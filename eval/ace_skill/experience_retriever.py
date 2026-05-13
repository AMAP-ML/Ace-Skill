"""
Experience Retrieval Module

Provides retrieval-based experience injection by selecting relevant experiences
based on query embedding similarity, rather than injecting all experiences.
"""

import os
import json
import hashlib
import time
import re
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path

import numpy as np
import requests
from sklearn.metrics.pairwise import cosine_similarity

import logging
logger = logging.getLogger(__name__)


# --------- Constants ---------

# Retry configuration
MAX_RETRIES = 5

# Timeout configuration
API_TIMEOUT = 60  # Timeout for embedding API requests (seconds)

# Batch processing configuration
DEFAULT_EMBEDDING_BATCH_SIZE = 10  # Batch size for embedding generation

# Retrieval configuration
TOP_K_RETRIEVE = 3  # Default number of top experiences to retrieve
MIN_SIMILARITY = 0.0  # Minimum similarity threshold for retrieval

# Token configuration
MAX_TOKENS_TASK_DECOMPOSITION = 2048  # Max tokens for task decomposition
MAX_TOKENS_REWRITE = 8192  # Max tokens for experience rewrite

# Temperature configuration
TEMPERATURE_TASK_DECOMPOSITION = 0.3  # Temperature for task decomposition
TEMPERATURE_REWRITE = 0.3  # Temperature for experience rewrite
TOP_P_DEFAULT = 1.0  # Default top_p parameter


# --------- Prompts ---------
from prompts.experience_prompts_test_time import TASK_DECOMPOSITION_PROMPT, EXPERIENCE_REWRITE_PROMPT


def _build_multimodal_user_content(prompt: str, images: List[Any]) -> List[dict]:
    """Build user content list for multimodal LLM call (text + image_url items)."""
    from PIL import Image
    from utils.context_utils import pil_to_base64_data_uri
    content = [{"type": "text", "text": prompt}]
    for img in images:
        if isinstance(img, Image.Image):
            content.append({
                "type": "image_url",
                "image_url": {"url": pil_to_base64_data_uri(img)}
            })
    return content


class ExperienceRetriever:
    """
    Retrieves relevant experiences based on query embedding similarity.
    
    Supports:
    - OpenAI-compatible embedding API
    - In-memory embedding storage with optional disk caching
    - Cosine similarity-based retrieval
    - Top-K experience selection
    """
    
    def __init__(
        self,
        experiences: Dict[str, str],
        embedding_model: str = "text-embedding-3-small",
        embedding_api_key: Optional[str] = None,
        embedding_endpoint: Optional[str] = None,
        cache_dir: Optional[str] = None,
        enable_cache: bool = True,
        llm_client: Optional[Any] = None,  # ExperienceLLM instance for task decomposition
        experience_library_path: Optional[str] = None,  # Path to experience library JSON file
    ):
        """
        Initialize the experience retriever.
        
        Args:
            experiences: Dictionary mapping experience IDs to experience text
            embedding_model: Embedding model name (default: "text-embedding-3-small")
            embedding_api_key: API key for embedding service (defaults to EXPERIENCE_EMBEDDING_API_KEY or OPENAI_API_KEY)
            embedding_endpoint: API endpoint for embedding service (defaults to EXPERIENCE_EMBEDDING_ENDPOINT or OPENAI_API_BASE)
            cache_dir: Directory to cache embeddings (defaults to experience/embeddings/{library_name}/)
            enable_cache: Whether to enable disk caching of embeddings
            llm_client: Optional ExperienceLLM instance for task decomposition (used by retrieve_with_decomposition)
            experience_library_path: Path to experience library JSON file (used to name cache subfolder)
        """
        self.experiences = dict(experiences)
        self.embedding_model = embedding_model
        self.cache_dir = cache_dir
        self.enable_cache = enable_cache
        self.llm_client = llm_client
        self.experience_library_path = experience_library_path
        
        # Store last retrieval info for saving to file
        self._last_retrieval_info: Optional[Dict[str, Any]] = None
        
        # API configuration
        self.embedding_api_key = (
            embedding_api_key 
            or os.environ.get("EXPERIENCE_EMBEDDING_API_KEY")
            or os.environ.get("OPENAI_API_KEY")
        )
        
        # Normalize endpoint
        self.embedding_endpoint = (
            embedding_endpoint
            or os.environ.get("EXPERIENCE_EMBEDDING_ENDPOINT")
            or os.environ.get("OPENAI_API_BASE", "https://api.openai.com/v1")
        )
        
        # Ensure endpoint format is correct
        if not self.embedding_endpoint.endswith("/v1"):
            self.embedding_endpoint = self.embedding_endpoint.rstrip("/") + "/v1"
        
        if not self.embedding_api_key:
            raise ValueError(
                "Embedding API key is required. Set EXPERIENCE_EMBEDDING_API_KEY "
                "or OPENAI_API_KEY environment variable, or pass embedding_api_key parameter."
            )
        
        # Embedding storage: {exp_id: np.ndarray}
        self._experience_embeddings: Dict[str, np.ndarray] = {}
        
        # Cache file paths
        self._cache_embeddings_path: Optional[str] = None
        self._cache_metadata_path: Optional[str] = None
        
        if self.enable_cache:
            if self.cache_dir:
                cache_base_dir = self.cache_dir
            else:
                experience_dir = os.path.dirname(os.path.abspath(__file__))
                cache_base_dir = os.path.join(experience_dir, "embeddings")
            
            if self.experience_library_path:
                library_filename = os.path.basename(self.experience_library_path)
                if library_filename.endswith('.json'):
                    library_name = library_filename[:-5]
                else:
                    library_name = library_filename
            else:
                library_name = self._compute_library_hash()[:8]
            
            cache_subdir = os.path.join(cache_base_dir, library_name)
            os.makedirs(cache_subdir, exist_ok=True)
            
            # Fixed filenames — cache is content-addressed by text hash internally,
            # so the filename does not change when library content changes.
            self._cache_embeddings_path = os.path.join(cache_subdir, "embedding_store.npy")
            self._cache_metadata_path = os.path.join(cache_subdir, "embedding_store_meta.json")
        
        # Load or generate embeddings
        self._load_or_generate_embeddings()
    
    def _compute_library_hash(self) -> str:
        """Compute hash of experience library (used only as fallback subfolder name)."""
        sorted_exps = sorted(self.experiences.items())
        content = json.dumps(sorted_exps, ensure_ascii=False, sort_keys=True)
        return hashlib.md5(content.encode('utf-8')).hexdigest()

    @staticmethod
    def _text_hash(text: str) -> str:
        """Compute hash of experience text for content-addressed caching."""
        return hashlib.sha256(text.encode('utf-8')).hexdigest()

    def _post_embedding_request(
        self, payload: dict, max_retries: int = MAX_RETRIES, is_batch: bool = False
    ) -> Optional[dict]:
        """
        POST to embedding API with retries. Returns response JSON body or None.
        """
        url = f"{self.embedding_endpoint}/embeddings"
        headers = {
            "Authorization": f"Bearer {self.embedding_api_key}",
            "Content-Type": "application/json",
        }
        prefix = "Batch embedding" if is_batch else "Embedding"
        for attempt in range(max_retries):
            try:
                response = requests.post(url, headers=headers, json=payload, timeout=API_TIMEOUT)
                response.raise_for_status()
                return response.json()
            except requests.exceptions.Timeout as e:
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    logger.warning(
                        f"{prefix} API timeout (attempt {attempt + 1}/{max_retries}): {e}. "
                        f"Retrying in {wait_time}s..."
                    )
                    time.sleep(wait_time)
                else:
                    logger.error(f"Failed to generate {'batch ' if is_batch else ''}embedding(s) after {max_retries} attempts (timeout): {e}")
                    print(f"  Error: {prefix} API timeout. Check endpoint: {url}")
                    return None
            except requests.exceptions.RequestException as e:
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    logger.warning(
                        f"{prefix} API request failed (attempt {attempt + 1}/{max_retries}): {e}. "
                        f"Retrying in {wait_time}s..."
                    )
                    time.sleep(wait_time)
                else:
                    logger.error(f"Failed to generate {'batch ' if is_batch else ''}embedding(s) after {max_retries} attempts: {e}")
                    print(f"  Error: {prefix} API request failed. Check endpoint: {url}, status: {getattr(e.response, 'status_code', 'N/A')}")
                    return None
            except Exception as e:
                logger.error(f"Unexpected error generating {'batch ' if is_batch else ''}embedding(s): {e}")
                print(f"  Error: Unexpected error: {e}")
                return None
        return None

    def _generate_embedding(self, text: str, max_retries: int = MAX_RETRIES) -> Optional[np.ndarray]:
        """
        Generate embedding for a single text using API.

        Args:
            text: Text to embed
            max_retries: Maximum number of retry attempts

        Returns:
            Embedding vector as numpy array, or None if failed
        """
        payload = {"model": self.embedding_model, "input": text}
        data = self._post_embedding_request(payload, max_retries=max_retries, is_batch=False)
        if not data:
            return None
        embedding = data.get("data", [{}])[0].get("embedding") if data.get("data") else None
        if embedding:
            return np.array(embedding, dtype=np.float32)
        logger.warning("Empty embedding response")
        logger.debug(f"Response data: {data}")
        return None
    
    def _generate_embeddings_batch(
        self, 
        texts: List[str], 
        batch_size: int = DEFAULT_EMBEDDING_BATCH_SIZE,
        max_retries: int = MAX_RETRIES
    ) -> List[Optional[np.ndarray]]:
        """
        Generate embeddings for multiple texts in batches using batch API.
        
        Args:
            texts: List of texts to embed
            batch_size: Number of texts per batch
            max_retries: Maximum retry attempts per batch
            
        Returns:
            List of embedding vectors (or None for failed texts)
        """
        all_embeddings = []
        total_batches = (len(texts) + batch_size - 1) // batch_size
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_num = i // batch_size + 1
            
            # Use batch API to generate embeddings for the entire batch at once
            batch_embeddings = self._generate_embeddings_batch_api(batch_texts, max_retries=max_retries)
            
            all_embeddings.extend(batch_embeddings)
            if total_batches > 1:
                successful = sum(1 for e in batch_embeddings if e is not None)
                print(f"  Progress: batch {batch_num}/{total_batches} ({successful}/{len(batch_texts)} successful)")
        return all_embeddings
    
    def _generate_embeddings_batch_api(
        self,
        texts: List[str],
        max_retries: int = MAX_RETRIES
    ) -> List[Optional[np.ndarray]]:
        """
        Generate embeddings for multiple texts using batch API call.

        Args:
            texts: List of texts to embed (will be sent in one API call)
            max_retries: Maximum retry attempts

        Returns:
            List of embedding vectors (or None for failed texts)
        """
        if not texts:
            return []
        payload = {"model": self.embedding_model, "input": texts}
        data = self._post_embedding_request(payload, max_retries=max_retries, is_batch=True)
        if not data:
            return [None] * len(texts)
        embeddings_data = data.get("data", [])
        batch_embeddings = []
        for item in embeddings_data:
            emb = item.get("embedding")
            batch_embeddings.append(np.array(emb, dtype=np.float32) if emb else None)
        while len(batch_embeddings) < len(texts):
            batch_embeddings.append(None)
        return batch_embeddings[:len(texts)]
    
    def _load_embedding_store(self) -> Dict[str, np.ndarray]:
        """Load content-addressed embedding store from disk.

        Returns:
            Dictionary mapping text_hash -> embedding vector.
            Empty dict if cache is missing, invalid, or fails to load.
        """
        if not self.enable_cache or not self._cache_embeddings_path:
            return {}

        if not os.path.exists(self._cache_embeddings_path):
            return {}

        if not os.path.exists(self._cache_metadata_path):
            return {}

        try:
            with open(self._cache_metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)

            if metadata.get("model") != self.embedding_model:
                logger.info("Embedding model changed, cache invalid, regenerating...")
                return {}

            store_raw = np.load(self._cache_embeddings_path, allow_pickle=True).item()
            store: Dict[str, np.ndarray] = {}
            for text_hash, emb_data in store_raw.items():
                store[text_hash] = np.array(emb_data, dtype=np.float32)

            logger.info(f"Loaded {len(store)} cached embeddings from {self._cache_embeddings_path}")
            return store

        except Exception as e:
            logger.warning(f"Failed to load embedding store: {e}")
            return {}
    
    def _save_embedding_store(self):
        """Save content-addressed embedding store to disk (text_hash -> embedding)."""
        if not self.enable_cache or not self._cache_embeddings_path:
            return

        try:
            store = {}
            for exp_id, emb in self._experience_embeddings.items():
                if exp_id in self.experiences:
                    text_hash = self._text_hash(self.experiences[exp_id])
                    store[text_hash] = emb.tolist()

            np.save(self._cache_embeddings_path, store, allow_pickle=True)

            metadata = {
                "model": self.embedding_model,
                "num_entries": len(store),
                "dimension": (
                    len(next(iter(self._experience_embeddings.values())))
                    if self._experience_embeddings else 0
                ),
                "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }

            with open(self._cache_metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)

            logger.info(
                f"Saved {len(store)} embeddings to store "
                f"({self._cache_embeddings_path})"
            )

        except Exception as e:
            logger.warning(f"Failed to save embedding store: {e}")
    
    def _load_or_generate_embeddings(self):
        """Load embeddings from content-addressed cache; only generate missing ones."""
        cached_store = self._load_embedding_store()

        texts_to_embed = []
        ids_to_embed = []
        reused_count = 0

        for exp_id, exp_text in self.experiences.items():
            text_hash = self._text_hash(exp_text)
            if text_hash in cached_store:
                self._experience_embeddings[exp_id] = cached_store[text_hash]
                reused_count += 1
            else:
                texts_to_embed.append(exp_text)
                ids_to_embed.append(exp_id)

        if texts_to_embed:
            total = len(self.experiences)
            print(f"\nGenerating embeddings for {len(texts_to_embed)}/{total} experiences "
                  f"(reused {reused_count} from cache) using model: {self.embedding_model}")

            start_time = time.time()
            embeddings = self._generate_embeddings_batch(
                texts_to_embed, batch_size=DEFAULT_EMBEDDING_BATCH_SIZE,
            )
            elapsed = time.time() - start_time

            successful_count = 0
            for exp_id, emb in zip(ids_to_embed, embeddings):
                if emb is not None:
                    self._experience_embeddings[exp_id] = emb
                    successful_count += 1
                else:
                    logger.warning(f"Failed to generate embedding for experience {exp_id}")

            print(f"Generated {successful_count}/{len(texts_to_embed)} new embeddings in {elapsed:.1f}s")
        else:
            print(f"Loaded all {reused_count} embeddings from cache (no API calls needed)")

        if self._experience_embeddings:
            self._save_embedding_store()
    
    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
        
        return float(dot_product / (norm_a * norm_b))
    
    def update_experiences(self, new_experiences: Dict[str, str], incremental: bool = True):
        """
        Update experiences and regenerate embeddings.

        When incremental=True, matches old embeddings by **text content hash**
        (not by experience ID) so that ID renumbering after batch_merge does not
        cause unnecessary re-embedding.

        Args:
            new_experiences: New or updated experiences dictionary
            incremental: If True, only generate embeddings for new/updated experiences.
                        If False, regenerate all embeddings.
        """
        if incremental:
            # Build text_hash -> embedding mapping from current embeddings.
            # This survives ID renumbering: even if E2 becomes E1, its text
            # hash stays the same and the embedding is reused.
            old_text_to_emb: Dict[str, np.ndarray] = {}
            for exp_id, emb in self._experience_embeddings.items():
                if exp_id in self.experiences:
                    text_hash = self._text_hash(self.experiences[exp_id])
                    old_text_to_emb[text_hash] = emb

            self.experiences = dict(new_experiences)

            new_embeddings: Dict[str, np.ndarray] = {}
            texts_to_embed: List[str] = []
            ids_to_embed: List[str] = []

            for exp_id, exp_text in new_experiences.items():
                text_hash = self._text_hash(exp_text)
                if text_hash in old_text_to_emb:
                    new_embeddings[exp_id] = old_text_to_emb[text_hash]
                else:
                    texts_to_embed.append(exp_text)
                    ids_to_embed.append(exp_id)

            self._experience_embeddings = new_embeddings

            if texts_to_embed:
                print(f"  Updating embeddings: {len(texts_to_embed)} new/changed, "
                      f"{len(new_embeddings)} reused from cache")
                embeddings = self._generate_embeddings_batch(
                    texts_to_embed, batch_size=DEFAULT_EMBEDDING_BATCH_SIZE,
                )
                for exp_id, emb in zip(ids_to_embed, embeddings):
                    if emb is not None:
                        self._experience_embeddings[exp_id] = emb
                    else:
                        logger.warning(f"Failed to generate embedding for experience {exp_id}")
            else:
                print(f"  All {len(new_embeddings)} embeddings reused from cache (no API calls)")

            if self._experience_embeddings:
                self._save_embedding_store()
        else:
            self.experiences = dict(new_experiences)
            self._experience_embeddings.clear()
            self._load_or_generate_embeddings()
    
    def get_last_retrieval_info(self) -> Optional[Dict[str, Any]]:
        """
        Get information about the last retrieval operation.
        
        Returns:
            Dictionary containing:
            - original_query: The original task description or query
            - decomposition_used: Whether task decomposition was used
            - subtasks: List of subtasks (if decomposition was used)
            - retrieved_experiences: List of experience IDs that were retrieved
            - retrieval_details: Details for each subtask retrieval (if decomposition was used)
            - total_unique_experiences: Total number of unique experiences retrieved
            None if no retrieval has been performed yet
        """
        return self._last_retrieval_info
    
    def retrieve(
        self,
        query: str,
        top_k: int = TOP_K_RETRIEVE,
        min_similarity: float = MIN_SIMILARITY,
        doc_id: Optional[str] = None,
    ) -> Tuple[Dict[str, str], Dict[str, Any]]:
        """
        Retrieve top-k most relevant experiences for a query.
        
        Args:
            query: Query text to search for
            top_k: Number of top experiences to retrieve
            min_similarity: Minimum similarity threshold (0.0 to 1.0)
            
        Returns:
            Tuple of:
            - Dictionary mapping experience IDs to experience text, sorted by relevance
            - Retrieval info dictionary containing:
                - original_query: The query used for retrieval
                - decomposition_used: False (always False for retrieve())
                - retrieved_experiences: List of retrieved experience IDs
                - total_unique_experiences: Number of unique experiences retrieved
        """
        empty_info = {
            "original_query": query,
            "decomposition_used": False,
            "subtasks": [],
            "retrieved_experiences": [],
            "retrieval_details": [],
            "total_unique_experiences": 0
        }
        if not self._experience_embeddings:
            logger.warning("No experience embeddings available, returning empty result")
            return {}, empty_info

        # Generate query embedding
        query_embedding = self._generate_embedding(query)
        if query_embedding is None:
            logger.warning("Failed to generate query embedding, returning empty result")
            return {}, empty_info

        # Compute similarities (vectorized)
        exp_ids = list(self._experience_embeddings.keys())
        exp_matrix = np.vstack([self._experience_embeddings[eid] for eid in exp_ids])
        q = query_embedding.reshape(1, -1)
        sim_scores = cosine_similarity(q, exp_matrix)[0]
        similarities = [(exp_ids[i], float(sim_scores[i])) for i in range(len(exp_ids)) if sim_scores[i] >= min_similarity]
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Get top-k
        top_similarities = similarities[:top_k]
        
        # Build result dictionary (preserve order)
        result = {
            exp_id: self.experiences[exp_id]
            for exp_id, _ in top_similarities
        }
        
        # Build retrieval info (thread-safe: returned directly, not stored in instance)
        retrieval_info = {
            "original_query": query,
            "decomposition_used": False,
            "subtasks": [],
            "retrieved_experiences": list(result.keys()),
            "retrieval_details": [],
            "total_unique_experiences": len(result)
        }
        
        # Also update instance variable for backward compatibility (get_last_retrieval_info)
        # Note: This may be overwritten by concurrent calls, but the return value is safe
        if not self._last_retrieval_info or not self._last_retrieval_info.get("decomposition_used", False):
            self._last_retrieval_info = retrieval_info.copy()
        
        if result:
            logger.debug(
                f"Retrieved {len(result)} experiences for query "
                f"(similarities: {[f'{s:.3f}' for _, s in top_similarities[:3]]}...)"
            )
        
        return result, retrieval_info
    
    def get_embedding_stats(self) -> Dict[str, Any]:
        """Get statistics about loaded embeddings."""
        return {
            "total_experiences": len(self.experiences),
            "embedded_count": len(self._experience_embeddings),
            "missing_count": len(self.experiences) - len(self._experience_embeddings),
            "embedding_model": self.embedding_model,
            "cache_enabled": self.enable_cache,
            "cache_path": self._cache_embeddings_path,
        }
    
    def _decompose_task_llm(self, task_description: str, images: Optional[List[Any]] = None) -> List[Dict[str, str]]:
        """
        Decompose a task into subtasks with queries using LLM.
        
        Args:
            task_description: The original task description
            images: Optional list of PIL.Image objects to include in the decomposition
            
        Returns:
            List of dictionaries, each containing:
                - "type": Subtask type/category
                - "query": Query string for retrieving relevant experiences
        """
        if not self.llm_client:
            raise ValueError("LLM client is required for task decomposition. Pass llm_client to ExperienceRetriever.__init__")
        
        prompt = TASK_DECOMPOSITION_PROMPT.format(task_description=task_description)
        
        # Retry logic: retry on any error until success or max retries
        for attempt in range(MAX_RETRIES):
            try:
                if images:
                    user_content = _build_multimodal_user_content(prompt, images)
                    response = self.llm_client._call_with_fallback(
                        user_content=user_content,
                        max_tokens=MAX_TOKENS_TASK_DECOMPOSITION,
                        temperature=TEMPERATURE_TASK_DECOMPOSITION,
                        top_p=TOP_P_DEFAULT,
                        primary_api_name="Primary Experience API",
                        fallback_api_name="Fallback Experience API",
                        require_chat_completions=True
                    )
                else:
                    response = self.llm_client.chat(prompt, max_tokens=MAX_TOKENS_TASK_DECOMPOSITION, temperature=TEMPERATURE_TASK_DECOMPOSITION)
                
                # Extract JSON from response (handle cases where LLM adds extra text)
                response = response.strip()
                
                # Try to find JSON array in the response
                json_match = re.search(r'\[.*\]', response, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                    subtasks = json.loads(json_str)
                    
                    # Validate format
                    if isinstance(subtasks, list) and all(
                        isinstance(item, dict) and "type" in item and "query" in item
                        for item in subtasks
                    ):
                        logger.debug(f"Decomposed task into {len(subtasks)} subtasks: {[s['type'] for s in subtasks]}")
                        return subtasks
                    else:
                        raise ValueError("LLM returned invalid subtask format")
                else:
                    raise ValueError("Could not extract JSON from LLM response")
                    
            except (json.JSONDecodeError, ValueError) as e:
                if attempt < MAX_RETRIES - 1:
                    logger.warning(f"Task decomposition failed (attempt {attempt + 1}/{MAX_RETRIES}): {e}, retrying...")
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    logger.warning(f"Task decomposition failed after {MAX_RETRIES} attempts: {e}, falling back to single query")
            except Exception as e:
                if attempt < MAX_RETRIES - 1:
                    logger.warning(f"LLM decomposition failed (attempt {attempt + 1}/{MAX_RETRIES}): {e}, retrying...")
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    logger.warning(f"LLM decomposition failed after {MAX_RETRIES} attempts: {e}, falling back to single query")
        
        # Fallback: return a single general query
        return [{
            "type": "general",
            "query": f"Methodology for handling this type of task: {task_description}"
        }]
    
    def retrieve_with_decomposition(
        self,
        task_description: str,
        top_k: int = TOP_K_RETRIEVE,
        min_similarity: float = MIN_SIMILARITY,
        subtask_top_k: Optional[int] = None,
        images: Optional[List[Any]] = None,
        doc_id: Optional[str] = None,
    ) -> Tuple[Dict[str, str], Dict[str, Any]]:
        """
        Retrieve experiences by decomposing task into subtasks and retrieving for each.
        
        This method:
        1. Decomposes the task into subtasks using LLM (with images if provided)
        2. Retrieves experiences for each subtask query
        3. Merges results (deduplicates, preserves order) - returns ALL unique experiences from all subtasks
        
        Args:
            task_description: The original task description
            top_k: Number of experiences to retrieve per subtask (if subtask_top_k not specified)
            min_similarity: Minimum similarity threshold for each retrieval
            subtask_top_k: Number of experiences to retrieve per subtask (defaults to top_k if not specified)
            images: Optional list of PIL.Image objects to include in task decomposition
            
        Returns:
            Tuple of:
            - Dictionary mapping experience IDs to experience text (deduplicated, ordered)
              Contains all unique experiences from all subtasks (no final limit applied)
            - Retrieval info dictionary containing:
                - original_query: The original task description
                - decomposition_used: True
                - subtasks: List of subtask dictionaries
                - retrieved_experiences: List of all retrieved experience IDs
                - retrieval_details: List of retrieval details for each subtask
                - total_unique_experiences: Total number of unique experiences retrieved
        """
        if not self.llm_client:
            raise ValueError(
                "LLM client is required for decomposition-based retrieval. "
                "Pass llm_client to ExperienceRetriever.__init__ or use retrieve() instead"
            )
        
        # Use subtask_top_k if specified, otherwise use top_k
        if subtask_top_k is None:
            subtask_top_k = top_k
        
        # Step 1: Decompose task
        print(f"  [Decomposition] Decomposing task into subtasks...")
        subtasks = self._decompose_task_llm(task_description, images=images)
        
        if not subtasks:
            print(f"  [Decomposition] Warning: Task decomposition returned no subtasks, falling back to direct retrieval")
            logger.warning("Task decomposition returned no subtasks, falling back to direct retrieval")
            result, retrieval_info = self.retrieve(task_description, top_k=top_k, min_similarity=min_similarity)
            # Also update instance variable for backward compatibility
            self._last_retrieval_info = retrieval_info.copy()
            return result, retrieval_info
        
        # Step 2: Retrieve for each subtask
        all_results: Dict[str, str] = {}  # Use dict to automatically deduplicate by key
        retrieval_details = []  # Store details for each subtask retrieval
        
        for subtask in subtasks:
            query = subtask.get("query", "")
            if not query:
                continue
            
            # Call retrieve() which now returns tuple
            result, _ = self.retrieve(
                query=query,
                top_k=subtask_top_k,  # Retrieve subtask_top_k experiences per subtask
                min_similarity=min_similarity
            )
            
            # Store retrieval details for this subtask
            retrieval_details.append({
                "subtask_type": subtask.get("type", "unknown"),
                "query": query,
                "retrieved_experience_ids": list(result.keys()),
                "count": len(result)
            })
            
            # Merge into all_results (preserves order: first occurrence wins)
            for exp_id, exp_text in result.items():
                if exp_id not in all_results:
                    all_results[exp_id] = exp_text
                
        # Build final retrieval info (thread-safe: returned directly)
        retrieval_info = {
            "original_query": task_description,
            "decomposition_used": True,
            "subtasks": subtasks,
            "retrieved_experiences": list(all_results.keys()),
            "retrieval_details": retrieval_details,
            "total_unique_experiences": len(all_results)
        }
        
        # Also update instance variable for backward compatibility (get_last_retrieval_info)
        self._last_retrieval_info = retrieval_info.copy()
        
        # Step 3: Return all unique experiences (no final limit - keep all from all subtasks)
        if all_results:
            print(f"  [Final] Merged {len(all_results)} unique experiences from {len(subtasks)} subtasks")
            logger.info(
                f"Retrieved {len(all_results)} unique experiences "
                f"from {len(subtasks)} subtasks (each retrieved up to {subtask_top_k} experiences)"
            )
        else:
            print(f"  [Final] No experiences retrieved from any subtask")
        
        return all_results, retrieval_info


def rewrite_experiences_for_task(
    experiences: Dict[str, str],
    task_description: str,
    llm_client: Any,  # ExperienceLLM instance
    images: Optional[List[Any]] = None,
) -> Dict[str, str]:
    """
    Rewrite retrieved experiences to better fit a specific task.
    
    Args:
        experiences: Dictionary mapping experience IDs to experience text
        task_description: The current task description
        llm_client: ExperienceLLM instance for calling LLM
        images: Optional list of PIL.Image objects (for multimodal rewrite)
        
    Returns:
        Dictionary mapping experience IDs to rewritten experience text
        (If rewrite fails, returns original experiences as fallback)
    """
    
    if not experiences:
        return experiences
    
    try:
        # Format experiences for prompt
        exp_items = []
        for exp_id, exp_text in experiences.items():
            exp_items.append(f"[{exp_id}]\n{exp_text}")
        experiences_text = "\n\n".join(exp_items)
        
        # Build rewrite prompt
        prompt = EXPERIENCE_REWRITE_PROMPT.format(
            task_description=task_description,
            experiences_text=experiences_text
        )
        
        # Call LLM with or without images
        if images:
            user_content = _build_multimodal_user_content(prompt, images)
            response = llm_client._call_with_fallback(
                user_content=user_content,
                max_tokens=MAX_TOKENS_REWRITE,
                temperature=TEMPERATURE_REWRITE,
                top_p=TOP_P_DEFAULT,
                system_prompt="You are an expert at adapting generalizable experiences to specific task contexts.",
                primary_api_name="Primary Experience Rewrite API",
                fallback_api_name="Fallback Experience Rewrite API",
                require_chat_completions=True,
                return_placeholder_on_error=False
            )
        else:
            response = llm_client.chat(prompt, max_tokens=MAX_TOKENS_REWRITE, temperature=TEMPERATURE_REWRITE)
        
        # Parse JSON response
        response = response.strip()
        
        # Try to find JSON object in the response
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
            rewritten_dict = json.loads(json_str)
            
            # Validate response format (allow skipping experiences as per prompt instructions)
            if isinstance(rewritten_dict, dict):
                original_count = len(experiences)
                rewritten_count = len(rewritten_dict)
                missing_ids = set(experiences.keys()) - set(rewritten_dict.keys())
                
                if missing_ids:
                    # Model skipped some experiences (this is allowed per prompt)
                    logger.info(f"Rewrite: Model skipped {len(missing_ids)} experience(s) (from {original_count} to {rewritten_count})")
                    print(f"  [Rewrite] Model skipped {len(missing_ids)} experience(s): {', '.join(sorted(missing_ids))}")
                else:
                    print(f"  [Rewrite] Successfully rewritten all {rewritten_count} experiences")
                
                return rewritten_dict
            else:
                logger.warning("Rewrite response is not a dictionary, using original experiences")
        else:
            logger.warning("Could not extract JSON from rewrite response, using original experiences")
    
    except Exception as e:
        logger.warning(f"Failed to rewrite experiences: {e}", exc_info=True)
        print(f"  [Rewrite] Warning: Failed to rewrite experiences: {e}, using original experiences")
    
    # Fallback to original experiences
    return experiences

