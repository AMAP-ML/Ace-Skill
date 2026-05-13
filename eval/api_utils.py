"""
Shared utility functions for train.py and infer.py.
"""

import argparse
import os
import re
import json
import logging
import copy
import yaml
from typing import Dict, List, Optional, Any, Callable
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from ace_skill import ExperienceLLM, load_experiences, format_for_prompt, rewrite_experiences_for_task, adapt_skill_for_task, ClusteredExperienceManager
from prompts.skill_prompts_test_time import SKILL_INJECTION_HEADER
from engine.api_processors import process_single_sample, set_global_prompts as set_processors_prompts
from qwen_vl_utils import fetch_image
from utils.context_utils import process_image

logger = logging.getLogger(__name__)

# Constants
RETRIEVAL_INFO_FILENAME = "tt_decomposition_info.txt"
SAMPLE_SUMMARY_FILENAME = "metrics_sample.json"
DATASET_SUMMARY_FILENAME = "metrics_at_k.json"


# ============================================================================
# Shared argument parser and setup
# ============================================================================

def create_base_parser(description="API-based inference script."):
    """
    Create argument parser with shared arguments for both train and inference.
    Train-specific arguments should be added by the caller after this returns.
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--input-file", type=str, required=True, help="Path to the input data file.")
    parser.add_argument("--image-folder", type=str, required=True, help="Path to the folder containing images.")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to save the output results.")
    parser.add_argument("--skip-completed", action='store_true',
                       help="Skip samples that are already completed (have traj.jsonl with turn_idx and metrics.json).")
    parser.add_argument("--max-samples", type=int, default=None,
                       help="Maximum number of samples to process (default: None, process all samples).")

    # Sampling Arguments
    parser.add_argument("--reasoning-temperature", type=float, required=True,
                       help="Reasoning-model temperature (required).")
    parser.add_argument("--reasoning-top-p", type=float, required=True,
                       help="Reasoning-model top-p (required).")
    parser.add_argument("--reasoning-top-k", type=int, default=None,
                       help="Optional reasoning-model top-k; omitted when unset.")
    parser.add_argument("--reasoning-presence-penalty", type=float, default=None,
                       help="Optional reasoning-model presence_penalty; omitted when unset.")
    parser.add_argument("--reasoning-repetition-penalty", type=float, default=None,
                       help="Optional reasoning-model repetition_penalty; omitted when unset.")
    parser.add_argument("--experience-temperature", type=float, required=True,
                       help="Experience-model temperature (required).")
    parser.add_argument("--experience-top-p", type=float, required=True,
                       help="Experience-model top-p (required).")
    parser.add_argument("--experience-top-k", type=int, default=None,
                       help="Optional experience-model top-k; omitted when unset.")
    parser.add_argument("--experience-presence-penalty", type=float, default=None,
                       help="Optional experience-model presence_penalty; omitted when unset.")
    parser.add_argument("--experience-repetition-penalty", type=float, default=None,
                       help="Optional experience-model repetition_penalty; omitted when unset.")
    parser.add_argument("--max-completion-tokens", type=int, default=8192, help="Maximum tokens for single model response (aligned with Baseline framework).")

    # Inference Arguments
    parser.add_argument("--max-turns", type=int, default=16, help="Maximum number of turns for inference.")
    parser.add_argument("--max-images", type=int, default=16, help="Maximum number of images per sample.")
    parser.add_argument("--max-total-tokens", type=int, default=8000, help="Maximum total tokens for context.")
    parser.add_argument("--max-pixels", type=int, default=2000000, help="Maximum pixels for image processing.")
    parser.add_argument("--min-pixels", type=int, default=40000, help="Minimum pixels for image processing.")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of workers for parallel processing.")
    parser.add_argument("--rollouts-per-sample", type=int, default=4, help="Number of independent rollouts per sample for pass@k evaluation.")
    parser.add_argument("--seed-base", type=int, default=42, help="Base seed for reproducibility. Each rollout uses seed_base + rollout_index.")

    # Prompt Arguments
    parser.add_argument("--inference-prompts-path", type=str, default="eval/prompts/inference_prompts.yaml",
                       help="Path to the inference prompts YAML file.")
    parser.add_argument("--system-prompt-key", type=str, default="multi_tool_agent_search",
                       choices=['direct_cot', 'agent_zoom', 'multi_tool_agent', 'multi_tool_agent_search', 'multi_tool_agent_code'],
                       help="The key for the system prompt to use from the inference prompts YAML file.")

    # Tool Arguments
    parser.add_argument("--tool-config-path", type=str, default="eval/configs/tool_configs.yaml",
                       help="Path to the tool configuration YAML file.")
    parser.add_argument("--image-search-max-calls", type=int, default=3,
                       help="Maximum number of image_search tool calls per sample (default: 3)")
    parser.add_argument("--web-search-max-calls", type=int, default=5,
                       help="Maximum number of web_search tool calls per sample (default: 5)")

    # Experience Arguments (shared between train and inference)
    parser.add_argument("--experience-enable", action='store_true',
                       help="Enable experience injection from library")
    parser.add_argument("--experience-library", type=str, default=None,
                       help="Path to consolidated experiences JSON")
    parser.add_argument("--experience-max-items", type=int, default=120,
                        help="Max number of experiences to inject/keep in library")

    # Skill Arguments (shared between train and inference)
    parser.add_argument("--skill-enable", action='store_true',
                       help="Enable dynamic skill generation")
    parser.add_argument("--skill-library", type=str, default=None,
                       help="Base path used to derive per-cluster skill files")
    parser.add_argument("--skill-inference", action='store_true',
                       help="Enable skill injection during inference")
    parser.add_argument("--no-skill-adaptation", action='store_false', dest='skill_adaptation',
                       default=True, help="Disable skill adaptation (use raw skill directly, default: adapt)")

    # Experience Retrieval Arguments
    parser.add_argument("--experience-retrieval", action='store_true',
                        help="Enable retrieval-based experience injection (default: False, uses all experiences)")
    parser.add_argument("--experience-retrieval-top-k", type=int, default=3,
                        help="Number of top experiences to retrieve per query (default: 3)")
    parser.add_argument("--experience-retrieval-min-similarity", type=float, default=0.0,
                        help="Minimum similarity threshold for retrieved experiences (0.0 to 1.0, default: 0.0)")
    parser.add_argument("--experience-embedding-api-key", type=str, default=None,
                        help="API key for embedding service (defaults to EXPERIENCE_EMBEDDING_API_KEY or OPENAI_API_KEY)")
    parser.add_argument("--experience-embedding-endpoint", type=str, default=None,
                        help="API endpoint for embedding service (defaults to EXPERIENCE_EMBEDDING_ENDPOINT or OPENAI_API_BASE)")
    parser.add_argument("--no-experience-embedding-cache", action='store_false', dest='experience_embedding_cache_enable',
                       default=True, help="Disable disk caching of experience embeddings (default: enabled)")
    parser.add_argument("--experience-retrieval-decomposition", action='store_true',
                        help="Enable task decomposition for retrieval (uses LLM to decompose task into subtasks, then retrieves for each) (default: False)")
    parser.add_argument("--experience-retrieval-rewrite", action='store_true',
                        help="Enable experience rewrite to adapt retrieved experiences to the current task (uses LLM to rewrite experiences) (default: False)")

    return parser


def _derive_cluster_mapping_path(input_file_path: str) -> Optional[str]:
    """Derive the cluster mapping file path from the input data file path.

    Convention: ``{stem}_doc_id_to_cluster.json`` sits alongside the input file.
    E.g. ``train_shuffled.json`` -> ``train_shuffled_doc_id_to_cluster.json``.
    """
    base, ext = os.path.splitext(input_file_path)
    return f"{base}_doc_id_to_cluster.json"


def _cluster_skill_path(base_skill_path: str, cluster_id: int) -> str:
    """Derive per-cluster skill file path from the base skill path.

    ``SKILL.md`` -> ``SKILL_cluster_0.md``, etc.
    Mirrors ``ClusteredExperienceManager._cluster_library_path``.
    """
    base, ext = os.path.splitext(base_skill_path)
    return f"{base}_cluster_{cluster_id}{ext}"


def setup_common(args):
    """
    Common initialization for both train and inference pipelines.
    
    Sets args.model_name and args.tool_configs as side effects.
    
    Returns:
        tuple (data, sampling_params, BASE_SYSTEM_PROMPT, experience_retriever)
        or None if initialization fails (e.g. prompt file not found).
    """
    args.model_name = os.environ.get("REASONING_MODEL_NAME")
    if not args.model_name:
        raise ValueError("REASONING_MODEL_NAME environment variable must be set")

    print(f"Starting API-based greedy inference...")
    print(f"Reasoning model: {args.model_name}")

    if args.rollouts_per_sample > 1:
        print(f"\n{'='*80}")
        print(f"Multi-rollout mode enabled: {args.rollouts_per_sample} rollouts per sample")
        print(f"Base seed: {args.seed_base}")
        print(f"This will enable pass@k and average@k evaluation")
        print(f"{'='*80}\n")

    # Load system prompt from YAML
    try:
        with open(args.inference_prompts_path, 'r', encoding='utf-8') as f:
            prompts_yaml = yaml.safe_load(f)
        SYSTEM_PROMPT = prompts_yaml['system_prompts'][args.system_prompt_key]
        print("Inference prompts loaded successfully.")
    except Exception as e:
        print(f"Error loading inference prompts from {args.inference_prompts_path}: {e}")
        return None

    BASE_SYSTEM_PROMPT = SYSTEM_PROMPT
    set_processors_prompts(SYSTEM_PROMPT)

    # Load tool configs
    try:
        tool_config_path = getattr(args, 'tool_config_path', 'eval/configs/tool_configs.yaml')
        if os.path.exists(tool_config_path):
            with open(tool_config_path, 'r', encoding='utf-8') as f:
                args.tool_configs = yaml.safe_load(f) or {}
            print(f"Loaded tool configs from {tool_config_path}")
        else:
            args.tool_configs = {}
            print(f"Tool config file not found at {tool_config_path}, using defaults")
    except Exception as e:
        print(f"Error loading tool configs: {e}")
        args.tool_configs = {}

    # Load data
    if args.input_file.endswith('.jsonl'):
        with open(args.input_file, 'r', encoding='utf-8') as f:
            data = [json.loads(line) for line in f]
    else:
        with open(args.input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

    original_count = len(data)

    if args.max_samples is not None and args.max_samples > 0:
        data = data[:args.max_samples]
        print(f"Loaded {original_count} samples, limiting to {len(data)} samples (--max-samples={args.max_samples}).")
    else:
        print(f"Loaded {len(data)} samples.")

    # Cluster mode is always enforced; the mapping file must exist.
    cluster_mapping_path = _derive_cluster_mapping_path(args.input_file)
    if not cluster_mapping_path or not os.path.exists(cluster_mapping_path):
        raise FileNotFoundError(
            f"Cluster mapping file not found at {cluster_mapping_path}. "
            f"This pipeline runs in cluster-only mode; please prepare "
            f"'<input>_doc_id_to_cluster.json' alongside --input-file."
        )

    with open(cluster_mapping_path, 'r', encoding='utf-8') as f:
        args._cluster_mapping = json.load(f)
    unmapped = 0
    for sample in data:
        doc_id = sample.get('doc_id', '')
        if doc_id in args._cluster_mapping:
            sample['_cluster_id'] = args._cluster_mapping[doc_id]
        else:
            sample['_cluster_id'] = -1
            unmapped += 1
    cluster_ids = sorted(set(s['_cluster_id'] for s in data if s['_cluster_id'] != -1))
    print(f"Cluster mapping loaded from {cluster_mapping_path}: "
          f"{len(args._cluster_mapping)} entries, {len(cluster_ids)} clusters, "
          f"{unmapped} unmapped samples")

    sampling_params = {
        "temperature": args.reasoning_temperature,
        "top_p": args.reasoning_top_p,
        "max_tokens": args.max_completion_tokens,
    }
    if args.reasoning_top_k is not None:
        sampling_params["top_k"] = args.reasoning_top_k
    if args.reasoning_presence_penalty is not None:
        sampling_params["presence_penalty"] = args.reasoning_presence_penalty
    if args.reasoning_repetition_penalty is not None:
        sampling_params["repetition_penalty"] = args.reasoning_repetition_penalty

    ExperienceLLM.configure_global_sampling(
        temperature=args.experience_temperature,
        top_p=args.experience_top_p,
        top_k=args.experience_top_k,
        presence_penalty=args.experience_presence_penalty,
        repetition_penalty=args.experience_repetition_penalty,
    )

    # In cluster-only mode the per-sample injection is handled by the retriever
    experience_retriever, _ = initialize_experience_retriever(args)

    return data, sampling_params, BASE_SYSTEM_PROMPT, experience_retriever


# ============================================================================
# Shared rollout execution
# ============================================================================

def run_single_rollout(sample, args, sampling_params, rollout_idx):
    """
    Run a single rollout for a given sample.
    
    Args:
        sample: Data sample to process
        args: Command line arguments
        sampling_params: Sampling parameters dictionary
        rollout_idx: Rollout index (0-based)
        
    Returns:
        Result dictionary with trajectory information
    """
    import random

    rollout_args = prepare_sample_args(args)

    if hasattr(args, '_sample_system_prompt'):
        set_processors_prompts(args._sample_system_prompt)

    seed = args.seed_base + rollout_idx
    random.seed(seed)

    rollout_sampling_params = sampling_params.copy()
    if 'seed' in rollout_sampling_params:
        rollout_sampling_params['seed'] = seed

    result = process_single_sample(sample, rollout_args, rollout_sampling_params, rollout_idx=rollout_idx)
    return result


# ============================================================================
# Experience retriever initialization and retrieval
# ============================================================================

def initialize_experience_retriever(args):
    """
    Initialize the (cluster-only) experience retriever.

    Cluster-only invariants:
      * ``--experience-enable`` and ``--experience-retrieval`` must both be set
        whenever ``args.experience_library`` is provided.
      * The retriever is always a ``ClusteredExperienceManager``.

    Returns:
        tuple: (experience_retriever, system_prompt)
            - experience_retriever: ClusteredExperienceManager or None
            - system_prompt: Always None in cluster-only mode (kept for
              signature compatibility with callers).
    """
    if not (getattr(args, 'experience_enable', False) and args.experience_library):
        return None, None

    if not getattr(args, 'experience_retrieval', False):
        raise ValueError(
            "Cluster-only pipeline requires --experience-retrieval to be enabled "
            "whenever --experience-enable is set."
        )

    print(f"\n{'='*80}")
    print(f"Initializing Clustered Experience Manager (cluster-only mode)...")
    print(f"{'='*80}\n")
    print(f"Experience library base path: {args.experience_library}")

    enable_cache = getattr(args, 'experience_embedding_cache_enable', True)
    cache_dir = os.path.dirname(args.experience_library) if args.experience_library else None
    embedding_model = os.environ.get("EXPERIENCE_EMBEDDING_MODEL", "text-embedding-3-small")
    embedding_api_key = getattr(args, 'experience_embedding_api_key', None) or os.environ.get("EXPERIENCE_EMBEDDING_API_KEY") or os.environ.get("OPENAI_API_KEY")
    embedding_endpoint = getattr(args, 'experience_embedding_endpoint', None) or os.environ.get("EXPERIENCE_EMBEDDING_ENDPOINT") or os.environ.get("OPENAI_API_BASE", "https://api.openai.com/v1")

    if embedding_endpoint and not embedding_endpoint.endswith("/v1"):
        embedding_endpoint = embedding_endpoint.rstrip("/") + "/v1"

    llm_client = None
    use_decomposition = getattr(args, 'experience_retrieval_decomposition', False)
    use_rewrite = getattr(args, 'experience_retrieval_rewrite', False)
    if use_decomposition or use_rewrite:
        features = []
        if use_decomposition:
            features.append("task decomposition")
        if use_rewrite:
            features.append("experience rewrite")
        print(f"Initializing LLM client for {', '.join(features)}...")
        llm_client = ExperienceLLM()
        print(f"LLM client initialized.")

    retriever_kwargs = dict(
        embedding_model=embedding_model,
        embedding_api_key=embedding_api_key,
        embedding_endpoint=embedding_endpoint,
        cache_dir=cache_dir,
        enable_cache=enable_cache,
    )

    experience_retriever = ClusteredExperienceManager(
        cluster_mapping=args._cluster_mapping,
        base_library_path=args.experience_library,
        llm_client=llm_client,
        **retriever_kwargs,
    )

    stats = experience_retriever.get_embedding_stats()
    print(f"Clustered Experience Manager initialized:")
    print(f"  - Total experiences: {stats['total_experiences']}")
    print(f"  - Embedded count: {stats['embedded_count']}")
    print(f"  - Embedding model: {stats['embedding_model']}")
    print(f"  - Cache enabled: {stats['cache_enabled']}")
    if stats.get('num_clusters'):
        print(f"  - Clusters: {stats['num_clusters']}")
    if stats.get('cache_path'):
        print(f"  - Cache path: {stats['cache_path']}")
    if use_decomposition:
        print(f"  - Task decomposition: ENABLED")
    if use_rewrite:
        print(f"  - Experience rewrite: ENABLED")
    print(f"\n{'='*80}\n")

    return experience_retriever, None


def retrieve_experiences_for_sample(sample, args, experience_retriever, base_system_prompt, 
                                     question_id, sample_dir, update_global_prompts=True):
    """
    Retrieve experiences and/or inject skills for a single sample.
    Supports: Experience only, Skill only, or both combined.
    """
    exp_enabled = experience_retriever is not None
    skill_enabled = getattr(args, 'skill_inference', False) and getattr(args, 'skill_library', None)
    
    if not (exp_enabled or skill_enabled):
        return _finalize_prompt(args, base_system_prompt, update_global_prompts)
    
    if sample_dir:
        os.makedirs(sample_dir, exist_ok=True)
    
    try:
        query_text = sample.get('problem', sample.get('question', ''))
        images = _load_images_for_retrieval(sample, args)
        
        retrieved_exps, retrieval_info, original_retrieved_exps = {}, {}, None
        use_rewrite = getattr(args, 'experience_retrieval_rewrite', False)
        llm_client = None
        
        if exp_enabled:
            doc_id = sample.get('doc_id')
            use_decomposition = getattr(args, 'experience_retrieval_decomposition', False)
            if use_decomposition:
                retrieved_exps, retrieval_info = experience_retriever.retrieve_with_decomposition(
                    task_description=query_text,
                    top_k=getattr(args, 'experience_retrieval_top_k', 5),
                    min_similarity=getattr(args, 'experience_retrieval_min_similarity', 0.0),
                    subtask_top_k=getattr(args, 'experience_retrieval_subtask_top_k', None),
                    images=images,
                    doc_id=doc_id,
                )
            else:
                retrieved_exps, retrieval_info = experience_retriever.retrieve(
                    query=query_text,
                    top_k=getattr(args, 'experience_retrieval_top_k', 5),
                    min_similarity=getattr(args, 'experience_retrieval_min_similarity', 0.0),
                    doc_id=doc_id,
                )
            
            if not retrieved_exps:
                print(f"  [Retrieval] No relevant experiences found for {question_id}")
            
            if retrieved_exps and use_rewrite:
                original_retrieved_exps = copy.deepcopy(retrieved_exps)
                llm_client = experience_retriever.llm_client or ExperienceLLM()
                print(f"  [Retrieval] Rewriting {len(retrieved_exps)} experiences...")
                retrieved_exps = rewrite_experiences_for_task(retrieved_exps, query_text, llm_client, images)
        
        skill_injection = ""
        if skill_enabled:
            skill_injection = _process_skill(args, query_text, retrieved_exps, images, llm_client, sample_dir, question_id, sample=sample)
        
        exp_injection = format_for_prompt(retrieved_exps, max_items=len(retrieved_exps)) if retrieved_exps else ""
        
        use_adaptation = getattr(args, 'skill_adaptation', True)
        
        if skill_injection and exp_injection:
            if skill_enabled and not use_adaptation:
                full_injection = f"{skill_injection}\n\n{exp_injection}"
                print(f"  [Injection] Combined raw skill + experiences for {question_id}")
            else:
                full_injection = skill_injection
        elif skill_injection:
            full_injection = skill_injection
        elif exp_injection:
            full_injection = exp_injection
        else:
            full_injection = ""
        
        dynamic_system_prompt = (
            f"{full_injection}\n\n{base_system_prompt}" if full_injection and base_system_prompt
            else (full_injection or base_system_prompt)
        )
        
        if exp_enabled and sample_dir:
            save_retrieval_info(experience_retriever, sample_dir, retrieved_exps or None, 
                              original_retrieved_exps, use_rewrite, retrieval_info=retrieval_info)
        
        return _finalize_prompt(args, dynamic_system_prompt, update_global_prompts)
        
    except Exception as e:
        logger.warning(f"Failed to process for {question_id}: {e}", exc_info=True)
        print(f"  [Pipeline] Warning: {e}")
        return _finalize_prompt(args, base_system_prompt, update_global_prompts)


def _process_skill(args, query_text, retrieved_exps, images, llm_client, sample_dir, question_id, sample=None):
    """Process skill: adapt or use raw based on skill_adaptation flag.

    Resolves the per-cluster skill file via the sample's ``doc_id`` and
    ``_cluster_mapping``. If the mapped cluster file does not exist, skip
    skill injection for this sample.
    """
    try:
        cluster_mapping = getattr(args, '_cluster_mapping', None)
        if cluster_mapping is None or sample is None:
            print("  [Skill] Cluster context unavailable, skipping skill injection")
            return ""

        doc_id = sample.get('doc_id', '')
        cluster_id = cluster_mapping.get(doc_id)
        if cluster_id is None:
            print(f"  [Skill] doc_id '{doc_id}' not in cluster mapping, skipping skill injection")
            return ""

        skill_path = _cluster_skill_path(args.skill_library, cluster_id)
        if not os.path.exists(skill_path):
            print(f"  [Skill] Cluster {cluster_id} skill not found at {skill_path}, skipping skill injection")
            return ""

        with open(skill_path, 'r', encoding='utf-8') as f:
            base_skill = f.read()
        
        use_adaptation = getattr(args, 'skill_adaptation', True)
        
        if use_adaptation and llm_client is None:
            llm_client = ExperienceLLM()
        
        exp_text = "\n\n".join(f"--- {k} ---\n{v}" for k, v in retrieved_exps.items()) if retrieved_exps else ""
        
        if use_adaptation:
            adapted_skill = adapt_skill_for_task(base_skill, exp_text, query_text, llm_client, images)
            skill_content = adapted_skill
            print(f"  [Skill] Adapted for {question_id}")
            
            if sample_dir:
                with open(os.path.join(sample_dir, "tt_skill_adapted.md"), 'w', encoding='utf-8') as f:
                    f.write(skill_content)
        
        else:
            skill_content = base_skill
            print(f"  [Skill] Using raw skill for {question_id}")
            
            if sample_dir:
                with open(os.path.join(sample_dir, "tt_skill_original.md"), 'w', encoding='utf-8') as f:
                    f.write(skill_content)
        
        return SKILL_INJECTION_HEADER.format(skill_content=skill_content)
    except Exception as e:
        print(f"  [Skill] Warning: {e}")
        return ""


def _finalize_prompt(args, prompt, update_global_prompts):
    """Helper to finalize prompt: update globals or set on args."""
    if update_global_prompts:
        set_processors_prompts(prompt)
    else:
        args._sample_system_prompt = prompt
    return prompt


def _load_images_for_retrieval(sample, args):
    """
    Load images from original data source for experience retrieval.
    
    Args:
        sample: Data sample
        args: Command line arguments
        
    Returns:
        list or None: List of processed images, or None if no images found
    """
    if not (hasattr(args, 'image_folder') and args.image_folder):
        return None
    
    image_paths = sample.get('images', [])
    if not image_paths:
        return None
    
    images = []
    max_pixels = getattr(args, 'max_pixels', 2000000)
    min_pixels = getattr(args, 'min_pixels', 40000)
    
    for img_path in image_paths:
        try:
            full_path = os.path.join(args.image_folder, img_path)
            if not os.path.exists(full_path):
                print(f"  [Retrieval] Warning: Image file not found: {full_path}")
                continue
            
            original_image = fetch_image({'image': full_path, 'max_pixels': max_pixels})
            processed_image = process_image(original_image, max_pixels, min_pixels)
            images.append(processed_image)
        except Exception as e:
            logger.warning(f"Failed to load image from {img_path}: {e}", exc_info=True)
            print(f"  [Retrieval] Warning: Failed to load image from {img_path}: {e}")
    
    if images:
        pass
    else:
        print(f"  [Retrieval] Warning: Failed to load any images from original data source")
    
    return images if images else None


# ============================================================================
# Experience parsing helpers
# ============================================================================

def _parse_exp_blocks(section: str) -> Dict[str, str]:
    """Parse [id]\\ncontent blocks from a section string. Returns dict id -> content."""
    result = {}
    exp_blocks = re.split(r'\n\[([^\]]+)\]\n', section)
    for i in range(1, len(exp_blocks), 2):
        exp_id = exp_blocks[i]
        exp_content = exp_blocks[i + 1].strip().strip('-').strip() if i + 1 < len(exp_blocks) else ""
        if exp_content:
            result[exp_id] = exp_content
    return result


def _parse_exp_ids(section: str) -> set:
    """Parse experience IDs from [id] markers in a section string."""
    return set(re.findall(r'\[([^\]]+)\]', section))


def get_used_original_experiences(sample_dir: str) -> Dict[str, str]:
    """
    Get original experiences that were actually used (not skipped during rewrite).
    
    Returns the original content of experiences that survived the rewrite process.
    If rewrite was not used, returns the retrieved experiences as-is.
    
    Args:
        sample_dir: Sample directory path
        
    Returns:
        Dict mapping experience ID to original content, or empty dict if not available
    """
    retrieval_info_file = os.path.join(sample_dir, RETRIEVAL_INFO_FILENAME)
    if not os.path.exists(retrieval_info_file):
        return {}
    
    try:
        with open(retrieval_info_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_exps = {}
        rewritten_ids = set()
        
        if "ORIGINAL RETRIEVED EXPERIENCES" in content:
            orig_section = content.split("ORIGINAL RETRIEVED EXPERIENCES")[1]
            if "REWRITTEN EXPERIENCES" in orig_section:
                orig_section = orig_section.split("REWRITTEN EXPERIENCES")[0]
            elif "RETRIEVED EXPERIENCE CONTENT" in orig_section:
                orig_section = orig_section.split("RETRIEVED EXPERIENCE CONTENT")[0]
            original_exps = _parse_exp_blocks(orig_section)
        
        if "REWRITTEN EXPERIENCES" in content:
            rewrite_section = content.split("REWRITTEN EXPERIENCES")[1]
            rewritten_ids = _parse_exp_ids(rewrite_section)
        elif "RETRIEVED EXPERIENCE CONTENT" in content:
            rewrite_section = content.split("RETRIEVED EXPERIENCE CONTENT")[1]
            rewritten_ids = _parse_exp_ids(rewrite_section)
            if not original_exps:
                original_exps = _parse_exp_blocks(rewrite_section)
                rewritten_ids = set(original_exps.keys())
        
        return {k: v for k, v in original_exps.items() if k in rewritten_ids}
    
    except Exception as e:
        logger.warning(f"Failed to parse retrieval info: {e}")
        return {}


def save_retrieval_info(experience_retriever, sample_dir, retrieved_exps=None, original_retrieved_exps=None, rewrite_used=False, retrieval_info=None):
    """
    Save retrieval information to file, including rewrite information if applicable.
    
    Args:
        experience_retriever: ExperienceRetriever instance
        sample_dir: Sample directory path
        retrieved_exps: Final experiences dict (after rewrite if rewrite was used) or None
        original_retrieved_exps: Original retrieved experiences dict (before rewrite) or None
        rewrite_used: Whether rewrite was applied
        retrieval_info: Optional retrieval info dict (if provided, used instead of getting from retriever)
    """
    if retrieval_info is None:
        retrieval_info = experience_retriever.get_last_retrieval_info() if experience_retriever else None
    if not retrieval_info:
        return
    
    retrieval_info_file = os.path.join(sample_dir, RETRIEVAL_INFO_FILENAME)
    os.makedirs(sample_dir, exist_ok=True)
    
    with open(retrieval_info_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("EXPERIENCE RETRIEVAL INFORMATION\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"Original Query:\n{retrieval_info.get('original_query', 'N/A')}\n\n")
        f.write(f"Decomposition Used: {retrieval_info.get('decomposition_used', False)}\n")
        f.write(f"Rewrite Used: {rewrite_used}\n\n")
        
        if retrieval_info.get('decomposition_used', False):
            f.write(f"Subtasks ({len(retrieval_info.get('subtasks', []))}):\n")
            for i, subtask in enumerate(retrieval_info.get('subtasks', []), 1):
                f.write(f"\n  {i}. Type: {subtask.get('type', 'unknown')}\n")
                f.write(f"     Query: {subtask.get('query', '')}\n")
            
            f.write(f"\n\nRetrieval Details:\n")
            for i, detail in enumerate(retrieval_info.get('retrieval_details', []), 1):
                f.write(f"\n  {i}. Subtask: {detail.get('subtask_type', 'unknown')}\n")
                f.write(f"     Query: {detail.get('query', '')}\n")
                f.write(f"     Retrieved {detail.get('count', 0)} experiences:\n")
                for exp_id in detail.get('retrieved_experience_ids', []):
                    f.write(f"       - {exp_id}\n")
        
        f.write(f"\n\nFinal Retrieved Experiences ({retrieval_info.get('total_unique_experiences', 0)} total):\n")
        for exp_id in retrieval_info.get('retrieved_experiences', []):
            f.write(f"  - {exp_id}\n")
        
        if rewrite_used and original_retrieved_exps:
            f.write(f"\n\n{'=' * 80}\n")
            f.write("ORIGINAL RETRIEVED EXPERIENCES (Before Rewrite)\n")
            f.write("=" * 80 + "\n")
            for exp_id, exp_text in original_retrieved_exps.items():
                f.write(f"\n[{exp_id}]\n")
                f.write("-" * 80 + "\n")
                f.write(exp_text)
                f.write("\n\n")
        
        if retrieved_exps:
            section_title = "REWRITTEN EXPERIENCES (After Rewrite)" if rewrite_used else "RETRIEVED EXPERIENCE CONTENT"
            f.write(f"\n\n{'=' * 80}\n")
            f.write(f"{section_title}\n")
            f.write("=" * 80 + "\n")
            for exp_id, exp_text in retrieved_exps.items():
                f.write(f"\n[{exp_id}]\n")
                f.write("-" * 80 + "\n")
                f.write(exp_text)
                f.write("\n\n")
        else:
            f.write("\nNo relevant experiences were retrieved for this query.\n")


# ============================================================================
# Sample processing helpers
# ============================================================================

def compute_and_save_sample_summary(question_id, sample_rollout_results, rollouts_per_sample, sample_dir):
    """
    Compute and save sample-level pass@k and average@k metrics.
    
    Args:
        question_id: Question ID
        sample_rollout_results: List of rollout result dictionaries
        rollouts_per_sample: Number of rollouts per sample
        sample_dir: Sample directory path
        
    Returns:
        dict: Sample summary dictionary
    """
    if not sample_rollout_results:
        return None
    
    if sample_rollout_results and '_rollout_idx' in sample_rollout_results[0]:
        sorted_results = sorted(sample_rollout_results, key=lambda r: r.get('_rollout_idx', 0))
    else:
        sorted_results = sample_rollout_results

    accuracies = [r["accuracy_score"] for r in sorted_results]
    sample_summary = {
        "question_id": question_id,
        "num_rollouts": len(sorted_results),
        "accuracies": accuracies,
    }

    for k in range(1, len(accuracies) + 1):
        acc_at_k = accuracies[:k]
        sample_summary[f"pass@{k}"] = 1.0 if any(acc == 1.0 for acc in acc_at_k) else 0.0
        sample_summary[f"average@{k}"] = sum(acc_at_k) / k

    with open(os.path.join(sample_dir, SAMPLE_SUMMARY_FILENAME), 'w', encoding='utf-8') as f:
        json.dump(sample_summary, f, indent=4, ensure_ascii=False)
    
    print(f"  \nSample {question_id}: pass@{rollouts_per_sample}={sample_summary[f'pass@{rollouts_per_sample}']:.2f}, "
          f"average@{rollouts_per_sample}={sample_summary[f'average@{rollouts_per_sample}']:.4f}")
    
    return sample_summary


def check_sample_completed(sample_dir, args):
    """
    Check if a sample is already completed (has traj.jsonl with turn_idx and metrics.json).
    
    Args:
        sample_dir: Sample directory path
        args: Command line arguments
        
    Returns:
        dict or None: Sample info dict if completed, None otherwise
    """
    if not getattr(args, 'skip_completed', False):
        return None
    
    traj_file = os.path.join(sample_dir, 'traj.jsonl')
    metrics_file = os.path.join(sample_dir, 'metrics.json')
    
    is_complete = False
    if os.path.exists(traj_file) and os.path.exists(metrics_file):
        try:
            with open(traj_file, 'r', encoding='utf-8') as f:
                lines = [l for l in f if l.strip()]
                for line in lines:
                    try:
                        traj_data = json.loads(line)
                        if 'turn_idx' in traj_data and traj_data['turn_idx'] is not None:
                            is_complete = True
                            break
                    except (json.JSONDecodeError, KeyError) as e:
                        logger.debug(f"Error parsing trajectory line: {e}")
                        pass
        except Exception as e:
            logger.debug(f"Error reading trajectory file: {e}")
            pass
    
    if not is_complete:
        return None
    
    sample_rollout_results = []
    try:
        with open(metrics_file, 'r', encoding='utf-8') as f:
            metrics = json.load(f)
        
        num_turns = 0
        try:
            with open(traj_file, 'r', encoding='utf-8') as f:
                max_turn_idx = -1
                for line in f:
                    if line.strip():
                        try:
                            traj_data = json.loads(line)
                            turn_idx = traj_data.get('turn_idx')
                            if turn_idx is not None and turn_idx > max_turn_idx:
                                max_turn_idx = turn_idx
                        except (json.JSONDecodeError, KeyError) as e:
                            logger.debug(f"Error parsing trajectory line: {e}")
                            pass
            num_turns = max_turn_idx + 1 if max_turn_idx >= 0 else 0
        except Exception as e:
            logger.debug(f"Error reading trajectory file for turn count: {e}")
            pass
        
        conversation_history = [{"role": "assistant", "content": ""} for _ in range(num_turns)] if num_turns > 0 else []
        
        sample_rollout_results.append({
            'question_id': os.path.basename(sample_dir),
            'accuracy_score': metrics.get('accuracy_score', 0.0),
            'conversation_history': conversation_history,
        })
    except Exception as e:
        logger.debug(f"Error loading existing metrics: {e}")
        pass
    
    return {
        'sample_rollout_results': sample_rollout_results
    }


def prepare_sample_args(args):
    """
    Create a thread-safe copy of args for parallel execution.
    
    Args:
        args: Original command line arguments
        
    Returns:
        A shallow copy of args with deep-copied tool_configs
    """
    sample_args = copy.copy(args)
    
    if hasattr(args, 'tool_configs') and args.tool_configs:
        sample_args.tool_configs = copy.deepcopy(args.tool_configs)
    
    return sample_args


def get_sample_metadata(sample, sample_idx, output_dir):
    """
    Extract question_id and sample_dir from a sample.
    
    Args:
        sample: Data sample dictionary
        sample_idx: Sample index (for fallback question_id)
        output_dir: Output directory path
        
    Returns:
        tuple: (question_id, sample_dir)
    """
    question_id = sample.get('doc_id', sample.get('question_id', f"sample_{sample_idx}"))
    sample_dir = os.path.join(output_dir, question_id)
    return question_id, sample_dir


def compute_dataset_summary(data, args):
    """
    Compute and save dataset-level pass@k and average@k metrics.
    
    Args:
        data: List of data samples
        args: Command line arguments
        
    Returns:
        dict: Dataset summary dictionary or None if no summaries found
    """
    sample_summaries = []
    for sample in data:
        question_id = sample.get('doc_id', sample.get('question_id', f"sample_{data.index(sample)}"))
        summary_path = os.path.join(args.output_dir, question_id, SAMPLE_SUMMARY_FILENAME)
        if os.path.exists(summary_path):
            with open(summary_path, 'r') as f:
                sample_summaries.append(json.load(f))
    
    if not sample_summaries:
        return None
    
    dataset_summary = {
        "num_samples": len(sample_summaries),
        "num_rollouts": args.rollouts_per_sample,
        "k_metrics": {}
    }
    
    for k in range(1, args.rollouts_per_sample + 1):
        pass_values = [s[f"pass@{k}"] for s in sample_summaries if f"pass@{k}" in s]
        avg_values = [s[f"average@{k}"] for s in sample_summaries if f"average@{k}" in s]
        
        dataset_summary["k_metrics"][k] = {
            "pass_at_k": round(sum(pass_values) / len(pass_values), 4) if pass_values else 0.0,
            "average_at_k": round(sum(avg_values) / len(avg_values), 4) if avg_values else 0.0,
        }
    
    with open(os.path.join(args.output_dir, DATASET_SUMMARY_FILENAME), 'w', encoding='utf-8') as f:
        json.dump(dataset_summary, f, indent=4, ensure_ascii=False)
    
    print(f"Total samples: {dataset_summary['num_samples']}")
    print(f"Rollouts per sample: {dataset_summary['num_rollouts']}\n")
    for k in range(1, args.rollouts_per_sample + 1):
        metrics = dataset_summary["k_metrics"][k]
        print(f"k={k}:")
        print(f"  Pass@{k}:    {metrics['pass_at_k']:.4f}")
        print(f"  Average@{k}: {metrics['average_at_k']:.4f}")
    print("="*80)
    
    return dataset_summary


def reload_experiences(args, base_system_prompt, experience_retriever=None):
    """
    Refresh the cluster experience retriever from disk.
    """
    if not (getattr(args, 'experience_enable', False) and args.experience_library):
        return base_system_prompt

    if experience_retriever is None:
        return base_system_prompt

    try:
        exps = load_experiences(args.experience_library)
        print(f"  Updating experience retriever with {len(exps)} experiences...")
        experience_retriever.update_experiences(exps, incremental=True)
        print(f"  Experience retriever updated successfully.")
    except Exception as e:
        print(f"  Warning: Failed to reload experiences: {e}")
        import traceback
        traceback.print_exc()

    return base_system_prompt


# ============================================================================
# Pipeline parallel processing
# ============================================================================

def execute_pipeline_parallel_processing(
    samples: List[Dict],
    sample_indices: List[int],
    sample_results_dict: Dict,
    args: Any,
    experience_retriever: Optional[Any],
    base_system_prompt: str,
    sampling_params: Dict,
    run_single_rollout_func: Callable,
    progress_desc: Optional[str] = None,
    all_results_list: Optional[List] = None
) -> None:
    """
    Executes a pipeline parallel processing of samples to maximize hardware utilization:
    
    Pipeline stages:
    1. Retrieval Stage: Submits experience retrieval tasks for samples. To prevent 
       "worker starvation" for the reasoning stage, retrieval tasks are submitted 
       gradually as slots open up.
    2. Reasoning (Rollout) Stage: As soon as a sample's retrieval (and optional rewrite)
       completes, all its rollouts (pass@k) are immediately submitted to the pool.
    3. Management Stage: A unified monitoring loop uses wait() to react to any completed 
       future (retrieve or rollout), ensuring smooth transition between stages.
    
    Args:
        samples: List of sample dictionaries from the dataset.
        sample_indices: List of actual sample indices (mapping to original dataset).
        sample_results_dict: Shared dictionary to store results for each sample.
        args: Command line arguments containing configuration (num_workers, etc).
        experience_retriever: ExperienceRetriever instance. If None, retrieval stage 
            may still run if skill_inference is enabled (skill-only mode).
        base_system_prompt: The initial system prompt used if retrieval is disabled or fails.
        sampling_params: Parameters for the LLM call (temperature, top_p, etc).
        run_single_rollout_func: The core function to execute one rollout (one pass).
        progress_desc: String description for the tqdm progress bar.
        all_results_list: Optional list to collect every single rollout result across all samples.
        
    Returns:
        None (results are mutated in sample_results_dict and all_results_list).
    """
    from tqdm import tqdm  # type: ignore

    with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        active_futures = {}
        pending_retrieval_samples = []
        
        for local_idx, sample in enumerate(samples):
            actual_sample_idx = sample_indices[local_idx]
            question_id = sample_results_dict[actual_sample_idx]['question_id']
            sample_dir = sample_results_dict[actual_sample_idx]['sample_dir']
            sample_args = prepare_sample_args(args)
            sample_args.output_dir = os.path.dirname(sample_dir)
            pending_retrieval_samples.append({
                'sample_idx': actual_sample_idx,
                'sample': sample,
                'sample_args': sample_args,
                'question_id': question_id,
                'sample_dir': sample_dir
            })

        total_rollouts = len(samples) * args.rollouts_per_sample
        pbar = tqdm(total=total_rollouts, desc=progress_desc or "Processing rollouts")
        
        rollout_counts = {}
        completed_samples = set()
        next_sample_idx = 0
        
        skill_enabled = getattr(args, 'skill_inference', False) and getattr(args, 'skill_library', None)
        needs_retrieval_stage = experience_retriever is not None or skill_enabled
        
        max_concurrent_retrievals = args.num_workers if needs_retrieval_stage else 0
        active_retrieval_count = 0

        if not needs_retrieval_stage:
            for item in pending_retrieval_samples:
                actual_sample_idx = item['sample_idx']
                sample = item['sample']
                sample_args = item['sample_args']
                sample_args._sample_system_prompt = base_system_prompt
                for rollout_idx in range(args.rollouts_per_sample):
                    f = executor.submit(run_single_rollout_func, sample, sample_args, sampling_params, rollout_idx)
                    active_futures[f] = {'type': 'rollout', 'sample_idx': actual_sample_idx, 'rollout_idx': rollout_idx}
            next_sample_idx = len(pending_retrieval_samples)
        else:
            while next_sample_idx < len(pending_retrieval_samples) and active_retrieval_count < max_concurrent_retrievals:
                item = pending_retrieval_samples[next_sample_idx]
                f = executor.submit(
                    retrieve_experiences_for_sample,
                    item['sample'], item['sample_args'], experience_retriever, base_system_prompt,
                    item['question_id'], item['sample_dir'], update_global_prompts=False
                )
                active_futures[f] = {
                    'type': 'retrieve', 
                    'sample_idx': item['sample_idx'], 
                    'sample': item['sample'], 
                    'sample_args': item['sample_args'],
                    'question_id': item['question_id'],
                    'sample_dir': item['sample_dir']
                }
                active_retrieval_count += 1
                next_sample_idx += 1

        while active_futures:
            done_futures, _ = wait(active_futures.keys(), return_when=FIRST_COMPLETED)
            
            for future in done_futures:
                if future not in active_futures:
                    continue
                    
                info = active_futures.pop(future)
                
                if info['type'] == 'retrieve':
                    active_retrieval_count -= 1
                    actual_sample_idx = info['sample_idx']
                    sample = info['sample']
                    sample_args = info['sample_args']
                    question_id = info['question_id']
                    
                    try:
                        future.result()
                        print(f"  [Pipeline] Sample {question_id}: Retrieve completed, submitting {args.rollouts_per_sample} rollouts...")
                    except Exception as e:
                        print(f"  [Pipeline] Error retrieving for sample {question_id}: {e}. Falling back to base prompt.")
                        sample_args._sample_system_prompt = base_system_prompt
                    
                    for rollout_idx in range(args.rollouts_per_sample):
                        f_rollout = executor.submit(run_single_rollout_func, sample, sample_args, sampling_params, rollout_idx)
                        active_futures[f_rollout] = {'type': 'rollout', 'sample_idx': actual_sample_idx, 'rollout_idx': rollout_idx}
                    
                    if next_sample_idx < len(pending_retrieval_samples):
                        item = pending_retrieval_samples[next_sample_idx]
                        f_new = executor.submit(
                            retrieve_experiences_for_sample,
                            item['sample'], item['sample_args'], experience_retriever, base_system_prompt,
                            item['question_id'], item['sample_dir'], update_global_prompts=False
                        )
                        active_futures[f_new] = {
                            'type': 'retrieve', 
                            'sample_idx': item['sample_idx'], 
                            'sample': item['sample'], 
                            'sample_args': item['sample_args'],
                            'question_id': item['question_id'],
                            'sample_dir': item['sample_dir']
                        }
                        active_retrieval_count += 1
                        next_sample_idx += 1
                
                elif info['type'] == 'rollout':
                    actual_sample_idx = info['sample_idx']
                    rollout_idx = info['rollout_idx']
                    
                    try:
                        result = future.result()
                        if result:
                            result['_rollout_idx'] = rollout_idx
                            sample_results_dict[actual_sample_idx]['results'].append(result)
                            if all_results_list is not None:
                                all_results_list.append(result)
                    except Exception as e:
                        print(f"  [Pipeline] Error in rollout {rollout_idx} for sample {actual_sample_idx}: {e}")
                    
                    pbar.update(1)
                    
                    if actual_sample_idx not in rollout_counts:
                        rollout_counts[actual_sample_idx] = 0
                    rollout_counts[actual_sample_idx] += 1
                    
                    if rollout_counts[actual_sample_idx] == args.rollouts_per_sample and actual_sample_idx not in completed_samples:
                        completed_samples.add(actual_sample_idx)
                        sample_info = sample_results_dict[actual_sample_idx]
                        if sample_info['results']:
                            compute_and_save_sample_summary(
                                sample_info['question_id'], sample_info['results'],
                                args.rollouts_per_sample, sample_info['sample_dir']
                            )

        pbar.close()
