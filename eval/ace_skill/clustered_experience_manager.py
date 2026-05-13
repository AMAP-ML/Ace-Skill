"""
Clustered Experience Manager

Partitions the experience pool by cluster so that retrieval and merging
happen within each cluster independently.  Provides the same duck-type
interface as ``ExperienceRetriever`` so call-sites can remain unchanged.
"""

import os
import json
import logging
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict

from .experience_retriever import ExperienceRetriever
from .experience_utils import load_existing, save_library

logger = logging.getLogger(__name__)


class ClusteredExperienceManager:
    """Per-cluster experience libraries and retrievers.

    Each cluster gets its own ``ExperienceRetriever`` instance backed by a
    separate JSON library file.  The public ``retrieve`` /
    ``retrieve_with_decomposition`` methods accept a ``doc_id`` argument that
    is used to route the call to the correct cluster retriever.
    """

    def __init__(
        self,
        cluster_mapping: Dict[str, int],
        base_library_path: str,
        llm_client: Optional[Any] = None,
        **retriever_kwargs,
    ):
        """
        Args:
            cluster_mapping: ``{doc_id: cluster_id}`` mapping loaded from
                ``*_doc_id_to_cluster.json``.
            base_library_path: Base path used to derive per-cluster experience
                library files (e.g. ``memory_bank/run1/experiences.json`` ->
                ``experiences_cluster_<id>.json``) via
                ``_cluster_library_path(cluster_id)``.  The base file itself is
                never read or written.
            llm_client: Optional ``ExperienceLLM`` instance shared among all
                cluster retrievers (used for decomposition / rewrite).
            **retriever_kwargs: Forwarded to each ``ExperienceRetriever``
                constructor (``embedding_model``, ``embedding_api_key``,
                ``embedding_endpoint``, ``cache_dir``, ``enable_cache``).
        """
        self.cluster_mapping = cluster_mapping
        self.base_library_path = base_library_path
        self._llm_client = llm_client
        self.retriever_kwargs = retriever_kwargs

        self.cluster_ids: List[int] = sorted(set(cluster_mapping.values()))
        self.retrievers: Dict[int, ExperienceRetriever] = {}

        for cid in self.cluster_ids:
            lib_path = self._cluster_library_path(cid)
            exps = load_existing(lib_path) if os.path.exists(lib_path) else {}
            self.retrievers[cid] = ExperienceRetriever(
                experiences=exps,
                llm_client=llm_client,
                experience_library_path=lib_path,
                **retriever_kwargs,
            )

        total_exps = sum(len(r.experiences) for r in self.retrievers.values())
        print(
            f"ClusteredExperienceManager: {len(self.cluster_ids)} clusters, "
            f"{total_exps} total experiences"
        )

    # ------------------------------------------------------------------
    # Path helpers
    # ------------------------------------------------------------------

    def _cluster_library_path(self, cluster_id: int) -> str:
        """Derive per-cluster library path from the base path.

        ``experiences.json`` -> ``experiences_cluster_0.json``, etc.
        """
        base, ext = os.path.splitext(self.base_library_path)
        return f"{base}_cluster_{cluster_id}{ext}"

    def get_cluster_id(self, doc_id: str) -> Optional[int]:
        return self.cluster_mapping.get(doc_id)

    # ------------------------------------------------------------------
    # Duck-type compatible retrieval interface
    # ------------------------------------------------------------------

    @property
    def llm_client(self):
        return self._llm_client

    def retrieve(
        self,
        query: str,
        top_k: int = 3,
        min_similarity: float = 0.0,
        doc_id: Optional[str] = None,
    ) -> Tuple[Dict[str, str], Dict[str, Any]]:
        retriever = self._retriever_for_doc(doc_id)
        if retriever is None:
            return {}, self._empty_info(query)
        return retriever.retrieve(query=query, top_k=top_k, min_similarity=min_similarity)

    def retrieve_with_decomposition(
        self,
        task_description: str,
        top_k: int = 3,
        min_similarity: float = 0.0,
        subtask_top_k: Optional[int] = None,
        images: Optional[list] = None,
        doc_id: Optional[str] = None,
    ) -> Tuple[Dict[str, str], Dict[str, Any]]:
        retriever = self._retriever_for_doc(doc_id)
        if retriever is None:
            return {}, self._empty_info(task_description)
        return retriever.retrieve_with_decomposition(
            task_description=task_description,
            top_k=top_k,
            min_similarity=min_similarity,
            subtask_top_k=subtask_top_k,
            images=images,
        )

    def get_embedding_stats(self) -> Dict[str, Any]:
        """Aggregate embedding stats across all cluster retrievers."""
        total_exps = 0
        embedded = 0
        for r in self.retrievers.values():
            stats = r.get_embedding_stats()
            total_exps += stats["total_experiences"]
            embedded += stats["embedded_count"]
        first = next(iter(self.retrievers.values()), None)
        return {
            "total_experiences": total_exps,
            "embedded_count": embedded,
            "missing_count": total_exps - embedded,
            "embedding_model": first.embedding_model if first else "N/A",
            "cache_enabled": first.enable_cache if first else False,
            "cache_path": "per-cluster",
            "num_clusters": len(self.cluster_ids),
        }

    def get_last_retrieval_info(self) -> Optional[Dict[str, Any]]:
        for r in self.retrievers.values():
            info = r.get_last_retrieval_info()
            if info is not None:
                return info
        return None

    # ------------------------------------------------------------------
    # Experience update helpers (used after batch merge in train.py)
    # ------------------------------------------------------------------

    def update_experiences(self, new_experiences: Dict[str, str], incremental: bool = True):
        """Reload ALL per-cluster libraries from disk.

        Called by ``reload_experiences`` between training batches to refresh
        every cluster retriever.  For per-cluster updates (e.g. right after a
        cluster-local merge), prefer ``update_cluster_experiences``.
        The ``new_experiences`` argument is ignored on purpose — each cluster
        loads its own JSON file from disk.
        """
        for cid in self.cluster_ids:
            lib_path = self._cluster_library_path(cid)
            if os.path.exists(lib_path):
                exps = load_existing(lib_path)
                if cid in self.retrievers:
                    self.retrievers[cid].update_experiences(exps, incremental=incremental)

    def update_cluster_experiences(
        self, cluster_id: int, new_experiences: Dict[str, str], incremental: bool = True,
    ):
        """Update a single cluster retriever after merge."""
        if cluster_id in self.retrievers:
            self.retrievers[cluster_id].update_experiences(new_experiences, incremental=incremental)
        else:
            logger.warning(f"No retriever for cluster {cluster_id}, creating one")
            lib_path = self._cluster_library_path(cluster_id)
            self.retrievers[cluster_id] = ExperienceRetriever(
                experiences=new_experiences,
                llm_client=self._llm_client,
                experience_library_path=lib_path,
                **self.retriever_kwargs,
            )

    def load_cluster_library(self, cluster_id: int) -> Dict[str, str]:
        lib_path = self._cluster_library_path(cluster_id)
        return load_existing(lib_path) if os.path.exists(lib_path) else {}

    def save_cluster_library(self, cluster_id: int, experiences: Dict[str, str]):
        lib_path = self._cluster_library_path(cluster_id)
        os.makedirs(os.path.dirname(lib_path), exist_ok=True)
        save_library(lib_path, experiences)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _retriever_for_doc(self, doc_id: Optional[str]) -> Optional[ExperienceRetriever]:
        if doc_id is None:
            logger.warning("doc_id is None in cluster mode; cannot route to cluster retriever")
            return None
        cid = self.get_cluster_id(doc_id)
        if cid is None:
            logger.warning(f"doc_id '{doc_id}' not found in cluster mapping")
            return None
        retriever = self.retrievers.get(cid)
        if retriever is None:
            logger.warning(f"No retriever for cluster {cid}")
        return retriever

    @staticmethod
    def _empty_info(query: str) -> Dict[str, Any]:
        return {
            "original_query": query,
            "decomposition_used": False,
            "subtasks": [],
            "retrieved_experiences": [],
            "retrieval_details": [],
            "total_unique_experiences": 0,
        }
