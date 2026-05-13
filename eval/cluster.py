#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import requests
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
DEFAULT_ENV_PATH = ROOT / ".env"

EMBEDDING_BATCH_SIZE = 10
EMBEDDING_API_TIMEOUT = 60
EMBEDDING_MAX_RETRIES = 3
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"


def load_dotenv(path: Path | None) -> None:
    """Load .env (KEY=VALUE) from path, ignoring comments and blank lines. Matches scripts/cluster_tir_questions.py."""
    p = path or DEFAULT_ENV_PATH
    if not p.exists():
        return
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                k, _, v = line.partition("=")
                k, v = k.strip(), v.strip().strip('"').strip("'")
                if k:
                    os.environ[k] = v


def get_embedding_config() -> tuple[str, str, str]:
    """Return (api_key, endpoint, model). Uses only EXPERIENCE_EMBEDDING_*; endpoint is normalized to end with /v1."""
    api_key = ""
    endpoint = ""
    if endpoint and not endpoint.endswith("/v1"):
        endpoint = endpoint + "/v1"
    model = os.environ.get("EXPERIENCE_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
    return api_key, endpoint, model


def post_embeddings(
    endpoint: str,
    api_key: str,
    model: str,
    texts: list[str],
    batch_size: int = EMBEDDING_BATCH_SIZE,
) -> list[np.ndarray | None]:
    """Batch-request embeddings for texts; returns a list aligned with texts, with None for failed positions."""
    url = f"{endpoint}/embeddings"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    all_embeddings: list[np.ndarray | None] = []
    total_batches = (len(texts) + batch_size - 1) // batch_size

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        batch_num = i // batch_size + 1
        payload = {"model": model, "input": batch}
        data = None
        for attempt in range(EMBEDDING_MAX_RETRIES):
            try:
                r = requests.post(
                    url, headers=headers, json=payload, timeout=EMBEDDING_API_TIMEOUT
                )
                r.raise_for_status()
                data = r.json()
                break
            except Exception as e:
                if attempt < EMBEDDING_MAX_RETRIES - 1:
                    time.sleep(2**attempt)
                    continue
                print(f"  Error: embedding batch {batch_num} failed: {e}", file=sys.stderr)
                all_embeddings.extend([None] * len(batch))
                break
        else:
            continue

        # If the request failed, data is None — skip processing
        if data is None:
            continue

        # Parse data (compatible with OpenAI format data[].embedding, and some APIs' nested data[].embedding fields)
        items = data.get("data", [])
        for j, item in enumerate(items):
            emb = None
            if isinstance(item, dict):
                emb = item.get("embedding")
                # A few APIs wrap the vector inside an array at data[].embedding, requiring an extra [0]
                if emb is None and "embedding" in item:
                    e = item["embedding"]
                    if isinstance(e, (list, tuple)) and len(e) > 0 and isinstance(e[0], (list, tuple)):
                        emb = e[0]
            all_embeddings.append(np.array(emb, dtype=np.float32) if emb else None)
        while len(all_embeddings) < i + len(batch):
            all_embeddings.append(None)

        ok = sum(1 for k in range(len(batch)) if all_embeddings[i + k] is not None)
        if ok == 0 and batch_num == 1:
            # When the first batch fully fails, print the response structure to aid debugging
            print("  Debug: no embeddings parsed. Response keys:", list(data.keys()), file=sys.stderr)
            if items:
                first = items[0] if isinstance(items[0], dict) else {}
                print("  Debug: first item keys:", list(first.keys()) if isinstance(first, dict) else type(first), file=sys.stderr)
            if "error" in data:
                print("  Debug: API error:", data.get("error"), file=sys.stderr)
            else:
                body_preview = json.dumps(data, ensure_ascii=False)[:500]
                print("  Debug: response preview:", body_preview, file=sys.stderr)
        if total_batches > 1 or ok > 0:
            print(f"  Progress: batch {batch_num}/{total_batches} ({ok}/{len(batch)} ok)")

    return all_embeddings[: len(texts)]


def load_benchmark_json(path: Path) -> list[dict[str, Any]]:
    """Load benchmark JSON; expects a list of dicts where each item has doc_id and problem."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON array, got {type(data)}")
    for i, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"Item at index {i} is not a dict")
        if "doc_id" not in item or "problem" not in item:
            raise ValueError(
                f"Item at index {i} missing 'doc_id' or 'problem' (keys: {list(item.keys())})"
            )
    return data


def embed_items(
    items: list[dict[str, Any]],
    endpoint: str,
    api_key: str,
    model: str,
    embedding_batch_size: int,
) -> tuple[list[Any], list[np.ndarray | None], list[int]]:
    """Embed each item's problem. Returns (doc_ids, embeddings, valid_idx)."""
    texts = [item["problem"] for item in items]
    doc_ids = [item["doc_id"] for item in items]
    embeddings = post_embeddings(endpoint, api_key, model, texts, batch_size=embedding_batch_size)
    valid_idx = [i for i, e in enumerate(embeddings) if e is not None]
    return doc_ids, embeddings, valid_idx


def build_doc_id_to_cluster(
    doc_ids: list[Any],
    total_len: int,
    valid_idx: list[int],
    labels: np.ndarray,
    allow_noise: bool,
) -> dict[Any, int]:
    """Build a doc_id -> cluster_id mapping from labels of valid samples."""
    idx_to_label = {valid_idx[j]: int(labels[j]) for j in range(len(valid_idx))}
    default_cid = -1 if allow_noise else 0
    doc_id_to_cluster = {}
    for i in range(total_len):
        cid = idx_to_label.get(i, default_cid)
        doc_id_to_cluster[doc_ids[i]] = cid
    return doc_id_to_cluster


def run_train_test_clustering(
    train_path: Path,
    test_path: Path,
    train_mapping_path: Path | None,
    test_mapping_path: Path | None,
    n_clusters: int,
    env_path: Path | None,
    embedding_batch_size: int,
    allow_noise: bool,
) -> None:
    """Fit K-Means on the train set, predict clusters for the test set, and write each mapping JSON."""
    load_dotenv(env_path or ROOT / ".env")
    api_key, endpoint, model = get_embedding_config()
    fallback_cluster_id = -1 if allow_noise else 0
    if not api_key or not endpoint:
        print("Please configure EXPERIENCE_EMBEDDING_API_KEY and EXPERIENCE_EMBEDDING_ENDPOINT in .env", file=sys.stderr)
        sys.exit(1)

    print(f"Embedding model: {model}, batch_size: {embedding_batch_size}")

    print(f"Loading train: {train_path}")
    train_items = load_benchmark_json(train_path)
    print(f"Train items: {len(train_items)}")
    train_doc_ids, train_embeddings, train_valid_idx = embed_items(
        train_items, endpoint, api_key, model, embedding_batch_size
    )
    if not train_valid_idx:
        print("Error: no successful train embeddings", file=sys.stderr)
        sys.exit(1)
    if len(train_valid_idx) < len(train_items):
        print(
            f"Warning: {len(train_items) - len(train_valid_idx)} train items failed embedding, "
            f"will assign cluster_id={fallback_cluster_id}",
            file=sys.stderr,
        )

    X_train = np.vstack([train_embeddings[i] for i in train_valid_idx]).astype(np.float32)
    n_valid = len(train_valid_idx)
    k = min(n_clusters, n_valid)
    if k <= 0:
        print("Error: invalid effective cluster count", file=sys.stderr)
        sys.exit(1)
    if k < n_clusters:
        print(
            f"Warning: requested n_clusters={n_clusters}, but only {n_valid} valid train embeddings; use k={k}",
            file=sys.stderr,
        )

    print(f"Training K-Means on train set: n_clusters={k}")
    X_train_norm = normalize(X_train, norm="l2", axis=1)
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    train_labels = kmeans.fit_predict(X_train_norm)

    train_doc_id_to_cluster = build_doc_id_to_cluster(
        train_doc_ids, len(train_items), train_valid_idx, train_labels, allow_noise
    )
    train_out_path = train_mapping_path or train_path.parent / (train_path.stem + "_doc_id_to_cluster.json")
    train_out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(train_out_path, "w", encoding="utf-8") as f:
        json.dump(train_doc_id_to_cluster, f, ensure_ascii=False, indent=2)
    print(f"Wrote train doc_id -> cluster_id mapping: {train_out_path}")

    print(f"Loading test: {test_path}")
    test_items = load_benchmark_json(test_path)
    print(f"Test items: {len(test_items)}")
    test_doc_ids, test_embeddings, test_valid_idx = embed_items(
        test_items, endpoint, api_key, model, embedding_batch_size
    )
    if len(test_valid_idx) < len(test_items):
        print(
            f"Warning: {len(test_items) - len(test_valid_idx)} test items failed embedding, "
            f"will assign cluster_id={fallback_cluster_id}",
            file=sys.stderr,
        )

    test_labels = np.array([], dtype=np.int32)
    if test_valid_idx:
        X_test = np.vstack([test_embeddings[i] for i in test_valid_idx]).astype(np.float32)
        X_test_norm = normalize(X_test, norm="l2", axis=1)
        test_labels = kmeans.predict(X_test_norm)

    test_doc_id_to_cluster = build_doc_id_to_cluster(
        test_doc_ids, len(test_items), test_valid_idx, test_labels, allow_noise
    )
    test_out_path = test_mapping_path or test_path.parent / (test_path.stem + "_doc_id_to_cluster.json")
    test_out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(test_out_path, "w", encoding="utf-8") as f:
        json.dump(test_doc_id_to_cluster, f, ensure_ascii=False, indent=2)
    print(f"Wrote test doc_id -> cluster_id mapping: {test_out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fit K-Means on the training set and predict cluster ids for the test set, writing two JSON files next to the inputs."
    )
    parser.add_argument(
        "train_json",
        type=Path,
        help="Training set JSON path, e.g. benchmark/TIR-Bench/train_shuffled.json",
    )
    parser.add_argument(
        "test_json",
        type=Path,
        help="Test set JSON path, e.g. benchmark/TIR-Bench/test_shuffled.json",
    )
    parser.add_argument(
        "--train-mapping",
        type=Path,
        default=None,
        help="Training set mapping JSON path (default: <stem>_doc_id_to_cluster.json in the training set directory)",
    )
    parser.add_argument(
        "--test-mapping",
        type=Path,
        default=None,
        help="Test set mapping JSON path (default: <stem>_doc_id_to_cluster.json in the test set directory)",
    )
    parser.add_argument(
        "-k", "--n-clusters",
        type=int,
        default=5,
        metavar="K",
        help="K-Means cluster count K (default: 5)",
    )
    parser.add_argument(
        "--allow-noise",
        action="store_true",
        help="mark samples with failed embedding as -1; default to cluster 0",
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        default=None,
        help=f".env path (default: {ROOT / '.env'})",
    )
    parser.add_argument(
        "--embedding-batch-size",
        type=int,
        default=EMBEDDING_BATCH_SIZE,
        help=f"Embedding batch size (default: {EMBEDDING_BATCH_SIZE})",
    )
    args = parser.parse_args()

    run_train_test_clustering(
        train_path=args.train_json,
        test_path=args.test_json,
        train_mapping_path=args.train_mapping,
        test_mapping_path=args.test_mapping,
        n_clusters=args.n_clusters,
        env_path=args.env_file,
        embedding_batch_size=args.embedding_batch_size,
        allow_noise=args.allow_noise,
    )


if __name__ == "__main__":
    main()
