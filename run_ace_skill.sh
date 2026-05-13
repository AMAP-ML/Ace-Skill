#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

usage() {
    echo "Usage: bash run_ace_skill.sh [train|inference|infer|all]"
}

MODE="${1:-}"

if [[ -z "$MODE" ]]; then
    usage
    exit 1
fi

# Allow overriding by environment variables.
DATASET_NAME="${DATASET_NAME:-tir-bench}"
RUN_ID="${RUN_ID:-run}"
INF_ID="${INF_ID:-run-infer}"




API_KEY_1="EMPTY"
END_POINT_1=""
API_KEY_2="EMPTY"
END_POINT_2=""
EMBEDDING_API_KEY=""
EMBEDDING_ENDPOINT=""


# ============================================================================
# Reasoning Model
# ============================================================================
export REASONING_MODEL_NAME="Qwen3.5-35B-A3B"
export REASONING_API_KEY="$API_KEY_1"
export REASONING_END_POINT="$END_POINT_1"
export REASONING_API_KEY_2="$API_KEY_2"
export REASONING_END_POINT_2="$END_POINT_2"

# ============================================================================
# Verifier Model
# ============================================================================
export VERIFIER_MODEL_NAME="Qwen3.5-35B-A3B"
export VERIFIER_API_KEY="$API_KEY_2"
export VERIFIER_END_POINT="$END_POINT_2"

# ============================================================================
# Experience Model
# ============================================================================
export EXPERIENCE_MODEL_NAME="Qwen3.5-35B-A3B"
export EXPERIENCE_API_KEY="$API_KEY_1"
export EXPERIENCE_END_POINT="$END_POINT_1"
export EXPERIENCE_API_KEY_2="$API_KEY_2"
export EXPERIENCE_END_POINT_2="$END_POINT_2"

# Optional: Fallback API for experience generation
export EXPERIENCE_EMBEDDING_MODEL="text-embedding-3-small"
export EXPERIENCE_EMBEDDING_API_KEY="$EMBEDDING_API_KEY"
export EXPERIENCE_EMBEDDING_ENDPOINT="$EMBEDDING_ENDPOINT"

# ============================================================================
# Function Calling Configuration
# ============================================================================
export JINA_API_KEY=""
export SERPAPI_KEY=""
export ENABLE_FUNCTION_CALLING="true"

TOOL_CONFIG_PATH="eval/configs/tool_configs.yaml"
IMAGE_SEARCH_MAX_CALLS=5
WEB_SEARCH_MAX_CALLS=7

# ============================================================================
# Inference Parameters
# ============================================================================

MAX_TOTAL_TOKENS=65536
MAX_COMPLETION_TOKENS=16384
MAX_TURNS=10
MAX_IMAGES=100
REASONING_TEMPERATURE=0.7
REASONING_TOP_P=0.8
REASONING_TOP_K=20
REASONING_PRESENCE_PENALTY=1.5
REASONING_REPETITION_PENALTY=1.0
EXPERIENCE_TEMPERATURE=0.7
EXPERIENCE_TOP_P=0.8
EXPERIENCE_TOP_K=20
EXPERIENCE_PRESENCE_PENALTY=1.5
EXPERIENCE_REPETITION_PENALTY=1.0

# ============================================================================
# Experience Parameters
# ============================================================================

EXPERIENCE_MAX_OPS=3
EXPERIENCE_MAX_ITEMS=120

# Experience Retrieval Parameters
EXPERIENCE_RETRIEVAL_TOP_K=3
EXPERIENCE_LIBRARY="memory_bank/${RUN_ID}/experiences.json"

# ============================================================================
# Skill Parameters
# ============================================================================

SKILL_LIBRARY="memory_bank/${RUN_ID}/SKILL.md"
SKILL_MAX_LENGTH=1000

# ============================================================================
# Weighted Sampling Parameters (optional)
# ============================================================================
SAMPLING_CONFIG="${SAMPLING_CONFIG:-}"
MAX_SAMPLING_SAMPLES="${MAX_SAMPLING_SAMPLES:-}"

# ============================================================================
# Running Settings
# ============================================================================
export ENABLE_FUNCTION_CALLING="true"
if [[ "$DATASET_NAME" == "visualtoolbench" ]]; then
  export ENABLED_TOOLS="web_search, visit, code_interpreter"
  SYSTEM_PROMPT_TYPE="multi_tool_agent"
  TRAIN_DATA_PATH="benchmark/VisualToolBench/train_single_turn_hybrid_tool_reasoning_shuffled.json"
  INFER_DATA_PATH="benchmark/VisualToolBench/test_single_turn_hybrid_tool_reasoning_shuffled.json"
elif [[ "$DATASET_NAME" == "tir-bench" ]]; then
  export ENABLED_TOOLS="code_interpreter"
  SYSTEM_PROMPT_TYPE="multi_tool_agent_code"
  TRAIN_DATA_PATH="benchmark/TIR-Bench/train_shuffled.json"
  INFER_DATA_PATH="benchmark/TIR-Bench/test_shuffled.json"
else
  echo "Unsupported DATASET_NAME: $DATASET_NAME"
  exit 1
fi

TRAIN_EXPERIENCE_LARGE_BATCH=8
INFER_EXPERIENCE_LARGE_BATCH=32
TRAIN_NUM_WORKERS=8
INFER_NUM_WORKERS=32
IMAGE_DIR="benchmark"
TRAIN_OUTPUT_DIR="output/${RUN_ID}"
TRAIN_LOG_FILE="logs/${RUN_ID}.log"
TRAIN_ROLLOUTS_PER_SAMPLE=4
INFER_OUTPUT_DIR="output/${INF_ID}"
INFER_LOG_FILE="logs/${INF_ID}.log"
INFER_ROLLOUTS_PER_SAMPLE=4

TRAIN_ONLY_ARGS=(
    --experience-online-generate
    --experience-library-update
    --experience-max-ops "$EXPERIENCE_MAX_OPS"
    --experience-refine
    --experience-max-items "$EXPERIENCE_MAX_ITEMS"
    --skill-refine
    --skill-max-length "$SKILL_MAX_LENGTH"
)

if [[ -n "$SAMPLING_CONFIG" ]]; then
    TRAIN_ONLY_ARGS+=(--sampling-config "$SAMPLING_CONFIG")
    if [[ -n "$MAX_SAMPLING_SAMPLES" ]]; then
        TRAIN_ONLY_ARGS+=(--max-sampling-samples "$MAX_SAMPLING_SAMPLES")
    fi
fi

derive_cluster_mapping_path() {
    local input_file="$1"
    echo "${input_file%.*}_doc_id_to_cluster.json"
}

run_phase() {
    local phase="$1"
    local entrypoint="$2"
    shift 2

    local data_path=""
    local output_dir=""
    local log_file=""
    local rollouts_per_sample=""
    local num_workers=""
    local experience_large_batch=""

    case "$phase" in
        train)
            data_path="$TRAIN_DATA_PATH"
            output_dir="$TRAIN_OUTPUT_DIR"
            log_file="$TRAIN_LOG_FILE"
            rollouts_per_sample="$TRAIN_ROLLOUTS_PER_SAMPLE"
            num_workers="$TRAIN_NUM_WORKERS"
            experience_large_batch="$TRAIN_EXPERIENCE_LARGE_BATCH"
            ;;
        inference)
            data_path="$INFER_DATA_PATH"
            output_dir="$INFER_OUTPUT_DIR"
            log_file="$INFER_LOG_FILE"
            rollouts_per_sample="$INFER_ROLLOUTS_PER_SAMPLE"
            num_workers="$INFER_NUM_WORKERS"
            experience_large_batch="$INFER_EXPERIENCE_LARGE_BATCH"
            ;;
        *)
            echo "Unknown phase: $phase"
            exit 1
            ;;
    esac

    local mapping_path
    mapping_path="$(derive_cluster_mapping_path "$data_path")"
    if [[ ! -f "$mapping_path" ]]; then
        echo "Cluster mapping file not found: $mapping_path"
        echo "Please prepare the cluster mapping file before running this script."
        exit 1
    fi

    mkdir -p "$(dirname "$output_dir")" \
             "$(dirname "$log_file")" \
             "$(dirname "$EXPERIENCE_LIBRARY")" \
             "$(dirname "$SKILL_LIBRARY")"

    local common_args=(
        --input-file "$data_path"
        --image-folder "$IMAGE_DIR"
        --output-dir "$output_dir"
        --reasoning-temperature "$REASONING_TEMPERATURE"
        --reasoning-top-p "$REASONING_TOP_P"
        --max-turns "$MAX_TURNS"
        --max-images "$MAX_IMAGES"
        --max-total-tokens "$MAX_TOTAL_TOKENS"
        --max-completion-tokens "$MAX_COMPLETION_TOKENS"
        --system-prompt-key "$SYSTEM_PROMPT_TYPE"
        --num-workers "$num_workers"
        --tool-config-path "$TOOL_CONFIG_PATH"
        --rollouts-per-sample "$rollouts_per_sample"
        --image-search-max-calls "$IMAGE_SEARCH_MAX_CALLS"
        --web-search-max-calls "$WEB_SEARCH_MAX_CALLS"
        --skill-enable
        --skill-library "$SKILL_LIBRARY"
        --skill-inference
        --experience-enable
        --experience-library "$EXPERIENCE_LIBRARY"
        --experience-retrieval
        --experience-retrieval-top-k "$EXPERIENCE_RETRIEVAL_TOP_K"
        --experience-retrieval-decomposition
        --experience-retrieval-rewrite
        --experience-temperature "$EXPERIENCE_TEMPERATURE"
        --experience-top-p "$EXPERIENCE_TOP_P"
    )

    local phase_args=()
    if [[ "$phase" == "train" ]]; then
        phase_args+=(--experience-large-batch "$experience_large_batch")
    fi

    if [[ -n "$REASONING_TOP_K" ]]; then
        common_args+=(--reasoning-top-k "$REASONING_TOP_K")
    fi
    if [[ -n "$REASONING_PRESENCE_PENALTY" ]]; then
        common_args+=(--reasoning-presence-penalty "$REASONING_PRESENCE_PENALTY")
    fi
    if [[ -n "$REASONING_REPETITION_PENALTY" ]]; then
        common_args+=(--reasoning-repetition-penalty "$REASONING_REPETITION_PENALTY")
    fi
    if [[ -n "$EXPERIENCE_TOP_K" ]]; then
        common_args+=(--experience-top-k "$EXPERIENCE_TOP_K")
    fi
    if [[ -n "$EXPERIENCE_PRESENCE_PENALTY" ]]; then
        common_args+=(--experience-presence-penalty "$EXPERIENCE_PRESENCE_PENALTY")
    fi
    if [[ -n "$EXPERIENCE_REPETITION_PENALTY" ]]; then
        common_args+=(--experience-repetition-penalty "$EXPERIENCE_REPETITION_PENALTY")
    fi
    echo "Running ${phase} with ${entrypoint}"

    python3 -u "$entrypoint" \
        "${common_args[@]}" \
        "${phase_args[@]}" \
        "$@" \
        2>&1 | tee "$log_file"
}

case "$MODE" in
    train)
        run_phase "train" "eval/train.py" "${TRAIN_ONLY_ARGS[@]}"
        ;;
    inference|infer)
        run_phase "inference" "eval/infer.py"
        ;;
    all)
        run_phase "train" "eval/train.py" "${TRAIN_ONLY_ARGS[@]}"
        run_phase "inference" "eval/infer.py"
        ;;
    *)
        usage
        exit 1
        ;;
esac
