#!/bin/bash --login
# ============================================================================
# WMT26 QE baseline — Gemma-4-31B-it, all variants, all language pairs.
#
# Job array layout: 27 pairs × 4 variants = 108 jobs (array 0-107)
#
#   Variant 0  (IDs   0-26): no thinking, no reference
#   Variant 1  (IDs  27-53): no thinking, with reference
#   Variant 2  (IDs  54-80): with thinking, no reference
#   Variant 3  (IDs  81-107): with thinking, with reference
#
# Pairs 0-20: 21 official WMT26 language pairs.
# Pairs 21-26: challenge-only pairs (en-ja, en-ko, en-el, en-hi, ja-zh, zh-en).
#
# Each job writes to its own output directory; the variant→directory mapping is:
#   Variant 0: quality_estimation_outputs_gemma4_no_ref
#   Variant 1: quality_estimation_outputs_gemma4_ref
#   Variant 2: quality_estimation_outputs_gemma4_thinking_no_ref
#   Variant 3: quality_estimation_outputs_gemma4_thinking_ref
#
# GPU memory (BF16 weights ~62 GB):
#   4× A100 80GB — ~258 GB free for KV cache; use --tensor-parallel-size 4
#
# Note: en-hy (English→Armenian) has long segments that approach the context
# window limit. This script automatically uses --max-model-len 32768 and
# --chunk-size 50 for en-hy jobs. If you hit OOM on en-hy, reduce
# --gpu-memory-utilization or use a node with more GPU memory.
#
# Submit all 108 jobs:
#   sbatch slurm/run_gemma4.sh
# Smoke-test a single job (cs-de, no-thinking, no-ref):
#   sbatch --array=0 slurm/run_gemma4.sh
# Run only no-thinking variants (IDs 0-53):
#   sbatch --array=0-53 slurm/run_gemma4.sh
# Run only thinking variants (IDs 54-107):
#   sbatch --array=54-107 slurm/run_gemma4.sh
# Run only no-ref variants (IDs 0-26 and 54-80):
#   sbatch --array=0-26,54-80 slurm/run_gemma4.sh
#
# After all jobs complete, merge each variant into one file (source order):
#   python merge_shards.py --model gemma4 \
#       --output-dir quality_estimation_outputs_gemma4_no_ref
#   python merge_shards.py --model gemma4 \
#       --output-dir quality_estimation_outputs_gemma4_ref
#   python merge_shards.py --model gemma4_thinking \
#       --output-dir quality_estimation_outputs_gemma4_thinking_no_ref
#   python merge_shards.py --model gemma4_thinking \
#       --output-dir quality_estimation_outputs_gemma4_thinking_ref
# ============================================================================

#SBATCH --time=48:00:00
#SBATCH --job-name=qe_gemma4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4
#SBATCH --mem=512G
#SBATCH --array=0-107
#SBATCH --qos=matrix
#SBATCH --output=slurm/logs/%x_%A_%a.out
#SBATCH --error=slurm/logs/%x_%A_%a.err
#SBATCH --exclude=dw-2-4

# ── Language pairs ────────────────────────────────────────────────────────────
# 21 official pairs (indices 0-20) + 6 challenge-only pairs (indices 21-26).
PAIRS=(
    cs-de cs-uk cs-vi en-areg en-be en-cs en-de en-et en-hy en-id en-is
    en-kk en-lij en-lld en-ru en-se en-th en-uk en-zhcn en-zhtw zhcn-ja
    en-ja en-ko en-el en-hi ja-zh zh-en
)
N_PAIRS=${#PAIRS[@]}   # 27

# ── Dispatch: variant = task_id // N_PAIRS, pair = task_id % N_PAIRS ─────────
VARIANT=$((SLURM_ARRAY_TASK_ID / N_PAIRS))
PAIR_IDX=$((SLURM_ARRAY_TASK_ID % N_PAIRS))
PAIR="${PAIRS[$PAIR_IDX]}"

case "$VARIANT" in
    0) THINKING=false; WITH_REF=false; OUTPUT_DIR="quality_estimation_outputs_gemma4_no_ref"          ;;
    1) THINKING=false; WITH_REF=true;  OUTPUT_DIR="quality_estimation_outputs_gemma4_ref"              ;;
    2) THINKING=true;  WITH_REF=false; OUTPUT_DIR="quality_estimation_outputs_gemma4_thinking_no_ref"  ;;
    3) THINKING=true;  WITH_REF=true;  OUTPUT_DIR="quality_estimation_outputs_gemma4_thinking_ref"     ;;
    *)
        echo "ERROR: Unexpected variant $VARIANT for SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID"
        exit 1
        ;;
esac

if [ -z "$PAIR" ]; then
    echo "ERROR: No pair for SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID (PAIR_IDX=$PAIR_IDX)"
    exit 1
fi

# ── Token budgets and chunk sizes ─────────────────────────────────────────────
if [ "$THINKING" = "true" ]; then
    MAX_NEW_TOKENS=8192
    CHUNK_SIZE=150
else
    MAX_NEW_TOKENS=512
    CHUNK_SIZE=400
fi

# ── en-hy override: longer segments need a bigger context window ──────────────
MAX_MODEL_LEN=16384
if [ "$PAIR" = "en-hy" ]; then
    MAX_MODEL_LEN=32768
    CHUNK_SIZE=50
fi

# ── Build optional flags ──────────────────────────────────────────────────────
THINKING_FLAG=""
[ "$THINKING" = "true" ] && THINKING_FLAG="--thinking"

REF_FLAG=""
[ "$WITH_REF" = "true" ] && REF_FLAG="--with-ref"

# ── Environment setup ─────────────────────────────────────────────────────────
SCRIPT_DIR="$( cd "$(dirname "$0")/.." && pwd )"

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export HF_HUB_OFFLINE=1
export VLLM_NO_USAGE_STATS=1
export DO_NOT_TRACK=1
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export FLASHINFER_DISABLE_VERSION_CHECK=1

module load cuda/12.8
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate qwen_new
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export LD_PRELOAD="/lib64/libssl.so.3:/lib64/libcrypto.so.3"

echo "=== Job ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}: model=gemma4 pair=${PAIR} variant=${VARIANT} thinking=${THINKING} ref=${WITH_REF} ==="
nvidia-smi
python -c "import torch; print('torch:', torch.__version__, '| CUDA:', torch.cuda.is_available(), '| GPUs:', torch.cuda.device_count())"

cd "$SCRIPT_DIR"
mkdir -p slurm/logs

python run_qe_vllm.py \
    --model gemma4 \
    $THINKING_FLAG \
    $REF_FLAG \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    --max-model-len "$MAX_MODEL_LEN" \
    --chunk-size "$CHUNK_SIZE" \
    --data-file mteval-test26.jsonl \
    --segment-type all \
    --pair "$PAIR" \
    --resume \
    --tensor-parallel-size 4 \
    --output-dir "$OUTPUT_DIR"
