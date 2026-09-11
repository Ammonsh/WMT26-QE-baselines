#!/bin/bash --login
# ============================================================================
# WMT26 QE baseline — XCOMET-XL and XCOMET-XXL, with and without references.
#
# Job array layout: 27 pairs × 4 variants = 108 jobs (array 0-107)
#
#   Variant 0  (IDs   0-26): XCOMET-XL,  no reference
#   Variant 1  (IDs  27-53): XCOMET-XL,  with reference
#   Variant 2  (IDs  54-80): XCOMET-XXL, no reference
#   Variant 3  (IDs  81-107): XCOMET-XXL, with reference
#
# Pairs 0-22: original 23 official WMT26 pairs (IDs unchanged).
# Pairs 23-26: challenge-only pairs added to cover previously missing segments:
#   23=en-el  24=en-hi  25=ja-zh  26=zh-en
#
# GPU memory guidance (fp32 weights):
#   XCOMET-XL  (~3.5B params, ~14 GB fp32 / ~7 GB fp16)  — 1 A100 80GB easily
#   XCOMET-XXL (~10.7B params, ~43 GB fp32 / ~21 GB fp16) — 1 A100 80GB fine
#
# Pre-download models before going offline:
#   python -c "from comet import download_model; \
#       download_model('Unbabel/XCOMET-XL'); \
#       download_model('Unbabel/XCOMET-XXL')"
#
# Submit all 108 jobs:
#   sbatch slurm/run_xcomet.sh
# Run only the 4 new challenge-only pairs (all 4 variants):
#   sbatch --array=23-26,50-53,77-80,104-107 slurm/run_xcomet.sh
# Run only XL jobs (no-ref + ref, IDs 0-53):
#   sbatch --array=0-53 slurm/run_xcomet.sh
# Run only XXL jobs (IDs 54-107):
#   sbatch --array=54-107 slurm/run_xcomet.sh
# Run only reference-aware jobs (variants 1 and 3):
#   sbatch --array=27-53,81-107 slurm/run_xcomet.sh
# Smoke-test XL no-ref on cs-de (ID 0):
#   sbatch --array=0 slurm/run_xcomet.sh
# ============================================================================

#SBATCH --time=8:00:00
#SBATCH --job-name=qe_xcomet
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --array=0-107
#SBATCH --qos=matrix
#SBATCH --output=slurm/logs/%x_%A_%a.out
#SBATCH --error=slurm/logs/%x_%A_%a.err
#SBATCH --exclude=dw-2-4

# ── Configuration (edit as needed) ─────────────────────────────────────────
BATCH_SIZE_XL=128     # batch size for XCOMET-XL (reduce if OOM)
BATCH_SIZE_XXL=16     # batch size for XCOMET-XXL (reduce if OOM)
GPUS=1                # number of GPUs per job (matches --gres=gpu:N above)
# ───────────────────────────────────────────────────────────────────────────

# ── Language pairs ───────────────────────────────────────────────────────────
# 23 original official pairs (indices 0-22) + 4 challenge-only pairs (23-26).
PAIRS=(
    cs-de cs-uk cs-vi en-areg en-be en-cs en-de en-et en-hy en-id en-is
    en-ja en-kk en-ko en-lij en-lld en-ru en-se en-th en-uk en-zhcn en-zhtw
    zhcn-ja
    en-el en-hi ja-zh zh-en
)
N_PAIRS=${#PAIRS[@]}   # 27

# ── Dispatch: variant = task_id // N_PAIRS, pair = task_id % N_PAIRS ────────
VARIANT=$((SLURM_ARRAY_TASK_ID / N_PAIRS))
PAIR_IDX=$((SLURM_ARRAY_TASK_ID % N_PAIRS))
PAIR="${PAIRS[$PAIR_IDX]}"

case "$VARIANT" in
    0) MODEL="xl";  WITH_REF=false ;;
    1) MODEL="xl";  WITH_REF=true  ;;
    2) MODEL="xxl"; WITH_REF=false ;;
    3) MODEL="xxl"; WITH_REF=true  ;;
    *)
        echo "ERROR: Unexpected variant $VARIANT for SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID"
        exit 1
        ;;
esac

if [ -z "$PAIR" ]; then
    echo "ERROR: No pair for SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID (PAIR_IDX=$PAIR_IDX)"
    exit 1
fi

BATCH_SIZE=$BATCH_SIZE_XL
if [ "$MODEL" = "xxl" ]; then
    BATCH_SIZE=$BATCH_SIZE_XXL
fi

REF_FLAG=""
REF_LABEL="no-ref"
if [ "$WITH_REF" = "true" ]; then
    REF_FLAG="--with-ref"
    REF_LABEL="ref"
fi

SCRIPT_DIR="$( cd "$(dirname "$0")/.." && pwd )"

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export HF_HUB_OFFLINE=1

module load cuda/12.8
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate qwen_new
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export LD_PRELOAD="/lib64/libssl.so.3:/lib64/libcrypto.so.3"

echo "=== Job ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}: model=xcomet-${MODEL} pair=${PAIR} ${REF_LABEL} ==="
nvidia-smi
python -c "import torch; print('torch:', torch.__version__, '| CUDA:', torch.cuda.is_available(), '| GPUs:', torch.cuda.device_count())"

cd "$SCRIPT_DIR"
mkdir -p slurm/logs

python run_qe_xcomet.py \
    --model "$MODEL" \
    $REF_FLAG \
    --data-file mteval-test26.jsonl \
    --segment-type all \
    --pair "$PAIR" \
    --batch-size "$BATCH_SIZE" \
    --gpus "$GPUS" \
    --resume
