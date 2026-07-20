#!/bin/bash --login
# ============================================================================
# WMT26 QE baseline — Qwen3.6-35B-A3B on A100s
# Job array: one node per language pair (21 jobs total).
#
# GPU memory guidance (BF16 weights ~70 GB):
#   2× A100 80GB  — fits comfortably, throughput ≈ 1× H200
#   4× A100 80GB  — more KV cache headroom, throughput > H200; use --batch-size 32
#
# Edit the Configuration block below, then:
#   sbatch slurm/run_qe_local_qwen.sh
# To run a single pair (e.g. for testing):
#   sbatch --array=0 slurm/run_qe_local_qwen.sh
# ============================================================================

#SBATCH --time=72:00:00
#SBATCH --job-name=qe_qwen
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:a100:4
#SBATCH --mem=1T
#SBATCH --array=0-20
#SBATCH --qos=matrix
#SBATCH --output=slurm/logs/%x_%A_%a.out
#SBATCH --error=slurm/logs/%x_%A_%a.err
#SBATCH --exclude=dw-2-4

# ── Configuration (edit as needed) ─────────────────────────────────────────
MODEL="qwen36"           # gemma4 | qwen36
THINKING=false           # true | false
MAX_NEW_TOKENS=512       # Stage 1 token budget; recommend 8192+ when THINKING=true
BATCH_SIZE=32            # 16 for 2× A100; 32 for 4× A100; 2 when THINKING=true
# ───────────────────────────────────────────────────────────────────────────

# Language pairs — must match TARGET_PAIRS keys in qe_utils.py (21 pairs).
PAIRS=(
    cs-de cs-vi en-areg en-be en-cs en-de en-et en-hy en-id en-is
    en-ja en-kk en-ko en-lij en-lld en-ru en-th en-uk en-zhcn en-zhtw
    zhcn-ja
)

PAIR="${PAIRS[$SLURM_ARRAY_TASK_ID]}"
if [ -z "$PAIR" ]; then
    echo "ERROR: No pair for SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID"
    exit 1
fi

SCRIPT_DIR=/home/acshurtz/nobackup/archive/wmt_qe/WMT26-QE-baselines

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export HF_HUB_OFFLINE=1
export OPENSSL_CONF=/dev/null

module load cuda/12.8
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate qwen
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export LD_PRELOAD="${CONDA_PREFIX}/lib/libssl.so.3:${CONDA_PREFIX}/lib/libcrypto.so.3"

echo "=== Job ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}: pair=${PAIR} model=${MODEL} ==="
nvidia-smi
python -c "import torch; print('torch:', torch.__version__, '| CUDA:', torch.cuda.is_available(), '| GPUs:', torch.cuda.device_count())"

cd "$SCRIPT_DIR"

THINKING_FLAG=""
if [ "$THINKING" = "true" ]; then
    THINKING_FLAG="--thinking"
fi

mkdir -p slurm/logs

python run_qe_local.py \
    --model "$MODEL" \
    $THINKING_FLAG \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    --pair "$PAIR" \
    --resume \
    --batch-size "$BATCH_SIZE"
