#!/bin/bash --login
# ============================================================================
# WMT26 QE baseline — Gemma-4-31B-it on A100s, sharded for speed
# Job array: 23 pairs × 4 shards = 92 jobs total.
#
# Each job processes ~25 % of one language pair's segments in parallel,
# targeting ~12 h wall time (vs 36-48 h for a full single-job run).
#
# GPU memory guidance (BF16 weights ~62 GB):
#   2× A100 80GB  — ~98 GB free for KV cache; use --batch-size 32
#
# Edit the Configuration block below, then:
#   sbatch slurm/run_qe_local_a100_sharded_gemma4.sh
# To smoke-test a single shard (pair=cs-de, shard 0):
#   sbatch --array=0 slurm/run_qe_local_a100_sharded_gemma4.sh
#
# After all jobs complete, merge shard files per pair:
#   for pair in cs-de cs-vi en-areg ...; do
#     cat quality_estimation_outputs_local/pred_gemma4_${pair}_s*of4.jsonl \
#       > quality_estimation_outputs_local/pred_gemma4_${pair}.jsonl
#   done
# ============================================================================

#SBATCH --time=24:00:00
#SBATCH --job-name=qe_gemma_shard
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:b200:1
#SBATCH --mem=512G
#SBATCH --array=0-91
#SBATCH --qos=cs
#SBATCH --output=slurm/logs/%x_%A_%a.out
#SBATCH --error=slurm/logs/%x_%A_%a.err
#SBATCH --exclude=dw-2-4

# ── Configuration (edit as needed) ─────────────────────────────────────────
MODEL="gemma4"           # gemma4 | qwen36
THINKING=false           # true | false
MAX_NEW_TOKENS=512       # Stage 1 token budget; recommend 8192+ when THINKING=true
BATCH_SIZE=32            # 32 for 2× A100; 2 when THINKING=true
NUM_SHARDS=4             # Number of shards per pair (array size must be pairs × shards)
# ───────────────────────────────────────────────────────────────────────────

# Language pairs — 23 pairs (21 original + cs-uk and en-se).
# Old pairs run with --resume and will only process new hypothesis systems.
# New pairs (cs-uk, en-se) run fresh.
PAIRS=(
    cs-de cs-uk cs-vi en-areg en-be en-cs en-de en-et en-hy en-id en-is
    en-ja en-kk en-ko en-lij en-lld en-ru en-se en-th en-uk en-zhcn en-zhtw
    zhcn-ja
)

PAIR_IDX=$((SLURM_ARRAY_TASK_ID / NUM_SHARDS))
SHARD=$((SLURM_ARRAY_TASK_ID % NUM_SHARDS))
PAIR="${PAIRS[$PAIR_IDX]}"

if [ -z "$PAIR" ]; then
    echo "ERROR: No pair for SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID (PAIR_IDX=$PAIR_IDX)"
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

echo "=== Job ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}: pair=${PAIR} shard=${SHARD}/${NUM_SHARDS} model=${MODEL} ==="
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
    --batch-size "$BATCH_SIZE" \
    --shard "$SHARD" \
    --num-shards "$NUM_SHARDS"
