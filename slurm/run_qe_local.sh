#!/bin/bash --login
# ============================================================================
# WMT26 QE baseline — local open-weight models (Gemma-4 / Qwen3.6)
#
# GPU memory guidance:
#   gemma4  (google/gemma-4-31B-it, ~62 GB bf16):
#     1× A100-80G or H100-80G or B200  — or 2× A100-40G
#   qwen36  (Qwen/Qwen3.6-35B-A3B, ~70 GB bf16):
#     2× A100-80G or H100-80G  — or 1× B200 (if 140 GB+)
#
# Edit the Configuration block below, then:
#   sbatch slurm/run_qe_local.sh
# ============================================================================

# ── Configuration (edit as needed) ─────────────────────────────────────────
GPU_TYPE="h100"          # a100 | h100 | b200
N_GPUS=2                 # see GPU memory guidance above
MODEL="gemma4"           # gemma4 | qwen36
THINKING=false           # true | false
MAX_NEW_TOKENS=512       # recommend 8192+ when THINKING=true
# ───────────────────────────────────────────────────────────────────────────

#SBATCH --job-name=qe_local
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:${GPU_TYPE}:${N_GPUS}
#SBATCH --mem=256G
#SBATCH --time=24:00:00
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/logs/%x_%j.err
#SBATCH --qos=cs

SCRIPT_DIR=/home/acshurtz/nobackup/archive/wmt_qe/WMT26-QE-baselines

export OMP_NUM_THREADS=$SLURM_CPUS_ON_NODE
export HF_HUB_OFFLINE=1
export OPENSSL_CONF=/dev/null

module load cuda/12.8
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate qwen
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export LD_PRELOAD="${CONDA_PREFIX}/lib/libssl.so.3:${CONDA_PREFIX}/lib/libcrypto.so.3"

nvidia-smi
python -c "import torch; print('torch:', torch.__version__, '| CUDA:', torch.cuda.is_available(), '| GPUs:', torch.cuda.device_count())"

cd "$SCRIPT_DIR"

# Build optional flags
THINKING_FLAG=""
if [ "$THINKING" = "true" ]; then
    THINKING_FLAG="--thinking"
fi

mkdir -p slurm/logs

python run_qe_local.py \
    --model "$MODEL" \
    $THINKING_FLAG \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    --resume
