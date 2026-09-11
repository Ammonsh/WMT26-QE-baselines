#!/bin/bash
# Run this on the LOGIN NODE (has internet access) to pre-download models
# into the HuggingFace cache before submitting compute jobs.
#
# Usage:
#   bash slurm/cache_models.sh
#
# Models downloaded:
#   google/gemma-4-31B-it      (~62 GB bf16)
#   Qwen/Qwen3.6-35B-A3B       (~70 GB bf16)

set -euo pipefail

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate qwen
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
# Use system's FIPS-certified libssl (RHEL9 nodes have kernel FIPS mode enabled).
export LD_PRELOAD="/lib64/libssl.so.3:/lib64/libcrypto.so.3"

echo "Python: $(which python) ($(python --version))"
echo "HF cache: $(python -c 'from huggingface_hub import constants; print(constants.HF_HUB_CACHE)')"
echo ""

echo "=== Caching google/gemma-4-31B-it ==="
hf download google/gemma-4-31B-it

echo ""
echo "=== Caching Qwen/Qwen3.6-35B-A3B ==="
hf download Qwen/Qwen3.6-35B-A3B

echo ""
echo "=== Done. Cache contents: ==="
hf cache list 2>/dev/null || echo "(cache listing not available)"
