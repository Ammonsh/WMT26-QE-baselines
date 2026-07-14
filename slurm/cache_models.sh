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

# Use conda's bundled OpenSSL instead of the system FIPS-enforced one.
# On RHEL9, the system OpenSSL is in FIPS mode and conda's libssl doesn't ship
# the FIPS integrity module, causing "FATAL FIPS SELFTEST FAILURE" on import.
export OPENSSL_CONF=/dev/null

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate qwen
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"

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
