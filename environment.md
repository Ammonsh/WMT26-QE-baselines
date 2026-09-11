# Environment Setup

## Conda Environment: `qwen_new`

Python 3.11, PyTorch 2.11.0+cu128.

The cluster provides CUDA 12.8 via `module load cuda/12.8` (resolves to `/apps/cudatoolkit/12.8.1`).
PyTorch must be built against CUDA 12.8 to match the available `nvcc`. PyTorch 2.13+cu130 (the
default from PyPI at time of writing) does **not** work because there is no CUDA 13 `nvcc` on the
cluster (even though `/usr/local/cuda-13.0` exists, it has no `bin/` directory).

### Base install

```bash
conda create -n qwen_new python=3.11 -y
conda activate qwen_new
pip install numpy
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

### causal-conv1d and flash-linear-attention (Qwen3.6 efficiency)

Must use `--no-build-isolation` so the build uses the installed torch (cu128) rather than
downloading the latest torch (cu130) into an isolated build environment.

```bash
module load cuda/12.8
pip install causal-conv1d --no-build-isolation
pip install flash-linear-attention --no-build-isolation
```

These compile from source (no pre-built wheel available for this torch/CUDA combo). Expect
10-30 minutes each.

### flash-attn (Gemma4 efficiency)

Requires `psutil` first. Also needs `--no-build-isolation` and `MAX_JOBS` to prevent OOM on the
login node (exit code 255 from nvcc = too many parallel compilations exhausting memory).
`TORCH_CUDA_ARCH_LIST` is set explicitly to cover A100s (sm_80), H100s (sm_90), B100/B200s
(sm_100), and sm_120 — all confirmed supported by CUDA 12.8 on this cluster.

```bash
pip install psutil
export TORCH_CUDA_ARCH_LIST="8.0;9.0;10.0;12.0"
MAX_JOBS=4 pip install flash-attn --no-build-isolation
```

Expect 30-60 minutes to compile.

## Key Pitfalls

- **CUDA version mismatch**: pip's isolated build environment downloads the latest torch (cu130),
  which mismatches nvcc 12.8. Always use `--no-build-isolation`.
- **nvcc not found**: must `module load cuda/12.8` before any compilation. The module sets
  `CUDA_HOME=/apps/cudatoolkit/12.8.1` and adds `nvcc` to `PATH`.
- **OOM during flash-attn build**: use `MAX_JOBS=4` (or lower) to limit parallel nvcc processes.
- **numpy missing**: install numpy before torch or any CUDA extension.

## Slurm Scripts

Both `slurm/run_reasoning_qe_local_sharded_qwen36.sh` and
`slurm/run_reasoning_qe_local_sharded_gemma4.sh` are already configured correctly:
- `module load cuda/12.8`
- `conda activate qwen_new`
