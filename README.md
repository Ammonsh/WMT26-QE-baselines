# WMT26-QE-baselines

LLM-as-judge baseline for the WMT 2026 Automated MT Evaluation Shared Task.

---

## Quick Start — Gemini API

**Requirements:** Python 3.8+, one file: `run_qe.py` (self-contained, no other project files needed)

**1. Install the Gemini client:**
```bash
pip install -U google-genai
```

**2. Set your API key:**
```bash
export GEMINI_API_KEY="your_key_here"
```

**3. Test on one segment** (prints prompts and responses, no file written):
```bash
python run_qe.py --data-file mteval-test26.jsonl --test
```

**4. Full run — official segments** (all 21 language pairs, output written to `quality_estimation_outputs_gemini/`):
```bash
python run_qe.py --data-file mteval-test26.jsonl --segment-type official
```

**5. Resume an interrupted run:**
```bash
python run_qe.py --data-file mteval-test26.jsonl --segment-type official --resume
```

**Optional — single language pair:**
```bash
python run_qe.py --data-file mteval-test26.jsonl --segment-type official --pair en-de
```

> The model is set by `MODEL_ID` near the top of `run_qe.py` (default: `gemini-3.6-flash`).
> Thinking is enabled at `medium` level by default (`THINKING_LEVEL` at the top of the script).
> On daily quota exhaustion the partial result is saved automatically — re-run with `--resume` the next day.

---

Supports closed-weight models via the Gemini API (`run_qe.py`) and open-weight models on local GPUs via vLLM (`run_qe_vllm.py`). Both scripts share common data loading, prompts, and output formatting via `qe_utils.py`.

---

## How it works

### Prompting: two-stage GEMBA-ESA

Each hypothesis is evaluated with two sequential LLM calls:

**Stage 1 — Error annotation**
The model receives the source segment and MT hypothesis and outputs a structured list of errors classified by category and severity (major/minor). The prompt is domain-specific: news/fact-checking, speech, social media, software data, and educational content each have tailored instructions. The domain is inferred automatically from the `item_id` field of each segment.

**Stage 2 — Scoring**
The Stage 1 annotations are fed back into the model alongside the source and hypothesis. The model outputs a single numeric score on a continuous 0–100 scale (0 = no meaning preserved, 100 = perfect).

The system prompt for both stages is: *"Your task is to identify machine translation errors and assess the quality of the translation."*

### All systems evaluated

Every MT system hypothesis in the `hyps` field is evaluated independently. Results are stored together per segment in `task1_pred` and `task2_pred` dicts keyed by system name.

### Per-segment checkpointing

Output is appended to the JSONL file after each segment completes (all systems). If a job is canceled, re-running with `--resume` skips segments already written and picks up where processing left off.

---

## Data

All language pairs are in a single combined JSONL file (`mteval-test26.jsonl`). Each line is one segment:

```json
{
  "item_id": "official_###_eng_Latn_###_deu_Latn_###_social_###_116262294091035303_###_0",
  "src": "source text",
  "ref": {"text": "reference translation", "type": "postedit"},
  "hyps": {
    "Gemini 3.1 Pro": "hypothesis text",
    "Gemma 4 - 31B": "hypothesis text",
    ...
  }
}
```

The `item_id` is `_###_`-separated: `{seg_type}_###_{src_code}_###_{tgt_code}_###_{domain}_###_{doc_id}_###_{seg_idx}`. The first field is `"official"` or `"challenge"`. Official segments use FLORES-200 language codes; challenge segments use short two-letter codes. 21 language pairs are configured in `TARGET_PAIRS` in `qe_utils.py`.

---

## Output format

Each input segment produces one JSONL line with `task1_pred` (error spans per system) and `task2_pred` (quality scores per system):

```json
{
  "item_id": "official_###_eng_Latn_###_deu_Latn_###_speech_###_id_rtWATtjAFUM_29.58-58.15_###_0",
  "task1_pred": {
    "Gemini 3.1 Pro": {
      "errors": [
        {"start": 197, "end": 202, "severity": "major", "category": "accuracy/addition"}
      ],
      "omission": null,
      "instruction_fault": null
    },
    "Gemma 4 - 31B": {
      "errors": [
        {"start": 171, "end": 177, "severity": "minor", "category": "fluency/register"}
      ],
      "omission": "minor",
      "instruction_fault": null
    },
    ...
  },
  "task2_pred": {
    "Gemini 3.1 Pro": 60.0,
    "Gemma 4 - 31B": 75.0,
    ...
  }
}
```

`errors` contains character-level span indices (half-open `[start, end)`) derived by string-matching the quoted error spans from Stage 1 back into the hypothesis text. `omission` is set to `"major"` or `"minor"` when Stage 1 identifies an `accuracy/omission` error (omitted content has no span in the hypothesis). `instruction_fault` is always `null` (not detectable from the current prompt).

---

## Running the models

### Gemini (closed-weight API)

See the **Quick Start** section at the top of this README for setup and run instructions.

Output is written to a single file `quality_estimation_outputs_gemini/pred_{OUTPUT_NAME}.jsonl` (all pairs combined). The model ID and thinking level are set by `MODEL_ID` and `THINKING_LEVEL` at the top of `run_qe.py`. On daily quota exhaustion the partial segment is saved and the script exits cleanly — re-run with `--resume` the next day.

Rate limiting is disabled by default (`MIN_INTERVAL_SEC = 0`). If you are using a free-tier key (~10 req/min limit), set `MIN_INTERVAL_SEC = 6.5` at the top of the script.

Use `--segment-type official` or `--segment-type challenge` to restrict to one segment type; default is `all`. Use `--workers N` (default 4) to control API parallelism.

---

### Gemma-4 / Qwen3.6 (open-weight, local GPU via vLLM)

**Requirements:** `pip install vllm`

Models must be pre-downloaded to the HuggingFace cache before running on compute nodes (which have no internet access). Run once on the login node:
```bash
bash slurm/cache_models.sh
```

**Test on one segment of one system** (no file written):
```bash
python run_qe_vllm.py --model gemma4 --data-file mteval-test26.jsonl --test
python run_qe_vllm.py --model qwen36 --data-file mteval-test26.jsonl --test
```

**Single pair** (output written to `--output-dir`):
```bash
python run_qe_vllm.py --model gemma4 --data-file mteval-test26.jsonl \
    --segment-type all --pair cs-de --tensor-parallel-size 4 \
    --output-dir quality_estimation_outputs_gemma4_no_ref
```

**With thinking mode** (chain-of-thought):
```bash
python run_qe_vllm.py --model gemma4 --thinking --max-new-tokens 8192 \
    --data-file mteval-test26.jsonl --pair cs-de --tensor-parallel-size 4 \
    --output-dir quality_estimation_outputs_gemma4_thinking_no_ref
```

**With reference translation:**
```bash
python run_qe_vllm.py --model gemma4 --with-ref --data-file mteval-test26.jsonl \
    --pair cs-de --tensor-parallel-size 4 \
    --output-dir quality_estimation_outputs_gemma4_ref
```

**Resume an interrupted run:**
```bash
python run_qe_vllm.py --model gemma4 --data-file mteval-test26.jsonl \
    --pair cs-de --tensor-parallel-size 4 --resume \
    --output-dir quality_estimation_outputs_gemma4_no_ref
```

Output files are named `pred_{model}_{pair}.jsonl` (or `pred_{model_thinking}_{pair}.jsonl` with `--thinking`) inside the specified `--output-dir`. Use `--segment-type official`, `challenge`, or `all` (default: `all`).

**Token budgets:** Stage 1 (error annotation) uses `--max-new-tokens` (default: 512 without thinking, 8192 with). Stage 2 (scoring) always uses 64 tokens.

**GPU memory (bf16):**
- `gemma4` (~62 GB): 4× A100-80G or 2× H200
- `qwen36` (~70 GB): 4× A100-80G or 2× B200/H200

> **Note on en-hy:** English→Armenian segments are unusually long and can exhaust the default context window. Use `--max-model-len 32768 --chunk-size 50` for this pair (the SLURM scripts handle this automatically).

> **Note on deduplication:** Deduplication of hypotheses between Stage 1 and Stage 2 is not performed in this baseline. If your submission pipeline requires it, apply deduplication between the two stages before running Stage 2 scoring.

---

### OpenAI Batch API (GPT)

**Requirements:** `pip install -U openai`

```bash
export OPENAI_API_KEY="your_key_here"
```

Set `MODEL_ID` and `REASONING_EFFORT` near the top of `run_qe_openai_batch.py`, then:

```bash
# Full run (all pairs, all segments)
python run_qe_openai_batch.py --data-file mteval-test26.jsonl

# Single pair test
python run_qe_openai_batch.py --data-file mteval-test26.jsonl --pair en-de --max-segments 5

# Resume after interruption
python run_qe_openai_batch.py --data-file mteval-test26.jsonl --resume
```

The script submits Stage 1 requests as an OpenAI batch job, polls until complete, then submits Stage 2. Intermediate state (`batch_state`, `stage1_lookup`, `stage1_parsed`, `stage2_lookup`) is saved to `OUTPUT_DIR` so any interruption is safely resumable. Final output is `pred_{MODEL_ID}.jsonl` in the same directory.

---

### XCOMET (metric model)

**Requirements:** `pip install unbabel-comet`

Pre-download models before going offline:
```bash
python -c "from comet import download_model; \
    download_model('Unbabel/XCOMET-XL'); \
    download_model('Unbabel/XCOMET-XXL')"
```

```bash
# XL, no reference, single pair
python run_qe_xcomet.py --model xl --data-file mteval-test26.jsonl --pair cs-de

# XXL, with reference
python run_qe_xcomet.py --model xxl --with-ref --data-file mteval-test26.jsonl

# Resume
python run_qe_xcomet.py --model xl --data-file mteval-test26.jsonl --resume
```

Output is written to `quality_estimation_outputs_xcomet/pred_xcomet_{model}_{pair}{_ref}.jsonl`.

---

### SLURM job arrays

Three consolidated scripts cover all model families. Each script uses a 108-job array (27 language pairs × 4 variants) so all configurations submit in a single `sbatch` call:

**Gemma-4 (`slurm/run_gemma4.sh`)**

| Variant | IDs | Thinking | Reference | Output directory |
|---------|-----|----------|-----------|-----------------|
| 0 | 0–26 | no | no | `quality_estimation_outputs_gemma4_no_ref` |
| 1 | 27–53 | no | yes | `quality_estimation_outputs_gemma4_ref` |
| 2 | 54–80 | yes | no | `quality_estimation_outputs_gemma4_thinking_no_ref` |
| 3 | 81–107 | yes | yes | `quality_estimation_outputs_gemma4_thinking_ref` |

```bash
sbatch slurm/run_gemma4.sh                   # all 108 jobs
sbatch --array=0 slurm/run_gemma4.sh         # smoke-test: cs-de, no-thinking, no-ref
sbatch --array=0-53 slurm/run_gemma4.sh      # no-thinking variants only
sbatch --array=54-107 slurm/run_gemma4.sh    # thinking variants only
```

**Qwen3.6 (`slurm/run_qwen36.sh`)** — identical variant layout, same commands with `run_qwen36.sh`.

**XCOMET (`slurm/run_xcomet.sh`)**

| Variant | IDs | Model | Reference |
|---------|-----|-------|-----------|
| 0 | 0–26 | XL | no |
| 1 | 27–53 | XL | yes |
| 2 | 54–80 | XXL | no |
| 3 | 81–107 | XXL | yes |

```bash
sbatch slurm/run_xcomet.sh                   # all 108 jobs
sbatch --array=0-53 slurm/run_xcomet.sh      # XL only
```

**After Gemma/Qwen jobs complete**, merge each variant's per-pair files into one:
```bash
python merge_shards.py --model gemma4 \
    --output-dir quality_estimation_outputs_gemma4_no_ref
python merge_shards.py --model gemma4_thinking \
    --output-dir quality_estimation_outputs_gemma4_thinking_no_ref
# (repeat for ref and qwen36 variants)
```

`merge_shards.py` deduplicates by `item_id` (preferring scored rows over null-score rows from crashed jobs) and writes a single `pred_{model}.jsonl` in source order.

**Resume behaviour:** `--resume` is always active. Re-submitting a failed array only processes segments not yet written to disk.

---

## Configuration

Key settings in `qe_utils.py`:

| Variable | Default | Description |
|---|---|---|
| `HYP_SYSTEM` | `"Gemini 3.1 Pro"` | Unused in full runs (all systems evaluated); kept as a reference |
| `N_INSTANCES_PER_PAIR` | `None` | Cap segments per pair; `None` = all |
| `TARGET_PAIRS` | 21 pairs | Language pairs and their FLORES-200 codes |
| `DOMAIN_REQUIREMENTS` | 7 domains | Per-domain prompt text for Stage 1 and Stage 2 (news, factchecking, speech, social, software, edu, general) |

Key settings in `run_qe_vllm.py`:

| Variable | Default | Description |
|---|---|---|
| `DEFAULT_MAX_NEW_TOKENS` | `512` | Stage 1 token budget without thinking |
| `DEFAULT_MAX_NEW_TOKENS_THINKING` | `8192` | Stage 1 token budget with `--thinking` |
| `MAX_NEW_TOKENS_STAGE2` | `64` | Stage 2 token budget (hardcoded; just a number) |
| `DEFAULT_CHUNK_SIZE_NON_THINKING` | `400` | Segments per vLLM call without thinking |
| `DEFAULT_CHUNK_SIZE_THINKING` | `150` | Segments per vLLM call with thinking |

---
