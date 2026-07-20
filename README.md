# WMT26-QE-baselines

LLM-as-judge baseline for the WMT 2026 Automated MT Evaluation Shared Task.

---

## Quick Start — Gemini API

**Requirements:** Python 3.8+, one file: `run_qe.py` (self-contained, no other project files needed)

**1. Install the Gemini client:**
```bash
pip install -U google-genai
```

**2. Extract the data tarball into the same directory as `run_qe.py`:**
```bash
tar -xzf test-set-files-v1.tar.gz
# Result: en-de.json, en-cs.json, ... sitting next to run_qe.py -- should be 23 total files
```

**3. Set your API key:**
```bash
export GEMINI_API_KEY="your_key_here"
```

**4. Test on one segment** (prints prompts and responses, no file written):
```bash
python run_qe.py --test
```

**5. Full run** (all 23 language pairs, outputs written to `quality_estimation_outputs_gemini/`):
```bash
python run_qe.py
```

**6. Resume an interrupted run:**
```bash
python run_qe.py --resume
```

**Optional — single language pair:**
```bash
python run_qe.py --pair en-de
```

**Optional — custom data directory** (if data files are not in the same directory as the script):
```bash
python run_qe.py --data-dir /path/to/data/
```

> The model is set by `MODEL_ID` near the top of `run_qe.py` (default: `gemini-3-flash-preview`).
> Thinking is enabled at `medium` level by default (`THINKING_LEVEL` at the top of the script).
> On daily quota exhaustion the partial result is saved automatically — re-run with `--resume` the next day.

---

Supports closed-weight models via the Gemini API (`run_qe.py`) and open-weight models on local GPUs (`run_qe_local.py`). Both scripts share common data loading, prompts, and output formatting via `qe_utils.py`.

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

Every MT system hypothesis in the `hyps` field is evaluated independently. Results are stored together per segment in a `qe_results` dict keyed by system name.

### Per-segment checkpointing

Output is appended to the JSONL file after each segment completes (all systems). If a job is canceled, re-running with `--resume` skips segments already written and picks up where processing left off.

---

## Data

Input files are one JSONL file per language pair (e.g. `en-de.json`), expected in the same directory as `run_qe.py` by default (override with `--data-dir`). Each line is one segment:

```json
{
  "item_id": "eng_Latn_###_deu_Latn_###_social_###_116262294091035303_###_0",
  "src": "source text",
  "ref": {"text": "reference translation", "type": "postedit"},
  "hyps": {
    "Gemini 3.1 Pro": "hypothesis text",
    "Gemma 4 - 31B": "hypothesis text",
    ...
  }
}
```

The `item_id` encodes source language, target language, domain, document ID, and segment index as `_###_`-separated fields. 21 of 23 language pairs are currently configured in `TARGET_PAIRS` in `qe_utils.py`.

---

## Output format

Each input segment is written to the output JSONL with a `qe_results` field added:

```json
{
  "item_id": "...",
  "src": "...",
  "ref": {...},
  "hyps": {...},
  "qe_results": {
    "Gemini 3.1 Pro": {
      "stage1_annotations": "Major:\naccuracy/mistranslation - \"wrong word\"\nMinor:\nno-error",
      "predicted_errors": [{"start_i": 5, "end_i": 14, "severity": "major"}],
      "score": 72.0
    },
    "Gemma 4 - 31B": {
      "stage1_annotations": "...",
      "predicted_errors": [...],
      "score": 85.0
    },
    ...
  }
}
```

`predicted_errors` contains character-level span indices derived by string-matching the quoted error spans from Stage 1 back into the hypothesis text.

---

## Running the models

### Gemini (closed-weight API)

See the **Quick Start** section at the top of this README for setup and run instructions.

Output is written to `quality_estimation_outputs_gemini/pred_{MODEL_ID}_{pair}.jsonl`. The model ID and thinking level are set by `MODEL_ID` and `THINKING_LEVEL` at the top of `run_qe.py`. On daily quota exhaustion the partial segment is saved and the script exits cleanly — re-run with `--resume` the next day.

Rate limiting is disabled by default (`MIN_INTERVAL_SEC = 0`). If you are using a free-tier key (~10 req/min limit), set `MIN_INTERVAL_SEC = 6.5` at the top of the script.

---

### Gemma-4 / Qwen3.6 (open-weight, local GPU)

**Requirements:** `pip install transformers torch accelerate`

Models must be pre-downloaded to the HuggingFace cache before running on compute nodes (which have no internet access). Run once on the login node:
```bash
bash slurm/cache_models.sh
```

**Test on one segment of one system** (no file written):
```bash
python run_qe_local.py --model gemma4 --test
python run_qe_local.py --model qwen36 --test
```

**Single pair (recommended for testing timing):**
```bash
python run_qe_local.py --model gemma4 --pair cs-de
```

**Resume an interrupted run:**
```bash
python run_qe_local.py --model gemma4 --pair cs-de --resume
```

**With thinking mode** (chain-of-thought, uses more tokens):
```bash
python run_qe_local.py --model gemma4 --thinking
python run_qe_local.py --model qwen36 --thinking --max-new-tokens 8192
```

Output is written to `quality_estimation_outputs_local/pred_{model}_{pair}.jsonl`.

**Token budgets:** Stage 1 (error annotation) uses `--max-new-tokens` (default 512). Stage 2 (scoring) always uses a fixed budget of 32 tokens since it only outputs a number.

GPU memory (bf16):
- `gemma4` (~62 GB): 1× H200 / A100-80G / H100-80G
- `qwen36` (~70 GB): 1× H200 (141 GB SXM), or 2× A100-80G / H100-80G

---

### SLURM job array (one H200 per language pair)

Edit the configuration block at the top of `slurm/run_qe_local.sh` (model, thinking flag, token budget), then submit:

```bash
sbatch slurm/run_qe_local.sh
```

This launches 21 jobs in parallel (`--array=0-20`), one per language pair, each on a single H200. Logs are written to `slurm/logs/qe_local_{jobid}_{arrayid}.out/err`.

To test a single pair before committing to the full array:
```bash
sbatch --array=0 slurm/run_qe_local.sh   # runs cs-de (index 0)
```

All jobs use `--resume`, so re-submitting the array after a failure or cancellation only processes segments not yet written.

---

## Configuration

Key settings in `qe_utils.py`:

| Variable | Default | Description |
|---|---|---|
| `HYP_SYSTEM` | `"Gemini 3.1 Pro"` | Unused in full runs (all systems evaluated); kept as a reference |
| `N_INSTANCES_PER_PAIR` | `None` | Cap segments per pair; `None` = all |
| `TARGET_PAIRS` | 21 pairs | Language pairs and their FLORES-200 codes |
| `DOMAIN_REQUIREMENTS` | 6 domains | Per-domain prompt text for Stage 1 and Stage 2 |

Key settings in `run_qe_local.py`:

| Variable | Default | Description |
|---|---|---|
| `DEFAULT_MAX_NEW_TOKENS` | `512` | Stage 1 token budget without thinking |
| `DEFAULT_MAX_NEW_TOKENS_THINKING` | `8192` | Stage 1 token budget with `--thinking` |
| `MAX_NEW_TOKENS_STAGE2` | `32` | Stage 2 token budget (hardcoded; just a number) |

---

## TODOs

- [ ] **Add remaining 2 language pairs** to `TARGET_PAIRS` once data is released (21 of 23 currently configured).