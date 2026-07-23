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

### Gemma-4 / Qwen3.6 (open-weight, local GPU)

**Requirements:** `pip install transformers torch accelerate`

Models must be pre-downloaded to the HuggingFace cache before running on compute nodes (which have no internet access). Run once on the login node:
```bash
bash slurm/cache_models.sh
```

**Test on one segment of one system** (no file written):
```bash
python run_qe_local.py --model gemma4 --data-file mteval-test26.jsonl --test
python run_qe_local.py --model qwen36 --data-file mteval-test26.jsonl --test
```

**Single pair (recommended for testing timing):**
```bash
python run_qe_local.py --model gemma4 --data-file mteval-test26.jsonl --segment-type official --pair cs-de
```

**Resume an interrupted run:**
```bash
python run_qe_local.py --model gemma4 --data-file mteval-test26.jsonl --segment-type official --pair cs-de --resume
```

**With thinking mode** (chain-of-thought, uses more tokens):
```bash
python run_qe_local.py --model gemma4 --data-file mteval-test26.jsonl --thinking
python run_qe_local.py --model qwen36 --data-file mteval-test26.jsonl --thinking --max-new-tokens 8192
```

Output is written to `quality_estimation_outputs_local/pred_{model}_{pair}.jsonl`. Use `--segment-type official` or `--segment-type challenge` to restrict to one segment type (default: `all`).

**Token budgets:** Stage 1 (error annotation) uses `--max-new-tokens` (default 512). Stage 2 (scoring) always uses a fixed budget of 64 tokens since it only outputs a number.

GPU memory (bf16):
- `gemma4` (~62 GB): 1× H200 / A100-80G / H100-80G
- `qwen36` (~70 GB): 1× H200 (141 GB SXM), or 2× A100-80G / H100-80G

---

### SLURM job array (one H200 per language pair)

Edit the configuration block at the top of `slurm/run_qe_local.sh` (model, thinking flag, token budget), then submit:

```bash
sbatch slurm/run_qe_local.sh
```

This launches 23 jobs in parallel (`--array=0-22`), one per language pair, each on a single H200. Logs are written to `slurm/logs/qe_local_{jobid}_{arrayid}.out/err`.

To test a single pair before committing to the full array:
```bash
sbatch --array=0 slurm/run_qe_local.sh   # runs cs-de (index 0)
```

All jobs use `--resume`, so re-submitting the array after a failure or cancellation only processes segments not yet written.

---

### Sharded SLURM job array (A100s, faster)

`slurm/run_qe_local_sharded_qwen36.sh` (and the equivalent gemma4 script) splits each language pair into 4 shards processed in parallel, targeting ~12 h wall time instead of 36–48 h. With 21 pairs × 4 shards = 84 jobs:

```bash
sbatch slurm/run_qe_local_sharded_qwen36.sh
```

Shard output files are named `pred_qwen36_{pair}_s0of4.jsonl` through `pred_qwen36_{pair}_s3of4.jsonl`. After all jobs complete, merge all pairs into a single output file in source order:

```bash
python merge_shards.py --model qwen36 --segment-type official          # all 21 pairs
python merge_shards.py --model qwen36 --segment-type official --dry-run  # check only
python merge_shards.py --model qwen36 --pair en-de                     # single pair
```

`merge_shards.py` collects rows from all shard files across all pairs, deduplicates by `item_id` (preferring scored rows over null-score rows from crashed jobs), and writes a single merged file `pred_{model}.jsonl` in the same order as the source data file.

**Resume behaviour:** `--resume` is always active. When re-running after a partial run or when new hypothesis systems are added to the data, each shard only processes segments/systems not yet scored — existing results are merged in before writing.

---

### Diagnosing job failures

**Check which array tasks failed or are still running:**
```bash
bash check_logs.sh                          # gemma4 shards (default)
bash check_logs.sh slurm/logs qe_qwen_shard # different pattern
```

Outputs succeeded/failed/unknown counts, failure reasons, and a ready-to-paste `sbatch --array=...` resubmit command for failed tasks.

**Analyze warning counts by language pair:**
```bash
python analyze_warnings.py                  # all pairs, sorted by total warnings
python analyze_warnings.py --top 10         # worst 10 pairs
python analyze_warnings.py --pattern qe_qwen_shard
```

Reports Stage 1 warnings (span not found in hypothesis) and Stage 2 warnings (failed to parse score) per pair, deduplicated to avoid counting crash-resume reruns twice.

---

## Configuration

Key settings in `qe_utils.py`:

| Variable | Default | Description |
|---|---|---|
| `HYP_SYSTEM` | `"Gemini 3.1 Pro"` | Unused in full runs (all systems evaluated); kept as a reference |
| `N_INSTANCES_PER_PAIR` | `None` | Cap segments per pair; `None` = all |
| `TARGET_PAIRS` | 21 pairs | Language pairs and their FLORES-200 codes |
| `DOMAIN_REQUIREMENTS` | 7 domains | Per-domain prompt text for Stage 1 and Stage 2 (news, factchecking, speech, social, software, edu, general) |

Key settings in `run_qe_local.py`:

| Variable | Default | Description |
|---|---|---|
| `DEFAULT_MAX_NEW_TOKENS` | `512` | Stage 1 token budget without thinking |
| `DEFAULT_MAX_NEW_TOKENS_THINKING` | `8192` | Stage 1 token budget with `--thinking` |
| `MAX_NEW_TOKENS_STAGE2` | `64` | Stage 2 token budget (hardcoded; just a number) |

---
