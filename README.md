# WMT26-QE-baselines

Plan Overview:

Create a codebase for running LLM-as-a-judge baselines for the WMT 2026 Automated MT Evaluation Shared Task. 
This codebase will be used for runnning both closed-weight (Gemini) and open-weight (Gemma 4, Qwen 3.6) models.

Note: Inputs and outputs will be slightly modified upon data release to match the specs of this year's shared task.

Inputs: jsonl with the following fields per line -- `src_text`, `tgt_text`, `doc_id`
(While developing, use just the `Claude-4` field under `tgt_text` as the single hypothesis translation to be evaluated (`wmt25-genmt-humeval.jsonl`))

Outputs: input jsonl with added field per line -- `predicted_errors` with system prepended to file name. (`wmt25-genmt-humeval.jsonl` becomes `pred_gemini_wmt25-genmt-humeval.jsonl`)

```
"predicted_errors": [
        {
            "start_i": 37,
            "end_i": 39,
            "severity": "minor"
        },
        ...
        ]
```

---

## Running the models

### Setup

All scripts expect the humeval data file at `../wmt25-genmt-humeval.jsonl` relative to this directory (i.e. one level up). The evaluated system and number of segments per language pair are configured in `qe_utils.py` (`HYP_SYSTEM`, `N_INSTANCES_PER_PAIR`, `TARGET_PAIRS`).

---

### Gemini (closed-weight, API)

**Requirements:** `pip install google-genai`

**Auth:**
```bash
export GEMINI_API_KEY="your_key_here"
```

**Test on one segment** (no file written, no credits consumed):
```bash
python run_qe.py --test
```

**Full run:**
```bash
python run_qe.py
```

Output is written to `quality_estimation_outputs_gemini/pred_gemini-3-flash-preview_wmt25-genmt-humeval.jsonl`. The model and output name are set by `MODEL_ID` at the top of `run_qe.py`. The script rate-limits automatically (~10 req/min free tier) and checkpoints after each language pair.

---

### Gemma-4 / Qwen3.6 (open-weight, local GPU)

**Requirements:** `pip install transformers torch accelerate`

Models must be downloaded to the HuggingFace cache before running on GPU nodes (which have no internet access). Run once on the login node:
```bash
bash slurm/cache_models.sh
```

**Test on one segment** (no file written, no GPU time consumed beyond loading):
```bash
python run_qe_local.py --model gemma4 --test
python run_qe_local.py --model qwen36 --test
```

**Full run:**
```bash
python run_qe_local.py --model gemma4
python run_qe_local.py --model qwen36
```

Output is written to `quality_estimation_outputs_local/pred_{model}_{humeval_file}`.

**With thinking mode** (chain-of-thought reasoning, uses more tokens):
```bash
python run_qe_local.py --model gemma4 --thinking
python run_qe_local.py --model qwen36 --thinking --max-new-tokens 8192
```

**Resume an interrupted run:**
```bash
python run_qe_local.py --model gemma4 --resume
```

**Submitting via Slurm:** Edit the configuration block at the top of `slurm/run_qe_local.sh` (GPU type, model, thinking flag, etc.) then:
```bash
sbatch slurm/run_qe_local.sh
```

GPU memory guidance (bf16):
- `gemma4` (~62 GB): 1× A100-80G / H100 / B200, or 2× A100-40G
- `qwen36` (~70 GB): 2× A100-80G / H100, or 1× B200

---

## TODOs for production (all 23 language pairs)

### Data / language pair support
- [ ] **Add all 23 language pairs to `TARGET_PAIRS`** in `qe_utils.py` once the official task data is released. Each entry needs `src_name`, `tgt_name`, `src_code`, `tgt_code`.
- [ ] **Update `extract_base_pair()` and `load_instances()`** to correctly parse `doc_id` prefixes for all new pairs. Currently validated against 7 pairs — some new language codes or regional variants (e.g. `zh-CN` vs `zh`, `pt-BR`) may not parse correctly. Audit against actual doc_ids in the released data.
- [ ] **Remove the `N_INSTANCES_PER_PAIR = 10` cap** in `qe_utils.py` for the production run (set to `None` or a large number to process all segments).
- [ ] **Verify output format** matches the official WMT26 task spec once it is published. Field names (`start_i`/`end_i` vs `start`/`end`, `severity` values, etc.) may differ from the dev data.

### Throughput and parallelisation
- [ ] **Benchmark each open-weight model** — time a full language pair run (all segments, one system) to estimate total GPU-hours needed across 23 pairs and plan Slurm allocations. Target: finish all pairs within the 1-week window.
- [ ] **Add batched inference** to `run_qe_local.py`. Currently processes one segment at a time (batch size 1). Adding left-padded batch generation would significantly improve throughput on H100/A100s, especially for shorter segments.
- [ ] **Parallelize across language pairs with Slurm array jobs.** Split the 23 pairs across multiple simultaneous jobs (e.g. one job per 3–4 pairs, or one job per model × pair) rather than running all pairs sequentially in one job. Add `--lang-pairs` CLI flag to `run_qe_local.py` to control which pairs a given job handles.
- [ ] **Finer-grained checkpointing** — currently `run_qe_local.py` writes output after each complete language pair. A job killed mid-pair loses all progress for that pair. Consider writing per-segment to avoid re-running large chunks on preemption.

### Robustness
- [ ] **Audit JSON parse failure rate** at scale. The current `parse_qe_output()` silently returns `[]` on failure. Add a counter and log a summary at the end of each run so failures are visible.
- [ ] **Handle missing hypotheses** — if a system has no translation for a given segment in `tgt_text`, `hyp_text` will be `""`. Currently this would send an empty string to the model. Add a skip/warn for empty hypotheses.
- [ ] **Validate `predicted_errors` field types** before writing output — ensure `start_i`/`end_i` are integers and `severity` is one of the expected values, so downstream evaluation scripts don't crash on malformed entries.