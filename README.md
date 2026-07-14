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