"""
WMT26 quality estimation — XCOMET (Unbabel/XCOMET-XL and Unbabel/XCOMET-XXL).

Unlike the LLM-as-judge scripts, XCOMET produces scores and error spans directly
from the model without a two-stage prompting pipeline. Both QE mode (no reference)
and reference-aware mode are supported via --with-ref.

Error spans from XCOMET metadata are mapped to the WMT26 task1_pred format
(character offsets into the hypothesis, severity, category). Scores are on a
[0, 1] scale from COMET and are multiplied by 100 for the WMT26 task2_pred format.

Setup:
  pip install -U unbabel-comet
  # Pre-download models (run once before going offline):
  python -c "from comet import download_model; download_model('Unbabel/XCOMET-XL'); download_model('Unbabel/XCOMET-XXL')"
  export HF_HUB_OFFLINE=1   # on compute nodes after pre-download

Run:
  python run_qe_xcomet.py --model xl --data-file mteval-test26.jsonl --test
  python run_qe_xcomet.py --model xxl --with-ref --data-file mteval-test26.jsonl --pair en-de
  python run_qe_xcomet.py --model xl --data-file mteval-test26.jsonl --resume
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

from qe_utils import (
    TARGET_PAIRS,
    load_instances,
    make_row,
    append_row,
    load_done_rows,
)


# ============================================================================
# CONFIG
# ============================================================================

MODELS = {
    "xl":  "Unbabel/XCOMET-XL",
    "xxl": "Unbabel/XCOMET-XXL",
}

DEFAULT_BATCH_SIZE = {
    "xl":  128,
    "xxl": 64,
}

OUTPUT_DIR = Path("quality_estimation_outputs_xcomet")

# Number of segments to accumulate before writing a checkpoint.
# Larger = fewer writes = faster, but coarser resume granularity.
CHUNK_SIZE = 500


# ============================================================================
# ERROR SPAN CONVERSION
# ============================================================================

def xcomet_spans_to_task1(spans: list) -> dict:
    """Convert XCOMET error_spans for one segment to the WMT26 task1_pred format.

    XCOMET returns a list of span dicts per segment, each with:
      start, end  — character offsets in the hypothesis (half-open [start, end))
      severity    — "major" or "minor"
      type        — MQM error category string (e.g. "fluency/grammar")

    Omissions (accuracy/omission) have no meaningful span in the hypothesis;
    they are captured in the top-level "omission" field instead.
    """
    errors = []
    omission_severities = []

    for span in (spans or []):
        raw_sev = (span.get("severity") or "minor").lower()
        # XCOMET uses "critical" in addition to "major"/"minor".
        # Map "critical" → "major"; anything else unknown → "minor".
        if raw_sev in ("major", "critical"):
            severity = "major"
        else:
            severity = "minor"

        # Field name varies across COMET versions: "type" is most common.
        category = span.get("type") or span.get("category") or "other"

        start = span.get("start", -1)
        end = span.get("end", -1)

        cat_lower = category.lower()
        if "omission" in cat_lower:
            omission_severities.append(severity)
        elif isinstance(start, int) and isinstance(end, int) and 0 <= start < end:
            errors.append({
                "start": start,
                "end": end,
                "severity": severity,
                "category": category,
            })

    def _max_sev(sevs):
        if "major" in sevs:
            return "major"
        if "minor" in sevs:
            return "minor"
        return None

    return {
        "errors": errors,
        "omission": _max_sev(omission_severities),
        "instruction_fault": None,
    }


# ============================================================================
# MAIN
# ============================================================================

def parse_args():
    p = argparse.ArgumentParser(description="WMT26 QE with XCOMET (XL or XXL)")
    p.add_argument("--model", required=True, choices=list(MODELS),
                   help="xl = Unbabel/XCOMET-XL  |  xxl = Unbabel/XCOMET-XXL")
    p.add_argument("--with-ref", action="store_true",
                   help="Include reference translation (reference-aware / MQM mode). "
                        "Falls back to QE mode for segments without a reference.")
    p.add_argument("--data-file", required=True,
                   help="Path to the combined JSONL data file (e.g. mteval-test26.jsonl).")
    p.add_argument("--segment-type", default="all", choices=["official", "challenge", "all"])
    p.add_argument("--pair", default=None,
                   help="Process only this language pair (e.g. en-de). "
                        "Useful for SLURM job arrays.")
    p.add_argument("--max-segments", type=int, default=None,
                   help="Cap segments per language pair; useful for testing.")
    p.add_argument("--batch-size", type=int, default=None,
                   help=f"Batch size for model.predict(). "
                        f"Defaults: xl={DEFAULT_BATCH_SIZE['xl']}, xxl={DEFAULT_BATCH_SIZE['xxl']}.")
    p.add_argument("--gpus", type=int, default=1,
                   help="Number of GPUs for COMET data-parallel inference (default: 1).")
    p.add_argument("--output-dir", default=str(OUTPUT_DIR),
                   help=f"Output directory (default: {OUTPUT_DIR}).")
    p.add_argument("--resume", action="store_true",
                   help="Skip segments already present in the output file.")
    p.add_argument("--test", action="store_true",
                   help="Run one segment of one system and print output; no file written.")
    return p.parse_args()


def main():
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    batch_size = args.batch_size or DEFAULT_BATCH_SIZE[args.model]
    model_id = MODELS[args.model]

    logging.info("Loading %s …", model_id)
    from comet import download_model, load_from_checkpoint
    model_path = download_model(model_id)
    model = load_from_checkpoint(model_path)
    logging.info("Model loaded.")

    # Compatibility fix: newer transformers loads XLMRobertaTokenizerFast by default,
    # which delegates attribute lookup to its internal tokenizer via __getattr__ and
    # does not expose build_inputs_with_special_tokens — a method COMET's encoder
    # calls directly in concat_sequences.  Patch it onto the class before any
    # DataLoader workers are forked so they inherit the fix automatically.
    try:
        import transformers as _hf
        _TokenizerCls = type(model.encoder.tokenizer)
        if not callable(getattr(_TokenizerCls, "build_inputs_with_special_tokens", None)):
            def _build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
                cls_id = self.cls_token_id if self.cls_token_id is not None else 0
                sep_id = self.sep_token_id if self.sep_token_id is not None else 2
                if token_ids_1 is None:
                    return [cls_id] + list(token_ids_0) + [sep_id]
                return [cls_id] + list(token_ids_0) + [sep_id, sep_id] + list(token_ids_1) + [sep_id]
            _TokenizerCls.build_inputs_with_special_tokens = _build_inputs_with_special_tokens
            logging.info("Patched %s.build_inputs_with_special_tokens for transformers compatibility.",
                         _TokenizerCls.__name__)
    except Exception as _e:
        logging.warning("Could not apply tokenizer compatibility patch: %s", _e)

    # Compatibility fix 2: newer transformers filters None values from tuple output
    # (return_dict=False).  XLM-R has no pooler head, so pooler_output=None gets
    # dropped, leaving a 2-tuple (last_hidden_state, hidden_states).  COMET's
    # XLMREncoder.forward expects a 3-tuple and unpacks it as:
    #   last_hidden_states, _, all_layers = self.model(...)
    # Wrap the backbone forward to re-insert the missing None at position 1.
    try:
        _orig_backbone_fwd = model.encoder.model.forward

        def _patched_backbone_forward(*args, **kwargs):
            result = _orig_backbone_fwd(*args, **kwargs)
            if isinstance(result, tuple) and len(result) == 2:
                last_hs, second = result
                # second is hidden_states (a tuple of tensors) when pooler_output
                # was filtered; pooler_output itself would be a plain tensor
                if isinstance(second, (tuple, list)):
                    return (last_hs, None, second)
            return result

        model.encoder.model.forward = _patched_backbone_forward
        logging.info("Patched encoder backbone forward for transformers tuple-output compatibility.")
    except Exception as _e:
        logging.warning("Could not apply encoder forward compatibility patch: %s", _e)

    if args.pair is not None:
        if args.pair not in TARGET_PAIRS:
            sys.exit(f"Unknown pair {args.pair!r}. Valid pairs: {list(TARGET_PAIRS)}")
        active_pairs = {args.pair: TARGET_PAIRS[args.pair]}
    else:
        active_pairs = TARGET_PAIRS

    instances_by_pair = load_instances(
        data_file=args.data_file,
        target_pairs=active_pairs,
        segment_type=args.segment_type,
    )

    # ── Smoke-test mode ───────────────────────────────────────────────────
    if args.test:
        pair = next(iter(active_pairs))
        instances = instances_by_pair.get(pair, [])
        if not instances:
            sys.exit(f"No instances found for {pair}")
        inst = instances[0]
        hyps = inst["_raw"].get("hyps", {})
        if not hyps:
            sys.exit("No hypotheses found in first instance")
        test_system, hyp = next(iter(hyps.items()))
        ref = inst.get("refA") if args.with_ref else None

        sample = {"src": inst["src_text"], "mt": hyp}
        if ref:
            sample["ref"] = ref

        print("=" * 60)
        ref_info = f" | REF: {ref[:60]}…" if ref else " | REF: none"
        print(f"PAIR: {pair} | SYSTEM: {test_system}{ref_info}")
        print(f"SRC: {inst['src_text'][:120]}")
        print(f"HYP: {hyp[:120]}")
        print("=" * 60)

        output = model.predict([sample], batch_size=1, gpus=args.gpus)
        score = output.scores[0]
        spans = output.metadata.error_spans[0] if output.metadata.error_spans else []
        task1 = xcomet_spans_to_task1(spans)

        print(f"Raw score (0-1): {score:.4f}  →  task2_pred: {score * 100:.2f}")
        print(f"Raw error_spans: {spans}")
        print(f"task1_pred: {json.dumps(task1, indent=2, ensure_ascii=False)}")
        return

    # ── Full run ──────────────────────────────────────────────────────────
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ref_tag = "_ref" if args.with_ref else ""

    for pair in active_pairs:
        output_path = output_dir / f"pred_{args.model}{ref_tag}_{pair}.jsonl"
        instances = instances_by_pair.get(pair, [])

        done_rows: dict = {}
        if args.resume:
            done_rows = load_done_rows(output_path)

        work_items = []
        for inst in instances:
            existing = done_rows.get(inst["doc_id"])
            done_sys = (
                {s for s, v in existing.get("task2_pred", {}).items() if v is not None}
                if existing else set()
            )
            new_sys = [
                (s, h) for s, h in inst["_raw"].get("hyps", {}).items()
                if s not in done_sys
            ]
            if new_sys:
                work_items.append((inst, new_sys, existing))

        if args.max_segments is not None:
            work_items = work_items[:args.max_segments]

        logging.info("[%s] %d/%d segments to process → %s",
                     pair, len(work_items), len(instances), output_path.name)

        pair_start = time.monotonic()
        n_done = 0

        for chunk_start in range(0, len(work_items), CHUNK_SIZE):
            chunk = work_items[chunk_start : chunk_start + CHUNK_SIZE]

            # Flatten all (segment, system) pairs in this chunk into a single
            # list for a single model.predict() call — COMET handles batching.
            flat_samples = []   # list of COMET input dicts
            flat_meta = []      # (chunk_idx, system, hyp) for each sample

            for i, (inst, systems, _) in enumerate(chunk):
                src = inst["src_text"]
                ref = inst.get("refA") if args.with_ref else None
                for system, hyp in systems:
                    if not hyp:
                        # Empty hypothesis — mark directly, skip model call.
                        flat_meta.append((i, system, ""))
                        flat_samples.append(None)
                        continue
                    sample = {"src": src, "mt": hyp}
                    if ref:
                        sample["ref"] = ref
                    flat_samples.append(sample)
                    flat_meta.append((i, system, hyp))

            # Run model only on non-empty hypotheses.
            valid_indices = [j for j, s in enumerate(flat_samples) if s is not None]
            valid_samples = [flat_samples[j] for j in valid_indices]

            scores_map = {}    # flat_index → float score
            spans_map = {}     # flat_index → list of span dicts

            if valid_samples:
                # COMET's DataLoader collate uses sample[0]'s keys as the template
                # for all samples in the batch. A mix of samples with and without
                # 'ref' triggers KeyError. Split into groups and predict separately.
                ref_positions   = [k for k, s in enumerate(valid_samples) if "ref" in s]
                noref_positions = [k for k, s in enumerate(valid_samples) if "ref" not in s]

                for positions in (ref_positions, noref_positions):
                    if not positions:
                        continue
                    group_samples = [valid_samples[k] for k in positions]
                    comet_output = model.predict(
                        group_samples,
                        batch_size=batch_size,
                        gpus=args.gpus,
                        progress_bar=False,
                    )
                    error_spans_list = comet_output.metadata.error_spans or []
                    for rank, k in enumerate(positions):
                        j = valid_indices[k]
                        scores_map[j] = comet_output.scores[rank]
                        spans_map[j] = (
                            error_spans_list[rank] if rank < len(error_spans_list) else []
                        )

            # Accumulate results per chunk position.
            task1_results = [{} for _ in chunk]
            task2_results = [{} for _ in chunk]

            for j, (i, system, hyp) in enumerate(flat_meta):
                if not hyp:
                    task1_results[i][system] = {
                        "errors": [], "omission": "major", "instruction_fault": None,
                    }
                    task2_results[i][system] = 0
                else:
                    raw_score = scores_map.get(j, 0.0)
                    # Clamp to [0, 100] after scaling.
                    task2_results[i][system] = max(0.0, min(100.0, raw_score * 100))
                    task1_results[i][system] = xcomet_spans_to_task1(spans_map.get(j, []))

            # Checkpoint.
            for i, (inst, _, existing_row) in enumerate(chunk):
                if existing_row is not None:
                    merged_t1 = {**existing_row.get("task1_pred", {}), **task1_results[i]}
                    merged_t2 = {**existing_row.get("task2_pred", {}), **task2_results[i]}
                else:
                    merged_t1, merged_t2 = task1_results[i], task2_results[i]
                append_row(make_row(inst, merged_t1, merged_t2), output_path)

            n_done += len(chunk)
            elapsed = int(time.monotonic() - pair_start)
            logging.info("[%s] %d/%d segments done | elapsed %dh%02dm%02ds",
                         pair, n_done, len(work_items),
                         elapsed // 3600, (elapsed % 3600) // 60, elapsed % 60)

        logging.info("[%s] complete → %s", pair, output_path.name)

    logging.info("Done. Output dir: %s", output_dir)


if __name__ == "__main__":
    main()
