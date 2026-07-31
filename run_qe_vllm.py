"""
WMT26 quality estimation — local models via vLLM (Gemma-4, Qwen3.6).

Drop-in replacement for run_qe_local.py that swaps the transformers.generate()
backend for vLLM's continuous-batching engine. All prompt building, parsing,
checkpointing, and sharding logic is unchanged and imported directly from
qe_utils.py.

Key differences from run_qe_local.py:
  - No manual left-padding / fixed batch-size loop. Work is flattened across
    a whole "chunk" of segments and handed to vLLM in one call; vLLM's
    scheduler packs variable-length sequences together (continuous batching),
    which is what actually fixes the long-tail-of-thinking-tokens slowdown.
  - --chunk-size replaces --batch-size: it controls how many *segments*
    (not individual generate calls) are batched together before a checkpoint
    write. Bigger chunks -> better GPU utilization but coarser --resume
    granularity (a crash loses the whole in-flight chunk, not just one
    segment). Tune based on how flaky your nodes are.
  - Assumes vLLM's tokenizer.apply_chat_template + skip_special_tokens=False
    decoding is sufficient to expose thinking-block markers, and relies on
    qe_utils._strip_thinking (regex-based) to strip them, rather than
    Gemma's custom processor.parse_response(). Recommend validating this
    with --test on gemma4 before a full run — see note in generate_batch().

Setup:
  pip install vllm
  export HF_HUB_OFFLINE=1   # on compute nodes (models must be pre-cached)
Run:
  python run_qe_vllm.py --model gemma4 --thinking --test
  python run_qe_vllm.py --model qwen36 --thinking --max-new-tokens 8192 \
      --data-file mteval-test26.jsonl --segment-type official --pair cs-de \
      --resume --chunk-size 200 --tensor-parallel-size 4
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")

from qe_utils import (
    TARGET_PAIRS,
    SYSTEM_PROMPT,
    REF_LABELS,
    load_instances,
    get_domain,
    build_stage1_prompt,
    build_stage2_prompt,
    parse_stage1_output,
    parse_stage2_output,
    stage1_to_predicted_errors,
    make_row,
    append_row,
    load_done_rows,
)

# ============================================================================
# MODEL DEFINITIONS
# ============================================================================

MODELS = {
    "gemma4": "google/gemma-4-31B-it",
    "qwen36": "Qwen/Qwen3.6-35B-A3B",
}

DEFAULT_MAX_NEW_TOKENS = 512
DEFAULT_MAX_NEW_TOKENS_THINKING = 8192
MAX_NEW_TOKENS_STAGE2 = 64
DEFAULT_CHUNK_SIZE_THINKING = 150
DEFAULT_CHUNK_SIZE_NON_THINKING = 400


# ============================================================================
# VLLM MODEL WRAPPER
# ============================================================================

class VLLMModelWrapper:
    """Unified text-only inference wrapper for Gemma-4 and Qwen3.6, via vLLM.

    generate_batch() takes a flat list of chat message lists and returns a
    list of (response_text, input_tokens, output_tokens), matching the
    interface of LocalModelWrapper.generate_batch() in run_qe_local.py so
    the calling code barely needs to change.
    """

    def __init__(
        self,
        model_type: str,
        thinking: bool = False,
        tensor_parallel_size: int = 4,
        gpu_memory_utilization: float = 0.90,
        max_model_len: int | None = None,
        dtype: str = "bfloat16",
    ) -> None:
        from vllm import LLM

        if model_type not in MODELS:
            raise ValueError(f"Unknown model_type {model_type!r}; expected one of {list(MODELS)}")

        self.model_type = model_type
        self.thinking = thinking
        model_id = MODELS[model_type]

        logging.info(
            "Loading %s with vLLM (tp=%d, gpu_mem_util=%.2f, max_model_len=%s) …",
            model_id, tensor_parallel_size, gpu_memory_utilization, max_model_len,
        )
        llm_kwargs = dict(
            model=model_id,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            dtype=dtype,
            trust_remote_code=True,
            # Shared system prompt + domain text across nearly every request in
            # a shard — prefix caching avoids re-computing that KV cache each time.
            enable_prefix_caching=True,
        )
        if max_model_len is not None:
            llm_kwargs["max_model_len"] = max_model_len

        self.llm = LLM(**llm_kwargs)
        self.tokenizer = self.llm.get_tokenizer()

        logging.info("Model loaded: %s | thinking: %s", model_id, thinking)

    def _sampling_params(self, max_new_tokens: int, thinking: bool):
        from vllm import SamplingParams

        if thinking:
            return SamplingParams(
                temperature=0.6,
                top_p=0.95,
                max_tokens=max_new_tokens,
                # Keep special tokens (e.g. <think>, <|channel>...) in the decoded
                # text so qe_utils._strip_thinking's regexes can find and strip them.
                skip_special_tokens=False,
            )
        return SamplingParams(
            temperature=0.0,
            max_tokens=max_new_tokens,
            skip_special_tokens=False,
        )

    def generate(
        self,
        messages: list[dict],
        max_new_tokens: int,
        enable_thinking: bool | None = None,
    ) -> tuple[str, int, int]:
        """Single-item convenience wrapper (used by --test mode)."""
        return self.generate_batch([messages], max_new_tokens, enable_thinking)[0]

    def generate_batch(
        self,
        batch_messages: list[list[dict]],
        max_new_tokens: int,
        enable_thinking: bool | None = None,
    ) -> list[tuple[str, int, int]]:
        """Run batched inference via vLLM. Returns list of (text, input_tokens, output_tokens)
        in the same order as batch_messages. No manual padding/looping needed —
        vLLM's internal scheduler continuously batches variable-length requests.
        """
        thinking = self.thinking if enable_thinking is None else enable_thinking
        prompts = [
            self.tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True, enable_thinking=thinking,
            )
            for msgs in batch_messages
        ]
        sp = self._sampling_params(max_new_tokens, thinking)
        outputs = self.llm.generate(prompts, sp, use_tqdm=False)

        results = []
        for out in outputs:
            gen = out.outputs[0]
            in_tok = len(out.prompt_token_ids)
            out_tok = len(gen.token_ids)
            results.append((gen.text, in_tok, out_tok))
        return results


# ============================================================================
# CLI
# ============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="WMT26 QE with local models via vLLM (Gemma-4, Qwen3.6)"
    )
    p.add_argument(
        "--model", required=True, choices=list(MODELS),
        help="Model to use: gemma4=google/gemma-4-31B-it, qwen36=Qwen/Qwen3.6-35B-A3B",
    )
    p.add_argument(
        "--thinking", action="store_true",
        help="Enable chain-of-thought thinking mode (uses sampling, more tokens).",
    )
    p.add_argument(
        "--max-new-tokens", type=int, default=None,
        help=f"Max tokens to generate for Stage 1 (default {DEFAULT_MAX_NEW_TOKENS} "
             f"without thinking, {DEFAULT_MAX_NEW_TOKENS_THINKING} with --thinking). "
             f"Stage 2 always uses {MAX_NEW_TOKENS_STAGE2} tokens (thinking disabled for Stage 2).",
    )
    p.add_argument(
        "--output-dir", default="quality_estimation_outputs_local",
        help="Output directory (default: quality_estimation_outputs_local).",
    )
    p.add_argument(
        "--data-file", required=True,
        help="Path to the combined JSONL data file (e.g. mteval-test26.jsonl).",
    )
    p.add_argument(
        "--segment-type", default="all", choices=["official", "challenge", "all"],
        help="Which segments to evaluate: 'official', 'challenge', or 'all' (default).",
    )
    p.add_argument(
        "--pair", default=None,
        help="Process only this language pair (e.g. cs-de). Useful for SLURM job arrays.",
    )
    p.add_argument(
        "--max-segments", type=int, default=None,
        help="Cap segments per language pair; useful for testing.",
    )
    p.add_argument(
        "--chunk-size", type=int, default=None,
        help="Number of segments to flatten into one vLLM generate() call before "
             f"checkpointing. Defaults to {DEFAULT_CHUNK_SIZE_NON_THINKING} without "
             f"--thinking, {DEFAULT_CHUNK_SIZE_THINKING} with --thinking. Larger values "
             "improve GPU utilization but coarsen --resume granularity. Reduce if OOM.",
    )
    p.add_argument(
        "--tensor-parallel-size", type=int, default=4,
        help="Number of GPUs to shard the model across (should match --gres=gpu:N). Default 4.",
    )
    p.add_argument(
        "--gpu-memory-utilization", type=float, default=0.90,
        help="Fraction of GPU memory vLLM may reserve for weights + KV cache. Default 0.90.",
    )
    p.add_argument(
        "--max-model-len", type=int, default=None,
        help="Max sequence length (prompt + generation) for vLLM's KV cache sizing. "
             "Defaults to the model's config. Set explicitly (e.g. 16384) if vLLM's "
             "auto-detected default doesn't leave enough room for prompt + max-new-tokens.",
    )
    p.add_argument(
        "--test", action="store_true",
        help="Run one segment of one system — prints prompts + responses, no file written.",
    )
    p.add_argument(
        "--resume", action="store_true",
        help="Skip segments already present in the output file (matched by item_id).",
    )
    p.add_argument(
        "--with-ref", action="store_true",
        help="Include the reference translation from the data 'ref' field in Stage 1 and "
             "Stage 2 prompts. The prompt wording adapts to the reference type "
             "(human/postedit/pseudo). Falls back to the no-reference prompt when "
             "ref text is absent for a given segment.",
    )
    p.add_argument(
        "--num-shards", type=int, default=1,
        help="Total number of shards for parallel array jobs. Default 1 = no sharding.",
    )
    p.add_argument(
        "--shard", type=int, default=0,
        help="Zero-based shard index for this job (0 .. num-shards-1).",
    )
    return p.parse_args()


# ============================================================================
# MAIN
# ============================================================================

def main():
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    max_new_tokens_s1 = args.max_new_tokens
    if max_new_tokens_s1 is None:
        max_new_tokens_s1 = DEFAULT_MAX_NEW_TOKENS_THINKING if args.thinking else DEFAULT_MAX_NEW_TOKENS

    chunk_size = args.chunk_size
    if chunk_size is None:
        chunk_size = DEFAULT_CHUNK_SIZE_THINKING if args.thinking else DEFAULT_CHUNK_SIZE_NON_THINKING

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

    wrapper = VLLMModelWrapper(
        args.model,
        thinking=args.thinking,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
    )

    # ── Smoke-test mode ───────────────────────────────────────────────────
    if args.test:
        pair = next(iter(active_pairs))
        cfg = active_pairs[pair]
        instances = instances_by_pair.get(pair, [])
        if not instances:
            sys.exit(f"No instances found for {pair}")
        inst = instances[0]

        hyps = inst["_raw"].get("hyps", {})
        if not hyps:
            sys.exit("No hypotheses found in first instance")
        test_system, hyp = next(iter(hyps.items()))
        src = inst["src_text"]
        domain = get_domain(inst["doc_id"])
        ref_text = inst.get("refA") if args.with_ref else None
        ref_type = inst.get("ref_type") if args.with_ref else None

        prompt1 = build_stage1_prompt(src, hyp, cfg, domain, ref_text=ref_text, ref_type=ref_type)
        messages1 = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt1}]
        print("=" * 60)
        ref_info = f" | REF_TYPE: {ref_type}" if ref_text else " | REF: none"
        print(f"PAIR: {pair} | DOMAIN: {domain} | SYSTEM: {test_system}{ref_info}")
        print("STAGE 1 PROMPT:")
        print(prompt1)
        print("=" * 60)
        raw1, in_tok1, out_tok1 = wrapper.generate(messages1, max_new_tokens_s1)
        print("STAGE 1 RESPONSE:")
        print(raw1)
        print("=" * 60)
        stage1_text = parse_stage1_output(raw1)
        parsed = stage1_to_predicted_errors(stage1_text, hyp)

        prompt2 = build_stage2_prompt(src, hyp, stage1_text, cfg, domain, ref_text=ref_text, ref_type=ref_type)
        messages2 = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt2}]
        print("STAGE 2 PROMPT:")
        print(prompt2)
        print("=" * 60)
        raw2, in_tok2, out_tok2 = wrapper.generate(messages2, MAX_NEW_TOKENS_STAGE2, enable_thinking=False)
        print("STAGE 2 RESPONSE:")
        print(raw2)
        print("=" * 60)
        score = parse_stage2_output(raw2)
        print(f"PARSED task1 result ({in_tok1}+{in_tok2} in / {out_tok1}+{out_tok2} out tokens):")
        print(json.dumps(parsed, indent=2, ensure_ascii=False))
        print(f"SCORE: {score}")
        return

    # ── Full run ──────────────────────────────────────────────────────────
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    thinking_tag = "_thinking" if args.thinking else ""
    model_tag = f"{args.model}{thinking_tag}"
    ref_tag = "_ref" if args.with_ref else ""
    shard_tag = f"_s{args.shard}of{args.num_shards}" if args.num_shards > 1 else ""

    for pair, cfg in active_pairs.items():
        output_path = output_dir / f"pred_{model_tag}_{pair}{ref_tag}{shard_tag}.jsonl"
        instances = instances_by_pair.get(pair, [])

        if args.num_shards > 1:
            instances = [inst for i, inst in enumerate(instances) if i % args.num_shards == args.shard]

        done_rows: dict = {}
        if args.resume:
            done_rows = load_done_rows(output_path)
            base_path = output_dir / f"pred_{model_tag}_{pair}{ref_tag}.jsonl"
            if args.num_shards > 1 and base_path.exists():
                for iid, row in load_done_rows(base_path).items():
                    if iid not in done_rows:
                        done_rows[iid] = row

        work_items = []
        for inst in instances:
            existing = done_rows.get(inst["doc_id"])
            done_sys = (
                {s for s, v in existing.get("task2_pred", {}).items() if v is not None}
                if existing else set()
            )
            new_sys = [(s, h) for s, h in inst["_raw"].get("hyps", {}).items()
                       if s not in done_sys]
            if new_sys:
                work_items.append((inst, new_sys, existing))

        if args.max_segments is not None:
            work_items = work_items[:args.max_segments]

        logging.info("[%s] %d/%d segments to process → %s (chunk_size=%d)",
                     pair, len(work_items), len(instances), output_path.name, chunk_size)

        pair_start = time.monotonic()
        n_done = 0
        for chunk_start in range(0, len(work_items), chunk_size):
            chunk = work_items[chunk_start:chunk_start + chunk_size]

            # Per-segment result accumulators, indexed by position within the chunk.
            task1_results_list = [dict() for _ in chunk]
            task2_results_list = [dict() for _ in chunk]

            # Flatten stage-1 work across the whole chunk: (idx_in_chunk, system, hyp).
            flat = []
            for i, (inst, systems, existing_row) in enumerate(chunk):
                for system, hyp in systems:
                    if not hyp:
                        task1_results_list[i][system] = {"errors": [], "omission": "major", "instruction_fault": None}
                        task2_results_list[i][system] = 0
                    else:
                        flat.append((i, system, hyp))

            if flat:
                domains = [get_domain(chunk[i][0]["doc_id"]) for i, _, _ in flat]
                srcs = [chunk[i][0]["src_text"] for i, _, _ in flat]
                refs = [
                    (chunk[i][0].get("refA") if args.with_ref else None,
                     chunk[i][0].get("ref_type") if args.with_ref else None)
                    for i, _, _ in flat
                ]

                s1_msgs = [
                    [{"role": "system", "content": SYSTEM_PROMPT},
                     {"role": "user", "content": build_stage1_prompt(
                         src, hyp, cfg, domain, ref_text=ref_text, ref_type=ref_type)}]
                    for (i, system, hyp), src, domain, (ref_text, ref_type)
                    in zip(flat, srcs, domains, refs)
                ]
                try:
                    s1_outputs = wrapper.generate_batch(s1_msgs, max_new_tokens_s1)
                except Exception as e:
                    if "CUDA error" in str(e):
                        raise
                    logging.error("[%s] chunk@%d stage1 batch failed (%d segments lost): %s",
                                   pair, chunk_start, len(chunk), e)
                    for i, system, hyp in flat:
                        task1_results_list[i][system] = {"errors": [], "omission": None, "instruction_fault": None}
                        task2_results_list[i][system] = None
                    flat = []  # skip stage 2

                if flat:
                    s1_texts = [parse_stage1_output(r[0]) for r in s1_outputs]

                    s2_msgs = [
                        [{"role": "system", "content": SYSTEM_PROMPT},
                         {"role": "user", "content": build_stage2_prompt(
                             src, hyp, s1_text, cfg, domain, ref_text=ref_text, ref_type=ref_type)}]
                        for (i, system, hyp), src, domain, s1_text, (ref_text, ref_type)
                        in zip(flat, srcs, domains, s1_texts, refs)
                    ]
                    try:
                        s2_outputs = wrapper.generate_batch(s2_msgs, MAX_NEW_TOKENS_STAGE2, enable_thinking=False)
                    except Exception as e:
                        logging.error("[%s] chunk@%d stage2 batch failed: %s", pair, chunk_start, e)
                        s2_outputs = [("", 0, 0)] * len(flat)

                    for (i, system, hyp), s1_text, s2_res in zip(flat, s1_texts, s2_outputs):
                        doc_id = chunk[i][0]["doc_id"]
                        parsed = stage1_to_predicted_errors(s1_text, hyp, log_ctx=f"[{doc_id} | {system}]")
                        score = parse_stage2_output(s2_res[0])
                        task1_results_list[i][system] = {
                            "errors": parsed["errors"],
                            "omission": parsed["omission"],
                            "instruction_fault": parsed["instruction_fault"],
                        }
                        task2_results_list[i][system] = score

            # Checkpoint the whole chunk.
            for i, (inst, systems, existing_row) in enumerate(chunk):
                if existing_row is not None:
                    merged_t1 = {**existing_row.get("task1_pred", {}), **task1_results_list[i]}
                    merged_t2 = {**existing_row.get("task2_pred", {}), **task2_results_list[i]}
                else:
                    merged_t1, merged_t2 = task1_results_list[i], task2_results_list[i]
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