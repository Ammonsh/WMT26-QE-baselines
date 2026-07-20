"""
WMT26 quality estimation — local HuggingFace models (Gemma-4, Qwen3.6).

Supports:
  google/gemma-4-31B-it     (--model gemma4)
  Qwen/Qwen3.6-35B-A3B      (--model qwen36)

Both run at segment level with optional thinking mode (--thinking).
Uses two-stage GEMBA-ESA prompting (same as run_qe.py) with domain-specific prompts.

Setup:
  conda activate qwen   # or whichever env has transformers + torch
  export HF_HUB_OFFLINE=1   # on compute nodes (models must be pre-cached)
Run:
  python run_qe_local.py --model gemma4 [--thinking] [--test]
  python run_qe_local.py --model qwen36 [--thinking] [--max-new-tokens 8192] [--test]
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

# Compute nodes have no internet — default to offline mode so transformers
# loads from cache without network retries. Override by setting
# HF_HUB_OFFLINE=0 in your environment before launching this script.
os.environ.setdefault("HF_HUB_OFFLINE", "1")

from qe_utils import (
    TARGET_PAIRS,
    SYSTEM_PROMPT,
    load_instances,
    get_domain,
    build_stage1_prompt,
    build_stage2_prompt,
    parse_stage1_output,
    parse_stage2_output,
    stage1_to_predicted_errors,
    make_row,
    append_row,
    load_done_ids,
)

# ============================================================================
# MODEL DEFINITIONS
# ============================================================================

MODELS = {
    "gemma4": "google/gemma-4-31B-it",
    "qwen36": "Qwen/Qwen3.6-35B-A3B",
}

# Default max_new_tokens. With thinking, much more budget is needed.
DEFAULT_MAX_NEW_TOKENS = 512
DEFAULT_MAX_NEW_TOKENS_THINKING = 8192
# Stage 2 outputs {"score": N} JSON — small budget is fine; 64 allows for
# markdown fences (```json...```) in case the model adds them.
MAX_NEW_TOKENS_STAGE2 = 64


# ============================================================================
# LOCAL MODEL WRAPPER
# ============================================================================

class LocalModelWrapper:
    """Unified text-only inference wrapper for Gemma-4 and Qwen3.6.

    Both models are loaded with device_map="auto" across all visible GPUs.
    The generate() method returns (response_text, input_tokens, output_tokens).
    """

    def __init__(self, model_type: str, thinking: bool = False) -> None:
        import torch
        from transformers import AutoModelForCausalLM

        if model_type not in MODELS:
            raise ValueError(f"Unknown model_type {model_type!r}; expected one of {list(MODELS)}")

        self.model_type = model_type
        self.thinking = thinking
        model_id = MODELS[model_type]

        logging.info("Loading processor for %s …", model_id)
        if model_type == "gemma4":
            from transformers import AutoProcessor
            self.processor = AutoProcessor.from_pretrained(model_id)
        else:
            # Use AutoTokenizer for Qwen3.6 (text-only).
            # Avoids importing qwen_vl_utils/torchaudio which trigger FIPS
            # selftest failures on RHEL9 nodes via RPATH-linked system libssl.
            from transformers import AutoTokenizer
            self.processor = AutoTokenizer.from_pretrained(model_id)

        logging.info("Loading model %s across available GPUs …", model_id)
        try:
            import flash_attn  # noqa: F401
            attn_impl = "flash_attention_2"
        except ImportError:
            attn_impl = "sdpa"
        logging.info("Attention implementation: %s", attn_impl)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype="auto",
            device_map="auto",
            low_cpu_mem_usage=True,
            attn_implementation=attn_impl,
        )
        self.model.eval()

        # Resolve tokenizer for token counting
        self._tokenizer = (
            self.processor.tokenizer
            if hasattr(self.processor, "tokenizer")
            else self.processor
        )

        logging.info(
            "Model loaded: %s | dtype: %s | thinking: %s",
            model_id, next(self.model.parameters()).dtype, thinking,
        )

    def _gen_kwargs(self, max_new_tokens: int) -> dict:
        kwargs: dict = {"max_new_tokens": max_new_tokens,
                        "pad_token_id": self._tokenizer.eos_token_id}
        if self.thinking:
            kwargs.update({"do_sample": True, "temperature": 0.6, "top_p": 0.95})
        else:
            kwargs.update({"do_sample": False, "temperature": None, "top_p": None})
        return kwargs

    def _decode_new_ids(self, new_ids) -> str:
        if self.model_type == "gemma4":
            # Decode keeping special tokens so parse_response() can strip thinking blocks.
            raw = self.processor.decode(new_ids, skip_special_tokens=False)
            parsed = self.processor.parse_response(raw)
            return parsed.get("content", "") if isinstance(parsed, dict) else str(parsed)
        else:
            return self.processor.decode(new_ids, skip_special_tokens=True)

    def generate(
        self,
        messages: list[dict],
        max_new_tokens: int,
    ) -> tuple[str, int, int]:
        """Run a single inference call. Returns (response_text, input_tokens, output_tokens)."""
        import torch

        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=self.thinking,
        )

        inputs = self.processor(text=text, return_tensors="pt").to(self.model.device)
        input_tokens = inputs["input_ids"].shape[-1]

        with torch.no_grad():
            output_ids = self.model.generate(**inputs, **self._gen_kwargs(max_new_tokens))

        new_ids = output_ids[0, input_tokens:]
        return self._decode_new_ids(new_ids), input_tokens, len(new_ids)

    def generate_batch(
        self,
        batch_messages: list[list[dict]],
        max_new_tokens: int,
    ) -> list[tuple[str, int, int]]:
        """Run batched inference. Returns list of (response_text, input_tokens, output_tokens).

        Uses left-padding so all sequences in the batch generate from the same
        position, which is required for decoder-only autoregressive models.
        """
        import torch

        texts = [
            self.processor.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True,
                enable_thinking=self.thinking,
            )
            for msgs in batch_messages
        ]

        # Left-pad: pad tokens go before the prompt so generation starts at the same index.
        if hasattr(self.processor, "tokenizer"):
            self.processor.tokenizer.padding_side = "left"
        else:
            self.processor.padding_side = "left"

        inputs = self.processor(text=texts, return_tensors="pt", padding=True).to(self.model.device)
        padded_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            output_ids = self.model.generate(**inputs, **self._gen_kwargs(max_new_tokens))

        results = []
        for i in range(len(batch_messages)):
            actual_input_len = int(inputs["attention_mask"][i].sum().item())
            new_ids = output_ids[i, padded_len:]
            results.append((self._decode_new_ids(new_ids), actual_input_len, len(new_ids)))
        return results


# ============================================================================
# CLI
# ============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="WMT26 QE with local HF models (Gemma-4, Qwen3.6)"
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
             f"Stage 2 always uses {MAX_NEW_TOKENS_STAGE2} tokens.",
    )
    p.add_argument(
        "--output-dir", default="quality_estimation_outputs_local",
        help="Output directory (default: quality_estimation_outputs_local).",
    )
    p.add_argument(
        "--pair", default=None,
        help="Process only this language pair (e.g. cs-de). Useful for SLURM job arrays.",
    )
    p.add_argument(
        "--batch-size", type=int, default=None,
        help="Number of (system, hyp) pairs to generate in parallel per model.generate() call. "
             "Defaults to 8 without --thinking, 2 with --thinking. Reduce if OOM.",
    )
    p.add_argument(
        "--test", action="store_true",
        help="Run one segment of one system — prints prompts + responses, no file written.",
    )
    p.add_argument(
        "--resume", action="store_true",
        help="Skip segments already present in the output file (matched by item_id).",
    )
    return p.parse_args()


# ============================================================================
# MAIN
# ============================================================================

def _run_two_stage(wrapper, src, hyp, cfg, domain, max_new_tokens_s1):
    """Run Stage 1 + Stage 2 for one (src, hyp) pair.

    Returns (stage1_text, parsed, score, tok_counts) where parsed is the dict
    from stage1_to_predicted_errors and tok_counts is (in1, out1, in2, out2).
    """
    messages1 = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_stage1_prompt(src, hyp, cfg, domain)},
    ]
    raw1, in_tok1, out_tok1 = wrapper.generate(messages1, max_new_tokens_s1)
    stage1_text = parse_stage1_output(raw1)
    parsed = stage1_to_predicted_errors(stage1_text, hyp)

    messages2 = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_stage2_prompt(src, hyp, stage1_text, cfg, domain)},
    ]
    raw2, in_tok2, out_tok2 = wrapper.generate(messages2, MAX_NEW_TOKENS_STAGE2)
    score = parse_stage2_output(raw2)
    return stage1_text, parsed, score, (in_tok1, out_tok1, in_tok2, out_tok2)


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

    batch_size = args.batch_size
    if batch_size is None:
        batch_size = 2 if args.thinking else 8

    # Determine which pairs to process
    if args.pair is not None:
        if args.pair not in TARGET_PAIRS:
            sys.exit(f"Unknown pair {args.pair!r}. Valid pairs: {list(TARGET_PAIRS)}")
        active_pairs = {args.pair: TARGET_PAIRS[args.pair]}
    else:
        active_pairs = TARGET_PAIRS

    instances_by_pair = load_instances(target_pairs=active_pairs)

    # ── Smoke-test mode ───────────────────────────────────────────────────
    if args.test:
        pair = next(iter(active_pairs))
        cfg = active_pairs[pair]
        instances = instances_by_pair.get(pair, [])
        if not instances:
            sys.exit(f"No instances found for {pair}")
        inst = instances[0]

        # Test with just the first available system
        hyps = inst["_raw"].get("hyps", {})
        if not hyps:
            sys.exit("No hypotheses found in first instance")
        test_system, hyp = next(iter(hyps.items()))
        src = inst["src_text"]
        domain = get_domain(inst["doc_id"])

        wrapper = LocalModelWrapper(args.model, thinking=args.thinking)

        prompt1 = build_stage1_prompt(src, hyp, cfg, domain)
        messages1 = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt1}]
        print("=" * 60)
        print(f"PAIR: {pair} | DOMAIN: {domain} | SYSTEM: {test_system}")
        print("STAGE 1 PROMPT:")
        print(prompt1)
        print("=" * 60)
        raw1, in_tok1, out_tok1 = wrapper.generate(messages1, max_new_tokens_s1)
        print("STAGE 1 RESPONSE:")
        print(raw1)
        print("=" * 60)
        stage1_text = parse_stage1_output(raw1)
        parsed = stage1_to_predicted_errors(stage1_text, hyp)

        prompt2 = build_stage2_prompt(src, hyp, stage1_text, cfg, domain)
        messages2 = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt2}]
        print("STAGE 2 PROMPT:")
        print(prompt2)
        print("=" * 60)
        raw2, in_tok2, out_tok2 = wrapper.generate(messages2, MAX_NEW_TOKENS_STAGE2)
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

    wrapper = LocalModelWrapper(args.model, thinking=args.thinking)

    for pair, cfg in active_pairs.items():
        output_path = output_dir / f"pred_{model_tag}_{pair}.jsonl"
        instances = instances_by_pair.get(pair, [])

        done_ids = load_done_ids(output_path) if args.resume else set()
        todo = [inst for inst in instances if inst["doc_id"] not in done_ids]
        logging.info("[%s] %d/%d segments to process → %s",
                     pair, len(todo), len(instances), output_path.name)

        pair_start = time.monotonic()
        for seg_i, inst in enumerate(todo):
            src = inst["src_text"]
            domain = get_domain(inst["doc_id"])
            systems = [(sys, hyp) for sys, hyp in inst["_raw"].get("hyps", {}).items() if hyp]
            task1_results = {}
            task2_results = {}

            # Process all systems for this segment in sub-batches.
            for b_start in range(0, len(systems), batch_size):
                sub = systems[b_start:b_start + batch_size]

                # Stage 1: error annotation
                s1_msgs = [
                    [{"role": "system", "content": SYSTEM_PROMPT},
                     {"role": "user", "content": build_stage1_prompt(src, hyp, cfg, domain)}]
                    for _, hyp in sub
                ]
                try:
                    s1_outputs = wrapper.generate_batch(s1_msgs, max_new_tokens_s1)
                except Exception as e:
                    if "CUDA error" in str(e):
                        raise  # CUDA context is broken; fail fast so --resume can recover cleanly
                    logging.error("[%s] %s stage1 batch failed: %s", pair, inst["doc_id"], e)
                    for system, _ in sub:
                        task1_results[system] = {"errors": [], "omission": None, "instruction_fault": None}
                        task2_results[system] = None
                    continue

                s1_texts = [parse_stage1_output(r[0]) for r in s1_outputs]

                # Stage 2: scoring (depends on stage 1 output)
                s2_msgs = [
                    [{"role": "system", "content": SYSTEM_PROMPT},
                     {"role": "user", "content": build_stage2_prompt(src, hyp, s1_text, cfg, domain)}]
                    for (_, hyp), s1_text in zip(sub, s1_texts)
                ]
                try:
                    s2_outputs = wrapper.generate_batch(s2_msgs, MAX_NEW_TOKENS_STAGE2)
                except Exception as e:
                    logging.error("[%s] %s stage2 batch failed: %s", pair, inst["doc_id"], e)
                    s2_outputs = [("", 0, 0)] * len(sub)

                for (system, hyp), s1_text, s2_res in zip(sub, s1_texts, s2_outputs):
                    parsed = stage1_to_predicted_errors(s1_text, hyp, log_ctx=f"[{inst['doc_id']} | {system}]")
                    score = parse_stage2_output(s2_res[0])
                    task1_results[system] = {
                        "errors": parsed["errors"],
                        "omission": parsed["omission"],
                        "instruction_fault": parsed["instruction_fault"],
                    }
                    task2_results[system] = score
                    logging.debug("[%s] %s | %s: %d errors, score=%s",
                                  pair, inst["doc_id"], system, len(parsed["errors"]), score)

            # Checkpoint: append this segment immediately so progress survives cancellation
            append_row(make_row(inst, task1_results, task2_results), output_path)
            if (seg_i + 1) % 10 == 0:
                elapsed = int(time.monotonic() - pair_start)
                logging.info("[%s] %d/%d segments done | elapsed %dh%02dm%02ds",
                             pair, seg_i + 1, len(todo),
                             elapsed // 3600, (elapsed % 3600) // 60, elapsed % 60)

        logging.info("[%s] complete → %s", pair, output_path.name)

    logging.info("Done. Output dir: %s", output_dir)


if __name__ == "__main__":
    main()
