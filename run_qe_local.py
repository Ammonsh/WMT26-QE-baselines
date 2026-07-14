"""
WMT26 quality estimation — local HuggingFace models (Gemma-4, Qwen3.6).

Supports:
  google/gemma-4-31B-it     (--model gemma4)
  Qwen/Qwen3.6-35B-A3B      (--model qwen36)

Both run at segment level with optional thinking mode (--thinking).
Uses the same prompt, data loading, and output format as run_qe.py (Gemini).

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
from pathlib import Path

# Compute nodes have no internet — default to offline mode so transformers
# loads from cache without network retries. Override by setting
# HF_HUB_OFFLINE=0 in your environment before launching this script.
os.environ.setdefault("HF_HUB_OFFLINE", "1")

from qe_utils import (
    HUMEVAL_FILE,
    TARGET_PAIRS,
    load_instances,
    build_qe_prompt,
    parse_qe_output,
    make_row,
    save_jsonl,
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
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype="auto",
            device_map="auto",
            low_cpu_mem_usage=True,
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

    def generate(
        self,
        messages: list[dict],
        max_new_tokens: int,
    ) -> tuple[str, int, int]:
        """Run a single inference call.

        Parameters
        ----------
        messages:
            Chat-format messages list (role/content dicts).
        max_new_tokens:
            Maximum tokens to generate.

        Returns
        -------
        response_text, input_tokens, output_tokens
        """
        import torch

        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=self.thinking,
        )

        inputs = self.processor(text=text, return_tensors="pt").to(self.model.device)
        input_tokens = inputs["input_ids"].shape[-1]

        gen_kwargs: dict = {"max_new_tokens": max_new_tokens,
                            "pad_token_id": self._tokenizer.eos_token_id}
        if self.thinking:
            gen_kwargs.update({"do_sample": True, "temperature": 0.6, "top_p": 0.95})
        else:
            gen_kwargs.update({"do_sample": False, "temperature": None, "top_p": None})

        with torch.no_grad():
            output_ids = self.model.generate(**inputs, **gen_kwargs)

        new_ids = output_ids[0, input_tokens:]
        output_tokens = len(new_ids)

        if self.model_type == "gemma4":
            # Decode keeping special tokens so parse_response() can strip thinking blocks.
            # parse_response() returns {'role': 'assistant', 'content': '...'}.
            raw = self.processor.decode(new_ids, skip_special_tokens=False)
            parsed = self.processor.parse_response(raw)
            response = parsed.get("content", "") if isinstance(parsed, dict) else str(parsed)
        else:
            # Qwen3.6: <think>…</think> blocks are stripped by parse_qe_output().
            response = self.processor.decode(new_ids, skip_special_tokens=True)

        return response, input_tokens, output_tokens


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
        help=f"Max tokens to generate (default {DEFAULT_MAX_NEW_TOKENS} without thinking, "
             f"{DEFAULT_MAX_NEW_TOKENS_THINKING} with --thinking).",
    )
    p.add_argument(
        "--output-dir", default="quality_estimation_outputs_local",
        help="Output directory (default: quality_estimation_outputs_local).",
    )
    p.add_argument(
        "--test", action="store_true",
        help="Run on one segment only — prints prompt + response, no file written.",
    )
    p.add_argument(
        "--resume", action="store_true",
        help="Skip doc_ids already present in the output file.",
    )
    return p.parse_args()


def _load_done_ids(output_path: Path) -> set:
    """Return set of doc_ids already written to output_path."""
    done = set()
    if not output_path.exists():
        return done
    with open(output_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
                done.add(r.get("doc_id"))
            except json.JSONDecodeError:
                pass
    return done


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

    max_new_tokens = args.max_new_tokens
    if max_new_tokens is None:
        max_new_tokens = DEFAULT_MAX_NEW_TOKENS_THINKING if args.thinking else DEFAULT_MAX_NEW_TOKENS

    instances_by_pair = load_instances()

    # ── Smoke-test mode ───────────────────────────────────────────────────
    if args.test:
        pair = next(iter(TARGET_PAIRS))
        cfg = TARGET_PAIRS[pair]
        instances = instances_by_pair.get(pair, [])
        if not instances:
            sys.exit(f"No instances found for {pair}")
        inst = instances[0]

        wrapper = LocalModelWrapper(args.model, thinking=args.thinking)

        prompt = build_qe_prompt(inst["src_text"], inst["hyp_text"], cfg)
        messages = [{"role": "user", "content": prompt}]
        print("=" * 60)
        print("PROMPT:")
        print(prompt)
        print("=" * 60)
        raw, in_tok, out_tok = wrapper.generate(messages, max_new_tokens)
        print("RAW RESPONSE:")
        print(raw)
        print("=" * 60)
        errors = parse_qe_output(raw)
        print(f"PARSED predicted_errors ({in_tok} input tokens, {out_tok} output tokens):")
        print(json.dumps(errors, indent=2, ensure_ascii=False))
        return

    # ── Full run ──────────────────────────────────────────────────────────
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    thinking_tag = "_thinking" if args.thinking else ""
    output_path = output_dir / f"pred_{args.model}{thinking_tag}_{HUMEVAL_FILE.name}"

    done_ids: set = set()
    if args.resume:
        done_ids = _load_done_ids(output_path)
        logging.info("Resuming: %d doc_ids already done", len(done_ids))

    # Load model only after verifying there's work to do
    total_todo = sum(
        1 for insts in instances_by_pair.values()
        for inst in insts
        if inst["doc_id"] not in done_ids
    )
    if total_todo == 0:
        logging.info("Nothing to do — all records already processed.")
        return
    logging.info("Segments to process: %d", total_todo)

    wrapper = LocalModelWrapper(args.model, thinking=args.thinking)

    rows: list = []
    mode = "a" if args.resume else "w"

    for pair, cfg in TARGET_PAIRS.items():
        instances = instances_by_pair.get(pair, [])
        todo = [inst for inst in instances if inst["doc_id"] not in done_ids]
        logging.info("[%s] %d/%d segments to process", pair, len(todo), len(instances))
        for inst in todo:
            prompt = build_qe_prompt(inst["src_text"], inst["hyp_text"], cfg)
            messages = [{"role": "user", "content": prompt}]
            try:
                raw, in_tok, out_tok = wrapper.generate(messages, max_new_tokens)
                predicted_errors = parse_qe_output(raw)
                logging.debug("[%s] %s: %d errors (%d/%d tokens)",
                              pair, inst["doc_id"], len(predicted_errors), in_tok, out_tok)
            except Exception as e:
                logging.error("[%s] %s failed: %s", pair, inst["doc_id"], e)
                predicted_errors = []
            rows.append(make_row(inst, predicted_errors))

        # Checkpoint: write completed pair to file
        with open(output_path, mode, encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        mode = "a"  # switch to append after first write
        rows = []
        logging.info("[%s] persisted checkpoint → %s", pair, output_path)

    logging.info("Done. Output: %s", output_path)


if __name__ == "__main__":
    main()
