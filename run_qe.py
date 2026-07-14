"""
WMT26 quality estimation — Gemini (Google AI Studio API).  [STANDALONE — no other files needed]

LLM-as-judge QE: predicts error spans (start_i, end_i, severity) for MT hypotheses.
Valid model IDs include: gemini-3.5-flash, gemini-3-flash, gemini-3-pro,
gemini-2.5-flash, gemini-2.5-pro. Change MODEL_ID below as needed.
Setup: pip install -U google-genai ; export GEMINI_API_KEY="..."
Run:   python run_qe.py [--test]
"""

import argparse
import os
import re
import sys
import time
import logging
from pathlib import Path

from google import genai
from google.genai import types

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
# CONFIG
# ============================================================================

MODEL_ID = "gemini-3-flash-preview"
OUTPUT_NAME = MODEL_ID
OUTPUT_DIR = Path("quality_estimation_outputs_gemini")

MIN_INTERVAL_SEC = 6.5    # ~10 req/min free tier; set 0 to disable
MAX_RETRIES = 6
MAX_BACKOFF_SEC = 120


# ============================================================================
# RATE LIMITING / RETRY HELPERS
# ============================================================================

class DailyQuotaExhausted(Exception):
    """Raised when the per-day API quota is exhausted (won't recover by waiting)."""


def parse_retry_delay(exc):
    m = re.search(r"['\"]retryDelay['\"]\s*:\s*['\"](\d+)s['\"]", str(exc))
    return float(m.group(1)) if m else None


def is_rate_limit(exc):
    msg = str(exc).lower()
    return "429" in msg or "resource_exhausted" in msg or "quota" in msg


def is_daily_quota_exhausted(exc):
    """
    Distinguish a per-DAY quota exhaustion from a transient per-minute rate limit.
    Per-day errors mention 'PerDay' / 'per day' in the quota metric and won't
    recover by waiting a minute, so we should stop rather than burn retries.
    """
    msg = str(exc).lower()
    return is_rate_limit(exc) and ("perday" in msg or "per day" in msg
                                   or "daily" in msg)


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true",
                        help="Run on one segment only (prints prompt+response, no file written)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
    logging.getLogger("httpx").setLevel(logging.WARNING)

    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        sys.exit("Set GEMINI_API_KEY in your environment.")
    client = genai.Client(api_key=api_key)

    def _build_config():
        """
        Build a generation config that disables thinking. Gemini 2.5 uses
        thinking_budget=0; Gemini 3.x uses thinking_level='minimal'. We try the
        most specific first and fall back so the script works across families.
        """
        base = dict(temperature=0.0, max_output_tokens=4096)
        for tc in (
            lambda: types.ThinkingConfig(thinking_level="minimal"),
            lambda: types.ThinkingConfig(thinking_budget=0),
            lambda: None,
        ):
            try:
                cfg_obj = tc()
            except Exception:
                continue
            try:
                if cfg_obj is None:
                    return types.GenerateContentConfig(**base)
                return types.GenerateContentConfig(thinking_config=cfg_obj, **base)
            except Exception:
                continue
        return types.GenerateContentConfig(**base)

    _GEN_CONFIG = _build_config()

    def call_api(prompt):
        resp = client.models.generate_content(
            model=MODEL_ID, contents=prompt, config=_GEN_CONFIG,
        )
        return resp.text or ""

    def call_api_with_retries(prompt):
        for attempt in range(MAX_RETRIES):
            try:
                return call_api(prompt)
            except Exception as e:
                if is_daily_quota_exhausted(e):
                    raise DailyQuotaExhausted(str(e)) from e
                if is_rate_limit(e):
                    wait = parse_retry_delay(e) or 30.0 * (attempt + 1)
                    wait = min(wait + 1.0, MAX_BACKOFF_SEC)
                    logging.warning(f"Rate-limited (try {attempt+1}/{MAX_RETRIES}); sleeping {wait:.0f}s")
                else:
                    wait = min(2 ** attempt, MAX_BACKOFF_SEC)
                    logging.warning(f"Error (try {attempt+1}/{MAX_RETRIES}): {e}; sleeping {wait}s")
                time.sleep(wait)
        raise RuntimeError("All retries failed")

    instances_by_pair = load_instances()

    if args.test:
        # Single-segment smoke test: first instance of first pair, no file written.
        import json
        pair = next(iter(TARGET_PAIRS))
        cfg = TARGET_PAIRS[pair]
        instances = instances_by_pair.get(pair, [])
        if not instances:
            sys.exit(f"No instances found for {pair}")
        inst = instances[0]
        prompt = build_qe_prompt(inst["src_text"], inst["hyp_text"], cfg)
        print("=" * 60)
        print("PROMPT:")
        print(prompt)
        print("=" * 60)
        raw = call_api_with_retries(prompt)
        print("RAW RESPONSE:")
        print(raw)
        print("=" * 60)
        errors = parse_qe_output(raw)
        print("PARSED predicted_errors:")
        print(json.dumps(errors, indent=2, ensure_ascii=False))
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / f"pred_{OUTPUT_NAME}_{HUMEVAL_FILE.name}"

    rows = []
    last_call = 0.0
    for pair, cfg in TARGET_PAIRS.items():
        instances = instances_by_pair.get(pair, [])
        logging.info(f"[{pair}] running QE on {len(instances)} segments")
        for inst in instances:
            if MIN_INTERVAL_SEC > 0:
                elapsed = time.time() - last_call
                if elapsed < MIN_INTERVAL_SEC:
                    time.sleep(MIN_INTERVAL_SEC - elapsed)
            prompt = build_qe_prompt(inst["src_text"], inst["hyp_text"], cfg)
            try:
                raw = call_api_with_retries(prompt)
                predicted_errors = parse_qe_output(raw)
            except DailyQuotaExhausted as e:
                save_jsonl(rows, output_path)
                logging.error(
                    f"DAILY QUOTA EXHAUSTED at [{pair}] {inst['doc_id']}. "
                    f"Saved {len(rows)} completed rows. Stopping.\n"
                    f"  -> Resume tomorrow, or enable billing on the API key, or "
                    f"switch MODEL_ID to a model with a higher free-tier daily limit.\n"
                    f"  -> To resume without redoing finished pairs, comment them out "
                    f"in TARGET_PAIRS in qe_utils.py.\n  Detail: {e}"
                )
                sys.exit(1)
            except Exception as e:
                logging.error(f"[{pair}] {inst['doc_id']} failed: {e}")
                predicted_errors = []
            finally:
                last_call = time.time()
            rows.append(make_row(inst, predicted_errors))
        save_jsonl(rows, output_path)
        logging.info(f"[{pair}] persisted — {len(rows)} rows total")
    logging.info("Done.")


if __name__ == "__main__":
    main()
