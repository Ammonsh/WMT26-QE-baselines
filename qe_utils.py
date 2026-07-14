"""
Shared utilities for WMT26 QE baseline scripts.

Imported by run_qe.py (Gemini) and run_qe_local.py (Gemma-4, Qwen3.6) so
that both use identical data loading, prompts, and output formatting.
"""

import json
import logging
import re
from pathlib import Path


# ============================================================================
# CONFIG  (defaults; callers may override by passing args to functions)
# ============================================================================

HUMEVAL_FILE = Path("../wmt25-genmt-humeval.jsonl")
OUTPUT_DIR = Path("quality_estimation_outputs")

HYP_SYSTEM = "Claude-4"   # which tgt_text system to evaluate
N_INSTANCES_PER_PAIR = 10

# Language pairs supported by the WMT26 QE task.
# tgt_code uses BCP-47-ish codes; en-ar, en-ru, en-uk excluded.
TARGET_PAIRS = {
    "cs-uk": {"src_name": "Czech",   "tgt_name": "Ukrainian",          "src_code": "cs", "tgt_code": "uk"},
    "cs-de": {"src_name": "Czech",   "tgt_name": "German",             "src_code": "cs", "tgt_code": "de-DE"},
    "en-zh": {"src_name": "English", "tgt_name": "Simplified Chinese", "src_code": "en", "tgt_code": "zh-CN"},
    "en-cs": {"src_name": "English", "tgt_name": "Czech",              "src_code": "en", "tgt_code": "cs"},
    "en-et": {"src_name": "English", "tgt_name": "Estonian",           "src_code": "en", "tgt_code": "et"},
    "en-is": {"src_name": "English", "tgt_name": "Icelandic",          "src_code": "en", "tgt_code": "is"},
    "en-ja": {"src_name": "English", "tgt_name": "Japanese",           "src_code": "en", "tgt_code": "ja"},
}


# ============================================================================
# DATA LOADING
# ============================================================================

def extract_base_pair(doc_id):
    """'cs-de_DE_#_news_#_...' -> ('cs-de', 'de_DE')"""
    first_chunk = doc_id.split("_#_", 1)[0]
    if "_" in first_chunk:
        base, variant = first_chunk.split("_", 1)
        return base, base.split("-", 1)[1] + "_" + variant
    return first_chunk, first_chunk.split("-", 1)[1]


def load_instances(
    humeval_file=None,
    hyp_system=None,
    n_instances=None,
    target_pairs=None,
):
    """Load QE instances from the humeval JSONL, one instance per source segment.

    Returns a dict mapping lang-pair key -> list of instance dicts, each with:
      doc_id, src_text, hyp_text, tgt_variant, refA, _raw (original JSON dict)
    """
    humeval_file = Path(humeval_file) if humeval_file else HUMEVAL_FILE
    hyp_system = hyp_system or HYP_SYSTEM
    n_instances = n_instances if n_instances is not None else N_INSTANCES_PER_PAIR
    target_pairs = target_pairs or TARGET_PAIRS

    buckets = {pair: [] for pair in target_pairs}
    with open(humeval_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            base, tgt_variant = extract_base_pair(d.get("doc_id", ""))
            if base not in target_pairs or len(buckets[base]) >= n_instances:
                continue
            buckets[base].append({
                "doc_id": d["doc_id"],
                "src_text": d.get("src_text", ""),
                "hyp_text": d.get("tgt_text", {}).get(hyp_system, ""),
                "tgt_variant": tgt_variant,
                "refA": d.get("tgt_text", {}).get("refA"),
                "_raw": d,  # keep original for output
            })
    return buckets


# ============================================================================
# PROMPT
# ============================================================================

def build_qe_prompt(src_text, hyp_text, cfg):
    """QE prompt (no reference) based on the MetricX25 prompt, slightly modified."""
    src_name = cfg['src_name']
    tgt_name = cfg['tgt_name']
    return (
        f"You are an annotator for the quality of machine translation. Your task is to identify errors and assess the quality of the translation.\n"
        f"Based on the source segment and machine translation surrounded with triple backticks, identify error types in the translation and classify them. "
        f"The categories of errors are: accuracy (addition, mistranslation, omission, untranslated text), fluency (character encoding, grammar, inconsistency, punctuation, register, spelling), style (awkward), terminology (inappropriate for context, inconsistent use), non-translation, other, or no-error.\n"
        f"Each error is classified as one of two severities: major and minor. Major errors confuse meaning, misrepresent the source, or violate the message (e.g., incorrect information, confusing wording). Minor errors are imperfections or stylistic issues that do not impact the core message (e.g., awkward phrasing).\n\n"
        f"Your response must be a strict and valid JSON object parseable with json.loads() in Python. "
        f"It must contain a single key \"predicted_errors\" whose value is a list of objects, each with keys \"start_i\", \"end_i\", and \"severity\". "
        f"If there are no errors, return {{\"predicted_errors\": []}}.\n\n"
        f"{src_name} source:\n```{src_text}```\n"
        f"{tgt_name} translation:\n```{hyp_text}```\n"
    )


# ============================================================================
# OUTPUT PARSING
# ============================================================================

def parse_qe_output(text):
    """Parse the model's QE JSON response.

    Handles:
      - <think>...</think> blocks from Qwen3 / Gemma-4 thinking mode
      - Optional ```json ... ``` markdown fences
      - Returns the predicted_errors list, or [] on failure.
    """
    if not text:
        return []
    text = text.strip()
    # Strip thinking blocks (Qwen3.6 thinking mode; no-op for others)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    # Strip optional ```json ... ``` fences
    text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
    text = re.sub(r"\s*```\s*$", "", text)
    try:
        obj = json.loads(text)
        return obj.get("predicted_errors", [])
    except json.JSONDecodeError as e:
        logging.warning(f"Failed to parse QE JSON response: {e}\nRaw text: {text[:200]}")
        return []


# ============================================================================
# OUTPUT
# ============================================================================

def make_row(inst, predicted_errors):
    """Return the original JSONL dict with predicted_errors added."""
    row = dict(inst["_raw"])
    row["predicted_errors"] = predicted_errors
    return row


def save_jsonl(rows, path):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
