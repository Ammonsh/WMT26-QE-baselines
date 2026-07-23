"""
WMT26 quality estimation — Gemini (Google AI Studio API).

LLM-as-judge QE using two-stage GEMBA-ESA prompting:
  Stage 1: Error annotation (domain-specific prompt → text annotations)
  Stage 2: Scoring (annotation + source/hyp → 0-100 score)
Output fields: stage1_annotations, predicted_errors (char spans), score.

This file is self-contained — no other project files are required.

Valid model IDs include: gemini-3-flash-preview, gemini-3-pro-preview,
gemini-2.5-flash, gemini-2.5-pro. Change MODEL_ID below as needed.

Setup:
  pip install -U google-genai
  export GEMINI_API_KEY="your_key_here"

Run:
  python run_qe.py --data-file mteval-test26.jsonl --test   # test on one segment, no file written
  python run_qe.py --data-file mteval-test26.jsonl          # full run, all language pairs
  python run_qe.py --data-file mteval-test26.jsonl --pair en-de    # single pair
  python run_qe.py --data-file mteval-test26.jsonl --resume        # resume an interrupted run
"""

import argparse
import concurrent.futures
import json
import logging
import os
import re
import sys
import time
from pathlib import Path
from threading import Lock

from google import genai
from google.genai import types


# ============================================================================
# CONFIG
# ============================================================================

MODEL_ID = "gemini-3.6-flash"
OUTPUT_NAME = MODEL_ID
OUTPUT_DIR = Path("quality_estimation_outputs_gemini")

THINKING_LEVEL = "medium"   # "low" / "medium" / "high" for Gemini 3.x; "none" to disable

MIN_INTERVAL_SEC = 0        # 0 = disabled (for enterprise/internal API access)
                            # set to 6.5 for free-tier (~10 req/min rate limit)
MAX_RETRIES = 6
MAX_BACKOFF_SEC = 120


# ============================================================================
# GEMBA-ESA PROMPTS  (embedded from qe_utils.py)
# ============================================================================

SYSTEM_PROMPT = "Your task is to identify machine translation errors and assess the quality of the translation."

# Domain requirement text inserted into Stage 1 and Stage 2 prompts.
# Keys match the domain component of item_id (4th _###_ field). Unknown domains fall back to "general".
DOMAIN_REQUIREMENTS = {
    "news": (
        "The source segment is from a news article. The translation should use a formal "
        "register consistent with journalistic standards and preserve the source HTML formatting."
    ),
    "factchecking": (
        "The source segment is from a news article. The translation should use a formal "
        "register consistent with journalistic standards and preserve the source HTML formatting."
    ),
    "speech": (
        "The source segment is a transcript of spoken content from a video. The translation "
        "should preserve the speaker's flow and colloquial style. It should omit non-linguistic "
        "sounds, such as laughter, groans, and hesitation sounds, while retaining interjections. "
        "Interrupted words should be completed when they can be inferred from context; otherwise, "
        "they should be omitted. Foreign words should remain unchanged. Each sentence should be "
        "placed on a separate line."
    ),
    "social": (
        "The source segment is user-generated content from a social media platform. Source "
        "spelling mistakes should not be reproduced. Meaningful expressiveness, such as "
        "capitalization or elongation, should be reproduced naturally in the target language. "
        "URLs and user handles should be copied unchanged, while hashtags should be translated "
        "when appropriate. Source punctuation should be followed as closely as possible, with "
        "additional punctuation only when needed to prevent serious loss of comprehension. The "
        "translation should use an informal style, like close friends talking, even if this "
        "changes the original tone, and preserve the source HTML formatting."
    ),
    "software": (
        "The source segment contains software data from a JSON. Only JSON content or values "
        "should be translated; keys and placeholders should be copied unchanged. The translation "
        "should contain only valid JSON content matching the input format."
    ),
    "edu": (
        "The source segment consists of biology, chemistry, and geography exercises from an "
        "educational web portal for children aged 9-16. The translation should be suitable for "
        "this educational context and age range, and preserve the source HTML formatting."
    ),
    "general": (
        "The translation should be accurate and fluent."
    ),
}

_STAGE1_ANNOTATION_BODY = (
    "Based on the source segment and machine translation surrounded by triple backticks, "
    "identify error types in the translation and classify them. The categories of errors are: "
    "accuracy (addition, mistranslation, omission, untranslated text), fluency (character "
    "encoding, grammar, inconsistency, punctuation, register, spelling), style (awkward), "
    "terminology (inappropriate for context, inconsistent use), non-translation, other, or "
    "no-error.\n\n\n"
    "Each error is classified as one of two categories: major or minor. Major errors disrupt "
    "the flow and make the understandability of the text difficult or impossible. Minor errors "
    "are errors that do not disrupt the flow significantly, and what the text is trying to say "
    "is still understandable.\n\n\n"
    "Return only the annotations in this format:\n"
    "Major:\n"
    "category/subcategory - \"error span\"\n"
    "Minor:\n"
    "category/subcategory - \"error span\"\n\n\n"
    "Use one error per line and write no-error when a section is empty. Quote spans from the "
    "translation; for omissions, quote the omitted source span."
)

_STAGE2_SCORING_BODY = (
    "Given the translation from {src_name} to {tgt_name} and the annotated error spans, assign "
    "a score on a continuous scale from 0 to 100. The scale has the following reference points: "
    "0=\"No meaning preserved\", 33=\"Some meaning preserved\", 66=\"Most meaning preserved and "
    "few grammar mistakes\", up to 100=\"Perfect meaning and grammar\".\n\n\n"
    "Domain requirements: {domain_req}\n\n\n"
    "Score the following translation:\n"
    "{src_name} source:\n"
    "```{src_text}```\n"
    "{tgt_name} translation:\n"
    "```{hyp_text}```\n"
    "Annotated error spans:\n"
    "```{error_spans}```\n\n\n"
    "Respond with ONLY a valid JSON object and nothing else: {{\"score\": N}}\n"
    "where N is an integer from 0 to 100."
)

# Language pairs for the WMT26 QE task (23 pairs).
# Keys match data filenames (e.g. "en-de" -> en-de.jsonl).
# src_code/tgt_code are the FLORES-200 codes from item_id fields.
TARGET_PAIRS = {
    "cs-de":   {"src_name": "Czech",              "tgt_name": "German",              "src_code": "ces_Latn", "tgt_code": "deu_Latn"},
    "cs-uk":   {"src_name": "Czech",              "tgt_name": "Ukrainian",           "src_code": "ces_Latn", "tgt_code": "ukr_Cyrl"},
    "cs-vi":   {"src_name": "Czech",              "tgt_name": "Vietnamese",          "src_code": "ces_Latn", "tgt_code": "vie_Latn"},
    "en-areg": {"src_name": "English",            "tgt_name": "Egyptian Arabic",     "src_code": "eng_Latn", "tgt_code": "arz_Arab"},
    "en-be":   {"src_name": "English",            "tgt_name": "Belarusian",          "src_code": "eng_Latn", "tgt_code": "bel_Cyrl"},
    "en-cs":   {"src_name": "English",            "tgt_name": "Czech",               "src_code": "eng_Latn", "tgt_code": "ces_Latn"},
    "en-de":   {"src_name": "English",            "tgt_name": "German",              "src_code": "eng_Latn", "tgt_code": "deu_Latn"},
    "en-et":   {"src_name": "English",            "tgt_name": "Estonian",            "src_code": "eng_Latn", "tgt_code": "ekk_Latn"},
    "en-hy":   {"src_name": "English",            "tgt_name": "Armenian",            "src_code": "eng_Latn", "tgt_code": "hye_Armn"},
    "en-id":   {"src_name": "English",            "tgt_name": "Indonesian",          "src_code": "eng_Latn", "tgt_code": "ind_Latn"},
    "en-is":   {"src_name": "English",            "tgt_name": "Icelandic",           "src_code": "eng_Latn", "tgt_code": "isl_Latn"},
    "en-ja":   {"src_name": "English",            "tgt_name": "Japanese",            "src_code": "eng_Latn", "tgt_code": "jpn_Jpan"},
    "en-kk":   {"src_name": "English",            "tgt_name": "Kazakh",              "src_code": "eng_Latn", "tgt_code": "kaz_Cyrl"},
    "en-ko":   {"src_name": "English",            "tgt_name": "Korean",              "src_code": "eng_Latn", "tgt_code": "kor_Hang"},
    "en-lij":  {"src_name": "English",            "tgt_name": "Ligurian",            "src_code": "eng_Latn", "tgt_code": "lij_Latn"},
    "en-lld":  {"src_name": "English",            "tgt_name": "Ladin",               "src_code": "eng_Latn", "tgt_code": "lld_Latn"},
    "en-ru":   {"src_name": "English",            "tgt_name": "Russian",             "src_code": "eng_Latn", "tgt_code": "rus_Cyrl"},
    "en-se":   {"src_name": "English",            "tgt_name": "Northern Sámi",       "src_code": "eng_Latn", "tgt_code": "sme_Latn"},
    "en-th":   {"src_name": "English",            "tgt_name": "Thai",                "src_code": "eng_Latn", "tgt_code": "tha_Thai"},
    "en-uk":   {"src_name": "English",            "tgt_name": "Ukrainian",           "src_code": "eng_Latn", "tgt_code": "ukr_Cyrl"},
    "en-zhcn": {"src_name": "English",            "tgt_name": "Simplified Chinese",  "src_code": "eng_Latn", "tgt_code": "zho_Hans"},
    "en-zhtw": {"src_name": "English",            "tgt_name": "Traditional Chinese", "src_code": "eng_Latn", "tgt_code": "zho_Hant_TW"},
    "zhcn-ja": {"src_name": "Simplified Chinese", "tgt_name": "Japanese",            "src_code": "zho_Hans", "tgt_code": "jpn_Jpan"},
}

HYP_SYSTEM = "Gemini 3.1 Pro"  # reference system name; all hyps are evaluated in full runs
N_INSTANCES_PER_PAIR = None     # None = all segments (no cap)

# Challenge segments use short 2-letter language codes instead of FLORES-200 codes.
# Maps (short_src, short_tgt) -> TARGET_PAIRS key. Pairs with no official equivalent are omitted.
CHALLENGE_CODE_MAP = {
    ("cs", "de"):    "cs-de",
    ("cs", "uk"):    "cs-uk",
    ("en", "ar"):    "en-areg",
    ("en", "cs"):    "en-cs",
    ("en", "de"):    "en-de",
    ("en", "de_DE"): "en-de",
    ("en", "is"):    "en-is",
    ("en", "ja"):    "en-ja",
    ("en", "ja_JP"): "en-ja",
    ("en", "ko"):    "en-ko",
    ("en", "ru"):    "en-ru",
    ("en", "uk"):    "en-uk",
    ("en", "zh"):    "en-zhcn",
    ("en", "zh_CN"): "en-zhcn",
    ("zh", "ja"):    "zhcn-ja",
}


# ============================================================================
# DATA LOADING
# ============================================================================

def get_domain(item_id: str) -> str:
    """Extract domain from item_id (4th _###_ component). Falls back to 'news'."""
    parts = tuple(item_id.split("_###_"))
    domain = parts[3] if len(parts) > 3 else "news"
    if domain not in DOMAIN_REQUIREMENTS:
        logging.warning("Unknown domain %r in item_id %r — falling back to 'general'", domain, item_id)
        return "general"
    return domain


def load_instances(data_file, target_pairs=None, segment_type="all"):
    """Load QE instances from a single combined JSONL file.

    Each line contains all language pairs. The language pair is derived from
    the src_code and tgt_code fields embedded in item_id
    (format: {seg_type}_###_{src_code}_###_{tgt_code}_###_{domain}_###_...).
    segment_type: "official", "challenge", or "all" (default).
    Returns a dict mapping lang-pair key -> list of instance dicts, each with:
      doc_id, src_text, hyp_text, refA, _raw (original JSON dict)
    """
    target_pairs = target_pairs or TARGET_PAIRS
    code_to_pair = {(v["src_code"], v["tgt_code"]): k for k, v in target_pairs.items()}
    _warned_challenge_codes = set()

    buckets = {pair: [] for pair in target_pairs}
    with open(data_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            item_id = d["item_id"]
            parts = item_id.split("_###_")
            if len(parts) < 3:
                continue
            if segment_type != "all" and parts[0] != segment_type:
                continue
            seg_type = parts[0]
            codes = (parts[1], parts[2])
            if seg_type == "challenge":
                pair = CHALLENGE_CODE_MAP.get(codes)
                if pair is None and codes not in _warned_challenge_codes:
                    logging.warning("Challenge pair %s-%s has no TARGET_PAIRS entry — skipping.", *codes)
                    _warned_challenge_codes.add(codes)
            else:
                pair = code_to_pair.get(codes)
            if pair is None or pair not in buckets:
                continue
            if N_INSTANCES_PER_PAIR is not None and len(buckets[pair]) >= N_INSTANCES_PER_PAIR:
                continue
            buckets[pair].append({
                "doc_id": item_id,
                "src_text": d.get("src", ""),
                "hyp_text": d.get("hyps", {}).get(HYP_SYSTEM, ""),
                "refA": d.get("ref", {}).get("text"),
                "_raw": d,
            })
    return buckets


# ============================================================================
# PROMPT BUILDERS
# ============================================================================

def build_stage1_prompt(src_text: str, hyp_text: str, cfg: dict, domain: str) -> str:
    src_name = cfg["src_name"]
    tgt_name = cfg["tgt_name"]
    domain_req = DOMAIN_REQUIREMENTS[domain]
    return (
        f"{src_name} source:\n"
        f"```{src_text}```\n"
        f"{tgt_name} translation:\n"
        f"```{hyp_text}```\n\n\n"
        f"{_STAGE1_ANNOTATION_BODY}\n\n\n"
        f"Domain requirements: {domain_req}"
    )


def build_stage2_prompt(src_text, hyp_text, stage1_output, cfg, domain):
    return _STAGE2_SCORING_BODY.format(
        src_name=cfg["src_name"],
        tgt_name=cfg["tgt_name"],
        domain_req=DOMAIN_REQUIREMENTS[domain],
        src_text=src_text,
        hyp_text=hyp_text,
        error_spans=stage1_output,
    )


# ============================================================================
# OUTPUT PARSING
# ============================================================================

def _strip_thinking(text: str) -> str:
    """Remove <think>…</think> blocks."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def parse_stage1_output(text: str) -> str:
    return _strip_thinking(text) if text else ""


_SPAN_RE = re.compile(r'^(\S[^"]*?)\s*-\s*"([^"]*)"')


def _find_span(span: str, hyp_text: str) -> tuple:
    idx = hyp_text.find(span)
    if idx != -1:
        return idx, idx + len(span)
    stripped = span.rstrip("\\")
    if stripped and stripped != span:
        idx = hyp_text.find(stripped)
        if idx != -1:
            return idx, idx + len(stripped)
    span_ws = re.sub(r"\s+", " ", span).strip()
    hyp_ws = re.sub(r"\s+", " ", hyp_text)
    if span_ws:
        idx = hyp_ws.find(span_ws)
        if idx != -1:
            return idx, idx + len(span_ws)
    return -1, -1


def stage1_to_predicted_errors(stage1_text: str, hyp_text: str) -> dict:
    """Convert Stage 1 annotation text to the task1_pred format for one system.

    Returns:
      {
        "errors": [{"start": int, "end": int, "severity": str, "category": str}],
        "omission": None | "minor" | "major",
        "instruction_fault": None,
      }
    Indices are half-open [start, end) matching Python slice conventions.
    omission is derived from accuracy/omission annotations (the omitted text is
    not in the hypothesis so no span is recorded). instruction_fault is always
    null as the current prompt is not designed to detect it.
    """
    errors = []
    omission_severities = []
    current_severity = None

    for line in stage1_text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.lower() == "major:":
            current_severity = "major"
        elif line.lower() == "minor:":
            current_severity = "minor"
        elif current_severity and line.lower() != "no-error":
            m = _SPAN_RE.match(line)
            if m:
                category = m.group(1).strip()
                span = m.group(2)
                cat_lower = category.lower()
                if "omission" in cat_lower and cat_lower.startswith("accuracy"):
                    omission_severities.append(current_severity)
                else:
                    start, end = _find_span(span, hyp_text)
                    if start != -1:
                        errors.append({"start": start, "end": end,
                                       "severity": current_severity, "category": category})
                    else:
                        logging.warning("Span not found in hyp_text: %r | hyp: %r", span, hyp_text[:120])

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


def parse_stage2_output(text: str):
    if not text:
        return None
    text = _strip_thinking(text).strip()
    stripped = re.sub(r"^```[a-zA-Z]*\s*", "", text).strip()
    stripped = re.sub(r"\s*```\s*$", "", stripped).strip()
    try:
        obj = json.loads(stripped)
        val = float(obj["score"])
        if 0.0 <= val <= 100.0:
            return val
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        pass
    matches = re.findall(r"\b(\d+(?:\.\d+)?)\b", text)
    for m in reversed(matches):
        try:
            val = float(m)
            if 0.0 <= val <= 100.0:
                return val
        except ValueError:
            pass
    logging.warning("Failed to parse score from Stage 2 output: %s", text[:200])
    return None


# ============================================================================
# OUTPUT HELPERS
# ============================================================================

def make_row(inst, task1_pred: dict, task2_pred: dict) -> dict:
    """Return the WMT26 QE submission row.

    task1_pred maps system_name -> {errors, omission, instruction_fault}.
    task2_pred maps system_name -> score (float or None).
    """
    return {
        "item_id": inst["doc_id"],
        "task1_pred": task1_pred,
        "task2_pred": task2_pred,
    }


def append_row(row: dict, path) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_done_ids(path) -> set:
    done = set()
    path = Path(path)
    if not path.exists():
        return done
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                t2 = obj.get("task2_pred", {})
                if any(v is not None for v in t2.values()):
                    done.add(obj.get("item_id"))
            except json.JSONDecodeError:
                pass
    return done


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
    msg = str(exc).lower()
    return is_rate_limit(exc) and ("perday" in msg or "per day" in msg or "daily" in msg)


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true",
                        help="Run one segment of one system (prints prompts+responses, no file written)")
    parser.add_argument("--pair", default=None,
                        help="Process only this language pair (e.g. en-de).")
    parser.add_argument("--data-file", required=True,
                        help="Path to the combined JSONL data file (e.g. mteval-test26.jsonl).")
    parser.add_argument("--segment-type", default="all", choices=["official", "challenge", "all"],
                        help="Which segments to evaluate: 'official', 'challenge', or 'all' (default).")
    parser.add_argument("--workers", type=int, default=4,
                        help="Number of parallel worker threads (default: 4).")
    parser.add_argument("--max-segments", type=int, default=None,
                        help="Cap segments per language pair; useful for testing.")
    parser.add_argument("--resume", action="store_true",
                        help="Skip segments already present in the output file.")
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
        Build a generation config with thinking enabled at THINKING_LEVEL.
        Gemini 3.x uses thinking_level; Gemini 2.5 uses thinking_budget.
        We try the most specific first and fall back so the script works
        across model families.
        """
        base = dict(
            system_instruction=SYSTEM_PROMPT,
            temperature=1.0,
            max_output_tokens=8192,
        )
        for tc in (
            lambda: types.ThinkingConfig(thinking_level=THINKING_LEVEL),  # Gemini 3.x
            lambda: types.ThinkingConfig(thinking_budget=1024),            # Gemini 2.5 fallback
            lambda: None,                                                   # no thinking config
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

    # Determine active pairs
    if args.pair is not None:
        if args.pair not in TARGET_PAIRS:
            sys.exit(f"Unknown pair {args.pair!r}. Valid pairs: {list(TARGET_PAIRS)}")
        active_pairs = {args.pair: TARGET_PAIRS[args.pair]}
    else:
        active_pairs = TARGET_PAIRS

    instances_by_pair = load_instances(data_file=args.data_file, target_pairs=active_pairs,
                                       segment_type=args.segment_type)

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

        prompt1 = build_stage1_prompt(src, hyp, cfg, domain)
        print("=" * 60)
        print(f"PAIR: {pair} | DOMAIN: {domain} | SYSTEM: {test_system}")
        print("STAGE 1 PROMPT:")
        print(prompt1)
        print("=" * 60)
        raw1 = call_api_with_retries(prompt1)
        print("STAGE 1 RESPONSE:")
        print(raw1)
        print("=" * 60)
        stage1_text = parse_stage1_output(raw1)
        parsed = stage1_to_predicted_errors(stage1_text, hyp)

        prompt2 = build_stage2_prompt(src, hyp, stage1_text, cfg, domain)
        print("STAGE 2 PROMPT:")
        print(prompt2)
        print("=" * 60)
        raw2 = call_api_with_retries(prompt2)
        print("STAGE 2 RESPONSE:")
        print(raw2)
        print("=" * 60)
        score = parse_stage2_output(raw2)
        print("PARSED task1 result:")
        print(json.dumps(parsed, indent=2, ensure_ascii=False))
        print(f"SCORE: {score}")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    _rl_lock = Lock()
    _last_call = [0.0]

    def _rate_limited_call(prompt):
        if MIN_INTERVAL_SEC > 0:
            with _rl_lock:
                elapsed = time.time() - _last_call[0]
                wait = MIN_INTERVAL_SEC - elapsed
                if wait > 0:
                    time.sleep(wait)
                _last_call[0] = time.time()
        return call_api_with_retries(prompt)

    def process_segment(pair, cfg, inst, output_path, file_lock):
        src = inst["src_text"]
        domain = get_domain(inst["doc_id"])
        hyps = inst["_raw"].get("hyps", {})
        task1_results = {}
        task2_results = {}

        for system, hyp in hyps.items():
            if not hyp:
                task1_results[system] = {"errors": [], "omission": "major", "instruction_fault": None}
                task2_results[system] = 0
                continue
            try:
                raw1 = _rate_limited_call(build_stage1_prompt(src, hyp, cfg, domain))
                stage1_text = parse_stage1_output(raw1)
                parsed = stage1_to_predicted_errors(stage1_text, hyp)
                raw2 = _rate_limited_call(build_stage2_prompt(src, hyp, stage1_text, cfg, domain))
                score = parse_stage2_output(raw2)
                task1_results[system] = {
                    "errors": parsed["errors"],
                    "omission": parsed["omission"],
                    "instruction_fault": parsed["instruction_fault"],
                }
                task2_results[system] = score
            except DailyQuotaExhausted:
                with file_lock:
                    append_row(make_row(inst, task1_results, task2_results), output_path)
                raise
            except Exception as e:
                logging.error(f"[{pair}] {inst['doc_id']} | {system} failed: {e}")
                task1_results[system] = {"errors": [], "omission": None, "instruction_fault": None}
                task2_results[system] = None

        with file_lock:
            append_row(make_row(inst, task1_results, task2_results), output_path)

    # Build work list across all pairs
    output_path = OUTPUT_DIR / f"pred_{OUTPUT_NAME}.jsonl"
    file_lock = Lock()
    done_ids = load_done_ids(output_path) if args.resume else set()

    work_items = []
    for pair, cfg in active_pairs.items():
        instances = instances_by_pair.get(pair, [])
        todo = [inst for inst in instances if inst["doc_id"] not in done_ids]
        if args.max_segments is not None:
            todo = todo[:args.max_segments]
        logging.info(f"[{pair}] {len(todo)}/{len(instances)} segments to process "
                     f"(2 API calls × N systems each) → {output_path.name}")
        for inst in todo:
            work_items.append((pair, cfg, inst, output_path, file_lock))

    n_total = len(work_items)
    n_done = 0
    logging.info(f"Processing {n_total} segments total with {args.workers} worker(s).")

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(process_segment, pair, cfg, inst, output_path, lock): (pair, inst["doc_id"])
            for pair, cfg, inst, output_path, lock in work_items
        }
        for future in concurrent.futures.as_completed(futures):
            pair, doc_id = futures[future]
            try:
                future.result()
            except DailyQuotaExhausted as e:
                logging.error(
                    f"DAILY QUOTA EXHAUSTED at [{pair}] {doc_id}. "
                    f"Partial segment saved. Resume with --resume tomorrow.\n  Detail: {e}"
                )
                executor.shutdown(wait=False, cancel_futures=True)
                sys.exit(1)
            except Exception as e:
                logging.error(f"[{pair}] {doc_id} unexpected error: {e}")
            n_done += 1
            if n_done % 10 == 0:
                logging.info(f"{n_done}/{n_total} segments done")

    logging.info("Done.")


if __name__ == "__main__":
    main()
