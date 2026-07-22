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

DATA_DIR = Path("../data")
OUTPUT_DIR = Path("quality_estimation_outputs")

HYP_SYSTEM = "Gemini 3.1 Pro"  # which hyps system to evaluate
N_INSTANCES_PER_PAIR = None     # None = all segments (no cap)

# ============================================================================
# GEMBA-ESA PROMPTS
# ============================================================================

SYSTEM_PROMPT = "Your task is to identify machine translation errors and assess the quality of the translation."

# Domain requirement text inserted into Stage 1 and Stage 2 prompts.
# Keys match the domain component of item_id (3rd _###_ field).
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
# Keys match data filenames (e.g. "en-de" -> ../data/en-de.jsonl).
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


# ============================================================================
# DATA LOADING
# ============================================================================

def extract_base_pair(item_id):
    """Parse a new-format item_id into (src_code, tgt_code, domain, doc_id).

    item_id format: '{src}_###_{tgt}_###_{domain}_###_{doc_id}[_###_{seg_idx}]'
    Returns a tuple of the '_###_'-separated parts (variable length).
    """
    return tuple(item_id.split("_###_"))


def get_domain(item_id: str) -> str:
    """Extract domain from item_id (3rd _###_ component). Falls back to 'news'."""
    parts = extract_base_pair(item_id)
    domain = parts[2] if len(parts) > 2 else "news"
    if domain not in DOMAIN_REQUIREMENTS:
        logging.warning("Unknown domain %r in item_id %r — falling back to 'news'", domain, item_id)
        return "news"
    return domain


def load_instances(
    data_dir=None,
    hyp_system=None,
    n_instances=None,
    target_pairs=None,
):
    """Load QE instances from per-pair JSONL files under data_dir.

    Each file is named '{pair}.jsonl' (e.g. 'en-de.jsonl').
    New field mapping vs. old humeval format:
      item_id -> doc_id  |  src -> src_text  |  hyps[system] -> hyp_text
      ref.text -> refA

    Returns a dict mapping lang-pair key -> list of instance dicts, each with:
      doc_id, src_text, hyp_text, refA, _raw (original JSON dict)
    """
    data_dir = Path(data_dir) if data_dir else DATA_DIR
    hyp_system = hyp_system or HYP_SYSTEM
    n_instances = n_instances if n_instances is not None else N_INSTANCES_PER_PAIR
    target_pairs = target_pairs or TARGET_PAIRS

    buckets = {pair: [] for pair in target_pairs}
    for pair in target_pairs:
        fpath = data_dir / f"{pair}.jsonl"
        with open(fpath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if n_instances is not None and len(buckets[pair]) >= n_instances:
                    break
                d = json.loads(line)
                buckets[pair].append({
                    "doc_id": d["item_id"],
                    "src_text": d.get("src", ""),
                    "hyp_text": d.get("hyps", {}).get(hyp_system, ""),
                    "refA": d.get("ref", {}).get("text"),
                    "_raw": d,  # keep original for output
                })
    return buckets


# ============================================================================
# PROMPT BUILDERS (GEMBA-ESA two-stage)
# ============================================================================

def build_stage1_prompt(src_text: str, hyp_text: str, cfg: dict, domain: str) -> str:
    """Build the Stage 1 error annotation prompt for the given domain."""
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


def build_stage2_prompt(
    src_text: str,
    hyp_text: str,
    stage1_output: str,
    cfg: dict,
    domain: str,
) -> str:
    """Build the Stage 2 scoring prompt using Stage 1 annotation output."""
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
    """Remove <think>…</think> blocks (Qwen3.6 / Gemma-4 thinking mode)."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def parse_stage1_output(text: str) -> str:
    """Return clean Stage 1 annotation text (thinking blocks stripped)."""
    if not text:
        return ""
    return _strip_thinking(text)


# Matches annotation lines: `category/subcategory - "error span"`
_SPAN_RE = re.compile(r'^(\S[^"]*?)\s*-\s*"([^"]*)"')


def _find_span(span: str, hyp_text: str) -> tuple:
    """Find span in hyp_text with normalization fallbacks.

    Tries in order:
      1. Exact match.
      2. Strip trailing backslashes from span (model sometimes appends a stray \\).
      3. Collapse all whitespace runs to a single space in both span and hyp_text
         (handles newline vs space mismatches in multi-line segments).

    Returns (start, end) on success, or (-1, -1) if not found.
    """
    if not span:  # empty string would always "match" at index 0 in Python
        return -1, -1
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


def stage1_to_predicted_errors(stage1_text: str, hyp_text: str, log_ctx: str = "") -> dict:
    """Convert Stage 1 annotation text to the task1_pred format for one system.

    Returns:
      {
        "errors": [{"start": int, "end": int, "severity": str, "category": str}],
        "omission": None | "minor" | "major",
        "instruction_fault": None | "minor" | "major",
      }

    Omissions (accuracy/omission) have no span in the hypothesis so they are
    captured as the top-level "omission" field rather than in "errors".
    instruction_fault errors are captured in both "errors" (if a span is found)
    and the top-level "instruction_fault" field.
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
                    # Omitted text is not present in the hypothesis — no span to locate
                    omission_severities.append(current_severity)
                else:
                    start, end = _find_span(span, hyp_text)
                    if start != -1:
                        errors.append({"start": start, "end": end,
                                       "severity": current_severity, "category": category})
                    else:
                        prefix = f"{log_ctx} " if log_ctx else ""
                        logging.warning("%sSpan not found in hyp_text: %r | hyp: %r",
                                        prefix, span, hyp_text[:120])

    def _max_sev(sevs):
        if "major" in sevs:
            return "major"
        if "minor" in sevs:
            return "minor"
        return None

    return {
        "errors": errors,
        "omission": _max_sev(omission_severities),
        "instruction_fault": None,  # not detectable from current prompt; set to null
    }


def parse_stage2_output(text: str):
    """Extract the numeric score (0–100) from Stage 2 output.

    The prompt asks for {"score": N} JSON. Tries JSON parse first, then falls
    back to scanning for any number in [0, 100] (handles markdown-fenced JSON
    or models that add brief explanatory text before the JSON).
    Returns a float, or None if no valid score found.
    """
    if not text:
        return None
    text = _strip_thinking(text).strip()
    # Strip optional ```json ... ``` fences
    stripped = re.sub(r"^```[a-zA-Z]*\s*", "", text).strip()
    stripped = re.sub(r"\s*```\s*$", "", stripped).strip()
    # Primary: JSON parse
    try:
        obj = json.loads(stripped)
        val = float(obj["score"])
        if 0.0 <= val <= 100.0:
            return val
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        pass
    # Fallback: scan for last number in [0, 100] anywhere in the text
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
# OUTPUT
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
    """Append a single row to a JSONL file (creates file if needed)."""
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_done_rows(path) -> dict:
    """Return dict mapping item_id -> merged row from a JSONL output file.

    When an item_id appears multiple times (e.g. from a crash-then-resume cycle),
    rows are merged at the system level: for each system, the latest non-null
    task2 score wins and the corresponding task1 result is kept alongside it.
    """
    rows: dict = {}
    path = Path(path)
    if not path.exists():
        return rows
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                iid = obj.get("item_id")
                if iid is None:
                    continue
                if iid not in rows:
                    rows[iid] = {
                        "item_id": iid,
                        "task1_pred": dict(obj.get("task1_pred", {})),
                        "task2_pred": dict(obj.get("task2_pred", {})),
                    }
                else:
                    prev = rows[iid]
                    for sys, score in obj.get("task2_pred", {}).items():
                        if score is not None:
                            prev["task2_pred"][sys] = score
                            t1 = obj.get("task1_pred", {}).get(sys)
                            if t1 is not None:
                                prev["task1_pred"][sys] = t1
                        elif sys not in prev["task2_pred"]:
                            prev["task2_pred"][sys] = None
                            t1 = obj.get("task1_pred", {}).get(sys)
                            if t1 is not None:
                                prev["task1_pred"][sys] = t1
            except json.JSONDecodeError:
                pass
    return rows


def load_done_ids(path) -> set:
    """Return set of item_ids already successfully written to a JSONL output file.

    A segment is considered done only if at least one system has a non-null score
    in task2_pred. Segments where all scores are null (e.g. from a CUDA error
    mid-run) are excluded so that --resume will reprocess them.
    """
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
