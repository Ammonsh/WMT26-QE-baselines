"""
WMT26 quality estimation — GPT (OpenAI Batch API).

Two-stage GEMBA-ESA QE using the OpenAI Batch API for high throughput
(50% cost savings vs. sync API, no per-minute rate limits).

Since Stage 2 prompts depend on Stage 1 outputs, two sequential batches are
submitted. The script polls until each batch completes, then moves on.
All intermediate state is saved to disk so the run can be resumed safely.

Setup:
  pip install -U openai
  export OPENAI_API_KEY="your_key_here"

Run:
  python run_qe_openai_batch.py --data-file mteval-test26.jsonl           # full run
  python run_qe_openai_batch.py --data-file mteval-test26.jsonl --pair en-de
  python run_qe_openai_batch.py --data-file mteval-test26.jsonl --resume  # resume after interruption
  python run_qe_openai_batch.py --data-file mteval-test26.jsonl --max-segments 5 --pair en-de  # test

Intermediate files written to OUTPUT_DIR:
  batch_state_{MODEL}.json    — batch IDs and output file IDs (for resume)
  stage1_lookup_{MODEL}.json  — maps custom_id → item_id / system / hyp_text
  stage1_parsed_{MODEL}.json  — cached Stage 1 parse results
  stage2_lookup_{MODEL}.json  — maps custom_id → item_id / system
  pred_{MODEL}.jsonl          — final output (WMT26 submission format)
"""

import argparse
import json
import logging
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

from openai import OpenAI


# ============================================================================
# CONFIG
# ============================================================================

MODEL_ID = "gpt-5.6-terra"
OUTPUT_NAME = MODEL_ID
OUTPUT_DIR = Path("quality_estimation_outputs_openai")

REASONING_EFFORT = "medium"   # "low" / "medium" / "high"

POLL_INTERVAL_SEC = 60        # seconds between batch status polls
BATCH_CHUNK_SIZE  = 800       # max requests per batch chunk (~800 K input tokens, respects 900 K queue limit)


# ============================================================================
# GEMBA-ESA PROMPTS
# ============================================================================

SYSTEM_PROMPT = "Your task is to identify machine translation errors and assess the quality of the translation."

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

HYP_SYSTEM = "Gemini 3.1 Pro"
N_INSTANCES_PER_PAIR = None

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
    parts = tuple(item_id.split("_###_"))
    domain = parts[3] if len(parts) > 3 else "news"
    if domain not in DOMAIN_REQUIREMENTS:
        logging.warning("Unknown domain %r in item_id %r — falling back to 'general'", domain, item_id)
        return "general"
    return domain


def load_instances(data_file, target_pairs=None, segment_type="all"):
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

import re

def _strip_thinking(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def parse_stage1_output(text: str) -> str:
    return _strip_thinking(text) if text else ""


_SPAN_RE = re.compile(r'^(\S[^"]*?)\s*-\s*"([^"]*)"', re.DOTALL)


def _merge_multiline_spans(text: str) -> list:
    """Rejoin annotation lines whose quoted error span contains a literal newline."""
    result = []
    buf = ""
    for line in text.splitlines():
        if buf:
            buf += "\n" + line
            if buf.count('"') % 2 == 0:
                result.append(buf)
                buf = ""
        else:
            if line.count('"') % 2 == 1:
                buf = line
            else:
                result.append(line)
    if buf:
        result.append(buf)
    return result


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
    errors = []
    omission_severities = []
    current_severity = None

    for line in _merge_multiline_spans(stage1_text):
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
# BATCH HELPERS
# ============================================================================

def _batch_request(custom_id: str, prompt: str) -> dict:
    return {
        "custom_id": custom_id,
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": MODEL_ID,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "reasoning_effort": REASONING_EFFORT,
            "max_completion_tokens": 8192,
        },
    }


def _upload_and_create_batch(client, requests: list, description: str) -> tuple:
    """Upload requests as a batch input file and submit. Returns (file_id, batch_id)."""
    content = "\n".join(json.dumps(r, ensure_ascii=False) for r in requests).encode("utf-8")
    file_obj = client.files.create(
        file=("batch_input.jsonl", content, "application/jsonl"),
        purpose="batch",
    )
    batch = client.batches.create(
        input_file_id=file_obj.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={"description": description},
    )
    return file_obj.id, batch.id


def _upload_and_create_batches(client, requests: list, description: str,
                               on_chunk_submitted=None) -> list:
    """Split requests into BATCH_CHUNK_SIZE chunks and submit each as a separate batch.
    Returns list of (file_id, batch_id) tuples.
    on_chunk_submitted(results_so_far) is called after each chunk so callers can save state
    incrementally — if interrupted mid-submission, already-submitted batches won't be lost."""
    chunks = [requests[i:i + BATCH_CHUNK_SIZE] for i in range(0, len(requests), BATCH_CHUNK_SIZE)]
    results = []
    for i, chunk in enumerate(chunks):
        logging.info("  Submitting chunk %d/%d (%d requests)...", i + 1, len(chunks), len(chunk))
        file_id, batch_id = _upload_and_create_batch(
            client, chunk, f"{description} [{i + 1}/{len(chunks)}]"
        )
        results.append((file_id, batch_id))
        if on_chunk_submitted:
            on_chunk_submitted(results)
    return results


def _poll_all_until_complete(client, batch_ids: list, poll_interval: int,
                              done: dict = None, on_complete=None) -> dict:
    """Poll multiple batches until all complete.
    Returns dict of batch_id -> output_file_id.
    on_complete(dict) is called whenever a new batch finishes (for incremental state saves)."""
    output_file_ids = dict(done or {})
    pending = [bid for bid in batch_ids if bid not in output_file_ids]
    while pending:
        still_pending = []
        for batch_id in pending:
            batch = client.batches.retrieve(batch_id)
            status = batch.status
            counts = batch.request_counts
            logging.info(
                "Batch %s: %s (completed=%d, failed=%d, total=%d)",
                batch_id, status, counts.completed, counts.failed, counts.total,
            )
            if status == "completed":
                if batch.output_file_id is None:
                    raise RuntimeError(f"Batch {batch_id} completed but output_file_id is None")
                output_file_ids[batch_id] = batch.output_file_id
                if on_complete:
                    on_complete(output_file_ids)
            elif status in ("failed", "expired", "cancelled"):
                raise RuntimeError(f"Batch {batch_id} ended with status: {status}")
            else:
                still_pending.append(batch_id)
        pending = still_pending
        if pending:
            time.sleep(poll_interval)
    return output_file_ids


def _download_results(client, output_file_id: str) -> list:
    """Download batch output JSONL and return list of result dicts."""
    content = client.files.content(output_file_id).text
    return [json.loads(line) for line in content.splitlines() if line.strip()]


def _download_all_results(client, output_file_ids: dict) -> list:
    """Download and merge results from multiple batch output files."""
    results = []
    for file_id in output_file_ids.values():
        results.extend(_download_results(client, file_id))
    return results


def _extract_text(result: dict) -> str:
    """Extract message content from a batch result dict, or '' on error."""
    if result.get("error"):
        logging.warning("Batch request error for %s: %s", result.get("custom_id"), result["error"])
        return ""
    resp = result.get("response", {})
    if resp.get("status_code") != 200:
        logging.warning("Non-200 response for %s: %s", result.get("custom_id"), resp.get("status_code"))
        return ""
    try:
        return resp["body"]["choices"][0]["message"]["content"] or ""
    except (KeyError, IndexError, TypeError):
        return ""


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-file", required=True,
                        help="Path to the combined JSONL data file.")
    parser.add_argument("--pair", default=None,
                        help="Process only this language pair (e.g. en-de).")
    parser.add_argument("--pairs", nargs="+", default=None,
                        help="Process only these language pairs (e.g. cs-de en-de zhcn-ja).")
    parser.add_argument("--segment-type", default="all", choices=["official", "challenge", "all"])
    parser.add_argument("--max-segments", type=int, default=None,
                        help="Cap segments per language pair; useful for testing.")
    parser.add_argument("--poll-interval", type=int, default=POLL_INTERVAL_SEC,
                        help=f"Seconds between batch status polls (default: {POLL_INTERVAL_SEC}).")
    parser.add_argument("--resume", action="store_true",
                        help="Resume using batch IDs saved in the state file.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)

    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit("Set OPENAI_API_KEY in your environment.")
    client = OpenAI()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    state_path    = OUTPUT_DIR / f"batch_state_{OUTPUT_NAME}.json"
    s1_lookup_path = OUTPUT_DIR / f"stage1_lookup_{OUTPUT_NAME}.json"
    s1_parsed_path = OUTPUT_DIR / f"stage1_parsed_{OUTPUT_NAME}.json"
    s2_lookup_path = OUTPUT_DIR / f"stage2_lookup_{OUTPUT_NAME}.json"
    output_path   = OUTPUT_DIR / f"pred_{OUTPUT_NAME}.jsonl"

    # Load or initialize state
    if state_path.exists() and not args.resume:
        sys.exit(
            f"State file exists at {state_path}.\n"
            "Use --resume to continue, or delete the state file to start fresh."
        )
    state = json.loads(state_path.read_text()) if (args.resume and state_path.exists()) else {}

    def save_state():
        state_path.write_text(json.dumps(state, indent=2))

    # Determine active pairs
    if args.pair is not None and args.pairs is not None:
        sys.exit("Use --pair OR --pairs, not both.")
    if args.pair is not None:
        if args.pair not in TARGET_PAIRS:
            sys.exit(f"Unknown pair {args.pair!r}. Valid pairs: {list(TARGET_PAIRS)}")
        active_pairs = {args.pair: TARGET_PAIRS[args.pair]}
    elif args.pairs is not None:
        invalid = [p for p in args.pairs if p not in TARGET_PAIRS]
        if invalid:
            sys.exit(f"Unknown pairs: {invalid}. Valid pairs: {list(TARGET_PAIRS)}")
        active_pairs = {p: TARGET_PAIRS[p] for p in args.pairs}
    else:
        active_pairs = TARGET_PAIRS

    instances_by_pair = load_instances(data_file=args.data_file, target_pairs=active_pairs,
                                       segment_type=args.segment_type)

    # Build flat work list: one entry per (segment, system) pair
    work_items = []  # list of (pair, cfg, inst, system, hyp)
    for pair, cfg in active_pairs.items():
        instances = instances_by_pair.get(pair, [])
        if args.max_segments is not None:
            instances = instances[:args.max_segments]
        for inst in instances:
            for system, hyp in inst["_raw"].get("hyps", {}).items():
                work_items.append((pair, cfg, inst, system, hyp or ""))

    logging.info("Total work items (segments × systems): %d", len(work_items))

    # =========================================================================
    # STAGE 1 — Error annotation
    # =========================================================================

    if not state.get("stage1_submission_complete"):
        # Build full request list (needed to know how many chunks and which remain).
        s1_requests = []
        s1_lookup = {}  # custom_id → {item_id, system, hyp_text}
        for idx, (pair, cfg, inst, system, hyp) in enumerate(work_items):
            cid = f"s1_{idx}"
            s1_lookup[cid] = {"item_id": inst["doc_id"], "system": system, "hyp_text": hyp}
            if not hyp:
                continue  # empty hyp: skip API call, handle at output time
            domain = get_domain(inst["doc_id"])
            s1_requests.append(_batch_request(cid, build_stage1_prompt(inst["src_text"], hyp, cfg, domain)))

        if "stage1_batch_ids" not in state:
            # Fresh start — write lookup and initialise state lists
            s1_lookup_path.write_text(json.dumps(s1_lookup, ensure_ascii=False))
            state["stage1_file_ids"]        = []
            state["stage1_batch_ids"]       = []
            state["stage1_output_file_ids"] = {}

        n_chunks = -(-len(s1_requests) // BATCH_CHUNK_SIZE)  # ceil division
        logging.info("Stage 1: %d requests → %d chunks of ≤%d (sequential to respect queue limit)",
                     len(s1_requests), n_chunks, BATCH_CHUNK_SIZE)

        # Process one chunk at a time: submit → poll → next.
        # This keeps at most one chunk's tokens in the OpenAI batch queue at once.
        for i in range(n_chunks):
            chunk = s1_requests[i * BATCH_CHUNK_SIZE : (i + 1) * BATCH_CHUNK_SIZE]

            if i >= len(state["stage1_batch_ids"]):
                logging.info("  Submitting chunk %d/%d (%d requests)...", i + 1, n_chunks, len(chunk))
                file_id, batch_id = _upload_and_create_batch(
                    client, chunk, f"WMT26 QE Stage 1 — {OUTPUT_NAME} [{i + 1}/{n_chunks}]"
                )
                state["stage1_file_ids"].append(file_id)
                state["stage1_batch_ids"].append(batch_id)
                save_state()

            batch_id = state["stage1_batch_ids"][i]
            if batch_id not in state["stage1_output_file_ids"]:
                logging.info("  Polling chunk %d/%d...", i + 1, n_chunks)

                def _save_s1_progress(fids):
                    state["stage1_output_file_ids"] = fids
                    save_state()

                _poll_all_until_complete(
                    client, [batch_id], args.poll_interval,
                    done=state["stage1_output_file_ids"],
                    on_complete=_save_s1_progress,
                )

        state["stage1_submission_complete"] = True
        save_state()
        logging.info("Stage 1: all %d chunks complete", n_chunks)
    else:
        s1_lookup = json.loads(s1_lookup_path.read_text())
        logging.info("Stage 1 already complete: %d chunks", len(state["stage1_batch_ids"]))

    # Parse Stage 1 results and cache to disk
    if not s1_parsed_path.exists():
        logging.info("Downloading Stage 1 results...")
        s1_parsed = {}  # key: "item_id|||system" → {stage1_text, errors, omission}
        for result in _download_all_results(client, state["stage1_output_file_ids"]):
            cid = result["custom_id"]
            meta = s1_lookup.get(cid, {})
            item_id = meta.get("item_id", "")
            system  = meta.get("system", "")
            hyp     = meta.get("hyp_text", "")
            key = f"{item_id}|||{system}"
            raw = _extract_text(result)
            stage1_text = parse_stage1_output(raw)
            parsed = stage1_to_predicted_errors(stage1_text, hyp)
            s1_parsed[key] = {
                "stage1_text": stage1_text,
                "errors": parsed["errors"],
                "omission": parsed["omission"],
            }
        s1_parsed_path.write_text(json.dumps(s1_parsed, ensure_ascii=False))
        logging.info("Stage 1 parsed: %d results", len(s1_parsed))
    else:
        s1_parsed = json.loads(s1_parsed_path.read_text())
        logging.info("Loaded cached Stage 1 results: %d entries", len(s1_parsed))

    # =========================================================================
    # STAGE 2 — Scoring
    # =========================================================================

    if not state.get("stage2_submission_complete"):
        s2_requests = []
        s2_lookup = {}  # custom_id → {item_id, system}
        for idx, (pair, cfg, inst, system, hyp) in enumerate(work_items):
            cid = f"s2_{idx}"
            s2_lookup[cid] = {"item_id": inst["doc_id"], "system": system}
            if not hyp:
                continue
            key = f"{inst['doc_id']}|||{system}"
            stage1_text = s1_parsed.get(key, {}).get("stage1_text", "")
            domain = get_domain(inst["doc_id"])
            s2_requests.append(
                _batch_request(cid, build_stage2_prompt(inst["src_text"], hyp, stage1_text, cfg, domain))
            )

        if "stage2_batch_ids" not in state:
            s2_lookup_path.write_text(json.dumps(s2_lookup, ensure_ascii=False))
            state["stage2_file_ids"]        = []
            state["stage2_batch_ids"]       = []
            state["stage2_output_file_ids"] = {}

        n_chunks = -(-len(s2_requests) // BATCH_CHUNK_SIZE)
        logging.info("Stage 2: %d requests → %d chunks of ≤%d (sequential to respect queue limit)",
                     len(s2_requests), n_chunks, BATCH_CHUNK_SIZE)

        for i in range(n_chunks):
            chunk = s2_requests[i * BATCH_CHUNK_SIZE : (i + 1) * BATCH_CHUNK_SIZE]

            if i >= len(state["stage2_batch_ids"]):
                logging.info("  Submitting chunk %d/%d (%d requests)...", i + 1, n_chunks, len(chunk))
                file_id, batch_id = _upload_and_create_batch(
                    client, chunk, f"WMT26 QE Stage 2 — {OUTPUT_NAME} [{i + 1}/{n_chunks}]"
                )
                state["stage2_file_ids"].append(file_id)
                state["stage2_batch_ids"].append(batch_id)
                save_state()

            batch_id = state["stage2_batch_ids"][i]
            if batch_id not in state["stage2_output_file_ids"]:
                logging.info("  Polling chunk %d/%d...", i + 1, n_chunks)

                def _save_s2_progress(fids):
                    state["stage2_output_file_ids"] = fids
                    save_state()

                _poll_all_until_complete(
                    client, [batch_id], args.poll_interval,
                    done=state["stage2_output_file_ids"],
                    on_complete=_save_s2_progress,
                )

        state["stage2_submission_complete"] = True
        save_state()
        logging.info("Stage 2: all %d chunks complete", n_chunks)
    else:
        s2_lookup = json.loads(s2_lookup_path.read_text())
        logging.info("Stage 2 already complete: %d chunks", len(state["stage2_batch_ids"]))

    # =========================================================================
    # ASSEMBLE FINAL OUTPUT
    # =========================================================================

    logging.info("Downloading Stage 2 results and writing output...")
    s2_scores = {}  # key: "item_id|||system" → score
    for result in _download_all_results(client, state["stage2_output_file_ids"]):
        cid = result["custom_id"]
        meta = s2_lookup.get(cid, {})
        key = f"{meta.get('item_id', '')}|||{meta.get('system', '')}"
        s2_scores[key] = parse_stage2_output(_extract_text(result))

    # Group predictions by item_id preserving work_items order
    items_by_id = defaultdict(lambda: {"task1_pred": {}, "task2_pred": {}})
    for pair, cfg, inst, system, hyp in work_items:
        item_id = inst["doc_id"]
        key = f"{item_id}|||{system}"
        if not hyp:
            items_by_id[item_id]["task1_pred"][system] = {
                "errors": [], "omission": "major", "instruction_fault": None,
            }
            items_by_id[item_id]["task2_pred"][system] = 0
        else:
            s1 = s1_parsed.get(key, {})
            items_by_id[item_id]["task1_pred"][system] = {
                "errors": s1.get("errors", []),
                "omission": s1.get("omission"),
                "instruction_fault": None,
            }
            items_by_id[item_id]["task2_pred"][system] = s2_scores.get(key)

    with open(output_path, "w", encoding="utf-8") as f:
        for item_id, preds in items_by_id.items():
            row = {"item_id": item_id,
                   "task1_pred": preds["task1_pred"],
                   "task2_pred": preds["task2_pred"]}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    logging.info("Done. %d rows written to %s", len(items_by_id), output_path)


if __name__ == "__main__":
    main()
