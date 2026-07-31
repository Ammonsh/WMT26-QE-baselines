"""
Merge shard JSONL files and verify completeness against source data.

Shard naming: pred_{model}_{pair}_s0of4.jsonl .. pred_{model}_{pair}_s3of4.jsonl
              (0-indexed: s0of4 through s3of4)
Unsharded (num-shards=1) runs write directly to pred_{model}_{pair}.jsonl.

All pairs are merged into a single output file in the same order as the
source data file (mteval-test26.jsonl by default):
  --segment-type official  → pred_{model}_official.jsonl
  --segment-type challenge → pred_{model}_challenge.jsonl
  --segment-type all       → pred_{model}.jsonl

Note: when THINKING=true the model tag includes a _thinking suffix
(e.g. qwen36_thinking), so pass --model qwen36_thinking accordingly.

Usage (from WMT26-QE-baselines/):
    python merge_shards.py --model qwen36_thinking --segment-type official
    python merge_shards.py --model qwen36_thinking --segment-type challenge
    python merge_shards.py --model gemma4_thinking --pair cs-de --segment-type official
    python merge_shards.py --dry-run                # check only, no merge
"""

import argparse
import json
from pathlib import Path

# Official test set — 21 pairs (en-ja and en-ko removed from final test set).
OFFICIAL_PAIRS = [
    "cs-de", "cs-uk", "cs-vi", "en-areg", "en-be", "en-cs", "en-de", "en-et", "en-hy",
    "en-id", "en-is", "en-kk", "en-lij", "en-lld", "en-ru",
    "en-se", "en-th", "en-uk", "en-zhcn", "en-zhtw", "zhcn-ja",
]

# Challenge sets — 12 pairs (includes en-ja and en-ko, absent from official).
CHALLENGE_PAIRS = [
    "cs-de", "cs-uk", "en-areg", "en-cs", "en-de", "en-is", "en-ja", "en-ko",
    "en-ru", "en-uk", "en-zhcn", "zhcn-ja",
]

SCRIPT_DIR = Path(__file__).parent
OUTPUT_DIR = SCRIPT_DIR / "quality_estimation_outputs_local"


def get_source_ids(data_file: Path, pairs: list[str], segment_type: str) -> list[str]:
    """Read item_ids from the combined source file in their original order.

    Filters by segment_type ('official', 'challenge', or 'all') and restricts
    to the requested pairs based on src/tgt codes embedded in the item_id.
    """
    from qe_utils import TARGET_PAIRS, CHALLENGE_CODE_MAP

    # Build set of pair keys we care about
    pair_set = set(pairs)

    # Reverse lookup: (src_code, tgt_code) -> pair key, for official segments
    code_to_pair = {}
    for key, cfg in TARGET_PAIRS.items():
        code_to_pair[(cfg["src_code"], cfg["tgt_code"])] = key

    ids = []
    with open(data_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item_id = json.loads(line)["item_id"]
            parts = item_id.split("_###_")
            if len(parts) < 3:
                continue
            seg_type = parts[0]
            if segment_type != "all" and seg_type != segment_type:
                continue
            codes = (parts[1], parts[2])
            if seg_type == "challenge":
                pair_key = CHALLENGE_CODE_MAP.get(codes)
            else:
                pair_key = code_to_pair.get(codes)
            if pair_key in pair_set:
                ids.append(item_id)
    return ids


def read_rows(path: Path) -> list[dict]:
    """Read all valid JSON rows from a JSONL file."""
    rows = []
    if not path.exists():
        return rows
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return rows


def has_score(row: dict) -> bool:
    """True if the row has at least one non-null task2 score."""
    return any(v is not None for v in row.get("task2_pred", {}).values())


def dedup_rows(rows: list[dict]) -> dict[str, dict]:
    """Deduplicate rows by item_id.

    For each item_id, prefer a row with at least one non-null score.
    Among scored rows (or among unscored rows), keep the last one seen.
    """
    best: dict[str, dict] = {}
    for row in rows:
        iid = row.get("item_id")
        if iid is None:
            continue
        prev = best.get(iid)
        if prev is None or (not has_score(prev) and has_score(row)):
            best[iid] = row
        elif has_score(prev) and has_score(row):
            best[iid] = row  # keep latest scored entry
    return best


def collect_shard_rows(pairs: list[str], model: str, num_shards: int) -> tuple[dict, list[str]]:
    """Collect and deduplicate rows from all shard files across all pairs.

    Returns (deduped_dict, list_of_missing_shard_filenames).
    When num_shards <= 1 (unsharded run), only the base file is consulted.
    Per-pair base files are also included as a fallback for sharded runs.
    """
    all_rows: list[dict] = []
    missing_shards: list[str] = []

    for pair in pairs:
        base_file = OUTPUT_DIR / f"pred_{model}_{pair}_ref.jsonl"

        if num_shards > 1:
            pair_has_shards = False
            for i in range(num_shards):
                sf = OUTPUT_DIR / f"pred_{model}_{pair}_s{i}of{num_shards}.jsonl"
                if sf.exists():
                    all_rows.extend(read_rows(sf))
                    pair_has_shards = True
                else:
                    missing_shards.append(sf.name)

            # Include the per-pair base file if it exists (prior unsharded run or previous merge).
            # Shard rows take priority (added first), base rows fill gaps.
            base_rows = read_rows(base_file)
            if base_rows:
                all_rows.extend(base_rows)
        else:
            # Unsharded run — output went directly to the base file.
            base_rows = read_rows(base_file)
            if base_rows:
                all_rows.extend(base_rows)
            else:
                missing_shards.append(base_file.name)

    deduped = dedup_rows(all_rows)
    return deduped, missing_shards


def verify_and_merge(pairs: list[str], model: str, num_shards: int,
                     data_file: Path, segment_type: str, dry_run: bool) -> bool:
    """Collect all shard rows, verify completeness, and write a single merged output file."""
    deduped, missing_shards = collect_shard_rows(pairs, model, num_shards)

    expected_ids = get_source_ids(data_file, pairs, segment_type)
    expected_set = set(expected_ids)
    found_set = set(deduped.keys())

    missing = expected_set - found_set
    extra = found_set - expected_set

    if missing_shards:
        print(f"Missing shard files ({len(missing_shards)}):")
        for name in sorted(missing_shards):
            print(f"  {name}")

    print(f"Expected: {len(expected_ids)} segments across {len(pairs)} pair(s)")
    print(f"Found:    {len(found_set)} unique rows")
    if missing:
        print(f"MISSING:  {len(missing)}")
        for iid in sorted(missing)[:10]:
            print(f"  {iid}")
        if len(missing) > 10:
            print(f"  ... and {len(missing) - 10} more")
    if extra:
        print(f"EXTRA:    {len(extra)} (not in source file for selected segment type)")

    ok = not missing
    if not ok:
        if not dry_run:
            print("→ skipped merge (incomplete)")
        return False

    if dry_run:
        print("→ dry run complete, no files written")
        return True

    seg_tag = f"_{segment_type}" if segment_type != "all" else ""
    out_path = OUTPUT_DIR / f"pred_{model}{seg_tag}.jsonl"
    with open(out_path, "w", encoding="utf-8") as out:
        written = 0
        for iid in expected_ids:
            row = deduped.get(iid)
            if row:
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
                written += 1
    print(f"→ wrote {out_path.name} ({written} rows, source order)")
    return True


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="gemma4", help="Model tag (default: gemma4). "
                   "Include _thinking suffix if the run used --thinking (e.g. qwen36_thinking).")
    p.add_argument("--num-shards", type=int, default=4, help="Number of shards (default: 4). "
                   "Use 1 for unsharded (challenge) runs.")
    p.add_argument("--pair", default=None, help="Single pair to process (default: all)")
    p.add_argument("--data-file", default=None,
                   help="Path to combined source JSONL (default: mteval-test26.jsonl next to script)")
    p.add_argument("--segment-type", default="all", choices=["official", "challenge", "all"],
                   help="Which segments to expect in output (default: all)")
    p.add_argument("--dry-run", action="store_true", help="Check only, do not write merged file")
    args = p.parse_args()

    if args.pair:
        pairs = [args.pair]
    elif args.segment_type == "challenge":
        pairs = CHALLENGE_PAIRS
    elif args.segment_type == "official":
        pairs = OFFICIAL_PAIRS
    else:
        # "all": union of both lists, preserving order and deduplicating
        seen = set()
        pairs = []
        for p_ in OFFICIAL_PAIRS + CHALLENGE_PAIRS:
            if p_ not in seen:
                pairs.append(p_)
                seen.add(p_)

    data_file = Path(args.data_file) if args.data_file else SCRIPT_DIR / "mteval-test26.jsonl"

    mode = "DRY RUN — " if args.dry_run else ""
    print(f"{mode}model={args.model}, {len(pairs)} pair(s), segment-type={args.segment_type}\n")

    verify_and_merge(pairs, args.model, args.num_shards, data_file, args.segment_type, args.dry_run)


if __name__ == "__main__":
    main()
