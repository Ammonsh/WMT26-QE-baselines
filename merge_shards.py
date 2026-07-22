"""
Merge shard JSONL files and verify completeness against source data.

Shard naming: pred_{model}_{pair}_s0of4.jsonl .. pred_{model}_{pair}_s3of4.jsonl
              (0-indexed: s0of4 through s3of4)

Usage (from WMT26-QE-baselines/):
    python merge_shards.py                          # gemma4, all pairs
    python merge_shards.py --model qwen36           # qwen36, all pairs
    python merge_shards.py --model gemma4 --pair cs-de  # single pair
    python merge_shards.py --dry-run                # check only, no merge
"""

import argparse
import json
from pathlib import Path

PAIRS = [
    "cs-de", "cs-uk", "cs-vi", "en-areg", "en-be", "en-cs", "en-de", "en-et", "en-hy",
    "en-id", "en-is", "en-ja", "en-kk", "en-ko", "en-lij", "en-lld", "en-ru",
    "en-se", "en-th", "en-uk", "en-zhcn", "en-zhtw", "zhcn-ja",
]

SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR / "../data"
OUTPUT_DIR = SCRIPT_DIR / "quality_estimation_outputs_local"


def get_source_ids(pair: str) -> list[str]:
    """Read item_ids from the source data file in their original order."""
    ids = []
    fpath = DATA_DIR / f"{pair}.jsonl"
    with open(fpath, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                ids.append(json.loads(line)["item_id"])
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


def merge_and_verify(pair: str, model: str, num_shards: int, dry_run: bool) -> bool:
    """Merge shards (+ any pre-existing base file) for one pair and verify completeness.

    Segments written to the base file by a prior unsharded run are included so
    they are not reported as missing.  Duplicate entries (e.g. from null-score
    rows that were re-processed) are collapsed, preferring scored rows.

    Returns True if all expected segments are accounted for.
    """
    shard_files = [
        OUTPUT_DIR / f"pred_{model}_{pair}_s{i}of{num_shards}.jsonl"
        for i in range(num_shards)
    ]
    base_file = OUTPUT_DIR / f"pred_{model}_{pair}.jsonl"

    missing_shards = [f for f in shard_files if not f.exists()]
    if missing_shards and not base_file.exists():
        print(f"  [{pair}] MISSING shard files:")
        for f in missing_shards:
            print(f"    {f.name}")
        return False

    # Collect all rows: shard files first, then base file (for resume-skipped segments)
    all_rows: list[dict] = []
    for sf in shard_files:
        all_rows.extend(read_rows(sf))
    base_rows = read_rows(base_file)

    # Deduplicate: shard rows take priority over base rows (shards are the new source of truth),
    # but base rows fill in segments that resume intentionally skipped.
    deduped = dedup_rows(all_rows)
    for row in base_rows:
        iid = row.get("item_id")
        if iid and iid not in deduped:
            deduped[iid] = row

    expected_ids = get_source_ids(pair)
    expected_set = set(expected_ids)
    found_set = set(deduped.keys())

    raw_count = len(all_rows) + len(base_rows)
    dupes = raw_count - len(deduped)
    missing = expected_set - found_set
    extra = found_set - expected_set

    ok = not missing and not extra
    status = "OK" if ok else "FAIL"
    print(f"  [{pair}] {status} — expected {len(expected_ids)}, unique found {len(found_set)}"
          + (f", missing {len(missing)}" if missing else "")
          + (f", extra {len(extra)}" if extra else "")
          + (f", duplicates collapsed {dupes}" if dupes else "")
          + (f" [{len(base_rows)} from base file]" if base_rows else ""))

    if missing:
        for iid in sorted(missing)[:5]:
            print(f"    MISSING: {iid}")
        if len(missing) > 5:
            print(f"    ... and {len(missing) - 5} more")

    if not dry_run and ok:
        merged_path = base_file
        with open(merged_path, "w", encoding="utf-8") as out:
            for iid in expected_ids:  # write in source order
                row = deduped.get(iid)
                if row:
                    out.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"    → wrote {merged_path.name} ({len(expected_ids)} rows, source order)")
    elif not dry_run and not ok:
        print(f"    → skipped merge (incomplete)")

    return ok


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="gemma4", help="Model tag (default: gemma4)")
    p.add_argument("--num-shards", type=int, default=4, help="Number of shards (default: 4)")
    p.add_argument("--pair", default=None, help="Single pair to process (default: all)")
    p.add_argument("--dry-run", action="store_true", help="Check only, do not write merged files")
    args = p.parse_args()

    pairs = [args.pair] if args.pair else PAIRS
    mode = "DRY RUN — " if args.dry_run else ""
    print(f"{mode}Merging {args.num_shards} shards for model={args.model}, {len(pairs)} pair(s)\n")

    results = {}
    for pair in pairs:
        results[pair] = merge_and_verify(pair, args.model, args.num_shards, args.dry_run)

    passed = sum(results.values())
    total = len(results)
    print(f"\n{'=' * 50}")
    print(f"Result: {passed}/{total} pairs complete")
    if passed < total:
        print("Incomplete pairs:")
        for pair, ok in results.items():
            if not ok:
                print(f"  {pair}")


if __name__ == "__main__":
    main()
