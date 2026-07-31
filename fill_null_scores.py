"""
Fill null task2_pred scores in a QE prediction JSONL file.

For each null score, imputes the mean score for that (language-pair, MT-system)
computed from non-null entries in the same file.  Fallback chain:
  1. Per-(pair, system) mean   — preferred; uses all non-null scores for that system in that pair
  2. Per-system global mean    — used when a system has nulls across all pairs
  3. Dataset-wide mean         — last resort if a system has zero non-null scores anywhere

Language pair is inferred from the item_id (format: {seg_type}_###_{src}_###_{tgt}_###_...).

Usage:
  python fill_null_scores.py pred_gemma4_thinking_official_ref.jsonl
  python fill_null_scores.py pred_gemma4_thinking_official_ref.jsonl --output filled.jsonl
  python fill_null_scores.py pred_gemma4_thinking_official_ref.jsonl --in-place
"""

import argparse
import json
import logging
import statistics
from collections import defaultdict
from pathlib import Path


def get_pair_key(item_id: str) -> str:
    """Extract 'src_###_tgt' from item_id as a pair key."""
    parts = item_id.split("_###_")
    if len(parts) >= 3:
        return f"{parts[1]}_###_{parts[2]}"
    return "unknown"


def load_rows(path: Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                logging.warning("Skipping malformed line %d: %s", lineno, e)
    return rows


def compute_means(rows: list[dict]) -> tuple[dict, dict, float]:
    """Return (pair_system_mean, system_mean, global_mean).

    pair_system_mean[(pair_key, system)] = mean of non-null scores
    system_mean[system]                  = mean across all pairs
    global_mean                          = mean across everything
    """
    pair_sys_scores: dict[tuple, list[float]] = defaultdict(list)
    sys_scores: dict[str, list[float]] = defaultdict(list)
    all_scores: list[float] = []

    for row in rows:
        pair_key = get_pair_key(row.get("item_id", ""))
        for system, score in row.get("task2_pred", {}).items():
            if score is not None:
                v = float(score)
                pair_sys_scores[(pair_key, system)].append(v)
                sys_scores[system].append(v)
                all_scores.append(v)

    pair_system_mean = {k: statistics.mean(v) for k, v in pair_sys_scores.items()}
    system_mean = {k: statistics.mean(v) for k, v in sys_scores.items()}
    global_mean = statistics.mean(all_scores) if all_scores else 50.0

    return pair_system_mean, system_mean, global_mean


def fill_nulls(rows: list[dict], pair_system_mean: dict, system_mean: dict, global_mean: float) -> tuple[list[dict], int]:
    filled_rows = []
    n_filled = 0
    for row in rows:
        pair_key = get_pair_key(row.get("item_id", ""))
        t2 = row.get("task2_pred", {})
        new_t2 = {}
        for system, score in t2.items():
            if score is not None:
                new_t2[system] = score
            else:
                imputed = (
                    pair_system_mean.get((pair_key, system))
                    or system_mean.get(system)
                    or global_mean
                )
                new_t2[system] = round(imputed, 2)
                n_filled += 1
        filled_rows.append({**row, "task2_pred": new_t2})
    return filled_rows, n_filled


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")

    p = argparse.ArgumentParser(description="Fill null task2_pred scores with per-system-per-pair means.")
    p.add_argument("input", help="Input JSONL prediction file.")
    p.add_argument(
        "--output", default=None,
        help="Output path. Defaults to <input_stem>_filled.jsonl in the same directory.",
    )
    p.add_argument(
        "--in-place", action="store_true",
        help="Overwrite the input file (a .bak backup is written first).",
    )
    args = p.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise SystemExit(f"Input file not found: {input_path}")

    if args.in_place:
        output_path = input_path
        backup_path = input_path.with_suffix(".bak.jsonl")
        import shutil
        shutil.copy2(input_path, backup_path)
        logging.info("Backup written to %s", backup_path)
    elif args.output:
        output_path = Path(args.output)
    else:
        output_path = input_path.with_name(input_path.stem + "_filled.jsonl")

    rows = load_rows(input_path)
    logging.info("Loaded %d rows from %s", len(rows), input_path)

    pair_system_mean, system_mean, global_mean = compute_means(rows)
    logging.info(
        "Computed means for %d (pair, system) combinations across %d systems. Global mean: %.2f",
        len(pair_system_mean), len(system_mean), global_mean,
    )

    # Log per-system means for transparency
    for sys_name, mean in sorted(system_mean.items()):
        logging.info("  System %-40s global mean: %.2f", sys_name, mean)

    filled_rows, n_filled = fill_nulls(rows, pair_system_mean, system_mean, global_mean)
    logging.info("Filled %d null score(s).", n_filled)

    with open(output_path, "w", encoding="utf-8") as f:
        for row in filled_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    logging.info("Written to %s", output_path)


if __name__ == "__main__":
    main()
