"""
Analyze WARNING counts in Slurm .err logs, broken down by language pair
and warning type (Stage 1 span misses vs Stage 2 score parse failures).

For each task ID, only the most recent job's log is used (same logic as
check_logs.sh), so resubmitted tasks don't inflate counts.

Usage (from WMT26-QE-baselines/):
    python analyze_warnings.py                        # default log dir
    python analyze_warnings.py --log-dir slurm/logs  # explicit path
    python analyze_warnings.py --pattern qe_gemma_shard
    python analyze_warnings.py --top 10              # show top N pairs
"""

import argparse
import re
from collections import defaultdict
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
DEFAULT_LOG_DIR = SCRIPT_DIR / "slurm/logs"

STAGE1_PATTERN = re.compile(r"\[WARNING\]\s+(\[.+?\])\s+Span not found in hyp_text:\s+'(.+?)'")
STAGE2_PATTERN = re.compile(r"\[WARNING\].*Failed to parse score from Stage 2 output:\s*(.*)")
PAIR_PATTERN   = re.compile(r"\[INFO\]\s+\[([a-z]{2,6}-[a-z]{2,6})\]")


def latest_logs(log_dir: Path, pattern: str) -> dict[str, Path]:
    """Return {task_id: path} keeping only the highest job_id per task."""
    latest: dict[str, tuple[int, Path]] = {}
    for f in log_dir.glob(f"{pattern}_*.err"):
        base = f.stem  # e.g. qe_gemma_shard_12864501_38
        suffix = base[len(pattern) + 1:]  # "12864501_38"
        parts = suffix.split("_", 1)
        if len(parts) != 2:
            continue
        job_id_str, task_id = parts
        try:
            job_id = int(job_id_str)
        except ValueError:
            continue
        if task_id not in latest or job_id > latest[task_id][0]:
            latest[task_id] = (job_id, f)
    return {tid: path for tid, (_, path) in latest.items()}


def count_warnings(path: Path) -> tuple[str | None, int, int]:
    """Return (pair, stage1_count, stage2_count) for a single log file.

    Deduplication within a file:
    - Stage 1: keyed by (item_id_bracket, span_text) — exact same span failure
      for the same segment+system is counted only once even if a resumed run
      re-processes it.
    - Stage 2: keyed by first 120 chars of the model output — catches the common
      case of identical degenerate outputs (<pad> floods, repetition) being
      logged twice for the same system after a crash+resume.
    """
    pair = None
    seen_s1: set[tuple[str, str]] = set()
    seen_s2: set[str] = set()
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            if pair is None:
                m = PAIR_PATTERN.search(line)
                if m:
                    pair = m.group(1)
            m1 = STAGE1_PATTERN.search(line)
            if m1:
                key = (m1.group(1), m1.group(2))
                seen_s1.add(key)
                continue
            m2 = STAGE2_PATTERN.search(line)
            if m2:
                key = m2.group(1)[:120]
                seen_s2.add(key)
    return pair, len(seen_s1), len(seen_s2)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--log-dir", default=str(DEFAULT_LOG_DIR))
    p.add_argument("--pattern", default="qe_gemma_shard")
    p.add_argument("--top", type=int, default=0, help="Show only top N pairs (0 = all)")
    args = p.parse_args()

    log_dir = Path(args.log_dir)
    logs = latest_logs(log_dir, args.pattern)
    if not logs:
        print(f"No .err files found matching '{args.pattern}_*.err' in {log_dir}")
        return

    # Aggregate by pair
    pair_s1: dict[str, int] = defaultdict(int)
    pair_s2: dict[str, int] = defaultdict(int)
    pair_files: dict[str, int] = defaultdict(int)
    unknown_s1 = unknown_s2 = 0

    for task_id, path in sorted(logs.items()):
        pair, s1, s2 = count_warnings(path)
        if pair:
            pair_s1[pair] += s1
            pair_s2[pair] += s2
            pair_files[pair] += 1
        else:
            unknown_s1 += s1
            unknown_s2 += s2

    all_pairs = sorted(set(pair_s1) | set(pair_s2),
                       key=lambda x: -(pair_s1[x] + pair_s2[x]))
    if args.top:
        all_pairs = all_pairs[: args.top]

    total_s1 = sum(pair_s1.values())
    total_s2 = sum(pair_s2.values())

    print(f"Analyzed {len(logs)} log files ({args.pattern})\n")
    print(f"{'Pair':<12}  {'Files':>5}  {'Stage1':>7}  {'Stage2':>7}  {'Total':>7}")
    print("-" * 46)
    for pair in all_pairs:
        s1 = pair_s1[pair]
        s2 = pair_s2[pair]
        print(f"{pair:<12}  {pair_files[pair]:>5}  {s1:>7}  {s2:>7}  {s1+s2:>7}")
    print("-" * 46)
    print(f"{'TOTAL':<12}  {'':>5}  {total_s1:>7}  {total_s2:>7}  {total_s1+total_s2:>7}")
    if unknown_s1 or unknown_s2:
        print(f"\n(unknown pair: stage1={unknown_s1}, stage2={unknown_s2})")

    # Summary: worst offenders per stage
    if len(all_pairs) > 1:
        worst_s1 = max(all_pairs, key=lambda x: pair_s1[x])
        worst_s2 = max(all_pairs, key=lambda x: pair_s2[x])
        print(f"\nMost Stage 1 warnings: {worst_s1} ({pair_s1[worst_s1]})")
        print(f"Most Stage 2 warnings: {worst_s2} ({pair_s2[worst_s2]})")


if __name__ == "__main__":
    main()
