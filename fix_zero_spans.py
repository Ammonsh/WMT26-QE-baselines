#!/usr/bin/env python3
"""Remove zero-length spans (start == end) from pred_gemma4 JSONL outputs.

Usage:
    python fix_zero_spans.py [file1.jsonl file2.jsonl ...]

If no files are given, processes all pred_gemma4_*.jsonl files in the default
output directory.  Files are edited in-place; a dry-run count is printed first.
"""
import json
import sys
from pathlib import Path

OUTPUT_DIR = Path(__file__).parent / "quality_estimation_outputs_local"


def count_and_fix(path: Path, dry_run: bool = False) -> tuple[int, int]:
    """Return (lines_changed, spans_removed). Writes in-place unless dry_run."""
    lines_changed = 0
    spans_removed = 0
    out_lines = []

    with open(path) as f:
        for raw in f:
            row = json.loads(raw)
            changed = False
            for sys_pred in row.get("task1_pred", {}).values():
                errors = sys_pred.get("errors")
                if not errors:
                    continue
                before = len(errors)
                sys_pred["errors"] = [e for e in errors if e["start"] != e["end"]]
                removed = before - len(sys_pred["errors"])
                if removed:
                    spans_removed += removed
                    changed = True
            if changed:
                lines_changed += 1
            out_lines.append(json.dumps(row, ensure_ascii=False))

    if not dry_run:
        with open(path, "w") as f:
            f.write("\n".join(out_lines) + "\n")

    return lines_changed, spans_removed


def main():
    if len(sys.argv) > 1:
        files = [Path(p) for p in sys.argv[1:]]
    else:
        files = sorted(OUTPUT_DIR.glob("pred_gemma4_*.jsonl"))

    if not files:
        print("No files found.")
        return

    # Dry run first
    print("Dry run — counting zero-length spans:")
    total_lines = total_spans = 0
    for path in files:
        lc, sr = count_and_fix(path, dry_run=True)
        if sr:
            print(f"  {path.name}: {sr} zero-length span(s) across {lc} line(s)")
        total_lines += lc
        total_spans += sr

    if total_spans == 0:
        print("No zero-length spans found. Nothing to do.")
        return

    print(f"\nTotal: {total_spans} zero-length span(s) in {total_lines} line(s) across {len(files)} file(s).")
    answer = input("Fix in place? [y/N] ").strip().lower()
    if answer != "y":
        print("Aborted.")
        return

    for path in files:
        lc, sr = count_and_fix(path, dry_run=False)
        if sr:
            print(f"  Fixed {path.name}: removed {sr} span(s)")
    print("Done.")


if __name__ == "__main__":
    main()
