"""Step 03b: package the raw judge JSONL logs into a cleaned, compressed copy.

The VM run appends every API attempt to results/judgements/<task>__<judge>.jsonl, including
failed calls (403s, 429s) that were later retried. This script:

  1. reads each raw log;
  2. drops records that have an error (failed API calls);
  3. keeps only the LATEST valid record per (item_id, order) — the log is append-only, so
     later lines are the retries that succeeded;
  4. checks that every task x judge has the expected number of judgements
     (tamil_pairwise 4,004 = 2,002 pairs x 2 orders; tamil_direct 280; hindi_pairwise 2,000);
  5. writes results/judgements_clean/<task>__<judge>.jsonl.gz (UTF-8 JSONL, gzip);
  6. prints a count table and the total logged cost per judge.

Consistency re-runs (results/judgements/consistency__*.jsonl) are a separate check and are
not part of the full-run package.

Usage:  python scripts/03b_package_judgements.py
Reads:  results/judgements/*.jsonl
Writes: results/judgements_clean/<task>__<judge>.jsonl.gz

No API keys are read. No annotator IDs or justification texts are present in these logs.
"""
from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "results" / "judgements"
OUT = ROOT / "results" / "judgements_clean"

# The three full-run tasks and the number of (item, order) rows each must have, per judge.
# Pairwise tasks are judged in both answer orders, so the count is pairs x 2.
EXPECTED = {
    "tamil_pairwise": 4004,   # 2,002 pairs x 2 orders
    "tamil_direct": 280,      # 280 responses, no order swap
    "hindi_pairwise": 2000,   # 1,000 unique pairs x 2 orders
}
JUDGES = ("frontier", "low_cost", "sea_lion", "base_model")


def load_latest_valid(path: Path) -> tuple[list[dict], int, int]:
    """Return (kept records, n_raw_lines, n_error_lines).

    A record is valid when it has no error field (or error is null). Among valid records
    with the same (item_id, order), the last one in the file wins, because the runner
    appends retries to the end of the log.
    """
    latest: dict[tuple[str, str], dict] = {}
    n_raw = n_err = 0
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            n_raw += 1
            rec = json.loads(line)
            if rec.get("error"):
                n_err += 1
                continue
            key = (str(rec["item_id"]), str(rec["order"]))
            latest[key] = rec  # later line overwrites earlier one for the same key
    kept = [latest[k] for k in sorted(latest)]  # stable order for a byte-identical gzip next time
    return kept, n_raw, n_err


def write_gzip_jsonl(path: Path, records: list[dict]) -> None:
    """Write one JSON object per line, gzip-compressed, with a trailing newline."""
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for rec in records:
            handle.write(json.dumps(rec, ensure_ascii=False) + "\n")


def main() -> None:
    if not RAW.is_dir():
        sys.exit(f"{RAW} not found: copy the VM logs into results/judgements/ first")
    OUT.mkdir(parents=True, exist_ok=True)

    print(f"{'task':<16} {'judge':<11} {'raw':>6} {'errors':>7} {'kept':>6} {'expect':>7}  status")
    cost_by_judge: dict[str, float] = {j: 0.0 for j in JUDGES}
    kept_by_judge: dict[str, int] = {j: 0 for j in JUDGES}
    problems: list[str] = []

    for task, expected in EXPECTED.items():
        for judge in JUDGES:
            src = RAW / f"{task}__{judge}.jsonl"
            if not src.is_file():
                problems.append(f"missing {src.name}")
                print(f"{task:<16} {judge:<11} {'—':>6} {'—':>7} {'—':>6} {expected:>7}  MISSING")
                continue
            kept, n_raw, n_err = load_latest_valid(src)
            dest = OUT / f"{task}__{judge}.jsonl.gz"
            write_gzip_jsonl(dest, kept)
            cost = sum(float(r.get("cost_usd") or 0.0) for r in kept)
            cost_by_judge[judge] += cost
            kept_by_judge[judge] += len(kept)
            ok = len(kept) == expected
            if not ok:
                problems.append(f"{src.name}: kept {len(kept)}, expected {expected}")
            # Flag parse failures among kept records so a silent bad JSON cannot slip into step 04.
            n_bad = sum(1 for r in kept if r.get("parse_ok") is False)
            extra = f"  parse_fail={n_bad}" if n_bad else ""
            print(f"{task:<16} {judge:<11} {n_raw:>6} {n_err:>7} {len(kept):>6} {expected:>7}  "
                  f"{'OK' if ok else 'COUNT MISMATCH'}{extra}")

    print("\nCost per judge (sum of cost_usd on kept records):")
    print(f"{'judge':<11} {'judgements':>12} {'cost_usd':>10}")
    total_cost = 0.0
    total_n = 0
    for judge in JUDGES:
        print(f"{judge:<11} {kept_by_judge[judge]:>12,} ${cost_by_judge[judge]:>9.4f}")
        total_cost += cost_by_judge[judge]
        total_n += kept_by_judge[judge]
    print(f"{'TOTAL':<11} {total_n:>12,} ${total_cost:>9.4f}")
    print(f"\nWrote gzip JSONL under {OUT.relative_to(ROOT).as_posix()}/")

    if problems:
        sys.exit("COUNT CHECK FAILED:\n  " + "\n  ".join(problems))


if __name__ == "__main__":
    main()
