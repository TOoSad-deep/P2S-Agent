"""Measure VLM judge vs human agreement on labeled render pairs.

CSV columns: reference,render_a,render_b,human   (human = A | B | tie)
Usage:  cd backend && python scripts/judge_agreement.py pairs.csv

Build pairs.csv by sampling candidate renders from backend/test_results/<run>/
candidates/ and labeling ~30 pairs by eye. Target agreement >= 85%; below
80% do NOT enable pairwise arbitration — iterate the judge prompt first.
"""
import csv
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.llm.vlm_judge import judge_pairwise


def main(csv_path: str) -> None:
    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    work_dir = Path(tempfile.mkdtemp(prefix="judge_cal_"))
    agree = total = 0
    for row in rows:
        verdict = judge_pairwise(
            row["reference"], row["render_a"], row["render_b"], work_dir=work_dir
        )
        if verdict is None:
            print(f"SKIP {row['render_a']}: judge call failed")
            continue
        human = row["human"].strip().upper()
        human = human if human in ("A", "B") else "tie"
        total += 1
        ok = verdict == human
        agree += ok
        print(f"{Path(row['render_a']).name} vs {Path(row['render_b']).name}: "
              f"judge={verdict} human={human} {'OK' if ok else 'MISS'}")
    if total:
        print(f"\nagreement: {agree}/{total} = {agree / total:.1%}  (target >= 85%)")
    else:
        print("no usable rows — check API config and CSV paths")


if __name__ == "__main__":
    main(sys.argv[1])
