"""Suggest quality-router band cutoffs from saved run artifacts.

Usage:  cd backend && python scripts/calibrate_thresholds.py [--results-dir test_results]

Collects selected-candidate final_score from every run's quality_router.json
and prints distribution percentiles to inform the 0.85/0.70/0.55/0.40 bands.
"""
import argparse
import json
import statistics
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="test_results")
    args = parser.parse_args()

    scores: list[float] = []
    for qr_path in sorted(Path(args.results_dir).glob("*/quality_router.json")):
        try:
            data = json.loads(qr_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if isinstance(data, dict) and "final_score" in data:
            scores.append(float(data["final_score"]))

    if len(scores) < 10:
        print(f"only {len(scores)} runs found in {args.results_dir!r} — need >= 10")
        return

    scores.sort()

    def pct(p: float) -> float:
        return scores[min(len(scores) - 1, int(p * len(scores)))]

    print(f"n={len(scores)}  mean={statistics.mean(scores):.4f}  median={pct(0.5):.4f}")
    print(f"p90 (suggest 'excellent' cut): {pct(0.90):.4f}")
    print(f"p70 (suggest 'good' cut):      {pct(0.70):.4f}")
    print(f"p40 (suggest 'acceptable' cut):{pct(0.40):.4f}")
    print(f"p15 (suggest 'poor' floor):    {pct(0.15):.4f}")


if __name__ == "__main__":
    main()
