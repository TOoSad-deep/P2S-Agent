"""Minimal E2E batch harness for P2S-Agent.

Usage:
  cd backend && python tests/e2e/run_batch.py                # all samples
  cd backend && python tests/e2e/run_batch.py circle box     # named samples

Prereq: backend running (./start.sh start), samples in tests/e2e/samples/*.png.
Writes a JSON report next to this file (report_latest.json).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import httpx

BASE_URL = "http://localhost:8001"
SAMPLES_DIR = Path(__file__).parent / "samples"
REPORT_PATH = Path(__file__).parent / "report_latest.json"
TIMEOUT_S = 600
POLL_INTERVAL_S = 2


def run_sample(path: Path) -> tuple[str, dict]:
    with path.open("rb") as f:
        resp = httpx.post(
            f"{BASE_URL}/png-shader/run",
            files={"image": (path.name, f, "image/png")},
            timeout=30,
        )
    resp.raise_for_status()
    run_id = resp.json()["run_id"]
    deadline = time.monotonic() + TIMEOUT_S
    while time.monotonic() < deadline:
        status = httpx.get(f"{BASE_URL}/png-shader/status/{run_id}", timeout=10).json()
        if status.get("status") in {"completed", "failed"}:
            return run_id, status
        time.sleep(POLL_INTERVAL_S)
    return run_id, {"status": "timeout"}


def main() -> None:
    names = sys.argv[1:]
    samples = (
        [SAMPLES_DIR / f"{n}.png" for n in names]
        if names
        else sorted(SAMPLES_DIR.glob("*.png"))
    )
    missing = [p for p in samples if not p.exists()]
    if missing or not samples:
        print(f"no usable samples in {SAMPLES_DIR} (missing: {[p.name for p in missing]})")
        return

    rows = []
    for path in samples:
        run_id, status = run_sample(path)
        score = (status.get("quality_router") or {}).get("final_score")
        rows.append({
            "sample": path.stem,
            "run_id": run_id,
            "status": status.get("status"),
            "final_score": score,
            "selected_source": (status.get("scoreboard") or {}).get("selected_source"),
        })
        print(f"{path.stem:28s} {str(status.get('status')):10s} score={score}")

    scored = [r["final_score"] for r in rows if isinstance(r["final_score"], (int, float))]
    n_pass = sum(1 for s in scored if s >= 0.85)
    n_acceptable = sum(1 for s in scored if 0.55 <= s < 0.85)
    summary = {
        "n": len(rows),
        "avg_final_score": round(sum(scored) / len(scored), 4) if scored else None,
        "pass": n_pass,
        "acceptable": n_acceptable,
        "fail": len(rows) - n_pass - n_acceptable,
        "rows": rows,
    }
    REPORT_PATH.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(
        f"\nn={summary['n']} avg={summary['avg_final_score']} "
        f"PASS(>=0.85)={n_pass} ACCEPTABLE(>=0.55)={n_acceptable} FAIL={summary['fail']}"
        f"\nreport -> {REPORT_PATH}"
    )


if __name__ == "__main__":
    main()
