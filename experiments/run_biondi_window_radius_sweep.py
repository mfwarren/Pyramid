#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


CASES = [
    {"name": "narrow", "line_radius": 6, "line_step": 2, "sample_radius": 120, "sample_step": 24},
    {"name": "medium", "line_radius": 12, "line_step": 3, "sample_radius": 180, "sample_step": 30},
    {"name": "wide", "line_radius": 18, "line_step": 4, "sample_radius": 240, "sample_step": 40},
]


def run_case(case: dict, output_root: Path) -> dict:
    case_dir = output_root / case["name"]
    cmd = [
        sys.executable,
        str(REPO_ROOT / "experiments" / "run_biondi_global_window.py"),
        "--output-dir",
        str(case_dir),
        "--line-radius",
        str(case["line_radius"]),
        "--line-step",
        str(case["line_step"]),
        "--sample-radius",
        str(case["sample_radius"]),
        "--sample-step",
        str(case["sample_step"]),
    ]
    completed = subprocess.run(cmd, cwd=REPO_ROOT, check=True, capture_output=True, text=True)
    summary_path = case_dir / "summary.json"
    summary = json.loads(summary_path.read_text())
    return {
        **case,
        "summary_path": str(summary_path),
        "stdout_tail": completed.stdout[-1000:],
        "depth_profile_peak_m": summary["depth_profile_peak_m"],
        "depth_profile_top5_m": summary["depth_profile_top5_m"],
        "edge_fraction": summary["edge_fraction"],
        "volume_mean": summary["volume_stats"]["mean"],
        "window_shape": summary["config"]["window_shape"],
    }


def main() -> None:
    output_root = Path("data/processed/biondi_window_radius_sweep")
    output_root.mkdir(parents=True, exist_ok=True)

    results = [run_case(case, output_root) for case in CASES]
    best = min(results, key=lambda item: (item["edge_fraction"], abs(item["depth_profile_peak_m"])))
    summary = {
        "cases": results,
        "best_case": best,
    }
    (output_root / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
