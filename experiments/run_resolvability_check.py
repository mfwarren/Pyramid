#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from giza_backend import build_geometry, interpolate_target, prepare_giza_dataset
from sardt import (
    SubApertureConfig,
    build_steering_matrix,
    compute_kz,
    preprocess_measurement,
    solve_tomography,
)


def main() -> None:
    output_dir = Path("data/processed/hypothesis_loop")
    output_dir.mkdir(parents=True, exist_ok=True)

    z_grid = np.linspace(-3.0, 3.0, 121)
    dataset = prepare_giza_dataset(
        cfg=SubApertureConfig(width_bins=192, step_bins=96, n_apertures=9, taper="hann"),
        z_grid=z_grid,
    )

    local_line, local_sample = dataset.default_local_point
    global_line = dataset.line_bounds[0] + local_line
    global_pixel = dataset.pixel_bounds[0] + local_sample
    target = interpolate_target(dataset.meta, global_line, global_pixel)
    geom, debug = build_geometry(dataset.meta, target, dataset.chip.shape[1], dataset.cfg)
    A = build_steering_matrix(geom, dataset.z_grid)
    kz = compute_kz(geom)

    test_cases = [
        {"name": "single_0m", "depths": [0.0], "amps": [1.0 + 0.0j]},
        {"name": "single_2m", "depths": [2.0], "amps": [1.0 + 0.0j]},
        {"name": "double_pm2m", "depths": [-2.0, 2.0], "amps": [1.0 + 0.0j, 1.0 + 0.0j]},
        {"name": "double_pm1m", "depths": [-1.0, 1.0], "amps": [1.0 + 0.0j, 1.0 + 0.0j]},
    ]

    rows = []
    fig, axes = plt.subplots(len(test_cases), 2, figsize=(10, 3.2 * len(test_cases)))
    if len(test_cases) == 1:
        axes = np.array([axes])

    for row_idx, case in enumerate(test_cases):
        h_true = np.zeros(len(z_grid), dtype=np.complex128)
        for depth, amp in zip(case["depths"], case["amps"]):
            idx = int(np.argmin(np.abs(z_grid - depth)))
            h_true[idx] += amp
        y = A @ h_true

        reconstructions = {}
        for mode in ["pinv", "tikhonov", "omp"]:
            h = solve_tomography(y, A, mode=mode, alpha=1e-3, sparsity=3)
            reconstructions[mode] = np.abs(h)

        truth = np.abs(h_true)
        axes[row_idx, 0].plot(z_grid, truth / (truth.max() + 1e-12), label="truth", color="black", linewidth=2)
        for mode, color in [("pinv", "tab:blue"), ("tikhonov", "tab:orange"), ("omp", "tab:green")]:
            prof = reconstructions[mode]
            axes[row_idx, 0].plot(z_grid, prof / (prof.max() + 1e-12), label=mode, color=color, alpha=0.9)
        axes[row_idx, 0].set_title(case["name"])
        axes[row_idx, 0].set_xlabel("Depth z (m)")
        axes[row_idx, 0].set_ylabel("Normalized amplitude")
        axes[row_idx, 0].legend(fontsize=8)

        gram = np.abs(A.conj().T @ A)
        axes[row_idx, 1].imshow(
            gram,
            cmap="viridis",
            aspect="auto",
            origin="lower",
            extent=[float(z_grid[0]), float(z_grid[-1]), float(z_grid[0]), float(z_grid[-1])],
        )
        axes[row_idx, 1].set_title("Steering Gram |A^H A|")
        axes[row_idx, 1].set_xlabel("Depth z (m)")
        axes[row_idx, 1].set_ylabel("Depth z (m)")

        rows.append(
            {
                "case": case["name"],
                "truth_depths_m": case["depths"],
                "pinv_peak_depth_m": float(z_grid[np.argmax(reconstructions["pinv"])]),
                "tikhonov_peak_depth_m": float(z_grid[np.argmax(reconstructions["tikhonov"])]),
                "omp_peak_depth_m": float(z_grid[np.argmax(reconstructions["omp"])]),
            }
        )

    fig.tight_layout()
    fig.savefig(output_dir / "resolvability_check.png", dpi=180)
    plt.close(fig)

    summary = {
        "kz_span_rad_per_m": float(kz.max() - kz.min()),
        "vertical_resolution_m": float((2.0 * np.pi) / max(float(kz.max() - kz.min()), 1e-12)),
        "condition_number": float(np.linalg.cond(A)),
        "cases": rows,
        "geometry_debug": debug,
    }
    (output_dir / "resolvability_check.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
