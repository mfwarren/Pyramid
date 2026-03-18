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

from biondi_core import (
    PixelTrackConfig,
    build_acoustic_steering_matrix,
    build_displacement_series,
    local_to_target,
    prepare_tracking_subapertures,
    solve_acoustic_tomography,
)
from giza_backend import TARGET_LAT, TARGET_LON, prepare_giza_dataset
from sardt import SubApertureConfig


def run_case(dataset, sub_slcs, line_points):
    track_cfg = PixelTrackConfig(patch_radius=8, search_radius=4)
    slice_matrix = np.empty((len(line_points), len(dataset.z_grid)), dtype=np.float64)
    tracking_stats = []
    for i, (line_f, sample_f) in enumerate(line_points):
        ll = int(np.clip(round(line_f), 16, dataset.chip_native.shape[0] - 17))
        ls = int(np.clip(round(sample_f), 16, dataset.chip_native.shape[1] - 17))
        target, _ = local_to_target(dataset, ll, ls)
        y, tracking_debug, obs_idx = build_displacement_series(
            sub_slcs,
            (ls, ll),
            track_cfg,
            observable="azimuth_phase",
            pair_stride=4,
            series_mode="cumulative",
            detrend_mode="linear",
        )
        A, steering_debug = build_acoustic_steering_matrix(
            dataset,
            target,
            sound_speed_mps=1500.0,
            vibration_hz=500.0,
            steering_mode="linear",
        )
        h = solve_acoustic_tomography(y, A[np.asarray(obs_idx)], inversion_mode="omp", sparsity=3)
        slice_matrix[i] = np.abs(h)
        tracking_stats.extend(tracking_debug)
    avg = slice_matrix.mean(axis=0)
    peaks = np.argsort(avg)[-5:][::-1]
    sat = sum(
        abs(item["shift_azimuth_px"]) >= 3.999 or abs(item["shift_range_px"]) >= 3.999
        for item in tracking_stats
    )
    return {
        "slice_matrix": slice_matrix,
        "peak_depths_m": [float(dataset.z_grid[p]) for p in peaks],
        "peak_vals": [float(avg[p]) for p in peaks],
        "saturated_fraction": float(sat / max(len(tracking_stats), 1)),
        "mean_correlation": float(np.mean([item["correlation"] for item in tracking_stats])),
        "mean_coherence": float(np.mean([item["coherence"] for item in tracking_stats])),
    }


def main() -> None:
    output_dir = Path("data/processed/biondi_tops_compare")
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = prepare_giza_dataset(
        target_lat=TARGET_LAT,
        target_lon=TARGET_LON,
        cfg=SubApertureConfig(width_bins=48, step_bins=8, n_apertures=96, taper="hann"),
        z_grid=np.linspace(-500.0, 500.0, 241),
        half_lines=2048,
        half_pixels=1024,
    )
    y0, x0 = dataset.default_local_point
    line_points = np.stack(
        [
            np.linspace(y0, y0, 24),
            np.linspace(max(32, x0 - 220), min(dataset.chip_native.shape[1] - 33, x0 + 220), 24),
        ],
        axis=1,
    )

    base = run_case(dataset, prepare_tracking_subapertures(dataset, "none"), line_points)
    deramped = run_case(dataset, prepare_tracking_subapertures(dataset, "tops_deramp"), line_points)
    deramped_2d = run_case(dataset, prepare_tracking_subapertures(dataset, "tops_deramp_2d"), line_points)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, title, case in zip(axes, ["No Deramp", "1D TOPS Deramp", "2D TOPS Deramp"], [base, deramped, deramped_2d]):
        vmax = np.percentile(case["slice_matrix"], 99)
        ax.imshow(
            case["slice_matrix"].T,
            cmap="magma",
            aspect="auto",
            origin="lower",
            vmin=0.0,
            vmax=max(vmax, 1e-6),
            extent=[0, case["slice_matrix"].shape[0] - 1, float(dataset.z_grid[0]), float(dataset.z_grid[-1])],
        )
        ax.set_title(title)
        ax.set_xlabel("Along-line sample")
        ax.set_ylabel("Depth (m)")
    fig.tight_layout()
    fig.savefig(output_dir / "compare.png", dpi=180)
    plt.close(fig)

    summary = {
        "base": base | {"slice_matrix": None},
        "deramped": deramped | {"slice_matrix": None},
        "deramped_2d": deramped_2d | {"slice_matrix": None},
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
