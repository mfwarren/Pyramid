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
)
from giza_backend import TARGET_LAT, TARGET_LON, prepare_giza_dataset
from sardt import SubApertureConfig


def laplacian_z(h: np.ndarray) -> np.ndarray:
    out = np.zeros_like(h)
    out[1:-1] = 2.0 * h[1:-1] - h[:-2] - h[2:]
    out[0] = h[0] - h[1]
    out[-1] = h[-1] - h[-2]
    return out


def laplacian_x(h: np.ndarray) -> np.ndarray:
    out = np.zeros_like(h)
    out[:, 1:-1] = 2.0 * h[:, 1:-1] - h[:, :-2] - h[:, 2:]
    out[:, 0] = h[:, 0] - h[:, 1]
    out[:, -1] = h[:, -1] - h[:, -2]
    return out


def complex_soft_threshold(x: np.ndarray, thresh: float) -> np.ndarray:
    mag = np.abs(x)
    shrink = np.maximum(0.0, mag - thresh) / np.maximum(mag, 1e-12)
    return shrink * x


def solve_global_slice(
    Y: np.ndarray,
    A: np.ndarray,
    lam_x: float = 0.25,
    lam_z: float = 0.5,
    lam_l1: float = 1e-3,
    n_iter: int = 160,
) -> np.ndarray:
    ah = A.conj().T
    spectral = float(np.linalg.norm(A, 2) ** 2)
    step = 1.0 / (spectral + 4.0 * lam_x + 4.0 * lam_z + 1e-6)
    h = np.zeros((A.shape[1], Y.shape[1]), dtype=np.complex128)
    y_state = h.copy()
    t = 1.0

    for _ in range(n_iter):
        grad = ah @ (A @ y_state - Y) + lam_x * laplacian_x(y_state) + lam_z * laplacian_z(y_state)
        candidate = complex_soft_threshold(y_state - step * grad, lam_l1 * step)
        t_next = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t * t))
        y_state = candidate + ((t - 1.0) / t_next) * (candidate - h)
        h = candidate
        t = t_next
    return h


def main() -> None:
    output_dir = Path("data/processed/biondi_global_slice")
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = prepare_giza_dataset(
        target_lat=TARGET_LAT,
        target_lon=TARGET_LON,
        cfg=SubApertureConfig(width_bins=48, step_bins=8, n_apertures=96, taper="hann"),
        z_grid=np.linspace(-500.0, 500.0, 241),
        half_lines=2048,
        half_pixels=1024,
    )

    track_cfg = PixelTrackConfig(patch_radius=8, search_radius=4)
    local_line, local_sample = dataset.default_local_point
    line_points = np.stack(
        [
            np.linspace(local_line, local_line, 36),
            np.linspace(max(32, local_sample - 280), min(dataset.chip_native.shape[1] - 33, local_sample + 280), 36),
        ],
        axis=1,
    )

    center_line = int(round(line_points[len(line_points) // 2, 0]))
    center_sample = int(round(line_points[len(line_points) // 2, 1]))
    center_target, center_debug = local_to_target(dataset, center_line, center_sample)
    A_full, steering_debug = build_acoustic_steering_matrix(
        dataset,
        center_target,
        sound_speed_mps=1500.0,
        vibration_hz=500.0,
        steering_mode="linear",
    )

    cases = {}
    for mode in ["none", "tops_deramp", "tops_deramp_2d"]:
        sub_slcs = prepare_tracking_subapertures(dataset, mode)
        Y_columns = []
        observation_indices = None
        tracking_examples = []
        for i, (line_f, sample_f) in enumerate(line_points):
            ll = int(np.clip(round(line_f), 16, dataset.chip_native.shape[0] - 17))
            ls = int(np.clip(round(sample_f), 16, dataset.chip_native.shape[1] - 17))
            y, tracking_debug, obs_idx = build_displacement_series(
                sub_slcs,
                (ls, ll),
                track_cfg,
                observable="azimuth_phase",
                pair_stride=4,
                series_mode="cumulative",
                detrend_mode="linear",
            )
            if observation_indices is None:
                observation_indices = obs_idx
            Y_columns.append(y)
            if i in {0, len(line_points) // 2, len(line_points) - 1}:
                tracking_examples.append({"index": i, "tracking_head": tracking_debug[:4], "local_line_sample": [ll, ls]})
        Y = np.stack(Y_columns, axis=1)
        A = A_full[np.asarray(observation_indices, dtype=np.int64)]
        H = solve_global_slice(Y, A, lam_x=0.35, lam_z=0.8, lam_l1=3e-3, n_iter=220)
        slice_matrix = np.abs(H.T)
        avg_profile = slice_matrix.mean(axis=0)
        cases[mode] = {
            "slice_matrix": slice_matrix,
            "peak_depth_m": float(dataset.z_grid[int(np.argmax(avg_profile))]),
            "max": float(slice_matrix.max()),
            "mean": float(slice_matrix.mean()),
            "tracking_examples": tracking_examples,
        }

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    mag = np.abs(dataset.chip_native)
    mag_db = 20.0 * np.log10(mag / (mag.max() + 1e-12) + 1e-6)
    axes[0].imshow(mag_db, cmap="gray", aspect="auto")
    axes[0].plot(line_points[:, 1], line_points[:, 0], color="tab:orange", linewidth=2)
    axes[0].set_title("Global inversion line")
    axes[0].set_xticks([])
    axes[0].set_yticks([])
    for ax, (mode, case) in zip(axes[1:], cases.items()):
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
        ax.set_title(f"Global slice: {mode}")
        ax.set_xlabel("Along-line sample")
        ax.set_ylabel("Depth (m)")
    fig.tight_layout()
    fig.savefig(output_dir / "preview.png", dpi=180)
    plt.close(fig)

    summary = {
        "config": {
            "sound_speed_mps": 1500.0,
            "vibration_hz": 500.0,
            "steering_mode": "linear",
            "observable": "azimuth_phase",
            "pair_stride": 4,
            "series_mode": "cumulative",
            "detrend_mode": "linear",
            "global_regularization": {"lam_x": 0.35, "lam_z": 0.8, "lam_l1": 3e-3, "n_iter": 220},
        },
        "center_target_debug": center_debug,
        "steering_debug": steering_debug,
        "cases": {
            mode: {
                "peak_depth_m": case["peak_depth_m"],
                "max": case["max"],
                "mean": case["mean"],
                "tracking_examples": case["tracking_examples"],
            }
            for mode, case in cases.items()
        },
    }
    np.savez_compressed(
        output_dir / "results.npz",
        none_slice=cases["none"]["slice_matrix"],
        deramp_slice=cases["tops_deramp"]["slice_matrix"],
        deramp2d_slice=cases["tops_deramp_2d"]["slice_matrix"],
        z_grid=dataset.z_grid,
        line_points=line_points,
    )
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
