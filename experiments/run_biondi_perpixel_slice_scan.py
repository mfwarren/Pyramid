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


def solve_variable_global_slice(
    y_columns: list[np.ndarray],
    a_columns: list[np.ndarray],
    lam_x: float = 0.35,
    lam_z: float = 0.8,
    lam_l1: float = 3e-3,
    n_iter: int = 220,
) -> np.ndarray:
    nz = a_columns[0].shape[1]
    nx = len(y_columns)
    spectral = max(float(np.linalg.norm(a, 2) ** 2) for a in a_columns)
    step = 1.0 / (spectral + 4.0 * lam_x + 4.0 * lam_z + 1e-6)
    h = np.zeros((nz, nx), dtype=np.complex128)
    y_state = h.copy()
    t = 1.0
    ah_columns = [a.conj().T for a in a_columns]

    for _ in range(n_iter):
        grad = np.zeros_like(y_state)
        for idx, (a, ah, y) in enumerate(zip(a_columns, ah_columns, y_columns)):
            grad[:, idx] = ah @ (a @ y_state[:, idx] - y)
        grad += lam_x * laplacian_x(y_state) + lam_z * laplacian_z(y_state)
        candidate = complex_soft_threshold(y_state - step * grad, lam_l1 * step)
        t_next = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t * t))
        y_state = candidate + ((t - 1.0) / t_next) * (candidate - h)
        h = candidate
        t = t_next
    return h


def peak_metrics(slice_matrix: np.ndarray, z_grid: np.ndarray, edge_guard_bins: int = 6) -> dict:
    peak_indices = np.argmax(slice_matrix, axis=1)
    peak_depths = z_grid[peak_indices]
    edge_mask = (peak_indices < edge_guard_bins) | (peak_indices >= len(z_grid) - edge_guard_bins)
    if len(peak_depths) > 1:
        stable_std = float(np.std(peak_depths))
        jumps = np.abs(np.diff(peak_depths))
        jump_p90 = float(np.percentile(jumps, 90))
    else:
        stable_std = 0.0
        jump_p90 = 0.0
    return {
        "peak_depths_m": peak_depths.tolist(),
        "median_peak_depth_m": float(np.median(peak_depths)),
        "std_peak_depth_m": stable_std,
        "edge_fraction": float(np.mean(edge_mask)),
        "jump_p90_m": jump_p90,
    }


def main() -> None:
    output_dir = Path("data/processed/biondi_perpixel_slice_scan")
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = prepare_giza_dataset(
        target_lat=TARGET_LAT,
        target_lon=TARGET_LON,
        cfg=SubApertureConfig(width_bins=48, step_bins=8, n_apertures=96, taper="hann"),
        z_grid=np.linspace(-500.0, 500.0, 241),
        half_lines=2048,
        half_pixels=1024,
    )
    sub_slcs = prepare_tracking_subapertures(dataset, "tops_deramp_2d")
    track_cfg = PixelTrackConfig(patch_radius=8, search_radius=4)

    local_line, local_sample = dataset.default_local_point
    line_points = np.stack(
        [
            np.linspace(local_line, local_line, 36),
            np.linspace(max(32, local_sample - 280), min(dataset.chip_native.shape[1] - 33, local_sample + 280), 36),
        ],
        axis=1,
    )

    y_columns: list[np.ndarray] = []
    targets = []
    tracking_examples = []
    observation_indices = None
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
        y_columns.append(y)
        target, target_debug = local_to_target(dataset, ll, ls)
        targets.append((target, target_debug))
        if i in {0, len(line_points) // 2, len(line_points) - 1}:
            tracking_examples.append(
                {
                    "index": i,
                    "local_line_sample": [ll, ls],
                    "tracking_head": tracking_debug[:4],
                    "target_debug": target_debug,
                }
            )

    assert observation_indices is not None
    freq_cases = [
        {"sound_speed_mps": 1500.0, "vibration_hz": 100.0},
        {"sound_speed_mps": 1500.0, "vibration_hz": 200.0},
        {"sound_speed_mps": 1500.0, "vibration_hz": 500.0},
        {"sound_speed_mps": 1500.0, "vibration_hz": 1000.0},
        {"sound_speed_mps": 3000.0, "vibration_hz": 200.0},
        {"sound_speed_mps": 3000.0, "vibration_hz": 500.0},
    ]

    cases = []
    for case in freq_cases:
        a_columns = []
        steering_examples = []
        for idx, (target, target_debug) in enumerate(targets):
            a_full, steering_debug = build_acoustic_steering_matrix(
                dataset,
                target,
                sound_speed_mps=case["sound_speed_mps"],
                vibration_hz=case["vibration_hz"],
                steering_mode="linear",
            )
            a_columns.append(a_full[np.asarray(observation_indices, dtype=np.int64)])
            if idx in {0, len(targets) // 2, len(targets) - 1}:
                steering_examples.append(
                    {
                        "index": idx,
                        "target_debug": target_debug,
                        "steering_debug": {
                            "approx_vertical_resolution_m": steering_debug["approx_vertical_resolution_m"],
                            "condition_number": steering_debug["condition_number"],
                            "effective_rank": steering_debug["effective_rank"],
                        },
                    }
                )

        h = solve_variable_global_slice(y_columns, a_columns)
        slice_matrix = np.abs(h.T)
        metrics = peak_metrics(slice_matrix, dataset.z_grid)
        cases.append(
            {
                **case,
                "slice_matrix": slice_matrix,
                "metrics": metrics,
                "score": float((1.0 - metrics["edge_fraction"]) / (1.0 + metrics["std_peak_depth_m"] + 0.25 * metrics["jump_p90_m"])),
                "steering_examples": steering_examples,
            }
        )

    cases.sort(key=lambda item: item["score"], reverse=True)
    best = cases[0]
    peak_depth_stack = np.stack([dataset.z_grid[np.argmax(case["slice_matrix"], axis=1)] for case in cases], axis=0)
    consensus_depth = np.median(peak_depth_stack, axis=0)
    consensus_std = np.std(peak_depth_stack, axis=0)

    fig, axes = plt.subplots(1, 4, figsize=(22, 5))
    mag = np.abs(dataset.chip_native)
    mag_db = 20.0 * np.log10(mag / (mag.max() + 1e-12) + 1e-6)
    axes[0].imshow(mag_db, cmap="gray", aspect="auto")
    axes[0].plot(line_points[:, 1], line_points[:, 0], color="tab:orange", linewidth=2)
    axes[0].set_title("Per-pixel steering line")
    axes[0].set_xticks([])
    axes[0].set_yticks([])

    vmax = np.percentile(best["slice_matrix"], 99)
    axes[1].imshow(
        best["slice_matrix"].T,
        cmap="magma",
        aspect="auto",
        origin="lower",
        vmin=0.0,
        vmax=max(vmax, 1e-6),
        extent=[0, best["slice_matrix"].shape[0] - 1, float(dataset.z_grid[0]), float(dataset.z_grid[-1])],
    )
    axes[1].set_title(f"Best case {best['sound_speed_mps']:.0f} m/s {best['vibration_hz']:.0f} Hz")
    axes[1].set_xlabel("Along-line sample")
    axes[1].set_ylabel("Depth (m)")

    axes[2].plot(consensus_depth, color="tab:orange")
    axes[2].fill_between(
        np.arange(len(consensus_depth)),
        consensus_depth - consensus_std,
        consensus_depth + consensus_std,
        color="tab:orange",
        alpha=0.25,
    )
    axes[2].set_title("Consensus peak depth")
    axes[2].set_xlabel("Along-line sample")
    axes[2].set_ylabel("Depth (m)")

    im = axes[3].imshow(
        peak_depth_stack,
        cmap="coolwarm",
        aspect="auto",
        origin="lower",
        vmin=float(dataset.z_grid[0]),
        vmax=float(dataset.z_grid[-1]),
    )
    axes[3].set_title("Peak depth by frequency case")
    axes[3].set_xlabel("Along-line sample")
    axes[3].set_ylabel("Case index")
    fig.colorbar(im, ax=axes[3], fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_dir / "preview.png", dpi=180)
    plt.close(fig)

    summary = {
        "config": {
            "preprocessing": "tops_deramp_2d",
            "observable": "azimuth_phase",
            "pair_stride": 4,
            "series_mode": "cumulative",
            "detrend_mode": "linear",
            "global_regularization": {"lam_x": 0.35, "lam_z": 0.8, "lam_l1": 3e-3, "n_iter": 220},
            "frequency_cases": freq_cases,
        },
        "best_case": {
            "sound_speed_mps": best["sound_speed_mps"],
            "vibration_hz": best["vibration_hz"],
            "score": best["score"],
            **best["metrics"],
            "steering_examples": best["steering_examples"],
        },
        "consensus": {
            "median_std_peak_depth_m": float(np.median(consensus_std)),
            "max_std_peak_depth_m": float(np.max(consensus_std)),
            "median_consensus_depth_m": float(np.median(consensus_depth)),
        },
        "cases": [
            {
                "sound_speed_mps": case["sound_speed_mps"],
                "vibration_hz": case["vibration_hz"],
                "score": case["score"],
                **case["metrics"],
            }
            for case in cases
        ],
        "tracking_examples": tracking_examples,
    }
    np.savez_compressed(
        output_dir / "results.npz",
        best_slice=best["slice_matrix"],
        peak_depth_stack=peak_depth_stack,
        consensus_depth=consensus_depth,
        consensus_std=consensus_std,
        z_grid=dataset.z_grid,
        line_points=line_points,
    )
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
