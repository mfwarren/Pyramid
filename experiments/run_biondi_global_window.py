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


def laplacian_axis(arr: np.ndarray, axis: int) -> np.ndarray:
    out = np.zeros_like(arr)
    slicer_mid = [slice(None)] * arr.ndim
    slicer_prev = [slice(None)] * arr.ndim
    slicer_next = [slice(None)] * arr.ndim
    slicer_mid[axis] = slice(1, -1)
    slicer_prev[axis] = slice(0, -2)
    slicer_next[axis] = slice(2, None)
    out[tuple(slicer_mid)] = 2.0 * arr[tuple(slicer_mid)] - arr[tuple(slicer_prev)] - arr[tuple(slicer_next)]

    slicer0 = [slice(None)] * arr.ndim
    slicer1 = [slice(None)] * arr.ndim
    slicer0[axis] = 0
    slicer1[axis] = 1
    out[tuple(slicer0)] = arr[tuple(slicer0)] - arr[tuple(slicer1)]

    slicer_last = [slice(None)] * arr.ndim
    slicer_prevlast = [slice(None)] * arr.ndim
    slicer_last[axis] = -1
    slicer_prevlast[axis] = -2
    out[tuple(slicer_last)] = arr[tuple(slicer_last)] - arr[tuple(slicer_prevlast)]
    return out


def complex_soft_threshold(x: np.ndarray, thresh: float) -> np.ndarray:
    mag = np.abs(x)
    shrink = np.maximum(0.0, mag - thresh) / np.maximum(mag, 1e-12)
    return shrink * x


def solve_global_volume(
    Y: np.ndarray,
    A: np.ndarray,
    grid_shape: tuple[int, int],
    lam_x: float = 0.25,
    lam_y: float = 0.25,
    lam_z: float = 0.75,
    lam_l1: float = 2e-3,
    n_iter: int = 200,
) -> np.ndarray:
    ah = A.conj().T
    nz = A.shape[1]
    ny, nx = grid_shape
    spectral = float(np.linalg.norm(A, 2) ** 2)
    step = 1.0 / (spectral + 4.0 * (lam_x + lam_y + lam_z) + 1e-6)
    h = np.zeros((nz, ny, nx), dtype=np.complex128)
    y_state = h.copy()
    t = 1.0

    for _ in range(n_iter):
        h2 = y_state.reshape(nz, ny * nx)
        grad_data = (ah @ (A @ h2 - Y)).reshape(nz, ny, nx)
        grad = (
            grad_data
            + lam_z * laplacian_axis(y_state, 0)
            + lam_y * laplacian_axis(y_state, 1)
            + lam_x * laplacian_axis(y_state, 2)
        )
        candidate = complex_soft_threshold(y_state - step * grad, lam_l1 * step)
        t_next = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t * t))
        y_state = candidate + ((t - 1.0) / t_next) * (candidate - h)
        h = candidate
        t = t_next
    return h


def main() -> None:
    output_dir = Path("data/processed/biondi_global_window")
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
    line_offsets = np.arange(-6, 7, 2)
    sample_offsets = np.arange(-120, 121, 24)
    ny = len(line_offsets)
    nx = len(sample_offsets)

    center_target, center_debug = local_to_target(dataset, int(local_line), int(local_sample))
    A_full, steering_debug = build_acoustic_steering_matrix(
        dataset,
        center_target,
        sound_speed_mps=1500.0,
        vibration_hz=500.0,
        steering_mode="linear",
    )

    observation_indices = None
    Y_cols = []
    debug_examples = []
    coords = []
    for yi, dy in enumerate(line_offsets):
        for xi, dx in enumerate(sample_offsets):
            ll = int(np.clip(local_line + dy, 16, dataset.chip_native.shape[0] - 17))
            ls = int(np.clip(local_sample + dx, 16, dataset.chip_native.shape[1] - 17))
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
            Y_cols.append(y)
            coords.append((yi, xi, ll, ls))
            if (yi, xi) in {(0, 0), (ny // 2, nx // 2), (ny - 1, nx - 1)}:
                debug_examples.append(
                    {
                        "grid_index": [yi, xi],
                        "local_line_sample": [ll, ls],
                        "tracking_head": tracking_debug[:4],
                    }
                )

    Y = np.stack(Y_cols, axis=1)
    A = A_full[np.asarray(observation_indices, dtype=np.int64)]
    H = solve_global_volume(Y, A, (ny, nx), lam_x=0.35, lam_y=0.35, lam_z=0.8, lam_l1=3e-3, n_iter=220)
    volume = np.abs(H)

    center_column = volume[:, ny // 2, :]
    depth_profile = volume.mean(axis=(1, 2))
    peak_depth = float(dataset.z_grid[int(np.argmax(depth_profile))])

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    mag = np.abs(dataset.chip_native)
    mag_db = 20.0 * np.log10(mag / (mag.max() + 1e-12) + 1e-6)
    axes[0].imshow(mag_db, cmap="gray", aspect="auto")
    plotted = np.array([[coords[j][3], coords[j][2]] for j in range(len(coords))]).reshape(ny, nx, 2)
    axes[0].scatter(plotted[:, :, 0], plotted[:, :, 1], s=14, c="tab:orange")
    axes[0].set_title("2D global window")
    axes[0].set_xticks([])
    axes[0].set_yticks([])

    vmax = np.percentile(center_column, 99)
    axes[1].imshow(
        center_column.T,
        cmap="magma",
        aspect="auto",
        origin="lower",
        vmin=0.0,
        vmax=max(vmax, 1e-6),
        extent=[0, center_column.shape[1] - 1, float(dataset.z_grid[0]), float(dataset.z_grid[-1])],
    )
    axes[1].set_title("Center-row depth slice")
    axes[1].set_xlabel("Window x")
    axes[1].set_ylabel("Depth (m)")

    axes[2].plot(dataset.z_grid, depth_profile / (depth_profile.max() + 1e-12), color="tab:orange")
    axes[2].set_title("Window-averaged depth profile")
    axes[2].set_xlabel("Depth (m)")
    axes[2].set_ylabel("Normalized amplitude")
    fig.tight_layout()
    fig.savefig(output_dir / "preview.png", dpi=180)
    plt.close(fig)

    summary = {
        "config": {
            "preprocessing": "tops_deramp_2d",
            "sound_speed_mps": 1500.0,
            "vibration_hz": 500.0,
            "steering_mode": "linear",
            "observable": "azimuth_phase",
            "pair_stride": 4,
            "series_mode": "cumulative",
            "detrend_mode": "linear",
            "window_shape": [ny, nx],
            "global_regularization": {"lam_x": 0.35, "lam_y": 0.35, "lam_z": 0.8, "lam_l1": 3e-3, "n_iter": 220},
        },
        "center_target_debug": center_debug,
        "steering_debug": steering_debug,
        "depth_profile_peak_m": peak_depth,
        "depth_profile_top5_m": [float(dataset.z_grid[i]) for i in np.argsort(depth_profile)[-5:][::-1]],
        "volume_stats": {
            "max": float(volume.max()),
            "mean": float(volume.mean()),
        },
        "debug_examples": debug_examples,
    }
    np.savez_compressed(output_dir / "results.npz", volume=volume, z_grid=dataset.z_grid, line_offsets=line_offsets, sample_offsets=sample_offsets)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
