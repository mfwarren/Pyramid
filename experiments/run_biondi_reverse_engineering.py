#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from giza_backend import (
    TARGET_LAT,
    TARGET_LON,
    build_geometry,
    compute_subaperture_centers,
    geodetic_to_ecef,
    interpolate_orbit,
    interpolate_target,
    prepare_giza_dataset,
)
from sardt import SubApertureConfig, build_doppler_subapertures


@dataclass
class PixelTrackConfig:
    patch_radius: int = 6
    search_radius: int = 4


def estimate_subpixel_peak(values: np.ndarray, peak_idx: int) -> float:
    if peak_idx <= 0 or peak_idx >= len(values) - 1:
        return float(peak_idx)
    left = float(values[peak_idx - 1])
    center = float(values[peak_idx])
    right = float(values[peak_idx + 1])
    denom = left - 2.0 * center + right
    if abs(denom) < 1e-12:
        return float(peak_idx)
    return float(peak_idx + 0.5 * (left - right) / denom)


def track_complex_shift(master: np.ndarray, slave: np.ndarray, pixel: tuple[int, int], cfg: PixelTrackConfig) -> tuple[complex, dict]:
    r, a = pixel
    pr = cfg.patch_radius
    sr = cfg.search_radius
    nr, na = master.shape
    r0 = max(pr + sr, min(nr - pr - sr - 1, r))
    a0 = max(pr + sr, min(na - pr - sr - 1, a))

    template = np.abs(master[r0 - pr:r0 + pr + 1, a0 - pr:a0 + pr + 1])
    template = template - template.mean()
    template_norm = float(np.linalg.norm(template) + 1e-12)

    score_map = np.full((2 * sr + 1, 2 * sr + 1), -np.inf, dtype=np.float64)
    phase_map = np.zeros_like(score_map, dtype=np.float64)

    for ir, dr in enumerate(range(-sr, sr + 1)):
        for ia, da in enumerate(range(-sr, sr + 1)):
            patch = slave[r0 + dr - pr:r0 + dr + pr + 1, a0 + da - pr:a0 + da + pr + 1]
            mag = np.abs(patch)
            mag = mag - mag.mean()
            norm = float(np.linalg.norm(mag) + 1e-12)
            score_map[ir, ia] = float(np.sum(template * mag) / (template_norm * norm))
            phase_map[ir, ia] = float(np.angle(np.vdot(master[r0 - pr:r0 + pr + 1, a0 - pr:a0 + pr + 1], patch)))

    peak_r, peak_a = np.unravel_index(np.argmax(score_map), score_map.shape)
    sub_r = estimate_subpixel_peak(score_map[:, peak_a], peak_r) - sr
    sub_a = estimate_subpixel_peak(score_map[peak_r, :], peak_a) - sr
    corr = float(score_map[peak_r, peak_a])
    phase = float(phase_map[peak_r, peak_a])

    displacement = complex(sub_a, sub_r) * np.exp(1j * phase)
    return displacement, {
        "shift_azimuth_px": float(sub_a),
        "shift_range_px": float(sub_r),
        "correlation": corr,
        "phase_rad": phase,
    }


def build_displacement_series(sub_slcs: list[np.ndarray], pixel: tuple[int, int], cfg: PixelTrackConfig) -> tuple[np.ndarray, list[dict]]:
    cumulative = [0.0j]
    debug = []
    for idx in range(len(sub_slcs) - 1):
        disp, dbg = track_complex_shift(sub_slcs[idx], sub_slcs[idx + 1], pixel, cfg)
        cumulative.append(cumulative[-1] + disp)
        debug.append({"pair": [idx, idx + 1], **dbg})
    y = np.asarray(cumulative, dtype=np.complex128)
    y = y - np.mean(y)
    return y, debug


def build_acoustic_geometry(dataset, target: dict, sound_speed_mps: float, vibration_hz: float) -> tuple[np.ndarray, dict]:
    geom, geom_debug = build_geometry(dataset.meta, target, dataset.chip.shape[1], dataset.cfg)
    centers = compute_subaperture_centers(dataset.chip.shape[1], dataset.cfg)
    center_axis = 0.5 * (dataset.chip.shape[1] - 1)
    dt = dataset.meta["azimuth_time_interval"]
    line_time = target["azimuth_time_seconds"]

    ref_pos, ref_vel = interpolate_orbit(dataset.meta["orbits"], line_time)
    target_ecef = geodetic_to_ecef(target["lat"], target["lon"], target["height"])
    los = target_ecef - ref_pos
    los /= np.linalg.norm(los)
    az_dir = ref_vel - np.dot(ref_vel, los) * los
    az_dir /= np.linalg.norm(az_dir)

    aperture_coords = []
    sub_times = []
    for center in centers:
        t_sub = line_time + (center - center_axis) * dt
        pos, _ = interpolate_orbit(dataset.meta["orbits"], t_sub)
        baseline = pos - ref_pos
        aperture_coords.append(float(np.dot(baseline, az_dir)))
        sub_times.append(t_sub)

    slant_range = geom_debug["slant_range_m"]
    acoustic_wavelength = sound_speed_mps / vibration_hz
    steering = np.exp(
        1j * 2.0 * np.pi * np.asarray(aperture_coords)[:, None] * dataset.z_grid[None, :] /
        (acoustic_wavelength * slant_range + 1e-12)
    )
    debug = {
        "sound_speed_mps": sound_speed_mps,
        "vibration_hz": vibration_hz,
        "acoustic_wavelength_m": acoustic_wavelength,
        "aperture_coords_m": aperture_coords,
        "subaperture_times_seconds": sub_times,
        "slant_range_m": slant_range,
        "approx_vertical_resolution_m": float(acoustic_wavelength * slant_range / max(max(aperture_coords) - min(aperture_coords), 1e-12)),
        "em_geometry_debug": geom_debug,
    }
    return steering, debug


def solve_pinv(y: np.ndarray, A: np.ndarray) -> np.ndarray:
    return np.linalg.pinv(A) @ y


def run_line(dataset, line_points: np.ndarray, sound_speed_mps: float, vibration_hz: float, track_cfg: PixelTrackConfig) -> tuple[np.ndarray, list[dict]]:
    slice_matrix = np.empty((len(line_points), len(dataset.z_grid)), dtype=np.float64)
    debug_points = []
    for idx, (local_line_f, local_sample_f) in enumerate(line_points):
        local_line = int(np.clip(round(local_line_f), 16, dataset.chip_native.shape[0] - 17))
        local_sample = int(np.clip(round(local_sample_f), 16, dataset.chip_native.shape[1] - 17))
        pixel = (local_sample, local_line)
        global_line = dataset.line_bounds[0] + local_line
        global_pixel = dataset.pixel_bounds[0] + local_sample
        target = interpolate_target(dataset.meta, global_line, global_pixel)
        y, tracking_debug = build_displacement_series(dataset.sub_slcs, pixel, track_cfg)
        A, acoustic_debug = build_acoustic_geometry(dataset, target, sound_speed_mps, vibration_hz)
        h = solve_pinv(y, A)
        slice_matrix[idx] = np.abs(h)
        if idx in {0, len(line_points) // 2, len(line_points) - 1}:
            debug_points.append(
                {
                    "index": idx,
                    "local_line_sample": [local_line, local_sample],
                    "tracking_debug_head": tracking_debug[:5],
                    "acoustic_debug": acoustic_debug,
                }
            )
    return slice_matrix, debug_points


def main() -> None:
    output_dir = Path("data/processed/biondi_reverse_engineer")
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = prepare_giza_dataset(
        target_lat=TARGET_LAT,
        target_lon=TARGET_LON,
        cfg=SubApertureConfig(width_bins=96, step_bins=16, n_apertures=48, taper="hann"),
        z_grid=np.linspace(-500.0, 500.0, 241),
        half_lines=2048,
        half_pixels=1024,
    )

    local_line, local_sample = dataset.default_local_point
    line_points = np.stack(
        [
            np.linspace(local_line, local_line, 48),
            np.linspace(max(32, local_sample - 320), min(dataset.chip_native.shape[1] - 33, local_sample + 320), 48),
        ],
        axis=1,
    )

    sound_speed_mps = 3000.0
    vibration_hz = 200.0
    track_cfg = PixelTrackConfig(patch_radius=8, search_radius=4)
    slice_matrix, debug_points = run_line(dataset, line_points, sound_speed_mps, vibration_hz, track_cfg)

    vmax = np.percentile(slice_matrix, 99)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    mag = np.abs(dataset.chip_native)
    mag_db = 20.0 * np.log10(mag / (mag.max() + 1e-12) + 1e-6)
    axes[0].imshow(mag_db, cmap="gray", aspect="auto")
    axes[0].plot(line_points[:, 1], line_points[:, 0], color="tab:orange", linewidth=2)
    axes[0].set_title("Giza magnitude with test line")
    axes[0].set_xticks([])
    axes[0].set_yticks([])

    axes[1].imshow(
        slice_matrix.T,
        cmap="magma",
        aspect="auto",
        origin="lower",
        vmin=0.0,
        vmax=max(vmax, 1e-6),
        extent=[0, slice_matrix.shape[0] - 1, float(dataset.z_grid[0]), float(dataset.z_grid[-1])],
    )
    axes[1].set_title("Reverse-engineered Biondi-style slice")
    axes[1].set_xlabel("Along-line sample")
    axes[1].set_ylabel("Depth (m)")
    fig.tight_layout()
    fig.savefig(output_dir / "preview.png", dpi=180)
    plt.close(fig)

    summary = {
        "config": {
            "subaperture": {
                "width_bins": dataset.cfg.width_bins,
                "step_bins": dataset.cfg.step_bins,
                "n_apertures": dataset.cfg.n_apertures,
            },
            "z_grid": [float(dataset.z_grid[0]), float(dataset.z_grid[-1]), int(len(dataset.z_grid))],
            "sound_speed_mps": sound_speed_mps,
            "vibration_hz": vibration_hz,
            "tracking": track_cfg.__dict__,
        },
        "dataset_summary": dataset.to_summary(),
        "slice_stats": {
            "max": float(slice_matrix.max()),
            "mean": float(slice_matrix.mean()),
            "peak_depth_m": float(dataset.z_grid[np.argmax(slice_matrix.mean(axis=0))]),
        },
        "debug_points": debug_points,
    }
    np.savez_compressed(output_dir / "results.npz", slice_matrix=slice_matrix, z_grid=dataset.z_grid, line_points=line_points)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
