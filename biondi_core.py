from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from giza_backend import (
    build_geometry,
    compute_subaperture_centers,
    evaluate_polynomial,
    geodetic_to_ecef,
    interpolate_azimuth_fm_rate,
    interpolate_dc_estimate,
    interpolate_orbit,
    interpolate_target,
)
from sardt import build_doppler_subapertures, solve_tomography


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


def track_complex_shift(master: np.ndarray, slave: np.ndarray, pixel: tuple[int, int], cfg: PixelTrackConfig) -> tuple[dict, dict]:
    r, a = pixel
    pr = cfg.patch_radius
    sr = cfg.search_radius
    nr, na = master.shape
    r0 = max(pr + sr, min(nr - pr - sr - 1, r))
    a0 = max(pr + sr, min(na - pr - sr - 1, a))

    master_patch = master[r0 - pr:r0 + pr + 1, a0 - pr:a0 + pr + 1]
    template = np.abs(master_patch)
    template = template - template.mean()
    template_norm = float(np.linalg.norm(template) + 1e-12)

    score_map = np.full((2 * sr + 1, 2 * sr + 1), -np.inf, dtype=np.float64)
    phase_map = np.zeros_like(score_map, dtype=np.float64)
    coherence_map = np.zeros_like(score_map, dtype=np.float64)

    for ir, dr in enumerate(range(-sr, sr + 1)):
        for ia, da in enumerate(range(-sr, sr + 1)):
            patch = slave[r0 + dr - pr:r0 + dr + pr + 1, a0 + da - pr:a0 + da + pr + 1]
            mag = np.abs(patch)
            mag = mag - mag.mean()
            norm = float(np.linalg.norm(mag) + 1e-12)
            score_map[ir, ia] = float(np.sum(template * mag) / (template_norm * norm))
            coherence = np.vdot(master_patch, patch) / np.sqrt(
                (np.vdot(master_patch, master_patch).real + 1e-12) *
                (np.vdot(patch, patch).real + 1e-12)
            )
            phase_map[ir, ia] = float(np.angle(coherence))
            coherence_map[ir, ia] = float(np.abs(coherence))

    peak_r, peak_a = np.unravel_index(np.argmax(score_map), score_map.shape)
    sub_r = estimate_subpixel_peak(score_map[:, peak_a], peak_r) - sr
    sub_a = estimate_subpixel_peak(score_map[peak_r, :], peak_a) - sr
    corr = float(score_map[peak_r, peak_a])
    phase = float(phase_map[peak_r, peak_a])
    coherence = float(coherence_map[peak_r, peak_a])

    obs = {
        "complex": complex(sub_a, sub_r) * np.exp(1j * phase),
        "azimuth": complex(sub_a, 0.0),
        "range": complex(sub_r, 0.0),
        "phase": np.exp(1j * phase),
        "coherence": complex(coherence, 0.0),
        "azimuth_phase": complex(sub_a, 0.0) * np.exp(1j * phase),
    }
    debug = {
        "shift_azimuth_px": float(sub_a),
        "shift_range_px": float(sub_r),
        "correlation": corr,
        "phase_rad": phase,
        "coherence": coherence,
    }
    return obs, debug


def detrend_series(y: np.ndarray, mode: str) -> np.ndarray:
    if mode == "none":
        return y
    if mode == "mean":
        return y - np.mean(y)
    if mode == "linear":
        t = np.arange(len(y), dtype=np.float64)
        design = np.c_[t, np.ones_like(t)]
        coef_real, *_ = np.linalg.lstsq(design, y.real, rcond=None)
        coef_imag, *_ = np.linalg.lstsq(design, y.imag, rcond=None)
        trend = (design @ coef_real) + 1j * (design @ coef_imag)
        return y - trend
    if mode == "first_difference":
        return np.diff(y, prepend=y[:1])
    raise ValueError(f"Unsupported detrend mode: {mode}")


def build_displacement_series(
    sub_slcs: list[np.ndarray],
    pixel: tuple[int, int],
    cfg: PixelTrackConfig,
    observable: str = "complex",
    pair_stride: int = 1,
    series_mode: str = "cumulative",
    detrend_mode: str = "mean",
) -> tuple[np.ndarray, list[dict], list[int]]:
    series = [0.0j]
    observation_indices = [0]
    debug = []
    idx = 0
    while idx + pair_stride < len(sub_slcs):
        obs, dbg = track_complex_shift(sub_slcs[idx], sub_slcs[idx + pair_stride], pixel, cfg)
        value = obs[observable]
        if series_mode == "cumulative":
            series.append(series[-1] + value)
        elif series_mode == "incremental":
            series.append(value)
        else:
            raise ValueError(f"Unsupported series mode: {series_mode}")
        observation_indices.append(idx + pair_stride)
        debug.append({"pair": [idx, idx + pair_stride], **dbg})
        idx += 1
    y = np.asarray(series, dtype=np.complex128)
    y = detrend_series(y, detrend_mode)
    return y, debug, observation_indices


def local_to_target(dataset, local_line: int, local_sample: int) -> tuple[dict, dict]:
    global_line = dataset.line_bounds[0] + local_line
    global_pixel = dataset.pixel_bounds[0] + local_sample
    target = interpolate_target(dataset.meta, global_line, global_pixel)
    return target, {"global_line_pixel": [global_line, global_pixel]}


def apply_tops_deramp(
    chip: np.ndarray,
    meta: dict,
    target: dict,
    center_azimuth_index: int | None = None,
) -> np.ndarray:
    if center_azimuth_index is None:
        center_azimuth_index = chip.shape[1] // 2
    tau = (np.arange(chip.shape[1], dtype=np.float64) - center_azimuth_index) * meta["azimuth_time_interval"]
    ka = interpolate_azimuth_fm_rate(meta, target["azimuth_time_seconds"], target["slant_range_time"])
    dc = interpolate_dc_estimate(meta, target["azimuth_time_seconds"], target["slant_range_time"])["data_dc_hz"]
    phase = 2.0 * np.pi * dc * tau + np.pi * ka * tau * tau
    return chip * np.exp(-1j * phase[None, :])


def apply_tops_deramp_2d(dataset) -> np.ndarray:
    chip = dataset.chip
    global_lines = np.arange(dataset.line_bounds[0], dataset.line_bounds[1], dtype=np.float64)
    global_pixels = np.arange(dataset.pixel_bounds[0], dataset.pixel_bounds[1], dtype=np.float64)
    az_times = dataset.meta["product_first_line_time_seconds"] + global_lines * dataset.meta["azimuth_time_interval"]
    center_time = float(az_times[len(az_times) // 2])
    eta = az_times - center_time

    slant_range_times = dataset.meta["slant_range_time"] + global_pixels / dataset.meta["range_sampling_rate"]

    fm_nodes = dataset.meta["azimuth_fm_rate_list"]
    dc_nodes = dataset.meta["dc_estimate_list"]
    ka_rows = np.empty((len(az_times), len(slant_range_times)), dtype=np.float64)
    dc_rows = np.empty_like(ka_rows)

    for i, t in enumerate(az_times):
        fm_node = min(fm_nodes, key=lambda item: abs(item["azimuth_time_seconds"] - t))
        dc_node = min(dc_nodes, key=lambda item: abs(item["azimuth_time_seconds"] - t))
        delta_tau_fm = slant_range_times - fm_node["t0"]
        delta_tau_dc = slant_range_times - dc_node["t0"]
        c0, c1, c2 = fm_node["coeffs"]
        ka_rows[i] = c0 + c1 * delta_tau_fm + c2 * delta_tau_fm * delta_tau_fm
        dc_rows[i] = (
            dc_node["data_coeffs"][0]
            + dc_node["data_coeffs"][1] * delta_tau_dc
            + dc_node["data_coeffs"][2] * delta_tau_dc * delta_tau_dc
        )

    phase = 2.0 * np.pi * dc_rows.T * eta[None, :] + np.pi * ka_rows.T * (eta[None, :] ** 2)
    return chip * np.exp(-1j * phase)


def prepare_tracking_subapertures(dataset, preprocessing_mode: str = "none") -> list[np.ndarray]:
    if preprocessing_mode == "none":
        return dataset.sub_slcs
    if preprocessing_mode == "tops_deramp":
        local_line, local_sample = dataset.default_local_point
        target, _ = local_to_target(dataset, local_line, local_sample)
        chip_pre = apply_tops_deramp(dataset.chip, dataset.meta, target)
        return build_doppler_subapertures(chip_pre, dataset.cfg)
    if preprocessing_mode == "tops_deramp_2d":
        chip_pre = apply_tops_deramp_2d(dataset)
        return build_doppler_subapertures(chip_pre, dataset.cfg)
    raise ValueError(f"Unsupported preprocessing mode: {preprocessing_mode}")


def build_acoustic_steering_matrix(
    dataset,
    target: dict,
    sound_speed_mps: float,
    vibration_hz: float,
    steering_mode: str = "linear",
) -> tuple[np.ndarray, dict]:
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

    aperture_coords_arr = np.asarray(aperture_coords, dtype=np.float64)
    z = np.asarray(dataset.z_grid, dtype=np.float64)
    acoustic_wavelength = sound_speed_mps / vibration_hz
    k = 2.0 * np.pi / max(acoustic_wavelength, 1e-12)
    slant_range = float(geom_debug["slant_range_m"])

    if steering_mode == "linear":
        phase = k * aperture_coords_arr[:, None] * z[None, :] / max(slant_range, 1e-12)
    elif steering_mode == "hyperbolic":
        path = np.sqrt(aperture_coords_arr[:, None] ** 2 + z[None, :] ** 2)
        ref_path = np.abs(aperture_coords_arr)[:, None]
        phase = k * (path - ref_path)
    elif steering_mode == "quadratic":
        phase = k * (aperture_coords_arr[:, None] ** 2) * z[None, :] / max(slant_range ** 2, 1e-12)
    else:
        raise ValueError(f"Unsupported steering mode: {steering_mode}")

    A = np.exp(1j * phase)
    col_norm = np.linalg.norm(A, axis=0, keepdims=True) + 1e-12
    A = A / col_norm
    singular_values = np.linalg.svd(A, compute_uv=False)
    debug = {
        "sound_speed_mps": sound_speed_mps,
        "vibration_hz": vibration_hz,
        "acoustic_wavelength_m": acoustic_wavelength,
        "aperture_coords_m": aperture_coords,
        "subaperture_times_seconds": sub_times,
        "slant_range_m": slant_range,
        "steering_mode": steering_mode,
        "approx_vertical_resolution_m": float(acoustic_wavelength * slant_range / max(np.ptp(aperture_coords_arr), 1e-12)),
        "condition_number": float(singular_values[0] / max(singular_values[-1], 1e-12)),
        "effective_rank": int(np.sum(singular_values / singular_values[0] > 1e-3)),
        "em_geometry_debug": geom_debug,
    }
    return A, debug


def solve_acoustic_tomography(
    y: np.ndarray,
    A: np.ndarray,
    inversion_mode: str = "pinv",
    alpha: float = 1e-2,
    sparsity: int = 3,
    lam: float = 1e-2,
    n_iter: int = 80,
) -> np.ndarray:
    return solve_tomography(
        y,
        A,
        mode=inversion_mode,
        alpha=alpha,
        sparsity=sparsity,
        lam=lam,
        n_iter=n_iter,
    )
