#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from biondi_core import build_acoustic_steering_matrix, local_to_target
from giza_backend import TARGET_LAT, TARGET_LON, build_geometry, prepare_giza_dataset
from sardt import SubApertureConfig, build_steering_matrix, compute_kz


def acoustic_depth_resolution(sound_speed_mps: float, vibration_hz: np.ndarray, slant_range_m: float, aperture_m: float) -> np.ndarray:
    acoustic_wavelength = sound_speed_mps / np.maximum(vibration_hz, 1e-12)
    return acoustic_wavelength * slant_range_m / np.maximum(aperture_m, 1e-12)


def required_aperture(sound_speed_mps: float, vibration_hz: np.ndarray, slant_range_m: float, target_resolution_m: float) -> np.ndarray:
    acoustic_wavelength = sound_speed_mps / np.maximum(vibration_hz, 1e-12)
    return acoustic_wavelength * slant_range_m / max(target_resolution_m, 1e-12)


def phase_displacement_sigma(radar_wavelength_m: float, coherence: float, looks: int = 1) -> dict:
    gamma = float(np.clip(coherence, 1e-4, 0.999999))
    sigma_phi = math.sqrt(max(0.0, 1.0 - gamma * gamma) / (2.0 * looks * gamma * gamma))
    sigma_u = radar_wavelength_m * sigma_phi / (4.0 * math.pi)
    return {"coherence": gamma, "phase_sigma_rad": sigma_phi, "los_displacement_sigma_m": sigma_u}


def make_linear_steering(z_grid: np.ndarray, aperture_coords_m: np.ndarray, slant_range_m: float, sound_speed_mps: float, vibration_hz: float) -> np.ndarray:
    acoustic_wavelength = sound_speed_mps / max(vibration_hz, 1e-12)
    k = 2.0 * np.pi / max(acoustic_wavelength, 1e-12)
    phase = k * aperture_coords_m[:, None] * z_grid[None, :] / max(slant_range_m, 1e-12)
    a = np.exp(1j * phase)
    return a / (np.linalg.norm(a, axis=0, keepdims=True) + 1e-12)


def psf_metrics(a: np.ndarray, z_grid: np.ndarray) -> dict:
    mid = len(z_grid) // 2
    target = a[:, mid]
    resp = np.abs(a.conj().T @ target)
    resp /= resp.max() + 1e-12
    peak_idx = int(np.argmax(resp))
    above = np.flatnonzero(resp >= 0.5)
    if len(above) >= 2:
        fwhm = float(abs(z_grid[above[-1]] - z_grid[above[0]]))
    else:
        fwhm = float("inf")
    s = np.linalg.svd(a, compute_uv=False)
    return {
        "fwhm_m": fwhm,
        "condition_number": float(s[0] / max(s[-1], 1e-12)),
        "effective_rank": int(np.sum(s / s[0] > 1e-3)),
        "psf_peak_depth_m": float(z_grid[peak_idx]),
    }


def main() -> None:
    output_dir = Path("data/processed/sensor_feasibility_model")
    output_dir.mkdir(parents=True, exist_ok=True)

    z_grid = np.linspace(-500.0, 500.0, 1001)
    cfg = SubApertureConfig(width_bins=48, step_bins=8, n_apertures=96, taper="hann")
    dataset = prepare_giza_dataset(
        target_lat=TARGET_LAT,
        target_lon=TARGET_LON,
        cfg=cfg,
        z_grid=z_grid,
        half_lines=2048,
        half_pixels=1024,
    )

    local_line, local_sample = dataset.default_local_point
    target, target_debug = local_to_target(dataset, local_line, local_sample)
    geom, geom_debug = build_geometry(dataset.meta, target, dataset.chip.shape[1], dataset.cfg)
    kz = compute_kz(geom)
    a_em = build_steering_matrix(geom, z_grid)

    a_acoustic_ref, acoustic_debug = build_acoustic_steering_matrix(
        dataset,
        target,
        sound_speed_mps=1500.0,
        vibration_hz=500.0,
        steering_mode="linear",
    )
    sentinel_aperture_coords = np.asarray(acoustic_debug["aperture_coords_m"], dtype=np.float64)
    sentinel_aperture_span_m = float(np.ptp(sentinel_aperture_coords))
    slant_range_m = float(acoustic_debug["slant_range_m"])
    radar_wavelength_s1 = float(geom.wavelength)

    freq_grid = np.array([10.0, 20.0, 50.0, 100.0, 200.0, 500.0, 1000.0], dtype=np.float64)
    wave_speeds = [500.0, 1500.0, 3000.0]
    target_resolutions = [1.0, 5.0, 10.0, 50.0]

    cosmo_like_cases = [
        {"name": "cosmo_like_500m", "aperture_m": 500.0, "radar_wavelength_m": 0.031},
        {"name": "cosmo_like_1000m", "aperture_m": 1000.0, "radar_wavelength_m": 0.031},
        {"name": "cosmo_like_5000m", "aperture_m": 5000.0, "radar_wavelength_m": 0.031},
        {"name": "cosmo_like_42000m", "aperture_m": 42000.0, "radar_wavelength_m": 0.031},
    ]

    resolution_rows = []
    psf_rows = []
    for sound_speed_mps in wave_speeds:
        sentinel_res = acoustic_depth_resolution(sound_speed_mps, freq_grid, slant_range_m, sentinel_aperture_span_m)
        resolution_rows.append(
            {
                "sensor": "sentinel1_actual",
                "sound_speed_mps": sound_speed_mps,
                "aperture_m": sentinel_aperture_span_m,
                "resolution_by_frequency_m": {str(int(f)): float(r) for f, r in zip(freq_grid, sentinel_res)},
                "required_aperture_for_resolution_m": {
                    str(res): {str(int(f)): float(required_aperture(sound_speed_mps, np.array([f]), slant_range_m, res)[0]) for f in freq_grid}
                    for res in target_resolutions
                },
            }
        )
        for vibration_hz in [100.0, 200.0, 500.0, 1000.0]:
            a = make_linear_steering(z_grid, sentinel_aperture_coords, slant_range_m, sound_speed_mps, vibration_hz)
            psf_rows.append(
                {
                    "sensor": "sentinel1_actual",
                    "sound_speed_mps": sound_speed_mps,
                    "vibration_hz": vibration_hz,
                    **psf_metrics(a, z_grid),
                }
            )

    for case in cosmo_like_cases:
        aperture_coords = np.linspace(-0.5 * case["aperture_m"], 0.5 * case["aperture_m"], len(sentinel_aperture_coords))
        for sound_speed_mps in wave_speeds:
            cosmo_res = acoustic_depth_resolution(sound_speed_mps, freq_grid, slant_range_m, case["aperture_m"])
            resolution_rows.append(
                {
                    "sensor": case["name"],
                    "sound_speed_mps": sound_speed_mps,
                    "aperture_m": case["aperture_m"],
                    "resolution_by_frequency_m": {str(int(f)): float(r) for f, r in zip(freq_grid, cosmo_res)},
                    "required_aperture_for_resolution_m": {
                        str(res): {str(int(f)): float(required_aperture(sound_speed_mps, np.array([f]), slant_range_m, res)[0]) for f in freq_grid}
                        for res in target_resolutions
                    },
                }
            )
            for vibration_hz in [100.0, 200.0, 500.0, 1000.0]:
                a = make_linear_steering(z_grid, aperture_coords, slant_range_m, sound_speed_mps, vibration_hz)
                psf_rows.append(
                    {
                        "sensor": case["name"],
                        "sound_speed_mps": sound_speed_mps,
                        "vibration_hz": vibration_hz,
                        **psf_metrics(a, z_grid),
                    }
                )

    phase_precision = {
        "sentinel1_c_band": {
            str(coh): phase_displacement_sigma(radar_wavelength_s1, coh, looks=1)
            for coh in [0.3, 0.5, 0.8, 0.95]
        },
        "cosmo_like_x_band": {
            str(coh): phase_displacement_sigma(0.031, coh, looks=1)
            for coh in [0.3, 0.5, 0.8, 0.95]
        },
    }

    em_summary = {
        "kz_span_rad_per_m": float(kz.max() - kz.min()),
        "vertical_resolution_m": float((2.0 * np.pi) / max(float(kz.max() - kz.min()), 1e-12)),
        "condition_number": float(np.linalg.cond(a_em)),
    }

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for sound_speed_mps, color in zip(wave_speeds, ["tab:blue", "tab:orange", "tab:green"]):
        res = acoustic_depth_resolution(sound_speed_mps, freq_grid, slant_range_m, sentinel_aperture_span_m)
        axes[0, 0].plot(freq_grid, res, marker="o", color=color, label=f"S1 c={sound_speed_mps:.0f}")
    axes[0, 0].set_title("Sentinel-1 actual aperture resolution")
    axes[0, 0].set_xlabel("Vibration frequency (Hz)")
    axes[0, 0].set_ylabel("Acoustic depth resolution (m)")
    axes[0, 0].set_yscale("log")
    axes[0, 0].legend(fontsize=8)

    for case, style in zip(cosmo_like_cases, ["-", "--", "-.", ":"]):
        res = acoustic_depth_resolution(1500.0, freq_grid, slant_range_m, case["aperture_m"])
        axes[0, 1].plot(freq_grid, res, marker="o", linestyle=style, label=case["name"])
    axes[0, 1].set_title("COSMO-like aperture sweep at c=1500 m/s")
    axes[0, 1].set_xlabel("Vibration frequency (Hz)")
    axes[0, 1].set_ylabel("Acoustic depth resolution (m)")
    axes[0, 1].set_yscale("log")
    axes[0, 1].legend(fontsize=8)

    coh_vals = [0.3, 0.5, 0.8, 0.95]
    s1_sigmas_mm = [1e3 * phase_precision["sentinel1_c_band"][str(coh)]["los_displacement_sigma_m"] for coh in coh_vals]
    cs_sigmas_mm = [1e3 * phase_precision["cosmo_like_x_band"][str(coh)]["los_displacement_sigma_m"] for coh in coh_vals]
    x = np.arange(len(coh_vals))
    axes[1, 0].bar(x - 0.18, s1_sigmas_mm, width=0.36, label="Sentinel-1 C-band")
    axes[1, 0].bar(x + 0.18, cs_sigmas_mm, width=0.36, label="COSMO-like X-band")
    axes[1, 0].set_xticks(x, [str(v) for v in coh_vals])
    axes[1, 0].set_title("Single-look LOS displacement sigma")
    axes[1, 0].set_xlabel("Coherence")
    axes[1, 0].set_ylabel("Sigma (mm)")
    axes[1, 0].legend(fontsize=8)

    for sensor_name, marker in [("sentinel1_actual", "o"), ("cosmo_like_5000m", "s"), ("cosmo_like_42000m", "^")]:
        rows = [row for row in psf_rows if row["sensor"] == sensor_name and row["sound_speed_mps"] == 1500.0]
        rows = sorted(rows, key=lambda item: item["vibration_hz"])
        axes[1, 1].plot(
            [row["vibration_hz"] for row in rows],
            [row["fwhm_m"] for row in rows],
            marker=marker,
            label=sensor_name,
        )
    axes[1, 1].set_title("Theoretical PSF width")
    axes[1, 1].set_xlabel("Vibration frequency (Hz)")
    axes[1, 1].set_ylabel("PSF FWHM (m)")
    axes[1, 1].set_yscale("log")
    axes[1, 1].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(output_dir / "feasibility.png", dpi=180)
    plt.close(fig)

    summary = {
        "scene": {
            "target_debug": target_debug,
            "slant_range_m": slant_range_m,
            "sentinel1_actual_aperture_span_m": sentinel_aperture_span_m,
            "sentinel1_radar_wavelength_m": radar_wavelength_s1,
        },
        "em_tomosar_limit": em_summary,
        "acoustic_reference_case": {
            "sound_speed_mps": 1500.0,
            "vibration_hz": 500.0,
            "approx_vertical_resolution_m": acoustic_debug["approx_vertical_resolution_m"],
            "condition_number": acoustic_debug["condition_number"],
            "effective_rank": acoustic_debug["effective_rank"],
        },
        "phase_precision": phase_precision,
        "resolution_rows": resolution_rows,
        "psf_rows": psf_rows,
        "notes": [
            "COSMO-like rows are illustrative aperture sweeps, not scene-derived commercial metadata.",
            "The acoustic model uses linear steering and should be treated as an upper-bound feasibility check.",
            "If the theoretical resolution and PSF remain broad, inversion improvements alone will not recover meter-scale layers.",
        ],
        "geometry_debug": geom_debug,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
