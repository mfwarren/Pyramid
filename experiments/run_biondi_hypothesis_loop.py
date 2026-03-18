#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import sys
from dataclasses import asdict
from dataclasses import dataclass
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
    solve_acoustic_tomography,
)
from giza_backend import TARGET_LAT, TARGET_LON, prepare_giza_dataset
from sardt import SubApertureConfig


@dataclass(frozen=True)
class BiondiConfig:
    name: str
    width_bins: int
    step_bins: int
    n_apertures: int
    z_min: float
    z_max: float
    z_bins: int
    sound_speed_mps: float
    vibration_hz: float
    steering_mode: str
    observable: str
    pair_stride: int
    series_mode: str
    detrend_mode: str
    patch_radius: int
    search_radius: int
    inversion_mode: str
    alpha: float = 1e-2
    sparsity: int = 3
    lam: float = 1e-2
    n_iter: int = 80
    half_lines: int = 2048
    half_pixels: int = 1024


def merged_config(base: BiondiConfig, name: str, **overrides) -> BiondiConfig:
    params = asdict(base)
    params.update(overrides)
    params["name"] = name
    return BiondiConfig(**params)


def local_maxima(y: np.ndarray, min_value: float) -> list[int]:
    peaks = []
    for i in range(1, len(y) - 1):
        if y[i] >= y[i - 1] and y[i] >= y[i + 1] and y[i] >= min_value:
            peaks.append(i)
    return peaks


def band_width(profile: np.ndarray, peak_idx: int) -> int:
    level = 0.5 * profile[peak_idx]
    left = peak_idx
    right = peak_idx
    while left > 0 and profile[left - 1] >= level:
        left -= 1
    while right < len(profile) - 1 and profile[right + 1] >= level:
        right += 1
    return right - left + 1


def slice_metrics(slice_matrix: np.ndarray, z_grid: np.ndarray) -> dict:
    avg_profile = slice_matrix.mean(axis=0)
    avg_profile /= avg_profile.max() + 1e-12
    peaks = local_maxima(avg_profile, min_value=max(0.35, float(np.quantile(avg_profile, 0.75))))
    if not peaks:
        peaks = [int(np.argmax(avg_profile))]
    peaks = sorted(peaks, key=lambda i: avg_profile[i], reverse=True)[:4]

    band_scores = []
    for idx in peaks:
        lo = max(0, idx - 1)
        hi = min(len(z_grid), idx + 2)
        support = float(np.mean(slice_matrix[:, lo:hi].sum(axis=1) / (slice_matrix.sum(axis=1) + 1e-12)))
        width = band_width(avg_profile, idx)
        depth_abs = abs(float(z_grid[idx]))
        edge_depth = max(abs(float(z_grid[0])), abs(float(z_grid[-1])), 1e-12)
        edge_penalty = max(0.0, 1.0 - (depth_abs / edge_depth) ** 3)
        center_penalty = 0.2 if depth_abs < 0.05 * edge_depth else 1.0
        sharpness = float(avg_profile[idx] / (np.mean(avg_profile[max(0, idx - 4): min(len(avg_profile), idx + 5)]) + 1e-12))
        band_scores.append(
            {
                "depth_m": float(z_grid[idx]),
                "value": float(avg_profile[idx]),
                "support": support,
                "width_bins": width,
                "sharpness": sharpness,
                "score": support * sharpness * edge_penalty * center_penalty / max(width, 1),
            }
        )

    return {
        "peakiness": float(avg_profile.max() / (avg_profile.mean() + 1e-12)),
        "entropy": float(-np.sum(avg_profile * np.log(avg_profile + 1e-12)) / math.log(len(avg_profile))),
        "bands": band_scores,
        "best_subsurface_score": float(max(item["score"] for item in band_scores)),
    }


def candidate_lines(dataset) -> list[dict]:
    h, w = dataset.chip_native.shape
    y0, x0 = dataset.default_local_point
    amp = np.abs(dataset.chip_native)
    search = amp[max(0, y0 - 96): min(h, y0 + 96), max(0, x0 - 256): min(w, x0 + 256)]
    iy, ix = np.unravel_index(np.argmax(search), search.shape)
    strong_y = max(0, y0 - 96) + iy
    strong_x = max(0, x0 - 256) + ix
    return [
        {"name": "target_horizontal", "points": np.stack([np.linspace(y0, y0, 40), np.linspace(max(32, x0 - 320), min(w - 33, x0 + 320), 40)], axis=1)},
        {"name": "strong_horizontal", "points": np.stack([np.linspace(strong_y, strong_y, 40), np.linspace(max(32, strong_x - 320), min(w - 33, strong_x + 320), 40)], axis=1)},
        {"name": "target_diag_up", "points": np.stack([np.linspace(min(h - 33, y0 + 80), max(32, y0 - 80), 40), np.linspace(max(32, x0 - 240), min(w - 33, x0 + 240), 40)], axis=1)},
    ]


def to_jsonable(value):
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [to_jsonable(v) for v in value]
    if isinstance(value, tuple):
        return [to_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def evaluate_config(config: BiondiConfig, output_dir: Path) -> dict:
    dataset = prepare_giza_dataset(
        target_lat=TARGET_LAT,
        target_lon=TARGET_LON,
        cfg=SubApertureConfig(width_bins=config.width_bins, step_bins=config.step_bins, n_apertures=config.n_apertures, taper="hann"),
        z_grid=np.linspace(config.z_min, config.z_max, config.z_bins),
        half_lines=config.half_lines,
        half_pixels=config.half_pixels,
    )
    track_cfg = PixelTrackConfig(patch_radius=config.patch_radius, search_radius=config.search_radius)
    best = None
    all_lines = []
    for line in candidate_lines(dataset):
        slice_matrix = np.empty((len(line["points"]), len(dataset.z_grid)), dtype=np.float64)
        debug_points = []
        steering_debug = None
        for i, (line_f, sample_f) in enumerate(line["points"]):
            local_line = int(np.clip(round(line_f), 16, dataset.chip_native.shape[0] - 17))
            local_sample = int(np.clip(round(sample_f), 16, dataset.chip_native.shape[1] - 17))
            target, target_debug = local_to_target(dataset, local_line, local_sample)
            y, tracking_debug, observation_indices = build_displacement_series(
                dataset.sub_slcs,
                (local_sample, local_line),
                track_cfg,
                observable=config.observable,
                pair_stride=config.pair_stride,
                series_mode=config.series_mode,
                detrend_mode=config.detrend_mode,
            )
            A, steering_debug = build_acoustic_steering_matrix(
                dataset,
                target,
                sound_speed_mps=config.sound_speed_mps,
                vibration_hz=config.vibration_hz,
                steering_mode=config.steering_mode,
            )
            A_used = A[np.asarray(observation_indices, dtype=np.int64)]
            h = solve_acoustic_tomography(
                y,
                A_used,
                inversion_mode=config.inversion_mode,
                alpha=config.alpha,
                sparsity=config.sparsity,
                lam=config.lam,
                n_iter=config.n_iter,
            )
            slice_matrix[i] = np.abs(h)
            if i in {0, len(line["points"]) // 2, len(line["points"]) - 1}:
                debug_points.append(
                    {
                        "index": i,
                        "local_line_sample": [local_line, local_sample],
                        "target_debug": target_debug,
                        "tracking_head": tracking_debug[:4],
                    }
                )

        metrics = slice_metrics(slice_matrix, dataset.z_grid)
        line_result = {
            "line_name": line["name"],
            "slice_metrics": metrics,
            "steering_debug": steering_debug,
            "debug_points": debug_points,
        }
        all_lines.append(line_result)
        score = metrics["best_subsurface_score"] / math.log10(steering_debug["condition_number"] + 10.0)
        if best is None or score > best["score"]:
            best = {
                "score": float(score),
                "slice_matrix": slice_matrix,
                "line_result": line_result,
                "line_points": line["points"],
                "dataset": dataset,
            }

    assert best is not None
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    mag = np.abs(best["dataset"].chip_native)
    mag_db = 20.0 * np.log10(mag / (mag.max() + 1e-12) + 1e-6)
    axes[0].imshow(mag_db, cmap="gray", aspect="auto")
    axes[0].plot(best["line_points"][:, 1], best["line_points"][:, 0], color="tab:orange", linewidth=2)
    axes[0].set_title(f"{config.name} map line")
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
        extent=[0, best["slice_matrix"].shape[0] - 1, float(best["dataset"].z_grid[0]), float(best["dataset"].z_grid[-1])],
    )
    axes[1].set_title(f"{config.name} best slice")
    axes[1].set_xlabel("Along-line sample")
    axes[1].set_ylabel("Depth (m)")
    fig.tight_layout()
    fig.savefig(output_dir / f"{config.name}.png", dpi=180)
    plt.close(fig)

    result = {
        "config": asdict(config),
        "best_score": best["score"],
        "best_line_result": best["line_result"],
        "all_line_results": [
            {"line_name": item["line_name"], "slice_metrics": item["slice_metrics"]}
            for item in all_lines
        ],
    }
    (output_dir / f"{config.name}.json").write_text(json.dumps(to_jsonable(result), indent=2))
    return result


def choose_best(results: list[dict]) -> dict:
    return max(results, key=lambda item: item["best_score"])


def main() -> None:
    output_dir = Path("data/processed/biondi_hypothesis_loop")
    output_dir.mkdir(parents=True, exist_ok=True)

    current = BiondiConfig(
        name="baseline",
        width_bins=96,
        step_bins=16,
        n_apertures=48,
        z_min=-500.0,
        z_max=500.0,
        z_bins=241,
        sound_speed_mps=3000.0,
        vibration_hz=200.0,
        steering_mode="linear",
        observable="complex",
        pair_stride=1,
        series_mode="cumulative",
        detrend_mode="mean",
        patch_radius=8,
        search_radius=4,
        inversion_mode="pinv",
    )

    history = []
    baseline = evaluate_config(current, output_dir)
    history.append(baseline)

    stages = [
        ("steering", [("hyperbolic", {"steering_mode": "hyperbolic"}), ("quadratic", {"steering_mode": "quadratic"})]),
        ("observable", [("azimuth_phase", {"observable": "azimuth_phase"}), ("phase", {"observable": "phase"}), ("azimuth", {"observable": "azimuth"}), ("range", {"observable": "range"})]),
        ("pairing", [("stride2", {"pair_stride": 2}), ("stride4", {"pair_stride": 4}), ("incremental", {"series_mode": "incremental"})]),
        ("detrend", [("linear_detrend", {"detrend_mode": "linear"}), ("first_difference", {"detrend_mode": "first_difference"})]),
        ("acoustic", [("fast_500hz", {"sound_speed_mps": 3000.0, "vibration_hz": 500.0}), ("slow_500hz", {"sound_speed_mps": 1500.0, "vibration_hz": 500.0}), ("slow_1000hz", {"sound_speed_mps": 1500.0, "vibration_hz": 1000.0})]),
        ("subaperture", [("dense_64_8_80", {"width_bins": 64, "step_bins": 8, "n_apertures": 80}), ("dense_48_8_96", {"width_bins": 48, "step_bins": 8, "n_apertures": 96}), ("wider_128_16_56", {"width_bins": 128, "step_bins": 16, "n_apertures": 56})]),
        ("inversion", [("tikhonov", {"inversion_mode": "tikhonov", "alpha": 1e-2}), ("omp3", {"inversion_mode": "omp", "sparsity": 3}), ("ista", {"inversion_mode": "ista", "lam": 5e-3, "n_iter": 120})]),
    ]

    stage_progression = []
    for stage_name, candidate_specs in stages:
        candidates = [merged_config(current, name, **overrides) for name, overrides in candidate_specs]
        results = [evaluate_config(candidate, output_dir) for candidate in candidates]
        best_stage = choose_best([history[-1], *results])
        history.extend(results)
        current = BiondiConfig(**best_stage["config"])
        stage_progression.append({"stage": stage_name, "selected": best_stage["config"]["name"]})

    best = choose_best(history)
    summary = {
        "selected_config": best["config"],
        "selected_score": best["best_score"],
        "selected_best_line": best["best_line_result"]["line_name"],
        "selected_slice_metrics": best["best_line_result"]["slice_metrics"],
        "selected_steering_debug": best["best_line_result"]["steering_debug"],
        "stage_progression": stage_progression,
        "all_results": [
            {
                "name": item["config"]["name"],
                "best_score": item["best_score"],
                "best_line": item["best_line_result"]["line_name"],
                "best_peak_depths": [band["depth_m"] for band in item["best_line_result"]["slice_metrics"]["bands"][:3]],
                "condition_number": item["best_line_result"]["steering_debug"]["condition_number"],
                "effective_rank": item["best_line_result"]["steering_debug"]["effective_rank"],
            }
            for item in history
        ],
    }
    (output_dir / "summary.json").write_text(json.dumps(to_jsonable(summary), indent=2))
    print(json.dumps(to_jsonable(summary), indent=2))


if __name__ == "__main__":
    main()
