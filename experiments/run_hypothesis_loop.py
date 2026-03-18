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

from giza_backend import TARGET_LAT, TARGET_LON, compute_vertical_slice, prepare_giza_dataset
from sardt import SubApertureConfig, build_steering_matrix, compute_kz


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    width_bins: int
    step_bins: int
    n_apertures: int
    z_min: float
    z_max: float
    z_bins: int
    measurement_mode: str
    patch_radius: int
    inversion_mode: str
    alpha: float = 1e-2
    sparsity: int = 3
    lam: float = 1e-2
    n_iter: int = 80
    preprocess_mode: str = "none"


def merged_config(base: ExperimentConfig, name: str, **overrides) -> ExperimentConfig:
    params = asdict(base)
    params.update(overrides)
    params["name"] = name
    return ExperimentConfig(**params)


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


def local_maxima(y: np.ndarray, min_value: float) -> list[int]:
    peaks = []
    for i in range(1, len(y) - 1):
        if y[i] >= y[i - 1] and y[i] >= y[i + 1] and y[i] >= min_value:
            peaks.append(i)
    return peaks


def band_width(profile: np.ndarray, peak_idx: int) -> int:
    peak = profile[peak_idx]
    threshold = 0.5 * peak
    left = peak_idx
    right = peak_idx
    while left > 0 and profile[left - 1] >= threshold:
        left -= 1
    while right < len(profile) - 1 and profile[right + 1] >= threshold:
        right += 1
    return right - left + 1


def slice_metrics(slice_matrix: np.ndarray, z_grid: np.ndarray) -> dict:
    avg_profile = slice_matrix.mean(axis=0)
    avg_profile /= avg_profile.max() + 1e-12
    floor = float(np.quantile(avg_profile, 0.75))
    peaks = local_maxima(avg_profile, min_value=max(0.35, floor))
    if not peaks:
        peaks = [int(np.argmax(avg_profile))]
    peaks = sorted(peaks, key=lambda idx: avg_profile[idx], reverse=True)[:3]

    supports = []
    widths = []
    sharpnesses = []
    depth_weights = []
    for peak_idx in peaks:
        lo = max(0, peak_idx - 1)
        hi = min(slice_matrix.shape[1], peak_idx + 2)
        band_energy = slice_matrix[:, lo:hi].sum(axis=1)
        col_energy = slice_matrix.sum(axis=1) + 1e-12
        support = float(np.mean(band_energy / col_energy))
        width = band_width(avg_profile, peak_idx)
        context = avg_profile[max(0, peak_idx - 4): min(len(avg_profile), peak_idx + 5)]
        sharpness = float(avg_profile[peak_idx] / (np.mean(context) + 1e-12))
        depth_norm = abs(float(z_grid[peak_idx])) / max(abs(float(z_grid[0])), abs(float(z_grid[-1])), 1e-12)
        depth_weight = max(0.0, 4.0 * depth_norm * (1.0 - depth_norm))
        supports.append(support)
        widths.append(width)
        sharpnesses.append(sharpness)
        depth_weights.append(depth_weight)

    peakiness = float(avg_profile.max() / (avg_profile.mean() + 1e-12))
    entropy = float(-np.sum(avg_profile * np.log(avg_profile + 1e-12)) / math.log(len(avg_profile)))
    best_band_score = max(
        support * sharpness / max(width, 1)
        for support, sharpness, width in zip(supports, sharpnesses, widths)
    )
    best_subsurface_score = max(
        support * sharpness * depth_weight / max(width, 1)
        for support, sharpness, width, depth_weight in zip(supports, sharpnesses, widths, depth_weights)
    )
    return {
        "peak_depths_m": [float(z_grid[idx]) for idx in peaks],
        "peak_values": [float(avg_profile[idx]) for idx in peaks],
        "peak_supports": supports,
        "peak_width_bins": widths,
        "peak_sharpness": sharpnesses,
        "peak_depth_weights": depth_weights,
        "peakiness": peakiness,
        "entropy": entropy,
        "best_band_score": float(best_band_score),
        "best_subsurface_score": float(best_subsurface_score),
    }


def steering_metrics(dataset) -> dict:
    local_line, local_sample = dataset.default_local_point
    global_line = dataset.line_bounds[0] + local_line
    global_pixel = dataset.pixel_bounds[0] + local_sample
    target = dataset.meta["grid_points"][0]
    from giza_backend import build_geometry, interpolate_target

    target = interpolate_target(dataset.meta, global_line, global_pixel)
    geom, debug = build_geometry(dataset.meta, target, dataset.chip.shape[1], dataset.cfg)
    A = build_steering_matrix(geom, dataset.z_grid)
    s = np.linalg.svd(A, compute_uv=False)
    kz = compute_kz(geom)
    kz_span = float(kz.max() - kz.min()) if len(kz) else 0.0
    vertical_resolution = float((2.0 * np.pi) / kz_span) if kz_span > 1e-12 else float("inf")
    return {
        "n_subapertures": int(len(dataset.sub_slcs)),
        "kz_span_rad_per_m": kz_span,
        "vertical_resolution_m": vertical_resolution,
        "condition_number": float(s[0] / max(s[-1], 1e-12)),
        "effective_rank": int(np.sum(s / s[0] > 1e-3)),
        "baselines_perp_m": [float(x) for x in geom.baselines_perp],
        "geometry_debug": debug,
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
        {"name": "target_horizontal", "x0": 0.2 * w, "y0": y0, "x1": 0.8 * w, "y1": y0},
        {"name": "target_diagonal_up", "x0": 0.2 * w, "y0": min(h - 1, y0 + 80), "x1": 0.8 * w, "y1": max(0, y0 - 80)},
        {"name": "target_diagonal_down", "x0": 0.2 * w, "y0": max(0, y0 - 80), "x1": 0.8 * w, "y1": min(h - 1, y0 + 80)},
        {"name": "strong_horizontal", "x0": max(0, strong_x - 280), "y0": strong_y, "x1": min(w - 1, strong_x + 280), "y1": strong_y},
    ]


def evaluate_config(config: ExperimentConfig, output_dir: Path) -> dict:
    cfg = SubApertureConfig(
        width_bins=config.width_bins,
        step_bins=config.step_bins,
        n_apertures=config.n_apertures,
        taper="hann",
    )
    z_grid = np.linspace(config.z_min, config.z_max, config.z_bins)
    dataset = prepare_giza_dataset(
        target_lat=TARGET_LAT,
        target_lon=TARGET_LON,
        cfg=cfg,
        z_grid=z_grid,
    )
    geom_metrics = steering_metrics(dataset)

    best = None
    all_lines = []
    for line in candidate_lines(dataset):
        line_params = {key: value for key, value in line.items() if key != "name"}
        slice_matrix, debug = compute_vertical_slice(
            dataset,
            n_samples=40,
            measurement_mode=config.measurement_mode,
            patch_radius=config.patch_radius,
            inversion_mode=config.inversion_mode,
            alpha=config.alpha,
            sparsity=config.sparsity,
            lam=config.lam,
            n_iter=config.n_iter,
            preprocess_mode=config.preprocess_mode,
            **line_params,
        )
        metrics = slice_metrics(slice_matrix, dataset.z_grid)
        line_result = {
            "line": line,
            "slice_metrics": metrics,
            "debug": debug,
        }
        all_lines.append(line_result)
        score = (
            metrics["best_subsurface_score"] *
            min(geom_metrics["effective_rank"], 8) /
            math.log10(geom_metrics["condition_number"] + 10.0)
        )
        if best is None or score > best["score"]:
            best = {
                "score": float(score),
                "slice_matrix": slice_matrix,
                "line_result": line_result,
            }

    assert best is not None
    result = {
        "config": asdict(config),
        "geometry_metrics": geom_metrics,
        "best_score": best["score"],
        "best_line": best["line_result"]["line"],
        "best_slice_metrics": best["line_result"]["slice_metrics"],
        "all_line_results": [
            {
                "line": item["line"],
                "slice_metrics": item["slice_metrics"],
            }
            for item in all_lines
        ],
    }

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    vmax = np.percentile(best["slice_matrix"], 99)
    axes[0].imshow(
        best["slice_matrix"].T,
        cmap="magma",
        aspect="auto",
        origin="lower",
        vmin=0.0,
        vmax=max(vmax, 1e-6),
        extent=[0, best["slice_matrix"].shape[0] - 1, float(dataset.z_grid[0]), float(dataset.z_grid[-1])],
    )
    axes[0].set_title(f"{config.name} best slice")
    axes[0].set_xlabel("Along-line sample")
    axes[0].set_ylabel("Depth z (m)")

    avg_profile = best["slice_matrix"].mean(axis=0)
    axes[1].plot(dataset.z_grid, avg_profile / (avg_profile.max() + 1e-12), color="tab:orange")
    for peak_depth in result["best_slice_metrics"]["peak_depths_m"]:
        axes[1].axvline(peak_depth, color="tab:blue", alpha=0.35, linewidth=1.0)
    axes[1].set_title("Average vertical profile")
    axes[1].set_xlabel("Depth z (m)")
    axes[1].set_ylabel("Normalized magnitude")
    fig.tight_layout()
    fig.savefig(output_dir / f"{config.name}.png", dpi=180)
    plt.close(fig)

    (output_dir / f"{config.name}.json").write_text(json.dumps(to_jsonable(result), indent=2))
    return result


def choose_best(results: list[dict]) -> dict:
    return max(results, key=lambda item: item["best_score"])


def main() -> None:
    output_dir = Path("data/processed/hypothesis_loop")
    output_dir.mkdir(parents=True, exist_ok=True)

    current = ExperimentConfig(
        name="baseline",
        width_bins=192,
        step_bins=96,
        n_apertures=9,
        z_min=-30.0,
        z_max=30.0,
        z_bins=121,
        measurement_mode="patch",
        patch_radius=2,
        inversion_mode="tikhonov",
        alpha=1e-2,
    )

    history = []
    baseline = evaluate_config(current, output_dir)
    history.append(baseline)

    stages = [
        (
            "depth_grid",
            [
                ("depth_15m", {"z_min": -15.0, "z_max": 15.0}),
                ("depth_6m", {"z_min": -6.0, "z_max": 6.0}),
                ("depth_3m", {"z_min": -3.0, "z_max": 3.0}),
            ],
        ),
        (
            "subaperture",
            [
                ("subap_128_48", {"width_bins": 128, "step_bins": 48, "n_apertures": 11}),
                ("subap_96_32", {"width_bins": 96, "step_bins": 32, "n_apertures": 15}),
                ("subap_64_24", {"width_bins": 64, "step_bins": 24, "n_apertures": 19}),
            ],
        ),
        (
            "measurement",
            [
                ("phase_raw", {"measurement_mode": "phase", "patch_radius": 0}),
                ("patch_r1", {"patch_radius": 1}),
                ("patch_r0", {"patch_radius": 0}),
            ],
        ),
        (
            "preprocess",
            [
                ("demean", {"preprocess_mode": "demean"}),
                ("remove_surface", {"preprocess_mode": "remove_surface"}),
                ("unit_phase", {"preprocess_mode": "unit_phase"}),
                ("unit_phase_remove_surface", {"preprocess_mode": "unit_phase_remove_surface"}),
            ],
        ),
        (
            "inversion",
            [
                ("tikhonov_lo", {"alpha": 1e-3}),
                ("pinv", {"inversion_mode": "pinv"}),
                ("omp_s2", {"inversion_mode": "omp", "sparsity": 2}),
                ("omp_s3", {"inversion_mode": "omp", "sparsity": 3}),
                ("ista", {"inversion_mode": "ista", "lam": 5e-3, "n_iter": 120}),
            ],
        ),
    ]

    stage_results = []
    for stage_name, candidate_specs in stages:
        candidates = [merged_config(current, name, **overrides) for name, overrides in candidate_specs]
        results = [evaluate_config(candidate, output_dir) for candidate in candidates]
        best_stage = choose_best([history[-1], *results])
        history.extend(results)
        current = ExperimentConfig(**best_stage["config"])
        stage_results.append({"stage": stage_name, "selected": best_stage["config"]["name"]})

    best = choose_best(history)
    summary = {
        "selected_config": best["config"],
        "selected_score": best["best_score"],
        "selected_geometry_metrics": best["geometry_metrics"],
        "selected_slice_metrics": best["best_slice_metrics"],
        "stage_progression": stage_results,
        "all_results": [
            {
                "name": item["config"]["name"],
                "best_score": item["best_score"],
                "vertical_resolution_m": item["geometry_metrics"]["vertical_resolution_m"],
                "condition_number": item["geometry_metrics"]["condition_number"],
                "effective_rank": item["geometry_metrics"]["effective_rank"],
                "peak_depths_m": item["best_slice_metrics"]["peak_depths_m"],
            }
            for item in history
        ],
    }
    (output_dir / "summary.json").write_text(json.dumps(to_jsonable(summary), indent=2))
    print(json.dumps(to_jsonable(summary), indent=2))


if __name__ == "__main__":
    main()
