from __future__ import annotations

import json
import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from dataclasses import field
from datetime import datetime
from io import BytesIO
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tifffile

from sardt import Geometry, SubApertureConfig, build_doppler_subapertures, compute_kz, compute_tomogram_for_pixels


C = 299_792_458.0
TARGET_LAT = 29.9792
TARGET_LON = 31.1342


def parse_utc(text: str) -> datetime:
    return datetime.fromisoformat(text.replace("Z", ""))


def dt_seconds(dt: datetime) -> float:
    return dt.timestamp()


def geodetic_to_ecef(lat_deg: float, lon_deg: float, h_m: float) -> np.ndarray:
    a = 6378137.0
    e2 = 6.69437999014e-3
    lat = math.radians(lat_deg)
    lon = math.radians(lon_deg)
    sin_lat = math.sin(lat)
    cos_lat = math.cos(lat)
    sin_lon = math.sin(lon)
    cos_lon = math.cos(lon)
    n = a / math.sqrt(1.0 - e2 * sin_lat * sin_lat)
    x = (n + h_m) * cos_lat * cos_lon
    y = (n + h_m) * cos_lat * sin_lon
    z = (n * (1.0 - e2) + h_m) * sin_lat
    return np.array([x, y, z], dtype=np.float64)


def parse_annotation(path: Path) -> dict:
    root = ET.parse(path).getroot()
    grid = []
    for point in root.findall(".//geolocationGridPoint"):
        grid.append(
            {
                "line": int(point.findtext("line")),
                "pixel": int(point.findtext("pixel")),
                "lat": float(point.findtext("latitude")),
                "lon": float(point.findtext("longitude")),
                "height": float(point.findtext("height")),
                "incidence": float(point.findtext("incidenceAngle")),
                "slant_range_time": float(point.findtext("slantRangeTime")),
                "azimuth_time_seconds": dt_seconds(parse_utc(point.findtext("azimuthTime"))),
            }
        )

    orbits = []
    for orbit in root.findall(".//orbit"):
        orbits.append(
            {
                "time_seconds": dt_seconds(parse_utc(orbit.findtext("time"))),
                "position": np.array(
                    [
                        float(orbit.findtext("position/x")),
                        float(orbit.findtext("position/y")),
                        float(orbit.findtext("position/z")),
                    ],
                    dtype=np.float64,
                ),
                "velocity": np.array(
                    [
                        float(orbit.findtext("velocity/x")),
                        float(orbit.findtext("velocity/y")),
                        float(orbit.findtext("velocity/z")),
                    ],
                    dtype=np.float64,
                ),
            }
        )

    def get_text(tag: str) -> str:
        value = root.findtext(f".//{tag}")
        if value is None:
            raise RuntimeError(f"Missing tag {tag} in {path}")
        return value

    return {
        "path": str(path),
        "number_of_lines": int(get_text("numberOfLines")),
        "number_of_samples": int(get_text("numberOfSamples")),
        "range_pixel_spacing": float(get_text("rangePixelSpacing")),
        "azimuth_pixel_spacing": float(get_text("azimuthPixelSpacing")),
        "azimuth_time_interval": float(get_text("azimuthTimeInterval")),
        "slant_range_time": float(get_text("slantRangeTime")),
        "incidence_mid_swath_deg": float(get_text("incidenceAngleMidSwath")),
        "range_sampling_rate": float(get_text("rangeSamplingRate")),
        "radar_frequency": float(get_text("radarFrequency")),
        "prf": float(get_text("prf")),
        "azimuth_steering_rate": float(get_text("azimuthSteeringRate")),
        "product_first_line_time_seconds": dt_seconds(parse_utc(get_text("productFirstLineUtcTime"))),
        "product_last_line_time_seconds": dt_seconds(parse_utc(get_text("productLastLineUtcTime"))),
        "grid_points": grid,
        "orbits": orbits,
        "azimuth_fm_rate_list": [
            {
                "azimuth_time_seconds": dt_seconds(parse_utc(node.findtext("azimuthTime"))),
                "t0": float(node.findtext("t0")),
                "coeffs": [float(x) for x in node.findtext("azimuthFmRatePolynomial").split()],
            }
            for node in root.findall(".//azimuthFmRateList/azimuthFmRate")
        ],
        "dc_estimate_list": [
            {
                "azimuth_time_seconds": dt_seconds(parse_utc(node.findtext("azimuthTime"))),
                "t0": float(node.findtext("t0")),
                "data_coeffs": [float(x) for x in node.findtext("dataDcPolynomial").split()],
                "geometry_coeffs": [float(x) for x in node.findtext("geometryDcPolynomial").split()],
            }
            for node in root.findall(".//dcEstimateList/dcEstimate")
        ],
        "burst_list": [
            {
                "azimuth_time_seconds": dt_seconds(parse_utc(node.findtext("azimuthTime"))),
                "azimuth_anx_time": float(node.findtext("azimuthAnxTime")),
                "first_valid_sample": [int(x) for x in node.findtext("firstValidSample").split()],
                "last_valid_sample": [int(x) for x in node.findtext("lastValidSample").split()],
            }
            for node in root.findall(".//burstList/burst")
        ],
    }


def choose_best_swath(annotation_dir: Path) -> tuple[Path, dict, dict]:
    best = None
    best_path = None
    best_meta = None
    for path in sorted(annotation_dir.glob("*vv*.xml")):
        meta = parse_annotation(path)
        for point in meta["grid_points"]:
            distance = (point["lat"] - TARGET_LAT) ** 2 + (point["lon"] - TARGET_LON) ** 2
            if best is None or distance < best["distance"]:
                best = {"distance": distance, **point}
                best_path = path
                best_meta = meta
    if best is None or best_path is None or best_meta is None:
        raise RuntimeError("No VV annotation found")
    return best_path, best_meta, best


def estimate_target_line_sample(meta: dict, target_lat: float, target_lon: float, neighbors: int = 16) -> tuple[float, float]:
    points = np.array([[p["lat"], p["lon"], p["line"], p["pixel"]] for p in meta["grid_points"]], dtype=np.float64)
    idx = np.argsort((points[:, 0] - target_lat) ** 2 + (points[:, 1] - target_lon) ** 2)[:neighbors]
    local = points[idx]
    design = np.c_[local[:, 0], local[:, 1], np.ones(len(local))]
    line_coef = np.linalg.lstsq(design, local[:, 2], rcond=None)[0]
    pixel_coef = np.linalg.lstsq(design, local[:, 3], rcond=None)[0]
    return float(np.dot([target_lat, target_lon, 1.0], line_coef)), float(np.dot([target_lat, target_lon, 1.0], pixel_coef))


def clip_target_to_valid_burst(meta: dict, line: float, pixel: float) -> tuple[int, int, dict]:
    lines_per_burst = len(meta["burst_list"][0]["first_valid_sample"])
    burst_index = int(np.clip(math.floor(line / lines_per_burst), 0, len(meta["burst_list"]) - 1))
    burst = meta["burst_list"][burst_index]
    within = int(np.clip(round(line - burst_index * lines_per_burst), 0, lines_per_burst - 1))
    first_valid = burst["first_valid_sample"][within]
    last_valid = burst["last_valid_sample"][within]
    pixel_i = int(round(pixel))
    if first_valid >= 0 and last_valid >= 0:
        pixel_i = int(np.clip(pixel_i, first_valid + 16, last_valid - 16))
    line_i = int(np.clip(round(line), burst_index * lines_per_burst + 16, min((burst_index + 1) * lines_per_burst - 17, meta["number_of_lines"] - 1)))
    return line_i, pixel_i, {"burst_index": burst_index, "line_within_burst": within, "first_valid_sample": first_valid, "last_valid_sample": last_valid}


def interpolate_orbit(orbits: list[dict], t_seconds: float) -> tuple[np.ndarray, np.ndarray]:
    ts = np.array([o["time_seconds"] for o in orbits], dtype=np.float64)
    positions = np.stack([o["position"] for o in orbits], axis=0)
    velocities = np.stack([o["velocity"] for o in orbits], axis=0)
    pos = np.array([np.interp(t_seconds, ts, positions[:, i]) for i in range(3)], dtype=np.float64)
    vel = np.array([np.interp(t_seconds, ts, velocities[:, i]) for i in range(3)], dtype=np.float64)
    return pos, vel


def interpolate_target(meta: dict, line: int, pixel: int) -> dict:
    weighted = []
    for p in meta["grid_points"]:
        d = math.hypot(line - p["line"], pixel - p["pixel"])
        w = 1.0 / max(d, 1e-6)
        weighted.append((w, p))
    weighted.sort(key=lambda item: item[0], reverse=True)
    nearest = weighted[:8]
    wsum = sum(w for w, _ in nearest)

    def avg(key: str) -> float:
        return sum(w * p[key] for w, p in nearest) / wsum

    return {
        "lat": avg("lat"),
        "lon": avg("lon"),
        "height": avg("height"),
        "incidence_deg": avg("incidence"),
        "slant_range_time": avg("slant_range_time"),
        "azimuth_time_seconds": meta["product_first_line_time_seconds"] + line * meta["azimuth_time_interval"],
    }


def interpolate_azimuth_fm_rate(meta: dict, t_seconds: float, slant_range_time: float) -> float:
    nearest = min(meta["azimuth_fm_rate_list"], key=lambda item: abs(item["azimuth_time_seconds"] - t_seconds))
    delta_tau = slant_range_time - nearest["t0"]
    c0, c1, c2 = nearest["coeffs"]
    return c0 + c1 * delta_tau + c2 * delta_tau * delta_tau


def evaluate_polynomial(coeffs: list[float], x: float) -> float:
    return float(sum(coeff * (x ** power) for power, coeff in enumerate(coeffs)))


def interpolate_dc_estimate(meta: dict, t_seconds: float, slant_range_time: float) -> dict:
    nearest = min(meta["dc_estimate_list"], key=lambda item: abs(item["azimuth_time_seconds"] - t_seconds))
    delta_tau = slant_range_time - nearest["t0"]
    return {
        "azimuth_time_seconds": nearest["azimuth_time_seconds"],
        "data_dc_hz": evaluate_polynomial(nearest["data_coeffs"], delta_tau),
        "geometry_dc_hz": evaluate_polynomial(nearest["geometry_coeffs"], delta_tau),
    }


def compute_effective_doppler_rate(meta: dict, wavelength: float, los_ref: np.ndarray, ref_vel: np.ndarray, azimuth_fm_rate: float) -> dict:
    az_velocity = ref_vel - np.dot(ref_vel, los_ref) * los_ref
    az_speed = float(np.linalg.norm(az_velocity))
    steering_rate_rad_s = math.radians(meta["azimuth_steering_rate"])
    steering_doppler_rate = -2.0 * az_speed * steering_rate_rad_s / wavelength
    denominator = azimuth_fm_rate - steering_doppler_rate
    if abs(denominator) < 1e-9:
        effective_rate = azimuth_fm_rate
    else:
        effective_rate = azimuth_fm_rate * steering_doppler_rate / denominator
    if not np.isfinite(effective_rate) or abs(effective_rate) < 1e-6:
        effective_rate = azimuth_fm_rate
    return {
        "azimuth_speed_mps": az_speed,
        "steering_rate_rad_per_s": steering_rate_rad_s,
        "steering_doppler_rate_hz_per_s": steering_doppler_rate,
        "effective_doppler_rate_hz_per_s": effective_rate,
    }


def compute_subaperture_centers(n_azimuth: int, cfg: SubApertureConfig) -> list[int]:
    centers = []
    start = cfg.width_bins // 2
    end = n_azimuth - cfg.width_bins // 2
    for c in range(start, end, cfg.step_bins):
        centers.append(c)
        if len(centers) >= cfg.n_apertures:
            break
    return centers


def build_geometry(meta: dict, target: dict, n_azimuth: int, cfg: SubApertureConfig) -> tuple[Geometry, dict]:
    wavelength = C / meta["radar_frequency"]
    incidence = math.radians(target["incidence_deg"])
    slant_range_m = 0.5 * C * target["slant_range_time"]
    target_ecef = geodetic_to_ecef(target["lat"], target["lon"], target["height"])
    sub_centers = compute_subaperture_centers(n_azimuth, cfg)
    center_axis = 0.5 * (n_azimuth - 1)
    ka = interpolate_azimuth_fm_rate(meta, target["azimuth_time_seconds"], target["slant_range_time"])
    dc = interpolate_dc_estimate(meta, target["azimuth_time_seconds"], target["slant_range_time"])
    ref_pos, ref_vel = interpolate_orbit(meta["orbits"], target["azimuth_time_seconds"])
    los_ref = target_ecef - ref_pos
    los_ref /= np.linalg.norm(los_ref)
    az_dir = ref_vel - np.dot(ref_vel, los_ref) * los_ref
    az_dir /= np.linalg.norm(az_dir)
    cross_track_dir = np.cross(los_ref, az_dir)
    cross_track_dir /= np.linalg.norm(cross_track_dir)
    doppler_rates = compute_effective_doppler_rate(meta, wavelength, los_ref, ref_vel, ka)
    kt = doppler_rates["effective_doppler_rate_hz_per_s"]

    baselines = []
    sub_times = []
    sub_freqs = []
    absolute_freqs = []
    for center in sub_centers:
        f_rel = (center - center_axis) * meta["prf"] / n_azimuth
        f_abs = dc["data_dc_hz"] + f_rel
        t_offset = f_rel / kt
        t_sub = target["azimuth_time_seconds"] + t_offset
        pos_k, _ = interpolate_orbit(meta["orbits"], t_sub)
        baseline = pos_k - ref_pos
        baseline_proj = baseline - np.dot(baseline, los_ref) * los_ref
        baselines.append(float(np.dot(baseline_proj, cross_track_dir)))
        sub_times.append(t_sub)
        sub_freqs.append(f_rel)
        absolute_freqs.append(f_abs)

    geom = Geometry(
        wavelength=wavelength,
        slant_ranges=np.full(len(sub_centers), slant_range_m, dtype=np.float64),
        baselines_perp=np.asarray(baselines, dtype=np.float64),
        incidence_angles=np.full(len(sub_centers), incidence, dtype=np.float64),
    )
    debug = {
        "subaperture_centers": sub_centers,
        "subaperture_relative_doppler_hz": sub_freqs,
        "subaperture_absolute_doppler_hz": absolute_freqs,
        "subaperture_times_seconds": sub_times,
        "baselines_perp_m": baselines,
        "slant_range_m": slant_range_m,
        "incidence_deg": target["incidence_deg"],
        "azimuth_fm_rate_hz_per_s": ka,
        "dc_estimate": dc,
        **doppler_rates,
    }
    return geom, debug


def load_chip(
    measurement_path: Path,
    center_line: int,
    center_pixel: int,
    half_lines: int,
    half_pixels: int,
    line_limits: tuple[int, int] | None = None,
    pixel_limits: tuple[int, int] | None = None,
) -> tuple[np.ndarray, tuple[int, int], tuple[int, int]]:
    with tifffile.TiffFile(measurement_path) as tf:
        arr = tf.pages[0].asarray()
    lines, pixels = arr.shape
    line_min, line_max = (0, lines) if line_limits is None else line_limits
    pixel_min, pixel_max = (0, pixels) if pixel_limits is None else pixel_limits
    line0 = max(line_min, center_line - half_lines)
    line1 = min(line_max, center_line + half_lines)
    pixel0 = max(pixel_min, center_pixel - half_pixels)
    pixel1 = min(pixel_max, center_pixel + half_pixels)
    chip = np.asarray(arr[line0:line1, pixel0:pixel1])
    return chip, (line0, line1), (pixel0, pixel1)


def render_magnitude_png(chip_native: np.ndarray) -> bytes:
    magnitude = np.abs(chip_native)
    mag_db = 20.0 * np.log10(magnitude / (magnitude.max() + 1e-12) + 1e-6)
    fig, ax = plt.subplots(figsize=(12, max(3, 12 * chip_native.shape[0] / chip_native.shape[1])))
    ax.imshow(mag_db, cmap="gray", aspect="auto")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout(pad=0)
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=140, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    return buf.getvalue()


@dataclass
class PreparedDataset:
    annotation_path: Path
    measurement_path: Path
    meta: dict
    nearest_grid_point: dict
    estimated_line_sample: tuple[float, float]
    burst_debug: dict
    chip_native: np.ndarray
    chip: np.ndarray
    line_bounds: tuple[int, int]
    pixel_bounds: tuple[int, int]
    cfg: SubApertureConfig
    sub_slcs: list[np.ndarray]
    z_grid: np.ndarray
    default_local_point: tuple[int, int]
    point_cache: dict[tuple[int, int], tuple[np.ndarray, dict]] = field(default_factory=dict)

    def to_summary(self) -> dict:
        return {
            "annotation_path": str(self.annotation_path),
            "measurement_path": str(self.measurement_path),
            "chip_line_bounds": list(self.line_bounds),
            "chip_pixel_bounds": list(self.pixel_bounds),
            "native_chip_shape": list(self.chip_native.shape),
            "chip_shape": list(self.chip.shape),
            "nearest_grid_point": self.nearest_grid_point,
            "estimated_target_line_sample": list(self.estimated_line_sample),
            "burst_debug": self.burst_debug,
            "default_local_point": list(self.default_local_point),
            "z_grid": self.z_grid.tolist(),
        }


def prepare_giza_dataset(
    subset_dir: Path = Path("data/processed/giza_safe_subset"),
    target_lat: float = TARGET_LAT,
    target_lon: float = TARGET_LON,
    cfg: SubApertureConfig | None = None,
    z_grid: np.ndarray | None = None,
    half_lines: int = 512,
    half_pixels: int = 1024,
) -> PreparedDataset:
    annotation_dir = subset_dir / "annotation"
    measurement_dir = subset_dir / "measurement"
    annotation_path, meta, nearest = choose_best_swath(annotation_dir)
    measurement_path = measurement_dir / annotation_path.name.replace(".xml", ".tiff")
    est_line, est_pixel = estimate_target_line_sample(meta, target_lat, target_lon)
    target_line, target_pixel, burst_debug = clip_target_to_valid_burst(meta, est_line, est_pixel)
    lines_per_burst = len(meta["burst_list"][0]["first_valid_sample"])
    burst_start = burst_debug["burst_index"] * lines_per_burst
    burst_stop = min((burst_debug["burst_index"] + 1) * lines_per_burst, meta["number_of_lines"])
    chip_native, line_bounds, pixel_bounds = load_chip(
        measurement_path,
        center_line=target_line,
        center_pixel=target_pixel,
        half_lines=half_lines,
        half_pixels=half_pixels,
        line_limits=(burst_start + 16, burst_stop - 16),
        pixel_limits=(max(0, burst_debug["first_valid_sample"] + 16), min(meta["number_of_samples"], burst_debug["last_valid_sample"] - 16)),
    )
    chip = chip_native.T
    cfg = cfg or SubApertureConfig(width_bins=192, step_bins=96, n_apertures=9, taper="hann")
    sub_slcs = build_doppler_subapertures(chip, cfg)
    z_grid = np.linspace(-30.0, 30.0, 121) if z_grid is None else np.asarray(z_grid, dtype=np.float64)
    local_line = target_line - line_bounds[0]
    local_pixel = target_pixel - pixel_bounds[0]
    return PreparedDataset(
        annotation_path=annotation_path,
        measurement_path=measurement_path,
        meta=meta,
        nearest_grid_point=nearest,
        estimated_line_sample=(est_line, est_pixel),
        burst_debug=burst_debug,
        chip_native=chip_native,
        chip=chip,
        line_bounds=line_bounds,
        pixel_bounds=pixel_bounds,
        cfg=cfg,
        sub_slcs=sub_slcs,
        z_grid=z_grid,
        default_local_point=(local_line, local_pixel),
    )


def local_to_global(dataset: PreparedDataset, local_line: int, local_sample: int) -> tuple[int, int]:
    global_line = dataset.line_bounds[0] + local_line
    global_pixel = dataset.pixel_bounds[0] + local_sample
    return global_line, global_pixel


def compute_point_tomogram(dataset: PreparedDataset, local_line: int, local_sample: int) -> tuple[np.ndarray, dict]:
    return compute_point_tomogram_with_options(dataset, local_line, local_sample)


def compute_point_tomogram_with_options(
    dataset: PreparedDataset,
    local_line: int,
    local_sample: int,
    measurement_mode: str = "patch",
    patch_radius: int = 2,
    inversion_mode: str = "tikhonov",
    alpha: float = 1e-2,
    sparsity: int = 3,
    lam: float = 1e-2,
    n_iter: int = 80,
    preprocess_mode: str = "none",
) -> tuple[np.ndarray, dict]:
    cache_key = (
        int(local_line),
        int(local_sample),
        measurement_mode,
        int(patch_radius),
        inversion_mode,
        float(alpha),
        int(sparsity),
        float(lam),
        int(n_iter),
        preprocess_mode,
    )
    cached = dataset.point_cache.get(cache_key)
    if cached is not None:
        return cached
    global_line, global_pixel = local_to_global(dataset, local_line, local_sample)
    target = interpolate_target(dataset.meta, global_line, global_pixel)
    geom, debug = build_geometry(dataset.meta, target, dataset.chip.shape[1], dataset.cfg)
    h = compute_tomogram_for_pixels(
        sub_slcs=dataset.sub_slcs,
        pixels=[(local_sample, local_line)],
        geom=geom,
        z_grid=dataset.z_grid,
        patch_radius=patch_radius,
        alpha=alpha,
        measurement_mode=measurement_mode,
        inversion_mode=inversion_mode,
        sparsity=sparsity,
        lam=lam,
        n_iter=n_iter,
        preprocess_mode=preprocess_mode,
    )[0]
    kz = compute_kz(geom)
    result = (
        h,
        {
            "target": target,
            "geometry": debug,
            "global_line_pixel": [global_line, global_pixel],
            "kz_rad_per_m": kz.tolist(),
            "kz_span_rad_per_m": float(kz.max() - kz.min()) if len(kz) else 0.0,
        },
    )
    dataset.point_cache[cache_key] = result
    return result


def sample_line_points(x0: float, y0: float, x1: float, y1: float, n_samples: int) -> np.ndarray:
    xs = np.linspace(x0, x1, n_samples)
    ys = np.linspace(y0, y1, n_samples)
    return np.stack([ys, xs], axis=1)


def compute_vertical_slice(
    dataset: PreparedDataset,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    n_samples: int = 64,
    measurement_mode: str = "patch",
    patch_radius: int = 2,
    inversion_mode: str = "tikhonov",
    alpha: float = 1e-2,
    sparsity: int = 3,
    lam: float = 1e-2,
    n_iter: int = 80,
    preprocess_mode: str = "none",
) -> tuple[np.ndarray, dict]:
    points = sample_line_points(x0, y0, x1, y1, n_samples)
    slice_matrix = np.empty((len(points), len(dataset.z_grid)), dtype=np.float64)
    debug_points = []
    for i, (line_f, sample_f) in enumerate(points):
        local_line = int(np.clip(round(line_f), 2, dataset.chip_native.shape[0] - 3))
        local_sample = int(np.clip(round(sample_f), 2, dataset.chip_native.shape[1] - 3))
        h, dbg = compute_point_tomogram_with_options(
            dataset,
            local_line,
            local_sample,
            measurement_mode=measurement_mode,
            patch_radius=patch_radius,
            inversion_mode=inversion_mode,
            alpha=alpha,
            sparsity=sparsity,
            lam=lam,
            n_iter=n_iter,
            preprocess_mode=preprocess_mode,
        )
        slice_matrix[i] = np.abs(h)
        if i in {0, len(points) // 2, len(points) - 1}:
            debug_points.append({"index": i, "local_line_sample": [local_line, local_sample], **dbg})
    debug = {
        "line": {"x0": x0, "y0": y0, "x1": x1, "y1": y1, "n_samples": n_samples},
        "sampled_debug_points": debug_points,
    }
    return slice_matrix, debug


def render_vertical_slice_png(slice_matrix: np.ndarray, z_grid: np.ndarray) -> bytes:
    fig, ax = plt.subplots(figsize=(12, 5))
    vmax = np.percentile(slice_matrix, 99)
    ax.imshow(
        slice_matrix.T,
        cmap="magma",
        aspect="auto",
        origin="lower",
        vmin=0.0,
        vmax=max(vmax, 1e-6),
        extent=[0, slice_matrix.shape[0] - 1, float(z_grid[0]), float(z_grid[-1])],
    )
    ax.set_xlabel("Line Distance Sample")
    ax.set_ylabel("Depth z (m)")
    ax.set_title("Vertical Tomographic Slice")
    fig.tight_layout()
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=160)
    plt.close(fig)
    return buf.getvalue()


def write_dataset_summary(dataset: PreparedDataset, output_path: Path) -> None:
    output_path.write_text(json.dumps(dataset.to_summary(), indent=2))
