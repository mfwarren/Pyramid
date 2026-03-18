#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
import zipfile
from pathlib import Path
from typing import Iterable

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sardt import (
    Geometry,
    SubApertureConfig,
    build_doppler_subapertures,
    compute_tomogram_for_pixels,
)


C = 299_792_458.0


def _iter_datasets(group: h5py.Group, prefix: str = "") -> Iterable[tuple[str, h5py.Dataset]]:
    for key, value in group.items():
        path = f"{prefix}/{key}"
        if isinstance(value, h5py.Dataset):
            yield path, value
        elif isinstance(value, h5py.Group):
            yield from _iter_datasets(value, path)


def _load_scalar_value(dataset: h5py.Dataset) -> float | None:
    try:
        value = dataset[()]
    except Exception:
        return None

    if isinstance(value, bytes):
        try:
            return float(value.decode("utf-8"))
        except Exception:
            return None

    arr = np.asarray(value)
    if arr.size != 1:
        return None

    scalar = arr.reshape(-1)[0]
    try:
        return float(scalar)
    except Exception:
        return None


def _center_slices(shape: tuple[int, int], max_rows: int, max_cols: int) -> tuple[slice, slice]:
    rows, cols = shape
    out_rows = min(rows, max_rows)
    out_cols = min(cols, max_cols)
    r0 = (rows - out_rows) // 2
    c0 = (cols - out_cols) // 2
    return slice(r0, r0 + out_rows), slice(c0, c0 + out_cols)


def _find_candidate_dataset(
    h5: h5py.File,
    max_rows: int,
    max_cols: int,
) -> tuple[str, np.ndarray]:
    if "/s_i" in h5 and "/s_q" in h5:
        ds_i = h5["/s_i"]
        ds_q = h5["/s_q"]
        rs, cs = _center_slices(ds_i.shape, max_rows=max_rows, max_cols=max_cols)
        real = ds_i[rs, cs].astype(np.float32)
        imag = ds_q[rs, cs].astype(np.float32)
        return "/s_i + /s_q", real + 1j * imag

    candidates: list[tuple[int, str, h5py.Dataset]] = []

    for path, dataset in _iter_datasets(h5):
        if dataset.ndim < 2:
            continue

        score = 0
        if np.issubdtype(dataset.dtype, np.complexfloating):
            score += 100
        if "slc" in path.lower():
            score += 20
        if "s_i" in path.lower() or "iq" in path.lower():
            score += 5
        if dataset.size > 4_096:
            score += 10

        if score > 0:
            candidates.append((score, path, dataset))

    if not candidates:
        raise RuntimeError("No 2D complex-like datasets found in HDF5 file")

    candidates.sort(key=lambda item: (item[0], item[2].size), reverse=True)
    _, path, dataset = candidates[0]
    rs, cs = _center_slices(dataset.shape[-2:], max_rows=max_rows, max_cols=max_cols)
    if dataset.ndim == 2:
        data = dataset[rs, cs]
    elif dataset.ndim == 3:
        data = dataset[:, rs, cs]
    else:
        data = dataset[()]
    return path, _ensure_complex_2d(data)


def _ensure_complex_2d(data: np.ndarray) -> np.ndarray:
    arr = np.asarray(data)

    if arr.ndim > 2:
        arr = np.squeeze(arr)

    if arr.ndim == 3 and arr.shape[-1] == 2:
        arr = arr[..., 0] + 1j * arr[..., 1]
    elif arr.ndim == 3 and arr.shape[0] == 2:
        arr = arr[0] + 1j * arr[1]

    if arr.ndim != 2:
        raise RuntimeError(f"Expected 2D complex data after squeeze, got shape {arr.shape}")

    if not np.iscomplexobj(arr):
        arr = arr.astype(np.complex64)

    return arr


def _pick_pixel(arr: np.ndarray, margin: int = 24) -> tuple[int, int]:
    mag = np.abs(arr)
    rows, cols = mag.shape
    r0 = min(margin, rows // 4)
    c0 = min(margin, cols // 4)
    core = mag[r0:rows - r0 or rows, c0:cols - c0 or cols]
    rr, cc = np.unravel_index(np.argmax(core), core.shape)
    return int(rr + r0), int(cc + c0)


def _extract_metadata(h5: h5py.File) -> dict[str, float]:
    keys = {
        "carrier_frequency": None,
        "center_frequency": None,
        "radar_frequency": None,
        "incidence_angle": None,
        "incidence_center": None,
        "avg_scene_incidence_angle": None,
        "near_range": None,
        "slant_range": None,
        "slant_range_to_first_pixel": None,
    }

    for path, dataset in _iter_datasets(h5):
        lower = path.lower()
        name = lower.rsplit("/", 1)[-1]
        for key in list(keys):
            if name == key and keys[key] is None:
                keys[key] = _load_scalar_value(dataset)

    frequency = (
        keys["carrier_frequency"]
        or keys["center_frequency"]
        or keys["radar_frequency"]
        or 9.65e9
    )
    incidence_deg = (
        keys["avg_scene_incidence_angle"]
        or keys["incidence_center"]
        or keys["incidence_angle"]
        or 35.0
    )
    slant_range = (
        keys["slant_range"]
        or keys["slant_range_to_first_pixel"]
        or keys["near_range"]
        or 700_000.0
    )

    return {
        "frequency_hz": frequency,
        "incidence_angle_deg": incidence_deg,
        "slant_range_m": slant_range,
    }


def _build_geometry(k: int, metadata: dict[str, float]) -> Geometry:
    wavelength = C / metadata["frequency_hz"]
    incidence_rad = math.radians(metadata["incidence_angle_deg"])

    # The repo research does not yet derive physical sub-aperture baselines from metadata.
    # For this first qualitative run, use a symmetric synthetic baseline sweep.
    baselines = np.linspace(-15.0, 15.0, k)

    return Geometry(
        wavelength=wavelength,
        slant_ranges=np.full(k, metadata["slant_range_m"], dtype=np.float64),
        baselines_perp=baselines.astype(np.float64),
        incidence_angles=np.full(k, incidence_rad, dtype=np.float64),
    )


def _load_slc(
    input_path: Path,
    extract_dir: Path,
    max_rows: int,
    max_cols: int,
) -> tuple[np.ndarray, str, dict[str, float]]:
    h5_path = input_path

    if input_path.suffix.lower() == ".zip":
        with zipfile.ZipFile(input_path) as zf:
            h5_names = [name for name in zf.namelist() if name.lower().endswith((".h5", ".hdf5"))]
            if not h5_names:
                raise RuntimeError("ZIP file does not contain any HDF5 products")
            target = h5_names[0]
            extract_dir.mkdir(parents=True, exist_ok=True)
            h5_path = extract_dir / Path(target).name
            if not h5_path.exists():
                zf.extract(target, path=extract_dir)
                extracted = extract_dir / target
                if extracted != h5_path:
                    extracted.rename(h5_path)

    with h5py.File(h5_path, "r") as h5:
        dataset_path, slc = _find_candidate_dataset(h5, max_rows=max_rows, max_cols=max_cols)
        metadata = _extract_metadata(h5)

    return slc, dataset_path, metadata


def _plot_results(
    slc: np.ndarray,
    sub_slcs: list[np.ndarray],
    pixel: tuple[int, int],
    z_grid: np.ndarray,
    tomogram: np.ndarray,
    output_png: Path,
) -> None:
    amp = np.abs(slc)
    amp_db = 20.0 * np.log10(amp / (np.max(amp) + 1e-12) + 1e-6)

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))

    axes[0, 0].imshow(amp_db, cmap="gray", aspect="auto")
    axes[0, 0].scatter([pixel[1]], [pixel[0]], s=50, c="tab:red", marker="x")
    axes[0, 0].set_title("Input SLC Magnitude (dB)")

    axes[0, 1].imshow(np.angle(slc), cmap="twilight", aspect="auto")
    axes[0, 1].set_title("Input SLC Phase")

    preview = np.hstack([np.abs(img) for img in sub_slcs[: min(4, len(sub_slcs))]])
    axes[1, 0].imshow(20.0 * np.log10(preview / (np.max(preview) + 1e-12) + 1e-6), cmap="magma", aspect="auto")
    axes[1, 0].set_title("First Sub-Aperture Magnitudes (dB)")

    axes[1, 1].plot(z_grid, np.abs(tomogram[0]))
    axes[1, 1].set_title("Tomogram at Selected Pixel")
    axes[1, 1].set_xlabel("Depth z (m)")
    axes[1, 1].set_ylabel("|h(z)|")

    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])
    axes[1, 1].set_xticks(np.linspace(z_grid.min(), z_grid.max(), 5))

    fig.tight_layout()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True, help="Path to .zip or .h5 open SAR product")
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/open_sar_test"))
    parser.add_argument("--max-rows", type=int, default=768)
    parser.add_argument("--max-cols", type=int, default=768)
    parser.add_argument("--width-bins", type=int, default=96)
    parser.add_argument("--step-bins", type=int, default=48)
    parser.add_argument("--n-apertures", type=int, default=9)
    args = parser.parse_args()

    extract_dir = args.output_dir / "extracted"
    slc, dataset_path, metadata = _load_slc(
        args.input,
        extract_dir,
        max_rows=args.max_rows,
        max_cols=args.max_cols,
    )

    cfg = SubApertureConfig(
        width_bins=min(args.width_bins, slc.shape[1]),
        step_bins=max(1, args.step_bins),
        n_apertures=args.n_apertures,
        taper="hann",
    )
    sub_slcs = build_doppler_subapertures(slc, cfg)
    if len(sub_slcs) < 3:
        raise RuntimeError(f"Only built {len(sub_slcs)} sub-apertures; need at least 3 for a meaningful test")

    pixel = _pick_pixel(slc)
    z_grid = np.linspace(-30.0, 30.0, 121)
    geom = _build_geometry(len(sub_slcs), metadata)
    tomogram = compute_tomogram_for_pixels(
        sub_slcs=sub_slcs,
        pixels=[pixel],
        geom=geom,
        z_grid=z_grid,
        patch_radius=2,
        alpha=1e-2,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_png = args.output_dir / "open_sar_preview.png"
    _plot_results(slc, sub_slcs, pixel, z_grid, tomogram, output_png)

    np.savez_compressed(
        args.output_dir / "open_sar_results.npz",
        slc=slc,
        pixel=np.asarray(pixel),
        z_grid=z_grid,
        tomogram=tomogram,
    )

    summary = {
        "input": str(args.input),
        "dataset_path": dataset_path,
        "slc_shape": list(map(int, slc.shape)),
        "selected_pixel": list(map(int, pixel)),
        "metadata": metadata,
        "n_subapertures": len(sub_slcs),
        "preview_png": str(output_png),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
