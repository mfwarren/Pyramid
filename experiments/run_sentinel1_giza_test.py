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

from giza_backend import compute_point_tomogram, prepare_giza_dataset


def plot_preview(
    chip_native: np.ndarray,
    sub_slcs: list[np.ndarray],
    pixel: tuple[int, int],
    z_grid: np.ndarray,
    tomogram: np.ndarray,
    output_path: Path,
) -> None:
    magnitude = np.abs(chip_native)
    magnitude_db = 20.0 * np.log10(magnitude / (magnitude.max() + 1e-12) + 1e-6)

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes[0, 0].imshow(magnitude_db, cmap="gray", aspect="auto")
    axes[0, 0].scatter([pixel[1]], [pixel[0]], c="tab:red", marker="x", s=40)
    axes[0, 0].set_title("Sentinel-1 IW2 VV Chip Magnitude (dB)")

    axes[0, 1].imshow(np.angle(chip_native), cmap="twilight", aspect="auto")
    axes[0, 1].set_title("Chip Phase")

    sub_preview = np.hstack([np.abs(img.T) for img in sub_slcs[: min(4, len(sub_slcs))]])
    sub_preview_db = 20.0 * np.log10(sub_preview / (sub_preview.max() + 1e-12) + 1e-6)
    axes[1, 0].imshow(sub_preview_db, cmap="magma", aspect="auto")
    axes[1, 0].set_title("First Sub-Aperture Magnitudes (dB)")

    axes[1, 1].plot(z_grid, np.abs(tomogram))
    axes[1, 1].set_title("Tomogram at Giza-Area Pixel")
    axes[1, 1].set_xlabel("Depth z (m)")
    axes[1, 1].set_ylabel("|h(z)|")

    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])
    axes[1, 1].set_xticks(np.linspace(z_grid.min(), z_grid.max(), 5))

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    output_dir = Path("data/processed/giza_test")
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = prepare_giza_dataset()
    local_line, local_pixel = dataset.default_local_point
    tomogram, point_debug = compute_point_tomogram(dataset, local_line, local_pixel)

    preview_path = output_dir / "giza_iw2_vv_preview.png"
    plot_preview(
        dataset.chip_native,
        dataset.sub_slcs,
        (local_line, local_pixel),
        dataset.z_grid,
        tomogram,
        preview_path,
    )

    np.savez_compressed(
        output_dir / "giza_iw2_vv_results.npz",
        chip=dataset.chip,
        pixel=np.asarray([local_pixel, local_line]),
        line_bounds=np.asarray(dataset.line_bounds),
        pixel_bounds=np.asarray(dataset.pixel_bounds),
        z_grid=dataset.z_grid,
        tomogram=tomogram,
        baselines_perp=np.asarray(point_debug["geometry"]["baselines_perp_m"]),
    )

    summary = dataset.to_summary() | {
        "local_pixel_line_sample": [int(local_line), int(local_pixel)],
        "tomography_pixel_range_azimuth": [int(local_pixel), int(local_line)],
        "point_debug": point_debug,
        "preview_path": str(preview_path),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
