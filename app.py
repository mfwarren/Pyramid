from __future__ import annotations

import base64
import json
from pathlib import Path

from flask import Flask, jsonify, render_template, request

from giza_backend import (
    compute_point_tomogram,
    compute_vertical_slice,
    prepare_giza_dataset,
    render_magnitude_png,
    render_vertical_slice_png,
)


app = Flask(__name__)
DATASET = prepare_giza_dataset(Path("data/processed/giza_safe_subset"))
OVERVIEW_PNG = render_magnitude_png(DATASET.chip_native)


def to_data_url(png_bytes: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(png_bytes).decode("ascii")


@app.get("/")
def index():
    local_line, local_sample = DATASET.default_local_point
    point_tomogram, point_debug = compute_point_tomogram(DATASET, local_line, local_sample)
    slice_matrix, slice_debug = compute_vertical_slice(
        DATASET,
        x0=DATASET.chip_native.shape[1] * 0.25,
        y0=local_line,
        x1=DATASET.chip_native.shape[1] * 0.75,
        y1=local_line,
        n_samples=48,
    )
    return render_template(
        "index.html",
        overview_data_url=to_data_url(OVERVIEW_PNG),
        slice_data_url=to_data_url(render_vertical_slice_png(slice_matrix, DATASET.z_grid)),
        dataset_summary=DATASET.to_summary(),
        initial_point_debug=point_debug,
        initial_slice_debug=slice_debug,
    )


@app.post("/api/slice")
def api_slice():
    payload = request.get_json(force=True)
    x0 = float(payload["x0"])
    y0 = float(payload["y0"])
    x1 = float(payload["x1"])
    y1 = float(payload["y1"])
    n_samples = int(payload.get("n_samples", 48))
    n_samples = max(12, min(n_samples, 160))
    slice_matrix, debug = compute_vertical_slice(DATASET, x0, y0, x1, y1, n_samples=n_samples)
    return jsonify(
        {
            "image": to_data_url(render_vertical_slice_png(slice_matrix, DATASET.z_grid)),
            "debug": debug,
            "stats": {
                "min": float(slice_matrix.min()),
                "max": float(slice_matrix.max()),
                "mean": float(slice_matrix.mean()),
            },
        }
    )


@app.get("/api/state")
def api_state():
    return jsonify(
        {
            "dataset": DATASET.to_summary(),
            "overview_image": to_data_url(OVERVIEW_PNG),
        }
    )


if __name__ == "__main__":
    app.run(debug=True, port=5001)
