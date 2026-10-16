from __future__ import annotations

import copy
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.ndimage import gaussian_filter

from .config import load_config
from .pipeline import build_selection


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {name: data[name] for name in data.files}


def plot_detector_region_images(
    *,
    analysis_dir: str | Path,
    config_path: str | Path,
    out_dir: str | Path,
    bin_width_mm: float = 4.0,
    smoothing_sigma_bins: float = 0.75,
) -> dict[str, Any]:
    """Show images reconstructed from different existing detector regions."""
    if bin_width_mm <= 0.0:
        raise ValueError("region-image bin width must be positive")
    if smoothing_sigma_bins < 0.0:
        raise ValueError("smoothing sigma cannot be negative")
    analysis_dir = Path(analysis_dir)
    config_path = Path(config_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    config = load_config(config_path)
    pairs = _load_npz(analysis_dir / "pairs.npz")
    reconstruction = _load_npz(analysis_dir / "reconstruction.npz")
    if len(pairs["key"]) != len(reconstruction["key"]) or not np.array_equal(
        pairs["key"],
        reconstruction["key"],
    ):
        raise ValueError("pair and reconstruction event keys are not aligned")
    with (analysis_dir / "summary.json").open(
        "r",
        encoding="utf-8",
    ) as handle:
        summary = json.load(handle)
    n_primaries = int(summary["event_counts"]["primaries"])

    observable_config = copy.deepcopy(config)
    observable_config["selection"]["phantom_truth_policy"] = "ignore"
    selected, _, diagnostics = build_selection(pairs, observable_config)
    valid = reconstruction["valid"].astype(bool)
    baseline = selected & valid
    if not np.any(baseline):
        raise ValueError("analysis has no observable-selected valid events")
    d1_radius = np.asarray(diagnostics["d1_radius_mm"], dtype=np.float64)
    d1_phi = np.mod(
        np.degrees(np.arctan2(pairs["d1_y_mm"], pairs["d1_x_mm"])),
        360.0,
    )
    d2x = np.asarray(pairs["d2_x_mm"], dtype=np.float64)
    d2y = np.asarray(pairs["d2_y_mm"], dtype=np.float64)
    d2radius = np.hypot(d2x, d2y)

    d1_regions: list[tuple[str, np.ndarray]] = [
        ("All D1", np.ones(len(baseline), dtype=bool)),
        ("D1 R=20–50 mm", (d1_radius >= 20.0) & (d1_radius < 50.0)),
        ("D1 R=50–100 mm", (d1_radius >= 50.0) & (d1_radius < 100.0)),
        ("D1 R=100–200 mm", (d1_radius >= 100.0) & (d1_radius <= 200.0)),
        ("D1 φ=0–90°", (d1_phi >= 0.0) & (d1_phi < 90.0)),
        ("D1 φ=90–180°", (d1_phi >= 90.0) & (d1_phi < 180.0)),
        ("D1 φ=180–270°", (d1_phi >= 180.0) & (d1_phi < 270.0)),
        ("D1 φ=270–360°", (d1_phi >= 270.0) & (d1_phi < 360.0)),
    ]
    d2_regions: list[tuple[str, np.ndarray]] = [
        ("All D2", np.ones(len(baseline), dtype=bool)),
        (
            "D2 center 10 mm",
            (np.abs(d2x) <= 5.0) & (np.abs(d2y) <= 5.0),
        ),
        (
            "D2 center 20 mm",
            (np.abs(d2x) <= 10.0) & (np.abs(d2y) <= 10.0),
        ),
        (
            "D2 outer r=16–36 mm",
            (d2radius >= 16.0) & (d2radius < 36.0),
        ),
    ]

    x = np.asarray(reconstruction["x_mm"], dtype=np.float64)
    depth = -np.asarray(reconstruction["z_mm"], dtype=np.float64)
    finite = np.isfinite(x) & np.isfinite(depth)
    z_lo, z_hi = map(float, config["phantom"]["z_interval_mm"])
    phantom_depth = (-z_hi, -z_lo)
    depth_lo = min(0.0, phantom_depth[0])
    depth_hi = max(100.0, phantom_depth[1])
    x_limit = 100.0
    x_edges = np.arange(
        -x_limit,
        x_limit + bin_width_mm * 0.5,
        bin_width_mm,
    )
    depth_edges = np.arange(
        depth_lo,
        depth_hi + bin_width_mm * 0.5,
        bin_width_mm,
    )
    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    depth_centers = 0.5 * (depth_edges[:-1] + depth_edges[1:])
    truth_signal = (
        (pairs["d1_n_phantom_compton"] == 1)
        & (pairs["d2_n_phantom_compton"] == 0)
    )
    truth_matched = reconstruction["truth_matched"].astype(bool)
    voxel_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []

    def calculate_regions(
        family: str,
        definitions: list[tuple[str, np.ndarray]],
    ) -> list[tuple[str, np.ndarray]]:
        output: list[tuple[str, np.ndarray]] = []
        baseline_count = int(np.count_nonzero(baseline))
        for label, condition in definitions:
            mask = baseline & condition & finite
            histogram, _, _ = np.histogram2d(
                x[mask],
                depth[mask],
                bins=(x_edges, depth_edges),
            )
            shown = (
                gaussian_filter(histogram, smoothing_sigma_bins)
                if smoothing_sigma_bins > 0.0
                else histogram
            )
            maximum = float(np.max(shown))
            normalized = shown / maximum if maximum > 0.0 else shown
            output.append((label, normalized))
            n = int(np.count_nonzero(mask))
            signal_n = int(np.count_nonzero(mask & truth_signal))
            matched_n = int(np.count_nonzero(mask & truth_matched))
            metric_rows.append(
                {
                    "family": family,
                    "region": label,
                    "n_observable_selected_valid": n,
                    "retention_vs_all": float(n / baseline_count),
                    "efficiency_per_primary": float(n / n_primaries),
                    "truth_single_scatter_fraction_score_only": (
                        float(signal_n / n) if n else None
                    ),
                    "actor_truth_match_fraction_score_only": (
                        float(matched_n / n) if n else None
                    ),
                }
            )
            for ix, x_center in enumerate(x_centers):
                for iz, depth_center in enumerate(depth_centers):
                    voxel_rows.append(
                        {
                            "family": family,
                            "region": label,
                            "x_mm": float(x_center),
                            "depth_mm": float(depth_center),
                            "raw_counts": int(histogram[ix, iz]),
                            "within_panel_normalized_smoothed_counts": float(
                                normalized[ix, iz]
                            ),
                        }
                    )
        return output

    d1_images = calculate_regions("D1", d1_regions)
    d2_images = calculate_regions("D2", d2_regions)
    voxel_csv = out_dir / "detector_region_voxels.csv"
    with voxel_csv.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(voxel_rows[0]))
        writer.writeheader()
        writer.writerows(voxel_rows)
    metrics_csv = out_dir / "detector_region_metrics.csv"
    with metrics_csv.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metric_rows[0]))
        writer.writeheader()
        writer.writerows(metric_rows)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = [
        "Arial",
        "DejaVu Sans",
        "Liberation Sans",
    ]
    plt.rcParams["svg.fonttype"] = "none"
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["font.size"] = 7
    plt.rcParams["axes.spines.right"] = False
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.linewidth"] = 0.8

    extent = [
        float(x_edges[0]),
        float(x_edges[-1]),
        float(depth_edges[-1]),
        float(depth_edges[0]),
    ]
    metrics_by_label = {row["region"]: row for row in metric_rows}
    mm_to_inch = 1.0 / 25.4
    figures: list[str] = []

    fig, axes = plt.subplots(
        2,
        4,
        figsize=(180.0 * mm_to_inch, 96.0 * mm_to_inch),
        sharex=True,
        sharey=True,
    )
    image = None
    for index, (ax, (label, values)) in enumerate(
        zip(axes.flat, d1_images)
    ):
        image = ax.imshow(
            values.T,
            extent=extent,
            aspect="equal",
            cmap="viridis",
            vmin=0.0,
            vmax=1.0,
            interpolation="nearest",
            rasterized=True,
        )
        row = metrics_by_label[label]
        ax.set_title(
            f"{label}\n"
            f"n={row['n_observable_selected_valid']:,}; "
            f"retain={row['retention_vs_all']:.1%}",
            fontsize=6.5,
        )
        ax.axhline(0.0, color="white", linestyle="--", linewidth=0.5)
        ax.set_xticks([-100.0, 0.0, 100.0])
        ax.text(
            -0.17,
            1.12,
            "abcdefgh"[index],
            transform=ax.transAxes,
            fontweight="bold",
            fontsize=8.5,
        )
    for ax in axes[-1, :]:
        ax.set_xlabel("Reconstructed x (mm)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Reconstructed depth (mm)")
    if image is not None:
        colorbar_axis = fig.add_axes([0.925, 0.18, 0.012, 0.64])
        colorbar = fig.colorbar(image, cax=colorbar_axis)
        colorbar.set_label("Within-panel normalized counts")
    fig.subplots_adjust(
        left=0.07,
        right=0.90,
        bottom=0.09,
        top=0.91,
        wspace=0.22,
        hspace=0.27,
    )
    for extension, dpi in (
        ("svg", 300),
        ("pdf", 300),
        ("tiff", 600),
        ("png", 300),
    ):
        path = out_dir / f"d1_region_images.{extension}"
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        figures.append(str(path.resolve()))
    plt.close(fig)

    fig, axes = plt.subplots(
        1,
        4,
        figsize=(180.0 * mm_to_inch, 52.0 * mm_to_inch),
        sharex=True,
        sharey=True,
    )
    image = None
    for index, (ax, (label, values)) in enumerate(zip(axes, d2_images)):
        image = ax.imshow(
            values.T,
            extent=extent,
            aspect="equal",
            cmap="viridis",
            vmin=0.0,
            vmax=1.0,
            interpolation="nearest",
            rasterized=True,
        )
        row = metrics_by_label[label]
        ax.set_title(
            f"{label}\n"
            f"n={row['n_observable_selected_valid']:,}; "
            f"retain={row['retention_vs_all']:.1%}",
            fontsize=6.3,
        )
        ax.axhline(0.0, color="white", linestyle="--", linewidth=0.5)
        ax.set_xticks([-100.0, 0.0, 100.0])
        ax.set_xlabel("Reconstructed x (mm)")
        ax.text(
            -0.17,
            1.12,
            "abcd"[index],
            transform=ax.transAxes,
            fontweight="bold",
            fontsize=8.5,
        )
    axes[0].set_ylabel("Reconstructed depth (mm)")
    if image is not None:
        colorbar_axis = fig.add_axes([0.91, 0.20, 0.012, 0.60])
        colorbar = fig.colorbar(image, cax=colorbar_axis)
        colorbar.set_label("Within-panel normalized counts")
    fig.subplots_adjust(
        left=0.07,
        right=0.885,
        bottom=0.17,
        top=0.84,
        wspace=0.22,
    )
    for extension, dpi in (
        ("svg", 300),
        ("pdf", 300),
        ("tiff", 600),
        ("png", 300),
    ):
        path = out_dir / f"d2_region_images.{extension}"
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        figures.append(str(path.resolve()))
    plt.close(fig)

    report = {
        "schema_version": "1.0",
        "analysis_dir": str(analysis_dir.resolve()),
        "config": str(config_path.resolve()),
        "selection_uses_monte_carlo_truth": False,
        "truth_use": "post-selection score columns only",
        "bin_width_mm": float(bin_width_mm),
        "smoothing_sigma_bins_visualization_only": float(
            smoothing_sigma_bins
        ),
        "rows": metric_rows,
        "voxel_source_csv": str(voxel_csv.resolve()),
        "metrics_source_csv": str(metrics_csv.resolve()),
        "figures": figures,
        "interpretation_guardrail": (
            "Each panel is normalized to its own maximum to compare spatial "
            "shape. Event count and retention are printed separately, so panel "
            "brightness must not be read as equal sensitivity. A homogeneous "
            "slab image is a region-use demonstration, not spatial resolution."
        ),
    }
    with (out_dir / "detector_region_images.json").open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    return report
