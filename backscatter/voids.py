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


def compute_void_cnr(
    *,
    void_roi_count: int,
    reference_roi_count: int,
    void_control_count: int,
    reference_control_count: int,
) -> dict[str, Any]:
    """Calculate a control-normalized Poisson deficit and its CNR.

    Control regions estimate the run-to-run normalization.  The uncertainty
    includes Poisson variance from the void ROI, reference ROI, and both
    control-region totals.
    """
    counts = (
        void_roi_count,
        reference_roi_count,
        void_control_count,
        reference_control_count,
    )
    if any(int(value) < 0 for value in counts):
        raise ValueError("ROI counts cannot be negative")
    if reference_roi_count == 0:
        raise ValueError("reference ROI contains no events")
    if void_control_count == 0 or reference_control_count == 0:
        raise ValueError("control regions contain no events")

    scale = float(void_control_count / reference_control_count)
    expected = float(scale * reference_roi_count)
    deficit = float(expected - void_roi_count)
    scale_variance = float(
        scale**2
        * (
            1.0 / float(void_control_count)
            + 1.0 / float(reference_control_count)
        )
    )
    variance = float(
        void_roi_count
        + scale**2 * reference_roi_count
        + reference_roi_count**2 * scale_variance
    )
    standard_deviation = float(np.sqrt(variance))
    contrast = float(deficit / expected) if expected > 0.0 else None
    cnr = float(deficit / standard_deviation) if standard_deviation > 0.0 else None
    contrast_standard_deviation = (
        float(standard_deviation / expected) if expected > 0.0 else None
    )
    contrast_ci = (
        [
            float(contrast - 1.96 * contrast_standard_deviation),
            float(contrast + 1.96 * contrast_standard_deviation),
        ]
        if contrast is not None and contrast_standard_deviation is not None
        else None
    )
    return {
        "void_roi_count": int(void_roi_count),
        "reference_roi_count": int(reference_roi_count),
        "void_control_count": int(void_control_count),
        "reference_control_count": int(reference_control_count),
        "control_normalization_scale": scale,
        "expected_no_void_roi_count": expected,
        "event_deficit": deficit,
        "event_deficit_standard_deviation": standard_deviation,
        "contrast": contrast,
        "contrast_approximate_95_ci": contrast_ci,
        "cnr": cnr,
        "uncertainty_model": (
            "Independent Poisson counting uncertainty for the void ROI, "
            "reference ROI, and pooled control regions; no detector-systematic "
            "or reconstruction-model uncertainty is included."
        ),
    }


def _observable_points(
    analysis_dir: Path,
    config: dict[str, Any],
    *,
    y_half_width_mm: float,
) -> dict[str, np.ndarray]:
    pairs = _load_npz(analysis_dir / "pairs.npz")
    reconstruction = _load_npz(analysis_dir / "reconstruction.npz")
    if len(pairs["key"]) != len(reconstruction["key"]) or not np.array_equal(
        pairs["key"],
        reconstruction["key"],
    ):
        raise ValueError("pair and reconstruction event keys are not aligned")

    observable_config = copy.deepcopy(config)
    observable_config["selection"]["phantom_truth_policy"] = "ignore"
    selected, _, _ = build_selection(pairs, observable_config)
    usable = selected & reconstruction["valid"].astype(bool)
    x = np.asarray(reconstruction["x_mm"][usable], dtype=np.float64)
    y = np.asarray(reconstruction["y_mm"][usable], dtype=np.float64)
    depth = -np.asarray(reconstruction["z_mm"][usable], dtype=np.float64)
    finite = np.isfinite(x) & np.isfinite(y) & np.isfinite(depth)
    within_y = np.abs(y) <= float(y_half_width_mm)
    keep = finite & within_y
    return {
        "x_mm": x[keep],
        "y_mm": y[keep],
        "depth_mm": depth[keep],
        "n_observable_selected_valid_all_y": np.asarray(
            [int(usable.sum())],
            dtype=np.int64,
        ),
    }


def _circle_count(
    points: dict[str, np.ndarray],
    *,
    center_x_mm: float,
    center_depth_mm: float,
    radius_mm: float,
) -> int:
    distance_squared = (
        (points["x_mm"] - float(center_x_mm)) ** 2
        + (points["depth_mm"] - float(center_depth_mm)) ** 2
    )
    return int(np.count_nonzero(distance_squared <= float(radius_mm) ** 2))


def _control_centers(
    *,
    void_x_mm: float,
    radius_mm: float,
    slab_x_interval_mm: tuple[float, float],
) -> list[float]:
    # The nearest controls are tangent to, but do not overlap, the target ROI.
    # The 30 mm floor keeps the layout unchanged for the 10 mm-diameter pilot.
    # A previous 4R spacing put every control for the 20 mm target outside the
    # useful reconstructed field of view and therefore supplied no counts.
    offset = max(2.0 * float(radius_mm), 30.0)
    candidates = [
        void_x_mm - 2.0 * offset,
        void_x_mm - offset,
        void_x_mm + offset,
        void_x_mm + 2.0 * offset,
    ]
    lo, hi = slab_x_interval_mm
    return [
        float(value)
        for value in candidates
        if value - radius_mm >= lo and value + radius_mm <= hi
    ]


def _summary_primary_count(analysis_dir: Path) -> int:
    with (analysis_dir / "summary.json").open(
        "r",
        encoding="utf-8",
    ) as handle:
        summary = json.load(handle)
    return int(summary["event_counts"]["primaries"])


def evaluate_concrete_void(
    *,
    void_analysis_dir: str | Path,
    reference_analysis_dir: str | Path,
    config_path: str | Path,
    out_dir: str | Path,
    roi_radius_scale: float = 1.5,
    map_bin_width_mm: float = 2.0,
    smoothing_sigma_bins: float = 1.0,
) -> dict[str, Any]:
    """Evaluate a cylindrical void against a homogeneous-slab reference.

    Only observable detector quantities define the selected reconstruction.
    Monte-Carlo truth fields are deliberately not used in the image or CNR.
    """
    if roi_radius_scale <= 0.0:
        raise ValueError("ROI radius scale must be positive")
    if map_bin_width_mm <= 0.0:
        raise ValueError("map bin width must be positive")
    if smoothing_sigma_bins < 0.0:
        raise ValueError("smoothing sigma cannot be negative")

    void_analysis_dir = Path(void_analysis_dir)
    reference_analysis_dir = Path(reference_analysis_dir)
    config_path = Path(config_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    config = load_config(config_path)
    void_geometry = config["phantom"].get("void")
    if not isinstance(void_geometry, dict):
        raise ValueError("configuration does not describe a concrete void")
    if void_geometry.get("axis") != "y":
        raise ValueError("only a y-axis cylindrical void is currently supported")

    void_radius = float(void_geometry["radius_mm"])
    void_length = float(void_geometry["length_mm"])
    void_x = float(void_geometry["center_mm"][0])
    void_depth = float(void_geometry["depth_mm"])
    y_half_width = 0.5 * void_length
    void_points = _observable_points(
        void_analysis_dir,
        config,
        y_half_width_mm=y_half_width,
    )
    reference_points = _observable_points(
        reference_analysis_dir,
        config,
        y_half_width_mm=y_half_width,
    )

    slab_center_x = float(config["phantom"]["center_mm"][0])
    slab_size_x = float(config["phantom"]["size_mm"][0])
    slab_x_interval = (
        slab_center_x - slab_size_x / 2.0,
        slab_center_x + slab_size_x / 2.0,
    )
    scales = sorted({1.0, 1.5, 2.0, float(roi_radius_scale)})
    scale_metrics: list[dict[str, Any]] = []
    for scale_value in scales:
        roi_radius = float(scale_value * void_radius)
        controls = _control_centers(
            void_x_mm=void_x,
            radius_mm=roi_radius,
            slab_x_interval_mm=slab_x_interval,
        )
        if len(controls) < 2:
            raise ValueError("fewer than two non-overlapping control ROIs fit")
        void_roi = _circle_count(
            void_points,
            center_x_mm=void_x,
            center_depth_mm=void_depth,
            radius_mm=roi_radius,
        )
        reference_roi = _circle_count(
            reference_points,
            center_x_mm=void_x,
            center_depth_mm=void_depth,
            radius_mm=roi_radius,
        )
        void_control_counts = [
            _circle_count(
                void_points,
                center_x_mm=x_center,
                center_depth_mm=void_depth,
                radius_mm=roi_radius,
            )
            for x_center in controls
        ]
        reference_control_counts = [
            _circle_count(
                reference_points,
                center_x_mm=x_center,
                center_depth_mm=void_depth,
                radius_mm=roi_radius,
            )
            for x_center in controls
        ]
        void_control = sum(void_control_counts)
        reference_control = sum(reference_control_counts)
        metrics = compute_void_cnr(
            void_roi_count=void_roi,
            reference_roi_count=reference_roi,
            void_control_count=void_control,
            reference_control_count=reference_control,
        )
        metrics.update(
            {
                "roi_radius_scale": float(scale_value),
                "roi_radius_mm": roi_radius,
                "control_x_centers_mm": controls,
                "void_control_counts_by_center": void_control_counts,
                "reference_control_counts_by_center": (
                    reference_control_counts
                ),
                "control_layout_rule": (
                    "Four same-depth lateral ROIs centered at +/-d and "
                    "+/-2d from the target, where "
                    "d=max(2*ROI radius, 30 mm); ROIs are tangent or "
                    "separated and clipped to the concrete slab."
                ),
            }
        )
        scale_metrics.append(metrics)

    primary_metrics = min(
        scale_metrics,
        key=lambda row: abs(float(row["roi_radius_scale"]) - roi_radius_scale),
    )
    n_void_primaries = _summary_primary_count(void_analysis_dir)
    n_reference_primaries = _summary_primary_count(reference_analysis_dir)

    x_edges = np.arange(
        slab_x_interval[0],
        slab_x_interval[1] + map_bin_width_mm * 0.5,
        map_bin_width_mm,
    )
    z_lo, z_hi = map(float, config["phantom"]["z_interval_mm"])
    depth_interval = (-z_hi, -z_lo)
    depth_edges = np.arange(
        depth_interval[0],
        depth_interval[1] + map_bin_width_mm * 0.5,
        map_bin_width_mm,
    )
    void_hist, _, _ = np.histogram2d(
        void_points["x_mm"],
        void_points["depth_mm"],
        bins=(x_edges, depth_edges),
    )
    reference_hist, _, _ = np.histogram2d(
        reference_points["x_mm"],
        reference_points["depth_mm"],
        bins=(x_edges, depth_edges),
    )
    normalization = float(primary_metrics["control_normalization_scale"])
    expected_hist = normalization * reference_hist
    deficit_hist = expected_hist - void_hist
    if smoothing_sigma_bins > 0.0:
        shown_void = gaussian_filter(void_hist, smoothing_sigma_bins)
        shown_expected = gaussian_filter(expected_hist, smoothing_sigma_bins)
        shown_deficit = gaussian_filter(deficit_hist, smoothing_sigma_bins)
    else:
        shown_void = void_hist
        shown_expected = expected_hist
        shown_deficit = deficit_hist

    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    depth_centers = 0.5 * (depth_edges[:-1] + depth_edges[1:])
    voxel_csv = out_dir / "void_map_source.csv"
    with voxel_csv.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "x_mm",
                "depth_mm",
                "void_counts",
                "reference_counts",
                "scaled_reference_counts",
                "expected_minus_void_counts",
            ]
        )
        for ix, x_center in enumerate(x_centers):
            for iz, depth_center in enumerate(depth_centers):
                writer.writerow(
                    [
                        float(x_center),
                        float(depth_center),
                        int(void_hist[ix, iz]),
                        int(reference_hist[ix, iz]),
                        float(expected_hist[ix, iz]),
                        float(deficit_hist[ix, iz]),
                    ]
                )

    depth_band = float(primary_metrics["roi_radius_mm"])
    void_band = np.abs(void_points["depth_mm"] - void_depth) <= depth_band
    reference_band = (
        np.abs(reference_points["depth_mm"] - void_depth) <= depth_band
    )
    void_profile, _ = np.histogram(void_points["x_mm"][void_band], bins=x_edges)
    reference_profile, _ = np.histogram(
        reference_points["x_mm"][reference_band],
        bins=x_edges,
    )
    expected_profile = normalization * reference_profile
    profile_csv = out_dir / "void_profile_source.csv"
    with profile_csv.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "x_mm",
                "void_counts",
                "reference_counts",
                "scaled_reference_counts",
            ]
        )
        writer.writerows(
            zip(
                x_centers,
                void_profile,
                reference_profile,
                expected_profile,
            )
        )

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

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
    plt.rcParams["legend.frameon"] = False

    mm_to_inch = 1.0 / 25.4
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(180.0 * mm_to_inch, 65.0 * mm_to_inch),
        gridspec_kw={"width_ratios": [1.0, 1.0, 1.15]},
    )
    extent = [
        float(x_edges[0]),
        float(x_edges[-1]),
        float(depth_edges[-1]),
        float(depth_edges[0]),
    ]
    x_display = max(50.0, abs(void_x) + 6.0 * void_radius)
    common_max = max(
        float(np.quantile(shown_void, 0.995)),
        float(np.quantile(shown_expected, 0.995)),
        1.0,
    )
    for ax, image_values, title in (
        (axes[0], shown_void, "Void phantom"),
        (axes[1], shown_expected, "Homogeneous reference"),
    ):
        image = ax.imshow(
            image_values.T,
            extent=extent,
            aspect="equal",
            cmap="viridis",
            vmin=0.0,
            vmax=common_max,
            interpolation="nearest",
            rasterized=True,
        )
        ax.add_patch(
            Circle(
                (void_x, void_depth),
                float(primary_metrics["roi_radius_mm"]),
                fill=False,
                edgecolor="white",
                linewidth=0.9,
                linestyle="--",
            )
        )
        ax.set_xlim(-x_display, x_display)
        ax.set_ylim(min(100.0, void_depth + 35.0), max(0.0, void_depth - 35.0))
        ax.set_xlabel("Reconstructed x (mm)")
        ax.set_ylabel("Reconstructed depth (mm)")
        ax.set_title(title, fontsize=7.2)
        colorbar = fig.colorbar(image, ax=ax, fraction=0.048, pad=0.03)
        colorbar.set_label("Smoothed counts / voxel")

    ax = axes[2]
    ax.step(
        x_centers,
        expected_profile,
        where="mid",
        color="#767676",
        linewidth=1.2,
        label="No-void expectation",
    )
    ax.step(
        x_centers,
        void_profile,
        where="mid",
        color="#0F4D92",
        linewidth=1.2,
        label="Void phantom",
    )
    ax.axvspan(
        void_x - float(primary_metrics["roi_radius_mm"]),
        void_x + float(primary_metrics["roi_radius_mm"]),
        color="#D8D8D8",
        alpha=0.7,
    )
    ax.set_xlim(-x_display, x_display)
    ax.set_xlabel("Reconstructed x (mm)")
    ax.set_ylabel(
        f"Events within depth {void_depth:g} ± {depth_band:g} mm"
    )
    ax.legend(loc="upper right", fontsize=6.2)
    cnr = primary_metrics["cnr"]
    contrast = primary_metrics["contrast"]
    ax.text(
        0.03,
        0.05,
        f"ROI radius = {roi_radius_scale:g}× void radius\n"
        f"Contrast = {contrast:.1%}\n"
        f"CNR = {cnr:.2f}",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=6.3,
    )

    for label, ax in zip("abc", axes):
        ax.text(
            -0.17,
            1.05,
            label,
            transform=ax.transAxes,
            fontweight="bold",
            fontsize=9,
        )
    fig.tight_layout(pad=1.1, w_pad=1.7)
    output_base = out_dir / "concrete_void_detectability"
    figures: list[str] = []
    for extension, dpi in (
        ("svg", 300),
        ("pdf", 300),
        ("tiff", 600),
        ("png", 300),
    ):
        path = output_base.with_suffix(f".{extension}")
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        figures.append(str(path.resolve()))
    plt.close(fig)

    report: dict[str, Any] = {
        "schema_version": "1.0",
        "void_analysis_dir": str(void_analysis_dir.resolve()),
        "reference_analysis_dir": str(reference_analysis_dir.resolve()),
        "config": str(config_path.resolve()),
        "selection": {
            "uses_monte_carlo_truth": False,
            "phantom_truth_policy_forced_to": "ignore",
            "y_half_width_mm": y_half_width,
        },
        "void_geometry": void_geometry,
        "event_counts": {
            "void_primaries": n_void_primaries,
            "reference_primaries": n_reference_primaries,
            "void_observable_selected_valid_all_y": int(
                void_points["n_observable_selected_valid_all_y"][0]
            ),
            "reference_observable_selected_valid_all_y": int(
                reference_points["n_observable_selected_valid_all_y"][0]
            ),
            "void_inside_y_extent": int(len(void_points["x_mm"])),
            "reference_inside_y_extent": int(len(reference_points["x_mm"])),
        },
        "predefined_primary_roi": primary_metrics,
        "roi_scale_sensitivity": scale_metrics,
        "map": {
            "bin_width_mm": float(map_bin_width_mm),
            "smoothing_sigma_bins_visualization_only": float(
                smoothing_sigma_bins
            ),
            "source_csv": str(voxel_csv.resolve()),
        },
        "profile_source_csv": str(profile_csv.resolve()),
        "figures": figures,
        "interpretation_guardrail": (
            "A positive CNR is single-pair pilot evidence for a count deficit "
            "at the known void location. It is not a blind-search detection "
            "probability or a limit of detection. Independent seeds and "
            "location-blinded trials are required for those claims."
        ),
    }
    with (out_dir / "concrete_void_metrics.json").open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    return report
