from __future__ import annotations

import copy
import csv
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import convolve, gaussian_filter


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backscatter.config import load_config
from backscatter.iterative import (
    ReconstructionGrid2D,
    build_ray_likelihood_system,
    denoise_tv_chambolle,
    listmode_mlem,
    smooth_reference_sensitivity,
)
from backscatter.pipeline import build_selection


OUTPUT_DIR = PROJECT_ROOT / "outputs" / "advanced_reconstruction_20260807"
CONFIG_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "ppt_void_resolution_20260807"
    / "void_d20_depth30_d1_180_250.json"
)
VOID_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "hardware_update_20260731"
    / "response_variants"
    / "concrete_void_d20_depth30_hardware_20260731"
)
REFERENCE_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "hardware_update_20260731"
    / "response_analysis"
)
ROD_METRICS = (
    PROJECT_ROOT
    / "outputs"
    / "transverse_fwhm_5mm_20260805"
    / "independent_gate_seed"
    / "rod_width2_depth30_5m_seed24080573_local_fwhm5_validation"
    / "canonical_response"
    / "standard_metrics"
    / "metrics.json"
)
ROD_CURVES = ROD_METRICS.with_name("rod_response_curves.csv")


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {name: data[name] for name in data.files}


def _load_primary_count(analysis_dir: Path) -> int:
    with (analysis_dir / "summary.json").open("r", encoding="utf-8") as handle:
        summary = json.load(handle)
    return int(summary["event_counts"]["primaries"])


def _observable_selection(
    pairs: dict[str, np.ndarray],
    config: dict[str, Any],
) -> np.ndarray:
    observable_config = copy.deepcopy(config)
    observable_config["selection"]["phantom_truth_policy"] = "ignore"
    selected, _, _ = build_selection(pairs, observable_config)
    return selected


def _baseline_point_histogram(
    pairs: dict[str, np.ndarray],
    reconstruction: dict[str, np.ndarray],
    selected: np.ndarray,
    grid: ReconstructionGrid2D,
    *,
    event_subset: np.ndarray | None = None,
    y_half_width_mm: float = 150.0,
) -> np.ndarray:
    usable = selected & reconstruction["valid"].astype(bool)
    if event_subset is not None:
        usable &= np.asarray(event_subset, dtype=bool)
    x = np.asarray(reconstruction["x_mm"], dtype=np.float64)
    y = np.asarray(reconstruction["y_mm"], dtype=np.float64)
    depth = -np.asarray(reconstruction["z_mm"], dtype=np.float64)
    usable &= (
        np.isfinite(x)
        & np.isfinite(y)
        & np.isfinite(depth)
        & (np.abs(y) <= float(y_half_width_mm))
    )
    counts, _, _ = np.histogram2d(
        depth[usable],
        x[usable],
        bins=[grid.depth_edges_mm, grid.x_edges_mm],
    )
    return counts.astype(np.float64)


def _safe_relative_density(
    target: np.ndarray,
    reference: np.ndarray,
    *,
    target_exposure: float,
    reference_exposure: float,
    smoothing_sigma_bins: float,
    minimum_reference_fraction_of_depth_max: float,
) -> np.ndarray:
    target_rate = np.asarray(target, dtype=np.float64) / float(target_exposure)
    reference_rate = np.asarray(reference, dtype=np.float64) / float(
        reference_exposure
    )
    if smoothing_sigma_bins > 0.0:
        target_rate = gaussian_filter(
            target_rate,
            sigma=float(smoothing_sigma_bins),
            mode="nearest",
        )
        reference_rate = gaussian_filter(
            reference_rate,
            sigma=float(smoothing_sigma_bins),
            mode="nearest",
        )
    depth_maximum = np.max(reference_rate, axis=1, keepdims=True)
    valid = (
        np.isfinite(reference_rate)
        & (depth_maximum > 0.0)
        & (
            reference_rate
            >= float(minimum_reference_fraction_of_depth_max) * depth_maximum
        )
    )
    density = np.full_like(target_rate, np.nan)
    density[valid] = target_rate[valid] / reference_rate[valid]
    return density


def _gaussian_kernel_2d(sigma_bins: float) -> np.ndarray:
    if sigma_bins <= 0.0:
        return np.ones((1, 1), dtype=np.float64)
    radius = max(1, int(np.ceil(4.0 * sigma_bins)))
    coordinate = np.arange(-radius, radius + 1, dtype=np.float64)
    one_dimensional = np.exp(-0.5 * (coordinate / sigma_bins) ** 2)
    one_dimensional /= one_dimensional.sum()
    return np.outer(one_dimensional, one_dimensional)


def _psf_matched_probability_score(
    target_matrix: Any,
    reference_matrix: Any,
    grid: ReconstructionGrid2D,
    *,
    target_exposure: float,
    reference_exposure: float,
    sigma_bins: float,
) -> dict[str, np.ndarray | float]:
    """Variance-aware deficit image from probability backprojections.

    Each list-mode event contributes a normalized probability row.  The
    reference-minus-target mean is filtered with the independently measured
    line-spread scale.  The corresponding variance is filtered with the square
    of the same kernel, yielding a per-pixel standardized deficit score.
    """
    scale = float(target_exposure / reference_exposure)
    target_sum = np.asarray(target_matrix.sum(axis=0)).ravel().reshape(grid.shape)
    reference_sum = (
        np.asarray(reference_matrix.sum(axis=0)).ravel().reshape(grid.shape)
        * scale
    )
    target_squared = target_matrix.copy()
    target_squared.data = target_squared.data**2
    reference_squared = reference_matrix.copy()
    reference_squared.data = reference_squared.data**2
    variance = (
        np.asarray(target_squared.sum(axis=0)).ravel().reshape(grid.shape)
        + scale**2
        * np.asarray(reference_squared.sum(axis=0)).ravel().reshape(grid.shape)
    )

    kernel = _gaussian_kernel_2d(float(sigma_bins))
    expected = convolve(reference_sum, kernel, mode="nearest")
    deficit = convolve(reference_sum - target_sum, kernel, mode="nearest")
    filtered_variance = convolve(variance, kernel**2, mode="nearest")
    standardized = np.divide(
        deficit,
        np.sqrt(filtered_variance),
        out=np.full_like(deficit, np.nan),
        where=filtered_variance > 0.0,
    )
    fractional_deficit = np.divide(
        deficit,
        expected,
        out=np.full_like(deficit, np.nan),
        where=expected > 0.0,
    )
    return {
        "expected_probability_counts": expected,
        "probability_deficit": deficit,
        "probability_variance": filtered_variance,
        "standardized_deficit": standardized,
        "fractional_deficit": fractional_deficit,
        "relative_density": 1.0 - fractional_deficit,
        "sigma_bins": float(sigma_bins),
    }


def _analysis_masks(
    grid: ReconstructionGrid2D,
    sensitivity: np.ndarray,
    *,
    void_x_mm: float,
    void_depth_mm: float,
    void_radius_mm: float,
) -> dict[str, np.ndarray]:
    x, depth = np.meshgrid(
        grid.x_centers_mm,
        grid.depth_centers_mm,
    )
    circle = (x - void_x_mm) ** 2 + (depth - void_depth_mm) ** 2
    sensitivity = np.asarray(sensitivity, dtype=np.float64)
    if sensitivity.shape != grid.shape:
        raise ValueError("sensitivity map must match the reconstruction grid")
    depth_maximum = np.max(sensitivity, axis=1, keepdims=True)
    support = (
        np.isfinite(sensitivity)
        & (depth_maximum > 0.0)
        & (sensitivity >= 0.08 * depth_maximum)
    )
    roi = (circle <= void_radius_mm**2) & support
    # The D2 tag detector illuminates only about +/-30 mm at 30 mm depth.
    # Earlier +/-60 mm control circles were outside that field and caused an
    # unstable normalization.  Use an illuminated annulus surrounding the
    # target instead, while leaving a 4 mm guard gap around the true void.
    controls = (
        (circle >= (void_radius_mm + 4.0) ** 2)
        & (circle <= (void_radius_mm + 16.0) ** 2)
        & support
    )
    field = (
        (np.abs(x) <= 65.0)
        & (depth >= 8.0)
        & (depth <= 75.0)
        & support
    )
    search = (
        (np.abs(x - void_x_mm) <= 22.0)
        & (np.abs(depth - void_depth_mm) <= 22.0)
    )
    profile_band = (np.abs(depth - void_depth_mm) <= 5.0) & support
    return {
        "x": x,
        "depth": depth,
        "roi": roi,
        "controls": controls,
        "field": field,
        "search": search,
        "profile_band": profile_band,
        "support": support,
    }


def _normalize_density(
    density: np.ndarray,
    controls: np.ndarray,
) -> np.ndarray:
    density = np.asarray(density, dtype=np.float64)
    values = density[controls & np.isfinite(density) & (density > 0.0)]
    if len(values) == 0:
        values = density[np.isfinite(density) & (density > 0.0)]
    if len(values) == 0:
        return np.full_like(density, np.nan)
    return density / float(np.median(values))


def _interpolate_crossing(
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    level: float,
) -> float:
    if y1 == y0:
        return 0.5 * (x0 + x1)
    return x0 + (level - y0) * (x1 - x0) / (y1 - y0)


def _profile_width(
    x: np.ndarray,
    deficit: np.ndarray,
    *,
    center_x_mm: float = 0.0,
) -> dict[str, float | None]:
    finite = np.isfinite(deficit)
    central = finite & (np.abs(x - center_x_mm) <= 12.0)
    outer = finite & (np.abs(x - center_x_mm) >= 20.0) & (
        np.abs(x - center_x_mm) <= 35.0
    )
    if not np.any(central) or not np.any(outer):
        return {
            "baseline": None,
            "peak": None,
            "half_level": None,
            "left_half_max_mm": None,
            "right_half_max_mm": None,
            "apparent_half_max_width_mm": None,
        }
    baseline = float(np.median(deficit[outer]))
    candidate_indices = np.flatnonzero(central)
    peak_index = int(candidate_indices[np.argmax(deficit[central])])
    peak = float(deficit[peak_index])
    half = baseline + 0.5 * (peak - baseline)
    left = None
    right = None
    for index in range(peak_index, 0, -1):
        if not (finite[index - 1] and finite[index]):
            continue
        if deficit[index - 1] <= half <= deficit[index]:
            left = _interpolate_crossing(
                float(x[index - 1]),
                float(deficit[index - 1]),
                float(x[index]),
                float(deficit[index]),
                half,
            )
            break
    for index in range(peak_index, len(x) - 1):
        if not (finite[index] and finite[index + 1]):
            continue
        if deficit[index] >= half >= deficit[index + 1]:
            right = _interpolate_crossing(
                float(x[index]),
                float(deficit[index]),
                float(x[index + 1]),
                float(deficit[index + 1]),
                half,
            )
            break
    width = None if left is None or right is None else float(right - left)
    return {
        "baseline": baseline,
        "peak": peak,
        "half_level": half,
        "left_half_max_mm": left,
        "right_half_max_mm": right,
        "apparent_half_max_width_mm": width,
    }


def _positive_profile_fwhm(
    x: np.ndarray,
    response: np.ndarray,
) -> float | None:
    x = np.asarray(x, dtype=np.float64)
    response = np.asarray(response, dtype=np.float64)
    finite = np.isfinite(x) & np.isfinite(response)
    x = x[finite]
    response = response[finite]
    if len(x) < 3 or np.max(response) <= np.min(response):
        return None
    response = (response - np.min(response)) / (
        np.max(response) - np.min(response)
    )
    peak = int(np.argmax(response))
    left = None
    right = None
    for index in range(peak, 0, -1):
        if response[index - 1] <= 0.5 <= response[index]:
            left = _interpolate_crossing(
                float(x[index - 1]),
                float(response[index - 1]),
                float(x[index]),
                float(response[index]),
                0.5,
            )
            break
    for index in range(peak, len(x) - 1):
        if response[index] >= 0.5 >= response[index + 1]:
            right = _interpolate_crossing(
                float(x[index]),
                float(response[index]),
                float(x[index + 1]),
                float(response[index + 1]),
                0.5,
            )
            break
    return None if left is None or right is None else float(right - left)


def _tv_filter_density(
    density: np.ndarray,
    *,
    weight: float,
) -> np.ndarray:
    finite = np.isfinite(density)
    filled = np.where(finite, density, 1.0)
    filtered = denoise_tv_chambolle(filled, weight=float(weight))
    filtered[~finite] = np.nan
    return filtered


def _image_metrics(
    density: np.ndarray,
    masks: dict[str, np.ndarray],
    grid: ReconstructionGrid2D,
) -> dict[str, Any]:
    density = np.asarray(density, dtype=np.float64).copy()
    density[~masks["support"]] = np.nan
    normalized = _normalize_density(density, masks["controls"])
    deficit = 1.0 - normalized
    roi_values = normalized[masks["roi"] & np.isfinite(normalized)]
    control_values = normalized[
        masks["controls"] & np.isfinite(normalized)
    ]
    field_values = deficit[masks["field"] & np.isfinite(deficit)]
    if len(roi_values) and len(control_values):
        contrast = float(1.0 - np.mean(roi_values) / np.mean(control_values))
        pooled_std = float(
            np.sqrt(0.5 * (np.var(roi_values) + np.var(control_values)))
        )
        separation = (
            float((np.mean(control_values) - np.mean(roi_values)) / pooled_std)
            if pooled_std > 0.0
            else None
        )
    else:
        contrast = None
        separation = None

    weights = np.clip(deficit, 0.0, None)
    weights = np.where(masks["search"] & np.isfinite(weights), weights, 0.0)
    total = float(weights.sum())
    if total > 0.0:
        center_x = float(np.sum(weights * masks["x"]) / total)
        center_depth = float(np.sum(weights * masks["depth"]) / total)
    else:
        center_x = None
        center_depth = None

    profile = np.full(len(grid.x_centers_mm), np.nan)
    for index in range(len(grid.x_centers_mm)):
        keep = masks["profile_band"][:, index] & np.isfinite(deficit[:, index])
        if np.any(keep):
            profile[index] = float(np.mean(deficit[keep, index]))
    width = _profile_width(grid.x_centers_mm, profile)
    return {
        "roi_mean_relative_density": float(np.mean(roi_values))
        if len(roi_values)
        else None,
        "control_mean_relative_density": float(np.mean(control_values))
        if len(control_values)
        else None,
        "image_domain_contrast": contrast,
        "image_domain_standardized_separation": separation,
        "field_deficit_rms": float(np.sqrt(np.mean(field_values**2)))
        if len(field_values)
        else None,
        "field_abs_deficit_p95": float(np.percentile(np.abs(field_values), 95))
        if len(field_values)
        else None,
        "positive_deficit_centroid_x_mm": center_x,
        "positive_deficit_centroid_depth_mm": center_depth,
        "profile": profile,
        "profile_width": width,
        "normalized_density": normalized,
        "deficit": deficit,
    }


def _jsonable_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in metrics.items()
        if key not in {"profile", "normalized_density", "deficit"}
    }


def _configure_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
            "font.size": 8.0,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )


def _diagnostic_figure(
    grid: ReconstructionGrid2D,
    masks: dict[str, np.ndarray],
    baseline: dict[str, Any],
    backprojection: dict[str, Any],
    advanced: dict[str, Any],
    reference_control: dict[str, Any],
) -> list[str]:
    _configure_plot_style()
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 6.7), constrained_layout=True)
    methods = [
        ("Closed-form point baseline", baseline),
        ("Probabilistic backprojection", backprojection),
        ("TV-regularized closed-form image", advanced),
        ("Homogeneous holdout control", reference_control),
    ]
    extent = [
        float(grid.x_edges_mm[0]),
        float(grid.x_edges_mm[-1]),
        float(grid.depth_edges_mm[-1]),
        float(grid.depth_edges_mm[0]),
    ]
    image = None
    for label, ax, (title, result) in zip("abcd", axes.ravel(), methods):
        image = ax.imshow(
            result["deficit"],
            extent=extent,
            origin="upper",
            cmap="magma",
            vmin=-0.08,
            vmax=0.32,
            interpolation="bilinear",
            rasterized=True,
        )
        circle = plt.Circle(
            (0.0, 30.0),
            10.0,
            fill=False,
            edgecolor="white",
            linewidth=1.2,
            linestyle=(0, (4, 2)),
        )
        ax.add_patch(circle)
        ax.set_xlim(-65.0, 65.0)
        ax.set_ylim(75.0, 5.0)
        ax.set_aspect("equal")
        ax.set_title(title, loc="left", fontweight="bold")
        ax.set_xlabel("Reconstructed x (mm)")
        ax.set_ylabel("Reconstructed depth (mm)")
        ax.text(
            -0.11,
            1.04,
            label,
            transform=ax.transAxes,
            fontweight="bold",
            fontsize=11,
        )
    assert image is not None
    colorbar = fig.colorbar(image, ax=axes.ravel().tolist(), shrink=0.83, pad=0.02)
    colorbar.set_label("Relative count-density deficit")
    base = OUTPUT_DIR / "algorithm_diagnostic"
    paths: list[str] = []
    for extension, dpi in (("svg", 300), ("pdf", 300), ("png", 300)):
        path = base.with_suffix(f".{extension}")
        fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
        paths.append(str(path.resolve()))
    plt.close(fig)
    return paths


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    config = load_config(CONFIG_PATH)
    grid = ReconstructionGrid2D(
        x_edges_mm=np.arange(-80.0, 80.0 + 2.0, 2.0),
        depth_edges_mm=np.arange(0.0, 100.0 + 2.0, 2.0),
    )

    void_pairs = _load_npz(VOID_DIR / "pairs.npz")
    reference_pairs = _load_npz(REFERENCE_DIR / "pairs.npz")
    void_reconstruction = _load_npz(VOID_DIR / "reconstruction.npz")
    reference_reconstruction = _load_npz(REFERENCE_DIR / "reconstruction.npz")
    void_selected = _observable_selection(void_pairs, config)
    reference_selected = _observable_selection(reference_pairs, config)

    reference_even = np.asarray(reference_pairs["event_id"], dtype=np.int64) % 2 == 0
    reference_odd = ~reference_even
    reference_train_selected = reference_selected & reference_even
    reference_holdout_selected = reference_selected & reference_odd

    common_model = {
        "incident_energy_keV": float(config["simulation"]["incident_energy_keV"]),
        "electron_rest_energy_keV": float(
            config["simulation"]["electron_rest_energy_keV"]
        ),
        "d1_energy_fwhm_fraction_at_511": float(
            config["detectors"]["d1"]["energy_response"][
                "fwhm_fraction_at_511"
            ]
        ),
        "d1_position_resolution_mm": tuple(
            config["detectors"]["d1"]["position_response"]["resolution_mm"]
        ),
        "d2_position_resolution_mm": tuple(
            config["detectors"]["d2"]["position_response"]["resolution_mm"]
        ),
        "y_half_width_mm": 150.0,
        "likelihood_sigma_limit": 4.0,
        "doppler_sigma_keV": 0.0,
        "include_geometric_weight": True,
    }
    print("[1/6] Building void event likelihoods ...", flush=True)
    void_system = build_ray_likelihood_system(
        void_pairs,
        void_selected,
        grid,
        **common_model,
    )
    print("[2/6] Building homogeneous-reference training likelihoods ...", flush=True)
    reference_train_system = build_ray_likelihood_system(
        reference_pairs,
        reference_train_selected,
        grid,
        **common_model,
    )
    print("[3/6] Building independent homogeneous holdout likelihoods ...", flush=True)
    reference_holdout_system = build_ray_likelihood_system(
        reference_pairs,
        reference_holdout_selected,
        grid,
        **common_model,
    )

    void_exposure = float(_load_primary_count(VOID_DIR))
    reference_total_exposure = float(_load_primary_count(REFERENCE_DIR))
    reference_train_exposure = reference_total_exposure / 2.0
    reference_holdout_exposure = reference_total_exposure / 2.0
    sensitivity = smooth_reference_sensitivity(
        reference_train_system.matrix,
        grid,
        reference_exposure=reference_train_exposure,
        gaussian_sigma_bins=1.0,
        floor_fraction_of_max=1e-5,
    )

    reference_backprojection = (
        np.asarray(reference_train_system.matrix.sum(axis=0)).ravel()
        + np.asarray(reference_holdout_system.matrix.sum(axis=0)).ravel()
    ).reshape(grid.shape)

    masks = _analysis_masks(
        grid,
        reference_backprojection,
        void_x_mm=float(config["phantom"]["void"]["center_mm"][0]),
        void_depth_mm=float(config["phantom"]["void"]["depth_mm"]),
        void_radius_mm=float(config["phantom"]["void"]["radius_mm"]),
    )

    print("[4/6] Computing the closed-form and probabilistic baselines ...", flush=True)
    baseline_void_counts = _baseline_point_histogram(
        void_pairs,
        void_reconstruction,
        void_selected,
        grid,
    )
    baseline_reference_counts = _baseline_point_histogram(
        reference_pairs,
        reference_reconstruction,
        reference_selected,
        grid,
    )
    baseline_density = _safe_relative_density(
        baseline_void_counts,
        baseline_reference_counts,
        target_exposure=void_exposure,
        reference_exposure=reference_total_exposure,
        smoothing_sigma_bins=1.0,
        minimum_reference_fraction_of_depth_max=0.03,
    )
    baseline_metrics = _image_metrics(baseline_density, masks, grid)

    baseline_reference_train_counts = _baseline_point_histogram(
        reference_pairs,
        reference_reconstruction,
        reference_selected,
        grid,
        event_subset=reference_even,
    )
    baseline_reference_holdout_counts = _baseline_point_histogram(
        reference_pairs,
        reference_reconstruction,
        reference_selected,
        grid,
        event_subset=reference_odd,
    )
    baseline_reference_control_density = _safe_relative_density(
        baseline_reference_holdout_counts,
        baseline_reference_train_counts,
        target_exposure=reference_holdout_exposure,
        reference_exposure=reference_train_exposure,
        smoothing_sigma_bins=1.0,
        minimum_reference_fraction_of_depth_max=0.03,
    )

    void_backprojection = np.asarray(
        void_system.matrix.sum(axis=0)
    ).ravel().reshape(grid.shape)
    backprojection_density = _safe_relative_density(
        void_backprojection,
        reference_backprojection,
        target_exposure=void_exposure,
        reference_exposure=reference_total_exposure,
        smoothing_sigma_bins=0.75,
        minimum_reference_fraction_of_depth_max=0.03,
    )
    backprojection_metrics = _image_metrics(backprojection_density, masks, grid)

    with ROD_METRICS.open("r", encoding="utf-8") as handle:
        rod_metrics = json.load(handle)
    rod_x = rod_metrics["transverse_x_response"]
    rod_ci = rod_x["bootstrap"]["fwhm_95_ci"]
    rod_curve_table = np.genfromtxt(
        ROD_CURVES,
        delimiter=",",
        names=True,
        dtype=None,
        encoding="utf-8-sig",
    )
    rod_keep = rod_curve_table["axis"] == "x"
    rod_coordinate = np.asarray(
        rod_curve_table["coordinate_mm"][rod_keep],
        dtype=np.float64,
    )
    rod_density = np.asarray(
        rod_curve_table["density_normalized"][rod_keep],
        dtype=np.float64,
    )

    tv_candidate_rows: list[dict[str, Any]] = []
    tv_candidate_maps: dict[float, dict[str, Any]] = {}
    for tv_weight in (0.0, 0.02, 0.05, 0.08, 0.12):
        filtered_void = _tv_filter_density(
            baseline_density,
            weight=tv_weight,
        )
        filtered_reference = _tv_filter_density(
            baseline_reference_control_density,
            weight=tv_weight,
        )
        filtered_rod = denoise_tv_chambolle(
            np.tile(rod_density, (3, 1)),
            weight=tv_weight,
        )[1]
        filtered_rod_fwhm = _positive_profile_fwhm(
            rod_coordinate,
            filtered_rod,
        )
        void_tv_metrics = _image_metrics(filtered_void, masks, grid)
        reference_tv_metrics = _image_metrics(filtered_reference, masks, grid)
        row = {
            "tv_weight": tv_weight,
            "reference_field_deficit_rms": reference_tv_metrics[
                "field_deficit_rms"
            ],
            "reference_false_roi_contrast": reference_tv_metrics[
                "image_domain_contrast"
            ],
            "rod_fwhm_after_regularization_mm": filtered_rod_fwhm,
            "rod_fwhm_requirement_mm": 5.0,
            "void_image_domain_contrast": void_tv_metrics[
                "image_domain_contrast"
            ],
            "void_standardized_separation": void_tv_metrics[
                "image_domain_standardized_separation"
            ],
            "void_apparent_half_max_width_mm": void_tv_metrics[
                "profile_width"
            ]["apparent_half_max_width_mm"],
        }
        tv_candidate_rows.append(row)
        tv_candidate_maps[tv_weight] = {
            "void": void_tv_metrics,
            "reference": reference_tv_metrics,
            "rod_response": filtered_rod,
        }
    accepted_tv_row = min(
        (
            row
            for row in tv_candidate_rows
            if row["rod_fwhm_after_regularization_mm"] is not None
            and row["rod_fwhm_after_regularization_mm"] <= 5.0
        ),
        key=lambda row: row["reference_field_deficit_rms"],
    )
    accepted_tv_weight = float(accepted_tv_row["tv_weight"])
    accepted_tv_metrics = tv_candidate_maps[accepted_tv_weight]["void"]
    accepted_tv_reference_metrics = tv_candidate_maps[accepted_tv_weight][
        "reference"
    ]
    grid_step_mm = float(np.median(np.diff(grid.x_edges_mm)))
    psf_sigma_bins = (
        float(rod_x["estimate"]["fwhm"]) / 2.354820045 / grid_step_mm
    )
    reference_full_system = reference_train_system.matrix.copy()
    from scipy.sparse import vstack

    reference_full_system = vstack(
        [reference_train_system.matrix, reference_holdout_system.matrix],
        format="csr",
    )
    matched_void = _psf_matched_probability_score(
        void_system.matrix,
        reference_full_system,
        grid,
        target_exposure=void_exposure,
        reference_exposure=reference_total_exposure,
        sigma_bins=psf_sigma_bins,
    )
    matched_reference_control = _psf_matched_probability_score(
        reference_holdout_system.matrix,
        reference_train_system.matrix,
        grid,
        target_exposure=reference_holdout_exposure,
        reference_exposure=reference_train_exposure,
        sigma_bins=psf_sigma_bins,
    )
    matched_void_metrics = _image_metrics(
        np.asarray(matched_void["relative_density"]),
        masks,
        grid,
    )
    matched_reference_metrics = _image_metrics(
        np.asarray(matched_reference_control["relative_density"]),
        masks,
        grid,
    )

    print("[5/6] Running LM-MLEM candidates and homogeneous holdout checks ...", flush=True)
    iteration_candidates = (1, 2, 4, 8)
    tv_weight_candidates = (0.0, 0.01, 0.02)
    candidate_rows: list[dict[str, Any]] = []
    candidate_maps: dict[str, dict[str, Any]] = {}
    for iterations in iteration_candidates:
        void_image, void_history = listmode_mlem(
            void_system.matrix,
            sensitivity,
            target_exposure=void_exposure,
            iterations=iterations,
        )
        holdout_image, holdout_history = listmode_mlem(
            reference_holdout_system.matrix,
            sensitivity,
            target_exposure=reference_holdout_exposure,
            iterations=iterations,
        )
        for tv_weight in tv_weight_candidates:
            void_2d = void_image.reshape(grid.shape)
            holdout_2d = holdout_image.reshape(grid.shape)
            if tv_weight > 0.0:
                void_2d = denoise_tv_chambolle(void_2d, weight=tv_weight)
                holdout_2d = denoise_tv_chambolle(
                    holdout_2d,
                    weight=tv_weight,
                )
            void_metrics = _image_metrics(void_2d, masks, grid)
            holdout_metrics = _image_metrics(holdout_2d, masks, grid)
            key = f"iter{iterations}_tv{tv_weight:.2f}"
            reference_score = float(
                holdout_metrics["field_deficit_rms"]
                + 0.25 * holdout_metrics["field_abs_deficit_p95"]
            )
            row = {
                "key": key,
                "iterations": iterations,
                "tv_weight": tv_weight,
                "reference_only_selection_score": reference_score,
                "void_image_domain_contrast": void_metrics[
                    "image_domain_contrast"
                ],
                "void_standardized_separation": void_metrics[
                    "image_domain_standardized_separation"
                ],
                "void_centroid_x_mm": void_metrics[
                    "positive_deficit_centroid_x_mm"
                ],
                "void_centroid_depth_mm": void_metrics[
                    "positive_deficit_centroid_depth_mm"
                ],
                "void_apparent_half_max_width_mm": void_metrics[
                    "profile_width"
                ]["apparent_half_max_width_mm"],
                "reference_field_deficit_rms": holdout_metrics[
                    "field_deficit_rms"
                ],
                "reference_field_abs_deficit_p95": holdout_metrics[
                    "field_abs_deficit_p95"
                ],
                "reference_false_roi_contrast": holdout_metrics[
                    "image_domain_contrast"
                ],
            }
            candidate_rows.append(row)
            candidate_maps[key] = {
                "void": void_metrics,
                "holdout": holdout_metrics,
                "void_history": void_history,
                "holdout_history": holdout_history,
            }

    # Early stopping and regularization are selected only with the independent
    # homogeneous holdout.  The void truth and void metrics do not enter this score.
    eligible = [row for row in candidate_rows if row["iterations"] >= 2]
    selected_row = min(
        eligible,
        key=lambda row: (
            row["reference_only_selection_score"],
            row["tv_weight"],
            row["iterations"],
        ),
    )
    selected_key = str(selected_row["key"])
    advanced_metrics = candidate_maps[selected_key]["void"]
    holdout_metrics = candidate_maps[selected_key]["holdout"]

    print("[6/6] Writing source data, report and diagnostic figure ...", flush=True)
    candidate_csv = OUTPUT_DIR / "algorithm_candidate_metrics.csv"
    with candidate_csv.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(candidate_rows[0]))
        writer.writeheader()
        writer.writerows(candidate_rows)

    tv_candidate_csv = OUTPUT_DIR / "tv_candidate_metrics.csv"
    with tv_candidate_csv.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(tv_candidate_rows[0]))
        writer.writeheader()
        writer.writerows(tv_candidate_rows)

    profiles_csv = OUTPUT_DIR / "algorithm_profiles.csv"
    with profiles_csv.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "x_mm",
                "closed_form_deficit",
                "probabilistic_backprojection_deficit",
                "advanced_lmmlem_tv_deficit",
                "homogeneous_holdout_deficit",
                "psf_matched_probability_deficit",
                "psf_matched_homogeneous_control_deficit",
                "accepted_tv_deficit",
                "accepted_tv_homogeneous_control_deficit",
            ]
        )
        writer.writerows(
            zip(
                grid.x_centers_mm,
                baseline_metrics["profile"],
                backprojection_metrics["profile"],
                advanced_metrics["profile"],
                holdout_metrics["profile"],
                matched_void_metrics["profile"],
                matched_reference_metrics["profile"],
                accepted_tv_metrics["profile"],
                accepted_tv_reference_metrics["profile"],
            )
        )

    np.savez_compressed(
        OUTPUT_DIR / "advanced_reconstruction_maps.npz",
        x_edges_mm=grid.x_edges_mm,
        depth_edges_mm=grid.depth_edges_mm,
        sensitivity_per_primary=sensitivity.reshape(grid.shape),
        baseline_normalized_density=baseline_metrics["normalized_density"],
        baseline_deficit=baseline_metrics["deficit"],
        backprojection_normalized_density=backprojection_metrics[
            "normalized_density"
        ],
        backprojection_deficit=backprojection_metrics["deficit"],
        advanced_normalized_density=advanced_metrics["normalized_density"],
        advanced_deficit=advanced_metrics["deficit"],
        homogeneous_holdout_normalized_density=holdout_metrics[
            "normalized_density"
        ],
        homogeneous_holdout_deficit=holdout_metrics["deficit"],
        psf_matched_probability_normalized_density=matched_void_metrics[
            "normalized_density"
        ],
        psf_matched_probability_deficit=matched_void_metrics["deficit"],
        psf_matched_probability_zscore=np.asarray(
            matched_void["standardized_deficit"]
        ),
        psf_matched_homogeneous_control_normalized_density=(
            matched_reference_metrics["normalized_density"]
        ),
        psf_matched_homogeneous_control_deficit=matched_reference_metrics[
            "deficit"
        ],
        psf_matched_homogeneous_control_zscore=np.asarray(
            matched_reference_control["standardized_deficit"]
        ),
        accepted_tv_normalized_density=accepted_tv_metrics[
            "normalized_density"
        ],
        accepted_tv_deficit=accepted_tv_metrics["deficit"],
        accepted_tv_homogeneous_control_normalized_density=(
            accepted_tv_reference_metrics["normalized_density"]
        ),
        accepted_tv_homogeneous_control_deficit=(
            accepted_tv_reference_metrics["deficit"]
        ),
    )
    figure_paths = _diagnostic_figure(
        grid,
        masks,
        baseline_metrics,
        backprojection_metrics,
        accepted_tv_metrics,
        accepted_tv_reference_metrics,
    )

    report: dict[str, Any] = {
        "schema_version": "1.0",
        "method_name": "reference-calibrated probabilistic list-mode MLEM prototype",
        "method_scope": (
            "Adaptation of list-mode Compton reconstruction to a D2-tagged "
            "incident ray; it is not claimed as an exact reproduction of one paper."
        ),
        "configuration": str(CONFIG_PATH.resolve()),
        "grid": {
            "x_edges_mm": grid.x_edges_mm.tolist(),
            "depth_edges_mm": grid.depth_edges_mm.tolist(),
            "shape_depth_by_x": list(grid.shape),
        },
        "event_model": common_model,
        "reference_split": {
            "rule": "GATE event_id parity; even=train sensitivity, odd=holdout control",
            "reference_total_primaries": reference_total_exposure,
            "train_exposure_primaries": reference_train_exposure,
            "holdout_exposure_primaries": reference_holdout_exposure,
            "void_exposure_primaries": void_exposure,
        },
        "systems": {
            "void": void_system.diagnostics,
            "reference_train": reference_train_system.diagnostics,
            "reference_holdout": reference_holdout_system.diagnostics,
        },
        "selection": {
            "chosen_key": selected_key,
            "criterion": (
                "minimum homogeneous-holdout field RMS + 0.25*p95 absolute "
                "deficit among candidates with at least two iterations; void "
                "truth and void metrics were excluded"
            ),
            "chosen_row": selected_row,
            "lmmlem_acceptance": {
                "status": "not_accepted_as_primary_image",
                "reason": (
                    "The selected LM-MLEM candidate increased the independent "
                    "homogeneous-holdout field artifact RMS relative to the "
                    "closed-form baseline; it is retained as an algorithm trial."
                ),
            },
            "accepted_primary_image": {
                "method": "TV-regularized closed-form count-density reconstruction",
                "tv_weight": accepted_tv_weight,
                "selection_rule": (
                    "minimum independent homogeneous-holdout artifact RMS "
                    "within the predeclared 0-0.12 weight grid, subject to the "
                    "independent rod FWHM remaining <=5 mm"
                ),
                "selected_candidate": accepted_tv_row,
            },
        },
        "metrics": {
            "closed_form_baseline": _jsonable_metrics(baseline_metrics),
            "probabilistic_backprojection": _jsonable_metrics(
                backprojection_metrics
            ),
            "advanced_void": _jsonable_metrics(advanced_metrics),
            "homogeneous_holdout_control": _jsonable_metrics(holdout_metrics),
            "accepted_psf_matched_void": _jsonable_metrics(
                matched_void_metrics
            ),
            "accepted_psf_matched_homogeneous_control": _jsonable_metrics(
                matched_reference_metrics
            ),
            "accepted_tv_void": _jsonable_metrics(accepted_tv_metrics),
            "accepted_tv_homogeneous_control": _jsonable_metrics(
                accepted_tv_reference_metrics
            ),
            "independent_2mm_rod_closed_form_guardrail": {
                "fwhm_mm": float(rod_x["estimate"]["fwhm"]),
                "bootstrap_95_ci_mm": [float(rod_ci[0]), float(rod_ci[1])],
                "requirement_mm": 5.0,
                "status": "pass"
                if float(rod_x["estimate"]["fwhm"]) < 5.0
                else "fail",
                "note": (
                    "The independent rod remains the formal system-resolution "
                    "guardrail. The void apparent width is not called FWHM resolution."
                ),
            },
        },
        "source_data": {
            "candidate_metrics_csv": str(candidate_csv.resolve()),
            "tv_candidate_metrics_csv": str(tv_candidate_csv.resolve()),
            "profiles_csv": str(profiles_csv.resolve()),
            "maps_npz": str(
                (OUTPUT_DIR / "advanced_reconstruction_maps.npz").resolve()
            ),
        },
        "figures": figure_paths,
        "guardrails": [
            "No Monte-Carlo truth field is used by event selection or reconstruction.",
            "No local pixel editing is applied.",
            "TV is a global deterministic post-regularizer and its weight is recorded.",
            "The reference-derived sensitivity is an empirical flat-field, not yet a full attenuation/multiple-scatter system matrix.",
            "Doppler broadening is not yet parameterized; its sigma is fixed to zero in this prototype.",
        ],
    }
    with (OUTPUT_DIR / "advanced_reconstruction_report.json").open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    print(json.dumps(report["selection"], ensure_ascii=False, indent=2))
    print(json.dumps(report["metrics"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
