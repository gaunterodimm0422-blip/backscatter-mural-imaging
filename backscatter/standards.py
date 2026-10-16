from __future__ import annotations

import copy
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import minimize
from scipy.signal import find_peaks
from scipy.special import erf
from scipy.stats import gaussian_kde

from .config import load_config


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {name: data[name] for name in data.files}


def _crossing(
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    level: float,
) -> float:
    if y1 == y0:
        return 0.5 * (x0 + x1)
    return x0 + (level - y0) * (x1 - x0) / (y1 - y0)


def estimate_fwhm_kde(
    values: np.ndarray,
    *,
    bandwidth_scale: float = 1.0,
    grid_size: int = 2048,
) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if len(values) < 20:
        return {
            "status": "insufficient_events",
            "n": int(len(values)),
            "fwhm": None,
        }
    if not np.isfinite(np.std(values)) or np.std(values) <= 0.0:
        return {
            "status": "zero_width",
            "n": int(len(values)),
            "fwhm": None,
        }
    if bandwidth_scale <= 0.0:
        raise ValueError("bandwidth_scale must be positive")

    q_lo, q_hi = np.quantile(values, [0.002, 0.998])
    span = max(float(q_hi - q_lo), float(np.std(values)), 1.0e-6)
    grid = np.linspace(q_lo - 0.15 * span, q_hi + 0.15 * span, grid_size)
    kde = gaussian_kde(
        values,
        bw_method=lambda obj: obj.scotts_factor() * bandwidth_scale,
    )
    density = np.asarray(kde(grid), dtype=np.float64)
    peak_index = int(np.argmax(density))
    peak_density = float(density[peak_index])
    half = 0.5 * peak_density

    peaks, properties = find_peaks(
        density,
        height=half,
        prominence=0.05 * peak_density,
    )
    if len(peaks) > 1:
        return {
            "status": "multiple_peaks_above_half_maximum",
            "n": int(len(values)),
            "fwhm": None,
            "grid": grid,
            "density": density,
            "mode": float(grid[peak_index]),
            "peak_count_above_half": int(len(peaks)),
        }

    above = density >= half
    left_below = np.flatnonzero(~above[:peak_index])
    right_below_relative = np.flatnonzero(~above[peak_index + 1 :])
    if len(left_below) == 0 or len(right_below_relative) == 0:
        return {
            "status": "half_maximum_crossing_not_bracketed",
            "n": int(len(values)),
            "fwhm": None,
            "grid": grid,
            "density": density,
            "mode": float(grid[peak_index]),
            "peak_count_above_half": int(len(peaks)),
        }

    left0 = int(left_below[-1])
    left1 = left0 + 1
    right1 = peak_index + 1 + int(right_below_relative[0])
    right0 = right1 - 1
    left = _crossing(
        float(grid[left0]),
        float(density[left0]),
        float(grid[left1]),
        float(density[left1]),
        half,
    )
    right = _crossing(
        float(grid[right0]),
        float(density[right0]),
        float(grid[right1]),
        float(density[right1]),
        half,
    )
    return {
        "status": "ok",
        "n": int(len(values)),
        "fwhm": float(right - left),
        "left_half_max": float(left),
        "right_half_max": float(right),
        "mode": float(grid[peak_index]),
        "peak_density": peak_density,
        "half_max_density": half,
        "peak_count_above_half": int(len(peaks)),
        "bandwidth_scale": float(bandwidth_scale),
        "grid": grid,
        "density": density,
    }


def bootstrap_fwhm(
    values: np.ndarray,
    *,
    n_bootstrap: int = 500,
    seed: int = 24061873,
    bandwidth_scale: float = 1.0,
) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if n_bootstrap <= 0:
        return {
            "requested": int(n_bootstrap),
            "valid": 0,
            "fwhm_95_ci": None,
            "mode_95_ci": None,
        }
    if len(values) < 20:
        return {
            "requested": int(n_bootstrap),
            "valid": 0,
            "fwhm_95_ci": None,
            "mode_95_ci": None,
        }

    rng = np.random.default_rng(seed)
    fwhm_values: list[float] = []
    modes: list[float] = []
    for _ in range(n_bootstrap):
        sample = values[rng.integers(0, len(values), size=len(values))]
        estimate = estimate_fwhm_kde(
            sample,
            bandwidth_scale=bandwidth_scale,
            grid_size=512,
        )
        if estimate["status"] == "ok":
            fwhm_values.append(float(estimate["fwhm"]))
            modes.append(float(estimate["mode"]))

    if not fwhm_values:
        return {
            "requested": int(n_bootstrap),
            "valid": 0,
            "fwhm_95_ci": None,
            "mode_95_ci": None,
        }
    return {
        "requested": int(n_bootstrap),
        "valid": int(len(fwhm_values)),
        "fwhm_95_ci": np.quantile(fwhm_values, [0.025, 0.975]).tolist(),
        "mode_95_ci": np.quantile(modes, [0.025, 0.975]).tolist(),
    }


def _write_curve_csv(
    path: Path,
    grid: np.ndarray,
    density: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    maximum = float(np.max(density))
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["depth_mm", "density", "density_normalized"])
        for x, y in zip(grid, density):
            writer.writerow([float(x), float(y), float(y / maximum)])


def _plot_thin_layer(
    *,
    depth_values: np.ndarray,
    depth_error: np.ndarray,
    target_interval: tuple[float, float],
    target_center: float,
    estimate: dict[str, Any],
    bootstrap: dict[str, Any],
    efficiency: float,
    output_base: Path,
) -> list[str]:
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
    plt.rcParams["legend.frameon"] = False

    mm_to_inch = 1.0 / 25.4
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(180.0 * mm_to_inch, 72.0 * mm_to_inch),
        gridspec_kw={"width_ratios": [1.55, 1.0]},
    )
    blue = "#0F4D92"
    blue_light = "#B4CFE8"
    neutral = "#767676"

    ax = axes[0]
    grid = np.asarray(estimate["grid"])
    density = np.asarray(estimate["density"])
    normalized = density / np.max(density)
    ax.fill_between(grid, normalized, color=blue_light, alpha=0.65, linewidth=0)
    ax.plot(grid, normalized, color=blue, linewidth=1.4)
    ax.axvspan(
        target_interval[0],
        target_interval[1],
        color="#D8D8D8",
        alpha=0.9,
        label="1 mm target",
    )
    ax.axvline(target_center, color=neutral, linestyle="--", linewidth=0.9)
    if estimate["status"] == "ok":
        left = float(estimate["left_half_max"])
        right = float(estimate["right_half_max"])
        ax.hlines(0.5, left, right, color="#B64342", linewidth=1.4)
        ax.vlines([left, right], 0.46, 0.54, color="#B64342", linewidth=1.0)
    ax.set_xlabel("Reconstructed axial distance from z = 0 (mm)")
    ax.set_ylabel("Normalized response")
    ax.set_ylim(0.0, 1.08)
    ax.text(
        -0.11,
        1.03,
        "a",
        transform=ax.transAxes,
        fontweight="bold",
        fontsize=9,
    )
    ci = bootstrap.get("fwhm_95_ci")
    ci_text = (
        f" [{ci[0]:.2f}, {ci[1]:.2f}]"
        if ci is not None
        else ""
    )
    fwhm_text = (
        f"FWHM = {estimate['fwhm']:.2f} mm{ci_text}"
        if estimate["status"] == "ok"
        else f"FWHM unavailable: {estimate['status']}"
    )
    bias = float(estimate.get("mode", np.nan) - target_center)
    ax.text(
        0.02,
        0.96,
        f"{fwhm_text}\nPeak bias = {bias:+.2f} mm\n"
        f"n = {len(depth_values):,}; efficiency = {efficiency:.3e}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=6.4,
    )

    ax = axes[1]
    finite_error = depth_error[np.isfinite(depth_error)]
    if len(finite_error):
        bins = np.linspace(
            np.quantile(finite_error, 0.005),
            np.quantile(finite_error, 0.995),
            45,
        )
        ax.hist(
            finite_error,
            bins=bins,
            density=True,
            color=blue_light,
            edgecolor=blue,
            linewidth=0.45,
        )
        ax.axvline(0.0, color=neutral, linestyle="--", linewidth=0.9)
        rmse = float(np.sqrt(np.mean(finite_error**2)))
        median = float(np.median(finite_error))
        p90 = float(np.quantile(np.abs(finite_error), 0.9))
        ax.text(
            0.98,
            0.96,
            f"RMSE = {rmse:.2f} mm\nMedian = {median:+.2f} mm\n"
            f"|error| P90 = {p90:.2f} mm",
            transform=ax.transAxes,
            va="top",
            ha="right",
            fontsize=6.4,
        )
    ax.set_xlabel("Event-wise depth error (mm)")
    ax.set_ylabel("Density")
    ax.text(
        -0.15,
        1.03,
        "b",
        transform=ax.transAxes,
        fontweight="bold",
        fontsize=9,
    )

    fig.tight_layout(pad=1.2, w_pad=2.0)
    output_base.parent.mkdir(parents=True, exist_ok=True)
    saved = []
    for extension, dpi in (
        ("svg", 300),
        ("pdf", 300),
        ("tiff", 600),
        ("png", 300),
    ):
        path = output_base.with_suffix(f".{extension}")
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        saved.append(str(path.resolve()))
    plt.close(fig)
    return saved


def _plot_rod(
    *,
    x_values: np.ndarray,
    depth_values: np.ndarray,
    x_error: np.ndarray,
    target_x_interval: tuple[float, float],
    target_depth_interval: tuple[float, float],
    target_depth: float,
    x_estimate: dict[str, Any],
    depth_estimate: dict[str, Any],
    x_bootstrap: dict[str, Any],
    efficiency: float,
    output_base: Path,
) -> list[str]:
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
    plt.rcParams["legend.frameon"] = False

    mm_to_inch = 1.0 / 25.4
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(180.0 * mm_to_inch, 65.0 * mm_to_inch),
    )
    blue = "#0F4D92"
    blue_light = "#B4CFE8"
    red = "#B64342"
    neutral = "#767676"
    band = "#D8D8D8"

    def response_panel(
        ax: Any,
        estimate: dict[str, Any],
        interval: tuple[float, float],
        truth_center: float,
        xlabel: str,
    ) -> None:
        grid = np.asarray(estimate["grid"])
        density = np.asarray(estimate["density"])
        normalized = density / np.max(density)
        ax.fill_between(
            grid,
            normalized,
            color=blue_light,
            alpha=0.65,
            linewidth=0,
        )
        ax.plot(grid, normalized, color=blue, linewidth=1.3)
        ax.axvspan(interval[0], interval[1], color=band, alpha=0.85)
        ax.axvline(
            truth_center,
            color=neutral,
            linestyle="--",
            linewidth=0.8,
        )
        if estimate["status"] == "ok":
            left = float(estimate["left_half_max"])
            right = float(estimate["right_half_max"])
            ax.hlines(0.5, left, right, color=red, linewidth=1.3)
            ax.vlines(
                [left, right],
                0.46,
                0.54,
                color=red,
                linewidth=0.9,
            )
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Normalized response")
        ax.set_ylim(0.0, 1.08)

    response_panel(
        axes[0],
        x_estimate,
        target_x_interval,
        0.0,
        "Reconstructed x (mm)",
    )
    ci = x_bootstrap.get("fwhm_95_ci")
    ci_text = f" [{ci[0]:.2f}, {ci[1]:.2f}]" if ci else ""
    axes[0].text(
        0.02,
        0.96,
        (
            f"x FWHM = {x_estimate['fwhm']:.2f} mm{ci_text}\n"
            f"Peak bias = {x_estimate['mode']:+.2f} mm\n"
            f"n = {len(x_values):,}; efficiency = {efficiency:.3e}"
            if x_estimate["status"] == "ok"
            else f"x FWHM unavailable: {x_estimate['status']}"
        ),
        transform=axes[0].transAxes,
        va="top",
        fontsize=6.2,
    )

    response_panel(
        axes[1],
        depth_estimate,
        target_depth_interval,
        target_depth,
        "Reconstructed depth (mm)",
    )
    axes[1].text(
        0.98,
        0.96,
        (
            f"Depth FWHM = {depth_estimate['fwhm']:.2f} mm\n"
            f"Peak bias = {depth_estimate['mode'] - target_depth:+.2f} mm"
            if depth_estimate["status"] == "ok"
            else f"Depth FWHM unavailable:\n{depth_estimate['status']}"
        ),
        transform=axes[1].transAxes,
        va="top",
        ha="right",
        fontsize=6.2,
    )

    finite_error = x_error[np.isfinite(x_error)]
    ax = axes[2]
    if len(finite_error):
        bins = np.linspace(
            np.quantile(finite_error, 0.005),
            np.quantile(finite_error, 0.995),
            45,
        )
        ax.hist(
            finite_error,
            bins=bins,
            density=True,
            color=blue_light,
            edgecolor=blue,
            linewidth=0.45,
        )
        ax.axvline(0.0, color=neutral, linestyle="--", linewidth=0.8)
        ax.text(
            0.98,
            0.96,
            f"RMSE = {np.sqrt(np.mean(finite_error**2)):.2f} mm\n"
            f"|error| P90 = {np.quantile(np.abs(finite_error), 0.9):.2f} mm",
            transform=ax.transAxes,
            va="top",
            ha="right",
            fontsize=6.2,
        )
    ax.set_xlabel("Event-wise x error (mm)")
    ax.set_ylabel("Density")

    for label, ax in zip("abc", axes):
        ax.text(
            -0.17,
            1.04,
            label,
            transform=ax.transAxes,
            fontweight="bold",
            fontsize=9,
        )
    fig.tight_layout(pad=1.2, w_pad=2.0)
    output_base.parent.mkdir(parents=True, exist_ok=True)
    saved = []
    for extension, dpi in (
        ("svg", 300),
        ("pdf", 300),
        ("tiff", 600),
        ("png", 300),
    ):
        path = output_base.with_suffix(f".{extension}")
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        saved.append(str(path.resolve()))
    plt.close(fig)
    return saved


def _evaluate_rod_target(
    *,
    analysis_dir: Path,
    config_path: Path,
    config: dict[str, Any],
    out_dir: Path,
    n_bootstrap: int,
    bandwidth_scale: float,
) -> dict[str, Any]:
    reconstruction = _load_npz(analysis_dir / "reconstruction.npz")
    with (analysis_dir / "summary.json").open("r", encoding="utf-8") as handle:
        summary = json.load(handle)
    usable = (
        reconstruction["selected"].astype(bool)
        & reconstruction["valid"].astype(bool)
    )
    x_values = np.asarray(reconstruction["x_mm"][usable], dtype=np.float64)
    depth_values = -np.asarray(
        reconstruction["z_mm"][usable],
        dtype=np.float64,
    )
    truth_matched = reconstruction["truth_matched"].astype(bool)
    truth_usable = usable & truth_matched
    x_error = (
        reconstruction["x_mm"][truth_usable]
        - reconstruction["truth_x_mm"][truth_usable]
    ).astype(np.float64)
    depth_error = (
        -reconstruction["z_mm"][truth_usable]
        + reconstruction["truth_z_mm"][truth_usable]
    ).astype(np.float64)

    x_estimate = estimate_fwhm_kde(
        x_values,
        bandwidth_scale=bandwidth_scale,
    )
    depth_estimate = estimate_fwhm_kde(
        depth_values,
        bandwidth_scale=bandwidth_scale,
    )
    x_bootstrap = bootstrap_fwhm(
        x_values,
        n_bootstrap=n_bootstrap,
        bandwidth_scale=bandwidth_scale,
    )
    depth_bootstrap = bootstrap_fwhm(
        depth_values,
        n_bootstrap=n_bootstrap,
        bandwidth_scale=bandwidth_scale,
    )
    radius = float(config["phantom"]["radius_mm"])
    target_depth = float(config["phantom"]["depth_mm"])
    n_primary = int(summary["event_counts"]["primaries"])
    efficiency = float(len(x_values) / n_primary)

    curve_csv = out_dir / "rod_response_curves.csv"
    with curve_csv.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["axis", "coordinate_mm", "density", "density_normalized"])
        for axis, estimate in (
            ("x", x_estimate),
            ("depth", depth_estimate),
        ):
            if "grid" not in estimate:
                continue
            density = np.asarray(estimate["density"])
            maximum = float(np.max(density))
            for coordinate, value in zip(estimate["grid"], density):
                writer.writerow(
                    [
                        axis,
                        float(coordinate),
                        float(value),
                        float(value / maximum),
                    ]
                )

    def compact(estimate: dict[str, Any]) -> dict[str, Any]:
        return {
            key: value
            for key, value in estimate.items()
            if key not in {"grid", "density"}
        }

    report: dict[str, Any] = {
        "schema_version": "1.0",
        "kind": "rod",
        "analysis_dir": str(analysis_dir.resolve()),
        "config": str(config_path.resolve()),
        "target_depth_mm": target_depth,
        "target_diameter_mm": 2.0 * radius,
        "n_primaries": n_primary,
        "n_selected_valid": int(len(x_values)),
        "efficiency_per_primary": efficiency,
        "transverse_x_response": {
            "estimate": compact(x_estimate),
            "bootstrap": x_bootstrap,
        },
        "axial_depth_response": {
            "estimate": compact(depth_estimate),
            "bootstrap": depth_bootstrap,
        },
        "event_truth_error": {
            "n": int(len(x_error)),
            "x_bias_mm": float(np.mean(x_error)) if len(x_error) else None,
            "x_rmse_mm": (
                float(np.sqrt(np.mean(x_error**2))) if len(x_error) else None
            ),
            "depth_bias_mm": (
                float(np.mean(depth_error)) if len(depth_error) else None
            ),
            "depth_rmse_mm": (
                float(np.sqrt(np.mean(depth_error**2)))
                if len(depth_error)
                else None
            ),
        },
        "source_data": str(curve_csv.resolve()),
        "figures": [],
        "interpretation_guardrail": (
            "The measured line-spread width includes the finite concrete rod "
            "diameter; no target-width deconvolution was applied."
        ),
    }
    if "grid" in x_estimate and "grid" in depth_estimate:
        report["figures"] = _plot_rod(
            x_values=x_values,
            depth_values=depth_values,
            x_error=x_error,
            target_x_interval=(-radius, radius),
            target_depth_interval=(target_depth - radius, target_depth + radius),
            target_depth=target_depth,
            x_estimate=x_estimate,
            depth_estimate=depth_estimate,
            x_bootstrap=x_bootstrap,
            efficiency=efficiency,
            output_base=out_dir / "rod_response",
        )
    with (out_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    return report


def _measure_double_rod_response(
    *,
    x_values: np.ndarray,
    centers: np.ndarray,
    bandwidth_scale: float,
) -> dict[str, Any]:
    """Measure two-peak and valley metrics for one KDE bandwidth."""
    midpoint = float(np.mean(centers))
    separation = float(centers[1] - centers[0])
    q_lo, q_hi = np.quantile(x_values, [0.002, 0.998])
    lo = min(float(q_lo), float(centers[0] - 2.0 * separation))
    hi = max(float(q_hi), float(centers[1] + 2.0 * separation))
    grid = np.linspace(lo, hi, 2048)
    kde = gaussian_kde(
        x_values,
        bw_method=lambda obj: obj.scotts_factor() * bandwidth_scale,
    )
    density = np.asarray(kde(grid), dtype=np.float64)
    peak_indices, _ = find_peaks(
        density,
        prominence=0.03 * float(np.max(density)),
    )
    left_candidates = peak_indices[grid[peak_indices] < midpoint]
    right_candidates = peak_indices[grid[peak_indices] > midpoint]
    empty_metrics = {
        "left_peak_x_mm": None,
        "right_peak_x_mm": None,
        "measured_peak_separation_mm": None,
        "valley_x_mm": None,
        "valley_to_mean_peak": None,
        "valley_to_lower_peak": None,
        "operationally_resolved_at_0p735": False,
    }
    if len(left_candidates) == 0 or len(right_candidates) == 0:
        return {
            "grid": grid,
            "density": density,
            "status": "two_peaks_not_detected",
            "chosen_indices": None,
            "metrics": empty_metrics,
        }

    left_index = int(
        left_candidates[np.argmax(density[left_candidates])]
    )
    right_index = int(
        right_candidates[np.argmax(density[right_candidates])]
    )
    if right_index <= left_index:
        return {
            "grid": grid,
            "density": density,
            "status": "invalid_peak_order",
            "chosen_indices": None,
            "metrics": empty_metrics,
        }

    valley_index = left_index + int(
        np.argmin(density[left_index : right_index + 1])
    )
    peak_values = np.asarray(
        [density[left_index], density[right_index]]
    )
    valley_value = float(density[valley_index])
    ratio_mean = valley_value / float(np.mean(peak_values))
    ratio_lower = valley_value / float(np.min(peak_values))
    return {
        "grid": grid,
        "density": density,
        "status": "ok",
        "chosen_indices": (left_index, right_index, valley_index),
        "metrics": {
            "left_peak_x_mm": float(grid[left_index]),
            "right_peak_x_mm": float(grid[right_index]),
            "measured_peak_separation_mm": float(
                grid[right_index] - grid[left_index]
            ),
            "valley_x_mm": float(grid[valley_index]),
            "valley_to_mean_peak": float(ratio_mean),
            "valley_to_lower_peak": float(ratio_lower),
            "operationally_resolved_at_0p735": bool(
                ratio_lower <= 0.735
            ),
        },
    }


def _evaluate_double_rod_target(
    *,
    analysis_dir: Path,
    config_path: Path,
    config: dict[str, Any],
    out_dir: Path,
    bandwidth_scale: float,
) -> dict[str, Any]:
    reconstruction = _load_npz(analysis_dir / "reconstruction.npz")
    with (analysis_dir / "summary.json").open("r", encoding="utf-8") as handle:
        summary = json.load(handle)
    usable = (
        reconstruction["selected"].astype(bool)
        & reconstruction["valid"].astype(bool)
    )
    x_values = np.asarray(reconstruction["x_mm"][usable], dtype=np.float64)
    if len(x_values) < 20:
        raise ValueError("double-rod analysis has fewer than 20 selected events")

    centers = np.asarray(
        [
            point[0]
            for point in config["phantom"]["target_centers_mm"]
        ],
        dtype=np.float64,
    )
    centers.sort()
    if len(centers) != 2:
        raise ValueError("double-rod config must contain exactly two target centers")
    separation = float(centers[1] - centers[0])
    radius = float(config["phantom"]["radius_mm"])

    primary_measurement = _measure_double_rod_response(
        x_values=x_values,
        centers=centers,
        bandwidth_scale=bandwidth_scale,
    )
    grid = primary_measurement["grid"]
    density = primary_measurement["density"]
    status = primary_measurement["status"]
    chosen = primary_measurement["chosen_indices"]
    valley_metrics = primary_measurement["metrics"]
    bandwidth_sensitivity: list[dict[str, Any]] = []
    for multiplier in (0.75, 1.0, 1.25):
        measurement = (
            primary_measurement
            if multiplier == 1.0
            else _measure_double_rod_response(
                x_values=x_values,
                centers=centers,
                bandwidth_scale=bandwidth_scale * multiplier,
            )
        )
        bandwidth_sensitivity.append(
            {
                "bandwidth_multiplier": float(multiplier),
                "effective_bandwidth_scale": float(
                    bandwidth_scale * multiplier
                ),
                "two_peak_status": measurement["status"],
                **measurement["metrics"],
            }
        )
    bandwidth_robust_resolved = bool(
        all(
            row["operationally_resolved_at_0p735"]
            for row in bandwidth_sensitivity
        )
    )

    truth_matched = reconstruction["truth_matched"].astype(bool)
    truth_usable = usable & truth_matched
    truth_x = np.asarray(
        reconstruction["truth_x_mm"][truth_usable],
        dtype=np.float64,
    )
    reconstructed_truth_x = np.asarray(
        reconstruction["x_mm"][truth_usable],
        dtype=np.float64,
    )
    nearest = (
        np.abs(truth_x[:, None] - centers[None, :]).argmin(axis=1)
        if len(truth_x)
        else np.empty(0, dtype=np.int64)
    )
    labelled: list[dict[str, Any]] = []
    for label, center in enumerate(centers):
        values = reconstructed_truth_x[nearest == label]
        estimate = estimate_fwhm_kde(
            values,
            bandwidth_scale=bandwidth_scale,
        )
        labelled.append(
            {
                "target_center_x_mm": float(center),
                "n": int(len(values)),
                "fwhm_status": estimate["status"],
                "reconstructed_mode_x_mm": estimate.get("mode"),
                "fwhm_x_mm": estimate.get("fwhm"),
                "localization_bias_x_mm": (
                    float(estimate["mode"] - center)
                    if estimate.get("mode") is not None
                    else None
                ),
            }
        )

    curve_csv = out_dir / "double_rod_x_response.csv"
    with curve_csv.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["x_mm", "density", "density_normalized"])
        maximum = float(np.max(density))
        for coordinate, value in zip(grid, density):
            writer.writerow(
                [float(coordinate), float(value), float(value / maximum)]
            )

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
    plt.rcParams["legend.frameon"] = False
    mm_to_inch = 1.0 / 25.4
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(180.0 * mm_to_inch, 66.0 * mm_to_inch),
        gridspec_kw={"width_ratios": [1.4, 1.0]},
    )
    blue = "#0F4D92"
    blue_light = "#B4CFE8"
    red = "#B64342"
    neutral = "#767676"
    normalized = density / np.max(density)

    ax = axes[0]
    ax.fill_between(grid, normalized, color=blue_light, alpha=0.65)
    ax.plot(grid, normalized, color=blue, linewidth=1.3)
    for center in centers:
        ax.axvspan(
            center - radius,
            center + radius,
            color="#D8D8D8",
            alpha=0.75,
        )
        ax.axvline(center, color=neutral, linestyle="--", linewidth=0.8)
    if chosen is not None:
        left_index, right_index, valley_index = chosen
        ax.scatter(
            grid[[left_index, right_index]],
            normalized[[left_index, right_index]],
            color=red,
            s=18,
            zorder=3,
        )
        ax.scatter(
            [grid[valley_index]],
            [normalized[valley_index]],
            facecolor="white",
            edgecolor=red,
            s=18,
            zorder=3,
        )
    ratio = valley_metrics["valley_to_lower_peak"]
    ax.text(
        0.98,
        0.96,
        (
            f"Valley/lower peak = {ratio:.3f}\n"
            f"Measured separation = "
            f"{valley_metrics['measured_peak_separation_mm']:.2f} mm\n"
            f"n = {len(x_values):,}"
            if ratio is not None
            else f"Two peaks not detected\nn = {len(x_values):,}"
        ),
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=6.4,
    )
    ax.set_xlabel("Reconstructed x (mm)")
    ax.set_ylabel("Normalized response")
    ax.set_ylim(0.0, 1.08)

    ax = axes[1]
    errors = reconstructed_truth_x - truth_x
    if len(errors):
        bins = np.linspace(
            np.quantile(errors, 0.005),
            np.quantile(errors, 0.995),
            45,
        )
        ax.hist(
            errors,
            bins=bins,
            density=True,
            color=blue_light,
            edgecolor=blue,
            linewidth=0.45,
        )
        ax.axvline(0.0, color=neutral, linestyle="--", linewidth=0.8)
        ax.text(
            0.98,
            0.96,
            f"Truth-matched n = {len(errors):,}\n"
            f"x RMSE = {np.sqrt(np.mean(errors**2)):.2f} mm",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=6.4,
        )
    ax.set_xlabel("Event-wise x error (mm)")
    ax.set_ylabel("Density")
    for label, ax in zip("ab", axes):
        ax.text(
            -0.13,
            1.04,
            label,
            transform=ax.transAxes,
            fontweight="bold",
            fontsize=9,
        )
    fig.tight_layout(pad=1.2, w_pad=2.0)
    output_base = out_dir / "double_rod_response"
    figures = []
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

    n_primary = int(summary["event_counts"]["primaries"])
    report = {
        "schema_version": "1.0",
        "kind": "double_rod",
        "analysis_dir": str(analysis_dir.resolve()),
        "config": str(config_path.resolve()),
        "target_center_separation_mm": separation,
        "target_diameter_mm": 2.0 * radius,
        "n_primaries": n_primary,
        "n_selected_valid": int(len(x_values)),
        "efficiency_per_primary": float(len(x_values) / n_primary),
        "two_peak_status": status,
        "two_peak_metrics": valley_metrics,
        "bandwidth_sensitivity": bandwidth_sensitivity,
        "bandwidth_robust_resolved_at_0p735": (
            bandwidth_robust_resolved
        ),
        "truth_labelled_rods": labelled,
        "source_data": str(curve_csv.resolve()),
        "figures": figures,
        "criterion_note": (
            "The 0.735 valley-to-lower-peak threshold is an explicitly chosen "
            "operational resolvability criterion; the continuous ratio and "
            "measured peak separation remain the primary outputs. Bandwidth "
            "robustness requires the same decision at 0.75x, 1.0x and 1.25x "
            "the nominal KDE bandwidth."
        ),
    }
    with (out_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    return report


def mtf_from_edge_spread(
    coordinate_mm: np.ndarray,
    edge_spread: np.ndarray,
) -> dict[str, Any]:
    coordinate_mm = np.asarray(coordinate_mm, dtype=np.float64)
    edge_spread = np.asarray(edge_spread, dtype=np.float64)
    if len(coordinate_mm) < 16 or len(coordinate_mm) != len(edge_spread):
        raise ValueError("edge-spread arrays must have the same length >= 16")
    spacing = np.diff(coordinate_mm)
    if not np.allclose(spacing, spacing[0], rtol=1.0e-5, atol=1.0e-8):
        raise ValueError("edge-spread coordinate grid must be uniform")
    line_spread = np.gradient(edge_spread, float(spacing[0]))
    mtf = np.abs(np.fft.rfft(line_spread))
    if mtf[0] <= 0.0:
        raise ValueError("edge-spread derivative has zero DC response")
    mtf /= mtf[0]
    frequency = np.fft.rfftfreq(
        len(line_spread),
        d=float(spacing[0]),
    )

    def crossing(level: float) -> float | None:
        below = np.flatnonzero(mtf <= level)
        below = below[below > 0]
        if len(below) == 0:
            return None
        right = int(below[0])
        left = right - 1
        return _crossing(
            float(frequency[left]),
            float(mtf[left]),
            float(frequency[right]),
            float(mtf[right]),
            level,
        )

    return {
        "frequency_cycles_per_mm": frequency,
        "mtf": mtf,
        "line_spread": line_spread,
        "mtf50_cycles_per_mm": crossing(0.5),
        "mtf10_cycles_per_mm": crossing(0.1),
    }


def fit_erf_edge_response(
    coordinate_mm: np.ndarray,
    edge_counts: np.ndarray,
    reference_counts: np.ndarray,
    *,
    edge_primaries: int,
    reference_primaries: int,
    n_bootstrap: int = 500,
    random_seed: int = 24061873,
) -> dict[str, Any]:
    """Fit a count-aware error-function ESF and derive a Gaussian-model MTF.

    The edge and full-sheet reference counts share an unknown, position-
    dependent acceptance. Conditioning on their binwise total removes that
    nuisance acceptance: the edge count is binomial with a probability set by
    the relative edge response and the two primary counts. This is materially
    more stable than differentiating a noisy ratio when only about one
    thousand selected events are available.
    """

    coordinate_mm = np.asarray(coordinate_mm, dtype=np.float64)
    edge_counts = np.asarray(edge_counts, dtype=np.int64)
    reference_counts = np.asarray(reference_counts, dtype=np.int64)
    if (
        len(coordinate_mm) < 16
        or len(edge_counts) != len(coordinate_mm)
        or len(reference_counts) != len(coordinate_mm)
    ):
        raise ValueError("edge fit arrays must have the same length >= 16")
    if np.any(edge_counts < 0) or np.any(reference_counts < 0):
        raise ValueError("edge/reference counts must be non-negative")
    if edge_primaries <= 0 or reference_primaries <= 0:
        raise ValueError("edge/reference primary counts must be positive")
    if n_bootstrap < 0:
        raise ValueError("n_bootstrap must be non-negative")

    spacing = np.diff(coordinate_mm)
    if not np.all(spacing > 0.0):
        raise ValueError("edge coordinates must be strictly increasing")
    total = edge_counts + reference_counts
    informative = total > 0
    if int(informative.sum()) < 16:
        raise ValueError("too few populated bins for the edge-response fit")
    x = coordinate_mm[informative]
    observed_edge = edge_counts[informative]
    observed_total = total[informative]
    primary_ratio = float(edge_primaries / reference_primaries)
    span = float(x[-1] - x[0])
    minimum_sigma = max(0.2, 0.2 * float(np.median(spacing)))
    maximum_sigma = max(minimum_sigma * 2.0, 0.5 * span)
    center_bound = max(abs(float(x[0])), abs(float(x[-1])))

    # Parameters are background ratio, log step amplitude, edge center and
    # log Gaussian sigma. The transforms enforce a rising, positive ESF.
    bounds = [
        (0.0, 1.0),
        (float(np.log(0.05)), float(np.log(3.0))),
        (-center_bound, center_bound),
        (float(np.log(minimum_sigma)), float(np.log(maximum_sigma))),
    ]

    def unpack(parameters: np.ndarray) -> tuple[float, float, float, float]:
        background = float(parameters[0])
        amplitude = float(np.exp(parameters[1]))
        center = float(parameters[2])
        sigma = float(np.exp(parameters[3]))
        return background, background + amplitude, center, sigma

    def relative_response(
        parameters: np.ndarray,
        points: np.ndarray,
    ) -> np.ndarray:
        background, plateau, center, sigma = unpack(parameters)
        transition = 0.5 * (
            1.0
            + erf(
                (points - center)
                / (np.sqrt(2.0) * sigma)
            )
        )
        return background + (plateau - background) * transition

    def probability(parameters: np.ndarray) -> np.ndarray:
        response = relative_response(parameters, x)
        result = primary_ratio * response / (
            1.0 + primary_ratio * response
        )
        return np.clip(result, 1.0e-12, 1.0 - 1.0e-12)

    def fit_counts(
        fitted_edge: np.ndarray,
        starts: list[np.ndarray],
    ) -> Any:
        fitted_reference = observed_total - fitted_edge

        def objective(parameters: np.ndarray) -> float:
            fitted_probability = probability(parameters)
            return float(
                -np.sum(
                    fitted_edge * np.log(fitted_probability)
                    + fitted_reference * np.log1p(-fitted_probability)
                )
            )

        candidates = [
            minimize(
                objective,
                start,
                method="L-BFGS-B",
                bounds=bounds,
                options={"maxiter": 2000, "ftol": 1.0e-12},
            )
            for start in starts
        ]
        finite = [
            candidate
            for candidate in candidates
            if np.isfinite(candidate.fun)
        ]
        if not finite:
            raise RuntimeError("edge-response optimizer returned no finite fit")
        return min(finite, key=lambda candidate: float(candidate.fun))

    center_starts = (0.0, float(np.median(x)))
    sigma_starts = (
        max(minimum_sigma, min(maximum_sigma, 2.0)),
        max(minimum_sigma, min(maximum_sigma, 5.0)),
        max(minimum_sigma, min(maximum_sigma, 10.0)),
    )
    initial_starts = [
        np.asarray(
            [0.01, np.log(1.0), center, np.log(sigma)],
            dtype=np.float64,
        )
        for center in center_starts
        for sigma in sigma_starts
    ]
    fitted = fit_counts(observed_edge, initial_starts)
    if not fitted.success:
        raise RuntimeError(
            "edge-response fit did not converge: "
            f"{fitted.message}"
        )
    parameters = np.asarray(fitted.x, dtype=np.float64)
    background, plateau, center, sigma = unpack(parameters)
    fitted_probability = probability(parameters)
    observed_fraction = observed_edge / observed_total
    clipped_observed = np.clip(
        observed_fraction,
        1.0e-12,
        1.0 - 1.0e-12,
    )
    saturated_nll = float(
        -np.sum(
            observed_edge * np.log(clipped_observed)
            + (observed_total - observed_edge)
            * np.log1p(-clipped_observed)
        )
    )
    deviance = float(2.0 * (float(fitted.fun) - saturated_nll))
    degrees_of_freedom = max(1, int(informative.sum()) - 4)
    pearson = float(
        np.sum(
            (
                observed_edge
                - observed_total * fitted_probability
            )
            ** 2
            / (
                observed_total
                * fitted_probability
                * (1.0 - fitted_probability)
            )
        )
    )

    def mtf_crossing(level: float, fitted_sigma: float) -> float:
        return float(
            np.sqrt(-np.log(level))
            / (np.sqrt(2.0) * np.pi * fitted_sigma)
        )

    bootstrap_parameters: list[np.ndarray] = []
    if n_bootstrap:
        rng = np.random.default_rng(int(random_seed))
        bootstrap_starts = [parameters]
        for _ in range(int(n_bootstrap)):
            sampled_edge = rng.binomial(
                observed_total,
                fitted_probability,
            )
            try:
                bootstrap_fit = fit_counts(
                    sampled_edge,
                    bootstrap_starts,
                )
            except RuntimeError:
                continue
            if bootstrap_fit.success and np.all(
                np.isfinite(bootstrap_fit.x)
            ):
                bootstrap_parameters.append(
                    np.asarray(bootstrap_fit.x, dtype=np.float64)
                )
    if bootstrap_parameters:
        bootstrap_array = np.vstack(bootstrap_parameters)
        bootstrap_centers = bootstrap_array[:, 2]
        bootstrap_sigmas = np.exp(bootstrap_array[:, 3])
        bootstrap_mtf50 = np.asarray(
            [mtf_crossing(0.5, value) for value in bootstrap_sigmas]
        )
        bootstrap_mtf10 = np.asarray(
            [mtf_crossing(0.1, value) for value in bootstrap_sigmas]
        )

        def interval(values: np.ndarray) -> list[float]:
            return [
                float(value)
                for value in np.quantile(values, [0.025, 0.975])
            ]

        bootstrap_report = {
            "requested": int(n_bootstrap),
            "successful": int(len(bootstrap_parameters)),
            "random_seed": int(random_seed),
            "center_95_ci_mm": interval(bootstrap_centers),
            "sigma_95_ci_mm": interval(bootstrap_sigmas),
            "mtf50_95_ci_cycles_per_mm": interval(bootstrap_mtf50),
            "mtf10_95_ci_cycles_per_mm": interval(bootstrap_mtf10),
        }
    else:
        bootstrap_sigmas = np.asarray([], dtype=np.float64)
        bootstrap_report = {
            "requested": int(n_bootstrap),
            "successful": 0,
            "random_seed": int(random_seed),
            "center_95_ci_mm": None,
            "sigma_95_ci_mm": None,
            "mtf50_95_ci_cycles_per_mm": None,
            "mtf10_95_ci_cycles_per_mm": None,
        }

    fitted_relative = relative_response(parameters, coordinate_mm)
    fitted_esf = (fitted_relative - background) / (
        plateau - background
    )
    fitted_lsf = np.exp(
        -0.5 * ((coordinate_mm - center) / sigma) ** 2
    ) / (np.sqrt(2.0 * np.pi) * sigma)
    return {
        "status": "ok",
        "background_ratio": background,
        "plateau_ratio": plateau,
        "edge_center_mm": center,
        "gaussian_sigma_mm": sigma,
        "fitted_relative_response": fitted_relative,
        "fitted_esf": fitted_esf,
        "fitted_lsf": fitted_lsf,
        "mtf50_cycles_per_mm": mtf_crossing(0.5, sigma),
        "mtf10_cycles_per_mm": mtf_crossing(0.1, sigma),
        "fit_diagnostics": {
            "informative_bins": int(informative.sum()),
            "binomial_deviance": deviance,
            "degrees_of_freedom": degrees_of_freedom,
            "deviance_per_dof": float(deviance / degrees_of_freedom),
            "pearson_chi_square": pearson,
            "pearson_chi_square_per_dof": float(
                pearson / degrees_of_freedom
            ),
        },
        "bootstrap": bootstrap_report,
        "_bootstrap_sigmas_mm": bootstrap_sigmas,
    }


def evaluate_edge_mtf(
    *,
    edge_analysis_dir: str | Path,
    reference_analysis_dir: str | Path,
    config_path: str | Path,
    out_dir: str | Path,
    bin_width_mm: float = 1.0,
    minimum_reference_count: int = 5,
    n_bootstrap: int = 500,
) -> dict[str, Any]:
    edge_analysis_dir = Path(edge_analysis_dir)
    reference_analysis_dir = Path(reference_analysis_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    config = load_config(config_path)
    if config["phantom"].get("target_kind") != "edge":
        raise ValueError("edge MTF config must have phantom.target_kind=edge")
    if bin_width_mm <= 0.0:
        raise ValueError("edge MTF bin width must be positive")

    edge_reconstruction = _load_npz(
        edge_analysis_dir / "reconstruction.npz"
    )
    reference_reconstruction = _load_npz(
        reference_analysis_dir / "reconstruction.npz"
    )

    def selected_x(table: dict[str, np.ndarray]) -> np.ndarray:
        usable = table["selected"].astype(bool) & table["valid"].astype(bool)
        return np.asarray(table["x_mm"][usable], dtype=np.float64)

    edge_x = selected_x(edge_reconstruction)
    reference_x = selected_x(reference_reconstruction)
    if len(edge_x) < 100 or len(reference_x) < 100:
        raise ValueError(
            "edge/reference analyses need at least 100 selected valid events"
        )
    with (edge_analysis_dir / "summary.json").open(
        "r",
        encoding="utf-8",
    ) as handle:
        edge_summary = json.load(handle)
    with (reference_analysis_dir / "summary.json").open(
        "r",
        encoding="utf-8",
    ) as handle:
        reference_summary = json.load(handle)
    edge_primaries = int(edge_summary["event_counts"]["primaries"])
    reference_primaries = int(
        reference_summary["event_counts"]["primaries"]
    )

    max_abs = float(
        min(
            100.0,
            max(30.0, np.quantile(np.abs(reference_x), 0.98)),
        )
    )
    edges = np.arange(
        -max_abs,
        max_abs + bin_width_mm,
        bin_width_mm,
    )
    centers = 0.5 * (edges[:-1] + edges[1:])
    edge_counts, _ = np.histogram(edge_x, bins=edges)
    reference_counts, _ = np.histogram(reference_x, bins=edges)
    edge_rate = edge_counts / edge_primaries
    reference_rate = reference_counts / reference_primaries
    valid_reference = reference_counts >= int(minimum_reference_count)
    if int(valid_reference.sum()) < 16:
        raise ValueError("too few edge bins have adequate reference counts")
    ratio = np.full(len(centers), np.nan)
    ratio[valid_reference] = (
        edge_rate[valid_reference] / reference_rate[valid_reference]
    )
    valid_indices = np.flatnonzero(np.isfinite(ratio))
    ratio = np.interp(
        np.arange(len(ratio)),
        valid_indices,
        ratio[valid_indices],
    )

    far = 0.4 * max_abs
    background_region = (centers >= -0.8 * max_abs) & (centers <= -far)
    plateau_region = (centers >= far) & (centers <= 0.8 * max_abs)
    background = float(np.median(ratio[background_region]))
    plateau = float(np.median(ratio[plateau_region]))
    if plateau <= background:
        raise ValueError("edge response plateau does not exceed its background")
    normalized = (ratio - background) / (plateau - background)
    edge_fit = fit_erf_edge_response(
        centers,
        edge_counts,
        reference_counts,
        edge_primaries=edge_primaries,
        reference_primaries=reference_primaries,
        n_bootstrap=n_bootstrap,
        random_seed=(
            int(config["simulation"].get("random_seed", 24061873))
            + int(round(float(config["phantom"]["depth_mm"]) * 1000.0))
        ),
    )
    edge_fit.pop("fitted_relative_response")
    fitted_esf = np.asarray(
        edge_fit.pop("fitted_esf"),
        dtype=np.float64,
    )
    fitted_lsf = np.asarray(
        edge_fit.pop("fitted_lsf"),
        dtype=np.float64,
    )
    frequency = np.linspace(
        0.0,
        0.5 / float(bin_width_mm),
        257,
    )
    fitted_sigma = float(edge_fit["gaussian_sigma_mm"])
    mtf = np.exp(
        -2.0 * np.pi**2 * fitted_sigma**2 * frequency**2
    )
    bootstrap_sigmas = np.asarray(
        edge_fit.pop("_bootstrap_sigmas_mm"),
        dtype=np.float64,
    )
    if len(bootstrap_sigmas):
        bootstrap_mtf = np.exp(
            -2.0
            * np.pi**2
            * bootstrap_sigmas[:, np.newaxis] ** 2
            * frequency[np.newaxis, :] ** 2
        )
        mtf_ci_low, mtf_ci_high = np.quantile(
            bootstrap_mtf,
            [0.025, 0.975],
            axis=0,
        )
    else:
        mtf_ci_low = np.full_like(frequency, np.nan)
        mtf_ci_high = np.full_like(frequency, np.nan)

    csv_path = out_dir / "edge_mtf_source.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "x_mm",
                "edge_counts",
                "reference_counts",
                "raw_rate_ratio",
                "raw_normalized_esf",
                "fitted_erf_esf",
                "fitted_gaussian_lsf",
            ]
        )
        writer.writerows(
            zip(
                centers,
                edge_counts,
                reference_counts,
                ratio,
                normalized,
                fitted_esf,
                fitted_lsf,
            )
        )
    mtf_csv = out_dir / "mtf_curve.csv"
    with mtf_csv.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "spatial_frequency_cycles_per_mm",
                "model_mtf",
                "bootstrap_95_ci_low",
                "bootstrap_95_ci_high",
            ]
        )
        writer.writerows(
            zip(
                frequency,
                mtf,
                mtf_ci_low,
                mtf_ci_high,
            )
        )

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
    mm_to_inch = 1.0 / 25.4
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(180.0 * mm_to_inch, 64.0 * mm_to_inch),
    )
    blue = "#0F4D92"
    red = "#B64342"
    neutral = "#767676"

    axes[0].plot(
        centers,
        normalized,
        color="#B4CFE8",
        linewidth=0.7,
        marker="o",
        markersize=1.5,
        label="Count-ratio bins",
    )
    axes[0].plot(
        centers,
        fitted_esf,
        color=blue,
        linewidth=1.4,
        label="Count-aware erf fit",
    )
    axes[0].axvline(0.0, color=neutral, linestyle="--", linewidth=0.8)
    axes[0].set_xlabel("Reconstructed x (mm)")
    axes[0].set_ylabel("Edge-spread response")
    axes[0].legend(fontsize=5.7)

    axes[1].plot(
        centers,
        fitted_lsf,
        color=blue,
        linewidth=1.3,
    )
    axes[1].axvline(0.0, color=neutral, linestyle="--", linewidth=0.8)
    axes[1].set_xlabel("Reconstructed x (mm)")
    axes[1].set_ylabel("Gaussian line-spread model (1/mm)")

    if len(bootstrap_sigmas):
        axes[2].fill_between(
            frequency,
            mtf_ci_low,
            mtf_ci_high,
            color="#B4CFE8",
            alpha=0.55,
            linewidth=0.0,
            label="95% bootstrap interval",
        )
    axes[2].plot(frequency, mtf, color=blue, linewidth=1.3)
    axes[2].axhline(0.5, color=neutral, linestyle="--", linewidth=0.8)
    axes[2].axhline(0.1, color=neutral, linestyle=":", linewidth=0.8)
    for value, level in (
        (edge_fit["mtf50_cycles_per_mm"], 0.5),
        (edge_fit["mtf10_cycles_per_mm"], 0.1),
    ):
        if value is not None:
            axes[2].scatter([value], [level], color=red, s=16, zorder=3)
    axes[2].set_xlim(
        0.0,
        0.5 / bin_width_mm,
    )
    axes[2].set_ylim(0.0, 1.05)
    axes[2].set_xlabel("Spatial frequency (cycles/mm)")
    axes[2].set_ylabel("MTF")
    mtf50_text = (
        f"{edge_fit['mtf50_cycles_per_mm']:.4f}"
        if edge_fit["mtf50_cycles_per_mm"] is not None
        else "unavailable"
    )
    mtf10_text = (
        f"{edge_fit['mtf10_cycles_per_mm']:.4f}"
        if edge_fit["mtf10_cycles_per_mm"] is not None
        else "unavailable"
    )
    axes[2].text(
        0.98,
        0.96,
        f"MTF50 = {mtf50_text} cyc/mm\n"
        f"MTF10 = {mtf10_text} cyc/mm",
        transform=axes[2].transAxes,
        ha="right",
        va="top",
        fontsize=6.3,
    )
    if len(bootstrap_sigmas):
        axes[2].legend(loc="upper right", bbox_to_anchor=(1.0, 0.77), fontsize=5.5)
    for label, ax in zip("abc", axes):
        ax.text(
            -0.17,
            1.04,
            label,
            transform=ax.transAxes,
            fontweight="bold",
            fontsize=9,
        )
    fig.tight_layout(pad=1.2, w_pad=2.0)
    output_base = out_dir / "edge_mtf"
    figures = []
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

    report = {
        "schema_version": "1.0",
        "kind": "edge",
        "edge_analysis_dir": str(edge_analysis_dir.resolve()),
        "reference_analysis_dir": str(reference_analysis_dir.resolve()),
        "config": str(Path(config_path).resolve()),
        "target_depth_mm": float(config["phantom"]["depth_mm"]),
        "bin_width_mm": float(bin_width_mm),
        "minimum_reference_count": int(minimum_reference_count),
        "edge_selected_valid": int(len(edge_x)),
        "reference_selected_valid": int(len(reference_x)),
        "raw_background_ratio": background,
        "raw_plateau_ratio": plateau,
        "model": "binomial_count_likelihood_erf_esf_gaussian_lsf",
        "model_fit": edge_fit,
        "mtf50_cycles_per_mm": edge_fit["mtf50_cycles_per_mm"],
        "mtf10_cycles_per_mm": edge_fit["mtf10_cycles_per_mm"],
        "source_data": [str(csv_path.resolve()), str(mtf_csv.resolve())],
        "figures": figures,
        "method_note": (
            "The edge run is normalized by a same-depth full thin-sheet "
            "reference. A conditional binomial likelihood cancels the unknown "
            "binwise illumination/acceptance, then fits an error-function ESF. "
            "The reported MTF is the analytic transform of the corresponding "
            "Gaussian LSF; its 95% interval is a conditional parametric "
            "bootstrap interval."
        ),
        "interpretation_guardrail": (
            "These are model-based MTF estimates from one simulation seed. "
            "They replace the unstable direct derivative/FFT diagnostic for "
            "this low-count data set and must be reported with the Gaussian-"
            "LSF assumption and bootstrap interval."
        ),
    }
    with (out_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    return report


def evaluate_standard_target(
    *,
    analysis_dir: str | Path,
    config_path: str | Path,
    out_dir: str | Path | None = None,
    n_bootstrap: int = 500,
    bandwidth_scale: float = 1.0,
) -> dict[str, Any]:
    analysis_dir = Path(analysis_dir)
    out_dir = Path(out_dir) if out_dir else analysis_dir / "standard_metrics"
    out_dir.mkdir(parents=True, exist_ok=True)
    config = load_config(config_path)
    kind = str(config["phantom"].get("target_kind", ""))
    if kind == "rod":
        return _evaluate_rod_target(
            analysis_dir=analysis_dir,
            config_path=Path(config_path),
            config=config,
            out_dir=out_dir,
            n_bootstrap=n_bootstrap,
            bandwidth_scale=bandwidth_scale,
        )
    if kind == "double_rod":
        return _evaluate_double_rod_target(
            analysis_dir=analysis_dir,
            config_path=Path(config_path),
            config=config,
            out_dir=out_dir,
            bandwidth_scale=bandwidth_scale,
        )
    if kind != "thin_layer":
        raise ValueError(
            "current evaluator supports thin_layer, rod and double_rod; "
            "edge evaluation requires a same-depth full-sheet reference"
        )

    reconstruction = _load_npz(analysis_dir / "reconstruction.npz")
    with (analysis_dir / "summary.json").open("r", encoding="utf-8") as handle:
        summary = json.load(handle)
    usable = (
        reconstruction["selected"].astype(bool)
        & reconstruction["valid"].astype(bool)
    )
    depth_values = -np.asarray(
        reconstruction["z_mm"][usable],
        dtype=np.float64,
    )
    truth_matched = reconstruction["truth_matched"].astype(bool)
    truth_usable = usable & truth_matched
    depth_error = (
        -reconstruction["z_mm"][truth_usable]
        + reconstruction["truth_z_mm"][truth_usable]
    ).astype(np.float64)

    estimate = estimate_fwhm_kde(
        depth_values,
        bandwidth_scale=bandwidth_scale,
    )
    bootstrap = bootstrap_fwhm(
        depth_values,
        n_bootstrap=n_bootstrap,
        bandwidth_scale=bandwidth_scale,
    )
    target_z_lo, target_z_hi = map(
        float,
        config["phantom"]["z_interval_mm"],
    )
    target_lo, target_hi = -target_z_hi, -target_z_lo
    target_center = float(config["phantom"]["depth_mm"])
    n_primary = int(summary["event_counts"]["primaries"])
    efficiency = float(len(depth_values) / n_primary)

    curve_csv = out_dir / "axial_psf_curve.csv"
    if "grid" in estimate:
        _write_curve_csv(curve_csv, estimate["grid"], estimate["density"])

    serializable_estimate = {
        key: value
        for key, value in estimate.items()
        if key not in {"grid", "density"}
    }
    report: dict[str, Any] = {
        "schema_version": "1.0",
        "kind": kind,
        "analysis_dir": str(analysis_dir.resolve()),
        "config": str(Path(config_path).resolve()),
        "target_depth_mm": float(config["phantom"]["depth_mm"]),
        "target_depth_interval_mm": [target_lo, target_hi],
        "n_primaries": n_primary,
        "n_selected_valid": int(len(depth_values)),
        "efficiency_per_primary": efficiency,
        "fwhm_method": {
            "name": "Gaussian KDE main-peak half-maximum crossings",
            "bandwidth": f"Scott factor x {bandwidth_scale:g}",
            "finite_target_deconvolution": False,
            "estimate": serializable_estimate,
            "bootstrap": bootstrap,
        },
        "peak_depth_bias_mm": (
            float(estimate["mode"] - target_center)
            if estimate.get("mode") is not None
            else None
        ),
        "event_truth_error": {
            "n": int(len(depth_error)),
            "bias_mm": (
                float(np.mean(depth_error)) if len(depth_error) else None
            ),
            "rmse_mm": (
                float(np.sqrt(np.mean(depth_error**2)))
                if len(depth_error)
                else None
            ),
            "absolute_error_p90_mm": (
                float(np.quantile(np.abs(depth_error), 0.9))
                if len(depth_error)
                else None
            ),
        },
        "source_data": (
            str(curve_csv.resolve()) if curve_csv.exists() else None
        ),
        "figures": [],
        "interpretation_guardrail": (
            "This is the intrinsic geometric response to a finite 1 mm target "
            "placed in vacuum. It is not a concrete penetration-depth result. "
            "No finite-target deconvolution was applied."
        ),
    }
    if "grid" in estimate:
        report["figures"] = _plot_thin_layer(
            depth_values=depth_values,
            depth_error=depth_error,
            target_interval=(target_lo, target_hi),
            target_center=target_center,
            estimate=estimate,
            bootstrap=bootstrap,
            efficiency=efficiency,
            output_base=out_dir / "axial_psf",
        )

    with (out_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    return report


def evaluate_concrete_depth_profile(
    *,
    analysis_dir: str | Path,
    config_path: str | Path,
    out_dir: str | Path | None = None,
    bin_width_mm: float = 10.0,
    minimum_events: int = 100,
    n_bootstrap: int = 200,
    bandwidth_scale: float = 1.0,
) -> dict[str, Any]:
    """Measure depth-dependent sensitivity and axial error in a full slab.

    Unlike the isolated thin-sheet calibration, this evaluator conditions on
    the true first Compton vertex inside the homogeneous slab.  It therefore
    includes the attenuation and scatter background produced by the material
    above each depth bin.
    """
    if bin_width_mm <= 0.0:
        raise ValueError("bin_width_mm must be positive")
    if minimum_events < 20:
        raise ValueError("minimum_events must be at least 20")

    analysis_dir = Path(analysis_dir)
    out_dir = (
        Path(out_dir)
        if out_dir
        else analysis_dir / "concrete_depth_profile"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    config = load_config(config_path)
    reconstruction = _load_npz(analysis_dir / "reconstruction.npz")
    pairs = _load_npz(analysis_dir / "pairs.npz")
    with (analysis_dir / "summary.json").open("r", encoding="utf-8") as handle:
        summary = json.load(handle)

    from .pipeline import build_selection

    observable_config = copy.deepcopy(config)
    observable_config["selection"]["phantom_truth_policy"] = "ignore"
    observable_selected, observable_chain, _ = build_selection(
        pairs,
        observable_config,
    )
    valid_truth = (
        reconstruction["valid"].astype(bool)
        & reconstruction["truth_matched"].astype(bool)
    )
    observable_valid_all = (
        observable_selected & reconstruction["valid"].astype(bool)
    )
    observable_usable = observable_selected & valid_truth
    volumes = np.char.lower(pairs["d1_compt_volume"].astype(str))
    truth_volume = str(
        config["selection"].get("phantom_truth_volume_contains", "concrete")
    ).lower()
    truth_pure_condition = (
        (pairs["d1_n_phantom_compton"] == 1)
        & (pairs["d2_n_phantom_compton"] == 0)
        & (np.char.find(volumes, truth_volume) >= 0)
    )
    truth_pure_usable = observable_usable & truth_pure_condition

    reconstructed_depth = -np.asarray(
        reconstruction["z_mm"][observable_usable],
        dtype=np.float64,
    )
    true_depth = -np.asarray(
        reconstruction["truth_z_mm"][observable_usable],
        dtype=np.float64,
    )
    residual = reconstructed_depth - true_depth
    truth_pure_depth = -np.asarray(
        reconstruction["truth_z_mm"][truth_pure_usable],
        dtype=np.float64,
    )
    truth_pure_residual = (
        -np.asarray(
            reconstruction["z_mm"][truth_pure_usable],
            dtype=np.float64,
        )
        - truth_pure_depth
    )

    z_lo, z_hi = map(float, config["phantom"]["z_interval_mm"])
    depth_lo, depth_hi = sorted((-z_hi, -z_lo))
    edges = np.arange(depth_lo, depth_hi, bin_width_mm, dtype=np.float64)
    if len(edges) == 0 or not np.isclose(edges[0], depth_lo):
        edges = np.r_[depth_lo, edges]
    if not np.isclose(edges[-1], depth_hi):
        edges = np.r_[edges, depth_hi]
    n_primary = int(summary["event_counts"]["primaries"])

    rows: list[dict[str, Any]] = []
    for index, (left, right) in enumerate(zip(edges[:-1], edges[1:])):
        in_bin = (true_depth >= left) & (
            true_depth <= right if index == len(edges) - 2 else true_depth < right
        )
        errors = residual[in_bin]
        count = int(len(errors))
        pure_in_bin = (truth_pure_depth >= left) & (
            truth_pure_depth <= right
            if index == len(edges) - 2
            else truth_pure_depth < right
        )
        pure_errors = truth_pure_residual[pure_in_bin]
        pure_count = int(len(pure_errors))
        width = float(right - left)
        efficiency = count / n_primary if n_primary > 0 else np.nan
        sensitivity_density = efficiency / width
        pure_efficiency = (
            pure_count / n_primary if n_primary > 0 else np.nan
        )
        pure_sensitivity_density = pure_efficiency / width
        estimate = estimate_fwhm_kde(
            errors,
            bandwidth_scale=bandwidth_scale,
        )
        pure_estimate = estimate_fwhm_kde(
            pure_errors,
            bandwidth_scale=bandwidth_scale,
        )
        if count >= minimum_events and estimate["status"] == "ok":
            bootstrap = bootstrap_fwhm(
                errors,
                n_bootstrap=n_bootstrap,
                seed=24061873 + index,
                bandwidth_scale=bandwidth_scale,
            )
        else:
            bootstrap = {
                "requested": int(n_bootstrap),
                "valid": 0,
                "fwhm_95_ci": None,
                "mode_95_ci": None,
            }
        ci = bootstrap.get("fwhm_95_ci")
        rows.append(
            {
                "depth_bin_left_mm": float(left),
                "depth_bin_right_mm": float(right),
                "depth_bin_center_mm": float(0.5 * (left + right)),
                "bin_width_mm": width,
                "observable_selected_valid_truth_matched": count,
                "truth_pure_single_compton_events": pure_count,
                "truth_pure_fraction_of_truth_matched": (
                    float(pure_count / count) if count else None
                ),
                "observable_efficiency_per_primary_in_bin": float(efficiency),
                "observable_sensitivity_per_primary_per_mm": float(
                    sensitivity_density
                ),
                "truth_pure_efficiency_per_primary_in_bin": float(
                    pure_efficiency
                ),
                "truth_pure_sensitivity_per_primary_per_mm": float(
                    pure_sensitivity_density
                ),
                "poisson_1sigma_per_primary_per_mm": (
                    float(np.sqrt(count) / n_primary / width)
                    if n_primary > 0
                    else None
                ),
                "axial_error_bias_mm": (
                    float(np.mean(errors)) if count else None
                ),
                "axial_error_rmse_mm": (
                    float(np.sqrt(np.mean(errors**2))) if count else None
                ),
                "axial_error_abs_p90_mm": (
                    float(np.quantile(np.abs(errors), 0.9))
                    if count
                    else None
                ),
                "axial_error_fwhm_status": estimate["status"],
                "axial_error_fwhm_mm": estimate.get("fwhm"),
                "axial_error_fwhm_ci_low_mm": ci[0] if ci else None,
                "axial_error_fwhm_ci_high_mm": ci[1] if ci else None,
                "bootstrap_valid": int(bootstrap["valid"]),
                "truth_pure_axial_error_bias_mm": (
                    float(np.mean(pure_errors)) if pure_count else None
                ),
                "truth_pure_axial_error_rmse_mm": (
                    float(np.sqrt(np.mean(pure_errors**2)))
                    if pure_count
                    else None
                ),
                "truth_pure_axial_error_fwhm_status": pure_estimate["status"],
                "truth_pure_axial_error_fwhm_mm": pure_estimate.get("fwhm"),
            }
        )

    csv_path = out_dir / "concrete_depth_profile.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    evaluable = [
        row
        for row in rows
        if row["observable_selected_valid_truth_matched"] >= minimum_events
        and row["axial_error_fwhm_status"] == "ok"
    ]
    deepest = (
        float(max(row["depth_bin_right_mm"] for row in evaluable))
        if evaluable
        else None
    )

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
    plt.rcParams["legend.frameon"] = False

    depth = np.asarray([row["depth_bin_center_mm"] for row in rows])
    counts = np.asarray(
        [row["observable_selected_valid_truth_matched"] for row in rows],
        dtype=np.int64,
    )
    sensitivity = (
        np.asarray(
            [
                row["observable_sensitivity_per_primary_per_mm"]
                for row in rows
            ],
            dtype=np.float64,
        )
        * 1.0e6
    )
    pure_sensitivity = (
        np.asarray(
            [
                row["truth_pure_sensitivity_per_primary_per_mm"]
                for row in rows
            ],
            dtype=np.float64,
        )
        * 1.0e6
    )
    sensitivity_sigma = (
        np.asarray(
            [
                row["poisson_1sigma_per_primary_per_mm"]
                for row in rows
            ],
            dtype=np.float64,
        )
        * 1.0e6
    )
    bias = np.asarray(
        [
            np.nan
            if row["axial_error_bias_mm"] is None
            else row["axial_error_bias_mm"]
            for row in rows
        ],
        dtype=np.float64,
    )
    rmse = np.asarray(
        [
            np.nan
            if row["axial_error_rmse_mm"] is None
            else row["axial_error_rmse_mm"]
            for row in rows
        ],
        dtype=np.float64,
    )
    fwhm = np.asarray(
        [
            np.nan
            if row["axial_error_fwhm_mm"] is None
            else row["axial_error_fwhm_mm"]
            for row in rows
        ],
        dtype=np.float64,
    )
    pure_fwhm = np.asarray(
        [
            np.nan
            if row["truth_pure_axial_error_fwhm_mm"] is None
            else row["truth_pure_axial_error_fwhm_mm"]
            for row in rows
        ],
        dtype=np.float64,
    )
    ci_low = np.asarray(
        [
            np.nan
            if row["axial_error_fwhm_ci_low_mm"] is None
            else row["axial_error_fwhm_ci_low_mm"]
            for row in rows
        ],
        dtype=np.float64,
    )
    ci_high = np.asarray(
        [
            np.nan
            if row["axial_error_fwhm_ci_high_mm"] is None
            else row["axial_error_fwhm_ci_high_mm"]
            for row in rows
        ],
        dtype=np.float64,
    )

    example_indices = [
        index
        for index, row in enumerate(rows)
        if row["observable_selected_valid_truth_matched"] >= minimum_events
        and row["axial_error_fwhm_status"] == "ok"
    ]
    if len(example_indices) > 1:
        example_indices = [example_indices[0], example_indices[-1]]
    example_curves: list[dict[str, Any]] = []
    for index in example_indices:
        left = float(rows[index]["depth_bin_left_mm"])
        right = float(rows[index]["depth_bin_right_mm"])
        in_bin = (true_depth >= left) & (
            true_depth <= right if index == len(rows) - 1 else true_depth < right
        )
        errors = residual[in_bin]
        estimate = estimate_fwhm_kde(
            errors,
            bandwidth_scale=bandwidth_scale,
        )
        if estimate["status"] != "ok":
            continue
        density = np.asarray(estimate["density"], dtype=np.float64)
        example_curves.append(
            {
                "left_mm": left,
                "right_mm": right,
                "n": int(len(errors)),
                "grid": np.asarray(estimate["grid"], dtype=np.float64),
                "normalized_density": density / float(np.max(density)),
                "fwhm_mm": float(estimate["fwhm"]),
                "left_half_max_mm": float(estimate["left_half_max"]),
                "right_half_max_mm": float(estimate["right_half_max"]),
            }
        )

    example_csv_path = out_dir / "concrete_depth_profile_example_distributions.csv"
    with example_csv_path.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "true_depth_bin_left_mm",
                "true_depth_bin_right_mm",
                "n_events",
                "axial_residual_mm",
                "normalized_kde_density",
                "fwhm_mm",
            ]
        )
        for curve in example_curves:
            for grid_value, density_value in zip(
                curve["grid"], curve["normalized_density"]
            ):
                writer.writerow(
                    [
                        curve["left_mm"],
                        curve["right_mm"],
                        curve["n"],
                        float(grid_value),
                        float(density_value),
                        curve["fwhm_mm"],
                    ]
                )

    mm_to_inch = 1.0 / 25.4
    fig, axes = plt.subplots(
        1,
        4,
        # Match the presentation's wide image frame without making the four
        # panels too short to read.
        figsize=(220.0 * mm_to_inch, 74.0 * mm_to_inch),
    )
    blue = "#0F4D92"
    red = "#B64342"
    grey = "#767676"

    ax = axes[0]
    positive = sensitivity > 0.0
    observable_handle = ax.errorbar(
        depth[positive],
        sensitivity[positive],
        yerr=sensitivity_sigma[positive],
        color=blue,
        marker="o",
        markersize=3.2,
        linewidth=1.1,
        capsize=2,
        label="Observable cuts",
    )
    pure_positive = pure_sensitivity > 0.0
    pure_handle = ax.plot(
        depth[pure_positive],
        pure_sensitivity[pure_positive],
        color="#D57A1F",
        marker="s",
        markersize=2.8,
        linewidth=1.0,
        label="Truth-pure subset",
    )[0]
    if np.count_nonzero(positive) > 1:
        ax.set_yscale("log")
    ax.set_xlabel("True Compton depth (mm)")
    ax.set_ylabel("Truth-matched events / $10^6$ primaries / mm")
    ax.legend(
        [observable_handle, pure_handle],
        ["Observable cuts", "Truth-pure subset"],
        fontsize=5.8,
        loc="best",
    )
    ax.text(-0.18, 1.03, "a", transform=ax.transAxes, fontweight="bold", fontsize=9)

    ax = axes[1]
    example_colors = [blue, red]
    for curve, color in zip(example_curves, example_colors):
        label = (
            f"{curve['left_mm']:g}–{curve['right_mm']:g} mm; "
            f"FWHM={curve['fwhm_mm']:.1f} mm"
        )
        ax.plot(
            curve["grid"],
            curve["normalized_density"],
            color=color,
            linewidth=1.1,
            label=label,
        )
        ax.hlines(
            0.5,
            curve["left_half_max_mm"],
            curve["right_half_max_mm"],
            color=color,
            linewidth=1.0,
        )
    ax.axvline(0.0, color=grey, linestyle="--", linewidth=0.8)
    ax.set_xlabel(r"Axial residual $d_{rec}-d_{true}$ (mm)")
    ax.set_ylabel("Normalized KDE")
    ax.set_ylim(bottom=0.0)
    if example_curves:
        ax.legend(fontsize=5.1, loc="best")
    ax.text(-0.18, 1.03, "b", transform=ax.transAxes, fontweight="bold", fontsize=9)

    ax = axes[2]
    ax.axhline(0.0, color=grey, linestyle="--", linewidth=0.8)
    ax.plot(depth, bias, color=blue, marker="o", markersize=3.2, label="Bias")
    ax.plot(depth, rmse, color=red, marker="s", markersize=3.0, label="RMSE")
    ax.set_xlabel("True Compton depth (mm)")
    ax.set_ylabel("Axial error (mm)")
    ax.legend(loc="best")
    ax.text(-0.18, 1.03, "c", transform=ax.transAxes, fontweight="bold", fontsize=9)

    ax = axes[3]
    finite = np.isfinite(fwhm)
    if np.any(finite):
        lower = np.where(
            np.isfinite(ci_low[finite]),
            np.maximum(0.0, fwhm[finite] - ci_low[finite]),
            0.0,
        )
        upper = np.where(
            np.isfinite(ci_high[finite]),
            np.maximum(0.0, ci_high[finite] - fwhm[finite]),
            0.0,
        )
        observable_fwhm_handle = ax.errorbar(
            depth[finite],
            fwhm[finite],
            yerr=np.vstack([lower, upper]),
            color=blue,
            marker="o",
            markersize=3.2,
            linewidth=1.1,
            capsize=2,
            label="Observable cuts",
        )
    finite_pure = np.isfinite(pure_fwhm)
    if np.any(finite_pure):
        pure_fwhm_handle = ax.plot(
            depth[finite_pure],
            pure_fwhm[finite_pure],
            color="#D57A1F",
            marker="s",
            markersize=2.8,
            linewidth=1.0,
            label="Truth-pure subset",
        )[0]
    else:
        pure_fwhm_handle = None
    insufficient = counts < minimum_events
    if np.any(insufficient):
        y_marker = np.nanmax(fwhm) if np.any(finite) else 1.0
        insufficient_handle = ax.scatter(
            depth[insufficient],
            np.full(np.count_nonzero(insufficient), 0.06 * y_marker),
            marker="x",
            color=grey,
            s=16,
            label=f"n < {minimum_events}",
        )
    else:
        insufficient_handle = None
    handles = []
    labels = []
    if np.any(finite):
        handles.append(observable_fwhm_handle)
        labels.append("Observable cuts")
    if pure_fwhm_handle is not None:
        handles.append(pure_fwhm_handle)
        labels.append("Truth-pure subset")
    if insufficient_handle is not None:
        handles.append(insufficient_handle)
        labels.append(f"n < {minimum_events}")
    if handles:
        ax.legend(handles, labels, fontsize=5.4, loc="upper right")
    ax.set_xlabel("True Compton depth (mm)")
    ax.set_ylabel("Axial-error FWHM (mm)")
    ax.set_ylim(bottom=0.0)
    ax.text(-0.18, 1.03, "d", transform=ax.transAxes, fontweight="bold", fontsize=9)

    fig.suptitle(
        f"Homogeneous concrete slab; n = {n_primary:,} primaries; "
        f"{bin_width_mm:g} mm truth bins",
        fontsize=7.2,
        y=1.01,
    )
    fig.tight_layout(w_pad=1.6)
    output_base = out_dir / "concrete_depth_profile"
    figures: list[str] = []
    for suffix, kwargs in (
        (".svg", {}),
        (".pdf", {}),
        (".tiff", {"dpi": 600}),
        (".png", {"dpi": 220}),
    ):
        path = output_base.with_suffix(suffix)
        fig.savefig(path, bbox_inches="tight", facecolor="white", **kwargs)
        figures.append(str(path.resolve()))
    plt.close(fig)

    report: dict[str, Any] = {
        "schema_version": "1.0",
        "kind": "homogeneous_concrete_depth_profile",
        "analysis_dir": str(analysis_dir.resolve()),
        "config": str(Path(config_path).resolve()),
        "n_primaries": n_primary,
        "n_observable_selected_valid_all": int(observable_valid_all.sum()),
        "n_observable_selected_valid_truth_matched": int(len(true_depth)),
        "observable_truth_match_fraction": (
            float(len(true_depth) / observable_valid_all.sum())
            if observable_valid_all.sum()
            else None
        ),
        "n_truth_pure_single_compton": int(len(truth_pure_depth)),
        "observable_selection_chain": observable_chain,
        "truth_depth_range_mm": [float(depth_lo), float(depth_hi)],
        "bin_width_mm": float(bin_width_mm),
        "minimum_events_for_resolution": int(minimum_events),
        "bootstrap_requested_per_evaluable_bin": int(n_bootstrap),
        "deepest_evaluable_bin_right_edge_mm": deepest,
        "rows": rows,
        "source_data": str(csv_path.resolve()),
        "example_error_distribution_source_data": str(
            example_csv_path.resolve()
        ),
        "figures": figures,
        "interpretation_guardrail": (
            "The deepest evaluable bin is a run-specific statistical limit "
            "defined by the stated event-count and single-peak FWHM criteria; "
            "it is not a universal maximum penetration depth."
        ),
        "method_note": (
            "Energy, timing, angle and detector-position cuts use observable "
            "fields only. Events are binned by the matched true first-Compton "
            "depth in the full concrete slab, so sensitivity includes "
            "overburden attenuation. Phantom truth is reported as a scoring "
            "subset, not used as an experimental cut. Axial resolution is the "
            "KDE FWHM of reconstructed depth minus true depth within each "
            "truth-depth bin. Panel b shows representative shallow and deep "
            "residual distributions before their FWHM values are summarized "
            "versus depth in panel d."
        ),
    }
    with (out_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    return report


def summarize_depth_series(
    *,
    metrics_paths: list[str | Path],
    out_dir: str | Path,
) -> dict[str, Any]:
    if not metrics_paths:
        raise ValueError("at least one metrics.json path is required")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for path_like in metrics_paths:
        path = Path(path_like)
        with path.open("r", encoding="utf-8") as handle:
            report = json.load(handle)
        estimate = report["fwhm_method"]["estimate"]
        bootstrap = report["fwhm_method"]["bootstrap"]
        ci = bootstrap.get("fwhm_95_ci")
        rows.append(
            {
                "depth_mm": float(report["target_depth_mm"]),
                "n_primaries": int(report["n_primaries"]),
                "n_selected_valid": int(report["n_selected_valid"]),
                "efficiency_per_primary": float(
                    report["efficiency_per_primary"]
                ),
                "fwhm_status": estimate["status"],
                "axial_fwhm_mm": estimate.get("fwhm"),
                "axial_fwhm_ci_low_mm": ci[0] if ci else None,
                "axial_fwhm_ci_high_mm": ci[1] if ci else None,
                "peak_depth_bias_mm": report.get("peak_depth_bias_mm"),
                "event_depth_bias_mm": report["event_truth_error"]["bias_mm"],
                "event_depth_rmse_mm": report["event_truth_error"]["rmse_mm"],
                "event_abs_depth_error_p90_mm": report["event_truth_error"][
                    "absolute_error_p90_mm"
                ],
                "metrics_path": str(path.resolve()),
            }
        )
    rows.sort(key=lambda row: row["depth_mm"])

    csv_path = out_dir / "depth_performance.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

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
    plt.rcParams["legend.frameon"] = False

    depth = np.asarray([row["depth_mm"] for row in rows])
    fwhm = np.asarray(
        [
            np.nan if row["axial_fwhm_mm"] is None else row["axial_fwhm_mm"]
            for row in rows
        ],
        dtype=np.float64,
    )
    low = np.asarray(
        [
            np.nan
            if row["axial_fwhm_ci_low_mm"] is None
            else row["axial_fwhm_ci_low_mm"]
            for row in rows
        ]
    )
    high = np.asarray(
        [
            np.nan
            if row["axial_fwhm_ci_high_mm"] is None
            else row["axial_fwhm_ci_high_mm"]
            for row in rows
        ]
    )
    peak_bias = np.asarray(
        [
            np.nan
            if row["peak_depth_bias_mm"] is None
            else row["peak_depth_bias_mm"]
            for row in rows
        ]
    )
    event_rmse = np.asarray(
        [
            np.nan
            if row["event_depth_rmse_mm"] is None
            else row["event_depth_rmse_mm"]
            for row in rows
        ]
    )
    efficiency = np.asarray(
        [row["efficiency_per_primary"] for row in rows]
    )
    counts = np.asarray([row["n_selected_valid"] for row in rows])

    mm_to_inch = 1.0 / 25.4
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(180.0 * mm_to_inch, 65.0 * mm_to_inch),
    )
    blue = "#0F4D92"
    red = "#B64342"
    neutral = "#767676"

    ax = axes[0]
    valid_ci = np.isfinite(fwhm) & np.isfinite(low) & np.isfinite(high)
    if np.any(valid_ci):
        yerr = np.vstack(
            [
                np.maximum(0.0, fwhm[valid_ci] - low[valid_ci]),
                np.maximum(0.0, high[valid_ci] - fwhm[valid_ci]),
            ]
        )
        ax.errorbar(
            depth[valid_ci],
            fwhm[valid_ci],
            yerr=yerr,
            color=blue,
            marker="o",
            linewidth=1.3,
            markersize=3.5,
            capsize=2.2,
        )
    ax.set_xlabel("Target axial distance (mm)")
    ax.set_ylabel("Axial FWHM (mm)")

    ax = axes[1]
    ax.axhline(0.0, color=neutral, linestyle="--", linewidth=0.8)
    ax.plot(
        depth,
        peak_bias,
        color=blue,
        marker="o",
        linewidth=1.2,
        markersize=3.5,
        label="Peak bias",
    )
    ax.plot(
        depth,
        event_rmse,
        color=red,
        marker="s",
        linewidth=1.2,
        markersize=3.2,
        label="Event RMSE",
    )
    ax.set_xlabel("Target axial distance (mm)")
    ax.set_ylabel("Depth error (mm)")
    ax.legend(fontsize=6)

    ax = axes[2]
    ax.plot(
        depth,
        efficiency,
        color=blue,
        marker="o",
        linewidth=1.3,
        markersize=3.5,
    )
    ax.set_yscale("log")
    ax.set_xlabel("Target axial distance (mm)")
    ax.set_ylabel("Selected events per primary")
    for x, y, n in zip(depth, efficiency, counts):
        ax.annotate(
            f"n={n:,}",
            (x, y),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            fontsize=5.5,
        )

    for label, ax in zip("abc", axes):
        ax.text(
            -0.18,
            1.04,
            label,
            transform=ax.transAxes,
            fontweight="bold",
            fontsize=9,
        )
        ax.set_xticks(depth)

    fig.tight_layout(pad=1.2, w_pad=2.0)
    output_base = out_dir / "depth_performance"
    figures = []
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

    output = {
        "schema_version": "1.0",
        "rows": rows,
        "source_data": str(csv_path.resolve()),
        "figures": figures,
        "interval_note": (
            "FWHM intervals are within-run nonparametric bootstrap intervals. "
            "An independent random-seed repeat is still required for simulation "
            "repeatability."
        ),
        "interpretation_guardrail": (
            "These isolated thin-sheet runs measure intrinsic geometric response "
            "versus axial target distance in vacuum. Use the homogeneous concrete "
            "depth profile for material penetration and depth sensitivity."
        ),
    }
    with (out_dir / "depth_performance.json").open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(output, handle, ensure_ascii=False, indent=2)
    return output
