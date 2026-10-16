from __future__ import annotations

import copy
import csv
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .config import load_config
from .pipeline import build_selection
from .standards import estimate_fwhm_kde


def _load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _ci_standard_error(ci: list[float] | None) -> float | None:
    if ci is None or len(ci) != 2:
        return None
    lo, hi = map(float, ci)
    if not lo < hi:
        return None
    return float((hi - lo) / (2.0 * 1.96))


def _difference_z(
    value_a: float,
    se_a: float | None,
    value_b: float,
    se_b: float | None,
) -> float | None:
    if se_a is None or se_b is None:
        return None
    denominator = float(np.hypot(se_a, se_b))
    if denominator <= 0.0:
        return None
    return float((value_b - value_a) / denominator)


def _configure_matplotlib() -> Any:
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
    return plt


def _save_figure(fig: Any, output_base: Path) -> list[str]:
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
    return figures


def compare_standard_repeatability(
    *,
    metrics_paths: Iterable[str | Path],
    out_dir: str | Path,
) -> dict[str, Any]:
    paths = [Path(path) for path in metrics_paths]
    if len(paths) != 2:
        raise ValueError("repeatability comparison currently requires two runs")
    metrics = [_load_json(path) for path in paths]
    if any(item.get("kind") != metrics[0].get("kind") for item in metrics):
        raise ValueError("standard-target metric kinds do not match")
    if any(
        float(item.get("target_depth_mm", np.nan))
        != float(metrics[0].get("target_depth_mm", np.nan))
        for item in metrics
    ):
        raise ValueError("standard-target depths do not match")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    curves: list[tuple[np.ndarray, np.ndarray]] = []
    for path, item in zip(paths, metrics):
        config = load_config(item["config"])
        estimate = item["fwhm_method"]["estimate"]
        ci = item["fwhm_method"]["bootstrap"].get("fwhm_95_ci")
        row = {
            "metrics_path": str(path.resolve()),
            "analysis_dir": item["analysis_dir"],
            "random_seed": int(config["simulation"]["random_seed"]),
            "n_primaries": int(item["n_primaries"]),
            "n_selected_valid": int(item["n_selected_valid"]),
            "efficiency_per_primary": float(item["efficiency_per_primary"]),
            "efficiency_poisson_se": float(
                np.sqrt(item["n_selected_valid"]) / item["n_primaries"]
            ),
            "fwhm_mm": (
                float(estimate["fwhm"])
                if estimate.get("fwhm") is not None
                else None
            ),
            "fwhm_ci_low_mm": float(ci[0]) if ci is not None else None,
            "fwhm_ci_high_mm": float(ci[1]) if ci is not None else None,
            "fwhm_approx_se_from_bootstrap_ci": _ci_standard_error(ci),
            "peak_depth_bias_mm": float(item["peak_depth_bias_mm"]),
            "event_truth_rmse_mm": float(
                item["event_truth_error"]["rmse_mm"]
            ),
        }
        rows.append(row)
        with Path(item["source_data"]).open(
            "r",
            encoding="utf-8-sig",
            newline="",
        ) as handle:
            curve_rows = list(csv.DictReader(handle))
        curves.append(
            (
                np.asarray(
                    [float(curve["depth_mm"]) for curve in curve_rows]
                ),
                np.asarray(
                    [float(curve["density_normalized"]) for curve in curve_rows]
                ),
            )
        )

    first, second = rows
    fwhm_z = (
        _difference_z(
            float(first["fwhm_mm"]),
            first["fwhm_approx_se_from_bootstrap_ci"],
            float(second["fwhm_mm"]),
            second["fwhm_approx_se_from_bootstrap_ci"],
        )
        if first["fwhm_mm"] is not None and second["fwhm_mm"] is not None
        else None
    )
    efficiency_z = _difference_z(
        first["efficiency_per_primary"],
        first["efficiency_poisson_se"],
        second["efficiency_per_primary"],
        second["efficiency_poisson_se"],
    )
    fwhm_ci_overlap = (
        max(first["fwhm_ci_low_mm"], second["fwhm_ci_low_mm"])
        <= min(first["fwhm_ci_high_mm"], second["fwhm_ci_high_mm"])
        if all(
            value is not None
            for value in (
                first["fwhm_ci_low_mm"],
                first["fwhm_ci_high_mm"],
                second["fwhm_ci_low_mm"],
                second["fwhm_ci_high_mm"],
            )
        )
        else None
    )
    pairwise = {
        "fwhm_difference_mm": (
            float(second["fwhm_mm"] - first["fwhm_mm"])
            if first["fwhm_mm"] is not None
            and second["fwhm_mm"] is not None
            else None
        ),
        "fwhm_relative_difference_vs_pair_mean": (
            float(
                (second["fwhm_mm"] - first["fwhm_mm"])
                / (0.5 * (first["fwhm_mm"] + second["fwhm_mm"]))
            )
            if first["fwhm_mm"] is not None
            and second["fwhm_mm"] is not None
            else None
        ),
        "fwhm_difference_z_approx": fwhm_z,
        "fwhm_bootstrap_95_ci_overlap": fwhm_ci_overlap,
        "efficiency_relative_difference_vs_pair_mean": float(
            (
                second["efficiency_per_primary"]
                - first["efficiency_per_primary"]
            )
            / (
                0.5
                * (
                    first["efficiency_per_primary"]
                    + second["efficiency_per_primary"]
                )
            )
        ),
        "efficiency_difference_z_poisson": efficiency_z,
        "provisional_consistency_within_2sigma": bool(
            (fwhm_z is None or abs(fwhm_z) <= 2.0)
            and (efficiency_z is None or abs(efficiency_z) <= 2.0)
        ),
        "pre_registered_repeatability_pass": bool(
            fwhm_ci_overlap is True
            and fwhm_z is not None
            and abs(fwhm_z) <= 2.0
            and efficiency_z is not None
            and abs(efficiency_z) <= 2.0
        ),
    }

    csv_path = out_dir / "standard_repeatability.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    plt = _configure_matplotlib()
    mm_to_inch = 1.0 / 25.4
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(180.0 * mm_to_inch, 62.0 * mm_to_inch),
    )
    colors = ["#0F4D92", "#B64342"]
    for row, curve, color in zip(rows, curves, colors):
        axes[0].plot(
            curve[0],
            curve[1],
            color=color,
            linewidth=1.2,
            label=f"Seed {row['random_seed']}",
        )
    axes[0].set_xlabel("Reconstructed depth (mm)")
    axes[0].set_ylabel("Normalized response")
    axes[0].legend(fontsize=6.2)

    labels = [str(row["random_seed"]) for row in rows]
    fwhm_values = [row["fwhm_mm"] for row in rows]
    fwhm_errors = [
        [
            row["fwhm_mm"] - row["fwhm_ci_low_mm"],
            row["fwhm_ci_high_mm"] - row["fwhm_mm"],
        ]
        for row in rows
    ]
    axes[1].errorbar(
        labels,
        fwhm_values,
        yerr=np.asarray(fwhm_errors).T,
        fmt="o",
        color="#0F4D92",
        capsize=3,
        markersize=4,
    )
    axes[1].set_xlabel("Random seed")
    axes[1].set_ylabel("Axial FWHM (mm)")
    axes[1].text(
        0.50,
        0.96,
        f"Approx. Δ / σ = {fwhm_z:.2f}"
        if fwhm_z is not None
        else "Approx. Δ / σ unavailable",
        transform=axes[1].transAxes,
        ha="center",
        va="top",
        fontsize=6.3,
    )

    efficiency_values = [row["efficiency_per_primary"] for row in rows]
    efficiency_errors = [row["efficiency_poisson_se"] for row in rows]
    axes[2].errorbar(
        labels,
        efficiency_values,
        yerr=efficiency_errors,
        fmt="o",
        color="#B64342",
        capsize=3,
        markersize=4,
    )
    axes[2].set_xlabel("Random seed")
    axes[2].set_ylabel("Selected valid events / primary")
    axes[2].text(
        0.50,
        0.96,
        f"Poisson Δ / σ = {efficiency_z:.2f}"
        if efficiency_z is not None
        else "Poisson Δ / σ unavailable",
        transform=axes[2].transAxes,
        ha="center",
        va="top",
        fontsize=6.3,
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
    fig.tight_layout(pad=1.1, w_pad=1.8)
    figures = _save_figure(fig, out_dir / "standard_repeatability")
    plt.close(fig)

    report = {
        "schema_version": "1.0",
        "kind": metrics[0]["kind"],
        "target_depth_mm": float(metrics[0]["target_depth_mm"]),
        "rows": rows,
        "pairwise": pairwise,
        "source_csv": str(csv_path.resolve()),
        "figures": figures,
        "guardrail": (
            "Two seeds test gross Monte-Carlo instability but are not enough "
            "to estimate a robust between-run variance. At least five seeds "
            "are preferred for a final uncertainty claim."
        ),
    }
    with (out_dir / "standard_repeatability.json").open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    return report


def validate_physical_d1_sizes(
    *,
    full_metrics_path: str | Path,
    physical_metrics_paths: Iterable[str | Path],
    posthoc_size_report_path: str | Path,
    out_dir: str | Path,
) -> dict[str, Any]:
    full_metrics = _load_json(full_metrics_path)
    physical_metrics = [
        _load_json(path) for path in physical_metrics_paths
    ]
    if not physical_metrics:
        raise ValueError("at least one physical D1-size run is required")
    posthoc = _load_json(posthoc_size_report_path)
    posthoc_by_radius = {
        float(row["outer_radius_mm"]): row
        for row in posthoc["posthoc_rows"]
    }
    full_config = load_config(full_metrics["config"])
    full_radius = float(full_config["detectors"]["d1"]["radius_mm"][1])
    full_efficiency = float(full_metrics["efficiency_per_primary"])
    full_fwhm_value = (
        full_metrics["fwhm_method"]["estimate"].get("fwhm")
    )
    full_fwhm = (
        float(full_fwhm_value)
        if full_fwhm_value is not None
        else None
    )
    rows: list[dict[str, Any]] = []

    all_metrics = [full_metrics, *physical_metrics]
    for item in all_metrics:
        config = load_config(item["config"])
        radius = float(config["detectors"]["d1"]["radius_mm"][1])
        if radius not in posthoc_by_radius:
            raise ValueError(
                f"no post-hoc prediction exists for D1 radius {radius:g} mm"
            )
        predicted = posthoc_by_radius[radius]
        actual_efficiency = float(item["efficiency_per_primary"])
        actual_fwhm = item["fwhm_method"]["estimate"].get("fwhm")
        actual_fwhm_ci = (
            item.get("fwhm_method", {})
            .get("bootstrap", {})
            .get("fwhm_95_ci")
        )
        predicted_fwhm = predicted.get("quality_value_mm")
        efficiency_prediction_error = float(
            actual_efficiency / full_efficiency
            - predicted["efficiency_retention"]
        )
        fwhm_relative_prediction_error = (
            float((actual_fwhm - predicted_fwhm) / predicted_fwhm)
            if actual_fwhm is not None
            and predicted_fwhm is not None
            and float(predicted_fwhm) != 0.0
            else None
        )
        direction_matches = (
            bool(
                (float(actual_fwhm) - full_fwhm)
                * (float(predicted_fwhm) - full_fwhm)
                >= 0.0
            )
            if radius != full_radius
            and actual_fwhm is not None
            and predicted_fwhm is not None
            and full_fwhm is not None
            else None
        )
        numeric_thresholds_pass = (
            bool(
                abs(efficiency_prediction_error) <= 0.02
                and fwhm_relative_prediction_error is not None
                and abs(fwhm_relative_prediction_error) <= 0.10
            )
            if radius != full_radius
            else None
        )
        row = {
            "outer_radius_mm": radius,
            "physical_run": radius != full_radius,
            "d1_mass_kg": float(predicted["d1_mass_kg"]),
            "actual_efficiency_per_primary": actual_efficiency,
            "actual_efficiency_retention_vs_full": float(
                actual_efficiency / full_efficiency
            ),
            "posthoc_efficiency_retention": float(
                predicted["efficiency_retention"]
            ),
            "efficiency_retention_prediction_error": (
                efficiency_prediction_error
            ),
            "efficiency_prediction_within_2_percentage_points": (
                bool(abs(efficiency_prediction_error) <= 0.02)
                if radius != full_radius
                else None
            ),
            "actual_fwhm_mm": (
                float(actual_fwhm) if actual_fwhm is not None else None
            ),
            "actual_fwhm_ci95_low_mm": (
                float(actual_fwhm_ci[0])
                if actual_fwhm_ci is not None
                else None
            ),
            "actual_fwhm_ci95_high_mm": (
                float(actual_fwhm_ci[1])
                if actual_fwhm_ci is not None
                else None
            ),
            "posthoc_fwhm_mm": (
                float(predicted_fwhm)
                if predicted_fwhm is not None
                else None
            ),
            "fwhm_prediction_error_mm": (
                float(actual_fwhm - predicted_fwhm)
                if actual_fwhm is not None and predicted_fwhm is not None
                else None
            ),
            "fwhm_relative_prediction_error": (
                fwhm_relative_prediction_error
            ),
            "fwhm_prediction_within_10_percent": (
                bool(
                    fwhm_relative_prediction_error is not None
                    and abs(fwhm_relative_prediction_error) <= 0.10
                )
                if radius != full_radius
                else None
            ),
            "fwhm_change_direction_matches_posthoc": direction_matches,
            "pre_registered_numeric_thresholds_pass": (
                numeric_thresholds_pass
            ),
            "analysis_dir": item["analysis_dir"],
        }
        rows.append(row)
    rows.sort(key=lambda row: row["outer_radius_mm"])

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "physical_d1_validation.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    radius_grid = np.asarray(sorted(posthoc_by_radius), dtype=np.float64)
    posthoc_efficiency = np.asarray(
        [
            posthoc_by_radius[radius]["efficiency_retention"]
            for radius in radius_grid
        ],
        dtype=np.float64,
    )
    posthoc_fwhm = np.asarray(
        [
            posthoc_by_radius[radius]["quality_value_mm"]
            for radius in radius_grid
        ],
        dtype=np.float64,
    )
    posthoc_mass = np.asarray(
        [
            posthoc_by_radius[radius]["d1_mass_kg"]
            for radius in radius_grid
        ],
        dtype=np.float64,
    )
    physical_radius = np.asarray(
        [row["outer_radius_mm"] for row in rows],
        dtype=np.float64,
    )
    actual_efficiency = np.asarray(
        [row["actual_efficiency_retention_vs_full"] for row in rows],
        dtype=np.float64,
    )
    actual_fwhm = np.asarray(
        [row["actual_fwhm_mm"] for row in rows],
        dtype=np.float64,
    )
    actual_fwhm_ci_low = np.asarray(
        [row["actual_fwhm_ci95_low_mm"] for row in rows],
        dtype=np.float64,
    )
    actual_fwhm_ci_high = np.asarray(
        [row["actual_fwhm_ci95_high_mm"] for row in rows],
        dtype=np.float64,
    )

    plt = _configure_matplotlib()
    mm_to_inch = 1.0 / 25.4
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(180.0 * mm_to_inch, 62.0 * mm_to_inch),
    )
    axes[0].plot(
        radius_grid,
        posthoc_efficiency,
        color="#767676",
        linewidth=1.1,
        label="Post-hoc active-area cut",
    )
    axes[0].scatter(
        physical_radius,
        actual_efficiency,
        color="#0F4D92",
        s=22,
        zorder=3,
        label="Physical GATE geometry",
    )
    axes[0].set_xlabel("D1 outer radius (mm)")
    axes[0].set_ylabel("Efficiency retention vs 200 mm")
    axes[0].legend(fontsize=6.1)

    axes[1].plot(
        radius_grid,
        posthoc_fwhm,
        color="#767676",
        linewidth=1.1,
        label="Post-hoc active-area cut",
    )
    axes[1].errorbar(
        physical_radius,
        actual_fwhm,
        yerr=np.vstack(
            [
                actual_fwhm - actual_fwhm_ci_low,
                actual_fwhm_ci_high - actual_fwhm,
            ]
        ),
        fmt="o",
        color="#B64342",
        markersize=5.2,
        capsize=2.5,
        elinewidth=0.9,
        zorder=3,
        label="Physical GATE geometry (95% CI)",
    )
    axes[1].set_xlabel("D1 outer radius (mm)")
    axes[1].set_ylabel("Axial FWHM (mm)")
    axes[1].legend(fontsize=6.1)

    axes[2].plot(
        radius_grid,
        posthoc_mass,
        "o-",
        color="#0F4D92",
        linewidth=1.1,
        markersize=3,
    )
    axes[2].set_xlabel("D1 outer radius (mm)")
    axes[2].set_ylabel("GAGG active mass (kg)")
    for label, ax in zip("abc", axes):
        ax.text(
            -0.18,
            1.04,
            label,
            transform=ax.transAxes,
            fontweight="bold",
            fontsize=9,
        )
    fig.tight_layout(pad=1.1, w_pad=1.8)
    figures = _save_figure(fig, out_dir / "physical_d1_validation")
    plt.close(fig)

    physical_only = [row for row in rows if row["physical_run"]]
    direction_results = [
        row["fwhm_change_direction_matches_posthoc"]
        for row in physical_only
        if row["fwhm_change_direction_matches_posthoc"] is not None
    ]
    systematic_direction_reversal = bool(
        len(direction_results) >= 2
        and all(result is False for result in direction_results)
    )
    report = {
        "schema_version": "1.0",
        "full_radius_mm": full_radius,
        "rows": rows,
        "maximum_absolute_efficiency_retention_prediction_error": max(
            abs(row["efficiency_retention_prediction_error"])
            for row in physical_only
        ),
        "maximum_absolute_fwhm_prediction_error_mm": max(
            abs(row["fwhm_prediction_error_mm"])
            for row in physical_only
            if row["fwhm_prediction_error_mm"] is not None
        ),
        "all_physical_sizes_pass_pre_registered_thresholds": bool(
            physical_only
            and all(
                row["pre_registered_numeric_thresholds_pass"] is True
                for row in physical_only
            )
            and not systematic_direction_reversal
        ),
        "systematic_fwhm_change_direction_reversal": (
            systematic_direction_reversal
        ),
        "pre_registered_thresholds": {
            "maximum_absolute_efficiency_retention_error": 0.02,
            "maximum_absolute_fwhm_relative_error": 0.10,
            "systematic_direction_reversal_disallowed": True,
            "systematic_direction_reversal_definition": (
                "At least two tested physical radii are available and every "
                "physical FWHM change relative to the 200 mm baseline has the "
                "opposite sign from its post-hoc prediction."
            ),
        },
        "source_csv": str(csv_path.resolve()),
        "figures": figures,
        "guardrail": (
            "Agreement validates the use of post-hoc active-area cuts for this "
            "unchanged passive geometry only. It does not include packaging, "
            "gaps, edge escape changes, optical transport, or readout channels."
        ),
    }
    with (out_dir / "physical_d1_validation.json").open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    return report


def compare_concrete_repeatability(
    *,
    metrics_paths: Iterable[str | Path],
    out_dir: str | Path,
) -> dict[str, Any]:
    paths = [Path(path) for path in metrics_paths]
    if len(paths) != 2:
        raise ValueError("concrete repeatability currently requires two runs")
    reports = [_load_json(path) for path in paths]
    if any(
        report.get("kind") != "homogeneous_concrete_depth_profile"
        for report in reports
    ):
        raise ValueError("inputs must be concrete depth-profile metrics")
    configs = [load_config(report["config"]) for report in reports]
    seeds = [int(config["simulation"]["random_seed"]) for config in configs]
    by_depth = [
        {
            float(row["depth_bin_center_mm"]): row
            for row in report["rows"]
        }
        for report in reports
    ]
    common_depths = sorted(set(by_depth[0]) & set(by_depth[1]))
    if not common_depths:
        raise ValueError("concrete profiles have no common depth bins")

    rows: list[dict[str, Any]] = []
    for depth in common_depths:
        first = by_depth[0][depth]
        second = by_depth[1][depth]
        sensitivity_a = float(
            first["observable_sensitivity_per_primary_per_mm"]
        )
        sensitivity_b = float(
            second["observable_sensitivity_per_primary_per_mm"]
        )
        fwhm_a = first.get("axial_error_fwhm_mm")
        fwhm_b = second.get("axial_error_fwhm_mm")
        ci_a = (
            [
                first["axial_error_fwhm_ci_low_mm"],
                first["axial_error_fwhm_ci_high_mm"],
            ]
            if first.get("axial_error_fwhm_ci_low_mm") is not None
            and first.get("axial_error_fwhm_ci_high_mm") is not None
            else None
        )
        ci_b = (
            [
                second["axial_error_fwhm_ci_low_mm"],
                second["axial_error_fwhm_ci_high_mm"],
            ]
            if second.get("axial_error_fwhm_ci_low_mm") is not None
            and second.get("axial_error_fwhm_ci_high_mm") is not None
            else None
        )
        rows.append(
            {
                "depth_bin_center_mm": depth,
                f"sensitivity_seed_{seeds[0]}": sensitivity_a,
                f"sensitivity_seed_{seeds[1]}": sensitivity_b,
                "sensitivity_relative_difference_vs_pair_mean": float(
                    (sensitivity_b - sensitivity_a)
                    / (0.5 * (sensitivity_a + sensitivity_b))
                )
                if sensitivity_a + sensitivity_b > 0.0
                else None,
                f"fwhm_seed_{seeds[0]}_mm": fwhm_a,
                f"fwhm_seed_{seeds[1]}_mm": fwhm_b,
                "fwhm_difference_mm": (
                    float(fwhm_b - fwhm_a)
                    if fwhm_a is not None and fwhm_b is not None
                    else None
                ),
                "fwhm_difference_z_approx": (
                    _difference_z(
                        float(fwhm_a),
                        _ci_standard_error(ci_a),
                        float(fwhm_b),
                        _ci_standard_error(ci_b),
                    )
                    if fwhm_a is not None and fwhm_b is not None
                    else None
                ),
            }
        )

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "concrete_repeatability.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    plt = _configure_matplotlib()
    mm_to_inch = 1.0 / 25.4
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(180.0 * mm_to_inch, 62.0 * mm_to_inch),
    )
    colors = ["#0F4D92", "#B64342"]
    for seed, profile, color in zip(seeds, by_depth, colors):
        depth = np.asarray(common_depths)
        sensitivity = np.asarray(
            [
                profile[value][
                    "observable_sensitivity_per_primary_per_mm"
                ]
                for value in common_depths
            ]
        )
        fwhm = np.asarray(
            [
                profile[value].get("axial_error_fwhm_mm", np.nan)
                if profile[value].get("axial_error_fwhm_mm") is not None
                else np.nan
                for value in common_depths
            ]
        )
        axes[0].plot(
            depth,
            sensitivity,
            "o-",
            color=color,
            linewidth=1.1,
            markersize=2.8,
            label=f"Seed {seed}",
        )
        axes[1].plot(
            depth,
            fwhm,
            "o-",
            color=color,
            linewidth=1.1,
            markersize=2.8,
            label=f"Seed {seed}",
        )
    axes[0].set_yscale("log")
    axes[0].set_xlabel("True first-Compton depth (mm)")
    axes[0].set_ylabel("Sensitivity / primary / mm")
    axes[0].legend(fontsize=6.1)
    axes[1].set_xlabel("True first-Compton depth (mm)")
    axes[1].set_ylabel("Axial-error FWHM (mm)")
    axes[1].legend(fontsize=6.1)

    relative = np.asarray(
        [
            row["sensitivity_relative_difference_vs_pair_mean"]
            for row in rows
        ],
        dtype=np.float64,
    )
    axes[2].axhline(0.0, color="#767676", linewidth=0.8)
    axes[2].plot(
        common_depths,
        100.0 * relative,
        "o-",
        color="#0F4D92",
        linewidth=1.1,
        markersize=2.8,
    )
    axes[2].set_xlabel("True first-Compton depth (mm)")
    axes[2].set_ylabel("Sensitivity difference vs pair mean (%)")
    for label, ax in zip("abc", axes):
        ax.text(
            -0.18,
            1.04,
            label,
            transform=ax.transAxes,
            fontweight="bold",
            fontsize=9,
        )
    fig.tight_layout(pad=1.1, w_pad=1.8)
    figures = _save_figure(fig, out_dir / "concrete_repeatability")
    plt.close(fig)

    evaluable_fwhm_z = [
        abs(row["fwhm_difference_z_approx"])
        for row in rows
        if row["fwhm_difference_z_approx"] is not None
    ]
    report = {
        "schema_version": "1.0",
        "seeds": seeds,
        "rows": rows,
        "maximum_absolute_sensitivity_relative_difference": float(
            np.max(np.abs(relative))
        ),
        "median_absolute_sensitivity_relative_difference": float(
            np.median(np.abs(relative))
        ),
        "maximum_absolute_fwhm_difference_z_approx": (
            float(max(evaluable_fwhm_z)) if evaluable_fwhm_z else None
        ),
        "source_csv": str(csv_path.resolve()),
        "figures": figures,
        "guardrail": (
            "Two seeds detect gross instability but do not establish final "
            "uncertainty. Deep bins can be count-limited and should be judged "
            "with their bootstrap status and event counts."
        ),
    }
    with (out_dir / "concrete_repeatability.json").open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    return report


def validate_concrete_physical_d1(
    *,
    full_analysis_dir: str | Path,
    physical_analysis_dir: str | Path,
    full_config_path: str | Path,
    physical_config_path: str | Path,
    out_dir: str | Path,
    bin_width_mm: float = 10.0,
) -> dict[str, Any]:
    """Validate a physical D1 radius against a post-hoc full-D1 radius cut."""
    if bin_width_mm <= 0.0:
        raise ValueError("depth bin width must be positive")
    full_analysis_dir = Path(full_analysis_dir)
    physical_analysis_dir = Path(physical_analysis_dir)
    full_config = load_config(full_config_path)
    physical_config = load_config(physical_config_path)
    physical_radius = float(
        physical_config["detectors"]["d1"]["radius_mm"][1]
    )
    full_radius = float(full_config["detectors"]["d1"]["radius_mm"][1])
    if physical_radius >= full_radius:
        raise ValueError("physical validation radius must be below full radius")

    def load_npz(path: Path) -> dict[str, np.ndarray]:
        with np.load(path, allow_pickle=False) as data:
            return {name: data[name] for name in data.files}

    def load_dataset(
        analysis_dir: Path,
        config: dict[str, Any],
    ) -> dict[str, Any]:
        pairs = load_npz(analysis_dir / "pairs.npz")
        reconstruction = load_npz(analysis_dir / "reconstruction.npz")
        if (
            len(pairs["key"]) != len(reconstruction["key"])
            or not np.array_equal(pairs["key"], reconstruction["key"])
        ):
            raise ValueError("pair and reconstruction keys are not aligned")
        observable_config = copy.deepcopy(config)
        observable_config["selection"]["phantom_truth_policy"] = "ignore"
        selected, _, _ = build_selection(pairs, observable_config)
        usable = selected & reconstruction["valid"].astype(bool)
        volumes = np.char.lower(pairs["d1_compt_volume"].astype(str))
        volume_name = str(
            config["selection"].get(
                "phantom_truth_volume_contains",
                "concrete",
            )
        ).lower()
        truth_signal = (
            (pairs["d1_n_phantom_compton"] == 1)
            & (pairs["d2_n_phantom_compton"] == 0)
            & (np.char.find(volumes, volume_name) >= 0)
        )
        with (analysis_dir / "summary.json").open(
            "r",
            encoding="utf-8",
        ) as handle:
            summary = json.load(handle)
        return {
            "pairs": pairs,
            "reconstruction": reconstruction,
            "usable": usable,
            "truth_signal": truth_signal,
            "n_primaries": int(summary["event_counts"]["primaries"]),
            "random_seed": int(config["simulation"]["random_seed"]),
        }

    full = load_dataset(full_analysis_dir, full_config)
    physical = load_dataset(physical_analysis_dir, physical_config)
    full_d1_radius = np.hypot(
        full["pairs"]["d1_x_mm"],
        full["pairs"]["d1_y_mm"],
    )
    full_mask = full["usable"]
    posthoc_mask = full["usable"] & (full_d1_radius <= physical_radius)
    actual_mask = physical["usable"]

    def count_summary(
        dataset: dict[str, Any],
        mask: np.ndarray,
    ) -> dict[str, Any]:
        reconstruction = dataset["reconstruction"]
        truth_matched = reconstruction["truth_matched"].astype(bool)
        return {
            "observable_selected_valid": int(mask.sum()),
            "truth_matched": int((mask & truth_matched).sum()),
            "truth_single_scatter_signal": int(
                (mask & dataset["truth_signal"]).sum()
            ),
            "observable_efficiency_per_primary": float(
                mask.sum() / dataset["n_primaries"]
            ),
            "truth_signal_efficiency_per_primary": float(
                (mask & dataset["truth_signal"]).sum()
                / dataset["n_primaries"]
            ),
        }

    full_counts = count_summary(full, full_mask)
    posthoc_counts = count_summary(full, posthoc_mask)
    actual_counts = count_summary(physical, actual_mask)
    posthoc_observable_retention = float(
        posthoc_counts["observable_selected_valid"]
        / full_counts["observable_selected_valid"]
    )
    actual_observable_retention = float(
        actual_counts["observable_efficiency_per_primary"]
        / full_counts["observable_efficiency_per_primary"]
    )
    posthoc_signal_retention = float(
        posthoc_counts["truth_single_scatter_signal"]
        / full_counts["truth_single_scatter_signal"]
    )
    actual_signal_retention = float(
        actual_counts["truth_signal_efficiency_per_primary"]
        / full_counts["truth_signal_efficiency_per_primary"]
    )

    z_lo, z_hi = map(float, full_config["phantom"]["z_interval_mm"])
    depth_edges = np.arange(
        -z_hi,
        -z_lo + bin_width_mm * 0.5,
        bin_width_mm,
    )
    rows: list[dict[str, Any]] = []

    def depth_values(
        dataset: dict[str, Any],
        mask: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        reconstruction = dataset["reconstruction"]
        matched = mask & reconstruction["truth_matched"].astype(bool)
        true_depth = -np.asarray(
            reconstruction["truth_z_mm"][matched],
            dtype=np.float64,
        )
        error = (
            -np.asarray(
                reconstruction["z_mm"][matched],
                dtype=np.float64,
            )
            - true_depth
        )
        return true_depth, error

    posthoc_depth, posthoc_error = depth_values(full, posthoc_mask)
    actual_depth, actual_error = depth_values(physical, actual_mask)
    for left, right in zip(depth_edges[:-1], depth_edges[1:]):
        posthoc_in_bin = (
            (posthoc_depth >= left)
            & (posthoc_depth < right)
        )
        actual_in_bin = (
            (actual_depth >= left)
            & (actual_depth < right)
        )
        posthoc_estimate = estimate_fwhm_kde(
            posthoc_error[posthoc_in_bin]
        )
        actual_estimate = estimate_fwhm_kde(actual_error[actual_in_bin])
        posthoc_sensitivity = float(
            posthoc_in_bin.sum()
            / full["n_primaries"]
            / (right - left)
        )
        actual_sensitivity = float(
            actual_in_bin.sum()
            / physical["n_primaries"]
            / (right - left)
        )
        rows.append(
            {
                "depth_bin_left_mm": float(left),
                "depth_bin_right_mm": float(right),
                "depth_bin_center_mm": float(0.5 * (left + right)),
                "posthoc_truth_matched_events": int(
                    posthoc_in_bin.sum()
                ),
                "physical_truth_matched_events": int(
                    actual_in_bin.sum()
                ),
                "posthoc_sensitivity_per_primary_per_mm": posthoc_sensitivity,
                "physical_sensitivity_per_primary_per_mm": actual_sensitivity,
                "physical_to_posthoc_sensitivity_ratio": (
                    float(actual_sensitivity / posthoc_sensitivity)
                    if posthoc_sensitivity > 0.0
                    else None
                ),
                "posthoc_axial_error_fwhm_mm": posthoc_estimate.get("fwhm"),
                "physical_axial_error_fwhm_mm": actual_estimate.get("fwhm"),
                "posthoc_fwhm_status": posthoc_estimate["status"],
                "physical_fwhm_status": actual_estimate["status"],
            }
        )

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "concrete_physical_d1_validation.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    depth = np.asarray([row["depth_bin_center_mm"] for row in rows])
    posthoc_sensitivity = np.asarray(
        [
            row["posthoc_sensitivity_per_primary_per_mm"]
            for row in rows
        ]
    )
    actual_sensitivity = np.asarray(
        [
            row["physical_sensitivity_per_primary_per_mm"]
            for row in rows
        ]
    )
    posthoc_fwhm = np.asarray(
        [
            row["posthoc_axial_error_fwhm_mm"]
            if row["posthoc_axial_error_fwhm_mm"] is not None
            else np.nan
            for row in rows
        ]
    )
    actual_fwhm = np.asarray(
        [
            row["physical_axial_error_fwhm_mm"]
            if row["physical_axial_error_fwhm_mm"] is not None
            else np.nan
            for row in rows
        ]
    )
    plt = _configure_matplotlib()
    mm_to_inch = 1.0 / 25.4
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(180.0 * mm_to_inch, 62.0 * mm_to_inch),
    )
    labels = ["Post-hoc cut", "Physical D1"]
    observable_retention = [
        posthoc_observable_retention,
        actual_observable_retention,
    ]
    signal_retention = [
        posthoc_signal_retention,
        actual_signal_retention,
    ]
    x_bar = np.arange(2)
    axes[0].bar(
        x_bar - 0.16,
        observable_retention,
        width=0.32,
        color="#0F4D92",
        label="Observable",
    )
    axes[0].bar(
        x_bar + 0.16,
        signal_retention,
        width=0.32,
        color="#B64342",
        label="Truth signal (score)",
    )
    axes[0].set_xticks(x_bar, labels)
    axes[0].set_ylabel(f"Retention vs physical R={full_radius:g} mm")
    axes[0].set_ylim(0.0, 1.08)
    axes[0].legend(fontsize=6.0)

    axes[1].plot(
        depth,
        posthoc_sensitivity,
        "o-",
        color="#767676",
        linewidth=1.1,
        markersize=2.8,
        label="Post-hoc cut",
    )
    axes[1].plot(
        depth,
        actual_sensitivity,
        "s-",
        color="#0F4D92",
        linewidth=1.1,
        markersize=2.8,
        label="Physical D1",
    )
    axes[1].set_yscale("log")
    axes[1].set_xlabel("True first-Compton depth (mm)")
    axes[1].set_ylabel("Sensitivity / primary / mm")
    axes[1].legend(fontsize=6.0)

    axes[2].plot(
        depth,
        posthoc_fwhm,
        "o-",
        color="#767676",
        linewidth=1.1,
        markersize=2.8,
        label="Post-hoc cut",
    )
    axes[2].plot(
        depth,
        actual_fwhm,
        "s-",
        color="#0F4D92",
        linewidth=1.1,
        markersize=2.8,
        label="Physical D1",
    )
    axes[2].set_xlabel("True first-Compton depth (mm)")
    axes[2].set_ylabel("Axial-error FWHM (mm)")
    axes[2].legend(fontsize=6.0)
    for label, ax in zip("abc", axes):
        ax.text(
            -0.18,
            1.04,
            label,
            transform=ax.transAxes,
            fontweight="bold",
            fontsize=9,
        )
    fig.tight_layout(pad=1.1, w_pad=1.8)
    figures = _save_figure(fig, out_dir / "concrete_physical_d1_validation")
    plt.close(fig)

    report = {
        "schema_version": "1.0",
        "full_analysis_dir": str(full_analysis_dir.resolve()),
        "physical_analysis_dir": str(physical_analysis_dir.resolve()),
        "full_radius_mm": full_radius,
        "physical_radius_mm": physical_radius,
        "random_seeds": {
            "full": full["random_seed"],
            "physical": physical["random_seed"],
        },
        "full_counts": full_counts,
        "posthoc_counts": posthoc_counts,
        "physical_counts": actual_counts,
        "retention": {
            "posthoc_observable": posthoc_observable_retention,
            "physical_observable": actual_observable_retention,
            "observable_prediction_error": (
                actual_observable_retention
                - posthoc_observable_retention
            ),
            "observable_prediction_within_2_percentage_points": bool(
                abs(
                    actual_observable_retention
                    - posthoc_observable_retention
                )
                <= 0.02
            ),
            "posthoc_truth_signal": posthoc_signal_retention,
            "physical_truth_signal": actual_signal_retention,
            "truth_signal_prediction_error": (
                actual_signal_retention - posthoc_signal_retention
            ),
            "truth_signal_prediction_within_2_percentage_points": bool(
                abs(actual_signal_retention - posthoc_signal_retention)
                <= 0.02
            ),
        },
        "depth_rows": rows,
        "source_csv": str(csv_path.resolve()),
        "figures": figures,
        "guardrail": (
            "The post-hoc prediction and physical run use identical observable "
            "cuts. Truth is used only to score signal and true-depth bins. "
            "Agreement does not model packaging, optical transport or channel "
            "gaps."
        ),
    }
    with (out_dir / "concrete_physical_d1_validation.json").open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    return report
