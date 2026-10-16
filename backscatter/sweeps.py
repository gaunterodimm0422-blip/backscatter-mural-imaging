from __future__ import annotations

import copy
import csv
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .config import load_config
from .pipeline import build_selection
from .standards import estimate_fwhm_kde


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {name: data[name] for name in data.files}


def _finite_quantile(values: np.ndarray, q: float) -> float | None:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    return float(np.quantile(values, q)) if len(values) else None


def _finite_std(values: np.ndarray) -> float | None:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    return float(np.std(values, ddof=1)) if len(values) > 1 else None


def _row_metrics(
    *,
    category: str,
    setting: str,
    selected: np.ndarray,
    truth_signal: np.ndarray | None,
    reconstruction: dict[str, np.ndarray],
    n_primaries: int,
    standard_target_kind: str | None,
    target_depth_mm: float | None,
) -> dict[str, Any]:
    observable_usable = selected & reconstruction["valid"].astype(bool)
    usable = (
        observable_usable & truth_signal
        if truth_signal is not None
        else observable_usable
    )
    n_observable = int(observable_usable.sum())
    n = int(usable.sum())
    x = reconstruction["x_mm"][usable]
    y = reconstruction["y_mm"][usable]
    z = reconstruction["z_mm"][usable]

    row: dict[str, Any] = {
        "category": category,
        "setting": setting,
        "n_selected_valid": n,
        "n_observable_selected_valid": n_observable,
        "n_truth_signal_valid": n if truth_signal is not None else None,
        "truth_signal_fraction": (
            n / n_observable
            if truth_signal is not None and n_observable
            else None
        ),
        "efficiency_per_primary": n / n_primaries if n_primaries else None,
        "observable_efficiency_per_primary": (
            n_observable / n_primaries if n_primaries else None
        ),
        "cps_per_MBq_low_rate_equivalent": (
            n / n_primaries * 1.0e6 if n_primaries else None
        ),
        "recon_x_median_mm": _finite_quantile(x, 0.5),
        "recon_y_median_mm": _finite_quantile(y, 0.5),
        "recon_z_median_mm": _finite_quantile(z, 0.5),
        "recon_x_iqr_mm": (
            _finite_quantile(x, 0.75) - _finite_quantile(x, 0.25)
            if n
            else None
        ),
        "recon_y_iqr_mm": (
            _finite_quantile(y, 0.75) - _finite_quantile(y, 0.25)
            if n
            else None
        ),
        "recon_z_iqr_mm": (
            _finite_quantile(z, 0.75) - _finite_quantile(z, 0.25)
            if n
            else None
        ),
        "truth_matched": 0,
        "truth_error_rmse_x_mm": None,
        "truth_error_rmse_y_mm": None,
        "truth_error_rmse_z_mm": None,
        "truth_error_bias_z_mm": None,
        "truth_error_abs_z_p90_mm": None,
        "axial_psf_fwhm_mm": None,
        "axial_psf_peak_bias_mm": None,
        "transverse_psf_fwhm_x_mm": None,
        "transverse_psf_peak_bias_x_mm": None,
        "resolution_status": "unavailable_without_truth_standard_target",
    }

    truth_usable = usable & reconstruction["truth_matched"].astype(bool)
    nt = int(truth_usable.sum())
    row["truth_matched"] = nt
    if nt:
        dx = (
            reconstruction["x_mm"][truth_usable]
            - reconstruction["truth_x_mm"][truth_usable]
        )
        dy = (
            reconstruction["y_mm"][truth_usable]
            - reconstruction["truth_y_mm"][truth_usable]
        )
        dz = (
            reconstruction["z_mm"][truth_usable]
            - reconstruction["truth_z_mm"][truth_usable]
        )
        row.update(
            {
                "truth_error_rmse_x_mm": float(np.sqrt(np.mean(dx**2))),
                "truth_error_rmse_y_mm": float(np.sqrt(np.mean(dy**2))),
                "truth_error_rmse_z_mm": float(np.sqrt(np.mean(dz**2))),
                "truth_error_bias_z_mm": float(np.mean(dz)),
                "truth_error_abs_z_p90_mm": float(np.quantile(np.abs(dz), 0.9)),
                "resolution_status": (
                    "truth_localization_error_available;"
                    "PSF_or_MTF_requires_standard_target"
                ),
            }
        )
    if standard_target_kind == "thin_layer" and target_depth_mm is not None:
        estimate = estimate_fwhm_kde(-np.asarray(z, dtype=np.float64))
        row["resolution_status"] = (
            "thin_layer_axial_psf_available"
            if estimate["status"] == "ok"
            else f"thin_layer_axial_psf_{estimate['status']}"
        )
        row["axial_psf_fwhm_mm"] = estimate.get("fwhm")
        row["axial_psf_peak_bias_mm"] = (
            float(estimate["mode"] - target_depth_mm)
            if estimate.get("mode") is not None
            else None
        )
    elif standard_target_kind == "rod":
        estimate = estimate_fwhm_kde(np.asarray(x, dtype=np.float64))
        row["resolution_status"] = (
            "rod_transverse_psf_available"
            if estimate["status"] == "ok"
            else f"rod_transverse_psf_{estimate['status']}"
        )
        row["transverse_psf_fwhm_x_mm"] = estimate.get("fwhm")
        row["transverse_psf_peak_bias_x_mm"] = estimate.get("mode")
    return row


def _truth_signal_mask(
    pairs: dict[str, np.ndarray],
    config: dict[str, Any],
) -> np.ndarray | None:
    volumes = np.char.lower(pairs["d1_compt_volume"].astype(str))
    available = bool(
        np.any(pairs["d1_n_phantom_compton"] > 0)
        or np.any((volumes != "null") & (volumes != "unavailable"))
    )
    if not available:
        return None
    volume = str(
        config["selection"].get(
            "phantom_truth_volume_contains",
            config["phantom"].get("gate_volume", config["phantom"]["name"]),
        )
    ).lower()
    return (
        (pairs["d1_n_phantom_compton"] == 1)
        & (pairs["d2_n_phantom_compton"] == 0)
        & (np.char.find(volumes, volume) >= 0)
    )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _annotate_tradeoffs(
    rows: list[dict[str, Any]],
    *,
    standard_target_kind: str | None,
) -> list[dict[str, Any]]:
    if not rows:
        return rows
    baseline = rows[0]
    baseline_efficiency = baseline["efficiency_per_primary"]
    min_events = max(100, int(np.ceil(0.05 * baseline["n_selected_valid"])))
    if standard_target_kind == "thin_layer":
        quality_field = "axial_psf_fwhm_mm"
    elif standard_target_kind == "rod":
        quality_field = "transverse_psf_fwhm_x_mm"
    else:
        quality_field = "truth_error_rmse_z_mm"
    baseline_quality = baseline[quality_field]
    for row in rows:
        quality = row[quality_field]
        row["quality_metric"] = quality_field
        row["quality_value_mm"] = quality
        row["efficiency_change_pct_vs_baseline"] = (
            100.0
            * (row["efficiency_per_primary"] / baseline_efficiency - 1.0)
            if baseline_efficiency
            else None
        )
        row["quality_change_pct_vs_baseline"] = (
            100.0 * (quality / baseline_quality - 1.0)
            if quality is not None and baseline_quality
            else None
        )
        row["pareto_quality_efficiency"] = False
        row["pareto_eligible"] = bool(
            row["n_selected_valid"] >= min_events
            and quality is not None
            and row["efficiency_per_primary"] is not None
            and row["efficiency_per_primary"] > 0.0
        )
        row["pareto_min_events"] = min_events

    valid = [
        index
        for index, row in enumerate(rows)
        if row["pareto_eligible"]
    ]
    for index in valid:
        candidate = rows[index]
        dominated = False
        for other_index in valid:
            if other_index == index:
                continue
            other = rows[other_index]
            same_point = np.isclose(
                other["quality_value_mm"],
                candidate["quality_value_mm"],
                rtol=1.0e-9,
                atol=1.0e-12,
            ) and np.isclose(
                other["efficiency_per_primary"],
                candidate["efficiency_per_primary"],
                rtol=1.0e-9,
                atol=1.0e-12,
            )
            if same_point and other_index < index:
                dominated = True
                break
            no_worse = (
                other["quality_value_mm"] <= candidate["quality_value_mm"]
                and other["efficiency_per_primary"]
                >= candidate["efficiency_per_primary"]
            )
            strictly_better = (
                other["quality_value_mm"] < candidate["quality_value_mm"]
                or other["efficiency_per_primary"]
                > candidate["efficiency_per_primary"]
            )
            if no_worse and strictly_better:
                dominated = True
                break
        candidate["pareto_quality_efficiency"] = not dominated
    return rows


def _plot_sweep_figures(
    *,
    rows: list[dict[str, Any]],
    row_masks: list[np.ndarray],
    reconstruction: dict[str, np.ndarray],
    config: dict[str, Any],
    out_dir: Path,
) -> dict[str, Any]:
    """Create response-distribution and efficiency-resolution trade-off figures."""
    kind = config["phantom"].get("target_kind")
    if kind not in {"thin_layer", "rod"}:
        return {
            "figures": [],
            "response_curve_source_data": None,
            "note": "Sweep figures require a thin-layer or rod standard target.",
        }
    if len(rows) != len(row_masks):
        raise ValueError("sweep row/mask alignment failed")

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

    def find_index(category: str, setting: str) -> int | None:
        for index, row in enumerate(rows):
            if row["category"] == category and row["setting"] == setting:
                return index
        return None

    baseline_index = 0
    selection_groups = [
        (
            "D1 energy window",
            [
                ("Baseline", baseline_index),
                ("180–250 keV", find_index("d1_energy_window", "180-250 keV")),
                ("190–240 keV", find_index("d1_energy_window", "190-240 keV")),
            ],
        ),
        (
            "Incident-angle range",
            [
                ("0–45°", baseline_index),
                ("0–30°", find_index("incident_angle", "0-30 deg")),
                ("0–20°", find_index("incident_angle", "0-20 deg")),
            ],
        ),
        (
            "D1 active outer radius",
            [
                ("R = 200 mm", baseline_index),
                ("R = 150 mm", find_index("d1_radius_cumulative", "20-150 mm")),
                ("R = 125 mm", find_index("d1_radius_cumulative", "20-125 mm")),
                ("R = 100 mm", find_index("d1_radius_cumulative", "20-100 mm")),
            ],
        ),
    ]
    colors = ["#0F4D92", "#D57A1F", "#2E7D5B", "#B64342"]
    values_all = (
        -np.asarray(reconstruction["z_mm"], dtype=np.float64)
        if kind == "thin_layer"
        else np.asarray(reconstruction["x_mm"], dtype=np.float64)
    )
    valid = reconstruction["valid"].astype(bool)
    curve_rows: list[dict[str, Any]] = []

    mm_to_inch = 1.0 / 25.4
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(180.0 * mm_to_inch, 62.0 * mm_to_inch),
        sharey=True,
    )
    for panel, (title, members) in enumerate(selection_groups):
        ax = axes[panel]
        for color, (label, index) in zip(colors, members):
            if index is None:
                continue
            usable = row_masks[index] & valid
            estimate = estimate_fwhm_kde(values_all[usable])
            if "grid" not in estimate:
                continue
            grid = np.asarray(estimate["grid"], dtype=np.float64)
            density = np.asarray(estimate["density"], dtype=np.float64)
            normalized = density / np.max(density)
            fwhm = estimate.get("fwhm")
            metric = f"; {fwhm:.1f} mm" if fwhm is not None else ""
            legend = f"{label} (n={int(usable.sum()):,}{metric})"
            ax.plot(grid, normalized, color=color, linewidth=1.25, label=legend)
            for coordinate, response in zip(grid, normalized):
                curve_rows.append(
                    {
                        "panel": panel + 1,
                        "factor": title,
                        "label": label,
                        "category": rows[index]["category"],
                        "setting": rows[index]["setting"],
                        "n_selected_valid": int(usable.sum()),
                        "truth_signal_fraction": rows[index][
                            "truth_signal_fraction"
                        ],
                        "response_coordinate_mm": float(coordinate),
                        "normalized_response": float(response),
                        "fwhm_mm": fwhm,
                    }
                )
        if kind == "thin_layer":
            z_lo, z_hi = map(float, config["phantom"]["z_interval_mm"])
            ax.axvspan(
                -z_hi,
                -z_lo,
                color="#D8D8D8",
                alpha=0.65,
                linewidth=0,
                zorder=0,
            )
            ax.set_xlabel("Reconstructed axial distance (mm)")
        else:
            ax.set_xlabel("Reconstructed x (mm)")
        ax.set_title(title, fontsize=7.2)
        ax.set_ylim(0.0, 1.06)
        ax.legend(fontsize=5.4, loc="best", handlelength=1.7)
        ax.text(
            -0.17,
            1.04,
            "abc"[panel],
            transform=ax.transAxes,
            fontweight="bold",
            fontsize=9,
        )
    axes[0].set_ylabel("Normalized response")
    fig.tight_layout(w_pad=1.4)

    curve_csv = out_dir / "sweep_response_curves.csv"
    _write_csv(curve_csv, curve_rows)
    figures: list[str] = []
    for extension, dpi in (
        ("svg", 300),
        ("pdf", 300),
        ("tiff", 600),
        ("png", 220),
    ):
        path = out_dir / f"filter_response_distributions.{extension}"
        fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
        figures.append(str(path.resolve()))
    plt.close(fig)

    category_style = {
        "baseline": ("Baseline", "#222222", "*"),
        "d1_energy_window": ("D1 energy", "#0F4D92", "o"),
        "d2_energy_window": ("D2 energy", "#5B8EBD", "s"),
        "incident_angle": ("Incident angle", "#D57A1F", "^"),
        "scatter_angle": ("Scatter angle", "#8B5AA5", "v"),
        "d1_radius_cumulative": ("D1 radius", "#2E7D5B", "D"),
        "d1_radial_band": ("D1 radial band", "#4C956C", "P"),
        "d1_azimuth_sector": ("D1 sector", "#B64342", "o"),
        "d2_region": ("D2 central area", "#117A8B", "s"),
        "d2_quadrant": ("D2 quadrant", "#7A5195", "^"),
        "d2_radial_band": ("D2 radial band", "#A66A2C", "D"),
    }
    tradeoff_groups = [
        (
            [
                "baseline",
                "d1_energy_window",
                "d2_energy_window",
                "incident_angle",
                "scatter_angle",
                "d1_radius_cumulative",
            ],
            "Event filters and active radius",
        ),
        (
            [
                "baseline",
                "d1_radial_band",
                "d1_azimuth_sector",
                "d2_region",
                "d2_quadrant",
                "d2_radial_band",
            ],
            "Detector-region use without geometry changes",
        ),
    ]
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(180.0 * mm_to_inch, 66.0 * mm_to_inch),
    )
    for panel, (allowed, title) in enumerate(tradeoff_groups):
        ax = axes[panel]
        for category in allowed:
            selected_rows = [
                row
                for row in rows
                if row["category"] == category
                and row["pareto_eligible"]
                and row["quality_value_mm"] is not None
            ]
            if not selected_rows:
                continue
            label, color, marker = category_style[category]
            ax.scatter(
                [row["quality_value_mm"] for row in selected_rows],
                [
                    row["efficiency_per_primary"] * 1.0e6
                    for row in selected_rows
                ],
                color=color,
                marker=marker,
                s=34 if category == "baseline" else 18,
                linewidths=0.7,
                edgecolors="white" if category != "baseline" else color,
                label=label,
                zorder=4 if category == "baseline" else 2,
            )
        ax.set_xlabel(
            "Axial FWHM (mm)"
            if kind == "thin_layer"
            else "Transverse x FWHM (mm)"
        )
        ax.set_ylabel(
            "Truth-target valid events / $10^6$ primaries"
            if rows[0]["n_truth_signal_valid"] is not None
            else "Valid events / $10^6$ primaries"
        )
        ax.set_yscale("log")
        ax.set_title(title, fontsize=7.2)
        ax.legend(fontsize=5.5, ncol=2, loc="best")
        ax.text(
            -0.15,
            1.04,
            "ab"[panel],
            transform=ax.transAxes,
            fontweight="bold",
            fontsize=9,
        )
    fig.tight_layout(w_pad=1.8)
    for extension, dpi in (
        ("svg", 300),
        ("pdf", 300),
        ("tiff", 600),
        ("png", 220),
    ):
        path = out_dir / f"sweep_tradeoffs.{extension}"
        fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
        figures.append(str(path.resolve()))
    plt.close(fig)

    return {
        "figures": figures,
        "response_curve_source_data": str(curve_csv.resolve()),
        "note": (
            "Cuts use observable energy, angle and detector-position fields. "
            "Truth labels are used only to score target-signal efficiency, "
            "purity and resolution. Response curves are target-signal curves "
            "from the same standard-target run; points with too few signal "
            "events are excluded by the recorded Pareto threshold."
        ),
    }


def run_sweeps(
    *,
    analysis_dir: str | Path,
    config_path: str | Path,
    out_dir: str | Path | None = None,
) -> dict[str, Any]:
    analysis_dir = Path(analysis_dir)
    out_dir = Path(out_dir) if out_dir else analysis_dir / "sweeps"
    out_dir.mkdir(parents=True, exist_ok=True)

    config = load_config(config_path)
    pairs = _load_npz(analysis_dir / "pairs.npz")
    reconstruction = _load_npz(analysis_dir / "reconstruction.npz")
    with (analysis_dir / "summary.json").open("r", encoding="utf-8") as handle:
        summary = json.load(handle)
    n_primaries = int(summary["event_counts"]["primaries"])
    standard_target_kind = config["phantom"].get("target_kind")
    target_depth_mm = config["phantom"].get("depth_mm")

    rows: list[dict[str, Any]] = []
    row_masks: list[np.ndarray] = []
    observable_config = copy.deepcopy(config)
    observable_config["selection"]["phantom_truth_policy"] = "ignore"
    base_selected, _, diagnostics = build_selection(pairs, observable_config)
    truth_signal = _truth_signal_mask(pairs, config)
    rows.append(
        _row_metrics(
            category="baseline",
            setting="observable cuts; truth-scored signal",
            selected=base_selected,
            truth_signal=truth_signal,
            reconstruction=reconstruction,
            n_primaries=n_primaries,
            standard_target_kind=standard_target_kind,
            target_depth_mm=(
                float(target_depth_mm)
                if target_depth_mm is not None
                else None
            ),
        )
    )
    row_masks.append(
        (base_selected & truth_signal).copy()
        if truth_signal is not None
        else base_selected.copy()
    )

    def add_config_row(category: str, setting: str, changed: dict[str, Any]) -> None:
        cfg = copy.deepcopy(observable_config)
        cfg["selection"].update(changed)
        selected, _, _ = build_selection(pairs, cfg)
        rows.append(
            _row_metrics(
                category=category,
                setting=setting,
                selected=selected,
                truth_signal=truth_signal,
                reconstruction=reconstruction,
                n_primaries=n_primaries,
                standard_target_kind=standard_target_kind,
                target_depth_mm=(
                    float(target_depth_mm)
                    if target_depth_mm is not None
                    else None
                ),
            )
        )
        row_masks.append(
            (selected & truth_signal).copy()
            if truth_signal is not None
            else selected.copy()
        )

    grid = config["sweep_grid"]
    for lo, hi in grid["d1_energy_windows_keV"]:
        add_config_row(
            "d1_energy_window",
            f"{lo:g}-{hi:g} keV",
            {"d1_energy_window_keV": [lo, hi]},
        )
    for lo, hi in grid["d2_energy_windows_keV"]:
        add_config_row(
            "d2_energy_window",
            f"{lo:g}-{hi:g} keV",
            {"d2_energy_window_keV": [lo, hi]},
        )
    beta_lo = float(config["selection"]["incident_angle_deg"][0])
    for maximum in grid["incident_angle_max_deg"]:
        add_config_row(
            "incident_angle",
            f"{beta_lo:g}-{maximum:g} deg",
            {"incident_angle_deg": [beta_lo, maximum]},
        )
    theta_hi = float(config["selection"]["scatter_angle_deg"][1])
    for minimum in grid["scatter_angle_min_deg"]:
        add_config_row(
            "scatter_angle",
            f"{minimum:g}-{theta_hi:g} deg",
            {"scatter_angle_deg": [minimum, theta_hi]},
        )
    r_lo = float(config["selection"]["d1_radius_mm"][0])
    for maximum in grid["d1_radius_max_mm"]:
        add_config_row(
            "d1_radius_cumulative",
            f"{r_lo:g}-{maximum:g} mm",
            {"d1_radius_mm": [r_lo, maximum]},
        )

    def add_mask_row(category: str, setting: str, condition: np.ndarray) -> None:
        selected = base_selected & condition
        rows.append(
            _row_metrics(
                category=category,
                setting=setting,
                selected=selected,
                truth_signal=truth_signal,
                reconstruction=reconstruction,
                n_primaries=n_primaries,
                standard_target_kind=standard_target_kind,
                target_depth_mm=(
                    float(target_depth_mm)
                    if target_depth_mm is not None
                    else None
                ),
            )
        )
        row_masks.append(
            (selected & truth_signal).copy()
            if truth_signal is not None
            else selected.copy()
        )

    radius = diagnostics["d1_radius_mm"]
    radial_edges = [20.0, 50.0, 75.0, 100.0, 125.0, 150.0, 175.0, 200.0]
    for lo, hi in zip(radial_edges[:-1], radial_edges[1:]):
        add_mask_row(
            "d1_radial_band",
            f"{lo:g}-{hi:g} mm",
            (radius >= lo) & (radius < hi),
        )

    phi = np.mod(
        np.degrees(np.arctan2(pairs["d1_y_mm"], pairs["d1_x_mm"])),
        360.0,
    )
    for sector_count in grid["d1_azimuth_sector_count"]:
        sector_count = int(sector_count)
        width = 360.0 / sector_count
        for sector in range(sector_count):
            lo = sector * width
            hi = (sector + 1) * width
            add_mask_row(
                "d1_azimuth_sector",
                f"N={sector_count}, sector={sector}, phi={lo:g}-{hi:g} deg",
                (phi >= lo) & (phi < hi),
            )

    d2x = pairs["d2_x_mm"]
    d2y = pairs["d2_y_mm"]
    d2r = np.hypot(d2x, d2y)
    add_mask_row("d2_region", "all", np.ones(len(d2x), dtype=bool))
    add_mask_row(
        "d2_region",
        "center_square_10mm",
        (np.abs(d2x) <= 5.0) & (np.abs(d2y) <= 5.0),
    )
    add_mask_row(
        "d2_region",
        "center_square_20mm",
        (np.abs(d2x) <= 10.0) & (np.abs(d2y) <= 10.0),
    )
    quadrant_conditions: list[tuple[str, np.ndarray]] = [
        ("Q1_x+_y+", (d2x >= 0.0) & (d2y >= 0.0)),
        ("Q2_x-_y+", (d2x < 0.0) & (d2y >= 0.0)),
        ("Q3_x-_y-", (d2x < 0.0) & (d2y < 0.0)),
        ("Q4_x+_y-", (d2x >= 0.0) & (d2y < 0.0)),
    ]
    for name, condition in quadrant_conditions:
        add_mask_row("d2_quadrant", name, condition)
    for lo, hi in ((0.0, 8.0), (8.0, 16.0), (16.0, 36.0)):
        add_mask_row(
            "d2_radial_band",
            f"{lo:g}-{hi:g} mm",
            (d2r >= lo) & (d2r < hi),
        )

    _annotate_tradeoffs(
        rows,
        standard_target_kind=standard_target_kind,
    )
    _write_csv(out_dir / "sweep_metrics.csv", rows)
    pareto_rows = [
        row for row in rows if row["pareto_quality_efficiency"]
    ]
    _write_csv(out_dir / "pareto_candidates.csv", pareto_rows)
    figure_report = _plot_sweep_figures(
        rows=rows,
        row_masks=row_masks,
        reconstruction=reconstruction,
        config=config,
        out_dir=out_dir,
    )
    machine = {
        "schema_version": "1.0",
        "analysis_dir": str(analysis_dir.resolve()),
        "config": str(Path(config_path).resolve()),
        "n_primaries": n_primaries,
        "truth_matched_in_dataset": int(
            reconstruction["truth_matched"].astype(bool).sum()
        ),
        "truth_scoring_available": truth_signal is not None,
        "pareto_candidates": pareto_rows,
        "figure_report": figure_report,
        "caveats": [
            "Homogeneous-slab coordinate widths are distributions, not system resolution.",
            "Truth error columns are localization errors only; PSF/MTF requires a standard thin-layer, rod, double-target or edge phantom.",
            "For thin-layer data, axial_psf_fwhm_mm is a within-row KDE estimate without bootstrap; use evaluate-standard for confidence intervals.",
            "For rod data, transverse_psf_fwhm_x_mm is the corresponding within-row line-spread estimate.",
            "Energy, angle, radius and region cuts are observable-only. Phantom truth is used after selection to score target-signal count, purity and resolution; it is not an experimental cut.",
            "D1 cumulative-radius rows are post-hoc active-area cuts. Final detector-size candidates require GATE reruns with the physical radius changed.",
        ],
        "rows": rows,
    }
    with (out_dir / "sweep_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(machine, handle, ensure_ascii=False, indent=2)
    return machine
