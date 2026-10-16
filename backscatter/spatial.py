from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


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
    paths: list[str] = []
    for extension, dpi in (
        ("svg", 300),
        ("pdf", 300),
        ("tiff", 600),
        ("png", 300),
    ):
        path = output_base.with_suffix(f".{extension}")
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        paths.append(str(path.resolve()))
    return paths


def summarize_rod_series(
    *,
    metrics_paths: Iterable[str | Path],
    out_dir: str | Path,
) -> dict[str, Any]:
    paths = [Path(path) for path in metrics_paths]
    if len(paths) < 2:
        raise ValueError("rod summary requires at least two widths")
    reports = [_load_json(path) for path in paths]
    if any(report.get("kind") != "rod" for report in reports):
        raise ValueError("all rod inputs must have kind='rod'")
    rows: list[dict[str, Any]] = []
    for path, report in zip(paths, reports):
        x_response = report["transverse_x_response"]
        axial_response = report["axial_depth_response"]
        x_estimate = x_response["estimate"]
        x_ci = x_response["bootstrap"].get("fwhm_95_ci")
        rows.append(
            {
                "rod_diameter_mm": float(report["target_diameter_mm"]),
                "n_primaries": int(report["n_primaries"]),
                "n_selected_valid": int(report["n_selected_valid"]),
                "efficiency_per_primary": float(
                    report["efficiency_per_primary"]
                ),
                "transverse_x_fwhm_mm": x_estimate.get("fwhm"),
                "transverse_x_fwhm_ci_low_mm": (
                    float(x_ci[0]) if x_ci is not None else None
                ),
                "transverse_x_fwhm_ci_high_mm": (
                    float(x_ci[1]) if x_ci is not None else None
                ),
                "axial_fwhm_mm": axial_response["estimate"].get("fwhm"),
                "x_localization_bias_mm": report["event_truth_error"][
                    "x_bias_mm"
                ],
                "x_eventwise_rmse_mm": report["event_truth_error"]["x_rmse_mm"],
                "metrics_path": str(path.resolve()),
            }
        )
    rows.sort(key=lambda row: row["rod_diameter_mm"])
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "rod_width_series.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    width = np.asarray([row["rod_diameter_mm"] for row in rows])
    fwhm = np.asarray([row["transverse_x_fwhm_mm"] for row in rows])
    efficiency = np.asarray([row["efficiency_per_primary"] for row in rows])
    lower = np.asarray(
        [
            row["transverse_x_fwhm_mm"]
            - row["transverse_x_fwhm_ci_low_mm"]
            for row in rows
        ]
    )
    upper = np.asarray(
        [
            row["transverse_x_fwhm_ci_high_mm"]
            - row["transverse_x_fwhm_mm"]
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
    axes[0].errorbar(
        width,
        fwhm,
        yerr=np.asarray([lower, upper]),
        fmt="o-",
        color="#0F4D92",
        capsize=3,
        linewidth=1.1,
        markersize=3.5,
    )
    axes[0].plot(
        width,
        width,
        "--",
        color="#767676",
        linewidth=0.8,
        label="Target diameter",
    )
    axes[0].set_xlabel("Rod diameter (mm)")
    axes[0].set_ylabel("Measured transverse x FWHM (mm)")
    axes[0].legend(fontsize=6.1)

    axes[1].plot(
        width,
        efficiency,
        "o-",
        color="#B64342",
        linewidth=1.1,
        markersize=3.5,
    )
    axes[1].set_xlabel("Rod diameter (mm)")
    axes[1].set_ylabel("Selected valid events / primary")

    axes[2].plot(
        width,
        [row["x_eventwise_rmse_mm"] for row in rows],
        "o-",
        color="#0F4D92",
        linewidth=1.1,
        markersize=3.5,
        label="x RMSE",
    )
    axes[2].plot(
        width,
        [row["axial_fwhm_mm"] for row in rows],
        "s--",
        color="#767676",
        linewidth=1.0,
        markersize=3.0,
        label="Axial FWHM",
    )
    axes[2].set_xlabel("Rod diameter (mm)")
    axes[2].set_ylabel("Error / response width (mm)")
    axes[2].legend(fontsize=6.1)
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
    figures = _save_figure(fig, out_dir / "rod_width_series")
    plt.close(fig)

    report = {
        "schema_version": "1.0",
        "rows": rows,
        "narrowest_rod_result": rows[0],
        "source_csv": str(csv_path.resolve()),
        "figures": figures,
        "interpretation_guardrail": (
            "The measured line-spread FWHM includes finite rod diameter. "
            "Narrower rods better approximate the system response but have "
            "lower event yield; no unsupported deconvolution is applied."
        ),
    }
    with (out_dir / "rod_width_series.json").open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    return report


def summarize_double_rod_series(
    *,
    metrics_paths: Iterable[str | Path],
    out_dir: str | Path,
) -> dict[str, Any]:
    paths = [Path(path) for path in metrics_paths]
    if len(paths) < 2:
        raise ValueError("double-rod summary requires at least two separations")
    reports = [_load_json(path) for path in paths]
    if any(report.get("kind") != "double_rod" for report in reports):
        raise ValueError("all double-rod inputs must have kind='double_rod'")
    rows: list[dict[str, Any]] = []
    curves: list[tuple[np.ndarray, np.ndarray, float]] = []
    for path, report in zip(paths, reports):
        metrics = report["two_peak_metrics"]
        separation = float(report["target_center_separation_mm"])
        rows.append(
            {
                "target_center_separation_mm": separation,
                "target_diameter_mm": float(report["target_diameter_mm"]),
                "n_primaries": int(report["n_primaries"]),
                "n_selected_valid": int(report["n_selected_valid"]),
                "efficiency_per_primary": float(
                    report["efficiency_per_primary"]
                ),
                "two_peak_status": report["two_peak_status"],
                "measured_peak_separation_mm": metrics[
                    "measured_peak_separation_mm"
                ],
                "valley_to_mean_peak": metrics["valley_to_mean_peak"],
                "valley_to_lower_peak": metrics["valley_to_lower_peak"],
                "operationally_resolved_at_0p735": bool(
                    metrics["operationally_resolved_at_0p735"]
                ),
                "bandwidth_robust_resolved_at_0p735": bool(
                    report.get(
                        "bandwidth_robust_resolved_at_0p735",
                        False,
                    )
                ),
                "metrics_path": str(path.resolve()),
            }
        )
        with Path(report["source_data"]).open(
            "r",
            encoding="utf-8-sig",
            newline="",
        ) as handle:
            curve_rows = list(csv.DictReader(handle))
        curves.append(
            (
                np.asarray(
                    [float(row["x_mm"]) for row in curve_rows],
                    dtype=np.float64,
                ),
                np.asarray(
                    [float(row["density_normalized"]) for row in curve_rows],
                    dtype=np.float64,
                ),
                separation,
            )
        )
    order = np.argsort(
        [row["target_center_separation_mm"] for row in rows]
    )
    rows = [rows[int(index)] for index in order]
    curves = [curves[int(index)] for index in order]
    resolved = [
        row
        for row in rows
        if row["operationally_resolved_at_0p735"]
    ]
    smallest_resolved = resolved[0] if resolved else None
    bandwidth_robust = [
        row
        for row in rows
        if row["bandwidth_robust_resolved_at_0p735"]
    ]
    smallest_bandwidth_robust = (
        bandwidth_robust[0] if bandwidth_robust else None
    )

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "double_rod_separation_series.csv"
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
        gridspec_kw={"width_ratios": [1.35, 0.9, 0.9]},
    )
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(curves)))
    for (x, response, separation), color in zip(curves, colors):
        axes[0].plot(
            x,
            response,
            color=color,
            linewidth=1.0,
            label=f"{separation:g} mm",
        )
    axes[0].set_xlabel("Reconstructed x (mm)")
    axes[0].set_ylabel("Normalized response")
    axes[0].legend(
        title="Center separation",
        fontsize=5.8,
        title_fontsize=5.8,
        ncol=2,
    )

    separation = np.asarray(
        [row["target_center_separation_mm"] for row in rows]
    )
    valley = np.asarray(
        [
            row["valley_to_lower_peak"]
            if row["valley_to_lower_peak"] is not None
            else np.nan
            for row in rows
        ]
    )
    axes[1].axhline(
        0.735,
        color="#767676",
        linestyle="--",
        linewidth=0.9,
        label="Operational threshold",
    )
    axes[1].plot(
        separation,
        valley,
        "o-",
        color="#B64342",
        linewidth=1.1,
        markersize=3.5,
    )
    axes[1].set_xlabel("Target center separation (mm)")
    axes[1].set_ylabel("Valley / lower peak")
    axes[1].set_ylim(0.0, 1.1)
    axes[1].legend(fontsize=5.8)

    measured = np.asarray(
        [
            row["measured_peak_separation_mm"]
            if row["measured_peak_separation_mm"] is not None
            else np.nan
            for row in rows
        ]
    )
    axes[2].plot(
        separation,
        separation,
        "--",
        color="#767676",
        linewidth=0.9,
        label="Identity",
    )
    axes[2].plot(
        separation,
        measured,
        "o-",
        color="#0F4D92",
        linewidth=1.1,
        markersize=3.5,
        label="Measured peaks",
    )
    axes[2].set_xlabel("Target center separation (mm)")
    axes[2].set_ylabel("Measured peak separation (mm)")
    axes[2].legend(fontsize=5.8)
    for label, ax in zip("abc", axes):
        ax.text(
            -0.18,
            1.04,
            label,
            transform=ax.transAxes,
            fontweight="bold",
            fontsize=9,
        )
    fig.tight_layout(pad=1.1, w_pad=1.7)
    figures = _save_figure(fig, out_dir / "double_rod_separation_series")
    plt.close(fig)

    report = {
        "schema_version": "1.0",
        "rows": rows,
        "smallest_operationally_resolved_separation": smallest_resolved,
        "smallest_bandwidth_robust_resolved_separation": (
            smallest_bandwidth_robust
        ),
        "source_csv": str(csv_path.resolve()),
        "figures": figures,
        "criterion": (
            "Operationally resolved when valley/lower-peak <= 0.735. "
            "The continuous ratio and measured peak separation are retained."
        ),
        "interpretation_guardrail": (
            "The threshold is an explicit operational criterion rather than a "
            "universal law. The bandwidth-robust result requires the same "
            "decision at 0.75x, 1.0x and 1.25x nominal KDE bandwidth. Final "
            "claims must additionally be stable across independent seeds."
        ),
    }
    with (out_dir / "double_rod_separation_series.json").open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    return report


def summarize_edge_series(
    *,
    metrics_paths: Iterable[str | Path],
    out_dir: str | Path,
) -> dict[str, Any]:
    paths = [Path(path) for path in metrics_paths]
    if len(paths) < 2:
        raise ValueError("edge summary requires at least two depths")
    reports = [_load_json(path) for path in paths]
    if any(report.get("kind") != "edge" for report in reports):
        raise ValueError("all edge inputs must have kind='edge'")

    rows: list[dict[str, Any]] = []
    curves: list[dict[str, Any]] = []
    for path, report in zip(paths, reports):
        model = report["model_fit"]
        bootstrap = model["bootstrap"]
        mtf50_ci = bootstrap["mtf50_95_ci_cycles_per_mm"]
        mtf10_ci = bootstrap["mtf10_95_ci_cycles_per_mm"]
        rows.append(
            {
                "target_depth_mm": float(report["target_depth_mm"]),
                "n_edge_selected_valid": int(
                    report["edge_selected_valid"]
                ),
                "n_reference_selected_valid": int(
                    report["reference_selected_valid"]
                ),
                "edge_center_mm": float(model["edge_center_mm"]),
                "gaussian_sigma_mm": float(model["gaussian_sigma_mm"]),
                "mtf50_cycles_per_mm": float(
                    report["mtf50_cycles_per_mm"]
                ),
                "mtf50_ci_low_cycles_per_mm": float(mtf50_ci[0]),
                "mtf50_ci_high_cycles_per_mm": float(mtf50_ci[1]),
                "mtf10_cycles_per_mm": float(
                    report["mtf10_cycles_per_mm"]
                ),
                "mtf10_ci_low_cycles_per_mm": float(mtf10_ci[0]),
                "mtf10_ci_high_cycles_per_mm": float(mtf10_ci[1]),
                "deviance_per_dof": float(
                    model["fit_diagnostics"]["deviance_per_dof"]
                ),
                "bootstrap_successful": int(bootstrap["successful"]),
                "bootstrap_requested": int(bootstrap["requested"]),
                "metrics_path": str(path.resolve()),
            }
        )
        with Path(report["source_data"][0]).open(
            "r",
            encoding="utf-8-sig",
            newline="",
        ) as handle:
            edge_rows = list(csv.DictReader(handle))
        with Path(report["source_data"][1]).open(
            "r",
            encoding="utf-8-sig",
            newline="",
        ) as handle:
            mtf_rows = list(csv.DictReader(handle))
        curves.append(
            {
                "depth": float(report["target_depth_mm"]),
                "x": np.asarray(
                    [float(row["x_mm"]) for row in edge_rows]
                ),
                "esf": np.asarray(
                    [float(row["fitted_erf_esf"]) for row in edge_rows]
                ),
                "frequency": np.asarray(
                    [
                        float(row["spatial_frequency_cycles_per_mm"])
                        for row in mtf_rows
                    ]
                ),
                "mtf": np.asarray(
                    [float(row["model_mtf"]) for row in mtf_rows]
                ),
                "mtf_low": np.asarray(
                    [float(row["bootstrap_95_ci_low"]) for row in mtf_rows]
                ),
                "mtf_high": np.asarray(
                    [float(row["bootstrap_95_ci_high"]) for row in mtf_rows]
                ),
            }
        )

    order = np.argsort([row["target_depth_mm"] for row in rows])
    rows = [rows[int(index)] for index in order]
    curves = [curves[int(index)] for index in order]
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "edge_depth_series.csv"
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
    colors = plt.cm.viridis(np.linspace(0.15, 0.8, len(curves)))
    for curve, color in zip(curves, colors):
        label = f"{curve['depth']:g} mm"
        axes[0].plot(
            curve["x"],
            curve["esf"],
            color=color,
            linewidth=1.3,
            label=label,
        )
        axes[1].fill_between(
            curve["frequency"],
            curve["mtf_low"],
            curve["mtf_high"],
            color=color,
            alpha=0.13,
            linewidth=0.0,
        )
        axes[1].plot(
            curve["frequency"],
            curve["mtf"],
            color=color,
            linewidth=1.3,
            label=label,
        )
    axes[0].axvline(
        0.0,
        color="#767676",
        linestyle="--",
        linewidth=0.8,
    )
    axes[0].set_xlabel("Reconstructed x (mm)")
    axes[0].set_ylabel("Fitted edge-spread response")
    axes[0].legend(title="Target depth", fontsize=5.8, title_fontsize=5.8)

    axes[1].axhline(
        0.5,
        color="#767676",
        linestyle="--",
        linewidth=0.8,
    )
    axes[1].axhline(
        0.1,
        color="#767676",
        linestyle=":",
        linewidth=0.8,
    )
    axes[1].set_xlim(0.0, 0.16)
    axes[1].set_ylim(0.0, 1.05)
    axes[1].set_xlabel("Spatial frequency (cycles/mm)")
    axes[1].set_ylabel("Model MTF")
    axes[1].legend(title="Target depth", fontsize=5.8, title_fontsize=5.8)

    depth = np.asarray([row["target_depth_mm"] for row in rows])
    for level, marker, color in (
        ("mtf50", "o", "#0F4D92"),
        ("mtf10", "s", "#B64342"),
    ):
        values = np.asarray(
            [row[f"{level}_cycles_per_mm"] for row in rows]
        )
        lower = values - np.asarray(
            [row[f"{level}_ci_low_cycles_per_mm"] for row in rows]
        )
        upper = np.asarray(
            [row[f"{level}_ci_high_cycles_per_mm"] for row in rows]
        ) - values
        axes[2].errorbar(
            depth,
            values,
            yerr=np.asarray([lower, upper]),
            fmt=f"{marker}-",
            color=color,
            capsize=3,
            linewidth=1.1,
            markersize=3.5,
            label=level.upper(),
        )
    axes[2].set_xlabel("Target depth (mm)")
    axes[2].set_ylabel("Spatial frequency (cycles/mm)")
    axes[2].legend(fontsize=5.8)
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
    figures = _save_figure(fig, out_dir / "edge_depth_series")
    plt.close(fig)

    shallow = rows[0]
    deep = rows[-1]
    report = {
        "schema_version": "1.0",
        "kind": "edge_depth_series",
        "rows": rows,
        "deep_to_shallow_mtf50_ratio": float(
            deep["mtf50_cycles_per_mm"]
            / shallow["mtf50_cycles_per_mm"]
        ),
        "deep_to_shallow_mtf10_ratio": float(
            deep["mtf10_cycles_per_mm"]
            / shallow["mtf10_cycles_per_mm"]
        ),
        "deep_to_shallow_sigma_ratio": float(
            deep["gaussian_sigma_mm"]
            / shallow["gaussian_sigma_mm"]
        ),
        "source_csv": str(csv_path.resolve()),
        "figures": figures,
        "interpretation_guardrail": (
            "The MTF values come from a count-aware error-function ESF / "
            "Gaussian LSF model with conditional bootstrap intervals. They "
            "are one-seed intrinsic-geometric simulation results, not final "
            "hardware performance claims."
        ),
    }
    with (out_dir / "edge_depth_series.json").open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    return report
