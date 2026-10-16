from __future__ import annotations

import argparse
import copy
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backscatter.hardware_figures import (
    materialize_response_analysis_view,
    prepare_hardware_response_analysis,
    resolve_hardware_profile_for_variant,
)
from backscatter.config import load_config
from backscatter.kinematics import solve_scatter_points
from backscatter.pipeline import build_selection
from backscatter.response import apply_detector_response
from backscatter.spatial import summarize_rod_series
from backscatter.standards import estimate_fwhm_kde, evaluate_standard_target
from backscatter.sweeps import run_sweeps


ROD_RUNS = {
    1: "rod_width1_depth30_5m_seed24061873_local_manual_retry2",
    2: "rod_width2_depth30_5m_seed24061873_local_manual_retry2",
    5: "rod_width5_depth30_5m_seed24061873_local_manual_retry2",
}


def _as_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if np.isfinite(result) else None


def _load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {name: data[name] for name in data.files}


def _positions(table: dict[str, np.ndarray], prefix: str) -> np.ndarray:
    return np.column_stack(
        [
            table[f"{prefix}_x_mm"],
            table[f"{prefix}_y_mm"],
            table[f"{prefix}_z_mm"],
        ]
    ).astype(np.float64)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _plot_requirement(
    *,
    rod_rows: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
    target_fwhm_mm: float,
    output: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "Arial",
        "DejaVu Sans",
    ]
    plt.rcParams["svg.fonttype"] = "none"
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["font.size"] = 8.5
    plt.rcParams["axes.spines.right"] = False
    plt.rcParams["axes.spines.top"] = False

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.0))
    widths = np.asarray([row["rod_diameter_mm"] for row in rod_rows])
    fwhm = np.asarray([row["transverse_x_fwhm_mm"] for row in rod_rows])
    lower = np.asarray(
        [
            row["transverse_x_fwhm_mm"]
            - row["transverse_x_fwhm_ci_low_mm"]
            for row in rod_rows
        ]
    )
    upper = np.asarray(
        [
            row["transverse_x_fwhm_ci_high_mm"]
            - row["transverse_x_fwhm_mm"]
            for row in rod_rows
        ]
    )
    axes[0].errorbar(
        widths,
        fwhm,
        yerr=np.asarray([lower, upper]),
        fmt="o-",
        color="#165A9F",
        capsize=4,
        label="Hardware-response result",
    )
    axes[0].axhline(
        target_fwhm_mm,
        color="#C2413B",
        linestyle="--",
        linewidth=1.2,
        label=f"Requirement: {target_fwhm_mm:g} mm",
    )
    axes[0].set_xlabel("Concrete rod diameter (mm)")
    axes[0].set_ylabel("Transverse x FWHM (mm)")
    axes[0].set_xticks(widths)
    axes[0].legend(fontsize=7.5)
    axes[0].set_title("Updated detector response")

    plotted = [row for row in candidate_rows if row["fwhm_mm"] is not None]
    colors = ["#165A9F" if row["eligible"] else "#AAB7C4" for row in plotted]
    axes[1].scatter(
        [row["count_retention"] * 100.0 for row in plotted],
        [row["fwhm_mm"] for row in plotted],
        c=colors,
        s=26,
        alpha=0.82,
    )
    axes[1].axhline(
        target_fwhm_mm,
        color="#C2413B",
        linestyle="--",
        linewidth=1.2,
    )
    axes[1].axvline(50.0, color="#767676", linestyle=":", linewidth=1.0)
    axes[1].set_xlabel("Valid-count retention vs baseline (%)")
    axes[1].set_ylabel("Transverse x FWHM (mm)")
    axes[1].set_title("Selection-quality trade-off (2 mm rod)")
    axes[1].text(
        0.02,
        0.03,
        "Blue: meets count, purity, bias and FWHM gates",
        transform=axes[1].transAxes,
        fontsize=7.2,
        color="#165A9F",
    )
    for label, axis in zip("ab", axes):
        axis.text(
            -0.14,
            1.04,
            label,
            transform=axis.transAxes,
            fontsize=11,
            fontweight="bold",
        )
    fig.tight_layout(pad=1.2, w_pad=2.4)
    for suffix, dpi in ((".png", 300), (".svg", 300), (".pdf", 300)):
        fig.savefig(output.with_suffix(suffix), dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _response_seed_stability(
    *,
    analysis_dir: Path,
    config_path: Path,
    seeds: list[int],
    target_fwhm_mm: float,
    output_dir: Path,
) -> dict[str, Any]:
    pairs = _load_npz(analysis_dir / "pairs.npz")
    config = load_config(config_path)
    observable_config = copy.deepcopy(config)
    observable_config["selection"]["phantom_truth_policy"] = "ignore"
    path_lo, path_hi = map(
        float, observable_config["reconstruction"]["path_length_mm"]
    )
    rows: list[dict[str, Any]] = []
    for seed in seeds:
        measured, _ = apply_detector_response(
            pairs,
            observable_config,
            random_seed=seed,
        )
        selected, _, _ = build_selection(measured, observable_config)
        solution = solve_scatter_points(
            _positions(measured, "source"),
            _positions(measured, "d1"),
            _positions(measured, "d2"),
            measured["d1_energy_keV"],
            incident_energy_keV=float(
                observable_config["simulation"]["incident_energy_keV"]
            ),
            electron_rest_energy_keV=float(
                observable_config["simulation"]["electron_rest_energy_keV"]
            ),
            path_min_mm=path_lo,
            path_max_mm=path_hi,
        )
        usable = selected & solution.valid
        estimate = estimate_fwhm_kde(solution.points_mm[usable, 0])
        rows.append(
            {
                "response_seed": seed,
                "n_selected_valid": int(usable.sum()),
                "transverse_x_fwhm_mm": estimate.get("fwhm"),
                "peak_bias_mm": estimate.get("mode"),
                "status": estimate["status"],
                "meets_5mm_target": bool(
                    estimate.get("fwhm") is not None
                    and float(estimate["fwhm"]) <= target_fwhm_mm
                ),
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "response_seed_stability.csv", rows)
    valid = [
        row
        for row in rows
        if row["status"] == "ok"
        and row["transverse_x_fwhm_mm"] is not None
    ]
    fwhm = np.asarray(
        [row["transverse_x_fwhm_mm"] for row in valid], dtype=np.float64
    )
    counts = np.asarray(
        [row["n_selected_valid"] for row in valid], dtype=np.int64
    )

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "Arial",
        "DejaVu Sans",
    ]
    plt.rcParams["svg.fonttype"] = "none"
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["font.size"] = 8.5
    plt.rcParams["axes.spines.right"] = False
    plt.rcParams["axes.spines.top"] = False
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.7))
    axes[0].plot(
        np.arange(1, len(valid) + 1),
        fwhm,
        "o-",
        color="#165A9F",
        linewidth=1.0,
        markersize=4,
    )
    axes[0].axhline(
        target_fwhm_mm,
        color="#C2413B",
        linestyle="--",
        linewidth=1.2,
        label=f"Requirement: {target_fwhm_mm:g} mm",
    )
    axes[0].set_xlabel("Detector-response random seed index")
    axes[0].set_ylabel("Transverse x FWHM (mm)")
    axes[0].legend(fontsize=7.5)
    axes[1].scatter(counts, fwhm, color="#0E7490", s=28, alpha=0.85)
    axes[1].axhline(
        target_fwhm_mm,
        color="#C2413B",
        linestyle="--",
        linewidth=1.2,
    )
    axes[1].set_xlabel("Selected valid events")
    axes[1].set_ylabel("Transverse x FWHM (mm)")
    axes[1].text(
        0.03,
        0.04,
        f"median={np.median(fwhm):.2f} mm\n"
        f"range={np.min(fwhm):.2f}–{np.max(fwhm):.2f} mm",
        transform=axes[1].transAxes,
        fontsize=7.5,
        color="#1F2937",
    )
    for label, axis in zip("ab", axes):
        axis.text(
            -0.14,
            1.04,
            label,
            transform=axis.transAxes,
            fontsize=11,
            fontweight="bold",
        )
    fig.tight_layout(pad=1.2, w_pad=2.4)
    figure_base = output_dir / "response_seed_stability"
    for suffix, dpi in ((".png", 300), (".svg", 300), (".pdf", 300)):
        fig.savefig(figure_base.with_suffix(suffix), dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    return {
        "n_requested": len(seeds),
        "n_valid": len(valid),
        "median_fwhm_mm": float(np.median(fwhm)),
        "fwhm_95_interval_across_response_seeds_mm": np.quantile(
            fwhm, [0.025, 0.975]
        ).tolist(),
        "minimum_fwhm_mm": float(np.min(fwhm)),
        "maximum_fwhm_mm": float(np.max(fwhm)),
        "all_response_seeds_meet_target": bool(
            np.all(fwhm <= target_fwhm_mm)
        ),
        "selected_valid_event_range": [int(np.min(counts)), int(np.max(counts))],
        "csv": str((output_dir / "response_seed_stability.csv").resolve()),
        "figure_png": str(figure_base.with_suffix(".png").resolve()),
        "guardrail": (
            "This varies only the parameterized detector-response smearing; "
            "it does not replace an independent GATE transport seed."
        ),
    }


def _truth_signal_mask(
    pairs: dict[str, np.ndarray], config: dict[str, Any]
) -> np.ndarray:
    volumes = np.char.lower(pairs["d1_compt_volume"].astype(str))
    volume_name = str(
        config["selection"].get(
            "phantom_truth_volume_contains",
            config["phantom"].get("gate_volume", config["phantom"]["name"]),
        )
    ).lower()
    return (
        (pairs["d1_n_phantom_compton"] == 1)
        & (pairs["d2_n_phantom_compton"] == 0)
        & (np.char.find(volumes, volume_name) >= 0)
    )


def _robust_selection_study(
    *,
    analysis_dir: Path,
    config_path: Path,
    seeds: list[int],
    target_fwhm_mm: float,
    output_dir: Path,
) -> dict[str, Any]:
    pairs = _load_npz(analysis_dir / "pairs.npz")
    base_config = load_config(config_path)
    base_config["selection"]["phantom_truth_policy"] = "ignore"
    path_lo, path_hi = map(
        float, base_config["reconstruction"]["path_length_mm"]
    )
    truth_signal = _truth_signal_mask(pairs, base_config)
    settings: list[dict[str, Any]] = [
        {"id": "baseline", "label": "Baseline", "changes": {}},
        {
            "id": "d1_180_250",
            "label": "D1 180–250 keV",
            "changes": {"d1_energy_window_keV": [180.0, 250.0]},
        },
        {
            "id": "d1_190_240",
            "label": "D1 190–240 keV",
            "changes": {"d1_energy_window_keV": [190.0, 240.0]},
        },
        {
            "id": "d2_450_560",
            "label": "D2 450–560 keV",
            "changes": {"d2_energy_window_keV": [450.0, 560.0]},
        },
        {
            "id": "scatter_100_180",
            "label": "Scatter 100–180°",
            "changes": {"scatter_angle_deg": [100.0, 180.0]},
        },
        {
            "id": "incident_0_30",
            "label": "Incident 0–30°",
            "changes": {"incident_angle_deg": [0.0, 30.0]},
        },
        {
            "id": "d1_radius_150",
            "label": "D1 radius 20–150 mm",
            "changes": {"d1_radius_mm": [20.0, 150.0]},
        },
        {
            "id": "d1_180_250_d2_450_560",
            "label": "D1 180–250 + D2 450–560",
            "changes": {
                "d1_energy_window_keV": [180.0, 250.0],
                "d2_energy_window_keV": [450.0, 560.0],
            },
        },
        {
            "id": "d1_180_250_incident_0_30",
            "label": "D1 180–250 + incident 0–30°",
            "changes": {
                "d1_energy_window_keV": [180.0, 250.0],
                "incident_angle_deg": [0.0, 30.0],
            },
        },
    ]

    detail_rows: list[dict[str, Any]] = []
    for seed in seeds:
        measured, _ = apply_detector_response(
            pairs,
            base_config,
            random_seed=seed,
        )
        solution = solve_scatter_points(
            _positions(measured, "source"),
            _positions(measured, "d1"),
            _positions(measured, "d2"),
            measured["d1_energy_keV"],
            incident_energy_keV=float(
                base_config["simulation"]["incident_energy_keV"]
            ),
            electron_rest_energy_keV=float(
                base_config["simulation"]["electron_rest_energy_keV"]
            ),
            path_min_mm=path_lo,
            path_max_mm=path_hi,
        )
        seed_rows: list[dict[str, Any]] = []
        for setting in settings:
            config = copy.deepcopy(base_config)
            config["selection"].update(setting["changes"])
            selected, _, _ = build_selection(measured, config)
            usable = selected & solution.valid
            estimate = estimate_fwhm_kde(solution.points_mm[usable, 0])
            count = int(usable.sum())
            truth_count = int((usable & truth_signal).sum())
            seed_rows.append(
                {
                    "response_seed": seed,
                    "setting_id": setting["id"],
                    "setting_label": setting["label"],
                    "n_selected_valid": count,
                    "truth_signal_fraction": (
                        truth_count / count if count else None
                    ),
                    "transverse_x_fwhm_mm": estimate.get("fwhm"),
                    "peak_bias_mm": estimate.get("mode"),
                    "status": estimate["status"],
                }
            )
        baseline_count = seed_rows[0]["n_selected_valid"]
        for row in seed_rows:
            row["count_retention_vs_same_seed_baseline"] = (
                row["n_selected_valid"] / baseline_count
                if baseline_count
                else None
            )
        detail_rows.extend(seed_rows)

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "robust_selection_seed_details.csv", detail_rows)
    summary_rows: list[dict[str, Any]] = []
    for setting in settings:
        subset = [
            row
            for row in detail_rows
            if row["setting_id"] == setting["id"] and row["status"] == "ok"
        ]
        fwhm = np.asarray(
            [row["transverse_x_fwhm_mm"] for row in subset],
            dtype=np.float64,
        )
        counts = np.asarray(
            [row["n_selected_valid"] for row in subset], dtype=np.int64
        )
        retention = np.asarray(
            [row["count_retention_vs_same_seed_baseline"] for row in subset],
            dtype=np.float64,
        )
        purity = np.asarray(
            [row["truth_signal_fraction"] for row in subset], dtype=np.float64
        )
        bias = np.asarray(
            [row["peak_bias_mm"] for row in subset], dtype=np.float64
        )
        q_low, q_high = np.quantile(fwhm, [0.025, 0.975])
        row = {
            "setting_id": setting["id"],
            "setting_label": setting["label"],
            "n_response_seeds": len(subset),
            "median_fwhm_mm": float(np.median(fwhm)),
            "fwhm_seed_q2p5_mm": float(q_low),
            "fwhm_seed_q97p5_mm": float(q_high),
            "maximum_fwhm_mm": float(np.max(fwhm)),
            "fraction_seeds_meeting_target": float(
                np.mean(fwhm <= target_fwhm_mm)
            ),
            "median_selected_valid": float(np.median(counts)),
            "median_count_retention": float(np.median(retention)),
            "minimum_truth_signal_fraction": float(np.min(purity)),
            "maximum_absolute_peak_bias_mm": float(np.max(np.abs(bias))),
            "robustly_eligible": bool(
                q_high <= target_fwhm_mm
                and np.median(retention) >= 0.75
                and np.min(purity) >= 0.95
                and np.max(np.abs(bias)) <= 1.0
            ),
            "selection_changes": json.dumps(
                setting["changes"], ensure_ascii=False, separators=(",", ":")
            ),
        }
        summary_rows.append(row)

    eligible = [row for row in summary_rows if row["robustly_eligible"]]
    eligible.sort(
        key=lambda row: (
            -row["median_count_retention"],
            row["fwhm_seed_q97p5_mm"],
        )
    )
    recommended = eligible[0] if eligible else None
    _write_csv(output_dir / "robust_selection_summary.csv", summary_rows)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "Arial",
        "DejaVu Sans",
    ]
    plt.rcParams["svg.fonttype"] = "none"
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["font.size"] = 8.2
    plt.rcParams["axes.spines.right"] = False
    plt.rcParams["axes.spines.top"] = False
    labels = [row["setting_label"] for row in summary_rows]
    y = np.arange(len(summary_rows))
    medians = np.asarray([row["median_fwhm_mm"] for row in summary_rows])
    q_low = np.asarray([row["fwhm_seed_q2p5_mm"] for row in summary_rows])
    q_high = np.asarray([row["fwhm_seed_q97p5_mm"] for row in summary_rows])
    retention = np.asarray(
        [row["median_count_retention"] * 100.0 for row in summary_rows]
    )
    colors = [
        "#165A9F" if row["robustly_eligible"] else "#AAB7C4"
        for row in summary_rows
    ]
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 5.1), sharey=True)
    axes[0].errorbar(
        medians,
        y,
        xerr=np.asarray([medians - q_low, q_high - medians]),
        fmt="none",
        ecolor="#7B8794",
        elinewidth=1.4,
        capsize=3,
    )
    axes[0].scatter(medians, y, c=colors, s=32, zorder=3)
    axes[0].axvline(
        target_fwhm_mm,
        color="#C2413B",
        linestyle="--",
        linewidth=1.2,
    )
    axes[0].set_xlabel("Transverse x FWHM across response seeds (mm)")
    axes[0].set_yticks(y, labels)
    axes[0].invert_yaxis()
    axes[0].set_title("Median and 2.5–97.5% interval")
    axes[1].barh(y, retention, color=colors, alpha=0.9)
    axes[1].axvline(75.0, color="#767676", linestyle=":", linewidth=1.0)
    axes[1].set_xlabel("Median valid-count retention (%)")
    axes[1].set_title("Signal retained relative to baseline")
    for label, axis in zip("ab", axes):
        axis.text(
            -0.10,
            1.03,
            label,
            transform=axis.transAxes,
            fontsize=11,
            fontweight="bold",
        )
    fig.tight_layout(pad=1.2, w_pad=2.0)
    figure_base = output_dir / "robust_selection_comparison"
    for suffix, dpi in ((".png", 300), (".svg", 300), (".pdf", 300)):
        fig.savefig(figure_base.with_suffix(suffix), dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    return {
        "n_response_seeds": len(seeds),
        "settings_tested": len(settings),
        "robustly_eligible_count": len(eligible),
        "recommended": recommended,
        "summary_rows": summary_rows,
        "summary_csv": str(
            (output_dir / "robust_selection_summary.csv").resolve()
        ),
        "detail_csv": str(
            (output_dir / "robust_selection_seed_details.csv").resolve()
        ),
        "figure_png": str(figure_base.with_suffix(".png").resolve()),
        "acceptance_rule": (
            "Across detector-response seeds: 97.5th-percentile FWHM <= target, "
            "median count retention >=75%, minimum truth-signal fraction >=95%, "
            "and maximum absolute peak bias <=1 mm."
        ),
    }


def run(
    *,
    project_root: Path,
    output_dir: Path,
    target_fwhm_mm: float,
    n_bootstrap: int,
    force: bool,
) -> dict[str, Any]:
    hardware = project_root / "configs" / "hardware_20260731.json"
    resolved_configs = output_dir / "resolved_configs"
    response_root = output_dir / "response_variants"
    figures = output_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    metric_paths: list[Path] = []
    response_reports: list[dict[str, Any]] = []
    response_dirs: dict[int, Path] = {}
    resolved_by_width: dict[int, Path] = {}

    for index, (width, run_name) in enumerate(ROD_RUNS.items()):
        analysis_dir = project_root / "runs" / run_name
        base_config = (
            project_root
            / "configs"
            / "generated"
            / f"rod_width{width}_depth30_5m.json"
        )
        tag = f"rod_width{width}_depth30_hardware_20260731"
        resolved = resolve_hardware_profile_for_variant(
            base_config_path=base_config,
            hardware_profile_path=hardware,
            output_path=resolved_configs / f"{tag}.json",
        )
        response_dir = response_root / tag
        report_path = response_dir / "response_analysis.json"
        if force or not report_path.exists():
            response_report = prepare_hardware_response_analysis(
                analysis_dir=analysis_dir,
                config_path=resolved,
                out_dir=response_dir,
                random_seed=24080510 + index,
            )
        else:
            with report_path.open("r", encoding="utf-8") as handle:
                response_report = json.load(handle)
        materialize_response_analysis_view(
            source_analysis_dir=analysis_dir,
            response_dir=response_dir,
        )
        metrics_dir = response_dir / "standard_metrics"
        metrics_path = metrics_dir / "metrics.json"
        if force or not metrics_path.exists():
            evaluate_standard_target(
                analysis_dir=response_dir,
                config_path=resolved,
                out_dir=metrics_dir,
                n_bootstrap=n_bootstrap,
            )
        metric_paths.append(metrics_path)
        response_reports.append(response_report)
        response_dirs[width] = response_dir
        resolved_by_width[width] = resolved

    rod_summary = summarize_rod_series(
        metrics_paths=metric_paths,
        out_dir=figures / "rod_width_series",
    )

    sweep_dir = response_dirs[2] / "sweeps"
    run_sweeps(
        analysis_dir=response_dirs[2],
        config_path=resolved_by_width[2],
        out_dir=sweep_dir,
    )
    sweep_rows = _load_csv(sweep_dir / "sweep_metrics.csv")
    baseline = sweep_rows[0]
    baseline_count = int(baseline["n_selected_valid"])
    min_count = max(150, int(np.ceil(0.50 * baseline_count)))

    candidates: list[dict[str, Any]] = []
    for row in sweep_rows:
        fwhm = _as_float(row.get("transverse_psf_fwhm_x_mm"))
        bias = _as_float(row.get("transverse_psf_peak_bias_x_mm"))
        purity = _as_float(row.get("truth_signal_fraction"))
        count = int(row["n_selected_valid"])
        retention = count / baseline_count if baseline_count else 0.0
        eligible = bool(
            fwhm is not None
            and fwhm <= target_fwhm_mm
            and count >= min_count
            and purity is not None
            and purity >= 0.95
            and bias is not None
            and abs(bias) <= 1.0
        )
        candidates.append(
            {
                "category": row["category"],
                "setting": row["setting"],
                "n_selected_valid": count,
                "count_retention": retention,
                "truth_signal_fraction": purity,
                "fwhm_mm": fwhm,
                "peak_bias_mm": bias,
                "eligible": eligible,
            }
        )

    eligible_candidates = [row for row in candidates if row["eligible"]]
    eligible_candidates.sort(
        key=lambda row: (-row["n_selected_valid"], row["fwhm_mm"])
    )
    chosen = eligible_candidates[0] if eligible_candidates else None

    seed_stability = _response_seed_stability(
        analysis_dir=(project_root / "runs" / ROD_RUNS[2]),
        config_path=resolved_by_width[2],
        seeds=list(range(24080511, 24080536)),
        target_fwhm_mm=target_fwhm_mm,
        output_dir=figures / "response_seed_stability",
    )
    robust_selection = _robust_selection_study(
        analysis_dir=(project_root / "runs" / ROD_RUNS[2]),
        config_path=resolved_by_width[2],
        seeds=list(range(24080511, 24080536)),
        target_fwhm_mm=target_fwhm_mm,
        output_dir=figures / "robust_selection",
    )

    _write_csv(output_dir / "selection_candidates.csv", candidates)
    _plot_requirement(
        rod_rows=rod_summary["rows"],
        candidate_rows=candidates,
        target_fwhm_mm=target_fwhm_mm,
        output=figures / "transverse_fwhm_requirement",
    )

    result = {
        "schema_version": "1.0",
        "objective": (
            f"Demonstrate transverse x FWHM near or below {target_fwhm_mm:g} mm "
            "after applying the updated D1/D2 detector response."
        ),
        "measurement_definition": {
            "standard_target": "concrete rod parallel to y at 30 mm depth",
            "primary_metric": "KDE line-spread FWHM of reconstructed x",
            "target_fwhm_mm": target_fwhm_mm,
            "guardrails": {
                "minimum_valid_events": min_count,
                "minimum_truth_signal_fraction": 0.95,
                "maximum_absolute_peak_bias_mm": 1.0,
                "minimum_count_retention_fraction": 0.50,
            },
        },
        "hardware_profile": str(hardware.resolve()),
        "rod_summary": rod_summary,
        "response_reports": response_reports,
        "candidate_count": len(candidates),
        "eligible_candidate_count": len(eligible_candidates),
        "recommended_selection": chosen,
        "response_seed_stability": seed_stability,
        "robust_selection": robust_selection,
        "requirement_met_on_2mm_rod": bool(
            chosen is not None
            and any(
                row["rod_diameter_mm"] == 2.0
                and row["transverse_x_fwhm_mm"] <= target_fwhm_mm
                and row["transverse_x_fwhm_ci_high_mm"] <= target_fwhm_mm
                for row in rod_summary["rows"]
            )
        ),
        "interpretation_guardrail": (
            "This is a deterministic post-processing application of the supplied "
            "energy and position responses to existing GATE events. Final acceptance "
            "requires an independent GATE random seed using the frozen selection."
        ),
        "outputs": {
            "candidate_csv": str((output_dir / "selection_candidates.csv").resolve()),
            "figure_png": str(
                (figures / "transverse_fwhm_requirement.png").resolve()
            ),
            "sweep_csv": str((sweep_dir / "sweep_metrics.csv").resolve()),
        },
    }
    with (output_dir / "transverse_fwhm_optimization.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate and optimize the updated-hardware transverse FWHM."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "transverse_fwhm_5mm_20260805",
    )
    parser.add_argument("--target-fwhm-mm", type=float, default=5.0)
    parser.add_argument("--bootstrap", type=int, default=300)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    result = run(
        project_root=PROJECT_ROOT,
        output_dir=args.output_dir,
        target_fwhm_mm=args.target_fwhm_mm,
        n_bootstrap=args.bootstrap,
        force=args.force,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
