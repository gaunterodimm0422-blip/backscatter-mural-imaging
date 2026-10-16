from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backscatter.config import load_config
from backscatter.hardware_figures import (
    materialize_response_analysis_view,
    prepare_hardware_response_analysis,
)
from backscatter.kinematics import solve_scatter_points
from backscatter.pipeline import build_selection
from backscatter.response import apply_detector_response
from backscatter.standards import estimate_fwhm_kde, evaluate_standard_target


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


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _evaluate_response_seed(
    *,
    pairs: dict[str, np.ndarray],
    config: dict[str, Any],
    truth_signal: np.ndarray,
    response_seed: int,
) -> dict[str, Any]:
    measured, _ = apply_detector_response(
        pairs,
        config,
        random_seed=response_seed,
    )
    selected, selection_chain, _ = build_selection(measured, config)
    path_lo, path_hi = map(float, config["reconstruction"]["path_length_mm"])
    solution = solve_scatter_points(
        _positions(measured, "source"),
        _positions(measured, "d1"),
        _positions(measured, "d2"),
        measured["d1_energy_keV"],
        incident_energy_keV=float(config["simulation"]["incident_energy_keV"]),
        electron_rest_energy_keV=float(
            config["simulation"]["electron_rest_energy_keV"]
        ),
        path_min_mm=path_lo,
        path_max_mm=path_hi,
    )
    usable = selected & solution.valid
    estimate = estimate_fwhm_kde(solution.points_mm[usable, 0])
    n_usable = int(usable.sum())
    n_truth_signal = int((usable & truth_signal).sum())
    return {
        "response_seed": int(response_seed),
        "status": estimate["status"],
        "transverse_x_fwhm_mm": estimate.get("fwhm"),
        "peak_bias_mm": estimate.get("mode"),
        "n_selected_valid": n_usable,
        "truth_signal_count": n_truth_signal,
        "truth_signal_fraction": (
            float(n_truth_signal / n_usable) if n_usable else None
        ),
        "n_after_d2_energy": int(selection_chain["d2_energy_window"]),
        "n_after_d1_energy": int(selection_chain["d1_energy_window"]),
    }


def _plot_validation(
    *,
    rows: list[dict[str, Any]],
    target_fwhm_mm: float,
    minimum_events: int,
    minimum_purity: float,
    output_base: Path,
) -> list[str]:
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

    fwhm = np.asarray([row["transverse_x_fwhm_mm"] for row in rows])
    counts = np.asarray([row["n_selected_valid"] for row in rows])
    purity = np.asarray([row["truth_signal_fraction"] for row in rows])
    index = np.arange(1, len(rows) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.0))
    axes[0].plot(index, fwhm, "o-", color="#165A9F", markersize=4)
    axes[0].axhline(
        target_fwhm_mm,
        color="#C2413B",
        linestyle="--",
        linewidth=1.2,
        label=f"Requirement: {target_fwhm_mm:g} mm",
    )
    axes[0].set_xlabel("Detector-response random-seed index")
    axes[0].set_ylabel("Transverse x FWHM (mm)")
    axes[0].set_title("Independent GATE transport seed")
    axes[0].legend(fontsize=7.5)

    scatter = axes[1].scatter(
        counts,
        fwhm,
        c=100.0 * purity,
        cmap="viridis",
        vmin=100.0 * minimum_purity,
        vmax=100.0,
        s=34,
        edgecolors="white",
        linewidths=0.35,
    )
    axes[1].axhline(
        target_fwhm_mm,
        color="#C2413B",
        linestyle="--",
        linewidth=1.2,
    )
    axes[1].axvline(
        minimum_events,
        color="#767676",
        linestyle=":",
        linewidth=1.0,
    )
    axes[1].set_xlabel("Selected valid events")
    axes[1].set_ylabel("Transverse x FWHM (mm)")
    axes[1].set_title("Resolution, count and truth-purity guardrails")
    colorbar = fig.colorbar(scatter, ax=axes[1], pad=0.02)
    colorbar.set_label("Truth-signal fraction (%)")

    for label, axis in zip("ab", axes):
        axis.text(
            -0.13,
            1.04,
            label,
            transform=axis.transAxes,
            fontsize=11,
            fontweight="bold",
        )
    fig.tight_layout(pad=1.2, w_pad=2.2)
    output_base.parent.mkdir(parents=True, exist_ok=True)
    outputs: list[str] = []
    for suffix, dpi in ((".png", 300), (".svg", 300), (".pdf", 300)):
        path = output_base.with_suffix(suffix)
        fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
        outputs.append(str(path.resolve()))
    plt.close(fig)
    return outputs


def _load_reference_result(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    with path.open("r", encoding="utf-8") as handle:
        report = json.load(handle)
    recommended = report.get("robust_selection", {}).get("recommended")
    if not recommended:
        return None
    return {
        "gate_seed": 24061873,
        "setting_id": recommended["setting_id"],
        "median_fwhm_mm": recommended["median_fwhm_mm"],
        "fwhm_seed_q2p5_mm": recommended["fwhm_seed_q2p5_mm"],
        "fwhm_seed_q97p5_mm": recommended["fwhm_seed_q97p5_mm"],
        "maximum_fwhm_mm": recommended["maximum_fwhm_mm"],
        "median_selected_valid": recommended["median_selected_valid"],
        "median_count_retention": recommended["median_count_retention"],
        "minimum_truth_signal_fraction": recommended[
            "minimum_truth_signal_fraction"
        ],
    }


def _write_markdown_report(path: Path, report: dict[str, Any]) -> None:
    summary = report["independent_seed_summary"]
    canonical = report["canonical_result"]
    checks = report["acceptance_checks"]
    result_text = "通过" if report["passed"] else "未通过"
    ci = canonical["bootstrap_fwhm_95_ci_mm"]
    lines = [
        "# 横向 FWHM 5 mm 独立验证",
        "",
        f"## 结论：{result_text}",
        "",
        (
            f"本轮使用独立 GATE 随机种子 `{report['gate_seed']}`，模拟 "
            f"{report['n_primaries']:,} 个初级光子。标准目标为 30 mm 深处、"
            "直径 2 mm、沿 y 方向放置的水泥杆。"
        ),
        "",
        (
            f"固定筛选条件为 D1 `{report['selection']['d1_energy_window_keV'][0]:g}–"
            f"{report['selection']['d1_energy_window_keV'][1]:g} keV`、D2 "
            f"`{report['selection']['d2_energy_window_keV'][0]:g}–"
            f"{report['selection']['d2_energy_window_keV'][1]:g} keV`。"
        ),
        "",
        (
            f"25 组探测器响应随机展宽下，横向 FWHM 中位数为 "
            f"**{summary['median_fwhm_mm']:.3f} mm**，范围为 "
            f"{summary['minimum_fwhm_mm']:.3f}–{summary['maximum_fwhm_mm']:.3f} mm；"
            f"25 组中有 {summary['n_meeting_target']}/{summary['n_valid']} 组不超过 5 mm。"
        ),
        "",
        (
            f"用于正式画响应曲线的固定电子学种子结果为 "
            f"**{canonical['fwhm_mm']:.3f} mm**，95% bootstrap 区间为 "
            f"**[{ci[0]:.3f}, {ci[1]:.3f}] mm**。"
            if ci is not None
            else (
                f"用于正式画响应曲线的固定电子学种子结果为 "
                f"**{canonical['fwhm_mm']:.3f} mm**；bootstrap 区间不可用。"
            )
        ),
        "",
        "## 验收条件",
        "",
        "| 条件 | 本轮结果 | 判定 |",
        "|---|---:|:---:|",
        (
            f"| 独立 GATE 种子与 500 万事件身份正确 | "
            f"seed={report['gate_seed']}, n={report['n_primaries']:,} | "
            f"{'通过' if checks['transport_identity'] else '未通过'} |"
        ),
        (
            f"| 所有响应种子 FWHM ≤ 5 mm | "
            f"max={summary['maximum_fwhm_mm']:.3f} mm | "
            f"{'通过' if checks['all_response_seeds_below_target'] else '未通过'} |"
        ),
        (
            f"| 固定种子的 FWHM 95% 区间上限 ≤ 5 mm | "
            f"{ci[1]:.3f} mm | "
            f"{'通过' if checks['canonical_bootstrap_upper_below_target'] else '未通过'} |"
            if ci is not None
            else (
                "| 固定种子的 FWHM 95% 区间上限 ≤ 5 mm | 无区间 | 未通过 |"
            )
        ),
        (
            f"| 有效事件中位数 ≥ {report['guardrails']['minimum_selected_valid_events']} | "
            f"{summary['median_selected_valid']:.0f} | "
            f"{'通过' if checks['enough_events'] else '未通过'} |"
        ),
        (
            f"| 最低真值信号比例 ≥ {100.0 * report['guardrails']['minimum_truth_signal_fraction']:.0f}% | "
            f"{100.0 * summary['minimum_truth_signal_fraction']:.2f}% | "
            f"{'通过' if checks['truth_purity'] else '未通过'} |"
        ),
        (
            f"| 最大峰位置偏差 ≤ {report['guardrails']['maximum_absolute_peak_bias_mm']:.1f} mm | "
            f"{summary['maximum_absolute_peak_bias_mm']:.3f} mm | "
            f"{'通过' if checks['peak_bias'] else '未通过'} |"
        ),
        "",
        "## 结果应如何解释",
        "",
        (
            "这里的横向 FWHM 是重建点在 x 方向形成的一维线扩展函数宽度。"
            "它包含 2 mm 标准杆本身的有限宽度，因此是系统对该标准目标的实测响应宽度，"
            "没有做缺乏依据的反卷积。"
        ),
        "",
        (
            "D1 能窗从原始的 170.33–255.5 keV 温和收窄到 180–250 keV，"
            "目的是去掉位于单次背散射能带边缘、对位置重建更不稳定的事件。"
            "验收同时限制事件数和真值纯度，所以结果不是靠只留下极少数事件得到的。"
        ),
        "",
        "## 文件",
        "",
        f"- 冻结配置：`{report['config']}`",
        f"- 逐种子数据：`{report['outputs']['seed_csv']}`",
        f"- 验证图：`{report['outputs']['figure_png']}`",
        f"- 标准杆响应图：`{canonical['figure_png']}`",
        f"- 机器可读报告：`{report['outputs']['json']}`",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def run(
    *,
    analysis_dir: Path,
    config_path: Path,
    output_dir: Path,
    gate_seed: int,
    response_seed_start: int,
    response_seed_count: int,
    n_bootstrap: int,
    skip_transport_identity_check: bool,
) -> dict[str, Any]:
    config = load_config(config_path)
    protocol = config.get("validation_protocol", {})
    target_fwhm_mm = float(protocol.get("target_fwhm_mm", 5.0))
    minimum_events = int(protocol.get("minimum_selected_valid_events", 150))
    minimum_purity = float(protocol.get("minimum_truth_signal_fraction", 0.95))
    maximum_bias = float(protocol.get("maximum_absolute_peak_bias_mm", 1.0))

    required = [
        analysis_dir / "pairs.npz",
        analysis_dir / "reconstruction.npz",
        analysis_dir / "summary.json",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing processed GATE analysis: " + ", ".join(missing))

    with (analysis_dir / "summary.json").open("r", encoding="utf-8") as handle:
        ideal_summary = json.load(handle)
    n_primaries = int(ideal_summary["event_counts"]["primaries"])
    run_id = analysis_dir.name
    configured_gate_seed = int(config["simulation"]["random_seed"])
    identity_ok = bool(
        n_primaries == int(config["simulation"]["primary_events"])
        and gate_seed == configured_gate_seed
        and f"seed{gate_seed}" in run_id
    )
    if skip_transport_identity_check:
        identity_ok = True

    output_dir.mkdir(parents=True, exist_ok=True)
    pairs = _load_npz(analysis_dir / "pairs.npz")
    observable_config = json.loads(json.dumps(config))
    observable_config["selection"]["phantom_truth_policy"] = "ignore"
    truth_signal = _truth_signal_mask(pairs, observable_config)
    response_seeds = list(
        range(response_seed_start, response_seed_start + response_seed_count)
    )
    rows = [
        _evaluate_response_seed(
            pairs=pairs,
            config=observable_config,
            truth_signal=truth_signal,
            response_seed=seed,
        )
        for seed in response_seeds
    ]
    _write_csv(output_dir / "independent_gate_seed_response_stability.csv", rows)

    valid_rows = [row for row in rows if row["status"] == "ok"]
    if len(valid_rows) != len(rows):
        bad = [row["response_seed"] for row in rows if row["status"] != "ok"]
        raise RuntimeError(f"FWHM estimation failed for response seeds: {bad}")
    fwhm = np.asarray(
        [row["transverse_x_fwhm_mm"] for row in valid_rows], dtype=np.float64
    )
    counts = np.asarray(
        [row["n_selected_valid"] for row in valid_rows], dtype=np.int64
    )
    purity = np.asarray(
        [row["truth_signal_fraction"] for row in valid_rows], dtype=np.float64
    )
    bias = np.asarray([row["peak_bias_mm"] for row in valid_rows], dtype=np.float64)

    canonical_response_dir = output_dir / "canonical_response"
    canonical_seed = response_seeds[0]
    prepare_hardware_response_analysis(
        analysis_dir=analysis_dir,
        config_path=config_path,
        out_dir=canonical_response_dir,
        random_seed=canonical_seed,
    )
    materialize_response_analysis_view(
        source_analysis_dir=analysis_dir,
        response_dir=canonical_response_dir,
    )
    metrics = evaluate_standard_target(
        analysis_dir=canonical_response_dir,
        config_path=config_path,
        out_dir=canonical_response_dir / "standard_metrics",
        n_bootstrap=n_bootstrap,
    )
    canonical_estimate = metrics["transverse_x_response"]["estimate"]
    canonical_ci = metrics["transverse_x_response"]["bootstrap"].get(
        "fwhm_95_ci"
    )

    figure_paths = _plot_validation(
        rows=valid_rows,
        target_fwhm_mm=target_fwhm_mm,
        minimum_events=minimum_events,
        minimum_purity=minimum_purity,
        output_base=output_dir / "independent_gate_seed_validation",
    )

    checks = {
        "transport_identity": identity_ok,
        "all_response_seeds_below_target": bool(np.all(fwhm <= target_fwhm_mm)),
        "canonical_bootstrap_upper_below_target": bool(
            canonical_ci is not None and float(canonical_ci[1]) <= target_fwhm_mm
        ),
        "enough_events": bool(np.median(counts) >= minimum_events),
        "truth_purity": bool(np.min(purity) >= minimum_purity),
        "peak_bias": bool(np.max(np.abs(bias)) <= maximum_bias),
    }
    passed = bool(all(checks.values()) and not skip_transport_identity_check)
    standard_figures = metrics.get("figures", [])
    canonical_png = next(
        (path for path in standard_figures if str(path).lower().endswith(".png")),
        None,
    )
    json_path = output_dir / "independent_gate_seed_validation.json"
    markdown_path = output_dir / "独立GATE种子_横向FWHM5mm验收.md"
    reference_path = (
        PROJECT_ROOT
        / "outputs"
        / "transverse_fwhm_5mm_20260805"
        / "transverse_fwhm_optimization.json"
    )
    report: dict[str, Any] = {
        "schema_version": "1.0",
        "passed": passed,
        "qualification_scope": (
            "official_independent_transport_seed_validation"
            if not skip_transport_identity_check
            else "nonacceptance_smoke_test"
        ),
        "analysis_dir": str(analysis_dir.resolve()),
        "run_id": run_id,
        "config": str(config_path.resolve()),
        "gate_seed": int(gate_seed),
        "configured_gate_seed": configured_gate_seed,
        "n_primaries": n_primaries,
        "target": {
            "kind": "concrete rod",
            "diameter_mm": float(config["phantom"]["diameter_mm"]),
            "depth_mm": float(config["phantom"]["depth_mm"]),
            "axis": "y",
        },
        "selection": {
            "d1_energy_window_keV": config["selection"]["d1_energy_window_keV"],
            "d2_energy_window_keV": config["selection"]["d2_energy_window_keV"],
            "incident_angle_deg": config["selection"]["incident_angle_deg"],
            "scatter_angle_deg": config["selection"]["scatter_angle_deg"],
            "truth_used_as_selection": False,
        },
        "guardrails": {
            "target_fwhm_mm": target_fwhm_mm,
            "minimum_selected_valid_events": minimum_events,
            "minimum_truth_signal_fraction": minimum_purity,
            "maximum_absolute_peak_bias_mm": maximum_bias,
        },
        "independent_seed_summary": {
            "n_response_seeds": len(rows),
            "n_valid": len(valid_rows),
            "n_meeting_target": int(np.sum(fwhm <= target_fwhm_mm)),
            "median_fwhm_mm": float(np.median(fwhm)),
            "fwhm_seed_q2p5_mm": float(np.quantile(fwhm, 0.025)),
            "fwhm_seed_q97p5_mm": float(np.quantile(fwhm, 0.975)),
            "minimum_fwhm_mm": float(np.min(fwhm)),
            "maximum_fwhm_mm": float(np.max(fwhm)),
            "median_selected_valid": float(np.median(counts)),
            "selected_valid_range": [int(np.min(counts)), int(np.max(counts))],
            "minimum_truth_signal_fraction": float(np.min(purity)),
            "maximum_absolute_peak_bias_mm": float(np.max(np.abs(bias))),
        },
        "canonical_result": {
            "response_seed": canonical_seed,
            "fwhm_mm": canonical_estimate.get("fwhm"),
            "peak_bias_mm": canonical_estimate.get("mode"),
            "bootstrap_fwhm_95_ci_mm": canonical_ci,
            "n_selected_valid": int(metrics["n_selected_valid"]),
            "metrics": str(
                (canonical_response_dir / "standard_metrics" / "metrics.json").resolve()
            ),
            "figure_png": canonical_png,
        },
        "reference_transport_seed_result": _load_reference_result(reference_path),
        "acceptance_checks": checks,
        "outputs": {
            "seed_csv": str(
                (output_dir / "independent_gate_seed_response_stability.csv").resolve()
            ),
            "figure_png": figure_paths[0],
            "figures": figure_paths,
            "json": str(json_path.resolve()),
            "markdown": str(markdown_path.resolve()),
        },
        "interpretation_guardrail": (
            "The reported transverse line-spread FWHM includes the finite 2 mm "
            "rod diameter. Detector response is parameterized postprocessing of "
            "event-summed GATE hits. Truth labels score purity but are not used "
            "as an observable event-selection condition."
        ),
    }
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    _write_markdown_report(markdown_path, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate the frozen transverse-FWHM selection on an independent GATE seed."
    )
    parser.add_argument("--analysis-dir", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "transverse_fwhm_5mm_frozen.json",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gate-seed", type=int, default=24080573)
    parser.add_argument("--response-seed-start", type=int, default=24080601)
    parser.add_argument("--response-seed-count", type=int, default=25)
    parser.add_argument("--bootstrap", type=int, default=300)
    parser.add_argument(
        "--skip-transport-identity-check",
        action="store_true",
        help="Smoke-test only; the generated report is forced to non-acceptance scope.",
    )
    args = parser.parse_args()
    if args.response_seed_count <= 0:
        parser.error("--response-seed-count must be positive")
    result = run(
        analysis_dir=args.analysis_dir,
        config_path=args.config,
        output_dir=args.output_dir,
        gate_seed=args.gate_seed,
        response_seed_start=args.response_seed_start,
        response_seed_count=args.response_seed_count,
        n_bootstrap=args.bootstrap,
        skip_transport_identity_check=args.skip_transport_identity_check,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
