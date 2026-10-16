from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .blind import evaluate_blind_void_campaign, score_blind_void_case
from .config import load_config
from .d2_energy import evaluate_d2_energy_acceptance
from .design import evaluate_detector_size
from .electronics import evaluate_electronics_sensitivity
from .experiments import (
    build_blind_analysis_config,
    build_concrete_void_config,
    build_config_variant,
    build_standard_config,
    write_standard_config,
)
from .figures import plot_energy_spectra, plot_reconstruction_distribution
from .kinematics import synthetic_selftest
from .materials import compare_standard_materials
from .pipeline import analyze_root
from .rootio import inspect_root
from .regions import plot_detector_region_images
from .readout import evaluate_position_readout_sensitivity
from .standards import (
    evaluate_concrete_depth_profile,
    evaluate_edge_mtf,
    evaluate_standard_target,
    summarize_depth_series,
)
from .spatial import (
    summarize_double_rod_series,
    summarize_edge_series,
    summarize_rod_series,
)
from .sweeps import run_sweeps
from .voids import evaluate_concrete_void
from .validation import (
    compare_concrete_repeatability,
    compare_standard_repeatability,
    validate_concrete_physical_d1,
    validate_physical_d1_sizes,
)


def _inspect(args: argparse.Namespace) -> int:
    report = inspect_root(args.root)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def _analyze(args: argparse.Namespace) -> int:
    summary = analyze_root(
        root_path=args.root,
        truth_path=args.truth,
        config_path=args.config,
        out_dir=args.out_dir,
    )
    print()
    print("分析完成")
    print(f"  输出目录: {Path(args.out_dir).resolve()}")
    print(
        "  事件链: "
        f"{summary['event_counts']['common_event_pairs_before_filters']:,} 对 "
        f"→ {summary['event_counts']['selected_before_geometric_solver']:,} 筛选 "
        f"→ {summary['event_counts']['selected_and_reconstructed']:,} 可重建"
    )
    peak = summary["energy_validation"]["d2_photopeak_mode_keV_450_to_530"]
    print(f"  D2 全能峰众数: {peak} keV")
    if summary["warnings"]:
        print("  警告:")
        for warning in summary["warnings"]:
            print(f"    - {warning}")
    return 0


def _selftest(_: argparse.Namespace) -> int:
    report = synthetic_selftest()
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["passed"] else 1


def _sweep(args: argparse.Namespace) -> int:
    report = run_sweeps(
        analysis_dir=args.analysis_dir,
        config_path=args.config,
        out_dir=args.out_dir,
    )
    output = (
        Path(args.out_dir)
        if args.out_dir
        else Path(args.analysis_dir) / "sweeps"
    )
    print(f"扫描完成: {output.resolve()}")
    print(f"  扫描行数: {len(report['rows'])}")
    print(f"  数据集中真值匹配事件: {report['truth_matched_in_dataset']}")
    return 0


def _d2_energy_study(args: argparse.Namespace) -> int:
    report = evaluate_d2_energy_acceptance(
        analysis_dir=args.analysis_dir,
        config_path=args.config,
        out_dir=args.out_dir,
        lower_thresholds_keV=args.lower_thresholds_kev,
        upper_energy_keV=args.upper_energy_kev,
    )
    baseline = report["photopeak_region_baseline"]
    lowest = report["lowest_threshold_result"]
    gain = (
        lowest["truth_signal_sensitivity_retention_vs_450keV"]
        - 1.0
    )
    print(f"D2 energy-acceptance study complete: {Path(args.out_dir).resolve()}")
    print(
        "  Truth-signal sensitivity gain vs "
        f"{baseline['setting']}: {gain:+.1%}"
    )
    return 0


def _standard_config(args: argparse.Namespace) -> int:
    base = load_config(args.base_config)
    config = build_standard_config(
        base,
        kind=args.kind,
        depth_mm=args.depth_mm,
        thickness_mm=args.thickness_mm,
        lateral_mm=args.lateral_mm,
        radius_mm=args.radius_mm,
        separation_mm=args.separation_mm,
        primary_events=args.primary_events,
        material=args.material,
        random_seed=args.random_seed,
        d1_rmax_mm=args.d1_rmax_mm,
    )
    output = write_standard_config(args.out, config)
    print(f"标准体模配置已生成: {output.resolve()}")
    return 0


def _variant_config(args: argparse.Namespace) -> int:
    base = load_config(args.base_config)
    config = build_config_variant(
        base,
        primary_events=args.primary_events,
        random_seed=args.random_seed,
        d1_rmax_mm=args.d1_rmax_mm,
        variant_id=args.variant_id,
    )
    output = write_standard_config(args.out, config)
    print(f"配置变体已生成: {output.resolve()}")
    return 0


def _blind_analysis_config(args: argparse.Namespace) -> int:
    base = load_config(args.base_config)
    config = build_blind_analysis_config(
        base,
        primary_events=args.primary_events,
    )
    output = write_standard_config(args.out, config)
    print(f"Blind neutral analysis configuration written: {output.resolve()}")
    return 0


def _void_config(args: argparse.Namespace) -> int:
    base = load_config(args.base_config)
    config = build_concrete_void_config(
        base,
        depth_mm=args.depth_mm,
        radius_mm=args.radius_mm,
        length_mm=args.length_mm,
        x_mm=args.x_mm,
        primary_events=args.primary_events,
        random_seed=args.random_seed,
        d1_rmax_mm=args.d1_rmax_mm,
    )
    output = write_standard_config(args.out, config)
    print(f"Concrete-void configuration written: {output.resolve()}")
    return 0


def _evaluate_standard(args: argparse.Namespace) -> int:
    report = evaluate_standard_target(
        analysis_dir=args.analysis_dir,
        config_path=args.config,
        out_dir=args.out_dir,
        n_bootstrap=args.bootstrap,
        bandwidth_scale=args.bandwidth_scale,
    )
    print(f"标准体模评估完成: {Path(args.out_dir).resolve() if args.out_dir else Path(args.analysis_dir).resolve() / 'standard_metrics'}")
    print(f"  有效事件: {report['n_selected_valid']:,}")
    estimate = (
        report["fwhm_method"]["estimate"]
        if report["kind"] == "thin_layer"
        else (
            report["transverse_x_response"]["estimate"]
            if report["kind"] == "rod"
            else None
        )
    )
    if estimate is not None:
        print(f"  FWHM 状态: {estimate['status']}")
    if estimate is not None and estimate.get("fwhm") is not None:
        label = "轴向" if report["kind"] == "thin_layer" else "横向 x"
        print(f"  {label} FWHM: {estimate['fwhm']:.3f} mm")
    if report["kind"] == "double_rod":
        ratio = report["two_peak_metrics"]["valley_to_lower_peak"]
        print(f"  双峰状态: {report['two_peak_status']}")
        if ratio is not None:
            print(f"  谷/较低峰比: {ratio:.3f}")
    return 0


def _depth_series(args: argparse.Namespace) -> int:
    report = summarize_depth_series(
        metrics_paths=args.metrics,
        out_dir=args.out_dir,
    )
    print(f"深度性能汇总完成: {Path(args.out_dir).resolve()}")
    print(f"  深度点数: {len(report['rows'])}")
    return 0


def _plot_spectra(args: argparse.Namespace) -> int:
    report = plot_energy_spectra(
        analysis_dir=args.analysis_dir,
        out_dir=args.out_dir,
        d2_fwhm_percent_at_511=args.d2_fwhm_percent_at_511,
        random_seed=args.random_seed,
    )
    print(f"能谱图已生成: {Path(report['figures'][0]).parent}")
    print(f"  D2 全能峰众数: {report['d2_photopeak_mode_keV']} keV")
    return 0


def _detector_size(args: argparse.Namespace) -> int:
    report = evaluate_detector_size(
        sweep_csv=args.sweep_csv,
        config_path=args.config,
        out_dir=args.out_dir,
    )
    candidate = report["smallest_radius_retaining_95pct_efficiency"]
    print(f"D1 尺寸权衡完成: {Path(args.out_dir).resolve()}")
    print(f"  200 mm D1 GAGG 质量: {report['d1_physical_mass_kg']:.3f} kg")
    if candidate is not None:
        print(
            "  后处理 95% 效率保留候选: "
            f"R={candidate['outer_radius_mm']:.0f} mm, "
            f"{candidate['d1_mass_kg']:.3f} kg"
        )
    return 0


def _plot_reconstruction(args: argparse.Namespace) -> int:
    report = plot_reconstruction_distribution(
        analysis_dir=args.analysis_dir,
        out_dir=args.out_dir,
    )
    print(f"重建分布图已生成: {Path(report['figures'][0]).parent}")
    print(
        "  重建点落在体模深度区间: "
        f"{report['fraction_inside_phantom_depth']:.1%}"
    )
    return 0


def _evaluate_edge(args: argparse.Namespace) -> int:
    report = evaluate_edge_mtf(
        edge_analysis_dir=args.edge_analysis_dir,
        reference_analysis_dir=args.reference_analysis_dir,
        config_path=args.config,
        out_dir=args.out_dir,
        bin_width_mm=args.bin_width_mm,
        minimum_reference_count=args.minimum_reference_count,
        n_bootstrap=args.bootstrap,
    )
    print(f"边缘 MTF 评估完成: {Path(args.out_dir).resolve()}")
    for label in ("mtf50", "mtf10"):
        value = report[f"{label}_cycles_per_mm"]
        text = f"{value:.4f}" if value is not None else "unavailable"
        print(f"  {label.upper()}: {text} cycles/mm")
    return 0


def _evaluate_concrete_depth(args: argparse.Namespace) -> int:
    report = evaluate_concrete_depth_profile(
        analysis_dir=args.analysis_dir,
        config_path=args.config,
        out_dir=args.out_dir,
        bin_width_mm=args.bin_width_mm,
        minimum_events=args.minimum_events,
        n_bootstrap=args.bootstrap,
        bandwidth_scale=args.bandwidth_scale,
    )
    output = (
        Path(args.out_dir)
        if args.out_dir
        else Path(args.analysis_dir) / "concrete_depth_profile"
    )
    print(f"Concrete depth profile complete: {output.resolve()}")
    print(
        "  Observable-selected valid events: "
        f"{report['n_observable_selected_valid_all']:,}"
    )
    print(
        "  Of these, truth-matched concrete events: "
        f"{report['n_observable_selected_valid_truth_matched']:,}"
    )
    deepest = report["deepest_evaluable_bin_right_edge_mm"]
    if deepest is not None:
        print(
            "  Deepest statistically evaluable bin right edge: "
            f"{deepest:g} mm"
        )
    return 0


def _evaluate_void(args: argparse.Namespace) -> int:
    report = evaluate_concrete_void(
        void_analysis_dir=args.void_analysis_dir,
        reference_analysis_dir=args.reference_analysis_dir,
        config_path=args.config,
        out_dir=args.out_dir,
        roi_radius_scale=args.roi_radius_scale,
        map_bin_width_mm=args.map_bin_width_mm,
        smoothing_sigma_bins=args.smoothing_sigma_bins,
    )
    metrics = report["predefined_primary_roi"]
    print(f"Concrete-void evaluation complete: {Path(args.out_dir).resolve()}")
    print(f"  Contrast: {metrics['contrast']:.1%}")
    print(f"  CNR: {metrics['cnr']:.3f}")
    return 0


def _score_blind_void(args: argparse.Namespace) -> int:
    report = score_blind_void_case(
        sample_analysis_dir=args.sample_analysis_dir,
        reference_analysis_dir=args.reference_analysis_dir,
        selection_config_path=args.selection_config,
        out_dir=args.out_dir,
        roi_radius_mm=args.roi_radius_mm,
        x_min_mm=args.x_min_mm,
        x_max_mm=args.x_max_mm,
        depth_min_mm=args.depth_min_mm,
        depth_max_mm=args.depth_max_mm,
        search_step_mm=args.search_step_mm,
        y_half_width_mm=args.y_half_width_mm,
        candidate_threshold=args.candidate_threshold,
    )
    center = report["estimated_center_mm"]
    print(f"Blind void score complete: {Path(args.out_dir).resolve()}")
    print(f"  Maximum score: {report['maximum_score']:.3f}")
    print(
        "  Estimated center: "
        f"x={center['x']:.1f} mm, depth={center['depth']:.1f} mm"
    )
    print(
        "  Candidate flag at frozen threshold "
        f"{report['candidate_threshold']:.2f}: "
        f"{report['candidate_flag']}"
    )
    return 0


def _evaluate_blind_void_campaign(args: argparse.Namespace) -> int:
    report = evaluate_blind_void_campaign(
        score_bundle_path=args.scores,
        committed_score_hash_path=args.score_hash,
        truth_key_path=args.truth_key,
        truth_salt_path=args.truth_salt,
        truth_commitment_path=args.truth_commitment,
        out_dir=args.out_dir,
    )
    positive = report["positive_cases"]
    negative = report["negative_cases"]
    print(f"Blind campaign reveal complete: {Path(args.out_dir).resolve()}")
    print(
        "  Correct localized detections: "
        f"{positive['correct_localized_detections']}/{positive['n']}"
    )
    print(
        f"  False alarms: {negative['false_alarms']}/{negative['n']}"
    )
    print(
        "  Workflow qualification passed: "
        f"{report['workflow_qualification_rule']['passed']}"
    )
    return 0


def _electronics_sensitivity(args: argparse.Namespace) -> int:
    report = evaluate_electronics_sensitivity(
        analysis_dir=args.analysis_dir,
        config_path=args.config,
        out_dir=args.out_dir,
        fwhm_percent_at_511=args.fwhm_percent,
        random_seed=args.random_seed,
    )
    print(f"Electronics sensitivity complete: {Path(args.out_dir).resolve()}")
    print(f"  Scenarios: {len(report['rows'])}")
    return 0


def _compare_standard_repeats(args: argparse.Namespace) -> int:
    report = compare_standard_repeatability(
        metrics_paths=args.metrics,
        out_dir=args.out_dir,
    )
    pairwise = report["pairwise"]
    print(f"Standard-target repeatability complete: {Path(args.out_dir).resolve()}")
    print(
        "  Pre-registered repeatability criteria passed: "
        f"{pairwise['pre_registered_repeatability_pass']}"
    )
    return 0


def _validate_d1_sizes(args: argparse.Namespace) -> int:
    report = validate_physical_d1_sizes(
        full_metrics_path=args.full_metrics,
        physical_metrics_paths=args.physical_metrics,
        posthoc_size_report_path=args.posthoc_report,
        out_dir=args.out_dir,
    )
    print(f"Physical D1 validation complete: {Path(args.out_dir).resolve()}")
    print(
        "  Maximum efficiency-retention prediction error: "
        f"{report['maximum_absolute_efficiency_retention_prediction_error']:.3%}"
    )
    print(
        "  All physical sizes passed pre-registered thresholds: "
        f"{report['all_physical_sizes_pass_pre_registered_thresholds']}"
    )
    return 0


def _compare_concrete_repeats(args: argparse.Namespace) -> int:
    report = compare_concrete_repeatability(
        metrics_paths=args.metrics,
        out_dir=args.out_dir,
    )
    print(f"Concrete repeatability complete: {Path(args.out_dir).resolve()}")
    print(
        "  Median absolute sensitivity difference: "
        f"{report['median_absolute_sensitivity_relative_difference']:.2%}"
    )
    return 0


def _compare_materials(args: argparse.Namespace) -> int:
    report = compare_standard_materials(
        metrics_paths=args.metrics,
        out_dir=args.out_dir,
    )
    print(f"Material comparison complete: {Path(args.out_dir).resolve()}")
    print(f"  Materials: {len(report['rows'])}")
    return 0


def _validate_concrete_d1(args: argparse.Namespace) -> int:
    report = validate_concrete_physical_d1(
        full_analysis_dir=args.full_analysis_dir,
        physical_analysis_dir=args.physical_analysis_dir,
        full_config_path=args.full_config,
        physical_config_path=args.physical_config,
        out_dir=args.out_dir,
        bin_width_mm=args.bin_width_mm,
    )
    retention = report["retention"]
    error = retention["observable_prediction_error"]
    print(f"Concrete physical-D1 validation complete: {Path(args.out_dir).resolve()}")
    print(f"  Observable retention prediction error: {error:.3%}")
    print(
        "  Observable retention passed 2-point threshold: "
        f"{retention['observable_prediction_within_2_percentage_points']}"
    )
    return 0


def _plot_region_images(args: argparse.Namespace) -> int:
    report = plot_detector_region_images(
        analysis_dir=args.analysis_dir,
        config_path=args.config,
        out_dir=args.out_dir,
        bin_width_mm=args.bin_width_mm,
        smoothing_sigma_bins=args.smoothing_sigma_bins,
    )
    print(f"Detector-region images complete: {Path(args.out_dir).resolve()}")
    print(f"  Region definitions: {len(report['rows'])}")
    return 0


def _summarize_rods(args: argparse.Namespace) -> int:
    report = summarize_rod_series(
        metrics_paths=args.metrics,
        out_dir=args.out_dir,
    )
    result = report["narrowest_rod_result"]
    print(f"Rod-width summary complete: {Path(args.out_dir).resolve()}")
    print(
        "  Narrowest-rod transverse FWHM: "
        f"{result['transverse_x_fwhm_mm']:.3f} mm"
    )
    return 0


def _summarize_double_rods(args: argparse.Namespace) -> int:
    report = summarize_double_rod_series(
        metrics_paths=args.metrics,
        out_dir=args.out_dir,
    )
    result = report["smallest_bandwidth_robust_resolved_separation"]
    print(f"Double-rod summary complete: {Path(args.out_dir).resolve()}")
    if result is None:
        print(
            "  No tested separation met the criterion across all KDE "
            "bandwidths"
        )
    else:
        print(
            "  Smallest bandwidth-robust resolved separation: "
            f"{result['target_center_separation_mm']:.3f} mm"
        )
    return 0


def _summarize_edges(args: argparse.Namespace) -> int:
    report = summarize_edge_series(
        metrics_paths=args.metrics,
        out_dir=args.out_dir,
    )
    shallow = report["rows"][0]
    deep = report["rows"][-1]
    print(f"Edge-depth summary complete: {Path(args.out_dir).resolve()}")
    print(
        f"  {shallow['target_depth_mm']:g} mm MTF50/MTF10: "
        f"{shallow['mtf50_cycles_per_mm']:.4f}/"
        f"{shallow['mtf10_cycles_per_mm']:.4f} cycles/mm"
    )
    print(
        f"  {deep['target_depth_mm']:g} mm MTF50/MTF10: "
        f"{deep['mtf50_cycles_per_mm']:.4f}/"
        f"{deep['mtf10_cycles_per_mm']:.4f} cycles/mm"
    )
    return 0


def _position_readout_sensitivity(args: argparse.Namespace) -> int:
    report = evaluate_position_readout_sensitivity(
        analysis_dir=args.analysis_dir,
        config_path=args.config,
        out_dir=args.out_dir,
        d1_xy_fwhm_mm=args.d1_xy_fwhm_mm,
        random_seed=args.random_seed,
    )
    print(f"Position-readout sensitivity complete: {Path(args.out_dir).resolve()}")
    print(f"  Scenarios: {len(report['rows'])}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m backscatter",
        description="511 keV 双光子标记背散射分析",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser(
        "inspect-root", help="只读检查 GATE ROOT 的树、分支和初级事件元数据"
    )
    inspect_parser.add_argument("--root", required=True)
    inspect_parser.set_defaults(func=_inspect)

    analyze_parser = subparsers.add_parser(
        "analyze", help="聚合 Hits、构建符合、重建并生成审计 JSON"
    )
    analyze_parser.add_argument("--root", required=True)
    analyze_parser.add_argument("--truth")
    analyze_parser.add_argument(
        "--config", default="configs/baseline.json"
    )
    analyze_parser.add_argument("--out-dir", required=True)
    analyze_parser.set_defaults(func=_analyze)

    selftest_parser = subparsers.add_parser(
        "selftest", help="运行带已知散射顶点的闭环公式自检"
    )
    selftest_parser.set_defaults(func=_selftest)

    sweep_parser = subparsers.add_parser(
        "sweep", help="扫描能量、角度、D1 半径/扇区和 D2 区域"
    )
    sweep_parser.add_argument("--analysis-dir", required=True)
    sweep_parser.add_argument(
        "--config", default="configs/baseline.json"
    )
    sweep_parser.add_argument("--out-dir")
    sweep_parser.set_defaults(func=_sweep)

    d2_energy_parser = subparsers.add_parser(
        "d2-energy-study",
        help=(
            "Test whether D2 partial-energy events below the 511 keV "
            "photopeak region add useful image sensitivity"
        ),
    )
    d2_energy_parser.add_argument("--analysis-dir", required=True)
    d2_energy_parser.add_argument("--config", required=True)
    d2_energy_parser.add_argument("--out-dir", required=True)
    d2_energy_parser.add_argument(
        "--lower-thresholds-kev",
        type=float,
        nargs="+",
        default=[1, 20, 50, 100, 150, 200, 250, 300, 350, 400, 450],
    )
    d2_energy_parser.add_argument(
        "--upper-energy-kev",
        type=float,
        default=560.0,
    )
    d2_energy_parser.set_defaults(func=_d2_energy_study)

    standard_parser = subparsers.add_parser(
        "standard-config", help="从冻结基线生成薄层、点、双点或边缘体模配置"
    )
    standard_parser.add_argument(
        "--kind",
        required=True,
        choices=[
            "thin_layer",
            "rod",
            "double_rod",
            "point",
            "double_point",
            "edge",
        ],
    )
    standard_parser.add_argument("--depth-mm", type=float, required=True)
    standard_parser.add_argument("--thickness-mm", type=float, default=1.0)
    standard_parser.add_argument("--lateral-mm", type=float, default=300.0)
    standard_parser.add_argument("--radius-mm", type=float, default=2.5)
    standard_parser.add_argument("--separation-mm", type=float, default=10.0)
    standard_parser.add_argument("--primary-events", type=int, default=2_000_000)
    standard_parser.add_argument("--random-seed", type=int)
    standard_parser.add_argument("--d1-rmax-mm", type=float)
    standard_parser.add_argument(
        "--material",
        choices=["Concrete", "Water", "Iron"],
        default="Concrete",
    )
    standard_parser.add_argument(
        "--base-config", default="configs/baseline.json"
    )
    standard_parser.add_argument("--out", required=True)
    standard_parser.set_defaults(func=_standard_config)

    variant_parser = subparsers.add_parser(
        "variant-config",
        help="从任意冻结配置生成事件数、随机种子或 D1 几何变体",
    )
    variant_parser.add_argument("--base-config", required=True)
    variant_parser.add_argument("--primary-events", type=int)
    variant_parser.add_argument("--random-seed", type=int)
    variant_parser.add_argument("--d1-rmax-mm", type=float)
    variant_parser.add_argument("--variant-id")
    variant_parser.add_argument("--out", required=True)
    variant_parser.set_defaults(func=_variant_config)

    blind_config_parser = subparsers.add_parser(
        "blind-analysis-config",
        help=(
            "Generate one neutral observable-only analysis configuration "
            "that contains no blinded case class or target position"
        ),
    )
    blind_config_parser.add_argument(
        "--primary-events",
        type=int,
        default=1_000_000,
    )
    blind_config_parser.add_argument(
        "--base-config",
        default="configs/baseline.json",
    )
    blind_config_parser.add_argument("--out", required=True)
    blind_config_parser.set_defaults(func=_blind_analysis_config)

    void_parser = subparsers.add_parser(
        "void-config",
        help="Generate a full concrete slab containing one cylindrical void",
    )
    void_parser.add_argument("--depth-mm", type=float, required=True)
    void_parser.add_argument("--radius-mm", type=float, required=True)
    void_parser.add_argument("--length-mm", type=float, default=300.0)
    void_parser.add_argument("--x-mm", type=float, default=0.0)
    void_parser.add_argument("--primary-events", type=int, default=2_000_000)
    void_parser.add_argument("--random-seed", type=int)
    void_parser.add_argument("--d1-rmax-mm", type=float)
    void_parser.add_argument(
        "--base-config",
        default="configs/baseline.json",
    )
    void_parser.add_argument("--out", required=True)
    void_parser.set_defaults(func=_void_config)

    evaluate_parser = subparsers.add_parser(
        "evaluate-standard",
        help="计算薄层、单杆或双杆的 PSF、可分辨性、偏差和效率",
    )
    evaluate_parser.add_argument("--analysis-dir", required=True)
    evaluate_parser.add_argument("--config", required=True)
    evaluate_parser.add_argument("--out-dir")
    evaluate_parser.add_argument("--bootstrap", type=int, default=500)
    evaluate_parser.add_argument("--bandwidth-scale", type=float, default=1.0)
    evaluate_parser.set_defaults(func=_evaluate_standard)

    depth_parser = subparsers.add_parser(
        "depth-series", help="汇总多个薄层深度的 FWHM、偏差与效率曲线"
    )
    depth_parser.add_argument("--metrics", nargs="+", required=True)
    depth_parser.add_argument("--out-dir", required=True)
    depth_parser.set_defaults(func=_depth_series)

    spectra_parser = subparsers.add_parser(
        "plot-spectra",
        help="绘制 D1/D2 原始能谱和带假设能量展宽的 D2 511 keV 峰",
    )
    spectra_parser.add_argument("--analysis-dir", required=True)
    spectra_parser.add_argument("--out-dir")
    spectra_parser.add_argument(
        "--d2-fwhm-percent-at-511",
        type=float,
        default=10.0,
    )
    spectra_parser.add_argument("--random-seed", type=int, default=24062001)
    spectra_parser.set_defaults(func=_plot_spectra)

    size_parser = subparsers.add_parser(
        "detector-size",
        help="绘制 D1 半径与 GAGG 质量、成像灵敏度和分辨率的关系",
    )
    size_parser.add_argument("--sweep-csv", required=True)
    size_parser.add_argument("--config", default="configs/baseline.json")
    size_parser.add_argument("--out-dir", required=True)
    size_parser.set_defaults(func=_detector_size)

    reconstruction_parser = subparsers.add_parser(
        "plot-reconstruction",
        help="绘制均匀体模重建分布，并明确标注其不是分辨率",
    )
    reconstruction_parser.add_argument("--analysis-dir", required=True)
    reconstruction_parser.add_argument("--out-dir")
    reconstruction_parser.set_defaults(func=_plot_reconstruction)

    edge_parser = subparsers.add_parser(
        "evaluate-edge",
        help="用半平面与同深度完整薄层参考计算 ESF、LSF、MTF10/MTF50",
    )
    edge_parser.add_argument("--edge-analysis-dir", required=True)
    edge_parser.add_argument("--reference-analysis-dir", required=True)
    edge_parser.add_argument("--config", required=True)
    edge_parser.add_argument("--out-dir", required=True)
    edge_parser.add_argument("--bin-width-mm", type=float, default=1.0)
    edge_parser.add_argument("--minimum-reference-count", type=int, default=5)
    edge_parser.add_argument("--bootstrap", type=int, default=500)
    edge_parser.set_defaults(func=_evaluate_edge)

    concrete_depth_parser = subparsers.add_parser(
        "evaluate-concrete-depth",
        help=(
            "Bin a full concrete slab by true first-Compton depth and report "
            "depth-dependent sensitivity and axial error"
        ),
    )
    concrete_depth_parser.add_argument("--analysis-dir", required=True)
    concrete_depth_parser.add_argument(
        "--config",
        default="configs/baseline.json",
    )
    concrete_depth_parser.add_argument("--out-dir")
    concrete_depth_parser.add_argument(
        "--bin-width-mm",
        type=float,
        default=10.0,
    )
    concrete_depth_parser.add_argument(
        "--minimum-events",
        type=int,
        default=100,
    )
    concrete_depth_parser.add_argument("--bootstrap", type=int, default=200)
    concrete_depth_parser.add_argument(
        "--bandwidth-scale",
        type=float,
        default=1.0,
    )
    concrete_depth_parser.set_defaults(func=_evaluate_concrete_depth)

    void_evaluation_parser = subparsers.add_parser(
        "evaluate-void",
        help=(
            "Compare a concrete-void reconstruction with a homogeneous "
            "reference using observable-only selection"
        ),
    )
    void_evaluation_parser.add_argument(
        "--void-analysis-dir",
        required=True,
    )
    void_evaluation_parser.add_argument(
        "--reference-analysis-dir",
        required=True,
    )
    void_evaluation_parser.add_argument("--config", required=True)
    void_evaluation_parser.add_argument("--out-dir", required=True)
    void_evaluation_parser.add_argument(
        "--roi-radius-scale",
        type=float,
        default=1.5,
    )
    void_evaluation_parser.add_argument(
        "--map-bin-width-mm",
        type=float,
        default=2.0,
    )
    void_evaluation_parser.add_argument(
        "--smoothing-sigma-bins",
        type=float,
        default=1.0,
    )
    void_evaluation_parser.set_defaults(func=_evaluate_void)

    blind_void_parser = subparsers.add_parser(
        "score-blind-void",
        help=(
            "Search an unknown concrete case for the maximum local "
            "control-normalized void-deficit score without reading truth"
        ),
    )
    blind_void_parser.add_argument(
        "--sample-analysis-dir",
        required=True,
    )
    blind_void_parser.add_argument(
        "--reference-analysis-dir",
        required=True,
    )
    blind_void_parser.add_argument("--selection-config", required=True)
    blind_void_parser.add_argument("--out-dir", required=True)
    blind_void_parser.add_argument(
        "--roi-radius-mm",
        type=float,
        default=15.0,
    )
    blind_void_parser.add_argument(
        "--x-min-mm",
        type=float,
        default=-30.0,
    )
    blind_void_parser.add_argument(
        "--x-max-mm",
        type=float,
        default=30.0,
    )
    blind_void_parser.add_argument(
        "--depth-min-mm",
        type=float,
        default=20.0,
    )
    blind_void_parser.add_argument(
        "--depth-max-mm",
        type=float,
        default=40.0,
    )
    blind_void_parser.add_argument(
        "--search-step-mm",
        type=float,
        default=2.0,
    )
    blind_void_parser.add_argument(
        "--y-half-width-mm",
        type=float,
        default=150.0,
    )
    blind_void_parser.add_argument(
        "--candidate-threshold",
        type=float,
        default=4.0,
    )
    blind_void_parser.set_defaults(func=_score_blind_void)

    blind_reveal_parser = subparsers.add_parser(
        "evaluate-blind-void-campaign",
        help=(
            "Verify frozen score and salted truth commitments, then reveal "
            "a blind qualification campaign"
        ),
    )
    blind_reveal_parser.add_argument("--scores", required=True)
    blind_reveal_parser.add_argument("--score-hash", required=True)
    blind_reveal_parser.add_argument("--truth-key", required=True)
    blind_reveal_parser.add_argument("--truth-salt", required=True)
    blind_reveal_parser.add_argument("--truth-commitment", required=True)
    blind_reveal_parser.add_argument("--out-dir", required=True)
    blind_reveal_parser.set_defaults(func=_evaluate_blind_void_campaign)

    electronics_parser = subparsers.add_parser(
        "electronics-sensitivity",
        help=(
            "Re-run observable selection and reconstruction under hypothetical "
            "D1/D2 energy-resolution scenarios"
        ),
    )
    electronics_parser.add_argument("--analysis-dir", required=True)
    electronics_parser.add_argument("--config", required=True)
    electronics_parser.add_argument("--out-dir", required=True)
    electronics_parser.add_argument(
        "--fwhm-percent",
        type=float,
        nargs="+",
        default=[0.0, 5.0, 10.0, 15.0],
    )
    electronics_parser.add_argument(
        "--random-seed",
        type=int,
        default=24062001,
    )
    electronics_parser.set_defaults(func=_electronics_sensitivity)

    standard_repeat_parser = subparsers.add_parser(
        "compare-standard-repeats",
        help="Compare two independent standard-target random seeds",
    )
    standard_repeat_parser.add_argument(
        "--metrics",
        nargs=2,
        required=True,
    )
    standard_repeat_parser.add_argument("--out-dir", required=True)
    standard_repeat_parser.set_defaults(func=_compare_standard_repeats)

    d1_validation_parser = subparsers.add_parser(
        "validate-d1-sizes",
        help=(
            "Compare physical D1-radius GATE runs with post-hoc active-area "
            "predictions"
        ),
    )
    d1_validation_parser.add_argument("--full-metrics", required=True)
    d1_validation_parser.add_argument(
        "--physical-metrics",
        nargs="+",
        required=True,
    )
    d1_validation_parser.add_argument("--posthoc-report", required=True)
    d1_validation_parser.add_argument("--out-dir", required=True)
    d1_validation_parser.set_defaults(func=_validate_d1_sizes)

    concrete_repeat_parser = subparsers.add_parser(
        "compare-concrete-repeats",
        help="Compare two homogeneous-concrete depth profiles by seed",
    )
    concrete_repeat_parser.add_argument("--metrics", nargs=2, required=True)
    concrete_repeat_parser.add_argument("--out-dir", required=True)
    concrete_repeat_parser.set_defaults(func=_compare_concrete_repeats)

    material_parser = subparsers.add_parser(
        "compare-materials",
        help="Compare equal-size standard targets made from different materials",
    )
    material_parser.add_argument("--metrics", nargs="+", required=True)
    material_parser.add_argument("--out-dir", required=True)
    material_parser.set_defaults(func=_compare_materials)

    concrete_d1_parser = subparsers.add_parser(
        "validate-concrete-d1",
        help=(
            "Compare a physical reduced-radius concrete run with a post-hoc "
            "radius cut from the full D1 run"
        ),
    )
    concrete_d1_parser.add_argument("--full-analysis-dir", required=True)
    concrete_d1_parser.add_argument(
        "--physical-analysis-dir",
        required=True,
    )
    concrete_d1_parser.add_argument("--full-config", required=True)
    concrete_d1_parser.add_argument("--physical-config", required=True)
    concrete_d1_parser.add_argument("--out-dir", required=True)
    concrete_d1_parser.add_argument(
        "--bin-width-mm",
        type=float,
        default=10.0,
    )
    concrete_d1_parser.set_defaults(func=_validate_concrete_d1)

    region_image_parser = subparsers.add_parser(
        "plot-region-images",
        help=(
            "Reconstruct images from D1 radial/azimuthal and D2 central/outer "
            "regions without changing detector geometry"
        ),
    )
    region_image_parser.add_argument("--analysis-dir", required=True)
    region_image_parser.add_argument("--config", required=True)
    region_image_parser.add_argument("--out-dir", required=True)
    region_image_parser.add_argument(
        "--bin-width-mm",
        type=float,
        default=4.0,
    )
    region_image_parser.add_argument(
        "--smoothing-sigma-bins",
        type=float,
        default=0.75,
    )
    region_image_parser.set_defaults(func=_plot_region_images)

    rod_summary_parser = subparsers.add_parser(
        "summarize-rods",
        help="Summarize transverse response across several rod diameters",
    )
    rod_summary_parser.add_argument("--metrics", nargs="+", required=True)
    rod_summary_parser.add_argument("--out-dir", required=True)
    rod_summary_parser.set_defaults(func=_summarize_rods)

    double_rod_summary_parser = subparsers.add_parser(
        "summarize-double-rods",
        help="Summarize resolvability across double-rod separations",
    )
    double_rod_summary_parser.add_argument(
        "--metrics",
        nargs="+",
        required=True,
    )
    double_rod_summary_parser.add_argument("--out-dir", required=True)
    double_rod_summary_parser.set_defaults(func=_summarize_double_rods)

    edge_summary_parser = subparsers.add_parser(
        "summarize-edges",
        help="Summarize model-based MTF across edge-target depths",
    )
    edge_summary_parser.add_argument("--metrics", nargs="+", required=True)
    edge_summary_parser.add_argument("--out-dir", required=True)
    edge_summary_parser.set_defaults(func=_summarize_edges)

    readout_parser = subparsers.add_parser(
        "position-readout-sensitivity",
        help=(
            "Compare ideal continuous GATE hit centroids with channel-center, "
            "no-DOI and finite D1 xy-resolution scenarios"
        ),
    )
    readout_parser.add_argument("--analysis-dir", required=True)
    readout_parser.add_argument("--config", required=True)
    readout_parser.add_argument("--out-dir", required=True)
    readout_parser.add_argument(
        "--d1-xy-fwhm-mm",
        type=float,
        nargs="+",
        default=[0.0, 2.0, 5.0, 10.0],
    )
    readout_parser.add_argument(
        "--random-seed",
        type=int,
        default=24062002,
    )
    readout_parser.set_defaults(func=_position_readout_sensitivity)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        code = int(args.func(args))
    except (FileNotFoundError, KeyError, ValueError) as exc:
        print(f"错误: {exc}", file=sys.stderr)
        code = 2
    raise SystemExit(code)
