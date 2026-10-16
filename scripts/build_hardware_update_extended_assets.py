from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backscatter.design import evaluate_detector_size
from backscatter.hardware_figures import (
    build_hardware_update_assets,
    materialize_response_analysis_view,
    prepare_hardware_response_analysis,
    resolve_hardware_profile_for_variant,
)
from backscatter.regions import plot_detector_region_images
from backscatter.spatial import summarize_double_rod_series
from backscatter.standards import evaluate_standard_target
from backscatter.sweeps import run_sweeps
from backscatter.validation import validate_physical_d1_sizes
from backscatter.voids import evaluate_concrete_void


def build_extended_assets(root: Path) -> dict[str, Any]:
    hardware_config = root / "configs" / "hardware_20260731.json"
    output = root / "outputs" / "hardware_update_20260731"
    resolved_configs = output / "resolved_configs"
    analyses = output / "response_variants"
    figures = output / "figures"

    base_concrete = root / "runs" / "concrete_slab_2m_seed24061873_local"
    base_thin30 = root / "runs" / "depth_psf_30mm_seed24061873_local"
    core = build_hardware_update_assets(
        analysis_dir=base_concrete,
        config_path=hardware_config,
        out_dir=output,
        threshold_analysis_dir=base_thin30,
        threshold_base_config_path=(
            root / "configs" / "generated" / "thin_layer_depth30.json"
        ),
    )
    concrete_response = output / "response_analysis"
    thin30_response = output / "threshold_response_analysis"
    materialize_response_analysis_view(
        source_analysis_dir=base_concrete,
        response_dir=concrete_response,
    )
    materialize_response_analysis_view(
        source_analysis_dir=base_thin30,
        response_dir=thin30_response,
    )
    thin30_config = (
        resolved_configs / "thin_layer_depth30_hardware_20260731.json"
    )

    def prepare_variant(
        *,
        tag: str,
        analysis_dir: Path,
        base_config: Path,
        random_seed: int,
    ) -> tuple[Path, Path, dict[str, Any]]:
        resolved = resolve_hardware_profile_for_variant(
            base_config_path=base_config,
            hardware_profile_path=hardware_config,
            output_path=resolved_configs / f"{tag}.json",
        )
        response_dir = analyses / tag
        report = prepare_hardware_response_analysis(
            analysis_dir=analysis_dir,
            config_path=resolved,
            out_dir=response_dir,
            random_seed=random_seed,
        )
        materialize_response_analysis_view(
            source_analysis_dir=analysis_dir,
            response_dir=response_dir,
        )
        return resolved, response_dir, report

    # Slide 10: the displayed 20 mm diameter, 30 mm deep void pilot.
    void_tag = "concrete_void_d20_depth30_hardware_20260731"
    void_config, void_response, void_response_report = prepare_variant(
        tag=void_tag,
        analysis_dir=(
            root
            / "runs"
            / "concrete_void_d20_depth30_2m_seed24061875_local_manual_void1"
        ),
        base_config=(
            root
            / "configs"
            / "generated"
            / "concrete_void_d20_depth30_2m.json"
        ),
        random_seed=24073120,
    )
    void_report = evaluate_concrete_void(
        void_analysis_dir=void_response,
        reference_analysis_dir=concrete_response,
        config_path=void_config,
        out_dir=figures / "slide10_void_response",
    )

    # Slide 11: 2 mm double rods at 5, 10 and 15 mm centre separation.
    double_metric_paths: list[Path] = []
    double_response_reports: list[dict[str, Any]] = []
    for index, separation in enumerate((5, 10, 15)):
        tag = f"double_rod_sep{separation}_hardware_20260731"
        config, response_dir, response_report = prepare_variant(
            tag=tag,
            analysis_dir=(
                root
                / "runs"
                / (
                    f"double_rod_width2_sep{separation}_depth30_5m_"
                    "seed24061873_local_manual_double_a_20260719_1"
                )
            ),
            base_config=(
                root
                / "configs"
                / "generated"
                / f"double_rod_width2_sep{separation}_depth30_5m.json"
            ),
            random_seed=24073130 + index,
        )
        metrics_dir = response_dir / "standard_metrics"
        evaluate_standard_target(
            analysis_dir=response_dir,
            config_path=config,
            out_dir=metrics_dir,
            n_bootstrap=200,
        )
        double_metric_paths.append(metrics_dir / "metrics.json")
        double_response_reports.append(response_report)
    double_report = summarize_double_rod_series(
        metrics_paths=double_metric_paths,
        out_dir=figures / "slide11_double_rods_response",
    )

    # Slide 13: D1 active-radius trade-off and physical R=150/175 validation.
    thin30_metrics_dir = thin30_response / "standard_metrics"
    evaluate_standard_target(
        analysis_dir=thin30_response,
        config_path=thin30_config,
        out_dir=thin30_metrics_dir,
        n_bootstrap=300,
    )
    sweep_dir = figures / "slide13_d1_size_response" / "sweeps"
    run_sweeps(
        analysis_dir=thin30_response,
        config_path=thin30_config,
        out_dir=sweep_dir,
    )
    posthoc_dir = figures / "slide13_d1_size_response" / "posthoc"
    size_report = evaluate_detector_size(
        sweep_csv=sweep_dir / "sweep_metrics.csv",
        config_path=thin30_config,
        out_dir=posthoc_dir,
    )
    physical_metric_paths: list[Path] = []
    physical_response_reports: list[dict[str, Any]] = []
    for index, radius in enumerate((150, 175)):
        tag = f"thin_layer_depth30_d1r{radius}_hardware_20260731"
        config, response_dir, response_report = prepare_variant(
            tag=tag,
            analysis_dir=(
                root
                / "runs"
                / f"physical_d1r{radius}_thin30_seed24061873_local"
            ),
            base_config=(
                root
                / "configs"
                / "generated"
                / f"thin_layer_depth30_d1r{radius}.json"
            ),
            random_seed=24073140 + index,
        )
        metrics_dir = response_dir / "standard_metrics"
        evaluate_standard_target(
            analysis_dir=response_dir,
            config_path=config,
            out_dir=metrics_dir,
            n_bootstrap=300,
        )
        physical_metric_paths.append(metrics_dir / "metrics.json")
        physical_response_reports.append(response_report)
    physical_size_report = validate_physical_d1_sizes(
        full_metrics_path=thin30_metrics_dir / "metrics.json",
        physical_metrics_paths=physical_metric_paths,
        posthoc_size_report_path=posthoc_dir / "d1_size_tradeoff.json",
        out_dir=figures / "slide13_d1_size_response" / "physical_validation",
    )

    # Slide 14: divide the response-applied full-concrete reconstruction by
    # detector position without changing the physical detector geometry.
    region_report = plot_detector_region_images(
        analysis_dir=concrete_response,
        config_path=hardware_config,
        out_dir=figures / "slide14_regions_response",
    )

    report = {
        "schema_version": "1.0",
        "hardware_config": str(hardware_config.resolve()),
        "core": core,
        "slide10_void": {
            "response": void_response_report,
            "evaluation": void_report,
        },
        "slide11_double_rods": {
            "responses": double_response_reports,
            "evaluation": double_report,
        },
        "slide13_d1_size": {
            "size_tradeoff": size_report,
            "physical_responses": physical_response_reports,
            "physical_validation": physical_size_report,
        },
        "slide14_regions": region_report,
    }
    with (output / "hardware_update_extended_assets.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    return report


if __name__ == "__main__":
    result = build_extended_assets(PROJECT_ROOT)
    print(result["slide10_void"]["evaluation"]["figures"][-1])
    print(result["slide11_double_rods"]["evaluation"]["figures"][-1])
    print(
        result["slide13_d1_size"]["physical_validation"]["figures"][-1]
    )
    print(result["slide14_regions"]["figures"][-1])
