from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Circle
from scipy.ndimage import gaussian_filter, gaussian_filter1d


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backscatter.voids import evaluate_concrete_void


WORK_DIR = PROJECT_ROOT / "outputs" / "ppt_void_resolution_20260807"
CONFIG_PATH = WORK_DIR / "void_d20_depth30_d1_180_250.json"
VOID_EVAL_DIR = WORK_DIR / "void_evaluation_d1_180_250"
FIGURE_DIR = WORK_DIR / "figures"

BASE_CONFIG = (
    PROJECT_ROOT
    / "outputs"
    / "hardware_update_20260731"
    / "resolved_configs"
    / "concrete_void_d20_depth30_hardware_20260731.json"
)
VOID_ANALYSIS = (
    PROJECT_ROOT
    / "outputs"
    / "hardware_update_20260731"
    / "response_variants"
    / "concrete_void_d20_depth30_hardware_20260731"
)
REFERENCE_ANALYSIS = (
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
VALIDATION_JSON = ROD_METRICS.parents[2] / "independent_gate_seed_validation.json"


def _write_updated_config() -> dict:
    with BASE_CONFIG.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    config["project"]["baseline_id"] = (
        "concrete_void_d20_depth30_hardware_response_d1_180_250"
    )
    config["selection"]["d1_energy_window_keV"] = [180.0, 250.0]
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    with CONFIG_PATH.open("w", encoding="utf-8") as handle:
        json.dump(config, handle, ensure_ascii=False, indent=2)
    return config


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


def _descriptive_profile_width(
    x: np.ndarray,
    contrast: np.ndarray,
) -> dict[str, float | None]:
    outer = ((x >= -30.0) & (x <= -18.0)) | (
        (x >= 18.0) & (x <= 30.0)
    )
    center = (x >= -12.0) & (x <= 12.0)
    baseline = float(np.median(contrast[outer]))
    center_indices = np.flatnonzero(center)
    peak_index = int(center_indices[np.argmax(contrast[center])])
    peak = float(contrast[peak_index])
    half_level = baseline + 0.5 * (peak - baseline)
    visible = np.flatnonzero((x >= -30.0) & (x <= 30.0))
    left = None
    right = None
    for index in range(peak_index, int(visible[0]), -1):
        if contrast[index - 1] <= half_level <= contrast[index]:
            left = _interpolate_crossing(
                float(x[index - 1]),
                float(contrast[index - 1]),
                float(x[index]),
                float(contrast[index]),
                half_level,
            )
            break
    for index in range(peak_index, int(visible[-1])):
        if contrast[index] >= half_level >= contrast[index + 1]:
            right = _interpolate_crossing(
                float(x[index]),
                float(contrast[index]),
                float(x[index + 1]),
                float(contrast[index + 1]),
                half_level,
            )
            break
    width = None if left is None or right is None else float(right - left)
    return {
        "baseline": baseline,
        "peak": peak,
        "peak_x_mm": float(x[peak_index]),
        "half_level": half_level,
        "left_half_max_mm": left,
        "right_half_max_mm": right,
        "descriptive_half_max_width_mm": width,
    }


def _load_map(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    frame = pd.read_csv(path)
    x = np.sort(frame["x_mm"].unique())
    depth = np.sort(frame["depth_mm"].unique())
    void = (
        frame.pivot(index="x_mm", columns="depth_mm", values="void_counts")
        .loc[x, depth]
        .to_numpy(dtype=float)
    )
    expected = (
        frame.pivot(
            index="x_mm",
            columns="depth_mm",
            values="scaled_reference_counts",
        )
        .loc[x, depth]
        .to_numpy(dtype=float)
    )
    return x, depth, void, expected


def _configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [
                "Microsoft YaHei",
                "SimHei",
                "Arial",
                "DejaVu Sans",
            ],
            "font.size": 9.0,
            "axes.titlesize": 10.0,
            "axes.labelsize": 9.0,
            "xtick.labelsize": 8.0,
            "ytick.labelsize": 8.0,
            "legend.fontsize": 7.8,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )


def _build_figure(void_report: dict) -> dict:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    metrics = void_report["predefined_primary_roi"]
    geometry = void_report["void_geometry"]
    void_radius = float(geometry["radius_mm"])
    void_depth = float(geometry["depth_mm"])
    roi_radius = float(metrics["roi_radius_mm"])

    map_path = Path(void_report["map"]["source_csv"])
    profile_path = Path(void_report["profile_source_csv"])
    x_map, depth_map, void_map, expected_map = _load_map(map_path)
    map_sigma = 1.0
    shown_void = gaussian_filter(void_map, map_sigma)
    shown_expected = gaussian_filter(expected_map, map_sigma)
    deficit_fraction = np.divide(
        shown_expected - shown_void,
        shown_expected,
        out=np.full_like(shown_expected, np.nan),
        where=shown_expected >= 4.0,
    )

    profile = pd.read_csv(profile_path)
    x_profile = profile["x_mm"].to_numpy(dtype=float)
    void_profile = profile["void_counts"].to_numpy(dtype=float)
    expected_profile = profile["scaled_reference_counts"].to_numpy(dtype=float)
    profile_sigma = 1.0
    shown_void_profile = gaussian_filter1d(void_profile, profile_sigma)
    shown_expected_profile = gaussian_filter1d(expected_profile, profile_sigma)
    contrast_profile = np.divide(
        shown_expected_profile - shown_void_profile,
        shown_expected_profile,
        out=np.full_like(shown_expected_profile, np.nan),
        where=shown_expected_profile >= 1.0,
    )
    profile_width = _descriptive_profile_width(x_profile, contrast_profile)

    with ROD_METRICS.open("r", encoding="utf-8") as handle:
        rod_metrics = json.load(handle)
    with VALIDATION_JSON.open("r", encoding="utf-8") as handle:
        validation = json.load(handle)
    rod_curve = pd.read_csv(ROD_CURVES)
    rod_curve = rod_curve[rod_curve["axis"] == "x"].copy()
    rod_x = rod_curve["coordinate_mm"].to_numpy(dtype=float)
    rod_y = rod_curve["density_normalized"].to_numpy(dtype=float)
    x_result = rod_metrics["transverse_x_response"]["estimate"]
    x_ci = rod_metrics["transverse_x_response"]["bootstrap"]["fwhm_95_ci"]

    _configure_style()
    mm_to_inch = 1.0 / 25.4
    fig = plt.figure(figsize=(264 * mm_to_inch, 82 * mm_to_inch))
    grid = fig.add_gridspec(
        1,
        3,
        width_ratios=[1.12, 1.0, 0.95],
        left=0.045,
        right=0.985,
        bottom=0.20,
        top=0.90,
        wspace=0.46,
    )
    axes = [fig.add_subplot(grid[0, index]) for index in range(3)]

    # a: localized, control-normalized deficit map.
    ax = axes[0]
    extent = [
        float(x_map[0] - 1.0),
        float(x_map[-1] + 1.0),
        float(depth_map[-1] + 1.0),
        float(depth_map[0] - 1.0),
    ]
    image = ax.imshow(
        deficit_fraction.T,
        extent=extent,
        aspect="equal",
        cmap="RdBu_r",
        norm=TwoSlopeNorm(vmin=-0.18, vcenter=0.0, vmax=0.42),
        interpolation="nearest",
        rasterized=True,
    )
    ax.add_patch(
        Circle(
            (0.0, void_depth),
            void_radius,
            fill=False,
            edgecolor="#FFFFFF",
            linewidth=1.8,
            linestyle="-",
            zorder=4,
        )
    )
    ax.add_patch(
        Circle(
            (0.0, void_depth),
            roi_radius,
            fill=False,
            edgecolor="#222222",
            linewidth=1.1,
            linestyle=(0, (4, 2)),
            zorder=4,
        )
    )
    ax.plot(0.0, void_depth, marker="+", color="#FFFFFF", ms=7, mew=1.4)
    ax.set_xlim(-35.0, 35.0)
    ax.set_ylim(54.0, 6.0)
    ax.set_xlabel("重建横向位置 x（mm）")
    ax.set_ylabel("重建深度（mm）")
    ax.set_title("参考归一化的孔洞计数缺失图")
    ax.legend(
        handles=[
            Line2D([0], [0], color="white", lw=1.8, label="真实孔洞边界：直径 20 mm"),
            Line2D(
                [0],
                [0],
                color="#222222",
                lw=1.1,
                ls=(0, (4, 2)),
                label="CNR 分析 ROI：直径 30 mm",
            ),
        ],
        loc="lower right",
        frameon=True,
        facecolor="#FFFFFF",
        framealpha=0.88,
    )
    colorbar = fig.colorbar(image, ax=ax, fraction=0.050, pad=0.025)
    colorbar.set_label("缺失比例", labelpad=2)

    # b: descriptive transverse profile through the void depth band.
    ax = axes[1]
    visible = (x_profile >= -30.0) & (x_profile <= 30.0)
    ax.axvspan(
        -void_radius,
        void_radius,
        color="#FDE6CF",
        alpha=0.95,
        label="真实孔洞横向范围",
    )
    ax.axvline(-void_radius, color="#D67A1F", lw=1.0, ls="--")
    ax.axvline(void_radius, color="#D67A1F", lw=1.0, ls="--")
    ax.axhline(0.0, color="#7A8793", lw=0.8, ls="--")
    ax.plot(
        x_profile[visible],
        contrast_profile[visible],
        color="#165A9F",
        lw=2.0,
        marker="o",
        ms=2.7,
        label="平滑后的缺失剖面",
    )
    left = profile_width["left_half_max_mm"]
    right = profile_width["right_half_max_mm"]
    half = float(profile_width["half_level"])
    width = profile_width["descriptive_half_max_width_mm"]
    if left is not None and right is not None and width is not None:
        ax.hlines(half, left, right, color="#C2413B", lw=2.2)
        ax.vlines([left, right], half - 0.012, half + 0.012, color="#C2413B", lw=1.2)
        ax.text(
            0.0,
            half + 0.018,
            f"缺失剖面半高宽 ≈ {width:.1f} mm",
            ha="center",
            va="bottom",
            color="#C2413B",
            fontsize=8.2,
            fontweight="bold",
        )
    ax.set_xlim(-30.0, 30.0)
    ax.set_ylim(-0.07, 0.31)
    ax.set_xlabel("重建横向位置 x（mm）")
    ax.set_ylabel("相对计数缺失")
    ax.set_title("孔洞中心深度带的横向剖面")
    ax.legend(loc="upper right")
    ax.text(
        0.03,
        0.04,
        "半高宽只描述孔洞影像范围，\n不等于系统空间分辨率。",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        color="#5F6B7A",
        fontsize=7.5,
        bbox={
            "boxstyle": "round,pad=0.22",
            "facecolor": "white",
            "edgecolor": "none",
            "alpha": 0.82,
        },
    )

    # c: formal transverse line-spread result from the independent rod run.
    ax = axes[2]
    visible_rod = (rod_x >= -15.0) & (rod_x <= 15.0)
    ax.fill_between(
        rod_x[visible_rod],
        0.0,
        rod_y[visible_rod],
        color="#D9E8F6",
        alpha=0.95,
    )
    ax.plot(
        rod_x[visible_rod],
        rod_y[visible_rod],
        color="#165A9F",
        lw=2.2,
    )
    left_rod = float(x_result["left_half_max"])
    right_rod = float(x_result["right_half_max"])
    ax.hlines(0.5, left_rod, right_rod, color="#C2413B", lw=2.6)
    ax.vlines(
        [left_rod, right_rod],
        0.46,
        0.54,
        color="#C2413B",
        lw=1.2,
    )
    ax.axvline(0.0, color="#7A8793", lw=0.9, ls="--")
    ax.set_xlim(-15.0, 15.0)
    ax.set_ylim(0.0, 1.08)
    ax.set_xlabel("重建横向位置 x（mm）")
    ax.set_ylabel("归一化响应")
    ax.set_title("独立 GATE 数据的横向线响应")
    ax.text(
        0.04,
        0.95,
        f"FWHM = {float(x_result['fwhm']):.2f} mm\n"
        f"95% CI：{float(x_ci[0]):.2f}–{float(x_ci[1]):.2f} mm",
        transform=ax.transAxes,
        ha="left",
        va="top",
        color="#111827",
        fontsize=8.5,
        fontweight="bold",
    )
    ax.text(
        0.96,
        0.08,
        "达标：上限 < 5 mm",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        color="#0E7490",
        fontsize=8.2,
        fontweight="bold",
    )
    ax.text(
        0.95,
        0.70,
        "2 mm 水泥细杆",
        transform=ax.transAxes,
        ha="right",
        va="center",
        color="#344054",
        fontsize=7.8,
    )

    for label, ax in zip("abc", axes):
        ax.text(
            -0.14,
            1.06,
            label,
            transform=ax.transAxes,
            fontsize=12,
            fontweight="bold",
            va="top",
        )

    fig.text(
        0.045,
        0.055,
        "孔洞图回答“缺陷是否形成局部计数缺失”；细杆线响应回答“系统横向能分多细”。两者不能用同一个 FWHM 代替。",
        ha="left",
        va="bottom",
        fontsize=8.7,
        color="#344054",
        fontweight="bold",
    )

    output_base = FIGURE_DIR / "void_imaging_with_transverse_resolution"
    figure_paths: list[str] = []
    for extension, dpi in (
        ("svg", 300),
        ("pdf", 300),
        ("tiff", 600),
        ("png", 300),
    ):
        path = output_base.with_suffix(f".{extension}")
        fig.savefig(path, dpi=dpi, facecolor="white")
        figure_paths.append(str(path.resolve()))
    plt.close(fig)

    figure_metrics = {
        "schema_version": "1.0",
        "void_selection": {
            "d1_energy_window_keV": [180.0, 250.0],
            "d2_energy_window_keV": [400.0, 600.0],
        },
        "void": {
            "diameter_mm": 2.0 * void_radius,
            "depth_mm": void_depth,
            "analysis_roi_diameter_mm": 2.0 * roi_radius,
            "contrast": float(metrics["contrast"]),
            "contrast_approximate_95_ci": metrics[
                "contrast_approximate_95_ci"
            ],
            "cnr": float(metrics["cnr"]),
            "descriptive_profile": profile_width,
            "guardrail": (
                "The profile half-maximum width describes the reconstructed "
                "void-deficit extent and is not the system point/line-spread "
                "FWHM."
            ),
        },
        "transverse_resolution": {
            "target": "2 mm-diameter concrete rod at 30 mm depth",
            "fwhm_mm": float(x_result["fwhm"]),
            "bootstrap_95_ci_mm": [float(x_ci[0]), float(x_ci[1])],
            "requirement_mm": float(validation["guardrails"]["target_fwhm_mm"]),
            "all_response_seeds_below_requirement": bool(
                validation["acceptance_checks"][
                    "all_response_seeds_below_target"
                ]
            ),
        },
        "visualization": {
            "map_gaussian_sigma_bins": map_sigma,
            "profile_gaussian_sigma_bins": profile_sigma,
            "manual_pixel_edits": False,
        },
        "source_files": {
            "void_map_csv": str(map_path.resolve()),
            "void_profile_csv": str(profile_path.resolve()),
            "void_metrics_json": str(
                (VOID_EVAL_DIR / "concrete_void_metrics.json").resolve()
            ),
            "rod_curve_csv": str(ROD_CURVES.resolve()),
            "rod_metrics_json": str(ROD_METRICS.resolve()),
            "independent_validation_json": str(VALIDATION_JSON.resolve()),
        },
        "figures": figure_paths,
    }
    with (FIGURE_DIR / "void_resolution_figure_metrics.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(figure_metrics, handle, ensure_ascii=False, indent=2)

    with (FIGURE_DIR / "void_resolution_figure_source.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "x_mm",
                "void_profile_smoothed",
                "expected_profile_smoothed",
                "relative_count_deficit",
            ]
        )
        writer.writerows(
            zip(
                x_profile,
                shown_void_profile,
                shown_expected_profile,
                contrast_profile,
            )
        )
    return figure_metrics


def main() -> None:
    _write_updated_config()
    void_report = evaluate_concrete_void(
        void_analysis_dir=VOID_ANALYSIS,
        reference_analysis_dir=REFERENCE_ANALYSIS,
        config_path=CONFIG_PATH,
        out_dir=VOID_EVAL_DIR,
        roi_radius_scale=1.5,
        map_bin_width_mm=2.0,
        smoothing_sigma_bins=1.0,
    )
    result = _build_figure(void_report)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
