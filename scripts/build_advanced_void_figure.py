from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Circle


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORK_DIR = PROJECT_ROOT / "outputs" / "advanced_reconstruction_20260807"
MAPS_PATH = WORK_DIR / "advanced_reconstruction_maps.npz"
REPORT_PATH = WORK_DIR / "advanced_reconstruction_report.json"
ROD_CURVES = (
    PROJECT_ROOT
    / "outputs"
    / "transverse_fwhm_5mm_20260805"
    / "independent_gate_seed"
    / "rod_width2_depth30_5m_seed24080573_local_fwhm5_validation"
    / "canonical_response"
    / "standard_metrics"
    / "rod_response_curves.csv"
)


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {name: data[name] for name in data.files}


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [
                "Microsoft YaHei",
                "SimHei",
                "Arial",
                "DejaVu Sans",
            ],
            "font.size": 8.4,
            "axes.titlesize": 10.2,
            "axes.labelsize": 8.7,
            "xtick.labelsize": 7.7,
            "ytick.labelsize": 7.7,
            "axes.linewidth": 0.85,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )


def _export(fig: plt.Figure, stem: str) -> list[str]:
    paths: list[str] = []
    for extension, dpi in (
        ("svg", 300),
        ("pdf", 300),
        ("tiff", 600),
        ("png", 300),
    ):
        path = WORK_DIR / f"{stem}.{extension}"
        fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
        paths.append(str(path.resolve()))
    return paths


def _imshow_deficit(
    ax: plt.Axes,
    deficit: np.ndarray,
    x_edges: np.ndarray,
    depth_edges: np.ndarray,
    *,
    title: str,
    show_truth: bool,
    centroid: tuple[float, float] | None = None,
) -> matplotlib.image.AxesImage:
    cmap = matplotlib.colormaps["magma"].copy()
    cmap.set_bad("#F1F5F9")
    image = ax.imshow(
        deficit,
        extent=[x_edges[0], x_edges[-1], depth_edges[-1], depth_edges[0]],
        cmap=cmap,
        vmin=-0.06,
        vmax=0.28,
        interpolation="bilinear",
        aspect="equal",
        rasterized=True,
    )
    finite = np.where(np.isfinite(deficit), deficit, -np.inf)
    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    depth_centers = 0.5 * (depth_edges[:-1] + depth_edges[1:])
    try:
        ax.contour(
            x_centers,
            depth_centers,
            finite,
            levels=[0.08, 0.12, 0.16],
            colors=["#FFF0D5", "#FFFFFF", "#77E2F2"],
            linewidths=[0.7, 0.9, 1.0],
            alpha=0.88,
        )
    except ValueError:
        pass
    support = np.isfinite(deficit).astype(np.float64)
    try:
        ax.contour(
            x_centers,
            depth_centers,
            support,
            levels=[0.5],
            colors=["#64748B"],
            linewidths=[0.9],
            linestyles=[(0, (3, 2))],
        )
    except ValueError:
        pass
    if show_truth:
        ax.add_patch(
            Circle(
                (0.0, 30.0),
                10.0,
                fill=False,
                edgecolor="white",
                linewidth=1.5,
                linestyle=(0, (4, 2)),
                zorder=5,
            )
        )
    if centroid is not None:
        ax.plot(
            centroid[0],
            centroid[1],
            marker="+",
            markersize=8,
            markeredgewidth=1.4,
            color="#6EE7F2",
            zorder=6,
        )
    ax.set_xlim(-35.0, 35.0)
    ax.set_ylim(55.0, 7.0)
    ax.set_xlabel("重建横向位置 x（mm）")
    ax.set_ylabel("重建深度（mm）")
    ax.set_title(title, loc="left", fontweight="bold", pad=6)
    return image


def _panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.13,
        1.06,
        label,
        transform=ax.transAxes,
        fontsize=12,
        fontweight="bold",
        va="top",
        color="#111827",
    )


def _nanmean_or_nan(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    return float(np.mean(finite)) if finite.size else float("nan")


def build_hero_figure(
    arrays: dict[str, np.ndarray],
    report: dict,
) -> dict[str, object]:
    _style()
    x_edges = arrays["x_edges_mm"]
    depth_edges = arrays["depth_edges_mm"]
    x = 0.5 * (x_edges[:-1] + x_edges[1:])
    depth = 0.5 * (depth_edges[:-1] + depth_edges[1:])
    baseline = arrays["baseline_deficit"]
    final = arrays["accepted_tv_deficit"]
    control = arrays["accepted_tv_homogeneous_control_deficit"]

    metrics = report["metrics"]["accepted_tv_void"]
    control_metrics = report["metrics"]["accepted_tv_homogeneous_control"]
    baseline_metrics = report["metrics"]["closed_form_baseline"]
    rod = report["metrics"]["independent_2mm_rod_closed_form_guardrail"]
    centroid = (
        float(metrics["positive_deficit_centroid_x_mm"]),
        float(metrics["positive_deficit_centroid_depth_mm"]),
    )
    profile = np.asarray(
        [
            _nanmean_or_nan(final[(depth >= 25.0) & (depth <= 35.0), index])
            for index in range(len(x))
        ],
        dtype=np.float64,
    )
    baseline_profile = np.asarray(
        [
            _nanmean_or_nan(baseline[(depth >= 25.0) & (depth <= 35.0), index])
            for index in range(len(x))
        ],
        dtype=np.float64,
    )
    depth_profile = np.asarray(
        [
            _nanmean_or_nan(final[index, (x >= -5.0) & (x <= 5.0)])
            for index in range(len(depth))
        ],
        dtype=np.float64,
    )
    control_depth_profile = np.asarray(
        [
            _nanmean_or_nan(control[index, (x >= -5.0) & (x <= 5.0)])
            for index in range(len(depth))
        ],
        dtype=np.float64,
    )

    mm_to_inch = 1.0 / 25.4
    fig = plt.figure(figsize=(264 * mm_to_inch, 111 * mm_to_inch))
    grid = fig.add_gridspec(
        2,
        4,
        width_ratios=[1.25, 0.78, 0.88, 0.88],
        height_ratios=[1.0, 1.0],
        left=0.048,
        right=0.985,
        bottom=0.15,
        top=0.93,
        wspace=0.48,
        hspace=0.48,
    )
    ax_main = fig.add_subplot(grid[:, 0:2])
    ax_horizontal = fig.add_subplot(grid[0, 2:4])
    ax_control = fig.add_subplot(grid[1, 2])
    ax_depth = fig.add_subplot(grid[1, 3])

    image = _imshow_deficit(
        ax_main,
        final,
        x_edges,
        depth_edges,
        title="20 mm 孔洞相对计数缺失图（梯形为有效视野）",
        show_truth=True,
        centroid=centroid,
    )
    _panel_label(ax_main, "a")
    colorbar = fig.colorbar(image, ax=ax_main, fraction=0.042, pad=0.025)
    colorbar.set_label("相对计数密度缺失")
    ax_main.legend(
        handles=[
            Line2D(
                [0],
                [0],
                color="white",
                lw=1.5,
                ls=(0, (4, 2)),
                label="孔洞真值边界",
            ),
            Line2D(
                [0],
                [0],
                color="#6EE7F2",
                marker="+",
                lw=0,
                markersize=7,
                label="缺失重心",
            ),
            Line2D(
                [0],
                [0],
                color="#94A3B8",
                lw=1.0,
                ls=(0, (3, 2)),
                label="有效视野边界",
            ),
        ],
        loc="lower right",
        frameon=True,
        facecolor="#160B28",
        edgecolor="none",
        labelcolor="white",
        framealpha=0.82,
        fontsize=7.2,
    )
    ax_main.text(
        0.015,
        0.985,
        "灰白区：完整水泥参考统计不足，未参与计算",
        transform=ax_main.transAxes,
        ha="left",
        va="top",
        color="#475569",
        fontsize=7.2,
        bbox={
            "boxstyle": "round,pad=0.24",
            "facecolor": "white",
            "edgecolor": "#CBD5E1",
            "alpha": 0.92,
        },
    )

    ax = ax_horizontal
    visible = (x >= -30.0) & (x <= 30.0)
    ax.axvspan(-10.0, 10.0, color="#F59E0B", alpha=0.13)
    ax.axvline(-10.0, color="#D97706", lw=0.9, ls="--")
    ax.axvline(10.0, color="#D97706", lw=0.9, ls="--")
    ax.axhline(0.0, color="#94A3B8", lw=0.8, ls="--")
    ax.plot(
        x[visible],
        baseline_profile[visible],
        color="#94A3B8",
        lw=1.2,
        label="原闭式图",
    )
    ax.plot(
        x[visible],
        profile[visible],
        color="#0F4D92",
        lw=2.2,
        label="TV 正则化",
    )
    width = metrics["profile_width"]["apparent_half_max_width_mm"]
    left = metrics["profile_width"]["left_half_max_mm"]
    right = metrics["profile_width"]["right_half_max_mm"]
    half = metrics["profile_width"]["half_level"]
    if all(value is not None for value in (width, left, right, half)):
        ax.hlines(half, left, right, color="#B64342", lw=2.5)
        ax.vlines([left, right], half - 0.012, half + 0.012, color="#B64342")
        ax.text(
            0.0,
            half + 0.018,
            f"表观半高宽 {width:.1f} mm（真值直径 20 mm）",
            ha="center",
            va="bottom",
            color="#B64342",
            fontweight="bold",
            fontsize=8.2,
        )
    ax.set_xlim(-30.0, 30.0)
    ax.set_ylim(-0.09, 0.30)
    ax.set_xlabel("横向位置 x（mm）")
    ax.set_ylabel("相对计数缺失")
    ax.set_title("横向剖面保留孔洞边缘，并抑制随机斑点", loc="left", fontweight="bold")
    ax.legend(loc="upper right", ncols=2)
    _panel_label(ax, "b")

    _imshow_deficit(
        ax_control,
        control,
        x_edges,
        depth_edges,
        title="完整水泥留出组",
        show_truth=False,
    )
    ax_control.set_xlim(-30.0, 30.0)
    ax_control.set_ylim(52.0, 10.0)
    ax_control.set_xlabel("x（mm）")
    ax_control.set_ylabel("深度（mm）")
    ax_control.text(
        0.04,
        0.04,
        f"同位置假缺失 {100*float(control_metrics['image_domain_contrast']):.1f}%",
        transform=ax_control.transAxes,
        ha="left",
        va="bottom",
        color="#FFFFFF",
        fontsize=7.4,
        fontweight="bold",
        bbox={
            "boxstyle": "round,pad=0.25",
            "facecolor": "#111827",
            "edgecolor": "none",
            "alpha": 0.76,
        },
    )
    _panel_label(ax_control, "c")

    ax = ax_depth
    visible_depth = (depth >= 5.0) & (depth <= 70.0)
    ax.axvspan(20.0, 40.0, color="#F59E0B", alpha=0.13)
    ax.plot(
        depth[visible_depth],
        depth_profile[visible_depth],
        color="#0F4D92",
        lw=2.1,
        label="孔洞数据",
    )
    ax.plot(
        depth[visible_depth],
        control_depth_profile[visible_depth],
        color="#94A3B8",
        lw=1.2,
        label="完整水泥",
    )
    ax.axhline(0.0, color="#94A3B8", lw=0.8, ls="--")
    ax.axvline(30.0, color="#D97706", lw=0.9, ls="--")
    ax.set_xlim(5.0, 70.0)
    ax.set_ylim(-0.09, 0.30)
    ax.set_xlabel("深度（mm）")
    ax.set_ylabel("相对计数缺失")
    ax.set_title("深度方向仍受能量分辨率限制", loc="left", fontweight="bold")
    ax.legend(loc="upper right")
    ax.text(
        0.04,
        0.06,
        "独立细杆：\n"
        f"横向 FWHM {float(rod['fwhm_mm']):.2f} mm\n"
        "深度 FWHM 38.47 mm",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        color="#334155",
        fontsize=7.4,
    )
    _panel_label(ax, "d")

    before_sep = float(baseline_metrics["image_domain_standardized_separation"])
    after_sep = float(metrics["image_domain_standardized_separation"])
    fig.text(
        0.048,
        0.045,
        "结果：TV 正则化将孔洞与邻近背景的图像域标准化分离度由 "
        f"{before_sep:.2f} 提高到 {after_sep:.2f}；缺失重心为 "
        f"({centroid[0]:.1f}, {centroid[1]:.1f}) mm。"
        "图像能较好恢复横向尺寸，但深度拉长仍是真实系统限制。",
        ha="left",
        va="bottom",
        fontsize=8.6,
        color="#334155",
        fontweight="bold",
    )

    paths = _export(fig, "advanced_void_imaging_hero")
    plt.close(fig)
    source_path = WORK_DIR / "advanced_void_figure_source.csv"
    with source_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "x_mm",
                "baseline_horizontal_deficit",
                "tv_horizontal_deficit",
                "depth_mm",
                "tv_depth_deficit",
                "homogeneous_depth_deficit",
            ]
        )
        length = max(len(x), len(depth))
        for index in range(length):
            writer.writerow(
                [
                    float(x[index]) if index < len(x) else "",
                    float(baseline_profile[index]) if index < len(x) else "",
                    float(profile[index]) if index < len(x) else "",
                    float(depth[index]) if index < len(depth) else "",
                    float(depth_profile[index]) if index < len(depth) else "",
                    float(control_depth_profile[index])
                    if index < len(depth)
                    else "",
                ]
            )
    return {
        "figures": paths,
        "source_csv": str(source_path.resolve()),
        "claim": (
            "TV regularization improves localized void/background separation "
            "while preserving the independent transverse FWHM; axial elongation remains."
        ),
    }


def build_method_screening_figure(
    arrays: dict[str, np.ndarray],
    report: dict,
) -> dict[str, object]:
    _style()
    x_edges = arrays["x_edges_mm"]
    depth_edges = arrays["depth_edges_mm"]
    panels = [
        (
            "闭式点重建",
            arrays["baseline_deficit"],
            report["metrics"]["closed_form_baseline"],
            "基线",
        ),
        (
            "概率射线反投影",
            arrays["psf_matched_probability_deficit"],
            report["metrics"]["accepted_psf_matched_void"],
            "深度条纹",
        ),
        (
            "LM-MLEM + TV",
            arrays["advanced_deficit"],
            report["metrics"]["advanced_void"],
            "留出组噪声放大",
        ),
        (
            "闭式重建 + TV",
            arrays["accepted_tv_deficit"],
            report["metrics"]["accepted_tv_void"],
            "当前采用",
        ),
    ]
    mm_to_inch = 1.0 / 25.4
    fig = plt.figure(figsize=(264 * mm_to_inch, 68 * mm_to_inch))
    grid = fig.add_gridspec(
        2,
        5,
        width_ratios=[1.0, 1.0, 1.0, 1.0, 0.055],
        height_ratios=[1.0, 0.18],
        left=0.045,
        right=0.955,
        bottom=0.16,
        top=0.88,
        wspace=0.34,
        hspace=0.48,
    )
    axes = [fig.add_subplot(grid[0, index]) for index in range(4)]
    summary_axes = [fig.add_subplot(grid[1, index]) for index in range(4)]
    colorbar_axis = fig.add_subplot(grid[0, 4])
    image = None
    for label, ax, summary_ax, (title, data, metrics, verdict) in zip(
        "abcd",
        axes,
        summary_axes,
        panels,
    ):
        image = _imshow_deficit(
            ax,
            data,
            x_edges,
            depth_edges,
            title=title,
            show_truth=True,
        )
        ax.set_xlim(-34.0, 34.0)
        ax.set_ylim(54.0, 8.0)
        ax.set_xlabel("横向位置 x（mm）", labelpad=2)
        ax.set_ylabel("深度（mm）")
        separation = metrics["image_domain_standardized_separation"]
        _panel_label(ax, label)

        selected = verdict == "当前采用"
        summary_ax.set_facecolor("#EAF3FF" if selected else "#F8FAFC")
        for spine in summary_ax.spines.values():
            spine.set_visible(False)
        summary_ax.set_xticks([])
        summary_ax.set_yticks([])
        summary_ax.text(
            0.5,
            0.66,
            f"分离度 {float(separation):.2f}",
            transform=summary_ax.transAxes,
            ha="center",
            va="center",
            fontsize=8.0,
            color="#0F4D92" if selected else "#334155",
            fontweight="bold",
        )
        summary_ax.text(
            0.5,
            0.24,
            verdict,
            transform=summary_ax.transAxes,
            ha="center",
            va="center",
            fontsize=7.3,
            color="#0F4D92" if selected else "#64748B",
            fontweight="bold" if selected else "normal",
        )
    assert image is not None
    colorbar = fig.colorbar(image, cax=colorbar_axis)
    colorbar.set_label("相对计数密度缺失")
    fig.text(
        0.045,
        0.035,
        "算法筛选结论：概率模型能够降低随机斑点，但 8% 能量分辨率使射线方向出现深度条纹；"
        "当前事件量下 LM-MLEM 放大了参考留出组伪影。闭式物理定位加 TV 正则化在孔洞可见性、负对照和 5 mm 横向分辨率之间最稳健。",
        ha="left",
        va="bottom",
        fontsize=7.8,
        color="#334155",
    )
    paths = _export(fig, "advanced_algorithm_screening")
    plt.close(fig)
    return {"figures": paths}


def main() -> None:
    arrays = _load_npz(MAPS_PATH)
    with REPORT_PATH.open("r", encoding="utf-8") as handle:
        report = json.load(handle)
    result = {
        "hero": build_hero_figure(arrays, report),
        "screening": build_method_screening_figure(arrays, report),
        "image_integrity": {
            "manual_pixel_edits": False,
            "global_processing": "2 mm binning, recorded Gaussian KDE baseline, globally applied TV regularization",
            "truth_overlay": "20 mm-diameter circle used only as a dashed visual overlay and not in reconstruction or parameter selection",
        },
    }
    with (WORK_DIR / "advanced_figure_manifest.json").open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
