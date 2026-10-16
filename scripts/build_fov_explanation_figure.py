from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Polygon, Rectangle


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORK_DIR = PROJECT_ROOT / "outputs" / "advanced_reconstruction_20260807"
MAPS_PATH = WORK_DIR / "advanced_reconstruction_maps.npz"
CONFIG_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "ppt_void_resolution_20260807"
    / "void_d20_depth30_d1_180_250.json"
)


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Microsoft YaHei", "SimHei", "Arial", "DejaVu Sans"],
            "font.size": 8.5,
            "axes.titlesize": 10.5,
            "axes.labelsize": 8.8,
            "xtick.labelsize": 7.8,
            "ytick.labelsize": 7.8,
            "axes.linewidth": 0.85,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )


def _export(fig: plt.Figure, stem: str) -> list[str]:
    result: list[str] = []
    for suffix, dpi in (("svg", 300), ("pdf", 300), ("tiff", 600), ("png", 300)):
        output = WORK_DIR / f"{stem}.{suffix}"
        fig.savefig(output, dpi=dpi, bbox_inches="tight", facecolor="white")
        result.append(str(output.resolve()))
    return result


def _panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.08,
        1.04,
        label,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=12,
        fontweight="bold",
        color="#111827",
    )


def main() -> None:
    _style()
    with np.load(MAPS_PATH, allow_pickle=False) as arrays:
        x_edges = arrays["x_edges_mm"]
        depth_edges = arrays["depth_edges_mm"]
        sensitivity = arrays["sensitivity_per_primary"]
    with CONFIG_PATH.open("r", encoding="utf-8") as handle:
        config = json.load(handle)

    x = 0.5 * (x_edges[:-1] + x_edges[1:])
    depth = 0.5 * (depth_edges[:-1] + depth_edges[1:])
    depth_max = np.max(sensitivity, axis=1, keepdims=True)
    relative = np.divide(
        sensitivity,
        depth_max,
        out=np.zeros_like(sensitivity),
        where=depth_max > 0.0,
    )
    support = relative >= 0.08

    width_rows: list[tuple[float, float, float, float]] = []
    for index, depth_value in enumerate(depth):
        valid_x = x[support[index]]
        if valid_x.size:
            step = float(np.median(np.diff(x)))
            width_rows.append(
                (
                    float(depth_value),
                    float(valid_x.min()),
                    float(valid_x.max()),
                    float(valid_x.max() - valid_x.min() + step),
                )
            )

    source_z = float(config["source"]["position_mm"][2])
    d2_z = float(config["detectors"]["d2"]["center_z_mm"])
    d2_half = float(config["detectors"]["d2"]["envelope_half_width_mm"])
    source_depth = -source_z
    d2_depth = -d2_z
    shown_depth = 70.0
    outer_x = (shown_depth + source_z) * d2_half / (d2_z - source_z)

    mm_to_inch = 1.0 / 25.4
    fig, (ax_geometry, ax_support) = plt.subplots(
        1,
        2,
        figsize=(183 * mm_to_inch, 92 * mm_to_inch),
        gridspec_kw={"width_ratios": [0.92, 1.18], "wspace": 0.34, "left": 0.075, "right": 0.965, "bottom": 0.26, "top": 0.79},
    )

    # a — the finite D2 plane tags a fan of object-going rays.
    ax = ax_geometry
    ax.add_patch(Rectangle((-70, 0), 140, 75, facecolor="#E2E8F0", edgecolor="#94A3B8", lw=0.9))
    ax.text(-66, 7, "水泥板", color="#475569", fontweight="bold")
    ax.axhline(0.0, color="#334155", lw=1.1)
    ax.text(68, -2.5, "表面", ha="right", va="bottom", color="#334155")
    ax.add_patch(Rectangle((-d2_half, d2_depth - 2.2), 2 * d2_half, 4.4, facecolor="#F59E0B", edgecolor="#B45309", lw=0.9))
    ax.text(0, d2_depth - 5.0, "D2 标记探测器（50 mm 宽）", ha="center", va="bottom", color="#92400E", fontweight="bold")
    ax.add_patch(Circle((0, source_depth), 2.8, facecolor="#C2413B", edgecolor="white", lw=0.8, zorder=6))
    ax.text(0, source_depth, "S", color="white", ha="center", va="center", fontweight="bold", zorder=7)
    ax.add_patch(
        Polygon(
            [(0, source_depth), (-outer_x, shown_depth), (outer_x, shown_depth)],
            closed=True,
            facecolor="#3B82F6",
            edgecolor="none",
            alpha=0.11,
        )
    )
    for d2_x in (-25.0, -12.5, 0.0, 12.5, 25.0):
        object_x = -(shown_depth + source_z) * d2_x / (d2_z - source_z)
        ax.plot([d2_x, 0.0, object_x], [d2_depth, source_depth, shown_depth], color="#2563EB", lw=0.9, alpha=0.65)
    ax.annotate(
        "D2 击中位置不同\n→ 反向光子进入水泥的方向不同",
        xy=(30, 48),
        xytext=(-61, 44),
        arrowprops={"arrowstyle": "->", "color": "#0F4D92", "lw": 1.0},
        color="#0F4D92",
        fontsize=8.2,
        fontweight="bold",
    )
    ax.text(65, 68, "射线随深度展开\n可测横向范围逐渐变宽", ha="right", va="bottom", color="#334155", fontsize=7.8)
    ax.set_xlim(-72, 72)
    ax.set_ylim(73, -68)
    ax.set_xlabel("横向位置 x（mm）")
    ax.set_ylabel("相对水泥表面的深度（mm）")
    ax.set_title("几何原因：有限 D2 面积形成扇束有效视野", loc="left", fontweight="bold")
    ax.set_aspect("equal", adjustable="box")
    _panel_label(ax, "a")

    # b — the exact support used by the reconstruction.
    ax = ax_support
    image = ax.imshow(
        relative,
        extent=[x_edges[0], x_edges[-1], depth_edges[-1], depth_edges[0]],
        cmap="Blues",
        vmin=0.0,
        vmax=1.0,
        interpolation="nearest",
        aspect="auto",
        rasterized=True,
    )
    ax.contour(x, depth, relative, levels=[0.08], colors=["#C2413B"], linewidths=[1.5], linestyles=[(0, (4, 2))])
    outside = np.ma.masked_where(support, np.ones_like(relative))
    ax.imshow(
        outside,
        extent=[x_edges[0], x_edges[-1], depth_edges[-1], depth_edges[0]],
        cmap=matplotlib.colors.ListedColormap(["#F1F5F9"]),
        vmin=0.0,
        vmax=1.0,
        interpolation="nearest",
        aspect="auto",
        alpha=0.82,
        rasterized=True,
    )
    ax.add_patch(Circle((0.0, 30.0), 10.0, fill=False, edgecolor="#F59E0B", lw=1.5, ls=(0, (4, 2))))
    for target_depth, label_y in ((9.0, 10.0), (29.0, 30.0), (49.0, 50.0)):
        idx = int(np.argmin(np.abs(depth - target_depth)))
        valid_x = x[support[idx]]
        left = float(valid_x.min())
        right = float(valid_x.max())
        width = right - left + float(np.median(np.diff(x)))
        ax.hlines(depth[idx], left, right, color="#0F4D92", lw=2.0)
        ax.text(right + 3.0, label_y, f"{width:.0f} mm", va="center", color="#0F4D92", fontsize=7.8, fontweight="bold")
    ax.text(
        0.025,
        0.03,
        "红色虚线：相对灵敏度 8% 阈值\n灰白区：参考统计不足，不做比值、不外推\n阶梯边缘：2 mm 重建网格",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        color="#334155",
        fontsize=7.6,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": "#CBD5E1", "alpha": 0.94},
    )
    ax.set_xlim(-62, 62)
    ax.set_ylim(60, 0)
    ax.set_xlabel("横向位置 x（mm）")
    ax.set_ylabel("深度（mm）")
    ax.set_title("数据原因：只保留完整水泥参考统计可靠的像素", loc="left", fontweight="bold")
    _panel_label(ax, "b")
    colorbar = fig.colorbar(image, ax=ax, fraction=0.045, pad=0.025)
    colorbar.set_label("该深度内的相对灵敏度")

    fig.suptitle("孔洞图为什么呈阶梯梯形：这是有效视野，不是缺陷外形", x=0.075, y=0.975, ha="left", fontsize=12.2, fontweight="bold", color="#111827")
    fig.text(
        0.075,
        0.055,
        "读图要点：真正与孔洞有关的是深度约 30 mm、x≈0 mm 附近的局部计数缺失；外侧梯形只说明哪些位置有足够参考事件可供可靠比较。",
        ha="left",
        va="bottom",
        fontsize=8.2,
        color="#334155",
        fontweight="bold",
    )

    outputs = _export(fig, "fov_trapezoid_explanation")
    plt.close(fig)

    csv_path = WORK_DIR / "fov_support_widths.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["depth_mm", "support_x_min_mm", "support_x_max_mm", "support_width_mm", "relative_sensitivity_threshold"])
        for row in width_rows:
            writer.writerow([*row, 0.08])
    print(json.dumps({"figure_files": outputs, "source_csv": str(csv_path.resolve())}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
