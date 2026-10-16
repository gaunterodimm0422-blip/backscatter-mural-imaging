"""孔洞成像图备选显示样式原型（2026-08-30）。

只读取已有的 advanced_reconstruction_maps.npz。
不重跑重建、不修改任何正式产物、不使用孔洞真值参与任何计算或参数选择。
真值圆与真值位置十字仅作视觉叠加。
输出：outputs/figure_style_options_20260830/
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Ellipse, Rectangle

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "outputs" / "advanced_reconstruction_20260807" / "advanced_reconstruction_maps.npz"
OUT = ROOT / "outputs" / "figure_style_options_20260830"
OUT.mkdir(parents=True, exist_ok=True)

ROD_FWHM_X_MM = 3.67
ROD_FWHM_Z_MM = 38.47
XLIM = (-35.0, 35.0)
ZLIM = (55.0, 7.0)
SENS_EXPONENT = 0.5   # 泊松型权重: sigma ∝ S^-0.5

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Noto Sans CJK JP", "Microsoft YaHei", "SimHei", "DejaVu Sans"],
    "font.size": 9.0,
    "axes.titlesize": 10.5,
    "axes.labelsize": 9.5,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "axes.linewidth": 0.9,
    "legend.frameon": False,
    "mathtext.fontset": "dejavusans",
})

d = np.load(SRC)
x_edges = d["x_edges_mm"]
z_edges = d["depth_edges_mm"]
xc = 0.5 * (x_edges[:-1] + x_edges[1:])
zc = 0.5 * (z_edges[:-1] + z_edges[1:])
X, Z = np.meshgrid(xc, zc)

S = d["sensitivity_per_primary"]
V = d["accepted_tv_deficit"]
C = d["accepted_tv_homogeneous_control_deficit"]

rowmax = np.nanmax(np.where(np.isfinite(S), S, -np.inf), axis=1, keepdims=True)
fov = np.isfinite(S) & (S >= 0.08 * rowmax)
win = (np.abs(X) <= 36) & (Z >= 6) & (Z <= 56)
mask = fov & np.isfinite(V) & np.isfinite(C) & win

Vm = np.where(mask, V, np.nan)
Cm = np.where(mask, C, np.nan)

# 噪声模型只由完整水泥独立留出组标定，不使用孔洞数据、不使用孔洞真值。
outside = mask & ((X ** 2 + (Z - 30.0) ** 2) > 400.0)
base = float(np.nanmedian(C[outside]))
shape = np.where(S > 0, S, np.nan) ** (-SENS_EXPONENT)
scale = float(np.nanstd(((C - base) / shape)[outside]))
sigma = shape * scale
Zv = (Vm - base) / sigma
Zc = (Cm - base) / sigma

roi = mask & ((X ** 2 + (Z - 30.0) ** 2) <= 100.0)
stats = {
    "sigma_median": float(np.nanmedian(sigma[mask])),
    "void_z_max": float(np.nanmax(Zv[mask])),
    "void_z_roi_mean": float(np.nanmean(Zv[roi])),
    "ctrl_z_max": float(np.nanmax(Zc[mask])),
    "ctrl_z_p99": float(np.nanpercentile(Zc[mask], 99)),
    "ctrl_z_roi_mean": float(np.nanmean(Zc[roi])),
}
for k, v in stats.items():
    print(f"{k:18s} {v: .4f}")

EXTENT = [x_edges[0], x_edges[-1], z_edges[-1], z_edges[0]]


def frame(ax, title, sub):
    ax.set_xlim(*XLIM)
    ax.set_ylim(*ZLIM)
    ax.set_xlabel("横向位置 x（mm）")
    ax.set_ylabel("深度（mm）")
    ax.set_title(title, loc="left", pad=20, fontweight="bold")
    ax.text(0.0, 1.012, sub, transform=ax.transAxes, fontsize=8.0,
            color="#4B5563", va="bottom", ha="left")


def nodata(ax):
    ax.add_patch(Rectangle((XLIM[0], ZLIM[1]), XLIM[1] - XLIM[0], ZLIM[0] - ZLIM[1],
                           facecolor="#EDF0F3", edgecolor="none", zorder=-5))
    ax.contourf(xc, zc, mask.astype(float), levels=[-0.5, 0.5],
                colors="none", hatches=["////"], zorder=-4)
    ax.contour(xc, zc, mask.astype(float), levels=[0.5],
               colors="#6B7280", linewidths=1.0, linestyles="dashed", zorder=4)


def truth(ax, color="#111827"):
    ax.add_patch(Circle((0.0, 30.0), 10.0, fill=False, edgecolor=color,
                        linewidth=1.6, linestyle=(0, (4, 2)), zorder=6))


def psf_marker(ax):
    ax.add_patch(Rectangle((-34.0, 53.0), 19.0, -22.0, facecolor="white",
                           alpha=0.82, edgecolor="#9CA3AF", linewidth=0.6, zorder=6))
    cx, cz = -31.5, 42.0
    ax.add_patch(Ellipse((cx, cz), ROD_FWHM_X_MM, ROD_FWHM_Z_MM, facecolor="#374151",
                         alpha=0.30, edgecolor="#111827", linewidth=1.0, zorder=7))
    ax.text(cx + 3.0, cz, "独立细杆测得的\n系统响应 FWHM\n横向 3.7 mm\n深度 38.5 mm",
            fontsize=6.8, va="center", ha="left", color="#111827", zorder=8)


fig, axes = plt.subplots(2, 2, figsize=(11.8, 9.4))

# ---- A 当前样式 ------------------------------------------------------
ax = axes[0, 0]
cmA = matplotlib.colormaps["magma"].copy()
cmA.set_bad("#F1F5F9")
imA = ax.imshow(Vm, extent=EXTENT, cmap=cmA, vmin=-0.06, vmax=0.28,
                interpolation="bilinear", aspect="equal")
ax.contour(xc, zc, np.where(np.isfinite(Vm), Vm, -np.inf), levels=[0.08, 0.12, 0.16],
           colors=["#FFF0D5", "#FFFFFF", "#77E2F2"], linewidths=[0.7, 0.9, 1.0])
truth(ax, "white")
frame(ax, "A  当前样式：顺序色标 magma + 双线性插值",
      "零点不可辨识；大片黑色其实是 D 约等于 0，不是无数据；色标无量名")
cb = fig.colorbar(imA, ax=ax, fraction=0.046, pad=0.03)
cb.set_label("相对计数缺失 D", fontsize=8.6)

# ---- B 发散色标 ------------------------------------------------------
ax = axes[0, 1]
cmB = matplotlib.colormaps["RdBu_r"].copy()
cmB.set_bad((0, 0, 0, 0))
imB = ax.imshow(Vm, extent=EXTENT, cmap=cmB, vmin=-0.25, vmax=0.25,
                interpolation="nearest", aspect="equal", zorder=1)
nodata(ax)
truth(ax)
psf_marker(ax)
ax.plot(0.5, 31.0, marker="+", ms=9, mew=1.6, color="#111827", zorder=7)
frame(ax, "B  发散色标、零点居中、原始像素、系统响应标尺",
      "白=D约等于0，红=计数缺失，蓝=计数偏多；斜线区为有效视野以外，未计算")
cb = fig.colorbar(imB, ax=ax, fraction=0.046, pad=0.03,
                  ticks=[-0.25, -0.125, 0.0, 0.125, 0.25])
cb.set_label(r"相对计数缺失 $D = 1 - R_{\mathrm{norm}}$", fontsize=8.6)

# ---- C 显著性图（孔洞） ---------------------------------------------
ax = axes[1, 0]
imC = ax.imshow(Zv, extent=EXTENT, cmap=cmB, vmin=-3.0, vmax=3.0,
                interpolation="nearest", aspect="equal", zorder=1)
cs = ax.contour(xc, zc, np.where(np.isfinite(Zv), Zv, 0.0), levels=[2.0, 2.5],
                colors="#111827", linewidths=[0.8, 1.2], zorder=5)
ax.clabel(cs, fmt={2.0: "2σ", 2.5: "2.5σ"}, fontsize=7.4, inline=True)
nodata(ax)
truth(ax)
frame(ax, "C  显著性图（孔洞数据）",
      f"σ 由完整水泥留出组按 S^-1/2 标定；峰值 {stats['void_z_max']:.1f}σ，ROI 平均 {stats['void_z_roi_mean']:.1f}σ")
cb = fig.colorbar(imC, ax=ax, fraction=0.046, pad=0.03)
cb.set_label(r"标准化偏离 $z = (D - D_0)/\sigma(x,d)$", fontsize=8.6)

# ---- D 显著性图（负对照） -------------------------------------------
ax = axes[1, 1]
imD = ax.imshow(Zc, extent=EXTENT, cmap=cmB, vmin=-3.0, vmax=3.0,
                interpolation="nearest", aspect="equal", zorder=1)
cs = ax.contour(xc, zc, np.where(np.isfinite(Zc), Zc, 0.0), levels=[2.0, 2.5],
                colors="#111827", linewidths=[0.8, 1.2], zorder=5)
ax.clabel(cs, fmt={2.0: "2σ", 2.5: "2.5σ"}, fontsize=7.4, inline=True)
nodata(ax)
truth(ax)
frame(ax, "D  负对照：完整水泥独立留出组，同一处理链",
      f"孔洞位置平均仅 {stats['ctrl_z_roi_mean']:.1f}σ；"
      f"但场边缘仍有 {stats['ctrl_z_max']:.1f}σ 的残余结构")
cb = fig.colorbar(imD, ax=ax, fraction=0.046, pad=0.03)
cb.set_label(r"标准化偏离 $z$", fontsize=8.6)

fig.suptitle("孔洞成像显示方案对比原型：仅改变呈现方式，未改变重建数据",
             fontsize=12.5, fontweight="bold", y=0.985)
fig.tight_layout(rect=(0, 0, 1, 0.955))
for ext, dpi in (("png", 220), ("pdf", 300), ("svg", 300)):
    fig.savefig(OUT / f"void_style_options.{ext}", dpi=dpi, bbox_inches="tight",
                facecolor="white")

with open(OUT / "style_option_metrics.txt", "w", encoding="utf-8") as fh:
    fh.write("孔洞成像显示方案原型指标（不替代 advanced_reconstruction_report.json）\n")
    fh.write(f"噪声模型: sigma(x,d) = k * S(x,d)^-{SENS_EXPONENT}, k 由完整水泥留出组标定\n")
    for k, v in stats.items():
        fh.write(f"{k} = {v:.4f}\n")
print("written to", OUT)
