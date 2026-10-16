"""按旧版 concrete_void_detectability 的克制风格重绘当前最优重建结果。

只读取 advanced_reconstruction_maps.npz，不重跑重建、不覆盖任何正式产物。
风格规则：无加粗结论标题、无彩色信息栏、无叠加箭头、低标注密度、
每个面板只有一个物理量和一个带单位的色标。
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "outputs" / "advanced_reconstruction_20260807" / "advanced_reconstruction_maps.npz"
OUT = ROOT / "outputs" / "figure_style_options_20260830"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Noto Sans CJK JP", "Microsoft YaHei", "SimHei", "DejaVu Sans"],
    "font.size": 10.0,
    "axes.titlesize": 10.5,
    "axes.labelsize": 10.0,
    "xtick.labelsize": 9.0,
    "ytick.labelsize": 9.0,
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
    "mathtext.fontset": "dejavusans",
})

d = np.load(SRC)
x_edges, z_edges = d["x_edges_mm"], d["depth_edges_mm"]
xc = 0.5 * (x_edges[:-1] + x_edges[1:])
zc = 0.5 * (z_edges[:-1] + z_edges[1:])
X, Z = np.meshgrid(xc, zc)

S = d["sensitivity_per_primary"]
Dv = d["accepted_tv_deficit"]
Nv = d["accepted_tv_normalized_density"]
Nr = d["accepted_tv_homogeneous_control_normalized_density"]

rowmax = np.nanmax(np.where(np.isfinite(S), S, -np.inf), axis=1, keepdims=True)
fov = np.isfinite(S) & (S >= 0.08 * rowmax)
win = (np.abs(X) <= 36) & (Z >= 6) & (Z <= 56)
mask = fov & np.isfinite(Dv) & np.isfinite(Nv) & np.isfinite(Nr) & win

EXTENT = [x_edges[0], x_edges[-1], z_edges[-1], z_edges[0]]
XLIM, ZLIM = (-35.0, 35.0), (55.0, 7.0)


def base(ax, title):
    ax.set_xlim(*XLIM)
    ax.set_ylim(*ZLIM)
    ax.set_xlabel("重建横向位置 x（mm）")
    ax.set_ylabel("重建深度（mm）")
    ax.set_title(title, pad=8)


def circle(ax, color="white"):
    ax.add_patch(Circle((0.0, 30.0), 10.0, fill=False, edgecolor=color,
                        linewidth=1.2, linestyle=(0, (5, 3)), zorder=5))


def letter(ax, s):
    ax.text(-0.20, 1.06, s, transform=ax.transAxes, fontsize=12,
            fontweight="bold", va="top", ha="left")


fig, axes = plt.subplots(1, 4, figsize=(17.0, 4.0))

# a 孔洞体模
ax = axes[0]
cm = matplotlib.colormaps["viridis"].copy()
cm.set_bad("white")
im = ax.imshow(np.where(mask, Nv, np.nan), extent=EXTENT, cmap=cm,
               vmin=0.5, vmax=1.5, aspect="equal", interpolation="nearest")
circle(ax)
base(ax, "孔洞体模")
letter(ax, "a")
cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.03)
cb.set_label("归一化计数密度")

# b 完整水泥参考
ax = axes[1]
im = ax.imshow(np.where(mask, Nr, np.nan), extent=EXTENT, cmap=cm,
               vmin=0.5, vmax=1.5, aspect="equal", interpolation="nearest")
circle(ax)
base(ax, "完整水泥参考")
letter(ax, "b")
cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.03)
cb.set_label("归一化计数密度")

# c 相对计数缺失
ax = axes[2]
cmd = matplotlib.colormaps["RdBu_r"].copy()
cmd.set_bad("white")
im = ax.imshow(np.where(mask, Dv, np.nan), extent=EXTENT, cmap=cmd,
               vmin=-0.25, vmax=0.25, aspect="equal", interpolation="nearest")
circle(ax, "#111827")
base(ax, "相对计数缺失")
letter(ax, "c")
cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.03,
                  ticks=[-0.25, -0.125, 0.0, 0.125, 0.25])
cb.set_label(r"$D = 1 - R_{\mathrm{norm}}$")

# d 横向剖面
ax = axes[3]
band = mask & (Z >= 25.0) & (Z <= 35.0)
prof_v = np.array([np.nanmean(np.where(band[:, i], Dv[:, i], np.nan))
                   if band[:, i].any() else np.nan for i in range(len(xc))])
prof_c = np.array([np.nanmean(np.where(band[:, i], d["accepted_tv_homogeneous_control_deficit"][:, i], np.nan))
                   if band[:, i].any() else np.nan for i in range(len(xc))])
ax.axhline(0.0, color="#9CA3AF", linewidth=0.8, zorder=1)
ax.axvspan(-10.0, 10.0, color="#E5E7EB", zorder=0)
ax.step(xc, prof_c, where="mid", color="#6B7280", linewidth=1.2, label="完整水泥留出组")
ax.step(xc, prof_v, where="mid", color="#1F4E9C", linewidth=1.6, label="孔洞体模")
ax.set_xlim(-32.0, 32.0)
ax.set_ylim(-0.30, 0.34)
ax.set_xlabel("重建横向位置 x（mm）")
ax.set_ylabel("深度 30 ± 5 mm 内的平均 D")
ax.set_title("横向剖面", pad=8)
letter(ax, "d")
ax.legend(loc="upper left", fontsize=9)
ax.text(0.97, 0.97,
        "灰带 = 孔洞真实横向范围\n"
        "表观半高宽 18.7 mm（真值直径 20 mm）\n"
        "系统横向 FWHM 3.67 mm（独立细杆）",
        transform=ax.transAxes, fontsize=8.6, va="top", ha="right")

fig.tight_layout()
for ext, dpi in (("png", 220), ("pdf", 300), ("svg", 300)):
    fig.savefig(OUT / f"void_sober_style.{ext}", dpi=dpi, bbox_inches="tight",
                facecolor="white")
print("written", OUT / "void_sober_style.png")
