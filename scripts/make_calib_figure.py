"""M2 标定曲线图：拟合出的边界离轴距离 vs 真实离轴距离，四档 GATE + 四条冻结模型。

数据来源：四档 analyze_mural_edge.py 的输出 JSON（main / offset / offset2 / offset3），
横轴与纵轴都不含真值以外的任何输入；误差棒是 profile likelihood 的 68% 区间。
四条模型线是在见到 offset2 / offset3 数据之前冻结在配置文件里的。
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

# 字体：Windows 主力机上没有 Noto CJK，必须先找 微软雅黑/黑体，否则中文全是方块。
# DejaVu Sans 垫底，补 U+2212 这类中文字体常缺的符号。
_CJK = ("Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Noto Sans CJK JP",
        "Noto Sans CJK HK", "WenQuanYi Zen Hei")
_stack = [fam for fam in _CJK
          if any(fam in f.name for f in font_manager.fontManager.ttflist)]
plt.rcParams["font.family"] = _stack + ["DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

import argparse
import os
_ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
_ap.add_argument("--calib", default="outputs/calib_fit.json",
                 help="四档标定拟合结果 calib_fit.json")
_ap.add_argument("--out", default="outputs", help="图件输出目录（默认 outputs）")
_args = _ap.parse_args()
os.makedirs(_args.out, exist_ok=True)

SURFACE = "#fcfcfb"
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#9a9891", "#e8e7e3"
FIT, GEOM = "#1c5cab", "#eb6834"          # 实测拟合 / 几何预期

c = json.load(open(_args.calib, encoding="utf-8"))
u = np.array(c["u"]); y = np.array(c["y"]); s = np.array(c["sig"]); res = np.array(c["res"])
k, sk, b, sb, r2 = c["k"], c["sk"], c["b"], c["sb"], c["r2"]
# 2026-09-28：图例里的几个 σ 改为按 calib_fit.json 现算（此前是 12% 的数写死在字符串里）。
# 固定平移模型 +2.2 mm 是见 offset3 数据前冻结的；它的偏离用 offset3（u = +8）那一点算。
_i8 = int(np.argmin(np.abs(u - 8.0)))
sig_k1 = (1.0 - k) / sk
sig_083 = abs(0.83 - k) / sk
sig_shift = abs(y[_i8] - (u[_i8] + 2.2)) / s[_i8]
order = np.argsort(u)
u, y, s, res = u[order], y[order], s[order], res[order]
labels = {0.0: "main", -8.0: "offset", -16.0: "offset2", 8.0: "offset3"}

fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11.6, 4.7), facecolor=SURFACE,
                              gridspec_kw=dict(width_ratios=[1.45, 1], wspace=0.26))
xs = np.linspace(-20, 12, 200)

ax.set_facecolor(SURFACE)
ax.plot(xs, xs, color=MUTED, lw=1.2, ls=(0, (5, 4)), zorder=1,
        label=f"无偏差 k = 1.00（排除 {sig_k1:.1f}σ）")
ax.plot(xs, xs + 2.2, color=MUTED, lw=1.0, ls=(0, (1, 2.5)), zorder=1,
        label=f"固定平移 +2.2 mm（排除 {sig_shift:.1f}σ）")
ax.plot(xs, 0.83 * xs, color=GEOM, lw=2.0, zorder=2,
        label=f"几何条件选择 k = 0.83（相容 {sig_083:.1f}σ）")
ax.plot(xs, k * xs + b, color=FIT, lw=2.4, zorder=3,
        label=f"四点拟合 k = {k:.3f} ± {sk:.3f}")
ax.fill_between(xs, (k - sk) * xs + b - sb, (k + sk) * xs + b + sb,
                color=FIT, alpha=0.13, lw=0, zorder=2)
ax.errorbar(u, y, yerr=s, fmt="o", ms=7, color=INK, mec=SURFACE, mew=1.2,
            elinewidth=1.4, capsize=0, zorder=5, label="GATE 实测（4 档 × 400 万事例）")
for uu, yy in zip(u, y):
    ax.annotate(labels[uu], (uu, yy), xytext=(7, -11), textcoords="offset points",
                fontsize=8.5, color=INK2)
ax.axhline(0, color=GRID, lw=0.8, zorder=0)
ax.axvline(0, color=GRID, lw=0.8, zorder=0)
ax.set_xlabel("真实边界离轴距离 u = 真值 − 探测头轴心（mm）", fontsize=10, color=INK2)
ax.set_ylabel("拟合边界离轴距离 û − 探测头轴心（mm）", fontsize=10, color=INK2)
ax.set_title("标定曲线：拟合边界系统性地缩向探测头轴心", fontsize=11.5,
             color=INK, pad=9, loc="left")
ax.set_xlim(-20, 12.5); ax.set_ylim(-16.5, 10.5)
ax.legend(frameon=False, fontsize=8.6, loc="upper left", labelcolor=INK2,
          handlelength=2.4, borderaxespad=0.4)

ax2.set_facecolor(SURFACE)
ax2.axhspan(-1, 1, color=GRID, zorder=0)
ax2.axhline(0, color=FIT, lw=1.6, zorder=2)
ax2.errorbar(u, res, yerr=s, fmt="o", ms=7, color=INK, mec=SURFACE, mew=1.2,
             elinewidth=1.4, capsize=0, zorder=3)
for uu, rr in zip(u, res):
    dy = 11 if rr >= 0 else -17          # 标签避开零线与数据点
    ax2.annotate(labels[uu], (uu, rr), xytext=(7, dy), textcoords="offset points",
                 fontsize=8.5, color=INK2)
ax2.set_xlabel("真实边界离轴距离 u（mm）", fontsize=10, color=INK2)
ax2.set_ylabel("实测 − 直线拟合（mm）", fontsize=10, color=INK2)
ax2.set_title("残差：线性成立", fontsize=11.5, color=INK, pad=9, loc="left")
ax2.set_xlim(-20, 12.5); ax2.set_ylim(-2.2, 2.2)
ax2.annotate(f"截距 b = {b:+.2f} ± {sb:.2f} mm（与 0 相容）\n"
             f"残差标准差 s = √(Σ残差²/(n−2)) = {c['srms']:.2f} mm     r² = {r2:.4f}\n"
             f"灰带 = ±1 mm",
             (0.97, 0.04), xycoords="axes fraction", ha="right", va="bottom",
             fontsize=8.8, color=INK,
             bbox=dict(boxstyle="round,pad=0.4", fc="#f3f2ee", ec="#dedcd6", lw=0.8))

for a in (ax, ax2):
    a.tick_params(colors=INK2, labelsize=9)
    for sp in ("top", "right"):
        a.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        a.spines[sp].set_color(MUTED); a.spines[sp].set_linewidth(0.8)
    a.grid(color=GRID, lw=0.7, zorder=0); a.set_axisbelow(True)

fig.suptitle("空鼓边界定位的横向刻度标定（单位姿，L = −15 mm，空鼓 20 mm 厚、深 20–40 mm）",
             fontsize=13, color=INK, x=0.008, ha="left", y=0.985)
fig.text(0.008, 0.012,
         "四档除空鼓横向范围外一字未改，共用同一参照场。误差棒为 profile likelihood 的 68% 区间。"
         "四条模型线在见到 offset2 / offset3 数据之前已冻结在配置文件中。"
         "横纵坐标均为重建量，真值只用于给出横轴与判据。\n"
         + (f"探测器响应：{c['response']}，D1 8% FWHM @511 keV（2026-09-28 重做分析，GATE 未重跑）。"
            if "response" in c else ""),
         fontsize=8.2, color=MUTED, ha="left")
fig.subplots_adjust(left=0.068, right=0.985, top=0.865,
                    bottom=(0.20 if "response" in c else 0.155))   # 图注两行时给横轴标签留位置
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(_args.out, f"m2_calibration.{ext}"), dpi=200,
                facecolor=SURFACE)
print(f"写出 {os.path.join(_args.out, 'm2_calibration')}.png / .pdf")
