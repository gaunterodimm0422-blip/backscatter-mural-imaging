"""M2 结果图：空鼓场/参照场的计数比值随横向位置的台阶，及其零假设对照。

数据来源：analyze_mural_edge.py 输出的 m2_edge_result.json（09-28 起默认读 D2 15% 的
M2 主档 m3/outputs/d2f15/m2_mural_edge/；outputs/m2_edge_result.json 是 12% 旧结果，别再用），
未做任何平滑、未使用真值、纵轴是原始计数比。误差棒为泊松误差。
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
_ap.add_argument("--result", default="m3/outputs/d2f15/m2_mural_edge/m2_edge_result.json",
                 help="analyze_mural_edge.py 输出的 m2_edge_result.json")
_ap.add_argument("--out", default="outputs", help="图件输出目录（默认 outputs）")
_ap.add_argument("--name", default="m2_edge_ratio",
                 help="输出文件名主干（默认 m2_edge_ratio；画 M3 时务必改名，免得覆盖 M2 的图）")
_ap.add_argument("--label", default="", help="标题前缀，例如 \"M3 新构型\"")
_ap.add_argument("--void-desc", default="空鼓 20 mm 厚、深 20–40 mm",
                 help="图注里对空鼓的描述（厚度扫描时要跟着改）")
_ap.add_argument("--null-cal", default=None,
                 help="可选：另一份结果 json，取其中的 null_calibration（例如 2000 次泊松抽样那份）")
_ap.add_argument("--note", default="",
                 help="图注末尾追加的一句（例如探测器响应口径），默认不加")
_args = _ap.parse_args()
os.makedirs(_args.out, exist_ok=True)

SURFACE = "#fcfcfb"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#9a9891"
RAMP = ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]   # 蓝色序数梯，浅->深

d = json.load(open(_args.result, encoding="utf-8"))
head = d["head_x_mm"]; truth = d["truth_edge_world_mm"]
bands = d["bands_mm"]
x = np.array(d["x_centres_mm"]) + head          # 换成世界坐标
H = {k: np.array(v, float) for k, v in d["hist"].items()}
npri = d["n_primary"]
fit_s = d["fits"]["signal"]; fit_n = d["fits"]["null"]

panels = [("空鼓场 / 参照场", "halfvoid", "ref_main", fit_s),
          ("零假设：参照场 / 参照场", "ref_null", "ref_main", fit_n)]

fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.9), sharey=True,
                         facecolor=SURFACE, gridspec_kw=dict(wspace=0.06))
_ymins = []          # 两图所有点的 (比值 − 误差) 下沿，用来定纵轴下限
_ymaxs = []          # 两图所有点的比值中心值上沿：纵轴上限至少要把每个点的中心值画进来
_void_side = []      # 左图空鼓侧（真值边界右侧 6 mm 以外）各带的平均比值，写进图注
for ax, (title, num, den, fit) in zip(axes, panels):
    ax.set_facecolor(SURFACE)
    s = npri[num] / npri[den]
    ax.axhline(1.0, color=MUTED, lw=1.0, ls=(0, (4, 3)), zorder=1)
    ax.axvline(truth, color=MUTED, lw=1.0, ls=(0, (1, 2)), zorder=1)
    for b, (lo, hi) in enumerate(bands):
        a, c = H[num][b], H[den][b]
        r = a / (s * c)
        err = r * np.sqrt(1.0 / np.maximum(a, 1) + 1.0 / np.maximum(c, 1))
        _ok = np.isfinite(r) & np.isfinite(err)
        if _ok.any():
            _ymins.append(float(np.min((r - err)[_ok])))
            _ymaxs.append(float(np.max(r[_ok])))
        if num == "halfvoid":
            _m = _ok & (x > truth + 6.0) & (c > 0)
            if _m.any():
                _void_side.append(float(a[_m].sum() / (s * c[_m].sum())))
        ax.errorbar(x, r, yerr=err, color=RAMP[b], lw=2.0, marker="o", ms=4.5,
                    elinewidth=1.0, capsize=0, zorder=3 + b,
                    label=f"{lo}–{hi} mm", mec=SURFACE, mew=0.8)
        # 右端直标已去掉（2026-09-23）：四条带的拟合箱右端不在同一个 x，
        # 直标要么落在曲线中间、要么只剩一条带有标，图例已足够识别。
    ax.set_title(title, fontsize=11.5, color=INK, pad=9, loc="left")
    ax.set_xlabel("横向位置 x（世界坐标，mm）", fontsize=10, color=INK2)
    ax.tick_params(colors=INK2, labelsize=9)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(MUTED); ax.spines[sp].set_linewidth(0.8)
    ax.set_xlim(x[0] - 4.5, x[-1] + 9.5)
    ax.grid(axis="y", color="#e8e7e3", lw=0.7, zorder=0)
    ax.set_axisbelow(True)

axes[0].set_ylabel("计数比值", fontsize=10, color=INK2)
_lo = min(_ymins) if _ymins else 0.62
_hi = max(_ymaxs) if _ymaxs else 1.22
axes[0].set_ylim(min(0.62, np.floor((_lo - 0.02) * 50) / 50),
                 max(1.22, np.ceil((_hi + 0.04) * 50) / 50))
axes[0].annotate(f"真值边界\n{truth:.0f} mm", (truth, 1.185), xytext=(-6, 0),
                 textcoords="offset points", ha="right", va="top",
                 fontsize=8.5, color=INK2)
axes[0].annotate(f"拟合边界 {fit_s['xe']+head:.2f} mm\n"
                 f"68% 区间 [{fit_s['xe_lo']+head:.2f}, {fit_s['xe_hi']+head:.2f}]\n"
                 f"T = {fit_s['T']:.1f}（阈值 30）",
                 (0.985, 0.965), xycoords="axes fraction", ha="right", va="top",
                 fontsize=9, color=INK,
                 bbox=dict(boxstyle="round,pad=0.42", fc="#f3f2ee", ec="#dedcd6", lw=0.8))
_nc = None
if _args.null_cal:
    _nc = json.load(open(_args.null_cal, encoding="utf-8")).get("null_calibration")
if _nc is None:
    _nc = d.get("null_calibration")
_nctxt = (f"\n零分布 {_nc['trials']} 次泊松抽样：99% 分位 {_nc['p99']:.1f}，最大 {_nc['max']:.1f}"
          if isinstance(_nc, dict) and "p99" in _nc else "")
axes[1].annotate(f"T = {fit_n['T']:.1f}（阈值 30，不显著，余量 {30 - fit_n['T']:.1f}）"
                 f"\n两档都没有空鼓，比值处处为 1{_nctxt}",
                 (0.018, 0.055), xycoords="axes fraction", ha="left", va="bottom",
                 fontsize=9, color=INK,
                 bbox=dict(boxstyle="round,pad=0.42", fc="#f3f2ee", ec="#dedcd6", lw=0.8))
axes[0].legend(title="重建深度带", frameon=False, fontsize=9, title_fontsize=9,
               loc="lower left", labelcolor=INK2, ncol=2,
               bbox_to_anchor=(0.0, 0.0))

_npri_txt = "、".join(f"{k} {v/1e4:.0f} 万" for k, v in npri.items())
_pre = f"{_args.label}：" if _args.label else ""
fig.suptitle(f"{_pre}半空间空鼓的横向边界：计数比值台阶（单位姿）",
             fontsize=13, color=INK, x=0.008, ha="left", y=0.985)
_vs = (f"左图空鼓侧（边界右侧 6 mm 以外）各深度带比值 {min(_void_side):.2f}–{max(_void_side):.2f}；"
       if _void_side else "")
fig.text(0.008, 0.012,
         f"{_args.void_desc}，边界真值 x = {truth:.0f} mm，探测头轴心 x = {head:.0f} mm。"
         f"初级粒子数：{_npri_txt}（比值已按初级数归一）。"
         "误差棒为泊松误差，未做平滑，未使用真值信息。\n"
         f"{_vs}右图两档均无空鼓、比值处处为 1。"
         + (f"{_args.note}。" if _args.note else ""),
         fontsize=8.2, color=MUTED, ha="left", linespacing=1.55)
fig.subplots_adjust(left=0.062, right=0.975, top=0.87, bottom=0.175)
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(_args.out, f"{_args.name}.{ext}"), dpi=200,
                facecolor=SURFACE)
print(f"写出 {os.path.join(_args.out, _args.name)}.png / .pdf")
print(f"  拟合边界 {fit_s['xe']+head:.2f} mm，68% 区间 [{fit_s['xe_lo']+head:.2f}, "
      f"{fit_s['xe_hi']+head:.2f}]（半宽 {(fit_s['xe_hi']-fit_s['xe_lo'])/2:.3f}），"
      f"xe_sigma {fit_s.get('xe_sigma', float('nan')):.3f}，T {fit_s['T']:.1f}，零假设 T {fit_n['T']:.1f}")
if _void_side:
    print("  空鼓侧各带平均比值:", " / ".join(f"{v:.3f}" for v in _void_side))
