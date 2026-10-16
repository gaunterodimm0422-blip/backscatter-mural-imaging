# -*- coding: utf-8 -*-
"""从 imaging_demo_data.npz 重画成像演示图（两张）。

图一 成像效果：(a) 参照场点密度  (b) 空鼓场点密度  (c) 比值图  (d) 四条冻结深度带的横向剖面
图二 定位能力：三档边界位置（轴心 / 离轴 −8 / 离轴 −16 mm）的比值图、剖面叠加、横向刻度标定

一维剖面优先用 npz 里的 hb_r / hb_v（就是冻结分析的四条深度带）。
老版 make_imaging_demo.py 存的 npz 没有这两项，那时才退回 5 mm 深度箱近似分组，
此时图注会自动加一行说明——别把近似分组的数当成冻结分析的 c_b 引用。

图二 (e) 的 k / b / r² 在脚本里有断言，必须复现 D2 15% 分析（2026-09-28）按 npz 里
两位小数的 x_e 算出的 k = 0.9200、b = +0.020 mm、r² = 0.99955，对不上就抛异常、图作废。
（json 全精度的 k = 0.9197 ± 0.0195，是正式引用值；两者只差在 x_e 的小数位。）
2026-09-28 起本脚本只接 D2 15% 的 npz（远控机 outputs\d2f15\imaging_demo_m3*）；
12% 版的脚本与图冻结在主力机 outputs\_12pct_frozen\。

用法：
  python3 redraw_imaging_demo.py --on-axis m3.npz --off8 m3x23.npz --off16 m3x31.npz --out out
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Rectangle
from matplotlib import font_manager

_have = {f.name for f in font_manager.fontManager.ttflist}
_stack = [f for f in ("WenQuanYi Zen Hei", "Noto Sans CJK SC", "Noto Sans CJK JP",
                      "Microsoft YaHei") if f in _have]
# DejaVu Sans 兜底：中文字体常缺 U+2212 真减号，靠 fallback 补上
plt.rcParams["font.family"] = _stack + ["DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = True

SURFACE = "#fcfcfb"
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#9a9891", "#e3e1dc"
TRUTH, FIT = "#1a7f4b", "#c0392b"
SEQ = LinearSegmentedColormap.from_list("seq", ["#f7f6f3", "#bcd4ef", "#5b93d4", "#1b4f8f"])
DIV = LinearSegmentedColormap.from_list("div", ["#1b4f8f", "#5b93d4", "#e9e7e2",
                                                "#eb9a6c", "#b3441c"])
DIV = DIV.copy(); DIV.set_bad(SURFACE)
SEQ = SEQ.copy(); SEQ.set_bad(SURFACE)

MIN_REF = 100
GCOL = ["#1b4f8f", "#5b93d4", "#eb9a6c", "#b3441c"]


class Case:
    def __init__(self, path, fit_override=None):
        d = np.load(path, allow_pickle=True)
        self.xe = np.asarray(d["xe"], float)
        self.de = np.asarray(d["de"], float)
        self.xc = 0.5 * (self.xe[1:] + self.xe[:-1])
        self.dc = 0.5 * (self.de[1:] + self.de[:-1])
        self.Hr = np.asarray(d["Hr"], float)
        self.Hv = np.asarray(d["Hv"], float)
        self.npr = int(d["n_primary_ref"]); self.npv = int(d["n_primary_void"])
        self.nsr = int(d["n_sel_ref"]); self.nsv = int(d["n_sel_void"])
        self.headx = float(d["head_x_mm"])
        self.truth_w = float(d["x_edge_truth"])
        self.fit_w = float(d["x_edge_fit"]) if fit_override is None else fit_override
        self.ute = self.truth_w - self.headx
        self.ufe = self.fit_w - self.headx
        self.void_depth = np.asarray(d["void_depth"], float)
        self.label = str(d["label"])
        self.scale = self.npr / self.npv if self.npr and self.npv else 1.0
        self.bands = [tuple(int(x) for x in b) for b in np.asarray(d["bands"])]
        self.exact = ("hb_r" in d.files) and ("hb_v" in d.files)
        if self.exact:
            self.hb_r = np.asarray(d["hb_r"], float)
            self.hb_v = np.asarray(d["hb_v"], float)
        else:
            self.hb_r = self.hb_v = None

    @property
    def R(self):
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(self.Hr >= MIN_REF, self.Hv * self.scale / self.Hr, np.nan)

    def _counts(self, idx):
        """按冻结深度带取一维计数；idx 是 bands 里的序号列表（可合并多带）。"""
        if self.exact:
            return self.hb_r[idx].sum(0), self.hb_v[idx].sum(0)
        lo = min(self.bands[i][0] for i in idx); hi = max(self.bands[i][1] for i in idx)
        m = (self.dc >= lo) & (self.dc < hi)
        return self.Hr[:, m].sum(1), self.Hv[:, m].sum(1)

    def profile(self, idx):
        r, v = self._counts(idx)
        g = r >= MIN_REF
        ratio = v[g] * self.scale / r[g]
        err = ratio * np.sqrt(1.0 / np.maximum(v[g], 1) + 1.0 / r[g])
        return self.xc[g], ratio, err

    def contrast(self, idx, guard=6.0):
        r, v = self._counts(idx)
        g = r >= MIN_REF
        L = (self.xc < self.ute - guard) & g
        Rg = (self.xc > self.ute + guard) & g
        if L.sum() == 0 or Rg.sum() == 0:
            return float("nan")
        rl = v[L].sum() / r[L].sum(); rr = v[Rg].sum() / r[Rg].sum()
        return 100.0 * (1.0 - rr / rl)


def style_axes(ax):
    ax.tick_params(colors=INK2, labelsize=8.5)
    ax.grid(True, color=GRID, lw=0.7, zorder=0)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(MUTED); ax.spines[sp].set_linewidth(0.8)


def map_axes(fig, ax, M, c, cmap, norm, title, cblabel, box=True, truth_line=False):
    ext = [c.xe[0], c.xe[-1], c.de[-1], c.de[0]]
    im = ax.imshow(np.ma.masked_invalid(M.T), origin="upper", extent=ext, aspect="auto",
                   cmap=cmap, norm=norm, interpolation="nearest")
    ax.set_title(title, fontsize=10.5, color=INK, loc="left", pad=6)
    ax.set_xlabel("横向 x（探测头坐标，mm）", fontsize=9, color=INK2)
    ax.set_ylabel("重建深度（mm）", fontsize=9, color=INK2)
    ax.tick_params(colors=INK2, labelsize=8)
    cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
    cb.set_label(cblabel, fontsize=8, color=INK2)
    cb.ax.tick_params(colors=INK2, labelsize=7.5)
    if box:
        lo_v, hi_v = c.void_depth
        ax.add_patch(Rectangle((c.ute, lo_v), c.xe[-1] - c.ute, hi_v - lo_v,
                               fc="none", ec=TRUTH, lw=1.5, ls=(0, (5, 3)), zorder=5))
    if truth_line:
        ax.axvline(c.ute, color=TRUTH, lw=3.0, ls=(0, (4, 2)), zorder=5)
    return im


# ---------------------------------------------------------------- 图一
def figure_one(c, out):
    fig = plt.figure(figsize=(15.0, 8.8), facecolor=SURFACE)
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 0.95], hspace=0.38, wspace=0.26,
                          left=0.05, right=0.975, top=0.868, bottom=0.185)
    vmax = float(np.percentile(np.concatenate([c.Hr.ravel(), c.Hv.ravel()]), 99.5))
    nrm = Normalize(vmin=0, vmax=vmax)
    for j, (H, name, bx) in enumerate(((c.Hr, "(a) 参照场（实心墙，无空鼓）", False),
                                       (c.Hv, "(b) 空鼓场（右半边 20–40 mm 深处有空鼓）", True))):
        ax = fig.add_subplot(gs[0, j])
        map_axes(fig, ax, H, c, SEQ, nrm, f"{name}  重建点密度", "每箱事例数", box=bx)

    ax = fig.add_subplot(gs[0, 2])
    map_axes(fig, ax, c.R, c, DIV, Normalize(vmin=0.6, vmax=1.4),
             "(c) 比值 = 空鼓场 / 参照场", "比值（1.0 = 无差别）")
    lo_v, hi_v = c.void_depth
    ax.annotate("真值空鼓范围", (c.ute + 1.5, lo_v - 1.5), fontsize=8.4, color=TRUTH,
                ha="left", va="bottom", zorder=6)
    ax.axvline(c.ute, color=TRUTH, lw=3.0, ls=(0, (4, 2)), zorder=5)
    ax.axvline(c.ufe, color=FIT, lw=1.3, zorder=6)
    ax.annotate(f"拟合边界 {c.fit_w:+.2f} mm", (c.ufe - 1.5, c.de[0] + 2.5),
                fontsize=8.4, color=FIT, ha="right", va="top", zorder=6)

    axp = fig.add_subplot(gs[1, :])
    for i, ((lo, hi), col) in enumerate(zip(c.bands, GCOL)):
        x, r, e = c.profile([i])
        axp.errorbar(x, r, yerr=e, fmt="o-", ms=4.5, lw=1.6, color=col, capsize=0,
                     label=f"重建深度 {lo}–{hi} mm（对比度 {c.contrast([i]):.1f}%）", zorder=4)
    axp.axhline(1.0, color=MUTED, lw=1.0, ls=(0, (4, 3)), zorder=2)
    y0, y1 = axp.get_ylim(); axp.set_ylim(y0 - 0.02, y1 + 0.18 * (y1 - y0))
    y0, y1 = axp.get_ylim()
    axp.axvline(c.ute, color=TRUTH, lw=3.0, ls=(0, (4, 2)), zorder=3)
    axp.annotate("真值边界", (c.ute - 1.2, y1 - 0.03 * (y1 - y0)), fontsize=8.6,
                 color=TRUTH, ha="right", va="top", zorder=5)
    axp.axvline(c.ufe, color=FIT, lw=1.3, zorder=3)
    axp.annotate("拟合边界", (c.ufe + 1.2, y1 - 0.12 * (y1 - y0)), fontsize=8.6,
                 color=FIT, ha="left", va="top", zorder=5)
    axp.set_title("(d) 沿横向的一维剖面：空鼓场 / 参照场（误差棒为泊松统计）",
                  fontsize=10.5, color=INK, loc="left", pad=6)
    axp.set_xlabel("横向 x（探测头坐标，mm）", fontsize=9.5, color=INK2)
    axp.set_ylabel("比值", fontsize=9.5, color=INK2)
    axp.set_xlim(c.xe[0], c.xe[-1])
    style_axes(axp)
    leg = axp.legend(loc="upper right", fontsize=8.4, frameon=True, framealpha=0.95,
                     edgecolor=GRID, ncol=2)
    for t in leg.get_texts():
        t.set_color(INK2)

    fig.suptitle("图一  成像效果演示（M3 新构型，边界在探测头轴心）："
                 "单看“像”看不出空鼓，信息在比值里",
                 fontsize=14.5, color=INK, x=0.006, ha="left", y=0.975)
    fig.text(0.006, 0.012,
             "(a)(b) 用同一色标：两张图肉眼几乎无差别——空鼓造成的计数亏空只有百分之几到三成，"
             "淹没在点密度的整体形状里。(c) 做了比值才显出来。\n"
             "深度方向是糊的（本构型深度误差半宽约 10.9 mm），所以亏空在纵向铺开、并不停在 20–40 mm；"
             "横向边界才是可信的那一维——这正是本方法只报横向边界、不报深度的原因。\n"
             f"参照场 {c.nsr:,} 个选出事例（{c.npr:,} 初级粒子），空鼓场 {c.nsv:,}（{c.npv:,}）；"
             f"比值已按初级粒子数归一，每箱参照计数 < {MIN_REF} 的不画。"
             "绿色虚线框 = 真值空鼓；红线 = 多深度带共享边界拟合给出的边界。\n"
             "(d) 的四条深度带就是冻结分析用的那四条；图例中的对比度是本图自己算的"
             "描述性双比值（边界两侧各留 6 mm 保护带），不是冻结分析里的 c_b —— 两者算法不同，不要互相替换引用。"
             f"{RESP_NOTE}。"
             + ("" if c.exact else
                "\n注意：本图的 npz 缺一维剖面，(d) 的分组由深度箱近似，与冻结深度带不完全一致。"),
             fontsize=8.2, color=MUTED, ha="left", va="bottom", linespacing=1.6)
    for ext_ in ("png", "pdf"):
        fig.savefig(Path(out) / f"imaging_demo_fig1.{ext_}", dpi=190, facecolor=SURFACE)
    plt.close(fig)




# ---------------------------------------------------------------- 图二
# 三个标定点的 68% 区间半宽，取自冻结分析（m3_edge_result 系列），不是本图算的。
# 三个标定点的 68% 区间（下侧, 上侧），单位 mm，逐字取自远控机冻结分析 json：
#   m3_edge_result / m3_offset / m3_offset2 里 fits.signal 的 xe − xe_lo 与 xe_hi − xe。
# 2026-09-23 更正：此前写的 ±0.35 / ±0.45 / ±1.25 是从"只保留一位小数"的区间端点反推的。
# 2026-09-28 换成 D2 15% 的值，取自远控机 outputs\d2f15\m3_edge_result / m3_offset / m3_offset2：
#   轴心 0.3299 / 0.3349；离轴 −8 0.3959 / 0.4296；离轴 −16 1.3946 / 0.9507（下侧 / 上侧）。
CI68 = {0.0: (0.330, 0.335), -8.0: (0.396, 0.430), -16.0: (1.395, 0.951)}
RESP_NOTE = "探测器响应：D1 8%、D2 15% FWHM @511 keV（2026-09-28 重做分析，GATE 未重跑）"
DEEP = [2, 3]          # bands 里的 15–25 与 25–35，合并起来画


def fit_scale(cases):
    """无权直线拟合 û−L = k·u + b，与 M2 四点标定逐字同一套做法。"""
    u = np.array([c.ute for c in cases], float)          # 真值离轴
    y = np.array([c.ufe for c in cases], float)          # 拟合离轴
    n = len(u)
    Suu = ((u - u.mean()) ** 2).sum()
    k = ((u - u.mean()) * (y - y.mean())).sum() / Suu
    b = y.mean() - k * u.mean()
    res = y - (k * u + b)
    s = np.sqrt((res ** 2).sum() / (n - 2))
    sk = s / np.sqrt(Suu)
    sb = s * np.sqrt(1.0 / n + u.mean() ** 2 / Suu)
    r2 = 1.0 - (res ** 2).sum() / ((y - y.mean()) ** 2).sum()
    return dict(u=u, y=y, k=k, b=b, sk=sk, sb=sb, r2=r2, res=res, s=s)


def figure_two(cases, out):
    cases = sorted(cases, key=lambda c: -c.ute)          # 轴心 → 离轴最远
    F = fit_scale(cases)
    # 自检：必须复现 D2 15% 分析按两位小数 x_e 算出的 k / b / r²，否则这张图作废
    assert abs(F["k"] - 0.9200) < 5e-4, f"k 对不上 15% 分析：{F['k']:.4f}"
    assert abs(F["b"] - 0.020) < 5e-3, f"b 对不上 15% 分析：{F['b']:.4f}"
    assert abs(F["r2"] - 0.99955) < 5e-5, f"r2 对不上 15% 分析：{F['r2']:.5f}"
    far = min(cases, key=lambda c: c.ute)                # 离轴最远的一档
    far_bias = far.fit_w - far.truth_w

    fig = plt.figure(figsize=(15.0, 9.0), facecolor=SURFACE)
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 0.95], hspace=0.40, wspace=0.30,
                          left=0.045, right=0.952, top=0.872, bottom=0.195)
    nrm = Normalize(vmin=0.6, vmax=1.4)
    tags = ["(a) 边界在探测头轴心", "(b) 边界离轴 −8 mm", "(c) 边界离轴 −16 mm"]
    for j, (c, tag) in enumerate(zip(cases, tags)):
        ax = fig.add_subplot(gs[0, j])
        map_axes(fig, ax, c.R, c, DIV, nrm, f"{tag}", "比值（1.0 = 无差别）",
                 truth_line=True)
        ax.axvline(c.ufe, color=FIT, lw=1.3, zorder=6)
        ax.annotate(f"真值 {c.ute:+.1f}", (c.ute - 1.2, 3.0), fontsize=8.4,
                    color=TRUTH, ha="right", va="top", zorder=6)
        ax.annotate(f"拟合 {c.ufe:+.2f}", (c.ufe + 1.2, 10.0), fontsize=8.4,
                    color=FIT, ha="left", va="top", zorder=6)

    # ---- 左下：三档的横向剖面叠在一起
    axp = fig.add_subplot(gs[1, :2])
    cols = ["#1b4f8f", "#5b93d4", "#b3441c"]
    mks = ["o-", "s-", "^-"]
    for c, col, mk in zip(cases, cols, mks):
        x, r, e = c.profile(DEEP)
        axp.errorbar(x, r, yerr=e, fmt=mk, ms=5.0, lw=1.7, color=col, capsize=0,
                     label=f"真值边界 {c.ute:+.0f} mm（对比度 {c.contrast(DEEP):.1f}%）",
                     zorder=4)
    axp.axhline(1.0, color=MUTED, lw=1.0, ls=(0, (4, 3)), zorder=2)
    y0, y1 = axp.get_ylim(); axp.set_ylim(y0 - 0.02, y1 + 0.22 * (y1 - y0))
    y0, y1 = axp.get_ylim()
    for c in cases:
        axp.axvline(c.ute, color=TRUTH, lw=2.6, ls=(0, (4, 2)), zorder=3)
        axp.axvline(c.ufe, color=FIT, lw=1.2, zorder=3)
    axp.annotate("每对竖线 = 真值（绿虚）与拟合（红实）", (cases[0].ute + 1.5,
                 y1 - 0.04 * (y1 - y0)), fontsize=8.6, color=INK2, ha="left", va="top")
    lo = min(c.bands[i][0] for c in cases for i in DEEP)
    hi = max(c.bands[i][1] for c in cases for i in DEEP)
    axp.set_title(f"(d) 重建深度 {lo}–{hi} mm（两条冻结深度带合并）的横向剖面："
                  "空鼓挪多远，台阶就挪多远",
                  fontsize=10.5, color=INK, loc="left", pad=6)
    axp.set_xlabel("横向 x（探测头坐标，mm）", fontsize=9.5, color=INK2)
    axp.set_ylabel("比值", fontsize=9.5, color=INK2)
    axp.set_xlim(cases[0].xe[0], cases[0].xe[-1])
    style_axes(axp)
    leg = axp.legend(loc="lower left", fontsize=8.6, frameon=True, framealpha=0.95,
                     edgecolor=GRID)
    for t in leg.get_texts():
        t.set_color(INK2)

    # ---- 右下：横向刻度标定曲线
    axc = fig.add_subplot(gs[1, 2])
    u, y = F["u"], F["y"]
    xs = np.linspace(u.min() - 2.5, u.max() + 2.5, 100)
    axc.plot(xs, xs, color=MUTED, lw=1.1, ls=(0, (5, 4)), zorder=1, label="无偏差 k = 1")
    axc.plot(xs, F["k"] * xs + F["b"], color="#1b4f8f", lw=1.9, zorder=2,
             label=f"拟合 k = {F['k']:.4f}")
    _lo = [CI68.get(float(t), (np.nan, np.nan))[0] for t in u]
    _up = [CI68.get(float(t), (np.nan, np.nan))[1] for t in u]
    axc.errorbar(u, y, yerr=[_lo, _up], fmt="o", ms=6.5,
                 color=FIT, capsize=3.0, lw=1.3, zorder=4, label="三个标定点")
    axc.set_title("(e) 横向刻度标定：拟合离轴 vs 真值离轴",
                  fontsize=10.5, color=INK, loc="left", pad=6)
    axc.set_xlabel("真值边界离轴 u（mm）", fontsize=9.5, color=INK2)
    axc.set_ylabel("拟合边界离轴 û − L（mm）", fontsize=9.5, color=INK2)
    style_axes(axc)
    axc.text(0.04, 0.955,
             f"k = {F['k']:.4f} ± {F['sk']:.4f}\n"
             f"b = {F['b']:+.3f} ± {F['sb']:.3f} mm\n"
             f"r² = {F['r2']:.5f}\n"
             f"残差 {F['res'][0]:+.3f} / {F['res'][1]:+.3f} / {F['res'][2]:+.3f} mm",
             transform=axc.transAxes, fontsize=8.2, color=INK2, ha="left", va="top",
             bbox=dict(fc=SURFACE, ec=GRID, lw=0.8, boxstyle="round,pad=0.45"))
    leg2 = axc.legend(loc="lower right", fontsize=8.0, frameon=True, framealpha=0.95,
                      edgecolor=GRID)
    for t in leg2.get_texts():
        t.set_color(INK2)

    fig.suptitle("图二  定位能力演示（M3 新构型）：空鼓边界挪到哪，重建出来的边界就跟到哪",
                 fontsize=14.5, color=INK, x=0.006, ha="left", y=0.975)
    nsr = cases[0].nsr
    fig.text(0.006, 0.012,
             f"三档用的是同一个参照场（同一次 GATE 参照运行，{nsr:,} 个选出事例），"
             "所以 (a)(b)(c) 三张比值图之间的全部差别都只来自空鼓边界位置。"
             "探测头轴心固定在 L = −15 mm。\n"
             + "；".join(f"{t} 真值 {c.truth_w:+.2f} mm、拟合 {c.fit_w:+.2f} mm、"
                         f"偏差 {c.fit_w - c.truth_w:+.2f} mm"
                         for t, c in zip(("(a)", "(b)", "(c)"), cases)) + "。\n"
             f"(e) 的直线是三点无权拟合（与 M2 四点标定逐字同一套做法）："
             f"k = {F['k']:.4f} ± {F['sk']:.4f}，b = {F['b']:+.3f} ± {F['sb']:.3f} mm，"
             f"r² = {F['r2']:.5f}，判据 9 通过。误差棒是各档自身的 68% 区间（不对称），取自冻结分析 json，不是本图算的。\n"
             "k < 1 的含义：离轴越远，测出的边界越往探测头中心缩；"
             f"新构型下这个系统偏差在 {abs(far.ute):.0f} mm 离轴处为 {far_bias:+.2f} mm，不做刻度改正也够用。\n"
             "(d) 图例中的对比度是本图自己算的描述性双比值（边界两侧各留 6 mm 保护带），"
             f"不是冻结分析里的 c_b；每箱参照计数 < {MIN_REF} 的点已剔除。{RESP_NOTE}。",
             fontsize=8.2, color=MUTED, ha="left", va="bottom", linespacing=1.6)
    for ext_ in ("png", "pdf"):
        fig.savefig(Path(out) / f"imaging_demo_fig2.{ext_}", dpi=190, facecolor=SURFACE)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="从 npz 重画成像演示图")
    ap.add_argument("--on-axis", required=True, help="边界在轴心那一档的 npz")
    ap.add_argument("--off8", required=True, help="边界离轴 −8 mm 那一档的 npz")
    ap.add_argument("--off16", required=True, help="边界离轴 −16 mm 那一档的 npz")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    ca = Case(a.on_axis); c8 = Case(a.off8); c16 = Case(a.off16)
    for c in (ca, c8, c16):
        if not c.exact:
            print(f"警告：{c.label} 的 npz 缺 hb_r/hb_v，一维剖面只能用深度箱近似。")
    figure_one(ca, out)
    figure_two([ca, c8, c16], out)
    print("图一：", out / "imaging_demo_fig1.png")
    print("图二：", out / "imaging_demo_fig2.png")
    for c in (ca, c8, c16):
        cb = " / ".join(f"{c.contrast([i]):.2f}%" for i in range(len(c.bands)))
        print(f"  {c.label}: 真值 {c.truth_w:+.2f} 拟合 {c.fit_w:+.2f} "
              f"选出 参照 {c.nsr:,} / 空鼓 {c.nsv:,}  四带对比度 {cb}")


if __name__ == "__main__":
    main()
