# -*- coding: utf-8 -*-
"""解释"导师那个圆锥是从哪来的"。

左：正着看。康普顿公式只给极角 θ，方位角 φ 一点信息都没有；φ 取遍 0~360°，
    出射方向就扫出一个以入射方向 v 为轴、半角 θ 的圆锥。
右：倒过来看。我们知道光子最后落在 H。"让 P→H 与 v 成 θ 角"的所有点 P，
    正好构成一个以 H 为顶点、以 −v 为轴、半角 θ 的圆锥；与已知入射直线一交，P 就定了。

θ 取 120°（= E′ 204.4 keV），与 5.4 节那张图同一个构造算例。
注意 θ > 90°，所以锥是**朝后开**的——背散射本来就是这样。
底面圆按透视系数压扁，纯为示意；其余几何量都是现算的。
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Ellipse, Arc, Polygon
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
_ap.add_argument("--out", default="outputs", help="图件输出目录（默认 outputs）")
_args = _ap.parse_args()
os.makedirs(_args.out, exist_ok=True)

SURFACE = "#fcfcfb"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#9a9891"
D1_COLOR, OBJ = "#2a78d6", "#eceae4"
GOOD, RAY, CONE = "#1a7f4b", "#b35c1e", "#6b4fbf"

S = np.array([0.0, 17.5]); T = np.array([-15.0, 52.5]); H = np.array([30.0, 7.5])
THETA = 120.0
v = (S - T) / np.linalg.norm(S - T)
d = H - S; nd = float(np.linalg.norm(d)); a = float(v @ d)
c = float(np.cos(np.radians(THETA))); Ep = 511.0 / (2.0 - c)
q = (a * a - c * c * nd * nd) / (1 - c * c)
t_plus = a + float(np.sqrt(a * a - q))
P = S + t_plus * v
FORE = 0.38                      # 底面圆的透视压扁系数（纯绘图参数）


def perp(u):
    return np.array([-u[1], u[0]])


def draw_cone(ax, apex, axis, theta_deg, L, n_rays=7, arrow=False):
    """圆锥 = 两条共面母线（粗，就是剖面上看到的）+ 透视底面圆 + 几条面外母线。"""
    u = axis / np.linalg.norm(axis)
    th = np.radians(theta_deg)
    C = apex + L * np.cos(th) * u
    rad = L * np.sin(th)
    n = perp(u)
    ax.add_patch(Polygon([apex, C + rad * n, C - rad * n], closed=True,
                         fc=CONE, ec="none", alpha=0.08, zorder=3))
    ax.add_patch(Ellipse(tuple(C), 2 * rad, 2 * rad * FORE,
                         angle=float(np.degrees(np.arctan2(n[1], n[0]))),
                         fc=CONE, ec=CONE, lw=1.1, ls=(0, (4, 3)), alpha=0.13, zorder=3))
    for k in range(n_rays):
        ph = np.pi * (k + 0.5) / n_rays          # 只画"前半圈"，避免线堆在一起
        pt = C + rad * np.cos(ph) * n + rad * FORE * np.sin(ph) * u
        if arrow:
            ax.annotate("", xy=tuple(pt), xytext=tuple(apex),
                        arrowprops=dict(arrowstyle="-|>", color=CONE, lw=0.9,
                                        alpha=0.65, shrinkA=3, shrinkB=0), zorder=4)
        else:
            ax.plot([apex[0], pt[0]], [apex[1], pt[1]], color=CONE, lw=0.8,
                    alpha=0.6, zorder=4)
    ends = []
    for sgn in (+1, -1):
        pt = C + sgn * rad * n
        ends.append(pt)
        if arrow:
            ax.annotate("", xy=tuple(pt), xytext=tuple(apex),
                        arrowprops=dict(arrowstyle="-|>", color=CONE, lw=1.8,
                                        shrinkA=3, shrinkB=0), zorder=6)
        else:
            ax.plot([apex[0], pt[0]], [apex[1], pt[1]], color=CONE, lw=1.8, zorder=6)
    return C, rad, n, ends


def mark_angle(ax, vertex, p1, p2, radius, label, color, lab_r, fs=9.5, lw=1.4,
               lab_xy=None, ha="center", va="center"):
    a1 = float(np.degrees(np.arctan2(*(np.asarray(p1) - vertex)[::-1])))
    a2 = float(np.degrees(np.arctan2(*(np.asarray(p2) - vertex)[::-1])))
    sweep = (a2 - a1 + 360) % 360
    if sweep > 180:
        a1, a2, sweep = a2, a1, 360 - sweep
    ax.add_patch(Arc(tuple(vertex), 2 * radius, 2 * radius, theta1=a1, theta2=a2,
                     color=color, lw=lw, zorder=8))
    m = np.radians(a1 + sweep / 2)
    pos = (vertex[0] + lab_r * np.cos(m), vertex[1] + lab_r * np.sin(m))
    if lab_xy is None:
        ax.annotate(label, pos, fontsize=fs, color=color, ha=ha, va=va, zorder=9)
    else:
        ax.annotate(label, pos, xytext=lab_xy, fontsize=fs, color=color, ha=ha, va=va, zorder=9,
                    arrowprops=dict(arrowstyle="-", color=color, lw=0.8, shrinkA=1, shrinkB=1))


fig = plt.figure(figsize=(13.0, 6.2), facecolor=SURFACE)
gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.35], wspace=0.06,
                      left=0.015, right=0.988, top=0.875, bottom=0.235)

# ================= 左：出射方向扫出一个圆锥 =================
axA = fig.add_subplot(gs[0, 0]); axA.set_facecolor(SURFACE)
O = np.array([0.0, 0.0])
CA, radA, nA, endsA = draw_cone(axA, O, v, THETA, 26.0, n_rays=7, arrow=True)
axA.annotate("", xy=tuple(O), xytext=tuple(O - 36 * v),
             arrowprops=dict(arrowstyle="-|>", color=RAY, lw=2.4, shrinkA=0, shrinkB=4), zorder=7)
axA.annotate("入射光子，方向 v", tuple(O - 20 * v + np.array([-2.5, 0])),
             fontsize=9.0, color=RAY, ha="right", va="center", zorder=7)
axA.plot([O[0], (O + 17 * v)[0]], [O[1], (O + 17 * v)[1]],
         color=MUTED, lw=1.0, ls=(0, (2, 2)), zorder=5)
axA.annotate("v 的延长线", tuple(O + 18 * v + np.array([1.0, 0])), fontsize=8.2, color=MUTED,
             ha="left", va="center", zorder=5)
mark_angle(axA, O, O + 15 * v, endsA[0], 11.0, f"θ = {THETA:.0f}°", CONE, lab_r=20.0)
axA.plot([O[0]], [O[1]], marker="o", ms=8.5, color=GOOD, mec="white", mew=1.3, zorder=10)
axA.annotate("P 散射点", (O[0] - 3.0, O[1] - 3.0), fontsize=9.5, color=GOOD,
             ha="right", va="top", zorder=10, weight="bold")
axA.annotate("出射方向可以是这上面的任意一条\nφ 取遍 0~360°，扫出一个半角 θ 的圆锥",
             (-38, 39), fontsize=9.0, color=CONE, ha="left", va="center", zorder=9)
axA.set_title("① 能量只说\"转了多大角 θ\"，没说\"往哪个方位 φ 转\"",
              fontsize=12.0, color=CONE, pad=8, loc="left")
axA.set_xlim(-40, 40); axA.set_ylim(-26, 46)

# ================= 右：候选散射点也是一个圆锥 =================
axB = fig.add_subplot(gs[0, 1]); axB.set_facecolor(SURFACE)
axB.add_patch(Rectangle((-56, -48), 136, 43, fc=OBJ, ec="none", zorder=1))
axB.plot([-56, 80], [-5, -5], color=MUTED, lw=0.8, zorder=2)
axB.annotate("被测物体", (-54, -12), fontsize=8.4, color=INK2, ha="left", va="center", zorder=3)
for x0, w in ((12.5, 67.5), (-56, 43.5)):
    axB.add_patch(Rectangle((x0, 2.5), w, 10.0, fc=D1_COLOR, ec="none", zorder=4))
axB.annotate("D1", (-54, 7.5), fontsize=8.4, color="white", ha="left", va="center", zorder=5)

CB, radB, nB, endsB = draw_cone(axB, H, -v, THETA, 32.0, n_rays=7)
L0, L1 = S - 13 * v, S + 50 * v
axB.plot([L0[0], L1[0]], [L0[1], L1[1]], color=RAY, lw=2.2, zorder=7)
axB.annotate("入射直线（已知，\n由背靠背标记测出来）", tuple(L1 + np.array([-3.0, 0])),
             fontsize=9.0, color=RAY, ha="right", va="center", zorder=8)
axB.plot([H[0], (H - 15 * v)[0]], [H[1], (H - 15 * v)[1]],
         color=CONE, lw=1.0, ls=(0, (4, 3)), zorder=6)
axB.annotate("锥轴 = −v", tuple(H - 16 * v + np.array([2.0, 0])), fontsize=8.4, color=CONE,
             ha="left", va="center", zorder=7)
mark_angle(axB, H, H - 14 * v, endsB[0], 12.0, f"θ = {THETA:.0f}°", CONE, lab_r=15.0,
           lab_xy=(-18.0, 20.0), ha="right")
axB.plot([H[0]], [H[1]], marker="o", ms=6.5, color=INK, mec=SURFACE, mew=1.1, zorder=9)
axB.annotate("H 背散射命中点（测到的）", (H[0] + 2.5, H[1] + 9.0), fontsize=9.0, color=INK,
             ha="left", va="bottom", zorder=9)
axB.plot([S[0]], [S[1]], marker="o", ms=6.5, color=INK, mec=SURFACE, mew=1.1, zorder=9)
axB.annotate("S 源", (S[0] + 2.0, S[1] + 1.5), fontsize=9.0, color=INK,
             ha="left", va="bottom", zorder=9)
axB.plot([P[0]], [P[1]], marker="o", ms=9.5, color=GOOD, mec="white", mew=1.4, zorder=10)
axB.annotate("P = 直线 ∩ 锥面", (P[0] - 3.0, P[1] - 2.0), fontsize=9.5, color=GOOD,
             ha="right", va="top", zorder=10, weight="bold")
axB.annotate("锥面上每一点都满足\"P→H 与 v 成 θ 角\"\n——它就是\"候选散射点\"的全集",
             (-54, 38), fontsize=9.0, color=CONE, ha="left", va="center", zorder=9)
axB.set_title("② 所以\"候选散射点\"的全集，也是一个圆锥面",
              fontsize=12.0, color=CONE, pad=8, loc="left")
axB.set_xlim(-56, 80); axB.set_ylim(-48, 46)

for ax in (axA, axB):
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)

fig.suptitle("圆锥是从哪来的：康普顿公式只定出极角 θ，方位角 φ 完全未知",
             fontsize=14.5, color=INK, x=0.006, ha="left", y=0.972)
fig.text(0.006, 0.018,
         "圆锥不是实物，它是一个集合：\"所有满足'转角 = θ'的方向（左）/ 位置（右）\"。"
         "cos θ = 2 − 511/E′ 只给极角 θ，方位角 φ 一点信息都没有，φ 取遍 0~360° 就扫出锥面。\n"
         "普通康普顿相机到这一步就得停手——它不知道入射方向，只能把锥面投到天球上做反投影；"
         "我们多了背靠背符合标记，入射直线是测出来的，所以线与锥一交，散射点就落成一个点。\n"
         "我们实际不画锥：P 必在\"过 S 沿 v 的那条直线\"上，H 又是测出来的一点——一条直线加一个线外的点只确定一个平面，"
         "P 与 H 都在这个平面里，于是整件事塌回平面几何；锥在该平面上的截口就是图中那两条粗母线，解三角形即可。\n"
         f"图中 θ = {THETA:.0f}°（即 E′ = {Ep:.1f} keV），与 5.4 节同一个构造算例。"
         f"θ > 90° 所以锥是朝后开的——背散射本来就是这样。底面圆按 {FORE:.2f} 的透视系数压扁，纯为示意；其余几何量均现算。",
         fontsize=8.4, color=MUTED, ha="left", va="bottom", linespacing=1.6)

for ex in ("png", "pdf"):
    fig.savefig(os.path.join(_args.out, f"cone_origin.{ex}"), dpi=200,
                facecolor=SURFACE)
print(f"写出 {os.path.join(_args.out, 'cone_origin')}.png / .pdf")
print(f"  θ={THETA}°  E'={Ep:.1f} keV  P=({P[0]:.2f},{P[1]:.2f})  |P−H|={np.linalg.norm(P-H):.2f} mm（母线取 36）")
