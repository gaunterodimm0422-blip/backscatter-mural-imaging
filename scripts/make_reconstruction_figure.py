# -*- coding: utf-8 -*-
"""第五节配图：闭式散射点重建的两种等价解法（同一个事件）。

左：我们用的"解三角形"（正弦定理，一步到位，自动落在物理根上）
右：导师写法"直线 ∩ 圆锥"（两边平方 -> 二次方程两个根，需要符号判据剔除伪根）

事件参数（T、H、θ）是为了把"两个根都为正"这件事讲清楚而构造的一个自洽算例，
**不是 GATE 实测事件**；探测器坐标取自新构型。
图上所有角度、长度、两个根都由脚本现算，未手填；正弦定理与二次方程的一致性有断言把关。
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Arc, Polygon
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
D1_COLOR, D2_COLOR = "#2a78d6", "#86b6ef"
OBJECT_FILL, SUPPORT_FILL = "#eceae4", "#dedad0"
GOOD, BAD, RAY, CONE = "#1a7f4b", "#c0392b", "#b35c1e", "#6b4fbf"

# ---------------- 事件（全部现算） ----------------
S = np.array([0.0, 17.5])        # 源 / 湮没点
T = np.array([-15.0, 52.5])      # D2 上的标记命中点
H = np.array([30.0, 7.5])        # D1 上的背散射命中点
THETA = 120.0                    # 散射角，由 D1 测到的 E' 反解

v = (S - T) / np.linalg.norm(S - T)
d = H - S
nd = float(np.linalg.norm(d))
a = float(v @ d)
alpha = float(np.degrees(np.arccos(a / nd)))
c = float(np.cos(np.radians(THETA)))
Ep = 511.0 / (2.0 - c)
q = (a * a - c * c * nd * nd) / (1.0 - c * c)
root = float(np.sqrt(a * a - q))
t_plus, t_minus = a + root, a - root
P, P_bad = S + t_plus * v, S + t_minus * v
t_sine = nd * np.sin(np.radians(THETA - alpha)) / np.sin(np.radians(THETA))
assert abs(t_sine - t_plus) < 1e-9, "正弦定理与二次方程不一致，图作废"
assert t_minus > 0, "这个算例本来是要演示两个根都为正的"

X0, X1, Z0, Z1 = -50.0, 64.0, -44.0, 68.0


def scene(ax):
    ax.set_facecolor(SURFACE)
    ax.add_patch(Rectangle((X0, -5.0), X1 - X0, 5.0, fc="#f2efe6", ec="none", zorder=1))
    ax.add_patch(Rectangle((X0, -40.0), X1 - X0, 35.0, fc=OBJECT_FILL, ec="none", zorder=1))
    ax.add_patch(Rectangle((X0, Z0), X1 - X0, -40.0 - Z0, fc=SUPPORT_FILL, ec="none", zorder=1))
    for z in (0.0, -5.0, -40.0):
        ax.plot([X0, X1], [z, z], color=MUTED, lw=0.7, zorder=2)
    ax.annotate("细泥 5 mm", (X0 + 2, -2.5), fontsize=7, color=INK2, va="center", zorder=3)
    ax.annotate("粗泥 35 mm", (X0 + 2, -36.5), fontsize=7, color=INK2, va="center", zorder=3)
    ax.annotate("砂砾岩", (X0 + 2, -42.0), fontsize=7, color=INK2, va="center", zorder=3)
    for x0, w in ((12.5, X1 - 12.5), (X0, -12.5 - X0)):
        ax.add_patch(Rectangle((x0, 2.5), w, 10.0, fc=D1_COLOR, ec="none", zorder=4))
    ax.annotate("D1 背散射探测器（环，内半径 12.5）", (X0 + 2, 7.5), fontsize=7,
                color="white", ha="left", va="center", zorder=5)
    ax.add_patch(Rectangle((-25, 42.5), 50, 20, fc=D2_COLOR, ec="none", zorder=4))
    ax.annotate("D2 标记探测器", (25 + 2, 52.5), fontsize=7, color="#1c5cab",
                ha="left", va="center", zorder=5)
    ax.annotate("", xy=tuple(T), xytext=tuple(S),
                arrowprops=dict(arrowstyle="-|>", color=D2_COLOR, lw=1.6,
                                shrinkA=4, shrinkB=0), zorder=6)
    ax.annotate("标记光子 511 keV\n（背靠背，由它定出 v）",
                (0.5 * (S[0] + T[0]) - 2.5, 0.5 * (S[1] + T[1])),
                fontsize=7.2, color="#1c5cab", ha="right", va="center", zorder=6)
    for pt, name, dx, dy, ha, va in ((S, "S 源", -2.5, 1.5, "right", "bottom"),
                                     (T, "T 标记命中点", 2.0, 2.0, "left", "bottom"),
                                     (H, "H 背散射命中点", 2.0, 3.0, "left", "bottom")):
        ax.plot([pt[0]], [pt[1]], marker="o", ms=6, color=INK, mec=SURFACE, mew=1.1, zorder=9)
        ax.annotate(name, (pt[0] + dx, pt[1] + dy), fontsize=8.4, color=INK,
                    ha=ha, va=va, zorder=9)


def finish(ax, title, color):
    ax.set_title(title, fontsize=12.5, color=color, pad=8, loc="left")
    ax.set_xlim(X0, X1); ax.set_ylim(Z0, Z1); ax.set_aspect("equal")
    ax.set_xlabel("x (mm)", fontsize=8.5, color=INK2)
    ax.set_ylabel("z (mm)", fontsize=8.5, color=INK2)
    ax.tick_params(colors=INK2, labelsize=7.5)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(MUTED); ax.spines[sp].set_linewidth(0.8)


def mark_angle(ax, vertex, p1, p2, radius, label, color, lab_r, fs=8.6, lw=1.4,
               lab_xy=None, ha="center", va="center"):
    """标出 p1-vertex-p2 的夹角（走 <180° 的那一侧），返回标签落点。"""
    a1 = float(np.degrees(np.arctan2(*(np.asarray(p1) - vertex)[::-1])))
    a2 = float(np.degrees(np.arctan2(*(np.asarray(p2) - vertex)[::-1])))
    sweep = (a2 - a1 + 360) % 360
    if sweep > 180:
        a1, a2, sweep = a2, a1, 360 - sweep
    ax.add_patch(Arc(tuple(vertex), 2 * radius, 2 * radius, angle=0,
                     theta1=a1, theta2=a2, color=color, lw=lw, zorder=8))
    mid = np.radians(a1 + sweep / 2)
    pos = (vertex[0] + lab_r * np.cos(mid), vertex[1] + lab_r * np.sin(mid))
    if lab_xy is None:
        ax.annotate(label, pos, fontsize=fs, color=color, ha=ha, va=va, zorder=9)
    else:
        ax.annotate(label, pos, xytext=lab_xy, fontsize=fs, color=color,
                    ha=ha, va=va, zorder=9,
                    arrowprops=dict(arrowstyle="-", color=color, lw=0.8,
                                    shrinkA=1, shrinkB=1))
    return pos


fig = plt.figure(figsize=(15.2, 10.4), facecolor=SURFACE)
gs = fig.add_gridspec(1, 2, wspace=0.13, left=0.045, right=0.988, top=0.905, bottom=0.335)

# ==================== 左：解三角形 ====================
axA = fig.add_subplot(gs[0, 0])
scene(axA)
axA.add_patch(Polygon([S, P, H], closed=True, fc="#f6d9a8", ec="none", alpha=0.6, zorder=3))
axA.annotate("", xy=tuple(P), xytext=tuple(S),
             arrowprops=dict(arrowstyle="-|>", color=RAY, lw=2.1, shrinkA=4, shrinkB=3), zorder=7)
axA.annotate("", xy=tuple(H), xytext=tuple(P),
             arrowprops=dict(arrowstyle="-|>", color=RAY, lw=2.1, shrinkA=3, shrinkB=4), zorder=7)
axA.plot([S[0], H[0]], [S[1], H[1]], color=INK2, lw=1.4, ls=(0, (5, 3)), zorder=6)
ext = P + 22 * v
axA.plot([P[0], ext[0]], [P[1], ext[1]], color=MUTED, lw=1.0, ls=(0, (2, 2)), zorder=6)
axA.annotate("v 的延长线", tuple(ext + np.array([1.5, -1.0])), fontsize=7,
             color=MUTED, ha="left", va="top", zorder=6)

mSP, mSH = 0.5 * (S + P), 0.5 * (S + H)
axA.annotate(f"沿入射方向 v 走 t\nt = {t_plus:.2f} mm（待求）", tuple(mSP),
             xytext=(-2.0, -7.0), fontsize=8.8, color=RAY, ha="right", va="center", zorder=8,
             arrowprops=dict(arrowstyle="-", color=RAY, lw=0.8, shrinkA=2, shrinkB=3))
axA.annotate(f"|d| = {nd:.2f} mm\n（已知，= H − S）", (mSH[0] - 1.0, mSH[1] + 2.5),
             fontsize=8.4, color=INK2, ha="center", va="bottom", zorder=8)

mark_angle(axA, S, P, H, 6.5, f"α = {alpha:.1f}°", RAY, lab_r=10.5)
mark_angle(axA, H, S, P, 8.0, f"θ − α = {THETA - alpha:.1f}°", GOOD, lab_r=11.0,
           lab_xy=(H[0] + 5.0, H[1] - 11.0), ha="left", va="center")
mark_angle(axA, P, S, H, 6.5, f"180° − θ = {180 - THETA:.0f}°", INK, lab_r=13.5, fs=8.2)
mark_angle(axA, P, ext, H, 14.0, f"θ = {THETA:.0f}°  （散射角）", CONE, lab_r=21.0, fs=9.2, lw=1.6)

axA.plot([P[0]], [P[1]], marker="o", ms=8.5, color=GOOD, mec="white", mew=1.4, zorder=10)
axA.annotate("P 散射点", (P[0] - 3.0, P[1] - 2.5), fontsize=9.2, color=GOOD,
             ha="right", va="top", zorder=10, weight="bold")
finish(axA, "我们用的解法：解三角形（正弦定理）", GOOD)

# ==================== 右：直线 ∩ 圆锥 ====================
axB = fig.add_subplot(gs[0, 1])
scene(axB)
lo, hi = -13.0, 50.0
L0, L1 = S + lo * v, S + hi * v
axB.plot([L0[0], L1[0]], [L0[1], L1[1]], color=RAY, lw=2.0, zorder=6)
axB.annotate("入射直线 S + t·v", (L0[0] + 2.5, L0[1] - 1.0), fontsize=8.4,
             color=RAY, ha="left", va="top", zorder=7)

axis = -v
base = float(np.degrees(np.arctan2(axis[1], axis[0])))


def ray(deg):
    r = np.radians(deg)
    return np.array([np.cos(r), np.sin(r)])


LEN = 46.0
axB.plot([H[0], (H + 17 * axis)[0]], [H[1], (H + 17 * axis)[1]],
         color=CONE, lw=1.0, ls=(0, (4, 3)), zorder=6)
axB.annotate("锥轴 = −v", tuple(H + 18 * axis + np.array([-1.0, 1.0])), fontsize=7.5,
             color=CONE, ha="right", va="bottom", zorder=7)
# 真锥面：半角 θ
for sgn in (+1, -1):
    e = H + LEN * ray(base + sgn * THETA)
    axB.plot([H[0], e[0]], [H[1], e[1]], color=CONE, lw=1.9, zorder=6)
# 伪锥面：半角 180° − θ（平方时被带进来的那一叶）
for sgn in (+1, -1):
    e = H + LEN * ray(base + sgn * (180.0 - THETA))
    axB.plot([H[0], e[0]], [H[1], e[1]], color=BAD, lw=1.5, ls=(0, (5, 3)), zorder=6)

mark_angle(axB, H, H + 17 * axis, H + 20 * ray(base - THETA), 12.0,
           f"θ = {THETA:.0f}°", CONE, lab_r=17.5, fs=9.2, lw=1.6)
mark_angle(axB, H, H + 17 * axis, H + 20 * ray(base + 180.0 - THETA), 7.0,
           f"180° − θ = {180 - THETA:.0f}°", BAD, lab_r=13.0, fs=8.2, lw=1.3)

axB.annotate(f"真锥面：半角 θ = {THETA:.0f}°\n（物理解落在这上面）",
             tuple(H + LEN * ray(base + THETA) + np.array([-1.5, -1.5])),
             fontsize=7.8, color=CONE, ha="right", va="top", zorder=7)
axB.annotate(f"伪锥面：半角 180° − θ = {180 - THETA:.0f}°\n（两边平方时被带进来的）",
             (46.0, 33.0), fontsize=7.8, color=BAD, ha="right", va="bottom", zorder=7)

axB.plot([P[0]], [P[1]], marker="o", ms=9, color=GOOD, mec="white", mew=1.4, zorder=10)
axB.annotate(f"P   t1 = {t_plus:.2f} mm\n物理根 √", (P[0] - 3.0, P[1] - 2.5), fontsize=9.0,
             color=GOOD, ha="right", va="top", zorder=10, weight="bold")
axB.plot([P_bad[0]], [P_bad[1]], marker="X", ms=10, color=BAD, mec="white", mew=1.2, zorder=10)
axB.annotate(f"P′   t2 = {t_minus:.2f} mm\n伪根 ×（注意它也 > 0）",
             tuple(P_bad), xytext=(-8.0, 25.0),
             fontsize=9.0, color=BAD, ha="right", va="bottom", zorder=10, weight="bold",
             arrowprops=dict(arrowstyle="-", color=BAD, lw=0.9, shrinkA=2, shrinkB=6))
finish(axB, "导师写法：直线 ∩ 圆锥（同一个事件）", CONE)

# ==================== 两个公式框 ====================
BOXA = ("三点 S、P、H 共面，构成一个三角形：\n"
        "   在 S 处内角 = α = arccos( v·d / |d| )                    ← 纯几何，只用命中点位置\n"
        "   在 P 处内角 = 180° − θ （入射方向转到出射方向转过 θ）      ← θ 由 D1 测到的 E′ 给出\n"
        "   在 H 处内角 = 180° − α − (180° − θ) = θ − α\n\n"
        f"cos θ = 2 − 511/E′ ，E′ = {Ep:.1f} keV  →  θ = {THETA:.0f}°\n"
        "正弦定理：  t / sin(θ − α)  =  |d| / sin(180° − θ)  =  |d| / sin θ\n"
        f"      t = |d|·sin(θ − α) / sin θ = {nd:.2f} × sin {THETA - alpha:.1f}° / sin {THETA:.0f}° "
        f"= {t_plus:.2f} mm\n"
        f"      P = S + t·v = ({P[0]:.2f}, {P[1]:.2f}) mm\n\n"
        "五行，没有迭代、没有查表、没有根的取舍 —— 正弦定理只给一个解，而且它就是物理解。")

BOXB = ("把“夹角等于 θ”这个条件两边平方，化成二次方程：\n"
        "      t² − 2a·t + (a² − c²|d|²)/(1 − c²) = 0 ，   a = v·d ，  c = cos θ\n"
        f"      a = {a:.2f} mm ，  c = {c:+.3f} ，  |d| = {nd:.2f} mm\n"
        f"      两根  t1 = {t_plus:.2f} mm ，  t2 = {t_minus:.2f} mm   ← 都 > 0，光靠 t > 0 去不掉\n\n"
        "为什么会多出一个根：平方以后，方程描述的不再是半角 θ 的那一叶锥面，\n"
        f"而是“半角 θ”与“半角 180° − θ = {180 - THETA:.0f}°”两叶的并集。P′ 就落在后一叶上——\n"
        f"在 P′ 处，v 与 P′→H 的夹角是 {180 - THETA:.0f}°，不是 {THETA:.0f}°。\n\n"
        "所以必须补一条符号判据（平方前的等式要求 sign(a − t) = sign(cos θ)）：\n"
        f"      t1 ： a − t = {a - t_plus:+.2f} ，与 cos θ = {c:+.3f} 同号   → 保留\n"
        f"      t2 ： a − t = {a - t_minus:+.2f} ，与 cos θ 反号             → 剔除")

fig.text(0.045, 0.300, BOXA, fontsize=8.4, color=INK, ha="left", va="top", linespacing=1.45,
         bbox=dict(boxstyle="round,pad=0.55", fc="#eef6f0", ec="#bcd9c8", lw=0.9))
fig.text(0.527, 0.300, BOXB, fontsize=8.4, color=INK, ha="left", va="top", linespacing=1.45,
         bbox=dict(boxstyle="round,pad=0.55", fc="#f3f0fb", ec="#cec1ea", lw=0.9))

fig.suptitle("闭式散射点重建：两种等价解法（画的是同一个事件）", fontsize=14.5, color=INK,
             x=0.006, ha="left", y=0.975)
fig.text(0.006, 0.012,
         f"两法数值完全等价：正弦定理给出 t = {t_sine:.6f} mm，二次方程的物理根 t1 = {t_plus:.6f} mm，"
         f"差 {abs(t_sine - t_plus):.1e} mm；在全量 4000 事件、2088 个有解事件的逐事件对比中，两法最大差 0.000000 mm。"
         f"区别只在于：三角形解法自动落在物理根上，不需要符号判据，代码分支更少、出错面更小。\n"
         f"图中探测器坐标取自新构型（源 z = 17.5；D1 z ∈ [2.5, 12.5]，内半径 12.5、外半径 100；D2 z ∈ [42.5, 62.5]，半宽 25）。"
         f"事件参数（T、H、θ）是为了演示“两个根都为正”而构造的一个自洽算例，不是 GATE 实测事件；"
         f"图上所有角度、长度与两个根均由脚本现算，未手填。",
         fontsize=8.0, color=MUTED, ha="left", va="bottom", linespacing=1.5)

for ex in ("png", "pdf"):
    fig.savefig(os.path.join(_args.out, f"closed_form_reconstruction.{ex}"),
                dpi=200, facecolor=SURFACE)
print(f"写出 {os.path.join(_args.out, 'closed_form_reconstruction')}.png / .pdf")
print(f"  v={v.round(4)}  |d|={nd:.3f}  a={a:.3f}  α={alpha:.2f}°  θ={THETA}°  E'={Ep:.1f} keV")
print(f"  t1={t_plus:.4f}  t2={t_minus:.4f}  正弦定理 t={t_sine:.4f}  差 {abs(t_sine-t_plus):.2e} mm")
print(f"  P=({P[0]:.2f}, {P[1]:.2f})   P'=({P_bad[0]:.2f}, {P_bad[1]:.2f})")
print(f"  符号判据: t1 -> a-t={a-t_plus:+.2f}; t2 -> a-t={a-t_minus:+.2f}; cosθ={c:+.3f}")


# ======================================================================
# 紧凑版：专供插进 .docx（两张，上下排）
# ----------------------------------------------------------------------
# 为什么要单独画：doc 的 docx 导出器对插图有硬上限，实测（800×500 / 1600×1000 /
# 3040×2080 三种像素数，96/300/340 三种 dpi，1.25/1.50/0.67/2.14 四种宽高比）
# 得到的规律是：保持宽高比，缩到 **最宽 3.12 英寸、最高 2.33 英寸** 为止；
# 像素数与 dpi 都不影响结果，{w=...} 属性 docx 导出器也不认。
# 所以面积最大的宽高比是 3.12/2.33 = 1.34，两个面板必须拆成两张分别插。
# 这一版按"最终就是 3.12 × 2.33 英寸"设计：字号就是纸面上的字号。
# 详图（15 英寸宽、带完整推导）仍作为独立文件交付。
# ======================================================================
KX0, KX1, KZ0, KZ1 = -27.0, 56.0, -35.0, 22.0
FS, FSS = 6.0, 5.3


def kscene(ax):
    ax.set_facecolor(SURFACE)
    ax.add_patch(Rectangle((KX0, -5.0), KX1 - KX0, 5.0, fc="#f2efe6", ec="none", zorder=1))
    ax.add_patch(Rectangle((KX0, KZ0), KX1 - KX0, -5.0 - KZ0, fc=OBJECT_FILL, ec="none", zorder=1))
    for z in (0.0, -5.0):
        ax.plot([KX0, KX1], [z, z], color=MUTED, lw=0.6, zorder=2)
    ax.annotate("被测物体（表面 z = 0）", (KX1 - 1.0, -2.5), fontsize=FSS, color=INK2,
                ha="right", va="center", zorder=3)
    for x0, w in ((12.5, KX1 - 12.5), (KX0, -12.5 - KX0)):
        ax.add_patch(Rectangle((x0, 2.5), w, 10.0, fc=D1_COLOR, ec="none", zorder=4))
    ax.annotate("D1", (KX0 + 1.2, 7.5), fontsize=FS, color="white", ha="left", va="center", zorder=5)
    ax.annotate("D1", (KX1 - 1.2, 7.5), fontsize=FS, color="white", ha="right", va="center", zorder=5)
    # T 在画外，用一段箭头代表"标记光子来自 D2 上的 T"
    stub = S + 4.0 * (T - S) / np.linalg.norm(T - S)
    ax.annotate("", xy=tuple(stub), xytext=tuple(S),
                arrowprops=dict(arrowstyle="-|>", color=D2_COLOR, lw=1.1,
                                shrinkA=2.5, shrinkB=0), zorder=6)
    ax.annotate("标记光子飞向 D2 上的 T\n（由 T 反推出方向 v）", (stub[0] - 2.0, stub[1] - 1.5),
                fontsize=FSS, color="#1c5cab", ha="right", va="center", zorder=6)
    for pt, nm, dx, dy, ha, va in ((S, "S 源", 1.5, 1.2, "left", "bottom"),
                                   (H, "H", 1.5, 1.5, "left", "bottom")):
        ax.plot([pt[0]], [pt[1]], marker="o", ms=3.4, color=INK, mec=SURFACE, mew=0.8, zorder=9)
        ax.annotate(nm, (pt[0] + dx, pt[1] + dy), fontsize=FS, color=INK,
                    ha=ha, va=va, zorder=9)
    ax.set_xlim(KX0, KX1); ax.set_ylim(KZ0, KZ1); ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)


def new_panel():
    f = plt.figure(figsize=(3.12, 2.33), facecolor=SURFACE)
    ax = f.add_axes([0.005, 0.005, 0.99, 0.90])
    kscene(ax)
    return f, ax


# ---- ①：解三角形 ----
f1, ax1 = new_panel()
ax1.add_patch(Polygon([S, P, H], closed=True, fc="#f6d9a8", ec="none", alpha=0.65, zorder=3))
ax1.annotate("", xy=tuple(P), xytext=tuple(S),
             arrowprops=dict(arrowstyle="-|>", color=RAY, lw=1.5, shrinkA=2.5, shrinkB=2), zorder=7)
ax1.annotate("", xy=tuple(H), xytext=tuple(P),
             arrowprops=dict(arrowstyle="-|>", color=RAY, lw=1.5, shrinkA=2, shrinkB=2.5), zorder=7)
ax1.plot([S[0], H[0]], [S[1], H[1]], color=INK2, lw=0.9, ls=(0, (4, 2.5)), zorder=6)
ext1 = P + 11 * v
ax1.plot([P[0], ext1[0]], [P[1], ext1[1]], color=MUTED, lw=0.8, ls=(0, (2, 2)), zorder=6)
ax1.annotate("|d| 已知", (16.0, 13.8), fontsize=FS, color=INK2, ha="center", va="bottom", zorder=8)
ax1.annotate("t 待求", tuple(0.5 * (S + P)), xytext=(-3.0, -4.0), fontsize=FS, color=RAY,
             ha="right", va="center", zorder=8,
             arrowprops=dict(arrowstyle="-", color=RAY, lw=0.6, shrinkA=1, shrinkB=2))
mark_angle(ax1, S, P, H, 4.5, "α", RAY, lab_r=7.2, fs=FS + 1.0, lw=1.0)
mark_angle(ax1, P, S, H, 4.5, "180°−θ", INK, lab_r=9.5, fs=FS, lw=1.0)
mark_angle(ax1, H, S, P, 5.5, "θ−α", GOOD, lab_r=8.0,
           lab_xy=(H[0] + 3.0, H[1] - 7.5), ha="left", va="center", fs=FS + 1.0, lw=1.0)
mark_angle(ax1, P, ext1, H, 8.5, f"θ = {THETA:.0f}°", CONE, lab_r=13.5, fs=FS + 1.0, lw=1.1)
ax1.plot([P[0]], [P[1]], marker="o", ms=4.6, color=GOOD, mec="white", mew=0.9, zorder=10)
ax1.annotate(f"P   t = {t_plus:.2f} mm", (P[0] - 1.8, P[1] - 1.6), fontsize=FS + 0.8, color=GOOD,
             ha="right", va="top", zorder=10, weight="bold")
f1.suptitle("① 解三角形（正弦定理）：只有一个解，而且就是物理解",
            fontsize=6.8, color=GOOD, x=0.008, ha="left", y=0.985, va="top")
for ex in ("png", "pdf"):
    f1.savefig(os.path.join(_args.out, f"closed_form_doc_a.{ex}"), dpi=400,
               facecolor=SURFACE)

# ---- ②：直线 ∩ 圆锥 ----
f2, ax2 = new_panel()
Lk0, Lk1 = S + (-3.0) * v, S + 44.0 * v
ax2.plot([Lk0[0], Lk1[0]], [Lk0[1], Lk1[1]], color=RAY, lw=1.5, zorder=6)
ax2.annotate("入射直线", (Lk1[0] + 2.2, Lk1[1] - 1.0), fontsize=FS, color=RAY,
             ha="left", va="top", zorder=7)
KLEN = 42.0
for sgn in (+1, -1):
    e = H + KLEN * ray(base + sgn * THETA)
    ax2.plot([H[0], e[0]], [H[1], e[1]], color=CONE, lw=1.3, zorder=6)
    e = H + KLEN * ray(base + sgn * (180.0 - THETA))
    ax2.plot([H[0], e[0]], [H[1], e[1]], color=BAD, lw=1.1, ls=(0, (4, 2.5)), zorder=6)
ax2.annotate("真锥面 半角 θ（物理解在这上面）", (KX0 + 1.0, -31.0), fontsize=FSS, color=CONE,
             ha="left", va="center", zorder=7)
ax2.annotate("伪锥面 半角 180°−θ\n（平方带进来的）", (KX1 - 1.0, 16.0), fontsize=FSS, color=BAD,
             ha="right", va="center", zorder=7)
ax2.plot([P[0]], [P[1]], marker="o", ms=4.6, color=GOOD, mec="white", mew=0.9, zorder=10)
ax2.annotate(f"P   t1 = {t_plus:.2f}\n物理根", (P[0] + 2.2, P[1] + 0.5), fontsize=FS + 0.8,
             color=GOOD, ha="left", va="center", zorder=10, weight="bold")
ax2.plot([P_bad[0]], [P_bad[1]], marker="X", ms=5.2, color=BAD, mec="white", mew=0.8, zorder=10)
ax2.annotate(f"P′  t2 = {t_minus:.2f}\n伪根（也 > 0）", tuple(P_bad), xytext=(-5.0, -3.0),
             fontsize=FS + 0.8, color=BAD, ha="right", va="center", zorder=10, weight="bold",
             arrowprops=dict(arrowstyle="-", color=BAD, lw=0.6, shrinkA=1, shrinkB=3))
f2.suptitle("② 直线 ∩ 圆锥：两个根都 > 0，要靠符号判据挑",
            fontsize=6.8, color=CONE, x=0.008, ha="left", y=0.985, va="top")
for ex in ("png", "pdf"):
    f2.savefig(os.path.join(_args.out, f"closed_form_doc_b.{ex}"), dpi=400,
               facecolor=SURFACE)

print(f"写出 {os.path.join(_args.out, 'closed_form_doc_a')} / closed_form_doc_b"
      "（.png/.pdf，各 3.12×2.33 英寸，专供 docx）")
