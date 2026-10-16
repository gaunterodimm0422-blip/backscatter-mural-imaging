# -*- coding: utf-8 -*-
"""探测头构型示意图 v3：旧构型（M1/M2 实跑） vs 新构型（2026-09-12 用户指定的改动）。

剖面按真实比例绘制，图中不含任何示意性夸张。
数字出处：
  * 立体角、选出率、对比度、k 预言 —— design/m3v2_predict.json（事例级蒙卡，冻结）
  * 几何自洽核查 —— 本脚本现算（与 run_mural_edge_v3.sh 的自检公式相同）
  * D1 质量 —— GAGG 密度 6.63 g/cm^3
橙色标注 = 本次改动项或由改动导出的约束；新构型尚未用 GATE 验证。
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Wedge, FancyArrowPatch
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
_ap.add_argument("--pred", default="outputs/m3v2_predict.json",
                 help="m3_predict.py 输出的 m3v2_predict.json")
_ap.add_argument("--out", default="outputs", help="图件输出目录（默认 outputs）")
_args = _ap.parse_args()
os.makedirs(_args.out, exist_ok=True)

SURFACE = "#fcfcfb"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#9a9891"
D1_COLOR, D2_COLOR = "#2a78d6", "#86b6ef"
CHANGE = "#eb6834"
OBJECT_FILL, SUPPORT_FILL = "#eceae4", "#dedad0"
BEAM = "#f6d9a8"
GAGG = 6.63

SOURCE_Z = 17.5
D2_HALF = 25.0
D2_THICK = 20.0

PRED = {e["tag"].split()[0]: e
        for e in json.load(open(_args.pred, encoding="utf-8"))}
OLD_P, NEW_P = PRED["旧构型"], PRED["新构型"]

CONFIGS = {
    "old": dict(name="旧构型（M1、M2 全部 GATE 运行使用的就是这一套）",
                theta_max=45.0, d1_c=17.5, d1_t=15.0, r_in=20.0, r_out=200.0,
                d2_z0=50.0, pred=OLD_P, mark=False),
    "new": dict(name="新构型（橙色为本次改动与由它导出的约束；尚未用 GATE 验证）",
                theta_max=35.0, d1_c=7.5, d1_t=10.0, r_in=12.5, r_out=100.0,
                d2_z0=42.5, pred=NEW_P, mark=True),
}

X_LIMIT, Z_LOW, Z_HIGH = 214.0, -62.0, 92.0


def d1_mass_kg(r_in, r_out, t):
    return np.pi * (r_out ** 2 - r_in ** 2) * t * 1e-3 * GAGG / 1000.0


def derived(cfg):
    """把所有几何导出量算一遍——与 run_mural_edge_v3.sh 的自检公式一致。"""
    d1_bot = cfg["d1_c"] - cfg["d1_t"] / 2.0
    d1_top = cfg["d1_c"] + cfg["d1_t"] / 2.0
    tan_c = np.tan(np.radians(cfg["theta_max"]))
    r_cone_at_d1bot = (SOURCE_Z - d1_bot) * tan_c
    d2_mid = cfg["d2_z0"] + D2_THICK / 2.0
    L2 = d2_mid - SOURCE_Z
    beta_d2 = np.degrees(np.arctan(D2_HALF / L2))
    beta_eff = min(cfg["theta_max"], beta_d2)
    fov30 = (SOURCE_Z + 30.0) * np.tan(np.radians(beta_eff))
    return dict(d1_bot=d1_bot, d1_top=d1_top, r_cone_at_d1bot=r_cone_at_d1bot,
                margin=cfg["r_in"] - r_cone_at_d1bot, d2_mid=d2_mid, L2=L2,
                beta_d2=beta_d2, beta_eff=beta_eff, fov30=fov30,
                limiter=("发射锥" if cfg["theta_max"] <= beta_d2 else "D2 尺寸"),
                mass=d1_mass_kg(cfg["r_in"], cfg["r_out"], cfg["d1_t"]))


def draw_side_view(ax, cfg):
    g = derived(cfg)
    m = cfg["mark"]
    ax.set_facecolor(SURFACE)

    # 壁画分层
    ax.add_patch(Rectangle((-X_LIMIT, -5.0), 2 * X_LIMIT, 5.0, fc="#f2efe6", ec="none", zorder=1))
    ax.add_patch(Rectangle((-X_LIMIT, -40.0), 2 * X_LIMIT, 35.0, fc=OBJECT_FILL, ec="none", zorder=1))
    ax.add_patch(Rectangle((-X_LIMIT, Z_LOW), 2 * X_LIMIT, -40.0 - Z_LOW, fc=SUPPORT_FILL, ec="none", zorder=1))
    for z in (0.0, -5.0, -40.0):
        ax.plot([-X_LIMIT, X_LIMIT], [z, z], color=MUTED, lw=0.7, zorder=2)
    ax.annotate("细泥 5 mm", (-X_LIMIT + 6, -2.5), fontsize=7.5, color=INK2, va="center")
    ax.annotate("粗泥 35 mm", (-X_LIMIT + 6, -22.0), fontsize=7.5, color=INK2, va="center")
    ax.annotate("砂砾岩支撑体", (-X_LIMIT + 6, -51.0), fontsize=7.5, color=INK2, va="center")
    ax.annotate("物体表面 z = 0", (X_LIMIT - 6, 2.0), fontsize=7.5, color=INK2, ha="right", va="bottom")

    # 有效照射锥（取发射锥与 D2 接受角的较小者）
    half = np.tan(np.radians(g["beta_eff"]))
    x_edge = (SOURCE_Z - Z_LOW) * half
    ax.fill([0, x_edge, -x_edge], [SOURCE_Z, Z_LOW, Z_LOW], color=BEAM, alpha=0.55, zorder=1.5, lw=0)
    for sign in (-1, 1):
        ax.plot([0, sign * x_edge], [SOURCE_Z, Z_LOW], color="#d9a441", lw=0.9, ls=(0, (5, 3)), zorder=2)
    # 发射锥（源宏 maxtheta）
    tanc = np.tan(np.radians(cfg["theta_max"]))
    for sign in (-1, 1):
        ax.plot([0, sign * (SOURCE_Z - Z_LOW) * tanc], [SOURCE_Z, Z_LOW],
                color="#b35c1e", lw=0.9, ls=(0, (2, 2)), zorder=2.2)
    ax.plot([-g["fov30"], g["fov30"]], [-30.0, -30.0], color="#c07d1e", lw=1.8, zorder=3)
    ax.annotate(f"深 30 mm 处视场半宽 {g['fov30']:.1f} mm（受{g['limiter']}限制）",
                (g["fov30"] + 6, -30.0), fontsize=7.8,
                color=(CHANGE if m else "#8a5a12"), va="center", zorder=8)

    # D1 环（剖面上左右两块）
    for sign in (-1, 1):
        x0 = cfg["r_in"] if sign > 0 else -cfg["r_out"]
        ax.add_patch(Rectangle((x0, g["d1_bot"]), cfg["r_out"] - cfg["r_in"], cfg["d1_t"],
                               fc=D1_COLOR, ec=(CHANGE if m else "none"),
                               lw=(1.4 if m else 0), zorder=4))
    # 标签写在 D1 板块内部，避免与源标签抢同一片位置（新构型里源就在 D1 上方 5 mm）
    ax.annotate("D1 背散射探测器（GAGG 环）",
                (-cfg["r_out"] + 5, cfg["d1_c"]), fontsize=8.0, color="white",
                ha="left", va="center", zorder=6)

    # D2
    ax.add_patch(Rectangle((-D2_HALF, cfg["d2_z0"]), 2 * D2_HALF, D2_THICK,
                           fc=D2_COLOR, ec=(CHANGE if m else "none"), lw=(1.4 if m else 0), zorder=4))
    ax.annotate(f"D2 标记探测器 50×50 颗 0.93 mm 晶体，厚 20 mm\n"
                f"下表面 z = {cfg['d2_z0']:.1f} mm，中面离源 {g['L2']:.1f} mm\n"
                f"边缘接受半角 atan(25/{g['L2']:.1f}) = {g['beta_d2']:.1f}°",
                (D2_HALF + 10, cfg["d2_z0"] + D2_THICK / 2), fontsize=8,
                color=(CHANGE if m else "#1c5cab"), va="center", zorder=6)

    # 源
    ax.plot([0], [SOURCE_Z], marker="o", ms=7, color=INK, mec=SURFACE, mew=1.2, zorder=7)
    ax.annotate("Na-22 源 (0, 0, 17.5)", (0, SOURCE_Z),
                xytext=(-0.35 * cfg["r_out"], g["d1_top"] + 8),
                fontsize=8, color=INK, ha="center", va="center", zorder=7,
                arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8,
                                shrinkA=2, shrinkB=6))

    # 内壁直照判据
    ok = g["margin"] >= 0
    ax.annotate(f"{cfg['theta_max']:.0f}° 发射锥在 D1 下表面（z = {g['d1_bot']:.1f}）处半径 "
                f"= {SOURCE_Z - g['d1_bot']:.1f} × tan{cfg['theta_max']:.0f}° = {g['r_cone_at_d1bot']:.2f} mm\n"
                f"内半径实取 {cfg['r_in']:.1f} mm，余量 {g['margin']:+.2f} mm —— "
                f"{'源不直接照射 D1 内壁' if ok else '源会直照 D1 内壁（不可接受）'}",
                (52, -7.0), fontsize=7.8, color=("#b35c1e" if ok else "#c0392b"),
                ha="left", va="top", zorder=8)

    # 尺寸标注
    def dim(x0, x1, z, text, color):
        ax.add_patch(FancyArrowPatch((x0, z), (x1, z), arrowstyle="<->",
                                     mutation_scale=8, color=color, lw=1.1, zorder=8))
        ax.annotate(text, (0.5 * (x0 + x1), z + 3.0), fontsize=8, color=color,
                    ha="center", va="bottom", zorder=8)

    cc = CHANGE if m else INK2
    dim(0, cfg["r_in"], g["d1_top"] + 15, f"内半径 {cfg['r_in']:.1f}", cc)
    dim(cfg["r_in"], cfg["r_out"], g["d1_top"] + 15, f"外半径 {cfg['r_out']:.0f}", cc)

    p = cfg["pred"]
    ax.annotate(
        f"发射锥 {cfg['theta_max']:.0f}°  立体角 {p['solid_angle_sr']:.3f} sr\n"
        f"D1 厚 {cfg['d1_t']:.0f}，内半径 {cfg['r_in']:.1f}，外半径 {cfg['r_out']:.0f} mm\n"
        f"D1 占 z ∈ [{g['d1_bot']:.1f}, {g['d1_top']:.1f}]，下表面离物体 {g['d1_bot']:.1f} mm\n"
        f"每初级粒子选出率 {p['eff_ref']:.4e}\n"
        f"每真实湮没选出率 {p['eff_per_annihilation']:.4e}\n"
        f"四带对比度 " + " / ".join(f"{b['contrast']*100:.1f}%" for b in p["bands"]) + "\n"
        f"横向刻度 k 预言 {p['k_predicted']:.3f}    D1 质量 {g['mass']:.2f} kg",
        (-X_LIMIT + 6, Z_HIGH - 3), fontsize=8.3, color=INK, ha="left", va="top",
        bbox=dict(boxstyle="round,pad=0.42", fc="#f3f2ee", ec="#dedcd6", lw=0.8), zorder=9)

    ax.set_title(cfg["name"], fontsize=12, color=(CHANGE if m else INK), pad=8, loc="left")
    ax.set_xlim(-X_LIMIT, X_LIMIT); ax.set_ylim(Z_LOW, Z_HIGH)
    ax.set_aspect("equal")
    ax.set_xlabel("x (mm)", fontsize=9, color=INK2)
    ax.set_ylabel("z (mm)", fontsize=9, color=INK2)
    ax.tick_params(colors=INK2, labelsize=8)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(MUTED); ax.spines[sp].set_linewidth(0.8)
    return g


def draw_plan_view(ax):
    old, new = CONFIGS["old"], CONFIGS["new"]
    ax.set_facecolor(SURFACE)
    ax.add_patch(Wedge((0, 0), old["r_out"], 0, 360, width=old["r_out"] - old["r_in"],
                       fc="#cfe0f5", ec=D1_COLOR, lw=1.0, zorder=2))
    ax.add_patch(Wedge((0, 0), new["r_out"], 0, 360, width=new["r_out"] - new["r_in"],
                       fc=D1_COLOR, ec=CHANGE, lw=1.4, alpha=0.92, zorder=3))
    ax.add_patch(Rectangle((-D2_HALF, -D2_HALF), 2 * D2_HALF, 2 * D2_HALF,
                           fc="none", ec="#1c5cab", lw=1.2, ls=(0, (4, 3)), zorder=4))
    # 35° 锥在 D1 下表面处的圆
    gn = derived(new)
    ax.add_patch(Wedge((0, 0), gn["r_cone_at_d1bot"], 0, 360, width=gn["r_cone_at_d1bot"],
                       fc="#f6d9a8", ec="#b35c1e", lw=1.0, alpha=0.8, zorder=5))
    ax.plot([0], [0], marker="o", ms=6, color=INK, mec=SURFACE, mew=1.0, zorder=6)
    ax.annotate(f"旧 R {old['r_in']:.0f}–{old['r_out']:.0f} · t{old['d1_t']:.0f}\n"
                f"{d1_mass_kg(old['r_in'], old['r_out'], old['d1_t']):.1f} kg",
                (0, old["r_out"] - 26), fontsize=8, color="#1c5cab", ha="center", va="center", zorder=7)
    ax.annotate(f"新 R {new['r_in']:.1f}–{new['r_out']:.0f} · t{new['d1_t']:.0f}\n"
                f"{d1_mass_kg(new['r_in'], new['r_out'], new['d1_t']):.2f} kg",
                (0, new["r_out"] - 30), fontsize=8.5, color="white", ha="center",
                va="center", zorder=7, weight="bold")
    ax.annotate(f"35° 锥在 D1 下表面处 直径 {2*gn['r_cone_at_d1bot']:.1f} mm\n"
                f"D1 内孔 直径 {2*new['r_in']:.0f} mm，单边余量 {gn['margin']:+.2f} mm",
                (0, -138), fontsize=8, color=CHANGE, ha="center", va="top", zorder=7)
    ax.annotate("D2 投影 50×50", (D2_HALF, D2_HALF), xytext=(150, 62), fontsize=7.5,
                color="#1c5cab", ha="center", va="bottom", zorder=7,
                arrowprops=dict(arrowstyle="-", color="#1c5cab", lw=0.7,
                                shrinkA=2, shrinkB=2))
    ax.set_title("D1 俯视（沿 z 看）", fontsize=12, color=INK, pad=8, loc="left")
    ax.set_xlim(-212, 212); ax.set_ylim(-212, 212)
    ax.set_aspect("equal")
    ax.set_xlabel("x (mm)", fontsize=9, color=INK2)
    ax.set_ylabel("y (mm)", fontsize=9, color=INK2)
    ax.tick_params(colors=INK2, labelsize=8)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(MUTED); ax.spines[sp].set_linewidth(0.8)


fig = plt.figure(figsize=(15.4, 8.6), facecolor=SURFACE)
gs = fig.add_gridspec(2, 2, width_ratios=[2.45, 1.0], height_ratios=[1, 1],
                      hspace=0.32, wspace=0.13, left=0.05, right=0.985, top=0.882, bottom=0.088)
g_old = draw_side_view(fig.add_subplot(gs[0, 0]), CONFIGS["old"])
g_new = draw_side_view(fig.add_subplot(gs[1, 0]), CONFIGS["new"])
draw_plan_view(fig.add_subplot(gs[:, 1]))

r_prim = NEW_P["eff_ref"] / OLD_P["eff_ref"]
r_ann = NEW_P["eff_per_annihilation"] / OLD_P["eff_per_annihilation"]

fig.suptitle("双 511 keV 符合标记背散射成像 —— 探测头构型：旧 vs 新（剖面按真实比例）",
             fontsize=14, color=INK, x=0.006, ha="left", y=0.975)
fig.text(0.006, 0.010,
         f"选出率、对比度、k 预言来自 design/m3v2_predict.json（事例级蒙卡，已冻结）。"
         f"新/旧：每个初级粒子 {r_prim:.3f}×（GATE 日志里直接看到的），"
         f"每次真实湮没 {r_ann:.3f}×（决定实际测量时长的）——只报前者会得出"
         f"“计数率没变”的错误结论。"
         f"D1 外半径 100 mm 与 D1 中心 z = 7.5 mm 是推断值，非用户指定。新构型尚未用 GATE 验证。",
         fontsize=8.2, color=MUTED, ha="left")
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(_args.out, f"head_geometry_v3.{ext}"), dpi=200,
                facecolor=SURFACE)

print(f"写出 {os.path.join(_args.out, 'head_geometry_v3')}.png / .pdf")
for tag, g, c in (("旧", g_old, CONFIGS["old"]), ("新", g_new, CONFIGS["new"])):
    print(f"{tag}构型  锥{c['theta_max']:.0f}°  D1 z∈[{g['d1_bot']:.1f},{g['d1_top']:.1f}] "
          f"R{c['r_in']:.1f}-{c['r_out']:.0f}  余量{g['margin']:+.2f} mm  "
          f"D2中面离源{g['L2']:.1f}  接受{g['beta_d2']:.1f}°  有效锥{g['beta_eff']:.1f}°({g['limiter']})  "
          f"FOV30={g['fov30']:.1f} mm  D1质量{g['mass']:.2f} kg")
print(f"选出率比 新/旧：每初级 {r_prim:.3f}×   每真实湮没 {r_ann:.3f}×")
