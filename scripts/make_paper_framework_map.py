# -*- coding: utf-8 -*-
"""把"仿真论文"的框架画成《韩景逸-框架》那种思维导图，一章一页。

颜色即状态：黑 = 数据已在手上；红 = 缺；橙 = 有但有前提（用了真值/上界/未冻结）。
每个叶子后面的中括号是数据出处或状态，拼稿时照着去项目文档里取数。
"""
import textwrap
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import font_manager

# 字体：Windows 主力机上没有 Noto CJK，必须先找 微软雅黑/黑体，否则中文全是方块。
# DejaVu Sans 垫底，补 U+2212 这类中文字体常缺的符号。
_CJK = ("Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Noto Sans CJK JP",
        "Noto Sans CJK HK", "WenQuanYi Zen Hei")
_stack = [fam for fam in _CJK
          if any(fam in f.name for f in font_manager.fontManager.ttflist)]
plt.rcParams["font.family"] = _stack + ["DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["pdf.fonttype"] = 42

import argparse
import os
_ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
_ap.add_argument("--out", default="outputs", help="图件输出目录（默认 outputs）")
_args = _ap.parse_args()
os.makedirs(_args.out, exist_ok=True)

SURFACE = "#fcfcfb"
HAVE, MISS, CARE = "#0b0b0b", "#c0392b", "#b35c1e"
C = {"有": HAVE, "缺": MISS, "慎": CARE}

CHAPTERS = [
 ("1  引言", [
   ("第一段：壁画空鼓的检测需求", [
     ("1.1.1 空鼓与修复中的边界定位需求；文保依据待补", "缺"),
     ("1.1.2 敲击、红外、雷达、超声的适用范围与局限；引用待补", "缺")]),
   ("第二段：为什么只能单侧", [
     ("1.2.1 墙上壁画的背面通常不可达，研究单侧非接触观测", "有")]),
   ("第三段：符合标记背散射（IBAG）", [
     ("1.3.1 双511 keV符合标记提供逐事件入射方向约束；非无误差探针", "有"),
     ("1.3.2 CSIRO、LANL、SAIC、PAITI等文献线索；引用与历史定位待核", "慎"),
     ("1.3.3 本文研究问题：构型取舍、边界判定、负对照与厚度扫描", "有")]),
   ("第四段：研究目标与贡献点", [
     ("1.4.1 重建实现、误差机制、构型比较及厚度扫描，对应第3章", "有")]),
 ]),
 ("2  原理、构型与仿真模型的建立", [
   ("2.1  符合标记背散射的原理", [
     ("2.1.1 理想背靠背511 keV源；D2位置约束物体侧光子入射线", "有"),
     ("2.1.2 单次康普顿能量约束散射角；D1位置与入射线共同定点", "有")]),
   ("2.2  闭式散射点重建", [
     ("2.2.1 正弦定理闭式解；角度、路程合法时选取物理单解", "有"),
     ("2.2.2 历史4000事例自检与线锥求交一致；伪根来自平方", "有"),
     ("2.2.3 当前Python状态码0-4：有效及方向、能量、几何、路程失败", "有")]),
   ("2.3  探测头构型", [
     ("2.3.1 D1环形GAGG背散射探测器；D2像素化GAGG标记探测器", "有"),
     ("2.3.2 M3孔径下限10.50 mm，取12.5 mm；35°锥与D2接受匹配", "有"),
     ("2.3.3 孔径与外半径历史扫描作构型依据；不混入当前M3性能", "慎")]),
   ("2.4  仿真模型的建立", [
     ("2.4.1 GATE/Geant4体模：细泥5、粗泥35、支撑体100 mm；材料为假设", "有"),
     ("2.4.2 合成→响应→筛选→重建；当前配对仍用模拟事例与初级身份", "慎"),
     ("2.4.3 D1 8%、D2 15%@511 keV；读出体素化后按180-250/400-600筛选", "有"),
     ("2.4.4 分别报告每模拟初级与每真实湮没的选出率", "有")]),
 ]),
 ("3  仿真结果与分析", [
   ("3.1  横向与深度误差的来源", [
     ("3.1.1 局部角度误差约5.5°@90°、18.9°@150°；180°线性传播失效", "有"),
     ("3.1.2 历史绝对误差中位：横向1.91、深度11.38 mm；非FWHM", "慎"),
     ("3.1.3 15%深度残差中央68%半宽15.09→10.93 mm，缩小27.6%", "有")]),
   ("3.2  估计量与判定框架", [
     ("3.2.1 四个重建深度带共享边界的二项似然；图上对比度≠拟合c_b", "有"),
     ("3.2.2 零分布蒙卡标定与冻结阈值T=30；不直接用Wilks卡方", "有"),
     ("3.2.3 负对照、随机对半及换参照场交叉检查", "有"),
     ("3.2.4 M3负对照T=25.40<30，但超出2000次零样本最大值23.44", "慎")]),
   ("3.3  构型优化与结果（旧构型作对照）", [
     ("3.3.1 M2对照与M3主构型：锥35°、D1厚10、孔半径12.5、下表面z=2.5 mm", "有"),
     ("3.3.2 T 325.2→630.1；轴心偏差-0.13→-0.07、68%半宽0.51→0.33 mm", "有"),
     ("3.3.3 浅带泄漏减少：真值诊断-0.48±2.00%，与零相容；非分辨率标定", "慎"),
     ("3.3.4 每真实湮没选出率比0.860；相同选出计数时间约×1.16", "有"),
     ("3.3.5 横向刻度k 0.795→0.920；4/3点、共享参照，独立验证尚缺", "有"),
     ("3.3.6 扫到中心或标定校正两种模式；已测离轴偏差+0.84/+1.21 mm", "有"),
     ("3.3.7 深带有效视场±34.5→±31.5 mm；六条主判据通过、三条预言未过", "有")]),
   ("3.4  方法边界与已测厚度档位", [
     ("3.4.1 当前图为闭式点云直方图与参照比值；未准确恢复空鼓深度和厚度", "有"),
     ("3.4.2 M3深带多次散射约48.6%；按次数拆分仅用于真值诊断", "慎"),
     ("3.4.3 简化事例级预测器有六类失效；正式阈值与判定取GATE结果", "有"),
     ("3.4.4 F3六档：最薄检出通过10 mm、定位通过20 mm；深40 mm界面条件", "有"),
     ("3.4.5 理想源与配对假设；实验、时间响应、偶然符合和剂量仍缺", "慎")]),
 ]),
 ("4  结论", [
   ("第一段：方法的有效性", [
     ("4.1.1 已测M3边界定位：轴心偏差-0.07、离轴-16 mm偏差+1.21 mm", "有"),
     ("4.1.2 本次无缺陷负对照未超阈值；M3偏尾，不等于现场零误报", "有"),
     ("4.1.3 F3已扫描档位：10 mm检出、20 mm定位；5 mm及以下未检出", "有")]),
   ("第二段：构型优化的可行性", [
     ("4.2.1 深度残差68%半宽缩小27.6%，横向刻度更接近1；多参数共同改变", "有"),
     ("4.2.2 D1晶体12.37→2.05 kg，减83.4%；选出率和深带视场有所牺牲", "有")]),
   ("第三段：未来研究方向", [
     ("4.3.1 实验验证、真实材料和辐射剂量评估", "缺"),
     ("4.3.2 时间分辨、偶然符合及可用于实验的符合配对", "缺"),
     ("4.3.3 厚度与深度独立变化及重复试验；是否开展待决定", "缺")]),
 ]),
]

X0, X1, X2 = 0.0, 3.2, 7.0
W0, W1, W2 = 2.9, 3.6, 8.6
SLOT = 1.0


def box_height(text, wrap):
    return 0.34 + 0.30 * len(textwrap.wrap(text, wrap) or [text])


def box(ax, x, y, w, text, color, fs, bold=False, wrap=28):
    lines = textwrap.wrap(text, wrap) or [text]
    t = "\n".join(lines)
    h = 0.34 + 0.30 * len(lines)
    ax.add_patch(FancyBboxPatch((x, y - h / 2), w, h,
                                boxstyle="round,pad=0.06,rounding_size=0.12",
                                fc="white", ec=color,
                                lw=1.6 if bold else 1.0, zorder=3))
    ax.annotate(t, (x + w / 2, y), fontsize=fs, color=color, ha="center", va="center",
                zorder=4, weight=("bold" if bold else "normal"), linespacing=1.35)
    return h


def draw(ax, title, groups):
    ax.set_facecolor(SURFACE)
    slots, y = [], 0.0
    for name, leaves in groups:
        ys = []
        for txt, _ in leaves:
            h = box_height(txt, 40)
            ys.append(y - h / 2)
            y -= h + 0.18
        y -= 0.45
        slots.append(ys)
    ally = [v for ys in slots for v in ys]
    rooty = float(np.mean(ally))
    for (name, leaves), ys in zip(groups, slots):
        my = float(np.mean(ys))
        worst = MISS if any(s == "缺" for _, s in leaves) else (
            CARE if any(s == "慎" for _, s in leaves) else HAVE)
        box(ax, X1, my, W1, name, worst, 8.4, bold=True, wrap=20)
        ax.plot([X0 + W0, X1], [rooty, my], color="#333", lw=1.1, zorder=2)
        for (txt, st), yy in zip(leaves, ys):
            box(ax, X2, yy, W2, txt, C[st], 8.0, wrap=40)
            ax.plot([X1 + W1, X2], [my, yy], color="#555", lw=0.8, zorder=2)
    box(ax, X0, rooty, W0, title, HAVE, 10.5, bold=True, wrap=12)
    ax.set_xlim(-0.4, X2 + W2 + 0.4)
    ax.set_ylim(y - 0.1, 0.55)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("auto")
    for sp in ax.spines.values():
        sp.set_visible(False)


pdf = PdfPages(os.path.join(_args.out, "framework_sim_paper.pdf"))
for i, (title, groups) in enumerate(CHAPTERS, 1):
    nleaf = sum(len(g[1]) for g in groups)
    fig = plt.figure(figsize=(13.6, max(4.2, 0.62 * nleaf + 1.6)), facecolor=SURFACE)
    ax = fig.add_axes([0.01, 0.045, 0.98, 0.90])
    draw(ax, title, groups)
    fig.text(0.012, 0.012,
             "黑 = 现有证据，须按节点条件引用；红 = 缺；橙 = 历史测试、真值诊断或有明确限制。\n"
             f"D2 15%正式口径，更新2026-09-30，第{i}/4页；逐项解释见《仿真论文框架逐项说明_20260930.md》。",
             fontsize=7.6, color="#8a8a85", ha="left", va="bottom")
    fig.savefig(os.path.join(_args.out, f"framework_sim_p{i}.png"), dpi=190,
                facecolor=SURFACE)
    pdf.savefig(fig, facecolor=SURFACE)
    plt.close(fig)
pdf.close()
print(f"写出 {os.path.join(_args.out, 'framework_sim_paper.pdf')} 与 framework_sim_p1..p4.png")
