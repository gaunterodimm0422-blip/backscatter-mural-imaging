# -*- coding: utf-8 -*-
"""成像效果演示：把重建点云画成图，并把"为什么必须做比值"讲清楚。

四张图：
  (a) 参照场（无空鼓）的重建点密度 (x, 深度)
  (b) 空鼓场的重建点密度 —— **和 (a) 几乎一模一样，肉眼看不出空鼓**
  (c) 比值 = (b)/(a)，空鼓才显出来；叠上真值空鼓轮廓与拟合出的边界
  (d) 沿 x 的一维剖面，按四个重建深度带分开

这张图本身就是本方法的诚实写照：**单看"像"看不出缺陷，信息在比值里，
结论来自统计检验；而且深度方向是糊的，横向才是可信的那一维。**

同时把两张二维直方图存成 imaging_demo_data.npz（几百 KB），方便拿回主力机重画。

用法（远控机）
--------------
  cd D:\\beisanshe
  <python> scripts\\make_imaging_demo.py ^
      --ref  runs\\m3_ref_t20_L-15_4m_seed24091302\\canonical_response ^
      --void runs\\m3_halfvoid_t20_L-15_4m_seed24091301\\canonical_response ^
      --config configs\\m3_edge_main_20260912.json ^
      --x-edge-truth -15.0 --x-edge-fit -15.07 ^
      --label "M3 新构型" --out outputs\\imaging_demo
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Rectangle
from matplotlib import font_manager

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

for fam in ("Noto Sans CJK SC", "Microsoft YaHei", "SimHei", "Noto Sans CJK JP"):
    if any(fam in f.name for f in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = fam
        break
plt.rcParams["axes.unicode_minus"] = False

SURFACE = "#fcfcfb"
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#9a9891", "#e3e1dc"
TRUTH, FIT = "#1a7f4b", "#c0392b"
# 顺序型：单一色相由浅到深
SEQ = LinearSegmentedColormap.from_list("seq", ["#f7f6f3", "#bcd4ef", "#5b93d4", "#1b4f8f"])
# 发散型：冷 ← 中性灰 → 暖（中点是灰，不是白，也不是任何色相）
DIV = LinearSegmentedColormap.from_list("div", ["#1b4f8f", "#5b93d4", "#e9e7e2",
                                                "#eb9a6c", "#b3441c"])


def load(ad: Path, config):
    from backscatter.pose_selection import build_selection_pose_aware
    ad = Path(ad)
    pairs = dict(np.load(ad / "pairs.npz", allow_pickle=True))
    recon = dict(np.load(ad / "reconstruction.npz", allow_pickle=True))
    selected, _c, _d = build_selection_pose_aware(pairs, config)
    keep = selected & np.asarray(recon["valid"], dtype=bool)
    yhw = float(config["observables"]["y_half_width_mm"])
    sy = np.asarray(pairs["source_y_mm"], float)
    if np.isfinite(yhw):
        keep &= np.abs(np.asarray(recon["y_mm"], float) - sy) <= yhw
    sx = np.asarray(pairs["source_x_mm"], float)
    n_primary = None
    for p in (ad / "summary.json", ad.parent / "summary.json"):
        if p.exists():
            n_primary = (json.loads(p.read_text(encoding="utf-8")).get("gate_metadata")
                         or {}).get("primary_event_count")
            if n_primary:
                break
    return (np.asarray(recon["x_mm"], float)[keep] - sx[keep],
            -np.asarray(recon["z_mm"], float)[keep],
            int(n_primary) if n_primary else None, int(keep.sum()),
            float(np.median(sx[keep])))


def draw(d, args):
    xe = np.asarray(d["xe"], float); de = np.asarray(d["de"], float)
    Hr = np.asarray(d["Hr"], float); Hv = np.asarray(d["Hv"], float)
    bands_ = [tuple(b) for b in np.asarray(d["bands"])]
    if "hb_r" in d and "hb_v" in d:
        hb_r = np.asarray(d["hb_r"], float); hb_v = np.asarray(d["hb_v"], float)
        band_note = ""
    else:
        # 老版本存的 npz 没有一维剖面，只能用 de 的深度箱近似分组。
        # 箱边切不出 8 mm 这种边界时，分组与冻结深度带不完全一致——图上必须写明。
        dc_ = 0.5 * (de[1:] + de[:-1])
        hb_r, hb_v = [], []
        for lo, hi in bands_:
            m = (dc_ >= lo) & (dc_ < hi)
            hb_r.append(Hr[:, m].sum(1)); hb_v.append(Hv[:, m].sum(1))
        hb_r = np.asarray(hb_r, float); hb_v = np.asarray(hb_v, float)
        band_note = ("（(d) 的深度分组由 %g mm 深度箱近似，与冻结深度带不完全一致）"
                     % (de[1] - de[0]))
        print("警告：npz 里没有 hb_r/hb_v，(d) 面板改用深度箱近似分组。" + band_note)
    npr = int(d["n_primary_ref"]); npv = int(d["n_primary_void"])
    nsr = int(d["n_sel_ref"]); nsv = int(d["n_sel_void"])
    headx = float(d["head_x_mm"]); bands = [tuple(b) for b in np.asarray(d["bands"])]
    scale = (npr / npv) if (npr and npv) else 1.0
    with np.errstate(invalid="ignore", divide="ignore"):
        R = np.where(Hr >= args.min_ref, Hv * scale / Hr, np.nan)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    ute = (args.x_edge_truth - headx) if args.x_edge_truth is not None else None
    ufe = (args.x_edge_fit - headx) if args.x_edge_fit is not None else None
    lo_v, hi_v = args.void_depth
    ttl = f"（{args.label}）" if args.label else ""

    fig = plt.figure(figsize=(15.0, 8.6), facecolor=SURFACE)
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 0.92], hspace=0.36, wspace=0.26,
                          left=0.05, right=0.975, top=0.875, bottom=0.175)
    ext = [xe[0], xe[-1], de[-1], de[0]]
    vmax = float(np.nanpercentile(np.concatenate([Hr.ravel(), Hv.ravel()]), 99.5))

    for j, (H, name) in enumerate(((Hr, "(a) 参照场（无空鼓）"), (Hv, "(b) 空鼓场"))):
        ax = fig.add_subplot(gs[0, j])
        im = ax.imshow(H.T, origin="upper", extent=ext, aspect="auto", cmap=SEQ,
                       vmin=0, vmax=vmax, interpolation="nearest")
        ax.set_title(f"{name}  重建点密度", fontsize=11, color=INK, loc="left", pad=6)
        ax.set_xlabel("横向 x（探测头坐标，mm）", fontsize=9, color=INK2)
        ax.set_ylabel("重建深度（mm）", fontsize=9, color=INK2)
        ax.tick_params(colors=INK2, labelsize=8)
        cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
        cb.set_label("每箱事例数", fontsize=8, color=INK2)
        cb.ax.tick_params(colors=INK2, labelsize=7.5)
        if ute is not None:
            ax.add_patch(Rectangle((ute, lo_v), xe[-1] - ute, hi_v - lo_v,
                                   fc="none", ec=TRUTH, lw=1.4, ls=(0, (5, 3)), zorder=5))

    ax = fig.add_subplot(gs[0, 2])
    im = ax.imshow(R.T, origin="upper", extent=ext, aspect="auto", cmap=DIV,
                   norm=Normalize(vmin=0.6, vmax=1.4), interpolation="nearest")
    ax.set_title("(c) 比值 = 空鼓场 / 参照场", fontsize=11, color=INK, loc="left", pad=6)
    ax.set_xlabel("横向 x（探测头坐标，mm）", fontsize=9, color=INK2)
    ax.set_ylabel("重建深度（mm）", fontsize=9, color=INK2)
    ax.tick_params(colors=INK2, labelsize=8)
    cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
    cb.set_label("比值（1.0 = 无差别）", fontsize=8, color=INK2)
    cb.ax.tick_params(colors=INK2, labelsize=7.5)
    if ute is not None:
        ax.add_patch(Rectangle((ute, lo_v), xe[-1] - ute, hi_v - lo_v,
                               fc="none", ec=TRUTH, lw=1.6, ls=(0, (5, 3)), zorder=5))
        ax.annotate("真值空鼓所在", (ute + 1.5, lo_v - 1.5), fontsize=8.4, color=TRUTH,
                    ha="left", va="bottom", zorder=6)
    if ufe is not None:
        ax.axvline(ufe, color=FIT, lw=1.8, zorder=6)
        ax.annotate(f"拟合边界 {args.x_edge_fit:+.2f} mm", (ufe - 1.5, de[0] + 2.5),
                    fontsize=8.4, color=FIT, ha="right", va="top", zorder=6)

    axp = fig.add_subplot(gs[1, :])
    cols = ["#1b4f8f", "#5b93d4", "#eb9a6c", "#b3441c"]
    x1 = 0.5 * (xe[1:] + xe[:-1])
    for (lo, hi), col, hr, hv in zip(bands, cols, hb_r, hb_v):
        g = hr >= args.min_ref
        r = hv[g] * scale / hr[g]
        er = r * np.sqrt(1.0 / np.maximum(hv[g], 1) + 1.0 / hr[g])
        axp.errorbar(x1[g], r, yerr=er, fmt="o-", ms=4.5, lw=1.6, color=col,
                     capsize=0, label=f"重建深度 {lo}–{hi} mm", zorder=4)
    axp.axhline(1.0, color=MUTED, lw=1.0, ls=(0, (4, 3)), zorder=2)
    y0, y1 = axp.get_ylim()
    axp.set_ylim(y0 - 0.02, y1 + 0.16 * (y1 - y0))
    y0, y1 = axp.get_ylim()
    if ute is not None:
        axp.axvline(ute, color=TRUTH, lw=1.6, ls=(0, (5, 3)), zorder=3)
        axp.annotate("真值边界", (ute - 1.2, y1 - 0.03 * (y1 - y0)), fontsize=8.6,
                     color=TRUTH, ha="right", va="top", zorder=5)
    if ufe is not None:
        axp.axvline(ufe, color=FIT, lw=1.8, zorder=3)
        axp.annotate("拟合边界", (ufe + 1.2, y1 - 0.11 * (y1 - y0)), fontsize=8.6,
                     color=FIT, ha="left", va="top", zorder=5)
    axp.set_title("(d) 沿横向的一维剖面：空鼓场 / 参照场（误差棒为泊松统计）",
                  fontsize=11, color=INK, loc="left", pad=6)
    axp.set_xlabel("横向 x（探测头坐标，mm）", fontsize=9.5, color=INK2)
    axp.set_ylabel("比值", fontsize=9.5, color=INK2)
    axp.set_xlim(xe[0], xe[-1])
    axp.tick_params(colors=INK2, labelsize=8.5)
    axp.grid(True, color=GRID, lw=0.7, zorder=0)
    axp.set_axisbelow(True)
    for sp in ("top", "right"):
        axp.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        axp.spines[sp].set_color(MUTED); axp.spines[sp].set_linewidth(0.8)
    leg = axp.legend(loc="upper right", fontsize=8.6, frameon=True, framealpha=0.95,
                     edgecolor=GRID, ncol=2)
    for t in leg.get_texts():
        t.set_color(INK2)

    fig.suptitle(f"成像效果演示{ttl}：单看“像”看不出空鼓，信息在比值里",
                 fontsize=14.5, color=INK, x=0.006, ha="left", y=0.975)
    fig.text(0.006, 0.012,
             f"(a)(b) 用同一色标：两张图肉眼几乎无差别 —— 空鼓造成的计数亏空只有百分之几到三成，"
             f"淹没在点密度的整体形状里。(c) 做了比值才显出来。\n"
             f"深度方向是糊的（本方法的深度分辨约 10 mm 半宽），所以亏空在纵向铺开；"
             f"横向边界才是可信的那一维 —— 这正是本方法只报横向边界、不报深度的原因。\n"
             f"{band_note}"
             f"参照场 {nsr:,} 个选出事例（{npr:,} 初级粒子），空鼓场 {nsv:,}（{npv:,}）；"
             f"比值已按初级粒子数归一，每箱参照计数 < {args.min_ref} 的不画。"
             f"绿色虚线框 = 真值空鼓；红线 = 多深度带共享边界拟合给出的边界。",
             fontsize=8.2, color=MUTED, ha="left", va="bottom", linespacing=1.6)

    for ext_ in ("png", "pdf"):
        fig.savefig(out / f"imaging_demo.{ext_}", dpi=190, facecolor=SURFACE)
    print(f"写出 {out/'imaging_demo.png'} / .pdf")
    if not getattr(args, "from_npz", None):
        print(f"二维直方图已存 {out/'imaging_demo_data.npz'}（可拷回主力机重画）")
    print(f"参照场 选出 {nsr:,} / 初级 {npr:,}    空鼓场 选出 {nsv:,} / 初级 {npv:,}")




def main():
    ap = argparse.ArgumentParser(description="成像效果演示图")
    ap.add_argument("--ref", required=True)
    ap.add_argument("--void", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--label", default="")
    ap.add_argument("--x-edge-truth", type=float, default=None, help="真值边界的世界坐标 mm")
    ap.add_argument("--x-edge-fit", type=float, default=None, help="拟合出的边界世界坐标 mm")
    ap.add_argument("--void-depth", nargs=2, type=float, default=[20.0, 40.0])
    ap.add_argument("--xlim", nargs=2, type=float, default=[-36.0, 36.0])
    ap.add_argument("--dlim", nargs=2, type=float, default=[0.0, 60.0])
    ap.add_argument("--dx", type=float, default=3.0)
    ap.add_argument("--dd", type=float, default=5.0)
    ap.add_argument("--min-ref", type=int, default=60)
    ap.add_argument("--out", required=True)
    ap.add_argument("--from-npz", default=None,
                    help="直接从已存的 imaging_demo_data.npz 重画，不需要 backscatter 包")
    args = ap.parse_args()

    if args.from_npz:
        d = dict(np.load(args.from_npz, allow_pickle=True))
        draw(d, args)
        return

    from backscatter.config import load_config
    config = load_config(Path(args.config))
    xr, dr, npr, nsr, headx = load(Path(args.ref), config)
    xv, dv, npv, nsv, _ = load(Path(args.void), config)
    bands = [tuple(b) for b in config["observables"]["depth_bands_mm"]]

    xe = np.arange(args.xlim[0], args.xlim[1] + 1e-9, args.dx)
    de = np.arange(args.dlim[0], args.dlim[1] + 1e-9, args.dd)
    xc = 0.5 * (xe[1:] + xe[:-1]); dc = 0.5 * (de[1:] + de[:-1])
    Hr, _, _ = np.histogram2d(xr, dr, bins=[xe, de])
    Hv, _, _ = np.histogram2d(xv, dv, bins=[xe, de])
    scale = (npr / npv) if (npr and npv) else 1.0
    with np.errstate(invalid="ignore", divide="ignore"):
        R = np.where(Hr >= args.min_ref, Hv * scale / Hr, np.nan)

    hb_r, hb_v = [], []
    for lo, hi in bands:
        mr = (dr >= lo) & (dr < hi); mv = (dv >= lo) & (dv < hi)
        hb_r.append(np.histogram(xr[mr], bins=xe)[0])
        hb_v.append(np.histogram(xv[mv], bins=xe)[0])
    hb_r = np.asarray(hb_r); hb_v = np.asarray(hb_v)

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "imaging_demo_data.npz", xe=xe, de=de, Hr=Hr, Hv=Hv,
                        hb_r=hb_r, hb_v=hb_v,
                        n_primary_ref=npr or 0, n_primary_void=npv or 0,
                        n_sel_ref=nsr, n_sel_void=nsv, head_x_mm=headx,
                        x_edge_truth=(args.x_edge_truth if args.x_edge_truth is not None else np.nan),
                        x_edge_fit=(args.x_edge_fit if args.x_edge_fit is not None else np.nan),
                        void_depth=np.asarray(args.void_depth), bands=np.asarray(bands),
                        label=args.label)

    draw(dict(xe=xe, de=de, Hr=Hr, Hv=Hv, hb_r=hb_r, hb_v=hb_v,
              n_primary_ref=npr or 0, n_primary_void=npv or 0,
              n_sel_ref=nsr, n_sel_void=nsv, head_x_mm=headx,
              bands=np.asarray(bands)), args)


if __name__ == "__main__":
    main()
