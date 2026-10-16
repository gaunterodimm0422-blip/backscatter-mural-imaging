# -*- coding: utf-8 -*-
"""诊断第二步：把对比度按"单次散射 / 多次散射"拆开，看浅带塌陷的另一半是哪来的。

第一步（diag_shallow_band.py）已经确认：
  * 杠杆臂 |d| 从 50.72 缩到 39.53 mm，深度误差半宽从 15.10 缩到 10.93 mm —— 机制成立
  * 0-8 带里"真实深度落在空鼓层"的份额从 10.4% 降到 6.4%（比 0.61）
  * 但实测 c_b 从 12.21% 掉到 2.40%（比 0.20）—— **泄漏只解释了大约一半**

剩下那一半的候选：**多次散射**。在幻影里散射两次以上的光子，重建顶点可以落在任何
地方，它携带的空鼓信息与"重建深度"脱钩。pairs.npz 里的 `d1_n_phantom_compton`
正好是"这个光子在幻影里康普顿散射了几次"，可以直接把两群分开。

观测量用**双比值**，与总流强和归一化无关：

    c = 1 − [N_void(右)/N_ref(右)] / [N_void(左)/N_ref(左)]

右 = 空鼓那一侧（探测头坐标 x > 0），左 = 完好侧。所有带、所有子群用**同一个**
横向窗口，保证可比。

**使用真值列只用于机制判定，任何对外报告的性能数字都不得出自这里。**

用法（远控机）
--------------
  <python> scripts\\diag_multiscatter.py ^
      --ref runs\\<REF_RUN>\\canonical_response ^
      --void runs\\<HALFVOID_RUN>\\canonical_response ^
      --config configs\\<配置>.json --label M3
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


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
    out = dict(
        x_head=np.asarray(recon["x_mm"], float)[keep] - sx[keep],
        d_rec=-np.asarray(recon["z_mm"], float)[keep],
        n_compt=np.asarray(pairs["d1_n_phantom_compton"], float)[keep],
        n_ray=np.asarray(pairs["d1_n_phantom_rayleigh"], float)[keep],
        matched=(np.asarray(recon["truth_matched"], bool)[keep]
                 if "truth_matched" in recon else None),
        d_true=(-np.asarray(recon["truth_z_mm"], float)[keep]
                if "truth_z_mm" in recon else None),
        n=int(keep.sum()))
    return out


def groups(n_compt):
    return [("单次 n=1", n_compt == 1),
            ("多次 n>=2", n_compt >= 2),
            ("n=0 或缺", ~((n_compt == 1) | (n_compt >= 2))),
            ("全部", np.ones_like(n_compt, dtype=bool))]


def main():
    ap = argparse.ArgumentParser(description="按散射次数拆解对比度")
    ap.add_argument("--ref", required=True)
    ap.add_argument("--void", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--label", default="")
    ap.add_argument("--xwin", type=float, default=16.5,
                    help="所有带共用的横向半窗口 mm（默认 16.5，= 0-8 带最窄的那个）")
    ap.add_argument("--void-depth", nargs=2, type=float, default=[20.0, 40.0])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from backscatter.config import load_config
    config = load_config(Path(args.config))
    R = load(Path(args.ref), config)
    V = load(Path(args.void), config)
    bands = [tuple(b) for b in config["observables"]["depth_bands_mm"]]
    W = float(args.xwin)
    lo_v, hi_v = args.void_depth

    print("=" * 92)
    print(f"对比度按散射次数拆解{('  [' + args.label + ']') if args.label else ''}"
          "   **仅用于机制判定，不得用作性能指标**")
    print("=" * 92)
    print(f"参照场 : {args.ref}   筛选后 {R['n']:,}")
    print(f"空鼓场 : {args.void}   筛选后 {V['n']:,}")
    print(f"横向窗口：所有带统一用 |x_head| <= {W:.1f} mm；右 = 空鼓侧(x>0)，左 = 完好侧(x<0)")
    print(f"对比度 c = 1 − [Nv(右)/Nr(右)] / [Nv(左)/Nr(左)]，双比值，与归一化无关\n")

    rows = []
    for lo, hi in bands:
        mR = (R["d_rec"] >= lo) & (R["d_rec"] < hi) & (np.abs(R["x_head"]) <= W)
        mV = (V["d_rec"] >= lo) & (V["d_rec"] < hi) & (np.abs(V["x_head"]) <= W)
        nR_band, nV_band = int(mR.sum()), int(mV.sum())
        print(f"--- 重建深度带 {lo}-{hi} mm   参照 {nR_band:,}   空鼓 {nV_band:,} ---")
        print(f"{'子群':>12}{'参照N':>9}{'份额':>8}{'空鼓N':>9}"
              f"{'左比值':>9}{'右比值':>9}{'c_b%':>9}{'该群真深空鼓层%':>16}")
        for name, gR in groups(R["n_compt"]):
            gV = groups(V["n_compt"])[[g[0] for g in groups(R["n_compt"])].index(name)][1]
            a, b = mR & gR, mV & gV
            nr, nv = int(a.sum()), int(b.sum())
            if nr < 50:
                print(f"{name:>12}{nr:9d}{'—':>8}{nv:9d}{'—':>9}{'—':>9}{'—':>9}{'—':>16}")
                rows.append(dict(band=[lo, hi], group=name, n_ref=nr, n_void=nv))
                continue
            nrL = int((a & (R["x_head"] < 0)).sum()); nrR = int((a & (R["x_head"] > 0)).sum())
            nvL = int((b & (V["x_head"] < 0)).sum()); nvR = int((b & (V["x_head"] > 0)).sum())
            rl = nvL / nrL if nrL else float("nan")
            rr = nvR / nrR if nrR else float("nan")
            c = 1 - rr / rl if rl else float("nan")
            fv = float("nan")
            if R["d_true"] is not None and R["matched"] is not None:
                mm = a & R["matched"] & np.isfinite(R["d_true"])
                if mm.sum() > 20:
                    dt = R["d_true"][mm]
                    # 按全体归一（未匹配的一律算作"不在空鼓层"，因为真值只记到 40 mm）
                    fv = 100.0 * float(((dt >= lo_v) & (dt <= hi_v)).sum()) / nr
            print(f"{name:>12}{nr:9d}{100*nr/max(nR_band,1):7.1f}%{nv:9d}"
                  f"{rl:9.4f}{rr:9.4f}{100*c:9.2f}{fv:15.1f}%")
            rows.append(dict(band=[lo, hi], group=name, n_ref=nr, n_void=nv,
                             ratio_left=rl, ratio_right=rr, c_b_pct=100*c,
                             f_void_pct_of_all=fv))
        print()

    print("判读：")
    print("  * 若『多次 n>=2』那一群的 c_b 明显高于『单次 n=1』，说明浅带原来的对比度")
    print("    主要是多次散射贡献的——它把深处的空鼓信息塞进了浅带。")
    print("  * 再比 M2 与 M3 的『多次』份额：份额掉下去多少，对比度就掉下去多少。")
    print("  * 『该群真深空鼓层%』按该群**全体**归一（真值只记到 40 mm，未匹配的算作不在层内），")
    print("    所以它是下限，不是精确值。")

    if args.out:
        out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
        json.dump(dict(label=args.label, ref=str(args.ref), void=str(args.void),
                       xwin=W, rows=rows),
                  open(out / "diag_multiscatter.json", "w", encoding="utf-8"),
                  ensure_ascii=False, indent=1)
        print(f"\n写入 {out / 'diag_multiscatter.json'}")


if __name__ == "__main__":
    main()
