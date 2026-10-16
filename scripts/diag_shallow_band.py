# -*- coding: utf-8 -*-
"""诊断：0-8 mm 重建深度带的对比度为什么从 M2 的 12.21% 塌到 M3 的 2.40%。

要检验的假说
------------
空鼓占真实深度 20~40 mm。**真正在 0~8 mm 处散射的事例，其入射腿和出射腿都不经过
空鼓层，所以它们根本不携带空鼓信息，对比度应当为零。**
那么"0-8 mm 重建带"里原本那 12.21% 的对比度只可能来自一件事：深度重建不准，
真实深度落在空鼓层里的事例被错误地重建到了浅带（泄漏）。

于是有一个可直接检验的预言：

    c_b(某重建带) ≈ f_void(该带) − k · f_deep(该带)

    f_void = 该带事例中"真实深度落在 20~40 mm"的份额（空鼓把它们整个抽掉 -> 正对比度）
    f_deep = 该带事例中"真实深度 > 40 mm"的份额（空鼓让入射/出射腿少衰减 -> 计数反而增加）
    真实深度 < 20 mm 的事例既不被抽掉也不被增强，纯粹稀释对比度

**如果 M3 的 0-8 带 f_void 比 M2 小得多，假说成立**：新构型 D1 压到离物体 2.5 mm，
杠杆臂 |d| = |H − S| 变短（dt/dθ = |d|·sinα/sin²θ），深度重建变准，泄漏减少，
同时浅层散射的接受度大增 —— 浅带被"真正的浅层事例"填满，而它们不带空鼓信息。

**本脚本使用真值（PhaseSpaceActor 康普顿顶点），只用于机制判定，
任何对外报告的性能数字都不得出自这里。**

应当跑在**参照场**（ref，空鼓挪到视场外、材料均匀）上：在均匀材料里统计
"哪个重建带的事例真实来自哪个深度"，才是干净的泄漏测量。

用法（远控机）
--------------
  <python> scripts\\diag_shallow_band.py ^
      --analysis-dir runs\\<RUN_ID>\\canonical_response ^
      --config configs\\<配置>.json ^
      --label M3-ref
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

TKEYS = ("truth_x_mm", "truth_y_mm", "truth_z_mm")


def attach_truth(ad: Path, pairs: dict, recon: dict):
    """真值顶点在 reconstruction.npz 里；canonical_response 视图不一定带，按 key 对齐去找。"""
    if all(k in recon for k in TKEYS):
        return {k: np.asarray(recon[k], float) for k in TKEYS}, "本目录 reconstruction.npz"
    k_here = np.asarray(recon["key"]) if "key" in recon else (
        np.asarray(pairs["key"]) if "key" in pairs else None)
    if k_here is None:
        raise SystemExit("npz 里没有 key 列，无法对齐真值。")
    for c in (ad / "pairs.npz", ad.parent / "reconstruction.npz", ad.parent / "pairs.npz"):
        if not c.exists():
            continue
        src = dict(np.load(c, allow_pickle=True))
        if not all(k in src for k in TKEYS) or "key" not in src:
            continue
        k_src = np.asarray(src["key"])
        order = np.argsort(k_src)
        pos = np.clip(np.searchsorted(k_src[order], k_here), 0, len(k_src) - 1)
        idx = order[pos]
        hit = k_src[idx] == k_here
        if hit.sum() < 0.99 * len(k_here):
            print(f"  （{c} 按 key 只匹配上 {int(hit.sum()):,}/{len(k_here):,}，跳过）")
            continue
        out = {}
        for k in TKEYS:
            v = np.full(len(k_here), np.nan, dtype=np.float64)
            v[hit] = np.asarray(src[k], dtype=np.float64)[idx[hit]]
            out[k] = v
        return out, f"{c.parent.name}/{c.name}（按 key 对齐 {int(hit.sum()):,}/{len(k_here):,}）"
    raise SystemExit("找不到 truth_x_mm/truth_y_mm/truth_z_mm。这一档 analyze 可能没带 --truth。")


def pick(d: dict, *cands):
    for c in cands:
        if c in d:
            return np.asarray(d[c], float), c
    return None, None


def main():
    ap = argparse.ArgumentParser(description="浅带对比度塌陷的真值诊断")
    ap.add_argument("--analysis-dir", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--label", default="")
    ap.add_argument("--void-depth", nargs=2, type=float, default=[20.0, 40.0])
    ap.add_argument("--out", default=None)
    ap.add_argument("--dump-keys", action="store_true", help="打印 npz 里所有列名")
    args = ap.parse_args()

    from backscatter.config import load_config
    from backscatter.pose_selection import build_selection_pose_aware

    config = load_config(Path(args.config))
    ad = Path(args.analysis_dir)
    pairs = dict(np.load(ad / "pairs.npz", allow_pickle=True))
    recon = dict(np.load(ad / "reconstruction.npz", allow_pickle=True))

    print("=" * 78)
    print(f"浅带对比度诊断{('  [' + args.label + ']') if args.label else ''}"
          "  **使用真值，仅用于机制判定，不得用作性能指标**")
    print("=" * 78)
    print(f"分析目录 : {ad}")
    if args.dump_keys:
        print(f"pairs.npz 列          : {sorted(pairs.keys())}")
        print(f"reconstruction.npz 列 : {sorted(recon.keys())}")

    truth, tsrc = attach_truth(ad, pairs, recon)
    for k in TKEYS:
        pairs[k] = truth[k]
    print(f"真值来源 : {tsrc}")

    # ---- 与 analyze_mural_edge.py 逐字相同的筛选，保证事例数能对上 ----
    selected, _c, _d = build_selection_pose_aware(pairs, config)
    keep = selected & np.asarray(recon["valid"], dtype=bool)
    yhw = float(config["observables"]["y_half_width_mm"])
    if np.isfinite(yhw):
        keep &= np.abs(np.asarray(recon["y_mm"], float)
                       - np.asarray(pairs["source_y_mm"], float)) <= yhw
    keep &= np.isfinite(pairs["truth_z_mm"])
    n = int(keep.sum())
    if n < 100:
        raise SystemExit(f"可用事例只有 {n} 个，太少。")

    sx = np.asarray(pairs["source_x_mm"], float)[keep]
    sz = np.asarray(pairs["source_z_mm"], float)[keep]
    d_rec = -np.asarray(recon["z_mm"], float)[keep]
    d_true = -np.asarray(pairs["truth_z_mm"], float)[keep]
    x_head = np.asarray(recon["x_mm"], float)[keep] - sx
    print(f"可用事例 : {n:,}（筛选与 analyze_mural_edge 一致；"
          f"**请核对下表 N 与那边 ref 列逐带相等**）")

    # ---- 深度重建质量 ----
    err = d_rec - d_true
    print(f"\n[深度重建质量] d_rec − d_true："
          f"中位 {np.median(err):+.2f} mm，"
          f"16~84 分位 [{np.percentile(err,16):+.1f}, {np.percentile(err,84):+.1f}] mm，"
          f"|误差| 中位 {np.median(np.abs(err)):.2f} mm，"
          f"半宽(84−16)/2 = {(np.percentile(err,84)-np.percentile(err,16))/2:.2f} mm")

    # ---- 杠杆臂 |d| = |H − S| ----
    hx, kx = pick(pairs, "d1_x_mm", "bs_x_mm", "backscatter_x_mm", "ideal_d1_x_mm")
    hy, _ = pick(pairs, "d1_y_mm", "bs_y_mm", "backscatter_y_mm", "ideal_d1_y_mm")
    hz, _ = pick(pairs, "d1_z_mm", "bs_z_mm", "backscatter_z_mm", "ideal_d1_z_mm")
    if hx is not None and hy is not None and hz is not None:
        sy = np.asarray(pairs["source_y_mm"], float)[keep]
        nd = np.sqrt((hx[keep]-sx)**2 + (hy[keep]-sy)**2 + (hz[keep]-sz)**2)
        print(f"[杠杆臂] |d| = |H − S|（列 {kx}）："
              f"中位 {np.median(nd):.2f} mm，"
              f"16~84 分位 [{np.percentile(nd,16):.1f}, {np.percentile(nd,84):.1f}] mm")
        print(f"         dt/dθ = |d|·sinα/sin²θ —— |d| 越短深度越准，这是机制的根")
    else:
        print("[杠杆臂] 没找到 D1 命中点的列，跳过（加 --dump-keys 看列名）")

    # ---- 关键表 ----
    lo_v, hi_v = args.void_depth
    bands = [tuple(b) for b in config["observables"]["depth_bands_mm"]]
    rows = []
    print(f"\n[关键表] 每个**重建**深度带里，事例的**真实**深度来自哪里"
          f"（空鼓层 = {lo_v:.0f}~{hi_v:.0f} mm）")
    print(f"{'重建带mm':>10}{'N':>8}{'真深<%.0f':>11}{'真深%.0f~%.0f':>13}{'真深>%.0f':>10}"
          % (lo_v, lo_v, hi_v, hi_v)
          + f"{'真深中位':>10}{'预言c_b%':>11}")
    print(f"{'':>10}{'':>8}{'(无信息)':>11}{'(被抽掉,正)':>13}{'(透射增强,负)':>10}{'':>10}{'':>11}")
    for lo, hi in bands:
        m = (d_rec >= lo) & (d_rec < hi)
        N = int(m.sum())
        if N == 0:
            rows.append(dict(band=[lo, hi], n=0)); continue
        dt = d_true[m]
        f_above = float(np.mean(dt < lo_v))
        f_void = float(np.mean((dt >= lo_v) & (dt <= hi_v)))
        f_deep = float(np.mean(dt > hi_v))
        print(f"{lo:4d}-{hi:<5d}{N:8d}{100*f_above:10.1f}%{100*f_void:12.1f}%"
              f"{100*f_deep:9.1f}%{np.median(dt):10.1f}{100*(f_void-f_deep):10.1f}%")
        rows.append(dict(band=[lo, hi], n=N, f_above=f_above, f_void=f_void,
                         f_deep=f_deep, d_true_median=float(np.median(dt)),
                         c_b_pred_pct=100*(f_void-f_deep)))
    print("  预言 c_b ≈ f(空鼓层) − f(更深)。这是一阶近似：只算'散射体被抽掉'与"
          "'腿上少衰减'两项，\n  不含多次散射，所以只用来比 M2/M3 的**相对变化**，不当绝对值。")

    # ---- 泄漏矩阵 ----
    tb = [(0, 5), (5, 10), (10, 20), (20, 30), (30, 40), (40, 60), (60, 140)]
    print(f"\n[泄漏矩阵] 行 = 真实深度，列 = 重建带，按行归一（每行加起来 100%，"
          f"行尾是落在四带之外的）")
    hdr = "".join(f"{f'{lo}-{hi}':>10}" for lo, hi in bands)
    print(f"{'真实深度':>12}{'N':>9}{hdr}{'带外':>8}")
    for tlo, thi in tb:
        mt = (d_true >= tlo) & (d_true < thi)
        Nt = int(mt.sum())
        if Nt == 0:
            continue
        cells, tot = [], 0.0
        for lo, hi in bands:
            f = float(np.mean((d_rec[mt] >= lo) & (d_rec[mt] < hi)))
            cells.append(f"{100*f:9.1f}%"); tot += f
        print(f"{f'{tlo}-{thi}':>12}{Nt:9d}" + "".join(cells) + f"{100*(1-tot):7.1f}%")

    if args.out:
        out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
        json.dump(dict(analysis_dir=str(ad), label=args.label, n_events=n,
                       truth_source=tsrc, void_depth=[lo_v, hi_v],
                       depth_err=dict(median=float(np.median(err)),
                                      p16=float(np.percentile(err, 16)),
                                      p84=float(np.percentile(err, 84)),
                                      abs_median=float(np.median(np.abs(err)))),
                       bands=rows),
                  open(out / "diag_shallow_band.json", "w", encoding="utf-8"),
                  ensure_ascii=False, indent=1)
        print(f"\n写入 {out / 'diag_shallow_band.json'}")


if __name__ == "__main__":
    main()
