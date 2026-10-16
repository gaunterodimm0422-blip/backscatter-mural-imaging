"""用真值顶点直接量"横向刻度压缩"，判定它是不是纯几何效应。

要检验的假说
------------
重建公式 P = S + t*v：方向 v 由符合标记给出，t 由能量反推。横向坐标与深度是
同一条射线上的两个分量，被同一个 t 锁在一起。记

    u = x - x_source（离探测头轴心的横向距离）
    h = z_source - z（源到该点的轴向距离 = 17.5 mm + 深度）

同一条射线上 u/h 是常数，于是对同一个事例应当**恒等**地有

    R := u_rec / u_true  ==  h_rec / h_true =: G

也就是说：**深度被低估多少倍，横向偏移就被压缩多少倍。** 如果 R == G 逐事例成立，
这就不是统计规律而是几何恒等式，机制即被证明；剩下的唯一问题是 G 的分布长什么样，
而那完全由深度重建的好坏决定。

R 与 G 的偏离只可能来自两件事：标记给出的方向不是真实入射方向（标记角误差，
实测 p50 约 1.3°），以及光子在到达该顶点之前已经散射过（多次散射，此时真值顶点
根本不在原射线上）。所以 R - G 的分布顺带量出这两项的合计影响。

**本脚本使用真值（PhaseSpaceActor 的康普顿顶点），只用于机制诊断，
不用于任何性能指标。** 任何对外报告的数字都不得出自这里。

用法
----
  <python> scripts\\check_lateral_scale.py ^
      --analysis-dir runs\\mural_ref2_t20_L-15_4m_seed24091122\\canonical_response ^
      --config configs\\mural_edge_20260911.json ^
      --void-depth 20 40
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


def main():
    ap = argparse.ArgumentParser(description="横向刻度压缩的真值诊断")
    ap.add_argument("--analysis-dir", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--void-depth", nargs=2, type=float, default=[20.0, 40.0],
                    help="空鼓所占的真实深度区间 mm，用于算携带边界信息的事例的平均压缩比")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from backscatter.config import load_config
    from backscatter.pose_selection import build_selection_pose_aware

    config = load_config(Path(args.config))
    ad = Path(args.analysis_dir)
    pairs = dict(np.load(ad / "pairs.npz", allow_pickle=True))
    recon = dict(np.load(ad / "reconstruction.npz", allow_pickle=True))

    # 真值顶点是 analyze 写进 reconstruction.npz 的（pipeline.py 的 reconstruction 字典），
    # 而 apply_response 生成的 canonical_response 视图不一定带。所以按下面的顺序找，
    # 找到之后一律用 key 列逐事例对齐，不假设行序相同。
    TKEYS = ("truth_x_mm", "truth_y_mm", "truth_z_mm")
    truth = {k: recon[k] for k in TKEYS} if all(k in recon for k in TKEYS) else None
    truth_src = "本目录 reconstruction.npz"
    if truth is None:
        cands = [ad / "pairs.npz", ad.parent / "reconstruction.npz", ad.parent / "pairs.npz"]
        k_here = np.asarray(recon["key"]) if "key" in recon else (
            np.asarray(pairs["key"]) if "key" in pairs else None)
        if k_here is None:
            raise SystemExit("本目录的 npz 里没有 key 列，无法对齐真值。")
        for c in cands:
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
            truth = {}
            for k in TKEYS:
                v = np.full(len(k_here), np.nan, dtype=np.float64)
                v[hit] = np.asarray(src[k], dtype=np.float64)[idx[hit]]
                truth[k] = v
            truth_src = f"{c.parent.name}/{c.name}（按 key 对齐 {int(hit.sum()):,}/{len(k_here):,}）"
            break
    if truth is None:
        raise SystemExit(
            "找不到真值顶点列 truth_x_mm/truth_y_mm/truth_z_mm。\n"
            f"已查：{ad/'reconstruction.npz'}、{ad/'pairs.npz'}、"
            f"{ad.parent/'reconstruction.npz'}、{ad.parent/'pairs.npz'}。\n"
            "这一档的 analyze 可能没有带 --truth 参数，或者 PhaseSpaceActor 顶点没匹配上。")
    for k in TKEYS:
        pairs[k] = np.asarray(truth[k], dtype=np.float64)

    selected, _chain, _diag = build_selection_pose_aware(pairs, config)
    keep = selected & np.asarray(recon["valid"], dtype=bool)
    sx = np.asarray(pairs["source_x_mm"], float)
    sz = np.asarray(pairs["source_z_mm"], float)
    tx = np.asarray(pairs["truth_x_mm"], float)
    tz = np.asarray(pairs["truth_z_mm"], float)
    keep &= np.isfinite(tx) & np.isfinite(tz)
    n_keep = int(keep.sum())
    if n_keep < 100:
        raise SystemExit(f"可用事例只有 {n_keep} 个，太少。")

    u_rec = np.asarray(recon["x_mm"], float)[keep] - sx[keep]
    u_true = tx[keep] - sx[keep]
    h_rec = sz[keep] - np.asarray(recon["z_mm"], float)[keep]
    h_true = sz[keep] - tz[keep]
    d_rec = -np.asarray(recon["z_mm"], float)[keep]
    d_true = -tz[keep]

    print("=" * 74)
    print("横向刻度压缩的真值诊断（**使用真值，仅用于机制判定，不得用作性能指标**）")
    print("=" * 74)
    print(f"\n分析目录 : {ad}")
    print(f"真值来源 : {truth_src}")
    print(f"可用事例 : {n_keep:,}（已过位姿感知筛选 + 可重建 + 真值顶点匹配上）")
    print(f"源轴向位置 z_source = {np.median(sz[keep]):.2f} mm，"
          f"探测头轴心 x = {np.median(sx[keep]):.2f} mm")

    # ---- 自检 1：真值顶点与重建点是否在同一条射线上 ----
    ok = (np.abs(u_true) > 2.0) & (h_true > 1.0) & (h_rec > 1.0)
    R = u_rec[ok] / u_true[ok]
    G = h_rec[ok] / h_true[ok]
    diff = R - G
    print(f"\n[自检 1] R = u_rec/u_true 与 G = h_rec/h_true 是否逐事例相等"
          f"（|u_true| > 2 mm 的 {int(ok.sum()):,} 个事例）")
    print(f"         R - G：中位 {np.median(diff):+.4f}   "
          f"16~84 分位 [{np.percentile(diff,16):+.3f}, {np.percentile(diff,84):+.3f}]   "
          f"|R-G| < 0.05 的比例 {100*np.mean(np.abs(diff)<0.05):.1f}%")
    print("         判读：中位接近 0 且多数事例落在 ±0.05 内 => 压缩是几何恒等式，"
          "机制成立；\n               分散则说明标记角误差与多次散射把射线打偏了。")

    # ---- 自检 2：深度确实被低估 ----
    print(f"\n[自检 2] 深度重建偏差：d_rec - d_true 中位 "
          f"{np.median(d_rec - d_true):+.2f} mm，"
          f"16~84 分位 [{np.percentile(d_rec-d_true,16):+.1f}, "
          f"{np.percentile(d_rec-d_true,84):+.1f}] mm")
    print(f"         d_rec 中位 {np.median(d_rec):.2f} mm，"
          f"d_true 中位 {np.median(d_true):.2f} mm")

    # ---- 压缩比 G 的分布，按重建深度带 ----
    bands = [tuple(b) for b in config["observables"]["depth_bands_mm"]]
    lo_v, hi_v = args.void_depth
    inslab = (d_true >= lo_v) & (d_true <= hi_v)
    print(f"\n压缩比 G = h_rec/h_true 的分布")
    print(f"{'重建深度带':>12s}{'全部事例':>10s}{'G中位':>8s}"
          f"{'空鼓层内事例':>14s}{'G中位':>8s}")
    for lo, hi in bands:
        m = (d_rec >= lo) & (d_rec < hi)
        ms = m & inslab
        g1 = np.median(G[m[ok]]) if m[ok].any() else float("nan")
        g2 = np.median(G[ms[ok]]) if ms[ok].any() else float("nan")
        print(f"{lo:5d}-{hi:<6d}{int(m.sum()):10d}{g1:8.3f}"
              f"{int(ms.sum()):14d}{g2:8.3f}")
    gslab = float(np.median(G[inslab[ok]])) if inslab[ok].any() else float("nan")
    print(f"\n真实深度落在空鼓层 [{lo_v:.0f}, {hi_v:.0f}] mm 内的事例："
          f"{int(inslab.sum()):,} 个，G 中位 = {gslab:.3f}")
    print(f"  => 这就是对边界拟合会被压缩多少的预言值。"
          f"M2 实测的边界压缩比是 0.70。")

    if args.out:
        out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
        json.dump(dict(
            analysis_dir=str(ad), n_events=n_keep,
            R_minus_G=dict(median=float(np.median(diff)),
                           p16=float(np.percentile(diff, 16)),
                           p84=float(np.percentile(diff, 84)),
                           frac_within_0p05=float(np.mean(np.abs(diff) < 0.05))),
            depth_bias_mm=float(np.median(d_rec - d_true)),
            d_rec_median=float(np.median(d_rec)), d_true_median=float(np.median(d_true)),
            G_median_in_slab=gslab, void_depth_mm=[lo_v, hi_v],
            G_by_band={f"{lo}-{hi}": float(np.median(G[((d_rec >= lo) & (d_rec < hi))[ok]]))
                       for lo, hi in bands},
        ), open(out / "lateral_scale.json", "w"), ensure_ascii=False, indent=1)
        print(f"\n结果写入 {out/'lateral_scale.json'}")


if __name__ == "__main__":
    main()
