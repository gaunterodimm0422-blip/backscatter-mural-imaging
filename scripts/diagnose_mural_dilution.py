"""M1 诊断：深处重建出来的事例到底是什么？稀释从哪来？

背景
----
M1 的预注册判据没过：45 mm 以深的合并显著性只有 0.43σ（门槛 5σ）。
对比参照档的重建深度剖面与事例级蒙卡预估，实测的深尾重 4 倍
（87 mm 处 4.24 倍）。怀疑那些深处事例是多次散射与误重建的产物，
**不携带空鼓信息**，把信号稀释掉了。本脚本就是去坐实或推翻这个怀疑。

做法
----
1. 直接问数据：重建深度 > 60 mm 的事例，它们在体模里散射了几次？在哪一层散射的？
   `d1_n_phantom_compton` 与 `d1_compt_volume` 都是 GATE Hits 的分支，
   对**每个**事例都有（三层都挂了 attachPhantomSD），不依赖 truth actor。
2. 按"体模内康普顿次数"分档，各自重做空鼓/参照的深度剖面比值，
   看凹陷（15~39 mm）与台阶（≥51 mm）的显著性怎么变。
3. 对能匹配上真值顶点的事例（truth actor 只挂在粗泥层，覆盖 5~40 mm），
   直接量重建深度与真深度的差。

**这些都是真值量，只能做诊断，不能当判选。** 它的用途是决定下一步该往哪走：
如果稀释主要来自多次散射，课题就变成"用什么**可观测量**剔除这类事例"；
如果不是，才轮到加统计量或上双位姿。

显著性口径
----------
先除掉整体计数比（即只看形状），再逐档算泊松显著性。
凹陷与台阶的合并显著性 = 各档 σ 之和 / sqrt(档数)。
**这是事后描述性指标，不是预注册判据。**

用法（在数据所在的那台机器上跑）
--------------------------------
python scripts\\diagnose_mural_dilution.py ^
    --void runs\\mural_void_t10_L-15_2m_seed24091011\\canonical_response ^
    --reference runs\\mural_ref_t10_L-15_2m_seed24091012\\canonical_response ^
    --config configs\\mural_void_20260910.json ^
    --out outputs\\m1_diagnosis
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

from backscatter.config import load_config                          # noqa: E402
from backscatter.pose_selection import build_selection_pose_aware    # noqa: E402


def _load(analysis_dir: Path):
    pairs = dict(np.load(analysis_dir / "pairs.npz", allow_pickle=True))
    recon = dict(np.load(analysis_dir / "reconstruction.npz", allow_pickle=True))
    sp = analysis_dir / "summary.json"
    if not sp.exists():
        sp = analysis_dir.parent / "summary.json"
    n_prim = None
    if sp.exists():
        n_prim = (json.loads(sp.read_text(encoding="utf-8")).get("gate_metadata")
                  or {}).get("primary_event_count")
    selected, _c, _d = build_selection_pose_aware(pairs, config=CONFIG)
    keep = selected & np.asarray(recon["valid"], dtype=bool)
    vol = pairs.get("d1_compt_volume")
    vol = (np.asarray(vol).astype(str)[keep] if vol is not None
           else np.full(int(keep.sum()), "?"))
    return dict(
        depth=-np.asarray(recon["z_mm"], dtype=np.float64)[keep],
        nph=np.asarray(pairs["d1_n_phantom_compton"], dtype=np.int32)[keep],
        ncr=np.asarray(pairs["d1_n_crystal_compton"], dtype=np.int32)[keep],
        vol=vol,
        matched=np.asarray(recon["truth_matched"], dtype=bool)[keep],
        tz=-np.asarray(recon["truth_z_mm"], dtype=np.float64)[keep],
        n_prim=int(n_prim) if n_prim else None, n=int(keep.sum()))


def shape_significance(dep_v, dep_r, edges):
    """除掉整体归一后的逐档泊松显著性，以及凹陷/台阶的合并值。"""
    ctr = 0.5 * (edges[1:] + edges[:-1])
    hv, _ = np.histogram(dep_v, bins=edges)
    hr, _ = np.histogram(dep_r, bins=edges)
    if hr.sum() == 0 or hv.sum() == 0:
        return ctr, hr, hv, np.full(len(ctr), np.nan), np.nan, np.nan, np.nan
    g = hv.sum() / hr.sum()
    exp_v = hr * g
    with np.errstate(divide="ignore", invalid="ignore"):
        sig = np.where(exp_v > 0, (hv - exp_v) / np.sqrt(np.maximum(hv + exp_v * g, 1e-9)),
                       np.nan)
    dip = (ctr >= 15) & (ctr <= 39) & np.isfinite(sig)
    step = (ctr >= 51) & np.isfinite(sig)
    cd = float(sig[dip].sum() / np.sqrt(max(dip.sum(), 1))) if dip.any() else np.nan
    cs = float(sig[step].sum() / np.sqrt(max(step.sum(), 1))) if step.any() else np.nan
    return ctr, hr, hv, sig, cd, cs, float(g)


def main() -> int:
    global CONFIG
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--void", required=True, type=Path)
    ap.add_argument("--reference", required=True, type=Path)
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--bin-width-mm", type=float, default=6.0)
    ap.add_argument("--depth-max-mm", type=float, default=120.0)
    ap.add_argument("--deep-from-mm", type=float, default=60.0)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    CONFIG = load_config(str(args.config))
    ph = CONFIG["phantom"]
    t_true = float(ph["void"]["thickness_mm"])
    d_top = float(ph["void"]["depth_mm"])

    print("[1/4] 读入与位姿感知筛选")
    V, R = _load(args.void), _load(args.reference)
    print(f"  空鼓 {V['n']:6d} 事例（{V['n_prim']} 初级）   "
          f"参照 {R['n']:6d} 事例（{R['n_prim']} 初级）")
    edges = np.arange(0.0, args.depth_max_mm + args.bin_width_mm, args.bin_width_mm)

    print(f"\n[2/4] 重建深度 > {args.deep_from_mm:.0f} mm 的事例，究竟是什么（用参照档）")
    deep = R["depth"] > args.deep_from_mm
    print(f"  这类事例占参照档 {deep.mean():.1%}（{int(deep.sum())} 个）")
    print("\n  体模内康普顿次数分布：")
    print(f"    {'次数':>5s}{'全部':>10s}{'占比':>8s}{'深处':>10s}{'占比':>8s}")
    vals = np.unique(R["nph"])
    for k in vals[:7]:
        a = R["nph"] == k
        b = a & deep
        print(f"    {int(k):5d}{int(a.sum()):10d}{a.mean():8.1%}"
              f"{int(b.sum()):10d}{(b.sum()/max(deep.sum(),1)):8.1%}")
    print("\n  散射发生在哪一层：")
    for v in np.unique(R["vol"])[:8]:
        a = R["vol"] == v
        b = a & deep
        print(f"    {str(v):>16s}{int(a.sum()):10d}{a.mean():8.1%}"
              f"{int(b.sum()):10d}{(b.sum()/max(deep.sum(),1)):8.1%}")
    m = R["matched"]
    if m.sum() > 100:
        dz = R["depth"][m] - R["tz"][m]
        print(f"\n  能匹配真值顶点的事例（truth actor 只挂粗泥层，覆盖 5~40 mm）："
              f"{m.mean():.1%}")
        print(f"    重建深度 - 真深度: 中位 {np.median(dz):+.1f} mm，"
              f"p5/p95 = {np.percentile(dz,5):+.1f} / {np.percentile(dz,95):+.1f} mm")
        md = m & deep
        if md.sum() > 20:
            print(f"    其中重建到 >{args.deep_from_mm:.0f} mm 的：{int(md.sum())} 个，"
                  f"真深度中位 {np.median(R['tz'][md]):.1f} mm"
                  f"  <- 真深度远小于重建深度就说明深尾是误重建")

    print("\n[3/4] 分档重做剖面比值（形状口径，事后描述，非判据）")
    subsets = [
        ("全部（现状）", lambda d: np.ones(len(d["depth"]), bool)),
        ("体模内单次康普顿", lambda d: d["nph"] == 1),
        ("体模内多次康普顿", lambda d: d["nph"] >= 2),
        ("单次 且 D1晶体内无康普顿", lambda d: (d["nph"] == 1) & (d["ncr"] == 0)),
    ]
    rows = []
    print(f"  {'子集':>26s}{'参照n':>9s}{'空鼓n':>9s}{'整体比':>9s}"
          f"{'>60mm占比':>11s}{'凹陷σ':>9s}{'台阶σ':>9s}")
    for name, fn in subsets:
        mv, mr = fn(V), fn(R)
        if mr.sum() < 200 or mv.sum() < 200:
            print(f"  {name:>26s}{int(mr.sum()):9d}{int(mv.sum()):9d}"
                  f"{'—':>9s}{'—':>11s}{'—':>9s}{'—':>9s}   事例太少，跳过")
            rows.append(dict(subset=name, n_ref=int(mr.sum()), n_void=int(mv.sum())))
            continue
        ctr, hr, hv, sig, cd, cs, g = shape_significance(V["depth"][mv], R["depth"][mr], edges)
        frac_deep = float((R["depth"][mr] > args.deep_from_mm).mean())
        print(f"  {name:>26s}{int(mr.sum()):9d}{int(mv.sum()):9d}{g:9.4f}"
              f"{frac_deep:11.1%}{cd:9.2f}{cs:9.2f}")
        rows.append(dict(subset=name, n_ref=int(mr.sum()), n_void=int(mv.sum()),
                         global_ratio=g, frac_deep=frac_deep,
                         dip_sigma=cd, step_sigma=cs,
                         depth_centres_mm=ctr.tolist(),
                         counts_ref=hr.tolist(), counts_void=hv.tolist(),
                         shape_sigma=[None if not np.isfinite(x) else float(x) for x in sig]))

    print("\n[4/4] 怎么读这张表")
    print("  · 若「体模内单次康普顿」这一档的 >60mm 占比明显下降、凹陷σ与台阶σ明显上升，")
    print("    说明稀释主要来自多次散射 —— 下一步是找可观测的抑制手段，不是加统计量。")
    print("  · 若各档的 σ 都差不多，说明稀释不是多次散射造成的 —— 那就是统计量不够，")
    print("    或者深度分辨本身不足以支撑这个观测量，得换路子（双位姿）。")
    print(f"  · 真值参考：空鼓深 {d_top:.0f}~{d_top+t_true:.0f} mm，中心 {d_top+t_true/2:.0f} mm。")

    json.dump({"void_dir": str(args.void), "reference_dir": str(args.reference),
               "bin_width_mm": args.bin_width_mm, "deep_from_mm": args.deep_from_mm,
               "n_primary": {"void": V["n_prim"], "reference": R["n_prim"]},
               "n_selected": {"void": V["n"], "reference": R["n"]},
               "note": "真值量诊断，非判选；显著性为事后描述性指标",
               "subsets": rows},
              open(args.out / "m1_diagnosis.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print(f"\n写出 {args.out / 'm1_diagnosis.json'}")
    return 0


CONFIG = None
if __name__ == "__main__":
    raise SystemExit(main())
