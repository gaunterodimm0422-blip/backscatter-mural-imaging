"""L3-A：D2 位置口径对照 —— 全事例质心 vs 按晶体分组（v4）。

为什么要做这件事
----------------
gate/geometry_detectors.mac 里 D2 是 50x50 颗 0.93 mm 晶体、厚 20 mm 的 GAGG 阵列，
晶体已作为 SPECThead 的 crystal 层挂上探测器，ROOT 里有 crystalID。
但 backscatter/rootio.py 聚合 Hits 时是对**整个事例**的所有 hit 求能量加权质心
（`centroid = np.divide(weighted, total_energy, ...)`），没有用晶体分割。
20 mm 厚 GAGG 里 511 keV 光子约 71% 会先在晶体内康普顿散射再被吸收，
散射点常落到相邻晶体，质心因此落在作用点之间，标记方向被带偏。

本脚本不改几何、不重跑 GATE，只重读已有 hits.root，按 (runID, eventID, crystalID)
分组后比较四种位置口径：

    A  全事例能量质心            现状。保留全部事例。
    B  沉积能量最高的晶体        保留全部事例。可实现。
    C  最早被击中的晶体          保留全部事例。**这是"入射晶体"的真值口径**——
                               光子穿过 20 mm GAGG 只要 ~67 ps，晶体之间的时间差
                               是 10-100 ps 量级，真实 GAGG+SiPM 的符合时间分辨
                               在 200-400 ps，**排不出这个序**。所以 C 只能当
                               可行性上界读：它回答"信息在不在事例里"，不回答
                               "能不能测出来"。
    D  康普顿序重建             对每颗候选晶体 i 检验"它是入射点"这一假设：
                               几何散射角 cos θ_geo = dir(S→r_i)·dir(r_i→其余晶体质心)
                               与能量学散射角 cos θ_kin = 2 − 511/(511 − e_i)
                               是否自洽，取偏差最小者。只用到源位置 S（已知硬件几何）、
                               各晶体位置与各晶体沉积能量——**全部是可观测量**。

自检（负对照，不过就别信结论）
------------------------------
1. 口径 A 的理想位置必须与 pairs.npz 的 ideal_d2_* 逐点相同（阈值 5e-3 mm）。
2. 单晶体事例（重数=1）在 A/B/C/D 四种口径下必须给出完全相同的结果。
3. 深度 FWHM 在四种口径下应基本不变——深度由 D1 能量分辨决定，与标记位置无关。

用法（Windows PowerShell，一整行）
----------------------------------
E:\\python3.12\\python.exe C:\\Users\\HOUWENDI\\Documents\\beisanshe\\scripts\\analyze_d2_crystal_readout.py --root "E:\\vbgate\\share\\beisanshe_v1\\gate\\output\\rod_x_depth30_5000000_seed24083101\\hits.root" --analysis-dir "C:\\Users\\HOUWENDI\\Documents\\beisanshe\\runs\\rod_x_depth30_5m_seed24083101\\canonical_response" --lateral-axis y --out "C:\\Users\\HOUWENDI\\Documents\\beisanshe\\outputs\\l3a_d2_crystal_readout"

第二次及以后加 --cache 即可跳过重读 ROOT（秒级）。

v3 相对 v2 的增补
-----------------
1. 横向 FWHM 同时按多个 KDE 带宽给出（默认 1.0 与 1.5），并额外给一列"自动升带宽"。
   带宽 1.0 与既往全部已发布数字同口径；位置口径修好之后 1 mm 晶体网格会把剖面离散成
   双瓣（50 列是偶数，没有中心晶体，中心在 ±0.5、±1.5 …），低带宽可能被判成多峰而拒绝估计。
   同时给出与带宽完全无关的 IQR 作为交叉验证。
2. 口径 D 增加"序重建置信裕度"判选：最优与次优假设的自洽度之差越大越可信，
   按裕度保留前 75% / 50% / 25%，给出分辨率与计数率的连续取舍曲线。
   单晶体事例的裕度记为 +inf，天然排在最前。
3. 口径 D 按点亮晶体数分档给出分辨率与纯度，用来定位序重建在哪一档失效。

v4 相对 v3 的增补与修正
-----------------------
1. **修正**：v3 的序重建判据用的是理想连续晶体位置，而最终重建用的是体素化位置——
   真实探测器只有后者。v4 默认用体素化后的位置做排序（--order-with measured），
   --order-with ideal 保留下来只为看"位置精度对排序的影响有多大"。
2. 新增**口径 E：最小 z 晶体**。光子从 D2 下表面（全局 z=50）射入、向上穿行 20 mm，
   第一次作用通常 z 更小。按 Klein–Nishina，511 keV 前向散射占多数，所以"z 最小"
   本身就是一个纯几何、零算术的入射晶体判据。
3. 新增**口径 F**：D 与 E 指向同一颗时用它，不一致时回退到 E。
4. 新增 --no-doi：把所有 z 压到晶体中面 60 mm，检验 D/E 对深度作用位置（DOI）能力的依赖。
   注意 2.5 mm 的 DOI 分辨是 configs 里的假设，20 mm 长晶体能否做到需向供应商确认。
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

from backscatter.electronics import energy_sigma_keV  # noqa: E402
from backscatter.kinematics import solve_scatter_points  # noqa: E402
from backscatter.response import quantize_positions_to_voxel_centers  # noqa: E402
from backscatter.rootio import _event_key  # noqa: E402
from backscatter.standards import bootstrap_fwhm, estimate_fwhm_kde  # noqa: E402

BASE_BRANCHES = ("runID", "eventID", "edep", "time", "posX", "posY", "posZ")
ID_CANDIDATES = ("crystalID", "level1ID", "volumeID", "level2ID")
# 与 configs/transverse_fwhm_5mm_frozen.json 的 detectors.d2.position_response 一致
D2_VOXEL_MM = (1.0, 1.0, 2.5)
D2_ERES_AT_511 = 0.12           # 同一份配置里的 D2 能量分辨
D2_CENTER_Z_MM = 60.0           # D2 晶体中面全局 z
E0_KEV = 511.0
MAX_CRYSTALS = 4096             # 50x50 = 2500 < 4096


def pick_id_branch(available: list[str]) -> str:
    for name in ID_CANDIDATES:
        if name in available:
            return name
    raise KeyError("Hits_crystal 里找不到晶体编号分支；可用分支：" + ", ".join(sorted(available)))


def read_target_hits(root_path, tree_name, want_keys, id_branch, step_size):
    import uproot
    want = np.sort(np.asarray(want_keys, dtype=np.int64))
    acc = {name: [] for name in (*BASE_BRANCHES, id_branch)}
    n_chunk = 0
    with uproot.open(root_path) as handle:
        for chunk in handle[tree_name].iterate([*BASE_BRANCHES, id_branch],
                                               library="np", step_size=step_size):
            n_chunk += 1
            key = _event_key(np.asarray(chunk["runID"]), np.asarray(chunk["eventID"]))
            mask = np.isin(key, want)
            if not mask.any():
                continue
            for name in acc:
                acc[name].append(np.asarray(chunk[name])[mask])
    print(f"    读了 {n_chunk} 个分块")
    if not acc["runID"]:
        raise RuntimeError("目标事例在 ROOT 里一个 hit 都没找到，键的定义可能不一致")
    return {name: np.concatenate(values) for name, values in acc.items()}


def group_by_crystal(rec, id_branch):
    """把 hit 聚成"每事例一组晶体"。返回不规则结构的扁平表 + 每事例的起止下标。"""
    key = _event_key(rec["runID"], rec["eventID"])
    cid = np.asarray(rec[id_branch], dtype=np.int64).ravel()
    if cid.min() < 0 or cid.max() >= MAX_CRYSTALS:
        raise ValueError(f"晶体编号超出 [0,{MAX_CRYSTALS}) 范围：{cid.min()}..{cid.max()}")
    comp = key * np.int64(MAX_CRYSTALS) + cid
    order = np.argsort(comp, kind="stable")
    comp_s, key_s = comp[order], key[order]
    edep = np.maximum(np.asarray(rec["edep"], dtype=np.float64)[order], 0.0) * 1000.0
    time_s = np.asarray(rec["time"], dtype=np.float64)[order]
    pos = np.column_stack([np.asarray(rec[b], dtype=np.float64)[order]
                           for b in ("posX", "posY", "posZ")])

    starts = np.r_[0, np.flatnonzero(comp_s[1:] != comp_s[:-1]) + 1]
    c_energy = np.add.reduceat(edep, starts)
    c_pos = np.column_stack([np.add.reduceat(edep * pos[:, i], starts) for i in range(3)])
    safe = np.where(c_energy > 0.0, c_energy, 1.0)
    c_pos = c_pos / safe[:, None]
    dead = c_energy <= 0.0
    if dead.any():
        c_pos[dead] = pos[starts][dead]
    c_time = np.minimum.reduceat(time_s, starts)
    c_key = key_s[starts]

    ev_starts = np.r_[0, np.flatnonzero(c_key[1:] != c_key[:-1]) + 1]
    ev_ends = np.r_[ev_starts[1:], len(c_key)]
    return dict(c_key=c_key, c_pos=c_pos, c_energy=c_energy, c_time=c_time,
                ev_key=c_key[ev_starts], ev_starts=ev_starts, ev_ends=ev_ends,
                mult=(ev_ends - ev_starts).astype(np.int32))


def choose_positions(g, source_xyz, rng, smear_crystal_energy=True,
                     order_with="measured", no_doi=False):
    """四种口径各返回 (N_events, 3) 的 D2 位置，外加 D 选中的晶体下标。"""
    c_pos, c_e, c_t = g["c_pos"], g["c_energy"], g["c_time"]
    if no_doi:                       # 没有 DOI 能力时，z 只能取晶体中面
        c_pos = c_pos.copy()
        c_pos[:, 2] = D2_CENTER_Z_MM
    # 序重建只能用真实读出的位置，不能用理想连续位置
    r_order = (quantize_positions_to_voxel_centers(c_pos, resolution_mm=D2_VOXEL_MM)
               if order_with == "measured" else c_pos)
    starts, ends = g["ev_starts"], g["ev_ends"]
    n = len(starts)
    e_meas = c_e.copy()
    if smear_crystal_energy:
        e_meas = np.maximum(c_e + rng.normal(0.0, 1.0, len(c_e)) *
                            energy_sigma_keV(c_e, fwhm_fraction_at_511=D2_ERES_AT_511), 0.0)

    pos_a = np.empty((n, 3)); idx_b = np.empty(n, np.int64)
    idx_c = np.empty(n, np.int64); idx_d = np.empty(n, np.int64)
    idx_e = np.empty(n, np.int64)        # 口径 E：z 最小（最靠近 D2 入射面）的晶体
    margin = np.full(n, np.inf)          # 最优与次优假设的自洽度之差，越大越确定
    for k, (s, e) in enumerate(zip(starts, ends)):
        w = c_e[s:e]
        pos_a[k] = (w[:, None] * c_pos[s:e]).sum(0) / max(w.sum(), 1e-12)
        idx_b[k] = s + int(np.argmax(e_meas[s:e]))
        idx_c[k] = s + int(np.argmin(c_t[s:e]))
        idx_e[k] = s + int(np.argmin(r_order[s:e, 2]))
        m = e - s
        if m == 1:
            idx_d[k] = s
            continue                      # 只有一颗晶体，序是确定的，裕度记为 +inf
        # 康普顿序重建：只用 S、各晶体位置、各晶体测量能量
        r = r_order[s:e]
        em = e_meas[s:e]
        tot = max(em.sum(), 1e-12)
        scores = np.full(m, np.inf)
        for i in range(m):
            others = np.ones(m, bool); others[i] = False
            wo = em[others]
            cen = (wo[:, None] * r[others]).sum(0) / max(wo.sum(), 1e-12)
            din = r[i] - source_xyz[k]
            dout = cen - r[i]
            nin, nout = np.linalg.norm(din), np.linalg.norm(dout)
            if nin <= 0 or nout <= 0:
                continue
            cos_geo = float(din @ dout / (nin * nout))
            rest = tot - em[i]
            if rest <= 1.0:
                continue
            cos_kin = 2.0 - E0_KEV / rest
            if not np.isfinite(cos_kin) or abs(cos_kin) > 1.2:
                continue
            scores[i] = abs(cos_geo - cos_kin)
        order_s = np.argsort(scores)
        idx_d[k] = s + int(order_s[0])
        if np.isfinite(scores[order_s[0]]) and np.isfinite(scores[order_s[1]]):
            margin[k] = float(scores[order_s[1]] - scores[order_s[0]])
        else:
            margin[k] = 0.0
    idx_f = np.where(idx_d == idx_e, idx_d, idx_e)   # 不一致时回退到 E
    return (pos_a, c_pos[idx_b], c_pos[idx_c], c_pos[idx_d], c_pos[idx_e], c_pos[idx_f],
            idx_c, idx_d, idx_e, margin)


def _auto_bw(values, start=1.0, stop=4.0, step=0.25):
    """从 start 起逐步放宽带宽直到 KDE 给得出 FWHM；返回 (fwhm, 用到的带宽, 状态)。"""
    bw = start
    while bw <= stop + 1e-9:
        est = estimate_fwhm_kde(values, bandwidth_scale=bw)
        if est["status"] == "ok":
            return float(est["fwhm"]), float(bw), "ok"
        bw += step
    return None, None, est["status"]


def measure(name, source, d1, d2, energy, lateral_index, keep, n_boot, bws):
    solution = solve_scatter_points(source, d1, d2, energy)
    good = solution.valid & keep
    row = {"variant": name, "n": int(good.sum())}
    if good.sum() < 20:
        row.update(lateral={}, depth={}, note="事例数不足 20，未估计")
        return row
    lat, dep = solution.points_mm[good, lateral_index], solution.points_mm[good, 2]
    row["lateral_iqr"] = float(np.subtract(*np.percentile(lat, [75, 25])))
    row["depth_iqr"] = float(np.subtract(*np.percentile(dep, [75, 25])))
    row["lateral"], row["depth"] = {}, {}
    for bw in bws:
        fl = estimate_fwhm_kde(lat, bandwidth_scale=bw)
        fd = estimate_fwhm_kde(dep, bandwidth_scale=bw)
        bl = bootstrap_fwhm(lat, n_bootstrap=n_boot, seed=24090411, bandwidth_scale=bw)
        bd = bootstrap_fwhm(dep, n_bootstrap=n_boot, seed=24090412, bandwidth_scale=bw)
        row["lateral"][f"{bw:g}"] = {"fwhm": fl.get("fwhm"), "status": fl["status"],
                                     "ci": bl.get("fwhm_95_ci")}
        row["depth"][f"{bw:g}"] = {"fwhm": fd.get("fwhm"), "status": fd["status"],
                                   "ci": bd.get("fwhm_95_ci")}
    f, b, st = _auto_bw(lat)
    row["lateral_auto"] = {"fwhm": f, "bandwidth": b, "status": st}
    return row


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, type=Path)
    ap.add_argument("--analysis-dir", required=True, type=Path)
    ap.add_argument("--tree", default="Hits_crystal")
    ap.add_argument("--lateral-axis", default="y", choices=("x", "y"))
    ap.add_argument("--step-size", default="300 MB")
    ap.add_argument("--bootstrap", type=int, default=400)
    ap.add_argument("--bandwidth-scales", default="1.0,1.5",
                    help="KDE 带宽倍数列表，逗号分隔。1.0 与既往全部已发布数字同口径；"
                         "位置口径修好之后 1 mm 晶体网格会把剖面离散成双瓣，"
                         "低带宽可能被判成多峰而拒绝估计，故同时给 1.5 与自动升带宽列。")
    ap.add_argument("--cache", action="store_true", help="复用 --out 里已缓存的晶体级数据")
    ap.add_argument("--order-with", default="measured", choices=("measured", "ideal"),
                    help="序重建判据用哪套晶体位置。measured=体素化后的真实读出位置（默认，"
                         "与最终重建口径一致）；ideal=理想连续位置（乐观，仅用于看位置精度的影响）")
    ap.add_argument("--no-doi", action="store_true",
                    help="假设 D2 没有深度作用位置（DOI）能力：所有 z 取晶体中面 60 mm。"
                         "用来检验口径 D/E 对 DOI 的依赖程度")
    ap.add_argument("--seed", type=int, default=24090431)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    bws = [float(v) for v in args.bandwidth_scales.split(",") if v.strip()]
    args.out.mkdir(parents=True, exist_ok=True)
    cache_path = args.out / "d2_crystal_cache.npz"

    pairs = dict(np.load(args.analysis_dir / "pairs.npz", allow_pickle=True))
    recon = dict(np.load(args.analysis_dir / "reconstruction.npz", allow_pickle=True))
    selected = np.asarray(recon["selected_valid"], dtype=bool)
    keys = np.asarray(pairs["key"], dtype=np.int64)[selected]
    print(f"[1/6] 目标事例 {keys.size} 个（selected_valid）")

    if args.cache and cache_path.exists():
        z = np.load(cache_path)
        g = {k: z[k] for k in z.files}
        id_branch = str(z["id_branch"]) if "id_branch" in z.files else "(缓存)"
        print(f"[2/6] 用缓存 {cache_path.name}")
        print(f"[3/6] 缓存里 {len(g['ev_key'])} 个事例")
    else:
        import uproot
        with uproot.open(args.root) as handle:
            id_branch = pick_id_branch(list(handle[args.tree].keys()))
        print(f"[2/6] 晶体编号分支 = {id_branch}")
        print(f"[3/6] 重读 {args.root.name} 的 {args.tree} ...")
        rec = read_target_hits(args.root, args.tree, keys, id_branch, args.step_size)
        g = group_by_crystal(rec, id_branch)
        np.savez_compressed(cache_path, id_branch=id_branch, **g)
        print(f"    还原出 {len(g['ev_key'])} 个事例，缓存写入 {cache_path.name}")

    mult_vals, mult_counts = np.unique(g["mult"], return_counts=True)
    print("    点亮晶体数分布：" + "  ".join(
        f"{int(v)}颗={int(c)}({c / g['mult'].size:.0%})" for v, c in zip(mult_vals, mult_counts)))

    order = np.searchsorted(g["ev_key"], keys)
    if not np.array_equal(g["ev_key"][order], keys):
        raise RuntimeError("重读的事例键与 pairs.npz 对不上")

    source = np.column_stack([pairs["source_x_mm"], pairs["source_y_mm"],
                              pairs["source_z_mm"]]).astype(np.float64)[selected]
    d1 = np.column_stack([pairs["d1_x_mm"], pairs["d1_y_mm"],
                          pairs["d1_z_mm"]]).astype(np.float64)[selected]
    energy = np.asarray(pairs["d1_energy_keV"], dtype=np.float64)[selected]

    rng = np.random.default_rng(args.seed)
    src_by_event = np.tile(source[0], (len(g["ev_key"]), 1)).astype(np.float64)
    src_by_event[order] = source
    pa, pb, pc, pd, pe, pf, idx_c, idx_d, idx_e, margin = choose_positions(
        g, src_by_event, rng, order_with=args.order_with, no_doi=args.no_doi)
    pa, pb, pc, pd = pa[order], pb[order], pc[order], pd[order]
    pe, pf = pe[order], pf[order]
    mult = g["mult"][order]
    margin = margin[order]
    hit_d = idx_d[order] == idx_c[order]
    hit_e = idx_e[order] == idx_c[order]
    hit_f = np.where(idx_d[order] == idx_e[order], hit_d, hit_e)
    de_agree = float((idx_d[order] == idx_e[order]).mean())
    agree = float(hit_d.mean())

    ideal = np.column_stack([pairs["ideal_d2_x_mm"], pairs["ideal_d2_y_mm"],
                             pairs["ideal_d2_z_mm"]]).astype(np.float64)[selected]
    dev = float(np.abs(pa - ideal).max())
    ok1 = dev < 5e-3
    print(f"[4/6] 自检 1a：口径 A 的理想位置 vs pairs 的 ideal_d2_*，最大偏差 = {dev:.6f} mm"
          f"{'   <-- 通过' if ok1 else '   <-- 不通过，结论不可用'}")
    响应位置 = np.column_stack([pairs["d2_x_mm"], pairs["d2_y_mm"],
                            pairs["d2_z_mm"]]).astype(np.float64)[selected]
    dev_q = float(np.abs(quantize_positions_to_voxel_centers(
        pa, resolution_mm=D2_VOXEL_MM) - 响应位置).max())
    ok1b = dev_q < 5e-3
    print(f"      自检 1b：口径 A 体素化后 vs pairs 的 d2_*（流水线响应后位置），"
          f"最大偏差 = {dev_q:.6f} mm{'   <-- 通过' if ok1b else '   <-- 不通过，体素规格对不上'}")
    print(f"[5/6] 选中真入射晶体（口径 C）的比例：D 康普顿序 {agree:.1%}｜"
          f"E 最小 z {hit_e.mean():.1%}｜F（D∧E，不一致回退 E）{hit_f.mean():.1%}"
          f"｜D 与 E 意见一致的事例占 {de_agree:.1%}")
    purity_by_mult = {}
    for m in sorted(set(mult.tolist())):
        sel_m = mult == m
        purity_by_mult[int(m)] = {"n": int(sel_m.sum()), "purity_D": float(hit_d[sel_m].mean()),
                                  "purity_E": float(hit_e[sel_m].mean()),
                                  "purity_F": float(hit_f[sel_m].mean())}
        print(f"      点亮 {int(m)} 颗（n={int(sel_m.sum()):3d}）：D 选对 {hit_d[sel_m].mean():5.1%}"
              f"｜E 选对 {hit_e[sel_m].mean():5.1%}｜F 选对 {hit_f[sel_m].mean():5.1%}")

    lat_index = 0 if args.lateral_axis == "x" else 1
    keep_all = np.ones(len(source), dtype=bool)
    single = mult == 1
    rows, singles = [], []
    quant_d = quantize_positions_to_voxel_centers(pd, resolution_mm=D2_VOXEL_MM)
    for tag, raw in (("A 全事例质心（现状）", pa), ("B 最大能量晶体", pb),
                     ("C 最早时间晶体（上界）", pc), ("D 康普顿序重建", pd),
                     ("E 最小 z 晶体", pe), ("F D∧E 一致性", pf)):
        quant = quantize_positions_to_voxel_centers(raw, resolution_mm=D2_VOXEL_MM)
        rows.append(measure(tag + "·理想位置", source, d1, raw, energy, lat_index,
                            keep_all, args.bootstrap, bws))
        rows.append(measure(tag + "·体素化后", source, d1, quant, energy, lat_index,
                            keep_all, args.bootstrap, bws))
        singles.append(measure(tag + "·体素化后·仅单晶体", source, d1, quant, energy,
                               lat_index, single, args.bootstrap, bws))

    # D 的置信裕度判选：按裕度从大到小保留前 q 比例（单晶体事例裕度为 +inf，天然排在最前）
    cuts = []
    finite_rank = np.argsort(-margin, kind="stable")
    for q in (0.75, 0.50, 0.25):
        k = max(int(round(q * len(margin))), 20)
        keep = np.zeros(len(margin), bool)
        keep[finite_rank[:k]] = True
        cuts.append(measure(f"D·体素化后·保留裕度前 {q:.0%}", source, d1, quant_d, energy,
                            lat_index, keep, args.bootstrap, bws))
        cuts[-1]["purity"] = float(hit_d[keep].mean())
        cuts[-1]["kept_fraction"] = float(keep.mean())

    # D 分点亮晶体数
    by_mult = []
    for m in (1, 2, 3):
        sel_m = mult == m
        if sel_m.sum() >= 20:
            r = measure(f"D·体素化后·仅点亮 {m} 颗", source, d1, quant_d, energy,
                        lat_index, sel_m, args.bootstrap, bws)
            r["purity"] = float(hit_d[sel_m].mean())
            by_mult.append(r)
    sel_m = mult >= 4
    if sel_m.sum() >= 20:
        r = measure("D·体素化后·点亮 >=4 颗", source, d1, quant_d, energy, lat_index,
                    sel_m, args.bootstrap, bws)
        r["purity"] = float(hit_d[sel_m].mean())
        by_mult.append(r)

    key0 = f"{bws[0]:g}"
    sig = {(r["n"], None if r["lateral"].get(key0, {}).get("fwhm") is None
            else round(r["lateral"][key0]["fwhm"], 6)) for r in singles}
    ok2 = len(sig) == 1
    print(f"[6/6] 自检 2：单晶体事例在四种口径下结果一致 = {ok2}"
          f"{'   <-- 通过' if ok2 else '   <-- 不通过，分组逻辑有错'}")

    fmt = lambda v: "  n/a" if v is None else f"{v:5.2f}"
    ci = lambda v: "      n/a" if v is None else f"[{v[0]:.2f},{v[1]:.2f}]"
    head = "".join(f"{'横向FWHM@' + f'{b:g}':>11s}{'95%CI':>14s}" for b in bws)
    print(f"\n    横向 FWHM 按多个 KDE 带宽给出；带宽 1.0 与既往已发布数字同口径。")
    print(f"    {'口径':<30s}{'n':>5s}{head}{'自动带宽':>16s}{'横向IQR':>9s}{'深度FWHM@' + f'{bws[0]:g}':>13s}")
    for r in rows + singles + cuts + by_mult:
        cells = "".join(f"{fmt(r['lateral'][f'{b:g}']['fwhm']):>11s}"
                        f"{ci(r['lateral'][f'{b:g}']['ci']):>14s}" for b in bws)
        a = r.get("lateral_auto", {})
        auto = "  n/a" if a.get("fwhm") is None else f"{a['fwhm']:5.2f}@bw{a['bandwidth']:.2f}"
        print(f"    {r['variant']:<30s}{r['n']:>5d}{cells}{auto:>16s}"
              f"{fmt(r.get('lateral_iqr')):>9s}{fmt(r['depth'][key0]['fwhm']):>13s}")
    print("\n    序重建判选的纯度（选中真入射晶体的比例）：")
    for r in cuts + by_mult:
        print(f"      {r['variant']:<30s} n={r['n']:>4d}  纯度 {r['purity']:.1%}")
    print(f"\n    排序用位置 = {args.order_with}｜DOI = {'关闭（z 取中面）' if args.no_doi else '按配置 2.5 mm'}")

    summary = {
        "root": str(args.root), "analysis_dir": str(args.analysis_dir),
        "tree": args.tree, "id_branch": id_branch, "lateral_axis": args.lateral_axis,
        "bandwidth_scales": bws, "d2_voxel_mm": list(D2_VOXEL_MM),
        "d2_eres_at_511": D2_ERES_AT_511, "seed": args.seed,
        "n_selected": int(selected.sum()),
        "selfcheck_A_vs_ideal_max_dev_mm": dev, "selfcheck_A_passed": ok1,
        "selfcheck_quantized_vs_pipeline_max_dev_mm": dev_q,
        "selfcheck_quantization_passed": ok1b,
        "selfcheck_single_crystal_consistent": ok2,
        "multiplicity_counts": {int(v): int(c) for v, c in zip(mult_vals, mult_counts)},
        "single_crystal_fraction": float(single.mean()),
        "order_with": args.order_with, "no_doi": bool(args.no_doi),
        "purity_D": agree, "purity_E": float(hit_e.mean()),
        "purity_F": float(hit_f.mean()), "D_agrees_E_fraction": de_agree,
        "D_purity_by_multiplicity": purity_by_mult,
        "rows": rows + singles + cuts + by_mult,
    }
    (args.out / "d2_crystal_readout.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n写出 {args.out / 'd2_crystal_readout.json'}")
    return 0 if (ok1 and ok1b and ok2) else 1


if __name__ == "__main__":
    raise SystemExit(main())
