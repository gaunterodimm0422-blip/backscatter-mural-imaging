"""L3-B：D2 厚度 × 像素间距 的构型扫描 —— 从同一份已有 hits.root 得到，无需重跑 GATE。

为什么不用重跑 GATE
-------------------
ROOT 里记着 D2 内每一次相互作用的**连续**坐标、沉积能量和时间。于是：

* **减薄 D2**：把深度超过 t 的相互作用直接丢掉，就等于这块板后面没有材料——
  光子穿过去、不再沉积。事例的总沉积能量随之下降，若掉出 [400,600] keV 的
  光电峰窗，这个事例就被判掉，这正是薄板的效率代价。
  （唯一被忽略的是"被移除的那部分材料反散射回保留区"的贡献，量级很小。）
* **改像素间距**：晶体是 0.93 mm 见方、缝隙 0.07 mm，对 511 keV 光子近似连续介质，
  相互作用发生在哪里几乎与分割方式无关。所以把连续坐标按任意间距重新分箱，
  就等于换了一块不同间距的阵列。

因此这两个构型维度可以从一份文件里扫出来。**结论仍应由一次真实 GATE 复核**，
本脚本的用途是把构型空间缩小到值得跑的那一两个点。

三种位置口径
------------
    A  全事例能量质心   现状（backscatter/rootio.py 的做法）
    D  康普顿序重建     只用源位置、晶体位置、晶体能量，可实现
    E  最小 z 晶体      光子自 D2 下表面 z=50 射入，先作用的那颗通常 z 更小；可实现，
                       但依赖深度作用位置（DOI）分辨——这是 configs 里的假设，需向供应商确认

起始事例集
----------
不是最终选出的那 279 个，而是**所有通过"与 D2 无关"的判选**的事例：
符合时间窗、不同初级光子、D1 能量窗、D1 半径。与 D2 有关的三条判选
（D2 能量窗、D2 半宽、入射角）在每一种构型下重新施加。
这样薄 D2 的统计量才不会被"按 20 mm 选出来的集合"人为限制。

用法（Windows PowerShell，一整行）
----------------------------------
E:\\python3.12\\python.exe C:\\Users\\HOUWENDI\\Documents\\beisanshe\\scripts\\scan_d2_geometry.py --root "E:\\vbgate\\share\\beisanshe_v1\\gate\\output\\rod_x_depth30_5000000_seed24083101\\hits.root" --analysis-dir "C:\\Users\\HOUWENDI\\Documents\\beisanshe\\runs\\rod_x_depth30_5m_seed24083101\\canonical_response" --lateral-axis y --out "C:\\Users\\HOUWENDI\\Documents\\beisanshe\\outputs\\l3b_d2_geometry"

第二次及以后加 --cache 跳过重读 ROOT。
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
from backscatter.rootio import _event_key  # noqa: E402
from backscatter.standards import bootstrap_fwhm, estimate_fwhm_kde  # noqa: E402

BRANCHES = ("runID", "eventID", "edep", "time", "posX", "posY", "posZ")
D2_Z_FACE = 50.0          # D2 入射面全局 z（晶体占 z ∈ [50, 70]）
D2_FULL_THICKNESS = 20.0
D2_DOI_MM = 2.5           # 深度作用位置分辨（configs 假设）
D2_ERES_AT_511 = 0.12
D2_HALF_WIDTH = 25.0
D2_E_WINDOW = (400.0, 600.0)
INCIDENT_MAX_DEG = 45.0
E0_KEV = 511.0


def read_hits(root_path, tree_name, want_keys, step_size):
    import uproot
    want = np.sort(np.asarray(want_keys, dtype=np.int64))
    acc = {n: [] for n in BRANCHES}
    n_chunk = 0
    with uproot.open(root_path) as handle:
        for chunk in handle[tree_name].iterate(list(BRANCHES), library="np",
                                               step_size=step_size):
            n_chunk += 1
            key = _event_key(np.asarray(chunk["runID"]), np.asarray(chunk["eventID"]))
            m = np.isin(key, want)
            if not m.any():
                continue
            for n in acc:
                acc[n].append(np.asarray(chunk[n])[m])
    print(f"    读了 {n_chunk} 个分块")
    out = {n: np.concatenate(v) for n, v in acc.items()}
    out["key"] = _event_key(out["runID"], out["eventID"])
    return out


def _bin_idx(v, pitch):
    return np.floor(v / pitch).astype(np.int64)


def emulate(hits, thickness, pitch, rng):
    """返回 {事例键: (A位置, D位置, E位置, 总能量, 点亮晶体数)}，按事例键排序。"""
    keep = hits["posZ"] <= D2_Z_FACE + thickness
    if keep.sum() < 20:
        return None
    key = hits["key"][keep]
    e = np.maximum(hits["edep"][keep], 0.0) * 1000.0
    x, y, z = hits["posX"][keep], hits["posY"][keep], hits["posZ"][keep]
    t = hits["time"][keep]

    ix, iy = _bin_idx(x, pitch), _bin_idx(y, pitch)
    span = int(np.ceil(2 * D2_HALF_WIDTH / pitch)) + 4
    cid = (ix - ix.min()) * span + (iy - iy.min())
    comp = key * np.int64(span * span + 8) + cid
    o = np.argsort(comp, kind="stable")
    comp_s, key_s, e_s, t_s = comp[o], key[o], e[o], t[o]
    pos_s = np.column_stack([x[o], y[o], z[o]])

    cs = np.r_[0, np.flatnonzero(comp_s[1:] != comp_s[:-1]) + 1]
    c_e = np.add.reduceat(e_s, cs)
    c_pos = np.column_stack([np.add.reduceat(e_s * pos_s[:, i], cs) for i in range(3)])
    c_pos /= np.where(c_e > 0, c_e, 1.0)[:, None]
    dead = c_e <= 0
    if dead.any():
        c_pos[dead] = pos_s[cs][dead]
    c_t = np.minimum.reduceat(t_s, cs)
    c_key = key_s[cs]

    es = np.r_[0, np.flatnonzero(c_key[1:] != c_key[:-1]) + 1]
    ee = np.r_[es[1:], len(c_key)]
    n_ev = len(es)
    ev_key = c_key[es]
    total = np.add.reduceat(c_e, es)
    mult = (ee - es).astype(np.int32)

    # 读出位置：x,y 落到间距格心，z 落到 DOI 体素中心
    rq = np.column_stack([(np.floor(c_pos[:, 0] / pitch) + 0.5) * pitch,
                          (np.floor(c_pos[:, 1] / pitch) + 0.5) * pitch,
                          (np.floor((c_pos[:, 2] - D2_Z_FACE) / D2_DOI_MM) + 0.5)
                          * D2_DOI_MM + D2_Z_FACE])
    em = np.maximum(c_e + rng.normal(0.0, 1.0, len(c_e)) *
                    energy_sigma_keV(c_e, fwhm_fraction_at_511=D2_ERES_AT_511), 0.0)
    return dict(ev_key=ev_key, es=es, ee=ee, total=total, mult=mult,
                c_pos=c_pos, rq=rq, c_e=c_e, em=em)


def positions(em_out, source_by_event):
    g = em_out
    es, ee, rq, c_e, em = g["es"], g["ee"], g["rq"], g["c_e"], g["em"]
    n = len(es)
    pa = np.empty((n, 3)); pe = np.empty((n, 3)); pd = np.empty((n, 3))
    for k, (s, e) in enumerate(zip(es, ee)):
        w = c_e[s:e]
        pa[k] = (w[:, None] * rq[s:e]).sum(0) / max(w.sum(), 1e-12)
        pe[k] = rq[s + int(np.argmin(rq[s:e, 2]))]
        m = e - s
        if m == 1:
            pd[k] = rq[s]
            continue
        r, ee_m = rq[s:e], em[s:e]
        tot = max(ee_m.sum(), 1e-12)
        best, best_score = 0, np.inf
        for i in range(m):
            oth = np.ones(m, bool); oth[i] = False
            wo = ee_m[oth]
            cen = (wo[:, None] * r[oth]).sum(0) / max(wo.sum(), 1e-12)
            din, dout = r[i] - source_by_event[k], cen - r[i]
            nin, nout = np.linalg.norm(din), np.linalg.norm(dout)
            rest = tot - ee_m[i]
            if nin <= 0 or nout <= 0 or rest <= 1.0:
                continue
            ck = 2.0 - E0_KEV / rest
            if not np.isfinite(ck) or abs(ck) > 1.2:
                continue
            sc = abs(float(din @ dout / (nin * nout)) - ck)
            if sc < best_score:
                best_score, best = sc, i
        pd[k] = r[best]
    return pa, pd, pe


def auto_fwhm(v, start=1.0, stop=4.0, step=0.25):
    bw = start
    while bw <= stop + 1e-9:
        est = estimate_fwhm_kde(v, bandwidth_scale=bw)
        if est["status"] == "ok":
            return float(est["fwhm"]), float(bw)
        bw += step
    return None, None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, type=Path)
    ap.add_argument("--analysis-dir", required=True, type=Path)
    ap.add_argument("--tree", default="Hits_crystal")
    ap.add_argument("--lateral-axis", default="y", choices=("x", "y"))
    ap.add_argument("--thicknesses", default="4,6,8,10,14,20")
    ap.add_argument("--pitches", default="0.5,1.0,2.0,4.0")
    ap.add_argument("--rules", default="A,D,E")
    ap.add_argument("--step-size", default="300 MB")
    ap.add_argument("--bootstrap", type=int, default=200,
                    help="每格自举次数，给 95% 置信区间。设 0 可跳过，扫得更快")
    ap.add_argument("--cache", action="store_true")
    ap.add_argument("--seed", type=int, default=24090541)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    cache = args.out / "d2_raw_hits_cache.npz"
    thicks = [float(v) for v in args.thicknesses.split(",")]
    pitches = [float(v) for v in args.pitches.split(",")]
    rules = [r.strip().upper() for r in args.rules.split(",")]

    pairs = dict(np.load(args.analysis_dir / "pairs.npz", allow_pickle=True))
    # 只施加"与 D2 无关"的判选，D2 相关的三条留到每个构型里重新施加
    base = (np.abs(pairs["dt_ns"]) <= 10.0)
    base &= np.asarray(pairs["different_primary_photons"], bool)
    base &= (pairs["d1_energy_keV"] >= 180.0) & (pairs["d1_energy_keV"] <= 250.0)
    base &= (pairs["d1_radius_mm"] >= 20.0) & (pairs["d1_radius_mm"] <= 200.0)
    keys = np.asarray(pairs["key"], dtype=np.int64)[base]
    order_pairs = np.argsort(keys)
    keys = keys[order_pairs]
    src = np.column_stack([pairs["source_x_mm"], pairs["source_y_mm"],
                           pairs["source_z_mm"]]).astype(np.float64)[base][order_pairs]
    d1 = np.column_stack([pairs["d1_x_mm"], pairs["d1_y_mm"],
                          pairs["d1_z_mm"]]).astype(np.float64)[base][order_pairs]
    d1e = np.asarray(pairs["d1_energy_keV"], dtype=np.float64)[base][order_pairs]
    print(f"[1/4] 起始事例（仅施加与 D2 无关的判选）= {keys.size} 个"
          f"（作为对照：完整判选后是 {int(np.asarray(np.load(args.analysis_dir / 'reconstruction.npz')['selected_valid']).sum())} 个）")

    if args.cache and cache.exists():
        z = np.load(cache); hits = {k: z[k] for k in z.files}
        print(f"[2/4] 用缓存 {cache.name}，{len(hits['key'])} 条 hit")
    else:
        print(f"[2/4] 重读 {args.root.name} 的 {args.tree} ...")
        hits = read_hits(args.root, args.tree, keys, args.step_size)
        np.savez_compressed(cache, **hits)
        print(f"    {len(hits['key'])} 条 hit，缓存写入 {cache.name}")

    lat_i = 0 if args.lateral_axis == "x" else 1
    rng = np.random.default_rng(args.seed)
    rows = []
    print("[3/4] 扫描 ...")
    for t in thicks:
        for p in pitches:
            g = emulate(hits, t, p, np.random.default_rng(args.seed))
            if g is None:
                continue
            idx = np.searchsorted(keys, g["ev_key"])
            ok_key = (idx < len(keys)) & (keys[np.minimum(idx, len(keys) - 1)] == g["ev_key"])
            src_e = np.zeros((len(g["ev_key"]), 3)); src_e[ok_key] = src[idx[ok_key]]
            pa, pd, pe = positions(g, src_e)
            tot_meas = g["total"] + np.random.default_rng(args.seed + 7).normal(
                0.0, 1.0, len(g["total"])) * energy_sigma_keV(
                g["total"], fwhm_fraction_at_511=D2_ERES_AT_511)
            keep_ev = ok_key & (tot_meas >= D2_E_WINDOW[0]) & (tot_meas <= D2_E_WINDOW[1])
            for rule, pos in (("A", pa), ("D", pd), ("E", pe)):
                if rule not in rules:
                    continue
                k = keep_ev & (np.abs(pos[:, 0]) <= D2_HALF_WIDTH) & (np.abs(pos[:, 1]) <= D2_HALF_WIDTH)
                v = pos - src_e
                nrm = np.linalg.norm(v, axis=1)
                inc = np.degrees(np.arccos(np.clip(np.where(nrm > 0, v[:, 2] / np.maximum(nrm, 1e-9), 0), -1, 1)))
                k &= inc <= INCIDENT_MAX_DEG
                ii = idx[k]
                if k.sum() < 25:
                    rows.append(dict(thickness=t, pitch=p, rule=rule, n=int(k.sum()),
                                     lateral_fwhm=None, lateral_bw=None, lateral_ci=None,
                                     lateral_iqr=None,
                                     depth_fwhm=None, mean_mult=float(g["mult"][k].mean()) if k.sum() else None))
                    continue
                sol = solve_scatter_points(src[ii], d1[ii], pos[k], d1e[ii])
                good = sol.valid
                lat = sol.points_mm[good, lat_i]; dep = sol.points_mm[good, 2]
                f, bw = auto_fwhm(lat)
                fd, _ = auto_fwhm(dep)
                ci = (bootstrap_fwhm(lat, n_bootstrap=args.bootstrap, seed=24090551,
                                     bandwidth_scale=bw).get("fwhm_95_ci")
                      if (f is not None and args.bootstrap > 0) else None)
                rows.append(dict(thickness=t, pitch=p, rule=rule, n=int(good.sum()),
                                 lateral_fwhm=f, lateral_bw=bw, lateral_ci=ci,
                                 lateral_iqr=float(np.subtract(*np.percentile(lat, [75, 25]))),
                                 depth_fwhm=fd,
                                 mean_mult=float(g["mult"][k][good].mean())))
        print(f"    厚度 {t:4.1f} mm 完成", flush=True)

    ref = max([r["n"] for r in rows if r["thickness"] == max(thicks)
               and r["pitch"] == pitches[0] and r["n"]] or [1])
    print("\n[4/4] 结果（横向 FWHM 用自动升带宽的 KDE；IQR 与带宽无关，用于交叉验证）")
    for rule in rules:
        print(f"\n  ── 位置口径 {rule} " + {"A": "全事例质心（现状）", "D": "康普顿序重建",
                                           "E": "最小 z 晶体"}.get(rule, "") + " ──")
        print("    " + f"{'厚度':>6s}"
              + "".join(f"{'间距 ' + f'{p:g} mm':>28s}" for p in pitches)
              + f"{'存活事例':>12s}{'平均点亮':>10s}")
        for t in thicks:
            cells = ""
            n_here, mm = None, None
            for p in pitches:
                r = next((x for x in rows if x["thickness"] == t and x["pitch"] == p
                          and x["rule"] == rule), None)
                if r is None or r["lateral_fwhm"] is None:
                    cells += f"{'n/a':>22s}"
                else:
                    c = r.get("lateral_ci")
                    ci_s = "" if c is None else f"[{c[0]:.1f},{c[1]:.1f}]"
                    cells += f"{r['lateral_fwhm']:6.2f}{ci_s:>12s} IQR{r['lateral_iqr']:5.2f}  "
                    n_here, mm = r["n"], r["mean_mult"]
            frac = "" if n_here is None else f"{n_here:5d} ({n_here / max(ref,1):4.0%})"
            print(f"    {t:6.1f}{cells}{frac:>10s}{'' if mm is None else f'{mm:10.2f}'}")
    json.dump({"thicknesses": thicks, "pitches": pitches, "rules": rules,
               "reference_n": int(ref), "rows": rows},
              open(args.out / "d2_geometry_scan.json", "w"), ensure_ascii=False, indent=1)
    print(f"\n写出 {args.out / 'd2_geometry_scan.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
