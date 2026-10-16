# -*- coding: utf-8 -*-
"""从远控机拷回的 outputs/d2f15/ 汇总 01 文档要用的全部 15% 数字（只读 json/npz，不改任何东西）。

用法：python collect_numbers_d2f15.py <d2f15目录> <输出 json>
每个数都在输出 json 里注明出自哪个文件的哪个字段。
"""
import json, math, sys
from pathlib import Path
import numpy as np

root = Path(sys.argv[1]); out = Path(sys.argv[2])
J = lambda p: json.load(open(root / p, encoding="utf-8"))
N = {}

def edge(name):
    d = J(f"{name}/m2_edge_result.json")
    s = d["fits"]["signal"]; hx = d["head_x_mm"]
    r = dict(src=f"{name}/m2_edge_result.json",
             inputs_all_d2f15=all(v.rstrip("\\/").endswith("response_d2f15") for v in d["inputs"].values()),
             truth=d["truth_edge_world_mm"], T=s["T"], xe_world=s["xe"] + hx, xe_local=s["xe"],
             bias=s["xe"] + hx - d["truth_edge_world_mm"],
             ci68_world=[s["xe_lo"] + hx, s["xe_hi"] + hx],
             hw=(s["xe_hi"] - s["xe_lo"]) / 2, hw_lo=s["xe"] - s["xe_lo"], hw_hi=s["xe_hi"] - s["xe"],
             xe_sigma=s["xe_sigma"], w=s["w"], c_b_pct=[100 * c for c in s["c"]],
             n_selected=d["n_selected"], n_primary=d["n_primary"])
    if "consistency" in d["fits"]:
        r["cross"] = d["fits"]["consistency"]["xe"] - s["xe"]
    if "null" in d["fits"]:
        n = d["fits"]["null"]
        r["null"] = dict(T=n["T"], xe_local=n["xe"], w=n["w"], c_b_pct=[100 * c for c in n["c"]])
    if "null_split" in d["fits"]:
        r["split_T"] = d["fits"]["null_split"]["T"]
    nc = d.get("null_calibration")
    if nc:
        r["null_calibration"] = {k: v for k, v in nc.items() if k != "T"}
    r["fov_bins_by_band"] = []
    for b, m in zip(d["bands_mm"], d["band_mask"]):
        xs = [x for x, mm in zip(d["x_centres_mm"], m) if mm]
        r["fov_bins_by_band"].append([b, xs[0], xs[-1]])
    return r

for nm in ["m3_edge_result", "m3_offset", "m3_offset2", "m3_widefov",
           "m2_mural_edge", "m2_mural_edge_offset", "m2_mural_edge_offset2",
           "m2_mural_edge_offset3", "m2_widefov"]:
    N[nm] = edge(nm)

def linfit(u, y):
    u = np.asarray(u, float); y = np.asarray(y, float); n = len(u)
    Suu = ((u - u.mean()) ** 2).sum()
    k = ((u - u.mean()) * (y - y.mean())).sum() / Suu
    b = y.mean() - k * u.mean()
    res = y - (k * u + b)
    s = math.sqrt((res ** 2).sum() / (n - 2))
    return dict(k=k, sk=s / math.sqrt(Suu), b=b, sb=s * math.sqrt(1 / n + u.mean() ** 2 / Suu),
                r2=1 - (res ** 2).sum() / ((y - y.mean()) ** 2).sum(), s=s, res=res.tolist(),
                u=u.tolist(), y=y.tolist())

# M3 三点标定（全精度 x_e）
m3 = [N["m3_edge_result"], N["m3_offset"], N["m3_offset2"]]
N["m3_calib_fullprec"] = linfit([r["truth"] + 15 for r in m3], [r["xe_local"] for r in m3])
N["m3_calib_fullprec"]["src"] = "m3_edge_result / m3_offset / m3_offset2 的 fits.signal.xe（全精度）"
# M3 三点标定（npz 里两位小数的 x_e，G8 (e) 面板用的就是它）
npz_fit = []
for nm in ["imaging_demo_m3", "imaging_demo_m3x23", "imaging_demo_m3x31"]:
    z = np.load(root / nm / "imaging_demo_data.npz", allow_pickle=True)
    npz_fit.append((float(z["x_edge_truth"]) - float(z["head_x_mm"]),
                    float(z["x_edge_fit"]) - float(z["head_x_mm"])))
N["m3_calib_npz2dp"] = linfit([a for a, _ in npz_fit], [b for _, b in npz_fit])
N["m3_calib_npz2dp"]["src"] = "imaging_demo_m3 / m3x23 / m3x31 的 npz 字段 x_edge_fit（两位小数）"
k3 = N["m3_calib_fullprec"]
N["m3_crit9"] = dict(abs_k_minus_0844=abs(k3["k"] - 0.844), margin=0.09 - abs(k3["k"] - 0.844),
                     k_vs_0844_sigma=(k3["k"] - 0.844) / k3["sk"])

# M2 四点标定（main / offset / offset2 / offset3）
m2n = ["m2_mural_edge", "m2_mural_edge_offset", "m2_mural_edge_offset2", "m2_mural_edge_offset3"]
m2 = [N[x] for x in m2n]
f2 = linfit([r["truth"] + 15 for r in m2], [r["xe_local"] for r in m2])
f2["sig"] = [r["xe_sigma"] for r in m2]
f2["k_vs_1_sigma"] = (1 - f2["k"]) / f2["sk"]
f2["k_vs_083_sigma"] = (0.83 - f2["k"]) / f2["sk"]
f2["k_vs_070_sigma"] = (f2["k"] - 0.70) / f2["sk"]
off3 = N["m2_mural_edge_offset3"]
f2["fixed_shift_22_sigma_at_offset3"] = abs(off3["xe_local"] - (8 + 2.2)) / off3["xe_sigma"]
f2["minus_b_over_k"] = -f2["b"] / f2["k"]
f2["src"] = "m2_mural_edge / _offset / _offset2 / _offset3 的 fits.signal.xe 与 xe_sigma"
N["m2_calib"] = f2
N["m2_diff_offset_minus_main"] = N["m2_mural_edge_offset"]["xe_world"] - N["m2_mural_edge"]["xe_world"]

# 计数率双记账
pm3 = N["m3_edge_result"]["n_selected"]["ref_main"] / N["m3_edge_result"]["n_primary"]["ref_main"] * 1e6
pm2 = N["m2_mural_edge"]["n_selected"]["ref_main"] / N["m2_mural_edge"]["n_primary"]["ref_main"] * 1e6
om3 = 2 * math.pi * (1 - math.cos(math.radians(35))); om2 = 2 * math.pi * (1 - math.cos(math.radians(45)))
ann3 = pm3 * 1e-6 * om3 / (4 * math.pi); ann2 = pm2 * 1e-6 * om2 / (4 * math.pi)
N["rate"] = dict(per_million_m3=pm3, per_million_m2=pm2, ratio_per_primary=pm3 / pm2,
                 omega_m3=om3, omega_m2=om2, per_annih_m3=ann3, per_annih_m2=ann2,
                 ratio_per_annih=ann3 / ann2, time_factor=ann2 / ann3,
                 frozen_interval=[15565, 25942], above_upper_pct=100 * (pm3 / 25942 - 1),
                 mc_pred_ratio_per_primary=1.047, mc_underest_pct=100 * (pm3 / pm2 / 1.047 - 1),
                 src="m3_edge_result 与 m2_mural_edge 的 n_selected.ref_main / n_primary.ref_main；立体角按发射锥 35° / 45°")

# 诊断：深度误差（B3）
for nm in ["diag_m3_ref", "diag_m2_ref"]:
    d = J(f"{nm}/diag_shallow_band.json")
    e = d["depth_err"]
    N[nm] = dict(src=f"{nm}/diag_shallow_band.json", n_events=d["n_events"],
                 half_width=(e["p84"] - e["p16"]) / 2, abs_median=e["abs_median"], median=e["median"])
N["B3"] = dict(half_width_ratio=N["diag_m3_ref"]["half_width"] / N["diag_m2_ref"]["half_width"],
               abs_median_ratio=N["diag_m3_ref"]["abs_median"] / N["diag_m2_ref"]["abs_median"])

# 诊断：单次/多次拆解（B4–B6），误差按四项泊松独立计数：左右两侧的计数由 json 里的比值精确反解
def recover(nr, nv, rl, rr):
    sols = []
    for nrL in range(1, nr):
        nvL = rl * nrL
        if abs(nvL - round(nvL)) < 1e-6:
            for nrR in (nr - nrL, nr - nrL - 1, nr - nrL - 2):
                if nrR <= 0:
                    continue
                nvR = rr * nrR
                if abs(nvR - round(nvR)) < 1e-6 and abs(round(nvR) + round(nvL) - nv) <= 2:
                    sols.append((nrL, nrR, round(nvL), round(nvR)))
    return sols

for nm in ["diag_ms_m3", "diag_ms_m2"]:
    d = J(f"{nm}/diag_multiscatter.json")
    rows = []
    tot = {}
    for r in d["rows"]:
        if r["group"].startswith("全部"):
            tot[tuple(r["band"])] = r["n_ref"]
    for r in d["rows"]:
        if "c_b_pct" not in r:
            continue
        s = recover(r["n_ref"], r["n_void"], r["ratio_left"], r["ratio_right"])
        assert len(s) == 1, (nm, r["band"], r["group"], s[:3])
        nrL, nrR, nvL, nvR = s[0]
        c = 1 - (nvR / nrR) / (nvL / nrL)
        assert abs(100 * c - r["c_b_pct"]) < 1e-9
        err4 = (1 - c) * math.sqrt(1 / nvR + 1 / nrR + 1 / nvL + 1 / nrL)
        rows.append(dict(band=r["band"], group=r["group"], c_b_pct=100 * c, err4_pct=100 * err4,
                         share_of_band_ref_pct=100 * r["n_ref"] / tot[tuple(r["band"])],
                         f_void_pct_of_all=r["f_void_pct_of_all"],
                         counts=dict(nrL=nrL, nrR=nrR, nvL=nvL, nvR=nvR)))
    N[nm] = dict(src=f"{nm}/diag_multiscatter.json", xwin=d["xwin"], rows=rows)

N["lateral_scale"] = J("m2_lateral_scale/lateral_scale.json")
N["lateral_scale"]["src"] = "m2_lateral_scale/lateral_scale.json"

# F3
f3 = J("f3_thickness/f3_summary.json")
N["f3"] = dict(src="f3_thickness/f3_summary.json", summary=f3, per_t={})
for t in (10, 5, 3, 2, 1):
    N["f3"]["per_t"][t] = edge(f"f3_thick_t{t}")

json.dump(N, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=float)
print("写入", out)
