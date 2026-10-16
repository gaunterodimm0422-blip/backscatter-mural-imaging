"""新构型（h1=10、D1 外半径 100）对 M2 那套观测量的预言，用于冻结 M3 判据。

同一套 m2_predict 蒙卡，只改三个几何参数。对每个构型都算：
  1. 四个重建深度带的空鼓侧/完好侧对比度与边界过渡宽度
  2. 全局效率（用于预言计数率之比）
  3. 横向刻度系数 k 的几何预言：空鼓层内事例在 0~35 mm 四个带里的
     G = (17.5 + d_rec) / (17.5 + d_true) 的计数加权中位
"""
import os, json, importlib.util
import numpy as np
from scipy.optimize import curve_fit
from scipy.special import erf

BANDS = [(0, 8), (8, 15), (15, 25), (25, 35)]
XE = np.arange(-45, 45.1, 1.5); XC = 0.5 * (XE[1:] + XE[:-1])
SLAB = (20.0, 40.0)


def step(x, A, c, xe, w):
    return A * (1 - c * 0.5 * (1 + erf((x - xe) / (np.sqrt(2) * w))))


def analyse(tag, env):
    os.environ.update({k: str(v) for k, v in env.items()})
    os.environ["VOID_T_MM"] = "20"
    spec = importlib.util.spec_from_file_location(f"m2_{tag}", "/home/claude/design/m2_predict.py")
    m2 = importlib.util.module_from_spec(spec); spec.loader.exec_module(m2)
    ref = m2.run(False, 24091201, 500_000, 16)
    voi = m2.run(True, 24091202, 500_000, 16)
    wr = ref["w"] * (ref["eff"] / ref["w"].sum())
    wv = voi["w"] * (voi["eff"] / voi["w"].sum())
    mr = np.abs(ref["y_rec"]) <= 40; mv = np.abs(voi["y_rec"]) <= 40
    out = dict(tag=tag, h_d1=m2.H_D1, r_in=m2.R1_IN, r_out=m2.R1_OUT,
               eff_ref=ref["eff"], eff_void=voi["eff"], bands=[])
    for lo, hi in BANDS:
        a = mr & (ref["depth_rec"] >= lo) & (ref["depth_rec"] < hi)
        b = mv & (voi["depth_rec"] >= lo) & (voi["depth_rec"] < hi)
        hr, _ = np.histogram(ref["x_rec"][a], bins=XE, weights=wr[a])
        hv, _ = np.histogram(voi["x_rec"][b], bins=XE, weights=wv[b])
        hr2, _ = np.histogram(ref["x_rec"][a], bins=XE, weights=wr[a] ** 2)
        hv2, _ = np.histogram(voi["x_rec"][b], bins=XE, weights=wv[b] ** 2)
        g = (hr > 0.05 * hr.max()) & (hv > 0)
        r = hv[g] / hr[g]
        sr = r * np.sqrt(hr2[g] / hr[g] ** 2 + hv2[g] / hv[g] ** 2)
        p, _ = curve_fit(step, XC[g], r, p0=[1., .2, 0., 3.], sigma=sr, absolute_sigma=True,
                         bounds=([.5, -1, -40, .5], [2., 1, 40, 60]), maxfev=40000)
        frac = wr[a].sum() / wr[mr].sum()
        out["bands"].append(dict(lo=lo, hi=hi, frac=float(frac), contrast=float(p[1]),
                                 edge=float(p[2]), width=float(p[3])))
    # 横向刻度系数 k 的几何预言（只用参照场，空鼓层内事例）
    slab = mr & (ref["depth_true"] >= SLAB[0]) & (ref["depth_true"] <= SLAB[1])
    G = (17.5 + ref["depth_rec"]) / (17.5 + ref["depth_true"])
    num = den = 0.0
    per_band = []
    for lo, hi in BANDS:
        m = slab & (ref["depth_rec"] >= lo) & (ref["depth_rec"] < hi)
        if not m.any():
            per_band.append(None); continue
        wsum = float(wr[m].sum()); gmed = float(np.median(G[m]))
        per_band.append(dict(lo=lo, hi=hi, weight=wsum, G_median=gmed))
        num += wsum * gmed; den += wsum
    out["k_predicted"] = num / den if den > 0 else float("nan")
    out["k_per_band"] = per_band
    out["depth_bias_median_mm"] = float(np.median(ref["depth_rec"] - ref["depth_true"]))
    d = ref["depth_rec"] - ref["depth_true"]
    out["depth_spread_p16_p84_mm"] = [float(np.percentile(d, 16)), float(np.percentile(d, 84))]
    return out


if __name__ == "__main__":
    results = []
    for tag, env in (("现状 h1=17.5 R20-200", dict(H_D1_MM=17.5, R1_IN_MM=20.0, R1_OUT_MM=200.0)),
                     ("新构型 h1=10 R20-100", dict(H_D1_MM=10.0, R1_IN_MM=20.0, R1_OUT_MM=100.0))):
        r = analyse(tag, env); results.append(r)
        print(f"\n===== {tag} =====")
        print(f"  效率 参照 {r['eff_ref']:.4e}/湮没   空鼓 {r['eff_void']:.4e}/湮没")
        print(f"  深度偏差中位 {r['depth_bias_median_mm']:+.2f} mm，"
              f"16~84 分位 [{r['depth_spread_p16_p84_mm'][0]:+.1f}, {r['depth_spread_p16_p84_mm'][1]:+.1f}] mm")
        print(f"  {'深度带':>10s}{'份额%':>8s}{'对比度%':>9s}{'边界mm':>9s}{'过渡宽mm':>10s}{'G中位':>8s}")
        for b, kb in zip(r["bands"], r["k_per_band"]):
            gm = kb["G_median"] if kb else float("nan")
            print(f"{b['lo']:5d}-{b['hi']:<4d}{100*b['frac']:8.1f}{100*b['contrast']:9.2f}"
                  f"{b['edge']:9.2f}{b['width']:10.2f}{gm:8.3f}")
        print(f"  横向刻度系数 k 的几何预言 = {r['k_predicted']:.3f}")
    a, b = results
    print(f"\n计数率之比（新/旧）= {b['eff_ref']/a['eff_ref']:.3f}")
    print(f"k 预言：旧 {a['k_predicted']:.3f}  新 {b['k_predicted']:.3f}  "
          f"（旧构型实测 k = 0.797 ± 0.028）")
    json.dump(results, open("m3_predict.json", "w"), ensure_ascii=False, indent=1)
