"""扫描：空鼓厚度 x 稀释因子 x 初级粒子数 -> 检出率与边界定位精度。

底片与对比度取自事例级蒙卡（m2_mc*.npz）。稀释因子 D 把蒙卡的对比度压低，
用来覆盖"蒙卡不含多次散射、实测对比度可能只有一半甚至三分之一"这件事。
每格做 N 次泊松重抽，跑与真分析完全相同的估计量。
"""
import sys, json, time
import numpy as np
sys.path.insert(0, "/home/claude/m2/scripts")
import importlib.util
spec = importlib.util.spec_from_file_location("ae", "/home/claude/m2/scripts/analyze_mural_edge.py")
ae = importlib.util.module_from_spec(spec); spec.loader.exec_module(ae)

XE = np.arange(-36.0, 36.01, 3.0)
XC = 0.5 * (XE[1:] + XE[:-1])
N_AT_2M = 29922.0          # M1 实测：200 万初级粒子选出 29,922 个重建事例


def templates(npz, bands):
    d = np.load(npz)
    wr = d["ref_w"]; wv = d["voi_w"]
    mr = np.abs(d["ref_y"]) <= 40; mv = np.abs(d["voi_y"]) <= 40
    lam, rat = [], []
    for lo, hi in bands:
        a = mr & (d["ref_z"] >= lo) & (d["ref_z"] < hi)
        b = mv & (d["voi_z"] >= lo) & (d["voi_z"] < hi)
        hr, _ = np.histogram(d["ref_x"][a], bins=XE, weights=wr[a])
        hv, _ = np.histogram(d["voi_x"][b], bins=XE, weights=wv[b])
        hv *= (float(d["voi_eff"]) / wv.sum()) / (float(d["ref_eff"]) / wr.sum())
        lam.append(hr); rat.append(np.where(hr > 0, hv / np.maximum(hr, 1e-30), 1.0))
    lam = np.array(lam); rat = np.array(rat)
    return lam / lam.sum(), rat


def trial_set(lam_frac, rat, D, n_ev, trials, rng, edge=True):
    T, XEs = [], []
    rmap = 1.0 + D * (rat - 1.0) if edge else np.ones_like(rat)
    mu_v = lam_frac * n_ev * rmap
    mu_r = lam_frac * n_ev
    for _ in range(trials):
        nv = rng.poisson(mu_v); nr = rng.poisson(mu_r)
        f1 = ae.fit_edge(XC, nv, nr, 1.0, True)
        f0 = ae.fit_edge(XC, nv, nr, 1.0, False)
        T.append(f0["nll"] - f1["nll"]); XEs.append(f1["xe"])
    return np.array(T), np.array(XEs)


if __name__ == "__main__":
    trials = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    cases = {"10 mm": ("/home/claude/design/m2_mc.npz", [(0,8),(8,15),(15,25),(25,35),(35,45)]),
             "20 mm": ("/home/claude/design/m2_mc_t20.npz", [(0,8),(8,15),(15,25),(25,35)])}
    rng = np.random.default_rng(5150)
    out = {}
    for tag, (npz, bands) in cases.items():
        lam_frac, rat = templates(npz, bands)
        Tn, _ = trial_set(lam_frac, rat, 1.0, N_AT_2M, trials, rng, edge=False)
        thr = float(np.percentile(Tn, 99))
        print(f"\n===== 空鼓厚度 {tag}，{len(bands)} 个深度带 =====")
        print(f"零假设 T：中位 {np.median(Tn):.1f}  90% {np.percentile(Tn,90):.1f}  "
              f"95% {np.percentile(Tn,95):.1f}  99% {thr:.1f}  最大 {Tn.max():.1f}   ({trials} 次)")
        print(f"{'初级粒子':>10s}{'稀释D':>8s}{'T中位':>9s}{'检出率%':>9s}"
              f"{'x_e偏差mm':>11s}{'x_e散布mm':>11s}")
        rows = []
        for nm in (2, 4, 8):
            for D in (1.0, 0.5, 0.33):
                T, xs = trial_set(lam_frac, rat, D, N_AT_2M * nm / 2.0, trials, rng)
                det = 100.0 * np.mean(T > thr)
                bias = float(np.median(xs))
                sp = float(0.5 * (np.percentile(xs, 84) - np.percentile(xs, 16)))
                print(f"{nm:8d}M{D:8.2f}{np.median(T):9.1f}{det:9.0f}{bias:11.2f}{sp:11.2f}")
                rows.append(dict(n_million=nm, D=D, T_median=float(np.median(T)),
                                 detect_pct=det, bias_mm=bias, spread_mm=sp))
        out[tag] = dict(thr=thr, null_median=float(np.median(Tn)), rows=rows,
                        bands=[list(b) for b in bands], trials=trials)
    json.dump(out, open("/home/claude/design/m2_scan.json", "w"), ensure_ascii=False, indent=1)
