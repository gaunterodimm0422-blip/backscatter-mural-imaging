"""M2 分析：从"半空间空鼓 / 参照"定出空鼓的**横向边界**，并判五条冻结判据。

为什么观测量是横向台阶
----------------------
M1 已经证明：单位姿的深度分辨约 38 mm，10 mm 的空鼓在深度方向被彻底糊平；
而同一台设备的横向分辨是 3~4 mm。2026-09-11 的事例级蒙卡进一步给出：

  重建深度带    空鼓侧/完好侧对比度   边界过渡宽度
     0- 8 mm            8.1%              1.0 mm
     8-15              13.6%              0.5 mm
    15-25              31.4%              1.1 mm
    25-35              33.3%              2.4 mm
    35-45             -16.7%（已翻号）     6.5 mm
    45-60            -100%（透射增强）    24.0 mm
    60-120           -100%（透射增强）    15.9 mm

浅带的台阶**几乎不糊**：空鼓抽掉的是散射体本身，这是横向坐标 x 的局部效应，
重建把深度糊掉、把横向留住了（本设备横向分辨 3~4 mm、深度分辨约 38 mm，差十倍）。
深带是透射增强（出射腿斜穿空鼓层，横向跑掉 16~24 mm），边界被抹开，
所以**深带只用于"有没有"，不参与"边在哪"**，边界拟合只用 0~35 mm 的四个带。

台阶高度不用蒙卡值：蒙卡不含多次散射，M1 实测证明它高估对比度约 2 倍。
高度由数据自己测，蒙卡只提供"哪些带可用、过渡有多宽"这两个形状结论。

估计量
------
按重建深度分带 b、横向分箱 i：

    n_ref [b,i] ~ Poisson(N_r * f_bi)
    n_void[b,i] ~ Poisson(N_v * f_bi * A_b * (1 - c_b * Phi((x_i - x_e)/w)))

x_e（边界位置）与 w（过渡宽度）**所有带共用**，A_b、c_b 每带自由。
未知的底片 f_bi 用条件化消掉：对每个箱条件在 n_ref+n_void 上，

    n_void | n_tot ~ Binomial(n_tot, rho/(1+rho)),  rho = (N_v/N_r)*A_b*(1-c_b*Phi)

这是精确的，两边的统计涨落都算进去了，不需要"参照无穷统计"的近似。

检验统计量  T = -2lnL(比值在 x 上是常数) + 2lnL(有边界)。
c_b = 0 时 x_e 与 w 不可辨识，渐近 chi2 不成立，所以 T 的零假设分布由蒙卡标定。

三个档
------
  halfvoid   半空间空鼓 20 mm，边界在世界坐标 x = -15 mm（= 探测头轴心），400 万
  ref_main   本轮参照场（无空鼓），400 万，主底片
  ref_null   M1 已跑完的无空鼓参照（seed 24091012，200 万），只当零假设与交叉检查

  信号   halfvoid/ref_main     零假设 ref_null/ref_main     交叉 halfvoid/ref_null
  另有一个不依赖 ref_null 的负对照：把 ref_main 自己随机对半，A/B 互比。
  两边初级粒子数不同不要紧：二项似然里用 s = N_v/N_r 精确归一。

判据（见到 M2 数据之前冻结，见 configs/mural_edge_20260911.json 的 observables 段）
  1. 检出：halfvoid/ref_main 的 T >= T_thr
  2. 边界位置：|x_e(世界) - (-15 mm)| <= 10 mm
  3. 零假设：ref_null/ref_main 的 T < T_thr
  4. 交叉：halfvoid/ref_null 与 halfvoid/ref_main 的 x_e 相差 <= 5 mm
  5. 台阶方向：四个带的 c_b 全部 > 0
  1、2、3、5 全过才算"能定出边界"。第 4 条为交叉检查，不过要如实记录。
  已知系统偏差：拟合的 x_e 系统性偏向空鼓一侧约 +1.1 mm（erf 台阶模型与真实
  过渡形状不完全一致，不是物理偏差）。判据 2 的阈值是 ±10 mm，不受影响。

用法（Windows，注意 Python 全路径）
-----------------------------------
  E:\\python3.12\\python.exe scripts\\analyze_mural_edge.py ^
      --halfvoid runs\\mural_halfvoid_t20_L-15_4m_seed24091121\\canonical_response ^
      --ref-main runs\\mural_ref2_t20_L-15_4m_seed24091122\\canonical_response ^
      --ref-null runs\\mural_ref_t10_L-15_2m_seed24091012\\canonical_response ^
      --config   configs\\mural_edge_20260911.json ^
      --out      outputs\\m2_mural_edge

先不跑 GATE 也能验估计量本身（在合成数据上注入已知边界再拟合）：
  E:\\python3.12\\python.exe scripts\\analyze_mural_edge.py --self-test --out outputs\\m2_selftest
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import erf

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

SQ2 = float(np.sqrt(2.0))


# --------------------------------------------------------------------------
# 估计量
# --------------------------------------------------------------------------
def _phi(x, xe, w):
    return 0.5 * (1.0 + erf((np.asarray(x) - xe) / (SQ2 * max(float(w), 1e-3))))


def _phi_pdf(x, xe, w):
    u = (np.asarray(x) - xe) / max(float(w), 1e-3)
    return np.exp(-0.5 * u * u) / np.sqrt(2.0 * np.pi)


def _nll(nv, nr, rho):
    """-2lnL 的二项形式（去掉与参数无关的常数）。"""
    rho = np.maximum(rho, 1e-12)
    return float(-2.0 * np.sum(nv * np.log(rho) - (nv + nr) * np.log1p(rho)))


def _nll_grad_rho(nv, nr, rho):
    """d(-2lnL)/d(rho)，形状同 rho。"""
    rho = np.maximum(rho, 1e-12)
    return -2.0 * (nv / rho - (nv + nr) / (1.0 + rho))


def _obj_full(q, x, nv, nr, s, nb):
    """参数顺序 [xe, w, A_1..A_nb, c_1..c_nb]，返回 (值, 解析梯度)。"""
    xe, w = q[0], q[1]
    A = q[2:2 + nb]
    c = q[2 + nb:2 + 2 * nb]
    w = max(w, 1e-3)
    Phi = _phi(x, xe, w)[None, :]
    pdf = _phi_pdf(x, xe, w)[None, :]
    rho = s * A[:, None] * (1.0 - c[:, None] * Phi)
    if np.any(rho <= 0):
        return 1e12, np.zeros_like(q)
    gr = _nll_grad_rho(nv, nr, rho)
    g = np.zeros_like(q)
    g[0] = float(np.sum(gr * (s * A[:, None] * c[:, None] * pdf / w)))
    g[1] = float(np.sum(gr * (s * A[:, None] * c[:, None]
                              * (np.asarray(x)[None, :] - xe) * pdf / w ** 2)))
    g[2:2 + nb] = np.sum(gr * (rho / A[:, None]), axis=1)
    g[2 + nb:] = np.sum(gr * (-s * A[:, None] * Phi), axis=1)
    return _nll(nv, nr, rho), g


def _init_A(nv, nr, s):
    return np.array([max(nv[b].sum(), 1.0) / max(s * nr[b].sum(), 1e-9)
                     for b in range(nv.shape[0])])


def fit_edge(x, nv, nr, s=1.0, free_edge=True,
             xe_bounds=(-35.0, 35.0), w_bounds=(0.5, 30.0)):
    """nv, nr 形状 (nb, nx)。s = N_primary(void) / N_primary(ref)。"""
    nv = np.asarray(nv, float)
    nr = np.asarray(nr, float)
    nb = nv.shape[0]
    A0 = _init_A(nv, nr, s)

    if not free_edge:
        rho = np.repeat((s * A0)[:, None], nv.shape[1], axis=1)
        return dict(nll=_nll(nv, nr, rho), A=A0.tolist(), c=[0.0] * nb,
                    xe=float("nan"), w=float("nan"))

    bnds = [xe_bounds, w_bounds] + [(0.2, 5.0)] * nb + [(-0.8, 0.8)] * nb
    best = None
    for xe0 in (-18.0, -6.0, 6.0, 18.0):
        q0 = np.concatenate([[xe0, 4.0], A0, np.full(nb, 0.05)])
        r = minimize(_obj_full, q0, args=(np.asarray(x, float), nv, nr, s, nb),
                     jac=True, method="L-BFGS-B", bounds=bnds,
                     options=dict(maxiter=3000, ftol=1e-14, gtol=1e-10))
        if best is None or r.fun < best.fun:
            best = r
    xe, w = best.x[0], best.x[1]
    A = best.x[2:2 + nb]
    c = best.x[2 + nb:]
    return dict(nll=float(best.fun), A=A.tolist(), c=c.tolist(),
                xe=float(xe), w=float(w))


def _obj_fixed_xe(q, xe, x, nv, nr, s, nb):
    full = np.concatenate([[xe], q])
    val, g = _obj_full(full, x, nv, nr, s, nb)
    return val, g[1:]


def _profile_curve(xx, nv, nr, s, fit, grid):
    nb = nv.shape[0]
    q0 = np.concatenate([[fit["w"]], fit["A"], fit["c"]])
    bnds = [(0.5, 30.0)] + [(0.2, 5.0)] * nb + [(-0.8, 0.8)] * nb
    vals = []
    for g in grid:
        r = minimize(_obj_fixed_xe, q0, args=(g, xx, nv, nr, s, nb), jac=True,
                     method="L-BFGS-B", bounds=bnds,
                     options=dict(maxiter=2000, ftol=1e-14, gtol=1e-10))
        vals.append(r.fun)
    return np.asarray(vals)


def _cross(grid, d, level=1.0):
    """d 是 Delta(-2lnL) 曲线，返回它两侧跨过 level 的位置（线性内插）。"""
    i0 = int(np.argmin(d))
    lo = hi = float("nan")
    for j in range(i0, 0, -1):
        if d[j - 1] >= level >= d[j]:
            lo = grid[j - 1] + (grid[j] - grid[j - 1]) * (d[j - 1] - level) / (d[j - 1] - d[j])
            break
    for j in range(i0, len(grid) - 1):
        if d[j] <= level <= d[j + 1]:
            hi = grid[j] + (grid[j + 1] - grid[j]) * (level - d[j]) / (d[j + 1] - d[j])
            break
    return lo, hi


def profile_xe(x, nv, nr, s, fit, half_range=20.0, n=41):
    """扫 x_e 的 profile likelihood，返回 Delta(-2lnL)=1 的区间与曲线。

    两遍：先粗扫定出量级，再按量级细扫，交点用线性内插，避免被网格步长量化。
    """
    nv = np.asarray(nv, float); nr = np.asarray(nr, float)
    xx = np.asarray(x, float)
    if not np.isfinite(fit.get("xe", np.nan)):
        return float("nan"), float("nan"), [], []
    grid = np.unique(np.clip(fit["xe"] + np.linspace(-half_range, half_range, n),
                             -35.0, 35.0))
    d = _profile_curve(xx, nv, nr, s, fit, grid); d = d - d.min()
    lo, hi = _cross(grid, d)
    if np.isfinite(lo) and np.isfinite(hi):
        hw = max(0.5 * (hi - lo), 0.05)
        if hw < 4.0 * (grid[1] - grid[0]):          # 太窄，网格分辨不出来，细扫一遍
            c = 0.5 * (lo + hi)
            grid = np.unique(np.clip(np.linspace(c - 4.0 * hw, c + 4.0 * hw, 41),
                                     -35.0, 35.0))
            d = _profile_curve(xx, nv, nr, s, fit, grid); d = d - d.min()
            lo, hi = _cross(grid, d)
    return float(lo), float(hi), grid.tolist(), d.tolist()


def compare(x, nv, nr, s, label, want_profile=True):
    f1 = fit_edge(x, nv, nr, s=s, free_edge=True)
    f0 = fit_edge(x, nv, nr, s=s, free_edge=False)
    out = dict(label=label, T=f0["nll"] - f1["nll"], s=s,
               n_void=int(np.sum(nv)), n_ref=int(np.sum(nr)), **f1)
    if want_profile:
        lo, hi, grid, curve = profile_xe(x, nv, nr, s, f1)
        out.update(xe_lo=lo, xe_hi=hi,
                   xe_sigma=(float("nan") if not np.isfinite(lo) else 0.5 * (hi - lo)),
                   profile_grid=grid, profile_curve=curve)
    return out


# --------------------------------------------------------------------------
# 数据装载
# --------------------------------------------------------------------------
def load_run(analysis_dir: Path, config, y_half_width_mm: float):
    """返回 (x_head_mm, depth_mm, n_primary, n_selected, head_x_mm)。"""
    from backscatter.pose_selection import build_selection_pose_aware

    analysis_dir = Path(analysis_dir)
    pairs = dict(np.load(analysis_dir / "pairs.npz", allow_pickle=True))
    recon = dict(np.load(analysis_dir / "reconstruction.npz", allow_pickle=True))
    summary_path = analysis_dir / "summary.json"
    if not summary_path.exists():
        summary_path = analysis_dir.parent / "summary.json"
    n_primary = None
    if summary_path.exists():
        s = json.loads(summary_path.read_text(encoding="utf-8"))
        n_primary = (s.get("gate_metadata") or {}).get("primary_event_count")
    selected, _chain, _diag = build_selection_pose_aware(pairs, config)
    valid = np.asarray(recon["valid"], dtype=bool)
    keep = selected & valid
    P = np.column_stack([recon["x_mm"], recon["y_mm"], recon["z_mm"]]).astype(np.float64)
    head_x = np.asarray(pairs["source_x_mm"], dtype=np.float64)
    if np.isfinite(y_half_width_mm):
        keep &= np.abs(P[:, 1] - np.asarray(pairs["source_y_mm"], float)) <= y_half_width_mm
    hx = head_x[keep]
    if hx.size and float(np.ptp(hx)) > 1e-6:
        raise SystemExit(f"{analysis_dir}: 源横向坐标不是常数（ptp={np.ptp(hx):.3g} mm），"
                         "说明这不是单一位姿的跑，本脚本不支持。")
    return (P[keep, 0] - hx, -P[keep, 2],
            (int(n_primary) if n_primary else None), int(keep.sum()),
            (float(hx[0]) if hx.size else float("nan")))


def histogram(x_head, depth, bands, xedges):
    h = np.zeros((len(bands), len(xedges) - 1), dtype=np.int64)
    for b, (lo, hi) in enumerate(bands):
        m = (depth >= lo) & (depth < hi)
        h[b], _ = np.histogram(x_head[m], bins=xedges)
    return h


# --------------------------------------------------------------------------
# 零假设标定
# --------------------------------------------------------------------------
def calibrate_null(x, shape, n_a, n_b, s, trials, seed=20260911):
    """从共同底片 shape（(nb,nx) 的相对份额）抽两份无边界数据，给出 T 的分布。"""
    rng = np.random.default_rng(seed)
    f = shape / shape.sum()
    Ts = []
    for _ in range(trials):
        na = rng.poisson(f * n_a)
        nb = rng.poisson(f * n_b)
        f1 = fit_edge(x, na, nb, s=s, free_edge=True)
        f0 = fit_edge(x, na, nb, s=s, free_edge=False)
        Ts.append(f0["nll"] - f1["nll"])
    return np.asarray(Ts)


# --------------------------------------------------------------------------
# 自检：在合成数据上注入已知边界
# --------------------------------------------------------------------------
def self_test(out_dir: Path, trials=60, seed=424242):
    """用一个与实测量级相符的底片，注入已知边界，检查偏差与误差棒。"""
    rng = np.random.default_rng(seed)
    xedges = np.arange(-36.0, 36.01, 3.0)
    xc = 0.5 * (xedges[1:] + xedges[:-1])
    bands = [(0, 8), (8, 15), (15, 25), (25, 35)]
    # 底片：各带份额取自蒙卡；横向形状取高斯，宽度随深度加大（照射锥）
    frac = np.array([0.228, 0.192, 0.196, 0.116])
    width = np.array([11.0, 13.0, 16.0, 20.0])
    shape = np.array([f * np.exp(-0.5 * (xc / w) ** 2) for f, w in zip(frac, width)])
    shape /= shape.sum()
    n_tot = 2 * 29922          # 400 万初级粒子，按 M1 实测的每百万产额外推
    results = {}
    C_MC = np.array([0.081, 0.136, 0.314, 0.333])       # 20 mm 空鼓，四个带的蒙卡对比度
    for D, cvec in (("蒙卡原值 D=1.00", C_MC), ("稀释一半 D=0.50", 0.50 * C_MC),
                    ("稀释到三分之一 D=0.33", 0.33 * C_MC)):
        xs, Ts, sig = [], [], []
        for _ in range(trials):
            step = _phi(xc, 0.0, 3.0)
            mu_v = shape * n_tot * (1.0 - cvec[:, None] * step[None, :])
            nv = rng.poisson(mu_v)
            nr = rng.poisson(shape * n_tot)
            r = compare(xc, nv, nr, 1.0, D, want_profile=True)
            xs.append(r["xe"]); Ts.append(r["T"]); sig.append(r["xe_sigma"])
        xs = np.asarray(xs); Ts = np.asarray(Ts); sig = np.asarray(sig)
        results[D] = dict(
            bias_mm=float(np.median(xs)),
            spread_mm=float(0.5 * (np.percentile(xs, 84) - np.percentile(xs, 16))),
            median_reported_sigma_mm=float(np.nanmedian(sig)),
            T_median=float(np.median(Ts)), T_p10=float(np.percentile(Ts, 10)),
            trials=int(trials))
    Tn = []
    for _ in range(trials):
        nv = rng.poisson(shape * n_tot); nr = rng.poisson(shape * n_tot)
        f1 = fit_edge(xc, nv, nr, 1.0, True); f0 = fit_edge(xc, nv, nr, 1.0, False)
        Tn.append(f0["nll"] - f1["nll"])
    Tn = np.asarray(Tn)
    results["零假设"] = dict(T_median=float(np.median(Tn)),
                             T_p90=float(np.percentile(Tn, 90)),
                             T_p95=float(np.percentile(Tn, 95)),
                             T_p99=float(np.percentile(Tn, 99)),
                             T_max=float(Tn.max()), trials=int(trials))
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "self_test.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
    print("\n估计量自检（合成数据，真值边界 x_e = 0）")
    print(f"{'情形':>22s}{'偏差mm':>9s}{'实际散布mm':>12s}{'报告误差mm':>12s}"
          f"{'T中位':>9s}{'T的10%分位':>12s}")
    for k, v in results.items():
        if k == "零假设":
            continue
        print(f"{k:>22s}{v['bias_mm']:9.2f}{v['spread_mm']:12.2f}"
              f"{v['median_reported_sigma_mm']:12.2f}{v['T_median']:9.1f}{v['T_p10']:12.1f}")
    z = results["零假设"]
    print(f"{'零假设（无边界）':>22s}{'-':>9s}{'-':>12s}{'-':>12s}"
          f"{z['T_median']:9.1f}{'90%:'+format(z['T_p90'],'.1f'):>12s}")
    print(f"{'':>22s}{'':>9s}{'':>12s}{'':>12s}"
          f"{'95%:'+format(z['T_p95'],'.1f'):>9s}{'99%:'+format(z['T_p99'],'.1f'):>12s}"
          f"  最大 {z['T_max']:.1f}")
    print(f"\n结果写入 {out_dir/'self_test.json'}")
    return results


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="M2 壁画空鼓横向边界定位")
    ap.add_argument("--halfvoid")
    ap.add_argument("--ref-main", dest="ref_main")
    ap.add_argument("--ref-null", dest="ref_null")
    ap.add_argument("--config"); ap.add_argument("--out", required=True)
    ap.add_argument("--null-trials", type=int, default=200,
                    help="零假设标定的抽样次数，0 表示跳过（默认 200，约几分钟）")
    ap.add_argument("--self-test", action="store_true",
                    help="只做估计量自检（合成数据），不需要任何 GATE 结果")
    ap.add_argument("--self-test-trials", type=int, default=60)
    args = ap.parse_args()
    out_dir = Path(args.out)

    if args.self_test:
        self_test(out_dir, trials=args.self_test_trials)
        return

    for need in ("halfvoid", "ref_main", "ref_null", "config"):
        if getattr(args, need) is None:
            raise SystemExit(f"缺少 --{need.replace(chr(95), chr(45))}")

    from backscatter.config import load_config
    config = load_config(Path(args.config))
    obs = config["observables"]
    bands = [tuple(b) for b in obs["depth_bands_mm"]]
    xb = obs["x_bin_edges_head_frame_mm"]
    xedges = np.arange(xb["start"], xb["stop"] + 1e-9, xb["step"])
    xc = 0.5 * (xedges[1:] + xedges[:-1])
    yhw = float(obs["y_half_width_mm"])
    min_ref = int(obs["min_ref_counts_per_bin"])
    truth_edge_world = float(config["phantom"]["void"]["edge_x_mm"])

    runs = {}
    print("=" * 74)
    print("M2：壁画空鼓横向边界定位")
    print("=" * 74)
    print(f"\n{'档':>10s}{'目录':>52s}")
    for name, path in (("halfvoid", args.halfvoid), ("ref_main", args.ref_main),
                       ("ref_null", args.ref_null)):
        x_head, depth, npri, nsel, hx = load_run(Path(path), config, yhw)
        runs[name] = dict(x=x_head, d=depth, npri=npri, nsel=nsel, hx=hx)
        print(f"{name:>10s}{str(path)[-52:]:>52s}")
    print(f"\n{'档':>10s}{'初级粒子':>12s}{'选出事例':>10s}{'探测头x/mm':>12s}"
          f"{'每百万选出':>12s}")
    for name in ("halfvoid", "ref_main", "ref_null"):
        r = runs[name]
        per = (r["nsel"] / r["npri"] * 1e6) if r["npri"] else float("nan")
        print(f"{name:>10s}{(r['npri'] if r['npri'] else -1):12d}{r['nsel']:10d}"
              f"{r['hx']:12.2f}{per:12.1f}")

    hxs = {n: runs[n]["hx"] for n in runs}
    if max(hxs.values()) - min(hxs.values()) > 1e-6:
        raise SystemExit(f"三个档的位姿不一致：{hxs}。本分析要求同一位姿。")
    head_x = float(runs["ref_main"]["hx"])

    hist = {n: histogram(runs[n]["x"], runs[n]["d"], bands, xedges) for n in runs}
    # 掩膜逐带独立：浅带的照射锥窄、横向足迹小，若全带取交集会被它把窗口拖窄，
    # 于是远离边界处的比值就看不到了。做法是保留各带自己够统计的箱，其余箱
    # 在三个档里一起置零——二项似然里 n_void = n_ref = 0 的箱贡献恰好为 0，
    # 等价于把它排除，且不必让各带的 x 网格变成不等长。
    band_good = hist["ref_main"] >= min_ref
    good = band_good.any(axis=0)
    kept = int(good.sum())
    print(f"\n横向分箱：{xb['start']:.0f} ~ {xb['stop']:.0f} mm，步长 {xb['step']:.0f} mm，"
          f"共 {len(xc)} 箱；每箱 ref_main 计数 >= {min_ref} 的带保留该箱，"
          f"并集 {kept} 箱：[{xc[good].min():.1f}, {xc[good].max():.1f}] mm（探测头坐标）")
    for b, (lo, hi) in enumerate(bands):
        m = band_good[b]
        rng = (f"[{xc[m].min():.1f}, {xc[m].max():.1f}]" if m.any() else "无")
        print(f"    {lo:>3d}-{hi:<3d} mm 带：{int(m.sum()):2d} 箱  {rng}")
    if kept < 8:
        raise SystemExit("可用横向箱少于 8，统计不足以拟合台阶。请降低 min_ref 或加大分箱。")
    xg = xc[good]
    bg = band_good[:, good]
    H = {n: np.where(bg, hist[n][:, good], 0) for n in hist}

    print(f"\n各深度带的事例数与比值（比值 = 空鼓场/参照场，已按初级粒子数归一）")
    print(f"{'深度带mm':>10s}{'halfvoid':>10s}{'ref_main':>10s}{'ref_null':>10s}"
          f"{'左半比值':>10s}{'右半比值':>10s}{'左-右':>9s}")
    edge_head = truth_edge_world - head_x        # 真值边界换算到探测头坐标
    left = xg < edge_head
    s_hr = (runs["halfvoid"]["npri"] / runs["ref_main"]["npri"]) if runs["ref_main"]["npri"] else 1.0
    s_2r = (runs["ref_null"]["npri"] / runs["ref_main"]["npri"]) if runs["ref_main"]["npri"] else 1.0
    for b, (lo, hi) in enumerate(bands):
        nv, n1, n2 = H["halfvoid"][b], H["ref_main"][b], H["ref_null"][b]
        vb = bg[b]
        rl = nv[left & vb].sum() / max(s_hr * n1[left & vb].sum(), 1e-9)
        rr = nv[(~left) & vb].sum() / max(s_hr * n1[(~left) & vb].sum(), 1e-9)
        print(f"{lo:4d}-{hi:<5d}{nv.sum():10d}{n1.sum():10d}{n2.sum():10d}"
              f"{rl:10.4f}{rr:10.4f}{rl-rr:9.4f}")

    t0 = time.time()
    res = {}
    res["signal"] = compare(xg, H["halfvoid"], H["ref_main"], s_hr, "halfvoid/ref_main")
    res["null"] = compare(xg, H["ref_null"], H["ref_main"], s_2r, "ref_null/ref_main")
    res["consistency"] = compare(xg, H["halfvoid"], H["ref_null"],
                                 (runs["halfvoid"]["npri"] / runs["ref_null"]["npri"])
                                 if runs["ref_null"]["npri"] else 1.0, "halfvoid/ref_null")
    rng = np.random.default_rng(20260911)
    m = rng.random(runs["ref_main"]["x"].size) < 0.5
    hA = np.where(bg, histogram(runs["ref_main"]["x"][m], runs["ref_main"]["d"][m],
                                bands, xedges)[:, good], 0)
    hB = np.where(bg, histogram(runs["ref_main"]["x"][~m], runs["ref_main"]["d"][~m],
                                bands, xedges)[:, good], 0)
    res["null_split"] = compare(xg, hA, hB, 1.0, "ref_main 随机对半 A/B")
    print(f"\n（拟合用时 {time.time()-t0:.0f} s）")

    T_thr = obs.get("t_threshold")
    null_cal = None
    if args.null_trials > 0:
        print(f"\n零假设标定：从 ref_main+ref_null 的共同底片抽 {args.null_trials} 对无边界数据 ...")
        t1 = time.time()
        shape = (H["ref_main"] + H["ref_null"]).astype(float)
        Ts = calibrate_null(xg, shape, H["ref_null"].sum(), H["ref_main"].sum(), s_2r,
                            args.null_trials)
        null_cal = dict(trials=int(args.null_trials), T=Ts.tolist(),
                        p50=float(np.percentile(Ts, 50)), p90=float(np.percentile(Ts, 90)),
                        p95=float(np.percentile(Ts, 95)), p99=float(np.percentile(Ts, 99)),
                        max=float(Ts.max()))
        print(f"  用时 {time.time()-t1:.0f} s   T 分位数  中位 {null_cal['p50']:.1f}  "
              f"90% {null_cal['p90']:.1f}  95% {null_cal['p95']:.1f}  "
              f"99% {null_cal['p99']:.1f}  最大 {null_cal['max']:.1f}")
        if T_thr is None:
            T_thr = null_cal["p99"]
            print(f"  配置里没有冻结阈值，采用本次标定的 99% 分位 T_thr = {T_thr:.1f}")
        else:
            print(f"  冻结阈值 T_thr = {T_thr:.1f}（配置里写死的，不随本次标定改）")
    if T_thr is None:
        raise SystemExit("配置里没有 t_threshold，且 --null-trials 为 0，无法判据。")

    print(f"\n{'比较':>18s}{'T':>9s}{'x_e(头坐标)':>13s}{'x_e(世界)':>11s}"
          f"{'68%区间(世界)':>18s}{'w/mm':>8s}")
    for key in ("signal", "null", "consistency", "null_split"):
        r = res[key]
        xw = r["xe"] + head_x
        lo = r.get("xe_lo", float("nan")) + head_x
        hi = r.get("xe_hi", float("nan")) + head_x
        itv = (f"[{lo:.1f}, {hi:.1f}]" if np.isfinite(lo) and np.isfinite(hi)
               else "无约束（无台阶）")
        print(f"{r['label']:>18s}{r['T']:9.1f}{r['xe']:13.2f}{xw:11.2f}"
              f"{itv:>18s}{r['w']:8.2f}")

    print(f"\n各带台阶高度 c_b（正 = 空鼓侧计数更少）")
    print(f"{'深度带mm':>10s}{'信号 c_b%':>12s}{'零假设 c_b%':>14s}{'交叉 c_b%':>12s}")
    for b, (lo, hi) in enumerate(bands):
        print(f"{lo:4d}-{hi:<5d}{100*res['signal']['c'][b]:12.2f}"
              f"{100*res['null']['c'][b]:14.2f}{100*res['consistency']['c'][b]:12.2f}")

    xe_world = res["signal"]["xe"] + head_x
    c_sig = np.asarray(res["signal"]["c"])
    checks = [
        ("1 检出：halfvoid/ref_main 的 T >= T_thr",
         res["signal"]["T"] >= T_thr,
         f"T = {res['signal']['T']:.1f}   T_thr = {T_thr:.1f}"),
        ("2 边界位置：|x_e(世界) - 真值| <= 10 mm",
         abs(xe_world - truth_edge_world) <= 10.0,
         f"x_e = {xe_world:.2f} mm   真值 {truth_edge_world:.1f} mm   "
         f"差 {xe_world-truth_edge_world:+.2f} mm"),
        ("3 零假设：ref_null/ref_main 的 T < T_thr",
         res["null"]["T"] < T_thr,
         f"T(零假设) = {res['null']['T']:.1f}   T_thr = {T_thr:.1f}   "
         f"另：ref_main 对半 T = {res['null_split']['T']:.1f}"),
        ("4 交叉：halfvoid/ref_null 与 halfvoid/ref_main 的 x_e 相差 <= 5 mm",
         abs(res["consistency"]["xe"] - res["signal"]["xe"]) <= 5.0,
         f"相差 {res['consistency']['xe']-res['signal']['xe']:+.2f} mm"),
        ("5 台阶方向：四个带的 c_b 全部 > 0",
         int(np.sum(c_sig > 0)) == len(bands),
         f"c_b > 0 的带数 = {int(np.sum(c_sig>0))} / {len(bands)}   "
         f"c_b(%) = {np.round(100*c_sig,2).tolist()}"),
    ]
    print("\n" + "=" * 74)
    print("冻结判据")
    print("=" * 74)
    for name, ok, detail in checks:
        print(f"  [{'通过' if ok else '不通过'}] {name}\n           {detail}")
    main_pass = all(ok for (n, ok, _) in checks if not n.startswith("4"))
    print(f"\n主判据（1、2、3、5）：{'全部通过' if main_pass else '有不通过项'}")
    print(f"交叉检查判据（4）：{'通过' if checks[3][1] else '不通过'}")

    out_dir.mkdir(parents=True, exist_ok=True)
    payload = dict(
        generated=time.strftime("%Y-%m-%dT%H:%M:%S"),
        config=str(Path(args.config).resolve()),
        inputs={k: str(Path(v).resolve())
                for k, v in (("halfvoid", args.halfvoid), ("ref_main", args.ref_main),
                             ("ref_null", args.ref_null))},
        head_x_mm=head_x, truth_edge_world_mm=truth_edge_world,
        bands_mm=[list(b) for b in bands], x_centres_mm=xg.tolist(),
        n_primary={k: runs[k]["npri"] for k in runs},
        n_selected={k: runs[k]["nsel"] for k in runs},
        hist={k: H[k].tolist() for k in H},
        band_mask=bg.tolist(),
        fits=res, t_threshold=float(T_thr), null_calibration=null_cal,
        criteria=[dict(name=n, passed=bool(o), detail=d) for n, o, d in checks],
        main_pass=bool(main_pass),
    )
    (out_dir / "m2_edge_result.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n结果写入 {out_dir/'m2_edge_result.json'}")


if __name__ == "__main__":
    main()
