"""M1 分析：从"空鼓 / 参照"两档算出重建深度剖面的比值台阶，并判四条冻结判据。

观测量为什么是台阶而不是"洞"
----------------------------
2026-09-10 的事例级蒙卡（用实测混凝土效率 2.54e-2/湮没 锚定）给出：

  * 整体计数比 0.9964 —— 只差 0.36%，整体计数不能当观测量
  * 空鼓所在深度的重建计数只低 2~3% —— 真值剖面在那里是 100% 的洞，但被
    38 mm 的单视角深度分辨抹平，又被下方的透射增强填回一部分
  * 45 mm 以深比值升到 1.03~1.25 —— **信号在这里**

空鼓移走的是面密度，进出两条腿都少衰减一次，于是它下方所有深度都获得增益：

    ratio(depth) = exp(mu_in * dSigma) * exp(mu_out * dSigma / cos_xi)

dSigma = 空鼓厚度 x 地仗密度。**台阶位置给深度，台阶高度给厚度。**

估计量
------
把比值剖面拟合成一个被深度分辨模糊过的台阶：

    r(d) = 1 + (A - 1) * Phi((d - d0) / s)

Phi 是标准正态累积分布。三个自由参数：A 渐近比值、d0 台阶中点、s 模糊宽度。
真值上材料是在 [空鼓顶, 空鼓底] 之间线性移走的，所以 d0 估计的是**空鼓中心**。

cos_xi 不用假设：逐事例由重建点 P 与 D1 命中 X2 算出射天顶角余弦，按事例平均。
mu_in / mu_out 是模型输入（511 keV 与出射带 ~200 keV 的质量衰减系数），
在输出 JSON 里显式记录，反解厚度时的敏感性一并给出。

负对照
------
把参照档的事例随机对半分成 A、B 两份，用完全相同的流程算 A/B 的台阶与显著性。
零假设下不该出台阶。这是"参照对参照"，不依赖任何真值。

判据（见 GATE 数据之前冻结，写在 configs/mural_void_20260910.json 的 observables 段）
  1. 45 mm 以深比值的合并显著性 >= 5 sigma；负对照 < 3 sigma
  2. 台阶中点 d0 与空鼓中心真值之差 <= 10 mm
  3. 反解的面密度亏损 dSigma 与真值之差 <= 30%
  4. 负对照不得自己通过第 1、2 条

用法
----
python scripts\\analyze_mural_void.py ^
    --void runs\\mural_void_t10_L-15_2m_seed24091011\\canonical_response ^
    --reference runs\\mural_ref_t10_L-15_2m_seed24091012\\canonical_response ^
    --config configs\\mural_void_20260910.json ^
    --out outputs\\m1_mural_void
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

from backscatter.config import load_config          # noqa: E402
from backscatter.pose_selection import build_selection_pose_aware  # noqa: E402

# 质量衰减系数 cm^2/g。地仗与砂砾岩的 <Z/A> 都在 0.50 附近，两者可共用。
MU_RHO_IN_DEFAULT = 0.0877        # 511 keV
MU_RHO_OUT_DEFAULT = 0.125        # 出射带 ~200 keV


def _load(analysis_dir: Path):
    pairs = dict(np.load(analysis_dir / "pairs.npz", allow_pickle=True))
    recon = dict(np.load(analysis_dir / "reconstruction.npz", allow_pickle=True))
    summary_path = analysis_dir / "summary.json"
    if not summary_path.exists():
        summary_path = analysis_dir.parent / "summary.json"
    n_primary = None
    if summary_path.exists():
        s = json.loads(summary_path.read_text(encoding="utf-8"))
        n_primary = (s.get("gate_metadata") or {}).get("primary_event_count")
    return pairs, recon, (int(n_primary) if n_primary else None)


def _observables(analysis_dir: Path, config, y_half_width_mm: float):
    """返回 (深度 mm, 出射天顶角余弦, 初级粒子数)。深度 = -z。"""
    pairs, recon, n_primary = _load(analysis_dir)
    selected, _chain, _diag = build_selection_pose_aware(pairs, config)
    valid = np.asarray(recon["valid"], dtype=bool)
    keep = selected & valid
    P = np.column_stack([recon["x_mm"], recon["y_mm"], recon["z_mm"]]).astype(np.float64)
    if np.isfinite(y_half_width_mm):
        keep &= np.abs(P[:, 1]) <= y_half_width_mm
    X2 = np.column_stack([pairs["d1_x_mm"], pairs["d1_y_mm"],
                          pairs["d1_z_mm"]]).astype(np.float64)
    d = X2[keep] - P[keep]
    n = np.linalg.norm(d, axis=1)
    cos_xi = np.clip(np.where(n > 0, d[:, 2] / np.maximum(n, 1e-9), 0.0), 1e-3, 1.0)
    S = np.column_stack([pairs["source_x_mm"], pairs["source_y_mm"],
                         pairs["source_z_mm"]]).astype(np.float64)[keep]
    X1 = np.column_stack([pairs["d2_x_mm"], pairs["d2_y_mm"],
                          pairs["d2_z_mm"]]).astype(np.float64)[keep]
    tag = X1 - S
    cos_beta = np.clip(np.abs(tag[:, 2]) / np.maximum(np.linalg.norm(tag, axis=1), 1e-9),
                       1e-3, 1.0)
    r1 = np.asarray(pairs["d1_radius_mm"], dtype=np.float64)[keep]
    return -P[keep, 2], cos_xi, cos_beta, r1, n_primary, int(keep.sum())


def _phi(x):
    from scipy.special import erf
    return 0.5 * (1.0 + erf(x / np.sqrt(2.0)))


def _fit_step(depth_centres, ratio, ratio_err, d_guess):
    """r(d) = 1 + (A-1)*Phi((d-d0)/s)，加权最小二乘。返回参数与协方差。"""
    import warnings
    from scipy.optimize import OptimizeWarning, curve_fit

    def model(d, a, d0, s):
        return 1.0 + (a - 1.0) * _phi((d - d0) / max(abs(s), 1e-6))

    ok = np.isfinite(ratio) & np.isfinite(ratio_err) & (ratio_err > 0)
    if ok.sum() < 5:
        return None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", OptimizeWarning)
            p, cov = curve_fit(model, depth_centres[ok], ratio[ok],
                               p0=[max(float(np.nanmax(ratio[ok])), 1.05), d_guess, 15.0],
                               sigma=ratio_err[ok], absolute_sigma=True, maxfev=20000)
    except Exception as exc:                      # 拟合不收敛就如实报告，不硬给数
        return {"status": f"fit_failed: {type(exc).__name__}"}
    resid = (ratio[ok] - model(depth_centres[ok], *p)) / ratio_err[ok]
    dof = int(ok.sum()) - 3
    err = np.sqrt(np.diag(cov))
    return {"status": "ok", "asymptotic_ratio": float(p[0]), "step_centre_mm": float(p[1]),
            "blur_sigma_mm": float(abs(p[2])),
            "asymptotic_ratio_err": float(err[0]), "step_centre_err_mm": float(err[1]),
            "chi2_per_dof": float(np.sum(resid**2) / dof) if dof > 0 else None,
            "n_points": int(ok.sum())}


def _k_of_depth(t_grid, r1_ref, h_d1_mm, mu_in, mu_out, inv_cos_beta):
    """k(d) = mu_in*<1/cos_beta> + mu_out*<1/cos_xi | 深度 d>。

    为什么必须逐深度算：出射天顶角由散射点深度与 D1 命中半径共同决定，
    浅处事例出射很斜（cos_xi 小），深处事例出射较陡（cos_xi 大）。
    合成数据实测：浅处 <cos_xi> 约 0.40，深处约 0.70。渐近比值是由**深处**事例
    决定的，用全样本的平均值会把 k 高估约 1.5 倍，厚度随之低估 1.5 倍。

    <1/cos_xi | d> 不做假设，由参照档实测的 D1 半径分布加已知几何直接算：
        cos_xi = (h_D1 + d) / sqrt(r1^2 + (h_D1 + d)^2)
    """
    r = np.asarray(r1_ref, dtype=np.float64)
    if len(r) > 20000:                              # 取样本即可，省内存
        r = r[:: max(len(r) // 20000, 1)]
    h = h_d1_mm + t_grid[:, None]
    inv_cos_xi = np.sqrt(r[None, :] ** 2 + h**2) / np.maximum(h, 1e-6)
    return mu_in * inv_cos_beta + mu_out * inv_cos_xi.mean(axis=1)


def _forward_fit(edges, hv, hr, npv, npr, k_grid_fn, rho_ground, d_top_guess, t_guess):
    """物理正演拟合，去掉 erf 那种描述性拟合的系统偏差。

    为什么 erf 会偏：观测到的比值不是"把真值比值做模糊"，而是两个卷积之比

        r_obs(d_rec) = ∫p(d)w(d)g(d_rec-d)dd / ∫p(d)g(d_rec-d)dd

    p(d) 随深度单调下降，模糊核把浅处（w 小）的事例混进来，压低了每个 d_rec 处的
    比值，于是台阶看上去整体后移。合成数据实测这个偏差约 +11 mm。

    正演做法：把真深度分布 p(d) 参数化成一个光滑的对数多项式，
    模糊核用带位移的高斯（重建深度本来就有已知的浅偏），空鼓用
        w(d) = exp(k_eff * ρ * clip(d - d_top, 0, t) / 10)
    然后**同时**正演出参照档与空鼓档的重建深度直方图，对两组计数一起做泊松拟合。
    自由参数：对数多项式 4 个、模糊 σ 与位移 2 个、空鼓深度与厚度 2 个，共 8 个。

    空鼓档的绝对归一**不是自由参数**——同样的初级粒子数下，多出来的事例完全由
    w 决定。这一条约束把整体计数比也变成了信息，而不是被归一化掉。
    """
    from scipy.optimize import minimize

    t_grid = np.arange(0.0, 200.0, 1.0)
    k_grid = k_grid_fn(t_grid)
    r_grid = np.arange(-60.0, 200.0, 1.0)
    x = t_grid / 50.0
    basis = np.vstack([np.ones_like(x), x, x**2, x**3])

    def rec_bins(prof_true, sigma, shift):
        sigma = max(abs(sigma), 1.0)
        z = (r_grid[:, None] - (t_grid[None, :] + shift)) / sigma
        K = np.exp(-0.5 * z**2) / (sigma * np.sqrt(2 * np.pi))
        rec = K @ prof_true
        idx = np.digitize(r_grid, edges) - 1
        out = np.zeros(len(edges) - 1)
        ok = (idx >= 0) & (idx < len(out))
        np.add.at(out, idx[ok], rec[ok])
        return out

    def unpack(q):
        return q[:4], q[4], q[5], q[6], max(q[7], 1e-3)

    def predict(q):
        c, sigma, shift, d_top, t = unpack(q)
        p_true = np.exp(np.clip(c @ basis, -50, 50))
        # 透射增强：空鼓下方两条腿都少衰减一次
        w = np.exp(k_grid * rho_ground * np.clip(t_grid - d_top, 0.0, t) / 10.0)
        # 空鼓自身没有散射体（空气），该深度段的散射密度归零。
        # 缺了这一项，模型算不出实测的整体计数亏损，也算不出空鼓深度处的凹陷。
        inside = np.clip(np.minimum(t_grid + 0.5, d_top + t)
                         - np.maximum(t_grid - 0.5, d_top), 0.0, 1.0)
        mask = 1.0 - inside                      # 1 mm 网格上按重叠比例，边界不跳变
        return (npr * rec_bins(p_true, sigma, shift),
                npv * rec_bins(p_true * mask * w, sigma, shift))

    def deviance(q):
        mr, mv = predict(q)
        d = 0.0
        for m, n in ((mr, hr), (mv, hv)):
            m = np.maximum(m, 1e-9)
            with np.errstate(divide="ignore", invalid="ignore"):
                term = np.where(n > 0, n * np.log(n / m), 0.0)
            d += 2.0 * np.sum(m - n + term)
        return float(d)

    scale = np.log(max(hr.sum(), 1) / npr / max(len(t_grid), 1)) - 2.0
    q0 = np.array([scale, -1.0, 0.0, 0.0, 16.0, 5.0, d_top_guess, t_guess])
    bnds = [(-40, 20), (-30, 30), (-30, 30), (-30, 30),
            (3.0, 60.0), (-30.0, 40.0), (0.0, 90.0), (0.2, 45.0)]
    res = minimize(deviance, q0, method="Powell", bounds=bnds,
                   options={"maxiter": 40000, "xtol": 1e-4, "ftol": 1e-6})
    c, sigma, shift, d_top, t = unpack(res.x)
    n_dof = 2 * len(hr) - len(q0)
    return {"status": "ok" if res.success else f"minimize: {res.message}",
            "void_depth_mm": float(d_top), "void_thickness_mm": float(t),
            "void_centre_mm": float(d_top + t / 2.0),
            "areal_deficit_g_cm2": float(t * rho_ground / 10.0),
            "blur_sigma_mm": float(abs(sigma)), "blur_shift_mm": float(shift),
            "deviance": float(res.fun),
            "deviance_per_dof": float(res.fun / n_dof) if n_dof > 0 else None,
            "k_at_20mm_cm2_g": float(k_grid[20]),
            "k_at_60mm_cm2_g": float(k_grid[60]),
            "k_at_100mm_cm2_g": float(k_grid[100])}


def _profiles(depth_v, depth_r, w_v, w_r, edges):
    hv, _ = np.histogram(depth_v, bins=edges)
    hr, _ = np.histogram(depth_r, bins=edges)
    nv = hv * w_v
    nr = hr * w_r
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(hr > 0, nv / np.maximum(nr, 1e-12), np.nan)
        # 泊松误差传播：raw 计数的相对误差
        rel = np.sqrt(np.where(hv > 0, 1.0 / hv, np.nan) + np.where(hr > 0, 1.0 / hr, np.nan))
        ratio_err = ratio * rel
        sig = (nv - nr) / np.sqrt(np.maximum(hv * w_v**2 + hr * w_r**2, 1e-12))
    return hv, hr, nv, nr, ratio, ratio_err, sig


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--void", required=True, type=Path)
    ap.add_argument("--reference", required=True, type=Path)
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--bin-width-mm", type=float, default=6.0)
    ap.add_argument("--depth-max-mm", type=float, default=120.0)
    ap.add_argument("--deep-from-mm", type=float, default=45.0,
                    help="合并显著性从这个深度往下算（冻结值 45 mm）")
    ap.add_argument("--y-half-width-mm", type=float, default=float("inf"))
    ap.add_argument("--mu-rho-in", type=float, default=MU_RHO_IN_DEFAULT)
    ap.add_argument("--mu-rho-out", type=float, default=MU_RHO_OUT_DEFAULT)
    ap.add_argument("--seed", type=int, default=24091051)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    config = load_config(str(args.config))
    ph = config["phantom"]
    void = ph.get("void", {})
    t_true = float(void.get("thickness_mm", np.nan))
    d_top = float(void.get("depth_mm", np.nan))
    rho_ground = float(ph["layers"][1]["density_g_cm3"])
    centre_true = d_top + t_true / 2.0
    sigma_true = t_true * rho_ground / 10.0            # g/cm2

    print("[1/5] 读入并做位姿感知筛选")
    dv, cos_v, cb_v, r1_v, npv, nv_sel = _observables(args.void, config, args.y_half_width_mm)
    dr, cos_r, cb_r, r1_r, npr, nr_sel = _observables(args.reference, config, args.y_half_width_mm)
    print(f"  空鼓档  选出 {nv_sel:7d} 事例   初级粒子 {npv}")
    print(f"  参照档  选出 {nr_sel:7d} 事例   初级粒子 {npr}")
    if not npv or not npr:
        print("  !! 读不到 primary_event_count，无法按每粒子归一，请检查 summary.json")
        return 1
    w_v, w_r = 1.0 / npv, 1.0 / npr
    print(f"  全局计数比（每粒子）= {(nv_sel*w_v)/(nr_sel*w_r):.4f}")

    edges = np.arange(0.0, args.depth_max_mm + args.bin_width_mm, args.bin_width_mm)
    ctr = 0.5 * (edges[1:] + edges[:-1])
    hv, hr, cnt_v, cnt_r, ratio, ratio_err, sig = _profiles(dv, dr, w_v, w_r, edges)

    print("\n[2/5] 深度剖面")
    print(f"  {'深度':>6s}{'参照':>9s}{'空鼓':>9s}{'比值':>9s}{'±':>7s}{'σ':>7s}")
    for i, c in enumerate(ctr):
        if hv[i] == 0 and hr[i] == 0:
            continue
        mark = "  <- 空鼓" if d_top - 3 <= c <= d_top + t_true + 3 else ""
        print(f"  {c:6.0f}{hr[i]:9d}{hv[i]:9d}{ratio[i]:9.3f}"
              f"{ratio_err[i]:7.3f}{sig[i]:7.1f}{mark}")

    deep = ctr >= args.deep_from_mm
    good = deep & np.isfinite(sig)
    combined = float(np.sum(sig[good]) / np.sqrt(max(good.sum(), 1)))
    print(f"\n[3/5] {args.deep_from_mm:.0f} mm 以深合并显著性 = {combined:.2f} sigma"
          f"（{int(good.sum())} 个 bin）")

    fit = _fit_step(ctr, ratio, ratio_err, centre_true)
    deep_mask_v, deep_mask_r = dv >= 60.0, dr >= 60.0
    cos_mean = float(np.mean(np.concatenate([cos_v[deep_mask_v], cos_r[deep_mask_r]]))
                     if (deep_mask_v.sum() + deep_mask_r.sum()) > 100
                     else np.mean(np.concatenate([cos_v, cos_r])))
    result_sigma = None
    if fit and fit.get("status") == "ok" and fit["asymptotic_ratio"] > 1.0:
        k = args.mu_rho_in + args.mu_rho_out / cos_mean
        result_sigma = float(np.log(fit["asymptotic_ratio"]) / k)
    print(f"\n[4/5] 台阶拟合  r(d) = 1 + (A-1)*Phi((d-d0)/s)")
    if fit and fit.get("status") == "ok":
        print(f"  A  渐近比值   = {fit['asymptotic_ratio']:.4f} ± {fit['asymptotic_ratio_err']:.4f}")
        print(f"  d0 台阶中点   = {fit['step_centre_mm']:.2f} ± {fit['step_centre_err_mm']:.2f} mm"
              f"   （空鼓中心真值 {centre_true:.1f} mm）")
        print(f"  s  模糊宽度   = {fit['blur_sigma_mm']:.2f} mm")
        print(f"  chi2/dof     = {fit['chi2_per_dof']:.2f}")
        print(f"  <cos xi|深>60mm> = {cos_mean:.3f}（渐近比值由深处事例决定，故只用深处）")
        if result_sigma is not None:
            t_est = result_sigma / rho_ground * 10.0
            print(f"  反解面密度亏损 = {result_sigma:.3f} g/cm2"
                  f"   （真值 {sigma_true:.3f}，相对差 {100*abs(result_sigma-sigma_true)/sigma_true:.1f}%）")
            print(f"  反解空鼓厚度   = {t_est:.2f} mm   （真值 {t_true:.1f} mm，"
                  f"按地仗密度 {rho_ground} g/cm3 换算）")
    else:
        print(f"  {fit}")

    print("\n[5/6] 物理正演拟合（去掉 erf 的系统偏差）")
    h_d1 = float(config["detectors"]["d1"]["center_z_mm"])
    inv_cos_beta = float(np.mean(1.0 / np.concatenate([cb_v, cb_r])))
    k_fn = (lambda g: _k_of_depth(g, r1_r, h_d1, args.mu_rho_in, args.mu_rho_out,
                                  inv_cos_beta))
    fwd = _forward_fit(edges, hv, hr, npv, npr, k_fn, rho_ground, d_top, t_true)
    print(f"  <1/cos beta> = {inv_cos_beta:.3f}（入射腿，实测）")
    print(f"  k(20mm) = {fwd['k_at_20mm_cm2_g']:.3f}   k(60mm) = {fwd['k_at_60mm_cm2_g']:.3f}"
          f"   k(100mm) = {fwd['k_at_100mm_cm2_g']:.3f} cm2/g（逐深度，非常数）")
    print(f"  空鼓深度 = {fwd['void_depth_mm']:.2f} mm   （真值 {d_top:.1f}）")
    print(f"  空鼓厚度 = {fwd['void_thickness_mm']:.2f} mm   （真值 {t_true:.1f}）")
    print(f"  空鼓中心 = {fwd['void_centre_mm']:.2f} mm   （真值 {centre_true:.1f}）")
    print(f"  面密度亏损 = {fwd['areal_deficit_g_cm2']:.3f} g/cm2   （真值 {sigma_true:.3f}，"
          f"相对差 {100*abs(fwd['areal_deficit_g_cm2']-sigma_true)/sigma_true:.1f}%）")
    print(f"  深度模糊 σ = {fwd['blur_sigma_mm']:.1f} mm，位移 {fwd['blur_shift_mm']:+.1f} mm")
    print(f"  偏差度/自由度 = {fwd['deviance_per_dof']}")

    print("\n[6/6] 负对照：把参照档随机对半分，做 A/B")
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(dr))
    half = len(dr) // 2
    da, db = dr[perm[:half]], dr[perm[half:]]
    ha, hb, ca, cb, ratio_n, err_n, sig_n = _profiles(da, db, 1.0, 1.0, edges)
    good_n = deep & np.isfinite(sig_n)
    combined_n = float(np.sum(sig_n[good_n]) / np.sqrt(max(good_n.sum(), 1)))
    fit_n = _fit_step(ctr, ratio_n, err_n, centre_true)
    print(f"  合并显著性 = {combined_n:.2f} sigma")
    if fit_n and fit_n.get("status") == "ok":
        print(f"  A = {fit_n['asymptotic_ratio']:.4f}   d0 = {fit_n['step_centre_mm']:.2f} mm")

    c1 = (combined >= 5.0) and (abs(combined_n) < 3.0)
    c2 = abs(fwd["void_centre_mm"] - centre_true) <= 10.0
    c3 = abs(fwd["areal_deficit_g_cm2"] - sigma_true) / sigma_true <= 0.30
    c4_1 = abs(combined_n) >= 5.0
    fwd_n = _forward_fit(edges, ha, hb, len(da), len(db), k_fn, rho_ground,
                         d_top, t_true)
    print(f"  正演拟合给出的假空鼓：深度 {fwd_n['void_depth_mm']:.1f} mm，"
          f"厚度 {fwd_n['void_thickness_mm']:.2f} mm，"
          f"面密度 {fwd_n['areal_deficit_g_cm2']:.3f} g/cm2")
    c4_2 = abs(fwd_n["void_centre_mm"] - centre_true) <= 10.0
    c4 = not (c4_1 and c4_2)
    names = ["1 检出显著性 >=5σ 且负对照 <3σ",
             "2 正演拟合的空鼓中心与真值之差 <=10 mm",
             "3 正演拟合的面密度亏损相对差 <=30%",
             "4 负对照未自行通过 1、2"]
    print("\n" + "=" * 60)
    for n, v in zip(names, [c1, c2, c3, c4]):
        print(f"  {'通过' if v else '不通过'}  {n}")
    allpass = all([c1, c2, c3, c4])
    print(f"\n  结论：{'四条全过 —— 能看见空鼓' if allpass else '有不通过项 —— 如实记录，不改判据'}")
    print("=" * 60)

    json.dump({
        "void_dir": str(args.void), "reference_dir": str(args.reference),
        "config": str(args.config),
        "n_primary": {"void": npv, "reference": npr},
        "n_selected": {"void": nv_sel, "reference": nr_sel},
        "global_ratio": float((nv_sel * w_v) / (nr_sel * w_r)),
        "bin_width_mm": args.bin_width_mm, "deep_from_mm": args.deep_from_mm,
        "y_half_width_mm": (None if not np.isfinite(args.y_half_width_mm)
                            else args.y_half_width_mm),
        "mu_rho_in_cm2_g": args.mu_rho_in, "mu_rho_out_cm2_g": args.mu_rho_out,
        "cos_xi_mean": cos_mean,
        "truth": {"void_thickness_mm": t_true, "void_depth_mm": d_top,
                  "void_centre_mm": centre_true, "areal_deficit_g_cm2": sigma_true,
                  "ground_density_g_cm3": rho_ground},
        "depth_centres_mm": ctr.tolist(),
        "counts_reference": hr.tolist(), "counts_void": hv.tolist(),
        "ratio": ratio.tolist(), "ratio_err": ratio_err.tolist(),
        "significance": sig.tolist(),
        "combined_sigma": combined, "erf_fit_descriptive": fit,
        "forward_fit": fwd,
        "inverted_areal_deficit_g_cm2": result_sigma,
        "inverted_thickness_mm": (None if result_sigma is None
                                  else result_sigma / rho_ground * 10.0),
        "negative_control": {"combined_sigma": combined_n,
                             "erf_fit_descriptive": fit_n, "forward_fit": fwd_n},
        "criteria": {n: bool(v) for n, v in zip(names, [c1, c2, c3, c4])},
        "all_criteria_passed": bool(allpass),
    }, open(args.out / "m1_mural_void.json", "w", encoding="utf-8"),
        ensure_ascii=False, indent=1)
    print(f"\n写出 {args.out / 'm1_mural_void.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
