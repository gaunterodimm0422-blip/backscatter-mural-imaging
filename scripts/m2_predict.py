"""M2 预估：半空间空鼓（边界在 x = 0 的头部局部坐标）能不能定出边界、定多准。

要回答的三件事
--------------
1. 重建横向剖面上，空鼓侧/完好侧的计数比值台阶有多宽（**边界模糊宽度**）。
   这一项由几何决定：入射腿与出射腿都要斜穿空鼓层，斜穿时横向会跑掉一段，
   把本来是阶跃的边界抹成一个过渡带。几何量蒙卡能算准。
2. 台阶落在哪儿（**有没有系统偏移**）。出射腿偏向 D1 一侧，可能把边界推偏。
3. 给定台阶宽度与实测对比度，2×200 万事例能把边界定到多少 mm（**σ(x_e)**）。

对比度（台阶高度）不用本蒙卡的值
--------------------------------
M1 已经证明本蒙卡低估对比度：它预言整体计数比 0.9964（几乎不变），
实测是 0.9679（-3.2%），差了约 9 倍，原因是它不含多次散射。
所以台阶高度一律用 **M1 实测的 -3.2%**，蒙卡只用来给形状（宽度、位置）。
这一条必须写在报告里：形状来自蒙卡、幅度来自实测。
"""
from __future__ import annotations
import sys, json, time
import numpy as np

sys.path.insert(0, "/home/claude/design/_stubs")
sys.path.insert(0, "/mnt/user-data/uploads/beisanshe")
from backscatter.kinematics import solve_scatter_points
from backscatter.electronics import energy_sigma_keV

E0 = 511.0
MU_RHO_IN, MU_RHO_OUT = 0.0877, 0.125      # cm^2/g @511 keV 与出射带 ~200 keV
MU_GAGG = 0.16                             # mm^-1 @~200 keV
GROUND_RHO, SUPPORT_RHO = 1.70, 2.20
GROUND_FINE, GROUND_COARSE, SUPPORT = 5.0, 35.0, 100.0
GROUND_BOT = GROUND_FINE + GROUND_COARSE   # 40 mm
import os
VOID_T = float(os.environ.get("VOID_T_MM", "10"))   # 空鼓厚度，可用环境变量覆盖
VOID_TOP_D, VOID_BOT_D = GROUND_BOT - VOID_T, GROUND_BOT     # 深度 30~40 mm
X_EDGE = 0.0                               # 头部局部坐标；空鼓占 x > X_EDGE
H_SRC, H_D1, L2, A2 = 17.5, 17.5, 42.5, 25.0
R1_IN, R1_OUT, T1 = 20.0, 200.0, 15.0
D1_PX, D1_PZ, D1_ERES = 1.55, 5.0, 0.08
E_LO, E_HI = 180.0, 250.0
TAG_KERNEL = np.asarray(json.load(open("/home/claude/design/tag_kernel.json"))["all"], float)


def column_density_full(depth_mm):
    """无空鼓时，从表面垂直往下到 depth 的面密度 g/cm2。"""
    d = np.asarray(depth_mm, float)
    return (np.clip(d, 0.0, GROUND_BOT) * GROUND_RHO
            + np.clip(d - GROUND_BOT, 0.0, SUPPORT) * SUPPORT_RHO) / 10.0


def seg_len_in_void(A, B):
    """线段 A->B 落在空鼓体（深度 30~40 mm 且 x > X_EDGE 的半空间）内的长度 mm。

    z 与 x 都沿线段线性变化，各自给出一个 s 区间，取交集即可。
    """
    D = B - A
    L = np.linalg.norm(D, axis=1)
    s_lo = np.zeros(len(A)); s_hi = np.ones(len(A))

    # --- z 方向：z ∈ [-VOID_BOT_D, -VOID_TOP_D] ---
    zlo, zhi = -VOID_BOT_D, -VOID_TOP_D
    dz = D[:, 2]
    flat = np.abs(dz) < 1e-12
    with np.errstate(divide="ignore", invalid="ignore"):
        sa = (zlo - A[:, 2]) / dz
        sb = (zhi - A[:, 2]) / dz
    s1 = np.where(flat, np.where((A[:, 2] >= zlo) & (A[:, 2] <= zhi), 0.0, 1.0),
                  np.minimum(sa, sb))
    s2 = np.where(flat, np.where((A[:, 2] >= zlo) & (A[:, 2] <= zhi), 1.0, 0.0),
                  np.maximum(sa, sb))
    s_lo = np.maximum(s_lo, s1); s_hi = np.minimum(s_hi, s2)

    # --- x 方向：x > X_EDGE ---
    dx = D[:, 0]
    flatx = np.abs(dx) < 1e-12
    with np.errstate(divide="ignore", invalid="ignore"):
        sx = (X_EDGE - A[:, 0]) / dx
    lo_x = np.where(flatx, np.where(A[:, 0] > X_EDGE, 0.0, 1.0),
                    np.where(dx > 0, sx, 0.0))
    hi_x = np.where(flatx, np.where(A[:, 0] > X_EDGE, 1.0, 0.0),
                    np.where(dx > 0, 1.0, sx))
    s_lo = np.maximum(s_lo, lo_x); s_hi = np.minimum(s_hi, hi_x)

    return L * np.clip(s_hi - s_lo, 0.0, 1.0)


def _bin(x, pitch, origin=0.0):
    return origin + (np.floor((x - origin) / pitch) + 0.5) * pitch


def _perp_basis(v):
    a = np.zeros_like(v); a[:, 0] = 1.0
    m = np.abs(v[:, 0]) > 0.9
    a[m] = 0.0; a[m, 1] = 1.0
    e1 = np.cross(v, a); e1 /= np.linalg.norm(e1, axis=1)[:, None]
    return e1, np.cross(v, e1)


def sample(n, with_void, rng, depth_max=120.0):
    S = np.array([0.0, 0.0, H_SRC])
    x2 = rng.uniform(-A2, A2, n); y2 = rng.uniform(-A2, A2, n)
    X2t = np.column_stack([x2, y2, np.full(n, H_SRC + L2)])
    d_tag = X2t - S
    dist_tag = np.linalg.norm(d_tag, axis=1)
    tag = d_tag / dist_tag[:, None]
    u = -tag
    w_tag = (L2 / dist_tag) / dist_tag ** 2
    cos_b = -u[:, 2]

    depth = rng.uniform(0.0, depth_max, n)
    P = S + (((H_SRC + depth) / cos_b)[:, None]) * u
    Sb = np.broadcast_to(S, P.shape)

    # 入射腿：从表面进入点到 P。用 S->P 即可（S 在真空里，真空段不贡献密度）。
    sigma_in = column_density_full(depth) / np.maximum(cos_b, 1e-6)
    rho_p = np.where((depth <= GROUND_BOT), GROUND_RHO,
                     np.where(depth <= GROUND_BOT + SUPPORT, SUPPORT_RHO, 0.0))
    if with_void:
        lv_in = seg_len_in_void(np.ascontiguousarray(Sb), P)
        sigma_in = sigma_in - lv_in * GROUND_RHO / 10.0
        in_void = ((depth >= VOID_TOP_D) & (depth <= VOID_BOT_D) & (P[:, 0] > X_EDGE))
        rho_p = np.where(in_void, 0.0, rho_p)
    w_int = rho_p / np.maximum(cos_b, 1e-6) * np.exp(-MU_RHO_IN * np.maximum(sigma_in, 0.0))
    ok = w_int > 0

    z_bot = H_D1 - T1 / 2
    r = np.sqrt(rng.uniform(R1_IN ** 2, R1_OUT ** 2, n))
    ph = rng.uniform(0, 2 * np.pi, n)
    ent = np.column_stack([r * np.cos(ph), r * np.sin(ph), np.full(n, z_bot)])
    d_out = ent - P
    dist1 = np.linalg.norm(d_out, axis=1)
    v = d_out / dist1[:, None]
    ok &= v[:, 2] > 1e-6
    cos_th = np.clip(np.einsum("ij,ij->i", u, v), -1.0, 1.0)
    theta = np.arccos(cos_th)
    Ep = E0 / (2.0 - cos_th)
    cos_xi = np.clip(v[:, 2], 1e-6, 1.0)
    sigma_out = column_density_full(depth) / cos_xi
    if with_void:
        lv_out = seg_len_in_void(P, ent)
        sigma_out = sigma_out - lv_out * GROUND_RHO / 10.0
    kn = np.clip((Ep / E0) ** 2 * (Ep / E0 + E0 / Ep - np.sin(theta) ** 2), 0.0, None)
    path_max = T1 / cos_xi
    p_abs = 1.0 - np.exp(-MU_GAGG * path_max)
    w = (w_tag * w_int * kn * cos_xi / dist1 ** 2 * p_abs
         * np.exp(-MU_RHO_OUT * np.maximum(sigma_out, 0.0)))

    E_meas = Ep + rng.normal(0, 1, n) * energy_sigma_keV(Ep, fwhm_fraction_at_511=D1_ERES)
    s_doi = -np.log(1 - rng.uniform(0, 1, n) * (1 - np.exp(-MU_GAGG * path_max))) / MU_GAGG
    X1 = ent + s_doi[:, None] * v
    X1d = np.column_stack([_bin(X1[:, 0], D1_PX), _bin(X1[:, 1], D1_PX),
                           _bin(X1[:, 2], D1_PZ, origin=z_bot)])
    ang = TAG_KERNEL[rng.integers(0, len(TAG_KERNEL), n)]
    az = rng.uniform(0, 2 * np.pi, n)
    e1, e2 = _perp_basis(tag)
    X2 = X2t + (dist_tag * np.tan(ang))[:, None] * (e1 * np.cos(az)[:, None]
                                                    + e2 * np.sin(az)[:, None])
    X2[:, 2] = H_SRC + L2
    ok &= (np.abs(X2[:, 0]) <= A2) & (np.abs(X2[:, 1]) <= A2)
    ok &= (E_meas >= E_LO) & (E_meas <= E_HI) & np.isfinite(w) & (w > 0)

    norm = (2 * A2) ** 2 * depth_max * np.pi * (R1_OUT ** 2 - R1_IN ** 2) / (4 * np.pi) ** 2
    norm *= MU_RHO_IN * 0.1
    return dict(P=P[ok], X1=X1d[ok], X2=X2[ok], E=E_meas[ok], w=w[ok],
                x_true=P[ok][:, 0], depth_true=depth[ok],
                eff=w[ok].sum() / n * norm)


def run(with_void, seed, n_chunk=500_000, n_rep=8):
    rng = np.random.default_rng(seed)
    P = X1 = X2 = E = w = dt = xt = None
    acc = {k: [] for k in ("X1", "X2", "E", "w", "depth_true", "x_true")}
    effs = []
    for _ in range(n_rep):
        s = sample(n_chunk, with_void, rng)
        for k in acc: acc[k].append(s[k])
        effs.append(s["eff"])
    X1 = np.concatenate(acc["X1"]); X2 = np.concatenate(acc["X2"])
    E = np.concatenate(acc["E"]); w = np.concatenate(acc["w"])
    dt = np.concatenate(acc["depth_true"]); xt = np.concatenate(acc["x_true"])
    S = np.broadcast_to(np.array([0.0, 0.0, H_SRC]), X1.shape).copy()
    sol = solve_scatter_points(S, X1, X2, E)
    g = sol.valid
    return dict(eff=float(np.mean(effs)), w=w[g],
                x_rec=sol.points_mm[g, 0], y_rec=sol.points_mm[g, 1],
                depth_rec=-sol.points_mm[g, 2], x_true=xt[g], depth_true=dt[g])


if __name__ == "__main__":
    t0 = time.time()
    ref = run(False, 24091101)
    voi = run(True, 24091102)
    print(f"蒙卡用时 {time.time()-t0:.0f} s   "
          f"参照事例 {len(ref['w']):,}   空鼓事例 {len(voi['w']):,}")
    print(f"效率：参照 {ref['eff']:.4e}/湮没   半空间空鼓 {voi['eff']:.4e}/湮没   "
          f"比值 {voi['eff']/ref['eff']:.4f}")

    edges = np.arange(-60, 61, 4.0)
    ctr = 0.5 * (edges[1:] + edges[:-1])
    sel = lambda d: (np.abs(d["y_rec"]) <= 40) & (d["depth_rec"] >= 0) & (d["depth_rec"] <= 120)
    out = {}
    for band, lo, hi in (("全深度", 0, 120), ("浅 0-30", 0, 30),
                         ("空鼓层 30-40", 30, 40), ("深 45+", 45, 120)):
        h = {}
        for name, d in (("ref", ref), ("void", voi)):
            m = sel(d) & (d["depth_rec"] >= lo) & (d["depth_rec"] < hi)
            h[name], _ = np.histogram(d["x_rec"][m], bins=edges, weights=d["w"][m])
        h["ref"] *= ref["eff"] / ref["w"].sum()
        h["void"] *= voi["eff"] / voi["w"].sum()
        ratio = h["void"] / np.maximum(h["ref"], 1e-30)
        out[band] = dict(ctr=ctr.tolist(), ref=h["ref"].tolist(),
                         void=h["void"].tolist(), ratio=ratio.tolist())
    json.dump(out, open("/home/claude/design/m2_predict.json", "w"),
              ensure_ascii=False, indent=1)

    print(f"\n{'x_rec':>7s}" + "".join(f"{b:>14s}" for b in out))
    for i, c in enumerate(ctr):
        if abs(c) > 56: continue
        row = f"{c:7.0f}"
        for b in out:
            row += f"{out[b]['ratio'][i]:14.4f}"
        print(row)
