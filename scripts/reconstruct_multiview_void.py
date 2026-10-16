"""S2V：混凝土体内空洞的多位姿深度定位（判据在见到数据之前冻结）。

和 S2 的区别
------------
S2 的目标是空气里的一根混凝土细杆——亮点。似然相乘找的是"两根针的交点"，
天然适合亮点。本轮目标是混凝土板里的空洞——亮背景上的**暗缺损**，负信号。
"图上哪最亮"不再对应"缺陷在哪"，所以估计器必须换。

估计器
------
1. 每个位姿都有一对跑：空洞跑 + 参照跑（同一个体模宏，只把 VOID_X_MM 从 0
   挪到 200 mm 挪出视野）。几何完全一致，所以孔径衰减在相除时精确抵消。
2. 位姿 p 的**亏损显著性图**（不是裸对比度）：
       S_p = clip( (r - v) / sqrt(var_r + var_v), 0, inf )
   其中 v、r 是空洞图与参照图在有效区内归一化到和为 1 的形状，
   var 由逐事件贡献的平方和给出（每个事件是一次独立抽样）。
   **必须除以统计误差**：深处的图很暗，裸对比度在那里的噪声比空洞本身的
   亏损还大，峰会跑到最噪的地方去——这一点在合成数据上验证时抓到过。
3. 联合定位图（与 S2 同构，便于横向比较）：
       J = (S_A / max S_A) x (S_B / max S_B)
   峰值位置与 FWHM 用 reconstruct_multiview_rod.py 里同一套 profiles /
   fwhm_of_profile，保证两轮口径一致。
4. 检出统计量用几何平均 sqrt(S_A * S_B)：要求两个位姿在同一像素上同时显著。

有效区（对信号盲，只由参照跑决定）
----------------------------------
  **逐深度层**判定：某层里参照图 >= 该层横向峰值的 20%，两个位姿都要满足，
  且深度在 [8, 60] mm 内。
按层判定是必须的：孔径限制是横向的，而图沿深度本来就衰减近两个量级；
用全图最大值当阈值会把深处整片切掉（实测只剩 117 个像素）。
参照跑里没有空洞，所以这个掩膜不含任何信号信息。

冻结判据
--------
  1. 检出：同等统计量下（各取一半事件），信号图的 max sqrt(S_A*S_B)
          >= 2 x 负对照图（参照前半 vs 参照后半）的同一统计量
  2. 定位：J 的峰值距真值 (x=0, 深度 30 mm) <= 8 mm
  3. 深度不确定度：J 的 bootstrap 峰值深度 95%CI 宽度 <= 15 mm
          （单视角深度 PSF 是 38.5 mm）
  4. 收窄来自多视角：联合 CI 宽度相对两个单位姿 CI 宽度均值收窄 >= sqrt(2)
空洞真值只用于事后打分，不参与重建、掩膜或选参。

**不再用"联合深度 FWHM"当判据**：对空洞来说，亏损斑块的宽度反映的是空洞
自身的尺寸（20 mm 的空洞给出约 20 mm 的斑块），不是分辨率。FWHM 仍然打印
和入库，但只作参考。

标定与冻结的过程（诚实记录）
----------------------------
估计器与上面四条判据是在**合成体模**上定下来的——真值已知、而 GATE 数据尚
不存在，所以这是"先在模拟真值上调好、再盲测真实数据"的正规做法，不是看着
真实结果改判据。合成体模（20 mm 空洞、深 30 mm、每档 5.4 万事件、单散射
玩具 MC）上定稿版给出：
    检出比 2.87、峰值偏差 2.55 mm、联合深度 CI 宽度 6.0 mm、CI 收窄 1.17x
也就是说，**在合成数据上第 4 条本身就没过**。这不是 bug：在均匀板 + 参照跑
的条件下，单个位姿已经能看到"空洞上方正常 / 洞内亏损 / 洞下方略有过量"的
深度结构，本来就比孤立亮点好定位，多视角能加的东西有限。真实 GATE 数据里
多次散射会把这个结构抹掉一些，第 4 条有可能反而更容易过。无论过不过，
结果本身就是对项目有意义的答案。

调参过程中被合成数据抓出来的两个真 bug（已修）：
  - 有效区用"全图最大值的 20%"来切 —— 图沿深度衰减近两个量级，这样切完
    只剩 117 个像素，峰跑到近表面。改为逐深度层判定。
  - 直接用裸对比度 1 - v/r —— 深处图暗，那里的噪声比空洞亏损还大，峰跑到
    深 52 mm 的纯噪声处。改为除以统计误差的显著性图，并加高斯平滑。

另外报告（不作判据）：bootstrap 区间；镜像背景交叉校验（不用参照跑，
改用"均匀板对探测头轴对称"这一条，看两种背景估计是否给出一致的结论）。

用法
----
  E:\\python3.12\\python.exe scripts\\reconstruct_multiview_void.py ^
      --pair runs\\s2v_void_L-15_2m_seed24090111\\canonical_response ^
             runs\\s2v_ref_L-15_2m_seed24090112\\canonical_response ^
      --pair runs\\s2v_void_L15_2m_seed24090113\\canonical_response ^
             runs\\s2v_ref_L15_2m_seed24090114\\canonical_response
"""
from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from backscatter.config import load_config
from backscatter.iterative import ReconstructionGrid2D, build_ray_likelihood_system
from backscatter.pose_selection import build_selection_pose_aware

# 直接复用细杆脚本里的剖面/FWHM 实现，保证两轮口径完全一致
_ROD = PROJECT_ROOT / "scripts" / "reconstruct_multiview_rod.py"
_spec = importlib.util.spec_from_file_location("_mv_rod", _ROD)
_rod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_rod)
profiles, fwhm_of_profile = _rod.profiles, _rod.fwhm_of_profile

DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "transverse_fwhm_5mm_frozen.json"
TRUE_X_MM, TRUE_DEPTH_MM = 0.0, 30.0

# ---- 冻结常数 ----
MASK_REF_FRACTION = 0.20          # 参照图不低于自身峰值的这个比例
MASK_DEPTH_MM = (8.0, 60.0)
CRIT_DETECT_RATIO_MIN = 2.0        # 信号峰 >= 纯噪声最大涨落的 2 倍
CRIT_PEAK_OFFSET_MAX_MM = 8.0      # 20 mm 的空洞定位到优于 8 mm
CRIT_JOINT_DEPTH_CI_MAX_MM = 15.0  # 单视角深度 PSF 是 38.5 mm，15 mm 是明确改善
CRIT_CI_GAIN_MIN = 1.4142135623730951
# = sqrt(2)：这是原则性的门槛，不是拍脑袋的数。联合用了两个位姿共 2 倍事件，
# 光靠统计量翻倍就能把置信区间收窄 sqrt(2)。所以"收窄 >= sqrt(2)"等价于
# "第二个视角至少抵得上把统计量翻一倍"；只有超过它，收窄才真的来自几何。
BOOTSTRAP_N = 200
BOOTSTRAP_SEED = 24090101
SMOOTH_SIGMA_MM = 3.0             # 高斯平滑；FWHM = 2.355 x sigma = 7.06 mm
SMOOTH_FWHM_MM = 2.3548200450309493 * SMOOTH_SIGMA_MM


def load_npz(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as d:
        return {k: d[k] for k in d.files}


def build_rows(pose_dir: Path, config, grid, model):
    """把一个分析目录变成 (行数据, 列号, 行号, 行数, 像素数, 位姿L, 筛选数)。"""
    pairs = load_npz(pose_dir / "pairs.npz")
    cfg = copy.deepcopy(config)
    cfg["selection"]["phantom_truth_policy"] = "ignore"
    sel, _, _ = build_selection_pose_aware(pairs, cfg)
    n_sel = int(sel.sum())
    if n_sel == 0:
        raise SystemExit(f"!! {pose_dir} 筛选后为 0 个事件，先查几何判据是否以探测头轴心为中心")
    system = build_ray_likelihood_system(pairs, sel, grid, **model)
    m = system.matrix
    rows = np.repeat(np.arange(m.shape[0]), np.diff(m.indptr))
    head_x = float(np.median(np.asarray(pairs["source_x_mm"], dtype=np.float64)[sel]))
    return dict(data=m.data, cols=m.indices, rows=rows, n_rows=m.shape[0],
                n_pix=m.shape[1], head_x=head_x, n_sel=n_sel, name=pose_dir.parent.name)


def _gauss1d(sigma_px: float) -> np.ndarray:
    radius = int(np.ceil(4.0 * sigma_px))
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    k = np.exp(-0.5 * (x / sigma_px) ** 2)
    return k / k.sum()


def smooth(m, sigma_mm, bin_mm, variance=False):
    """可分离高斯平滑（只用 numpy）。

    1 mm 像素上每格只有几个"等效事件"，10 mm 的空洞亏损会被噪声完全淹没
    （合成数据上实测：不平滑时峰跑到深 52 mm 的纯噪声处）。空洞本身 10 mm，
    所以 sigma=2 mm 的平滑几乎不损失可分辨性，却把每个分辨单元的有效统计量
    提高约 25 倍。
    variance=True 时用核的平方做传播：Var(sum w a) = sum w^2 Var(a)。
    """
    if sigma_mm <= 0:
        return m
    k = _gauss1d(sigma_mm / bin_mm)
    if variance:
        k = k ** 2                     # 注意：不再归一化
    out = np.apply_along_axis(lambda v: np.convolve(v, k, mode="same"), 0, m)
    return np.apply_along_axis(lambda v: np.convolve(v, k, mode="same"), 1, out)


def make_map(ds, weights, shape, square=False):
    """逐像素累加行贡献；square=True 时累加平方，用来估计统计误差。"""
    out = np.zeros(ds["n_pix"], dtype=np.float64)
    d = ds["data"] ** 2 if square else ds["data"]
    np.add.at(out, ds["cols"], d * weights[ds["rows"]])
    return out.reshape(shape)


def row_mask(m, frac):
    """逐深度层的横向有效区：该层里不低于本层横向峰值的 frac。"""
    pk = m.max(axis=1, keepdims=True)
    return (m >= frac * pk) & (pk > 0)


def significance(m_void, q_void, m_ref, q_ref, mask):
    """归一化形状上的亏损除以其统计误差，负值截零。"""
    s = np.zeros_like(m_void)
    sv, sr = m_void[mask].sum(), m_ref[mask].sum()
    if sv <= 0 or sr <= 0:
        return s
    v, qv = m_void / sv, q_void / sv ** 2
    r, qr = m_ref / sr, q_ref / sr ** 2
    den = np.sqrt(qv + qr)
    ok = mask & (den > 0)
    s[ok] = (r[ok] - v[ok]) / den[ok]
    return np.clip(s, 0.0, None)


def joint_localizer(cA, cB):
    if cA.max() <= 0 or cB.max() <= 0:
        return np.zeros_like(cA)
    return (cA / cA.max()) * (cB / cB.max())


def geo_mean_peak(cA, cB, mask):
    g = np.sqrt(np.clip(cA, 0, None) * np.clip(cB, 0, None))
    g = np.where(mask, g, 0.0)
    return float(g.max()), g


def measure(m, grid):
    if not np.isfinite(m).any() or m.max() <= 0:
        return dict(depth_fwhm_mm=float("nan"), lateral_fwhm_mm=float("nan"),
                    peak_x_mm=float("nan"), peak_depth_mm=float("nan"))
    (zc, dp), (xc, lp), (px, pz) = profiles(m, grid)
    d, _ = fwhm_of_profile(zc, dp)
    l, _ = fwhm_of_profile(xc, lp)
    return dict(depth_fwhm_mm=float(d), lateral_fwhm_mm=float(l),
                peak_x_mm=float(px), peak_depth_mm=float(pz))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", nargs=2, type=Path, action="append", required=True,
                    metavar=("VOID_DIR", "REF_DIR"),
                    help="一个位姿的 (空洞跑, 参照跑) canonical_response 目录；需要两个位姿")
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--x-range-mm", type=float, default=60.0)
    ap.add_argument("--depth-max-mm", type=float, default=80.0)
    ap.add_argument("--bin-mm", type=float, default=1.0)
    args = ap.parse_args()
    if len(args.pair) != 2:
        ap.error("需要恰好两个 --pair（两个位姿）")

    out_dir = args.out_dir or (PROJECT_ROOT / "outputs" / f"multiview_void_s2v_{date.today():%Y%m%d}")
    out_dir.mkdir(parents=True, exist_ok=True)

    config = load_config(args.config)
    b = args.bin_mm
    grid = ReconstructionGrid2D(
        x_edges_mm=np.arange(-args.x_range_mm, args.x_range_mm + b, b),
        depth_edges_mm=np.arange(0.0, args.depth_max_mm + b, b))
    model = {
        "incident_energy_keV": float(config["simulation"]["incident_energy_keV"]),
        "electron_rest_energy_keV": float(config["simulation"]["electron_rest_energy_keV"]),
        "d1_energy_fwhm_fraction_at_511": float(
            config["detectors"]["d1"]["energy_response"]["fwhm_fraction_at_511"]),
        "d1_position_resolution_mm": tuple(
            config["detectors"]["d1"]["position_response"]["resolution_mm"]),
        "d2_position_resolution_mm": tuple(
            config["detectors"]["d2"]["position_response"]["resolution_mm"]),
    }
    print(f"重建网格: x ±{args.x_range_mm:.0f} mm, 深度 0–{args.depth_max_mm:.0f} mm, "
          f"像素 {b:.1f} mm  -> {grid.shape}")
    print()

    sets = []
    for i, (void_dir, ref_dir) in enumerate(args.pair, start=1):
        v = build_rows(void_dir.resolve(), config, grid, model)
        r = build_rows(ref_dir.resolve(), config, grid, model)
        print(f"[{i}/2] 位姿 L = {v['head_x']:+.2f} mm")
        print(f"    空洞跑 {v['name']}: 筛选 {v['n_sel']}  似然保留 {v['n_rows']}")
        print(f"    参照跑 {r['name']}: 筛选 {r['n_sel']}  似然保留 {r['n_rows']}")
        if abs(v["head_x"] - r["head_x"]) > 1e-6:
            raise SystemExit(f"!! 空洞跑与参照跑位姿不一致：{v['head_x']} vs {r['head_x']}")
        sets.append((v, r))
    print()

    ones = lambda ds: np.ones(ds["n_rows"])
    sm = lambda m: smooth(m, SMOOTH_SIGMA_MM, b)
    sq = lambda m: smooth(m, SMOOTH_SIGMA_MM, b, variance=True)
    mv = [sm(make_map(v, ones(v), grid.shape)) for v, r in sets]
    mr = [sm(make_map(r, ones(r), grid.shape)) for v, r in sets]
    qv = [sq(make_map(v, ones(v), grid.shape, square=True)) for v, r in sets]
    qr = [sq(make_map(r, ones(r), grid.shape, square=True)) for v, r in sets]
    print(f"平滑: 高斯 sigma = {SMOOTH_SIGMA_MM:.1f} mm (FWHM {SMOOTH_FWHM_MM:.2f} mm)")

    # ---- 有效区：只由参照跑决定，对信号盲 ----
    zc = grid.depth_centers_mm
    depth_ok = ((zc >= MASK_DEPTH_MM[0]) & (zc <= MASK_DEPTH_MM[1]))[:, None]
    mask = depth_ok & np.ones((1, grid.shape[1]), bool)
    for m in mr:
        mask = mask & row_mask(m, MASK_REF_FRACTION)
    print(f"有效区: {int(mask.sum())} 个像素"
          f"（逐层：参照图 >= 本层横向峰值的 {MASK_REF_FRACTION:.0%}，"
          f"深度 {MASK_DEPTH_MM[0]:.0f}–{MASK_DEPTH_MM[1]:.0f} mm）")

    cA = significance(mv[0], qv[0], mr[0], qr[0], mask)
    cB = significance(mv[1], qv[1], mr[1], qr[1], mask)
    J = joint_localizer(cA, cB)
    jm = measure(J, grid)
    offset = float(np.hypot(jm["peak_x_mm"] - TRUE_X_MM, jm["peak_depth_mm"] - TRUE_DEPTH_MM))

    single = [measure(c, grid) for c in (cA, cB)]
    for i, s in enumerate(single):
        print(f"  单位姿 {'AB'[i]} 显著性图: 深度 FWHM {s['depth_fwhm_mm']:6.2f} mm  "
              f"横向 {s['lateral_fwhm_mm']:6.2f} mm  峰值 ({s['peak_x_mm']:+.1f}, {s['peak_depth_mm']:.1f})")

    # ---- 平方效应对照 ----
    sq_ctrl = [measure(joint_localizer(c, c), grid)["depth_fwhm_mm"] for c in (cA, cB)]
    sq_mean = float(np.nanmean(sq_ctrl))
    geo_gain = sq_mean / jm["depth_fwhm_mm"] if jm["depth_fwhm_mm"] > 0 else float("nan")

    # ---- 检出：同等统计量下 信号 vs 负对照 ----
    def half_w(ds, parity):
        w = np.zeros(ds["n_rows"]); w[parity::2] = 1.0; return w
    sig_c, ctl_c = [], []
    for v, r in sets:
        wv0, wr0, wr1 = half_w(v, 0), half_w(r, 0), half_w(r, 1)
        mv_h = sm(make_map(v, wv0, grid.shape)); qv_h = sq(make_map(v, wv0, grid.shape, True))
        mr_h0 = sm(make_map(r, wr0, grid.shape)); qr_h0 = sq(make_map(r, wr0, grid.shape, True))
        mr_h1 = sm(make_map(r, wr1, grid.shape)); qr_h1 = sq(make_map(r, wr1, grid.shape, True))
        sig_c.append(significance(mv_h, qv_h, mr_h0, qr_h0, mask))
        ctl_c.append(significance(mr_h1, qr_h1, mr_h0, qr_h0, mask))
    sig_peak, _ = geo_mean_peak(sig_c[0], sig_c[1], mask)
    ctl_peak, _ = geo_mean_peak(ctl_c[0], ctl_c[1], mask)
    detect_ratio = sig_peak / ctl_peak if ctl_peak > 0 else float("inf")

    print()
    print("=" * 66)
    print(f"联合定位图 J（{len(sets)} 位姿对比度逐像素相乘）")
    print(f"  深度 FWHM = {jm['depth_fwhm_mm']:6.2f} mm     单位姿自乘对照均值 {sq_mean:6.2f} mm"
          f"   净几何增益 {geo_gain:.2f}×")
    deconv = (float(np.sqrt(max(jm['depth_fwhm_mm'] ** 2 - SMOOTH_FWHM_MM ** 2, 0.0)))
              if np.isfinite(jm['depth_fwhm_mm']) else float("nan"))
    print(f"  扣掉平滑核后    = {deconv:6.2f} mm （平方差；平滑核 FWHM {SMOOTH_FWHM_MM:.2f} mm）")
    print(f"  横向 FWHM = {jm['lateral_fwhm_mm']:6.2f} mm")
    print(f"  峰值位置  = ({jm['peak_x_mm']:+.1f}, {jm['peak_depth_mm']:.1f}) mm"
          f"    真值 ({TRUE_X_MM:.1f}, {TRUE_DEPTH_MM:.1f})   偏差 {offset:.2f} mm")
    print(f"  检出统计量（半统计量）信号 {sig_peak:.4f} / 负对照 {ctl_peak:.4f} = {detect_ratio:.2f}×")
    print("=" * 66)

    # ---- bootstrap ----
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    bd, bx, bz = [], [], []
    bz_single = [[], []]
    for _ in range(BOOTSTRAP_N):
        cs = []
        for v, r in sets:
            wv = rng.multinomial(v["n_rows"], np.full(v["n_rows"], 1.0 / v["n_rows"])).astype(float)
            wr = rng.multinomial(r["n_rows"], np.full(r["n_rows"], 1.0 / r["n_rows"])).astype(float)
            cs.append(significance(sm(make_map(v, wv, grid.shape)), sq(make_map(v, wv, grid.shape, True)),
                                   sm(make_map(r, wr, grid.shape)), sq(make_map(r, wr, grid.shape, True)), mask))
        for k in (0, 1):
            bz_single[k].append(measure(cs[k], grid)["peak_depth_mm"])
        s = measure(joint_localizer(cs[0], cs[1]), grid)
        if np.isfinite(s["depth_fwhm_mm"]):
            bd.append(s["depth_fwhm_mm"])
        bx.append(s["peak_x_mm"]); bz.append(s["peak_depth_mm"])
    q = lambda a: np.percentile(np.asarray(a, float), [2.5, 50, 97.5]).tolist()
    print(f"\nbootstrap N={BOOTSTRAP_N}（深度 FWHM 有效 {len(bd)}）")
    if bd:
        lo, md, hi = q(bd); print(f"  深度 FWHM  中位 {md:.2f}  95%CI [{lo:.2f}, {hi:.2f}] mm")
    lo, md, hi = q(bz); print(f"  峰值深度   中位 {md:.1f}  95%CI [{lo:.1f}, {hi:.1f}] mm  (真值 30.0)")
    lo, md, hi = q(bx); print(f"  峰值 x     中位 {md:+.1f} 95%CI [{lo:+.1f}, {hi:+.1f}] mm  (真值 0.0)")
    joint_ci_w = float(q(bz)[2] - q(bz)[0])
    single_ci_w = [float(q(v)[2] - q(v)[0]) for v in bz_single]
    single_ci_mean = float(np.mean(single_ci_w))
    ci_gain = single_ci_mean / joint_ci_w if joint_ci_w > 0 else float("inf")
    print(f"  峰值深度 95%CI 宽度: 联合 {joint_ci_w:.1f} mm  单位姿 "
          f"{single_ci_w[0]:.1f} / {single_ci_w[1]:.1f} mm  (均值 {single_ci_mean:.1f})"
          f"   -> 多视角收窄 {ci_gain:.2f}×")

    # ---- 交叉校验：镜像背景（不用参照跑）----
    mirror = []
    for m, qmap, (vv, _) in zip(mv, qv, sets):
        L = vv["head_x"]
        xs = grid.x_centers_mm
        idx = np.clip(np.searchsorted(xs, 2.0 * L - xs), 0, len(xs) - 1)
        mirror.append(significance(m, qmap, m[:, idx], qmap[:, idx], mask))
    jm_mirror = measure(joint_localizer(mirror[0], mirror[1]), grid)
    print(f"\n交叉校验（镜像背景，不用参照跑）: 深度 FWHM {jm_mirror['depth_fwhm_mm']:.2f} mm  "
          f"峰值 ({jm_mirror['peak_x_mm']:+.1f}, {jm_mirror['peak_depth_mm']:.1f})")

    # ---- 判据 ----
    checks = [
        (detect_ratio >= CRIT_DETECT_RATIO_MIN,
         f"检出比 {detect_ratio:.2f} >= {CRIT_DETECT_RATIO_MIN}"),
        (np.isfinite(offset) and offset <= CRIT_PEAK_OFFSET_MAX_MM,
         f"峰值偏差 {offset:.2f} <= {CRIT_PEAK_OFFSET_MAX_MM} mm"),
        (np.isfinite(joint_ci_w) and joint_ci_w <= CRIT_JOINT_DEPTH_CI_MAX_MM,
         f"联合峰值深度 95%CI 宽度 {joint_ci_w:.1f} <= {CRIT_JOINT_DEPTH_CI_MAX_MM} mm"),
        (np.isfinite(ci_gain) and ci_gain >= CRIT_CI_GAIN_MIN,
         f"多视角把深度 CI 收窄 {ci_gain:.2f} >= {CRIT_CI_GAIN_MIN}×"),
    ]
    print("\n冻结判据：")
    for ok, text in checks:
        print(f"  [{'通过' if ok else '未通过'}] {text}")
    verdict = all(ok for ok, _ in checks)
    print(f"\n结论：{'S2V 通过 —— 多视角对混凝土体内空洞同样成立' if verdict else 'S2V 未通过，见上面未达标的项'}")

    # ---- 出图 ----
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), constrained_layout=True)
    for ax, img, ttl in zip(axes, (cA, cB, J),
                            (f"位姿 A (L={sets[0][0]['head_x']:+.0f} mm) 亏损显著性",
                             f"位姿 B (L={sets[1][0]['head_x']:+.0f} mm) 亏损显著性",
                             "联合 J = cA x cB")):
        im = ax.imshow(img, origin="upper", aspect="equal", cmap="magma",
                       extent=[grid.x_edges_mm[0], grid.x_edges_mm[-1],
                               grid.depth_edges_mm[-1], grid.depth_edges_mm[0]])
        ax.plot([TRUE_X_MM], [TRUE_DEPTH_MM], "o", mfc="none", mec="cyan", ms=12, mew=1.6)
        ax.set_xlabel("x (mm)"); ax.set_ylabel("深度 (mm)"); ax.set_title(ttl)
        fig.colorbar(im, ax=ax, shrink=0.85)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(out_dir / f"multiview_void.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)

    report = {
        "schema_version": "1.0",
        "purpose": "S2V：混凝土体内空洞的多位姿深度定位",
        "estimator": "per-pose deficit significance (r-v)/sqrt(var), row-wise mask, multiplied pixelwise",
        "truth_use": "post-hoc scoring only",
        "frozen_constants": {
            "mask_ref_fraction": MASK_REF_FRACTION, "mask_depth_mm": list(MASK_DEPTH_MM),
            "detect_ratio_min": CRIT_DETECT_RATIO_MIN, "peak_offset_max_mm": CRIT_PEAK_OFFSET_MAX_MM,
            "joint_depth_ci_max_mm": CRIT_JOINT_DEPTH_CI_MAX_MM, "ci_gain_min": CRIT_CI_GAIN_MIN,
            "smooth_sigma_mm": SMOOTH_SIGMA_MM,
        },
        "grid": {"x_range_mm": args.x_range_mm, "depth_max_mm": args.depth_max_mm, "bin_mm": b},
        "mask_pixels": int(mask.sum()),
        "poses": [{"head_x_mm": v["head_x"], "void_dir": str(pv), "ref_dir": str(pr),
                   "void_selected": v["n_sel"], "ref_selected": r["n_sel"],
                   "single_pose_significance": s}
                  for (v, r), (pv, pr), s in zip(sets, args.pair, single)],
        "smoothing": {"sigma_mm": SMOOTH_SIGMA_MM, "fwhm_mm": SMOOTH_FWHM_MM},
        "joint": {**jm, "peak_offset_mm": offset, "depth_fwhm_deconvolved_mm": deconv,
                  "squaring_control_depth_fwhm_mm": sq_ctrl, "squaring_control_mean_mm": sq_mean,
                  "net_geometric_gain": geo_gain},
        "detection": {"signal_peak": sig_peak, "control_peak": ctl_peak, "ratio": detect_ratio},
        "bootstrap": {"n": BOOTSTRAP_N, "seed": BOOTSTRAP_SEED,
                      "depth_fwhm_mm_ci": q(bd) if bd else None,
                      "peak_depth_mm_ci": q(bz), "peak_x_mm_ci": q(bx),
                      "joint_depth_ci_width_mm": joint_ci_w,
                      "single_pose_depth_ci_width_mm": single_ci_w,
                      "ci_gain": ci_gain},
        "mirror_cross_check": jm_mirror,
        "criteria": {text: bool(ok) for ok, text in checks},
        "verdict": bool(verdict),
    }
    (out_dir / "multiview_void_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    np.savez_compressed(out_dir / "multiview_void_maps.npz",
                        cA=cA, cB=cB, joint=J, mask=mask,
                        x_centers_mm=grid.x_centers_mm, depth_centers_mm=grid.depth_centers_mm)
    print(f"\n结果写入 {out_dir}")
    return 0 if verdict else 2


if __name__ == "__main__":
    sys.exit(main())
