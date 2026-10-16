"""S2 窄版：多位姿联合重建，检验深度分辨率能否从单视角的约 38 mm 降到毫米级。

原理
----
单个位姿下，每个事件的定位误差是一根**沿入射射线方向的"针"**：垂直射线方向很准
（横向约 3.7 mm），沿射线方向很差（深度约 38 mm）。源是点源，所以对某个体素而言
针的方向是固定的，单视角内怎么重建都改不了。

把探测头平移到另一个位姿，同一体素的射线方向就变了，针也跟着转。两根方向不同的
针交叉，交点在各个方向都小——这就是导师提出的思路。

**关键：不能把两个位姿的事件简单合并。** 那样得到的是两根针的并集（X 形），图不会
变锐。要得到交集必须做似然相乘：每个位姿各生成一张概率图，再逐像素相乘。

实现
----
概率图直接用项目已有的 backscatter.iterative.build_ray_likelihood_system 生成——
它是逐事件读取源位置的，平移位姿天生支持，不需要改动。

判据（在看到结果之前冻结）
--------------------------
  1. 联合图深度 FWHM <= 10 mm        （解析预测 5.1 mm）
  2. 单位姿深度 FWHM / 联合 >= 3     （证明增益来自多视角本身）
  3. 联合图横向 FWHM <= 5 mm         （不为深度牺牲横向）
  4. 联合图峰值距真值 (0, 30) mm <= 5 mm
细杆真值不参与重建或选参，只用于事后打分。

用法
----
  E:\\python3.12\\python.exe scripts\\reconstruct_multiview_rod.py ^
      --pose runs\\s2_rod_L-27_5m_seed24083201\\canonical_response ^
      --pose runs\\s2_rod_L27_5m_seed24083202\\canonical_response
可选 --pose 传更多位姿；--out-dir 默认 outputs\\multiview_s2_<日期>
"""
from __future__ import annotations

import argparse
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

import copy

DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "transverse_fwhm_5mm_frozen.json"
TRUE_X_MM, TRUE_DEPTH_MM = 0.0, 30.0

# 冻结判据
CRIT_JOINT_DEPTH_MAX_MM = 10.0
CRIT_GAIN_MIN = 3.0
CRIT_JOINT_LATERAL_MAX_MM = 5.0
CRIT_PEAK_OFFSET_MAX_MM = 5.0


def observable_selection(pairs, config):
    """位姿感知筛选。

    pipeline.build_selection 的 D1 有效半径与 D2 有效面积是以**世界原点**为中心
    写的，探测头平移后会在错误的中心上切割（L=-27 mm 实测把 154 个事件砍成 0）。
    这里改用 backscatter.pose_selection，以逐事件的探测头轴心为中心；L=0 时两者
    逐位相同，历史结果不受影响。
    """
    cfg = copy.deepcopy(config)
    cfg["selection"]["phantom_truth_policy"] = "ignore"
    selected, _, _ = build_selection_pose_aware(pairs, cfg)
    return selected


def load_npz(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as d:
        return {k: d[k] for k in d.files}


def fwhm_of_profile(coord: np.ndarray, value: np.ndarray) -> tuple[float, float]:
    """峰两侧半高线性插值。返回 (FWHM, 峰位)。"""
    v = np.asarray(value, dtype=np.float64)
    if not np.any(np.isfinite(v)) or np.nanmax(v) <= 0:
        return float("nan"), float("nan")
    v = np.where(np.isfinite(v), v, 0.0)
    base = np.median(np.concatenate([v[:5], v[-5:]]))
    v = v - base
    peak = int(np.argmax(v))
    half = v[peak] / 2.0
    if half <= 0:
        return float("nan"), float(coord[peak])

    def cross(idx_iter):
        prev = peak
        for i in idx_iter:
            if v[i] <= half:
                x0, x1, y0, y1 = coord[prev], coord[i], v[prev], v[i]
                return float(x1 if y0 == y1 else x0 + (half - y0) * (x1 - x0) / (y1 - y0))
            prev = i
        return float("nan")

    left = cross(range(peak - 1, -1, -1))
    right = cross(range(peak + 1, len(v)))
    if not (np.isfinite(left) and np.isfinite(right)):
        return float("nan"), float(coord[peak])
    return right - left, float(coord[peak])


def profiles(m: np.ndarray, grid, half_window_mm: float = 3.0):
    """过峰值取剖面：深度剖面在峰的 x 附近平均，横向剖面在峰的深度附近平均。"""
    x, z = grid.x_centers_mm, grid.depth_centers_mm
    flat = int(np.nanargmax(m))
    iz, ix = np.unravel_index(flat, m.shape)
    kx = np.abs(x - x[ix]) <= half_window_mm
    kz = np.abs(z - z[iz]) <= half_window_mm
    depth_prof = np.nanmean(m[:, kx], axis=1)
    lat_prof = np.nanmean(m[kz, :], axis=0)
    return (z, depth_prof), (x, lat_prof), (float(x[ix]), float(z[iz]))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pose", type=Path, action="append", required=True,
                    help="每个位姿的 canonical_response 目录，至少两个")
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--x-range-mm", type=float, default=60.0)
    ap.add_argument("--depth-max-mm", type=float, default=80.0)
    ap.add_argument("--bin-mm", type=float, default=1.0)
    args = ap.parse_args()

    if len(args.pose) < 2:
        ap.error("至少要两个 --pose")
    out_dir = args.out_dir or (PROJECT_ROOT / "outputs" /
                               f"multiview_s2_{date.today():%Y%m%d}")
    out_dir.mkdir(parents=True, exist_ok=True)

    config = load_config(args.config)
    b = args.bin_mm
    grid = ReconstructionGrid2D(
        x_edges_mm=np.arange(-args.x_range_mm, args.x_range_mm + b, b),
        depth_edges_mm=np.arange(0.0, args.depth_max_mm + b, b),
    )
    model = {
        "incident_energy_keV": float(config["simulation"]["incident_energy_keV"]),
        "electron_rest_energy_keV": float(config["simulation"]["electron_rest_energy_keV"]),
        "d1_energy_fwhm_fraction_at_511": float(
            config["detectors"]["d1"]["energy_response"]["fwhm_fraction_at_511"]),
        "d1_position_resolution_mm": tuple(
            config["detectors"]["d1"]["position_response"]["resolution_mm"]),
        "d2_position_resolution_mm": tuple(
            config["detectors"]["d2"]["position_response"]["resolution_mm"]),
        "y_half_width_mm": 150.0,
        "likelihood_sigma_limit": 4.0,
        "doppler_sigma_keV": 0.0,
        "include_geometric_weight": True,
    }

    print(f"重建网格: x ±{args.x_range_mm:.0f} mm, 深度 0–{args.depth_max_mm:.0f} mm, "
          f"像素 {b:.1f} mm  -> {grid.shape}")
    print()

    maps, rows = [], []
    for i, pose_dir in enumerate(args.pose, start=1):
        pose_dir = pose_dir.resolve()
        pairs = load_npz(pose_dir / "pairs.npz")
        sel = observable_selection(pairs, config)
        src_x = np.asarray(pairs["source_x_mm"], dtype=np.float64)
        head_x = float(np.median(src_x))
        print(f"[{i}/{len(args.pose)}] {pose_dir.name}")
        print(f"    探测头位姿 L = {head_x:+.2f} mm （由逐事件源位置中位数得到）")
        print(f"    通过筛选的事件: {int(sel.sum())}")
        if int(sel.sum()) == 0:
            print("    !! 该位姿筛选后为 0 个事件，无法重建。")
            print("       先检查 D1/D2 几何判据是否以探测头轴心为中心，"
                  "再检查目标是否落在照射锥外。")
            return 1
        system = build_ray_likelihood_system(pairs, sel, grid, **model)
        m = np.asarray(system.matrix.sum(axis=0)).ravel().reshape(grid.shape)
        s = m.sum()
        if s <= 0:
            print("    !! 概率图全零，该位姿无法使用")
            return 1
        m = m / s
        maps.append(m)
        (zc, dp), (xc, lp), (px, pz) = profiles(m, grid)
        d_fwhm, _ = fwhm_of_profile(zc, dp)
        l_fwhm, _ = fwhm_of_profile(xc, lp)
        print(f"    单位姿 深度 FWHM = {d_fwhm:6.2f} mm   横向 FWHM = {l_fwhm:6.2f} mm"
              f"   峰值 ({px:+.1f}, {pz:.1f})")
        rows.append({"pose_dir": str(pose_dir), "head_x_mm": head_x,
                     "n_selected": int(sel.sum()), "depth_fwhm_mm": d_fwhm,
                     "lateral_fwhm_mm": l_fwhm, "peak_x_mm": px, "peak_depth_mm": pz})
        print()

    # ---- 联合：逐像素相乘（似然相交，不是事件合并）----
    joint = np.ones_like(maps[0])
    for m in maps:
        joint = joint * (m / m.max())
    joint = joint / joint.sum()
    (zc, jdp), (xc, jlp), (jpx, jpz) = profiles(joint, grid)
    j_depth, _ = fwhm_of_profile(zc, jdp)
    j_lat, _ = fwhm_of_profile(xc, jlp)
    single_depth = float(np.nanmean([r["depth_fwhm_mm"] for r in rows]))
    gain = single_depth / j_depth if j_depth and np.isfinite(j_depth) else float("nan")
    offset = float(np.hypot(jpx - TRUE_X_MM, jpz - TRUE_DEPTH_MM))

    print("=" * 62)
    print(f"联合重建（{len(maps)} 个位姿，似然逐像素相乘）")
    print(f"  深度 FWHM = {j_depth:6.2f} mm      单位姿平均 {single_depth:6.2f} mm"
          f"      增益 {gain:5.2f}×")
    print(f"  横向 FWHM = {j_lat:6.2f} mm")
    print(f"  峰值位置  = ({jpx:+.1f}, {jpz:.1f}) mm    真值 (0.0, 30.0)   偏差 {offset:.2f} mm")
    print("=" * 62)

    checks = [
        (f"联合深度 FWHM {j_depth:.2f} <= {CRIT_JOINT_DEPTH_MAX_MM} mm",
         np.isfinite(j_depth) and j_depth <= CRIT_JOINT_DEPTH_MAX_MM),
        (f"增益 {gain:.2f} >= {CRIT_GAIN_MIN}",
         np.isfinite(gain) and gain >= CRIT_GAIN_MIN),
        (f"联合横向 FWHM {j_lat:.2f} <= {CRIT_JOINT_LATERAL_MAX_MM} mm",
         np.isfinite(j_lat) and j_lat <= CRIT_JOINT_LATERAL_MAX_MM),
        (f"峰值偏差 {offset:.2f} <= {CRIT_PEAK_OFFSET_MAX_MM} mm",
         offset <= CRIT_PEAK_OFFSET_MAX_MM),
    ]
    print("\n冻结判据：")
    for text, ok in checks:
        print(f"  [{'通过' if ok else '未通过'}] {text}")
    verdict = all(ok for _, ok in checks)
    print(f"\n结论：{'S2 通过 —— 多视角确实把深度带进毫米级' if verdict else 'S2 未通过，见上面未达标的项'}")

    # ---- 出图 ----
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "DejaVu Sans"],
        "font.size": 9.5, "axes.spines.top": False, "axes.spines.right": False})
    n = len(maps)
    fig, axes = plt.subplots(1, n + 2, figsize=(3.4 * (n + 2), 3.8))
    extent = [grid.x_edges_mm[0], grid.x_edges_mm[-1],
              grid.depth_edges_mm[-1], grid.depth_edges_mm[0]]
    for ax, m, r in zip(axes[:n], maps, rows):
        ax.imshow(m / m.max(), extent=extent, cmap="magma", aspect="equal",
                  interpolation="nearest", vmin=0, vmax=1)
        ax.set_title(f"位姿 L={r['head_x_mm']:+.0f} mm\n深度 FWHM {r['depth_fwhm_mm']:.1f} mm",
                     fontsize=9.5, loc="left")
    ax = axes[n]
    ax.imshow(joint / joint.max(), extent=extent, cmap="magma", aspect="equal",
              interpolation="nearest", vmin=0, vmax=1)
    ax.set_title(f"联合（似然相乘）\n深度 FWHM {j_depth:.1f} mm", fontsize=9.5, loc="left")
    for ax in axes[:n + 1]:
        ax.plot([TRUE_X_MM], [TRUE_DEPTH_MM], marker="+", ms=10, mew=1.4, color="#6EE7F2")
        ax.set_xlim(-30, 30); ax.set_ylim(60, 5)
        ax.set_xlabel("横向 x（mm）"); ax.set_ylabel("深度（mm）")
    ax = axes[n + 1]
    for m, r in zip(maps, rows):
        (zc2, dp2), _, _ = profiles(m, grid)
        ax.plot(zc2, dp2 / dp2.max(), lw=1.2, alpha=0.75,
                label=f"L={r['head_x_mm']:+.0f}  {r['depth_fwhm_mm']:.0f} mm")
    ax.plot(zc, jdp / jdp.max(), lw=2.2, color="#B3261E",
            label=f"联合  {j_depth:.1f} mm")
    ax.axvline(TRUE_DEPTH_MM, color="#6B7280", ls=":", lw=1.1)
    ax.set_xlim(0, 70); ax.set_ylim(0, 1.1)
    ax.set_xlabel("深度（mm）"); ax.set_ylabel("归一化深度剖面")
    ax.set_title("深度剖面对比", fontsize=9.5, loc="left")
    ax.legend(fontsize=8.2)
    fig.suptitle("S2：多位姿联合重建对深度分辨率的影响", fontsize=11.5,
                 fontweight="bold", x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    for ext in ("png", "pdf", "svg"):
        fig.savefig(out_dir / f"multiview_depth.{ext}", dpi=200,
                    bbox_inches="tight", facecolor="white")

    np.savez_compressed(out_dir / "multiview_maps.npz",
                        x_edges_mm=grid.x_edges_mm, depth_edges_mm=grid.depth_edges_mm,
                        joint=joint, **{f"pose_{i}": m for i, m in enumerate(maps)})
    with (out_dir / "multiview_report.json").open("w", encoding="utf-8") as fh:
        json.dump({
            "schema_version": "1.0",
            "purpose": "S2 窄版：多位姿似然相乘联合重建的深度分辨率",
            "combination": "per-pose normalized backprojection maps multiplied pixelwise",
            "truth_use": "post-hoc scoring only",
            "grid": {"x_range_mm": args.x_range_mm, "depth_max_mm": args.depth_max_mm,
                     "bin_mm": b},
            "poses": rows,
            "joint": {"depth_fwhm_mm": j_depth, "lateral_fwhm_mm": j_lat,
                      "peak_x_mm": jpx, "peak_depth_mm": jpz,
                      "peak_offset_mm": offset,
                      "single_pose_mean_depth_fwhm_mm": single_depth,
                      "gain": gain},
            "criteria": {t: bool(o) for t, o in checks},
            "verdict": bool(verdict),
        }, fh, ensure_ascii=False, indent=2)
    print(f"\n结果写入 {out_dir}")
    return 0 if verdict else 2


if __name__ == "__main__":
    sys.exit(main())
