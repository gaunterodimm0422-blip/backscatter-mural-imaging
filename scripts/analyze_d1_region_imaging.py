"""工作一：最小可行 D1 的成像验证（纯后处理，不需要新的 GATE 模拟）。

问题：把 D1 从 R=20-200 mm 的 GAGG 整环（12.37 kg）缩到实验室买得起的小模块，
孔洞成像还能不能成立？效率保留不等于成像质量保留，必须逐个重建来看。

做法：复用 scripts/reconstruct_void_advanced.py 的同一条闭式重建 + TV 链路，
只在事件层面按 D1 命中的半径和方位角切子集，其余处理完全不变。

关键设计：
  * 分析掩膜（有效视野、ROI、控制环、场区）固定为整环定义，
    这样各档之间的指标可直接比较，且整环一档必须复现官方数字。
  * 每一档额外报告"该档自身参考计数能通过 3% 阈值的像素数"，作为视野收缩指标。
  * 全程不使用孔洞真值参与筛选、重建或选参；真值仅用于事后打分。

输出：outputs/d1_region_imaging_20260830/
用法（在项目根目录）：python scripts/analyze_d1_region_imaging.py
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import reconstruct_void_advanced as adv          # 复用官方链路
from backscatter.config import load_config
from backscatter.iterative import (
    ReconstructionGrid2D,
    build_ray_likelihood_system,
    denoise_tv_chambolle,
)

OUT_DIR = PROJECT_ROOT / "outputs" / "d1_region_imaging_20260830"
ROD_DIR = (
    PROJECT_ROOT / "outputs" / "transverse_fwhm_5mm_20260805" / "independent_gate_seed"
    / "rod_width2_depth30_5m_seed24080573_local_fwhm5_validation" / "canonical_response"
)
TV_WEIGHT = 0.12
D1_THICKNESS_CM = 1.5
GAGG_DENSITY = 6.63          # g/cm^3
SOURCE_ACTIVITY_BQ = 950e3   # 实测源活度
CONE_FRACTION = (1.0 - np.cos(np.radians(45.0))) / 2.0
BETA_PLUS_BRANCH = 0.903

# (标签, R内, R外, 方位角规格)
# 方位角规格 = None 表示整周；否则 (中心角 deg, 全宽 deg, 是否对置双扇区)
REGIONS = [
    ("整环 R20-200（验证基准）",        20.0, 200.0, None),
    ("整环 R20-100",                    20.0, 100.0, None),
    ("整环 R20-75",                     20.0,  75.0, None),
    ("整环 R20-50",                     20.0,  50.0, None),
    ("整环 R50-100",                    50.0, 100.0, None),
    ("整环 R100-200",                  100.0, 200.0, None),
    ("R20-50 半环 中心0deg(+x侧)",      20.0,  50.0, (0.0, 180.0, False)),
    ("R20-50 半环 中心90deg(+y侧)",     20.0,  50.0, (90.0, 180.0, False)),
    ("R20-50 对置双90deg扇区 沿x",      20.0,  50.0, (0.0, 90.0, True)),
    ("R20-50 对置双90deg扇区 沿y",      20.0,  50.0, (90.0, 90.0, True)),
    ("R20-50 单90deg扇区 中心0deg",     20.0,  50.0, (0.0, 90.0, False)),
    ("R20-50 单90deg扇区 中心90deg",    20.0,  50.0, (90.0, 90.0, False)),
    ("R20-50 单45deg扇区 中心0deg",     20.0,  50.0, (0.0, 45.0, False)),
    ("R20-75 单90deg扇区 中心0deg",     20.0,  75.0, (0.0, 90.0, False)),
]


def region_mask(pairs, r_in, r_out, sector):
    radius = np.asarray(pairs["d1_radius_mm"], dtype=np.float64)
    mask = (radius >= r_in) & (radius < r_out)
    if sector is None:
        return mask
    center, width, opposed = sector
    phi = np.degrees(np.arctan2(
        np.asarray(pairs["d1_y_mm"], dtype=np.float64),
        np.asarray(pairs["d1_x_mm"], dtype=np.float64),
    ))
    delta = np.abs((phi - center + 180.0) % 360.0 - 180.0)
    inside = delta <= width / 2.0
    if opposed:
        delta2 = np.abs((phi - center - 180.0 + 180.0) % 360.0 - 180.0)
        inside = inside | (delta2 <= width / 2.0)
    return mask & inside


def region_geometry(r_in, r_out, sector):
    fraction = 1.0 if sector is None else (
        (2.0 if sector[2] else 1.0) * sector[1] / 360.0
    )
    area_cm2 = np.pi * (r_out ** 2 - r_in ** 2) / 100.0 * fraction
    volume_cm3 = area_cm2 * D1_THICKNESS_CM
    return area_cm2, volume_cm3, volume_cm3 * GAGG_DENSITY / 1000.0


def transverse_fwhm(x_mm, values):
    """在峰两侧做半高线性插值，返回 FWHM(mm)；失败返回 nan。"""
    values = np.asarray(values, dtype=np.float64)
    if not np.any(np.isfinite(values)) or np.nanmax(values) <= 0:
        return float("nan")
    peak = int(np.nanargmax(values))
    half = values[peak] / 2.0

    def cross(indices):
        previous = peak
        for i in indices:
            if not np.isfinite(values[i]):
                return float("nan")
            if values[i] <= half:
                x0, x1 = x_mm[previous], x_mm[i]
                y0, y1 = values[previous], values[i]
                if y0 == y1:
                    return float(x1)
                return float(x0 + (half - y0) * (x1 - x0) / (y1 - y0))
            previous = i
        return float("nan")

    left = cross(range(peak - 1, -1, -1))
    right = cross(range(peak + 1, len(values)))
    if not np.isfinite(left) or not np.isfinite(right):
        return float("nan")
    return right - left


def rod_transverse_fwhm(rod_pairs, rod_recon, selected, subset):
    usable = selected & rod_recon["valid"].astype(bool) & subset
    x = np.asarray(rod_recon["x_mm"], dtype=np.float64)
    depth = -np.asarray(rod_recon["z_mm"], dtype=np.float64)
    y = np.asarray(rod_recon["y_mm"], dtype=np.float64)
    usable &= np.isfinite(x) & np.isfinite(depth) & (np.abs(y) <= 150.0)
    usable &= np.abs(depth - 30.0) <= 5.0
    if usable.sum() < 50:
        return float("nan"), int(usable.sum())
    edges = np.arange(-30.0, 30.0 + 1.0, 1.0)
    counts, _ = np.histogram(x[usable], bins=edges)
    centers = 0.5 * (edges[:-1] + edges[1:])
    kernel = np.exp(-0.5 * (np.arange(-4, 5) / 1.5) ** 2)
    kernel /= kernel.sum()
    smoothed = np.convolve(counts.astype(float), kernel, mode="same")
    baseline = np.median(np.concatenate([smoothed[:8], smoothed[-8:]]))
    return transverse_fwhm(centers, smoothed - baseline), int(usable.sum())


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    config = load_config(adv.CONFIG_PATH)
    grid = ReconstructionGrid2D(
        x_edges_mm=np.arange(-80.0, 80.0 + 2.0, 2.0),
        depth_edges_mm=np.arange(0.0, 100.0 + 2.0, 2.0),
    )

    print("载入事件数据 ...", flush=True)
    void_pairs = adv._load_npz(adv.VOID_DIR / "pairs.npz")
    ref_pairs = adv._load_npz(adv.REFERENCE_DIR / "pairs.npz")
    void_recon = adv._load_npz(adv.VOID_DIR / "reconstruction.npz")
    ref_recon = adv._load_npz(adv.REFERENCE_DIR / "reconstruction.npz")
    void_selected = adv._observable_selection(void_pairs, config)
    ref_selected = adv._observable_selection(ref_pairs, config)
    ref_even = np.asarray(ref_pairs["event_id"], dtype=np.int64) % 2 == 0
    ref_odd = ~ref_even

    void_exposure = float(adv._load_primary_count(adv.VOID_DIR))
    ref_exposure = float(adv._load_primary_count(adv.REFERENCE_DIR))

    print("构建整环参考灵敏度（用于固定分析掩膜，只做一次）...", flush=True)
    common_model = {
        "incident_energy_keV": float(config["simulation"]["incident_energy_keV"]),
        "electron_rest_energy_keV": float(
            config["simulation"]["electron_rest_energy_keV"]),
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
    train_system = build_ray_likelihood_system(
        ref_pairs, ref_selected & ref_even, grid, **common_model)
    holdout_system = build_ray_likelihood_system(
        ref_pairs, ref_selected & ref_odd, grid, **common_model)
    ref_backprojection = (
        np.asarray(train_system.matrix.sum(axis=0)).ravel()
        + np.asarray(holdout_system.matrix.sum(axis=0)).ravel()
    ).reshape(grid.shape)
    masks = adv._analysis_masks(
        grid, ref_backprojection,
        void_x_mm=float(config["phantom"]["void"]["center_mm"][0]),
        void_depth_mm=float(config["phantom"]["void"]["depth_mm"]),
        void_radius_mm=float(config["phantom"]["void"]["radius_mm"]),
    )

    print("载入独立细杆数据 ...", flush=True)
    rod_pairs = adv._load_npz(ROD_DIR / "pairs.npz")
    rod_recon = adv._load_npz(ROD_DIR / "reconstruction.npz")
    rod_selected = adv._observable_selection(rod_pairs, config)

    rows = []
    maps = {}
    for index, (label, r_in, r_out, sector) in enumerate(REGIONS, start=1):
        print(f"[{index}/{len(REGIONS)}] {label}", flush=True)
        v_mask = region_mask(void_pairs, r_in, r_out, sector)
        r_mask = region_mask(ref_pairs, r_in, r_out, sector)
        d_mask = region_mask(rod_pairs, r_in, r_out, sector)

        v_counts = adv._baseline_point_histogram(
            void_pairs, void_recon, void_selected, grid, event_subset=v_mask)
        r_counts = adv._baseline_point_histogram(
            ref_pairs, ref_recon, ref_selected, grid, event_subset=r_mask)
        train_counts = adv._baseline_point_histogram(
            ref_pairs, ref_recon, ref_selected, grid, event_subset=r_mask & ref_even)
        holdout_counts = adv._baseline_point_histogram(
            ref_pairs, ref_recon, ref_selected, grid, event_subset=r_mask & ref_odd)

        density = adv._safe_relative_density(
            v_counts, r_counts,
            target_exposure=void_exposure, reference_exposure=ref_exposure,
            smoothing_sigma_bins=1.0, minimum_reference_fraction_of_depth_max=0.03)
        control_density = adv._safe_relative_density(
            holdout_counts, train_counts,
            target_exposure=ref_exposure / 2.0, reference_exposure=ref_exposure / 2.0,
            smoothing_sigma_bins=1.0, minimum_reference_fraction_of_depth_max=0.03)

        baseline_metrics = adv._image_metrics(density, masks, grid)
        tv_metrics = adv._image_metrics(
            adv._tv_filter_density(density, weight=TV_WEIGHT), masks, grid)
        tv_control_metrics = adv._image_metrics(
            adv._tv_filter_density(control_density, weight=TV_WEIGHT), masks, grid)

        n_void = int((void_selected & v_mask & void_recon["valid"].astype(bool)).sum())
        n_ref = int((ref_selected & r_mask & ref_recon["valid"].astype(bool)).sum())
        area, volume, mass = region_geometry(r_in, r_out, sector)
        eff_primary = n_ref / ref_exposure
        rate = SOURCE_ACTIVITY_BQ * eff_primary * CONE_FRACTION * BETA_PLUS_BRANCH
        fov_pixels = int(np.isfinite(density).sum())
        rod_fwhm, rod_n = rod_transverse_fwhm(
            rod_pairs, rod_recon, rod_selected, d_mask)

        control_roi = tv_control_metrics["deficit"][
            masks["roi"] & np.isfinite(tv_control_metrics["deficit"])]

        rows.append({
            "region": label,
            "r_inner_mm": r_in, "r_outer_mm": r_out,
            "sector_center_deg": "" if sector is None else sector[0],
            "sector_width_deg": "" if sector is None else sector[1],
            "sector_opposed": "" if sector is None else int(sector[2]),
            "area_cm2": round(area, 1),
            "gagg_volume_cm3": round(volume, 1),
            "gagg_mass_kg": round(mass, 3),
            "n_void_events": n_void,
            "n_reference_events": n_ref,
            "efficiency_per_primary": round(eff_primary, 6),
            "true_coincidences_per_s_at_950kBq": round(rate, 1),
            "fov_pixels": fov_pixels,
            "separation_no_tv": _r(baseline_metrics["image_domain_standardized_separation"]),
            "separation_tv012": _r(tv_metrics["image_domain_standardized_separation"]),
            "contrast_tv012": _r(tv_metrics["image_domain_contrast"]),
            "centroid_x_mm": _r(tv_metrics["positive_deficit_centroid_x_mm"]),
            "centroid_depth_mm": _r(tv_metrics["positive_deficit_centroid_depth_mm"]),
            "apparent_width_mm": _r(
                tv_metrics["profile_width"].get("apparent_half_max_width_mm")),
            "field_deficit_rms_tv012": _r(tv_metrics["field_deficit_rms"]),
            "control_roi_false_deficit": _r(float(np.mean(control_roi)) if control_roi.size else None),
            "rod_transverse_fwhm_mm": _r(rod_fwhm),
            "rod_events": rod_n,
        })
        maps[label] = tv_metrics["deficit"]

    fields = list(rows[0].keys())
    with (OUT_DIR / "d1_region_imaging_metrics.csv").open(
            "w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    with (OUT_DIR / "d1_region_imaging_report.json").open(
            "w", encoding="utf-8") as handle:
        json.dump({
            "schema_version": "1.0",
            "purpose": "最小可行 D1 的成像验证；固定整环分析掩膜，逐区域重建。",
            "tv_weight": TV_WEIGHT,
            "analysis_mask": "fixed at full-ring definition so regions are comparable",
            "truth_use": "post-hoc scoring only; not used for selection or reconstruction",
            "source_activity_Bq": SOURCE_ACTIVITY_BQ,
            "rate_model": "rate = A * eff_per_cone_primary * 0.1464 * 0.903",
            "rows": rows,
        }, handle, ensure_ascii=False, indent=2)

    np.savez_compressed(
        OUT_DIR / "d1_region_deficit_maps.npz",
        x_edges_mm=grid.x_edges_mm, depth_edges_mm=grid.depth_edges_mm,
        **{f"deficit_{i}": maps[label] for i, (label, *_rest) in enumerate(REGIONS)},
    )

    _figure(rows, maps, grid)

    print("\n=== 整环一档与官方结果对照（应当一致）===")
    top = rows[0]
    print(f"  分离度 无TV : {top['separation_no_tv']}   官方 1.80")
    print(f"  分离度 TV0.12: {top['separation_tv012']}   官方 2.84")
    print(f"  重心          : ({top['centroid_x_mm']}, {top['centroid_depth_mm']})   官方 (0.5, 31.0)")
    print(f"  参考事件数    : {top['n_reference_events']}")
    print(f"\n结果已写入 {OUT_DIR}")


def _r(value, digits=4):
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return ""
    return round(float(value), digits)


def _figure(rows, maps, grid):
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Microsoft YaHei", "SimHei", "Noto Sans CJK JP", "DejaVu Sans"],
        "font.size": 9.5, "axes.spines.top": False, "axes.spines.right": False,
    })
    mass = [r["gagg_mass_kg"] for r in rows]
    sep = [r["separation_tv012"] if r["separation_tv012"] != "" else np.nan for r in rows]
    rate = [r["true_coincidences_per_s_at_950kBq"] for r in rows]

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))
    ax = axes[0]
    ax.scatter(mass, sep, s=48, color="#1F4E9C", zorder=3)
    for r, m, s in zip(rows, mass, sep):
        ax.annotate(r["region"], (m, s), fontsize=6.6, xytext=(4, 4),
                    textcoords="offset points")
    ax.set_xscale("log")
    ax.set_xlabel("D1 晶体质量（kg，对数轴）")
    ax.set_ylabel("孔洞图像域标准化分离度（TV 0.12）")
    ax.set_title("晶体用量 → 成像质量", loc="left")
    ax.axhline(2.0, color="#9CA3AF", linestyle="--", linewidth=0.9)
    ax.text(min(mass), 2.03, "分离度 2.0 参考线", fontsize=7.5, color="#6B7280")

    ax = axes[1]
    ax.scatter(mass, rate, s=48, color="#B45309", zorder=3)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("D1 晶体质量（kg，对数轴）")
    ax.set_ylabel("950 kBq 下真符合率（/s，对数轴）")
    ax.set_title("晶体用量 → 计数率", loc="left")
    fig.tight_layout()
    for ext in ("png", "pdf", "svg"):
        fig.savefig(OUT_DIR / f"d1_region_tradeoff.{ext}", dpi=220,
                    bbox_inches="tight", facecolor="white")

    order = [i for i, _ in enumerate(REGIONS)]
    cols = 5
    rows_n = int(np.ceil(len(order) / cols))
    fig, axes = plt.subplots(rows_n, cols, figsize=(3.1 * cols, 2.7 * rows_n))
    extent = [grid.x_edges_mm[0], grid.x_edges_mm[-1],
              grid.depth_edges_mm[-1], grid.depth_edges_mm[0]]
    cmap = matplotlib.colormaps["RdBu_r"].copy(); cmap.set_bad("white")
    for ax, (label, *_rest) in zip(np.ravel(axes), REGIONS):
        ax.imshow(maps[label], extent=extent, cmap=cmap, vmin=-0.25, vmax=0.25,
                  aspect="equal", interpolation="nearest")
        ax.set_xlim(-35, 35); ax.set_ylim(55, 7)
        ax.set_title(label, fontsize=7.5, loc="left")
        ax.tick_params(labelsize=7)
    for ax in np.ravel(axes)[len(REGIONS):]:
        ax.axis("off")
    fig.suptitle("各 D1 区域的相对计数缺失图（同一色标，固定掩膜）", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    for ext in ("png", "pdf"):
        fig.savefig(OUT_DIR / f"d1_region_gallery.{ext}", dpi=200,
                    bbox_inches="tight", facecolor="white")


if __name__ == "__main__":
    main()
