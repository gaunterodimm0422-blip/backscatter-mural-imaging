"""S1：测量细杆在 x / y / 深度三个方向的线扩散函数 FWHM。

为什么要这个脚本：项目原有的 evaluate-standard --kind rod 只算 x 和深度两个方向
（因为原细杆沿 y 放，y 方向没有结构）。要拿到 y 方向的数就得单独算。

本脚本直接复用 backscatter.standards 里的 estimate_fwhm_kde 和 bootstrap_fwhm，
方法与官方 x 方向那次完全一致，所以三个方向的数字可以直接互相比较。

用法（Windows，本机无全局 Python）：
  自检 —— 在已有的沿 y 细杆数据上跑，x 方向应复现 3.67 mm：
    E:\\python3.12\\python.exe scripts\\analyze_rod_axes.py --selfcheck
  测新数据：
    E:\\python3.12\\python.exe scripts\\analyze_rod_axes.py --analysis-dir runs\\rod_x_depth30_...
  两个一起（推荐，能直接对比）：
    E:\\python3.12\\python.exe scripts\\analyze_rod_axes.py --selfcheck --analysis-dir runs\\rod_x_depth30_...

只读，不写任何结果文件。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backscatter.standards import bootstrap_fwhm, estimate_fwhm_kde

# 官方沿 y 细杆（测 x 方向）的数据，用作方法自检
REFERENCE_DIR = (
    PROJECT_ROOT / "outputs" / "transverse_fwhm_5mm_20260805" / "independent_gate_seed"
    / "rod_width2_depth30_5m_seed24080573_local_fwhm5_validation" / "canonical_response"
)
REFERENCE_X_FWHM_MM = 3.67          # 官方值，自检要对上
N_BOOTSTRAP = 200
BANDWIDTH_SCALE = 1.0


def load_axes(analysis_dir: Path) -> dict:
    npz = analysis_dir / "reconstruction.npz"
    if not npz.exists():
        raise FileNotFoundError(
            f"{npz} 不存在。这个目录必须是 python -m backscatter analyze 的输出。")
    with np.load(npz, allow_pickle=False) as data:
        rec = {k: data[k] for k in data.files}
    usable = rec["selected"].astype(bool) & rec["valid"].astype(bool)
    n_prim = None
    summary = analysis_dir / "summary.json"
    if summary.exists():
        with summary.open(encoding="utf-8") as fh:
            n_prim = json.load(fh).get("event_counts", {}).get("primaries")
    return {
        "n_primaries": n_prim,
        "n_usable": int(usable.sum()),
        "x": np.asarray(rec["x_mm"][usable], dtype=np.float64),
        "y": np.asarray(rec["y_mm"][usable], dtype=np.float64),
        "depth": -np.asarray(rec["z_mm"][usable], dtype=np.float64),
    }


def measure(values: np.ndarray) -> tuple[float, tuple[float, float] | None, str]:
    if values.size < 30:
        return float("nan"), None, "too_few_events"
    est = estimate_fwhm_kde(values, bandwidth_scale=BANDWIDTH_SCALE)
    boot = bootstrap_fwhm(values, n_bootstrap=N_BOOTSTRAP,
                          bandwidth_scale=BANDWIDTH_SCALE)
    fwhm = est.get("fwhm")                    # 字段名就是 fwhm
    status = str(est.get("status", "unknown"))
    ci = boot.get("fwhm_95_ci") if isinstance(boot, dict) else None
    return (float(fwhm) if fwhm is not None else float("nan"),
            tuple(ci) if ci else None, status)


def report(label: str, analysis_dir: Path) -> dict:
    d = load_axes(analysis_dir)
    print(f"### {label}")
    print(f"    目录：{analysis_dir}")
    print(f"    初级事件：{d['n_primaries']}    可用重建事件：{d['n_usable']}")
    out = {}
    for axis in ("x", "y", "depth"):
        fwhm, ci, status = measure(d[axis])
        out[axis] = fwhm
        ci_txt = f"  bootstrap 95% 区间 {ci[0]:.2f} – {ci[1]:.2f} mm" if ci else "  （未做 bootstrap）"
        bad = "" if status == "ok" else f"   [点估计状态: {status}]"
        name = {"x": "x 方向", "y": "y 方向", "depth": "深度方向"}[axis]
        print(f"    {name} FWHM = {fwhm:8.2f} mm{ci_txt}{bad}")
    print()
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--analysis-dir", type=Path, default=None,
                    help="新数据的 analyze 输出目录（沿 x 细杆）")
    ap.add_argument("--selfcheck", action="store_true",
                    help="在已有的沿 y 细杆数据上自检，x 方向应复现 3.67 mm")
    args = ap.parse_args()

    if not args.selfcheck and args.analysis_dir is None:
        ap.error("至少要给 --selfcheck 或 --analysis-dir 之一")

    ref = None
    if args.selfcheck:
        if not REFERENCE_DIR.exists():
            print(f"自检数据不存在：{REFERENCE_DIR}")
            return 1
        ref = report("自检：已有的沿 y 细杆（测 x 方向）", REFERENCE_DIR)
        err = abs(ref["x"] - REFERENCE_X_FWHM_MM)
        if not np.isfinite(ref["x"]) or err > 0.3:
            print(f"自检未通过：x 方向算出 {ref['x']:.2f} mm，官方值 {REFERENCE_X_FWHM_MM} mm，"
                  f"差 {err:.2f} mm > 0.3 mm。")
            print("说明本脚本的算法与官方链路不一致，新数据的结果不可信，先停下来。")
            return 1
        print(f"自检通过：x 方向 {ref['x']:.2f} mm，与官方 {REFERENCE_X_FWHM_MM} mm 一致。\n")

    if args.analysis_dir is not None:
        new = report("新数据：沿 x 细杆（测 y 方向）", args.analysis_dir)
        print("=" * 60)
        print("结论：")
        print(f"  y 方向 FWHM = {new['y']:.2f} mm")
        if ref is not None:
            print(f"  对比已有的 x 方向 = {ref['x']:.2f} mm")
            if np.isfinite(new["y"]) and np.isfinite(ref["x"]):
                rel = abs(new["y"] - ref["x"]) / ref["x"]
                if rel < 0.25:
                    print(f"  两者相差 {rel*100:.0f}%，符合几何 x/y 对称的预期。")
                    print("  → 二维位置分辨率成立，可以对导师给出 x 和 y 两个方向的实测值。")
                else:
                    print(f"  两者相差 {rel*100:.0f}%，超出预期。几何是 x/y 对称的，")
                    print("  差这么多说明重建链路在 y 方向可能有问题，需要排查。")
        print("  注意：沿 x 细杆的 x 方向数字没有意义（那是沿杆方向，没有结构）。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
