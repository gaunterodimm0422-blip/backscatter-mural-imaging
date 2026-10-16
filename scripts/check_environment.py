"""环境自检：换机器 / 换 Python 版本之后，先确认分析结果没有漂移。

为什么要这一步
--------------
项目此前发布的数字（横向 y FWHM 4.146 mm、深度 38.32 mm）都是在主力机的
E:\\python3.12 环境下算出来的。远控机是 Python 3.13 + 另一套 numpy/scipy。
KDE 与 bootstrap 都走 scipy，版本不同可能有微小漂移。本脚本用一份**现成的**
重建产物（不需要 ROOT）重算同样的量，与已发布值逐条对照。

判据
----
  1. 闭式解自检：2 万个已知顶点全部解出，最大顶点误差 < 1e-8 mm，非法能量全被拒
  2. 横向 y FWHM 与 4.146 mm 相差 < 1%
  3. 深度 z FWHM 与 38.317 mm 相差 < 1%
  4. bootstrap 95% CI 与 [3.29, 5.01] 相差 < 3%
三条以上不过就别在这台机器上出正式数字，先把差异查清楚。

用法
----
  python scripts/check_environment.py --reconstruction reference/rod_x_reconstruction.npz
"""
from __future__ import annotations

import argparse
import platform
import sys
from pathlib import Path

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

REFERENCE = {
    "lateral_y_fwhm_mm": 4.146170763993865,
    "depth_z_fwhm_mm": 38.31723835209995,
    "lateral_ci_mm": (3.29, 5.01),
    "n_selected_valid": 279,
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reconstruction", type=Path,
                    default=ROOT_DIR / "reference" / "rod_x_reconstruction.npz")
    ap.add_argument("--bootstrap", type=int, default=400)
    args = ap.parse_args()

    import scipy
    print("[环境]")
    print(f"  Python      {platform.python_version()}  ({sys.executable})")
    print(f"  numpy       {np.__version__}")
    print(f"  scipy       {scipy.__version__}")
    try:
        import uproot
        print(f"  uproot      {uproot.__version__}")
    except Exception as exc:                       # 只用于打印版本，缺了不影响本自检
        print(f"  uproot      不可用（{type(exc).__name__}）——读 ROOT 时才需要它")

    from backscatter.kinematics import synthetic_selftest
    from backscatter.standards import bootstrap_fwhm, estimate_fwhm_kde

    ok = []
    print("\n[1/4] 闭式解自检")
    r = synthetic_selftest()
    p1 = bool(r["passed"])
    print(f"  生成 {r['n_generated']}  解出 {r['n_reconstructed']}  "
          f"最大顶点误差 {r['max_vertex_error_mm']:.3e} mm  "
          f"非法能量被拒 {r['invalid_160keV_rejected']}")
    print(f"  -> {'通过' if p1 else '不通过'}")
    ok.append(p1)

    if not args.reconstruction.exists():
        print(f"\n找不到参照数据 {args.reconstruction}，后三项跳过")
        return 0 if all(ok) else 1
    d = np.load(args.reconstruction)
    m = np.asarray(d["selected_valid"], dtype=bool)
    n = int(m.sum())
    print(f"\n[2/4] 参照数据：{args.reconstruction.name}，selected_valid = {n}"
          f"（应为 {REFERENCE['n_selected_valid']}）")
    ok.append(n == REFERENCE["n_selected_valid"])

    def compare(label, value, expected, tol):
        rel = abs(value - expected) / expected
        good = rel <= tol
        print(f"  {label}: 本机 {value:.6f}  已发布 {expected:.6f}  "
              f"相对差 {rel*100:.3f}%  -> {'通过' if good else '不通过'}")
        return good

    print("\n[3/4] FWHM 复现")
    fy = estimate_fwhm_kde(d["y_mm"][m])
    fz = estimate_fwhm_kde(d["z_mm"][m])
    ok.append(compare("横向 y FWHM", float(fy["fwhm"]), REFERENCE["lateral_y_fwhm_mm"], 0.01))
    ok.append(compare("深度 z FWHM", float(fz["fwhm"]), REFERENCE["depth_z_fwhm_mm"], 0.01))

    print("\n[4/4] bootstrap 95% CI 复现")
    by = bootstrap_fwhm(d["y_mm"][m], n_bootstrap=args.bootstrap, seed=11)
    lo, hi = by["fwhm_95_ci"]
    elo, ehi = REFERENCE["lateral_ci_mm"]
    ok.append(compare("CI 下界", lo, elo, 0.03))
    ok.append(compare("CI 上界", hi, ehi, 0.03))

    print("\n" + "=" * 58)
    print("环境自检：" + ("全部通过，可以在这台机器上出正式数字"
                        if all(ok) else "有不通过项，先查清差异再出数字"))
    print("=" * 58)
    return 0 if all(ok) else 1


if __name__ == "__main__":
    raise SystemExit(main())
