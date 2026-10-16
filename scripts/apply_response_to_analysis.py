"""把冻结的探测器响应施加到一个 analyze 输出目录上，生成 canonical_response 视图。

背景：`python -m backscatter analyze` 只做到"理想 GATE 事件对"，**不施加探测器响应**
（D1 8% / D2 12% 能量展宽、位置离散化）。官方的 3.67 mm / 38.47 mm 都是在施加响应
之后的 canonical_response 目录上算的。少了这一步，深度会偏乐观（接近 12.39 mm 的
理想能量分辨率极限），横向影响很小。

本脚本只做响应这一步，直接调用项目自己的
backscatter.hardware_figures.prepare_hardware_response_analysis 和
materialize_response_analysis_view，参数与官方流程一致。
不包含官方脚本里针对"沿 y 细杆 + x 方向"的验收判定，所以对任意朝向的体模都能用。

用法：
  E:\\python3.12\\python.exe scripts\\apply_response_to_analysis.py ^
      --analysis-dir runs\\rod_x_depth30_5m_seed24083101 ^
      --config configs\\transverse_fwhm_5mm_frozen.json

默认响应随机种子 24080601，与官方 canonical_response 用的是同一个。
输出写到 <analysis-dir>\\canonical_response\\，不改动原目录。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backscatter.hardware_figures import (
    materialize_response_analysis_view,
    prepare_hardware_response_analysis,
)

OFFICIAL_RESPONSE_SEED = 24080601      # 官方 canonical_response 用的响应种子


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--analysis-dir", type=Path, required=True,
                    help="python -m backscatter analyze 的输出目录")
    ap.add_argument("--config", type=Path,
                    default=PROJECT_ROOT / "configs" / "transverse_fwhm_5mm_frozen.json")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="默认 <analysis-dir>/canonical_response")
    ap.add_argument("--seed", type=int, default=OFFICIAL_RESPONSE_SEED)
    args = ap.parse_args()

    analysis_dir = args.analysis_dir.resolve()
    out_dir = (args.out_dir or analysis_dir / "canonical_response").resolve()
    if not (analysis_dir / "pairs.npz").exists():
        print(f"找不到 {analysis_dir / 'pairs.npz'}，请先跑 python -m backscatter analyze")
        return 1
    if (out_dir / "reconstruction.npz").exists():
        print(f"{out_dir} 里已经有 reconstruction.npz，为避免覆盖已中止。")
        print("要重做请先删掉该目录，或用 --out-dir 换一个位置。")
        return 1

    print(f"输入（理想响应）: {analysis_dir}")
    print(f"配置            : {args.config}")
    print(f"响应随机种子    : {args.seed}"
          f"{'（与官方 canonical_response 一致）' if args.seed == OFFICIAL_RESPONSE_SEED else ''}")
    print(f"输出            : {out_dir}")
    print()

    print("[1/2] 施加 D1/D2 能量展宽与位置离散化 ...", flush=True)
    info = prepare_hardware_response_analysis(
        analysis_dir=analysis_dir,
        config_path=args.config,
        out_dir=out_dir,
        random_seed=args.seed,
    )
    print("[2/2] 生成标准评估器可读的 pairs/reconstruction/summary 视图 ...", flush=True)
    materialize_response_analysis_view(
        source_analysis_dir=analysis_dir,
        response_dir=out_dir,
    )

    for name in ("pairs.npz", "reconstruction.npz", "summary.json"):
        f = out_dir / name
        print(f"  {name:22s} {'存在' if f.exists() else '缺失'}"
              f"  {f.stat().st_size:,} B" if f.exists() else f"  {name:22s} 缺失")
    print()
    print("完成。接下来量三个方向：")
    print(f"  E:\\python3.12\\python.exe scripts\\analyze_rod_axes.py --selfcheck --analysis-dir {out_dir}")

    manifest = out_dir / "response_application.json"
    with manifest.open("w", encoding="utf-8") as fh:
        json.dump({
            "schema_version": "1.0",
            "purpose": "对 analyze 的理想输出施加冻结探测器响应，等价于官方 canonical_response",
            "source_analysis_dir": str(analysis_dir),
            "config": str(Path(args.config).resolve()),
            "response_seed": args.seed,
            "official_response_seed": OFFICIAL_RESPONSE_SEED,
            "prepare_info_keys": sorted(info.keys()) if isinstance(info, dict) else None,
        }, fh, ensure_ascii=False, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
