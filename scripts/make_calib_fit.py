# -*- coding: utf-8 -*-
"""由 M2 四档边界分析的结果 json 生成 calib_fit.json（make_calib_figure.py 的输入）。

2026-09-28 新写：此前的 calib_fit.json（D2 12%）是手工生成的，没有留脚本；
为了让 G6 可复现，改由本脚本从四档 m2_edge_result.json 直接算。
做法与 redo_d2f15.py summary 的 M2 四点检验逐字相同：无权直线拟合
  y = x_e(探测头坐标) = k·u + b，u = 真值 − 探测头轴心，
s = √(Σ残差²/(n−2))，σ_k = s/√Suu，σ_b = s·√(1/n + ū²/Suu)。
sig 取各档 fits.signal.xe_sigma（profile likelihood 68% 区间的半宽），只用于画误差棒，不进拟合。

用法：
  python make_calib_fit.py --main <m2_mural_edge 目录> --offset <…_offset 目录>
         --offset2 <…_offset2 目录> --offset3 <…_offset3 目录> --out outputs/calib_fit.json
"""
import argparse, json, math
from pathlib import Path

ap = argparse.ArgumentParser()
for k in ("main", "offset", "offset2", "offset3"):
    ap.add_argument(f"--{k}", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()

u, y, sig, src = [], [], [], []
for k in ("main", "offset", "offset2", "offset3"):
    p = Path(getattr(a, k)) / "m2_edge_result.json"
    d = json.load(open(p, encoding="utf-8"))
    bad = [v for v in d["inputs"].values() if not v.rstrip("\\/").endswith("response_d2f15")]
    if bad:
        raise SystemExit(f"{p} 的输入不全是 response_d2f15：{bad}")
    s = d["fits"]["signal"]
    u.append(d["truth_edge_world_mm"] - d["head_x_mm"])
    y.append(s["xe"]); sig.append(s["xe_sigma"]); src.append(str(p))

n = len(u); ub = sum(u) / n; yb = sum(y) / n
Suu = sum((x - ub) ** 2 for x in u)
k = sum((x - ub) * (v - yb) for x, v in zip(u, y)) / Suu
b = yb - k * ub
res = [v - (k * x + b) for x, v in zip(u, y)]
s = math.sqrt(sum(r * r for r in res) / (n - 2))
out = dict(k=k, sk=s / math.sqrt(Suu), b=b, sb=s * math.sqrt(1 / n + ub ** 2 / Suu),
           r2=1 - sum(r * r for r in res) / sum((v - yb) ** 2 for v in y),
           srms=s, srms_definition="s = sqrt(sum(res^2)/(n-2))（不是 RMS）",
           u=u, y=y, sig=sig, res=res, response="D2 15%（response_d2f15）", sources=src)
Path(a.out).parent.mkdir(parents=True, exist_ok=True)
json.dump(out, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(f"写入 {a.out}：k = {k:.4f} ± {out['sk']:.4f}，b = {b:+.3f} ± {out['sb']:.3f} mm，"
      f"r² = {out['r2']:.5f}，s = {s:.3f} mm")
