# -*- coding: utf-8 -*-
r"""F3 空鼓厚度扫描：远控机上的分析驱动与判据判定（2026-09-23 冻结）。

2026-09-23 修订（见到任何 F3 数据之前）：导师确认 D2 能量分辨为 15 %。本脚本改用
  配置 configs\m3_thick_t*_d2f15_20260923.json（只把 D2 的 0.12 改成 0.15），
  响应目录 runs\<RUN_ID>\response_d2f15\（参照场、零假设底片也用这个目录），
  输出 outputs\d2f15\f3_thick_t*\，锚点 outputs\d2f15\m3_edge_result\。
  前提：先用 scripts\redo_d2f15.py run 把 M3 的参照场、零假设底片和锚点按 15 % 重做完。
  冻结判据一个字没改。

两个子命令
----------
  pipeline  逐档跑三步：python -m backscatter analyze → apply_response_to_analysis.py
            → analyze_mural_edge.py。某一步的产物已经存在就跳过这一步，可以放心重复执行。
  judge     读各档的 m2_edge_result.json，按冻结的 F3 判据判定，
            给出"检出下限"与"定位下限"，写 outputs\f3_thickness\f3_summary.json。

用法（远控机 Windows 侧，在 D:\beisanshe 下）
------------------------------------------
  <python> scripts\f3_thickness_scan.py pipeline --dry-run      只打印将要执行的命令，什么都不跑
  <python> scripts\f3_thickness_scan.py pipeline                真跑五档
  <python> scripts\f3_thickness_scan.py pipeline --only 10,5    只跑指定厚度（跑完一档就能先分析一档）
  <python> scripts\f3_thickness_scan.py judge                   判定

  <python> = C:\Users\Administrator\AppData\Local\Programs\Python\Python313\python.exe

冻结判据（见到数据之前定死，与 configs\m3_thick_t*_20260923.json 里的 frozen_criteria 逐字对应）
--------------------------------------------------------------------------------------------
  F3-1 检出  halfvoid/ref_main 的 T >= 30，且深两带（15–25、25–35 mm）拟合出的 c_b 至少一个 > 0
  F3-2 定位  满足 F3-1，且 |x_e − (−15)| <= 2.0 mm，且 68% 区间半宽 (xe_hi − xe_lo)/2 <= 2.0 mm
  F3-3 交叉  满足 F3-1 的档，halfvoid/ref_null 与 halfvoid/ref_main 的 x_e 相差 <= 5 mm
  检出下限 = 过 F3-1 的最薄一档；定位下限 = 同时过 F3-1/2/3 的最薄一档。
  若出现"薄档过、厚档不过"（不单调），不下单一下限，先查原因。

  analyze_mural_edge.py 自己打印的判据 1–5 是 M3 主档的，F3 不看（判据 5 要求四带 c_b 全正，
  薄空鼓的 0–8 带本来就约为 0，正负各半）。

参照场与零假设底片沿用 M3：m3_ref_t20_L-15_4m_seed24091302、m3_refnull_t20_L-15_4m_seed24091303（用 response_d2f15）。
20 mm 档不重跑 GATE，锚点读 outputs\d2f15\m3_edge_result\m2_edge_result.json（M3 主档按 15 % 重做的结果）。
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------- 冻结常量
T_THR = 30.0
XE_TRUE_WORLD = -15.0
XE_TOL_MM = 2.0
HW_TOL_MM = 2.0
CROSS_TOL_MM = 5.0
DEEP_BANDS = ([15, 25], [25, 35])

THICK_SEED = {10: 24092301, 5: 24092302, 3: 24092303, 2: 24092304, 1: 24092305}
REF_MAIN = "m3_ref_t20_L-15_4m_seed24091302"
REF_NULL = "m3_refnull_t20_L-15_4m_seed24091303"
RESP = "response_d2f15"
ANCHOR_T20 = ("m3_edge_result", "m3_halfvoid_t20_L-15_4m_seed24091301")


def run_id(t: int) -> str:
    return f"m3_halfvoid_t{t}_L-15_4m_seed{THICK_SEED[t]}"


def cfg_path(repo: Path, t: int) -> Path:
    return repo / "configs" / f"m3_thick_t{t}_d2f15_20260923.json"


def out_dir(repo: Path, t: int) -> Path:
    return repo / "outputs" / "d2f15" / f"f3_thick_t{t}"


# ---------------------------------------------------------------- pipeline
def pipeline(args) -> int:
    repo, share, py = args.repo, args.share, sys.executable
    todo = sorted(THICK_SEED, reverse=True)
    if args.only:
        want = {int(s) for s in args.only.split(",") if s.strip()}
        bad = want - set(THICK_SEED)
        if bad:
            print(f"--only 里有不认识的厚度 {sorted(bad)}，可选 {sorted(THICK_SEED, reverse=True)}")
            return 2
        todo = [t for t in todo if t in want]

    problems = []
    for ref in (REF_MAIN, REF_NULL):
        p = repo / "runs" / ref / RESP / "reconstruction.npz"
        if not p.exists():
            problems.append(f"找不到参照场 {p}（先跑 scripts\\redo_d2f15.py run）")
    for t in todo:
        c = cfg_path(repo, t)
        if not c.exists():
            problems.append(f"找不到配置 {c}")
    if problems:
        print("\n".join("  !! " + s for s in problems))
        if not args.dry_run:
            print("先把上面缺的东西补齐再跑。")
            return 1

    print(f"仓库      : {repo}")
    print(f"共享盘    : {share}")
    print(f"Python    : {py}")
    print(f"本次厚度  : {todo} mm")
    print(f"模式      : {'只打印命令（--dry-run）' if args.dry_run else '真跑'}\n")

    for t in todo:
        rid = run_id(t)
        adir = repo / "runs" / rid
        cfg = cfg_path(repo, t)
        res = out_dir(repo, t)
        hits = share / rid / "hits.root"
        truth = share / rid / "hits_truth_mural.root"
        steps = [
            ("1/3 analyze（聚合 Hits、符合、重建）", adir / "pairs.npz",
             [py, "-m", "backscatter", "analyze", "--root", str(hits), "--truth", str(truth),
              "--config", str(cfg), "--out-dir", str(adir)]),
            ("2/3 施加探测器响应（D2 15 %）→ response_d2f15",
             adir / RESP / "reconstruction.npz",
             [py, str(repo / "scripts" / "apply_response_to_analysis.py"),
              "--analysis-dir", str(adir), "--config", str(cfg), "--out-dir", str(adir / RESP)]),
            ("3/3 边界拟合（与 M3 逐字同一套估计量）", res / "m2_edge_result.json",
             [py, str(repo / "scripts" / "analyze_mural_edge.py"),
              "--halfvoid", str(adir / RESP),
              "--ref-main", str(repo / "runs" / REF_MAIN / RESP),
              "--ref-null", str(repo / "runs" / REF_NULL / RESP),
              "--config", str(cfg), "--out", str(res), "--null-trials", "200"]),
        ]
        print(f"===== 厚 {t} mm   {rid} =====")
        if not hits.exists() and not args.dry_run:
            print(f"  !! GATE 结果还没到：{hits}\n     这一档跳过。")
            print()
            continue
        for name, product, cmd in steps:
            if product.exists():
                print(f"  [跳过] {name}：已有 {product.name}")
                continue
            print(f"  [{'命令' if args.dry_run else '执行'}] {name}")
            print("    " + " ".join(f'"{a}"' if " " in a else a for a in cmd))
            if args.dry_run:
                continue
            t0 = time.time()
            r = subprocess.run(cmd, cwd=str(repo))
            if r.returncode != 0:
                print(f"  !! 这一步失败（返回码 {r.returncode}），停止。把上面的输出发回来。")
                return r.returncode
            print(f"    完成，用时 {time.time() - t0:.0f} s")
        print()
    print("全部结束。下一步：scripts\\f3_thickness_scan.py judge")
    return 0


# ---------------------------------------------------------------- judge
def _read(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _row(t: int, d: dict, expect_halfvoid: str) -> dict:
    head = float(d["head_x_mm"])
    s, cons, null = d["fits"]["signal"], d["fits"]["consistency"], d["fits"]["null"]
    bands = [list(map(int, b)) for b in d["bands_mm"]]
    c = [float(v) for v in s["c"]]
    deep = [c[bands.index(list(b))] for b in DEEP_BANDS if list(b) in bands]
    xe = float(s["xe"]) + head
    lo, hi = float(s["xe_lo"]) + head, float(s["xe_hi"]) + head
    hw = (hi - lo) / 2.0
    xe_cons = float(cons["xe"]) + head
    T = float(s["T"])

    guard = []
    raw = {k: Path(str(v).replace("\\", "/")) for k, v in (d.get("inputs") or {}).items()}
    inp = {k: v.parent.name for k, v in raw.items()}
    for k, v in raw.items():
        if v.name != RESP:
            guard.append(f"{k} 用的响应目录是 {v.name}，应为 {RESP}（D2 15 %）")
    if inp.get("halfvoid") != expect_halfvoid:
        guard.append(f"halfvoid 输入是 {inp.get('halfvoid')}，应为 {expect_halfvoid}")
    if inp.get("ref_main") != REF_MAIN:
        guard.append(f"ref_main 输入是 {inp.get('ref_main')}，应为 {REF_MAIN}")
    if inp.get("ref_null") != REF_NULL:
        guard.append(f"ref_null 输入是 {inp.get('ref_null')}，应为 {REF_NULL}")
    if abs(float(d.get("truth_edge_world_mm", XE_TRUE_WORLD)) - XE_TRUE_WORLD) > 1e-9:
        guard.append(f"真值边界是 {d.get('truth_edge_world_mm')}，应为 {XE_TRUE_WORLD}")

    f1 = (T >= T_THR) and any(v > 0 for v in deep)
    f2 = f1 and abs(xe - XE_TRUE_WORLD) <= XE_TOL_MM and hw <= HW_TOL_MM
    f3 = f1 and abs(xe_cons - xe) <= CROSS_TOL_MM
    return dict(t_mm=t, T=T, T_null=float(null["T"]), xe_world=xe, bias=xe - XE_TRUE_WORLD,
                ci68=[lo, hi], half_width=hw, xe_cross_diff=xe_cons - xe,
                c_b_pct=[round(100 * v, 2) for v in c], deep_c_b_pct=[round(100 * v, 2) for v in deep],
                F3_1=f1, F3_2=f2, F3_3=f3, located=bool(f1 and f2 and f3), guard=guard)


def judge(args) -> int:
    repo = args.repo
    rows, missing = [], []
    a = _read(repo / "outputs" / "d2f15" / ANCHOR_T20[0] / "m2_edge_result.json")
    if a is None:
        missing.append(20)
    else:
        rows.append(_row(20, a, ANCHOR_T20[1]))
    for t in sorted(THICK_SEED, reverse=True):
        d = _read(out_dir(repo, t) / "m2_edge_result.json")
        if d is None:
            missing.append(t)
            continue
        rows.append(_row(t, d, run_id(t)))

    if not rows:
        print("一档结果都没有。先跑 pipeline。")
        return 1

    print("F3 空鼓厚度扫描：冻结判据判定（D2 能量分辨 15 %，响应目录 response_d2f15）")
    print(f"判据：F3-1 T >= {T_THR:g} 且深两带 c_b 至少一个 > 0；"
          f"F3-2 |偏差| <= {XE_TOL_MM:g} mm 且 68% 半宽 <= {HW_TOL_MM:g} mm；"
          f"F3-3 交叉差 <= {CROSS_TOL_MM:g} mm\n")
    hdr = (f"{'厚度':>5} {'T':>8} {'x_e(世界)':>10} {'偏差':>7} {'68%半宽':>8} {'交叉差':>7} "
           f"{'c_b(15-25)':>11} {'c_b(25-35)':>11}  {'F3-1':>4} {'F3-2':>4} {'F3-3':>4}")
    print(hdr)
    print("-" * len(hdr.encode("gbk", "ignore")))
    yn = lambda b: "过" if b else "—"
    for r in rows:
        dc = r["deep_c_b_pct"] + [float("nan")] * (2 - len(r["deep_c_b_pct"]))
        print(f"{r['t_mm']:>4g}  {r['T']:>8.1f} {r['xe_world']:>10.2f} {r['bias']:>+7.2f} "
              f"{r['half_width']:>8.2f} {r['xe_cross_diff']:>+7.2f} {dc[0]:>10.2f}% {dc[1]:>10.2f}%  "
              f"{yn(r['F3_1']):>4} {yn(r['F3_2']):>4} {yn(r['F3_3']):>4}"
              + ("   (20 mm 为 M3 主档，锚点)" if r["t_mm"] == 20 else ""))
    guards = [(r["t_mm"], g) for r in rows for g in r["guard"]]
    if guards:
        print("\n!! 输入核对不通过（这几档的结果不能用，先查）：")
        for t, g in guards:
            print(f"   厚 {t} mm：{g}")
    if missing:
        print(f"\n缺结果的厚度：{missing} mm")

    ordered = sorted(rows, key=lambda r: -r["t_mm"])
    det = [r["F3_1"] for r in ordered]
    loc = [r["located"] for r in ordered]

    def threshold(flags):
        # 单调 = 从厚到薄先全"过"、后全"不过"
        k = next((i for i, f in enumerate(flags) if not f), len(flags))
        mono = all(not f for f in flags[k:])
        if not mono:
            return None, False
        return (ordered[k - 1]["t_mm"] if k > 0 else None), True

    t_det, mono_det = threshold(det)
    t_loc, mono_loc = threshold(loc)
    Ts = [r["T"] for r in ordered]
    T_mono = all(Ts[i] >= Ts[i + 1] for i in range(len(Ts) - 1))

    print()
    if not mono_det:
        print("检出下限：不单调（有薄档过而厚档不过），按冻结规则不下单一下限，先查原因。")
    else:
        print(f"检出下限：{f'{t_det:g} mm' if t_det is not None else '所有已跑的厚度都没检出'}"
              + ("（注意：还有厚度没跑完）" if missing else ""))
    if not mono_loc:
        print("定位下限：不单调，按冻结规则不下单一下限，先查原因。")
    else:
        print(f"定位下限：{f'{t_loc:g} mm' if t_loc is not None else '所有已跑的厚度都没定出边界'}"
              + ("（注意：还有厚度没跑完）" if missing else ""))
    print(f"一致性检查（不是判据）：T 随厚度{'单调下降' if T_mono else '不单调 —— 厚度小时 T 由涨落主导，先看是不是都在 30 以下'}")
    if any(g for g in guards):
        print("有档的输入核对没通过，上面的下限不作数。")

    summ = dict(generated=time.strftime("%Y-%m-%d %H:%M:%S"),
                criteria=dict(T_thr=T_THR, xe_true_world=XE_TRUE_WORLD, xe_tol_mm=XE_TOL_MM,
                              half_width_tol_mm=HW_TOL_MM, cross_tol_mm=CROSS_TOL_MM,
                              deep_bands=[list(b) for b in DEEP_BANDS]),
                rows=rows, missing_thickness_mm=missing,
                detection_limit_mm=t_det if mono_det else None, detection_monotonic=mono_det,
                localization_limit_mm=t_loc if mono_loc else None, localization_monotonic=mono_loc,
                T_monotonic=T_mono, input_guard_failures=[f"t{t}: {g}" for t, g in guards])
    od = repo / "outputs" / "d2f15" / "f3_thickness"
    od.mkdir(parents=True, exist_ok=True)
    (od / "f3_summary.json").write_text(json.dumps(summ, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n写出 {od / 'f3_summary.json'}")
    return 0


def main() -> int:
    here = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser(description="F3 空鼓厚度扫描：分析驱动与判据判定")
    ap.add_argument("--repo", type=Path, default=here, help="仓库根目录（默认脚本所在仓库）")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p1 = sub.add_parser("pipeline", help="逐档跑 analyze → apply_response → analyze_mural_edge")
    p1.add_argument("--share", type=Path, default=Path(r"D:\vgate_shared\output"),
                    help=r"GATE 结果所在目录（默认 D:\vgate_shared\output）")
    p1.add_argument("--only", default="", help="只跑这些厚度，逗号分隔，例如 10,5")
    p1.add_argument("--dry-run", action="store_true", help="只打印命令，不执行")
    sub.add_parser("judge", help="按冻结判据判定并写 f3_summary.json")
    args = ap.parse_args()
    return pipeline(args) if args.cmd == "pipeline" else judge(args)


if __name__ == "__main__":
    sys.exit(main())
