# -*- coding: utf-8 -*-
r"""M3 与 M2 按 D2 能量分辨 15 % 重做分析（不重跑 GATE），2026-09-23，第二版（加入 M2）。

背景
----
导师确认 D2 能量分辨应为 15 %（原来一直用 12 %）。能量展宽与位置离散都是在分析时对
GATE 理想沉积事后施加的（backscatter/response.py），GATE 宏里没有任何展宽，所以只需：
  0) 准备 D2 15 % 的配置：由各自的 12 % 配置复制，只改 detectors.d2.energy_response.fwhm_fraction_at_511
     （0.12 -> 0.15），另加说明键 response_revision；程序逐键核对，多改一个键就停。
     M3 的配置随包交付，这里只核对；M2 的配置在远控机上由 12 % 原件现场生成（其中两份原件只在远控机上）。
  1) 对每个运行目录重新施加响应（响应种子仍是 24080601），写到 runs\<RUN_ID>\response_d2f15\，
     原来的 canonical_response\（12 %）一个字不动；
  2) 用同一个 analyze_mural_edge.py 重做边界分析（M3 四次、M2 五次）；
  3) 重出成像演示 npz 与诊断（M3：成像 3 份、诊断 2 份；M2：成像 1 份、诊断 3 份）；
  4) 按各自原来冻结的判据重判（M3 九条；M2 主档五条、偏移档六条、两个标定档各四条、四点线性检验）。
12 % 的结果作为冻结记录保留，不覆盖、不并列报告；只在某条判据的通过/不通过翻转时标出来。
所有新产物都在 outputs\d2f15\ 下。

判据 9（M3）与 M2 的线性检验都用 json 里的全精度 x_e 做无权拟合；12 % 冻结版当时用的是保留两位
小数的 x_e（M3 的 k 因此是 0.9263，全精度 0.9262），不影响判定。

用法（远控机 Windows，PowerShell，在 D:\beisanshe 下）
----------------------------------------------------
  <python> scripts\redo_d2f15.py run --dry-run          只打印将要执行的命令（M3 与 M2 都做）
  <python> scripts\redo_d2f15.py run                    真跑（已有产物的步骤自动跳过，可重复执行）
  <python> scripts\redo_d2f15.py summary                按冻结判据重判，写 outputs\d2f15\m3_d2f15_summary.json 与 m2_d2f15_summary.json
  加 --set m3 或 --set m2 只做其中一组；默认 --set all。

  <python> = C:\Users\Administrator\AppData\Local\Programs\Python\Python313\python.exe
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
import time
from pathlib import Path

RESP = "response_d2f15"
KEY_OLD = '"fwhm_fraction_at_511": 0.12'
KEY_NEW = '"fwhm_fraction_at_511": 0.15'
REV_TEMPLATE = ('  "response_revision": {\n'
                '    "date": "2026-09-23",\n'
                '    "change": "detectors.d2.energy_response.fwhm_fraction_at_511: 0.12 -> 0.15",\n'
                '    "reason": "导师确认 D2 能量分辨应为 15 %（需求文档值）；原 0.12 出自 2026-07-31 硬件参数更新。其余键与源文件逐字相同。",\n'
                '    "source_config": "SRCNAME",\n'
                '    "gate_rerun_needed": false,\n'
                '    "response_output_dir_name": "response_d2f15"\n'
                '  },\n')
M2_MAIN_SRC_SHA = "7e9db6722320d0476631a015ece031118ca9dbd4362e7a141cdec4e4d854f6ea"  # M2 设计文档第 10 节的冻结值

SETS = {
    "m3": dict(
        label="M3（新构型）",
        runs={"halfvoid": "m3_halfvoid_t20_L-15_4m_seed24091301",
              "ref": "m3_ref_t20_L-15_4m_seed24091302",
              "refnull": "m3_refnull_t20_L-15_4m_seed24091303",
              "x23": "m3_halfvoidx23_t20_L-15_4m_seed24091311",
              "x31": "m3_halfvoidx31_t20_L-15_4m_seed24091312"},
        cfg={"main": ("m3_edge_main_20260912.json", "m3_edge_main_d2f15_20260923.json"),
             "offset": ("m3_edge_offset_20260912.json", "m3_edge_offset_d2f15_20260923.json"),
             "offset2": ("m3_edge_offset2_20260912.json", "m3_edge_offset2_d2f15_20260923.json"),
             "widefov": ("m3_edge_main_widefov_20260912.json", "m3_edge_main_widefov_d2f15_20260923.json")},
        generate_cfg=False,
        resp_cfg={"halfvoid": "main", "ref": "main", "refnull": "main", "x23": "offset", "x31": "offset2"},
        edge=[("m3_edge_result", "halfvoid", "main", 2000),
              ("m3_offset", "x23", "offset", 200),
              ("m3_offset2", "x31", "offset2", 200),
              ("m3_widefov", "halfvoid", "widefov", 0)],
        img=[("imaging_demo_m3", "halfvoid", "m3_edge_result", -15.0, "M3 新构型"),
             ("imaging_demo_m3x23", "x23", "m3_offset", -23.0, "M3 边界离轴 -8 mm"),
             ("imaging_demo_m3x31", "x31", "m3_offset2", -31.0, "M3 边界离轴 -16 mm")],
        diag=[("diag_ms_m3", "diag_multiscatter.py", "M3 D2 15%"),
              ("diag_m3_ref", "diag_shallow_band.py", "M3 D2 15%")],
    ),
    "m2": dict(
        label="M2（旧构型，对照组）",
        runs={"halfvoid": "mural_halfvoid_t20_L-15_4m_seed24091121",
              "ref": "mural_ref2_t20_L-15_4m_seed24091122",
              "refnull": "mural_ref_t10_L-15_2m_seed24091012",
              "x23": "mural_halfvoidx23_t20_L-15_4m_seed24091131",
              "x31": "mural_halfvoidx31_t20_L-15_4m_seed24091141",
              "x07": "mural_halfvoidx07_t20_L-15_4m_seed24091151"},
        cfg={"main": ("mural_edge_20260911.json", "mural_edge_d2f15_20260923.json"),
             "offset": ("mural_edge_offset_20260911.json", "mural_edge_offset_d2f15_20260923.json"),
             "offset2": ("mural_edge_offset2_20260911.json", "mural_edge_offset2_d2f15_20260923.json"),
             "offset3": ("mural_edge_offset3_20260911.json", "mural_edge_offset3_d2f15_20260923.json"),
             "widefov": ("mural_edge_widefov_20260913.json", "mural_edge_widefov_d2f15_20260923.json")},
        generate_cfg=True,
        resp_cfg={"halfvoid": "main", "ref": "main", "refnull": "main",
                  "x23": "offset", "x31": "offset2", "x07": "offset3"},
        edge=[("m2_mural_edge", "halfvoid", "main", 200),
              ("m2_mural_edge_offset", "x23", "offset", 200),
              ("m2_mural_edge_offset2", "x31", "offset2", 200),
              ("m2_mural_edge_offset3", "x07", "offset3", 200),
              ("m2_widefov", "halfvoid", "widefov", 0)],
        img=[("imaging_demo_m2", "halfvoid", "m2_mural_edge", -15.0, "M2 旧构型")],
        diag=[("diag_ms_m2", "diag_multiscatter.py", "M2 D2 15%"),
              ("diag_m2_ref", "diag_shallow_band.py", "M2 D2 15%"),
              ("m2_lateral_scale", "check_lateral_scale.py", "")],
    ),
}
DIAG_PRODUCT = {"diag_multiscatter.py": "diag_multiscatter.json",
                "diag_shallow_band.py": "diag_shallow_band.json",
                "check_lateral_scale.py": "lateral_scale.json"}


def P(repo: Path, *parts) -> Path:
    return repo.joinpath(*parts)


def out_root(repo):
    return P(repo, "outputs", "d2f15")


def resp_dir(repo, S, key):
    return P(repo, "runs", S["runs"][key], RESP)


def cfg_new(repo, S, key):
    return P(repo, "configs", S["cfg"][key][1])


def _q(a: str) -> str:
    return f'"{a}"' if " " in a else a


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _edge_json(repo, name):
    f = out_root(repo) / name / "m2_edge_result.json"
    return json.loads(f.read_text(encoding="utf-8")) if f.exists() else None


def _diff(x, y, p=""):
    out = []
    if isinstance(x, dict) and isinstance(y, dict):
        for k in sorted(set(x) | set(y)):
            if k not in x or k not in y:
                out.append(p + "." + k)
            else:
                out += _diff(x[k], y[k], p + "." + k)
    elif x != y:
        out.append(p)
    return out


ALLOWED_DIFF = ['.detectors.d2.energy_response.fwhm_fraction_at_511', '.response_revision']


# ---------------------------------------------------------------- 0) 配置
def prepare_configs(repo, S, dry) -> bool:
    ok = True
    for key, (src, dst) in S["cfg"].items():
        ps, pd = P(repo, "configs", src), P(repo, "configs", dst)
        if not ps.exists():
            print(f"  !! 找不到 12 % 原件 {ps}")
            ok = False
            continue
        if src == "mural_edge_20260911.json" and _sha(ps) != M2_MAIN_SRC_SHA:
            print(f"  !! {src} 的 sha256 与 M2 冻结值不同（{_sha(ps)[:12]}… vs {M2_MAIN_SRC_SHA[:12]}…），先停下查")
            ok = False
            continue
        t_src = ps.read_text(encoding="utf-8")
        if pd.exists():
            t_dst = pd.read_text(encoding="utf-8")
        elif S["generate_cfg"]:
            if t_src.count(KEY_OLD) != 1 or not t_src.startswith("{\n"):
                print(f"  !! {src} 里 D2 分辨率的写法与预期不同，不自动改，停下")
                ok = False
                continue
            t_dst = "{\n" + REV_TEMPLATE.replace("SRCNAME", src) + t_src.replace(KEY_OLD, KEY_NEW)[2:]
            if not dry:
                pd.write_text(t_dst, encoding="utf-8")
        else:
            print(f"  !! 找不到 {pd}（M3 的配置应随包交付）")
            ok = False
            continue
        j0, j1 = json.loads(t_src), json.loads(t_dst)
        d = _diff(j0, j1)
        good = (d == ALLOWED_DIFF and j0["detectors"]["d2"]["energy_response"]["fwhm_fraction_at_511"] == 0.12
                and j1["detectors"]["d2"]["energy_response"]["fwhm_fraction_at_511"] == 0.15)
        tag = "核对通过" if good else f"!! 与原件的差别不止 D2 分辨率：{d}"
        print(f"  配置 {dst:44s} {tag}  原件 {_sha(ps)[:12]}…  新件 {hashlib.sha256(t_dst.encode('utf-8')).hexdigest()[:12]}…"
              + ("（dry-run，未写盘）" if dry and not pd.exists() else ""))
        ok = ok and good
    return ok


# ---------------------------------------------------------------- run
def run_set(args, name) -> int:
    S, repo, py = SETS[name], args.repo, sys.executable
    print(f"==================== {S['label']} ====================")
    missing = []
    for k, rid in S["runs"].items():
        if not P(repo, "runs", rid, "pairs.npz").exists():
            missing.append(f"找不到 {P(repo, 'runs', rid, 'pairs.npz')}")
    scripts = {"apply_response_to_analysis.py", "analyze_mural_edge.py", "make_imaging_demo.py"} | {d[1] for d in S["diag"]}
    for s in sorted(scripts):
        if not P(repo, "scripts", s).exists():
            missing.append(f"找不到脚本 {P(repo, 'scripts', s)}")
    if missing:
        print("\n".join("  !! " + m for m in missing))
        if not args.dry_run:
            print("先把缺的补齐。")
            return 1
    print("[配置] 准备并核对 D2 15 % 配置")
    if not prepare_configs(repo, S, args.dry_run) and not args.dry_run:
        print("配置核对不通过，停止。把上面的输出发回来。")
        return 1

    only = {s.strip() for s in args.only.split(",") if s.strip()} or {"response", "edge", "extras"}
    steps = []
    if "response" in only:
        for k in S["runs"]:
            steps.append((f"响应 D2 15 %  {k:9s}", resp_dir(repo, S, k) / "reconstruction.npz",
                          [py, str(P(repo, "scripts", "apply_response_to_analysis.py")),
                           "--analysis-dir", str(P(repo, "runs", S["runs"][k])),
                           "--config", str(cfg_new(repo, S, S["resp_cfg"][k])),
                           "--out-dir", str(resp_dir(repo, S, k))]))
    if "edge" in only:
        for oname, hv, cfg, trials in S["edge"]:
            steps.append((f"边界分析 {oname}（零假设 {trials} 次）", out_root(repo) / oname / "m2_edge_result.json",
                          [py, str(P(repo, "scripts", "analyze_mural_edge.py")),
                           "--halfvoid", str(resp_dir(repo, S, hv)),
                           "--ref-main", str(resp_dir(repo, S, "ref")),
                           "--ref-null", str(resp_dir(repo, S, "refnull")),
                           "--config", str(cfg_new(repo, S, cfg)),
                           "--out", str(out_root(repo) / oname),
                           "--null-trials", str(trials)]))
    if "extras" in only:
        for oname, hv, src, truth, label in S["img"]:
            steps.append((f"成像 npz {oname}", out_root(repo) / oname / "imaging_demo_data.npz",
                          ("IMG", oname, hv, src, truth, label)))
        for oname, script, label in S["diag"]:
            if script == "diag_multiscatter.py":
                cmd = [py, str(P(repo, "scripts", script)), "--ref", str(resp_dir(repo, S, "ref")),
                       "--void", str(resp_dir(repo, S, "halfvoid")), "--config", str(cfg_new(repo, S, "main")),
                       "--label", label, "--out", str(out_root(repo) / oname)]
            elif script == "diag_shallow_band.py":
                cmd = [py, str(P(repo, "scripts", script)), "--analysis-dir", str(resp_dir(repo, S, "ref")),
                       "--config", str(cfg_new(repo, S, "main")), "--label", label, "--out", str(out_root(repo) / oname)]
            else:
                cmd = [py, str(P(repo, "scripts", script)), "--analysis-dir", str(resp_dir(repo, S, "ref")),
                       "--config", str(cfg_new(repo, S, "main")), "--void-depth", "20", "40",
                       "--out", str(out_root(repo) / oname)]
            steps.append((f"诊断 {oname}", out_root(repo) / oname / DIAG_PRODUCT[script], cmd))

    for label, product, cmd in steps:
        if product.exists():
            print(f"[跳过] {label}：已有 {product.name}")
            continue
        if isinstance(cmd, tuple) and cmd[0] == "IMG":
            _, oname, hv, src, truth, lab = cmd
            d = _edge_json(repo, src)
            if d is None:
                fit_txt = "<等 " + src + " 边界分析做完后自动读取>"
                if not args.dry_run:
                    print(f"  !! {label}：需要先有 {src} 的结果，跳过。")
                    continue
            else:
                fit_txt = f"{float(d['fits']['signal']['xe']) + float(d['head_x_mm']):.2f}"
            cmd = [py, str(P(repo, "scripts", "make_imaging_demo.py")),
                   "--ref", str(resp_dir(repo, S, "ref")), "--void", str(resp_dir(repo, S, hv)),
                   "--config", str(cfg_new(repo, S, "main")),
                   "--x-edge-truth", f"{truth}", "--x-edge-fit", fit_txt,
                   "--label", lab, "--out", str(out_root(repo) / oname)]
        print(f"[{'命令' if args.dry_run else '执行'}] {label}")
        print("    " + " ".join(_q(a) for a in cmd))
        if args.dry_run:
            continue
        t0 = time.time()
        r = subprocess.run(cmd, cwd=str(repo))
        if r.returncode != 0:
            print(f"  !! 这一步失败（返回码 {r.returncode}），停止。把上面的输出发回来。")
            return r.returncode
        print(f"    完成，用时 {time.time() - t0:.0f} s")
    print()
    return 0


def run(args) -> int:
    for name in (["m3", "m2"] if args.set == "all" else [args.set]):
        rc = run_set(args, name)
        if rc:
            return rc
    print("结束。下一步：scripts\\redo_d2f15.py summary")
    return 0


# ---------------------------------------------------------------- summary 公用
def _fit(d):
    head = float(d["head_x_mm"])
    s, cons, null, split = (d["fits"][k] for k in ("signal", "consistency", "null", "null_split"))
    bands = [tuple(map(int, b)) for b in d["bands_mm"]]
    return dict(T=float(s["T"]), xe_world=float(s["xe"]) + head, xe_head=float(s["xe"]),
                lo=float(s["xe_lo"]) + head, hi=float(s["xe_hi"]) + head, sigma=float(s.get("xe_sigma", float("nan"))),
                truth=float(d["truth_edge_world_mm"]), head=head,
                cross=float(cons["xe"]) - float(s["xe"]),
                c=dict(zip(bands, [float(v) for v in s["c"]])),
                T_null=float(null["T"]), T_split=float(split["T"]),
                inputs=d.get("inputs") or {}, nsel=d.get("n_selected") or {}, npri=d.get("n_primary") or {},
                ncal=d.get("null_calibration") or {})


def _guard(res):
    g = []
    for n, d in res.items():
        for k, v in (d.get("inputs") or {}).items():
            if Path(str(v).replace("\\", "/")).name != RESP:
                g.append(f"{n} 的 {k} 输入不是 {RESP}：{v}")
    return g


def _linfit(pts):
    n = len(pts)
    mu = sum(u for u, _ in pts) / n
    my = sum(y for _, y in pts) / n
    sxx = sum((u - mu) ** 2 for u, _ in pts)
    sxy = sum((u - mu) * (y - my) for u, y in pts)
    k = sxy / sxx
    b = my - k * mu
    resid = [y - (k * u + b) for u, y in pts]
    ssr = sum(r * r for r in resid)
    sst = sum((y - my) ** 2 for _, y in pts)
    s2 = ssr / (n - 2)
    return dict(k=k, k_sigma=math.sqrt(s2 / sxx), b=b, b_sigma=math.sqrt(s2 * (1 / n + mu * mu / sxx)),
                r2=(1 - ssr / sst) if sst > 0 else float('nan'), residuals=resid, s=math.sqrt(s2), points=pts)


def _windows(w):
    xc = [float(v) for v in w["x_centres_mm"]]
    win = []
    for row in w["band_mask"]:
        on = [x for x, b in zip(xc, row) if b]
        win.append((min(on), max(on)) if on else (float("nan"), float("nan")))
    return win


def _print_rows(F, lab):
    print(f"{'档':<10}{'T':>8}{'x_e(世界)':>11}{'偏差':>8}{'68% 区间':>20}{'半宽':>7}{'交叉差':>8}   c_b(%) 0–8 / 8–15 / 15–25 / 25–35")
    for key, f in F.items():
        cb = " / ".join(f"{100 * f['c'][bd]:.2f}" for bd in sorted(f["c"]))
        print(f"{lab[key]:<10}{f['T']:>8.1f}{f['xe_world']:>11.2f}{f['xe_world'] - f['truth']:>+8.2f}"
              f"{'[' + format(f['lo'], '.3f') + ', ' + format(f['hi'], '.3f') + ']':>20}{(f['hi'] - f['lo']) / 2:>7.3f}"
              f"{f['cross']:>+8.2f}   {cb}")


def _print_verdicts(names, V, frozen):
    print("\n判据                                         结果    与 12 % 冻结版判定")
    flips = []
    for key in names:
        same = V[key] == frozen[key]
        if not same:
            flips.append(key)
        print(f"  {key:<5} {names[key]:<40} {'通过' if V[key] else '不通过':<6} {'相同' if same else '!! 翻转'}")
    if flips:
        print(f"\n!! 有判据的通过/不通过与 12 % 冻结版不同：{flips}。把这张表原样发回来。")
    return flips


def _dump(repo, fname, obj):
    od = out_root(repo)
    od.mkdir(parents=True, exist_ok=True)
    (od / fname).write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"\n写出 {od / fname}")


def _ser(F):
    return {kk: {x: ({f"{a}-{bb}": c for (a, bb), c in v.items()} if x == "c" else v)
                 for x, v in f.items() if x not in ("inputs", "nsel", "npri", "ncal")} for kk, f in F.items()}


# ---------------------------------------------------------------- summary M3（九条，与 M3 冻结值逐字相同）
M3_FROZEN = {"1": True, "2": True, "3": True, "4": True, "5": True,
             "6a": False, "6b": True, "7": False, "8a": True, "8b": False, "9": True}


def summary_m3(repo) -> int:
    names_edge = [e[0] for e in SETS["m3"]["edge"]]
    res = {n: _edge_json(repo, n) for n in names_edge}
    miss = [n for n, v in res.items() if v is None]
    if miss:
        print(f"M3 缺结果：{miss}。先跑完 run。")
        return 1
    guard = _guard(res)
    F = {n: _fit(res[n]) for n in ("m3_edge_result", "m3_offset", "m3_offset2")}
    lab = {"m3_edge_result": "轴心 −15", "m3_offset": "离轴 −23", "m3_offset2": "离轴 −31"}
    m = F["m3_edge_result"]
    V = {}
    V["1"] = all(f["T"] >= 30.0 for f in F.values())
    V["2"] = all(abs(f["xe_world"] - f["truth"]) <= 10.0 for f in F.values())
    V["3"] = m["T_null"] < 30.0 and m["T_split"] < 30.0
    V["4"] = all(abs(f["cross"]) <= 5.0 for f in F.values())
    V["5"] = all(all(c > 0 for c in f["c"].values()) for f in F.values())
    V["6a"] = m["c"][(15, 25)] >= 0.319
    V["6b"] = m["c"][(25, 35)] >= 0.263
    per_m = m["nsel"]["ref_main"] / m["npri"]["ref_main"] * 1e6
    V["7"] = 15565.0 <= per_m <= 25942.0
    per_annih = per_m * 1e-6 * 1.1363 / (4 * math.pi)
    win = _windows(res["m3_widefov"])
    V["8a"] = win[0][0] <= -16.5 and win[0][1] >= 16.5
    V["8b"] = win[3][0] < -34.5 and win[3][1] > 34.5
    cal = _linfit([(f["truth"] - f["head"], f["xe_head"]) for f in F.values()])
    V["9"] = abs(cal["k"] - 0.844) <= 0.09 and abs(cal["b"]) <= 1.0 and cal["r2"] >= 0.98

    print("M3 按 D2 能量分辨 15 % 重做：九条冻结判据重判（响应目录 response_d2f15）\n")
    _print_rows(F, lab)
    nc = m["ncal"]
    print(f"\n零假设 ref_null/ref_main T = {m['T_null']:.1f}（阈值 30，余量 {30.0 - m['T_null']:.1f}）；ref_main 对半 T = {m['T_split']:.1f}")
    if nc:
        print(f"零分布 {nc.get('trials')} 次自举：中位 {nc.get('p50', float('nan')):.1f}，99% {nc.get('p99', float('nan')):.1f}，最大 {nc.get('max', float('nan')):.1f}")
    print(f"ref_main 每百万初级选出 {per_m:.1f}（冻结区间 [15565, 25942]）；每真实湮没选出率 {per_annih:.4e}")
    print("有效窗（widefov，探测头坐标，箱中心）：" + "；".join(
        f"{a}–{bb} mm [{lo:+.1f}, {hi:+.1f}]" for (a, bb), (lo, hi) in zip([(0, 8), (8, 15), (15, 25), (25, 35)], win)))
    print(f"三点标定：k = {cal['k']:.4f} ± {cal['k_sigma']:.4f}，b = {cal['b']:+.3f} ± {cal['b_sigma']:.3f} mm，"
          f"r² = {cal['r2']:.5f}，残差 " + " / ".join(f"{r:+.3f}" for r in cal["residuals"]) + f" mm，s = {cal['s']:.3f} mm")
    names = {"1": "T ≥ 30（三档）", "2": "|偏差| ≤ 10 mm（三档）", "3": "零假设 T 与对半 T 都 < 30",
             "4": "交叉差 ≤ 5 mm（三档）", "5": "四带 c_b 全正（三档）", "6a": "c_b(15–25) ≥ 31.9 %",
             "6b": "c_b(25–35) ≥ 26.3 %", "7": "每百万初级选出落在冻结区间",
             "8a": "0–8 带有效窗 ≥ ±16.5 mm", "8b": "25–35 带有效窗 > ±34.5 mm",
             "9": "|k − 0.844| ≤ 0.09、|b| ≤ 1、r² ≥ 0.98"}
    flips = _print_verdicts(names, V, M3_FROZEN)
    if guard:
        print("\n!! 输入核对不通过：\n   " + "\n   ".join(guard))
    _dump(repo, "m3_d2f15_summary.json", dict(
        generated=time.strftime("%Y-%m-%d %H:%M:%S"), response_dir=RESP, d2_fwhm_fraction_at_511=0.15,
        fits=_ser(F), null_calibration=nc, per_million_ref_main=per_m, per_real_annihilation=per_annih,
        fov_windows_head_mm={f"{a}-{bb}": list(wv) for (a, bb), wv in zip([(0, 8), (8, 15), (15, 25), (25, 35)], win)},
        calibration=cal, verdicts=V, verdict_flips_vs_frozen=flips, input_guard_failures=guard))
    return 0


# ---------------------------------------------------------------- summary M2（按 M2 各档配置里冻结的判据）
M2_FROZEN = {"main1": True, "main2": True, "main3": True, "main4": True, "main5": True,
             "off1": True, "off2": True, "off3": True, "off4": True, "off5": True, "off6": True,
             "o2_1": True, "o2_2": True, "o2_3": True, "o2_4": True,
             "o3_1": True, "o3_2": True, "o3_3": True, "o3_4": True, "lin": True}


def summary_m2(repo) -> int:
    names_edge = [e[0] for e in SETS["m2"]["edge"]]
    res = {n: _edge_json(repo, n) for n in names_edge}
    miss = [n for n, v in res.items() if v is None]
    if miss:
        print(f"M2 缺结果：{miss}。先跑完 run。")
        return 1
    guard = _guard(res)
    keys = ("m2_mural_edge", "m2_mural_edge_offset", "m2_mural_edge_offset2", "m2_mural_edge_offset3")
    F = {n: _fit(res[n]) for n in keys}
    lab = dict(zip(keys, ("轴心 −15", "离轴 −23", "离轴 −31", "离轴 −7")))
    m, o, o2, o3 = (F[k] for k in keys)
    V = {}
    for tag, f in (("main", m), ("off", o)):
        V[tag + "1"] = f["T"] >= 30.0
        V[tag + "2"] = abs(f["xe_world"] - f["truth"]) <= 10.0
        V[tag + "3"] = f["T_null"] < 30.0 and f["T_split"] < 30.0
        V[tag + "4"] = abs(f["cross"]) <= 5.0
        V[tag + "5"] = all(c > 0 for c in f["c"].values())
    dshift = o["xe_world"] - m["xe_world"]
    V["off6"] = abs(dshift - (-8.0)) <= 3.0
    model_rows = {}
    for tag, f, ck in (("o2_", o2, "offset2"), ("o3_", o3, "offset3")):
        V[tag + "1"] = f["T"] >= 30.0
        V[tag + "2"] = f["T_null"] < 30.0 and f["T_split"] < 30.0
        V[tag + "3"] = all(c > 0 for c in f["c"].values())
        cfg = json.loads(cfg_new(repo, SETS["m2"], ck).read_text(encoding="utf-8"))
        pred = cfg["observables"]["frozen_predictions"]["predictions_mm"]
        dist = {mn: f["xe_world"] - float(pv) for mn, pv in pred.items()}
        near = min(dist, key=lambda k: abs(dist[k]))
        model_rows[ck] = dict(predictions_mm=pred, distance_mm=dist, nearest=near,
                              nearest_sigma=abs(dist[near]) / f["sigma"] if f["sigma"] > 0 else float("nan"))
        V[tag + "4"] = abs(dist[near]) <= 5.0
    cal = _linfit([(f["truth"] - f["head"], f["xe_world"] - f["head"]) for f in F.values()])
    V["lin"] = abs(cal["b"]) <= 1.0 and cal["r2"] >= 0.98
    win = _windows(res["m2_widefov"])

    print("M2（旧构型，对照组）按 D2 能量分辨 15 % 重做：冻结判据重判（响应目录 response_d2f15）\n")
    _print_rows(F, lab)
    print(f"\n零假设 ref_null/ref_main T = {m['T_null']:.1f}；ref_main 对半 T = {m['T_split']:.1f}（四档共用同一对参照场）")
    print(f"偏移档差分：x_e(−23 档) − x_e(主档) = {dshift:+.2f} mm（冻结要求 −8.0 ± 3.0）")
    for ck, r in model_rows.items():
        print(f"模型判别 {ck}：" + "，".join(f"{mn} 差 {dv:+.2f}" for mn, dv in r["distance_mm"].items())
              + f"；最近 {r['nearest']}（{r['nearest_sigma']:.1f}σ）")
    print(f"四点线性：k = {cal['k']:.4f} ± {cal['k_sigma']:.4f}，b = {cal['b']:+.3f} ± {cal['b_sigma']:.3f} mm，"
          f"r² = {cal['r2']:.5f}，残差 " + " / ".join(f"{r:+.3f}" for r in cal["residuals"]) + f" mm，s = {cal['s']:.3f} mm；扫描对中剩余偏差 −b/k = {-cal['b'] / cal['k']:+.2f} mm")
    ks = cal['k_sigma'] if cal['k_sigma'] > 0 else float('nan')
    print(f"描述（不是判据）：k 与 1 相差 {(1 - cal['k']) / ks:.1f}σ，与几何预期 0.83 相差 {(cal['k'] - 0.83) / ks:+.1f}σ")
    print("有效窗（widefov，探测头坐标，箱中心）：" + "；".join(
        f"{a}–{bb} mm [{lo:+.1f}, {hi:+.1f}]" for (a, bb), (lo, hi) in zip([(0, 8), (8, 15), (15, 25), (25, 35)], win)))
    names = {"main1": "主档 T ≥ 30", "main2": "主档 |偏差| ≤ 10 mm", "main3": "主档 零假设与对半 T < 30",
             "main4": "主档 交叉差 ≤ 5 mm", "main5": "主档 四带 c_b 全正",
             "off1": "−23 档 T ≥ 30", "off2": "−23 档 |偏差| ≤ 10 mm", "off3": "−23 档 零假设与对半 T < 30",
             "off4": "−23 档 交叉差 ≤ 5 mm", "off5": "−23 档 四带 c_b 全正", "off6": "−23 档 差分 −8 ± 3 mm",
             "o2_1": "−31 档 T ≥ 30", "o2_2": "−31 档 零假设与对半 T < 30", "o2_3": "−31 档 四带 c_b 全正",
             "o2_4": "−31 档 最近的冻结模型相距 ≤ 5 mm",
             "o3_1": "−7 档 T ≥ 30", "o3_2": "−7 档 零假设与对半 T < 30", "o3_3": "−7 档 四带 c_b 全正",
             "o3_4": "−7 档 最近的冻结模型相距 ≤ 5 mm",
             "lin": "四点线性 |b| ≤ 1 mm 且 r² ≥ 0.98"}
    flips = _print_verdicts(names, V, M2_FROZEN)
    if guard:
        print("\n!! 输入核对不通过：\n   " + "\n   ".join(guard))
    _dump(repo, "m2_d2f15_summary.json", dict(
        generated=time.strftime("%Y-%m-%d %H:%M:%S"), response_dir=RESP, d2_fwhm_fraction_at_511=0.15,
        fits=_ser(F), offset_shift_mm=dshift, model_discrimination=model_rows, calibration=cal,
        fov_windows_head_mm={f"{a}-{bb}": list(wv) for (a, bb), wv in zip([(0, 8), (8, 15), (15, 25), (25, 35)], win)},
        verdicts=V, verdict_flips_vs_frozen=flips, input_guard_failures=guard))
    return 0


def summary(args) -> int:
    rc = 0
    if args.set in ("all", "m3"):
        rc |= summary_m3(args.repo)
        print()
    if args.set in ("all", "m2"):
        rc |= summary_m2(args.repo)
    return rc


def main() -> int:
    here = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser(description="M3 与 M2 按 D2 15 % 重做分析（不重跑 GATE）")
    ap.add_argument("--repo", type=Path, default=here)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p1 = sub.add_parser("run")
    p1.add_argument("--set", choices=("all", "m3", "m2"), default="all")
    p1.add_argument("--dry-run", action="store_true")
    p1.add_argument("--only", default="", help="response,edge,extras 中的若干个，逗号分隔；默认全部")
    p2 = sub.add_parser("summary")
    p2.add_argument("--set", choices=("all", "m3", "m2"), default="all")
    a = ap.parse_args()
    return run(a) if a.cmd == "run" else summary(a)


if __name__ == "__main__":
    sys.exit(main())
