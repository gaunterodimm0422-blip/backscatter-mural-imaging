"""S0b 检查：确认"探测头横向平移"的位姿实现是否正确。

在 Windows 上运行（本机无全局 Python，要写全路径）：
    E:\\python3.12\\python.exe C:\\Users\\HOUWENDI\\Documents\\beisanshe\\scripts\\check_pose_smoke.py
可选指定数据目录（装着 pose_smoke_L* 那一层）：
    ... check_pose_smoke.py --root "E:\\vbgate\\share\\beisanshe_v1\\gate\\output"

只读 ROOT 文件，不写任何结果，不碰已有产物。

核心思想：水泥板横向 500 mm、均匀，探测头平行平移时，
**在探测头自身坐标系里，各档的分布应当完全重合**。任何不重合都说明平移实现有问题。

六条判据：
  1. 每档都有 hits.root；
  2. D2 命中的 x 质心（世界坐标）≈ 平移量 L，偏差 < 3 mm —— 探测头确实动了；
  3. 符合事件数（D1、D2 同一事件且都有能量沉积）各档之间差异 < 4 倍泊松标准差；
  4. D1 落在 180-250 keV 背散射窗内的事件占比，各档相对偏差 < 15%；
  5. D1 命中半径中位数（相对探测头中心，即 x-L）各档偏差 < 5 mm；
  6. D2 全能峰位各档偏差 < 5 keV。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOTS = [
    PROJECT_ROOT / "runs",
    PROJECT_ROOT / "runs" / "pose_smoke",
    PROJECT_ROOT / "gate" / "output",
    PROJECT_ROOT / "output",
]
D1_TREE, D2_TREE = "Hits_BSdetector", "Hits_crystal"
D1_WINDOW_KEV = (180.0, 250.0)
BRANCHES = ["eventID", "edep", "posX", "posY", "posZ"]


def find_runs(root: Path) -> list[tuple[float, Path]]:
    runs = []
    for d in sorted(root.glob("pose_smoke_L*")):
        if not (d / "hits.root").exists():
            continue
        try:
            runs.append((float(d.name.split("_L")[-1]), d))
        except ValueError:
            continue
    return sorted(runs)


def read_params(d: Path) -> dict:
    """读取 run_parameters.txt，用于核对这批数据到底是什么运行。"""
    out = {}
    f = d / "run_parameters.txt"
    if f.exists():
        for line in f.read_text(encoding="utf-8", errors="replace").splitlines():
            if "=" in line:
                k, _, v = line.partition("=")
                out[k.strip()] = v.strip()
    return out


def per_event(arrays: dict) -> dict:
    """把 Hits 按 eventID 聚合成每事件一条记录（能量求和、位置按能量加权）。"""
    edep = np.asarray(arrays["edep"], dtype=np.float64) * 1000.0     # MeV -> keV
    keep = edep > 0.0
    if not keep.any():
        return {"event": np.array([], dtype=np.int64), "energy": np.array([]),
                "x": np.array([]), "y": np.array([])}
    ev = np.asarray(arrays["eventID"], dtype=np.int64)[keep]
    x = np.asarray(arrays["posX"], dtype=np.float64)[keep]
    y = np.asarray(arrays["posY"], dtype=np.float64)[keep]
    e = edep[keep]
    order = np.argsort(ev, kind="stable")
    ev, x, y, e = ev[order], x[order], y[order], e[order]
    starts = np.concatenate([[0], np.flatnonzero(np.diff(ev)) + 1])
    etot = np.add.reduceat(e, starts)
    safe = np.maximum(etot, 1e-12)
    return {
        "event": ev[starts],
        "energy": etot,
        "x": np.add.reduceat(e * x, starts) / safe,
        "y": np.add.reduceat(e * y, starts) / safe,
    }


def peak_keV(energy: np.ndarray) -> float:
    near = energy[(energy > 400.0) & (energy < 600.0)]
    if near.size < 20:
        return float("nan")
    h, edges = np.histogram(near, bins=60, range=(400.0, 600.0))
    i = int(h.argmax())
    return float(0.5 * (edges[i] + edges[i + 1]))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=None)
    args = ap.parse_args()

    try:
        import uproot
    except ImportError:
        print("需要 uproot：E:\\python3.12\\python.exe -m pip install uproot")
        return 2

    runs = []
    for r in ([args.root] if args.root else DEFAULT_ROOTS):
        if r and r.exists():
            runs = find_runs(r)
            if runs:
                print(f"数据目录：{r}\n")
                break
    if not runs:
        print("没找到 pose_smoke_L* 目录。请把虚拟机 gate/output/pose_smoke_L* 回传到下列任一位置：")
        for r in DEFAULT_ROOTS:
            print(f"  {r}")
        return 1

    # ---- 数据来源核对：防止在旧数据上误报通过 ----
    print("本次读到的数据：")
    bad = []
    for L, d in runs:
        pm = read_params(d)
        phantom = pm.get("PHANTOM_MAC", "未知")
        nprim = pm.get("N_PRIMARIES", "未知")
        print(f"  L={L:>4.0f}  体模={phantom:<24} 初级事件={nprim:<10} {d.name}")
        if phantom != "phantom_concrete.mac":
            bad.append(f"L={L:.0f} 的体模是 {phantom}，S0b 要求 phantom_concrete.mac")
        try:
            if int(nprim) < 50000:
                bad.append(f"L={L:.0f} 只有 {nprim} 个初级事件，S0b 要求至少 50000")
        except ValueError:
            bad.append(f"L={L:.0f} 读不到 N_PRIMARIES")
    print()
    if bad:
        print("这不是 S0b 的数据，检查中止：")
        for b in bad:
            print(f"  - {b}")
        print("\n请确认新一轮已经在虚拟机跑完，并把 output/pose_smoke_L* 覆盖回")
        print(f"  {runs[0][1].parent}")
        return 1

    rows = []
    for L, d in runs:
        with uproot.open(d / "hits.root") as f:
            keys = {k.split(";")[0] for k in f.keys()}
            if D1_TREE not in keys or D2_TREE not in keys:
                print(f"L={L:.0f}: 缺少 {D1_TREE} 或 {D2_TREE}，实际有 {sorted(keys)}")
                return 1
            e1 = per_event(f[D1_TREE].arrays(BRANCHES, library="np"))
            e2 = per_event(f[D2_TREE].arrays(BRANCHES, library="np"))

        common = np.intersect1d(e1["event"], e2["event"], assume_unique=True)
        n_coin = int(common.size)
        sel1 = np.isin(e1["event"], common)
        d1_e = e1["energy"][sel1]
        d1_r = np.hypot(e1["x"][sel1] - L, e1["y"][sel1])          # 相对探测头中心
        in_win = (d1_e >= D1_WINDOW_KEV[0]) & (d1_e <= D1_WINDOW_KEV[1])
        rows.append({
            "L": L,
            "n_d1": int(e1["event"].size),
            "n_d2": int(e2["event"].size),
            "n_coin": n_coin,
            "d2_cx": float(np.median(e2["x"])) if e2["x"].size else float("nan"),
            "d2_cx_head": (float(np.median(e2["x"])) - L) if e2["x"].size else float("nan"),
            "d1_r_med": float(np.median(d1_r)) if d1_r.size else float("nan"),
            "n_win": int(in_win.sum()),
            "f_win": float(in_win.mean()) if d1_e.size else float("nan"),
            "peak": peak_keV(e2["energy"]),
        })

    hdr = (f"{'L(mm)':>6}{'D1事件':>8}{'D2事件':>8}{'符合':>7}"
           f"{'D2质心x':>9}{'头系D2x':>9}{'D1半径中位':>11}"
           f"{'180-250窗':>10}{'窗内占比':>9}{'D2峰位':>8}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['L']:>6.0f}{r['n_d1']:>8d}{r['n_d2']:>8d}{r['n_coin']:>7d}"
              f"{r['d2_cx']:>9.2f}{r['d2_cx_head']:>9.2f}{r['d1_r_med']:>11.1f}"
              f"{r['n_win']:>10d}{r['f_win']:>9.3f}{r['peak']:>8.1f}")
    print("-" * len(hdr))

    # ---- 判据 ----
    fails = []
    for r in rows:
        if abs(r["d2_cx"] - r["L"]) >= 3.0:
            fails.append(f"L={r['L']:.0f}: D2 质心 {r['d2_cx']:.2f} 与平移量 {r['L']:.0f} 差 >3 mm")

    coin = np.array([r["n_coin"] for r in rows], dtype=float)
    if coin.min() > 0:
        ref = coin.mean()
        tol = 4.0 * np.sqrt(ref)
        if (np.abs(coin - ref) > tol).any():
            fails.append(f"符合事件数各档差异超出 4σ 泊松容差（均值 {ref:.0f}，容差 ±{tol:.0f}，"
                         f"实际 {coin.astype(int).tolist()}）")
    else:
        fails.append("有位姿的符合事件数为 0")

    fw = np.array([r["f_win"] for r in rows], dtype=float)
    if np.isfinite(fw).all() and fw.mean() > 0:
        if (np.abs(fw - fw.mean()) / fw.mean() > 0.15).any():
            fails.append(f"180-250 keV 窗内占比各档相对偏差 >15%：{np.round(fw, 3).tolist()}")

    rr = np.array([r["d1_r_med"] for r in rows], dtype=float)
    if np.isfinite(rr).all() and (rr.max() - rr.min() > 5.0):
        fails.append(f"D1 半径中位数（头系）各档跨度 {rr.max()-rr.min():.1f} mm >5 mm：{np.round(rr,1).tolist()}")

    pk = np.array([r["peak"] for r in rows], dtype=float)
    if np.isfinite(pk).all() and (pk.max() - pk.min() > 5.0):
        fails.append(f"D2 峰位各档跨度 {pk.max()-pk.min():.1f} keV >5 keV：{np.round(pk,1).tolist()}")

    if fails:
        print("S0b 未通过：")
        for f in fails:
            print(f"  - {f}")
        print("\n把上表和 gate.log 发回，先不要跑正式任务。")
        return 1

    print("S0b 通过：探测头平移实现正确，在探测头坐标系里四档分布一致。")
    print("可以进入 S1/S2。")
    print("\n提醒：本次用的是均匀水泥板，符合计数本来就不该随 L 变化——")
    print("它证明的是平移实现干净，不代表定点靶的产额不随 L 衰减。")
    print("定点靶的产额衰减要由 S2 的孔洞体模来测。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
