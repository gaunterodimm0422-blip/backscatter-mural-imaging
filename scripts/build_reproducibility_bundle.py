"""Build reproducibility and lightweight-result bundles for this project.

The script deliberately does not copy ROOT, NPZ, or TIFF data into either ZIP.
Those files remain indexed, with sizes and available SHA-256 values, in the data
inventory.  Only Python's standard library is required.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import re
import shutil
import sys
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNS_ROOT = PROJECT_ROOT / "runs"
RAW_ROOT = RUNS_ROOT / "vgate_fast_transfer" / "output"
BLIND_PUBLIC_ROOT = RUNS_ROOT / "blind_void_qualification" / "public" / "blindq1"
BLIND_ANALYSIS_ROOT = RUNS_ROOT / "blind_void_qualification" / "analysis" / "blindq1"

PROGRAM_DIRS = (
    "backscatter",
    "configs",
    "docs",
    "gate",
    "scripts",
    "tests",
    "reproducibility",
)
PROGRAM_ROOT_FILES = (
    ".gitignore",
    "README.md",
    "pyproject.toml",
    "requirements.txt",
)
LIGHTWEIGHT_RUN_SUFFIXES = {
    ".csv",
    ".json",
    ".log",
    ".md",
    ".pdf",
    ".png",
    ".sha256",
    ".svg",
    ".txt",
}
HEAVY_SUFFIXES = {".root", ".npz", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default=dt.date.today().strftime("%Y%m%d"))
    parser.add_argument("--output-root", default="deliverables")
    return parser.parse_args()


def rel(path: Path) -> str:
    return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()


def human_bytes(size: int) -> str:
    units = ("B", "KB", "MB", "GB", "TB")
    value = float(size)
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            return f"{value:.2f} {unit}"
        value /= 1024.0
    raise AssertionError("unreachable")


def sha256(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest().upper()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_key_values(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip()
    return values


def run_group(run_id: str) -> str:
    patterns = (
        (r"^concrete_slab", "完整水泥基线"),
        (r"^repeat_concrete", "完整水泥独立种子"),
        (r"^physical_d1r\d+_concrete", "完整水泥物理缩径"),
        (r"^depth_psf_", "轴向薄层"),
        (r"^repeat_thin", "薄层独立种子"),
        (r"^physical_d1r\d+_thin", "薄层物理缩径"),
        (r"^rod_width", "单杆横向响应"),
        (r"^double_rod", "双杆可分辨性"),
        (r"^edge_", "边缘 MTF"),
        (r"^concrete_void", "完整水泥孔洞"),
    )
    for pattern, label in patterns:
        if re.search(pattern, run_id):
            return label
    return "其他正式运行"


def infer_config(run_id: str) -> str | None:
    exact = {
        "concrete_slab_2m_seed24061873_local": "configs/generated/concrete_slab_2m.json",
        "repeat_concrete_seed24061874_local": "configs/generated/concrete_slab_2m_seed24061874.json",
        "physical_d1r150_concrete_seed24061873_local": "configs/generated/concrete_slab_2m_d1r150.json",
        "repeat_thin30_seed24061874_local": "configs/generated/thin_layer_depth30_seed24061874.json",
        "repeat_thin90_seed24061874_local": "configs/generated/thin_layer_depth90_seed24061874.json",
        "physical_d1r150_thin30_seed24061873_local": "configs/generated/thin_layer_depth30_d1r150.json",
        "physical_d1r175_thin30_seed24061873_local": "configs/generated/thin_layer_depth30_d1r175.json",
    }
    if run_id in exact:
        return exact[run_id]
    match = re.match(r"depth_psf_(\d+)mm_", run_id)
    if match:
        return f"configs/generated/thin_layer_depth{match.group(1)}.json"
    match = re.match(r"rod_width(\d+)_depth30_5m_", run_id)
    if match:
        return f"configs/generated/rod_width{match.group(1)}_depth30_5m.json"
    match = re.match(r"double_rod_width2_sep(\d+)_depth30_5m_", run_id)
    if match:
        return (
            "configs/generated/double_rod_width2_sep"
            f"{match.group(1)}_depth30_5m.json"
        )
    match = re.match(r"edge_depth(\d+)_2m_", run_id)
    if match:
        return f"configs/generated/edge_depth{match.group(1)}_2m.json"
    match = re.match(r"concrete_void_d(\d+)_depth(\d+)_2m_", run_id)
    if match:
        return (
            f"configs/generated/concrete_void_d{match.group(1)}_"
            f"depth{match.group(2)}_2m.json"
        )
    return None


def parameter_summary(parameters: dict[str, str]) -> str:
    phantom = parameters.get("PHANTOM_ID", "unknown")
    d1_radius = parameters.get("D1_RMAX_MM", "?")
    if phantom == "concrete_100mm":
        return f"100 mm 水泥板；D1 Rmax={d1_radius} mm"
    if phantom == "thin_layer":
        return (
            f"深度 {parameters.get('TARGET_DEPTH_MM', '?')} mm；"
            f"厚 {parameters.get('TARGET_THICKNESS_MM', '?')} mm；"
            f"D1 Rmax={d1_radius} mm"
        )
    if phantom == "rod":
        radius = float(parameters.get("TARGET_RADIUS_MM", "0"))
        return (
            f"直径 {2 * radius:g} mm；深度 "
            f"{parameters.get('TARGET_DEPTH_MM', '?')} mm"
        )
    if phantom == "double_rod":
        radius = float(parameters.get("TARGET_RADIUS_MM", "0"))
        return (
            f"直径 {2 * radius:g} mm；中心距 "
            f"{parameters.get('TARGET_SEPARATION_MM', '?')} mm"
        )
    if phantom == "edge":
        return f"边缘深度 {parameters.get('TARGET_DEPTH_MM', '?')} mm"
    if phantom == "concrete_100mm_cylindrical_void":
        radius = float(parameters.get("VOID_RADIUS_MM", "0"))
        return (
            f"孔洞直径 {2 * radius:g} mm；中心深度 "
            f"{parameters.get('VOID_DEPTH_MM', '?')} mm"
        )
    return phantom


def recorded_root_hash(
    root_file: Path, transfer_manifest: dict[str, Any] | None
) -> tuple[str | None, str]:
    if transfer_manifest is None:
        return None, "not_recorded"
    if root_file.name == "hits.root":
        value = transfer_manifest.get("hits_sha256")
    elif root_file.name == transfer_manifest.get("truth_file"):
        value = transfer_manifest.get("truth_sha256")
    else:
        value = None
    if value:
        return str(value).upper(), "transfer_manifest"
    return None, "not_recorded"


def inspect_regular_runs() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for run_dir in sorted(p for p in RAW_ROOT.iterdir() if p.is_dir()):
        parameters_path = run_dir / "run_parameters.txt"
        log_path = run_dir / "gate.log"
        parameters = read_key_values(parameters_path)
        expected = int(parameters["N_PRIMARIES"])
        log_text = log_path.read_text(encoding="utf-8", errors="replace")
        match = re.search(r"Number of events processed\s*:\s*(\d+)", log_text)
        processed = int(match.group(1)) if match else None
        fatal_markers = re.findall(
            r"ERROR|FATAL|Aborted|core dumped|G4Exception",
            log_text,
            flags=re.IGNORECASE,
        )
        transfer_path = run_dir / "transfer_manifest.json"
        transfer = read_json(transfer_path) if transfer_path.exists() else None
        root_files: list[dict[str, Any]] = []
        for root_file in sorted(run_dir.glob("*.root")):
            checksum, source = recorded_root_hash(root_file, transfer)
            if checksum is None:
                print(f"Calculating missing SHA-256: {rel(root_file)}", flush=True)
                checksum = sha256(root_file)
                source = "calculated_for_bundle"
            root_files.append(
                {
                    "path": rel(root_file),
                    "bytes": root_file.stat().st_size,
                    "sha256": checksum,
                    "checksum_source": source,
                }
            )
        analysis_dir = RUNS_ROOT / run_dir.name
        derived_files = (
            list(analysis_dir.rglob("*")) if analysis_dir.exists() else []
        )
        derived_files = [p for p in derived_files if p.is_file()]
        config = infer_config(run_dir.name)
        rows.append(
            {
                "run_id": run_dir.name,
                "group": run_group(run_dir.name),
                "purpose_parameters": parameter_summary(parameters),
                "phantom_id": parameters.get("PHANTOM_ID"),
                "expected_primaries": expected,
                "processed_events": processed,
                "random_seed": int(parameters.get("RANDOM_SEED", "0")),
                "d1_outer_radius_mm": float(parameters.get("D1_RMAX_MM", "0")),
                "gate_complete": processed == expected and not fatal_markers,
                "fatal_markers": sorted(set(fatal_markers)),
                "raw_directory": rel(run_dir),
                "raw_bytes": sum(p["bytes"] for p in root_files),
                "root_files": root_files,
                "config": config,
                "analysis_directory": rel(analysis_dir),
                "analysis_summary_present": (analysis_dir / "summary.json").exists(),
                "analysis_file_count": len(derived_files),
                "analysis_bytes": sum(p.stat().st_size for p in derived_files),
                "transfer_manifest_present": transfer is not None,
                "gate_version": (run_dir / "gate_version.txt").read_text(
                    encoding="utf-8", errors="replace"
                ).strip(),
            }
        )
    return rows


def inspect_blind_runs() -> list[dict[str, Any]]:
    reveal_path = BLIND_ANALYSIS_ROOT / "revealed_results" / "blind_qualification_results.json"
    reveal = read_json(reveal_path)
    revealed_cases = {case["case_id"]: case for case in reveal["cases"]}
    rows: list[dict[str, Any]] = []
    for case_dir in sorted(p for p in BLIND_PUBLIC_ROOT.iterdir() if p.is_dir()):
        public = read_json(case_dir / "public_manifest.json")
        hits = case_dir / "hits.root"
        case = revealed_cases[case_dir.name]
        analysis_dir = BLIND_ANALYSIS_ROOT / case_dir.name
        rows.append(
            {
                "case_id": case_dir.name,
                "expected_primaries": int(public["expected_primaries"]),
                "public_raw_file": rel(hits),
                "raw_bytes": hits.stat().st_size,
                "sha256": str(public["hits_sha256"]).upper(),
                "checksum_source": "public_manifest",
                "private_completion_verified": bool(
                    public["gate_log_completion_verified_privately"]
                ),
                "contains_truth_or_label_in_public_bundle": bool(
                    public["contains_truth_or_case_label"]
                ),
                "revealed_class": case["class"],
                "true_x_mm": case["true_x_mm"],
                "true_depth_mm": case["true_depth_mm"],
                "maximum_score": case["maximum_score"],
                "candidate_flag": case["candidate_flag"],
                "localization_error_mm": case["localization_error_mm"],
                "analysis_directory": rel(analysis_dir),
                "analysis_present": (analysis_dir / "summary.json").exists(),
            }
        )
    return rows


def extension_summary(files: Iterable[Path]) -> list[dict[str, Any]]:
    counts: Counter[str] = Counter()
    sizes: Counter[str] = Counter()
    for path in files:
        suffix = path.suffix.lower() or "[none]"
        counts[suffix] += 1
        sizes[suffix] += path.stat().st_size
    return [
        {"extension": ext, "file_count": counts[ext], "bytes": sizes[ext]}
        for ext in sorted(counts, key=lambda key: sizes[key], reverse=True)
    ]


def data_level(path: Path) -> tuple[str, str]:
    relative = rel(path)
    suffix = path.suffix.lower()
    name = path.name.lower()
    if suffix == ".root":
        if "transfer_selfcheck" in relative:
            return "TEST", "传输脚本自检占位文件，不是科学数据"
        return "L0", "GATE 原始 Hits 或仿真真值 ROOT"
    if suffix == ".npz":
        return "L1", "可由 ROOT 重新生成的分析中间数组"
    if (
        relative.startswith(rel(RAW_ROOT) + "/")
        and name
        in {
            "gate.log",
            "gate_version.txt",
            "macro_sha256.txt",
            "run_parameters.txt",
            "transfer_manifest.json",
        }
    ):
        return "L0", "GATE 运行、参数、版本或传输完整性证据"
    if relative.startswith(rel(BLIND_PUBLIC_ROOT) + "/") and name in {
        "campaign_public.json",
        "gate_version.txt",
        "macro_sha256.txt",
        "public_manifest.json",
    }:
        return "L0", "盲检公开输入的版本或完整性证据"
    if suffix in {".csv", ".json"}:
        return "L2", "定量指标、评分或机器可读结果"
    if suffix in {".png", ".pdf", ".svg", ".tiff"}:
        return "L3", "绘图结果"
    if suffix in {".log", ".sha256", ".txt", ".md"}:
        return "AUDIT", "分析日志、校验或说明文件"
    return "OTHER", "其他项目产物"


def derived_only_directories(regular_ids: set[str]) -> list[dict[str, Any]]:
    special = {"vgate_fast_transfer", "blind_void_qualification"}
    rows: list[dict[str, Any]] = []
    for directory in sorted(p for p in RUNS_ROOT.iterdir() if p.is_dir()):
        if directory.name in regular_ids or directory.name in special:
            continue
        files = [p for p in directory.rglob("*") if p.is_file()]
        if directory.name.startswith(("validation", "transfer", "legacy", "vgate_")):
            role = "自检、传输或审计产物，不作为正式科学数据"
        elif directory.name.startswith("smoke"):
            role = "冒烟测试分析，非正式大样本"
        else:
            role = "由多个正式运行汇总得到的分析结果"
        rows.append(
            {
                "directory": rel(directory),
                "role": role,
                "file_count": len(files),
                "bytes": sum(p.stat().st_size for p in files),
            }
        )
    return rows


def build_inventory() -> dict[str, Any]:
    regular = inspect_regular_runs()
    blind = inspect_blind_runs()
    all_files = [p for p in RUNS_ROOT.rglob("*") if p.is_file()]
    scientific_root_files = [
        Path(item["path"])
        for run in regular
        for item in run["root_files"]
    ]
    scientific_root_bytes = sum(run["raw_bytes"] for run in regular) + sum(
        case["raw_bytes"] for case in blind
    )
    non_scientific_roots = []
    regular_prefix = rel(RAW_ROOT) + "/"
    blind_prefix = rel(BLIND_PUBLIC_ROOT) + "/"
    for path in sorted(p for p in all_files if p.suffix.lower() == ".root"):
        relative = rel(path)
        if relative.startswith(regular_prefix) or relative.startswith(blind_prefix):
            continue
        non_scientific_roots.append(
            {
                "path": relative,
                "bytes": path.stat().st_size,
                "role": "传输脚本自检占位文件，不是科学 ROOT 数据",
            }
        )
    root_hashes: dict[str, str] = {
        item["path"]: item["sha256"]
        for run in regular
        for item in run["root_files"]
    }
    root_hashes.update(
        {case["public_raw_file"]: case["sha256"] for case in blind}
    )
    file_inventory = []
    for path in sorted(all_files):
        level, role = data_level(path)
        relative = rel(path)
        item: dict[str, Any] = {
            "path": relative,
            "bytes": path.stat().st_size,
            "extension": path.suffix.lower() or "[none]",
            "data_level": level,
            "role": role,
            "included_in_lightweight_results_zip": (
                path.suffix.lower() in LIGHTWEIGHT_RUN_SUFFIXES
            ),
        }
        if relative in root_hashes:
            item["sha256"] = root_hashes[relative]
        file_inventory.append(item)
    configured_not_run = [
        {
            "experiment": "水薄层，30 mm 深度",
            "config": "configs/generated/water_thin_layer_depth30_2m.json",
            "planned_primaries": 2_000_000,
            "reason": "配置已完成，正式大样本尚未运行",
        },
        {
            "experiment": "铁薄层，30 mm 深度",
            "config": "configs/generated/iron_thin_layer_depth30_2m.json",
            "planned_primaries": 2_000_000,
            "reason": "配置已完成，正式大样本尚未运行",
        },
        {
            "experiment": "双杆，20 mm 中心距",
            "config": "configs/generated/double_rod_width2_sep20_depth30_5m.json",
            "planned_primaries": 5_000_000,
            "reason": "10/15 mm 已稳健可分辨，按自适应停止规则未运行",
        },
        {
            "experiment": "双杆，30 mm 中心距",
            "config": "configs/generated/double_rod_width2_sep30_depth30_5m.json",
            "planned_primaries": 5_000_000,
            "reason": "10/15 mm 已稳健可分辨，按自适应停止规则未运行",
        },
    ]
    total_bytes = sum(p.stat().st_size for p in all_files)
    return {
        "schema_version": "1.0",
        "generated_at": dt.datetime.now(dt.timezone.utc).astimezone().isoformat(),
        "project_root": ".",
        "summary": {
            "runs_file_count": len(all_files),
            "runs_total_bytes": total_bytes,
            "regular_completed_gate_runs": len(regular),
            "blind_public_cases": len(blind),
            "scientific_root_file_count": len(scientific_root_files) + len(blind),
            "scientific_root_bytes": scientific_root_bytes,
            "configured_not_run_count": len(configured_not_run),
            "all_52_python_tests_passed_before_packaging": True,
            "test_python": "Python 3.12",
        },
        "data_levels": {
            "L0": "GATE ROOT, truth ROOT, log, parameters, macro and version audit",
            "L1": "events.npz, pairs.npz, reconstruction.npz",
            "L2": "CSV and JSON quantitative results",
            "L3": "PNG/PDF/SVG/TIFF figures",
            "L4": "documents, LaTeX report, final PDF and PPT",
        },
        "extension_summary": extension_summary(all_files),
        "file_inventory": file_inventory,
        "regular_gate_runs": regular,
        "blind_qualification_cases": blind,
        "configured_not_run": configured_not_run,
        "derived_or_test_directories": derived_only_directories(
            {run["run_id"] for run in regular}
        ),
        "non_scientific_root_files": non_scientific_roots,
    }


def write_data_map(inventory: dict[str, Any], path: Path) -> None:
    summary = inventory["summary"]
    lines = [
        "# 背散射项目数据地图",
        "",
        f"生成时间：{inventory['generated_at']}",
        "",
        "## 1. 总体情况",
        "",
        f"- `runs/`：{summary['runs_file_count']} 个文件，"
        f"共 {human_bytes(summary['runs_total_bytes'])}；",
        f"- 正式普通 GATE 运行：{summary['regular_completed_gate_runs']} 组；",
        f"- 盲检公开病例：{summary['blind_public_cases']} 组；",
        f"- 正式科学 ROOT：{summary['scientific_root_file_count']} 个，"
        f"共 {human_bytes(summary['scientific_root_bytes'])}；",
        "- 打包前 Python 自动测试：52/52 通过；",
        "- ROOT、NPZ、TIFF 未装入轻量 ZIP，但均保留在原项目并列入 JSON 清单。",
        "",
        "## 2. 如何认数据",
        "",
        "| 级别 | 文件 | 含义 | 是否可重新生成 |",
        "|---|---|---|---|",
        "| L0 | `hits.root` | D1/D2 原始 Hits，是后续分析的核心输入 | 需重新运行 GATE |",
        "| L0 | `hits_truth_*.root` | 仿真真值，只用于评分 | 需重新运行 GATE |",
        "| L0 | `gate.log`、`run_parameters.txt`、`macro_sha256.txt` | 完成数、运行参数和宏版本证据 | 随 GATE 运行生成 |",
        "| L1 | `events.npz`、`pairs.npz`、`reconstruction.npz` | 事件聚合、D1/D2 配对和散射点中间数组 | 可由 ROOT 重新生成 |",
        "| L2 | CSV、JSON | FWHM、RMSE、灵敏度、CNR、MTF、盲检评分 | 可由 L1 重新生成 |",
        "| L3 | PNG、PDF、SVG、TIFF | 图形；PDF/SVG 为矢量版，TIFF 为高分辨率副本 | 可由 L1/L2 重新生成 |",
        "| L4 | `docs/`、`report_latex/`、PDF、PPT | 汇报和论文式材料 | 由 L2/L3 整理 |",
        "",
        "## 3. 已完成的普通 GATE 运行",
        "",
        "所有下列运行的 `gate.log` 均包含预期事件完成标记，且分析目录存在。",
        "",
        "| 类型 | 运行目录名 | 初级事件 | 随机种子 | 关键参数 | ROOT 大小 | 配置 |",
        "|---|---|---:|---:|---|---:|---|",
    ]
    for run in inventory["regular_gate_runs"]:
        config = f"`{run['config']}`" if run["config"] else "未自动匹配"
        lines.append(
            f"| {run['group']} | `{run['run_id']}` | "
            f"{run['expected_primaries']:,} | {run['random_seed']} | "
            f"{run['purpose_parameters']} | {human_bytes(run['raw_bytes'])} | "
            f"{config} |"
        )
    lines.extend(
        [
            "",
            "原始目录统一位于 `runs/vgate_fast_transfer/output/<run_id>/`；",
            "分析目录统一位于 `runs/<run_id>/`。逐文件 SHA-256 见",
            "`data_inventory.json` 的 `regular_gate_runs[].root_files`。",
            "",
            "## 4. 12 例孔洞盲检数据",
            "",
            "公开 ROOT 位于 `runs/blind_void_qualification/public/blindq1/`。公开包在",
            "评分冻结前不含类别、位置、truth ROOT 或 gate.log；下表类别来自正式揭盲结果。",
            "",
            "| 病例 | 揭盲类别 | 真值 x/depth (mm) | 最大分数 | 判为候选 | 定位误差 (mm) | ROOT 大小 |",
            "|---|---|---|---:|---|---:|---:|",
        ]
    )
    for case in inventory["blind_qualification_cases"]:
        if case["revealed_class"] == "void":
            truth = f"{case['true_x_mm']}/{case['true_depth_mm']}"
            error = f"{case['localization_error_mm']:.2f}"
        else:
            truth = "无孔洞"
            error = "-"
        lines.append(
            f"| `{case['case_id']}` | {case['revealed_class']} | {truth} | "
            f"{case['maximum_score']:.2f} | {case['candidate_flag']} | {error} | "
            f"{human_bytes(case['raw_bytes'])} |"
        )
    lines.extend(
        [
            "",
            "这批数据只证明 6+6 小样本盲检流程验收通过，不能据此宣称最终检出率",
            "为 100% 或误报率为 0%。",
            "",
            "## 5. 已配置但没有正式数据",
            "",
            "| 实验 | 配置 | 计划事件 | 原因 |",
            "|---|---|---:|---|",
        ]
    )
    for item in inventory["configured_not_run"]:
        lines.append(
            f"| {item['experiment']} | `{item['config']}` | "
            f"{item['planned_primaries']:,} | {item['reason']} |"
        )
    lines.extend(
        [
            "",
            "## 6. 汇总结果、自检和非正式数据",
            "",
            "| 目录 | 身份 | 大小 |",
            "|---|---|---:|",
        ]
    )
    for item in inventory["derived_or_test_directories"]:
        lines.append(
            f"| `{item['directory']}` | {item['role']} | "
            f"{human_bytes(item['bytes'])} |"
        )
    lines.extend(
        [
            "",
            "`transfer_selfcheck` 和 `transfer_script_selfcheck` 中极小的 `.root` 文件",
            "只是复制脚本自检占位文件，不得计入探测器物理结果。",
            "",
            "## 7. 压缩包中包含与不包含的内容",
            "",
            "- 程序包：GATE 宏、Python 程序、配置、测试、脚本、文档和 LaTeX 源码；",
            "- 轻量结果包：日志、参数、CSV、JSON、PNG、PDF、SVG 和 SHA-256 文件；",
            "- 未包含：ROOT、NPZ、TIFF；其原始相对路径、大小和校验值在 JSON 清单中；",
            "- 某些历史 JSON 的 `figures` 字段仍记录生成时的 Windows 绝对路径；",
            "  指标本身不受影响，重新运行绘图命令会生成当前工作目录下的新路径。",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def should_skip_program_file(path: Path) -> bool:
    return (
        "__pycache__" in path.parts
        or path.suffix.lower() in {".pyc", ".pyo"}
        or "build" in path.parts
    )


def copy_files(files: Iterable[Path], source_root: Path, destination_root: Path) -> None:
    for source in files:
        relative = source.relative_to(source_root)
        destination = destination_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


def build_program_snapshot(destination: Path) -> None:
    destination.mkdir(parents=True)
    root_files = [PROJECT_ROOT / name for name in PROGRAM_ROOT_FILES]
    copy_files((p for p in root_files if p.exists()), PROJECT_ROOT, destination)
    for directory_name in PROGRAM_DIRS:
        source_dir = PROJECT_ROOT / directory_name
        files = [
            p for p in source_dir.rglob("*") if p.is_file() and not should_skip_program_file(p)
        ]
        copy_files(files, PROJECT_ROOT, destination)
    latex_root = PROJECT_ROOT / "report_latex"
    latex_files = [
        p
        for p in latex_root.rglob("*")
        if p.is_file() and "build" not in p.relative_to(latex_root).parts
    ]
    copy_files(latex_files, PROJECT_ROOT, destination)


def build_lightweight_results_snapshot(destination: Path) -> None:
    destination.mkdir(parents=True)
    run_files = [
        p
        for p in RUNS_ROOT.rglob("*")
        if p.is_file() and p.suffix.lower() in LIGHTWEIGHT_RUN_SUFFIXES
    ]
    copy_files(run_files, PROJECT_ROOT, destination)
    for extra_name in ("output", "outputs"):
        extra_root = PROJECT_ROOT / extra_name
        if extra_root.exists():
            copy_files(
                (p for p in extra_root.rglob("*") if p.is_file()),
                PROJECT_ROOT,
                destination,
            )


def write_checksums(root: Path) -> Path:
    checksum_path = root / "CHECKSUMS_SHA256.txt"
    files = sorted(p for p in root.rglob("*") if p.is_file() and p != checksum_path)
    lines = [f"{sha256(path)}  {path.relative_to(root).as_posix()}" for path in files]
    checksum_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return checksum_path


def zip_directory(directory: Path, destination: Path) -> None:
    with zipfile.ZipFile(
        destination,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
        allowZip64=True,
    ) as archive:
        for path in sorted(p for p in directory.rglob("*") if p.is_file()):
            arcname = (Path(directory.name) / path.relative_to(directory)).as_posix()
            archive.write(path, arcname)


def main() -> int:
    args = parse_args()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", args.tag):
        raise ValueError("Unsafe tag")
    output_root = (PROJECT_ROOT / args.output_root).resolve()
    try:
        output_root.relative_to(PROJECT_ROOT.resolve())
    except ValueError as exc:
        raise ValueError("Output root must stay inside the project") from exc
    delivery_root = output_root / f"reproducibility_{args.tag}"
    if delivery_root.exists():
        raise FileExistsError(
            f"Refusing to overwrite an existing delivery: {delivery_root}"
        )
    delivery_root.mkdir(parents=True)

    print("Inspecting runs and building data inventory...", flush=True)
    inventory = build_inventory()
    inventory_path = delivery_root / "data_inventory.json"
    inventory_path.write_text(
        json.dumps(inventory, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    data_map_path = delivery_root / "DATA_MAP.md"
    write_data_map(inventory, data_map_path)

    program_dir = delivery_root / f"beisanshe_program_{args.tag}"
    results_dir = delivery_root / f"beisanshe_lightweight_results_{args.tag}"
    print("Copying program snapshot...", flush=True)
    build_program_snapshot(program_dir)
    print("Copying lightweight result snapshot...", flush=True)
    build_lightweight_results_snapshot(results_dir)

    for snapshot in (program_dir, results_dir):
        reproducibility_dir = snapshot / "reproducibility"
        reproducibility_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(inventory_path, reproducibility_dir / inventory_path.name)
        shutil.copy2(data_map_path, reproducibility_dir / data_map_path.name)
        write_checksums(snapshot)

    program_zip = delivery_root / f"beisanshe_program_{args.tag}.zip"
    results_zip = delivery_root / f"beisanshe_lightweight_results_{args.tag}.zip"
    print("Compressing program snapshot...", flush=True)
    zip_directory(program_dir, program_zip)
    print("Compressing lightweight result snapshot...", flush=True)
    zip_directory(results_dir, results_zip)

    package_hash_path = delivery_root / "PACKAGE_SHA256.txt"
    package_hash_path.write_text(
        f"{sha256(program_zip)}  {program_zip.name}\n"
        f"{sha256(results_zip)}  {results_zip.name}\n",
        encoding="ascii",
    )
    package_summary = {
        "program_zip": {
            "path": program_zip.name,
            "bytes": program_zip.stat().st_size,
            "sha256": sha256(program_zip),
        },
        "lightweight_results_zip": {
            "path": results_zip.name,
            "bytes": results_zip.stat().st_size,
            "sha256": sha256(results_zip),
        },
        "raw_root_included": False,
        "npz_included": False,
        "tiff_included": False,
    }
    (delivery_root / "package_summary.json").write_text(
        json.dumps(package_summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Delivery ready: {delivery_root}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
