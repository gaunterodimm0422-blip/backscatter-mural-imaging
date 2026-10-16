# 背散射项目复现资料说明

本目录用于回答两个问题：

1. 哪些程序和参数能够重新产生当前结果；
2. 当前硬盘上的数据分别是什么、位于哪里、是否属于正式科学数据。

## 交付物

执行 `scripts/build_reproducibility_bundle.ps1` 后会产生：

- `beisanshe_program_<日期>.zip`：GATE 宏、运行脚本、Python 分析程序、配置、测试、研究文档和 LaTeX 报告源码；
- `beisanshe_lightweight_results_<日期>.zip`：CSV、JSON、PNG、PDF、SVG、日志和参数文件；
- `DATA_MAP.md`：人可以直接阅读的数据地图；
- `data_inventory.json`：机器可读的完整数据清单；
- `PACKAGE_SHA256.txt`：两个压缩包的 SHA-256。

## 为什么原始 ROOT 没放进程序压缩包

当前 `runs/` 约 43 GB，其中科学计算用 ROOT 原始数据约 40 GB。把它们塞进程序
包会使交付物难以复制，也会混淆“程序”和“数据”。因此：

- 程序包只保存重新运行所需的程序、配置和说明；
- 轻量结果包保存表格、指标、图片、运行参数和日志；
- ROOT、NPZ 和 TIFF 在 `DATA_MAP.md` 与 `data_inventory.json` 中逐项标出，
  仍保留在原项目 `runs/` 下。

其中 TIFF 是由同一绘图程序导出的高分辨率副本，PNG/PDF/SVG 已放入轻量结果包；
NPZ 是由 ROOT 生成的中间数组，可用分析命令重新产生。

## 数据分级

- **L0 原始仿真数据**：`hits.root`、`hits_truth_*.root`、`gate.log`、
  `run_parameters.txt`、`macro_sha256.txt`；
- **L1 分析中间数据**：`events.npz`、`pairs.npz`、`reconstruction.npz`；
- **L2 定量结果**：CSV 和 JSON；
- **L3 图形结果**：PNG、PDF、SVG、TIFF；
- **L4 汇报材料**：`docs/`、`report_latex/`、阶段报告 PDF 和 PPT。

完整执行顺序见 [REPRODUCE_ALL.md](REPRODUCE_ALL.md)。

