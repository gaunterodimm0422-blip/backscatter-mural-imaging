# 当前壁画GATE运行包：冻结依赖备份

本目录于2026-10-16从本机`E:\vbgate\share\beisanshe_v1\gate`复制，用于让代码仓库同时保存当前壁画空气夹层方案所需的GATE运行文件。原有`gate/`保留早期水泥/标准体模，本目录没有覆盖它。

这是运行程序的备份，不是新的仿真任务。本次仅复制文件、解析依赖并核对SHA-256，**没有启动GATE、运行分析或连接远控机**。复制的`.sh`、`.mac`和`.db`保持源文件原始字节；注释中的历史预测、旧构型比较不代表当前正式性能，结果以`../m3/outputs/d2f15/`的JSON及`../项目文档/00_交接入口_当前状态与文件索引_20261016.md`为准。

## 文件与实际依赖

运行入口是`run_mural_edge_v3.sh`（轴心/参照/负对照及离轴批次）和`run_mural_thick_v1.sh`（界面空气层厚度扫描）。两个脚本均设置：

```text
MAIN_MAC=main_baseline_pose_v2.mac
PHANTOM_MAC=phantom_mural.mac
TRUTH_MAC=truth_mural.mac
```

主宏的全部文件依赖均已包含：

```text
main_baseline_pose_v2.mac
  verbosity.mac
  materials/BackscatterMaterials.db
  geometry_detectors_pose_v2.mac
  phantom_mural.mac
  physics.mac
  output_hits.mac
  truth_mural.mac
```

上述子宏没有进一步的`/control/execute`引用。材料库自包含所需元素与材料，不依赖源目录中未使用的`.bak`文件。两个入口脚本不再调用其它本项目shell脚本，计算用的Python片段内嵌于脚本中。总计10个原始运行文件，另加本README。这里只保存实际依赖闭包，不包含E盘目录中的其它历史体模、ZIP和输出数据；新运行生成的`macro_sha256.txt`清单因此可能比历史大全目录的清单短。

## 预期运行环境与输出位置

历史正式运行环境记录为Linux vGate/GATE 9.4.2、Geant4 11.4.0，需已配置好的`Gate`命令、`bash`、`python3`及常用GNU命令行工具。它们是外部运行环境，本目录没有安装或附带GATE二进制、Geant4数据集或虚拟机。

远控机历史布局为：

- 虚拟机运行目录：`/home/vgate/beisanshe/gate_bundle_20260910/gate`。
- 虚拟机共享盘：`/media/sf_vgate_shared`，对应远控Windows的`D:\vgate_shared`。
- 原始结果：`/media/sf_vgate_shared/output/<RUN_ID>/`，对应`D:\vgate_shared\output\<RUN_ID>\`。
- Windows分析目录：`D:\beisanshe\runs\<RUN_ID>\response_d2f15`。

这些是既有记录，**不表示已实时核验远控机在线或上述数据仍完整**。`hits.root`、`hits_truth_mural.root`及事件级响应缓存按记录留在远控机，没有复制到本目录或随本次代码上传。仓库中的约5 KB绘图NPZ是聚合直方图，不是原始事件库。

两个脚本都会切换到自身所在目录后运行。实际安排运行时，应**显式设置`SHARE_OUT=/media/sf_vgate_shared/output`**，不要依赖自动探测或主力机的共享盘布局。仅需查看参数时可在Linux运行包目录使用下列干运行形式，示例本次并未执行：

```bash
DRY_RUN=1 SHARE_OUT=/media/sf_vgate_shared/output bash run_mural_edge_v3.sh
DRY_RUN=1 SHARE_OUT=/media/sf_vgate_shared/output bash run_mural_thick_v1.sh
```

正式长模拟需另获用户安排。原脚本会先写本地`output/<RUN_ID>/`，完成后导出共享盘，默认删除成功导出的本地副本；`KEEP_LOCAL=1`可保留。若共享盘已有同名运行目录，脚本直接跳过，不在跳过前重新验证ROOT完整性，所以不能把“目录存在”当作验收通过。

本包只记录理想输运沉积；D1 8%、D2 15%能量响应及位置量化在后续Python分析中施加。复分析应使用`../configs/*_d2f15_20260923.json`，按现行说明生成`response_d2f15`，不要把脚本中历史注释或旧12%缓存当成正式结果。

## 四个关键冻结指纹

复制后已确认这四个副本与E盘原件及02操作手册的冻结记录一致；其它六个原始运行文件也逐一通过源/副本SHA-256一致性检查。

| 文件 | SHA-256 |
|---|---|
| `geometry_detectors_pose_v2.mac` | `105d94710e272670f532b6ab459c9ae7fc4d4b113234f104ff2bd90c22b3fa9f` |
| `main_baseline_pose_v2.mac` | `54264c5b9995c6b69991cf92e67bb4f702ab7e22a6a04143814c7ca7b3c6d531` |
| `run_mural_edge_v3.sh` | `abd0a4c1c3d43c013f75da09048429040d0ffd26aa64bc079ae0e6b9affacb51` |
| `run_mural_thick_v1.sh` | `0e743bca0f553fd8eaba1dd951110394795e2de47810a6ab6e12cf9702bd7a46` |
