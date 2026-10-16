# GATE 9.4 仿真运行说明

本目录是项目唯一的 GATE 运行包。不要逐条手工执行子宏；统一通过 shell
入口运行，这样初级事件数、随机种子、几何参数、GATE 版本、宏哈希和完整
日志都会随结果保存。

## 1. 最小水泥测试

在 vGate 中进入本目录后执行：

```bash
N_PRIMARIES=20000 bash run_baseline.sh smoke_concrete
```

输出位于 `output/smoke_concrete/`，至少应包含：

- `hits.root`：D1/D2 Hits；
- `hits_truth_concrete.root`：水泥中的康普顿顶点真值；
- `gate.log`：完整终端日志；
- `run_parameters.txt`：本次参数；
- `macro_sha256.txt` 和 `gate_version.txt`：版本审计。

## 2. 标准分辨率体模

统一入口：

```bash
N_PRIMARIES=2000000 TARGET_DEPTH_MM=30 TARGET_THICKNESS_MM=1 \
  bash run_standard.sh thin_layer depth_psf_30mm
```

支持：

- `thin_layer`：1 mm 孤立薄层，测轴向响应；
- `rod`：沿 y 的单细杆，测横向线扩散响应；
- `double_rod`：双细杆，测双目标可分辨性；
- `edge`：半平面边缘，测 MTF；
- `point` / `double_point`：统计量足够时测三维点响应。

杆或点的半径由 `TARGET_RADIUS_MM` 控制，双目标中心距由
`TARGET_SEPARATION_MM` 控制，材料可设为 `Concrete`、`Water` 或 `Iron`。
这些孤立目标位于 Vacuum 中，表示本征几何响应，不表示水泥覆盖下的穿透
能力。

项目所说的“1 mm 杆”是直径 1 mm、半径 0.5 mm、沿 y 方向长 300 mm 的
孤立水泥圆柱测试体。它不是探测器，也不是射线束；它相当于一条已知很细的
线，用重建后的 x 分布宽度测量横向线扩散响应。

## 3. 水泥孔洞体模

`phantom_concrete_void.mac` 在 100 mm 完整水泥板中嵌入一条沿 y 的圆柱形
Vacuum 孔洞。孔洞是水泥的子体积，因此不存在用两个同级体积互相覆盖的问题。

示例：直径 10 mm、中心深度 30 mm、长度 300 mm：

```bash
N_PRIMARIES=2000000 RANDOM_SEED=24061875 \
VOID_RADIUS_MM=5 VOID_LENGTH_MM=300 VOID_X_MM=0 VOID_DEPTH_MM=30 \
  bash run_concrete_void.sh concrete_void_d10_depth30_seed24061875
```

`run_baseline.sh` 自动计算水泥母体坐标中的
`VOID_LOCAL_Z_MM = 50 - VOID_DEPTH_MM`，并同时记录全局深度和局部坐标。

## 4. 正式大样本应写入虚拟机本地盘

VirtualBox 共享目录不适合多个 GATE 进程持续并发写大型 ROOT。正式运行使用：

```bash
N_PRIMARIES=2000000 SKIP_EXPORT=1 \
  bash run_local_export.sh baseline concrete_seed24061874_local
```

标准体模把 `baseline` 换成 `thin_layer`、`rod`、`double_rod`、`edge` 等；
孔洞使用 `concrete_void`。默认本地根目录为
`/home/vgate/backscatter_runs`。

`SKIP_EXPORT=1` 表示仿真结束后先保留在虚拟机本地盘，由主机完成哈希校验后
再复制并清理，避免同时保留两份大文件把虚拟机磁盘写满。不设置该变量时，
脚本会使用锁和临时目录顺序导出到共享目录。

## 5. 人工托管正式单杆批次

正式运行包已复制到虚拟机本地目录
`/home/vgate/beisanshe_v1/gate`。在 vGate 的第一个终端中运行以下命令，
并保持该终端和虚拟机开启：

```bash
cd /home/vgate/beisanshe_v1/gate
RUN_TAG=manual_retry1 bash run_campaign_batch.sh rods
```

`RUN_TAG` 会加入三个任务名，避免碰到以前中断后留下的同名本地目录。一次
重试只能使用一次 tag；如果这一批再次中断，应改成 `manual_retry2`，不要把
不完整 ROOT 当作可续跑文件。

在第二个 vGate 终端中查看三个任务的进程数、本地 ROOT 目录大小和完成数：

```bash
cd /home/vgate/beisanshe_v1/gate
bash watch_rods.sh manual_retry1 30
```

目录大小只是进度代理，不等于有效事件数。只有第一个终端显示三个
`Completed`、观察脚本显示 `Completed: 3/3`，并且后续检查到 500 万事件
完成标记和 SHA-256 一致，才算正式完成。

## 6. 人工托管孔洞、双杆和边缘批次

建议按“孔洞 pilot → 边缘 → 双杆”的顺序逐批运行；每一批完成、回传并清理
虚拟机本地副本后再启动下一批。孔洞 pilot 的正式默认值为每组 200 万事件：

```bash
cd /home/vgate/beisanshe_v1/gate
RUN_TAG=manual_void1 bash run_campaign_batch.sh void_pilot
```

第二个终端监视：

```bash
cd /home/vgate/beisanshe_v1/gate
bash watch_campaign.sh void_pilot manual_void1 30
```

边缘体模使用 `edge` 批次，每组默认 200 万事件；双杆分成 5/10/15 mm 和
20/30 mm 两批，每组默认 500 万事件。先分析第一批；只有 15 mm 仍不能
稳健分辨时，才按自适应规则运行第二批：

```bash
RUN_TAG=manual_edge1 bash run_campaign_batch.sh edge
RUN_TAG=manual_double_a1 bash run_campaign_batch.sh double_rods_a
# 仅在第一批未夹住分辨边界时运行：
RUN_TAG=manual_double_b1 bash run_campaign_batch.sh double_rods_b
```

短样本诊断时可分别设置 `VOID_PRIMARIES`、`EDGE_PRIMARIES` 或
`DOUBLE_ROD_PRIMARIES`；正式任务不要覆盖默认值。

## 7. 文件职责

- `main_baseline.mac`：初始化顺序、511 keV back-to-back 源和采集；
- `geometry_detectors.mac`：D1 环与 D2 50×50 阵列；
- `phantom_concrete.mac`：均匀 100 mm 水泥；
- `phantom_concrete_void.mac`：带圆柱 Vacuum 孔洞的完整水泥；
- `phantom_*.mac`：薄层、单杆、双杆、点、双点和边缘体模；
- `physics.mac`：Livermore 电磁物理；
- `materials/BackscatterMaterials.db`：自包含材料库；
- `output_hits.mac`：D1/D2 ROOT Hits；
- `truth_concrete.mac` / `truth_target.mac`：反冲电子顶点真值。

`phantomSD` 写入 `nPhantomCompton`、`nPhantomRayleigh` 和
`comptVolName`；PhaseSpaceActor 记录精确顶点。两者都是仿真真值，只用于
事后评分，不得冒充真实实验可用的筛选量。

## 8. 20 mm 孔洞盲检流程验收批次

这一步不是把文件名写成 `void/no_void` 后再分析，而是由虚拟机在运行时随机
生成 6 个孔洞样本和 6 个无孔洞样本。公开文件只叫 `case001–case012`；
答案和位置只保存在虚拟机私有目录，评分锁定前不得导出。

先把三个小脚本从共享目录复制到已经存在的虚拟机本地运行包，并做语法检查：

```bash
cd /home/vgate/beisanshe_v1/gate
cp /media/sf_share/beisanshe_v1/gate/run_blind_void_qualification.sh .
cp /media/sf_share/beisanshe_v1/gate/watch_blind_void_qualification.sh .
cp /media/sf_share/beisanshe_v1/gate/reveal_blind_void_qualification.sh .
chmod +x *_blind_void_qualification.sh
bash -n run_blind_void_qualification.sh
bash -n watch_blind_void_qualification.sh
bash -n reveal_blind_void_qualification.sh
```

确认三个 `bash -n` 都没有输出后，在第一个终端启动：

```bash
cd /home/vgate/beisanshe_v1/gate
bash run_blind_void_qualification.sh blindq1
```

默认共 12 例、每例 100 万事件、最多同时运行 3 个 GATE。按已有 ROOT 大小
估算，公开主 ROOT 最终约占 6–8 GB，运行时还会有私有 truth 和最多三个
临时主 ROOT，建议启动前 `/home/vgate` 至少有约 12 GB 可用空间。

第二个终端只查看不泄露答案的进度：

```bash
cd /home/vgate/beisanshe_v1/gate
bash watch_blind_void_qualification.sh blindq1
```

完成后不要查看
`/home/vgate/backscatter_blind_private/blindq1`，也不要运行 reveal 脚本。
先让主机只回传 `/home/vgate/backscatter_blind_public/blindq1` 并锁定 12 个
评分；主机给出 `SCORE_BUNDLE_SHA256` 后才允许揭盲。

## 9. 完成验收

正式结果必须同时满足：

1. `gate.log` 含准确的 `Number of events processed`；
2. 主 ROOT 和对应 truth ROOT 非空；
3. 日志不含 `ERROR`、`FATAL`、`Aborted`、`core dumped` 或 G4Exception；
4. 从虚拟机复制到主机后，两端 `hits.root` 的 SHA-256 一致；
5. `run_parameters.txt` 与所用 JSON 配置一致。

本项目已在 vGate-9.4.1、GATE 9.4.1、Geant4 11.3.0 上验证。材料库由项目
自带，不依赖 `/opt/gate/GateMaterials.db` 等镜像内部绝对路径。
