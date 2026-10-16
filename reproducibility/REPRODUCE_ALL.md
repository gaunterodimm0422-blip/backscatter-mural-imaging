# 从零复现当前背散射项目

## 1. 固定环境

GATE 仿真已经验证的环境为：

- vGate 9.4.1；
- GATE 9.4.1；
- Geant4 11.3.0；
- Python 3.10 或更高版本。

Windows 分析环境建议在项目根目录建立独立虚拟环境：

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

正式快照打包前已在 Python 3.12 环境执行测试，52 项全部通过。

## 2. GATE 端运行原则

将整个 `gate/` 复制到虚拟机本地磁盘，例如：

```bash
cp -a /media/sf_share/beisanshe_v1/gate /home/vgate/beisanshe_v1/
cd /home/vgate/beisanshe_v1/gate
chmod +x *.sh
bash -n run_baseline.sh
bash -n run_standard.sh
bash -n run_local_export.sh
bash -n run_campaign_batch.sh
```

正式数据写到 `/home/vgate/backscatter_runs`，不要让多个 GATE 进程持续向
VirtualBox 共享文件夹写 ROOT。每次运行必须保留：

- `hits.root`；
- `hits_truth_concrete.root` 或 `hits_truth_target.root`；
- `gate.log`；
- `run_parameters.txt`；
- `macro_sha256.txt`；
- `gate_version.txt`。

只有 `gate.log` 中出现准确的 `Number of events processed`，且没有
`ERROR/FATAL/Aborted/core dumped/G4Exception`，才能标为完成。

## 3. 当前正式仿真的重新运行命令

以下命令中的运行名与当前数据目录一致。若原目录仍存在，不应覆盖；重新运行时
应更换 `RUN_TAG`，然后在比较时明确新旧随机种子和运行名。

### 3.1 完整水泥基线

```bash
N_PRIMARIES=2000000 RANDOM_SEED=24061873 SKIP_EXPORT=1 \
  bash run_local_export.sh baseline concrete_slab_2m_seed24061873_local
```

### 3.2 10--90 mm 轴向薄层

```bash
for depth in 10 30 50 70 90; do
  run_id="depth_psf_${depth}mm_seed24061873_local"
  N_PRIMARIES=2000000 RANDOM_SEED=24061873 \
  TARGET_DEPTH_MM="$depth" TARGET_THICKNESS_MM=1 \
  TARGET_LATERAL_MM=300 SKIP_EXPORT=1 \
    bash run_local_export.sh thin_layer "$run_id"
done
```

### 3.3 独立随机种子复现与 D1 物理缩径

```bash
N_PRIMARIES=2000000 RANDOM_SEED=24061874 SKIP_EXPORT=1 \
  bash run_local_export.sh baseline repeat_concrete_seed24061874_local

N_PRIMARIES=2000000 RANDOM_SEED=24061874 TARGET_DEPTH_MM=30 \
TARGET_THICKNESS_MM=1 TARGET_LATERAL_MM=300 SKIP_EXPORT=1 \
  bash run_local_export.sh thin_layer repeat_thin30_seed24061874_local

N_PRIMARIES=2000000 RANDOM_SEED=24061874 TARGET_DEPTH_MM=90 \
TARGET_THICKNESS_MM=1 TARGET_LATERAL_MM=300 SKIP_EXPORT=1 \
  bash run_local_export.sh thin_layer repeat_thin90_seed24061874_local

N_PRIMARIES=2000000 RANDOM_SEED=24061873 D1_RMAX_MM=150 \
TARGET_DEPTH_MM=30 TARGET_THICKNESS_MM=1 TARGET_LATERAL_MM=300 SKIP_EXPORT=1 \
  bash run_local_export.sh thin_layer physical_d1r150_thin30_seed24061873_local

N_PRIMARIES=2000000 RANDOM_SEED=24061873 D1_RMAX_MM=175 \
TARGET_DEPTH_MM=30 TARGET_THICKNESS_MM=1 TARGET_LATERAL_MM=300 SKIP_EXPORT=1 \
  bash run_local_export.sh thin_layer physical_d1r175_thin30_seed24061873_local

N_PRIMARIES=2000000 RANDOM_SEED=24061873 D1_RMAX_MM=150 SKIP_EXPORT=1 \
  bash run_local_export.sh baseline physical_d1r150_concrete_seed24061873_local
```

### 3.4 1/2/5 mm 单杆

```bash
RUN_TAG=manual_retry2 bash run_campaign_batch.sh rods
```

这里的 1/2/5 mm 是实体水泥杆的直径，目标位于真空中，用于测量本征横向响应。

### 3.5 10/20 mm 水泥孔洞，30/60 mm 深度

```bash
RUN_TAG=manual_void1 bash run_campaign_batch.sh void_pilot
```

这里是完整水泥板中的圆柱形 Vacuum 区，与单杆体模不是同一种对象。

### 3.6 30/70 mm 边缘体模

```bash
RUN_TAG=manual_edge_20260719_1 bash run_campaign_batch.sh edge
```

### 3.7 5/10/15 mm 双杆中心距

```bash
RUN_TAG=manual_double_a_20260719_1 bash run_campaign_batch.sh double_rods_a
```

20/30 mm 中心距没有运行，因为 10/15 mm 已经稳定形成双峰，按照预先规定的
自适应停止规则无需继续消耗计算量。对应配置仍保留在 `configs/generated/`。

### 3.8 12 例孔洞盲检流程验收

```bash
bash run_blind_void_qualification.sh blindq1
```

公开数据只能包含 `case001`--`case012`、`hits.root` 和公开承诺信息。必须先在
Windows 端冻结评分和 SHA-256，之后才能运行 `reveal_blind_void_qualification.sh`。

### 3.9 尚未产生正式数据的配置

- `water_thin_layer_depth30_2m.json`：水薄层；
- `iron_thin_layer_depth30_2m.json`：铁薄层；
- 双杆中心距 20/30 mm：按自适应停止规则未运行。

因此不能把水/铁写成已经完成的正式跨材料结果。

## 4. 从虚拟机回传原始数据

在 Windows 项目根目录设置虚拟机密码，仅在当前进程中使用：

```powershell
$env:VGATE_PASSWORD = "你的虚拟机密码"
```

普通运行通过以下脚本回传并校验两端 SHA-256：

```powershell
powershell -ExecutionPolicy Bypass -File scripts\transfer_vgate_runs.ps1 `
  -RunIds "运行目录名1","运行目录名2"
```

盲检公开包使用：

```powershell
powershell -ExecutionPolicy Bypass -File scripts\transfer_blind_public_campaign.ps1 `
  -CampaignTag blindq1
```

不要把密码写进脚本、文档或压缩包。

## 5. 从 ROOT 重新生成分析结果

当前正式 ROOT 位于：

```text
runs/vgate_fast_transfer/output/<run_id>/
```

### 5.1 基线和轴向薄层

```powershell
powershell -ExecutionPolicy Bypass -File scripts\process_concrete_depth.ps1 `
  -SharedGateRoot "runs\vgate_fast_transfer"

powershell -ExecutionPolicy Bypass -File scripts\process_thin_layer_campaign.ps1 `
  -SharedGateRoot "runs\vgate_fast_transfer"
```

### 5.2 独立种子和物理缩径验证

```powershell
powershell -ExecutionPolicy Bypass -File scripts\process_stage2_validation.ps1 `
  -RawRoot "runs\vgate_fast_transfer\output"
```

### 5.3 单杆、双杆和边缘

```powershell
powershell -ExecutionPolicy Bypass -File scripts\process_spatial_campaign.ps1 `
  -RawRoot "runs\vgate_fast_transfer\output" -Campaign rods `
  -RodRunSuffix "_manual_retry2"

powershell -ExecutionPolicy Bypass -File scripts\process_spatial_campaign.ps1 `
  -RawRoot "runs\vgate_fast_transfer\output" -Campaign double_rods_a `
  -DoubleRodRunSuffix "_manual_double_a_20260719_1"

powershell -ExecutionPolicy Bypass -File scripts\process_spatial_campaign.ps1 `
  -RawRoot "runs\vgate_fast_transfer\output" -Campaign edge `
  -EdgeRunSuffix "_manual_edge_20260719_1"
```

### 5.4 水泥孔洞

```powershell
powershell -ExecutionPolicy Bypass -File scripts\process_void_material_campaign.ps1 `
  -RawRoot "runs\vgate_fast_transfer\output" -Campaign voids `
  -VoidRunSuffix "_manual_void1"
```

不要使用 `-Campaign all`，因为水和铁正式数据尚未运行。

### 5.5 盲检评分

```powershell
powershell -ExecutionPolicy Bypass -File scripts\process_blind_void_qualification.ps1 `
  -CampaignTag blindq1
```

评分脚本会拒绝覆盖已有分析目录。要完整复算，应先复制项目到新的工作目录，
或使用新的 campaign tag，不能直接删掉当前冻结评分。

## 6. 可复现性的边界

- 相同随机种子、GATE/Geant4 版本、宏和材料库应产生一致统计过程；不同平台的
  ROOT 二进制文件不要求逐字节相同，应比较预注册指标和统计区间；
- `macro_sha256.txt` 记录每次仿真实际使用的 GATE 宏；
- `run_parameters.txt` 是每个运行目录的参数事实来源；
- `transfer_manifest.json` 或公开盲检 manifest 保存回传时的 ROOT SHA-256；
- 真值分支只能用于仿真评分，不能作为将来真实实验的事件筛选条件。

