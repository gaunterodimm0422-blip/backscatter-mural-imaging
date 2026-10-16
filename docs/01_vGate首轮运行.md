# vGate 首轮运行

首轮只跑 2 万个初级事件，用来确认宏命令、ROOT 树和散射真值。它不是正式统计结果。

## 1. 已准备好的共享运行包

为了复用现有 vGate 共享盘，运行包已经放在 Windows：

`E:\vbgate\share\beisanshe_v1\gate`

如果 `E:\vbgate\share` 在虚拟机中的共享名是 `share`，常见路径为：

```bash
/media/sf_share/beisanshe_v1/gate
```

如果不是这个路径，可执行：

```bash
find /media -maxdepth 4 -type d -name beisanshe_v1 2>/dev/null
```

找到后进入其 `gate` 子目录。

## 2. 检查 GATE

```bash
Gate --version
cd /media/sf_share/beisanshe_v1/gate
```

本机 `vGate-9.4.1` 已实测为 GATE 9.4.1、Geant4 11.3.0。宏使用项目内自包含
材料库，不依赖虚拟机中的 `GateMaterials.db` 绝对路径。

## 3. 冒烟测试

```bash
N_PRIMARIES=20000 bash run_baseline.sh smoke_concrete
```

脚本使用固定随机种子并把屏幕输出保存到日志。正常完成后应出现：

```text
gate/output/smoke_concrete/hits.root
gate/output/smoke_concrete/hits_truth_concrete.root
gate/output/smoke_concrete/gate.log
gate/output/smoke_concrete/gate_version.txt
gate/output/smoke_concrete/macro_sha256.txt
gate/output/smoke_concrete/run_parameters.txt
```

请不要只截图终端。把整个 `gate/output/smoke_concrete` 文件夹保留在共享目录中。

## 4. 在 Windows 主机分析

回到 PowerShell：

```powershell
cd "C:\Users\HOUWENDI\Documents\beisanshe"

python -m backscatter inspect-root `
  --root "E:\vbgate\share\beisanshe_v1\gate\output\smoke_concrete\hits.root"

python -m backscatter analyze `
  --root "E:\vbgate\share\beisanshe_v1\gate\output\smoke_concrete\hits.root" `
  --truth "E:\vbgate\share\beisanshe_v1\gate\output\smoke_concrete\hits_truth_concrete.root" `
  --config "configs\baseline.json" `
  --out-dir "runs\smoke_concrete"
```

## 5. 首轮验收

`runs/smoke_concrete/summary.json` 中应满足：

- 初级事件数约为 20,000；
- 两个 Hits 树均非空；
- D2 事件能谱在 511 keV 附近有峰；
- `phantom_truth.available=true`；
- D1 中至少有一部分事件 `nPhantomCompton>=1`；
- 真值文件中可提取 Compton 电子顶点；
- 运动学自检和散射点求解没有 NaN 扩散或把非法事件裁剪到 180°。

如果 GATE 在某一条宏命令处停止，请把 `gate.log` 原样交给我；不要手动删命令后继续，
否则会失去版本和真值的一致性。

## 6. 正式基线

百万级任务不要让 GATE 持续写 VirtualBox 共享盘。冒烟测试通过后，统一先写
虚拟机本地盘，ROOT 正常关闭后再导出：

```bash
N_PRIMARIES=1000000 \
  bash run_local_export.sh baseline baseline_concrete_seed24061873
```

第二个独立随机种子由脚本环境变量指定：

```bash
N_PRIMARIES=1000000 RANDOM_SEED=24061874 \
  bash run_local_export.sh baseline baseline_concrete_seed24061874
```

两个正式基线用于检查结果的统计重复性。只有控制终端同时出现 `Completed:` 和
`Exported:` 后，主机端的最终目录才可进入分析。
