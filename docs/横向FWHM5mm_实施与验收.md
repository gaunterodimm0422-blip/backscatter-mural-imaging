# 横向 FWHM 约 5 mm：实施与验收说明

## 1. 本阶段要证明什么

目标不是单纯找一组能让曲线变窄的参数，而是在更新后的 D1/D2 性能条件下，证明系统对标准细杆的横向响应宽度约为 5 mm 或更小，同时保留足够的有效事件。

本阶段采用的标准目标是：

- 直径 2 mm 的水泥杆；
- 杆沿 y 方向放置，因此 x 方向是横跨细杆的方向；
- 杆中心位于 30 mm 深处；
- 每组运行 5,000,000 个初级光子。

把重建点沿 x 方向投影，形成一条一维响应曲线。该曲线在峰值一半高度处的全宽就是横向 FWHM。

## 2. 已完成的参数优化

更新后的探测器响应为：

- D1：GAGG；位置读出尺寸为 1.55 mm × 1.55 mm × 5 mm；能量分辨率为 8% FWHM @ 511 keV；
- D2：GAGG；位置读出尺寸为 1 mm × 1 mm × 2.5 mm；能量分辨率为 12% FWHM @ 511 keV。

利用已完成的 GATE 原始事件，先在事件能量求和后加入上述能量展宽和位置读出响应，再进行筛选和重建。更新响应后的已有结果为：

| 标准杆直径 | 横向 FWHM | 95% bootstrap 区间 | 有效事件数 |
|---:|---:|---:|---:|
| 1 mm | 4.117 mm | 3.135–5.181 mm | 84 |
| 2 mm | 3.808 mm | 3.375–4.269 mm | 314 |
| 5 mm | 4.485 mm | 4.254–4.765 mm | 1,965 |

1 mm 杆的中心值低于 5 mm，但事件只有 84 个，区间上限超过 5 mm，所以不适合单独作为正式验收依据。2 mm 杆兼顾了目标宽度和统计量，因此选为正式验证目标。

## 3. 冻结的筛选方案

在不改变探测器结构的情况下，最终只对 D1 能窗作温和调整：

- D1 能窗：180–250 keV；
- D2 能窗：400–600 keV；
- 入射角：0–45°；
- 康普顿散射角：90–180°；
- D1 有效半径：20–200 mm。

在同一组 GATE 输运事件上改变 25 次探测器电子学随机展宽，D1 180–250 keV 方案得到：

- 横向 FWHM 中位数：3.573 mm；
- 2.5%–97.5% 分位区间：3.367–3.883 mm；
- 最大值：3.953 mm；
- 25/25 组均不超过 5 mm；
- 有效事件中位数：273；
- 保留原始基准事件的 86.0%；
- 最低真值信号比例：96.74%；
- 最大峰位置偏差：0.137 mm。

因此该方案不是通过只保留极少数事件来人为压窄曲线。

冻结配置位于：

`configs/transverse_fwhm_5mm_frozen.json`

## 4. 为什么还要运行一个独立 GATE 随机种子

前面的 25 组检查只改变了探测器能量展宽所用的随机数，GATE 中光子的输运历史仍来自同一个随机种子。正式结论还需要一组独立的 GATE 输运随机种子，以排除原始输运样本的偶然波动。

本轮独立验证固定为：

- GATE 随机种子：24080573；
- 初级光子数：5,000,000；
- 目标：2 mm 水泥杆，30 mm 深度；
- 运行位置：虚拟机本地 `/home/vgate/backscatter_runs`；
- 预计只需运行这一组，不再重复 1 mm 和 5 mm 杆。

## 5. 在虚拟机中启动

共享目录中的两个新脚本已经与 Windows 工作目录做过 SHA-256 一致性检查。先把它们复制到虚拟机本地项目目录：

```bash
cd /home/vgate/beisanshe_v1/gate
cp /media/sf_share/beisanshe_v1/gate/run_transverse_fwhm_validation.sh .
cp /media/sf_share/beisanshe_v1/gate/watch_transverse_fwhm_validation.sh .
chmod +x run_transverse_fwhm_validation.sh watch_transverse_fwhm_validation.sh
bash -n run_transverse_fwhm_validation.sh
bash -n watch_transverse_fwhm_validation.sh
```

以上两条 `bash -n` 没有输出即表示语法正常。

第一个终端启动正式运行：

```bash
cd /home/vgate/beisanshe_v1/gate
bash run_transverse_fwhm_validation.sh
```

第二个终端查看进度：

```bash
cd /home/vgate/beisanshe_v1/gate
bash watch_transverse_fwhm_validation.sh \
  rod_width2_depth30_5m_seed24080573_local_fwhm5_validation 30
```

监视器同时显示输出目录大小、进程运行时间、CPU 占用和 `/home` 剩余空间。目录大小只能反映大致进度；正式完成必须出现精确的 5,000,000 事件完成标志。

## 6. 完成后的校验和传输

在 Windows PowerShell 中执行。密码只输入到当前 PowerShell 环境变量，不要把密码写进脚本或聊天记录：

```powershell
cd C:\Users\HOUWENDI\Documents\beisanshe
$env:VGATE_PASSWORD = Read-Host "请输入虚拟机 vgate 用户密码"
powershell -ExecutionPolicy Bypass -File scripts\transfer_vgate_runs.ps1 `
  -RunIds rod_width2_depth30_5m_seed24080573_local_fwhm5_validation
Remove-Item Env:VGATE_PASSWORD
```

传输脚本会检查：

1. `hits.root`、truth ROOT、日志和参数文件均存在且非空；
2. 日志含精确的 5,000,000 事件完成标志；
3. 日志不含致命错误；
4. 虚拟机端与 Windows 端 ROOT 文件的 SHA-256 完全一致。

校验完成前不要删除虚拟机本地结果。

## 7. 自动处理与最终验收

传输成功后运行：

```powershell
cd C:\Users\HOUWENDI\Documents\beisanshe
powershell -ExecutionPolicy Bypass -File `
  scripts\process_transverse_fwhm_validation.ps1
```

程序会自动完成：

1. 从 ROOT 中聚合 D1/D2 事件并匹配 truth；
2. 应用 D1 8%、D2 12% 的能量展宽；
3. 应用 D1/D2 的位置读出尺寸；
4. 使用冻结的 D1 180–250 keV 条件进行可观测量筛选；
5. 闭式重建散射点；
6. 用 25 组探测器响应随机种子检查结果稳定性；
7. 对固定响应种子进行 300 次 bootstrap；
8. 输出逐种子 CSV、验证图、标准杆响应图、JSON 和中文验收报告。

正式通过必须同时满足：

- 运行身份确为随机种子 24080573、5,000,000 个初级光子；
- 25/25 组探测器响应随机种子的横向 FWHM 都不超过 5 mm；
- 固定响应种子的 bootstrap 95% 区间上限不超过 5 mm；
- 有效事件数中位数不少于 150；
- 最低真值信号比例不少于 95%；
- 最大峰位置偏差不超过 1 mm。

最终输出目录为：

`outputs/transverse_fwhm_5mm_20260805/independent_gate_seed/rod_width2_depth30_5m_seed24080573_local_fwhm5_validation`

## 8. 当前状态

截至 2026-08-05，本阶段已完成并通过正式验收：

- 更新探测器响应后的基准计算已完成；
- 参数扫描和 25 组电子学随机展宽稳定性检查已完成；
- D1 180–250 keV 方案已冻结；
- 虚拟机启动脚本、监视脚本、传输校验和自动验收程序均已完成；
- 已用旧 GATE 数据完成整条链路的非正式冒烟测试，程序运行正常；
- 独立 GATE 随机种子 24080573 的 5,000,000 事件运行已完整结束；
- 虚拟机原始 ROOT、共享副本和项目副本的 SHA-256 已核对一致；
- 独立种子下 25/25 组探测器响应随机展宽的横向 FWHM 均不超过 5 mm；
- 横向 FWHM 中位数为 3.672 mm，最大值为 4.297 mm；
- 固定响应种子的横向 FWHM 为 3.672 mm，bootstrap 95% 区间为 3.109–4.418 mm；
- 有效事件数中位数为 288，最低真值信号比例为 97.18%，最大峰位置偏差为 0.147 mm；
- 全部正式验收条件均通过。

最终中文验收报告位于：

`outputs/transverse_fwhm_5mm_20260805/independent_gate_seed/rod_width2_depth30_5m_seed24080573_local_fwhm5_validation/独立GATE种子_横向FWHM5mm验收.md`
