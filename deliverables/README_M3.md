# M3：新探测头构型下重跑边界定位 —— 运行说明与冻结判据

日期 2026-09-12 · 上一轮见 `README_M2.md`

---

## 一、这一轮改了什么

只动探测头几何，**体模、源位置、能量窗、符合窗、判选链条、重建算法、估计量全部不动**。

| 项 | 旧（M1/M2 全部运行） | 新（本轮） | 谁定的 |
|---|---|---|---|
| 发射锥半角 `maxtheta` | 45° | **35°** | 用户指定 |
| D1 厚度 | 15 mm | **10 mm** | 用户指定 |
| D1 内半径 | 20 mm | **12.5 mm** | 用户指定 |
| D2 下表面 z | 50 mm | **42.5 mm** | 用户指定 |
| D1 外半径 | 200 mm | 100 mm | **我推断的，非用户指定** |
| D1 中面 z | 17.5 mm | 7.5 mm（下表面离物体 2.5 mm） | **我推断的，非用户指定** |
| 源位置 | z = 17.5 mm | z = 17.5 mm | 不变 |
| D2 厚度 / 有效面 | 20 mm / 50×50 mm | 20 mm / 50×50 mm | 不变 |

后两行是我为了让前四个数自洽而补的。**要改就改 `run_mural_edge_v3.sh` 里的 `D1_RMAX_MM` 和 `D1_CENTER_Z_MM` 两行**，脚本会自动重算平移量并重跑几何自检。

### 四个数为什么是自洽的一组

```
D1 下表面 z = 2.5 mm，源在 z = 17.5 mm
35° 锥在 D1 下表面处的半径 = 15.0 × tan35° = 10.50 mm
D1 内半径实取 12.5 mm  →  余量 +2.00 mm        源不直接照射 D1 内壁
源到 D2 中面 = 52.5 − 17.5 = 35.0 mm，D2 半宽 25 mm
D2 边缘接受半角 = atan(25/35) = 35.5°  ≈  发射锥 35°   锥恰好填满 D2 内接圆
深 30 mm 处视场半宽 = 47.5 × tan35° = 33.3 mm（旧构型 27.9 mm，被 D2 边缘限制）
```

**内半径 12.5 mm 这个数只有在"锥收到 35°"且"D1 压到离物体 2.5 mm"时才成立，三者绑定。**
单独把内半径改小、锥不收，源会直照 D1 内壁。

---

## 二、文件清单与校验和

| 文件 | sha256 | 说明 |
|---|---|---|
| `gate/geometry_detectors_pose_v2.mac` | `105d94710e272670f532b6ab459c9ae7fc4d4b113234f104ff2bd90c22b3fa9f` | 新几何，参数化 D1 内外半径/厚度/平移、D2 平移 |
| `gate/main_baseline_pose_v2.mac` | `54264c5b9995c6b69991cf92e67bb4f702ab7e22a6a04143814c7ca7b3c6d531` | 调用上面那个几何宏 |
| `gate/run_mural_edge_v3.sh` | `abd0a4c1c3d43c013f75da09048429040d0ffd26aa64bc079ae0e6b9affacb51` | 跑批脚本，带几何自检 |
| `scripts/analyze_mural_edge.py` | 与 M2 逐字相同 | 分析与判据 |
| `configs/m3_edge_main_20260912.json` | — | 边界 −15 mm，判据 1~6 |
| `configs/m3_edge_offset_20260912.json` | — | 边界 −23 mm，判据 9 |
| `configs/m3_edge_offset2_20260912.json` | — | 边界 −31 mm，判据 9 |
| `configs/m3_edge_main_widefov_20260912.json` | — | **只量视场宽度，不参与任何 T 判据**，见第五节判据 8 |

**原来的 `geometry_detectors_pose.mac` / `main_baseline_pose.mac` / `run_mural_edge.sh` 一个字都没动**，
M1/M2 的结果仍可原样复现。

---

## 三、怎么跑（远控机里的 vGate 虚拟机）

### 第 0 步：把压缩包放到共享盘

主力机上的 `m3_bundle_20260912.zip` 拷到远控机的 `D:\vgate_shared\`。

### 第 1 步：解包与校验（虚拟机终端，逐行）

```
cd /home/vgate/beisanshe/gate_bundle_20260910/gate
unzip -o -j /media/sf_vgate_shared/m3_bundle_20260912.zip 'gate/*'
sha256sum geometry_detectors_pose_v2.mac main_baseline_pose_v2.mac run_mural_edge_v3.sh
```

三个值必须分别等于：

```
105d94710e272670f532b6ab459c9ae7fc4d4b113234f104ff2bd90c22b3fa9f  geometry_detectors_pose_v2.mac
54264c5b9995c6b69991cf92e67bb4f702ab7e22a6a04143814c7ca7b3c6d531  main_baseline_pose_v2.mac
abd0a4c1c3d43c013f75da09048429040d0ffd26aa64bc079ae0e6b9affacb51  run_mural_edge_v3.sh
```

对不上就停下，不要往下跑。

### 第 2 步：语法检查与几何自检（不跑 GATE，秒级）

```
bash -n run_mural_edge_v3.sh
STAGE=a DRY_RUN=1 SHARE_OUT=/media/sf_vgate_shared/output bash run_mural_edge_v3.sh
```

应当看到：

```
几何自检通过
  空鼓 z ∈ [-40, -20] mm，粗泥层 z ∈ [-40, -5] mm
  D1 占 z ∈ [2.5, 12.5] mm，下表面离物体 2.5 mm，内半径下限 10.50 mm，实取 12.5 mm，余量 +2.00 mm
  D2 占 z ∈ [42.5, 62.5] mm，中面 52.5 mm，源到中面 35.0 mm，边缘接受半角 35.5°，发射锥 35°
  深 30 mm 处视场半宽 33.3 mm
```

### 第 3 步：STAGE=a（**三档** × 400 万事例）

```
cd /home/vgate/beisanshe/gate_bundle_20260910/gate
nohup env STAGE=a SHARE_OUT=/media/sf_vgate_shared/output bash run_mural_edge_v3.sh > edge_v3a.out 2>&1 &
echo "PID: $!"
```

监控：

```
tail -f edge_v3a.out
grep -c "^" edge_v3a.out                      # 行数在涨就说明活着
df -h /                                        # 根分区可用要 > 3 G
ls -la /media/sf_vgate_shared/output/ | tail   # 结果有没有搬到共享盘
```

**预计时长**：M2 的**两档** 400 万事例实测 76 分钟（我当时预测 45 分钟，低估了 69%，这次不给乐观数）。
本轮 STAGE=a 是**三档**。新构型 D1 体积小了 6 倍、锥也收窄，单事例应当更快，
但我没有实测依据，**就按 110~150 分钟准备**（= 76 × 1.5 再留余量）。
STAGE=b 两档，按 75~100 分钟准备。

### 第 4 步：a 判完再决定要不要跑 b

**a 不通过就不要跑 b。** 通过了再：

```
nohup env STAGE=b SHARE_OUT=/media/sf_vgate_shared/output bash run_mural_edge_v3.sh > edge_v3b.out 2>&1 &
echo "PID: $!"
```

### 第 5 步：分析（远控机 Windows 侧，不在虚拟机里）

三档跑完后，`D:\beisanshe\runs\` 下应当出现（目录名由脚本按 `m3_<档>_t20_L-15_4m_seed<种子>` 生成）：

| 角色 | 目录名 |
|---|---|
| halfvoid（有空鼓） | `m3_halfvoid_t20_L-15_4m_seed24091301` |
| ref_main（底片） | `m3_ref_t20_L-15_4m_seed24091302` |
| ref_null（零假设底片） | `m3_refnull_t20_L-15_4m_seed24091303` |

判据 1~7 与 9 那一跑：

```
C:\Users\Administrator\AppData\Local\Programs\Python\Python313\python.exe D:\beisanshe\scripts\analyze_mural_edge.py --halfvoid "D:\beisanshe\runs\m3_halfvoid_t20_L-15_4m_seed24091301\canonical_response" --ref-main "D:\beisanshe\runs\m3_ref_t20_L-15_4m_seed24091302\canonical_response" --ref-null "D:\beisanshe\runs\m3_refnull_t20_L-15_4m_seed24091303\canonical_response" --config "D:\beisanshe\configs\m3_edge_main_20260912.json" --out "D:\beisanshe\outputs\m3_edge_result" --null-trials 200
```

判据 8（视场宽度）那一跑，**只换 `--config` 和 `--out`，其余逐字相同**：

```
C:\Users\Administrator\AppData\Local\Programs\Python\Python313\python.exe D:\beisanshe\scripts\analyze_mural_edge.py --halfvoid "D:\beisanshe\runs\m3_halfvoid_t20_L-15_4m_seed24091301\canonical_response" --ref-main "D:\beisanshe\runs\m3_ref_t20_L-15_4m_seed24091302\canonical_response" --ref-null "D:\beisanshe\runs\m3_refnull_t20_L-15_4m_seed24091303\canonical_response" --config "D:\beisanshe\configs\m3_edge_main_widefov_20260912.json" --out "D:\beisanshe\outputs\m3_widefov" --null-trials 0
```

`--null-trials 0` 是故意的：这一跑**只取 band_mask 的宽度，T 值一概不看**（原因见判据 8）。

STAGE=b 跑完后，标定档的两跑（`--ref-main` / `--ref-null` 沿用上面同两个底片）：

```
C:\Users\Administrator\AppData\Local\Programs\Python\Python313\python.exe D:\beisanshe\scripts\analyze_mural_edge.py --halfvoid "D:\beisanshe\runs\m3_halfvoidx23_t20_L-15_4m_seed24091311\canonical_response" --ref-main "D:\beisanshe\runs\m3_ref_t20_L-15_4m_seed24091302\canonical_response" --ref-null "D:\beisanshe\runs\m3_refnull_t20_L-15_4m_seed24091303\canonical_response" --config "D:\beisanshe\configs\m3_edge_offset_20260912.json" --out "D:\beisanshe\outputs\m3_offset" --null-trials 200
```

```
C:\Users\Administrator\AppData\Local\Programs\Python\Python313\python.exe D:\beisanshe\scripts\analyze_mural_edge.py --halfvoid "D:\beisanshe\runs\m3_halfvoidx31_t20_L-15_4m_seed24091312\canonical_response" --ref-main "D:\beisanshe\runs\m3_ref_t20_L-15_4m_seed24091302\canonical_response" --ref-null "D:\beisanshe\runs\m3_refnull_t20_L-15_4m_seed24091303\canonical_response" --config "D:\beisanshe\configs\m3_edge_offset2_20260912.json" --out "D:\beisanshe\outputs\m3_offset2" --null-trials 200
```

> **本轮相对 M2 多跑了一档 `refnull`，原因必须写清楚：**
> M2 的 `--ref-null` 用的是 `mural_ref_t10_L-15_2m_seed24091012`，那是 **M1 时期、旧几何**跑的底片。
> 新构型下再拿它当底片，`refnull / ref` 的比值里会混进**几何差异**而不只是统计涨落，
> 负对照（判据 3、4）会假报警，整套判据就废了。
> 所以 STAGE=a 加了一档 `refnull:120:200:120:24091303` —— 与 `ref` 的空鼓摆位逐字相同
> （都挪到 x = 200 mm 看不见），**只换随机种子**，是同一个物理场的两次独立实现。

---

## 四、蒙卡预言（在见到任何 GATE 数据之前冻结）

出处 `design/m3v2_predict.json`，事例级蒙卡，**不含多次散射**——M1 已实测到这会让它**高估对比度**，
所以下面的对比度是上界，不是期望值。

| 重建深度带 | 旧预言 | 新预言 | 旧实测（M2） |
|---|---|---|---|
| 0–8 mm | 9.53% | 23.50% | 12.21% |
| 8–15 mm | 16.89% | 13.25% | 14.07% |
| 15–25 mm | 31.59% | 37.47% | 31.93% |
| 25–35 mm | 34.12% | 44.13% | 26.34% |

| 量 | 旧 | 新 | 新/旧 |
|---|---|---|---|
| 立体角 | 1.840 sr | 1.136 sr | 0.6175 |
| 深 30 mm 处视场半宽 | 27.9 mm | 33.3 mm | 1.19 |
| 每个初级粒子的选出率 | — | — | **1.047** |
| 每次真实湮没的选出率 | — | — | **0.646** |
| 横向刻度系数 k | 0.819（实测 0.797 ± 0.028） | 0.844 | — |
| D1 晶体质量 | 12.37 kg | 2.05 kg | 0.17 |

### 记账陷阱（必读，判据 7 就是为它设的）

`SOURCE_THETA_MAX_DEG` 是**抽样设置**，不是硬件。真实 Na-22 源各向同性发射，
把锥从 45° 收到 35°，等于"同样的初级粒子数代表了更多的真实湮没"。

- **每个初级粒子的选出率：新/旧 = 1.047** ← GATE 日志里直接看到的就是这个
- **每次真实湮没的选出率：新/旧 = 0.646** ← 决定真实测量时长的是这个

**看到"每百万初级粒子选出的事例数几乎没变"不等于计数率没变。**
达到同样统计量所需的真实测量时间是原来的 1 / 0.646 ≈ **1.55 倍**。两个数都要记进结果。

---

## 五、冻结判据（全部在见到 GATE 数据之前定死）

### STAGE=a（halfvoid 边界 −15 mm + ref_main + ref_null）

| # | 判据 | 阈值 | 来源 |
|---|---|---|---|
| 1 | 检出：halfvoid / ref_main 的 `T` | `≥ 30` | M2 用 24 箱、3000 次零试验标定（中位 8.0，99% 19.0，99.9% 27.2，最大 28.0） |
| 2 | 边界位置：`|x_e(世界) − (−15.00)|` | `≤ 10 mm` | 与 M2 逐字相同 |
| 3 | 零假设：ref_null / ref_main 的 `T`；以及 ref_main 对半的 `T` | 两者都 `< 30` | 与 M2 逐字相同（M2 实测 4.5 与 6.7）。**ref_null 必须是本轮新跑的 `m3_refnull`，不能沿用 M2 的旧几何底片** |
| 4 | 交叉：halfvoid/ref_null 与 halfvoid/ref_main 的 `x_e` 相差 | `≤ 5 mm` | 与 M2 逐字相同（M2 实测 +0.49 mm） |
| 5 | 台阶方向：四个带的 `c_b` | 全部 `> 0` | 与 M2 逐字相同 |
| 6 | 对比度不比 M2 差：`c_b(15–25) ≥ 31.9%` 且 `c_b(25–35) ≥ 26.3%` | 见左 | M2 实测值 31.93% / 26.34%，取到小数点后一位 |
| 7 | **两个计数率都记录**：ref_main 每百万初级粒子选出数 | 落在 `[15565, 25942]` | M2 实测 19822/百万 × 预言 1.047 = 20754，取 ±25% |
| 7b | 同时换算并记录每次真实湮没的选出率 = 每初级 × 0.6175 | 应 ≈ M2 的 0.646 倍 | **只报 7 不报 7b 视为判据未完成** |
| 8 | 视场：用 widefov 配置量各带有效窗（ref ≥ 30 计数的箱） | 0–8 带 `≥ ±16.5 mm`，25–35 带 `> ±34.5 mm` | M2 实测四带有效窗 ±16.5 / [−22.5,+19.5] / [−25.5,+28.5] / ±34.5 mm |

**判据 6 用 15–25 与 25–35 两个带，不用 0–8**：浅带的单带拟合一向不稳（M2 里它的有效窗只有 ±16.5 mm，
边界又恰在 −15，几乎顶到窗边），蒙卡对它的预言从 9.5% 跳到 23.5% 也最不可信。

**判据 8 为什么要单开一个配置**：M2 实测里最深的 25–35 带有效窗已经顶到 ±34.5 mm，
也就是**被直方图边界截断而不是被物理截断**，在 ±36 mm 的窗里根本量不出 +19% 的视场增益。
但箱数一变零分布就变，`T_thr = 30` 这个阈值就不再适用，M3 与 M2 的 `T` 也不能直接比。
所以：**判据 1~7 与 9 全部在 ±36 mm 的原配置上做，判据 8 单独在 ±48 mm 的配置上量宽度，那一跑的 `T` 一概不看。**

### STAGE=b（offset −23 mm + offset2 −31 mm，重新标定横向刻度）

| # | 判据 | 阈值 | 来源 |
|---|---|---|---|
| 9 | 三点（main / offset / offset2）自由截距拟合 `û − L = k·u + b` | `|k − 0.844| ≤ 0.09` 且 `|b| ≤ 1 mm` 且 `r² ≥ 0.98` | 0.09 ≈ 3 × σ(k)，σ(k) 取 0.03（M2 四点拟合给出 0.0275，三点会略大） |

> **判据 9 的能力边界，必须写清楚：**
> 蒙卡预言 `k` 从 0.819 涨到 0.844，变化量 **0.025 比标定精度 σ(k) ≈ 0.03 还小**。
> 也就是说**这条判据没有能力验证那 +0.025 的预言**，它只能证伪"条件选择这套机制完全错了"
> （比如量出 k ≈ 1.0 或 k ≈ 0.6）。
> 想真正检验 +0.025，需要把每档的事例数提高到 4 倍以上，或者加更多离轴档。本轮不做。

---

## 六、判完之后要报的数（一条都不能省）

1. 九条判据逐条：通过 / 不通过 + 实测值
2. `x_e` 及其 68% 区间；与真值的差
3. 四个带的 `c_b` 与有效窗
4. **两个计数率**：每百万初级粒子选出数、每次真实湮没的选出率（判据 7 / 7b）
5. `T` 与零假设的两个 `T`
6. 与 M2 的逐项对照（上面第四节的表里已经有 M2 那一列）
7. 实际墙上时间，以及它与"60~90 分钟"预估的差

---

## 七、没做的事（不掩饰）

| 项 | 现状 |
|---|---|
| 时间分辨、偶然符合 | **未建模**，所以本轮的对比度不含偶然符合造成的本底抬高 |
| 空鼓厚度 | 固定 20 mm，**没有扫描**，不知道多薄还能测出来 |
| MLEM 收敛判据 | **无自动停止**，固定迭代次数 |
| D2 位置口径 | 仍用重心法（4.15 mm）；康普顿序列重建（2.53 mm）只留了接口，未实现 |
| 新构型的横向 / 深度 FWHM | **没有跑标准件**，本轮只量边界定位，不报分辨率数字 |
| D1 外半径 100 mm、D1 中面 7.5 mm | **我推断的，未经用户确认，也未经 GATE 之外的任何验证** |
