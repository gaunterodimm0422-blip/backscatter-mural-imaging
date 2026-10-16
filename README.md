# 壁画空鼓的511 keV符合标记背散射检测

## 当前项目入口（2026-10-16）

本仓库当前主线是：用GATE / Geant4输运仿真、闭式候选点重建和多重建深度带共享边界统计判定，研究**壁画空鼓的检出与横向边界定位**。目前已有仿真及分析结果，尚无实物实验数据。

先阅读[交接入口与文件索引](项目文档/00_交接入口_当前状态与文件索引_20261016.md)，或直接查看[最新16页项目PPT](deliverables/壁画空鼓背散射检测_项目汇报_20261013_修正版.pptx)与[逐页讲稿](deliverables/壁画空鼓背散射检测_逐页讲稿_20261013.md)。[框架逐项说明](deliverables/仿真论文框架逐项说明_20260930.md)专门解释[论文框架PDF](deliverables/framework_sim_paper.pdf)。

当前方案的D1是贴近壁面的环形GAGG背散射探测器，D2是源另一侧的像素化GAGG标记阵列。正式能量响应为D1 8%、D2 15% FWHM @511 keV。壁面z=0、+z朝探测器；D1内/外半径12.5/100 mm、厚10 mm、z=2.5–12.5 mm，源位于局部z=17.5 mm，D2有效面约50×50 mm、厚20 mm、z=42.5–62.5 mm。以[当前主配置](configs/m3_edge_main_d2f15_20260923.json)及相应输入结果为准。

本次界面空鼓扫描中，20 mm档通过检出与定位；10 mm档通过检出、但边界区间过宽；5、3、2、1 mm未通过检出。各档空气层下边界固定深40 mm、各缺陷场400万初级事件，不能将结果外推为任意材料/埋深的通用厚度极限。边界参数的统计区间半宽不等于系统空间FWHM。

| 目录 | 当前用途 |
|---|---|
| `gate_mural/` | 从现行E盘副本复制的壁画GATE宏、材料库和冻结跑批脚本；见其README |
| `backscatter/`、`scripts/`、`configs/`、`tests/` | 分析实现、驱动脚本、配置与测试 |
| `m3/outputs/d2f15/` | 正式15%分析结果和4份小型聚合绘图NPZ；不是逐事件数据库 |
| `outputs/` | 当前保留的24个小图/标定文件及历史设计预测输入 |
| `deliverables/`、`项目文档/` | 报告、演示稿、交接及研究记录 |
| `gate/`、`docs/`、`report_latex/` | 较早水泥基线及历史阶段材料，不是当前壁画几何的唯一入口 |

**仓库不包含正式原始ROOT、完整事件级响应缓存、虚拟机镜像和历史大图缓存。** 这些按记录仍在远控机或本地归档中，详细位置见交接入口。当前GATE运行文件已补到`gate_mural/`；原E盘文件未修改。大数据未上传不等于已经删除或已完成异地备份。

本次仓库先按私有研究备份准备；发布与后续公开范围由用户决定。首次发布步骤及上传范围见[GITHUB_UPLOAD.md](GITHUB_UPLOAD.md)。

---

## 历史阶段记录：早期水泥基线

以下旧正文保留用于追溯；其中“当前”“权威几何”等字样指早期水泥研究阶段，不取代上面壁画主线的正式配置与结果。

### 511 keV双光子标记背散射成像：原基线说明

这是项目的干净基线版本。旧项目保留在
`C:\Users\HOUWENDI\Claude\Projects\背散射`，这里不直接修改旧文件。

当前阶段的任务是先得到一条可验证、可重复的链路：

1. 在 vGate / GATE 9.4 中运行固定几何、固定随机种子的水泥基线；
2. 同时输出 D1 背散射探测器和 D2 标记探测器的 Hits；
3. 记录水泥中的康普顿次数、末次散射体积以及散射顶点真值；
4. 在 Windows 主机端汇总事件、构建双光子符合、重建散射点；
5. 先验证能谱、运动学和真值闭环，再开展分辨率体模与参数扫描。

这条链路现已在 vGate-9.4.1 / GATE 9.4.1 / Geant4 11.3.0 上完成：

- 200 万事件完整水泥板的 D1/D2 能谱、重建和 10 mm 深度分层；
- 10、30、50、70、90 mm 五个 200 万事件孤立薄层的本征轴向响应；
- 能量窗、入射/散射角、D1 半径/扇区和 D2 区域的统一扫描；
- D1 尺寸—质量—效率的后处理候选评估。

单杆、双杆和边缘体模已通过宏与分析程序验收；1/2/5 mm 单杆正式大样本
已经完成，横向 x FWHM 分别为 4.43、3.49、4.59 mm，其中 1 mm 杆仅有
88 个有效事件、仍需补统计。30/70 mm 边缘体模正式大样本也已完成，模型化
MTF50 分别为 0.0491/0.0263 cycles/mm，MTF10 分别为
0.0895/0.0480 cycles/mm。2 mm 直径双杆的 5/10/15 mm 中心距正式任务也已
完成：5 mm 为带宽敏感的临界情况，10/15 mm 稳健可分辨。孤立薄层只用于测量
系统本征几何响应；完整水泥板中的成像深度和灵敏度按真实康普顿顶点深度
分层评估。导师要求的逐条阶段结果见
[docs/05_导师要求逐条阶段汇报.md](docs/05_导师要求逐条阶段汇报.md)。
第二阶段在揭盲前冻结的比较方法和通过条件见
[docs/06_第二阶段实验设计与预注册判据.md](docs/06_第二阶段实验设计与预注册判据.md)。
面向跨专业初学者、可直接用于和导师交流的通俗版记录见
[docs/07_给导师讲明白目前做了什么_通俗版.md](docs/07_给导师讲明白目前做了什么_通俗版.md)。
四组正式水泥孔洞 pilot 的方法、图像、CNR 和结论边界见
[docs/08_水泥孔洞pilot结果.md](docs/08_水泥孔洞pilot结果.md)。
孔洞盲检为什么需要、评分怎样计算以及答案如何隔离见
[docs/09_水泥孔洞盲检与评分方案.md](docs/09_水泥孔洞盲检与评分方案.md)。
12 例盲检流程验收的揭盲结果和结论边界见
[docs/10_水泥孔洞盲检流程验收结果.md](docs/10_水泥孔洞盲检流程验收结果.md)。
30/70 mm 边缘 MTF 的正式结果、模型假设和初算方法修订说明见
[docs/11_边缘MTF正式结果.md](docs/11_边缘MTF正式结果.md)。
5/10/15 mm 双杆的谷峰比、带宽稳健性和停止 20/30 mm 的依据见
[docs/12_双杆可分辨性正式结果.md](docs/12_双杆可分辨性正式结果.md)。
导师新增的能谱线性坐标、D2 能量展宽、非全能峰事件利用和尺寸—灵敏度曲线见
[docs/13_导师新增三项要求落实结果.md](docs/13_导师新增三项要求落实结果.md)。
按论文结构组织、图片直接嵌入正文的阶段性研究报告见
[docs/14_背散射成像系统蒙特卡洛构建与性能评价_阶段性研究报告.md](docs/14_背散射成像系统蒙特卡洛构建与性能评价_阶段性研究报告.md)。
面向跨专业初学者的项目原理、十分钟讲稿、常见问答和速记卡见
[docs/15_项目全面介绍与导师汇报教程_跨专业入门版.md](docs/15_项目全面介绍与导师汇报教程_跨专业入门版.md)。

第二阶段的独立随机种子复现和 D1 R=150/175 mm 物理缩径复算已经完成，
均通过揭盲前冻结的判据；R=150 mm 是当前理想几何下的首选尺寸候选。
1/2/5 mm 单杆的 500 万事件正式任务已经完成并通过回传验收。完整水泥内的圆柱
真空孔洞体模四组 200 万事件 pilot 已完成：10 mm 孔洞在 30/60 mm 深度的
CNR 为 2.38/1.42，20 mm 孔洞为 9.13/0.98；目前只有 20 mm、30 mm 深度
满足进入多随机种子盲检的预注册门槛。0/5/10/15% FWHM@511 keV 电子学
敏感性和探测器位置读出/DOI 敏感性分析也已加入。20 mm、30 mm 深度孔洞
的 12 例答案隔离盲检流程验收已经完成：6/6 孔洞样本正确定位检出，0/6
无孔洞样本误报，评分和带盐答案承诺均通过 SHA-256 验证。该小样本结果只
说明流程验收通过，不能写成最终 100% 检出率或 0% 误报率。30/70 mm 边缘
正式任务也已完成；较深目标的模型化 MTF50/MTF10 均降至 30 mm 结果的约
54%。双杆第一批也已完成，当前最小已测试的带宽稳健可分辨中心距为 10 mm；
20/30 mm 按自适应停止规则不再运行。
导师新增的三项后处理要求也已完成：图 b 改为线性计数坐标，图 c 加入明确
标注的假设能量展宽；D2 非全能峰下限扫描确认理想仿真中最多可增加约 50%
真信号；D1 尺寸图已直接绘制每 100 万初级事件的成像灵敏度。自动测试现为
52 项。

## 历史水泥基线几何（非当前壁画构型）

- 坐标约定：`+z` 从被测物体指向探测器，水泥上表面为 `z = 0 mm`。
- 源：`(0, 0, 17.5) mm`，两个反向的 511 keV 光子，锥角 `0–45°`。
- D1 背散射探测器：GAGG 环，`R = 20–200 mm`，厚 `15 mm`，中心 `z = 17.5 mm`。
- D2 标记探测器：`50×50` 个 GAGG 晶体，晶体 `0.93×0.93×20 mm³`，
  阵列节距 `1.00 mm`，中心 `z = 60 mm`。
- 基线物体：`500×500×100 mm³` 水泥，`z = [-100, 0] mm`。

完整的冻结参数和判据见
[configs/baseline.json](configs/baseline.json) 与
[docs/00_权威基线与判据.md](docs/00_权威基线与判据.md)。

## 第一次运行

先按 [docs/01_vGate首轮运行.md](docs/01_vGate首轮运行.md) 在虚拟机中运行
2 万初级事件的冒烟测试。得到 ROOT 文件后，在本机执行：

```powershell
python -m backscatter inspect-root --root "ROOT文件路径"
python -m backscatter analyze `
  --root "ROOT文件路径" `
  --truth "真值ROOT文件路径" `
  --config "configs/baseline.json" `
  --out-dir "runs/smoke_baseline"
```

无 ROOT 文件也可以先验证运动学求解器：

```powershell
python -m unittest discover -s tests -v
python -m backscatter selftest
```

已有一次 `analyze` 输出后，可生成导师第 3、4 条所需的统一筛选/分区表：

```powershell
python -m backscatter sweep `
  --analysis-dir "runs/smoke_concrete" `
  --config "configs/baseline.json"
```

没有 actor 真值和标准体模时，扫描表只报告事件效率与分布位置，不会把这些量
误标为系统分辨率。

标准薄层配置与评估示例：

```powershell
python -m backscatter standard-config `
  --kind thin_layer --depth-mm 30 --thickness-mm 1 `
  --primary-events 2000000 `
  --out "configs\generated\thin_layer_depth30.json"

python -m backscatter evaluate-standard `
  --analysis-dir "runs\depth_psf_30mm_seed24061873_local" `
  --config "configs\generated\thin_layer_depth30.json"
```

单杆横向 PSF 使用 `--kind rod`，并以 `--radius-mm` 的两倍作为实际杆宽。

完整 100 mm 水泥板的大样本分析使用：

```powershell
python -m backscatter evaluate-concrete-depth `
  --analysis-dir "runs\concrete_slab_2m_seed24061873_local" `
  --config "configs\generated\concrete_slab_2m.json" `
  --bin-width-mm 10 --minimum-events 100 --bootstrap 200
```

该命令给出每个真实深度分箱的有效事件灵敏度、轴向偏差/RMSE、轴向误差
FWHM 和置信区间。它报告的是带明确统计门槛的“本次运行可评估深度”，不会
把它表述成与计数时间无关的绝对最大穿透深度。

水泥孔洞配置与均匀水泥参照的对比评价示例：

```powershell
python -m backscatter void-config `
  --depth-mm 30 --radius-mm 5 --length-mm 300 `
  --primary-events 2000000 `
  --out "configs\generated\concrete_void_d10_depth30_2m.json"

python -m backscatter evaluate-void `
  --void-analysis-dir "runs\concrete_void_d10_depth30_seed24061875_local" `
  --reference-analysis-dir "runs\concrete_slab_2m_seed24061873_local" `
  --config "configs\generated\concrete_void_d10_depth30_2m.json" `
  --out-dir "runs\concrete_void_d10_depth30_seed24061875_local\void_metrics"
```

该评价强制使用 `phantom_truth_policy=ignore` 构图。已知位置的单次 CNR 只作
pilot 效应量，不能直接称为盲检检测概率。

电子学分辨率假设情景：

```powershell
python -m backscatter electronics-sensitivity `
  --analysis-dir "runs\concrete_slab_2m_seed24061873_local" `
  --config "configs\generated\concrete_slab_2m.json" `
  --out-dir "runs\concrete_slab_2m_seed24061873_local\electronics_sensitivity" `
  --fwhm-percent 0 5 10 15
```

位置读出与 DOI 假设情景：

```powershell
python -m backscatter position-readout-sensitivity `
  --analysis-dir "runs\concrete_slab_2m_seed24061873_local" `
  --config "configs\generated\concrete_slab_2m.json" `
  --out-dir "runs\concrete_slab_2m_seed24061873_local\position_readout_sensitivity" `
  --d1-xy-fwhm-mm 0 2 5 10
```

其中理想行使用连续 GATE 能量质心；其他行把 D2 量化到晶体中心、取消
D1/D2 DOI，并扫描 D1 平面位置分辨率。它们是硬件敏感性情景，不是实测规格。

## 目录

- `gate/`：在 vGate 中执行的 GATE 9.4 宏与批处理脚本。
- `backscatter/`：Windows 主机端的 ROOT 读取、符合、重建和审计程序。
- `configs/`：权威参数与后续实验矩阵。
- `docs/`：坐标、验收标准、运行交接和阶段说明。
- `tests/`：不依赖 GATE 的公式与配置自检。

## 重要口径

- 平板水泥的 X/Y/Z 分布宽度不是系统分辨率。
- 系统本征分辨率必须用已知真值的薄层、细杆、双杆或边缘体模测量；材料内
  的实际深度性能必须再用完整介质中的真值深度分层验证。
- 灵敏度同时报告“每个初级事件的有效计数效率”和
  `cps/MBq`；后者只在低计数率、一个初级事件对应一次衰变的口径下换算。
- 参数扫描必须基于同一批原始事件或同一统计量级，并同时报告
  分辨率、CNR/对比度与灵敏度，不能只优化其中一项。
- `nPhantomCompton`、`comptVolName` 和 actor 顶点属于仿真真值，不能冒充
  实验筛选条件。扫描先用能量、时间、角度和探测器位置等可观测量选事件，
  再用真值评价目标信号效率、纯度和分辨率。
