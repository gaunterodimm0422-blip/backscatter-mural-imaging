# 背散射成像先进重建算法调研、复现与阶段评价

图像逐项读法、有效视野梯形来源、公式和可复现命令见：`outputs/advanced_reconstruction_20260807/孔洞成像图解与先进算法复现说明.md`。

## 1. 研究目的

本次工作的目标不是单纯把孔洞图“画得更圆”，而是在当前 GATE 数据和探测器性能条件下，比较不同重建方法能否同时做到：提高孔洞可见性、抑制完整水泥中的假结构，并保持已经通过独立验证的横向空间分辨率。

当前硬件响应按以下参数处理：D1、D2 均为 GAGG；D1 位置分辨率为 `(1.55, 1.55, 5.0) mm`、能量分辨率为 `8%@511 keV`；D2 位置分辨率为 `(1.0, 1.0, 2.5) mm`、能量分辨率为 `12%@511 keV`。

## 2. 当前闭式重建的物理含义

对每一个 D1–D2 符合事件，D2 击中位置与源位置先确定入射射线；D1 沉积能量通过康普顿公式给出散射角；D1 击中位置再提供出射方向约束。程序逐事件求解入射射线上的相互作用点，最后统计散射点密度形成图像。

这种方法计算快、物理含义清楚，而且与本项目“D2 标记入射方向”的装置结构最匹配。但它把具有能量和位置不确定度的一次测量压缩为单个重建点，因此有限能量分辨率会转化为空间散布，尤其会拉长深度方向响应。

## 3. 文献方法与本项目适配性

| 方法 | 文献中的核心思想 | 本项目中的判断 |
|---|---|---|
| 标记射线的能量闭式定位 | 由入射方向、散射能量与出射方向求相互作用位置 | 与当前结构最匹配，保留为物理基线 |
| 飞行时间定位 | 用两路时间差估计散射点沿射线的位置 | 当前数据没有足够的时间分辨信息，暂不复现 |
| 概率射线反投影 | 不把事件压成一个点，而是在整条候选射线上按测量似然分配权重 | 已复现；斑点减少，但形成明显深度条纹 |
| List-mode MLEM / OSEM | 每个事件以概率系统矩阵进入最大似然迭代 | 已实现参考场校正原型；当前系统矩阵尚不完整，负对照伪影被放大 |
| 分辨率恢复 LM-OSEM | 将位置相关点扩散函数写入迭代模型 | 有潜力，但需要更完整的探测器响应、衰减和多次散射模型 |
| SOE | 以马尔可夫链抽样康普顿事件的可能起点 | 更适合普通康普顿锥几何；本装置候选空间已被 D2 压缩到射线，不是当前首选 |
| TV 正则化 | 全局抑制随机纹理，同时尽量保留边缘 | 已复现；用独立负对照与细杆约束选参后效果最稳健 |

## 4. 本项目实际复现的三条路线

### 4.1 概率射线反投影

程序在 D2 标记射线与水泥板相交的每一个候选位置上，计算该位置对应的几何散射角和理论 D1 能量，再根据 D1 的 `8%@511 keV` 能量响应以及 D1/D2 位置分辨率计算事件似然。每个事件不再只贡献一个点，而是在候选射线上分配概率。

该方法确实减少了闭式图中的随机斑点，但由于 D1 能量展宽对应较大的康普顿角不确定度，概率沿射线方向被摊开，形成明显的轴向条纹。它适合继续发展系统响应模型，但当前不作为正式孔洞图。

### 4.2 参考场校正的 list-mode MLEM + TV 原型

程序把逐事件候选位置似然组成稀疏系统矩阵，以完整水泥偶数事件估计经验灵敏度，用奇数事件作为独立留出负对照，然后执行 list-mode MLEM 更新并测试轻度 TV 正则化。

按预先规定的参考场选择规则得到的候选为 2 次迭代、TV 权重 0.02。它在孔洞数据上的图像域分离度为 1.22，但完整水泥留出组的场内伪影 RMS 为 0.307，高于未正则化闭式参考场的 0.254。因此该原型被保留为算法试验，没有进入正式成像结果。

主要原因不是 MLEM 原理失效，而是当前原型系统矩阵还没有完整描述 511 keV 光子在水泥中的衰减、多次散射、多普勒展宽和空间变化的探测效率。若在这种条件下继续增加迭代次数，算法会把模型误差和统计涨落一起放大。

### 4.3 闭式物理重建 + TV 正则化

程序先使用原有逐事件闭式物理定位形成计数密度图，再对整幅图施加同一个 TV 正则化参数。没有对孔洞区域进行局部手工修图，也没有让孔洞真值参与重建。

TV 权重从预先给定的 `0、0.02、0.05、0.08、0.12` 中选择。选择依据是：完整水泥独立留出组的场内伪影 RMS 最小，同时独立 2 mm 细杆的横向 FWHM 必须不超过 5 mm。最终采用权重 0.12。

## 5. 定量结果

| 指标 | 闭式基线 | 概率射线 | LM-MLEM 原型 | 闭式 + TV（采用） |
|---|---:|---:|---:|---:|
| 孔洞图像域标准化分离度 | 1.80 | 2.00 | 1.22 | **2.84** |
| 孔洞表观横向半高宽 | 10.1 mm | 12.4 mm | 11.7 mm | **18.7 mm** |
| 完整水泥同位置假缺失 | 3.62% | 3.19% | 5.64% | **2.45%** |
| 完整水泥场内伪影 RMS | 0.254 | 0.0736 | 0.307 | **0.192** |

20 mm 孔洞在最终图上的缺失重心为 `(0.5, 31.0) mm`，接近真值 `(0, 30) mm`。18.7 mm 是孔洞影像的表观宽度，不是系统空间分辨率。

正式横向空间分辨率仍由独立的 2 mm 细杆验证给出：FWHM 为 **3.67 mm**，bootstrap 95% 区间为 **3.11–4.42 mm**，满足不超过 5 mm 的要求。TV 处理前后该值分别为 3.67225 mm 和 3.67236 mm，可视为没有变化。

深度 FWHM 约为 **38.47 mm**。因此当前系统可以支持“横向约 5 mm”，不能宣称“横向与深度方向都达到 5 mm”。新版图保留轴向拉长，正是为了如实显示这一限制。

## 6. 阶段结论与下一步

当前最稳健的工程方案是“D2 标记射线的闭式物理定位 + 全局 TV 正则化”。它比原图更平滑、更易辨认孔洞，同时通过完整水泥负对照和独立细杆试验限制了过度平滑。

概率模型和 list-mode MLEM 仍值得继续研究，但下一阶段应先补齐系统矩阵，而不是直接增加迭代次数。建议依次加入：水泥内路径衰减、单次与多次散射分类、多普勒展宽、位置相关探测效率，以及多个独立随机种子的训练/验证数据。模型完整后再重新比较 LM-OSEM、分辨率恢复和正则化参数。

## 7. 主要原始文献

- D. Lee et al., *Three-Dimensional Imaging of Hidden Objects Using Positron Emission Backscatter*, 2009, DOI: `10.1109/NSSMIC.2009.5402164`。
- Q. Looker et al., *Demonstration of imaging via backscattering of annihilation gamma rays*, 2010, DOI: `10.1016/j.nima.2010.01.078`。
- J. Gerl et al., *High-resolution gamma backscatter imaging for technical applications*, 2004, DOI: `10.1016/j.nima.2004.03.130`。
- S. J. Wilderman et al., *List-Mode Maximum Likelihood Reconstruction of Compton Scatter Camera Images in Nuclear Medicine*, 1998。
- A. Andreyev et al., *Fast image reconstruction for Compton camera using stochastic origin ensemble approach*, 2011, DOI: `10.1118/1.3528170`。
- S. M. Kim et al., *Resolution recovery reconstruction for a Compton camera*, 2013, DOI: `10.1088/0031-9155/58/9/2823`。
- C. Wu et al., *An accurate probabilistic model with detector resolution and Doppler broadening correction in list-mode MLEM reconstruction for Compton camera*, 2022, DOI: `10.1088/1361-6560/ac73d2`。
- Y. Gu et al., *Image reconstruction based on total variation minimization and alternating direction method for Compton scatter tomography*, 2014, DOI: `10.7498/aps.63.018701`。

## 8. 可复现文件

- 主结果与全部参数：`outputs/advanced_reconstruction_20260807/advanced_reconstruction_report.json`
- 图源数值：`outputs/advanced_reconstruction_20260807/advanced_void_figure_source.csv`
- 算法候选指标：`outputs/advanced_reconstruction_20260807/algorithm_candidate_metrics.csv`
- TV 候选指标：`outputs/advanced_reconstruction_20260807/tv_candidate_metrics.csv`
- 重建矩阵与图像：`outputs/advanced_reconstruction_20260807/advanced_reconstruction_maps.npz`
- 实现代码：`backscatter/iterative.py`、`scripts/reconstruct_void_advanced.py`、`scripts/build_advanced_void_figure.py`
