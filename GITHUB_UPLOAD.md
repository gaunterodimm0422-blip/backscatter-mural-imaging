# GitHub首次发布与上传范围

日期：2026-10-16。适用目录：`C:\Users\HOUWENDI\Documents\beisanshe`。

## 本次准备了什么

- 保留源码、脚本、配置、测试、项目记录、根目录交付文件及正式15%结果。
- 纳入`gate_mural/`：现行壁画GATE必要宏和材料库、两个冻结跑批脚本，复制后与E盘原件逐项核对SHA-256。
- 纳入4个约5 KB的`imaging_demo_data.npz`，供从已有聚合数据重画图。
- `outputs/`只纳入24个具名正式小图/标定文件及`m3v2_predict.json`设计输入；后者是历史预测，不是正式性能判据。
- 排除ROOT、事件级NPZ、`runs/`、`.codex_work/`、`tmp/`、历史大图/嵌套导出目录、凭据文件和虚拟环境；只是忽略上传，不删除本机原件。
- `.gitattributes`禁止自动换行转换，避免克隆时破坏冻结文件的SHA-256或shell脚本换行。

根目录`README.md`已补当前壁画方案入口，保留的早期水泥正文被明确标为历史。原仿真结果、判据和图件没有重算或改写。

## 由用户手动发布

当前截图是GitHub Desktop的“Let's get started!”页面。请按以下顺序操作：

1. 点击右侧最下面的 **Add an Existing Repository from your local drive...**。不要选择新建仓库，也不要复制出另一个嵌套项目目录。
2. 在本地路径中选择或粘贴：

   ```text
   C:\Users\HOUWENDI\Documents\beisanshe
   ```

3. 点击 **Add Repository**。
4. 本地首次提交准备完成后，Changes页显示没有待提交变更是正常的；顶部应提供 **Publish repository**。
5. 点击 **Publish repository**，仓库名建议填写：

   ```text
   backscatter-mural-imaging
   ```

6. **保留勾选“Keep this code private”**。这是研究备份，不是现在就公开论文与拟专利技术内容。不要添加不必要的开放许可或更改可见性。
7. 选择自己的个人账号，点击窗口下方 **Publish Repository**，等待完成。
8. 点击 **View on GitHub**，确认页面标记为 **Private**，并能看到`README.md`、`gate_mural`、`backscatter`和`项目文档`。这一步才能确认云端已经有备份。

如果出现红色错误、要求重新登录、已有同名仓库、或界面与以上不同，请截图发回来，不要强制推送、删除本地文件或另外新建嵌套目录。登录、验证码及授权确认由用户自己完成，不向agent发送密码或token。

## 后续更新怎么操作

修改文件后，在GitHub Desktop左侧查看差异，填写一条说明，点击 **Commit to main**，再点击 **Push origin**。

- Commit：在本机保存一个版本。
- Publish repository：第一次在GitHub建立远程仓库并上传。
- Push origin：把后续本机版本上传到已经关联的远程仓库。

若只是修改了被忽略的原始数据或临时文件，Changes中不会出现，不能把“没有变更”理解为这些大数据也自动备份了。

## 数据与复现边界

正式原始ROOT据记录保存在远控机`D:\vgate_shared\output\<RUN_ID>\`，完整15%事件级响应缓存据记录在`D:\beisanshe\runs\<RUN_ID>\response_d2f15\`；本次没有连接远控机重新核验或迁移它们。

本仓库能提供代码、必要壁画宏、配置、正式结果和聚合绘图数据，不能代替原始数据归档。克隆到新电脑后若要从ROOT重新全分析，仍需另行取得这些原始文件并按机器/路径手册配置运行环境。仅检查现有结果、阅读报告和重画现有聚合图不需要重新跑GATE。
