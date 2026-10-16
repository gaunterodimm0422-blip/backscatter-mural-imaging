# 阶段性研究报告 LaTeX 工程

主文件：`main.tex`

图片：`figures/`（均为从正式分析结果复制的矢量 PDF，文件名已固定）

## 推荐编译方式

使用 XeLaTeX 连续编译两次，以生成正确的目录、图表编号和交叉引用：

```powershell
xelatex main.tex
xelatex main.tex
```

在 TeXstudio 中，将默认编译器设为 **XeLaTeX**，然后连续点击两次“构建并查看”。

本报告使用 `ctexart` 和 Fandol 中文字体，避免依赖某一台电脑上的宋体或黑体。
不要使用 pdfLaTeX 编译中文正文。

## 文件关系

- 原始内容来源：`../docs/14_背散射成像系统蒙特卡洛构建与性能评价_阶段性研究报告.md`
- LaTeX 主文件：`main.tex`
- 报告图件：`figures/`
- 本机已编译版本：`build/main.pdf`

原 Markdown 文件未被修改。
