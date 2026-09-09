# 三阶段研究概览

- [英文概览图](../three_stage_research_public.pdf)：中英文报告共用，当前为中文报告图 2。
- [图形源](figure_three_stage_research.tikz.tex)：文字、坐标与连接关系。
- [字体与颜色](figure_three_stage_research.tex)：统一设置。

在仓库根目录运行：

```bash
bash reports/zh/figures/program/build.sh
```

`make report` 也会自动重建本图。成品宽度为 160 mm，适合整栏使用；更窄的版面应重排，不应直接缩小文字。

上方为研究目标，中间三个科学阶段共同构成 Qiushi Discovery Loop，下方突出两代前沿模型、数据高效学习原则与开放研究材料。树形分叉表示多方向并行探索，节点数不编码实际实验数量。模型符号表示 Transformer 类模型，不代替实际架构说明。
