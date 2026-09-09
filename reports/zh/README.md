# 中文研究报告

[阅读 PDF](qiushi-engine-babylm-report-zh.pdf)：《数据高效语言建模：从前沿突破到原理指导的模型改进》。报告介绍 Qiushi Engine 在 BabyLM 2026 Strict-Small 上的连续自主研究：前沿模型的创造怎样产生新的学习认识，这些认识又怎样改变下一代模型的训练。

正文依次介绍研究背景、三个研究阶段、独立发现、综合讨论和结论。附录保留算法、完整结果、配套工程入口与 74 个研究主题。报告中的每类材料均可从本仓库继续查阅。

[英文报告](../en/qiushi-engine-babylm-report-en.pdf)与本版共用结果数据、[参考文献库](../references.bib)和八幅英文科研图。参考文献按正文首次引用顺序编号，两版条目与编号一致；英文源文件及构建说明见 [reports/en](../en/README.md)。

每个阶段都包含问题提出、方法构造、实验检验和成果整理。摘要凝练关系学习、能力复用与保持的设计原则；研究谱系章按经验、表示、学习目标、训练动力学及测量实现归纳更广泛的成果。Hugging Face 提供已公开的两代模型，配套研究工程将模型与完整科研材料联系起来。发布范围统一记录在下方的发布状态中。

综合讨论简述 Qiushi Engine 的分层多智能体组织及其前期公开研究；附录说明 AI Lab 计算支持、两张 H100 的研究用途、计算平台与软件环境，并区分原始实验配置和模型发布验证环境。

报告以 Research RSI（科研过程的递归自我改进）讨论科学认识、方法创新与实验经验如何改变后续研究。摘要、阶段衔接、讨论和结论保持一致，并与 STOP、Darwin Gödel Machine、Reflexion 和 Co-Scientist 等相关工作比较。实证范围是同一 BabyLM 项目内的连续研究与模型改进。

16 位作者按四行、每行四位排列；陈红胜（Hongsheng Chen）倒数第二，杨怡豪（Yihao Yang）末位署名。两位通讯作者以星号标注，通讯联系优先列出杨怡豪，邮箱分别为 yangyihao@zju.edu.cn 与 hansomchen@zju.edu.cn。

## 报告与配套材料

| 阅读内容 | 对应材料 |
| --- | --- |
| 两代模型及实际使用 | [模型目录](../../models/README.md) |
| 方法与实验设计 | [方法说明](../../methods/README.md)、[核心程序](../../experiments/CORE_PROGRAMS.md) |
| 训练、数据与评测 | [运行指南](../../reproducibility/TRAINING.md)、[数据构造](../../data/RECONSTRUCTION.md) |
| 图表与完整结果 | [统一结果数据](../../results/)、[实验与章节对应](../../experiments/index.csv) |
| 主要发现及其依据 | [结论与证据](../../evidence/claim_evidence_map.md) |
| 科学问题与设计演进 | [32 条专题阅读路径](../../research/reading_paths.md)、[科学导读](../../research/scientific_guide.md) |
| 科研笔记、推导与分析 | [研究笔记](../../research/notes/README.md)、[测量与技术记录](../../research/documents/README.md) |
| 实验假设与规划 | [实验规划](../../research/plans/README.md) |
| 独立发现与术语解释 | [主题材料地图](../../research/materials.md)、[英文术语与实验条件](../../research/terms.md) |

## 修改与重建

编辑入口为 `main.tex` 和 `chapters/`；样式、作者在 `latex/`，两版共用的参考文献为 `../references.bib`，矢量图与 TikZ 源在 `figures/`。科研图全部使用英文标签，供中英文报告共同使用；中文图注保留必要解释。数据直接读取仓库根目录的 `results/`，不要另建数据副本。

在仓库根目录运行 `make report`，或者在本目录运行 `bash build.sh`。构建会先生成图表，再编译并检查 PDF，通过后更新阅读文件。中间文件在 `build/`，生成表格在 `tables/`，均不纳入源码发布。

中文报告、英文对齐和配套材料的发布状态见 [发布状态](../../reproducibility/STATUS.md)。报告构建不执行训练、模型评测或远端发布。
