# Scientific figures

Chinese and English reports share these English-language vector figures.
Captions follow the language of each report. Numerical results come from the
repository's `results/` directory.

| Figure | Editable source |
| --- | --- |
| [Leaderboard](leaderboard.pdf) | [Python](../render.py) |
| [Three-stage research](three_stage_research_public.pdf) | [TikZ](program/figure_three_stage_research.tikz.tex) |
| [Relation and context use](relation_context_use.pdf) | [Python](../render.py) |
| [Functional access](interface_reach.pdf) | [Python](../render.py) |
| [Input masking and supervision](input_supervision.pdf) | [TikZ](input_supervision.tex) |
| [Training strategies](strategy_comparison.pdf) | [Python](../render.py) |
| [Task contributions](task_contributions.pdf) | [Python](../render.py) |
| [Research lineage](research_lineage.pdf) | [TikZ](research_lineage.tex) |

Run `make report` from the repository root to rebuild all figures and the Chinese
report. TikZ produces conceptual diagrams; Python/Matplotlib with the LaTeX PGF
backend produces data plots. Typeface, colors and layout rules are defined in
[STYLE.md](../../../assets/STYLE.md).
