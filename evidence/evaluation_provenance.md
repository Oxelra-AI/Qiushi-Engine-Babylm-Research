# Evaluation Provenance and Interpretation

The public comparison uses the report's snapshot from 8 September 2026 at 20:53 Beijing time. Local training controls use the same loading protocol with all adapters loaded and retain all nine original component values and continuation seeds. Public scores displayed to two decimal places and local score vectors remain separate.

The nine top-level components are BLiMP, BLiMP Supplement, EWoK, Entity Tracking, COMPS, GlobalPIQA, (Super)GLUE, Reading and AoA. Overall is their arithmetic mean; NLP and Human-like are displayed group summaries. Overall is not accuracy pooled over a common set of questions.

GlobalPIQA's two subsets contain 103 and 100 questions and are aggregated by averaging the subset scores. Reading measures correspondence with reading behavior, while AoA uses learning trajectories; these scores cannot all be interpreted as percentage accuracy. The final gains do not mean every component improves. Consult the original vectors for the complete changes.

The pinned public model documentation records SuperGLUE fine-tuning seed 42 and the source version of the AoA estimator. Model directories retain checkpoint records, environment records and original public evaluation summaries. This release preparation did not rerun the official evaluation or re-audit every item-level file.

- [Nine local endpoints](../results/training_strategy_comparison.csv)
- [Ten-model public comparison](../results/leaderboard_comparison.csv)
- [Frontier public evaluation](../models/frontier/EVALUATION.json)
- [Principle-guided public evaluation](../models/principle_guided/EVALUATION.json)
- [Full introduction to the evaluation framework](../reports/zh/chapters/01_background.tex)
