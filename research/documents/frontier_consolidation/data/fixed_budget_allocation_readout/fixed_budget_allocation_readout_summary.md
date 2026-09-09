# allocation specificity state fixed-budget allocation readout

Use V-C for fixed-budget allocation, V-B for compact re-expression specificity, B-R for same-population breadth versus duplicate recurrence, and the exact V-R=(V-B)+(B-R) identity to avoid treating Entity aggregate movement as a single mechanism.

Score rows present: 342; missing score rows: 258.
Contrast rows present: 460; missing contrast rows: 420.
Triangle rows present: 82; broad triangle rows: 16.

## Selected quantities
- deberta_basin1_VminusB exEntity5 late80_100: n=2 mean=+0.1065 range=[+0.0830, +0.1300] checkpoints=chck_90M;chck_100M
- deberta_basin1_BminusR exEntity5 late80_100: n=2 mean=+0.2565 range=[+0.2500, +0.2630] checkpoints=chck_90M;chck_100M
- deberta_basin1_VminusR exEntity5 late80_100: n=2 mean=+0.3630 range=[+0.3330, +0.3930] checkpoints=chck_90M;chck_100M
- deberta_basin1_VminusB cheap6_no_GlobalPIQA late80_100: n=2 mean=+0.5787 range=[+0.5750, +0.5825] checkpoints=chck_90M;chck_100M
- deberta_basin1_BminusR cheap6_no_GlobalPIQA late80_100: n=2 mean=+0.3529 range=[+0.3050, +0.4008] checkpoints=chck_90M;chck_100M
- roberta_VminusC_maxgeom exEntity5 late80_100: n=2 mean=-0.7210 range=[-0.7430, -0.6990] checkpoints=chck_90M;chck_100M
- roberta_VminusC_maxgeom cheap6_no_GlobalPIQA late80_100: n=2 mean=-0.5933 range=[-0.6008, -0.5858] checkpoints=chck_90M;chck_100M
- deberta_basin1_VminusC_maxgeom exEntity5 late80_100: n=2 mean=+0.2990 range=[+0.2380, +0.3600] checkpoints=chck_90M;chck_100M
- deberta_basin2_VminusC_maxgeom exEntity5 late80_100: not yet complete

Triangle rows now complete for selected quantities:
- chck_80M Entity: V-R=+4.5000, V-B=+3.1000, B-R=+1.4000, residual=+0.0000
- chck_80M cheap6_no_GlobalPIQA: V-R=+1.0225, V-B=+0.4650, B-R=+0.5575, residual=+0.0000
- chck_80M exEntity5: V-R=+0.3270, V-B=-0.0620, B-R=+0.3890, residual=+0.0000
- chck_90M Entity: V-R=+3.6600, V-B=+3.0800, B-R=+0.5800, residual=+0.0000
- chck_90M cheap6_no_GlobalPIQA: V-R=+0.8875, V-B=+0.5825, B-R=+0.3050, residual=+0.0000
- chck_90M exEntity5: V-R=+0.3330, V-B=+0.0830, B-R=+0.2500, residual=+0.0000
- chck_100M Entity: V-R=+3.8900, V-B=+2.8000, B-R=+1.0900, residual=+0.0000
- chck_100M cheap6_no_GlobalPIQA: V-R=+0.9758, V-B=+0.5750, B-R=+0.4008, residual=+0.0000
- chck_100M exEntity5: V-R=+0.3930, V-B=+0.1300, B-R=+0.2630, residual=+0.0000

## Files
- score_rows_csv: `experiments/archive/frontier_consolidation/data/fixed_budget_allocation_readout/score_rows.csv`
- missing_score_rows_csv: `experiments/archive/frontier_consolidation/data/fixed_budget_allocation_readout/missing_score_rows.csv`
- contrast_rows_csv: `experiments/archive/frontier_consolidation/data/fixed_budget_allocation_readout/contrast_rows.csv`
- missing_contrast_rows_csv: `experiments/archive/frontier_consolidation/data/fixed_budget_allocation_readout/missing_contrast_rows.csv`
- triangle_rows_csv: `experiments/archive/frontier_consolidation/data/fixed_budget_allocation_readout/triangle_rows.csv`
- summary_rows_csv: `experiments/archive/frontier_consolidation/data/fixed_budget_allocation_readout/summary_rows.csv`
- summary_json: `experiments/archive/frontier_consolidation/data/fixed_budget_allocation_readout/fixed_budget_allocation_readout_summary.json`
- summary_md: `research/documents/frontier_consolidation/data/fixed_budget_allocation_readout/fixed_budget_allocation_readout_summary.md`
