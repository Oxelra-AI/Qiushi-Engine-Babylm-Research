# earlier analysis RoBERTa transfer readout

Status: `ROBERTA_TRANSFER_READOUT_COMPLETE`

The inherited 1x RoBERTa endpoint has Entity V-R -0.410, too small/negative to resolve the high-dose DeBERTa carrier. Existing MAX RoBERTa view-clean Entity deltas are available for 2 checkpoints with mean -1.105; this is total fixed-budget transfer, not semantic isolation. MAX RoBERTa view-repeat Entity common-window delta is now +0.510; compare this with DeBERTa MAX +2.0625 and RoBERTa 1x -0.410.

## Key quantities

- Low-dose RoBERTa 1x endpoint Entity V-R: `-0.41000000000000014`
- MAX RoBERTa semantic Entity common-window V-R: `0.509999999999998`
- MAX minus low-dose Entity delta: `0.9199999999999982`
- DeBERTa reference: first-basin MAX Entity common-window V-R `+2.0625`; 1x `+0.55875`.

## Missing MAX view-repeat rows

- chck_20M: view_present=True repeat_present=False
- chck_30M: view_present=True repeat_present=False
- chck_40M: view_present=True repeat_present=False
- chck_50M: view_present=True repeat_present=False
- chck_60M: view_present=True repeat_present=False
- chck_70M: view_present=True repeat_present=False

## Files

- contrast_rows_csv: `experiments/archive/frontier_consolidation/data/roberta_transfer_after_late_entity/roberta_transfer_contrast_rows.csv`
- summary_rows_csv: `experiments/archive/frontier_consolidation/data/roberta_transfer_after_late_entity/roberta_transfer_summary_rows.csv`
- summary_json: `experiments/archive/frontier_consolidation/data/roberta_transfer_after_late_entity/roberta_transfer_readout_summary.json`
- summary_md: `research/documents/frontier_consolidation/data/roberta_transfer_after_late_entity/roberta_transfer_readout_summary.md`
