# chunk stream preflight — Experience-utilization experiment design

CPU-only design derived from the sequence curriculum loop measurement direct loop measurement. No model was trained or evaluated.

## Mechanism

The current prefix length path charges words that the model never receives. A useful next experiment should therefore read *experience utilization*: whether the model benefits when every charged word becomes visible and eligible for prediction. A short-to-long sequence order is a second factor, not the same factor.

## Clean comparisons

1. Current row256 -> U256: fixed 256-token length in both; U256 uses word-boundary chunks to reveal suffix words that current row truncation hides. This reads recovered charged-word visibility without short-to-long order.
2. U256 -> U64_128_256: both expose every tokenized word once per epoch. With shared mask probability and matched stage-reset updates, total active tokens and expected target tokens are the same; the remaining change is local context/order/packing.
3. Target-matched variants reduce the chunked-arm mask probability so expected masked-target count equals the current row256 baseline; these are interpretation controls if a first wave improves.

## legal40k

- Current row256 hides 1,286,640 of 100M charged words (1.2866%) and has 137,061,620 active tokens at mask 0.15.
- U256 chunking exposes all charged words, has 139,426,440 active tokens (1.0173x current), and can match current expected targets with mask probability 0.147456.
- U64_128_256 after chunking has the same active-token total as U256 (139,426,440) and the same 2,530 stage-reset updates; this separates ordering from visibility recovery.
- The broken prefix 64x3_128x4_256x3 path would use only 86,639,145 active tokens (0.6321x current row256) and hide 36.86% of charged word opportunities across the schedule.

## minfreq25

- Current row256 hides 1,448,730 of 100M charged words (1.4487%) and has 138,713,400 active tokens at mask 0.15.
- U256 chunking exposes all charged words, has 141,385,120 active tokens (1.0193x current), and can match current expected targets with mask probability 0.147165.
- U64_128_256 after chunking has the same active-token total as U256 (141,385,120) and the same 2,530 stage-reset updates; this separates ordering from visibility recovery.
- The broken prefix 64x3_128x4_256x3 path would use only 87,141,062 active tokens (0.6282x current row256) and hide 37.38% of charged word opportunities across the schedule.

## Expensive-run use

Do not launch these arms while the legal40k 12x384 depth training is pending. If depth and support-floor evidence still leave the SOTA gap, the clean first wave is U256_chunked_visibility_mask015 versus U64_128_256_chunked_order_mask015 on the same tokenizer/backbone seed: the pair distinguishes recovered experience utilization from length ordering while preserving charged words and target-token totals between the two chunked arms.

JSON: `experiments/archive/representation_and_objectives/data/experience_utilization_design/experience_utilization_experiment_design.json`
CSV: `experiments/archive/representation_and_objectives/data/experience_utilization_design/experience_utilization_arm_summary.csv`, `experiments/archive/representation_and_objectives/data/experience_utilization_design/experience_utilization_batch_plan.csv`
