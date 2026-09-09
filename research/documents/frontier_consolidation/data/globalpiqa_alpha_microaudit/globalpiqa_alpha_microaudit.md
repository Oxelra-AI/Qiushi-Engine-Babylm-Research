# private scale endpoint vs mechanism synthesis GlobalPIQA alpha micro-audit

Status: **COMPLETE**
Common GlobalPIQA examples: `203`; alpha-sensitive examples: `5`.

## Alpha scores

| alpha | correct | score | net items vs anchor | score delta | cheap7 contribution |
|---|---:|---:|---:|---:|---:|
| a0_anchor | 76 | 37.438424 | +0 | +0.000000 | +0.000000 |
| a0p5 | 79 | 38.916256 | +3 | +1.477833 | +0.211119 |
| a0p75 | 78 | 38.423645 | +2 | +0.985222 | +0.140746 |
| a1 | 77 | 37.931034 | +1 | +0.492611 | +0.070373 |

## Changed examples

| item | pattern | gold | alpha predictions |
|---|---|---|---|
| GlobalPIQA:GlobalPIQA_nonparallel:group0123_ex000054_eng_latn_0_v1 | 0111 | wait for the edges to brown and the top to bubble. Slide the spatula under the pancake so it's centered. Quickly flip yo | a0_anchor:✗; a0p5:✓; a0p75:✓; a1:✓ |
| GlobalPIQA:GlobalPIQA_nonparallel:group0123_ex000082_eng_latn_0_v1 | 1110 | Let it sit before cutting into it. | a0_anchor:✓; a0p5:✓; a0p75:✓; a1:✗ |
| GlobalPIQA:GlobalPIQA_parallel:parallel_ex000018_eng_latn | 0111 | Watching a movie | a0_anchor:✗; a0p5:✓; a0p75:✓; a1:✓ |
| GlobalPIQA:GlobalPIQA_parallel:parallel_ex000039_eng_latn | 0111 | Chew thoroughly then swallow | a0_anchor:✗; a0p5:✓; a0p75:✓; a1:✓ |
| GlobalPIQA:GlobalPIQA_parallel:parallel_ex000071_eng_latn | 1100 | The wick should be longer than the height of the candle wax | a0_anchor:✓; a0p5:✓; a0p75:✗; a1:✗ |

## Scientific reading

- GlobalPIQA has 203 common examples; only 5 change across alpha0/0.5/0.75/1.0.
- Alpha0.5 GlobalPIQA gain is 3 net examples = +1.477833 score points = +0.211119 cheap7 points.
- Alpha0.75 GlobalPIQA gain is 2 net examples = +0.985222 score points = +0.140746 cheap7 points.
- This supports the interpretation that GlobalPIQA-driven alpha ranking is a few-example endpoint effect, not broad commonsense acquisition.

JSON: `experiments/archive/frontier_consolidation/data/globalpiqa_alpha_microaudit/globalpiqa_alpha_microaudit.json`
