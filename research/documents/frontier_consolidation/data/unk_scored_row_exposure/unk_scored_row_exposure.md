# dual compliant tokenizer endpoint policy row-level spatial repair route status-tokenizer `<unk>` exposure

The spatial repair route status tokenizer is rule-compliant because it was fitted only on the allowed 10M reinvest pool. This report quantifies where its missing byte-level alphabet entries create `<unk>` tokens under the official strict decode/tokenization path. The byte-alphabet tokenizer is the only structural repair now under study; no benchmark-informed vocabulary tuning is allowed.

spatial repair route status tokenizer SHA: `91b775514b9f4e2d1f37c28445ab98181007e16d14f763955168547e293ee8f9`
Byte-alphabet tokenizer SHA: `b2b317e655a96f2c14eb559ce9fa7573f180904d84819cacfb5f61174cf355cf`

## Zero-shot official-decode families

| family | decoded rows | spatial repair route status rows with `<unk>` | spatial repair route status rows with target-span `<unk>` | target row rate | spatial repair route status `<unk>` tokens | ByteAlpha `<unk>` tokens | token ratio B/A |
|---|---:|---:|---:|---:|---:|---:|---:|
| BLiMP | 59875 | 0 | 0 | 0.000000 | 0 | 0 | 1.000089 |
| Supplement | 5218 | 473 | 473 | 0.090648 | 946 | 0 | 1.000868 |
| EWoK | 7618 | 0 | 0 | 0.000000 | 0 | 0 | 1.017346 |
| Entity | 6780 | 0 | 0 | 0.000000 | 0 | 0 | 1.000000 |
| COMPS | 91028 | 0 | 0 | 0.000000 | 0 | 0 | 1.000628 |
| GlobalPIQA_parallel | 103 | 0 | 0 | 0.000000 | 0 | 0 | 1.000880 |
| GlobalPIQA_nonparallel | 100 | 0 | 0 | 0.000000 | 0 | 0 | 1.001022 |

## Supplement task localization

| task | rows | rows with spatial repair route status `<unk>` | rows with target-span `<unk>` | target row rate | spatial repair route status `<unk>` tokens | target `<unk>` tokens | token ratio B/A |
|---|---:|---:|---:|---:|---:|---:|---:|
| hypernym | 842 | 0 | 0 | 0.000000 | 0 | 0 | 1.000307 |
| qa_congruence_easy | 64 | 64 | 64 | 1.000000 | 128 | 128 | 1.000000 |
| qa_congruence_tricky | 165 | 165 | 165 | 1.000000 | 330 | 330 | 1.000000 |
| subject_aux_inversion | 3867 | 0 | 0 | 0.000000 | 0 | 0 | 1.001103 |
| turn_taking | 280 | 244 | 244 | 0.871429 | 488 | 488 | 1.000000 |

For MLM zero-shot, tokens whose offsets overlap the completion span are individually masked and scored. In Supplement, the newline is part of the whole sentence/completion, so the affected rows above are direct target-span exposure, not only context exposure.

## SuperGLUE finetuning input exposure

SuperGLUE train/valid text rows scanned: 92959; rows with spatial repair route status `<unk>`: 129 (0.001388); spatial repair route status `<unk>` tokens: 523; byte-alphabet `<unk>` tokens: 0; token ratio B/A 1.000472.

| task.split | rows | rows with spatial repair route status `<unk>` | row rate | spatial repair route status `<unk>` tokens | token ratio B/A |
|---|---:|---:|---:|---:|---:|
| boolq.train | 9427 | 91 | 0.009653 | 458 | 1.000631 |
| qqp.valid | 20215 | 14 | 0.000693 | 28 | 1.000321 |
| boolq.valid | 1635 | 11 | 0.006728 | 22 | 1.000553 |
| qqp.train | 10000 | 5 | 0.000500 | 6 | 1.000330 |
| mnli.train | 10000 | 4 | 0.000400 | 4 | 1.000413 |
| mrpc.train | 3668 | 2 | 0.000545 | 3 | 1.000697 |
| wsc.train | 554 | 2 | 0.003610 | 2 | 1.000490 |
| mnli.valid | 4908 | 0 | 0.000000 | 0 | 1.000289 |
| mrpc.valid | 204 | 0 | 0.000000 | 0 | 1.000392 |
| multirc.train | 27243 | 0 | 0.000000 | 0 | 1.000454 |
| multirc.valid | 2424 | 0 | 0.000000 | 0 | 1.000562 |
| rte.train | 2490 | 0 | 0.000000 | 0 | 1.000507 |

## Reading/AoA input surface

Input strings scanned: 16430; spatial repair route status `<unk>` tokens: 44; byte-alphabet `<unk>` tokens: 0; token ratio B/A 1.000431.

## Interpretation

spatial repair route status remains a valid compliant endpoint and should not be discarded without its official score. The `<unk>` exposure is sharply localized: Supplement has 473/5218 rows with direct target-span `<unk>` tokens, because the completion equals the full string and contains dialogue newlines in QA/turn-taking; SuperGLUE exposure is sparse and mostly BoolQ/QQP input text; BLiMP/EWoK/Entity/COMPS/GlobalPIQA show no spatial repair route status `<unk>` under the official decode scan. The byte-alphabet repair removes the coverage defect with almost unchanged length geometry, but its newline/rare-byte tokens are structurally present rather than necessarily well-trained from the 10M stream. Therefore pristine official evaluation, not speculation, must decide between the two compliant endpoints if both finish without blocking resource priority.

Full JSON: `experiments/archive/frontier_consolidation/data/unk_scored_row_exposure/unk_scored_row_exposure.json`
