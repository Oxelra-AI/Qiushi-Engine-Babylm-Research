# mlm rtd gdes 20m screen plan — MLM+RTD-GDES 20M result and delayed-tail design

## What was run

A bounded from-scratch MLM+RTD-GDES training screen was launched from random initialization on the protected legal compact-view-reinvest stream.

Run: `training/runs/mlm_rtd_lambda1_seed43022_20M`

Configuration:

- DeBERTa-v2 masked LM, 8×480, 34,467,424 MLM parameters;
- RTD head: 231,842 parameters;
- spatial repair route status compliant tokenizer, same 100M JSONL stream prefix, same seed trio 43/43022/43023, batch 256, seq256, AdamW betas (0.9,0.98), weight decay 0.01, LR 0.001, warmup fraction 0.06, 100M schedule horizon (`lr_total_steps=2529`), fixed WWM p=0.15;
- RTD self-corruptions sampled from detached MLM logits at masked positions with T=1.0;
- class-balanced RTD CE;
- GDES restores MLM-only gradients on `deberta.embeddings.word_embeddings.weight` and `deberta.encoder.rel_embeddings.weight`;
- trained for 20,000,000 counted words, 506 steps; checkpoints at 5M/10M/15M/20M.

Training metrics: `training/runs/mlm_rtd_lambda1_seed43022_20M/scientific_metrics.json`.

The first MLM loss exactly matched the spatial repair route status legal baseline first loss: 9.837543487548828, validating the random initialization, first batch, masking path, and MLM loss before adding RTD. RTD loss decreased from 0.6960 to 0.5505. Matched-step MLM loss against spatial repair route status remained essentially unchanged: mean delta over 506 steps -0.001946; last-50 mean +0.002963. Thus RTD did not grossly damage scalar MLM optimization.

## Cheap official-compatible evaluation

Evaluation file: `data/mlm_rtd_20M_eval/per_target/mlm_rtd_lambda1_seed43022_20M.json`.
Summary/anatomy: `data/mlm_rtd_20M_summary/mlm_rtd_20M_summary.{json,md}` and `data/mlm_rtd_20M_anatomy/mlm_rtd_20M_anatomy.{json,md}`.

Primary comparator: spatial repair route status legal 20M, `data/legal20m_treatment_effect_eval/per_target/complianttok_reinvest_seed43022_20M.json`.

| column | baseline20M | MLM+RTD20M | delta |
|---|---:|---:|---:|
| BLiMP | 59.6900 | 60.1600 | +0.4700 |
| Supplement | 55.4500 | 56.9400 | +1.4900 |
| EWoK | 50.7300 | 50.0700 | -0.6600 |
| Entity | 18.6500 | 19.3500 | +0.7000 |
| COMPS | 50.2600 | 50.2400 | -0.0200 |
| GlobalPIQA_parallel | 20.3900 | 18.4500 | -1.9400 |
| GlobalPIQA_nonparallel | 48.0000 | 48.0000 | +0.0000 |
| GlobalPIQA | 34.1950 | 33.2250 | -0.9700 |
| Reading | 8.6700 | 8.4350 | -0.2350 |
| cheap7 | 39.6636 | 39.7743 | +0.1107 |

The screen did **not** meet the continuation signal of a broad cheap7 rise near +0.35 without fragile-column damage. It should not be extended unchanged to a 100M from-scratch endpoint.

## Behavioral interpretation

The result is not a simple failure to optimize MLM. It is an objective-induced rotation of ranking behavior:

- Positive: Supplement improves mainly through QA congruence (`qa_congruence_easy` +4.69, `qa_congruence_tricky` +3.03), Entity improves in `regular` and `move_contents`, BLiMP improves syntax/syntax-semantics and binding (`binding` +5.62).
- Negative: EWoK aggregate falls through social/material/physical dynamics and negation (`material-dynamics` -7.27, `physical-dynamics` -5.00, `social-properties` -7.93, `negation` -10.53); GlobalPIQA_parallel falls -1.94; Reading falls -0.235; BLiMP semantics falls -3.65 and NPI licensing -4.58.

This pattern is compatible with RTD encouraging local context-token compatibility and some structural regularity while disturbing semantic/polarity/event-dynamics MLM margins, which are read by official evaluation through pseudo-likelihood.

## independent_review scientific reading

independent_review verifier result: .

Main points:

- The from-scratch fixed-λ=1 GDES intervention should not proceed unchanged to 100M.
- RTD itself should not yet be closed, because the positive mechanism probe was on a mature 80M encoder, while the negative/mixed score screen was from initialization.
- A delayed RTD continuation from an existing 80M checkpoint tests a distinct hypothesis: RTD may be a consolidation signal after MLM lexical/semantic likelihood structure exists, but harmful or redistributive during early acquisition.
- A delayed test must be paired with an MLM-only continuation control and should be dose-calibrated; λ=1 already rotates gradients by a nontrivial amount even with small positive trunk cosine.

## Delayed-tail experiment design, if pursued

The next GPU work should not be another automatic same-route continuation. It should be a sharply paired delayed-consolidation test.

### Scientific question

Does hard, model-sampled RTD-GDES improve mature contextual compatibility and official-compatible cheap behavior when applied only after the protected compact-view-reinvest model has already reached 80M words, without the EWoK/GlobalPIQA/Reading damage observed when RTD is active from initialization?

### Minimal reliable design

Use the spatial repair route status legal 80M checkpoint:

`training/runs/complianttok_reinvest_seed43022_r2/hf_model/chck_80M`

Use the exact post-80M row segment from the frozen 100M stream, starting at row `2024 * 256 = 518144`, as in tail rephase probe and experience utilization route original-tail replay. tail rephase probe and experience utilization route records: `PARENT_ACTUAL_WORDS=80,034,368`; tail to 100M uses rows 518144 onward.

Preferred paired design if two H100s are free:

1. **Tail-MLM control, 80→90M**: load chck_80M, discard optimizer state, train fixed WWM MLM only on the exact original tail rows, same train RNG, matched low continuation LR or chosen continuation schedule.
2. **Tail-RTD-GDES, 80→90M**: identical to control except add RTD-GDES. Use λ below the from-scratch λ=1 dose, ideally λ≈0.25–0.5 or a norm-targeted setting, because the earlier analysis mature probe found RTD/MLM trunk norm ratio 0.241 at λ=1 and independent_review estimated a substantial update rotation.

Score both at chck_85M and chck_90M if available; if evaluation budget must be minimal, score chck_90M only first. Use cheap official-compatible columns only. Continue to 100M only if the paired RTD-minus-MLM tail delta is broad and avoids EWoK/Reading/GlobalPIQA damage.

### Relation to existing tail rephase probe and experience utilization route control

There is an existing original-tail replay control:

`training/runs/original_tail_replay_matched_lr_seed43044`

Its evaluated chck_90M file is:

`data/tail_replay_eval/per_target/tail_replay_matched_lr_chck_90M.json`

Scores: BLiMP 65.84, Supplement 61.24, EWoK 50.53, Entity 28.13, COMPS 51.66, GlobalPIQA 36.065, Reading 8.17.

This can be used as a provisional control if and only if the tail RTD trainer exactly matches its choices: chck_80M load, row_start 518144, train_rng_seed 43044, matched-LR schedule, no optimizer state, exact row dataset, fixed WWM p=0.15, batch256, seq256. A cleaner comparison is to run a new paired MLM-only tail alongside tail RTD if resources allow, because the delayed-RTD result should not inherit interpretive ambiguity from an old optimizer-state/rephase experiment.

### Required trainer changes

A tail RTD trainer should be derived from `scripts/original_tail_replay_trainer.py` plus the earlier analysis RTD/GDES code, not from the from-scratch earlier analysis trainer. It must:

- load `DebertaV2ForMaskedLM.from_pretrained(chck_80M)`;
- load tokenizer from the same checkpoint;
- use `load_tail_examples(train_file, row_start=518144)`;
- save total-exposure checkpoints named chck_85M/chck_90M (or chck_85M/90M/95M/100M if running full tail);
- record tail metadata, first/last rows, total words, actual parent exposure, and schedule;
- record RTD replacement statistics, RTD loss, MLM loss, and, if possible, occasional trunk RTD/MLM norm ratio/cosine on the actual continuation batch;
- protect word embeddings and relative embeddings exactly as in earlier analysis unless deliberately testing an ablation.

### Stop/continue interpretation

Continue beyond 90M only if RTD-minus-MLM tail cheap7 is meaningfully positive and broad, with no recurrence of the from-scratch damage in EWoK dynamics/negation, GlobalPIQA_parallel, and Reading. If RTD discrimination improves while these official columns worsen, the route should be closed as an objective mismatch: the auxiliary is solving its own task while harming the MLM likelihood margins that BabyLM evaluates.

## Current route status

- From-scratch λ=1 MLM+RTD-GDES: not a 100M continuation route.
- RTD-GDES as mature delayed consolidation: open, but requires a paired, dose-calibrated tail experiment before any endpoint commitment.
- No final packaging, full official evaluation, or submission step is justified.
