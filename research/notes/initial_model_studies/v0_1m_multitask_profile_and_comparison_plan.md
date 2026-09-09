# v0 1m multitask profile and comparison plan — V0 1M Multi-task Profile and Same-exposure Mechanism Comparison Plan

Active goal: BabyLM Strict-Small official Overall SOTA under strict 2026 constraints.

## 1. Why this step was run

The clean 1M V0 dense causal pilot from v0 1m pipeline and first signal proved the official-compatible lifecycle but only had BLiMP fast evidence. Weak tasks must **not** be treated as causal evidence for sparse routing or used to select V1 prematurely. This step therefore built a broader official-compatible 1M phenotype for the existing V0 checkpoint, then connected it to a fair same-exposure comparison plan among mechanism-distinct candidates.

The model profiled here is not a SOTA candidate:

- run dir: `experiments/archive/initial_model_studies/training/runs/babylm_pilot_dense_v0_1M`
- model: dense GPT2LMHeadModel-style causal decoder
- parameters: 7,419,392
- tokenizer: official 16,384-token baseline tokenizer, saved portably as `PreTrainedTokenizerFast`
- training exposure: exactly 1,000,000 official whitespace words
- official corpus revision: `BabyLM-community/BabyLM-2026-Strict-Small` at `c92ab16b4f08858304b0815706065b3354d8fc0a`
- clean HF load evidence: `checkpoint_verify.json`

## 2. Official-compatible 1M V0 profile

Evaluation repository:

`experiments/archive/initial_model_studies/repos/babylm-eval/strict`

Model path used for most profile runs:

`experiments/archive/initial_model_studies/training/runs/babylm_pilot_dense_v0_1M/hf_model` with `revision_name chck_1M`

### Scores and report paths

| benchmark/profile item | score | report path | interpretation |
|---|---:|---|---|
| BLiMP fast | 53.63 | `runs/babylm_pilot_dense_v0_1M/eval_results_blimp_fast_1M/chck_1M/chck_1M/zero_shot/causal/blimp/blimp_fast/best_temperature_report.txt` | Above chance after 1M words; the dense causal scaffold learns some syntax/semantic regularities quickly. |
| BLiMP Supplement fast | 48.00 | `runs/babylm_pilot_dense_v0_1M/eval_results_fast_profile_1M/hf_model/chck_1M/zero_shot/causal/blimp/supplement_fast/best_temperature_report.txt` | Below chance/weak on supplemental generalization tasks; not yet robust linguistic competence. |
| EWoK fast | 48.36 | `runs/babylm_pilot_dense_v0_1M/eval_results_fast_profile_1M/hf_model/chck_1M/zero_shot/causal/ewok/ewok_fast/best_temperature_report.txt` | Weak world-knowledge/commonsense discrimination at 1M. The EWoK zip had to be unpacked from nested `evaluation_data/fast_eval/evaluation_data/fast_eval/ewok_fast`. |
| Entity Tracking fast | 17.83 | `runs/babylm_pilot_dense_v0_1M/eval_results_fast_profile_1M/hf_model/chck_1M/zero_shot/causal/entity_tracking/entity_tracking_fast/best_temperature_report.txt` | Severe weakness; far below balanced-choice intuition. This suggests the tiny dense causal model is not tracking discourse state at 1M, but does **not** prove sparse routing is the right fix. |
| COMPS full local | 50.38 | `runs/babylm_pilot_dense_v0_1M/eval_results_fast_profile_1M/hf_model/chck_1M/zero_shot/causal/comps/comps/best_temperature_report.txt` | Near chance on compositional/property tasks at 1M; useful as a morphology/compositionality probe. |
| Reading eye-tracking score | 9.64 | `runs/babylm_pilot_dense_v0_1M/eval_results_fast_profile_1M/hf_model/chck_1M/zero_shot/causal/reading/report.txt` | Positive early reading signal; should be compared across mechanisms because 2026 Overall gives Reading one of nine columns. |
| Reading self-paced score | 2.65 | same as above | Small positive self-paced signal. |

Machine-readable summary before EWoK/COMPS was saved at:

`experiments/archive/initial_model_studies/training/runs/babylm_pilot_dense_v0_1M/fast_profile_1M_summary.json`

This note supersedes that summary by adding EWoK fast and COMPS.

## 3. What this profile does and does not show

The profile shows a 1M-word dense causal phenotype:

- BLiMP fast rises above chance by 1M words.
- Supplement, EWoK, and COMPS remain near chance.
- Entity tracking is extremely weak.
- Reading has small-to-moderate positive signal.

It does **not** identify the cause of the profile. Several explanations remain live:

1. **Exposure/scale**: 1M words and 7.4M parameters may simply be too early/small for entity, EWoK, and COMPS; a denser or longer V0 could improve without new mechanisms.
2. **Training-objective mismatch**: pure next-token prediction may learn local syntax earlier than discourse state or word-level developmental trajectories.
3. **Representation bottleneck**: a 16k BPE stream may not expose morphology/entity features efficiently under low data.
4. **Memory/state bottleneck**: shallow dense attention at context 256 may be weak at entity state maintenance.
5. **Optimization/data-order effect**: the simple one-pass deterministic word selection/shuffle may bias early competence toward high-frequency local distributional patterns.

Therefore this evidence should guide fair comparisons, not justify immediately locking onto sparse routing.

## 4. Mechanism-distinct candidates for the next 1M comparisons

All candidates below must use the same official corpus revision, same 1M word exposure, same tokenizer source unless the candidate explicitly tests tokenizer/representation, same checkpoint naming, same verifier, and the same fast profile tasks: BLiMP fast, BLiMP Supplement fast, EWoK fast, Entity Tracking fast, COMPS, Reading.

### Candidate A — stronger dense causal control

Purpose: distinguish true mechanism benefit from underpowered V0.

- Same training script/interface.
- Increase model toward a still-cheap baseline: e.g. 8 layers, width 384, 6 heads, context 512 if memory/time permits, or train the current 7.4M model to 10M exposure for curve evidence.
- Scientific use: if entity/reading/EWoK improve substantially with dense scale/exposure alone, sparse routing is less urgent; if BLiMP improves but entity remains collapsed, state/representation mechanisms become more important.

### Candidate B — sparse-routing causal model

Purpose: test whether path diversity improves entity/reading/developmental-like trajectories at equal exposure.

- MoEP-like routing or simpler routed parallel blocks, but keep HF causal interface.
- Required logs: router load distribution, route entropy, token-frequency vs route assignment, and route distribution for entity/closed-class/open-class tokens.
- Scientific use: improvement in entity/reading at same BLiMP/EWoK would support the route-diversity hypothesis. Weak or noisy improvement would not be enough without controls.

### Candidate C — recurrent/stateful or memory-light causal model

Purpose: test whether the entity collapse is mainly a state-maintenance problem rather than an expert-path problem.

- Compact recurrent/linear-recurrent block or gated memory channel inside a causal LM interface.
- Keep parameter count comparable to sparse candidate.
- Scientific use: if this candidate improves entity tracking more cleanly than sparse routing, the main mechanism should shift toward state/memory rather than MoE-style path diversity.

### Candidate D — multi-granularity token/morphology side channel

Purpose: test whether COMPS/entity/AoA-related weaknesses arise from BPE representation under low data.

- Character/byte CNN over token strings gated into token embeddings; no external morphology model.
- Same tokenizer IDs and causal output head, with side information learned only from the official 10M text/token strings.
- Scientific use: improvements on COMPS and entity without BLiMP loss would support representation reconstruction rather than routing/objective changes.

### Candidate E — low-weight difficulty-aware auxiliary objective

Purpose: inherit AMLM's hard-token focus without causing the human-like collapse seen in earlier literature.

- Add a small training-only denoising/span or token-difficulty auxiliary loss from token-level causal loss EMA, not from eval data.
- Keep the causal head and official scoring unchanged.
- Use a low auxiliary weight and log difficulty distribution by token frequency/POS proxy if possible.
- Scientific use: if BLiMP Supplement/EWoK/COMPS improve while Reading/entity remain stable, the route is promising; if Reading/entity collapse, the AMLM tradeoff is reproduced.

## 5. Near-term experimental matrix

Do **not** launch a 100M run yet. The next matrix should compare mechanism classes under a common cheap setting:

| run | model/mechanism | exposure | key comparison | decision-relevant profiles |
|---|---|---:|---|---|
| V0-1M existing | tiny dense causal | 1M | reference phenotype | BLiMP 53.63, supplement 48.00, EWoK 48.36, entity 17.83, COMPS 50.38, reading 9.64/2.65 |
| V0-scale-1M | stronger dense causal | 1M | underpowered-control | Does entity/reading improve without new mechanism? |
| V1-sparse-1M | sparse-routing causal | 1M | path-diversity mechanism | Entity/reading vs BLiMP/EWoK tradeoff; router logs required. |
| V2-state-1M | recurrent/stateful causal | 1M | state-maintenance mechanism | Entity tracking and reading vs syntax. |
| V3-morph-1M | multi-granularity side channel | 1M | representation mechanism | COMPS/entity/supplement without BLiMP loss. |
| V4-aux-1M | dense or sparse + low-weight difficulty auxiliary | 1M | objective mechanism | Supplement/EWoK/COMPS gains without human-like collapse. |

The proposed comparison should select two or three candidates that are maximally mechanism-distinct and fast to implement under the existing V0 trainer framework, likely: stronger dense control, sparse-routing causal, and either recurrent/stateful or morph side-channel. The comparison should be official-eval driven.

## 6. Next experiment

The next work should turn this comparison plan into runnable code/configs under `training/scripts/`, preserving the existing clean HF save/load path. The sparse route remains a serious candidate, but not the default conclusion. It must win under equal 1M exposure against at least a stronger dense control and one different mechanism before becoming the main scientific line.
