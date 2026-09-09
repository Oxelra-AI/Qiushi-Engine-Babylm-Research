# allocation balanced interpretation balanced allocation learner result

## Purpose

The earlier analysis allocation design asked whether faithful compact second views can preserve the base source-aligned signal while freeing processed-word budget for additional reviewed source-supported rows. allocation balanced interpretation repaired the unfair deterministic packing in the recurrence and current+aux schedules, then executed the learner comparison on six private-adapter training seeds.

This is a concentrated learner diagnostic on 17 base faithful-shortening rows and 7 additional reviewed rows, not a legal BabyLM endpoint.

## Schedule repair

Script: `experiments/archive/functional_learning/scripts/allocation_learner_comparison.py`.

The auxiliary schedule from earlier analysis was preserved. The source-identity-dependent packing choices were replaced by balanced seeded schedules:

- Recurrence control: each base source receives 9--11 extra compact presentations per seed (most seeds 10 or 10--11), instead of some receiving 37 and others none. Across six seeds, used extra words are 11,457--11,564 versus 11,791 target auxiliary words; 227--334 saved words are left unused rather than letting exact packing choose sources.
- Current+aux substitution: each base source drops 10--11 current presentations, leaving 69--70 current presentations per pair, instead of one source dropping to 33 presentations. The arm uses 102,915--103,020 row words versus 104,480 current-base capacity; 1,460--1,565 words remain unused to keep source exposure fair.

Schedule summaries: `experiments/archive/functional_learning/data/allocation_balanced_synthesis/balanced_schedule_summary.csv`.

## Runs and artifacts

Two run roots, six total seeds:

- `experiments/archive/functional_learning/data/allocation_balanced_comparison` with seeds 60001--60003.
- `experiments/archive/functional_learning/data/allocation_balanced_comparison_rep2` with seeds 60004--60006.

Combined synthesis:

- `experiments/archive/functional_learning/data/allocation_balanced_synthesis/summary.json`
- `combined_per_seed_arm_metrics.csv`
- `combined_pair_g_summary.csv`
- `combined_pair_contrasts.csv`
- `combined_source_follow_success.csv`

Core metrics use parent-relative common-target decomposition:

\[
  g=\frac{\Delta m_{\text{source-original}}+\Delta m_{\text{source-altered}}}{2},\quad
  b=\frac{\Delta m_{\text{source-original}}-\Delta m_{\text{source-altered}}}{2},
\]
with `p` as the no-source original-answer prior shift. Surface metrics are parent-relative masked NLL deltas; negative is better.

## Six-seed arm means

| arm | base common g | aux common g | base compact NLL Δ | base current NLL Δ | aux compact NLL Δ | aux current NLL Δ |
|---|---:|---:|---:|---:|---:|---:|
| current_base80 | 0.008 | 0.004 | -0.185 | -0.593 | -0.159 | -0.202 |
| current_aux_substitution | 0.020 | 0.082 | -0.157 | -0.506 | -0.609 | -0.314 |
| compact_base80_unspent | 0.130 | 0.060 | -0.550 | -0.199 | -0.118 | -0.075 |
| compact_interleaved_recurrence | 0.147 | 0.069 | -0.593 | -0.222 | -0.121 | -0.079 |
| compact_aux_support | 0.143 | 0.128 | -0.540 | -0.198 | -0.610 | -0.228 |

Main mean contrasts:

- `compact_aux_support - compact_base80_unspent`: base common g +0.0126; aux common g +0.0684; base compact surface NLL is 0.0107 worse, but aux compact surface NLL is 0.492 better.
- `compact_aux_support - compact_interleaved_recurrence`: base common g -0.0043; aux common g +0.0590; base compact surface NLL is 0.0528 worse; aux compact surface NLL is 0.490 better.
- `compact_aux_support - current_aux_substitution`: base common g +0.1226; aux common g +0.0465; base compact surface NLL is 0.382 better; base current surface NLL is 0.307 worse; aux compact surface NLL is essentially tied (-0.001 difference); aux current surface NLL is 0.0867 worse.

## Directional interpretation

The useful comparison is joint preservation plus additional learning.

1. **Compact+aux learns the auxiliary compact surface as well as current+aux while preserving compact-base/source-following behavior far better.** Aux compact NLL gains are nearly identical (`-0.6104` compact+aux vs `-0.6094` current+aux), but compact+aux has higher auxiliary common-source movement (`g=0.1282` vs `0.0817`) and much higher base common movement (`0.1427` vs `0.0201`).

2. **Current+aux substitution is not simply an equal alternative.** It learns auxiliary surfaces but base common g stays near current_base80 rather than the compact arms. It also shifts aux common margins asymmetrically: aux common `b=-0.1810` and `p=-0.1659`, so much of its auxiliary decision movement favors the altered answer/no-source direction. For current+aux, implied aux deltas are approximately Δsource-original = -0.099 and Δsource-altered = +0.263. For compact+aux, aux `b=-0.0693` and `p=-0.0940`, implying Δsource-original = +0.0588 and Δsource-altered = +0.1975. Compact+aux is therefore more symmetric source-supported movement on the auxiliary bank.

3. **Recurrence is a strong base-preservation control, not an auxiliary-learning substitute.** Compact interleaved recurrence has the best base compact surface NLL (`-0.5926`) and base common g (`0.1470`, essentially tied with compact+aux `0.1427`), but it does not learn auxiliary surfaces (`aux compact NLL -0.1208`) and has lower auxiliary common g (`0.0691`). Thus saved budget spent on old source IDs preserves base signal and improves old surfaces, while saved budget spent on reviewed auxiliary rows teaches new support.

4. **The auxiliary effect is not a single-row artifact, but the small bank remains uneven.** Compact+aux exceeds current+aux on auxiliary g for 5/7 source pairs and exceeds recurrence on 4/7. It is worse than current+aux on the gas-price and money-range pairs, with the money-range task dominating some current+aux advantage. Pair table: `combined_pair_contrasts.csv`.

5. **Base common movement in compact arms is still directionally uneven.** Compact arms improve base common g mainly with positive original-source movement; this continues the earlier analysis observation that compact adaptation is partly a stronger familiar/original-answer preference. This does not invalidate the allocation comparison, but it means this small result cannot yet be called a general source-use principle.

## Scientific status

The result supports the *mechanism-facing route* that correspondence-preserving shorter views can preserve a compact/source-conditioned base signal and let the saved budget teach extra reviewed support better than pure recurrence and better, in common-source directions, than current-view substitution. It is a real positive learner result for the Stage III compact-and-reinvestment hypothesis.

It is not sufficient for `Qiushi-BabyLM-36M-Strict-Small-v5` by itself. The next step should scale this into a more realistic legal-stream candidate only if the selection/review/admission policy can produce a large enough verified pool with the same separation of policies: faithful shortening, supported partial-view allocation, and correspondence repair. The model-improvement route should preserve original views/source IDs as controls and include an early broad preservation readout before investing in full official evaluation.
