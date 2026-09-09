# matched support source contrast result result: source-specific decomposition of the synthetic bridge

## Why this step was needed

identity shortcut bridge result v2 showed that exact repetition could create large absolute damage on the unpracticed content/restatement probe, but the subsequent audit showed that the source-specific decomposition did not match the BabyLM relation-learning term: in identity shortcut bridge result the unrelated-source/no-match control was damaged even more than the true-source probe, so the model still used the true source better rather than showing an excess true-source cost. matched support source contrast result therefore repaired train/probe support before any multi-operation extension.

## Experiments executed

1. `matched_support_source_contrast.py`: same attribute token for copy and content templates, but train and probes share source-slot distribution, target-slot distribution, and one matched plus one unmatched final event per relation-arm sequence. `repeat_full`/`repeat_masked` and `varied_full`/`varied_masked` are paired-token-identical; the mask blocks only target-event attention to its source event during training.

2. `revision_008b_nonoverlap_source_contrast.py`: same support and attention intervention, but the content/rewrite target uses a disjoint token `r_i` while the source contains `s_i`. Exact repetition trains `s_i→s_i`; varied restatement trains `s_i→r_i`. This is the closer analogue of relation_learning's tokenizer-nonoverlap rewrite probe.

## Main numerical results

Positive excess true-source cost is `(T_A-T_B)-(U_A-U_B)`, so it means A is worse on true-source use beyond its unrelated-source/no-match change.

| experiment | contrast | T delta | U delta | excess true-source cost | gain delta | reading |
|---|---|---:|---:|---:|---:|---|
| same-token content | R_full − support | +0.040±0.069 | +0.383±0.028 | -0.343±0.061 | +0.343±0.061 | true source helps more than unrelated control; not BabyLM-like recurrence cost |
| nonoverlap rewrite | R_full − support | +1.827±0.779 | +1.881±0.762 | -0.054±0.045 | +0.054±0.045 | large absolute rewrite harm, but T and U move together; not source-specific |
| nonoverlap rewrite | R_full − R_masked | +1.350±1.388 | +1.376±1.354 | -0.026±0.034 | +0.026±0.034 | attention access changes absolute NLL but not selective true-source cost |
| nonoverlap rewrite | V_full − support | -1.862±0.696 | +0.166±0.150 | -2.028±0.839 | +2.028±0.839 | varied restatement gives strong true-source rewrite use |
| nonoverlap rewrite | R_full − V_full | +3.689±0.147 | +1.715±0.899 | +1.974±0.832 | -1.974±0.832 | R is much worse than V on true-source rewrite; this matches V−R ordering but not R−C |

## What changed scientifically

The repaired support tests do not support treating identity shortcut bridge result's absolute content damage as the same source-specific recurrence cost measured by relation_learning. In the same-token version, exact repetition actually improves source-conditioned content relative to the unrelated control (R_full − support excess ≈ −0.34). This exposed that the earlier content target could be solved by copying the source attribute token.

The nonoverlap version removes that shortcut. Exact repetition then causes large absolute rewrite-token damage relative to clean/support controls: R_full − support has T≈+1.83 nats and U≈+1.88 nats. But the excess true-source cost remains near zero and slightly negative (≈−0.054±0.045). Thus the source is not being used selectively worse than the unrelated control; both true-source and no-match rewrite targets are broadly disfavored.

The paired attention intervention remains important but its meaning is narrower. In nonoverlap rewrite, R_full − R_masked shifts content T and U by similarly large amounts (T≈+1.35, U≈+1.38, excess≈−0.026), so within-context attention access can install a broad final-slot/target-space expectation that damages rewrite tokens, but it does not by itself produce the BabyLM-like positive excess true-source cost. For varied restatement, attention access is necessary for strong rewrite use: V_full − V_masked has excess≈−2.03 and content gain≈+2.03.

The most robust synthetic result after repair is therefore relational support specificity rather than direct BabyLM recurrence-cost identity: finite experience installs the relation it demonstrates. Repetition installs `same entity → same source token` and varied restatement installs `same entity + source token → disjoint rewrite token`. When evaluation asks for the other relation, damage can be large, but the T/U decomposition distinguishes broad target-prior damage from source-specific content competition.

## Learning-curve scan

Across stored eval epochs, the nonoverlap R_full − support content excess did not show a large hidden positive phase. Final per-seed excesses are roughly −0.011, −0.101, and −0.051. R_full − R_masked final excesses are roughly −0.007, −0.066, and −0.006. Seed-level curves occasionally fluctuate, but the repeated pattern is not a +0.4 to +1.0 nat positive excess comparable to the BabyLM DeBERTa term. The exact curve values are saved in `data/source_contrast_synthesis/source_contrast_synthesis.json`.

## Implication for the next research step

Do not extend this exact substrate to Entity-depth state tracking as though the BabyLM source-specific mechanism has already been reproduced. A depth extension could still test how relation-specific training affects multi-update tracking, but the bridge to the BabyLM nonoverlap R−C cost needs a sharper substrate first: the clean/control arm must share broad final-slot and target-token distribution with repetition while lacking only the repeated identity relation, and the probe must preserve disjoint target tokens while preventing broad rewrite-token suppression from masquerading as source-use impairment. A useful next construction is a counterbalanced dual-relation corpus where each arm sees the same marginal rates of copy tokens and rewrite tokens at matched and unmatched final slots, while only the joint source→target pairing differs (identity-paired vs rewrite-paired vs independent-paired).

## Files

- overlap script: `scripts/matched_support_source_contrast.py`

- overlap data: `data/matched_support_s42_43_100_e300/`

- nonoverlap script: `scripts/revision_008b_nonoverlap_source_contrast.py`

- nonoverlap data: `data/revision_008b_nonoverlap_s42_43_100_e300/`

- synthesis data: `data/source_contrast_synthesis/source_contrast_synthesis.json`

- figure: `figures/source_decomposition.png`
