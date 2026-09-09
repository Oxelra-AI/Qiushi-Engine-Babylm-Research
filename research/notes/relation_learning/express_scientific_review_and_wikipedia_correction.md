# causal lm objective generality plan Express scientific review: corrected Wikipedia transfer reading

This note is research-facing. It corrects the natural-domain interpretation before any final paper statement is hardened.

## What was checked

- Frozen probe: `experiments/archive/relation_learning/analysis/wikipedia_simplification_restatement_probe/wikipedia_simplification_pairs.jsonl` and `experiments/archive/relation_learning/analysis/wikipedia_simplification_restatement_probe/wikipedia_simplification_probe_records.jsonl`.
- Validation: `experiments/archive/relation_learning/analysis/wikipedia_simplification_restatement_probe/validation_results.json` reports `PASS`.
- Scored contrasts: `experiments/archive/relation_learning/data/integrated_scope_and_mechanism/wikipedia_scope_key_across_seed.csv` and `experiments/archive/relation_learning/data/wikipedia_simplification_score/wikipedia_late_arm_contrasts.csv`.
- Direction: WikiLarge-clean `source` is the English Wikipedia source sentence and `target` is the Simple English target sentence. Example source→target:
  - source: At a board meeting in January 2004 , Geir Ivarsøy announced his wish to resign as a board member in Opera Software , though he remained active in the company even after that .
  - target: In January of 2004 , Geir Ivarsøy said he wanted to resign as a board member in Opera Software , though he kept working in the company after that .
- Target class definition was rechecked from the constructor and the frozen pairs: `overlap` means the exact target tokenizer ID occurs in the true source token-ID set; `nonoverlap` means it does not. The frozen design has 1,200 pairs, 1,200 overlap targets and 2,400 nonoverlap targets; each target has T/U/N records, for 10,800 records. No class-definition failures were found.

## Corrected natural-domain result

The argument map corrections synthesis over-emphasized the nonoverlap slice and therefore made VIEW's natural transfer sound absent. The full WikiLarge/Simple-English readout has a more informative target-class structure. Values below are across the three original DeBERTa seeds; positive `gain_T_vs_N` contrast means the first arm extracts more benefit from the true aligned Wikipedia source than the second arm, relative to the ordinary N source.

| target class | R−C Δ(T−N) | V−C Δ(T−N) | V−R Δ(T−N) | V−C Δ(U−N) |
|---|---:|---:|---:|---:|
| overlap | +0.109 ± 0.122 | +0.238 ± 0.071 | +0.129 ± 0.053 | +0.007 ± 0.016 |
| nonoverlap | -0.217 ± 0.029 | +0.032 ± 0.040 | +0.249 ± 0.065 | -0.005 ± 0.025 |
| all targets | -0.054 ± 0.054 | +0.135 ± 0.054 | +0.189 ± 0.012 | +0.001 ± 0.017 |

Per-seed signs for `gain_T_vs_N` on the load-bearing classes:

```
seed                 43022 43122 43222
contrast token_class                  
RminusC  ALL             -     -     -
         nonoverlap      -     -     -
         overlap         +     +     -
VminusC  ALL             +     +     +
         nonoverlap      +     +     -
         overlap         +     +     +
VminusR  ALL             +     +     +
         nonoverlap      +     +     +
         overlap         +     +     +
```

The natural-domain statement should therefore be:

1. Exact natural repeats preserve the old ordering from the copy probe: REPEAT > VIEW > CLEAN for identical recurrence.
2. Natural restatements split by target class. On target tokens that recur in the source under changed sentence form, VIEW extracts more true-source benefit than both CLEAN and REPEAT (`V−C` +0.238 ± 0.071; `V−R` +0.129 ± 0.053). The nuisance term `V−C Δ(U−N)` is near zero (+0.007 ± 0.016), so this is source-specific rather than a general register advantage.
3. On target tokens absent from the source, exact recurrence has the stable cost (`R−C` -0.217 ± 0.029), while VIEW is approximately CLEAN (`V−C` +0.032 ± 0.040) but still exceeds REPEAT (`V−R` +0.249 ± 0.065).
4. The VIEW nonoverlap null is not a failure of transfer. It is a relation boundary: the VIEW intervention practiced nonidentical restatement of source-supported content, not lexical substitution requiring a source-absent target token.

## Revised principle wording

The general principle should be phrased as relation-typed form robustness under fixed budget: the relation repeatedly composed inside a context window determines not only whether a source is used, but the surface relation under which retrieval remains useful. Exact recurrence installs a source-recognition routine whose precise readout is bound to identical surface form; when the related source is nonidentical, the trigger persists but the readout can pull probability toward source content and away from changed targets. Restatement practice installs a source-use routine that survives sentence-form changes for source-recurring content. The boundary on source-absent substitute words is predicted by the fact that lexical substitution was not the practiced relation.

## Artifact changes for the next synthesis

- Wrote `experiments/archive/relation_learning/data/express_review/wikipedia_transfer_by_target_class.csv`.
- Wrote `experiments/archive/relation_learning/data/express_review/wikipedia_transfer_by_seed_signs.csv`.
- Wrote `experiments/archive/relation_learning/figures/express_review/fig_wikipedia_target_class_transfer.png` and `experiments/archive/relation_learning/figures/express_review/fig_wikipedia_target_class_transfer.pdf`.

The natural-domain figure for the paper should use the arm × target-class design above, not a nonoverlap-only line. The older nonoverlap-only reading should remain only as the evidence for the exact-recurrence cost on changed/substituted words.
