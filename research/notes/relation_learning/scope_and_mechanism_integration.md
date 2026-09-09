# argument map corrections scope and mechanism integration

Created: 2026-09-06T10:25:35Z

This note integrates the delivered Wikipedia natural-restatement scores and seed43122 split Entity scores, then repairs two wording-sensitive parts of the relation-typed composition argument using already-scored data: ordinary held-out decomposition and source-content concentration.

## 1. Wikipedia natural-restatement T/U/N

- Wikipedia natural-restatement scorer convention: `gain_T_vs_N = NLL(N) - NLL(T)`, so positive contrast means an arm gains more from the true source relative to the neutral source than the comparison arm.
- The frozen WikiLarge/Simple-English probe direction is English Wikipedia source sentence to Simple English target sentence. The overlap class is exact tokenizer-ID recurrence of the masked target in the true source; nonoverlap means the target tokenizer ID is absent from the true source.
- Overlap targets: V−C gain_T_vs_N is +0.2376 ± 0.0706 and V−R is +0.1286 ± 0.0533, with V−C gain_U_vs_N near zero (+0.0069 ± 0.0159). VIEW therefore transfers to natural restatement where the target token recurs in the source under changed sentence form.
- Nonoverlap targets: R−C gain_T_vs_N is -0.2171 ± 0.0285 and gain_U_vs_N is +0.0110 ± 0.0499. Exact recurrence has a source-specific natural-domain cost for source-absent substituted tokens.
- Nonoverlap targets: V−C gain_T_vs_N is small (+0.0316 ± 0.0398), while V−R is +0.2487 ± 0.0646 because R is actively below C. This is a relation-by-target-form pattern, not absence of VIEW transfer.
- All targets: V−C gain_T_vs_N is +0.1346 ± 0.0539 and V−R is +0.1887 ± 0.0116; R−C is -0.0540 ± 0.0540.

Interpretation: the adequate natural restatement probe carries two separable effects. VIEW-trained models use a true aligned source better than CLEAN and REPEAT on source-recurring restatement tokens; exact-recurrence models incur a stable cost on source-absent substituted tokens. This strengthens the relation-typed form-robustness principle: the practiced within-window relation determines which surface transformations remain supported by the installed source-use routine.

## 2. Seed43122 official Entity split integration

- Seed43122 official Entity split integration: RS−C zero-update -2.57 versus original R−C +9.09; RS−R -11.66.
- Seed43122 deeper updates: rel_ge3 RS−C +1.21 versus original R−C -1.30; RS−R +2.50.
- Seed43122 VIEW split behavior differs from seed43022: rel_eq0 VS−C +1.00, rel_ge3 VS−C -3.25, VS−V at rel_ge3 -8.81. The seed43022 VIEW_SPLIT positive-update behavior does not replicate.

Interpretation: REPEAT's zero-update Entity benefit is localized at two seeds; splitting removes it at seed43122 as at seed43022. The VIEW_SPLIT positive-update behavior from seed43022 does not reproduce at seed43122, so the behavioral face should remain REPEAT-asymmetric rather than two-sided.

## 3. Ordinary held-out decomposition of local versus split share

- Ordinary held-out decomposition at seed43022: R−C +0.1074 nats = RS−C +0.0864 + R−RS +0.0210; the in-window share is 19.5% of the small ordinary-loss excess.
- For VIEW: V−C +0.0994 nats = VS−C +0.0791 + V−VS +0.0203; the in-window share is 20.4%. Most ordinary held-out loss difference is not the same-window relation component, in contrast to compact T/U/N.

This directly corrects the relation to ICLM: the present experiment separates corpus exposure from window adjacency. At this BabyLM dose, the window-adjacency component is small on ordinary held-out text and large only on compact/source-conditioned probes. Therefore the present data do not explain ICLM's broad perplexity loss; they identify a targeted in-window computation that ICLM did not isolate.

## 4. Source-content concentration from relation practice principle

- relation practice principle concentration readout is restricted to tokenizer-nonoverlap rewrite targets, so the target token is absent from the true source. It cannot show exact copying of the answer token; it tests whether the recognized-source mass movement supports or competes with the target.
- R−C under true-source T: source-content mass changes by +0.0750 ± 0.0212, while target probability changes by -0.0081 ± 0.0022.
- R−C target/source-content-mass ratio changes by -174.1052 ± 387.2074; the shifted mass is not precision on the correct nonoverlap target, but diffuse source-content pull competing with the target.
- V−C under T also raises source-content mass by +0.0660 ± 0.0054, but raises target probability by +0.0580 ± 0.0046; this separates VIEW from REPEAT as content-conditioned support rather than source-content competition.

Mechanism wording should change from 'identity pull' to: exact recurrence trains a trigger that transfers to related spans and moves probability mass onto source-content tokens, but its answer-level readout does not transfer to nonidentical targets. VIEW also recognizes source content, but the target probability moves with it rather than against it.

## Output files

- `experiments/archive/relation_learning/data/integrated_scope_and_mechanism/wikipedia_scope_key_across_seed.csv`
- `experiments/archive/relation_learning/data/integrated_scope_and_mechanism/wikipedia_nonoverlap_seed_values.csv`
- `experiments/archive/relation_learning/data/integrated_scope_and_mechanism/entity_seed43122_late_contrasts_by_group.csv`
- `experiments/archive/relation_learning/data/integrated_scope_and_mechanism/ordinary_loss_decomposition.csv`
- `experiments/archive/relation_learning/data/integrated_scope_and_mechanism/source_specificity_T_concentration_across_seed.csv`
