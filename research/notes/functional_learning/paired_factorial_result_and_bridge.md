# Paired factorial result and natural-language transfer design

## Paired template result

The repaired paired run is `scripts/paired_factorial_and_update_probe.py`; outputs are in `data/paired_factorial_update_probe/`.

This run corrected the corruption vs loss design pairing problem. The two corrupted arms used the same epoch/batch corruption stream and reset dropout seeds before each forward pass. The template substrate was also tightened: 36 train groups and 12 held groups are unique, query side and source order are independently cycled, and the target entity appears first and second equally often in both train and held groups. Tokenization validation found no issues.

Endpoint results on the repaired task:

| Arm | Held flip | Held U | Held R | Held N | Train flip | bg corrupted positions | answer labels |
|---|---:|---:|---:|---:|---:|---:|---:|
| paired corrupted answer-only | 2/12 | +1.937 | -2.007 | +1.844 | 33/36 | 187,731 | 36,000 |
| paired corrupted answer + bg | 1/12 | +1.284 | -0.979 | +1.306 | 20/36 | 187,731 | 36,000 |

The stronger corruption vs loss design endpoint statement is therefore narrowed. Adding background loss under matched corruption/dropout streams still weakens training acquisition on the controlled task (33/36 to 20/36 train flips) and slightly lowers held joint correctness (2/12 to 1/12), but the paired held result is much smaller than the unpaired corruption vs loss design separation. The answer-only arm itself generalizes poorly on the decoupled unique task, so the template substrate is now mainly evidence for fragility of transfer rather than a stable source of a broad rule.

The retain side remains the central failure. In the paired answer-only arm, held U becomes strongly positive (+1.937) while held R is strongly negative (-2.007). The answer + bg arm also keeps positive held U (+1.284) and negative held R (-0.979). This says the model often accepts the newly stated update but does not robustly preserve the target source state when the other entity is updated.

## One-step update measurement

From the final corrupted answer-only snapshot, a fixed-batch measurement found:

- ||g_answer|| = 2.391
- ||g_bg|| = 8.396
- cos(g_answer, g_bg) = +0.0413
- ||g_answer + g_bg|| = 8.825
- with max norm 1.0, the combined update is clipped to coefficient 0.1133, while answer-alone would be clipped to 0.4182.

The background gradient is much larger and the combined update is therefore dominated by the background component, but it is not anti-aligned with the answer gradient in this measured batch. One optimizer step from the same snapshot reduced both answer and background evaluation losses for all three update types:

| Update | Δ answer loss | Δ bg loss | Δ held flip | Δ held U | Δ held R |
|---|---:|---:|---:|---:|---:|
| answer | -0.023793 | -0.023685 | 0 | -0.072462 | +0.077705 |
| bg | -0.003784 | -0.107630 | 0 | -0.090957 | +0.094410 |
| combined | -0.009868 | -0.104371 | 0 | -0.095032 | +0.098392 |

This does not support a simple immediate directional-conflict story. The better current interpretation is an optimizer/allocation interaction over training: background targets are numerous and their gradients dominate clipped AdamW steps through the same private-adapter parameters, reducing how much the relation-answer objective shapes a transferable retain/update computation. That is weaker and more precise than saying background gradients inherently oppose binding.

## Interleave result read with the same caution

The Step034b interleave run installed binding to 6/12 held flips after 200 answer-only corrupted epochs. Continuing answer-only ended at 4/12 held flips. Interleaving answer-only and background-only steps ended at 0/12 held flips, with held U=+1.822 and held R=-1.157.

This run shows that the tested mixed schedule damages retain-side held transfer in the template task. It should not be read as proof that a separate parameter path is required. The script alternates updates through the same parameters and optimizer state, and its actual counts are 750 answer steps and 750 background steps, not the same answer dose plus additional background steps. The loss of held flips may reflect reduced answer reinforcement, background pressure, optimizer history, or their combination.

## Natural-language scorer status

The multi-token scorer `scripts/multitoken_scorer.py` was implemented and evaluated on the earlier accepted state-update pilot rows. Outputs are in `data/multitoken_scorer_pilot512/`.

The scorer consumed all 110 rows / 55 pairs with zero scoring errors. It masks the candidate span in the final use frame and scores answer versus foil symmetrically by mean per-token log probability; summed log probability is also saved for equal-token-length subsets. The pair-level signed recipient-change measure is U+R, and joint UPDATE-and-RETAIN correctness is recorded separately.

Baseline coherent86 on the pilot rows:

- UPDATE correct 42/55.
- RETAIN correct 13/55.
- Joint UPDATE-and-RETAIN correct 3/55.
- Mean U = +1.9352.
- Mean R = -1.9542.
- Mean U+R = -0.0189.
- Equal-token-length subset: 15 pairs, joint 1/15 by summed score.

The scorer is usable for cleaner natural state-update rows, but the pilot rows are not yet strong training material. Manual samples and literal source-position extraction show residual semantic noise and role-detection imperfections. The baseline readout nevertheless matches the emerging pattern: coherent86 often accepts an update state but fails the other-entity update retain side.

## Implication for the next BabyLM-facing work

The current evidence argues against spending more time deriving a universal result from the tiny template task. Its strongest value is now methodological and mechanistic: it exposes the update/retain asymmetry, transfer fragility, and optimizer/allocation interactions that any BabyLM intervention must measure.

The next meaningful bridge should use the natural-language contrastive substrate once it has cleaner accepted rows. A bounded coherent86 pilot should preserve inherited ALN material and displace only filler-like material. It should compare matched ordinary continuation with an intervention in which strict contrastive rows receive explicit relation-answer credit while ordinary ALN learning continues in a controlled schedule. The readout must include held strict-packet U, R, U+R, joint correctness, and cheap7/Entity preservation against the exact coherent86 and matched continuation references. Parent anchoring is still only a preservation factor to cross later, not evidence of added competence by itself.

## Natural-row answer-only preflight on noisy state-update pilot rows

The additional preflight used `scripts/revision_035b_answer_only_train_pilot.py` on the 55 accepted earlier analysis pilot pairs, using 40 train pairs and 15 held pairs. This was a pipeline and mechanism preflight, not evidence that the pilot rows are ready for BabyLM training.

The learner masked only the final answer span in `use_sentence_frame` and trained the coherent86 private adapter with answer-token CE for 90 epochs at lr=5e-5. Results are in `data/revision_035b_answer_only_train_pilot/`.

| Eval | Joint | Update | Retain | mean U | mean R | mean U+R |
|---|---:|---:|---:|---:|---:|---:|
| baseline train | 2/40 | 30/40 | 9/40 | +2.171 | -2.086 | +0.085 |
| baseline held | 1/15 | 12/15 | 4/15 | +1.307 | -1.603 | -0.296 |
| final train | 14/40 | 34/40 | 20/40 | +2.009 | +0.594 | +2.603 |
| final held | 0/15 | 11/15 | 3/15 | +2.273 | -2.040 | +0.233 |

The multi-token answer-loss path can shape the private adapter on natural rows: train joint correctness improves from 2/40 to 14/40 and train R becomes positive. But there is no held pair transfer in this noisy pilot; held joint correctness falls from 1/15 to 0/15 and held R remains strongly negative. This reinforces the need for cleaner accepted rows and an exposure-matched ALN-preserving pilot, rather than drawing a general law from either the template task or noisy Qwen-generated rows.

Important procedural lesson for the next script: save trained checkpoints and final pair-level score files, not only compact summaries, whenever the result may need later error analysis.
