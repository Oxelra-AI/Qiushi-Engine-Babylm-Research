# reciprocal multiview mechanism and scaffold reciprocal multi-view mechanism note

## Scientific Motivation was run

The causal transfer result synthesis causal GPT compact-vs-repeat result did more than close a direct architecture-transfer claim. Its negative pattern suggests an objective/topology interaction: DeBERTa MLM can use both sides of a source/compact pair as bidirectional evidence for masked tokens, whereas the tested causal packet gives cross-view evidence only to the later segment. The proposed next mechanism test is: finish the pending DeBERTa trajectory comparison, then pursue objective-compatible reciprocal multi-view learning rather than repeating the same one-direction causal packet or endpoint alpha engineering.

CPU/file-only analysis while the DeBERTa selected grids remained incomplete. No new training was performed.

## Companion experiment evidence

DeBERTa compact-view triangle controls completed training at 100M. The reference-equivalence discrepancy was a scheduler artifact: a 26-step schedule had been used instead of the historical 2529-step schedule; this was repaired. Official-compatible triangle scores were still unavailable. Reciprocal/objective-asymmetry probes therefore remained insufficient for a new training decision.

## New probe: reciprocal view-lift

Script: `scripts/reciprocal_view_lift_probe.py`.

Definition: for a token on one side of a source/view pair, compare NLL when both sides are present to NLL when the target side is alone. Lift = NLL(side-only) - NLL(paired). Positive lift means the opposite view helps prediction. The probe separates compact vs exact-repeat controls, target side, order, and exact-token/text copy strata.

### DeBERTa MLM chck82/chck100

A dry construction smoke over 8 pairs produced 96 target records with valid accounting. A 128-pair CPU probe was then run on trusted-code scale1.75 `chck_82M` and `chck_100M`:

- chck82 output: `data/reciprocal_view_lift_probe_chck82_128/`.
- chck100 output: `data/reciprocal_view_lift_probe_chck100_128/`.
- comparison: `data/reciprocal_view_lift_comparison_82_100/`.

Key strata from the comparison:

| pair type | target | copy_by_text | chck82 mean lift | chck100 mean lift | 100M-82M |
|---|---|---:|---:|---:|---:|
| compact | compact/rewrite side | false | +0.061620 | +0.040022 | -0.021598 |
| compact | source side | false | -0.106480 | -0.112485 | -0.006005 |
| compact | compact/rewrite side | true | +5.371522 | +5.316419 | -0.055104 |
| compact | source side | true | +5.131248 | +5.106071 | -0.025177 |
| repeat | repeated side | true | +5.120093 | +5.143040 | +0.022947 |
| repeat | source side, non-copy | false | -0.312812 | -0.310210 | +0.002602 |

Scientific reading: DeBERTa MLM has strong paired-context lift, but the measured lift is overwhelmingly exact lexical availability/copy. The load-bearing compact non-copy lift is tiny and not noticeably linked to the 82M-to-100M official decline. This warns against another mask-target or innovation-weighting route: simply selecting more source-absent or copied pair tokens is unlikely to become the missing general principle and would overlap closed earlier analysis/077/160 evidence.

### Causal GPT chck82

Small CPU probes on the trained earlier analysis causal compact and repeat GPT checkpoints (`chck_82M`) confirmed the causal objective asymmetry directly:

- compact output: `data/reciprocal_view_lift_probe_causal_compact_chck82_32/`.
- repeat output: `data/reciprocal_view_lift_probe_causal_repeat_chck82_32/`.

For causal compact:

| order | target segment | mean lift |
|---|---|---:|
| compact-view -> source | first segment (view) | 0.000000 |
| compact-view -> source | second segment (source) | +1.761393 |
| source -> compact-view | first segment (source) | 0.000000 |
| source -> compact-view | second segment (view) | +1.710828 |

For causal repeat:

| order | target segment | mean lift |
|---|---|---:|
| repeat -> source | first segment (repeat) | 0.000000 |
| repeat -> source | second segment (source) | +1.714659 |
| source -> repeat | first segment (source) | 0.000000 |
| source -> repeat | second segment (repeat) | +2.628201 |

This is the expected one-way structure: the first segment cannot use future evidence, while the second segment can. Exact repetition is easier than compact semantic transfer, especially for source->repeat. The causal negative is therefore consistent with missing reciprocal conditioning for each pair, not merely with bad pool bookkeeping. It still does not prove that a reciprocal causal variant will work, because repeat controls may remain advantaged by exact lexical copying and because the DeBERTa MLM non-copy lift measured above is small.

## Candidate construction: reciprocal causal pools

Script: `scripts/build_reciprocal_causal_pools.py`.

Two scaffolds were materialized and audited. The all-pair version uses every pair in both directions but doubles the pair-view word dose relative to causal gpt transfer experiment design, so it is not the cleanest first discriminator.

### Preferred scaffold: dose-matched shuffled reciprocal pairs

Directory: `data/reciprocal_causal_transfer_scaffold_dosematched_shuffle/`.

- deterministic shuffled pair seed: 178585
- selected pairs: 6,071
- pair units per arm: 12,142 (each selected pair appears source->view and view->source, or source->repeat and repeat->source)
- pair words per arm: 423,512, matching the causal gpt transfer experiment design original pair budget of 423,511 to within +1 word
- filler words per arm: 9,576,488 from the same neutral filler source
- exact legal words: 10,000,000 per pool
- rows: 73,877 per pool
- pair positions match: true, 12,142 positions
- row word delta compact-minus-repeat: exactly 0
- compact pool SHA: `13a01574bc90cd1c8a6e3792a0e637d7f00ae372b42a480b27c768d83faeecd3`
- repeat pool SHA: `1fcb64ce1380906534721537597def8d50765f2ed162e67b94c04e378d852e49`
- selected-pair manifest SHA: `d2729b6752561dac1483e8f7e34c71393de288fc8ed81537fce5e1c1cd6eedbd`

Token audit: `data/reciprocal_causal_token_exposure_audit_dosematched_shuffle/`.

Under the neutral causal gpt transfer experiment design tokenizer:

| quantity | compact | repeat | compact-repeat | rel. |
|---|---:|---:|---:|---:|
| raw tokens incl EOS / epoch | 14,587,760 | 14,556,070 | +31,690 | +0.2177% |
| active 256-token positions / epoch | 14,587,648 | 14,555,904 | +31,744 | +0.2181% |
| chunks / epoch | 56,983 | 56,859 | +124 | +0.2181% |
| token/word | 1.458776 | 1.455607 | +0.003169 | +0.2177% |
| total updates at batch 128 | 4,460 | 4,450 | +10 | +0.2247% |
| total updates at batch 256 | 2,230 | 2,230 | 0 | 0.0000% |

If this candidate is ever trained, batch size 256 is preferable because it removes the update-count asymmetry; the remaining effective-token asymmetry is roughly the same magnitude as causal gpt transfer experiment design (+0.218% vs +0.222%). A valid first H100 test would be the lowest reliable matched real-training screen, not another 100M run by default: both arms, same neutral tokenizer, same GPT2 config, same seed, batch 256, selected checkpoint(s), official-compatible cheap7 family comparison, and early stop if compact does not beat reciprocal repeat beyond GlobalPIQA/Reading volatility.

## Current route judgment

The new evidence strengthens the objective/topology interpretation but also places a hard warning on it:

1. Causal one-direction asymmetry is real and directly measured.
2. Reciprocal causal construction is possible under exact legal word accounting and matched pair-word dose.
3. However, DeBERTa MLM pair lift is mostly lexical copy lift; compact non-copy reciprocal lift is small and not a visible 82M-retention signal in this bounded probe.
4. Therefore no new H100 training should be launched until the pending DeBERTa common-grid comparison is complete and DeBERTa triangle control scores are considered.

The immediate execution priority remains the pending DeBERTa trajectory comparison:

- protected scale1.75 seed43022 reference common 70M-100M grid (GPU0; skips valid cached 80M/100M).
- scale1.25 seed43022 dense common 70M-100M grid (GPU1).
- pending: score scale1.75 seed43122 dense on the same common grid.

Only after those results should the system decide whether objective-compatible reciprocal multi-view learning deserves a small real-training screen, and the dose-matched scaffold above is the technically cleanest candidate if that decision is positive.
