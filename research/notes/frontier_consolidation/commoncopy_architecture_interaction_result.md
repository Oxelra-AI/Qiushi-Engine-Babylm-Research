# commoncopy architecture interaction result — Common-copy DeBERTa positional-ablation interaction: causal adjudication

## Question resolved
Does the architecture-interaction finding — that removing DeBERTa's disentangled p2c/c2p positional-score
terms collapses the stable-family compact-minus-repeat advantage — reflect the removed
positional-score pathway, or was it an initialization / optimization-basin artifact of the
plain no-disentangle construction?

initialization parity and interaction interpretation established that omitting the pos_key_proj/pos_query_proj modules shifts the torch
construction RNG stream, so the architecture-interaction plain no-disentangle arm did NOT share common
initialization with full (53/140 common tensors nonexact at init; initial-logit mean_abs 0.494).
The common-copy wrapper (common copy disambiguation tool) copies all 140 same-shaped common tensors from the same
full DeBERTa init into the pos_att_type=[] target before training, removing that confound
while still omitting the 32 positional-projection tensors (target still 30,773,344 params vs
full 34,467,424). After-copy 140/140 exact; initial full-vs-target logit mean_abs 0.0024677.

## Cells (all legal, verified)
Both common-copy arms trained to 100,000,000 counted words, 2,529 steps, 10 checkpoints,
WWM 0.15, AdamW lr0.001, batch256/seq256, tokenizer `compliant16k_reinvest10M`, seeds
43/43022/43023, params 30,773,344, finite final losses (compact 4.234443; repeat 4.421523).
Integrity: `experiments/archive/frontier_consolidation/data/commoncopy_integrity_final` all_ready=true,
problem_count=0. (Reader repaired: `batch_size` is not written to scientific_metrics.json by
the base trainer; it is verified from the architecture-interaction run command `--batch_size 256`.)

Full compact/repeat reference cells reuse the validated architecture-interaction per-target selected payloads.

## Selected 80M/100M panel (stable families)
`experiments/archive/frontier_consolidation/data/commoncopy_architecture_interaction_selected_panel`

compact-minus-repeat by architecture (mean over 80M/100M):
- full DeBERTa:      cheap6_no_GlobalPIQA +0.2721, cheap5 +0.2440, EWoK+Entity_sum +1.340, cheap7 -0.266
- common-copy nodis: cheap6_no_GlobalPIQA +0.3400, cheap5 +0.3990, EWoK+Entity_sum +0.955, cheap7 +0.787

interaction (nodis_minus_full), mean over 80M/100M:
- cheap6_no_GlobalPIQA **+0.0679**  (architecture-interaction plain: **-0.6908**)
- cheap5_no_GlobalPIQA_Reading **+0.1550**  (architecture-interaction plain: -0.6080)
- EWoK+Entity_sum **-0.3850**  (architecture-interaction plain: -3.1200)
- cheap7 +1.0529 (mostly GlobalPIQA-carried; GlobalPIQA interaction +6.96)

## Four-cell item bootstrap (400 resamples, 170,722 common items)
`experiments/archive/frontier_consolidation/data/commoncopy_architecture_interaction_four_cell_bootstrap`
Sign convention: interaction = nodis_minus_full; negative = removing p2c/c2p attenuates compact-minus-repeat.

- chck_80M stable_five item interaction **+0.372 pp**, item interval [0.030, 0.740] (excludes zero, positive)
- chck_100M stable_five item interaction **+0.547 pp**, item interval [0.188, 0.874] (excludes zero, positive)
- chck_80M EWoK+Entity item interaction -0.208 pp, item interval [-1.292, 0.945] (crosses zero)
- chck_100M EWoK+Entity item interaction -0.028 pp, item interval [-1.181, 1.104] (crosses zero)
- all cluster intervals cross zero

## Conclusion (causal adjudication)
The architecture-interaction stable-family collapse of compact-minus-repeat under plain no-disentangle was
**largely an initialization / optimization-basin effect**, not evidence that DeBERTa's
disentangled p2c/c2p positional-score pathway is necessary for the compact semantic
second-view advantage.

When common initialization is controlled:
- the strong negative interaction disappears (cheap6 interaction from -0.6908 to +0.0679);
- stable_five item interaction is small-POSITIVE with intervals excluding zero at both checkpoints;
- EWoK+Entity item interaction is indistinguishable from zero;
- the common-copy no-disentangle arm itself preserves the compact advantage on stable families
  (nodis compact-minus-repeat cheap6 +0.340, cheap5 +0.399).

So compact-minus-repeat's stable-family benefit does NOT require the p2c/c2p positional-score
package. Two things remain true and jointly bound the DeBERTa story:
1. Removing p2c/c2p still hurts overall competence (nodis arms are ~4-5 points lower on
   cheap6/cheap5 absolute than full; Entity ~18 vs ~27), i.e. the positional-score pathway is
   important for absolute capability but not for the compact-view *treatment effect*.
2. The residual difference between full and common-copy nodis remains bundled with capacity
   (30.77M vs 34.47M) and score composition; c2p_only vs p2c_only is the only bit-identical-init
   contrast for a narrower directional question, but the treatment-effect question is now answered.

## What this means for the transfer puzzle
The earlier hypothesis that "structured semantic compression becomes sample-efficient only when
DeBERTa's positional-score machinery binds recurring content to changed roles" is NOT supported:
the compact advantage survives removal of that machinery under matched init. The GPT-2/RoBERTa
transfer failures therefore cannot be attributed to the absence of disentangled positional scores.
The architecture-interaction line has delivered its decisive result and does not justify further
DeBERTa subdivision (c2p-only/p2c-only masking-rate screens excluded per prior stopping rule).

The open field-level question returns to: what property of the DeBERTa MLM coordinate (vs GPT-2
causal, vs stock RoBERTa) lets compact second views produce stable downstream benefit, given that
it is not the positional-score pathway. Candidate remaining factors: bidirectional MLM objective +
absolute-position-biased-input + encoder relative embeddings retained even in nodis (both nodis
checkpoints keep relative_attention=true, position_biased_input=true). The RoBERTa arm already had
bidirectional MLM and still failed broad transfer, so the surviving distinguishing factor is narrow.

## Boundary
No new BabyLM training, SuperGLUE, AoA, upload, or leaderboard submission. Protected assets
unchanged: coherent86 alpha0.75 Overall 42.1210247099666; chck_82M 41.942481167385985 public;
chck_84M 42.0189129742181.
