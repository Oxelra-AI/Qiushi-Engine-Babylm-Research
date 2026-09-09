# seed43022 dose practical predictions fit and COMPACT_EXPERIENCE column analysis

Summary JSON: `experiments/archive/relation_learning/data/fit_and_paired_context_columns/summary.json`

## Strict-complement OFF -> ALN per-source late contrasts

- seed 43022 ALL: ALN-minus-OFF -0.1005 nats, SE 0.0047, ALN lower on 63.8% rows.
- seed 43122 ALL: ALN-minus-OFF -0.1037 nats, SE 0.0051, ALN lower on 64.0% rows.

Per-source CSV: `experiments/archive/relation_learning/data/fit_and_paired_context_columns/paired_context_strict_complement_aln_minus_off_per_source.csv`

## Seed43022 dose ordinary-fit costs

- heldout2647_full: dose21-base +0.0047, dose25-base +0.0239, dose25-dose21 +0.0192 nats.
- heldout2647_inherited_aln_text_hit: dose21-base +0.0060, dose25-base +0.0254, dose25-dose21 +0.0194 nats.
- heldout2647_no_inherited_aln_text: dose21-base +0.0041, dose25-base +0.0231, dose25-dose21 +0.0190 nats.
- strict_complement_ngram_3000: dose21-base +0.0388, dose25-base +0.0315, dose25-dose21 -0.0074 nats.

Dose per-source CSV: `experiments/archive/relation_learning/data/fit_and_paired_context_columns/seed43022_dose_ordinary_fit_per_source_contrasts.csv`

## COMPACT_EXPERIENCE available cheap-column contrasts

- seed43022_ALN_minus_OFF: cheap7 +0.513, BLiMP -0.450, EWoK -0.610, Entity +2.450, Reading -0.190.
- seed43022_SHUF_minus_OFF: cheap7 +0.024, BLiMP +1.000, EWoK -0.470, Entity -0.440, Reading +0.570.
- seed43022_ALN_minus_SHUF: cheap7 +0.489, BLiMP -1.450, EWoK -0.140, Entity +2.890, Reading -0.760.
- seed43122_ALN_minus_OFF: cheap7 +0.181, BLiMP -1.260, EWoK +2.640, Entity +2.430, Reading -0.740.

Budget/source notes: `experiments/archive/relation_learning/data/fit_and_paired_context_columns/paired_context_budget_and_source_notes.json`
