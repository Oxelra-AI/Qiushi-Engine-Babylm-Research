# compact order result and roberta transfer decision: compact ordered-vs-scrambled result integrated + RoBERTa transfer redesign

## 1. Completed Evidence

Both compact-order arms trained to exact 40M legal words (1,012 steps each, stock DeBERTa-v2
8x480, 34,467,424 params, legal spatial repair route status tokenizer). Selected official-compatible cheap scores
and a fixed ordered source+compact channel probe were computed for chck_20M and chck_40M.

Authoritative artifacts:
- `data/compact_order_integrated_readout/compact_order_integrated_readout.json`
- `data/compact_order_channel_probe/compact_order_channel_probe.json`
- `data/compact_order_selected_eval/{ordered,scrambled}/{chck_20M,chck_40M}/...`

### Official ordered-minus-scrambled cheap deltas

chck_20M:
- cheap7 +0.172143, cheap6_no_GlobalPIQA +0.195833, cheap5_no_GlobalPIQA_Reading +0.262,
  EWoK+Entity **+1.675** (EWoK +2.30, Entity +1.05), BLiMP -0.06, Supplement -1.43,
  COMPS -0.55, GlobalPIQA +0.03, Reading -0.135.

chck_40M:
- cheap7 **-0.67**, cheap6_no_GlobalPIQA **-0.288333**, cheap5_no_GlobalPIQA_Reading **-0.298**,
  EWoK+Entity -0.02 (EWoK -0.53, Entity +0.49), BLiMP -0.67, Supplement -1.04,
  COMPS +0.26, GlobalPIQA -2.96, Reading -0.24.

### Channel probe (fixed ordered compact events; negative = ordered lower NLL)

chck_20M piece-weighted category deltas: retained_content -0.080398, source_absent_content
-0.053718, function_other -0.079314. Source-absent interaction vs controls_mean **+0.026137**
(P(diff<0)=0.015): source-absent is LESS ordered-favorable than controls -> `source_absent_category_selective=false`.

chck_40M piece-weighted category deltas: retained_content -0.010249, source_absent_content
**-0.25524**, function_other -0.024953. Source-absent interaction vs controls_mean **-0.237639**
(P(diff<0)=1.0): source-absent is much MORE ordered-favorable than controls ->
`source_absent_category_selective=true`.

## 2. Scientific interpretation

Two clean, opposite facts:

1. **Downstream (official) verdict is negative for coherent order at 40M.** At the later checkpoint,
   coherent compact word order is worse than scrambled on cheap7 and, more importantly, on the
   stable families cheap6_no_GlobalPIQA (-0.288) and cheap5_no_GlobalPIQA_Reading (-0.298), with
   EWoK+Entity essentially flat (-0.02). The chck_20M positive is an early general/native-order
   familiarity effect that does not survive to 40M and is not source-absent-specific.

2. **The source-absent local NLL channel does emerge with order by 40M** (interaction -0.238, P=1.0),
   but it coincides with WORSE selected competence. This is the same local-vs-selected-surface
   DISSOCIATION seen at earlier analysis/077: a source-absent local target signal can improve its intended
   local NLL while damaging the BabyLM-compatible surface.

**Committed conclusion.** The unnatural ordered-versus-scrambled contrast should receive no further
H100 spending: at 40M, preserving compact word order at fixed compact lexical multiset is net-negative
on the official-compatible stable readouts, despite creating a strong local source-absent NLL channel.
This resolves only the artificial scrambling subquestion. It does **not** prove that fluent syntax is
unnecessary in the mature natural compact treatment, because the validated compact-vs-repeat marginal
bundles faithful compression, content density, source-wide lexical coverage, natural surface form, and
diversity reinvestment, and its strongest downstream effect emerged much later (~80M). Do NOT extend
the ordered/scrambled line, and do NOT turn the source-absent channel into explicit target prioritization
(closed since earlier analysis/077).

## 3. What remains open — the real transfer question

The validated positive result is the natural compact-view + reinvestment MARGINAL (compact vs matched
repeat, +1.35 mean7 at 80M under DeBERTa MLM). That effect works through content density, source-wide
lexical coverage, semantic transformation, and reinvested source diversity — NOT through fluent order
(now excluded) and NOT through copied-token retrieval or reciprocal conditioning (previously excluded).

The ordered/scrambled result does not touch this marginal. The open, high-value question is:

> Does the natural compact-vs-repeat marginal reproduce under a DIFFERENT bidirectional MLM encoder
> (stock absolute-position RoBERTa/BERT) evaluated across the full late trajectory?

This is a generalizability test relevant to larger models. A positive result would show that the bundled
natural compact-vs-repeat data marginal transfers across two different bidirectional-MLM coordinates;
an independent seed would still be needed before treating it as robust. A negative result would bound
the effect in this tested stock absolute-position RoBERTa coordinate; it would not identify relative
position or disentangled attention as the cause.

## 4. Corrected RoBERTa transfer design (supersedes the roberta transfer pair scaffold ready 40M scaffold)

The roberta transfer pair scaffold ready scaffold used 40M streams. 20M/40M neutrality cannot close
transfer because the DeBERTa effect is late-emerging (peaked ~80M). The transfer test must:

- use the NATURAL compact-vs-repeat marginal (not ordered vs scrambled);
- train BOTH arms to full 100M legal exposure with dense late checkpoints (every 10M, and denser
  20M in the 60-100M band where the DeBERTa effect matured), matched initialization/mask stream;
- score selected cheap official-compatible metrics across the late trajectory;
- judge on stable families (cheap6_no_GlobalPIQA, cheap5_no_GlobalPIQA_Reading, EWoK+Entity, Supplement,
  Entity, COMPS), not on GlobalPIQA/Reading alone;
- account for the known tokenization difference: compact produces ~+96,000 active tokens and -784 word
  groups per 40M (concentrated in the changed block) vs repeat, so a fixed word budget gives compact
  slightly more BPE supervision — this is part of the treatment, not a confound to remove.

Data: the existing original DeBERTa 100M streams
`data/density_cleanqwen_overlay_medium_riskhard/cleanqwen_fineweb_compact_view_reinvest_100M.jsonl`
and `.../cleanqwen_fineweb_repeat_compact_reinvest_100M.jsonl` are the matched pair (same row order,
differ only in the 3,005 compact-pair rows per 10M block). compact order result and roberta transfer decision verifies row-matching before any launch.

Recipe (matches DeBERTa reference): RoBERTa 8x480 (30,528,064 params), legal spatial repair route status tokenizer,
seeds 43/43022/43023, AdamW lr 0.001 wd 0.01 warmup 0.06 betas (0.9,0.98), batch 256, seq 256,
WWM fixed p=0.15, `--max_word_exposure 100000000`, `--checkpoint_words 10000000`,
`--lr_total_steps` set to the actual 100M step count (~2529 for this stream geometry; confirm from preflight).

## 5. Cost/admission

This is expensive (two full 100M RoBERTa runs on the two H100s, ~parallel). It is admitted because it
answers a decisive generalizability question still unresolved by the existing studies, and no
cheaper reliable evidence exists (the DeBERTa marginal is established; the transfer is not). It will
decide whether the compact-view marginal survives at least one non-DeBERTa bidirectional-MLM coordinate
or whether the effect must be treated as bounded to the coordinates actually tested so far (DeBERTa
positive, GPT2 causal negative, RoBERTa pending). Both outcomes materially change the research judgment.

## 6. Boundaries

- No explicit source-absent target prioritization.
- Ordered/scrambled line closed; do not add trajectories, doses, or renamed variants.

## 7. compact order result and roberta transfer decision launch record

After the 100M pair audit and explicit cost admission, the full RoBERTa transfer was launched as managed work:

- Repeat training: `training/runs/roberta_repeat_compact_reinvest_100M_seed43022`.
- Compact training: `training/runs/roberta_compact_reinvest_100M_seed43022`.
- The first selected-evaluation attempt was cancelled before evaluation because the wrapper would have passed an inconsistent GPU selection; it provides no model evidence.
- Corrected selected evaluation uses `scripts/after_roberta_transfer_eval.py`, with outputs under `data/roberta_full100m_selected_eval/` and `data/roberta_full100m_integrated/`.

The corrected evaluator waits for both arms to finish, then scores all 10M-grid checkpoints (`chck_10M` ... `chck_100M`) with compact on GPU0 and repeat on GPU1, and integrates compact-minus-repeat late-band metrics over 60M-100M. Do not poll these tasks; collect delivered results and read the saved artifacts.
