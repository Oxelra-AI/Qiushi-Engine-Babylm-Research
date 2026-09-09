# paired alignment accounting audit — Repair plan after paired-alignment Wave 1

## Immediate decision

Do **not** launch `experiments/archive/compact_experience/scripts/wave2_train.sh` in its current form.

Reason: the current expanded training files all target 100,000,000 words, but the base pools are not exactly 10,000,000 words. ALIGNED/MISMATCHED/SINGLE_REPEAT are 9,999,840 words, so 100M is 10 full passes plus 1,600 words. SINGLE_ORIG is only 9,551,200 words, so 100M is about 10.47 passes. This is unacceptable for an official-candidate arm.

## What paired alignment accounting audit established

The paired Wave-1 result is a strong mechanism signal, but not yet a broad frontier shift.

Evidence:
- Fast screen: `experiments/archive/compact_experience/data/paired_alignment_eval/paired_alignment_wave1_eval_summary.json`
- Full Entity probe: `experiments/archive/compact_experience/data/paired_alignment_full_entity/full_entity_probe.json`
- Accounting audit: `experiments/archive/compact_experience/data/paired_alignment_accounting_audit.json`

Key result:
- ALIGNED − MISMATCHED fast Entity +10.640; full Entity +10.710.
- ALIGNED − INITIAL_MODEL_STUDIES baseline full Entity +7.380.
- Aggregate fast equal-7 vs baseline is flat (-0.006), with BLiMP/Supplement/COMPS down and Entity/EWoK/GPIQA up.

Interpretation:
- There is a real within-sequence adjacency effect, concentrated in Entity.
- The current evidence does not yet separate same-meaning rewrite correspondence from mere local coherence.
- The low ALIGNED loss (1.946 vs mismatched 3.227) may partly reflect in-context copyability rather than better transferable structure.

## Required next construction: COHERENT-NONSYN control

Build a coherent non-synonymous adjacency arm that keeps local topical/entity coherence but removes same-meaning rewrite correspondence.

The semantic ladder should be:

1. **PARAPHRASE/ALIGNED** — adjacent same-meaning rewrite pair.
2. **COHERENT-NONSYN** — adjacent same-topic or document-adjacent non-paraphrase text.
3. **MISMATCHED** — adjacent unrelated text.

Contrasts:
- `COHERENT-NONSYN − MISMATCHED`: contribution of local coherence.
- `PARAPHRASE − COHERENT-NONSYN`: additional contribution of rewrite correspondence.

Decision rule:
- If `PARAPHRASE ≈ COHERENT-NONSYN > MISMATCHED`, the effect is mostly local coherence. The best next route becomes coherent document/window packing of legal corpus text rather than paired rewrite data.
- If `PARAPHRASE > COHERENT-NONSYN > MISMATCHED`, same-meaning rewrite correspondence is a genuine paired-data signal worth scaling under exact legal accounting.

## Construction constraints

Any new arm must:
- use exactly legal data sources under the current rules,
- use exactly 10,000,000 whitespace words in the base pool, or explicitly cap training at `10 * base_pool_words` and handle checkpoint naming,
- preserve the same 160-word example packing and source accounting as Wave 1,
- keep sentence inventory and word budget matched as closely as possible,
- avoid relying on model-generated text that would trip the external-model/distillation constraints.

Preferred practical route:
- Build COHERENT-NONSYN from existing paired-pool originals plus same-source or same-document neighboring text if recoverable, otherwise from lexical/topic clustering using text-side statistics only.
- If lexical/topic clustering is used, record the method and show overlap/confound statistics: sentence length, source mix, entity overlap, lexical overlap/Jaccard, topic similarity, and independent-sentence count per window.

## Before any SOTA claim

The current paired-alignment runs are not official-candidate compliant under a literal ≤10-epoch interpretation because all matched arms exceed 10 passes by 1,600 words and SINGLE_ORIG exceeds 10 passes by 4.488M words. Any official candidate must be rebuilt with exact accounting.

A promising arm should eventually be evaluated on the full nine-entry official coordinate (including SuperGLUE and AoA), not only the fast screen.
