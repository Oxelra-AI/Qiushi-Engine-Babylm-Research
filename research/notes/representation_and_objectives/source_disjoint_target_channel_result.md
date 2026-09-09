# source disjoint target channel result — source-disjoint held-out test of the compact source-absent target channel

## Source-Disjoint Target-Channel Question

The target selective source absent result/226 target-selective source-absent effect was measured on the same 12,155 compact
pairs used as training inputs. Even probe words never selected by WWM stayed visible as input
text, so the effect could be local reformulation learning inside repeated rows rather than an
abstract, transferable denoising channel. The decisive test: does the same fixed target-type
contrast survive on **compact pairs whose source sentence never entered training**?

## Design (evaluation-only, no new training)

- Reservoir of held-out pairs: the accepted medium-compact rewrites at
  `experiments/archive/frontier_consolidation/data/medium_compact_analysis/medium_compact_ws_rows.jsonl`
  (SHA `4768758e...`), 21,465 analyzed rows, 18,682 accepted. The 12,155 training pairs are the
  `selected_compact_reinvest_pairs.jsonl` subset (SHA `d2a3110c...`).
- Disjointness tiers built by comparing each accepted rewrite against the training pair identity
  (`key = sid|doc`, exact `source_text`, `doc_id`):
  - `source_disjoint_quality`: 1,509 pairs whose exact source sentence never appeared in training
    input and whose content_recall ≥ 0.75 and compact length ratio ∈ [0.5, 1.05] (matches training
    pair quality: candidate content_recall mean 0.850). Source-absent-content candidate events: 974.
  - `doc_disjoint_quality`: 105 pairs whose entire source document never appeared, same quality
    filter. Source-absent candidates: 71 (small; corroboration only).
  - `doc_disjoint_all_accepted`: 475 doc-disjoint accepted pairs, no quality filter (lower quality
    tail). Source-absent candidates: 1,012.
- Held-out target events matched to the in-training fixed probe by target category, BPE piece
  length, tokenizer support (computed under the exact realized 20M own-visible input stream), and
  compact-side position. Matcher output for `source_disjoint_quality/source_absent_content`:
  974 events, 710 pairs, support/position distributions close to the in-training probe.
- Arms evaluated (existing 20M checkpoints, no retraining): `full` (crossview v2 20M result own-visible full loss),
  `drop_abs` (target selective source absent result drop source-absent content labels), `drop_copied_tok` (target selective source absent result token-piece
  matched copied drop), `drop_copied_word` (wholeword copied control plan whole-word matched copied-content drop).
- Central contrast: `drop_abs − drop_copied_word` piece-weighted loss with pair-cluster bootstrap.
  Positive on source_absent_content = removing source-absent labels hurts held-out source-absent
  denoising more than removing whole-word matched copied-content labels.

Artifacts:
- Script: `experiments/archive/representation_and_objectives/scripts/source_disjoint_target_type_probe.py`
- Result: `experiments/archive/representation_and_objectives/data/source_disjoint_target_probe/source_disjoint_target_type_probe.json`
- Preflight: `.../source_disjoint_target_type_preflight.json`
- Per-event losses: `.../source_disjoint_event_losses.jsonl`
- Held-out event list: `.../source_disjoint_probe_events.jsonl`
- Training-input token support cache: `.../training_input_token_freq_20M.json`

## Result: the source-absent channel survives on source-disjoint pairs and is category-specific

`drop_abs − drop_copied_word` at chck_20M, piece-weighted nats, [p025, p975], fraction above zero:

| eval set | retained_content | source_absent_content | function_other |
|---|---:|---:|---:|
| source_disjoint_quality | **−0.0470** [−0.069,−0.024] 0.0 | **+0.0796** [+0.031,+0.125] **1.0** | **−0.0452** [−0.069,−0.022] 0.0 |
| doc_disjoint_quality (n=71) | −0.0029 [−0.044,+0.041] 0.48 | **+0.1141** [−0.063,+0.313] 0.912 | −0.0031 [−0.057,+0.053] 0.47 |
| doc_disjoint_all_accepted | −0.0525 [−0.076,−0.029] 0.0 | **+0.0356** [−0.008,+0.076] 0.948 | −0.0596 [−0.089,−0.030] 0.0 |

Against the token-piece control and the full-loss baseline, the source-absent effect is if anything
larger and the retained/function categories stay near zero or negative:

- `drop_abs − drop_copied_tok`, source_disjoint_quality/source_absent_content: **+0.1404** [+0.095,+0.186], frac>0 1.0.
- `drop_abs − full`, source_disjoint_quality/source_absent_content: **+0.1077** [+0.063,+0.152], frac>0 1.0; retained −0.039, function −0.002.

Piece-weighted category loss (chck_20M, source_disjoint_quality): source_absent_content is 7.0046
for full, **7.1123 for drop_abs** (worse), 6.9720 for drop_copied_tok, 7.0328 for drop_copied_word.
Removing source-absent labels raises held-out source-absent loss; removing copied labels does not.

## Scientific reading

1. The target selective source absent result/226 source-absent target-type channel is **abstraction beyond repeated-row
   exposure**, not local reformulation memorization. All probe source+rewrite rows here were absent
   from training inputs, yet training on sparse source-absent compact-content labels still improves
   later denoising of unseen source-absent compact-content targets, and this cannot be replaced by
   an equal amount of whole-word matched copied-content label learning.
2. The effect is **category-specific**: it is positive only for source-absent content and moves the
   opposite way for retained content and function words. This rules out a generic "drop_abs is a
   better model" explanation; the improvement is localized to the abstractive compact-content
   prediction distribution.
3. Boundaries preserved: this remains within the teacher-generated FineWeb compact reservoir, one
   tokenizer, and one DeBERTa masked-denoising family at 20M words. The result shows the channel is
   a real, transferable-within-distribution denoising mechanism; it does not yet establish transfer
   across data source, architecture, or the 100M compact-view endpoint (+2.4164 equal7).
4. The strict doc-disjoint quality subset (n=71) agrees in sign and magnitude but is too small to
   settle on its own; the larger source-disjoint quality set (974 events, 710 pairs, frac>0 = 1.0)
   is the load-bearing population.

## What this justifies next

This survival on source-disjoint pairs is the condition the strategist set for advancing the
channel into exact-parity packed-trajectory work. The next decisive question is whether this local
source-absent denoising channel connects to the 100M compact-view competence gain (Supplement,
relational EWoK domains) under the historical packed geometry, tested with an exact-parity
target-selective or loss-mask intervention that preserves identical text, tokenizer, WWM schedule,
exposure accounting, and packed sequence layout. The early 20M external surfaces (wholeword copied control plan:
drop_abs > whole-word copied drop on cheap7 +0.797, EWoK +0.82, Supplement +1.3, GlobalPIQA +3.475)
are consistent but must not be treated as endpoint evidence because the compact-view advantage
matured late.
