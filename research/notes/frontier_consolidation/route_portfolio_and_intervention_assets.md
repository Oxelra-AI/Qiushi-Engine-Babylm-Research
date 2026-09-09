# route portfolio and intervention assets — Route portfolio, literature grounding, and prepared intervention assets

The proposed masking and consistency interventions are specified for comparison with the mature legal-tokenizer evidence. Their implementation readiness does not determine which scientific route is justified.

This note is CPU-only research state. It does not choose a route, does not
launch GPU work, and does not change any corpus, tokenizer, or recipe. The route
decision waits on the mature 70M/80M matched clean-vs-reinvest comparison
(`s50_t5/t11/t19` → `compare_legal_treatment_trajectory.py` →
`interpret_legal_treatment_pattern.py`).

## Decision Rule

The earlier analysis legal endpoints both miss 41.8 (spatial repair route status 41.258, byte-alphabet 40.704).
The score deficit alone does not tell us whether the *data mechanism* failed or
the *tokenizer coordinate* shifted. The mature matched comparison distinguishes:

- **broad_positive_reinvestment_survival** → compact-view reinvestment still
  transfers under the legal tokenizer; the deficit is coordinate/representation/
  optimization, not the data principle. Next work: legal representation or
  optimization, coordinated with 40k route. Do NOT launch a learning-signal
  objective merely because assets are ready.
- **selective_relation_sensitive_weakness** → relation domains (Supplement/EWoK/
  Entity) lag broad columns. Next work: one single-variable relation-weighted
  masking screen using the repaired `wwm_static_prior`.
- **broad_reinvestment_disappearance** → the compact-view data no longer helps
  under the legal tokenizer. Next work: explicit source-view consistency
  (a separate construction), not masking.

`interpret_legal_treatment_pattern.py` encodes this and refuses to label
a pattern until both 70M and 80M reinvest-clean pairs exist.

## Asset A — static relation-weighted masking (`wwm_static_prior`), repaired

- Trainer: `masking_curriculum_trainer_static_prior.py` (SHA `18f6975a...`).
- Prior: `static_token_mask_prior/static_token_mask_prior.json` (SHA
  `9e60a9bb...`), corpus-only, from the frozen legal 10M pool + spatial repair route status tokenizer.
- Repair (earlier analysis): group probabilities are normalized by WWM-group token lengths
  so the *expected selected-token rate* matches 0.15, not just the group rate.
  Deterministic 4096-row check: fixed=static=0.150000 token rate, zero clipping;
  `relation_only_v1` relation lift 1.592, high-prior lift 1.764; `relation_info_v1`
  milder (1.406/1.533).
- Full 64740-row deterministic expected-budget pass is the remaining CPU check
  before any launch (`expected_static_prior_budget.py --limit-rows 0`).
- Launch wrapper: `train_static_prior_model.py` (hashes + 100M word count
  now mandatory). Only change vs the spatial repair route status endpoint: masking_curriculum +
  static prior; everything else frozen.
- Use only if the mature pattern is **selective_relation_sensitive_weakness**.

## Asset B — source-view consistency map (feasibility → durable spans)

- Feasibility (`pair_consistency_feasibility.py`): 3005 changed rows,
  12145/12155 pair records both source+rewrite visible at seq256, 7 source-only,
  52 true-truncation rows.
- Durable span map (`build_pair_span_map.py` →
  `pair_span_map/pair_span_map.jsonl` + summary): per changed row, exact
  trainer-visible token ranges for source and rewrite of every selected pair,
  visibility flags, and counts. All 3005 rows align exactly (0 tiny-suffix); 2995
  rows have all pairs both-visible. Row token means: source 118.9, rewrite 81.4.
- This is a construction asset, NOT score evidence. It requires a new
  span-carrying dataset/collate + auxiliary-loss path, because the
  inherited trainer masking function sees only input_ids/attention_mask/word_group.
- Use only if the mature pattern is **broad_reinvestment_disappearance**.

## Literature grounding (read from saved Knowledge sources)

- **UDA** \cite{xie2019unsupervised}: consistency between predictions on original
  and augmented text, with a **stop-gradient fixed copy** for the target, weight
  λ≈1, optional confidence masking + prediction sharpening. Key lesson: consistency
  helps *most* when the augmentation is **valid** (label/semantics preserving) and
  **diverse**; its Theorem 1 ties label efficiency to augmentation connectivity.
  Our compact views are semantically noisy (density cleanqwen overlay medium riskhard: ~8-20% materially change the
  assertion), so a source-view consistency loss should be robust to imperfect
  views: prefer low-weight symmetric agreement + possibly confidence gating, not a
  hard equality constraint.
- **DeCLUTR** \cite{giorgi2021declutr}: MLM + InfoNCE over same-document spans
  (anchor/positive), summed losses, temperature τ=5e-2, mean pooling; in-batch
  and same-document hard negatives. Directly analogous to source↔rewrite as
  anchor↔positive from the *same* packed row.
- **ConSERT** \cite{yan2021consert}: NT-Xent over two augmented views, effective
  with as few as ~1000 texts; warns that even "None-None" contrastive training
  reshapes the space by pushing representations apart — relevant caution that
  in-batch negatives in tiny-data MLM pretraining may distort LM behavior. First
  consistency test should therefore avoid aggressive negatives.
- Net design guidance if Asset B is chosen: start with a **low-weight,
  stop-gradient symmetric agreement** between source and rewrite pooled
  representations (or masked-token logit distributions) on the ~3005 changed rows,
  keep MLM primary, add negatives only after a positive-only screen; keep it
  strictly separate from Asset A.

## Status of managed evidence (not polled)

- Clean 80M training; reinvest 70M/80M cheap eval;
  clean 20M/70M/80M cheap eval — all running; results pending.
- Full 64740-row expected-budget pass — running (CPU).
- Reinvest 20M reference present (mean7 39.6636); not decisive alone (  calibration: 70M/80M first informative).
