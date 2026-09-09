# legal deficit representation vs candidate alignment — Is the legal-tokenizer deficit a representation problem, and does any candidate fix align with it?

CPU-only, evidence-only. Does not poll or infer the mature clean-vs-reinvest tasks
(`s50_t5/t11/t19`), does not launch GPU work, does not change corpus/tokenizer, and does
not tune a tokenizer from evaluation data.

## Three Analyses

### A. Tokenization geometry vs legal deficit (`tokenizer_fragmentation_deficit_alignment`)
Read the old inherited (non-submittable) tokenizer, the two legal 16k tokenizers, and the
legal 24k/32k/40k/support-floored candidates (all fit only on the allowed 10M pool), and
correlated per-UID tokens-per-word geometry with the legal deficit decomposition and prior mismatch legal_step35−old_ref score
deltas.

- **Fragmentation does not explain the deficit.** Correlation between the spatial repair route status-vs-old
  tokens/word inflation and score delta is weak and mostly wrong-signed: ALL Pearson
  −0.105, BLiMP −0.047, EWoK **+0.247** (worse-signed). The worst losses have almost no
  inflation: `wh_questions_object_gap` +0.0009 tpw / −17.34; `tough_vs_raising_1` −0.0096
  tpw / −10.97; `material-dynamics` −0.0215 tpw / −8.83. QA congruence is the partial
  exception (+0.118/+0.130 tpw). **No evaluated string overflows 256 tokens** for any
  tokenizer, so eval-input truncation is not the cause.
- **The legal 40k candidate lowers tokens/word roughly uniformly** (mean 40k−spatial repair route status tpw
  −0.133), so it does not selectively target the losing subtasks. The "gap-closed"
  percentage is numerically unstable where the spatial repair route status−old denominator ≈ 0 (values of
  164%, 511%, negative), so it must not be read as a recovery estimate.

### B. Token support and identity vs legal deficit (`tokenizer_support_identity_alignment`)
Counted every subword's occurrence in the allowed 10M pool for each tokenizer, then per
UID measured (i) mean log pool-count and fraction of eval tokens with pool count <50/<100,
and (ii) token-identity change vs the old tokenizer (multiset Jaccard, new-token fraction).

- **No simple low-support or identity story across all 83 UIDs.** ALL correlations are
  weak and mixed in sign: `mean_log10_pool_count` Pearson −0.154 (wrong sign), `frac<50`
  +0.104, `new-vs-old fraction` Pearson −0.041 / Spearman −0.132, Jaccard +0.041/+0.128.
- **Supplement is a small, internally consistent exception (n=5 only).** There, higher
  token-identity change and lower support align with larger loss: new-vs-old fraction
  Pearson −0.75, frac<50 −0.53, 40k-vs-spatial repair route status new fraction −0.78. This is suggestive that
  QA-congruence sensitivity may be partly a tokenization-identity effect, but with only 5
  UIDs it cannot support a route by itself.
- **Legal 16k tokenizers actually have BETTER low-frequency support than the old
  tokenizer** on the pool (frac vocab <50: old 0.1275 vs spatial repair route status 0.0803), so "legal
  tokenizer has worse support" is false at 16k. Larger legal vocabularies (32k 0.53, 40k
  0.63 of vocab below 50 count) trade support for shorter sequences.

### C. Mature-trajectory subtask interpreter (`mature_treatment_subtask_interpreter`)
Built and validated a not-ready-safe interpreter that, once the mature clean pairs exist,
computes reinvest-minus-clean BLiMP/Supplement UID and EWoK-domain deltas, EWoK
relation-vs-property split, and the alignment of those treatment deltas with the existing
`relation_only_v1`/`relation_info_v1` pressure map. It currently reports only the delivered
reinvest side (20M mean7 39.6636, 70M mean7 42.6086) and refuses to label a route until
both clean 70M and 80M pairs arrive.

## Synthesis for the route choice (does not replace the mature-trajectory decision)

legal deficit decomposition and prior mismatch converge on one conclusion: **the legal-tokenizer deficit is a structured
representation/optimization effect, not a fragmentation, truncation, low-support, or
token-identity artifact, and it is not concentrated on relation content.** Independent
verification (independent_review) agrees and adds the concrete falsification patterns below.

Implications for the prewritten three-way route rule:
1. **relation_only_v1 static prior is mismatched twice over.** legal deficit decomposition and prior mismatch showed positive
   pressure–deficit correlation (wrong direction); legal deficit representation vs candidate alignment shows the loss is dominated by
   binding/filler-gap syntax and QA congruence, which the relation lexicon does not target.
   Even if the mature deltas look "relational," running `relation_only_v1` would amplify
   tokens away from the losses. A masking route needs a **binding/predicate/QA-targeted**
   prior with its own corpus-pressure alignment evidence — not the existing asset.
2. **Legal representation/optimization stays the most consistent hypothesis**, but NOT via
   tokens/word. Candidate mechanisms not yet ruled out: merge inventory / lexical
   allocation, embedding estimation under 10M words, WWM target granularity, parameter
   allocation, and optimization. legal-40k package is a valid test of several of
   these at once but is confounded by (a) the accumulated-microbatch trainer (realized-
   gradient, not objective, difference) and (b) 45.8M vs 34.5M params with heavy
   low-support embedding breadth (90.7% of new-over-16k units occur <50 times).
3. **Source-view consistency is neither supported nor rejected** by legal deficit decomposition and prior mismatch, because
   aggregate benchmark deltas cannot see view agreement. It becomes plausible only if the
   mature vector is complementary (clean stronger on syntax/QA, reinvest stronger on
   knowledge columns) at both 70M and 80M AND measured paired-view disagreement predicts
   the losses.

## Falsification patterns to read in the mature 70M/80M reinvest-minus-clean vector
- **Preserve reinvest + legal representation/optimization:** positive, stable
  reinvest−clean on Entity/COMPS/GlobalPIQA/Reading/EWoK-knowledge at both 70M and 80M,
  with residual syntax/QA weakness; UID-level 70M and 80M deltas positively correlated.
- **Binding/predicate masking (NOT relation_only_v1):** stable weakness concentrated in a
  corpus-targetable predicate/event/QA class that aligns with a *new* prior's pressure map,
  while already-strong spatial relations do not share the deficit.
- **Source-view consistency:** clean-vs-reinvest advantages oppose by column and recur at
  70M/80M, and paired-view disagreement on proposition-equivalent pairs predicts losses.
- **Broad disappearance / reinvest below clean everywhere:** reconsider whether the data
  mechanism survives the legal coordinate at all.

## Artifacts
- `experiments/archive/frontier_consolidation/scripts/tokenizer_fragmentation_deficit_alignment.py`
- `experiments/archive/frontier_consolidation/data/tokenizer_fragmentation_deficit_alignment`
- `experiments/archive/frontier_consolidation/scripts/tokenizer_support_identity_alignment.py`
- `experiments/archive/frontier_consolidation/data/tokenizer_support_identity_alignment`
- `experiments/archive/frontier_consolidation/scripts/mature_treatment_subtask_interpreter.py`
- `experiments/archive/frontier_consolidation/data/mature_treatment_subtask_interpreter`
- independent_review memo: `data/external/independent_review01_verifier1_integration.md`

## Unresolved Comparison
Clean 80M training, reinvest 70M/80M cheap evaluation and clean 20M/70M/80M cheap evaluation remain pending. Route selection depends on both mature clean comparisons; the representation-versus-candidate analysis constrains their interpretation.
