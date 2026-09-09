# legal40k decision and next route — Legal40k two-seed decision and the genuinely different next route

## Authoritative legal40k official vectors (both delivered, hardened collation integrity PASS)

| task | legal40k mean | legal16k mean | inherited16k (non-submission) | Δ40−16 | Δ40−inherited |
|---|---|---|---|---|---|
| BLiMP | 67.795 | 66.407 | 66.268 | **+1.388** | +1.526 |
| Supplement | 60.137 | 57.524 | 62.614 | **+2.612** | −2.477 |
| EWoK | 50.738 | 49.632 | 52.714 | **+1.106** | −1.976 |
| Entity | 25.977 | 27.391 | 27.016 | **−1.414** | −1.038 |
| COMPS | 52.157 | 52.029 | 51.754 | +0.128 | +0.403 |
| SuperGLUE | 68.912 | 69.393 | 70.468 | −0.481 | −1.557 |
| GlobalPIQA | 33.415 | 37.357 | 35.379 | **−3.942** | −1.964 |
| Reading | 7.894 | 8.043 | 8.553 | −0.149 | −0.660 |
| AoA | 0.0 | 0.0 | 0.0 | 0 | 0 |
| **Overall** | **40.780** | **40.864** | **41.641** | **−0.084** | **−0.860** |

- seed43022 Overall 41.1406 (margin vs 41.8 = −0.659); seed43122 Overall 40.4201 (margin −1.380).
- Both seeds below 41.8. Mean essentially tied with legal16k. **Legal40k did NOT solve the compliant SOTA problem.**

## Decisive mechanism reading

Legal40k is not a null result — it is a clean **representation trade-off**:
- Wider vocabulary + shorter segmentation + extra embedding capacity **recovered the support-rich linguistic surfaces** that legal16k lost: BLiMP +1.39, Supplement +2.61, EWoK +1.11.
- But it **collapsed the support-thin / rare / cross-lingual surfaces**: GlobalPIQA −3.94 (parallel 26.2→22.3 for BOTH seeds; nonparallel wildly seed-unstable 47 vs 42), Entity −1.41 (hardest `move_contents_0_ops`/`regular_0_ops` tracking dropped most).
- These two movements cancel → flat Overall.

This exactly matches the tokenizer support spectrum token-support spectrum: legal40k has 63.2% of used vocabulary below 50 pool occurrences (median support 33 vs legal16k 140). Rare-token surfaces (GlobalPIQA parallel is cross-lingual completion; Entity 0-ops needs stable surface segmentation of novel object names) are damaged by under-trained subword units. Legal16k is the mirror image: well-supported units but coarser segmentation hurting Supplement/EWoK.

**Neither pure-16k nor pure-40k legal tokenizer reaches the inherited 42.03**, because the inherited tokenizer (learned on 100M Strict) had both segmentation richness AND support. A legal tokenizer trained on only 10M cannot get both at a single flat vocab size — this is a genuine sample-efficiency problem in the representation layer.

## Convergence Across Experiments

matched aoa mincontext discrepancy audit-tokenizer clean-control confirms the **compact-view reinvestment mechanism survives strongly on the legal recipe**: 80M cheap-column reinvest-vs-clean +1.49 mean7 (Supplement +2.06, EWoK +2.03, Entity +2.33, COMPS +0.65). So the 42.033 mechanism is NOT tokenizer leakage — compact semantic views + reinvestment genuinely improve learning. The remaining gap to SOTA is the **general legal representation deficit**, not the data mechanism. Independent analyses converged on support-floored tokenizers as the proposed repair; the complementary design uses minfreq50 / equal-word MLM credit.

## The genuinely different next route (not a vocab-size sweep)

Build a **support-floored / merge-quality-controlled legal tokenizer** that keeps 40k's fine segmentation for frequent linguistic material while flooring rare merges so every retained unit has enough pool support to be well-trained — protecting GlobalPIQA/Entity while retaining the Supplement/EWoK gains. The tokenizer support spectrum minfreq tokenizers already exist:
- **minfreq25**: vocab 29,529, tokens/word 1.4139 (between failed 16k=1.4669 and 40k=1.3943), median used support 105-ish.
- minfreq50: vocab 19,609, tokens/word 1.4484.

To avoid duplicating companion analysis and create an adversarial contrast:
- **companion analysis takes minfreq25 (~29.5k)** — the middle support/segmentation operating point, on the confirmed compact_view_reinvest 10M/100M corpus, same 8×480 fixed-WWM accum recipe, both seeds → full official eval. Prediction to test: minfreq25 should keep most of legal40k's Supplement/EWoK gain (shorter than 16k) while NOT collapsing GlobalPIQA/Entity (support floor ≥25). If Overall > legal40k mean AND GlobalPIQA recovers toward legal16k, the support-floor mechanism is validated and points toward the optimal legal operating point.
- **companion analysis takes minfreq50 / equal-word MLM credit** (their stated choice).

This is one general legal representation repair chosen directly from the paired column pattern, satisfying the expensive-work admission bar: it will decide whether the compliant SOTA gap is a tokenizer-support operating-point problem (repairable by support flooring) or a deeper deficit needing architecture/objective change.

## Preserved protected evidence
- Inherited 42.033 remains non-submission mechanism evidence only (tokenizer learned on 100M).
- Compact-view reinvestment treatment is a confirmed positive mechanism across the paired studies. Keep it in all next routes.

## Training status and reassessment criterion

The minfreq25 H100 trainings were **not** launched in legal40k decision and next route, although the preflight passed. Two expensive same-family representation runs had failed to improve the full official result, so the scientific justification required reassessment before another run. The route was technically ready but not yet selected. Preflight/manifest assets:
- Launcher: `experiments/archive/representation_and_objectives/scripts/minfreq25_supportfloor_train.py`
- Preflight: `experiments/archive/representation_and_objectives/data/minfreq25_supportfloor_training/minfreq25_supportfloor_preflight.json`
- Seed manifests: `experiments/archive/representation_and_objectives/data/minfreq25_supportfloor_training/minfreq25_supportfloor_train_manifest_seed43022.json` and `...seed43122.json`

Preflight passed with exact expected hashes: tokenizer SHA `311e7a20cd8b20f512ab574b8d62b574b5106408e742c32b0bce152dfe8162e0`, 10M pool SHA `215944978157394dbecf2039f2c1e1806bfcbb9701421920f78605eb58975a23`, 100M stream SHA `3dd19f09deeca44d6b07340e70cd15baa4b5c7bffe0bd462ff37536e470e7691`, vocab size 29,529, fast tokenizer, special token IDs `[unk=0, bos=1, eos=2, pad=3, mask=4]`, and working `word_ids()`.
