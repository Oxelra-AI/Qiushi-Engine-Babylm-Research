# reference and margin state reference and margin state

This note records the scientific representation fixed before the in-flight breadth and clean jobs terminate.  It should be read with the actual readout files:

- Reference decomposition script: `scripts/reference_decomposition_readout.py`
- Current reference output: `data/reference_decomposition_readout/reference_decomposition_summary.md`
- Direct-margin script: `scripts/margin_domain_window_readout.py`
- Direct-margin output: `data/margin_domain_window_readout/margin_domain_window_summary.md`
- Entity margin-numops output: `data/entity_margin_numops_common10_80/entity_margin_numops_summary.md`

## Reference arm choice

The broad mechanism-bearing reference is **B**, not **R**.

Definitions for MAX dose:

- `V`: compact source-conditioned view packet stream.
- `R`: exact repeat stream.  This measures duplicate recurrence, but it fuses two differences: the text is source-related and also exactly duplicated.
- `B`: same-population independent-sentence breadth stream.  This preserves source population, rows, words, passes, and packing while removing both exact duplication and own-source relatedness.  It is the current best broad reference for asking whether the companion must be about its own source.
- `C`: clean filler stream.  This is the absolute fixed-budget placement; it says whether the full packet stream beats ordinary clean finite experience under a specified geometry.

Therefore the live broad identities are:

```text
V-C = (V-B) + (B-C)
V-R = (V-B) + (B-R)
B-C = (B-R) + (R-C)
```

The old earlier analysis ex-Entity MAX common-window numbers against the older 1x-geometry clean arm were:

- cheap6 ex-Entity `V-C = +0.5981`, `R-C = +0.5345`, `V-R = +0.0636`.
- cheap5 ex-Entity `V-C = +0.6194`, `R-C = +0.5231`, `V-R = +0.0963`.

Thus roughly 0.53 of the roughly 0.60 broad ex-Entity positive total sits in `R-C` if `R` is used as the internal reference.  That makes `V-R` a poor broad mechanism carrier: against exact duplication, compact re-expression currently adds little ex-Entity on the old-clean common window.  The question that can still produce a meaningful principle is whether `V-B` is broad and positive.  If `V-B` is near zero outside Entity, the broad total is stream/corpus composition or geometry rather than source-conditioned companions.  If `V-B` is large and broad, within-packet referential coherence becomes the mechanism-bearing quantity.

## Current interim breadth evidence

The in-flight scorer has already made some breadth files visible, so the reference and margin state reference script currently sees **one** broad breadth checkpoint at 80M.  This is not the terminated managed-task result and must not be treated as a final ladder.

At that visible 80M checkpoint only:

- ex-Entity `V-B = -0.0620`, while `B-C_old = +0.8960` and `V-C_old = +0.8340`.
- cheap6 `V-B = +0.4650`, but the family rows show this is dominated by Entity: `V-B` Entity is +3.10, while ex-Entity is slightly negative.
- Family `V-B` at 80M: BLiMP -0.87, Supplement +1.82, EWoK -0.15, Entity +3.10, COMPS -1.18, Reading +0.07.

This single checkpoint is useful for arithmetic and for sharpening the risk: the apparently positive cheap6 `V-B` can be almost entirely Entity-carried.  It is not enough to close the companion mechanism.  The correct next read is the terminated breadth ladder, then rerun `reference_decomposition_readout.py` and inspect ex-Entity and per-family `V-B`, `B-C`, and the identities across checkpoints.

## Direct-margin evidence

The delivered sampled direct-margin ladder scored 3,480 margin rows: 2,160 Entity and 1,320 EWoK.  It supports the operation-sensitive reading of Entity `V-R`, but it does not establish a clean monotone dose law.

From `data/margin_domain_window_readout/margin_domain_window_summary.md`:

- Entity common10_80: dose1 all +0.4197, zero -0.4345, nonzero +0.5906; dose2p64 all +0.3605, zero -1.9462, nonzero +0.8218.
- Entity endpoint80: dose1 all +0.8673, zero -0.1729, nonzero +1.0754; dose2p64 all +0.9792, zero -2.6744, nonzero +1.7100.
- The MAX endpoint80 nonzero-minus-zero gap is +4.3844 margin units, aligned with the official prediction-stratum split.
- EWoK all-domain margins are unstable: common10_80 means are +0.0599, -0.2117, +0.0090 for 1x/mid/MAX; endpoint80 means are +0.3385, +0.0111, -0.2960.

This threshold-free readout strengthens the conclusion that Entity `V-R` is tied to update/non-update allocation and not broad reusable record formation by itself.  It also keeps the broad official stable-family breadth readout central, because the sampled margins do not provide an independent broad positive carrier.

## Consequence for expensive work

Do not launch the permuted-companion training from the current Entity evidence alone.  It becomes high-value if broad ex-Entity `V-B` survives, because it equalizes compact-rewrite fertility and multiset properties that B cannot match while breaking own-source correspondence.  If broad ex-Entity `V-B` is near zero, permuting compact companions would answer a mechanism that has already failed to appear broadly.

The in-flight MAX-geometry clean controls remain useful, but they answer a different question: absolute placement against matched clean experience.  A positive `V-C` with `V-B≈0` supports a stream/corpus composition or geometry effect, not source-conditioned companion learning.  A positive `V-B` with weak or null `V-C` would still be a relative allocation principle but not an absolute clean-data improvement.  Both distinctions must be preserved.

## Scope facts for `V-B`

Existing pre result reading order and replication plan records already specify what `B` does and does not equalize:

- Whole-pool 10M accounting (`data/pool_token_accounting/pool_token_accounting.md`): `max_view` and `max_breadth` are exactly word- and row-matched, but `max_breadth` has -0.2864% legal16k tokens and -0.2831% visible seq256 tokens relative to `max_view`; WWM groups are almost identical (+0.0086%). `max_clean` differs in the other direction: +0.4095% legal16k tokens, +0.3326% visible tokens, and -0.0638% WWM groups versus `max_view`.
- Changed-block record (`data/breadth_arm_geometry_audit/breadth_geometry_audit.md`): `max_breadth` has -2.602% legal16k tokens and -2.515% visible tokens versus `max_view` inside the changed rows, while WWM groups remain nearly equal (+0.075%). Breadth rows also have about 2.03 fewer sentence-final marks per changed row.
- Replaced companion population: compact rewrites and breadth sentences each spend 428,122 words, but rewrite fertility is 1.5286 legal16k tokens/word while selected breadth is 1.4308; unigram JS(rewrite,breadth)=0.1242 bits. Selected breadth uses 15,123 whole sentences from 4,871 docs; 94.9% of selected docs overlap MAX source docs, so this is same-population sentence breadth, not new-document breadth.

These facts make `B` the best current broad reference for own-source relatedness, but not a pure text-multiset control.  If broad `V-B` is positive, the next compact-text-held reference is the already materialized permuted-companion stream, because it preserves the compact rewrite multiset and nearly matches token geometry while breaking own-source correspondence.  If broad `V-B` is near zero, these B-versus-V text-population differences are not the main obstacle; the companion mechanism simply has not appeared broadly in the official stable-family scores.
