# earlier analysis — Route decision after strict ordered-prefix surface

## Scientific update from strict ordered target surface

The ordered-prefix route interpretation closed the cross-sentence ordered mechanism too strongly. The stricter strict ordered target surface measurement added local same-source neighbors and an independent selection/evaluation split:

- selection model: seed42 WWM at 80M;
- held-out measurement model: seed43 WWM at 100M;
- fixed suffix and target positions;
- variants include full, deleted, block-shuffled, random same-source, local-neighbor same-source, and cross-source prefixes.

The ordered-prefix surface is not empty:

- local-neighbor same + block selected 153/1200 targets;
- 68/1200 survived held-out measurement;
- survivors come from both exact 1M slice and broader official text.

But it is still not a scale-ready training route:

- after removing explicit target-in-prefix, local-leak, and repeated-capitalized proxies, only 10/1200 held-out local survivors remain;
- decoded survivors are mostly lexical, phrase, topical, numeric, or broad discourse continuation (`pounds`, `foot`, `discussing`, `crops`, `child`, `fourth`, `taken`, `north`), not clean entity-state update targets;
- all local-neighbor cases use distance 1 in original chunk order, so many positives are adjacent-continuation effects rather than reusable state transitions.

This supports a narrow statement: official text contains a small measurable ordered-prefix-dependent surface in mature WWM. It does not support broad CPC, contrastive, RTD, or continuation training. Those objectives would still train mostly the much larger locally predictable/topic-continuity population and repeat the failure mode already observed.

## Route decision

Do not build another broad cross-sentence auxiliary loss from the strict ordered target surface surface.

Preserve the strict ordered target surface rows as an analysis surface for future mechanisms:

- `data/strict_ordered_target_surface_rows.csv`
- `data/strict_ordered_target_surface_summary.json`
- `data/strict_surface_filter_analysis.json`

A further semantic classification of all 68 survivors is useful only if a future route explicitly targets ordered discourse/entity-state learning. It should not block the next training direction, because the conservative clean surface is only 10/1200 and is not aligned tightly enough with the current Entity/EWoK/GlobalPIQA gap.

The next training line should move to mechanisms that do not rely on broad ordered prefix-to-suffix credit assignment. The strongest immediate candidate remains factorized WWM cadence, but only with a truly matched comparison.

## Matched cadence experiment requirements

The ordered prefix route decision cadence trainer smoke worked mechanically, but the first proposed comparison was not yet safe.

Important correction: the prior data event binding panel matched WWM baseline already has `max_position_embeddings=512`, so there is no large 512-position-capacity mismatch. The ordered prefix route decision smoke had config `max_position_embeddings=520` because `build_model()` uses `max(max_position_embeddings, max_seq_length + 8)`. The actual parameter difference is small (+3,840), not the large capacity difference stated in ordered prefix route decision. Still, a cadence run reaching 512 tokens should be compared against a fixed-length reference built by the same trainer and model constructor, not directly against prior data event binding panel.

For a clean 1M cadence comparison, run four arms with `scripts/cadence_wwm_trainer.py`, all using the same selected official 1M examples, seeds 42/456/789, batch 128, lr_total_steps 49, tokenizer baseline16k, DeBERTa-v2 8x480, `max_seq_length=512`, and `max_position_embeddings=512` so all models instantiate as 520-position configs and 34,471,264 params:

1. fixed-256 reference: `seq_length=256`, no sequence schedule, `mask_prob=0.15`, `max_seq_length=512`;
2. length-only: `seq_length=128`, `seq_len_schedule='0.0:128,0.4:256,0.7:512'`, `mask_prob=0.15`, `max_seq_length=512`;
3. mask-decay-only: `seq_length=256`, no sequence schedule, `mask_prob_start=0.30`, `mask_prob_end=0.15`, `max_seq_length=512`;
4. combined: length schedule plus mask decay, `max_seq_length=512`.

Record for each arm:

- exact word exposure and training steps;
- source-word composition and consumed example order hash/equality;
- untruncated and kept token totals at max_seq_length=512;
- per-step active token counts and masked target counts;
- final model parameter count and config hash;
- direct chck_1M fast-profile results for BLiMP, Supplement, EWoK, Entity, COMPS, and Reading.

Interpretation of a 1M cadence result:

- If a cadence arm improves Entity/EWoK/GlobalPIQA-related fast columns while preserving Supplement/COMPS/Reading relative to the fixed-256 reference, run seed43 or a modest 3M two-arm follow-up.
- If gains come only from extra kept tokens under `max_seq_length=512`, then the mechanism is longer untruncated exposure, not cadence; that may still be valuable, but it must be named and compared against fixed-512 reference before scaling.
- If mask-decay helps early fast columns but damages Supplement/Reading, do not scale it.
- Do not compare cadence arms directly to the prior data event binding panel fixed-256 `max_seq_length=256` run for route decisions; prior data event binding panel remains a historical baseline, not the matched reference for this factorized comparison.

## Current-best endpoint recheck remains unfinished

Before endpoint-level SOTA statements, still recover or rerun the exact-checkpoint recheck for seed43 chck_80M versus chck_100M on the columns used in the current-best coordinate. The new best verification and assessment coordinate remains useful, but exact direct checkpoint paths should be confirmed after the known local checkpoint-loading issue discovered earlier.

## Next concrete work

The proposed implementation is a small runner/evaluator for the four matched 1M cadence arms above and run the fixed-256 reference plus one active arm first if runtime is tight. The matched fixed-256 reference is essential; without it, cadence results are not interpretable.
