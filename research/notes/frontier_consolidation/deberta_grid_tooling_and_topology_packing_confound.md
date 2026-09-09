# deberta grid tooling and topology packing confound — DeBERTa grid interpreter, coordinate verification, and topology packing confound

No training, official-compatible evaluation, HF upload, or leaderboard submission occurred. All work is CPU/file-only and prepares/validates evidence for the pending runtime-managed DeBERTa common-grid tasks and constrains the reciprocal/topology direction.

## Pending Comparisons

- protected reference scale1.75 seed43022 DeBERTa common 70M–100M grid (GPU0), output `data/selected_trajectory_eval_scale1p75_seed43022_reference_common2M` (skips cached 80M/100M).
- scale1.25 seed43022 dense DeBERTa common grid (GPU1), output `data/selected_trajectory_eval_scale1p25_seed43022_dense_common2M` (all 16 endpoints).
- Still pending launch when a GPU frees: scale1.75 seed43122 dense common grid to `data/selected_trajectory_eval_scale1p75_seed43122_dense_common2M`. Dry plan validated at `data/seed43122_common_grid_launch_plan/selected_eval_plan.json` (all endpoints present).

## Coordinate verification (deberta grid tooling and topology packing confound, done)

`scripts/deberta_checkpoint_coordinate_audit.py` → `data/deberta_checkpoint_coordinate_audit/`.

Result OK for all three trajectories:
- All share identical tokenizer.json SHA `a9cbb830495cb92bbb2996adc256207746282ee67f4f8d40a4f646a634ec139a`, tokenizer_config, special_tokens_map, and the same custom modeling code `adapter_scaled_modeling.py` SHA `9fd4ef104f5f8e7d458c7ed10b996d238a6af53a866e8dea907a3cc0e4e156d2`.
- All are `AdapterDebertaV2ForMaskedLM` with adapter auto_map (trusted-code loading required), 35,463,008 params, vocab 16384, geometry 8×480×8, exposure 100M, source words total 100M, tokenizer label `compliant16k_reinvest10M`, same corpus `cleanqwen_fineweb_compact_view_reinvest_100M.jsonl`.
- Adapter scales differ exactly as intended: reference 1.75, scale1.25 1.25, seed43122 1.75.
- Reference has 100 checkpoint dirs; both dense runs have 50; all common-grid endpoints present in every run.

This confirms the compared trajectories share the legal representation coordinate; the intended contrasts (adapter amplitude for scale1.25; joint init+mask stream for seed43122) are real, not confounded by tokenizer/code/data/geometry.

## Score interpreter (deberta grid tooling and topology packing confound, ready, smoked)

`scripts/interpret_deberta_common_grid.py` → outputs `deberta_common_grid_interpretation.{json,md}`.

Improvements after independent_review verification:
- Reports both the loose near-best span and the **contiguous** near-best band containing the actual maximum (the loose span can overstate width when qualifying points are disconnected).
- Adds **endpoint-resolved** adapter energy: per-checkpoint effective up/stock ratio, 82M and 100M energy, 82→100 energy delta, and 70→100 slope, plus endpoint ratios to reference.
- Keeps cheap7, cheap6(no GlobalPIQA), cheap5(no GlobalPIQA/Reading), EWoK/Entity, syntax/COMPS, volatile summaries, per-column peaks, and wide-family-positive detection (≥4 positive columns, positive cheap6 and cheap5, volatile gain share ≤0.5).

Smoked on synthetic complete grids with the real adapter-norm JSON; passes `--strict-complete`.

Command after all three selected outputs exist and pass `selected_mlm_integrity_check.py`:

```bash
PYTHONDONTWRITEBYTECODE=1 python -B experiments/archive/frontier_consolidation/scripts/interpret_deberta_common_grid.py \
  --trajectory scale1p75_seed43022_reference=experiments/archive/frontier_consolidation/data/selected_trajectory_eval_scale1p75_seed43022_reference_common2M/selected_trajectory.json \
  --trajectory scale1p25_seed43022_dense=experiments/archive/frontier_consolidation/data/selected_trajectory_eval_scale1p25_seed43022_dense_common2M/selected_trajectory.json \
  --trajectory scale1p75_seed43122_dense=experiments/archive/frontier_consolidation/data/selected_trajectory_eval_scale1p75_seed43122_dense_common2M/selected_trajectory.json \
  --reference-label scale1p75_seed43022_reference \
  --adapter-json experiments/archive/frontier_consolidation/data/adapter_norm_trajectory/adapter_norm_trajectory.json \
  --out-dir experiments/archive/frontier_consolidation/data/deberta_common_grid_interpretation \
  --strict-complete
```

## Interpretation bounds (from deberta grid tooling and topology packing confound independent_review verification)

The DeBERTa grid can support only two bounded readings, not a general principle:
- **scale1.25 vs reference**: seed/mask-matched adapter-amplitude contrast. earlier analysis shows scale1.25 has ~0.78× the reference effective up/stock ratio, a real lower-residual-energy trajectory. A convincing timing result needs an internally peaked later/broader high band with reduced late falloff and a recognizably similar family profile — not merely a rising curve to the 100M boundary (which is ambiguous with underfitting). Inspect 90M–100M shape.
- **scale1.75 seed43122 vs reference**: joint initialization + mask-stream (and possibly dataloader-order) robustness; a single alternative draw, not a variance estimate. Reappearance near 78M–86M with neighboring and family-wide support is useful replication only.

Remaining cheap checks before any further training: evaluator-version/backend/parser equivalence between cache-seeded reference points (80M/100M) and freshly scored points; actual optimizer/LR/step/RNG coordinates at each nominal endpoint; and neighboring-checkpoint/family consistency in place of missing item-level uncertainty.

## Topology 2×2 packing confound (deberta grid tooling and topology packing confound, new, consequential)

`scripts/causal_topology_packing_audit.py` → `data/causal_topology_packing_audit/`.

Under the actual GPT trainer geometry (concatenate rows + EOS, chunk to 256, per-epoch whole-row shuffle with `random.Random(seed+epoch)`):

- The two units of each semantic pair **never share a 256-token chunk** and are **never adjacent** after packing (pair units share-chunk fraction 0.0; adjacent-doc fraction 0.0) at epochs 0/1/2. The intended reciprocal/oneway cross-view conditioning therefore lives strictly **inside each pair-unit row**, not across the pair.
- About **19–20%** of pair-unit rows **cross a chunk boundary**, so for those rows the tokens after the boundary lose the earlier within-row (source or view) context. This is a real exposure-geometry effect the row-word matching does not remove.
- The compact arm crosses boundaries **more often** than repeat (epoch0 cross-chunk fraction: compact_oneway 0.2021 vs repeat_oneway 0.1870; compact_reciprocal 0.1958 vs repeat_reciprocal 0.1870), because compact pair rows are slightly longer (mean pair-row tokens 51.70 vs 49.09). This compounds the already-known +0.2216% compact active-token asymmetry.
- Topology (oneway vs reciprocal) changes active tokens/chunks by 0 within each semantic arm, as intended; the compact-vs-repeat token gap is the same (+31,744 active tokens) in both topology cells.

Consequence for the reciprocal/topology direction: even the corrected 2×2 does **not** cleanly isolate "reciprocal cross-view conditioning" because (a) that conditioning is confined within-row and partly severed by chunk boundaries, and (b) compact rows differ from repeat rows in length/boundary-crossing beyond token count. If evidence ever revives this direction, the realization needs a boundary-aware construction (e.g., pad/segment so each pair-unit row is not split across chunks, or an example-as-row trainer), and the interaction must still survive removal of GlobalPIQA/Reading and the copied-token overlap stratum (selected-subset mean word-multiset overlap 0.8295).

## Standing constraints unchanged

- Protected public endpoint: scale1.75 chck_82M, displayed 41.94, HF `leslie721007/babylm-strict-small-scale1p75-chck82`. Strongest local endpoint: coherent86 α=0.75, projected Overall(AoA0) 42.1210, HF `leslie721007/babylm-strict-small-coherent86-alpha075`. No new submission.
- Trusted custom-code loading remains mandatory for all adapter DeBERTa checkpoints.
- Do not reopen alpha/anchor/retention/edit/IG/graph/coherence-margin routes.
