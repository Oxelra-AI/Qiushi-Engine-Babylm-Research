# paired tail factor separation design — factor-separated paired tail continuations after corrected curriculum

## Why this replaces the contaminated lamb result and sleep restart readiness single "sleep" continuation

The contaminated lamb result and sleep restart readiness held continuation bundled two scientifically different interventions:

1. **fresh AdamW moments** after loading the 80M weights, and
2. **a restarted/reheated LR trajectory** over the 80M→100M tail.

The strategist note correctly points out that this opaque package would not tell us whether any movement came from moment reset or from renewed learning-rate energy. Prior optimizer/schedule packages have often redistributed relational surfaces without broad improvement, so if the corrected curriculum misses we need the smallest paired comparison that separates the factors rather than another endpoint whose meaning is unclear.

No new GPU work was launched. The corrected curriculum endpoint remains the active decision. The paired tail runs below are prepared only if that endpoint is below the leader region.

## Paired design now implemented

Script: `scripts/paired_tail_continuation_trainer.py`

Common contract for both arms:

- source weights: `training/runs/legal40k_accum_compact_view_reinvest_seed43022/hf_model/chck_80M`
- source checkpoint accounting: baseline earlier analysis, `actual_cumulative_word_exposure=80,034,368`
- legal tail data: same compact-view 100M stream, starting exactly after the 80M checkpoint row boundary
- tail rows: row index 518,144 onward
- tail size: 129,256 examples / 19,965,632 words
- effective steps: 505 at batch 256, microbatch 64
- final total exposure: exactly 100,000,000 words; no partial-example split
- tokenizer/model: legal byte-BPE 40k and DeBERTa-v2 8×480 baseline checkpoint
- masking/objective: fixed WWM 0.15
- both arms use fresh AdamW moments (`betas=(0.9,0.98)`, weight decay 0.01)
- both arms use the same `train_rng_seed=53023`, so WWM masks and dropout stream are replay-matched between the two arms; the uninterrupted 100M endpoint remains the anchor, not either tail arm

The dry-run manifests confirm identical first effective batch and CPU WWM mask hashes between arms:

- first tail batch token IDs: `f890c152a8eeaf871b018113b6df6c63630af61149421875e504f48711119d72`
- first tail batch word groups: `29a5b585632a3d6d15631ce9b5660f9322945c3197f9704200d2e43256a76f75`
- first CPU selected-mask positions: `319950e1e6aab858c2144c3f9848481f1a0222467281729ec7afda893010abf9`
- first CPU labels: `b26d52d27ffa568da53f7a1ba75431368663d1725009f746d059f47ee8800b56`
- first batch words: 39,780, exactly matching the uninterrupted baseline's logged earlier analysis batch words
- first batch masked tokens under the dry-run CPU mask replay: 8,232

## Arms

### Arm A — `residual_low_lr`

Question: what does a fresh-moment reset do if the LR path remains the baseline's residual low-LR trajectory?

The script reconstructs the baseline global-step cosine LR over steps 2025–2529. The baseline trainer logs `get_last_lr()` after `sched.step()`, so the LR recorded at earlier analysis is the optimizer LR prepared for the first post-80M update (global earlier analysis). The dry run now records both the applied update LR and the next LR after the update:

- source checkpoint logged LR after earlier analysis: `1.0720866372880134e-4`
- residual arm first update LR: `1.0720866372880134e-4` (matches source logged LR)
- baseline earlier analysis logged LR / residual arm first-after-update LR: `1.0680028468509506e-4`
- residual arm last update LR: `4.3633092056127864e-10`
- residual arm final-after-update LR: 0.0
- manifest: `data/paired_tail_dryrun_residual/paired_tail_manifest.json`

This is the cleanest available fresh-moment residual-LR arm. Because the original 80M optimizer/RNG state was not saved, it cannot be bit-identical to the uninterrupted 80M→100M baseline; the comparison to the anchor includes ordinary stochastic replay differences plus the fresh moments. The **residual-vs-reheat** comparison, however, isolates LR trajectory under matched weights, data, masks, dropout seed, optimizer class, and fresh moments.

### Arm B — `reheat_cosine`

Question: with the same fresh moments and same replayed tail stochasticity, does LR reheating add reusable learning beyond moment reset?

Tail-only cosine over 505 steps, peak LR `3e-4`, warmup fraction 0.06 = 30 steps. It uses the analogous optimizer-step convention: the first update starts at 0.0 and the LR after that update is `1.0e-5`:

- first update LR: `0.0`
- first-after-update LR: `1.0e-5`
- last update LR: `3.2807429663106456e-9`
- final-after-update LR: 0.0
- manifest: `data/paired_tail_dryrun_reheat/paired_tail_manifest.json`

The peak `3e-4` is deliberately below the original 1e-3 base LR but above the residual tail LR after warmup, so it tests renewed learning energy without making the tail a high-LR destructive restart.

## If and only if the corrected curriculum misses: launch logic

Do **not** launch these while the corrected curriculum comparison is still unresolved. If corrected curriculum reaches the leader region (cheap7 roughly ≥43.77), spend the GPUs on full official-compatible evaluation instead.

If corrected curriculum is below the leader region and especially if it is not above the fixed-256 baseline cheap7 43.108, launch the two paired arms from identical 80M weights, preferably simultaneously on the two H100s:

```bash
python3 -B experiments/archive/representation_and_objectives/scripts/paired_tail_continuation_trainer.py \
  --mode residual_low_lr \
  --example_jsonl experiments/archive/frontier_consolidation/data/density_cleanqwen_overlay_medium_riskhard/cleanqwen_fineweb_compact_view_reinvest_100M.jsonl \
  --tokenizer_path experiments/archive/representation_and_objectives/data/legal_representation_route_map/tokenizers/legal_byte_bpe_40k \
  --source_model_path experiments/archive/representation_and_objectives/training/runs/legal40k_accum_compact_view_reinvest_seed43022/hf_model/chck_80M \
  --source_metrics_path experiments/archive/representation_and_objectives/training/runs/legal40k_accum_compact_view_reinvest_seed43022/scientific_metrics.json \
  --source_training_log experiments/archive/representation_and_objectives/training/runs/legal40k_accum_compact_view_reinvest_seed43022/training_log.jsonl \
  --source_checkpoint_name chck_80M \
  --output_dir experiments/archive/representation_and_objectives/training/runs/tail_freshmom_residual_lr_from80M_seed53023 \
  --gpu 0 --train_rng_seed 53023 \
  --original_base_lr 0.001 --original_warmup_fraction 0.06 \
  --reheat_peak_lr 0.0003 --reheat_warmup_fraction 0.06
```

```bash
python3 -B experiments/archive/representation_and_objectives/scripts/paired_tail_continuation_trainer.py \
  --mode reheat_cosine \
  --example_jsonl experiments/archive/frontier_consolidation/data/density_cleanqwen_overlay_medium_riskhard/cleanqwen_fineweb_compact_view_reinvest_100M.jsonl \
  --tokenizer_path experiments/archive/representation_and_objectives/data/legal_representation_route_map/tokenizers/legal_byte_bpe_40k \
  --source_model_path experiments/archive/representation_and_objectives/training/runs/legal40k_accum_compact_view_reinvest_seed43022/hf_model/chck_80M \
  --source_metrics_path experiments/archive/representation_and_objectives/training/runs/legal40k_accum_compact_view_reinvest_seed43022/scientific_metrics.json \
  --source_training_log experiments/archive/representation_and_objectives/training/runs/legal40k_accum_compact_view_reinvest_seed43022/training_log.jsonl \
  --source_checkpoint_name chck_80M \
  --output_dir experiments/archive/representation_and_objectives/training/runs/tail_freshmom_reheat_lr3e4_from80M_seed53023 \
  --gpu 1 --train_rng_seed 53023 \
  --original_base_lr 0.001 --original_warmup_fraction 0.06 \
  --reheat_peak_lr 0.0003 --reheat_warmup_fraction 0.06
```

Each arm is ~505 steps from an existing checkpoint, much cheaper than a new 100M endpoint. This is the minimum reliable paired test of whether reheating adds useful learning beyond moment reset.

## Readout logic

Evaluate both arms with the same cheap7 wrapper used for other endpoints, using `hf_model/chck_100M` if present. Compare against:

- uninterrupted legal40k fixed-256 100M baseline cheap7: **43.108**; full Overall **41.1406**
- legal40k baseline 80M reference cheap7: **42.9486**
- visible leader cheap7 region: roughly **43.77**

Scientific interpretation:

1. `residual_low_lr` ≈ anchor and `reheat_cosine` > residual: reheating provides useful late learning beyond moment reset and deserves a stronger follow-up or composition.
2. `residual_low_lr` > anchor and `reheat_cosine` ≤ residual: fresh moments/state reset is the active factor; renewed LR is unnecessary or harmful.
3. both arms ≤ anchor: late reset/reheating does not solve the acceptance coordinate; close this tail route.
4. any arm improves GlobalPIQA only through the known parallel/nonparallel redistribution while damaging Supplement/Entity/BLiMP/EWoK: not a SOTA path even if one column moves.
5. if either arm reaches leader cheap7 region, immediately run full official-compatible evaluation including SuperGLUE and AoA before making route conclusions.

The key comparison is not just cheap7; it is whether broad columns and the known hard relation surfaces improve together rather than repeating prior tradeoffs.
