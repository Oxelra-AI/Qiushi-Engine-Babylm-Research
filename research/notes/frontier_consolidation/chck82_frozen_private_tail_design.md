# chck82 frozen private tail design — verified 82M frozen-slow private-tail design

## Why the route changed

An independently repeated full evaluation of the existing scale1.75 `chck_82M` checkpoint reports Overall `41.942481167385985`, above the displayed 41.80 Strict-Small leader. The separate verification (`data/chck82_independent_verification/`) confirms:

- all nine columns are present and arithmetic-recompute to the hardened Overall/cheap7 exactly;
- measurement repeatability against the first score is tight (Overall delta `0.00022695351662349594`, max column delta `0.005893446487505116`);
- the legal 10M pool and 100M stream hashes match the established compliant substrate;
- the tokenizer was trained on the legal 10M pool and endpoint-vs-training vocab maps are identical;
- selected checkpoint exposure is `82,012,495` words, within the `100,000,000` total exposure allowance;
- the checkpoint is CPU-loadable through the local custom `AdapterDebertaV2ForMaskedLM` class and produces finite logits.

The endpoint strategy is to preserve `chck_82M` while the independent from-corpus reproduction remains pending. The 82M-to-100M degradation is evidence that further ordinary continuation moves the mature competence surface in the wrong direction.

## How this interacts with detached private experiment design

detached private experiment design's factorized 20M panel still matters, but only as a mechanism discriminator. Its outcome should not automatically launch another 100M training run from random initialization. If the panel shows separated sparse aligned > separated sparse shuffled and separated sparse aligned > exact `mlm_only`, then the next experiment should be anchored on the already-leading 82M function:

1. freeze the entire verified `chck_82M` scale1.75 function as a slow path;
2. attach a new zero-output private adapter after each encoder layer;
3. spend only the remaining legal exposure (`100,000,000 - 82,012,495 = 17,987,505` charged words) on sparse aligned/source-free acquisition plus neutrality;
4. compare aligned, shuffled, and base/neutral branches under identical remaining-exposure accounting.

This directly tests whether source-correspondence knowledge can be added to an already-leading function without moving the mature slow competence that made 82M successful.

## Mechanical feasibility already checked

Files:

- `scripts/frozen82_private_modeling.py`
- `scripts/frozen82_private_mech_check.py`
- output `data/frozen82_private_mech_check/frozen82_private_mech_check.{json,md}`

Mechanical result: PASS.

Key values:

- original `chck_82M` params: `35,463,008`;
- frozen-slow + fresh private params: `36,458,592`;
- new private params: `995,584`;
- slow path vs original max logit diff: `0.0`;
- private ON vs OFF at attachment max logit diff: `0.0`;
- private RMS at attachment: `0.0`;
- slow adapter remains active (max RMS `0.12243394553661346` on the smoke batch);
- with slow parameters frozen, private grad norm is nonzero (`21.6071`) when enabled and slow grad norm is `0.0`;
- disabling the private branch removes the gradient path entirely, as expected.

## Proposed tail arms if the detached private experiment design panel supports the mechanism

All arms start from exactly the verified `chck_82M` endpoint (`model.safetensors` SHA `93ceb76adf5a33d349f1de33e988e6ed0c2b2a547dbd92cf83cc952f8e2591b3`) and use the original legal stream after `82,012,495` already-charged words.

| Arm | Source correspondence | Trainable params | Losses | Scientific role |
|---|---|---|---|---|
| `frozen82_sparse20_aligned_tail` | true source→rewrite | fresh `.private_adapter.*` only | sparse aux CE + neutrality KL | tests whether true correspondence adds source-free competence to an already-leading function |
| `frozen82_sparse20_shuffled_tail` | batch-deranged source multiset | fresh `.private_adapter.*` only | same | distinguishes true source structure from source-text exposure and private optimizer dynamics |
| `frozen82_neutral_base_tail` | no aux | fresh private stays zero under neutrality | neutrality only or direct original checkpoint score | confirms base function and evaluation carrier remain unchanged |

A full official check of any tail candidate would need a compatible checkpoint ladder for AoA. A natural construction is to preserve original scale1.75 checkpoints up to `chck_82M` and then tail private checkpoints afterward; earlier private pathway is identically zero, so its predictions match the original slow path.

## What would make the tail experiment worth running

The current detached private experiment design 20M evidence must first show more than reduced damage:

- separated sparse aligned beats exact `mlm_only` at 20M;
- separated sparse aligned beats separated sparse shuffled;
- fragile columns (EWoK, Reading, Supplement; eventually SuperGLUE) are not damaged in the pattern seen in broad dual-view, scale1.75 endpoint decline, or U256.

If detached private experiment design fails these conditions, the frozen-82M private path remains a reusable capability, but the sparse source-free correspondence objective should be rebuilt before any additional GPU exposure is spent.

## Exact stream boundary for the frozen tail

Computed by `scripts/chck82_stream_boundary.py`; output `data/chck82_stream_boundary/chck82_stream_boundary.{json,md}`.

- `chck_82M` was saved at original scale1.75 update `2074` / loader step `2074`.
- Previous cumulative words: `81,973,343`.
- Boundary batch words: `39,152`.
- Actual cumulative words after the boundary batch: `82,012,495`.
- Target overshoot over nominal 82M: `12,495` words.
- Remaining legal charged words before the 100M cap: `17,987,505`.
- If the same stream and batch size 256 are reused, the tail should start from the next loader step (`2075`) after skipping `530,944` rows.

This exact boundary is necessary because continuing from the nominal 82M target rather than the actual saved function would double-count or omit part of the batch that created the score-bearing checkpoint.
