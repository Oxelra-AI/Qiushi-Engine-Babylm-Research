# causal interface trajectory — Common-screen evaluation and corrected pairing result for carrier-residual credit

## Research question

The real-pretraining loop from causal intervention tested whether the frozen-carrier/private-adapter BabyLM recipe can learn more transferable competence by weighting MLM credit toward tokens the protected `chck_82M` carrier cannot predict. The first score screen suggested EWoK/Entity gains over a newly replayed standard arm, but two issues had to be resolved before treating that as evidence for the weighting mechanism:

1. The exact inherited coherent86 alpha0.75 endpoint, not only the replayed standard arm, must sit on the same evaluation screen.
2. causal intervention `compute_carrier_weights` ran the carrier forward in training mode with dropout and consumed the RNG stream before the weighted student forward, so shared seeds did not create a clean paired comparison.

## Execution changes made in causal interface trajectory

### Corrected paired replay

Implemented `scripts/context_credit_paired_trainer.py`, wrapping the causal intervention trainer but replacing carrier scoring so it:

- switches to `eval()` during carrier scoring;
- disables the private adapter for the carrier distribution;
- isolates teacher RNG with `torch.random.fork_rng` and a fixed internal seed;
- restores the model's train/eval state and private-adapter flags before the student forward.

This repairs the dropout/RNG pairing issue. An initial wrapper smoke command did not propagate the `--smoke` flag into the imported causal intervention trainer, so it actually produced full legal-tail runs. This is useful evidence:

- `training/runs/smoke_standard/` is a full 82M→86M run and is bit-identical to causal intervention standard. Alpha0.75 final SHA256: `f3f46e30f5a7af5c95e9af394d15b47e1ea8ec9d16e385471da7f52b8e5b2059`.
- `training/runs/smoke_carrier_residual/` is a full 82M→86M deterministic carrier-residual run. Alpha0.75 final SHA256: `e3b722fab62f7a3af41b828f0e3eb3ed926cb59664a22d4fb7da17da14e5db38`.
- The original causal intervention stochastic carrier-residual alpha0.75 final SHA256 was `6a98c76354e59d6db01869aa9a9a127ebed0f1d9f4376783688c02e620a84a48`.
- The inherited coherent86 alpha0.75 final SHA256 is `e14d757ae51b41e33bf0813f841248fecd1eefeb9e040f520c4c6203343b15c8`, identical to `frontier_consolidation/training/runs/frozen82_fastpath4M_coherent_seed43022/hf_model/final/model.safetensors`, and different from the gradient-accumulated causal intervention/026 standard replay.

Training dynamics: deterministic carrier-residual credit produced a larger private correction than standard and even larger than the original stochastic carrier-residual in neutral KL/RMS. Its final neutral KL was `0.009448` versus causal intervention stochastic carrier-residual `0.007145` and standard `0.002551`. Its private RMS remained strongest in deep layers, e.g. L7/L8 about `0.0481/0.0583` versus standard `0.0229/0.0204`.

### Evaluation repair

Implemented `scripts/eval_common_screen.py`, which fixes Reading by passing `--data_path evaluation_data/fast_eval/reading/reading_data.csv`. It also materializes alpha-scaled checkpoints by changing every layer's executed `private_adapter.scale`, not just `config.private_adapter_scale`. A first alpha materialization attempt failed because Hugging Face dynamic modules tried to write into a read-only global cache; the script was repaired to set `HF_MODULES_CACHE` and related caches inside the writable evaluation output directory.

Merged causal intervention six-column results with repaired Reading-only runs using `scripts/merge_reading.py`. Collated all common and alpha results with `scripts/collate_common_eval.py` and `scripts/alpha_collate.py`.

Main result paths:

- `data/common_eval/collated_common_eval.json`
- `data/alpha_eval/collated_alpha_eval.json`
- Exact common-screen JSONs under `data/common_eval/*/*_eval.json`
- Repaired causal intervention Reading-only JSONs under `data/reading_only/`

## Common cheap7 screen with fixed Reading

All values below are the same fast/cheap7 research screen: fast BLiMP/Supplement/EWoK/Entity/GlobalPIQA, full COMPS, fast Reading. It is not the complete official stack with SuperGLUE/AoA, but it is the correct common screen for this decision.

| Model | equal7 | BLiMP | Supp | EWoK | Entity | COMPS | GPIQA | Reading |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `chck_82M` frozen carrier | 44.4943 | 69.22 | 66.00 | 50.64 | 27.68 | 52.19 | 37.58 | 8.150 |
| exact coherent86 alpha0.75 | **44.5643** | 69.17 | 66.40 | 49.82 | 27.78 | 52.05 | 38.565 | 8.165 |
| causal intervention standard replay alpha0.75 | 44.2293 | 69.01 | 64.80 | 49.45 | 27.93 | 52.19 | 38.08 | 8.145 |
| causal intervention stochastic carrier-residual alpha0.75 | 44.4143 | 68.93 | 64.80 | **50.82** | **28.29** | 52.29 | 37.59 | 8.180 |
| causal interface trajectory deterministic carrier-residual alpha0.75 | 44.2064 | 69.00 | 64.80 | 50.09 | 27.41 | **52.35** | 37.58 | 8.215 |

Deltas versus exact coherent86 alpha0.75:

- causal intervention standard replay: `-0.3350` equal7.
- causal intervention stochastic carrier-residual: `-0.1500` equal7. It gains EWoK `+1.00`, Entity `+0.51`, COMPS `+0.24`, and Reading `+0.015`, but loses Supplement `-1.60` and GlobalPIQA `-0.975`.
- causal interface trajectory deterministic carrier-residual: `-0.3579` equal7. It gains only EWoK `+0.27`, COMPS `+0.30`, Reading `+0.05`, but loses Entity `-0.37`, Supplement `-1.60`, and GlobalPIQA `-0.985`.

The stochastic causal intervention carrier-residual arm did beat the newly replayed standard arm by `+0.1850` equal7, but that contrast is not trustworthy as a mechanism comparison because of carrier-score dropout/RNG mismatch and because both replay arms trail the inherited coherent86 reference. The corrected deterministic carrier-residual arm is slightly worse than the standard replay (`44.2064` vs `44.2293`) and much worse than coherent86.

## Alpha check: better correction or larger correction?

The weighted path had higher KL and adapter RMS, so I tested alpha0.5 for both stochastic and deterministic carrier-residual models with executed scale changed at the module level.

| Model | equal7 | BLiMP | Supp | EWoK | Entity | COMPS | GPIQA | Reading |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| causal intervention stochastic carrier-residual alpha0.5 | 44.3950 | 69.04 | 65.20 | 50.18 | 27.82 | 52.28 | 38.065 | 8.180 |
| causal interface trajectory deterministic carrier-residual alpha0.5 | 44.1900 | 69.05 | 65.20 | 49.91 | 27.61 | 52.28 | 37.08 | 8.200 |

Reducing alpha partly recovers Supplement/GlobalPIQA for the stochastic model but loses most of the EWoK/Entity gain; the net is slightly below the stochastic alpha0.75 endpoint and still below coherent86. For the deterministic model, alpha0.5 also does not help. This looks more like a larger correction with a tradeoff than a better retention–gain frontier.

## Scientific interpretation

The raw carrier-error credit hypothesis is weakened. Directing private-adapter credit to low-confidence carrier tokens can create an EWoK/Entity signal in the flawed stochastic candidate, but the common-baseline and corrected-pair evidence show that it is not a trustworthy improvement over the existing coherent86 recipe. The result supports the following qualification: low carrier confidence mixes useful unfamiliar relations with ambiguity or locally hard targets whose gradients can damage broad competence. In this stable-carrier/private-path architecture, the lever cannot be stated as “train more on what the carrier does not know.” It must distinguish credit for information made available by relation/context structure from credit on hard but unresolved or distribution-shifting local prediction targets.

The real-pretraining loop is therefore productive but not yet a score advance. The correct lesson is not to discard selective credit entirely; it is to refine the scientific object. A useful next intervention should either:

1. change the data composition so the practiced relation is explicitly useful, as in the state-update companion-packet intervention while preserving packed pair-row topology; or
2. build a selective-credit scheme whose weights depend on context-resolvable relation evidence, not only low carrier probability.

Because no carrier-residual candidate survived the exact coherent86 common screen, complete official evaluation is not yet warranted for these arms. The next official-stack run should be reserved for a candidate that first beats coherent86 on a common cheap7 screen or shows a strong complementary signal with a clear route to recover the lost columns.

## Subsequent research route

The relation-retargeted stream with UPDATED_USE and UNCHANGED_DISTRACTOR_USE packets was judged more promising than further local tuning of raw carrier-error weighting. Its resulting model would require evaluation on the same common screen. A controlled hybrid remained conditional on the relation-retargeted results revealing which columns moved.
