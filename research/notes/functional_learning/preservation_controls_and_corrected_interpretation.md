# preservation controls and corrected interpretation preservation controls and corrected interpretation

## Why the preservation experiment design and evidence preservation arms are exploratory whole-policy tests

preservation experiment design and evidence launched two ordinary-corruption preservation branches (`lambda_pres=1.0` and `3.0`) on top of the `(M,S)` acquisition condition. A subsequent methodological review identified several issues that change their scientific meaning. preservation controls and corrected interpretation audited those issues and built corrected control machinery.

### 1. The preservation rendering is not genuinely evidence-absent

preservation experiment design and evidence uses standard 15% WWM on the *full packed Qwen pair row* as the parent-distillation rendering. This avoids pair-boundary construction errors, but the row still contains the source and current views; source evidence remains available to the model. Therefore the arm tests ordinary-corruption parent distillation on the same lawful text, not the stronger hypothesis of conditional competence separation on evidence-absent or source-removed renderings.

A successful preservation experiment design and evidence-style arm could show that parent distillation on ordinary corruption reduces damage while preserving the `(M,S)` acquisition effect. It would not by itself establish that the mechanism is specifically preserving underdetermined/evidence-absent competence. Conversely, failure would not refute preservation on truly evidence-absent renderings.

### 2. preservation experiment design and evidence did not keep the model stochastic stream fixed

`targeted_preservation_train.py` sets the global seed as:

```python
stable_seed("preservation experiment design and evidence-preservation", train_seed, focus_lambda, lambda_pres)
```

The acquisition examples themselves are row-keyed and lambda-independent, but dropout/RNG during training differs across `lambda_pres` values and differs from the original `(M,S)` run. The no-model audit `data/acquisition_stream_audit/acquisition_stream_audit.json` found:

- original earlier analysis/earlier analysis global seed: `135392207`
- preservation experiment design and evidence lambda seeds: `0.0 -> 940740990`, `1.0 -> 253484362`, `3.0 -> 228406199`
- acquisition masks/labels over the first five updates are deterministic and fixed: digest `b6be50e4b0f9e3aaf8e5fdb19d004d0779905c616c247d9e15855b102d48bbea`

Thus differences among the preservation experiment design and evidence λ arms include acquisition stochasticity changes, not only preservation pressure.

### 3. Identical-weight KL has a large dropout component in preservation experiment design and evidence mode

The preservation probe `data/preservation_kl_mode_probe_cpu8/preservation_kl_mode_probe.json` loaded two identical coherent86 checkpoints and evaluated the exact first-macro preservation renderings. It found:

- eval/eval identical weights: KL(teacher||student) ≈ `9.78e-10` nats/target, i.e. numerical zero.
- train/eval identical weights (preservation experiment design and evidence style): KL(teacher||student) ≈ `0.1806` nats/target on 244 targets.
- train/eval same seed exactly repeats, different seed changes the value but remains large (`0.1699`).

Therefore the initial KL around 0.15-0.16 in preservation experiment design and evidence training is substantially dropout/stochastic consistency pressure before any learned drift. It should not be interpreted as acquired functional divergence. Loss ratios are objective-value ratios, not gradient-contribution percentages.

### 4. KL direction corrected

The implemented PyTorch expression is:

```python
F.kl_div(log_softmax(student_logits), softmax(teacher_logits), reduction="sum")
```

This is KL(teacher || student), not KL(student || teacher). All later notes and comparisons should use this orientation.

## Clean controlled trainer

Created `scripts/clean_preservation_train.py`. Its purpose is to preserve the useful `(M,S)` acquisition branch while making later preservation experiments interpretable:

- uses the original earlier analysis/earlier analysis acquisition global seed, independent of `lambda_pres`;
- restores RNG after teacher loading;
- optionally forks preservation-forward RNG into a separate seed family and restores the acquisition RNG after auxiliary forwards;
- exposes `--pres-student-mode eval|train`, where `eval` measures parent-function preservation without dropout and `train` reproduces stochastic-consistency pressure in a contained RNG branch;
- records that the rendering is ordinary full-row WWM with source evidence present;
- documents KL direction as KL(teacher || student).

Dry run at `data/clean_preservation_dryrun/dry_run.json` matched the first macro data geometry: 252 acquisition examples, 247 focus targets, 7894 ordinary targets, 33 Qwen rows, 972 preservation targets, 4657 preservation words, acquisition seed `135392207`.

## Zero-preservation replay control

Created `scripts/zero_preservation_replay_compare.py` and ran a one-update CPU replay comparing the original earlier analysis `(M,S)` trainer against the clean trainer with `lambda_pres=0`.

Output: `data/zero_preservation_replay_compare_u1_cpu/zero_preservation_replay_compare.json`.

Result:

- checkpoint `model.safetensors` hashes are identical: `daabf480539502686c3227c5e1586a5011e2450685545593fb9795090db0a732` for both;
- all overlapping loss, target, row, and gradient values match;
- the only recorded difference is a harmless field-name mismatch (`pooled_loss` in original vs `pooled_ce` in clean), so the script returned the conservative status `ZERO_PRESERVATION_REPLAY_DIFFERS` but the model-state equivalence control passes.

This supports using the preservation controls and corrected interpretation clean trainer for future controlled preservation tests.

## Status of preservation experiment design and evidence background arms

Both earlier preservation runs ended at a 900s execution limit. The λ=1 arm wrote a complete 80-update summary and update_0080 checkpoint before termination; its output is at `data/preservation_lambda1p0_seed62064/`. Successful process completion was not established despite the saved endpoint; the files therefore require caution, and the training policy remains confounded by lambda-dependent global RNG and train/eval dropout preservation. The λ=3 arm only reached update 60 and should not be treated as an 80-update endpoint; it has checkpoints through update_0060 under `data/preservation_lambda3p0_seed62064/checkpoints/`.

## Next scientific work

Do not launch another coefficient merely by interpolation. The next meaningful experiment should use the clean trainer and compare the preservation mechanism against `(M,S)` under fixed acquisition stochasticity. A bounded next run should prefer `--pres-student-mode eval` first if the scientific target is parent-function preservation rather than dropout consistency. If a train-mode auxiliary branch is later tested, it should be interpreted separately as stochastic consistency regularization, not just stronger parent preservation.

For evaluation, the λ=1 preservation experiment design and evidence completed checkpoint may still be useful as an exploratory whole-policy signal, but it cannot be treated as a causal test of preservation strength. Before expensive official evaluation, run cheap diagnostics on it and on any clean controlled pilot to see whether ordinary-corruption distillation changes the gain-cost trade-off or only damps the update.
