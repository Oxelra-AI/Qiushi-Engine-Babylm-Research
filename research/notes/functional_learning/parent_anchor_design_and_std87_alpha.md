# held fitted direction test — parent-anchored continuation design and std87 alpha finding

## Why the new comparison is scientifically different

clean d component ablation showed that ordinary continuation from exact coherent86 can still lower masked-LM CE on fixed suffix probes, but the common-screen score drifts after a near tie at 87M. It also showed that carrier-residual confidence weighting is not useful: it moves the private path more, worsens fixed-probe CE, and harms EWoK/Entity.

The remaining alternative is not another token confidence rule. In clean d component ablation the neutral KL term still anchored the adapted private-on model to the private-off 82M carrier on neutral examples. This is a holdover from the inherited frozen-carrier/private-adapter recipe. When starting from exact coherent86, that anchor can pull against the useful private correction already present in the parent. The relevant scientific question is whether additional legal suffix learning can be *consolidated onto the existing useful correction* if the neutral preservation reference is the frozen private-on coherent86 parent instead of the private-off carrier.

This tests cumulative learning rather than raw hard-token weighting:

- Same parent: `models/frontier`.
- Same stream suffix: compact-view-reinvest JSONL rows after coherent86, `skip_rows=556791`, total cap 100M.
- Same trainable path: existing private adapters only.
- Same ordinary WWM main objective, optimizer, seed, schedule offset, executed private scale, and neutral-example subset protocol as clean d component ablation standard.
- Changed factor: neutral KL target is frozen coherent86 private-on, using KL(parent private-on || current private-on), not private-off carrier.

A positive result must beat the coherent86 parent on the common screen and then survive stronger official-compatible evaluation. Merely matching the parent would mean better preservation, not useful added competence.

## Implementation

Script: `experiments/archive/functional_learning/scripts/parent_anchor_continuation.py`.

It imports and reuses the clean d component ablation trainer to avoid changing the data loader, masking, accounting, optimizer, checkpoint naming, and save format. It monkey-patches only `compute_neutral_kl`:

- loads a frozen teacher copy of the exact coherent86 parent on the same GPU;
- keeps teacher private adapters enabled at scale 0.75;
- computes `target_logits = teacher(masked_inputs)` on the same neutral subset;
- computes token-averaged `KL(parent_private_on || current_private_on)` over attention tokens;
- backpropagates this loss through the current private-on model.

Smoke command succeeded:

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=0 python experiments/archive/functional_learning/scripts/parent_anchor_continuation.py \
  --gpu 0 \
  --output_dir experiments/archive/functional_learning/training/runs/parent_anchor_smoke \
  --max_updates 1 \
  --checkpoint_words 0
```

Smoke result: one update, `main_loss=2.4905`, neutral loss approximately zero at initialization (`-9.37e-10`, numerical), `tail_words=40244`, `total_consumed_words=86045539`. This is expected because current and frozen parent initially match exactly.

The full run had been submitted and remained incomplete at the time:

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=0 python experiments/archive/functional_learning/scripts/parent_anchor_continuation.py \
  --gpu 0 \
  --output_dir experiments/archive/functional_learning/training/runs/coherent86_continue_parent_anchor_seed43023
```

## clean d component ablation std87 alpha finding

The pending clean d component ablation alpha checks finished. The near-tie 87M ordinary continuation checkpoint is not a clear advance after executed-scale recalibration:

| model | alpha | equal7 | Δ vs coherent86 | BLiMP | Supp | EWoK | Entity | COMPS | GPIQA | Reading |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| coherent86 parent | 0.75 | 44.5643 | — | 69.17 | 66.40 | 49.82 | 27.78 | 52.05 | 38.565 | 8.165 |
| std87 | 0.75 | 44.5543 | -0.0100 | 69.04 | 67.20 | 50.18 | 27.54 | 52.11 | 37.565 | 8.245 |
| std87 | 0.50 | 44.5671 | +0.0029 | 69.19 | 67.20 | 50.09 | 27.57 | 52.14 | 37.565 | 8.215 |
| std87 | 1.00 | 44.5543 | -0.0100 | 69.07 | 66.80 | 50.27 | 27.25 | 52.16 | 38.065 | 8.265 |

The alpha0.5 mean is only +0.0029 equal7 above parent and still has lower Entity and GlobalPIQA than parent. This is a competence redistribution, not yet an improvement that warrants full official evaluation.

## Prepared analysis path after full parent-anchor run

Script prepared: `experiments/archive/functional_learning/scripts/collate_parent_anchor.py`.

The proposed completed-run comparison evaluates a selected parent-anchor ladder on the common screen using the repaired causal interface trajectory evaluator. The planned tags mirror clean d component ablation:

- `pa_87M = .../hf_model/chck_total_87005295w`
- `pa_90M = .../hf_model/chck_total_90005295w`
- `pa_94M = .../hf_model/chck_total_94005295w`
- `pa_98M = .../hf_model/chck_total_98005295w`
- `pa_100M_final = .../hf_model/final`

Then run the collator:

```bash
PYTHONDONTWRITEBYTECODE=1 python experiments/archive/functional_learning/scripts/collate_parent_anchor.py
```

If any parent-anchor checkpoint beats coherent86 by more than measurement-level noise while preserving Entity/GPIQA better than std87 alpha0.5, run fixed-probe drift against the parent and then a stronger official-style evaluation. If it only returns to parent score, record it as a consolidation/stability result and prioritize the hypothesis that better experience structure is needed, especially the changed-form state-update stream.
