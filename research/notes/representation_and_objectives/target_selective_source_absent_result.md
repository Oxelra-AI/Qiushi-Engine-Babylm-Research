# target selective source absent result — identical-text target-selection screen for compact rewrite source-absent labels

## Why this screen was run

earlier analysis showed that source-extractive S/G controls could not simultaneously preserve semantic separation and match the actual learning interface: BPE burden, WWM target burden, and fragmentation traded off against one another. The cleaner intervention is therefore to hold the **input text and corruption schedule fixed** and vary only which masked targets contribute loss.

This screen uses the corrected crossview v2 20M result compact MLM geometry (`own_visible`) because it already has:

- identical source+compact input text for all arms;
- identical tokenizer, untrained init, WWM masks, optimizer, LR schedule, filler dose, source dose, and 20M word exposure;
- intrinsic rewrite-origin labels carried by rewrite identity;
- a completed full-loss arm (`crossview_own_visible_20M`).

The specific scientific question was whether the rare source-absent compact-content labels are causally useful beyond their count. If removing those labels hurts compact-side semantic denoising more than removing an equal number of copied compact-side labels, direct prediction of abstractive/content-reformulated tokens remains a live mechanism. If not, the compact marginal advantage is more likely carried by exposure/context/distribution shaping rather than those labels.

## Scripts and artifacts

- Trainer: `experiments/archive/representation_and_objectives/scripts/target_selective_deberta_trainer.py`
- Readout: `experiments/archive/representation_and_objectives/scripts/target_selective_readout.py`
- Paired bootstrap: `experiments/archive/representation_and_objectives/scripts/target_selective_event_bootstrap.py`
- Fixed probe events inherited from earlier analysis: `experiments/archive/representation_and_objectives/data/existing_trajectory_denoising_probe/probe_events.jsonl`
- Training arms:
  - full baseline: `experiments/archive/representation_and_objectives/training/runs/crossview_own_visible_20M`
  - drop source-absent compact-content labels: `experiments/archive/representation_and_objectives/training/runs/target_selective_drop_abs_content_20M`
  - drop matched copied compact-side labels: `experiments/archive/representation_and_objectives/training/runs/target_selective_drop_copied_matched_20M`
- Aggregate readout: `experiments/archive/representation_and_objectives/data/target_selective_readout/target_selective_readout.json`
- Event-level bootstrap: `experiments/archive/representation_and_objectives/data/target_selective_event_bootstrap/target_selective_event_bootstrap.json`
- Per-event losses: `experiments/archive/representation_and_objectives/data/target_selective_event_bootstrap/event_losses.jsonl`

## Preflight and exact intervention

Both target-selective arms trained on the same 20M-word stream as crossview v2 20M result own-visible: 147,780 examples, exactly 20,000,000 words, two epochs. Original realized masked-target counts in the crossview v2 20M result full-loss arm were:

- filler: 4,134,298
- source: 110,478
- rewrite copied: 65,723
- rewrite source-absent content: 7,649
- rewrite source-absent other: 2,537
- total: 4,320,685

`drop_abs_content` set all 7,649 `rw_abs_content` labels to `-100`, keeping 4,313,036 targets.  
`drop_copied_matched` set exactly 7,649 deterministic copied-side labels to `-100`, also keeping 4,313,036 targets.

The smoke tests for both arms passed with custom full-visible DeBERTa matching stock forward exactly (`full_vis_delta=0.0`), finite gradients, and identical initial batch loss up to the intended target removal.

Two initial attempts failed before training and provide no model evidence. The subsequent training runs completed successfully:

- `drop_abs_content`: 578 updates, 20M words, final loss 3.55970, kept targets 4,313,036, dropped absent-content 7,649.
- `drop_copied_matched`: 578 updates, 20M words, final loss 3.55174, kept targets 4,313,036, dropped copied 7,649.

The full crossview v2 20M result own-visible reference had final loss 3.55358. Thus the overall training loss did not move in a simple way that could by itself explain the category-specific probe result.

## Evaluation-only anchor on existing compact/repeat/adjbreak trajectories

Before training the target-selective arms, I ran the fixed semantic denoising probe from earlier analysis on 4,096 events/category (12,288 events total) across existing compact, repeat, and adjbreak trajectories at 20M/60M/100M.

At 100M, compact minus repeat fixed-event loss was:

- retained content: **−0.959886** nats
- source-absent content: **−1.661613** nats
- function/other: **−0.318863** nats

At 100M, compact minus adjbreak was:

- retained content: **−0.535834** nats
- source-absent content: **−0.611586** nats
- function/other: **−0.078048** nats

This says that the successful compact trajectory has a strong late compact-side denoising advantage over repeat/adjbreak under identical source+compact probe inputs, especially on source-absent content. It does not prove endpoint utility by itself; it motivated the causal target-selection screen.

## Target-selection result: source-absent labels carry a specific local causal signal

Readout compared 10M and 20M checkpoints from the two target-selective arms plus crossview v2 20M result full own-visible, on exactly the same 12,288 fixed compact-side probe events.

### 20M aggregate piece-weighted fixed-event deltas

Positive `drop_abs_minus_drop_copied` means removing source-absent content labels hurt more than removing the same number of copied labels.

| Probe target category | drop_abs − full | drop_copied − full | drop_abs − drop_copied |
|---|---:|---:|---:|
| retained content | −0.043185 | −0.006640 | −0.036545 |
| source-absent content | **+0.119934** | **+0.000140** | **+0.119794** |
| function/other | −0.000686 | −0.012435 | +0.011749 |

At 20M, withholding source-absent compact-content labels selectively worsens later denoising of source-absent compact-content events by about **0.12 nats** relative to withholding matched copied labels. Withholding copied labels has essentially no effect on source-absent fixed-event loss.

### Pair-cluster bootstrap

The event-level paired bootstrap clusters by compact pair identity (1,000 resamples). For the main 20M comparison `drop_abs_minus_drop_copied`:

- source-absent content:
  - event-mean delta: **+0.187203** nats
  - piece-weighted delta: **+0.119794** nats
  - piece-weighted 95% interval: **[+0.097717, +0.140308]**
  - bootstrap fraction above zero: **1.000**
- retained content:
  - event-mean delta: −0.022000
  - piece-weighted delta: −0.036545
  - piece-weighted 95% interval: **[−0.057512, −0.011310]**
- function/other:
  - event-mean delta: +0.017084
  - piece-weighted delta: +0.011749
  - piece-weighted 95% interval: [−0.011610, +0.032772]

Thus the robust signal is category-specific: direct prediction of source-absent compact-content labels improves later reconstruction of source-absent compact-content probes, but does not improve retained-content probes in this 20M geometry.

At 10M the same source-absent comparison was even larger: piece-weighted +0.162417 with interval [+0.146506, +0.177621]. The 20M effect is still clearly positive, not a one-checkpoint artifact.

## Scientific interpretation

This screen revives a narrower source-absent mechanism after crossview v2 20M result closed direct correct-source reachability as dominant. The two statements are compatible:

- crossview v2 20M result: seeing the correct source, versus a wrong source or blocked source, gives only a modest extra loss benefit on source-absent content; copied targets show larger partner-specific visibility effects.
- target selective source absent result: when the compact text and masks are held fixed, **actually training on source-absent content labels** is specifically useful for later source-absent compact-side denoising, beyond an equal number of copied-side labels.

So the live mechanism is not primarily `source attends to paired source and infers absent word`. It is more like: compact rewrite marginals introduce sparse but semantically important noncopy target events; predicting them teaches the model part of the reformulation/content-density distribution that literal repetition and copied-token learning do not supply.

This is still an internal 20M denoising mechanism result, not an endpoint or architecture-general principle. It was run in the crossview v2 20M result pair-aware compact geometry, not the historical 100M packed compact-view stream. The result should therefore guide the next experiment rather than be treated as final evidence for BabyLM score movement.

## Pending external-surface check

A bounded cheap7 evaluation has been submitted for the three 20M checkpoints:

- `full_step221_own_visible/chck_20M`
- `drop_abs_content/chck_20M`
- `drop_copied_matched/chck_20M`

Output root: `experiments/archive/representation_and_objectives/data/target_selective_cheap7_eval`  

No conclusion about official BabyLM task movement should be drawn until this task returns and its summaries are inspected.

## Next interpretation after cheap7 returns

If the 20M cheap7 readout shows no external-surface movement, this still supports the local source-absent learning channel but argues against spending two full 100M runs in the crossview v2 20M result geometry. The better next step would be a historical-geometry loss-mask intervention or a representation probe linking source-absent event improvement to Supplement/EWoK transition rows.

If `drop_abs_content` is worse than `drop_copied_matched` on Supplement/EWoK/COMPS or the correctness transition analysis relational domains while broad columns are comparable, then the source-absent target channel becomes strong enough to justify a 100M target-selective extension or a more faithful historical-stream implementation.

If `drop_abs_content` is not worse than `drop_copied_matched`, then the local denoising effect does not immediately transfer to BabyLM surfaces at 20M; the compact marginal advantage likely requires late accumulation, broader rewrite-marginal context, or interaction with historical packing/exposure rather than the rare labels alone.

## Exposure split: not only exact selected-event memorization

Because the fixed probe samples compact-pair words from the same population used in training, I split the per-event losses by whether the probed word was actually selected by the deterministic crossview v2 20M result/225 WWM schedule in epoch 0 or epoch 1. Artifact:

- `experiments/archive/representation_and_objectives/data/event_exposure_split/event_exposure_split.json`
- enriched per-event records: `experiments/archive/representation_and_objectives/data/event_exposure_split/event_losses_with_training_exposure.jsonl`

For source-absent content probe events, selection counts were:

- selected in neither epoch: 2,990 events
- selected in exactly one epoch: 1,031 events
- selected in both epochs: 75 events

At `chck_20M`, `drop_abs_minus_drop_copied` on source-absent content was:

| WWM exposure of probed word | events | piece-weighted delta | 95% pair-cluster interval |
|---|---:|---:|---:|
| selected 0 times | 2,990 | **+0.092402** | [+0.065540, +0.117660] |
| selected 1 time | 1,031 | **+0.184915** | [+0.144577, +0.227687] |
| selected 2 times | 75 | **+0.314981** | [+0.139388, +0.510565] |
| selected any | 1,106 | **+0.194881** | [+0.153499, +0.236433] |
| all | 4,096 | **+0.119794** | [+0.096799, +0.141509] |

The effect is larger on words directly selected as training targets, as expected, but remains clearly positive for source-absent content words that were **not selected at all** in the two-epoch WWM schedule. This makes the result stronger than exact target-event memorization: training on the sparse source-absent content target class appears to shape nearby/category-level compact-side prediction, at least within the same compact-pair distribution.

This still does not prove official-task movement or cross-architecture generality. It supports a narrower causal statement: inside the compact rewrite marginal, source-absent content labels provide a category-specific denoising signal that transfers beyond the exact masked instances and cannot be replaced by the same count of copied-side labels over the first 20M words.
