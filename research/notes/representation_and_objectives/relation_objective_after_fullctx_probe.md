# fullctx aux budget audit — relation-objective update after full-context probe across FW arms

## What changed

The full-context pivot-substitution view is real as a local likelihood phenomenon, but fullctx aux budget audit shows it is not yet a training target that can be expected to repair the project's persistent relation failures.

Across the completed 100M FW checkpoints, the same 2,266 legal held-out events were scored:

| arm | true-minus-shuffle mean | SE | EWoK wrong-row interaction median | GP parallel | hard52 margin | GP nonparallel | cheap7 |
|---|---:|---:|---:|---:|---:|---:|---:|
| compact | 0.487175 | 0.029113 | -0.956 | 24.27 | 1.717 | 53.0 | 43.181 |
| row-block breadth | 0.545990 | 0.030148 | -0.505 | 29.13 | 1.433 | 45.0 | 42.639 |
| interleaved breadth | 0.540041 | 0.031580 | -0.666 | 26.21 | 1.660 | 56.0 | 43.079 |

The arm ordering of the full-context margin agrees with the relation-side tradeoff (row-block/interleaved above compact), but the effect is small relative to event variability: range 0.0588 nats, only 0.0408 of the average event SD. Event-level Pearson correlations across arms are 0.864–0.891. All three checkpoints, including arms with thousands of EWoK stable conditional reversals and 61/103 GlobalPIQA-parallel rows wrong for all arms, already strongly prefer the true pivot over a matched shuffled pivot on these events.

## Why the full context pivot substitution probe trainer must not be launched unchanged

1. **The static signal is mostly shared local compatibility.** Increasing the true-vs-shuffled-pivot margin would likely reinforce a relation-word/context fit that the models already express, not necessarily the conditional alternative selection that fails in EWoK and GlobalPIQA.

2. **The proposed control pressure was unequal.** full context pivot substitution probe gradient preflight measured `true_vs_shuffle` at model-gradient ratio ~0.150 versus MLM, while `placebo_shuffle_vs_anchor` was ~0.209. A semantic/control difference after training could reflect stronger nonsemantic pressure in the control rather than true relation learning.

3. **The exposure accounting was wrong.** The unchanged trainer counts only base WWM stream words. The audit over the exact 80M→90M segment found 11,346 auxiliary events. A repaired two-view true+shuffle objective would add 3,527,690 row-word passes, so the full 9,971,308 base-word segment would debit 13,498,998 word-passes. The unchanged four-view implementation would debit 17,026,688 word-passes. To spend a 10M-word debit after 80M, observed ratios imply at most 7,386,702 base words for a two-view objective or 5,856,281 base words for the unchanged four-view trainer.

## Mechanistic update

The current evidence says the models have already learned many local pivot-consequence compatibilities in attested sentences, but they fail when a context must choose among plausible alternatives. The next objective should therefore move from **pivot substitution** to **masked alternative consequence selection**:

- leave ordinary WWM input/masking unchanged;
- use only events whose dependent/consequence target group is actually masked in the ordinary WWM pass;
- reuse the same MLM forward logits at the masked target positions, adding no extra auxiliary input views and no extra word exposure;
- contrast the true dependent target against a matched cross-event dependent target with the same token length, category, target class, and frequency band, avoiding identical targets and same-row shortcuts;
- measure whether this true-target-vs-plausible-alternative margin is unsaturated and follows EWoK/GlobalPIQA relation readouts more strongly than the full-context pivot-substitution margin.

This candidate is not yet authorized for training. It needs a no-update calibration first: coverage of naturally masked event targets, target-negative match quality, margin distributions across standard 80M and the three 100M FW arms, and gradient ratio/cosine using the existing WWM forward. If it proceeds to training later, a same-exposure standard-WWM reference must be trained to the identical debited exposure.

## Artifacts

- Probe synthesis: `experiments/archive/representation_and_objectives/data/fullctx_probe_tradeoff_synthesis/fullctx_probe_tradeoff_synthesis.json`
- Probe note: `research/notes/representation_and_objectives/fullctx_probe_tradeoff_synthesis.md`
- Budget audit: `experiments/archive/representation_and_objectives/data/fullctx_aux_budget_audit/fullctx_aux_budget_audit.json`
- Budget note: `research/notes/representation_and_objectives/fullctx_aux_budget_audit.md`
