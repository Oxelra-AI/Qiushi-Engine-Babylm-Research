# mask budget and pad contract — Mask budget and pad/label contract for stream-order chunk training

This check integrates independent_review feedback before any H100 launch of the experience-utilization trainer.

## Baseline parity

The matched legal40k row256 baseline (`legal40k_accum_compact_view_reinvest_seed43022`) used fixed WWM 0.15, AdamW LR 0.001, warmup fraction 0.06, and 2,529 optimizer updates. Its training log sums to 20,568,519 masked targets over 100M charged words.
The stream-order U256/U64 chunk dry-runs use the same data stream SHA `3dd19f09deeca44d6b07340e70cd15baa4b5c7bffe0bd462ff37536e470e7691` and recover 139,426,440 active tokens versus the chunk stream preflight row256 active-token measurement 137,061,620.

## Mask budget

At mask_prob=0.15, U256 dry-run produced 20,916,090 targets, +347,571 versus the realized row256 baseline (1.016898x). U64_128_256 produced 20,915,364 targets, +346,845 (1.016863x).
A target-matched launch should pass mask_prob about 0.14745584 by the chunk stream preflight expected-active-token formula, or 0.14752237 if matching the exact realized baseline target count from the training log.

## Pad/label contract

- U256 first batch L256: chunks=302, words=40170, active=55799, masked=8393, pad positions=21513, labels-on-pad=0, pad-input changes=0, active missing word groups=0.
- U64_128_256 first batch L64: chunks=998, words=39624, active=55113, masked=8275, pad positions=8759, labels-on-pad=0, pad-input changes=0, active missing word groups=0.

The pad/label contract is clean for the first real stream-order chunked update in both arms. The remaining design choice is scientific: mask_prob=0.15 tests full recovered experience including extra target pressure; mask_prob≈0.147456 tests target-matched visibility. The first expensive run after the depth result should choose this explicitly and record it in the launch manifest.

JSON: `experiments/archive/representation_and_objectives/data/mask_budget_and_pad_contract/mask_budget_and_pad_contract.json`
CSV: `experiments/archive/representation_and_objectives/data/mask_budget_and_pad_contract/mask_budget_comparison.csv`
