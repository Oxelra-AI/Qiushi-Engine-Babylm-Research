# coupled correspondence control — coupled dual-view true-correspondence matched control

## Why this experiment
endpoint resolved dualview hardsurface established that `coupled_sparse20_aligned_20M` is the first intervention
among these experiments to strongly move the load-bearing context-conditioned alternative-binding
surface:
- fixed full ewok interaction synthesis EWoK stable-reversal subset: stable failures 922 (mlm_only) -> 430; accuracy 0.26581 -> 0.51258
- fixed fw globalpiqa relevant substrate GlobalPIQA hard52: 3.85% -> 9.62%; mean top-minus-correct 1.6229 -> 1.4432 nats

But the same coupled run collapses broad cheap7 (~38.79 vs 39.79 mlm_only). The separated
sparse20 arm preserves broad cheap7 (40.29) but does NOT touch the mechanism (EWoK stable
rows worse than mlm_only). The open scientific question: is the coupled hard-surface repair
caused by **true source-rewrite correspondence**, or merely by the coupled auxiliary
perturbation of the training trajectory?

## The decisive control
`coupled_sparse20_shuffled_20M`: identical to `coupled_sparse20_aligned_20M`
except `--mode shuffled`. The trainer (`dual_view_corrected_trainer.py`) applies a
batch-level derangement of the SAME source texts, so source multiset and charged exposure
are identical per batch; only true correspondence differs.

Matched coordinate confirmed:
- trainer: `experiments/archive/frontier_consolidation/scripts/dual_view_corrected_trainer.py` (DUAL_VIEW_CORRECTED, coupled single-optimizer pathway)
- corpus: `.../cleanqwen_fineweb_compact_view_reinvest_100M.jsonl` (SHA 3dd19f09...)
- tokenizer: `.../compliant_tokenizer` (legal16k, SHA 91b775...)
- aux: `.../sparse_aux_pair_data/top20/sparse_aux_pair_data_top20.json`
- seeds 43/43022/43023, adapter_scale 1.0, bottleneck 128, batch 256, seq 256,
  LR 0.001, warmup 0.06, WWM 0.15, max_word_exposure 20M, lr_total_steps 2529,
  aux_pair_shuffle_seed 43022
- params 35,463,008; first loss 9.837543487548828 (bit-identical to aligned) — verified in launch log

Run output
`experiments/archive/representation_and_objectives/training/runs/coupled_sparse20_shuffled_20M_seed43022`.

## Interpretation tool
`experiments/archive/representation_and_objectives/scripts/coupled_shuffled_control_readout.py` reuses the
endpoint resolved dualview hardsurface fixed fw globalpiqa relevant substrate GlobalPIQA hard52 all-option reader and fixed full ewok interaction synthesis EWoK stable-row
reader verbatim, adds the shuffled control target, and computes the decisive delta
`coupled_aligned_minus_coupled_shuffled` on both hard surfaces (parses OK; resolves
mlm_only + coupled aligned as ready; waits for shuffled `final`).

## Decision rule (predeclared, read at natural resolution)
Per strategist coupled control interpretation infrastructure guidance, do NOT treat correspondence as a binary vote between two
unequal readouts. Read each surface at its natural resolution:
- **EWoK (1,471 rows) is the primary, cleaner interaction measure.** Read as a graded effect:
  aligned-minus-shuffled deltas in accuracy, stable-failure COUNT, and the interaction-sum
  distribution (mean/median shift toward less-negative interaction). A large aligned-over-
  shuffled reduction in stable failures (upgraded interpreter flags >=50 fewer, or accuracy
  >= +0.03) is the load-bearing evidence.
- **GlobalPIQA (52 rows) read through paired ranks and margins, not only discrete accuracy.**
  Aligned-minus-shuffled hard52 mean top-minus-correct margin becoming more negative, or more
  correct-at-rank1, counts as coherent movement even if few discrete choices flip.

Verdicts:
- **True-correspondence-specific (preserve mechanism):** a large graded aligned-over-shuffled
  EWoK interaction effect WITH coherent GlobalPIQA margin/rank movement (even if only a few
  hard52 choices flip). Then design a minimal broad-preserving coupled
  variant that keeps cross-view true-alignment pressure while restoring
  BLiMP/Supplement/Entity/Reading/COMPS.
- **Generic auxiliary/trajectory effect (rethink route):** coupled SHUFFLED cuts EWoK stable
  failures comparably to aligned (correspondence-free repair share ~= total repair), i.e. broad
  movement without alignment-specific hard-row transitions. Then the mechanism claim collapses;
  the route must be rethought, not scaled.

The interpreter also decomposes the stable-failure repair: total = aligned-minus-mlm_only;
correspondence-free = shuffled-minus-mlm_only; true-correspondence share = (residual-total)/|total|.

## Observed matched-coordinate confirmation (running control, mid-run tail)
From the `s177_t14_tool1` status tail, the shuffled control charges aux exposure exactly as the
aligned run: update 50 shows cumulative_main 1,976,087 + cumulative_aux 18,414 = charged
1,994,501, mirroring the aligned run's aux-charging that made it stop at 19,995,185 charged
words / 501 updates (`stopped_before_cap: true`). mlm_only had zero aux and reached exactly
20,000,000 / 506 updates. So the correct matched comparison is coupled aligned vs coupled
shuffled (both aux-charged, both ~19.99M/~501 updates); mlm_only is the payload baseline.
first loss 9.837543487548828 is bit-identical across all three, confirming identical init and
main-stream WWM.

## Boundaries
- chck_82M endpoint (Overall 41.942481, reproduced, packaged) is left untouched.
- These hard surfaces are diagnostics; not tuned, not a submission scoring rule.
- No 82M tail launched from any sparse recipe until the correspondence question is settled.
