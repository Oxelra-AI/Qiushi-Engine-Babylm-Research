# compact order evaluation and channel readout plan compact ordered-vs-scrambled evaluation and source-absent channel readout

## Scientific purpose

The compact order experiment design ordered-vs-scrambled compact-view experiment is no longer just an aggregate `ordered > scrambled` screen. source-absent target-selection result gives the correct causal readout: coherent compact order is interesting only if it helps the same rare compact-rewrite channel that survived the reciprocal-conditioning closure.

Current mechanism state:

- causal directional fork closed reciprocal-view conditioning as the compact mechanism: aggregate interaction was negative (`I = -0.077` cheap7, EWoK -0.425).
- source-conditioned ordering probes showed ordered source retrieval is general to bidirectional MLM and is weaker for compact than extractive families; it is not a compact-specific copied-token mechanism.
- earlier analysis found a narrow but robust local channel: direct training on source-absent compact-content labels improves later fixed-event source-absent compact-side denoising by about +0.12 nats relative to dropping the same count of copied labels, with pair-cluster interval roughly [+0.098,+0.140], and remains positive for probe words never selected by WWM.

Therefore the compact order experiment design arms test whether coherent word order in compact views preserves or strengthens this source-absent compact-content prediction channel under the historical DeBERTa MLM training interface. Official-compatible family scores are necessary but not sufficient.

## Expensive work admitted and bounded

Already running/pending work:

- Ordered 40M stock DeBERTa training was launched first.
- Scrambled 40M training was deferred until ordered training completed after approximately 23GB of residual GPU memory caused an OOM in the batch-256 trainer. The OOM does not establish model behavior.

Queued after-training work launched in compact order evaluation and channel readout plan:

- After both arms complete, the planned readout covers selected cheap official-compatible evaluation for 2 arms x 2 checkpoints (`chck_20M`, `chck_40M`) and the source-absent compact-channel probe. It excludes SuperGLUE and AoA.
- CPU/file training-log and mask-count auditing is required before interpreting narrow gaps, to account for realized BPE/WWM and mask drift.

This is the minimum reliable evidence for the current route decision: if ordered and scrambled are indistinguishable on stable official families and on the source-absent channel, then fluent compact order should not receive an 80M extension. If ordered has an aggregate gain without selective source-absent improvement, the most likely interpretation is generic sensitivity to corrupted word order, not a compact compression principle. Only ordered-selective improvement on source-absent compact-content plus stable official-family movement should justify mature extension.

## Scripts prepared

- `scripts/compact_order_channel_probe.py`
  - Rebuilds fixed ordered compact-side denoising events from `data/factorial_view_candidate_audit/compact_candidate_pairs.jsonl` using the legal spatial repair route status tokenizer.
  - Categories: `retained_content`, `source_absent_content`, `function_other`.
  - Evaluates the *same ordered source+compact masked events* on both ordered and scrambled arms at `chck_20M` and `chck_40M`.
  - Negative `ordered_minus_scrambled` means the ordered-trained model has lower NLL and better preserves the channel.
  - Pair-cluster bootstrap is saved for each category/checkpoint.

- `scripts/after_compact_order_training_eval.py`
  - Waits for both training arms to write `scientific_metrics.json`, `training_log.jsonl`, and `hf_model/chck_20M`/`chck_40M`.
  - Runs `scripts/eval_custom_checkpoint.py` for each arm/checkpoint.
  - Runs the channel probe and then the integrator.

- `scripts/integrate_compact_order_results.py`
  - Combines official-compatible selected cheap scores with source-absent channel results.
  - Reports cheap7, cheap6 without GlobalPIQA, cheap5 without GlobalPIQA/Reading, EWoK+Entity, and per-column deltas.
  - Marks each checkpoint as `ordered-selective` only when cheap6 movement is positive enough, EWoK+Entity is nonnegative, and source-absent NLL is lower for ordered.

- `scripts/training_log_mask_audit_wait.py`
  - Compares actual training logs across arms: finite losses, step count, exact word exposure, checkpoint presence, total masked-token counts, and batch/region-level masked-token drift.
  - Includes lead route assessment after source use probe realized-context numbers: compact-minus-scrambled changed block has Δ active tokens 0, Δ candidate groups +8, Δ view active +165, CPU-proxy Δ masked tokens +557, and CPU-proxy Δ masked view/source +472/+183.

## Interpretation thresholds

Primary official-compatible readout is still from `notes/compact_order_experiment_design.md`:

- `|Δ cheap6 without GlobalPIQA| >= 0.3`: decisive early signal.
- `0.1 <= |Δ cheap6 without GlobalPIQA| < 0.3`: suggestive; extension only if channel and family movement agree.
- `< 0.1`: no useful early aggregate order effect.

compact order evaluation and channel readout plan adds the source-absent condition:

- A mature extension is scientifically justified only if ordered has lower source-absent-content fixed-event NLL than scrambled, with direction stronger or more specific than retained/function categories, and official movement is not carried only by GlobalPIQA/Reading.
- Aggregate official gain without a source-absent channel advantage should not promote the route; it is consistent with generic damage from unnatural scrambled order.
- A source-absent channel advantage without official stable-family movement is a local mechanism result only; it should motivate a more faithful historical-stream or representation probe, not immediate full maturation.
- Noisy minibatch MLM loss is not a stop signal by itself. Finite stable training, exact exposure, valid checkpoints, official-family scores, and the fixed-event channel are the relevant evidence.

## Pending result locations

Expected outputs after the pending comparisons complete:

- Selected official-compatible score panel: `data/compact_order_selected_eval/`
- Source-absent channel probe: `data/compact_order_channel_probe/compact_order_channel_probe.{json,md}`
- Integrated readout: `data/compact_order_integrated_readout/compact_order_integrated_readout.{json,md}`
- Training-log/mask audit: `data/compact_order_training_log_mask_audit/training_log_mask_audit.{json,md}`

No leaderboard submission is allowed from these tasks.
