# RoBERTa transfer synthesis readiness and static target-label context

## Current state read

Managed RoBERTa work remains unresolved in the runtime ledger:

- compact 100M RoBERTa arm.
- selected official-compatible evaluation supervisor for the compact/repeat RoBERTa pair.
- CPU local pair-stratified bridge after the RoBERTa arms finish.

No new H100 training, SuperGLUE or AoA evaluation was performed.

## bridge dose response interpretation evidence integrated

bridge dose response interpretation produced a static packed-stream label-load profile for its target-selective source-absent vs copied-content whole-word deletion arms:

- File: `experiments/archive/representation_and_objectives/data/packed_static_label_load_profile_v2/packed_static_label_load_profile_v2.json`
- Note: `research/notes/representation_and_objectives/packed_targetselect_static_load_and_readout_patch.md`
- Stream SHA: `3dd19f09deeca44d6b07340e70cd15baa4b5c7bffe0bd462ff37536e470e7691`
- 100M words / 647,400 rows / 2,529 optimizer batches.
- Source-absent candidate BPE pieces: 319,580.
- Matched copied-content candidate BPE pieces: 322,370.
- Copied-minus-source-absent candidate BPE pieces: +2,790 over 100M.
- Expected realized WWM deleted copied-minus-source-absent pieces: +418.5 at p=0.15.
- This matches the reported realized CUDA-WWM scale (48,105 vs 48,387 deleted BPE; copied-minus-absent +282).
- LR-weighted denominator ratio `drop_abs/drop_copied`: 0.99998037; maximum batch denominator deviation from 1: 0.00182084.

Scientific use: this makes simple loss-normalization or training-time label-load differences a weak alternate explanation for any later endpoint separation. It does **not** provide endpoint evidence by itself. The source-absent mechanism still needs the postrun readout: local source-absent denoising movement together with endpoint movement on Supplement and the compact order result and roberta transfer decision relational EWoK domains.

## RoBERTa reader repair

I patched `scripts/roberta_stratified_bridge_integrator.py` so its early fallback now points to the repaired event-set readout:

- old fallback: `data/roberta_pair_stratified_response_probe_early_cpu/stratified_response.json`
- new fallback: `data/roberta_pair_stratified_response_probe_early_cpu_repaired/stratified_response.json`

The repaired bridge check was run to:

- `data/roberta_stratified_bridge_integrator_early_repaired_check/stratified_bridge_integration.{json,md}`

Current readout is only early local response:

- selected official trajectory absent;
- local response present but not late-band;
- event SHA match true against repaired manifest SHA `c172b378873553f209a6bfd9bf53a63e0bde251c1f77690f8abea0ef2d18709a`;
- repaired early local means: `compact|source_absent_content` +0.09423, `compact|retained_content` +0.05493, `compact|function_other` -0.02932, `repeat|retained_content` +0.02685, `repeat|function_other` -0.069997;
- the only positive measured feature marker is compact content density on compact source-absent events (+0.04790 mean high-minus-low across 10M/20M).

This is still local pseudolikelihood only. It must not trigger a new expensive arm before selected official-compatible late-band scores and late local bridge results are read.

## New synthesis reader

Created:

- `scripts/roberta_compact_result_synthesizer.py`
- current smoke output: `data/roberta_compact_result_synthesis_current/roberta_compact_result_synthesis.{json,md}`

The synthesizer reads existing files only. It combines:

1. RoBERTa selected compact-minus-repeat late-band deltas when `data/roberta_full100m_integrated/roberta_transfer_integrated.json` exists.
2. local pair-stratified response from the late CPU bridge if present, otherwise repaired early response only with an explicit local-only warning.
3. bridge dose response interpretation static label-load profile.
4. target-selective postrun summary when `experiments/archive/representation_and_objectives/data/packed_targetselect_postrun_readout/postrun_readout_summary.json` exists.

Current smoke result:

- RoBERTa selected ready: false.
- RoBERTa local ready: true, using late band: false.
- static ready: true.
- postrun ready: false.
- Reading: the repaired early local source-absent response remains a local pseudolikelihood signal; no new expensive arm follows before selected trajectory and late local bridge are read.

## Interpretation of the Pending Results

After the pending experimental results become available:

1. Verify the compact training integrity: exact 100M exposure, 2,529 updates, 30,528,064 params, legal tokenizer, all ten checkpoints.
2. Read the selected-eval output, then run/read `scripts/roberta_transfer_result_reader.py` if the supervisor did not already create the ready report.
3. Read the late local bridge and confirm event SHA matches repaired manifest SHA `c172b378...18709a`.
4. Rerun `scripts/roberta_compact_result_synthesizer.py` (default args) to combine the selected trajectory, late local bridge, static profile, and any available postrun target-selective result.

Interpretation before more H100 work:

- Stable RoBERTa compact gains on cheap6/cheap5/EWoK+Entity/Supplement/Entity/COMPS plus late source-absent/content-density local concordance: next expensive work is a paired independent-seed RoBERTa replication.
- Stable RoBERTa gains without local concordance: compact marginal may transfer, but measured source-absent/content-density mechanism is incomplete; inspect item/family movement before one clean intervention.
- Local source-absent response without stable downstream transfer: another local-vs-selected dissociation; do not replicate from local response alone.
- Neutral/negative RoBERTa stable readouts: bound this tested stock absolute-position bidirectional MLM coordinate and return to natural compact-marginal decomposition around content density, source-wide coverage, lexical recurrence, and diversity reinvestment.

No leaderboard submission is permitted or performed.
