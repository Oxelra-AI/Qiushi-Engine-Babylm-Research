# dose training and leakage repair cleaned dose training and inherited-leakage repair

## What changed

The cleaned up-dose streams from earlier analysis are now the active dose substrates. I patched `experiments/archive/relation_learning/scripts/launch_restatement_dose_training.py` so a run directory pre-created by the background runtime is accepted if and only if it is empty. Validate-only checks then succeeded for:

- `dose21`, seed `43022`, stream `experiments/archive/relation_learning/data/probe_clean_nested_dose_streams/dose21/dose21_compact_view_reinvest_100M.jsonl`, SHA `80f8151764cc0a4865a0a8dfc80e092cfba56186eeb41997a96c3bc7db057779`, run `experiments/archive/relation_learning/training/runs/probe_clean_restatement_dose21_seed43022`.
- `dose25`, seed `43022`, stream `experiments/archive/relation_learning/data/probe_clean_nested_dose_streams/dose25/dose25_compact_view_reinvest_100M.jsonl`, SHA `3949e70367f9455a382f5e2e48a8500c8eb8f08cb5ad9454a34c6bfc93af5924`, run `experiments/archive/relation_learning/training/runs/probe_clean_restatement_dose25_seed43022`.

Both training runs had started at this stage:

- Dose21 seed43022, probe-clean.
- Dose25 seed43022, probe-clean.

A deliberate launch check showed both are running and saving early checkpoints. Dose21 printed earlier analysis loss `9.837388`, earlier analysis loss `6.804815`, and saved `chck_1M`/`chck_2M` during the check. Dose25 printed earlier analysis loss `9.835737`, earlier analysis loss `4.652988`, earlier analysis loss `4.306013`, and saved through at least `chck_10M` during the check. Final training results have not yet been delivered and must not be inferred.

## Inherited ALN leakage audit

earlier analysis repaired the *new* dose pairs, but the same blind official-corpus draw also affected the inherited COMPACT_EXPERIENCE aligned-restatement block. I built and ran `experiments/archive/relation_learning/scripts/audit_inherited_aln_and_paired_context_stream_leakage.py`. It applied the earlier analysis exact/subsequence/high-overlap screen to `experiments/archive/compact_experience/data/qwen_clean_aligned/selected_pairs.jsonl` and checked COMPACT_EXPERIENCE streams.

Main results in `experiments/archive/relation_learning/data/inherited_aln_leakage_audit/audit_summary.json`:

- Inherited selected ALN material: 37,594 pairs, 1,656,800 pair words, 75,188 source/rewrite texts audited.
- Pair-text hits: 33,431 blocking hit rows across 3,395 selected pairs and 11,975 reference snippets.
- Hit screens: 24,071 hits to the later 6,992-row heldout set, 9,359 hits to the seed43222 entity clean integration 2,647-row heldout set, and 1 hit to WikiLarge probe text.
- Query field: 32,608 hits from ALN originals and 823 from rewrites.
- For the later 6,992 rows, 2,354 rows contain inherited ALN source/rewrite text and 4,638 do not. Clean subset: `experiments/archive/relation_learning/data/inherited_aln_leakage_audit/heldout6992_no_inherited_aln_pair_text_hits.jsonl`.

The same script also found that the later 6,992-row set is not a true heldout set for COMPACT_EXPERIENCE OFF/ALN: both `official_only_10M` and `qwen_aligned_10M` contain all 6,992 rows as exact token subsequences, and `qwen_aligned_10M` contains them as ordinary row identities. Therefore the old ALN−OFF ordinary-loss improvement on those 6,992 rows is not a clean generalization ordinate.

## Current compact-view-reinvest heldout exposure

I built and ran `experiments/archive/relation_learning/scripts/audit_current_dose_stream_heldout_exposure.py` for the later 6,992-row set and `experiments/archive/relation_learning/scripts/audit_current_dose_stream_heldout2647_exposure.py` for the seed43222 entity clean integration 2,647-row set.

For the later 6,992 rows, the compact-view-reinvest base and repaired dose streams all contain the same 4,345 unique rows as ordinary row identities; 2,895 of those are not already pair-text-hit rows. This makes the later 6,992 set unsuitable as the ordinary-fit ordinate for the current up-dose curve, although the exposure set is shared across base/dose arms.

For the seed43222 entity clean integration 2,647 rows, the current base and both repaired up-dose streams have zero exact stream-identity hits. However 904/2,647 rows contain inherited ALN source/rewrite text. Thus the useful ordinary-fit readouts for the up-dose curve are:

- Full seed43222 entity clean integration heldout: `experiments/archive/frontier_consolidation/data/density_cleanqwen_overlay_medium_riskhard/heldout_cleanqwen_rows.jsonl` (2,647 rows), absent as exact stream rows from base/dose streams.
- Pair-text-clean seed43222 entity clean integration subset: `experiments/archive/relation_learning/data/current_dose_stream_heldout2647_exposure/heldout2647_no_inherited_aln_pair_text_hits.jsonl` (1,743 rows), absent as exact stream rows and without inherited ALN source/rewrite text under the dose training and leakage repair screen.

The pair-text-clean subset is the safest ordinary-language loss ordinate for the current dose substitution law. The full 2,647-row result should also be reported because it corresponds to the original seed43222 entity clean integration heldout instrument and shows how much inherited ALN text exposure matters.

## Re-reading the old ALN−OFF calibration

I recomputed the existing causal gpt relation boundary/032 ALN−OFF row-level tables by subset with `experiments/archive/relation_learning/scripts/recompute_paired_context_aln_off_heldout_subsets.py`; outputs are under `experiments/archive/relation_learning/data/paired_context_aln_off_heldout_subsets`.

On the later 6,992 rows, the two-seed old result was ALN−OFF `-0.01249` and `-0.01281`. After removing rows that contain inherited ALN source/rewrite text, the same old tables give `-0.01039` and `-0.00838` on 4,638 rows. On the 2,354 rows with inherited pair-text hits, the gaps are larger: `-0.01662` and `-0.02153`. This means part of the old ordinary-loss improvement was concentrated on ALN-exposed source/rewrite rows, while a smaller negative difference remains in the pair-text-clean subset. Because both OFF and ALN trained on the 6,992 rows as official text, these numbers quantify concentration of the old readout rather than clean generalization.

The prior linear expectations for up-dose ordinary loss (`~ -0.0033` at dose21, `~ -0.0063` at dose25) should be treated only as contaminated order-of-magnitude background. If the seed43222 entity clean integration pair-text-clean ordinary loss is flat while practiced-class T/U/N rises, the correct reading is not that restatement stopped buying ordinary fit; it is that the reliable substitution law is carried by practiced source-recurring readout rather than broad ordinary loss.

## Ordinary-fit scoring started

I built `experiments/archive/relation_learning/scripts/score_compactview_ordinary_fit_subsets.py`, a deterministic-mask ordinary MLM scorer for the current compact-view-reinvest dose curve. It scores:

- `heldout2647_full` (2,647 rows)
- `heldout2647_no_inherited_aln_text` (1,743 rows)

for inherited baselines `base43022` and `base43122`, and later the dose arms when checkpoints exist. Dry-run verified that baseline checkpoints `chck_80M`, `chck_90M`, and `chck_100M` exist for both inherited baselines. Baseline CPU scoring of ordinary fit on clean subsets had started; its result was still pending.

## Binding route update from functional_learning

A subsequent analysis corrected the objective-side binding interpretation. The useful decomposition is now `beta=(U+R)/2`, `alpha=(U-R)/2`, and `gamma=beta-|alpha|`; joint UPDATE+RETAIN corresponds to positive per-pair gamma. Binary candidate CE already rewards both U and R, so ordinary answer CE remains a serious competitor. Paired-context loss is only a candidate shaping term if it improves held gamma/joint on structurally valid rows while broader learning survives. The earlier pilot baseline was joint `3/55`, mean beta `-0.009`, mean `|alpha|` `3.324`, and mean gamma `-3.333`. This supported pausing further natural binding-packet construction at this stage; binding remained separate from the dose curve.
