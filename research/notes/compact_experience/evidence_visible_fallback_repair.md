# cluster continuation noaoa result evidence-visible masking fallback repair

This note records the dormant fallback route work done while the corrected natural-cluster no-AoA evaluation was running.

## Why this route exists

If the natural entity-relation cluster arm E1 does not beat E2 anchor-shuffle, E3 repeat, and E4 untouched continuation, the next genuinely distinct mechanism should not keep manipulating anchor packs. The prepared fallback changes the learning signal: which whole words become prediction targets under MLM. The scientific idea is to keep the evidence context visible while masking entity, numeric, and attribute/object words more often, so gradients are concentrated on recoverable entity/relation/attribute knowledge rather than mostly on local syntax/function words.

## Measurement verification and repair


The verifier judged the fallback route mostly clean as an 80M→~100M continuation test because it keeps data pool, initialization, optimizer family, batch size, sequence length, word exposure, and checkpoint/evaluation path matched. The important confound it identified was that the first implementation equalized masked whole-word count but not masked subword-token count. Since named entities and long content words can fragment into more subword positions, a positive result could otherwise be caused by a larger or harder loss-bearing token budget rather than by the intended credit-assignment mechanism.

I repaired this before any H100 training:

- `scripts/evidence_visible_continuation_trainer.py` now has `--mask_budget token_count` as the default. In token-budget mode it first samples a uniform-WWM target token count for each sequence, then priority-samples whole words while keeping masked subword positions close to that target. `--mask_budget word_count` remains available only to reproduce the older word-count behavior if needed.
- The trainer records `masked_tokens`, `target_masked_tokens`, and `masked_token_budget_ratio` in logs and `scientific_metrics.json`.
- Sentence-start detection was broadened for punctuation plus quotes/brackets so sentence-initial capitals are less often misread as named entities.
- `scripts/launch_evidence_visible_masking_training.sh` now uses the token-budget default explicitly and refuses to overwrite existing run directories.
- `scripts/launch_evidence_visible_eval.sh` now discovers checkpoint names from each arm's `scientific_metrics.json` rather than assuming `chck_99M` or `chck_100M`.

## Dormant assets

- Runnable trainer: `scripts/evidence_visible_continuation_trainer.py`
- Training launcher: `scripts/launch_evidence_visible_masking_training.sh`
- No-AoA evaluator: `scripts/launch_evidence_visible_eval.sh`
- No-AoA summarizer: `scripts/summarize_evidence_visible_eval.py`

The route has not been launched in this step as of this note. It should be launched only if the natural-cluster no-AoA result and held-out diagnostic fail to support E1 as a stronger mechanism.

## Required interpretation if run

The decisive contrast is `evidence_visible` against all three controls: `uniform_control`, `random_priority`, and `inverse_priority`. A useful result must improve the target recovery columns EWoK, COMPS, and GlobalPIQA without spending too much of BLiMP, Supplement, Entity, or Reading, and it must show `masked_token_budget_ratio` near 1.0 across arms. If a single seed looks promising, a second seed should follow before any 100M-from-scratch or leaderboard-facing investment.
