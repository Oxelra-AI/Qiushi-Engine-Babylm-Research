# aoa safety audit and route — AoA measurement safety audit and load-bearing route

## What this step verified about the corrected ledger

Ran `grep_search` over all active scripts/notes for AoA scoring usage. Findings:

- **Canonical scoring is centralized and correct.** `scripts/babylm_official_scoring.py`
  is the single source of Overall arithmetic: it stores `aoa_raw_correlation` and
  `aoa_leaderboard_score = 100 * raw`, and Overall uses the leaderboard score.
  Regression `self_test()` passes: leaderboard row with AoA=22.9 recomputes Overall
  40.6222 (expected 40.62); raw 0.229 → 22.9. Top-20 non-zero-AoA rows match within
  leaderboard rounding.
- **Active evaluation runners import the canonical helper.** `full_overall_eval_runner.py`,
  `full_eval_runner.py`, `full_eval_runner.py`, and the new
  `full_eval_runner.py` all delegate to the corrected full overall eval runner, so any
  future full eval scales AoA correctly. The four summaries
  (`full overall eval/028/030/032`) label AoA as leaderboard units and preserve raw correlation.

## Two concrete safety fixes applied this step

1. **Retired an unsafe legacy evaluator.** `scripts/eval_full_official.py`
   computed Overall directly from raw AoA (`result["scores"]["AoA"]` with no ×100)
   and would have produced a mis-scaled Overall if ever run. Its `main()` now raises
   immediately with a message pointing to the corrected runners. It is retained only
   for provenance; it targets `step027_*` runs that were superseded by the clean
   clean qwen compliance and validity/clean qwen experiment state corpora anyway.
2. **Hardened the legacy-field path in the scoring helper.** When only the legacy
   `aoa_for_provisional_overall` field survives, the helper now infers units from
   magnitude (`|v|>1` → leaderboard units; else raw correlation) and records
   `aoa_unit_inferred_from_legacy_field`. Zero is invariant either way. This prevents
   a silent double-scaling or under-scaling on old records.

## Why AoA is now a load-bearing route (not bookkeeping)

Corrected leaderboard-unit ledger (AoA = 100×raw):

| model | Overall | AoA (leaderboard) | AoA raw |
|---|---:|---:|---:|
| qwen_clean_aligned (s43022) | 41.3443 | 0.0 | 0.0 |
| qwen_clean_aligned (s43122) | 40.6501 | 0.0 | 0.0 |
| official_lengthmatched (s43022) | 38.9404 | -15.70 | -0.157 |
| official_lengthmatched (s43122) | 38.9424 | -11.72 | -0.117 |
| qwen_shuffled_control | 39.2240 | -12.79 | -0.128 |
| official_original_dup | 37.8266 | -13.32 | -0.133 |
| official_sourcematched | 39.6838 | 0.0 | 0.0 |
| selected_original_dup_all | 40.9915 | 0.0 | 0.0 |
| qwen_separated_pair | 39.5643 | 0.0 | 0.0 |
| visible leader | 41.8 | 0.0 | 0.0 |

Two facts stand out:
- The clean-Qwen treatments have raw AoA exactly 0.0 (they produce no measurable
  developmental AoA fit), so their Overall is unchanged by the correction. The gain
  vs the official controls is now much larger in Overall (e.g. +2.40 vs seed43022
  lengthmatched) but that gap is dominated by the controls' large **negative** AoA
  (−12 to −16 leaderboard points), not by an AoA advantage of the treatment.
- If a clean model reached raw AoA r≈0.229 (leaderboard 22.9) while holding
  qwen_clean_aligned's other eight columns fixed, its Overall would be **43.89**
  (aoa unit correction leverage calc). So AoA is the single largest reachable lever on Overall.

The visible leader shows AoA 0.0, so the leader is *not* winning on AoA. This means
two independent routes to beat 41.8 exist and can compound:
1. lift the NLP/Reading columns of the clean-Qwen model, and
2. move AoA from ~0 toward a positive developmental fit **legally**.

## Legality boundary for the AoA route

- **Forbidden:** using official AoA/CDI evaluation words, the CDI human curve, AoA
  predictions, or AoA scores as any training or model-selection signal.
- **Allowed:** shaping training experience with training-corpus statistics, official
  source labels (CHILDES/simple_wiki/dialogue/prose), corpus-internal word frequency,
  general public developmental priors (earlier = child-directed/high-frequency/short),
  and model/checkpoint dynamics. Selection among candidates must use the training-side
  order design and the general Overall, not the AoA column output.

## Next AoA-safe intervention (materialized this step)

`scripts/materialize_devcurr_qwen_order.py` builds a developmental first-pass
order over the **exact** clean qwen compliance and validity clean-Qwen 10M multiset (multiset digest preserved),
then repeats the original order for passes 2–10. First 1M words become pure CHILDES,
then simple_wiki, then aligned pair rows / dialogue, then dense prose last. This makes
the AoA checkpoint ladder (chck_1M..chck_10M) meaningfully order-sensitive while
minimally disturbing the final training trajectory. Dual-seed training launched
(`launch_devcurr_training.sh`), evaluated by `full_eval_runner.py`.
