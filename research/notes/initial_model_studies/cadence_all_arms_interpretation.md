# cadence all arms interpretation — Matched cadence all-arms interpretation

## Evidence files

- All-arms evaluator: `scripts/eval_cadence_all_arms.py`
- JSON: `data/cadence_all_arms_fast_profile.json`
- Summary note: `notes/cadence_all_arms_fast_profile.md`
- Runs:
  - fixed reference: `training/runs/cadence_fixed256_ref_1M_b128_seed42/`
  - length-only: `training/runs/cadence_length_only_1M_b128_seed42/`
  - mask-decay-only: `training/runs/cadence_mask_decay_only_1M_b128_seed42/`
  - combined: `training/runs/cadence_combined_1M_b128_seed42/`

## Matching

The comparison is interpretable. All arms used the same trainer/model constructor, official 1M words, 49 steps, batch 128, seeds 42/456/789, baseline16k tokenizer, and identical 520-position DeBERTa-v2 config with 34,471,264 parameters. Source-word composition was identical: BNC spoken 762,080 words and CHILDES 237,920 words.

The active-token / masked-target differences are the experimental factors:

| arm | active tokens | masked targets | mechanism |
|---|---:|---:|---|
| fixed256_ref | 1,335,322 | 199,522 | fixed length 256, mask 0.15 |
| length_only | 1,122,911 | 167,557 | early length 128 reduces active tokens and targets; late 512 sees longer examples |
| mask_decay_only | 1,335,322 | 302,811 | same active tokens, much denser early masking |
| combined | 1,122,911 | 244,267 | fewer active tokens than fixed, more targets than fixed |

## Fast-profile result versus fixed256 reference

| arm | BLiMP | Supplement | EWoK | Entity | COMPS | Reading |
|---|---:|---:|---:|---:|---:|---:|
| length_only | +0.54 | +2.80 | -3.00 | +0.64 | +0.67 | -0.120 |
| mask_decay_only | -0.41 | +0.40 | -1.63 | +1.11 | -0.02 | +0.145 |
| combined | +0.18 | +4.00 | -3.54 | -0.44 | -0.09 | -0.120 |

## Scientific interpretation

The cadence route does not currently solve the BabyLM Strict-Small deficit. The main target cluster remains Entity/EWoK/GlobalPIQA, and every tested cadence factor damages EWoK at 1M:

- Length scheduling improves BLiMP, Supplement, Entity, and COMPS mildly, but EWoK falls by 3.00 and Reading falls slightly. The gain is therefore not aligned with the missing leader phenotype; it may mainly alter shallow grammar/supplement behavior while reducing early broad world-knowledge signal.
- Mask decay improves Entity by +1.11 and Reading slightly, but EWoK falls by 1.63 and BLiMP falls. This is the only arm with a notable Entity bump, but it trades against EWoK and gives no multi-column SOTA route.
- The combined arm gives a large Supplement gain (+4.00) but has the worst EWoK loss (-3.54) and Entity becomes negative. The two factors do not combine constructively for the target cluster.

This pattern is consistent with cadence changing optimization emphasis rather than producing the missing reusable knowledge mechanism. More masked targets early can move Entity a little, but broad world-knowledge/commonsense performance is weakened. Longer-context scheduling as implemented reduces the total active-token exposure by ~16%, which likely contributes to the EWoK loss.

## Route consequence

Do not launch seed43, 3M, 10M, or 100M from these cadence arms. The current factorized cadence schedules are not scale-ready for Overall SOTA because none improves Entity and EWoK together while preserving protected columns.

The only possible small follow-up within cadence would be a different length schedule that keeps active-token exposure matched, such as fixed-512 reference versus late-512 schedule, but this would test token exposure/long-context access rather than the current cadence hypothesis. It should not outrank a stronger route unless independent evidence supports long-context exposure itself.

The next route should move toward a different bottleneck: word/morph anchoring, tokenizer/representation changes, or another architecture-level mechanism grounded in BabyLM findings and able to address low-frequency entities/properties without relying on sparse ordered-prefix targets.

The current-best exact-checkpoint recheck remains important for endpoint-level interpretation and should be executed before making SOTA-gap conclusions.
