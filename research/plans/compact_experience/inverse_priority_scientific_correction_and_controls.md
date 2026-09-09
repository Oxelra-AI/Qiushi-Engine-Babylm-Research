# all mask endpoint interpretation inverse-priority correction and control assets

## What changed scientifically in all mask endpoint interpretation

The restart/masking inverse-priority result exposes two issues in the initial interpretation:

1. The common restart/rephasing effect must be separated from mask-target redistribution. The relevant mask-target number is inverse minus uniform, not inverse minus clean.
2. The current arms are confounded by target geometry and randomness: inverse masks more shorter whole words at fixed token budget, and the original trainer uses one RNG stream for selection and corruption, so non-uniform selection consumes a different number of draws and shifts downstream corruption/replacement noise.

The strongest no-AoA signal remains real enough to pursue, but mechanism attribution must be sharpened.

## Endpoint-matched correction from existing trajectory files

The missing uniform-control `chck_100M` no-AoA row was recovered from `data/external/mask_uniform_control_trajectory_summary.json`:

- uniform_control `chck_100M` equal7 = 43.2114; equal6 without GlobalPIQA = 44.5625.
- inverse_priority `chck_100M` equal7 = 43.7050; equal6 without GlobalPIQA = 45.0500.
- inverse−uniform at true `chck_100M`: equal7 +0.4936; equal6 without GlobalPIQA +0.4875.
- inverse−uniform per-column at true `chck_100M`: BLiMP +0.37, Supplement +2.28, EWoK -0.17, Entity +0.17, COMPS +0.25, GlobalPIQA +0.53, Reading +0.025.

This corrects the initial interpretation: the true-100M inverse signal is not mostly a GlobalPIQA artifact. In contrast, evidence-visible chck_100M is +0.3393 equal7 but -0.1042 equal6 vs uniform, with GlobalPIQA +3.0, so evidence-visible is GlobalPIQA-driven.

Full staged calculation: `data/external/endpoint_matched_mask_contrast.json` and `.md`. These calculations were recorded as staged outputs.

## Decoupled and length/count-matched controls prepared

New scripts:

- `scripts/decoupled_mask_control_trainer.py`
  - separates mask-selection RNG from corruption/replacement RNG;
  - adds `length_matched_random` mode, which draws the inverse-priority target-length template but randomizes word identity within exact wordpiece-length bins;
  - records fixed-basis `base_high_fraction_among_selected_words`, `selected_wordpiece_len_mean`, and `selected_base_priority_mean`, avoiding the old arm-internal high-priority metric trap.
- `scripts/audit_decoupled_mask_control.py`
  - CPU-only sampler audit.

CPU audit results:

- initial: `data/external/decoupled_mask_audit.json`;
- after patching `length_matched_random` to avoid inverse-template word identities when same-length alternatives exist: `data/external/decoupled_mask_audit_after_avoid_template.json`.

After-patch audit on 24 chunks confirmed:

| mode | masked words | masked tokens | token ratio | base-high fraction | mean wordpiece length | mean base priority |
|---|---:|---:|---:|---:|---:|---:|
| uniform | 618 | 911 | 1.0 | 0.401 | 1.474 | 0.975 |
| inverse_priority | 701 | 911 | 1.0 | 0.251 | 1.300 | 0.761 |
| length_matched_random | 701 | 911 | 1.0 | 0.382 | 1.300 | 0.962 |
| evidence_visible | 580 | 911 | 1.0 | 0.588 | 1.571 | 1.244 |
| random_priority | 627 | 911 | 1.0 | 0.389 | 1.453 | 0.987 |

`all_inverse_length_histograms_matched_by_control=true`; `corruption_mismatches_on_overlap=0` for overlapping inverse/length-matched selected token positions. The length-matched control now exactly matches inverse target geometry while more strongly removing low-priority word identity, so it is a cleaner P0 control if the route remains alive after full SuperGLUE+AoA and seed replication.

## How to use these assets

Immediate priority remains the pending full evaluation of seed43022 inverse-priority 95M/100M. If the true 100M endpoint crosses or remains near the visible 41.8 row, launch the already-compiled original-seed43122 inverse-vs-uniform replication first. The decoupled controls should follow once replication shows the signal is not a one-seed artifact.

If the seed43122 original replication is positive, run a three-arm decoupled control from the same clean parent:

1. decoupled uniform;
2. decoupled inverse_priority;
3. decoupled length_matched_random.

This distinguishes three possible mechanisms:

- inverse > length_matched_random > uniform: both target geometry and low-priority identity matter;
- length_matched_random ≈ inverse > uniform: target count/length/fragmentation dominates;
- inverse ≈ uniform after RNG decoupling: original result partly came from corruption-RNG realization.

Do not describe a final data-efficient learning principle until these controls and seed replication are measured.
