# context dependence ig probe result context-dependence IG probe: pilot result and interpretation

## What ran

Forward-only masked-token NLL probe on legal corpus text `cleanqwen_fineweb_compact_view_reinvest_10M.jsonl` (corpus SHA in JSON), no training, no upload, no submission.

- checkpoints: scale1.75 `chck_82M` and `chck_100M`
- 256 sampled rows, 4 targets/row, 1024 items, radii 4/8/16, max_seq_len 256, CUDA
- artifact: `data/context_dependence_ig_probe_chck82_chck100_256rows/context_dependence_summary.{json,md}` and `context_dependence_items.csv`
- script: `scripts/context_dependence_ig_probe.py` (CPU smoke: `data/context_dependence_ig_probe_smoke/`)

IG(i) = local_window_NLL(target_i) - full_context_NLL(target_i), target piece masked in both.

## Key numbers (chck82, radius 4)

- full NLL mean 2.6116, median 1.3105
- IG mean 1.5996 (wide context is worth ~1.6 nats on average)
- corr(IG, full NLL) pearson -0.2911; partial corr controlling log sample frequency -0.5242
- high-IG top20% mean full NLL 1.0876; low-IG bottom20% mean full NLL 4.2782

Same qualitative pattern at radii 8/16 and at chck100.

Between chck82 and chck100: mean full-NLL delta -0.0307; corr(refIG, full-NLL delta) pearson 0.036/0.055/0.082 at r=4/8/16; partial controlling logfreq 0.067/0.081/0.107.

## Interpretation

1. **Context dependence is real and large, but it is not where residual error lives.** High-IG tokens (identity strongly determined by wider context) are the ones the mature model already predicts well once context is present (full NLL ~1.09 for the top-20% IG). The underlearned residual (high full-context NLL ~4.28) sits in **low-IG** tokens — tokens that remain hard even with full context and that a local window does not rescue. The partial correlation controlling frequency is strongly negative (-0.52), so this is not a frequency artifact.

2. **The specific target-allocation hypothesis is weakened.** The context dependence ig probe result route idea was: shift the fixed MLM mask/loss budget toward high context-dependence targets because those carry cross-clause binding. The probe shows those high-IG targets are already low-error at maturity. Masking more of them would concentrate supervision on already-solved positions, matching the earlier finding that lower MLM loss does not imply broader competence. This is not a promising direct training target.

3. **Context dependence does not track the 82M->100M competence movement.** IG at 82M has almost no relation to which tokens change full-context NLL by 100M (partial corr ~0.07-0.11). So context-dependence is also not a benchmark-independent selector for the late reallocation phenomenon.

## Consequence for routing

- Do not launch an IG-target-allocation training run. The cheap probe did its job: it removed a plausible-looking route before any H100 training, exactly the intended discrimination.
- The residual-error mass is in **low-IG, locally-and-globally-hard tokens** whose difficulty is not explained by context availability or frequency. That is a different and possibly more interesting object: what makes these tokens hard, and whether their difficulty is intrinsic (rare sense, entity, number, morphology) rather than a supervision-allocation problem. This should be characterized (target kind, source family, morphology, entity/number) before any objective is proposed, and it must be checked against the closed benchmark-shaped and innovation-masking families so it does not re-enter them.
- The endpoint branch is unaffected: alpha0.75 remains the strongest local validator-pass endpoint (Overall(AoA0) 42.1210, carrier SHA 40181994...), reconciled in context dependence ig probe result as a config-scaled single function over coherent86 weights. Endpoint hardening (HF bundle validation, no submission) remains available independently of this mechanism result.

## Suggested next work

1. Characterize the low-IG high-error residual: stratify the existing CSV and a larger sample by target kind, morphology, entity/number/rarity, and source family; determine whether the hard mass is intrinsic token difficulty (not addressable by masking reallocation under the 10M rule) or a specific learnable structure.
2. If a learnable structure appears, design a matched-exposure cheap screen with an explicit ordinary-MLM control; if the mass is intrinsic rarity/entity difficulty, this data-efficiency lever is closed and the search should move to a genuinely different upstream mechanism.
3. Keep endpoint hardening (alpha0.75 HF bundle, no submission) as a parallel, non-blocking track.
