# lead route decision after legal40k — Lead route decision after legal16k/legal40k endpoints

## Current endpoint state

The completed compliant endpoints in this comparison do not establish a new Strict-Small SOTA.

- Inherited compact_view_reinvest with baseline16k: 42.0331 Overall, but not submission-compliant because tokenizer was trained outside the Strict-Small 10M budget.
- Legal16k compact_view_reinvest: seed43022 40.7040, seed43122 41.0240, mean 40.8640.
- Legal40k compact_view_reinvest: seed43022 41.1406, seed43122 40.4201, mean 40.7804. Hardened official-coordinate collation passed for both.
- Visible leader anchor remains `go76dof/wwm_curriculum_simplification_40k` at 41.80 Overall with 12×384 DeBERTa-v2, 40k tokenizer, LAMB/cosine LR 0.007, 64→256 sequence curriculum, and WWM7→Token3 masking curriculum.

## Interpretation of legal40k

Legal40k closes the pure-vocabulary question as a trade-off, not as a route:

| Movement vs legal16k mean | Direction |
|---|---:|
| BLiMP | +1.388 |
| Supplement | +2.612 |
| EWoK | +1.106 |
| Entity | -1.414 |
| GlobalPIQA | -3.942 |
| Overall | -0.084 |

The evidence supports a representation-layer support/segmentation tension: 40k restores some syntactic/linguistic and EWoK surface, but loses support-thin GlobalPIQA/Entity behavior. It does not justify another full two-seed vocabulary-axis interpolation as main next route. Minfreq25 remains useful as a conditional operating-point probe, but companion analysis is already pursuing a support-floor/equal-word line. companion analysis should not duplicate it unless vector makes minfreq25 a specifically necessary curve-localization point.

## Decision on minfreq25

Do not launch the prepared minfreq25 two-seed H100 trainings now.

Reason: minfreq25 is scientifically coherent, hash-checked, and ready, but it is an interior point on the same legal vocabulary/support axis. After legal16k and legal40k both missed 41.8 and companion analysis is already running minfreq50/equal-word work, companion analysis should reserve H100s for genuinely different factors. If later shows minfreq50 restores GlobalPIQA/Entity while losing segmentation gains, minfreq25 becomes a targeted one-seed curve-localization experiment rather than a blind full pair.

## Immediate low-cost evidence launched

The already-trained clean-Qwen WWM→token run differs from its fixed-WWM reference only after 70M words. It is not a compliant compact-view endpoint and uses the inherited baseline16k coordinate, but it is a clean low-cost masking-factor probe of the leader's WWM7→Token3 component.

Evaluation was launched using:

```bash
GPU=1 MEM_FREE_MB=65000 UTIL_MAX=25 FORCE=0 DRY_RUN=0 bash experiments/archive/representation_and_objectives/training/scripts/eval_wwm_to_token_postswitch_minimal.sh
```

It should evaluate chck_80M, chck_90M, chck_100M zero-shot + Reading and summarize against the fixed-WWM clean-Qwen trajectory. If it exits because the GPU is not isolated or because evaluation tooling fails, repair/launch only when doing so does not delay higher-value training.

Use this evidence only as directional transfer:
- Continue masking-factor route if post-switch official-compatible vectors show sustained gains beyond lower MLM loss, especially in EWoK/BLiMP/Supplement/COMPS without damaging Entity/GlobalPIQA.
- Do not treat lower token-level MLM loss as competence.
- If 80M/90M peak and 100M falls, the token phase may need shorter dose.
- If flat or harmful, do not spend a compact-view retrain on WWM→token alone.

## Selected New-Factor Route
The strongest immediate expensive route is true leader-shape depth under the preserved compact_view_reinvest data mechanism:

- legal40k tokenizer, same compact_view_reinvest 100M stream;
- DeBERTa-v2 12 layers × hidden 384 × 12 heads × intermediate 1280;
- AdamW/cosine as current recipe, fixed WWM at 0.15, fixed visible length 256;
- start with seed43022 only.

Scientific purpose: test whether depth-over-width / leader geometry repairs relational and stateful columns that a wide 8×480 legal40k model cannot, without duplicating support-floor work or confounding with an unverified masking/sequence schedule. This run is distinct from the failed vocabulary axis and directly tests one untested factor that distinguishes the 41.8 leader.

Decision conditions after full official eval:
- If seed43022 deep-fixed gives a coherent vector gain over legal40k seed43022 (not only loss), especially Entity/EWoK/COMPS/SuperGLUE and no severe GlobalPIQA collapse, reproduce with seed43122 and consider adding masking/sequence curricula.
- If depth improves relational/state columns but loses surface columns, inspect training dynamics and consider sequence curriculum or optimizer rather than rejecting depth outright.
- If depth is flat or worse across the relevant columns, deprioritize depth and use WWM→token evidence / support-floor result to choose the next factor.
- If it unexpectedly nears or clears 41.8, protect and reproduce before additional mechanism work.

## Sequence curriculum caution

The current COMPACT_EXPERIENCE/relation bias trainer preflight trainer supports `--seq_len_schedule`, but inspection shows the dataset tokenizes each row at max length 256 and the training loop then slices to the current shorter length while still charging `ex.words` for the full row. This is a staged visible-context truncation intervention, not a faithful leader-style batch-size-scaled 64→256 curriculum. Do not launch a bundled deep+64→256 experiment with the current implementation as if it exactly reproduces the leader factor. First construct and preflight a faithful sequence curriculum that records per-phase optimizer steps, charged words, row/chunk formation, padding/truncation, and batch-size/gradient-accumulation policy.

## Relation-cue masking

Relation boost 2.0 is well audited and remains a valid learning-signal probe, but it should not occupy the first new H100 route. Legal40k already improves EWoK somewhat while the largest loss is GlobalPIQA. Use relation boost after depth/curriculum evidence if a selective EWoK/COMPS/Entity deficit remains.

## Distinct Experimental Factors

- Support-floor and equal-word-tokenizer comparisons require complete official vectors.
- The architecture/recipe comparison begins with legal40k 12x384 fixed-WWM seed43022 and evaluation of the WWM-to-token conversion.
These vary different factors and should not be interpreted as a single additive experiment.

These complementary experiments avoid duplicate vocabulary interpolation and give independent axes for route choice.
