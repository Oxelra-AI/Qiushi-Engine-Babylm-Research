# sequence curriculum loop measurement — WWM→token directional evidence and sequence-curriculum loop measurement

CPU-only interpretation of the completed WWM-to-token evaluation. No new GPU training was launched.

## 1. WWM→token late-switch is not a compliant crossing route by itself

The completed evaluation scored the already-trained clean-Qwen WWM→token post-switch
checkpoints (chck_80M/90M/100M) on the inherited 16k coordinate, no-AoA zero-shot +
Reading, matched to the clean-Qwen fixed-WWM trajectory.

Source: `experiments/archive/representation_and_objectives/data/wwm_to_token_postswitch_fullzeroshot_reading/wwm_to_token_vs_clean_qwen_delta_summary.json`

Matched equal7 delta (WWM→token minus fixed WWM), same seed, same data, same stream:
- chck_80M: +0.9486 (GlobalPIQA +4.99, EWoK +1.80, Supplement +1.44; Entity −0.86, Reading −0.82)
- chck_90M: +0.4229 (EWoK +1.79, GlobalPIQA +1.52; COMPS −0.49, Reading −0.49)
- chck_100M: +0.0314 (EWoK +1.48; GlobalPIQA −0.47, Reading −0.59, COMPS −0.47)

Reading of the evidence:
- The late token-masking switch has a real early advantage that is largest at 80M and
  monotonically decays to essentially zero by 100M matched exposure. EWoK stays +1.5
  at endpoint, but that gain is offset by GlobalPIQA/Reading/COMPS losses so equal7 is
  flat.
- This is on the noncompliant inherited 16k coordinate and does not include SuperGLUE
  or AoA, so it cannot be an endpoint score. It is directional masking-factor evidence
  only.
- Conclusion: late WWM→token masking alone is not a compliant SOTA crossing route. It
  should not be carried as a standalone next expensive move. Its endpoint EWoK gain is
  the only durable column effect and would only matter combined with a route that does
  not lose GlobalPIQA/Reading.

## 2. Direct current-loop sequence measurement (confirms family specific tokenizer predictor attribution)

Source: `experiments/archive/representation_and_objectives/data/sequence_curriculum_loop_measurement/sequence_curriculum_loop_measurement.json`

First effective batch (256 rows, 39,370 debited words), legal40k tokenizer, using the
exact legal40k accum training completion/relation bias trainer preflight dataset + 64-row microbatch + combine path and the WWM mask
function (no model forward):
- L64: sees 12,119 of 38,915 full-256 word groups (0.311) while debiting the full row
  words; sampled WWM selects 1,781 groups / 2,436 tokens.
- L128: sees 0.620 of full-256 word groups.
- L256: sees 1.000.

This is a direct measurement, not code inspection: the trainer pops `batch.words` and
sums it into `cumulative_words` before slicing input_ids/attention_mask/word_group to
the stage length, so suffix word groups hidden by L64/L128 still consume word exposure.
The existing `seq_len_schedule` prefix path therefore cannot be credited as the leader's
batch-scaled 64→256 sequence factor.

## 3. Compliant faithful chunking regime (defined, ratios measured)

Regime: each stage epoch partitions the same 10M whitespace words into stage-length
word-boundary chunks; every whitespace word is charged exactly once per epoch. A
10-epoch 64/128/256 curriculum therefore stays at 100M charged words — compliant. The
extra target tokens come only from revealing suffix words that prefix slicing hid, not
from adding words or epochs.

legal40k, schedule 64×3/128×4/256×3: faithful chunking gives 1.609× active tokens at
1.072× optimizer steps, 100,000,000 charged words. minfreq25: 1.613× at 1.085×.
legal40k, schedule 64×7/256×3: 1.976× active tokens at 1.028× steps. minfreq25: 1.990×.

Per-epoch hidden-by-prefix words: L64 0.692, L128 0.388, L256 0.012 (legal40k).
Faithful chunks per epoch with inverse row-batch: L64 250,872 rows at batch 1024;
L128 139,524 at batch 512; L256 75,469 at batch 256.

Measurement repair note: standalone leading-space `Ġ` tokens with whitespace-only
offsets are attached to the following whitespace word (they are visible non-special
candidates in the trainer), matching family specific tokenizer predictor ratios exactly; zero unassigned offsets
remain in the accounting sense.

## Route implication

- Late WWM→token masking alone: closed as a standalone compliant crossing route
  (endpoint equal7 delta ≈ 0).
- Faithful 64→256 sequence curriculum: a genuine, currently-unbuilt leader factor that
  raises target-token exposure ~1.6× at ~1.07× steps while remaining ≤100M words. It is
  a real training intervention, not a free re-slice. It should be built and launched
  only after the running depth vector (`s74_t5_tool1`) and support-floor vector
  are read, so the next expensive run attacks the actual remaining deficit rather than
  bundling untested factors.

JSON: `experiments/archive/representation_and_objectives/data/sequence_curriculum_loop_measurement/sequence_curriculum_loop_measurement.json`
Delta: `experiments/archive/representation_and_objectives/data/wwm_to_token_postswitch_fullzeroshot_reading/wwm_to_token_vs_clean_qwen_delta_summary.json`
