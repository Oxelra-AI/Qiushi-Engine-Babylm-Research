# Format-attribution controls

## Coherent-unsplit-special result now read at two seeds

Source files:

- `research/documents/relation_learning/data/coherent_special_sofar/coherent_special_sofar.md`
- training summaries under `experiments/archive/relation_learning/data/format_replay_corrected/coherent_unsplit_special/seed98097` and `seed98098`
- per-target payloads under `experiments/archive/relation_learning/data/eval_format_replay_corrected/coherent_unsplit_special/eval`

Both coherent-unsplit-special seeds trained cleanly: 101 updates, 3,992,800 words, first/last target ratios near 0.15, initial private-on/off CE equality, and final coherent readout KL at scale1.0 of 0.0032269 and 0.0037048. The endpoint nevertheless does not produce a broad format lift.

Seed 98097 completed cheap7: 43.5000, which is -0.45945 against chck82 and -0.68143 against coherent86_s43022. Column movements against chck82 were BLiMP +0.3787, Supplement -1.9478, EWoK -0.3655, Entity -0.3340, COMPS +0.0688, GlobalPIQA -0.9877, Reading -0.0287. Seed 98098 repeats the important face of the pattern: BLiMP +0.3187, Supplement -2.1978, EWoK -0.8355, COMPS +0.0688, Reading -0.0837, with Entity +0.4060 and GlobalPIQA +1.4723. Because Supplement and EWoK damage is large and repeated, coherent-unsplit-special was not admitted into a composed candidate. Adapter-amplitude tuning was not part of the proposed repair route at this stage.

## What the result changes about the format hypothesis

The original dead-weight hypothesis for official special tokens was too simple. The added tokens give the private branch a real channel, but the channel is not neutral: it lifts BLiMP while damaging Supplement and EWoK. The coherent readout used by the trainer was too narrow as a preservation measurement. It stayed small while official short-format columns moved strongly, so every format endpoint now needs a no-gradient short-row readout beside the coherent-row number.

A fast bounded readout on 32 matched examples was produced at `research/documents/relation_learning/data/short_format_readout_fast/short_format_readout_fast.md`. It compares private-on/private-off KL on the same content as coherent rows and as isolated short rows. At alpha0.75:

| endpoint | coherent+special KL | short+special KL | ratio |
|---|---:|---:|---:|
| coherent86_alpha075 | 0.00082465 | 0.00183627 | 2.227 |
| coherent_special_98097_alpha075 | 0.00193546 | 0.00974691 | 5.036 |
| coherent_special_98098_alpha075 | 0.00240540 | 0.01199229 | 4.986 |

This is direct evidence that the private branch can diverge several-fold more on short isolated rows than on coherent rows even when trained on the same suffix words. The full 256-row version had not been run at the time of this note. Its original analysis entrypoint is `experiments/archive/relation_learning/scripts/short_format_readout.py` using `endpoints_alpha_only.json`.

## Pending attribution controls

The following two controls had been submitted and were pending when this note was written:

1. No-special coherent control: `experiments/archive/relation_learning/scripts/coherent_no_special_control_sequence.py`. It trains the exact coherent86 suffix through the earlier analysis word-paced trainer with `add_special_tokens=False` at seeds 98097 and 98098 and evaluates cheap7. If this recovers the coherent replay profile, the Supplement loss in coherent-unsplit-special is attributable to added special-token exposure. If it also loses Supplement, the earlier analysis trainer implementation itself pushes the format family downward and isolated/half-format results should not be read as pure row-format effects.

2. Special-embedding-only control: `experiments/archive/relation_learning/scripts/special_embedding_only_sequence.py`. It trains only the `<s>` and `</s>` embedding rows of the protected chck82 model on the same coherent suffix with added special tokens, then evaluates cheap7 at two seeds 98197 and 98198. Because DeBERTa ties input embeddings and decoder weights, the two output rows move too; the result is a restricted two-row test of whether special-token mismatch can be repaired without a full private branch, not a pure input-only intervention.

## Evaluation validation

The repaired AutoModel path had been independently validated through real `AutoModel.from_pretrained`, with exact hidden-state agreement for coherent86 and dense repaired checkpoints and a stock original-path negative control. Shared AoA evaluation computes common ancestral surprisals once, labels endpoints by actual exposure such as `endpoint_86.005295M`, and distinguishes measured zero from placeholder zero. Full AoA manifests were still pending for the comparisons in this note.

## Outstanding attribution tests

The following measurements were still needed for the corresponding interpretations:

1. Completed format-arm results, interpreted under the same fixed admission criterion.
2. The no-special control before attributing coherent-unsplit-special loss to special tokens.
3. The special-embedding-only control to decide whether restricted special-token repair is useful or the mismatch hypothesis should be withdrawn.
4. Faithful v4 SuperGLUE seed42/seed43 comparisons.
5. Full AoA and repaired SuperGLUE comparisons.

The central practical target is still a legal Strict-Small v5 that beats faithful v4 through a real training improvement. The current coherent-special result is a negative training lever and a positive scientific finding about how a small private branch can use unconstrained special-token channels on short evaluation inputs.
