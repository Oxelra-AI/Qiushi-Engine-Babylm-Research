# Post-WESS evidence and route reassessment

## Scientific judgment

WESS should remain closed. wess route closed showed that the unlabeled-address version cannot parse the query entity on independent templates, so the causal slot mechanism disappears when gold addresses are removed. exported base threeway official interpretation also showed that annotated WESS auxiliary training does not move the official Entity/EWoK target cluster and that GlobalPIQA movement is reproducible by no-address auxiliary training. Continuing WESS would keep optimizing a synthetic success condition that does not operate on official inputs.

The remaining official gap is concentrated in Entity Tracking, EWoK, GlobalPIQA, COMPS, and SuperGLUE. Protected DeBERTa 8×480 WWM already leads the public leader on Supplement and Reading. The next experiment must therefore look for a mechanism that can raise Entity/EWoK/GlobalPIQA without spending the protected advantages.

## Route ordering

1. **Route B: morphology/surface adapter on protected DeBERTa** — first execution.
2. **Route A: seq512 protected DeBERTa WWM** — second, after checking how much official training text actually crosses the 256-token window.
3. **Route C: approximate influence/prototypicality ordering** — useful later, but its proxy variables are not yet well separated.
4. **Route D: adaptive WWM** — implementable, but the available literature points more toward Entity/GLUE than EWoK and has masking-volume confounds.

## Why Route B first

The morpheme-aware BabyLM source is the only refreshed source that directly reports large movement in both EWoK and Entity. In its 10M GPT-BERT table, BPE to Morfessor changes EWoK from 50.40 to 70.01 and Entity from 25.30 to 62.17, while also showing a severe Supplement/Reading penalty. Those numbers cannot be copied into our expected effect size because the paper changes the tokenizer, wordpiece inventory, segmentation length, mask targets, and model family. But it is strong directional evidence that shared surface/morphological structure can alter exactly the target cluster that remains unsolved.

The local negative 40k tokenizer result does not refute this. It tested a larger frequency-based ByteLevel vocabulary and repeatedly damaged Reading; it did not test morphology-aware sharing while keeping the baseline16k tokenizer and official input pipeline intact.

Route B is scientifically sharper than Route A as the first screen because it distinguishes a real mechanism question: does cross-token surface sharing improve Entity/EWoK/COMPS beyond a parameter-matched extra-capacity adapter? Route A is cheaper, but long windows have a less direct path to EWoK and need a truncation/subsequence analysis before their mechanism premise is strong.

## Required Route B design

Run at least three arms at the protected 8×480 DeBERTa WWM 10M scale:

1. Existing or freshly matched protected baseline: DeBERTa-v2 8×480, baseline16k, WWM, official corpus.
2. Surface-sharing adapter: add a small feature-derived vector to the input embedding from normalized tokenizer string features such as char n-grams, prefixes, suffixes, and simple affix flags. Use a learned scalar or per-channel scalar so the model can ignore it if harmful.
3. Token-ID adapter control: same parameter count, same initialization, same projection shape, same insertion point, but features are token-specific with no surface sharing.

The comparison that matters is surface adapter minus token-ID adapter on Entity/EWoK/COMPS and on protected strengths. If both adapters move similarly, the effect is capacity or optimization. If the surface adapter alone moves Entity/EWoK/COMPS while Supplement/Reading/BLiMP stay acceptable, the route is live.

## Implementation invariants

- Keep baseline16k tokenizer and existing WWM grouping.
- Keep official corpus only for the first screen.
- Keep exact whitespace word exposure accounting and checkpoint save/load compatibility.
- Match random seeds, document order, optimizer settings, masked-word count, and learning-rate trajectory across arms as closely as the trainer allows.
- Normalize token strings consistently: strip byte/space artifacts only in a documented way, preserve case choices, and define what happens to punctuation, digits, and special tokens before training.
- Keep the surface basis fixed and learned only through the projection; do not train a separate tokenizer or external segmenter unless its text exposure is counted.
- Record feature coverage, collision rate if hashing is used, number of active features per token, learned scalar magnitude, and surface-feature usage by frequency band.

## Measurements needed before any scale-up

Official-compatible columns at 10M:

- BLiMP
- Supplement
- full EWoK with the established word-tokenize path
- Entity Tracking
- COMPS
- GlobalPIQA parallel and nonparallel separately, plus mean
- Reading eye-tracking and self-paced separately, plus mean
- ordinary MLM validation loss

Mechanism-focused measurements:

- surface arm minus token-ID arm per column;
- per-phenomenon BLiMP/Supplement movement;
- Entity subgroups regular/ambiref/move_contents and operation depth if reports expose them;
- EWoK per UID/context/contrast groups if reports expose them;
- GlobalPIQA parallel vs nonparallel separately;
- frequency and token-length stratified MLM accuracy/loss;
- ablation of the learned surface contribution at evaluation, if easy to implement.

Scale to 100M only if the surface arm has an Entity/EWoK/COMPS advantage over both baseline and token-ID adapter without a large loss in Supplement/Reading. A GlobalPIQA-only improvement is not enough; exported base threeway official interpretation showed such movement can be unrelated to the missing mechanism.

## What Route A needs before it runs

If Route B construction stalls, Route A can be executed quickly, but first measure how much the current official example construction actually truncates at 256 tokens and whether truncated spans contain repeated entities, multi-sentence context, or long-range references. Route A changes training window/packing, not merely position capacity: protected models already use max_position_embeddings 512. A seq512 run must record word exposure, masked predictions, optimizer steps, and LR position carefully because fixed word exposure and fixed update count cannot both remain identical if batching changes.

## Final route decision

The next execution should construct Route B. It is the best available post-WESS test of a mechanism plausibly tied to Entity/EWoK, uses official unlabeled input, avoids gold-route proxies, and includes a control that can separate representation sharing from mere capacity.
