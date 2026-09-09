# Post-compact-order natural-data test plan (not launched)

## Status

This is a non-launch plan written while the compact ordered-vs-scrambled result is still pending. It incorporates the whole-word copied-control findings and the target-prioritization failure.

The plan is usable only after:

1. matched ordered/scrambled official-selected, channel, training-log, and exposure-split outputs are delivered and read.
2. The source-absent channel readout uses category interactions, not raw source_absent movement alone.
3. The channel exposure reconstruction readiness exposure split is used only if CUDA mask-count reconstruction matches both arm logs exactly.
4. The result is not only GlobalPIQA/Reading movement.

No new GPU work is authorized by this file.

## Scientific boundary

The next experiment, if needed, should estimate a **compact-data marginal**, not a source-absent-target marginal. The live mechanism is that faithful compression naturally changes the text distribution: source-absent compact-side words appear in ordinary compact contexts, along with retained content, copied material, source-side material, higher content density, source-tail coverage, and additional distinct sources purchased by shorter views. Ordinary WWM samples from that distribution without hand-marked origin priority.

earlier analysis is the reason this boundary matters: explicit source-absent target pressure improved the intended local strict-target loss yet lowered the mature selected BabyLM-compatible surface by -1.0529 cheap7 versus spatial repair route status reinvest at 80M. Therefore a positive source-absent local signal should not be translated into explicit mask/loss reweighting.

## Candidate 1: natural compact-data transfer to another bidirectional MLM encoder

### Question

Does compact-view/reinvestment help because of the data distribution itself, or because it is unusually well matched to DeBERTa-v2's disentangled-attention coordinate?

### Design

Train a materially different bidirectional masked-LM encoder, such as a RoBERTa/BERT-style encoder without DeBERTa-v2 relative disentangled attention, on a matched compact-vs-control pair:

- Arm C: faithful compact plus diversity-reinvestment pool.
- Arm R/B: matched noncompact/repetition or breadth control with the same word budget and source/filler accounting.

Keep fixed:

- legal tokenizer/data accounting;
- source identities and source-side mass;
- row schedule and packing;
- ordinary WWM p=0.15;
- total word exposure and LR horizon;
- corruption RNG for shared examples where mechanically possible;
- official-compatible selected cheap evaluation.

The current `experiments/archive/compact_experience/scripts/masking_curriculum_trainer.py` is DeBERTa-only, so this requires a new small trainer or careful extension. It should not be treated as a command-line switch.

### Minimum reliable run

Start with one paired seed at 20M and 40M. Continue both arms to 80M only if stable columns show an emerging distributed signal: |cheap5/cheap6 delta| about 0.1-0.3, or Supplement/EWoK+Entity move consistently even if aggregate is small. A 20M local result alone is not decisive because the original compact benefit matured late.

### Readouts

Primary score readouts:

- cheap6 excluding GlobalPIQA;
- cheap5 excluding GlobalPIQA/Reading;
- Supplement, EWoK, Entity, COMPS, BLiMP individually;
- EWoK+Entity.

Mechanism readout:

- held-out-source compact-side denoising probe with source_absent_content, retained_content, and function_other events matched for BPE length/frequency;
- source_absent interaction versus retained/function controls, not raw loss alone.

### Interpretations

| Result | Meaning |
|---|---|
| compact beats matched control on stable columns and source-absent selectivity transfers | strongest evidence yet for architecture-transferable compact-data principle within bidirectional MLM |
| compact beats matched control without source-absent selectivity | compactness transfers, but density/tail coverage/diversity may be more load-bearing than source-absent labels |
| local selectivity transfers but stable task movement does not | source-absent channel is real but not sufficient for external competence at this budget/architecture |
| DeBERTa-only positive | compact data may depend on DeBERTa-v2's inductive coordinate; general law must be narrower |

## Candidate 2: natural source-absent-density scaling through faithful compression style

### Question

Does the useful compact signal scale when source-absent content density arises naturally from faithful compression style, rather than from loss/mask allocation?

Define:

`d_abs = source_absent_content_words / compact_view_words`.

### Design

Construct matched-source compact-style pools where every source receives faithful fluent compressions at different abstraction levels:

1. low-abstraction mostly extractive compression;
2. current compact style;
3. higher-abstraction faithful compression.

No origin labels may affect sampling, corruption, or loss. Ordinary WWM remains unchanged.

Match or audit:

- total 10M word budget and counted exposure;
- source set and source-side mass;
- compact-view mass;
- retained-content mass;
- reinvested distinct-source allocation;
- row schedule and packing;
- view length and lexical-frequency bins where feasible;
- content density, compression ratio, tail/source-position coverage, fluency, and semantic faithfulness.

### Minimum reliable run

Start with the current style plus the two most separated faithful styles. Train to 20M for local dynamics; carry the two best-matched extremes to 40M, and to 80M only if stable columns show an emerging signal.

### Interpretations

| Pattern | Meaning |
|---|---|
| broad score rises with `d_abs` then saturates/turns over | natural compression-response law, possible optimal novelty regime |
| source-absent denoising rises but broad scores flatten/fall | local channel is real but decoupled from reusable BabyLM competence |
| scores track content density/tail coverage more than `d_abs` | source-absent labels are a correlate of better compact data, not principal cause |
| no structured response after matching | compact benefit depends on exact rewrite distribution or diversity reinvestment, not abstraction density alone |

## Candidate 3: compactness versus diversity reinvestment factorial

The validated compact-reinvestment intervention couples two mechanisms: compact rewrite supervision and extra distinct sources bought with saved words. If the pending ordered/scrambled result does not settle which is load-bearing, a data-level factorial under ordinary WWM is the cleanest causal split:

| View family | Saved-word allocation |
|---|---|
| matched noncompact/extractive | repeated existing sources |
| matched noncompact/extractive | additional distinct sources |
| faithful compact | repeated existing sources |
| faithful compact | additional distinct sources |

Estimate compact main effect, diversity main effect, and interaction:

`E_comp = 0.5 * [(C-A) + (D-B)]`

`E_div = 0.5 * [(B-A) + (D-C)]`

`I = (D-C) - (B-A)`

This advances the principle only if compact rewriting has a stable main effect or a reproducible positive interaction with diversity reinvestment. If only diversity contributes, the transferable lesson is broader source coverage rather than source-absent compact supervision.

## Candidate 4: objective transfer after architecture transfer

Only after the architecture-transfer result is known, hold one encoder fixed and compare ordinary WWM with a content-agnostic random-span denoising objective:

| Data | ordinary WWM | content-agnostic random spans |
|---|---:|---:|
| compact/reinvested | yes | yes |
| matched baseline | yes | yes |

The span sampler must use only positions and RNG, never rewrite-origin annotations. Match exposure, predicted-token mass, packing, initialization, optimizer, and LR horizon.

Interpret by the interaction:

`I_obj = (S_compact - S_base)_span - (S_compact - S_base)_WWM`.

This asks whether compact data helps under another content-agnostic denoising objective. It is cleaner than jumping to T5 or causal generation because those change architecture and objective simultaneously. The prior GPT2 causal null warns against assuming autoregressive transfer, but does not settle masked-objective transfer.

## Common selected official-compatible readout

All candidate runs should be evaluated using:

- cheap6 without GlobalPIQA;
- cheap5 without GlobalPIQA/Reading;
- Supplement;
- EWoK;
- Entity;
- COMPS;
- EWoK+Entity;
- BLiMP to catch grammatical loss;
- held-out-source category interactions for source_absent_content versus retained/function controls.

A route should not continue from a GlobalPIQA-only, Reading-only, or local denoising-only movement.

## Recommendation after review of the whole-word control

If the pending matched result supports the source-absent compact channel on stable families, the highest-priority next scientific test is Candidate 1: a contrasting bidirectional MLM encoder trained on compact versus matched control under ordinary WWM. It directly tests whether the compact-data marginal is DeBERTa-specific or transfers within the masked-LM family. Candidate 2 is the next best test if the ordered/scrambled result suggests that lexical/content selection, not fluent order, is load-bearing. Candidate 3 is useful if compact-vs-diversity attribution remains confounded. Candidate 4 should wait until one model-coordinate result is known.
