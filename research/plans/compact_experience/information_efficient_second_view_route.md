# second view route compliance and source basis — information-efficient same-window second views

## Why the route changed

The high-LR LAMB interaction screen did not show the broad, persistent optimizer × clean-Qwen paired-data interaction that would justify another serial extension of the public leader recipe. At the no-AoA 10M screen, interaction equal7 was negative at `chck_3M` (-0.5529), negative at `chck_5M` (-0.2871), and only mildly positive at `chck_10M` (+0.3036), with the late positive value partly GlobalPIQA-driven and Supplement still negative. This ends the current search through remaining leader components as the main route.

The positive anchor that remains is the same-window second-view mechanism: clean-Qwen aligned pairs beat selected-original duplication and separated-window coexistence under the same seed, while shuffling correspondence damages the effect. crossview denoising design then showed that forcing particular cross-view mask targets is worse than ordinary WWM. The current best interpretation is therefore not “change the objective” or “copy the leader,” but “improve the relation carried by the second view itself.”

## New scientific object

A useful second view should spend synthetic words on relation information that the small model can bind to a real official sentence in the same attention window. The object is not paraphrase dose; it is relation density under a strict word budget:

\[
\text{relation density} = \frac{\text{faithful, specific, complementary relations exposed next to official text}}{\text{synthetic words}}.
\]

The current clean-Qwen view is positive but inefficient:

- 37,594 selected pairs use 1,656,800 pair words in the 10M pool.
- Generated side words: 807,787, or 8.08% of the whole 10M budget.
- Rewrite/source length ratio mean: 0.966, median 0.963.
- 33,325/37,594 pairs are near length-preserving; 10,449 pairs have content overlap ≥0.82.
- The same pair-word budget could, in simple coverage arithmetic, expose roughly 50.8k-58.7k originals if the generated view length were 0.35-0.65 of the original, rather than 37.6k pairs now.

This arithmetic is not a result by itself. It only defines a route worth testing: can a faithful shorter second view keep or improve the positive same-window effect while exposing more distinct official sentences and more relation instances?

## Transformation families worth comparing

### 1. Proposition capsule

One or two short natural English sentences preserving the core event, roles, negation, modality, quantities, dates, names, comparisons, and essential relations, while deleting redundant detail. This tests whether relation-dense compression can preserve the same-window benefit with more distinct official originals per synthetic word.

Two intensities are prepared:

- `capsule_065`: target 0.60-0.70x original length.
- `capsule_050`: target 0.45-0.55x original length.

Expected positive columns: Entity, EWoK, SuperGLUE-like inference, some GlobalPIQA/COMPS when role and property relations are preserved. Main risk: losing natural grammar breadth or Reading if the view becomes summary-like or telegraphic.

### 2. Near-length paraphrase anchor

A fresh near-length prompt on the same originals as the capsule arms. This recreates the current positive family on a shared source cohort, so later screens can distinguish transformation form from source selection.

### 3. Typed relation QA

One short question-answer pair about a relation explicitly present in the original. This tests whether making a relation queryable gives value beyond a compact declarative view. It is risky because template language can reduce natural distributional diversity and shallow questions may only repeat anchors. It should be compared to a declarative capsule before any long training.

### 4. Later structural alternation and bridge families

Two additional families are proposed for later use if the first panel is promising: single-operation structural alternation (voice, clause order, referent explicitation, nominalization/verbalization) and two-official-sentence bridge rows. These are mechanistically distinct from compression, but require more specialized construction. They should follow after the capsule panel identifies acceptable fidelity and length behavior.

## Immediate small panel

Created by `scripts/make_transform_panel_prompts.py`:

- 3,000 shared official originals sampled from `data/mixture/official_pool.jsonl`.
- Source counts approximate official source-word proportions: open_subtitles 685, gutenberg 767, childes 852, simple_wiki 460, bnc_spoken 229, switchboard 7.
- Length bins: 12-17 words 1,692; 18-24 words 707; 25-35 words 405; 36-55 words 196.
- Four transforms per original: `near_paraphrase`, `capsule_065`, `capsule_050`, `typed_relation_qa`.
- Total prompts: 12,000.
- Prompt files:
  - `data/transform_panel/panel_originals.jsonl`
  - `data/transform_panel/panel_prompts.jsonl`
  - `training/data/transform_panel_prompts_shard0.jsonl`
  - `training/data/transform_panel_prompts_shard1.jsonl`

The panel is intentionally small. It is for generation quality and route selection, not for model pretraining.

## After generation

The generation design uses two shards, whose completion is not established in this note:

- `transform_panel_qwen_shard0`
- `transform_panel_qwen_shard1`

After it completes, run:

```bash
cd experiments/archive/compact_experience
python -B scripts/audit_transform_panel.py
```

Expected output:

- `data/transform_panel/audit/transform_panel_audit.json`
- `data/transform_panel/audit/accepted_panel_records.jsonl`
- `data/transform_panel/audit/rejected_panel_records.jsonl`
- `data/transform_panel/audit/stratified_examples.json`

The script checks surface fidelity and usability without benchmark labels: length, content overlap, entity and number retention, negation and modality retention, prompt artifacts, QA format, repetition, and accepted shared-original sets. It cannot prove semantic adequacy; it tells whether a family is viable enough for deeper review and a real training screen.

## Training screen design if a family survives generation review

The first training screen should not increase pair dose. It should fix the current pair-word budget (1,656,800 words) and same-window topology, then compare at least:

1. existing clean-Qwen near-length corpus or a fresh near-length shared-cohort corpus;
2. `capsule_050` or `capsule_065` on the same originals, with saved pair words replaced by the same official filler rather than by repeated pairs;
3. the same capsule family with expanded distinct-original coverage at the same pair-word budget;
4. selected-original duplication for the capsule cohort;
5. separated-window capsule control if the same-window arm shows promise.

The key contrast is two-step:

- same originals: transformation form and length;
- more originals at the same pair-word budget: coverage and relation-density gain.

The 10M screen should use the trusted fixed coordinate unless a later construction reason changes it: DeBERTa-v2 8×480, baseline16k, AdamW, fixed seq256, WWM, seed43022, no AoA/CDI internals, official-compatible no-AoA columns plus Reading. If a compact view keeps the same-window benefit and avoids BLiMP/Supplement/Reading/COMPS damage, use a second seed before any 100M run. Any 100M candidate still needs full nine-column evaluation.
