# route decision mechanism — Frontier route decision after paired continuation eval causal granularity result

This plan chooses the next experiment now that the cleanest granularity measurement in these experiments closed unconditional late WWM→token on the strongest inherited backbone.

## Decision base

- The live target is `wwm_curriculum_simplification_40k`, Overall **41.8**; the strongest inherited complete internal coordinate is **40.7028** (INITIAL_MODEL_STUDIES earlier analysis). The remaining gap is **1.098 Overall**, concentrated in Entity **6.25**, EWoK **5.63**, GlobalPIQA **2.08**, (Super)GLUE **1.53**.
- The three backbone-internal mechanisms did not shift the frontier:
  1. residualized C/S content selection: negative on the matched 4M screen;
  2. same-content easy-to-hard ordering: negative weighted proxy;
  3. unconditional late WWM→token: paired continuation eval isolated causal effect is **negative in aggregate** on the earlier analysis-family DeBERTa-v2 8×480 WWM backbone (equal-7 mean −0.301; legacy weighted screen −0.237). The restart control reproduced the INITIAL_MODEL_STUDIES tail almost exactly, so this is not an optimizer-restart artifact.
- The b256 fixedseq pair eval raw b256 apparent token gain was therefore interpreted correctly as recovery of a degraded fresh-init baseline, not a real frontier shift.

## Route comparison

| route | frontier evidence | expected SOTA potential | cost/risk | decision |
|---|---|---|---|---|
| paired-rewrite / meaning-preserving multi-view data | strongest: current leader card + Edman 2024 paraphrase evidence | highest; only route tied to moving frontier location rather than redistributing supervision | must construct a legal paired corpus and match content/budget carefully | **run first** |
| tokenizer + capacity (40k, 384×12, 12 layers) | real but dataset-conditional; Edman Table 5 shows 40k helps on new paraphrase data but not necessarily original data | high only after the data effect is isolated; otherwise confounded | expensive and changes several load-bearing variables at once | second interaction cell, not first |
| optimizer consolidation (LAMB / Muon-family + tail averaging) | cheap and externally supported, especially tail averaging → (Super)GLUE ~1.6 | bounded; cannot cover a 1.1 Overall gap spread across Entity/EWoK by itself | low | fold in as separable add-on, not main route |

The paired-rewrite route is the only candidate with converging external evidence that it can move the frontier itself rather than slide along it. The key correction from independent verification is that the evidence is strongest on (Super)GLUE, while the leader card suggests EWoK/Entity gains as well; therefore a GLUE-free fast screen is not enough for the decision.

## First experiment: matched paired-rewrite on the fixed earlier analysis backbone

The next experiment must answer one question only: does **aligned paired-rewrite data**, consumed through ordinary MLM on the same backbone, beat matched official or unpaired text under BabyLM Strict-Small accounting?

### Fixed conditions

- Backbone: INITIAL_MODEL_STUDIES earlier analysis-family DeBERTa-v2 **8×480**, WWM recipe, baseline16k tokenizer, seed 43, official-compatible checkpoint cadence, official corpus accounting.
- Word budget: **≤10M whitespace words total for the training corpus**; **≤100M word exposures**; checkpoint schedule must keep the BabyLM naming convention (`chck_1M` … `chck_10M`, then `chck_20M` … `chck_100M`).
- Objective: **plain MLM** over the paired text. No contrastive head, no paired-encoder objective, no auxiliary decoder. This avoids measuring an objective penalty rather than a data benefit.
- Evaluation: **full official-compatible nine-entry coordinate** — BLiMP, Supplement, EWoK, Entity, COMPS, (Super)GLUE, GlobalPIQA, Reading, AoA. The fast screen is not sufficient because the route's strongest supported effect is on (Super)GLUE and AoA and Entity must remain in accepted evaluation formats.

### Candidate data source

Primary immediate candidate: **SynCSE-partial-NLI** (`hkust-nlp/SynCSE-partial-NLI`) plus, if needed, **WikiLarge** (`Nechba/wikilarge-text-simplification`, Apache-2.0) as a secondary aligned simplification source.

Observed facts from local probes:
- `hkust-nlp/SynCSE-partial-NLI`: MIT license, one CSV with `sent0`, `sent1`, `hard_neg`; examples preserve sentence/triple structure and provide a meaning-preserving positive plus a hard negative.
- `Nechba/wikilarge-text-simplification`: Apache-2.0 license, `Normal`/`Simple` aligned simplification pairs, 148,843 train rows.
- `facebook/asset`: CC-BY-SA-4.0, but only validation/test splits are available in the probed repo snapshot, so it is not the first training source.
- `google-research-datasets/paws`: aligned paraphrase-style pairs but labels include non-paraphrases; useful only if filtered to label 1 and if the license/conditions are explicitly handled. Keep as a fallback, not the first choice.

### Arms

1. **A_official**: official 10M BabyLM Strict-Small corpus, baseline recipe. This is the reference arm.
2. **B_pair**: ≤10M words of paired aligned text constructed from SynCSE-partial + aligned WikiLarge, mixed so each original and its rewrite both contribute to the same 10M budget.
3. **C_simple_only**: the same simplified/rewritten sentences **without** their originals, also ≤10M words. This is the content-matched control that distinguishes true multi-view pairing from a mere domain/style shift toward simplified text.

If only two arms can be run immediately, run **B_pair** and **C_simple_only** first, because A_official is already the inherited earlier analysis-family reference; but the three-arm design is the cleanest scientific object.

### Pair construction protocol

- Build a deterministic materializer that:
  - loads SynCSE-partial (`sent0`, `sent1`) and WikiLarge (`Normal`, `Simple`);
  - keeps only aligned positive pairs;
  - optionally stores `hard_neg` as metadata but does **not** feed hard negatives through a contrastive objective;
  - tokenizes / counts words with the exact whitespace-word accounting;
  - selects a ≤10M-word paired corpus with source balance and duplicate checks;
  - writes per-arm JSONL rows with `text`, `words`, `example_id`, `source`, and pair metadata.
- For each pair, prefer keeping both views in the training corpus. If a strict ≤10M total requires sampling, sample pairs, not individual sides, so alignment remains intact.
- Record corpus provenance, license surface, source mix, pair counts, word counts, and duplicate/self-overlap statistics in a `screen_summary.json`.

### Accounting and legality

- The BabyLM 2026 rules count the **sum of all text seen by all training** toward the 10M limit. For Strict-Small, external/generated text also counts toward the 10M word budget. The pair corpus itself must fit ≤10M words; no separate free rewrite budget exists.
- Distillation is not allowed: do not expose an external model's tokenizer, weights, hidden states, or output distribution to the submission model. Using published text pairs is allowed as custom data if the total word budget is respected.
- If any self-generation is later used, the generator must come from the BabyLM-approved external-model list and every generated word must be counted. For this first experiment, prefer already published pair corpora to avoid that dependency.

### Optimizer / tail-average add-on

- Keep the primary backbone and optimizer fixed for the first data test.
- In parallel or immediately afterward, run a **near-free tail-average probe** on the same checkpoints: average the final 20% of checkpoints for each arm and compare averaged vs non-averaged endpoints. This tests the cheap (Super)GLUE-sensitive effect without contaminating the data comparison.

### Success standard

- A positive result must beat the inherited **40.7028** internal coordinate on the **full nine-entry official coordinate**, not merely the fast screen.
- The scientific bar is not just “higher score”; it is a frontier shift: B_pair should improve Entity/EWoK/(Super)GLUE or GlobalPIQA without collapsing Supplement/BLiMP/Reading, and C_simple_only should show that the gain depends on pairing rather than simplified text alone.
- If B_pair fails but C_simple_only succeeds, the leader clue is mostly a domain/style effect, not multi-view alignment. If B_pair succeeds only after the 40k/capacity cell, then the leader edge is a coupling effect between pair data and representation granularity.

## Immediate next work

1. Implement the paired-corpus materializer and run a dry validation on SynCSE-partial + WikiLarge with word counts, overlap checks, source mix, and pair integrity.
2. Train the first matched arms on both H100s under the earlier analysis-compatible full-cycle recipe.
3. Evaluate on the full official-compatible nine-entry coordinate and compare against the 40.7028 coordinate and 41.8 live target.

## Research state after this decision

The evidence does not support further unconditional masking-granularity variants or text-scoring selection as the main route. The active research question is now: **does aligned meaning-preserving multi-view data expand the small-model competence frontier under BabyLM Strict-Small, and if so, is the effect data-alone or coupled to 40k/capacity?**
