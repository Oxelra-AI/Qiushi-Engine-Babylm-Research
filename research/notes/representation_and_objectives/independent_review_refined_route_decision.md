# route portfolio context alternative binding — independent_review-refined route decision after clean curriculum and adapter evidence

## Additional Scientific Evidence

The previous reflection correctly recentered the research on context-conditioned alternative binding, but it risked narrowing the next portfolio back into corpus-mined completion objects. route portfolio context alternative binding rebuilt the route comparison from the full accumulated evidence, read COMPACT_EXPERIENCE/frontier_consolidation objective and adapter records, used independent_review for independent route generation and criticism, and checked the official-rule sources available in Knowledge/local evaluation code.

The core state remains unchanged:

- No compliant Strict-Small SOTA exists.
- Best completed compliant full endpoint: legal40k 8×480 fixed-256 compact-view AdamW, Overall 41.1406, cheap7 43.1079.
- Visible leader remains around Overall 41.80 / cheap7 43.77.
- The noncompliant inherited-tokenizer endpoint at Overall 42.0331 is mechanism evidence only.
- Clean word-boundary AdamW sequence curriculum scored cheap7 42.3529 and is closed as a crossing route.
- Ordinary joint residual adapters are not a free route: they train, but they shift competence. inference readout shows GlobalPIQA damage mainly direct residual output; EWoK damage mainly altered stock trajectory. Amplitude tuning does not recover the aoa mincontext discrepancy audit key-column anchor.
- The running paired tails `s140_t39_tool1` and `s140_t40_tool1` remain unresolved. They are the final bounded read on fresh AdamW moments and LR reheating, not a license for optimizer-route proliferation.

## Evaluation Constraints

From BabyLM FAQ Knowledge source `data/external/FAQs.md`:

- Line 116: any training objective/regime is permitted as long as data restrictions are followed, and models must provide a function to score a word sequence without extra fine-tuning.
- Lines 126–128: language-learned external tools count toward the 100M word budget; synthetic text is allowed under accounting and can be generated in any legal way if text facilitates learning.
- Lines 139–141: Qwen 2.5/3/3.5 up to 9B are approved teacher-model families.

From the 2026 evaluation material and local official-compatible checkout:

- Strict/Strict-Small submission uploads predictions from the official pipeline, including checkpoint evaluation for full submission (`data/external/BabyLM-Evaluation-2026.md`, lines 160–173).
- The local 2026 pipeline loads strict zero-shot/AoA/fine-tuning models with `trust_remote_code=True` in several places, and the strict checkout includes an HF conversion tutorial for custom modeling files (`data/pristine_official_coordinate/babylm-eval/strict/hf_conversion_tutorial/create_new_hf_repo.py`).
- A public Strict-Small submission record in Knowledge (`Serdar404/RecGPT-10M`) uses custom Transformers code loaded with `trust_remote_code=True`, which makes custom architecture plausible for the challenge, though a final submission would still need end-to-end official-pipeline packaging and prediction generation.

Implication: route A custom side-path architectures and route B synthetic/hand-written role-switch text are not ruled out by the sources, but both need exact accounting, HF save/load, and official-compatible scoring before any endpoint. No official-eval rows or labels should be used to create training data.

## independent_review corrections that changed the portfolio

### 1. Naive detached logit side path is not enough

independent_review verifier argued correctly that a detached side path trained by ordinary CE on `base_logits.detach() + r(h.detach())` can become a static or local output-scale adapter. curriculum result adapter amplitude and paired tail relaunch already showed GlobalPIQA is fragile to direct residual logits: GP is best near zero adapter output. Therefore architecture route A is useful only if its side branch is forced to model context-difference, not generic logit sharpening.

Refined A mechanism:

- Preserve base training exactly: ordinary WWM CE on base logits updates the base DeBERTa-v2 path as before.
- Side branch input uses detached features, so side loss does not alter base gradients.
- Side branch must remove or separately parameterize static vocabulary bias and be evaluated by whether `r(Ca)-r(Cb)` tracks the sign of a target-pair context difference under matched alternatives.
- A branch that improves only held-out MLM loss or frequent-token calibration but not context-difference margins should be rejected before H100 training.

### 2. Role-switch packets need transfer, not just packet self-score

independent_review verifier argued that frozen scoring of synthetic packets only tests whether current models solve those packets, not whether training on them induces general binding rather than template rules. Balanced target counts remove marginal priors but do not remove conditional template priors such as `above -> high` or first-noun shortcuts.

Refined B mechanism:

- Generate small legal role-switch packets only from hand-written grammar and in-budget vocabulary lists, never from official evaluation item text.
- Partition families/templates/entities so a training packet family is withheld from the held-out packet evaluation.
- The first useful screen must test held-out family transfer plus real EWoK variable-swap movement, not seen-template accuracy.
- Before any training, simulate useful WWM density under the actual collator and verify replacement-text costs.

### 3. GlobalPIQA is a severe surface but not yet uniquely isolated as binding

EWoK variable-swap and stable-reversal evidence directly isolate context-conditioned role failure. GlobalPIQA_parallel is severe and persistent, but independent_review correctly warned that pseudo-likelihood normalization, length, frequency, or calibration could contribute. Before treating GlobalPIQA as the same object, run cheap row-overlap and token-contribution anatomy across legal40k/legal16k/depth/inherited endpoints.

## Updated route ranking

### Route A: protected context-difference logit side path (primary)

**Scientific purpose:** test whether context-conditioned alternative competition can be added as a protected readout without damaging the compact-view base trajectory.

**What must be built before training:**

1. An HF-loadable custom DeBERTa-v2 side-path wrapper that can export as `AutoModelForMaskedLM` with `trust_remote_code=True` and reproduce disabled-path logits/loss exactly.
2. A mechanical identity probe: with side branch present but disabled or fully detached, base gradients, base optimizer state updates, RNG/dropout consumption, and first several base parameter updates match the baseline control. One-step equality is insufficient if RNG draw order changes.
3. A frozen residual-content probe on 20M/80M/100M checkpoints: fit side parameters only for tiny legal-text batches, then measure residual mass by base-entropy decile, target frequency, and whether context-difference residuals track held-out alternative-pair signs better than target-only/local-window controls.

**Expensive-work admission:** only a 20M matched-horizon enabled-vs-disabled screen, and only after the three cheap tests above pass. It should compare against the same-horizon anchor and report cheap7 plus EWoK variable-swap/stable failures and GlobalPIQA rank distributions. Continue beyond 20M only if broad columns are preserved while EWoK/GlobalPIQA/Reading avoid the ordinary-adapter damage signature.

### Route B: balanced role-switch experience in the primary MLM stream (primary)

**Scientific purpose:** supply the missing high-density experience where identical alternatives exchange roles under different contexts, because the legal natural corpus is too sparse for clean exchange training.

**What must be built before training:**

1. A 20k–50k-word packet generator with families such as spatial above/below, temporal before/after, transfer giver/receiver, open/closed state, container/support, comparative more/less. It should use common in-budget nouns/verbs and ordinary prose mini-discourses rather than brittle `A above B -> A high` transforms.
2. A provenance manifest: grammar source, vocabulary source from allowed training text, seeds, template IDs, word counts, removed/replaced material candidates, and near-duplicate checks against official held-out text. Training material must not be derived from official eval item text or labels.
3. Actual WWM-density simulation: how many masked targets per 10 epochs truly require role-conditioned prediction, under the same word grouping and mask RNG as the baseline.
4. Frozen checkpoint scoring: target-main-effect-cancelled exchange margins, context erasure, target/packet permutation nulls, and held-out family/entity split. Accept only if current endpoints are unsaturated and target-only/local-template cues are near-null.

**Expensive-work admission:** if the object passes the no-training checks, use a small matched legal replacement pool, not an append. Include a matched natural-text replacement control or, at minimum, quantify that removed material is low-yield by source/length/frequency/readability. First H100 screen should be 20M from initialization or another low-cost design that can compare to a same-horizon anchor. Continue only if held-out packet families and real EWoK variable-swap/GlobalPIQA hard ranks move together without broad damage.

### Route C: cheap mature checkpoint consolidation

**Scientific purpose:** test whether existing legal40k fixed-256 checkpoints contain complementary mature predictions without new training.

**Action:** build predeclared 80–100, 90–100, and 70–100 weighted checkpoint averages from existing legal40k fixed-256 checkpoints, plus a logit-ensemble reference if practical. Evaluate cheap7 and hard surfaces. This should not consume H100 training; it can proceed as a cheap artifact/evaluation line once the active paired tails are read or if CPU/GPU-eval lanes are free.

**Interpretation:** unlikely to close the frontier gap alone, but if it improves fragile surfaces without damage it may supply a stronger base for A/B. Stop if it simply averages the broad-vs-relation tradeoff.

### Route D: selective update geometry held

Continuous Muon had real EWoK signal but broad cost; abrupt switches are closed. Selective core Muon/AdaMuon remains scientifically distinct but should not be the next H100 commitment unless the already-running paired tails deliver a coherent gain or a no-training update-geometry probe predicts joint broad+relation movement. The strategist instruction to avoid an optimizer chain is preserved.

## Immediate next work

1. Once the paired-tail results are available, evaluate cheap7 against 43.1079 using the recorded outputs. If neither improves broad cheap7 together with EWoK/GlobalPIQA hard surfaces, close reset/reheat family without more optimizer variants.
2. Build the Route A mechanical/frozen probe using the existing adapter machinery; the key risk is architecture/gradient coupling.
3. Build the complementary Route B packet generator and WWM-density/frozen-scoring scripts without training. H100 training requires the packet object to pass the no-training tests first.
4. A shared model/result registry should be consolidated because best legal full endpoint (41.1406 legal40k) and aoa mincontext discrepancy audit (41.2578 legal16k) are distinct anchors; future notes must state which coordinate a test uses.

## independent_review files

- Generator integration: `data/external/independent_review01_generator1_integration.md`
- Verifier integration: `data/external/independent_review01_verifier1_integration.md`
- Original branch outputs: `independent_review01_sub1_generator.md`, `independent_review01_sub2_generator.md`, `independent_review01_sub3_verifier.md`, `independent_review01_sub4_verifier.md`
