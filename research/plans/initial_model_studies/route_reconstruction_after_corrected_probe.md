# earlier analysis — Route Reconstruction after the Corrected History/Content Probe

## Current scientific state

The recent internal score improvement is real but not the mechanism needed for SOTA:

- Current best internal coordinate: `wwm_seed43 chck_80M`, Overall ≈ **40.703**.
- Protected seed42 100M: Overall ≈ **40.527**.
- Public leader reference: Overall ≈ **41.801**.
- Remaining gap: ≈ **1.10 Overall**, concentrated in **Entity −6.25**, **EWoK −5.63**, **GlobalPIQA −2.07**.
- The 40k tokenizer run does **not** explain the leader’s Entity score: official40k gave Entity 22.16, below the protected 16k reference.

Thus endpoint choice and seed variance are useful, but the central problem remains unchanged: create durable entity/relation/state and commonsense ability under strict-small training.

## What official remention probe changed

### official remention probe
A lexical-split remention probe found full-history hidden states classify first mention vs later mention better than local-only states. The signal is stronger for AMLM, especially at 100M.

This shows that some discourse/history geometry is readable, not that the model uses relation-specific state for prediction.

### corrected history content probe
The corrected probe fixed the cf smoke offset bug, verified identical target/content/local text, and added an equal-distance unrelated-history control. It measured:

\[
I_{\rm rel}
=
[\ell(y\mid x_{\rm entity\text{-}cf})-\ell(y\mid x)]
-
[\ell(y\mid x_{\rm unrelated\text{-}cf})-\ell(y\mid x)].
\]

Across eight WWM/AMLM checkpoints, all content effects were negative:

- wwm42_40M −0.0024
- wwm42_100M −0.0078
- wwm43_40M −0.0102
- wwm43_100M −0.0127
- amlm42_40M −0.0026
- amlm42_100M −0.0193
- amlm43_40M −0.0038
- amlm43_100M −0.0079

The fraction of samples where same-entity replacement exceeded unrelated replacement was only 0.20–0.28. Late checkpoints show some same-entity hidden-state divergence, but it is not converted into masked content prediction under this target.

### Route consequence

Do **not** launch:

- MCEC from the flawed cf smoke evidence;
- RMEC 100M;
- AMLM scaling;
- self-distillation as a main route;
- generic entity/semantic contrastive objectives that can solve by identity, topic, or local noise.

The next mechanism must create a setting where relation-specific prior history changes output probabilities, and the test must separate that effect from generic upstream perturbation.

---

## Selected next object: official-text minimal state counterfactual ranking

The most valuable immediate question is:

> In official BabyLM text, can a model use an earlier **state-changing fact** about an entity to rank the correct later state/content token over a matched counterfactual token, when the local continuation is identical and nuisance controls are matched?

This is stronger than corrected history content probe because it does not use arbitrary content after a remention. It targets slots whose later word is tied to an earlier explicit relation/fact.

### Target case form

Find or construct from counted official text passages cases with:

1. An early state/fact sentence: entity `e` has relation/filler `r=y1`.
2. A later local continuation around the same entity with a masked state/content slot whose observed filler is `y1`.
3. A counterfactual early fact replacing `y1` with matched `y2`, while leaving the later local continuation identical.
4. A matched unrelated-history edit at similar distance and token length.
5. A state-preserving control that changes surface form but not `y1`.

Then score a difference-in-differences:

\[
R=
[\log p(y_1\mid h_1,\ell)-\log p(y_2\mid h_1,\ell)]
-
[\log p(y_1\mid h_2,\ell)-\log p(y_2\mid h_2,\ell)],
\]

and its unrelated-control analogue. A useful state-specific signal requires positive `R` and positive `R - R_unrelated` on held-out entities/fillers/templates.

### High-precision relation families to mine first

Use deterministic official-text patterns; do not use evaluation data.

1. **Location / preposition state**
   - Early: `E is/was/lived/stayed/went/sat/stood in/at/on Y`.
   - Later: `E ... in/at/on [MASK]` or local continuation where `Y` is the observed filler.

2. **Possession / holder state**
   - Early: `E has/had/held/carried/took/found/brought Y`.
   - Later: `E ... [MASK]` where `Y` is repeated as the object/filler.

3. **Container / part-whole / availability**
   - Early: `Y was in/on/inside E`, `E contained Y`, `E opened/closed Y`.
   - Later: state phrase repeats the object/state filler.

4. **Attribute / role facts**
   - Early: `E is/was a Y`, `E became Y`, `E was called/named Y`.
   - Later: appositive/role phrase requiring `Y`.

5. **Action consequence candidates**
   - Early: `E broke/repaired/opened/closed/moved Y`.
   - Later: a state word (`broken`, `open`, `closed`, place/object filler) appears in relation to the same entity/object.
   - This family is higher risk and should be separated, not pooled blindly.

### Case quality checks

Every candidate must pass:

- target `y1` and counterfactual `y2` have matched tokenizer length, coarse frequency band, capitalization, and grammatical type;
- `y1` is not locally leaked in the later window except at the masked target;
- later local context is identical across original, state-changing counterfactual, state-preserving control, and unrelated-history edit;
- edit distance, character/token length, and isolated marginal pseudo-loss of the changed early span are recorded;
- entity/filler/template splits are available for held-out tests;
- relation family is recorded so effects are not hidden by averaging incompatible phenomena.

If fewer than about 150 high-quality cases survive, the inventory itself becomes an evidence result: natural official text may not contain enough clean relation-conditional slots for this route, and training should not be launched from weak mined pairs.

---

## Stage 0: pretraining-free state-ranking test before any H100-scale training

### Models to evaluate

Start with existing checkpoints; no training:

- protected WWM seed42: 40M, 80/100M if available;
- WWM seed43: 40M, 80M, 100M;
- AMLM seed42/43: 40M and 100M;
- optionally public leader checkpoint if locally loadable, as a phenotype reference.

### Metrics

For each relation family and split:

1. `R_state`: state counterfactual ranking score above.
2. `R_unrelated`: matched unrelated-history ranking score.
3. `R_extra = R_state - R_unrelated`.
4. Local-only score with history removed.
5. State-preserving surface edit score.
6. Bootstrap confidence intervals over passages.
7. Correlation across checkpoints with Entity/EWoK/GlobalPIQA changes (exploratory only).

### Decisive outcomes

**Positive:** `R_extra > 0` with held-out entity/filler/template generalization and family-specific consistency.

- Then the model has some task-relevant relation/history signal, but the ordinary MLM objective does not emphasize it. A short objective/interface experiment is justified.

**Negative:** `R_extra ≤ 0` even after matching control severity and selecting genuine state slots.

- Then the current checkpoints do not have an output-usable state signal. The next route must form state through a stronger interface or experience structure, not extract it from existing DeBERTa geometry.

---

## Stage 1: causal-use intervention if Stage 0 is positive

Before training, test whether the signal is causal to output:

1. **Attention/path blocking:** block target-position attention to early state span vs matched unrelated span.
2. **Activation patching:** patch early state span, event boundary, entity span, and target span between original and state-cf conditions.
3. **Entity/state swap intervention:** swap target entity’s state representation vs other entity’s state representation, if a reliable entity span can be isolated.

A real mechanism should show that perturbing the relevant state span changes the `y1/y2` log-odds more than perturbing unrelated spans. If the signal is only readable but non-causal, do not train a side head that can remain private.

---

## Stage 2: short matched training only if Stage 0/1 support it

The first training should be **10–30M**, not 100M.

### Preferred objective: State-Contrastive Delayed Prediction (SCDP)

For each mined case:

\[
L_{\rm SCDP}
=
-\log
\frac{
\exp s(h_1,\ell,y_1,y_2)/\tau
}{
\exp s(h_1,\ell,y_1,y_2)/\tau+
\exp s(h_2,\ell,y_2,y_1)/\tau+
\sum_{u}\exp s(h_u,\ell,y_u,y_1)/\tau
}.
\]

In practice, use ordinary MLM logits for `y1/y2` at the masked later slot; avoid an auxiliary-only private classifier unless it is tied to the MLM hidden state/logits.

### Matched arms

1. **WWM control.**
2. **WWM + random delayed contrast.** Same number of contrastive pairs, random/mismatched targets.
3. **WWM + topic/unrelated matched contrast.** Controls generic passage perturbation.
4. **WWM + hard state contrast + delayed masked reconstruction.** The real SCDP arm.

If Stage 0 suggests AMLM has more state-readable geometry, a second 2-arm continuation can start from AMLM 40M, but only after the WWM-first mechanism has a signal.

### Evaluation

At 5/10/20/30M:

- official 7-column zero-shot/Reading;
- state-counterfactual ranking inventory;
- layer mixing learning response result-style unseen-entity/template state probe if relevant;
- local-only and unrelated-control measurements;
- Supplement/Reading preservation.

Continuation needs all of:

- state ranking `R_extra` becomes positive or increases materially over controls;
- Entity/EWoK/GlobalPIQA move in the same direction and persist beyond the early phase;
- Supplement/Reading degradation is small;
- gains are not explained by random/topic contrast arms.

---

## If Stage 0 is negative: stronger interface routes

If no existing checkpoint ranks state-conditioned continuations, then the research should move to state formation, not readout extraction.

The two most serious routes are:

1. **Event-state bottleneck on official-derived passages**
   - Parse high-confidence event/state facts into low-dimensional relation slots.
   - Use shared transition operators for location/possession/container/action consequences.
   - Train only after frozen probes show the targets have acceptable precision.
   - All generated/modified text or extra views count under official word exposure if used for training.

2. **Causal event side-channel re-injected into DeBERTa**
   - Keep WWM bidirectional path to preserve Supplement/Reading.
   - Add a narrow prefix event stream that forms state before the later masked slot.
   - Use SCDP/hard state contrast as the training signal, not generic causal LM.
   - Compare against an equal-parameter bidirectional adapter and a causal-stream-without-state-objective control.

These are stronger than MCEC/RMEC because relation history is given a distinct computation path and an intervention target. They also avoid WESS/R1’s failure mode by deriving facts from official text and forcing predictions through masked-output consequences rather than side labels or annotated addresses.

---

## Immediate next work

The proposed implementation is the Stage 0 inventory/ranking test:

`scripts/state_counterfactual_inventory.py`

Minimum output files:

- `data/state_counterfactual_inventory.jsonl`
- `data/state_counterfactual_inventory_summary.json`
- `data/state_counterfactual_ranking_results.json`
- `notes/state_counterfactual_ranking_results.md`

First run target:

- mine up to 3000 official documents;
- retain up to 300 high-quality cases;
- evaluate WWM seed43 80M and 100M first, plus protected seed42 100M;
- if yield and signal are nontrivial, extend to AMLM and public leader.

No pretraining should start until this Stage 0 result is known.
