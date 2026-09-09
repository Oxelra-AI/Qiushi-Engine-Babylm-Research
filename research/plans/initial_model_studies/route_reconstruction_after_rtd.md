# transfer gain per word gate — Route reconstruction after RTD context-probe failure

## Research state now inherited

The active goal remains BabyLM 2026 Strict-Small Overall SOTA under official constraints, with real pretraining and full official-compatible evaluation. The strongest internal coordinate is still the validated DeBERTa-v2 8×480 WWM endpoint (`wwm_seed43 chck_80M`, Overall about 40.703), not a new mechanism. Current leader gap is concentrated in Entity, EWoK, GlobalPIQA and SuperGLUE; Supplement is not the weakness.

The last Execute block changed the route picture:

- Relation-explicit FineWeb selection: genuine 1M EWoK movement but volatile and non-persistent by 3M; Entity flat.
- Token-aware paired-restatement: true pair worse than orig/shuffled on Entity and EWoK under equal-row/no-truncation control.
- Simple RTD+MLM: at 1M it partially escapes all-original RTD shortcut, but remains weak (replaced recall 0.255, above-majority +0.019), gives EWoK +2.18 while damaging Entity −1.39 and Supplement −2.80, and fails the matched WWM-vs-RTD true-prefix specificity test.
- History probes: hidden states contain remention/history variables, especially at late checkpoints and AMLM, but same-entity history changes do not causally support later MLM content over unrelated-history changes.

Therefore the next mechanism cannot be another data filter, relation-word priority mask, downstream-content mask, generic RTD generator anomaly target, or auxiliary representation/readout objective that can succeed without changing the MLM logits.

## What the next mechanism must do

For a fixed later suffix and fixed masked target, changing the true earlier context must change the gold-token logit more than matched unrelated or shuffled earlier context, and more than it changes under the protected WWM baseline.

A useful next training event has this structure:

\[
\Delta_{\rm specificity} =
[\log p(x_t\mid h^+, s_{\setminus t}) - \log p(x_t\mid h^-, s_{\setminus t})]_{\rm candidate}
-
[\log p(x_t\mid h^+, s_{\setminus t}) - \log p(x_t\mid h^-, s_{\setminus t})]_{\rm WWM}.
\]

The wrongness of the negative example must depend on the earlier context itself, not on local token weirdness, document topic, length, source, generator artifacts, or relation-word presence.

## Candidate mechanisms worth constructing

### 1. CPC-MLM: counterfactual-prefix conditioned logit margin

For natural training snippets, split each example into prefix `h` and later suffix `s`. Select masked targets in `s`. Build a matched counterfactual prefix `h-` from another snippet with similar length/source/rough topic but incompatible continuation. Add a margin on the ordinary MLM gold-token logit:

\[
L_{\rm CPC} = [m - \log p_\theta(x_t\mid h^+,s_{\setminus t}) + \log p_\theta(x_t\mid h^-,s_{\setminus t})]_+.
\]

Single active factor: add this true-prefix versus matched-negative-prefix MLM-logit margin to protected WWM. Same architecture, tokenizer, official corpus, WWM masks, word exposure, seeds, and direct checkpoint evaluation pattern.

Controls:

- protected WWM;
- WWM + CPC;
- same extra forward/compute with ineffective margin, e.g. true prefix used on both sides or loss weight zero while preserving compute path;
- optional random-prefix arm only to test whether hard matching matters, not as the main route.

Small validation:

- seeds 42 and 43;
- exact official 1M and 3M exposures;
- direct checkpoint paths only;
- fast profile at 1M and 3M: BLiMP, Supplement, EWoK, Entity, COMPS, Reading;
- per-domain/item resampling for EWoK and Entity because single-seed fast EWoK is noisy;
- matched intervention probe: true, deleted, shuffled, topic/length/source-matched incompatible, and unrelated prefixes on held-out official examples.

Reason to prioritize: it most directly trains the missing quantity from corrected history content probe and 307: the ordinary MLM gold logit must prefer the true earlier context over a matched wrong earlier context for the same later suffix. It avoids a separate classifier head and avoids generator artifacts.

Main risk: constructing negative prefixes that are genuinely incompatible rather than merely different topic. The first implementation should not try to infer rich semantics; it should begin with tightly matched within-source, near-neighbor prefixes and measure whether the learned effect is specific.

### 2. Prefix-LM / dense continuation auxiliary on the same DeBERTa backbone

Use the same official examples, but add a continuation objective over later suffix tokens conditioned on earlier prefix. The purpose is to create many supervised suffix tokens whose gradients must pass through prefix-to-suffix attention, without RTD class imbalance and without generator replacements.

Single active factor: add a causal/prefix-LM continuation loss on suffix tokens while retaining WWM and the DeBERTa MLM evaluation interface. Architecture, tokenizer, data, seeds and word exposure remain matched.

Controls:

- protected WWM;
- WWM + prefix-LM continuation;
- compute-matched continuation loss with shuffled prefix or prefix detached if implementation permits.

Small validation:

- seeds 42 and 43;
- 1M and 3M exposures;
- compare suffix loss under true versus matched counterfactual prefix;
- matched MLM-logit intervention probe identical to CPC-MLM.

Reason to test after CPC-MLM or in parallel only if implementation is cheap: it tests whether the bottleneck is sparse MLM supervision rather than hard-negative construction. It may improve general language efficiency, but if it increases perplexity-style learning without true-prefix specificity it should stop.

Main risk: implementation changes the training interface and compute more than CPC-MLM, and broad next-token style loss may overfit local suffix patterns rather than stateful context.

### 3. State-channel MLM with prefix state exchange

Insert a small number of learned state tokens at a prefix/suffix boundary. In a state mode, suffix tokens read a fixed-capacity state produced from the prefix instead of directly reading all prefix tokens. Add state-exchange training: for the same suffix and target, the true prefix state must outperform a matched wrong prefix state.

Reason to keep as a construction reserve: it directly attacks the observed disconnect between history in hidden states and history in logits, and differs from WESS because it uses no entity labels or predicted entity routing. It is engineering-heavier and easier to damage Supplement, so it should follow CPC-MLM unless a low-risk implementation is available.

## Immediate next work

Construct CPC-MLM as the primary next experiment. Do not launch full 100M. Do not launch all candidates. The next step should produce an implementable CPC-MLM trainer/probe pair, with exact data mining and matching rules, then run a tiny compile/smoke and a 1M seed42 comparison only after the trainer is inspectable.

Minimal CPC-MLM implementation plan:

1. Reuse `babylm_masked_train.py` data selection, tokenizer, WWM masking, DeBERTa-v2 8×480 config, checkpoint saving, direct checkpoint evaluation.
2. For each batch, split each 160-word example into prefix and suffix by word/token position, choose WWM targets in suffix, and build a negative prefix from another example with similar prefix length and source; initially use batch negatives plus a deterministic fallback pool.
3. Forward true-prefix+suffix and negative-prefix+same-suffix through the same MLM model; compute WWM loss on the true sequence and margin loss on selected suffix targets.
4. Keep auxiliary loss small at first: fixed margin and weight chosen once so CPC gradient norm is visible but not dominant; record margin activation rate and gradient norm ratio.
5. Save exact pair metadata for the first 1M run: source, target token, prefix lengths, negative-source match, logit true/negative before and after training.
6. Run matched WWM/CPC seed42 at 1M only after smoke; evaluate fast profile with direct checkpoint path and run the same matched intervention probe. If CPC fails true-prefix specificity, repair matching/loss before any seed43 or 3M.

## What would stop this route

Stop CPC-MLM as currently formed if any of these occurs in the matched 1M/3M validation:

- candidate minus WWM true-prefix specificity is near zero or weaker than unrelated-prefix sensitivity;
- the effect lives only in an auxiliary score and not in MLM gold logits;
- EWoK improves only at 1M but reverses by 3M as relation-explicit did;
- Entity drops while EWoK rises, repeating the RTD tradeoff;
- Supplement or BLiMP takes a sustained hit.

If CPC-MLM fails cleanly, the next most informative route is prefix-LM/dense continuation, not another data-composition filter or generic contrastive head.
