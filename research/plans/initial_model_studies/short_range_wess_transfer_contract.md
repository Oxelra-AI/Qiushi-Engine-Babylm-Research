# earlier analysis — Short-range DeBERTa+WESS transfer experiment contract

## Evidence boundary

v4 (wess v4 address challenge results) proved the persistent entity-addressing algorithm is real and causally necessary in the micro-world: destroying addressing (eventwise_random, wrong_entity) collapses accuracy and interventions; consistent permutation preserves it (swap 0.986 in internal coordinate); no-bottleneck still uses slots; extra Transformer capacity does not help.

But v4 does NOT prove that WESS changes **natural masked-token prediction** in a mixed BabyLM corpus under MLM, nor that it improves Entity-class ability. The next step is neither a full 100M run nor more micro-world patching. It is a **short-range DeBERTa-v2 + WESS transfer experiment** on real data and budget, comparing effective addressing vs destroyed addressing, and confirming that slot state can causally change natural MLM targets and Entity-type behavior.

Only after a clear signal appears do we expand to a full nine-column candidate.

## Scientific question

Under real MLM training on a mixed BabyLM-like corpus at short range (≤10M-word budget slice), does a DeBERTa-v2 encoder augmented with a WESS entity-slot module:

1. reduce masked-token loss specifically on entity-state-dependent target tokens, beyond a matched no-slot baseline;
2. show that slot state **causally** changes the natural masked-target prediction (slot-swap / write-ablation change the predicted token in the expected direction);
3. lose both effects when addressing is destroyed (eventwise-random / wrong-entity routing), proving the effect comes from entity-indexed storage and not from added parameters;
4. begin to move Entity-class probing accuracy.

If yes, expand immediately to a full 100M nine-column candidate. If no, the mechanism has a transfer gap and we must fix representation/routing on real text before scaling.

## Design

### Shared backbone

- DeBERTa-v2, small but real: 8×384 or the S1 12×384 shape at short range. Use the protected trainer's tokenizer (baseline16k) and MLM collation from `babylm_masked_train_leadershape.py`.
- Short budget: a fixed slice, e.g. 5M–10M word exposure, matched across all arms. This is a transfer probe, not a SOTA run; keep it small and fast.

### Data

Mixed corpus = official BabyLM text (real natural language) + a modest fraction (e.g. 10–20%) of **entity-state episodes rendered as natural sentences** with gold entity/state span annotations. All words count against the budget. The episodes must contain multi-entity, overwrite, and interference structure so that some masked targets are only predictable from correct entity→state binding.

Crucially, include a held-out set of **binding-critical masked targets**: natural sentences where the masked token (a location/attribute/state word) depends on which entity performed which action. This is the natural-language analogue of the micro-world query.

### WESS integration into MLM

- Encoder produces hidden states as usual.
- WESS reads gold entity/state spans from the episode annotations (Track B accepts gold spans for this first transfer test, per wess v4 address challenge results decision), maintains GRU entity slots, and produces a slot-informed representation.
- At binding-critical masked positions, the MLM head receives a combination of the standard encoder hidden state and the queried entity's slot readout (no strict bottleneck — wess v4 address challenge results showed slots are used even with full-text access).
- Standard MLM loss on all masked tokens; the slot pathway adds information at entity-referring positions.

### Arms (matched data, budget, init, tokenizer, steps)

1. **baseline_mlm**: DeBERTa-v2 MLM, no slots. Trains on the same mixed corpus.
2. **wess_gold**: DeBERTa-v2 + WESS with correct entity→slot routing.
3. **wess_eventwise_random**: same module, each event writes to a random slot (destroyed addressing).
4. **wess_wrong_entity**: writes to a different entity's slot (destroyed addressing).
5. (optional) **wess_no_episodes**: WESS module present but episodes replaced by equal-word plain text, to check the module isn't just extra capacity.

### Primary measurements (short-range)

1. **Masked-target loss / accuracy on binding-critical held-out targets**: wess_gold should be lower-loss / higher-accuracy than baseline_mlm and than destroyed-addressing arms.
2. **Overall MLM loss on natural text**: must not degrade for wess_gold (the module should not hurt general language modeling).
3. **Causal slot interventions on natural targets**: for held-out binding-critical sentences, swap the queried entity's slot with another entity's slot before the MLM readout; the predicted masked token should shift toward the other entity's state. Measure the shift rate. Destroyed-addressing arms should show no coherent shift.
4. **Write-ablation on natural targets**: remove the last state-update write for the queried entity; the predicted token should revert to the previous state.
5. **Entity-class probe**: run a small slice of the official Entity Tracking eval (or a faithful local proxy) on each arm's checkpoint to see whether wess_gold begins to move Entity accuracy relative to baseline_mlm.

### Decision rule

- **Clear positive**: wess_gold reduces binding-critical held-out loss and raises binding-critical accuracy vs baseline_mlm AND destroyed-addressing arms, with coherent causal slot shift on natural targets, and no degradation of general MLM loss, and non-negative Entity-probe movement. → expand to full 100M nine-column candidate immediately.
- **Ambiguous**: general MLM fine but binding-critical signal weak or not addressing-specific. → fix span/readout integration on real text before scaling.
- **Negative**: no binding-critical advantage or addressing-destruction doesn't remove the effect. → the effect is capacity/co-occurrence, not entity binding; return to representation design.

## Efficiency and scoreboard discipline

- Keep this short-range: small model, ≤10M matched slice, few arms. The recorded compute configuration has two H100s for matched-arm comparisons.
- The 100M candidate is not part of the short-range test and is gated on a clear positive.
- Any checkpoint that is later promoted to a full candidate must be scored into `data/babylm_scoreboard.json` (or successor) with all nine official columns.

## Proposed implementation

The proposed short-range DeBERTa+WESS MLM transfer trainer adapts `babylm_masked_train_leadershape.py` for the encoder/tokenizer/collation and adds the WESS slot module with configurable routing, plus the binding-critical held-out set and the causal-intervention-on-natural-targets evaluation. The matched arms and primary measurements at the short budget test whether the validated micro-world mechanism transfers to BabyLM. This contract is a design, not a report of completed training.
