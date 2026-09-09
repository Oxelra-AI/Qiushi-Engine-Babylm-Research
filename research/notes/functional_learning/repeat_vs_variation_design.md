# mechanism macro convergence: Repetition vs. Variation Experiment Design

## Scientific question

topology phase1 result established that **correct relational content** determines h1/h3 coordinate transport:
adversarial label inversion selectively anti-orients h1 while h3 (connected by correct labels)
remains correctly oriented. Exposure alone cannot explain this because h1 exposure is preserved
in the adversarial condition.

mechanism macro convergence asks the next question: **why might varied experience be more efficient than repeated
experience for forming reusable coordinates?** This connects the coordinate-transport mechanism to the
BabyLM finding that VIEW (varied rewrites) reproducibly helps Entity tracking (+3.21) and
Reading (+0.37) while REPEAT helps 0-operation recall.

## Scientific constraint

The experiment must distinguish:
1. **Variation that breaks a plausible shortcut** from
2. **Equally diverse variation that leaves the shortcut intact**

under the same semantic support and training budget. A selective benefit would explain
**which** variation buys useful information.

Exact repetition must be preserved as a competing explanation when optimization is incomplete
(the DETACH underfit at e60 makes this concrete).

## Three conditions

### REPEAT (baseline)
- Same 192 comparison events every epoch
- The learner sees the same (name, verb, object, voice) tuples repeatedly
- A surface shortcut exists: the model can memorize specific token patterns
  rather than abstract the nonce verb meaning

### VARY_CONTEXT (breaks shortcut)
- Fresh 192 comparison events each epoch with **different name pairs, objects, and voices**
- Same verbs, relation structure, labels (identical comparison graph)
- Forces abstraction: the model sees "daxed" with many different name pairs
  and must learn the abstract verb meaning across contexts
- Pool: 32 extra names × 24 extra objects × 2 voices = ~47K configs per relation pair

### VARY_NOISE (preserves shortcut)
- Same 192 comparison events as REPEAT, but each epoch adds **different filler tokens**
  (noise prefix + suffix) to each event string
- The diagnostic (name, verb, object) pattern is preserved
- Surface diversity exists (unique token sequences each epoch) but the shortcut remains

### Controls
- State training is **identical** across all three conditions: same 576 state queries
  (512 s_give/s_receive + 32 h0_dax anchors + 32 h2_norp anchors)
- Equality pretraining is **identical**: same 300 char-pair epochs, same seed
- Model architecture (shared_trunk PosAlign) is **identical**
- Total comparison gradient steps per epoch is **identical**: 192

### Evaluation
- **Held-out templates**: the standard eval set uses 16 names (Tara, Ben, Luca, Vera, etc.)
  that never appear in ANY training condition
- **Per-relation accuracy**: h0_dax (direct anchor), h1_mep (graph transport), 
  h2_norp (direct anchor), h3_ziv (graph transport)
- **Bridge sign pairs**: +1/-1 for sign-reversal analysis

## Predictions

### If variation helps by breaking surface shortcuts (proposed mechanism):
- VARY_CONTEXT > REPEAT on held-template h1/h3 transport
- VARY_CONTEXT > VARY_NOISE on held-template h1/h3 transport
- VARY_NOISE ≈ REPEAT (irrelevant diversity doesn't help)
- REPEAT ≥ VARY_CONTEXT on direct-anchor (h0/h2) recall (repetition aids memorization)
- Budget sweep: REPEAT may catch up at higher epochs

This maps to relation_learning's BabyLM finding: VIEW helps Entity 3-4ops (graph transport)
while REPEAT helps Entity 0-ops (direct recall).

### If variation helps by generic diversity/regularization:
- VARY_CONTEXT ≈ VARY_NOISE > REPEAT on all metrics
- Any diversity helps equally

### If optimization completeness dominates:
- All three converge at enough epochs
- REPEAT catches up to VARY at higher budgets

## Connection to general principle

topology phase1 result's adversarial result shows: finite experience is efficient when it supplies **correct**
reusable constraints that enter a **shared coordinate**. mechanism macro convergence extends this: varied experience
may be efficient because it forces the learner to **extract the abstract relation** rather than
memorize surface patterns. The shortcut-specific design distinguishes this from a generic
diversity/regularization effect.

If confirmed, the combined mechanism is:
1. Finite evidence forms a shared coordinate (equality pretraining)
2. Sparse absolute labels orient the coordinate (h0/h2 state training)
3. Correct relational constraints transport the orientation (comparison training)
4. **Varied rendering forces abstraction of the constraint**, making transport robust to
   novel contexts (this experiment)
5. Wrong constraints can actively install wrong competence (topology phase1 result adversarial)

This connects to relation_learning's natural-language finding: VIEW rewrites vary the rendering
while preserving the semantic content, helping entity tracking (requires graph transport)
but not BLiMP (form-sensitive, no transport needed).

## Run plan

- 3 conditions × 2 bridge signs × 1 seed (30000) × 60 epochs = 6 cells
- All were in progress via `run_all.py` at the time of the note
- Output: `experiments/archive/functional_learning/data/repeat_vs_variation`
- Analysis: `combined_analysis.md` and `combined_analysis.json`

## Scripts

- Experiment: `experiments/archive/functional_learning/scripts/repeat_vs_variation.py`
- Runner: `experiments/archive/functional_learning/scripts/run_all.py`
