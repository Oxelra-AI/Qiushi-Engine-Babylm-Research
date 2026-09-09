# earlier analysis — Custom Corpus Route: Entity-State Narrative Data

## Evidence synthesis driving the route

### Why mechanism-only approaches failed on official data

related experiments tested every reasonable training mechanism on the official corpus:
- Entity-density selection: Entity −1.93 (structure density conclusion and objective pivot)
- WikiAuto aligned pairs: Entity −0.42, EWoK −1.75 (aligned vs shuffled 10m available comparison)
- AMLM: transient 40M gain, reversed at 100M (related experiments)
- RMEC/MCEC: rejected before launch by corrected probe (new best verification and assessment)
- Structured SynCSE: Entity −0.10 (two seed zero shot summary)
- 40k tokenizer: Entity 22.16, below protected 22.62 (official40k oom and accum repair)
- WESS/R1: learned mechanism doesn't transfer (related experiments)

The official corpus (Gutenberg, Simple Wiki, CHILDES, OpenSubtitles, BNC, Switchboard) is dominated by literary narrative, conversational fragments, and transcribed speech. These domains contain entity references but rarely in the explicit entity-relation-state form that Entity Tracking evaluations test.

### Why the leader succeeds and WikiAuto fails

The leader uses FineWeb simplification pairs — **web/informational text** with entity-property-relation content, simplified to make relations explicit. This works because:
1. Web text naturally contains "X is located in Y," "X has Z," "X moved to Y" patterns
2. Simplification decomposes complex sentences into atomic entity-relation statements
3. Adjacent original+simplified pairs create a learning signal where the model sees the SAME entity-relation in two forms

WikiAuto fails because Wikipedia simplification produces:
- Encyclopedic date/location/attribution facts, not trackable state
- Very long source sentences with relative clauses, not narrative events
- Passive voice and formal register that Entity Tracking doesn't test

### The key differentiator

The leader's data works because it creates **narrative entity-state learning signal** — text where entities DO things, GO places, HAVE properties, and where the simplified version makes these relations EXPLICIT and ADJACENT. This is fundamentally different from either:
- The official corpus (implicit, scattered, complex)
- WikiAuto (encyclopedic, formal, non-narrative)

---

## Design: Entity-State Narrative Corpus (ESNC)

### Core principle

Construct a 10M-word corpus where every training window contains:
1. At least 2 named/specific entities with trackable properties
2. Explicit entity-relation statements (location, possession, attribute, action)
3. Simplified restatements that decompose complex sentences into atomic relations
4. Narrative structure where entity states change over time

### Source selection

Use publicly available, license-compatible text sources. Priority:
1. **FineWeb-Edu** (CC-BY): high-quality educational web text filtered for quality
2. **C4-en** (ODC-BY): diverse web text 
3. **BookCorpus/open alternatives** (narrative fiction)
4. **Simple English Wikipedia** (already simplified, entity-rich)

Filter all sources for:
- Contains ≥2 capitalized entity names per 100-word window
- Contains ≥1 explicit relation cue (location/possession/action verb + entity + filler)
- Does NOT contain formulaic/tabular/code/list content
- Readability level appropriate (Flesch 50-80 or sentence length 8-25 words average)

### Processing pipeline

For each selected passage:
1. **Extract entity-relation triples** deterministically using dependency parsing + entity cues
2. **Generate simplified sentences** from complex sentences using rule-based splitting:
   - Relative clause extraction: "X, who was in Y, did Z" → "X was in Y. X did Z."
   - Coordination splitting: "X went to Y and took Z" → "X went to Y. X took Z."
   - Passive→active: "Z was taken by X" → "X took Z."
   - Appositive extraction: "X, a doctor, lived in Y" → "X is a doctor. X lived in Y."
3. **Pair original + simplified** as adjacent document blocks
4. **Entity-state threading**: within each document block, ensure the same entities appear in both original and simplified with matching properties

### Format
```
[original complex sentence with entity relations]
[simplified: entity-relation statement 1]
[simplified: entity-relation statement 2]
[blank line]
[next original complex sentence...]
[simplified...]
```

### Quality filters
- Discard pairs where entity overlap < 1 shared entity between original and simplified
- Discard if simplified doesn't contain at least 1 explicit relation verb + entity + filler
- Discard if relation is purely temporal/definitional without state content
- Keep pairs where entity-state content is about location, possession, attribute, physical state, or action consequence

### Size budget
- Target: exactly 10,000,000 whitespace words (legal BabyLM Strict-Small)
- Estimated yield from FineWeb-Edu filtering: ~2M qualifying passages per 100M scanned
- From 2M qualifying passages, extract and pair to reach 10M words
- If insufficient from one source, combine with Simple Wikipedia and filtered BookCorpus

---

## Why this differs from WikiAuto (which failed)

| Property | WikiAuto (failed) | ESNC (proposed) |
|---|---|---|
| Source domain | Wikipedia (encyclopedic) | Web/narrative (informational, story-like) |
| Relation types | Dates, locations, awards, demographics | Possession, location, physical state, action |
| Sentence structure | Very long → shorter formal | Complex → atomic entity-relation |
| Entity tracking signal | Weak (entities defined, not tracked) | Strong (entities DO things, CHANGE state) |
| Narrative continuity | None (single-sentence pairs) | Within-block entity threading |
| Match to Entity eval | Poor (eval tests narrative state) | Direct (eval tests same relation types) |

---

## Why this IS innovation, not replication

1. **Targeted for entity-state**, not generic simplification — we filter and construct specifically for trackable entity properties
2. **Rule-based generation**, not model-based — all text is deterministically derived, reproducible, and clearly legal
3. **Entity-threaded blocks**, not random sentence pairs — later words in a block depend on earlier entity-state facts
4. **Combined with our architectural advantage**: DeBERTa-v2 8×480 + baseline16k already achieves Supplement 61.00 (leader: 56.04), so we preserve this while gaining Entity/EWoK

---

## Immediate execution plan

### Phase 1: source selection and filtering (few hours)
- Download FineWeb-Edu sample (e.g., first 50GB shard)
- Filter for entity density and relation quality
- Estimate yield per 100M source words

### Phase 2: processing and corpus construction (~1 day)
- Apply rule-based simplification to filtered passages
- Build entity-threaded blocks
- Quality-filter pairs
- Assemble exactly 10M words with word count manifest

### Phase 3: training (2-3 hours per arm on H100)
- Arm A: DeBERTa-v2 8×480, baseline16k, ESNC corpus, flat WWM, 100M exposure, seed 42
- Arm B: same as Arm A but pair-shuffled (simplifications placed after random entities, not their own)
- Both compared against protected official-corpus reference

### Phase 4: evaluation
- chck_60M/70M/80M/90M/100M on 7 zero-shot columns
- SuperGLUE at best endpoint
- AoA over full trajectory
- Target: Entity ≥ 26, EWoK ≥ 54 at best single endpoint while Reading ≥ 4.5

---

## Fallback: if FineWeb-Edu yield is insufficient

If filtering FineWeb-Edu doesn't produce 10M words of quality entity-state content:
1. Combine with Simple Wikipedia entity-rich passages (avoid pure encyclopedic)
2. Add procedural entity-state text from rule-based templates (the second proposed mechanism)
3. Use official corpus narrative sections (Gutenberg stories, CHILDES narratives) as a quality anchor

---

## Evidence files
- state counterfactual ranking smoke results v2 showed that explicit repeated entity+cue+filler slots DO activate state-specific ranking (R_extra = +5.24, albeit on only 11 cases)
- This means the DeBERTa architecture CAN use entity-state when the training text provides enough explicit examples
- The path to SOTA is: give it a corpus FULL of such examples

## Compliance
- Total corpus: exactly 10M whitespace words
- All text publicly available and license-compatible
- All processing is deterministic/rule-based (no external LM generation)
- Word counting follows exact BabyLM specification
- Dataset composition recorded in datasheet for submission
