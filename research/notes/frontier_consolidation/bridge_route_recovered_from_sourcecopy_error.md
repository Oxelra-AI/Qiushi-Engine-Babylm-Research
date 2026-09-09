# bridge route recovered from sourcecopy error — source-attested bridge route recovered from the `source_copy_degree` error

## Why mechanism evidence synthesis's central closure was unsupported

The mechanism evidence synthesis improved source-attested bridge prototype measured `source_copy_degree`, defined essentially as whether the candidate's content words occur anywhere in the source. Under the source-attested-lemma instruction this value is expected to be near one and does **not** distinguish:

- contiguous extraction vs clause reordering;
- source-order deletion vs predicate-argument recoding;
- copied content lemmas with new function-word structure;
- proposition-preserving compression vs proposition-damaging deletion.

Therefore the mechanism evidence synthesis conclusions that the route inherently collapses into extraction, that novel vocabulary is necessary, or that DeBERTa's compact effect should be attributed to relative attention were too strong. GPT2 and RoBERTa comparisons also changed multiple factors beyond relative position.

## Repaired evidence produced in bridge route recovered from sourcecopy error

### 1. Automatic structural analysis

Script: `experiments/archive/frontier_consolidation/scripts/bridge_structural_transformation.py`  
Output: `experiments/archive/frontier_consolidation/data/bridge_structural_transformation/structural_transformation_analysis.json`

The analyzer measured source-order preservation, longest contiguous copied source run, novel word/function-word edits, and edit class on all 103 retained candidates. Main result:

| class | n | reading |
|---|---:|---|
| verbatim source substring | 14 | exact contiguous source window |
| deletion/reorder | 34 | source order preserved with long copied run |
| light restructure | 47 | small but real reordering/function-word/inflection edits |
| substantive restructure | 8 | stronger clause/predicate/function recoding |

Aggregate:

- extraction-like = 48/103 = 46.6%
- transformation-like by structural heuristic = 55/103 = 53.4%
- mean longest source-contiguous run fraction = 0.583
- mean source-content order agreement = 0.962
- mean novel word fraction = 0.0475

This directly falsifies the claim that source-copy overlap alone closes the route: more than half of retained outputs show some structural transformation. But it also shows the pool is not a clean transformation treatment: source order remains very high and exact/deletion-like cases are numerous.

### 2. Blinded independent_review scientist reading

Packet: `research/documents/frontier_consolidation/data/bridge_blinded_reading/bridge_blinded_reader_packet.md`  
independent_review integration: `data/external/independent_review01_verifier1_integration.md`

independent_review independently read the 103 cases without using `source_copy_degree` as evidence. Its integrated finding:

- the 103 candidates are a heterogeneous near-even mixture, not simply mostly extracts;
- a meaningful transformation core exists (e.g. evidential frame -> active predicate, adjunct movement, passive-to-active causation, finite/nonfinite conversion, connector/function-word restructuring);
- many cases are exact/near-exact extracts or source-order deletion; several nominal transformations damage proposition, qualifier, scope, modality, list membership, argument assignment, or grammar;
- only 8/103 are substantive restructurings under the automatic structural label, while most transformation-like cases are light edits.

independent_review bottom line: the concept is feasible, but the 103 retained outputs are not a clean fluent-bridge training surface.

## Correct scientific state after bridge route recovered from sourcecopy error

The source-attested fluent bridge route is **not closed** by the mechanism evidence synthesis source-copy statistic. It remains a potentially valuable same-DeBERTa control because the prototype contains real proposition-preserving structural transformations while keeping source-attested lexical material.

But the present 103 retained outputs should **not** be used directly for full BabyLM pretraining. Direct training on this mixture would entangle at least four treatments:

1. exact/near-exact extraction;
2. source-order deletion compression;
3. genuine grammatical and predicate-argument recoding;
4. damaged partial summaries or malformed compression.

That would not isolate fluent source-attested re-expression from the telegraphic source-only extractive arms already tested in extractive selected readout result.

## What would make a matched same-DeBERTa bridge experiment valuable

A future bridge pool must first pass a stricter construction/selection standard:

- exclude exact source windows and pure source-order deletions;
- require an explicit structural operation such as voice alternation, adjunct movement, finite/nonfinite conversion, relative-clause recoding, argument-frame shift, or connector/function-word restructuring;
- preserve negation, modality, quantifiers, comparison, causal/temporal structure, attribution, geographic/temporal scope, and list members;
- reject malformed source fragments and missing auxiliaries/determiners/copulas/orphaned complements;
- keep exact/deletion, light grammatical compression, and substantive predicate-argument recoding as separate strata;
- verify corpus-level geometry relative to compact and extractive arms: word count, content/function fraction, packed BPE tokens, maskable exposure, sentence length, source span, fragmentation, and longest copied-run fraction.

The immediate discovery task is therefore not pretraining. It is to determine whether a scalable generator/selector can yield enough high-fidelity structural transformations to fill the FineWeb compact block at useful volume. If yes, a matched same-DeBERTa bridge experiment is valuable. If the high-fidelity core remains too small, the bridge route is not ready for expensive training.

## Interaction with coherent88 endpoint work

The already-trained coherent88 private replay should still be read out because the model exists and the cheap7 alpha evaluation is now inexpensive relative to training. But its result is endpoint evidence, not a replacement for the mechanism discovery line.
