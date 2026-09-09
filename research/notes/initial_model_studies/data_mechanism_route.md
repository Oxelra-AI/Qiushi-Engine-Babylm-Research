# Short-budget evidence and the data-mechanism route

## Scope of the short-budget evidence

These short-budget experiments did not establish BabyLM Strict-Small Overall superiority. The scientific limitations are specific:

- The only cross-seed-stable positive component we found is WWM, and its gains are small: mean WWM-minus-token BLiMP +2.15, EWoK +2.50, Entity +0.51, with Supplement/COMPS/Reading flat or slightly negative.
- Three components taken directly from the current top recipe each failed to close the SOTA gap under strict paired two-seed 1M tests:
  - short-to-long length schedule: Supplement -3.00, COMPS -0.61, no stable Entity/EWoK/Reading gain (rejected);
  - official-corpus 40k ByteLevel-BPE tokenizer: Reading eye -8.31, Reading SPR -3.56, Supplement -3.20, no Entity gain (rejected);
  - parameter-matched DeBERTa-v2 architecture: mean BLiMP -0.285, EWoK -0.59, Entity +0.075, COMPS -0.20, gains unstable across seeds (not a useful base).
- The absolute level of our WWM base (BLiMP ~56, EWoK ~49-51, Entity ~17.6-18.1, Supplement ~50-51) is far below the top system (BLiMP 67.20, EWoK 56.07, Entity 28.45, Supplement 56.01, Overall 41.80).

Mechanically stacking a fourth champion-recipe component is not the right move. The champion's own strongest reported gains are on EWoK, Entity Tracking, and GLUE, and its distinctive ingredient is **meaning-preserving rewrite/simplification pairs**, not any single architectural or tokenizer trick. The next research must therefore attack the data-semantics mechanism directly.

## Why the data route, grounded in sources

Two primary sources make the data route the highest-value next direction:

1. `edman2024babylmsa` (Are BabyLMs Second Language Learners?, BabyLM 2024 strict-small): across lexical (Wiktionary), grammar, and paraphrase data types under DeBERTa MLM, **paraphrase/contrastive data was by far the most impactful ingredient**, giving the two best models (paraphrase-only and BabyLM+paraphrase mix), with GLUE gains around +8. Lexical data hurt; grammar barely helped. This directly supports rewrite-pair semantics as the mechanism, not order or architecture.

2. `roque2025beyond` (Beyond Repetition, BabyLM 2025): builds a parallel corpus of human paragraphs aligned with LLM-simplified variants and tests repeated-exposure vs. complexity-order vs. interleaved schedules. Reported that adding simplified data improves fine-tuning and zero-shot over a repeated-exposure baseline, with size-dependent ordering effects. This is essentially the pair/simplification-adjacency mechanism at BabyLM scale.

Additional support: `charpentier2025babylm`/Velasco-Roque abstract reports that in complexity-controlled pretraining, **simpler texts benefit linguistic-knowledge tasks while more complex texts favor world-knowledge and entity tracking** — meaning the pair structure (complex + simple aligned) may combine both benefits.

## Rules compliance (from sources)

- `edman2024babylmsa` explicitly states the BabyLM 2024+ challenge allows participants to **construct their own datasets within the track word budget** (10M words for strict-small). This is corroborated by the 2023 findings and the strict-small track description.
- Therefore a custom rewrite-pair corpus is compliant **as long as total training-word exposure is accounted within budget**. Any generated/aligned text counts as training words; the budget accounting we already enforce (`max_word_exposure`, whitespace-word counts, manifests) covers this.
- The top FineWeb simplification-pair train file (`go76dof/Fineweb_simplification_pairs`) is gated (401). We do not use it. We use an accessible, licensed alternative.

## Accessible, verified data source

`GEM/wiki_auto_asset_turk` (dataset sha `ac2b97468b38cb35fcebe327ac8e1cb6b55b6b99`, gated=False):

- `wiki_auto_asset_turk/train-00000-of-00001.parquet` — 484k rows, fields `source` (complex Wikipedia sentence) and `target` (Simple Wikipedia aligned sentence).
- Component licenses: WikiAuto CC BY-NC 3.0, ASSET CC BY-NC 4.0, TURK GPL v3.0 — research use; acceptable for a research submission, to be recorded in the data manifest.
- The train split is WikiAuto-derived aligned complex→simple sentence pairs (verified in the dataset card sample), i.e. exactly the meaning-preserving rewrite structure the mechanism needs, with strong entity overlap between source and target (names, dates, entities largely preserved across the pair).

This gives the material for both a data-swap experiment and, more importantly, a **decisive pair-vs-shuffle mechanism experiment**.

## Route: several mechanism-distinct, compliant, executable data lines

All lines keep the protected base fixed: BERT-WWM, baseline 16k tokenizer, fixed 256 length, `max_position_embeddings=512`, AdamW schedule, exact 1M whitespace-word exposure, paired seeds 42/43, official `mlm` profiling (BLiMP, Supplement, EWoK, Entity, COMPS, Reading). Only the training text/adjacency changes. Each line isolates a different mechanism hypothesis.

### Line A (decisive): pair-adjacent vs. pair-shuffled — isolates *semantic-equivalent adjacency*

Same exact sentence set and same total words in both arms; only whether each (source, target) rewrite pair is kept adjacent in the same training example/context or separated by shuffling target sentences across pairs.

- `pair_adjacent`: build examples that place `source` immediately followed by its aligned `target` (meaning-preserving rewrite adjacency), packed to the words/example budget with pair boundaries preserved.
- `pair_shuffled`: identical multiset of source and target sentences and identical word budget, but targets are permuted so a source is adjacent to an unrelated target (adjacency destroyed, distribution preserved).

This is the cleanest test of the mechanism hypothesis: if adjacency of meaning-equivalent rewrites is what teaches entity consistency and semantic equivalence, `pair_adjacent` should beat `pair_shuffled` on Entity/EWoK/COMPS at equal data and words. If they are equal, the benefit is distributional (vocabulary/style), not adjacency.

### Line B: rewrite-pair corpus vs. official corpus — isolates *data distribution/source*

`wiki_auto_pairs` (pair-adjacent) vs. the existing official-corpus WWM baseline, both at 1M words. This measures the total effect of switching to rewrite-pair data (confounds distribution + adjacency + style), and connects to the champion's data choice. It is not decisive alone, which is why Line A is the priority.

### Line C (optional, later): mixed official + rewrite pairs — isolates *augmentation vs. replacement*

A 50/50 mix of official corpus and pair-adjacent rewrite data, matching `edman2024babylmsa`'s finding that a BabyLM+paraphrase mix was among the best. Run only if A/B show a positive signal.

## Priority and decision rule

1. Build a reusable pair-data path in `babylm_masked_train.py` that reads a local rewrite-pair file and supports `pair_adjacent` / `pair_shuffled` example construction, with exact whitespace-word accounting, file hash, dataset sha, and manifest — without breaking the official-corpus path.
2. Run a 10k WWM smoke on pair-adjacent data (save root/`chck_1M`, load `AutoModelForMaskedLM`, official fast BLiMP+Entity).
3. The primary mechanism test is **Line A**, the pair-adjacent vs. pair-shuffled 1M two-seed experiment, with retained content and exposure matched.
4. In parallel/after, run **Line B** rewrite-pairs vs. official at 1M.
5. Decision: carry the data route forward only if pair-adjacent beats pair-shuffled (Line A) on Entity and at least one of EWoK/COMPS with stable cross-seed sign, and/or rewrite-pairs beat official (Line B) on the same columns without collapsing Reading/Supplement. If adjacency shows no effect, the benefit is distributional and the next factor becomes data mixture/scale, not adjacency.

No 10M/100M scaling, full nine-column evaluation, or submission work until a data line shows a controlled, cross-seed positive 1M signal on the human-like/semantic columns that currently block us.

## Compliance and integrity notes

- Record dataset id, sha, file hashes, license, and exact word budget in every run manifest.
- Keep total training-word exposure within the strict-small budget; rewrite-pair words count as training words.
- Do not use the gated FineWeb file; use only the accessible licensed corpus.
- Preserve exact paired seeds and the WWM+16k+fixed256 base so any gain is attributable to the data-adjacency factor.
