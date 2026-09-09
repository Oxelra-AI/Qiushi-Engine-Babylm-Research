# earlier analysis — route after legal model-side decomposition

## Active target

The target remains a real BabyLM Strict-Small Overall SOTA result under official-compatible training and evaluation. Current internally trained models remain below the locally re-scored public leader, so the next work must attack the remaining scientific gap rather than polish any existing artifact.

## Evidence now available

The public leader checkpoint `go76dof/wwm_curriculum_simplification_40k` has been staged and locally re-scored without using its access-restricted training data. Its Entity/EWoK/GlobalPIQA profile is real under our evaluator:

| model / arm | exposure | BLiMP | Supp. | EWoK | Entity | COMPS | GlobalPIQA | Reading | main interpretation |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| public leader local | public ckpt | 67.20 | 56.04 | 56.07 | 28.45 | 53.57 | 39.665 | 5.425 | target profile; access-restricted pair data not used locally |
| protected 8×480 baseline16k WWM | 100M | 66.76 | 59.88 | 52.19 | 22.62 | 52.19 | 35.635 | 7.62 | best internally trained complete 9-column reference, Overall 40.5269 |
| S1 12×384 baseline16k WWM AdamW | 100M | 66.84 | 60.31 | 52.02 | 20.24 | 52.26 | 37.605 | 7.25 | leader shape raises GlobalPIQA but not Entity/EWoK |
| S2 S1 + word-clock length/mask curriculum | 100M | 64.24 | 59.09 | 51.64 | 18.47 | 50.61 | 38.635 | 7.52 | curriculum helps GlobalPIQA but harms the target cluster |
| S3 S1 + legal official40k tokenizer | 10M | 54.34 | 52.50 | 49.55 | 17.77 | 50.02 | 37.165 | 1.17 | 40k shifts PIQA but not Entity/EWoK and damages Reading |
| validated LAMB reference-default S1 | 10M | 51.75 | 51.63 | 49.52 | 16.56 | 50.33 | 36.225 | 6.77 | stable, but early target movement is negative |
| S1 AdamW same 10M comparison | 10M | 53.37 | 52.34 | not run | 17.73 | 50.34 | 35.21 | 8.35 | direct optimizer comparison point |

Important correction to preserve: the lamb s1 10m available coordinate LAMB run did **not** reduce MLM loss faster than S1 AdamW at 10M. It was stable but worse in pretraining loss: LAMB final loss 5.0807 versus S1 AdamW final loss 4.0523 at the same 10M exposure, batch geometry, data, model, tokenizer, and seeds. LAMB improves GlobalPIQA mean only through nonparallel PIQA, while BLiMP, Entity, EWoK, and Reading move away from the public leader profile.

LAMB implementation evidence is now solid for the tested variant: `data/lamb_numeric_comparison.json` shows exact zero parameter delta against the staged torch-optimizer/cybertronai reference for both default `debias=false` and optional `debias=true`. Therefore the negative 10M result is not explained by a broken local optimizer.

## Route judgment

A full 100M LAMB run remains scientifically possible, because the 10M run is only earlier analysis/2442 of a high-LR trajectory and does not sample the later cosine-decay regime. It should not be declared impossible from the 10M result alone.

However, an immediate single full 100M LAMB run is not now the highest-value next experiment. The reasons are concrete:

1. The tested LAMB factor is stable and recognized-reference accurate, but at 10M it is worse than AdamW on MLM loss and on the target Entity/EWoK/BLiMP cluster.
2. S1, S2, S3, and LAMB each move some part of the public leader recipe but none reproduces the leader's key Entity/EWoK gain.
3. The public leader's name and metadata point to simplification-pair data, and the strongest remaining unexplained capability gap is semantic/entity/world-knowledge behavior rather than grammar.
4. The model-side factors have now been tested enough to make data mechanism the most information-rich next route, while a full 100M LAMB run would mostly answer a late-trajectory optimizer question with weak early target support.

The LAMB route should be preserved as a secondary trajectory, not erased. A useful way to preserve it is a medium continuation only if it directly compares with the data route: original-data LAMB to around 30M, or a later 2×2 data/optimizer interaction test if the aligned-pair data shows promise. It should not consume the next main execution slot before a clean data-alignment experiment exists.

## Next main route: legal aligned simplification/paraphrase reconstruction

Scientific hypothesis: the public leader's Entity/EWoK gains may come from seeing semantically equivalent or simplified renderings of the same content within a tiny word budget. This can raise surface-invariant entity and world-knowledge representations in a way ordinary official-corpus MLM, shape changes, tokenizer changes, curricula, and LAMB on raw data do not.

The first experiment must separate three effects:

- seeing rewritten/simplified text at all;
- seeing original and rewritten text in the same window;
- seeing the **correct** semantic correspondence between the two surfaces.

### Fixed base for the first data experiment

Use the S1 base to avoid mixing in known harmful or ambiguous factors:

- DeBERTa-v2 12 layers, hidden 384, 12 heads, intermediate 1280;
- baseline16k tokenizer;
- AdamW default from S1;
- flat whole-word masking;
- sequence length 256;
- official-compatible exact whitespace word exposure;
- no 40k tokenizer, no LAMB, no length/mask curriculum, no access-restricted FineWeb pair data.

### Data source ladder

Start with a corpus construction that is legal and auditable before scaling:

1. **Official-corpus-derived deterministic rewrites**: derive pairs `(x_i, y_i)` from BabyLM official training text with rule-based transformations that preserve named entities, quantities, polarity, core predicates, and argument roles. This is the cleanest source for attribution because all content originates in the official corpus.
2. **Open simplification/paraphrase resources** only after source files, license, and BabyLM rule compatibility are confirmed from primary sources. Candidate resources include ASSET, WikiAuto/ASSET/Turk, Newsela-derived material, Wikipedia/Simple-Wikipedia pairs, and semantic/entity-preservation evaluation sources. These should be read and confirmed before use.
3. **Generated paraphrases** only if the official rule source confirms generated/custom text is allowed under the word budget and the generation process can be recorded. If used, keep it as a separate source condition; do not mix teacher-model knowledge with the pure semantic-alignment test.

### Four-arm first experiment

Construct one shared pair pool with exact word counts and stable hashes. For every pair `(x_i, y_i)`, build arms with matched text multisets:

- **Aligned**: `x_i [PAIR_SEP] y_i` in one example/window.
- **Shuffled-pair**: `x_i [PAIR_SEP] y_{π(i)}` where `π` has no fixed points and is permuted within buckets for source, length, entity count, transformation type, and rough topic if available.
- **Unpaired-mixture**: the same `x_i` and `y_i` texts appear as independent examples, not as paired windows.
- **Source-only / rewrite-only matched arms**: source-only and rewrite-only streams resampled to the same word exposure, so text style and content changes can be separated from alignment.

The aligned-vs-shuffled comparison isolates the correct semantic correspondence because the two arms contain the same source and rewrite text multisets and nearly the same local format. The unpaired/source/rewrite arms determine whether any gain comes from rewritten text distribution rather than pair alignment.

### Word accounting and manifests

Every arm must have an explicit manifest before training:

```json
{
  "pair_id": "...",
  "arm": "aligned|shuffled|unpaired|source_only|rewrite_only",
  "source_file": "...",
  "source_span": "...",
  "rewrite_rule_or_source": "...",
  "source_text_sha256": "...",
  "rewrite_text_sha256": "...",
  "source_words": 0,
  "rewrite_words": 0,
  "selected_words": 0,
  "cumulative_selected_words": 0
}
```

Special separator tokens and padding do not count as words. Do not truncate in the middle of a sentence to hit the budget; construct the manifest so the selected examples sum exactly or deterministically to the chosen word budget. For pair windows, preferentially select pairs fitting within 256 tokens; otherwise report source-side and rewrite-side truncation separately.

### First scale

The first useful screen should be 10M words per arm with at least one seed and preferably paired seeds if runtime permits. Checkpoints at 2M/5M/10M would be valuable if the trainer supports them cleanly; otherwise 10M direct evaluation is acceptable for the first pass.

Measure:

- BLiMP, Supplement, Entity, COMPS, GlobalPIQA parallel/nonparallel, Reading;
- full local EWoK;
- pretraining loss, masked tokens per word, truncation, source/rewrite word contributions;
- entity-focused readouts if simple to compute: Entity split scores, entity-token logit margins, or pair retrieval on held-out legal pairs.

Signal patterns that would justify scaling:

- Aligned > shuffled on Entity and EWoK with no grammar/Reading collapse large enough to erase the gain;
- Aligned > unpaired-mixture, showing same-window correct correspondence matters;
- gains persist or grow from 5M to 10M if intermediate checkpoints exist;
- entity/world-knowledge improvement is not merely lower MLM loss or shorter/easier text.

If aligned and shuffled are similar but both beat source-only, the benefit is rewrite-text distribution or same-window dual-style exposure rather than correct pair correspondence. If no arm improves Entity/EWoK, the legal rewrite source or transformation family is not capturing the leader's mechanism and should be rebuilt rather than scaled blindly.

## Secondary LAMB path

The LAMB path is retained as follows:

- do not launch a full 100M LAMB run as the next main experiment;
- if data construction takes substantial time or if aligned-pair results interact with optimization, run a medium original-data LAMB continuation around 30M to test whether the loss and target columns begin to catch up after the peak-LR period;
- if aligned-pair data is promising, later test AdamW vs LAMB on the same reconstructed data to see whether optimization matters only under the new data distribution.

## Immediate next work

The proposed first construction is the data materialization layer before any training:

1. read the official 2026 rule source to confirm treatment of custom/generated text and word accounting;
2. inspect the acquired simplification resources and decide whether the first pair pool is official-corpus-derived, open-resource-derived, or both as separate source conditions;
3. write `scripts/build_aligned_rewrite_pair_corpus.py` to produce manifests and JSONL examples for the aligned, shuffled, unpaired, source-only, and rewrite-only arms;
4. compute source/rewrite word counts, hash multiset equality, tokenization/truncation summaries, and sample pairs for human-readable review;
5. only after manifests are clean, train the first 10M matched arms on the fixed S1 base.
