# recgpt differential route update — RecGPT differential route update

## What the completed full coordinate changes

The official-corpus public RecGPT run is now a completed 9-column result, not an unfinished run:

- File: `data/recgpt_official_full_overall.json`
- Columns: BLiMP 55.13, Supplement 44.23, EWoK 50.49, Entity 16.38, COMPS 50.70, SuperGLUE 58.58, GlobalPIQA 32.725, Reading 0.215, AoA 0.0.
- Overall: 34.2723.
- Gap to protected DeBERTa: −6.2546 Overall.
- Gap to visible leader: −7.5288 Overall.

The full public RecGPT learning recipe ran through 10 epochs on legal official text with word-faithful checkpoint trajectory and compatible full evaluation. The low score is therefore a real learning-result difference between public RecGPT's original training experience and legal official-corpus RecGPT, not an incomplete artifact.

## Behavioral differential

Under the same local evaluation harness, public RecGPT is close to its model-card phenotype on the main zero-shot columns:

- Public RecGPT local: BLiMP 73.20, Supplement 63.24, EWoK 52.82, Entity 16.73, COMPS 55.47, GlobalPIQA 38.71, Reading 2.03 with space-prefix Reading repair.
- Official-corpus RecGPT: BLiMP 55.13, Supplement 44.23, EWoK 50.49, Entity 16.38, COMPS 50.70, GlobalPIQA 32.725, Reading 0.215.

The same architecture/optimizer/NextLat family on official data loses especially BLiMP (−18.07), Supplement (−19.01), COMPS (−4.77), GlobalPIQA (−5.99), and Reading (−1.82 relative to space-fixed local public reference), while Entity stays weak in both cases. The public RecGPT phenotype is therefore not “recursive causal model solves BabyLM”; it is “recursive causal model plus a very different experience distribution produces strong grammar/supplement/COMPS/GlobalPIQA, but still weak Entity.”

## Source-grounded clues about the missing experience distribution

Grounded source facts:

1. The public RecGPT model card says it used a custom 10M-word English corpus and a 32,768-token BPE tokenizer.
2. The public training-source README names `babylm-2024-baby-cosmo-fine-100m-train.jsonl` as the 100M JSONL data to tokenize and train, and names `bblm10M-bpe` / `bblm100M-bpe` tokenizers.
3. The public dataloader comment says row-group shuffling is needed because “the way we sample climbmix leads to our raw data being clustered.”
4. The public model card links a separate dataset-construction repository, but the current retrieval did not acquire that repository. The details of that construction should not be invented.

Together these point to Baby-COSMO-Fine / ClimbMix-like sampling or filtering as the likely missing load-bearing factor, not merely RecGPT architecture.

## Tokenizer differential result

Script: `scripts/recgpt_tokenizer_diff.py`
Output: `data/recgpt_tokenizer_diff.json`

Tokenizer comparison between public RecGPT tokenizer and the official-corpus RecGPT tokenizer on BabyLM probe words:

| word set | n | public leading-space single-token frac | official leading-space single-token frac | mean public-minus-official leading-space length |
|---|---:|---:|---:|---:|
| Reading targets | 1726 | 0.8528 | 0.8598 | +0.0104 |
| AoA words | 504 | 0.9425 | 0.9702 | +0.0337 |
| union | 1144 | 0.7946 | 0.8173 | +0.0306 |

This weakens a simple BPE-coverage explanation for the official-corpus RecGPT collapse: the official-corpus tokenizer is slightly *more* single-token on Reading/AoA word lists. Tokenization still matters for Reading evaluator compatibility because no-space vs leading-space token IDs differ strongly for byte-level BPE, but the public-vs-official RecGPT capability gap is not explained by worse official-tokenizer lexical coverage on these probes.

## Best current scientific interpretation

The missing factor is likely the distribution of training experiences: source mixture, document granularity, lexical/register diversity, web-like sentence quality, and the degree to which the corpus teaches sentence-level plausibility, entailment-like contrasts, and broad commonsense associations. The official BabyLM corpus is conversational/developmental and preserves protected DeBERTa Reading/Supplement strengths, but it does not let the RecGPT recipe develop the public high-BLiMP/high-Supplement/high-COMPS phenotype.

Entity remains the central unsolved column. Public RecGPT also has weak Entity, so copying its experience distribution would not by itself solve the protected DeBERTa gap to the visible leader. The useful lesson is narrower: a web/filtered custom experience distribution can strongly move grammar/supplement/COMPS/GlobalPIQA under a causal recursive learner, but a separate mechanism or data structure is needed for Entity/EWoK.

## Route implication for the SOTA goal

Do not return to ordinary WWM data-ratio tuning or blind architecture changes. The next high-value route should infer and test load-bearing experience structure against protected DeBERTa gaps:

1. Recover public-data construction details if accessible: `serdardoesml/bblm26-dataset`, Baby-COSMO-Fine, and ClimbMix sampling/filtering. If the repository is inaccessible, infer measurable properties from available public model-card/source clues and from accessible Baby-COSMO-Fine / ClimbMix lineage sources.
2. Build a legal, word-counted experience ledger before training: source identity, license, word count, document segmentation, tokenizer training text, and checkpoint trajectory contract.
3. Test a small set of sharply different legal experience distributions on the protected DeBERTa backbone first, because protected DeBERTa is much closer to SOTA than official-corpus RecGPT:
   - a high-quality web/sentence-plausibility distribution meant to restore BLiMP/Supplement/COMPS/GlobalPIQA without destroying Reading;
   - a relation/entity event distribution meant to lift Entity/EWoK, but only if it preserves Supplement/Reading;
   - a hybrid source schedule that keeps enough official conversational/developmental text to protect Reading.
4. Use 10M or 20M exposure screens with the sharded fast evaluation infrastructure, not a new 100M run, until a route shows real movement on Entity/EWoK/GlobalPIQA/SuperGLUE together with preserved Supplement/Reading.
5. Before any future full-budget challenger, verify the full artifact contract up front: final checkpoint, chck_1M–chck_9M trajectory, sharded AoA path, SuperGLUE task shard path, and aggregation script.

## Immediate next research action

The next step should be a source-and-measurement construction step, not training yet:

- retrieve or inspect dataset-construction sources for Baby-COSMO-Fine / ClimbMix / `bblm26-dataset`;
- measure accessible candidate corpora against official text and past custom screens using features aligned to the observed columns: sentence length and punctuation, lexical Zipf/frequency profile, capitalized entity density, pronoun/dialogue rate, dependency/relation cue density, answer-choice plausibility overlap, and tokenizer compression under baseline16k vs 32k BPE;
- from those measurements choose two legal 10M-word candidate experience distributions for protected DeBERTa 10M–20M screens.

The core target remains the original SOTA goal: beat Overall 41.8011, not just explain RecGPT. The RecGPT differential now tells us that experience distribution is load-bearing, while also warning that a RecGPT-like distribution alone probably will not solve Entity.
