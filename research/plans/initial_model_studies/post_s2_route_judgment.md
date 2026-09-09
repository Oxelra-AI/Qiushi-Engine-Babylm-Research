# earlier analysis — post-S2 route judgment

## Scientific situation

The active goal remains BabyLM Strict-Small Overall SOTA under the official constraints. The current evidence after S1/S2 is:

- Protected 8×480 baseline16k DeBERTa-v2 WWM: complete Overall 40.5269; strong Supplement/Reading; weak Entity/EWoK/GlobalPIQA.
- S1 12×384/intermediate1280, baseline16k, official corpus, flat WWM, 100M: BLiMP 66.84, Supplement 60.31, EWoK 52.02, Entity 20.24, COMPS 52.26, GlobalPIQA 37.605, Reading 7.25. S1 moved GlobalPIQA but did not move Entity/EWoK.
- True S2 word-clock curriculum on S1 base, 100M: BLiMP 64.24, Supplement 59.09, EWoK 51.64, Entity 18.47, COMPS 50.61, GlobalPIQA 38.635, Reading 7.52. S2 further moved GlobalPIQA/Reading but damaged BLiMP, Entity, EWoK, and COMPS. It is not a positive overall component on official-corpus/baseline16k/AdamW.
- Visible leader card: 12×384/intermediate1280 DeBERTa-v2, 40k SentencePiece, LAMB, 64→256 length curriculum, WWM7→Token3, FineWeb original+simplified sentence pairs, Overall 41.80 with large Entity/EWoK/GlobalPIQA gains. Exact paired data remains access-restricted in this environment.

The post-S2 inference is now sharper: architecture shape and true word-clock curriculum are not enough. But it is too early to attribute the remaining gap only to the paired data, because two legal model-side factors from the leader recipe remain untested in the leader-shape setting: 40k tokenizer under 12×384 and LAMB/cosine high-LR optimization. The earlier official40k result was on 8×480 with a large embedding increase and severe Reading loss; it does not settle 40k under S1/S2 geometry. LAMB has not been tested at all.

## Route comparison

### 1. Re-score the public leader checkpoint locally

The staged local leader folder currently contains config, tokenizer, spm.model, and README, but not model.safetensors. `hf_info.json` shows the model repo has a public `model.safetensors` LFS object of ~69 MB. A direct local re-score is the cheapest way to determine whether the public model's Entity/EWoK/GlobalPIQA numbers reproduce under our evaluator path and local EWoK data.

Why it matters:

- If the public checkpoint re-scores near the card, the target gap is real and our evaluator is aligned enough for column-by-column reasoning.
- If it does not, the route target and scoring comparability need repair before new expensive training.

Immediate execution: download/load `go76dof/wwm_curriculum_simplification_40k` directly or stage model.safetensors into `data/leader_package_revision_124/model_side/local/`, then run the same direct-checkpoint available-coordinate and full-EWoK scripts used for S1/S2. Do not use the gated training data.

### 2. Complete legal model-side decomposition before data reconstruction

The clean next training work is not more S1/S2 curriculum variants. It is to test the two unrun leader-side factors on legal official data:

- **S3 tokenizer arm:** S1 12×384/intermediate1280 with legal official-corpus 40k tokenizer, no new data. Start with a 10M screen and tokenizer-aware Reading interpretation; if any Entity/EWoK movement appears without catastrophic Reading collapse, scale. Because earlier 40k damaged Reading under 8×480, this arm should record word-level scoring details and embedding/non-embedding parameters.
- **LAMB optimizer arm:** S1 12×384/intermediate1280, baseline16k, official corpus, flat WWM or the leader length/mask curriculum as a second cell, LAMB/cosine with a small LR stability sweep around the leader card regime. This tests whether optimization, not data, induced the relational cluster.
- If both single factors show no relational movement, one co-tuned S3+LAMB+leader-curriculum arm becomes informative. If it still fails, legal rewrite-pair data becomes the strongest remaining explanation.

### 3. Data reconstruction route, but only with decisive controls

If model-side factors fail, legal rewrite-pair reconstruction becomes high value. The mechanism should be framed as two adjacent views of the same proposition, not generic simplification. The important comparison is aligned adjacent pairs versus the same pair members shuffled apart, plus repetition and simplified-only controls. A positive result must show adjacency/semantic alignment, not simply easier text, duplicated exposure, or topical batching.

Potential low-cost data object: official-corpus sentences transformed into simpler or more explicit variants under strict word accounting. Before use, current BabyLM 2026 rules on generated/rewritten text and tokenizer-training text must be rechecked.

### 4. Hybrid objective route

The low-ratio causal+masked hybrid remains scientifically promising for GlobalPIQA/EWoK because it changes information flow and left-context supervision. It is not the most direct answer to Entity, since RecGPT-like causal routes can be strong on BLiMP/GlobalPIQA but weak on Entity. If pursued in parallel, it must first pass a no-future-leak test for DeBERTa relative attention and compare low-ratio CLM against a dense non-causal auxiliary with matched supervised-token count.

## Next experiment

The first proposed measurement is local re-scoring of the public leader checkpoint if weights can be downloaded without gated data. The subsequent model-side comparison is: either S3 40k under S1 geometry or LAMB under S1 geometry. The order should favor whichever can be cleanly implemented fastest after the leader re-score, but neither should be skipped when interpreting whether the data condition is load-bearing.
