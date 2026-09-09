# route synthesis for independent_review Route Synthesis — grounding for independent scientific reading

## Fixed target
Real, fully-legal BabyLM Strict-Small Overall SOTA (visible target 41.8; need to beat it). Best fully-legal complete endpoint: spatial repair route status same-pool-tokenizer compact-view-reinvest, **Overall 41.2578** (−0.5422 below 41.8). Non-submittable old-tokenizer endpoint 42.0331 is mechanism evidence only (its tokenizer was fit on 100M text, violating ≤10M budget).

## Protected scientific core (do not reopen)
Compact-view reinvestment: under the SAME legal tokenizer, reinvest − clean is −0.83 at 20M but **+1.29 at 70M and +1.35 at 80M** mean7 (BLiMP +1.53, Supplement +2.06, EWoK +2.03, Entity +2.33, COMPS +0.65, GPIQA −0.03, Reading +0.85 at 80M). The corpus mechanism is strongly positive at mature exposure and late-emerging. It is not the bottleneck.

## The recoverable quantity
Old-tokenizer 42.033 vs legal-spatial repair route status 41.258 = **−0.775 Overall, same corpus + recipe + seeds; ONLY the tokenizer differs.**
- Largest losses: EWoK −3.14, Supplement −2.11, BLiMP −1.00, SuperGLUE −0.76. GPIQA +0.44.
- Deficit is STRUCTURED and reproduces across two independent legal 16k tokenizers (Supplement UID Pearson 0.82, sign agreement 1.0; BLiMP 0.65). So it is a property of the legal 10M-fit budget, not tokenizer noise.
- Largest UID losses: BLiMP `principle_A_reconstruction` −20.89, `wh_questions_object_gap` −17.34, `tough_vs_raising` −10.97, `anaphor_gender_agreement` −10.71; Supplement `qa_congruence_easy` −12.5; EWoK `physical-dynamics` −19.5, `social-properties` −13.75, `material-dynamics` −8.83.

## Scalar explanations already falsified (do NOT re-test)
- Tokenization fragmentation: legal/old token ratios ≈ 0.9915–0.9922 (near 1.0); no new seq256 truncation.
- Token-string identity: only ~15.7% of token strings changed.
- Low support / rare tokens: ruled out (support error conditioned probe support probe; earlier analysis high-support errors).
- Byte coverage: byte-alphabet repair (full ByteLevel alphabet) made it WORSE (40.70 vs 41.26).
- Vocabulary size / support floor: minfreq50 (vocab 19,609) closed at 80M (−0.168).

## Closed routes (do not reopen without new mechanism)
word-mean MLM credit; minfreq50 tokenizer; byte-alphabet; spatial lengthening; relation-only masking; late stopping / checkpoint averaging; naive source-view agreement; earlier analysis innovation-biased WWM probability; exact-swap (paused, too small ~0.174% perturbation); depth alone (41.03); EWoK crossed-interaction objective; natural-pair self-contrast.

## Pending Representation Comparison
SGCR support-sharing training/evaluation remains pending. Its result is required before another legal40k, SGCR or depth comparison is justified.

## Two competing mechanism hypotheses for the −0.775 gap

### H-A: Tokenizer-fit distribution mismatch (data/representation mechanism)
The old tokenizer was BPE-fit on 100M of DISTINCT Strict text. The legal tokenizer was fit on the 10M pool (which contains compact views) that is then repeated 10×. Its merge table encodes the statistics of a narrow, repeated corpus. In-corpus segmentation is fine (token ratio ~1.0), but the segmentation of UNSEEN evaluation text (BLiMP/EWoK drawn from held-out distributions) is systematically worse / less compositional. The deficit concentrates in long-range syntactic binding (principle-A, wh-object-gap) and relational reasoning (physical-dynamics) — exactly the subtasks whose eval sentences differ most from the training pool. This is a REPRESENTATION mechanism in the data/tokenizer-fit layer, orthogonal to embedding-support-sharing route.

### H-B: Untuned optimization for the legal coordinate (optimization mechanism)
The inherited recipe (AdamW lr=0.001, warmup 0.06, WD 0.01, batch 256, WWM 0.15, fixed seq256) was tuned for the OLD tokenizer and never re-optimized for the legal tokenizer. The legal tokenizer changes token frequencies and gradient geometry; the same lr/warmup may be suboptimal. The leader uses LAMB lr=0.007 (very different point). A bounded lr/optimizer screen could recover part of the gap.

## Scientific Constraints
- Identify a SPECIFIC learning-dynamics mechanism; test with ONE bounded experiment. No generic hyperparameter sweep.
- The compact-view advantage emerges LATE. A 40M ranking is decisive ONLY if prior trajectories show the relevant ordering is stable by 40M; otherwise use the smallest MATURE comparison that genuinely distinguishes the mechanism.
- A 20M→70M→80M treatment trajectory exists; 100 checkpoints exist for the legal run.

## Open question for independent reading
Which hypothesis (H-A tokenizer-fit distribution mismatch, H-B optimization, or a genuinely different third mechanism) is best supported by the evidence, and what is the single cheapest decisive experiment that distinguishes it — given that 40M may be too early for a late-emerging effect, and given the late-emergence signature?
