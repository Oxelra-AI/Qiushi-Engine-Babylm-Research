# earlier analysis — Route rebuild after SCMLM-R final-answer failure

## Immediate scientific update

The earlier analysis four-arm response experiment and earlier analysis capacity probe sharply narrow the current state-counterfactual route.

Key files:

- Four-arm evaluation: `data/scmlm_r_fourarm_eval.json`
- Capacity probe: `data/capacity_probe.json`
- R1 learning-response result: `notes/r1_learning_response_result.md`
- SCMLM-R route plan: `plans/scmlm_r_refined_first_experiment.md`

The repeated quantitative pattern is the same mirror failure:

- In held-out counterfactual pairs the model has a fixed candidate-answer preference, e.g. mean `m_A` near `-0.5` and mean `m_B` near `+0.5` in the opposite context.
- Pair-summed interaction margin stays numerically zero.
- Both-contexts-correct fraction stays 0 for WWM, random-negative ranking, and SCMLM-R.
- Answer-span MLM produces only a tiny held-out signal at earlier analysis (both-contexts-correct 0.04), likely answer familiarity rather than state computation.
- A strong small-set overfit attempt with `tau=0.5`, `gamma=1.0`, and `lambda_inter=2.0` over 32 fixed pairs for 60 epochs still could not make both contexts correct on the trained pairs.

This is stronger than the r1 learning response result WWM result: not only does ordinary WWM fail to discover the state variable, but a final-query answer-ranking loss through the ordinary MLM head also does not break the answer-prior mirror in this implementation.

## Correct interpretation

Do not overstate the result as a proof that transformers or masked LMs cannot represent state.

The tested SCMLM-R implementation did **not** include the full proposed design:

- no prefix/intermediate-state supervision;
- no state-irrelevant and noncausal swap families in training;
- no harder multi-candidate pool beyond limited equal-length locations;
- short 200-step continuations for the multi-arm run;
- the capacity probe used final-answer ranking only.

So the result does not close every possible state-counterfactual training design. It closes the current **final-answer MLM-head ranking** branch as a practical route to the BabyLM SOTA goal.

The deeper lesson is that the state signal is too far from the ordinary masked answer score. When the two passages are same-word-bag counterfactuals, the query-position MLM distribution remains dominated by candidate-token prior and local form. Final-answer ranking has weak leverage because the two contexts are represented almost identically at the scoring head.

If state-counterfactual work is revisited, the next meaningful version must train state along the trajectory: prefix queries, operation-local consistency, noncausal swap invariance, and perhaps a state-directed RTD or object-state representation. It should not be another final-answer-only ranking run.

## Relation to the SOTA goal

The protected internal best remains:

- baseline16k DeBERTa-v2 8×480 WWM, official corpus, 100M exposure;
- complete Overall 40.5269;
- strengths: Supplement 59.88, Reading 7.62;
- weaknesses: Entity 22.62, EWoK 52.19, GlobalPIQA 35.635, SuperGLUE 68.02.

The visible leader remains:

- `go76dof/wwm_curriculum_simplification_40k`, Overall 41.80;
- strengths: Entity 28.45, EWoK 56.07, GlobalPIQA 39.665, SuperGLUE 69.79;
- weaker than protected on Supplement and Reading.

RecGPT-10M is also important:

- Overall 41.53, close to leader;
- very strong BLiMP 73.11, COMPS 55.43, GlobalPIQA 40.68, Reading 6.92;
- weak Entity 16.59, EWoK 52.62.

This matters because there are at least two high-scoring phenotypes:

1. leader phenotype: masked DeBERTa + custom simplification-pair-like data, high Entity/EWoK/GlobalPIQA;
2. RecGPT phenotype: recursive causal + custom corpus/optimizer/tokenizer, high BLiMP/Supplement/COMPS/GlobalPIQA/Reading but weak Entity.

Repeated experiments attempted to construct the missing Entity/EWoK state variable and repeatedly failed. The SOTA path should no longer be trapped in synthetic entity-state micro-worlds. The next route must either make a substantially stronger representational intervention or broaden to the full leaderboard phenotype space.

## Route comparison now

### Route A — trajectory-supervised state representation

Scientific object: make state information available before the final answer by training intermediate prefix states and operation-local consistency, possibly with a state-directed RTD auxiliary or an exportable object-state channel.

Why it is still scientifically meaningful:

- It directly targets the mechanism that failed.
- It could produce a real learning principle if it works.

Why it is risky now:

- WESS already showed that object-state machinery works only when addresses are supplied, and predicted/unlabeled routing collapsed.
- Final-answer SCMLM-R could not even memorize 32 pairs in the current scoring setup.
- No evidence yet links this synthetic state machinery to official Entity/EWoK improvement.

If pursued, it should be a small mechanism experiment, not a full 100M route. The first construction should implement prefix/intermediate supervision and state-directed RTD on generated pairs, then test both-contexts-correct and causal/noncausal swap sensitivity. It should stop if trained-pair success remains near zero.

### Route B — legal custom-corpus route inspired by the leaderboard phenotypes

Scientific object: reconstruct a broad, legal, high-signal 10M training corpus rather than trying to inject a micro-world state variable. The corpus should combine natural linguistic diversity, readability control, high entity/relation density, and paired meaning-preserving views where available, while preserving official-data developmental signals that protect Reading/Supplement.

Why it is strong now:

- Both current top rows used custom 10M corpora, not only official BabyLM text.
- BabyLM rules allow custom training datasets within the 10M-word and exposure limits.
- The leader’s exact FineWeb simplification-pair file is gated, but its phenotype strongly suggests broad custom data is load-bearing.
- RecGPT demonstrates that near-leader Overall can be reached even with weak Entity if other NLP/Reading columns are high.
- This route aims at the actual leaderboard objective instead of one synthetic proxy.

Risks:

- The accessible simplification-pair substitutes tested so far (WikiAuto/ASSET/TurkCorpus) did not improve Entity/EWoK under plain WWM.
- A weak custom corpus can lose protected Supplement/Reading.
- External generation by large language models is not allowed unless the closed-system word accounting is satisfied, so corpus construction must use legal sources and non-learned processing or sources with counted words.

Concrete next version:

- Start from protected 8×480 baseline16k DeBERTa WWM, because it remains the best complete internal coordinate and protects Supplement/Reading.
- Build a 10M-word candidate corpus with exact accounting, not more than 30–40% non-official words in the first screen.
- Candidate sources should be legally accessible and source-recorded: official BabyLM core, SimpleWiki/Gutenberg/OpenSubtitles-like public material if allowed, high-quality human simplification/alignment sources only after filtering, and selected high-readability/high-diversity text. Do not use the gated `go76dof/Fineweb_simplification_pairs` training file.
- Compare official-only vs custom-mix at 10M or 20M exposure using fast but official-compatible columns: BLiMP, Supplement, Entity, EWoK, COMPS, GlobalPIQA, Reading, plus loss and truncation stats.
- If custom-mix moves only GlobalPIQA, it repeats the known weak pattern. It needs either Entity/EWoK movement or a RecGPT-like broad NLP/Reading gain large enough to improve Overall.

### Route C — RecGPT-style recursive causal route

Scientific object: reproduce or adapt the recursive shared-block causal phenotype.

Why it is interesting:

- RecGPT is close to the leader without solving Entity.
- It suggests a different route to Overall through grammar, COMPS, GlobalPIQA, and Reading.
- Its public resources include model card and code/dataset repositories.

Risks:

- It is a substantial new backend and official evaluation route.
- Weak Entity means it may not beat the leader unless combined with better data or protected masked strengths.
- A direct reproduction would be less scientifically original unless it reveals a new transferable principle.

Use it as a source of ideas and a possible parallel route if custom-corpus masked runs stall, not as the immediate main route.

## earlier analysis recommendation

The current final-answer state-counterfactual branch should stop. The highest-value next research work is to move out of synthetic state micro-worlds and rebuild around the real leaderboard phenotypes.

Primary next route: **legal custom-corpus reconstruction and controlled custom-mix screening on the protected DeBERTa WWM backbone**.

A narrow trajectory-supervised state experiment remains scientifically interesting but should be secondary and short. It should include prefix/intermediate supervision from the start and should not consume full official-scale compute unless it first shows trained-pair and held-out-pair state sensitivity.

The proposed alternatives are:

1. design the legal custom-corpus construction and its first official-compatible screen; or
2. compare this proposed pivot against the accumulated negative mechanism evidence before implementing it.

A new large-scale branch requires an explicit comparison of the proposed mechanism with the alternatives supported by existing evidence.
