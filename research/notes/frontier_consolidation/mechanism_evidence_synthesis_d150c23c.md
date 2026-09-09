# mechanism evidence synthesis: Mechanism evidence synthesis — what compact-view data efficiency establishes

## The DeBERTa compact-view reinvestment effect

Under the legal BabyLM Strict-Small coordinate (≤10M words, ≤100M exposure), replacing ~4.2% of the word budget with compact generated semantic views of FineWeb sources while reinvesting the saved words into additional source diversity produces:

- **+1.35 mean7 at 80M** over a matched clean control under the compliant tokenizer
- Component gains: BLiMP +1.53, Supplement +2.06, EWoK +2.03, Entity +2.33, COMPS +0.65, Reading +0.85
- The effect is **late-emerging** (weak at 20M, growing through 70-80M)
- The effect is **reproduced across two seeds** (43022: +0.685, 43123: +0.480)

## Architecture specificity: the effect does NOT transfer

| Architecture | Objective | Compact-minus-control | Stable families |
|---|---|---|---|
| DeBERTa-v2 8×480 (relative attention) | Masked LM | **+1.35 mean7** | Positive all 7 |
| GPT-2 8×480 (causal, absolute) | Next-token | +0.078 cheap7, -0.145 cheap6 | **Negative without GlobalPIQA** |
| RoBERTa 8×480 (bidirectional, absolute) | Masked LM | +0.042 cheap7, -0.115 cheap6 | **Negative without GlobalPIQA** |
| RoBERTa HS/LS/HD/LD factorial | Masked LM | +0.97 cheap6 interaction | **Composition artifact** (zero stable-relational) |

The compact advantage is specific to DeBERTa's relative position attention + masked denoising interaction. It does not transfer to absolute-position attention (RoBERTa) or causal objectives (GPT-2). factorial decomposition confirmed the interaction is a composition artifact at item level.

## Source-only construction fails: three negative arms

| Construction | Source-absent content | Fluency | Source span | Stable cheap6 vs compact |
|---|---|---|---|---|
| Extractive balanced | Zero | Telegraphic | 0.972 | **-0.257** |
| Extractive wide | Zero | Telegraphic | 0.928 | **-0.662** |
| Source-attested bridge | Zero | Fluent but 87% source-copy | — | **Not trained (closed)** |

- **Extractive views** provide broad source/tail access but lose on EWoK+Entity (the most informative stable families)
- **Source-attested bridge** outputs are overwhelmingly direct source copies (87.4% copy degree ≥ 0.999); the constraint collapses fluent re-expression into polished extraction
- Neither approach can introduce the novel vocabulary and genuine semantic recoding that natural compact views provide

## What the evidence rules out as the dominant mechanism

1. ~~Simple reciprocal source conditioning~~ — compact non-copy rewrite lift only +0.0616
2. ~~Compact-specific ordered source retrieval~~ — source-conditioned ordering is general to bidirectional MLM
3. ~~Same-sequence source-absent/copied target complementarity~~ — cross-realization probe: full-minus-drop intervals cross zero
4. ~~Source-absent innovation masking/loss weighting~~ — innovation masking failed (-1.05 mean7); controlling boundary prohibits reopening
5. ~~Edit-state/source correspondence as mature transfer~~ — source-absent channel is generic editing-distribution information
6. ~~Coherent word order alone~~ — ordered vs scrambled shows selective local NLL channel but -0.67 cheap7 at 40M
7. ~~Source access/tail coverage alone~~ — extractive wide with 98% source coverage still -0.66 cheap6
8. ~~Architecture-general contextual diversification~~ — RoBERTa, GPT-2, and factorial all negative on stable families

## What survives as the working explanation

Natural compact views create a **coupled multi-factor data treatment** whose components cannot be isolated by source-only construction:

1. **Novel vocabulary (source-absent content)**: ~17,891 instances across the compact pool; provides genuinely new prediction targets for masked denoising that are absent from source repetition
2. **Faithful semantic compression**: the compact view preserves the source's core proposition while changing its lexical/syntactic expression, creating diverse prediction contexts around shared content
3. **Fluent syntactic form**: grammatically complete sentences that DeBERTa's relative attention can model (vs telegraphic fragments that lack syntactic structure)
4. **Source-wide coverage with reinvested diversity**: compact views extract content from across the full source position range (66.99% coverage), then the saved words are reinvested into additional diverse sources

The working hypothesis is that DeBERTa's **relative position attention interacts with the compact-view structure** to create a learning signal that absolute-position attention cannot extract. Specifically:
- Relative attention makes DeBERTa sensitive to local syntactic/semantic structure rather than absolute position
- When the same content word appears in different syntactic contexts (source vs compact view), relative attention can learn the structural invariance
- This is unavailable to absolute-position models, where position encodes location rather than structure
- Novel vocabulary (source-absent words) provides additional prediction targets that are not deducible from surface position patterns

This is consistent with the target-deletion triangle: full compact > drop_copied_word (+1.104) > drop_abs (+0.459), showing that copied/shared target supervision is more consequential than source-absent targets alone, but both contribute.

## Score-improvement path

The existing endpoints are:
- chck_82M: Overall 41.94 (submitted, frontier-displayed)
- chck_84M: projected Overall 42.02 (not submitted)
- coherent86 α=0.75: projected Overall 42.12 (not submitted)

The chck_84M coherent replay is now running. Expected: frozen 84M + ~4M coherent private-path replay, then alpha sweep. If α=0.75 produces cheap7 > 44.18 with non-GlobalPIQA-only gain, it would supersede coherent86 as the strongest local endpoint.

## What this means for transferable principles

The compact-view effect is a real sample-efficiency phenomenon but currently **architecture-specific** (DeBERTa relative attention + MLM). Making it transferable would require either:
- Understanding the relative-attention mechanism well enough to replicate the learning signal in other architectures
- Finding a different form of the same principle (diverse contextual re-expression) that works with absolute attention or causal objectives
- Developing a new training objective that captures the invariance-learning benefit across architectures

This remains an open research question with potential high-impact implications for data-efficient pretraining.

## Files
- Bridge prototype result: `notes/source_attested_bridge_prototype_result.md`
- Bridge data: `data/improved_fluent_bridge/`
- Alpha sweep script: `scripts/coherent88_alpha_sweep.py`
- Mechanism evidence chain: related experiments→177→211→221→222→223→231→232→236
- RoBERTa factorial: confirms composition artifact for HS/LS/HD/LD interaction
