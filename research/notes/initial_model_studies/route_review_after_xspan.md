# XSpan results and alternative objective designs

## Bottom line

The earlier analysis low-ratio WWM+CLM hybrid route is technically possible and scientifically worth testing, but it should not be treated as the single best next route anymore. A direct Hugging Face probe found the visible Strict-Small leader card for `go76dof/wwm_curriculum_simplification_40k`, and it changes the competitive picture.

The immediate next research work should compare live 2026 competitive mechanisms, not only build a local hybrid variant:

1. **Leader-style DeBERTa + simplification-pair data + WWM/token curriculum + 40k tokenizer**.
2. **Official-corpus GPT-BERT/AdaMuon-style hybrid objective**, because multiple 2026 entries show strong BLiMP/Supplement/Entity tradeoffs with this family.
3. **Protected baseline16k DeBERTa WWM**, as the stable reference with strong Supplement and Reading.

The current XSpan rho0.15 route should remain stopped as-is.

## New evidence from direct HF card fetch

### Visible leader: `go76dof/wwm_curriculum_simplification_40k`

Downloaded card: `data/hf_probe_revision_123/go76dof__wwm_curriculum_simplification_40k__README.md`

Key facts:

- 34.7M-parameter DeBERTa-v2 style masked LM.
- Trained on `go76dof/Fineweb_simplification_pairs`, 9,999,969 whitespace words.
- Original FineWeb sentence followed by simplified rewrite; blank lines separate pairs.
- 40k SentencePiece BPE tokenizer trained for this data condition.
- Architecture: 12 layers, hidden 384, intermediate 1280, 12 heads, dropout 0.1, DeBERTa-v2 relative attention.
- Optimizer: LAMB, max LR 0.007, cosine schedule.
- Training: 10 epochs, sequence length curriculum 64→256, masking curriculum WWM for epochs 1–7 then token masking for epochs 8–10.
- Leaderboard scores: Overall 41.80, EWoK 56.07, Entity 28.45, GlobalPIQA 39.67, SuperGLUE 69.79, but lower Supplement/Reading than our protected model.

This is not merely a tokenizer result. It is a package: external simplification-pair data, 40k tokenizer, smaller-hidden/deeper DeBERTa, LAMB, sequence-length curriculum, and WWM→token masking curriculum. Our earlier official40k experiment was on the official corpus and hurt Reading; it does not falsify this leader package.

### 2026 hybrid entries

Downloaded cards:

- `data/hf_probe_revision_123/qyxu1994__instanton-hybrid__README.md`
- `data/hf_probe_revision_123/svsatheesh__BabySteps_MurphysLaw-10M-mixed__README.md`

`instanton-hybrid`:

- 40.2M GPT-BERT hybrid, official corpus, 50/50 masked/causal mix.
- LAMB, LR 0.007, z-loss, mask ratio 0.30→0.15, sequence length 128→256→512.
- Scores: BLiMP 72.13, Supplement 60.86, EWoK 50.15, Entity 19.87, GlobalPIQA 37.14, Reading 6.35, Overall 40.78.

`BabySteps_MurphysLaw-10M-mixed`:

- GPT-BERT hybrid on official corpus, AdaMuon/AdamW-style split, tail weight averaging.
- 12 layers, hidden 384, heads 6, intermediate 1280, 16k vocab, sequence length 128.
- Scores: BLiMP 71.59, Supplement 63.93, EWoK 51.94, Entity 27.95, GlobalPIQA about 35, SuperGLUE about 69.9.
- Card reports data-centric experiments did not beat a faithful GPT-BERT reconstruction with Muon-family optimizer and tuned LR.

These cards strengthen the hybrid route, but also show the key pattern: hybrids can lift BLiMP/Supplement and sometimes Entity, yet may not solve GlobalPIQA or EWoK as strongly as the leader simplification-pair DeBERTa.

### RecGPT

`Serdar404/RecGPT-10M` remains important but not sufficient alone:

- Recursive causal model, custom data, Muon/AdamW split.
- Strong BLiMP, COMPS, GlobalPIQA, but Entity only 16.59.
- It supports the idea that causal/sequential modeling helps sequence plausibility, but not that pure causal modeling solves the largest Entity gap.

## Review of the earlier analysis proposed low-ratio hybrid

### What is strong

- The route is grounded in GPT-BERT and AntLM, not invented from local noise.
- DeBERTa-v2 can accept a 3D attention mask: Transformers inspection shows `DebertaV2Encoder.get_attention_mask` unsqueezes 3D masks and self-attention masks attention scores directly, so a causal branch can be made technically leak-free if tested.
- It preserves a standard MLM checkpoint for official evaluation.
- It directly addresses a failure of XSpan: narrow selected spans from random init were too easy to learn without correct s1; a causal stream supervises every token under left-context information flow.

### What is weak

- The proposed “label position t with token t+1” branch is not exactly GPT-BERT’s MNTP. GPT-BERT unifies MLM and CLM by shifting outputs one token to the right: when token k+1 is masked, the prediction is output at k. A faithful implementation should decide between (a) plain CLM branch and (b) MNTP-style shifted masked branch. Those are scientifically different.
- A low causal ratio may be too weak at 1M to show a signal, while a 50/50 ratio may damage the WWM interface. The prior 2026 cards include both 50/50 and mixed-objective variants, but the leader is pure MLM with better data/curriculum.
- Hybrid route is not obviously the fastest path to the current leader because the leader is not hybrid; it is DeBERTa+FineWeb simplification pairs+curricula.
- Entity is the hardest ambiguity: BabySteps hybrid reports Entity 27.95, near the leader; instanton hybrid is only 19.87; RecGPT is 16.59. This suggests optimizer/architecture/data details dominate objective family alone.

## What exact tests are required if building hybrid

Before any 1M hybrid screen:

1. **Causal no-leak unit test**: create two inputs sharing prefix but differing in future tokens. Under the causal branch, logits/loss for prefix positions must be unchanged to numerical tolerance. Under bidirectional branch they should change. This must be saved as a runnable script and result JSON.
2. **Attention mask shape test**: verify the DeBERTa branch receives a [batch, query, key] lower-triangular-and-padding mask and that all saved checkpoints still load as `AutoModelForMaskedLM`.
3. **Objective identity test**: choose and record whether the branch is plain CLM or GPT-BERT-style MNTP. If MNTP, labels and positions must match the paper’s shifted-output formulation.
4. **Small smoke**: record WWM loss, causal/MNTP loss, token counts, exposure counts, and direct checkpoint loadability.

1M comparison should include:

- protected WWM-only matched code path,
- low-ratio hybrid,
- 50/50 hybrid or MNTP high-signal arm if smoke is clean,
- possibly leader-package probe if data/tokenizer construction is already available.

Do not proceed by only looking at BLiMP. Require relation-gap movement: Entity plus EWoK or GlobalPIQA direction, while preserving Supplement and Reading close to WWM.

## Recommended next route after review

The strongest next step is not pure implementation of the earlier analysis low-ratio hybrid alone. The proposal prepares two executable branches and then chooses by a small pilot:

### Branch A — faithful 2026 hybrid reconstruction on our infrastructure

Build a leak-tested GPT-BERT/MNTP or CLM+WWM trainer on the protected DeBERTa/LTG-like size. Inherit from the 2026 hybrid cards:

- consider hidden 384, 12 layers, intermediate 1280 as a parameter-compatible alternative to 8×480;
- consider LAMB or Muon-family optimizer only after plain AdamW code path is verified;
- include mask schedule 30→15 only as a later single-factor repair, not in the first smoke.

### Branch B — leader-style simplification-pair DeBERTa package

The leader card is now the most directly relevant source for closing the exact gap. Before expensive training, retrieve or inspect `go76dof/Fineweb_simplification_pairs` metadata and verify legal word accounting. Then design a faithful small-screen:

- 40k SP tokenizer and 12×384 DeBERTa shape,
- LAMB optimizer and 64→256 length curriculum,
- WWM7→Token3 mask curriculum,
- FineWeb simplification pairs or a legally reconstructed equivalent.

This branch directly targets the columns where our protected model is behind: Entity, EWoK, GlobalPIQA, SuperGLUE. It risks Reading/Supplement, but the leader’s Overall shows the tradeoff can win.

## Next experiment

The XSpan evidence does not justify scaling. The best next work is to make a concrete execution plan for a two-branch pilot:

1. A faithful, leak-tested hybrid trainer smoke and 1M screen.
2. A leader-package feasibility probe: dataset access, tokenizer files, and a tiny tokenizer/model smoke with exact word counting.

If execution budget forces one first action, build the hybrid no-leak smoke only if it is fast; otherwise inspect and stage the leader dataset because it is the current visible SOTA and was newly recovered in this step.
