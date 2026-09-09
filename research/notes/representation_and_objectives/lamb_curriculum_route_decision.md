# lamb curriculum route decision — Route pivot: LAMB + sequence-length curriculum

## Decision

After exhaustive testing of relation-targeted training signals (corpus directional pair census), pivoting to
the two biggest untested factors from the visible leader's recipe: **LAMB optimizer** and
**sequence length curriculum** (64→128→256).

## Evidence supporting the pivot

### Relation route closed
- Dense PVDM (pvdm full readout synthesis): worsened all relation surfaces
- Visible-target sparse auxiliary (sparse relation aux calibration): visible-context lookup, not conditional prediction
- Full-context pivot substitution (full context pivot substitution probe): already saturated in failing models
- Naturally masked alternative ranking (fullctx aux budget audit): 94% saturated
- Broad two-context/two-target pair pool (relation filtered pair pools): unrelated targets, not competing alternatives
- Cross-row opposition funnel (intrarow calibration synthesis): weak function-word shells
- Same-context two-argument funnel (intrarow calibration synthesis): compact-dominant slot assignment
- Counterfactual exchange (counterfactual exchange clean subset): first unsaturated signal, but sparse and contaminated;
  v4 exact microtemplates only 49 cases from 20k rows (2 spatial, 1 temporal)

Scientific conclusion: the model's local pivot-consequence compatibility and slot assignment
are already strong. The unsolved weakness (context-conditioned alternative binding) cannot be
addressed by any local training signal we've been able to construct at useful density.

### Leader recipe gap analysis
The visible leader (41.80) uses a fundamentally different training recipe:
- **LAMB optimizer** at LR 0.007 (vs our AdamW at LR 0.001)
- **Sequence length curriculum** 64→256 (vs our fixed 256)
- DeBERTa-v2 12×384, 12 heads, FFN 1280 (vs our 8×480)
- 40k SentencePiece tokenizer (vs our legal40k byte-BPE)
- WWM→token masking transition (tested, near-zero effect at 100M with AdamW)
- FineWeb simplification pairs (we use compact-view reinvestment pairs)

The two untested factors are LAMB and curriculum. These are not mere hyperparameter changes:
- LAMB normalizes updates per-layer by ||weight||/||adam_step||, enabling 7× higher LR
- Sequence curriculum starts with 4× more gradient steps per word (at seq_len 64)
- Together they change the learning dynamics fundamentally

### Muon branch closed
50M broad columns (s128_t47_tool1, collected lamb curriculum route decision):
- Continuous Muon50: cheap7 42.57 (+0.59 vs AdamW 41.97); EWoK +3.89, GlobalPIQA +1.01
- Muon20→AdamW50: cheap7 42.52 (+0.55); GlobalPIQA +2.03, Entity -0.85
- Muon40→AdamW50: cheap7 41.89 (-0.08); GlobalPIQA -2.43

Combined with 50M EWoK (intrarow calibration synthesis): continuous Muon improves EWoK but abrupt empty-moment
switches don't repair hard GlobalPIQA rows. Abrupt switch artifacts formally closed.
Smoother/moment-preserving consolidation remains open as a future direction.

## Training runs launched

| Task | Architecture | Optimizer | Curriculum | GPU | Comparison target |
|------|-------------|-----------|------------|-----|-------------------|
| s131_t18_tool1 | 12×384/12h/FFN1280 | LAMB 0.007 | 64→128→256 | 0 | depth 41.03 |
| s131_t18_tool2 | 8×480/8h/FFN1920 | LAMB 0.007 | 64→128→256 | 1 | legal40k 41.14 |

Both use: legal40k tokenizer, compact view reinvest 100M data, bf16 autocast,
cosine warmup 6%, weight decay 0.01, fixed WWM 0.15, seed 43022.

## Expected outcomes and next actions
- If LAMB + curriculum improves over baseline → full official evaluation
- If 12×384 > 8×480 with LAMB → leader's architecture confirmed as better with LAMB
- If neither improves → gap is not in optimizer/curriculum; return to data or representation
- After training: run cheap7 columns first, then full official evaluation if promising

## Files
- Trainer: `scripts/lamb_curriculum_trainer.py`
- Smoke test: `data/smoke_test/`
- Training runs: `training/runs/lamb_curriculum_12x384_legal40k_seed43022/`
  and `training/runs/lamb_curriculum_8x480_legal40k_seed43022/`
- 50M Muon broad: `data/muon_switch_50m_eval/muon_switch_50m_summary.json`
