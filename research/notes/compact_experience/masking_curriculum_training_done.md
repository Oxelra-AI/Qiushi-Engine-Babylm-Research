# masking curriculum training done: Masking Curriculum Trainer — Training Complete, Evaluation Pending

## Trainer
- Script: `experiments/archive/compact_experience/scripts/masking_curriculum_trainer.py`
- Unified framework treating WWM→token, mask-prob decay, and AMLM-hard as three orthogonal
  dimensions of prediction-granularity curriculum control
- Produces dynamics traces (loss/accuracy by frequency band, effective mask rate, entropy)

## Trained Arms (4M words, official corpus, DeBERTa-v2 8x480, seed43, baseline16k)
| Arm | Curriculum | Final Loss | Key Dynamic |
|-----|-----------|-----------|-------------|
| wwm_fixed_4m_seed43 | wwm_fixed (0.15) | 6.644 | Baseline control |
| wwm_to_token_4m_seed43 | wwm→token at 70% | 6.718 | Switch visible at step ~274 |
| amlm_hard_4m_seed43 | WWM + decay 0.30→0.15 + difficulty | 6.508 | Eff rate 0.41→0.27, 39 AMLM updates |
| amlm_hard_switch_4m_seed43 | AMLM-hard + switch at 70% | 6.660 | Combined dynamics |

## Evaluation Status
- Evaluator: `experiments/archive/compact_experience/scripts/eval_curriculum_4m.py`
- FIXED: evaluation repo path is `experiments/archive/initial_model_studies/repos/babylm-eval/strict`
  with data at `evaluation_data/` (not `eval_data/`)
- Need to verify BLiMP smoke works with corrected paths, then run full screen
- Both H100s available and idle

## Key Scientific Observations (pre-evaluation)
1. AMLM-hard achieves lowest loss (6.508) — difficulty weighting helps optimization
2. wwm_to_token has HIGHER loss than wwm_fixed — expected since token-level masking
   creates more independent hard predictions
3. Loss ordering does NOT predict task performance (confirmed in cs 4m residualized eval C/S experiment)
4. AMLM effective mask rate exceeds nominal, proving difficulty concentration is active
5. All arms produce dynamics_traces.jsonl for post-hoc analysis of WHEN transitions help

## Next: Run full fast evaluation with corrected paths, aggregate contrasts
