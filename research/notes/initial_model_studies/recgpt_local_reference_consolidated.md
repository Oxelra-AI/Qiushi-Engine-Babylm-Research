# recgpt local reference consolidated — RecGPT local reference: consolidated status and route forward

## Local Public RecGPT Reference (verified under the retained evaluator)

| Column | Local score | Leaderboard | Delta | Status |
|---|---:|---:|---:|---|
| BLiMP | 73.20 | 73.11 | +0.09 | ✓ reproduced |
| Supplement | 63.24 | 61.73 | +1.51 | ✓ reproduced |
| EWoK | 52.82 | 52.62 | +0.20 | ✓ reproduced |
| Entity | 16.73 | 16.59 | +0.14 | ✓ reproduced (weak, as expected) |
| COMPS | 55.47 | 55.43 | +0.04 | ✓ reproduced |
| GlobalPIQA mean | 38.71 | 40.68 | -1.97 | partial (evaluator/version) |
| Reading mean (space-fix) | 2.03 | 6.92 | -4.89 | partial (tokenizer+evaluator) |

## Reading Fix Analysis

- **Without space-fix:** eye 0.63, self-paced 0.01 (Reading 0.32) — target tokenized as beginning-of-string
- **With space-fix:** eye 2.31, self-paced 1.75 (Reading 2.03) — target tokenized as continuation
- **Leaderboard:** eye 9.35, self-paced 4.49 (Reading 6.92)

The space-prefix fix improved Reading 6× (0.32→2.03) and made all RT regression coefficients highly significant (p<1e-4). The remaining gap is likely:
1. RecGPT's 32k BPE frequency-weights correlate more strongly with length/Subtlex, leaving less unique contextual variance after baseline regression
2. Possible evaluator version or torch.compile numerical differences between author's submission and our local unfused FlexAttention run
3. Not a problem with our loading/patching (confirmed by correct NLP columns)

## Scientific judgment for route forward

The RecGPT local reference is now verified at the level needed for same-budget mechanism comparison:
- **For within-evaluator comparisons**, local RecGPT is the correct causal/recursive target
- **If our own trained model beats local RecGPT under the same evaluator**, that is real progress
- **The leaderboard Reading gap does not invalidate the reference** because all our models will be evaluated under the same local harness

## Evidence files
- Full column evaluation: `data/recgpt_public_causal_scores.json`
- Reading space-fix: `training/runs/recgpt_reading_spacefix/detailed_results.json`
- Reading anomaly diagnostic: `data/recgpt_reading_anomaly_diagnostic.json`
- Patched model: `data/recgpt_local/patched_model`
- CUDA smoke test: `data/recgpt_local/patched_cuda_smoke.json`

## Next: official-corpus RecGPT-style training

The public RecGPT reference is now locally real. The next constructive work is:

1. **Extract and adapt the RecGPT training code** from `data/external/38424423a97c_serdardoesml-bblm26-recgpt-*.zip` for our legal official BabyLM corpus
2. **Design a controlled same-budget training** that:
   - Uses the official 10M-word BabyLM corpus (not the gated custom corpus)
   - Matches RecGPT's 10-epoch / 100M-word exposure budget
   - Uses the same architecture (RecGPTForCausalLM, recursive_depth=16, hidden=768, embedding=192, FFN=12288)
   - Trains with the public RecGPT optimizer split (Muon for recursive block, AdamW for embeddings)
   - Produces HF-compatible checkpoints for local evaluation with `--backend causal`
3. **Isolate mechanism contributions** by comparing:
   - Full RecGPT recipe on official data → tests data vs architecture
   - Ablations: depth=1 (no recursion), AdamW-only (no Muon split), tied embeddings
4. **Evaluate under the same local harness** so all scores are directly comparable to local RecGPT reference

The target: if official-corpus RecGPT matches or exceeds local public RecGPT on BLiMP/Supplement/COMPS/GlobalPIQA/EWoK, the causal/recursive interface IS the value-add, not the gated data. If it falls substantially short, the gated custom corpus was load-bearing.
