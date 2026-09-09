# corruption vs loss design design note: corruption-vs-loss factorial and BabyLM intervention design

## Factorial experiment

### Design
Three arms, each from fresh coherent86 private-adapter state:
1. **clean_answer_only**: bg_prob=0, answer-only loss (replication of earlier analysis answer-only, expected ~10/12 held)
2. **corrupted_answer_only**: bg_prob=0.15 applied to INPUT, loss only at forced answer position
3. **corrupted_answer_plus_bg**: bg_prob=0.15 applied to INPUT, answer_loss + bg_loss (answer at full weight, bg separate)

Arms 2 and 3 see IDENTICAL corrupted inputs. The only difference is whether background masked positions contribute gradients.

### Separation logic
- Arm 2 vs Arm 1 → **evidence effect**: does input corruption destroy the relational information needed to resolve the answer?
- Arm 3 vs Arm 2 → **gradient effect**: do background gradients interfere with answer learning, even when the answer gets full weight?

### Possible outcomes and implications

**Outcome A: Arm 2 ≈ Arm 1 (corruption harmless), Arm 3 << Arm 2 (bg gradients destructive)**
→ Background gradients are the problem. In BabyLM training:
- Contrastive packets can coexist with ordinary WWM on other text
- Answer positions need forced masking and explicit weighting
- Background targets from the SAME packet can be suppressed, but ALN text can train normally
- This motivates a separated loss: answer_loss(contrastive) + bg_loss(ALN), with an explicit ratio

**Outcome B: Arm 2 << Arm 1 (corruption destructive), regardless of Arm 3**
→ Evidence corruption matters. In BabyLM training:
- Contrastive packets need their relational structure protected from masking
- Entity names, source states, and evidence positions should not be corrupted
- This motivates a selective masking strategy: only mask answer positions in contrastive packets
- ALN text can still use standard WWM

**Outcome C: Arm 2 ≈ Arm 1 (corruption harmless), Arm 3 ≈ Arm 2 (bg gradients harmless)**
→ The earlier analysis focused+background failure was loss DILUTION (answer weight diluted by bg target count)
- The fix is simply proper loss weighting, not masking strategy changes
- In BabyLM training, standard masking everywhere plus explicit answer loss weighting should suffice
- This is the most optimistic outcome for practical integration

**Outcome D: All arms fail**
→ The template task itself may be too difficult or the 500-epoch schedule insufficient
- Need to revisit the task design or training parameters

## Natural state-update row compatibility

### Natural state-update row format
- 55 accepted pairs (110 training rows) from pilot
- Fields: pair_id, packet_type, source_sentence, update_sentence, use_sentence, use_sentence_frame, full_text, answer_text, foil_text, entity_name, distractor_entity, answer_in_use_spans
- Answer token counts: 8/110 one-token, 102/110 multi-token (2-6 tokens)
- The current one-token scorer requires one-token answers → cannot consume most natural state-update rows

### Multi-token scorer requirements
Compatibility with natural state-update rows requires a scorer that:
1. Locates answer span in use_sentence via use_sentence_frame + {STATE} placeholder
2. Handles different token counts between answer and foil
3. Scores symmetrically (same method for both candidates)

Options:
- **MLM pseudo-log-likelihood**: mask all answer positions, score each answer token independently, sum/average
- **Length-normalized sum**: sum log p(token_i | all masked) / n_tokens
- **Min-probability**: use the minimum token probability as a robustness measure

### Integration priority
Entity/source-state filtering remained unresolved (only 55/512 accepted). The proposed compatibility adapter depended on:
1. The factorial result decides the masking strategy
2. Production-quality rows become available under tightened filters
3. The multi-token scorer is validated against the one-token scorer on shared examples

## Four-cell coherent86 comparison (after factorial result)

### Cells
1. **ALN continuation** (control): coherent86 + remaining legal ALN text, standard WWM, no contrastive packets
2. **ALN + contrastive, standard WWM**: coherent86 + ALN + contrastive packets, all standard 15% WWM
3. **ALN + contrastive, answer-weighted**: coherent86 + ALN + contrastive packets with explicit answer-position forcing and loss weighting (ratio determined by factorial)
4. **ALN + contrastive, answer-weighted + parent anchor**: same as cell 3 plus KL to coherent86 parent

### Success criteria
- Joint improvement on held-out UPDATE and RETAIN recipient-sensitivity (measured by recipient-sensitivity scorer)
- No unacceptable cheap7 decline (< 0.3 points)
- Entity column stability
- Honest comparison against coherent86 alpha0.75 baseline

### Experimental prerequisites
The proposed four-cell comparison depended on:
1. Factorial results deciding the masking strategy
2. Enough validated contrastive rows, from natural packets or scaled templates
3. A small pilot at the same 87M checkpoint as the clean d-component ablation before the full legal suffix
