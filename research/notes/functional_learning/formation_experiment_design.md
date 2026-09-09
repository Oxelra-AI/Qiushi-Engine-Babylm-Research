# dense focus common target analysis Formation-stage experiment design

## Scientific question

Can a different private-formation policy from chck_82M produce a model that exceeds
coherent86 (v4, Overall 42.1210)?

## Why this differs from failed recipes

Recent functional_learning experiments (causal intervention) all started from mature coherent86 and used:
- Small private-adapter learning rate continuation
- Short training (80 updates)
- Late modification of already-learned representations
- Various objective/credit modifications (weighted focus, source-dependence, answer-only)

All failed to produce broad improvement. The earlier analysis review concluded these were evidence
about late modification, not about the underlying principle.

This experiment returns to the chck_82M formation stage where:
- Private adapters are FRESH (never trained)
- Learning rate starts from warmup (not continuation)
- Ordinary WWM MLM credit (not specialized objectives)
- The learner coordinate is at maximum plasticity

## Key observation: coherent86 used only 22% of its planned schedule

The earlier analysis trainer used:
- lr_total_steps = 455 (designed for full 18M tail)
- max_tail_charged_words = 3,992,918 (~4M)
- Only 101 updates were taken (22% of schedule)
- At update 101, LR was at ~93% of peak (barely decayed)

This means coherent86 stopped BEFORE the learning rate could decay naturally.
The remaining 14M words and 354 schedule steps represent unused legal formation opportunity.

## Arms

### Arm A: `reference_4M` (reproduce coherent86)
- chck_82M → 4M words private formation → evaluate at alpha=0.75
- Same data, schedule, and hyperparameters as earlier analysis coherent_replay
- Expected: cheap7 ~44.106, reproduced coherent86

### Arm B: `full_18M` (extended formation)
- chck_82M → 18M words private formation → evaluate at alpha=0.75
- Same stream, starting from same row 530,944
- Same schedule (lr_total_steps=455), runs to natural completion
- 455 updates, LR decays to zero naturally
- Checkpoints every 1M words → fine-grained trajectory

### Arm C: `full_18M_warmed` (extended formation, extended schedule)
- chck_82M → 18M words private formation
- lr_total_steps increased to 1818 (4.5× longer) so LR at 455 updates matches Arm A's LR
- This separates "more data" from "different LR regime"
- LR at 455 updates ≈ 93% of peak (same as coherent86 endpoint)

## Evaluation plan
- Cheap7 at alpha=0.75 for all checkpoints
- If any checkpoint exceeds coherent86 fast screen (44.5643):
  - Run at additional alpha values (0.5, 0.75, 1.0)
  - If best exceeds by >0.3 points: full official-compatible evaluation

## Data streams
- Stream: frontier_consolidation's `cleanqwen_fineweb_compact_view_reinvest_100M.jsonl`
- Skip rows: 530,944 (same as earlier analysis)
- Full tail from 530,944: 116,456 rows, 17,987,505 words

## Reference values
- chck_82M cheap7: 43.959 (from earlier analysis summary)
- coherent86 (alpha=0.75) cheap7: 44.564, Overall: 42.121
- earlier analysis coherent_replay cheap7: 44.106 (at alpha=1.0!)
- earlier analysis spanbreak_replay cheap7: 43.121 (structure-destroyed control)

## Key artifact
- Trainer script: `scripts/formation_replay_trainer.py`
- Output: `data/formation_experiment/`
