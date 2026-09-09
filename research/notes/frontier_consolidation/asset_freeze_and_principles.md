# asset freeze and principles — Protected Assets and Distilled Principles

## Part A: Protected assets (frozen, do not modify)

### 1. Submitted public endpoint: chck_82M (Overall 41.94)
- Model SHA256: `93ceb76adf5a33d349f1de33e988e6ed0c2b2a547dbd92cf83cc952f8e2591b3`
- Architecture: AdapterDebertaV2ForMaskedLM, 35,463,008 params (8×480 DeBERTa-v2 + bottleneck-128 adapters, scale 1.75)
- HF: `leslie721007/babylm-strict-small-scale1p75-chck82` rev `f49775dc5eafbf3a14d6f2107ec5368188d2b1dd`
- Carrier SHA: `dcad3d8cf285a69d31910a32c62c80689022a3ff459a0956401ab7f4b3542237`
- Public Overall 41.94, rank 1
- Exposure: 82,012,495 words; legal pool SHA `215944978157394dbecf2039f2c1e1806bfcbb9701421920f78605eb58975a23`
- Tokenizer SHA: `91b775514b9f4e2d1f37c28445ab98181007e16d14f763955168547e293ee8f9`
- Bit-identical reproduction confirmed (chck82 reproducibility and peak characterization)

### 2. Strongest local endpoint: coherent86 α=0.75 (Overall(AoA0) 42.121)
- Model SHA: `e14d757ae51b41e33bf0813f841248fecd1eefeb9e040f520c4c6203343b15c8`
- Config: `private_adapter_scale: 0.75`, same base weights as chck_82M with 995,584 coherent-replay private params
- HF: `leslie721007/babylm-strict-small-coherent86` rev `ad128352c154702704e8b29c24cf94d986fe5c7f`
- Carrier SHA: `40181994810e21bc823474a3e4ac84c8eb42213e03904d36a60a4698477d1994`
- cheap7: 44.1814, SuperGLUE: 69.8192, Overall(AoA0): 42.1210

### 3. Reproduction recipe
```
Phase 1: Train scale1.75 adapter DeBERTa-v2 on legal compact-view reinvest pool for 82M words
Phase 2: Freeze all non-private, coherent replay through private adapter for 4M words  
Phase 3: Inference-time private_adapter_scale = 0.75
```
- Training script: scripts/adapter_scaled_trainer.py
- Modeling: scripts/adapter_modeling.py (→ adapter_scaled_modeling.py)
- Replay: scripts/frozen82_fastpath_replay_trainer.py
- Pool: data/density_cleanqwen_overlay_medium_riskhard/
- Tokenizer: data/compliant_tokenizer/

---

## Part B: Distilled principles from 168 steps of evidence

### Principle 1: Compact Semantic Redundancy Reduction + Source Diversity Reinvestment
**Statement**: Replacing verbose natural text with semantically faithful compact paraphrases, while reinvesting freed word budget in diverse source material, produces broadly better language competence than either simple repetition or adding diverse unrelated text.

**Evidence**: 
- Compact-view reinvest vs clean control: +1.35 mean7 at 80M maturity (earlier analysis)
- Compact views beat independent breadth: +0.54 cheap7 at 100M (fw absolute progress decision discipline)
- Two-seed treatment effect: +0.58 Overall average (endpoint decision ledger and route quality)
- Effect survives legal tokenizer change (related experiments)

**Generalizability hypothesis**: In any data-constrained pretraining, deduplicating semantic content through faithful compaction and reinvesting the budget in source diversity improves sample efficiency. This should scale to larger corpora and models because the mechanism is about information structure, not model capacity.

### Principle 2: Function-Preserving Residual Capacity
**Statement**: Adding zero-initialized nonlinear side branches to a pretrained backbone — so the composite function is EXACTLY the original at initialization — allows the model to discover new learning trajectories that the original architecture could not reach, without destroying existing competence.

**Evidence**:
- Scale1.75 adapter produced the 82M peak at 41.94 Overall (chck82 frozen private tail design)
- Disabled adapter exactly reproduced the base model (adapter 20M closure)
- The adapter redirected learning trajectory: different checkpoint had different capability allocations (chck82 reproducibility and peak characterization)
- Scale2.0 barely improved over 1.75; scale matters nonlinearly (adapter scale sweep plan)

**Generalizability hypothesis**: Function-preserving capacity expansion is a general principle for continued learning. It differs from LoRA (which starts from a trained model and finetunes) because it modifies the learning trajectory from scratch. It should apply to any architecture where residual connections allow zero-output initialization.

### Principle 3: The Narrow Peak Phenomenon (Capability Forgetting Under Continued Training)
**Statement**: When training loss continues to improve, downstream task performance on relational, commonsense, and entity-tracking tasks peaks at an intermediate point and then declines. The peak is narrow (±2M word window), reproducible, but not predicted by any tested corpus-internal metric.

**Evidence**:
- cheap7 trajectory: 77M=43.28, 80M=43.81, 82M=43.96, 83M=43.81, 100M=43.54 (chck82 reproducibility and peak characterization)
- 82M→100M decline driven by EWoK/Entity/GlobalPIQA/SuperGLUE (chck82 reproducibility and peak characterization)
- Three benchmark-independent selectors all failed to locate the peak (related experiments)
- MLM loss, token-level IG, structural probes all declined or were uncorrelated (related experiments, 162)

**Generalizability hypothesis**: This phenomenon likely occurs in all LLM pretraining but is masked by increasing model size. Understanding and controlling it could improve training efficiency for LLMs of any scale. The peak may correspond to a transition between memorization-dominated and generalization-dominated learning phases.

### Principle 4: Pair Atomicity (Same-Window Multi-View Learning)
**Statement**: A model learns more from seeing source text and its semantic paraphrase in the same training window than from seeing them separately. Joint visibility of 0.999 vs 0.572 correlated with better downstream performance.

**Evidence**:
- Joint visibility audit: row-atomic 0.999 vs row-chunked 0.572 (mechanism preservation summary)
- Pairfit with 1.0 complete visibility outperformed chunked (rowchunk pairfit screen and route decision)
- Source+compact directly adjacent (0 word gap) in all 12,155 pairs (compact recurrence schedule audit)

**Generalizability hypothesis**: Implicit consistency learning from multi-view text in the same context window is a general data augmentation principle for language models. It creates invariance to surface form variation without explicit objectives (explicit R-Drop and coherence-margin both failed: dualview panel readiness and budget, 158-159).

### Principle 5: Competence Redistribution vs. Broad Acquisition
**Statement**: Most learning interventions redistribute competence across task families rather than adding broad new competence. An intervention that improves aggregate score while losing net discrete items is redistributive, not accumulative.

**Evidence**:
- 14 legal interventions: mean mature cheap7 delta -0.320 (scale1p75 item mechanism and cross route tradeoff)
- Coherent86 vs chck82: +0.147 cheap7 but net -115 discrete items (frozen anchor coherent replay item reading)
- Alpha scaling monotonically redistributes: 3,116 activations entangled with 3,231 damages (coherent86 multiarm mechanism reading)
- Private-scale interpolation: amplitude knob, not mechanism (private scale endpoint vs mechanism synthesis)

**Generalizability hypothesis**: In data-constrained settings, most optimization and architectural modifications move competence between evaluation axes rather than creating it. Distinguishing redistribution from acquisition requires item-level analysis, not just aggregate scores. This matters for all LLM evaluation methodology.

---

## Part C: Open scientific questions worth pursuing

1. **Does compact-view reinvestment scale beyond 10M words?** If the principle holds at 100M, 1B, or 10B words, it would transform pretraining data preparation.

2. **Can the narrow peak be widened or eliminated?** If we understand why downstream competence peaks while loss improves, we could save massive compute across all LLM training.

3. **What is the optimal information density for pretraining?** The compact-view works because it changes the density of semantic information per word. Is there a theoretical optimum?

4. **Does function-preserving residual capacity generalize to larger architectures?** Zero-output adapters at scale could be a general architecture principle for efficient pretraining.

5. **Can multi-view consistency be made explicit without destroying learning?** Implicit same-window pairing works; explicit objectives (R-Drop, coherence-margin) failed. Why? What is the correct explicit formulation?
