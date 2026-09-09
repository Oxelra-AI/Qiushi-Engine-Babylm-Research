# earlier analysis — Mechanism chain for the next route: order information, causal recursion, and SOTA path

## Why this route note exists

The earlier analysis custom-mix screen showed that replacing official text with more encyclopedic/simplified text under ordinary WWM does not produce the desired Entity–EWoK–GlobalPIQA movement and costs Supplement/Reading. The next work should compare routes that change the learning interface or experience structure, not another corpus-ratio patch.

The order-sensitivity result is from one checkpoint and one task family. It does not support further final-answer state-pair objectives in that setting, but it must not be promoted into a theorem that DeBERTa always ignores order. The unresolved question is whether order information is absent at all layers or is present and then lost before the final MLM score.

## Evidence to connect

### Order-sensitivity result

On `training/runs/r1_ordered_dynamic_5M/hf_model/chck_5M`, same-word-bag counterfactual passages differed by mean 16.5 tokens, but the final masked query-position hidden state had cosine 0.99999999 between the two operation orders, and the same candidate answer scored identically in both contexts. This explains why WWM, answer-span MLM, random negatives, and SCMLM-R final-answer ranking could not make both contexts correct.

Boundary: this was the final layer of one checkpoint on R1-style pairs. It does not settle intermediate layers, other checkpoints, or other training interfaces.

### RecGPT phenotype and implementation facts

RecGPT-10M is close to the public leader but has a very different column shape:

- BLiMP 73.11, Supplement 61.73, EWoK 52.62, Entity 16.59, COMPS 55.43, GlobalPIQA 40.68, SuperGLUE 66.64, Reading 6.92.
- Its Entity is worse than the protected DeBERTa reference (22.62), and EWoK is only comparable (52.62 vs 52.19).
- Its strength is mainly BLiMP/COMPS/GlobalPIQA and good Reading/Supplement, not Entity tracking.

Recovered RecGPT training-code facts from `serdardoesml/bblm26-recgpt`:

- MIT-licensed repository at commit `178566ec38540e91dc951d6b094efc89c0ee19b3`.
- Custom `RecGPTForCausalLM` with `RecGPTConfig` and HF causal interface.
- Shared Transformer block recursively applied for configurable `recursive_depth` (model card: 16 iterations).
- Factorized embeddings: token embedding size 192 projected to hidden size; untied LM head because the code comments say tied embeddings hurt recursive models.
- Causal/RoPE/FlexAttention stack with RMSNorm, fused QKV, gated attention, zero-initialized attention output and MLP down projection.
- Optimizer split: RecGPT reports Muon for recursive block and AdamW for embeddings/embedding-related parameters; code exposes separate `lr_embed`, `lr_block`, and `lr_nl`, and adds an auxiliary `nl_loss` from `nl_aux_model.py` during training.
- Tokenizer is a 32,768-token BPE; training uses a token-batch interface and official checkpoint marks.

Therefore a future RecGPT-style screen should not be a generic GPT run. The relevant mechanism package is causal scoring + recursive weight reuse + factorized/untied embeddings + gated attention + Muon/AdamW split + auxiliary hidden/embedding loss. We should isolate which part matters before spending full training compute.

## Mechanism chain as currently understood

1. A causal scorer makes each position depend only on its prefix, so operation order can be represented naturally.
2. Recursion and shared-block refinement may improve grammar, composition, and commonsense scoring per parameter, explaining high BLiMP/COMPS/GlobalPIQA.
3. Order-sensitive prefix representation alone does not solve Entity, because RecGPT's Entity is only 16.59. Entity requires persistent entity identity and state binding across re-mentions, which neither plain WWM nor RecGPT clearly supplies.
4. The desired phenotype is not pure RecGPT replacement; it is to keep protected Entity/Supplement/Reading strengths while gaining RecGPT-like BLiMP/COMPS/GlobalPIQA.

## Immediate probe before architecture choice

Build a no-training probe on the R1 ordered-dynamic chck_5M checkpoint:

1. Regenerate the same R1 counterfactual pair family used in the order-sensitivity comparison.
2. For each layer from embeddings through final layer, extract masked query-position hidden states for context A and context B.
3. Compute by layer:
   - cosine similarity between `h_A^l` and `h_B^l`;
   - L2 norm of `h_A^l - h_B^l`;
   - whether the difference direction aligns with the answer-score direction if passed through the existing MLM head transform where possible.
4. Compute gradients of the true-vs-counterfactual answer score with respect to the hidden states at each layer for paired examples.
5. Report whether order information is absent at every layer or appears at intermediate layers and collapses before the final score.

Interpretation:

- If every layer is nearly identical across order pairs, then ordinary bidirectional WWM in this checkpoint never formed the needed order representation; Route C becomes stronger.
- If middle layers separate the orders but the final layer collapses, a bidirectional repair may exist: intermediate residual readout, layer mixing, shallower scoring, or auxiliary supervision at the layer that contains order.
- If gradients into earlier layers are too small or symmetric to separate the paired orders, loss-side repairs through the final MLM head are unlikely to work; a different interface is needed.

## Route after the probe

Do not launch full RecGPT reproduction yet. RecGPT itself has weak Entity. If the probe supports a causal/recursive route, the first training screen should be a small, controlled comparison that isolates mechanism parts:

- dense causal control with the same 10M official corpus;
- lightweight RecGPT-style recursive causal model with official data and a reproducible tokenizer choice;
- optionally a causal/WWM hybrid only after no-future-token leakage and score-interface checks are clean.

The first score target is not SOTA; it is whether the causal/recursive interface produces the RecGPT-like BLiMP/COMPS/GlobalPIQA/Reading phenotype under our legal data and whether it can be combined with protected masked strengths. Full 100M training should wait until this mechanism chain is resolved.
