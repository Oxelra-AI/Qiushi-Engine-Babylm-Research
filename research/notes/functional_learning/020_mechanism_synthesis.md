# Step019b/020 mechanism synthesis: held-symbol transfer loss comes from non-answer pressure on the contextual body, not held-row drift

## Scientific question

budget matched design note exposed a dissociation in the query-first orbit-binding task: answer-only preparation can create counterfactual binding on trained entities and sometimes on held entity symbols, but later full next-token training can recover perfect trained-entity binding while held-symbol transfer collapses. The sharp mechanistic question was whether this loss is caused by tied input/output embedding rows for held entities being moved as output classes, or by specialization of the contextual computation itself.

The distinction matters for the broader data-efficient learning goal. If transfer loss is only tied-row damage, it is a lexical-parameter sharing artifact. If the contextual body itself is rewritten by non-answer prediction pressure, the phenomenon is closer to a general learning principle: finite experience can install a reusable selector, but later objectives can allocate credit to easier local predictive work that overwrites the state features that made the selector reusable beyond familiar symbols.

## Literature and implementation basis

The synthetic CLM uses tied embeddings: logits are `self.ln(h) @ self.tok.weight.T`. This is not incidental: tied input/output word vectors are a known efficiency-improving language-model design. Inan et al. motivate reuse of the input embedding matrix as the output classifier by constraining output logits to the embedding subspace and reducing parameters; they also stress that input and output words live in identical spaces rather than isolated class systems (`Knowledge/objects/papers/Tying-Word-Vectors-and-Word-Classifiers-A-Loss-Framework-for-Language-Mo--d2960e9f353c--7b87b808fccc/object.md`, `\cite{inan2016tying}`). Yu et al. analyze rare-token embedding gradients under the softmax and show that embeddings can be pushed by non-target terms; they decompose rare-token gradients into target-pull and non-target push terms and improve generation by gating rare-token gradients (`Knowledge/objects/papers/Rare-Tokens-Degenerate-All-Tokens-Improving-Neural-Text-Generation-via-A--c5bbc24660d8--9a9a89cd34ae/object.md`, `\cite{yu2021rare}`). These sources support the plausibility of a tied-row interference path, but they do not establish it in the budget matched design note task.

## Step019c: the tied-row gradient path exists

`scripts/revision_019c_tied_gradient_sanity.py` tested the exact gradient path on a query-first bound continuation batch with seed 100, epoch 401, n=128. The held entity rows were absent from both inputs and targets (`held_input_count=0`, `held_target_count=0`).

Key output: `notes/019c_tied_gradient_sanity.md`, JSON at `data/revision_019c_tied_gradient_sanity/results.json`.

- Tied model held-row data-gradient L2: `[1.47e-07, 8.96e-09]`.
- Manual non-target softmax gradient matched tied autograd with max error `1.07e-14`.
- Untied held input gradient L2: `[0.0, 0.0]`.
- Untied held output gradient matched the tied held-row gradient with max error `0.0`.
- With one AdamW step and no weight decay, tied held input rows moved `[0.001309, 0.000232]`, whereas untied held input rows did not move; with weight decay, untied held input row motion matched decay-only expectation.

This verifies the possible parameter path: when input and output rows are tied, a token absent from input and target experience can still be trained as a non-target output class, and that update changes the row later used as its input embedding. It does not show that this path causes the held-transfer collapse.

## Step019b: held-row drift is not the cause of held-transfer loss in this task

Step019b reran bound query-first answer-only preparation at the budget matched design note acquisition epochs and then continued with full next-token training under tied/untied and row-freezing interventions. It also built zero-training untied hybrids from final tied checkpoints.

Canonical files:

- Script: `scripts/revision_019b_embedding_role_decomposition.py`
- Data: `data/revision_019b_embedding_role_decomposition/results.json`
- Analysis: `data/revision_019b_embedding_role_decomposition/analysis.json`
- Notes: `notes/019b_embedding_role_decomposition_result.md`, `notes/019b_embedding_role_decomposition_analysis.md`
- Figure: `figures/revision_019b_embedding_role_decomposition.png`

The key invariants held: epoch-0 branch records exactly reproduced preparation metrics; untied `final_FF` hybrids reproduced final tied behavior exactly; frozen held rows had final held-input L2 equal to zero; `tied_carry_full` reproduced budget matched design note’s bound-preparation trajectory exactly on shared epochs.

Central endpoint results:

| branch | final trained top4 | final held top4 | final held B-swap | final held selectivity | held input L2 |
|---|---:|---:|---:|---:|---:|
| tied_carry_full | 1.000 | 0.389 | +3.119 | +0.188 | 0.691 |
| tied_carry_freeze_held_shared | 1.000 | 0.367 | +2.745 | +0.175 | 0.000 |
| tied_reset_full | 0.921 | 0.389 | +3.003 | +0.217 | 0.779 |
| tied_reset_freeze_held_shared | 0.922 | 0.370 | +2.633 | +0.191 | 0.000 |
| untied_reset_full | 0.921 | 0.395 | +3.047 | +0.217 | 0.112 |
| untied_reset_freeze_held_input | 0.921 | 0.378 | +2.938 | +0.207 | 0.000 |

The decisive seed remains seed100, where preparation had held top4 `0.953`, held B `+7.774`, held selectivity `+0.914`. After full continuation:

- `tied_carry_full`: held top4 `0.238`, held B `+1.035`, held selectivity `-0.029`, held input L2 `0.609`.
- `tied_carry_freeze_held_shared`: held top4 `0.238`, held B `+0.916`, held selectivity `-0.029`, held input L2 `0.000`.
- `untied_reset_full`: held top4 `0.312`, held B `+2.338`, held selectivity `+0.078`, held input L2 `0.112`.
- `untied_reset_freeze_held_input`: held top4 `0.285`, held B `+2.178`, held selectivity `+0.061`, held input L2 `0.000`.

Freezing the held rows did not rescue held transfer. Untying reduced held-input drift but did not preserve held transfer. The zero-training hybrid tests were still sharper: restoring only preparation held input rows into the final tied model (`final_PF`) did not rescue held transfer, and inserting final held input rows into the preparation network (`prep_FP`) did not damage preparation behavior. For seed100 tied carry:

- prep held top4 `0.953`
- final_FF held top4 `0.238`
- final_PF held top4 `0.238`
- final_FP held top4 `0.238`
- final_PP held top4 `0.238`
- prep network with final held input rows held top4 `0.965`

Thus held input-row drift is neither necessary nor sufficient for the held-symbol collapse observed here. The tied-row path exists, but it is not the load-bearing mechanism in this task.

## body objective ablation: non-answer training pressure on the contextual body is sufficient for early collapse

After Step019b ruled out held input-row drift, `scripts/component_ablation_after_switch.py` loaded the saved Step019b preparation checkpoints and ran short, 25-epoch continuations with selective trainable parameter sets. This tests where the destructive update acts before long full-objective reacquisition.

Canonical files:

- Script: `scripts/component_ablation_after_switch.py`
- Data: `data/component_ablation_after_switch/results.json`
- Note: `notes/component_ablation_after_switch.md`
- Added body-objective comparison: `data/body_objective_ablation/results.json`, `notes/body_objective_ablation.md`

Branch means after 25 epochs:

| branch | final train top4 | final held top4 | final held B | final held selectivity |
|---|---:|---:|---:|---:|
| tied_all_full | 0.346 | 0.315 | +0.206 | +0.058 |
| untied_all_full | 0.346 | 0.316 | +0.193 | +0.055 |
| untied_all_answer_only | 0.997 | 0.777 | +8.448 | +0.730 |
| untied_all_context_only | 0.276 | 0.255 | -0.024 | -0.002 |
| untied_body_only_full | 0.348 | 0.315 | +0.194 | +0.054 |
| untied_out_only_full | 0.997 | 0.749 | +5.670 | +0.668 |
| untied_input_only_full | 0.996 | 0.772 | +6.092 | +0.693 |

Per-seed body-only objective ablation confirms the objective source:

| branch | final train top4 | final held top4 | final held B | final held selectivity |
|---|---:|---:|---:|---:|
| untied_body_only_full | 0.348 | 0.315 | +0.194 | +0.054 |
| untied_body_only_answer_only | 0.999 | 0.773 | +8.503 | +0.729 |
| untied_body_only_context_only | 0.263 | 0.253 | -0.024 | -0.002 |

The seed100 one-epoch effect shows the same direction at the very start of switching:

- start: held top4 `0.953`, held B `+7.774`, held selectivity `+0.914`
- body-only full after 1 epoch: held top4 `0.648`, held B `+3.349`, held selectivity `+0.447`
- body-only context-only after 1 epoch: held top4 `0.641`, held B `+3.259`, held selectivity `+0.437`
- body-only answer-only after 1 epoch: held top4 `0.941`, held B `+8.478`, held selectivity `+0.932`
- output-only full after 25 epochs: held top4 `0.949`, held B `+7.691`, held selectivity `+0.914`
- input-only full after 25 epochs: held top4 `0.957`, held B `+7.957`, held selectivity `+0.916`

Therefore the early collapse is reproduced when only the contextual body/position/LN parameters are trainable under the full objective, with input and output tables fixed. It is also reproduced by context-only non-answer losses on the same body. It is not reproduced by output-head-only full training or input-table-only full training. Continuing answer-only training on the same body preserves or strengthens held transfer.

## Current controlled mechanism

The evidence now supports a more precise mechanism than the embedding-specialization hypothesis:

1. Answer-only query-first training can form a query-conditioned contextual marker that supports counterfactual binding on trained entities and, in some seeds, held entity symbols.
2. Full next-token training imposes strong non-answer losses on query and context positions. Those losses update the contextual stack in a way that rapidly suppresses the held-symbol selector/marker behavior, even when input and output token tables are fixed.
3. The same architecture can later reacquire trained-entity binding under the full objective, but reacquisition is specialized to symbols and contexts with ongoing answer supervision; broader held-symbol transfer remains weak.
4. Tied input/output rows create a real extra gradient path for absent or rare symbols, but in this orbit-binding task that path is not the main cause of held-transfer loss.

This is a controlled candidate for a general data-efficient learning principle: useful finite experience is retained and transferred only when later objectives preserve the internal state variables that carry reusable relations. Prediction pressure allocated to easier local statistics can overwrite those state variables while still improving or recovering familiar-distribution performance. The relevant unit is not simply the token, the example, or the aggregate loss, but the alignment between objective terms and the computation that must remain reusable.

## Remaining scientific work

The current evidence identifies the destructive site at the level of the contextual body and the destructive source at the level of non-answer context/query losses. It does not yet reveal the internal feature geometry of the query-conditioned marker or the exact parameter subset/layer where it is overwritten. The natural next work is to measure activation-level marking before and after the switch, layer by layer, and to test whether protecting the marker-bearing pathway or replaying sparse answer loss during context training preserves held transfer while allowing context prediction to improve.
