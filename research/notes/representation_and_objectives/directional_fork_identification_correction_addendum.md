# Directional Forks and Reciprocal-Learning Identification

Status: corrected fork checks passed and compact-only training began; final interaction and extractive controls remained pending.

Forward-then-reverse minus forward-then-forward cannot identify reciprocal learning, because reverse prediction alone could explain the difference. The corrected metric was

`I_m(data) = (m_FR + m_RF)/2 - (m_FF + m_RR)/2`.

Evidence for compact specificity would additionally subtract this interaction on copy-matched semantic or random extractive data. The initial FR-versus-FF design was superseded.

FF/FR shared an exact forward first-epoch state; RR/RF shared an exact reverse first-epoch state. Forks preserved model, optimizer, scheduler and random-generator states. A continuation defect that moved CPU random state onto the accelerator was repaired by loading fork state on CPU and moving optimizer moments separately. Pair-aware chunking preserved pair boundaries and avoided truncating tokenized pairs longer than 256 tokens.

The check recorded 13 log entries for each prefix, exact prefix sharing, equal branch start updates and divergent final branch hashes. All four continuations had 26 updates. These checks establish a valid intervention construction, not a positive interaction.

The compact screen used a 10M-word pass, shared filler, a neutral tokenizer, GPT2 8x480, model seed 43, training seed 43022, learning rate 0.0006 and batch 256. A meaningful official-compatible interaction, especially in relational EWoK domains, was required before extending to extractive controls. A null compact interaction would not support reciprocal compact learning in this causal-model coordinate.
