# layerwise exchange readout and gradient anatomy — layerwise exchange readouts before spending another endpoint

## Research situation

The shared-tokenizer legal role-switch and role-fixed 80M screens from earlier analysis were still managed asynchronously during this step. I did not query their status or start another training run. The purpose here was to use existing checkpoints to understand whether a stronger next mechanism should be a readout of already-formed context-conditioned exchange information, or a training mechanism that creates such information.

companion analysis's new group finding (message 158) independently reinforced that recent mature interventions rotate decision surfaces rather than adding broad missing competence: its scale1.75 adapter helps some BLiMP/Entity subfamilies while damaging NPI/island/EWoK material/quantitative surfaces; cross-route score vectors are not one scalar improvement axis. This makes a broad SOTA route unlikely unless it repairs specific relational/conditional substructures while preserving the gains.

## Files produced

- Layerwise decoded-M script: `experiments/archive/representation_and_objectives/scripts/layerwise_packet_M_decomposition.py`
- Subset layerwise result: `experiments/archive/representation_and_objectives/data/layerwise_packet_M_decomposition/layerwise_packet_M_summary_score.json`
- Full-suite layerwise result: `experiments/archive/representation_and_objectives/data/layerwise_packet_M_decomposition_fullall/layerwise_packet_M_summary_score.json`
- Exchange-gradient script: `experiments/archive/representation_and_objectives/scripts/exchange_gradient_anatomy.py`
- Exchange-gradient result: `experiments/archive/representation_and_objectives/data/exchange_gradient_anatomy/exchange_gradient_anatomy_summary_score.json`
- Frozen diagonal-bilinear readout script: `experiments/archive/representation_and_objectives/scripts/frozen_bilinear_exchange_probe.py`
- Frozen diagonal-bilinear result: `experiments/archive/representation_and_objectives/data/frozen_bilinear_exchange_probe/frozen_bilinear_exchange_probe_summary_score.json`

All three tests train no endpoint. They use existing legal40k checkpoints and the role switch packet screen synthesis packet scoring suite as a mechanistic object.

## 1. Layerwise decoded four-cell M

The script masks the candidate target exactly as the frozen exchange scoring analysis/role switch exposure geometry and route implication packet scorers do, but decodes every hidden-state layer through the trained MLM head. On the full 1,560-pair suite including temporal and container families, final-layer summaries were:

| checkpoint | final M mean | final both-correct | M positive fraction |
|---|---:|---:|---:|
| fixed-256 anchor 80M | 1.1709 | 0.2981 | 0.6179 |
| fixed-256 anchor 100M | 1.0765 | 0.2673 | 0.5929 |
| reheat 100M | 1.3771 | 0.2974 | 0.6173 |

Layer growth is late and mostly monotone for M. At anchor 80M, M progresses approximately: layer0 -0.016, layer5 0.041, layer6 0.202, layer7 0.650, layer8 1.171. At reheat 100M: layer0 -0.019, layer5 0.117, layer6 0.328, layer7 0.798, layer8 1.377. There is no hidden layer decoded by the existing head that cleanly exceeds the final layer as a strong exchange readout. Layer 7 sometimes has slightly different accuracy balance, but not a large missing signal.

The family map is more important than the aggregate:

| family | anchor80 final M / both | anchor100 final M / both | reheat final M / both | interpretation |
|---|---:|---:|---:|---|
| transfer | 5.573 / 0.600 | 5.140 / 0.537 | 6.867 / 0.642 | already relatively learnable |
| comparative | 4.055 / 0.475 | 4.178 / 0.438 | 4.106 / 0.429 | large margin but not robust both-direction accuracy |
| state_change | 1.747 / 0.108 | 2.229 / 0.117 | 2.097 / 0.196 | positive M but highly one-sided |
| spatial | 0.544 / 0.203 | 0.126 / 0.156 | 0.611 / 0.172 | weak exchange despite being central to role-switch goal |
| temporal | -1.587 / 0.327 | -1.080 / 0.303 | -1.235 / 0.310 | negative M; remains special/asymmetric |
| container | -3.460 / 0.056 | -4.519 / 0.044 | -4.655 / 0.028 | withheld family remains collapsed |

This explains why packet self-scores can look partly positive while natural transfer remains weak: the current synthetic object is not uniformly measuring one transferable operation. Transfer/comparative behave very differently from spatial, state_change, temporal, and container.

For a softplus exchange loss `softplus(1-M)` over all families, the final-layer gradient weight would be dominated by weak or negative families. At reheat 100M, approximate weight shares are spatial 0.265, temporal 0.242, container 0.209, comparative 0.159, state_change 0.092, transfer 0.033. Any future exchange objective must therefore avoid letting the known asymmetric temporal/container cases and weak template artifacts dominate; temporal and container should remain held-out until their grammar and scoring asymmetries are repaired.

## 2. Centered exchange gradient versus ordinary CE

On a smaller 54-pair stratified subset, I compared gradients from:

- centered exchange loss `mean softplus(1-M)`;
- ordinary correct AB/BA CE;
- role-fixed CE.

Results:

| checkpoint | exchange loss | correct CE | fixed CE | cos(exchange, correct CE) | cos(exchange, fixed CE) | cos(correct CE, fixed CE) |
|---|---:|---:|---:|---:|---:|---:|
| anchor 80M | 1.5758 | 4.7346 | 5.5849 | 0.0679 | -0.0150 | 0.9397 |
| reheat 100M | 1.6558 | 4.3870 | 5.2169 | 0.0391 | -0.0415 | 0.9200 |

This is a real credit-assignment difference: ordinary correct CE and role-fixed CE are almost the same gradient direction, while centered exchange is nearly orthogonal to both. The exchange gradient is not an output-bias trick: MLM-head bias norm is only about 3e-6 to 4e-6 of total norm-squared, head weights about 0.002–0.003, while embeddings plus encoder layers carry almost all gradient. The encoder-layer norm-squared fractions for anchor80 exchange are roughly layers 0..7 = 0.052, 0.067, 0.061, 0.089, 0.117, 0.146, 0.144, 0.121.

This supports the scientific premise that a centered exchange objective is materially different from the ordinary packet CE route. It does not by itself justify a long endpoint, because it says the gradient is different, not that it generalizes to natural BabyLM surfaces.

## 3. Frozen diagonal-bilinear exchange readout

To test the protected-readout premise directly, I froze the base model and trained a tiny 480-dimensional diagonal-bilinear vector per layer:

`score(C; alt0, alt1) = ((E_alt0 - E_alt1) * h_masked_context) @ w`

Training used only train-style pairs from the train families (spatial, transfer, comparative, state_change). It then evaluated held-out templates and the withheld container/temporal families. After filtering multi-token candidates, 1,437 pairs remained.

The result is strongly negative for "the information is already present and only unread by the head." The probe fits train examples increasingly well with depth, but it does not transfer:

| checkpoint | best layer | train-family train context acc | held-out train-family context acc | held-out train-family pair both | container context acc | container pair both |
|---|---:|---:|---:|---:|---:|---:|
| anchor 20M | 8 | 0.815 | 0.512 | 0.067 | 0.503 | 0.022 |
| anchor 80M | 7 | 0.832 | 0.512 | 0.089 | 0.483 | 0.050 |
| anchor 100M | 7 | 0.808 | 0.528 | 0.074 | 0.503 | 0.061 |
| reheat 100M | 7/8 | 0.834–0.878 | 0.512/0.495 | 0.074 | 0.500/0.458 | 0.050/0.033 |

Temporal context accuracy also stays near chance and pair-both remains near zero. Thus a small antisymmetric readout cannot simply uncover a robust exchange algebra already latent in these frozen states. If Route A becomes the next architecture route, it should be treated as a representation-forming mechanism trained by an interaction objective, not as a detached readout of an already solved hidden variable.

## Scientific update

These no-training tests sharpen the route choice:

1. The current backbone has some packet-like exchange behavior, mostly in late layers and final logits, but it is family-specific. Transfer/comparative are much easier than spatial/state_change; container and temporal are negative.
2. The centered four-cell exchange loss gives a genuinely different gradient from ordinary CE/role-fixed packet CE and largely removes direct output-bias learning.
3. A frozen small bilinear readout fails to transfer from train templates to held-out templates/families, so the missing mechanism is not merely an available internal feature waiting for a side head.
4. If the running earlier analysis legal sparse replacement screen is flat on natural hard surfaces, the next serious route should not be more packet placement or packet dose renaming. It should be a representation-forming interaction mechanism: centered exchange credit, ordered-role/equivariant state, or protected context-difference side path, with temporal/container repaired or withheld and with natural EWoK/GlobalPIQA readouts as the decisive evidence.
5. The immediate next action remains to read the delivered earlier analysis role_switch/role_fixed checkpoints and apply the natural hard-surface contrast together with synthetic acquisition readout. The layerwise exchange readout and gradient anatomy results mainly prepare the route after that result: detached readout alone is too weak, but centered exchange credit remains scientifically distinct enough to deserve a minimal construction if the sparse WWM recipe fails.
