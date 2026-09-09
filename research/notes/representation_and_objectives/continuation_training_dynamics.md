# continuation training dynamics — 70M→80M continuation training dynamics

This compares only training traces for the standard WWM staged branch and the PVDM treatment/control branches. It does not replace the downstream readout.


## Shared accounting

- same segment: True; same hashes: True; same schedule/exposure: True

- selected predicted tokens total: {'standard': 2162049, 'treatment': 2147486, 'control': 2147486}

- PVDM selected-token delta vs standard: {'treatment': -14563, 'control': -14563}


## Loss and mask-rate traces

- `standard`: loss first/last=2.563990960915346/2.6666369943320576; loss mean=2.7841661734406484; effective mask-rate mean=0.15005554799999998; masked-token mean=8648.196

- `treatment`: loss first/last=2.611742846831649/2.793831899309365; loss mean=2.805715700671479; effective mask-rate mean=0.149042116; masked-token mean=8589.944

- `control`: loss first/last=2.6961954851673826/2.848363915997498; loss mean=2.8701104122739296; effective mask-rate mean=0.149042116; masked-token mean=8589.944


## Stepwise deltas

- `standard_minus_treatment`: loss_delta mean=-0.02154952723083035, median=-0.018403373409212165, first={'step': 1, 'loss_delta': -0.04775188591630286, 'masked_tokens_delta': 224}, last={'step': 250, 'loss_delta': -0.12719490497730757, 'masked_tokens_delta': 105}; same words=True, same exposure=True, same LR=True

- `standard_minus_control`: loss_delta mean=-0.08594423883328167, median=-0.08640550036873518, first={'step': 1, 'loss_delta': -0.13220452425203666, 'masked_tokens_delta': 224}, last={'step': 250, 'loss_delta': -0.18172692166544024, 'masked_tokens_delta': 105}; same words=True, same exposure=True, same LR=True

- `treatment_minus_control`: loss_delta mean=-0.06439471160245132, median=-0.06353630476602556, first={'step': 1, 'loss_delta': -0.0844526383357338, 'masked_tokens_delta': 0}, last={'step': 250, 'loss_delta': -0.05453201668813268, 'masked_tokens_delta': 0}; same words=True, same exposure=True, same LR=True


## Interpretation before downstream scores

- The three branches share the exact tail rows, word exposure, tokenizer hash, and schedule. Standard WWM is therefore a valid control for staged replay plus optimizer reset/microbatch execution.

- Standard WWM is not target-identical to PVDM treatment/control, by design: it retains ordinary whole-word target sampling. Its selected-token total is only about 0.68% higher than the PVDM arms, so a large downstream difference is unlikely to be explained by total mask mass alone, but target identity and relation-anchor swapping remain the intended contrast.

- Standard WWM has lower loss than both PVDM arms on this segment. This cannot be interpreted as better downstream relation learning; it mainly indicates the modified dependent-target distribution is harder than ordinary WWM. The fixed readout decides causal meaning.


Files: `experiments/archive/representation_and_objectives/data/continuation_training_dynamics/continuation_training_dynamics_compare.json`
