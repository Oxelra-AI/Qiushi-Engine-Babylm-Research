# pvdm 80m ewok fourcell reader — staged PVDM 80M readout plan

This wrapper evaluates only explicit `hf_model/chck_80M` checkpoints from the staged treatment/control runs. The fixed readouts are Supplement, Entity, GlobalPIQA all-option margins, and EWoK four-cell. It is prepared before the training tasks finish so no new metric is invented after seeing results.

Preflight: `experiments/archive/representation_and_objectives/data/pvdm_80m_readouts/preflight.json`
