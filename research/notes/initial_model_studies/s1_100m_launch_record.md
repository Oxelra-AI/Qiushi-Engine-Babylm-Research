# protected 8x480 10m available coordinate — S1 100M launch rationale and command record

## Evidence supporting this launch

Matched 10M direct-checkpoint comparison:

- S1 12×384 10M: `data/s1_10m_available_coordinate.json`
- Protected 8×480 10M: `data/protected_8x480_10m_available_coordinate.json`
- Synthesis: `data/s1_vs_protected_10m_comparison.json`, `notes/s1_vs_protected_10m_comparison.md`

Key 10M deltas (S1 − protected 8×480):

- BLiMP −0.80
- Supplement −1.07
- Entity +0.29
- COMPS +0.39
- GlobalPIQA parallel +3.89
- GlobalPIQA nonparallel +5.00
- GlobalPIQA mean +4.445
- Reading mean +0.125

Interpretation: shape alone is not a 10M endpoint solution, but the +4.45 GlobalPIQA mean at matched exposure is a real target-cluster signal, comparable in magnitude to the protected 8×480 model's full 10M→100M GlobalPIQA gain. Entity remains weak and grammar/supplement are lower, so S1 is not sufficient by itself. But the signal is strong enough that the next clean execution is to train S1 to 100M before adding S2 curriculum, to see whether exposure recovers BLiMP/Supplement and preserves/improves GlobalPIQA.

## Run identity

Run id: `babylm_leadershape_s1_100M_aligned_micro128`

Scientific object: legal S1 architecture-shape isolation at full allowed exposure, official corpus only, baseline16k tokenizer, flat WWM. Not leader reproduction; exact `go76dof/Fineweb_simplification_pairs` data remains gated.

Training geometry:

- isolated fork: `training/scripts/babylm_masked_train_leadershape.py`
- DeBERTa-v2 12 layers, hidden 384, 12 heads, intermediate 1280
- p2c/c2p relative attention, position buckets 256
- baseline16k tokenizer, flat WWM mask_prob 0.15
- effective batch 256, micro_batch_size 128
- max exposure 100M over official 10M corpus (10 epochs), checkpoint every 10M
- lr_total_steps 2442, warmup_fraction 0.05, same seed discipline as protected runs

## Evaluation after completion

Evaluate direct checkpoints, starting with `hf_model/chck_100M`, on BLiMP, Supplement, Entity, COMPS, GlobalPIQA parallel/nonparallel, Reading, full EWoK if ready, plus SuperGLUE/AoA if the route remains promising. Compare to protected 8×480 100M and visible leader gaps. If S1 100M preserves a GlobalPIQA advantage and recovers grammar/supplement enough, build S2 curriculum (length schedule + WWM→token). If not, prioritize curriculum/tokenizer/data reconstruction or GPT-BERT/MNTP hybrid.
