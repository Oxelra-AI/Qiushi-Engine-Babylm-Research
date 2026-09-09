# causal transfer result synthesis: Causal GPT compact-view vs repeat architecture-transfer result

## What was tested

Cross-architecture transfer of the strongest validated data mechanism (compact
semantic same-window views vs exact source repetition), moved from the bidirectional
DeBERTa MLM coordinate into a decoder-only GPT2 causal-LM coordinate.

- Both arms: GPT2 8x480, 30,156,480 params, next-token objective.
- Neutral 16k BPE tokenizer trained ONLY on shared common filler (SHA `e6723383...`),
  arm-neutral by construction.
- Compact pool SHA `fa2216ce...`, repeat pool SHA `e40b3a89...`, each exactly 10M legal
  words, 73,890 rows, 12,155 pairs, balanced source->view / view->source order (seed 171043).
- Exactly 100M legal charged words, 2230 steps, 10 epochs each. Dense 2M checkpoints.
- Only implementation asymmetry: compact has +32,256 active tokens/epoch (+0.2216%).
- Repeat had slightly LOWER training loss at every selected checkpoint (final 3.28013 vs
  3.292451). Training loss is corpus-fit only, not competence.
- Selected official-compatible cheap7 (report-parsed, --backend causal) at
  chck_20M/50M/70M/82M/90M/100M.

## Selected cheap7 (official-compatible)

| endpoint | compact | repeat | Δcheap7 | Δcheap6 noGP | Δcheap5 noGP/Read | Δrel/state | Δvolatile(GP+Read) | +cols |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| chck_20M | 38.0814 | 38.1943 | -0.1129 | +0.2825 | +0.2360 | +0.3450 | -0.9850 | 5 |
| chck_50M | 40.1493 | 40.2121 | -0.0629 | -0.2375 | -0.3180 | +0.6500 | +0.5750 | 5 |
| chck_70M | 39.9036 | 40.5964 | -0.6929 | -0.5633 | -0.5940 | -1.2250 | -0.9400 | 2 |
| chck_82M | 40.8214 | 40.2714 | +0.5500 | -0.1033 | -0.0580 | -0.2300 | +2.0700 | 4 |
| chck_90M | 40.6757 | 40.3471 | +0.3286 | -0.1142 | -0.0760 | -0.7250 | +1.3400 | 3 |
| chck_100M | 40.7114 | 40.2550 | +0.4564 | -0.1317 | -0.0980 | -0.7300 | +1.8425 | 4 |

Mean Δcheap7 = +0.0777 (3/6 positive; 2/6 broad).
Mean Δcheap6(no GlobalPIQA) = **-0.1446**.
Mean Δcheap5(no GlobalPIQA/Reading) = **-0.1513**.
Mean Δrelation_state(EWoK+Entity) = **-0.3192** (negative at 4/6 checkpoints).

## Scientific conclusion (decisive, negative for architecture-general transfer)

The compact-view mechanism does **not** transfer as a broad, family-general data-efficient
principle into the decoder-only causal coordinate.

1. The only positive aggregate signal (mean Δcheap7 +0.078) vanishes and turns negative
   once the small volatile GlobalPIQA column is removed (Δcheap6noGP -0.145, Δcheap5 -0.151).
2. At every positive-cheap7 endpoint (82M/90M/100M) the advantage is carried by
   GlobalPIQA+Reading (Δvolatile +2.07 / +1.34 / +1.84) while syntax/surface and
   relation/state are flat-to-negative. Max single-positive-column share 0.80-0.88.
3. The relation/state families (EWoK+Entity) that the DeBERTa compact-view mechanism
   was supposed to strengthen are on average **worse** for compact (-0.319), opposite to
   the DeBERTa evidence.
4. The compact arm even carries a small +0.2216% active-token advantage and still does
   not achieve broad wins.

## What this establishes and does not establish

- Establishes: the same-window compact semantic-view benefit observed under DeBERTa MLM
  (corpus geometry interpretation and pending clean vector: +1.35 mean7 at 80M, compact > independent breadth) is NOT reproduced under
  a decoder-only causal-LM objective+attention. The benefit appears coupled to the
  bidirectional MLM coordinate / inductive bias, not to the data organization alone.
- Does NOT erase the DeBERTa compact-view/reinvestment causal evidence, which remains the
  strongest validated corpus mechanism in this study's own coordinate.
- Does NOT prove the mechanism is intrinsically DeBERTa-specific in all decoder settings;
  this bounds one causal implementation (GPT2 8x480, this tokenizer/order/budget).

## Route implication

Architecture-general transfer of compact same-window views is not supported by this test.
The transferable-principle search should not assume the compact-view data benefit carries
into causal LMs as-is. The next generality question worth resolving is the DeBERTa
seed/mask-robustness and adapter-scale behavior of the 82M peak (common-grid scoring
launched: reference and scale1.25 seed43022;
scale1.75 seed43122 pending).

Artifacts:
- `data/selected_causal_eval_compact/selected_causal_trajectory.json`
- `data/selected_causal_eval_repeat/selected_causal_trajectory.json`
- `data/causal_compact_repeat_selected_contrast/causal_compact_repeat_selected_contrast.{json,md}`
- accounting: `data/causal_training_accounting_audit/`


## Packet topology confirmation (causal transfer result synthesis audit)

`data/causal_packet_topology_audit/` confirms the two causal pools are
matched in construction:
- Both 73,890 rows, exactly 10,000,000 legal words.
- Both 12,236 packed pair rows (1,656,800 pair-packed words), identical pair positions.
- Per-row word field delta compact-minus-repeat = 0.0 (word budget matched row-by-row).
- 61,736 rows have identical text (shared filler); 12,154 differ only in the paired
  packet text (compact semantic view vs exact source repeat).
- The only field-level differences at matched indices are the source LABEL
  (`causal_compact_view` vs `causal_repeat`) at the 12,155 paired rows, which is by design.

Therefore the negative causal transfer result is a genuine content-difference result
(compact semantic view vs exact repetition), not an artifact of row order, legal-word
budget, or packet topology. The only residual implementation asymmetry is the recorded
+0.2216% compact active-token count and a tiny median +0.08 sentence-count / +1.33 char
difference in the paired rows, all far too small to explain a broad transfer signal —
and compact does not show one.

Integrity of the selected causal payloads is verified in
`data/causal_selected_integrity_check/` (both arms OK: 8 cheap tasks
each, cheap7 recomputed to <1e-9 error, model SHAs recorded).
