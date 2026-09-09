# muon wdmatched causal plan and pvdm challenge — Muon matched-decay causal test and PVDM challenge

## Research continuity

The active substrate remains the protected legal compact-view reinvest corpus. Prior work established that compact semantic second views plus reinvested source diversity produce a large late-emerging legal-tokenizer treatment effect, while data allocation, tokenizer variants, word-mean credit, minfreq support, innovation masking, depth, interaction objectives, tail rephase, expanded FineWeb allocation, checkpoint recombination, and RTD have not solved the remaining legal-coordinate gap.

rtd closure and muon route decision moved the route to hidden-matrix update geometry because the actual spatial repair route status legal system showed very low-rank replayed AdamW hidden-matrix updates. earlier analysis implemented the first real test: Muon on 48 attention/FFN matrices, AdamW on embeddings, relative-position channels, MLM head, norms, biases, and all vectors, with data/init/WWM/schedule held fixed. The 20M screen was strongly positive at LR 0.008: cheap7 40.9271 versus spatial repair route status 39.6636, delta +1.2636. LR 0.012 was much weaker at +0.4543.

The essential confound is that earlier analysis did not isolate orthogonalized update geometry. It also passed the same numeric weight_decay=0.01 to the Muon group. Since Muon LR was 0.008 or 0.012 while spatial repair route status AdamW LR was 0.001, decoupled shrinkage p <- p * (1 - lr * wd) was about 8x or 12x stronger on hidden matrices. Therefore the LR 0.008 versus 0.012 difference combines update magnitude, norm growth, and shrinkage, and cannot be interpreted from score alone.

## Verified optimizer formula and geometry

The wrapper code applies decoupled decay inside the Muon group as `p.mul_(1.0 - lr * weight_decay)` before `p.add_(update, alpha=-lr)`. The HuggingFace scheduler changes each param group's current lr by the same multiplier, so setting Muon weight decay to `0.01 * 0.001 / 0.008 = 0.00125` matches spatial repair route status per-step shrinkage at every scheduler fraction.

I ran `scripts/muon_weight_geometry.py`, producing `data/muon_weight_geometry/muon_weight_geometry.{json,md}`. The spectra support the caution:

| model | Frobenius/init | stable-rank Δ/init | entropy-rank Δ/init | top8-energy Δ/init |
|---|---:|---:|---:|---:|
| spatial repair route status 20M | 1.2179 | -109.01 | -60.31 | +0.0798 |
| Muon 0.008 wd0.01 20M | 1.5754 | -32.56 | -9.83 | +0.0075 |
| Muon 0.012 wd0.01 20M | 1.9860 | -32.17 | -7.96 | +0.0053 |

Absolute hidden-matrix stable rank is 43.96 for spatial repair route status, 120.41 for Muon 0.008, and 120.81 for Muon 0.012. Top-8 energy is 0.133 for spatial repair route status, 0.061 for Muon 0.008, and 0.059 for Muon 0.012. Thus Muon did broaden the learned hidden matrices, but both Muon LRs broadened similarly while scoring very differently. The differentiating geometry at 20M is Frobenius growth and step magnitude, not spectral broadening alone. A mature arm must therefore be chosen by the matched-decay causal comparison, not by the spectra.

## muon wdmatched causal plan and pvdm challenge causal arm

New trainer: `scripts/muon_wdmatched_trainer.py`.

The smoke run at 999,918 words verified:

- exact 48 Muon matrices and 122 AdamW tensors, 34,467,424 unique parameters;
- Muon LR 0.008, Muon weight decay 0.00125;
- nominal Muon shrink per base step = 1e-05;
- spatial repair route status AdamW shrink per base step = 1e-05;
- shrink ratio vs spatial repair route status = 1.0;
- exact first loss 9.837543487548828;
- checkpoint loads as AutoModelForMaskedLM.

Caveat: the 1M smoke used its own 26-step short schedule only as a numerical and file-system check. The actual 20M causal run uses `--lr_total_steps 2529`, matching earlier analysis and the spatial repair route status-compatible 100M horizon.

One causal comparison was launched, with run dir `training/runs/muon_lr008_wd00125_seed43022_20M/`. It keeps all earlier analysis/spatial repair route status coordinates fixed except Muon hidden-matrix weight decay:

- corpus: `cleanqwen_fineweb_compact_view_reinvest_100M.jsonl`;
- tokenizer: spatial repair route status legal 16k tokenizer;
- DeBERTa-v2 8x480, seed 43, extra init seed 43022, train RNG 43023;
- batch size 256, seq length 256, WWM fixed p=0.15;
- AdamW interface lr 0.001 wd 0.01; Muon hidden lr 0.008 wd 0.00125; warmup 0.06; lr_total_steps 2529;
- checkpoints 5M/10M/15M/20M.

The decisive interpretation is:

- If matched-decay LR0.008 remains broadly positive near the earlier analysis +1.26 cheap7 scale, the principle is hidden-update orthogonalization/broadening rather than excess shrinkage, and it earns one mature readout around 70M or 80M.
- If matched-decay loses most of the gain, the bundled strong hidden-matrix shrinkage or norm control was load-bearing; do not launch 70-80M until the mechanism is repaired around norm/step-size control.
- If matched-decay improves EWoK/Reading while preserving most broad gains, it suggests earlier analysis's damage partly came from excess shrinkage, and the matched-decay arm is the preferred mature candidate.

Prepared evaluator: `scripts/eval_muon_wdmatched_20m.py`. It evaluates all seven cheap columns for the muon wdmatched causal plan and pvdm challenge arm and consolidates it with spatial repair route status and the earlier analysis arms.

## Challenge to PVDM evidence

The PVDM design resolved compliance by excluding spaCy-derived labels and replacing them with deterministic in-budget labels. The real-batch invariant check is strong: treatment and control share dependent target labels, realized mask mass, and replacement actions while differing in true-pivot visibility.

The earlier analysis readout is negative for the visible-pivot dependent-prediction treatment: treatment had lower local training loss than control at all 250 steps but was worse on GlobalPIQA parallel/nonparallel, GP hard52 rank/margins, Supplement, and EWoK; only Entity improved. Control itself was worse than the uninterrupted reference on EWoK, so part of the damage may come from staged optimizer reset/microbatch replay or target redistribution common to both arms.

The substantive methodological limitation is that the decomposition is not sufficient for a new relational objective until the identical-segment legacy-WWM branch separates shared staged-continuation damage from target-redistribution damage. Any repaired PVDM should increase relation-pivot prediction pressure or relation contrast rather than simply making the pivot visible, because the PVDM evidence suggests hiding the relation pivot may be the useful pressure. A further hypothesis is whether PVDM and Muon can be unified as an update-geometry principle: PVDM changes target-gradient geometry and damaged relation surfaces; Muon changes hidden matrix update geometry and may improve broad reuse if the matched-decay result survives.
