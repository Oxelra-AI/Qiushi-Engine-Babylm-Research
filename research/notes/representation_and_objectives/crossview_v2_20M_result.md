# crossview v2 20M result corrected 20M MLM cross-view screen result

## Question

The earlier analysis/221 causal-separation question was whether correct source--rewrite semantic correspondence makes cross-boundary attention useful for intrinsically source-absent rewrite content, beyond generic cross-boundary context and beyond copied-token retrieval. This matters because adjusted source use residual ruled against a simple enhanced ordered copied-token retrieval account in a frozen source-use probe, leaving source-absent semantic supervision as a live but unproved explanation for the compact-view benefit.

## Design actually run

The earlier cross-view design was revised before training. The crossview v2 20M result repaired design used:

- data v2: `experiments/archive/representation_and_objectives/data/crossview_data_v2/crossview_consolidated_v2.jsonl`, SHA256 `1123d5ab1b0127618d9f9fd72accf39c078ecef047ecf9213d0b9622dea7ccff`;
- identical untrained DeBERTa-v2 8x480 init for every arm: `experiments/archive/representation_and_objectives/training/runs/crossview_identical_init_seed43022/hf_model_init`, `model.safetensors` SHA256 `ddc8532a6eb81d2eed0e7671fdd06d68dfeeac08f96f4d7d8e057391b6ed5e68`;
- intrinsic rewrite-origin labels carried with the rewrite identity through derangement;
- deterministic identity-attached WWM masks: source masks keyed by source identity, rewrite masks keyed by rewrite identity, filler masks keyed by filler identity;
- common legal16k tokenizer from `experiments/archive/representation_and_objectives/training/runs/gc_compact_repeat_reinvest_16k_seed43022_r2/hf_model/chck_1M`;
- 20M words per arm, two exact epochs, 578 optimizer steps, LR schedule geometry `lr_total_steps=2529`, warmup 126.

Preflight `experiments/archive/representation_and_objectives/data/crossview_v2_preflight/crossview_v2_preflight.json` passed all identity checks: 12,155 common rewrite identities, zero own/wrong stratum mismatches, zero rewrite-mask mismatches, zero visible/blocked target mismatches, zero source-mask mismatches. The custom full-visible DeBERTa forward matched stock exactly (max delta 0.0); blocked mask changed logits by 0.2691181004 and gradients were finite.

## Runs

All four arms completed successfully:

| Arm | Output root | Loss last | rw_copied mean | rw_abs_content mean | rw_abs_other mean |
|---|---|---:|---:|---:|---:|
| own_visible | `experiments/archive/representation_and_objectives/training/runs/crossview_own_visible_20M` | 3.55358 | 6.28766 | 7.27849 | 5.55748 |
| own_blocked | `experiments/archive/representation_and_objectives/training/runs/crossview_own_blocked_20M` | 3.59419 | 6.37888 | 7.32224 | 5.63700 |
| wrong_visible | `experiments/archive/representation_and_objectives/training/runs/crossview_wrong_visible_20M` | 3.60661 | 6.35177 | 7.29906 | 5.56618 |
| wrong_blocked | `experiments/archive/representation_and_objectives/training/runs/crossview_wrong_blocked_20M` | 3.58044 | 6.36318 | 7.30453 | 5.63262 |

Each arm logged identical stratum counts:

- filler targets: 4,134,298;
- source targets: 110,478;
- rewrite copied targets: 65,723;
- rewrite absent-content targets: 7,649;
- rewrite absent-other targets: 2,537.

These equal counts matter: the interaction is not a target-mixture artifact after crossview v2 20M result repair.

## Interaction readout

Analyzer: `experiments/archive/representation_and_objectives/scripts/crossview_interaction_analyzer.py`

Output: `experiments/archive/representation_and_objectives/data/crossview_interaction/crossview_partner_visibility_interaction.json`

Definition, with lower loss better:

`I_partner(X) = (own_visible_X - own_blocked_X) - (wrong_visible_X - wrong_blocked_X)`.

A negative value means correct-partner visibility helps stratum X beyond wrong-partner generic visibility.

### Exposure-weighted over all 20M words

| Stratum | own visibility effect | wrong visibility effect | I_partner |
|---|---:|---:|---:|
| filler | +0.012282 | +0.003964 | +0.008319 |
| source | -0.019142 | +0.016904 | -0.036046 |
| rw_copied | -0.091228 | -0.011401 | **-0.079827** |
| rw_abs_content | -0.043753 | -0.005475 | **-0.038278** |
| rw_abs_other | -0.079517 | -0.066443 | -0.013074 |

### End-window (second epoch, cum >= 10M)

| Stratum | own visibility effect | wrong visibility effect | I_partner |
|---|---:|---:|---:|
| filler | +0.000721 | +0.003777 | -0.003057 |
| source | -0.048092 | +0.024125 | -0.072217 |
| rw_copied | -0.138941 | -0.003917 | **-0.135024** |
| rw_abs_content | -0.071499 | -0.023284 | **-0.048215** |
| rw_abs_other | -0.076673 | -0.021383 | -0.055290 |

## Scientific interpretation

The corrected four-arm study finds a real but modest correct-partner visibility effect on intrinsically source-absent content:

- all-run `rw_abs_content` I_partner = -0.038278 nats;
- end-window `rw_abs_content` I_partner = -0.048215 nats.

However, the copied-token interaction is substantially larger:

- all-run `rw_copied` I_partner = -0.079827 nats;
- end-window `rw_copied` I_partner = -0.135024 nats.

Thus the effect does **not** concentrate on source-absent semantic content. The strongest cross-boundary partner-specific signal remains copied tokens and source/rewrite lexical coherence, not the target class that would support the source-absent semantic-supervision explanation of the compact-view downstream gain. This is consistent with adjusted source use residual's warning that ordered/copy-like source use and tail/coverage effects remain serious alternatives.

The result does not prove that source-absent content never matters in compact-view learning: this is one 20M, one-seed, loss-level screen with only 7,649 masked absent-content targets. It does, however, fail the predeclared condition for continuing this exact route into 100M or endpoint evaluation. Running official endpoint evaluation on these four 20M arms would be score fishing unless a stronger source-absent-specific signal is first established.

## Consequence for the mechanism route

Closed at loss-screen level: **correct cross-view attention to own source as a dominant source-absent semantic-supervision mechanism**.

Preserved weaker fact: correct partner visibility helps source-absent content a little, but less than it helps copied targets. This can be used as a constraint on future accounts: any compact-view principle must explain why large downstream gains arise even though direct own-source reachability mostly improves literal/copied channels in this controlled 20M MLM screen.

Recommended next scientific move: return to mechanism formulation rather than extend this run. Plausible next directions are (i) coverage-matched extractive/tail-content controls, because copied/lexical channels dominate this screen; (ii) a representation or data-layout account in which compact views shape the denoising distribution without requiring direct source-to-source-absent target prediction; or (iii) a genuinely target-selective loss intervention only if another analysis finds a stronger source-absent signal. Do not launch 100M or endpoint evaluation from this four-arm screen as if the predeclared source-absent route had survived.
