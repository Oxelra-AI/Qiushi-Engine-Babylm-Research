# adapter 20M closure — Function-preserving residual-adapter experiment

## Scientific rationale

Cross-route evidence (related experiments) shows that interventions altering the shared
prediction target or shared optimizer path repeatedly redistribute competence
across official evaluation surfaces rather than expanding total capacity:

- Muon broadened hidden spectra → +1.36 cheap7 at 20M, but −0.16 at 80M
- LAMB with trust-ratio saturation → −2.50 at 20M 
- Word-mean MLM, innovation masking, RTD, minfreq50 → local target success
  but broad competence damage

The protected corpus mechanism (compact semantic second views + reinvested source
diversity) produces a +1.35 mean7 gain at 80M under the legal tokenizer, but the
remaining gap to 41.8 requires the learning system to convert that signal into
broader mature competence, not just redistribute it.

## Hypothesis

A function-preserving residual branch after each DeBERTa encoder layer can absorb
the compact-view signal into additional nonlinear capacity without rewriting the
baseline (spatial repair route status) trajectory.  At initialization, the augmented model is exactly
the stock model (verified: all diffs = 0.0).  The branch recruits gradually as
W_up departs from zero.  If the branch is recruited and improves broad competence
without Entity/relation/Reading damage, the approach adds genuinely new capacity
rather than rotating existing capacity.

## Architecture

    h_out = h_layer + Adapter(h_layer)
    Adapter(h) = W_up( GELU( W_down( LayerNorm(h) ) ) )

- W_up and b_up initialized to zero → adapter output = 0 at init
- W_down initialized by PyTorch default (kaiming_uniform_)
- LayerNorm initialized to weight=1, bias=0
- One adapter per DeBERTa-v2 encoder layer (8 adapters total)
- Adapter output factors (W_up, b_up) excluded from weight decay
- All other parameters use spatial repair route status weight_decay=0.01

### Parameter counts

| bottleneck | adapter params | stock params   | total     | overhead |
|-----------|---------------|----------------|-----------|----------|
| 64        | 503,552       | 34,467,424     | 34,970,976| +1.46%   |
| 128       | 995,584       | 34,467,424     | 35,463,008| +2.89%   |

## Mechanical verification (adapter 20M closure)

All tests passed on GPU with the exact spatial repair route status RNG:

| Test | Result |
|------|--------|
| Stock parameter names match | ✓ |
| Stock parameter max abs diff | 0.0 |
| Eval logit max abs diff | 0.0 |
| Eval loss abs diff | 0.0 |
| Train logit max abs diff | 0.0 |
| Train loss abs diff | 0.0 |
| Stock gradient max abs diff | 0.0 |
| First backward W_up grad norm | 4.97 (useful) |
| First backward W_down grad norm | 0.0 (expected) |
| component gap analysis and leader reverse engineering W_down grad norm | 0.243 (branch recruiting) |
| HF roundtrip logit diff | 0.0 |
| Reloaded class | AdapterDebertaV2ForMaskedLM |

## Training setup

- Corpus: frozen compact-view-reinvest 10M pool × 2 passes = 20M words
- Tokenizer: legal16k (spatial repair route status same-pool tokenizer, vocab 16,384)
- Architecture: DeBERTa-v2 8×480 + zero-output bottleneck adapters
- Seeds: 43 / 43022 / 43023 (exact spatial repair route status seeds)
- Optimizer: AdamW lr=0.001, betas (0.9, 0.98), weight_decay 0.01
  (adapter output group: weight_decay 0.0)
- Schedule: cosine with warmup_fraction=0.06
- Masking: WWM fixed at 0.15
- batch_size=256, seq_length=256
- Gradient checkpointing enabled (to share GPU with concurrent runs)
- Checkpoint cadence: every 5M words

## Training arms

| arm | bottleneck |
|---|---:|
| adapter64 | 64 |
| adapter128 | 128 |

## Decision criteria for continuation to 80M+

Continue to 80M only if ALL of:
1. First loss matches spatial repair route status (9.837543)
2. Adapter branch is recruited (adapter output RMS > 0 at 20M)
3. Cheap7 ≥ spatial repair route status 20M (39.66) for at least one arm
4. No dominant Entity or Reading damage (≤ −2 vs spatial repair route status)
5. Trunk displacement is small (mean cosine to spatial repair route status > 0.95)

If gains are concentrated in GlobalPIQA only (as with Muon), do not continue.

## Evaluation plan

After 20M training completes:
1. Run cheap7 evaluation for both arms via eval_adapter_20m.py
2. Measure adapter recruitment (output RMS) and trunk displacement
3. Compare against spatial repair route status 20M and Muon 20M references
4. If criteria met, launch 80M continuation on the better arm
