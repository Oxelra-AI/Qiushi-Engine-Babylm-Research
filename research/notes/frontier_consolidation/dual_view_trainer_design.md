# dual view trainer design — Dual-View Adapter Trainer Design

## Scientific Basis
source free transfer synthesis shared-readout transfer test showed that jointly training on source-conditioned
and source-free views through shared weights enables transfer that survives source removal.
True alignment dramatically outperforms shuffled (source-absent shared_true -0.895 vs
shared_shuffled -0.572 NLL improvement over spatial repair route status baseline).

## Design Corrections Applied
1. **Do not continue from spatial repair route status 100M** — embed the dual-view mechanism within a fresh
   100M trajectory so a positive result matures into a legal endpoint.
2. **Same-target, same-mask** — both views mask the same rewrite word groups and predict
   the same rewrite tokens.
3. **Shuffled control changes only source correspondence** — same rewrite, masks, targets.
4. **Reproduce the probe mechanism** — shared weights, paired initialization.

## Implementation: dual_view_adapter_trainer.py

### Pair Data
- Pre-computed by build_and_validate_pair_data.py
- 3,005 pair rows (eid 950000-953004), 100% word-group mapping coverage
- Token-level coverage 94.9% (BPE context effects at boundaries; irrelevant since WWM
  operates at word-group level)
- Saved: data/pair_data/pair_data.json

### Mechanism
For ALL rows: standard WWM masking (15% word groups) → main MLM forward → main loss.

For pair rows (~4.6% of data, ~12 per batch of 256):
1. Identify masked word groups that are in the rewrite portion (pre-computed set)
2. Map to rewrite-only word groups (pre-computed mapping)
3. Build rewrite-only input with those word groups masked as [MASK]
4. Forward rewrite-only through SAME model → auxiliary logits
5. CE loss on same rewrite targets

Total loss = main_MLM + lambda * aux_CE (lambda=1.0 default)

### Architecture
Same as scale1.75: DeBERTa-v2 8×480 + zero-init bottleneck-128 adapter
Adapter scale: 1.0 (not 1.75; the dual-view signal replaces the amplitude mechanism)

### Training Arms for 20M Panel
- dual_view_true: correct source-rewrite alignment (treatment)
- mlm_only: no auxiliary view (matched adapter scale=1.0 baseline)
- (dual_view_shuffled: deferred; needs data reconstruction)

### Gate Criteria (from token value private readout synthesis)
Before 100M: dual_view_true must beat mlm_only on cheap7 while preserving sentinels
(Supplement subject-aux, Reading, EWoK material/spatial/quantitative).

## Files
- Pre-computed pair data: data/pair_data/pair_data.json
- Pair data summary: data/pair_data/pair_data_summary.json
- Trainer script: scripts/dual_view_adapter_trainer.py
- Validation script: scripts/build_and_validate_pair_data.py
