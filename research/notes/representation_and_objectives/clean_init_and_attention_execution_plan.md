# clean init and attention execution plan — contamination-aware clean-init recheck and raw-name attention pilot

## Scientific Motivation exists

raw name binding construction and design discovered a real sequential reuse bug in `raw_name_binding_probe.py`: for a given condition, the `bs=-1` cell trained from the already-updated `bs=+1` model. That makes the raw-name mean-pooled result unsuitable for sign-pair interpretation. It also raises a general scientific warning: in a nonconvex learner, complete local fit does not by itself establish path independence. Even though direct code inspection of causal gauge experimental design shows `run_one` deep-copies `paired_models[condition]` for every bridge sign, the central causal gauge experimental design/291 mechanism is important enough to re-establish with an explicit untouched-initialization audit.

## Minimal decision-changing expensive work

The load-bearing GPU recheck is intentionally small:

```bash
CUDA_VISIBLE_DEVICES=0 python3 -B experiments/archive/representation_and_objectives/training/scripts/clean_gauge_from_init.py \
  --out experiments/archive/representation_and_objectives/data/clean_gauge_from_init \
  --models shared_trunk untied --bridge-signs 1 -1 --seeds 29000 \
  --epochs 220 --device cuda --print-every 55
```

This tests only the minimal supplied-harness sign pair and contrast:
- `shared_trunk` must train both signs from the same untouched initial hash and reproduce bridge-sign-controlled h1/h3 graph reversal;
- `untied` must also fit local state/comparison rows but not transport the comparison coordinate to state decisions;
- the stored initialization hashes must remain unchanged after all cells.

A CPU smoke run already passed at `data/clean_gauge_smoke/` and verified the hash-audit machinery.

## Parallel raw-name attention pilot

In parallel, I launched the smallest useful query-attention raw-name pilot:

```bash
CUDA_VISIBLE_DEVICES=1 python3 -B experiments/archive/representation_and_objectives/training/scripts/revision_292b_raw_name_attention_probe.py \
  --out experiments/archive/representation_and_objectives/data/raw_name_attention_primary \
  --conditions tied shared_trunk untied --bridge-signs 1 -1 --seeds 29300 \
  --epochs 220 --device cuda --print-every 55
```

This follows raw name binding construction and design's diagnosis that mean-pooled GRU state binding can fit but comparison binding fails for `shared_trunk`. The query-attention module separates event and query at `<QRY>`, mean-pools the query segment, and uses cross-attention from the query vector to event positions. CPU audit at `data/attention_smoke_audit/` confirmed:
- exactly one `<QRY>` per candidate sequence;
- nonempty event and query masks;
- candidate queries produce different untrained scores;
- comparison scoring runs without malformed inputs.

Interpretation is deliberately constrained: this pilot first asks whether learned candidate matching makes train comparison fit possible. Transport-like raw-name patterns should not be read as mechanism evidence until the clean supplied-harness recheck reestablishes the causal gauge experimental design/291 mechanism.

## Architectural caveat for the attention variant

In `revision_292b_raw_name_attention_probe.py`, `shared_trunk` shares the GRU trunk between state and comparison scorers but has separate scalar heads and separate `QueryAttentionScorer` modules. Their `W_q` and `W_k` projections are initialized identically by deep-copy but become separate trainable modules. Therefore, if raw-name attention succeeds, the exact statement is that a learned shared recurrent representation plus query-conditioned matching can support the gauge-carrying pathway under the tested training objective. It does not yet prove that the matcher itself is shared, nor that natural language supplies the matcher without architectural bias. That would require a later stricter architecture comparison.

## Analysis prepared

`scripts/clean_attention_result_analysis.py` is syntax-checked and smoke-tested. It reads completed output files and reports:
- clean-init status and initialization audit;
- bridge-sign pairs for supplied harness;
- raw-name attention binding fit versus provisional transport-like signals;
- row-paired state `d_e` sign reversal by direct/graph/unchanged categories.

The script will be run on the final GPU output directories before interpreting results.
