# aoa analysis and curriculum design — Concise View Protocol: Results and Resource Constraints

## Protocol Design (Completed)
Built `concise_view_protocol.py` with:
1. Length-adaptive compression targets (75-95% for ≤14 words, down to 45-65% for 36+ words)
2. Entity pre-extraction with explicit MUST-INCLUDE constraints in prompts
3. Source-aware handling (CHILDES/BNC speaker labels)
4. Integrated verifier checking entity recall, negation, modality, numbers, copy overlap

## Generation Test Results
- Model used: Qwen3-1.7B (ONLY complete model available locally)
- Acceptance rate: **5.5% (11/200)** — unacceptably low
- Primary failure: model doesn't follow instructions (generates repetitive text, unrelated content, or copies input verbatim)
- Root cause: 1.7B model too weak for faithful instruction-following paraphrasing

## Resource Constraints Blocking This Route
1. **GPU memory**: Both H100s have ~11 GiB usable (stale allocations from completed training blocking ~68 GiB each; nvidia-smi --gpu-reset permission-denied)
2. **Model availability**: Qwen3-4B missing shard 3/3; Qwen3-8B missing shards 3-5/5; all other 8B+ models not cached locally
3. **Network**: HuggingFace unreachable (cannot download missing shards or models)
4. **Consequence**: Cannot run any model ≥4B for generation. Only Qwen3-1.7B works but is too weak.

## Key Rule Discovery
BabyLM 2026 Strict-Small allows **free dataset construction** within 10M word budget.
The leader uses FineWeb-Edu simplification pairs — NOT official BabyLM sources.
This means our limitation isn't the paraphrasing mechanism but the SOURCE CONTENT.

## Implication for Route
The concise-view protocol DESIGN is sound but cannot be EXECUTED without:
- Network access (to download Qwen3-4B shard3, or FineWeb-Edu data directly), OR
- GPU memory recovery (to load qwen3.5-9b at 18 GiB)

## Next: Pivot to AoA (Highest Arithmetic Leverage)
Current 41.3 has AoA=0.0. Leader has AoA=0.0. Entry `deberta-base-75k-sam_ext-s1` has AoA=22.9 worth +2.54 Overall.
If our 41.3 model achieves even AoA=+5, that adds +0.56 → 41.9 → ABOVE LEADER.
AoA requires: full checkpoint trajectory (chck_1M through chck_100M) + learning dynamics that correlate with child age of acquisition.
Training DeBERTa-v2 (35M params) works fine on 11 GiB GPUs.
