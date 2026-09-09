#!/usr/bin/env bash
# research: Full Qwen3.5-9B compact generation for FW mechanism-family new sources.
# Run from user root. Uses the same generator family as the validated compact views.
# This is generation/data construction only, not BabyLM model training or official evaluation.
set -uo pipefail
PROMPTS="experiments/archive/representation_and_objectives/data/fw_mechanism_source_selection/fw_mechanism_compact_prompts.jsonl"
OUTDIR="experiments/archive/representation_and_objectives/training/runs/qwen35_fw_compact_full_26015"
OUT="${OUTDIR}/outputs.jsonl"
mkdir -p "${OUTDIR}"
python3 - <<'PY'
import json, pathlib
p=pathlib.Path('experiments/archive/representation_and_objectives/data/fw_mechanism_source_selection/fw_mechanism_compact_prompts.jsonl')
n=sum(1 for line in p.open(encoding='utf-8') if line.strip())
print(json.dumps({'event':'preflight_prompts','path':str(p),'rows':n}, ensure_ascii=False), flush=True)
if n != 26015:
    raise SystemExit(f'expected 26015 prompts, got {n}')
PY
: "${CUDA_VISIBLE_DEVICES:=0}"
export CUDA_VISIBLE_DEVICES
"${BABYLM_GENERATOR:?configure-an-external-generator}" \
  --model qwen3.5-9b \
  --prompts-jsonl "${PROMPTS}" \
  --output-jsonl "${OUT}" \
  --batch-size 64 \
  --max-new-tokens 80 \
  --temperature 0.1 \
  --device cuda
python3 - <<'PY'
import json, pathlib, hashlib
out=pathlib.Path('experiments/archive/representation_and_objectives/training/runs/qwen35_fw_compact_full_26015/outputs.jsonl')
h=hashlib.sha256(); n=0; toks=0; models=set()
with out.open('rb') as fb:
    for chunk in iter(lambda: fb.read(1<<20), b''):
        h.update(chunk)
with out.open(encoding='utf-8') as f:
    for line in f:
        if not line.strip(): continue
        r=json.loads(line); n+=1; toks+=int(r.get('generated_tokens') or 0); models.add(str(r.get('model')))
print(json.dumps({'event':'postflight_outputs','path':str(out),'rows':n,'generated_tokens':toks,'models':sorted(models),'sha256':h.hexdigest()}, ensure_ascii=False), flush=True)
if n != 26015:
    raise SystemExit(f'expected 26015 outputs, got {n}')
PY
