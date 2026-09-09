#!/usr/bin/env python3
"""research: Score SHUF seed43122 replication. 

research: Run Entity evaluation for SHUF on GPU 0 to create per_target payload.
research: Run the validated mechanism scorer (compact T/U/N, Wikipedia, copy, Entity stratification).
research: Run ordinary held-out loss.

Follows pre-stated replication criteria from notes/038_shuf_dup_prestate.md.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative

import pathlib, sys

ROOT0 = _public_path('experiments/archive/relation_learning/scripts/score_shuf_seed43122.py')
ROOT = _PUBLIC_ROOT

WS = ROOT / "experiments/archive/relation_learning"
COMPACT_EXPERIENCE = ROOT / "experiments/archive/compact_experience"
OUT = WS / "data/shuf_seed43122_probe"
OUT.mkdir(parents=True, exist_ok=True)
PT_DIR = OUT / "per_target"
PT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(WS / "scripts"))

# ---- Phase 1: Entity evaluation for SHUF ----
import eval_split_entity_official as entity_eval  # noqa: E402
import json, os, subprocess, re, math, time

STRICT = ROOT / "experiments/archive/representation_and_objectives/data/pristine_official_coordinate/babylm-eval/strict"
ENTITY_DATA = STRICT / "evaluation_data/full_eval/entity_tracking"
NLP_DATA_ROOT = ROOT / "experiments/archive/initial_model_studies/data/nltk_data"

def parse_score(text):
    for pat in [r"### AVERAGE [A-Z_ '\\-]*\s*\n\s*([+-]?[0-9]+(?:\.[0-9]+)?)",
                r"AVERAGE ACCURACY\s*\n\s*([+-]?[0-9]+(?:\.[0-9]+)?)"]:
        m = re.search(pat, text)
        if m:
            val = float(m.group(1))
            return val if math.isfinite(val) and -5 <= val <= 105 else None
    return None

SHUF_RUN = WS / "training/runs/qwen_shuffled_control_16k_seed43122"
SHUF_PT = PT_DIR / "qwen_shuffled_control_seed43122.json"

if not SHUF_PT.exists():
    print("Phase 1: Entity evaluation for SHUF seed43122", flush=True)
    model_path = SHUF_RUN / "hf_model/chck_100M"
    out_dir = OUT / "official_outputs/SHUF/Entity/chck_100M"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = OUT / "logs/entity_SHUF_chck_100M.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0"
    env["TOKENIZERS_PARALLELISM"] = "false"
    if NLP_DATA_ROOT.exists():
        env["NLTK_DATA"] = str(NLP_DATA_ROOT)
    hf_cache = OUT / "hf_cache/SHUF"
    for key, val in {"HF_HOME": hf_cache, "HF_HUB_CACHE": hf_cache/"hub",
                     "TRANSFORMERS_CACHE": hf_cache/"transformers",
                     "HF_MODULES_CACHE": hf_cache/"modules",
                     "HF_DATASETS_CACHE": hf_cache/"datasets"}.items():
        val.mkdir(parents=True, exist_ok=True)
        env[key] = str(val)
    
    argv = [
        sys.executable, "-B", "-m", "evaluation_pipeline.sentence_zero_shot.run",
        "--model_path_or_name", str(model_path),
        "--backend", "mlm",
        "--task", "entity_tracking",
        "--data_path", str(ENTITY_DATA),
        "--revision_name", "SHUF_seed43122_Entity",
        "--save_predictions",
        "--batch_size", "128",
        "--non_causal_batch_size", "64",
        "--output_dir", str(out_dir),
    ]
    with log_path.open("w") as fh:
        proc = subprocess.run(argv, cwd=str(STRICT), env=env, stdout=fh, stderr=subprocess.STDOUT, timeout=2400)
    
    reports = sorted(out_dir.rglob("best_temperature_report.txt"), key=lambda p: p.stat().st_mtime)
    preds = sorted(out_dir.rglob("predictions.json"), key=lambda p: p.stat().st_mtime)
    score = parse_score(reports[-1].read_text() if reports else "")
    
    payload = {
        "target": "qwen_shuffled_control_seed43122",
        "description": "SHUF seed43122 replication arm",
        "family": "debertav2_8x480_16k_compact_experience_seed43122_relation_replication",
        "run_dir": str(SHUF_RUN),
        "endpoint": "chck_100M",
        "tasks": {"Entity": {
            "score": score,
            "returncode": proc.returncode,
            "predictions": str(preds[-1]) if preds else None,
            "report": str(reports[-1]) if reports else None,
        }},
        "stable_scores": {"Entity": score},
    }
    SHUF_PT.write_text(json.dumps(payload, indent=2))
    print(f"Entity score SHUF seed43122: {score}", flush=True)
else:
    data = json.loads(SHUF_PT.read_text())
    print(f"Entity already evaluated: {data.get('stable_scores', {}).get('Entity')}", flush=True)

# ---- Phase 2: Mechanism scorer ----
print("\nPhase 2: Mechanism scoring (compact T/U/N, Wikipedia, copy, Entity strata)", flush=True)
import score_paired_context_relation_design as base  # noqa: E402

base.DEFAULT_OUT = OUT
base.DEFAULT_NOTE = (_PUBLIC_ROOT / 'research/notes/relation_learning/041_shuf_seed43122_probe.md')
base.ARMS = {
    "OFF": {
        "role": "OFF",
        "description": "seed43122 official_lengthmatched baseline",
        "run": COMPACT_EXPERIENCE / "training/runs/official_lengthmatched_16k_seed43122",
        "per_target": COMPACT_EXPERIENCE / "data/full_eval/per_target/official_lengthmatched_seed43122.json",
    },
    "SHUF": {
        "role": "SHUF",
        "description": "seed43122 qwen_shuffled_control originals paired with WRONG rewrites",
        "run": WS / "training/runs/qwen_shuffled_control_16k_seed43122",
        "per_target": SHUF_PT,
    },
}
base.CONTRASTS = [("SHUFminusOFF", "SHUF", "OFF")]
base.main()

# ---- Phase 3: Ordinary held-out loss ----
print("\nPhase 3: Ordinary held-out loss", flush=True)
import ordinary_heldout_price_probe as ord_base  # noqa: E402

ord_base.OUT = WS / "data/shuf_seed43122_ordinary_heldout"
ord_base.NOTE = (_PUBLIC_ROOT / 'research/notes/relation_learning/041_shuf_seed43122_ordinary_heldout.md')
ord_base.CKPTS = ["chck_80M", "chck_90M", "chck_100M"]
ord_base.ARM_CONFIGS = {
    "OFF": COMPACT_EXPERIENCE / "training/runs/official_lengthmatched_16k_seed43122",
    "SHUF": WS / "training/runs/qwen_shuffled_control_16k_seed43122",
}
ord_base.ROLE = {"OFF": "OFF", "SHUF": "SHUF"}
ord_base.CONTRASTS = [("SHUF", "OFF")]
ord_base.main()
