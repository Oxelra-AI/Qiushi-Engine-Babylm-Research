#!/usr/bin/env python3
"""research independent CPU/file verification of the scale1.75 chck_82M endpoint.

Purpose: inspect the reported above-frontier chck_82M endpoint evidence across
the representation and consolidation studies:
complete nine-column arithmetic, legal exposure/tokenizer/data provenance, file
preservation, and CPU HuggingFace loadability.

This script does not launch evaluation or training. It reads existing evidence,
recomputes deterministic checks, hashes files, and performs a tiny CPU forward pass.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

ROOT = _public_path('.')
STUDY = _public_path('experiments/archive/frontier_consolidation')
A01 = _public_path('experiments/archive/representation_and_objectives')
OUT_DIR = _public_path('experiments/archive/frontier_consolidation/data/chck82_independent_verification')
OUT_JSON = _public_path('experiments/archive/frontier_consolidation/data/chck82_independent_verification/chck82_independent_verification.json')
OUT_MD = _public_path('research/documents/frontier_consolidation/data/chck82_independent_verification/chck82_independent_verification.md')

SUMMARY = _public_path('experiments/archive/representation_and_objectives/data/scale1p75_chck82_full_eval_reproduction/summary/scale1p75_100M_full_eval_hardened_summary.json')
REPEAT = _public_path('experiments/archive/representation_and_objectives/data/chck82_eval_comparison/chck82_eval_comparison.json')
PREFLIGHT = _public_path('experiments/archive/representation_and_objectives/data/chck82_reproducibility_preflight/chck82_reproducibility_preflight.json')
LEADER = _public_path('experiments/archive/representation_and_objectives/data/chck82_public_leader_comparison/chck82_public_leader_comparison.json')
SNAPSHOT_MANIFEST = _public_path('experiments/archive/representation_and_objectives/data/chck82_endpoint_snapshot/manifest.json')
RUN_DIR = _public_path('experiments/archive/frontier_consolidation/training/runs/adapter128_scale1p75_h100M100M_seed43022_official_ladder')
ENDPOINT = _public_path('experiments/archive/frontier_consolidation/training/runs/adapter128_scale1p75_h100M100M_seed43022_official_ladder/hf_model/chck_82M')
RUN_METRICS = _public_path('experiments/archive/frontier_consolidation/training/runs/adapter128_scale1p75_h100M100M_seed43022_official_ladder/scientific_metrics.json')

CHEAP7_COLS = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA", "Reading"]
ALL_COLS = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "SuperGLUE", "GlobalPIQA", "Reading", "AoA"]
EXPECTED_MODEL_SHA = "93ceb76adf5a33d349f1de33e988e6ed0c2b2a547dbd92cf83cc952f8e2591b3"
EXPECTED_POOL_SHA = "215944978157394dbecf2039f2c1e1806bfcbb9701421920f78605eb58975a23"
EXPECTED_STREAM_SHA = "3dd19f09deeca44d6b07340e70cd15baa4b5c7bffe0bd462ff37536e470e7691"
EXPECTED_TOKENIZER_SHA = "91b775514b9f4e2d1f37c28445ab98181007e16d14f763955168547e293ee8f9"


def rel(p: Path | str | None) -> str | None:
    if p is None:
        return None
    p = Path(p)
    try:
        return str(p.relative_to(ROOT))
    except Exception:
        return str(p)


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path, block: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(block)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def close(a, b, tol=1e-9):
    return a is not None and b is not None and abs(float(a) - float(b)) <= tol


def loadability_check(endpoint: Path):
    """CPU-only tiny forward pass with local HF dynamic-module cache."""
    out = {"attempted": True, "ok": False, "error": None}
    try:
        # Keep HuggingFace dynamic module writes under the output directory.
        hf_cache = _public_path('experiments/archive/frontier_consolidation/data/chck82_independent_verification/hf_cache')
        (hf_cache / "modules").mkdir(parents=True, exist_ok=True)
        os.environ["HF_HOME"] = str(hf_cache)
        os.environ["TRANSFORMERS_CACHE"] = str(hf_cache)
        os.environ["HF_MODULES_CACHE"] = str(hf_cache / "modules")
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

        import torch
        from transformers import AutoModelForMaskedLM, AutoTokenizer

        tok = AutoTokenizer.from_pretrained(str(endpoint), local_files_only=True)
        model = AutoModelForMaskedLM.from_pretrained(
            str(endpoint), trust_remote_code=True, local_files_only=True
        )
        model.eval()
        param_count = sum(p.numel() for p in model.parameters())
        adapter_param_count = sum(p.numel() for n, p in model.named_parameters() if ".adapter." in n)
        enc = tok("The small dog runs in the garden.", return_tensors="pt")
        with torch.no_grad():
            pred = model(**enc)
        logits = pred.logits
        finite = bool(torch.isfinite(logits).all().item())
        out.update({
            "ok": finite and list(logits.shape)[-1] == int(model.config.vocab_size),
            "model_class": model.__class__.__name__,
            "tokenizer_class": tok.__class__.__name__,
            "param_count": int(param_count),
            "adapter_param_count": int(adapter_param_count),
            "vocab_size_config": int(model.config.vocab_size),
            "vocab_size_tokenizer": int(len(tok)),
            "logits_shape": list(logits.shape),
            "logits_all_finite": finite,
            "adapter_scale": float(getattr(model.config, "adapter_scale", -1.0)),
            "adapter_bottleneck": int(getattr(model.config, "adapter_bottleneck", -1)),
            "auto_map": getattr(model.config, "auto_map", None),
        })
    except Exception as e:  # intentionally captured into dossier
        out["error"] = repr(e)
    return out


def main():
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    summary = read_json(SUMMARY)
    repeat = read_json(REPEAT)
    preflight = read_json(PREFLIGHT)
    leader = read_json(LEADER)
    snapshot = read_json(SNAPSHOT_MANIFEST)
    metrics = read_json(RUN_METRICS)

    scores = summary.get("scores", {})
    missing_scores = [c for c in ALL_COLS if scores.get(c) is None]
    cheap7 = sum(float(scores[c]) for c in CHEAP7_COLS) / len(CHEAP7_COLS) if not any(c not in scores for c in CHEAP7_COLS) else None
    overall = sum(float(scores[c]) for c in ALL_COLS) / len(ALL_COLS) if not missing_scores else None

    leader_scores = leader.get("public_leader", {}).get("scores", {})
    leader_overall_recomputed = sum(float(leader_scores[c]) for c in ALL_COLS) / len(ALL_COLS)

    saved = metrics.get("saved_checkpoints", [])
    chck82_records = [x for x in saved if x.get("name") == "chck_82M"]
    chck82 = chck82_records[0] if chck82_records else {}
    actual_chck82_words = int(chck82.get("actual_cumulative_word_exposure", -1))
    target_chck82_words = int(chck82.get("target_word_exposure", -1))
    pool_words = int(preflight.get("legal_data", {}).get("counts", {}).get("pool_10m", {}).get("words", -1))
    stream_words = int(preflight.get("legal_data", {}).get("counts", {}).get("stream_100m", {}).get("words", -1))
    selected_epochs = actual_chck82_words / pool_words if pool_words > 0 else None

    endpoint_files = {}
    for name in ["config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "adapter_scaled_modeling.py"]:
        p = ENDPOINT / name
        endpoint_files[name] = {
            "path": rel(p),
            "exists": p.exists(),
            "size_bytes": p.stat().st_size if p.exists() else None,
            "sha256": sha256_file(p) if p.exists() else None,
        }

    snapshot_model = _public_path('experiments/archive/representation_and_objectives/data/chck82_endpoint_snapshot/checkpoint/chck_82M/model.safetensors')
    snapshot_model_sha = sha256_file(snapshot_model) if snapshot_model.exists() else None

    loadability = loadability_check(ENDPOINT)

    checks = {
        "all_nine_scores_present": len(missing_scores) == 0,
        "cheap7_matches_summary": close(cheap7, summary.get("cheap7"), 1e-9),
        "overall_matches_summary": close(overall, summary.get("Overall"), 1e-9),
        "overall_above_reported_41p8": overall is not None and overall > 41.8,
        "overall_above_recomputed_public_leader": overall is not None and overall > leader_overall_recomputed,
        "repeatability_pass": repeat.get("status") == "PASS" and repeat.get("deltas_hardened_minus_first", {}).get("max_abs_column_delta", 999) < 0.01,
        "preflight_pass": preflight.get("status") == "PASS",
        "snapshot_manifest_status": snapshot.get("status") == "CHCK82_ENDPOINT_SNAPSHOT",
        "source_model_sha_matches_expected": endpoint_files["model.safetensors"]["sha256"] == EXPECTED_MODEL_SHA,
        "snapshot_model_sha_matches_source": snapshot_model_sha == endpoint_files["model.safetensors"]["sha256"],
        "pool_hash_matches_expected": preflight.get("legal_data", {}).get("hashes", {}).get("pool_10m") == EXPECTED_POOL_SHA,
        "stream_hash_matches_expected": preflight.get("legal_data", {}).get("hashes", {}).get("stream_100m") == EXPECTED_STREAM_SHA,
        "tokenizer_training_hash_expected": preflight.get("hashes", {}).get("training_tokenizer", {}).get("sha256") == EXPECTED_TOKENIZER_SHA,
        "tokenizer_vocab_maps_identical": bool(preflight.get("legal_data", {}).get("tokenizer_vocab_compare", {}).get("vocab_maps_identical")),
        "pool_words_eq_10M": pool_words == 10_000_000,
        "stream_words_eq_100M": stream_words == 100_000_000,
        "selected_checkpoint_target_82M": target_chck82_words == 82_000_000,
        "selected_checkpoint_actual_within_10_epochs": actual_chck82_words > 0 and actual_chck82_words <= 100_000_000,
        "selected_checkpoint_before_100M_endpoint": actual_chck82_words > 0 and actual_chck82_words < int(metrics.get("word_exposure", -1)),
        "loadability_cpu_forward_ok": bool(loadability.get("ok")),
    }
    checks["all_core_checks_pass"] = all(checks.values())

    result = {
        "status": "PASS" if checks["all_core_checks_pass"] else "CHECKS_FAILED",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "purpose": "Independent A02-side CPU/file verification of the A01-reported scale1.75 chck_82M above-frontier endpoint while research factorized runs continue.",
        "paths": {
            "endpoint": rel(ENDPOINT),
            "hardened_summary": rel(SUMMARY),
            "repeatability": rel(REPEAT),
            "preflight": rel(PREFLIGHT),
            "public_leader_comparison": rel(LEADER),
            "snapshot_manifest": rel(SNAPSHOT_MANIFEST),
            "run_metrics": rel(RUN_METRICS),
        },
        "score_arithmetic": {
            "scores": {c: scores.get(c) for c in ALL_COLS},
            "missing_scores": missing_scores,
            "cheap7_recomputed": cheap7,
            "cheap7_reported": summary.get("cheap7"),
            "overall_recomputed": overall,
            "overall_reported": summary.get("Overall"),
            "public_leader_reported": leader.get("public_leader", {}).get("overall_reported"),
            "public_leader_recomputed_from_displayed_columns": leader_overall_recomputed,
            "margin_vs_reported_41p8": None if overall is None else overall - 41.8,
            "margin_vs_recomputed_public_leader": None if overall is None else overall - leader_overall_recomputed,
            "repeatability_overall_delta": repeat.get("deltas_hardened_minus_first", {}).get("Overall"),
            "repeatability_max_abs_column_delta": repeat.get("deltas_hardened_minus_first", {}).get("max_abs_column_delta"),
        },
        "legal_exposure_and_provenance": {
            "pool_words": pool_words,
            "stream_words": stream_words,
            "pool_sha256": preflight.get("legal_data", {}).get("hashes", {}).get("pool_10m"),
            "stream_sha256": preflight.get("legal_data", {}).get("hashes", {}).get("stream_100m"),
            "tokenizer_training_sha256": preflight.get("hashes", {}).get("training_tokenizer", {}).get("sha256"),
            "tokenizer_vocab_maps_identical": preflight.get("legal_data", {}).get("tokenizer_vocab_compare", {}).get("vocab_maps_identical"),
            "tokenizer_vocab_size": preflight.get("legal_data", {}).get("tokenizer_metadata", {}).get("tokenizer", {}).get("vocab_size"),
            "checkpoint_record_from_run_metrics": chck82,
            "selected_actual_words": actual_chck82_words,
            "selected_target_words": target_chck82_words,
            "selected_epochs_against_10M_pool": selected_epochs,
            "full_run_word_exposure": metrics.get("word_exposure"),
            "actual_training_steps_full_run": metrics.get("actual_training_steps"),
            "saved_checkpoint_count_full_run": len(saved),
            "aoa_ladder_complete_in_hardened_summary": summary.get("endpoint_ready", {}).get("aoa_ladder_complete"),
            "source_words_consumed_full_run": metrics.get("source_words_consumed"),
        },
        "artifact_identity": {
            "expected_model_sha256": EXPECTED_MODEL_SHA,
            "endpoint_files": endpoint_files,
            "snapshot_model_path": rel(snapshot_model),
            "snapshot_model_exists": snapshot_model.exists(),
            "snapshot_model_sha256": snapshot_model_sha,
            "snapshot_candidate_sha256_from_manifest": snapshot.get("candidate_model_sha256"),
            "config_core_from_preflight": preflight.get("original_endpoint", {}).get("config_core"),
        },
        "cpu_loadability": loadability,
        "checks": checks,
        "scientific_reading": (
            "The chck_82M endpoint is a legal-exposure, loadable, repeated-score above-frontier checkpoint candidate if the fixed from-corpus reproduction also matches or is re-evaluated. "
            "It establishes a practical score-bearing endpoint strategy and proves late 82M-to-100M competence loss in the scale1.75 trajectory, but it does not by itself settle the general source-free correspondence-learning mechanism."
        ),
        "elapsed_sec": round(time.time() - t0, 3),
    }

    OUT_JSON.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    md = []
    md.append("# research independent verification — scale1.75 chck_82M\n")
    md.append(f"Status: **{result['status']}**\n")
    md.append("\n## Score arithmetic\n")
    md.append(f"- Recomputed Overall: `{overall}`; hardened summary Overall: `{summary.get('Overall')}`.\n")
    md.append(f"- Recomputed cheap7: `{cheap7}`; hardened summary cheap7: `{summary.get('cheap7')}`.\n")
    md.append(f"- Margin vs displayed 41.80: `{result['score_arithmetic']['margin_vs_reported_41p8']}`.\n")
    md.append(f"- Margin vs recomputed public leader columns: `{result['score_arithmetic']['margin_vs_recomputed_public_leader']}`.\n")
    md.append(f"- Measurement repeatability: Overall delta `{result['score_arithmetic']['repeatability_overall_delta']}`, max column delta `{result['score_arithmetic']['repeatability_max_abs_column_delta']}`.\n")
    md.append("\n## Legal exposure and provenance\n")
    md.append(f"- Legal pool words `{pool_words}`, SHA `{result['legal_exposure_and_provenance']['pool_sha256']}`.\n")
    md.append(f"- 100M stream words `{stream_words}`, SHA `{result['legal_exposure_and_provenance']['stream_sha256']}`.\n")
    md.append(f"- Selected checkpoint actual words `{actual_chck82_words}` = `{selected_epochs}` epochs of the 10M pool; target `{target_chck82_words}`.\n")
    md.append(f"- Tokenizer training SHA `{result['legal_exposure_and_provenance']['tokenizer_training_sha256']}`; vocab-map identity `{result['legal_exposure_and_provenance']['tokenizer_vocab_maps_identical']}`.\n")
    md.append("\n## Artifact identity and loadability\n")
    md.append(f"- Source model SHA `{endpoint_files['model.safetensors']['sha256']}`; snapshot SHA `{snapshot_model_sha}`.\n")
    md.append(f"- CPU HF loadability OK: `{loadability.get('ok')}`; class `{loadability.get('model_class')}`; params `{loadability.get('param_count')}`; logits shape `{loadability.get('logits_shape')}`.\n")
    md.append("\n## Core checks\n")
    for k, v in checks.items():
        md.append(f"- `{k}`: `{v}`\n")
    md.append("\n## Scientific reading\n")
    md.append(result["scientific_reading"] + "\n")
    md.append(f"\nJSON: `{rel(OUT_JSON)}`\n")
    OUT_MD.write_text("".join(md), encoding="utf-8")

    print(json.dumps({"status": result["status"], "out_json": rel(OUT_JSON), "out_md": rel(OUT_MD), "checks_pass": checks["all_core_checks_pass"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
