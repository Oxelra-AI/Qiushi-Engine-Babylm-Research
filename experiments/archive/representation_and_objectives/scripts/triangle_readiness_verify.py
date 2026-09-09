#!/usr/bin/env python3
"""research compact triangle readiness/provenance verifier.

Cheap CPU check before endpoint interpretation: verifies that the three trained arms are
on the intended data streams, recipe, exposure schedule, tokenizer lineage, and 100M
endpoints. Does not run benchmark evaluation.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = _public_path('.')
OUT_ROOT = _public_path('experiments/archive/representation_and_objectives/data/triangle_readiness')

ARMS = {
    "compact_view_reinvest_historical": {
        "run": _public_path('experiments/archive/frontier_consolidation/training/runs/cleanqwen_fineweb_compact_view_reinvest_16k_seed43022'),
        "expected_example_jsonl": _public_path('experiments/archive/frontier_consolidation/data/density_cleanqwen_overlay_medium_riskhard/cleanqwen_fineweb_compact_view_reinvest_100M.jsonl'),
        "expected_label": "cleanqwen_fineweb_compact_view_reinvest",
        "expected_sha256": "3dd19f09deeca44d6b07340e70cd15baa4b5c7bffe0bd462ff37536e470e7691",
    },
    "compact_repeat_reinvest_gc": {
        "run": _public_path('experiments/archive/representation_and_objectives/training/runs/gc_compact_repeat_reinvest_16k_seed43022_r2'),
        "expected_example_jsonl": _public_path('experiments/archive/frontier_consolidation/data/density_cleanqwen_overlay_medium_riskhard/cleanqwen_fineweb_repeat_compact_reinvest_100M.jsonl'),
        "expected_label": "cleanqwen_fineweb_repeat_compact_reinvest",
        "expected_sha256": "91e8817e25f234b87481fa6ca96d968d6c460b84317273629eda21ea3be24eaa",
    },
    "adjbreak_reinvest_gc": {
        "run": _public_path('experiments/archive/representation_and_objectives/training/runs/gc_adjbreak_reinvest_16k_seed43022_r2'),
        "expected_example_jsonl": _public_path('experiments/archive/representation_and_objectives/data/compact_reinvest_adjbreak_control/cleanqwen_fineweb_compact_view_reinvest_adjbreak_100M.jsonl'),
        "expected_label": "cleanqwen_fineweb_compact_view_reinvest_adjbreak",
        "expected_sha256": "3097308293081a784d383f9bcaaf01a792e3be3484565e3b63d180ebe7017312",
    },
}

EXPECTED_RECIPE = {
    "variant": "masking_curriculum_wwm_fixed",
    "backend": "mlm",
    "model_family": "DebertaV2ForMaskedLM",
    "parameter_count": 34467424,
    "vocab_size": 16384,
    "tokenizer_label": "baseline16k",
    "word_exposure": 100000000,
    "actual_training_steps": 2529,
    "masking_curriculum": "wwm_fixed",
    "mask_prob_start": 0.15,
    "mask_prob_end": 0.15,
    "hidden_size": 480,
    "n_layer": 8,
    "n_head": 8,
    "ffn_mult": 4,
    "seed": 43,
    "seq_length": 256,
    "max_seq_length": 256,
}


def rel(path: Path | str) -> str:
    p = Path(path)
    try:
        return str(p.relative_to(ROOT))
    except Exception:
        return str(p)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def file_tail(path: Path, n: int = 1) -> list[str]:
    if not path.exists():
        return []
    # training logs are small enough here; keep simple and robust.
    lines = path.read_text(encoding="utf-8").splitlines()
    return lines[-n:]


def verify_arm(name: str, cfg: dict[str, Any]) -> dict[str, Any]:
    run = cfg["run"]
    metrics_path = run / "scientific_metrics.json"
    manifest_path = run / "example_order_manifest.json"
    log_path = run / "training_log.jsonl"
    endpoint = run / "hf_model/chck_100M"
    metrics = load_json(metrics_path) if metrics_path.exists() else {}
    manifest = load_json(manifest_path) if manifest_path.exists() else {}
    recipe_diffs = {k: {"actual": metrics.get(k), "expected": v} for k, v in EXPECTED_RECIPE.items() if metrics.get(k) != v}
    stream = Path(manifest.get("example_jsonl") or metrics.get("example_jsonl") or "")
    # If stream was recorded as user-root-relative, resolve under ROOT.
    stream_abs = stream if stream.is_absolute() else ROOT / stream
    stream_sha = sha256_file(stream_abs) if stream_abs.exists() else None
    log_last = file_tail(log_path, 1)
    last_record = json.loads(log_last[0]) if log_last else {}
    saved = metrics.get("saved_checkpoints", [])
    return {
        "run": rel(run),
        "metrics_exists": metrics_path.exists(),
        "manifest_exists": manifest_path.exists(),
        "training_log_exists": log_path.exists(),
        "endpoint_model_exists": (endpoint / "model.safetensors").exists(),
        "recorded_example_jsonl": manifest.get("example_jsonl") or metrics.get("example_jsonl"),
        "expected_example_jsonl": rel(cfg["expected_example_jsonl"]),
        "example_jsonl_path_ok": stream_abs.resolve() == cfg["expected_example_jsonl"].resolve() if stream_abs.exists() and cfg["expected_example_jsonl"].exists() else False,
        "example_jsonl_sha256": stream_sha,
        "expected_sha256": cfg["expected_sha256"],
        "example_jsonl_sha_ok": stream_sha == cfg["expected_sha256"],
        "label_actual": manifest.get("example_jsonl_label") or metrics.get("example_jsonl_label"),
        "label_expected": cfg["expected_label"],
        "label_ok": (manifest.get("example_jsonl_label") or metrics.get("example_jsonl_label")) == cfg["expected_label"],
        "recipe_diffs": recipe_diffs,
        "recipe_ok": not recipe_diffs,
        "checkpoint_count": len(saved),
        "first_checkpoint": saved[0] if saved else None,
        "last_checkpoint": saved[-1] if saved else None,
        "last_training_record": last_record,
    }


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    arms = {name: verify_arm(name, cfg) for name, cfg in ARMS.items()}
    all_ok = all(a["metrics_exists"] and a["manifest_exists"] and a["training_log_exists"] and a["endpoint_model_exists"] and a["example_jsonl_sha_ok"] and a["label_ok"] and a["recipe_ok"] and a["checkpoint_count"] == 100 for a in arms.values())
    payload = {
        "status": "TRIANGLE_READY_FOR_GUARDED_ENDPOINT_READOUT" if all_ok else "TRIANGLE_READINESS_PROBLEMS",
        "purpose": "Verify data/provenance/recipe readiness before interpreting compact-view triangle no-AoA endpoint results.",
        "all_ok": all_ok,
        "arms": arms,
        "interpretation": "If all_ok is true and research GC equivalence remains true, endpoint/posthoc differences can be read as data-mechanism evidence under research seed-spread and broad-competence rules; benchmark evaluation is still required.",
    }
    out_json = _public_path('experiments/archive/representation_and_objectives/data/triangle_readiness/triangle_readiness.json')
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    md = _public_path('research/documents/representation_and_objectives/data/triangle_readiness/triangle_readiness.md')
    lines = ["# research triangle readiness", "", f"Status: `{payload['status']}`", ""]
    for name, a in arms.items():
        lines.append(f"- `{name}`: recipe_ok={a['recipe_ok']}, sha_ok={a['example_jsonl_sha_ok']}, label_ok={a['label_ok']}, endpoint_model_exists={a['endpoint_model_exists']}, checkpoints={a['checkpoint_count']}")
    lines.append("")
    lines.append(f"JSON: `{rel(out_json)}`")
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "all_ok": all_ok, "out_json": rel(out_json), "out_md": rel(md)}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
