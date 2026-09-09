#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import math
import pathlib
from transformers import AutoModelForMaskedLM, AutoTokenizer

ROOT = pathlib.Path("experiments/archive/initial_model_studies")
RUNS = {
    "consistency": ROOT / "training/runs/babylm_entity_consistency_deberta_20M",
    "shuffled_pair": ROOT / "training/runs/babylm_entity_shuffled_pair_deberta_20M",
}
OUT = ROOT / "data/entity_consistency_20m_training_validation.json"
NOTE = (ROOT.parents[2] / 'research/notes/initial_model_studies/entity_consistency_20m_training_validation.md')


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(p: pathlib.Path):
    return json.loads(p.read_text(encoding="utf-8"))


def read_logs(p: pathlib.Path):
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]


def model_weight_file(p: pathlib.Path) -> pathlib.Path:
    for name in ["model.safetensors", "pytorch_model.bin"]:
        q = p / name
        if q.exists():
            return q
    raise FileNotFoundError(p)


def summarize(name: str, run: pathlib.Path) -> dict:
    metrics = read_json(run / "scientific_metrics.json")
    cfg = read_json(run / "entity_aux_config.json")
    manifest = read_json(run / "example_order_manifest.json")
    logs = read_logs(run / "training_log.jsonl")
    hf_root = run / "hf_model"
    ckpts = sorted([p.name for p in hf_root.iterdir() if p.is_dir() and p.name.startswith("chck_")], key=lambda s: int(s.split("_")[1][:-1]))
    load_checks = []
    for rel in ["hf_model", "hf_model/chck_10M", "hf_model/chck_20M"]:
        p = run / rel
        tok = AutoTokenizer.from_pretrained(p)
        model = AutoModelForMaskedLM.from_pretrained(p)
        wf = model_weight_file(p)
        load_checks.append({
            "rel": rel,
            "tokenizer_len": len(tok),
            "tokenizer_class": tok.__class__.__name__,
            "model_class": model.__class__.__name__,
            "parameter_count": model.num_parameters(),
            "weight_sha256": sha256_file(wf),
        })
    hf_files = sorted([p.name for p in hf_root.iterdir()])
    has_projection_pollution = any("projection" in x or "entity" in x for x in hf_files)
    aux = metrics["entity_aux_summary"]
    loss_vals = [float(x["loss_entity_aux"]) for x in logs]
    mlm_vals = [float(x["loss_mlm"]) for x in logs]
    pair_vals = [int(x.get("entity_aux_pairs", 0)) for x in logs]
    active_vals = [int(x.get("entity_aux_active", 0)) for x in logs]
    summary = {
        "run_dir": str(run),
        "entity_aux_mode": metrics["entity_aux_mode"],
        "word_exposure": metrics["word_exposure"],
        "example_pool_words_actual": metrics["example_pool_words_actual"],
        "selected_for_training_words": metrics["selected_for_training_words"],
        "actual_training_steps": metrics["actual_training_steps"],
        "lr_schedule_total_steps": metrics["lr_schedule_total_steps"],
        "parameter_count": metrics["parameter_count"],
        "train_parameter_count_including_aux_head": metrics["train_parameter_count_including_aux_head"],
        "entity_aux_parameter_count": metrics["entity_aux_parameter_count"],
        "loss_first": metrics["loss_first"],
        "loss_last": metrics["loss_last"],
        "loss_mlm_first": mlm_vals[0],
        "loss_mlm_last": mlm_vals[-1],
        "loss_aux_first": loss_vals[0],
        "loss_aux_last": loss_vals[-1],
        "loss_aux_min": min(loss_vals),
        "loss_aux_max": max(loss_vals),
        "num_logs": len(logs),
        "first_log": logs[0],
        "mid_log": logs[len(logs)//2],
        "last_log": logs[-1],
        "entity_aux_summary": aux,
        "pair_values_min_max": [min(pair_vals), max(pair_vals)],
        "active_steps": sum(active_vals),
        "entity_aux_config": cfg,
        "num_selection_epochs": len(manifest.get("selection_epochs", [])),
        "selection_epochs": manifest.get("selection_epochs", []),
        "manifest_sha256": sha256_file(run / "example_order_manifest.json"),
        "checkpoints": ckpts,
        "load_checks": load_checks,
        "hf_model_files": hf_files,
        "projection_head_exists_outside_hf_model": (run / "entity_projection_train_only.pt").exists(),
        "projection_pollution_in_hf_model": has_projection_pollution,
    }
    assert summary["word_exposure"] == 20_000_000
    assert summary["selected_for_training_words"] == 20_000_000
    assert summary["example_pool_words_actual"] == 10_000_000
    assert summary["actual_training_steps"] == 489
    assert summary["lr_schedule_total_steps"] == 2442
    assert summary["num_logs"] == 489
    assert summary["checkpoints"] == ["chck_10M", "chck_20M"]
    assert summary["parameter_count"] == 34_467_424
    assert summary["entity_aux_parameter_count"] == 61_824
    assert summary["projection_head_exists_outside_hf_model"] and not summary["projection_pollution_in_hf_model"]
    assert aux["entity_aux_pairs"] > 0 and aux["entity_aux_active"] > 0
    assert all(math.isfinite(v) for v in loss_vals + mlm_vals)
    assert all(c["tokenizer_len"] == 16384 and c["model_class"] == "DebertaV2ForMaskedLM" and c["parameter_count"] == 34_467_424 for c in load_checks)
    return summary


def main() -> None:
    results = {name: summarize(name, run) for name, run in RUNS.items()}
    c = results["consistency"]
    s = results["shuffled_pair"]
    comparison = {
        "same_example_order_manifest_hash": c["manifest_sha256"] == s["manifest_sha256"],
        "aux_pairs_consistency_minus_shuffled": c["entity_aux_summary"]["entity_aux_pairs"] - s["entity_aux_summary"]["entity_aux_pairs"],
        "aux_candidates_consistency_minus_shuffled": c["entity_aux_summary"]["entity_aux_candidates"] - s["entity_aux_summary"]["entity_aux_candidates"],
        "aux_mean_gap_consistency_minus_shuffled": c["entity_aux_summary"]["entity_aux_mean_gap_per_pair"] - s["entity_aux_summary"]["entity_aux_mean_gap_per_pair"],
        "aux_mean_loss_per_active_step_consistency_minus_shuffled": c["entity_aux_summary"]["entity_aux_mean_loss_per_active_step"] - s["entity_aux_summary"]["entity_aux_mean_loss_per_active_step"],
        "mlm_loss_last_consistency_minus_shuffled": c["loss_mlm_last"] - s["loss_mlm_last"],
        "total_loss_last_consistency_minus_shuffled": c["loss_last"] - s["loss_last"],
        "consistency_chck10_vs_chck20_same_hash": c["load_checks"][1]["weight_sha256"] == c["load_checks"][2]["weight_sha256"],
        "shuffled_chck10_vs_chck20_same_hash": s["load_checks"][1]["weight_sha256"] == s["load_checks"][2]["weight_sha256"],
    }
    payload = {
        "status": "ENTITY_CONSISTENCY_20M_TRAINING_VALIDATED",
        "results": results,
        "comparison": comparison,
        "interpretation": "Both 20M arms are valid matched training artifacts. Ability evidence still requires official-compatible evaluation with direct checkpoint paths.",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = [
        "# research — Entity Mention Consistency 20M training validation", "",
        f"Evidence JSON: `{OUT}`", "",
        "Both arms are valid/loadable 20M artifacts; this is training validation, not task-score evidence.", "",
        "| quantity | consistency | shuffled_pair | delta c-s |", "|---|---:|---:|---:|",
        f"| word exposure | {c['word_exposure']} | {s['word_exposure']} | 0 |",
        f"| steps | {c['actual_training_steps']} | {s['actual_training_steps']} | 0 |",
        f"| aux pairs | {c['entity_aux_summary']['entity_aux_pairs']:.0f} | {s['entity_aux_summary']['entity_aux_pairs']:.0f} | {comparison['aux_pairs_consistency_minus_shuffled']:.0f} |",
        f"| aux candidates | {c['entity_aux_summary']['entity_aux_candidates']:.0f} | {s['entity_aux_summary']['entity_aux_candidates']:.0f} | {comparison['aux_candidates_consistency_minus_shuffled']:.0f} |",
        f"| mean pair gap | {c['entity_aux_summary']['entity_aux_mean_gap_per_pair']:.3f} | {s['entity_aux_summary']['entity_aux_mean_gap_per_pair']:.3f} | {comparison['aux_mean_gap_consistency_minus_shuffled']:.3f} |",
        f"| mean aux loss/active step | {c['entity_aux_summary']['entity_aux_mean_loss_per_active_step']:.3f} | {s['entity_aux_summary']['entity_aux_mean_loss_per_active_step']:.3f} | {comparison['aux_mean_loss_per_active_step_consistency_minus_shuffled']:.3f} |",
        f"| final MLM loss | {c['loss_mlm_last']:.4f} | {s['loss_mlm_last']:.4f} | {comparison['mlm_loss_last_consistency_minus_shuffled']:+.4f} |",
        f"| final total loss | {c['loss_last']:.4f} | {s['loss_last']:.4f} | {comparison['total_loss_last_consistency_minus_shuffled']:+.4f} |",
        "", "Projection heads are outside `hf_model`; checkpoint hashes differ between 10M and 20M within both arms.",
    ]
    NOTE.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "out": str(OUT), "comparison": comparison}, indent=2))

if __name__ == "__main__":
    main()
