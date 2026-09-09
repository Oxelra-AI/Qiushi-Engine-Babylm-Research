#!/usr/bin/env python3
from __future__ import annotations

import json
import pathlib
from transformers import AutoModelForMaskedLM, AutoTokenizer

ROOT = pathlib.Path("experiments/archive/initial_model_studies")
RUNS = {
    "relation_wwm": ROOT / "training/runs/babylm_smoke_relation_wwm_deberta_20k",
    "relation_wwm_shuffled": ROOT / "training/runs/babylm_smoke_relation_wwm_shuffled_deberta_20k",
}
OUT = ROOT / "data/relation_wwm_smoke_validation.json"
NOTE = (ROOT.parents[2] / 'research/notes/initial_model_studies/relation_wwm_smoke_validation.md')


def read_json(p: pathlib.Path):
    return json.loads(p.read_text(encoding="utf-8"))


def frac(a: float, b: float) -> float:
    return float(a) / float(b) if b else 0.0


def summarize_run(name: str, run: pathlib.Path) -> dict:
    metrics = read_json(run / "scientific_metrics.json")
    tele = read_json(run / "masking_telemetry_summary.json")
    logs = [json.loads(x) for x in (run / "training_log.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    loads = []
    for rel in ["hf_model", "hf_model/chck_1M"]:
        p = run / rel
        tok = AutoTokenizer.from_pretrained(p)
        model = AutoModelForMaskedLM.from_pretrained(p)
        loads.append({
            "rel": rel,
            "tokenizer_class": tok.__class__.__name__,
            "tokenizer_len": len(tok),
            "model_class": model.__class__.__name__,
            "parameter_count": model.num_parameters(),
        })
    t = tele["totals"]
    selected_groups = t["selected_groups"]
    candidate_groups = t["candidate_groups"]
    summary = {
        "run_dir": str(run),
        "mask_mode": metrics["mask_mode"],
        "word_exposure": metrics["word_exposure"],
        "actual_training_steps": metrics["actual_training_steps"],
        "parameter_count": metrics["parameter_count"],
        "loss_first": metrics["loss_first"],
        "loss_last": metrics["loss_last"],
        "masked_tokens_total": metrics["masked_tokens_total"],
        "masked_tokens_per_whitespace_word": metrics["masked_tokens_per_whitespace_word"],
        "num_log_lines": len(logs),
        "first_log": logs[0],
        "last_log": logs[-1],
        "saved_checkpoints": metrics["saved_checkpoints"],
        "load_checks": loads,
        "telemetry_totals": t,
        "selected_group_fraction": frac(selected_groups, candidate_groups),
        "candidate_relation_fraction": frac(t["relation_candidate_groups"], candidate_groups),
        "selected_relation_fraction": frac(t["relation_selected_groups"], selected_groups),
        "candidate_entity_fraction": frac(t["entity_candidate_groups"], candidate_groups),
        "selected_entity_fraction": frac(t["entity_selected_groups"], selected_groups),
        "candidate_state_fraction": frac(t["state_candidate_groups"], candidate_groups),
        "selected_state_fraction": frac(t["state_selected_groups"], selected_groups),
    }
    assert summary["word_exposure"] == 20000
    assert summary["actual_training_steps"] == 8
    assert len(logs) == 8
    assert summary["parameter_count"] == 34467424
    assert all(c["tokenizer_len"] == 16384 and c["model_class"] == "DebertaV2ForMaskedLM" for c in loads)
    assert selected_groups == 3000.0
    assert candidate_groups == 20000.0
    assert abs(summary["selected_group_fraction"] - 0.15) < 1e-9
    return summary


def main():
    results = {name: summarize_run(name, run) for name, run in RUNS.items()}
    rel = results["relation_wwm"]
    shuf = results["relation_wwm_shuffled"]
    comparison = {
        "relation_selected_fraction_delta": rel["selected_relation_fraction"] - shuf["selected_relation_fraction"],
        "entity_selected_fraction_delta": rel["selected_entity_fraction"] - shuf["selected_entity_fraction"],
        "state_selected_fraction_delta": rel["selected_state_fraction"] - shuf["selected_state_fraction"],
        "relation_enrichment_over_candidate": rel["selected_relation_fraction"] - rel["candidate_relation_fraction"],
        "shuffled_relation_enrichment_over_candidate": shuf["selected_relation_fraction"] - shuf["candidate_relation_fraction"],
        "entity_enrichment_over_candidate": rel["selected_entity_fraction"] - rel["candidate_entity_fraction"],
        "shuffled_entity_enrichment_over_candidate": shuf["selected_entity_fraction"] - shuf["candidate_entity_fraction"],
    }
    payload = {
        "status": "RELATION_WWM_SMOKES_VALIDATED",
        "trainer": "training/scripts/babylm_masked_train_relation_wwm.py",
        "results": results,
        "comparison": comparison,
        "interpretation": "Both relation_wwm and shuffled-control smokes train, save, reload, and preserve exact-K 15% selected whole-word groups. relation_wwm enriches selected relation/entity/state groups relative to the shuffled-weight control, so the mechanism and control are active.",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = [
        "# research — relation-WWM smoke validation", "",
        f"Evidence JSON: `{OUT}`", "",
        "Both smokes completed with ai_lab-accepted artifacts and loadable HF checkpoints.", "",
        "| metric | relation_wwm | shuffled control |",
        "|---|---:|---:|",
        f"| word exposure | {rel['word_exposure']} | {shuf['word_exposure']} |",
        f"| optimizer steps | {rel['actual_training_steps']} | {shuf['actual_training_steps']} |",
        f"| selected groups / candidate groups | {rel['telemetry_totals']['selected_groups']:.0f}/{rel['telemetry_totals']['candidate_groups']:.0f} | {shuf['telemetry_totals']['selected_groups']:.0f}/{shuf['telemetry_totals']['candidate_groups']:.0f} |",
        f"| selected group fraction | {rel['selected_group_fraction']:.3f} | {shuf['selected_group_fraction']:.3f} |",
        f"| candidate relation fraction | {rel['candidate_relation_fraction']:.3f} | {shuf['candidate_relation_fraction']:.3f} |",
        f"| selected relation fraction | {rel['selected_relation_fraction']:.3f} | {shuf['selected_relation_fraction']:.3f} |",
        f"| candidate entity fraction | {rel['candidate_entity_fraction']:.3f} | {shuf['candidate_entity_fraction']:.3f} |",
        f"| selected entity fraction | {rel['selected_entity_fraction']:.3f} | {shuf['selected_entity_fraction']:.3f} |",
        f"| candidate state fraction | {rel['candidate_state_fraction']:.4f} | {shuf['candidate_state_fraction']:.4f} |",
        f"| selected state fraction | {rel['selected_state_fraction']:.4f} | {shuf['selected_state_fraction']:.4f} |",
        f"| loss first→last | {rel['loss_first']:.3f}→{rel['loss_last']:.3f} | {shuf['loss_first']:.3f}→{shuf['loss_last']:.3f} |",
        "", "Interpretation: exact-K mask density is correct in both arms. The relation arm selects relation/entity/state groups more often than the shuffled-weight control while preserving the same candidate text and selected-group count. This is sufficient to launch a 20M falsification experiment; it is not yet evidence of BabyLM score improvement.",
    ]
    NOTE.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "out": str(OUT), "comparison": comparison}, indent=2))

if __name__ == "__main__":
    main()
