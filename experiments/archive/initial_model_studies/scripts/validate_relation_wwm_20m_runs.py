#!/usr/bin/env python3
from __future__ import annotations

import json
import pathlib
from transformers import AutoModelForMaskedLM, AutoTokenizer

ROOT = pathlib.Path("experiments/archive/initial_model_studies")
RUNS = {
    "relation_wwm": ROOT / "training/runs/babylm_relation_wwm_deberta_20M",
    "relation_wwm_shuffled": ROOT / "training/runs/babylm_relation_wwm_shuffled_deberta_20M",
}
OUT = ROOT / "data/relation_wwm_20m_training_validation.json"
NOTE = (ROOT.parents[2] / 'research/notes/initial_model_studies/relation_wwm_20m_training_validation.md')


def read_json(p: pathlib.Path):
    return json.loads(p.read_text(encoding="utf-8"))


def frac(a: float, b: float) -> float:
    return float(a) / float(b) if b else 0.0


def summarize_run(name: str, run: pathlib.Path) -> dict:
    metrics = read_json(run / "scientific_metrics.json")
    tele = read_json(run / "masking_telemetry_summary.json")
    tok_sum = read_json(run / "tokenization_coupling_summary.json")
    manifest = read_json(run / "example_order_manifest.json")
    logs = [json.loads(x) for x in (run / "training_log.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    ckpts = sorted([p.name for p in (run / "hf_model").iterdir() if p.is_dir() and p.name.startswith("chck_")], key=lambda x: int(x.split("_")[1][:-1]))
    load_checks = []
    for rel in ["hf_model", "hf_model/chck_10M", "hf_model/chck_20M"]:
        p = run / rel
        tok = AutoTokenizer.from_pretrained(p)
        model = AutoModelForMaskedLM.from_pretrained(p)
        load_checks.append({
            "rel": rel,
            "tokenizer_class": tok.__class__.__name__,
            "tokenizer_len": len(tok),
            "model_class": model.__class__.__name__,
            "parameter_count": model.num_parameters(),
        })
    t = tele["totals"]
    selected_groups = float(t["selected_groups"])
    candidate_groups = float(t["candidate_groups"])
    out = {
        "run_dir": str(run),
        "mask_mode": metrics["mask_mode"],
        "word_exposure": metrics["word_exposure"],
        "selected_for_training_words": metrics["selected_for_training_words"],
        "example_pool_words_actual": metrics["example_pool_words_actual"],
        "actual_training_steps": metrics["actual_training_steps"],
        "lr_schedule_total_steps": metrics["lr_schedule_total_steps"],
        "parameter_count": metrics["parameter_count"],
        "embedding_parameter_count": metrics["embedding_parameter_count"],
        "non_embedding_parameter_count": metrics["non_embedding_parameter_count"],
        "vocab_size": metrics["vocab_size"],
        "loss_first": metrics["loss_first"],
        "loss_last": metrics["loss_last"],
        "masked_tokens_total": metrics["masked_tokens_total"],
        "masked_tokens_per_whitespace_word": metrics["masked_tokens_per_whitespace_word"],
        "first_log": logs[0],
        "mid_log": logs[len(logs)//2],
        "last_log": logs[-1],
        "num_log_lines": len(logs),
        "num_selection_epochs": len(manifest.get("selection_epochs", [])),
        "selection_epochs": manifest.get("selection_epochs", []),
        "unique_official_pool_words": manifest.get("unique_official_pool_words"),
        "num_checkpoints": len(ckpts),
        "checkpoints": ckpts,
        "load_checks": load_checks,
        "tokenization_coupling_summary": tok_sum,
        "telemetry_totals": t,
        "selected_group_fraction": frac(selected_groups, candidate_groups),
        "candidate_relation_fraction": frac(t["relation_candidate_groups"], candidate_groups),
        "selected_relation_fraction": frac(t["relation_selected_groups"], selected_groups),
        "candidate_entity_fraction": frac(t["entity_candidate_groups"], candidate_groups),
        "selected_entity_fraction": frac(t["entity_selected_groups"], selected_groups),
        "candidate_state_fraction": frac(t["state_candidate_groups"], candidate_groups),
        "selected_state_fraction": frac(t["state_selected_groups"], selected_groups),
    }
    assert out["word_exposure"] == 20_000_000
    assert out["selected_for_training_words"] == 20_000_000
    assert out["example_pool_words_actual"] == 10_000_000
    assert out["actual_training_steps"] == 489
    assert out["lr_schedule_total_steps"] == 2442
    assert out["num_log_lines"] == 489
    assert out["last_log"]["cumulative_word_exposure"] == 20_000_000
    assert out["num_selection_epochs"] == 2
    assert out["num_checkpoints"] == 2
    assert out["checkpoints"] == ["chck_10M", "chck_20M"]
    assert out["parameter_count"] == 34_467_424
    assert out["vocab_size"] == 16_384
    assert all(c["tokenizer_len"] == 16_384 and c["model_class"] == "DebertaV2ForMaskedLM" and c["parameter_count"] == 34_467_424 for c in load_checks)
    # exact-K is applied per example after truncation; aggregate selected/candidate can differ slightly from 0.15.
    assert abs(out["selected_group_fraction"] - 0.15) < 1e-4
    return out


def main():
    results = {name: summarize_run(name, run) for name, run in RUNS.items()}
    rel = results["relation_wwm"]
    sh = results["relation_wwm_shuffled"]
    comparison = {
        "loss_last_relation_minus_shuffled": rel["loss_last"] - sh["loss_last"],
        "masked_tokens_per_word_relation_minus_shuffled": rel["masked_tokens_per_whitespace_word"] - sh["masked_tokens_per_whitespace_word"],
        "selected_relation_fraction_delta": rel["selected_relation_fraction"] - sh["selected_relation_fraction"],
        "selected_entity_fraction_delta": rel["selected_entity_fraction"] - sh["selected_entity_fraction"],
        "selected_state_fraction_delta": rel["selected_state_fraction"] - sh["selected_state_fraction"],
        "relation_enrichment_over_candidate": rel["selected_relation_fraction"] - rel["candidate_relation_fraction"],
        "shuffled_relation_enrichment_over_candidate": sh["selected_relation_fraction"] - sh["candidate_relation_fraction"],
        "entity_enrichment_over_candidate": rel["selected_entity_fraction"] - rel["candidate_entity_fraction"],
        "shuffled_entity_enrichment_over_candidate": sh["selected_entity_fraction"] - sh["candidate_entity_fraction"],
    }
    payload = {
        "status": "RELATION_WWM_20M_TRAINING_VALIDATED",
        "results": results,
        "comparison": comparison,
        "interpretation": "Both 20M runs are valid matched training artifacts with exact-K 15% whole-word group density. relation_wwm strongly enriches selected relation/entity groups compared with shuffled-control; scores are not yet evaluated.",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = [
        "# research — relation-WWM 20M training validation", "",
        f"Evidence JSON: `{OUT}`", "",
        "Both 20M runs are complete and loadable. These are mechanism-screening artifacts, not BabyLM score evidence until evaluated.", "",
        "| quantity | relation_wwm | shuffled control |", "|---|---:|---:|",
        f"| word exposure | {rel['word_exposure']} | {sh['word_exposure']} |",
        f"| optimizer steps | {rel['actual_training_steps']} | {sh['actual_training_steps']} |",
        f"| LR schedule total | {rel['lr_schedule_total_steps']} | {sh['lr_schedule_total_steps']} |",
        f"| checkpoints | {', '.join(rel['checkpoints'])} | {', '.join(sh['checkpoints'])} |",
        f"| loss first→last | {rel['loss_first']:.4f}→{rel['loss_last']:.4f} | {sh['loss_first']:.4f}→{sh['loss_last']:.4f} |",
        f"| selected group fraction | {rel['selected_group_fraction']:.3f} | {sh['selected_group_fraction']:.3f} |",
        f"| candidate relation fraction | {rel['candidate_relation_fraction']:.3f} | {sh['candidate_relation_fraction']:.3f} |",
        f"| selected relation fraction | {rel['selected_relation_fraction']:.3f} | {sh['selected_relation_fraction']:.3f} |",
        f"| candidate entity fraction | {rel['candidate_entity_fraction']:.3f} | {sh['candidate_entity_fraction']:.3f} |",
        f"| selected entity fraction | {rel['selected_entity_fraction']:.3f} | {sh['selected_entity_fraction']:.3f} |",
        f"| selected state fraction | {rel['selected_state_fraction']:.4f} | {sh['selected_state_fraction']:.4f} |",
        "", "Next evidence required: official-compatible scores at chck_10M and chck_20M for Entity, GlobalPIQA, Reading, BLiMP, Supplement, and COMPS, compared to each other and to the ordinary WWM trajectory.",
    ]
    NOTE.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "out": str(OUT), "comparison": comparison}, indent=2))

if __name__ == "__main__":
    main()
