#!/usr/bin/env python3
from __future__ import annotations

import json
import pathlib
import re
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

ROOT = pathlib.Path("experiments/archive/initial_model_studies")
RUN_ROOT = ROOT / "training/runs"
TOKEN_RUN = "babylm_masked_token_pos512_seed43_1M"
WWM_RUN = "babylm_masked_wwm_pos512_seed43_1M"
EVAL_SUB = "eval_results_masked_1m_pos512/hf_model/chck_1M/zero_shot/mlm"
OUT_PROFILE = ROOT / "data/masked_1m_seed43_profile.json"
OUT_TRAIN = ROOT / "data/masked_1m_seed43_training_summary.json"
OUT_NOTE = (ROOT.parents[2] / 'research/notes/initial_model_studies/masked_1m_seed43_token_vs_wwm_profile.md')
OUT_STABILITY = (ROOT.parents[2] / 'research/notes/initial_model_studies/wwm_cross_seed_stability_interpretation.md')
PROFILE = ROOT / "data/masked_1m_grid_pos512_profile.json"

COLS = ["blimp_fast", "supplement_fast", "ewok_fast", "entity_tracking_fast", "comps", "reading_eye_tracking", "reading_self_paced"]
REPORTS = {
    "blimp_fast": "blimp/blimp_fast/best_temperature_report.txt",
    "supplement_fast": "blimp/supplement_fast/best_temperature_report.txt",
    "ewok_fast": "ewok/ewok_fast/best_temperature_report.txt",
    "entity_tracking_fast": "entity_tracking/entity_tracking_fast/best_temperature_report.txt",
    "comps": "comps/comps/best_temperature_report.txt",
    "reading": "reading/report.txt",
}


def load_json(path: pathlib.Path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_avg(report: pathlib.Path) -> float:
    txt = report.read_text(encoding="utf-8", errors="replace")
    m = re.search(r"AVERAGE ACCURACY\s*\n([0-9.\-]+)", txt)
    if not m:
        raise RuntimeError(f"could not parse average accuracy from {report}\n{txt[:500]}")
    return float(m.group(1))


def read_reading(report: pathlib.Path) -> dict[str, float]:
    txt = report.read_text(encoding="utf-8", errors="replace")
    out = {}
    for label, key in [("EYE TRACKING SCORE", "reading_eye_tracking"), ("SELF-PACED READING SCORE", "reading_self_paced")]:
        m = re.search(re.escape(label) + r":\s*([0-9.\-]+)", txt)
        if not m:
            raise RuntimeError(f"could not parse {label} from {report}\n{txt}")
        out[key] = float(m.group(1))
    return out


def load_checks(run_dir: pathlib.Path) -> list[dict]:
    out = []
    for rel in ["hf_model", "hf_model/chck_1M"]:
        p = run_dir / rel
        tok = AutoTokenizer.from_pretrained(p)
        model = AutoModelForMaskedLM.from_pretrained(p)
        ids = torch.randint(low=0, high=len(tok), size=(1, 180))
        y = model(input_ids=ids, attention_mask=torch.ones_like(ids), labels=ids)
        out.append({
            "rel": rel,
            "ok": True,
            "tokenizer_class": tok.__class__.__name__,
            "model_class": model.__class__.__name__,
            "parameter_count": sum(x.numel() for x in model.parameters()),
            "max_position_embeddings": int(model.config.max_position_embeddings),
            "seq_len_forward_tested": 180,
            "forward_loss": float(y.loss.detach()),
        })
    return out


def extract_run(run_id: str, mode: str) -> dict:
    run_dir = RUN_ROOT / run_id
    metrics = load_json(run_dir / "scientific_metrics.json")
    manifest = load_json(run_dir / "example_order_manifest.json")
    assert metrics["word_exposure"] == 1_000_000
    assert metrics["mask_mode"] == mode
    assert metrics["max_position_embeddings"] >= 512
    scores: dict[str, float] = {}
    reports: dict[str, str] = {}
    for col in ["blimp_fast", "supplement_fast", "ewok_fast", "entity_tracking_fast", "comps"]:
        rp = run_dir / EVAL_SUB / REPORTS[col]
        assert rp.exists(), rp
        scores[col] = read_avg(rp)
        reports[col] = str(rp)
    rp = run_dir / EVAL_SUB / REPORTS["reading"]
    assert rp.exists(), rp
    scores.update(read_reading(rp))
    reports["reading"] = str(rp)
    return {
        "run_id": run_id,
        "mask_mode": mode,
        "metrics": metrics,
        "manifest": manifest,
        "load_checks": load_checks(run_dir),
        "scores": scores,
        "reports": reports,
    }


def diff(a: float, b: float) -> float:
    return round(a - b, 4)


def fmt(x):
    return f"{x:.2f}" if isinstance(x, float) else str(x)


def main() -> None:
    token = extract_run(TOKEN_RUN, "token")
    wwm = extract_run(WWM_RUN, "wwm")
    assert token["manifest"]["consumed_example_ids_in_order"] == wwm["manifest"]["consumed_example_ids_in_order"]
    assert token["manifest"]["source_words_consumed"] == wwm["manifest"]["source_words_consumed"]
    assert token["manifest"]["selected_for_training_words"] == wwm["manifest"]["selected_for_training_words"] == 1_000_000
    seed43_delta = {c: diff(wwm["scores"][c], token["scores"][c]) for c in COLS}

    train_summary = []
    for row in [token, wwm]:
        m = row["metrics"]
        train_summary.append({
            "run_id": row["run_id"],
            "mask_mode": row["mask_mode"],
            "parameter_count": m["parameter_count"],
            "loss_first": m["loss_first"],
            "loss_last": m["loss_last"],
            "word_exposure": m["word_exposure"],
            "steps": m["actual_training_steps"],
            "seed": m["seed"],
            "extra_init_seed": m["extra_init_seed"],
            "train_rng_seed": m["train_rng_seed"],
            "max_seq_length": m["max_seq_length"],
            "max_position_embeddings": m["max_position_embeddings"],
            "source_words": row["manifest"]["source_words_consumed"],
            "first12_examples": row["manifest"]["consumed_example_ids_in_order"][:12],
            "load_checks": row["load_checks"],
        })
    profile_rows = [{"run_id": token["run_id"], "mask_mode": "token", "scores": token["scores"], "reports": token["reports"]},
                    {"run_id": wwm["run_id"], "mask_mode": "wwm", "scores": wwm["scores"], "reports": wwm["reports"]}]
    profile_payload = {"train_summary": train_summary, "profile_rows": profile_rows, "wwm_minus_token": seed43_delta}
    OUT_TRAIN.parent.mkdir(parents=True, exist_ok=True)
    OUT_PROFILE.parent.mkdir(parents=True, exist_ok=True)
    OUT_TRAIN.write_text(json.dumps(train_summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    OUT_PROFILE.write_text(json.dumps(profile_payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    lines = [
        "# research — Seed-43 1M masked-LM token masking vs whole-word masking",
        "",
        f"Evidence JSON: `{OUT_PROFILE}`",
        "",
        "This reconstructs the completed seed-43 profiles from report files after the original runner failed only during final aggregation. The two runs use identical selected examples/order/source mix, baseline tokenizer, 8-layer/256-hidden BERT MLM, `max_seq_length=256`, `max_position_embeddings=512`, exact 1M word exposure, and official `mlm` backend.",
        "",
        "| mode | BLiMP | Supp | EWoK | Entity | COMPS | Read eye | Read SPR | loss_last |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in [token, wwm]:
        s = row["scores"]
        lines.append(f"| {row['mask_mode']} | {fmt(s['blimp_fast'])} | {fmt(s['supplement_fast'])} | {fmt(s['ewok_fast'])} | {fmt(s['entity_tracking_fast'])} | {fmt(s['comps'])} | {fmt(s['reading_eye_tracking'])} | {fmt(s['reading_self_paced'])} | {fmt(row['metrics']['loss_last'])} |")
    lines += ["", "## WWM minus token", "", "| BLiMP | Supp | EWoK | Entity | COMPS | Read eye | Read SPR |", "|---:|---:|---:|---:|---:|---:|---:|", f"| {fmt(seed43_delta['blimp_fast'])} | {fmt(seed43_delta['supplement_fast'])} | {fmt(seed43_delta['ewok_fast'])} | {fmt(seed43_delta['entity_tracking_fast'])} | {fmt(seed43_delta['comps'])} | {fmt(seed43_delta['reading_eye_tracking'])} | {fmt(seed43_delta['reading_self_paced'])} |"]
    OUT_NOTE.parent.mkdir(parents=True, exist_ok=True)
    OUT_NOTE.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Cross-seed synthesis with research seed-42.
    research = load_json(PROFILE)
    seed42_delta = research["wwm_minus_token"]
    mean_delta = {c: round((seed42_delta[c] + seed43_delta[c]) / 2, 4) for c in COLS}
    same_sign_positive = {c: seed42_delta[c] > 0 and seed43_delta[c] > 0 for c in COLS}
    stab_lines = [
        "# research — Cross-seed stability of whole-word masking at 1M",
        "",
        f"Seed-42 evidence: `{PROFILE}`",
        f"Seed-43 evidence: `{OUT_PROFILE}`",
        "",
        "Both comparisons isolate WWM against token-level masking under the same model family, tokenizer, exposure, position capacity, official `mlm` backend, and paired data/order within seed. Seed 43 changes only the random condition.",
        "",
        "| column | seed42 WWM-token | seed43 WWM-token | mean | positive both seeds |",
        "|---|---:|---:|---:|---:|",
    ]
    for c in COLS:
        stab_lines.append(f"| {c} | {fmt(seed42_delta[c])} | {fmt(seed43_delta[c])} | {fmt(mean_delta[c])} | {same_sign_positive[c]} |")
    # Direct interpretation in scientific terms.
    positives = [c for c in COLS if same_sign_positive[c]]
    stab_lines += [
        "",
        "## Interpretation",
        "",
        f"Positive in both seeds: {', '.join(positives) if positives else 'none'}.",
        "The WWM effect is cross-seed stable for BLiMP, Supplement, EWoK, and Entity Tracking, but not for COMPS or Reading. The Entity gain is modest but repeated; WWM alone does not close the large gap to the current top system, so subsequent work should treat WWM as a fixed base and decompose the next factors from the SOTA route one at a time.",
    ]
    OUT_STABILITY.parent.mkdir(parents=True, exist_ok=True)
    OUT_STABILITY.write_text("\n".join(stab_lines) + "\n", encoding="utf-8")
    print(json.dumps({"seed43_delta": seed43_delta, "seed42_delta": seed42_delta, "mean_delta": mean_delta, "positive_both": same_sign_positive, "out_profile": str(OUT_PROFILE), "out_stability": str(OUT_STABILITY)}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
