#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import pathlib
import re
import shutil
import subprocess

from training_process import run_training
import sys

from transformers import AutoModelForMaskedLM, AutoTokenizer

ROOT = pathlib.Path("experiments/archive/initial_model_studies")
TRAIN = ROOT / "training/scripts/babylm_crossview_mask_train.py"
STRICT_DIR = ROOT / "repos/babylm-eval/strict"
RUN_ROOT = ROOT / "training/runs"
META = ROOT / "data/state_revision_68/state_materialization_all_seeds.json"
OUT_JSON = ROOT / "data/state_coherent_vs_corrupted_1m_profile.json"
OUT_TRAIN_JSON = ROOT / "data/state_coherent_vs_corrupted_1m_training_summary.json"
OUT_NOTE = (ROOT.parents[2] / 'research/notes/initial_model_studies/state_coherent_vs_corrupted_1m_profile.md')
LOG = (ROOT.parents[2] / 'research/notes/initial_model_studies/state_coherent_vs_corrupted_1m_train_and_profile.log')
SEED_SPECS = {42: {"extra_init_seed": 456, "train_rng_seed": 789}, 43: {"extra_init_seed": 457, "train_rng_seed": 790}}
TASKS = [
    ("blimp_fast", "blimp", "evaluation_data/fast_eval/blimp_fast"),
    ("supplement_fast", "blimp", "evaluation_data/fast_eval/supplement_fast"),
    ("ewok_fast", "ewok", "evaluation_data/fast_eval/evaluation_data/fast_eval/ewok_fast"),
    ("entity_tracking_fast", "entity_tracking", "evaluation_data/fast_eval/entity_tracking_fast"),
    ("comps", "comps", "evaluation_data/full_eval/comps"),
]
COLS = ["blimp_fast", "supplement_fast", "ewok_fast", "entity_tracking_fast", "comps", "reading_eye_tracking", "reading_self_paced"]
COMMON = [
    "--max_word_exposure", "1000000",
    "--example_pool_words", "1000000",
    "--checkpoint_words", "1000000",
    "--tokenizer_label", "baseline16k",
    "--mask_prob", "0.15",
    "--seq_length", "256",
    "--max_seq_length", "256",
    "--max_position_embeddings", "512",
    "--batch_size", "83",
    "--lr_total_steps", "98",
    "--hidden_size", "256",
    "--n_layer", "8",
    "--n_head", "8",
    "--ffn_mult", "4",
    "--learning_rate", "0.001",
    "--log_every", "50",
]


def setup_env() -> dict[str, str]:
    env = os.environ.copy()
    hf_home = ROOT / "training/hf_home"
    env["HF_HOME"] = str(hf_home.resolve())
    env["HF_HUB_CACHE"] = str((hf_home / "hub").resolve())
    env["TRANSFORMERS_CACHE"] = str((hf_home / "transformers").resolve())
    env["HF_MODULES_CACHE"] = str((ROOT / "training/hf_modules_cache").resolve())
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    for key in ["HF_HOME", "HF_HUB_CACHE", "TRANSFORMERS_CACHE", "HF_MODULES_CACHE"]:
        pathlib.Path(env[key]).mkdir(parents=True, exist_ok=True)
    return env


def run(cmd: list[str], env: dict[str, str], cwd: pathlib.Path | None = None, logf=None) -> None:
    line = "$ " + " ".join(cmd)
    print(line, flush=True)
    if logf:
        logf.write("\n" + line + "\n"); logf.flush()
    p = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env, cwd=str(cwd) if cwd else None)
    print(p.stdout[-5000:], flush=True)
    if logf:
        logf.write(p.stdout + f"\n[returncode={p.returncode}]\n"); logf.flush()
    if p.returncode != 0:
        raise RuntimeError(f"command failed with {p.returncode}: {' '.join(cmd)}\n{p.stdout[-10000:]}")


def load_json(path: pathlib.Path):
    return json.loads(path.read_text(encoding="utf-8"))


def run_id_for(arm: str, seed: int) -> str:
    return f"babylm_step69_state_{arm}_wwm_seed{seed}_1M"


def read_avg(report: pathlib.Path) -> float:
    txt = report.read_text(encoding="utf-8", errors="replace")
    m = re.search(r"AVERAGE ACCURACY\s*\n([0-9.\-]+)", txt)
    if not m:
        raise RuntimeError(f"could not parse {report}\n{txt[:500]}")
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


def compare_supervision(coh_manifest: list[dict], cor_manifest: list[dict]) -> dict:
    if len(coh_manifest) != len(cor_manifest):
        return {"ok": False, "reason": "length mismatch", "len_coh": len(coh_manifest), "len_cor": len(cor_manifest)}
    mismatches = []
    state_count = 0
    official_count = 0
    for a, b in zip(coh_manifest, cor_manifest):
        if a.get("row_index") != b.get("row_index") or a.get("example_id") != b.get("example_id") or a.get("kind") != b.get("kind"):
            mismatches.append({"row_index": a.get("row_index"), "bad": {"row/example/kind": ((a.get("row_index"), a.get("example_id"), a.get("kind")), (b.get("row_index"), b.get("example_id"), b.get("kind")))}})
            continue
        if a["kind"] == "pair_crossview":
            state_count += 1
            fields = ["source_pair_id", "anchor_group_count", "anchor_token_count", "anchor_source_word_indices", "anchor_norms", "words", "kept_tokens", "untruncated_tokens", "tokens_lost_to_truncation"]
            bad = {f: (a.get(f), b.get(f)) for f in fields if a.get(f) != b.get(f)}
            if bad:
                mismatches.append({"row_index": a.get("row_index"), "bad": bad})
        else:
            official_count += 1
            fields = ["words", "kept_tokens", "untruncated_tokens", "tokens_lost_to_truncation"]
            bad = {f: (a.get(f), b.get(f)) for f in fields if a.get(f) != b.get(f)}
            if bad:
                mismatches.append({"row_index": a.get("row_index"), "bad": bad})
    return {"ok": len(mismatches) == 0, "state_count": state_count, "official_count": official_count, "mismatches_first10": mismatches[:10]}


def arm_path(seed_meta: dict, arm: str) -> str:
    return seed_meta[f"{arm}_path"]


def train_and_check(seed: int, arm: str, seed_meta: dict, env: dict[str, str], logf) -> dict:
    run_id = run_id_for(arm, seed)
    run_dir = RUN_ROOT / run_id
    cmd = [
        sys.executable, str(TRAIN),
        "--output_dir", str(run_dir),
        "--example_jsonl", arm_path(seed_meta, arm),
        "--example_jsonl_label", f"state_{arm}_90_10_1m",
        "--example_jsonl_meta", str(META),
        "--seed", str(seed),
        "--extra_init_seed", str(SEED_SPECS[seed]["extra_init_seed"]),
        "--train_rng_seed", str(SEED_SPECS[seed]["train_rng_seed"]),
        *COMMON,
    ]
    if (run_dir / "scientific_metrics.json").exists() and (run_dir / "hf_model" / "chck_1M" / "config.json").exists():
        print(f"REUSE_COMPLETED_TRAINING {run_id}", flush=True)
        if logf:
            logf.write(f"\nREUSE_COMPLETED_TRAINING {run_id}\n")
    else:
        if run_dir.exists():
            print(f"REMOVE_INCOMPLETE_RUN_DIR {run_id}", flush=True)
            shutil.rmtree(run_dir)
        run_training(cmd, output_dir=run_dir, timeout=2400, env=env, logf=logf)
    for rel in ["hf_model", "hf_model/chck_1M"]:
        AutoTokenizer.from_pretrained(run_dir / rel)
        AutoModelForMaskedLM.from_pretrained(run_dir / rel)
    metrics = load_json(run_dir / "scientific_metrics.json")
    summary = load_json(run_dir / "crossview_supervision_summary.json")
    manifest = load_json(run_dir / "crossview_supervision_manifest.json")
    assert metrics["data_source_type"] == "explicit_crossview_jsonl", metrics
    assert metrics["word_exposure"] == 1_000_000, metrics
    assert metrics["actual_training_steps"] == 98, metrics
    assert summary["total_words"] == 1_000_000, summary
    return {"run_id": run_id, "seed": seed, "arm": arm, "metrics": metrics, "supervision_summary": summary, "supervision_manifest": manifest}


def profile(run_id: str, env: dict[str, str], logf) -> dict:
    run_dir = RUN_ROOT / run_id
    model_path = (run_dir / "hf_model").resolve()
    outdir = (run_dir / "eval_results_revision_69").resolve()
    scores, reports = {}, {}
    for task_name, task, data_path in TASKS:
        run([
            sys.executable, "-m", "evaluation_pipeline.sentence_zero_shot.run",
            "--model_path_or_name", str(model_path), "--backend", "mlm", "--task", task,
            "--data_path", data_path, "--save_predictions", "--revision_name", "chck_1M",
            "--batch_size", "64", "--output_dir", str(outdir),
        ], env, cwd=STRICT_DIR, logf=logf)
        report = outdir / "hf_model" / "chck_1M" / "zero_shot" / "mlm" / task / task_name / "best_temperature_report.txt"
        scores[task_name] = read_avg(report); reports[task_name] = str(report)
    run([
        sys.executable, "-m", "evaluation_pipeline.reading.run",
        "--model_path_or_name", str(model_path), "--backend", "mlm",
        "--data_path", "evaluation_data/fast_eval/reading/reading_data.csv",
        "--revision_name", "chck_1M", "--output_dir", str(outdir),
    ], env, cwd=STRICT_DIR, logf=logf)
    rreport = outdir / "hf_model" / "chck_1M" / "zero_shot" / "mlm" / "reading" / "report.txt"
    scores.update(read_reading(rreport)); reports["reading"] = str(rreport)
    return {"run_id": run_id, "scores": scores, "reports": reports}


def fmt(x):
    return f"{x:.2f}" if isinstance(x, float) else str(x)


def main() -> None:
    env = setup_env()
    meta = load_json(META)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    train_rows, profile_rows, sup_cmp = {}, {}, {}
    with LOG.open("a", encoding="utf-8") as logf:
        for seed_str, seed_meta in meta["seeds"].items():
            seed = int(seed_str)
            assert seed_meta["actual_total_words"] == 1_000_000
            assert seed_meta["validation"]["hard_validation_ok"] is True
            assert seed_meta["validation"]["combined_kept_token_delta_total"] == 0
            for arm in ["coherent", "corrupted"]:
                row = train_and_check(seed, arm, seed_meta, env, logf)
                train_rows[row["run_id"]] = row
            coh_id = run_id_for("coherent", seed); cor_id = run_id_for("corrupted", seed)
            cmp = compare_supervision(train_rows[coh_id]["supervision_manifest"], train_rows[cor_id]["supervision_manifest"])
            if not cmp["ok"]:
                raise RuntimeError(f"seed {seed} supervision mismatch: {cmp}")
            sup_cmp[str(seed)] = cmp
        for run_id in list(train_rows.keys()):
            profile_rows[run_id] = profile(run_id, env, logf)

    per_seed_scores, deltas = {}, {}
    for seed_str in meta["seeds"]:
        seed = int(seed_str)
        coh_id = run_id_for("coherent", seed); cor_id = run_id_for("corrupted", seed)
        coh = profile_rows[coh_id]["scores"]; cor = profile_rows[cor_id]["scores"]
        per_seed_scores[seed_str] = {"state_coherent": coh, "state_corrupted": cor}
        deltas[seed_str] = {c: round(coh[c] - cor[c], 4) for c in COLS}
    mean_delta = {c: round(sum(deltas[s][c] for s in deltas) / len(deltas), 4) for c in COLS}
    train_summary = []
    for run_id, row in train_rows.items():
        m = row["metrics"]; s = row["supervision_summary"]
        train_summary.append({
            "run_id": run_id, "seed": row["seed"], "arm": row["arm"],
            "parameter_count": m["parameter_count"], "loss_first": m["loss_first"], "loss_last": m["loss_last"],
            "word_exposure": m["word_exposure"], "steps": m["actual_training_steps"],
            "selected_tokens_per_word": m["selected_tokens_per_whitespace_word"],
            "crossview_tokens_total": m["crossview_tokens_total"], "wwm_tokens_total": m["wwm_tokens_total"],
            "crossview_rows_total": m["crossview_rows_total"], "wwm_rows_total": m["wwm_rows_total"],
            "truncated_examples": s["truncated_examples_at_max_seq_length"], "tokens_lost_to_truncation": s["total_tokens_lost_to_truncation"],
            "anchor_groups_total": s["anchor_groups_total"], "anchor_tokens_total": s["anchor_tokens_total"],
        })
    payload = {"state_materialization_meta": str(META), "per_seed_supervision_identity": sup_cmp, "train_summary": train_summary,
               "per_seed_scores": per_seed_scores, "state_coherent_minus_state_corrupted_by_seed": deltas,
               "mean_state_coherent_minus_state_corrupted": mean_delta}
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_TRAIN_JSON.write_text(json.dumps(train_summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    OUT_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = [
        "# research — procedural state stories: coherent vs corrupted",
        "",
        f"Evidence JSON: `{OUT_JSON}`",
        "",
        "900k official words + 100k hand-coded procedural state stories. Coherent/corrupted arms share official examples, story templates, target answer multiset, target mask positions, word counts, kept-token totals, and update geometry; corrupted swaps decisive bindings so events no longer support the recorded target state.",
        "",
        "| seed | arm | BLiMP | Supp | EWoK | Entity | COMPS | Read eye | Read SPR |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for seed_str in meta["seeds"]:
        for key in ["state_coherent", "state_corrupted"]:
            sc = per_seed_scores[seed_str][key]
            lines.append(f"| {seed_str} | {key} | {fmt(sc['blimp_fast'])} | {fmt(sc['supplement_fast'])} | {fmt(sc['ewok_fast'])} | {fmt(sc['entity_tracking_fast'])} | {fmt(sc['comps'])} | {fmt(sc['reading_eye_tracking'])} | {fmt(sc['reading_self_paced'])} |")
    lines += ["", "## coherent minus corrupted", "", "| seed | BLiMP | Supp | EWoK | Entity | COMPS | Read eye | Read SPR |", "|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for seed_str in meta["seeds"]:
        d = deltas[seed_str]
        lines.append(f"| {seed_str} | {fmt(d['blimp_fast'])} | {fmt(d['supplement_fast'])} | {fmt(d['ewok_fast'])} | {fmt(d['entity_tracking_fast'])} | {fmt(d['comps'])} | {fmt(d['reading_eye_tracking'])} | {fmt(d['reading_self_paced'])} |")
    lines += ["", "## mean delta", "", "| BLiMP | Supp | EWoK | Entity | COMPS | Read eye | Read SPR |", "|---:|---:|---:|---:|---:|---:|---:|", f"| {fmt(mean_delta['blimp_fast'])} | {fmt(mean_delta['supplement_fast'])} | {fmt(mean_delta['ewok_fast'])} | {fmt(mean_delta['entity_tracking_fast'])} | {fmt(mean_delta['comps'])} | {fmt(mean_delta['reading_eye_tracking'])} | {fmt(mean_delta['reading_self_paced'])} |"]
    OUT_NOTE.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"mean_state_coherent_minus_state_corrupted": mean_delta, "by_seed": deltas, "supervision_identity": sup_cmp, "out": str(OUT_JSON)}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
