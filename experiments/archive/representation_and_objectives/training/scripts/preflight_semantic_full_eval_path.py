#!/usr/bin/env python3
"""Preflight the selected semantic-view full-evaluation path without launching evaluation.

This CPU-only check protects the next decision after the no-AoA semantic-view
trajectory.  It checks that the REPRESENTATION_FRONTIER_STUDIES wrapper scripts parse, the COMPACT_EXPERIENCE inherited
full evaluator and repaired AoA helper are available, BabyLM Overall scoring uses
the corrected AoA leaderboard-unit convention, and both semantic-view model roots
have the complete 1M..100M checkpoint ladder needed for official strict-small AoA.
"""
from __future__ import annotations

import ast
import json
import pathlib
import subprocess
import sys
from typing import Any

ROOT = pathlib.Path("experiments/archive/representation_and_objectives")
SCRIPTS = ROOT / "training" / "scripts"
COMPACT_EXPERIENCE_SCRIPTS = pathlib.Path("experiments/archive/compact_experience/scripts")
RUNS = ROOT / "training" / "runs"
OUT_DIR = ROOT / "data" / "semantic_full_eval_preflight"
NOTE = (ROOT / 'notes'.parents[3] / 'research/notes/representation_and_objectives/semantic_full_eval_preflight.md')

PY_FILES = [
    SCRIPTS / "prefill_semantic_view_full_eval_from_noaoa.py",
    SCRIPTS / "semantic_view_full_eval_runner.py",
    SCRIPTS / "summarize_semantic_view_full_eval.py",
    COMPACT_EXPERIENCE_SCRIPTS / "full_overall_eval_runner.py",
    COMPACT_EXPERIENCE_SCRIPTS / "babylm_official_scoring.py",
    COMPACT_EXPERIENCE_SCRIPTS / "aoa_local_ckpts_for_model.py",
]
SH_FILES = [
    SCRIPTS / "launch_semantic_view_full_eval_selected.sh",
]
RUN_INFO = {
    "semantic_view_treatment": RUNS / "semantic_view_treatment_8x480_16k_wwm_seed43022",
    "original_packet_local": RUNS / "original_packet_local_8x480_16k_wwm_seed43022",
}
AOA_REQUIRED = [f"chck_{i}M" for i in range(1, 10)] + [f"chck_{i}M" for i in range(10, 101, 10)]
ALL_REQUIRED = [f"chck_{i}M" for i in range(1, 101)]


def ast_check(path: pathlib.Path) -> dict[str, Any]:
    rec = {"path": str(path), "exists": path.exists(), "ok": False}
    if not path.exists():
        rec["error"] = "missing"
        return rec
    try:
        ast.parse(path.read_text(encoding="utf-8"))
        rec["ok"] = True
    except Exception as exc:
        rec["error"] = repr(exc)
    return rec


def bash_check(path: pathlib.Path) -> dict[str, Any]:
    rec = {"path": str(path), "exists": path.exists(), "ok": False}
    if not path.exists():
        rec["error"] = "missing"
        return rec
    proc = subprocess.run(["bash", "-n", str(path)], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    rec.update({"returncode": proc.returncode, "stdout": proc.stdout[-1000:], "stderr": proc.stderr[-1000:], "ok": proc.returncode == 0})
    return rec


def checkpoint_check(run: pathlib.Path) -> dict[str, Any]:
    model_root = run / "hf_model"
    existing = sorted([p.name for p in model_root.iterdir() if p.is_dir()]) if model_root.exists() else []
    existing_set = set(existing)
    all_missing = [x for x in ALL_REQUIRED if x not in existing_set]
    aoa_missing = [x for x in AOA_REQUIRED if x not in existing_set]
    metrics = run / "scientific_metrics.json"
    summary: dict[str, Any] = {}
    if metrics.exists():
        m = json.loads(metrics.read_text(encoding="utf-8"))
        summary = {
            "word_exposure": m.get("word_exposure"),
            "actual_training_steps": m.get("actual_training_steps"),
            "parameter_count": m.get("parameter_count"),
            "vocab_size": m.get("vocab_size"),
            "tokenizer_label": m.get("tokenizer_label"),
            "loss_last": m.get("loss_last"),
            "example_jsonl": m.get("example_jsonl"),
        }
    return {
        "run": str(run),
        "model_root": str(model_root),
        "metrics_exists": metrics.exists(),
        "metrics_summary": summary,
        "checkpoint_dir_count": len(existing),
        "all_1M_to_100M_complete": len(all_missing) == 0,
        "aoa_19_step_ladder_complete": len(aoa_missing) == 0,
        "missing_all_required": all_missing,
        "missing_aoa_required": aoa_missing,
    }


def scoring_self_test() -> dict[str, Any]:
    sys.path.insert(0, str(COMPACT_EXPERIENCE_SCRIPTS.resolve()))
    import babylm_official_scoring as scoring  # noqa: WPS433
    out = scoring.self_test()
    return {"ok": True, "result": out, "official_keys": scoring.OFFICIAL_OVERALL_KEYS, "aoa_scale": scoring.AOA_SCALE}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    py = {p.name: ast_check(p) for p in PY_FILES}
    sh = {p.name: bash_check(p) for p in SH_FILES}
    ck = {name: checkpoint_check(run) for name, run in RUN_INFO.items()}
    score = scoring_self_test()

    prefill_script = SCRIPTS / "prefill_semantic_view_full_eval_from_noaoa.py"
    runner_script = SCRIPTS / "semantic_view_full_eval_runner.py"
    prefill_text = prefill_script.read_text(encoding="utf-8") if prefill_script.exists() else ""
    runner_text = runner_script.read_text(encoding="utf-8") if runner_script.exists() else ""
    structural = {
        "prefill_overrides_base_out_root": "base.OUT_ROOT = OUT_ROOT" in prefill_text and "base.PER_TARGET_DIR = OUT_ROOT / \"per_target\"" in prefill_text,
        "prefill_records_noaoa_source": "prefill_source" in prefill_text and "non_leakage_statement" in prefill_text,
        "runner_overrides_base_targets": "base.TARGETS" in runner_text and "REPRESENTATION_FRONTIER_STUDIES_semantic_view_packet_local_contrast" in runner_text,
        "runner_has_columns_subset": "--columns" in runner_text,
        "compact_experience_repaired_aoa_helper_available": (COMPACT_EXPERIENCE_SCRIPTS / "aoa_local_ckpts_for_model.py").exists(),
    }
    all_ok = (
        all(v.get("ok") for v in py.values())
        and all(v.get("ok") for v in sh.values())
        and all(v.get("aoa_19_step_ladder_complete") and v.get("metrics_exists") for v in ck.values())
        and score.get("ok") is True
        and all(structural.values())
    )
    payload = {
        "status": "SEMANTIC_FULL_EVAL_PREFLIGHT",
        "all_ok": all_ok,
        "python_ast_checks": py,
        "bash_syntax_checks": sh,
        "checkpoint_checks": ck,
        "scoring_self_test": score,
        "structural_checks": structural,
        "full_eval_if_noaoa_positive": {
            "launcher": str(SCRIPTS / "launch_semantic_view_full_eval_selected.sh"),
            "output_root": "experiments/archive/representation_and_objectives/data/semantic_view_full_eval",
            "recommended_endpoint_policy": "If no-AoA delta is meaningful, evaluate the treatment endpoint selected by no-AoA and the packet-local control at the same endpoint for causal full-Overall delta; if the treatment's no-AoA maximum is at another endpoint, consider an additional treatment-only full eval only after the first full measurement.",
        },
    }
    out_json = OUT_DIR / "semantic_full_eval_preflight.json"
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    lines = ["# research semantic-view selected full-eval preflight\n\n"]
    lines.append(f"All checked components ready: **{all_ok}**. This did not launch evaluation.\n\n")
    lines.append("## Checks\n\n")
    lines.append(f"- Python script AST checks all pass: {all(v.get('ok') for v in py.values())}.\n")
    lines.append(f"- Shell launch syntax pass: {all(v.get('ok') for v in sh.values())}.\n")
    lines.append(f"- Corrected BabyLM scoring self-test pass: {score.get('ok')}, AoA scale {score.get('aoa_scale')}.\n")
    lines.append(f"- COMPACT_EXPERIENCE repaired AoA helper available: {structural['compact_experience_repaired_aoa_helper_available']}.\n")
    for name, rec in ck.items():
        ms = rec.get("metrics_summary", {})
        lines.append(f"- `{name}`: AoA 19-step ladder complete={rec['aoa_19_step_ladder_complete']}, all 1M..100M checkpoints={rec['all_1M_to_100M_complete']}, exposure={ms.get('word_exposure')}, last_loss={ms.get('loss_last')}.\n")
    lines.append("\n## Use after no-AoA\n\n")
    lines.append("If the semantic-view no-AoA trajectory shows a meaningful treatment gain, run selected full official-style evaluation with `TREAT_ENDPOINT=<ckpt> PACKET_ENDPOINT=<same_ckpt> bash experiments/archive/representation_and_objectives/training/scripts/launch_semantic_view_full_eval_selected.sh`. The zero-shot/Reading values will be prefilled from the frozen no-AoA run; SuperGLUE and AoA will be measured.\n\n")
    lines.append(f"JSON: `{out_json}`\n")
    NOTE.write_text("".join(lines), encoding="utf-8")
    print(json.dumps({"status": payload["status"], "all_ok": all_ok, "out": str(out_json), "note": str(NOTE)}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
