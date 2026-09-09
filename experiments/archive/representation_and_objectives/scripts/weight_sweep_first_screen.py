#!/usr/bin/env python3
"""research: first-screen readout for same-initialization FW mixture checkpoints.

This is the minimum post-hoc compatibility test before any new 100M branch or full
BabyLM endpoint evaluation.  It evaluates only broad-preservation sentinels
(Supplement, Entity), a current-coordinate EWoK aggregate, and GlobalPIQA all-option
row/margin readouts for a small predetermined mixture set from
`fw_weight_space_sweep.py`.

No training is performed.  The mixture coefficients were fixed before reading these
results; do not use this script for fine-grained lambda tuning on official rows.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(".").resolve()
WS = ROOT / "experiments/archive/representation_and_objectives"
A02_WS = ROOT / "experiments/archive/frontier_consolidation"
COMPACT_EXPERIENCE_SCRIPTS = ROOT / "experiments/archive/compact_experience/scripts"
A01_AI_SCRIPTS = WS / "training/scripts"
A01_SCRIPTS = WS / "scripts"
PRISTINE_STRICT = WS / "data/pristine_official_coordinate/babylm-eval/strict"
OUT_ROOT = WS / "data/weight_sweep_first_screen"
ZERO_OUT = OUT_ROOT / "zero_surface"
EWOK_OUT = OUT_ROOT / "official_ewok"
MARGIN_OUT = OUT_ROOT / "globalpiqa_margin"
LOG_ROOT = OUT_ROOT / "logs"
NOTE = (ROOT / 'research/notes/representation_and_objectives/weight_sweep_first_screen.md')

MIXTURE_RUN_ROOT = WS / "training/runs/fw_weight_space_sweep"
MIXTURES = {
    "ci_a0p25": {"run_dir": MIXTURE_RUN_ROOT / "ci_a0p25", "endpoint": "chck_100M", "label": "0.75 compact + 0.25 interleaved"},
    "ci_a0p50": {"run_dir": MIXTURE_RUN_ROOT / "ci_a0p50", "endpoint": "chck_100M", "label": "0.50 compact + 0.50 interleaved"},
    "ci_a0p75": {"run_dir": MIXTURE_RUN_ROOT / "ci_a0p75", "endpoint": "chck_100M", "label": "0.25 compact + 0.75 interleaved"},
    "cr_a0p25": {"run_dir": MIXTURE_RUN_ROOT / "cr_a0p25", "endpoint": "chck_100M", "label": "0.75 compact + 0.25 row-block"},
    "cir_i0p25_r0p25": {"run_dir": MIXTURE_RUN_ROOT / "cir_i0p25_r0p25", "endpoint": "chck_100M", "label": "0.50 compact + 0.25 interleaved + 0.25 row-block"},
}

# Completed endpoint readouts used only for comparison.
ENDPOINTS = {
    "compact": {
        "label": "A02 compact endpoint",
        "Supplement": 58.86,
        "Entity": 28.36,
        "EWoK": 50.25,
        "GlobalPIQA_parallel": 24.27,
        "GlobalPIQA_nonparallel": 53.0,
        "GlobalPIQA": 38.635,
        "hard52_accuracy": 3.8461538461538463,
        "hard52_mean_top_minus_correct": 1.7169,
        "stable_failure_frac_wrong": 0.6890889830508474,
        "cheap7": 43.18142857142857,
    },
    "interleaved": {
        "label": "A01 interleaved endpoint",
        "Supplement": 56.42,
        "Entity": 25.60,
        "EWoK": 51.85,
        "GlobalPIQA_parallel": 26.21,
        "GlobalPIQA_nonparallel": 56.0,
        "GlobalPIQA": 41.105,
        "hard52_accuracy": 3.8461538461538463,
        "hard52_mean_top_minus_correct": 1.660,
        "stable_failure_frac_wrong": None,
        "cheap7": 43.07928571428572,
    },
    "rowblock": {
        "label": "A02 row-block endpoint",
        "Supplement": 59.69,
        "Entity": 23.88,
        "EWoK": 50.50,
        "GlobalPIQA_parallel": 29.13,
        "GlobalPIQA_nonparallel": 45.0,
        "GlobalPIQA": 37.065,
        "hard52_accuracy": 5.769230769230769,
        "hard52_mean_top_minus_correct": 1.4333,
        "stable_failure_frac_wrong": 0.6457574180114524,
        "cheap7": 42.63857142857143,
    },
}

FIRST_SCREEN_COLUMNS = ["Supplement", "Entity"]


def now_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def run_logged(name: str, cmd: list[str], log: Path, gpu: int | None, timeout: int) -> dict[str, Any]:
    env = os.environ.copy()
    env["TOKENIZERS_PARALLELISM"] = "false"
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    log.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with log.open("a", encoding="utf-8") as f:
        f.write(f"\n[{now_utc()}] $ {' '.join(cmd)}\n")
        proc = subprocess.run(cmd, cwd=str(ROOT), env=env, stdout=f, stderr=subprocess.STDOUT, text=True, timeout=timeout)
        elapsed = time.time() - t0
        f.write(f"[returncode={proc.returncode} elapsed_sec={elapsed:.3f}]\n")
    return {"name": name, "cmd": cmd, "returncode": proc.returncode, "elapsed_sec": round(elapsed, 3), "log": str(log)}


def import_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def configure_zero_module():
    if str(COMPACT_EXPERIENCE_SCRIPTS) not in sys.path:
        sys.path.insert(0, str(COMPACT_EXPERIENCE_SCRIPTS))
    import full_overall_eval_runner as base  # type: ignore

    base.STRICT = PRISTINE_STRICT
    base.OUT_ROOT = ZERO_OUT
    base.PER_TARGET_DIR = ZERO_OUT / "per_target"
    base.TARGETS = {
        name: {
            "run_dir": spec["run_dir"],
            "endpoint": spec["endpoint"],
            "description": spec["label"],
            "family": "same_initialization_mixture",
        }
        for name, spec in MIXTURES.items()
    }
    base.ZERO_BY_COL = {t["column"]: t for t in base.ZERO_SHOT_TASKS}
    return base


def preflight_targets(targets: list[str]) -> dict[str, Any]:
    out = {"status": "READY", "created_utc": now_utc(), "targets": {}}
    for t in targets:
        spec = MIXTURES[t]
        model = Path(spec["run_dir"]) / "hf_model" / spec["endpoint"]
        files = {name: (model / name).exists() for name in ["config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"]}
        errs = []
        if not model.exists():
            errs.append(f"missing model path {model}")
        for name, exists in files.items():
            if not exists:
                errs.append(f"missing {name}")
        out["targets"][t] = {"model_path": str(model), "label": spec["label"], "files": files, "errors": errs}
        if errs:
            out["status"] = "NOT_READY"
    return out


def run_zero_surface(targets: list[str], gpu: int, force: bool) -> list[dict[str, Any]]:
    base = configure_zero_module()
    runs = []
    for target in targets:
        old_argv = sys.argv
        try:
            argv = ["full_overall_eval_runner.py", "--target", target, "--gpu", str(gpu), "--columns", *FIRST_SCREEN_COLUMNS]
            if force:
                argv.insert(4, "--force")
            sys.argv = argv
            t0 = time.time()
            base.main()
            runs.append({"target": target, "columns": FIRST_SCREEN_COLUMNS, "returncode": 0, "elapsed_sec": round(time.time() - t0, 3), "out_json": str(ZERO_OUT / "per_target" / f"{target}.json")})
        finally:
            sys.argv = old_argv
    return runs


def run_ewok(targets: list[str], gpu: int, force: bool) -> list[dict[str, Any]]:
    runs = []
    for target in targets:
        model_path = Path(MIXTURES[target]["run_dir"]) / "hf_model" / MIXTURES[target]["endpoint"]
        out_root = EWOK_OUT / target
        out_json = out_root / f"official_ewok_reeval_{target}.json"
        if out_json.exists() and not force:
            rec = load_json(out_json)
            runs.append({"target": target, "status": "existing", "score": rec.get("official_ewok_score"), "out_json": str(out_json)})
            continue
        cmd = [
            sys.executable, "-B", str(A01_AI_SCRIPTS / "official_ewok_reeval.py"),
            "--gpu", str(gpu), "--model-path", str(model_path.resolve()), "--out-root", str(out_root), "--tag", target,
        ]
        rec = run_logged(f"ewok_{target}", cmd, LOG_ROOT / target / "official_ewok.log", gpu, timeout=3600)
        if rec["returncode"] != 0:
            raise RuntimeError(rec)
        payload = load_json(out_json)
        rec.update({"target": target, "score": payload.get("official_ewok_score"), "out_json": str(out_json)})
        runs.append(rec)
    return runs


def run_globalpiqa_margin(targets: list[str], force: bool) -> dict[str, Any]:
    # The base margin reader itself loads to CPU and records all option scores.
    if (MARGIN_OUT / "globalpiqa_margin_reader_results.json").exists() and not force:
        return {"status": "existing", "combined_json": str(MARGIN_OUT / "globalpiqa_margin_reader_results.json")}
    mod = import_module(A01_SCRIPTS / "globalpiqa_margin_reader.py", "globalpiqa_margin_reader_step116")
    mod.OUT_ROOT = MARGIN_OUT
    mod.NOTE = (ROOT / 'research/notes/representation_and_objectives/weight_sweep_globalpiqa_margin.md')
    mixture_targets = {
        name: {
            "label": MIXTURES[name]["label"],
            "model_root": Path(spec["run_dir"]) / "hf_model",
            "revision": spec["endpoint"],
            "official_parallel": None,
            "official_nonparallel": None,
        }
        for name, spec in MIXTURES.items()
    }
    mod.TARGETS.update(mixture_targets)
    old_argv = sys.argv
    try:
        sys.argv = ["globalpiqa_margin_reader.py", "--targets", *targets, "--modes", "parallel", "nonparallel", "--batch_size", "8", "--non_causal_batch_size", "32", "--threads", "12"]
        mod.main()
    finally:
        sys.argv = old_argv
    return {"status": "ran", "combined_json": str(MARGIN_OUT / "globalpiqa_margin_reader_results.json")}


def zero_scores(target: str) -> dict[str, float]:
    p = ZERO_OUT / "per_target" / f"{target}.json"
    d = load_json(p)
    out = {}
    for col in FIRST_SCREEN_COLUMNS:
        rec = d.get("tasks", {}).get(col, {})
        if rec.get("score") is None:
            raise RuntimeError({"missing_zero_score": col, "target": target, "path": str(p), "record": rec})
        out[col] = float(rec["score"])
    return out


def ewok_score(target: str) -> float:
    p = EWOK_OUT / target / f"official_ewok_reeval_{target}.json"
    d = load_json(p)
    val = d.get("official_ewok_score")
    if val is None:
        raise RuntimeError({"missing_ewok_score": target, "path": str(p)})
    return float(val)


def margin_scores(target: str) -> dict[str, Any]:
    p = MARGIN_OUT / f"{target}_margins.json"
    d = load_json(p)
    modes = d["modes"]
    par = modes["parallel"]["summary"]
    non = modes["nonparallel"]["summary"]
    aw = par.get("always_wrong_subset") or {}
    return {
        "GlobalPIQA_parallel": float(par["accuracy"]),
        "GlobalPIQA_nonparallel": float(non["accuracy"]),
        "GlobalPIQA": (float(par["accuracy"]) + float(non["accuracy"])) / 2.0,
        "hard52_accuracy": aw.get("accuracy"),
        "hard52_correct_rank_counts": aw.get("correct_rank_counts"),
        "hard52_mean_top_minus_correct": aw.get("mean_top_minus_correct"),
        "parallel_correct_rank_counts": par.get("correct_rank_counts"),
        "nonparallel_correct_rank_counts": non.get("correct_rank_counts"),
        "margin_json": str(p),
    }


def finite_or_none(x: Any) -> float | None:
    try:
        v = float(x)
        return v if math.isfinite(v) else None
    except Exception:
        return None


def summarize(targets: list[str], run_records: dict[str, Any]) -> dict[str, Any]:
    arms: dict[str, Any] = {}
    for target in targets:
        z = zero_scores(target)
        e = ewok_score(target)
        g = margin_scores(target)
        scores = {**z, "EWoK": e, **{k: v for k, v in g.items() if k.startswith("GlobalPIQA")}}
        broad_pair = (scores["Supplement"] + scores["Entity"]) / 2.0
        relation_pair = (scores["EWoK"] + scores["GlobalPIQA"]) / 2.0
        screen4 = (scores["Supplement"] + scores["Entity"] + scores["EWoK"] + scores["GlobalPIQA"]) / 4.0
        arms[target] = {
            "label": MIXTURES[target]["label"],
            "scores": scores,
            "broad_pair_mean_supp_entity": broad_pair,
            "relation_pair_mean_ewok_globalpiqa": relation_pair,
            "screen4_mean": screen4,
            "hard52_accuracy": g.get("hard52_accuracy"),
            "hard52_mean_top_minus_correct": g.get("hard52_mean_top_minus_correct"),
            "hard52_correct_rank_counts": g.get("hard52_correct_rank_counts"),
            "parallel_correct_rank_counts": g.get("parallel_correct_rank_counts"),
            "nonparallel_correct_rank_counts": g.get("nonparallel_correct_rank_counts"),
            "deltas_vs_compact_endpoint": {k: finite_or_none(scores.get(k)) - ENDPOINTS["compact"][k] for k in ["Supplement", "Entity", "EWoK", "GlobalPIQA", "GlobalPIQA_parallel", "GlobalPIQA_nonparallel"]},
            "deltas_vs_interleaved_endpoint": {k: finite_or_none(scores.get(k)) - ENDPOINTS["interleaved"][k] for k in ["Supplement", "Entity", "EWoK", "GlobalPIQA", "GlobalPIQA_parallel", "GlobalPIQA_nonparallel"]},
            "deltas_vs_rowblock_endpoint": {k: finite_or_none(scores.get(k)) - ENDPOINTS["rowblock"][k] for k in ["Supplement", "Entity", "EWoK", "GlobalPIQA", "GlobalPIQA_parallel", "GlobalPIQA_nonparallel"]},
        }
    # Comparison reference values.
    endpoints = {}
    for name, d in ENDPOINTS.items():
        endpoints[name] = dict(d)
        endpoints[name]["broad_pair_mean_supp_entity"] = (d["Supplement"] + d["Entity"]) / 2.0
        endpoints[name]["relation_pair_mean_ewok_globalpiqa"] = (d["EWoK"] + d["GlobalPIQA"]) / 2.0
        endpoints[name]["screen4_mean"] = (d["Supplement"] + d["Entity"] + d["EWoK"] + d["GlobalPIQA"]) / 4.0
    best = {
        "screen4": max(arms, key=lambda k: arms[k]["screen4_mean"]),
        "broad_pair": max(arms, key=lambda k: arms[k]["broad_pair_mean_supp_entity"]),
        "relation_pair": max(arms, key=lambda k: arms[k]["relation_pair_mean_ewok_globalpiqa"]),
        "globalpiqa_parallel": max(arms, key=lambda k: arms[k]["scores"]["GlobalPIQA_parallel"]),
        "hard52_margin_lowest": min(arms, key=lambda k: arms[k]["hard52_mean_top_minus_correct"] if arms[k]["hard52_mean_top_minus_correct"] is not None else 1e9),
    }
    payload = {
        "status": "WEIGHT_SWEEP_FIRST_SCREEN_DONE",
        "created_utc": now_utc(),
        "boundary": "Fixed small mixture set; no training; no SuperGLUE, AoA, full collation, or fine-grained official-row lambda tuning.",
        "targets": targets,
        "endpoints_reference": endpoints,
        "mixtures": arms,
        "best_mixture_by_readout": best,
        "runs": run_records,
        "interpretation_fields": {
            "broad_pair_mean_supp_entity": "Preservation of compact broad capability sentinels; compact endpoint is 43.61.",
            "relation_pair_mean_ewok_globalpiqa": "Combined EWoK+GlobalPIQA movement; interleaved endpoint is 46.4775 and compact endpoint is 44.4425.",
            "hard52_mean_top_minus_correct": "Lower is better on the research GlobalPIQA_parallel hard52 deep-rank subset.",
        },
    }
    return payload


def write_note(summary: dict[str, Any]) -> None:
    lines = []
    lines.append("# research — first-screen readout for FW weight-space mixtures")
    lines.append("")
    lines.append("This readout tested a fixed, small set of same-initialization parameter mixtures before any new training or full official evaluation. It reads broad preservation through Supplement/Entity, relation movement through current-coordinate EWoK and GlobalPIQA all-option margins, and does not run SuperGLUE/AoA.")
    lines.append("")
    lines.append("## Endpoint references")
    lines.append("")
    lines.append("| endpoint | Supplement | Entity | EWoK | GlobalPIQA | GP-parallel | GP-nonparallel | broad pair | relation pair | hard52 margin |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for name, d in summary["endpoints_reference"].items():
        hm = d.get("hard52_mean_top_minus_correct")
        lines.append(f"| {name} | {d['Supplement']:.2f} | {d['Entity']:.2f} | {d['EWoK']:.2f} | {d['GlobalPIQA']:.3f} | {d['GlobalPIQA_parallel']:.2f} | {d['GlobalPIQA_nonparallel']:.2f} | {d['broad_pair_mean_supp_entity']:.3f} | {d['relation_pair_mean_ewok_globalpiqa']:.3f} | {hm if hm is None else f'{hm:.3f}'} |")
    lines.append("")
    lines.append("## Mixture readout")
    lines.append("")
    lines.append("| mixture | Supplement | Entity | EWoK | GlobalPIQA | GP-parallel | GP-nonparallel | broad pair | relation pair | screen4 | hard52 acc | hard52 margin |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for name, d in summary["mixtures"].items():
        s = d["scores"]
        ha = d.get("hard52_accuracy")
        hm = d.get("hard52_mean_top_minus_correct")
        lines.append(f"| {name} | {s['Supplement']:.2f} | {s['Entity']:.2f} | {s['EWoK']:.2f} | {s['GlobalPIQA']:.3f} | {s['GlobalPIQA_parallel']:.2f} | {s['GlobalPIQA_nonparallel']:.2f} | {d['broad_pair_mean_supp_entity']:.3f} | {d['relation_pair_mean_ewok_globalpiqa']:.3f} | {d['screen4_mean']:.3f} | {ha if ha is None else f'{ha:.2f}'} | {hm if hm is None else f'{hm:.3f}'} |")
    lines.append("")
    lines.append("Best mixture by readout: " + json.dumps(summary["best_mixture_by_readout"], ensure_ascii=False))
    lines.append("")
    lines.append("Interpretation should compare these mixtures to the endpoints rather than choose a fine-grained official-row lambda. A useful branch-consolidation model would keep compact-like Supplement/Entity while moving EWoK/GlobalPIQA and hard52 margins toward the breadth directions.")
    lines.append("")
    lines.append("Files:")
    lines.append(f"- summary JSON: `{OUT_ROOT / 'weight_sweep_first_screen_summary.json'}`")
    lines.append(f"- zero surface: `{ZERO_OUT}`")
    lines.append(f"- EWoK: `{EWOK_OUT}`")
    lines.append(f"- GlobalPIQA margins: `{MARGIN_OUT}`")
    NOTE.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", nargs="*", default=list(MIXTURES), choices=sorted(MIXTURES))
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    pf = preflight_targets(args.targets)
    (OUT_ROOT / "preflight.json").write_text(json.dumps(pf, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if args.dry_run:
        print(json.dumps({"status": "DRY_RUN", "ready": pf["status"], "preflight": str(OUT_ROOT / "preflight.json")}, indent=2), flush=True)
        return
    if pf["status"] != "READY":
        raise RuntimeError(pf)

    runs: dict[str, Any] = {}
    runs["zero_surface"] = run_zero_surface(args.targets, args.gpu, args.force)
    runs["ewok"] = run_ewok(args.targets, args.gpu, args.force)
    runs["globalpiqa_margin"] = run_globalpiqa_margin(args.targets, args.force)
    summary = summarize(args.targets, runs)
    out_json = OUT_ROOT / "weight_sweep_first_screen_summary.json"
    out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_note(summary)
    print(json.dumps({
        "status": summary["status"],
        "best_mixture_by_readout": summary["best_mixture_by_readout"],
        "out_json": str(out_json),
        "note": str(NOTE),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
