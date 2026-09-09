#!/usr/bin/env python3
"""research/120 fixed readouts for staged PVDM 80M continuations.

This wrapper registers the two staged checkpoints into already validated readers:
  * Supplement + Entity official-compatible zero-shot wrapper from research;
  * GlobalPIQA all-option margin reader from research/108;
  * EWoK four-cell interaction reader from research.

It never launches training and never evaluates parent smoke directories.  Each
model path is explicitly `hf_model/chck_80M` under the staged 70M->80M run.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(".").resolve()
A01_WS = ROOT / "experiments/archive/representation_and_objectives"
SCRIPT_DIR = A01_WS / "scripts"
AI_SCRIPT_DIR = A01_WS / "training/scripts"
OUT_ROOT = A01_WS / "data/pvdm_80m_readouts"
FULL_ROOT = OUT_ROOT / "supplement_entity"
GP_ROOT = OUT_ROOT / "globalpiqa_margin"
EWOK_ROOT = OUT_ROOT / "ewok_fourcell"
NOTE = (ROOT / 'research/notes/representation_and_objectives/pvdm_80m_readout_plan.md')

TARGETS: dict[str, dict[str, Any]] = {
    "pvdm_treatment_80m": {
        "run_dir": A01_WS / "training/runs/pvdm_treatment_70M_to_80M_seed43022",
        "model_path": A01_WS / "training/runs/pvdm_treatment_70M_to_80M_seed43022/hf_model/chck_80M",
        "label": "PVDM treatment: true relational pivot visible, matched surrogate masked, staged 80M",
        "endpoint": "chck_80M",
    },
    "pvdm_control_80m": {
        "run_dir": A01_WS / "training/runs/pvdm_control_70M_to_80M_seed43022",
        "model_path": A01_WS / "training/runs/pvdm_control_70M_to_80M_seed43022/hf_model/chck_80M",
        "label": "PVDM matched control: surrogate visible, true relational pivot masked, same dependent targets, staged 80M",
        "endpoint": "chck_80M",
    },
    "compact_80m_reference": {
        "run_dir": ROOT / "experiments/archive/frontier_consolidation/training/runs/fw_compact_view_shared16k_seed43022",
        "model_path": ROOT / "experiments/archive/frontier_consolidation/training/runs/fw_compact_view_shared16k_seed43022/hf_model/chck_80M",
        "label": "A02 compact full-batch reference at archived chck_80M; not optimizer-reset matched but useful trajectory anchor",
        "endpoint": "chck_80M",
    },
}


def now_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def import_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def preflight(targets: list[str]) -> dict[str, Any]:
    out = {"status": "READY", "created_utc": now_utc(), "targets": {}}
    for t in targets:
        spec = TARGETS[t]
        mp = Path(spec["model_path"])
        rd = Path(spec["run_dir"])
        mpath = rd / "scientific_metrics.json"
        rec: dict[str, Any] = {
            "run_dir": str(rd), "run_dir_exists": rd.exists(),
            "model_path": str(mp), "model_path_exists": mp.exists(),
            "scientific_metrics": str(mpath), "scientific_metrics_exists": mpath.exists(),
            "errors": [],
        }
        if not rd.exists(): rec["errors"].append("missing run_dir")
        if not mp.exists(): rec["errors"].append("missing hf_model/chck_80M")
        if mpath.exists():
            try:
                m = load_json(mpath)
                rec["metrics_summary"] = {
                    "word_exposure": m.get("word_exposure"),
                    "continuation_words": m.get("continuation_words"),
                    "actual_training_steps": m.get("actual_training_steps"),
                    "loss_first": m.get("loss_first"),
                    "loss_last": m.get("loss_last"),
                    "parameter_count": m.get("parameter_count"),
                    "vocab_size": m.get("vocab_size"),
                    "stage_stop_name": m.get("stage_stop_name"),
                }
                if t.startswith("pvdm_"):
                    if m.get("stage_stop_name") != "chck_80M": rec["errors"].append("stage_stop_name not chck_80M")
                    if m.get("word_exposure") != 80011326: rec["errors"].append(f"word_exposure {m.get('word_exposure')} != 80011326")
            except Exception as exc:
                rec["errors"].append(f"cannot parse scientific_metrics: {exc}")
        elif t.startswith("pvdm_"):
            rec["errors"].append("missing scientific_metrics")
        if rec["errors"]: out["status"] = "NOT_READY"
        out["targets"][t] = rec
    return out


def run_command(cmd: list[str], log: Path, gpu: int | None = None, force_cpu: bool = False) -> dict[str, Any]:
    log.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["TOKENIZERS_PARALLELISM"] = "false"
    if force_cpu:
        env["CUDA_VISIBLE_DEVICES"] = ""
    elif gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    t0 = time.time()
    with log.open("a", encoding="utf-8") as f:
        f.write(f"\n[{now_utc()}] $ {' '.join(cmd)}\n")
        proc = subprocess.run(cmd, cwd=str(ROOT), env=env, stdout=f, stderr=subprocess.STDOUT, text=True)
        elapsed = time.time() - t0
        f.write(f"[returncode={proc.returncode} elapsed_sec={elapsed:.3f}]\n")
    return {"cmd": cmd, "returncode": proc.returncode, "elapsed_sec": round(elapsed, 3), "log": str(log)}


def run_supp_entity(targets: list[str], gpu: int, force: bool) -> dict[str, Any]:
    mod = import_module(AI_SCRIPT_DIR / "full_eval_fw_shared_anchor.py", "full_eval_pvdm_adapter")
    mod.OUT_ROOT = FULL_ROOT
    mod.PER_TARGET_DIR = FULL_ROOT / "per_target"
    for t in targets:
        spec = TARGETS[t]
        mod.TARGET_REGISTRY[t] = {
            "run_dir": Path(spec["run_dir"]),
            "endpoint": spec["endpoint"],
            "description": spec["label"],
            "family": "pvdm_staged_80m" if t.startswith("pvdm_") else "reference_80m",
        }
    runs = []
    for t in targets:
        if t == "compact_80m_reference":
            # research already has its cheap columns; still allow explicit rerun if requested by caller.
            pass
        cmd = [sys.executable, "-B", str(AI_SCRIPT_DIR / "full_eval_fw_shared_anchor.py"), "--target", t, "--endpoint", "chck_80M", "--gpu", str(gpu), "--columns", "Supplement", "Entity"]
        # The standalone script does not know our injected registry, so use in-process main instead.
        old_argv = sys.argv
        try:
            sys.argv = ["full_eval_fw_shared_anchor.py", "--target", t, "--endpoint", "chck_80M", "--gpu", str(gpu), "--columns", "Supplement", "Entity"] + (["--force"] if force else [])
            t0 = time.time(); mod.main(); elapsed = time.time() - t0
            runs.append({"target": t, "returncode": 0, "elapsed_sec": round(elapsed, 3), "mode": "in_process_step106_adapter"})
        finally:
            sys.argv = old_argv
    return {"runs": runs, "out_root": str(FULL_ROOT)}


def run_globalpiqa(targets: list[str], max_items: int | None = None, threads: int = 8) -> dict[str, Any]:
    mod = import_module(SCRIPT_DIR / "globalpiqa_margin_reader.py", "globalpiqa_pvdm_adapter")
    mod.OUT_ROOT = GP_ROOT
    mod.NOTE = (ROOT / 'research/notes/representation_and_objectives/pvdm_80m_globalpiqa_margin_reader.md')
    for t in targets:
        spec = TARGETS[t]
        mod.TARGETS[t] = {"label": spec["label"], "model_root": Path(spec["run_dir"]) / "hf_model", "revision": "chck_80M", "official_parallel": None, "official_nonparallel": None}
    old_argv = sys.argv
    try:
        sys.argv = ["globalpiqa_margin_reader.py", "--targets", *targets, "--modes", "parallel", "nonparallel", "--batch_size", "8", "--non_causal_batch_size", "32", "--threads", str(threads)]
        if max_items is not None:
            sys.argv += ["--max_items", str(max_items)]
        t0 = time.time(); mod.main(); elapsed = time.time() - t0
    finally:
        sys.argv = old_argv
    return {"returncode": 0, "elapsed_sec": round(elapsed, 3), "out_root": str(GP_ROOT)}


def run_ewok(targets: list[str], device: str, row_limit: int, threads: int) -> dict[str, Any]:
    mod = import_module(SCRIPT_DIR / "fw_ewok_interaction_reader.py", "ewok_pvdm_adapter")
    mod.OUT_ROOT = EWOK_ROOT
    mod.NOTE = (ROOT / 'research/notes/representation_and_objectives/pvdm_80m_ewok_fourcell_reader.md')
    mod.DEFAULT_TARGETS.clear()
    for t in targets:
        spec = TARGETS[t]
        mod.DEFAULT_TARGETS[t] = {"model_path": Path(spec["model_path"]), "label": spec["label"]}
    old_argv = sys.argv
    try:
        sys.argv = ["fw_ewok_interaction_reader.py", "--targets", *targets, "--device", device, "--threads", str(threads), "--row_batch_size", "64", "--masked_batch_size", "128"]
        if row_limit:
            sys.argv += ["--row_limit", str(row_limit)]
        t0 = time.time(); mod.main(); elapsed = time.time() - t0
    finally:
        sys.argv = old_argv
    return {"returncode": 0, "elapsed_sec": round(elapsed, 3), "out_root": str(EWOK_ROOT)}


def synthesize(targets: list[str]) -> dict[str, Any]:
    rows: dict[str, Any] = {}
    archived_compact_80m = ROOT / "experiments/archive/frontier_consolidation/data/fw_comparison_eval/per_target/compact_view_chck_80M.json"
    for t in targets:
        rec: dict[str, Any] = {"target": t}
        p_full = FULL_ROOT / "per_target" / f"{t}.json"
        if (not p_full.exists()) and t == "compact_80m_reference" and archived_compact_80m.exists():
            p_full = archived_compact_80m
            rec["Supplement_Entity_source"] = "archived_A02_step086_compact_chck_80M"
        if p_full.exists():
            d = load_json(p_full)
            tasks = d.get("tasks", {})
            rec["Supplement"] = tasks.get("Supplement", {}).get("score")
            rec["Entity"] = tasks.get("Entity", {}).get("score")
            if t == "compact_80m_reference":
                rec["archived_official_EWoK"] = tasks.get("EWoK", {}).get("score")
                gp_p = tasks.get("GlobalPIQA_parallel", {}).get("score")
                gp_np = tasks.get("GlobalPIQA_nonparallel", {}).get("score")
                if isinstance(gp_p, (int, float)) and isinstance(gp_np, (int, float)):
                    rec["archived_official_GlobalPIQA_mean"] = (gp_p + gp_np) / 2.0
        p_gp_candidates = [GP_ROOT / f"{t}_margins.json", GP_ROOT / f"{t}_summary.json"]
        p_gp = next((p for p in p_gp_candidates if p.exists()), None)
        if p_gp is not None:
            gd = load_json(p_gp)
            modes = gd.get("modes", {})
            par = modes.get("parallel", {}).get("summary", modes.get("parallel", {}))
            non = modes.get("nonparallel", {}).get("summary", modes.get("nonparallel", {}))
            rec["GlobalPIQA_parallel"] = par.get("accuracy")
            rec["GlobalPIQA_nonparallel"] = non.get("accuracy")
            aw = par.get("always_wrong_subset") or {}
            rec["GlobalPIQA_parallel_hard52_mean_top_minus_correct"] = aw.get("mean_top_minus_correct")
            rec["GlobalPIQA_parallel_hard52_rank_counts"] = aw.get("correct_rank_counts")
        p_ew = EWOK_ROOT / t / "ewok_interaction_summary.json"
        if p_ew.exists():
            ed = load_json(p_ew); s = ed.get("summary", {})
            rec["EWoK_fourcell_accuracy"] = s.get("accuracy")
            rec["EWoK_stable_failure_frac_wrong"] = s.get("stable_failure_frac_wrong")
            rec["EWoK_interaction_sum_wrong_median"] = s.get("interaction_sum_wrong", {}).get("median")
        rows[t] = rec
    comp = {"status": "PVDM_80M_READOUT_SYNTHESIS", "created_utc": now_utc(), "targets": rows}
    if "pvdm_treatment_80m" in rows and "pvdm_control_80m" in rows:
        delta = {}
        a = rows["pvdm_treatment_80m"]; b = rows["pvdm_control_80m"]
        for k, v in a.items():
            if k == "target": continue
            if isinstance(v, (int, float)) and isinstance(b.get(k), (int, float)):
                delta[k] = v - b[k]
        comp["treatment_minus_control"] = delta
    out = OUT_ROOT / "pvdm_80m_readout_synthesis.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(comp, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return {"synthesis": str(out), "payload": comp}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", nargs="+", default=["pvdm_treatment_80m", "pvdm_control_80m"], choices=sorted(TARGETS))
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--skip-supp-entity", action="store_true")
    ap.add_argument("--skip-globalpiqa", action="store_true")
    ap.add_argument("--skip-ewok", action="store_true")
    ap.add_argument("--ewok-device", choices=["cpu", "cuda"], default="cpu")
    ap.add_argument("--ewok-row-limit", type=int, default=0)
    ap.add_argument("--globalpiqa-max-items", type=int, default=None)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    pf = preflight(args.targets)
    pf_path = OUT_ROOT / "preflight.json"
    pf_path.write_text(json.dumps(pf, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    NOTE.write_text(
        "# research — staged PVDM 80M readout plan\n\n"
        "This wrapper evaluates only explicit `hf_model/chck_80M` checkpoints from the staged treatment/control runs. "
        "The fixed readouts are Supplement, Entity, GlobalPIQA all-option margins, and EWoK four-cell. It is prepared before the training tasks finish so no new metric is invented after seeing results.\n\n"
        f"Preflight: `{pf_path}`\n",
        encoding="utf-8",
    )
    if args.dry_run:
        print(json.dumps({"status": "DRY_RUN", "ready": pf["status"], "preflight": str(pf_path)}, indent=2), flush=True)
        return
    if pf["status"] != "READY":
        raise RuntimeError({"preflight": str(pf_path), "errors": {k:v['errors'] for k,v in pf['targets'].items() if v['errors']}})
    runs: dict[str, Any] = {}
    if not args.skip_supp_entity:
        runs["supplement_entity"] = run_supp_entity(args.targets, args.gpu, args.force)
    if not args.skip_globalpiqa:
        runs["globalpiqa"] = run_globalpiqa(args.targets, args.globalpiqa_max_items, args.threads)
    if not args.skip_ewok:
        runs["ewok"] = run_ewok(args.targets, args.ewok_device, args.ewok_row_limit, args.threads)
    syn = synthesize(args.targets)
    payload = {"status": "PVDM_80M_READOUTS_DONE", "created_utc": now_utc(), "targets": args.targets, "preflight": str(pf_path), "runs": runs, "synthesis": syn["synthesis"]}
    out = OUT_ROOT / "pvdm_80m_readout_run_summary.json"
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "synthesis": syn["synthesis"], "run_summary": str(out), "delta": syn["payload"].get("treatment_minus_control")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
