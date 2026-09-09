#!/usr/bin/env python3
"""research: evaluate a saved relation-first acquired state on BabyLM fast Cheap7/Entity.

This wrapper is deliberately separate from the research replication run.  research
retrained a transient model and therefore could not connect the acquired
state-selection behavior to broader BabyLM competence.  research saves exact HF
checkpoints.  This script reads those saved states, records model identity with
trust_remote_code, and runs a selected subset of the already repaired research
fast common-screen evaluator (especially Entity, optionally all Cheap7 columns).

The result is a compatibility diagnostic, not an official BabyLM submission:
fast_eval splits are used for BLiMP/Supp/EWoK/Entity/GlobalPIQA and full COMPS,
matching the inherited research research screen.  Full official-style evaluation
must still follow for any surviving SOTA candidate.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import hashlib
import json
import os
import pathlib
import sys
import time
from typing import Any, Dict, Iterable, List, Optional

import torch

ROOT = _public_path('.')
STUDY = _public_path('experiments/archive/functional_learning')
SCRIPTS = _public_path('experiments/archive/functional_learning/scripts')
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import eval_common_screen as common  # noqa: E402

DEFAULT_STEP42_ROOT = _public_path('experiments/archive/functional_learning/data/saved_state_replicate_neutral_threeentity')
DEFAULT_OUT = _public_path('experiments/archive/functional_learning/data/saved_state_cheap7_entity')

ZERO_TASK_BY_COL = {col: (col, task, data, batch) for col, task, data, batch in common.TASKS}
CHEAP7_ZERO_COLUMNS = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA_parallel", "GlobalPIQA_nonparallel"]
CHEAP7_COLUMNS = CHEAP7_ZERO_COLUMNS + ["Reading"]


def rel(path: pathlib.Path | str) -> str:
    try:
        return str(pathlib.Path(path).resolve().relative_to(ROOT))
    except Exception:
        return str(path)


def now_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha256_file(path: pathlib.Path) -> Optional[str]:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path: pathlib.Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def setup_identity_cache(out_root: pathlib.Path, tag: str) -> Dict[str, str]:
    env = os.environ.copy()
    hf = out_root / "hf_cache" / f"{tag}_identity"
    env["HF_HOME"] = str(hf.resolve())
    env["HF_HUB_CACHE"] = str((hf / "hub").resolve())
    env["HF_DATASETS_CACHE"] = str((hf / "datasets").resolve())
    env["TRANSFORMERS_CACHE"] = str((hf / "transformers").resolve())
    env["HF_MODULES_CACHE"] = str((hf / "modules").resolve())
    env["TOKENIZERS_PARALLELISM"] = "false"
    for k in ["HF_HOME", "HF_HUB_CACHE", "HF_DATASETS_CACHE", "TRANSFORMERS_CACHE", "HF_MODULES_CACHE"]:
        pathlib.Path(env[k]).mkdir(parents=True, exist_ok=True)
    os.environ.update({k: env[k] for k in ["HF_HOME", "HF_HUB_CACHE", "HF_DATASETS_CACHE", "TRANSFORMERS_CACHE", "HF_MODULES_CACHE", "TOKENIZERS_PARALLELISM"]})
    return env


def checkpoint_identity(model_dir: pathlib.Path, out_root: pathlib.Path, tag: str) -> Dict[str, Any]:
    setup_identity_cache(out_root, tag)
    # Import Transformers only after redirecting HF_HOME/HF_MODULES_CACHE into the
    # writable local output directory.  If imported earlier, its dynamic-module
    # cache can remain bound to a read-only default and trusted loading fails.
    from transformers import AutoConfig, AutoModelForMaskedLM

    cfg = AutoConfig.from_pretrained(str(model_dir), trust_remote_code=True, local_files_only=True)
    dynamic_files = sorted([p.name for p in model_dir.glob("*modeling*.py")])
    out: Dict[str, Any] = {
        "checkpoint_path": rel(model_dir),
        "exists": model_dir.exists(),
        "model_safetensors_sha256": sha256_file(model_dir / "model.safetensors"),
        "config_architectures": list(getattr(cfg, "architectures", []) or []),
        "config_model_type": getattr(cfg, "model_type", None),
        "config_auto_map": getattr(cfg, "auto_map", None),
        "config_adapter_scale": getattr(cfg, "adapter_scale", None),
        "config_private_adapter_scale": getattr(cfg, "private_adapter_scale", None),
        "dynamic_modeling_files": dynamic_files,
    }
    try:
        model = AutoModelForMaskedLM.from_pretrained(str(model_dir), trust_remote_code=True, local_files_only=True)
        named = list(model.named_parameters())
        private = [(n, p) for n, p in named if ".private_adapter." in n]
        scales: List[float] = []
        try:
            scales = [float(layer.private_adapter.scale) for layer in model.deberta.encoder.layer]
        except Exception:
            pass
        out.update({
            "trust_remote_code_loaded_class": type(model).__name__,
            "trust_remote_code_loaded_module": type(model).__module__,
            "total_params": int(sum(p.numel() for _, p in named)),
            "private_adapter_params": int(sum(p.numel() for _, p in private)),
            "private_adapter_tensor_count": len(private),
            "executed_private_scales": scales,
        })
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    except Exception as exc:
        out["trust_remote_code_load_error"] = repr(exc)
    try:
        plain, info = AutoModelForMaskedLM.from_pretrained(
            str(model_dir), trust_remote_code=False, local_files_only=True, output_loading_info=True
        )
        unexpected = list(info.get("unexpected_keys") or [])
        missing = list(info.get("missing_keys") or [])
        out["plain_loader"] = {
            "class": type(plain).__name__,
            "module": type(plain).__module__,
            "total_params": int(sum(p.numel() for p in plain.parameters())),
            "private_adapter_params": int(sum(p.numel() for n, p in plain.named_parameters() if ".private_adapter." in n)),
            "unexpected_key_count": len(unexpected),
            "unexpected_private_key_count": sum(1 for k in unexpected if "private_adapter" in k),
            "missing_key_count": len(missing),
        }
        del plain
    except Exception as exc:
        out["plain_loader"] = {"error": repr(exc)}
    return out


def choose_seed(summary: Dict[str, Any], seed_arg: Optional[int]) -> Dict[str, Any]:
    per_seed = list(summary.get("per_seed") or [])
    if not per_seed:
        raise RuntimeError("research summary has no per_seed entries")
    if seed_arg is not None:
        for rec in per_seed:
            if int(rec.get("seed")) == int(seed_arg):
                return rec
        raise RuntimeError(f"Requested seed {seed_arg} not found; available {[r.get('seed') for r in per_seed]}")

    def key(rec: Dict[str, Any]):
        three = rec.get("trained_three_entity", {})
        ext = rec.get("trained_extended_held", {})
        reas = rec.get("trained_reassignment_held", {})
        # Prefer a state that solves the harder held probes; ties use cross-source margins.
        return (
            float(three.get("update_c_both_query_indexed", -1)),
            float(ext.get("retain_both_full_source", -1)),
            float(reas.get("n_swap_both_follow", -1)),
            float(ext.get("mean_retain_cross_source", float("nan"))),
        )
    return max(per_seed, key=key)


def parse_columns(texts: Iterable[str]) -> List[str]:
    cols: List[str] = []
    for t in texts:
        if t.lower() == "entity":
            cols.append("Entity")
        elif t.lower() == "cheap7":
            cols.extend(CHEAP7_COLUMNS)
        elif t in ZERO_TASK_BY_COL or t == "Reading":
            cols.append(t)
        else:
            raise ValueError(f"Unknown column {t}; use Entity, Cheap7, Reading, or one of {list(ZERO_TASK_BY_COL)}")
    # preserve order, remove duplicates
    out: List[str] = []
    for c in cols:
        if c not in out:
            out.append(c)
    return out


def run_selected(model_path: pathlib.Path, tag: str, gpu: int, out_root: pathlib.Path, columns: List[str], reading_data: str, force: bool) -> Dict[str, Any]:
    out_root.mkdir(parents=True, exist_ok=True)
    env = common.setup_env(out_root, tag, gpu)
    result: Dict[str, Any] = {
        "status": "SAVED_STATE_SELECTED_EVAL",
        "created_utc": now_utc(),
        "source_model_path": rel(model_path),
        "tag": tag,
        "gpu": gpu,
        "columns_requested": columns,
    }
    scores: Dict[str, Optional[float]] = {}
    existing_path = out_root / f"{tag}_eval.json"
    if existing_path.exists() and not force:
        prev = read_json(existing_path)
        prev_scores = prev.get("scores") or {}
    else:
        prev_scores = {}

    for col in columns:
        if col in ZERO_TASK_BY_COL:
            if col in prev_scores and prev_scores[col] is not None and not force:
                scores[col] = prev_scores[col]
                result[col] = {"column": col, "score": prev_scores[col], "status": "cached_score"}
                print(json.dumps({"event": "cached", "tag": tag, "column": col, "score": prev_scores[col]}), flush=True)
                continue
            print(f"[{tag}] Evaluating {col} on {rel(model_path)}", flush=True)
            _, task, data, batch = ZERO_TASK_BY_COL[col]
            r = common.eval_sentence(model_path, col, task, data, batch, tag, env, out_root)
            scores[col] = r.get("score")
            result[col] = r
            print(json.dumps({"event": "column_done", "tag": tag, "column": col, "score": r.get("score"), "returncode": r.get("returncode")}), flush=True)
        elif col == "Reading":
            if "Reading" in prev_scores and prev_scores["Reading"] is not None and not force:
                scores["Reading"] = prev_scores["Reading"]
                scores["Reading_eye"] = prev_scores.get("Reading_eye")
                scores["Reading_self_paced"] = prev_scores.get("Reading_self_paced")
                result["Reading"] = {"Reading": prev_scores["Reading"], "status": "cached_score"}
                print(json.dumps({"event": "cached", "tag": tag, "column": col, "score": prev_scores["Reading"]}), flush=True)
                continue
            print(f"[{tag}] Evaluating Reading on {rel(model_path)}", flush=True)
            r = common.eval_reading(model_path, tag, env, out_root, reading_data)
            scores.update({"Reading": r.get("Reading"), "Reading_eye": r.get("Reading_eye"), "Reading_self_paced": r.get("Reading_self_paced")})
            result["Reading"] = r
            print(json.dumps({"event": "column_done", "tag": tag, "column": col, "score": r.get("Reading"), "returncode": r.get("returncode")}), flush=True)

    if "GlobalPIQA_parallel" in scores or "GlobalPIQA_nonparallel" in scores:
        gp, gn = scores.get("GlobalPIQA_parallel"), scores.get("GlobalPIQA_nonparallel")
        if gp is not None and gn is not None:
            scores["GlobalPIQA_mean"] = (float(gp) + float(gn)) / 2.0
    eq_vals = [scores.get(k) for k in common.EQ7_KEYS]
    valid = [float(v) for v in eq_vals if v is not None]
    if valid:
        scores["equal_valid_mean"] = sum(valid) / len(valid)
        scores["n_valid_equal_columns"] = len(valid)
    result["scores"] = scores
    out_file = out_root / f"{tag}_eval.json"
    out_file.write_text(json.dumps(result, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    print(json.dumps({"status": "EVAL_DONE", "tag": tag, "out": rel(out_file), "scores": scores}, ensure_ascii=False), flush=True)
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--research-root", default=str(DEFAULT_STEP42_ROOT))
    ap.add_argument("--model-path", default="", help="Evaluate an explicit saved HF checkpoint instead of reading research summary.")
    ap.add_argument("--seed", type=int, default=None, help="Seed to choose from research summary; default chooses the best behavioral seed.")
    ap.add_argument("--target", default="", help="Output tag; default derived from selected seed/checkpoint.")
    ap.add_argument("--out-root", default=str(DEFAULT_OUT))
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--columns", nargs="+", default=["Entity"], help="Entity, Cheap7, Reading, or individual zero-shot column names.")
    ap.add_argument("--reading-data", default="evaluation_data/fast_eval/reading/reading_data.csv")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    out_root = pathlib.Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    columns = parse_columns(args.columns)

    selected_seed: Optional[Dict[str, Any]] = None
    if args.model_path:
        model_path = pathlib.Path(args.model_path)
        tag = args.target or model_path.parent.parent.parent.name if model_path.name == "final" else (args.target or model_path.name)
    else:
        root = pathlib.Path(args.root)
        summary_path = root / "saved_state_summary.json"
        if not summary_path.exists():
            raise FileNotFoundError(f"Missing research summary: {summary_path}")
        summary = read_json(summary_path)
        selected_seed = choose_seed(summary, args.seed)
        model_path = ROOT / str(selected_seed["model_dir"])
        tag = args.target or f"seed{selected_seed['seed']}_saved_state"

    identity = checkpoint_identity(model_path, out_root, tag)
    preflight = {
        "status": "SAVED_STATE_EVAL_PREFLIGHT",
        "created_utc": now_utc(),
        "selected_seed_record": selected_seed,
        "model_path": rel(model_path),
        "tag": tag,
        "columns": columns,
        "identity": identity,
        "out_root": rel(out_root),
        "scientific_use": "Compatibility readout for the same saved state that showed relation-first acquisition; not an official BabyLM submission.",
    }
    preflight_path = out_root / f"{tag}_preflight.json"
    preflight_path.write_text(json.dumps(preflight, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    print(json.dumps({"status": preflight["status"], "preflight": rel(preflight_path), "tag": tag, "columns": columns, "class": identity.get("trust_remote_code_loaded_class"), "private_params": identity.get("private_adapter_params")}, ensure_ascii=False), flush=True)
    if args.dry_run:
        return
    if identity.get("trust_remote_code_loaded_class") != "FrozenSlowPrivateDebertaV2ForMaskedLM" or int(identity.get("private_adapter_params") or 0) == 0:
        raise RuntimeError(f"Checkpoint identity is not the trusted private-adapter model: {identity}")
    run_selected(model_path, tag, int(args.gpu), out_root, columns, args.reading_data, args.force)


if __name__ == "__main__":
    main()
