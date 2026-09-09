#!/usr/bin/env python3
"""research: strict evaluator for corrected bridge checkpoints.

Relation readout uses the working research/research interface: simultaneous full-span
candidate scoring with correct source required to outrank both alternatives.  No
ordering between the two incorrect alternatives is required.  Checkpoint loading
fails if any private-adapter tensor is missing; a model that merely instantiates
private adapters without loading their weights is not accepted.

The optional Cheap7 path reuses research's repaired fast-screen wrapper after first
checking private-adapter integrity with the explicit model class.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import gc
import json
import os
import pathlib
import subprocess
import sys
import time
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from safetensors.torch import load_file
from transformers import AutoTokenizer, DebertaV2Config

ROOT = _public_path('.')
STUDY = _public_path('experiments/archive/functional_learning')
A02 = _public_path('experiments/archive/frontier_consolidation')
SCRIPTS = _public_path('experiments/archive/functional_learning/scripts')
A02_SCRIPTS = _public_path('experiments/archive/frontier_consolidation/scripts')
sys.path.insert(0, str(A02_SCRIPTS))
sys.path.insert(0, str(SCRIPTS))

from frozen82_private_modeling import FrozenSlowPrivateDebertaV2ForMaskedLM  # noqa: E402
import threeway_and_reassignment_probe as probe  # noqa: E402
import saved_state_replicate_neutral_threeentity as repl  # noqa: E402

PARENT_PATH = _public_path('models/frontier')
EXPECTED_PRIVATE_PARAMS = 995584


def rel(path: pathlib.Path | str) -> str:
    try:
        return str(pathlib.Path(path).resolve().relative_to(ROOT))
    except Exception:
        return str(path)


def finite_mean(xs: Iterable[float]) -> float:
    vals = [float(x) for x in xs if x is not None]
    vals = [x for x in vals if x == x and abs(x) != float("inf")]
    return sum(vals) / len(vals) if vals else float("nan")


def write_jsonl(path: pathlib.Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def strict_load_checkpoint(ckpt_path: pathlib.Path, device: torch.device, private_scale: float) -> Tuple[Any, Dict[str, Any]]:
    cfg = DebertaV2Config.from_pretrained(str(ckpt_path), local_files_only=True)
    cfg.private_adapter_bottleneck = 128
    cfg.private_adapter_scale = float(private_scale)
    cfg.private_adapter_enabled = True
    model = FrozenSlowPrivateDebertaV2ForMaskedLM(cfg)
    sd_path = ckpt_path / "model.safetensors"
    if not sd_path.exists():
        raise FileNotFoundError(f"missing model.safetensors at {ckpt_path}")
    sd = load_file(str(sd_path), device="cpu")
    missing, unexpected = model.load_state_dict(sd, strict=False)
    tied_missing = {"cls.predictions.decoder.weight", "cls.predictions.decoder.bias"}
    missing_private = [k for k in missing if "private_adapter" in k]
    bad_missing = [k for k in missing if k not in tied_missing]
    if missing_private or bad_missing or unexpected:
        raise RuntimeError({
            "checkpoint": rel(ckpt_path),
            "missing_private": missing_private[:20],
            "bad_missing": bad_missing[:20],
            "unexpected": list(unexpected)[:20],
        })
    model.tie_weights()
    if hasattr(model, "set_private_enabled"):
        model.set_private_enabled(True)
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    model.to(device)
    model.eval()
    named = list(model.named_parameters())
    private = [(n, p) for n, p in named if ".private_adapter." in n]
    scales: List[float] = []
    try:
        scales = [float(layer.private_adapter.scale) for layer in model.deberta.encoder.layer]
    except Exception:
        pass
    ident = {
        "checkpoint_path": rel(ckpt_path),
        "class": type(model).__name__,
        "module": type(model).__module__,
        "total_params": int(sum(p.numel() for _, p in named)),
        "private_adapter_tensors": len(private),
        "private_adapter_params": int(sum(p.numel() for _, p in private)),
        "executed_private_scales": scales,
        "missing_key_count": len(missing),
        "unexpected_key_count": len(unexpected),
        "missing_private_key_count": len(missing_private),
    }
    if ident["class"] != "FrozenSlowPrivateDebertaV2ForMaskedLM" or ident["private_adapter_params"] != EXPECTED_PRIVATE_PARAMS:
        raise RuntimeError(f"untrusted checkpoint identity: {ident}")
    return model, ident


def load_parent(device: torch.device, private_scale: float):
    import coherent86_continuation_trainer as parent_loader
    model, missing, unexpected = parent_loader.load_model(PARENT_PATH, device, 128, float(private_scale))
    if [k for k in missing if "private_adapter" in k] or unexpected:
        raise RuntimeError(f"parent load mismatch: missing={missing[:10]}, unexpected={unexpected[:10]}")
    if hasattr(model, "set_private_enabled"):
        model.set_private_enabled(True)
    model.eval()
    named = list(model.named_parameters())
    private = [(n, p) for n, p in named if ".private_adapter." in n]
    return model, {
        "checkpoint_path": rel(PARENT_PATH),
        "class": type(model).__name__,
        "module": type(model).__module__,
        "total_params": int(sum(p.numel() for _, p in named)),
        "private_adapter_tensors": len(private),
        "private_adapter_params": int(sum(p.numel() for _, p in private)),
        "executed_private_scales": [float(layer.private_adapter.scale) for layer in model.deberta.encoder.layer],
    }


def extra_margin_summary(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Separate correct-source vs wrong-source and correct-source vs replacement margins."""
    return {
        "n_records": len(records),
        "neutral_qa_full_source": sum(1 for r in records if r["neutral_qa_full_source"]),
        "neutral_qb_full_source": sum(1 for r in records if r["neutral_qb_full_source"]),
        "retain_qa_full_source": sum(1 for r in records if r["retain_qa_full_source"]),
        "retain_qb_full_source": sum(1 for r in records if r["retain_qb_full_source"]),
        "mean_neutral_correct_vs_wrong": finite_mean([r["neutral_qa_cross_source"] for r in records] + [r["neutral_qb_cross_source"] for r in records]),
        "mean_neutral_correct_vs_replacement": finite_mean([r["neutral_qa_correct_over_new"] for r in records] + [r["neutral_qb_correct_over_new"] for r in records]),
        "mean_retain_correct_vs_wrong": finite_mean([r["retain_qa_cross_source"] for r in records] + [r["retain_qb_cross_source"] for r in records]),
        "mean_retain_correct_vs_replacement": finite_mean([r["retain_qa_correct_over_new"] for r in records] + [r["retain_qb_correct_over_new"] for r in records]),
        "full_source_rule": "correct source must outrank wrong source and shared replacement; no ordering between wrong source and replacement is required",
    }


def eval_relation(model, tokenizer, held_pairs, three_cases, device, seq_length: int, label: str, raw_dir: pathlib.Path) -> Dict[str, Any]:
    held_tw = []
    for i, p in enumerate(held_pairs):
        held_tw.append(probe.threeway_score_pair(model, tokenizer, p, device, seq_length))
        if (i + 1) % 10 == 0:
            print(json.dumps({"event": "relation_threeway_progress", "label": label, "n": i + 1}), flush=True)
    tw_summary, tw_records = repl.analyze_threeway_extended(held_tw, held_pairs, label)
    reassign_summary, reassign_records, reassign_n = repl.score_reassignment_held(
        model, tokenizer, held_pairs, held_tw, device, seq_length)
    three_scored = [repl.score_three_entity_case(model, tokenizer, c, device, seq_length) for c in three_cases]
    three_summary, three_records = repl.analyze_three_entity(three_scored, three_cases, f"{label}_three_entity")
    write_jsonl(raw_dir / f"{label}_held_threeway_records.jsonl", tw_records)
    write_jsonl(raw_dir / f"{label}_reassignment_records.jsonl", reassign_records)
    write_jsonl(raw_dir / f"{label}_three_entity_records.jsonl", three_records)
    return {
        "threeway": tw_summary,
        "margin_summary": extra_margin_summary(tw_records),
        "reassignment": reassign_summary,
        "reassignment_valid_n": reassign_n,
        "three_entity": three_summary,
    }


def bridge_checkpoints(bridge_dir: pathlib.Path, spec: str) -> List[pathlib.Path]:
    base = bridge_dir / "checkpoints"
    if not base.exists():
        raise FileNotFoundError(f"no checkpoints under {bridge_dir}")
    if spec == "all":
        return sorted([p for p in base.iterdir() if p.is_dir()])
    out = []
    for item in spec.split(","):
        name = item.strip()
        if not name:
            continue
        p = base / name
        if not p.exists() and name.isdigit():
            p = base / f"update_{int(name):04d}"
        out.append(p)
    return out


def eval_cheap7_with_step043(ckpt_path: pathlib.Path, tag: str, out_root: pathlib.Path, gpu: int, columns: List[str], reading_data: str, force: bool) -> Dict[str, Any]:
    """Run research as a fresh process after strict in-process adapter loading.

    Importing Transformers before research redirects HF_MODULES_CACHE can bind the
    dynamic-module loader to a read-only default.  A subprocess avoids that cache
    ordering problem while the in-process strict_load_checkpoint verifies that the
    checkpoint contains all private-adapter tensors before any score is accepted.
    """
    device = torch.device("cpu")
    model, ident = strict_load_checkpoint(ckpt_path, device, private_scale=0.75)
    del model
    gc.collect()
    cmd = [
        sys.executable,
        str(_public_path('experiments/archive/functional_learning/scripts/eval_saved_state_cheap7_entity.py')),
        "--model-path", str(ckpt_path),
        "--target", tag,
        "--out-root", str(out_root),
        "--gpu", str(int(gpu)),
        "--columns", *columns,
        "--reading-data", reading_data,
    ]
    if force:
        cmd.append("--force")
    env = os.environ.copy()
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    proc = subprocess.run(cmd, cwd=str(ROOT), text=True, capture_output=True, env=env)
    if proc.returncode != 0:
        raise RuntimeError({"cmd": cmd, "returncode": proc.returncode, "stdout_tail": proc.stdout[-2000:], "stderr_tail": proc.stderr[-4000:]})
    out_file = out_root / f"{tag}_eval.json"
    if not out_file.exists():
        raise FileNotFoundError(f"research subprocess did not produce {out_file}")
    result = json.loads(out_file.read_text(encoding="utf-8"))
    preflight_file = out_root / f"{tag}_preflight.json"
    preflight = json.loads(preflight_file.read_text(encoding="utf-8")) if preflight_file.exists() else {}
    identity = preflight.get("identity") or {}
    if identity.get("trust_remote_code_loaded_class") != "FrozenSlowPrivateDebertaV2ForMaskedLM" or int(identity.get("private_adapter_params") or 0) != EXPECTED_PRIVATE_PARAMS:
        raise RuntimeError(f"research identity check failed after subprocess: {identity}")
    return {"strict_identity": ident, "identity": identity, "result": result, "scores": result.get("scores", {}), "stdout_tail": proc.stdout[-2000:], "stderr_tail": proc.stderr[-2000:]}


def parse_columns(args_columns: Sequence[str]) -> List[str]:
    import eval_saved_state_cheap7_entity as cheap
    return cheap.parse_columns(args_columns)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bridge-dir", default="", help="Bridge training directory containing checkpoints/.")
    ap.add_argument("--model-path", default="", help="Evaluate one explicit HF checkpoint path instead of a bridge-dir checkpoint list.")
    ap.add_argument("--include-parent", action="store_true", help="Also relation-score the coherent86 parent baseline.")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--checkpoints", default="all", help="all, comma names, or update numbers")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--seq-length", type=int, default=512)
    ap.add_argument("--private-scale", type=float, default=0.75)
    ap.add_argument("--construction-seed", type=int, default=40040)
    ap.add_argument("--eval-relation", action="store_true", default=False)
    ap.add_argument("--eval-cheap7", action="store_true", default=False)
    ap.add_argument("--cheap7-columns", nargs="+", default=["Entity"], help="Entity, Cheap7, Reading, or selected research columns")
    ap.add_argument("--reading-data", default="evaluation_data/fast_eval/reading/reading_data.csv")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    if not args.eval_relation and not args.eval_cheap7:
        raise ValueError("Select --eval-relation and/or --eval-cheap7")

    out_dir = pathlib.Path(args.out_dir)
    raw_dir = out_dir / "raw_records"
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    targets: List[Tuple[str, pathlib.Path, Optional[pathlib.Path]]] = []
    if args.model_path:
        p = pathlib.Path(args.model_path)
        targets.append((p.name if p.name != "final" else p.parent.parent.name, p, None))
    else:
        bridge_dir = pathlib.Path(args.bridge_dir)
        for p in bridge_checkpoints(bridge_dir, args.checkpoints):
            targets.append((p.name, p, bridge_dir))
    if args.include_parent:
        targets.insert(0, ("coherent86_parent", PARENT_PATH, None))

    tokenizer = AutoTokenizer.from_pretrained(str(PARENT_PATH), local_files_only=True, use_fast=True)
    pairs, train_pairs, held_pairs, _train_rows, _held_rows = repl.fixed_repaired_pairs(int(args.construction_seed))
    three_cases = repl.build_three_entity_cases(pairs, train_pairs, held_pairs, int(args.construction_seed), max_cases=12)
    columns = parse_columns(args.cheap7_columns) if args.eval_cheap7 else []

    all_results: List[Dict[str, Any]] = []
    for label, ckpt_path, bridge_dir in targets:
        if not ckpt_path.exists():
            raise FileNotFoundError(ckpt_path)
        tag = label.replace("/", "_")
        if bridge_dir is not None:
            tag = f"{bridge_dir.name}_{tag}"
        print(json.dumps({"event": "target_start", "label": label, "path": rel(ckpt_path)}), flush=True)
        t0 = time.time()
        rec: Dict[str, Any] = {"label": label, "path": rel(ckpt_path), "bridge_dir": rel(bridge_dir) if bridge_dir else None}
        meta_path = ckpt_path / "bridge_metadata.json"
        if meta_path.exists():
            rec["bridge_metadata"] = json.loads(meta_path.read_text(encoding="utf-8"))

        if args.eval_relation:
            if ckpt_path == PARENT_PATH:
                model, ident = load_parent(device, float(args.private_scale))
            else:
                model, ident = strict_load_checkpoint(ckpt_path, device, float(args.private_scale))
            rec["model_identity"] = ident
            rec["relation"] = eval_relation(model, tokenizer, held_pairs, three_cases, device, int(args.seq_length), tag, raw_dir)
            relres = rec["relation"]
            print(json.dumps({
                "event": "relation_done", "label": label,
                "neutral_full": relres["threeway"].get("neutral_both_full_source"),
                "retain_full": relres["threeway"].get("retain_both_full_source"),
                "reassign_both": relres["reassignment"].get("n_swap_both_follow"),
                "three_entity": relres["three_entity"].get("update_c_both_query_indexed"),
            }), flush=True)
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if args.eval_cheap7:
            if ckpt_path == PARENT_PATH:
                import eval_saved_state_cheap7_entity as cheap
                # Parent has the same exact class when loaded by trusted research path; run selected directly.
                cheap_res = cheap.run_selected(ckpt_path, tag, int(args.gpu), out_dir / "cheap7", columns, args.reading_data, args.force)
                rec["cheap7"] = {"scores": cheap_res.get("scores", {}), "result": cheap_res}
            else:
                rec["cheap7"] = eval_cheap7_with_step043(ckpt_path, tag, out_dir / "cheap7", int(args.gpu), columns, args.reading_data, args.force)
            print(json.dumps({
                "event": "cheap7_done", "label": label,
                "Entity": rec["cheap7"].get("scores", {}).get("Entity"),
                "equal_valid_mean": rec["cheap7"].get("scores", {}).get("equal_valid_mean"),
            }), flush=True)

        rec["elapsed_sec"] = round(time.time() - t0, 1)
        all_results.append(rec)
        (out_dir / "bridge_eval_partial.json").write_text(json.dumps({"results": all_results}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    summary = {
        "status": "BRIDGE_EVAL_DONE",
        "eval_relation": bool(args.eval_relation),
        "eval_cheap7": bool(args.eval_cheap7),
        "n_targets": len(all_results),
        "full_source_rule": "retention requires correct source > wrong source and correct source > shared replacement; no wrong-vs-replacement ordering is required",
        "results": all_results,
    }
    (out_dir / "bridge_eval_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
