#!/usr/bin/env python3
"""research: compare scale-1.75 20M standalone trajectory with the embedded 20M checkpoint in the 50M continuation run.

The scientific purpose is to test whether the apparent research 20M residual-adapter gain is an execution-stable prefix of the 50M trajectory before interpreting the 50M score or spending further exposure.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import torch
from safetensors.torch import load_file

ROOT = Path("experiments/archive/frontier_consolidation")
RUN20 = ROOT / "training/runs/adapter128_scale1p75_h100M20M_seed43022"
RUN50 = ROOT / "training/runs/adapter128_scale1p75_h100M50M_seed43022"
OUTDIR = ROOT / "data/scale1p75_prefix_repro"
CKPT = "hf_model/chck_20M"
IGNORE_LOG_KEYS = {"elapsed_sec"}
FLOAT_TOL = 0.0


def sha256(path: Path, block: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(block)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        out.append(json.loads(line))
    return out


def normalize_log_record(rec: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in rec.items() if k not in IGNORE_LOG_KEYS}


def train_records(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    # The research training_log.jsonl contains per-step train records only; checkpoint_saved
    # events are printed to stdout, not stored in this file. Therefore the embedded 20M
    # prefix of a longer run is the same number of train rows as the standalone 20M run.
    return [normalize_log_record(r) for r in records if "step" in r and "cumulative_word_exposure" in r]


def compare_records(a: List[Dict[str, Any]], b: List[Dict[str, Any]]) -> Dict[str, Any]:
    b = b[:len(a)]
    same_len = len(a) == len(b)
    mismatches = []
    for i, (ra, rb) in enumerate(zip(a, b)):
        if ra != rb:
            keys = sorted(set(ra) | set(rb))
            diff = {k: {"a": ra.get(k), "b": rb.get(k)} for k in keys if ra.get(k) != rb.get(k)}
            mismatches.append({"index": i, "diff": diff, "a": ra, "b": rb})
            if len(mismatches) >= 20:
                break
    return {
        "same_len": same_len,
        "len_a": len(a),
        "len_b": len(b),
        "n_mismatches_first20": len(mismatches),
        "first_mismatches": mismatches,
        "exact_equal_ignoring_elapsed": same_len and not mismatches,
        "last_a": a[-1] if a else None,
        "last_b": b[-1] if b else None,
    }


def tensor_compare(path_a: Path, path_b: Path) -> Dict[str, Any]:
    state_a = load_file(str(path_a), device="cpu")
    state_b = load_file(str(path_b), device="cpu")
    keys_a, keys_b = set(state_a), set(state_b)
    common = sorted(keys_a & keys_b)
    missing_a = sorted(keys_b - keys_a)
    missing_b = sorted(keys_a - keys_b)
    total_numel = 0
    total_sq = 0.0
    max_abs = 0.0
    max_rel = 0.0
    mismatching = []
    exact_all = True
    for k in common:
        ta = state_a[k]
        tb = state_b[k]
        if ta.shape != tb.shape or ta.dtype != tb.dtype:
            exact_all = False
            mismatching.append({"name": k, "shape_a": list(ta.shape), "shape_b": list(tb.shape), "dtype_a": str(ta.dtype), "dtype_b": str(tb.dtype)})
            continue
        eq = torch.equal(ta, tb)
        if not eq:
            exact_all = False
            diff = (ta.to(torch.float64) - tb.to(torch.float64)).abs()
            local_max = float(diff.max().item()) if diff.numel() else 0.0
            denom = float(ta.to(torch.float64).norm().item())
            rel = float(diff.norm().item() / (denom + 1e-30)) if diff.numel() else 0.0
            max_abs = max(max_abs, local_max)
            max_rel = max(max_rel, rel)
            if len(mismatching) < 25:
                mismatching.append({"name": k, "max_abs": local_max, "rel_l2_vs_a": rel, "numel": int(diff.numel())})
            total_sq += float((diff * diff).sum().item())
        total_numel += int(ta.numel())
    return {
        "n_keys_a": len(keys_a),
        "n_keys_b": len(keys_b),
        "n_common": len(common),
        "missing_in_20M": missing_a,
        "missing_in_50M": missing_b,
        "exact_tensor_equal": exact_all and not missing_a and not missing_b,
        "total_numel_common": total_numel,
        "global_l2_diff": total_sq ** 0.5,
        "max_abs_diff": max_abs,
        "max_rel_l2_tensor": max_rel,
        "mismatching_examples": mismatching,
    }


def selected_file_hashes(run: Path) -> Dict[str, str]:
    rels = [
        "example_order_manifest.json",
        "scientific_metrics.json",
        "hf_model/config.json",
        "hf_model/tokenizer.json",
        "hf_model/tokenizer_config.json",
        "hf_model/special_tokens_map.json",
        f"{CKPT}/config.json",
        f"{CKPT}/tokenizer.json",
        f"{CKPT}/tokenizer_config.json",
        f"{CKPT}/special_tokens_map.json",
        f"{CKPT}/model.safetensors",
    ]
    out = {}
    for r in rels:
        p = run / r
        out[r] = sha256(p) if p.exists() else "MISSING"
    return out


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    for p in [RUN20, RUN50, RUN20 / CKPT, RUN50 / CKPT]:
        if not p.exists():
            raise FileNotFoundError(p)

    hashes20 = selected_file_hashes(RUN20)
    hashes50 = selected_file_hashes(RUN50)
    hash_cmp = {r: {"20M_run": hashes20.get(r), "50M_run": hashes50.get(r), "equal": hashes20.get(r) == hashes50.get(r)} for r in sorted(set(hashes20) | set(hashes50))}

    config_cmp = {}
    for rel in ["hf_model/config.json", f"{CKPT}/config.json", "example_order_manifest.json", "scientific_metrics.json"]:
        pa, pb = RUN20 / rel, RUN50 / rel
        if pa.exists() and pb.exists():
            ja, jb = load_json(pa), load_json(pb)
            config_cmp[rel] = {"equal_json": ja == jb, "a": ja if ja != jb else None, "b": jb if ja != jb else None}
        else:
            config_cmp[rel] = {"equal_json": False, "missing": [str(p) for p in [pa, pb] if not p.exists()]}

    log20 = train_records(read_jsonl(RUN20 / "training_log.jsonl"))
    log50_all = train_records(read_jsonl(RUN50 / "training_log.jsonl"))
    log50 = log50_all[:len(log20)]
    log_cmp = compare_records(log20, log50)

    tensors = tensor_compare(RUN20 / CKPT / "model.safetensors", RUN50 / CKPT / "model.safetensors")

    conclusion = {
        "prefix_reproduces": bool(log_cmp["exact_equal_ignoring_elapsed"] and tensors["exact_tensor_equal"]),
        "hash_model_safetensors_equal": hash_cmp[f"{CKPT}/model.safetensors"]["equal"],
        "log_exact_equal_ignoring_elapsed": log_cmp["exact_equal_ignoring_elapsed"],
        "tensor_exact_equal": tensors["exact_tensor_equal"],
    }

    out = {
        "status": "SCALE1P75_PREFIX_REPRO",
        "scientific_question": "Does the 50M scale1.75 run contain the same 20M trajectory that scored +0.642 cheap7 in research?",
        "run20": str(RUN20),
        "run50": str(RUN50),
        "checkpoint": CKPT,
        "conclusion": conclusion,
        "file_hash_comparison": hash_cmp,
        "json_comparison": config_cmp,
        "training_log_prefix_comparison": log_cmp,
        "training_log_rows_in_50M_run_total": len(log50_all),
        "tensor_comparison": tensors,
    }
    out_json = OUTDIR / "scale1p75_prefix_repro.json"
    out_md = (OUTDIR.parents[4] / 'research/documents/frontier_consolidation/data/scale1p75_prefix_repro/scale1p75_prefix_repro.md')
    out_json.write_text(json.dumps(out, indent=2, sort_keys=True))

    lines = [
        "# research scale1.75 embedded-prefix reproduction",
        "",
        f"Run20: `{RUN20}`",
        f"Run50: `{RUN50}`",
        f"Checkpoint compared: `{CKPT}`",
        "",
        "## Conclusion",
        f"- Prefix reproduces exactly (training log except elapsed + tensors): **{conclusion['prefix_reproduces']}**",
        f"- Model safetensors hash equal: {conclusion['hash_model_safetensors_equal']}",
        f"- Training-log prefix equal ignoring elapsed seconds: {conclusion['log_exact_equal_ignoring_elapsed']}",
        f"- State-dict tensors exactly equal: {conclusion['tensor_exact_equal']}",
        "",
        "## Training-log prefix",
        f"- records compared through standalone chck_20M: {log_cmp['len_a']} vs {log_cmp['len_b']} (50M total train rows: {len(log50_all)})",
        f"- first mismatches after dropping elapsed_sec: {log_cmp['n_mismatches_first20']}",
        f"- last 20M-run prefix record: `{json.dumps(log_cmp['last_a'], sort_keys=True)}`",
        f"- last 50M-run prefix record: `{json.dumps(log_cmp['last_b'], sort_keys=True)}`",
        "",
        "## Tensor comparison",
        f"- keys: {tensors['n_keys_a']} vs {tensors['n_keys_b']}, common {tensors['n_common']}",
        f"- exact tensor equal: {tensors['exact_tensor_equal']}",
        f"- global L2 diff: {tensors['global_l2_diff']}",
        f"- max abs diff: {tensors['max_abs_diff']}",
        f"- max per-tensor rel L2: {tensors['max_rel_l2_tensor']}",
        "",
        "## Selected hash equality",
        "| relative file | equal |",
        "|---|---:|",
    ]
    for rel, rec in hash_cmp.items():
        lines.append(f"| `{rel}` | {rec['equal']} |")
    if log_cmp["first_mismatches"]:
        lines += ["", "## First log mismatches", "```json", json.dumps(log_cmp["first_mismatches"], indent=2), "```"]
    if tensors["mismatching_examples"]:
        lines += ["", "## First tensor mismatches", "```json", json.dumps(tensors["mismatching_examples"], indent=2), "```"]
    out_md.write_text("\n".join(lines) + "\n")
    print(json.dumps({"status": out["status"], "out_json": str(out_json), "out_md": str(out_md), **conclusion}, indent=2))


if __name__ == "__main__":
    main()
