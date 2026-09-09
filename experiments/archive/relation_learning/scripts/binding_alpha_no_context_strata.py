#!/usr/bin/env python3
"""research: alpha scan, no-context shortcut check, and conflict strata for the
chck_82M frozen-private binding candidate.

This is a local margin readout, not official BabyLM scoring. It reuses the
research candidate-phrase scorer on the 200 held binding pairs.

Key coordinates:
  * full_context: original source+update+query row.
  * no_context: only "The relevant state of {query_entity} is ___."; if joint
    remains high here, the candidate may be using entity-answer associations or
    answer priors rather than source/update state assignment.
  * alpha: inference-time private_adapter_scale applied to the trained private
    branch. alpha=0 is the protected chck_82M slow function; alpha=0.75 is the
    saved research candidate setting.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import csv
import json
import math
import os
import pathlib
import re
import sys
import time
from collections import defaultdict
from typing import Any

import torch

SCRIPT = _public_path('experiments/archive/relation_learning/scripts/binding_alpha_no_context_strata.py')
ROOT = _public_path('.')
WS = _public_path('experiments/archive/relation_learning')
DEFAULT_CKPT = _public_path('experiments/archive/relation_learning/data/chck82_binding_candidate/scale_0.75/checkpoint')
DEFAULT_HELD = _public_path('experiments/archive/relation_learning/data/recombination_rows/recombination_heldout.jsonl')
DEFAULT_PAIRS = _public_path('experiments/archive/relation_learning/data/recombination_rows/binding_pairs_heldout.jsonl')
DEFAULT_OUT = _public_path('experiments/archive/relation_learning/data/binding_alpha_no_context_strata')

CACHE_BASE = os.environ.get("CACHE_BASE", str(_public_path('experiments/archive/relation_learning/data/binding_alpha_no_context_strata/hf_cache')))
os.environ["HF_HOME"] = CACHE_BASE
os.environ["TRANSFORMERS_CACHE"] = CACHE_BASE
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# research creates cache directories at import time; point it into this research
# writable output tree before importing, otherwise its default cache path may be
# outside writable cache directories.
os.environ["CACHE_BASE"] = str(pathlib.Path(CACHE_BASE).resolve() / "import_cache")
sys.path.insert(0, str(_public_path('experiments/archive/relation_learning/scripts')))
import binding_factorial_train as S73  # noqa: E402


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def rel(p: pathlib.Path | str) -> str:
    pp = pathlib.Path(p)
    try:
        return str(pp.resolve().relative_to(ROOT))
    except Exception:
        return str(pp)


def read_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    out = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
    return out


def write_json(path: pathlib.Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_csv(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    seen = set()
    for r in rows:
        for k in r:
            if k not in seen:
                seen.add(k); keys.append(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader(); w.writerows(rows)


def norm(s: Any) -> str:
    return re.sub(r"\s+", " ", str(s or "").strip().lower())


def contains_phrase(text: str, phrase: str) -> bool:
    t = norm(text); p = norm(phrase)
    return bool(p) and p in t


def first_pos(text: str, phrase: str) -> int | None:
    t = norm(text); p = norm(phrase)
    if not p:
        return None
    idx = t.find(p)
    return idx if idx >= 0 else None


def word_jaccard(a: str, b: str) -> float:
    ta = set(re.findall(r"[a-z0-9]+", norm(a)))
    tb = set(re.findall(r"[a-z0-9]+", norm(b)))
    if not ta and not tb:
        return 1.0
    return len(ta & tb) / max(1, len(ta | tb))


def set_private_scale(model: Any, alpha: float) -> None:
    if hasattr(model, "config"):
        model.config.private_adapter_scale = float(alpha)
    n = 0
    for layer in getattr(getattr(model, "deberta", None), "encoder", object()).layer:
        if hasattr(layer, "private_adapter"):
            layer.private_adapter.scale = float(alpha)
            layer.private_adapter.enabled = (float(alpha) != 0.0)
            n += 1
    if hasattr(model, "set_private_enabled"):
        model.set_private_enabled(float(alpha) != 0.0)
        # set_private_enabled may leave module scales untouched; set again.
        for layer in model.deberta.encoder.layer:
            if hasattr(layer, "private_adapter"):
                layer.private_adapter.scale = float(alpha)
    if n == 0:
        raise RuntimeError("No private_adapter modules found; wrong model class for alpha scan")


def model_identity(model: Any, alpha: float) -> dict[str, Any]:
    total = sum(p.numel() for p in model.parameters())
    private = sum(p.numel() for n, p in model.named_parameters() if ".private_adapter." in n)
    slow_adapter = sum(p.numel() for n, p in model.named_parameters() if ".adapter." in n and ".private_adapter." not in n)
    return {
        "model_class": type(model).__name__,
        "total_params": int(total),
        "private_params": int(private),
        "slow_adapter_params": int(slow_adapter),
        "alpha": float(alpha),
        "config_private_adapter_scale": float(getattr(model.config, "private_adapter_scale", float("nan"))),
        "config_adapter_scale": float(getattr(model.config, "adapter_scale", float("nan"))),
    }


def no_context_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for r in rows:
        rr = dict(r)
        prefix = f"The relevant state of {r['query_entity']} is "
        text = prefix + str(r["answer_text"]).strip() + "."
        rr["context_text"] = text
        rr["answer_char_start"] = len(prefix)
        rr["answer_char_end"] = len(prefix) + len(str(r["answer_text"]).strip())
        rr["shortcut_context_mode"] = "no_context_query_only"
        out.append(rr)
    return out


def pair_meta_from_rows(rows: list[dict[str, Any]], pairs: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    by_id = {r["row_id"]: r for r in rows}
    meta = {}
    for bp in pairs:
        a = by_id[bp["row_a_id"]]
        b = by_id[bp["row_b_id"]]
        src = a.get("source_sentence", "")
        update = a.get("update_sentence", "")
        shared = (src + " " + update).strip()
        target = a.get("target_entity", bp.get("entity_a", ""))
        updated = a.get("updated_entity", bp.get("entity_b", ""))
        target_in_source = contains_phrase(src, target)
        updated_in_source = contains_phrase(src, updated)
        target_pos = first_pos(shared, target)
        updated_pos = first_pos(shared, updated)
        if target_pos is None or updated_pos is None or target_pos == updated_pos:
            order = "ambiguous"
        elif target_pos < updated_pos:
            order = "unchanged_first"
        else:
            order = "updated_first"
        ans_j = word_jaccard(a.get("source_state", bp.get("answer_a", "")), a.get("new_state", bp.get("answer_b", "")))
        genuine = bool(target_in_source and updated_in_source and ans_j < 0.5)
        meta[bp["pair_id"]] = {
            "pair_id": bp["pair_id"],
            "row_a_id": bp["row_a_id"],
            "row_b_id": bp["row_b_id"],
            "target_entity": target,
            "updated_entity": updated,
            "target_entity_in_source": int(target_in_source),
            "updated_entity_in_source": int(updated_in_source),
            "both_entities_in_source": int(target_in_source and updated_in_source),
            "position_order": order,
            "answer_jaccard": round(ans_j, 6),
            "conflict_quality": "genuine_both_source_low_answer_overlap" if genuine else "weak_or_location_separable",
        }
    return meta


def pair_outcomes(eval_result: dict[str, Any], pairs: list[dict[str, Any]], meta: dict[str, dict[str, Any]], alpha: float, mode: str) -> list[dict[str, Any]]:
    rows = {r["row_id"]: r for r in eval_result.get("row_margins", [])}
    out = []
    for bp in pairs:
        a = rows.get(bp["row_a_id"])
        b = rows.get(bp["row_b_id"])
        if not a or not b:
            continue
        ac = int(a.get("correct_by_margin", 0)); bc = int(b.get("correct_by_margin", 0))
        joint = int(ac and bc)
        rec = {
            "alpha": alpha,
            "context_mode": mode,
            "pair_id": bp["pair_id"],
            "a_correct": ac,
            "b_correct": bc,
            "joint": joint,
            "both_wrong": int((not ac) and (not bc)),
            "a_margin": float(a.get("correct_margin", float("nan"))),
            "b_margin": float(b.get("correct_margin", float("nan"))),
            "joint_min_margin": min(float(a.get("correct_margin", float("nan"))), float(b.get("correct_margin", float("nan")))),
        }
        rec.update(meta.get(bp["pair_id"], {}))
        out.append(rec)
    return out


def summarize_pair_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    def add_group(r: dict[str, Any]) -> list[str]:
        groups = ["all_pairs"]
        groups.append(str(r.get("conflict_quality", "unknown")))
        if int(r.get("updated_entity_in_source", 0)):
            groups.append("updated_entity_in_source")
        else:
            groups.append("updated_entity_not_in_source")
        if int(r.get("both_entities_in_source", 0)):
            groups.append("both_entities_in_source")
        else:
            groups.append("not_both_entities_in_source")
        groups.append("position_" + str(r.get("position_order", "unknown")))
        if float(r.get("answer_jaccard", 0.0)) >= 0.25:
            groups.append("answer_overlap_ge0p25")
        else:
            groups.append("answer_overlap_lt0p25")
        return groups
    buckets: dict[tuple[float, str, str], list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        for g in add_group(r):
            buckets[(float(r["alpha"]), str(r["context_mode"]), g)].append(r)
    out = []
    for (alpha, mode, group), vals in sorted(buckets.items(), key=lambda x: (x[0][0], x[0][1], x[0][2])):
        n = len(vals)
        j = sum(int(v["joint"]) for v in vals)
        ac = sum(int(v["a_correct"]) for v in vals)
        bc = sum(int(v["b_correct"]) for v in vals)
        bw = sum(int(v["both_wrong"]) for v in vals)
        out.append({
            "alpha": alpha,
            "context_mode": mode,
            "group": group,
            "n": n,
            "joint": j,
            "gated_frac": j / n if n else float("nan"),
            "a_correct": ac,
            "b_correct": bc,
            "both_wrong": bw,
            "a_acc": ac / n if n else float("nan"),
            "b_acc": bc / n if n else float("nan"),
            "mean_joint_min_margin": sum(float(v["joint_min_margin"]) for v in vals) / n if n else float("nan"),
        })
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=pathlib.Path, default=DEFAULT_CKPT)
    ap.add_argument("--held-data", type=pathlib.Path, default=DEFAULT_HELD)
    ap.add_argument("--binding-pairs", type=pathlib.Path, default=DEFAULT_PAIRS)
    ap.add_argument("--out-dir", type=pathlib.Path, default=DEFAULT_OUT)
    ap.add_argument("--alphas", nargs="*", type=float, default=[0.0, 0.25, 0.5, 0.75, 1.0, 1.25])
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(json.dumps({"event": "load_model", "checkpoint": rel(args.checkpoint), "device": str(device), "utc": now()}), flush=True)
    from transformers import AutoModelForMaskedLM, AutoTokenizer
    model = AutoModelForMaskedLM.from_pretrained(str(args.checkpoint), trust_remote_code=True).to(device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(str(args.checkpoint), use_fast=True)

    held_all = read_jsonl(args.held_data)
    held_rows = [r for r in held_all if r.get("pair_half") in ("A", "B")]
    binding_pairs = read_jsonl(args.binding_pairs)
    meta = pair_meta_from_rows(held_rows, binding_pairs)
    context_sets = {
        "full_context": held_rows,
        "no_context": no_context_rows(held_rows),
    }

    all_pair_rows: list[dict[str, Any]] = []
    compact_evals: dict[str, Any] = {}
    identities: dict[str, Any] = {}
    for alpha in args.alphas:
        set_private_scale(model, alpha)
        identities[str(alpha)] = model_identity(model, alpha)
        for mode, rows in context_sets.items():
            print(json.dumps({"event": "score", "alpha": alpha, "mode": mode, "utc": now()}), flush=True)
            result = S73.score_margin_eval(model, tokenizer, rows, binding_pairs, device, args.max_length, batch_size=args.batch_size)
            pair_rows = pair_outcomes(result, binding_pairs, meta, alpha, mode)
            all_pair_rows.extend(pair_rows)
            compact_evals[f"alpha{alpha:g}_{mode}"] = {
                "binding_pairs_scored": result.get("binding_pairs_scored"),
                "joint": result.get("binding_joint_correct"),
                "gated_frac": result.get("binding_joint_accuracy"),
                "a_correct": result.get("a_correct"),
                "b_correct": result.get("b_correct"),
                "both_wrong": sum(int(r["both_wrong"]) for r in pair_rows),
                "by_role": result.get("by_role"),
            }

    strata = summarize_pair_rows(all_pair_rows)
    write_csv(args.out_dir / "pair_outcomes.csv", all_pair_rows)
    write_csv(args.out_dir / "strata_summary.csv", strata)

    # Human-readable highlights: all/full, all/no-context, genuine subset at alpha .75 and best full alpha.
    all_full = [r for r in strata if r["context_mode"] == "full_context" and r["group"] == "all_pairs"]
    best_full = max(all_full, key=lambda r: (r["joint"], r["a_correct"] + r["b_correct"])) if all_full else None
    summary = {
        "status": "BINDING_ALPHA_NO_CONTEXT_STRATA",
        "created_utc": now(),
        "checkpoint": rel(args.checkpoint),
        "device": str(device),
        "alphas": args.alphas,
        "identities": identities,
        "compact_evals": compact_evals,
        "best_full_context_all_pairs": best_full,
        "files": {
            "pair_outcomes": rel(args.out_dir / "pair_outcomes.csv"),
            "strata_summary": rel(args.out_dir / "strata_summary.csv"),
        },
        "interpretation_keys": {
            "full_context_joint": "paired-null gating count; no-gating null is joint=0 unless both_wrong is high",
            "no_context_joint": "shortcut pressure: entity-answer or answer-prior success without source/update evidence",
            "genuine_both_source_low_answer_overlap": "operational stronger-conflict subset: both entities appear in source and answer word Jaccard < 0.5",
        },
    }
    write_json(args.out_dir / "summary.json", summary)

    # Markdown table of principal rows.
    lines = ["# research binding alpha / no-context / conflict strata", "", f"Checkpoint: `{rel(args.checkpoint)}`", ""]
    lines.append("## All-pair alpha scan")
    lines.append("| alpha | full joint | full A | full B | full both-wrong | no-context joint | no-context A | no-context B | no-context both-wrong |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for alpha in args.alphas:
        f = compact_evals.get(f"alpha{alpha:g}_full_context", {})
        n = compact_evals.get(f"alpha{alpha:g}_no_context", {})
        lines.append(f"| {alpha:g} | {f.get('joint')} | {f.get('a_correct')} | {f.get('b_correct')} | {f.get('both_wrong')} | {n.get('joint')} | {n.get('a_correct')} | {n.get('b_correct')} | {n.get('both_wrong')} |")
    lines.append("")
    lines.append("## Conflict strata (selected)")
    lines.append("| alpha | mode | group | n | joint | gated_frac | A | B | both_wrong |")
    lines.append("|---:|---|---|---:|---:|---:|---:|---:|---:|")
    selected_groups = {"all_pairs", "genuine_both_source_low_answer_overlap", "weak_or_location_separable", "updated_entity_in_source", "updated_entity_not_in_source", "position_updated_first", "position_unchanged_first"}
    for r in strata:
        if r["group"] in selected_groups:
            lines.append(f"| {r['alpha']:g} | {r['context_mode']} | {r['group']} | {r['n']} | {r['joint']} | {r['gated_frac']:.3f} | {r['a_correct']} | {r['b_correct']} | {r['both_wrong']} |")
    (args.out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": summary["status"], "summary": rel(args.out_dir / "summary.json"), "md": rel(args.out_dir / "summary.md"), "best": best_full}, indent=2), flush=True)


if __name__ == "__main__":
    main()
