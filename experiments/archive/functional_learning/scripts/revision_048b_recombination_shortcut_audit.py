#!/usr/bin/env python3
"""research: shortcut audit for research recombination export.

The recombination rows are valuable as a descriptive cross-format screen, but a
pair can pass the full-phrase contract without contextual state binding if the
query entity and candidate phrases already make one assignment semantically much
more plausible than the other.  This script scores the same source_state/new_state
candidate contrast under multiple information regimes:

  no_context:  The relevant state of {query_entity} is {STATE}.
  source_only: source_sentence + final frame
  update_only: update_sentence + final frame
  full:        source_sentence + update_sentence + final frame

If no_context already satisfies row_a source>new and row_b new>source, the pair is
not a decisive selector-transfer test even if full scoring succeeds.  Full-only
improvement is descriptive evidence of using the presented context, but because
these rows are not recipient-only reversals it still should not be equated with
transfer of the relation-first selector.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import csv
import gc
import json
import math
import pathlib
import sys
import time
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

ROOT = _public_path('.')
STUDY = _public_path('experiments/archive/functional_learning')
SCRIPTS = _public_path('experiments/archive/functional_learning/scripts')
sys.path.insert(0, str(SCRIPTS))

import bridge_eval as bridge_eval  # noqa: E402

PARENT_PATH = _public_path('models/frontier')
SPECIALIST_PATH = _public_path('experiments/archive/functional_learning/data/saved_state_replicate_neutral_threeentity/seed_40040/hf_model/final')
A02_EXPORT = _public_path('experiments/archive/relation_learning/data/recombination_rows')
DEFAULT_BINDING_PAIRS = _public_path('experiments/archive/relation_learning/data/recombination_rows/binding_pairs_heldout.jsonl')
DEFAULT_HELDOUT_ROWS = _public_path('experiments/archive/relation_learning/data/recombination_rows/recombination_heldout.jsonl')
DEFAULT_STRATA = _public_path('experiments/archive/relation_learning/data/binding_pair_strata_and_export/heldout_binding_pair_source_presence.csv')


def rel(path: pathlib.Path | str) -> str:
    try:
        return str(pathlib.Path(path).resolve().relative_to(ROOT))
    except Exception:
        return str(path)


def read_jsonl(path: pathlib.Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: pathlib.Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def finite_mean(xs: Iterable[float]) -> float:
    vals = [float(x) for x in xs if x is not None and math.isfinite(float(x))]
    return sum(vals) / len(vals) if vals else float("nan")


def read_strata(path: pathlib.Path) -> Dict[str, Dict[str, Any]]:
    if not path.exists():
        return {}
    out: Dict[str, Dict[str, Any]] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            clean = dict(row)
            for k in ["target_entity_in_source", "updated_entity_in_source", "both_entities_in_source", "same_source_update_context"]:
                if k in clean:
                    clean[k] = str(clean[k]).strip().lower() in {"1", "true", "yes"}
            out[clean["pair_id"]] = clean
    return out


def locate_span_positions(offsets: Sequence[Tuple[int, int]], start: int, end: int) -> List[int]:
    pos: List[int] = []
    for i, (s, e) in enumerate(offsets):
        s, e = int(s), int(e)
        if e <= s:
            continue
        if s < end and e > start:
            pos.append(i)
    return pos


def make_item(tokenizer, frame: str, candidate: str, seq_length: int, device: torch.device) -> Dict[str, Any]:
    if frame.count("{STATE}") != 1:
        raise ValueError(f"bad frame with {frame.count('{STATE}')} slots: {frame[:120]!r}")
    before, after = frame.split("{STATE}")
    full = before + candidate + after
    start = len(before)
    end = start + len(candidate)
    enc = tokenizer(full, add_special_tokens=True, return_offsets_mapping=True, return_tensors="pt", max_length=seq_length, truncation=True, padding="max_length")
    offsets = [(int(a), int(b)) for a, b in enc["offset_mapping"].squeeze(0).tolist()]
    pos = locate_span_positions(offsets, start, end)
    if not pos:
        raise ValueError(f"candidate span unavailable: {candidate!r}")
    if min(offsets[p][0] for p in pos) > start or max(offsets[p][1] for p in pos) < end:
        raise ValueError(f"candidate span partial: {candidate!r}")
    input_ids = enc["input_ids"].squeeze(0)
    masked = input_ids.clone()
    target = input_ids[pos].clone()
    for p in pos:
        masked[p] = int(tokenizer.mask_token_id)
    return {
        "input_ids": masked.to(device),
        "attention_mask": enc["attention_mask"].squeeze(0).to(device),
        "positions": torch.tensor(pos, dtype=torch.long, device=device),
        "target_ids": target.to(device),
        "n_tokens": len(pos),
    }


def score_frame(model, tokenizer, frame: str, candidates: Sequence[Tuple[str, str]], device: torch.device, seq_length: int) -> Dict[str, Dict[str, Any]]:
    items = [make_item(tokenizer, frame, cand, seq_length, device) for _, cand in candidates]
    input_ids = torch.stack([x["input_ids"] for x in items], dim=0)
    attention_mask = torch.stack([x["attention_mask"] for x in items], dim=0)
    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
    out = {}
    for bi, ((lab, cand), item) in enumerate(zip(candidates, items)):
        pos = item["positions"]
        target = item["target_ids"]
        lp = F.log_softmax(logits[bi, pos], dim=-1)
        vals = lp[torch.arange(len(pos), device=device), target].detach().cpu().tolist()
        out[lab] = {"candidate": cand, "n_tokens": int(item["n_tokens"]), "sum_logp": float(sum(vals)), "mean_logp": float(sum(vals) / len(vals))}
    return out


def context_frame(row: Dict[str, Any], mode: str) -> str:
    final = f"The relevant state of {row['query_entity']} is {{STATE}}."
    if mode == "no_context":
        return final
    if mode == "source_only":
        return row["source_sentence"].rstrip() + " " + final
    if mode == "update_only":
        return row["update_sentence"].rstrip() + " " + final
    if mode == "full":
        return row["source_sentence"].rstrip() + " " + row["update_sentence"].rstrip() + " " + final
    raise ValueError(f"unknown mode {mode}")


def load_model(label: str, device: torch.device, private_scale: float):
    if label == "parent":
        return bridge_eval.load_parent(device, private_scale)
    if label == "specialist_seed40040":
        return bridge_eval.strict_load_checkpoint(SPECIALIST_PATH, device, private_scale)
    if label.startswith("checkpoint:"):
        p = pathlib.Path(label.split(":", 1)[1])
        if not p.is_absolute():
            p = ROOT / p
        return bridge_eval.strict_load_checkpoint(p, device, private_scale)
    raise ValueError(f"unknown model {label}")


def select_pairs(pairs: List[Dict[str, Any]], pair_ids: Sequence[str], max_pairs: int) -> List[Dict[str, Any]]:
    by_id = {p["pair_id"]: p for p in pairs}
    if pair_ids:
        out = []
        for pid in pair_ids:
            if pid not in by_id:
                raise KeyError(f"pair_id not found: {pid}")
            out.append(by_id[pid])
        return out
    pairs = sorted(pairs, key=lambda p: p["pair_id"])
    if max_pairs and max_pairs > 0:
        return pairs[:max_pairs]
    return pairs


def summarize(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    summary = {}
    keys = ["all", "updated_entity_in_source", "both_entities_in_source", "not_both_entities_in_source"]
    for model in sorted({r["model"] for r in records}):
        mrs = [r for r in records if r["model"] == model]
        for mode in sorted({r["mode"] for r in mrs}):
            rs0 = [r for r in mrs if r["mode"] == mode]
            for subset in keys:
                if subset == "all":
                    rs = rs0
                elif subset == "updated_entity_in_source":
                    rs = [r for r in rs0 if r.get("updated_entity_in_source")]
                elif subset == "both_entities_in_source":
                    rs = [r for r in rs0 if r.get("both_entities_in_source")]
                else:
                    rs = [r for r in rs0 if not r.get("both_entities_in_source")]
                if not rs:
                    continue
                summary[f"{model}::{mode}::{subset}"] = {
                    "model": model,
                    "mode": mode,
                    "subset": subset,
                    "n_pairs": len(rs),
                    "joint_success": sum(1 for r in rs if r["joint_success"]),
                    "row_a_source_gt_new": sum(1 for r in rs if r["row_a_source_gt_new"]),
                    "row_b_new_gt_source": sum(1 for r in rs if r["row_b_new_gt_source"]),
                    "mean_row_a_margin_source_minus_new": finite_mean([r["row_a_margin_source_minus_new"] for r in rs]),
                    "mean_row_b_margin_new_minus_source": finite_mean([r["row_b_margin_new_minus_source"] for r in rs]),
                    "mean_joint_min_margin": finite_mean([min(r["row_a_margin_source_minus_new"], r["row_b_margin_new_minus_source"]) for r in rs]),
                }
    # Pair-level shortcut labels compare no_context and full where both were run.
    pair_records: Dict[Tuple[str, str], Dict[str, Dict[str, Any]]] = {}
    for r in records:
        pair_records.setdefault((r["model"], r["pair_id"]), {})[r["mode"]] = r
    shortcut = []
    full_only = []
    full_failed = []
    for (model, pid), d in pair_records.items():
        if "full" not in d:
            continue
        full = d["full"]
        noctx = d.get("no_context")
        if noctx and noctx["joint_success"]:
            shortcut.append({"model": model, "pair_id": pid, "full_joint": full["joint_success"], "no_context_min_margin": min(noctx["row_a_margin_source_minus_new"], noctx["row_b_margin_new_minus_source"]), "both_entities_in_source": noctx.get("both_entities_in_source")})
        elif noctx and full["joint_success"]:
            full_only.append({"model": model, "pair_id": pid, "full_min_margin": min(full["row_a_margin_source_minus_new"], full["row_b_margin_new_minus_source"]), "both_entities_in_source": full.get("both_entities_in_source")})
        elif not full["joint_success"]:
            full_failed.append({"model": model, "pair_id": pid, "full_min_margin": min(full["row_a_margin_source_minus_new"], full["row_b_margin_new_minus_source"]), "both_entities_in_source": full.get("both_entities_in_source")})
    return {"groups": summary, "no_context_shortcut_pairs": shortcut[:50], "full_only_pairs": full_only[:50], "full_failed_pairs": full_failed[:50]}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--binding-pairs", default=str(DEFAULT_BINDING_PAIRS))
    ap.add_argument("--heldout-rows", default=str(DEFAULT_HELDOUT_ROWS))
    ap.add_argument("--strata", default=str(DEFAULT_STRATA))
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--models", nargs="+", default=["parent"])
    ap.add_argument("--modes", nargs="+", default=["no_context", "full"])
    ap.add_argument("--pair-ids", nargs="*", default=[])
    ap.add_argument("--max-pairs", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--torch-threads", type=int, default=6)
    ap.add_argument("--seq-length", type=int, default=512)
    ap.add_argument("--private-scale", type=float, default=0.75)
    args = ap.parse_args()

    if args.device == "cpu" and args.torch_threads > 0:
        torch.set_num_threads(int(args.torch_threads))
    device = torch.device(args.device if args.device != "cuda" else "cuda:0")
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pairs = read_jsonl(pathlib.Path(args.binding_pairs))
    rows = {r["row_id"]: r for r in read_jsonl(pathlib.Path(args.heldout_rows))}
    strata = read_strata(pathlib.Path(args.strata))
    selected = select_pairs(pairs, args.pair_ids, args.max_pairs)
    tokenizer = AutoTokenizer.from_pretrained(str(PARENT_PATH), local_files_only=True, use_fast=True)

    all_records: List[Dict[str, Any]] = []
    identities: Dict[str, Any] = {}
    errors: List[Dict[str, Any]] = []
    start_time = time.time()
    for model_label in args.models:
        print(json.dumps({"event": "model_start", "model": model_label, "n_pairs": len(selected), "modes": args.modes}), flush=True)
        model, ident = load_model(model_label, device, args.private_scale)
        model.eval()
        identities[model_label] = ident
        for i, pair in enumerate(selected):
            row_a = rows[pair["row_a_id"]]
            row_b = rows[pair["row_b_id"]]
            if str(row_a.get("query_entity")) == str(row_b.get("query_entity")):
                errors.append({"pair_id": pair["pair_id"], "error": "rows have identical query_entity"})
            candidates = [("source_state", row_a["source_state"]), ("new_state", row_a["new_state"])]
            for mode in args.modes:
                try:
                    score_a = score_frame(model, tokenizer, context_frame(row_a, mode), candidates, device, args.seq_length)
                    score_b = score_frame(model, tokenizer, context_frame(row_b, mode), candidates, device, args.seq_length)
                    ma = float(score_a["source_state"]["mean_logp"] - score_a["new_state"]["mean_logp"])
                    mb = float(score_b["new_state"]["mean_logp"] - score_b["source_state"]["mean_logp"])
                    st = strata.get(pair["pair_id"], {})
                    rec = {
                        "model": model_label,
                        "pair_id": pair["pair_id"],
                        "mode": mode,
                        "row_a_id": pair["row_a_id"],
                        "row_b_id": pair["row_b_id"],
                        "query_a": row_a["query_entity"],
                        "query_b": row_b["query_entity"],
                        "source_state": row_a["source_state"],
                        "new_state": row_a["new_state"],
                        "target_entity_in_source": st.get("target_entity_in_source"),
                        "updated_entity_in_source": st.get("updated_entity_in_source"),
                        "both_entities_in_source": st.get("both_entities_in_source"),
                        "row_a_source_gt_new": ma > 0,
                        "row_b_new_gt_source": mb > 0,
                        "joint_success": ma > 0 and mb > 0,
                        "row_a_margin_source_minus_new": ma,
                        "row_b_margin_new_minus_source": mb,
                        "row_a_scores": score_a,
                        "row_b_scores": score_b,
                    }
                    all_records.append(rec)
                except Exception as e:
                    errors.append({"model": model_label, "pair_id": pair["pair_id"], "mode": mode, "error": repr(e)})
            if (i + 1) % 20 == 0:
                print(json.dumps({"event": "score_progress", "model": model_label, "n": i + 1}), flush=True)
        del model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    write_jsonl(out_dir / "recombination_shortcut_records.jsonl", all_records)
    summary = {
        "status": "A02_RECOMBINATION_SHORTCUT_AUDIT_DONE",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "scientific_purpose": "Quantify whether full-phrase binding pairs can be solved by query/candidate priors without contextual state assignment. no_context success marks shortcut susceptibility, not selector transfer.",
        "binding_pairs": rel(args.binding_pairs),
        "heldout_rows": rel(args.heldout_rows),
        "strata": rel(args.strata),
        "n_selected_pairs": len(selected),
        "models": args.models,
        "modes": args.modes,
        "pair_ids": [p["pair_id"] for p in selected],
        "identities": identities,
        "summary": summarize(all_records),
        "n_errors": len(errors),
        "errors": errors[:30],
        "outputs": {
            "records": rel(out_dir / "recombination_shortcut_records.jsonl"),
            "summary": rel(out_dir / "recombination_shortcut_summary.json"),
        },
        "elapsed_sec": round(time.time() - start_time, 3),
    }
    (out_dir / "recombination_shortcut_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
