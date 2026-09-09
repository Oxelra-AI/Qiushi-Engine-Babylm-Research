#!/usr/bin/env python3
"""research: score strict semantic Route 3 pairs and controls on existing checkpoints.

This is a forward-only, no-weight-changing screen.  It extracts the strict
same-head/same-category subset from the research rebuilt pair object and scores
four matched files:

  * true_semantic correspondence
  * target/family shuffled B context
  * family/length-matched shuffled B context
  * target-swapped null

For each pair, the antisymmetric two-context margin is
  delta = [logit(a|C_a)-logit(b|C_a)] + [logit(b|C_b)-logit(a|C_b)].
The paired construction cancels target priors.  The controls test whether the
object contains correspondence-level signal beyond target/family/length balance.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

SCRIPT_DIR = _public_path('experiments/archive/representation_and_objectives/scripts')
WORKSPACE = _public_path('experiments/archive/representation_and_objectives')
STUDY = _public_path('experiments/archive/representation_and_objectives')
USER_ROOT = _public_path('.')
IN_DIR = _public_path('experiments/archive/representation_and_objectives/data/semantic_route3')
OUT_DIR = _public_path('experiments/archive/representation_and_objectives/data/semantic_route3_scores')
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATHS = {
    "chck82_scale1p75": _public_path('experiments/archive/frontier_consolidation/training/runs/adapter128_scale1p75_h100M100M_seed43022_official_ladder/hf_model/chck_82M'),
    "legal16k_base100": _public_path('experiments/archive/representation_and_objectives/training/runs/strictsmalltok_compact_view_reinvest_seed43022/hf_model/chck_100M'),
    "scale1p75_100M": _public_path('experiments/archive/frontier_consolidation/training/runs/adapter128_scale1p75_h100M100M_seed43022_official_ladder/hf_model/chck_100M'),
}

PAIR_FILES = {
    "true_semantic": _public_path('experiments/archive/representation_and_objectives/data/semantic_route3/route3_semantic_true_pairs.jsonl'),
    "target_family_shuffle": _public_path('experiments/archive/representation_and_objectives/data/semantic_route3/route3_control_target_family_shuffle.jsonl'),
    "family_length_matched": _public_path('experiments/archive/representation_and_objectives/data/semantic_route3/route3_control_family_length_matched.jsonl'),
    "target_swapped_null": _public_path('experiments/archive/representation_and_objectives/data/semantic_route3/route3_control_target_swapped_null.jsonl'),
}


def rel(p: Path | str) -> str:
    pp = Path(p)
    try:
        return str(pp.resolve().relative_to(USER_ROOT))
    except Exception:
        return str(pp)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def key_of(p: dict[str, Any]) -> tuple[str, int]:
    return str(p["antonym_pair"]), int(p["semantic_pair_rank"])


def prepare_strict_files() -> dict[str, Path]:
    true_all = read_jsonl(PAIR_FILES["true_semantic"])
    strict_true = [p for p in true_all if p.get("control_type") == "true_semantic"]
    strict_keys = {key_of(p) for p in strict_true}
    out_paths = {"true_semantic": _public_path('experiments/archive/representation_and_objectives/data/semantic_route3_scores/strict_true_semantic.jsonl')}
    write_jsonl(out_paths["true_semantic"], strict_true)
    for name in ["target_family_shuffle", "family_length_matched", "target_swapped_null"]:
        rows = read_jsonl(PAIR_FILES[name])
        by_key = {key_of(p): p for p in rows}
        kept = []
        missing = []
        for p in strict_true:
            k = key_of(p)
            q = by_key.get(k)
            if q is None:
                missing.append(k)
            else:
                kept.append(q)
        if missing:
            print(json.dumps({"warning": "missing_control_rows", "control": name, "n_missing": len(missing), "head": missing[:5]}), flush=True)
        out_paths[name] = OUT_DIR / f"strict_{name}.jsonl"
        write_jsonl(out_paths[name], kept)
    manifest = {
        "status": "STRICT_CONTROL_FILES",
        "strict_true_n": len(strict_true),
        "strict_key_n": len(strict_keys),
        "files": {k: rel(v) for k, v in out_paths.items()},
        "family_counts": dict(sorted(Counter(p["antonym_pair"] for p in strict_true).items())),
        "same_head_frac": sum(1 for p in strict_true if p.get("same_entity_head")) / len(strict_true) if strict_true else None,
        "same_category_frac": sum(1 for p in strict_true if p.get("same_entity_category")) / len(strict_true) if strict_true else None,
    }
    (_public_path('experiments/archive/representation_and_objectives/data/semantic_route3_scores/strict_control_manifest.json')).write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return out_paths


def mask_state(context: str, start: int, end: int, mask_token: str) -> str:
    if not (0 <= start < end <= len(context)):
        raise ValueError(f"bad span {start}:{end} for len {len(context)}")
    return context[:start] + mask_token + context[end:]


def token_id(tok, word: str) -> int | None:
    ids = tok.encode(" " + word, add_special_tokens=False)
    if len(ids) == 1:
        return int(ids[0])
    ids2 = tok.encode(word, add_special_tokens=False)
    if len(ids2) == 1:
        return int(ids2[0])
    return None


def qstats(vals: list[float]) -> dict[str, Any]:
    xs = [float(v) for v in vals if math.isfinite(float(v))]
    if not xs:
        return {"n": 0}
    xs_sorted = sorted(xs)
    return {
        "n": len(xs),
        "mean": statistics.fmean(xs),
        "median": statistics.median(xs),
        "q10": float(np.quantile(xs_sorted, 0.10)),
        "q25": float(np.quantile(xs_sorted, 0.25)),
        "q75": float(np.quantile(xs_sorted, 0.75)),
        "q90": float(np.quantile(xs_sorted, 0.90)),
        "min": xs_sorted[0],
        "max": xs_sorted[-1],
    }


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        return {"n": 0}
    return {
        "n": len(records),
        "crossed_success": sum(1 for r in records if r["margin_a"] > 0 and r["margin_b"] > 0) / len(records),
        "a_success": sum(1 for r in records if r["margin_a"] > 0) / len(records),
        "b_success": sum(1 for r in records if r["margin_b"] > 0) / len(records),
        "delta_positive_frac": sum(1 for r in records if r["delta"] > 0) / len(records),
        "margin_a": qstats([r["margin_a"] for r in records]),
        "margin_b": qstats([r["margin_b"] for r in records]),
        "delta": qstats([r["delta"] for r in records]),
        "min_margin": qstats([min(r["margin_a"], r["margin_b"]) for r in records]),
    }


def score_pair_file(model, tok, device: torch.device, path: Path, batch_size: int, max_length: int) -> dict[str, Any]:
    pairs = read_jsonl(path)
    records: list[dict[str, Any]] = []
    skipped = Counter()
    texts: list[str] = []
    candidates: list[tuple[int, int]] = []
    metas: list[dict[str, Any]] = []

    def flush() -> None:
        nonlocal texts, candidates, metas, records
        if not texts:
            return
        enc = tok(texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        mask_id = tok.mask_token_id
        mask_pos = []
        valid = []
        for ids in enc["input_ids"]:
            pos = (ids == mask_id).nonzero(as_tuple=False).flatten()
            if len(pos) == 1:
                mask_pos.append(int(pos[0])); valid.append(True)
            else:
                mask_pos.append(0); valid.append(False)
        with torch.inference_mode():
            logits = model(**enc).logits
        vals = []
        for i, ((target, foil), ok) in enumerate(zip(candidates, valid)):
            if not ok:
                vals.append(None); continue
            p = mask_pos[i]
            vals.append(float((logits[i, p, target] - logits[i, p, foil]).detach().cpu()))
        for i, meta in enumerate(metas):
            ma, mb = vals[2 * i], vals[2 * i + 1]
            if ma is None or mb is None:
                skipped["mask_missing_or_truncated"] += 1
                continue
            records.append({
                **{k: meta.get(k) for k in ["antonym_pair", "control_type", "semantic_pair_rank", "same_entity_head", "same_entity_category", "entity_head_a", "entity_head_b", "entity_category_a", "entity_category_b"]},
                "margin_a": ma,
                "margin_b": mb,
                "delta": ma + mb,
                "crossed_success": bool(ma > 0 and mb > 0),
            })
        texts = []; candidates = []; metas = []

    for p in pairs:
        ida, idb = token_id(tok, p["target_a"]), token_id(tok, p["target_b"])
        if ida is None or idb is None:
            skipped["multitoken"] += 1
            continue
        try:
            ca = mask_state(p["context_a"], int(p["state_start_a"]), int(p["state_end_a"]), tok.mask_token)
            cb = mask_state(p["context_b"], int(p["state_start_b"]), int(p["state_end_b"]), tok.mask_token)
        except Exception:
            skipped["bad_span"] += 1
            continue
        if ca.count(tok.mask_token) != 1 or cb.count(tok.mask_token) != 1:
            skipped["mask_count"] += 1
            continue
        texts.extend([ca, cb])
        candidates.extend([(ida, idb), (idb, ida)])
        metas.append(p)
        if len(metas) >= batch_size:
            flush()
    flush()

    by_family = defaultdict(list)
    by_ctrl = defaultdict(list)
    for r in records:
        by_family[r["antonym_pair"]].append(r)
        by_ctrl[r.get("control_type", "unknown")].append(r)
    return {
        "path": rel(path),
        "input_pairs": len(pairs),
        "skipped": dict(skipped),
        "overall": summarize(records),
        "by_family": {k: summarize(v) for k, v in sorted(by_family.items())},
        "by_control_type": {k: summarize(v) for k, v in sorted(by_ctrl.items())},
        "records": records,
    }


def paired_control_compare(scored: dict[str, Any]) -> dict[str, Any]:
    # Compare each control against the true rows on common (family, rank).  Positive
    # true_minus_control delta means the real correspondence has higher antisymmetric margin.
    true_recs = scored["pair_sets"]["true_semantic"]["records"]
    true_by_key = {(r["antonym_pair"], int(r["semantic_pair_rank"])): r for r in true_recs}
    out = {}
    for cname, payload in scored["pair_sets"].items():
        if cname == "true_semantic":
            continue
        diffs = []
        for r in payload["records"]:
            k = (r["antonym_pair"], int(r["semantic_pair_rank"]))
            tr = true_by_key.get(k)
            if tr is not None:
                diffs.append(tr["delta"] - r["delta"])
        out[cname] = {
            "n_common": len(diffs),
            "true_minus_control_delta": qstats(diffs),
            "positive_frac": sum(1 for d in diffs if d > 0) / len(diffs) if diffs else None,
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["chck82_scale1p75"], choices=sorted(MODEL_PATHS))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-length", type=int, default=256)
    args = ap.parse_args()

    strict_paths = prepare_strict_files()
    all_results: dict[str, Any] = {
        "status": "SEMANTIC_ROUTE3_CONTROL_SCORE",
        "strict_files": {k: rel(v) for k, v in strict_paths.items()},
        "models": {},
    }
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    for mname in args.models:
        mpath = MODEL_PATHS[mname]
        print(json.dumps({"event": "load_model", "model_name": mname, "model_path": rel(mpath), "device": str(device)}), flush=True)
        tok = AutoTokenizer.from_pretrained(str(mpath), trust_remote_code=True, use_fast=True)
        model = AutoModelForMaskedLM.from_pretrained(str(mpath), trust_remote_code=True).to(device).eval()
        scored = {"model_path": rel(mpath), "pair_sets": {}}
        for pname, ppath in strict_paths.items():
            scored["pair_sets"][pname] = score_pair_file(model, tok, device, ppath, args.batch_size, args.max_length)
            print(json.dumps({"event": "scored", "model": mname, "pair_set": pname, "overall": scored["pair_sets"][pname]["overall"], "skipped": scored["pair_sets"][pname]["skipped"]}, default=str), flush=True)
        scored["control_comparison"] = paired_control_compare(scored)
        all_results["models"][mname] = scored
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    out_json = _public_path('experiments/archive/representation_and_objectives/data/semantic_route3_scores/semantic_route3_control_scores.json')
    out_json.write_text(json.dumps(all_results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": all_results["status"], "out_json": rel(out_json), "models": list(all_results["models"].keys())}, indent=2), flush=True)


if __name__ == "__main__":
    main()
