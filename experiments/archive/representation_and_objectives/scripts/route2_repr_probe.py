#!/usr/bin/env python3
"""research: decoder-robust representation-vs-readout probe on the interference ladder.

Scientific question (Route 2, the only orthogonal lever left after Route1/Route3
closed): for the contradiction+action conditions where the MLM head fails at 0.0
crossed success, is the correct-state information present in the hidden
representation but unreadable at the output head, or is it never formed?

This is NOT the research 65%-trigger probe.  the
probe must isolate the paired interaction and survive held-out and null controls:

  * Splits: leave-one-out by base frame, by object, and by family-actor group,
    so the probe cannot memorize a specific frame instance.
  * Task: predict which context (c1 vs c2) a hidden [MASK] representation came
    from, using ONLY the difference structure that requires reading the action /
    state, per condition.  We phrase it as: given the two rendered contexts of a
    frame (c1 with correct state a, c2 with correct state b), can a linear probe
    on the layer-L [MASK] hidden state separate c1-representations from
    c2-representations better than matched nulls?
  * Nulls:
      - renamed control frames (same structure, different object/actor): a real
        signal should transfer; a frame-instance artifact should not.
      - label-permutation null: shuffle the c1/c2 labels within CV folds.
  * Readout baseline: the MLM-head crossed-success for the same condition, so we
    can compare probe accuracy to head accuracy directly.

A positive Route 2 result requires: contradict_bare probe accuracy clearly above
both nulls AND above the MLM-head crossed success, held out across base frames,
for >=2 checkpoints.  Otherwise the information is not separably formed and the
deficit is representation formation, not readout.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import json
import math
import os
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

USER_ROOT = _public_path('.')
A01_WS = _public_path('experiments/archive/representation_and_objectives')
A02_WS = _public_path('experiments/archive/frontier_consolidation')
OUT_ROOT = _public_path('experiments/archive/representation_and_objectives/data/route2_repr_probe')
CACHE = _public_path('experiments/archive/representation_and_objectives/data/route2_repr_probe/hf_cache')
for name, path in {
    "HOME": _public_path('experiments/archive/representation_and_objectives/data/route2_repr_probe/hf_cache/home'), "XDG_CACHE_HOME": _public_path('experiments/archive/representation_and_objectives/data/route2_repr_probe/hf_cache/xdg'), "HF_HOME": _public_path('experiments/archive/representation_and_objectives/data/route2_repr_probe/hf_cache/hf_home'),
    "HF_MODULES_CACHE": _public_path('experiments/archive/representation_and_objectives/data/route2_repr_probe/hf_cache/hf_modules'), "TRANSFORMERS_CACHE": _public_path('experiments/archive/representation_and_objectives/data/route2_repr_probe/hf_cache/transformers'),
    "TORCH_HOME": _public_path('experiments/archive/representation_and_objectives/data/route2_repr_probe/hf_cache/torch'), "TMPDIR": _public_path('experiments/archive/representation_and_objectives/data/route2_repr_probe/hf_cache/tmp'),
}.items():
    os.environ.setdefault(name, str(path))
    Path(os.environ[name]).mkdir(parents=True, exist_ok=True)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from transformers import AutoModelForMaskedLM, AutoTokenizer

FRAME_PATH = _public_path('experiments/archive/representation_and_objectives/data/interference_ladder/interference_ladder_frames.jsonl')
RENAMED_PATH = _public_path('experiments/archive/representation_and_objectives/data/interference_ladder/interference_ladder_renamed_controls.jsonl')

MODELS = {
    "chck82_scale1p75": _public_path('experiments/archive/frontier_consolidation/training/runs/adapter128_scale1p75_h100M100M_seed43022_official_ladder/hf_model/chck_82M'),
    "legal16k_base100": _public_path('experiments/archive/representation_and_objectives/training/runs/strictsmalltok_compact_view_reinvest_seed43022/hf_model/chck_100M'),
    "scale1p75_100M": _public_path('experiments/archive/frontier_consolidation/training/runs/adapter128_scale1p75_h100M100M_seed43022_official_ladder/hf_model/chck_100M'),
}

CONDITIONS = ["contradict_bare", "last_event", "explicit_final"]


def rel(p: Path | str) -> str:
    pp = Path(p)
    try:
        return str(pp.resolve().relative_to(USER_ROOT))
    except Exception:
        return str(pp)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def render(context: str, query_template: str, target: str) -> tuple[str, tuple[int, int]]:
    pre, post = query_template.split("{target}", 1)
    q = pre + target + post
    lead = (context.strip() + " ") if context.strip() else ""
    sent = lead + q
    return sent, (len(lead) + len(pre), len(lead) + len(pre) + len(target))


def mask_target_position(tok, sent: str, span: tuple[int, int]) -> tuple[list[int], list[int], int, int] | None:
    """Return masked input and the original token id at the target span.

    Offset-based selection is the robust single-token check used by the validated
    research scorer; raw tok.encode(' word') is too strict for some byte-BPE cases.
    """
    enc = tok(sent, return_offsets_mapping=True, return_tensors=None)
    ids = list(enc["input_ids"]); att = list(enc["attention_mask"]); offs = list(enc["offset_mapping"])
    s, e = span
    sel = [i for i, (a, b) in enumerate(offs) if not (a == b == 0) and b > s and a < e]
    if len(sel) != 1:
        return None
    pos = sel[0]
    target_id = int(ids[pos])
    ids2 = list(ids); ids2[pos] = tok.mask_token_id
    return ids2, att, pos, target_id


def context_token_id(tok, word: str) -> int | None:
    """Find token id for a state word in a realistic query sentence."""
    sent = f"The item is now {word}."
    start = sent.index(word); end = start + len(word)
    prep = mask_target_position(tok, sent, (start, end))
    if prep is None:
        return None
    return prep[3]


@torch.inference_mode()
def collect_hidden_and_head(model, tok, device, frames: list[dict[str, Any]], conditions: list[str], batch_size: int = 48):
    """For each frame+condition, collect layerwise [MASK] hidden states for c1 and c2
    (query masked), plus the MLM-head crossed-success flag using the mask logits."""
    n_layers = int(model.config.num_hidden_layers) + 1
    items = []  # (frame_idx, condition, which_ctx, ids, att, pos, target_id, foil_id)
    frame_meta = []
    for fi, r in enumerate(frames):
        if r["condition"] not in conditions:
            continue
        a, b = r["alt_a"], r["alt_b"]
        qt = r["query_template"]
        ida = context_token_id(tok, a)
        idb = context_token_id(tok, b)
        if ida is None or idb is None:
            continue
        frame_meta.append({
            "frame_idx": fi, "frame_id": r["frame_id"], "base_id": r["base_id"],
            "condition": r["condition"], "family": r["family"], "object": r.get("object"),
            "actor": r.get("actor"), "alt_a": a, "alt_b": b,
        })
        for which, ctx, tgt, foil in [("c1", r["context1"], ida, idb), ("c2", r["context2"], idb, ida)]:
            sent, span = render(ctx, qt, r["alt_a"] if which == "c1" else r["alt_b"])
            prep = mask_target_position(tok, sent, span)
            if prep is None:
                items.append(None)
                continue
            ids, att, pos, original_target_id = prep
            # Use offset-derived original token id for the correct target and the
            # independently derived counterpart for the foil.
            target_id = original_target_id
            foil_id = foil
            items.append({"frame_key": (fi, r["condition"]), "which": which, "ids": ids, "att": att,
                          "pos": pos, "target_id": target_id, "foil_id": foil_id})
    # Batch forward
    hidden = defaultdict(dict)   # (frame_idx,cond) -> {"c1": [L x H], "c2": ...}
    head_margin = defaultdict(dict)  # (frame_idx,cond) -> {"c1": logit(correct)-logit(foil)}
    valid_items = [it for it in items if it is not None]
    pad = tok.pad_token_id if tok.pad_token_id is not None else 0
    valid_items.sort(key=lambda x: len(x["ids"]))
    for st in range(0, len(valid_items), batch_size):
        batch = valid_items[st:st + batch_size]
        ml = max(len(x["ids"]) for x in batch)
        ids_t = torch.tensor([x["ids"] + [pad] * (ml - len(x["ids"])) for x in batch], dtype=torch.long, device=device)
        att_t = torch.tensor([x["att"] + [0] * (ml - len(x["att"])) for x in batch], dtype=torch.long, device=device)
        pos_t = torch.tensor([x["pos"] for x in batch], dtype=torch.long, device=device)
        out = model(input_ids=ids_t, attention_mask=att_t, output_hidden_states=True, return_dict=True)
        mb = torch.arange(ids_t.shape[0], device=device)
        hs = out.hidden_states  # tuple len n_layers
        logits = out.logits[mb, pos_t]  # [B, V]
        for bi, it in enumerate(batch):
            key = it["frame_key"]; which = it["which"]
            layer_vecs = np.stack([hs[li][bi, it["pos"], :].float().cpu().numpy() for li in range(n_layers)], axis=0)
            hidden[key][which] = layer_vecs
            head_margin[key][which] = float((logits[bi, it["target_id"]] - logits[bi, it["foil_id"]]).cpu())
    return hidden, head_margin, frame_meta, n_layers


def logistic_cv_by_group(X: np.ndarray, y: np.ndarray, groups: np.ndarray, permute: bool = False, seed: int = 7) -> float:
    """Leave-one-group-out CV accuracy of a linear probe.  If permute, shuffle y within
    the whole set (label-permutation null) before CV."""
    rng = np.random.RandomState(seed)
    yv = y.copy()
    if permute:
        yv = yv.copy()
        rng.shuffle(yv)
    uniq = np.unique(groups)
    correct = 0; total = 0
    for g in uniq:
        test = groups == g
        train = ~test
        if len(np.unique(yv[train])) < 2 or test.sum() == 0:
            continue
        clf = LogisticRegression(max_iter=2000, C=1.0)
        # Standardize using train stats to stabilize.
        mu = X[train].mean(0); sd = X[train].std(0) + 1e-6
        clf.fit((X[train] - mu) / sd, yv[train])
        pred = clf.predict((X[test] - mu) / sd)
        correct += int((pred == yv[test]).sum()); total += int(test.sum())
    return correct / total if total else float("nan")


def build_probe_matrix(hidden: dict, frame_meta: list[dict], condition: str, layer: int):
    """Rows: one per (frame, context). Feature: layer-L [MASK] hidden state.
    Label: 0 for c1, 1 for c2. Group: base_id (leave-one-base-out)."""
    X = []; y = []; grp_base = []; grp_obj = []
    for fm in frame_meta:
        if fm["condition"] != condition:
            continue
        key = (fm["frame_idx"], condition)
        h = hidden.get(key)
        if not h or "c1" not in h or "c2" not in h:
            continue
        X.append(h["c1"][layer]); y.append(0); grp_base.append(fm["base_id"]); grp_obj.append(fm["object"])
        X.append(h["c2"][layer]); y.append(1); grp_base.append(fm["base_id"]); grp_obj.append(fm["object"])
    if not X:
        return None
    return np.array(X), np.array(y), np.array(grp_base), np.array(grp_obj)


def head_crossed_success(head_margin: dict, frame_meta: list[dict], condition: str) -> float:
    ok = 0; n = 0
    for fm in frame_meta:
        if fm["condition"] != condition:
            continue
        hm = head_margin.get((fm["frame_idx"], condition))
        if not hm or "c1" not in hm or "c2" not in hm:
            continue
        n += 1
        if hm["c1"] > 0 and hm["c2"] > 0:
            ok += 1
    return ok / n if n else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=list(MODELS), choices=list(MODELS))
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    frames = read_jsonl(FRAME_PATH)
    renamed = read_jsonl(RENAMED_PATH)

    results = {"status": "ROUTE2_REPR_PROBE", "conditions": CONDITIONS, "models": {}}
    for mname in args.models:
        mpath = MODELS[mname]
        print(json.dumps({"event": "load_model", "model": mname, "path": rel(mpath)}), flush=True)
        tok = AutoTokenizer.from_pretrained(str(mpath), trust_remote_code=True, use_fast=True)
        model = AutoModelForMaskedLM.from_pretrained(str(mpath), trust_remote_code=True).to(device).eval()

        hidden, head_margin, fmeta, n_layers = collect_hidden_and_head(model, tok, device, frames, CONDITIONS)
        rhidden, rhead, rfmeta, _ = collect_hidden_and_head(model, tok, device, renamed, CONDITIONS)

        mres = {"model_path": rel(mpath), "n_layers": n_layers, "conditions": {}}
        for cond in CONDITIONS:
            head_cs = head_crossed_success(head_margin, fmeta, cond)
            layer_rows = []
            for layer in range(n_layers):
                built = build_probe_matrix(hidden, fmeta, cond, layer)
                if built is None:
                    continue
                X, y, gb, go = built
                acc_base = logistic_cv_by_group(X, y, gb, permute=False)
                acc_obj = logistic_cv_by_group(X, y, go, permute=False)
                acc_perm = logistic_cv_by_group(X, y, gb, permute=True)
                # Renamed transfer: train on original, test on renamed at same layer.
                rbuilt = build_probe_matrix(rhidden, rfmeta, cond, layer)
                acc_renamed_transfer = float("nan")
                if rbuilt is not None:
                    Xr, yr, _, _ = rbuilt
                    mu = X.mean(0); sd = X.std(0) + 1e-6
                    clf = LogisticRegression(max_iter=2000, C=1.0)
                    if len(np.unique(y)) == 2:
                        clf.fit((X - mu) / sd, y)
                        pr = clf.predict((Xr - mu) / sd)
                        acc_renamed_transfer = float((pr == yr).mean())
                layer_rows.append({
                    "layer": layer, "n_pairs": int(len(y) // 2),
                    "acc_loo_base": acc_base, "acc_loo_object": acc_obj,
                    "acc_label_perm_null": acc_perm, "acc_renamed_transfer": acc_renamed_transfer,
                })
            best = max((r for r in layer_rows), key=lambda r: (r["acc_loo_base"] if math.isfinite(r["acc_loo_base"]) else -1), default=None)
            mres["conditions"][cond] = {
                "mlm_head_crossed_success": head_cs,
                "layers": layer_rows,
                "best_layer_by_loo_base": best,
            }
            if best:
                print(json.dumps({"event": "cond_summary", "model": mname, "cond": cond,
                                  "head_cs": round(head_cs, 3),
                                  "best_layer": best["layer"],
                                  "acc_loo_base": round(best["acc_loo_base"], 3),
                                  "acc_perm_null": round(best["acc_label_perm_null"], 3),
                                  "acc_renamed_transfer": round(best["acc_renamed_transfer"], 3)}), flush=True)
        results["models"][mname] = mres
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    out_json = _public_path('experiments/archive/representation_and_objectives/data/route2_repr_probe/route2_repr_probe.json')
    out_json.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": results["status"], "out_json": rel(out_json)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
