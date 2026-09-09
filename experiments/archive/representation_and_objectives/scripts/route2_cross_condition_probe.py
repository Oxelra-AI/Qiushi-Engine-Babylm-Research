#!/usr/bin/env python3
"""research: cross-condition hidden-state probe for action-vs-prior information.

The same-condition logistic probe was trivially perfect because c1/c2 contexts have
obvious lexical differences.  This script isolates the important question:

  Does an action-derived final-state code learned from action-only contexts
  (last_event) transfer to contradictory-prior contexts (contradict_bare), where
  the MLM head has 0.0 crossed success?

Controls:
  * source conditions explicit_final and consistent_prior show whether state/prior
    lexical codes invert or transfer.
  * target conditions contradict_bare/temporal/reinforced test increasing prior
    interference.
  * grouped transfer holds out base_id or object: train on source condition with
    that group excluded, test on target condition for that group.
  * renamed controls: train on original source and test renamed targets.
  * label permutation null: shuffle source labels before transfer.

This is forward-only and changes no weights.
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
OUT_ROOT = _public_path('experiments/archive/representation_and_objectives/data/route2_cross_condition_probe')
CACHE = _public_path('experiments/archive/representation_and_objectives/data/route2_cross_condition_probe/hf_cache')
for name, path in {
    "HOME": _public_path('experiments/archive/representation_and_objectives/data/route2_cross_condition_probe/hf_cache/home'), "XDG_CACHE_HOME": _public_path('experiments/archive/representation_and_objectives/data/route2_cross_condition_probe/hf_cache/xdg'), "HF_HOME": _public_path('experiments/archive/representation_and_objectives/data/route2_cross_condition_probe/hf_cache/hf_home'),
    "HF_MODULES_CACHE": _public_path('experiments/archive/representation_and_objectives/data/route2_cross_condition_probe/hf_cache/hf_modules'), "TRANSFORMERS_CACHE": _public_path('experiments/archive/representation_and_objectives/data/route2_cross_condition_probe/hf_cache/transformers'),
    "TORCH_HOME": _public_path('experiments/archive/representation_and_objectives/data/route2_cross_condition_probe/hf_cache/torch'), "TMPDIR": _public_path('experiments/archive/representation_and_objectives/data/route2_cross_condition_probe/hf_cache/tmp'),
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
SOURCE_CONDITIONS = ["last_event", "explicit_final", "consistent_prior"]
TARGET_CONDITIONS = ["contradict_bare", "contradict_temporal", "contradict_reinforced"]
ALL_CONDITIONS = sorted(set(SOURCE_CONDITIONS + TARGET_CONDITIONS))


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
    lead = (context.strip() + " ") if context.strip() else ""
    sent = lead + pre + target + post
    return sent, (len(lead) + len(pre), len(lead) + len(pre) + len(target))


def mask_target_position(tok, sent: str, span: tuple[int, int]) -> tuple[list[int], list[int], int, int] | None:
    enc = tok(sent, return_offsets_mapping=True, return_tensors=None)
    ids = list(enc["input_ids"]); att = list(enc["attention_mask"]); offs = list(enc["offset_mapping"])
    s, e = span
    sel = [i for i, (a, b) in enumerate(offs) if not (a == b == 0) and b > s and a < e]
    if len(sel) != 1:
        return None
    pos = sel[0]
    tid = int(ids[pos])
    ids2 = list(ids); ids2[pos] = int(tok.mask_token_id)
    return ids2, att, pos, tid


def context_token_id(tok, word: str) -> int | None:
    sent = f"The item is now {word}."
    start = sent.index(word); end = start + len(word)
    prep = mask_target_position(tok, sent, (start, end))
    return None if prep is None else prep[3]


@torch.inference_mode()
def collect(model, tok, device, frames: list[dict[str, Any]], conditions: list[str], batch_size: int = 48):
    n_layers = int(model.config.num_hidden_layers) + 1
    items = []
    meta_by_key = {}
    for r in frames:
        cond = r["condition"]
        if cond not in conditions:
            continue
        ida = context_token_id(tok, r["alt_a"]); idb = context_token_id(tok, r["alt_b"])
        if ida is None or idb is None:
            continue
        key = (r["base_id"], cond)
        meta_by_key[key] = {"base_id": r["base_id"], "frame_id": r["frame_id"], "condition": cond,
                            "family": r["family"], "object": r.get("object"), "actor": r.get("actor"),
                            "alt_a": r["alt_a"], "alt_b": r["alt_b"]}
        for which, ctx, label, correct, foil in [
            ("c1", r["context1"], 0, ida, idb),
            ("c2", r["context2"], 1, idb, ida),
        ]:
            # Render with correct target then mask it; label is final-state polarity.
            target_word = r["alt_a"] if label == 0 else r["alt_b"]
            sent, span = render(ctx, r["query_template"], target_word)
            prep = mask_target_position(tok, sent, span)
            if prep is None:
                continue
            ids, att, pos, orig_tid = prep
            items.append({"key": key, "which": which, "ids": ids, "att": att, "pos": pos,
                          "label": label, "correct_id": orig_tid, "foil_id": foil})
    pad = tok.pad_token_id if tok.pad_token_id is not None else 0
    rows = []
    items.sort(key=lambda x: len(x["ids"]))
    for st in range(0, len(items), batch_size):
        batch = items[st:st + batch_size]
        ml = max(len(x["ids"]) for x in batch)
        ids_t = torch.tensor([x["ids"] + [pad] * (ml - len(x["ids"])) for x in batch], dtype=torch.long, device=device)
        att_t = torch.tensor([x["att"] + [0] * (ml - len(x["att"])) for x in batch], dtype=torch.long, device=device)
        pos_t = torch.tensor([x["pos"] for x in batch], dtype=torch.long, device=device)
        out = model(input_ids=ids_t, attention_mask=att_t, output_hidden_states=True, return_dict=True)
        mb = torch.arange(ids_t.shape[0], device=device)
        logits = out.logits[mb, pos_t]
        for bi, it in enumerate(batch):
            meta = meta_by_key[it["key"]]
            hs = np.stack([out.hidden_states[li][bi, it["pos"], :].float().cpu().numpy() for li in range(n_layers)], axis=0)
            margin = float((logits[bi, it["correct_id"]] - logits[bi, it["foil_id"]]).cpu())
            rows.append({**meta, "which": it["which"], "label": it["label"], "hidden": hs, "head_margin": margin})
    return rows, n_layers


def matrix(rows: list[dict[str, Any]], cond: str, layer: int):
    sub = [r for r in rows if r["condition"] == cond]
    if not sub:
        return None
    X = np.stack([r["hidden"][layer] for r in sub], axis=0)
    y = np.array([int(r["label"]) for r in sub])
    gb = np.array([str(r["base_id"]) for r in sub])
    go = np.array([str(r["object"]) for r in sub])
    return X, y, gb, go


def train_test_acc(Xtr, ytr, Xte, yte, permute: bool = False, seed: int = 0) -> float:
    ytrain = ytr.copy()
    if permute:
        rng = np.random.RandomState(seed); rng.shuffle(ytrain)
    if len(np.unique(ytrain)) < 2:
        return float("nan")
    mu = Xtr.mean(0); sd = Xtr.std(0) + 1e-6
    clf = LogisticRegression(max_iter=2000, C=1.0)
    clf.fit((Xtr - mu) / sd, ytrain)
    pred = clf.predict((Xte - mu) / sd)
    return float((pred == yte).mean())


def grouped_transfer(rows, source: str, target: str, layer: int, group_kind: str, permute: bool = False) -> float:
    src = matrix(rows, source, layer); tgt = matrix(rows, target, layer)
    if src is None or tgt is None:
        return float("nan")
    Xs, ys, gbs, gos = src; Xt, yt, gbt, got = tgt
    gs = gbs if group_kind == "base" else gos
    gt = gbt if group_kind == "base" else got
    uniq = sorted(set(gt.tolist()))
    correct = 0; total = 0
    for g in uniq:
        tr = gs != g
        te = gt == g
        if not tr.any() or not te.any() or len(np.unique(ys[tr])) < 2:
            continue
        acc = train_test_acc(Xs[tr], ys[tr], Xt[te], yt[te], permute=permute, seed=13 + layer)
        # recover counts from acc; each held group has integer count.
        n = int(te.sum())
        correct += int(round(acc * n)); total += n
    return correct / total if total else float("nan")


def renamed_transfer(train_rows, renamed_rows, source: str, target: str, layer: int, permute: bool = False) -> float:
    src = matrix(train_rows, source, layer); tgt = matrix(renamed_rows, target, layer)
    if src is None or tgt is None:
        return float("nan")
    Xs, ys, _, _ = src; Xt, yt, _, _ = tgt
    return train_test_acc(Xs, ys, Xt, yt, permute=permute, seed=99 + layer)


def head_crossed(rows: list[dict[str, Any]], cond: str) -> float:
    by_base = defaultdict(dict)
    for r in rows:
        if r["condition"] == cond:
            by_base[r["base_id"]][r["which"]] = r["head_margin"]
    vals = [v for v in by_base.values() if "c1" in v and "c2" in v]
    return sum(1 for v in vals if v["c1"] > 0 and v["c2"] > 0) / len(vals) if vals else float("nan")


def summarize_layer_rows(layer_rows: list[dict[str, Any]], key: str) -> dict[str, Any] | None:
    vals = [r for r in layer_rows if math.isfinite(float(r.get(key, float("nan"))))]
    if not vals:
        return None
    best = max(vals, key=lambda r: r[key])
    return {"best_layer": best["layer"], "best_value": best[key], "all_values": [r[key] for r in layer_rows]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=list(MODELS), choices=list(MODELS))
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    frames = read_jsonl(FRAME_PATH)
    renamed = read_jsonl(RENAMED_PATH)
    results = {"status": "ROUTE2_CROSS_CONDITION_PROBE", "models": {}, "source_conditions": SOURCE_CONDITIONS, "target_conditions": TARGET_CONDITIONS}
    for mname in args.models:
        mpath = MODELS[mname]
        print(json.dumps({"event": "load_model", "model": mname, "path": rel(mpath)}), flush=True)
        tok = AutoTokenizer.from_pretrained(str(mpath), trust_remote_code=True, use_fast=True)
        model = AutoModelForMaskedLM.from_pretrained(str(mpath), trust_remote_code=True).to(device).eval()
        rows, n_layers = collect(model, tok, device, frames, ALL_CONDITIONS)
        rrows, _ = collect(model, tok, device, renamed, ALL_CONDITIONS)
        mres = {"model_path": rel(mpath), "n_layers": n_layers, "head_crossed": {c: head_crossed(rows, c) for c in ALL_CONDITIONS}, "transfers": {}}
        for source in SOURCE_CONDITIONS:
            for target in TARGET_CONDITIONS:
                k = f"{source}_to_{target}"
                layer_rows = []
                for layer in range(n_layers):
                    row = {
                        "layer": layer,
                        "acc_base_heldout": grouped_transfer(rows, source, target, layer, "base", permute=False),
                        "acc_object_heldout": grouped_transfer(rows, source, target, layer, "object", permute=False),
                        "acc_renamed_transfer": renamed_transfer(rows, rrows, source, target, layer, permute=False),
                        "perm_base_null": grouped_transfer(rows, source, target, layer, "base", permute=True),
                        "perm_renamed_null": renamed_transfer(rows, rrows, source, target, layer, permute=True),
                    }
                    row["swapped_label_acc_base"] = 1.0 - row["acc_base_heldout"] if math.isfinite(row["acc_base_heldout"]) else float("nan")
                    layer_rows.append(row)
                mres["transfers"][k] = {
                    "layers": layer_rows,
                    "best_base": summarize_layer_rows(layer_rows, "acc_base_heldout"),
                    "best_object": summarize_layer_rows(layer_rows, "acc_object_heldout"),
                    "best_renamed": summarize_layer_rows(layer_rows, "acc_renamed_transfer"),
                }
                bb = mres["transfers"][k]["best_base"]
                br = mres["transfers"][k]["best_renamed"]
                print(json.dumps({"event": "transfer", "model": mname, "transfer": k,
                                  "head_target": round(mres["head_crossed"][target], 3),
                                  "best_base_layer": None if bb is None else bb["best_layer"],
                                  "best_base_acc": None if bb is None else round(bb["best_value"], 3),
                                  "best_renamed_acc": None if br is None else round(br["best_value"], 3)}), flush=True)
        results["models"][mname] = mres
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    out = _public_path('experiments/archive/representation_and_objectives/data/route2_cross_condition_probe/route2_cross_condition_probe.json')
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": results["status"], "out_json": rel(out)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
