#!/usr/bin/env python3
"""research: factorial and causal entity-binding probe for the Route-2 lead.

Forward-only experiment on existing BabyLM checkpoints.  It addresses the research
ambiguity: a last_event-trained linear direction transfers to contradictory-prior
hidden states, but that could be an action/state/template cue rather than an
entity-bound result-state representation.

Experiment:
  * Build matched two-entity action/state frames.  Action/state tokens and sequence
    geometry are held fixed while the affected entity and queried entity swap.
  * Decode final-state labels and affected-query labels from hidden states at the
    mask position under held-out family and held-out action splits.
  * Derive a pre-fixed affected-query component at layers 4/5/6 and causally patch
    or remove it at the layer transition.  Compare against random and swapped-sign
    directions.  No weights are updated.

Outputs live under data/factorial_entity_binding_probe/<model>/.
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
import random
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

USER_ROOT = _public_path('.')
A01_WS = _public_path('experiments/archive/representation_and_objectives')
A02_WS = _public_path('experiments/archive/frontier_consolidation')
OUT_ROOT = _public_path('experiments/archive/representation_and_objectives/data/factorial_entity_binding_probe')
CACHE = _public_path('experiments/archive/representation_and_objectives/data/factorial_entity_binding_probe/hf_cache')
for name, path in {
    "HOME": _public_path('experiments/archive/representation_and_objectives/data/factorial_entity_binding_probe/hf_cache/home'),
    "XDG_CACHE_HOME": _public_path('experiments/archive/representation_and_objectives/data/factorial_entity_binding_probe/hf_cache/xdg'),
    "HF_HOME": _public_path('experiments/archive/representation_and_objectives/data/factorial_entity_binding_probe/hf_cache/hf_home'),
    "HF_MODULES_CACHE": _public_path('experiments/archive/representation_and_objectives/data/factorial_entity_binding_probe/hf_cache/hf_modules'),
    "TRANSFORMERS_CACHE": _public_path('experiments/archive/representation_and_objectives/data/factorial_entity_binding_probe/hf_cache/transformers'),
    "TORCH_HOME": _public_path('experiments/archive/representation_and_objectives/data/factorial_entity_binding_probe/hf_cache/torch'),
    "TMPDIR": _public_path('experiments/archive/representation_and_objectives/data/factorial_entity_binding_probe/hf_cache/tmp'),
}.items():
    os.environ.setdefault(name, str(path))
    Path(os.environ[name]).mkdir(parents=True, exist_ok=True)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from transformers import AutoModelForMaskedLM, AutoTokenizer

MODELS = {
    "chck82_scale1p75": _public_path('experiments/archive/frontier_consolidation/training/runs/adapter128_scale1p75_h100M100M_seed43022_official_ladder/hf_model/chck_82M'),
    "legal16k_base100": _public_path('experiments/archive/representation_and_objectives/training/runs/strictsmalltok_compact_view_reinvest_seed43022/hf_model/chck_100M'),
    "scale1p75_100M": _public_path('experiments/archive/frontier_consolidation/training/runs/adapter128_scale1p75_h100M100M_seed43022_official_ladder/hf_model/chck_100M'),
}

# Each family has state0/state1 and event0/event1 where eventi normally produces statei.
# Candidate actions need not be one-token; target states must be one token in the tested tokenizer.
FAMILIES = [
    {"family": "open_closed", "state0": "open", "state1": "closed", "event0": "opened", "event1": "closed", "train_group": "train"},
    {"family": "empty_full", "state0": "empty", "state1": "full", "event0": "emptied", "event1": "filled", "train_group": "train"},
    {"family": "wet_dry", "state0": "wet", "state1": "dry", "event0": "wetted", "event1": "dried", "train_group": "train"},
    {"family": "clean_dirty", "state0": "clean", "state1": "dirty", "event0": "cleaned", "event1": "dirtied", "train_group": "heldout_family"},
    {"family": "warm_cool", "state0": "warm", "state1": "cool", "event0": "warmed", "event1": "cooled", "train_group": "heldout_family"},
    {"family": "loose_tight", "state0": "loose", "state1": "tight", "event0": "loosened", "event1": "tightened", "train_group": "heldout_family"},
]

OBJECTS = ["box", "cup", "bag", "bottle", "door", "drawer", "window", "jar", "pot", "pan"]
ACTORS = ["Kate", "Jack", "Anna", "Emma", "Mary", "Seth", "Amy", "Tom"]
# Held-out actors only test name/position robustness, not the scientific object.
HELDOUT_ACTORS = {"Amy", "Tom"}


def rel(p: Path | str) -> str:
    pp = Path(p)
    try:
        return str(pp.resolve().relative_to(USER_ROOT))
    except Exception:
        return str(pp)


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def qstats(vals) -> dict[str, Any]:
    xs = sorted(float(v) for v in vals if isinstance(v, (int, float)) and math.isfinite(float(v)))
    if not xs:
        return {"n": 0}
    def q(p: float) -> float:
        if len(xs) == 1:
            return xs[0]
        idx = p * (len(xs) - 1)
        lo, hi = math.floor(idx), math.ceil(idx)
        if lo == hi:
            return xs[lo]
        return xs[lo] * (hi - idx) + xs[hi] * (idx - lo)
    return {"n": len(xs), "mean": statistics.fmean(xs), "median": statistics.median(xs),
            "p10": q(0.1), "p25": q(0.25), "p75": q(0.75), "p90": q(0.9), "min": xs[0], "max": xs[-1]}


def render(ctx: str, query: str) -> tuple[str, tuple[int, int]]:
    # Query is "The obj is now {target}."; this function renders with the explicit
    # target for offset discovery; it will be masked before scoring/representation.
    pre, post = query.split("{target}", 1)
    lead = (ctx.strip() + " ") if ctx.strip() else ""
    sent = lead + pre + "TARGETTOKEN" + post
    s = len(lead) + len(pre)
    return sent, (s, s + len("TARGETTOKEN"))


def mask_target(tok, sent: str, span: tuple[int, int], target_word: str):
    sent2 = sent.replace("TARGETTOKEN", target_word)
    # span changes only by target length replacing TARGETTOKEN
    s = span[0]; e = s + len(target_word)
    enc = tok(sent2, return_offsets_mapping=True, return_tensors=None)
    ids = list(enc["input_ids"]); att = list(enc["attention_mask"]); offs = list(enc["offset_mapping"])
    selected = [i for i, (a, b) in enumerate(offs) if not (a == b == 0) and b > s and a < e]
    if len(selected) != 1:
        return None
    pos = selected[0]
    tid = int(ids[pos])
    ids[pos] = int(tok.mask_token_id)
    return ids, att, pos, tid


def single_target_id(tok, word: str) -> int | None:
    ctx = "The item is now {target}."
    sent, span = render("", ctx)
    prep = mask_target(tok, sent, span, word)
    if prep is None:
        return None
    return prep[3]


def build_frames(tok, max_cases: int = 0) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rng = random.Random(19601)
    target_check = {}
    usable_fams = []
    for fam in FAMILIES:
        ids = {fam["state0"]: single_target_id(tok, fam["state0"]), fam["state1"]: single_target_id(tok, fam["state1"])}
        ok = all(v is not None for v in ids.values())
        target_check[fam["family"]] = {"ok": ok, "target_ids": ids}
        if ok:
            usable_fams.append(fam)
    if not usable_fams:
        raise RuntimeError("no usable one-token target families")

    frames = []
    counter = 0
    obj_pairs = [(OBJECTS[i], OBJECTS[i+1]) for i in range(0, len(OBJECTS)-1, 2)]
    for fam in usable_fams:
        for pair_i, (obj_a, obj_b) in enumerate(obj_pairs):
            for actor in ACTORS:
                for event_pol in [0, 1]:
                    result_state = fam[f"state{event_pol}"]
                    prior_state = fam[f"state{1-event_pol}"]
                    event = fam[f"event{event_pol}"]
                    for affected_slot, affected_obj in [("A", obj_a), ("B", obj_b)]:
                        other_obj = obj_b if affected_slot == "A" else obj_a
                        # Both objects are explicitly set to the opposite prior; the action updates exactly one.
                        ctx = f"The {obj_a} was {prior_state}. The {obj_b} was {prior_state}. {actor} {event} the {affected_obj}."
                        for query_slot, query_obj in [("A", obj_a), ("B", obj_b)]:
                            affected_query = int(query_slot == affected_slot)
                            final_pol = event_pol if affected_query else 1 - event_pol
                            correct = fam[f"state{final_pol}"]
                            foil = fam[f"state{1-final_pol}"]
                            counter += 1
                            frames.append({
                                "case_id": f"F196_{counter:05d}",
                                "family": fam["family"],
                                "train_group": fam["train_group"],
                                "actor": actor,
                                "actor_group": "heldout_actor" if actor in HELDOUT_ACTORS else "train_actor",
                                "obj_pair": f"{obj_a}/{obj_b}",
                                "obj_a": obj_a,
                                "obj_b": obj_b,
                                "affected_slot": affected_slot,
                                "query_slot": query_slot,
                                "affected_obj": affected_obj,
                                "query_obj": query_obj,
                                "affected_query": affected_query,
                                "event_pol": event_pol,
                                "final_pol": final_pol,
                                "prior_state": prior_state,
                                "result_state": result_state,
                                "event": event,
                                "context": ctx,
                                "query_template": f"The {query_obj} is now {{target}}.",
                                "correct": correct,
                                "foil": foil,
                            })
    # deterministic downsample only if requested by pilot
    if max_cases and len(frames) > max_cases:
        rng.shuffle(frames)
        frames = sorted(frames[:max_cases], key=lambda r: r["case_id"])
    return frames, {"target_check": target_check, "n_frames": len(frames), "families": [f["family"] for f in usable_fams]}


@torch.inference_mode()
def collect_rows(model, tok, device, frames: list[dict[str, Any]], batch_size: int):
    n_layers = int(model.config.num_hidden_layers) + 1
    examples = []
    for i, r in enumerate(frames):
        sent, span = render(r["context"], r["query_template"])
        for key, word in [("correct", r["correct"]), ("foil", r["foil"]), ("state0", next(f["state0"] for f in FAMILIES if f["family"] == r["family"])), ("state1", next(f["state1"] for f in FAMILIES if f["family"] == r["family"]))]:
            prep = mask_target(tok, sent, span, word)
            if prep is None:
                continue
            ids, att, pos, tid = prep
            examples.append({"frame_i": i, "score_key": key, "ids": ids, "att": att, "pos": pos, "tid": tid, "length": len(ids)})
    examples.sort(key=lambda x: x["length"])
    pad = tok.pad_token_id if tok.pad_token_id is not None else 0
    # Per frame representation is identical across target alternatives because the target is masked.
    reps: dict[int, np.ndarray] = {}
    logits_by: dict[tuple[int, str], float] = {}
    for st in range(0, len(examples), batch_size):
        batch = examples[st:st+batch_size]
        ml = max(x["length"] for x in batch)
        ids_t = torch.tensor([x["ids"] + [pad]*(ml-x["length"]) for x in batch], dtype=torch.long, device=device)
        att_t = torch.tensor([x["att"] + [0]*(ml-x["length"]) for x in batch], dtype=torch.long, device=device)
        pos_t = torch.tensor([x["pos"] for x in batch], dtype=torch.long, device=device)
        tid_t = torch.tensor([x["tid"] for x in batch], dtype=torch.long, device=device)
        out = model(input_ids=ids_t, attention_mask=att_t, output_hidden_states=True, return_dict=True)
        mb = torch.arange(ids_t.shape[0], device=device)
        vals = torch.log_softmax(out.logits[mb, pos_t].float(), dim=-1).gather(-1, tid_t.unsqueeze(-1)).squeeze(-1)
        for bi, ex in enumerate(batch):
            logits_by[(ex["frame_i"], ex["score_key"])] = float(vals[bi].cpu())
            if ex["score_key"] == "correct" and ex["frame_i"] not in reps:
                reps[ex["frame_i"]] = np.stack([out.hidden_states[li][bi, ex["pos"], :].float().cpu().numpy() for li in range(n_layers)], axis=0)
    rows = []
    for i, r in enumerate(frames):
        if i not in reps:
            continue
        c = logits_by.get((i, "correct"), float("nan")); f = logits_by.get((i, "foil"), float("nan"))
        s0 = logits_by.get((i, "state0"), float("nan")); s1 = logits_by.get((i, "state1"), float("nan"))
        row = dict(r)
        row["hidden"] = reps[i]
        row["margin_correct_foil"] = c - f
        row["score_correct"] = c
        row["score_foil"] = f
        row["state0_minus_state1"] = s0 - s1
        row["head_correct"] = bool(c > f)
        rows.append(row)
    return rows, n_layers


def matrix(rows: list[dict[str, Any]], layer: int, label_key: str, keep_fn=None):
    sub = [r for r in rows if keep_fn is None or keep_fn(r)]
    X = np.stack([r["hidden"][layer] for r in sub], axis=0)
    y = np.array([int(r[label_key]) for r in sub], dtype=np.int64)
    return X, y, sub


def train_lr_predict(Xtr, ytr, Xte, yte, seed=0, permute=False):
    yfit = ytr.copy()
    if permute:
        rng = np.random.RandomState(seed); rng.shuffle(yfit)
    if len(np.unique(yfit)) < 2:
        return float("nan"), None, None, None
    mu = Xtr.mean(0); sd = Xtr.std(0) + 1e-6
    clf = LogisticRegression(max_iter=2000, C=0.5, random_state=seed)
    clf.fit((Xtr-mu)/sd, yfit)
    pred = clf.predict((Xte-mu)/sd)
    return float((pred == yte).mean()), clf, mu, sd


def split_masks(rows: list[dict[str, Any]]):
    return {
        "heldout_family": lambda r: r["train_group"] == "heldout_family",
        "heldout_actor": lambda r: r["actor_group"] == "heldout_actor" and r["train_group"] == "train",
        "heldout_event0": lambda r: r["event_pol"] == 0 and r["train_group"] == "train",
        "heldout_event1": lambda r: r["event_pol"] == 1 and r["train_group"] == "train",
    }


def decoder_panel(rows: list[dict[str, Any]], n_layers: int) -> dict[str, Any]:
    panel = {"affected_query": {}, "final_pol": {}}
    masks = split_masks(rows)
    for label_key in ["affected_query", "final_pol"]:
        for split_name, test_fn in masks.items():
            layer_rows = []
            for layer in range(n_layers):
                Xte, yte, _ = matrix(rows, layer, label_key, test_fn)
                Xtr, ytr, _ = matrix(rows, layer, label_key, lambda r, tf=test_fn: not tf(r))
                acc, _, _, _ = train_lr_predict(Xtr, ytr, Xte, yte, seed=196 + layer)
                pacc, _, _, _ = train_lr_predict(Xtr, ytr, Xte, yte, seed=9196 + layer, permute=True)
                layer_rows.append({"layer": layer, "acc": acc, "perm_null": pacc})
            best = max(layer_rows, key=lambda r: -1 if not math.isfinite(r["acc"]) else r["acc"])
            panel[label_key][split_name] = {"layers": layer_rows, "best_layer": best["layer"], "best_acc": best["acc"]}
    return panel


def component_from_training(rows: list[dict[str, Any]], layer: int, label_key: str, exclude_heldout_family: bool = True):
    train = [r for r in rows if (r["train_group"] == "train" if exclude_heldout_family else True)]
    X = np.stack([r["hidden"][layer] for r in train], axis=0)
    y = np.array([int(r[label_key]) for r in train])
    mu = X.mean(0)
    m1 = X[y == 1].mean(0)
    m0 = X[y == 0].mean(0)
    d = m1 - m0
    norm = float(np.linalg.norm(d) + 1e-12)
    w = d / norm
    proj = (X - mu) @ w
    amp = 0.5 * abs(float(np.median(proj[y == 1]) - np.median(proj[y == 0])))
    if not math.isfinite(amp) or amp <= 1e-6:
        amp = float(np.std(proj) + 1e-6)
    rng = np.random.RandomState(19600 + layer)
    rand = rng.normal(size=w.shape); rand = rand / (np.linalg.norm(rand) + 1e-12)
    return {"layer": layer, "label_key": label_key, "w": w.astype("float32"), "mu": mu.astype("float32"), "amp": amp, "random_w": rand.astype("float32"),
            "train_n": len(train), "train_proj_stats": {"label1": qstats(proj[y == 1]), "label0": qstats(proj[y == 0])}}


def make_batch_examples(tok, frames: list[dict[str, Any]]):
    exs = []
    for i, r in enumerate(frames):
        sent, span = render(r["context"], r["query_template"])
        for key, word in [("correct", r["correct"]), ("foil", r["foil"]), ("state0", next(f["state0"] for f in FAMILIES if f["family"] == r["family"])), ("state1", next(f["state1"] for f in FAMILIES if f["family"] == r["family"]))]:
            prep = mask_target(tok, sent, span, word)
            if prep is None:
                continue
            ids, att, pos, tid = prep
            exs.append({"frame_i": i, "score_key": key, "ids": ids, "att": att, "pos": pos, "tid": tid, "length": len(ids)})
    exs.sort(key=lambda x: x["length"])
    return exs


@torch.inference_mode()
def score_with_hook(model, tok, device, frames: list[dict[str, Any]], comp: dict[str, Any], mode: str, alpha: float, batch_size: int, eval_filter=None):
    eval_frames = [r for r in frames if eval_filter is None or eval_filter(r)]
    examples = make_batch_examples(tok, eval_frames)
    pad = tok.pad_token_id if tok.pad_token_id is not None else 0
    layer_h = int(comp["layer"])
    module_idx = layer_h - 1
    if module_idx < 0:
        raise ValueError("hook patch only supports hidden layer >=1")
    w = torch.tensor(comp["w"], dtype=torch.float32, device=device)
    rw = torch.tensor(comp["random_w"], dtype=torch.float32, device=device)
    mu = torch.tensor(comp["mu"], dtype=torch.float32, device=device)
    amp = float(comp["amp"]) * float(alpha)
    active: dict[str, torch.Tensor] = {}

    def hook(_module, _inp, out):
        h = out[0] if isinstance(out, tuple) else out
        pos = active["pos"]
        sign = active["sign"]
        mb = torch.arange(h.shape[0], device=h.device)
        hp = h[mb, pos, :].float()
        hh = h.clone()
        if mode == "none":
            return out
        elif mode == "remove":
            ww = w
            coeff = ((hp - mu) @ ww).unsqueeze(-1)
            newp = hp - coeff * ww
        elif mode == "add":
            newp = hp + (amp * sign).unsqueeze(-1) * w
        elif mode == "add_swapped":
            newp = hp - (amp * sign).unsqueeze(-1) * w
        elif mode == "add_random":
            newp = hp + (amp * sign).unsqueeze(-1) * rw
        elif mode == "remove_random":
            ww = rw
            coeff = ((hp - mu) @ ww).unsqueeze(-1)
            newp = hp - coeff * ww
        else:
            raise ValueError(mode)
        hh[mb, pos, :] = newp.to(h.dtype)
        if isinstance(out, tuple):
            return (hh,) + tuple(out[1:])
        return hh

    handle = model.deberta.encoder.layer[module_idx].register_forward_hook(hook)
    vals: dict[tuple[int, str], float] = {}
    try:
        for st in range(0, len(examples), batch_size):
            batch = examples[st:st+batch_size]
            ml = max(x["length"] for x in batch)
            ids_t = torch.tensor([x["ids"] + [pad]*(ml-x["length"]) for x in batch], dtype=torch.long, device=device)
            att_t = torch.tensor([x["att"] + [0]*(ml-x["length"]) for x in batch], dtype=torch.long, device=device)
            pos_t = torch.tensor([x["pos"] for x in batch], dtype=torch.long, device=device)
            tid_t = torch.tensor([x["tid"] for x in batch], dtype=torch.long, device=device)
            signs = []
            for ex in batch:
                r = eval_frames[ex["frame_i"]]
                # For affected-query component: + for affected query, - for unaffected.
                # This sign is fixed by the factorial object, not by model output.
                s = 1.0 if int(r["affected_query"]) == 1 else -1.0
                signs.append(s)
            active["pos"] = pos_t
            active["sign"] = torch.tensor(signs, dtype=torch.float32, device=device)
            out = model(input_ids=ids_t, attention_mask=att_t, return_dict=True)
            mb = torch.arange(ids_t.shape[0], device=device)
            lp = torch.log_softmax(out.logits[mb, pos_t].float(), dim=-1).gather(-1, tid_t.unsqueeze(-1)).squeeze(-1)
            for ex, val in zip(batch, lp.detach().cpu().tolist()):
                vals[(ex["frame_i"], ex["score_key"])] = float(val)
    finally:
        handle.remove()
    rows = []
    for i, r in enumerate(eval_frames):
        c = vals.get((i, "correct"), float("nan")); f = vals.get((i, "foil"), float("nan"))
        s0 = vals.get((i, "state0"), float("nan")); s1 = vals.get((i, "state1"), float("nan"))
        rows.append({**{k: r[k] for k in ["case_id", "family", "train_group", "actor_group", "event_pol", "affected_query", "final_pol", "affected_slot", "query_slot"]},
                     "mode": mode, "alpha": alpha, "layer": layer_h, "margin_correct_foil": c-f, "head_correct": bool(c > f), "state0_minus_state1": s0-s1})
    return rows


def summarize_behavior(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"n": 0}
    out = {"n": len(rows), "accuracy": sum(1 for r in rows if r["head_correct"]) / len(rows), "margin": qstats(r["margin_correct_foil"] for r in rows)}
    for field in ["family", "train_group", "actor_group", "affected_query", "event_pol"]:
        groups = defaultdict(list)
        for r in rows:
            groups[str(r[field])].append(r)
        out[f"by_{field}"] = {k: {"n": len(v), "accuracy": sum(1 for x in v if x["head_correct"])/len(v), "margin_mean": statistics.fmean(x["margin_correct_foil"] for x in v)} for k,v in sorted(groups.items())}
    return out


def write_csv(path: Path, rows: list[dict[str, Any]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8"); return
    keys=[]; seen=set()
    for r in rows:
        for k in r:
            if k not in seen:
                seen.add(k); keys.append(k)
    with path.open("w", encoding="utf-8", newline="") as f:
        w=csv.DictWriter(f, fieldnames=keys, extrasaction="ignore"); w.writeheader(); w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(MODELS), default="chck82_scale1p75")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--max-cases", type=int, default=0)
    ap.add_argument("--alpha", type=float, default=1.0)
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    out_dir = OUT_ROOT / args.model
    out_dir.mkdir(parents=True, exist_ok=True)
    print(json.dumps({"event": "load_model", "model": args.model, "path": rel(MODELS[args.model]), "device": str(device)}), flush=True)
    tok = AutoTokenizer.from_pretrained(str(MODELS[args.model]), trust_remote_code=True, use_fast=True)
    model = AutoModelForMaskedLM.from_pretrained(str(MODELS[args.model]), trust_remote_code=True).to(device).eval()

    frames, manifest = build_frames(tok, args.max_cases)
    frames_path = out_dir / "factorial_entity_binding_frames.jsonl"
    frames_path.write_text("".join(json.dumps(r, sort_keys=True, ensure_ascii=False) + "\n" for r in frames), encoding="utf-8")
    print(json.dumps({"event": "frames", **manifest, "frames_path": rel(frames_path)}), flush=True)

    rows, n_layers = collect_rows(model, tok, device, frames, args.batch_size)
    # Strip hidden arrays before writing case rows.
    slim_rows = []
    for r in rows:
        rr = {k:v for k,v in r.items() if k != "hidden"}
        slim_rows.append(rr)
    write_csv(out_dir / "baseline_factorial_rows.csv", slim_rows)
    baseline_summary = summarize_behavior(slim_rows)
    panel = decoder_panel(rows, n_layers)

    interventions = []
    intervention_summaries = {}
    # Pre-fixed layers from research lead: high layers before reversal.
    for layer in [4, 5, 6]:
        if layer >= n_layers:
            continue
        comp = component_from_training(rows, layer, "affected_query", exclude_heldout_family=True)
        csummary = {k:v for k,v in comp.items() if k not in {"w", "mu", "random_w"}}
        for mode in ["none", "remove", "remove_random", "add", "add_random", "add_swapped"]:
            # Evaluate only held-out families to force action/state-family transfer.
            irows = score_with_hook(model, tok, device, frames, comp, mode, args.alpha, args.batch_size, eval_filter=lambda r: r["train_group"] == "heldout_family")
            interventions.extend(irows)
            key = f"L{layer}_{mode}"
            intervention_summaries[key] = {"component": csummary, "heldout_family": summarize_behavior(irows)}
            print(json.dumps({"event": "intervention", "model": args.model, "layer": layer, "mode": mode, "acc": intervention_summaries[key]["heldout_family"].get("accuracy"), "margin_mean": intervention_summaries[key]["heldout_family"].get("margin",{}).get("mean")}), flush=True)
    write_csv(out_dir / "intervention_heldout_family_rows.csv", interventions)

    # Compact layer curves for stdout.
    for label_key, splits in panel.items():
        for split, info in splits.items():
            print(json.dumps({"event": "decoder", "model": args.model, "label": label_key, "split": split,
                              "best_layer": info["best_layer"], "best_acc": round(info["best_acc"], 4),
                              "curve": [round(x["acc"], 3) for x in info["layers"]]}), flush=True)

    summary = {
        "status": "FACTORIAL_ENTITY_BINDING_PROBE",
        "created_utc": now(),
        "model": args.model,
        "model_path": rel(MODELS[args.model]),
        "device": str(device),
        "n_layers": n_layers,
        "manifest": manifest,
        "baseline_summary": baseline_summary,
        "decoder_panel": panel,
        "intervention_summaries": intervention_summaries,
        "files": {
            "frames": rel(frames_path),
            "baseline_rows": rel(out_dir / "baseline_factorial_rows.csv"),
            "intervention_rows": rel(out_dir / "intervention_heldout_family_rows.csv"),
        },
        "interpretation_guard": "No weights changed; official labels not used; held-out family interventions test whether a pre-fixed affected-query component from train families causally moves final-state predictions.",
    }
    out_json = out_dir / "factorial_entity_binding_probe.json"
    out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": summary["status"], "out_json": rel(out_json)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
