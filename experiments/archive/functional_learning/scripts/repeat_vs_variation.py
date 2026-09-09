#!/usr/bin/env python3
"""research functional_learning: Repetition vs. Variation experiment.

Scientific purpose
------------------
Test whether varied renderings of the same relational structure produce better
h1/h3 transport than exact repetition, and whether the benefit comes from
breaking surface shortcuts (VARY_CONTEXT) or diversity per se (VARY_NOISE).

Three conditions (identical comparison gradient budget per epoch):
  REPEAT:       Same 192 comparison events every epoch.
  VARY_CONTEXT: Fresh comparison events each epoch — new name pairs, objects,
                and voices, but same verbs, relation structure, and labels.
                Forces the learner to abstract the nonce verb meaning across
                contexts rather than memorize specific (name, verb, object) tuples.
  VARY_NOISE:   Same 192 events, but each epoch a different filler token is
                prepended/appended to each event string.  Surface diversity is
                introduced without changing the diagnostic token pattern.

State training (h0/h2 anchors + s_give/s_receive) is identical across conditions.
Evaluation uses the existing held-out names (Tara, Ben, Luca, Vera …) which
never appear in any training condition.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

# ── Import inherited harness ──
A01_SCRIPT_DIR = Path("experiments/archive/representation_and_objectives/training/scripts")
sys.path.insert(0, str(A01_SCRIPT_DIR))

import raw_span_discovery_probe as base   # noqa: E402
import posalign_gauge_probe     as p298   # noqa: E402
import dualpos_learned_equality_gauge_probe as s299  # noqa: E402

DEFAULT_OUT = Path("experiments/archive/functional_learning/data/repeat_vs_variation")

# ── Name and object pools ──
# Training names (used in REPEAT and state training — original harness names)
TRAIN_NAMES = [
    "Mira", "Omar", "Jonas", "Nia", "Rina", "Tomas", "Noel", "Iris",
    "Milo", "Keira", "Theo", "Felix", "Sara", "Lena", "Pavel", "Ava",
]
# Extra names for VARY_CONTEXT (disjoint from train AND eval names)
EXTRA_NAMES = [
    "Dani", "Ravi", "Yuki", "Luis", "Faye", "Owen", "Neve", "Cruz",
    "Rhea", "Nash", "Tala", "Kyle", "Isla", "Erik", "Lola", "Seth",
    "Cleo", "Jude", "Rosa", "Axel", "Wren", "Dale", "Hope", "Bram",
    "Skye", "Dane", "Ruth", "Elio", "Gwen", "Remy", "Zane", "Mara",
]
# Eval names already in the harness (never used for training in any condition)
EVAL_NAMES = [
    "Arun", "Ben", "Caleb", "Eli", "Hugo", "June", "Leah", "Luca",
    "Mateo", "Maya", "Nora", "Simon", "Tara", "Vera", "Yara", "Zara",
]

TRAIN_OBJECTS = [
    "lantern", "basket", "blanket", "camera", "compass", "helmet",
    "jacket", "map", "medal", "notebook", "parcel", "sketchbook",
    "tablet", "teapot", "violin", "wallet",
]
EXTRA_OBJECTS = [
    "mirror", "feather", "puzzle", "brush", "globe", "trophy",
    "pendant", "canteen", "satchel", "wreath", "figurine", "gazette",
    "thimble", "prism", "fossil", "locket", "brooch", "monocle",
    "spindle", "chalice", "baton", "garland", "mosaic", "beacon",
]

# Filler tokens for VARY_NOISE (common adverbs / short phrases)
NOISE_PREFIXES = [
    "Notably,", "Reportedly,", "Indeed,", "Clearly,", "Apparently,",
    "Perhaps,", "Allegedly,", "Evidently,", "Ultimately,", "Certainly,",
    "Supposedly,", "Plainly,", "Essentially,", "Surprisingly,", "Admittedly,",
    "Interestingly,", "Predictably,", "Obviously,", "Coincidentally,",
    "Fortunately,", "Unexpectedly,", "Incidentally,", "Naturally,",
    "Strangely,", "Remarkably,", "Regrettably,", "Undoubtedly,",
    "Actually,", "Frankly,", "Curiously,",
]
NOISE_SUFFIXES = [
    "It seems.", "One notes.", "As expected.", "So it was.",
    "This happened.", "Time passed.", "Fair enough.", "Quite so.",
    "They say.", "In short.", "End note.", "For now.",
    "Moving on.", "To be sure.", "All told.", "In brief.",
]

VERB_MAP = {
    "h0_dax":  "daxed",
    "h1_mep":  "meped",
    "h2_norp": "norped",
    "h3_ziv":  "zived",
}

RELATION_PAIRS = [
    ("h0_dax", "h1_mep"),
    ("h0_dax", "h2_norp"),
    ("h1_mep", "h2_norp"),
    ("h2_norp", "h3_ziv"),
]


# ── Helpers ──

def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, sort_keys=True) + "\n")


def write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")


def module_hash(module: torch.nn.Module, n: int = 16) -> str:
    h = hashlib.sha256()
    with torch.no_grad():
        for name, p in module.named_parameters():
            h.update(name.encode("utf-8"))
            h.update(p.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()[:n]


# ── Comparison rendering generation ──

def make_event(name1: str, name2: str, obj: str, verb: str, voice: str) -> str:
    """Generate a single event string in active or passive voice."""
    if voice == "active":
        return f"During the {obj} episode, {name1} {verb} {name2}."
    else:
        return f"During the {obj} episode, {name2} was {verb} by {name1}."


def make_comparison(rel1: str, rel2: str, name1: str, name2: str, obj: str,
                    voice1: str, voice2: str, row_id: str,
                    prefix1: str = "", suffix1: str = "",
                    prefix2: str = "", suffix2: str = "") -> base.ComparisonExample:
    """Create a ComparisonExample from rendering parameters."""
    v1, v2 = VERB_MAP[rel1], VERB_MAP[rel2]
    e1 = make_event(name1, name2, obj, v1, voice1)
    e2 = make_event(name1, name2, obj, v2, voice2)
    if prefix1:
        e1 = prefix1 + " " + e1
    if suffix1:
        e1 = e1 + " " + suffix1
    if prefix2:
        e2 = prefix2 + " " + e2
    if suffix2:
        e2 = e2 + " " + suffix2
    return base.ComparisonExample(
        row_id=row_id, suite="synthetic_step005", split="train",
        arm="aligned_state_bridge",
        event1=e1, event2=e2, names=(name1, name2),
        label=True, inverted_target=True,
        relation1=rel1, relation2=rel2,
        metadata={"voice1": voice1, "voice2": voice2, "object": obj},
    )


def generate_pool(names: List[str], objects: List[str], n_per_rel: int,
                  seed: int) -> Dict[Tuple[str,str], List[dict]]:
    """Pre-generate a large pool of rendering configs per relation pair.
    
    Returns {(rel1, rel2): [{"name1", "name2", "obj", "voice1", "voice2"}, ...]}
    """
    rng = random.Random(seed)
    # Build all possible name pairs (ordered)
    pairs = [(a, b) for i, a in enumerate(names) for b in names if a != b and i < names.index(b)]
    if len(pairs) < 2:
        pairs = [(names[0], names[1])] if len(names) >= 2 else []
    voices = ["active", "passive"]

    pool = {}
    for rel1, rel2 in RELATION_PAIRS:
        configs = []
        for n1, n2 in pairs:
            for obj in objects:
                for v1 in voices:
                    for v2 in voices:
                        configs.append({"name1": n1, "name2": n2, "obj": obj,
                                        "voice1": v1, "voice2": v2})
        rng.shuffle(configs)
        pool[(rel1, rel2)] = configs
    return pool


def sample_epoch_comps(pool: Dict, n_per_rel: int, epoch: int,
                       seed: int) -> List[base.ComparisonExample]:
    """Sample n_per_rel comparison configs per relation pair for one epoch."""
    rng = random.Random(seed * 100003 + epoch * 7)
    out = []
    for (rel1, rel2), configs in pool.items():
        # Deterministically sample without replacement within this epoch
        indices = list(range(len(configs)))
        rng.shuffle(indices)
        selected = indices[:n_per_rel]
        for i, idx in enumerate(selected):
            cfg = configs[idx]
            cmp = make_comparison(
                rel1, rel2, cfg["name1"], cfg["name2"], cfg["obj"],
                cfg["voice1"], cfg["voice2"],
                row_id=f"vary_ctx_e{epoch}_{rel1}_{rel2}_{i}",
            )
            out.append(cmp)
    return out


def add_noise_to_comps(comps: List[base.ComparisonExample], epoch: int,
                       seed: int) -> List[base.ComparisonExample]:
    """Return copies of comparisons with epoch-specific noise tokens added."""
    rng = random.Random(seed * 100003 + epoch * 13 + 999983)
    out = []
    for c in comps:
        pfx = rng.choice(NOISE_PREFIXES)
        sfx = rng.choice(NOISE_SUFFIXES)
        e1 = pfx + " " + c.event1 + " " + sfx
        e2 = pfx + " " + c.event2 + " " + sfx
        c_new = base.ComparisonExample(
            row_id=f"noise_e{epoch}_{c.row_id}",
            suite=c.suite, split=c.split, arm=c.arm,
            event1=e1, event2=e2, names=c.names,
            label=c.label, inverted_target=c.inverted_target,
            relation1=c.relation1, relation2=c.relation2,
            metadata={**c.metadata, "noise_prefix": pfx, "noise_suffix": sfx},
        )
        out.append(c_new)
    return out


# ── Vocabulary collection ──

def collect_extended_vocab(vocab: base.RawVocab, ts, tc, es, ec,
                           extra_names: List[str], extra_objects: List[str]):
    """Ensure vocab covers all possible tokens across conditions."""
    base.collect_raw_vocab(vocab, ts, tc)
    base.collect_raw_vocab(vocab, es, ec)
    # Add extra name tokens
    for n in extra_names + EVAL_NAMES:
        tok, _ = base.raw_tokenize(f"During the obj episode, {n} daxed Other.")
        vocab.add(tok)
    # Add extra object tokens
    for obj in extra_objects + TRAIN_OBJECTS:
        tok, _ = base.raw_tokenize(f"During the {obj} episode, Name daxed Other.")
        vocab.add(tok)
    # Add noise tokens
    for pfx in NOISE_PREFIXES:
        tok, _ = base.raw_tokenize(pfx)
        vocab.add(tok)
    for sfx in NOISE_SUFFIXES:
        tok, _ = base.raw_tokenize(sfx)
        vocab.add(tok)


# ── Training loop (epoch-varying comparisons) ──

def train_with_condition(model, vocab, ts, tc_fixed, device, bridge_sign: int,
                         condition: str, epochs: int, lr: float, wd: float,
                         seed: int, cmp_w: float, state_w: float, static_w: float,
                         print_every: int, hard: bool,
                         vary_pool=None, n_per_rel: int = 48) -> dict:
    """Train with condition-specific comparison renderings each epoch."""
    model.to(device)
    all_params = []
    seen = set()
    for p in list(model.event_state.parameters()) + list(model.static.parameters()):
        if p.requires_grad:
            pid = id(p)
            if pid not in seen:
                seen.add(pid); all_params.append(p)
    if model.event_cmp is not model.event_state:
        for p in model.event_cmp.parameters():
            if p.requires_grad:
                pid = id(p)
                if pid not in seen:
                    seen.add(pid); all_params.append(p)
    if not all_params:
        return {"error": "no trainable params"}

    opt = torch.optim.AdamW([{"params": all_params}], lr=lr, weight_decay=wd)
    rng = random.Random(seed)
    history = []
    t0 = time.time()
    unique_events_seen = set()

    for ep in range(1, epochs + 1):
        model.train()

        # Get comparison set for this epoch
        if condition == "repeat":
            tc_epoch = list(tc_fixed)
        elif condition == "vary_context":
            tc_epoch = sample_epoch_comps(vary_pool, n_per_rel, ep, seed)
        elif condition == "vary_noise":
            tc_epoch = add_noise_to_comps(tc_fixed, ep, seed)
        else:
            raise ValueError(f"Unknown condition: {condition}")

        # Track unique events
        for c in tc_epoch:
            unique_events_seen.add((c.event1, c.event2))

        states = list(ts); comps = tc_epoch
        rng.shuffle(states); rng.shuffle(comps)
        opt.zero_grad(set_to_none=True)
        losses = []

        for q in states:
            if q.is_changed:
                sc, _ = p298.score_event_pa(model.event_state, vocab, q.event, q.names, device, hard)
            else:
                sc = torch.stack([
                    p298.score_static_pa(model.static, vocab, q.prefix,
                                         q.hypothesis_text_by_name[cn], q.names, cn, device, hard)
                    for cn in q.names
                ])

            target = base.get_target_idx(q)
            if bridge_sign == -1 and q.is_direct_anchor:
                sc_eff = sc.flip(0)
            else:
                sc_eff = sc
            loss = F.cross_entropy(sc_eff.view(1, -1),
                                   torch.tensor([target], dtype=torch.long, device=device))
            losses.append((state_w if q.is_changed else static_w) * loss)

        for c in comps:
            s1, _ = p298.score_event_pa(model.event_cmp, vocab, c.event1, c.names, device, hard)
            s2, _ = p298.score_event_pa(model.event_cmp, vocab, c.event2, c.names, device, hard)
            p1, p2 = F.softmax(s1, dim=0), F.softmax(s2, dim=0)
            psame = (p1 * p2).sum().clamp(1e-6, 1 - 1e-6)
            y = torch.tensor(float(c.label), dtype=torch.float32, device=psame.device)
            losses.append(cmp_w * (-(y * torch.log(psame) + (1 - y) * torch.log(1 - psame))))

        total = torch.stack(losses).mean()
        total.backward()
        torch.nn.utils.clip_grad_norm_(all_params, 5.0)
        opt.step()

        if ep == 1 or ep == epochs or (print_every and ep % print_every == 0):
            model.eval()
            with torch.no_grad():
                ns = cs = nch = cch = nun = cun = 0
                for q in ts:
                    if q.is_changed:
                        sc, _ = p298.score_event_pa(model.event_state, vocab,
                                                     q.event, q.names, device, hard)
                    else:
                        sc = torch.stack([
                            p298.score_static_pa(model.static, vocab, q.prefix,
                                                 q.hypothesis_text_by_name[cn],
                                                 q.names, cn, device, hard)
                            for cn in q.names
                        ])
                    if bridge_sign == -1 and q.is_direct_anchor:
                        sc_eff = sc.flip(0)
                    else:
                        sc_eff = sc
                    t_idx = base.get_target_idx(q)
                    ok = int(sc_eff.argmax().item() == t_idx)
                    ns += 1; cs += ok
                    if q.is_changed: nch += 1; cch += ok
                    else: nun += 1; cun += ok

                nc_cmp = cc_cmp = 0
                for c in tc_fixed:  # Always eval on FIXED comps for comparability
                    s1, _ = p298.score_event_pa(model.event_cmp, vocab, c.event1, c.names, device, hard)
                    s2, _ = p298.score_event_pa(model.event_cmp, vocab, c.event2, c.names, device, hard)
                    ps = (F.softmax(s1, dim=0) * F.softmax(s2, dim=0)).sum()
                    cc_cmp += int(bool(ps >= 0.5) == bool(c.label)); nc_cmp += 1

            rec = {
                "epoch": ep,
                "loss": float(total.detach().cpu()),
                "train_state": cs / ns if ns else math.nan,
                "train_changed": cch / nch if nch else math.nan,
                "train_unchanged": cun / nun if nun else math.nan,
                "train_cmp_fixed": cc_cmp / nc_cmp if nc_cmp else math.nan,
            }
            history.append(rec)
            if print_every:
                print(json.dumps(rec, sort_keys=True), flush=True)

    return {
        "elapsed_seconds": time.time() - t0,
        "unique_comparison_event_pairs_seen": len(unique_events_seen),
        "history": history,
    }


# ── Full evaluation ──

def eval_per_relation(model, vocab, es, device, bridge_sign: int, hard: bool) -> dict:
    """Per-relation accuracy on eval state queries."""
    by_rel = defaultdict(lambda: {"n": 0, "correct": 0, "margins": []})
    model.eval()
    with torch.no_grad():
        for q in es:
            if not q.is_changed:
                continue
            sc, _ = p298.score_event_pa(model.event_state, vocab, q.event, q.names, device, hard)
            target = base.get_target_idx(q)
            if bridge_sign == -1 and q.is_direct_anchor:
                sc_eff = sc.flip(0)
            else:
                sc_eff = sc
            ok = int(sc_eff.argmax().item() == target)
            margin = float((sc_eff[target] - sc_eff[1 - target]).cpu())
            r = q.relation or "unknown"
            by_rel[r]["n"] += 1
            by_rel[r]["correct"] += ok
            by_rel[r]["margins"].append(margin)
    out = {}
    for r, d in sorted(by_rel.items()):
        out[r] = {
            "n": d["n"],
            "acc": d["correct"] / d["n"] if d["n"] else math.nan,
            "mean_signed_margin": float(np.mean(d["margins"])) if d["margins"] else math.nan,
        }
    return out


# ── Main run ──

def run_one(args) -> dict:
    s299.patch_dual_matcher()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device("cpu")  # This tiny harness is faster on CPU

    # Load base data
    ts, tc, es, ec, _pe, counts = base.load_dataset(args.data_root, args.arm)
    print(f"Loaded: {len(ts)} train state, {len(tc)} train cmp, "
          f"{len(es)} eval state, {len(ec)} eval cmp", flush=True)

    # Build extended vocabulary
    vocab = base.RawVocab()
    collect_extended_vocab(vocab, ts, tc, es, ec, EXTRA_NAMES, EXTRA_OBJECTS)
    print(f"Vocabulary: {len(vocab.itos)} tokens", flush=True)

    # Pre-generate VARY_CONTEXT pool
    vary_pool = generate_pool(EXTRA_NAMES, EXTRA_OBJECTS, args.n_per_rel, args.seed)
    pool_sizes = {f"{r1}->{r2}": len(v) for (r1, r2), v in vary_pool.items()}
    print(f"VARY_CONTEXT pool sizes: {pool_sizes}", flush=True)

    # Create model
    if args.char_pair_mode == "none":
        char_ids = []
    elif args.char_pair_mode == "train_alphabet":
        char_ids = s299.char_ids_from_names(s299.names_from_training(ts, tc))
    else:
        char_ids = list(range(1, 27))

    paired = p298.create_paired_posalign(len(vocab.itos), [args.condition], args.seed,
                                         args.emb_dim, args.hidden, args.char_dim)
    model = copy.deepcopy(paired[args.condition]).to(device)

    init_hash = module_hash(model, 16)
    print(json.dumps({
        "run": "repeat_vs_variation",
        "condition_type": args.condition_type,
        "model_condition": args.condition,
        "bridge_sign": args.bridge_sign,
        "seed": args.seed,
        "epochs": args.epochs,
        "init_hash": init_hash,
        "n_per_rel": args.n_per_rel,
    }, indent=2), flush=True)

    # Equality pretraining (identical across conditions)
    if args.char_pair_mode != "none" and args.char_pair_epochs > 0:
        pre_hist = s299.pretrain_char_pairs(model, char_ids, device,
                                            args.char_pair_epochs, args.char_pair_lr,
                                            args.print_every)
    else:
        pre_hist = []

    pretrain_hash = module_hash(model, 16)
    p298.freeze_matcher(model)

    # Relational training (condition-specific)
    train_info = train_with_condition(
        model, vocab, ts, tc, device, args.bridge_sign,
        condition=args.condition_type,
        epochs=args.epochs, lr=args.lr, wd=args.wd, seed=args.seed,
        cmp_w=args.cmp_weight, state_w=args.state_weight,
        static_w=args.static_weight, print_every=args.print_every,
        hard=True, vary_pool=vary_pool, n_per_rel=args.n_per_rel,
    )

    final_hash = module_hash(model, 16)

    # Evaluate
    eval_per_rel = eval_per_relation(model, vocab, es, device, args.bridge_sign, hard=True)

    # Also use research-style full eval for comparability
    eval_sr, eval_cr = p298.eval_predictions(model, vocab, es, ec, device,
                                              args.condition, args.arm, args.seed,
                                              args.bridge_sign, hard=True)
    choices = base.state_choice_records(eval_sr)
    boths = base.pair_both_records(choices)
    central = base.compute_central(choices, boths, eval_cr, args.condition, args.arm,
                                   args.seed, args.bridge_sign)

    # Also eval train predictions on FIXED comparisons
    train_sr, train_cr = p298.eval_predictions(model, vocab, ts, tc, device,
                                                args.condition, args.arm, args.seed,
                                                args.bridge_sign, hard=True)

    # Manifest
    manifest = {
        "condition_type": args.condition_type,
        "bridge_sign": args.bridge_sign,
        "seed": args.seed,
        "epochs": args.epochs,
        "n_per_rel": args.n_per_rel,
        "train_state_count": len(ts),
        "train_cmp_fixed_count": len(tc),
        "eval_state_count": len(es),
        "eval_cmp_count": len(ec),
        "unique_comparison_pairs_seen": train_info.get("unique_comparison_event_pairs_seen", 0),
        "vocab_size": len(vocab.itos),
        "vary_pool_sizes": pool_sizes,
    }

    # Save results
    sign = "+1" if args.bridge_sign == 1 else "-1"
    tag = f"{args.condition_type}_bs{sign}_e{args.epochs}_seed{args.seed}"
    rd = args.out / tag
    rd.mkdir(parents=True, exist_ok=True)

    last_hist = train_info.get("history", [{}])[-1] if train_info.get("history") else {}

    result = {
        "condition_type": args.condition_type,
        "model_condition": args.condition,
        "arm": args.arm,
        "seed": args.seed,
        "bridge_sign": args.bridge_sign,
        "epochs": args.epochs,
        "hashes": {"init": init_hash, "pretrain": pretrain_hash, "final": final_hash},
        "train_info": train_info,
        "eval_per_relation": eval_per_rel,
        "central_eval": central,
        "manifest": manifest,
        "train_last": last_hist,
    }
    write_json(rd / "result.json", result)
    write_json(rd / "manifest.json", manifest)
    write_jsonl(rd / "eval_state_predictions.jsonl", eval_sr)
    write_jsonl(rd / "train_state_predictions.jsonl", train_sr)

    # Summary
    lines = [
        f"# research: {args.condition_type} bs{sign} e{args.epochs}",
        "",
        f"- model: {args.condition}",
        f"- seed: {args.seed}",
        f"- epochs: {args.epochs}",
        f"- comparison events per epoch: {len(tc)}",
        f"- unique comparison pairs seen: {train_info.get('unique_comparison_event_pairs_seen', 0)}",
        f"- final model hash: {final_hash}",
        "",
        "## Per-relation eval accuracy (changed state, held-out names)",
        "",
        "| relation | n | accuracy | mean_margin |",
        "|---|---:|---:|---:|",
    ]
    for r, d in sorted(eval_per_rel.items()):
        lines.append(f"| {r} | {d['n']} | {d['acc']:.4f} | {d['mean_signed_margin']:.4f} |")
    lines.extend([
        "",
        "## Central eval",
        "",
        "```json",
        json.dumps(central, indent=2, sort_keys=True),
        "```",
        "",
        "## Train last epoch",
        "",
        "```json",
        json.dumps(last_hist, indent=2, sort_keys=True),
        "```",
    ])
    (rd / "summary.md").write_text("\n".join(lines) + "\n")

    print(json.dumps({
        "status": "COMPLETE",
        "tag": tag,
        "run_dir": str(rd),
        "eval_per_relation": eval_per_rel,
        "central_direct_same": central.get("direct_same"),
        "central_graph_same": central.get("graph_same"),
        "central_hh_closure": central.get("hh_closure"),
        "train_last": last_hist,
        "unique_pairs_seen": train_info.get("unique_comparison_event_pairs_seen", 0),
    }, indent=2, sort_keys=True), flush=True)

    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition-type", required=True,
                    choices=["repeat", "vary_context", "vary_noise"])
    ap.add_argument("--bridge-sign", type=int, required=True, choices=[1, -1])
    ap.add_argument("--seed", type=int, default=30000)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--n-per-rel", type=int, default=48,
                    help="Comparison events per relation pair per epoch")
    ap.add_argument("--data-root", type=Path, default=base.DEFAULT_DATA_ROOT)
    ap.add_argument("--arm", default="aligned_state_bridge")
    ap.add_argument("--condition", default="shared_trunk",
                    help="Model architecture condition")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--lr", type=float, default=5e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--cmp-weight", type=float, default=1.0)
    ap.add_argument("--state-weight", type=float, default=1.0)
    ap.add_argument("--static-weight", type=float, default=0.5)
    ap.add_argument("--emb-dim", type=int, default=48)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--char-dim", type=int, default=16)
    ap.add_argument("--char-pair-mode", default="full_alphabet",
                    choices=["none", "train_alphabet", "full_alphabet"])
    ap.add_argument("--char-pair-epochs", type=int, default=300)
    ap.add_argument("--char-pair-lr", type=float, default=1e-2)
    ap.add_argument("--print-every", type=int, default=10)
    args = ap.parse_args()
    run_one(args)


if __name__ == "__main__":
    main()
