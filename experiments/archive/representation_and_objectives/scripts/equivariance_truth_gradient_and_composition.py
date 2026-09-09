#!/usr/bin/env python3
"""research: repair the cross-predicate equivariance measurement.

This is a CPU-only execution diagnostic, not BabyLM training. It addresses the
research all-chance result by separating four questions that were confounded:

1. Are the generated labels a coherent symbolic truth table?
2. Is there a live gradient path, and can a tiny balanced slice be forced to
   memorize?
3. If every predicate receives independent semantic evidence, can a small
   learner compose context-predicate and hypothesis-predicate meanings on held
   predicate-pair combinations?
4. Under that non-zero-shot-predicate setting, does a representation matching
   (equivariance) loss improve ordinary supervised learning on new pair/world/
   source-domain compositions?

The benchmark is deliberately small and controlled. It uses the source-attested
paired-world family metadata from research only as the source of A/B role flips;
all text is canonicalized to ENTITY_A/ENTITY_B so the measurement is about the
role-operation and optimization path, not about name memorization.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import collections
import copy
import hashlib
import json
import math
import os
import random
import re
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

STUDY = Path("experiments/archive/representation_and_objectives")
WORKSPACE = STUDY
TRAIN = WORKSPACE / "data/paired_world_pilot/families_train.jsonl"
HELD = WORKSPACE / "data/paired_world_pilot/families_held.jsonl"
OUT_DIR = WORKSPACE / "data/equivariance_diagnostics"

# subject_role tells which world-role the grammatical subject of the predicate
# bears. wf = subject is winner/positive role; lf = subject is loser/negative role.
PREDS: dict[str, dict[str, str]] = {
    "defeated":      {"polarity": "wf", "ctx": "{S} defeated {O}", "hyp": "{S} defeated {O}"},
    "beat":          {"polarity": "wf", "ctx": "{S} beat {O}", "hyp": "{S} beat {O}"},
    "overcame":      {"polarity": "wf", "ctx": "{S} overcame {O}", "hyp": "{S} overcame {O}"},
    "prevailed":     {"polarity": "wf", "ctx": "{S} prevailed over {O}", "hyp": "{S} prevailed over {O}"},
    "lost_to":       {"polarity": "lf", "ctx": "{S} lost to {O}", "hyp": "{S} lost to {O}"},
    "was_beaten":    {"polarity": "lf", "ctx": "{S} was beaten by {O}", "hyp": "{S} was beaten by {O}"},
    "fell_to":       {"polarity": "lf", "ctx": "{S} fell to {O}", "hyp": "{S} fell to {O}"},
    "was_defeated":  {"polarity": "lf", "ctx": "{S} was defeated by {O}", "hyp": "{S} was defeated by {O}"},
}
PRED_LIST = list(PREDS.keys())
WF_PREDS = [p for p in PRED_LIST if PREDS[p]["polarity"] == "wf"]
LF_PREDS = [p for p in PRED_LIST if PREDS[p]["polarity"] == "lf"]


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def toks(s: str) -> list[str]:
    return re.findall(r"[A-Za-z_]+|\d+|[^\sA-Za-z_\d]", str(s).lower())


def stable_hash(s: str) -> int:
    return int(hashlib.sha1(s.encode("utf-8")).hexdigest()[:12], 16)


def winner_loser(ctx: dict[str, Any]) -> tuple[str, str]:
    wlab = ctx["winner_label"]
    if wlab == "A":
        return "ENTITY_A", "ENTITY_B"
    if wlab == "B":
        return "ENTITY_B", "ENTITY_A"
    raise ValueError(f"bad winner_label {wlab!r}")


def context_sentence(cp: str, ctx: dict[str, Any], fam: dict[str, Any], *, canonical: bool = True) -> str:
    winner, loser = winner_loser(ctx)
    if PREDS[cp]["polarity"] == "wf":
        S, O = winner, loser
    else:
        S, O = loser, winner
    body = PREDS[cp]["ctx"].format(S=S, O=O)
    # Keep a weak source-domain/world marker to preserve that examples originate
    # from separate worlds, but do not include scores or rankings.
    sport = re.sub(r"[^A-Za-z0-9_]+", "_", str(fam.get("sport", fam.get("source_type", "sport")))).strip("_") or "sport"
    world = "world_one" if ctx.get("winner_label") == fam.get("context1", {}).get("winner_label") else "world_two"
    return f"In DOMAIN_{sport} {world}, {body}."


def hypothesis_sentence(hp: str, orient: str) -> str:
    S, O = ("ENTITY_A", "ENTITY_B") if orient == "AB" else ("ENTITY_B", "ENTITY_A")
    return PREDS[hp]["hyp"].format(S=S, O=O) + "."


def role_hypothesis(entity: str, role: str) -> str:
    return f"{entity} was the {role}."


def canonical_role_context(ctx: dict[str, Any], fam: dict[str, Any]) -> str:
    winner, loser = winner_loser(ctx)
    sport = re.sub(r"[^A-Za-z0-9_]+", "_", str(fam.get("sport", fam.get("source_type", "sport")))).strip("_") or "sport"
    return f"In DOMAIN_{sport}, {winner} was the winner and {loser} was the loser."


def entity_role(entity: str, ctx: dict[str, Any]) -> str:
    winner, loser = winner_loser(ctx)
    if entity == winner:
        return "winner"
    if entity == loser:
        return "loser"
    raise ValueError(entity)


def hyp_label(hp: str, orient: str, ctx: dict[str, Any]) -> int:
    S = "ENTITY_A" if orient == "AB" else "ENTITY_B"
    role = entity_role(S, ctx)
    subj_role = "winner" if PREDS[hp]["polarity"] == "wf" else "loser"
    return int(role == subj_role)


def make_composition_row(fam: dict[str, Any], wk: str, cp: str, hp: str, orient: str, split: str, source: str) -> dict[str, Any]:
    ctx = fam[wk]
    ctxt = context_sentence(cp, ctx, fam)
    hyp = hypothesis_sentence(hp, orient)
    y = hyp_label(hp, orient, ctx)
    return {
        "id": f"{fam['family_id']}_{wk}_{cp}_{hp}_{orient}_{source}",
        "source": source,
        "family_id": fam["family_id"],
        "family_split": split,
        "sport": fam.get("sport", fam.get("source_type", "unknown")),
        "world": wk,
        "cp": cp,
        "hp": hp,
        "orient": orient,
        "text": ctxt + " [SEP] " + hyp,
        "context": ctxt,
        "hypothesis": hyp,
        "label": y,
        "winner_label": ctx["winner_label"],
        "cp_polarity": PREDS[cp]["polarity"],
        "hp_polarity": PREDS[hp]["polarity"],
    }


def make_support_rows(fams: list[dict[str, Any]], split: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for fam in fams:
        for wk in ["context1", "context2"]:
            ctx = fam[wk]
            # Context-predicate semantic evidence: every context predicate is
            # directly labeled against role words, independent of any held
            # predicate-pair composition.
            for cp in PRED_LIST:
                ctxt = context_sentence(cp, ctx, fam)
                for entity in ["ENTITY_A", "ENTITY_B"]:
                    for role in ["winner", "loser"]:
                        y = int(entity_role(entity, ctx) == role)
                        rows.append({
                            "id": f"{fam['family_id']}_{wk}_support_ctx_{cp}_{entity}_{role}",
                            "source": "semantic_support_context_predicate",
                            "family_id": fam["family_id"],
                            "family_split": split,
                            "sport": fam.get("sport", fam.get("source_type", "unknown")),
                            "world": wk,
                            "cp": cp,
                            "hp": f"role_{role}",
                            "orient": entity[-1],
                            "text": ctxt + " [SEP] " + role_hypothesis(entity, role),
                            "context": ctxt,
                            "hypothesis": role_hypothesis(entity, role),
                            "label": y,
                            "winner_label": ctx["winner_label"],
                            "cp_polarity": PREDS[cp]["polarity"],
                            "hp_polarity": role,
                        })
            # Hypothesis-predicate semantic evidence: every hypothesis predicate
            # is directly labeled against an explicit role-state context.
            role_ctxt = canonical_role_context(ctx, fam)
            for hp in PRED_LIST:
                for orient in ["AB", "BA"]:
                    hyp = hypothesis_sentence(hp, orient)
                    y = hyp_label(hp, orient, ctx)
                    rows.append({
                        "id": f"{fam['family_id']}_{wk}_support_hyp_{hp}_{orient}",
                        "source": "semantic_support_hypothesis_predicate",
                        "family_id": fam["family_id"],
                        "family_split": split,
                        "sport": fam.get("sport", fam.get("source_type", "unknown")),
                        "world": wk,
                        "cp": "explicit_role_state",
                        "hp": hp,
                        "orient": orient,
                        "text": role_ctxt + " [SEP] " + hyp,
                        "context": role_ctxt,
                        "hypothesis": hyp,
                        "label": y,
                        "winner_label": ctx["winner_label"],
                        "cp_polarity": "explicit",
                        "hp_polarity": PREDS[hp]["polarity"],
                    })
    return rows


def pred_pair_split() -> tuple[set[tuple[str, str]], set[tuple[str, str]]]:
    """Balanced held cp×hp combinations; every predicate appears in train and held.

    A 64-cell matrix is split by a fixed Latin-like rule. The held set contains
    16 cells. Each context predicate has 2 held and 6 train hypothesis partners;
    each hypothesis predicate has 2 held and 6 train context partners.
    """
    held: set[tuple[str, str]] = set()
    for i, cp in enumerate(PRED_LIST):
        for j, hp in enumerate(PRED_LIST):
            if (i + 3 * j) % 4 == 0:
                held.add((cp, hp))
    allp = {(cp, hp) for cp in PRED_LIST for hp in PRED_LIST}
    train = allp - held
    return train, held


def make_pair_rows(fams: list[dict[str, Any]], split: str, pairs: set[tuple[str, str]], source: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for fam in fams:
        for wk in ["context1", "context2"]:
            for cp, hp in sorted(pairs):
                for orient in ["AB", "BA"]:
                    rows.append(make_composition_row(fam, wk, cp, hp, orient, split, source))
    return rows


def audit_rows(rows: list[dict[str, Any]], name: str) -> dict[str, Any]:
    n = len(rows)
    labels = collections.Counter(int(r["label"]) for r in rows)
    by_source = collections.defaultdict(collections.Counter)
    by_pair = collections.defaultdict(collections.Counter)
    conflicts: dict[str, set[int]] = collections.defaultdict(set)
    for r in rows:
        y = int(r["label"])
        by_source[r.get("source", "?")][y] += 1
        by_pair[(r.get("cp"), r.get("hp"))][y] += 1
        conflicts[r["text"]].add(y)
    bad_texts = [(t, sorted(v)) for t, v in conflicts.items() if len(v) > 1]
    return {
        "name": name,
        "n": n,
        "label_counts": dict(labels),
        "label_fraction_true": labels.get(1, 0) / max(1, n),
        "source_label_counts": {k: dict(v) for k, v in sorted(by_source.items())},
        "n_pair_cells": len(by_pair),
        "pair_cell_unbalanced_examples": [
            {"pair": list(k), "counts": dict(v)}
            for k, v in list(by_pair.items())[:10]
            if len(v) != 2 or min(v.values()) == 0
        ],
        "duplicate_text_conflicts": len(bad_texts),
        "duplicate_text_conflict_examples": bad_texts[:5],
    }


class Vocab:
    def __init__(self) -> None:
        self.w2i = {"<pad>": 0, "<unk>": 1}

    def fit(self, rows: list[dict[str, Any]]) -> None:
        for r in rows:
            for t in toks(r["text"]):
                if t not in self.w2i:
                    self.w2i[t] = len(self.w2i)

    def encode(self, text: str, max_len: int) -> list[int]:
        ids = [self.w2i.get(t, 1) for t in toks(text)][:max_len]
        return ids + [0] * (max_len - len(ids))


def encode_rows(rows: list[dict[str, Any]], vocab: Vocab, max_len: int) -> tuple[torch.Tensor, torch.Tensor]:
    x = [vocab.encode(r["text"], max_len) for r in rows]
    y = [int(r["label"]) for r in rows]
    return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)


class TextBiGRU(nn.Module):
    def __init__(self, vocab_size: int, emb: int = 64, hid: int = 96, dropout: float = 0.05):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb, padding_idx=0)
        self.gru = nn.GRU(emb, hid, batch_first=True, bidirectional=True)
        self.head = nn.Sequential(nn.LayerNorm(2 * hid), nn.Dropout(dropout), nn.Linear(2 * hid, 64), nn.ReLU(), nn.Linear(64, 2))

    def encode_repr(self, x: torch.Tensor) -> torch.Tensor:
        e = self.emb(x)
        _, h = self.gru(e)
        return torch.cat([h[-2], h[-1]], dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encode_repr(x))


def acc_loss(model: nn.Module, x: torch.Tensor, y: torch.Tensor, batch: int = 512) -> tuple[float, float]:
    model.eval()
    ok = 0
    total_loss = 0.0
    n = 0
    with torch.no_grad():
        for s in range(0, len(x), batch):
            xb = x[s:s + batch]
            yb = y[s:s + batch]
            logits = model(xb)
            loss = F.cross_entropy(logits, yb, reduction="sum")
            total_loss += float(loss.item())
            ok += int((logits.argmax(1) == yb).sum().item())
            n += len(xb)
    return ok / max(1, n), total_loss / max(1, n)


def balanced_sample(rows: list[dict[str, Any]], n_per_label: int, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    groups = {0: [], 1: []}
    for r in rows:
        groups[int(r["label"])].append(r)
    out: list[dict[str, Any]] = []
    for y in [0, 1]:
        rng.shuffle(groups[y])
        out.extend(groups[y][:n_per_label])
    rng.shuffle(out)
    return out


def gradient_and_memorization_probe(rows: list[dict[str, Any]], out_dir: Path, seed: int = 25401) -> dict[str, Any]:
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    sl = balanced_sample(rows, n_per_label=32, seed=seed)
    vocab = Vocab(); vocab.fit(sl)
    max_len = max(len(toks(r["text"])) for r in sl)
    x, y = encode_rows(sl, vocab, max_len)
    model = TextBiGRU(len(vocab.w2i), emb=48, hid=64, dropout=0.0)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2, weight_decay=0.0)

    init_acc, init_loss = acc_loss(model, x, y)
    params0 = torch.cat([p.detach().flatten().cpu() for p in model.parameters()])

    # First-step gradient audit.
    model.train()
    logits = model(x)
    loss = F.cross_entropy(logits, y)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    grad_sq = 0.0
    nonzero_grads = 0
    for p in model.parameters():
        if p.grad is not None:
            grad_sq += float((p.grad.detach() ** 2).sum().item())
            nonzero_grads += int((p.grad.detach().abs() > 0).sum().item())
    grad_norm = math.sqrt(grad_sq)
    opt.step()
    params1 = torch.cat([p.detach().flatten().cpu() for p in model.parameters()])
    param_delta_first_step = float(torch.norm(params1 - params0).item())
    acc1, loss1 = acc_loss(model, x, y)

    hist = [{"epoch": 0, "acc": init_acc, "loss": init_loss}, {"epoch": 1, "acc": acc1, "loss": loss1}]
    # Continue until exact memorization or a hard cap. Full-batch is deliberate:
    # this checks the model/data/gradient path, not stochastic optimization.
    best_acc = acc1
    best_loss = loss1
    reached_epoch = None
    for ep in range(2, 501):
        model.train()
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        if ep <= 10 or ep % 10 == 0:
            a, l = acc_loss(model, x, y)
            hist.append({"epoch": ep, "acc": a, "loss": l})
            best_acc, best_loss = max(best_acc, a), min(best_loss, l)
            if a >= 0.999 and reached_epoch is None:
                reached_epoch = ep
                break
    final_acc, final_loss = acc_loss(model, x, y)
    if reached_epoch is None and final_acc >= 0.999:
        reached_epoch = hist[-1]["epoch"] if hist else None

    write_jsonl(out_dir / "memorized_slice_rows.jsonl", sl)
    return {
        "slice_rows": len(sl),
        "labels": dict(collections.Counter(int(r["label"]) for r in sl)),
        "vocab_size": len(vocab.w2i),
        "max_len": max_len,
        "initial_acc": init_acc,
        "initial_loss": init_loss,
        "first_step_grad_norm": grad_norm,
        "first_step_nonzero_grad_entries": nonzero_grads,
        "first_step_param_delta_l2": param_delta_first_step,
        "after_first_step_acc": acc1,
        "after_first_step_loss": loss1,
        "final_acc": final_acc,
        "final_loss": final_loss,
        "reached_999_acc_epoch": reached_epoch,
        "history": hist,
        "passed_gradient_path": bool(grad_norm > 0 and param_delta_first_step > 0 and nonzero_grads > 0),
        "passed_memorization": bool(final_acc >= 0.999),
    }


def build_equiv_pairs(rows: list[dict[str, Any]], max_pairs: int = 50000, seed: int = 25402) -> list[tuple[int, int]]:
    """Pair same world/query rows across different context predicates.

    Only composition rows are paired. Each pair holds family, world, hypothesis
    predicate, orientation and label fixed while varying the context predicate.
    """
    groups: dict[tuple[str, str, str, str, int], list[int]] = collections.defaultdict(list)
    for i, r in enumerate(rows):
        if r.get("source") != "composition_train_pair":
            continue
        groups[(r["family_id"], r["world"], r["hp"], r["orient"], int(r["label"]))].append(i)
    pairs: list[tuple[int, int]] = []
    for idxs in groups.values():
        for a in range(len(idxs)):
            for b in range(a + 1, len(idxs)):
                if rows[idxs[a]]["cp"] != rows[idxs[b]]["cp"]:
                    pairs.append((idxs[a], idxs[b]))
    rng = random.Random(seed)
    rng.shuffle(pairs)
    return pairs[:max_pairs]


def train_model(
    train_rows: list[dict[str, Any]],
    eval_sets: dict[str, list[dict[str, Any]]],
    *,
    seed: int,
    epochs: int,
    lr: float,
    lam: float,
    eq_pairs: list[tuple[int, int]] | None,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(min(8, os.cpu_count() or 1))

    vocab = Vocab(); vocab.fit(train_rows)
    max_len = min(96, max(len(toks(r["text"])) for r in train_rows + [r for rows in eval_sets.values() for r in rows]))
    train_x, train_y = encode_rows(train_rows, vocab, max_len)
    model = TextBiGRU(len(vocab.w2i), emb=64, hid=96, dropout=0.05)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    ds = TensorDataset(train_x, train_y)
    loader = DataLoader(ds, batch_size=128, shuffle=True, generator=torch.Generator().manual_seed(seed))
    eq_t = torch.tensor(eq_pairs, dtype=torch.long) if eq_pairs and lam > 0 else None
    hist: list[dict[str, float]] = []

    best_state = None
    best_score = -1.0
    for ep in range(1, epochs + 1):
        model.train()
        total = 0.0
        n = 0
        for xb, yb in loader:
            logits = model(xb)
            loss = F.cross_entropy(logits, yb)
            if eq_t is not None:
                # Sample representation matches from the global training tensor.
                pi = torch.randint(0, len(eq_t), (min(256, len(eq_t)),))
                p = eq_t[pi]
                ha = model.encode_repr(train_x[p[:, 0]])
                hb = model.encode_repr(train_x[p[:, 1]])
                # Normalize before MSE so the loss aligns directions rather than
                # collapsing norms.
                loss = loss + lam * F.mse_loss(F.normalize(ha, dim=-1), F.normalize(hb, dim=-1))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += float(loss.item()) * len(xb)
            n += len(xb)
        train_acc, train_loss = acc_loss(model, train_x, train_y)
        rec = {"epoch": ep, "train_acc": train_acc, "train_loss": train_loss, "opt_loss": total / max(1, n)}
        # Use held-pair on train source families only as a model-selection proxy;
        # it is not part of CE/equivariance training, and it tests held pair
        # composition without new worlds/domains.
        if "held_pair_seen_family" in eval_sets:
            ex, ey = encode_rows(eval_sets["held_pair_seen_family"], vocab, max_len)
            rec["held_pair_seen_family_acc"] = acc_loss(model, ex, ey)[0]
            score = rec["held_pair_seen_family_acc"]
        else:
            score = train_acc
        hist.append(rec)
        if score > best_score:
            best_score = score
            best_state = copy.deepcopy(model.state_dict())
    if best_state is not None:
        model.load_state_dict(best_state)

    eval_out: dict[str, Any] = {}
    for name, rows in eval_sets.items():
        x, y = encode_rows(rows, vocab, max_len)
        a, l = acc_loss(model, x, y)
        eval_out[name] = {"n": len(rows), "acc": a, "loss": l, "label_true_frac": sum(int(r["label"]) for r in rows) / max(1, len(rows))}
        # Per source-domain breakdown when useful.
        by_sport: dict[str, list[int]] = collections.defaultdict(list)
        model.eval()
        with torch.no_grad():
            logits = model(x)
            pred = logits.argmax(1).cpu().numpy().tolist()
        for i, r in enumerate(rows):
            by_sport[str(r.get("sport", "?"))].append(int(pred[i] == int(r["label"])))
        eval_out[name]["by_sport_acc"] = {k: float(np.mean(v)) for k, v in sorted(by_sport.items()) if v}

    return {
        "seed": seed,
        "epochs": epochs,
        "lr": lr,
        "lambda": lam,
        "vocab_size": len(vocab.w2i),
        "max_len": max_len,
        "best_selector_score": best_score,
        "history_tail": hist[-8:],
        "eval": eval_out,
    }


def choose_family_splits(all_train: list[dict[str, Any]], all_held: list[dict[str, Any]], train_n: int, held_n: int, domain_n: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], str, dict[str, int]]:
    # Prefer football as held source-domain if present, because it differs most
    # from tennis/BWF; otherwise use the smallest available domain with enough rows.
    all_fams = all_train + all_held
    counts = collections.Counter(str(f.get("sport", f.get("source_type", "unknown"))) for f in all_fams)
    held_domain = "football" if counts.get("football", 0) >= max(4, min(domain_n, 8)) else sorted(counts, key=lambda k: (counts[k], k))[0]
    train_pool = [f for f in all_train if str(f.get("sport", f.get("source_type", "unknown"))) != held_domain]
    held_pool = [f for f in all_held if str(f.get("sport", f.get("source_type", "unknown"))) != held_domain]
    held_domain_pool = [f for f in all_fams if str(f.get("sport", f.get("source_type", "unknown"))) == held_domain]
    rng = random.Random(25403)
    rng.shuffle(train_pool); rng.shuffle(held_pool); rng.shuffle(held_domain_pool)
    return train_pool[:train_n], held_pool[:held_n], held_domain_pool[:domain_n], held_domain, dict(counts)


def run_composition_experiment(args: argparse.Namespace, out_dir: Path) -> dict[str, Any]:
    all_train = read_jsonl(TRAIN)
    all_held = read_jsonl(HELD)
    train_fams, held_fams, held_domain_fams, held_domain, domain_counts = choose_family_splits(
        all_train, all_held, args.train_fams, args.held_fams, args.held_domain_fams
    )
    train_pairs, held_pairs = pred_pair_split()
    support_rows = make_support_rows(train_fams, "train")
    comp_train_rows = make_pair_rows(train_fams, "train", train_pairs, "composition_train_pair")
    train_rows = support_rows + comp_train_rows
    eval_sets = {
        "train_pair_held_family": make_pair_rows(held_fams, "held", train_pairs, "eval_train_pair_held_family"),
        "held_pair_seen_family": make_pair_rows(train_fams, "train", held_pairs, "eval_held_pair_seen_family"),
        "held_pair_held_family": make_pair_rows(held_fams, "held", held_pairs, "eval_held_pair_held_family"),
        "held_pair_held_domain": make_pair_rows(held_domain_fams, "held_domain", held_pairs, "eval_held_pair_held_domain"),
        "semantic_support_held_family": make_support_rows(held_fams, "held"),
    }
    eq_pairs = build_equiv_pairs(train_rows, max_pairs=args.max_eq_pairs)

    # Persist compact row sets needed to reproduce/inspect the benchmark.
    write_jsonl(out_dir / "semantic_support_train.jsonl", support_rows)
    write_jsonl(out_dir / "composition_train_pairs.jsonl", comp_train_rows)
    write_jsonl(out_dir / "held_pair_held_family_eval.jsonl", eval_sets["held_pair_held_family"])
    write_jsonl(out_dir / "held_pair_held_domain_eval.jsonl", eval_sets["held_pair_held_domain"])

    audits = {
        "support_train": audit_rows(support_rows, "semantic_support_train"),
        "composition_train": audit_rows(comp_train_rows, "composition_train_pair"),
        "held_pair_seen_family": audit_rows(eval_sets["held_pair_seen_family"], "held_pair_seen_family"),
        "held_pair_held_family": audit_rows(eval_sets["held_pair_held_family"], "held_pair_held_family"),
        "held_pair_held_domain": audit_rows(eval_sets["held_pair_held_domain"], "held_pair_held_domain"),
    }

    results: dict[str, list[dict[str, Any]]] = {"standard": [], "equivariant": []}
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    for seed in seeds:
        results["standard"].append(train_model(train_rows, eval_sets, seed=seed, epochs=args.epochs, lr=args.lr, lam=0.0, eq_pairs=None))
        results["equivariant"].append(train_model(train_rows, eval_sets, seed=seed, epochs=args.epochs, lr=args.lr, lam=args.lam, eq_pairs=eq_pairs))

    def mean_std(mode: str, eval_name: str, field: str = "acc") -> dict[str, Any]:
        vals = [r["eval"][eval_name][field] for r in results[mode]]
        return {"mean": float(np.mean(vals)), "std": float(np.std(vals)), "seeds": vals}

    aggregates: dict[str, Any] = {}
    for eval_name in eval_sets:
        aggregates[eval_name] = {
            "standard": mean_std("standard", eval_name),
            "equivariant": mean_std("equivariant", eval_name),
        }
        aggregates[eval_name]["delta_equiv_minus_standard"] = aggregates[eval_name]["equivariant"]["mean"] - aggregates[eval_name]["standard"]["mean"]

    summary = {
        "family_domain_counts_all": domain_counts,
        "held_domain": held_domain,
        "family_counts": {"train": len(train_fams), "held_family": len(held_fams), "held_domain": len(held_domain_fams)},
        "predicate_sets": {"all": PRED_LIST, "wf": WF_PREDS, "lf": LF_PREDS},
        "pair_split": {
            "train_pairs": len(train_pairs),
            "held_pairs": len(held_pairs),
            "held_pairs_list": [list(x) for x in sorted(held_pairs)],
            "property": "Each predicate appears in both training and held pair-composition cells; held cells test new cp×hp combinations, not zero-shot predicate words.",
        },
        "row_counts": {"train_total": len(train_rows), "support": len(support_rows), "composition_train": len(comp_train_rows), **{k: len(v) for k, v in eval_sets.items()}},
        "equivariance_pairs": len(eq_pairs),
        "audits": audits,
        "aggregates": aggregates,
        "per_seed": results,
    }
    return summary


def old_step253_rows_for_debug(train_fams: list[dict[str, Any]], limit_fams: int = 20) -> list[dict[str, Any]]:
    # Rebuild the research-style full predicate matrix, but with the corrected
    # tokenizer/model used only for truth/gradient debugging.
    fams = train_fams[:limit_fams]
    all_pairs = {(cp, hp) for cp in PRED_LIST for hp in PRED_LIST}
    return make_pair_rows(fams, "train", all_pairs, "legacy_step253_full_matrix_debug")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-fams", type=int, default=48)
    ap.add_argument("--held-fams", type=int, default=24)
    ap.add_argument("--held-domain-fams", type=int, default=24)
    ap.add_argument("--epochs", type=int, default=18)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--lam", type=float, default=0.35)
    ap.add_argument("--seeds", type=str, default="254,255,256")
    ap.add_argument("--max-eq-pairs", type=int, default=40000)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_train = read_jsonl(TRAIN)
    legacy_rows = old_step253_rows_for_debug(all_train, limit_fams=20)
    truth_audit = audit_rows(legacy_rows, "legacy_step253_full_matrix_debug")
    memorize = gradient_and_memorization_probe(legacy_rows, OUT_DIR)
    comp = run_composition_experiment(args, OUT_DIR)

    summary = {
        "status": "EQUIVARIANCE_TRUTH_GRADIENT_AND_COMPOSITION",
        "created_utc": now(),
        "script": str(_public_path('experiments/archive/representation_and_objectives/scripts/equivariance_truth_gradient_and_composition.py')),
        "purpose": "Repair research all-chance interpretation by verifying truth table/gradient/memorization, then comparing ordinary vs equivariant learning when every predicate has independent semantic evidence.",
        "inputs": {"train": str(TRAIN), "held": str(HELD)},
        "legacy_truth_table_audit": truth_audit,
        "gradient_and_memorization_probe": memorize,
        "composition_experiment": comp,
        "interpretation_boundary": "This controlled CPU probe can show whether the objective/data format is measurable and whether equivariance helps after predicate words are semantically supported. It is not BabyLM training and does not by itself establish a general natural data-efficient learning principle.",
    }
    write_json(OUT_DIR / "summary.json", summary)

    # Markdown synthesis.
    agg = comp["aggregates"]
    md: list[str] = []
    md.append("# research — Equivariance truth-table, gradient, and composition diagnostic")
    md.append("")
    md.append("## Why this run exists")
    md.append("")
    md.append("research produced all-chance GRU/BiGRU numbers. This run treats that as an unmeasured experiment: it first checks the symbolic labels and gradient path, then only compares standard versus equivariant learning after every predicate has independent semantic support.")
    md.append("")
    md.append("## Truth table and gradient path")
    md.append("")
    md.append(f"- Legacy full-matrix debug rows: {truth_audit['n']} rows; true fraction {truth_audit['label_fraction_true']:.3f}; duplicate text conflicts {truth_audit['duplicate_text_conflicts']}.")
    md.append(f"- Memorization slice: {memorize['slice_rows']} rows, labels {memorize['labels']}; first-step grad norm {memorize['first_step_grad_norm']:.4g}, first-step parameter delta {memorize['first_step_param_delta_l2']:.4g}.")
    md.append(f"- Initial acc/loss {memorize['initial_acc']:.3f}/{memorize['initial_loss']:.4f}; final acc/loss {memorize['final_acc']:.3f}/{memorize['final_loss']:.4f}; reached ≥0.999 acc epoch {memorize['reached_999_acc_epoch']}.")
    md.append("")
    md.append("## Corrected composition experiment")
    md.append("")
    md.append(f"- Train families {comp['family_counts']['train']}, held families {comp['family_counts']['held_family']}, held source-domain `{comp['held_domain']}` families {comp['family_counts']['held_domain']}.")
    md.append(f"- Train pair cells {comp['pair_split']['train_pairs']}; held pair cells {comp['pair_split']['held_pairs']}; equivariance pairs {comp['equivariance_pairs']}.")
    md.append(f"- Train rows {comp['row_counts']['train_total']} = support {comp['row_counts']['support']} + pair-composition {comp['row_counts']['composition_train']}.")
    md.append("")
    md.append("| Evaluation surface | Standard acc | Equivariant acc | Δ equiv-std |")
    md.append("|---|---:|---:|---:|")
    for name in ["semantic_support_held_family", "train_pair_held_family", "held_pair_seen_family", "held_pair_held_family", "held_pair_held_domain"]:
        st = agg[name]["standard"]
        eq = agg[name]["equivariant"]
        delta = agg[name]["delta_equiv_minus_standard"]
        md.append(f"| {name} | {st['mean']:.3f}±{st['std']:.3f} | {eq['mean']:.3f}±{eq['std']:.3f} | {delta:+.3f} |")
    md.append("")
    md.append("## Scientific reading")
    md.append("")
    if memorize["passed_gradient_path"] and memorize["passed_memorization"]:
        md.append("The research all-chance result should not be treated as evidence for a semantic-prior requirement: the corrected diagnostic has a live gradient path and can memorize a balanced slice.")
    else:
        md.append("The diagnostic still failed to establish a clean gradient/memorization path; inspect the saved slice before interpreting any transfer numbers.")
    md.append("The composition table is the first relevant ordinary-vs-equivariant comparison because predicate meanings are no longer zero-shot: each predicate appears in independent role-labeled support before held cp×hp combinations are tested.")
    md.append("These numbers remain a controlled small-model measurement. Natural-source work still needs independently attested event/state families, stronger domains, and connection to the EWoK/Entity bridge panel before any BabyLM-scale run is justified.")
    md.append("")
    md.append("## Files")
    md.append("")
    md.append(f"- Summary JSON: `{OUT_DIR / 'summary.json'}`")
    md.append(f"- Memorized slice rows: `{OUT_DIR / 'memorized_slice_rows.jsonl'}`")
    md.append(f"- Train/eval rows: `{OUT_DIR / 'semantic_support_train.jsonl'}`, `{OUT_DIR / 'composition_train_pairs.jsonl'}`, `{OUT_DIR / 'held_pair_held_family_eval.jsonl'}`")
    ((_PUBLIC_ROOT / 'research/documents/representation_and_objectives/data/equivariance_diagnostics/summary.md')).write_text("\n".join(md) + "\n", encoding="utf-8")

    print(json.dumps({
        "status": summary["status"],
        "summary_json": str(OUT_DIR / "summary.json"),
        "memorization_final_acc": memorize["final_acc"],
        "memorization_epoch_999": memorize["reached_999_acc_epoch"],
        "held_pair_held_family_standard": agg["held_pair_held_family"]["standard"]["mean"],
        "held_pair_held_family_equivariant": agg["held_pair_held_family"]["equivariant"]["mean"],
        "held_pair_held_domain_standard": agg["held_pair_held_domain"]["standard"]["mean"],
        "held_pair_held_domain_equivariant": agg["held_pair_held_domain"]["equivariant"]["mean"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
