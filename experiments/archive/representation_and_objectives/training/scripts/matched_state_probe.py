#!/usr/bin/env python3
"""research: matched state-format probe on the repaired research substrate.

Scientific purpose
------------------
research showed state-query out-of-distribution improvement without bridge-controlled
mixed relation orientation. This runner separates four explanations using the
lowest-cost learned comparison before any BabyLM-scale training:

  * aligned held-state evidence;
  * inverted held-state evidence;
  * row/update-matched neutral seen-state evidence with a held distractor;
  * row/update-matched random state labels on the same held-state surfaces;
  * a relation-only row-count control.

All state-format arms use 192 held-held comparison rows + 128 state-query rows
(320 rows total), the same epochs, seeds, head initialization, and optimizer.
The script saves per-row logits and grouped state-choice summaries, so later
steps can inspect local bridge fit, same-train-name transfer, one-new-name
transfer, two-new-name transfer, signed true-vs-inverted orientation, and
changed+unchanged conservation.

No official BabyLM evaluation, upload, or leaderboard interaction occurs.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import copy
import csv
import gc
import importlib.util
import json
import math
import os
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


DEFAULT_ARMS = [
    "heldheld_only",
    "heldheld_repeat_control",
    "aligned_matched",
    "inverted_matched",
    "neutral_matched",
    "random_label_matched",
]
DEFAULT_SEEDS = [28100, 28101, 28102]
EPOCHS = 50
LR_ENCODER = 2e-5
LR_HEAD = 1e-3
BATCH = 16
MAX_LEN = 196
WARMUP_FRAC = 0.1

BASE_EVAL_SUITES = [
    "heldheld_unseen_edge_closure",
    "mixed_held_seen_orientation",
    "paired_state_conservation",
    "cross_template_state_readout",
    "name_permutation_counterfactual",
]


def load_step278_module(workspace: Path):
    mod_path = workspace / "scripts" / "equivariant_symmetry_substrate.py"
    spec = importlib.util.spec_from_file_location("equivariant_symmetry_substrate", mod_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {mod_path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")


def key_for_train_copy(r: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(r)
    return out


def neutral_matched_rows(neutral_state_rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Downsample neutral decoupled rows to match the aligned/inverted state rows.

    We use only s_give active/passive rows, then select 16 two-candidate groups
    for each query_kind × voice. This yields exactly 128 candidate rows:
    2 query kinds × 2 voices × 16 pair groups × 2 candidates.
    """
    groups: Dict[Tuple[str, str], Dict[str, List[Dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for r in neutral_state_rows:
        if r.get("relation") != "s_give":
            continue
        if r.get("voice") not in {"active", "passive"}:
            continue
        groups[(str(r.get("query_kind")), str(r.get("voice")))][str(r.get("pair_id"))].append(r)

    selected: List[Dict[str, Any]] = []
    for qk in ["changed", "unchanged"]:
        for voice in ["active", "passive"]:
            pair_groups = groups[(qk, voice)]
            usable = []
            for pid, rows in sorted(pair_groups.items()):
                cands = sorted(rows, key=lambda x: int(x.get("candidate_slot", 0)))
                if len(cands) == 2 and {int(x.get("candidate_slot", -1)) for x in cands} == {0, 1}:
                    usable.append(cands)
            if len(usable) < 16:
                raise RuntimeError(f"Not enough neutral groups for {(qk, voice)}: {len(usable)}")
            for pair in usable[:16]:
                for r in pair:
                    nr = dict(r)
                    nr["row_id"] = "matched_neutral_" + str(nr["row_id"])
                    nr["pair_id"] = "matched_neutral_" + str(nr["pair_id"])
                    nr["suite"] = "neutral_matched_seen_state_with_held_distractor"
                    nr["matched_control"] = "neutral_seen_state_s_give"
                    selected.append(nr)
    if len(selected) != 128:
        raise RuntimeError(f"neutral matched row count {len(selected)} != 128")
    return selected


def random_label_state_rows(aligned_state_rows: Sequence[Dict[str, Any]], seed: int = 28100) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    groups: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for r in aligned_state_rows:
        groups[(str(r.get("pair_id")), str(r.get("query_kind")))].append(r)
    out: List[Dict[str, Any]] = []
    for (pid, qk), rows in sorted(groups.items()):
        cands = sorted(rows, key=lambda x: int(x.get("candidate_slot", 0)))
        if len(cands) != 2 or {int(x.get("candidate_slot", -1)) for x in cands} != {0, 1}:
            raise RuntimeError(f"Bad aligned state group {pid} {qk}")
        true_slot = rng.randint(0, 1)
        for r in cands:
            nr = dict(r)
            slot = int(nr.get("candidate_slot"))
            nr["row_id"] = "random_label_" + str(nr["row_id"])
            nr["pair_id"] = "random_label_" + str(nr["pair_id"])
            nr["suite"] = "random_label_state_format"
            nr["original_label"] = bool(nr.get("label"))
            nr["random_label_seed"] = seed
            nr["random_true_slot"] = true_slot
            nr["label"] = bool(slot == true_slot)
            nr["correct_slot"] = true_slot
            nr["matched_control"] = "random_state_label_on_held_surface"
            out.append(nr)
    if len(out) != 128:
        raise RuntimeError(f"random label row count {len(out)} != 128")
    return out


def repeat_hh_rows(hh_rows: Sequence[Dict[str, Any]], n_extra: int = 128) -> List[Dict[str, Any]]:
    out = [dict(r) for r in hh_rows]
    for i, r in enumerate(hh_rows[:n_extra]):
        nr = dict(r)
        nr["row_id"] = f"repeat_control_{i:04d}_" + str(nr["row_id"])
        nr["suite"] = "heldheld_repeat_rowcount_control"
        nr["matched_control"] = "relation_only_repeated_rows"
        out.append(nr)
    if len(out) != len(hh_rows) + n_extra:
        raise RuntimeError("bad repeat control size")
    return out


def build_matched_train_sets(substrate: Path) -> Dict[str, List[Dict[str, Any]]]:
    hh_all = load_jsonl(substrate / "arms" / "heldheld_only" / "train_supervised.jsonl")
    hh_rows = [r for r in hh_all if r.get("task") == "relation_comparison"]
    aligned_all = load_jsonl(substrate / "arms" / "aligned_state_bridge" / "train_supervised.jsonl")
    inverted_all = load_jsonl(substrate / "arms" / "inverted_state_bridge" / "train_supervised.jsonl")
    neutral_all = load_jsonl(substrate / "arms" / "neutral_decoupled" / "train_supervised.jsonl")
    aligned_state = [r for r in aligned_all if r.get("task") == "state_query"]
    inverted_state = [r for r in inverted_all if r.get("task") == "state_query"]
    neutral_state = [r for r in neutral_all if r.get("task") == "state_query"]
    if len(hh_rows) != 192 or len(aligned_state) != 128 or len(inverted_state) != 128:
        raise RuntimeError((len(hh_rows), len(aligned_state), len(inverted_state)))
    neutral_state = neutral_matched_rows(neutral_state)
    random_state = random_label_state_rows(aligned_state, seed=28100)
    return {
        "heldheld_only": [dict(r) for r in hh_rows],
        "heldheld_repeat_control": repeat_hh_rows(hh_rows, 128),
        "aligned_matched": [dict(r) for r in hh_rows] + [dict(r) for r in aligned_state],
        "inverted_matched": [dict(r) for r in hh_rows] + [dict(r) for r in inverted_state],
        "neutral_matched": [dict(r) for r in hh_rows] + neutral_state,
        "random_label_matched": [dict(r) for r in hh_rows] + random_state,
    }


def voice_to_int(v: str) -> int:
    return 0 if v == "active" else 1


def unique_state_groups(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    groups = []
    for r in rows:
        if r.get("task") != "state_query":
            continue
        pid = str(r.get("pair_id"))
        if pid in seen:
            continue
        seen.add(pid)
        groups.append(r)
    return groups


def build_support_suites(train_sets: Dict[str, List[Dict[str, Any]]], mod: Any) -> Dict[str, List[Dict[str, Any]]]:
    """Generate small support-distance readouts with true labels.

    They reuse the sparse aligned bridge relation/name/voice/static patterns, but
    vary object and entity novelty. The same rows are evaluated for every arm;
    summaries compute both true- and inverted-orientation agreement.
    """
    aligned_state = [r for r in train_sets["aligned_matched"] if r.get("task") == "state_query"]
    bases = unique_state_groups(aligned_state)
    suites: Dict[str, List[Dict[str, Any]]] = {
        "support_exact_aligned_train_true": [],
        "support_train_names_new_objects": [],
        "support_one_new_second_name": [],
        "support_one_new_first_name": [],
    }
    # Exact aligned train rows with their true labels, included as a common reference.
    suites["support_exact_aligned_train_true"] = [dict(r, suite="support_exact_aligned_train_true") for r in aligned_state]
    for i, r in enumerate(bases):
        rel = str(r["relation"])
        a, b = r["arg_order"]
        voice = voice_to_int(str(r["voice"]))
        static_slot = int(r["static_slot"])
        chg = mod.EVAL_CHANGED[(i * 3 + 1) % len(mod.EVAL_CHANGED)]
        st = mod.EVAL_STATIC[(i * 5 + 2) % len(mod.EVAL_STATIC)]
        rows = mod.state_rows(f"support_trainname_newobj_{i:04d}", rel, a, b, chg, st,
                              "eval", "support_train_names_new_objects", voice, static_slot, mod.TRUE_ASSIGNMENT)
        for x in rows:
            x["support_relation_seen_in_bridge"] = rel in {"h0_dax", "h2_norp"}
            x["name_novelty"] = "same_two_train_names"
        suites["support_train_names_new_objects"].extend(rows)

        b2 = mod.EVAL_NAMES[(i * 7 + 1) % len(mod.EVAL_NAMES)]
        rows = mod.state_rows(f"support_one_new_second_{i:04d}", rel, a, b2, chg, st,
                              "eval", "support_one_new_second_name", voice, static_slot, mod.TRUE_ASSIGNMENT)
        for x in rows:
            x["support_relation_seen_in_bridge"] = rel in {"h0_dax", "h2_norp"}
            x["name_novelty"] = "first_train_second_eval"
        suites["support_one_new_second_name"].extend(rows)

        a2 = mod.EVAL_NAMES[(i * 7 + 3) % len(mod.EVAL_NAMES)]
        rows = mod.state_rows(f"support_one_new_first_{i:04d}", rel, a2, b, chg, st,
                              "eval", "support_one_new_first_name", voice, static_slot, mod.TRUE_ASSIGNMENT)
        for x in rows:
            x["support_relation_seen_in_bridge"] = rel in {"h0_dax", "h2_norp"}
            x["name_novelty"] = "first_eval_second_train"
        suites["support_one_new_first_name"].extend(rows)
    return suites


class ClassificationModel(nn.Module):
    def __init__(self, base_model: nn.Module, hidden_size: int, dropout: float = 0.1):
        super().__init__()
        self.base = base_model
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_size, 2)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, token_type_ids: torch.Tensor | None = None) -> torch.Tensor:
        out = self.base(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        cls_rep = out.last_hidden_state[:, 0]
        return self.head(self.dropout(cls_rep))


def tokenize_row(row: Dict[str, Any], tokenizer: Any, max_len: int) -> Dict[str, Any]:
    if row["task"] == "relation_comparison":
        text_a, text_b = row["event1"], row["event2"]
    elif row["task"] == "state_query":
        text_a, text_b = row["premise"], row["hypothesis"]
    else:
        raise ValueError(row["task"])
    enc = tokenizer(text_a, text_b, max_length=max_len, truncation=True, padding=False)
    enc["label"] = 1 if row.get("label") else 0
    return enc


def make_batches(encoded_rows: Sequence[Dict[str, Any]], batch_size: int, pad_id: int, device: torch.device, shuffle: bool = False):
    idxs = list(range(len(encoded_rows)))
    if shuffle:
        random.shuffle(idxs)
    for start in range(0, len(idxs), batch_size):
        sel = idxs[start:start + batch_size]
        batch = [encoded_rows[i] for i in sel]
        max_len = max(len(b["input_ids"]) for b in batch)
        ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long, device=device)
        mask = torch.zeros(len(batch), max_len, dtype=torch.long, device=device)
        tids = torch.zeros(len(batch), max_len, dtype=torch.long, device=device)
        labels = torch.zeros(len(batch), dtype=torch.long, device=device)
        for i, b in enumerate(batch):
            L = len(b["input_ids"])
            ids[i, :L] = torch.tensor(b["input_ids"], dtype=torch.long, device=device)
            mask[i, :L] = torch.tensor(b["attention_mask"], dtype=torch.long, device=device)
            if "token_type_ids" in b:
                tids[i, :L] = torch.tensor(b["token_type_ids"], dtype=torch.long, device=device)
            labels[i] = int(b["label"])
        yield sel, ids, mask, tids, labels


def train_one_arm(model: nn.Module, train_encoded: Sequence[Dict[str, Any]], epochs: int, pad_id: int, device: torch.device) -> Dict[str, float]:
    if len(train_encoded) == 0:
        return {"train_acc_last_epoch": float("nan"), "train_loss_last_epoch": float("nan")}
    encoder_params = [p for n, p in model.named_parameters() if not n.startswith("head")]
    head_params = [p for n, p in model.named_parameters() if n.startswith("head")]
    optimizer = torch.optim.AdamW([
        {"params": encoder_params, "lr": LR_ENCODER, "weight_decay": 0.01},
        {"params": head_params, "lr": LR_HEAD, "weight_decay": 0.0},
    ])
    total_steps = epochs * max(1, (len(train_encoded) + BATCH - 1) // BATCH)
    warmup_steps = int(total_steps * WARMUP_FRAC)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        return max(0.0, 1.0 - (step - warmup_steps) / max(total_steps - warmup_steps, 1))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    last_acc = 0.0
    last_loss = 0.0
    model.train()
    for _ep in range(epochs):
        total_loss = 0.0
        total_correct = 0
        total_n = 0
        for _sel, ids, mask, tids, labels in make_batches(train_encoded, BATCH, pad_id, device, shuffle=True):
            logits = model(ids, mask, tids)
            loss = F.cross_entropy(logits, labels)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += float(loss.item()) * len(labels)
            total_correct += int((logits.argmax(1) == labels).sum().item())
            total_n += int(len(labels))
        last_acc = total_correct / max(1, total_n)
        last_loss = total_loss / max(1, total_n)
    return {"train_acc_last_epoch": last_acc, "train_loss_last_epoch": last_loss}


@torch.no_grad()
def predict(model: nn.Module, encoded: Sequence[Dict[str, Any]], raw_rows: Sequence[Dict[str, Any]], suite_name: str, arm: str, seed: int, pad_id: int, device: torch.device, batch_size: int = 64) -> List[Dict[str, Any]]:
    model.eval()
    pred_rows: List[Dict[str, Any]] = []
    for sel, ids, mask, tids, labels in make_batches(encoded, batch_size, pad_id, device, shuffle=False):
        logits = model(ids, mask, tids)
        probs = torch.softmax(logits, dim=1)
        for j, row_idx in enumerate(sel):
            r = raw_rows[row_idx]
            l0 = float(logits[j, 0].detach().cpu().item())
            l1 = float(logits[j, 1].detach().cpu().item())
            p1 = float(probs[j, 1].detach().cpu().item())
            rec = {
                "arm": arm,
                "seed": seed,
                "suite_eval": suite_name,
                "row_id": str(r.get("row_id")),
                "pair_id": str(r.get("pair_id", "")),
                "task": str(r.get("task")),
                "query_kind": str(r.get("query_kind", "")),
                "relation": str(r.get("relation", r.get("relation1", ""))),
                "relation1": str(r.get("relation1", "")),
                "relation2": str(r.get("relation2", "")),
                "component": str(r.get("component", "")),
                "component1": str(r.get("component1", "")),
                "component2": str(r.get("component2", "")),
                "voice": str(r.get("voice", "")),
                "voice1": str(r.get("voice1", "")),
                "voice2": str(r.get("voice2", "")),
                "candidate_slot": int(r.get("candidate_slot", -1)) if str(r.get("candidate_slot", "")) != "" else -1,
                "correct_slot": int(r.get("correct_slot", -1)) if str(r.get("correct_slot", "")) != "" else -1,
                "static_slot": int(r.get("static_slot", -1)) if str(r.get("static_slot", "")) != "" else -1,
                "label": bool(r.get("label")),
                "logit_false": l0,
                "logit_true": l1,
                "prob_true": p1,
                "pred_label": bool(l1 > l0),
                "label_margin": l1 - l0,
                "orientation_dependency": str(r.get("orientation_dependency", "")),
                "name_novelty": str(r.get("name_novelty", "")),
                "support_relation_seen_in_bridge": r.get("support_relation_seen_in_bridge", None),
                "matched_control": str(r.get("matched_control", "")),
            }
            pred_rows.append(rec)
    return pred_rows


def safe_mean(xs: Sequence[float]) -> float | None:
    vals = [float(x) for x in xs if x is not None and not math.isnan(float(x))]
    return float(np.mean(vals)) if vals else None


def safe_std(xs: Sequence[float]) -> float | None:
    vals = [float(x) for x in xs if x is not None and not math.isnan(float(x))]
    return float(np.std(vals)) if vals else None


def label_for_assignment(row: Dict[str, Any], mod: Any, assignment_name: str) -> bool:
    if assignment_name == "row":
        return bool(row.get("label"))
    assignment = mod.TRUE_ASSIGNMENT if assignment_name == "true" else mod.INVERTED_ASSIGNMENT
    return bool(mod.predict_label(row, assignment))


def row_statement_summary(pred_rows: Sequence[Dict[str, Any]], raw_rows: Sequence[Dict[str, Any]], mod: Any, assignment_name: str) -> Dict[str, Any]:
    vals = []
    margins = []
    for pr, rr in zip(pred_rows, raw_rows):
        target = label_for_assignment(rr, mod, assignment_name)
        pred = bool(pr["pred_label"])
        vals.append(1.0 if pred == target else 0.0)
        margins.append(float(pr["label_margin"]) * (1.0 if target else -1.0))
    return {"acc": safe_mean(vals), "margin": safe_mean(margins), "n": len(vals)}


def state_choice_summary(pred_rows: Sequence[Dict[str, Any]], raw_rows: Sequence[Dict[str, Any]], mod: Any, assignment_name: str) -> Dict[str, Any]:
    groups: Dict[Tuple[str, str], List[Tuple[Dict[str, Any], Dict[str, Any]]]] = defaultdict(list)
    for pr, rr in zip(pred_rows, raw_rows):
        if rr.get("task") != "state_query":
            continue
        groups[(str(rr.get("pair_id")), str(rr.get("query_kind")))].append((pr, rr))
    choices = []
    margins = []
    by_qk: Dict[str, List[float]] = defaultdict(list)
    by_rel: Dict[str, List[float]] = defaultdict(list)
    by_voice: Dict[str, List[float]] = defaultdict(list)
    by_rel_qk: Dict[Tuple[str, str], List[float]] = defaultdict(list)
    pair_choice: Dict[Tuple[str, str], bool] = {}
    for (pid, qk), items in groups.items():
        if len(items) != 2:
            continue
        # target slot under selected label interpretation
        target_slots = []
        for _pr, rr in items:
            if label_for_assignment(rr, mod, assignment_name):
                target_slots.append(int(rr.get("candidate_slot")))
        if len(target_slots) != 1:
            continue
        target_slot = target_slots[0]
        score_by_slot = {int(rr.get("candidate_slot")): float(pr["logit_true"]) for pr, rr in items}
        if set(score_by_slot) != {0, 1}:
            continue
        choice_slot = 0 if score_by_slot[0] >= score_by_slot[1] else 1
        ok = float(choice_slot == target_slot)
        choices.append(ok)
        margin = score_by_slot[target_slot] - score_by_slot[1 - target_slot]
        margins.append(margin)
        rr0 = items[0][1]
        rel = str(rr0.get("relation"))
        voice = str(rr0.get("voice"))
        by_qk[qk].append(ok)
        by_rel[rel].append(ok)
        by_voice[voice].append(ok)
        by_rel_qk[(rel, qk)].append(ok)
        pair_choice[(pid, qk)] = bool(ok)
    # changed+unchanged conservation at the pair_id level
    both = []
    changed_only = []
    unchanged_only = []
    pids = sorted({pid for pid, _qk in pair_choice})
    for pid in pids:
        if (pid, "changed") in pair_choice and (pid, "unchanged") in pair_choice:
            ch = pair_choice[(pid, "changed")]
            un = pair_choice[(pid, "unchanged")]
            both.append(1.0 if ch and un else 0.0)
            changed_only.append(1.0 if ch and not un else 0.0)
            unchanged_only.append(1.0 if un and not ch else 0.0)
    return {
        "choice_acc": safe_mean(choices),
        "choice_margin": safe_mean(margins),
        "n_choice_groups": len(choices),
        "pair_both": safe_mean(both),
        "changed_only": safe_mean(changed_only),
        "unchanged_only": safe_mean(unchanged_only),
        "by_query_kind": {k: safe_mean(v) for k, v in sorted(by_qk.items())},
        "by_relation": {k: safe_mean(v) for k, v in sorted(by_rel.items())},
        "by_voice": {k: safe_mean(v) for k, v in sorted(by_voice.items())},
        "by_relation_query_kind": {f"{k[0]}::{k[1]}": safe_mean(v) for k, v in sorted(by_rel_qk.items())},
    }


def summarize_by_seed(suite_predictions: Dict[str, Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]], mod: Any) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for suite, (pred_rows, raw_rows) in suite_predictions.items():
        suite_out: Dict[str, Any] = {}
        for assignment_name in ["row", "true", "inverted"]:
            suite_out[f"statement_{assignment_name}"] = row_statement_summary(pred_rows, raw_rows, mod, assignment_name)
            if any(r.get("task") == "state_query" for r in raw_rows):
                suite_out[f"state_choice_{assignment_name}"] = state_choice_summary(pred_rows, raw_rows, mod, assignment_name)
        out[suite] = suite_out
    return out


def aggregate_seed_summaries(all_seed_summaries: Dict[str, Dict[int, Dict[str, Any]]]) -> Dict[str, Any]:
    # Flatten numeric leaves across seeds for arm/suite/metric paths.
    agg: Dict[str, Any] = {}
    for arm, seed_map in all_seed_summaries.items():
        arm_out: Dict[str, Any] = {}
        paths: Dict[Tuple[str, ...], List[float]] = defaultdict(list)

        def collect(prefix: Tuple[str, ...], obj: Any) -> None:
            if isinstance(obj, dict):
                for k, v in obj.items():
                    collect(prefix + (str(k),), v)
            elif isinstance(obj, (int, float)) and not isinstance(obj, bool):
                if not math.isnan(float(obj)):
                    paths[prefix].append(float(obj))

        for _seed, summary in seed_map.items():
            collect((), summary)
        for path, vals in paths.items():
            cur = arm_out
            for p in path[:-1]:
                cur = cur.setdefault(p, {})
            cur[path[-1] + "_mean"] = safe_mean(vals)
            cur[path[-1] + "_std"] = safe_std(vals)
            cur[path[-1] + "_n"] = len(vals)
        agg[arm] = arm_out
    return agg


def get_path(obj: Dict[str, Any], path: Sequence[str]) -> Any:
    cur: Any = obj
    for p in path:
        if not isinstance(cur, dict) or p not in cur:
            return None
        cur = cur[p]
    return cur


def fmt(x: Any, digits: int = 3) -> str:
    if x is None:
        return "nan"
    try:
        xf = float(x)
    except Exception:
        return str(x)
    if math.isnan(xf):
        return "nan"
    return f"{xf:.{digits}f}"


def write_summary_md(path: Path, args: argparse.Namespace, train_sets: Dict[str, List[Dict[str, Any]]], aggregate: Dict[str, Any], all_seed_summaries: Dict[str, Dict[int, Dict[str, Any]]], per_arm_elapsed: Dict[str, List[float]]) -> None:
    lines: List[str] = []
    lines.append("# research matched state-format probe")
    lines.append("")
    lines.append("This learned probe keeps the repaired research text surface but equalizes the state-format arms: each state arm has 192 held-held comparison rows plus 128 state-query rows and the same number of epochs/update passes. It tests whether research's state-query transfer came from bridge polarity or from generic state-format/optimization effects.")
    lines.append("")
    lines.append(f"- seeds: `{args.seeds}`")
    lines.append(f"- epochs: `{args.epochs}`")
    lines.append(f"- arms: `{args.arms}`")
    lines.append("- no official evaluation, upload, or leaderboard interaction")
    lines.append("")
    lines.append("## Training row construction")
    lines.append("")
    lines.append("| arm | total rows | relation rows | state rows | state source |")
    lines.append("|---|---:|---:|---:|---|")
    for arm in args.arms:
        rows = train_sets[arm]
        n_rel = sum(1 for r in rows if r.get("task") == "relation_comparison")
        n_state = sum(1 for r in rows if r.get("task") == "state_query")
        src = Counter(str(r.get("suite")) for r in rows if r.get("task") == "state_query")
        src_s = "; ".join(f"{k}:{v}" for k, v in sorted(src.items())) or "none"
        lines.append(f"| {arm} | {len(rows)} | {n_rel} | {n_state} | {src_s} |")
    lines.append("")

    lines.append("## Core readouts (mean over seeds)")
    lines.append("")
    lines.append("`mixed_true_stmt` is true-assignment statement accuracy on mixed held-seen relation comparisons; `state_true_choice` is pair-choice accuracy on paired held-state rows; `state_inverted_choice` is the same rows scored against the globally inverted held-role orientation; `pair_both_true` requires changed and unchanged state choices to be simultaneously true.")
    lines.append("")
    lines.append("| arm | train row-label stmt | mixed true stmt | mixed inverted stmt | paired state true choice | paired state inverted choice | pair_both true | pair_both inverted | exact train state row-label choice |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for arm in args.arms:
        a = aggregate[arm]
        row_train = get_path(a, ["train_all", "statement_row", "acc_mean"])
        mixed_t = get_path(a, ["mixed_held_seen_orientation", "statement_true", "acc_mean"])
        mixed_i = get_path(a, ["mixed_held_seen_orientation", "statement_inverted", "acc_mean"])
        st_t = get_path(a, ["paired_state_conservation", "state_choice_true", "choice_acc_mean"])
        st_i = get_path(a, ["paired_state_conservation", "state_choice_inverted", "choice_acc_mean"])
        both_t = get_path(a, ["paired_state_conservation", "state_choice_true", "pair_both_mean"])
        both_i = get_path(a, ["paired_state_conservation", "state_choice_inverted", "pair_both_mean"])
        exact_train_state = get_path(a, ["train_state_only", "state_choice_row", "choice_acc_mean"])
        lines.append(f"| {arm} | {fmt(row_train)} | {fmt(mixed_t)} | {fmt(mixed_i)} | {fmt(st_t)} | {fmt(st_i)} | {fmt(both_t)} | {fmt(both_i)} | {fmt(exact_train_state)} |")
    lines.append("")

    lines.append("## Support-distance state choices (mean over seeds, true orientation)")
    lines.append("")
    lines.append("| arm | exact aligned train surface | same train names/new objects | first train + second eval | first eval + second train | paired two-eval names | cross-template two-eval |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for arm in args.arms:
        a = aggregate[arm]
        vals = [
            get_path(a, ["support_exact_aligned_train_true", "state_choice_true", "choice_acc_mean"]),
            get_path(a, ["support_train_names_new_objects", "state_choice_true", "choice_acc_mean"]),
            get_path(a, ["support_one_new_second_name", "state_choice_true", "choice_acc_mean"]),
            get_path(a, ["support_one_new_first_name", "state_choice_true", "choice_acc_mean"]),
            get_path(a, ["paired_state_conservation", "state_choice_true", "choice_acc_mean"]),
            get_path(a, ["cross_template_state_readout", "state_choice_true", "choice_acc_mean"]),
        ]
        lines.append(f"| {arm} | " + " | ".join(fmt(v) for v in vals) + " |")
    lines.append("")

    lines.append("## Same support-distance rows scored against inverted orientation")
    lines.append("")
    lines.append("| arm | exact aligned train surface | same train names/new objects | first train + second eval | first eval + second train | paired two-eval names | cross-template two-eval |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for arm in args.arms:
        a = aggregate[arm]
        vals = [
            get_path(a, ["support_exact_aligned_train_true", "state_choice_inverted", "choice_acc_mean"]),
            get_path(a, ["support_train_names_new_objects", "state_choice_inverted", "choice_acc_mean"]),
            get_path(a, ["support_one_new_second_name", "state_choice_inverted", "choice_acc_mean"]),
            get_path(a, ["support_one_new_first_name", "state_choice_inverted", "choice_acc_mean"]),
            get_path(a, ["paired_state_conservation", "state_choice_inverted", "choice_acc_mean"]),
            get_path(a, ["cross_template_state_readout", "state_choice_inverted", "choice_acc_mean"]),
        ]
        lines.append(f"| {arm} | " + " | ".join(fmt(v) for v in vals) + " |")
    lines.append("")

    # Highlight aligned vs inverted vs neutral/random on changed rows.
    lines.append("## Changed-state relation split on paired two-eval names")
    lines.append("")
    lines.append("True-orientation choice accuracy by held relation and changed/unchanged query. Relations h0/h2 were directly bridged in aligned/inverted arms; h1/h3 were only connected through held-held comparison rows.")
    lines.append("")
    lines.append("| arm | h0 changed | h1 changed | h2 changed | h3 changed | h0 unchanged | h1 unchanged | h2 unchanged | h3 unchanged |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for arm in args.arms:
        a = aggregate[arm]
        vals = []
        for rel in ["h0_dax", "h1_mep", "h2_norp", "h3_ziv"]:
            vals.append(get_path(a, ["paired_state_conservation", "state_choice_true", "by_relation_query_kind", f"{rel}::changed_mean"]))
        for rel in ["h0_dax", "h1_mep", "h2_norp", "h3_ziv"]:
            vals.append(get_path(a, ["paired_state_conservation", "state_choice_true", "by_relation_query_kind", f"{rel}::unchanged_mean"]))
        lines.append(f"| {arm} | " + " | ".join(fmt(v) for v in vals) + " |")
    lines.append("")

    lines.append("## Interpretation from this run")
    lines.append("")
    # Compute a small evidence paragraph from aggregate values.
    aligned = aggregate.get("aligned_matched", {})
    inverted = aggregate.get("inverted_matched", {})
    neutral = aggregate.get("neutral_matched", {})
    random_arm = aggregate.get("random_label_matched", {})
    al_state = get_path(aligned, ["paired_state_conservation", "state_choice_true", "choice_acc_mean"])
    inv_state_true = get_path(inverted, ["paired_state_conservation", "state_choice_true", "choice_acc_mean"])
    inv_state_inv = get_path(inverted, ["paired_state_conservation", "state_choice_inverted", "choice_acc_mean"])
    neu_state = get_path(neutral, ["paired_state_conservation", "state_choice_true", "choice_acc_mean"])
    rnd_state = get_path(random_arm, ["paired_state_conservation", "state_choice_true", "choice_acc_mean"])
    al_mixed = get_path(aligned, ["mixed_held_seen_orientation", "statement_true", "acc_mean"])
    inv_mixed = get_path(inverted, ["mixed_held_seen_orientation", "statement_true", "acc_mean"])
    lines.append(f"- Mixed held-seen relation orientation remains near chance in the polarity arms: aligned true-statement {fmt(al_mixed)}, inverted true-statement {fmt(inv_mixed)}. This run therefore does not turn state evidence into a reusable relation coordinate.")
    lines.append(f"- Paired held-state true-choice accuracy: aligned {fmt(al_state)}, inverted scored as true {fmt(inv_state_true)}, inverted scored as inverted {fmt(inv_state_inv)}, neutral {fmt(neu_state)}, random-label {fmt(rnd_state)}.")
    lines.append("- Compare the exact train-state row-label fit and the support-distance tables to decide whether inverted evidence was locally learned, whether any effect follows training identities, and whether neutral/random state-format exposure reproduces held-state transfer.")
    lines.append("")
    lines.append("## Files")
    lines.append("")
    lines.append("- per-row logits: `per_row_predictions.jsonl`")
    lines.append("- seed-level summaries: `seed_summaries.json`")
    lines.append("- aggregate summaries: `aggregate_summary.json`")
    lines.append("- train construction: `matched_train_construction.json`")
    path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--arms", type=str, nargs="+", default=DEFAULT_ARMS)
    parser.add_argument("--outdir", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--dry-run-rows", action="store_true", help="Only construct datasets and support suites, then exit.")
    args = parser.parse_args()

    workspace = _public_path('experiments/archive/representation_and_objectives')
    substrate = workspace / "data" / "equivariant_symmetry_substrate"
    ckpt = workspace / "training" / "runs" / "qwen_8x480_16k_wwm_to_token_100M_seed43022" / "hf_model" / "chck_80M"
    outdir = Path(args.outdir) if args.outdir else workspace / "data" / "matched_state_probe"
    outdir.mkdir(parents=True, exist_ok=True)

    mod = load_step278_module(workspace)
    train_sets_all = build_matched_train_sets(substrate)
    for arm in args.arms:
        if arm not in train_sets_all:
            raise ValueError(f"Unknown arm {arm}; options {sorted(train_sets_all)}")
    train_sets = {arm: train_sets_all[arm] for arm in args.arms}
    support_suites = build_support_suites(train_sets_all, mod)

    construction = {}
    for arm, rows in train_sets.items():
        construction[arm] = {
            "total_rows": len(rows),
            "relation_rows": sum(1 for r in rows if r.get("task") == "relation_comparison"),
            "state_rows": sum(1 for r in rows if r.get("task") == "state_query"),
            "task_counts": dict(Counter(str(r.get("task")) for r in rows)),
            "state_suite_counts": dict(Counter(str(r.get("suite")) for r in rows if r.get("task") == "state_query")),
            "state_query_counts": dict(Counter(str(r.get("query_kind")) for r in rows if r.get("task") == "state_query")),
            "state_voice_counts": dict(Counter(str(r.get("voice")) for r in rows if r.get("task") == "state_query")),
            "state_candidate_slot_counts": dict(Counter(str(r.get("candidate_slot")) for r in rows if r.get("task") == "state_query")),
        }
    construction["support_suites"] = {k: len(v) for k, v in support_suites.items()}
    (outdir / "matched_train_construction.json").write_text(json.dumps(construction, indent=2, sort_keys=True))
    for arm, rows in train_sets.items():
        write_jsonl(outdir / f"train_{arm}.jsonl", rows)
    for suite, rows in support_suites.items():
        write_jsonl(outdir / f"eval_{suite}.jsonl", rows)
    if args.dry_run_rows:
        print(json.dumps({"status": "ROWS_ONLY", "outdir": str(outdir), "construction": construction}, indent=2), flush=True)
        return

    device = torch.device(args.device if args.device else ("cuda:0" if torch.cuda.is_available() else "cpu"))
    print(f"Device: {device}", flush=True)
    print(f"Checkpoint: {ckpt}", flush=True)
    print(f"Output: {outdir}", flush=True)

    from transformers import AutoConfig, AutoTokenizer, DebertaV2Model
    tokenizer = AutoTokenizer.from_pretrained(str(ckpt), local_files_only=True)
    pad_id = tokenizer.pad_token_id or 0
    config = AutoConfig.from_pretrained(str(ckpt), local_files_only=True)
    hidden_size = int(config.hidden_size)
    print(f"Hidden size: {hidden_size}", flush=True)

    # Raw eval suites shared by every arm.
    eval_raw: Dict[str, List[Dict[str, Any]]] = {}
    for suite in BASE_EVAL_SUITES:
        eval_raw[suite] = load_jsonl(substrate / "eval" / f"{suite}.jsonl")
    eval_raw.update(support_suites)

    # Tokenize once.
    train_encoded = {arm: [tokenize_row(r, tokenizer, MAX_LEN) for r in rows] for arm, rows in train_sets.items()}
    train_state_raw = {arm: [r for r in rows if r.get("task") == "state_query"] for arm, rows in train_sets.items()}
    train_state_encoded = {arm: [tokenize_row(r, tokenizer, MAX_LEN) for r in rows] for arm, rows in train_state_raw.items()}
    eval_encoded = {suite: [tokenize_row(r, tokenizer, MAX_LEN) for r in rows] for suite, rows in eval_raw.items()}

    all_seed_summaries: Dict[str, Dict[int, Dict[str, Any]]] = {arm: {} for arm in args.arms}
    per_arm_elapsed: Dict[str, List[float]] = {arm: [] for arm in args.arms}
    per_row_path = outdir / "per_row_predictions.jsonl"
    if per_row_path.exists():
        per_row_path.unlink()

    for seed in args.seeds:
        set_seed(seed)
        ref_head = nn.Linear(hidden_size, 2)
        head_init_state = copy.deepcopy(ref_head.state_dict())
        del ref_head

        for arm in args.arms:
            t0 = time.time()
            set_seed(seed)
            base_model = DebertaV2Model.from_pretrained(str(ckpt), local_files_only=True)
            model = ClassificationModel(base_model, hidden_size)
            model.head.load_state_dict(head_init_state)
            model.to(device)

            train_stats = train_one_arm(model, train_encoded[arm], args.epochs, pad_id, device)

            suite_predictions: Dict[str, Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]] = {}
            # Training rows under row labels; this measures exact local fit for every arm.
            pred_train_all = predict(model, train_encoded[arm], train_sets[arm], "train_all", arm, seed, pad_id, device)
            suite_predictions["train_all"] = (pred_train_all, train_sets[arm])
            if train_state_raw[arm]:
                pred_train_state = predict(model, train_state_encoded[arm], train_state_raw[arm], "train_state_only", arm, seed, pad_id, device)
                suite_predictions["train_state_only"] = (pred_train_state, train_state_raw[arm])
            else:
                pred_train_state = []
            # Shared held-out and support rows.
            for suite, rows in eval_raw.items():
                preds = predict(model, eval_encoded[suite], rows, suite, arm, seed, pad_id, device)
                suite_predictions[suite] = (preds, rows)

            seed_summary = summarize_by_seed(suite_predictions, mod)
            seed_summary["train_stats_last_epoch"] = train_stats
            elapsed = time.time() - t0
            seed_summary["elapsed_sec"] = elapsed
            all_seed_summaries[arm][seed] = seed_summary
            per_arm_elapsed[arm].append(elapsed)

            # Stream predictions to disk with true/inverted target labels added.
            with per_row_path.open("a") as f:
                for suite, (preds, raws) in suite_predictions.items():
                    for pr, rr in zip(preds, raws):
                        pr2 = dict(pr)
                        pr2["target_true_assignment"] = label_for_assignment(rr, mod, "true")
                        pr2["target_inverted_assignment"] = label_for_assignment(rr, mod, "inverted")
                        pr2["row_label_correct"] = bool(pr2["pred_label"] == bool(rr.get("label")))
                        pr2["true_assignment_correct"] = bool(pr2["pred_label"] == pr2["target_true_assignment"])
                        pr2["inverted_assignment_correct"] = bool(pr2["pred_label"] == pr2["target_inverted_assignment"])
                        f.write(json.dumps(pr2, ensure_ascii=False, sort_keys=True) + "\n")

            paired_true = get_path(seed_summary, ["paired_state_conservation", "state_choice_true", "choice_acc"])
            paired_inv = get_path(seed_summary, ["paired_state_conservation", "state_choice_inverted", "choice_acc"])
            mixed_true = get_path(seed_summary, ["mixed_held_seen_orientation", "statement_true", "acc"])
            train_row = get_path(seed_summary, ["train_all", "statement_row", "acc"])
            exact_state = get_path(seed_summary, ["train_state_only", "state_choice_row", "choice_acc"])
            print(
                f"arm={arm} seed={seed} train_row={fmt(train_row,4)} exact_state={fmt(exact_state,4)} "
                f"mixed_true={fmt(mixed_true,4)} paired_true={fmt(paired_true,4)} paired_inv={fmt(paired_inv,4)} "
                f"[{elapsed:.1f}s]",
                flush=True,
            )

            model.cpu()
            del model, base_model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Save after each seed for resilience.
        (outdir / "seed_summaries.json").write_text(json.dumps(all_seed_summaries, indent=2, sort_keys=True))

    aggregate = aggregate_seed_summaries(all_seed_summaries)
    (outdir / "seed_summaries.json").write_text(json.dumps(all_seed_summaries, indent=2, sort_keys=True))
    (outdir / "aggregate_summary.json").write_text(json.dumps(aggregate, indent=2, sort_keys=True))
    summary_md = outdir / "matched_state_probe_summary.md"
    write_summary_md(summary_md, args, train_sets, aggregate, all_seed_summaries, per_arm_elapsed)

    # Print compact scientific signal.
    compact = {
        "status": "MATCHED_STATE_PROBE_COMPLETE",
        "summary": str(summary_md),
        "aggregate": str(outdir / "aggregate_summary.json"),
        "per_row_predictions": str(per_row_path),
        "core": {},
        "no_official_evaluation_upload_or_leaderboard": True,
    }
    for arm in args.arms:
        a = aggregate[arm]
        compact["core"][arm] = {
            "train_row_acc": get_path(a, ["train_all", "statement_row", "acc_mean"]),
            "exact_train_state_choice": get_path(a, ["train_state_only", "state_choice_row", "choice_acc_mean"]),
            "mixed_true_stmt": get_path(a, ["mixed_held_seen_orientation", "statement_true", "acc_mean"]),
            "mixed_inverted_stmt": get_path(a, ["mixed_held_seen_orientation", "statement_inverted", "acc_mean"]),
            "paired_state_true_choice": get_path(a, ["paired_state_conservation", "state_choice_true", "choice_acc_mean"]),
            "paired_state_inverted_choice": get_path(a, ["paired_state_conservation", "state_choice_inverted", "choice_acc_mean"]),
            "pair_both_true": get_path(a, ["paired_state_conservation", "state_choice_true", "pair_both_mean"]),
        }
    print(json.dumps(compact, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
