#!/usr/bin/env python3
"""research: trusted private-adapter answer-phase harness for plain-use packets.

The input packets are research plain-use state-update rows.  Each row has a
source sentence, an update sentence, and a use sentence that expresses either the
new state (UPDATED_USE) or the source state after a distractor update
(UNCHANGED_DISTRACTOR_USE).  This harness maps those rows into the same kind of
answer-span-only supervision used in the repaired relation-first tests while
preserving the trusted coherent86 private-adapter loader and an explicit model
identity record.

For non-exact paraphrastic use sentences, the supervised answer span is the
smallest use-sentence token window that covers the validator-provided state-hit
terms.  The foil is the competing canonical state phrase.  Scores are mean
masked-token log-probability differences for the answer span versus the foil
phrase in the same source+update+use frame.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import collections
import hashlib
import json
import math
import os
import pathlib
import random
import re
import sys
import time
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

SCRIPT_DIR = _public_path('experiments/archive/functional_learning/scripts')
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import relation_first_repaired_harness as h  # noqa: E402

ROOT = _public_path('.')
STUDY = _public_path('experiments/archive/functional_learning')
A02 = _public_path('experiments/archive/relation_learning')
DEFAULT_TRAIN = _public_path('experiments/archive/relation_learning/data/state_use_generation/validated_state_use_packets_plainuse_full_train.jsonl')
DEFAULT_HELD = _public_path('experiments/archive/relation_learning/data/state_use_generation/validated_state_use_packets_plainuse_full_heldout.jsonl')
DEFAULT_OUT = _public_path('experiments/archive/functional_learning/data/plainuse_answer_harness')
WORD_RE = re.compile(r"[A-Za-z0-9]+(?:[\u2019'][A-Za-z0-9]+)?")


def rel(path: pathlib.Path | str) -> str:
    try:
        return str(pathlib.Path(path).resolve().relative_to(ROOT))
    except Exception:
        return str(path)


def now_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha256_file(path: pathlib.Path) -> str:
    hsh = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            hsh.update(chunk)
    return hsh.hexdigest()


def read_jsonl(path: pathlib.Path) -> List[Dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: pathlib.Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def finite_mean(xs: Iterable[float]) -> float:
    vals = [float(x) for x in xs if x is not None and math.isfinite(float(x))]
    return sum(vals) / len(vals) if vals else float("nan")


def stem(word: str) -> str:
    w = str(word).lower().strip("\u2019'")
    if w.endswith("'s"):
        w = w[:-2]
    if len(w) > 6 and w.endswith("ing"):
        return w[:-3]
    if len(w) > 5 and w.endswith("ied"):
        return w[:-3] + "y"
    if len(w) > 5 and w.endswith("ed"):
        return w[:-2]
    if len(w) > 5 and w.endswith("es"):
        return w[:-2]
    if len(w) > 4 and w.endswith("s") and not w.endswith("ss"):
        return w[:-1]
    return w


def word_tokens(text: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for m in WORD_RE.finditer(str(text)):
        raw = m.group(0)
        out.append({"raw": raw, "stem": stem(raw), "start": m.start(), "end": m.end()})
    return out


def find_case_insensitive_span(text: str, needle: str) -> Optional[Tuple[int, int]]:
    low = text.lower()
    n = needle.strip().lower()
    if not n:
        return None
    i = low.find(n)
    if i < 0:
        return None
    return i, i + len(n)


def choose_hit_window(use_sentence: str, hits: Sequence[str], max_window_tokens: int) -> Tuple[int, int, Dict[str, Any]]:
    toks = word_tokens(use_sentence)
    # Validator hits are already lightweight stems in the validation files (e.g. `enclos`,
    # `analysi`).  Stemming them again can corrupt final-s stems, so match both
    # the raw lowercased hit and this script's stemmed variant against raw and
    # stemmed use-sentence tokens.
    hitset_raw = {str(x).lower().strip("\u2019'") for x in hits if str(x).strip()}
    hitset = {x for x in hitset_raw if x} | {stem(x) for x in hitset_raw if x}
    if not toks or not hitset:
        raise ValueError("empty use tokens or state-hit terms")
    best = None
    for i in range(len(toks)):
        have: set[str] = set()
        for j in range(i, min(len(toks), i + int(max_window_tokens))):
            token_forms = {str(toks[j]["raw"]).lower(), str(toks[j]["stem"])}
            matched = sorted(hitset & token_forms)
            if matched:
                have.update(matched)
            if have:
                width = j - i + 1
                coverage = len(have)
                contains_all = coverage == len(hitset)
                # Prefer all validator hits, then short windows, then later windows
                # when an entity name at the sentence start repeats inside the state.
                key = (contains_all, coverage, -width, i)
                if best is None or key > best[0]:
                    best = (key, i, j, sorted(have))
    if best is None:
        raise ValueError(f"no state-hit token found in use sentence: hits={hits!r} use={use_sentence!r}")
    _key, i, j, have = best
    missing = sorted(hitset - set(have))
    return int(toks[i]["start"]), int(toks[j]["end"]), {
        "method": "validator_hit_window",
        "hit_terms": sorted(hitset),
        "covered_hit_terms": have,
        "missing_hit_terms": missing,
        "window_token_count": int(j - i + 1),
        "window_tokens": [t["raw"] for t in toks[i:j+1]],
    }


def normalize_packet(row: Dict[str, Any], max_window_tokens: int = 12) -> Dict[str, Any]:
    ptype = str(row.get("packet_type", "")).upper()
    if ptype not in {"UPDATED_USE", "UNCHANGED_DISTRACTOR_USE"}:
        raise ValueError(f"unknown packet_type {ptype}")
    if ptype == "UPDATED_USE":
        answer_state = str(row["new_state"]).strip()
        foil_state = str(row["source_state"]).strip()
        hits = list(row.get("use_new_state_hits") or [])
        answer_state_kind = "new_state"
        signed_margin_name = "U_new_over_source"
    else:
        answer_state = str(row["source_state"]).strip()
        foil_state = str(row["new_state"]).strip()
        hits = list(row.get("use_source_state_hits") or [])
        answer_state_kind = "source_state"
        signed_margin_name = "R_source_over_new"
    use = str(row["use_sentence"]).strip()
    exact = find_case_insensitive_span(use, answer_state)
    if exact is not None:
        a0, a1 = exact
        span_info = {"method": "exact_answer_state", "hit_terms": [stem(x) for x in hits], "covered_hit_terms": [stem(x) for x in hits], "missing_hit_terms": [], "window_token_count": len(word_tokens(use[a0:a1])), "window_tokens": [t["raw"] for t in word_tokens(use[a0:a1])]}
    else:
        a0, a1, span_info = choose_hit_window(use, hits, max_window_tokens)
    answer_surface = use[a0:a1].strip()
    if not answer_surface:
        raise ValueError(f"empty answer surface for {row.get('pair_id')}")
    frame = use[:a0] + "{STATE}" + use[a1:]
    return {
        **row,
        "packet_type": ptype,
        "answer_state_kind": answer_state_kind,
        "answer_text": answer_surface,
        "canonical_answer_state": answer_state,
        "foil_text": foil_state,
        "use_sentence_frame": frame,
        "answer_surface_char_span_in_use": [int(a0), int(a1)],
        "answer_span_info": span_info,
        "signed_margin_name": signed_margin_name,
        "a01_harness_version": "a02_plainuse_answer_harness_v1",
    }


def full_text_and_span(row: Dict[str, Any], candidate: str) -> Tuple[str, int, int]:
    frame = str(row["use_sentence_frame"])
    if frame.count("{STATE}") != 1:
        raise ValueError(f"bad frame for {row.get('pair_id')}: {frame!r}")
    source = str(row["source_sentence"]).strip()
    update = str(row["update_sentence"]).strip()
    before, after = frame.split("{STATE}")
    prefix = source + " " + update + " " + before
    full = prefix + candidate + after
    return full, len(prefix), len(prefix) + len(candidate)


def locate_span_positions(offsets: Sequence[Tuple[int, int]], start: int, end: int) -> List[int]:
    pos = []
    for i, (a, b) in enumerate(offsets):
        if b <= start or a >= end:
            continue
        if a == b:
            continue
        pos.append(i)
    return pos


def score_candidate(model, tokenizer, row: Dict[str, Any], candidate: str, device: torch.device, seq_length: int) -> Dict[str, Any]:
    full, start, end = full_text_and_span(row, candidate)
    enc = tokenizer(full, add_special_tokens=True, return_offsets_mapping=True, return_tensors="pt", max_length=seq_length, truncation=True, padding="max_length")
    offsets = [(int(a), int(b)) for a, b in enc["offset_mapping"].squeeze(0).tolist()]
    positions = locate_span_positions(offsets, start, end)
    if not positions:
        raise ValueError(f"candidate span unavailable or truncated: pair={row.get('pair_id')} candidate={candidate!r}")
    covered_start = min(offsets[p][0] for p in positions)
    covered_end = max(offsets[p][1] for p in positions)
    if covered_start > start or covered_end < end:
        raise ValueError(f"candidate span partial: pair={row.get('pair_id')} candidate={candidate!r} span={(start,end)} covered={(covered_start,covered_end)}")
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)
    target_ids = input_ids[0, positions].detach().clone()
    masked = input_ids.clone()
    masked[0, positions] = int(tokenizer.mask_token_id)
    with torch.no_grad():
        logits = model(input_ids=masked, attention_mask=attention_mask).logits[0, positions]
        lp = F.log_softmax(logits, dim=-1)
        token_logps = lp[torch.arange(len(positions), device=device), target_ids].detach().cpu().tolist()
        top_ids = logits.argmax(dim=-1).detach().cpu().tolist()
    tids = [int(x) for x in target_ids.detach().cpu().tolist()]
    return {
        "candidate": candidate,
        "char_start": int(start),
        "char_end": int(end),
        "token_positions": [int(p) for p in positions],
        "token_ids": tids,
        "tokens": [tokenizer.convert_ids_to_tokens(t) for t in tids],
        "n_tokens": len(positions),
        "sum_logp": float(sum(token_logps)),
        "mean_logp": float(sum(token_logps) / len(token_logps)),
        "token_logps": [float(x) for x in token_logps],
        "top_tokens": [tokenizer.convert_ids_to_tokens(int(t)) for t in top_ids],
    }


def score_rows(model, tokenizer, rows: Sequence[Dict[str, Any]], device: torch.device, seq_length: int, progress_every: int = 100) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    model.eval()
    scored: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []
    for i, r in enumerate(rows):
        try:
            ans = score_candidate(model, tokenizer, r, str(r["answer_text"]), device, seq_length)
            foil = score_candidate(model, tokenizer, r, str(r["foil_text"]), device, seq_length)
            m = ans["mean_logp"] - foil["mean_logp"]
            s = ans["sum_logp"] - foil["sum_logp"]
            out = dict(r)
            out.update({
                "answer_score": ans,
                "foil_score": foil,
                "margin_mean_answer_minus_foil": float(m),
                "margin_sum_answer_minus_foil": float(s),
                "correct_mean": bool(m > 0),
                "correct_sum": bool(s > 0),
                "same_token_length": ans["n_tokens"] == foil["n_tokens"],
            })
            scored.append(out)
        except Exception as exc:
            errors.append({"index": i, "pair_id": r.get("pair_id"), "packet_type": r.get("packet_type"), "error": repr(exc)})
        if progress_every and (i + 1) % int(progress_every) == 0:
            print(f"scored {i+1}/{len(rows)} rows; errors={len(errors)}", flush=True)
    return scored, errors


class AnswerSpanDataset(Dataset):
    def __init__(self, rows: Sequence[Dict[str, Any]], tokenizer, seq_length: int):
        self.items: List[Dict[str, Any]] = []
        for r in rows:
            full, start, end = full_text_and_span(r, str(r["answer_text"]))
            enc = tokenizer(full, add_special_tokens=True, return_offsets_mapping=True, return_tensors="pt", max_length=seq_length, truncation=True, padding="max_length")
            offsets = [(int(a), int(b)) for a, b in enc["offset_mapping"].squeeze(0).tolist()]
            pos = locate_span_positions(offsets, start, end)
            if not pos:
                raise ValueError(f"answer span unavailable: {r.get('pair_id')}")
            covered_start = min(offsets[p][0] for p in pos)
            covered_end = max(offsets[p][1] for p in pos)
            if covered_start > start or covered_end < end:
                raise ValueError(f"answer span partial: {r.get('pair_id')} span={(start,end)} covered={(covered_start,covered_end)}")
            input_ids = enc["input_ids"].squeeze(0)
            labels = torch.full_like(input_ids, -100)
            masked = input_ids.clone()
            for p in pos:
                labels[p] = input_ids[p]
                masked[p] = int(tokenizer.mask_token_id)
            self.items.append({
                "input_ids": masked,
                "attention_mask": enc["attention_mask"].squeeze(0),
                "labels": labels,
                "n_answer_tokens": len(pos),
                "pair_id": r.get("pair_id"),
                "packet_type": r.get("packet_type"),
            })

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.items[idx]


def collate(batch: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "input_ids": torch.stack([b["input_ids"] for b in batch]),
        "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
        "labels": torch.stack([b["labels"] for b in batch]),
        "n_answer_tokens": torch.tensor([int(b["n_answer_tokens"]) for b in batch], dtype=torch.long),
        "packet_type": [b["packet_type"] for b in batch],
    }


def freeze_to_private_optimizer(model, lr: float, weight_decay: float):
    for _, p in model.named_parameters():
        p.requires_grad_(False)
    private_names = {n for n, _ in model.named_parameters() if ".private_adapter." in n}
    if not private_names:
        raise RuntimeError("No private-adapter parameters found")
    for n, p in model.named_parameters():
        p.requires_grad_(n in private_names)
    up_ids = {id(p) for n, p in model.named_parameters() if ".private_adapter.up." in n}
    normal = [p for n, p in model.named_parameters() if n in private_names and id(p) not in up_ids]
    zero_wd = [p for n, p in model.named_parameters() if n in private_names and id(p) in up_ids]
    opt = torch.optim.AdamW([
        {"params": normal, "weight_decay": float(weight_decay)},
        {"params": zero_wd, "weight_decay": 0.0},
    ], lr=float(lr), betas=(0.9, 0.98), eps=1e-6)
    return opt, {
        "trainable_tensor_count": sum(1 for _, p in model.named_parameters() if p.requires_grad),
        "trainable_param_count": int(sum(p.numel() for _, p in model.named_parameters() if p.requires_grad)),
        "nonprivate_trainable_tensors": sum(1 for n, p in model.named_parameters() if p.requires_grad and n not in private_names),
    }


def setup_writable_hf_cache(out_dir: pathlib.Path, tag: str) -> None:
    hf = out_dir / "hf_cache" / tag
    for k, sub in [("HF_HOME", "home"), ("HF_HUB_CACHE", "hub"), ("HF_DATASETS_CACHE", "datasets"), ("TRANSFORMERS_CACHE", "transformers"), ("HF_MODULES_CACHE", "modules")]:
        p = hf / sub
        p.mkdir(parents=True, exist_ok=True)
        os.environ[k] = str(p.resolve())
    os.environ["TOKENIZERS_PARALLELISM"] = "false"


def compact_identity(model, load_info: Dict[str, Any]) -> Dict[str, Any]:
    ident = h.model_identity(model, load_info)
    return {
        "class": ident.get("class"),
        "module": ident.get("module"),
        "total_params": ident.get("total_params"),
        "private_params": ident.get("private_params"),
        "executed_private_scales": ident.get("executed_private_scales"),
        "load_missing_count": len(ident.get("load_missing") or []),
        "load_unexpected_count": len(ident.get("load_unexpected") or []),
    }


def summarize_normalization(rows: Sequence[Dict[str, Any]], label: str) -> Dict[str, Any]:
    by_method = collections.Counter(str(r.get("answer_span_info", {}).get("method")) for r in rows)
    by_type = collections.Counter(str(r.get("packet_type")) for r in rows)
    width = [int(r.get("answer_span_info", {}).get("window_token_count") or 0) for r in rows]
    missing = sum(1 for r in rows if r.get("answer_span_info", {}).get("missing_hit_terms"))
    return {
        "label": label,
        "n_rows": len(rows),
        "packet_type_counts": dict(by_type),
        "answer_span_methods": dict(by_method),
        "state_hit_terms_missing_in_window_rows": int(missing),
        "answer_window_tokens_mean": finite_mean(width),
        "answer_window_tokens_max": max(width) if width else None,
    }


def summarize_scores(scored: Sequence[Dict[str, Any]], label: str) -> Dict[str, Any]:
    by_type: Dict[str, List[Dict[str, Any]]] = collections.defaultdict(list)
    for r in scored:
        by_type[str(r.get("packet_type"))].append(r)
    out: Dict[str, Any] = {
        "label": label,
        "n_rows": len(scored),
        "n_correct_mean": sum(1 for r in scored if r.get("correct_mean")),
        "accuracy_mean": sum(1 for r in scored if r.get("correct_mean")) / max(1, len(scored)),
        "mean_margin_mean": finite_mean([r["margin_mean_answer_minus_foil"] for r in scored]),
        "mean_margin_sum": finite_mean([r["margin_sum_answer_minus_foil"] for r in scored]),
        "same_token_length_rows": sum(1 for r in scored if r.get("same_token_length")),
        "packet_type": {},
    }
    for ptype, rows in sorted(by_type.items()):
        out["packet_type"][ptype] = {
            "n": len(rows),
            "n_correct_mean": sum(1 for r in rows if r.get("correct_mean")),
            "accuracy_mean": sum(1 for r in rows if r.get("correct_mean")) / max(1, len(rows)),
            "mean_margin_mean": finite_mean([r["margin_mean_answer_minus_foil"] for r in rows]),
            "mean_margin_sum": finite_mean([r["margin_sum_answer_minus_foil"] for r in rows]),
        }
    return out


def load_and_normalize(train_path: pathlib.Path, held_path: pathlib.Path, max_window_tokens: int) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    train_raw = read_jsonl(train_path)
    held_raw = read_jsonl(held_path)
    train = [normalize_packet(r, max_window_tokens) for r in train_raw]
    held = [normalize_packet(r, max_window_tokens) for r in held_raw]
    meta = {
        "status": "A02_PLAINUSE_NORMALIZED",
        "created_utc": now_utc(),
        "inputs": {"train": rel(train_path), "held": rel(held_path)},
        "input_sha256": {"train": sha256_file(train_path), "held": sha256_file(held_path)},
        "normalization_rule": "Exact answer-state substring if present; otherwise smallest use-sentence token window covering validator state-hit terms, scored as answer surface against competing canonical state phrase.",
        "train": summarize_normalization(train, "train"),
        "held": summarize_normalization(held, "held"),
    }
    return train, held, meta


def run_parent_score(args, train: Sequence[Dict[str, Any]], held: Sequence[Dict[str, Any]], out_dir: pathlib.Path, device: torch.device, tokenizer) -> Dict[str, Any]:
    model, load_info = h.load_trusted_model(device, private_scale=float(args.private_scale))
    identity = compact_identity(model, load_info)
    held_scored, held_errors = score_rows(model, tokenizer, held, device, int(args.seq_length), progress_every=100)
    train_subset = list(train[: int(args.train_score_limit)]) if int(args.train_score_limit) > 0 else []
    train_scored, train_errors = score_rows(model, tokenizer, train_subset, device, int(args.seq_length), progress_every=200) if train_subset else ([], [])
    result = {
        "status": "A02_PLAINUSE_PARENT_SCORE",
        "created_utc": now_utc(),
        "model_identity": identity,
        "held_summary": summarize_scores(held_scored, "parent_held"),
        "held_errors": held_errors,
        "train_subset_summary": summarize_scores(train_scored, f"parent_train_first{len(train_scored)}") if train_scored else None,
        "train_subset_errors": train_errors,
    }
    write_jsonl(out_dir / "parent_held_scored_rows.jsonl", held_scored)
    (out_dir / "parent_score_summary.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    return result


def train_one_seed(args, seed: int, train: Sequence[Dict[str, Any]], held: Sequence[Dict[str, Any]], out_dir: pathlib.Path, device: torch.device, tokenizer) -> Dict[str, Any]:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    model, load_info = h.load_trusted_model(device, private_scale=float(args.private_scale))
    identity = compact_identity(model, load_info)
    opt, trainable_info = freeze_to_private_optimizer(model, float(args.lr), float(args.weight_decay))
    if trainable_info["nonprivate_trainable_tensors"] != 0:
        raise RuntimeError(f"non-private trainable tensors present: {trainable_info}")
    ds = AnswerSpanDataset(train, tokenizer, int(args.seq_length))
    gen = torch.Generator()
    gen.manual_seed(seed + 44044)
    loader = DataLoader(ds, batch_size=int(args.batch_size), shuffle=True, collate_fn=collate, generator=gen, num_workers=0)
    trajectory: List[Dict[str, Any]] = []
    cumulative_answer_tokens = 0
    t0 = time.time()

    def evaluate(epoch: int, train_loss: Optional[float]) -> None:
        held_scored, held_errors = score_rows(model, tokenizer, held, device, int(args.seq_length), progress_every=0)
        if held_errors:
            raise RuntimeError(f"held scoring errors: {held_errors[:5]}")
        entry = {
            "epoch": int(epoch),
            "train_loss": train_loss,
            "held": summarize_scores(held_scored, f"seed{seed}_held_e{epoch}"),
        }
        trajectory.append(entry)
        write_jsonl(out_dir / f"held_scored_seed{seed}_e{epoch:03d}.jsonl", held_scored)
        print(json.dumps({"event": "eval", "seed": seed, "epoch": epoch, "train_loss": train_loss, "held_acc": entry["held"]["accuracy_mean"], "updated_acc": entry["held"]["packet_type"].get("UPDATED_USE", {}).get("accuracy_mean"), "unchanged_acc": entry["held"]["packet_type"].get("UNCHANGED_DISTRACTOR_USE", {}).get("accuracy_mean")}, ensure_ascii=False), flush=True)

    evaluate(0, None)
    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        num = 0.0
        den = 0
        for batch in loader:
            inp = batch["input_ids"].to(device)
            attn = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            out = model(input_ids=inp, attention_mask=attn)
            loss = F.cross_entropy(out.logits[labels != -100].view(-1, out.logits.size(-1)), labels[labels != -100].view(-1), reduction="mean")
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], float(args.max_grad_norm))
            opt.step()
            opt.zero_grad(set_to_none=True)
            n = int((labels != -100).sum().item())
            num += float(loss.detach().cpu()) * max(1, n)
            den += n
        cumulative_answer_tokens += den
        if epoch % int(args.eval_every) == 0 or epoch == int(args.epochs):
            evaluate(epoch, num / max(1, den))
    model_dir = out_dir / f"seed_{seed}" / "hf_model" / "final"
    if args.save_model:
        model_dir.mkdir(parents=True, exist_ok=True)
        model.eval()
        model.save_pretrained(str(model_dir), safe_serialization=True)
        tokenizer.save_pretrained(str(model_dir))
    result = {
        "seed": int(seed),
        "model_identity_in_memory": identity,
        "trainable_info": trainable_info,
        "epochs": int(args.epochs),
        "n_train_rows": len(train),
        "train_example_exposures": len(train) * int(args.epochs),
        "cumulative_answer_label_tokens": int(cumulative_answer_tokens),
        "lr": float(args.lr),
        "batch_size": int(args.batch_size),
        "elapsed_sec": time.time() - t0,
        "trajectory": trajectory,
        "final_held": trajectory[-1]["held"],
        "model_dir": rel(model_dir) if args.save_model else None,
    }
    (out_dir / f"seed_{seed}" / "seed_summary.json").parent.mkdir(parents=True, exist_ok=True)
    (out_dir / f"seed_{seed}" / "seed_summary.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    del model, opt
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train", default=str(DEFAULT_TRAIN))
    ap.add_argument("--held", default=str(DEFAULT_HELD))
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT))
    ap.add_argument("--gpu", type=int, default=1)
    ap.add_argument("--seq-length", type=int, default=160)
    ap.add_argument("--private-scale", type=float, default=0.75)
    ap.add_argument("--max-window-tokens", type=int, default=12)
    ap.add_argument("--mode", choices=["normalize", "parent", "train"], default="parent")
    ap.add_argument("--train-score-limit", type=int, default=512)
    ap.add_argument("--seeds", nargs="+", type=int, default=[44040, 44041])
    ap.add_argument("--epochs", type=int, default=9)
    ap.add_argument("--eval-every", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--max-grad-norm", type=float, default=1.0)
    ap.add_argument("--save-model", action="store_true")
    args = ap.parse_args()

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    setup_writable_hf_cache(out_dir, "a02_plainuse")
    train, held, norm_meta = load_and_normalize(pathlib.Path(args.train), pathlib.Path(args.held), int(args.max_window_tokens))
    write_jsonl(out_dir / "normalized_train_rows.jsonl", train)
    write_jsonl(out_dir / "normalized_held_rows.jsonl", held)
    (out_dir / "normalization_summary.json").write_text(json.dumps(norm_meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": norm_meta["status"], "train": norm_meta["train"], "held": norm_meta["held"], "out_dir": rel(out_dir)}, ensure_ascii=False), flush=True)
    if args.mode == "normalize":
        return
    device = torch.device(f"cuda:{args.gpu}" if int(args.gpu) >= 0 and torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(str(h.MODEL_PATH), local_files_only=True, use_fast=True)
    if args.mode == "parent":
        result = run_parent_score(args, train, held, out_dir, device, tokenizer)
        print(json.dumps({"status": result["status"], "held": result["held_summary"], "out": rel(out_dir / "parent_score_summary.json")}, ensure_ascii=False), flush=True)
        return
    seed_results = []
    parent_summary = run_parent_score(args, train, held, out_dir, device, tokenizer)
    for seed in args.seeds:
        seed_results.append(train_one_seed(args, int(seed), train, held, out_dir, device, tokenizer))
    aggregate = {
        "status": "A02_PLAINUSE_ANSWER_TRAIN_DONE",
        "created_utc": now_utc(),
        "normalization": norm_meta,
        "parent_summary": parent_summary,
        "seed_results": seed_results,
        "aggregate": {
            "seeds": [int(s) for s in args.seeds],
            "mean_final_held_accuracy": finite_mean([r["final_held"]["accuracy_mean"] for r in seed_results]),
            "mean_final_updated_accuracy": finite_mean([r["final_held"]["packet_type"].get("UPDATED_USE", {}).get("accuracy_mean", float("nan")) for r in seed_results]),
            "mean_final_unchanged_accuracy": finite_mean([r["final_held"]["packet_type"].get("UNCHANGED_DISTRACTOR_USE", {}).get("accuracy_mean", float("nan")) for r in seed_results]),
        },
    }
    (out_dir / "plainuse_train_summary.json").write_text(json.dumps(aggregate, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": aggregate["status"], "out": rel(out_dir / "plainuse_train_summary.json"), "aggregate": aggregate["aggregate"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
