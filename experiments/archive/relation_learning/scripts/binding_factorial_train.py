#!/usr/bin/env python3
"""research: matched adapter-phase factorial on deterministic recombination rows.

Scientific purpose
------------------
research built the right natural-packet object for entity-conditioned state use:
for every DISTRACTOR packet the source+update prefix is identical and the queried
entity alone flips the correct answer between source_state and new_state.  A
single answer-only phase would not identify the mechanism: in-format performance
can rise just because the format is practiced.  This script runs matched arms on
the same rows, same base checkpoint, same adapter parameters, same seed, same
batch size, same LR and same number of epochs/updates:

  answer_clean:
      evidence-preserving concentrated credit.  Source/update/query visible;
      answer tokens are replaced by [MASK]; labels only on answer tokens.

  uniform_wwm:
      ordinary MLM-style credit.  15% whole-word masking over the entire row;
      labels on all selected word tokens; no forced answer slot.

  answer_corrupt_update_state:
      concentrated answer credit but the update-state evidence tokens in the
      update sentence are masked in the input.  Labels remain only on answer
      tokens.  This separates concentrated credit from support preservation.

Evaluation uses a margin-based pair instrument, not exact generated strings.
For each held-out DISTRACTOR pair, score the two candidate phrases in the same
final frame by one-token-at-a-time pseudo-log-likelihood:
  - unchanged entity row: source_state must beat new_state
  - updated entity row:   new_state must beat source_state
The gate is the fraction of pairs where both margins have the correct sign.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import pathlib
import random
import re
import statistics
import time
from collections import Counter, defaultdict
from typing import Any, Iterable

# ---------- HF cache setup BEFORE transformers import ----------
ROOT = pathlib.Path.cwd()
CACHE_BASE = pathlib.Path(os.environ.get("CACHE_BASE", str(ROOT / "experiments/archive/relation_learning/data/binding_factorial/hf_cache")))
if not CACHE_BASE.is_absolute():
    CACHE_BASE = ROOT / CACHE_BASE
for _sub in ["hf_home", "hf_home/hub", "transformers", "modules", "datasets", "tmp"]:
    (CACHE_BASE / _sub).mkdir(parents=True, exist_ok=True)
os.environ["HF_HOME"] = str(CACHE_BASE / "hf_home")
os.environ["HF_HUB_CACHE"] = str(CACHE_BASE / "hf_home/hub")
os.environ["HUGGINGFACE_HUB_CACHE"] = str(CACHE_BASE / "hf_home/hub")
os.environ["TRANSFORMERS_CACHE"] = str(CACHE_BASE / "transformers")
os.environ["HF_MODULES_CACHE"] = str(CACHE_BASE / "modules")
os.environ["HF_DATASETS_CACHE"] = str(CACHE_BASE / "datasets")
os.environ["TMPDIR"] = str(CACHE_BASE / "tmp")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
import torch.nn.functional as F
from transformers import AutoModelForMaskedLM, AutoTokenizer

# ---------- Paths ----------
TRAIN_ROWS = ROOT / "experiments/archive/relation_learning/data/recombination_rows/recombination_train.jsonl"
HELDOUT_ROWS = ROOT / "experiments/archive/relation_learning/data/recombination_rows/recombination_heldout.jsonl"
BINDING_PAIRS = ROOT / "experiments/archive/relation_learning/data/recombination_rows/binding_pairs_heldout.jsonl"
BASE_43022_CHCK = ROOT / "experiments/archive/frontier_consolidation/training/runs/adapter128_scale1p75_h100M100M_seed43022_official_ladder/hf_model/chck_100M"
DEFAULT_OUT_ROOT = ROOT / "experiments/archive/relation_learning/data/binding_factorial"

WORD_RE = re.compile(r"[A-Za-z0-9]+(?:[\u2019'][A-Za-z0-9]+)?")
STOP = {
    "the", "a", "an", "and", "or", "but", "of", "to", "in", "on", "at", "for", "from", "with",
    "by", "as", "into", "onto", "over", "under", "through", "during", "after", "before", "within",
    "without", "near", "about", "around", "is", "are", "was", "were", "be", "been", "being", "has",
    "have", "had", "do", "does", "did", "this", "that", "these", "those", "his", "her", "their",
    "its", "our", "your", "my", "he", "she", "it", "they", "we", "you", "i", "not", "no", "than",
    "then", "there", "here", "which", "who", "whom", "whose", "what", "where", "when", "why", "how",
    "one", "two", "three", "new", "old", "same", "current", "former", "formerly", "still", "now",
}


def now_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def rel(p: pathlib.Path) -> str:
    try:
        return str(p.relative_to(ROOT))
    except Exception:
        return str(p)


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_json(path: pathlib.Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_csv(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("\n", encoding="utf-8")
        return
    fields = sorted(set().union(*(r.keys() for r in rows)))
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rows)


def norm_text(s: str | None) -> str:
    return re.sub(r"\s+", " ", str(s or "")).strip()


def word_spans(text: str) -> list[tuple[int, int, str]]:
    return [(m.start(), m.end(), m.group(0)) for m in WORD_RE.finditer(text)]


def stem(tok: str) -> str:
    t = re.sub(r"[^a-z0-9]+", "", tok.lower())
    if len(t) > 5 and t.endswith("ing"):
        t = t[:-3]
    elif len(t) > 4 and t.endswith("ed"):
        t = t[:-2]
    elif len(t) > 4 and t.endswith("es"):
        t = t[:-2]
    elif len(t) > 3 and t.endswith("s"):
        t = t[:-1]
    return t


def content_stems(text: str) -> set[str]:
    out = set()
    for _a, _b, w in word_spans(text):
        sw = stem(w)
        if sw and sw not in STOP and len(sw) > 2:
            out.add(sw)
    return out


def trusted_load_model(model_path: pathlib.Path, device: torch.device) -> tuple[Any, Any, dict[str, Any]]:
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), use_fast=True)
    model = AutoModelForMaskedLM.from_pretrained(
        str(model_path),
        trust_remote_code=True,
        local_files_only=True,
        torch_dtype=torch.float32,
    )
    total = int(sum(p.numel() for p in model.parameters()))
    adapter_p = int(sum(p.numel() for n, p in model.named_parameters() if "adapter" in n.lower()))
    identity = {
        "checkpoint_path": rel(model_path),
        "loaded_class": type(model).__module__ + "." + type(model).__qualname__,
        "total_params_loaded": total,
        "adapter_params_loaded": adapter_p,
        "architectures": getattr(model.config, "architectures", []),
        "model_type": getattr(model.config, "model_type", None),
        "auto_map_present": hasattr(model.config, "auto_map") and bool(getattr(model.config, "auto_map")),
        "adapter_enabled_config": getattr(model.config, "adapter_enabled", None),
        "adapter_scale_config": getattr(model.config, "adapter_scale", None),
        "adapter_bottleneck_config": getattr(model.config, "adapter_bottleneck", None),
        "trust_remote_code": True,
        "local_files_only": True,
    }
    model.to(device)
    return model, tokenizer, identity


def freeze_base_train_adapter(model: Any) -> dict[str, int]:
    total_tensors = adapter_tensors = frozen_tensors = 0
    trainable_params = frozen_params = 0
    for name, param in model.named_parameters():
        total_tensors += 1
        if "adapter" in name.lower():
            param.requires_grad = True
            adapter_tensors += 1
            trainable_params += int(param.numel())
        else:
            param.requires_grad = False
            frozen_tensors += 1
            frozen_params += int(param.numel())
    return {
        "total_parameter_tensors": total_tensors,
        "adapter_trainable_tensors": adapter_tensors,
        "frozen_tensors": frozen_tensors,
        "adapter_trainable_params": trainable_params,
        "frozen_params": frozen_params,
    }


def locate_token_positions(offsets: list[tuple[int, int]], a0: int, b0: int) -> list[int]:
    pos = []
    for i, (a, b) in enumerate(offsets):
        if a == b:
            continue
        if b > a0 and a < b0:
            pos.append(i)
    return pos


def token_word_groups(text: str, offsets: list[tuple[int, int]], seq_len: int) -> list[int]:
    spans = word_spans(text)
    groups = [-1] * seq_len
    j = 0
    for i, (a, b) in enumerate(offsets):
        if a == b:
            continue
        while j < len(spans) and spans[j][1] <= a:
            j += 1
        if j < len(spans) and spans[j][0] < b and spans[j][1] > a:
            groups[i] = j
    return groups


def update_state_token_positions(row: dict[str, Any], offsets: list[tuple[int, int]]) -> tuple[list[int], dict[str, Any]]:
    """Find update-state evidence tokens in the update sentence.

    Exact new_state spans inside update_sentence are used when available.  When the
    generated update uses a paraphrase, corrupt content-word overlaps between
    `new_state` and the update sentence.  The summary records coverage so the arm
    is not misread as uniformly destroying all support.
    """
    context = row["context_text"]
    source = norm_text(row.get("source_sentence"))
    update = norm_text(row.get("update_sentence"))
    new_state = norm_text(row.get("new_state"))
    if not update:
        return [], {"mode": "missing_update", "n_positions": 0, "matched_terms": []}
    source_prefix = source + " "
    update_start = context.find(update, len(source_prefix) - 1)
    if update_start < 0:
        update_start = context.find(update)
    if update_start < 0:
        return [], {"mode": "update_not_found", "n_positions": 0, "matched_terms": []}
    update_end = update_start + len(update)
    # Exact phrase first.
    exact = update.lower().find(new_state.lower()) if new_state else -1
    if exact >= 0:
        a = update_start + exact
        b = a + len(new_state)
        pos = locate_token_positions(offsets, a, b)
        return pos, {"mode": "exact_new_state_phrase", "n_positions": len(pos), "matched_terms": [new_state]}
    # Fallback: corrupt update content words whose crude stems appear in new_state.
    targets = content_stems(new_state)
    matched_terms: list[str] = []
    char_spans: list[tuple[int, int]] = []
    for a, b, w in word_spans(update):
        sw = stem(w)
        if sw in targets:
            matched_terms.append(w)
            char_spans.append((update_start + a, update_start + b))
    positions: list[int] = []
    for a, b in char_spans:
        positions.extend(locate_token_positions(offsets, a, b))
    positions = sorted(set(positions))
    return positions, {"mode": "content_stem_overlap" if positions else "no_update_state_overlap", "n_positions": len(positions), "matched_terms": matched_terms}


def tokenize_row(tokenizer: Any, row: dict[str, Any], max_length: int = 256) -> dict[str, Any] | None:
    context = row["context_text"]
    enc = tokenizer(context, add_special_tokens=True, truncation=True, max_length=max_length, return_offsets_mapping=True)
    ids = [int(x) for x in enc["input_ids"]]
    offsets = [(int(a), int(b)) for a, b in enc["offset_mapping"]]
    answer_positions = locate_token_positions(offsets, int(row["answer_char_start"]), int(row["answer_char_end"]))
    if not answer_positions:
        return None
    update_positions, update_meta = update_state_token_positions(row, offsets)
    return {
        "input_ids": ids,
        "attention_mask": [1] * len(ids),
        "answer_positions": answer_positions,
        "update_state_positions": update_positions,
        "word_groups": token_word_groups(context, offsets, len(ids)),
        "row_id": row["row_id"],
        "pair_id": row["pair_id"],
        "pair_half": row.get("pair_half", ""),
        "packet_type": row.get("packet_type", ""),
        "role": row.get("role", ""),
        "answer_kind": row.get("answer_kind", ""),
        "query_entity": row.get("query_entity", ""),
        "answer_text": row.get("answer_text", ""),
        "source_state": row.get("source_state", ""),
        "new_state": row.get("new_state", ""),
        "update_corruption_meta": update_meta,
    }


def collate(batch: list[dict[str, Any]], pad_id: int) -> dict[str, torch.Tensor]:
    max_len = max(len(b["input_ids"]) for b in batch)
    bsz = len(batch)
    input_ids = torch.full((bsz, max_len), int(pad_id), dtype=torch.long)
    att = torch.zeros((bsz, max_len), dtype=torch.long)
    word_groups = torch.full((bsz, max_len), -1, dtype=torch.long)
    answer_mask = torch.zeros((bsz, max_len), dtype=torch.bool)
    update_state_mask = torch.zeros((bsz, max_len), dtype=torch.bool)
    for i, b in enumerate(batch):
        L = len(b["input_ids"])
        input_ids[i, :L] = torch.tensor(b["input_ids"], dtype=torch.long)
        att[i, :L] = 1
        word_groups[i, :L] = torch.tensor(b["word_groups"], dtype=torch.long)
        for p in b["answer_positions"]:
            if p < max_len:
                answer_mask[i, p] = True
        for p in b["update_state_positions"]:
            if p < max_len:
                update_state_mask[i, p] = True
    return {"input_ids": input_ids, "attention_mask": att, "word_groups": word_groups,
            "answer_mask": answer_mask, "update_state_mask": update_state_mask}


def apply_uniform_wwm(batch_t: dict[str, torch.Tensor], tokenizer: Any, gen: torch.Generator, mask_prob: float) -> tuple[torch.Tensor, torch.Tensor, dict[str, int]]:
    ids = batch_t["input_ids"]
    att = batch_t["attention_mask"].bool()
    wg = batch_t["word_groups"]
    device = ids.device
    selected = torch.zeros_like(ids, dtype=torch.bool)
    for b in range(ids.shape[0]):
        max_gid = int(wg[b].max().item())
        if max_gid < 0:
            continue
        choose = torch.rand(max_gid + 1, generator=gen, device=device) < float(mask_prob)
        for g in torch.nonzero(choose, as_tuple=False).flatten().tolist():
            selected[b] |= (wg[b] == int(g))
    selected &= att
    # Avoid special tokens/pad/mask.
    special = {x for x in [getattr(tokenizer, "cls_token_id", None), getattr(tokenizer, "sep_token_id", None), getattr(tokenizer, "pad_token_id", None), getattr(tokenizer, "mask_token_id", None)] if x is not None}
    if special:
        special_mask = torch.zeros_like(selected)
        for sid in special:
            special_mask |= (ids == int(sid))
        selected &= ~special_mask
    labels = torch.full_like(ids, -100)
    labels[selected] = ids[selected]
    masked = ids.clone()
    if selected.any():
        replace_mask = (torch.rand(ids.shape, generator=gen, device=device) < 0.8) & selected
        masked[replace_mask] = int(tokenizer.mask_token_id)
        rand_mask = (torch.rand(ids.shape, generator=gen, device=device) < 0.5) & selected & ~replace_mask
        if rand_mask.any():
            rand_ids = torch.randint(0, int(tokenizer.vocab_size), (int(rand_mask.sum().item()),), generator=gen, device=device)
            masked[rand_mask] = rand_ids
    stats = {"masked_label_positions": int(selected.sum().item()), "wwm_rows": int(ids.shape[0])}
    return masked, labels, stats


def apply_answer_arm(batch_t: dict[str, torch.Tensor], tokenizer: Any, arm: str) -> tuple[torch.Tensor, torch.Tensor, dict[str, int]]:
    ids = batch_t["input_ids"]
    answer_mask = batch_t["answer_mask"]
    update_state_mask = batch_t["update_state_mask"]
    labels = torch.full_like(ids, -100)
    labels[answer_mask] = ids[answer_mask]
    masked = ids.clone()
    masked[answer_mask] = int(tokenizer.mask_token_id)
    if arm == "answer_corrupt_update_state":
        # Corrupt support evidence in the input only, never as a target label.
        support = update_state_mask & ~answer_mask
        masked[support] = int(tokenizer.mask_token_id)
        n_support = int(support.sum().item())
    else:
        n_support = 0
    stats = {
        "answer_label_positions": int(answer_mask.sum().item()),
        "support_mask_positions": n_support,
    }
    return masked, labels, stats


def batch_train_step(model: Any, optimizer: Any, batch_t: dict[str, torch.Tensor], tokenizer: Any,
                     arm: str, gen: torch.Generator, wwm_prob: float) -> dict[str, float]:
    if arm == "uniform_wwm":
        masked, labels, st = apply_uniform_wwm(batch_t, tokenizer, gen, wwm_prob)
    else:
        masked, labels, st = apply_answer_arm(batch_t, tokenizer, arm)
    if int((labels != -100).sum().item()) == 0:
        return {"loss": float("nan"), "labeled_positions": 0, **st}
    out = model(input_ids=masked, attention_mask=batch_t["attention_mask"], labels=labels)
    loss = out.loss
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    return {"loss": float(loss.item()), "labeled_positions": int((labels != -100).sum().item()), **st}


def make_candidate_context(row: dict[str, Any], candidate: str) -> tuple[str, int, int]:
    prefix = row["context_text"][:int(row["answer_char_start"])]
    cand = norm_text(candidate)
    text = prefix + cand + "."
    return text, len(prefix), len(prefix) + len(cand)


def build_candidate_mask_records(tokenizer: Any, rows: list[dict[str, Any]], max_length: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    records: list[dict[str, Any]] = []
    skips = Counter()
    for row in rows:
        candidates = {"source_state": row.get("source_state", ""), "new_state": row.get("new_state", "")}
        for cand_role, cand in candidates.items():
            text, a0, b0 = make_candidate_context(row, cand)
            enc = tokenizer(text, add_special_tokens=True, truncation=True, max_length=max_length, return_offsets_mapping=True)
            ids = [int(x) for x in enc["input_ids"]]
            offsets = [(int(a), int(b)) for a, b in enc["offset_mapping"]]
            pos = locate_token_positions(offsets, a0, b0)
            if not pos:
                skips["no_candidate_tokens"] += 1
                continue
            for j, p in enumerate(pos):
                masked = list(ids)
                masked[p] = int(tokenizer.mask_token_id)
                records.append({
                    "row_id": row["row_id"],
                    "pair_id": row["pair_id"],
                    "pair_half": row.get("pair_half", ""),
                    "role": row.get("role", ""),
                    "packet_type": row.get("packet_type", ""),
                    "query_entity": row.get("query_entity", ""),
                    "answer_kind": row.get("answer_kind", ""),
                    "candidate_role": cand_role,
                    "candidate_text": cand,
                    "candidate_token_index": j,
                    "candidate_n_tokens": len(pos),
                    "target_token_id": int(ids[p]),
                    "input_ids": masked,
                    "attention_mask": [1] * len(masked),
                })
    return records, {"candidate_mask_records": len(records), "skips": dict(skips)}




@torch.no_grad()
def score_margin_eval(model: Any, tokenizer: Any, heldout_rows: list[dict[str, Any]], binding_pairs: list[dict[str, Any]],
                      device: torch.device, max_length: int, batch_size: int, max_pairs: int = 0) -> dict[str, Any]:
    """Margin-based held-out evaluation for source_state vs new_state candidates."""
    if max_pairs and max_pairs > 0:
        keep_pair_ids = {bp["pair_id"] for bp in binding_pairs[:max_pairs]}
        binding_pairs = [bp for bp in binding_pairs if bp["pair_id"] in keep_pair_ids]
        heldout_rows = [r for r in heldout_rows if r.get("pair_id") in keep_pair_ids or r.get("pair_half") == "single"]
    # Candidate record construction with explicit mask_token_id in every record.
    records: list[dict[str, Any]] = []
    skips = Counter()
    for row in heldout_rows:
        for cand_role, cand in [("source_state", row.get("source_state", "")), ("new_state", row.get("new_state", ""))]:
            text, a0, b0 = make_candidate_context(row, cand)
            enc = tokenizer(text, add_special_tokens=True, truncation=True, max_length=max_length, return_offsets_mapping=True)
            ids0 = [int(x) for x in enc["input_ids"]]
            offsets = [(int(a), int(b)) for a, b in enc["offset_mapping"]]
            pos = locate_token_positions(offsets, a0, b0)
            if not pos:
                skips["no_candidate_tokens"] += 1
                continue
            for j, p in enumerate(pos):
                masked = list(ids0)
                masked[p] = int(tokenizer.mask_token_id)
                records.append({
                    "row_id": row["row_id"], "pair_id": row["pair_id"], "pair_half": row.get("pair_half", ""),
                    "role": row.get("role", ""), "packet_type": row.get("packet_type", ""),
                    "answer_kind": row.get("answer_kind", ""), "query_entity": row.get("query_entity", ""),
                    "candidate_role": cand_role, "candidate_text": cand, "candidate_token_index": j,
                    "candidate_n_tokens": len(pos), "target_token_id": int(ids0[p]),
                    "input_ids": masked, "attention_mask": [1] * len(masked),
                })
    pad_id = int(tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0)
    token_rows: list[dict[str, Any]] = []
    model.eval()
    for start in range(0, len(records), batch_size):
        batch = records[start:start+batch_size]
        max_len = max(len(r["input_ids"]) for r in batch)
        ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long, device=device)
        att = torch.zeros((len(batch), max_len), dtype=torch.long, device=device)
        for i, r in enumerate(batch):
            L = len(r["input_ids"])
            ids[i, :L] = torch.tensor(r["input_ids"], dtype=torch.long, device=device)
            att[i, :L] = 1
        logits = model(input_ids=ids, attention_mask=att).logits.float()
        logp = F.log_softmax(logits, dim=-1)
        for i, r in enumerate(batch):
            mpos = [k for k, tid in enumerate(r["input_ids"]) if tid == int(tokenizer.mask_token_id)]
            if len(mpos) != 1:
                skips[f"bad_mask_count_{len(mpos)}"] += 1
                continue
            lp = float(logp[i, mpos[0], int(r["target_token_id"])].detach().cpu())
            token_rows.append({k: r[k] for k in ["row_id", "pair_id", "pair_half", "role", "packet_type", "answer_kind", "query_entity", "candidate_role", "candidate_text", "candidate_n_tokens"]} | {"candidate_token_index": r["candidate_token_index"], "logprob": lp})
    # Aggregate candidate phrase logprob per token.
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for tr in token_rows:
        grouped[(tr["row_id"], tr["candidate_role"])].append(tr)
    cand_rows: list[dict[str, Any]] = []
    for (rid, cand_role), rs in grouped.items():
        base = rs[0]
        logprob_sum = sum(float(x["logprob"]) for x in rs)
        cand_rows.append({
            "row_id": rid, "pair_id": base["pair_id"], "pair_half": base["pair_half"], "role": base["role"],
            "packet_type": base["packet_type"], "answer_kind": base["answer_kind"], "query_entity": base["query_entity"],
            "candidate_role": cand_role, "candidate_text": base["candidate_text"],
            "candidate_n_tokens": int(base["candidate_n_tokens"]),
            "logprob_sum": logprob_sum,
            "logprob_per_token": logprob_sum / max(1, int(base["candidate_n_tokens"])),
        })
    by_row: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for cr in cand_rows:
        by_row[cr["row_id"]][cr["candidate_role"]] = cr
    row_margins: list[dict[str, Any]] = []
    for rid, d in by_row.items():
        if "source_state" not in d or "new_state" not in d:
            continue
        src, new = d["source_state"], d["new_state"]
        margin_new_minus_source = float(new["logprob_per_token"]) - float(src["logprob_per_token"])
        role = src["role"]
        correct_margin = -margin_new_minus_source if role == "unchanged_entity" else margin_new_minus_source
        row_margins.append({
            "row_id": rid, "pair_id": src["pair_id"], "pair_half": src["pair_half"], "role": role,
            "packet_type": src["packet_type"], "answer_kind": src["answer_kind"], "query_entity": src["query_entity"],
            "source_lp_per_token": float(src["logprob_per_token"]), "new_lp_per_token": float(new["logprob_per_token"]),
            "margin_new_minus_source": margin_new_minus_source,
            "correct_margin": correct_margin,
            "correct_by_margin": int(correct_margin > 0.0),
        })
    by_id = {r["row_id"]: r for r in row_margins}
    pair_rows: list[dict[str, Any]] = []
    for bp in binding_pairs:
        a = by_id.get(bp["row_a_id"]); b = by_id.get(bp["row_b_id"])
        if not a or not b:
            continue
        pair_rows.append({
            "pair_id": bp["pair_id"], "row_a_id": bp["row_a_id"], "row_b_id": bp["row_b_id"],
            "a_correct": int(a["correct_by_margin"]), "b_correct": int(b["correct_by_margin"]),
            "joint_correct": int(a["correct_by_margin"] and b["correct_by_margin"]),
            "a_correct_margin": float(a["correct_margin"]), "b_correct_margin": float(b["correct_margin"]),
            "joint_min_margin": min(float(a["correct_margin"]), float(b["correct_margin"])),
        })
    def avg(xs: Iterable[float]) -> float:
        vals = [float(x) for x in xs if math.isfinite(float(x))]
        return float(statistics.mean(vals)) if vals else float("nan")
    def se(xs: Iterable[float]) -> float:
        vals = [float(x) for x in xs if math.isfinite(float(x))]
        return float(statistics.stdev(vals) / math.sqrt(len(vals))) if len(vals) > 1 else float("nan")
    by_role: dict[str, Any] = {}
    for role, rows in defaultdict(list, {k: [r for r in row_margins if r["role"] == k] for k in sorted(set(r["role"] for r in row_margins))}).items():
        by_role[role] = {
            "n": len(rows),
            "correct_by_margin": int(sum(r["correct_by_margin"] for r in rows)),
            "accuracy_by_margin": float(sum(r["correct_by_margin"] for r in rows) / len(rows)) if rows else float("nan"),
            "mean_correct_margin": avg(r["correct_margin"] for r in rows),
            "se_correct_margin": se(r["correct_margin"] for r in rows),
        }
    result = {
        "status": "MARGIN_EVAL_DONE",
        "created_utc": now_utc(),
        "n_heldout_rows_input": len(heldout_rows),
        "candidate_mask_records": len(records),
        "candidate_phrase_rows": len(cand_rows),
        "row_margin_rows": len(row_margins),
        "binding_pairs_scored": len(pair_rows),
        "binding_joint_correct": int(sum(p["joint_correct"] for p in pair_rows)),
        "binding_joint_accuracy": float(sum(p["joint_correct"] for p in pair_rows) / len(pair_rows)) if pair_rows else float("nan"),
        "binding_mean_joint_min_margin": avg(p["joint_min_margin"] for p in pair_rows),
        "binding_se_joint_min_margin": se(p["joint_min_margin"] for p in pair_rows),
        "a_correct": int(sum(p["a_correct"] for p in pair_rows)),
        "b_correct": int(sum(p["b_correct"] for p in pair_rows)),
        "by_role": by_role,
        "skips": dict(skips),
        "row_margins": row_margins,
        "pair_margins": pair_rows,
    }
    return result


def tokenization_audit(train_toks: list[dict[str, Any]], held_toks: list[dict[str, Any]]) -> dict[str, Any]:
    all_toks = train_toks + held_toks
    modes = Counter(t["update_corruption_meta"].get("mode") for t in all_toks)
    n_update_pos = [len(t["update_state_positions"]) for t in all_toks]
    n_answer = [len(t["answer_positions"]) for t in all_toks]
    return {
        "n_rows_tokenized": len(all_toks),
        "answer_token_positions_mean": float(statistics.mean(n_answer)) if n_answer else float("nan"),
        "answer_token_positions_median": float(statistics.median(n_answer)) if n_answer else float("nan"),
        "update_state_corruption_modes": dict(modes),
        "update_state_positions_mean": float(statistics.mean(n_update_pos)) if n_update_pos else float("nan"),
        "update_state_positions_median": float(statistics.median(n_update_pos)) if n_update_pos else float("nan"),
        "rows_with_update_state_positions": int(sum(1 for x in n_update_pos if x > 0)),
        "fraction_rows_with_update_state_positions": float(sum(1 for x in n_update_pos if x > 0) / len(n_update_pos)) if n_update_pos else float("nan"),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arm", choices=["answer_clean", "uniform_wwm", "answer_corrupt_update_state"], required=True)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--eval-batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=73073)
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--wwm-prob", type=float, default=0.15)
    ap.add_argument("--eval-every", type=int, default=5, help="Evaluate every N epochs; epoch 0 and final are always evaluated")
    ap.add_argument("--max-train-rows", type=int, default=0)
    ap.add_argument("--max-heldout-pairs", type=int, default=0)
    ap.add_argument("--out-root", type=pathlib.Path, default=DEFAULT_OUT_ROOT)
    ap.add_argument("--plan-only", action="store_true")
    ap.add_argument("--eval-only", action="store_true", help="Run margin baseline and exit without training or saving a checkpoint")
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    out_dir = args.out_root / args.arm
    out_dir.mkdir(parents=True, exist_ok=True)

    plan = {
        "status": "BINDING_FACTORIAL_PLAN",
        "created_utc": now_utc(),
        "arm": args.arm,
        "train_rows": rel(TRAIN_ROWS),
        "heldout_rows": rel(HELDOUT_ROWS),
        "binding_pairs": rel(BINDING_PAIRS),
        "base_model": rel(BASE_43022_CHCK),
        "source_sha256": {"train": sha256_file(TRAIN_ROWS), "heldout": sha256_file(HELDOUT_ROWS), "binding_pairs": sha256_file(BINDING_PAIRS)},
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "eval_batch_size": args.eval_batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "device_requested": args.device,
        "seed": args.seed,
        "max_length": args.max_length,
        "wwm_prob": args.wwm_prob,
        "max_train_rows": args.max_train_rows,
        "max_heldout_pairs": args.max_heldout_pairs,
        "out_dir": rel(out_dir),
        "design": {
            "answer_clean": "answer tokens masked and labeled; source/update/query visible",
            "uniform_wwm": "15% whole-word MLM over the same full rows; no forced answer target",
            "answer_corrupt_update_state": "answer tokens masked/labeled; update-state evidence tokens masked in input only",
            "readout": "margin-based source_state vs new_state PLL in the same final frame; joint correct requires both halves of each DISTRACTOR pair to have the correct margin sign",
        },
    }
    write_json(out_dir / "plan.json", plan)
    print(json.dumps(plan, indent=2, ensure_ascii=False), flush=True)
    if args.plan_only:
        return

    train_rows = read_jsonl(TRAIN_ROWS)
    heldout_rows = read_jsonl(HELDOUT_ROWS)
    binding_pairs = read_jsonl(BINDING_PAIRS)
    if args.max_train_rows and args.max_train_rows > 0:
        train_rows = train_rows[:args.max_train_rows]
    if args.max_heldout_pairs and args.max_heldout_pairs > 0:
        keep = {bp["pair_id"] for bp in binding_pairs[:args.max_heldout_pairs]}
        binding_pairs = [bp for bp in binding_pairs if bp["pair_id"] in keep]
        heldout_rows = [r for r in heldout_rows if r.get("pair_id") in keep]

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    print(f"Loading trusted base on {device}", flush=True)
    model, tokenizer, identity = trusted_load_model(BASE_43022_CHCK, device)
    if identity["adapter_params_loaded"] <= 0:
        raise RuntimeError(f"Trusted adapter load failed: {identity}")
    write_json(out_dir / "model_identity.json", identity)
    print(json.dumps({"model_identity": identity}, indent=2), flush=True)

    pad_id = int(tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0)
    train_toks = [x for x in (tokenize_row(tokenizer, r, args.max_length) for r in train_rows) if x is not None]
    held_toks = [x for x in (tokenize_row(tokenizer, r, args.max_length) for r in heldout_rows) if x is not None]
    audit = tokenization_audit(train_toks, held_toks)
    write_json(out_dir / "tokenization_audit.json", audit)
    print(json.dumps({"tokenization_audit": audit}, indent=2), flush=True)

    freeze_stats = freeze_base_train_adapter(model)
    write_json(out_dir / "freeze_stats.json", freeze_stats)
    print(json.dumps({"freeze_stats": freeze_stats}, indent=2), flush=True)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=args.weight_decay)
    gen = torch.Generator(device=device)
    gen.manual_seed(args.seed + 1009)

    # Baseline margin readout with the same model object before any updates.
    base_eval = score_margin_eval(model, tokenizer, heldout_rows, binding_pairs, device, args.max_length, args.eval_batch_size, args.max_heldout_pairs)
    trajectory: list[dict[str, Any]] = [{
        "epoch": 0,
        "updates_seen": 0,
        "eval": {k: v for k, v in base_eval.items() if k not in {"row_margins", "pair_margins"}},
    }]
    write_json(out_dir / "eval_epoch_000.json", base_eval)
    print(json.dumps({"epoch": 0, "eval": trajectory[-1]["eval"]}, indent=2, ensure_ascii=False), flush=True)
    if args.eval_only or args.epochs == 0:
        summary = {
            "status": "BINDING_FACTORIAL_EVAL_ONLY_DONE",
            "created_utc": now_utc(),
            "arm": args.arm,
            "model_identity": identity,
            "freeze_stats": freeze_stats,
            "tokenization_audit": audit,
            "trajectory": trajectory,
            "outputs": {"eval_epoch_000": rel(out_dir / "eval_epoch_000.json")},
        }
        write_json(out_dir / "summary.json", summary)
        print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
        return

    updates_seen = 0
    epoch_summaries: list[dict[str, Any]] = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        random.shuffle(train_toks)
        losses: list[float] = []
        labeled_positions = support_masks = answer_positions = wwm_positions = 0
        t0 = time.time()
        for start in range(0, len(train_toks), args.batch_size):
            batch = train_toks[start:start+args.batch_size]
            batch_t = collate(batch, pad_id)
            batch_t = {k: v.to(device) for k, v in batch_t.items()}
            st = batch_train_step(model, opt, batch_t, tokenizer, args.arm, gen, args.wwm_prob)
            if math.isfinite(float(st.get("loss", float("nan")))):
                losses.append(float(st["loss"]))
            labeled_positions += int(st.get("labeled_positions", 0))
            support_masks += int(st.get("support_mask_positions", 0))
            answer_positions += int(st.get("answer_label_positions", 0))
            wwm_positions += int(st.get("masked_label_positions", 0))
            updates_seen += 1
        epoch_summary = {
            "epoch": epoch,
            "updates_seen": updates_seen,
            "mean_train_loss": float(statistics.mean(losses)) if losses else float("nan"),
            "n_train_batches": len(losses),
            "labeled_positions": labeled_positions,
            "answer_label_positions": answer_positions,
            "support_mask_positions": support_masks,
            "wwm_label_positions": wwm_positions,
            "elapsed_sec": round(time.time() - t0, 2),
        }
        epoch_summaries.append(epoch_summary)
        print(json.dumps({"epoch": epoch, "train": epoch_summary}, indent=2), flush=True)
        if (epoch % args.eval_every == 0) or (epoch == args.epochs):
            ev = score_margin_eval(model, tokenizer, heldout_rows, binding_pairs, device, args.max_length, args.eval_batch_size, args.max_heldout_pairs)
            write_json(out_dir / f"eval_epoch_{epoch:03d}.json", ev)
            trajectory.append({
                "epoch": epoch,
                "updates_seen": updates_seen,
                "train": epoch_summary,
                "eval": {k: v for k, v in ev.items() if k not in {"row_margins", "pair_margins"}},
            })
            print(json.dumps({"epoch": epoch, "eval": trajectory[-1]["eval"]}, indent=2, ensure_ascii=False), flush=True)

    ckpt_dir = out_dir / "checkpoint"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(ckpt_dir), safe_serialization=True)
    tokenizer.save_pretrained(str(ckpt_dir))
    summary = {
        "status": "BINDING_FACTORIAL_TRAIN_DONE",
        "created_utc": now_utc(),
        "arm": args.arm,
        "model_identity": identity,
        "freeze_stats": freeze_stats,
        "tokenization_audit": audit,
        "n_train_tokenized": len(train_toks),
        "n_heldout_tokenized": len(held_toks),
        "n_binding_pairs": len(binding_pairs),
        "updates_seen": updates_seen,
        "epoch_summaries": epoch_summaries,
        "trajectory": trajectory,
        "checkpoint_dir": rel(ckpt_dir),
    }
    write_json(out_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
