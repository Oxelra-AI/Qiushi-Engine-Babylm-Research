#!/usr/bin/env python3
"""research: score the state-update intervention with a T/U/N current-state margin.

This script implements the primary held-out readout requested after research.
It treats each generated packet as a current-state query rather than as an
ordinary MLM row.  For each non-training packet it scores candidate state
phrases in a fixed state slot under three contexts:

  T: source sentence + the packet's true update sentence.
  U: source sentence + a deterministic type-compatible swapped update from
     another packet; this measures response to a recent foreign-state phrase,
     not the same relation as a co-established distractor entity.
  N: source sentence only, with the update removed.

Candidate phrases are:
  source_state, original_new_state, and swapped_new_state.

Phrase scores are pseudo-log-likelihoods: one candidate token is masked at a
time while the rest of the phrase and context remain visible; scores are
normalized by the number of candidate tokens.  A content-token variant averages
only candidate-token positions overlapping non-stopword content words.

The outputs preserve raw T/U/N margins for base and intervention side-by-side
and arm-minus-base paired deltas, because N movement alone can reflect generator
surface familiarity rather than update use.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import pathlib
import random
import re
import statistics
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Iterable

import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

ROOT = pathlib.Path.cwd()
PROBE_DIR = ROOT / "experiments/archive/relation_learning/data/state_margin_probe"
DEFAULT_EXTENDED = PROBE_DIR / "state_margin_probe_extended_nontrain.jsonl"
DEFAULT_BALANCED = PROBE_DIR / "state_margin_probe_balanced_heldout.jsonl"
OUT_DEFAULT = ROOT / "experiments/archive/relation_learning/data/state_margin_scores"

BASE_43022 = ROOT / "experiments/archive/frontier_consolidation/training/runs/adapter128_scale1p75_h100M100M_seed43022_official_ladder/hf_model"
BASE_43122 = ROOT / "experiments/archive/frontier_consolidation/training/runs/adapter128_scale1p75_seed43122_dense100M/hf_model"
INT_43022 = ROOT / "experiments/archive/relation_learning/training/runs/state_update_plainuse_seed43022/hf_model"
INT_43122 = ROOT / "experiments/archive/relation_learning/training/runs/state_update_plainuse_seed43122/hf_model"

MODEL_ROOTS = {
    "seed43022_base": {"seed": "43022", "arm": "base", "root": BASE_43022},
    "seed43022_intervention": {"seed": "43022", "arm": "intervention", "root": INT_43022},
    "seed43122_base": {"seed": "43122", "arm": "base", "root": BASE_43122},
    "seed43122_intervention": {"seed": "43122", "arm": "intervention", "root": INT_43122},
}

WORD_RE = re.compile(r"[A-Za-z0-9]+(?:[\u2019'][A-Za-z0-9]+)?")
STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "of", "to", "in", "on", "at", "for", "from", "with",
    "by", "as", "into", "onto", "over", "under", "through", "during", "after", "before", "within",
    "without", "near", "about", "around", "is", "are", "was", "were", "be", "been", "being", "has",
    "have", "had", "do", "does", "did", "this", "that", "these", "those", "his", "her", "their",
    "its", "our", "your", "my", "he", "she", "it", "they", "we", "you", "i", "not", "no", "than",
    "then", "there", "here", "which", "who", "whom", "whose", "what", "where", "when", "why", "how",
    "one", "two", "three", "new", "old", "same", "current", "former", "formerly", "still", "now",
}

@dataclass
class CandidateExample:
    packet_id: str
    probe_set: str
    packet_type: str
    condition: str
    candidate_role: str
    candidate_text: str
    text: str
    candidate_start: int
    candidate_end: int
    slot_mode: str = ""
    slot_original_text: str = ""
    n_tokens: int = 0
    n_content_tokens: int = 0
    skip_reason: str | None = None


def now_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def rel(p: pathlib.Path) -> str:
    try:
        return str(p.relative_to(ROOT))
    except Exception:
        return str(p)


def read_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_json(path: pathlib.Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_csv(path: pathlib.Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        keys: list[str] = []
        seen = set()
        for r in rows:
            for k in r.keys():
                if k not in seen:
                    seen.add(k); keys.append(k)
        fieldnames = keys
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def norm_space(s: Any) -> str:
    return " ".join(str(s or "").strip().split())


def stem(word: str) -> str:
    w = word.lower().strip("\u2019'")
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


def content_word_spans(text: str) -> list[tuple[int, int, str]]:
    spans = []
    for m in WORD_RE.finditer(text):
        raw = m.group(0)
        st = stem(raw)
        if len(st) >= 3 and st not in STOPWORDS:
            spans.append((m.start(), m.end(), st))
    return spans


def se(vals: Iterable[float]) -> float:
    xs = [float(x) for x in vals if math.isfinite(float(x))]
    if len(xs) <= 1:
        return float("nan")
    return statistics.stdev(xs) / math.sqrt(len(xs))


def mean(vals: Iterable[float]) -> float:
    xs = [float(x) for x in vals if math.isfinite(float(x))]
    return float(statistics.mean(xs)) if xs else float("nan")


def pct_pos(vals: Iterable[float]) -> float:
    xs = [float(x) for x in vals if math.isfinite(float(x))]
    return float(sum(1 for x in xs if x > 0) / len(xs)) if xs else float("nan")


def checkpoint_path(root: pathlib.Path, checkpoint: str) -> pathlib.Path:
    if checkpoint in {"final", "root", "hf_model"}:
        return root
    return root / checkpoint


def model_plan(model_names: list[str], checkpoints: list[str]) -> list[dict[str, Any]]:
    plan = []
    for name in model_names:
        info = MODEL_ROOTS[name]
        for ck in checkpoints:
            p = checkpoint_path(info["root"], ck)
            plan.append({
                "model_name": name,
                "seed": info["seed"],
                "arm": info["arm"],
                "checkpoint": ck,
                "path": p,
                "exists": p.exists() and (p / "config.json").exists(),
            })
    return plan


def choose_swaps(rows: list[dict[str, Any]], seed: int = 51051) -> dict[str, dict[str, Any]]:
    """Deterministically choose a same-type, similar-length, similar-new-state-type swapped update.

    For UPDATED_USE, the swapped update belongs to another target entity, so the U
    context is source + a recent update about somebody else.  For DISTRACTOR it
    remains a distractor-style update.  We prefer matching `state_type_new`, then
    matching update length, then a stable seeded shuffle.
    """
    by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_type[str(r.get("packet_type"))].append(r)
    rng = random.Random(seed)
    jitter: dict[str, float] = {str(r.get("pair_id")): rng.random() for r in rows}
    swaps: dict[str, dict[str, Any]] = {}
    for r in rows:
        rid = str(r.get("pair_id"))
        ptype = str(r.get("packet_type"))
        target_terms = set(str(x).lower() for x in r.get("target_entity_terms") or [])
        target_entity_low = str(r.get("target_entity", "")).lower()
        new_type = str(r.get("state_type_new") or r.get("competitor_state_type") or "")
        uw = int(r.get("update_words") or len(norm_space(r.get("update_sentence", "")).split()))
        candidates = []
        for c in by_type.get(ptype, []):
            cid = str(c.get("pair_id"))
            if cid == rid:
                continue
            upd_low = str(c.get("update_sentence", "")).lower()
            # Avoid trivial leakage: swapped update should not mention this target entity.
            if target_entity_low and target_entity_low in upd_low:
                continue
            if target_terms and any(t and re.search(r"\b" + re.escape(t) + r"\b", upd_low) for t in target_terms):
                continue
            c_new_type = str(c.get("state_type_new") or c.get("competitor_state_type") or "")
            cw = int(c.get("update_words") or len(norm_space(c.get("update_sentence", "")).split()))
            type_pen = 0 if c_new_type == new_type else 1
            len_pen = abs(cw - uw)
            cand_token_len_pen = abs(len(norm_space(c.get("new_state", "")).split()) - len(norm_space(r.get("new_state", "")).split()))
            candidates.append((type_pen, len_pen, cand_token_len_pen, jitter[cid], c))
        if not candidates:
            # Fall back to same type only; record can still be interpreted because pair id differs.
            for c in by_type.get(ptype, []):
                if str(c.get("pair_id")) != rid:
                    cw = int(c.get("update_words") or len(norm_space(c.get("update_sentence", "")).split()))
                    candidates.append((9, abs(cw - uw), 9, jitter[str(c.get("pair_id"))], c))
        if not candidates:
            raise RuntimeError(f"No swap candidate for {rid}")
        candidates.sort(key=lambda x: x[:4])
        swaps[rid] = candidates[0][4]
    return swaps


def locate_candidate_span(text: str, candidate: str) -> tuple[int, int, str | None]:
    cand = norm_space(candidate)
    if not cand:
        return -1, -1, "empty_candidate"
    idx = text.lower().find(cand.lower())
    if idx >= 0:
        return idx, idx + len(cand), None
    # Fall back to the longest content word from the candidate.  This keeps
    # phrase-slot scoring usable when the original generated use sentence inflects
    # or paraphrases part of the state, and the token coverage is recorded.
    spans = sorted(content_word_spans(cand), key=lambda x: (-(x[1] - x[0]), x[2]))
    for _a, _b, st in spans:
        m = re.search(r"\b" + re.escape(st) + r"[A-Za-z0-9]*\b", text, flags=re.IGNORECASE)
        if m:
            return m.start(), m.end(), "fallback_content_subspan"
    return -1, -1, "candidate_not_in_text"


def build_context_text(pkt: dict[str, Any], condition: str, swap: dict[str, Any] | None, candidate: str, slot_mode: str) -> tuple[str, int, int, str | None, str]:
    source = norm_space(pkt.get("source_sentence"))
    target = norm_space(pkt.get("target_entity"))
    if not source or not target:
        return "", -1, -1, "missing_source_or_target", ""
    cand = norm_space(candidate)
    if not cand:
        return "", -1, -1, "empty_candidate", ""
    prefix_parts = [source]
    if condition == "T":
        upd = norm_space(pkt.get("update_sentence"))
        if not upd:
            return "", -1, -1, "missing_true_update", ""
        prefix_parts.append(upd)
    elif condition == "U":
        if swap is None:
            return "", -1, -1, "missing_swap", ""
        upd = norm_space(swap.get("update_sentence"))
        if not upd:
            return "", -1, -1, "missing_swapped_update", ""
        prefix_parts.append(upd)
    elif condition == "N":
        pass
    else:
        return "", -1, -1, f"bad_condition_{condition}", ""
    context = " ".join(prefix_parts)
    if slot_mode == "template":
        query_prefix = f"The relevant state of {target} is "
        before = context + " " + query_prefix
        text = before + cand + "."
        return text, len(before), len(before) + len(cand), None, cand
    if slot_mode == "use":
        # Preserve the held-out generated use sentence as the frame and replace the
        # gold/used state span with each candidate.  This is closer to an Entity-style
        # final query than a bare template, but can only be used when a state span is
        # identifiable in the use sentence.
        use = norm_space(pkt.get("use_sentence"))
        if not use:
            return "", -1, -1, "missing_use_sentence", ""
        pkt_type = str(pkt.get("packet_type"))
        slot_text = norm_space(pkt.get("new_state" if pkt_type == "UPDATED_USE" else "source_state"))
        sa, sb, err = locate_candidate_span(use, slot_text)
        if err == "candidate_not_in_text":
            return "", -1, -1, "use_slot_not_identified", slot_text
        replacement_use = use[:sa] + cand + use[sb:]
        before = context + " " + replacement_use[:sa]
        text = context + " " + replacement_use
        return text, len(before), len(before) + len(cand), err, slot_text
    return "", -1, -1, f"bad_slot_mode_{slot_mode}", ""


def build_examples(rows: list[dict[str, Any]], probe_set: str, slot_modes: list[str]) -> tuple[list[CandidateExample], dict[str, Any]]:
    swaps = choose_swaps(rows)
    examples: list[CandidateExample] = []
    swap_rows = []
    for pkt in rows:
        rid = str(pkt.get("pair_id"))
        swap = swaps[rid]
        candidates = {
            "source_state": norm_space(pkt.get("source_state")),
            "original_new_state": norm_space(pkt.get("new_state")),
            "swapped_new_state": norm_space(swap.get("new_state")),
        }
        swap_rows.append({
            "probe_set": probe_set,
            "pair_id": rid,
            "packet_type": pkt.get("packet_type"),
            "swap_pair_id": swap.get("pair_id"),
            "swap_packet_type": swap.get("packet_type"),
            "swap_update_words": swap.get("update_words"),
            "swap_state_type_new": swap.get("state_type_new"),
            "swap_new_state": swap.get("new_state"),
            "swap_update_sentence": swap.get("update_sentence"),
        })
        for slot_mode in slot_modes:
            for cond in ["T", "U", "N"]:
                for role, cand in candidates.items():
                    text, a, b, err, slot_original = build_context_text(pkt, cond, swap, cand, slot_mode)
                    examples.append(CandidateExample(
                        packet_id=rid,
                        probe_set=probe_set,
                        packet_type=str(pkt.get("packet_type")),
                        condition=cond,
                        candidate_role=role,
                        candidate_text=cand,
                        text=text,
                        candidate_start=a,
                        candidate_end=b,
                        slot_mode=slot_mode,
                        slot_original_text=slot_original,
                        skip_reason=err,
                    ))
    meta = {
        "swap_count": len(swaps),
        "swap_rows": swap_rows,
        "swap_same_new_state_type_rate": mean(1.0 if str(next(r for r in rows if str(r.get('pair_id')) == sr['pair_id']).get('state_type_new')) == str(sr.get('swap_state_type_new')) else 0.0 for sr in swap_rows) if swap_rows else float('nan'),
    }
    return examples, meta


def encode_examples(tokenizer, examples: list[CandidateExample], max_length: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    encoded = []
    stats = Counter()
    for ex in examples:
        if ex.skip_reason:
            stats[ex.skip_reason] += 1
            continue
        enc = tokenizer(ex.text, add_special_tokens=True, truncation=False, return_offsets_mapping=True)
        ids = [int(x) for x in enc["input_ids"]]
        offsets = [(int(a), int(b)) for a, b in enc["offset_mapping"]]
        if len(ids) > max_length:
            stats["over_max_length"] += 1
            continue
        positions = [i for i, (a, b) in enumerate(offsets) if b > ex.candidate_start and a < ex.candidate_end and a != b]
        if not positions:
            stats["no_candidate_token_positions"] += 1
            continue
        content_spans = [(ex.candidate_start + a, ex.candidate_start + b, st) for a, b, st in content_word_spans(ex.candidate_text)]
        content_positions = []
        if content_spans:
            for i, (a, b) in enumerate(offsets):
                if a == b:
                    continue
                if any(b > ca and a < cb for ca, cb, _ in content_spans):
                    content_positions.append(i)
        if not content_positions:
            content_positions = positions[:]
        ex.n_tokens = len(positions)
        ex.n_content_tokens = len(content_positions)
        encoded.append({
            "packet_id": ex.packet_id,
            "probe_set": ex.probe_set,
            "packet_type": ex.packet_type,
            "condition": ex.condition,
            "candidate_role": ex.candidate_role,
            "candidate_text": ex.candidate_text,
            "slot_mode": ex.slot_mode,
            "slot_original_text": ex.slot_original_text,
            "text": ex.text,
            "input_ids": ids,
            "positions": positions,
            "content_positions": content_positions,
            "candidate_start": ex.candidate_start,
            "candidate_end": ex.candidate_end,
            "seq_len": len(ids),
            "n_candidate_tokens": len(positions),
            "n_content_tokens": len(content_positions),
        })
        stats["encoded"] += 1
    return encoded, dict(stats)


@torch.no_grad()
def score_encoded(model, encoded: list[dict[str, Any]], tokenizer, device: torch.device, batch_size: int) -> list[dict[str, Any]]:
    mask_id = int(tokenizer.mask_token_id)
    pad_id = int(tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0)
    tasks = []
    for idx, rec in enumerate(encoded):
        ids = rec["input_ids"]
        for pos in rec["positions"]:
            tasks.append((idx, int(pos), int(ids[pos]), pos in set(rec["content_positions"])))
    full_sums = [0.0 for _ in encoded]
    content_sums = [0.0 for _ in encoded]
    counts = [0 for _ in encoded]
    content_counts = [0 for _ in encoded]
    model.eval()
    for start in range(0, len(tasks), batch_size):
        batch = tasks[start:start + batch_size]
        max_len = max(len(encoded[idx]["input_ids"]) for idx, _, _, _ in batch)
        input_ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
        att = torch.zeros((len(batch), max_len), dtype=torch.long)
        pos_t = []
        target_t = []
        meta = []
        for bi, (idx, pos, target, is_content) in enumerate(batch):
            ids = list(encoded[idx]["input_ids"])
            ids[pos] = mask_id
            input_ids[bi, :len(ids)] = torch.tensor(ids, dtype=torch.long)
            att[bi, :len(ids)] = 1
            pos_t.append(pos); target_t.append(target); meta.append((idx, is_content))
        input_ids = input_ids.to(device)
        att = att.to(device)
        logits = model(input_ids=input_ids, attention_mask=att).logits.float()
        logp = torch.nn.functional.log_softmax(logits, dim=-1)
        pos_tensor = torch.tensor(pos_t, dtype=torch.long, device=device)
        target_tensor = torch.tensor(target_t, dtype=torch.long, device=device)
        vals = logp[torch.arange(len(batch), device=device), pos_tensor, target_tensor].detach().cpu().tolist()
        for (idx, is_content), val in zip(meta, vals):
            full_sums[idx] += float(val); counts[idx] += 1
            if is_content:
                content_sums[idx] += float(val); content_counts[idx] += 1
    rows = []
    for rec, s, cs, n, cn in zip(encoded, full_sums, content_sums, counts, content_counts):
        rr = {k: rec[k] for k in [
            "packet_id", "probe_set", "packet_type", "condition", "candidate_role",
            "candidate_text", "slot_mode", "slot_original_text", "seq_len", "n_candidate_tokens", "n_content_tokens",
        ]}
        rr.update({
            "pll_logprob_sum": s,
            "pll_logprob_per_token": s / max(1, n),
            "pll_nll_per_token": -s / max(1, n),
            "content_logprob_sum": cs,
            "content_logprob_per_token": cs / max(1, cn),
            "content_nll_per_token": -cs / max(1, cn),
            "scored_tokens": n,
            "scored_content_tokens": cn,
        })
        rows.append(rr)
    return rows


def attach_packet_meta(score_rows: list[dict[str, Any]], packets_by_id: dict[str, dict[str, Any]], swap_by_id: dict[str, dict[str, Any]]) -> None:
    for r in score_rows:
        pkt = packets_by_id.get(str(r["packet_id"]), {})
        sw = swap_by_id.get(str(r["packet_id"]), {})
        r.update({
            "gold_state_type": pkt.get("gold_state_type"),
            "competitor_state_type": pkt.get("competitor_state_type"),
            "state_type_source": pkt.get("state_type_source"),
            "state_type_new": pkt.get("state_type_new"),
            "updated_conflict_heuristic": pkt.get("updated_conflict_heuristic"),
            "source_state": pkt.get("source_state"),
            "original_new_state": pkt.get("new_state"),
            "target_entity": pkt.get("target_entity"),
            "swapped_new_state": sw.get("new_state"),
            "swap_pair_id": sw.get("pair_id"),
            "swap_state_type_new": sw.get("state_type_new"),
        })


def pivot_margins(score_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple, dict[str, dict[str, Any]]] = defaultdict(dict)
    for r in score_rows:
        key = (r["model_name"], r["seed"], r["arm"], r["checkpoint"], r["probe_set"], r["packet_id"], r["slot_mode"], r["condition"])
        groups[key][r["candidate_role"]] = r
    out = []
    for key, d in groups.items():
        if not {"source_state", "original_new_state", "swapped_new_state"} <= set(d):
            continue
        model_name, seed, arm, ck, probe_set, pid, slot_mode, cond = key
        src = d["source_state"]; orig = d["original_new_state"]; sw = d["swapped_new_state"]
        pkt_type = src.get("packet_type")
        # higher logprob margin means the first candidate is more supported in the slot.
        row = {
            "model_name": model_name,
            "seed": seed,
            "arm": arm,
            "checkpoint": ck,
            "probe_set": probe_set,
            "packet_id": pid,
            "slot_mode": slot_mode,
            "condition": cond,
            "packet_type": pkt_type,
            "gold_state_type": src.get("gold_state_type"),
            "competitor_state_type": src.get("competitor_state_type"),
            "updated_conflict_heuristic": src.get("updated_conflict_heuristic"),
            "state_type_source": src.get("state_type_source"),
            "state_type_new": src.get("state_type_new"),
            "swap_pair_id": src.get("swap_pair_id"),
            "source_state": src.get("source_state"),
            "original_new_state": src.get("original_new_state"),
            "swapped_new_state": src.get("swapped_new_state"),
        }
        for prefix, field in [("full", "pll_logprob_per_token"), ("content", "content_logprob_per_token")]:
            s_src = float(src[field]); s_orig = float(orig[field]); s_sw = float(sw[field])
            row[f"{prefix}_lp_source"] = s_src
            row[f"{prefix}_lp_original_new"] = s_orig
            row[f"{prefix}_lp_swapped_new"] = s_sw
            row[f"{prefix}_m_orig_new_minus_source"] = s_orig - s_src
            row[f"{prefix}_m_source_minus_orig_new"] = s_src - s_orig
            row[f"{prefix}_m_source_minus_swapped_new"] = s_src - s_sw
            row[f"{prefix}_m_orig_new_minus_swapped_new"] = s_orig - s_sw
            # Packet-specific gold margin against the original competitor.
            if pkt_type == "UPDATED_USE":
                row[f"{prefix}_gold_margin_vs_original_competitor"] = s_orig - s_src
                row[f"{prefix}_retention_margin_vs_swapped_new"] = s_src - s_sw
            else:
                row[f"{prefix}_gold_margin_vs_original_competitor"] = s_src - s_orig
                row[f"{prefix}_retention_margin_vs_swapped_new"] = s_src - s_sw
        out.append(row)
    return out


def summarize_margins(margin_rows: list[dict[str, Any]], out_dir: pathlib.Path) -> list[dict[str, Any]]:
    fields = [
        "full_m_orig_new_minus_source", "full_m_source_minus_orig_new", "full_m_source_minus_swapped_new",
        "full_m_orig_new_minus_swapped_new", "full_gold_margin_vs_original_competitor", "full_retention_margin_vs_swapped_new",
        "content_m_orig_new_minus_source", "content_m_source_minus_orig_new", "content_m_source_minus_swapped_new",
        "content_m_orig_new_minus_swapped_new", "content_gold_margin_vs_original_competitor", "content_retention_margin_vs_swapped_new",
    ]
    group_keys_sets = [
        ["seed", "arm", "checkpoint", "probe_set", "slot_mode", "packet_type", "condition"],
        ["seed", "arm", "checkpoint", "probe_set", "slot_mode", "packet_type", "condition", "updated_conflict_heuristic"],
        ["seed", "arm", "checkpoint", "probe_set", "slot_mode", "packet_type", "condition", "gold_state_type"],
    ]
    rows = []
    for keys in group_keys_sets:
        groups: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
        for r in margin_rows:
            groups[tuple(r.get(k) for k in keys)].append(r)
        for kvals, vals in groups.items():
            rec = {k: v for k, v in zip(keys, kvals)}
            rec["grouping"] = "+".join(keys)
            rec["n_packets"] = len({v["packet_id"] for v in vals})
            for f in fields:
                xs = [float(v[f]) for v in vals if f in v and math.isfinite(float(v[f]))]
                rec[f"mean_{f}"] = mean(xs)
                rec[f"se_{f}"] = se(xs)
                rec[f"fraction_positive_{f}"] = pct_pos(xs)
            rows.append(rec)
    write_csv(out_dir / "raw_TUN_margin_summary.csv", rows)
    return rows


def paired_arm_deltas(margin_rows: list[dict[str, Any]], out_dir: pathlib.Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    # Pair base and intervention within seed/checkpoint/probe/packet/condition.
    idx: dict[tuple, dict[str, dict[str, Any]]] = defaultdict(dict)
    for r in margin_rows:
        key = (r["seed"], r["checkpoint"], r["probe_set"], r["packet_id"], r["slot_mode"], r["condition"])
        idx[key][r["arm"]] = r
    fields = [
        "full_m_orig_new_minus_source", "full_m_source_minus_orig_new", "full_m_source_minus_swapped_new",
        "full_m_orig_new_minus_swapped_new", "full_gold_margin_vs_original_competitor", "full_retention_margin_vs_swapped_new",
        "content_m_orig_new_minus_source", "content_m_source_minus_orig_new", "content_m_source_minus_swapped_new",
        "content_m_orig_new_minus_swapped_new", "content_gold_margin_vs_original_competitor", "content_retention_margin_vs_swapped_new",
    ]
    delta_rows = []
    for key, d in idx.items():
        if not {"base", "intervention"} <= set(d):
            continue
        b = d["base"]; a = d["intervention"]
        rec = {
            "seed": key[0], "checkpoint": key[1], "probe_set": key[2], "packet_id": key[3], "slot_mode": key[4], "condition": key[5],
            "packet_type": a.get("packet_type"),
            "gold_state_type": a.get("gold_state_type"),
            "updated_conflict_heuristic": a.get("updated_conflict_heuristic"),
            "swap_pair_id": a.get("swap_pair_id"),
        }
        for f in fields:
            rec[f"delta_{f}"] = float(a[f]) - float(b[f])
            rec[f"base_{f}"] = float(b[f])
            rec[f"intervention_{f}"] = float(a[f])
        delta_rows.append(rec)
    write_csv(out_dir / "paired_arm_delta_rows.csv", delta_rows)

    # Summaries by condition and packet type.
    group_key_sets = [
        ["seed", "checkpoint", "probe_set", "slot_mode", "packet_type", "condition"],
        ["seed", "checkpoint", "probe_set", "slot_mode", "packet_type", "condition", "updated_conflict_heuristic"],
        ["seed", "checkpoint", "probe_set", "slot_mode", "packet_type", "condition", "gold_state_type"],
    ]
    summary_rows = []
    for keys in group_key_sets:
        groups: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
        for r in delta_rows:
            groups[tuple(r.get(k) for k in keys)].append(r)
        for kvals, vals in groups.items():
            rec = {k: v for k, v in zip(keys, kvals)}
            rec["grouping"] = "+".join(keys)
            rec["n_packets"] = len({v["packet_id"] for v in vals})
            for f in fields:
                col = f"delta_{f}"
                xs = [float(v[col]) for v in vals if math.isfinite(float(v[col]))]
                rec[f"mean_{col}"] = mean(xs)
                rec[f"se_{col}"] = se(xs)
                rec[f"fraction_positive_{col}"] = pct_pos(xs)
            summary_rows.append(rec)
    write_csv(out_dir / "paired_arm_delta_summary.csv", summary_rows)

    # Relation-specific T/U/N composites on the same packets.  For UPDATED_USE,
    # T orig-new>source tests true same-entity update use, while U source>swapped-new
    # tests high-power retention/distractor behavior after an update about another entity.
    # For DISTRACTOR, T gold margin and U retention margin are both source-state retention.
    by_packet: dict[tuple, dict[str, dict[str, Any]]] = defaultdict(dict)
    for r in delta_rows:
        by_packet[(r["seed"], r["checkpoint"], r["probe_set"], r["packet_id"], r["slot_mode"])][r["condition"]] = r
    comp_rows = []
    for key, d in by_packet.items():
        if not {"T", "U", "N"} <= set(d):
            continue
        t, u, n = d["T"], d["U"], d["N"]
        ptype = t.get("packet_type")
        rec = {
            "seed": key[0], "checkpoint": key[1], "probe_set": key[2], "packet_id": key[3], "slot_mode": key[4],
            "packet_type": ptype,
            "gold_state_type": t.get("gold_state_type"),
            "updated_conflict_heuristic": t.get("updated_conflict_heuristic"),
        }
        for prefix in ["full", "content"]:
            if ptype == "UPDATED_USE":
                rec[f"{prefix}_delta_T_update_use"] = t[f"delta_{prefix}_m_orig_new_minus_source"]
                rec[f"{prefix}_delta_U_retention"] = u[f"delta_{prefix}_m_source_minus_swapped_new"]
                rec[f"{prefix}_delta_N_update_prior"] = n[f"delta_{prefix}_m_orig_new_minus_source"]
                rec[f"{prefix}_net_T_minus_U_update_use"] = t[f"delta_{prefix}_m_orig_new_minus_source"] - u[f"delta_{prefix}_m_orig_new_minus_source"]
                rec[f"{prefix}_net_T_minus_N_update_use"] = t[f"delta_{prefix}_m_orig_new_minus_source"] - n[f"delta_{prefix}_m_orig_new_minus_source"]
                rec[f"{prefix}_binding_pair_sum_Tupdate_Uretention"] = 0.5 * (t[f"delta_{prefix}_m_orig_new_minus_source"] + u[f"delta_{prefix}_m_source_minus_swapped_new"])
            else:
                rec[f"{prefix}_delta_T_retention"] = t[f"delta_{prefix}_m_source_minus_orig_new"]
                rec[f"{prefix}_delta_U_retention"] = u[f"delta_{prefix}_m_source_minus_swapped_new"]
                rec[f"{prefix}_delta_N_retention_prior"] = n[f"delta_{prefix}_m_source_minus_orig_new"]
                rec[f"{prefix}_net_T_minus_U_retention"] = t[f"delta_{prefix}_m_source_minus_orig_new"] - u[f"delta_{prefix}_m_source_minus_orig_new"]
                rec[f"{prefix}_net_T_minus_N_retention"] = t[f"delta_{prefix}_m_source_minus_orig_new"] - n[f"delta_{prefix}_m_source_minus_orig_new"]
                rec[f"{prefix}_binding_pair_sum_Tretention_Uretention"] = 0.5 * (t[f"delta_{prefix}_m_source_minus_orig_new"] + u[f"delta_{prefix}_m_source_minus_swapped_new"])
        comp_rows.append(rec)
    write_csv(out_dir / "relation_composite_delta_rows.csv", comp_rows)

    comp_summary = []
    comp_fields = sorted(k for r in comp_rows for k in r if k.startswith("full_") or k.startswith("content_"))
    comp_group_keys = [
        ["seed", "checkpoint", "probe_set", "slot_mode", "packet_type"],
        ["seed", "checkpoint", "probe_set", "slot_mode", "packet_type", "updated_conflict_heuristic"],
        ["seed", "checkpoint", "probe_set", "slot_mode", "packet_type", "gold_state_type"],
    ]
    for keys in comp_group_keys:
        groups: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
        for r in comp_rows:
            groups[tuple(r.get(k) for k in keys)].append(r)
        for kvals, vals in groups.items():
            rec = {k: v for k, v in zip(keys, kvals)}
            rec["grouping"] = "+".join(keys)
            rec["n_packets"] = len({v["packet_id"] for v in vals})
            for f in comp_fields:
                xs = [float(v[f]) for v in vals if f in v and math.isfinite(float(v[f]))]
                rec[f"mean_{f}"] = mean(xs)
                rec[f"se_{f}"] = se(xs)
                rec[f"fraction_positive_{f}"] = pct_pos(xs)
            comp_summary.append(rec)
    write_csv(out_dir / "relation_composite_delta_summary.csv", comp_summary)
    return delta_rows, summary_rows, comp_summary


def write_note(out_dir: pathlib.Path, meta: dict[str, Any], raw_summary: list[dict[str, Any]], delta_summary: list[dict[str, Any]], comp_summary: list[dict[str, Any]]) -> None:
    def find(rows: list[dict[str, Any]], **kw) -> list[dict[str, Any]]:
        res = []
        for r in rows:
            ok = True
            for k, v in kw.items():
                if str(r.get(k)) != str(v):
                    ok = False; break
            if ok:
                res.append(r)
        return res
    def fmt(x: Any) -> str:
        try:
            xf = float(x)
            if math.isnan(xf): return "NA"
            return f"{xf:+.4f}"
        except Exception:
            return "NA"
    lines = []
    lines.append("# research state-margin scorer result\n\n")
    lines.append(f"Created UTC: {meta.get('created_utc')}\n\n")
    lines.append("## Instrument\n\n")
    lines.append("Each packet is scored under T=source+true update, U=source+swapped compatible update, and N=source only. Candidate phrases are source_state, original_new_state, and swapped_new_state. Phrase scores use one-token-at-a-time pseudo-log-likelihood normalized by candidate token count; content-token margins are written separately. The primary `template` slot is an explicit current-state query; optional `use` slot scoring keeps the generated use-sentence frame when the state span is identifiable. Higher margins mean the first named state phrase is more probable in the scored slot.\n\n")
    lines.append("This instrument intentionally preserves raw T/U/N values for base and intervention; a movement under N alone is surface/generator-format movement, not evidence of update use. The small training dose means tenths of a nat are scientifically meaningful; packet-level paired deltas and SEs are the relevant quantities.\n\n")
    lines.append("## Main paired deltas\n\n")
    lines.append("For UPDATED_USE, `delta_T_update_use` is intervention-base movement of original_new_state over source_state under the true update; `delta_U_retention` is source_state over swapped_new_state when the update belongs to another entity. For DISTRACTOR, T and U retention rows compare source_state against original or swapped distractor states.\n\n")
    cols = [
        "seed", "checkpoint", "probe_set", "slot_mode", "packet_type", "n_packets",
        "mean_full_delta_T_update_use", "se_full_delta_T_update_use",
        "mean_full_delta_U_retention", "se_full_delta_U_retention",
        "mean_full_binding_pair_sum_Tupdate_Uretention", "se_full_binding_pair_sum_Tupdate_Uretention",
        "mean_full_delta_T_retention", "se_full_delta_T_retention",
        "mean_full_binding_pair_sum_Tretention_Uretention", "se_full_binding_pair_sum_Tretention_Uretention",
    ]
    lines.append("| seed | ck | set | slot | type | n | T update | se | U retention | se | update+Uret avg | se | T retention | se | retention avg | se |\n")
    lines.append("|---|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
    for r in comp_summary:
        if r.get("grouping") != "seed+checkpoint+probe_set+slot_mode+packet_type":
            continue
        lines.append(
            f"| {r.get('seed')} | {r.get('checkpoint')} | {r.get('probe_set')} | {r.get('slot_mode')} | {r.get('packet_type')} | {r.get('n_packets')} | "
            f"{fmt(r.get('mean_full_delta_T_update_use'))} | {fmt(r.get('se_full_delta_T_update_use'))} | "
            f"{fmt(r.get('mean_full_delta_U_retention'))} | {fmt(r.get('se_full_delta_U_retention'))} | "
            f"{fmt(r.get('mean_full_binding_pair_sum_Tupdate_Uretention'))} | {fmt(r.get('se_full_binding_pair_sum_Tupdate_Uretention'))} | "
            f"{fmt(r.get('mean_full_delta_T_retention'))} | {fmt(r.get('se_full_delta_T_retention'))} | "
            f"{fmt(r.get('mean_full_binding_pair_sum_Tretention_Uretention'))} | {fmt(r.get('se_full_binding_pair_sum_Tretention_Uretention'))} |\n"
        )
    lines.append("\n## Output files\n\n")
    for k, v in meta.get("outputs", {}).items():
        lines.append(f"- {k}: `{v}`\n")
    lines.append("\n## Notes for interpretation\n\n")
    lines.append("Use the extended set for UPDATED power and the balanced set when comparing UPDATED and DISTRACTOR with equal denominators. Report state-type and conflict-stratified rows from `relation_composite_delta_summary.csv` before mapping to official Entity, because the probe distribution is dominated by other relation states rather than Entity-like containment/location.\n")
    (out_dir / "state_margin_score_note.md").write_text("".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--probe", action="append", default=[], help="JSONL probe path; may be repeated. Defaults to extended and balanced sets.")
    ap.add_argument("--out-dir", default=str(OUT_DEFAULT))
    ap.add_argument("--models", nargs="+", default=["seed43022_base", "seed43022_intervention"], choices=sorted(MODEL_ROOTS))
    ap.add_argument("--checkpoints", nargs="+", default=["chck_86M", "final"])
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--max-items", type=int, default=0, help="Use only the first N packets from each probe for a smoke run.")
    ap.add_argument("--slot-modes", nargs="+", default=["template"], choices=["template", "use"], help="State slot surfaces to score. `template` is the primary explicit state query; `use` keeps the generated use-sentence frame when possible.")
    ap.add_argument("--plan-only", action="store_true")
    ap.add_argument("--skip-missing", action="store_true")
    args = ap.parse_args()

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    probe_paths = [pathlib.Path(p) for p in args.probe] if args.probe else [DEFAULT_EXTENDED, DEFAULT_BALANCED]
    model_entries = model_plan(args.models, args.checkpoints)

    plan = {
        "status": "STATE_MARGIN_SCORE_PLAN",
        "created_utc": now_utc(),
        "probe_paths": [rel(p) for p in probe_paths],
        "probe_exists": {rel(p): p.exists() for p in probe_paths},
        "probe_sha256": {rel(p): sha256_file(p) for p in probe_paths if p.exists()},
        "models": [{**m, "path": rel(m["path"])} for m in model_entries],
        "max_items": args.max_items,
        "slot_modes": args.slot_modes,
        "device": args.device,
        "batch_size": args.batch_size,
        "max_length": args.max_length,
        "out_dir": rel(out_dir),
        "measurement": "T/U/N current-state slot; candidates source_state/original_new_state/swapped_new_state; one-token-at-a-time pseudo-log-likelihood normalized by candidate token count. Primary slot_mode=template; optional slot_mode=use keeps the generated use-sentence frame when possible.",
    }
    write_json(out_dir / "score_plan.json", plan)
    if args.plan_only:
        print(json.dumps(plan, indent=2, ensure_ascii=False), flush=True)
        return
    missing_models = [m for m in model_entries if not m["exists"]]
    if missing_models and not args.skip_missing:
        raise FileNotFoundError("Missing model paths: " + "; ".join(rel(m["path"]) for m in missing_models))
    model_entries = [m for m in model_entries if m["exists"]]
    if not model_entries:
        raise RuntimeError("No model entries to score")

    all_packets: list[dict[str, Any]] = []
    probe_packet_counts = {}
    for p in probe_paths:
        rows = read_jsonl(p)
        if args.max_items > 0:
            rows = rows[:args.max_items]
        probe_set = p.stem.replace("state_margin_probe_", "")
        for r in rows:
            rr = dict(r)
            rr["probe_set"] = probe_set
            all_packets.append(rr)
        probe_packet_counts[probe_set] = len(rows)
    # Encode each probe separately so swaps are chosen within that probe population.
    # Deduplicated packet rows can appear in both balanced and extended sets; they are
    # intentionally retained with distinct probe_set labels.
    tokenizer = AutoTokenizer.from_pretrained(str(model_entries[0]["path"]), use_fast=True)
    if tokenizer.mask_token_id is None:
        raise RuntimeError("Tokenizer has no mask token")
    encoded_all = []
    packet_meta: dict[str, dict[str, Any]] = {}
    swap_meta_by_pid_probe: dict[tuple[str, str], dict[str, Any]] = {}
    encoding_stats = {}
    swap_rows_out = []
    for p in probe_paths:
        rows = read_jsonl(p)
        if args.max_items > 0:
            rows = rows[:args.max_items]
        probe_set = p.stem.replace("state_margin_probe_", "")
        examples, smeta = build_examples(rows, probe_set, args.slot_modes)
        enc, estats = encode_examples(tokenizer, examples, args.max_length)
        encoded_all.extend(enc)
        encoding_stats[probe_set] = estats
        # Reconstruct swap metadata for attach using the rows written by build_examples.
        by_id = {str(r.get("pair_id")): r for r in rows}
        for r in rows:
            packet_meta[f"{probe_set}::{r.get('pair_id')}"] = r
        for sr in smeta["swap_rows"]:
            swap_rows_out.append(sr)
            swap_meta_by_pid_probe[(probe_set, str(sr["pair_id"]))] = {
                "pair_id": sr.get("swap_pair_id"),
                "new_state": sr.get("swap_new_state"),
                "state_type_new": sr.get("swap_state_type_new"),
                "update_sentence": sr.get("swap_update_sentence"),
            }
    write_csv(out_dir / "swap_assignments.csv", swap_rows_out)
    write_json(out_dir / "encoding_metadata.json", {
        "status": "STATE_MARGIN_ENCODING_READY",
        "created_utc": now_utc(),
        "probe_packet_counts": probe_packet_counts,
        "encoded_candidate_records": len(encoded_all),
        "encoding_stats": encoding_stats,
        "candidate_records_expected_per_packet_per_slot_mode": 9,
        "slot_modes": args.slot_modes,
        "note": "Each packet contributes T/U/N x source/original-new/swapped-new candidate phrase records before token-wise PLL expansion.",
    })

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    score_rows: list[dict[str, Any]] = []
    started = time.time()
    for mi, m in enumerate(model_entries, start=1):
        print(f"[LOAD {mi}/{len(model_entries)}] {m['model_name']} {m['checkpoint']} {rel(m['path'])}", flush=True)
        model = AutoModelForMaskedLM.from_pretrained(str(m["path"]), torch_dtype=torch.float32).eval().to(device)
        rows = score_encoded(model, encoded_all, tokenizer, device, args.batch_size)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        for r in rows:
            probe_pid = f"{r['probe_set']}::{r['packet_id']}"
            pkt = packet_meta.get(probe_pid, {})
            sw = swap_meta_by_pid_probe.get((r["probe_set"], str(r["packet_id"])), {})
            r.update({
                "model_name": m["model_name"],
                "seed": m["seed"],
                "arm": m["arm"],
                "checkpoint": m["checkpoint"],
                "model_path": rel(m["path"]),
                "gold_state_type": pkt.get("gold_state_type"),
                "competitor_state_type": pkt.get("competitor_state_type"),
                "state_type_source": pkt.get("state_type_source"),
                "state_type_new": pkt.get("state_type_new"),
                "updated_conflict_heuristic": pkt.get("updated_conflict_heuristic"),
                "source_state": pkt.get("source_state"),
                "original_new_state": pkt.get("new_state"),
                "target_entity": pkt.get("target_entity"),
                "swapped_new_state": sw.get("new_state"),
                "swap_pair_id": sw.get("pair_id"),
                "swap_state_type_new": sw.get("state_type_new"),
            })
        score_rows.extend(rows)
        print(f"[DONE] {m['model_name']} {m['checkpoint']} candidate_records={len(rows)}", flush=True)
    score_path = out_dir / "candidate_phrase_score_rows.csv"
    write_csv(score_path, score_rows)
    margin_rows = pivot_margins(score_rows)
    margin_path = out_dir / "raw_TUN_margin_rows.csv"
    write_csv(margin_path, margin_rows)
    raw_summary = summarize_margins(margin_rows, out_dir)
    delta_rows, delta_summary, comp_summary = paired_arm_deltas(margin_rows, out_dir)

    meta = {
        "status": "STATE_MARGIN_SCORE_DONE",
        "created_utc": now_utc(),
        "elapsed_sec": round(time.time() - started, 2),
        "probe_paths": [rel(p) for p in probe_paths],
        "probe_packet_counts": probe_packet_counts,
        "encoded_candidate_records": len(encoded_all),
        "models_scored": [{**m, "path": rel(m["path"])} for m in model_entries],
        "score_rows": len(score_rows),
        "margin_rows": len(margin_rows),
        "paired_delta_rows": len(delta_rows),
        "relation_composite_rows": sum(1 for _ in open(out_dir / "relation_composite_delta_rows.csv", encoding="utf-8")) - 1 if (out_dir / "relation_composite_delta_rows.csv").exists() else 0,
        "outputs": {
            "plan": rel(out_dir / "score_plan.json"),
            "encoding_metadata": rel(out_dir / "encoding_metadata.json"),
            "swap_assignments": rel(out_dir / "swap_assignments.csv"),
            "candidate_phrase_score_rows": rel(score_path),
            "raw_TUN_margin_rows": rel(margin_path),
            "raw_TUN_margin_summary": rel(out_dir / "raw_TUN_margin_summary.csv"),
            "paired_arm_delta_rows": rel(out_dir / "paired_arm_delta_rows.csv"),
            "paired_arm_delta_summary": rel(out_dir / "paired_arm_delta_summary.csv"),
            "relation_composite_delta_rows": rel(out_dir / "relation_composite_delta_rows.csv"),
            "relation_composite_delta_summary": rel(out_dir / "relation_composite_delta_summary.csv"),
            "note": rel(out_dir / "state_margin_score_note.md"),
        },
    }
    write_json(out_dir / "score_metadata.json", meta)
    write_note(out_dir, meta, raw_summary, delta_summary, comp_summary)
    print(json.dumps(meta, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
