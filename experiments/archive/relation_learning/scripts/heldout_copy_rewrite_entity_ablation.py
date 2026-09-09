#!/usr/bin/env python3
"""research: held-out copy/rewrite competence probes and Entity cue ablations.

Scientific purpose
------------------
research established that the VIEW/REPEAT Entity crossover follows whether the queried
state is changed, and that REPEAT has a larger repeated-vs-unrepeated copy gain on the
MAX packets it trained on.  This script closes two remaining holes without new
training:

1. held-out natural copy gain: rebuild source+rotated-copy packets from the research
   held-out rows, which were never in any arm's training stream.  If REPEAT learned a
   natural copy computation, not just its own packet rows, its copy gain should exceed
   CLEAN and VIEW here.
2. held-out rewrite content-conditioning gain: use accepted compact source/rewrite
   pairs that were *not* selected into the MAX training pairs.  Score masked rewrite
   content tokens with the true source present versus an unrelated length-matched
   source.  If VIEW learned to use an earlier span by content, VIEW should gain more
   than CLEAN and REPEAT, especially on rewrite tokens that are not exact source-token
   repeats.
3. Entity cue ablations: on official Entity items whose stale initial state is a
   non-gold option after relevant updates, remove the queried-box initial clause or
   the relevant update sentence(s) and measure gold-vs-stale margin dependence.  The
   copy/content-reading account predicts REPEAT depends more on the stale initial
   clause, while VIEW depends more on relevant updates.

No training, no upload, and no official leaderboard evaluation is performed.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


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
from collections import defaultdict
from typing import Any

import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer


def find_root() -> pathlib.Path:
    return _PUBLIC_ROOT


ROOT = find_root()
WS = ROOT / "experiments/archive" / 'relation_learning'
OUT_DEFAULT = WS / "data" / "heldout_copy_rewrite_entity_ablation"
frontier_consolidation_RUNS = ROOT / "experiments/archive" / 'frontier_consolidation' / "training" / "runs"
relation_learning_RUNS = WS / "training" / "runs"
POOL_DIR = ROOT / "experiments/archive" / 'frontier_consolidation' / "data" / "dose_2p64x_rowholdout_pools"
HELDOUT_ROWS = POOL_DIR / "heldout_cleanqwen_rows.jsonl"
SELECTED_MAX_PAIRS = ROOT / "experiments/archive" / 'frontier_consolidation' / "data" / "dose_distribution_select" / "selected_matched_max_pairs.jsonl"
ALL_ACCEPTED_PAIRS = ROOT / "experiments/archive" / 'frontier_consolidation' / "data" / "expansion_analysis" / "combined_all_accepted_pairs.jsonl"
ENTITY_DIR = ROOT / "experiments/archive" / 'representation_and_objectives' / "data" / "pristine_official_coordinate" / "babylm-eval" / "strict" / "evaluation_data" / "full_eval" / "entity_tracking"

ARM_CONFIGS = {
    "D_V_43022": frontier_consolidation_RUNS / "full_p2c_c2p_abs_view_dose2p64x_matched_rowholdout_deberta100M_seed43022",
    "D_C_43022": frontier_consolidation_RUNS / "full_p2c_c2p_abs_clean_dose2p64x_matched_rowholdout_deberta100M_seed43022",
    "D_R_43022": frontier_consolidation_RUNS / "full_p2c_c2p_abs_repeat_dose2p64x_matched_rowholdout_deberta100M_seed43022",
    "D_V_43122": frontier_consolidation_RUNS / "full_p2c_c2p_abs_view_dose2p64x_matched_rowholdout_deberta100M_seed43122",
    "D_C_43122": frontier_consolidation_RUNS / "full_p2c_c2p_abs_clean_dose2p64x_matched_rowholdout_deberta100M_seed43122",
    "D_R_43122": frontier_consolidation_RUNS / "full_p2c_c2p_abs_repeat_dose2p64x_matched_rowholdout_deberta100M_seed43122",
    "D_V_43222": relation_learning_RUNS / "full_p2c_c2p_abs_view_dose2p64x_matched_rowholdout_deberta100M_seed43222",
    "D_C_43222": relation_learning_RUNS / "full_p2c_c2p_abs_clean_dose2p64x_matched_rowholdout_deberta100M_seed43222",
    "D_R_43222": relation_learning_RUNS / "full_p2c_c2p_abs_repeat_dose2p64x_matched_rowholdout_deberta100M_seed43222",
    "RBT_V_43022": frontier_consolidation_RUNS / "roberta_view_dose2p64x_matched_rowholdout_100M_seed43022",
    "RBT_C_43022": frontier_consolidation_RUNS / "roberta_clean_dose2p64x_matched_rowholdout_100M_seed43022",
    "RBT_R_43022": frontier_consolidation_RUNS / "roberta_repeat_dose2p64x_matched_rowholdout_100M_seed43022",
}
CKS = ["chck_80M", "chck_90M", "chck_100M"]
ROLE_ALIASES = {"V": "V", "C": "C", "R": "R", "VIEW": "V", "CLEAN": "C", "REPEAT": "R"}

STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "if", "then", "than", "that", "this", "these", "those",
    "of", "in", "on", "at", "to", "for", "from", "with", "without", "by", "as", "is", "are", "was",
    "were", "be", "been", "being", "it", "its", "they", "them", "their", "he", "she", "his", "her",
    "we", "you", "i", "not", "no", "do", "does", "did", "can", "could", "would", "should", "will",
}


def rel(p: pathlib.Path) -> str:
    try:
        return str(p.relative_to(ROOT))
    except Exception:
        return str(p)


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def read_jsonl(path: pathlib.Path, limit: int | None = None):
    n = 0
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)
                n += 1
                if limit is not None and n >= limit:
                    return


def parse_arm_name(arm: str) -> tuple[str, str, str]:
    """Return (architecture_label, role V/C/R, seed) for names like D_V_43022 or RBT_REPEAT_43022."""
    parts = str(arm).split("_")
    seed = parts[-1] if parts and parts[-1].isdigit() else "NA"
    role_raw = parts[-2] if len(parts) >= 2 and parts[-1].isdigit() else (parts[1] if len(parts) > 1 else "")
    role = ROLE_ALIASES.get(role_raw.upper(), role_raw.upper()[:1])
    arch = "_".join(parts[:-2]) if len(parts) >= 3 and parts[-1].isdigit() else (parts[0] if parts else "")
    return arch, role, seed


def arm_meta(arm: str) -> dict[str, str]:
    arch, role, seed = parse_arm_name(arm)
    return {"arch": arch, "role": role, "seed": seed}


def wc(text: str) -> int:
    return len(str(text).split())


def mean(xs: list[float]) -> float:
    xs = [float(x) for x in xs if math.isfinite(float(x))]
    return float(statistics.mean(xs)) if xs else float("nan")


def pstdev(xs: list[float]) -> float:
    xs = [float(x) for x in xs if math.isfinite(float(x))]
    return float(statistics.pstdev(xs)) if len(xs) > 1 else 0.0


def norm_text(s: str) -> str:
    return re.sub(r"\s+", " ", str(s).strip().lower()).strip(" .")


def repeat_source_words(source_text: str, n_words: int, salt: str) -> str:
    toks = str(source_text).split()
    if not toks or n_words <= 0:
        return ""
    h = int(hashlib.sha1(salt.encode("utf-8")).hexdigest()[:8], 16)
    start = h % len(toks)
    rot = toks[start:] + toks[:start]
    out: list[str] = []
    while len(out) < n_words:
        out.extend(rot[: n_words - len(out)])
    return " ".join(out)


def start_end_mask(tokenizer) -> tuple[int, int, int]:
    start_tok = tokenizer.cls_token_id if tokenizer.cls_token_id is not None else tokenizer.bos_token_id
    end_tok = tokenizer.sep_token_id if tokenizer.sep_token_id is not None else tokenizer.eos_token_id
    mask_tok = tokenizer.mask_token_id
    if start_tok is None or end_tok is None or mask_tok is None:
        raise RuntimeError("tokenizer lacks start/end/mask token id")
    return int(start_tok), int(end_tok), int(mask_tok)


def wrap_body(tokenizer, body: list[int], positions: list[int], targets: list[int], meta: dict[str, Any], max_len: int) -> dict[str, Any] | None:
    start_tok, end_tok, mask_tok = start_end_mask(tokenizer)
    if len(body) + 2 > max_len:
        return None
    if len(positions) != len(targets) or not positions:
        return None
    ids = [start_tok] + [int(x) for x in body] + [end_tok]
    out_positions = []
    for p in positions:
        q = int(p) + 1
        if q <= 0 or q >= len(ids) - 1:
            return None
        out_positions.append(q)
        ids[q] = mask_tok
    return {"input_ids": ids, "attention_mask": [1] * len(ids), "positions": out_positions, "targets": [int(x) for x in targets], **meta}


def encode_mask_span(tokenizer, text: str, start: int, end: int, meta: dict[str, Any], max_len: int = 512) -> dict[str, Any] | None:
    enc = tokenizer(text, return_offsets_mapping=True, add_special_tokens=True, truncation=True, max_length=max_len)
    ids = list(enc["input_ids"])
    offs = enc["offset_mapping"]
    positions: list[int] = []
    targets: list[int] = []
    for i, (a, b) in enumerate(offs):
        if b <= a:
            continue
        if b > start and a < end:
            positions.append(i)
            targets.append(int(ids[i]))
    if not positions:
        return None
    masked = list(ids)
    mask_id = int(tokenizer.mask_token_id)
    for p in positions:
        masked[p] = mask_id
    return {"input_ids": masked, "attention_mask": [1] * len(masked), "positions": positions, "targets": targets, **meta}


def safe_vocab_ids(tokenizer) -> list[int]:
    special = set(int(x) for x in tokenizer.all_special_ids)
    ids: list[int] = []
    for i in range(len(tokenizer)):
        if i in special:
            continue
        tok = tokenizer.convert_ids_to_tokens(i)
        s = str(tok) if tok is not None else ""
        if not s or s.startswith("[") or s.startswith("<"):
            continue
        ids.append(i)
    if len(ids) < 1000:
        raise RuntimeError(f"too few safe vocab ids: {len(ids)}")
    return ids


def rand_ids(rng: random.Random, vocab_ids: list[int], n: int, forbid: set[int] | None = None) -> list[int]:
    forbid = forbid or set()
    out: list[int] = []
    while len(out) < n:
        x = int(rng.choice(vocab_ids))
        if x not in forbid:
            out.append(x)
    return out


def find_subseq(hay: list[int], needle: list[int]) -> int | None:
    if not needle or len(needle) > len(hay):
        return None
    first = needle[0]
    L = len(needle)
    for i, x in enumerate(hay[: len(hay) - L + 1]):
        if x == first and hay[i : i + L] == needle:
            return i
    return None


def sentence_or_window(text: str, rng: random.Random) -> str | None:
    text = " ".join(str(text).split())
    sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    candidates = [s for s in sents if 10 <= wc(s) <= 55]
    if candidates:
        return rng.choice(candidates)
    toks = text.split()
    if len(toks) < 10:
        return None
    L = min(55, max(20, min(len(toks), 36)))
    if len(toks) <= L:
        return " ".join(toks)
    start = rng.randrange(0, len(toks) - L + 1)
    return " ".join(toks[start : start + L])


def build_heldout_copy_records(tokenizer, vocab_ids: list[int], rng: random.Random, n_per_span: int, span_lengths: list[int], max_len: int, scan_rows: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = list(read_jsonl(HELDOUT_ROWS, scan_rows))
    records: list[dict[str, Any]] = []
    made = {L: 0 for L in span_lengths}
    stats = defaultdict(int)
    for obj in rows:
        if all(made[L] >= n_per_span for L in span_lengths):
            break
        source_text = sentence_or_window(str(obj["text"]), rng)
        if not source_text:
            stats["no_source"] += 1
            continue
        source_words = wc(source_text)
        # Match the compact-rewrite budget approximately while keeping enough copied span.
        companion_words = max(8, min(28, int(round(0.62 * source_words))))
        rep_text = repeat_source_words(source_text, companion_words, str(obj.get("example_id", stats["seen"])))
        src_ids = tokenizer(source_text, add_special_tokens=False)["input_ids"]
        rep_ids = tokenizer(rep_text, add_special_tokens=False)["input_ids"]
        stats["seen"] += 1
        if not src_ids or not rep_ids or len(src_ids) + len(rep_ids) + 2 > max_len:
            stats["too_long_or_empty"] += 1
            continue
        for span_len in span_lengths:
            if made[span_len] >= n_per_span:
                continue
            candidates: list[tuple[int, int]] = []
            for j in range(0, len(rep_ids) - span_len + 1):
                needle = rep_ids[j : j + span_len]
                si = find_subseq(src_ids, needle)
                if si is not None:
                    candidates.append((si, j))
            if not candidates:
                stats[f"no_match_span_{span_len}"] += 1
                continue
            si, j = rng.choice(candidates)
            k = rng.randrange(span_len)
            target = int(rep_ids[j + k])
            src_control = list(src_ids)
            src_control[si : si + span_len] = rand_ids(rng, vocab_ids, span_len, forbid=set(rep_ids[j : j + span_len]))
            body_rep = list(src_ids) + list(rep_ids)
            body_unr = src_control + list(rep_ids)
            mask_pos = len(src_ids) + j + k
            probe_id = f"heldoutcopy:{span_len}:{made[span_len]}:{obj.get('example_id')}"
            for cond, body in [("repeated", body_rep), ("unrepeated", body_unr)]:
                rec = wrap_body(tokenizer, body, [mask_pos], [target], {
                    "probe_family": "heldout_natural_source_repeat",
                    "probe_id": probe_id,
                    "condition": cond,
                    "span_len": span_len,
                    "source_example_id": obj.get("example_id"),
                    "source_domain": obj.get("source"),
                    "source_words": source_words,
                    "companion_words": companion_words,
                    "body_len": len(body),
                }, max_len=max_len)
                if rec is not None:
                    records.append(rec)
            made[span_len] += 1
    stats.update({f"made_span_{L}": made[L] for L in span_lengths})
    return records, dict(stats)


def selected_pair_ids() -> set[str]:
    out: set[str] = set()
    for obj in read_jsonl(SELECTED_MAX_PAIRS):
        pid = str(obj.get("pair_id") or obj.get("prompt_id"))
        out.add(pid)
        out.add(pid.replace("compact:", ""))
    return out


def canonical_pair_id(obj: dict[str, Any]) -> str:
    return str(obj.get("pair_id") or obj.get("prompt_id") or f"sid:{obj.get('sentence_id')}|doc:{obj.get('doc_id')}")


def load_unselected_rewrite_pairs(limit: int | None = None) -> list[dict[str, Any]]:
    selected = selected_pair_ids()
    out: list[dict[str, Any]] = []
    for obj in read_jsonl(ALL_ACCEPTED_PAIRS):
        pid = canonical_pair_id(obj)
        if pid in selected or pid.replace("compact:", "") in selected or f"compact:{pid}" in selected:
            continue
        if "source_text" not in obj or "rewrite_text" not in obj:
            continue
        d = dict(obj)
        d["pair_id"] = pid if pid.startswith("compact:") else f"compact:{pid}"
        out.append(d)
        if limit is not None and len(out) >= limit:
            break
    return out


def token_content_class(piece: str) -> bool:
    word = re.sub(r"[^A-Za-z0-9]+", "", piece).lower()
    return bool(word) and word not in STOPWORDS and len(word) >= 2


def length_match_control_source(source_pool: list[list[int]], idx: int, target_len: int, rng: random.Random, forbid: set[int]) -> list[int]:
    if not source_pool or target_len <= 0:
        return []
    out: list[int] = []
    j = (idx * 7919 + 17) % len(source_pool)
    attempts = 0
    while len(out) < target_len and attempts < len(source_pool) + 5:
        cand = source_pool[j % len(source_pool)]
        if len(cand) > 0:
            if len(cand) >= target_len - len(out):
                if len(cand) > target_len - len(out):
                    start = rng.randrange(0, len(cand) - (target_len - len(out)) + 1)
                    out.extend(cand[start : start + (target_len - len(out))])
                else:
                    out.extend(cand)
            else:
                out.extend(cand)
        j += 104729
        attempts += 1
    if len(out) < target_len:
        out.extend(rand_ids(rng, safe_vocab_cache, target_len - len(out), forbid=forbid))  # replaced after cache init
    out = out[:target_len]
    if forbid:
        repl = None
        for k, x in enumerate(out):
            if x in forbid:
                if repl is None:
                    repl = rand_ids(rng, safe_vocab_cache, 1, forbid=forbid)[0]
                out[k] = repl
    return out


# Global cache used only inside length_match_control_source; assigned in main after tokenizer load.
safe_vocab_cache: list[int] = []


def build_rewrite_conditioning_records(tokenizer, vocab_ids: list[int], rng: random.Random, max_pairs: int | None, tokens_per_class: int, max_len: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    global safe_vocab_cache
    safe_vocab_cache = vocab_ids
    pairs = load_unselected_rewrite_pairs(max_pairs)
    source_pool = [tokenizer(str(p["source_text"]), add_special_tokens=False)["input_ids"] for p in pairs]
    records: list[dict[str, Any]] = []
    stats = defaultdict(int)
    for i, p in enumerate(pairs):
        src_text = " ".join(str(p["source_text"]).split())
        rew_text = " ".join(str(p["rewrite_text"]).split())
        src_ids = tokenizer(src_text, add_special_tokens=False)["input_ids"]
        rew_enc = tokenizer(rew_text, add_special_tokens=False, return_offsets_mapping=True)
        rew_ids = list(rew_enc["input_ids"])
        if not src_ids or not rew_ids or len(src_ids) + len(rew_ids) + 2 > max_len:
            stats["too_long_or_empty"] += 1
            continue
        src_set = set(int(x) for x in src_ids)
        content_pos: list[tuple[int, str]] = []
        for j, (a, b) in enumerate(rew_enc["offset_mapping"]):
            if b <= a:
                continue
            piece = rew_text[a:b]
            if not token_content_class(piece):
                continue
            cls = "overlap" if int(rew_ids[j]) in src_set else "nonoverlap"
            content_pos.append((j, cls))
        by_cls: dict[str, list[int]] = {"overlap": [], "nonoverlap": []}
        for j, cls in content_pos:
            by_cls[cls].append(j)
        chosen: list[tuple[int, str]] = []
        for cls in ["nonoverlap", "overlap"]:
            vals = by_cls[cls]
            if not vals:
                continue
            if len(vals) <= tokens_per_class:
                picks = vals
            else:
                # deterministic spread through the rewrite rather than a prefix-only slice
                picks = [vals[round(k * (len(vals) - 1) / (tokens_per_class - 1))] for k in range(tokens_per_class)] if tokens_per_class > 1 else [vals[len(vals)//2]]
            chosen.extend((j, cls) for j in picks)
        if not chosen:
            stats["no_content_tokens"] += 1
            continue
        for j, cls in chosen:
            target = int(rew_ids[j])
            ctrl_src = length_match_control_source(source_pool, i, len(src_ids), rng, forbid={target})
            body_true = list(src_ids) + rew_ids
            body_ctrl = ctrl_src + rew_ids
            mask_pos = len(src_ids) + j
            probe_id = f"rewritecond:{i}:{j}:{p['pair_id']}"
            common = {
                "probe_family": "heldout_rewrite_content_conditioning",
                "probe_id": probe_id,
                "condition": "",
                "token_class": cls,
                "pair_id": p["pair_id"],
                "sentence_id": p.get("sentence_id"),
                "doc_id": p.get("doc_id"),
                "source_words": int(p.get("source_words", wc(src_text))),
                "rewrite_words": int(p.get("rewrite_words", wc(rew_text))),
                "source_token_len": len(src_ids),
                "rewrite_token_len": len(rew_ids),
                "body_len": len(body_true),
                "target_token_id": target,
            }
            for cond, body in [("true_source", body_true), ("unrelated_source", body_ctrl)]:
                rec = wrap_body(tokenizer, body, [mask_pos], [target], {**common, "condition": cond}, max_len=max_len)
                if rec is not None:
                    records.append(rec)
        stats["pairs_used"] += 1
        stats[f"pairs_with_nonoverlap"] += int(bool(by_cls["nonoverlap"]))
        stats[f"pairs_with_overlap"] += int(bool(by_cls["overlap"]))
    stats["unselected_pairs_loaded"] = len(pairs)
    return records, dict(stats)


def split_entity_prefix(prefix: str) -> tuple[str, list[str], str]:
    parts = [x.strip() for x in re.split(r"(?<=\.)\s+", prefix.strip()) if x.strip()]
    if len(parts) < 2:
        return prefix.strip(), [], ""
    return parts[0], parts[1:-1], parts[-1]


def parse_query_box(query: str) -> int | None:
    m = re.search(r"Box\s+(\d+)\s+contains\s*$", query.strip())
    return int(m.group(1)) if m else None


def parse_initial_clauses(initial_sentence: str) -> list[tuple[int, str]]:
    s = initial_sentence.strip()
    if s.endswith("."):
        s = s[:-1]
    out: list[tuple[int, str]] = []
    for m in re.finditer(r"Box\s+(\d+)\s+contains\s+(.*?)(?=,\s*Box\s+\d+\s+contains\s+|$)", s):
        out.append((int(m.group(1)), m.group(2).strip()))
    return out


def initial_without_qbox(initial_sentence: str, qbox: int | None) -> str:
    clauses = parse_initial_clauses(initial_sentence)
    kept = [(b, c) for b, c in clauses if qbox is None or b != qbox]
    if not kept:
        return ""
    return ", ".join(f"Box {b} contains {c}" for b, c in kept) + "."


def op_boxes(op: str) -> dict[str, int | None]:
    out = {"from_box": None, "to_box": None, "into_box": None, "any_box": None}
    boxes = [int(x) for x in re.findall(r"Box\s+(\d+)", op)]
    if boxes:
        out["any_box"] = boxes[0]
    m = re.search(r"from\s+Box\s+(\d+)", op)
    if m:
        out["from_box"] = int(m.group(1))
    m = re.search(r"to\s+Box\s+(\d+)", op)
    if m:
        out["to_box"] = int(m.group(1))
    m = re.search(r"into\s+Box\s+(\d+)", op)
    if m:
        out["into_box"] = int(m.group(1))
    return out


def op_relevant_to_query(op: str, qbox: int | None) -> bool:
    if qbox is None:
        return False
    b = op_boxes(op)
    low = op.lower().strip()
    if low.startswith("move"):
        return b["from_box"] == qbox or b["to_box"] == qbox
    if low.startswith("remove"):
        return b["from_box"] == qbox
    if low.startswith("put"):
        return b["into_box"] == qbox
    return qbox in {v for v in b.values() if v is not None}


def build_entity_prefix(initial: str, ops: list[str], query: str) -> str:
    parts: list[str] = []
    init = initial.strip()
    if init:
        parts.append(init if init.endswith(".") else init + ".")
    for op in ops:
        o = op.strip()
        if o:
            parts.append(o if o.endswith(".") else o + ".")
    q = query.strip()
    if q:
        parts.append(q)
    prefix = " ".join(parts).strip()
    if prefix and not prefix.endswith(" "):
        prefix += " "
    return prefix


def load_entity_ablation_items() -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for fn in ["regular.jsonl", "ambiref.jsonl", "move_contents.jsonl"]:
        typ = fn[:-6]
        counters: dict[int, int] = defaultdict(int)
        for obj in read_jsonl(ENTITY_DIR / fn):
            if any("nothing" in str(o).lower() for o in obj.get("options", [])):
                continue
            reported = int(obj["numops"])
            item_index = counters[reported]
            counters[reported] += 1
            initial, ops, query = split_entity_prefix(str(obj["input_prefix"]))
            qbox = parse_query_box(query)
            rel_flags = [op_relevant_to_query(op, qbox) for op in ops]
            rel_updates = sum(int(x) for x in rel_flags)
            if rel_updates < 1:
                continue
            clauses = parse_initial_clauses(initial)
            init_map = {b: c for b, c in clauses}
            stale = init_map.get(qbox) if qbox is not None else None
            options = [str(x) for x in obj.get("options", [])]
            if not options or stale is None or norm_text(stale) == norm_text(options[0]):
                continue
            stale_idx = None
            for oi, opt in enumerate(options):
                if norm_text(opt) == norm_text(stale):
                    stale_idx = oi
                    break
            if stale_idx is None:
                continue
            rel_indices = [i for i, x in enumerate(rel_flags) if x]
            last_rel = max(rel_indices)
            prefix_full = build_entity_prefix(initial, ops, query)
            prefix_no_initial = build_entity_prefix(initial_without_qbox(initial, qbox), ops, query)
            ops_no_last = [op for i, op in enumerate(ops) if i != last_rel]
            ops_no_all = [op for i, op in enumerate(ops) if not rel_flags[i]]
            item = {
                "task": "Entity",
                "entity_type": typ,
                "uid": f"{typ}_{reported}_ops",
                "item_index": item_index,
                "sample_id": obj.get("sample_id"),
                "example_id": obj.get("example_id"),
                "reported_numops": reported,
                "relevant_updates": rel_updates,
                "total_ops": len(ops),
                "irrelevant_ops": len(ops) - rel_updates,
                "query_box": qbox if qbox is not None else -1,
                "gold": options[0],
                "stale": options[stale_idx],
                "stale_idx": stale_idx,
                "prefix_words_full": wc(prefix_full),
                "variants": {
                    "full": prefix_full,
                    "no_initial_qbox": prefix_no_initial,
                    "no_last_relevant_update": build_entity_prefix(initial, ops_no_last, query),
                    "no_all_relevant_updates": build_entity_prefix(initial, ops_no_all, query),
                },
            }
            items.append(item)
    return items


def build_entity_ablation_records(tokenizer, max_items: int | None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    items = load_entity_ablation_items()
    if max_items is not None:
        items = items[:max_items]
    records: list[dict[str, Any]] = []
    by_rel = defaultdict(int)
    for i, item in enumerate(items):
        by_rel[int(item["relevant_updates"])] += 1
        for variant, prefix in item["variants"].items():
            for cand_label in ["gold", "stale"]:
                cand = item[cand_label]
                text = prefix + cand
                rec = encode_mask_span(tokenizer, text, len(prefix), len(text), {
                    "probe_family": "entity_gold_stale_cue_ablation",
                    "probe_id": f"entityabl:{i}:{item['uid']}:{item['item_index']}",
                    "variant": variant,
                    "candidate": cand_label,
                    "uid": item["uid"],
                    "entity_type": item["entity_type"],
                    "item_index": item["item_index"],
                    "reported_numops": item["reported_numops"],
                    "relevant_updates": item["relevant_updates"],
                    "total_ops": item["total_ops"],
                    "irrelevant_ops": item["irrelevant_ops"],
                    "prefix_words_full": item["prefix_words_full"],
                    "stale_idx": item["stale_idx"],
                    "query_box": item["query_box"],
                }, max_len=512)
                if rec is not None:
                    records.append(rec)
    return records, {"items": len(items), "by_relevant_updates": dict(sorted(by_rel.items()))}


@torch.no_grad()
def score_records(model, records: list[dict[str, Any]], device: torch.device, pad_id: int, batch_size: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for start in range(0, len(records), batch_size):
        batch = records[start : start + batch_size]
        max_len = max(len(r["input_ids"]) for r in batch)
        ids = torch.full((len(batch), max_len), int(pad_id), dtype=torch.long)
        att = torch.zeros((len(batch), max_len), dtype=torch.long)
        for i, r in enumerate(batch):
            L = len(r["input_ids"])
            ids[i, :L] = torch.tensor(r["input_ids"], dtype=torch.long)
            att[i, :L] = torch.tensor(r["attention_mask"], dtype=torch.long)
        ids = ids.to(device)
        att = att.to(device)
        logits = model(input_ids=ids, attention_mask=att).logits.float()
        logp = torch.nn.functional.log_softmax(logits, dim=-1)
        for i, r in enumerate(batch):
            lp_sum = 0.0
            for pos, target in zip(r["positions"], r["targets"]):
                lp_sum += float(logp[i, int(pos), int(target)].detach().cpu())
            meta = {k: v for k, v in r.items() if k not in {"input_ids", "attention_mask", "positions", "targets"}}
            meta.update({
                "logprob_sum": lp_sum,
                "nll_per_token": -lp_sum / max(1, len(r["positions"])),
                "n_masked_tokens": len(r["positions"]),
                "seq_len": len(r["input_ids"]),
            })
            rows.append(meta)
    return rows


def aggregate_copy(scored: list[dict[str, Any]], arm: str, ck: str) -> list[dict[str, Any]]:
    by: dict[tuple[str, int], dict[str, dict[str, Any]]] = defaultdict(dict)
    for r in scored:
        if r.get("probe_family") != "heldout_natural_source_repeat":
            continue
        by[(str(r["probe_id"]), int(r["span_len"]))][str(r["condition"])] = r
    out: list[dict[str, Any]] = []
    for (pid, span), d in sorted(by.items()):
        if "repeated" not in d or "unrepeated" not in d:
            continue
        rr, ur = d["repeated"], d["unrepeated"]
        out.append({
            "arm": arm, "checkpoint": ck, **arm_meta(arm),
            "probe_family": "heldout_natural_source_repeat", "probe_id": pid, "span_len": span,
            "source_example_id": rr.get("source_example_id"), "source_domain": rr.get("source_domain"),
            "source_words": rr.get("source_words"), "companion_words": rr.get("companion_words"),
            "repeated_nll": rr["nll_per_token"], "unrepeated_nll": ur["nll_per_token"],
            "gain": float(ur["nll_per_token"]) - float(rr["nll_per_token"]),
        })
    return out


def aggregate_rewrite(scored: list[dict[str, Any]], arm: str, ck: str) -> list[dict[str, Any]]:
    by: dict[tuple[str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for r in scored:
        if r.get("probe_family") != "heldout_rewrite_content_conditioning":
            continue
        by[(str(r["probe_id"]), str(r["token_class"]))][str(r["condition"])] = r
    out: list[dict[str, Any]] = []
    for (pid, tokcls), d in sorted(by.items()):
        if "true_source" not in d or "unrelated_source" not in d:
            continue
        tr, ur = d["true_source"], d["unrelated_source"]
        out.append({
            "arm": arm, "checkpoint": ck, **arm_meta(arm),
            "probe_family": "heldout_rewrite_content_conditioning", "probe_id": pid, "token_class": tokcls,
            "pair_id": tr.get("pair_id"), "source_words": tr.get("source_words"), "rewrite_words": tr.get("rewrite_words"),
            "source_token_len": tr.get("source_token_len"), "rewrite_token_len": tr.get("rewrite_token_len"),
            "true_source_nll": tr["nll_per_token"], "unrelated_source_nll": ur["nll_per_token"],
            "gain": float(ur["nll_per_token"]) - float(tr["nll_per_token"]),
        })
    return out


def aggregate_entity(scored: list[dict[str, Any]], arm: str, ck: str) -> list[dict[str, Any]]:
    by: dict[tuple[str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for r in scored:
        if r.get("probe_family") != "entity_gold_stale_cue_ablation":
            continue
        by[(str(r["probe_id"]), str(r["variant"]))][str(r["candidate"])] = r
    margins: dict[str, dict[str, Any]] = defaultdict(dict)
    meta_by_probe: dict[str, dict[str, Any]] = {}
    for (pid, variant), d in by.items():
        if "gold" not in d or "stale" not in d:
            continue
        gr, sr = d["gold"], d["stale"]
        meta_by_probe[pid] = gr
        margins[pid][variant] = float(gr["logprob_sum"]) - float(sr["logprob_sum"])
    out: list[dict[str, Any]] = []
    for pid, md in sorted(margins.items()):
        if "full" not in md:
            continue
        m = meta_by_probe[pid]
        row = {
            "arm": arm, "checkpoint": ck, **arm_meta(arm),
            "probe_family": "entity_gold_stale_cue_ablation", "probe_id": pid,
            "uid": m.get("uid"), "entity_type": m.get("entity_type"), "item_index": m.get("item_index"),
            "reported_numops": m.get("reported_numops"), "relevant_updates": m.get("relevant_updates"),
            "total_ops": m.get("total_ops"), "irrelevant_ops": m.get("irrelevant_ops"),
            "prefix_words_full": m.get("prefix_words_full"), "stale_idx": m.get("stale_idx"), "query_box": m.get("query_box"),
            "margin_full": md.get("full", float("nan")),
            "margin_no_initial_qbox": md.get("no_initial_qbox", float("nan")),
            "margin_no_last_relevant_update": md.get("no_last_relevant_update", float("nan")),
            "margin_no_all_relevant_updates": md.get("no_all_relevant_updates", float("nan")),
        }
        row["effect_no_initial_qbox"] = row["margin_no_initial_qbox"] - row["margin_full"]
        row["effect_no_last_relevant_update"] = row["margin_no_last_relevant_update"] - row["margin_full"]
        row["effect_no_all_relevant_updates"] = row["margin_no_all_relevant_updates"] - row["margin_full"]
        out.append(row)
    return out


def groups_for_copy(r: dict[str, Any]) -> list[str]:
    return ["ALL", f"span_{r['span_len']}", f"domain_{r.get('source_domain','')}"]


def groups_for_rewrite(r: dict[str, Any]) -> list[str]:
    return ["ALL", f"token_{r['token_class']}"]


def groups_for_entity(r: dict[str, Any]) -> list[str]:
    relu = int(r["relevant_updates"])
    groups = ["ALL", f"rel_updates_{relu}", f"type_{r['entity_type']}"]
    if relu >= 2:
        groups.append("rel_ge2")
    if relu >= 3:
        groups.append("rel_ge3")
    return groups


def summarize_gain(rows: list[dict[str, Any]], family: str) -> list[dict[str, Any]]:
    groups: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
    grouper = groups_for_copy if family == "copy" else groups_for_rewrite
    for r in rows:
        for g in grouper(r):
            groups[(r["arm"], r["checkpoint"], g)].append(r)
    out: list[dict[str, Any]] = []
    for (arm, ck, g), vals in sorted(groups.items()):
        gains = [float(v["gain"]) for v in vals]
        out.append({
            "family": family, "arm": arm, "checkpoint": ck, **arm_meta(arm),
            "group": g, "n": len(vals), "mean_gain": mean(gains), "sd_gain": pstdev(gains),
            "se_gain": pstdev(gains) / math.sqrt(len(gains)) if len(gains) > 1 else float("nan"),
            "mean_a_nll": mean([float(v.get("repeated_nll", v.get("true_source_nll", float("nan")))) for v in vals]),
            "mean_b_nll": mean([float(v.get("unrepeated_nll", v.get("unrelated_source_nll", float("nan")))) for v in vals]),
        })
    return out


def summarize_entity(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        for g in groups_for_entity(r):
            groups[(r["arm"], r["checkpoint"], g)].append(r)
    out: list[dict[str, Any]] = []
    keys = ["margin_full", "effect_no_initial_qbox", "effect_no_last_relevant_update", "effect_no_all_relevant_updates", "margin_no_initial_qbox", "margin_no_last_relevant_update", "margin_no_all_relevant_updates"]
    for (arm, ck, g), vals in sorted(groups.items()):
        row = {"family": "entity_ablation", "arm": arm, "checkpoint": ck, **arm_meta(arm), "group": g, "n": len(vals)}
        for k in keys:
            row[f"mean_{k}"] = mean([float(v[k]) for v in vals])
        row["mean_relevant_updates"] = mean([float(v["relevant_updates"]) for v in vals])
        row["mean_total_ops"] = mean([float(v["total_ops"]) for v in vals])
        out.append(row)
    return out


def contrast_summaries(summary: list[dict[str, Any]], value_prefixes: list[str], family: str) -> list[dict[str, Any]]:
    idx = {(r["arm"], r["checkpoint"], r["group"]): r for r in summary}
    groups = sorted({r["group"] for r in summary})
    checkpoints = sorted({r["checkpoint"] for r in summary})
    by_arch_seed_role: dict[tuple[str, str, str], str] = {}
    for r in summary:
        role = str(r.get("role", ""))
        if role in {"V", "C", "R"}:
            by_arch_seed_role[(str(r.get("arch", "")), str(r.get("seed", "")), role)] = str(r["arm"])
    arch_seeds = sorted({(a, s) for (a, s, _role) in by_arch_seed_role})
    out: list[dict[str, Any]] = []
    for arch, seed in arch_seeds:
        arms = {role: by_arch_seed_role.get((arch, seed, role)) for role in ["V", "C", "R"]}
        for ck in checkpoints:
            for g in groups:
                for cname, a, b in [("RminusC", "R", "C"), ("RminusV", "R", "V"), ("VminusC", "V", "C"), ("VminusR", "V", "R"), ("CminusR", "C", "R")]:
                    if not arms.get(a) or not arms.get(b):
                        continue
                    ka = (arms[a], ck, g); kb = (arms[b], ck, g)
                    if ka not in idx or kb not in idx:
                        continue
                    ra, rb = idx[ka], idx[kb]
                    row = {"family": family, "arch": arch, "seed": seed, "checkpoint": ck, "group": g, "contrast": cname, "arm_a": arms[a], "arm_b": arms[b], "n": int(ra["n"])}
                    for vp in value_prefixes:
                        row[f"delta_{vp}_a_minus_b"] = float(ra[vp]) - float(rb[vp])
                        row[f"{vp}_a"] = float(ra[vp])
                        row[f"{vp}_b"] = float(rb[vp])
                    out.append(row)
    return out


def late_contrasts(con_rows: list[dict[str, Any]], value_delta_keys: list[str], family: str) -> list[dict[str, Any]]:
    groups: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
    for r in con_rows:
        groups[(r.get("arch", ""), r["seed"], r["group"], r["contrast"])].append(r)
    out: list[dict[str, Any]] = []
    for (arch, seed, g, c), vals in sorted(groups.items()):
        row = {"family": family, "arch": arch, "seed": seed, "group": g, "contrast": c, "n_checkpoints": len(vals), "n": int(vals[0]["n"])}
        for k in value_delta_keys:
            row[f"late_mean_{k}"] = mean([float(v[k]) for v in vals])
            # Preserve decomposed NLL terms for interpreting whether a gain contrast comes from
            # the cued/intact side or the corrupted/unrelated denominator.
            base = k.replace("delta_", "")
            a_key = f"{base}_a"; b_key = f"{base}_b"
            if a_key in vals[0]:
                row[f"late_mean_{a_key}"] = mean([float(v[a_key]) for v in vals])
            if b_key in vals[0]:
                row[f"late_mean_{b_key}"] = mean([float(v[b_key]) for v in vals])
        out.append(row)
    return out


def write_csv(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("\n", encoding="utf-8")
        return
    fields = sorted(set().union(*(r.keys() for r in rows)))
    preferred = [
        "family", "arm", "checkpoint", "seed", "role", "group", "contrast", "n", "n_checkpoints",
        "mean_gain", "delta_mean_gain_a_minus_b", "late_mean_delta_mean_gain_a_minus_b",
        "mean_margin_full", "delta_mean_margin_full_a_minus_b", "late_mean_delta_mean_margin_full_a_minus_b",
        "mean_effect_no_initial_qbox", "delta_mean_effect_no_initial_qbox_a_minus_b", "late_mean_delta_mean_effect_no_initial_qbox_a_minus_b",
        "mean_effect_no_last_relevant_update", "delta_mean_effect_no_last_relevant_update_a_minus_b", "late_mean_delta_mean_effect_no_last_relevant_update_a_minus_b",
        "mean_effect_no_all_relevant_updates", "delta_mean_effect_no_all_relevant_updates_a_minus_b", "late_mean_delta_mean_effect_no_all_relevant_updates_a_minus_b",
        "probe_family", "probe_id", "span_len", "token_class", "gain",
    ]
    preferred = preferred[:1] + ["arch", "arm_a", "arm_b"] + preferred[1:]
    fields = [f for f in preferred if f in fields] + [f for f in fields if f not in preferred]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rows)


def make_note(out_dir: pathlib.Path, plan: dict[str, Any], copy_late: list[dict[str, Any]], rewrite_late: list[dict[str, Any]], entity_late: list[dict[str, Any]]) -> None:
    def pick(rows, group, contrast, key):
        return [r for r in rows if r.get("group") == group and r.get("contrast") == contrast and key in r]
    lines: list[str] = []
    lines.append("# research held-out copy, rewrite-conditioning, and Entity cue-ablation probes")
    lines.append("")
    lines.append("## Record counts")
    lines.append("")
    lines.append(f"- Held-out natural copy records: {plan['record_counts']['heldout_natural_source_repeat']} from `{rel(HELDOUT_ROWS)}`.")
    lines.append(f"- Held-out rewrite-conditioning records: {plan['record_counts']['heldout_rewrite_content_conditioning']} from unselected accepted pairs in `{rel(ALL_ACCEPTED_PAIRS)}` after excluding `{rel(SELECTED_MAX_PAIRS)}`.")
    lines.append(f"- Entity cue-ablation records: {plan['record_counts']['entity_gold_stale_cue_ablation']} on stale-non-gold relevant-update Entity items.")
    lines.append("")
    lines.append("## Key late contrasts")
    lines.append("")
    lines.append("For copy and rewrite probes, positive gain contrast means the first arm benefits more from the relevant context cue. For Entity cue ablations, a positive `effect_no_initial` means removing the queried-box initial clause raises the gold-over-stale margin; a negative `effect_no_last/all_relevant` means removing update evidence lowers the gold-over-stale margin.")
    lines.append("")
    lines.append("### Held-out natural copy gain")
    lines.append("")
    lines.append("| seed | contrast | late delta gain | n |")
    lines.append("|---:|---|---:|---:|")
    for r in copy_late:
        if r["group"] == "ALL" and r["contrast"] in {"RminusC", "RminusV", "VminusC"}:
            lines.append(f"| {r['seed']} | {r['contrast']} | {r['late_mean_delta_mean_gain_a_minus_b']:+.4f} | {r['n']} |")
    lines.append("")
    lines.append("### Held-out rewrite content-conditioning gain")
    lines.append("")
    lines.append("| seed | group | contrast | late delta gain | n |")
    lines.append("|---:|---|---|---:|---:|")
    for r in rewrite_late:
        if r["group"] in {"ALL", "token_nonoverlap", "token_overlap"} and r["contrast"] in {"VminusC", "VminusR", "CminusR"}:
            lines.append(f"| {r['seed']} | {r['group']} | {r['contrast']} | {r['late_mean_delta_mean_gain_a_minus_b']:+.4f} | {r['n']} |")
    lines.append("")
    lines.append("### Entity gold-vs-stale cue ablations")
    lines.append("")
    lines.append("| seed | group | contrast | d effect no-initial | d effect no-last-update | d effect no-all-updates | n |")
    lines.append("|---:|---|---|---:|---:|---:|---:|")
    for r in entity_late:
        if r["group"] in {"ALL", "rel_ge2", "rel_ge3"} and r["contrast"] in {"RminusV", "VminusR", "VminusC", "RminusC"}:
            lines.append(f"| {r['seed']} | {r['group']} | {r['contrast']} | {r['late_mean_delta_mean_effect_no_initial_qbox_a_minus_b']:+.4f} | {r['late_mean_delta_mean_effect_no_last_relevant_update_a_minus_b']:+.4f} | {r['late_mean_delta_mean_effect_no_all_relevant_updates_a_minus_b']:+.4f} | {r['n']} |")
    lines.append("")
    lines.append("## Scientific reading")
    lines.append("")
    lines.append("This note is generated mechanically from the research CSV outputs. The important interpretation is not any single NLL level but whether the held-out context gains and the causal Entity ablation effects separate the arms in the same direction as the official relevant-update crossover.")
    (out_dir / "probe_readout.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arms", nargs="+", default=list(ARM_CONFIGS))
    ap.add_argument("--checkpoints", nargs="+", default=CKS)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=192)
    ap.add_argument("--copy-n-per-span", type=int, default=1000)
    ap.add_argument("--copy-span-lengths", nargs="+", type=int, default=[1, 4])
    ap.add_argument("--copy-scan-rows", type=int, default=6992)
    ap.add_argument("--rewrite-max-pairs", type=int, default=0, help="0 means use all unselected pairs")
    ap.add_argument("--rewrite-tokens-per-class", type=int, default=2)
    ap.add_argument("--entity-max-items", type=int, default=0, help="0 means use all eligible items")
    ap.add_argument("--max-len-copy-rewrite", type=int, default=256)
    ap.add_argument("--seed", type=int, default=909009)
    ap.add_argument("--out-dir", default=str(OUT_DEFAULT))
    ap.add_argument("--plan-only", action="store_true")
    args = ap.parse_args()

    out_dir = pathlib.Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    for arm in args.arms:
        for ck in args.checkpoints:
            mp = ARM_CONFIGS[arm] / "hf_model" / ck
            if not mp.exists():
                raise FileNotFoundError(mp)

    tokenizer = AutoTokenizer.from_pretrained(str(ARM_CONFIGS[args.arms[0]] / "hf_model"), use_fast=True)
    vocab_ids = safe_vocab_ids(tokenizer)
    rng = random.Random(args.seed)

    copy_records, copy_stats = build_heldout_copy_records(tokenizer, vocab_ids, rng, args.copy_n_per_span, args.copy_span_lengths, args.max_len_copy_rewrite, args.copy_scan_rows)
    rewrite_limit = None if args.rewrite_max_pairs <= 0 else args.rewrite_max_pairs
    rewrite_records, rewrite_stats = build_rewrite_conditioning_records(tokenizer, vocab_ids, rng, rewrite_limit, args.rewrite_tokens_per_class, args.max_len_copy_rewrite)
    entity_limit = None if args.entity_max_items <= 0 else args.entity_max_items
    entity_records, entity_stats = build_entity_ablation_records(tokenizer, entity_limit)
    records = copy_records + rewrite_records + entity_records

    plan = {
        "status": "PROBE_PLAN",
        "created_utc": now(),
        "arms": args.arms,
        "checkpoints": args.checkpoints,
        "record_counts": {
            "heldout_natural_source_repeat": len(copy_records),
            "heldout_rewrite_content_conditioning": len(rewrite_records),
            "entity_gold_stale_cue_ablation": len(entity_records),
            "total": len(records),
        },
        "pair_counts": {
            "heldout_copy_pairs": len(copy_records) // 2,
            "rewrite_conditioning_pairs": len(rewrite_records) // 2,
            "entity_candidate_pairs": len(entity_records) // 2,
        },
        "copy_stats": copy_stats,
        "rewrite_stats": rewrite_stats,
        "entity_stats": entity_stats,
        "inputs": {
            "heldout_rows": rel(HELDOUT_ROWS),
            "all_accepted_pairs": rel(ALL_ACCEPTED_PAIRS),
            "selected_max_pairs_excluded": rel(SELECTED_MAX_PAIRS),
            "entity_dir": rel(ENTITY_DIR),
        },
        "scoring": {
            "copy_gain": "NLL(unrepeated/corrupted source) - NLL(repeated/source-present)",
            "rewrite_gain": "NLL(unrelated length-matched source) - NLL(true source)",
            "entity_margin": "gold answer logprob sum - stale-initial answer logprob sum under each ablated context",
        },
    }
    (out_dir / "probe_plan.json").write_text(json.dumps(plan, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if args.plan_only:
        print(json.dumps(plan, indent=2, ensure_ascii=False), flush=True)
        return

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    pad_id = int(tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0)
    print(json.dumps({**plan, "device": str(device), "started_utc": now()}, indent=2, ensure_ascii=False), flush=True)

    all_copy: list[dict[str, Any]] = []
    all_rewrite: list[dict[str, Any]] = []
    all_entity: list[dict[str, Any]] = []
    meta_rows: list[dict[str, Any]] = []
    for arm in args.arms:
        for ck in args.checkpoints:
            t0 = time.time()
            model_path = ARM_CONFIGS[arm] / "hf_model" / ck
            print(f"[LOAD] {arm} {ck} {model_path}", flush=True)
            model = AutoModelForMaskedLM.from_pretrained(str(model_path), torch_dtype=torch.float32)
            model.eval().to(device)
            scored = score_records(model, records, device, pad_id, args.batch_size)
            copy_rows = aggregate_copy(scored, arm, ck)
            rewrite_rows = aggregate_rewrite(scored, arm, ck)
            entity_rows = aggregate_entity(scored, arm, ck)
            write_csv(out_dir / f"copy_pair_rows_{arm}_{ck}.csv", copy_rows)
            write_csv(out_dir / f"rewrite_pair_rows_{arm}_{ck}.csv", rewrite_rows)
            write_csv(out_dir / f"entity_ablation_rows_{arm}_{ck}.csv", entity_rows)
            all_copy.extend(copy_rows); all_rewrite.extend(rewrite_rows); all_entity.extend(entity_rows)
            meta_rows.append({"arm": arm, "checkpoint": ck, "scored_records": len(scored), "copy_pairs": len(copy_rows), "rewrite_pairs": len(rewrite_rows), "entity_items": len(entity_rows), "elapsed_sec": round(time.time() - t0, 2)})
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
            print(f"[DONE] {arm} {ck} copy={len(copy_rows)} rewrite={len(rewrite_rows)} entity={len(entity_rows)} elapsed={time.time()-t0:.1f}s", flush=True)

    copy_summary = summarize_gain(all_copy, "copy")
    rewrite_summary = summarize_gain(all_rewrite, "rewrite")
    entity_summary = summarize_entity(all_entity)
    copy_con = contrast_summaries(copy_summary, ["mean_gain"], "copy")
    rewrite_con = contrast_summaries(rewrite_summary, ["mean_gain"], "rewrite")
    ent_keys = ["mean_margin_full", "mean_effect_no_initial_qbox", "mean_effect_no_last_relevant_update", "mean_effect_no_all_relevant_updates"]
    entity_con = contrast_summaries(entity_summary, ent_keys, "entity_ablation")
    copy_late = late_contrasts(copy_con, ["delta_mean_gain_a_minus_b"], "copy")
    rewrite_late = late_contrasts(rewrite_con, ["delta_mean_gain_a_minus_b"], "rewrite")
    entity_late = late_contrasts(entity_con, [f"delta_{k}_a_minus_b" for k in ent_keys], "entity_ablation")

    write_csv(out_dir / "copy_pair_rows.csv", all_copy)
    write_csv(out_dir / "rewrite_pair_rows.csv", all_rewrite)
    write_csv(out_dir / "entity_ablation_rows.csv", all_entity)
    write_csv(out_dir / "copy_summary.csv", copy_summary)
    write_csv(out_dir / "rewrite_summary.csv", rewrite_summary)
    write_csv(out_dir / "entity_ablation_summary.csv", entity_summary)
    write_csv(out_dir / "copy_contrasts.csv", copy_con)
    write_csv(out_dir / "rewrite_contrasts.csv", rewrite_con)
    write_csv(out_dir / "entity_ablation_contrasts.csv", entity_con)
    write_csv(out_dir / "copy_late_contrasts.csv", copy_late)
    write_csv(out_dir / "rewrite_late_contrasts.csv", rewrite_late)
    write_csv(out_dir / "entity_ablation_late_contrasts.csv", entity_late)
    write_csv(out_dir / "probe_meta.csv", meta_rows)
    make_note(out_dir, plan, copy_late, rewrite_late, entity_late)

    result = {
        "status": "PROBES_DONE",
        "finished_utc": now(),
        "device": str(device),
        "files": {
            "plan": rel(out_dir / "probe_plan.json"),
            "copy_late": rel(out_dir / "copy_late_contrasts.csv"),
            "rewrite_late": rel(out_dir / "rewrite_late_contrasts.csv"),
            "entity_late": rel(out_dir / "entity_ablation_late_contrasts.csv"),
            "copy_rows": rel(out_dir / "copy_pair_rows.csv"),
            "rewrite_rows": rel(out_dir / "rewrite_pair_rows.csv"),
            "entity_rows": rel(out_dir / "entity_ablation_rows.csv"),
            "note": rel(out_dir / "probe_readout.md"),
        },
    }
    (out_dir / "probe_result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("\nKEY HELD-OUT COPY")
    for r in copy_late:
        if r["group"] == "ALL" and r["contrast"] in {"RminusC", "RminusV", "VminusC"}:
            print(f"seed={r['seed']} {r['contrast']} dGain={r['late_mean_delta_mean_gain_a_minus_b']:+.4f} n={r['n']}")
    print("\nKEY REWRITE CONDITIONING")
    for r in rewrite_late:
        if r["group"] in {"ALL", "token_nonoverlap"} and r["contrast"] in {"VminusC", "VminusR", "CminusR"}:
            print(f"seed={r['seed']} {r['group']} {r['contrast']} dGain={r['late_mean_delta_mean_gain_a_minus_b']:+.4f} n={r['n']}")
    print("\nKEY ENTITY ABLATION")
    for r in entity_late:
        if r["group"] in {"ALL", "rel_ge2", "rel_ge3"} and r["contrast"] in {"RminusV", "VminusR", "VminusC", "RminusC"}:
            print(f"seed={r['seed']} {r['group']} {r['contrast']} dNoInit={r['late_mean_delta_mean_effect_no_initial_qbox_a_minus_b']:+.4f} dNoLast={r['late_mean_delta_mean_effect_no_last_relevant_update_a_minus_b']:+.4f} dNoAll={r['late_mean_delta_mean_effect_no_all_relevant_updates_a_minus_b']:+.4f} n={r['n']}")
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
