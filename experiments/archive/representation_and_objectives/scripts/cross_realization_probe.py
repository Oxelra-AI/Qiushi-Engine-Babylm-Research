#!/usr/bin/env python3
"""research: paired cross-realization probe for compact-view mechanisms.

Evaluation-only saved-checkpoint experiment.  For source-shared retained content
words that occur once in both a source sentence and its compact rewrite, score the
same target word under three contexts:

  source_context          source sentence with the target occurrence masked
  rewrite_context         genuine compact rewrite with the target occurrence masked
  counterfactual_context  another compact rewrite, matched on surface/interface
                          features, whose same-length donor word is masked but
                          whose visible context does not contain the target.

The target labels are identical token ids across the source and genuine rewrite
contexts; the counterfactual donor slot is required to have the same BPE-piece
length so the same labels can be placed at the same mask mass.  No model outputs
are used for item selection.  The probe measures whether existing checkpoints
have a genuine-rewrite advantage over matched counterfactual context, and whether
that advantage is carried by compact input exposure/shared target supervision or
requires the full target complement.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import pathlib
import random
import re
import statistics
import sys
import time
from typing import Any

import torch
import torch.nn.functional as F
from transformers import AutoModelForMaskedLM, AutoTokenizer

ROOT = pathlib.Path("experiments/archive/representation_and_objectives")
TOKENIZER = ROOT / "training/runs/gc_compact_repeat_reinvest_16k_seed43022_r2/hf_model/chck_1M"
TRAIN_FIXED_EVENTS = ROOT / "data/existing_trajectory_denoising_probe/probe_events.jsonl"
SOURCE_DISJOINT_EVENTS = ROOT / "data/source_disjoint_target_probe/source_disjoint_probe_events.jsonl"
DEFAULT_OUT = ROOT / "data/cross_realization_probe"
SEQ_LEN = 256

ARMS: dict[str, pathlib.Path] = {
    "full_compact_100M": pathlib.Path("experiments/archive/frontier_consolidation/training/runs/cleanqwen_fineweb_compact_view_reinvest_16k_seed43022/hf_model/chck_100M"),
    "drop_abs_100M": ROOT / "training/runs/packed_drop_abs_content_100M_fast/hf_model/chck_100M",
    "drop_copied_word_100M": ROOT / "training/runs/packed_drop_copied_content_wholeword_100M_fast/hf_model/chck_100M",
    "compact_repeat_100M": ROOT / "training/runs/gc_compact_repeat_reinvest_16k_seed43022_r2/hf_model/chck_100M",
    "adjbreak_100M": ROOT / "training/runs/gc_adjbreak_reinvest_16k_seed43022_r2/hf_model/chck_100M",
}

FUNC_WORDS = frozenset("""
a an the this that these those my your his her its our their is am are was were be been being
have has had do does did will would shall should can could may might must need ought to of in on
at by for with from into through during before after above below between under over about against
along across around behind beside beyond down near off since toward upon within without and but or
nor so yet both either neither not no if then else than as which who whom whose what when where how
while until because although though even also just only still already very much more most less least
too quite really rather such same other another each every all some any few many i me we us you he
him she her it they them one ones there here up out away again back now however therefore thus hence
moreover furthermore nevertheless meanwhile otherwise instead indeed perhaps maybe probably
""".split())
RELATIONAL_WORDS = set("""
able about across action active activity actor actors actually added adds affects after allows also among another appears area areas around associated because become becomes became before being between based called cause caused causes change changes changing common commonly compared connected consists contain contained contains created creates defined described different during effects either especially events example examples formed found from function functions general generally gives group groups having including indicates involves known leads made makes means method methods more most occurs often part parts process processes produces provides related relation relations relationship result results same separate shows similar since specific structure system systems through used using usually while within without works
""".split())
EVENT_STATE_WORDS = set("""
add added adds become becomes became broken built carried changed changes changing closed created creates damaged developed discovered done drop dropped dry eaten fall falls fell filled found frozen gave gives got grow grows has held hidden increased left located made make makes moved moves named opened placed produced removed said sent set shown stuck taken turn turned used went won wrote
""".split())
WORD_RE = re.compile(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)?")


def norm(w: str) -> str:
    m = WORD_RE.search(w)
    return m.group(0).lower() if m else re.sub(r"[^a-z0-9]", "", w.lower())


def is_content(w: str) -> bool:
    n = norm(w)
    return bool(n) and n not in FUNC_WORDS and len(n) > 1


def flags(w: str) -> dict[str, Any]:
    nw = norm(w)
    return {
        "capitalized": bool(w[:1].isupper() and not w.isupper()),
        "number": any(c.isdigit() for c in w),
        "hyphenated": "-" in w,
        "rel_event": nw in RELATIONAL_WORDS or nw in EVENT_STATE_WORDS,
        "short_norm": len(nw) <= 3,
    }


def token_count(tok, text: str) -> int:
    if not text:
        return 0
    return len(tok(text, add_special_tokens=False)["input_ids"])


def word_span(tok, words: list[str], word_i: int) -> tuple[int, int] | None:
    if word_i < 0 or word_i >= len(words):
        return None
    before = " ".join(words[:word_i])
    through = " ".join(words[: word_i + 1])
    st = token_count(tok, before)
    en = token_count(tok, through)
    if st < en and en <= SEQ_LEN:
        return st, en
    return None


def text_ids(tok, words: list[str]) -> list[int]:
    return list(tok(" ".join(words), add_special_tokens=False, truncation=True, max_length=SEQ_LEN)["input_ids"])


def source_side_overlap(src_words: list[str], side_words: list[str]) -> float:
    src = {norm(w) for w in src_words if is_content(w)}
    side = [norm(w) for w in side_words if is_content(w)]
    if not side:
        return 0.0
    return sum(1 for w in side if w in src) / len(side)


def load_retained_events(path: pathlib.Path, source_name: str, keep_sets: set[str] | None = None) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            d = json.loads(line)
            if str(d.get("category")) != "retained_content":
                continue
            es = str(d.get("eval_set", source_name))
            if keep_sets is not None and es not in keep_sets:
                continue
            d = dict(d)
            d.setdefault("eval_set", es)
            d.setdefault("event_uid", f"{source_name}|{i}|{d.get('pair_id')}|wi{d.get('word_index')}|retained")
            d["source_name"] = source_name
            d["line_index"] = i
            out.append(d)
    return out


def build_candidate(e: dict[str, Any], tok) -> tuple[dict[str, Any] | None, str]:
    src = str(e.get("source_text", ""))
    side = str(e.get("side_text", e.get("rewrite_text", "")))
    word = str(e.get("word", ""))
    if not src or not side or not word:
        return None, "missing_text_or_word"
    target_norm = norm(word)
    if not target_norm or not is_content(word):
        return None, "not_content"
    src_words = src.split()
    side_words = side.split()
    try:
        side_i = int(e.get("word_index"))
    except Exception:
        return None, "bad_word_index"
    if side_i < 0 or side_i >= len(side_words):
        return None, "side_index_oob"
    if norm(side_words[side_i]) != target_norm:
        return None, "side_word_norm_mismatch"
    # Clean leakage control: target normalized form appears exactly once in each context.
    src_pos = [i for i, w in enumerate(src_words) if norm(w) == target_norm]
    side_pos = [i for i, w in enumerate(side_words) if norm(w) == target_norm]
    if len(src_pos) != 1 or len(side_pos) != 1:
        return None, "target_not_unique_in_both_contexts"
    if side_pos[0] != side_i:
        return None, "side_unique_position_mismatch"
    src_span = word_span(tok, src_words, src_pos[0])
    side_span = word_span(tok, side_words, side_i)
    if src_span is None or side_span is None:
        return None, "span_failed"
    src_ids = text_ids(tok, src_words)
    side_ids = text_ids(tok, side_words)
    ss, se = src_span
    rs, re = side_span
    if se > len(src_ids) or re > len(side_ids):
        return None, "span_after_truncation"
    src_toks = src_ids[ss:se]
    side_toks = side_ids[rs:re]
    if not src_toks or src_toks != side_toks:
        return None, "source_rewrite_token_ids_not_identical"
    if len(src_ids) > SEQ_LEN or len(side_ids) > SEQ_LEN:
        return None, "too_long"
    src_norms = {norm(w) for w in src_words if norm(w)}
    side_norms = {norm(w) for w in side_words if norm(w)}
    f = flags(word)
    cand = {
        "event_uid": str(e.get("event_uid")),
        "eval_set": str(e.get("eval_set", "unknown")),
        "source_name": str(e.get("source_name", "unknown")),
        "pair_id": str(e.get("pair_id", e.get("event_uid"))),
        "doc_id": str(e.get("doc_id", "")),
        "source_text": src,
        "side_text": side,
        "word": word,
        "target_norm": target_norm,
        "source_word_index": int(src_pos[0]),
        "rewrite_word_index": int(side_i),
        "target_token_ids": src_toks,
        "target_pieces": len(src_toks),
        "source_ids": src_ids,
        "rewrite_ids": side_ids,
        "source_span": [ss, se],
        "rewrite_span": [rs, re],
        "source_word_len": len(src_words),
        "rewrite_word_len": len(side_words),
        "source_bpe_len": len(src_ids),
        "rewrite_bpe_len": len(side_ids),
        "rewrite_rel_pos": (side_i + 0.5) / max(1, len(side_words)),
        "compression_ratio": len(side_words) / max(1, len(src_words)),
        "source_side_overlap": source_side_overlap(src_words, side_words),
        "target_flags": f,
        "target_capitalized": f["capitalized"],
        "target_number": f["number"],
        "target_rel_event": f["rel_event"],
        "target_hyphenated": f["hyphenated"],
        "source_norm_count": len(src_norms),
        "rewrite_norm_count": len(side_norms),
    }
    return cand, "ok"


def build_slot(c: dict[str, Any], tok) -> dict[str, Any] | None:
    # The donor slot is the retained word in the donor rewrite.  Its context is not
    # required to imply the target; it is only an interface-matched compact context.
    side_words = str(c["side_text"]).split()
    wi = int(c["rewrite_word_index"])
    if wi < 0 or wi >= len(side_words):
        return None
    sp = tuple(c["rewrite_span"])
    if int(sp[1]) - int(sp[0]) != int(c["target_pieces"]):
        return None
    word = side_words[wi]
    f = flags(word)
    return {
        "donor_event_uid": c["event_uid"],
        "donor_pair_id": c["pair_id"],
        "donor_eval_set": c["eval_set"],
        "donor_word": word,
        "donor_norm": norm(word),
        "donor_ids": c["rewrite_ids"],
        "donor_span": list(c["rewrite_span"]),
        "target_pieces": int(c["target_pieces"]),
        "rewrite_word_len": int(c["rewrite_word_len"]),
        "rewrite_bpe_len": int(c["rewrite_bpe_len"]),
        "rewrite_rel_pos": float(c["rewrite_rel_pos"]),
        "compression_ratio": float(c["compression_ratio"]),
        "source_side_overlap": float(c["source_side_overlap"]),
        "capitalized": f["capitalized"],
        "number": f["number"],
        "rel_event": f["rel_event"],
        "hyphenated": f["hyphenated"],
        "side_norms": {norm(w) for w in side_words if norm(w)},
    }


def choose_counterfactual(target: dict[str, Any], slots_by_pieces: dict[int, list[dict[str, Any]]], rng: random.Random) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    tpieces = int(target["target_pieces"])
    target_norm = str(target["target_norm"])
    candidates = []
    for s in slots_by_pieces.get(tpieces, []):
        if s["donor_pair_id"] == target["pair_id"]:
            continue
        if s["donor_norm"] == target_norm:
            continue
        if target_norm in s["side_norms"]:
            continue
        # Strong interface matching for capitalization/number; otherwise the
        # counterfactual baseline would mix obvious orthographic classes.
        if bool(s["capitalized"]) != bool(target["target_capitalized"]):
            continue
        if bool(s["number"]) != bool(target["target_number"]):
            continue
        candidates.append(s)
    if not candidates:
        return None, {"reason": "no_candidate_same_piece_capnum_no_target"}

    def score(s: dict[str, Any]) -> tuple[float, float]:
        rel_penalty = 0.15 if bool(s["rel_event"]) != bool(target["target_rel_event"]) else 0.0
        hyph_penalty = 0.05 if bool(s["hyphenated"]) != bool(target["target_hyphenated"]) else 0.0
        val = (
            abs(float(s["rewrite_rel_pos"]) - float(target["rewrite_rel_pos"]))
            + 0.015 * abs(int(s["rewrite_word_len"]) - int(target["rewrite_word_len"]))
            + 0.010 * abs(int(s["rewrite_bpe_len"]) - int(target["rewrite_bpe_len"]))
            + 0.50 * abs(float(s["source_side_overlap"]) - float(target["source_side_overlap"]))
            + 0.50 * abs(float(s["compression_ratio"]) - float(target["compression_ratio"]))
            + rel_penalty
            + hyph_penalty
        )
        return val, rng.random() * 1e-9

    best = min(candidates, key=score)
    audit = {
        "reason": "ok",
        "candidate_count": len(candidates),
        "donor_event_uid": best["donor_event_uid"],
        "donor_pair_id": best["donor_pair_id"],
        "donor_word": best["donor_word"],
        "donor_norm": best["donor_norm"],
        "abs_rewrite_rel_pos": round(abs(float(best["rewrite_rel_pos"]) - float(target["rewrite_rel_pos"])), 6),
        "rewrite_word_len_delta": int(best["rewrite_word_len"]) - int(target["rewrite_word_len"]),
        "rewrite_bpe_len_delta": int(best["rewrite_bpe_len"]) - int(target["rewrite_bpe_len"]),
        "compression_ratio_delta": round(float(best["compression_ratio"]) - float(target["compression_ratio"]), 6),
        "source_side_overlap_delta": round(float(best["source_side_overlap"]) - float(target["source_side_overlap"]), 6),
        "rel_event_match": bool(best["rel_event"]) == bool(target["target_rel_event"]),
        "hyphenated_match": bool(best["hyphenated"]) == bool(target["target_hyphenated"]),
        "capitalized_match": True,
        "number_match": True,
        "piece_match": True,
        "counterfactual_contains_target": False,
    }
    return best, audit


def prepare_events(tok, max_per_eval_set: int | None, seed: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    keep_sets = {"source_disjoint_quality", "doc_disjoint_all_accepted", "doc_disjoint_quality"}
    raw = load_retained_events(TRAIN_FIXED_EVENTS, "train_fixed_probe")
    raw.extend(load_retained_events(SOURCE_DISJOINT_EVENTS, "source_disjoint", keep_sets=keep_sets))
    rng = random.Random(seed)
    by_set: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for e in raw:
        by_set[str(e.get("eval_set", "unknown"))].append(e)
    capped = []
    for es, rows in sorted(by_set.items()):
        rng.shuffle(rows)
        capped.extend(rows if max_per_eval_set is None else rows[:max_per_eval_set])

    skip = collections.Counter()
    candidates: list[dict[str, Any]] = []
    for e in capped:
        c, why = build_candidate(e, tok)
        if c is None:
            skip[why] += 1
        else:
            candidates.append(c)
    slots_by_pieces: dict[int, list[dict[str, Any]]] = collections.defaultdict(list)
    for c in candidates:
        s = build_slot(c, tok)
        if s is not None:
            slots_by_pieces[int(s["target_pieces"])].append(s)

    prepared: list[dict[str, Any]] = []
    donor_skips = collections.Counter()
    match_audits: list[dict[str, Any]] = []
    for c in candidates:
        donor, audit = choose_counterfactual(c, slots_by_pieces, rng)
        if donor is None:
            donor_skips[str(audit.get("reason", "no_donor"))] += 1
            continue
        c = dict(c)
        c["counterfactual_ids"] = list(donor["donor_ids"])
        c["counterfactual_span"] = list(donor["donor_span"])
        c["counterfactual_donor_event_uid"] = donor["donor_event_uid"]
        c["counterfactual_donor_pair_id"] = donor["donor_pair_id"]
        c["counterfactual_donor_word"] = donor["donor_word"]
        c["counterfactual_audit"] = audit
        prepared.append(c)
        match_audits.append(audit)

    if max_per_eval_set is not None:
        # Re-cap after donor matching so every set contributes similarly in pilots.
        by_set2: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
        for c in prepared:
            by_set2[c["eval_set"]].append(c)
        prepared2 = []
        for es, rows in sorted(by_set2.items()):
            rng.shuffle(rows)
            prepared2.extend(rows[:max_per_eval_set])
        prepared = prepared2
        match_audits = [c["counterfactual_audit"] for c in prepared]

    def mean(xs: list[float]) -> float | None:
        return round(float(statistics.mean(xs)), 6) if xs else None

    audit_summary = {
        "raw_retained_events": len(raw),
        "capped_events": len(capped),
        "candidate_events_after_clean_shared_target": len(candidates),
        "prepared_events": len(prepared),
        "skip": dict(skip),
        "donor_skip": dict(donor_skips),
        "by_eval_set": dict(collections.Counter(c["eval_set"] for c in prepared)),
        "by_target_pieces": dict(collections.Counter(str(c["target_pieces"]) for c in prepared)),
        "target_capitalized_fraction": mean([1.0 if c["target_capitalized"] else 0.0 for c in prepared]),
        "target_number_fraction": mean([1.0 if c["target_number"] else 0.0 for c in prepared]),
        "target_rel_event_fraction": mean([1.0 if c["target_rel_event"] else 0.0 for c in prepared]),
        "match": {
            "fraction_piece_match": mean([1.0 if a.get("piece_match") else 0.0 for a in match_audits]),
            "fraction_capitalized_match": mean([1.0 if a.get("capitalized_match") else 0.0 for a in match_audits]),
            "fraction_number_match": mean([1.0 if a.get("number_match") else 0.0 for a in match_audits]),
            "fraction_rel_event_match": mean([1.0 if a.get("rel_event_match") else 0.0 for a in match_audits]),
            "fraction_hyphenated_match": mean([1.0 if a.get("hyphenated_match") else 0.0 for a in match_audits]),
            "counterfactual_contains_target_fraction": mean([1.0 if a.get("counterfactual_contains_target") else 0.0 for a in match_audits]),
            "abs_rel_pos_mean": mean([float(a.get("abs_rewrite_rel_pos", 0.0)) for a in match_audits]),
            "rewrite_word_len_delta_mean": mean([float(a.get("rewrite_word_len_delta", 0.0)) for a in match_audits]),
            "rewrite_bpe_len_delta_mean": mean([float(a.get("rewrite_bpe_len_delta", 0.0)) for a in match_audits]),
            "compression_ratio_delta_mean": mean([float(a.get("compression_ratio_delta", 0.0)) for a in match_audits]),
            "source_side_overlap_delta_mean": mean([float(a.get("source_side_overlap_delta", 0.0)) for a in match_audits]),
            "candidate_count_median": round(float(statistics.median([int(a.get("candidate_count", 0)) for a in match_audits])), 3) if match_audits else None,
        },
    }
    return prepared, audit_summary


def make_examples(events: list[dict[str, Any]], tok) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    for e in events:
        for ctx, ids_key, span_key in [
            ("source", "source_ids", "source_span"),
            ("rewrite", "rewrite_ids", "rewrite_span"),
            ("counterfactual", "counterfactual_ids", "counterfactual_span"),
        ]:
            ids = list(e[ids_key])
            st, en = map(int, e[span_key])
            if st < 0 or en > len(ids) or st >= en:
                continue
            if en - st != int(e["target_pieces"]):
                continue
            labels = [-100] * len(ids)
            target_ids = list(map(int, e["target_token_ids"]))
            labels[st:en] = target_ids
            masked = list(ids)
            masked[st:en] = [int(tok.mask_token_id)] * (en - st)
            examples.append({
                "event_uid": e["event_uid"],
                "eval_set": e["eval_set"],
                "pair_id": e["pair_id"],
                "doc_id": e.get("doc_id", ""),
                "context": ctx,
                "input_ids": masked,
                "labels": labels,
                "span": [st, en],
                "pieces": en - st,
                "target_norm": e["target_norm"],
                "target_word": e["word"],
                "target_rel_event": bool(e["target_rel_event"]),
                "target_capitalized": bool(e["target_capitalized"]),
                "target_number": bool(e["target_number"]),
                "target_pieces": int(e["target_pieces"]),
            })
    return examples


def sha256_file(path: pathlib.Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def eval_arm(arm: str, model_path: pathlib.Path, examples: list[dict[str, Any]], tok, device: str, batch_size: int) -> list[dict[str, Any]]:
    print(json.dumps({"event": "arm_start", "arm": arm, "model_path": str(model_path), "n_examples": len(examples)}), flush=True)
    model = AutoModelForMaskedLM.from_pretrained(str(model_path), local_files_only=True)
    model.to(device)
    model.eval()
    rows: list[dict[str, Any]] = []
    pad_id = int(tok.pad_token_id)
    with torch.no_grad():
        for start in range(0, len(examples), batch_size):
            batch = examples[start : start + batch_size]
            max_len = min(SEQ_LEN, max(len(ex["input_ids"]) for ex in batch))
            input_ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
            labels = torch.full((len(batch), max_len), -100, dtype=torch.long)
            for i, ex in enumerate(batch):
                ids = list(ex["input_ids"][:max_len])
                labs = list(ex["labels"][:max_len])
                input_ids[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)
                labels[i, : len(labs)] = torch.tensor(labs, dtype=torch.long)
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            attn = (input_ids != pad_id).long().to(device)
            out = model(input_ids=input_ids, attention_mask=attn, output_hidden_states=True)
            logits = out.logits
            hidden = out.hidden_states[-1]
            vocab = logits.shape[-1]
            per_tok = F.cross_entropy(logits.reshape(-1, vocab), labels.reshape(-1), reduction="none", ignore_index=-100).reshape(labels.shape)
            mask = labels != -100
            for b, ex in enumerate(batch):
                vals = per_tok[b][mask[b]]
                if vals.numel() == 0:
                    continue
                hs = hidden[b][mask[b]].float().mean(dim=0)
                rows.append({
                    "arm": arm,
                    "event_uid": ex["event_uid"],
                    "eval_set": ex["eval_set"],
                    "pair_id": ex["pair_id"],
                    "doc_id": ex.get("doc_id", ""),
                    "context": ex["context"],
                    "loss_sum": float(vals.sum().detach().cpu()),
                    "loss_mean": float(vals.mean().detach().cpu()),
                    "pieces": int(vals.numel()),
                    "hidden": hs.detach().cpu(),
                    "target_norm": ex["target_norm"],
                    "target_word": ex["target_word"],
                    "target_rel_event": bool(ex["target_rel_event"]),
                    "target_capitalized": bool(ex["target_capitalized"]),
                    "target_number": bool(ex["target_number"]),
                    "target_pieces": int(ex["target_pieces"]),
                })
    del model
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    print(json.dumps({"event": "arm_done", "arm": arm, "n_rows": len(rows)}), flush=True)
    return rows


def cluster_key(r: dict[str, Any]) -> str:
    d = str(r.get("doc_id", ""))
    if d and d.lower() != "none":
        return "doc:" + d
    return "pair:" + str(r.get("pair_id", r.get("event_uid")))


def bootstrap_mean(rows: list[dict[str, Any]], key: str, n_boot: int, seed: int) -> dict[str, Any]:
    vals = [float(r[key]) for r in rows if key in r and r[key] is not None and math.isfinite(float(r[key]))]
    if not vals:
        return {"n": 0}
    point = float(statistics.mean(vals))
    by_cluster: dict[str, list[float]] = collections.defaultdict(list)
    for r in rows:
        if key in r and r[key] is not None and math.isfinite(float(r[key])):
            by_cluster[cluster_key(r)].append(float(r[key]))
    clusters = list(by_cluster.items())
    rng = random.Random(seed)
    boots = []
    for _ in range(n_boot):
        acc = []
        for _j in range(len(clusters)):
            _cid, xs = clusters[rng.randrange(len(clusters))]
            acc.extend(xs)
        boots.append(float(statistics.mean(acc)))
    boots.sort()
    def q(p: float) -> float:
        return boots[min(len(boots)-1, max(0, int(round(p*(len(boots)-1)))))]
    return {
        "n": len(vals),
        "n_clusters": len(clusters),
        "mean": round(point, 6),
        "p025": round(q(0.025), 6),
        "p50": round(q(0.5), 6),
        "p975": round(q(0.975), 6),
        "fraction_gt0": round(sum(1 for x in boots if x > 0)/len(boots), 6),
    }


def summarize_arm(rows: list[dict[str, Any]], n_boot: int, seed: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    grouped: dict[str, dict[str, dict[str, Any]]] = collections.defaultdict(dict)
    meta: dict[str, dict[str, Any]] = {}
    for r in rows:
        grouped[str(r["event_uid"])][str(r["context"])] = r
        meta[str(r["event_uid"])] = r
    per_event: list[dict[str, Any]] = []
    for uid, ctxs in grouped.items():
        if not {"source", "rewrite", "counterfactual"}.issubset(ctxs):
            continue
        src = ctxs["source"]
        rw = ctxs["rewrite"]
        cf = ctxs["counterfactual"]
        # Piece-normalized context losses.  Source/rewrite/cf all have identical target ids and piece count by construction.
        src_loss = float(src["loss_sum"]) / int(src["pieces"])
        rw_loss = float(rw["loss_sum"]) / int(rw["pieces"])
        cf_loss = float(cf["loss_sum"]) / int(cf["pieces"])
        hs = src["hidden"]
        hr = rw["hidden"]
        hc = cf["hidden"]
        cos_sr = float(F.cosine_similarity(hs, hr, dim=0).item())
        cos_sc = float(F.cosine_similarity(hs, hc, dim=0).item())
        rec = {
            "event_uid": uid,
            "eval_set": src["eval_set"],
            "pair_id": src["pair_id"],
            "doc_id": src.get("doc_id", ""),
            "target_norm": src["target_norm"],
            "target_word": src["target_word"],
            "target_rel_event": bool(src["target_rel_event"]),
            "target_capitalized": bool(src["target_capitalized"]),
            "target_number": bool(src["target_number"]),
            "target_pieces": int(src["target_pieces"]),
            "source_loss": src_loss,
            "rewrite_loss": rw_loss,
            "counterfactual_loss": cf_loss,
            "rewrite_advantage": cf_loss - rw_loss,
            "source_advantage": cf_loss - src_loss,
            "rewrite_minus_source_loss": rw_loss - src_loss,
            "cos_source_rewrite": cos_sr,
            "cos_source_counterfactual": cos_sc,
            "agreement_advantage": cos_sr - cos_sc,
        }
        per_event.append(rec)
    strata: dict[str, list[dict[str, Any]]] = {
        "all": per_event,
        "target_rel_event": [r for r in per_event if r["target_rel_event"]],
        "target_non_rel_event": [r for r in per_event if not r["target_rel_event"]],
        "target_cap_or_number": [r for r in per_event if r["target_capitalized"] or r["target_number"]],
        "target_plain": [r for r in per_event if not (r["target_capitalized"] or r["target_number"])],
    }
    for es in sorted(set(r["eval_set"] for r in per_event)):
        strata[f"eval_set/{es}"] = [r for r in per_event if r["eval_set"] == es]
    metrics = [
        "source_loss", "rewrite_loss", "counterfactual_loss",
        "rewrite_advantage", "source_advantage", "rewrite_minus_source_loss",
        "cos_source_rewrite", "cos_source_counterfactual", "agreement_advantage",
    ]
    summary: dict[str, Any] = {"n_complete_events": len(per_event), "strata": {}}
    for sname, srows in strata.items():
        summary["strata"][sname] = {m: bootstrap_mean(srows, m, n_boot, seed + stable_int(sname + m) % 1000000) for m in metrics}
    return summary, per_event


def stable_int(s: str) -> int:
    return int.from_bytes(hashlib.blake2b(s.encode("utf-8"), digest_size=8).digest(), "little")


def summarize_between_arms(per_arm_events: dict[str, list[dict[str, Any]]], n_boot: int, seed: int) -> dict[str, Any]:
    by_arm_uid = {arm: {r["event_uid"]: r for r in rows} for arm, rows in per_arm_events.items()}
    contrasts = [
        ("full_minus_drop_abs", "full_compact_100M", "drop_abs_100M"),
        ("full_minus_drop_copied_word", "full_compact_100M", "drop_copied_word_100M"),
        ("drop_abs_minus_drop_copied_word", "drop_abs_100M", "drop_copied_word_100M"),
        ("drop_abs_minus_compact_repeat", "drop_abs_100M", "compact_repeat_100M"),
        ("full_minus_compact_repeat", "full_compact_100M", "compact_repeat_100M"),
        ("adjbreak_minus_compact_repeat", "adjbreak_100M", "compact_repeat_100M"),
        ("drop_abs_minus_adjbreak", "drop_abs_100M", "adjbreak_100M"),
    ]
    metrics = ["rewrite_advantage", "agreement_advantage", "rewrite_loss", "counterfactual_loss", "rewrite_minus_source_loss"]
    out: dict[str, Any] = {}
    for cname, a, b in contrasts:
        if a not in by_arm_uid or b not in by_arm_uid:
            continue
        common = sorted(set(by_arm_uid[a]) & set(by_arm_uid[b]))
        rows = []
        for uid in common:
            ra = by_arm_uid[a][uid]
            rb = by_arm_uid[b][uid]
            rec = {
                "event_uid": uid,
                "eval_set": ra["eval_set"],
                "pair_id": ra["pair_id"],
                "doc_id": ra.get("doc_id", ""),
                "target_rel_event": ra["target_rel_event"],
                "target_capitalized": ra["target_capitalized"],
                "target_number": ra["target_number"],
            }
            for m in metrics:
                rec[m] = float(ra[m]) - float(rb[m])
            rows.append(rec)
        strata: dict[str, list[dict[str, Any]]] = {
            "all": rows,
            "target_rel_event": [r for r in rows if r["target_rel_event"]],
            "target_non_rel_event": [r for r in rows if not r["target_rel_event"]],
            "target_cap_or_number": [r for r in rows if r["target_capitalized"] or r["target_number"]],
            "target_plain": [r for r in rows if not (r["target_capitalized"] or r["target_number"])],
        }
        for es in sorted(set(r["eval_set"] for r in rows)):
            strata[f"eval_set/{es}"] = [r for r in rows if r["eval_set"] == es]
        out[cname] = {"a": a, "b": b, "n_common": len(common), "strata": {}}
        for sname, srows in strata.items():
            out[cname]["strata"][sname] = {m: bootstrap_mean(srows, m, n_boot, seed + stable_int(cname + sname + m) % 1000000) for m in metrics}
    return out


def write_jsonl(path: pathlib.Path, rows: list[dict[str, Any]], include_hidden: bool = False) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            rr = dict(r)
            if not include_hidden and "hidden" in rr:
                rr.pop("hidden", None)
            f.write(json.dumps(rr, ensure_ascii=False) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=pathlib.Path, default=DEFAULT_OUT)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch_size", type=int, default=96)
    ap.add_argument("--max_per_eval_set", type=int, default=None)
    ap.add_argument("--bootstrap_iters", type=int, default=500)
    ap.add_argument("--seed", type=int, default=238013)
    ap.add_argument("--arms", default="full_compact_100M,drop_abs_100M,drop_copied_word_100M,compact_repeat_100M,adjbreak_100M")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    if args.out_dir.exists() and any(args.out_dir.iterdir()) and not args.force:
        raise SystemExit(f"Output dir exists and nonempty: {args.out_dir}; use --force")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(str(TOKENIZER), local_files_only=True, use_fast=True)
    selected_arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    missing = [a for a in selected_arms if a not in ARMS]
    if missing:
        raise SystemExit(f"Unknown arms: {missing}")
    model_paths = {a: ARMS[a] for a in selected_arms if ARMS[a].exists()}
    missing_paths = {a: str(ARMS[a]) for a in selected_arms if not ARMS[a].exists()}
    if missing_paths:
        print(json.dumps({"event": "missing_model_paths", "missing": missing_paths}), flush=True)

    prepared, prep_summary = prepare_events(tok, args.max_per_eval_set, args.seed)
    examples = make_examples(prepared, tok)
    prep_summary.update({
        "status": "CROSS_REALIZATION_PROBE_PREPARED",
        "tokenizer": str(TOKENIZER),
        "tokenizer_vocab_size": len(tok),
        "seq_len": SEQ_LEN,
        "n_examples": len(examples),
        "selected_arms": selected_arms,
        "available_arms": sorted(model_paths),
        "missing_model_paths": missing_paths,
    })
    with (args.out_dir / "prepared_events.jsonl").open("w", encoding="utf-8") as f:
        for c in prepared:
            cc = {k: v for k, v in c.items() if k not in {"source_ids", "rewrite_ids", "counterfactual_ids", "target_token_ids"}}
            cc["target_token_ids_sha"] = hashlib.sha256(json.dumps(c["target_token_ids"]).encode()).hexdigest()[:16]
            f.write(json.dumps(cc, ensure_ascii=False) + "\n")
    (args.out_dir / "prep_summary.json").write_text(json.dumps(prep_summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(prep_summary), flush=True)
    if not prepared:
        raise SystemExit("No prepared events")

    per_arm_summary: dict[str, Any] = {}
    per_arm_events: dict[str, list[dict[str, Any]]] = {}
    for arm, path in model_paths.items():
        rows = eval_arm(arm, path, examples, tok, args.device, args.batch_size)
        write_jsonl(args.out_dir / f"{arm}_context_rows.jsonl", rows, include_hidden=False)
        summary, ev = summarize_arm(rows, args.bootstrap_iters, args.seed + stable_int(arm) % 1000000)
        per_arm_summary[arm] = {
            "model_path": str(path),
            "model_file_sha256": sha256_file(path / "pytorch_model.bin") or sha256_file(path / "model.safetensors"),
            **summary,
        }
        per_arm_events[arm] = ev
        write_jsonl(args.out_dir / f"{arm}_per_event_metrics.jsonl", ev, include_hidden=False)

    between = summarize_between_arms(per_arm_events, args.bootstrap_iters, args.seed)
    result = {
        "status": "CROSS_REALIZATION_PROBE_DONE",
        "created_at": time.time(),
        "interpretation": {
            "rewrite_advantage": "counterfactual_loss - rewrite_loss; positive means genuine compact rewrite context predicts the shared target better than matched counterfactual compact context",
            "agreement_advantage": "cos(final_hidden_source_mask, final_hidden_rewrite_mask) - cos(final_hidden_source_mask, final_hidden_counterfactual_mask); positive means source/rewrite masked-target representations agree more than source/counterfactual",
            "arm_contrast_sign": "a_minus_b for each metric; positive rewrite_advantage contrast means arm a has stronger genuine-rewrite advantage than arm b",
            "caution": "This saved-checkpoint probe sharpens causal separation but cannot by itself prove input exposure alone, because dense shared-target gradients remain coupled in the trained arms.",
        },
        "prep": prep_summary,
        "arms": per_arm_summary,
        "between_arm_contrasts": between,
    }
    out_json = args.out_dir / "cross_realization_probe.json"
    out_json.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    # Human-readable compact summary.
    lines = ["# research cross-realization probe", "", f"JSON: `{out_json}`", "", "## Preparation", f"- prepared_events: {prep_summary['prepared_events']}", f"- n_examples: {prep_summary['n_examples']}", f"- by_eval_set: {prep_summary['by_eval_set']}", f"- match: {prep_summary['match']}", "", "## Arm main metrics (all events)"]
    for arm in selected_arms:
        if arm not in per_arm_summary:
            continue
        allm = per_arm_summary[arm]["strata"]["all"]
        lines.append(f"### {arm}")
        for m in ["rewrite_advantage", "agreement_advantage", "rewrite_loss", "counterfactual_loss", "rewrite_minus_source_loss"]:
            s = allm[m]
            lines.append(f"- {m}: n={s.get('n')}, mean={s.get('mean')}, boot[{s.get('p025')},{s.get('p975')}], frac_gt0={s.get('fraction_gt0')}")
    lines.extend(["", "## Key between-arm contrasts (all events)"])
    for cname in ["full_minus_drop_abs", "full_minus_drop_copied_word", "drop_abs_minus_drop_copied_word", "drop_abs_minus_compact_repeat", "adjbreak_minus_compact_repeat", "drop_abs_minus_adjbreak"]:
        if cname not in between:
            continue
        lines.append(f"### {cname}")
        allc = between[cname]["strata"]["all"]
        for m in ["rewrite_advantage", "agreement_advantage", "rewrite_loss"]:
            s = allc[m]
            lines.append(f"- {m}: n={s.get('n')}, mean={s.get('mean')}, boot[{s.get('p025')},{s.get('p975')}], frac_gt0={s.get('fraction_gt0')}")
    (args.out_dir / "cross_realization_probe.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": "CROSS_REALIZATION_PROBE_DONE", "out_json": str(out_json), "prepared_events": len(prepared), "arms": sorted(per_arm_summary)}), flush=True)


if __name__ == "__main__":
    main()
