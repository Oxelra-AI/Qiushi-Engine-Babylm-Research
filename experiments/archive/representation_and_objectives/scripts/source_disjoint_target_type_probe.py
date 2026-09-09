#!/usr/bin/env python3
"""research: source-disjoint held-out target-type probe for compact source-absent channel.

The research/226 target-selective effect was measured on compact pairs that were also present
as training inputs.  Even probe words not selected by WWM had remained visible in the same
source+rewrite rows.  This script asks whether the contrast survives on compact pairs whose
source sentence never entered training, using the unused medium-compact rewrite reservoir.

It evaluates existing 20M checkpoints only; it does not train a model.  Main contrast:

    drop_abs_content  -  drop_copied_content_wholeword

Positive loss delta on held-out source_absent_content means that deleting source-absent compact
content labels during training hurts prediction of unseen source-absent compact-content targets
more than deleting matched whole-word copied-content labels.  Survival on source-disjoint pairs
supports abstraction beyond repeated-row exposure; failure localizes the effect to reformulation
learning inside repeated rows.
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
from transformers import AutoModelForMaskedLM

USER_ROOT = pathlib.Path(".").resolve()
ROOT = pathlib.Path("experiments/archive/representation_and_objectives")
A02 = pathlib.Path("experiments/archive/frontier_consolidation")
SCRIPT_DIR = USER_ROOT / "experiments/archive" / 'representation_and_objectives' / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import crossview_deberta_trainer_v2 as base  # noqa: E402

TRAIN_PAIRS = A02 / "data/density_core_reinvestment_medium_riskhard/selected_compact_reinvest_pairs.jsonl"
RESERVOIR = A02 / "data/medium_compact_analysis/medium_compact_ws_rows.jsonl"
TRAIN_DATA = ROOT / "data/crossview_data_v2/crossview_consolidated_v2.jsonl"
TRAIN_PROBE_EVENTS = ROOT / "data/existing_trajectory_denoising_probe/probe_events.jsonl"
TOKENIZER = ROOT / "training/runs/gc_compact_repeat_reinvest_16k_seed43022_r2/hf_model/chck_1M"
OUT_DIR = ROOT / "data/source_disjoint_target_probe"
SEQ_LEN = 256

ARMS = {
    "full": ROOT / "training/runs/crossview_own_visible_20M",
    "drop_abs": ROOT / "training/runs/target_selective_drop_abs_content_20M",
    "drop_copied_tok": ROOT / "training/runs/target_selective_drop_copied_matched_20M",
    "drop_copied_word": ROOT / "training/runs/target_selective_drop_copied_content_wholeword_20M",
}
CHECKPOINTS = ["chck_10M", "chck_20M"]
CATEGORIES = ["retained_content", "source_absent_content", "function_other"]

FUNC_WORDS = frozenset("a an the this that these those my your his her its our their "
    "is am are was were be been being have has had do does did will would shall should "
    "can could may might must need ought to of in on at by for with from into through "
    "during before after above below between under over about against along across "
    "around behind beside beyond down near off since toward upon within without "
    "and but or nor so yet both either neither not no nor if then else than as "
    "which who whom whose what when where how while until because although though "
    "even also just only still already very much more most less least too quite "
    "really rather than such same other another each every all some any few many "
    "no more several enough another each own same different many little much few "
    "i me we us you he him she her it they them myself yourself himself herself "
    "itself ourselves themselves one ones there here up out off away again back "
    "now then so however therefore thus hence moreover furthermore nevertheless "
    "meanwhile otherwise instead indeed certainly perhaps maybe probably "
    "been being getting going doing having making taking".split())


def norm(w: str) -> str:
    return re.sub(r"[^a-z0-9]", "", w.lower())


def is_content(w: str) -> bool:
    n = norm(w)
    return n not in FUNC_WORDS and len(n) > 1


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def q_stats(xs: list[float]) -> dict[str, Any]:
    if not xs:
        return {"n": 0}
    s = sorted(float(x) for x in xs)
    n = len(s)

    def q(p: float) -> float:
        return s[min(n - 1, max(0, int(round(p * (n - 1)))))]

    return {
        "n": n,
        "mean": round(statistics.mean(s), 6),
        "median": round(statistics.median(s), 6),
        "p05": round(q(0.05), 6),
        "p10": round(q(0.10), 6),
        "p25": round(q(0.25), 6),
        "p75": round(q(0.75), 6),
        "p90": round(q(0.90), 6),
        "p95": round(q(0.95), 6),
        "min": round(s[0], 6),
        "max": round(s[-1], 6),
    }


def token_count(tok, text: str) -> int:
    return len(tok(text, add_special_tokens=False)["input_ids"])


def target_span(tok, source_text: str, side_words: list[str], word_i: int) -> tuple[int, int] | None:
    before_side = " ".join(side_words[:word_i])
    through_side = " ".join(side_words[: word_i + 1])
    prefix_before = source_text if not before_side else source_text + " " + before_side
    prefix_after = source_text + " " + through_side
    st = token_count(tok, prefix_before)
    en = token_count(tok, prefix_after)
    if st < en and st < SEQ_LEN:
        return st, min(en, SEQ_LEN)
    return None


def stable_u64(s: str) -> int:
    return int.from_bytes(hashlib.blake2b(s.encode("utf-8"), digest_size=8).digest(), "little")


def read_training_identity() -> dict[str, Any]:
    keys, docs, sources = set(), set(), set()
    with TRAIN_PAIRS.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            keys.add(str(d["key"]))
            docs.add(str(d["doc_id"]))
            sources.add(str(d["source_text"]))
    return {"keys": keys, "docs": docs, "sources": sources}


def build_training_token_freq(tok, out_dir: pathlib.Path, args) -> dict[int, int]:
    cache = out_dir / "training_input_token_freq_20M.json"
    meta_cache = out_dir / "training_input_token_freq_20M_meta.json"
    if cache.exists() and not args.force_token_freq:
        raw = json.loads(cache.read_text(encoding="utf-8"))
        return {int(k): int(v) for k, v in raw.items()}

    pool = base.load_pool(str(TRAIN_DATA))
    examples, aw, ep = base.build_stream(pool, args.max_word_exposure, "own", args.stream_seed)
    if aw != args.max_word_exposure:
        raise RuntimeError(f"stream filled {aw}, expected {args.max_word_exposure}")
    ds = base.PairAwareDataset(examples, tok, SEQ_LEN, "own", False, args.mask_prob, args.mask_seed)
    special_ids = set(int(x) for x in tok.all_special_ids)
    token_freq: collections.Counter[int] = collections.Counter()
    t0 = time.time()
    for i in range(len(ds)):
        item = ds[i]
        ids = item["input_ids"]
        att = item["attention_mask"]
        for tid in ids[att.bool()].tolist():
            it = int(tid)
            if it not in special_ids:
                token_freq[it] += 1
        if args.progress_every and (i + 1) % args.progress_every == 0:
            print(json.dumps({"event": "token_freq_progress", "examples": i + 1, "unique": len(token_freq), "elapsed_sec": round(time.time() - t0, 1)}), flush=True)
    cache.write_text(json.dumps({str(k): int(v) for k, v in sorted(token_freq.items())}, indent=2) + "\n", encoding="utf-8")
    meta = {
        "status": "TRAINING_INPUT_TOKEN_FREQ_BUILT",
        "train_data": str(TRAIN_DATA),
        "train_data_sha256": sha256_file(TRAIN_DATA),
        "tokenizer": str(TOKENIZER),
        "max_word_exposure": args.max_word_exposure,
        "stream_seed": args.stream_seed,
        "mask_seed": args.mask_seed,
        "mask_prob": args.mask_prob,
        "examples": len(ds),
        "actual_words": aw,
        "epochs_realized": ep,
        "unique_ids": len(token_freq),
        "elapsed_sec": round(time.time() - t0, 1),
    }
    meta_cache.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return dict(token_freq)


def attach_support_and_features(e: dict[str, Any], tok, token_freq: dict[int, int]) -> dict[str, Any] | None:
    src = e["source_text"]
    side = e["side_text"]
    side_words = side.split()
    wi = int(e["word_index"])
    full_ids = tok(src + " " + side, add_special_tokens=False, truncation=True, max_length=SEQ_LEN)["input_ids"]
    span = e.get("span")
    if span is None:
        span = target_span(tok, src, side_words, wi)
    if span is None:
        return None
    st, en = int(span[0]), int(span[1])
    en = min(en, len(full_ids), SEQ_LEN)
    if st >= en:
        return None
    toks = [int(x) for x in full_ids[st:en]]
    supports = [int(token_freq.get(int(t), 0)) for t in toks]
    logs = [math.log1p(x) for x in supports]
    bnd = token_count(tok, src)
    rewrite_len_bpe = max(1, len(full_ids) - bnd)
    mid = 0.5 * (st + en - 1)
    out = dict(e)
    out["span"] = [st, en]
    out["n_pieces"] = en - st
    out["token_ids"] = toks
    out["support_min"] = min(supports) if supports else 0
    out["support_mean"] = sum(supports) / max(1, len(supports))
    out["support_log_min"] = min(logs) if logs else 0.0
    out["support_log_mean"] = sum(logs) / max(1, len(logs))
    out["rel_word_pos"] = wi / max(1, len(side_words) - 1)
    out["seq_token_mid"] = mid / max(1, SEQ_LEN - 1)
    out["rel_token_mid"] = (mid - bnd) / max(1, rewrite_len_bpe - 1)
    out["input_ids"] = full_ids
    return out


def load_reference_events(tok, token_freq: dict[int, int]) -> list[dict[str, Any]]:
    events = []
    with TRAIN_PROBE_EVENTS.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            d.setdefault("eval_set", "train_probe_reference")
            e = attach_support_and_features(d, tok, token_freq)
            if e is not None:
                events.append(e)
    return events


def reservoir_candidates(tok, token_freq: dict[int, int], train_id: dict[str, Any], args) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    subsets = {
        "source_disjoint_quality": [],
        "doc_disjoint_quality": [],
        "doc_disjoint_all_accepted": [],
    }
    counts = collections.Counter()
    source_norm_cache: dict[str, set[str]] = {}
    with RESERVOIR.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            counts["total_rows"] += 1
            if not d.get("accepted_for_next_construction"):
                continue
            counts["accepted_rows"] += 1
            key = f"sid:{d['sentence_id']}|doc:{d['doc_id']}"
            exact_source_disjoint = key not in train_id["keys"] and str(d["source_text"]) not in train_id["sources"]
            doc_disjoint = str(d["doc_id"]) not in train_id["docs"]
            if exact_source_disjoint:
                counts["accepted_exact_source_disjoint"] += 1
            if doc_disjoint:
                counts["accepted_doc_disjoint"] += 1
            cr = float(d.get("content_recall") or 0.0)
            lr = float(d.get("length_ratio") or 0.0)
            quality = cr >= args.min_content_recall and args.min_length_ratio <= lr <= args.max_length_ratio
            if not exact_source_disjoint and not doc_disjoint:
                continue
            pair = {
                "pair_id": f"heldout_compact:sid{d['sentence_id']}:doc{d['doc_id']}",
                "key": key,
                "doc_id": str(d["doc_id"]),
                "sentence_id": str(d["sentence_id"]),
                "source_text": str(d["source_text"]),
                "side_text": str(d["rewrite_text"]),
                "rewrite_text": str(d["rewrite_text"]),
                "content_recall": cr,
                "length_ratio": lr,
                "domain_hits": d.get("domain_hits") or [],
                "exact_source_disjoint": exact_source_disjoint,
                "doc_disjoint": doc_disjoint,
                "quality_matched": quality,
            }
            if exact_source_disjoint and quality:
                subsets["source_disjoint_quality"].append(pair)
            if doc_disjoint and quality:
                subsets["doc_disjoint_quality"].append(pair)
            if doc_disjoint:
                subsets["doc_disjoint_all_accepted"].append(pair)

    # Convert pairs to word-level target events.
    event_subsets: dict[str, list[dict[str, Any]]] = {k: [] for k in subsets}
    for subset, pairs in subsets.items():
        for p in pairs:
            src = p["source_text"]
            side = p["side_text"]
            source_norms = source_norm_cache.get(src)
            if source_norms is None:
                source_norms = set(norm(w) for w in src.split() if norm(w))
                source_norm_cache[src] = source_norms
            side_words = side.split()
            full_ids = tok(src + " " + side, add_special_tokens=False, truncation=True, max_length=SEQ_LEN)["input_ids"]
            if len(full_ids) < 4:
                continue
            for wi, w in enumerate(side_words):
                nw = norm(w)
                if not nw:
                    continue
                span = target_span(tok, src, side_words, wi)
                if span is None or span[0] >= span[1] or span[1] > len(full_ids):
                    continue
                if is_content(w) and nw in source_norms:
                    cat = "retained_content"
                elif is_content(w) and nw not in source_norms:
                    cat = "source_absent_content"
                else:
                    cat = "function_other"
                e = {
                    "eval_set": subset,
                    "pair_id": p["pair_id"],
                    "key": p["key"],
                    "doc_id": p["doc_id"],
                    "sentence_id": p["sentence_id"],
                    "source_text": src,
                    "side_text": side,
                    "word": w,
                    "word_index": wi,
                    "category": cat,
                    "span": [span[0], span[1]],
                    "n_pieces": span[1] - span[0],
                    "domain_hits": p["domain_hits"],
                    "content_recall": p["content_recall"],
                    "length_ratio": p["length_ratio"],
                    "exact_source_disjoint": p["exact_source_disjoint"],
                    "doc_disjoint": p["doc_disjoint"],
                    "quality_matched": p["quality_matched"],
                }
                ee = attach_support_and_features(e, tok, token_freq)
                if ee is not None:
                    event_subsets[subset].append(ee)

    meta = {
        "reservoir_counts": dict(counts),
        "pair_counts_by_subset": {k: len(v) for k, v in subsets.items()},
        "event_candidate_counts_by_subset_category": {
            k: dict(collections.Counter(e["category"] for e in v)) for k, v in event_subsets.items()
        },
        "quality_filters": {
            "min_content_recall": args.min_content_recall,
            "min_length_ratio": args.min_length_ratio,
            "max_length_ratio": args.max_length_ratio,
        },
    }
    return event_subsets, meta


def summarize_events(events: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {
        "n_events": len(events),
        "n_pairs": len(set(e["pair_id"] for e in events)),
        "category_counts": dict(collections.Counter(e["category"] for e in events)),
        "piece_total_by_category": dict(collections.Counter({})),
    }
    piece_counter = collections.Counter()
    for e in events:
        piece_counter[e["category"]] += int(e["n_pieces"])
    out["piece_total_by_category"] = dict(piece_counter)
    for feat in ["n_pieces", "support_log_mean", "support_log_min", "support_min", "support_mean", "rel_word_pos", "rel_token_mid", "seq_token_mid", "content_recall", "length_ratio"]:
        out[feat] = q_stats([float(e[feat]) for e in events if feat in e])
    out["by_category"] = {}
    for cat in CATEGORIES:
        ce = [e for e in events if e["category"] == cat]
        out["by_category"][cat] = {
            "n_events": len(ce),
            "n_pairs": len(set(e["pair_id"] for e in ce)),
            "piece_total": int(sum(int(e["n_pieces"]) for e in ce)),
            "n_pieces": q_stats([float(e["n_pieces"]) for e in ce]),
            "support_log_mean": q_stats([float(e["support_log_mean"]) for e in ce]),
            "support_log_min": q_stats([float(e["support_log_min"]) for e in ce]),
            "rel_word_pos": q_stats([float(e["rel_word_pos"]) for e in ce]),
            "seq_token_mid": q_stats([float(e["seq_token_mid"]) for e in ce]),
        }
    return out


def event_score(ref: dict[str, Any], cand: dict[str, Any]) -> float:
    # BPE length and support are the main matching variables; position prevents tail/head artifacts.
    s = 0.0
    s += 1000.0 * abs(int(ref["n_pieces"]) - int(cand["n_pieces"]))
    s += 8.0 * abs(float(ref["support_log_mean"]) - float(cand["support_log_mean"]))
    s += 5.0 * abs(float(ref["support_log_min"]) - float(cand["support_log_min"]))
    s += 3.0 * abs(float(ref["rel_word_pos"]) - float(cand["rel_word_pos"]))
    s += 2.0 * abs(float(ref["seq_token_mid"]) - float(cand["seq_token_mid"]))
    return s


def choose_matched_events(ref_events: list[dict[str, Any]], candidates: list[dict[str, Any]], requested: int, seed: int, pair_cap: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rng = random.Random(seed)
    by_len: dict[int, list[int]] = collections.defaultdict(list)
    for j, c in enumerate(candidates):
        by_len[int(c["n_pieces"])].append(j)
    for lst in by_len.values():
        rng.shuffle(lst)
    refs = list(ref_events)
    rng.shuffle(refs)
    # Harder references first: longer/rarer targets make matching difficult.
    refs.sort(key=lambda e: (-int(e["n_pieces"]), float(e["support_log_mean"]), stable_u64(str(e.get("pair_id", "")) + str(e.get("word_index", "")))))
    used: set[int] = set()
    pair_counts: collections.Counter[str] = collections.Counter()
    selected: list[dict[str, Any]] = []
    fallback_counts: collections.Counter[str] = collections.Counter()
    score_sum = 0.0
    length_mismatch = 0

    def pool_for(ref: dict[str, Any]) -> tuple[str, list[int]]:
        L = int(ref["n_pieces"])
        for radius in [0, 1, 2, 3, 4, 8, 999]:
            if radius == 0:
                lens = [L]
                label = "exact_len"
            elif radius == 999:
                lens = list(by_len.keys())
                label = "global_fallback"
            else:
                lens = [x for x in range(max(1, L - radius), L + radius + 1) if x in by_len]
                label = f"len_radius_{radius}"
            pool = []
            for le in lens:
                for j in by_len.get(le, []):
                    if j in used:
                        continue
                    if pair_counts[str(candidates[j]["pair_id"])] >= pair_cap:
                        continue
                    pool.append(j)
            if pool:
                return label, pool
        return "none", []

    for ref in refs:
        if len(selected) >= requested:
            break
        label, pool = pool_for(ref)
        if not pool:
            break
        best = min(pool, key=lambda j: (event_score(ref, candidates[j]), stable_u64(str(candidates[j]["pair_id"]) + str(candidates[j]["word_index"]))))
        cand = dict(candidates[best])
        used.add(best)
        pair_counts[str(cand["pair_id"])] += 1
        sc = event_score(ref, cand)
        score_sum += sc
        if int(ref["n_pieces"]) != int(cand["n_pieces"]):
            length_mismatch += 1
        cand["matched_ref_pair_id"] = ref.get("pair_id")
        cand["matched_ref_word_index"] = ref.get("word_index")
        cand["match_score_to_train_probe"] = round(sc, 6)
        cand["match_class"] = label
        cand["feature_delta_to_train_probe"] = {
            "n_pieces": int(cand["n_pieces"]) - int(ref["n_pieces"]),
            "support_log_mean": round(float(cand["support_log_mean"]) - float(ref["support_log_mean"]), 6),
            "support_log_min": round(float(cand["support_log_min"]) - float(ref["support_log_min"]), 6),
            "rel_word_pos": round(float(cand["rel_word_pos"]) - float(ref["rel_word_pos"]), 6),
            "seq_token_mid": round(float(cand["seq_token_mid"]) - float(ref["seq_token_mid"]), 6),
        }
        fallback_counts[label] += 1
        selected.append(cand)
    diag = {
        "requested": requested,
        "selected": len(selected),
        "candidate_events": len(candidates),
        "unique_pairs_selected": len(set(e["pair_id"] for e in selected)),
        "length_mismatch_events": length_mismatch,
        "fraction_exact_bpe_len": round(1.0 - length_mismatch / max(1, len(selected)), 6),
        "match_class_counts": dict(fallback_counts),
        "mean_match_score": round(score_sum / max(1, len(selected)), 6),
        "pair_cap": pair_cap,
    }
    return selected, diag


def build_matched_probe_sets(ref_events: list[dict[str, Any]], event_subsets: dict[str, list[dict[str, Any]]], args) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    ref_by_cat = {cat: [e for e in ref_events if e["category"] == cat] for cat in CATEGORIES}
    selected_all: list[dict[str, Any]] = []
    meta: dict[str, Any] = {}
    for subset, candidates in event_subsets.items():
        meta[subset] = {"selection_by_category": {}, "candidate_summary": summarize_events(candidates)}
        cand_by_cat = {cat: [e for e in candidates if e["category"] == cat] for cat in CATEGORIES}
        if subset == "source_disjoint_quality":
            req = args.source_per_category
        elif subset == "doc_disjoint_quality":
            req = args.doc_quality_per_category
        else:
            req = args.doc_all_per_category
        # Equal category composition where possible; if a strict subset lacks enough targets, use the available matched sample.
        for cat in CATEGORIES:
            requested = min(req, len(ref_by_cat[cat]), len(cand_by_cat[cat]))
            selected, diag = choose_matched_events(ref_by_cat[cat], cand_by_cat[cat], requested, args.match_seed + 1009 * (1 + CATEGORIES.index(cat)) + stable_u64(subset) % 100000, args.pair_cap)
            for e in selected:
                e["eval_set"] = subset
                e["category"] = cat
            selected_all.extend(selected)
            meta[subset]["selection_by_category"][cat] = diag
        meta[subset]["selected_summary"] = summarize_events([e for e in selected_all if e["eval_set"] == subset])
    return selected_all, meta


def make_batch(events: list[dict[str, Any]], tok) -> tuple[torch.Tensor, torch.Tensor, list[dict[str, Any]]]:
    max_len = min(SEQ_LEN, max(len(e["input_ids"]) for e in events))
    input_batch = torch.full((len(events), max_len), tok.pad_token_id, dtype=torch.long)
    labels = torch.full((len(events), max_len), -100, dtype=torch.long)
    for i, e in enumerate(events):
        ids = list(e["input_ids"][:max_len])
        st, en = int(e["span"][0]), int(e["span"][1])
        en = min(en, len(ids), max_len)
        input_batch[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)
        if st < en:
            labels[i, st:en] = torch.tensor(ids[st:en], dtype=torch.long)
            input_batch[i, st:en] = tok.mask_token_id
    return input_batch, labels, events


def eval_model_events(model_path: pathlib.Path, events: list[dict[str, Any]], tok, device: str, batch_size: int) -> list[dict[str, Any]]:
    model = AutoModelForMaskedLM.from_pretrained(str(model_path), local_files_only=True)
    model.to(device)
    model.eval()
    per_event = []
    with torch.no_grad():
        for i in range(0, len(events), batch_size):
            batch = events[i : i + batch_size]
            input_ids, labels, raw = make_batch(batch, tok)
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            attn = (input_ids != tok.pad_token_id).long().to(device)
            logits = model(input_ids=input_ids, attention_mask=attn).logits
            vocab = logits.shape[-1]
            per_tok = F.cross_entropy(logits.view(-1, vocab), labels.view(-1), reduction="none", ignore_index=-100).view(labels.shape)
            mask = labels != -100
            for b, e in enumerate(raw):
                vals = per_tok[b][mask[b]]
                if vals.numel() == 0:
                    continue
                per_event.append({
                    "event_uid": e["event_uid"],
                    "eval_set": e["eval_set"],
                    "category": e["category"],
                    "pair_id": e["pair_id"],
                    "doc_id": e.get("doc_id"),
                    "loss_sum": float(vals.sum().detach().cpu()),
                    "pieces": int(vals.numel()),
                })
    del model
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return per_event


def piece_weighted(records: list[dict[str, Any]]) -> float | None:
    p = sum(int(r["pieces"]) for r in records)
    if p == 0:
        return None
    return sum(float(r["loss_sum"]) for r in records) / p


def cluster_bootstrap(delta_records: list[dict[str, Any]], n_boot: int, seed: int) -> dict[str, Any] | None:
    by_cluster: dict[str, list[tuple[float, int]]] = collections.defaultdict(list)
    for r in delta_records:
        by_cluster[str(r.get("pair_id") or r.get("event_uid"))].append((float(r["delta_sum"]), int(r["pieces"])))
    clusters = list(by_cluster.keys())
    if not clusters:
        return None
    rng = random.Random(seed)
    vals = []
    for _ in range(n_boot):
        num = 0.0
        den = 0
        for _ in range(len(clusters)):
            c = clusters[rng.randrange(len(clusters))]
            for ds, p in by_cluster[c]:
                num += ds
                den += p
        vals.append(num / den if den else 0.0)
    vals.sort()
    n = len(vals)

    def q(p: float) -> float:
        return vals[min(n - 1, max(0, int(round(p * (n - 1)))))]

    return {
        "median": round(statistics.median(vals), 6),
        "p025": round(q(0.025), 6),
        "p05": round(q(0.05), 6),
        "p95": round(q(0.95), 6),
        "p975": round(q(0.975), 6),
        "fraction_gt0": round(sum(1 for x in vals if x > 0) / n, 4),
    }


def summarize_eval(per_arm_ck: dict[str, dict[str, list[dict[str, Any]]]], events: list[dict[str, Any]], args) -> dict[str, Any]:
    event_index = {e["event_uid"]: e for e in events}
    eval_sets = sorted(set(e["eval_set"] for e in events))
    out: dict[str, Any] = {"piece_weighted_category_loss": {}, "contrasts": {}, "decision_fields": {}}
    for ck in CHECKPOINTS:
        out["piece_weighted_category_loss"][ck] = {}
        for arm in ARMS:
            recs = per_arm_ck.get(arm, {}).get(ck, [])
            by_key: dict[tuple[str, str], list[dict[str, Any]]] = collections.defaultdict(list)
            for r in recs:
                by_key[(r["eval_set"], r["category"])].append(r)
            out["piece_weighted_category_loss"][ck][arm] = {}
            for es in eval_sets:
                out["piece_weighted_category_loss"][ck][arm][es] = {}
                for cat in CATEGORIES:
                    v = piece_weighted(by_key.get((es, cat), []))
                    out["piece_weighted_category_loss"][ck][arm][es][cat] = round(v, 6) if v is not None else None

        contrasts = [
            ("drop_abs_minus_drop_copied_word", "drop_abs", "drop_copied_word"),
            ("drop_abs_minus_drop_copied_tok", "drop_abs", "drop_copied_tok"),
            ("drop_abs_minus_full", "drop_abs", "full"),
            ("drop_copied_word_minus_full", "drop_copied_word", "full"),
        ]
        out["contrasts"][ck] = {}
        for name, a_arm, b_arm in contrasts:
            a_by = {r["event_uid"]: r for r in per_arm_ck[a_arm][ck]}
            b_by = {r["event_uid"]: r for r in per_arm_ck[b_arm][ck]}
            out["contrasts"][ck][name] = {}
            for es in eval_sets:
                out["contrasts"][ck][name][es] = {}
                for cat in CATEGORIES:
                    delta_records = []
                    for uid, ar in a_by.items():
                        ev = event_index.get(uid)
                        br = b_by.get(uid)
                        if ev is None or br is None:
                            continue
                        if ev["eval_set"] != es or ev["category"] != cat:
                            continue
                        if int(ar["pieces"]) != int(br["pieces"]):
                            continue
                        delta_records.append({
                            "event_uid": uid,
                            "pair_id": ev["pair_id"],
                            "delta_sum": float(ar["loss_sum"]) - float(br["loss_sum"]),
                            "pieces": int(ar["pieces"]),
                        })
                    v = piece_weighted([{"loss_sum": r["delta_sum"], "pieces": r["pieces"]} for r in delta_records])
                    out["contrasts"][ck][name][es][cat] = {
                        "n_events": len(delta_records),
                        "n_pairs": len(set(r["pair_id"] for r in delta_records)),
                        "piece_weighted_delta": round(v, 6) if v is not None else None,
                        "pair_cluster_bootstrap": cluster_bootstrap(delta_records, args.n_boot, args.seed + stable_u64(ck + name + es + cat) % 1_000_000),
                    }
    # Compact decision fields for the specified contrast.
    for es in eval_sets:
        key = f"{es}/chck_20M/drop_abs_minus_drop_copied_word/source_absent_content"
        out["decision_fields"][key] = out["contrasts"].get("chck_20M", {}).get("drop_abs_minus_drop_copied_word", {}).get(es, {}).get("source_absent_content")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output_dir", default=str(OUT_DIR))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch_size", type=int, default=96)
    ap.add_argument("--source_per_category", type=int, default=2048)
    ap.add_argument("--doc_quality_per_category", type=int, default=192)
    ap.add_argument("--doc_all_per_category", type=int, default=512)
    ap.add_argument("--pair_cap", type=int, default=2)
    ap.add_argument("--min_content_recall", type=float, default=0.75)
    ap.add_argument("--min_length_ratio", type=float, default=0.50)
    ap.add_argument("--max_length_ratio", type=float, default=1.05)
    ap.add_argument("--max_word_exposure", type=int, default=20_000_000)
    ap.add_argument("--stream_seed", type=int, default=43)
    ap.add_argument("--mask_seed", type=int, default=430230221)
    ap.add_argument("--mask_prob", type=float, default=0.15)
    ap.add_argument("--match_seed", type=int, default=22743023)
    ap.add_argument("--seed", type=int, default=227)
    ap.add_argument("--n_boot", type=int, default=1000)
    ap.add_argument("--progress_every", type=int, default=30000)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--force_token_freq", action="store_true")
    ap.add_argument("--preflight_only", action="store_true")
    args = ap.parse_args()

    out_dir = pathlib.Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / "source_disjoint_target_type_probe.json"
    events_jsonl = out_dir / "source_disjoint_probe_events.jsonl"
    per_event_jsonl = out_dir / "source_disjoint_event_losses.jsonl"
    if out_json.exists() and not args.force and not args.preflight_only:
        print(out_json)
        return

    t0 = time.time()
    tok = base.make_portable_tokenizer(str(TOKENIZER))
    train_id = read_training_identity()
    token_freq = build_training_token_freq(tok, out_dir, args)
    ref_events = load_reference_events(tok, token_freq)
    event_subsets, reservoir_meta = reservoir_candidates(tok, token_freq, train_id, args)
    selected_events, selection_meta = build_matched_probe_sets(ref_events, event_subsets, args)
    for i, e in enumerate(selected_events):
        e["event_uid"] = f"{e['eval_set']}|{i}|{e['pair_id']}|wi{e['word_index']}|{e['category']}"
    with events_jsonl.open("w", encoding="utf-8") as f:
        for e in selected_events:
            # Omit input ids from durable event list to keep it readable; they can be recomputed from text/span.
            row = {k: v for k, v in e.items() if k != "input_ids"}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    preflight_payload = {
        "status": "SOURCE_DISJOINT_TARGET_TYPE_PREFLIGHT" if args.preflight_only else "SOURCE_DISJOINT_TARGET_TYPE_RUNNING",
        "meaning": "Build source-disjoint compact-pair held-out probe sets matched to the in-training fixed probe by target category, BPE length, tokenizer support, and position.",
        "inputs": {
            "train_pairs": str(TRAIN_PAIRS),
            "train_pairs_sha256": sha256_file(TRAIN_PAIRS),
            "reservoir": str(RESERVOIR),
            "reservoir_sha256": sha256_file(RESERVOIR),
            "train_data": str(TRAIN_DATA),
            "tokenizer": str(TOKENIZER),
            "reference_probe_events": str(TRAIN_PROBE_EVENTS),
            "source_per_category": args.source_per_category,
            "doc_quality_per_category": args.doc_quality_per_category,
            "doc_all_per_category": args.doc_all_per_category,
            "pair_cap": args.pair_cap,
        },
        "training_identity_counts": {"keys": len(train_id["keys"]), "docs": len(train_id["docs"]), "sources": len(train_id["sources"])},
        "reservoir_meta": reservoir_meta,
        "reference_summary": summarize_events(ref_events),
        "selection_meta": selection_meta,
        "selected_summary": summarize_events(selected_events),
        "events_jsonl": str(events_jsonl),
        "elapsed_preflight_sec": round(time.time() - t0, 1),
    }
    preflight_path = out_dir / "source_disjoint_target_type_preflight.json"
    preflight_path.write_text(json.dumps(preflight_payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if args.preflight_only:
        print(json.dumps({
            "status": preflight_payload["status"],
            "preflight": str(preflight_path),
            "selected_summary": preflight_payload["selected_summary"],
            "reservoir_meta": reservoir_meta,
            "elapsed_sec": preflight_payload["elapsed_preflight_sec"],
        }, indent=2, ensure_ascii=False), flush=True)
        return

    missing = []
    per_arm_ck: dict[str, dict[str, list[dict[str, Any]]]] = {arm: {} for arm in ARMS}
    for arm, root in ARMS.items():
        for ck in CHECKPOINTS:
            mp = root / "hf_model" / ck
            if not mp.exists():
                missing.append(str(mp))
                per_arm_ck[arm][ck] = []
                continue
            print(json.dumps({"event": "eval_start", "arm": arm, "checkpoint": ck, "model": str(mp), "n_events": len(selected_events), "device": args.device}), flush=True)
            per_arm_ck[arm][ck] = eval_model_events(mp, selected_events, tok, args.device, args.batch_size)
            print(json.dumps({"event": "eval_done", "arm": arm, "checkpoint": ck, "records": len(per_arm_ck[arm][ck])}), flush=True)

    with per_event_jsonl.open("w", encoding="utf-8") as f:
        by_arm_ck = {arm: {ck: {r["event_uid"]: r for r in recs} for ck, recs in per_arm_ck[arm].items()} for arm in ARMS}
        for e in selected_events:
            row = {
                "event_uid": e["event_uid"],
                "eval_set": e["eval_set"],
                "pair_id": e["pair_id"],
                "doc_id": e.get("doc_id"),
                "category": e["category"],
                "word": e.get("word"),
                "word_index": e.get("word_index"),
                "n_pieces": e.get("n_pieces"),
                "support_log_mean": e.get("support_log_mean"),
                "support_log_min": e.get("support_log_min"),
            }
            for ck in CHECKPOINTS:
                row[ck] = {}
                for arm in ARMS:
                    r = by_arm_ck[arm][ck].get(e["event_uid"])
                    if r is not None:
                        row[ck][arm] = {"loss_sum": r["loss_sum"], "pieces": r["pieces"], "loss_mean": r["loss_sum"] / max(1, r["pieces"])}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    eval_summary = summarize_eval(per_arm_ck, selected_events, args)
    payload = dict(preflight_payload)
    payload.update({
        "status": "SOURCE_DISJOINT_TARGET_TYPE_PROBE_DONE",
        "meaning": "Evaluation-only test of whether the research/226 source-absent target-type effect survives on compact pairs whose source sentence never entered the training pair set; strict document-disjoint subsets are reported separately.",
        "device": args.device,
        "checkpoints": CHECKPOINTS,
        "arms": {k: str(v) for k, v in ARMS.items()},
        "missing_models": missing,
        "event_losses_jsonl": str(per_event_jsonl),
        "eval_summary": eval_summary,
        "scientific_reading": [
            "Positive drop_abs_minus_drop_copied_word on source_disjoint_quality/source_absent_content supports abstraction beyond repeated-row exposure because all probe source+rewrite rows were absent from training input.",
            "The doc_disjoint_quality subset is stricter but small; use it as corroboration or warning, not as the only statistical population.",
            "Retained_content and function_other should not show the same positive pattern if the effect is specifically source-absent compact-content rather than broad model quality.",
            "This readout is still within the same teacher-generated FineWeb compact reservoir and same tokenizer/model family; transfer to other data or architectures remains a later question."
        ],
        "elapsed_total_sec": round(time.time() - t0, 1),
    })
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": payload["status"],
        "out": str(out_json),
        "missing": missing,
        "decision_fields": eval_summary["decision_fields"],
        "elapsed_sec": payload["elapsed_total_sec"],
    }, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
