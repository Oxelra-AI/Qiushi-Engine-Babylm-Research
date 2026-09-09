#!/usr/bin/env python3
"""research: audit target-selection matching under the actual packed 100M WWM schedule.

The research packed intervention matched source-absent compact-content and copied-content
whole-word groups at the 10M pool level.  This script simulates the exact deterministic
WWM target schedule used by packed_target_selective_trainer over the historical
647,400-row / 100M stream, then measures what masked whole-word events and BPE pieces
would actually be removed by

  * drop_abs_content (all rw_abs_content masked groups), and
  * the research copied-content whole-word selection.

If the pool-level selection is not well matched after realized WWM, the script also builds
an event-level copied-content control: one copied masked whole-word event matched to each
realized source-absent-content masked event, preferring exact BPE length and close epoch,
token-support, and position features.  A follow-up trainer can use the emitted
stream_index/group_id event map.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import pathlib
import random
import statistics
import sys
import time
from typing import Any

import torch

USER_ROOT = pathlib.Path(".").resolve()
SCRIPT_DIR = USER_ROOT / "experiments/archive" / 'representation_and_objectives' / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import packed_target_selective_trainer as train228  # noqa: E402
from annotate_packed_pool import word_class  # noqa: E402

DEFAULT_STREAM = "experiments/archive/frontier_consolidation/data/density_cleanqwen_overlay_medium_riskhard/cleanqwen_fineweb_compact_view_reinvest_100M.jsonl"
DEFAULT_ANNOTATION = "experiments/archive/representation_and_objectives/data/packed_pool_annotation/packed_pool_annotations.jsonl"
DEFAULT_SELECTION = "experiments/archive/representation_and_objectives/data/packed_pool_annotation/packed_wholeword_copied_selection.jsonl"
DEFAULT_TOKENIZER = "experiments/archive/representation_and_objectives/training/runs/gc_compact_repeat_reinvest_16k_seed43022_r2/hf_model/chck_1M"
DEFAULT_OUT = "experiments/archive/representation_and_objectives/data/realized_wwm_selection_audit"

CAT_RW_COPIED = train228.CAT_TO_IDX[train228.CAT_RW_COPIED]
CAT_RW_ABS_CONTENT = train228.CAT_TO_IDX[train228.CAT_RW_ABS_CONTENT]


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def stable_u64(s: str) -> int:
    return int.from_bytes(hashlib.blake2b(s.encode("utf-8"), digest_size=8).digest(), "little")


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


def summarize(events: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {
        "n_events": len(events),
        "piece_total": int(sum(int(e["bpe_len"]) for e in events)),
        "unique_example_group_count": len({(int(e["example_id"]), int(e["group_id"])) for e in events}),
        "epoch_counts": dict(sorted(collections.Counter(str(int(e["epoch"])) for e in events).items(), key=lambda kv: int(kv[0]))),
        "bpe_len_counts": dict(sorted(collections.Counter(str(int(e["bpe_len"])) for e in events).items(), key=lambda kv: int(kv[0]))),
    }
    for key in [
        "bpe_len",
        "support_log_mean",
        "support_log_min",
        "support_mean",
        "support_min",
        "seq_token_mid",
        "rel_group_pos",
        "row_order_frac",
    ]:
        out[key] = q_stats([float(e[key]) for e in events])
    # content of copied selection purity, when present
    if events and "word_class" in events[0]:
        out["word_class_counts"] = dict(collections.Counter(str(e.get("word_class", "")) for e in events))
    return out


def mean_delta(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    # Return b - a, used as selected/control minus absent.
    out: dict[str, Any] = {
        "event_delta": int(b.get("n_events", 0) - a.get("n_events", 0)),
        "piece_delta": int(b.get("piece_total", 0) - a.get("piece_total", 0)),
    }
    for key in [
        "bpe_len",
        "support_log_mean",
        "support_log_min",
        "support_mean",
        "support_min",
        "seq_token_mid",
        "rel_group_pos",
        "row_order_frac",
    ]:
        av = a.get(key, {}).get("mean")
        bv = b.get(key, {}).get("mean")
        if av is not None and bv is not None:
            out[f"{key}_mean_delta"] = round(float(bv) - float(av), 6)
    return out


def load_annotations(path: pathlib.Path) -> tuple[dict[int, list[int]], dict[int, int], dict[int, list[str]], set[int]]:
    annotations: dict[int, list[int]] = {}
    row_to_eid: dict[int, int] = {}
    word_cats_str: dict[int, list[str]] = {}
    changed_eids: set[int] = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            a = json.loads(line)
            eid = int(a["example_id"])
            cats_s = list(a["word_categories"])
            annotations[eid] = [train228.CAT_TO_IDX.get(c, 0) for c in cats_s]
            word_cats_str[eid] = cats_s
            row_to_eid[int(a["row_index"])] = eid
            if any(c.startswith("rw_") for c in cats_s):
                changed_eids.add(eid)
    return annotations, row_to_eid, word_cats_str, changed_eids


def load_pool_selection(path: pathlib.Path, row_to_eid: dict[int, int]) -> set[tuple[int, int]]:
    pos: set[tuple[int, int]] = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            g = json.loads(line)
            eid = row_to_eid.get(int(g["row_index"]))
            if eid is not None:
                pos.add((int(eid), int(g["word_index"])))
    return pos


def load_stream(path: pathlib.Path) -> list[dict[str, Any]]:
    stream: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if not line.strip():
                continue
            r = json.loads(line)
            stream.append({
                "stream_index": idx,
                "example_id": int(r.get("example_id", idx)),
                "text": str(r["text"]),
                "words": int(r.get("words", len(str(r["text"]).split()))),
            })
    return stream


class EncoderCache:
    def __init__(self, tokenizer, annotations: dict[int, list[int]], seq_length: int):
        self.tok = tokenizer
        self.annotations = annotations
        self.seq_length = seq_length
        self.special_ids = set(int(x) for x in tokenizer.all_special_ids)
        self.word_start_cache: dict[int, bool] = {}
        self.cache: dict[int, dict[str, Any]] = {}

    def is_word_start(self, token_id: int) -> bool:
        v = self.word_start_cache.get(int(token_id))
        if v is None:
            s = self.tok.convert_ids_to_tokens(int(token_id))
            v = bool(s is not None and train228.is_word_start(str(s)))
            self.word_start_cache[int(token_id)] = v
        return v

    def encode(self, entry: dict[str, Any]) -> dict[str, Any]:
        eid = int(entry["example_id"])
        got = self.cache.get(eid)
        if got is not None:
            return got
        enc = self.tok(
            entry["text"], add_special_tokens=False, truncation=True,
            max_length=self.seq_length, padding="max_length", return_tensors="pt",
        )
        input_ids = enc["input_ids"].squeeze(0).long()
        attention_mask = enc["attention_mask"].squeeze(0).long()
        group = torch.full_like(input_ids, -1)
        gid = -1
        for i in range(input_ids.shape[0]):
            if int(attention_mask[i]) == 0:
                continue
            tid = int(input_ids[i])
            if tid in self.special_ids:
                continue
            if gid < 0 or self.is_word_start(tid) or i == 0:
                gid += 1
            group[i] = gid
        token_cat = torch.zeros_like(input_ids)
        word_cats = self.annotations.get(eid)
        if word_cats is not None:
            for i in range(input_ids.shape[0]):
                g = int(group[i])
                if g >= 0 and g < len(word_cats):
                    token_cat[i] = int(word_cats[g])
        words = entry["text"].split()
        max_gid = int(group.max().item()) if (group >= 0).any() else -1
        # quick flag for rows with rewrite-side target categories inside the visible 256-token window
        has_interest = bool(((token_cat == CAT_RW_ABS_CONTENT) | (token_cat == CAT_RW_COPIED)).any().item())
        out = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "word_group": group,
            "token_cat": token_cat,
            "words": words,
            "max_gid": max_gid,
            "has_interest": has_interest,
        }
        self.cache[eid] = out
        return out


def build_token_freq(stream: list[dict[str, Any]], cache: EncoderCache) -> dict[int, int]:
    occ = collections.Counter(int(e["example_id"]) for e in stream)
    first_entry: dict[int, dict[str, Any]] = {}
    for e in stream:
        eid = int(e["example_id"])
        if eid not in first_entry:
            first_entry[eid] = e
    freq: collections.Counter[int] = collections.Counter()
    for eid, e in first_entry.items():
        enc = cache.encode(e)
        ids = enc["input_ids"]
        att = enc["attention_mask"].bool()
        mult = int(occ[eid])
        for tid in ids[att].tolist():
            it = int(tid)
            if it not in cache.special_ids:
                freq[it] += mult
    return dict(freq)


def make_event(
    *,
    origin: str,
    entry: dict[str, Any],
    epoch: int,
    gid: int,
    positions: list[int],
    label_ids: list[int],
    token_freq: dict[int, int],
    words: list[str],
    max_gid: int,
    n_stream: int,
    selected_by_pool: bool,
) -> dict[str, Any]:
    supports = [int(token_freq.get(int(t), 0)) for t in label_ids]
    logs = [math.log1p(x) for x in supports]
    word_text = words[gid] if 0 <= gid < len(words) else ""
    mid = 0.5 * (min(positions) + max(positions))
    return {
        "origin": origin,
        "stream_index": int(entry["stream_index"]),
        "example_id": int(entry["example_id"]),
        "epoch": int(epoch),
        "group_id": int(gid),
        "word_text": word_text,
        "word_class": word_class(word_text) if word_text else "",
        "positions": [int(p) for p in positions],
        "token_ids": [int(t) for t in label_ids],
        "bpe_len": int(len(label_ids)),
        "seq_token_mid": float(mid / 255.0),
        "rel_group_pos": float(gid / max(1, max_gid)),
        "row_order_frac": float(entry["stream_index"] / max(1, n_stream - 1)),
        "support_min": int(min(supports) if supports else 0),
        "support_mean": float(sum(supports) / max(1, len(supports))),
        "support_log_min": float(min(logs) if logs else 0.0),
        "support_log_mean": float(sum(logs) / max(1, len(logs))),
        "selected_by_pool_selection": bool(selected_by_pool),
        "event_key": f"s{int(entry['stream_index'])}|e{int(entry['example_id'])}|g{int(gid)}",
    }


def collect_realized_events(args, stream: list[dict[str, Any]], cache: EncoderCache, pool_drop_positions: set[tuple[int, int]], token_freq: dict[int, int]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    device = torch.device(args.device)
    gen = torch.Generator(device=device)
    gen.manual_seed(args.train_rng_seed)
    abs_events: list[dict[str, Any]] = []
    copied_content_events: list[dict[str, Any]] = []
    pool_selected_copied_events: list[dict[str, Any]] = []
    occ_seen: collections.Counter[int] = collections.Counter()
    purity = collections.Counter()
    t0 = time.time()
    n = len(stream)

    for start in range(0, n, args.batch_size):
        entries = stream[start:start + args.batch_size]
        encs = [cache.encode(e) for e in entries]
        input_ids_cpu = torch.stack([e["input_ids"] for e in encs])
        attention_cpu = torch.stack([e["attention_mask"] for e in encs])
        group_cpu = torch.stack([e["word_group"] for e in encs])
        token_cat_cpu = torch.stack([e["token_cat"] for e in encs])
        input_ids = input_ids_cpu.to(device, non_blocking=True)
        attention_mask = attention_cpu.to(device, non_blocking=True)
        word_group = group_cpu.to(device, non_blocking=True)

        # This function consumes the same dedicated CUDA generator calls as training:
        # group selection, replacement randoms, and random-token ids.
        _, labels = train228.apply_wwm_masking(input_ids, attention_mask, word_group, cache.tok, args.mask_prob, gen)
        labels_cpu = labels.cpu()
        mask_cpu = labels_cpu != -100

        for b, entry in enumerate(entries):
            eid = int(entry["example_id"])
            epoch = int(occ_seen[eid])
            occ_seen[eid] += 1
            if not encs[b]["has_interest"]:
                continue
            words = encs[b]["words"]
            max_gid = int(encs[b]["max_gid"])
            cats = token_cat_cpu[b]
            groups = group_cpu[b]
            labs = labels_cpu[b]
            m = mask_cpu[b]
            interest_gids = torch.unique(groups[m & ((cats == CAT_RW_ABS_CONTENT) | (cats == CAT_RW_COPIED))])
            for g0 in interest_gids.tolist():
                gid = int(g0)
                if gid < 0:
                    continue
                pos_all = ((groups == gid) & m).nonzero(as_tuple=False).view(-1).tolist()
                if not pos_all:
                    continue
                # A visible word-group should be category-pure; record violations if not.
                cat_vals = [int(cats[p]) for p in pos_all]
                cat_count = collections.Counter(cat_vals)
                cat_idx = cat_count.most_common(1)[0][0]
                selected_by_pool = (eid, gid) in pool_drop_positions
                if len(cat_count) > 1:
                    purity["mixed_category_masked_groups"] += 1
                if cat_idx == CAT_RW_ABS_CONTENT:
                    ev = make_event(
                        origin="absent_content", entry=entry, epoch=epoch, gid=gid,
                        positions=pos_all, label_ids=[int(labs[p]) for p in pos_all],
                        token_freq=token_freq, words=words, max_gid=max_gid, n_stream=n,
                        selected_by_pool=False,
                    )
                    abs_events.append(ev)
                elif cat_idx == CAT_RW_COPIED:
                    ev = make_event(
                        origin="copied_content", entry=entry, epoch=epoch, gid=gid,
                        positions=pos_all, label_ids=[int(labs[p]) for p in pos_all],
                        token_freq=token_freq, words=words, max_gid=max_gid, n_stream=n,
                        selected_by_pool=selected_by_pool,
                    )
                    # copied-content pool excludes function-like copied rewrite words, as in research/228 controls
                    if ev["word_class"] in {"content", "number"}:
                        copied_content_events.append(ev)
                        if selected_by_pool:
                            pool_selected_copied_events.append(ev)
                    elif selected_by_pool:
                        purity["pool_selected_noncontent_copied_events"] += 1

        if args.progress_every > 0 and (start + len(entries)) % args.progress_every < args.batch_size:
            print(json.dumps({
                "event": "audit_progress",
                "stream_rows": start + len(entries),
                "abs_events": len(abs_events),
                "copied_content_events": len(copied_content_events),
                "pool_selected_events": len(pool_selected_copied_events),
                "elapsed_sec": round(time.time() - t0, 1),
            }), flush=True)

    meta = {
        "occurrence_count_stats": dict(collections.Counter(str(v) for v in occ_seen.values())),
        "purity_counters": dict(purity),
        "elapsed_collect_sec": round(time.time() - t0, 1),
    }
    return abs_events, copied_content_events, pool_selected_copied_events, meta


def choose_event_matches(abs_events: list[dict[str, Any]], copied_events: list[dict[str, Any]], seed: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_len: dict[int, list[int]] = collections.defaultdict(list)
    for j, e in enumerate(copied_events):
        by_len[int(e["bpe_len"])].append(j)
    rng = random.Random(seed)
    for js in by_len.values():
        rng.shuffle(js)
    used: set[int] = set()
    selected: list[dict[str, Any]] = []
    fallback = collections.Counter()
    score_sum = 0.0
    len_mismatch = 0

    targets = list(abs_events)
    targets.sort(key=lambda e: (-int(e["bpe_len"]), float(e["support_log_mean"]), stable_u64(e["event_key"])))

    def score(a: dict[str, Any], c: dict[str, Any]) -> float:
        s = 0.0
        s += 20000.0 * abs(int(a["bpe_len"]) - int(c["bpe_len"]))
        s += 90.0 * (0 if int(a["epoch"]) == int(c["epoch"]) else 1)
        s += 10.0 * abs(float(a["support_log_mean"]) - float(c["support_log_mean"]))
        s += 7.0 * abs(float(a["support_log_min"]) - float(c["support_log_min"]))
        s += 5.0 * abs(float(a["seq_token_mid"]) - float(c["seq_token_mid"]))
        s += 4.0 * abs(float(a["rel_group_pos"]) - float(c["rel_group_pos"]))
        s += 1.0 * abs(float(a["row_order_frac"]) - float(c["row_order_frac"]))
        return s

    def candidate_pool(a: dict[str, Any]) -> tuple[str, list[int]]:
        L = int(a["bpe_len"])
        for radius in [0, 1, 2, 4, 8, 16, 999]:
            if radius == 0:
                lens = [L]
                label = "exact_len"
            elif radius == 999:
                lens = list(by_len.keys())
                label = "global_fallback"
            else:
                lens = [x for x in range(max(1, L - radius), L + radius + 1) if x in by_len]
                label = f"len_radius_{radius}"
            pool = [j for le in lens for j in by_len.get(le, []) if j not in used]
            if pool:
                return label, pool
        return "none", []

    for a in targets:
        label, pool = candidate_pool(a)
        if not pool:
            raise RuntimeError("ran out of realized copied-content events")
        same_epoch = [j for j in pool if int(copied_events[j]["epoch"]) == int(a["epoch"])]
        if same_epoch:
            pool = same_epoch
            label += "|same_epoch"
        best = min(pool, key=lambda j: (score(a, copied_events[j]), stable_u64(copied_events[j]["event_key"])))
        c = dict(copied_events[best])
        used.add(best)
        sc = score(a, c)
        score_sum += sc
        if int(a["bpe_len"]) != int(c["bpe_len"]):
            len_mismatch += 1
        fallback[label] += 1
        c["matched_abs_event_key"] = a["event_key"]
        c["matched_abs_stream_index"] = int(a["stream_index"])
        c["matched_abs_example_id"] = int(a["example_id"])
        c["matched_abs_group_id"] = int(a["group_id"])
        c["matched_abs_bpe_len"] = int(a["bpe_len"])
        c["matched_abs_epoch"] = int(a["epoch"])
        c["match_score"] = round(sc, 6)
        c["match_class"] = label
        c["feature_delta_to_abs"] = {
            "bpe_len": int(c["bpe_len"]) - int(a["bpe_len"]),
            "epoch": int(c["epoch"]) - int(a["epoch"]),
            "support_log_mean": round(float(c["support_log_mean"]) - float(a["support_log_mean"]), 6),
            "support_log_min": round(float(c["support_log_min"]) - float(a["support_log_min"]), 6),
            "seq_token_mid": round(float(c["seq_token_mid"]) - float(a["seq_token_mid"]), 6),
            "rel_group_pos": round(float(c["rel_group_pos"]) - float(a["rel_group_pos"]), 6),
            "row_order_frac": round(float(c["row_order_frac"]) - float(a["row_order_frac"]), 6),
        }
        selected.append(c)

    diag = {
        "selected_events": len(selected),
        "used_copied_events": len(used),
        "length_mismatch_events": len_mismatch,
        "fraction_exact_bpe_len": round(1.0 - len_mismatch / max(1, len(selected)), 6),
        "match_class_counts": dict(fallback),
        "mean_match_score": round(score_sum / max(1, len(selected)), 6),
    }
    return selected, diag


def write_jsonl(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stream", default=DEFAULT_STREAM)
    ap.add_argument("--annotation", default=DEFAULT_ANNOTATION)
    ap.add_argument("--selection", default=DEFAULT_SELECTION)
    ap.add_argument("--tokenizer_path", default=DEFAULT_TOKENIZER)
    ap.add_argument("--output_dir", default=DEFAULT_OUT)
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--max_seq_length", type=int, default=256)
    ap.add_argument("--mask_prob", type=float, default=0.15)
    ap.add_argument("--train_rng_seed", type=int, default=43023)
    ap.add_argument("--match_seed", type=int, default=22943023)
    ap.add_argument("--progress_every", type=int, default=50000)
    args = ap.parse_args()

    t0 = time.time()
    out = pathlib.Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    stream_path = pathlib.Path(args.stream)
    ann_path = pathlib.Path(args.annotation)
    sel_path = pathlib.Path(args.selection)

    print(json.dumps({"event": "load_annotations"}), flush=True)
    annotations, row_to_eid, word_cats_str, changed_eids = load_annotations(ann_path)
    pool_drop_positions = load_pool_selection(sel_path, row_to_eid)
    print(json.dumps({"event": "load_stream"}), flush=True)
    stream = load_stream(stream_path)
    if len(stream) != 647400:
        print(json.dumps({"event": "warning_stream_len", "len": len(stream)}), flush=True)
    if not torch.cuda.is_available() and str(args.device).startswith("cuda"):
        raise RuntimeError("CUDA requested but not available; exact trainer WWM schedule uses CUDA generator")
    tokenizer = train228.make_portable_tokenizer(args.tokenizer_path)
    cache = EncoderCache(tokenizer, annotations, args.max_seq_length)
    print(json.dumps({"event": "build_token_freq", "stream_rows": len(stream), "unique_annotations": len(annotations)}), flush=True)
    token_freq = build_token_freq(stream, cache)
    print(json.dumps({"event": "collect_realized_events", "token_freq_ids": len(token_freq)}), flush=True)
    abs_events, copied_events, pool_selected_events, meta = collect_realized_events(args, stream, cache, pool_drop_positions, token_freq)

    print(json.dumps({"event": "choose_event_matches", "abs_events": len(abs_events), "copied_events": len(copied_events)}), flush=True)
    eventmatched, match_diag = choose_event_matches(abs_events, copied_events, args.match_seed)

    abs_summary = summarize(abs_events)
    pool_summary = summarize(pool_selected_events)
    eventmatched_summary = summarize(eventmatched)
    copied_pool_summary = summarize(copied_events)

    # Save full scientific artifacts.
    write_jsonl(out / "realized_absent_content_events.jsonl", abs_events)
    write_jsonl(out / "realized_pool_selected_copied_events.jsonl", pool_selected_events)
    write_jsonl(out / "realized_eventmatched_copied_events.jsonl", eventmatched)
    # Trainer-ready event map: stream index -> group IDs.
    pos_map: dict[str, list[int]] = collections.defaultdict(list)
    for e in eventmatched:
        pos_map[str(int(e["stream_index"]))].append(int(e["group_id"]))
    pos_map = {k: sorted(set(v)) for k, v in pos_map.items()}
    (out / "eventmatched_drop_group_map_by_stream_index.json").write_text(json.dumps(pos_map, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    audit = {
        "status": "REALIZED_WWM_SELECTION_AUDIT",
        "meaning": "Exact-schedule audit of source-absent versus copied-content deletions under the historical packed 100M deterministic WWM schedule. Pool-level research selection is compared to an event-level rematch.",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "inputs": {
            "stream": args.stream,
            "stream_sha256": sha256_file(stream_path),
            "annotation": args.annotation,
            "annotation_sha256": sha256_file(ann_path),
            "pool_selection": args.selection,
            "pool_selection_sha256": sha256_file(sel_path),
            "tokenizer": args.tokenizer_path,
            "device_for_wwm_rng": args.device,
            "batch_size": args.batch_size,
            "max_seq_length": args.max_seq_length,
            "mask_prob": args.mask_prob,
            "train_rng_seed": args.train_rng_seed,
            "match_seed": args.match_seed,
        },
        "stream_rows": len(stream),
        "changed_example_ids": len(changed_eids),
        "pool_drop_positions": len(pool_drop_positions),
        "collection_meta": meta,
        "source_absent_content_realized": abs_summary,
        "copied_content_realized_pool": copied_pool_summary,
        "pool_selected_copied_realized": pool_summary,
        "eventmatched_copied_realized": eventmatched_summary,
        "pool_selected_minus_abs": mean_delta(abs_summary, pool_summary),
        "eventmatched_minus_abs": mean_delta(abs_summary, eventmatched_summary),
        "eventmatch_diagnostics": match_diag,
        "files": {
            "abs_events": str(out / "realized_absent_content_events.jsonl"),
            "pool_selected_copied_events": str(out / "realized_pool_selected_copied_events.jsonl"),
            "eventmatched_copied_events": str(out / "realized_eventmatched_copied_events.jsonl"),
            "eventmatched_drop_group_map_by_stream_index": str(out / "eventmatched_drop_group_map_by_stream_index.json"),
        },
        "elapsed_sec": round(time.time() - t0, 1),
    }
    (out / "realized_wwm_selection_audit.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": audit["status"],
        "out": str(out / "realized_wwm_selection_audit.json"),
        "abs_events": abs_summary["n_events"],
        "abs_pieces": abs_summary["piece_total"],
        "pool_selected_events": pool_summary["n_events"],
        "pool_selected_pieces": pool_summary["piece_total"],
        "pool_minus_abs": audit["pool_selected_minus_abs"],
        "eventmatched_events": eventmatched_summary["n_events"],
        "eventmatched_pieces": eventmatched_summary["piece_total"],
        "eventmatched_minus_abs": audit["eventmatched_minus_abs"],
        "eventmatch_diag": match_diag,
        "elapsed_sec": audit["elapsed_sec"],
    }, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
