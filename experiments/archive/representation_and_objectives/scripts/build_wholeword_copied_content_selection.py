#!/usr/bin/env python3
"""research: build a whole-word copied-content deletion control.

The research copied-drop control removed an equal number of copied token pieces, but often
removed only part of a multi-piece copied word.  This script constructs a stronger control
for the same research own-visible 20M training interface: every selected control unit is a
whole copied-content target word group.  The selected copied-content groups are matched to
all source-absent-content target word groups as closely as possible in

  * total deleted BPE pieces / wordpiece length,
  * target-token support under the exact realized 20M input stream,
  * compact-side position,
  * epoch and identity-level deterministic-WWM exposure count.

It writes selected copied-content groups and an audit.  It does not train a model.
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

import crossview_deberta_trainer_v2 as base  # noqa: E402

DEFAULT_DATA = "experiments/archive/representation_and_objectives/data/crossview_data_v2/crossview_consolidated_v2.jsonl"
DEFAULT_TOKENIZER = "experiments/archive/representation_and_objectives/training/runs/gc_compact_repeat_reinvest_16k_seed43022_r2/hf_model/chck_1M"
DEFAULT_OUT = "experiments/archive/representation_and_objectives/data/wholeword_copied_content_selection"

SPECIAL_ORIGINS = {"absent_content", "copied_content"}


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


def summarize_groups(groups: list[dict[str, Any]]) -> dict[str, Any]:
    if not groups:
        return {"n_groups": 0}
    out: dict[str, Any] = {
        "n_groups": len(groups),
        "piece_total": int(sum(g["bpe_len"] for g in groups)),
        "epoch_counts": dict(collections.Counter(str(g["epoch"]) for g in groups)),
        "identity_exposure_counts": dict(collections.Counter(str(g["identity_exposure_count"]) for g in groups)),
        "bpe_len_counts": dict(sorted(collections.Counter(str(g["bpe_len"]) for g in groups).items(), key=lambda kv: int(kv[0]))),
    }
    for key in [
        "bpe_len",
        "support_log_mean",
        "support_log_min",
        "support_min",
        "support_mean",
        "rel_word_pos",
        "rel_token_mid",
        "seq_token_mid",
        "row_order_frac",
    ]:
        out[key] = q_stats([float(g[key]) for g in groups])
    return out


def feature_deltas(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key in [
        "bpe_len",
        "support_log_mean",
        "support_log_min",
        "support_min",
        "support_mean",
        "rel_word_pos",
        "rel_token_mid",
        "seq_token_mid",
        "row_order_frac",
    ]:
        av = a.get(key, {}).get("mean")
        bv = b.get(key, {}).get("mean")
        if av is not None and bv is not None:
            out[f"selected_minus_abs_{key}_mean"] = round(float(bv) - float(av), 6)
    out["selected_minus_abs_piece_total"] = int(b.get("piece_total", 0) - a.get("piece_total", 0))
    out["selected_minus_abs_group_count"] = int(b.get("n_groups", 0) - a.get("n_groups", 0))
    return out


def iter_inventory(examples: list[dict[str, Any]], tok, args) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[int, int], dict[str, Any]]:
    """Enumerate realized rewrite-side masked target word groups and realized input-token support."""
    full = base.PairAwareDataset(examples, tok, args.max_seq_length, "own", False, args.mask_prob, args.mask_seed)
    token_freq: collections.Counter[int] = collections.Counter()
    raw_groups: list[dict[str, Any]] = []
    identity_counts: collections.Counter[tuple[str, str, int]] = collections.Counter()
    origin_piece_counts: collections.Counter[str] = collections.Counter()
    origin_group_counts: collections.Counter[str] = collections.Counter()
    skipped_pair = 0
    t_loop = time.time()
    half = args.max_seq_length // 2
    special_ids = set(int(x) for x in tok.all_special_ids)

    for idx in range(len(full)):
        item = full[idx]
        ids = item["input_ids"]
        att = item["attention_mask"]
        labs = item["labels"]
        # support under the exact realized input stream, excluding pads/specials
        for tid in ids[att.bool()].tolist():
            it = int(tid)
            if it not in special_ids:
                token_freq[it] += 1

        ex = examples[idx]
        if ex.get("type") != "pair":
            continue
        src = ex["source_text"]
        rw = ex["own_rewrite"]
        origins = ex["own_rewrite_origins_true"]
        rewrite_id = str(ex.get("own_rewrite_pair_id", ex.get("source_pair_id", ex.get("pair_id", idx))))
        source_pair_id = str(ex.get("source_pair_id", ex.get("pair_id", idx)))
        bnd = int(item["boundary"])
        rw_len = int(item["rewrite_len"])
        # Re-tokenize only the rewrite segment to recover word_ids exactly as the base dataset does.
        se = tok(src, add_special_tokens=False, truncation=True, max_length=half)
        # guard exact boundary identity; if this ever fails, use item boundary but record a warning.
        if len(se["input_ids"]) != bnd:
            skipped_pair += 1
        re = tok(rw, add_special_tokens=False, truncation=True, max_length=args.max_seq_length - bnd)
        rw_wids = re.word_ids(batch_index=0) if hasattr(re, "word_ids") else list(range(len(re["input_ids"])))
        rw_wids = rw_wids[:rw_len]

        groups: dict[tuple[int, str], dict[str, Any]] = {}
        for rel, wid in enumerate(rw_wids):
            if wid is None or wid >= len(origins):
                continue
            pos = bnd + rel
            if pos >= args.max_seq_length or int(labs[pos]) == -100:
                continue
            origin = str(origins[int(wid)])
            if origin not in SPECIAL_ORIGINS:
                continue
            key = (int(wid), origin)
            if key not in groups:
                word_list = rw.split()
                groups[key] = {
                    "group_key": f"ex{idx}|wid{int(wid)}|{origin}",
                    "example_index": idx,
                    "source_pair_id": source_pair_id,
                    "rewrite_id": rewrite_id,
                    "identity_key": f"{rewrite_id}|wid={int(wid)}|origin={origin}",
                    "word_id": int(wid),
                    "word_text": word_list[int(wid)] if int(wid) < len(word_list) else "",
                    "origin": origin,
                    "epoch": int(ex.get("_epoch", -1)),
                    "positions": [],
                    "token_ids": [],
                    "boundary": bnd,
                    "rewrite_len_bpe": rw_len,
                    "rewrite_words": len(origins),
                    "row_order_frac": idx / max(1, len(full) - 1),
                }
            groups[key]["positions"].append(pos)
            groups[key]["token_ids"].append(int(labs[pos]))
        for g in groups.values():
            if not g["positions"]:
                continue
            g["positions"] = sorted(g["positions"])
            g["bpe_len"] = len(g["positions"])
            g["token_start"] = min(g["positions"])
            g["token_end_exclusive"] = max(g["positions"]) + 1
            mid = 0.5 * (g["token_start"] + g["token_end_exclusive"] - 1)
            g["seq_token_mid"] = mid / max(1, args.max_seq_length - 1)
            g["rel_token_mid"] = (mid - bnd) / max(1, rw_len - 1)
            g["rel_word_pos"] = int(g["word_id"]) / max(1, len(origins) - 1)
            raw_groups.append(g)
            identity_counts[(g["origin"], g["rewrite_id"], int(g["word_id"]))] += 1
            origin_piece_counts[g["origin"]] += int(g["bpe_len"])
            origin_group_counts[g["origin"]] += 1
        if args.progress_every > 0 and (idx + 1) % args.progress_every == 0:
            print(json.dumps({"event": "inventory_progress", "examples": idx + 1, "groups": len(raw_groups), "elapsed_sec": round(time.time() - t_loop, 1)}), flush=True)

    # Add identity-level selected count and support features.
    for g in raw_groups:
        g["identity_exposure_count"] = int(identity_counts[(g["origin"], g["rewrite_id"], int(g["word_id"]))])
        supports = [int(token_freq.get(int(t), 0)) for t in g["token_ids"]]
        g["support_min"] = min(supports) if supports else 0
        g["support_mean"] = sum(supports) / max(1, len(supports))
        logs = [math.log1p(x) for x in supports]
        g["support_log_min"] = min(logs) if logs else 0.0
        g["support_log_mean"] = sum(logs) / max(1, len(logs))

    meta = {
        "examples": len(examples),
        "skipped_pair_boundary_mismatch": skipped_pair,
        "token_freq_unique_ids": len(token_freq),
        "origin_piece_counts": dict(origin_piece_counts),
        "origin_group_counts": dict(origin_group_counts),
        "elapsed_inventory_sec": round(time.time() - t_loop, 1),
    }
    return raw_groups, [g for g in raw_groups if g["origin"] == "absent_content"], token_freq, meta


def choose_matches(abs_groups: list[dict[str, Any]], copied_groups: list[dict[str, Any]], seed: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    # Group candidate indices by BPE length.  Exact BPE length matching is strongly preferred because
    # it makes the deleted piece mass equal when possible.
    by_len: dict[int, list[int]] = collections.defaultdict(list)
    for j, g in enumerate(copied_groups):
        by_len[int(g["bpe_len"])].append(j)
    rng = random.Random(seed)
    for lst in by_len.values():
        rng.shuffle(lst)

    # Hard examples first: long / rare absent groups have fewer plausible copied controls.
    targets = list(abs_groups)
    targets.sort(key=lambda g: (-int(g["bpe_len"]), float(g["support_log_mean"]), stable_u64(g["group_key"])))
    used: set[int] = set()
    selected_records: list[dict[str, Any]] = []
    len_mismatch = 0
    fallback_counts: collections.Counter[str] = collections.Counter()
    score_sum = 0.0

    # Feature scales chosen in raw units.  The large penalties keep exposure/epoch matched when feasible;
    # support and compact-side position decide among many same-length candidates.
    def score(a: dict[str, Any], c: dict[str, Any]) -> float:
        s = 0.0
        s += 10000.0 * abs(int(a["bpe_len"]) - int(c["bpe_len"]))
        s += 70.0 * (0 if int(a["epoch"]) == int(c["epoch"]) else 1)
        s += 45.0 * abs(int(a["identity_exposure_count"]) - int(c["identity_exposure_count"]))
        s += 8.0 * abs(float(a["support_log_mean"]) - float(c["support_log_mean"]))
        s += 5.0 * abs(float(a["support_log_min"]) - float(c["support_log_min"]))
        s += 4.0 * abs(float(a["rel_word_pos"]) - float(c["rel_word_pos"]))
        s += 3.0 * abs(float(a["rel_token_mid"]) - float(c["rel_token_mid"]))
        s += 1.0 * abs(float(a["row_order_frac"]) - float(c["row_order_frac"]))
        return s

    def candidate_pool_for(a: dict[str, Any]) -> tuple[str, list[int]]:
        L = int(a["bpe_len"])
        # Try exact length first; if exhausted, allow nearest lengths.  The inventory is large enough that
        # exact length should cover almost all targets; the fallback is only fail-safe.
        for radius in [0, 1, 2, 3, 4, 8, 16, 999]:
            lens: list[int]
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
        label, pool = candidate_pool_for(a)
        if not pool:
            raise RuntimeError("ran out of copied-content target groups")
        # First filter to same epoch/exposure if feasible, without making it a hard requirement.
        same_epoch_exposure = [j for j in pool if int(copied_groups[j]["epoch"]) == int(a["epoch"]) and int(copied_groups[j]["identity_exposure_count"]) == int(a["identity_exposure_count"])]
        if same_epoch_exposure:
            pool = same_epoch_exposure
            label += "|same_epoch_exposure"
        else:
            same_epoch = [j for j in pool if int(copied_groups[j]["epoch"]) == int(a["epoch"])]
            if same_epoch:
                pool = same_epoch
                label += "|same_epoch"
        best = min(pool, key=lambda j: (score(a, copied_groups[j]), stable_u64(copied_groups[j]["group_key"])))
        c = copied_groups[best]
        used.add(best)
        sc = score(a, c)
        score_sum += sc
        if int(a["bpe_len"]) != int(c["bpe_len"]):
            len_mismatch += 1
        fallback_counts[label] += 1
        rec = dict(c)
        rec["matched_abs_group_key"] = a["group_key"]
        rec["matched_abs_bpe_len"] = int(a["bpe_len"])
        rec["matched_abs_epoch"] = int(a["epoch"])
        rec["matched_abs_identity_exposure_count"] = int(a["identity_exposure_count"])
        rec["match_score"] = round(sc, 6)
        rec["match_class"] = label
        rec["feature_delta_to_abs"] = {
            "bpe_len": int(c["bpe_len"]) - int(a["bpe_len"]),
            "epoch": int(c["epoch"]) - int(a["epoch"]),
            "identity_exposure_count": int(c["identity_exposure_count"]) - int(a["identity_exposure_count"]),
            "support_log_mean": round(float(c["support_log_mean"]) - float(a["support_log_mean"]), 6),
            "support_log_min": round(float(c["support_log_min"]) - float(a["support_log_min"]), 6),
            "rel_word_pos": round(float(c["rel_word_pos"]) - float(a["rel_word_pos"]), 6),
            "rel_token_mid": round(float(c["rel_token_mid"]) - float(a["rel_token_mid"]), 6),
            "row_order_frac": round(float(c["row_order_frac"]) - float(a["row_order_frac"]), 6),
        }
        selected_records.append(rec)

    diagnostics = {
        "selected_groups": len(selected_records),
        "used_candidate_groups": len(used),
        "length_mismatch_groups": len_mismatch,
        "fraction_exact_bpe_len": round(1.0 - len_mismatch / max(1, len(selected_records)), 6),
        "match_class_counts": dict(fallback_counts),
        "mean_match_score": round(score_sum / max(1, len(selected_records)), 6),
    }
    return selected_records, diagnostics


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=DEFAULT_DATA)
    ap.add_argument("--tokenizer_path", default=DEFAULT_TOKENIZER)
    ap.add_argument("--output_dir", default=DEFAULT_OUT)
    ap.add_argument("--max_word_exposure", type=int, default=20_000_000)
    ap.add_argument("--max_seq_length", type=int, default=256)
    ap.add_argument("--mask_prob", type=float, default=0.15)
    ap.add_argument("--mask_seed", type=int, default=430230221)
    ap.add_argument("--seed", type=int, default=43)
    ap.add_argument("--match_seed", type=int, default=22643023)
    ap.add_argument("--progress_every", type=int, default=20000)
    args = ap.parse_args()

    t0 = time.time()
    out = pathlib.Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    tok = base.make_portable_tokenizer(args.tokenizer_path)
    pool = base.load_pool(args.data)
    examples, aw, ep = base.build_stream(pool, args.max_word_exposure, "own", args.seed)
    if aw != args.max_word_exposure:
        raise RuntimeError(f"stream filled {aw}, expected {args.max_word_exposure}")

    raw_groups, abs_groups, token_freq, inv_meta = iter_inventory(examples, tok, args)
    copied_groups = [g for g in raw_groups if g["origin"] == "copied_content"]
    if not abs_groups:
        raise RuntimeError("no absent_content groups found")
    if not copied_groups:
        raise RuntimeError("no copied_content groups found")
    selected, diag = choose_matches(abs_groups, copied_groups, args.match_seed)

    abs_summary = summarize_groups(abs_groups)
    copied_pool_summary = summarize_groups(copied_groups)
    selected_summary = summarize_groups(selected)
    deltas = feature_deltas(abs_summary, selected_summary)
    if selected_summary["piece_total"] != abs_summary["piece_total"]:
        print(json.dumps({"event": "warning_piece_mass_mismatch", "abs": abs_summary["piece_total"], "selected": selected_summary["piece_total"]}), flush=True)

    selected_jsonl = out / "selected_copied_content_wholeword_groups.jsonl"
    with selected_jsonl.open("w", encoding="utf-8") as f:
        for g in selected:
            # keep the positions and enough features to audit/replay; omit no scientific fields.
            f.write(json.dumps(g, ensure_ascii=False) + "\n")
    abs_jsonl = out / "absent_content_target_groups.jsonl"
    with abs_jsonl.open("w", encoding="utf-8") as f:
        for g in abs_groups:
            f.write(json.dumps(g, ensure_ascii=False) + "\n")

    # A compact position map for fast trainer loading.
    pos_map: dict[str, list[int]] = collections.defaultdict(list)
    for g in selected:
        for p in g["positions"]:
            pos_map[str(g["example_index"])].append(int(p))
    pos_map_out = {k: sorted(set(v)) for k, v in pos_map.items()}
    pos_json = out / "drop_position_map.json"
    pos_json.write_text(json.dumps(pos_map_out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    audit = {
        "status": "WHOLEWORD_COPIED_CONTENT_SELECTION_BUILT",
        "meaning": "Whole copied-content target-word groups selected to match all source-absent-content target groups under the exact research own-visible 20M interface. This is a stronger control than research token-piece copied deletion.",
        "inputs": {
            "data": args.data,
            "data_sha256": sha256_file(pathlib.Path(args.data)),
            "tokenizer": args.tokenizer_path,
            "max_word_exposure": args.max_word_exposure,
            "word_exposure_realized": aw,
            "epochs_realized": ep,
            "seed": args.seed,
            "mask_seed": args.mask_seed,
            "mask_prob": args.mask_prob,
            "match_seed": args.match_seed,
            "max_seq_length": args.max_seq_length,
        },
        "inventory_meta": inv_meta,
        "absent_content_groups": abs_summary,
        "copied_content_pool": copied_pool_summary,
        "selected_copied_content_wholeword_groups": selected_summary,
        "match_diagnostics": diag,
        "selected_minus_abs_feature_deltas": deltas,
        "files": {
            "selected_groups_jsonl": str(selected_jsonl),
            "absent_groups_jsonl": str(abs_jsonl),
            "drop_position_map_json": str(pos_json),
        },
        "scientific_use": [
            "Train exactly one additional 20M arm with the same research own-visible text/interface/masks/init, deleting these whole copied-content target groups.",
            "Compare drop_abs_content against this whole-word copied-content control on the fixed compact-side source-absent denoising readout before interpreting target-type specificity.",
            "The pending cheap7 result should be read only as early external coupling; the compact-view effect matured late, so 20M task scores should not decide route survival alone."
        ],
        "elapsed_sec": round(time.time() - t0, 1),
    }
    audit_path = out / "wholeword_copied_content_selection_audit.json"
    audit_path.write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": audit["status"],
        "out": str(audit_path),
        "abs_groups": abs_summary["n_groups"],
        "abs_pieces": abs_summary["piece_total"],
        "candidate_groups": copied_pool_summary["n_groups"],
        "selected_groups": selected_summary["n_groups"],
        "selected_pieces": selected_summary["piece_total"],
        "piece_delta": selected_summary["piece_total"] - abs_summary["piece_total"],
        "fraction_exact_bpe_len": diag["fraction_exact_bpe_len"],
        "deltas": deltas,
        "elapsed_sec": audit["elapsed_sec"],
    }, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
