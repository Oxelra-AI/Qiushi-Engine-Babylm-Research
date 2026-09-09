#!/usr/bin/env python3
"""research: official Entity relevance/recency cuts and training-window design check.

This CPU script sharpens the copy-vs-state-reading fork using existing official
Entity predictions. It verifies whether Entity `numops` is actually the number of
updates touching the queried box, separates total operation count / context length /
post-relevant-operation distance, and checks whether the MAX VIEW/REPEAT changed
rows concatenate each source with its own companion inside the training examples.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import csv
import hashlib
import json
import math
import pathlib
import re
import statistics
import time
from collections import defaultdict
from typing import Any

from transformers import AutoTokenizer


def find_root() -> pathlib.Path:
    return _PUBLIC_ROOT


ROOT = find_root()
WS = ROOT / "experiments/archive" / 'relation_learning'
OUT = WS / "data" / "entity_relevant_update_analysis"
ENTITY_DIR = ROOT / "experiments/archive" / 'representation_and_objectives' / "data" / "pristine_official_coordinate" / "babylm-eval" / "strict" / "evaluation_data" / "full_eval" / "entity_tracking"
PRED_ITEM_ROWS = WS / "data" / "entity_error_comps_readout" / "entity_prediction_item_rows.csv"
PAIR_PATH = ROOT / "experiments/archive" / 'frontier_consolidation' / "data" / "dose_distribution_select" / "selected_matched_max_pairs.jsonl"
POOL_DIR = ROOT / "experiments/archive" / 'frontier_consolidation' / "data" / "dose_2p64x_rowholdout_pools"
VIEW_STREAM = POOL_DIR / "compact_view_dose2p64x_10M.jsonl"
REPEAT_STREAM = POOL_DIR / "compact_repeat_dose2p64x_10M.jsonl"
VIEW_META = POOL_DIR / "compact_view_dose2p64x_changed_block_rows_meta.jsonl"
REPEAT_META = POOL_DIR / "compact_repeat_dose2p64x_changed_block_rows_meta.jsonl"
TOKENIZER_ROOT = ROOT / "experiments/archive" / 'frontier_consolidation' / "training" / "runs" / "full_p2c_c2p_abs_view_dose2p64x_matched_rowholdout_deberta100M_seed43022" / "hf_model"
CKS = ["chck_80M", "chck_90M", "chck_100M"]


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


def norm(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s.strip(" .")


def wc(text: str) -> int:
    return len(text.split())


def split_entity_prefix(prefix: str) -> tuple[str, list[str], str]:
    parts = [x.strip() for x in re.split(r"(?<=\.)\s+", prefix.strip()) if x.strip()]
    if len(parts) < 2:
        return prefix.strip(), [], ""
    return parts[0], parts[1:-1], parts[-1]


def parse_query_box(query: str) -> int | None:
    m = re.search(r"Box\s+(\d+)\s+contains\s*$", query.strip())
    return int(m.group(1)) if m else None


def parse_initial_contents(initial_sentence: str) -> dict[int, str]:
    s = initial_sentence.strip()
    if s.endswith("."):
        s = s[:-1]
    out: dict[int, str] = {}
    for m in re.finditer(r"Box\s+(\d+)\s+contains\s+(.*?)(?=,\s*Box\s+\d+\s+contains\s+|$)", s):
        out[int(m.group(1))] = m.group(2).strip()
    return out


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
    op_low = op.lower().strip()
    if op_low.startswith("move"):
        return b["from_box"] == qbox or b["to_box"] == qbox
    if op_low.startswith("remove"):
        return b["from_box"] == qbox
    if op_low.startswith("put"):
        return b["into_box"] == qbox
    return qbox in {v for v in b.values() if v is not None}


def quantile_bins(values: list[int], qs: int = 4) -> list[float]:
    vals = sorted(values)
    if not vals:
        return []
    cuts = []
    for q in range(1, qs):
        idx = int(round(q * (len(vals) - 1) / qs))
        cuts.append(vals[idx])
    return cuts


def assign_bin(x: int, cuts: list[float]) -> str:
    for i, c in enumerate(cuts, start=1):
        if x <= c:
            return f"q{i}"
    return f"q{len(cuts)+1}"


def load_entity_metadata(tokenizer) -> dict[tuple[str, int], dict[str, Any]]:
    tmp: list[dict[str, Any]] = []
    for fn in ["regular.jsonl", "ambiref.jsonl", "move_contents.jsonl"]:
        typ = fn[:-6]
        counters: dict[int, int] = defaultdict(int)
        for obj in read_jsonl(ENTITY_DIR / fn):
            if any("nothing" in str(o).lower() for o in obj.get("options", [])):
                continue
            reported = int(obj["numops"])
            uid = f"{typ}_{reported}_ops"
            item_index = counters[reported]
            counters[reported] += 1
            initial, ops, query = split_entity_prefix(obj["input_prefix"])
            qbox = parse_query_box(query)
            relevant_flags = [op_relevant_to_query(op, qbox) for op in ops]
            rel_updates = sum(int(x) for x in relevant_flags)
            last_rel_idx = max([i for i, x in enumerate(relevant_flags) if x], default=-1)
            ops_after_last_rel = (len(ops) - 1 - last_rel_idx) if last_rel_idx >= 0 else len(ops)
            ops_before_first_rel = min([i for i, x in enumerate(relevant_flags) if x], default=len(ops))
            init_map = parse_initial_contents(initial)
            stale = init_map.get(qbox) if qbox is not None else None
            options = [str(x) for x in obj.get("options", [])]
            stale_idx = None
            if stale is not None:
                for oi, opt in enumerate(options):
                    if norm(opt) == norm(stale):
                        stale_idx = oi
                        break
            # Untruncated and truncated token lengths for context-length controls.
            enc_full = tokenizer(obj["input_prefix"], add_special_tokens=True, truncation=False)
            enc_trunc = tokenizer(obj["input_prefix"], add_special_tokens=True, truncation=True, max_length=256)
            tmp.append({
                "uid": uid,
                "entity_type": typ,
                "reported_numops": reported,
                "item_index": item_index,
                "sample_id": obj.get("sample_id"),
                "example_id": obj.get("example_id"),
                "query_box": qbox if qbox is not None else -1,
                "total_ops": len(ops),
                "relevant_updates": rel_updates,
                "irrelevant_ops": len(ops) - rel_updates,
                "ops_after_last_relevant": ops_after_last_rel,
                "ops_before_first_relevant": ops_before_first_rel,
                "last_relevant_is_final_op": int(last_rel_idx == len(ops) - 1 and last_rel_idx >= 0),
                "prefix_words": wc(obj["input_prefix"]),
                "prefix_chars": len(obj["input_prefix"]),
                "prefix_tokens_untruncated": len(enc_full["input_ids"]),
                "prefix_tokens_trunc256": len(enc_trunc["input_ids"]),
                "prefix_truncated_at_256": int(len(enc_full["input_ids"]) > 256),
                "stale_initial": stale or "",
                "stale_available": int(stale_idx is not None),
                "stale_is_gold": int(stale_idx == 0) if stale_idx is not None else 0,
                "gold": options[0] if options else "",
            })
    cuts_words = quantile_bins([int(x["prefix_words"]) for x in tmp], 4)
    cuts_tokens = quantile_bins([int(x["prefix_tokens_untruncated"]) for x in tmp], 4)
    out = {}
    for r in tmp:
        r["prefix_words_bin"] = assign_bin(int(r["prefix_words"]), cuts_words)
        r["prefix_tokens_bin"] = assign_bin(int(r["prefix_tokens_untruncated"]), cuts_tokens)
        out[(r["uid"], int(r["item_index"]))] = r
    return out


def repeat_source_words(source_text: str, n_words: int, salt: str) -> str:
    toks = source_text.split()
    if not toks or n_words <= 0:
        return ""
    h = int(hashlib.sha1(salt.encode("utf-8")).hexdigest()[:8], 16)
    start = h % len(toks)
    rot = toks[start:] + toks[:start]
    out: list[str] = []
    while len(out) < n_words:
        out.extend(rot[: n_words - len(out)])
    return " ".join(out)


def verify_training_design(tokenizer) -> dict[str, Any]:
    pair_map = {str(p["pair_id"]): p for p in read_jsonl(PAIR_PATH)}
    view_rows = list(read_jsonl(VIEW_STREAM))
    repeat_rows = list(read_jsonl(REPEAT_STREAM))
    view_meta = list(read_jsonl(VIEW_META))
    repeat_meta = list(read_jsonl(REPEAT_META))
    changed_n = len(view_meta)
    if len(repeat_meta) != changed_n:
        raise RuntimeError("view/repeat changed meta length mismatch")
    if len(view_rows) < changed_n or len(repeat_rows) < changed_n:
        raise RuntimeError("stream shorter than changed block")

    exact_view = 0
    exact_repeat = 0
    length_seq_same = 0
    row_token_lens_view = []
    row_token_lens_repeat = []
    row_fits_view = 0
    row_fits_repeat = 0
    pair_total = 0
    pair_source_then_companion = 0
    pair_text_fits_alone_view = 0
    pair_text_fits_alone_repeat = 0
    pair_row_visible_full_view = 0
    pair_row_visible_full_repeat = 0
    examples = []

    for i in range(changed_n):
        pid_list = [str(x) for x in view_meta[i].get("pair_ids", [])]
        if pid_list != [str(x) for x in repeat_meta[i].get("pair_ids", [])]:
            raise RuntimeError(f"pair id mismatch row {i}")
        v_segments = []
        r_segments = []
        for pid in pid_list:
            p = pair_map[pid]
            src = " ".join(str(p["source_text"]).split())
            rew = " ".join(str(p["rewrite_text"]).split())
            rep = repeat_source_words(src, int(p["rewrite_words"]), pid)
            v_segments.append(f"{src} {rew}".strip())
            r_segments.append(f"{src} {rep}".strip())
            pair_total += 1
            if wc(src) + wc(rew) == int(p["pair_words"]) and wc(src) + wc(rep) == int(p["pair_words"]):
                pair_source_then_companion += 1
            if len(tokenizer(f"{src} {rew}".strip(), add_special_tokens=True, truncation=False)["input_ids"]) <= 256:
                pair_text_fits_alone_view += 1
            if len(tokenizer(f"{src} {rep}".strip(), add_special_tokens=True, truncation=False)["input_ids"]) <= 256:
                pair_text_fits_alone_repeat += 1
        v_rebuilt = " ".join(v_segments)
        r_rebuilt = " ".join(r_segments)
        vtext = " ".join(str(view_rows[i]["text"]).split())
        rtext = " ".join(str(repeat_rows[i]["text"]).split())
        vok = int(vtext == v_rebuilt)
        rok = int(rtext == r_rebuilt)
        exact_view += vok
        exact_repeat += rok
        length_seq_same += int(int(view_rows[i]["words"]) == int(repeat_rows[i]["words"]) == int(view_meta[i]["words"]))
        vlen = len(tokenizer(vtext, add_special_tokens=True, truncation=False)["input_ids"])
        rlen = len(tokenizer(rtext, add_special_tokens=True, truncation=False)["input_ids"])
        row_token_lens_view.append(vlen)
        row_token_lens_repeat.append(rlen)
        row_fits_view += int(vlen <= 256)
        row_fits_repeat += int(rlen <= 256)
        # Conservative row-level full visibility: if whole packed row fits, every companion in it is visible.
        pair_row_visible_full_view += len(pid_list) if vlen <= 256 else 0
        pair_row_visible_full_repeat += len(pid_list) if rlen <= 256 else 0
        if i < 6:
            examples.append({
                "row_index": i,
                "pair_count": len(pid_list),
                "words": int(view_rows[i]["words"]),
                "view_rebuild_exact": vok,
                "repeat_rebuild_exact": rok,
                "view_tokens_with_special": vlen,
                "repeat_tokens_with_special": rlen,
                "view_fits_256": int(vlen <= 256),
                "repeat_fits_256": int(rlen <= 256),
                "view_text_prefix": vtext[:220],
                "repeat_text_prefix": rtext[:220],
            })
    def mean(xs):
        return float(statistics.mean(xs)) if xs else float("nan")
    def pct(a, b):
        return 100.0 * a / b if b else float("nan")
    return {
        "pair_path": rel(PAIR_PATH),
        "view_stream": rel(VIEW_STREAM),
        "repeat_stream": rel(REPEAT_STREAM),
        "view_row_meta": rel(VIEW_META),
        "repeat_row_meta": rel(REPEAT_META),
        "changed_rows": changed_n,
        "changed_pairs": pair_total,
        "view_rows_rebuilt_exact": exact_view,
        "repeat_rows_rebuilt_exact": exact_repeat,
        "row_length_sequence_equal_in_changed_block": int(length_seq_same == changed_n),
        "pair_source_then_own_companion_count": pair_source_then_companion,
        "pair_source_then_own_companion_pct": pct(pair_source_then_companion, pair_total),
        "pair_text_fits_256_alone_view_pct": pct(pair_text_fits_alone_view, pair_total),
        "pair_text_fits_256_alone_repeat_pct": pct(pair_text_fits_alone_repeat, pair_total),
        "packed_row_fits_256_view_pct": pct(row_fits_view, changed_n),
        "packed_row_fits_256_repeat_pct": pct(row_fits_repeat, changed_n),
        "packed_row_tokens_view_mean_max": [mean(row_token_lens_view), max(row_token_lens_view)],
        "packed_row_tokens_repeat_mean_max": [mean(row_token_lens_repeat), max(row_token_lens_repeat)],
        "pairs_in_fully_visible_packed_rows_view_pct": pct(pair_row_visible_full_view, pair_total),
        "pairs_in_fully_visible_packed_rows_repeat_pct": pct(pair_row_visible_full_repeat, pair_total),
        "examples": examples,
        "interpretation": "All changed rows are source+own-companion packets by construction; each individual source+companion pair fits the 256-token window, but packed rows can exceed 256 BPE tokens, so 'inside the training window' is exact at the pair design level and approximate at packed-row full-visibility level.",
    }


def load_prediction_augmented(meta: dict[tuple[str, int], dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    with PRED_ITEM_ROWS.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            key = (r["uid"], int(r["item_index"]))
            if key not in meta:
                raise RuntimeError(f"missing metadata for {key}")
            m = meta[key]
            out = dict(r)
            for k, v in m.items():
                if k not in out:
                    out[k] = v
            # normalize numeric fields
            for k in ["seed", "item_index", "correct", "pred_idx", "stale_available", "stale_is_gold", "pred_is_stale_initial", "wrong_and_stale_initial", "reported_numops", "total_ops", "relevant_updates", "irrelevant_ops", "ops_after_last_relevant", "ops_before_first_relevant", "last_relevant_is_final_op", "prefix_words", "prefix_tokens_untruncated", "prefix_truncated_at_256"]:
                if k in out:
                    out[k] = int(out[k])
            rows.append(out)
    return rows


def item_groups(r: dict[str, Any]) -> list[str]:
    relu = int(r["relevant_updates"])
    total = int(r["total_ops"])
    groups = [
        "ALL",
        f"reported_numops_{r['reported_numops']}",
        f"rel_updates_{relu}",
        f"total_ops_{total}",
        f"rel{relu}_total{total}",
        f"prefix_words_{r['prefix_words_bin']}",
        f"prefix_tokens_{r['prefix_tokens_bin']}",
        f"type_{r['entity_type']}",
    ]
    if relu == 0:
        groups.append(f"rel0_total_ops_{total}")
        groups.append(f"rel0_prefix_words_{r['prefix_words_bin']}")
    else:
        groups.append("rel_ge1")
        groups.append(f"rel_ge1_postrel_ops_{r['ops_after_last_relevant']}")
        groups.append(f"rel{relu}_postrel_ops_{r['ops_after_last_relevant']}")
        groups.append(f"rel{relu}_lastrel_final_{r['last_relevant_is_final_op']}")
    if int(r.get("stale_available", 0)) and not int(r.get("stale_is_gold", 0)):
        groups.append("stale_available_not_gold")
        groups.append(f"rel{relu}_stale_available_not_gold")
    return groups


def mean(xs: list[float]) -> float:
    xs = [x for x in xs if math.isfinite(x)]
    return float(statistics.mean(xs)) if xs else float("nan")


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        for g in item_groups(r):
            groups[(r["seed"], r["arm"], r["checkpoint"], g)].append(r)
    out = []
    for (seed, arm, ck, g), vals in sorted(groups.items()):
        wrong = [v for v in vals if not int(v["correct"])]
        stale_cand = [v for v in vals if int(v.get("stale_available", 0)) and not int(v.get("stale_is_gold", 0))]
        wrong_stale_cand = [v for v in stale_cand if not int(v["correct"])]
        out.append({
            "seed": seed,
            "arm": arm,
            "checkpoint": ck,
            "group": g,
            "n": len(vals),
            "accuracy_pct": 100.0 * sum(int(v["correct"]) for v in vals) / len(vals),
            "mean_reported_numops": mean([float(v["reported_numops"]) for v in vals]),
            "mean_relevant_updates": mean([float(v["relevant_updates"]) for v in vals]),
            "mean_total_ops": mean([float(v["total_ops"]) for v in vals]),
            "mean_prefix_words": mean([float(v["prefix_words"]) for v in vals]),
            "mean_prefix_tokens": mean([float(v["prefix_tokens_untruncated"]) for v in vals]),
            "stale_available_not_gold_n": len(stale_cand),
            "stale_pick_pct_among_wrong_stale_available_not_gold": 100.0 * sum(int(v["pred_is_stale_initial"]) for v in wrong_stale_cand) / len(wrong_stale_cand) if wrong_stale_cand else float("nan"),
            "stale_pick_pct_among_all_wrong": 100.0 * sum(int(v["pred_is_stale_initial"]) for v in wrong) / len(wrong) if wrong else float("nan"),
        })
    return out


def contrasts(summary: list[dict[str, Any]]) -> list[dict[str, Any]]:
    idx = {(int(r["seed"]), r["arm"], r["checkpoint"], r["group"]): r for r in summary}
    groups = sorted({r["group"] for r in summary})
    out = []
    for seed in [43022, 43122]:
        for ck in CKS:
            for g in groups:
                for name, a, b in [("VminusC", "V", "C"), ("VminusR", "V", "R"), ("CminusR", "C", "R"), ("RminusC", "R", "C"), ("RminusV", "R", "V")]:
                    ka = (seed, a, ck, g)
                    kb = (seed, b, ck, g)
                    if ka not in idx or kb not in idx:
                        continue
                    ra, rb = idx[ka], idx[kb]
                    out.append({
                        "seed": seed,
                        "checkpoint": ck,
                        "group": g,
                        "contrast": name,
                        "n": int(ra["n"]),
                        "delta_accuracy_pct_a_minus_b": float(ra["accuracy_pct"]) - float(rb["accuracy_pct"]),
                        "acc_a": float(ra["accuracy_pct"]),
                        "acc_b": float(rb["accuracy_pct"]),
                        "delta_stale_wrong_pct_a_minus_b": float(ra["stale_pick_pct_among_wrong_stale_available_not_gold"]) - float(rb["stale_pick_pct_among_wrong_stale_available_not_gold"]),
                        "stale_wrong_a": float(ra["stale_pick_pct_among_wrong_stale_available_not_gold"]),
                        "stale_wrong_b": float(rb["stale_pick_pct_among_wrong_stale_available_not_gold"]),
                        "mean_reported_numops": float(ra["mean_reported_numops"]),
                        "mean_relevant_updates": float(ra["mean_relevant_updates"]),
                        "mean_total_ops": float(ra["mean_total_ops"]),
                        "mean_prefix_words": float(ra["mean_prefix_words"]),
                    })
    return out


def late_summary(con_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
    for r in con_rows:
        groups[(r["seed"], r["group"], r["contrast"])].append(r)
    out = []
    for (seed, g, contrast), vals in sorted(groups.items()):
        out.append({
            "seed": seed,
            "group": g,
            "contrast": contrast,
            "n_checkpoints": len(vals),
            "n": int(vals[0]["n"]),
            "late_mean_delta_accuracy_pct_a_minus_b": mean([float(v["delta_accuracy_pct_a_minus_b"]) for v in vals]),
            "late_mean_acc_a": mean([float(v["acc_a"]) for v in vals]),
            "late_mean_acc_b": mean([float(v["acc_b"]) for v in vals]),
            "late_mean_delta_stale_wrong_pct_a_minus_b": mean([float(v["delta_stale_wrong_pct_a_minus_b"]) for v in vals]),
            "mean_reported_numops": mean([float(v["mean_reported_numops"]) for v in vals]),
            "mean_relevant_updates": mean([float(v["mean_relevant_updates"]) for v in vals]),
            "mean_total_ops": mean([float(v["mean_total_ops"]) for v in vals]),
            "mean_prefix_words": mean([float(v["mean_prefix_words"]) for v in vals]),
        })
    return out


def write_csv(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("\n", encoding="utf-8")
        return
    fields = sorted(set().union(*(r.keys() for r in rows)))
    preferred = [
        "seed", "arm", "checkpoint", "group", "contrast", "n", "n_checkpoints",
        "accuracy_pct", "delta_accuracy_pct_a_minus_b", "late_mean_delta_accuracy_pct_a_minus_b", "acc_a", "acc_b", "late_mean_acc_a", "late_mean_acc_b",
        "reported_numops", "relevant_updates", "total_ops", "irrelevant_ops", "ops_after_last_relevant", "prefix_words", "prefix_tokens_untruncated",
        "mean_reported_numops", "mean_relevant_updates", "mean_total_ops", "mean_prefix_words", "mean_prefix_tokens",
        "stale_available_not_gold_n", "stale_pick_pct_among_wrong_stale_available_not_gold", "delta_stale_wrong_pct_a_minus_b", "late_mean_delta_stale_wrong_pct_a_minus_b",
        "uid", "item_index", "entity_type", "sample_id", "example_id", "query_box", "gold", "stale_initial",
    ]
    fields = [f for f in preferred if f in fields] + [f for f in fields if f not in preferred]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rows)


def make_note(path: pathlib.Path, design: dict[str, Any], meta_rows: list[dict[str, Any]], late_rows: list[dict[str, Any]]) -> None:
    def pick(group: str, contrast: str):
        vals = [r for r in late_rows if r["group"] == group and r["contrast"] == contrast]
        return sorted(vals, key=lambda r: int(r["seed"]))
    mismatches = [r for r in meta_rows if int(r["reported_numops"]) != int(r["relevant_updates"])]
    total_ops_rel0 = [r for r in late_rows if r["group"].startswith("rel0_total_ops_") and r["contrast"] in {"RminusV", "RminusC"}]
    rel_groups = []
    for k in range(0, 6):
        rel_groups.extend([r for r in late_rows if r["group"] == f"rel_updates_{k}" and r["contrast"] in {"RminusV", "RminusC", "VminusC"}])
    lines = []
    lines.append("# research Entity relevant-update and copy-window readout")
    lines.append("")
    lines.append("## Training packet design facts")
    lines.append("")
    lines.append(f"- Changed rows scanned: {design['changed_rows']:,}; changed source/companion pairs: {design['changed_pairs']:,}.")
    lines.append(f"- VIEW and REPEAT rows rebuilt exactly from `source_text + companion`: {design['view_rows_rebuilt_exact']:,}/{design['changed_rows']:,} and {design['repeat_rows_rebuilt_exact']:,}/{design['changed_rows']:,}.")
    lines.append(f"- Pair-local construction count: {design['pair_source_then_own_companion_count']:,}/{design['changed_pairs']:,} ({design['pair_source_then_own_companion_pct']:.2f}%).")
    lines.append(f"- Each individual source+companion pair fits the 256-token tokenizer window: VIEW {design['pair_text_fits_256_alone_view_pct']:.2f}%, REPEAT {design['pair_text_fits_256_alone_repeat_pct']:.2f}%. Packed rows are longer: full packed-row fit is VIEW {design['packed_row_fits_256_view_pct']:.2f}%, REPEAT {design['packed_row_fits_256_repeat_pct']:.2f}%; mean/max packed-row token lengths are VIEW {design['packed_row_tokens_view_mean_max'][0]:.1f}/{design['packed_row_tokens_view_mean_max'][1]} and REPEAT {design['packed_row_tokens_repeat_mean_max'][0]:.1f}/{design['packed_row_tokens_repeat_mean_max'][1]}.")
    lines.append("- Interpretation: the changed-block construction really is source plus its own companion, and the REPEAT companion is a rotated exact source-token segment. Because multiple pairs can be packed into a row, not every entire packed row is visible under seq256; the pair-level copy signal is nevertheless canonical for most individual source+companion packets.")
    lines.append("")
    lines.append("## Entity variable audit")
    lines.append("")
    lines.append(f"- Official-filtered Entity items: {len(meta_rows):,}. Parsed relevant-update count mismatches with the dataset `numops`: {len(mismatches)}.")
    if mismatches[:3]:
        lines.append(f"  First mismatches: {mismatches[:3]}")
    lines.append("- Thus the earlier depth variable is best read as number of state updates touching the queried box, not merely total operation sentences or context length.")
    lines.append("")
    lines.append("## Late official accuracy by relevant updates")
    lines.append("")
    lines.append("| seed | group | contrast | delta acc pct | acc a | acc b | n | total ops mean | words mean |")
    lines.append("|---:|---|---|---:|---:|---:|---:|---:|---:|")
    for r in sorted(rel_groups, key=lambda x: (int(x["seed"]), x["group"], x["contrast"])):
        lines.append(f"| {r['seed']} | {r['group']} | {r['contrast']} | {r['late_mean_delta_accuracy_pct_a_minus_b']:+.2f} | {r['late_mean_acc_a']:.2f} | {r['late_mean_acc_b']:.2f} | {r['n']} | {r['mean_total_ops']:.2f} | {r['mean_prefix_words']:.1f} |")
    lines.append("")
    lines.append("## Zero-relevant-update split by total operation count")
    lines.append("")
    lines.append("| seed | group | contrast | delta acc pct | acc a | acc b | n | words mean |")
    lines.append("|---:|---|---|---:|---:|---:|---:|---:|")
    for r in sorted(total_ops_rel0, key=lambda x: (int(x["seed"]), x["group"], x["contrast"])):
        lines.append(f"| {r['seed']} | {r['group']} | {r['contrast']} | {r['late_mean_delta_accuracy_pct_a_minus_b']:+.2f} | {r['late_mean_acc_a']:.2f} | {r['late_mean_acc_b']:.2f} | {r['n']} | {r['mean_prefix_words']:.1f} |")
    lines.append("")
    lines.append("## Scientific reading")
    lines.append("")
    lines.append("The official Entity depth result is already organized by relevant updates to the queried box. The key alternative to test is therefore whether REPEAT's no-update advantage persists when irrelevant operations and longer contexts are present, and whether the arm crossover starts when the queried state is actually changed. The tables above preserve that split for the two existing seeds and late checkpoints. Stale-initial percentages remain a secondary error clue because the eligible stale option is not present for every item and the wrong-answer denominator changes with accuracy.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER_ROOT), use_fast=True)
    t0 = time.time()
    design = verify_training_design(tokenizer)
    meta = load_entity_metadata(tokenizer)
    meta_rows = list(meta.values())
    augmented = load_prediction_augmented(meta)
    summary_rows = summarize(augmented)
    con_rows = contrasts(summary_rows)
    late_rows = late_summary(con_rows)
    write_csv(OUT / "entity_item_metadata.csv", meta_rows)
    write_csv(OUT / "entity_prediction_augmented_rows.csv", augmented)
    write_csv(OUT / "entity_relevant_update_summary.csv", summary_rows)
    write_csv(OUT / "entity_relevant_update_contrasts.csv", con_rows)
    write_csv(OUT / "entity_relevant_update_late_contrasts.csv", late_rows)
    (OUT / "training_packet_design_visibility.json").write_text(json.dumps(design, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    make_note(OUT / "entity_relevant_update_readout.md", design, meta_rows, late_rows)
    result = {
        "status": "ENTITY_RELEVANT_UPDATE_ANALYSIS_DONE",
        "finished_utc": now(),
        "elapsed_sec": round(time.time() - t0, 2),
        "official_filtered_entity_items": len(meta_rows),
        "prediction_rows": len(augmented),
        "numops_relevant_update_mismatches": sum(1 for r in meta_rows if int(r["reported_numops"]) != int(r["relevant_updates"])),
        "files": {
            "training_packet_design_visibility": rel(OUT / "training_packet_design_visibility.json"),
            "item_metadata": rel(OUT / "entity_item_metadata.csv"),
            "augmented_prediction_rows": rel(OUT / "entity_prediction_augmented_rows.csv"),
            "summary": rel(OUT / "entity_relevant_update_summary.csv"),
            "contrasts": rel(OUT / "entity_relevant_update_contrasts.csv"),
            "late_contrasts": rel(OUT / "entity_relevant_update_late_contrasts.csv"),
            "note": rel(OUT / "entity_relevant_update_readout.md"),
        },
    }
    (OUT / "entity_relevant_update_summary.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)
    print("\nKEY LATE GROUPS")
    for r in late_rows:
        if r["group"] in {"rel_updates_0", "rel_updates_1", "rel_updates_2", "rel_updates_3", "rel_updates_4", "rel_updates_5"} and r["contrast"] in {"RminusV", "RminusC", "VminusC"}:
            print(f"seed={r['seed']} {r['group']} {r['contrast']} dAcc={r['late_mean_delta_accuracy_pct_a_minus_b']:+.2f} acc_a={r['late_mean_acc_a']:.2f} acc_b={r['late_mean_acc_b']:.2f} n={r['n']} total_ops={r['mean_total_ops']:.2f}")
    print("\nZERO-RELEVANT TOTAL-OPS CUT")
    for r in late_rows:
        if r["group"].startswith("rel0_total_ops_") and r["contrast"] in {"RminusV", "RminusC"}:
            print(f"seed={r['seed']} {r['group']} {r['contrast']} dAcc={r['late_mean_delta_accuracy_pct_a_minus_b']:+.2f} n={r['n']} words={r['mean_prefix_words']:.1f}")


if __name__ == "__main__":
    main()
