#!/usr/bin/env python3
"""research exact WWM exposure split for compact ordered-vs-scrambled channel events.

This post-hoc analyzer sharpens the research fixed ordered-text channel probe.  It
reconstructs, without model training, whether each probed compact-side word was actually
selected as a WWM target during the research 40M ordered and scrambled training streams.
It then splits ordered-minus-scrambled event losses by selected-count states.

Why this matters: the fixed channel probe evaluates ordered source+compact text, so the
ordered arm is on its native sequence distribution while the scrambled arm is not.  A
source_absent_content NLL advantage is strongest when (a) it is larger than retained/function
advantages and (b) it persists for events whose probed word was never selected as a training
target in the relevant 40M streams.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import collections
import csv
import hashlib
import importlib.util
import json
import math
import random
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import torch
from transformers import AutoTokenizer

USER_ROOT = _public_path('experiments/archive/frontier_consolidation/scripts/compact_order_event_exposure_split.py')
for _ in range(12):
    if (_public_path("experiments")).exists():
        break
    USER_ROOT = _public_path('experiments/archive/frontier_consolidation/scripts')
STUDY = _public_path('experiments/archive/frontier_consolidation')
WS = _public_path('experiments/archive/frontier_consolidation')
TOKENIZER = _public_path('experiments/archive/frontier_consolidation/data/compliant_tokenizer')
META_PATH = _public_path('experiments/archive/frontier_consolidation/data/density_cleanqwen_overlay_medium_riskhard/cleanqwen_fineweb_compact_view_reinvest_changed_block_rows_meta.jsonl')
PAIR_DIR = _public_path('experiments/archive/frontier_consolidation/data/factorial_view_candidate_audit')
CHANNEL_DIR = _public_path('experiments/archive/frontier_consolidation/data/compact_order_channel_probe')
DEFAULT_OUT = _public_path('experiments/archive/frontier_consolidation/data/compact_order_event_exposure_split')
SEQ_LEN = 256
BATCH_SIZE = 256
ROWS_PER_10M = 64740
PASSES = 4
MASK_PROB = 0.15
TRAIN_RNG_SEED = 43023
CATEGORIES = ["retained_content", "source_absent_content", "function_other"]
ARMS = {
    "ordered": _public_path('experiments/archive/frontier_consolidation/data/factorial_view_candidate_audit/compact_candidate_pairs.jsonl'),
    "scrambled": _public_path('experiments/archive/frontier_consolidation/data/factorial_view_candidate_audit/compact_scrambled_candidate_pairs.jsonl'),
}
RUN_ROOTS = {
    "ordered": _public_path('experiments/archive/frontier_consolidation/training/runs/compact_order_ordered_40M_seed43022'),
    "scrambled": _public_path('experiments/archive/frontier_consolidation/training/runs/compact_order_scrambled_40M_seed43022'),
}
POOLS_10M = {
    "ordered": _public_path('experiments/archive/frontier_consolidation/data/compact_order_factorial_pool_scaffold/compact_ordered_10M.jsonl'),
    "scrambled": _public_path('experiments/archive/frontier_consolidation/data/compact_order_factorial_pool_scaffold/compact_scrambled_10M.jsonl'),
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows



def resolve_existing_path(path_like: str | Path, base: Path | None = None) -> Path:
    """Resolve absolute, user-root-relative, or directory-relative saved paths."""
    p = Path(path_like)
    if p.is_absolute():
        return p
    if base is not None and (base / p).exists():
        return base / p
    return USER_ROOT / p


def load_channel_events(channel: dict[str, Any], channel_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Merge research loss rows with probe-event metadata.

    `compact_order_channel_probe.py` writes compact loss rows without `word_index`
    or ordered-side spans, because those are normally only needed by the channel probe
    itself.  Exact training-exposure reconstruction needs `word_index`, so load the
    parallel `probe_events.jsonl` and merge by stable `event_index` before simulating WWM.
    """
    loss_path = resolve_existing_path(channel.get("event_losses_jsonl") or (channel_dir / "event_losses.jsonl"), channel_dir)
    probe_raw = channel.get("event_file") or channel.get("probe_events_jsonl") or (channel_dir / "probe_events.jsonl")
    probe_path = resolve_existing_path(probe_raw, channel_dir)
    if not loss_path.exists():
        raise FileNotFoundError(loss_path)
    if not probe_path.exists():
        raise FileNotFoundError(probe_path)
    loss_rows = read_jsonl(loss_path)
    probe_rows = read_jsonl(probe_path)
    probe_by_idx = {int(r.get("event_index", i)): r for i, r in enumerate(probe_rows)}
    events: list[dict[str, Any]] = []
    missing_probe = 0
    missing_word_index = 0
    for i, loss in enumerate(loss_rows):
        idx = int(loss.get("event_index", i))
        probe = probe_by_idx.get(idx)
        if probe is None:
            missing_probe += 1
            merged = dict(loss)
        else:
            merged = dict(probe)
            merged.update(loss)
        merged["event_index"] = idx
        if "word_index" not in merged:
            missing_word_index += 1
        events.append(merged)
    if missing_word_index:
        raise RuntimeError(f"{missing_word_index} events lack word_index after merging {loss_path} with {probe_path}")
    return events, {
        "event_losses_jsonl": str(loss_path),
        "probe_events_jsonl": str(probe_path),
        "loss_rows": len(loss_rows),
        "probe_rows": len(probe_rows),
        "missing_probe_rows": missing_probe,
        "missing_word_index_after_merge": missing_word_index,
    }


def norm(w: str) -> str:
    import re
    return re.sub(r"[^a-z0-9]", "", str(w).lower())


def token_count(tok, text: str) -> int:
    if not text:
        return 0
    return len(tok(text, add_special_tokens=False)["input_ids"])


def is_word_start(token_str: str) -> bool:
    return token_str.startswith("Ġ") or token_str.startswith("▁")


def encode_group(tok, text: str, special_ids: set[int]) -> tuple[list[int], list[int], list[int]]:
    enc = tok(text, add_special_tokens=False, truncation=True, max_length=SEQ_LEN, padding="max_length")
    ids = [int(x) for x in enc["input_ids"]]
    att = [int(x) for x in enc["attention_mask"]]
    group = [-1] * SEQ_LEN
    gid = -1
    cache: dict[int, bool] = {}
    for i, (tid, am) in enumerate(zip(ids, att)):
        if not am or tid in special_ids:
            continue
        flag = cache.get(tid)
        if flag is None:
            s = tok.convert_ids_to_tokens(int(tid))
            flag = bool(s is not None and is_word_start(str(s)))
            cache[tid] = flag
        if gid < 0 or flag or i == 0:
            gid += 1
        group[i] = gid
    return ids, att, group


def map_word_index(original_words: list[str], target_i: int, arm_words: list[str]) -> tuple[int | None, dict[str, Any]]:
    """Map ordered-view word index to the corresponding arm-view occurrence.

    For the scrambled arm the whitespace-token multiset is identical but positions differ.  Use
    normalized occurrence rank; exact word occurrence rank is recorded for ambiguity context.
    """
    target = original_words[target_i]
    tn = norm(target)
    before_same_norm = [i for i, w in enumerate(original_words[:target_i + 1]) if norm(w) == tn]
    rank_norm = len(before_same_norm) - 1
    positions_norm = [i for i, w in enumerate(arm_words) if norm(w) == tn]
    positions_exact = [i for i, w in enumerate(arm_words) if w == target]
    meta = {
        "target_word": target,
        "target_norm": tn,
        "norm_rank": rank_norm,
        "arm_norm_count": len(positions_norm),
        "arm_exact_count": len(positions_exact),
        "duplicate_norm_in_ordered": sum(1 for w in original_words if norm(w) == tn) > 1,
    }
    if rank_norm < len(positions_norm):
        return int(positions_norm[rank_norm]), meta
    return None, meta


def selected_summary(records: list[dict[str, Any]], ck: str, pred) -> dict[str, Any]:
    sub = [r for r in records if pred(r) and ck in r]
    if not sub:
        return {"n_events": 0}
    by_pair: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for r in sub:
        by_pair[str(r["pair_id"])].append(r)
    pairs = sorted(by_pair)
    def weighted(rows: list[dict[str, Any]]) -> float:
        num = sum(x[ck]["ordered"]["loss_sum"] - x[ck]["scrambled"]["loss_sum"] for x in rows)
        den = sum(x[ck]["ordered"]["pieces"] for x in rows)
        return num / den if den else float("nan")
    def evmean(rows: list[dict[str, Any]]) -> float:
        return statistics.mean(x[ck]["ordered"]["loss_mean"] - x[ck]["scrambled"]["loss_mean"] for x in rows) if rows else float("nan")
    rng = random.Random(205700 + len(ck) + len(sub))
    boots_w = []
    boots_e = []
    for _ in range(500):
        sample = []
        for _j in range(len(pairs)):
            sample.extend(by_pair[rng.choice(pairs)])
        boots_w.append(weighted(sample))
        boots_e.append(evmean(sample))
    def q(xs: list[float], p: float) -> float:
        xs = sorted(float(x) for x in xs if math.isfinite(float(x)))
        if not xs:
            return float("nan")
        return xs[min(len(xs)-1, max(0, int(round(p*(len(xs)-1)))))]
    return {
        "n_events": len(sub),
        "n_pair_clusters": len(pairs),
        "piece_weighted_delta": round(weighted(sub), 6),
        "event_mean_delta": round(evmean(sub), 6),
        "piece_interval": [round(q(boots_w, 0.025), 6), round(q(boots_w, 0.975), 6)],
        "event_interval": [round(q(boots_e, 0.025), 6), round(q(boots_e, 0.975), 6)],
        "p_piece_delta_lt0": round(sum(x < 0 for x in boots_w) / len(boots_w), 4),
        "meaning": "negative ordered_minus_scrambled means ordered arm lower NLL on fixed ordered compact event",
    }


def build_event_maps(events: list[dict[str, Any]], ordered_pairs: dict[str, dict[str, Any]], arm_pairs: dict[str, dict[str, Any]], meta_rows: list[dict[str, Any]], tok, special_ids: set[int]) -> tuple[dict[int, list[dict[str, Any]]], dict[str, Any]]:
    by_pair: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for e in events:
        by_pair[str(e["pair_id"])].append(e)
    row_events: dict[int, list[dict[str, Any]]] = collections.defaultdict(list)
    mapping_stats = collections.Counter()
    mapping_examples = []
    for meta in meta_rows:
        row_idx = int(meta["row_index"])
        pair_ids = [str(x) for x in meta.get("pair_ids", [])]
        if not pair_ids:
            continue
        parts: list[str] = []
        for pid in pair_ids:
            if pid not in arm_pairs:
                parts.append("")
                continue
            src = str(arm_pairs[pid]["source_text"]).strip()
            view = str(arm_pairs[pid]["view_text"]).strip()
            original_words = str(ordered_pairs[pid]["view_text"]).split()
            arm_words = view.split()
            prefix_before_pair = " ".join(p for p in parts if p)
            prefix_source = " ".join(p for p in [prefix_before_pair, src] if p)
            for e in by_pair.get(pid, []):
                ev_i = int(e["word_index"])
                arm_i, m = map_word_index(original_words, ev_i, arm_words)
                if arm_i is None:
                    mapping_stats["missing_word_mapping"] += 1
                    if len(mapping_examples) < 20:
                        mapping_examples.append({"event_index": e.get("event_index"), "pair_id": pid, "word": e.get("word"), "meta": m})
                    continue
                prefix_before = " ".join(p for p in [prefix_source, " ".join(arm_words[:arm_i])] if p)
                prefix_after = " ".join(p for p in [prefix_source, " ".join(arm_words[:arm_i + 1])] if p)
                st = token_count(tok, prefix_before)
                en = token_count(tok, prefix_after)
                if st >= en or st >= SEQ_LEN:
                    mapping_stats["not_visible_or_empty_span"] += 1
                    continue
                row_events[row_idx].append({
                    "event_index": int(e["event_index"]),
                    "span": [int(st), int(min(en, SEQ_LEN))],
                    "pair_id": pid,
                    "category": e.get("category"),
                    "arm_word_index": int(arm_i),
                    "mapping_meta": m,
                })
                mapping_stats["mapped_visible"] += 1
            parts.append(src)
            parts.append(view)
    return dict(row_events), {"mapping_stats": dict(mapping_stats), "mapping_examples": mapping_examples, "rows_with_events": len(row_events)}


def read_training_log_mask_counts(arm: str) -> tuple[list[int], Path | None]:
    log_path = RUN_ROOTS[arm] / "training_log.jsonl"
    if not log_path.exists():
        return [], log_path
    counts: list[int] = []
    with log_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if "masked_tokens" in row:
                counts.append(int(row["masked_tokens"]))
    return counts, log_path


def compare_mask_count_trace(arm: str, simulated: list[int]) -> dict[str, Any]:
    observed, log_path = read_training_log_mask_counts(arm)
    if not observed:
        return {"status": "no_training_log_mask_counts", "training_log": str(log_path) if log_path else None, "simulated_batches": len(simulated)}
    n = min(len(observed), len(simulated))
    diffs = [int(simulated[i]) - int(observed[i]) for i in range(n)]
    mismatches = [i for i, d in enumerate(diffs) if d != 0]
    return {
        "status": "exact_prefix_match" if (not mismatches and len(observed) == len(simulated)) else "mismatch_or_prefix_only",
        "training_log": str(log_path),
        "observed_log_rows": len(observed),
        "simulated_batches": len(simulated),
        "n_compared": n,
        "exact_prefix_match": not mismatches,
        "same_length": len(observed) == len(simulated),
        "mismatch_count": len(mismatches),
        "max_abs_diff": max((abs(d) for d in diffs), default=0),
        "sum_sim_minus_log": int(sum(diffs)),
        "first_mismatches": [
            {"batch_index0": int(i), "step": int(i + 1), "simulated": int(simulated[i]), "logged": int(observed[i]), "diff": int(simulated[i] - observed[i])}
            for i in mismatches[:20]
        ],
        "first20_simulated": [int(x) for x in simulated[:20]],
        "first20_logged": [int(x) for x in observed[:20]],
    }


def simulate_arm(arm: str, tok, events: list[dict[str, Any]], device: str, max_batches: int | None = None) -> dict[str, Any]:
    meta_rows = read_jsonl(META_PATH)
    ordered_pairs = {str(r["pair_id"]): r for r in read_jsonl(ARMS["ordered"])}
    arm_pairs = {str(r["pair_id"]): r for r in read_jsonl(ARMS[arm])}
    special_ids = set(int(x) for x in tok.all_special_ids)
    row_events, map_meta = build_event_maps(events, ordered_pairs, arm_pairs, meta_rows, tok, special_ids)
    # Precompute only the 10M row tensors once; research 40M pools are four exact passes.
    pool_path = POOLS_10M[arm]
    rows = read_jsonl(pool_path)
    if len(rows) != ROWS_PER_10M:
        raise RuntimeError(f"{arm} row count {len(rows)} != {ROWS_PER_10M}")
    ids_all = torch.empty((ROWS_PER_10M, SEQ_LEN), dtype=torch.long)
    att_all = torch.empty((ROWS_PER_10M, SEQ_LEN), dtype=torch.long)
    grp_all = torch.empty((ROWS_PER_10M, SEQ_LEN), dtype=torch.long)
    for i, r in enumerate(rows):
        ids, att, grp = encode_group(tok, str(r["text"]), special_ids)
        ids_all[i] = torch.tensor(ids, dtype=torch.long)
        att_all[i] = torch.tensor(att, dtype=torch.long)
        grp_all[i] = torch.tensor(grp, dtype=torch.long)
    counts: dict[int, dict[str, Any]] = {int(e["event_index"]): {"selected_count": 0, "visible_passes": 0, "selected_passes": []} for e in events}
    dev = torch.device(device if device == "cpu" or torch.cuda.is_available() else "cpu")
    gen = torch.Generator(device=dev)
    gen.manual_seed(TRAIN_RNG_SEED)
    special_tensor = torch.tensor(sorted(special_ids), device=dev)
    t0 = time.time()
    total_batches = math.ceil((ROWS_PER_10M * PASSES) / BATCH_SIZE)
    batches_done = 0
    batch_masked_counts: list[int] = []
    for global_start in range(0, ROWS_PER_10M * PASSES, BATCH_SIZE):
        global_end = min(ROWS_PER_10M * PASSES, global_start + BATCH_SIZE)
        row_indices = [g % ROWS_PER_10M for g in range(global_start, global_end)]
        inp = ids_all[row_indices].to(dev)
        att = att_all[row_indices].to(dev)
        groups_batch = grp_all[row_indices].to(dev)
        candidate = att.bool() & ~torch.isin(inp, special_tensor)
        select = torch.zeros_like(candidate)
        for b in range(inp.shape[0]):
            groups = groups_batch[b]
            valid_groups = torch.unique(groups[groups >= 0])
            if valid_groups.numel() == 0:
                continue
            gp = torch.rand(valid_groups.numel(), generator=gen, device=dev)
            chosen = valid_groups[gp < MASK_PROB]
            if chosen.numel() == 0:
                continue
            select[b] = torch.isin(groups, chosen) & candidate[b]
        if select.sum() == 0:
            flat = candidate.view(-1).nonzero(as_tuple=False)
            if flat.numel() > 0:
                select.view(-1)[flat[0, 0]] = True
        batch_masked_counts.append(int(select.sum().item()))
        # Advance RNG exactly through the trainer's 80/10/10 corruption draw and optional random-token draw.
        r = torch.rand(inp.shape[0], inp.shape[1], generator=gen, device=dev)
        rand_tok = select & (r >= 0.8) & (r < 0.9)
        if rand_tok.any():
            _ = torch.randint(0, len(tok), (int(rand_tok.sum()),), generator=gen, device=dev)
        # Record event exposure for rows that contain compact pairs.
        for b, row_i in enumerate(row_indices):
            evs = row_events.get(int(row_i))
            if not evs:
                continue
            pass_i = (global_start + b) // ROWS_PER_10M
            sel_b = select[b]
            for ev in evs:
                st, en = ev["span"]
                eid = int(ev["event_index"])
                counts[eid]["visible_passes"] += 1
                if bool(sel_b[int(st):int(en)].any().item()):
                    counts[eid]["selected_count"] += 1
                    counts[eid]["selected_passes"].append(int(pass_i))
        batches_done += 1
        if max_batches is not None and batches_done >= max_batches:
            break
    if dev.type == "cuda":
        torch.cuda.empty_cache()
    complete = max_batches is None or batches_done >= total_batches
    mask_validation = compare_mask_count_trace(arm, batch_masked_counts)
    return {
        "arm": arm,
        "pool_10m": str(pool_path),
        "pool_10m_sha256": sha256_file(pool_path),
        "map_meta": map_meta,
        "events": counts,
        "batches_done": batches_done,
        "total_batches_full": total_batches,
        "complete_simulation": complete,
        "elapsed_sec": round(time.time() - t0, 1),
        "device": str(dev),
        "mask_count_validation": mask_validation,
        "batch_masked_counts_first20": [int(x) for x in batch_masked_counts[:20]],
        "batch_masked_counts_last20": [int(x) for x in batch_masked_counts[-20:]],
        "batch_masked_counts_sum": int(sum(batch_masked_counts)),
    }


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--channel-dir", type=Path, default=CHANNEL_DIR)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--max-batches", type=int, default=0, help="debug only: stop after this many training batches per arm; 0 means full 40M")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_json = args.out_dir / "compact_order_event_exposure_split.json"
    if out_json.exists() and not args.force:
        print(out_json)
        return
    channel_json = args.channel_dir / "compact_order_channel_probe.json"
    if not channel_json.exists():
        raise FileNotFoundError(channel_json)
    channel = json.loads(channel_json.read_text(encoding="utf-8"))
    events, event_source_meta = load_channel_events(channel, args.channel_dir)
    tok = AutoTokenizer.from_pretrained(str(TOKENIZER), local_files_only=True, use_fast=True)
    # Ensure event_index is present and stable after merging probe metadata and loss rows.
    for i, e in enumerate(events):
        e.setdefault("event_index", i)
    max_batches = args.max_batches if args.max_batches and args.max_batches > 0 else None
    arm_exposure = {arm: simulate_arm(arm, tok, events, args.device, max_batches=max_batches) for arm in ["ordered", "scrambled"]}
    enriched = []
    counts = {cat: collections.Counter() for cat in CATEGORIES}
    for e in events:
        eid = int(e["event_index"])
        row = dict(e)
        for arm in ["ordered", "scrambled"]:
            ex = arm_exposure[arm]["events"].get(eid, {"selected_count": 0, "visible_passes": 0, "selected_passes": []})
            row[f"{arm}_selected_count_4passes"] = int(ex.get("selected_count", 0))
            row[f"{arm}_visible_passes"] = int(ex.get("visible_passes", 0))
            row[f"{arm}_selected_passes"] = ex.get("selected_passes", [])
        row["either_arm_selected_count_4passes"] = row["ordered_selected_count_4passes"] + row["scrambled_selected_count_4passes"]
        row["neither_arm_selected"] = row["either_arm_selected_count_4passes"] == 0
        row["ordered_never_selected"] = row["ordered_selected_count_4passes"] == 0
        row["scrambled_never_selected"] = row["scrambled_selected_count_4passes"] == 0
        enriched.append(row)
        cat = row.get("category")
        if cat in counts:
            counts[cat][f"ordered_selected_{row['ordered_selected_count_4passes']}"] += 1
            counts[cat][f"scrambled_selected_{row['scrambled_selected_count_4passes']}"] += 1
            counts[cat]["neither_selected" if row["neither_arm_selected"] else "either_selected"] += 1
    # Summaries by checkpoint/category/split.
    summary: dict[str, Any] = {}
    checkpoints = [ck for ck in channel.get("summary", {}).keys() if any(ck in e for e in enriched)]
    for ck in checkpoints:
        summary[ck] = {}
        for cat in CATEGORIES:
            cr = [r for r in enriched if r.get("category") == cat]
            summary[ck][cat] = {}
            splits = [
                ("all", lambda r: True),
                ("neither_arm_selected", lambda r: r.get("neither_arm_selected") is True),
                ("either_arm_selected", lambda r: not r.get("neither_arm_selected")),
                ("ordered_never_selected", lambda r: int(r.get("ordered_selected_count_4passes", 0)) == 0),
                ("ordered_selected_any", lambda r: int(r.get("ordered_selected_count_4passes", 0)) > 0),
                ("scrambled_never_selected", lambda r: int(r.get("scrambled_selected_count_4passes", 0)) == 0),
                ("scrambled_selected_any", lambda r: int(r.get("scrambled_selected_count_4passes", 0)) > 0),
            ]
            for name, pred in splits:
                summary[ck][cat][name] = selected_summary(cr, ck, pred)
    # Minimal category-interaction view for never-selected events.
    interaction: dict[str, Any] = {}
    for ck in checkpoints:
        interaction[ck] = {}
        for split_name, pred in [
            ("all", lambda r: True),
            ("neither_arm_selected", lambda r: r.get("neither_arm_selected") is True),
            ("ordered_never_selected", lambda r: int(r.get("ordered_selected_count_4passes", 0)) == 0),
        ]:
            vals = {cat: summary[ck][cat][split_name].get("piece_weighted_delta") for cat in CATEGORIES}
            if all(v is not None for v in vals.values()):
                controls = statistics.mean([float(vals["retained_content"]), float(vals["function_other"])])
                interaction[ck][split_name] = {
                    "source_absent_delta": vals["source_absent_content"],
                    "retained_content_delta": vals["retained_content"],
                    "function_other_delta": vals["function_other"],
                    "source_absent_minus_controls_mean": round(float(vals["source_absent_content"]) - controls, 6),
                    "meaning": "negative source_absent_minus_controls_mean means source-absent has larger ordered NLL advantage than retained/function controls within this exposure split",
                }
    out_rows = []
    for r in enriched:
        flat = {k: r.get(k) for k in ["event_index", "pair_id", "category", "word", "word_index", "n_pieces", "ordered_selected_count_4passes", "scrambled_selected_count_4passes", "either_arm_selected_count_4passes", "neither_arm_selected", "ordered_never_selected", "scrambled_never_selected"]}
        out_rows.append(flat)
    write_csv(args.out_dir / "event_training_exposure_rows.csv", out_rows)
    (args.out_dir / "event_losses_with_training_exposure.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in enriched), encoding="utf-8")
    payload = {
        "status": "COMPACT_ORDER_EVENT_EXPOSURE_SPLIT",
        "meaning": "Exact/no-model reconstruction of whether each fixed channel-probe event word was selected as a WWM target in the ordered and scrambled research 40M training streams. Use never-selected splits to separate category-level source-absent behavior from direct target exposure.",
        "inputs": {
            "channel_json": str(channel_json),
            "event_source_meta": event_source_meta,
            "event_losses_jsonl": event_source_meta.get("event_losses_jsonl"),
            "probe_events_jsonl": event_source_meta.get("probe_events_jsonl"),
            "tokenizer_sha256": sha256_file(_public_path('experiments/archive/frontier_consolidation/data/compliant_tokenizer/tokenizer.json')),
            "meta_path": str(META_PATH),
            "meta_sha256": sha256_file(META_PATH),
            "seq_len": SEQ_LEN,
            "batch_size": BATCH_SIZE,
            "passes": PASSES,
            "mask_prob": MASK_PROB,
            "train_rng_seed": TRAIN_RNG_SEED,
            "device": args.device,
            "max_batches": max_batches,
        },
        "arm_exposure_meta": {arm: {k: v for k, v in rec.items() if k != "events"} for arm, rec in arm_exposure.items()},
        "counts": {cat: dict(c) for cat, c in counts.items()},
        "summary": summary,
        "category_interaction_by_exposure_split": interaction,
        "enriched_event_losses": str(args.out_dir / "event_losses_with_training_exposure.jsonl"),
        "event_training_exposure_csv": str(args.out_dir / "event_training_exposure_rows.csv"),
        "json_path": str(out_json),
    }
    write_json(out_json, payload)
    lines = ["# research compact order event exposure split", "", payload["meaning"], ""]
    lines.append(f"Events: {len(enriched)}; checkpoints with losses: {checkpoints}; debug max_batches={max_batches}.")
    lines.append("")
    lines.append("## Counts")
    for cat, c in payload["counts"].items():
        lines.append(f"- {cat}: {dict(c)}")
    lines.append("")
    lines.append("## Source-absent minus controls by exposure split")
    for ck, by_split in interaction.items():
        lines.append(f"### {ck}")
        for split, rec in by_split.items():
            lines.append(f"- {split}: {rec}")
    lines.append("")
    lines.append(f"JSON: `{out_json}`")
    (args.out_dir / "compact_order_event_exposure_split.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "out": str(out_json), "counts": payload["counts"], "interaction": interaction, "complete": {a: arm_exposure[a]["complete_simulation"] for a in arm_exposure}}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
