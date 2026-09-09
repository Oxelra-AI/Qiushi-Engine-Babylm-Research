#!/usr/bin/env python3
"""research: build a frozen pair-span map for possible source-view consistency.

The research/52 feasibility analysis showed that source/rewrite roles can be
reconstructed for compact-view-reinvest changed rows.  This script turns that
feasibility evidence into a durable CPU artifact: a JSONL map from each changed
training row (example_id) to tokenizer-visible source/rewrite token spans for each
selected compact pair under the research legal tokenizer and seq256 truncation.

It reads only training-side files already inside the legal 10M reinvest corpus
construction.  It does not train, evaluate, or change any corpus/tokenizer.  The
map is intended for a future separate source-view consistency trainer only if the
mature legal-tokenizer clean-vs-reinvest trajectory indicates broad disappearance
of the compact-view mechanism.
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
import pathlib
import re
import statistics
import time
from typing import Any

from transformers import AutoTokenizer


def find_user_root() -> pathlib.Path:
    return _PUBLIC_ROOT


USER_ROOT = find_user_root()
WORKSPACE = USER_ROOT / "experiments/archive/frontier_consolidation"
POOL = WORKSPACE / "data/density_cleanqwen_overlay_medium_riskhard/cleanqwen_fineweb_compact_view_reinvest_10M.jsonl"
ROW_META = WORKSPACE / "data/density_cleanqwen_overlay_medium_riskhard/cleanqwen_fineweb_compact_view_reinvest_changed_block_rows_meta.jsonl"
PAIR_ROWS = WORKSPACE / "data/medium_compact_analysis/medium_compact_ws_rows.jsonl"
TOKENIZER = WORKSPACE / "data/compliant_tokenizer"
OUT_DIR = WORKSPACE / "data/pair_span_map"
EXPECTED_POOL_SHA = "215944978157394dbecf2039f2c1e1806bfcbb9701421920f78605eb58975a23"
EXPECTED_TOKENIZER_SHA = "91b775514b9f4e2d1f37c28445ab98181007e16d14f763955168547e293ee8f9"
CHANGED_SOURCE = "cleanqwen_fineweb_compact_view_reinvest"
WORD_RE = re.compile(r"\S+")
STRIP_RE = re.compile(r"^[^\w]+|[^\w]+$", re.UNICODE)


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def norm_word(w: str) -> str:
    return STRIP_RE.sub("", w).lower().replace("’", "'")


def read_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_meta() -> tuple[set[str], dict[int, dict[str, Any]]]:
    selected: set[str] = set()
    by_ex: dict[int, dict[str, Any]] = {}
    for obj in read_jsonl(ROW_META):
        pair_ids = [str(x).split(":", 1)[-1] for x in obj.get("pair_ids") or []]
        selected.update(pair_ids)
        by_ex[int(obj["example_id"])] = {**obj, "pair_ids_norm": pair_ids}
    return selected, by_ex


def load_pairs(selected: set[str]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for obj in read_jsonl(PAIR_ROWS):
        pid = str(obj.get("prompt_id") or obj.get("pair_id") or "")
        if pid in selected:
            out[pid] = obj
    return out


def construct_word_roles(meta: dict[str, Any], pair_map: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    seq: list[dict[str, Any]] = []
    for pid in meta.get("pair_ids_norm") or []:
        p = pair_map[pid]
        for role, key in [("source", "source_text"), ("rewrite", "rewrite_text")]:
            for local_i, w in enumerate(str(p.get(key) or "").split()):
                seq.append({"norm": norm_word(w), "pair_id": pid, "role": role, "local_word_index": local_i})
    return seq


def token_word_indices(text: str, offsets: list[tuple[int, int]]) -> list[int]:
    spans = [(m.start(), m.end()) for m in WORD_RE.finditer(text)]
    out: list[int] = []
    j = 0
    for s, e in offsets:
        if e <= s:
            out.append(-1)
            continue
        while j + 1 < len(spans) and spans[j][1] <= s:
            j += 1
        best = -1
        best_ov = 0
        for k in (j - 1, j, j + 1):
            if 0 <= k < len(spans):
                ws, we = spans[k]
                ov = max(0, min(e, we) - max(s, ws))
                if ov > best_ov:
                    best = k
                    best_ov = ov
        out.append(best)
    return out


def row_words(text: str) -> list[str]:
    return [norm_word(m.group(0)) for m in WORD_RE.finditer(text)]


def contiguous_ranges(indices: list[int]) -> list[list[int]]:
    xs = sorted(set(int(i) for i in indices))
    if not xs:
        return []
    ranges: list[list[int]] = []
    start = prev = xs[0]
    for x in xs[1:]:
        if x == prev + 1:
            prev = x
        else:
            ranges.append([start, prev + 1])  # half-open
            start = prev = x
    ranges.append([start, prev + 1])
    return ranges


def q(vals: list[float], p: float) -> float:
    if not vals:
        return 0.0
    xs = sorted(vals)
    pos = (len(xs) - 1) * p
    lo = int(pos)
    hi = min(lo + 1, len(xs) - 1)
    frac = pos - lo
    return xs[lo] * (1 - frac) + xs[hi] * frac


def stats(vals: list[float]) -> dict[str, float | int]:
    return {
        "n": len(vals),
        "mean": statistics.mean(vals) if vals else 0.0,
        "median": statistics.median(vals) if vals else 0.0,
        "p05": q(vals, 0.05),
        "p95": q(vals, 0.95),
        "min": min(vals) if vals else 0.0,
        "max": max(vals) if vals else 0.0,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--max-rows", type=int, default=0, help="debug only; 0 means all changed rows")
    args = ap.parse_args()
    t0 = time.time()
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pool_sha = sha256_file(POOL)
    tok_sha = sha256_file(TOKENIZER / "tokenizer.json")
    if pool_sha != EXPECTED_POOL_SHA:
        raise RuntimeError(f"pool SHA mismatch: {pool_sha}")
    if tok_sha != EXPECTED_TOKENIZER_SHA:
        raise RuntimeError(f"tokenizer SHA mismatch: {tok_sha}")

    selected, meta_by_ex = load_meta()
    pair_map = load_pairs(selected)
    missing_pair_records = sorted(selected - set(pair_map))
    if missing_pair_records:
        raise RuntimeError(f"missing pair records for selected ids: {missing_pair_records[:5]} ... n={len(missing_pair_records)}")
    tok = AutoTokenizer.from_pretrained(str(TOKENIZER), use_fast=True)
    special_ids = set(tok.all_special_ids)

    out_jsonl = out_dir / "pair_span_map.jsonl"
    row_count = 0
    changed_rows_seen = 0
    rows_written = 0
    exact_alignment_rows = 0
    tiny_suffix_rows = 0
    token_truncation_rows = 0
    pair_visibility = collections.Counter()
    row_visibility = collections.Counter()
    source_tok_counts: list[float] = []
    rewrite_tok_counts: list[float] = []
    both_tok_counts: list[float] = []
    visible_pair_totals: list[float] = []
    contiguous_span_counts: list[float] = []
    records_samples: list[dict[str, Any]] = []

    with POOL.open("r", encoding="utf-8") as f, out_jsonl.open("w", encoding="utf-8") as out:
        for line in f:
            if not line.strip():
                continue
            row_count += 1
            obj = json.loads(line)
            if obj.get("source") != CHANGED_SOURCE:
                continue
            changed_rows_seen += 1
            if args.max_rows and changed_rows_seen > args.max_rows:
                break
            ex_id = int(obj["example_id"])
            text = str(obj["text"])
            meta = meta_by_ex.get(ex_id)
            if meta is None:
                raise RuntimeError(f"changed row {ex_id} missing meta")
            constructed = construct_word_roles(meta, pair_map)
            words = row_words(text)
            n = min(len(words), len(constructed))
            prefix = all(words[i] == constructed[i]["norm"] for i in range(n))
            tiny_suffix = prefix and len(words) >= len(constructed) and len(words) - len(constructed) <= 16
            if prefix and len(words) == len(constructed):
                exact_alignment_rows += 1
            elif tiny_suffix:
                tiny_suffix_rows += 1
            elif not prefix:
                bad = next((i for i in range(n) if words[i] != constructed[i]["norm"]), None)
                raise RuntimeError(f"alignment mismatch ex_id={ex_id} at word {bad}: row={words[bad] if bad is not None else None}, constructed={constructed[bad]['norm'] if bad is not None else None}")
            roles: list[dict[str, Any] | None] = [None] * len(words)
            for i in range(min(len(words), len(constructed))):
                roles[i] = constructed[i]
            enc = tok(text, add_special_tokens=False, truncation=True, max_length=args.max_length, return_offsets_mapping=True)
            ids = [int(x) for x in enc["input_ids"]]
            offsets = [(int(a), int(b)) for a, b in enc["offset_mapping"]]
            full_len = len(tok(text, add_special_tokens=False)["input_ids"])
            truncated = full_len > args.max_length
            if truncated:
                token_truncation_rows += 1
            t2w = token_word_indices(text, offsets)
            per_pair: dict[str, dict[str, Any]] = {}
            for tok_i, (tid, wi) in enumerate(zip(ids, t2w)):
                if tid in special_ids or wi < 0 or wi >= len(roles):
                    continue
                role_rec = roles[wi]
                if role_rec is None:
                    continue
                pid = str(role_rec["pair_id"])
                role = str(role_rec["role"])
                rec = per_pair.setdefault(pid, {
                    "pair_id": pid,
                    "source_token_indices": [],
                    "rewrite_token_indices": [],
                    "source_token_count": 0,
                    "rewrite_token_count": 0,
                })
                if role == "source":
                    rec["source_token_indices"].append(tok_i)
                    rec["source_token_count"] += 1
                elif role == "rewrite":
                    rec["rewrite_token_indices"].append(tok_i)
                    rec["rewrite_token_count"] += 1
            pair_records = []
            row_source = 0
            row_rewrite = 0
            n_both = 0
            n_source_only = 0
            n_rewrite_only = 0
            n_invisible = 0
            for pid in meta.get("pair_ids_norm") or []:
                base = per_pair.get(pid, {"pair_id": pid, "source_token_indices": [], "rewrite_token_indices": [], "source_token_count": 0, "rewrite_token_count": 0})
                s_idx = list(base["source_token_indices"])
                r_idx = list(base["rewrite_token_indices"])
                s_count = int(base["source_token_count"])
                r_count = int(base["rewrite_token_count"])
                row_source += s_count
                row_rewrite += r_count
                if s_count and r_count:
                    vis = "both_visible"
                    n_both += 1
                elif s_count:
                    vis = "source_only"
                    n_source_only += 1
                elif r_count:
                    vis = "rewrite_only"
                    n_rewrite_only += 1
                else:
                    vis = "invisible"
                    n_invisible += 1
                pair_visibility[vis] += 1
                visible_pair_totals.append(float(s_count + r_count))
                span_n = len(contiguous_ranges(s_idx)) + len(contiguous_ranges(r_idx))
                contiguous_span_counts.append(float(span_n))
                p = pair_map[pid]
                pair_records.append({
                    "pair_id": pid,
                    "visibility": vis,
                    "source_token_count": s_count,
                    "rewrite_token_count": r_count,
                    "source_token_ranges": contiguous_ranges(s_idx),
                    "rewrite_token_ranges": contiguous_ranges(r_idx),
                    "source_words": len(str(p.get("source_text") or "").split()),
                    "rewrite_words": len(str(p.get("rewrite_text") or "").split()),
                })
            if n_both == len(pair_records):
                row_visibility["all_pairs_both_visible"] += 1
            elif n_both > 0:
                row_visibility["some_pairs_both_visible"] += 1
            else:
                row_visibility["no_pairs_both_visible"] += 1
            source_tok_counts.append(float(row_source))
            rewrite_tok_counts.append(float(row_rewrite))
            both_tok_counts.append(float(row_source + row_rewrite))
            record = {
                "row_index_in_pool_1based": row_count,
                "example_id": ex_id,
                "words": int(obj.get("words", len(text.split()))),
                "token_len_truncated": len(ids),
                "token_len_full": full_len,
                "seq_length": args.max_length,
                "truncated_by_seq256": truncated,
                "pair_count": len(pair_records),
                "row_token_counts": {"source": row_source, "rewrite": row_rewrite, "source_plus_rewrite": row_source + row_rewrite},
                "row_pair_visibility": {"both_visible": n_both, "source_only": n_source_only, "rewrite_only": n_rewrite_only, "invisible": n_invisible},
                "alignment": "exact" if len(words) == len(constructed) else "tiny_suffix",
                "pairs": pair_records,
            }
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            rows_written += 1
            if len(records_samples) < 6:
                sample = {k: v for k, v in record.items() if k != "pairs"}
                sample["first_pairs"] = pair_records[:3]
                sample["text_preview"] = text[:260]
                records_samples.append(sample)

    summary = {
        "status": "PAIR_SPAN_MAP_BUILT",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "purpose": "Durable trainer-visible token-span map for possible source-view consistency; derived only from the frozen legal compact-view-reinvest training corpus and sidecars.",
        "inputs": {
            "pool_10m": str(POOL),
            "pool_sha256": pool_sha,
            "row_meta": str(ROW_META),
            "pair_rows": str(PAIR_ROWS),
            "tokenizer": str(TOKENIZER),
            "tokenizer_sha256": tok_sha,
            "max_length": args.max_length,
        },
        "outputs": {"pair_span_map_jsonl": str(out_jsonl)},
        "counts": {
            "pool_rows_scanned": row_count,
            "changed_rows_seen": changed_rows_seen if not args.max_rows else min(changed_rows_seen, args.max_rows),
            "rows_written": rows_written,
            "selected_pair_ids": len(selected),
            "pair_rows_loaded": len(pair_map),
            "exact_alignment_rows": exact_alignment_rows,
            "tiny_suffix_alignment_rows": tiny_suffix_rows,
            "token_truncation_rows": token_truncation_rows,
            "pair_visibility": dict(pair_visibility),
            "row_visibility": dict(row_visibility),
        },
        "row_token_stats": {
            "source_tokens": stats(source_tok_counts),
            "rewrite_tokens": stats(rewrite_tok_counts),
            "source_plus_rewrite_tokens": stats(both_tok_counts),
        },
        "pair_token_stats": {
            "source_plus_rewrite_visible_tokens_per_pair_record": stats(visible_pair_totals),
            "contiguous_ranges_per_pair_record": stats(contiguous_span_counts),
        },
        "samples": records_samples,
        "use_constraints": [
            "This map is a construction asset, not evidence that source-view consistency improves BabyLM scores.",
            "Do not combine this objective with static-prior masking in a first GPU test; the mature 70M/80M pattern must select the intervention family first.",
            "A future trainer using this map should keep corpus, tokenizer, model, optimizer, seeds, exposure accounting, and evaluation fixed; only the auxiliary source-view consistency loss should change.",
            "Contrastive in-batch negatives are risky in tiny-data MLM pretraining; a first implementation should consider stop-gradient or low-weight symmetric representation/logit agreement over true source-rewrite pairs before adding negatives.",
        ],
        "elapsed_sec": round(time.time() - t0, 3),
    }
    out_json = out_dir / "pair_span_map_summary.json"
    out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = [
        "# research pair-span map",
        "",
        summary["purpose"],
        "",
        "This is CPU-only training-side metadata. It does not choose or launch a route.",
        "",
        "## Counts",
    ]
    for k, v in summary["counts"].items():
        lines.append(f"- {k}: `{v}`")
    lines += ["", "## Row token stats"]
    for k, rec in summary["row_token_stats"].items():
        lines.append(f"- {k}: mean={rec['mean']:.3f}, median={rec['median']:.3f}, p05={rec['p05']:.3f}, p95={rec['p95']:.3f}, min={rec['min']}, max={rec['max']}")
    lines += ["", "## Pair token stats"]
    for k, rec in summary["pair_token_stats"].items():
        lines.append(f"- {k}: mean={rec['mean']:.3f}, median={rec['median']:.3f}, p05={rec['p05']:.3f}, p95={rec['p95']:.3f}, min={rec['min']}, max={rec['max']}")
    lines += ["", "## Use constraints"]
    for item in summary["use_constraints"]:
        lines.append(f"- {item}")
    lines += ["", f"Map JSONL: `{out_jsonl}`", f"Summary JSON: `{out_json}`"]
    out_md = out_dir / "pair_span_map_summary.md"
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": summary["status"],
        "out_json": str(out_json),
        "out_md": str(out_md),
        "out_jsonl": str(out_jsonl),
        "rows_written": rows_written,
        "pair_visibility": dict(pair_visibility),
        "elapsed_sec": summary["elapsed_sec"],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
