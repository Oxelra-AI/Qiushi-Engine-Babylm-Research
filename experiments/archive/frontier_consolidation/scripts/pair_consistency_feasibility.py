#!/usr/bin/env python3
"""research: CPU feasibility analysis for source/rewrite pair consistency.

A true pair-consistency repair would require the trainer to know which tokens in
a packed changed-block row belong to the original source sentence and which to
its compact rewrite.  This script reconstructs that mapping from the frozen row
sidecar and accepted compact-pair metadata, using only training-corpus files.
It produces span/token statistics and an implementation map; it does not train.
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
OUT_DIR = WORKSPACE / "data/pair_consistency_feasibility"
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


def wc(text: str) -> int:
    return len(text.split())


def read_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_meta() -> tuple[set[str], dict[int, dict[str, Any]]]:
    selected = set()
    by_ex = {}
    for obj in read_jsonl(ROW_META):
        pair_ids = [str(x).split(":", 1)[-1] for x in obj.get("pair_ids") or []]
        selected.update(pair_ids)
        by_ex[int(obj["example_id"])] = {**obj, "pair_ids_norm": pair_ids}
    return selected, by_ex


def load_pairs(selected: set[str]) -> dict[str, dict[str, Any]]:
    out = {}
    for obj in read_jsonl(PAIR_ROWS):
        pid = str(obj.get("prompt_id") or obj.get("pair_id") or "")
        if pid in selected:
            out[pid] = obj
    return out


def construct_word_roles(meta: dict[str, Any], pair_map: dict[str, dict[str, Any]]) -> list[tuple[str, str, str]]:
    seq = []
    for pid in meta.get("pair_ids_norm") or []:
        p = pair_map[pid]
        for w in str(p.get("source_text") or "").split():
            seq.append((norm_word(w), pid, "source"))
        for w in str(p.get("rewrite_text") or "").split():
            seq.append((norm_word(w), pid, "rewrite"))
    return seq


def row_word_list(text: str) -> list[str]:
    return [norm_word(m.group(0)) for m in WORD_RE.finditer(text)]


def token_word_indices(text: str, offsets: list[tuple[int, int]]) -> tuple[list[int], list[tuple[int, int]]]:
    spans = [(m.start(), m.end()) for m in WORD_RE.finditer(text)]
    out = []
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
    return out, spans


def q(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    xs = sorted(values)
    pos = (len(xs) - 1) * p
    lo = int(pos)
    hi = min(lo + 1, len(xs) - 1)
    frac = pos - lo
    return xs[lo] * (1 - frac) + xs[hi] * frac


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    ap.add_argument("--max-length", type=int, default=256)
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
    tok = AutoTokenizer.from_pretrained(str(TOKENIZER), use_fast=True)
    special_ids = set(tok.all_special_ids)

    rows_total = 0
    changed_total = 0
    exact_alignment = 0
    token_role_rows = 0
    rows_any_truncated = 0
    pair_counts = []
    word_lengths = []
    token_lengths = []
    source_tokens = []
    rewrite_tokens = []
    both_tokens = []
    per_pair_token_lens = []
    unusable_examples = []
    row_samples = []
    pair_visibility_counter = collections.Counter()
    source_rewrite_ratio = []

    with POOL.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rows_total += 1
            obj = json.loads(line)
            if obj.get("source") != CHANGED_SOURCE:
                continue
            changed_total += 1
            ex_id = int(obj["example_id"])
            text = str(obj["text"])
            meta = meta_by_ex.get(ex_id)
            if not meta:
                unusable_examples.append({"example_id": ex_id, "reason": "missing row meta"})
                continue
            constructed = construct_word_roles(meta, pair_map)
            row_words = row_word_list(text)
            n = min(len(row_words), len(constructed))
            prefix_ok = all(row_words[i] == constructed[i][0] for i in range(n)) and n == len(constructed)
            # allow a tiny 9-word clean topup suffix in the final changed block row
            if all(row_words[i] == constructed[i][0] for i in range(n)) and len(row_words) - len(constructed) <= 16:
                prefix_ok = True
            if prefix_ok:
                exact_alignment += 1
            roles = [(None, "unknown")] * len(row_words)
            for i in range(min(len(row_words), len(constructed))):
                roles[i] = (constructed[i][1], constructed[i][2])
            enc = tok(text, add_special_tokens=False, truncation=True, max_length=args.max_length, return_offsets_mapping=True)
            ids = [int(x) for x in enc["input_ids"]]
            offsets = [(int(a), int(b)) for a, b in enc["offset_mapping"]]
            if len(ids) >= args.max_length and len(tok(text, add_special_tokens=False)["input_ids"]) > args.max_length:
                rows_any_truncated += 1
            t2w, spans = token_word_indices(text, offsets)
            counts = collections.Counter()
            pair_local = collections.defaultdict(lambda: collections.Counter())
            for tid, wi in zip(ids, t2w):
                if tid in special_ids or wi < 0 or wi >= len(roles):
                    continue
                pid, role = roles[wi]
                counts[role] += 1
                if pid is not None and role in ("source", "rewrite"):
                    pair_local[pid][role] += 1
            st = counts["source"]
            rt = counts["rewrite"]
            if st or rt:
                token_role_rows += 1
            source_tokens.append(st)
            rewrite_tokens.append(rt)
            both_tokens.append(st + rt)
            if st and rt:
                source_rewrite_ratio.append(rt / st)
            for pid, c in pair_local.items():
                if c["source"] > 0 and c["rewrite"] > 0:
                    pair_visibility_counter["both_visible"] += 1
                elif c["source"] > 0:
                    pair_visibility_counter["source_only"] += 1
                elif c["rewrite"] > 0:
                    pair_visibility_counter["rewrite_only"] += 1
                per_pair_token_lens.append({"source": int(c["source"]), "rewrite": int(c["rewrite"]), "total": int(c["source"] + c["rewrite"])})
            pair_counts.append(len(meta.get("pair_ids_norm") or []))
            word_lengths.append(len(row_words))
            token_lengths.append(len(ids))
            if len(row_samples) < 8:
                row_samples.append({
                    "example_id": ex_id,
                    "words": len(row_words),
                    "token_len": len(ids),
                    "pair_count": len(meta.get("pair_ids_norm") or []),
                    "source_tokens": st,
                    "rewrite_tokens": rt,
                    "pair_ids": meta.get("pair_ids_norm") or [],
                    "text_preview": text[:300],
                })

    def stats(xs: list[float]) -> dict[str, float | int]:
        return {"n": len(xs), "mean": statistics.mean(xs) if xs else 0.0, "median": statistics.median(xs) if xs else 0.0, "p05": q(xs, 0.05), "p95": q(xs, 0.95), "min": min(xs) if xs else 0.0, "max": max(xs) if xs else 0.0}

    pair_total = len(per_pair_token_lens)
    pair_token_totals = [x["total"] for x in per_pair_token_lens]
    feasible_rows = token_role_rows
    payload = {
        "status": "PAIR_CONSISTENCY_FEASIBILITY",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "purpose": "Determine whether a future source-rewrite consistency learning signal can be implemented under the frozen compact-view-reinvest corpus without changing the data stream or reading evaluation examples.",
        "inputs": {"pool_10m": str(POOL), "pool_sha256": pool_sha, "row_meta": str(ROW_META), "pair_rows": str(PAIR_ROWS), "tokenizer": str(TOKENIZER), "tokenizer_sha256": tok_sha, "max_length": args.max_length},
        "counts": {"rows_total": rows_total, "changed_rows": changed_total, "selected_pair_ids": len(selected), "pair_rows_loaded": len(pair_map), "exact_or_tiny_suffix_alignment_rows": exact_alignment, "token_role_rows": token_role_rows, "rows_with_true_token_truncation": rows_any_truncated, "pair_token_records": pair_total},
        "row_stats": {"pair_count": stats(pair_counts), "word_length": stats(word_lengths), "token_length": stats(token_lengths), "source_tokens": stats(source_tokens), "rewrite_tokens": stats(rewrite_tokens), "source_plus_rewrite_tokens": stats(both_tokens), "rewrite_to_source_token_ratio": stats(source_rewrite_ratio)},
        "pair_visibility_counter_at_seq256": dict(pair_visibility_counter),
        "per_pair_token_total_stats": stats(pair_token_totals),
        "samples": row_samples,
        "implementation_map": [
            "The inherited trainer dataset does not return example_id, source label, pair_id, role, or token span fields; a true consistency objective requires a new dataset/collate path, not only a masking-function patch.",
            "The changed-block sidecar plus accepted pair metadata reconstruct source/rewrite roles for essentially all changed rows; token role spans are therefore feasible as an auxiliary supervision map derived only from the training corpus.",
            "A lowest-risk consistency screen would add a small auxiliary loss only on changed-block rows: encode source and rewrite token subsets from the same packed row, pool hidden states over matching pair spans, and penalize distance between source and rewrite representations after the MLM forward. This changes the learning signal but not corpus, tokenizer, data order, model architecture, or evaluation.",
            "Implementation should be separate from the static-prior trainer; do not mix consistency and static prior in the first test. Use one cheap short/mature screen only if the matched clean-control trajectory shows compact-view reinvestment needs a learning-signal repair rather than a representation-only repair."
        ],
        "elapsed_sec": round(time.time() - t0, 3),
    }
    out_json = out_dir / "pair_consistency_feasibility.json"
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = [
        "# research pair-consistency feasibility",
        "",
        payload["purpose"],
        "",
        "## Counts",
    ]
    for k, v in payload["counts"].items():
        lines.append(f"- {k}: {v}")
    lines += ["", "## Row/token stats"]
    for k, rec in payload["row_stats"].items():
        lines.append(f"- {k}: mean={rec['mean']:.3f}, median={rec['median']:.3f}, p05={rec['p05']:.3f}, p95={rec['p95']:.3f}, min={rec['min']}, max={rec['max']}")
    lines += ["", "## Pair visibility at seq256", f"- {dict(pair_visibility_counter)}", "", "## Implementation map"]
    for x in payload["implementation_map"]:
        lines.append(f"- {x}")
    lines += ["", f"Full JSON: `{out_json}`"]
    out_md = out_dir / "pair_consistency_feasibility.md"
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "out_json": str(out_json), "out_md": str(out_md), "changed_rows": changed_total, "token_role_rows": token_role_rows, "rows_with_true_token_truncation": rows_any_truncated, "elapsed_sec": payload["elapsed_sec"]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
