#!/usr/bin/env python3
"""research: Build strict content-innovation group map for isolated masking reallocation.

This repairs the map's definition of strict innovation.

The strict target is the repaired research object:
  - a rewrite word group inside a both-visible compact source/rewrite pair;
  - normalized form absent from the paired source;
  - normalized form absent from every other word-group occurrence in the row;
  - content-like: not a stopword, not a relation/function cue, length >= 3,
    contains an alphabetic character.

The only intended training intervention is to shift expected WWM supervision mass
from copyable rewrite groups into these strict content innovations. Source groups,
filler groups, duplicate source-absent rewrite groups, and function-like innovations
must remain at the ordinary 0.15 WWM rate.

CPU-only. No model training and no official evaluation.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import hashlib
import json
import pathlib
import re
import statistics
import sys
import time
from collections import Counter, defaultdict
from typing import Any

from transformers import AutoTokenizer


def find_user_root() -> pathlib.Path:
    return _PUBLIC_ROOT


USER_ROOT = find_user_root()
WORKSPACE = USER_ROOT / "experiments/archive/frontier_consolidation"
POOL_10M = WORKSPACE / "data/density_cleanqwen_overlay_medium_riskhard/cleanqwen_fineweb_compact_view_reinvest_10M.jsonl"
SPAN_JSONL = WORKSPACE / "data/pair_span_map/pair_span_map.jsonl"
TOKENIZER_DIR = WORKSPACE / "data/compliant_tokenizer"
OUT_DIR = WORKSPACE / "data/strict_innovation_group_map"

EXPECTED_POOL_SHA = "215944978157394dbecf2039f2c1e1806bfcbb9701421920f78605eb58975a23"
EXPECTED_TOK_SHA = "91b775514b9f4e2d1f37c28445ab98181007e16d14f763955168547e293ee8f9"
SEQ_LEN = 256
MASK_PROB = 0.15
P_STRICT_DEFAULT = 0.50

RELATION_WORDS = {
    "in", "on", "under", "over", "above", "below", "behind", "beside", "near", "inside", "outside",
    "into", "onto", "between", "through", "across", "around", "from", "to", "with", "without", "before",
    "after", "during", "while", "when", "because", "therefore", "then", "if", "unless", "although",
    "but", "not", "no", "never", "only", "all", "some", "every", "any", "more", "less", "same", "different",
    "cause", "causes", "caused", "make", "makes", "made", "move", "moves", "moved", "put", "puts", "placed",
    "go", "goes", "went", "fall", "falls", "fell", "open", "opens", "closed", "break", "breaks", "broke",
}

STOPWORDS = set("""
a an the and or but if then when while because so for to of in on at by from with without into onto through over under above below after before during as is are was were be been being has have had do does did it its they them their he she his her we our you your this that these those there here not no only all some any every more less same different like than can could should would may might must will shall just also very
""".split())


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def is_word_start(token_str: str) -> bool:
    return token_str.startswith("Ġ") or token_str.startswith("▁")


def compute_word_groups(input_ids: list[int], attention_mask: list[int], tokenizer, special_ids: set[int]) -> list[int]:
    groups = [-1] * len(input_ids)
    ws_cache: dict[int, bool] = {}
    gid = -1
    for i, tid in enumerate(input_ids):
        if attention_mask[i] == 0:
            continue
        if tid in special_ids:
            continue
        if tid not in ws_cache:
            tok = tokenizer.convert_ids_to_tokens(int(tid))
            ws_cache[tid] = bool(tok is not None and is_word_start(str(tok)))
        if gid < 0 or ws_cache[tid] or i == 0:
            gid += 1
        groups[i] = gid
    return groups


def norm_text(text: str) -> str:
    text = text.lower().replace("Ġ", " ").replace("▁", " ")
    pieces = re.findall(r"[a-z0-9]+", text)
    return " ".join(pieces)


def token_norm(tokenizer, ids: list[int]) -> str:
    if not ids:
        return ""
    return norm_text(tokenizer.decode([int(x) for x in ids], clean_up_tokenization_spaces=False))


def is_content_like(norm: str) -> bool:
    parts = norm.split()
    if not parts:
        return False
    w = parts[0]
    if w in STOPWORDS or w in RELATION_WORDS:
        return False
    if len(w) < 3:
        return False
    return bool(re.search(r"[a-z]", w))


def group_positions_in_ranges(groups: list[int], ranges: list[list[int]]) -> dict[int, list[int]]:
    out: dict[int, list[int]] = defaultdict(list)
    for start, end in ranges:
        for pos in range(int(start), int(end)):
            if 0 <= pos < len(groups):
                gid = int(groups[pos])
                if gid >= 0:
                    out[gid].append(pos)
    return out


def stat(vals: list[float]) -> dict[str, Any]:
    if not vals:
        return {"n": 0, "mean": None, "median": None, "min": None, "max": None}
    return {
        "n": len(vals),
        "mean": float(statistics.mean(vals)),
        "median": float(statistics.median(vals)),
        "min": float(min(vals)),
        "max": float(max(vals)),
    }


def main() -> None:
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    pool_sha = sha256_file(POOL_10M)
    tok_sha = sha256_file(TOKENIZER_DIR / "tokenizer.json")
    if pool_sha != EXPECTED_POOL_SHA:
        raise RuntimeError(f"pool SHA mismatch: {pool_sha}")
    if tok_sha != EXPECTED_TOK_SHA:
        raise RuntimeError(f"tokenizer SHA mismatch: {tok_sha}")

    span_by_ex: dict[int, dict[str, Any]] = {}
    with SPAN_JSONL.open("r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            span_by_ex[int(obj["example_id"])] = obj
    changed_ids = set(span_by_ex)

    pool_texts: dict[int, str] = {}
    with POOL_10M.open("r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            eid = int(obj.get("example_id", -1))
            if eid in changed_ids:
                pool_texts[eid] = str(obj["text"])
    if len(pool_texts) != len(changed_ids):
        missing = len(changed_ids) - len(pool_texts)
        raise RuntimeError(f"missing changed rows from pool: {missing}")

    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER_DIR), use_fast=True)
    special_ids = set(tokenizer.all_special_ids)

    out_map: dict[int, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    totals = Counter()
    strict_norms = Counter()
    duplicate_norms = Counter()
    function_norms = Counter()
    copyable_norms = Counter()

    for eid in sorted(changed_ids):
        text = pool_texts[eid]
        span = span_by_ex[eid]
        enc = tokenizer(text, add_special_tokens=False, truncation=True, max_length=SEQ_LEN, padding="max_length")
        input_ids = [int(x) for x in enc["input_ids"]]
        attention_mask = [int(x) for x in enc["attention_mask"]]
        groups = compute_word_groups(input_ids, attention_mask, tokenizer, special_ids)
        n_total = max([g for g in groups if g >= 0], default=-1) + 1

        all_gid_pos: dict[int, list[int]] = defaultdict(list)
        for pos, gid in enumerate(groups):
            if gid >= 0:
                all_gid_pos[int(gid)].append(pos)
        row_norm_to_gids: dict[str, set[int]] = defaultdict(set)
        row_gid_to_norm: dict[int, str] = {}
        row_gid_to_tokens: dict[int, int] = {}
        for gid, posns in all_gid_pos.items():
            norm = token_norm(tokenizer, [input_ids[p] for p in posns])
            if norm:
                row_gid_to_norm[gid] = norm
                row_norm_to_gids[norm].add(gid)
                row_gid_to_tokens[gid] = len(posns)

        source_gids: set[int] = set()
        rewrite_gids: set[int] = set()
        strict_gids: set[int] = set()
        copyable_gids: set[int] = set()
        duplicate_source_absent_gids: set[int] = set()
        function_source_absent_unique_gids: set[int] = set()
        short_or_empty_source_absent_gids: set[int] = set()

        for pair in span.get("pairs", []):
            if pair.get("visibility") != "both_visible":
                continue
            src_groups = group_positions_in_ranges(groups, pair.get("source_token_ranges", []))
            rew_groups = group_positions_in_ranges(groups, pair.get("rewrite_token_ranges", []))
            src_norms: set[str] = set()
            for gid, posns in src_groups.items():
                source_gids.add(gid)
                norm = token_norm(tokenizer, [input_ids[p] for p in posns])
                if norm:
                    src_norms.add(norm)
            for gid, posns in rew_groups.items():
                rewrite_gids.add(gid)
                norm = token_norm(tokenizer, [input_ids[p] for p in posns])
                if not norm or len(norm) <= 1:
                    short_or_empty_source_absent_gids.add(gid)
                    totals["skip_empty_or_single"] += 1
                    continue
                if norm in src_norms:
                    copyable_gids.add(gid)
                    copyable_norms[norm] += 1
                    continue
                # Source-absent with respect to this paired source. It must also be row-unique.
                other_gids = row_norm_to_gids.get(norm, set()) - {gid}
                if other_gids:
                    duplicate_source_absent_gids.add(gid)
                    duplicate_norms[norm] += 1
                    continue
                if is_content_like(norm):
                    strict_gids.add(gid)
                    strict_norms[norm] += 1
                else:
                    function_source_absent_unique_gids.add(gid)
                    function_norms[norm] += 1

        # Ensure disjointness by precedence. Copyable is never a strict innovation.
        strict_gids -= copyable_gids
        duplicate_source_absent_gids -= strict_gids | copyable_gids
        function_source_absent_unique_gids -= strict_gids | copyable_gids | duplicate_source_absent_gids
        short_or_empty_source_absent_gids -= strict_gids | copyable_gids | duplicate_source_absent_gids | function_source_absent_unique_gids
        source_only_gids = source_gids - strict_gids - copyable_gids
        rewrite_other_gids = rewrite_gids - strict_gids - copyable_gids
        filler_gids = set(range(n_total)) - (rewrite_gids | source_gids)
        ordinary_gids = set(range(n_total)) - strict_gids - copyable_gids

        n_strict = len(strict_gids)
        n_copy = len(copyable_gids)
        n_source_only = len(source_only_gids)
        n_rewrite_other = len(rewrite_other_gids)
        n_duplicate_source_absent = len(duplicate_source_absent_gids)
        n_function_source_absent_unique = len(function_source_absent_unique_gids)
        n_short_or_empty_source_absent = len(short_or_empty_source_absent_gids)
        n_nonrewrite_filler = len(filler_gids)
        n_ordinary = len(ordinary_gids)

        strict_tokens = sum(row_gid_to_tokens.get(g, 0) for g in strict_gids)
        copy_tokens = sum(row_gid_to_tokens.get(g, 0) for g in copyable_gids)
        ordinary_tokens = sum(row_gid_to_tokens.get(g, 0) for g in ordinary_gids)

        out_map[eid] = {
            "strict_content_innovation_gids": sorted(strict_gids),
            "copyable_gids": sorted(copyable_gids),
            "ordinary_gids": sorted(ordinary_gids),
            "source_gids": sorted(source_only_gids),
            "duplicate_source_absent_gids": sorted(duplicate_source_absent_gids),
            "function_source_absent_unique_gids": sorted(function_source_absent_unique_gids),
            "short_or_empty_source_absent_gids": sorted(short_or_empty_source_absent_gids),
            "rewrite_other_gids": sorted(rewrite_other_gids),
            "filler_gids": sorted(filler_gids),
            "n_total_groups": n_total,
            "n_strict_content_innovation": n_strict,
            "n_copyable": n_copy,
            "n_source_only": n_source_only,
            "n_rewrite_other": n_rewrite_other,
            "n_duplicate_source_absent": n_duplicate_source_absent,
            "n_function_source_absent_unique": n_function_source_absent_unique,
            "n_short_or_empty_source_absent": n_short_or_empty_source_absent,
            "n_nonrewrite_filler": n_nonrewrite_filler,
            "n_ordinary": n_ordinary,
            "strict_tokens": strict_tokens,
            "copyable_tokens": copy_tokens,
            "ordinary_tokens": ordinary_tokens,
        }
        row_rec = {
            "example_id": eid,
            "n_total": n_total,
            "n_strict": n_strict,
            "n_copyable": n_copy,
            "n_source_only": n_source_only,
            "n_rewrite_other": n_rewrite_other,
            "n_duplicate_source_absent": n_duplicate_source_absent,
            "n_function_source_absent_unique": n_function_source_absent_unique,
            "n_short_or_empty_source_absent": n_short_or_empty_source_absent,
            "n_nonrewrite_filler": n_nonrewrite_filler,
            "n_ordinary": n_ordinary,
            "strict_tokens": strict_tokens,
            "copyable_tokens": copy_tokens,
            "ordinary_tokens": ordinary_tokens,
        }
        rows.append(row_rec)
        for k, v in row_rec.items():
            if k != "example_id":
                totals[k] += int(v)
        totals["rows_processed"] += 1
        if n_strict > 0:
            totals["rows_with_strict"] += 1
        if n_copy > 0:
            totals["rows_with_copyable"] += 1

    total_strict = int(totals["n_strict"])
    total_copy = int(totals["n_copyable"])
    if total_copy <= 0:
        raise RuntimeError("no copyable groups to fund strict innovation uplift")
    p_copy_default = MASK_PROB - (P_STRICT_DEFAULT - MASK_PROB) * total_strict / total_copy
    if not (0.0 <= p_copy_default <= MASK_PROB):
        raise RuntimeError(f"p_copy_default out of isolated-shift range: {p_copy_default}")
    expected_extra_strict = (P_STRICT_DEFAULT - MASK_PROB) * total_strict
    expected_removed_copy = (MASK_PROB - p_copy_default) * total_copy

    map_json = OUT_DIR / "strict_innovation_group_map.json"
    map_json.write_text(json.dumps({str(k): v for k, v in out_map.items()}, separators=(",", ":")) + "\n", encoding="utf-8")
    rows_jsonl = OUT_DIR / "strict_innovation_rows.jsonl"
    with rows_jsonl.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, separators=(",", ":")) + "\n")

    summary = {
        "status": "STRICT_INNOVATION_GROUP_MAP",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "purpose": "Strict repaired-research target map: source-absent, row-unique, content-like rewrite groups; copyable groups used only to fund expected mass shift; all other groups ordinary.",
        "inputs": {
            "pool_10m": str(POOL_10M),
            "pair_span_map": str(SPAN_JSONL),
            "tokenizer_dir": str(TOKENIZER_DIR),
            "pool_sha256": pool_sha,
            "tokenizer_sha256": tok_sha,
            "changed_rows": len(changed_ids),
        },
        "totals": {k: int(v) for k, v in sorted(totals.items())},
        "per_row_stats": {
            "n_total": stat([r["n_total"] for r in rows]),
            "n_strict": stat([r["n_strict"] for r in rows]),
            "n_copyable": stat([r["n_copyable"] for r in rows]),
            "n_source_only": stat([r["n_source_only"] for r in rows]),
            "n_rewrite_other": stat([r["n_rewrite_other"] for r in rows]),
            "n_duplicate_source_absent": stat([r["n_duplicate_source_absent"] for r in rows]),
            "n_function_source_absent_unique": stat([r["n_function_source_absent_unique"] for r in rows]),
            "n_short_or_empty_source_absent": stat([r["n_short_or_empty_source_absent"] for r in rows]),
            "n_nonrewrite_filler": stat([r["n_nonrewrite_filler"] for r in rows]),
        },
        "isolated_reallocation_default": {
            "mask_prob": MASK_PROB,
            "p_strict": P_STRICT_DEFAULT,
            "p_copy": p_copy_default,
            "total_strict_groups": total_strict,
            "total_copyable_groups": total_copy,
            "expected_extra_strict_groups": expected_extra_strict,
            "expected_removed_copyable_groups": expected_removed_copy,
            "expected_mass_balance_error": expected_extra_strict - expected_removed_copy,
            "ordinary_group_probability": MASK_PROB,
            "interpretation": "Only strict content innovations and copyable rewrite groups change probabilities; source, filler, duplicate source-absent, and function-like groups remain at baseline.",
        },
        "top_strict_norms": strict_norms.most_common(50),
        "top_copyable_norms": copyable_norms.most_common(40),
        "top_duplicate_source_absent_norms": duplicate_norms.most_common(30),
        "top_function_source_absent_unique_norms": function_norms.most_common(30),
        "outputs": {
            "map_json": str(map_json),
            "rows_jsonl": str(rows_jsonl),
            "summary_json": str(OUT_DIR / "strict_innovation_group_map_summary.json"),
            "note_md": str(OUT_DIR / "strict_innovation_group_map.md"),
        },
        "elapsed_sec": round(time.time() - t0, 3),
    }
    (OUT_DIR / "strict_innovation_group_map_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    lines = [
        "# research — strict content-innovation group map",
        "",
        "CPU-only; no model update, no official evaluation, no H100 work.",
        "",
        "This replaces the research loose map. Strict innovation means a rewrite word group that is source-absent relative to its paired source, absent from every other word-group occurrence in the row, and content-like. Duplicate source-absent and relation/function-like groups remain ordinary.",
        "",
        f"- changed rows processed: `{totals['rows_processed']}`",
        f"- rows with strict innovation: `{totals['rows_with_strict']}`",
        f"- strict content-innovation groups per 10M pass: `{total_strict}`",
        f"- copyable rewrite groups per 10M pass: `{total_copy}`",
        f"- duplicate source-absent rewrite groups kept ordinary: `{totals['n_duplicate_source_absent']}`",
        f"- unique relation/function-like source-absent rewrite groups kept ordinary: `{totals['n_function_source_absent_unique']}`",
        f"- short/empty source-absent rewrite groups kept ordinary: `{totals['n_short_or_empty_source_absent']}`",
        "",
        "## Category totals per 10M pass",
        f"- total groups in changed rows: `{totals['n_total']}`",
        f"- strict content innovations: `{totals['n_strict']}`",
        f"- copyable rewrite: `{totals['n_copyable']}`",
        f"- source-only groups: `{totals['n_source_only']}`",
        f"- non-strict rewrite-other groups: `{totals['n_rewrite_other']}`",
        f"  - duplicate source-absent: `{totals['n_duplicate_source_absent']}`",
        f"  - unique relation/function-like source-absent: `{totals['n_function_source_absent_unique']}`",
        f"  - short/empty source-absent: `{totals['n_short_or_empty_source_absent']}`",
        f"- nonrewrite filler groups: `{totals['n_nonrewrite_filler']}`",
        "",
        "## Default isolated mass reallocation",
        f"- baseline WWM rate: `{MASK_PROB}`",
        f"- strict innovation rate: `{P_STRICT_DEFAULT}`",
        f"- copyable rewrite rate needed for aggregate mass match: `{p_copy_default:.8f}`",
        f"- expected extra strict groups: `{expected_extra_strict:.6f}`",
        f"- expected removed copyable groups: `{expected_removed_copy:.6f}`",
        f"- aggregate balance error: `{expected_extra_strict - expected_removed_copy:.12f}`",
        "- all ordinary groups, including source/filler/duplicate/function-like groups, remain at baseline 0.15.",
        "",
        "## Per-row means",
    ]
    for key, label in [("n_total", "total groups"), ("n_strict", "strict"), ("n_copyable", "copyable"), ("n_source_only", "source-only"), ("n_rewrite_other", "rewrite-other"), ("n_duplicate_source_absent", "duplicate source-absent"), ("n_function_source_absent_unique", "function/relation source-absent"), ("n_nonrewrite_filler", "nonrewrite filler")]:
        s = summary["per_row_stats"][key]
        lines.append(f"- {label}: mean={s['mean']:.4f}, median={s['median']}, min={s['min']}, max={s['max']}")
    lines += ["", "## Top strict normalized targets"]
    for w, c in strict_norms.most_common(25):
        lines.append(f"- `{w}`: {c}")
    lines += ["", f"Map: `{map_json}`", f"Summary JSON: `{OUT_DIR / 'strict_innovation_group_map_summary.json'}`"]
    (OUT_DIR / "strict_innovation_group_map.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps({
        "status": summary["status"],
        "rows": totals["rows_processed"],
        "rows_with_strict": totals["rows_with_strict"],
        "strict_groups": total_strict,
        "copyable_groups": total_copy,
        "p_strict": P_STRICT_DEFAULT,
        "p_copy_default": p_copy_default,
        "map_json": str(map_json),
        "summary_json": str(OUT_DIR / "strict_innovation_group_map_summary.json"),
        "elapsed_sec": summary["elapsed_sec"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
