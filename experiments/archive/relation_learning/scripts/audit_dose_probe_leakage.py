#!/usr/bin/env python3
"""research: audit nested restatement-dose selected pairs for probe/heldout leakage.

The nested dose streams from research add validated source->rewrite pairs to the
compact-view-reinvest substrate.  Before training, the selected pair texts must be
checked against the evaluation/readout texts whose small effects will define the
dose curve.  This script compares every selected source and rewrite string in the
current dose21/dose25 selections against:

  * the 6,992-row BabyLM official row-holdout slice used by research/research;
  * the older 2,647-row compact-view-reinvest row-holdout slice, for completeness;
  * the 1,626 compact rewrite-conditioning probe pairs used in research;
  * the 1,200 WikiLarge simplification probe pairs used in Steps023-025.

It performs exact normalized string matching, heldout-row token-subsequence
matching, and high lexical-overlap matching.  Results are written to
data/dose_leakage_audit/ and optionally inserted into the
research stream metadata.  A nonzero number of blocking hits means the selected
pair pool must be repaired and the dose streams re-materialized before GPU time is
spent.
"""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import math
import pathlib
import re
import statistics
import time
from dataclasses import dataclass
from typing import Any, Iterable

ROOT = pathlib.Path.cwd()
STUDY = ROOT / "experiments/archive/relation_learning"
OUT_DEFAULT = STUDY / "data/dose_leakage_audit"
META_PATH = STUDY / "data/nested_restatement_dose_streams/nested_dose_materialization_metadata.json"
DOSE21_SELECTED = STUDY / "data/nested_restatement_dose_streams/dose21_selected_pairs.jsonl"
DOSE25_SUPERSET = STUDY / "data/nested_restatement_dose_streams/dose25_selected_pairs_superset.jsonl"
DOSE25_TOPUP = STUDY / "data/nested_restatement_dose_streams/dose25_topup_selected_pairs_from_shard0.jsonl"

HELDOUT_6992 = ROOT / "experiments/archive/frontier_consolidation/data/dose_2p64x_rowholdout_pools/heldout_cleanqwen_rows.jsonl"
HELDOUT_2647 = ROOT / "experiments/archive/frontier_consolidation/data/density_cleanqwen_overlay_medium_riskhard/heldout_cleanqwen_rows.jsonl"
ALL_ACCEPTED_PAIRS = ROOT / "experiments/archive/frontier_consolidation/data/expansion_analysis/combined_all_accepted_pairs.jsonl"
SELECTED_MAX_PAIRS = ROOT / "experiments/archive/frontier_consolidation/data/dose_distribution_select/selected_matched_max_pairs.jsonl"
WIKI_PAIRS = STUDY / "analysis/wikipedia_simplification_restatement_probe/wikipedia_simplification_pairs.jsonl"

WORD_RE = re.compile(r"[A-Za-z0-9]+(?:[\u2019'][A-Za-z0-9]+)?")
SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")
STOP = {
    "the", "a", "an", "and", "or", "but", "of", "to", "in", "on", "for", "with", "by", "at", "as", "from",
    "that", "this", "these", "those", "is", "are", "was", "were", "be", "been", "being", "it", "its", "he",
    "she", "they", "them", "his", "her", "their", "we", "you", "i", "me", "my", "our", "your", "not", "no",
    "do", "does", "did", "so", "if", "then", "than", "there", "here", "who", "what", "when", "where", "why",
    "how", "have", "has", "had", "will", "would", "can", "could", "should", "about", "into", "out", "up", "down",
}


def rel(p: pathlib.Path) -> str:
    try:
        return str(p.relative_to(ROOT))
    except Exception:
        return str(p)


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_jsonl(path: pathlib.Path, limit: int | None = None) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            out.append(json.loads(line))
            if limit is not None and len(out) >= limit:
                break
    return out


def norm_text(s: str) -> str:
    toks = [m.group(0).lower().replace("\u2019", "'") for m in WORD_RE.finditer(str(s))]
    return " ".join(toks)


def tokens(s: str) -> list[str]:
    return [m.group(0).lower().replace("\u2019", "'") for m in WORD_RE.finditer(str(s))]


def informative_tokens(tok: Iterable[str]) -> set[str]:
    return {t for t in tok if len(t) >= 3 and t not in STOP}


def canonical_pair_id(obj: dict[str, Any]) -> str:
    if obj.get("pair_id"):
        return str(obj["pair_id"])
    if obj.get("prompt_id"):
        return str(obj["prompt_id"])
    return f"sid:{obj.get('sentence_id')}|doc:{obj.get('doc_id')}"


def canonical_selected_key(pid: str) -> str:
    pid = str(pid)
    return pid.replace("compact:", "")


def selected_pair_ids_from_max() -> set[str]:
    selected: set[str] = set()
    for obj in read_jsonl(SELECTED_MAX_PAIRS):
        pid = canonical_pair_id(obj)
        selected.add(pid)
        selected.add(canonical_selected_key(pid))
        selected.add(f"compact:{canonical_selected_key(pid)}")
    return selected


def load_compact_probe_pairs(limit: int = 1626) -> list[dict[str, Any]]:
    # Matches research load_unselected_rewrite_pairs(max_pairs): iterate the all-accepted file,
    # exclude selected MAX-dose pairs, and take the first 1626 usable source/rewrite pairs.
    selected = selected_pair_ids_from_max()
    out: list[dict[str, Any]] = []
    for obj in read_jsonl(ALL_ACCEPTED_PAIRS):
        pid = canonical_pair_id(obj)
        key = canonical_selected_key(pid)
        if pid in selected or key in selected or f"compact:{key}" in selected:
            continue
        if "source_text" not in obj or "rewrite_text" not in obj:
            continue
        d = dict(obj)
        d["pair_id"] = pid if pid.startswith("compact:") else f"compact:{pid}"
        out.append(d)
        if len(out) >= limit:
            break
    return out


@dataclass
class RefText:
    ref_id: str
    screen: str
    field: str
    text: str
    norm: str
    toks: list[str]
    tokset: set[str]
    source_path: str


@dataclass
class QueryText:
    pair_id: str
    dose21: bool
    dose25_topup: bool
    dose25: bool
    field: str  # original or rewrite
    text: str
    norm: str
    toks: list[str]
    tokset: set[str]
    source: str
    original_id: str


def add_ref(refs: list[RefText], *, screen: str, field: str, ref_id: str, text: str, source_path: pathlib.Path) -> None:
    text = " ".join(str(text).split())
    if not text:
        return
    nt = norm_text(text)
    ts = tokens(text)
    if len(ts) < 3:
        return
    refs.append(RefText(ref_id=ref_id, screen=screen, field=field, text=text, norm=nt, toks=ts,
                        tokset=informative_tokens(ts), source_path=rel(source_path)))


def sentence_chunks(text: str) -> list[str]:
    text = " ".join(str(text).split())
    if not text:
        return []
    chunks: list[str] = []
    for part in SENT_SPLIT_RE.split(text):
        p = part.strip()
        if len(tokens(p)) >= 4:
            chunks.append(p)
    # Also add short sliding windows so transcript rows without punctuation still expose
    # exact sentence-like subsequences for overlap matching.
    row_toks = str(text).split()
    if len(row_toks) > 35:
        for start in range(0, len(row_toks), 20):
            win = " ".join(row_toks[start:start + 35]).strip()
            if len(tokens(win)) >= 8:
                chunks.append(win)
    return chunks


def load_references() -> tuple[list[RefText], dict[str, Any]]:
    refs: list[RefText] = []
    ref_meta: dict[str, Any] = {}

    for tag, path in [("heldout6992", HELDOUT_6992), ("heldout2647", HELDOUT_2647)]:
        rows = read_jsonl(path)
        ref_meta[f"{tag}_rows"] = len(rows)
        for obj in rows:
            rid = f"{tag}:row:{obj.get('source')}:{obj.get('example_id')}"
            text = str(obj.get("text", ""))
            add_ref(refs, screen=tag, field="row_full", ref_id=rid, text=text, source_path=path)
            for j, chunk in enumerate(sentence_chunks(text)):
                add_ref(refs, screen=tag, field="row_chunk", ref_id=f"{rid}:chunk:{j}", text=chunk, source_path=path)

    compact = load_compact_probe_pairs(1626)
    ref_meta["compact_probe_pairs"] = len(compact)
    for obj in compact:
        pid = str(obj.get("pair_id"))
        add_ref(refs, screen="compact_probe1626", field="source_text", ref_id=pid, text=str(obj.get("source_text", "")), source_path=ALL_ACCEPTED_PAIRS)
        add_ref(refs, screen="compact_probe1626", field="rewrite_text", ref_id=pid, text=str(obj.get("rewrite_text", "")), source_path=ALL_ACCEPTED_PAIRS)

    wiki = read_jsonl(WIKI_PAIRS)
    ref_meta["wiki_pairs"] = len(wiki)
    for obj in wiki:
        pid = str(obj.get("pair_id"))
        for field in ["source_text", "target_text", "unrelated_source_text"]:
            if obj.get(field):
                add_ref(refs, screen="wikilarge_probe1200", field=field, ref_id=pid, text=str(obj.get(field, "")), source_path=WIKI_PAIRS)
    ref_meta["reference_texts"] = len(refs)
    ref_meta["reference_texts_by_screen"] = dict(collections.Counter(r.screen for r in refs))
    return refs, ref_meta


def load_queries() -> tuple[list[QueryText], dict[str, Any]]:
    dose21_ids = {str(r.get("pair_id")) for r in read_jsonl(DOSE21_SELECTED)}
    topup_ids = {str(r.get("pair_id")) for r in read_jsonl(DOSE25_TOPUP)}
    superset = read_jsonl(DOSE25_SUPERSET)
    queries: list[QueryText] = []
    for obj in superset:
        pid = str(obj.get("pair_id"))
        for field, key in [("original", "original"), ("rewrite", "rewrite")]:
            text = " ".join(str(obj.get(key, "")).split())
            if not text:
                continue
            ts = tokens(text)
            queries.append(QueryText(
                pair_id=pid,
                dose21=pid in dose21_ids,
                dose25_topup=pid in topup_ids,
                dose25=True,
                field=field,
                text=text,
                norm=norm_text(text),
                toks=ts,
                tokset=informative_tokens(ts),
                source=str(obj.get("source", "")),
                original_id=str(obj.get("original_id", "")),
            ))
    meta = {
        "dose21_pairs": len(dose21_ids),
        "dose25_topup_pairs": len(topup_ids),
        "dose25_superset_pairs": len(superset),
        "selected_texts_audited": len(queries),
    }
    return queries, meta


def build_exact_map(refs: list[RefText]) -> dict[str, list[int]]:
    exact: dict[str, list[int]] = collections.defaultdict(list)
    for i, r in enumerate(refs):
        if r.norm:
            exact[r.norm].append(i)
    return exact


def build_fivegram_index(refs: list[RefText]) -> dict[tuple[str, ...], list[int]]:
    idx: dict[tuple[str, ...], list[int]] = collections.defaultdict(list)
    for i, r in enumerate(refs):
        # Only full rows/chunks need sequence containment. Compact/wiki exact equality is enough.
        if not r.screen.startswith("heldout"):
            continue
        toks = r.toks
        if len(toks) < 5:
            continue
        seen = set()
        for j in range(0, len(toks) - 4):
            gram = tuple(toks[j:j + 5])
            if gram not in seen:
                idx[gram].append(i)
                seen.add(gram)
    return idx


def has_subsequence(haystack: list[str], needle: list[str]) -> bool:
    if not needle or len(needle) > len(haystack):
        return False
    first = needle[0]
    L = len(needle)
    for i, t in enumerate(haystack[:len(haystack) - L + 1]):
        if t == first and haystack[i:i + L] == needle:
            return True
    return False


def build_overlap_index(refs: list[RefText]) -> tuple[dict[str, int], dict[str, list[int]]]:
    df: dict[str, int] = collections.Counter()
    for r in refs:
        for t in r.tokset:
            df[t] += 1
    inv: dict[str, list[int]] = collections.defaultdict(list)
    for i, r in enumerate(refs):
        for t in r.tokset:
            # Extremely common tokens are not useful candidate keys.
            if df[t] <= 2500:
                inv[t].append(i)
    return df, inv


def screen_hits(queries: list[QueryText], refs: list[RefText], *, jaccard_threshold: float, containment_threshold: float) -> list[dict[str, Any]]:
    exact_map = build_exact_map(refs)
    gram_index = build_fivegram_index(refs)
    df, inv = build_overlap_index(refs)
    hits: list[dict[str, Any]] = []
    seen_hit_keys: set[tuple[str, str, str, str, str]] = set()

    def add_hit(q: QueryText, r: RefText, hit_type: str, jac: float, contain_q: float, extra: dict[str, Any] | None = None) -> None:
        key = (q.pair_id, q.field, r.screen, r.field, r.ref_id + ":" + hit_type)
        if key in seen_hit_keys:
            return
        seen_hit_keys.add(key)
        row = {
            "pair_id": q.pair_id,
            "original_id": q.original_id,
            "query_field": q.field,
            "query_source": q.source,
            "dose21": q.dose21,
            "dose25_topup": q.dose25_topup,
            "dose25": q.dose25,
            "hit_type": hit_type,
            "screen": r.screen,
            "ref_field": r.field,
            "ref_id": r.ref_id,
            "jaccard": round(jac, 6),
            "contain_query_in_ref_tokenset": round(contain_q, 6),
            "query_tokens": len(q.toks),
            "query_informative_tokens": len(q.tokset),
            "ref_tokens": len(r.toks),
            "ref_informative_tokens": len(r.tokset),
            "query_text": q.text[:500],
            "ref_text": r.text[:500],
            "ref_source_path": r.source_path,
        }
        if extra:
            row.update(extra)
        hits.append(row)

    for qi, q in enumerate(queries):
        # Exact normalized text equality against reference snippets.
        for ri in exact_map.get(q.norm, []):
            r = refs[ri]
            add_hit(q, r, "exact_norm", 1.0, 1.0)

        # Exact token-sequence containment in heldout rows/chunks.  This catches selected
        # sentences that are substrings of a 160-word heldout row.
        if len(q.toks) >= 5:
            grams: list[tuple[str, ...]] = []
            if len(q.toks) >= 5:
                grams.append(tuple(q.toks[:5]))
                mid = max(0, len(q.toks)//2 - 2)
                grams.append(tuple(q.toks[mid:mid+5]))
                grams.append(tuple(q.toks[-5:]))
            cand = set()
            for g in grams:
                cand.update(gram_index.get(g, []))
            for ri in cand:
                r = refs[ri]
                if has_subsequence(r.toks, q.toks):
                    add_hit(q, r, "exact_token_subsequence", 1.0 if len(q.toks) == len(r.toks) else 0.0, 1.0)

        # High lexical overlap. Choose rare informative query terms to find candidates.
        if len(q.tokset) >= 4:
            rare_terms = sorted(q.tokset, key=lambda t: (df.get(t, 10**9), t))[:8]
            candidates: set[int] = set()
            for t in rare_terms:
                candidates.update(inv.get(t, []))
            for ri in candidates:
                r = refs[ri]
                if not r.tokset:
                    continue
                inter = len(q.tokset & r.tokset)
                if inter < 3:
                    continue
                union = len(q.tokset | r.tokset)
                jac = inter / union if union else 0.0
                contain_q = inter / len(q.tokset) if q.tokset else 0.0
                # For full heldout rows the row is much longer, so high containment of the
                # query's informative terms is the safer signal. For probe sentences use true
                # high Jaccard.
                if jac >= jaccard_threshold:
                    add_hit(q, r, "high_jaccard", jac, contain_q)
                elif r.screen.startswith("heldout") and contain_q >= containment_threshold and len(q.tokset) >= 6:
                    add_hit(q, r, "heldout_high_query_containment", jac, contain_q)

    return hits


def write_csv(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("\n", encoding="utf-8")
        return
    preferred = [
        "pair_id", "original_id", "query_field", "query_source", "dose21", "dose25_topup", "hit_type", "screen",
        "ref_field", "ref_id", "jaccard", "contain_query_in_ref_tokenset", "query_tokens", "ref_tokens",
        "query_text", "ref_text", "ref_source_path",
    ]
    fields = [f for f in preferred if f in rows[0]] + [f for f in sorted(set().union(*(r.keys() for r in rows))) if f not in preferred]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", default=str(OUT_DEFAULT))
    ap.add_argument("--jaccard-threshold", type=float, default=0.80)
    ap.add_argument("--containment-threshold", type=float, default=0.95)
    ap.add_argument("--update-metadata", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    queries, qmeta = load_queries()
    refs, rmeta = load_references()
    hits = screen_hits(queries, refs, jaccard_threshold=args.jaccard_threshold, containment_threshold=args.containment_threshold)

    blocking_types = {"exact_norm", "exact_token_subsequence", "high_jaccard", "heldout_high_query_containment"}
    blocking_hits = [h for h in hits if h["hit_type"] in blocking_types]
    by_screen = collections.Counter(h["screen"] for h in blocking_hits)
    by_type = collections.Counter(h["hit_type"] for h in blocking_hits)
    by_pair = collections.defaultdict(list)
    for h in blocking_hits:
        by_pair[h["pair_id"]].append(h)
    affected_dose21 = sorted({pid for pid, hs in by_pair.items() if any(h["dose21"] for h in hs)})
    affected_topup = sorted({pid for pid, hs in by_pair.items() if any(h["dose25_topup"] for h in hs)})

    # A small file of pair IDs to exclude if rematerialization is needed.
    exclusion_path = out_dir / "blocking_pair_ids_to_exclude.txt"
    exclusion_path.write_text("\n".join(sorted(by_pair)) + ("\n" if by_pair else ""), encoding="utf-8")

    write_csv(out_dir / "blocking_hits.csv", blocking_hits)
    with (out_dir / "blocking_hits.jsonl").open("w", encoding="utf-8") as f:
        for h in blocking_hits:
            f.write(json.dumps(h, ensure_ascii=False) + "\n")

    summary = {
        "status": "DOSE_PROBE_LEAKAGE_AUDIT",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_sec": round(time.time() - t0, 2),
        "query_meta": qmeta,
        "reference_meta": rmeta,
        "thresholds": {
            "jaccard_threshold": args.jaccard_threshold,
            "heldout_query_containment_threshold": args.containment_threshold,
        },
        "blocking_hit_count": len(blocking_hits),
        "blocking_pair_count": len(by_pair),
        "affected_dose21_pair_count": len(affected_dose21),
        "affected_dose25_topup_pair_count": len(affected_topup),
        "blocking_hits_by_screen": dict(sorted(by_screen.items())),
        "blocking_hits_by_type": dict(sorted(by_type.items())),
        "blocking_pair_ids_to_exclude": rel(exclusion_path),
        "outputs": {
            "blocking_hits_csv": rel(out_dir / "blocking_hits.csv"),
            "blocking_hits_jsonl": rel(out_dir / "blocking_hits.jsonl"),
        },
        "inputs": {
            "dose21_selected": rel(DOSE21_SELECTED),
            "dose25_superset": rel(DOSE25_SUPERSET),
            "dose25_topup": rel(DOSE25_TOPUP),
            "heldout6992": rel(HELDOUT_6992),
            "heldout2647": rel(HELDOUT_2647),
            "compact_all_accepted": rel(ALL_ACCEPTED_PAIRS),
            "compact_selected_max_excluded": rel(SELECTED_MAX_PAIRS),
            "wiki_pairs": rel(WIKI_PAIRS),
        },
        "input_sha256": {
            "dose21_selected": sha256_file(DOSE21_SELECTED),
            "dose25_superset": sha256_file(DOSE25_SUPERSET),
            "dose25_topup": sha256_file(DOSE25_TOPUP),
            "heldout6992": sha256_file(HELDOUT_6992),
            "heldout2647": sha256_file(HELDOUT_2647),
            "wiki_pairs": sha256_file(WIKI_PAIRS),
        },
        "scientific_interpretation": (
            "blocking_hit_count == 0 means no selected dose source/rewrite matched the primary heldout/probe texts "
            "under exact normalized, heldout token-subsequence, high-Jaccard, or high-query-containment screens. "
            "A positive count means the selected pair IDs should be excluded, dose streams re-materialized, and the audit rerun before training."
        ),
    }
    (out_dir / "audit_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    if args.update_metadata:
        meta = json.loads(META_PATH.read_text(encoding="utf-8"))
        meta["probe_leakage_audit"] = summary
        META_PATH.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        summary["metadata_updated"] = rel(META_PATH)
        (out_dir / "audit_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
