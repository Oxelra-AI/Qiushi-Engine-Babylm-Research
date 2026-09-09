#!/usr/bin/env python3
"""research: strict source-only candidates for structural experience tests.

This selects legal-corpus rows that look suitable for a small graph-verified
relational/event-state packet.  It does not use BabyLM evaluation items or model
scores, excludes already generated compact/rewrite rows, and is meant to feed a
manual or semi-automatic verification step, not training directly.
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
import re
from pathlib import Path
from typing import Dict, List, Tuple

ROOT = _public_path('.')
CORPUS = _public_path('experiments/archive/frontier_consolidation/data/density_cleanqwen_overlay_medium_riskhard/cleanqwen_fineweb_compact_view_reinvest_10M.jsonl')
OUT = _public_path('experiments/archive/frontier_consolidation/data/structural_packet_candidates')
TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)?")
SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")
NAMED = re.compile(r"\b[A-Z][a-z]{2,}(?:\s+[A-Z][a-z]{2,}){0,2}\b")
TRANSCRIPT_TAG = re.compile(r"\*[A-Z]{2,4}:|\b[A-Z]{2,4}:\s")

RX = {
    "temporal": re.compile(r"\b(before|after|then|when|while|until|later|next|finally|first|during|as soon as|meanwhile|subsequently|previously)\b", re.I),
    "causal": re.compile(r"\b(because|so that|therefore|thus|hence|due to|as a result|result(?:ed|s|ing)? in|caus(?:e|ed|es|ing)|led to|if|unless)\b", re.I),
    "state": re.compile(r"\b(put|puts|placed?|moved?|went|goes|returned?|left|entered?|arrived?|took|takes|gave|gives|changed?|became|become|turned|opened?|closed?|built|broke|formed?|created?|lost|found|kept|removed?|added|mixed|sent|brought|carried|filled|emptied)\b", re.I),
    "quantity": re.compile(r"\b(\d+(?:[\.,]\d+)?|one|two|three|four|five|six|seven|eight|nine|ten|more than|less than|fewer than|at least|at most|percent|per cent|half|double|twice|increased?|decreased?|reduced?|grew|larger|smaller|higher|lower)\b", re.I),
    "social": re.compile(r"\b(said|says|told|asked|answered|believed?|thought|knew|wanted|decided|promised|agreed|refused|claimed|reported|explained|heard|called)\b", re.I),
    "modality": re.compile(r"\b(not|no|never|without|cannot|can't|should|would|could|might|may|must|perhaps|maybe|likely|unlikely|possible|but|however|although|though|whereas|instead|rather than)\b", re.I),
    "pronoun": re.compile(r"\b(he|she|they|him|her|them|his|their|it|its|who|which|that|this|these|those)\b", re.I),
}

FAMILY_DEFS = {
    "event_temporal_causal": {"need": ["temporal", "state"], "bonus": ["causal", "pronoun"]},
    "entity_state_update": {"need": ["state", "pronoun"], "bonus": ["temporal", "social"]},
    "quantity_change_compare": {"need": ["quantity"], "bonus": ["state", "causal", "temporal"]},
    "social_belief_report": {"need": ["social", "pronoun"], "bonus": ["modality", "temporal", "causal"]},
    "polarity_contrast_event": {"need": ["modality", "state"], "bonus": ["causal", "temporal", "pronoun"]},
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def bucket(src: str) -> str:
    s = (src or "").lower()
    if "compact" in s or "qwen" in s or "rewrite" in s:
        return "generated_compact_or_rewrite"
    if "child" in s:
        return "CHILDES"
    if "bnc" in s:
        return "BNC_spoken"
    if "wiki" in s:
        return "SimpleWiki"
    if "guten" in s:
        return "Gutenberg"
    if "subtitle" in s:
        return "OpenSubtitles"
    if "switch" in s:
        return "Switchboard"
    if "fineweb" in s:
        return "FineWeb_or_web"
    return src or "UNKNOWN"


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip())


def row_quality(text: str, src_bucket: str) -> float:
    toks = TOKEN_RE.findall(text)
    n = len(toks)
    sents = [x.strip() for x in SENT_SPLIT.split(text.strip()) if x.strip()]
    transcript = len(TRANSCRIPT_TAG.findall(text))
    url_markup = len(re.findall(r"https?://|www\.|<[^>]+>|\{\{|\}\}|[_=]{2,}", text))
    quote_noise = (text.count("{") + text.count("}") + text.count("[") + text.count("]")) / max(1, len(text))
    if n < 25 or n > 180:
        return -10.0
    q = 2.0
    q += min(2.0, len(sents) * 0.4)
    q += min(1.0, len(NAMED.findall(text)) * 0.2)
    q -= 0.35 * transcript
    q -= 1.2 * url_markup
    q -= 20.0 * quote_noise
    if src_bucket in {"Gutenberg", "SimpleWiki", "OpenSubtitles"}:
        q += 0.4
    if src_bucket == "CHILDES" and transcript > 6:
        q -= 1.5
    return q


def hits(text: str) -> Dict[str, int]:
    return {k: len(rx.findall(text)) for k, rx in RX.items()}


def family_score(fam: str, h: Dict[str, int], text: str, src_bucket: str) -> float:
    spec = FAMILY_DEFS[fam]
    if any(h[k] == 0 for k in spec["need"]):
        return -1e9
    q = row_quality(text, src_bucket)
    if q < 0:
        return -1e9
    score = q + sum(min(3, h[k]) * 1.5 for k in spec["need"])
    score += sum(min(3, h[k]) * 0.6 for k in spec["bonus"])
    # Reward multiple sentences and named/referential surface, but avoid ultra-dense transcripts.
    score += 0.15 * min(8, len(NAMED.findall(text)))
    score -= 0.25 * max(0, len(TRANSCRIPT_TAG.findall(text)) - 3)
    return score


def evidence_snippet(text: str, h: Dict[str, int]) -> str:
    sents = [norm(s) for s in SENT_SPLIT.split(text.strip()) if s.strip()]
    scored = []
    for s in sents:
        sh = hits(s)
        sc = sum(1 for v in sh.values() if v) + 0.2 * sum(min(3, v) for v in sh.values())
        scored.append((sc, s))
    scored.sort(reverse=True)
    chosen = [s for _, s in scored[:3]] or [norm(text)]
    snip = " / ".join(chosen)
    return snip[:650] + ("..." if len(snip) > 650 else "")


def select(corpus: Path, per_family: int) -> Tuple[dict, List[dict]]:
    pool: Dict[str, List[dict]] = {fam: [] for fam in FAMILY_DEFS}
    counts = collections.Counter()
    source_counts = collections.Counter()
    with corpus.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if not line.strip():
                continue
            row = json.loads(line)
            text = str(row.get("text") or "")
            src = str(row.get("source") or "UNKNOWN")
            b = bucket(src)
            counts["rows"] += 1
            counts[f"source::{b}"] += 1
            if b == "generated_compact_or_rewrite":
                counts["excluded_generated"] += 1
                continue
            h = hits(text)
            for fam in FAMILY_DEFS:
                sc = family_score(fam, h, text, b)
                if sc > -1e8:
                    rec = {
                        "family": fam,
                        "row_index": idx,
                        "example_id": row.get("example_id"),
                        "source": src,
                        "source_bucket": b,
                        "words": int(row.get("words") or len(TOKEN_RE.findall(text))),
                        "token_count": len(TOKEN_RE.findall(text)),
                        "sentence_count": len([s for s in SENT_SPLIT.split(text.strip()) if s.strip()]),
                        "score": round(sc, 4),
                        "hits": {k: v for k, v in h.items() if v},
                        "named_spans": NAMED.findall(text)[:12],
                        "snippet": evidence_snippet(text, h),
                        "text": norm(text)[:1200],
                    }
                    pool[fam].append(rec)
    selected: List[dict] = []
    seen_rows = set()
    per_source_cap = max(8, per_family // 3)
    for fam, rows in pool.items():
        rows.sort(key=lambda r: (-r["score"], r["row_index"]))
        fam_src = collections.Counter()
        fam_sel = []
        # First pass: balanced source buckets and no duplicate rows across families.
        for r in rows:
            if len(fam_sel) >= per_family:
                break
            if r["row_index"] in seen_rows:
                continue
            if fam_src[r["source_bucket"]] >= per_source_cap:
                continue
            fam_sel.append(r)
            fam_src[r["source_bucket"]] += 1
            seen_rows.add(r["row_index"])
        # Second pass: fill if balancing was too strict.
        for r in rows:
            if len(fam_sel) >= per_family:
                break
            if r["row_index"] in seen_rows:
                continue
            fam_sel.append(r)
            fam_src[r["source_bucket"]] += 1
            seen_rows.add(r["row_index"])
        selected.extend(fam_sel)
        counts[f"selected::{fam}"] = len(fam_sel)
        for src, n in fam_src.items():
            counts[f"selected::{fam}::{src}"] = n
    summary = {
        "status": "STRUCTURAL_PACKET_CANDIDATES",
        "corpus": str(corpus.relative_to(ROOT)),
        "corpus_sha256": sha256(corpus),
        "rows_scanned": counts["rows"],
        "excluded_generated_rows": counts["excluded_generated"],
        "per_family_target": per_family,
        "selected_total": len(selected),
        "selected_by_family": {fam: counts[f"selected::{fam}"] for fam in FAMILY_DEFS},
        "selected_by_family_source": {
            fam: {k.split("::")[-1]: v for k, v in counts.items() if k.startswith(f"selected::{fam}::")}
            for fam in FAMILY_DEFS
        },
        "notes": "Rows are candidates for verification, not accepted relational graphs or training data.",
    }
    return summary, selected


def write(summary: dict, selected: List[dict], out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    json_path = out / "structural_packet_candidate_summary.json"
    jsonl_path = out / "structural_packet_candidates.jsonl"
    md_path = out / "structural_packet_candidates.md"
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    with jsonl_path.open("w", encoding="utf-8") as f:
        for r in selected:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    byfam = collections.defaultdict(list)
    for r in selected:
        byfam[r["family"]].append(r)
    lines = [
        "# research strict structural packet candidates",
        "",
        "This source-only selector excludes generated compact/rewrite rows and balances explicit relation/event-state candidates by family. It does not use official evaluation items or model scores. Each row still needs proposition-level verification before it can be used in a packet or experiment.",
        "",
        f"- corpus: `{summary['corpus']}`",
        f"- SHA256: `{summary['corpus_sha256']}`",
        f"- rows scanned: `{summary['rows_scanned']}`; generated rows excluded: `{summary['excluded_generated_rows']}`",
        f"- selected total: `{summary['selected_total']}`",
        "",
        "## Family counts",
    ]
    for fam, n in summary["selected_by_family"].items():
        lines.append(f"- {fam}: {n}; sources {summary['selected_by_family_source'].get(fam,{})}")
    lines.append("")
    lines.append("## Representative snippets")
    for fam in FAMILY_DEFS:
        lines.append(f"\n### {fam}")
        for r in byfam[fam][:8]:
            lines.append(f"- row {r['row_index']} ex={r['example_id']} src={r['source_bucket']} score={r['score']} hits={r['hits']}: {r['snippet']}")
    lines += [
        "",
        "## Direct reading",
        "These candidates make the graph-preserving experience route feasible as a small source-derived packet, but not yet worth training. The next scientific work is to choose a small verified subset and test cross-view/role/order/state transfer against same-source equal-word ordinary compaction and structure-changed controls. If the selected rows reduce to obvious marker completion or proposition drift, this route should stop before generation scale-up.",
        "",
        f"JSON: `{json_path.relative_to(ROOT)}`",
        f"JSONL: `{jsonl_path.relative_to(ROOT)}`",
    ]
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", type=Path, default=CORPUS)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--per-family", type=int, default=80)
    args = ap.parse_args()
    corpus = args.corpus if args.corpus.is_absolute() else ROOT / args.corpus
    out = args.out if args.out.is_absolute() else ROOT / args.out
    summary, selected = select(corpus, args.per_family)
    write(summary, selected, out)
    print(json.dumps({
        "status": summary["status"],
        "selected_total": summary["selected_total"],
        "selected_by_family": summary["selected_by_family"],
        "out_summary": str((out / "structural_packet_candidate_summary.json").relative_to(ROOT)),
        "out_jsonl": str((out / "structural_packet_candidates.jsonl").relative_to(ROOT)),
        "out_md": str((out / "structural_packet_candidates.md").relative_to(ROOT)),
    }, indent=2))


if __name__ == "__main__":
    main()
