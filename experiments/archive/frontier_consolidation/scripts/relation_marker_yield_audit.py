#!/usr/bin/env python3
"""research: source-only audit for relation/event-state material in the legal corpus.

This is deliberately not an official-benchmark coverage search.  It scans only the
legal BabyLM training pool used by the compact-view reinvest substrate and asks
whether there is enough explicit, auditable source material for a possible
future graph-preserving relational-experience diagnostic.
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
import math
import re
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

ROOT = _public_path('.')
DEFAULT_CORPUS = _public_path('experiments/archive/frontier_consolidation/data/density_cleanqwen_overlay_medium_riskhard/cleanqwen_fineweb_compact_view_reinvest_10M.jsonl')
DEFAULT_OUT = _public_path('experiments/archive/frontier_consolidation/data/relation_marker_yield_audit')

# Broad, official-free linguistic/event categories.  These are not selected from
# BabyLM eval item text; they are generic corpus phenomena relevant to event/state
# and discourse learning.
FAMILIES: Dict[str, List[str]] = {
    "temporal_order": [
        r"\bbefore\b", r"\bafter\b", r"\bthen\b", r"\bwhen\b", r"\bwhile\b",
        r"\bonce\b", r"\blater\b", r"\bpreviously\b", r"\bsubsequently\b",
        r"\bfinally\b", r"\bfirst\b", r"\bnext\b", r"\buntil\b", r"\bsince\b",
        r"\bduring\b", r"\bas soon as\b", r"\bmeanwhile\b",
    ],
    "causal_consequence": [
        r"\bbecause\b", r"\bso that\b", r"\bso\b", r"\btherefore\b",
        r"\bthus\b", r"\bhence\b", r"\bdue to\b", r"\bas a result\b",
        r"\bresult(?:ed|s|ing)? in\b", r"\bcaus(?:e|ed|es|ing)\b",
        r"\bled to\b", r"\bconsequently\b", r"\bif\b", r"\bunless\b",
    ],
    "entity_reference": [
        r"\bhe\b", r"\bshe\b", r"\bthey\b", r"\bhim\b", r"\bher\b", r"\bthem\b",
        r"\bhis\b", r"\btheir\b", r"\bhers\b", r"\btheirs\b", r"\bit\b", r"\bits\b",
        r"\bwho\b", r"\bwhich\b", r"\bthat\b", r"\bthis\b", r"\bthese\b", r"\bthose\b",
    ],
    "state_change_motion": [
        r"\bput\b", r"\bputs\b", r"\bplaced?\b", r"\bmoved?\b", r"\bwent\b",
        r"\bgoes\b", r"\breturned?\b", r"\bleft\b", r"\bentered?\b", r"\barrived?\b",
        r"\btook\b", r"\btakes\b", r"\bgave\b", r"\bgives\b", r"\bchanged?\b",
        r"\bbecame\b", r"\bbecome\b", r"\bturned\b", r"\bopened?\b", r"\bclosed?\b",
        r"\bbuilt\b", r"\bbroke\b", r"\bdied\b", r"\bcreated?\b", r"\bformed?\b",
    ],
    "quantity_comparison": [
        r"\b\d+(?:[\.,]\d+)?\b", r"\bone\b", r"\btwo\b", r"\bthree\b", r"\bfour\b",
        r"\bfive\b", r"\bsix\b", r"\bseven\b", r"\beight\b", r"\bnine\b", r"\bten\b",
        r"\bmore than\b", r"\bless than\b", r"\bfewer than\b", r"\bat least\b", r"\bat most\b",
        r"\bpercent\b", r"\bper cent\b", r"\bhalf\b", r"\bdouble\b", r"\btwice\b",
        r"\bincreased?\b", r"\bdecreased?\b", r"\breduced?\b", r"\bgrew\b",
    ],
    "social_mental_speech": [
        r"\bsaid\b", r"\bsays\b", r"\btold\b", r"\basked\b", r"\banswered\b",
        r"\bbelieved?\b", r"\bthought\b", r"\bknew\b", r"\bwanted\b", r"\bdecided\b",
        r"\bpromised\b", r"\bagreed\b", r"\brefused\b", r"\bclaimed\b", r"\breported\b",
    ],
    "polarity_modality": [
        r"\bnot\b", r"\bno\b", r"\bnever\b", r"\bwithout\b", r"\bcan't\b", r"\bcannot\b",
        r"\bshould\b", r"\bwould\b", r"\bcould\b", r"\bmight\b", r"\bmay\b", r"\bmust\b",
        r"\bperhaps\b", r"\bmaybe\b", r"\blikely\b", r"\bunlikely\b", r"\bpossible\b",
    ],
    "contrast_alternative": [
        r"\bbut\b", r"\bhowever\b", r"\balthough\b", r"\bthough\b", r"\bwhereas\b",
        r"\binstead\b", r"\brather than\b", r"\beither\b", r"\bor\b", r"\bneither\b",
    ],
}
COMPILED = {k: [re.compile(p, flags=re.I) for p in pats] for k, pats in FAMILIES.items()}
SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")
CAPITAL_SEQ = re.compile(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3}\b")
TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)?")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def source_bucket(src: str) -> str:
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


def family_hits(text: str) -> Dict[str, int]:
    return {fam: sum(len(rx.findall(text)) for rx in rxs) for fam, rxs in COMPILED.items()}


def clean_text_score(text: str) -> float:
    # Penalize markup and very short fragments; favor multiple sentences and normal punctuation.
    words = TOKEN_RE.findall(text)
    n_words = len(words)
    sent_count = max(1, len([s for s in SENT_SPLIT.split(text.strip()) if s.strip()]))
    weird = len(re.findall(r"https?://|www\.|<[^>]+>|[_=]{2,}|\{\{|\}\}", text))
    quote_density = (text.count('"') + text.count("'") + text.count("`") + text.count("{") + text.count("}")) / max(1, len(text))
    length_term = 1.0 if 25 <= n_words <= 180 else 0.5 if 12 <= n_words <= 250 else 0.1
    sentence_term = min(1.0, sent_count / 3.0)
    return length_term + 0.5 * sentence_term - 0.8 * weird - 5.0 * quote_density


def row_score(text: str, hits: Dict[str, int]) -> float:
    active = sum(1 for v in hits.values() if v > 0)
    dense = sum(min(v, 3) for v in hits.values())
    # Favor rows with one explicit state/event family and one discourse/relation family.
    eventish = int(hits.get("state_change_motion", 0) > 0) + int(hits.get("temporal_order", 0) > 0) + int(hits.get("causal_consequence", 0) > 0)
    referential = int(hits.get("entity_reference", 0) > 0) + int(hits.get("social_mental_speech", 0) > 0)
    quantitative = int(hits.get("quantity_comparison", 0) > 0)
    return 2.0 * active + 0.4 * dense + 2.0 * eventish + 1.2 * referential + 1.0 * quantitative + clean_text_score(text)


def truncate(s: str, n: int = 420) -> str:
    s = re.sub(r"\s+", " ", s.strip())
    return s if len(s) <= n else s[: n - 3] + "..."


def scan(corpus: Path, max_rows: int | None = None) -> Tuple[dict, List[dict]]:
    counters = {
        "rows": 0,
        "words_field_sum": 0,
        "token_count_sum": 0,
        "sha256": sha256_file(corpus),
        "family_rows": collections.Counter(),
        "family_hits": collections.Counter(),
        "source_rows": collections.Counter(),
        "source_words": collections.Counter(),
        "source_family_rows": collections.defaultdict(collections.Counter),
        "active_family_hist": collections.Counter(),
        "candidate_rows": 0,
        "candidate_words": 0,
        "multi_family_rows": 0,
        "clean_candidate_rows": 0,
    }
    top: List[dict] = []
    by_combo: Dict[Tuple[str, ...], List[dict]] = collections.defaultdict(list)
    rng_mod = 1000003

    with corpus.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if max_rows is not None and idx >= max_rows:
                break
            if not line.strip():
                continue
            row = json.loads(line)
            text = str(row.get("text") or "")
            src = str(row.get("source") or "UNKNOWN")
            bucket = source_bucket(src)
            words_field = int(row.get("words") or len(TOKEN_RE.findall(text)))
            token_count = len(TOKEN_RE.findall(text))
            hits = family_hits(text)
            active = tuple(sorted(k for k, v in hits.items() if v > 0))
            score = row_score(text, hits)
            is_candidate = (
                len(active) >= 2
                and (hits["temporal_order"] or hits["causal_consequence"] or hits["state_change_motion"] or hits["quantity_comparison"])
                and clean_text_score(text) > 0.45
                and 20 <= token_count <= 220
            )
            counters["rows"] += 1
            counters["words_field_sum"] += words_field
            counters["token_count_sum"] += token_count
            counters["source_rows"][bucket] += 1
            counters["source_words"][bucket] += words_field
            counters["active_family_hist"][len(active)] += 1
            if len(active) >= 2:
                counters["multi_family_rows"] += 1
            if is_candidate:
                counters["candidate_rows"] += 1
                counters["candidate_words"] += words_field
            if clean_text_score(text) > 0.45 and len(active) >= 2:
                counters["clean_candidate_rows"] += 1
            for fam, n in hits.items():
                if n:
                    counters["family_rows"][fam] += 1
                    counters["family_hits"][fam] += n
                    counters["source_family_rows"][bucket][fam] += 1
            if is_candidate:
                rec = {
                    "row_index": idx,
                    "example_id": row.get("example_id"),
                    "source": src,
                    "source_bucket": bucket,
                    "words": words_field,
                    "token_count": token_count,
                    "active_families": list(active),
                    "family_hits": {k: hits[k] for k in active},
                    "score": round(score, 4),
                    "capitalized_spans": CAPITAL_SEQ.findall(text)[:8],
                    "text": truncate(text, 650),
                }
                # Keep deterministic high-score examples, with light diversity by combo/source.
                top.append(rec)
                combo = active[:4]
                if len(by_combo[combo]) < 5:
                    by_combo[combo].append(rec)
    top.sort(key=lambda r: (-r["score"], r["row_index"]))
    sampled = top[:200]
    # Add combo-diverse examples not already in top200.
    seen = {r["row_index"] for r in sampled}
    for combo, rows in sorted(by_combo.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        for r in rows:
            if len(sampled) >= 400:
                break
            if r["row_index"] not in seen:
                sampled.append(r)
                seen.add(r["row_index"])
        if len(sampled) >= 400:
            break
    return counters, sampled


def counter_to_dict(c: collections.Counter) -> dict:
    return {str(k): v for k, v in c.most_common()}


def write_outputs(corpus: Path, out_dir: Path, counters: dict, samples: List[dict]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = counters["rows"]
    summary = {
        "status": "RELATION_MARKER_YIELD_AUDIT",
        "corpus": str(corpus.relative_to(ROOT)),
        "corpus_sha256": counters["sha256"],
        "rows": rows,
        "words_field_sum": counters["words_field_sum"],
        "token_count_sum": counters["token_count_sum"],
        "candidate_rows": counters["candidate_rows"],
        "candidate_words": counters["candidate_words"],
        "candidate_row_fraction": counters["candidate_rows"] / max(1, rows),
        "multi_family_rows": counters["multi_family_rows"],
        "clean_multi_family_rows": counters["clean_candidate_rows"],
        "active_family_hist": counter_to_dict(counters["active_family_hist"]),
        "family_rows": counter_to_dict(counters["family_rows"]),
        "family_hits": counter_to_dict(counters["family_hits"]),
        "source_rows": counter_to_dict(counters["source_rows"]),
        "source_words": counter_to_dict(counters["source_words"]),
        "source_family_rows": {src: counter_to_dict(cnt) for src, cnt in counters["source_family_rows"].items()},
        "sample_records_path": str((out_dir / "candidate_samples.jsonl").relative_to(ROOT)),
    }
    with (out_dir / "relation_marker_yield_audit.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    with (out_dir / "candidate_samples.jsonl").open("w", encoding="utf-8") as f:
        for r in samples:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    fam_lines = []
    for fam, n in counters["family_rows"].most_common():
        fam_lines.append(f"- {fam}: {n} rows ({n/max(1, rows):.3%}), hits={counters['family_hits'][fam]}")
    src_lines = []
    for src, n in counters["source_rows"].most_common():
        src_lines.append(f"- {src}: {n} rows, words={counters['source_words'][src]}")
    examples = []
    for r in samples[:20]:
        examples.append(
            f"- row {r['row_index']} ex={r['example_id']} src={r['source_bucket']} score={r['score']} fam={','.join(r['active_families'])}: {r['text']}"
        )
    md = f"""# research relation/event-state marker yield audit

This is a source-only legal-corpus audit, not a search over official evaluation items.  It estimates whether the current compact-view reinvest pool contains enough explicit relational/event-state material for a possible graph-verified experience diagnostic before any new generation or training.

- corpus: `{summary['corpus']}`
- SHA256: `{summary['corpus_sha256']}`
- rows scanned: `{rows}`
- words from row field: `{summary['words_field_sum']}`
- candidate rows: `{summary['candidate_rows']}` ({summary['candidate_row_fraction']:.3%}), candidate words `{summary['candidate_words']}`
- rows with >=2 broad families: `{summary['multi_family_rows']}`; clean multi-family rows `{summary['clean_multi_family_rows']}`

## Family row counts
{chr(10).join(fam_lines)}

## Source buckets
{chr(10).join(src_lines)}

## First 20 high-score candidate snippets
{chr(10).join(examples)}

## Direct reading
The regex families are intentionally broad and noisy.  Positive yield means only that there is enough source material to build a manually or semi-automatically verified packet.  It does not validate relation/state transfer, generation fidelity, or a training route.  A future discriminator must use same-source equal-word packets and structure-preserving/corrupted controls, and must measure cross-view or renamed-entity transfer rather than raw sentence NLL.

JSON: `{(out_dir / 'relation_marker_yield_audit.json').relative_to(ROOT)}`
Samples: `{(out_dir / 'candidate_samples.jsonl').relative_to(ROOT)}`
"""
    (out_dir / "relation_marker_yield_audit.md").write_text(md, encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--max-rows", type=int, default=0)
    args = ap.parse_args()
    corpus = args.corpus if args.corpus.is_absolute() else (ROOT / args.corpus)
    out = args.out if args.out.is_absolute() else (ROOT / args.out)
    counters, samples = scan(corpus, max_rows=(args.max_rows or None))
    write_outputs(corpus, out, counters, samples)
    print(json.dumps({
        "status": "RELATION_MARKER_YIELD_AUDIT",
        "rows": counters["rows"],
        "candidate_rows": counters["candidate_rows"],
        "candidate_words": counters["candidate_words"],
        "out_json": str((out / "relation_marker_yield_audit.json").relative_to(ROOT)),
        "out_md": str((out / "relation_marker_yield_audit.md").relative_to(ROOT)),
    }, indent=2))


if __name__ == "__main__":
    main()
