#!/usr/bin/env python3
"""Attribution of research generated-rewrite score-text overlaps.

The broad research scan found exact score-text spans in the inherited clean-Qwen
paraphrase rewrites, but none in the compact FineWeb rewrites. This script asks a
narrower question: for each generated-rewrite span, was the exact same span already
present in that pair's original/source text, or is it rewrite-only for that pair?
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import csv
import json
import pathlib
import re
import time
from collections import Counter, defaultdict
from typing import Any

ROOT = _public_path('.')
STUDY = _public_path('experiments/archive/representation_and_objectives')
OUT_DIR = _public_path('experiments/archive/representation_and_objectives/data/generated_view_overlap_scan')
DETAIL_CSV = _public_path('experiments/archive/representation_and_objectives/data/generated_view_overlap_scan/generated_rewrite_overlap_details.csv')
OUT_JSON = _public_path('experiments/archive/representation_and_objectives/data/generated_view_overlap_scan/rewrite_overlap_attribution.json')
OUT_CSV = _public_path('experiments/archive/representation_and_objectives/data/generated_view_overlap_scan/rewrite_overlap_attribution_pair_ngrams.csv')
NOTE = _public_path('research/notes/representation_and_objectives/rewrite_overlap_attribution.md')

COMPACT_PAIRS = _public_path('experiments/archive/frontier_consolidation/data/density_core_reinvestment_medium_riskhard/selected_compact_reinvest_pairs.jsonl')
QWEN_SELECTED_PAIRS = _public_path('experiments/archive/compact_experience/data/qwen_clean_aligned/selected_pairs.jsonl')
WORD_RE = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?")


def toks(text: str) -> list[str]:
    return [m.group(0).lower() for m in WORD_RE.finditer(text or "")]


def ngrams(ts: list[str], n: int) -> set[str]:
    if len(ts) < n:
        return set()
    return {" ".join(ts[i:i+n]) for i in range(len(ts) - n + 1)}


def load_jsonl_by(path: pathlib.Path, key: str) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            out[str(obj[key])] = obj
    return out


def classify_source(row: dict[str, str], qwen_pairs: dict[str, dict[str, Any]], compact_pairs: dict[str, dict[str, Any]]) -> dict[str, Any]:
    pid = row["pair_id"]
    n = int(row["n"])
    ng = row["ngram"]
    kind = row["component_kind"]
    if kind == "official_source_qwen_paraphrase_rewrite":
        p = qwen_pairs[pid]
        original = p.get("original", "")
        rewrite = p.get("rewrite", "")
        source = p.get("source")
        example_id = p.get("example_id")
        cohort = p.get("cohort")
        original_words = p.get("original_words")
        rewrite_words = p.get("rewrite_words")
    elif kind == "fineweb_compact_qwen_rewrite":
        p = compact_pairs[pid]
        original = p.get("source_text", "")
        rewrite = p.get("rewrite_text", "")
        source = "fineweb"
        example_id = p.get("sentence_id")
        cohort = p.get("doc_id")
        original_words = p.get("source_words")
        rewrite_words = p.get("rewrite_words")
    else:
        raise ValueError(f"unexpected generated rewrite kind {kind}")
    original_has_ngram = ng in ngrams(toks(original), n)
    rewrite_has_ngram = ng in ngrams(toks(rewrite), n)
    return {
        "n": n,
        "ngram": ng,
        "component_kind": kind,
        "pair_id": pid,
        "source": source,
        "example_id": example_id,
        "cohort_or_doc": cohort,
        "original_has_ngram": original_has_ngram,
        "rewrite_has_ngram": rewrite_has_ngram,
        "attribution": "copied_or_preserved_from_pair_original" if original_has_ngram else "rewrite_only_within_pair",
        "original_words": original_words,
        "rewrite_words": rewrite_words,
        "original_text_prefix": original[:260].replace("\n", " "),
        "rewrite_text_prefix": rewrite[:260].replace("\n", " "),
    }


def main() -> None:
    t0 = time.time()
    qwen_pairs = load_jsonl_by(QWEN_SELECTED_PAIRS, "pair_id")
    compact_pairs = load_jsonl_by(COMPACT_PAIRS, "pair_id")

    raw_rows: list[dict[str, str]] = []
    with DETAIL_CSV.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            raw_rows.append(row)

    # Collapse duplicate eval hits; the pair-level attribution should count a generated phrase once per pair/ngram/n/kind.
    grouped: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    eval_hit_counter: Counter[tuple[str, str, str, str]] = Counter()
    eval_top_dirs: dict[tuple[str, str, str, str], Counter[str]] = defaultdict(Counter)
    for row in raw_rows:
        key = (row["component_kind"], row["pair_id"], row["n"], row["ngram"])
        eval_hit_counter[key] += 1
        eval_top_dirs[key].update([row.get("eval_top_dir", "")])
        if key not in grouped:
            rec = classify_source(row, qwen_pairs, compact_pairs)
            grouped[key] = rec

    pair_ngram_rows: list[dict[str, Any]] = []
    for key, rec in grouped.items():
        rr = dict(rec)
        rr["eval_hit_rows_in_detail_csv"] = eval_hit_counter[key]
        rr["eval_top_dirs"] = dict(eval_top_dirs[key])
        pair_ngram_rows.append(rr)
    pair_ngram_rows.sort(key=lambda r: (r["component_kind"], r["attribution"], -int(r["n"]), r["pair_id"], r["ngram"]))

    summary: dict[str, Any] = {
        "raw_detail_rows": len(raw_rows),
        "unique_pair_ngrams": len(pair_ngram_rows),
        "by_component_kind": {},
        "by_component_kind_and_n": {},
        "by_attribution": {},
        "rewrite_only_examples": [],
        "copied_or_preserved_examples": [],
    }
    for r in pair_ngram_rows:
        kind = r["component_kind"]
        attr = r["attribution"]
        n = r["n"]
        summary["by_component_kind"].setdefault(kind, {"unique_pair_ngrams": 0, "eval_hit_rows": 0, "attribution": Counter()})
        summary["by_component_kind"][kind]["unique_pair_ngrams"] += 1
        summary["by_component_kind"][kind]["eval_hit_rows"] += r["eval_hit_rows_in_detail_csv"]
        summary["by_component_kind"][kind]["attribution"].update([attr])
        kn = f"{kind}::n{n}"
        summary["by_component_kind_and_n"].setdefault(kn, {"unique_pair_ngrams": 0, "eval_hit_rows": 0, "attribution": Counter()})
        summary["by_component_kind_and_n"][kn]["unique_pair_ngrams"] += 1
        summary["by_component_kind_and_n"][kn]["eval_hit_rows"] += r["eval_hit_rows_in_detail_csv"]
        summary["by_component_kind_and_n"][kn]["attribution"].update([attr])
        summary["by_attribution"].setdefault(attr, {"unique_pair_ngrams": 0, "eval_hit_rows": 0, "component_kinds": Counter(), "n": Counter()})
        summary["by_attribution"][attr]["unique_pair_ngrams"] += 1
        summary["by_attribution"][attr]["eval_hit_rows"] += r["eval_hit_rows_in_detail_csv"]
        summary["by_attribution"][attr]["component_kinds"].update([kind])
        summary["by_attribution"][attr]["n"].update([str(n)])
        target_list = "rewrite_only_examples" if attr == "rewrite_only_within_pair" else "copied_or_preserved_examples"
        if len(summary[target_list]) < 25:
            summary[target_list].append({k: r[k] for k in ["n", "ngram", "component_kind", "pair_id", "source", "original_has_ngram", "rewrite_has_ngram", "original_text_prefix", "rewrite_text_prefix", "eval_top_dirs"] if k in r})

    def jsonable(x: Any) -> Any:
        if isinstance(x, Counter):
            return dict(x)
        if isinstance(x, dict):
            return {k: jsonable(v) for k, v in x.items()}
        if isinstance(x, list):
            return [jsonable(v) for v in x]
        return x

    payload = {
        "status": "REWRITE_OVERLAP_ATTRIBUTION",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "purpose": "Attribute generated-rewrite exact score-text spans to pair-original-preserved versus rewrite-only within the same source/rewrite pair.",
        "inputs": {
            "detail_csv": str(DETAIL_CSV),
            "qwen_selected_pairs": str(QWEN_SELECTED_PAIRS),
            "compact_pairs": str(COMPACT_PAIRS),
        },
        "summary": jsonable(summary),
        "outputs": {"json": str(OUT_JSON), "csv": str(OUT_CSV), "note": str(NOTE)},
        "interpretation": "Pair-level rewrite-only spans show teacher-generated wording that exactly matches scored text; copied/preserved spans were already present in the allowed original counterpart for that training pair. Both are sparse compared with the full generated-rewrite mass, and compact FineWeb rewrites had zero spans in the upstream research detail file.",
        "elapsed_sec": round(time.time() - t0, 3),
    }

    # CSV output.
    keys = [
        "n", "ngram", "component_kind", "pair_id", "source", "example_id", "cohort_or_doc",
        "attribution", "original_has_ngram", "rewrite_has_ngram", "eval_hit_rows_in_detail_csv",
        "eval_top_dirs", "original_words", "rewrite_words", "original_text_prefix", "rewrite_text_prefix",
    ]
    with OUT_CSV.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in pair_ngram_rows:
            rr = {k: r.get(k, "") for k in keys}
            rr["eval_top_dirs"] = json.dumps(rr["eval_top_dirs"], ensure_ascii=False, sort_keys=True)
            w.writerow(rr)

    OUT_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    by_attr = payload["summary"]["by_attribution"]
    note = [
        "# research rewrite-overlap attribution",
        "",
        "This CPU-only follow-up attributes exact score-text spans found in generated rewrites to the same pair's original/source text.",
        "",
        f"- Raw detail rows from research: {len(raw_rows)}",
        f"- Unique generated-rewrite pair/ngram records: {len(pair_ngram_rows)}",
        f"- Attribution summary: `{json.dumps(by_attr, ensure_ascii=False, sort_keys=True)}`",
        "",
        "## Interpretation",
        "The compact FineWeb Qwen rewrites had no exact 7/8/10-token scored-text spans in research. The inherited clean-Qwen paraphrase rewrites have sparse spans; this file separates spans already present in the paired original from rewrite-only spans introduced or preserved by the teacher rewrite process. This is a defensibility/provenance record, not a model-score explanation.",
        "",
        f"JSON: `{OUT_JSON}`",
        f"CSV: `{OUT_CSV}`",
    ]
    NOTE.write_text("\n".join(note) + "\n", encoding="utf-8")

    print(json.dumps({
        "status": payload["status"],
        "elapsed_sec": payload["elapsed_sec"],
        "summary": payload["summary"],
        "out_json": str(OUT_JSON),
        "out_csv": str(OUT_CSV),
        "note": str(NOTE),
    }, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
