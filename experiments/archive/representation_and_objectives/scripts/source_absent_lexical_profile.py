#!/usr/bin/env python3
"""research: lexical profile of packed source-absent labels vs copied control.

This script is CPU-only and does not inspect pending model outputs.  It describes
what kinds of compact-rewrite words the two target-selective arms remove in the
historical packed pool: all source-absent content words versus the matched copied
whole-word control.  The goal is to understand whether the channel is mainly rare
lexical coverage, numbers/names, or a broad set of compact relational/abstractive
rewriting tokens.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import pathlib
import re
import statistics
import sys
import time
from typing import Any

USER_ROOT = pathlib.Path(".").resolve()
SCRIPT_DIR = USER_ROOT / "experiments/archive/representation_and_objectives/scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from annotate_packed_pool import norm_word, word_class, CAT_RW_ABS_CONTENT, CAT_RW_COPIED  # noqa: E402

POOL = pathlib.Path("experiments/archive/frontier_consolidation/data/density_cleanqwen_overlay_medium_riskhard/cleanqwen_fineweb_compact_view_reinvest_10M.jsonl")
ANNOTATION = pathlib.Path("experiments/archive/representation_and_objectives/data/packed_pool_annotation/packed_pool_annotations.jsonl")
SELECTION = pathlib.Path("experiments/archive/representation_and_objectives/data/packed_pool_annotation/packed_wholeword_copied_selection.jsonl")
OUT_DIR = pathlib.Path("experiments/archive/representation_and_objectives/data/source_absent_lexical_profile")

RELATIONAL_WORDS = set("""
able about across action active activity actor actors actually added adds affects after allows also among another appears area areas around associated because become becomes became before being between based called cause caused causes change changes changing common commonly compared connected consists contain contained contains created creates defined described different during effects either especially events example examples formed found from function functions general generally gives group groups happens include included includes including into known leads located made major means needed occurs often part parts possible process processes produced produces property properties provides refers related remains result results says several shows since states system systems than through type types used uses using where while within works
""".split())
GENERIC_ENTITY_WORDS = set("""
animal animals area areas body building buildings city cities country countries culture cultures family families group groups language languages material materials member members people person place places plant plants region regions species state states system systems thing things time times water work works world year years
""".split())
EVENT_STATE_WORDS = set("""
add added adds become becomes became broken built carried changed changes changing closed created creates damaged developed discovered done drop dropped dry eaten fall falls fell filled found frozen gave gives got grow grows has held hidden increased left located made make makes moved moves named opened placed produced removed said sent set shown stuck taken turn turned used went won wrote
""".split())


def sha256_file(path: pathlib.Path) -> str | None:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def qstats(xs: list[float]) -> dict[str, Any]:
    if not xs:
        return {"n": 0}
    s = sorted(float(x) for x in xs)
    n = len(s)
    def q(p: float) -> float:
        return s[min(n - 1, max(0, int(round(p * (n - 1)))))]
    return {"n": n, "mean": round(statistics.mean(s), 6), "median": round(statistics.median(s), 6), "p05": round(q(0.05), 6), "p25": round(q(0.25), 6), "p75": round(q(0.75), 6), "p95": round(q(0.95), 6), "min": round(s[0], 6), "max": round(s[-1], 6)}


def entropy(counter: collections.Counter[str]) -> float:
    tot = sum(counter.values())
    if tot <= 0:
        return 0.0
    return -sum((v / tot) * math.log(v / tot) for v in counter.values() if v > 0)


def load_annotations(path: pathlib.Path) -> tuple[list[dict[str, Any]], dict[int, int]]:
    anns = []
    row_to_eid = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            a = json.loads(line)
            anns.append(a)
            row_to_eid[int(a["row_index"])] = int(a["example_id"])
    return anns, row_to_eid


def load_selection(path: pathlib.Path, row_to_eid: dict[int, int]) -> set[tuple[int, int]]:
    s = set()
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            g = json.loads(line)
            eid = row_to_eid.get(int(g["row_index"]))
            if eid is not None:
                s.add((eid, int(g["word_index"])))
    return s


def flags(surface: str, nw: str) -> list[str]:
    fs = []
    if any(c.isdigit() for c in surface):
        fs.append("number_or_year")
    if surface[:1].isupper() and not surface.isupper():
        fs.append("capitalized")
    if "-" in surface:
        fs.append("hyphenated")
    if nw in RELATIONAL_WORDS:
        fs.append("relational_or_abstracting_word")
    if nw in GENERIC_ENTITY_WORDS:
        fs.append("generic_entity_word")
    if nw in EVENT_STATE_WORDS:
        fs.append("event_or_state_word")
    if len(nw) <= 3:
        fs.append("short_norm")
    if not fs:
        fs.append("other_content")
    return fs


def summarize_records(records: list[dict[str, Any]], pool_freq: collections.Counter[str]) -> dict[str, Any]:
    norms = [r["norm"] for r in records]
    type_counts = collections.Counter(norms)
    cls = collections.Counter(r["word_class"] for r in records)
    flag_counts = collections.Counter(f for r in records for f in r["flags"])
    top = [{"norm": w, "count": int(c), "pool_freq": int(pool_freq[w])} for w, c in type_counts.most_common(60)]
    tot = len(records)
    type_tot = len(type_counts)
    singleton_types = sum(1 for v in type_counts.values() if v == 1)
    top10_mass = sum(c for _, c in type_counts.most_common(10)) / max(1, tot)
    support_values = [float(pool_freq[w]) for w in norms]
    return {
        "n_tokens_whitespace_words": tot,
        "n_types": type_tot,
        "type_token_ratio": round(type_tot / max(1, tot), 6),
        "singleton_type_fraction": round(singleton_types / max(1, type_tot), 6),
        "top10_token_mass": round(top10_mass, 6),
        "entropy_nats": round(entropy(type_counts), 6),
        "word_class_counts": dict(cls),
        "flag_counts": dict(flag_counts),
        "flag_fraction": {k: round(v / max(1, tot), 6) for k, v in sorted(flag_counts.items())},
        "norm_length": qstats([len(w) for w in norms]),
        "pool_whitespace_support_for_norm": qstats(support_values),
        "top_norms": top,
    }


def example_rows(records: list[dict[str, Any]], wanted_flags: list[str], n: int = 30) -> list[dict[str, Any]]:
    out = []
    seen = set()
    wanted = set(wanted_flags)
    for r in records:
        if wanted and not (wanted & set(r["flags"])):
            continue
        key = (r["row_index"], r["word_index"])
        if key in seen:
            continue
        seen.add(key)
        out.append({k: r[k] for k in ["row_index", "word_index", "surface", "norm", "word_class", "flags", "left_context", "right_context"]})
        if len(out) >= n:
            break
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default=str(POOL))
    ap.add_argument("--annotation", default=str(ANNOTATION))
    ap.add_argument("--selection", default=str(SELECTION))
    ap.add_argument("--output_dir", default=str(OUT_DIR))
    args = ap.parse_args()

    t0 = time.time()
    out_dir = pathlib.Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    anns, row_to_eid = load_annotations(pathlib.Path(args.annotation))
    selected = load_selection(pathlib.Path(args.selection), row_to_eid)
    pool_rows = []
    pool_freq = collections.Counter()
    with pathlib.Path(args.pool).open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            words = str(obj["text"]).split()
            pool_rows.append({"text": str(obj["text"]), "words": words, "example_id": int(obj.get("example_id", len(pool_rows)))})
            pool_freq.update(norm_word(w) for w in words)

    abs_records: list[dict[str, Any]] = []
    copied_selected_records: list[dict[str, Any]] = []
    copied_all_records: list[dict[str, Any]] = []
    for ann in anns:
        row_idx = int(ann["row_index"])
        eid = int(ann["example_id"])
        if row_idx >= len(pool_rows):
            continue
        words = pool_rows[row_idx]["words"]
        for wi, cat in enumerate(ann["word_categories"]):
            if wi >= len(words):
                continue
            if cat not in {CAT_RW_ABS_CONTENT, CAT_RW_COPIED}:
                continue
            surf = words[wi]
            nw = norm_word(surf)
            rec = {
                "row_index": row_idx,
                "example_id": eid,
                "word_index": wi,
                "surface": surf,
                "norm": nw,
                "word_class": word_class(surf),
                "flags": flags(surf, nw),
                "left_context": " ".join(words[max(0, wi - 5):wi]),
                "right_context": " ".join(words[wi + 1: min(len(words), wi + 6)]),
            }
            if cat == CAT_RW_ABS_CONTENT:
                abs_records.append(rec)
            elif cat == CAT_RW_COPIED and word_class(surf) in {"content", "number"}:
                copied_all_records.append(rec)
                if (eid, wi) in selected:
                    copied_selected_records.append(rec)

    abs_types = set(r["norm"] for r in abs_records)
    sel_types = set(r["norm"] for r in copied_selected_records)
    allcop_types = set(r["norm"] for r in copied_all_records)
    payload = {
        "status": "SOURCE_ABSENT_LEXICAL_PROFILE",
        "meaning": "Lexical/heuristic profile of compact source-absent content words and the matched copied whole-word control in the packed 10M pool. CPU-only; no endpoint outputs read.",
        "inputs": {"pool": args.pool, "pool_sha256": sha256_file(pathlib.Path(args.pool)), "annotation": args.annotation, "annotation_sha256": sha256_file(pathlib.Path(args.annotation)), "selection": args.selection, "selection_sha256": sha256_file(pathlib.Path(args.selection))},
        "record_counts": {"source_absent_content_words_per_10M_pool": len(abs_records), "all_copied_content_words_per_10M_pool": len(copied_all_records), "selected_copied_control_words_per_10M_pool": len(copied_selected_records)},
        "type_overlap": {
            "abs_types": len(abs_types),
            "selected_copied_types": len(sel_types),
            "all_copied_content_types": len(allcop_types),
            "abs_selected_jaccard": round(len(abs_types & sel_types) / max(1, len(abs_types | sel_types)), 6),
            "abs_covered_by_selected_type_fraction": round(len(abs_types & sel_types) / max(1, len(abs_types)), 6),
            "abs_covered_by_any_copied_content_type_fraction": round(len(abs_types & allcop_types) / max(1, len(abs_types)), 6),
        },
        "source_absent_content": summarize_records(abs_records, pool_freq),
        "selected_copied_control": summarize_records(copied_selected_records, pool_freq),
        "all_copied_content_available": summarize_records(copied_all_records, pool_freq),
        "examples": {
            "source_absent_relational_or_abstracting": example_rows(abs_records, ["relational_or_abstracting_word"], 40),
            "source_absent_event_or_state": example_rows(abs_records, ["event_or_state_word"], 40),
            "source_absent_number_or_year": example_rows(abs_records, ["number_or_year"], 30),
            "selected_copied_relational_or_abstracting": example_rows(copied_selected_records, ["relational_or_abstracting_word"], 40),
        },
        "interpretation": {
            "why_it_matters": "If source-absent compact labels are diverse and rich in relation/state/abstraction words rather than just rare names or numbers, then a later endpoint effect is more plausibly tied to compact rewriting semantics. If they are mostly lexical tail coverage, interpretation should be narrower.",
            "limits": "Heuristic word flags are descriptive only and use no learned tagger; final mechanism reading depends on the trained arm comparison.",
        },
        "elapsed_sec": round(time.time() - t0, 1),
    }
    out_json = out_dir / "source_absent_lexical_profile.json"
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    md = [
        "# research source-absent lexical profile",
        "",
        f"JSON: `{out_json}`",
        "",
        "## Counts",
        "",
        json.dumps(payload["record_counts"], indent=2, ensure_ascii=False),
        "",
        "## Type overlap",
        "",
        json.dumps(payload["type_overlap"], indent=2, ensure_ascii=False),
        "",
        "## Flag fractions",
        "",
        "Source-absent content:",
        json.dumps(payload["source_absent_content"]["flag_fraction"], indent=2, ensure_ascii=False),
        "",
        "Selected copied control:",
        json.dumps(payload["selected_copied_control"]["flag_fraction"], indent=2, ensure_ascii=False),
        "",
        "## Top source-absent norms",
        "",
    ]
    for r in payload["source_absent_content"]["top_norms"][:40]:
        md.append(f"- {r['norm']}: {r['count']} (pool freq {r['pool_freq']})")
    md += ["", "## Reading", "", payload["interpretation"]["why_it_matters"], ""]
    (out_dir / "source_absent_lexical_profile.md").write_text("\n".join(md), encoding="utf-8")
    print(json.dumps({"status": payload["status"], "out_json": str(out_json), "record_counts": payload["record_counts"], "type_overlap": payload["type_overlap"], "source_absent_flag_fraction": payload["source_absent_content"]["flag_fraction"], "selected_copied_flag_fraction": payload["selected_copied_control"]["flag_fraction"], "elapsed_sec": payload["elapsed_sec"]}, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
