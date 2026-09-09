#!/usr/bin/env python3
"""research tight semantic assignment-reversal export.

This is a stricter companion to `semantic_repaired_assignment_reversal_export.py`.
The first research repair removed `located_in` and connector-prefix failures, but independent review
review found additional retained defects: non-ASCII truncations (Jundia/Jundiaí),
hyphenated truncations (Fort/Fort-de-France), administrative/granularity mismatches,
attachment-ambiguous death places, and non-exclusive descriptor relations.

This script audits the ~300 high-confidence underlying triples again and generates a
primary conservative export.  It intentionally favors interpretability over row count:
primary rows include only birth/founded years and direct place-valued birthplace or
death_place facts whose answer strings pass stricter boundary and value-type tests.
Nationality and occupation are recorded but excluded from the primary export because
"actually X" does not reliably revoke multi-valued descriptors.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import importlib.util
import json
import pathlib
import re
import time
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List, Tuple

ROOT = _public_path('.')
STUDY = _public_path('experiments/archive/functional_learning')
SCRIPT = _public_path('experiments/archive/functional_learning/scripts/export_larger_assignment_reversal_pool.py')
SOURCE_SCRIPT = _public_path('experiments/archive/functional_learning/scripts/source_disjoint_assignment_reversal_export.py')
STRICT_SCRIPT = _public_path('experiments/archive/functional_learning/scripts/strict_disjoint_assignment_reversal_export.py')
HIGHCONF_SCRIPT = _public_path('experiments/archive/functional_learning/scripts/highconfidence_strict_assignment_reversal_export.py')
PREV_STEP066_DIR = _public_path('experiments/archive/functional_learning/data/semantic_repaired_assignment_reversal_export')
DEFAULT_OUT = _public_path('experiments/archive/functional_learning/data/tight_semantic_assignment_reversal_export')

TIGHT_CAPS = {
    "death_place": 190,
    "birthplace": 165,
    "birth_year": 150,
    "located_in": 0,
    "founded_year": 20,
    "nationality": 0,
    "occupation": 0,
}

# Known ambiguous propositions identified by the independent review semantic reading.  These are not
# necessarily false, but the retained answer may attach to an event/cause rather than the
# death itself or may be too ambiguous for a clean controlled packet.
KNOWN_ATTACHMENT_AMBIGUOUS = {
    ("death_place", "rw2_041759", "ehiogu", "london"),
    ("death_place", "rw2_046080", "modi", "ahmedabad"),
    ("death_place", "rw2_043689", "issaacson", "melbourne"),
}

# Obvious non-primary/broad or administrative values when the update pool supplies city
# names.  This is a conservative automatic rule, not a world-knowledge oracle.
BROAD_PLACE_TERMS = {
    "county", "counties", "township", "province", "region", "state", "states",
    "country", "republic", "kingdom", "district", "government", "area", "territory",
    "africa", "germany", "philippines", "nigeria", "england", "france", "india",
    "wales", "scotland", "australia", "california", "arizona", "mississippi",
}
ADMIN_FOLLOWERS = {
    "area", "city", "district", "county", "counties", "township", "province",
    "region", "state", "states", "municipality", "department", "borough",
    "suburb", "prefecture", "territory", "republic", "kingdom",
}
NAME_CONNECTORS = {"de", "del", "la", "le", "el", "di", "du", "dos", "das", "van", "von"}


def import_module(name: str, path: pathlib.Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def rel(p: pathlib.Path | str) -> str:
    try:
        return str(pathlib.Path(p).resolve().relative_to(ROOT))
    except Exception:
        return str(p)


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def write_jsonl(path: pathlib.Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def clean_space(s: str) -> str:
    return re.sub(r"\s+", " ", str(s or "").strip())


def lower_key(t: Dict[str, Any]) -> Tuple[str, str, str, str]:
    return (str(t.get("relation", "")), str(t.get("sentence_id", "")), str(t.get("entity", "")).lower(), str(t.get("value", "")).lower())


def find_value_pos(t: Dict[str, Any]) -> int:
    sent = str(t.get("sentence", ""))
    val = str(t.get("value", ""))
    span = t.get("value_span") or []
    if isinstance(span, list) and len(span) == 2:
        try:
            s, e = int(span[0]), int(span[1])
            if 0 <= s <= e <= len(sent) and sent[s:e] == val:
                return s
        except Exception:
            pass
    return sent.find(val)


def exact_entity_occurrences(sentence: str, entity: str) -> int:
    if not entity:
        return 0
    count = 0
    for m in re.finditer(re.escape(entity), sentence, flags=re.I):
        before = sentence[m.start() - 1] if m.start() else " "
        after = sentence[m.end()] if m.end() < len(sentence) else " "
        if before.isalnum() or after.isalnum() or before in "'-’" or after in "'-’":
            continue
        count += 1
    return count


def value_boundary_issue(t: Dict[str, Any]) -> str | None:
    sent = str(t.get("sentence", ""))
    val = str(t.get("value", ""))
    pos = find_value_pos(t)
    if pos < 0:
        return "value_not_literal"
    end = pos + len(val)
    if end < len(sent):
        ch = sent[end]
        if ch.isalpha() or ch.isdigit() or ch in "-'’":
            return "value_immediate_token_continues"
    after = sent[end:end + 60]
    # Incomplete toponyms such as Rio de Janeiro, Benque Viejo del Carmen.
    m = re.match(r"^\s+([A-Za-z][A-Za-z'’.-]*)\b", after)
    if m:
        nxt = m.group(1).lower().strip(".,;:")
        if nxt in NAME_CONNECTORS:
            return f"value_prefix_before_name_connector:{nxt}"
        if nxt in ADMIN_FOLLOWERS and nxt not in {w.lower() for w in val.split()}:
            return f"value_prefix_before_admin_type:{nxt}"
    return None


def place_granularity_issue(t: Dict[str, Any]) -> str | None:
    value = clean_space(str(t.get("value", ""))).strip(".,")
    words = [w.strip(".,;:()[]{}\"'“”").lower() for w in value.split()]
    # Allow proper city names containing City as part of a title, e.g. Rapid City.
    broad = [w for w in words if w in BROAD_PLACE_TERMS and not (w == "city" and len(words) == 2)]
    if broad:
        return "place_value_granularity_mismatch:" + ",".join(sorted(set(broad)))
    return None


def relation_cue_ok(t: Dict[str, Any]) -> Tuple[bool, str]:
    relation = str(t.get("relation", ""))
    sent = clean_space(str(t.get("sentence", "")))
    value = str(t.get("value", ""))
    pos = find_value_pos(t)
    if pos < 0:
        return False, "value_not_literal"
    before = sent[max(0, pos - 180):pos].lower()
    nearby = sent[max(0, pos - 120): pos + len(value) + 80].lower()
    if relation == "birth_year":
        return (bool(re.search(r"\bborn\b", before)) and re.fullmatch(r"\d{4}", value) is not None, "birth_year_born_cue" if "born" in before else "birth_year_without_born_cue")
    if relation == "founded_year":
        ok = bool(re.search(r"\b(founded|established|created|formed|started|opened)\b", before)) and re.fullmatch(r"\d{4}", value) is not None
        return ok, "founded_year_cue" if ok else "founded_year_without_found_cue"
    if relation == "birthplace":
        ok = bool(re.search(r"\bborn\b.{0,140}\bin\b", before))
        return ok, "birthplace_born_in_cue" if ok else "birthplace_without_born_in_cue"
    if relation == "death_place":
        ok = bool(re.search(r"\bdied\b.{0,160}\b(in|at)\b", before))
        # Very short attachment window does not fix all cases; known ambiguous cases are separate.
        return ok, "death_place_died_place_cue" if ok else "death_place_without_died_place_cue"
    if relation == "nationality":
        ok = bool(re.search(r"\b(is|was)\s+(a|an)\s+" + re.escape(value.lower()) + r"\s+[a-z-]+", nearby))
        return ok, "nationality_descriptor_cue" if ok else "nationality_without_descriptor_cue"
    if relation == "occupation":
        ok = bool(re.search(r"\b(is|was)\s+(a|an)\s+(?:[a-z]+\s+)?" + re.escape(value.lower()) + r"\b", nearby))
        return ok, "occupation_descriptor_cue" if ok else "occupation_without_descriptor_cue"
    if relation == "located_in":
        return False, "located_in_excluded_value_type_mismatch"
    return False, "unknown_relation"


def classify_triple(t: Dict[str, Any]) -> Tuple[str, List[str]]:
    """Return primary_status and reasons.

    primary_status is one of: accept_primary, accept_nonprimary, quarantine.
    """
    relation = str(t.get("relation", ""))
    reasons: List[str] = []
    cue_ok, cue_reason = relation_cue_ok(t)
    if not cue_ok:
        reasons.append(cue_reason)
    boundary = value_boundary_issue(t)
    if boundary:
        reasons.append(boundary)
    key = lower_key(t)
    if key in KNOWN_ATTACHMENT_AMBIGUOUS:
        reasons.append("known_attachment_ambiguous_death_place")
    if relation in {"birthplace", "death_place"}:
        gran = place_granularity_issue(t)
        if gran:
            reasons.append(gran)
    entity_count = exact_entity_occurrences(str(t.get("sentence", "")), str(t.get("entity", "")))
    if entity_count == 0:
        reasons.append("entity_not_literal_whole")
    # Keep nonunique entities as a flag, not a primary rejection; single surnames are often
    # genuine source subjects and over-filtering would silently change the problem.
    if entity_count > 1:
        reasons.append(f"entity_exact_occurs_{entity_count}_times_flag")
    if relation in {"located_in"}:
        reasons.append("relation_excluded_from_primary_located_in")
    if relation in {"nationality", "occupation"}:
        reasons.append("relation_nonexclusive_descriptor_excluded_from_primary")
    fatal = [r for r in reasons if not r.startswith("entity_exact_occurs_")]
    if relation in {"nationality", "occupation"}:
        if cue_ok and boundary is None:
            return "accept_nonprimary", reasons
        return "quarantine", reasons
    if relation == "located_in":
        return "quarantine", reasons
    if fatal:
        return "quarantine", reasons
    return "accept_primary", reasons


def audit_highconf_triples(s64, s65high) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    triples, highconf_summary = s65high.high_confidence_triples(s64)
    primary: List[Dict[str, Any]] = []
    nonprimary: List[Dict[str, Any]] = []
    quarantined: List[Dict[str, Any]] = []
    for t in triples:
        status, reasons = classify_triple(t)
        tt = dict(t)
        tt["tight_semantic_status"] = status
        tt["tight_semantic_reasons"] = reasons
        if status == "accept_primary":
            primary.append(tt)
        elif status == "accept_nonprimary":
            nonprimary.append(tt)
        else:
            quarantined.append(tt)
    summary = {
        "highconfidence_summary": highconf_summary,
        "audited_triples": len(triples),
        "primary_accepted_triples": len(primary),
        "nonprimary_accepted_triples": len(nonprimary),
        "quarantined_triples": len(quarantined),
        "primary_by_relation": dict(Counter(t["relation"] for t in primary)),
        "nonprimary_by_relation": dict(Counter(t["relation"] for t in nonprimary)),
        "quarantined_by_relation": dict(Counter(t["relation"] for t in quarantined)),
        "quarantine_reasons": dict(Counter(r for t in quarantined for r in t["tight_semantic_reasons"] if not r.startswith("entity_exact_occurs_"))),
        "entity_occurrence_flags_in_primary": dict(Counter(r for t in primary for r in t["tight_semantic_reasons"] if r.startswith("entity_exact_occurs_"))),
        "quarantined_examples": quarantined[:30],
        "nonprimary_examples": nonprimary[:20],
    }
    return primary, nonprimary, quarantined, summary


def filter_rows(rows: List[Dict[str, Any]], split: str | None = None, frame_split: str | None = None, original_only: bool = False) -> List[Dict[str, Any]]:
    out = rows
    if split is not None:
        out = [r for r in out if r.get("split") == split]
    if original_only:
        out = [r for r in out if r.get("frame_id") == "f00_original_a01"]
    if frame_split is not None:
        out = [r for r in out if r.get("frame_split") == frame_split]
    return out


def pair_records_for(rows: List[Dict[str, Any]], pair_meta: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [pair_meta[pid] for pid in sorted({r["pair_id"] for r in rows})]


def build_rows(s64, maps: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    all_rows: List[Dict[str, Any]] = []
    pair_meta: Dict[str, Dict[str, Any]] = {}
    for m in maps:
        for spec in s64.assignment_specs(m):
            for frame in s64.FRAMES:
                rows, bp = s64.make_two_rows(m, spec, frame)
                for r in rows:
                    r["packet_type"] = "A01_TIGHT_SEMANTIC_ASSIGNMENT_REVERSAL_USE"
                    r["map_policy"] = "tight_semantic_primary_source_entity_disjoint"
                    r["training_contract"] = "tight semantic assignment-reversal A/B answer-slot: primary relation-value triples passed stricter boundary, cue and value-type filters; answer spans validated."
                bp["map_policy"] = "tight_semantic_primary_source_entity_disjoint"
                all_rows.extend(rows)
                pair_meta[bp["pair_id"]] = bp
    return all_rows, pair_meta


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=pathlib.Path, default=DEFAULT_OUT)
    ap.add_argument("--seed", type=int, default=65066)
    ap.add_argument("--min-large-relation-triples", type=int, default=20)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    s64 = import_module("assignment_export", SCRIPT)
    s65src = import_module("source_assignment_export", SOURCE_SCRIPT)
    s65strict = import_module("strict_assignment_export", STRICT_SCRIPT)
    s65high = import_module("highconf_assignment_export", HIGHCONF_SCRIPT)

    old_caps = dict(s64.RELATION_CAPS)
    s64.RELATION_CAPS.clear(); s64.RELATION_CAPS.update(TIGHT_CAPS)
    try:
        primary_triples, nonprimary_triples, quarantined_triples, audit_summary = audit_highconf_triples(s64, s65high)
        raw_maps, split_notes = s65src.make_maps(s64, primary_triples, args.seed, args.min_large_relation_triples)
        maps, removed_maps, filter_info = s65strict.strict_filter_train_against_held(raw_maps)
    finally:
        s64.RELATION_CAPS.clear(); s64.RELATION_CAPS.update(old_caps)

    maps.sort(key=lambda x: (x["split"], x["relation"], x["base_pair_id"]))
    all_rows, pair_meta = build_rows(s64, maps)
    files: Dict[str, Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]] = {}
    for split, label in [("train", "train"), ("heldout", "heldout")]:
        rows_all = filter_rows(all_rows, split=split)
        rows_seen = filter_rows(all_rows, split=split, frame_split="train_seen")
        rows_unseen = filter_rows(all_rows, split=split, frame_split="eval_unseen")
        rows_original = filter_rows(all_rows, split=split, original_only=True)
        files[f"{label}_frame_all"] = (rows_all, pair_records_for(rows_all, pair_meta))
        files[f"{label}_frame_seen"] = (rows_seen, pair_records_for(rows_seen, pair_meta))
        files[f"{label}_frame_unseen"] = (rows_unseen, pair_records_for(rows_unseen, pair_meta))
        files[f"{label}_original"] = (rows_original, pair_records_for(rows_original, pair_meta))

    outputs, checks = {}, {}
    for name, (rows, pairs) in files.items():
        row_path = args.out_dir / f"a01_tight_semantic_assignment_reversal_{name}.jsonl"
        pair_path = args.out_dir / f"binding_pairs_{name}.jsonl"
        write_jsonl(row_path, rows)
        write_jsonl(pair_path, pairs)
        chk = s64.static_check(rows, pairs)
        checks[name] = chk
        outputs[name] = {"rows_path": rel(row_path), "binding_pairs_path": rel(pair_path), "rows": len(rows), "pairs": len(pairs)}

    op_path = args.out_dir / "a01_tight_semantic_assignment_reversal_operation_maps.jsonl"
    primary_path = args.out_dir / "primary_accepted_triples.jsonl"
    nonprimary_path = args.out_dir / "nonprimary_descriptor_triples.jsonl"
    quarantine_path = args.out_dir / "quarantined_tight_semantic_triples.jsonl"
    removed_path = args.out_dir / "removed_train_overlap_maps.jsonl"
    write_jsonl(op_path, maps)
    write_jsonl(primary_path, primary_triples)
    write_jsonl(nonprimary_path, nonprimary_triples)
    write_jsonl(quarantine_path, quarantined_triples)
    write_jsonl(removed_path, removed_maps)

    sample: List[Dict[str, Any]] = []
    for relation in sorted(set(m["relation"] for m in maps)):
        sample.extend([m for m in maps if m["relation"] == relation][:2])
    sample_path = args.out_dir / "operation_map_sample.json"
    sample_path.write_text(json.dumps(sample, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    fatal = sum(ch["duplicate_row_ids"] + ch["duplicate_pair_ids"] + ch["n_span_errors"] + ch["n_bad_pair_groups"] for ch in checks.values())
    overlap = s65strict.split_overlap(maps)
    status = "TIGHT_SEMANTIC_ASSIGNMENT_REVERSAL_EXPORT_READY" if fatal == 0 and overlap["source_ids"]["n_overlap"] == 0 and overlap["entities"]["n_overlap"] == 0 else "TIGHT_SEMANTIC_ASSIGNMENT_REVERSAL_EXPORT_HAS_WARNINGS"
    summary = {
        "status": status,
        "created_utc": now(),
        "scientific_purpose": "Primary conservative assignment-reversal export after independent_review semantic review: exclude malformed spans, incompatible place granularities, located_in, and nonexclusive descriptor relations from clean mechanism substrate.",
        "relation_caps": TIGHT_CAPS,
        "input_previous_repair": rel(PREV_STEP066_DIR),
        "scripts": {
            "generator": rel(SCRIPT),
            "source_splitter": rel(SOURCE_SCRIPT),
            "strict_filter": rel(STRICT_SCRIPT),
            "highconfidence_filter": rel(HIGHCONF_SCRIPT),
            "tight_export": rel(_public_path('experiments/archive/functional_learning/scripts/tight_semantic_assignment_reversal_export.py')),
        },
        "triple_audit": {
            **audit_summary,
            "primary_accepted_triples_path": rel(primary_path),
            "nonprimary_descriptor_triples_path": rel(nonprimary_path),
            "quarantined_tight_semantic_triples_path": rel(quarantine_path),
        },
        "split_notes_by_relation": split_notes,
        "raw_maps_before_strict_filter": len(raw_maps),
        "strict_filter_info": {**filter_info, "removed_maps_path": rel(removed_path)},
        "n_base_maps": len(maps),
        "base_split_counts": dict(Counter(m["split"] for m in maps)),
        "base_relation_counts": dict(Counter(m["relation"] for m in maps)),
        "base_split_relation_counts": {sp: dict(Counter(m["relation"] for m in maps if m["split"] == sp)) for sp in ["train", "heldout"]},
        "outputs": outputs,
        "operation_maps": rel(op_path),
        "sample": rel(sample_path),
        "static_check_fatal_counts": {name: chk["duplicate_row_ids"] + chk["duplicate_pair_ids"] + chk["n_span_errors"] + chk["n_bad_pair_groups"] for name, chk in checks.items()},
        "static_checks": checks,
        "split_overlap": overlap,
        "reuse_profile": s65strict.map_reuse_profile(maps),
        "contract": {
            "primary_scope": "Cleanest automatic substrate: birth_year, founded_year, and direct place-valued birthplace/death_place; no located_in, nationality, or occupation in primary rows.",
            "still_not_manual_certification": "The script encodes a conservative semantic audit and independent_review-identified examples, but it does not replace full human semantic review of every source sentence.",
            "interpretation": "Supports controlled recipient-conditioned update routing for primary rows. Stronger entity-relation binding still requires wrong-recipient, no-update, relation-contrast, source/entity-disjoint and broad-preservation tests.",
        },
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    if status.endswith("HAS_WARNINGS") or fatal:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
