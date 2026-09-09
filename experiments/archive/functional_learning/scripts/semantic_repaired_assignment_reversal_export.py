#!/usr/bin/env python3
"""research: semantically repair the research high-confidence assignment export.

The research high-confidence filter checked literal spans, whole entity mentions, and a
nearby relation cue.  That was still not enough: e.g. a located_in source sentence
"La Banda is a city in the Santiago del Estero Province" was represented as the
source value "Santiago", and "Milton is a city in King and Pierce counties" as
"King".  Those are literal substrings, not complete interchangeable values for a
controlled assignment-reversal operation.

This script keeps the useful source/entity-disjoint A/B assignment schema but adds a
semantic-value audit over the underlying triples before deriving rows.  It produces a
repaired export in a new directory and a quarantine summary for the superseded research
export.  The intended use remains controlled mechanism testing, not direct BabyLM SOTA
evidence.
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
OLD_HIGHCONF_DIR = _public_path('experiments/archive/functional_learning/data/highconfidence_strict_assignment_reversal_export')
DEFAULT_OUT = _public_path('experiments/archive/functional_learning/data/semantic_repaired_assignment_reversal_export')

# Keep the research caps except that located_in is quarantined by the semantic audit.
SEMANTIC_CAPS = {
    "death_place": 190,
    "birthplace": 165,
    "birth_year": 150,
    "located_in": 0,
    "founded_year": 20,
    "nationality": 16,
    "occupation": 16,
}

SAFE_PLACE_RELATIONS = {"birthplace", "death_place"}
SAFE_YEAR_RELATIONS = {"birth_year", "founded_year"}
SAFE_DESCRIPTOR_RELATIONS = {"nationality", "occupation"}

PLACE_ADMIN_WORDS = {
    "province", "county", "counties", "state", "states", "district", "region",
    "regions", "prefecture", "municipality", "department", "departments", "oblast",
    "territory", "territories", "republic", "kingdom", "island", "islands",
}
LOWERCASE_CONNECTORS = {"of", "and", "del", "de", "la", "el", "du", "di", "van", "von"}


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


def load_jsonl(path: pathlib.Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    out: List[Dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def clean_space(s: str) -> str:
    return re.sub(r"\s+", " ", str(s or "").strip())


def triple_key(t: Dict[str, Any]) -> Tuple[str, str, str, str]:
    return (
        str(t.get("relation", "")),
        str(t.get("sentence_id", "")),
        str(t.get("entity", "")).lower(),
        str(t.get("value", "")).lower(),
    )


def sentence_after_value(sentence: str, value: str) -> str:
    i = sentence.find(value)
    if i < 0:
        return ""
    return sentence[i + len(value): i + len(value) + 80]


def sentence_before_value(sentence: str, value: str) -> str:
    i = sentence.find(value)
    if i < 0:
        return ""
    return sentence[max(0, i - 160): i]


def immediate_value_continuation(sentence: str, value: str) -> str | None:
    """Detect obvious evidence that the stored value is only a prefix fragment."""
    after = sentence_after_value(sentence, value)
    if not after:
        return None
    # If the next characters continue with lowercase words or a coordinated noun phrase,
    # the extracted value is very likely a substring of a larger location/name phrase.
    m = re.match(r"^\s+([a-z][A-Za-z'’.-]*)\b", after)
    if m and m.group(1).lower() in LOWERCASE_CONNECTORS:
        return f"value_prefix_before_lowercase_connector:{m.group(1)}"
    if re.match(r"^\s+and\s+", after, re.I):
        return "value_prefix_before_and_coordination"
    # Examples: Santiago del Estero Province, King and Pierce counties.  Also catch
    # value=Los followed by Angeles-like capital continuation, but only as a warning;
    # existing regex normally includes adjacent capital words.
    m2 = re.match(r"^\s+([A-Z][A-Za-z'’.-]*)\b", after)
    if m2:
        next_word = m2.group(1).lower().strip(".,;:()")
        if next_word in PLACE_ADMIN_WORDS:
            return f"value_prefix_before_admin_word:{m2.group(1)}"
    return None


def relation_phrase_support(relation: str, sentence: str, entity: str, value: str) -> Tuple[bool, str]:
    """Heuristic support check for relation renderings over exact source text.

    This intentionally remains conservative and auditable.  It cannot prove truth, but
    it catches cases where a structural regex no longer supports the proposition used by
    the controlled packet.
    """
    sent = clean_space(sentence)
    ent = clean_space(entity)
    val = clean_space(value)
    low = sent.lower()
    ent_low = ent.lower()
    val_low = val.lower()
    if val_low not in low:
        return False, "value_not_literal_in_sentence"
    if ent_low not in low:
        return False, "entity_not_literal_in_sentence"
    before = sentence_before_value(sent, val).lower()
    nearby = low[max(0, low.find(val_low) - 180): low.find(val_low) + len(val_low) + 80]
    if relation == "birth_year":
        if re.search(r"\bborn\b", before) and re.fullmatch(r"\d{4}", val):
            return True, "born_cue_before_year"
        return False, "birth_year_without_born_cue"
    if relation == "birthplace":
        if re.search(r"\bborn\b.{0,120}\bin\b", before):
            cont = immediate_value_continuation(sent, val)
            if cont:
                return False, cont
            return True, "born_in_place_cue"
        return False, "birthplace_without_born_in_cue"
    if relation == "death_place":
        if re.search(r"\bdied\b.{0,140}\bin\b", before):
            cont = immediate_value_continuation(sent, val)
            if cont:
                return False, cont
            return True, "died_in_place_cue"
        return False, "death_place_without_died_in_cue"
    if relation == "founded_year":
        if re.search(r"\b(founded|established|created|formed|started|opened)\b", before) and re.fullmatch(r"\d{4}", val):
            return True, "founding_cue_before_year"
        return False, "founded_year_without_founding_cue"
    if relation == "nationality":
        if re.search(r"\b(is|was)\s+(a|an)\s+" + re.escape(val_low) + r"\s+[a-z-]+", nearby):
            return True, "nationality_occupation_phrase"
        return False, "nationality_without_descriptor_phrase"
    if relation == "occupation":
        if re.search(r"\b(is|was)\s+(a|an)\s+(?:[a-z]+\s+)?" + re.escape(val_low) + r"\b", nearby):
            return True, "occupation_descriptor_phrase"
        return False, "occupation_without_descriptor_phrase"
    if relation == "located_in":
        # Quarantine this relation for the current A/B update contract.  The source
        # extraction gives containing jurisdictions (province/county/state/country),
        # while the update/replacement pool uses cities.  Prefix fragments such as
        # Santiago and King show that literal span checks are not enough here.
        return False, "located_in_quarantined_value_type_not_interchangeable"
    return False, "unknown_relation"


def semantic_audit_triples(triples: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    kept: List[Dict[str, Any]] = []
    quarantined: List[Dict[str, Any]] = []
    for t in triples:
        relation = str(t.get("relation", ""))
        ok, reason = relation_phrase_support(relation, str(t.get("sentence", "")), str(t.get("entity", "")), str(t.get("value", "")))
        tt = dict(t)
        tt["semantic_audit"] = reason
        if ok:
            tt["semantic_status"] = "accepted_for_assignment_reversal"
            kept.append(tt)
        else:
            tt["semantic_status"] = "quarantined_before_row_generation"
            tt["quarantine_reason"] = reason
            quarantined.append(tt)
    summary = {
        "audited_triples": len(triples),
        "accepted_triples": len(kept),
        "quarantined_triples": len(quarantined),
        "accepted_by_relation": dict(Counter(t["relation"] for t in kept)),
        "quarantined_by_relation": dict(Counter(t["relation"] for t in quarantined)),
        "quarantined_by_reason": dict(Counter(t["quarantine_reason"] for t in quarantined)),
        "quarantined_examples": quarantined[:20],
    }
    return kept, quarantined, summary


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


def old_export_affected_maps(quarantined: List[Dict[str, Any]]) -> Dict[str, Any]:
    qkeys = {triple_key(t) for t in quarantined}
    op_path = _public_path('experiments/archive/functional_learning/data/highconfidence_strict_assignment_reversal_export/a01_highconf_strict_assignment_reversal_operation_maps.jsonl')
    maps = load_jsonl(op_path)
    affected = []
    for m in maps:
        akey = (str(m.get("relation", "")), str(m.get("source_a_id", "")), str(m.get("entity_a", "")).lower(), str(m.get("value_a", "")).lower())
        bkey = (str(m.get("relation", "")), str(m.get("source_b_id", "")), str(m.get("entity_b", "")).lower(), str(m.get("value_b", "")).lower())
        if akey in qkeys or bkey in qkeys:
            mm = dict(m)
            mm["affected_side_a"] = akey in qkeys
            mm["affected_side_b"] = bkey in qkeys
            affected.append(mm)
    return {
        "old_operation_maps_path": rel(op_path),
        "old_maps_total": len(maps),
        "affected_old_maps": len(affected),
        "affected_old_maps_by_relation": dict(Counter(m["relation"] for m in affected)),
        "affected_old_rows_estimate_frame_all": len(affected) * 16,
        "affected_old_binding_pairs_estimate_frame_all": len(affected) * 8,
        "affected_old_examples": affected[:20],
        "affected_old_maps": affected,
    }


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
    s64.RELATION_CAPS.clear(); s64.RELATION_CAPS.update(SEMANTIC_CAPS)
    try:
        highconf_triples, highconf_summary = s65high.high_confidence_triples(s64)
        accepted_triples, quarantined_triples, semantic_summary = semantic_audit_triples(highconf_triples)
        raw_maps, split_notes = s65src.make_maps(s64, accepted_triples, args.seed, args.min_large_relation_triples)
        maps, removed_maps, filter_info = s65strict.strict_filter_train_against_held(raw_maps)
    finally:
        s64.RELATION_CAPS.clear(); s64.RELATION_CAPS.update(old_caps)

    maps.sort(key=lambda x: (x["split"], x["relation"], x["base_pair_id"]))

    all_rows: List[Dict[str, Any]] = []
    pair_meta: Dict[str, Dict[str, Any]] = {}
    for m in maps:
        for spec in s64.assignment_specs(m):
            for frame in s64.FRAMES:
                rows, bp = s64.make_two_rows(m, spec, frame)
                for r in rows:
                    r["packet_type"] = "A01_SEMANTIC_REPAIRED_ASSIGNMENT_REVERSAL_USE"
                    r["map_policy"] = "semantic_repaired_source_entity_disjoint_literals"
                    r["training_contract"] = "semantic-repaired assignment-reversal A/B answer-slot: same source_context and shared_new value; two rows per assignment; answer_text spans validated; source triples passed research relation-value audit."
                bp["map_policy"] = "semantic_repaired_source_entity_disjoint_literals"
                all_rows.extend(rows)
                pair_meta[bp["pair_id"]] = bp

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

    outputs: Dict[str, Any] = {}
    checks: Dict[str, Any] = {}
    for name, (rows, pairs) in files.items():
        row_path = args.out_dir / f"a01_semantic_repaired_assignment_reversal_{name}.jsonl"
        pair_path = args.out_dir / f"binding_pairs_{name}.jsonl"
        write_jsonl(row_path, rows)
        write_jsonl(pair_path, pairs)
        chk = s64.static_check(rows, pairs)
        checks[name] = chk
        outputs[name] = {"rows_path": rel(row_path), "binding_pairs_path": rel(pair_path), "rows": len(rows), "pairs": len(pairs)}

    op_path = args.out_dir / "a01_semantic_repaired_assignment_reversal_operation_maps.jsonl"
    write_jsonl(op_path, maps)
    accepted_path = args.out_dir / "accepted_semantic_triples.jsonl"
    quarantined_path = args.out_dir / "quarantined_semantic_triples.jsonl"
    removed_path = args.out_dir / "removed_train_overlap_maps.jsonl"
    affected_path = args.out_dir / "superseded_step065_affected_maps.jsonl"
    write_jsonl(op_path, maps)
    write_jsonl(accepted_path, accepted_triples)
    write_jsonl(quarantined_path, quarantined_triples)
    write_jsonl(removed_path, removed_maps)

    affected = old_export_affected_maps(quarantined_triples)
    write_jsonl(affected_path, affected.get("affected_old_maps", []))
    affected_public = {k: v for k, v in affected.items() if k != "affected_old_maps"}
    affected_public["affected_old_maps_path"] = rel(affected_path)

    sample: List[Dict[str, Any]] = []
    for relation in sorted(set(m["relation"] for m in maps)):
        sample.extend([m for m in maps if m["relation"] == relation][:2])
    sample_path = args.out_dir / "operation_map_sample.json"
    sample_path.write_text(json.dumps(sample, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    fatal = sum(ch["duplicate_row_ids"] + ch["duplicate_pair_ids"] + ch["n_span_errors"] + ch["n_bad_pair_groups"] for ch in checks.values())
    overlap = s65strict.split_overlap(maps)
    status = "SEMANTIC_REPAIRED_ASSIGNMENT_REVERSAL_EXPORT_READY" if fatal == 0 and overlap["source_ids"]["n_overlap"] == 0 and overlap["entities"]["n_overlap"] == 0 else "SEMANTIC_REPAIRED_ASSIGNMENT_REVERSAL_EXPORT_HAS_WARNINGS"
    summary = {
        "status": status,
        "created_utc": now(),
        "scientific_purpose": "Semantically repaired literal-value assignment-reversal pool: quarantine structural-but-incomplete relation values, regenerate source/entity-disjoint A/B rows, preserve controlled operation substrate.",
        "supersedes_with_qualification": {
            "old_export_dir": rel(OLD_HIGHCONF_DIR),
            "reason": "research whole-entity/value-span checks admitted at least located_in values that are substrings of larger jurisdictions, e.g. Santiago from Santiago del Estero Province and King from King and Pierce counties.",
        },
        "base_scripts": {
            "generator": rel(SCRIPT),
            "source_splitter": rel(SOURCE_SCRIPT),
            "strict_filter": rel(STRICT_SCRIPT),
            "highconfidence_filter": rel(HIGHCONF_SCRIPT),
            "semantic_repair": rel(_public_path('experiments/archive/functional_learning/scripts/semantic_repaired_assignment_reversal_export.py')),
        },
        "highconfidence_triple_summary_before_semantic_audit": highconf_summary,
        "semantic_triple_audit": {**semantic_summary, "accepted_triples_path": rel(accepted_path), "quarantined_triples_path": rel(quarantined_path)},
        "semantic_relation_caps": SEMANTIC_CAPS,
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
        "superseded_step065_affected_maps": affected_public,
        "contract": {
            "row_schema": "A/B answer-slot: two rows per assignment-specific pair_id; pair_half A is unchanged/source_state, pair_half B is updated/new_state.",
            "semantic_repair": "Located_in triples are excluded for the current packet because their source values denote containing jurisdictions while replacement values are cities, and observed extractions were incomplete substrings. Other retained triples passed relation-specific cue/value checks over exact source text.",
            "strict_split": "No held source id or held entity is used in train maps after filtering; source values may overlap.",
            "scope": "Controlled counterfactual maps from exact BabyLM source sentences; update sentences and replacement values are constructed; not direct broad BabyLM improvement evidence.",
            "interpretation_boundary": "Structural/static success remains weaker than semantic source review. In-format learner success can still reflect recipient/update gating unless wrong-recipient, no-update, source/entity-disjoint and relation-contrast probes are included.",
        },
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    if status.endswith("HAS_WARNINGS") or fatal != 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
