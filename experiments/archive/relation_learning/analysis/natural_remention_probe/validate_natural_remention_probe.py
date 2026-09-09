#!/usr/bin/env python3
"""Structural and provenance validation for the natural re-mention probe."""
from __future__ import annotations

import collections
import hashlib
import json
import pathlib
import re

ROOT = pathlib.Path("/workspace")
BASE = ROOT / "experiments/archive/relation_learning/analysis/natural_remention_probe"
PROBE = BASE / "natural_remention_probe.jsonl"
SUMMARY = BASE / "natural_remention_probe_summary.json"
SAMPLE = BASE / "natural_remention_probe_sample.md"
SOURCE = ROOT / "experiments/archive/representation_and_objectives/training/data/cached_fineweb_seqsafe96_candidate/cleanqwen_seqsafe_fineweb_single_doc_10M.jsonl"
WORD_RE = re.compile(r"[A-Za-z0-9]+(?:['’][A-Za-z0-9]+)?")


def sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    records = [json.loads(x) for x in PROBE.read_text(encoding="utf-8").splitlines() if x.strip()]
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    errors: list[str] = []
    checks = collections.Counter()

    wanted_lines = {r["source_line_index"] for r in records}
    source_rows = {}
    with SOURCE.open(encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i in wanted_lines:
                source_rows[i] = json.loads(line)
    checks["source_rows_loaded"] = len(source_rows)

    ids = set()
    rows = set()
    match_pairs = collections.defaultdict(list)
    for r in records:
        pid = r["probe_id"]
        if pid in ids: errors.append(f"duplicate probe_id: {pid}")
        ids.add(pid)
        line_i = r["source_line_index"]
        if line_i in rows: errors.append(f"duplicate source line: {line_i}")
        rows.add(line_i)
        match_pairs[r["class_matching"]["pair_id"]].append(r)
        src = source_rows.get(line_i)
        if src is None or src.get("example_id") != r["row_id"] or src.get("text") != r["full_text_original"]:
            errors.append(f"source provenance mismatch: {pid}")
        text = r["full_text_original"]
        ant, rem = r["antecedent"], r["remention"]
        if text[ant["char_start"]:ant["char_end"]] != ant["text"]:
            errors.append(f"antecedent char offset mismatch: {pid}")
        if text[rem["char_start"]:rem["char_end"]] != rem["text"]:
            errors.append(f"remention char offset mismatch: {pid}")
        if not ant["char_end"] <= rem["char_start"]:
            errors.append(f"overlap/order error: {pid}")
        mask = r["mask_plan"]
        if (mask["char_start"], mask["char_end"], mask["target_text"]) != (rem["char_start"], rem["char_end"], rem["text"]):
            errors.append(f"mask plan mismatch: {pid}")
        ctx = r["local_context"]
        if text[ctx["char_start"]:ctx["char_end"]] != ctx["text"]:
            errors.append(f"context offset mismatch: {pid}")
        cp = r["corruption_plan"]
        rebuilt = text[:ant["char_start"]] + cp["replacement_text"] + text[ant["char_end"]:]
        if rebuilt != cp["full_text_antecedent_replaced"]:
            errors.append(f"corruption reconstruction mismatch: {pid}")
        rs, re_ = cp["remention_char_start_replaced"], cp["remention_char_end_replaced"]
        if rebuilt[rs:re_] != rem["text"]:
            errors.append(f"replaced-condition remention offset mismatch: {pid}")
        wc = len(WORD_RE.findall(cp["replacement_text"]))
        if wc != ant["word_count"] or not cp["exact_word_count_match"]:
            errors.append(f"replacement word length mismatch: {pid}")
        if r["training_overlap_audit"]["arms_with_overlap"]:
            errors.append(f"training-overlap-marked record retained: {pid}")
        if r["class_label"] == "nonidentical_remention" and re.sub(r"\W", "", ant["text"].lower()) == re.sub(r"\W", "", rem["text"].lower()):
            errors.append(f"nonidentical surface forms normalize equal: {pid}")
        checks["records_checked"] += 1
        checks["replaced_condition_offsets_checked"] += 1

    dist = summary["final_distribution"]
    class_counts = collections.Counter(r["class_label"] for r in records)
    if class_counts["verbatim_remention"] != class_counts["nonidentical_remention"]:
        errors.append("class denominators differ")
    for b in set(dist["verbatim_remention"]["distance_bin"]) | set(dist["nonidentical_remention"]["distance_bin"]):
        if dist["verbatim_remention"]["distance_bin"].get(b, 0) != dist["nonidentical_remention"]["distance_bin"].get(b, 0):
            errors.append(f"distance-bin mismatch: {b}")
    for pair_id, pair in match_pairs.items():
        if len(pair) != 2 or {r["class_label"] for r in pair} != {"verbatim_remention", "nonidentical_remention"}:
            errors.append(f"invalid class-matching pair: {pair_id}")
    checks["class_matching_pairs_checked"] = len(match_pairs)
    if summary["output"]["probe_sha256"] != sha256(PROBE):
        errors.append("probe SHA-256 does not match summary")
    sample_ids = re.findall(r"^## \d+\. (\S+)", SAMPLE.read_text(encoding="utf-8"), re.M)
    if len(sample_ids) != 100 or len(set(sample_ids)) != 100 or not set(sample_ids) <= ids:
        errors.append("sample membership/count invalid")
    checks["sample_records_checked"] = len(sample_ids)

    result = {
        "status": "PASS" if not errors else "FAIL",
        "probe_sha256": sha256(PROBE),
        "records": len(records),
        "unique_probe_ids": len(ids),
        "unique_source_rows": len(rows),
        "class_counts": dict(sorted(class_counts.items())),
        "checks": dict(checks),
        "errors": errors,
    }
    (BASE / "validation_results.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    raise SystemExit(bool(errors))


if __name__ == "__main__":
    main()
