#!/usr/bin/env python3
"""research: containment and future probe design for the ultra-clean transition asset.

This script is CPU/file-only. It checks whether research transition/control
sentences are exact substrings of the current FW compact, row-block breadth, and
interleaved breadth 10M corpora. The key scientific question is whether a later
small continuation probe can use only text already present in a legal 10M pool,
so it changes relative exposure rather than adding new unique language material.
"""
from __future__ import annotations

import csv
import hashlib
import json
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

WS = Path("experiments/archive/representation_and_objectives")
OUT = WS / "data/transition_containment"
NOTE = (WS.parents[2] / 'research/notes/representation_and_objectives/transition_containment_and_probe_design.md')

ARMS = {
    "compact": WS / "data/fw_full_arms/fw_preserved_compact_view_10M.jsonl",
    "rowblock_breadth": WS / "data/fw_source_breadth_wholesentence_arm/fw_preserved_source_breadth_wholesentence_10M.jsonl",
    "interleaved_breadth": WS / "data/fw_source_breadth_interleaved_wholesentence_arm/fw_preserved_source_breadth_interleaved_wholesentence_10M.jsonl",
}
TREAT = WS / "data/ultraclean_transition_probe_v3/ultraclean_transition_30k.jsonl"
CTRL = WS / "data/ultraclean_transition_probe_v3/ultraclean_anchor_control_30k.jsonl"
SUMMARY_IN = WS / "data/ultraclean_transition_probe_v3/ultraclean_transition_probe_assets_v3.json"

WORD_RE = re.compile(r"\S+")
NORM_RE = re.compile(r"[^a-z0-9]+")


def now_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def norm_words(text: str) -> list[str]:
    out: list[str] = []
    for w in WORD_RE.findall(text.lower()):
        z = NORM_RE.sub("", w)
        if z:
            out.append(z)
    return out


def key_for_text(text: str, n: int = 8) -> tuple[str, ...]:
    ws = norm_words(text)
    if not ws:
        return tuple()
    m = min(n, len(ws))
    return tuple(ws[:m])


def iter_jsonl(path: Path):
    with path.open(encoding="utf-8") as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            obj = json.loads(line)
            obj["_line_index"] = i
            yield obj


def load_targets() -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for arm, path in [("transition", TREAT), ("anchor_control", CTRL)]:
        for obj in iter_jsonl(path):
            text = str(obj["text"]).strip()
            words_field = int(obj.get("words", len(text.split())))
            actual_words = len(text.split())
            rec = {
                "probe_arm": arm,
                "line_index": int(obj["_line_index"]),
                "text": text,
                # Use actual whitespace words for any future training stream because
                # the BabyLM trainers verify len(text.split()) against the field.
                "words": actual_words,
                "words_field": words_field,
                "word_field_matches_actual": words_field == actual_words,
                "source_label": obj.get("source_label"),
                "source_kind": obj.get("source_kind"),
                "origin_source": obj.get("origin_source"),
                "selected_route_bucket": obj.get("selected_route_bucket"),
                "required_capability": obj.get("required_capability"),
                "length_bin": obj.get("length_bin"),
                "relation_marker_count": obj.get("relation_marker_count", obj.get("marker", {}).get("causal", None) if isinstance(obj.get("marker"), dict) else None),
                "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                "contains": {},
            }
            records.append(rec)
    return records


def build_target_index(records: list[dict[str, Any]]) -> dict[tuple[str, ...], list[int]]:
    idx: dict[tuple[str, ...], list[int]] = defaultdict(list)
    for i, r in enumerate(records):
        key = key_for_text(r["text"])
        if not key:
            raise RuntimeError(f"empty key for target {i}")
        idx[key].append(i)
    return idx


def scan_arm(arm_name: str, path: Path, records: list[dict[str, Any]], target_index: dict[tuple[str, ...], list[int]]) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    found: dict[int, dict[str, Any]] = {}
    rows = 0
    words_total = 0
    candidate_hits = 0
    key_lengths = sorted({len(k) for k in target_index if k})
    # Most targets have at least 8 normalized words. We still support shorter texts.
    for obj in iter_jsonl(path):
        rows += 1
        text = str(obj["text"])
        row_words_field = int(obj.get("words", len(text.split())))
        words_total += row_words_field
        nws = norm_words(text)
        if not nws:
            continue
        keys_here = set()
        for n in key_lengths:
            if len(nws) < n:
                continue
            for j in range(0, len(nws) - n + 1):
                k = tuple(nws[j:j+n])
                if k in target_index:
                    keys_here.add(k)
        if not keys_here:
            continue
        for k in keys_here:
            for ridx in target_index[k]:
                if ridx in found:
                    continue
                target_text = records[ridx]["text"]
                if target_text in text:
                    found[ridx] = {
                        "row_index": int(obj.get("example_id", obj.get("_line_index", -1))),
                        "line_index": int(obj.get("_line_index", -1)),
                        "row_source": obj.get("source"),
                        "row_words": row_words_field,
                    }
                    candidate_hits += 1
    for i, r in enumerate(records):
        r["contains"][arm_name] = found.get(i)
    return {
        "arm": arm_name,
        "path": str(path),
        "sha256": sha256_file(path),
        "rows": rows,
        "words": words_total,
        "targets_found": len(found),
        "candidate_hits": candidate_hits,
    }


def aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for probe_arm in ["transition", "anchor_control"]:
        recs = [r for r in records if r["probe_arm"] == probe_arm]
        mismatches = [r for r in recs if not r.get("word_field_matches_actual", True)]
        base = {
            "sentences": len(recs),
            "words": sum(r["words"] for r in recs),
            "stored_word_field_sum": sum(int(r.get("words_field", r["words"])) for r in recs),
            "word_field_mismatch_sentences": len(mismatches),
            "word_field_mismatch_delta_actual_minus_field": sum(r["words"] - int(r.get("words_field", r["words"])) for r in mismatches),
            "by_source_label_words": dict(Counter({})),
        }
        for arm_name in ARMS:
            kept = [r for r in recs if r["contains"].get(arm_name)]
            missing = [r for r in recs if not r["contains"].get(arm_name)]
            base[f"{arm_name}_sentences"] = len(kept)
            base[f"{arm_name}_words"] = sum(r["words"] for r in kept)
            base[f"{arm_name}_missing_sentences"] = len(missing)
            base[f"{arm_name}_missing_words"] = sum(r["words"] for r in missing)
        c_by_source = Counter()
        c_by_bucket = Counter()
        c_by_len = Counter()
        for r in recs:
            c_by_source[str(r.get("source_label"))] += r["words"]
            c_by_bucket[str(r.get("selected_route_bucket"))] += r["words"]
            c_by_len[str(r.get("length_bin"))] += r["words"]
        base["by_source_label_words"] = dict(sorted(c_by_source.items()))
        base["by_bucket_words"] = dict(sorted(c_by_bucket.items()))
        base["by_length_bin_words"] = dict(sorted(c_by_len.items()))
        summary[probe_arm] = base
    # A conservative common-contained pair size if both treatment and control must be exact in compact.
    compact_transition_words = summary["transition"]["compact_words"]
    compact_control_words = summary["anchor_control"]["compact_words"]
    summary["compact_contained_pair_word_cap"] = min(compact_transition_words, compact_control_words)
    # Same for all three arm corpora.
    all3_t = sum(r["words"] for r in records if r["probe_arm"] == "transition" and all(r["contains"].get(a) for a in ARMS))
    all3_c = sum(r["words"] for r in records if r["probe_arm"] == "anchor_control" and all(r["contains"].get(a) for a in ARMS))
    summary["all_three_contained_pair_word_cap"] = min(all3_t, all3_c)
    return summary


def write_filtered(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Write compact-contained candidate subsets for later exact stream construction.

    These files are not a training stream. They are candidate inserts for a later
    matched continuation probe.
    """
    files = {}
    for probe_arm in ["transition", "anchor_control"]:
        rows = [r for r in records if r["probe_arm"] == probe_arm and r["contains"].get("compact")]
        out_path = OUT / f"{probe_arm}_exact_in_compact.jsonl"
        with out_path.open("w", encoding="utf-8") as f:
            for r in rows:
                obj = {k: v for k, v in r.items() if k != "contains"}
                obj["compact_location"] = r["contains"].get("compact")
                obj["rowblock_breadth_location"] = r["contains"].get("rowblock_breadth")
                obj["interleaved_breadth_location"] = r["contains"].get("interleaved_breadth")
                f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        files[f"{probe_arm}_exact_in_compact"] = str(out_path)
    return files


def write_records_csv(records: list[dict[str, Any]], path: Path) -> None:
    fieldnames = [
        "probe_arm", "line_index", "words", "words_field", "word_field_matches_actual", "source_label", "source_kind", "origin_source",
        "selected_route_bucket", "required_capability", "length_bin", "text_sha256",
        "in_compact", "in_rowblock_breadth", "in_interleaved_breadth", "compact_row_source",
        "rowblock_row_source", "interleaved_row_source",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in records:
            w.writerow({
                "probe_arm": r["probe_arm"],
                "line_index": r["line_index"],
                "words": r["words"],
                "words_field": r.get("words_field"),
                "word_field_matches_actual": r.get("word_field_matches_actual"),
                "source_label": r.get("source_label"),
                "source_kind": r.get("source_kind"),
                "origin_source": r.get("origin_source"),
                "selected_route_bucket": r.get("selected_route_bucket"),
                "required_capability": r.get("required_capability"),
                "length_bin": r.get("length_bin"),
                "text_sha256": r.get("text_sha256"),
                "in_compact": bool(r["contains"].get("compact")),
                "in_rowblock_breadth": bool(r["contains"].get("rowblock_breadth")),
                "in_interleaved_breadth": bool(r["contains"].get("interleaved_breadth")),
                "compact_row_source": (r["contains"].get("compact") or {}).get("row_source"),
                "rowblock_row_source": (r["contains"].get("rowblock_breadth") or {}).get("row_source"),
                "interleaved_row_source": (r["contains"].get("interleaved_breadth") or {}).get("row_source"),
            })


def note_text(result: dict[str, Any]) -> str:
    s = result["containment_summary"]
    probe = result["future_probe_reading"]
    lines = []
    lines.append("# research — transition asset containment and future probe design")
    lines.append("")
    lines.append("## What was measured")
    lines.append("")
    lines.append("The research ultra-clean transition/control asset was checked against three legal 10M FW corpora: A02 compact, row-block breadth, and A01 interleaved breadth. Exact containment means the sentence text appears as a substring of a row in that 10M corpus. This matters because a later warm-start probe can remain inside a legal 10M word set only if it reweights text already present in the checkpoint's training pool.")
    lines.append("")
    lines.append("## Main counts")
    lines.append("")
    lines.append("Stored word fields in the research JSONL are not all identical to Python whitespace counts, so this research output uses actual `len(text.split())` for any future training-compatible accounting and records the mismatch explicitly.")
    lines.append("")
    lines.append("| asset | actual words | stored field sum | field mismatches | in compact | in row-block breadth | in interleaved breadth |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for arm in ["transition", "anchor_control"]:
        x = s[arm]
        lines.append(f"| {arm} | {x['words']} | {x['stored_word_field_sum']} | {x['word_field_mismatch_sentences']} ({x['word_field_mismatch_delta_actual_minus_field']:+d}) | {x['compact_words']} | {x['rowblock_breadth_words']} | {x['interleaved_breadth_words']} |")
    lines.append("")
    lines.append(f"Compact-contained matched word cap: **{s['compact_contained_pair_word_cap']}** words.")
    lines.append(f"Contained in all three FW corpora matched word cap: **{s['all_three_contained_pair_word_cap']}** words.")
    lines.append("")
    lines.append("## Scientific reading")
    lines.append("")
    lines.append(probe["reading"])
    lines.append("")
    lines.append("## Future minimum-cost probe, if FW allocation fails")
    lines.append("")
    for item in probe["minimum_cost_probe"]:
        lines.append(f"- {item}")
    lines.append("")
    lines.append("## Files")
    lines.append("")
    for k, v in result["files"].items():
        lines.append(f"- {k}: `{v}`")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    records = load_targets()
    target_index = build_target_index(records)
    arm_scans = []
    for arm_name, path in ARMS.items():
        arm_scans.append(scan_arm(arm_name, path, records, target_index))
    containment_summary = aggregate(records)
    record_csv = OUT / "transition_control_sentence_containment.csv"
    write_records_csv(records, record_csv)
    filtered_files = write_filtered(records)

    summary_in = json.loads(SUMMARY_IN.read_text(encoding="utf-8"))
    future_probe_reading = {
        "reading": (
            "Most of the useful transition/control material is inherited from the current compact-compatible pool rather than being novel external text. "
            "A future probe should therefore use a legal warm-start contrast only by changing relative exposure inside the original 10M pool. "
            "It should not continue a completed 100M endpoint, and it should not add non-contained sentences to an existing checkpoint."
        ),
        "minimum_cost_probe": [
            "Use a compact-family intermediate checkpoint, preferably chck_70M or chck_80M, not chck_100M, so the total exposure can still end at exactly 100M.",
            "Construct two remaining-exposure streams with identical total words and identical common removals: one repeats the compact-contained transition subset, the other repeats the compact-contained anchor-control subset.",
            "Use only sentences exact-contained in the checkpoint's 10M pool; otherwise the warm-start branch would add new unique text beyond the pool used before the checkpoint.",
            "Interpret the result as a relative-exposure probe, not as a final data recipe: it tests whether explicit physical/spatial/quantity transitions can move EWoK four-cell and GlobalPIQA hard-row ranks without the broad damage seen in row-block breadth.",
            "Read cheap7, GlobalPIQA all-option margins, and EWoK four-cell before any full official endpoint work. A useful signal is compact-level broad columns plus reduced hard-row margins; an EWoK-only rise with GlobalPIQA or Reading loss repeats the earlier relation-data failure pattern.",
        ],
    }
    result = {
        "status": "TRANSITION_CONTAINMENT_DONE",
        "created_utc": now_utc(),
        "inputs": {
            "transition": str(TREAT),
            "anchor_control": str(CTRL),
            "summary": str(SUMMARY_IN),
            "arms": {k: str(v) for k, v in ARMS.items()},
        },
        "original_counts": {
            "transition_words_stored": summary_in.get("treatment_summary", {}).get("words"),
            "anchor_control_words_stored": summary_in.get("control_summary", {}).get("words"),
            "match_metrics_stored": summary_in.get("match_metrics"),
        },
        "arm_scans": arm_scans,
        "containment_summary": containment_summary,
        "future_probe_reading": future_probe_reading,
        "files": {
            "json": str(OUT / "transition_containment_and_probe_design.json"),
            "sentence_csv": str(record_csv),
            "transition_exact_in_compact": filtered_files["transition_exact_in_compact"],
            "anchor_control_exact_in_compact": filtered_files["anchor_control_exact_in_compact"],
            "note": str(NOTE),
        },
    }
    json_path = OUT / "transition_containment_and_probe_design.json"
    json_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    NOTE.write_text(note_text(result), encoding="utf-8")
    print(json.dumps({
        "status": result["status"],
        "compact_transition_words": containment_summary["transition"]["compact_words"],
        "compact_control_words": containment_summary["anchor_control"]["compact_words"],
        "compact_contained_pair_word_cap": containment_summary["compact_contained_pair_word_cap"],
        "all_three_contained_pair_word_cap": containment_summary["all_three_contained_pair_word_cap"],
        "json": str(json_path),
        "note": str(NOTE),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
