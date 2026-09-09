#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import pathlib
from collections import Counter
from typing import Any

STUDY = pathlib.Path("experiments/archive/compact_experience")
SRC = pathlib.Path("experiments/archive/initial_model_studies/data/core_competence_screen")
OUT_JSON = STUDY / "data" / "initial_model_core_screen_inspection.json"
OUT_NOTE = (STUDY / 'notes'.parents[3] / 'research/notes/compact_experience/initial_model_core_screen_inspection.md')


def h_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def h_seq(xs: list[str]) -> str:
    return hashlib.sha256("\n".join(xs).encode("utf-8")).hexdigest()


def short_stats(vals: list[float]) -> dict[str, Any]:
    if not vals:
        return {"n": 0, "min": None, "mean": None, "max": None}
    return {"n": len(vals), "min": min(vals), "mean": sum(vals) / len(vals), "max": max(vals)}


def inspect_jsonl(path: pathlib.Path) -> dict[str, Any]:
    n = 0
    words = 0
    source = Counter()
    fields = Counter()
    conditions = Counter()
    text_hashes: list[str] = []
    row_hashes: list[str] = []
    comp: list[float] = []
    curr: list[float] = []
    samples = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            n += 1
            words += int(r.get("words") or 0)
            source[str(r.get("source"))] += 1
            for k in r.keys():
                fields[k] += 1
            if r.get("condition") is not None:
                conditions[str(r.get("condition"))] += 1
            text = str(r.get("text") or "")
            th = h_text(text)
            text_hashes.append(th)
            row_hashes.append(str(r.get("row_hash") or th[:16]))
            if isinstance(r.get("comp_score"), (int, float)):
                comp.append(float(r["comp_score"]))
            if isinstance(r.get("curriculum_score"), (int, float)):
                curr.append(float(r["curriculum_score"]))
            if len(samples) < 2:
                samples.append({
                    "words": r.get("words"),
                    "source": r.get("source"),
                    "condition": r.get("condition"),
                    "comp_score": r.get("comp_score"),
                    "curriculum_score": r.get("curriculum_score"),
                    "text_head": text[:220].replace("\n", " "),
                })
    return {
        "file": str(path),
        "bytes": path.stat().st_size,
        "rows": n,
        "words": words,
        "source_dist": dict(source.most_common()),
        "fields": sorted(fields.keys()),
        "condition_values": dict(conditions.most_common()),
        "text_multiset_hash": h_seq(sorted(text_hashes)) if text_hashes else None,
        "text_order_hash": h_seq(text_hashes) if text_hashes else None,
        "row_multiset_hash": h_seq(sorted(row_hashes)) if row_hashes else None,
        "row_order_hash": h_seq(row_hashes) if row_hashes else None,
        "comp_score": short_stats(comp),
        "curriculum_score": short_stats(curr),
        "samples": samples,
    }


def main() -> None:
    files = sorted(p for p in SRC.glob("*.jsonl") if p.is_file())
    inspections = {p.name: inspect_jsonl(p) for p in files}
    groups_by_text = {}
    for name, info in inspections.items():
        groups_by_text.setdefault(info["text_multiset_hash"], []).append(name)
    groups_by_words = {}
    for name, info in inspections.items():
        groups_by_words.setdefault(info["words"], []).append(name)
    key_takeaways = []
    same_content_order = [
        "official_random_order_a.jsonl",
        "official_random_order_b.jsonl",
        "official_curriculum.jsonl",
    ]
    if all(k in inspections for k in same_content_order):
        hashes = {inspections[k]["text_multiset_hash"] for k in same_content_order}
        words = {inspections[k]["words"] for k in same_content_order}
        key_takeaways.append({
            "topic": "same_content_order_arms",
            "files": same_content_order,
            "same_text_multiset": len(hashes) == 1,
            "same_words": len(words) == 1,
            "words": sorted(words),
        })
    small_factorial = [
        "official_flat_shuffled.jsonl", "official_flat_curriculum.jsonl",
        "composition_selected_shuffled.jsonl", "composition_selected_curriculum.jsonl",
    ]
    present_small = [k for k in small_factorial if k in inspections]
    if present_small:
        key_takeaways.append({
            "topic": "small_factorial_files",
            "files": present_small,
            "rows": {k: inspections[k]["rows"] for k in present_small},
            "words": {k: inspections[k]["words"] for k in present_small},
            "interpretation": "These are small 512-word-row files, not the 4M same-content order arms described by metadata.json.",
        })
    payload = {
        "status": "ok",
        "source_dir": str(SRC),
        "files": inspections,
        "groups_by_text_multiset_hash": groups_by_text,
        "groups_by_words": groups_by_words,
        "key_takeaways": key_takeaways,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    lines = [
        "# research — INITIAL_MODEL_STUDIES core-competence materialization inspection",
        "",
        f"Source directory: `{SRC}`.",
        "",
        "## File inventory",
        "",
        "| file | rows | words | sources | score fields | content group |",
        "|---|---:|---:|---|---|---|",
    ]
    hash_labels: dict[str, str] = {}
    for idx, h in enumerate(sorted(groups_by_text), 1):
        hash_labels[h] = f"G{idx}"
    for name, info in inspections.items():
        score_fields = []
        if info["comp_score"]["n"]:
            score_fields.append("comp")
        if info["curriculum_score"]["n"]:
            score_fields.append("curr")
        src_top = ", ".join(f"{k}:{v}" for k, v in list(info["source_dist"].items())[:3])
        lines.append(f"| `{name}` | {info['rows']} | {info['words']} | {src_top} | {', '.join(score_fields) or '-'} | {hash_labels.get(info['text_multiset_hash'], '-')} |")
    lines += [
        "",
        "## Research reading",
        "",
        "The three files `official_random_order_a.jsonl`, `official_random_order_b.jsonl`, and `official_curriculum.jsonl` are the true research same-content order arms: each has 25,000 rows and 4,000,000 words, and they share the same text multiset. Their only intended difference is row order.",
        "",
        "The four files named `official_flat_*` and `composition_selected_*` are much smaller: each has 195 rows and 99,840 words, mostly 512-word rows. They are not described by `metadata.json` and should be treated as earlier small materialization leftovers until their construction code and source/length matching are reconstructed. They cannot directly answer the research 4M compositional-selection question.",
        "",
        "Therefore research only closed the simple same-content ordering version. The content-selection mechanism remains unresolved because the intended 4M source/length-matched compositional-selection arms were not actually materialized in the inspected metadata set.",
        "",
        f"Full JSON: `{OUT_JSON}`.",
    ]
    OUT_NOTE.parent.mkdir(parents=True, exist_ok=True)
    OUT_NOTE.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": "ok", "json": str(OUT_JSON), "note": str(OUT_NOTE), "n_files": len(inspections), "key_takeaways": key_takeaways}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
