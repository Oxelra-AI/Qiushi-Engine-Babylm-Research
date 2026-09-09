#!/usr/bin/env python3
"""research: recover original Qwen3.5-9B compact-view generation path and measure quality.

This script is CPU-only except for generation itself, which is run separately by
`external generator`. It prepares representative prompt samples and then
merges model outputs with source records to measure whether compact rewrites keep
factual anchors and relation structure well enough for the FW1.5M mechanism-family
corpus construction.
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
import pathlib
import random
import re
import statistics
import time
from typing import Any, Iterable

SCRIPT_DIR = _public_path('experiments/archive/representation_and_objectives/scripts')
WORKSPACE = _public_path('experiments/archive/representation_and_objectives')
DEFAULT_PROMPTS = _public_path('experiments/archive/representation_and_objectives/data/fw_mechanism_source_selection/fw_mechanism_compact_prompts.jsonl')
OUT_DIR = _public_path('experiments/archive/representation_and_objectives/data/qwen35_teacher_recovery')
NOTE_PATH = _public_path('research/notes/representation_and_objectives/qwen35_teacher_recovery.md')

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "being", "by", "for", "from", "has", "have",
    "had", "he", "her", "his", "i", "in", "is", "it", "its", "of", "on", "or", "she", "that", "the",
    "their", "they", "this", "to", "was", "were", "which", "who", "will", "with", "would", "you",
}
NEGATION_MARKERS = {
    "not", "no", "never", "none", "neither", "nor", "without", "lack", "lacks", "lacked", "cannot", "can't", "won't", "isn't", "aren't", "doesn't", "don't", "didn't", "wasn't", "weren't", "non",
}
MODAL_MARKERS = {
    "may", "might", "can", "could", "should", "would", "must", "possible", "possibly", "likely", "unlikely", "potential", "potentially", "probable", "probably", "suspect", "suspected", "suggest", "suggests", "seems", "appears",
}
CAUSAL_MARKERS = {
    "because", "cause", "caused", "causes", "causing", "due", "therefore", "thereby", "hence", "since", "so", "led", "leads", "result", "results", "resulted", "resulting", "allow", "allows", "allowed", "enable", "enables", "enabled", "prevent", "prevents", "prevented", "if", "when",
}
COMPARISON_MARKERS = {
    "more", "less", "fewer", "greater", "smaller", "larger", "higher", "lower", "than", "before", "after", "between", "while", "whereas", "unlike", "compared", "similar", "different", "differences",
}


def read_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: pathlib.Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def wc(text: str) -> int:
    return len((text or "").split())


def lexical_words(text: str) -> list[str]:
    return re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?|\d[\d,\.]*", (text or "").lower())


def content_words(text: str) -> list[str]:
    return [w for w in lexical_words(text) if w not in STOPWORDS and len(w) > 1]


def marker_set(text: str, markers: set[str]) -> set[str]:
    words = set(lexical_words(text))
    out = words & markers
    # Capture common multi-word causal forms that word-set matching misses.
    t = " " + " ".join(lexical_words(text)) + " "
    if " due to " in t:
        out.add("due")
    if " led to " in t:
        out.add("led")
    if " as a result " in t:
        out.add("result")
    return out


def norm_text(text: str) -> str:
    return " ".join(lexical_words(text))


def norm_hash(text: str) -> str:
    return hashlib.sha256(norm_text(text).encode("utf-8")).hexdigest()[:32]


def extract_entities(source_text: str, row: dict[str, Any]) -> list[str]:
    for key in ("source_entities", "entities"):
        vals = row.get(key)
        if vals:
            return [str(x).strip() for x in vals if str(x).strip()]
    caps = re.findall(r"\b[A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)*\b", source_text or "")
    bad = {"The", "A", "An", "In", "On", "At", "This", "That", "These", "Those"}
    return [c for c in caps if c not in bad and len(c) > 2]


def extract_numbers(text: str, row: dict[str, Any] | None = None) -> list[str]:
    if row:
        vals = row.get("source_numbers") or row.get("numbers")
        if vals:
            return [str(x).strip() for x in vals if str(x).strip()]
    return re.findall(r"\b\d[\d,\.]*\b", text or "")


def recall(items: list[str], text: str) -> float:
    if not items:
        return 1.0
    tl = (text or "").lower()
    found = 0
    for x in items:
        xl = str(x).lower().strip()
        if not xl:
            continue
        if xl in tl:
            found += 1
    denom = len([x for x in items if str(x).strip()])
    return found / max(1, denom)


def word_recall(source: str, rewrite: str, content_only: bool = False) -> float:
    sw = set(content_words(source) if content_only else lexical_words(source))
    rw = set(content_words(rewrite) if content_only else lexical_words(rewrite))
    if not sw:
        return 1.0
    return len(sw & rw) / len(sw)


def marker_recall(source: str, rewrite: str, markers: set[str]) -> float:
    sm = marker_set(source, markers)
    if not sm:
        return 1.0
    rm = marker_set(rewrite, markers)
    return len(sm & rm) / len(sm)


def clean_output(text: str) -> str:
    s = (text or "").strip()
    s = re.sub(r"^```(?:\w+)?", "", s).strip()
    s = re.sub(r"```$", "", s).strip()
    s = re.sub(r"^(Rewrite|Output|Answer|Simplified sentence|Sentence)\s*:\s*", "", s, flags=re.I).strip()
    if "\n" in s:
        parts = [p.strip() for p in s.splitlines() if p.strip()]
        if parts:
            s = parts[0]
    s = s.strip(" \t\"'`“”")
    return s


def feature_record(row: dict[str, Any]) -> dict[str, Any]:
    src = str(row.get("source_text") or row.get("text") or "")
    domains = row.get("domains") or row.get("domain_hits") or []
    if not domains:
        domains = ["no_domain"]
    ents = extract_entities(src, row)
    nums = extract_numbers(src, row)
    sw = int(row.get("source_words") or row.get("words") or wc(src))
    if sw <= 18:
        length_bin = "short"
    elif sw <= 30:
        length_bin = "medium"
    else:
        length_bin = "long"
    feats = {
        "length_bin": length_bin,
        "has_entity": bool(ents),
        "has_number": bool(nums),
        "has_negation": bool(marker_set(src, NEGATION_MARKERS)),
        "has_modality": bool(marker_set(src, MODAL_MARKERS)),
        "has_causal": bool(marker_set(src, CAUSAL_MARKERS)),
        "has_comparison": bool(marker_set(src, COMPARISON_MARKERS)),
        "primary_domain": str(domains[0]),
        "domains": [str(d) for d in domains],
    }
    return feats


def prepare_pilot(args: argparse.Namespace) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    prompts = read_jsonl(pathlib.Path(args.prompts))
    rng = random.Random(args.seed)

    scored = []
    for i, p in enumerate(prompts):
        feats = feature_record(p)
        bucket = (
            feats["primary_domain"], feats["length_bin"],
            "E" if feats["has_entity"] else "e0",
            "N" if feats["has_number"] else "n0",
            "G" if feats["has_negation"] else "g0",
            "M" if feats["has_modality"] else "m0",
            "C" if feats["has_causal"] else "c0",
            "R" if feats["has_comparison"] else "r0",
        )
        scored.append((bucket, i, p, feats))

    by_bucket: dict[tuple[Any, ...], list[tuple[int, dict[str, Any], dict[str, Any]]]] = collections.defaultdict(list)
    for bucket, i, p, feats in scored:
        by_bucket[bucket].append((i, p, feats))
    for vals in by_bucket.values():
        rng.shuffle(vals)

    selected_indices: set[int] = set()
    # First pass: one from the richer buckets, prioritizing structure markers and nonempty domains.
    bucket_order = sorted(
        by_bucket,
        key=lambda b: (
            -sum(1 for x in b[2:] if x in {"E", "N", "G", "M", "C", "R"}),
            b[0] == "no_domain",
            b[0], b[1], b[2:],
        ),
    )
    for b in bucket_order:
        if len(selected_indices) >= args.n:
            break
        i, _p, _f = by_bucket[b][0]
        selected_indices.add(i)

    # Second pass: enforce explicit marker coverage if present in the pool.
    markers = ["has_negation", "has_modality", "has_causal", "has_comparison", "has_number", "has_entity"]
    for marker in markers:
        candidates = [(i, p, f) for _b, i, p, f in scored if f[marker] and i not in selected_indices]
        rng.shuffle(candidates)
        for i, _p, _f in candidates[: max(0, min(args.n // 12, len(candidates)) )]:
            if len(selected_indices) >= args.n:
                break
            selected_indices.add(i)
        if len(selected_indices) >= args.n:
            break

    # Final fill using deterministic random order over all remaining prompts.
    remaining = [i for _b, i, _p, _f in scored if i not in selected_indices]
    rng.shuffle(remaining)
    for i in remaining:
        if len(selected_indices) >= args.n:
            break
        selected_indices.add(i)

    selected = []
    for new_index, i in enumerate(sorted(selected_indices)):
        rec = dict(prompts[i])
        rec["pilot_original_index"] = i
        rec["pilot_index"] = new_index
        rec["pilot_features"] = feature_record(rec)
        selected.append(rec)

    out_path = pathlib.Path(args.out_prompts)
    write_jsonl(out_path, selected)
    feat_counts: dict[str, Any] = {
        "rows": len(selected),
        "pool_rows": len(prompts),
        "pool_sha256": sha256_file(pathlib.Path(args.prompts)),
    }
    for marker in markers:
        feat_counts[marker] = sum(1 for r in selected if r["pilot_features"][marker])
    domains = collections.Counter(r["pilot_features"]["primary_domain"] for r in selected)
    lengths = collections.Counter(r["pilot_features"]["length_bin"] for r in selected)
    feat_counts["domain_counts"] = dict(domains.most_common())
    feat_counts["length_counts"] = dict(lengths.most_common())
    meta = {
        "status": "QWEN35_RECOVERY_PILOT_PROMPTS",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "purpose": "Representative same-source prompt sample to verify the original Qwen3.5-9B compact-view generation process before producing the FW1.5M pair family.",
        "input_prompts": str(args.prompts),
        "output_prompts": str(out_path),
        "seed": args.seed,
        "feature_counts": feat_counts,
    }
    meta_path = _public_path('experiments/archive/representation_and_objectives/data/qwen35_teacher_recovery/qwen35_recovery_pilot_prompts_manifest.json')
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": meta["status"], "output": str(out_path), "rows": len(selected), "feature_counts": feat_counts}, indent=2, ensure_ascii=False))


def stat(vals: list[float]) -> dict[str, Any]:
    if not vals:
        return {"n": 0}
    xs = sorted(float(x) for x in vals)
    def q(frac: float) -> float:
        return xs[int(round(frac * (len(xs) - 1)))]
    return {
        "n": len(xs),
        "mean": statistics.fmean(xs),
        "median": statistics.median(xs),
        "p05": q(0.05),
        "p25": q(0.25),
        "p75": q(0.75),
        "p95": q(0.95),
        "min": xs[0],
        "max": xs[-1],
    }


def analyze_outputs(args: argparse.Namespace) -> None:
    prompts = read_jsonl(pathlib.Path(args.prompts))
    outputs = read_jsonl(pathlib.Path(args.outputs))
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    out_by_index: dict[int, dict[str, Any]] = {}
    for j, r in enumerate(outputs):
        idx = int(r.get("index", j))
        out_by_index[idx] = r

    merged: list[dict[str, Any]] = []
    accepted_core: list[dict[str, Any]] = []
    accepted_struct: list[dict[str, Any]] = []
    for i, p in enumerate(prompts):
        o = out_by_index.get(i, {})
        source_text = str(p.get("source_text") or p.get("text") or "")
        output_text = clean_output(str(o.get("output") or o.get("generated") or o.get("text") or ""))
        src_words = int(p.get("source_words") or p.get("words") or wc(source_text))
        out_words = wc(output_text)
        ratio = out_words / max(1, src_words)
        ents = extract_entities(source_text, p)
        nums = extract_numbers(source_text, p)
        lex_recall = word_recall(source_text, output_text, content_only=False)
        cont_recall = word_recall(source_text, output_text, content_only=True)
        ent_recall = recall(ents, output_text)
        num_recall = recall(nums, output_text)
        neg_recall = marker_recall(source_text, output_text, NEGATION_MARKERS)
        modal_recall = marker_recall(source_text, output_text, MODAL_MARKERS)
        causal_recall = marker_recall(source_text, output_text, CAUSAL_MARKERS)
        comp_recall = marker_recall(source_text, output_text, COMPARISON_MARKERS)
        norm_src = norm_text(source_text)
        norm_out = norm_text(output_text)
        exact_copy = norm_src == norm_out
        copy_like = exact_copy or (ratio >= 0.90 and lex_recall >= 0.85)
        too_short = out_words < 5
        malformed = (not output_text) or len(output_text) > 800 or output_text.count("\n") > 0
        core_accept = (
            not malformed and not too_short and not copy_like and
            0.35 <= ratio <= 0.85 and
            cont_recall >= 0.45 and
            ent_recall >= 0.75 and
            num_recall >= 1.0
        )
        struct_accept = (
            core_accept and
            neg_recall >= 1.0 and
            modal_recall >= 1.0 and
            causal_recall >= 0.5 and
            comp_recall >= 0.5
        )
        rec = {
            "prompt_index": i,
            "prompt_id": p.get("prompt_id", ""),
            "norm_hash": p.get("norm_hash") or norm_hash(source_text),
            "source_text": source_text,
            "rewrite_text": output_text,
            "source_words": src_words,
            "rewrite_words": out_words,
            "compression_ratio": ratio,
            "lexical_recall": lex_recall,
            "content_recall": cont_recall,
            "entity_recall": ent_recall,
            "number_recall": num_recall,
            "negation_marker_recall": neg_recall,
            "modality_marker_recall": modal_recall,
            "causal_marker_recall": causal_recall,
            "comparison_marker_recall": comp_recall,
            "exact_copy": exact_copy,
            "copy_like": copy_like,
            "malformed": malformed,
            "too_short": too_short,
            "accepted_core": core_accept,
            "accepted_structure_strict": struct_accept,
            "accepted": struct_accept,
            "model_id": o.get("model", args.model_label),
            "generated_tokens": o.get("generated_tokens"),
            "doc_id": p.get("doc_id", ""),
            "domains": p.get("domains") or p.get("domain_hits") or [],
            "source_entities": ents,
            "source_numbers": nums,
        }
        merged.append(rec)
        if core_accept:
            accepted_core.append(rec)
        if struct_accept:
            accepted_struct.append(rec)

    def frac(pred) -> float:
        return sum(1 for r in merged if pred(r)) / max(1, len(merged))

    summary = {
        "status": "QWEN35_GENERATION_ANALYZED",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "purpose": "Measure whether the recovered original Qwen3.5-9B compact generator preserves anchors and relation structure on the FW1.5M new-source subset.",
        "prompts": str(args.prompts),
        "prompts_sha256": sha256_file(pathlib.Path(args.prompts)),
        "outputs": str(args.outputs),
        "outputs_sha256": sha256_file(pathlib.Path(args.outputs)),
        "model_label": args.model_label,
        "n_prompts": len(prompts),
        "n_outputs": len(outputs),
        "n_merged": len(merged),
        "n_accepted_core": len(accepted_core),
        "n_accepted_structure_strict": len(accepted_struct),
        "core_accept_rate": len(accepted_core) / max(1, len(merged)),
        "structure_strict_accept_rate": len(accepted_struct) / max(1, len(merged)),
        "accepted_structure_rewrite_words": sum(r["rewrite_words"] for r in accepted_struct),
        "accepted_core_rewrite_words": sum(r["rewrite_words"] for r in accepted_core),
        "stats_all": {
            "compression_ratio": stat([r["compression_ratio"] for r in merged]),
            "rewrite_words": stat([r["rewrite_words"] for r in merged]),
            "content_recall": stat([r["content_recall"] for r in merged]),
            "entity_recall": stat([r["entity_recall"] for r in merged]),
            "number_recall": stat([r["number_recall"] for r in merged]),
            "negation_marker_recall": stat([r["negation_marker_recall"] for r in merged if marker_set(r["source_text"], NEGATION_MARKERS)]),
            "modality_marker_recall": stat([r["modality_marker_recall"] for r in merged if marker_set(r["source_text"], MODAL_MARKERS)]),
            "causal_marker_recall": stat([r["causal_marker_recall"] for r in merged if marker_set(r["source_text"], CAUSAL_MARKERS)]),
            "comparison_marker_recall": stat([r["comparison_marker_recall"] for r in merged if marker_set(r["source_text"], COMPARISON_MARKERS)]),
        },
        "stats_accepted_structure": {
            "compression_ratio": stat([r["compression_ratio"] for r in accepted_struct]),
            "rewrite_words": stat([r["rewrite_words"] for r in accepted_struct]),
            "content_recall": stat([r["content_recall"] for r in accepted_struct]),
            "entity_recall": stat([r["entity_recall"] for r in accepted_struct]),
            "number_recall": stat([r["number_recall"] for r in accepted_struct]),
        },
        "fractions": {
            "copy_like": frac(lambda r: r["copy_like"]),
            "too_short": frac(lambda r: r["too_short"]),
            "malformed": frac(lambda r: r["malformed"]),
            "entity_loss_when_source_has_entities": frac(lambda r: bool(r["source_entities"]) and r["entity_recall"] < 1.0),
            "number_loss_when_source_has_numbers": frac(lambda r: bool(r["source_numbers"]) and r["number_recall"] < 1.0),
            "negation_loss_when_source_has_negation": frac(lambda r: bool(marker_set(r["source_text"], NEGATION_MARKERS)) and r["negation_marker_recall"] < 1.0),
            "modality_loss_when_source_has_modality": frac(lambda r: bool(marker_set(r["source_text"], MODAL_MARKERS)) and r["modality_marker_recall"] < 1.0),
            "causal_low_when_source_has_causal": frac(lambda r: bool(marker_set(r["source_text"], CAUSAL_MARKERS)) and r["causal_marker_recall"] < 0.5),
            "comparison_low_when_source_has_comparison": frac(lambda r: bool(marker_set(r["source_text"], COMPARISON_MARKERS)) and r["comparison_marker_recall"] < 0.5),
        },
        "outputs_written": {
            "merged": str(out_dir / "qwen35_generation_merged_quality.jsonl"),
            "accepted_core": str(out_dir / "qwen35_generation_accepted_core.jsonl"),
            "accepted_structure_strict": str(out_dir / "qwen35_generation_accepted_structure_strict.jsonl"),
            "summary": str(out_dir / "qwen35_generation_quality_summary.json"),
        },
    }
    write_jsonl(out_dir / "qwen35_generation_merged_quality.jsonl", merged)
    write_jsonl(out_dir / "qwen35_generation_accepted_core.jsonl", accepted_core)
    write_jsonl(out_dir / "qwen35_generation_accepted_structure_strict.jsonl", accepted_struct)
    (out_dir / "qwen35_generation_quality_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    note_lines = [
        "# research — Qwen3.5-9B teacher recovery quality\n\n",
        "This analysis checks the original Qwen3.5-9B compact-view generation process on the FW mechanism-route sources before any BabyLM pretraining is launched.\n\n",
        f"Prompts: `{args.prompts}`\n\n",
        f"Outputs: `{args.outputs}`\n\n",
        f"Rows merged: {len(merged):,}; core accepted: {len(accepted_core):,} ({summary['core_accept_rate']:.3f}); structure-strict accepted: {len(accepted_struct):,} ({summary['structure_strict_accept_rate']:.3f}).\n\n",
        f"Accepted structure-strict rewrite words: {summary['accepted_structure_rewrite_words']:,}.\n\n",
        "## Main measurements\n\n",
        f"- Compression ratio median/all: {summary['stats_all']['compression_ratio'].get('median')} (mean {summary['stats_all']['compression_ratio'].get('mean')}).\n",
        f"- Content recall median/all: {summary['stats_all']['content_recall'].get('median')} (mean {summary['stats_all']['content_recall'].get('mean')}).\n",
        f"- Copy-like fraction: {summary['fractions']['copy_like']:.3f}.\n",
        f"- Entity loss when source has entities: {summary['fractions']['entity_loss_when_source_has_entities']:.3f}; number loss when source has numbers: {summary['fractions']['number_loss_when_source_has_numbers']:.3f}.\n",
        f"- Negation loss: {summary['fractions']['negation_loss_when_source_has_negation']:.3f}; modality loss: {summary['fractions']['modality_loss_when_source_has_modality']:.3f}; causal low-recall: {summary['fractions']['causal_low_when_source_has_causal']:.3f}; comparison low-recall: {summary['fractions']['comparison_low_when_source_has_comparison']:.3f}.\n\n",
        "## Files\n\n",
        f"- Summary JSON: `{out_dir / 'qwen35_generation_quality_summary.json'}`\n",
        f"- Merged rows: `{out_dir / 'qwen35_generation_merged_quality.jsonl'}`\n",
        f"- Accepted structure-strict rows: `{out_dir / 'qwen35_generation_accepted_structure_strict.jsonl'}`\n",
    ]
    NOTE_PATH.write_text("".join(note_lines), encoding="utf-8")
    print(json.dumps({
        "status": summary["status"],
        "n_merged": len(merged),
        "core_accept_rate": summary["core_accept_rate"],
        "structure_strict_accept_rate": summary["structure_strict_accept_rate"],
        "accepted_structure_rewrite_words": summary["accepted_structure_rewrite_words"],
        "summary": str(out_dir / "qwen35_generation_quality_summary.json"),
    }, indent=2, ensure_ascii=False))


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p1 = sub.add_parser("prepare-pilot")
    p1.add_argument("--prompts", default=str(DEFAULT_PROMPTS))
    p1.add_argument("--n", type=int, default=256)
    p1.add_argument("--seed", type=int, default=99035)
    p1.add_argument("--out-prompts", default=str(_public_path('experiments/archive/representation_and_objectives/data/qwen35_teacher_recovery/qwen35_recovery_pilot_prompts.jsonl')))
    p2 = sub.add_parser("analyze")
    p2.add_argument("--prompts", required=True)
    p2.add_argument("--outputs", required=True)
    p2.add_argument("--out-dir", required=True)
    p2.add_argument("--model-label", default="qwen3.5-9b")
    args = ap.parse_args()
    if args.cmd == "prepare-pilot":
        prepare_pilot(args)
    elif args.cmd == "analyze":
        analyze_outputs(args)
    else:
        raise ValueError(args.cmd)


if __name__ == "__main__":
    main()
