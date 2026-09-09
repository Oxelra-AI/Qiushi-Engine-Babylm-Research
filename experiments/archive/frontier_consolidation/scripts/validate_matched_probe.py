#!/usr/bin/env python3
"""research: validate generated matched graph-transfer probe examples.

This script performs hard, local checks before any frozen-model scoring:
- target/distractor from approved opposite pairs;
- target/distractor absent from all context views;
- structured/neutral/reversed contexts are close in length and lexical style;
- reversed context is a minimal-ish edit of structured context;
- query uses a different lexical surface and is not copied verbatim from a view.

The checks are deliberately conservative enough to prevent the research lexical-reuse
failure mode from becoming a false positive, but they cannot prove semantic
correctness by themselves.  They produce a packet for a small frozen transfer
measurement and a reviewable Markdown summary.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import json
import math
import re
import statistics
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

ROOT = _public_path('.')
STUDY = _public_path('experiments/archive/frontier_consolidation')
WORKSPACE = _public_path('experiments/archive/frontier_consolidation')
BASE = _public_path('experiments/archive/frontier_consolidation/data/matched_graph_transfer_probe')
PROMPTS = _public_path('experiments/archive/frontier_consolidation/data/matched_graph_transfer_probe/matched_probe_generation_prompts.jsonl')
OUTPUTS = _public_path('experiments/archive/frontier_consolidation/data/matched_graph_transfer_probe/matched_probe_generation_outputs.jsonl')
ACCEPTED = _public_path('experiments/archive/frontier_consolidation/data/matched_graph_transfer_probe/matched_probe_accepted.jsonl')
REJECTED = _public_path('experiments/archive/frontier_consolidation/data/matched_graph_transfer_probe/matched_probe_rejected.jsonl')
SUMMARY_JSON = _public_path('experiments/archive/frontier_consolidation/data/matched_graph_transfer_probe/matched_probe_validation_summary.json')
SUMMARY_MD = _public_path('research/documents/frontier_consolidation/data/matched_graph_transfer_probe/matched_probe_validation_summary.md')

ALLOWED_PAIRS = {
    ("after", "before"), ("before", "after"),
    ("because", "despite"), ("despite", "because"),
    ("more", "less"), ("less", "more"),
    ("inside", "outside"), ("outside", "inside"),
    ("accepted", "rejected"), ("rejected", "accepted"),
    ("gained", "lost"), ("lost", "gained"),
    ("entered", "left"), ("left", "entered"),
    ("true", "false"), ("false", "true"),
}

STOP = {
    "the", "a", "an", "and", "or", "but", "to", "of", "in", "on", "for", "with", "by", "as", "at", "from",
    "that", "this", "these", "those", "was", "were", "is", "are", "be", "been", "being", "had", "has", "have",
    "he", "she", "it", "they", "them", "his", "her", "their", "its", "who", "which", "while", "when", "then",
}


def rel(p: Path | str) -> str:
    try:
        return str(Path(p).resolve().relative_to(ROOT))
    except Exception:
        return str(p)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows=[]
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def extract_json(text: str) -> tuple[dict[str, Any] | None, str | None]:
    text = (text or "").strip()
    if not text:
        return None, "empty-output"
    # Remove common code fences without trusting them.
    text2 = re.sub(r"^```(?:json)?", "", text, flags=re.I).strip()
    text2 = re.sub(r"```$", "", text2).strip()
    # First direct parse.
    try:
        obj = json.loads(text2)
        if isinstance(obj, dict):
            return obj, None
    except Exception:
        pass
    # Extract first balanced-ish object.
    start = text2.find("{")
    end = text2.rfind("}")
    if start >= 0 and end > start:
        chunk = text2[start:end+1]
        try:
            obj = json.loads(chunk)
            if isinstance(obj, dict):
                return obj, None
        except Exception as e:
            return None, f"json-parse-failed:{type(e).__name__}"
    return None, "no-json-object"


def words(text: str) -> list[str]:
    return re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?|\d+", (text or "").lower())


def content_words(text: str) -> set[str]:
    return {w for w in words(text) if len(w) > 2 and w not in STOP}


def word_count(text: str) -> int:
    return len(words(text))


def contains_word_or_substring(text: str, word: str) -> bool:
    t = (text or "").lower()
    w = (word or "").lower().strip()
    if not w:
        return False
    # Treat exact word occurrence as fatal and also reject obvious inflection/subword copies.
    if re.search(r"(?<![a-z])" + re.escape(w) + r"(?![a-z])", t):
        return True
    if len(w) >= 5 and w in t:
        return True
    return False


def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / max(1, len(a | b))


def ratio(a: str, b: str) -> float:
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def candidate_token_check(tokenizer, target: str, distractor: str) -> tuple[bool, dict[str, Any]]:
    info={}
    ok=True
    for label, w in [("target", target), ("distractor", distractor)]:
        ids = tokenizer.encode(" " + w, add_special_tokens=False)
        toks = tokenizer.convert_ids_to_tokens(ids)
        # Ignore a standalone whitespace-marker token when present; require that the lexical word is not split into too many pieces.
        lexical = [t for t in toks if t not in {"Ġ", "▁"}]
        info[label] = {"ids": [int(x) for x in ids], "tokens": toks, "lexical_piece_count": len(lexical)}
        if len(lexical) < 1 or len(lexical) > 2:
            ok=False
    return ok, info


def validate_record(meta: dict[str, Any], obj: dict[str, Any], tokenizer=None) -> tuple[bool, list[str], dict[str, Any]]:
    issues=[]
    rec={"prompt_meta": {k: meta.get(k) for k in ["prompt_id","row_index","family","source_bucket","selection_score"]}}
    if obj.get("ok") is False:
        return False, ["generator-ok-false:" + str(obj.get("reason", ""))[:120]], rec
    required = ["relation_family", "target", "distractor", "view_structured", "view_neutral", "view_reversed", "query_prefix", "query_suffix", "correct_full_query", "distractor_full_query", "edge_correct", "edge_reversed"]
    for k in required:
        if not isinstance(obj.get(k), str) or not obj.get(k).strip():
            issues.append(f"missing-{k}")
    if issues:
        return False, issues, rec
    target = obj["target"].strip().lower()
    distractor = obj["distractor"].strip().lower()
    if (target, distractor) not in ALLOWED_PAIRS:
        issues.append("target-distractor-not-allowed-pair")
    if not re.fullmatch(r"[a-z]+", target or "") or not re.fullmatch(r"[a-z]+", distractor or ""):
        issues.append("target-or-distractor-not-single-alpha-word")
    views = {k: obj[k].strip() for k in ["view_structured", "view_neutral", "view_reversed"]}
    for vk, txt in views.items():
        wc = word_count(txt)
        if wc < 14 or wc > 42:
            issues.append(f"{vk}-word-count-{wc}-outside-14-42")
        for w in [target, distractor]:
            if contains_word_or_substring(txt, w):
                issues.append(f"{vk}-contains-{w}")
        if any(marker in txt for marker in ["->", "=>", "{", "}", "[", "]"]):
            issues.append(f"{vk}-contains-unnatural-marker")
    wc_s = word_count(views["view_structured"]); wc_n = word_count(views["view_neutral"]); wc_r = word_count(views["view_reversed"])
    if abs(wc_s - wc_n) > 6:
        issues.append(f"structured-neutral-length-diff-{abs(wc_s-wc_n)}")
    if abs(wc_s - wc_r) > 6:
        issues.append(f"structured-reversed-length-diff-{abs(wc_s-wc_r)}")
    cw_s = content_words(views["view_structured"]); cw_n = content_words(views["view_neutral"]); cw_r = content_words(views["view_reversed"])
    jac_sn = jaccard(cw_s, cw_n); jac_sr = jaccard(cw_s, cw_r)
    sim_sn = ratio(views["view_structured"], views["view_neutral"]); sim_sr = ratio(views["view_structured"], views["view_reversed"])
    if jac_sn < 0.35:
        issues.append(f"structured-neutral-content-jaccard-low-{jac_sn:.3f}")
    if jac_sr < 0.45:
        issues.append(f"structured-reversed-content-jaccard-low-{jac_sr:.3f}")
    if sim_sr < 0.55:
        issues.append(f"structured-reversed-char-sim-low-{sim_sr:.3f}")
    if views["view_structured"].lower() == views["view_neutral"].lower():
        issues.append("structured-neutral-identical")
    if views["view_structured"].lower() == views["view_reversed"].lower():
        issues.append("structured-reversed-identical")
    qp = obj["query_prefix"].strip()
    qs = obj["query_suffix"].strip()
    if word_count(qp + " " + qs) < 4:
        issues.append("query-too-short")
    query_surface = (qp + " " + qs).lower()
    # Query should not be a verbatim substring of either view.
    if len(query_surface) > 16 and any(query_surface in v.lower() for v in views.values()):
        issues.append("query-copied-from-view")
    if tokenizer is not None:
        tok_ok, tok_info = candidate_token_check(tokenizer, target, distractor)
        rec["tokenizer_target_info"] = tok_info
        if not tok_ok:
            issues.append("target-or-distractor-tokenization-too-fragmented")
    clean_obj = {k: obj.get(k) for k in required}
    clean_obj["new_names"] = obj.get("new_names") if isinstance(obj.get("new_names"), list) else []
    rec.update({
        "prompt_id": meta.get("prompt_id"),
        "row_index": meta.get("row_index"),
        "family": meta.get("family"),
        "source_bucket": meta.get("source_bucket"),
        "generated": clean_obj,
        "validation_metrics": {
            "wc_structured": wc_s,
            "wc_neutral": wc_n,
            "wc_reversed": wc_r,
            "content_jaccard_structured_neutral": jac_sn,
            "content_jaccard_structured_reversed": jac_sr,
            "char_sim_structured_neutral": sim_sn,
            "char_sim_structured_reversed": sim_sr,
        },
        "validation_issues": issues,
    })
    return len(issues) == 0, issues, rec


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", default=str(PROMPTS))
    ap.add_argument("--outputs", default=str(OUTPUTS))
    ap.add_argument("--accepted", default=str(ACCEPTED))
    ap.add_argument("--rejected", default=str(REJECTED))
    ap.add_argument("--summary-json", default=str(SUMMARY_JSON))
    ap.add_argument("--summary-md", default=str(SUMMARY_MD))
    args = ap.parse_args()
    prompts_path = Path(args.prompts)
    outputs_path = Path(args.outputs)
    accepted_path = Path(args.accepted)
    rejected_path = Path(args.rejected)
    summary_json_path = Path(args.summary_json)
    summary_md_path = Path(args.summary_md)
    for p in [accepted_path.parent, rejected_path.parent, summary_json_path.parent, summary_md_path.parent]:
        p.mkdir(parents=True, exist_ok=True)
    prompts = load_jsonl(prompts_path)
    outputs = load_jsonl(outputs_path)
    p_by_index = {i: p for i, p in enumerate(prompts)}

    tokenizer = None
    try:
        from transformers import AutoTokenizer
        ckpt = _public_path('experiments/archive/frontier_consolidation/training/runs/adapter128_scale1p75_h100M100M_seed43022_official_ladder/hf_model/chck_82M')
        tokenizer = AutoTokenizer.from_pretrained(ckpt, trust_remote_code=True, local_files_only=True)
    except Exception:
        tokenizer = None

    accepted=[]; rejected=[]; issue_counts=Counter(); family_counts=Counter(); fam_acc=Counter()
    for out in outputs:
        idx = int(out.get("index", len(accepted)+len(rejected)))
        meta = p_by_index.get(idx, {})
        obj, err = extract_json(str(out.get("output", "")))
        if obj is None:
            rec = {"prompt_meta": meta, "raw_output": out.get("output"), "validation_issues": [err or "parse-failed"]}
            rejected.append(rec); issue_counts[err or "parse-failed"] += 1
            continue
        ok, issues, rec = validate_record(meta, obj, tokenizer=tokenizer)
        rec["raw_generated_tokens"] = out.get("generated_tokens")
        family_counts[str(meta.get("family"))] += 1
        if ok:
            accepted.append(rec); fam_acc[str(meta.get("family"))] += 1
        else:
            rejected.append(rec)
            for issue in issues:
                issue_counts[issue] += 1
    with accepted_path.open("w", encoding="utf-8") as f:
        for r in accepted:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with rejected_path.open("w", encoding="utf-8") as f:
        for r in rejected:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    metrics = [r["validation_metrics"] for r in accepted]
    summary = {
        "status": "MATCHED_PROBE_VALIDATION",
        "prompts": len(prompts),
        "outputs": len(outputs),
        "accepted": len(accepted),
        "rejected": len(rejected),
        "accepted_by_family": dict(fam_acc),
        "prompted_by_family": dict(family_counts),
        "issue_counts": dict(issue_counts.most_common()),
        "accepted_file": rel(accepted_path),
        "rejected_file": rel(rejected_path),
        "tokenizer_loaded": tokenizer is not None,
    }
    if metrics:
        for k in sorted(metrics[0]):
            vals=[float(m[k]) for m in metrics if isinstance(m.get(k), (int,float)) and math.isfinite(float(m[k]))]
            if vals:
                summary[k] = {"mean": statistics.fmean(vals), "min": min(vals), "max": max(vals)}
    summary_json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines=[
        "# research matched graph-transfer probe validation",
        "",
        f"Prompts: {len(prompts)}; outputs: {len(outputs)}; accepted: {len(accepted)}; rejected: {len(rejected)}",
        f"Accepted by family: `{dict(fam_acc)}`",
        "",
        "## Why this validation exists",
        "The research packet was not decisive because graph_compact exposed relation words that ordinary_compact dropped. This validation accepts only examples where the answer word and its opposite are absent from every context view, where structured and neutral views have similar length and content overlap, and where reversed views are close surface edits. A positive frozen score on this packet would still need interpretation; a failed construction or failed transfer stops the route before any charged pretraining.",
        "",
        "## Frequent issues",
    ]
    for k,v in issue_counts.most_common(20):
        lines.append(f"- {k}: {v}")
    if metrics:
        lines.extend(["", "## Accepted metric ranges"])
        for k,v in summary.items():
            if isinstance(v, dict) and {"mean","min","max"} <= set(v):
                lines.append(f"- {k}: mean {v['mean']:.3f}, min {v['min']:.3f}, max {v['max']:.3f}")
    lines.extend(["", f"Accepted JSONL: `{rel(accepted_path)}`", f"Rejected JSONL: `{rel(rejected_path)}`", f"Summary JSON: `{rel(summary_json_path)}`"])
    summary_md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": summary["status"], "accepted": len(accepted), "rejected": len(rejected), "out_json": rel(summary_json_path), "out_md": rel(summary_md_path)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
