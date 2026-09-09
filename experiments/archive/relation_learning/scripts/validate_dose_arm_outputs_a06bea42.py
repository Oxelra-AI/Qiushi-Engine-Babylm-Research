#!/usr/bin/env python3
"""research: validate corrected restatement-dose rewrite outputs.

This validates Qwen3.5-9B rewrites for the ALN dose arm.  It fixes two research
issues:
  1. generation outputs from batch generation carry an `index`, so source
     metadata must be recovered from the prompt JSONL;
  2. the anti-copy metric is longest common CONTIGUOUS token span, matching the
     state-update validators, not longest common subsequence.

The main semantic validity filters reuse the COMPACT_EXPERIENCE research validator logic:
source/rewrite completeness, length ratio, number matching, entity recall, lower
content overlap, duplicate filtering, and copy-overlap.  Additional dose-arm
surface controls reject content-token Jaccard >= 0.80 and contiguous-copy spans
of >= 6 tokens.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import pathlib
import random
import re
import statistics
import sys
import time
from collections import Counter, defaultdict
from typing import Any

ROOT = pathlib.Path.cwd()
COMPACT_EXPERIENCE_SCRIPT = ROOT / "experiments/archive/compact_experience/scripts/clean_materialize_qwen_pairs.py"
DEFAULT_OUT_DIR = ROOT / "experiments/archive/relation_learning/data/dose_arm_validated"

MAX_CONTENT_JACCARD = 0.80
MAX_CONTIGUOUS_LCS = 5


def now_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def import_compact_experience():
    spec = importlib.util.spec_from_file_location("compact_experience_step028", COMPACT_EXPERIENCE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {COMPACT_EXPERIENCE_SCRIPT}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def read_jsonl(path: pathlib.Path, tolerate_partial: bool = False) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                if tolerate_partial:
                    continue
                raise RuntimeError(f"Bad JSON in {path} line {line_no}")
    return rows


def write_jsonl(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
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


def contiguous_lcs(a: list[str], b: list[str]) -> tuple[int, str]:
    best = 0
    best_i = 0
    dp = [0] * (len(b) + 1)
    for i, x in enumerate(a, start=1):
        new = [0] * (len(b) + 1)
        for j, y in enumerate(b, start=1):
            if x == y:
                new[j] = dp[j - 1] + 1
                if new[j] > best:
                    best = new[j]
                    best_i = i - best
        dp = new
    return best, " ".join(a[best_i:best_i + best]) if best else ""


def stat(xs: list[float]) -> dict[str, float | int | None]:
    ys = [float(x) for x in xs if math.isfinite(float(x))]
    if not ys:
        return {"n": 0, "mean": None, "median": None, "p95": None, "max": None}
    ys_sorted = sorted(ys)
    p95_idx = min(len(ys_sorted) - 1, int(math.ceil(0.95 * len(ys_sorted))) - 1)
    return {
        "n": len(ys),
        "mean": round(statistics.mean(ys), 4),
        "median": round(statistics.median(ys), 4),
        "p95": round(ys_sorted[p95_idx], 4),
        "max": round(max(ys), 4),
    }


def format_rec_for_source(compact_experience, meta: dict[str, Any]):
    return compact_experience.SourceSentence(
        pair_id=str(meta.get("original_id") or meta.get("id", "")).replace("dose56_", ""),
        text=compact_experience.normalize_ws(str(meta.get("source_sentence", ""))),
        words=int(meta.get("source_words") or len(str(meta.get("source_sentence", "")).split())),
        source=str(meta.get("source_name", meta.get("source", ""))),
        example_id=int(meta.get("source_example_id") or -1),
        cohort=str(meta.get("cohort", "dose")),
    )


def clean_generation_output(compact_experience, raw: str) -> str:
    out = compact_experience.clean_output(raw or "")
    # Keep only first nonempty line unless the model echoed prose after the first sentence.
    lines = [x.strip() for x in out.splitlines() if x.strip()]
    if lines:
        out = lines[0]
    low = out.lower()
    for pref in ("rewritten:", "rewrite:", "paraphrase:", "rewritten sentence:", "output:"):
        if low.startswith(pref):
            out = out[len(pref):].strip()
            low = out.lower()
    return compact_experience.strip_outer_quotes(out)


def validate_one(compact_experience, meta: dict[str, Any], raw_output: str, duplicate_seen: Counter[str]) -> tuple[dict[str, Any] | None, list[str], dict[str, Any]]:
    src = format_rec_for_source(compact_experience, meta)
    out = clean_generation_output(compact_experience, raw_output)
    reasons: list[str] = []
    detail: dict[str, Any] = {"raw_output": raw_output, "clean_output": out}
    if not out:
        return None, ["empty_output"], detail
    if "\n" in out or "\r" in out:
        reasons.append("newline")
    low = out.lower().strip()
    if low.startswith(compact_experience.META_PREFIXES):
        reasons.append("meta_prefix")
    if out[-1:] in {",", ";", ":", "-", "—"}:
        reasons.append("bad_terminal_fragment")
    if not compact_experience.is_complete_pair_text(out, is_source=False):
        reasons.append("rewrite_incomplete_or_fragment")
    if not compact_experience.is_complete_pair_text(src.text, is_source=True):
        reasons.append("source_incomplete_or_fragment")
    if compact_experience.has_repetition(out):
        reasons.append("repetition")

    ow = src.words
    rw = len(compact_experience.words(out))
    detail["source_words"] = ow
    detail["rewrite_words"] = rw
    if rw < compact_experience.MIN_REWRITE_WORDS:
        reasons.append(f"too_short_{rw}")
    if rw > compact_experience.MAX_REWRITE_WORDS:
        reasons.append(f"too_long_{rw}")
    ratio = rw / max(1, ow)
    detail["len_ratio"] = round(ratio, 4)
    if ratio < compact_experience.MIN_LEN_RATIO or ratio > compact_experience.MAX_LEN_RATIO:
        reasons.append(f"len_ratio_{ratio:.2f}")

    src_nums = compact_experience.number_tokens(src.text)
    out_nums = compact_experience.number_tokens(out)
    detail["num_source"] = src_nums
    detail["num_rewrite"] = out_nums
    if Counter(src_nums) != Counter(out_nums):
        reasons.append("number_mismatch")

    src_ents = compact_experience.entity_tokens(src.text)
    out_ents = compact_experience.entity_tokens(out)
    detail["entity_source"] = src_ents
    detail["entity_rewrite"] = out_ents
    if src_ents:
        out_norm = set(out_ents)
        matched = 0
        for e in src_ents:
            if e in out_norm or e.replace("'s", "") in out_norm or e.rstrip("s") in out_norm:
                matched += 1
        recall = matched / len(src_ents)
        needed = compact_experience.MIN_ENTITY_RECALL_SMALL if len(src_ents) <= 2 else compact_experience.MIN_ENTITY_RECALL_LARGE
        if recall < needed:
            reasons.append(f"entity_recall_{recall:.2f}")
    else:
        recall = 1.0
    detail["entity_recall"] = round(recall, 4)

    src_content = set(compact_experience.content_tokens(src.text))
    out_content = set(compact_experience.content_tokens(out))
    if src_content or out_content:
        overlap_min = len(src_content & out_content) / max(1, min(len(src_content), len(out_content)))
        jaccard = len(src_content & out_content) / max(1, len(src_content | out_content))
    else:
        overlap_min = 0.0
        jaccard = 0.0
    detail["content_overlap_min"] = round(overlap_min, 4)
    detail["content_jaccard"] = round(jaccard, 4)
    if overlap_min < compact_experience.MIN_CONTENT_OVERLAP and src.words >= 10:
        reasons.append(f"low_content_overlap_{overlap_min:.2f}")
    if overlap_min > compact_experience.MAX_CONTENT_OVERLAP and rw >= ow * 0.85:
        reasons.append(f"copy_overlap_{overlap_min:.2f}")
    if jaccard >= MAX_CONTENT_JACCARD:
        reasons.append(f"high_jaccard_{jaccard:.2f}")

    src_tokens = compact_experience.norm_tokens(src.text)
    out_tokens = compact_experience.norm_tokens(out)
    lcs, span = contiguous_lcs(src_tokens, out_tokens)
    detail["longest_common_contiguous_words"] = lcs
    detail["longest_common_span"] = span
    if lcs > MAX_CONTIGUOUS_LCS:
        reasons.append(f"contiguous_lcs_gt5_{lcs}")

    fp = compact_experience.output_fingerprint(out)
    if not fp:
        reasons.append("empty_fingerprint")
    if not reasons:
        duplicate_seen[fp] += 1
        if duplicate_seen[fp] > compact_experience.MAX_DUPLICATE_OUTPUTS_PER_FINGERPRINT:
            reasons.append("duplicate_output")

    if reasons:
        return None, reasons, detail

    rec = {
        "pair_id": str(meta.get("id", src.pair_id)),
        "original_id": str(meta.get("original_id", src.pair_id)),
        "original": src.text,
        "rewrite": out,
        "original_words": ow,
        "rewrite_words": rw,
        "pair_words": ow + rw,
        "source": src.source,
        "example_id": src.example_id,
        "source_sentence_idx": meta.get("source_sentence_idx"),
        "cohort": src.cohort,
        "len_ratio": ratio,
        "content_overlap": overlap_min,
        "content_jaccard": jaccard,
        "entity_recall": recall,
        "num_source": src_nums,
        "num_rewrite": out_nums,
        "entity_source": src_ents,
        "entity_rewrite": out_ents,
        "longest_common_contiguous_words": lcs,
        "longest_common_span": span,
    }
    return rec, [], detail


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--outputs", required=True)
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    ap.add_argument("--tag", default="pilot")
    ap.add_argument("--tolerate-partial", action="store_true")
    ap.add_argument("--manual-sample-n", type=int, default=80)
    args = ap.parse_args()

    started = time.time()
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    compact_experience = import_compact_experience()

    outputs_path = pathlib.Path(args.outputs)
    prompts_path = pathlib.Path(args.prompts)
    outputs = read_jsonl(outputs_path, tolerate_partial=args.tolerate_partial)
    prompts = read_jsonl(prompts_path)
    print(f"[LOAD] outputs={len(outputs)} prompts={len(prompts)}", flush=True)

    accepted: list[dict[str, Any]] = []
    rejected_rows: list[dict[str, Any]] = []
    rejection_reasons = Counter()
    duplicate_seen: Counter[str] = Counter()
    accepted_details: list[dict[str, Any]] = []

    for local_i, rec in enumerate(outputs):
        idx = rec.get("index")
        meta: dict[str, Any]
        if isinstance(idx, int) and 0 <= idx < len(prompts):
            meta = prompts[idx]
        elif rec.get("source_sentence"):
            meta = rec
            idx = local_i
        else:
            rejection_reasons["index_out_of_range"] += 1
            rejected_rows.append({"index": idx, "reasons": ["index_out_of_range"], "record": rec})
            continue
        raw = rec.get("output", rec.get("generated_text", rec.get("generated", "")))
        pair, reasons, detail = validate_one(compact_experience, meta, str(raw), duplicate_seen)
        if pair is None:
            buckets = []
            for r in reasons:
                b = re.sub(r"_(?:0\.\d+|\d+\.\d+|\d+)$", "", r)
                b = re.sub(r"_\d+$", "", b)
                buckets.append(b)
                rejection_reasons[b] += 1
            if len(rejected_rows) < 1000:
                rejected_rows.append({
                    "index": idx,
                    "id": meta.get("id"),
                    "original_id": meta.get("original_id"),
                    "source": meta.get("source_sentence"),
                    "raw_output": raw,
                    "clean_output": detail.get("clean_output"),
                    "reasons": reasons,
                    "reason_buckets": buckets,
                    "detail": detail,
                })
        else:
            pair["local_output_index"] = idx
            accepted.append(pair)
            accepted_details.append(detail)

    accepted_path = out_dir / f"accepted_dose_arm_pairs_{args.tag}.jsonl"
    rejected_path = out_dir / f"rejected_examples_{args.tag}.jsonl"
    write_jsonl(accepted_path, accepted)
    write_jsonl(rejected_path, rejected_rows)

    rng = random.Random(56056)
    sample = accepted[:]
    rng.shuffle(sample)
    sample = sample[:min(args.manual_sample_n, len(sample))]
    sample_path = out_dir / f"accepted_manual_read_sample_{args.tag}.md"
    lines = [f"# research dose-arm accepted manual sample ({args.tag}, n={len(sample)})\n\n"]
    for i, r in enumerate(sample, start=1):
        lines.append(f"## {i}. {r['pair_id']} ({r['source']}; {r['original_words']}→{r['rewrite_words']} words; J={r['content_jaccard']:.3f}; overlap_min={r['content_overlap']:.3f}; contig={r['longest_common_contiguous_words']})\n\n")
        lines.append(f"**Original:** {r['original']}\n\n")
        lines.append(f"**Rewrite:** {r['rewrite']}\n\n")
        if r.get("entity_source") or r.get("num_source"):
            lines.append(f"Entities: {r.get('entity_source')} → {r.get('entity_rewrite')}; Numbers: {r.get('num_source')} → {r.get('num_rewrite')}\n\n")
    sample_path.write_text("".join(lines), encoding="utf-8")

    reg = Counter(r["source"] for r in accepted)
    pair_words = [int(r["pair_words"]) for r in accepted]
    lcs_vals = [int(r["longest_common_contiguous_words"]) for r in accepted]
    j_vals = [float(r["content_jaccard"]) for r in accepted]
    o_vals = [float(r["content_overlap"]) for r in accepted]
    accept_rate = len(accepted) / max(1, len(outputs))

    # Estimate prompts needed for 21% and 25% doses from observed accepted pair words.
    dose_targets = []
    existing_pair_words = 1_656_800
    total_words = 10_000_000
    mean_pair_words = statistics.mean(pair_words) if pair_words else None
    mean_accepted_pair_words_per_prompt = (sum(pair_words) / len(outputs)) if outputs else 0
    for frac in (0.210, 0.250):
        add_words = int(round(frac * total_words)) - existing_pair_words
        if mean_accepted_pair_words_per_prompt > 0:
            prompts_needed = math.ceil(add_words / mean_accepted_pair_words_per_prompt)
        else:
            prompts_needed = None
        if mean_pair_words:
            accepted_pairs_needed = math.ceil(add_words / mean_pair_words)
        else:
            accepted_pairs_needed = None
        dose_targets.append({
            "target_total_pair_fraction": frac,
            "additional_pair_words_needed_per_10M": add_words,
            "observed_mean_accepted_pair_words": round(mean_pair_words, 2) if mean_pair_words else None,
            "observed_accepted_pair_words_per_prompt": round(mean_accepted_pair_words_per_prompt, 2),
            "estimated_accepted_pairs_needed": accepted_pairs_needed,
            "estimated_prompts_needed_at_pilot_yield": prompts_needed,
        })

    meta = {
        "status": "DOSE_ARM_VALIDATED",
        "created_utc": now_utc(),
        "elapsed_sec": round(time.time() - started, 2),
        "tag": args.tag,
        "outputs_path": str(outputs_path),
        "prompts_path": str(prompts_path),
        "outputs_sha256": sha256_file(outputs_path),
        "prompts_sha256": sha256_file(prompts_path),
        "total_outputs": len(outputs),
        "prompt_records": len(prompts),
        "accepted": len(accepted),
        "acceptance_rate": round(accept_rate, 4),
        "rejection_reasons": dict(rejection_reasons.most_common()),
        "accepted_pair_word_total": sum(pair_words),
        "accepted_pair_word_stats": stat([float(x) for x in pair_words]),
        "accepted_content_jaccard_stats": stat(j_vals),
        "accepted_content_overlap_min_stats": stat(o_vals),
        "accepted_contiguous_lcs_stats": stat([float(x) for x in lcs_vals]),
        "register_distribution": dict(reg.most_common()),
        "dose_targets_from_observed_yield": dose_targets,
        "thresholds": {
            "compact_experience_source_and_rewrite_completeness": True,
            "compact_experience_len_ratio": [compact_experience.MIN_LEN_RATIO, compact_experience.MAX_LEN_RATIO],
            "compact_experience_content_overlap_min": [compact_experience.MIN_CONTENT_OVERLAP, compact_experience.MAX_CONTENT_OVERLAP],
            "entity_recall_small": compact_experience.MIN_ENTITY_RECALL_SMALL,
            "entity_recall_large": compact_experience.MIN_ENTITY_RECALL_LARGE,
            "max_content_jaccard": MAX_CONTENT_JACCARD,
            "max_longest_common_contiguous_words": MAX_CONTIGUOUS_LCS,
        },
        "outputs": {
            "accepted_pairs": str(accepted_path),
            "rejected_examples": str(rejected_path),
            "manual_sample": str(sample_path),
        },
        "sha256": {"accepted_pairs": sha256_file(accepted_path), "rejected_examples": sha256_file(rejected_path), "manual_sample": sha256_file(sample_path)},
    }
    meta_path = out_dir / f"validation_metadata_{args.tag}.json"
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(meta, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
