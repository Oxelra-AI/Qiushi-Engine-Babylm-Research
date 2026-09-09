#!/usr/bin/env python3
"""research: provenance-preserving structural compact policy and legal-tail materializer.

The research risk=0 file incorrectly encoded a structural screen as a semantic
`faithful_shortening` label. This script repairs the provenance:

* risk=0/no-tag generated rows are treated as *unverified structural candidates*,
  not as reviewed faithful shortenings;
* demonstrated semantic failures are excluded explicitly;
* exact returns to the source sentence are counted separately as repair/recurrence
  candidates and are not included in the default non-exact shortening policy;
* full original/current/compact texts are written for bounded independent review;
* the legal tail is rebuilt with qwen-pair segment metadata so a real-stream
  trainer can compare inherited WWM with source-visible, view-focused learning on
  the same packed stream.

The default built stream therefore tests a specific practical policy:
`risk0_no_tag_nonexact_structural_shortening + reinvested topup`, with uncertainty
about unreviewed semantic composition preserved in metadata. It does not certify
rows as faithful semantic shortenings.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import hashlib
import json
import random
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

ROOT = _public_path('.')
STUDY = _public_path('experiments/archive/functional_learning')
OUT_ROOT = _public_path('experiments/archive/functional_learning/data/structural_policy_materialization')

REFERENCE_TAIL = _public_path('experiments/archive/functional_learning/data/reference_tail/reference_tail_ordinary_wordpaced.jsonl')
JOINED = _public_path('experiments/archive/functional_learning/data/selective_full_generation/selective_full_joined.jsonl')
SCREEN = _public_path('experiments/archive/functional_learning/data/selective_full_generation/semantic_screen/semantic_screen_pilot.jsonl')
SELECTED_PAIRS = _public_path('experiments/archive/compact_experience/data/qwen_clean_aligned/selected_pairs.jsonl')
PACKED_META = _public_path('experiments/archive/compact_experience/data/qwen_clean_aligned/qwen_pair_packed_rows_meta.jsonl')
TOPUP_ROWS = _public_path('experiments/archive/representation_and_objectives/data/qwen_internal_density_preflight/preferred_len90_r060_affordance_topup_rows.jsonl')
NEUTRAL_ROWS = _public_path('experiments/archive/representation_and_objectives/data/qwen_internal_density_preflight/preferred_len90_r060_neutral_source_matched_rows.jsonl')

# Demonstrated by full-text semantic inspection or earlier known altered rows.
DEMONSTRATED_EXCLUDE = {
    "rw2s0_030370": "sample counterexample: belief/hedging in current/source becomes unqualified assertion in compact",
    "rw2s0_029078": "known altered claim in research semantic screen: navigation order reversed near Goddess Mother temple/idol makers",
    "rw_035551": "known altered claim in research semantic screen: conditional self-attribution shifted toward direct assertion",
}

# Include the previous 30-row sample plus salient examples for full-text review.
SAMPLE = _public_path('experiments/archive/functional_learning/data/risk0_admission/risk0_validation_sample.jsonl')
SALIENT_REVIEW_IDS = [
    "rw2s0_030370",  # demonstrated failure
    "rw2s0_001402",  # exact source return / repair-or-recurrence
    "rw_036172",     # exact source return from sample
    "rw_017177",     # exact source return from sample
    "rw2s0_024236",
    "rw2s1_026885",
]

NUM_RE = re.compile(r"\b\d+(?:[.,:]\d+)*(?:st|nd|rd|th)?\b", re.I)


def rel(p: Path | str) -> str:
    try:
        return str(Path(p).resolve().relative_to(ROOT))
    except Exception:
        return str(p)


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def wc(text: str) -> int:
    return len((text or "").strip().split())


def norm_text(text: str) -> str:
    # Keep punctuation/case mostly intact but normalize whitespace and paired curly quotes.
    t = (text or "").strip()
    t = t.replace("\u201c", '"').replace("\u201d", '"').replace("\u2018", "'").replace("\u2019", "'")
    return re.sub(r"\s+", " ", t)


def sha_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def iter_jsonl(path: Path):
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def load_selected_pairs() -> Dict[str, Dict[str, Any]]:
    return {r["pair_id"]: r for r in iter_jsonl(SELECTED_PAIRS)}


def pair_current_rewrite(pair: Dict[str, Any]) -> str:
    return str(pair.get("rewrite", pair.get("current_rewrite", ""))).strip()


def pair_text(pair: Dict[str, Any], rewrite: Optional[str] = None) -> str:
    rw = pair_current_rewrite(pair) if rewrite is None else str(rewrite).strip()
    return (str(pair.get("original", "")).strip() + (" " + rw if rw else "")).strip()


def build_meta_hash_map(pairs: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    dup = 0
    for rm in iter_jsonl(PACKED_META):
        txt = " ".join(pair_text(pairs[pid]) for pid in rm["pair_ids"] if pid in pairs)
        h = sha_text(txt)
        if h in out:
            dup += 1
        out[h] = {**rm, "canonical_text_sha256": h, "canonical_text_words": wc(txt)}
    if dup:
        print(json.dumps({"event": "warning_duplicate_packed_hash", "duplicates": dup}), file=sys.stderr, flush=True)
    return out


def extract_numbers(text: str) -> set[str]:
    return {m.group(0).lower() for m in NUM_RE.finditer(text or "")}


def numbers_ok(original: str, compact: str) -> bool:
    nums = extract_numbers(original)
    if not nums:
        return True
    low = (compact or "").lower()
    return all(n in low for n in nums)


def load_screen() -> Dict[str, Dict[str, Any]]:
    return {r["pair_id"]: r for r in iter_jsonl(SCREEN)}


def classify_candidate(r: Dict[str, Any], sc: Dict[str, Any], min_words: int) -> Dict[str, Any]:
    pid = r["pair_id"]
    original = str(r.get("original", "")).strip()
    current = str(r.get("current_rewrite", "")).strip()
    compact = str(r.get("compact_rewrite", "")).strip()
    keep = bool(r.get("selective_keep_current", False)) or compact == "KEEP_CURRENT"
    risk = sc.get("risk_score", 999)
    tags = list(sc.get("risk_tags", ["unknown"]) or [])
    cw = wc(current)
    kw = wc(compact)
    ow = wc(original)
    has_saving = bool(compact) and kw < cw and kw >= int(min_words)
    structural = bool(risk == 0 and len(tags) == 0 and not keep and has_saving and numbers_ok(original, compact))
    compact_equals_original = bool(norm_text(compact) == norm_text(original))
    compact_equals_current = bool(norm_text(compact) == norm_text(current))

    if pid in DEMONSTRATED_EXCLUDE:
        kind = "demonstrated_exclude"
        admit = False
        reason = DEMONSTRATED_EXCLUDE[pid]
    elif keep:
        kind = "keep_current"
        admit = False
        reason = "generator chose KEEP_CURRENT"
    elif not structural:
        kind = "not_structural_candidate"
        admit = False
        reason = f"risk={risk} tags={len(tags)} has_saving={has_saving} numbers_ok={numbers_ok(original, compact)}"
    elif compact_equals_current:
        kind = "exact_current_return"
        admit = False
        reason = "compact equals inherited current rewrite after whitespace/quote normalization"
    elif compact_equals_original:
        kind = "exact_source_return_repair_or_recurrence"
        admit = False
        reason = "compact returns exactly to source text; useful as repair/recurrence candidate but not a genuine second-expression shortening"
    else:
        kind = "unverified_structural_shortening_candidate"
        admit = True
        reason = "risk=0/no-tag structural screen only; not independently certified faithful"

    return {
        "pair_id": pid,
        "source": r.get("source", ""),
        "example_id": r.get("example_id"),
        "admit_compact": bool(admit),
        "candidate_kind": kind,
        "semantic_label": "unreviewed_structural_candidate" if admit else kind,
        "semantic_confidence": None,
        "admission_provenance": "selective_generation + automatic structural screen; no full semantic certification",
        "semantic_reason": reason,
        "current_rewrite_words": cw,
        "compact_rewrite_words": kw,
        "original_words": ow,
        "saved_words_unique_pair": max(0, cw - kw) if admit else 0,
        "structural_screen_pass": bool(structural),
        "risk_score": risk,
        "risk_tags": tags,
        "selective_keep_current": keep,
        "compact_equals_original": compact_equals_original,
        "compact_equals_current": compact_equals_current,
        "compact_rewrite": compact,
        "current_rewrite": current,
        "original": original,
        "reject_reasons": [] if admit else [kind, reason],
    }


def build_policy(args: argparse.Namespace) -> Dict[str, Any]:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    screen = load_screen()
    decisions: List[Dict[str, Any]] = []
    for r in iter_jsonl(Path(args.joined)):
        decisions.append(classify_candidate(r, screen.get(r["pair_id"], {}), int(args.min_words)))

    out_path = out_dir / "structural_nonexact_decisions.jsonl"
    write_jsonl(out_path, decisions)

    admitted = [d for d in decisions if d["admit_compact"]]
    kind_counts = Counter(d["candidate_kind"] for d in decisions)
    structural_kinds = Counter(d["candidate_kind"] for d in decisions if d["structural_screen_pass"])
    by_source = Counter(d["source"] for d in admitted)
    exact_source = [d for d in decisions if d["candidate_kind"] == "exact_source_return_repair_or_recurrence"]
    demonstrated = [d for d in decisions if d["candidate_kind"] == "demonstrated_exclude"]

    saved_dist = Counter()
    for d in admitted:
        s = int(d["saved_words_unique_pair"])
        if s <= 3:
            saved_dist["1-3"] += 1
        elif s <= 5:
            saved_dist["4-5"] += 1
        elif s <= 10:
            saved_dist["6-10"] += 1
        elif s <= 15:
            saved_dist["11-15"] += 1
        else:
            saved_dist["16+"] += 1

    # Full-text review packet: previous sample IDs plus salient examples plus random
    # examples from admitted non-exact candidates and exact-source returns.
    review_ids: set[str] = set(SALIENT_REVIEW_IDS)
    if SAMPLE.exists():
        for r in iter_jsonl(SAMPLE):
            if r.get("pair_id"):
                review_ids.add(r["pair_id"])
    rng = random.Random(int(args.review_seed))
    admitted_pool = [d for d in admitted if d["pair_id"] not in review_ids]
    exact_pool = [d for d in exact_source if d["pair_id"] not in review_ids]
    other_struct = [d for d in decisions if d["structural_screen_pass"] and not d["admit_compact"] and d["pair_id"] not in review_ids]
    for d in rng.sample(admitted_pool, min(int(args.extra_review_admitted), len(admitted_pool))):
        review_ids.add(d["pair_id"])
    for d in rng.sample(exact_pool, min(int(args.extra_review_exact_source), len(exact_pool))):
        review_ids.add(d["pair_id"])
    for d in rng.sample(other_struct, min(int(args.extra_review_other_structural), len(other_struct))):
        review_ids.add(d["pair_id"])
    review_rows = [d for d in decisions if d["pair_id"] in review_ids]
    review_rows.sort(key=lambda d: (d["candidate_kind"], d["pair_id"]))
    review_jsonl = out_dir / "bounded_fulltext_review_packet.jsonl"
    write_jsonl(review_jsonl, review_rows)
    review_md = out_dir / "bounded_fulltext_review_packet.md"
    lines = [
        "# research bounded full-text review packet\n",
        "Rows in this file are not semantically certified by inclusion. They are supplied to inspect the composition of the risk=0/no-tag structural policy and salient counterexamples.\n",
        "Labels to use in review: faithful_shortening, supported_partial_view, exact_source_return_repair_or_recurrence, altered_meaning, information_losing_summary, unclear.\n",
    ]
    for i, d in enumerate(review_rows, 1):
        lines.append(f"\n## {i}. {d['pair_id']} ({d['candidate_kind']}, source={d['source']}, saved={d['compact_rewrite_words']} vs current {d['current_rewrite_words']})\n")
        lines.append(f"- risk_score: {d['risk_score']}\n")
        lines.append(f"- risk_tags: {d['risk_tags']}\n")
        lines.append(f"- provenance: {d['admission_provenance']}\n")
        lines.append("\nOriginal/source:\n")
        lines.append(d["original"] + "\n")
        lines.append("\nInherited current rewrite:\n")
        lines.append(d["current_rewrite"] + "\n")
        lines.append("\nGenerated compact:\n")
        lines.append(d["compact_rewrite"] + "\n")
    review_md.write_text("".join(lines), encoding="utf-8")

    summary = {
        "status": "STRUCTURAL_POLICY_READY",
        "created_utc": now(),
        "joined": rel(Path(args.joined)),
        "screen": rel(SCREEN),
        "n_pairs": len(decisions),
        "candidate_kind_counts": dict(kind_counts),
        "structural_screen_pass_kind_counts": dict(structural_kinds),
        "n_admitted_nonexact_unverified_structural_shortening": len(admitted),
        "unique_pair_saved_words_admitted_nonexact": sum(int(d["saved_words_unique_pair"]) for d in admitted),
        "mean_saved_words_admitted_nonexact": (sum(int(d["saved_words_unique_pair"]) for d in admitted) / len(admitted)) if admitted else 0.0,
        "admitted_by_source": dict(by_source.most_common()),
        "admitted_savings_distribution": dict(sorted(saved_dist.items())),
        "exact_source_return_count_not_admitted": len(exact_source),
        "exact_source_return_saved_words_if_used": sum(max(0, int(d["current_rewrite_words"]) - int(d["compact_rewrite_words"])) for d in exact_source),
        "demonstrated_exclusions": [{"pair_id": d["pair_id"], "reason": d["semantic_reason"]} for d in demonstrated],
        "decisions": rel(out_path),
        "bounded_fulltext_review_jsonl": rel(review_jsonl),
        "bounded_fulltext_review_md": rel(review_md),
        "review_packet_rows": len(review_rows),
        "policy_interpretation": "admit_compact=True means structurally filtered, non-exact generated shortening candidate; it is not a semantic faithful_shortening label. Exact source returns and demonstrated failures are separated.",
    }
    (out_dir / "structural_policy_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    return summary


def load_decisions(path: Path) -> Dict[str, Dict[str, Any]]:
    return {r["pair_id"]: r for r in iter_jsonl(path)}


def build_pair_piece(pair: Dict[str, Any], decision: Optional[Dict[str, Any]]) -> Tuple[str, Dict[str, Any], int]:
    original = str(pair.get("original", "")).strip()
    if decision and decision.get("admit_compact"):
        view = str(decision.get("compact_rewrite", "")).strip()
        view_kind = "compact_structural_nonexact"
        candidate_kind = decision.get("candidate_kind", "unverified_structural_shortening_candidate")
    else:
        view = pair_current_rewrite(pair)
        view_kind = "current_inherited"
        candidate_kind = decision.get("candidate_kind", "not_in_policy") if decision else "not_in_policy"
    piece = (original + (" " + view if view else "")).strip()
    saved = max(0, wc(pair_current_rewrite(pair)) - wc(view)) if decision and decision.get("admit_compact") else 0
    segment = {
        "pair_id": pair["pair_id"],
        "source_text": original,
        "view_text": view,
        "view_kind": view_kind,
        "candidate_kind": candidate_kind,
        "source_start": 0,
        "source_end": len(original),
        "view_start": len(original) + (1 if view else 0),
        "view_end": len(piece),
        "source_words": wc(original),
        "view_words": wc(view),
        "current_view_words": wc(pair_current_rewrite(pair)),
        "saved_words": saved,
    }
    return piece, segment, saved


def compacted_text_segments_for_meta(rm: Dict[str, Any], pairs: Dict[str, Dict[str, Any]], decisions: Dict[str, Dict[str, Any]]) -> Tuple[str, int, List[str], List[Dict[str, Any]], Counter]:
    parts: List[str] = []
    segments: List[Dict[str, Any]] = []
    saved = 0
    touched: List[str] = []
    kinds: Counter = Counter()
    cursor = 0
    for pid in rm.get("pair_ids", []):
        if pid not in pairs:
            continue
        pair = dict(pairs[pid])
        pair["pair_id"] = pid
        d = decisions.get(pid)
        piece, seg, s = build_pair_piece(pair, d)
        if parts:
            cursor += 1  # separating space inserted by join below
        seg = dict(seg)
        for key in ["source_start", "source_end", "view_start", "view_end"]:
            seg[key] += cursor
        parts.append(piece)
        cursor += len(piece)
        segments.append(seg)
        kinds[str(seg.get("candidate_kind", "unknown"))] += 1
        if d and d.get("admit_compact"):
            saved += int(s)
            touched.append(pid)
    return " ".join(parts), saved, touched, segments, kinds


def build_tail(args: argparse.Namespace) -> Dict[str, Any]:
    out_dir = Path(args.out_dir) / args.arm
    out_dir.mkdir(parents=True, exist_ok=True)
    pairs = load_selected_pairs()
    decisions = load_decisions(Path(args.decisions))
    meta_by_hash = build_meta_hash_map(pairs)

    out_rows: List[Dict[str, Any]] = []
    match_stats = Counter()
    saved_total = 0
    rows_modified = 0
    pair_occurrences = 0
    qwen_rows_with_segments = 0
    row_kind_counts = Counter()
    modified_occ_by_pair = Counter()
    ref_words = 0

    for line_i, row0 in enumerate(iter_jsonl(REFERENCE_TAIL)):
        row = dict(row0)
        ref_words += int(row.get("words", wc(row.get("text", ""))))
        if row.get("source") == "qwen_pair_packed":
            match_stats["qwen_rows"] += 1
            rm = meta_by_hash.get(sha_text(str(row.get("text", ""))))
            if not rm:
                match_stats["missing_qwen_rows"] += 1
            else:
                match_stats["matched_qwen_rows"] += 1
                new_text, saved, touched, segments, kinds = compacted_text_segments_for_meta(rm, pairs, decisions)
                old_words = int(row.get("words", wc(row.get("text", ""))))
                new_words = wc(new_text)
                # Even unchanged qwen rows receive segment metadata for correspondence-focused training.
                row["text"] = new_text
                row["words"] = new_words
                row["qwen_pair_ids"] = list(rm.get("pair_ids", []))
                row["qwen_pair_segments"] = segments
                row["qwen_segment_policy"] = "source-visible view-focused targetable metadata; semantic status follows per-pair candidate_kind"
                qwen_rows_with_segments += 1
                row_kind_counts.update(kinds)
                if touched:
                    saved_by_wc = old_words - new_words
                    row["compact_modified_pair_ids"] = touched
                    row["compact_saved_words"] = saved_by_wc
                    row["compact_admission_policy"] = "risk0_no_tag_nonexact_structural_candidate_unverified; exact_source_returns_excluded"
                    row["compact_policy_counts"] = dict(kinds)
                    rows_modified += 1
                    pair_occurrences += len(touched)
                    saved_total += saved_by_wc
                    for pid in touched:
                        modified_occ_by_pair[pid] += 1
        row["reference_tail_position"] = row.get("reference_tail_position", line_i)
        out_rows.append(row)

    topup_source = None
    topup_kind = None
    if args.arm == "compact_structural_reinvest":
        topup_source = TOPUP_ROWS
        topup_kind = "structural_compact_affordance_topup"
    elif args.arm == "compact_structural_neutral":
        topup_source = NEUTRAL_ROWS
        topup_kind = "structural_compact_neutral_topup"
    elif args.arm == "compact_structural_unspent":
        pass
    else:
        raise ValueError(args.arm)

    topup_rows_added = 0
    topup_words_added = 0
    if topup_source:
        budget = saved_total
        for tr in iter_jsonl(topup_source):
            tw = int(tr.get("words", wc(tr.get("text", ""))))
            if topup_words_added + tw > budget:
                continue
            new = dict(tr)
            new["bridge_kind"] = "ordinary_tail"
            new["source"] = f"{topup_kind}::{tr.get('source', 'unknown')}"
            new["structural_compact_topup"] = True
            new["orig_row_index"] = f"{topup_kind}_{topup_rows_added:06d}"
            new["presentation_source_index"] = topup_rows_added
            new["reference_tail_position"] = len(out_rows)
            new["words"] = tw
            out_rows.append(new)
            topup_rows_added += 1
            topup_words_added += tw
            if topup_words_added >= budget:
                break

    out_jsonl = out_dir / f"{args.arm}_reference_tail_wordpaced_segments.jsonl"
    write_jsonl(out_jsonl, out_rows)
    h = hashlib.sha256()
    with out_jsonl.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    total_words = sum(int(r.get("words", wc(r.get("text", "")))) for r in out_rows)
    src_rows = Counter(r.get("source", "") for r in out_rows)
    src_words = Counter()
    for r in out_rows:
        src_words[r.get("source", "")] += int(r.get("words", wc(r.get("text", ""))))

    summary = {
        "status": "STRUCTURAL_TAIL_BUILT",
        "created_utc": now(),
        "arm": args.arm,
        "reference_tail": rel(REFERENCE_TAIL),
        "decisions": rel(Path(args.decisions)),
        "rows_written": len(out_rows),
        "total_words": total_words,
        "reference_tail_words": ref_words,
        "net_word_change_vs_reference_tail": total_words - ref_words,
        "legal_total_words_if_from_coherent86": 86005295 + total_words,
        "qwen_rows": int(match_stats["qwen_rows"]),
        "matched_qwen_rows": int(match_stats["matched_qwen_rows"]),
        "missing_qwen_rows": int(match_stats["missing_qwen_rows"]),
        "qwen_rows_with_segments": qwen_rows_with_segments,
        "rows_modified": rows_modified,
        "pair_occurrences_compacted": pair_occurrences,
        "words_saved_by_compaction": saved_total,
        "modified_pair_occurrence_stats": {
            "n_pairs_seen_in_tail": len(modified_occ_by_pair),
            "min": min(modified_occ_by_pair.values()) if modified_occ_by_pair else 0,
            "max": max(modified_occ_by_pair.values()) if modified_occ_by_pair else 0,
            "mean": (sum(modified_occ_by_pair.values()) / len(modified_occ_by_pair)) if modified_occ_by_pair else 0.0,
        },
        "qwen_segment_candidate_kind_counts": dict(row_kind_counts),
        "topup_rows_added": topup_rows_added,
        "topup_words_added": topup_words_added,
        "unspent_saved_words": saved_total - topup_words_added,
        "source_rows": dict(src_rows),
        "source_words": dict(src_words),
        "output_jsonl": rel(out_jsonl),
        "sha256": h.hexdigest(),
        "interpretation": "Legal-tail stream for testing a structurally filtered, non-exact generated shortening policy. It carries qwen segment metadata for correspondence-focused training. Admitted rows are not semantic faithful labels; exact source returns and demonstrated failures were excluded from compaction.",
    }
    (out_dir / "tail_build_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    return summary


def inspect_prefix(args: argparse.Namespace) -> Dict[str, Any]:
    rows = []
    words = 0
    compact_rows = 0
    compact_occ = 0
    qwen_rows = 0
    qwen_targets_possible = 0
    max_words = int(args.words_per_update) * int(args.max_updates)
    for row in iter_jsonl(Path(args.tail_jsonl)):
        if words >= max_words:
            break
        rows.append(row)
        w = int(row.get("words", wc(row.get("text", ""))))
        words += w
        if row.get("source") == "qwen_pair_packed":
            qwen_rows += 1
            qwen_targets_possible += sum(1 for seg in row.get("qwen_pair_segments", []) if seg.get("view_text"))
        if row.get("compact_modified_pair_ids"):
            compact_rows += 1
            compact_occ += len(row.get("compact_modified_pair_ids") or [])
    summary = {
        "status": "PREFIX_INSPECTED",
        "tail_jsonl": rel(Path(args.tail_jsonl)),
        "max_updates": int(args.max_updates),
        "words_per_update": int(args.words_per_update),
        "prefix_rows": len(rows),
        "prefix_words": words,
        "qwen_rows": qwen_rows,
        "qwen_pair_segments": qwen_targets_possible,
        "compact_modified_rows": compact_rows,
        "compact_modified_pair_occurrences": compact_occ,
        "interpretation": "This is the initial packed legal-stream slice used to compare ordinary WWM against correspondence-focused learning under realistic row order and competing ordinary experience.",
    }
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"prefix_inspect_{args.max_updates:04d}updates.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("policy")
    p.add_argument("--joined", default=str(JOINED))
    p.add_argument("--out-dir", default=str(OUT_ROOT))
    p.add_argument("--min-words", type=int, default=4)
    p.add_argument("--review-seed", type=int, default=62062)
    p.add_argument("--extra-review-admitted", type=int, default=40)
    p.add_argument("--extra-review-exact-source", type=int, default=12)
    p.add_argument("--extra-review-other-structural", type=int, default=8)

    b = sub.add_parser("build-tail")
    b.add_argument("--decisions", default=str(_public_path('experiments/archive/functional_learning/data/structural_policy_materialization/structural_nonexact_decisions.jsonl')))
    b.add_argument("--out-dir", default=str(OUT_ROOT))
    b.add_argument("--arm", choices=["compact_structural_reinvest", "compact_structural_unspent", "compact_structural_neutral"], default="compact_structural_reinvest")

    i = sub.add_parser("inspect-prefix")
    i.add_argument("--tail-jsonl", required=True)
    i.add_argument("--out-dir", default=str(OUT_ROOT))
    i.add_argument("--max-updates", type=int, default=80)
    i.add_argument("--words-per-update", type=int, default=39533)

    args = ap.parse_args()
    if args.cmd == "policy":
        build_policy(args)
    elif args.cmd == "build-tail":
        build_tail(args)
    elif args.cmd == "inspect-prefix":
        inspect_prefix(args)


if __name__ == "__main__":
    main()
