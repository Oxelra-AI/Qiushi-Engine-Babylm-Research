#!/usr/bin/env python3
"""research: semantic-admission materializer for Qwen-internal compact tail.

This replaces research's draft materializer for the practical route.  The core repair is
semantic admission: compact outputs are candidate rewrites, not accepted data.  A pair
is compacted only when an explicit semantic label says faithful_shortening; otherwise
the inherited rewrite is retained in every arm.  The materializer operates on the
actual coherent86->100M reference tail, not on the earlier 10M pool, and maps repeated
qwen_pair_packed tail rows back to their selected-pair metadata by exact row-text hash.

Subcommands:
  join-full          Join full generation outputs to the compaction manifest by index.
  admit-from-labels  Convert semantic labels/judge outputs to pair admission decisions.
  audit-tail         Compute realized tail savings under an admission file.
  build-tail         Build compact_unspent, compact_reinvest, or compact_neutral tails.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

ROOT = _public_path('.')
STUDY = _public_path('experiments/archive/functional_learning')
OUT_ROOT = _public_path('experiments/archive/functional_learning/data/compact_tail_materialization')

REFERENCE_TAIL = _public_path('experiments/archive/functional_learning/data/reference_tail/reference_tail_ordinary_wordpaced.jsonl')
FULL_GEN_INPUT = _public_path('experiments/archive/functional_learning/data/compact_pilot/full_generation_input_26567.jsonl')
FULL_GEN_OUTPUT = _public_path('experiments/archive/functional_learning/data/compact_pilot/full_generation_output_26567.jsonl')
COMPACTION_MANIFEST = _public_path('experiments/archive/representation_and_objectives/data/qwen_internal_density_preflight/preferred_len90_r060_compaction_manifest.jsonl')
SELECTED_PAIRS = _public_path('experiments/archive/compact_experience/data/qwen_clean_aligned/selected_pairs.jsonl')
PACKED_META = _public_path('experiments/archive/compact_experience/data/qwen_clean_aligned/qwen_pair_packed_rows_meta.jsonl')
TOPUP_ROWS = _public_path('experiments/archive/representation_and_objectives/data/qwen_internal_density_preflight/preferred_len90_r060_affordance_topup_rows.jsonl')
NEUTRAL_ROWS = _public_path('experiments/archive/representation_and_objectives/data/qwen_internal_density_preflight/preferred_len90_r060_neutral_source_matched_rows.jsonl')

NUM_RE = re.compile(r"\b\d+(?:[.,:]\d+)*(?:st|nd|rd|th)?\b", re.I)
FAITHFUL = {"faithful_shortening", "faithful", "faithful_compression"}
NONFAITHFUL_ALIASES = {
    "supported_summary": "supported_summary_with_lost_detail",
    "information_losing": "supported_summary_with_lost_detail",
    "unclear_or_information_losing": "supported_summary_with_lost_detail",
    "unsafe": "unclear",
    "unclear_or_unsafe_source": "unclear",
    "changed_meaning": "altered_meaning",
    "altered": "altered_meaning",
}


def rel(p: Path | str) -> str:
    try:
        return str(Path(p).resolve().relative_to(ROOT))
    except Exception:
        return str(p)


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def wc(text: str) -> int:
    return len((text or "").strip().split())


def sha_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_selected_pairs() -> Dict[str, Dict[str, Any]]:
    return {r["pair_id"]: r for r in load_jsonl(SELECTED_PAIRS)}


def pair_current_rewrite(pair: Dict[str, Any]) -> str:
    return str(pair.get("rewrite", pair.get("current_rewrite", ""))).strip()


def pair_text(pair: Dict[str, Any], rewrite: Optional[str] = None) -> str:
    rw = pair_current_rewrite(pair) if rewrite is None else str(rewrite).strip()
    return (str(pair.get("original", "")).strip() + (" " + rw if rw else "")).strip()


def build_meta_hash_map(pairs: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    dup = 0
    for rm in load_jsonl(PACKED_META):
        txt = " ".join(pair_text(pairs[pid]) for pid in rm["pair_ids"] if pid in pairs)
        h = sha_text(txt)
        if h in out:
            dup += 1
        out[h] = {**rm, "canonical_text_sha256": h, "canonical_text_words": wc(txt)}
    if dup:
        print(json.dumps({"event": "warning_duplicate_packed_hash", "duplicates": dup}), file=sys.stderr, flush=True)
    return out


def clean_generation_output(text: str) -> str:
    text = (text or "").strip()
    if text.startswith('"') and text.endswith('"') and len(text) >= 2:
        text = text[1:-1]
    # Remove common assistant preambles if a generator ignored the instruction.
    prefixes = ["Here is the rewritten sentence:", "Rewritten sentence:", "Sentence:"]
    for p in prefixes:
        if text.lower().startswith(p.lower()):
            text = text[len(p):].strip()
    return text


def join_full(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = load_jsonl(Path(args.manifest))
    gen_input = load_jsonl(Path(args.generation_input)) if Path(args.generation_input).exists() else []
    gen = load_jsonl(Path(args.generation_output))
    manifest_by_pid = {r["pair_id"]: r for r in manifest}
    input_by_idx = {i: r for i, r in enumerate(gen_input)}
    joined = []
    missing = 0
    for j, g in enumerate(gen):
        idx = int(g.get("index", j))
        pid = g.get("pair_id") or input_by_idx.get(idx, {}).get("pair_id")
        m = manifest_by_pid.get(pid)
        if not m:
            missing += 1
            m = {"pair_id": pid or f"idx_{idx}"}
        compact = clean_generation_output(str(g.get("output", g.get("text", g.get("completion", "")))))
        joined.append({
            "pair_id": m.get("pair_id"),
            "source": m.get("source", ""),
            "example_id": m.get("example_id"),
            "original": m.get("original", ""),
            "current_rewrite": m.get("current_rewrite", m.get("rewrite", "")),
            "compact_rewrite": compact,
            "target_compact_words": m.get("target_compact_rewrite_words", 0),
            "expected_saved_words": m.get("expected_saved_words", 0),
            "entity_source": m.get("entity_source", []),
            "num_source": m.get("num_source", []),
            "content_overlap": m.get("content_overlap", 0),
            "generation_index": idx,
            "raw_model": g.get("model"),
        })
    out_path = out_dir / "full_generation_joined_26567.jsonl"
    write_jsonl(out_path, joined)
    summary = {
        "status": "FULL_GENERATION_JOINED",
        "n_generation_rows": len(gen),
        "n_joined": len(joined),
        "n_manifest_missing": missing,
        "generation_output": rel(Path(args.generation_output)),
        "output": rel(out_path),
    }
    (out_dir / "full_generation_join_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


def normalize_label(label: str) -> str:
    lab = (label or "").strip().lower().replace(" ", "_").replace("-", "_")
    return NONFAITHFUL_ALIASES.get(lab, lab)


def extract_numbers(text: str) -> set[str]:
    return {m.group(0).lower() for m in NUM_RE.finditer(text or "")}


def numbers_ok(original: str, compact: str) -> bool:
    nums = extract_numbers(original)
    if not nums:
        return True
    low = (compact or "").lower()
    return all(n in low for n in nums)


def load_label_file(path: Path) -> Dict[str, Dict[str, Any]]:
    labels: Dict[str, Dict[str, Any]] = {}
    for r in load_jsonl(path):
        pid = r.get("pair_id")
        if not pid:
            continue
        label = normalize_label(str(r.get("judge_label", r.get("label", ""))))
        confidence = r.get("judge_confidence", r.get("confidence", None))
        try:
            confidence = float(confidence) if confidence is not None else None
        except Exception:
            confidence = None
        labels[pid] = {
            "semantic_label": label,
            "semantic_confidence": confidence,
            "semantic_reason": r.get("judge_reason", r.get("note", r.get("brief_reason", ""))),
            "semantic_source": str(path),
            "raw_label_row": r,
        }
    return labels


def admit_from_labels(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    joined = load_jsonl(Path(args.joined_outputs))
    labels = load_label_file(Path(args.labels))
    decisions = []
    for r in joined:
        pid = r["pair_id"]
        lab = labels.get(pid, {})
        label = lab.get("semantic_label", "unlabeled")
        conf = lab.get("semantic_confidence")
        curr = str(r.get("current_rewrite", "")).strip()
        comp = str(r.get("compact_rewrite", "")).strip()
        current_words = wc(curr)
        compact_words = wc(comp)
        basic_ok = bool(comp) and compact_words >= int(args.min_words) and compact_words < current_words and numbers_ok(str(r.get("original", "")), comp)
        label_ok = label in FAITHFUL
        conf_ok = (conf is None) or (conf >= float(args.min_confidence))
        admit = bool(label_ok and conf_ok and basic_ok)
        reason = []
        if not label_ok:
            reason.append(f"semantic_label={label}")
        if not conf_ok:
            reason.append(f"confidence={conf} < {args.min_confidence}")
        if not comp:
            reason.append("empty_compact")
        if compact_words < int(args.min_words):
            reason.append(f"compact_words={compact_words} < {args.min_words}")
        if compact_words >= current_words:
            reason.append(f"compact_words={compact_words} >= current_words={current_words}")
        if not numbers_ok(str(r.get("original", "")), comp):
            reason.append("number_string_missing_from_compact")
        decisions.append({
            "pair_id": pid,
            "source": r.get("source", ""),
            "example_id": r.get("example_id"),
            "admit_compact": admit,
            "semantic_label": label,
            "semantic_confidence": conf,
            "semantic_reason": lab.get("semantic_reason", ""),
            "current_rewrite_words": current_words,
            "compact_rewrite_words": compact_words,
            "saved_words_unique_pair": max(0, current_words - compact_words) if admit else 0,
            "reject_reasons": reason,
            "compact_rewrite": comp,
            "current_rewrite": curr,
            "original": r.get("original", ""),
        })
    out_path = out_dir / args.output_name
    write_jsonl(out_path, decisions)
    admitted = [d for d in decisions if d["admit_compact"]]
    by_label = Counter(d["semantic_label"] for d in decisions)
    summary = {
        "status": "ADMISSION_DECISIONS_BUILT",
        "n_joined_outputs": len(joined),
        "n_label_rows": len(labels),
        "n_admitted_unique_pairs": len(admitted),
        "unique_pair_saved_words": sum(d["saved_words_unique_pair"] for d in admitted),
        "label_counts": dict(by_label),
        "min_confidence": float(args.min_confidence),
        "output": rel(out_path),
        "interpretation": "Only semantically labeled faithful_shortening rows are admitted; every other compact candidate keeps the inherited rewrite in matched arms.",
    }
    (out_dir / (Path(args.output_name).stem + "_summary.json")).write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


def load_decisions(path: Path) -> Dict[str, Dict[str, Any]]:
    return {r["pair_id"]: r for r in load_jsonl(path)}


def tail_rows_with_meta(pairs: Dict[str, Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    meta_by_hash = build_meta_hash_map(pairs)
    rows = []
    qwen_rows = 0
    matched = 0
    for line_i, row in enumerate(load_jsonl(REFERENCE_TAIL)):
        new = dict(row)
        new["tail_line_index"] = line_i
        if row.get("source") == "qwen_pair_packed":
            qwen_rows += 1
            rm = meta_by_hash.get(sha_text(row.get("text", "")))
            if rm:
                matched += 1
                new["packed_meta"] = rm
            else:
                new["packed_meta_missing"] = True
        rows.append(new)
    return rows, {"qwen_rows": qwen_rows, "matched_qwen_rows": matched, "missing_qwen_rows": qwen_rows - matched}


def compacted_text_for_meta(rm: Dict[str, Any], pairs: Dict[str, Dict[str, Any]], decisions: Dict[str, Dict[str, Any]]) -> Tuple[str, int, List[str]]:
    parts = []
    saved = 0
    touched = []
    for pid in rm.get("pair_ids", []):
        pair = pairs[pid]
        d = decisions.get(pid)
        if d and d.get("admit_compact"):
            rw = d.get("compact_rewrite", "")
            saved += max(0, wc(pair_current_rewrite(pair)) - wc(rw))
            touched.append(pid)
        else:
            rw = pair_current_rewrite(pair)
        parts.append(pair_text(pair, rw))
    text = " ".join(parts)
    return text, saved, touched


def audit_tail(args: argparse.Namespace) -> Dict[str, Any]:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pairs = load_selected_pairs()
    decisions = load_decisions(Path(args.decisions))
    tail_rows, match_stats = tail_rows_with_meta(pairs)
    unique_admitted = {pid for pid, d in decisions.items() if d.get("admit_compact")}
    touched_occurrences = 0
    saved_occurrences = 0
    row_touches = 0
    rows_by_source = Counter()
    admitted_occ_by_pair = Counter()
    for row in tail_rows:
        rm = row.get("packed_meta")
        if not rm:
            continue
        text, saved, touched = compacted_text_for_meta(rm, pairs, decisions)
        if touched:
            row_touches += 1
            touched_occurrences += len(touched)
            saved_occurrences += saved
            rows_by_source[row.get("source", "")] += 1
            for pid in touched:
                admitted_occ_by_pair[pid] += 1
    summary = {
        "status": "COMPACT_TAIL_AUDITED",
        "reference_tail": rel(REFERENCE_TAIL),
        **match_stats,
        "n_unique_admitted_pairs": len(unique_admitted),
        "n_tail_pair_occurrences_touched": touched_occurrences,
        "n_tail_rows_touched": row_touches,
        "realized_tail_saved_words": saved_occurrences,
        "topup_full_rows_possible_160w_approx": saved_occurrences // 160,
        "rows_by_source": dict(rows_by_source),
        "admitted_pair_occurrence_stats": {
            "n_pairs_seen_in_tail": len(admitted_occ_by_pair),
            "min": min(admitted_occ_by_pair.values()) if admitted_occ_by_pair else 0,
            "max": max(admitted_occ_by_pair.values()) if admitted_occ_by_pair else 0,
            "mean": sum(admitted_occ_by_pair.values()) / max(1, len(admitted_occ_by_pair)),
        },
        "decisions": rel(Path(args.decisions)),
    }
    out_path = out_dir / "tail_audit_summary.json"
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    return summary


def build_tail(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir) / args.arm
    out_dir.mkdir(parents=True, exist_ok=True)
    pairs = load_selected_pairs()
    decisions = load_decisions(Path(args.decisions))
    tail_rows, match_stats = tail_rows_with_meta(pairs)
    arm = args.arm
    if arm not in {"compact_unspent", "compact_reinvest", "compact_neutral"}:
        raise ValueError(arm)

    out_rows = []
    saved_total = 0
    rows_modified = 0
    pair_occurrences = 0
    for row in tail_rows:
        row = dict(row)
        rm = row.pop("packed_meta", None)
        row.pop("tail_line_index", None)
        row.pop("packed_meta_missing", None)
        if rm:
            new_text, saved, touched = compacted_text_for_meta(rm, pairs, decisions)
            if touched:
                old_words = int(row.get("words", wc(row.get("text", ""))))
                new_words = wc(new_text)
                # Saved by text word count is authoritative for training.
                saved_by_wc = old_words - new_words
                row["text"] = new_text
                row["words"] = new_words
                row["compact_modified_pair_ids"] = touched
                row["compact_saved_words"] = saved_by_wc
                row["compact_admission_policy"] = "semantic_label_faithful_shortening_only"
                rows_modified += 1
                pair_occurrences += len(touched)
                saved_total += saved_by_wc
        out_rows.append(row)

    topup_source = None
    topup_kind = None
    if arm == "compact_reinvest":
        topup_source = TOPUP_ROWS
        topup_kind = "semantic_compact_affordance_topup"
    elif arm == "compact_neutral":
        topup_source = NEUTRAL_ROWS
        topup_kind = "semantic_compact_neutral_topup"

    topup_rows_added = 0
    topup_words_added = 0
    if topup_source:
        budget = saved_total
        for tr in load_jsonl(topup_source):
            tw = int(tr.get("words", wc(tr.get("text", ""))))
            if topup_words_added + tw > budget:
                continue
            new = dict(tr)
            new["bridge_kind"] = "ordinary_tail"
            new["source"] = f"{topup_kind}::{tr.get('source', 'unknown')}"
            new["semantic_compact_topup"] = True
            # Give top-up rows stable unique ordinary_tail keys for deterministic WWM.
            # The research trainer keys ordinary rows by (orig_row_index, example_id,
            # presentation_source_index); inherited tail rows already have orig_row_index,
            # but the top-up files do not.
            new["orig_row_index"] = f"{topup_kind}_{topup_rows_added:06d}"
            new["presentation_source_index"] = topup_rows_added
            new["reference_tail_position"] = len(out_rows)
            new["words"] = tw
            out_rows.append(new)
            topup_rows_added += 1
            topup_words_added += tw
            if topup_words_added >= budget:
                break

    out_jsonl = out_dir / f"{arm}_reference_tail_wordpaced.jsonl"
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
    ref_words = sum(int(r.get("words", wc(r.get("text", "")))) for r in load_jsonl(REFERENCE_TAIL))
    summary = {
        "status": "COMPACT_TAIL_BUILT",
        "arm": arm,
        "reference_tail": rel(REFERENCE_TAIL),
        "decisions": rel(Path(args.decisions)),
        **match_stats,
        "rows_written": len(out_rows),
        "total_words": total_words,
        "reference_tail_words": ref_words,
        "net_word_change_vs_reference_tail": total_words - ref_words,
        "rows_modified": rows_modified,
        "pair_occurrences_compacted": pair_occurrences,
        "words_saved_by_compaction": saved_total,
        "topup_rows_added": topup_rows_added,
        "topup_words_added": topup_words_added,
        "unspent_saved_words": saved_total - topup_words_added,
        "source_rows": dict(src_rows),
        "source_words": dict(src_words),
        "output_jsonl": rel(out_jsonl),
        "sha256": h.hexdigest(),
        "legal_total_words_if_from_coherent86": 86005295 + total_words,
        "interpretation": "Rows without admitted faithful compact labels retain the inherited rewrite; reinvestment uses only realized saved words and keeps total words under the BabyLM cap.",
    }
    (out_dir / "tail_build_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd")

    p = sub.add_parser("join-full")
    p.add_argument("--generation-output", default=str(FULL_GEN_OUTPUT))
    p.add_argument("--generation-input", default=str(FULL_GEN_INPUT))
    p.add_argument("--manifest", default=str(COMPACTION_MANIFEST))
    p.add_argument("--out-dir", default=str(OUT_ROOT))

    a = sub.add_parser("admit-from-labels")
    a.add_argument("--joined-outputs", required=True)
    a.add_argument("--labels", required=True)
    a.add_argument("--out-dir", default=str(OUT_ROOT))
    a.add_argument("--output-name", default="admission_decisions.jsonl")
    a.add_argument("--min-confidence", type=float, default=0.70)
    a.add_argument("--min-words", type=int, default=8)

    au = sub.add_parser("audit-tail")
    au.add_argument("--decisions", required=True)
    au.add_argument("--out-dir", default=str(OUT_ROOT))

    b = sub.add_parser("build-tail")
    b.add_argument("--decisions", required=True)
    b.add_argument("--arm", choices=["compact_unspent", "compact_reinvest", "compact_neutral"], required=True)
    b.add_argument("--out-dir", default=str(OUT_ROOT))

    args = ap.parse_args()
    if args.cmd == "join-full":
        join_full(args)
    elif args.cmd == "admit-from-labels":
        admit_from_labels(args)
    elif args.cmd == "audit-tail":
        audit_tail(args)
    elif args.cmd == "build-tail":
        build_tail(args)
    else:
        ap.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
