#!/usr/bin/env python3
"""research: Matched corpus materializer for compact-view/reinvest comparison.

Takes actual compact rewrites, rebuilds qwen_pair_packed rows with shorter rewrites,
accounts saved words, and materializes matched training corpora for a clean comparison.

Arms:
  baseline:     Current Qwen pool (coherent86 substrate). No changes.
  compact_only: Same pool but targeted rewrites shortened. Total words decrease.
  compact_reinvest: Compact + saved words spent on top-up official rows (new source).
  compact_neutral:  Compact + saved words spent on neutral source-matched rows (control).

The materializer rebuilds pair rows from individual pairs rather than text replacement,
which avoids fragile substring matching in concatenated text.

Usage:
    python matched_corpus_materializer.py audit --compact-outputs COMPACT.jsonl
    python matched_corpus_materializer.py build --compact-outputs COMPACT.jsonl --arm compact_reinvest
"""
import argparse, json, sys, os, hashlib, statistics
from pathlib import Path
from collections import defaultdict, Counter

STUDY = Path("experiments/archive/functional_learning")
DATA = STUDY / "data" / "matched_corpus"
PILOT_DATA = STUDY / "data" / "compact_pilot"

# Source files
SELECTED_PAIRS = Path("experiments/archive/compact_experience/data/qwen_clean_aligned/selected_pairs.jsonl")
PACKED_ROWS_META = Path("experiments/archive/compact_experience/data/qwen_clean_aligned/qwen_pair_packed_rows_meta.jsonl")
QWEN_POOL = Path("experiments/archive/compact_experience/data/qwen_clean_aligned/training_corpora/qwen_aligned_10M.jsonl")
TOPUP_ROWS = Path("experiments/archive/representation_and_objectives/data/qwen_internal_density_preflight/preferred_len90_r060_affordance_topup_rows.jsonl")
NEUTRAL_ROWS = Path("experiments/archive/representation_and_objectives/data/qwen_internal_density_preflight/preferred_len90_r060_neutral_source_matched_rows.jsonl")


def load_selected_pairs():
    """Load all selected source/rewrite pairs."""
    pairs = {}
    with open(SELECTED_PAIRS) as f:
        for line in f:
            rec = json.loads(line)
            pairs[rec["pair_id"]] = rec
    return pairs


def load_packed_row_meta():
    """Load the packed row metadata (pair_ids per row)."""
    rows = []
    with open(PACKED_ROWS_META) as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def load_compact_outputs(path):
    """Load compact rewrite outputs and index by pair_id."""
    compact = {}
    with open(path) as f:
        for line in f:
            rec = json.loads(line)
            pid = rec.get("pair_id", "")
            text = rec.get("compact_rewrite", rec.get("output", rec.get("text", "")))
            if isinstance(text, str):
                text = text.strip()
                if text.startswith('"') and text.endswith('"'):
                    text = text[1:-1]
            compact[pid] = text
    return compact


def word_count(text):
    return len(text.strip().split())


def rebuild_pair_text(pair, compact_text=None):
    """Rebuild text for one pair: source + (compact or current) rewrite."""
    source = pair["original"]
    rewrite = compact_text if compact_text else pair.get("rewrite", pair.get("current_rewrite", ""))
    # Clean rewrite: ensure no leading/trailing junk
    rewrite = rewrite.strip() if rewrite else ""
    return f"{source} {rewrite}" if rewrite else source


def audit(args):
    """Audit compact outputs for quality-filtered feasibility."""
    DATA.mkdir(parents=True, exist_ok=True)

    # Load data
    pairs = load_selected_pairs()
    packed_meta = load_packed_row_meta()
    compact = load_compact_outputs(args.compact_outputs)

    print(f"Selected pairs: {len(pairs)}", flush=True)
    print(f"Packed rows: {len(packed_meta)}", flush=True)
    print(f"Compact outputs: {len(compact)}", flush=True)

    # Map pair_id -> row_index for affected rows
    pair_to_row = {}
    for rm in packed_meta:
        for pid in rm["pair_ids"]:
            pair_to_row[pid] = rm["row_index"]

    # Analyze each compact output
    results = []
    total_current_words = 0
    total_compact_words = 0
    total_saved = 0
    n_passed = 0
    n_failed_entity = 0
    n_failed_number = 0
    n_failed_length = 0
    n_failed_empty = 0

    for pid, compact_text in compact.items():
        pair = pairs.get(pid)
        if not pair:
            continue

        if not compact_text or not compact_text.strip():
            n_failed_empty += 1
            continue

        current_rewrite = pair.get("rewrite", pair.get("current_rewrite", ""))
        source_text = pair["original"]

        curr_wc = word_count(current_rewrite)
        comp_wc = word_count(compact_text)
        saved = curr_wc - comp_wc

        # Quality checks
        entity_ok = True
        entities = pair.get("entity_source", [])
        if entities:
            text_lower = compact_text.lower()
            for ent in entities:
                if ent.lower() not in text_lower:
                    entity_ok = False
                    break

        number_ok = True
        numbers = pair.get("num_source", [])
        if numbers:
            for num in numbers:
                if num not in compact_text:
                    number_ok = False
                    break

        # Length check: compact must be strictly shorter than current
        length_ok = comp_wc < curr_wc

        passed = entity_ok and number_ok and length_ok

        if not entity_ok:
            n_failed_entity += 1
        if not number_ok:
            n_failed_number += 1
        if not length_ok:
            n_failed_length += 1

        if passed:
            n_passed += 1
            total_current_words += curr_wc
            total_compact_words += comp_wc
            total_saved += saved

        results.append({
            "pair_id": pid,
            "source": pair.get("source", ""),
            "current_words": curr_wc,
            "compact_words": comp_wc,
            "saved": saved,
            "entity_ok": entity_ok,
            "number_ok": number_ok,
            "length_ok": length_ok,
            "passed": passed,
        })

    # Affected rows
    affected_rows = set()
    for r in results:
        if r["passed"]:
            rid = pair_to_row.get(r["pair_id"])
            if rid is not None:
                affected_rows.add(rid)

    # Top-up capacity
    n_topup_available = 0
    topup_words_available = 0
    with open(TOPUP_ROWS) as f:
        for line in f:
            row = json.loads(line)
            n_topup_available += 1
            topup_words_available += row.get("words", word_count(row.get("text", "")))

    topup_rows_needed = total_saved // 160  # approximate 160-word rows
    topup_slack = total_saved - (topup_rows_needed * 160)

    summary = {
        "status": "COMPACT_AUDIT",
        "compact_outputs_examined": len(compact),
        "matched_to_pairs": len(results),
        "n_passed_quality": n_passed,
        "n_failed_entity": n_failed_entity,
        "n_failed_number": n_failed_number,
        "n_failed_length": n_failed_length,
        "n_failed_empty": n_failed_empty,
        "pass_rate": round(n_passed / max(len(results), 1), 4),
        "total_current_rewrite_words": total_current_words,
        "total_compact_rewrite_words": total_compact_words,
        "total_saved_words": total_saved,
        "mean_savings_per_pair": round(total_saved / max(n_passed, 1), 2),
        "affected_packed_rows": len(affected_rows),
        "topup_rows_available": n_topup_available,
        "topup_words_available": topup_words_available,
        "topup_rows_needed_approx": topup_rows_needed,
        "topup_slack_words": topup_slack,
        "extrapolation": {
            "note": "If pilot pass rate and savings hold for the full 26,567 manifest",
            "estimated_total_passed": round(n_passed / max(len(results), 1) * 26567),
            "estimated_total_saved": round(total_saved / max(n_passed, 1) * (n_passed / max(len(results), 1) * 26567)),
            "estimated_topup_160w_rows": round(total_saved / max(n_passed, 1) * (n_passed / max(len(results), 1) * 26567) / 160),
        },
    }

    with open(DATA / "compact_audit_summary.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    with open(DATA / "compact_audit_details.jsonl", "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


def build(args):
    """Build a modified 10M pool with compact rewrites and optional reinvestment."""
    DATA.mkdir(parents=True, exist_ok=True)
    arm = args.arm

    if arm not in ("compact_only", "compact_reinvest", "compact_neutral"):
        print(f"ERROR: unknown arm '{arm}'", file=sys.stderr)
        sys.exit(1)

    # Load data
    pairs = load_selected_pairs()
    packed_meta = load_packed_row_meta()
    compact = load_compact_outputs(args.compact_outputs)

    # Build pair_id -> compact_text map (quality-filtered)
    compact_map = {}
    for pid, text in compact.items():
        pair = pairs.get(pid)
        if not pair or not text or not text.strip():
            continue
        curr_wc = word_count(pair.get("rewrite", pair.get("current_rewrite", "")))
        comp_wc = word_count(text)
        if comp_wc >= curr_wc:
            continue  # no savings
        # Entity check
        entities = pair.get("entity_source", [])
        text_lower = text.lower()
        if entities and any(e.lower() not in text_lower for e in entities):
            continue
        # Number check
        numbers = pair.get("num_source", [])
        if numbers and any(n not in text for n in numbers):
            continue
        compact_map[pid] = text

    print(f"Quality-filtered compact rewrites: {len(compact_map)}", flush=True)

    # Build pair_id -> row_index lookup
    pair_to_row_idx = {}
    row_meta_by_idx = {}
    for rm in packed_meta:
        row_meta_by_idx[rm["row_index"]] = rm
        for pid in rm["pair_ids"]:
            pair_to_row_idx[pid] = rm["row_index"]

    # Process the pool: read, modify pair rows, write
    arm_dir = DATA / arm
    arm_dir.mkdir(parents=True, exist_ok=True)
    out_pool = arm_dir / "qwen_compact_10M.jsonl"

    total_saved = 0
    rows_modified = 0
    rows_written = 0
    total_words = 0
    pair_rows_touched = set()

    # Identify all affected row indices
    affected_row_indices = set()
    for pid in compact_map:
        ridx = pair_to_row_idx.get(pid)
        if ridx is not None:
            affected_row_indices.add(ridx)

    # Need to know which row_index -> line_index in the pool
    # The pool may not be in row_index order, so we scan
    row_index_to_line = {}
    with open(QWEN_POOL) as f:
        for line_idx, line in enumerate(f):
            row = json.loads(line)
            if row.get("source") == "qwen_pair_packed":
                eid = row["example_id"]
                # Find matching packed_meta entry
                for rm in packed_meta:
                    if rm["example_id"] == eid:
                        row_index_to_line[rm["row_index"]] = line_idx
                        break

    print(f"Mapped {len(row_index_to_line)} pair rows to pool lines", flush=True)
    print(f"Affected pair rows: {len(affected_row_indices)}", flush=True)

    # Read all pool rows
    pool_rows = []
    with open(QWEN_POOL) as f:
        for line in f:
            pool_rows.append(json.loads(line))

    # Modify affected pair rows by rebuilding from pairs
    for ridx in affected_row_indices:
        rm = row_meta_by_idx.get(ridx)
        if not rm:
            continue
        line_idx = row_index_to_line.get(ridx)
        if line_idx is None:
            continue

        # Rebuild text from pairs
        parts = []
        row_saved = 0
        for pid in rm["pair_ids"]:
            pair = pairs.get(pid)
            if not pair:
                continue
            source_text = pair["original"]
            if pid in compact_map:
                rewrite_text = compact_map[pid]
                curr_wc = word_count(pair.get("rewrite", pair.get("current_rewrite", "")))
                comp_wc = word_count(rewrite_text)
                row_saved += (curr_wc - comp_wc)
            else:
                rewrite_text = pair.get("rewrite", pair.get("current_rewrite", ""))
            parts.append(f"{source_text} {rewrite_text}")

        new_text = " ".join(parts)
        new_wc = word_count(new_text)

        old_row = pool_rows[line_idx]
        old_wc = old_row["words"]

        pool_rows[line_idx] = {
            "text": new_text,
            "words": new_wc,
            "example_id": old_row["example_id"],
            "source": "qwen_pair_packed",
        }
        total_saved += (old_wc - new_wc)
        rows_modified += 1

    print(f"Rows modified: {rows_modified}, total saved words: {total_saved}", flush=True)

    # For reinvestment arms, add top-up rows
    topup_added = 0
    topup_words_added = 0

    if arm in ("compact_reinvest", "compact_neutral"):
        topup_source = TOPUP_ROWS if arm == "compact_reinvest" else NEUTRAL_ROWS
        words_budget = total_saved
        topup_candidates = []
        with open(topup_source) as f:
            for line in f:
                topup_candidates.append(json.loads(line))

        # Add top-up rows up to the saved word budget
        for tr in topup_candidates:
            tw = tr.get("words", word_count(tr.get("text", "")))
            if topup_words_added + tw <= words_budget:
                pool_rows.append({
                    "text": tr["text"],
                    "words": tw,
                    "example_id": tr.get("example_id", 900000 + topup_added),
                    "source": tr.get("source", "topup"),
                })
                topup_added += 1
                topup_words_added += tw
            if topup_words_added >= words_budget:
                break

        print(f"Top-up rows added: {topup_added}, words added: {topup_words_added}", flush=True)

    # Write modified pool
    total_words = sum(r["words"] for r in pool_rows)
    with open(out_pool, "w") as f:
        for r in pool_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    rows_written = len(pool_rows)

    # Compute SHA
    h = hashlib.sha256()
    with open(out_pool, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)

    # Source distribution
    src_words = Counter()
    src_rows = Counter()
    for r in pool_rows:
        src_words[r["source"]] += r["words"]
        src_rows[r["source"]] += 1

    summary = {
        "status": f"CORPUS_{arm.upper()}_BUILT",
        "arm": arm,
        "rows_written": rows_written,
        "total_words": total_words,
        "pair_rows_modified": rows_modified,
        "total_words_saved_from_compaction": total_saved,
        "topup_rows_added": topup_added,
        "topup_words_added": topup_words_added,
        "net_word_change": topup_words_added - total_saved,
        "source_words": dict(src_words),
        "source_rows": dict(src_rows),
        "output_pool": str(out_pool),
        "sha256": h.hexdigest(),
        "quality_filtered_compact_pairs": len(compact_map),
    }

    with open(arm_dir / "corpus_build_summary.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd")

    audit_p = sub.add_parser("audit")
    audit_p.add_argument("--compact-outputs", required=True,
                        help="JSONL with pair_id and compact_rewrite fields")

    build_p = sub.add_parser("build")
    build_p.add_argument("--compact-outputs", required=True)
    build_p.add_argument("--arm", required=True,
                        choices=["compact_only", "compact_reinvest", "compact_neutral"])

    args = ap.parse_args()
    if args.cmd == "audit":
        audit(args)
    elif args.cmd == "build":
        build(args)
    else:
        ap.print_help()
        sys.exit(1)
