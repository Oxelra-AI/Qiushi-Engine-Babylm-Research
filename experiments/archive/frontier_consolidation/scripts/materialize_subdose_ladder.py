#!/usr/bin/env python3
"""research: Materialize sub-1x dose ladder in MAX geometry.

Creates nested dose arms at quarter-1x, half-1x, and full-1x using the
MAX rowholdout row geometry, with row-level mixing of MAX view and clean arms.

All arms share the same 65,313-row 10M pool structure (7,923 pair rows +
1 topup row + 57,389 shared filler rows) and 653,130-row 100M stream.
The rho=0 reference is the already-trained MAX-geometry DeBERTa clean control.

Design:
  - Lines 0-7922 (0-indexed) are pair/clean rows. Active pair rows use
    the MAX view arm content (source+compact rewrite); inactive pair rows
    use the MAX clean arm content (clean-Qwen replacement).
  - Line 7923 is the topup row (same word count in both arms; uses view arm).
  - Lines 7924-65312 are shared filler rows (identical in both arms).
  - The 100M stream is 10x concatenation of the 10M pool (no shuffle).

Nesting: quarter_1x ⊆ half_1x ⊆ full_1x ⊆ all pair rows.
Rows are selected by cumulative 1x pair words: sort 1x-containing rows
by 1x-pair-word fraction descending (pure 1x first, then mixed), and
select until cumulative 1x pair words reach the target fraction.

Coverage is computed from the active pairs' source+rewrite text: unique
docs, unique sentence IDs, content types (lowercase 4+ letter words),
and domains. A cumulative CSV tracks marginal coverage growth.

No GPU, model loading, training, evaluation, or FineWeb streaming.
"""

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import json
import hashlib
import pathlib
import collections
import re
import sys
import statistics
import time

ROOT = _public_path('experiments/archive/frontier_consolidation')  # experiments/archive/frontier_consolidation
WS = _public_path('experiments/archive/frontier_consolidation')

# Source files
MAX_VIEW_10M = _public_path('experiments/archive/frontier_consolidation/data/dose_2p64x_rowholdout_pools/compact_view_dose2p64x_10M.jsonl')
MAX_CLEAN_10M = _public_path('experiments/archive/frontier_consolidation/data/dose_2p64x_rowholdout_pools/cleanqwen_lengthmatched_dose2p64x_10M.jsonl')
MAX_PAIRS = _public_path('experiments/archive/frontier_consolidation/data/dose_distribution_select/selected_matched_max_pairs.jsonl')
MAX_VIEW_META = _public_path('experiments/archive/frontier_consolidation/data/dose_2p64x_rowholdout_pools/compact_view_dose2p64x_changed_block_rows_meta.jsonl')

# Structure constants
PAIR_ROWS = 7923          # pair rows with content: lines 0..7922 (0-indexed)
META_ROWS = 7924          # metadata includes topup row at index 7923
TOPUP_LINE_0IDX = 7923    # line 7923 (0-indexed) = topup with 0 pairs
TOTAL_LINES = 65313
BUDGET_WORDS = 10_000_000

OUT_DIR = _public_path('experiments/archive/frontier_consolidation/data/subdose_ladder_maxgeom_pools')

# Dose targets as fractions of total 1x pair words
DOSE_FRACS = collections.OrderedDict([
    ("quarter_1x", 0.25),
    ("half_1x",    0.50),
    ("full_1x",    1.00),
])

CONTENT_RE = re.compile(r'[A-Za-z]{4,}')

def content_toks(text: str) -> list[str]:
    return [m.group().lower() for m in CONTENT_RE.finditer(text)]


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def rel(p: pathlib.Path) -> str:
    try:
        return str(p.relative_to(_public_path('.')))
    except ValueError:
        return str(p)


def main():
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Phase 1: Load pair origins and text ──────────────────────────
    print("Phase 1: Loading MAX pair selection...")
    pair_origin = {}   # pair_id → origin string
    pair_info = {}     # pair_id → dict with text, doc_id, pair_words, etc.
    n_1x = n_inc = 0
    for line in open(MAX_PAIRS):
        r = json.loads(line)
        pid = r["pair_id"]
        origin = r.get("origin", "unknown")
        pair_origin[pid] = origin
        pair_info[pid] = dict(
            doc_id=str(r.get("doc_id", "")),
            sentence_id=str(r.get("sentence_id", "")),
            pair_words=int(r.get("pair_words", 0)),
            source_words=int(r.get("source_words", 0)),
            rewrite_words=int(r.get("rewrite_words", 0)),
            source_text=r.get("source_text", ""),
            rewrite_text=r.get("rewrite_text", ""),
            domain_hits=r.get("domain_hits", []),
        )
        if "old_selected_1x" in origin:
            n_1x += 1
        else:
            n_inc += 1
    print(f"  Total pairs: {len(pair_origin)}, 1x: {n_1x}, increment: {n_inc}")
    assert len(pair_origin) == 33291, f"Expected 33291 pairs, got {len(pair_origin)}"
    assert n_1x == 12155, f"Expected 12155 1x pairs, got {n_1x}"

    # ── Phase 2: Classify changed rows ───────────────────────────────
    print("Phase 2: Loading changed-row metadata...")
    row_meta = []
    for line in open(MAX_VIEW_META):
        r = json.loads(line)
        pair_ids = r.get("pair_ids", [])
        words = int(r["words"])
        n1 = sum(1 for p in pair_ids if "old_selected_1x" in pair_origin.get(p, ""))
        ni = len(pair_ids) - n1
        pw1 = sum(pair_info[p]["pair_words"] for p in pair_ids
                   if "old_selected_1x" in pair_origin.get(p, ""))
        pwi = sum(pair_info[p]["pair_words"] for p in pair_ids
                   if "old_selected_1x" not in pair_origin.get(p, ""))
        row_meta.append(dict(
            row_index=int(r["row_index"]),
            words=words,
            n_pairs=len(pair_ids),
            pair_ids=pair_ids,
            n_1x=n1, n_inc=ni,
            pw_1x=pw1, pw_inc=pwi,
            row_class=("pure_1x" if ni == 0 and n1 > 0 else
                       "pure_increment" if n1 == 0 else "mixed"),
        ))
    assert len(row_meta) == META_ROWS, f"Expected {META_ROWS} metadata rows, got {len(row_meta)}"
    # The last metadata row is the topup (0 pairs); exclude from pair classification
    pair_row_meta = row_meta[:PAIR_ROWS]
    # Verify pair words match row words for pair rows
    for rm in pair_row_meta:
        assert rm["pw_1x"] + rm["pw_inc"] == rm["words"], (
            f"Row {rm['row_index']}: pw_1x={rm['pw_1x']} + pw_inc={rm['pw_inc']} != words={rm['words']}")
    # Topup row should have 0 pairs
    assert row_meta[PAIR_ROWS]["n_pairs"] == 0, f"Topup row has {row_meta[PAIR_ROWS]['n_pairs']} pairs"

    n_pure1 = sum(1 for r in pair_row_meta if r["row_class"] == "pure_1x")
    n_mixed = sum(1 for r in pair_row_meta if r["row_class"] == "mixed")
    n_pureinc = sum(1 for r in pair_row_meta if r["row_class"] == "pure_increment")
    pw_pure1 = sum(r["words"] for r in pair_row_meta if r["row_class"] == "pure_1x")
    pw_mixed_total = sum(r["words"] for r in pair_row_meta if r["row_class"] == "mixed")
    pw_mixed_1x = sum(r["pw_1x"] for r in pair_row_meta if r["row_class"] == "mixed")
    print(f"  Changed rows: {len(row_meta)}")
    print(f"  Pure 1x:       {n_pure1} rows, {pw_pure1} words")
    print(f"  Mixed:         {n_mixed} rows, {pw_mixed_total} words (1x portion: {pw_mixed_1x})")
    print(f"  Pure increment:{n_pureinc} rows")

    total_1x_pw = sum(r["pw_1x"] for r in pair_row_meta)
    print(f"  Total 1x pair words across all rows: {total_1x_pw}")
    assert total_1x_pw == 423511, f"Expected 423511, got {total_1x_pw}"

    # ── Phase 3: Build nested dose selections ────────────────────────
    print("Phase 3: Building dose selections...")
    # Sort rows with any 1x content: pure_1x first (by row_index),
    # then mixed (by 1x fraction descending, then row_index)
    rows_with_1x = [r for r in pair_row_meta if r["n_1x"] > 0]
    rows_with_1x.sort(key=lambda r: (
        0 if r["row_class"] == "pure_1x" else 1,
        -r["pw_1x"] / max(1, r["words"]),
        r["row_index"],
    ))

    # Build cumulative pair-word ladder
    cum_1x_pw = 0
    cum_all_pw = 0
    cum_rows = 0
    dose_selections = {}  # dose_name → set of row indices (0-based)
    thresholds_hit = {}   # dose_name → bool

    for dose_name in DOSE_FRACS:
        thresholds_hit[dose_name] = False
        dose_selections[dose_name] = set()

    ordered_row_indices = []
    for r in rows_with_1x:
        ordered_row_indices.append(r["row_index"])
        cum_1x_pw += r["pw_1x"]
        cum_all_pw += r["words"]
        cum_rows += 1
        for dose_name, frac in DOSE_FRACS.items():
            target = int(total_1x_pw * frac)
            if not thresholds_hit[dose_name]:
                dose_selections[dose_name].add(r["row_index"])
                if cum_1x_pw >= target:
                    thresholds_hit[dose_name] = True

    # Verify nesting
    for a, b in [("quarter_1x", "half_1x"), ("half_1x", "full_1x")]:
        assert dose_selections[a].issubset(dose_selections[b]), f"{a} not subset of {b}"

    dose_info = {}
    for dose_name, indices in dose_selections.items():
        n1 = sum(row_meta[i]["n_1x"] for i in indices)
        ni = sum(row_meta[i]["n_inc"] for i in indices)
        pw = sum(row_meta[i]["words"] for i in indices)
        pw1 = sum(row_meta[i]["pw_1x"] for i in indices)
        rho = pw / BUDGET_WORDS
        dose_info[dose_name] = dict(
            active_rows=len(indices),
            n_1x_pairs=n1, n_inc_pairs=ni,
            active_total_pw=pw, active_1x_pw=pw1,
            rho=rho,
        )
        print(f"  {dose_name}: {len(indices)} rows, "
              f"{n1}+{ni} pairs, {pw} pw (1x: {pw1}), ρ={rho:.6f}")

    # ── Phase 4: Cumulative coverage CSV ─────────────────────────────
    print("Phase 4: Computing cumulative coverage...")
    cum_csv_path = _public_path('experiments/archive/frontier_consolidation/data/subdose_ladder_maxgeom_pools/subdose_cumulative_coverage.csv')
    cum_docs = set()
    cum_sids = set()
    cum_content = collections.Counter()
    cum_domains = collections.Counter()
    cum_pw = 0
    cum_pairs = 0
    prev_types = 0

    with open(cum_csv_path, 'w') as f:
        f.write("rank,row_index,row_class,row_pw,row_pw_1x,row_pw_inc,"
                "cum_rows,cum_pairs,cum_pw,cum_1x_pw,"
                "cum_docs,cum_sids,cum_content_types,cum_hapax,"
                "marginal_new_types,marginal_new_types_per_100k_pw\n")
        cum_1x_pw_running = 0
        for rank, ridx in enumerate(ordered_row_indices):
            rm = row_meta[ridx]
            row_pw = rm["words"]
            cum_pw += row_pw
            cum_1x_pw_running += rm["pw_1x"]
            cum_pairs += rm["n_pairs"]
            for pid in rm["pair_ids"]:
                pi = pair_info[pid]
                if pi["doc_id"]:
                    cum_docs.add(pi["doc_id"])
                if pi["sentence_id"]:
                    cum_sids.add(pi["sentence_id"])
                cum_content.update(content_toks(pi["source_text"]))
                cum_content.update(content_toks(pi["rewrite_text"]))
                for d in pi["domain_hits"]:
                    cum_domains[d] += 1
            new_types = len(cum_content) - prev_types
            marginal = (new_types / max(1, row_pw)) * 100_000
            prev_types = len(cum_content)
            hapax = sum(1 for v in cum_content.values() if v == 1)
            f.write(f"{rank},{ridx},{rm['row_class']},{row_pw},{rm['pw_1x']},{rm['pw_inc']},"
                    f"{rank+1},{cum_pairs},{cum_pw},{cum_1x_pw_running},"
                    f"{len(cum_docs)},{len(cum_sids)},{len(cum_content)},{hapax},"
                    f"{new_types},{marginal:.1f}\n")
    print(f"  Wrote {cum_csv_path.name}: {len(ordered_row_indices)} rows")

    # Per-dose coverage snapshots
    dose_coverage = {}
    for dose_name, indices in dose_selections.items():
        docs = set()
        sids = set()
        cc = collections.Counter()
        doms = collections.Counter()
        n_pairs = 0
        for ridx in indices:
            for pid in row_meta[ridx]["pair_ids"]:
                pi = pair_info[pid]
                n_pairs += 1
                if pi["doc_id"]: docs.add(pi["doc_id"])
                if pi["sentence_id"]: sids.add(pi["sentence_id"])
                cc.update(content_toks(pi["source_text"]))
                cc.update(content_toks(pi["rewrite_text"]))
                for d in pi["domain_hits"]:
                    doms[d] += 1
        dose_coverage[dose_name] = dict(
            unique_docs=len(docs),
            unique_sids=len(sids),
            content_types=len(cc),
            hapax=sum(1 for v in cc.values() if v == 1),
            domains=dict(doms.most_common()),
        )
        print(f"  {dose_name}: {n_pairs} active pairs, "
              f"{len(docs)} docs, {len(cc)} content types")

    # ── Phase 5: Construct 10M pools ─────────────────────────────────
    print("Phase 5: Reading source 10M files...")
    with open(MAX_VIEW_10M) as f:
        view_lines = f.readlines()
    with open(MAX_CLEAN_10M) as f:
        clean_lines = f.readlines()
    assert len(view_lines) == TOTAL_LINES, f"View: {len(view_lines)} lines"
    assert len(clean_lines) == TOTAL_LINES, f"Clean: {len(clean_lines)} lines"

    # Verify shared filler lines are identical
    n_filler_checked = 0
    n_filler_mismatch = 0
    for i in range(TOPUP_LINE_0IDX + 1, TOTAL_LINES):
        if view_lines[i] != clean_lines[i]:
            n_filler_mismatch += 1
        n_filler_checked += 1
    print(f"  Filler lines checked: {n_filler_checked}, mismatches: {n_filler_mismatch}")
    if n_filler_mismatch > 0:
        print("  WARNING: some filler lines differ; using view arm for shared lines")

    # Verify per-row word count identity for pair rows
    for i in range(PAIR_ROWS):
        vr = json.loads(view_lines[i])
        cr = json.loads(clean_lines[i])
        vw = int(vr.get("words", len(vr.get("text","").split())))
        cw = int(cr.get("words", len(cr.get("text","").split())))
        assert vw == cw, f"Line {i}: view words {vw} != clean words {cw}"

    print("  Per-row word count identity verified for all pair rows ✓")

    for dose_name in DOSE_FRACS:
        indices = dose_selections[dose_name]
        out_10m = OUT_DIR / f"subdose_{dose_name}_view_10M.jsonl"
        print(f"  Writing {out_10m.name} ({len(indices)} active pair rows)...")

        with open(out_10m, 'w') as f:
            for i in range(TOTAL_LINES):
                if i < PAIR_ROWS:
                    if i in indices:
                        f.write(view_lines[i])
                    else:
                        f.write(clean_lines[i])
                else:
                    f.write(view_lines[i])

        # Verify
        total_words = 0
        n_rows = 0
        with open(out_10m) as f:
            for line in f:
                r = json.loads(line)
                total_words += int(r.get("words", len(r.get("text","").split())))
                n_rows += 1
        assert n_rows == TOTAL_LINES, f"{dose_name}: {n_rows} rows"
        assert total_words == BUDGET_WORDS, f"{dose_name}: {total_words} words != {BUDGET_WORDS}"
        print(f"    Verified: {n_rows} rows, {total_words} words ✓")

    # Free memory
    del view_lines, clean_lines

    # ── Phase 6: Build 100M streams ──────────────────────────────────
    print("Phase 6: Building 100M streams (10x concatenation)...")
    for dose_name in DOSE_FRACS:
        in_10m = OUT_DIR / f"subdose_{dose_name}_view_10M.jsonl"
        out_100m = OUT_DIR / f"subdose_{dose_name}_view_100M.jsonl"
        print(f"  Writing {out_100m.name}...")
        with open(in_10m, 'rb') as src:
            content_bytes = src.read()
        with open(out_100m, 'wb') as dst:
            for _ in range(10):
                dst.write(content_bytes)
        # Quick check: file size should be 10x
        sz_10 = in_10m.stat().st_size
        sz_100 = out_100m.stat().st_size
        assert sz_100 == 10 * sz_10, f"{dose_name}: 100M={sz_100} != 10*{sz_10}"
        print(f"    10M: {sz_10:,} bytes, 100M: {sz_100:,} bytes ✓")

    # ── Phase 7: Compute hashes ──────────────────────────────────────
    print("Phase 7: Computing SHA256 hashes...")
    hashes = {}
    for dose_name in DOSE_FRACS:
        h10 = sha256_file(OUT_DIR / f"subdose_{dose_name}_view_10M.jsonl")
        h100 = sha256_file(OUT_DIR / f"subdose_{dose_name}_view_100M.jsonl")
        hashes[dose_name] = dict(sha_10m=h10, sha_100m=h100)
        print(f"  {dose_name}: 10M={h10[:16]}... 100M={h100[:16]}...")

    # ── Phase 8: Write metadata and summary ──────────────────────────
    print("Phase 8: Writing metadata and summary...")
    metadata = {
        "status": "SUBDOSE_LADDER_MATERIALIZED",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "purpose": ("Sub-1x dose ladder in MAX rowholdout geometry. "
                     "Tests the onset/knee of the admission effect below rho=0.042. "
                     "The rho=0 reference is the already-trained MAX-geometry DeBERTa clean."),
        "source_files": {
            "max_view_10m": rel(MAX_VIEW_10M),
            "max_clean_10m": rel(MAX_CLEAN_10M),
            "max_pairs": rel(MAX_PAIRS),
            "max_view_meta": rel(MAX_VIEW_META),
        },
        "geometry": {
            "total_lines_10m": TOTAL_LINES,
            "pair_rows": PAIR_ROWS,
            "topup_line_0idx": TOPUP_LINE_0IDX,
            "filler_lines": TOTAL_LINES - PAIR_ROWS - 1,
            "budget_words": BUDGET_WORDS,
            "repeats_100m": 10,
        },
        "row_classification": {
            "pure_1x": n_pure1,
            "mixed": n_mixed,
            "pure_increment": n_pureinc,
            "pure_1x_words": pw_pure1,
            "mixed_words_total": pw_mixed_total,
            "mixed_words_1x_portion": pw_mixed_1x,
        },
        "pair_counts": {
            "total_max_pairs": 33291,
            "total_1x_pairs": n_1x,
            "total_1x_pair_words": total_1x_pw,
        },
        "filler_verification": {
            "checked": n_filler_checked,
            "mismatches": n_filler_mismatch,
        },
        "doses": {},
    }

    for dose_name in DOSE_FRACS:
        di = dose_info[dose_name]
        dc = dose_coverage[dose_name]
        metadata["doses"][dose_name] = {
            "fraction_of_1x": DOSE_FRACS[dose_name],
            "active_rows": di["active_rows"],
            "n_1x_pairs": di["n_1x_pairs"],
            "n_inc_pairs": di["n_inc_pairs"],
            "active_total_pw": di["active_total_pw"],
            "active_1x_pw": di["active_1x_pw"],
            "rho": di["rho"],
            "unique_docs": dc["unique_docs"],
            "unique_sids": dc["unique_sids"],
            "content_types": dc["content_types"],
            "hapax_content_types": dc["hapax"],
            "domains": dc["domains"],
            "sha_10m": hashes[dose_name]["sha_10m"],
            "sha_100m": hashes[dose_name]["sha_100m"],
            "file_10m": rel(OUT_DIR / f"subdose_{dose_name}_view_10M.jsonl"),
            "file_100m": rel(OUT_DIR / f"subdose_{dose_name}_view_100M.jsonl"),
        }

    meta_path = _public_path('experiments/archive/frontier_consolidation/data/subdose_ladder_maxgeom_pools/subdose_ladder_metadata.json')
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    # Summary markdown
    summary_path = _public_path('research/documents/frontier_consolidation/data/subdose_ladder_maxgeom_pools/subdose_ladder_summary.md')
    with open(summary_path, 'w') as f:
        f.write("# research: Sub-1x dose ladder in MAX geometry\n\n")
        f.write("All arms share the MAX rowholdout row-word-count sequence (65,313 rows, 10M words).\n")
        f.write("The rho=0 reference is the already-trained MAX-geometry DeBERTa clean control.\n\n")
        f.write("## Row classification\n\n")
        f.write(f"- Pure 1x rows: {n_pure1} ({pw_pure1} words)\n")
        f.write(f"- Mixed rows: {n_mixed} ({pw_mixed_total} words, 1x portion: {pw_mixed_1x})\n")
        f.write(f"- Pure increment rows: {n_pureinc}\n\n")
        f.write("## Dose ladder\n\n")
        f.write("| Dose | Rows | 1x pairs | Inc pairs | Total PW | 1x PW | ρ | Docs | Types |\n")
        f.write("|------|------|----------|-----------|----------|-------|---|------|-------|\n")
        f.write(f"| clean | 0 | 0 | 0 | 0 | 0 | 0.0 | 0 | 0 |\n")
        for dn in DOSE_FRACS:
            di = dose_info[dn]
            dc = dose_coverage[dn]
            f.write(f"| {dn} | {di['active_rows']} | {di['n_1x_pairs']} | {di['n_inc_pairs']} | "
                    f"{di['active_total_pw']} | {di['active_1x_pw']} | {di['rho']:.6f} | "
                    f"{dc['unique_docs']} | {dc['content_types']} |\n")
        f.write(f"| dose1p82 | (existing) | | | 771199 | | 0.077120 | 5089 | 36016 |\n")
        f.write(f"| MAX | (existing) | | | 1118587 | | 0.111872 | 5261 | 43540 |\n")
        f.write("\n## Files\n\n")
        for dn in DOSE_FRACS:
            f.write(f"### {dn}\n")
            f.write(f"- 10M: `{rel(OUT_DIR / f'subdose_{dn}_view_10M.jsonl')}`\n")
            f.write(f"- 100M: `{rel(OUT_DIR / f'subdose_{dn}_view_100M.jsonl')}`\n")
            f.write(f"- SHA 10M: `{hashes[dn]['sha_10m']}`\n")
            f.write(f"- SHA 100M: `{hashes[dn]['sha_100m']}`\n\n")
        f.write("## Coverage saturation\n\n")
        f.write("See `subdose_cumulative_coverage.csv` for the per-row cumulative coverage curve.\n")
        f.write("The key question: does the score knee coincide with the coverage knee?\n")

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s")
    print(json.dumps({
        "status": "SUBDOSE_LADDER_MATERIALIZED",
        "output_dir": rel(OUT_DIR),
        "metadata": rel(meta_path),
        "summary": rel(summary_path),
        "coverage_csv": rel(cum_csv_path),
        "doses": {dn: {
            "active_rows": dose_info[dn]["active_rows"],
            "rho": dose_info[dn]["rho"],
            "content_types": dose_coverage[dn]["content_types"],
            "unique_docs": dose_coverage[dn]["unique_docs"],
        } for dn in DOSE_FRACS},
        "elapsed_sec": round(elapsed, 1),
        "no_gpu_training_eval_streaming": True,
    }, indent=2))


if __name__ == "__main__":
    main()
