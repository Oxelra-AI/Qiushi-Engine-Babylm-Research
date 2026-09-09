#!/usr/bin/env python3
"""Interleave research reinvestment top-up rows after saved compact qwen rows.

The initial structural materializer appended top-up rows at the end of the tail. That
keeps total words legal, but an early prefix would see no reinvested experience and a
full continuation would place most reinvestment at the lowest learning-rate tail. This
script preserves the exact same compacted qwen rows and top-up rows, but moves top-ups
forward using an online saved-word budget: as compacted rows save words, add the next
top-up row once enough saved capacity has accumulated. Total words and row identities
are preserved.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative

import argparse, json, pathlib, hashlib, time, sys
from collections import Counter
from typing import Any, Dict, Iterable, List

ROOT = _public_path('.')
STUDY = _public_path('experiments/archive/functional_learning')
IN_TAIL = _public_path('experiments/archive/functional_learning/data/structural_policy_materialization/compact_structural_reinvest/compact_structural_reinvest_reference_tail_wordpaced_segments.jsonl')
OUT_DIR = _public_path('experiments/archive/functional_learning/data/structural_policy_materialization/compact_structural_reinvest_interleaved')


def rel(p):
    try: return str(pathlib.Path(p).resolve().relative_to(ROOT))
    except Exception: return str(p)

def now(): return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

def wc(t): return len((t or "").strip().split())

def iter_jsonl(path):
    with pathlib.Path(path).open(encoding="utf-8") as f:
        for line in f:
            line=line.strip()
            if line: yield json.loads(line)

def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with pathlib.Path(path).open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False)+"\n")

def is_topup(row):
    return bool(row.get("structural_compact_topup")) or str(row.get("source", "")).startswith("structural_compact_")

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input-tail", type=pathlib.Path, default=IN_TAIL)
    ap.add_argument("--out-dir", type=pathlib.Path, default=OUT_DIR)
    ap.add_argument("--output-name", default="compact_structural_reinvest_interleaved_reference_tail_wordpaced_segments.jsonl")
    args=ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    ordinary=[]; topups=[]
    input_words=0
    for row in iter_jsonl(args.input_tail):
        input_words += int(row.get("words", wc(row.get("text", ""))))
        if is_topup(row):
            topups.append(row)
        else:
            ordinary.append(row)
    topup_queue=list(topups)
    out=[]
    saved_budget=0
    spent=0
    inserted=0
    insertion_events=[]
    for i,row in enumerate(ordinary):
        out.append(row)
        saved_budget += int(row.get("compact_saved_words", 0) or 0)
        # Insert topups as soon as their word count fits the accumulated saved budget.
        while topup_queue:
            tw=int(topup_queue[0].get("words", wc(topup_queue[0].get("text", ""))))
            if spent + tw > saved_budget:
                break
            tr=dict(topup_queue.pop(0))
            tr["interleaved_after_reference_tail_position"] = row.get("reference_tail_position", i)
            tr["interleaved_after_output_index"] = len(out)-1
            tr["interleaved_saved_budget_words_available"] = saved_budget
            tr["interleaved_saved_budget_words_spent_after"] = spent + tw
            out.append(tr)
            spent += tw
            inserted += 1
            if len(insertion_events) < 30:
                insertion_events.append({"after_input_index": i, "after_reference_tail_position": row.get("reference_tail_position", i), "topup_words": tw, "saved_budget": saved_budget, "spent": spent})
    # If any topups remain, append them only if legal saved budget still allows them.
    leftover_added=0
    for tr0 in topup_queue:
        tw=int(tr0.get("words", wc(tr0.get("text", ""))))
        if spent + tw <= saved_budget:
            tr=dict(tr0)
            tr["interleaved_after_reference_tail_position"] = "end_leftover"
            tr["interleaved_after_output_index"] = len(out)-1
            out.append(tr)
            spent += tw
            leftover_added += 1
        else:
            break
    out_path=args.out_dir/args.output_name
    write_jsonl(out_path,out)
    h=hashlib.sha256()
    with out_path.open("rb") as f:
        for chunk in iter(lambda:f.read(1<<20), b""):
            h.update(chunk)
    total_words=sum(int(r.get("words", wc(r.get("text", "")))) for r in out)
    src_rows=Counter(r.get("source","") for r in out)
    prefix_stats=[]
    for nupd in [20,40,80,160,354]:
        limit=nupd*39533
        rows=words=qwen=mods=tops=0
        for r in out:
            if words>=limit: break
            rows+=1
            words+=int(r.get("words", wc(r.get("text", ""))))
            qwen += int(r.get("source")=="qwen_pair_packed")
            mods += int(bool(r.get("compact_modified_pair_ids")))
            tops += int(is_topup(r))
        prefix_stats.append({"updates": nupd, "rows": rows, "words": words, "qwen_rows": qwen, "compact_modified_rows": mods, "topup_rows": tops})
    summary={
        "status":"INTERLEAVED_REINVEST_TAIL_BUILT",
        "created_utc":now(),
        "input_tail":rel(args.input_tail),
        "output_jsonl":rel(out_path),
        "input_rows":len(ordinary)+len(topups),
        "input_words":input_words,
        "ordinary_rows_non_topup":len(ordinary),
        "topup_rows_from_input":len(topups),
        "interleaved_topup_rows":inserted+leftover_added,
        "leftover_added_at_end":leftover_added,
        "topup_rows_not_inserted":len(topup_queue)-leftover_added,
        "saved_budget_total":saved_budget,
        "topup_words_spent":spent,
        "unspent_saved_words":saved_budget-spent,
        "output_rows":len(out),
        "output_words":total_words,
        "word_change_vs_input":total_words-input_words,
        "legal_total_words_if_from_coherent86":86005295+total_words,
        "source_rows":dict(src_rows),
        "sha256":h.hexdigest(),
        "first_insertion_events":insertion_events,
        "prefix_stats":prefix_stats,
        "interpretation":"Same compacted qwen rows and top-up rows as research structural reinvest tail, but top-ups are interleaved by accumulated saved-word budget so early real-stream prefixes test reinvested experience rather than only compaction.",
    }
    (args.out_dir/"interleaved_tail_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    if total_words != input_words:
        print("word mismatch", file=sys.stderr)
        sys.exit(1)

if __name__=="__main__": main()
