#!/usr/bin/env python3
"""Audit whether research posalign hard assignment canonicalizes all name occurrences."""
from __future__ import annotations
import json, sys
from collections import Counter, defaultdict
from pathlib import Path

PROJECT = Path("experiments/archive/representation_and_objectives")
sys.path.insert(0, str(PROJECT / "training/scripts"))
import raw_span_discovery_probe as base  # noqa: E402

OUT = PROJECT / "data/posalign_occurrence_audit"


def count_occurrences(text, names):
    _, forms = base.raw_tokenize(text)
    forms = [str(f).lower() for f in forms]
    return {n.lower(): [i for i, f in enumerate(forms) if f == n.lower()] for n in names}, forms


def audit():
    ts, tc, es, ec, _, counts = base.load_dataset(base.DEFAULT_DATA_ROOT, base.DEFAULT_ARM)
    rows = []
    for split, states, comps in [("train", ts, tc), ("eval", es, ec)]:
        for q in states:
            if not q.is_changed: continue
            occ, forms = count_occurrences(q.event, q.names)
            rows.append({"split": split, "task": "state", "suite": q.suite, "key": q.key,
                         "names": list(q.names), "n_tokens": len(forms),
                         "counts": {n: len(occ[n.lower()]) for n in q.names},
                         "max_occ": max(len(v) for v in occ.values()),
                         "both_single": all(len(v) == 1 for v in occ.values()),
                         "text": q.event})
        for c in comps:
            for ei, ev in enumerate([c.event1, c.event2], start=1):
                occ, forms = count_occurrences(ev, c.names)
                rows.append({"split": split, "task": f"cmp_event{ei}", "suite": c.suite, "key": c.row_id,
                             "names": list(c.names), "n_tokens": len(forms),
                             "counts": {n: len(occ[n.lower()]) for n in c.names},
                             "max_occ": max(len(v) for v in occ.values()),
                             "both_single": all(len(v) == 1 for v in occ.values()),
                             "text": ev})
    def subset_summary(xs):
        return {
            "n_events": len(xs),
            "both_names_single_occurrence_frac": sum(r["both_single"] for r in xs)/len(xs) if xs else 0,
            "any_name_repeated_frac": sum(r["max_occ"] > 1 for r in xs)/len(xs) if xs else 0,
            "max_occurrence_count": max((r["max_occ"] for r in xs), default=0),
            "occurrence_count_hist": dict(sorted(Counter(r["max_occ"] for r in xs).items())),
        }
    summary = {"counts": counts}
    for split in ["train", "eval"]:
        for task_group in ["state", "cmp"]:
            if task_group == "state": xs = [r for r in rows if r["split"] == split and r["task"] == "state"]
            else: xs = [r for r in rows if r["split"] == split and r["task"].startswith("cmp_")]
            summary[f"{split}_{task_group}"] = subset_summary(xs)
    examples = [r for r in rows if r["max_occ"] > 1][:20]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "occurrence_rows.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    (OUT / "occurrence_repeated_examples.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in examples))
    (OUT / "occurrence_audit.json").write_text(json.dumps({"summary": summary, "n_rows": len(rows)}, indent=2, ensure_ascii=False, sort_keys=True) + "\n")
    lines = ["# research posalign occurrence audit", "",
             "This checks whether raw event strings contain multiple occurrences of a candidate/other name. research hard assignment canonicalizes one argmax token per query; research hard oracle canonicalizes every exact occurrence.", "",
             "| subset | n events | both single frac | any repeated frac | max occurrence | hist max-occ |",
             "|---|---:|---:|---:|---:|---|"]
    for k, s in summary.items():
        if k == "counts": continue
        lines.append(f"| {k} | {s['n_events']} | {s['both_names_single_occurrence_frac']:.3f} | {s['any_name_repeated_frac']:.3f} | {s['max_occurrence_count']} | {s['occurrence_count_hist']} |")
    lines += ["", "## First repeated examples", ""]
    for r in examples[:8]:
        lines.append(f"- {r['split']} {r['task']} {r['suite']} {r['key']} counts={r['counts']} text=`{r['text']}`")
    ((OUT.parents[4] / 'research/documents/representation_and_objectives/data/posalign_occurrence_audit/occurrence_audit.md')).write_text("\n".join(lines) + "\n")
    print(json.dumps({"status": "OCCURRENCE_AUDIT_COMPLETE",
                      "md": str((OUT.parents[4] / 'research/documents/representation_and_objectives/data/posalign_occurrence_audit/occurrence_audit.md')),
                      "json": str(OUT / "occurrence_audit.json")}, indent=2))


if __name__ == "__main__":
    audit()
