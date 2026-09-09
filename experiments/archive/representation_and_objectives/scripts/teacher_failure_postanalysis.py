#!/usr/bin/env python3
"""research teacher failure postanalysis for paired-world realization check."""
from __future__ import annotations
import collections, json, re
from pathlib import Path
from typing import Any

ROOT = Path("experiments/archive/representation_and_objectives")
IN = ROOT / "data/paired_world_sequence_teacher/teacher_labeled_rows.jsonl"
OUT = ROOT / "data/paired_world_sequence_teacher/teacher_failure_postanalysis.json"
MD = (ROOT.parents[2] / 'research/documents/representation_and_objectives/data/paired_world_sequence_teacher/teacher_failure_postanalysis.md')

TEMPLATE_META = {
    1:("defeated", "w", "train"), 2:("beat", "w", "train"), 3:("won against", "w", "train"),
    4:("overcame", "w", "train"), 5:("proved too strong for", "w", "train"),
    6:("lost to", "l", "train"), 7:("fell to", "l", "train"), 8:("was defeated by", "l", "train"),
    9:("was beaten by", "l", "train"), 10:("was unable to overcome", "l", "train"),
    11:("saw triumph over", "m", "train"), 12:("emerged victorious over", "m", "train"),
    13:("prevailed against", "m", "train"), 14:("came out on top", "m", "train"),
    15:("ended with victorious over", "m", "train"), 16:("edged out", "w", "held"),
    17:("succumbed to", "l", "held"), 18:("victory for W over L", "m", "held"),
    19:("claimed the win", "w", "held"), 20:("went down to", "l", "held"),
}
TEACHERS = ["qwen3.5-9b", "llama3.1-8b-instruct"]

def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]

def rate(n: int, d: int) -> float:
    return n / d if d else 0.0

def summarize(rows: list[dict[str, Any]], key_fn) -> dict[str, Any]:
    groups = collections.defaultdict(list)
    for r in rows:
        groups[str(key_fn(r))].append(r)
    out = {}
    for k in sorted(groups):
        rs = groups[k]
        out[k] = {"n": len(rs), "accuracy": rate(sum(1 for r in rs if r["correct"]), len(rs))}
    return out

def main() -> None:
    rows = read_jsonl(IN)
    by_id = collections.defaultdict(dict)
    for r in rows:
        tid = int(r["context_template_id"])
        phrase, first, split = TEMPLATE_META.get(tid, ("?", "?", "?"))
        r["template_phrase"] = phrase
        r["template_first"] = first
        r["template_meta_split"] = split
        by_id[r["id"]][r["teacher"]] = r
    pairs = []
    for pid, d in by_id.items():
        if not all(t in d for t in TEACHERS):
            continue
        q, l = d[TEACHERS[0]], d[TEACHERS[1]]
        qlab, llab = q.get("parsed_label"), l.get("parsed_label")
        gold = q["gold_label"]
        if qlab == llab == gold:
            state = "both_correct"
        elif qlab == gold and llab != gold:
            state = "qwen_correct_llama_wrong"
        elif llab == gold and qlab != gold:
            state = "llama_correct_qwen_wrong"
        elif qlab == llab and qlab != gold:
            state = "both_same_wrong"
        else:
            state = "both_different_wrong_or_invalid"
        pairs.append({**{k: q[k] for k in ["id","family_id","family_split","sport","variant","context_template_id","context_template_split","template_phrase","template_first","hyp_kind","hyp_direction","gold_label","context","hypothesis"]}, "qwen_label": qlab, "llama_label": llab, "state": state})
    def pair_summarize(key_fn):
        groups = collections.defaultdict(list)
        for p in pairs:
            groups[str(key_fn(p))].append(p)
        out = {}
        for k in sorted(groups):
            ps = groups[k]
            counts = collections.Counter(p["state"] for p in ps)
            out[k] = {"n": len(ps), "both_correct": rate(counts["both_correct"], len(ps)), "state_counts": dict(counts)}
        return out
    out = {
        "status": "TEACHER_FAILURE_POSTANALYSIS",
        "n_pairs": len(pairs),
        "state_counts": dict(collections.Counter(p["state"] for p in pairs)),
        "by_template_id": pair_summarize(lambda p: f"{p['context_template_id']:02d}:{p['template_phrase']}:{p['template_first']}:{p['context_template_split']}"),
        "by_template_first": pair_summarize(lambda p: p["template_first"]),
        "by_variant_template_split": pair_summarize(lambda p: f"{p['variant']}::{p['context_template_split']}"),
        "by_hyp_kind_gold": pair_summarize(lambda p: f"{p['hyp_kind']}::{p['gold_label']}"),
        "by_teacher_template_accuracy": {t: summarize([r for r in rows if r["teacher"] == t], lambda r: f"{int(r['context_template_id']):02d}:{r['template_phrase']}:{r['template_first']}:{r['context_template_split']}") for t in TEACHERS},
        "failure_examples_by_template": {},
        "interpretation": "Failures are disagreements rather than shared wrong labels if both_same_wrong is zero. Concentration by held templates or by lost_to hypotheses indicates realization/predicate brittleness, not a stable teacher-supervised substrate.",
    }
    failures = [p for p in pairs if p["state"] != "both_correct"]
    by_t = collections.defaultdict(list)
    for p in failures:
        by_t[f"{p['context_template_id']:02d}:{p['template_phrase']}"] .append(p)
    for k, ps in sorted(by_t.items(), key=lambda kv: (-len(kv[1]), kv[0]))[:10]:
        out["failure_examples_by_template"][k] = ps[:5]
    OUT.write_text(json.dumps(out, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")
    md = ["# research teacher failure postanalysis", "", f"Pairs: {len(pairs)}", "", f"State counts: {dict(collections.Counter(p['state'] for p in pairs))}", "", "## Template both-correct rates", "", "| template | n | both-correct | states |", "|---|---:|---:|---|"]
    for k, v in out["by_template_id"].items():
        md.append(f"| {k} | {v['n']} | {v['both_correct']:.3f} | {v['state_counts']} |")
    md += ["", "## By template-first role", "", "| first role | n | both-correct | states |", "|---|---:|---:|---|"]
    for k, v in out["by_template_first"].items():
        md.append(f"| {k} | {v['n']} | {v['both_correct']:.3f} | {v['state_counts']} |")
    md += ["", "## Interpretation", "", out["interpretation"]]
    MD.write_text("\n".join(md)+"\n", encoding="utf-8")
    print(json.dumps({"status": out["status"], "n_pairs": len(pairs), "state_counts": out["state_counts"], "worst_templates": list(out["failure_examples_by_template"].keys())[:5], "out": str(OUT), "md": str(MD)}, indent=2), flush=True)

if __name__ == "__main__":
    main()
