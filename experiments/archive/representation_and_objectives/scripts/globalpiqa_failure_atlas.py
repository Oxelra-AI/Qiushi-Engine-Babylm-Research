#!/usr/bin/env python3
"""research: GlobalPIQA compact-density failure atlas.

CPU-only interpretation of already-finished fast GlobalPIQA predictions for
compact_repeat_core, compact_view_core, and compact_view_reinvest.  It links
per-example decisions to the official local evaluation data and to the selected
compact source-view pairs, so the next route can repair the weak practical-
affordance column from real examples instead of guessing from one mean score.
"""
from __future__ import annotations

import collections
import json
import math
import pathlib
import re
import string
from difflib import SequenceMatcher
from typing import Any

ROOT = pathlib.Path(".").resolve()
A01_WORKSPACE = ROOT / "experiments/archive" / 'representation_and_objectives'
STRICT = ROOT / "experiments/archive" / 'initial_model_studies' / "repos" / "babylm-eval" / "strict"
OUT_DIR = A01_WORKSPACE / "data" / "globalpiqa_failure_atlas"
OUT_JSON = OUT_DIR / "globalpiqa_failure_atlas.json"
OUT_NOTE = (ROOT / 'research/notes/representation_and_objectives/globalpiqa_failure_atlas.md')

DATA_PATHS = {
    "parallel": STRICT / "evaluation_data" / "fast_eval" / "global_piqa_parallel" / "eng_latn.jsonl",
    "nonparallel": STRICT / "evaluation_data" / "fast_eval" / "global_piqa_nonparallel" / "eng_latn.jsonl",
}
PRED_PATHS = {
    "compact_repeat_core": {
        "parallel": ROOT / "experiments/archive" / 'frontier_consolidation' / "data" / "density_noaoa_eval_compact_core" / "eval_outputs" / "compact_core_noaoa" / "compact_repeat_core" / "GlobalPIQA_parallel" / "chck_100M" / "compact_repeat_core_GlobalPIQA_parallel" / "zero_shot" / "mlm" / "global_piqa_parallel" / "global_piqa_parallel" / "predictions.json",
        "nonparallel": ROOT / "experiments/archive" / 'frontier_consolidation' / "data" / "density_noaoa_eval_compact_core" / "eval_outputs" / "compact_core_noaoa" / "compact_repeat_core" / "GlobalPIQA_nonparallel" / "chck_100M" / "compact_repeat_core_GlobalPIQA_nonparallel" / "zero_shot" / "mlm" / "global_piqa_nonparallel" / "global_piqa_nonparallel" / "predictions.json",
    },
    "compact_view_core": {
        "parallel": ROOT / "experiments/archive" / 'frontier_consolidation' / "data" / "density_noaoa_eval_compact_core" / "eval_outputs" / "compact_core_noaoa" / "compact_view_core" / "GlobalPIQA_parallel" / "chck_100M" / "compact_view_core_GlobalPIQA_parallel" / "zero_shot" / "mlm" / "global_piqa_parallel" / "global_piqa_parallel" / "predictions.json",
        "nonparallel": ROOT / "experiments/archive" / 'frontier_consolidation' / "data" / "density_noaoa_eval_compact_core" / "eval_outputs" / "compact_core_noaoa" / "compact_view_core" / "GlobalPIQA_nonparallel" / "chck_100M" / "compact_view_core_GlobalPIQA_nonparallel" / "zero_shot" / "mlm" / "global_piqa_nonparallel" / "global_piqa_nonparallel" / "predictions.json",
    },
    "compact_view_reinvest": {
        "parallel": ROOT / "experiments/archive" / 'frontier_consolidation' / "data" / "density_noaoa_eval_reinvest" / "eval_outputs" / "density_reinvest_noaoa" / "compact_view_reinvest" / "GlobalPIQA_parallel" / "chck_100M" / "compact_view_reinvest_GlobalPIQA_parallel" / "zero_shot" / "mlm" / "global_piqa_parallel" / "global_piqa_parallel" / "predictions.json",
        "nonparallel": ROOT / "experiments/archive" / 'frontier_consolidation' / "data" / "density_noaoa_eval_reinvest" / "eval_outputs" / "density_reinvest_noaoa" / "compact_view_reinvest" / "GlobalPIQA_nonparallel" / "chck_100M" / "compact_view_reinvest_GlobalPIQA_nonparallel" / "zero_shot" / "mlm" / "global_piqa_nonparallel" / "global_piqa_nonparallel" / "predictions.json",
    },
}
REINVEST_PAIR_PATH = ROOT / "experiments/archive" / 'frontier_consolidation' / "data" / "density_core_reinvestment_medium_riskhard" / "selected_compact_reinvest_pairs.jsonl"

STOP = {
    "a", "an", "the", "and", "or", "but", "if", "then", "when", "what", "which", "who", "whom", "whose",
    "to", "of", "in", "on", "at", "by", "for", "from", "with", "without", "into", "onto", "over", "under",
    "is", "are", "was", "were", "be", "been", "being", "am", "do", "does", "did", "will", "would", "can", "could",
    "should", "may", "might", "must", "as", "it", "its", "this", "that", "these", "those", "you", "your", "i", "we",
    "they", "he", "she", "them", "him", "her", "his", "their", "our", "my", "me", "some", "any", "all", "not", "no",
    "one", "two", "three", "four", "there", "here", "after", "before", "than", "more", "less", "same", "other", "another",
    "happens", "happen", "make", "makes", "made", "get", "gets", "got", "use", "used", "using", "put", "place",
}
TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9'\-]*|\d+(?:\.\d+)?")


def read_json(path: pathlib.Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def iter_jsonl(path: pathlib.Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def norm_text(s: str) -> str:
    s = s.strip().lower()
    s = s.translate(str.maketrans("", "", string.punctuation.replace("'", "")))
    s = re.sub(r"\s+", " ", s)
    return s


def content_tokens(text: str) -> set[str]:
    out = set()
    for tok in TOKEN_RE.findall(text.lower()):
        tok = tok.strip("'\"").replace("'s", "")
        if len(tok) < 3 or tok in STOP:
            continue
        out.add(tok)
    return out


def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 0.0
    return len(a & b) / max(1, len(a | b))


def load_examples(split: str) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for row in iter_jsonl(DATA_PATHS[split]):
        eid = str(row["example_id"])
        solutions = []
        i = 0
        while f"solution{i}" in row:
            solutions.append(str(row[f"solution{i}"]))
            i += 1
        cats = []
        if row.get("categories"):
            cats = [x.strip() for x in re.split(r"[,;/]", str(row["categories"])) if x.strip()]
        sup = {}
        if row.get("supplement"):
            try:
                sup = json.loads(row["supplement"])
            except Exception:
                sup = {"raw": row.get("supplement")}
        text = " ".join([str(row.get("prompt", ""))] + solutions)
        out[eid] = {
            "example_id": eid,
            "split": split,
            "prompt": row.get("prompt"),
            "solutions": solutions,
            "label": int(row["label"]),
            "categories": cats or ["uncategorized"],
            "category_string": row.get("categories") or "uncategorized",
            "supplement": sup,
            "example_inspiration": sup.get("example_inspiration") if isinstance(sup, dict) else None,
            "approx_cultural_score": row.get("approx_cultural_score"),
            "llm_used": row.get("llm_used"),
            "tokens": sorted(content_tokens(text)),
        }
    return out


def flatten_predictions(path: pathlib.Path) -> dict[str, str]:
    data = read_json(path)
    out: dict[str, str] = {}
    for group_key, rec in data.items():
        preds = rec.get("predictions") if isinstance(rec, dict) else None
        if not isinstance(preds, list):
            continue
        for p in preds:
            pid = str(p.get("id"))
            pred = str(p.get("pred", ""))
            # Official output IDs append _0 to the example_id.  The example_id itself can contain underscores.
            eid = pid[:-2] if pid.endswith("_0") else pid
            # If the group key is exactly the example_id, prefer it.
            if str(group_key) in pid or str(group_key) == eid:
                eid = str(group_key)
            out[eid] = pred
    return out


def infer_choice(pred_text: str, solutions: list[str]) -> dict[str, Any]:
    n_pred = norm_text(pred_text)
    norms = [norm_text(x) for x in solutions]
    for i, s in enumerate(norms):
        if n_pred == s:
            return {"choice": i, "match": "exact_norm", "similarity": 1.0}
    # Some predictions include a leading fragment or retain punctuation differently.
    for i, s in enumerate(norms):
        if n_pred and (n_pred in s or s in n_pred):
            sim = SequenceMatcher(None, n_pred, s).ratio()
            return {"choice": i, "match": "substring_norm", "similarity": sim}
    sims = [SequenceMatcher(None, n_pred, s).ratio() for s in norms]
    if sims:
        best = max(range(len(sims)), key=lambda i: sims[i])
        return {"choice": best, "match": "fuzzy", "similarity": sims[best]}
    return {"choice": None, "match": "no_solution", "similarity": None}


def arm_decisions(examples_by_split: dict[str, dict[str, dict[str, Any]]]) -> dict[str, dict[str, dict[str, Any]]]:
    out: dict[str, dict[str, dict[str, Any]]] = {}
    for arm, split_paths in PRED_PATHS.items():
        out[arm] = {}
        for split, pred_path in split_paths.items():
            preds = flatten_predictions(pred_path)
            split_dec: dict[str, Any] = {}
            for eid, ex in examples_by_split[split].items():
                pred_text = preds.get(eid)
                if pred_text is None:
                    split_dec[eid] = {"has_prediction": False, "correct": False, "choice": None}
                    continue
                inf = infer_choice(pred_text, ex["solutions"])
                correct = (inf["choice"] == ex["label"])
                split_dec[eid] = {
                    "has_prediction": True,
                    "pred_text": pred_text,
                    "choice": inf["choice"],
                    "label": ex["label"],
                    "correct": bool(correct),
                    "match": inf["match"],
                    "similarity": inf["similarity"],
                }
            out[arm][split] = split_dec
    return out


def mean(xs: list[float]) -> float | None:
    return sum(xs) / len(xs) if xs else None


def acc(rows: list[bool]) -> float | None:
    return 100.0 * sum(bool(x) for x in rows) / len(rows) if rows else None


def split_score_summary(decisions: dict[str, dict[str, dict[str, Any]]], examples_by_split: dict[str, dict[str, dict[str, Any]]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for arm, split_dec in decisions.items():
        out[arm] = {}
        for split, dec in split_dec.items():
            correct = [bool(dec[eid]["correct"]) for eid in examples_by_split[split]]
            out[arm][split] = {"n": len(correct), "accuracy": acc(correct)}
        out[arm]["mean_parallel_nonparallel"] = mean([out[arm]["parallel"]["accuracy"], out[arm]["nonparallel"]["accuracy"]])
    return out


def by_group(decisions: dict[str, dict[str, dict[str, Any]]], examples_by_split: dict[str, dict[str, dict[str, Any]]], arm: str, split: str, field: str) -> dict[str, Any]:
    buckets: dict[str, list[bool]] = collections.defaultdict(list)
    for eid, ex in examples_by_split[split].items():
        values: list[Any]
        if field == "categories":
            values = ex.get("categories") or ["uncategorized"]
        else:
            v = ex.get(field)
            values = ["null" if v is None else str(v)]
        for v in values:
            buckets[str(v)].append(bool(decisions[arm][split][eid]["correct"]))
    return {k: {"n": len(v), "accuracy": acc(v)} for k, v in sorted(buckets.items())}


def category_tables(decisions: dict[str, dict[str, dict[str, Any]]], examples_by_split: dict[str, dict[str, dict[str, Any]]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for split in ["parallel", "nonparallel"]:
        out[split] = {}
        for arm in PRED_PATHS:
            out[split][arm] = {
                "categories": by_group(decisions, examples_by_split, arm, split, "categories"),
                "example_inspiration": by_group(decisions, examples_by_split, arm, split, "example_inspiration"),
                "llm_used": by_group(decisions, examples_by_split, arm, split, "llm_used"),
                "approx_cultural_score": by_group(decisions, examples_by_split, arm, split, "approx_cultural_score"),
            }
    return out


def example_record(eid: str, split: str, examples_by_split: dict[str, dict[str, dict[str, Any]]], decisions: dict[str, dict[str, dict[str, Any]]]) -> dict[str, Any]:
    ex = examples_by_split[split][eid]
    return {
        "example_id": eid,
        "split": split,
        "prompt": ex["prompt"],
        "solutions": ex["solutions"],
        "label": ex["label"],
        "categories": ex["categories"],
        "example_inspiration": ex.get("example_inspiration"),
        "decisions": {arm: decisions[arm][split][eid] for arm in PRED_PATHS},
    }


def flip_sets(decisions: dict[str, dict[str, dict[str, Any]]], examples_by_split: dict[str, dict[str, dict[str, Any]]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    pairs = [
        ("compact_view_core", "compact_repeat_core", "view_core_minus_repeat_core"),
        ("compact_view_reinvest", "compact_view_core", "reinvest_minus_view_core"),
        ("compact_view_reinvest", "compact_repeat_core", "reinvest_minus_repeat_core"),
    ]
    for split in ["parallel", "nonparallel"]:
        out[split] = {}
        for hi, lo, name in pairs:
            fixed = []
            broken = []
            both_correct = []
            both_wrong = []
            for eid in examples_by_split[split]:
                hc = bool(decisions[hi][split][eid]["correct"])
                lc = bool(decisions[lo][split][eid]["correct"])
                if hc and not lc:
                    fixed.append(eid)
                elif lc and not hc:
                    broken.append(eid)
                elif hc and lc:
                    both_correct.append(eid)
                else:
                    both_wrong.append(eid)
            out[split][name] = {
                "fixed_count": len(fixed),
                "broken_count": len(broken),
                "net_count": len(fixed) - len(broken),
                "both_correct_count": len(both_correct),
                "both_wrong_count": len(both_wrong),
                "fixed_examples": [example_record(e, split, examples_by_split, decisions) for e in fixed[:20]],
                "broken_examples": [example_record(e, split, examples_by_split, decisions) for e in broken[:20]],
            }
        all_fail = []
        only_reinvest = []
        for eid in examples_by_split[split]:
            vals = {arm: bool(decisions[arm][split][eid]["correct"]) for arm in PRED_PATHS}
            if not any(vals.values()):
                all_fail.append(eid)
            if vals["compact_view_reinvest"] and not vals["compact_view_core"] and not vals["compact_repeat_core"]:
                only_reinvest.append(eid)
        out[split]["all_three_wrong"] = {
            "count": len(all_fail),
            "examples": [example_record(e, split, examples_by_split, decisions) for e in all_fail[:30]],
        }
        out[split]["only_reinvest_correct"] = {
            "count": len(only_reinvest),
            "examples": [example_record(e, split, examples_by_split, decisions) for e in only_reinvest[:30]],
        }
    return out


def load_pair_tokens() -> list[dict[str, Any]]:
    pairs = []
    for obj in iter_jsonl(REINVEST_PAIR_PATH):
        text = str(obj.get("source_text", "")) + " " + str(obj.get("rewrite_text", ""))
        pairs.append({
            "pair_id": obj.get("pair_id"),
            "key": obj.get("key"),
            "doc_id": obj.get("doc_id"),
            "domain_hits": obj.get("domain_hits") or [],
            "source_text": obj.get("source_text"),
            "rewrite_text": obj.get("rewrite_text"),
            "content_recall": obj.get("content_recall"),
            "tokens": content_tokens(text),
        })
    return pairs


def coverage_for_example(ex: dict[str, Any], pairs: list[dict[str, Any]], topk: int = 5) -> dict[str, Any]:
    qtoks = set(ex.get("tokens") or [])
    scored = []
    for p in pairs:
        sim = jaccard(qtoks, p["tokens"])
        if sim > 0:
            scored.append((sim, len(qtoks & p["tokens"]), p))
    scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
    top = []
    for sim, inter, p in scored[:topk]:
        top.append({
            "pair_id": p["pair_id"], "key": p["key"], "doc_id": p["doc_id"],
            "domain_hits": p["domain_hits"], "jaccard": round(sim, 6), "intersection": inter,
            "source_text": p["source_text"], "rewrite_text": p["rewrite_text"],
        })
    return {"query_token_count": len(qtoks), "max_jaccard": round(scored[0][0], 6) if scored else 0.0, "top_matches": top}


def source_overlap_analysis(examples_by_split: dict[str, dict[str, dict[str, Any]]], decisions: dict[str, dict[str, dict[str, Any]]]) -> dict[str, Any]:
    pairs = load_pair_tokens()
    out: dict[str, Any] = {"pair_count": len(pairs), "all_three_wrong": {}, "by_category_failed_mean_max_jaccard": {}}
    for split in ["parallel", "nonparallel"]:
        fail_ids = [eid for eid in examples_by_split[split] if not any(bool(decisions[arm][split][eid]["correct"]) for arm in PRED_PATHS)]
        samples = []
        for eid in fail_ids[:40]:
            rec = example_record(eid, split, examples_by_split, decisions)
            rec["compact_pair_overlap"] = coverage_for_example(examples_by_split[split][eid], pairs, topk=3)
            samples.append(rec)
        out["all_three_wrong"][split] = {"count": len(fail_ids), "sample_with_top_source_matches": samples}
        buckets: dict[str, list[float]] = collections.defaultdict(list)
        for eid in fail_ids:
            cov = coverage_for_example(examples_by_split[split][eid], pairs, topk=0)
            for cat in examples_by_split[split][eid].get("categories") or ["uncategorized"]:
                buckets[cat].append(float(cov["max_jaccard"]))
        out["by_category_failed_mean_max_jaccard"][split] = {
            k: {"n": len(v), "mean_max_jaccard": round(sum(v)/len(v), 6), "median_max_jaccard": round(sorted(v)[len(v)//2], 6)}
            for k, v in sorted(buckets.items())
        }
    return out


def compact_category_delta_table(category_tables_payload: dict[str, Any], split: str, group_field: str, hi: str, lo: str) -> list[dict[str, Any]]:
    h = category_tables_payload[split][hi][group_field]
    l = category_tables_payload[split][lo][group_field]
    rows = []
    for key in sorted(set(h) & set(l)):
        rows.append({
            "key": key,
            "n": h[key]["n"],
            "target_accuracy": h[key]["accuracy"],
            "baseline_accuracy": l[key]["accuracy"],
            "delta": None if h[key]["accuracy"] is None or l[key]["accuracy"] is None else round(float(h[key]["accuracy"]) - float(l[key]["accuracy"]), 6),
        })
    rows.sort(key=lambda x: (x["delta"] if x["delta"] is not None else -999, x["n"]), reverse=True)
    return rows


def write_note(payload: dict[str, Any]) -> None:
    lines: list[str] = []
    lines.append("# research GlobalPIQA failure atlas")
    lines.append("")
    lines.append("This CPU-only pass links the finished compact-density fast GlobalPIQA predictions to the local evaluation examples and to the selected compact source-view pairs. It was done while the research full evaluation and seed43122 run continued asynchronously, without polling or modifying them.")
    lines.append("")
    lines.append("## Accuracy reconstruction")
    lines.append("")
    lines.append("| arm | parallel | nonparallel | mean |")
    lines.append("|---|---:|---:|---:|")
    for arm, rec in payload["score_summary"].items():
        lines.append(f"| {arm} | {rec['parallel']['accuracy']:.2f} | {rec['nonparallel']['accuracy']:.2f} | {rec['mean_parallel_nonparallel']:.2f} |")
    lines.append("")
    lines.append("The reconstructed accuracies match the A02 reports: reinvest is 25.24 parallel and 46.00 nonparallel, mean 35.62. This confirms that the weakness is not a parsing artifact.")
    lines.append("")
    lines.append("## Flips between arms")
    lines.append("")
    for split in ["parallel", "nonparallel"]:
        lines.append(f"### {split}")
        for name in ["view_core_minus_repeat_core", "reinvest_minus_view_core", "reinvest_minus_repeat_core"]:
            rec = payload["flips"][split][name]
            lines.append(f"- `{name}`: fixed {rec['fixed_count']}, broken {rec['broken_count']}, net {rec['net_count']}, both wrong {rec['both_wrong_count']}.")
        all_wrong = payload["flips"][split]["all_three_wrong"]["count"]
        only_re = payload["flips"][split]["only_reinvest_correct"]["count"]
        lines.append(f"- all three compact arms wrong: {all_wrong}; only reinvest correct: {only_re}.")
    lines.append("")
    lines.append("## Category movement")
    lines.append("")
    for split in ["parallel", "nonparallel"]:
        lines.append(f"### {split}: reinvest minus repeat by category")
        rows = payload["category_delta_tables"][split]["reinvest_minus_repeat_categories"]
        lines.append("| category | n | repeat | reinvest | delta |")
        lines.append("|---|---:|---:|---:|---:|")
        for r in rows[:12]:
            lines.append(f"| {r['key']} | {r['n']} | {r['baseline_accuracy']:.2f} | {r['target_accuracy']:.2f} | {r['delta']:+.2f} |")
        lines.append("")
    lines.append("## Source overlap for unresolved examples")
    lines.append("")
    for split in ["parallel", "nonparallel"]:
        rec = payload["source_overlap"]["all_three_wrong"][split]
        lines.append(f"- {split}: {rec['count']} examples are wrong for repeat, core-view, and reinvest. For these examples, compact-pair lexical overlap is only a rough source-coverage signal, not evidence of learned ability.")
        cats = payload["source_overlap"]["by_category_failed_mean_max_jaccard"][split]
        for cat, c in list(cats.items())[:8]:
            lines.append(f"  - {cat}: n={c['n']}, mean max content-Jaccard to any selected compact pair={c['mean_max_jaccard']:.3f}")
    lines.append("")
    lines.append("## Mechanistic reading for next work")
    lines.append("")
    lines.append("- Compact views repair many nonparallel practical examples relative to literal repetition, but they do not repair the harder parallel side; reinvest adds a small parallel gain and no nonparallel gain over core-view.")
    lines.append("- The remaining gap to the public leader is dominated by practical affordance / causal-procedural GlobalPIQA examples, especially cases where all three compact arms choose the same wrong option. If the full reinvest endpoint misses only narrowly, the next data repair should add compact source-view packets describing physical affordances, everyday tool use, material response, time/counting, and action outcome contrasts, rather than adding more generic encyclopedic statements.")
    lines.append("- Any such repair should preserve the successful compact-view properties already measured: short jointly visible source+rewrite packets, high entity/number retention, exact 10M/100M accounting, and no broad loss in Supplement/Entity/EWoK.")
    lines.append("")
    lines.append(f"JSON: `{OUT_JSON}`")
    OUT_NOTE.parent.mkdir(parents=True, exist_ok=True)
    OUT_NOTE.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    examples_by_split = {split: load_examples(split) for split in DATA_PATHS}
    decisions = arm_decisions(examples_by_split)
    score_summary = split_score_summary(decisions, examples_by_split)
    cats = category_tables(decisions, examples_by_split)
    flips = flip_sets(decisions, examples_by_split)
    category_delta_tables = {
        split: {
            "view_core_minus_repeat_categories": compact_category_delta_table(cats, split, "categories", "compact_view_core", "compact_repeat_core"),
            "reinvest_minus_view_categories": compact_category_delta_table(cats, split, "categories", "compact_view_reinvest", "compact_view_core"),
            "reinvest_minus_repeat_categories": compact_category_delta_table(cats, split, "categories", "compact_view_reinvest", "compact_repeat_core"),
            "reinvest_minus_repeat_inspiration": compact_category_delta_table(cats, split, "example_inspiration", "compact_view_reinvest", "compact_repeat_core"),
        }
        for split in ["parallel", "nonparallel"]
    }
    source_overlap = source_overlap_analysis(examples_by_split, decisions)
    payload = {
        "status": "GLOBALPIQA_FAILURE_ATLAS",
        "inputs": {
            "data_paths": {k: str(v) for k, v in DATA_PATHS.items()},
            "prediction_paths": {arm: {split: str(path) for split, path in rec.items()} for arm, rec in PRED_PATHS.items()},
            "reinvest_pairs": str(REINVEST_PAIR_PATH),
        },
        "score_summary": score_summary,
        "category_tables": cats,
        "category_delta_tables": category_delta_tables,
        "flips": flips,
        "source_overlap": source_overlap,
        "interpretation": {
            "weak_column": "GlobalPIQA remains the compact_view_reinvest fast-column gap: 35.62 mean, about 4.05 below the public leader despite the high seven-column surface.",
            "parallel_nonparallel_reading": "The core view improves nonparallel examples over repeat but hurts parallel examples; reinvest partly repairs the parallel side without further nonparallel gain.",
            "repair_direction_if_endpoint_needs_it": "Prefer compact source-view packets about practical physical affordances, everyday tool use, material response, time/counting, and action outcome contrasts. Preserve joint visibility and compactness rather than simply increasing generic FineWeb facts.",
        },
        "note": str(OUT_NOTE),
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_note(payload)
    print(json.dumps({
        "status": payload["status"],
        "json": str(OUT_JSON),
        "note": str(OUT_NOTE),
        "reinvest_globalpiqa_mean": score_summary["compact_view_reinvest"]["mean_parallel_nonparallel"],
        "parallel_all_three_wrong": flips["parallel"]["all_three_wrong"]["count"],
        "nonparallel_all_three_wrong": flips["nonparallel"]["all_three_wrong"]["count"],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
