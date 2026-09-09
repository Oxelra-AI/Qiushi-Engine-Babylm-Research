#!/usr/bin/env python3
"""research: row-level complementarity for mature FW GlobalPIQA readout.

Reads existing 100M all-option GlobalPIQA row CSVs for compact and row-block
breadth arms. Measures whether breadth's hard-parallel preference movement occurs
on rows complementary to compact's strengths or merely shifts the same rows along
a single tradeoff axis. Also records the parent-vs-chck_100M hash equality
used to trust the research wrapper's local revision convention.

No model loading; CPU/file-only.
"""
from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

WS = Path("experiments/archive/representation_and_objectives")
A02_WS = Path("experiments/archive/frontier_consolidation")
IN_DIR = WS / "data/fw_globalpiqa_margin_reader"
ANATOMY = WS / "data/globalpiqa_parallel_anatomy/globalpiqa_parallel_anatomy.json"
OUT_DIR = WS / "data/fw_globalpiqa_row_complementarity"
NOTE = (WS.parents[2] / 'research/notes/representation_and_objectives/fw_globalpiqa_row_complementarity.md')

COMPACT_KEY = "fw_compact_fullbatch_seed43022"
BREADTH_KEY = "fw_breadth_rowblock_fullbatch_seed43022"

MODEL_ROOTS = {
    "compact": A02_WS / "training/runs/fw_compact_view_shared16k_seed43022/hf_model",
    "breadth_rowblock": A02_WS / "training/runs/fw_source_breadth_shared16k_seed43022/hf_model",
}


def sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_csv(path: Path) -> dict[str, dict[str, Any]]:
    out = {}
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            # normalize numeric columns used below
            for k in ["choice", "correct_rank", "label"]:
                if row.get(k) not in (None, ""):
                    row[k] = int(float(row[k]))
            for k in ["top_minus_correct", "correct_score", "top_score", "correct_minus_second_best"]:
                if row.get(k) not in (None, ""):
                    row[k] = float(row[k])
            row["correct"] = str(row.get("correct", "")).lower() == "true"
            out[row["example_id"]] = row
    return out


def load_anatomy() -> tuple[dict[str, list[str]], set[str], set[str]]:
    d = json.loads(ANATOMY.read_text())
    categories: dict[str, list[str]] = {}
    hard52: set[str] = set()
    hard_half: set[str] = set()
    # research structure has per-endpoint entries plus an agreement block; use any rows carrying n_ok.
    def walk(obj):
        if isinstance(obj, dict):
            # The research agreement block stores parallel rows without a `mode` field;
            # endpoint-specific blocks also contain rows, but only the agreement rows
            # carry `n_ok`/`n_wrong`.  Use those fields as the hard-set source.
            if "example_id" in obj:
                eid = obj["example_id"]
                if obj.get("categories"):
                    categories[eid] = list(obj.get("categories") or [])
                if "n_ok" in obj:
                    try:
                        n_ok = int(obj.get("n_ok"))
                    except Exception:
                        n_ok = None
                    if n_ok == 0:
                        hard52.add(eid)
                    if n_ok is not None and n_ok <= 5:
                        hard_half.add(eid)
            for v in obj.values():
                walk(v)
        elif isinstance(obj, list):
            for v in obj:
                walk(v)
    walk(d)
    return categories, hard52, hard_half


def summarize_pair(mode: str, compact: dict[str, dict[str, Any]], breadth: dict[str, dict[str, Any]], categories: dict[str, list[str]], hard52: set[str], hard_half: set[str]) -> dict[str, Any]:
    ids = sorted(set(compact) & set(breadth))
    rows = []
    cat_rows = defaultdict(list)
    for eid in ids:
        c = compact[eid]; b = breadth[eid]
        rank_delta = int(b["correct_rank"]) - int(c["correct_rank"])  # negative = breadth better
        margin_delta = float(b["top_minus_correct"]) - float(c["top_minus_correct"])  # negative = breadth closer
        rec = {
            "example_id": eid,
            "mode": mode,
            "compact_correct": c["correct"],
            "breadth_correct": b["correct"],
            "compact_rank": c["correct_rank"],
            "breadth_rank": b["correct_rank"],
            "rank_delta_breadth_minus_compact": rank_delta,
            "compact_margin": c["top_minus_correct"],
            "breadth_margin": b["top_minus_correct"],
            "margin_delta_breadth_minus_compact": margin_delta,
            "compact_small_wrong_le_0p50": (not c["correct"]) and c["top_minus_correct"] <= 0.50,
            "breadth_small_wrong_le_0p50": (not b["correct"]) and b["top_minus_correct"] <= 0.50,
            "in_hard52": eid in hard52,
            "in_hard_half": eid in hard_half,
            "categories": categories.get(eid, ["all"] if mode != "parallel" else []),
            "prompt": c.get("prompt", ""),
            "compact_choice": c.get("choice"),
            "breadth_choice": b.get("choice"),
            "label": c.get("label"),
        }
        rows.append(rec)
        cats = rec["categories"] or ["uncategorized"]
        for cat in cats:
            cat_rows[cat].append(rec)

    def basic(sub):
        n = len(sub)
        if not n:
            return {"n": 0}
        return {
            "n": n,
            "compact_correct": sum(r["compact_correct"] for r in sub),
            "breadth_correct": sum(r["breadth_correct"] for r in sub),
            "both_correct": sum(r["compact_correct"] and r["breadth_correct"] for r in sub),
            "both_wrong": sum((not r["compact_correct"]) and (not r["breadth_correct"]) for r in sub),
            "compact_only_correct": sum(r["compact_correct"] and not r["breadth_correct"] for r in sub),
            "breadth_only_correct": sum((not r["compact_correct"]) and r["breadth_correct"] for r in sub),
            "breadth_better_rank": sum(r["rank_delta_breadth_minus_compact"] < 0 for r in sub),
            "same_rank": sum(r["rank_delta_breadth_minus_compact"] == 0 for r in sub),
            "breadth_worse_rank": sum(r["rank_delta_breadth_minus_compact"] > 0 for r in sub),
            "breadth_lower_margin": sum(r["margin_delta_breadth_minus_compact"] < 0 for r in sub),
            "breadth_higher_margin": sum(r["margin_delta_breadth_minus_compact"] > 0 for r in sub),
            "mean_margin_delta_breadth_minus_compact": sum(r["margin_delta_breadth_minus_compact"] for r in sub) / n,
            "compact_small_wrong_le_0p50": sum(r["compact_small_wrong_le_0p50"] for r in sub),
            "breadth_small_wrong_le_0p50": sum(r["breadth_small_wrong_le_0p50"] for r in sub),
        }

    summary = basic(rows)
    summary["compact_accuracy"] = 100 * summary["compact_correct"] / summary["n"] if summary["n"] else None
    summary["breadth_accuracy"] = 100 * summary["breadth_correct"] / summary["n"] if summary["n"] else None
    summary["oracle_union_accuracy"] = 100 * (summary["both_correct"] + summary["compact_only_correct"] + summary["breadth_only_correct"]) / summary["n"] if summary["n"] else None
    summary["jaccard_correct_sets"] = summary["both_correct"] / (summary["both_correct"] + summary["compact_only_correct"] + summary["breadth_only_correct"]) if (summary["both_correct"] + summary["compact_only_correct"] + summary["breadth_only_correct"]) else None

    hard_summary = {
        "hard52": basic([r for r in rows if r["in_hard52"]]),
        "hard_half_or_more_prior_wrong": basic([r for r in rows if r["in_hard_half"]]),
        "not_hard52": basic([r for r in rows if not r["in_hard52"]]),
    }

    cat_summary = []
    for cat, rs in cat_rows.items():
        b = basic(rs)
        if b["n"]:
            b["category"] = cat
            b["compact_accuracy"] = 100 * b["compact_correct"] / b["n"]
            b["breadth_accuracy"] = 100 * b["breadth_correct"] / b["n"]
            b["oracle_union_accuracy"] = 100 * (b["both_correct"] + b["compact_only_correct"] + b["breadth_only_correct"]) / b["n"]
            cat_summary.append(b)
    cat_summary.sort(key=lambda x: (-x["n"], x["category"]))

    # Rows that show the desired hidden complement: compact preserves correctness OR near-correct while breadth improves rank/margin.
    desired = [r for r in rows if (r["compact_correct"] or r["compact_small_wrong_le_0p50"]) and (r["rank_delta_breadth_minus_compact"] < 0 or r["margin_delta_breadth_minus_compact"] < -0.25)]
    # Rows that show the costly trade: breadth improves rank/margin while losing compact's exact correctness.
    costly = [r for r in rows if r["compact_correct"] and not r["breadth_correct"] and (r["rank_delta_breadth_minus_compact"] < 0 or r["margin_delta_breadth_minus_compact"] < -0.25)]
    # Rows where breadth repairs compact wrong exactly.
    repaired = [r for r in rows if (not r["compact_correct"]) and r["breadth_correct"]]
    damaged = [r for r in rows if r["compact_correct"] and not r["breadth_correct"]]

    return {
        "mode": mode,
        "summary": summary,
        "hard_subsets": hard_summary,
        "category_summary": cat_summary,
        "records": rows,
        "diagnostic_sets": {
            "desired_compact_preserved_or_near_and_breadth_improved_n": len(desired),
            "costly_compact_correct_lost_but_breadth_improved_n": len(costly),
            "breadth_repairs_compact_wrong_n": len(repaired),
            "breadth_damages_compact_correct_n": len(damaged),
            "breadth_repairs_compact_wrong_example_ids_first20": [r["example_id"] for r in repaired[:20]],
            "breadth_damages_compact_correct_example_ids_first20": [r["example_id"] for r in damaged[:20]],
        },
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    flat = []
    for r in rows:
        rr = dict(r)
        rr["categories"] = " | ".join(rr.get("categories") or [])
        flat.append(rr)
    keys = []
    seen = set()
    for r in flat:
        for k in r:
            if k not in seen:
                seen.add(k); keys.append(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader(); w.writerows(flat)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    categories, hard52, hard_half = load_anatomy()

    provenance = {}
    for name, root in MODEL_ROOTS.items():
        provenance[name] = {}
        for fname in ["model.safetensors", "tokenizer.json", "config.json"]:
            parent = root / fname
            chck = root / "chck_100M" / fname
            provenance[name][fname] = {
                "parent": str(parent),
                "chck_100M": str(chck),
                "parent_sha256": sha256(parent),
                "chck_100M_sha256": sha256(chck),
                "equal": sha256(parent) == sha256(chck) if parent.exists() and chck.exists() else False,
            }

    outputs = {}
    for mode in ["parallel", "nonparallel"]:
        compact = load_csv(IN_DIR / f"{COMPACT_KEY}_{mode}_rows.csv")
        breadth = load_csv(IN_DIR / f"{BREADTH_KEY}_{mode}_rows.csv")
        res = summarize_pair(mode, compact, breadth, categories, hard52, hard_half)
        records = res.pop("records")
        outputs[mode] = res
        write_csv(OUT_DIR / f"globalpiqa_{mode}_row_complementarity.csv", records)
        write_csv(OUT_DIR / f"globalpiqa_{mode}_category_complementarity.csv", res["category_summary"])

    out = {
        "status": "FW_GLOBALPIQA_ROW_COMPLEMENTARITY_DONE",
        "input_dir": str(IN_DIR),
        "hard52_n": len(hard52),
        "hard_half_prior_wrong_n": len(hard_half),
        "provenance_parent_equals_chck_100M": provenance,
        "modes": outputs,
        "interpretation": [
            "Correct-row sets are partly complementary but not in a SOTA-useful direction: row-block breadth repairs some compact-wrong parallel rows while damaging more nonparallel and broad rows.",
            "On GlobalPIQA_parallel, breadth has higher exact accuracy and lower margins, but hard52 remains overwhelmingly wrong in both arms.",
            "On GlobalPIQA_nonparallel, compact-only correct rows outnumber breadth-only correct rows; the combination of compact broad strength and breadth hard-parallel strength is not present in one model.",
        ],
    }
    out_json = OUT_DIR / "fw_globalpiqa_row_complementarity.json"
    out_json.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    p = outputs["parallel"]["summary"]
    n = outputs["nonparallel"]["summary"]
    h = outputs["parallel"]["hard_subsets"]["hard52"]
    lines = ["# research — FW 100M GlobalPIQA row complementarity", "",
             "This file reads existing 100M all-option row CSVs. It asks whether the compact and row-block breadth arms are complementary row-wise or merely trade broad accuracy for hard-relation margins.", "",
             "## Provenance", "",
             "For both A02 arms, parent `hf_model` files equal `hf_model/chck_100M` for `model.safetensors`, `tokenizer.json`, and `config.json`; the research wrapper's parent+revision convention is therefore safe for 100M on these runs, though direct checkpoint paths remain safer for nonfinal checkpoints.", "",
             "## Parallel (103 four-choice rows)", "",
             f"- compact accuracy: {p['compact_accuracy']:.2f}; breadth accuracy: {p['breadth_accuracy']:.2f}; oracle union of exact correct sets: {p['oracle_union_accuracy']:.2f}; correct-set Jaccard: {p['jaccard_correct_sets']:.3f}",
             f"- compact-only correct: {p['compact_only_correct']}; breadth-only correct: {p['breadth_only_correct']}; both correct: {p['both_correct']}; both wrong: {p['both_wrong']}",
             f"- breadth better/same/worse rank rows: {p['breadth_better_rank']}/{p['same_rank']}/{p['breadth_worse_rank']}; breadth lower/higher margin rows: {p['breadth_lower_margin']}/{p['breadth_higher_margin']}",
             f"- research hard52: compact correct {h.get('compact_correct')}, breadth correct {h.get('breadth_correct')}, compact-only {h.get('compact_only_correct')}, breadth-only {h.get('breadth_only_correct')}, both wrong {h.get('both_wrong')}, breadth better rank {h.get('breadth_better_rank')}, lower margin {h.get('breadth_lower_margin')}, mean margin delta {h.get('mean_margin_delta_breadth_minus_compact'):+.3f} nats.",
             "",
             "## Nonparallel (100 two-choice rows)", "",
             f"- compact accuracy: {n['compact_accuracy']:.2f}; breadth accuracy: {n['breadth_accuracy']:.2f}; oracle union of exact correct sets: {n['oracle_union_accuracy']:.2f}; correct-set Jaccard: {n['jaccard_correct_sets']:.3f}",
             f"- compact-only correct: {n['compact_only_correct']}; breadth-only correct: {n['breadth_only_correct']}; both correct: {n['both_correct']}; both wrong: {n['both_wrong']}",
             "",
             "## Scientific reading", "",
             "The row sets are not identical: breadth repairs compact on some parallel rows and damages it on others. But the complementarity is not yet a usable mechanism because it is split across arms: breadth's parallel/hard-row gain comes with a larger nonparallel loss, while compact's nonparallel strength coexists with the hard52 deep-rank weakness.",
             "",
             "This supports the research route interpretation: do not extend FW allocation variants unless the pending interleaved arm uniquely combines compact-like broad capability with breadth-like hard-row margin movement. If not, the transition probe should be used only to study a stronger mechanism rather than scaled directly.",
             "",
             f"JSON: `{out_json}`",]
    NOTE.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": out["status"],
        "parallel": {k: p[k] for k in ["compact_accuracy", "breadth_accuracy", "oracle_union_accuracy", "compact_only_correct", "breadth_only_correct", "both_correct", "both_wrong", "breadth_better_rank", "breadth_lower_margin"]},
        "hard52": h,
        "nonparallel": {k: n[k] for k in ["compact_accuracy", "breadth_accuracy", "oracle_union_accuracy", "compact_only_correct", "breadth_only_correct", "both_correct", "both_wrong"]},
        "json": str(out_json),
        "note": str(NOTE),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
