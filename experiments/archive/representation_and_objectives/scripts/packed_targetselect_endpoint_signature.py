#!/usr/bin/env python3
"""research: endpoint signature comparison for packed target-selective arms.

This is the scientific readout for the historical packed-geometry intervention.
It does not treat an aggregate score delta as sufficient.  It compares each
100M target-selective endpoint against the historical full compact_view_reinvest
reference and reports:

  * cheap7-style columns and deltas,
  * item-level transitions for Supplement, EWoK, COMPS, GlobalPIQA,
  * EWoK domain movement, especially the research source-own-adjacency relational
    domains: social-properties, physical-dynamics, spatial-relations,
    physical-relations.

Expected use after evaluation:
  python3 -B .../packed_targetselect_endpoint_signature.py \
    --full-json experiments/archive/representation_and_objectives/data/compact_triangle_noaoa_eval/per_target/compact_view_reinvest.json \
    --arm-json drop_abs=.../per_target/dropabs100.json \
    --arm-json drop_copied=.../per_target/dropcopied100.json
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import pathlib
from typing import Any

ROOT = pathlib.Path("experiments/archive/representation_and_objectives")
OFFICIAL_FULL = ROOT / "data/pristine_official_coordinate/babylm-eval/strict/evaluation_data/full_eval"
GLOBALPIQA_FULL = pathlib.Path("experiments/archive/initial_model_studies/repos/babylm-eval/strict/evaluation_data/full_eval")
DEFAULT_OUT = ROOT / "data/packed_targetselect_endpoint_signature"

COLUMNS = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA", "Reading"]
RELATIONAL = ["social-properties", "physical-dynamics", "spatial-relations", "physical-relations"]
ADJ_INDEPENDENT = ["material-properties", "social-interactions"]


def load_json(p: pathlib.Path) -> Any:
    return json.loads(p.read_text(encoding="utf-8"))


def task_score(payload: dict[str, Any], col: str) -> float | None:
    oo = payload.get("official_overall", {})
    if isinstance(oo, dict) and isinstance(oo.get("scores"), dict) and col in oo["scores"]:
        return oo["scores"].get(col)
    t = payload.get("tasks", {}).get(col)
    if not t:
        if col == "GlobalPIQA":
            a = task_score(payload, "GlobalPIQA_parallel")
            b = task_score(payload, "GlobalPIQA_nonparallel")
            return None if a is None or b is None else (float(a) + float(b)) / 2.0
        return None
    if col == "Reading":
        return (t.get("scores") or {}).get("Reading")
    return t.get("score")


def score_table(arms: dict[str, dict[str, Any]]) -> dict[str, Any]:
    scores: dict[str, dict[str, float | None]] = {}
    for name, payload in arms.items():
        scores[name] = {c: task_score(payload, c) for c in COLUMNS}
        vals = [scores[name][c] for c in COLUMNS]
        if all(v is not None for v in vals):
            scores[name]["cheap7_equal_mean"] = round(sum(float(v) for v in vals) / len(vals), 6)
        else:
            scores[name]["cheap7_equal_mean"] = None
    deltas: dict[str, Any] = {}
    names = [n for n in arms if n != "full"]
    for n in names:
        key = f"{n}_minus_full"
        deltas[key] = {}
        for c in COLUMNS + ["cheap7_equal_mean"]:
            a = scores[n].get(c)
            b = scores["full"].get(c)
            deltas[key][c] = None if a is None or b is None else round(float(a) - float(b), 6)
    if len(names) >= 2:
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                a, b = names[i], names[j]
                key = f"{a}_minus_{b}"
                deltas[key] = {}
                for c in COLUMNS + ["cheap7_equal_mean"]:
                    av = scores[a].get(c); bv = scores[b].get(c)
                    deltas[key][c] = None if av is None or bv is None else round(float(av) - float(bv), 6)
    return {"scores": scores, "deltas": deltas}


def pred_from_payload(payload: dict[str, Any], task: str) -> pathlib.Path:
    t = payload.get("tasks", {}).get(task)
    if isinstance(t, dict) and t.get("predictions"):
        p = pathlib.Path(t["predictions"])
        if p.exists():
            return p
    # Some older per-target JSONs only record output_dir.  Recover by glob.
    if isinstance(t, dict) and t.get("output_dir"):
        od = pathlib.Path(t["output_dir"])
        patterns = {
            "BLiMP": "*/**/zero_shot/mlm/blimp/*/predictions.json",
            "Supplement": "*/**/zero_shot/mlm/blimp/*/predictions.json",
            "EWoK": "*/**/zero_shot/mlm/ewok/*/predictions.json",
            "Entity": "*/**/zero_shot/mlm/entity_tracking/*/predictions.json",
            "COMPS": "*/**/zero_shot/mlm/comps/*/predictions.json",
            "GlobalPIQA_parallel": "*/**/zero_shot/mlm/global_piqa_parallel/*/predictions.json",
            "GlobalPIQA_nonparallel": "*/**/zero_shot/mlm/global_piqa_nonparallel/*/predictions.json",
        }
        cands = [pathlib.Path(x) for x in glob.glob(str(od / patterns[task]), recursive=True)]
        if len(cands) == 1:
            return cands[0]
        raise FileNotFoundError(f"{task}: expected 1 pred under {od}, found {len(cands)}")
    raise FileNotFoundError(f"No predictions path for {task} in payload {payload.get('target')}")


def iter_gold_jsonl(gold_dir: pathlib.Path):
    for gf in sorted(gold_dir.glob("*.jsonl")):
        rows = [json.loads(l) for l in gf.read_text(encoding="utf-8").splitlines() if l.strip()]
        yield gf.stem, rows


def score_blimp_like(pred_path: pathlib.Path, gold_dir: pathlib.Path) -> list[dict[str, Any]]:
    pred = load_json(pred_path)
    items = []
    for sub, rows in iter_gold_jsonl(gold_dir):
        if sub not in pred:
            continue
        preds = pred[sub]["predictions"]
        if len(preds) != len(rows):
            raise ValueError(f"{pred_path}: {sub} pred {len(preds)} != gold {len(rows)}")
        for pr, g in zip(preds, rows):
            items.append({"id": pr["id"], "subtask": sub, "correct": (pr.get("pred") or "").strip() == g["sentence_good"].strip()})
    return items


def score_ewok(pred_path: pathlib.Path, gold_dir: pathlib.Path) -> list[dict[str, Any]]:
    pred = load_json(pred_path)
    items = []
    for sub, rows in iter_gold_jsonl(gold_dir):
        if sub not in pred:
            continue
        preds = pred[sub]["predictions"]
        if len(preds) != len(rows):
            raise ValueError(f"{pred_path}: {sub} pred {len(preds)} != gold {len(rows)}")
        for pr, g in zip(preds, rows):
            target = " ".join([g["Context1"], g["Target1"]]).strip()
            items.append({"id": pr["id"], "subtask": sub, "domain": g.get("Domain", sub), "correct": (pr.get("pred") or "").strip() == target})
    return items


def score_comps(pred_path: pathlib.Path, gold_dir: pathlib.Path) -> list[dict[str, Any]]:
    pred = load_json(pred_path)
    sub_to_stem = {"base": "comps_base", "wugs_dist_before": "comps_wugs_dist-before", "wugs_dist_in_between": "comps_wugs_dist-in-between", "wugs": "comps_wugs"}
    items = []
    for sub, data in pred.items():
        stem = sub_to_stem.get(sub)
        if not stem:
            continue
        rows = [json.loads(l) for l in (gold_dir / f"{stem}.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
        preds = data["predictions"]
        if len(preds) != len(rows):
            raise ValueError(f"{pred_path}: {sub} pred {len(preds)} != gold {len(rows)}")
        for pr, g in zip(preds, rows):
            target = " ".join([g["prefix_acceptable"], g["property_phrase"]]).strip()
            items.append({"id": pr["id"], "subtask": sub, "correct": (pr.get("pred") or "").strip() == target})
    return items


def score_globalpiqa(pred_path: pathlib.Path, gold_file: pathlib.Path) -> list[dict[str, Any]]:
    pred = load_json(pred_path)
    rows = [json.loads(l) for l in gold_file.read_text(encoding="utf-8").splitlines() if l.strip()]
    gold_by_id = {r.get("example_id", ""): r for r in rows}
    items = []
    for eid, data in pred.items():
        g = gold_by_id.get(eid)
        if not g:
            continue
        correct_answer = g[f"solution{g['label']}"].strip()
        for pr in data["predictions"]:
            items.append({"id": pr["id"], "example_id": eid, "correct": (pr.get("pred") or "").strip() == correct_answer})
    return items


def compute_trans(a: list[dict[str, Any]], b: list[dict[str, Any]]) -> dict[str, Any]:
    ad = {x["id"]: bool(x["correct"]) for x in a}
    bd = {x["id"]: bool(x["correct"]) for x in b}
    ids = sorted(set(ad) & set(bd))
    cc = cw = wc = ww = 0
    for i in ids:
        ca = ad[i]; cb = bd[i]
        if ca and cb: cc += 1
        elif ca and not cb: cw += 1
        elif (not ca) and cb: wc += 1
        else: ww += 1
    total = cc + cw + wc + ww
    return {
        "n_common": total,
        "CC": cc,
        "CW_lost_from_a": cw,
        "WC_new_in_b": wc,
        "WW": ww,
        "a_correct": cc + cw,
        "b_correct": cc + wc,
        "accuracy_delta_b_minus_a_pp": round(100.0 * (wc - cw) / max(1, total), 6),
        "net_gain_b_minus_a_items": wc - cw,
        "churn_items": cw + wc,
        "retained_rate_of_a_correct_pct": round(100.0 * cc / max(1, cc + cw), 6),
        "loss_rate_of_a_correct_pct": round(100.0 * cw / max(1, cc + cw), 6),
    }


def group_transitions(items_a: list[dict[str, Any]], items_b: list[dict[str, Any]], key: str) -> dict[str, Any]:
    ga: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    gb: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for x in items_a:
        ga[str(x.get(key, "unknown"))].append(x)
    for x in items_b:
        gb[str(x.get(key, "unknown"))].append(x)
    out = {}
    for k in sorted(set(ga) & set(gb)):
        out[k] = compute_trans(ga[k], gb[k])
    return out


def aggregate_group(items: list[dict[str, Any]], domains: list[str]) -> dict[str, Any]:
    chosen = [x for x in items if str(x.get("domain")) in domains]
    c = sum(1 for x in chosen if x["correct"])
    n = len(chosen)
    return {"n": n, "correct": c, "accuracy": round(100.0 * c / max(1, n), 6)}


def build_items(arms: dict[str, dict[str, Any]]) -> dict[str, dict[str, list[dict[str, Any]]]]:
    tasks = {
        "Supplement": lambda p: score_blimp_like(p, OFFICIAL_FULL / "supplement_filtered"),
        "EWoK": lambda p: score_ewok(p, OFFICIAL_FULL / "ewok_filtered"),
        "COMPS": lambda p: score_comps(p, OFFICIAL_FULL / "comps"),
        "GlobalPIQA_parallel": lambda p: score_globalpiqa(p, GLOBALPIQA_FULL / "global_piqa_parallel/eng_latn.jsonl"),
        "GlobalPIQA_nonparallel": lambda p: score_globalpiqa(p, GLOBALPIQA_FULL / "global_piqa_nonparallel/eng_latn.jsonl"),
    }
    out: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for task, fn in tasks.items():
        out[task] = {}
        for name, payload in arms.items():
            try:
                out[task][name] = fn(pred_from_payload(payload, task))
            except Exception as e:
                out[task][name] = []
                print(json.dumps({"event": "item_scoring_failed", "task": task, "arm": name, "error": str(e)}), flush=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--full-json", default=str(ROOT / "data/compact_triangle_noaoa_eval/per_target/compact_view_reinvest.json"))
    ap.add_argument("--arm-json", action="append", default=[], help="NAME=path to per-target JSON")
    ap.add_argument("--output_dir", default=str(DEFAULT_OUT))
    args = ap.parse_args()

    arms = {"full": load_json(pathlib.Path(args.full_json))}
    for spec in args.arm_json:
        name, sep, path = spec.partition("=")
        if not sep:
            raise SystemExit(f"--arm-json must be NAME=path, got {spec}")
        arms[name] = load_json(pathlib.Path(path))
    out_dir = pathlib.Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cheap = score_table(arms)
    items = build_items(arms)
    transitions: dict[str, Any] = {}
    names = [n for n in arms if n != "full"]
    for task, by_arm in items.items():
        transitions[task] = {}
        for n in names:
            if by_arm.get("full") and by_arm.get(n):
                transitions[task][f"{n}_minus_full"] = compute_trans(by_arm["full"], by_arm[n])
                if task in {"Supplement", "COMPS"}:
                    transitions[task][f"{n}_minus_full_by_subtask"] = group_transitions(by_arm["full"], by_arm[n], "subtask")
                if task == "EWoK":
                    transitions[task][f"{n}_minus_full_by_domain"] = group_transitions(by_arm["full"], by_arm[n], "domain")
        if len(names) >= 2:
            for i in range(len(names)):
                for j in range(i + 1, len(names)):
                    a, b = names[i], names[j]
                    if by_arm.get(a) and by_arm.get(b):
                        transitions[task][f"{a}_minus_{b}"] = compute_trans(by_arm[b], by_arm[a])

    ewok_groups: dict[str, Any] = {}
    ewok_items = items.get("EWoK", {})
    for group_name, domains in [("relational_domains", RELATIONAL), ("adjacency_independent_domains", ADJ_INDEPENDENT)]:
        per = {name: aggregate_group(ewok_items.get(name, []), domains) for name in arms}
        rec = {"domains": domains, "scores": per, "deltas": {}}
        for n in names:
            rec["deltas"][f"{n}_minus_full"] = round(per[n]["accuracy"] - per["full"]["accuracy"], 6) if per[n]["n"] and per["full"]["n"] else None
        if len(names) >= 2:
            for i in range(len(names)):
                for j in range(i + 1, len(names)):
                    a, b = names[i], names[j]
                    rec["deltas"][f"{a}_minus_{b}"] = round(per[a]["accuracy"] - per[b]["accuracy"], 6) if per[a]["n"] and per[b]["n"] else None
        ewok_groups[group_name] = rec

    payload = {
        "status": "PACKED_TARGETSELECT_ENDPOINT_SIGNATURE",
        "meaning": "Endpoint readout for source-absent versus copied-content loss deletion in the historical compact-view packed geometry. Interpret through Supplement and relational-EWoK transition pattern, not aggregate score alone.",
        "inputs": {"full_json": args.full_json, "arm_jsons": args.arm_json},
        "cheap7": cheap,
        "item_transitions": transitions,
        "ewok_predeclared_groups": ewok_groups,
        "interpretation_template": {
            "source_absent_channel_load_bearing": "Supported only if drop_abs loses the established compact-view Supplement/EWoK relational advantages versus full more than the matched copied-content arm does, with retained/lost/new item transitions aligning to the research source-own-adjacency signature.",
            "aggregate_score_only": "An equal7 or GlobalPIQA-only delta is not sufficient to connect the target channel to the compact-view mechanism."
        },
    }
    out = out_dir / "packed_targetselect_endpoint_signature.json"
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    # Short markdown summary.
    md = ["# research packed target-selective endpoint signature", "", f"JSON: `{out}`", "", "## Cheap7-style scores", "", "| arm | " + " | ".join(COLUMNS + ["equal7"]) + " |", "|---" + "|---:" * (len(COLUMNS) + 1) + "|"]
    for name, sc in cheap["scores"].items():
        md.append("| " + name + " | " + " | ".join("NA" if sc.get(c) is None else f"{float(sc[c]):.3f}" for c in COLUMNS + ["cheap7_equal_mean"]) + " |")
    md += ["", "## Deltas", "", "| contrast | " + " | ".join(COLUMNS + ["equal7"]) + " |", "|---" + "|---:" * (len(COLUMNS) + 1) + "|"]
    for name, de in cheap["deltas"].items():
        md.append("| " + name + " | " + " | ".join("NA" if de.get(c) is None else f"{float(de[c]):+.3f}" for c in COLUMNS + ["cheap7_equal_mean"]) + " |")
    md += ["", "## EWoK predeclared groups", ""]
    for g, rec in ewok_groups.items():
        md.append(f"### {g}")
        md.append("| arm | n | acc |")
        md.append("|---|---:|---:|")
        for arm, s in rec["scores"].items():
            md.append(f"| {arm} | {s['n']} | {s['accuracy']:.3f} |")
        md.append("Deltas: " + json.dumps(rec["deltas"], ensure_ascii=False))
        md.append("")
    (out_dir / "packed_targetselect_endpoint_signature.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "out": str(out), "cheap7": cheap, "ewok_groups": ewok_groups}, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
