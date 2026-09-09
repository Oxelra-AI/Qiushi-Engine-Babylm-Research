#!/usr/bin/env python3
"""research: conservation/transition readout for compact-view Entity evidence.

Scientific purpose
------------------
research showed that official Entity V-B is positive across zero and nonzero
operation strata, but a counterbalanced binding panel showed affected gains with
unaffected costs.  This script reads the *existing* files only and asks whether
view training moves examples into jointly conserved/update-correct states, or
mostly swaps one side of a binding pair for the other.

It produces two cheap readouts:
  1. paired binding-state flows on the counterbalanced binding substrate, grouping
     affected and unaffected queries from the same event into states
     {both_correct, affected_only, unaffected_only, neither};
  2. official Entity item-level gain/loss and option-index transitions for
     first-basin DeBERTa MAX view/repeat/breadth predictions by split and
     numops.

No model loading, no training, no GPU, no official scoring, no upload.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import csv
import json
import pathlib
import statistics
import time
from collections import Counter, defaultdict
from typing import Any


def find_user_root() -> pathlib.Path:
    return _PUBLIC_ROOT


ROOT = find_user_root()
A01_WS = ROOT / "experiments/archive" / 'representation_and_objectives'
A02_WS = ROOT / "experiments/archive" / 'frontier_consolidation'
OUT = A01_WS / "data" / "conservation_transition_readout"
BINDING_EVAL = A01_WS / "data" / "annotated_corpus" / "eval.jsonl"
BINDING_PER_ITEM = A01_WS / "data" / "breadth_binding_affunaff_probe" / "binding_per_item_records.csv"
ENTITY_ROOT = A01_WS / "data" / "pristine_official_coordinate" / "babylm-eval" / "strict" / "evaluation_data" / "full_eval" / "entity_tracking"
FIXED_BUDGET_SUMMARY = A02_WS / "data" / "fixed_budget_allocation_readout" / "fixed_budget_allocation_readout_summary.md"

CHECKPOINTS = ["chck_80M", "chck_90M", "chck_100M"]
SPLITS = ["ambiref", "regular", "move_contents"]
NUMOPS = list(range(6))
STATES = ["both_correct", "affected_only", "unaffected_only", "neither"]
PRED_INDEXES = [-1, 0, 1, 2, 3, 4]

ARM_CONFIGS: dict[str, dict[str, Any]] = {
    "view": {
        "arm_key": "deberta_basin1_view",
        "eval_root": A02_WS / "data/dose_ladder_stable_eval/eval",
        "target_prefix": "dose_max_view",
    },
    "repeat": {
        "arm_key": "deberta_basin1_repeat",
        "eval_root": A02_WS / "data/dose_ladder_stable_eval/eval",
        "target_prefix": "dose_max_repeat",
    },
    "breadth": {
        "arm_key": "deberta_breadth",
        "eval_root": A02_WS / "data/breadth_entity_ewok_eval/eval",
        "target_prefix": "max_breadth_seed43022",
    },
}
CONTRASTS = [
    ("VminusR", "view", "repeat"),
    ("VminusB", "view", "breadth"),
    ("BminusR", "breadth", "repeat"),
]


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def rel(p: pathlib.Path | str | None) -> str | None:
    if p is None:
        return None
    pp = pathlib.Path(p)
    try:
        return str(pp.relative_to(ROOT))
    except Exception:
        return str(pp)


def boolish(x: Any) -> bool:
    return str(x).strip().lower() in {"true", "1", "yes"}


def read_json(path: pathlib.Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: pathlib.Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_csv(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for r in rows:
        for k in r.keys():
            if k not in fields:
                fields.append(k)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def mean(xs: list[float]) -> float | None:
    return sum(xs) / len(xs) if xs else None


def fmt_pp(x: Any) -> str:
    return "NA" if x is None else f"{float(x):+.3f}"


def fmt_frac_pp(x: Any) -> str:
    return "NA" if x is None else f"{100.0 * float(x):+.3f}"


# ---------------------------------------------------------------------------
# Binding paired-state readout
# ---------------------------------------------------------------------------


def load_binding_meta() -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    with BINDING_EVAL.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("kind") != "binding":
                continue
            qid = str(r.get("quartet_id", ""))
            # quartet_id is ..._<actor_entity>_<query_entity>; remove the final
            # query field to pair affected and unaffected queries for one event.
            pair_key = qid.rsplit("_", 1)[0] if "_" in qid else qid
            out[str(r["id"])] = {
                "id": r["id"],
                "pair_key": pair_key,
                "split": r.get("split"),
                "family": r.get("family"),
                "template_id": r.get("template_id"),
                "actor_entity": r.get("actor_entity"),
                "query_entity": r.get("query_entity"),
                "is_affected_query": bool(r.get("is_affected_query")),
                "answer": r.get("answer"),
                "foil": r.get("foil"),
                "text": r.get("text"),
            }
    return out


def classify_pair(rows: list[dict[str, Any]]) -> tuple[str | None, dict[str, Any]]:
    aff = [r for r in rows if r["is_affected"]]
    unaff = [r for r in rows if not r["is_affected"]]
    if len(aff) != 1 or len(unaff) != 1:
        return None, {"pair_error": f"expected one affected and one unaffected; got {len(aff)}/{len(unaff)}"}
    ac = boolish(aff[0]["correct"])
    uc = boolish(unaff[0]["correct"])
    if ac and uc:
        state = "both_correct"
    elif ac and not uc:
        state = "affected_only"
    elif (not ac) and uc:
        state = "unaffected_only"
    else:
        state = "neither"
    return state, {
        "affected_id": aff[0]["id"],
        "unaffected_id": unaff[0]["id"],
        "affected_correct": ac,
        "unaffected_correct": uc,
        "affected_margin": float(aff[0]["margin"]),
        "unaffected_margin": float(unaff[0]["margin"]),
    }


def binding_readout() -> dict[str, Any]:
    meta = load_binding_meta()
    if not BINDING_PER_ITEM.exists():
        raise FileNotFoundError(BINDING_PER_ITEM)
    item_rows: list[dict[str, Any]] = []
    with BINDING_PER_ITEM.open("r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            mid = r.get("id", "")
            if mid not in meta:
                continue
            if r.get("split") != "eval_held_recomb":
                continue
            rr = {**r, **meta[mid]}
            rr["is_affected"] = boolish(r.get("is_affected"))
            rr["correct"] = boolish(r.get("correct"))
            rr["margin"] = float(r.get("margin", "nan"))
            item_rows.append(rr)

    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for r in item_rows:
        grouped[(r["arm"], r["checkpoint"], r["pair_key"])].append(r)

    pair_rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for (arm, ck, pair_key), rows in sorted(grouped.items()):
        state, info = classify_pair(rows)
        base = rows[0]
        if state is None:
            errors.append({"arm": arm, "checkpoint": ck, "pair_key": pair_key, **info})
            continue
        pair_rows.append({
            "arm": arm,
            "checkpoint": ck,
            "pair_key": pair_key,
            "state": state,
            "family": base.get("family"),
            "template_id": base.get("template_id"),
            "actor_entity": base.get("actor_entity"),
            **info,
        })

    state_counts: list[dict[str, Any]] = []
    by_model: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for r in pair_rows:
        by_model[(r["arm"], r["checkpoint"])].append(r)
    for (arm, ck), rows in sorted(by_model.items()):
        n = len(rows)
        c = Counter(r["state"] for r in rows)
        rec: dict[str, Any] = {"arm": arm, "checkpoint": ck, "n_pairs": n}
        for st in STATES:
            rec[f"{st}_count"] = c.get(st, 0)
            rec[f"{st}_frac"] = c.get(st, 0) / n if n else None
        rec["affected_accuracy_from_pairs"] = (c["both_correct"] + c["affected_only"]) / n if n else None
        rec["unaffected_accuracy_from_pairs"] = (c["both_correct"] + c["unaffected_only"]) / n if n else None
        rec["mean_affected_margin"] = mean([float(r["affected_margin"]) for r in rows])
        rec["mean_unaffected_margin"] = mean([float(r["unaffected_margin"]) for r in rows])
        state_counts.append(rec)

    idx: dict[tuple[str, str, str], dict[str, Any]] = {(r["arm"], r["checkpoint"], r["pair_key"]): r for r in pair_rows}
    flow_rows: list[dict[str, Any]] = []
    flow_detail_rows: list[dict[str, Any]] = []
    for cname, aarm, barm in CONTRASTS:
        for ck in CHECKPOINTS:
            pair_keys = sorted({k for (arm, c, k) in idx if c == ck and arm in {aarm, barm}})
            common = [k for k in pair_keys if (aarm, ck, k) in idx and (barm, ck, k) in idx]
            n = len(common)
            matrix = Counter()
            for k in common:
                a = idx[(aarm, ck, k)]
                b = idx[(barm, ck, k)]
                trans = (b["state"], a["state"])
                matrix[trans] += 1
                flow_detail_rows.append({
                    "contrast": cname,
                    "checkpoint": ck,
                    "pair_key": k,
                    "from_arm": barm,
                    "to_arm": aarm,
                    "from_state": b["state"],
                    "to_state": a["state"],
                    "family": a.get("family"),
                    "template_id": a.get("template_id"),
                    "actor_entity": a.get("actor_entity"),
                    "from_affected_correct": b["affected_correct"],
                    "from_unaffected_correct": b["unaffected_correct"],
                    "to_affected_correct": a["affected_correct"],
                    "to_unaffected_correct": a["unaffected_correct"],
                    "from_affected_margin": b["affected_margin"],
                    "from_unaffected_margin": b["unaffected_margin"],
                    "to_affected_margin": a["affected_margin"],
                    "to_unaffected_margin": a["unaffected_margin"],
                })
            a_counts = Counter(idx[(aarm, ck, k)]["state"] for k in common)
            b_counts = Counter(idx[(barm, ck, k)]["state"] for k in common)
            rec: dict[str, Any] = {"contrast": cname, "checkpoint": ck, "from_arm": barm, "to_arm": aarm, "n_pairs": n}
            for st in STATES:
                rec[f"to_{st}_frac"] = a_counts.get(st, 0) / n if n else None
                rec[f"from_{st}_frac"] = b_counts.get(st, 0) / n if n else None
                rec[f"delta_{st}_frac"] = (a_counts.get(st, 0) - b_counts.get(st, 0)) / n if n else None
            # Conservation-oriented flows.
            def frac(frm: str | None = None, to: str | None = None) -> float | None:
                if not n:
                    return None
                s = 0
                for (f, t), cnt in matrix.items():
                    if (frm is None or f == frm) and (to is None or t == to):
                        s += cnt
                return s / n
            rec.update({
                "from_not_both_to_both_frac": sum(cnt for (f, t), cnt in matrix.items() if f != "both_correct" and t == "both_correct") / n if n else None,
                "from_both_to_not_both_frac": sum(cnt for (f, t), cnt in matrix.items() if f == "both_correct" and t != "both_correct") / n if n else None,
                "both_correct_retained_frac_of_all": frac("both_correct", "both_correct"),
                "unaffected_only_to_both_frac": frac("unaffected_only", "both_correct"),
                "unaffected_only_to_affected_only_frac": frac("unaffected_only", "affected_only"),
                "both_to_affected_only_frac": frac("both_correct", "affected_only"),
                "both_to_unaffected_only_frac": frac("both_correct", "unaffected_only"),
                "both_to_neither_frac": frac("both_correct", "neither"),
                "neither_to_both_frac": frac("neither", "both_correct"),
                "affected_only_to_both_frac": frac("affected_only", "both_correct"),
                "flow_matrix_json": json.dumps({f"{f}->{t}": cnt for (f, t), cnt in sorted(matrix.items())}, sort_keys=True),
            })
            flow_rows.append(rec)

    late_summary: list[dict[str, Any]] = []
    quantities = [
        "delta_both_correct_frac", "delta_affected_only_frac", "delta_unaffected_only_frac", "delta_neither_frac",
        "from_not_both_to_both_frac", "from_both_to_not_both_frac", "unaffected_only_to_both_frac",
        "unaffected_only_to_affected_only_frac", "both_to_affected_only_frac", "both_to_unaffected_only_frac", "both_to_neither_frac",
        "neither_to_both_frac", "affected_only_to_both_frac",
    ]
    for cname, _, _ in CONTRASTS:
        rows = [r for r in flow_rows if r["contrast"] == cname]
        for q in quantities:
            vals = [float(r[q]) for r in rows if r.get(q) is not None]
            if vals:
                late_summary.append({
                    "contrast": cname, "window": "late_80_90_100M", "quantity": q,
                    "n": len(vals), "mean_frac": mean(vals), "mean_pp": 100.0 * mean(vals),
                    "min_pp": 100.0 * min(vals), "max_pp": 100.0 * max(vals),
                    "checkpoints": ";".join(r["checkpoint"] for r in rows if r.get(q) is not None),
                })

    return {
        "binding_item_row_count": len(item_rows),
        "binding_pair_count": len(pair_rows),
        "binding_pair_errors": errors,
        "binding_pair_rows": pair_rows,
        "binding_state_counts": state_counts,
        "binding_flow_rows": flow_rows,
        "binding_flow_detail_rows": flow_detail_rows,
        "binding_late_flow_summary": late_summary,
    }


# ---------------------------------------------------------------------------
# Official Entity option-response/gain-loss transitions
# ---------------------------------------------------------------------------


def pred_path(cfg: dict[str, Any], ck: str) -> pathlib.Path | None:
    target = f"{cfg['target_prefix']}_{ck}"
    root = pathlib.Path(cfg["eval_root"])
    direct = root / "official_outputs" / target / "Entity"
    hits = sorted(direct.glob("**/zero_shot/mlm/entity_tracking/entity_tracking/predictions.json")) if direct.exists() else []
    if hits:
        return hits[0]
    hits = sorted(root.glob(f"**/{target}/Entity/**/zero_shot/mlm/entity_tracking/entity_tracking/predictions.json"))
    if hits:
        return hits[0]
    hits = sorted(root.glob(f"**/{target}*Entity*/zero_shot/mlm/entity_tracking/entity_tracking/predictions.json"))
    return hits[0] if hits else None


def load_entity_gold() -> dict[tuple[str, int], list[dict[str, Any]]]:
    out: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for split in SPLITS:
        by_num: dict[int, list[dict[str, Any]]] = defaultdict(list)
        with (ENTITY_ROOT / f"{split}.jsonl").open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                # Match the file-only scorer used for the current macro readout.
                if str(obj.get("options", [""])[0]).strip() == "nothing.":
                    continue
                by_num[int(obj["numops"])].append(obj)
        for n in NUMOPS:
            out[(split, n)] = by_num.get(n, [])
    return out


def load_entity_preds(path: pathlib.Path) -> dict[tuple[str, int], list[dict[str, Any]]]:
    raw = read_json(path)
    out: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for split in SPLITS:
        for n in NUMOPS:
            key = f"{split}_{n}_ops"
            rec = raw.get(key) if isinstance(raw, dict) else None
            out[(split, n)] = list((rec or {}).get("predictions") or []) if isinstance(rec, dict) else []
    return out


def entity_item_records() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    gold = load_entity_gold()
    path_rows: list[dict[str, Any]] = []
    item_rows: list[dict[str, Any]] = []
    for arm, cfg in ARM_CONFIGS.items():
        for ck in CHECKPOINTS:
            p = pred_path(cfg, ck)
            path_rows.append({"arm": arm, "checkpoint": ck, "predictions": rel(p), "exists": bool(p and p.exists())})
            if not p or not p.exists():
                continue
            preds = load_entity_preds(p)
            for split in SPLITS:
                for nops in NUMOPS:
                    g_rows = gold[(split, nops)]
                    p_rows = preds.get((split, nops), [])
                    m = min(len(g_rows), len(p_rows))
                    for i in range(m):
                        g = g_rows[i]
                        pred = str(p_rows[i].get("pred", "")).strip()
                        options = [str(o).strip() for o in g.get("options", [])]
                        try:
                            pred_idx = options.index(pred)
                        except ValueError:
                            pred_idx = -1
                        item_rows.append({
                            "arm": arm,
                            "checkpoint": ck,
                            "split": split,
                            "numops": nops,
                            "local_index": i,
                            "sample_id": g.get("sample_id"),
                            "example_id": g.get("example_id"),
                            "item_key": f"{split}:{nops}:{i}",
                            "pred": pred,
                            "pred_idx": pred_idx,
                            "correct": pred_idx == 0,
                            "gold": options[0] if options else None,
                            "option_count": len(options),
                        })
    return path_rows, item_rows


def add_group_rows(items: list[dict[str, Any]]) -> list[tuple[str, list[dict[str, Any]]]]:
    groups: list[tuple[str, list[dict[str, Any]]]] = []
    groups.append(("all_18_subtasks", items))
    groups.append(("zero_ops", [r for r in items if int(r["numops"]) == 0]))
    groups.append(("nonzero_ops", [r for r in items if int(r["numops"]) > 0]))
    for n in NUMOPS:
        groups.append((f"numops_{n}", [r for r in items if int(r["numops"]) == n]))
    for split in SPLITS:
        sr = [r for r in items if r["split"] == split]
        groups.append((f"split_{split}", sr))
        groups.append((f"{split}_zero_ops", [r for r in sr if int(r["numops"]) == 0]))
        groups.append((f"{split}_nonzero_ops", [r for r in sr if int(r["numops"]) > 0]))
        for n in NUMOPS:
            groups.append((f"{split}_numops_{n}", [r for r in sr if int(r["numops"]) == n]))
    return groups


def entity_transition_readout(item_rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_model: dict[tuple[str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for r in item_rows:
        by_model[(r["arm"], r["checkpoint"])][r["item_key"]] = r

    model_group_rows: list[dict[str, Any]] = []
    for (arm, ck), idx in sorted(by_model.items()):
        rows = list(idx.values())
        for group, gr in add_group_rows(rows):
            if not gr:
                continue
            n = len(gr)
            counts = Counter(int(r["pred_idx"]) for r in gr)
            rec: dict[str, Any] = {
                "arm": arm, "checkpoint": ck, "group": group, "n": n,
                "accuracy_frac": sum(1 for r in gr if r["correct"]) / n,
                "accuracy_pp": 100.0 * sum(1 for r in gr if r["correct"]) / n,
            }
            for pi in PRED_INDEXES:
                label = "not_in_options" if pi == -1 else f"option{pi}"
                rec[f"pred_{label}_frac"] = counts.get(pi, 0) / n
            rec["pred_index_counts_json"] = json.dumps({str(k): v for k, v in sorted(counts.items())}, sort_keys=True)
            model_group_rows.append(rec)

    transition_rows: list[dict[str, Any]] = []
    transition_detail_rows: list[dict[str, Any]] = []
    for cname, aarm, barm in CONTRASTS:
        for ck in CHECKPOINTS:
            aidx = by_model.get((aarm, ck), {})
            bidx = by_model.get((barm, ck), {})
            common_keys = sorted(set(aidx) & set(bidx))
            paired = []
            for k in common_keys:
                a = aidx[k]
                b = bidx[k]
                paired.append({"a": a, "b": b})
                transition_detail_rows.append({
                    "contrast": cname,
                    "checkpoint": ck,
                    "item_key": k,
                    "split": a["split"],
                    "numops": a["numops"],
                    "sample_id": a.get("sample_id"),
                    "example_id": a.get("example_id"),
                    "from_arm": barm,
                    "to_arm": aarm,
                    "from_pred_idx": b["pred_idx"],
                    "to_pred_idx": a["pred_idx"],
                    "from_correct": b["correct"],
                    "to_correct": a["correct"],
                    "gain_loss_state": "gain" if (not b["correct"] and a["correct"]) else ("loss" if (b["correct"] and not a["correct"]) else ("both_correct" if a["correct"] and b["correct"] else "both_wrong")),
                })
            # construct grouping by applying predicates to the a-side metadata
            pseudo_items = [{**x["a"], "_pair": x} for x in paired]
            for group, gr in add_group_rows(pseudo_items):
                if not gr:
                    continue
                n = len(gr)
                a_correct = sum(1 for r in gr if r["_pair"]["a"]["correct"])
                b_correct = sum(1 for r in gr if r["_pair"]["b"]["correct"])
                gains = sum(1 for r in gr if (not r["_pair"]["b"]["correct"] and r["_pair"]["a"]["correct"]))
                losses = sum(1 for r in gr if (r["_pair"]["b"]["correct"] and not r["_pair"]["a"]["correct"]))
                both_correct = sum(1 for r in gr if (r["_pair"]["b"]["correct"] and r["_pair"]["a"]["correct"]))
                both_wrong = sum(1 for r in gr if ((not r["_pair"]["b"]["correct"]) and (not r["_pair"]["a"]["correct"])))
                a_counts = Counter(int(r["_pair"]["a"]["pred_idx"]) for r in gr)
                b_counts = Counter(int(r["_pair"]["b"]["pred_idx"]) for r in gr)
                trans_counts = Counter((int(r["_pair"]["b"]["pred_idx"]), int(r["_pair"]["a"]["pred_idx"])) for r in gr)
                rec: dict[str, Any] = {
                    "contrast": cname, "checkpoint": ck, "from_arm": barm, "to_arm": aarm, "group": group, "n": n,
                    "from_accuracy_frac": b_correct / n,
                    "to_accuracy_frac": a_correct / n,
                    "delta_accuracy_pp": 100.0 * (a_correct - b_correct) / n,
                    "both_correct_frac": both_correct / n,
                    "gain_wrong_to_correct_frac": gains / n,
                    "loss_correct_to_wrong_frac": losses / n,
                    "both_wrong_frac": both_wrong / n,
                    "gain_minus_loss_pp": 100.0 * (gains - losses) / n,
                    "predidx_transition_json": json.dumps({f"{f}->{t}": cnt for (f, t), cnt in sorted(trans_counts.items())}, sort_keys=True),
                }
                for pi in PRED_INDEXES:
                    label = "not_in_options" if pi == -1 else f"option{pi}"
                    rec[f"from_pred_{label}_frac"] = b_counts.get(pi, 0) / n
                    rec[f"to_pred_{label}_frac"] = a_counts.get(pi, 0) / n
                    rec[f"delta_pred_{label}_frac"] = (a_counts.get(pi, 0) - b_counts.get(pi, 0)) / n
                transition_rows.append(rec)

    # Late summaries for primary groups.
    primary_groups = ["all_18_subtasks", "zero_ops", "nonzero_ops"] + [f"numops_{n}" for n in NUMOPS] + [f"split_{s}" for s in SPLITS]
    late_summary: list[dict[str, Any]] = []
    for cname, _, _ in CONTRASTS:
        for group in primary_groups:
            rows = [r for r in transition_rows if r["contrast"] == cname and r["group"] == group]
            for q in ["delta_accuracy_pp", "gain_wrong_to_correct_frac", "loss_correct_to_wrong_frac", "both_correct_frac", "both_wrong_frac", "delta_pred_option0_frac", "delta_pred_option1_frac", "delta_pred_option2_frac", "delta_pred_option3_frac", "delta_pred_option4_frac"]:
                vals = [float(r[q]) for r in rows if r.get(q) is not None]
                if vals:
                    scale = 100.0 if q.endswith("_frac") else 1.0
                    late_summary.append({
                        "contrast": cname, "group": group, "quantity": q,
                        "n": len(vals), "mean": mean(vals), "mean_scaled": scale * mean(vals),
                        "min_scaled": scale * min(vals), "max_scaled": scale * max(vals),
                        "checkpoints": ";".join(r["checkpoint"] for r in rows),
                    })
    return {
        "entity_model_group_rows": model_group_rows,
        "entity_transition_rows": transition_rows,
        "entity_transition_detail_rows": transition_detail_rows,
        "entity_late_transition_summary": late_summary,
    }


# ---------------------------------------------------------------------------
# Markdown synthesis
# ---------------------------------------------------------------------------


def lookup(rows: list[dict[str, Any]], **kw: Any) -> dict[str, Any] | None:
    for r in rows:
        if all(r.get(k) == v for k, v in kw.items()):
            return r
    return None


def late_value(rows: list[dict[str, Any]], contrast: str, quantity: str) -> float | None:
    r = lookup(rows, contrast=contrast, quantity=quantity)
    return None if r is None else r.get("mean_pp")


def entity_late_value(rows: list[dict[str, Any]], contrast: str, group: str, quantity: str) -> float | None:
    r = lookup(rows, contrast=contrast, group=group, quantity=quantity)
    return None if r is None else r.get("mean_scaled")


def write_md(payload: dict[str, Any], path: pathlib.Path) -> None:
    bflows = payload["binding"]["binding_late_flow_summary"]
    eflows = payload["entity"]["entity_late_transition_summary"]
    lines: list[str] = []
    lines.append("# research conservation-transition readout\n\n")
    lines.append("This readout uses only existing prediction/per-item files. It interprets Entity movement through conservation: does an arm move event pairs into `both_correct` (changed entity updated and unaffected entity retained), or does it exchange one side of the pair for the other?\n\n")
    lines.append("## Binding paired-state flow: late 80/90/100M means\n\n")
    lines.append("Percentages below are fractions of paired affected/unaffected event rows, averaged over the three late checkpoints. `from` is the second arm in the contrast (e.g. B for V-B), `to` is the first arm.\n\n")
    lines.append("| contrast | Δ both-correct pp | Δ affected-only pp | Δ unaffected-only pp | Δ neither pp | not-both→both pp | both→not-both pp | unaffected-only→affected-only pp | both→affected-only pp |\n")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|\n")
    for cname, _, _ in CONTRASTS:
        lines.append(
            f"| {cname} | {fmt_pp(late_value(bflows,cname,'delta_both_correct_frac'))} | {fmt_pp(late_value(bflows,cname,'delta_affected_only_frac'))} | {fmt_pp(late_value(bflows,cname,'delta_unaffected_only_frac'))} | {fmt_pp(late_value(bflows,cname,'delta_neither_frac'))} | {fmt_pp(late_value(bflows,cname,'from_not_both_to_both_frac'))} | {fmt_pp(late_value(bflows,cname,'from_both_to_not_both_frac'))} | {fmt_pp(late_value(bflows,cname,'unaffected_only_to_affected_only_frac'))} | {fmt_pp(late_value(bflows,cname,'both_to_affected_only_frac'))} |\n"
        )
    lines.append("\n## Binding per-checkpoint V-B state flows\n\n")
    lines.append("| checkpoint | from both | to both | Δ both pp | from affected-only | to affected-only | from unaffected-only | to unaffected-only | from neither | to neither | flow matrix |\n")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|\n")
    for r in [x for x in payload["binding"]["binding_flow_rows"] if x["contrast"] == "VminusB"]:
        lines.append(
            f"| {r['checkpoint']} | {fmt_frac_pp(r['from_both_correct_frac'])} | {fmt_frac_pp(r['to_both_correct_frac'])} | {fmt_frac_pp(r['delta_both_correct_frac'])} | {fmt_frac_pp(r['from_affected_only_frac'])} | {fmt_frac_pp(r['to_affected_only_frac'])} | {fmt_frac_pp(r['from_unaffected_only_frac'])} | {fmt_frac_pp(r['to_unaffected_only_frac'])} | {fmt_frac_pp(r['from_neither_frac'])} | {fmt_frac_pp(r['to_neither_frac'])} | `{r['flow_matrix_json']}` |\n"
        )
    lines.append("\n## Official Entity gain/loss and option response\n\n")
    lines.append("Option0 is the correct completion in the official Entity format. The table separates net accuracy from gain/loss churn and from response-index movement.\n\n")
    lines.append("| contrast | group | Δ accuracy pp | wrong→correct pp | correct→wrong pp | both-correct pp | both-wrong pp | Δ option0 pp | Δ option1 pp | Δ option2 pp | Δ option3 pp | Δ option4 pp |\n")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
    for cname in ["VminusB", "BminusR", "VminusR"]:
        for group in ["all_18_subtasks", "zero_ops", "nonzero_ops", "numops_0", "numops_1", "numops_2", "numops_3", "numops_4", "numops_5"]:
            lines.append(
                f"| {cname} | {group} | {fmt_pp(entity_late_value(eflows,cname,group,'delta_accuracy_pp'))} | {fmt_pp(entity_late_value(eflows,cname,group,'gain_wrong_to_correct_frac'))} | {fmt_pp(entity_late_value(eflows,cname,group,'loss_correct_to_wrong_frac'))} | {fmt_pp(entity_late_value(eflows,cname,group,'both_correct_frac'))} | {fmt_pp(entity_late_value(eflows,cname,group,'both_wrong_frac'))} | {fmt_pp(entity_late_value(eflows,cname,group,'delta_pred_option0_frac'))} | {fmt_pp(entity_late_value(eflows,cname,group,'delta_pred_option1_frac'))} | {fmt_pp(entity_late_value(eflows,cname,group,'delta_pred_option2_frac'))} | {fmt_pp(entity_late_value(eflows,cname,group,'delta_pred_option3_frac'))} | {fmt_pp(entity_late_value(eflows,cname,group,'delta_pred_option4_frac'))} |\n"
            )
    lines.append("\n## Scientific interpretation\n\n")
    lines.extend(payload["scientific_interpretation_md"])
    lines.append("\n## Current broad-score file status\n\n")
    lines.append(payload.get("fixed_budget_status_excerpt", "No fixed-budget summary file found.") + "\n\n")
    lines.append("## Files\n\n")
    for k, v in payload["files"].items():
        lines.append(f"- {k}: `{v}`\n")
    path.write_text("".join(lines), encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    binding = binding_readout()
    path_rows, entity_items = entity_item_records()
    entity = entity_transition_readout(entity_items)

    files = {
        "summary_md": rel(OUT / "conservation_transition_summary.md"),
        "summary_json": rel(OUT / "conservation_transition_summary.json"),
        "binding_pair_states_csv": rel(OUT / "binding_pair_states.csv"),
        "binding_state_counts_csv": rel(OUT / "binding_state_counts.csv"),
        "binding_flow_rows_csv": rel(OUT / "binding_flow_rows.csv"),
        "binding_flow_detail_csv": rel(OUT / "binding_flow_detail_rows.csv"),
        "binding_late_flow_summary_csv": rel(OUT / "binding_late_flow_summary.csv"),
        "entity_prediction_paths_csv": rel(OUT / "entity_prediction_paths.csv"),
        "entity_item_records_csv": rel(OUT / "entity_item_records.csv"),
        "entity_model_group_rows_csv": rel(OUT / "entity_model_group_rows.csv"),
        "entity_transition_rows_csv": rel(OUT / "entity_transition_rows.csv"),
        "entity_transition_detail_rows_csv": rel(OUT / "entity_transition_detail_rows.csv"),
        "entity_late_transition_summary_csv": rel(OUT / "entity_late_transition_summary.csv"),
    }

    # Save heavy/detail CSVs before synthesis.
    write_csv(OUT / "binding_pair_states.csv", binding["binding_pair_rows"])
    write_csv(OUT / "binding_state_counts.csv", binding["binding_state_counts"])
    write_csv(OUT / "binding_flow_rows.csv", binding["binding_flow_rows"])
    write_csv(OUT / "binding_flow_detail_rows.csv", binding["binding_flow_detail_rows"])
    write_csv(OUT / "binding_late_flow_summary.csv", binding["binding_late_flow_summary"])
    write_csv(OUT / "entity_prediction_paths.csv", path_rows)
    write_csv(OUT / "entity_item_records.csv", entity_items)
    write_csv(OUT / "entity_model_group_rows.csv", entity["entity_model_group_rows"])
    write_csv(OUT / "entity_transition_rows.csv", entity["entity_transition_rows"])
    write_csv(OUT / "entity_transition_detail_rows.csv", entity["entity_transition_detail_rows"])
    write_csv(OUT / "entity_late_transition_summary.csv", entity["entity_late_transition_summary"])

    # Pull only a small status excerpt from the current fixed-budget table.
    fixed_excerpt = "No fixed-budget allocation summary was present at run time."
    if FIXED_BUDGET_SUMMARY.exists():
        text = FIXED_BUDGET_SUMMARY.read_text(encoding="utf-8")
        keep = []
        for line in text.splitlines():
            if "Score rows present" in line or "Contrast rows present" in line or "Triangle rows present" in line or "not yet complete" in line or "chck_" in line or "Entity:" in line:
                keep.append(line)
        fixed_excerpt = "\n".join(keep[:80]) if keep else text[:2000]

    # Direct scientific reading computed from table lookups.
    b_late = binding["binding_late_flow_summary"]
    e_late = entity["entity_late_transition_summary"]
    vb_delta_both = late_value(b_late, "VminusB", "delta_both_correct_frac")
    vb_to_both = late_value(b_late, "VminusB", "from_not_both_to_both_frac")
    vb_from_both = late_value(b_late, "VminusB", "from_both_to_not_both_frac")
    vb_unaff_to_aff = late_value(b_late, "VminusB", "unaffected_only_to_affected_only_frac")
    vb_both_to_aff = late_value(b_late, "VminusB", "both_to_affected_only_frac")
    vb_off_zero = entity_late_value(e_late, "VminusB", "zero_ops", "delta_accuracy_pp")
    vb_off_nonzero = entity_late_value(e_late, "VminusB", "nonzero_ops", "delta_accuracy_pp")
    br_off_zero = entity_late_value(e_late, "BminusR", "zero_ops", "delta_accuracy_pp")
    br_off_nonzero = entity_late_value(e_late, "BminusR", "nonzero_ops", "delta_accuracy_pp")
    interpretation_lines = [
        f"- Binding V-B does **not** show the conservation pattern needed for source-conditioned state-record formation. Late mean Δ both-correct is {fmt_pp(vb_delta_both)} pp, while not-both→both is {fmt_pp(vb_to_both)} pp and both→not-both is {fmt_pp(vb_from_both)} pp. The paired flow is therefore churn around a near-chance boundary rather than robust movement into the state where both changed and stable entity readouts are correct.\n",
        f"- The erosive part is visible directly: V-B contains unaffected-only→affected-only flow {fmt_pp(vb_unaff_to_aff)} pp and both→affected-only flow {fmt_pp(vb_both_to_aff)} pp. This is exactly the conservation failure highlighted by the strategist: view can improve the changed query while damaging the stable query in the same event.\n",
        f"- Official Entity still has a real V-B surface effect: late zero-op Δ accuracy {fmt_pp(vb_off_zero)} pp and nonzero-op {fmt_pp(vb_off_nonzero)} pp. But the official item transition table shows this as ordinary gain/loss movement and option0/correct-response shifts, not as paired retention of unaffected state.\n",
        f"- B-R remains a changed-state/operation prior relative to repeat: official zero-op {fmt_pp(br_off_zero)} pp versus nonzero {fmt_pp(br_off_nonzero)} pp. V-B is arithmetically different from B-R on official Entity, but the paired binding table says that difference should not be promoted to source-correspondence without an aligned-versus-permuted arm.\n",
        "- Existing broad score files have not supplied a completed broad ex-Entity V-B/B-C/V-C triangle beyond the already noted 80M point. Because the conservation pattern is absent in the cheap binding evidence and broad V-B is still incomplete/weak, a new H100 permuted-companion run is not supported by the current evidence. The compact macro branch should be closed unless A02 later delivers already-running breadth/clean scores that materially change broad V-B or paired conservation.\n",
    ]

    payload = {
        "status": "CONSERVATION_TRANSITION_READOUT_COMPLETE",
        "created_utc": now(),
        "inputs": {
            "binding_eval": rel(BINDING_EVAL),
            "binding_per_item": rel(BINDING_PER_ITEM),
            "entity_root": rel(ENTITY_ROOT),
            "fixed_budget_summary": rel(FIXED_BUDGET_SUMMARY) if FIXED_BUDGET_SUMMARY.exists() else None,
        },
        "no_model_loading_training_evaluation_gpu_upload_or_leaderboard": True,
        "entity_prediction_paths": path_rows,
        "binding": {k: v for k, v in binding.items() if k not in {"binding_pair_rows", "binding_flow_detail_rows"}},
        "entity": {k: v for k, v in entity.items() if k not in {"entity_transition_detail_rows"}},
        "fixed_budget_status_excerpt": fixed_excerpt,
        "scientific_interpretation_md": interpretation_lines,
        "files": files,
    }
    write_json(OUT / "conservation_transition_summary.json", payload)
    write_md(payload, OUT / "conservation_transition_summary.md")
    print(json.dumps({
        "status": payload["status"],
        "summary_md": files["summary_md"],
        "summary_json": files["summary_json"],
        "binding_vb_delta_both_correct_pp": vb_delta_both,
        "binding_vb_not_both_to_both_pp": vb_to_both,
        "binding_vb_both_to_not_both_pp": vb_from_both,
        "official_vb_zero_ops_delta_pp": vb_off_zero,
        "official_vb_nonzero_ops_delta_pp": vb_off_nonzero,
        "prediction_missing_count": sum(1 for r in path_rows if not r["exists"]),
        "no_model_loading_training_evaluation_gpu_upload_or_leaderboard": True,
    }, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
