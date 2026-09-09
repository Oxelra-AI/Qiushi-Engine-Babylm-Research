#!/usr/bin/env python3
"""research: strengthened endpoint signature for packed target-selective 100M arms.

This supersedes the quick research signature reader for mechanism interpretation.
It uses compound item keys, reports official-score vs micro-score differences,
paired uncertainty for arm-to-arm task/domain contrasts, and similarity of the
intervention vector to the historical compact-view triangle transition vectors.

Primary scientific contrast:
    drop_abs - drop_copied_word
where positive accuracy delta means drop_abs is more correct than the copied-word
control.  For interpreting source-absent supervision as beneficial, also inspect
`copied_word_minus_drop_abs` vectors, where positive means retaining source-absent
labels helped relative to deleting them.
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import math
import pathlib
import random
import statistics
from typing import Any

ROOT = pathlib.Path("experiments/archive/representation_and_objectives")
OFFICIAL_FULL = ROOT / "data/pristine_official_coordinate/babylm-eval/strict/evaluation_data/full_eval"
GLOBALPIQA_FULL = pathlib.Path("experiments/archive/initial_model_studies/repos/babylm-eval/strict/evaluation_data/full_eval")
research = ROOT / "data/correctness_transitions/correctness_transitions.json"
DEFAULT_FULL = ROOT / "data/compact_triangle_noaoa_eval/per_target/compact_view_reinvest.json"
DEFAULT_OUT = ROOT / "data/packed_targetselect_endpoint_signature_v2"

SCORE_COLUMNS = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA", "Reading"]
ITEM_TASKS = ["Supplement", "EWoK", "COMPS", "GlobalPIQA_parallel", "GlobalPIQA_nonparallel"]
RELATIONAL = ["social-properties", "physical-dynamics", "spatial-relations", "physical-relations"]
ADJ_INDEPENDENT = ["material-properties", "social-interactions"]
HIST_VECTOR_KEYS = ["Supplement", *RELATIONAL, *ADJ_INDEPENDENT]


def load_json(p: pathlib.Path) -> Any:
    return json.loads(p.read_text(encoding="utf-8"))


def task_score(payload: dict[str, Any], col: str) -> float | None:
    oo = payload.get("official_overall", {})
    if isinstance(oo, dict) and isinstance(oo.get("scores"), dict) and col in oo["scores"]:
        v = oo["scores"].get(col)
        return None if v is None else float(v)
    if col == "GlobalPIQA":
        a = task_score(payload, "GlobalPIQA_parallel")
        b = task_score(payload, "GlobalPIQA_nonparallel")
        return None if a is None or b is None else (a + b) / 2.0
    t = payload.get("tasks", {}).get(col)
    if not isinstance(t, dict):
        return None
    if col == "Reading":
        v = (t.get("scores") or {}).get("Reading")
    else:
        v = t.get("score")
    return None if v is None else float(v)


def score_table(arms: dict[str, dict[str, Any]]) -> dict[str, Any]:
    scores = {}
    for name, payload in arms.items():
        scores[name] = {c: task_score(payload, c) for c in SCORE_COLUMNS}
        vals = [scores[name][c] for c in SCORE_COLUMNS]
        scores[name]["cheap7_equal_mean"] = None if any(v is None for v in vals) else round(sum(float(v) for v in vals) / len(vals), 6)
    deltas = {}
    names = [n for n in arms if n != "full"]
    for n in names:
        key = f"{n}_minus_full"
        deltas[key] = {}
        for c in SCORE_COLUMNS + ["cheap7_equal_mean"]:
            a = scores[n].get(c); b = scores["full"].get(c)
            deltas[key][c] = None if a is None or b is None else round(float(a) - float(b), 6)
    for i, a in enumerate(names):
        for b in names[i+1:]:
            key = f"{a}_minus_{b}"
            deltas[key] = {}
            for c in SCORE_COLUMNS + ["cheap7_equal_mean"]:
                av = scores[a].get(c); bv = scores[b].get(c)
                deltas[key][c] = None if av is None or bv is None else round(float(av) - float(bv), 6)
            key2 = f"{b}_minus_{a}"
            deltas[key2] = {}
            for c in SCORE_COLUMNS + ["cheap7_equal_mean"]:
                av = scores[b].get(c); bv = scores[a].get(c)
                deltas[key2][c] = None if av is None or bv is None else round(float(av) - float(bv), 6)
    return {"scores": scores, "deltas": deltas}


def pred_from_payload(payload: dict[str, Any], task: str) -> pathlib.Path:
    t = payload.get("tasks", {}).get(task)
    if isinstance(t, dict) and t.get("predictions"):
        p = pathlib.Path(t["predictions"])
        if p.exists():
            return p
    if isinstance(t, dict) and t.get("output_dir"):
        od = pathlib.Path(t["output_dir"])
        patterns = {
            "Supplement": "*/**/zero_shot/mlm/blimp/*/predictions.json",
            "EWoK": "*/**/zero_shot/mlm/ewok/*/predictions.json",
            "COMPS": "*/**/zero_shot/mlm/comps/*/predictions.json",
            "GlobalPIQA_parallel": "*/**/zero_shot/mlm/global_piqa_parallel/*/predictions.json",
            "GlobalPIQA_nonparallel": "*/**/zero_shot/mlm/global_piqa_nonparallel/*/predictions.json",
        }
        cands = [pathlib.Path(x) for x in glob.glob(str(od / patterns[task]), recursive=True)]
        if len(cands) == 1:
            return cands[0]
        raise FileNotFoundError(f"{task}: expected 1 pred under {od}, found {len(cands)}")
    raise FileNotFoundError(f"No prediction path for {task} in payload {payload.get('target')}")


def iter_gold_jsonl(gold_dir: pathlib.Path):
    for gf in sorted(gold_dir.glob("*.jsonl")):
        rows = [json.loads(l) for l in gf.read_text(encoding="utf-8").splitlines() if l.strip()]
        yield gf.stem, rows


def dup_info(items: list[dict[str, Any]]) -> dict[str, Any]:
    cnt = collections.Counter(x["key"] for x in items)
    dups = [k for k, v in cnt.items() if v > 1]
    return {"n_items": len(items), "n_unique_keys": len(cnt), "duplicate_key_count": len(dups), "max_duplicate_multiplicity": max(cnt.values()) if cnt else 0}


def score_blimp_like(pred_path: pathlib.Path, gold_dir: pathlib.Path) -> list[dict[str, Any]]:
    pred = load_json(pred_path)
    out = []
    for sub, rows in iter_gold_jsonl(gold_dir):
        if sub not in pred:
            continue
        preds = pred[sub]["predictions"]
        if len(preds) != len(rows):
            raise ValueError(f"{pred_path}: {sub} pred {len(preds)} != gold {len(rows)}")
        for idx, (pr, g) in enumerate(zip(preds, rows)):
            key = f"{sub}|{idx}|{pr.get('id','')}"
            out.append({"key": key, "subtask": sub, "correct": (pr.get("pred") or "").strip() == g["sentence_good"].strip()})
    return out


def score_ewok(pred_path: pathlib.Path, gold_dir: pathlib.Path) -> list[dict[str, Any]]:
    pred = load_json(pred_path)
    out = []
    for sub, rows in iter_gold_jsonl(gold_dir):
        if sub not in pred:
            continue
        preds = pred[sub]["predictions"]
        if len(preds) != len(rows):
            raise ValueError(f"{pred_path}: {sub} pred {len(preds)} != gold {len(rows)}")
        for idx, (pr, g) in enumerate(zip(preds, rows)):
            domain = str(g.get("Domain", sub))
            key = f"{domain}|{sub}|{idx}|{pr.get('id','')}"
            target = " ".join([g["Context1"], g["Target1"]]).strip()
            out.append({"key": key, "subtask": sub, "domain": domain, "correct": (pr.get("pred") or "").strip() == target})
    return out


def score_comps(pred_path: pathlib.Path, gold_dir: pathlib.Path) -> list[dict[str, Any]]:
    pred = load_json(pred_path)
    sub_to_stem = {"base": "comps_base", "wugs_dist_before": "comps_wugs_dist-before", "wugs_dist_in_between": "comps_wugs_dist-in-between", "wugs": "comps_wugs"}
    out = []
    for sub, data in pred.items():
        stem = sub_to_stem.get(sub)
        if not stem:
            continue
        rows = [json.loads(l) for l in (gold_dir / f"{stem}.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
        preds = data["predictions"]
        if len(preds) != len(rows):
            raise ValueError(f"{pred_path}: {sub} pred {len(preds)} != gold {len(rows)}")
        for idx, (pr, g) in enumerate(zip(preds, rows)):
            key = f"{sub}|{idx}|{pr.get('id','')}"
            target = " ".join([g["prefix_acceptable"], g["property_phrase"]]).strip()
            out.append({"key": key, "subtask": sub, "correct": (pr.get("pred") or "").strip() == target})
    return out


def score_globalpiqa(pred_path: pathlib.Path, gold_file: pathlib.Path) -> list[dict[str, Any]]:
    pred = load_json(pred_path)
    rows = [json.loads(l) for l in gold_file.read_text(encoding="utf-8").splitlines() if l.strip()]
    gold_by_id = {str(r.get("example_id", "")): r for r in rows}
    out = []
    for eid, data in pred.items():
        g = gold_by_id.get(str(eid))
        if not g:
            continue
        target = g[f"solution{g['label']}"].strip()
        for idx, pr in enumerate(data["predictions"]):
            key = f"{eid}|{idx}|{pr.get('id','')}"
            out.append({"key": key, "example_id": str(eid), "correct": (pr.get("pred") or "").strip() == target})
    return out


def build_items(arms: dict[str, dict[str, Any]]) -> dict[str, dict[str, list[dict[str, Any]]]]:
    task_fns = {
        "Supplement": lambda p: score_blimp_like(p, OFFICIAL_FULL / "supplement_filtered"),
        "EWoK": lambda p: score_ewok(p, OFFICIAL_FULL / "ewok_filtered"),
        "COMPS": lambda p: score_comps(p, OFFICIAL_FULL / "comps"),
        "GlobalPIQA_parallel": lambda p: score_globalpiqa(p, GLOBALPIQA_FULL / "global_piqa_parallel/eng_latn.jsonl"),
        "GlobalPIQA_nonparallel": lambda p: score_globalpiqa(p, GLOBALPIQA_FULL / "global_piqa_nonparallel/eng_latn.jsonl"),
    }
    out = {}
    for task, fn in task_fns.items():
        out[task] = {}
        for arm, payload in arms.items():
            out[task][arm] = fn(pred_from_payload(payload, task))
    return out


def transition(a_items: list[dict[str, Any]], b_items: list[dict[str, Any]], a_name: str, b_name: str) -> dict[str, Any]:
    ad = {x["key"]: bool(x["correct"]) for x in a_items}
    bd = {x["key"]: bool(x["correct"]) for x in b_items}
    keys = sorted(set(ad) & set(bd))
    cc = a_only = b_only = ww = 0
    discord = []
    for k in keys:
        a = ad[k]; b = bd[k]
        if a and b: cc += 1
        elif a and not b:
            a_only += 1
            discord.append(-1)
        elif (not a) and b:
            b_only += 1
            discord.append(1)
        else: ww += 1
    n = cc + a_only + b_only + ww
    delta = 100.0 * (b_only - a_only) / max(1, n)
    boot = paired_bootstrap(keys, ad, bd, seed=stable_u64(a_name + b_name + str(n)) % 1_000_000)
    return {
        "a": a_name,
        "b": b_name,
        "meaning": f"positive accuracy_delta_b_minus_a_pp means {b_name} is better than {a_name}",
        "n_common": n,
        "both_correct": cc,
        f"{a_name}_only_correct": a_only,
        f"{b_name}_only_correct": b_only,
        "both_wrong": ww,
        "a_correct": cc + a_only,
        "b_correct": cc + b_only,
        "a_micro_accuracy": round(100.0 * (cc + a_only) / max(1, n), 6),
        "b_micro_accuracy": round(100.0 * (cc + b_only) / max(1, n), 6),
        "accuracy_delta_b_minus_a_pp": round(delta, 6),
        "net_gain_b_minus_a_items": b_only - a_only,
        "discordant_items": a_only + b_only,
        "retained_rate_of_a_correct_pct": round(100.0 * cc / max(1, cc + a_only), 6),
        "loss_rate_of_a_correct_pct": round(100.0 * a_only / max(1, cc + a_only), 6),
        "bootstrap_accuracy_delta_b_minus_a_pp": boot,
        "mcnemar_asymptotic_chi2_continuity": mcnemar(a_only, b_only),
    }


def stable_u64(s: str) -> int:
    import hashlib
    return int.from_bytes(hashlib.blake2b(s.encode("utf-8"), digest_size=8).digest(), "little")


def paired_bootstrap(keys: list[str], ad: dict[str, bool], bd: dict[str, bool], seed: int, n_boot: int = 1000) -> dict[str, Any] | None:
    if not keys:
        return None
    # The paired difference for one item is in {-1, 0, +1}.  Sampling n items
    # item-by-item is too slow for COMPS (~91k items × many contrasts).  A
    # multinomial over the three difference categories is equivalent and fast.
    counts = collections.Counter((1 if bd[k] else 0) - (1 if ad[k] else 0) for k in keys)
    n = len(keys)
    probs = [counts[-1] / n, counts[0] / n, counts[1] / n]
    try:
        import numpy as np
        rng = np.random.default_rng(seed)
        draws = rng.multinomial(n, probs, size=n_boot)
        vals = (100.0 * (draws[:, 2] - draws[:, 0]) / n).tolist()
    except Exception:
        # Conservative fallback with much smaller loop cost than item bootstrap.
        rng_py = random.Random(seed)
        vals = []
        for _ in range(n_boot):
            c_neg = c_pos = 0
            for _ in range(n):
                r = rng_py.random()
                if r < probs[0]:
                    c_neg += 1
                elif r >= probs[0] + probs[1]:
                    c_pos += 1
            vals.append(100.0 * (c_pos - c_neg) / n)
    vals.sort()

    def q(p: float) -> float:
        return vals[min(n_boot - 1, max(0, int(round(p * (n_boot - 1)))))]

    return {"median": round(statistics.median(vals), 6), "p025": round(q(0.025), 6), "p05": round(q(0.05), 6), "p95": round(q(0.95), 6), "p975": round(q(0.975), 6), "fraction_gt0": round(sum(1 for x in vals if x > 0) / n_boot, 4), "diff_counts": {str(k): int(v) for k, v in sorted(counts.items())}}


def mcnemar(a_only: int, b_only: int) -> dict[str, Any]:
    # Exact p is not needed for routing; continuity-corrected chi2 gives scale of asymmetry.
    disc = a_only + b_only
    if disc == 0:
        return {"stat": 0.0, "discordant": 0}
    stat = (abs(a_only - b_only) - 1) ** 2 / disc if abs(a_only - b_only) >= 1 else 0.0
    return {"stat": round(stat, 6), "discordant": disc, "a_only": a_only, "b_only": b_only}


def group_items(items: list[dict[str, Any]], key: str, values: list[str] | None = None) -> list[dict[str, Any]]:
    if values is None:
        return items
    vs = set(values)
    return [x for x in items if str(x.get(key)) in vs]


def by_group_transitions(items: dict[str, list[dict[str, Any]]], a: str, b: str, group_key: str) -> dict[str, Any]:
    groups = sorted(set(str(x.get(group_key, "unknown")) for x in items.get(a, [])) | set(str(x.get(group_key, "unknown")) for x in items.get(b, [])))
    return {g: transition(group_items(items[a], group_key, [g]), group_items(items[b], group_key, [g]), a, b) for g in groups}


def aggregate_accuracy(items: list[dict[str, Any]], key: str | None = None, values: list[str] | None = None) -> dict[str, Any]:
    xs = items if key is None else group_items(items, key, values)
    n = len(xs); c = sum(1 for x in xs if x["correct"])
    return {"n": n, "correct": c, "micro_accuracy": round(100.0 * c / max(1, n), 6)}


def official_micro_report(arms: dict[str, dict[str, Any]], items: dict[str, dict[str, list[dict[str, Any]]]]) -> dict[str, Any]:
    out = {}
    for task in ITEM_TASKS:
        col = "GlobalPIQA" if task.startswith("GlobalPIQA") else task
        out[task] = {}
        for arm in arms:
            mic = aggregate_accuracy(items[task][arm])
            off = task_score(arms[arm], task)
            if off is None and task.startswith("GlobalPIQA"):
                off = task_score(arms[arm], task)
            out[task][arm] = {"micro": mic, "official_task_score": off, "official_minus_micro": None if off is None else round(float(off) - mic["micro_accuracy"], 6), "key_integrity": dup_info(items[task][arm])}
    return out


def load_step211_vector(pair_key: str) -> dict[str, float]:
    data = load_json(research)
    vec: dict[str, float] = {}
    supp = data.get("Supplement", {}).get(pair_key)
    if supp:
        vec["Supplement"] = float(supp.get("accuracy_delta", supp.get("accuracy_delta_b_minus_a_pp", 0.0)))
    ew = data.get("EWoK", {})
    # research used `pair_key_by_subtask` with group labels being EWoK subtask names but stored domain too.
    # The saved note gives exact net item gains by Domain for view-vs-adjbreak and adjbreak-vs-repeat.
    # Prefer the saved JSON if it has by_domain; otherwise reconstruct from by_subtask domain labels if available.
    for cand_key in [f"{pair_key}_by_domain", f"{pair_key}_by_subtask"]:
        gd = ew.get(cand_key)
        if isinstance(gd, dict):
            for k, rec in gd.items():
                if k in RELATIONAL + ADJ_INDEPENDENT:
                    vec[k] = float(rec.get("accuracy_delta", rec.get("accuracy_delta_b_minus_a_pp", 0.0)))
            if all(k in vec for k in RELATIONAL + ADJ_INDEPENDENT):
                break
    return vec


def transition_vector(transitions: dict[str, Any], contrast_key: str) -> dict[str, float]:
    # Positive means first non-full arm is better than the comparator in contrast_key's string direction.
    out = {}
    supp = transitions.get("Supplement", {}).get(contrast_key)
    if supp:
        out["Supplement"] = float(supp["accuracy_delta_b_minus_a_pp"])
    ewgroups = transitions.get("EWoK_groups", {})
    for g in RELATIONAL + ADJ_INDEPENDENT:
        rec = ewgroups.get(g, {}).get(contrast_key)
        if rec:
            out[g] = float(rec["accuracy_delta_b_minus_a_pp"])
    return out


def cosine(a: dict[str, float], b: dict[str, float], keys: list[str]) -> float | None:
    xs = [float(a[k]) for k in keys if k in a and k in b]
    ys = [float(b[k]) for k in keys if k in a and k in b]
    if len(xs) < 2:
        return None
    na = math.sqrt(sum(x*x for x in xs)); nb = math.sqrt(sum(y*y for y in ys))
    if na == 0 or nb == 0:
        return None
    return sum(x*y for x, y in zip(xs, ys)) / (na * nb)


def corr(a: dict[str, float], b: dict[str, float], keys: list[str]) -> float | None:
    xs = [float(a[k]) for k in keys if k in a and k in b]
    ys = [float(b[k]) for k in keys if k in a and k in b]
    if len(xs) < 3:
        return None
    mx = statistics.mean(xs); my = statistics.mean(ys)
    vx = sum((x-mx)**2 for x in xs); vy = sum((y-my)**2 for y in ys)
    if vx == 0 or vy == 0:
        return None
    return sum((x-mx)*(y-my) for x,y in zip(xs,ys)) / math.sqrt(vx*vy)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--full-json", default=str(DEFAULT_FULL))
    ap.add_argument("--arm-json", action="append", default=[], help="NAME=path")
    ap.add_argument("--output_dir", default=str(DEFAULT_OUT))
    args = ap.parse_args()

    out_dir = pathlib.Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    arms = {"full": load_json(pathlib.Path(args.full_json))}
    for spec in args.arm_json:
        name, sep, path = spec.partition("=")
        if not sep:
            raise SystemExit(f"Bad --arm-json {spec}; expected NAME=path")
        arms[name] = load_json(pathlib.Path(path))

    scores = score_table(arms)
    items = build_items(arms)
    micro = official_micro_report(arms, items)
    names = [n for n in arms if n != "full"]
    transitions: dict[str, Any] = {}
    for task in ITEM_TASKS:
        transitions[task] = {}
        for n in names:
            transitions[task][f"{n}_minus_full"] = transition(items[task]["full"], items[task][n], "full", n)
            transitions[task][f"full_minus_{n}"] = transition(items[task][n], items[task]["full"], n, "full")
        for i, a in enumerate(names):
            for b in names[i+1:]:
                transitions[task][f"{a}_minus_{b}"] = transition(items[task][b], items[task][a], b, a)
                transitions[task][f"{b}_minus_{a}"] = transition(items[task][a], items[task][b], a, b)
        if task in {"Supplement", "COMPS"}:
            transitions[task]["by_subtask"] = {}
            for i, a in enumerate(names):
                for b in names[i+1:]:
                    transitions[task]["by_subtask"][f"{a}_minus_{b}"] = by_group_transitions(items[task], b, a, "subtask")
        if task == "EWoK":
            transitions[task]["by_domain"] = {}
            for i, a in enumerate(names):
                for b in names[i+1:]:
                    transitions[task]["by_domain"][f"{a}_minus_{b}"] = by_group_transitions(items[task], b, a, "domain")

    # EWoK group transitions and domain-level vector for each arm contrast.
    ewok_group = {}
    ewok_domain = {}
    contrast_keys = []
    for n in names:
        contrast_keys.extend([f"{n}_minus_full", f"full_minus_{n}"])
    for i, a in enumerate(names):
        for b in names[i+1:]:
            contrast_keys.extend([f"{a}_minus_{b}", f"{b}_minus_{a}"])
    for group_name, domains in [("relational_domains", RELATIONAL), ("adjacency_independent_domains", ADJ_INDEPENDENT)]:
        ewok_group[group_name] = {"domains": domains, "scores": {}, "contrasts": {}}
        for arm in arms:
            ewok_group[group_name]["scores"][arm] = aggregate_accuracy(items["EWoK"][arm], "domain", domains)
        for ck in contrast_keys:
            b, _, a = ck.partition("_minus_")
            if a in arms and b in arms:
                ewok_group[group_name]["contrasts"][ck] = transition(group_items(items["EWoK"][a], "domain", domains), group_items(items["EWoK"][b], "domain", domains), a, b)
    for dom in sorted(set(str(x.get("domain")) for arm in arms for x in items["EWoK"][arm])):
        ewok_domain[dom] = {}
        for ck in contrast_keys:
            b, _, a = ck.partition("_minus_")
            if a in arms and b in arms:
                ewok_domain[dom][ck] = transition(group_items(items["EWoK"][a], "domain", [dom]), group_items(items["EWoK"][b], "domain", [dom]), a, b)
    transitions["EWoK_groups"] = ewok_group
    transitions["EWoK_domain_vectors"] = ewok_domain

    # Historical vectors and similarity.  Use research JSON `accuracy_delta` values
    # directly, which are all on the same percentage-point scale for Supplement and
    # EWoK domain files.  Avoid mixing note-level net-item counts with percentages.
    hist = {
        "view_vs_adjbreak": load_step211_vector("view_vs_adjbreak"),
        "view_vs_repeat": load_step211_vector("view_vs_repeat"),
    }
    hist_source = {
        "path": str(research),
        "scale": "accuracy_delta_percentage_points_from_saved_Step211_JSON",
        "required_profile_keys": HIST_VECTOR_KEYS,
        "missing_keys": {name: [k for k in HIST_VECTOR_KEYS if k not in vec] for name, vec in hist.items()},
    }
    intervention_vectors = {}
    vector_similarity = {}
    for ck in contrast_keys:
        vec = {}
        # Supplement and EWoK domains from direct transitions. Positive follows contrast key direction.
        supp = transitions.get("Supplement", {}).get(ck)
        if supp:
            vec["Supplement"] = float(supp["accuracy_delta_b_minus_a_pp"])
        for dom in RELATIONAL + ADJ_INDEPENDENT:
            dr = ewok_domain.get(dom, {}).get(ck)
            if dr:
                vec[dom] = float(dr["accuracy_delta_b_minus_a_pp"])
        intervention_vectors[ck] = vec
        vector_similarity[ck] = {}
        for hname, hvec in hist.items():
            if not isinstance(hvec, dict):
                continue
            use_keys = [k for k in HIST_VECTOR_KEYS if k in hvec and k in vec]
            vector_similarity[ck][hname] = {
                "common_keys": use_keys,
                "sign_agreement_fraction": None if not use_keys else round(sum(1 for k in use_keys if (vec[k] == 0 and hvec[k] == 0) or (vec[k] > 0) == (hvec[k] > 0)) / len(use_keys), 6),
                "cosine": None if cosine(vec, hvec, use_keys) is None else round(cosine(vec, hvec, use_keys), 6),
                "pearson": None if corr(vec, hvec, use_keys) is None else round(corr(vec, hvec, use_keys), 6),
            }

    payload = {
        "status": "PACKED_TARGETSELECT_ENDPOINT_SIGNATURE_V2",
        "meaning": "Strengthened official-task transition readout for packed target-selective endpoint arms using compound item keys and uncertainty. Primary mechanism contrast is drop_abs versus drop_copied_word; historical full is a shape reference.",
        "inputs": {"full_json": args.full_json, "arm_jsons": args.arm_json},
        "cheap7": scores,
        "official_vs_micro": micro,
        "item_transitions": transitions,
        "historical_vectors": hist,
        "historical_vector_source": hist_source,
        "intervention_vectors": intervention_vectors,
        "vector_similarity_to_historical": vector_similarity,
        "interpretation": {
            "primary_estimand": "drop_abs_minus_drop_copied_word arm-to-arm contrast because both arms are trained by the same fast implementation; historical full supplies reference shape and scale.",
            "load_bearing_support": "Requires copied_word_minus_drop_abs to align with historical compact-view Supplement and relational-EWoK advantages, while local denoising probe shows source_absent_content-specific degradation under drop_abs.",
            "globalpiqa_warning": "GlobalPIQA has tiny item counts; do not use a GlobalPIQA-only delta as mechanism evidence.",
        },
    }
    out = out_dir / "packed_targetselect_endpoint_signature_v2.json"
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    md = ["# research packed target-selective endpoint signature v2", "", f"JSON: `{out}`", "", "## Cheap7-style scores", "", "| arm | " + " | ".join(SCORE_COLUMNS + ["equal7"]) + " |", "|---" + "|---:" * (len(SCORE_COLUMNS) + 1) + "|"]
    for arm, sc in scores["scores"].items():
        md.append("| " + arm + " | " + " | ".join("NA" if sc.get(c) is None else f"{float(sc[c]):.3f}" for c in SCORE_COLUMNS + ["cheap7_equal_mean"]) + " |")
    md += ["", "## Primary arm-to-arm deltas", "", "| contrast | Supplement | EWoK | COMPS | GlobalPIQA | equal7 |", "|---|---:|---:|---:|---:|---:|"]
    for ck, de in scores["deltas"].items():
        md.append(f"| {ck} | {de.get('Supplement')} | {de.get('EWoK')} | {de.get('COMPS')} | {de.get('GlobalPIQA')} | {de.get('cheap7_equal_mean')} |")
    md += ["", "## EWoK predeclared groups", ""]
    for g, rec in ewok_group.items():
        md.append(f"### {g}")
        md.append("| arm | n | micro_acc |")
        md.append("|---|---:|---:|")
        for arm, s in rec["scores"].items():
            md.append(f"| {arm} | {s['n']} | {s['micro_accuracy']:.3f} |")
        md.append("")
    md += ["## Vector similarity", ""]
    for ck, sims in vector_similarity.items():
        md.append(f"- `{ck}`: " + json.dumps(sims, ensure_ascii=False))
    (out_dir / "packed_targetselect_endpoint_signature_v2.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "out": str(out), "cheap7": scores, "ewok_groups": ewok_group, "vector_similarity": vector_similarity}, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
