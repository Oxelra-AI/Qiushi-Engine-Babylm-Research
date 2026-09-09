#!/usr/bin/env python3
"""research: official Entity deployment readout for the state-update intervention.

The research/052 template-slot T/U/N probe suggests the intervention may shift
preference toward source-state retention when a recent update concerns an entity
already present in the source, rather than installing precise entity-gated update
use.  This script tests the corresponding BabyLM Entity prediction at matched
checkpoints by running the official strict Entity evaluator and then stratifying
prediction rows by parsed relevant updates, irrelevant operations, and post-last-
relevant-operation distance.

Main prediction recorded before scoring:
  - if the learned movement is source-retention/anti-recent-competitor, accuracy
    should improve most for relevant_updates==0 items with irrelevant operations;
  - it should not improve, and may cost, items where the queried entity's own
    last relevant update supplies the answer (especially rel>=3 and/or no later
    irrelevant operations);
  - a clear rel>=3 gain would contradict the template readout and imply the
    probe underestimated entity-conditioned binding.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import csv
import json
import math
import os
import pathlib
import re
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from typing import Any


def find_root() -> pathlib.Path:
    return _PUBLIC_ROOT


ROOT = find_root()
WS = ROOT / "experiments/archive/relation_learning"
STRICT = ROOT / "experiments/archive/representation_and_objectives/data/pristine_official_coordinate/babylm-eval/strict"
ENTITY_DATA = STRICT / "evaluation_data/full_eval/entity_tracking"
NLP_DATA_ROOT = ROOT / "experiments/archive/initial_model_studies/data/nltk_data"
OUT_DEFAULT = WS / "data/state_update_entity_eval"

MODEL_ROOTS = {
    "seed43022_base": {
        "seed": "43022",
        "arm": "base",
        "root": ROOT / "experiments/archive/frontier_consolidation/training/runs/adapter128_scale1p75_h100M100M_seed43022_official_ladder/hf_model",
        "description": "frontier_consolidation research compact-view-reinvest/coherent-prestate seed43022 base stream",
    },
    "seed43022_intervention": {
        "seed": "43022",
        "arm": "intervention",
        "root": ROOT / "experiments/archive/relation_learning/training/runs/state_update_plainuse_seed43022/hf_model",
        "description": "relation_learning state-update plain-use replacement seed43022, 99,909,920 words",
    },
    "seed43122_base": {
        "seed": "43122",
        "arm": "base",
        "root": ROOT / "experiments/archive/frontier_consolidation/training/runs/adapter128_scale1p75_seed43122_dense100M/hf_model",
        "description": "frontier_consolidation research seed43122 base stream",
    },
    "seed43122_intervention": {
        "seed": "43122",
        "arm": "intervention",
        "root": ROOT / "experiments/archive/relation_learning/training/runs/state_update_plainuse_seed43122/hf_model",
        "description": "relation_learning state-update plain-use replacement seed43122, 99,909,920 words",
    },
}

CHECKPOINTS_DEFAULT = ["chck_86M", "final"]
ENTITY_FILES = ["regular.jsonl", "ambiref.jsonl", "move_contents.jsonl"]


def now_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def rel(p: pathlib.Path | str) -> str:
    pp = pathlib.Path(p)
    try:
        return str(pp.relative_to(ROOT))
    except Exception:
        return str(pp)


def read_json(path: pathlib.Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: pathlib.Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_csv(path: pathlib.Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        keys: list[str] = []
        seen = set()
        for r in rows:
            for k in r:
                if k not in seen:
                    seen.add(k); keys.append(k)
        fieldnames = keys
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def checkpoint_path(root: pathlib.Path, checkpoint: str) -> pathlib.Path:
    if checkpoint in {"final", "root", "hf_model"}:
        return root
    return root / checkpoint


def parse_score(text: str) -> float | None:
    for pat in [
        r"### AVERAGE [A-Z_ '\\-]*\s*\n\s*([+-]?[0-9]+(?:\.[0-9]+)?)",
        r"AVERAGE ACCURACY\s*\n\s*([+-]?[0-9]+(?:\.[0-9]+)?)",
    ]:
        m = re.search(pat, text)
        if m:
            val = float(m.group(1))
            return val if math.isfinite(val) and -5.0 <= val <= 105.0 else None
    return None


def norm_answer(s: Any) -> str:
    t = str(s or "").strip().lower()
    t = re.sub(r"\s+", " ", t)
    return t.rstrip(" .")


def read_jsonl(path: pathlib.Path):
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def split_entity_prefix(prefix: str) -> tuple[str, list[str], str]:
    parts = [x.strip() for x in re.split(r"(?<=\.)\s+", prefix.strip()) if x.strip()]
    if len(parts) < 2:
        return prefix.strip(), [], ""
    return parts[0], parts[1:-1], parts[-1]


def parse_query_box(query: str) -> int | None:
    m = re.search(r"Box\s+(\d+)\s+contains\s*$", query.strip())
    return int(m.group(1)) if m else None


def parse_initial_contents(initial_sentence: str) -> dict[int, str]:
    s = initial_sentence.strip()
    if s.endswith("."):
        s = s[:-1]
    out: dict[int, str] = {}
    for m in re.finditer(r"Box\s+(\d+)\s+contains\s+(.*?)(?=,\s*Box\s+\d+\s+contains\s+|$)", s):
        out[int(m.group(1))] = m.group(2).strip()
    return out


def op_boxes(op: str) -> dict[str, int | None]:
    out = {"from_box": None, "to_box": None, "into_box": None, "any_box": None}
    boxes = [int(x) for x in re.findall(r"Box\s+(\d+)", op)]
    if boxes:
        out["any_box"] = boxes[0]
    for key, pat in [("from_box", r"from\s+Box\s+(\d+)"), ("to_box", r"to\s+Box\s+(\d+)"), ("into_box", r"into\s+Box\s+(\d+)")]:
        m = re.search(pat, op)
        if m:
            out[key] = int(m.group(1))
    return out


def op_relevant_to_query(op: str, qbox: int | None) -> bool:
    if qbox is None:
        return False
    b = op_boxes(op)
    lo = op.lower().strip()
    if lo.startswith("move"):
        return b["from_box"] == qbox or b["to_box"] == qbox
    if lo.startswith("remove"):
        return b["from_box"] == qbox
    if lo.startswith("put"):
        return b["into_box"] == qbox
    return qbox in {v for v in b.values() if v is not None}


def load_entity_metadata() -> dict[tuple[str, int], dict[str, Any]]:
    meta: dict[tuple[str, int], dict[str, Any]] = {}
    skipped_nothing = 0
    for fn in ENTITY_FILES:
        typ = fn[:-6]
        counters: dict[int, int] = defaultdict(int)
        for obj in read_jsonl(ENTITY_DATA / fn):
            # The official evaluation filters "nothing" options. Match the prior research row universe.
            if any("nothing" in str(o).lower() for o in obj.get("options", [])):
                skipped_nothing += 1
                continue
            reported = int(obj["numops"])
            uid = f"{typ}_{reported}_ops"
            item_index = counters[reported]
            counters[reported] += 1
            initial, ops, query = split_entity_prefix(obj.get("input_prefix", ""))
            qbox = parse_query_box(query)
            relevant_flags = [op_relevant_to_query(op, qbox) for op in ops]
            rel_updates = int(sum(relevant_flags))
            last_rel_idx = max([i for i, x in enumerate(relevant_flags) if x], default=-1)
            ops_after_last_rel = (len(ops) - 1 - last_rel_idx) if last_rel_idx >= 0 else len(ops)
            init_map = parse_initial_contents(initial)
            stale = init_map.get(qbox) if qbox is not None else None
            options = [str(x) for x in obj.get("options", [])]
            stale_idx = None
            if stale is not None:
                for oi, opt in enumerate(options):
                    if norm_answer(opt) == norm_answer(stale):
                        stale_idx = oi
                        break
            rec = {
                "uid": uid,
                "entity_type": typ,
                "reported_numops": reported,
                "item_index": item_index,
                "sample_id": obj.get("sample_id"),
                "example_id": obj.get("example_id"),
                "query_box": qbox if qbox is not None else -1,
                "total_ops": len(ops),
                "relevant_updates": rel_updates,
                "irrelevant_ops": len(ops) - rel_updates,
                "ops_after_last_relevant": ops_after_last_rel,
                "last_relevant_is_final_op": int(last_rel_idx == len(ops) - 1 and last_rel_idx >= 0),
                "gold": options[0] if options else "",
                "stale_initial": stale or "",
                "stale_available": int(stale_idx is not None),
                "stale_is_gold": int(stale_idx == 0) if stale_idx is not None else 0,
                "input_prefix_words": len(str(obj.get("input_prefix", "")).split()),
            }
            rec["rel0_with_irrelevant_ops"] = int(rel_updates == 0 and len(ops) > 0)
            rec["rel_ge3"] = int(rel_updates >= 3)
            rec["last_relevant_answer_no_later_ops"] = int(rel_updates > 0 and ops_after_last_rel == 0)
            meta[(uid, item_index)] = rec
    # Store skipped count in a sentinel; callers ignore it for item lookup.
    meta[("__meta__", -1)] = {"skipped_nothing": skipped_nothing}
    return meta


def eval_env(out_dir: pathlib.Path, model_name: str, checkpoint: str, gpu: int) -> dict[str, str]:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["TOKENIZERS_PARALLELISM"] = "false"
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    if NLP_DATA_ROOT.exists():
        env["NLTK_DATA"] = str(NLP_DATA_ROOT.resolve())
    cache = out_dir / "cache" / f"{model_name}_{checkpoint}"
    tmp = out_dir / "tmp" / f"{model_name}_{checkpoint}"
    for key, p in {
        "HF_HOME": cache,
        "HF_HUB_CACHE": cache / "hub",
        "TRANSFORMERS_CACHE": cache / "transformers",
        "HF_MODULES_CACHE": cache / "modules",
        "HF_DATASETS_CACHE": cache / "datasets",
        "TMPDIR": tmp,
    }.items():
        p.mkdir(parents=True, exist_ok=True)
        env[key] = str(p.resolve())
    return env


def run_entity_eval(model_name: str, checkpoint: str, gpu: int, out_dir: pathlib.Path, timeout: int, force: bool = False) -> dict[str, Any]:
    info = MODEL_ROOTS[model_name]
    model_path = checkpoint_path(info["root"], checkpoint)
    target = f"step053_{model_name}_{checkpoint}"
    target_dir = out_dir / "outputs" / target / "Entity"
    log_path = out_dir / "logs" / f"{target}_Entity.log"
    payload_path = out_dir / "per_target" / f"{target}.json"
    if not (model_path.exists() and (model_path / "config.json").exists()):
        rec = {"status": "missing_model", "model_name": model_name, "checkpoint": checkpoint, "model_path": rel(model_path)}
        write_json(payload_path, rec)
        return rec
    if payload_path.exists() and not force:
        old = read_json(payload_path)
        rec = old.get("entity_eval", old)
        pred = rec.get("predictions")
        if rec.get("returncode") == 0 and rec.get("score") is not None and pred and (ROOT / pred).exists():
            print(f"[SKIP] {model_name} {checkpoint} score={rec.get('score')}", flush=True)
            return rec | {"status": "skip_existing"}
    target_dir.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    argv = [
        sys.executable, "-B", "-m", "evaluation_pipeline.sentence_zero_shot.run",
        "--model_path_or_name", str(model_path.resolve()),
        "--backend", "mlm",
        "--task", "entity_tracking",
        "--data_path", str(ENTITY_DATA.resolve()),
        "--revision_name", target,
        "--save_predictions",
        "--batch_size", "128",
        "--non_causal_batch_size", "64",
        "--output_dir", str(target_dir.resolve()),
    ]
    print(f"[RUN] {model_name} {checkpoint} on GPU {gpu}", flush=True)
    t0 = time.time()
    err = None
    try:
        with log_path.open("w", encoding="utf-8") as fh:
            fh.write(json.dumps({"event": "start", "utc": now_utc(), "model_name": model_name, "checkpoint": checkpoint, "model_path": rel(model_path)}) + "\n")
            proc = subprocess.run(argv, cwd=str(STRICT.resolve()), env=eval_env(out_dir, model_name, checkpoint, gpu), stdout=fh, stderr=subprocess.STDOUT, text=True, timeout=timeout)
            rc = proc.returncode
    except subprocess.TimeoutExpired:
        rc = -124
        err = f"timeout after {timeout}s"
    elapsed = round(time.time() - t0, 1)
    report_files = sorted(target_dir.rglob("best_temperature_report.txt"), key=lambda p: p.stat().st_mtime)
    pred_files = sorted(target_dir.rglob("predictions.json"), key=lambda p: p.stat().st_mtime)
    score = parse_score(report_files[-1].read_text(encoding="utf-8", errors="replace")) if report_files else None
    rec = {
        "status": "ok" if rc == 0 and score is not None and pred_files else "failed",
        "model_name": model_name,
        "seed": info["seed"],
        "arm": info["arm"],
        "checkpoint": checkpoint,
        "model_path": rel(model_path),
        "returncode": rc,
        "score": score,
        "elapsed_sec": elapsed,
        "gpu": gpu,
        "log": rel(log_path),
        "report": rel(report_files[-1]) if report_files else None,
        "predictions": rel(pred_files[-1]) if pred_files else None,
        "description": info["description"],
    }
    if err:
        rec["error"] = err
    write_json(payload_path, {"target": target, "entity_eval": rec, "updated_utc": now_utc()})
    print(f"[DONE] {model_name} {checkpoint} score={score} rc={rc} elapsed={elapsed}s", flush=True)
    return rec


def parse_pred_id(pid: str) -> tuple[str, int] | None:
    # Example: ambiref_0_ops_123 -> uid=ambiref_0_ops, item_index=123
    m = re.match(r"^(.+_\d+_ops)_(\d+)$", str(pid))
    if not m:
        return None
    return m.group(1), int(m.group(2))


def flatten_predictions(pred_path: pathlib.Path) -> list[tuple[str, int, str, str]]:
    obj = read_json(pred_path)
    rows: list[tuple[str, int, str, str]] = []
    for key, payload in obj.items():
        if isinstance(payload, dict):
            plist = payload.get("predictions", [])
        else:
            plist = payload
        for idx, pred in enumerate(plist):
            if isinstance(pred, dict):
                pid = str(pred.get("id") or f"{key}_{idx}")
                text = str(pred.get("pred", ""))
            else:
                pid = f"{key}_{idx}"
                text = str(pred)
            parsed = parse_pred_id(pid)
            if parsed is None:
                uid, item_index = key, idx
            else:
                uid, item_index = parsed
            rows.append((uid, item_index, pid, text))
    return rows


def attach_predictions(eval_recs: list[dict[str, Any]], meta: dict[tuple[str, int], dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for ev in eval_recs:
        pred_rel = ev.get("predictions")
        if not pred_rel:
            continue
        pred_path = ROOT / str(pred_rel)
        if not pred_path.exists():
            continue
        for uid, item_index, pred_id, pred_text in flatten_predictions(pred_path):
            m = meta.get((uid, item_index))
            if not m:
                continue
            gold = m.get("gold", "")
            correct = int(norm_answer(pred_text) == norm_answer(gold))
            pred_is_stale = int(bool(m.get("stale_initial")) and norm_answer(pred_text) == norm_answer(m.get("stale_initial")))
            r = {
                "model_name": ev["model_name"],
                "seed": ev["seed"],
                "arm": ev["arm"],
                "checkpoint": ev["checkpoint"],
                "uid": uid,
                "item_index": item_index,
                "pred_id": pred_id,
                "pred": pred_text,
                "correct": correct,
                "pred_is_stale_initial": pred_is_stale,
            }
            r.update(m)
            rows.append(r)
    return rows


def item_groups(r: dict[str, Any]) -> list[str]:
    relu = int(r["relevant_updates"])
    total = int(r["total_ops"])
    irr = int(r["irrelevant_ops"])
    post = int(r["ops_after_last_relevant"])
    groups = [
        "ALL",
        f"type_{r['entity_type']}",
        f"reported_numops_{r['reported_numops']}",
        f"rel_updates_{relu}",
        f"total_ops_{total}",
    ]
    if relu == 0:
        groups += ["rel_eq0", "rel_eq0_total_ops0" if total == 0 else "rel_eq0_irrelevant_ops_gt0"]
        if total > 0:
            groups.append(f"rel_eq0_irrelevant_ops_{irr}")
            if total <= 3:
                groups.append("rel_eq0_irrelevant_ops_1to3")
            elif total <= 6:
                groups.append("rel_eq0_irrelevant_ops_4to6")
            else:
                groups.append("rel_eq0_irrelevant_ops_ge7")
    else:
        groups += ["rel_ge1", f"rel_ge1_postrel_ops_{post}"]
        groups.append("rel_ge1_postrel_ops0" if post == 0 else "rel_ge1_postrel_ops_gt0")
        if relu >= 3:
            groups.append("rel_ge3")
            groups.append("rel_ge3_postrel_ops0" if post == 0 else "rel_ge3_postrel_ops_gt0")
        if relu in {1, 2, 3, 4, 5}:
            groups.append(f"rel{relu}_postrel_ops0" if post == 0 else f"rel{relu}_postrel_ops_gt0")
        if int(r.get("stale_available", 0)) and not int(r.get("stale_is_gold", 0)):
            groups += ["stale_available_not_gold", f"rel{relu}_stale_available_not_gold"]
    return groups


def mean(xs: list[float]) -> float:
    xs = [float(x) for x in xs if math.isfinite(float(x))]
    return statistics.mean(xs) if xs else float("nan")


def se(xs: list[float]) -> float:
    xs = [float(x) for x in xs if math.isfinite(float(x))]
    if len(xs) <= 1:
        return float("nan")
    return statistics.stdev(xs) / math.sqrt(len(xs))


def summarize_rows(pred_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
    for r in pred_rows:
        for g in item_groups(r):
            groups[(r["seed"], r["arm"], r["checkpoint"], g)].append(r)
    out = []
    for (seed, arm, ck, g), vals in sorted(groups.items()):
        wrong = [v for v in vals if not int(v["correct"])]
        stale_cand = [v for v in vals if int(v.get("stale_available", 0)) and not int(v.get("stale_is_gold", 0))]
        wrong_stale_cand = [v for v in stale_cand if not int(v["correct"])]
        out.append({
            "seed": seed,
            "arm": arm,
            "checkpoint": ck,
            "group": g,
            "n": len(vals),
            "accuracy_pct": 100.0 * sum(int(v["correct"]) for v in vals) / len(vals),
            "mean_relevant_updates": mean([v["relevant_updates"] for v in vals]),
            "mean_total_ops": mean([v["total_ops"] for v in vals]),
            "mean_irrelevant_ops": mean([v["irrelevant_ops"] for v in vals]),
            "mean_ops_after_last_relevant": mean([v["ops_after_last_relevant"] for v in vals]),
            "stale_available_not_gold_n": len(stale_cand),
            "stale_pick_pct_among_wrong_stale_available_not_gold": 100.0 * sum(int(v["pred_is_stale_initial"]) for v in wrong_stale_cand) / len(wrong_stale_cand) if wrong_stale_cand else float("nan"),
            "stale_pick_pct_among_all_wrong": 100.0 * sum(int(v["pred_is_stale_initial"]) for v in wrong) / len(wrong) if wrong else float("nan"),
        })
    return out


def paired_deltas(summary_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    idx = {(r["seed"], r["checkpoint"], r["arm"], r["group"]): r for r in summary_rows}
    groups = sorted({r["group"] for r in summary_rows})
    delta_rows = []
    for seed in sorted({r["seed"] for r in summary_rows}):
        for ck in sorted({r["checkpoint"] for r in summary_rows}):
            for g in groups:
                kb = (seed, ck, "base", g)
                ki = (seed, ck, "intervention", g)
                if kb not in idx or ki not in idx:
                    continue
                b, a = idx[kb], idx[ki]
                if int(b["n"]) != int(a["n"]):
                    # The official predictions should have identical row universes.
                    pass
                delta_rows.append({
                    "seed": seed,
                    "checkpoint": ck,
                    "group": g,
                    "n": int(a["n"]),
                    "delta_accuracy_pct_intervention_minus_base": float(a["accuracy_pct"]) - float(b["accuracy_pct"]),
                    "base_accuracy_pct": float(b["accuracy_pct"]),
                    "intervention_accuracy_pct": float(a["accuracy_pct"]),
                    "delta_stale_pick_pct_among_wrong_stale_available_not_gold": float(a["stale_pick_pct_among_wrong_stale_available_not_gold"]) - float(b["stale_pick_pct_among_wrong_stale_available_not_gold"]),
                    "base_stale_pick_pct_among_wrong_stale_available_not_gold": float(b["stale_pick_pct_among_wrong_stale_available_not_gold"]),
                    "intervention_stale_pick_pct_among_wrong_stale_available_not_gold": float(a["stale_pick_pct_among_wrong_stale_available_not_gold"]),
                    "mean_relevant_updates": float(a["mean_relevant_updates"]),
                    "mean_total_ops": float(a["mean_total_ops"]),
                    "mean_irrelevant_ops": float(a["mean_irrelevant_ops"]),
                    "mean_ops_after_last_relevant": float(a["mean_ops_after_last_relevant"]),
                })
    # Across-seed means by checkpoint and group, plus late mean over scored checkpoints.
    cross = []
    by = defaultdict(list)
    for r in delta_rows:
        by[(r["checkpoint"], r["group"])].append(r)
        by[("late_mean_over_checkpoints", r["group"])].append(r)
    for (ck, g), vals in sorted(by.items()):
        ds = [float(v["delta_accuracy_pct_intervention_minus_base"]) for v in vals]
        cross.append({
            "checkpoint": ck,
            "group": g,
            "n_seed_checkpoint_rows": len(vals),
            "seeds": ",".join(sorted({str(v["seed"]) for v in vals})),
            "mean_delta_accuracy_pct_intervention_minus_base": mean(ds),
            "se_delta_accuracy_pct_intervention_minus_base": se(ds),
            "mean_base_accuracy_pct": mean([v["base_accuracy_pct"] for v in vals]),
            "mean_intervention_accuracy_pct": mean([v["intervention_accuracy_pct"] for v in vals]),
            "mean_n": mean([v["n"] for v in vals]),
            "mean_relevant_updates": mean([v["mean_relevant_updates"] for v in vals]),
            "mean_total_ops": mean([v["mean_total_ops"] for v in vals]),
        })
    return delta_rows, cross


def write_prediction_note(path: pathlib.Path) -> None:
    path.write_text(
        "# research Entity deployment prediction\n\n"
        "Written before running the research Entity scores. The research/052 template-slot margins show that the base already uses true target updates, while the intervention mostly shifts source-state retention when an update concerns an entity present in the source and leaves the foreign-update contrast near flat. Therefore the Entity prediction is: (1) possible gains should concentrate on `rel_eq0_irrelevant_ops_gt0`, where irrelevant operations create a recency trap for a source-state answer; (2) items with the queried entity's own relevant updates, especially `rel_ge3` and `postrel_ops0`, should be flat or lower if the learned relation is a coarse anti-recency/source-retention shift; (3) a clear rel_ge3 gain would contradict the template readout and indicate that the template probe underestimates entity-conditioned binding. Interpret any aggregate score against displaced ALN and the 99,909,920-word intervention exposure.\n",
        encoding="utf-8",
    )


def write_note(path: pathlib.Path, eval_recs: list[dict[str, Any]], delta_rows: list[dict[str, Any]], cross_rows: list[dict[str, Any]]) -> None:
    key_groups = [
        "ALL", "rel_eq0", "rel_eq0_total_ops0", "rel_eq0_irrelevant_ops_gt0", "rel_eq0_irrelevant_ops_1to3", "rel_eq0_irrelevant_ops_4to6", "rel_eq0_irrelevant_ops_ge7",
        "rel_ge1", "rel_ge1_postrel_ops0", "rel_ge1_postrel_ops_gt0", "rel_updates_1", "rel_updates_2", "rel_updates_3", "rel_updates_4", "rel_updates_5", "rel_ge3", "rel_ge3_postrel_ops0", "rel_ge3_postrel_ops_gt0",
        "stale_available_not_gold",
    ]
    def f(x: Any) -> str:
        try:
            y = float(x)
            if math.isnan(y):
                return "NA"
            return f"{y:+.2f}"
        except Exception:
            return "NA"
    lines = []
    lines.append("# research state-update Entity deployment readout\n\n")
    lines.append(f"Created UTC: {now_utc()}\n\n")
    lines.append("## Official Entity scores\n\n")
    lines.append("| seed | arm | checkpoint | score | elapsed s | predictions |\n")
    lines.append("|---:|---|---|---:|---:|---|\n")
    for ev in sorted(eval_recs, key=lambda r: (r.get("seed"), r.get("checkpoint"), r.get("arm"))):
        lines.append(f"| {ev.get('seed')} | {ev.get('arm')} | {ev.get('checkpoint')} | {f(ev.get('score'))} | {ev.get('elapsed_sec')} | `{ev.get('predictions')}` |\n")
    lines.append("\n## Within-seed intervention minus base by Entity operation structure\n\n")
    lines.append("Positive delta means the state-update intervention is more accurate than its matched base for the same seed/checkpoint/item group.\n\n")
    lines.append("| seed | ck | group | n | base | int | delta |\n")
    lines.append("|---:|---|---|---:|---:|---:|---:|\n")
    for r in sorted(delta_rows, key=lambda r: (r["seed"], r["checkpoint"], key_groups.index(r["group"]) if r["group"] in key_groups else 999, r["group"])):
        if r["group"] not in key_groups:
            continue
        lines.append(f"| {r['seed']} | {r['checkpoint']} | {r['group']} | {r['n']} | {r['base_accuracy_pct']:.2f} | {r['intervention_accuracy_pct']:.2f} | {r['delta_accuracy_pct_intervention_minus_base']:+.2f} |\n")
    lines.append("\n## Across-seed / late compact rows\n\n")
    lines.append("| checkpoint | group | rows | mean base | mean int | mean delta | se |\n")
    lines.append("|---|---|---:|---:|---:|---:|---:|\n")
    for r in sorted(cross_rows, key=lambda r: (r["checkpoint"], key_groups.index(r["group"]) if r["group"] in key_groups else 999, r["group"])):
        if r["group"] not in key_groups:
            continue
        lines.append(f"| {r['checkpoint']} | {r['group']} | {r['n_seed_checkpoint_rows']} | {r['mean_base_accuracy_pct']:.2f} | {r['mean_intervention_accuracy_pct']:.2f} | {r['mean_delta_accuracy_pct_intervention_minus_base']:+.2f} | {f(r['se_delta_accuracy_pct_intervention_minus_base'])} |\n")
    lines.append("\n## Interpretation reminders\n\n")
    lines.append("The pre-scored prediction was gain on `rel_eq0_irrelevant_ops_gt0` and cost/flat behavior on relevant-update items if the arm learned a coarse source-retention/anti-recency relation rather than entity-gated state binding. A rel_ge3 gain would contradict that margin-based reading. Keep the 99,909,920-word exposure and replacement of inherited ALN rows visible; this script does not run cheap7, ordinary held-out, SuperGLUE, or AoA.\n")
    path.write_text("".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", nargs="+", default=list(MODEL_ROOTS), choices=sorted(MODEL_ROOTS))
    ap.add_argument("--checkpoints", nargs="+", default=CHECKPOINTS_DEFAULT)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--out-dir", default=str(OUT_DEFAULT))
    ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("--plan-only", action="store_true")
    ap.add_argument("--analyze-only", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_prediction_note(out_dir / "pre_scored_entity_prediction.md")
    plan = {
        "status": "ENTITY_EVAL_PLAN",
        "created_utc": now_utc(),
        "models": args.models,
        "checkpoints": args.checkpoints,
        "gpu": args.gpu,
        "out_dir": rel(out_dir),
        "model_paths": [],
        "prediction": "gain concentrated on rel_eq0_irrelevant_ops_gt0 and flat/cost on own relevant-update items if template-margin readout transfers; rel_ge3 gain would contradict it",
    }
    for name in args.models:
        info = MODEL_ROOTS[name]
        for ck in args.checkpoints:
            p = checkpoint_path(info["root"], ck)
            plan["model_paths"].append({"model_name": name, "seed": info["seed"], "arm": info["arm"], "checkpoint": ck, "path": rel(p), "exists": p.exists() and (p / "config.json").exists()})
    write_json(out_dir / "entity_eval_plan.json", plan)
    if args.plan_only:
        print(json.dumps(plan, indent=2, ensure_ascii=False), flush=True)
        return

    eval_recs: list[dict[str, Any]] = []
    if args.analyze_only:
        for name in args.models:
            for ck in args.checkpoints:
                payload = out_dir / "per_target" / f"step053_{name}_{ck}.json"
                if payload.exists():
                    obj = read_json(payload)
                    eval_recs.append(obj.get("entity_eval", obj))
    else:
        for name in args.models:
            for ck in args.checkpoints:
                eval_recs.append(run_entity_eval(name, ck, args.gpu, out_dir, args.timeout, force=args.force))

    write_json(out_dir / "entity_eval_results.json", {"status": "ENTITY_EVAL_RESULTS", "results": eval_recs, "updated_utc": now_utc()})
    meta = load_entity_metadata()
    item_meta = [v for k, v in meta.items() if k[0] != "__meta__"]
    write_csv(out_dir / "entity_item_metadata.csv", item_meta)
    pred_rows = attach_predictions(eval_recs, meta)
    write_csv(out_dir / "entity_prediction_rows.csv", pred_rows)
    summary = summarize_rows(pred_rows)
    write_csv(out_dir / "entity_group_summary.csv", summary)
    deltas, cross = paired_deltas(summary)
    write_csv(out_dir / "entity_intervention_minus_base_by_group.csv", deltas)
    write_csv(out_dir / "entity_cross_seed_summary.csv", cross)
    note_path = out_dir / "state_update_entity_note.md"
    write_note(note_path, eval_recs, deltas, cross)
    done = {
        "status": "STATE_UPDATE_ENTITY_DONE",
        "created_utc": now_utc(),
        "out_dir": rel(out_dir),
        "n_eval_records": len(eval_recs),
        "n_prediction_rows": len(pred_rows),
        "n_entity_items": len(item_meta),
        "outputs": {
            "prediction": rel(out_dir / "pre_scored_entity_prediction.md"),
            "plan": rel(out_dir / "entity_eval_plan.json"),
            "eval_results": rel(out_dir / "entity_eval_results.json"),
            "prediction_rows": rel(out_dir / "entity_prediction_rows.csv"),
            "group_summary": rel(out_dir / "entity_group_summary.csv"),
            "deltas": rel(out_dir / "entity_intervention_minus_base_by_group.csv"),
            "cross_seed": rel(out_dir / "entity_cross_seed_summary.csv"),
            "note": rel(note_path),
        },
    }
    write_json(out_dir / "state_update_entity_done.json", done)
    print(json.dumps(done, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
