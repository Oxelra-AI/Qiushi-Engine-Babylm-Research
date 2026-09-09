#!/usr/bin/env python3
"""research: small EWoK continuous-margin seed-noise probe for full DeBERTa.

This script recomputes official-style EWoK masked-LM margins on a bounded sample
of EWoK rows for the same-architecture full-DeBERTa compact/repeat seed-reference
cells. It is a small follow-up to the binary item-flip analysis: if compact-minus-
repeat continuous margin deltas are correlated across seeds even when binary flips
are unstable, the principle should be formulated over margin allocation rather than
item identity; if they also decorrelate against same-architecture seed spread, the
reproducible result is mainly aggregate family-level reallocation.

The script uses the already validated research official-style EWoK scorer and
runs only on checkpoints that already exist. It performs no training, SuperGLUE,
AoA, upload, or leaderboard action.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import csv
import importlib.util
import json
import math
import pathlib
import statistics
import sys
import time
from typing import Any


def find_user_root() -> pathlib.Path:
    return _PUBLIC_ROOT


USER_ROOT = find_user_root()
STUDY = USER_ROOT / "experiments/archive/frontier_consolidation"
WS = STUDY
PATH = WS / "scripts/ewok_fourway_margin_readout.py"
DEFAULT_OUT = WS / "data/ewok_margin_seed_noise_probe"
LADDER_OUT = WS / "data/full_deberta_seed_ladder_stable_eval"

RUN_DIRS = {
    "seed43022_compact": WS / "training/runs/complianttok_reinvest_seed43022_r2",
    "seed43022_repeat": WS / "training/runs/full_p2c_c2p_abs_repeat_deberta100M_seed43022",
    "seed43122_compact": WS / "training/runs/full_p2c_c2p_abs_compact_deberta100M_seed43122",
    "seed43122_repeat": WS / "training/runs/full_p2c_c2p_abs_repeat_deberta100M_seed43122_retry2",
}
ARM_META = {
    "seed43022_compact": {"seed": 43022, "data_arm": "compact"},
    "seed43022_repeat": {"seed": 43022, "data_arm": "repeat"},
    "seed43122_compact": {"seed": 43122, "data_arm": "compact"},
    "seed43122_repeat": {"seed": 43122, "data_arm": "repeat"},
}
EXISTING_PER_TARGETS: dict[tuple[str, str], pathlib.Path] = {
    ("seed43022_compact", "chck_80M"): WS / "data/legal_mature_treatment_effect_eval/per_target/complianttok_reinvest_seed43022_80M.json",
    ("seed43022_compact", "chck_100M"): WS / "data/compliant_full_eval/per_target/complianttok_reinvest_seed43022.json",
    ("seed43022_repeat", "chck_80M"): WS / "data/architecture_interaction_selected_panel_final/full_repeat/chck_80M/eval/per_target/arch_full_repeat_chck_80M.json",
    ("seed43022_repeat", "chck_100M"): WS / "data/architecture_interaction_selected_panel_final/full_repeat/chck_100M/eval/per_target/arch_full_repeat_chck_100M.json",
}
DEFAULT_CHECKPOINTS = ["chck_80M", "chck_100M"]


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def read_json(path: pathlib.Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_step080():
    if not PATH.exists():
        raise FileNotFoundError(PATH)
    spec = importlib.util.spec_from_file_location("ewok_fourway_margin_readout", PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {PATH}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def ck_words(ck: str) -> int:
    return int(ck[5:-1]) * 1_000_000


def checkpoint_path(arm: str, ck: str) -> pathlib.Path:
    return RUN_DIRS[arm] / "hf_model" / ck


def checkpoint_exists(arm: str, ck: str) -> bool:
    p = checkpoint_path(arm, ck)
    return (p / "model.safetensors").exists() or (p / "pytorch_model.bin").exists()


def per_target_from_ladder(arm: str, ck: str) -> pathlib.Path:
    return LADDER_OUT / "eval" / "per_target" / f"ladder_{arm}_{ck}.json"


def prediction_from_per_target(path: pathlib.Path) -> str:
    if not path.exists():
        return ""
    try:
        obj = read_json(path)
        rec = (obj.get("tasks") or {}).get("EWoK") or {}
        pred = rec.get("predictions") or ""
        p = pathlib.Path(str(pred))
        if pred and (p.is_absolute() and p.exists() or (USER_ROOT / p).exists()):
            return str(pred)
    except Exception:
        return ""
    return ""


def prediction_path_for(arm: str, ck: str) -> str:
    p = EXISTING_PER_TARGETS.get((arm, ck))
    if p is not None:
        pred = prediction_from_per_target(p)
        if pred:
            return pred
    p2 = per_target_from_ladder(arm, ck)
    pred = prediction_from_per_target(p2)
    return pred


def wait_for_file(path: pathlib.Path, timeout_sec: int, poll_sec: int, log_path: pathlib.Path) -> dict[str, Any]:
    start = time.time()
    while True:
        exists = path.exists()
        rec = {"event": "file_wait_sample", "utc": now(), "path": str(path), "exists": exists, "timeout_sec": timeout_sec}
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(json.dumps(rec), flush=True)
        if exists:
            return {"status": "file_ready", "waited_sec": round(time.time() - start, 1), "path": str(path)}
        if time.time() - start > timeout_sec:
            return {"status": "file_wait_timeout", "waited_sec": round(time.time() - start, 1), "path": str(path)}
        time.sleep(poll_sec)


def wait_for_checkpoints(arms: list[str], checkpoints: list[str], timeout_sec: int, poll_sec: int, log_path: pathlib.Path) -> dict[str, Any]:
    start = time.time()
    while True:
        missing = [f"{a}:{ck}" for a in arms for ck in checkpoints if not checkpoint_exists(a, ck)]
        rec = {"event": "checkpoint_wait_sample", "utc": now(), "missing": missing[:20], "missing_count": len(missing)}
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(json.dumps(rec), flush=True)
        if not missing:
            return {"status": "checkpoints_ready", "waited_sec": round(time.time() - start, 1)}
        if time.time() - start > timeout_sec:
            return {"status": "checkpoint_wait_timeout", "waited_sec": round(time.time() - start, 1), "missing": missing}
        time.sleep(poll_sec)


def finite(x: Any) -> bool:
    try:
        return math.isfinite(float(x))
    except Exception:
        return False


def pearson(xs: list[float], ys: list[float]) -> float | None:
    pairs = [(float(x), float(y)) for x, y in zip(xs, ys) if finite(x) and finite(y)]
    if len(pairs) < 3:
        return None
    mx = statistics.mean(x for x, _ in pairs)
    my = statistics.mean(y for _, y in pairs)
    vx = sum((x - mx) ** 2 for x, _ in pairs)
    vy = sum((y - my) ** 2 for _, y in pairs)
    if vx <= 0 or vy <= 0:
        return None
    return sum((x - mx) * (y - my) for x, y in pairs) / math.sqrt(vx * vy)


def summarize_delta(vals: list[float]) -> dict[str, Any]:
    xs = [float(v) for v in vals if finite(v)]
    if not xs:
        return {"n": 0}
    ys = sorted(xs)
    return {
        "n": len(xs),
        "mean": statistics.mean(xs),
        "median": statistics.median(xs),
        "stdev": statistics.stdev(xs) if len(xs) > 1 else 0.0,
        "p10": ys[max(0, int(0.10 * (len(ys) - 1)))],
        "p90": ys[min(len(ys) - 1, int(0.90 * (len(ys) - 1)))],
        "frac_positive": sum(v > 0 for v in xs) / len(xs),
    }


def write_comparison(out_dir: pathlib.Path, model_summaries: dict[str, dict[str, Any]], checkpoints: list[str]) -> dict[str, Any]:
    # Flatten all scored records.
    rows: list[dict[str, Any]] = []
    by_model_uid: dict[tuple[str, str], dict[str, Any]] = {}
    for model_name, summ in model_summaries.items():
        arm, ck = model_name.rsplit("_", 1)
        # model names contain ck like chck_80M; rsplit is not safe. Use stored metadata instead.
        arm = summ["arm"]
        ck = summ["checkpoint_name"]
        for r in summ["records"]:
            row = {
                "model": model_name,
                "arm": arm,
                "seed": ARM_META[arm]["seed"],
                "data_arm": ARM_META[arm]["data_arm"],
                "checkpoint": ck,
                "words": ck_words(ck),
                "uid": r["uid"],
                "domain": r["domain"],
                "index": r["index"],
                "official_margin": r["official_margin"],
                "target2_context_margin": r["target2_context_margin"],
                "interaction": r["interaction"],
                "correct": bool(r["computed_official_pred_correct"]),
                "saved_pred_match": r.get("saved_pred_match"),
            }
            rows.append(row)
            by_model_uid[(model_name, str(r["uid"]))] = row
    flat_csv = out_dir / "ewok_margin_records_flat.csv"
    fields = ["model", "arm", "seed", "data_arm", "checkpoint", "words", "uid", "domain", "index", "official_margin", "target2_context_margin", "interaction", "correct", "saved_pred_match"]
    with flat_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # Compute compact-minus-repeat margin deltas per seed/checkpoint/item.
    delta_rows: list[dict[str, Any]] = []
    for seed in [43022, 43122]:
        for ck in checkpoints:
            comp_model = f"seed{seed}_compact_{ck}"
            rep_model = f"seed{seed}_repeat_{ck}"
            uids = sorted(set(k[1] for k in by_model_uid if k[0] == comp_model) & set(k[1] for k in by_model_uid if k[0] == rep_model))
            for uid in uids:
                c = by_model_uid[(comp_model, uid)]
                r = by_model_uid[(rep_model, uid)]
                delta_rows.append({
                    "seed": seed,
                    "checkpoint": ck,
                    "words": ck_words(ck),
                    "uid": uid,
                    "domain": c["domain"],
                    "compact_margin": c["official_margin"],
                    "repeat_margin": r["official_margin"],
                    "margin_delta": float(c["official_margin"]) - float(r["official_margin"]),
                    "compact_correct": c["correct"],
                    "repeat_correct": r["correct"],
                    "binary_delta": int(c["correct"]) - int(r["correct"]),
                    "interaction_delta": float(c["interaction"]) - float(r["interaction"]),
                })
    delta_csv = out_dir / "ewok_compact_minus_repeat_margin_deltas.csv"
    dfields = ["seed", "checkpoint", "words", "uid", "domain", "compact_margin", "repeat_margin", "margin_delta", "compact_correct", "repeat_correct", "binary_delta", "interaction_delta"]
    with delta_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=dfields)
        w.writeheader()
        for r in delta_rows:
            w.writerow(r)

    delta_index = {(int(r["seed"]), str(r["checkpoint"]), str(r["uid"])): r for r in delta_rows}
    comparisons: dict[str, Any] = {}
    for ck in checkpoints:
        uids = sorted({uid for (seed, c, uid) in delta_index if c == ck and seed == 43022} & {uid for (seed, c, uid) in delta_index if c == ck and seed == 43122})
        x = [float(delta_index[(43022, ck, uid)]["margin_delta"]) for uid in uids]
        y = [float(delta_index[(43122, ck, uid)]["margin_delta"]) for uid in uids]
        bx = [float(delta_index[(43022, ck, uid)]["binary_delta"]) for uid in uids]
        by = [float(delta_index[(43122, ck, uid)]["binary_delta"]) for uid in uids]
        comparisons[f"cross_seed_{ck}"] = {
            "n_common_items": len(uids),
            "pearson_margin_delta": pearson(x, y),
            "pearson_binary_delta": pearson(bx, by),
            "seed43022_margin_delta": summarize_delta(x),
            "seed43122_margin_delta": summarize_delta(y),
            "mean_delta_seed43122_minus_43022": (statistics.mean(y) - statistics.mean(x)) if x and y else None,
        }
    for seed in [43022, 43122]:
        if len(checkpoints) >= 2:
            ck0, ck1 = checkpoints[0], checkpoints[-1]
            uids = sorted({uid for (s, c, uid) in delta_index if c == ck0 and s == seed} & {uid for (s, c, uid) in delta_index if c == ck1 and s == seed})
            x = [float(delta_index[(seed, ck0, uid)]["margin_delta"]) for uid in uids]
            y = [float(delta_index[(seed, ck1, uid)]["margin_delta"]) for uid in uids]
            comparisons[f"within_seed_{seed}_{ck0}_vs_{ck1}"] = {"n_common_items": len(uids), "pearson_margin_delta": pearson(x, y), "first": summarize_delta(x), "last": summarize_delta(y)}

    summary = {
        "status": "EWOK_MARGIN_SEED_NOISE_PROBE_DONE",
        "created_utc": now(),
        "records_csv": str(flat_csv),
        "deltas_csv": str(delta_csv),
        "n_model_records": len(rows),
        "n_delta_records": len(delta_rows),
        "comparisons": comparisons,
        "validation": {name: summ.get("validation_against_saved_predictions") for name, summ in model_summaries.items()},
        "boundary": "Official-style EWoK margin recomputation on a bounded sampled row set; no training, SuperGLUE, AoA, upload, or leaderboard action.",
    }
    (out_dir / "ewok_margin_seed_noise_probe_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = ["# research EWoK continuous-margin seed-noise probe", "", summary["boundary"], "", "## Main comparisons"]
    for k, v in comparisons.items():
        lines.append(f"- {k}: {v}")
    lines += ["", f"Records: `{flat_csv}`", f"Deltas: `{delta_csv}`", f"JSON: `{out_dir / 'ewok_margin_seed_noise_probe_summary.json'}`"]
    (out_dir / "ewok_margin_seed_noise_probe_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT))
    ap.add_argument("--checkpoints", nargs="*", default=DEFAULT_CHECKPOINTS)
    ap.add_argument("--rows-per-domain", type=int, default=10)
    ap.add_argument("--seed", type=int, default=25380)
    ap.add_argument("--domains", nargs="*", default=None)
    ap.add_argument("--hard-domains", action="store_true")
    ap.add_argument("--device", default=None)
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--row-batch-size", type=int, default=8)
    ap.add_argument("--non-causal-batch-size", type=int, default=64)
    ap.add_argument("--wait-for-checkpoints", action="store_true")
    ap.add_argument("--wait-for-file", default="")
    ap.add_argument("--wait-timeout-sec", type=int, default=36_000)
    ap.add_argument("--poll-sec", type=int, default=120)
    ap.add_argument("--plan-only", action="store_true")
    args = ap.parse_args()

    out_dir = pathlib.Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = USER_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "ewok_margin_seed_noise_probe_log.jsonl"
    checkpoints = args.checkpoints
    arms = list(RUN_DIRS)

    plan = {
        "status": "EWOK_MARGIN_SEED_NOISE_PROBE_PLAN",
        "created_utc": now(),
        "decision_role": "Test whether compact-minus-repeat continuous EWoK margin deltas are shared across full-DeBERTa seeds beneath binary threshold churn; if not, strengthen the basin-specific aggregate-reallocation interpretation.",
        "lowest_cost_scope": "EWoK only, sampled rows_per_domain before any BLiMP/COMPS expansion; waits for the stable-ladder result file before using GPU if requested.",
        "out_dir": str(out_dir),
        "checkpoints": checkpoints,
        "rows_per_domain": args.rows_per_domain,
        "arms": {a: {"run_dir": str(RUN_DIRS[a]), "meta": ARM_META[a], "checkpoint_exists": {ck: checkpoint_exists(a, ck) for ck in checkpoints}, "prediction_paths": {ck: prediction_path_for(a, ck) for ck in checkpoints}} for a in arms},
        "wait_for_file": args.wait_for_file,
        "no_training_superglue_aoa_upload_or_leaderboard": True,
    }
    (out_dir / "ewok_margin_seed_noise_probe_plan.json").write_text(json.dumps(plan, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": plan["status"], "out_dir": str(out_dir), "checkpoints": checkpoints, "rows_per_domain": args.rows_per_domain, "plan_json": str(out_dir / "ewok_margin_seed_noise_probe_plan.json")}, indent=2), flush=True)
    if args.plan_only:
        return

    if args.wait_for_file:
        wr = wait_for_file(pathlib.Path(args.wait_for_file) if pathlib.Path(args.wait_for_file).is_absolute() else USER_ROOT / args.wait_for_file, args.wait_timeout_sec, args.poll_sec, log_path)
        (out_dir / "wait_for_file_record.json").write_text(json.dumps(wr, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        if wr.get("status") != "file_ready":
            raise SystemExit(2)
    if args.wait_for_checkpoints:
        wr = wait_for_checkpoints(arms, checkpoints, args.wait_timeout_sec, args.poll_sec, log_path)
        (out_dir / "wait_for_checkpoints_record.json").write_text(json.dumps(wr, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        if wr.get("status") != "checkpoints_ready":
            raise SystemExit(2)

    mod = load_step080()
    all_rows = mod.load_ewok_rows(mod.EVAL_EWOK)
    domains = set(args.domains) if args.domains else None
    sample_rows = mod.choose_rows(all_rows, args.rows_per_domain, args.seed, domains, args.hard_domains)
    if not sample_rows:
        raise RuntimeError("No EWoK rows selected")

    class ScoreArgs:
        pass
    score_args = ScoreArgs()
    score_args.device = args.device
    score_args.bf16 = args.bf16
    score_args.max_length = args.max_length
    score_args.row_batch_size = args.row_batch_size
    score_args.non_causal_batch_size = args.non_causal_batch_size

    model_summaries: dict[str, dict[str, Any]] = {}
    for arm in arms:
        for ck in checkpoints:
            if not checkpoint_exists(arm, ck):
                raise FileNotFoundError(checkpoint_path(arm, ck))
            model_name = f"{arm}_{ck}"
            cfg = {
                "ckpt": str(checkpoint_path(arm, ck).relative_to(USER_ROOT)),
                "pred": prediction_path_for(arm, ck),
                "broad": {},
                "note": f"full-DeBERTa seed-noise EWoK margin probe: {arm} {ck}",
            }
            print(json.dumps({"event": "model_start", "model": model_name, "checkpoint": cfg["ckpt"], "pred": cfg["pred"], "sample_rows": len(sample_rows)}, ensure_ascii=False), flush=True)
            summ = mod.score_model(model_name, cfg, sample_rows, score_args)
            summ["arm"] = arm
            summ["checkpoint_name"] = ck
            summ["seed"] = ARM_META[arm]["seed"]
            summ["data_arm"] = ARM_META[arm]["data_arm"]
            model_summaries[model_name] = summ
            (out_dir / f"{model_name}.json").write_text(json.dumps({k: v for k, v in summ.items() if k != "records"}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            vr = summ.get("validation_against_saved_predictions") or {}
            print(json.dumps({"event": "model_done", "model": model_name, "validation_match_rate": vr.get("match_rate"), "overall": summ.get("overall")}, ensure_ascii=False), flush=True)

    summary = write_comparison(out_dir, model_summaries, checkpoints)
    print(json.dumps({"status": summary["status"], "summary_json": str(out_dir / "ewok_margin_seed_noise_probe_summary.json"), "comparisons": summary["comparisons"]}, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
