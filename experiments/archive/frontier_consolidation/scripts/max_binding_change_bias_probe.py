#!/usr/bin/env python3
"""research: MAX-dose counterbalanced binding affected/unaffected probe.

CPU-only inference on the entity-event-state binding substrate.  This
reuses the research counterbalanced probe idea, but applies it to the current
MAX view/repeat arms in both DeBERTa basins and reads affected and unaffected
rows separately rather than hiding the signed movement in EEBF.

Scientific role: test whether the amplified Entity carrier could be a simple
shift toward predicting changed state.  A pure changed-state bias predicts
view-minus-repeat affected gains together with unaffected losses and little
balanced gain.  A record-addressability interpretation is strengthened only if
affected gains survive without a comparable unaffected cost.

No training, no official scoring job, no GPU, no upload, no leaderboard action.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import csv
import gc
import json
import os
import pathlib
import time
from collections import defaultdict
from typing import Any

# Force CPU before importing torch/transformers.
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

ROOT = _public_path('.')
WS = _public_path('experiments/archive/frontier_consolidation')
EVAL_PATH = _public_path('experiments/archive/representation_and_objectives/data/annotated_corpus/eval.jsonl')
OUT_DIR = _public_path('experiments/archive/frontier_consolidation/data/max_binding_change_bias_probe')
CHECKPOINTS = [f"chck_{i}M" for i in range(10, 101, 10)]

ARMS = {
    "basin1_max_view": {
        "basin": "seed43022_first_basin",
        "data_arm": "view",
        "run_dir": _public_path('experiments/archive/frontier_consolidation/training/runs/full_p2c_c2p_abs_view_dose2p64x_matched_rowholdout_deberta100M_seed43022'),
        "seed": 43022,
    },
    "basin1_max_repeat": {
        "basin": "seed43022_first_basin",
        "data_arm": "repeat",
        "run_dir": _public_path('experiments/archive/frontier_consolidation/training/runs/full_p2c_c2p_abs_repeat_dose2p64x_matched_rowholdout_deberta100M_seed43022'),
        "seed": 43022,
    },
    "basin2_max_view": {
        "basin": "seed43122_second_basin",
        "data_arm": "view",
        "run_dir": _public_path('experiments/archive/frontier_consolidation/training/runs/full_p2c_c2p_abs_view_dose2p64x_matched_rowholdout_deberta100M_seed43122'),
        "seed": 43122,
    },
    "basin2_max_repeat": {
        "basin": "seed43122_second_basin",
        "data_arm": "repeat",
        "run_dir": _public_path('experiments/archive/frontier_consolidation/training/runs/full_p2c_c2p_abs_repeat_dose2p64x_matched_rowholdout_deberta100M_seed43122'),
        "seed": 43122,
    },
}


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def rel(p: pathlib.Path) -> str:
    try:
        return str(p.relative_to(ROOT))
    except Exception:
        return str(p)


def load_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def first_answer_token(tok: Any, word: str) -> int:
    ids = tok.encode(" " + str(word), add_special_tokens=False)
    if not ids:
        raise ValueError(f"empty encoding for {word!r}")
    return int(ids[1] if len(ids) > 1 else ids[0])


def evaluate_one(model_key: str, model_path: pathlib.Path, eval_rows: list[dict[str, Any]], out_dir: pathlib.Path, batch_size: int) -> tuple[list[dict[str, Any]], float]:
    if torch.cuda.is_available() or torch.cuda.device_count() != 0:
        raise RuntimeError("CPU safety failure: torch can see CUDA")
    cache_home = out_dir / "hf_cache" / model_key
    os.environ["HF_HOME"] = str(cache_home)
    os.environ["TRANSFORMERS_CACHE"] = str(cache_home / "transformers")
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)
    model = AutoModelForMaskedLM.from_pretrained(model_path, trust_remote_code=True, local_files_only=True).to("cpu").eval()
    mask_id = tok.mask_token_id
    work_rows = []
    for r in eval_rows:
        if r.get("kind") != "binding" or r.get("split") not in {"eval_held_recomb", "train_binding", "eval_held_order"}:
            continue
        rr = dict(r)
        rr["_answer_id"] = first_answer_token(tok, rr["answer"])
        rr["_foil_id"] = first_answer_token(tok, rr["foil"])
        work_rows.append(rr)
    results: list[dict[str, Any]] = []
    start = time.time()
    for i in range(0, len(work_rows), batch_size):
        batch = work_rows[i:i + batch_size]
        enc = tok([r["text"] for r in batch], padding=True, truncation=True, max_length=128, return_tensors="pt")
        with torch.no_grad():
            logits = model(**enc).logits
        ids_batch = enc["input_ids"]
        for j, r in enumerate(batch):
            ids = ids_batch[j]
            mask_positions = (ids == mask_id).nonzero(as_tuple=True)[0]
            if len(mask_positions) == 0:
                margin = float("nan")
                correct = False
                err = "no_mask"
            else:
                pos = int(mask_positions[0].item())
                margin = float(logits[j, pos, int(r["_answer_id"])].item() - logits[j, pos, int(r["_foil_id"])].item())
                correct = bool(margin > 0)
                err = ""
            results.append({
                "model_key": model_key,
                "id": r.get("id", ""),
                "split": r.get("split", ""),
                "kind": r.get("kind", ""),
                "family": r.get("family", ""),
                "template_id": r.get("template_id", ""),
                "is_affected": bool(r.get("is_affected_query")),
                "correct": correct,
                "margin": margin,
                "answer": r.get("answer"),
                "foil": r.get("foil"),
                "error": err,
            })
    elapsed = time.time() - start
    del model
    del tok
    gc.collect()
    return results, elapsed


def acc(rows: list[dict[str, Any]]) -> float | None:
    if not rows:
        return None
    return sum(1 for r in rows if r.get("correct")) / len(rows)


def mean_margin(rows: list[dict[str, Any]]) -> float | None:
    vals = [float(r["margin"]) for r in rows if r.get("margin") == r.get("margin")]
    if not vals:
        return None
    return sum(vals) / len(vals)


def summarize_model(rows: list[dict[str, Any]]) -> dict[str, Any]:
    held = [r for r in rows if r["split"] == "eval_held_recomb" and r["kind"] == "binding"]
    affected = [r for r in held if r["is_affected"]]
    unaffected = [r for r in held if not r["is_affected"]]
    train = [r for r in rows if r["split"] == "train_binding"]
    order = [r for r in rows if r["split"] == "eval_held_order"]
    fam: dict[str, dict[str, Any]] = {}
    for f in sorted({r["family"] for r in held}):
        fr = [r for r in held if r["family"] == f]
        fa = [r for r in fr if r["is_affected"]]
        fu = [r for r in fr if not r["is_affected"]]
        fam[f] = {
            "affected_n": len(fa),
            "affected_accuracy": acc(fa),
            "affected_margin_mean": mean_margin(fa),
            "unaffected_n": len(fu),
            "unaffected_accuracy": acc(fu),
            "unaffected_margin_mean": mean_margin(fu),
        }
    aa = acc(affected)
    ua = acc(unaffected)
    am = mean_margin(affected)
    um = mean_margin(unaffected)
    return {
        "held_n": len(held),
        "affected_n": len(affected),
        "unaffected_n": len(unaffected),
        "affected_accuracy": aa,
        "unaffected_accuracy": ua,
        "EEBF_composite": None if aa is None or ua is None else (aa + ua) / 2.0,
        "affected_margin_mean": am,
        "unaffected_margin_mean": um,
        "margin_signed_shift_toward_changed": None if am is None or um is None else am - um,
        "train_binding_accuracy": acc(train),
        "eval_held_order_accuracy": acc(order),
        "by_family": fam,
    }


def delta(a: float | None, b: float | None) -> float | None:
    return None if a is None or b is None else a - b


def build_pair_deltas(summary_rows: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for basin in ["seed43022_first_basin", "seed43122_second_basin"]:
        for ck in CHECKPOINTS:
            view_key = f"{basin}:view:{ck}"
            repeat_key = f"{basin}:repeat:{ck}"
            v = summary_rows.get(view_key)
            r = summary_rows.get(repeat_key)
            if not v or not r or v.get("status") != "ok" or r.get("status") != "ok":
                out.append({"basin": basin, "checkpoint": ck, "status": "missing_pair"})
                continue
            vm = v["metrics"]
            rm = r["metrics"]
            affected_delta = delta(vm["affected_accuracy"], rm["affected_accuracy"])
            unaffected_delta = delta(vm["unaffected_accuracy"], rm["unaffected_accuracy"])
            eebf_delta = delta(vm["EEBF_composite"], rm["EEBF_composite"])
            affected_margin_delta = delta(vm["affected_margin_mean"], rm["affected_margin_mean"])
            unaffected_margin_delta = delta(vm["unaffected_margin_mean"], rm["unaffected_margin_mean"])
            signed_bias_margin_delta = delta(affected_margin_delta, unaffected_margin_delta)
            out.append({
                "basin": basin,
                "checkpoint": ck,
                "words": int(ck.replace("chck_", "").replace("M", "")) * 1_000_000,
                "status": "ok",
                "view_affected_accuracy": vm["affected_accuracy"],
                "repeat_affected_accuracy": rm["affected_accuracy"],
                "affected_delta_view_minus_repeat": affected_delta,
                "view_unaffected_accuracy": vm["unaffected_accuracy"],
                "repeat_unaffected_accuracy": rm["unaffected_accuracy"],
                "unaffected_delta_view_minus_repeat": unaffected_delta,
                "EEBF_delta_view_minus_repeat": eebf_delta,
                "affected_margin_delta_view_minus_repeat": affected_margin_delta,
                "unaffected_margin_delta_view_minus_repeat": unaffected_margin_delta,
                "signed_bias_margin_delta": signed_bias_margin_delta,
                "change_bias_signature": bool((affected_delta or 0.0) > 0 and (unaffected_delta or 0.0) < 0 and abs(eebf_delta or 0.0) < abs(affected_delta or 0.0)),
            })
    return out


def write_csv(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields: list[str] = []
    for row in rows:
        for k in row.keys():
            if k not in fields:
                fields.append(k)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def write_md(payload: dict[str, Any], path: pathlib.Path) -> None:
    pair_rows = [r for r in payload["pair_deltas"] if r.get("status") == "ok"]
    lines: list[str] = []
    lines.append("# research MAX binding changed-state-bias probe\n\n")
    lines.append("CPU-only research-style counterbalanced binding readout on MAX view/repeat in both DeBERTa basins. The key threat model is a signed shift toward predicting changed state: affected rows improve while unaffected rows decline, leaving EEBF nearly unchanged.\n\n")
    lines.append("## View-minus-repeat held recombination deltas\n\n")
    lines.append("| basin | checkpoint | affected Δ pp | unaffected Δ pp | EEBF Δ pp | affected margin Δ | unaffected margin Δ | signed margin shift | bias-like |\n")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|\n")
    for r in pair_rows:
        lines.append(
            f"| {r['basin']} | {r['checkpoint']} | "
            f"{100*r['affected_delta_view_minus_repeat']:+.3f} | "
            f"{100*r['unaffected_delta_view_minus_repeat']:+.3f} | "
            f"{100*r['EEBF_delta_view_minus_repeat']:+.3f} | "
            f"{r['affected_margin_delta_view_minus_repeat']:+.4f} | "
            f"{r['unaffected_margin_delta_view_minus_repeat']:+.4f} | "
            f"{r['signed_bias_margin_delta']:+.4f} | {r['change_bias_signature']} |\n"
        )
    lines.append("\n## Compact interpretation\n\n")
    interp = payload.get("interpretation", {})
    for k, v in interp.items():
        lines.append(f"- **{k}**: {v}\n")
    lines.append("\nFiles: summary JSON and CSVs in this directory.\n")
    path.write_text("".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    ap.add_argument("--checkpoints", nargs="*", default=CHECKPOINTS)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--plan-only", action="store_true")
    args = ap.parse_args()

    out_dir = pathlib.Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoints = args.checkpoints or CHECKPOINTS

    plan_models: list[dict[str, Any]] = []
    for arm_key, cfg in ARMS.items():
        for ck in checkpoints:
            model_path = pathlib.Path(cfg["run_dir"]) / "hf_model" / ck
            plan_models.append({
                "arm_key": arm_key,
                "basin": cfg["basin"],
                "data_arm": cfg["data_arm"],
                "checkpoint": ck,
                "model_path": rel(model_path),
                "exists": model_path.exists(),
            })
    plan = {
        "status": "MAX_BINDING_CHANGE_BIAS_PLAN" if args.plan_only else "MAX_BINDING_CHANGE_BIAS_START",
        "created_utc": now(),
        "eval_path": rel(EVAL_PATH),
        "torch_cuda_available": bool(torch.cuda.is_available()),
        "torch_cuda_device_count": int(torch.cuda.device_count()),
        "model_count": len(plan_models),
        "missing_models": [m for m in plan_models if not m["exists"]],
        "models": plan_models,
        "scientific_role": "Cheap CPU inference separates affected versus unaffected movement to test changed-state bias before any permuted-companion H100 run.",
        "no_training_official_eval_gpu_upload_or_leaderboard": True,
    }
    (out_dir / "binding_change_bias_plan.json").write_text(json.dumps(plan, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(plan, indent=2, ensure_ascii=False), flush=True)
    if args.plan_only:
        return
    if plan["missing_models"]:
        raise SystemExit("missing model paths; refusing partial probe unless plan is repaired")
    if torch.cuda.is_available() or torch.cuda.device_count() != 0:
        raise SystemExit("CPU safety failure before inference")

    eval_rows = load_jsonl(EVAL_PATH)
    per_item_all: list[dict[str, Any]] = []
    summary_rows: dict[str, dict[str, Any]] = {}
    per_model_records: list[dict[str, Any]] = []
    for arm_key, cfg in ARMS.items():
        for ck in checkpoints:
            basin = str(cfg["basin"])
            data_arm = str(cfg["data_arm"])
            model_key = f"{basin}:{data_arm}:{ck}"
            model_path = pathlib.Path(cfg["run_dir"]) / "hf_model" / ck
            rows, elapsed = evaluate_one(model_key, model_path, eval_rows, out_dir, args.batch_size)
            metrics = summarize_model(rows)
            rec = {
                "status": "ok",
                "model_key": model_key,
                "arm_key": arm_key,
                "basin": basin,
                "data_arm": data_arm,
                "seed": cfg["seed"],
                "checkpoint": ck,
                "words": int(ck.replace("chck_", "").replace("M", "")) * 1_000_000,
                "model_path": rel(model_path),
                "elapsed_sec": round(elapsed, 3),
                "metrics": metrics,
            }
            summary_rows[model_key] = rec
            per_model_records.append({
                "model_key": model_key,
                "basin": basin,
                "data_arm": data_arm,
                "checkpoint": ck,
                "affected_accuracy": metrics["affected_accuracy"],
                "unaffected_accuracy": metrics["unaffected_accuracy"],
                "EEBF_composite": metrics["EEBF_composite"],
                "affected_margin_mean": metrics["affected_margin_mean"],
                "unaffected_margin_mean": metrics["unaffected_margin_mean"],
                "margin_signed_shift_toward_changed": metrics["margin_signed_shift_toward_changed"],
                "train_binding_accuracy": metrics["train_binding_accuracy"],
                "eval_held_order_accuracy": metrics["eval_held_order_accuracy"],
                "elapsed_sec": round(elapsed, 3),
            })
            for row in rows:
                row.update({"basin": basin, "data_arm": data_arm, "checkpoint": ck})
            per_item_all.extend(rows)
            print(json.dumps({"model_key": model_key, "affected_acc": metrics["affected_accuracy"], "unaffected_acc": metrics["unaffected_accuracy"], "EEBF": metrics["EEBF_composite"], "elapsed_sec": round(elapsed, 3)}, ensure_ascii=False), flush=True)

    pair_deltas = build_pair_deltas(summary_rows)
    ok_pairs = [r for r in pair_deltas if r.get("status") == "ok"]
    bias_like_count = sum(1 for r in ok_pairs if r.get("change_bias_signature"))
    late = [r for r in ok_pairs if r.get("checkpoint") in {"chck_80M", "chck_90M", "chck_100M"}]
    late_bias_like_count = sum(1 for r in late if r.get("change_bias_signature"))
    mean_late_eebf_delta = sum(r["EEBF_delta_view_minus_repeat"] for r in late) / len(late) if late else None
    mean_late_affected_delta = sum(r["affected_delta_view_minus_repeat"] for r in late) / len(late) if late else None
    mean_late_unaffected_delta = sum(r["unaffected_delta_view_minus_repeat"] for r in late) / len(late) if late else None
    interpretation = {
        "late_pair_summary": f"Across late 80/90/100M pairs, mean affected Δ={mean_late_affected_delta}, mean unaffected Δ={mean_late_unaffected_delta}, mean EEBF Δ={mean_late_eebf_delta}.",
        "bias_like_pair_count": f"{bias_like_count}/{len(ok_pairs)} all checkpoint pairs and {late_bias_like_count}/{len(late)} late pairs have affected gain, unaffected loss, and smaller balanced movement.",
        "use_before_permuted_training": "If second-basin and numops splits show the same signed bias, do not treat Entity V-R as record-addressability evidence; if affected gains persist without an unaffected cost, the aligned-versus-permuted arm becomes more informative.",
    }
    payload = {
        "status": "MAX_BINDING_CHANGE_BIAS_DONE",
        "created_utc": plan["created_utc"],
        "finished_utc": now(),
        "eval_path": rel(EVAL_PATH),
        "cpu_safety": {"cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "torch_cuda_available": bool(torch.cuda.is_available()), "torch_cuda_device_count": int(torch.cuda.device_count())},
        "summary_rows": summary_rows,
        "per_model_records": per_model_records,
        "pair_deltas": pair_deltas,
        "interpretation": interpretation,
        "no_training_official_eval_gpu_upload_or_leaderboard": True,
    }
    write_csv(out_dir / "binding_per_model_summary.csv", per_model_records)
    write_csv(out_dir / "binding_view_minus_repeat_deltas.csv", pair_deltas)
    write_csv(out_dir / "binding_per_item_records.csv", per_item_all)
    (out_dir / "binding_change_bias_summary.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_md(payload, out_dir / "binding_change_bias_summary.md")
    print(json.dumps({
        "status": payload["status"],
        "out_dir": rel(out_dir),
        "ok_pair_count": len(ok_pairs),
        "bias_like_pair_count": bias_like_count,
        "late_bias_like_count": late_bias_like_count,
        "mean_late_affected_delta": mean_late_affected_delta,
        "mean_late_unaffected_delta": mean_late_unaffected_delta,
        "mean_late_eebf_delta": mean_late_eebf_delta,
        "no_training_official_eval_gpu_upload_or_leaderboard": True,
    }, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
