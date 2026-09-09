#!/usr/bin/env python3
"""research: DeBERTa MAX view/repeat/breadth binding affected/unaffected probe.

Scientific purpose
------------------
Use an already-built counterbalanced binding substrate to ask
whether the first-basin Entity view-minus-breadth (V-B) signal behaves like a
source-correspondence/state-update effect or like the view-minus-repeat (V-R)
and breadth-minus-repeat (B-R) changed-state offset.

This is intentionally cheap relative to new training:
  * no training
  * no official scorer
  * no GPU (CUDA is hidden before torch import)
  * no upload / leaderboard action
  * only late 80/90/100M checkpoints of already-trained DeBERTa arms

Rows are split into affected queries (the queried entity was changed by the
event/update) and unaffected queries (the queried entity should retain a stable
state despite another entity changing). A bias toward the changed value predicts
positive affected deltas and negative unaffected deltas, with little balanced
EEBF movement. A stronger correspondence/addressing story should preserve or
improve affected rows without paying a comparable unaffected cost.
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
from typing import Any

# CPU safety before importing torch/transformers.
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer


def find_user_root() -> pathlib.Path:
    return _PUBLIC_ROOT


ROOT = find_user_root()
A01_WS = ROOT / "experiments/archive" / 'representation_and_objectives'
A02_WS = ROOT / "experiments/archive" / 'frontier_consolidation'
EVAL_PATH = A01_WS / "data" / "annotated_corpus" / "eval.jsonl"
DEFAULT_OUT = A01_WS / "data" / "breadth_binding_affunaff_probe"
CHECKPOINTS = ["chck_80M", "chck_90M", "chck_100M"]

ARMS: dict[str, dict[str, Any]] = {
    "view": {
        "arm_role": "V",
        "run_dir": A02_WS / "training/runs/full_p2c_c2p_abs_view_dose2p64x_matched_rowholdout_deberta100M_seed43022",
        "description": "first-basin MAX compact source+rewrite view",
    },
    "repeat": {
        "arm_role": "R",
        "run_dir": A02_WS / "training/runs/full_p2c_c2p_abs_repeat_dose2p64x_matched_rowholdout_deberta100M_seed43022",
        "description": "first-basin MAX exact source recurrence/repeat",
    },
    "breadth": {
        "arm_role": "B",
        "run_dir": A02_WS / "training/runs/full_p2c_c2p_abs_breadth_dose2p64x_matched_rowholdout_deberta100M_seed43022",
        "description": "first-basin MAX same-population sentence breadth replacing compact companions",
    },
}

CONTRASTS = [
    ("VminusR", "view", "repeat"),
    ("VminusB", "view", "breadth"),
    ("BminusR", "breadth", "repeat"),
]


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def rel(p: pathlib.Path | str) -> str:
    pp = pathlib.Path(p)
    try:
        return str(pp.relative_to(ROOT))
    except Exception:
        return str(pp)


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
    # Preserve the research scoring convention for multi-piece words.
    return int(ids[1] if len(ids) > 1 else ids[0])


def eval_rows_for_binding(raw_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for r in raw_rows:
        if r.get("kind") != "binding":
            continue
        if r.get("split") not in {"eval_held_recomb", "train_binding", "eval_held_order"}:
            continue
        out.append(dict(r))
    return out


def evaluate_one(model_key: str, model_path: pathlib.Path, eval_rows: list[dict[str, Any]], out_dir: pathlib.Path, batch_size: int, max_len: int) -> tuple[list[dict[str, Any]], float]:
    if torch.cuda.is_available() or torch.cuda.device_count() != 0:
        raise RuntimeError("CPU safety failure: torch can see CUDA")
    cache_home = out_dir / "hf_cache" / model_key.replace(":", "_")
    os.environ["HF_HOME"] = str(cache_home)
    os.environ["TRANSFORMERS_CACHE"] = str(cache_home / "transformers")

    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)
    model = AutoModelForMaskedLM.from_pretrained(model_path, trust_remote_code=True, local_files_only=True).to("cpu").eval()
    mask_id = tok.mask_token_id
    work: list[dict[str, Any]] = []
    for r in eval_rows:
        rr = dict(r)
        rr["_answer_id"] = first_answer_token(tok, rr["answer"])
        rr["_foil_id"] = first_answer_token(tok, rr["foil"])
        work.append(rr)

    results: list[dict[str, Any]] = []
    start = time.time()
    for i in range(0, len(work), batch_size):
        batch = work[i:i + batch_size]
        enc = tok([r["text"] for r in batch], padding=True, truncation=True, max_length=max_len, return_tensors="pt")
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
    return None if not rows else sum(1 for r in rows if r.get("correct")) / len(rows)


def mean_margin(rows: list[dict[str, Any]]) -> float | None:
    vals = [float(r["margin"]) for r in rows if r.get("margin") == r.get("margin")]
    return None if not vals else sum(vals) / len(vals)


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    held = [r for r in rows if r["split"] == "eval_held_recomb" and r["kind"] == "binding"]
    affected = [r for r in held if r["is_affected"]]
    unaffected = [r for r in held if not r["is_affected"]]
    train = [r for r in rows if r["split"] == "train_binding"]
    order = [r for r in rows if r["split"] == "eval_held_order"]
    aa = acc(affected)
    ua = acc(unaffected)
    am = mean_margin(affected)
    um = mean_margin(unaffected)
    fam: dict[str, dict[str, Any]] = {}
    for f in sorted({r.get("family", "") for r in held}):
        fr = [r for r in held if r.get("family", "") == f]
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
    return {
        "held_n": len(held),
        "affected_n": len(affected),
        "unaffected_n": len(unaffected),
        "affected_accuracy": aa,
        "unaffected_accuracy": ua,
        "EEBF_composite": None if aa is None or ua is None else 0.5 * (aa + ua),
        "affected_margin_mean": am,
        "unaffected_margin_mean": um,
        "margin_signed_shift_toward_changed": None if am is None or um is None else am - um,
        "train_binding_accuracy": acc(train),
        "eval_held_order_accuracy": acc(order),
        "by_family": fam,
    }


def sub(a: float | None, b: float | None) -> float | None:
    return None if a is None or b is None else a - b


def build_contrasts(per_model: dict[tuple[str, str], dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for ck in CHECKPOINTS:
        for cname, akey, bkey in CONTRASTS:
            a = per_model.get((akey, ck))
            b = per_model.get((bkey, ck))
            if not a or not b:
                out.append({"checkpoint": ck, "contrast": cname, "status": "missing_pair"})
                continue
            am = a["metrics"]
            bm = b["metrics"]
            aff_d = sub(am["affected_accuracy"], bm["affected_accuracy"])
            unaff_d = sub(am["unaffected_accuracy"], bm["unaffected_accuracy"])
            eebf_d = sub(am["EEBF_composite"], bm["EEBF_composite"])
            aff_md = sub(am["affected_margin_mean"], bm["affected_margin_mean"])
            unaff_md = sub(am["unaffected_margin_mean"], bm["unaffected_margin_mean"])
            signed_md = sub(aff_md, unaff_md)
            out.append({
                "checkpoint": ck,
                "words": int(ck.replace("chck_", "").replace("M", "")) * 1_000_000,
                "contrast": cname,
                "arm_a": akey,
                "arm_b": bkey,
                "status": "ok",
                "A_affected_accuracy": am["affected_accuracy"],
                "B_affected_accuracy": bm["affected_accuracy"],
                "affected_delta": aff_d,
                "A_unaffected_accuracy": am["unaffected_accuracy"],
                "B_unaffected_accuracy": bm["unaffected_accuracy"],
                "unaffected_delta": unaff_d,
                "EEBF_delta": eebf_d,
                "affected_margin_delta": aff_md,
                "unaffected_margin_delta": unaff_md,
                "signed_bias_margin_delta": signed_md,
                "offset_like_signature": bool((aff_d or 0.0) > 0 and (unaff_d or 0.0) < 0 and abs(eebf_d or 0.0) < abs(aff_d or 0.0)),
            })
    return out


def mean(xs: list[float]) -> float | None:
    return None if not xs else sum(xs) / len(xs)


def build_window_summary(contrast_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for cname in [c[0] for c in CONTRASTS]:
        rows = [r for r in contrast_rows if r.get("status") == "ok" and r.get("contrast") == cname]
        for key in ["affected_delta", "unaffected_delta", "EEBF_delta", "affected_margin_delta", "unaffected_margin_delta", "signed_bias_margin_delta"]:
            vals = [float(r[key]) for r in rows if r.get(key) is not None]
            out.append({
                "contrast": cname,
                "window": "late_80_90_100M",
                "quantity": key,
                "n": len(vals),
                "mean": mean(vals),
                "min": min(vals) if vals else None,
                "max": max(vals) if vals else None,
            })
        offset_count = sum(1 for r in rows if r.get("offset_like_signature"))
        out.append({"contrast": cname, "window": "late_80_90_100M", "quantity": "offset_like_count", "n": len(rows), "mean": offset_count, "min": offset_count, "max": offset_count})
    return out


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


def fmt(x: Any, scale: float = 1.0) -> str:
    if x is None:
        return "NA"
    return f"{scale * float(x):+.3f}"


def write_md(payload: dict[str, Any], path: pathlib.Path) -> None:
    lines: list[str] = []
    lines.append("# research breadth binding affected/unaffected probe\n\n")
    lines.append("CPU-only inference on A01 research counterbalanced binding rows, applied to already-trained first-basin DeBERTa MAX view/repeat/breadth checkpoints. It asks whether V-B is a balanced correspondence-like gain or an affected/unaffected offset.\n\n")
    lines.append("## Per-arm held recombination readout\n\n")
    lines.append("| arm | checkpoint | affected acc | unaffected acc | EEBF | affected margin | unaffected margin | signed margin | train binding |\n")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|\n")
    for r in payload["per_model_records"]:
        lines.append(
            f"| {r['arm']} | {r['checkpoint']} | {fmt(r['affected_accuracy'],100)} | {fmt(r['unaffected_accuracy'],100)} | {fmt(r['EEBF_composite'],100)} | {fmt(r['affected_margin_mean'])} | {fmt(r['unaffected_margin_mean'])} | {fmt(r['margin_signed_shift_toward_changed'])} | {fmt(r['train_binding_accuracy'],100)} |\n"
        )
    lines.append("\n## Contrasts: affected/unaffected deltas\n\n")
    lines.append("| contrast | checkpoint | affected Δ pp | unaffected Δ pp | EEBF Δ pp | affected margin Δ | unaffected margin Δ | signed margin Δ | offset-like |\n")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---|\n")
    for r in payload["contrast_rows"]:
        if r.get("status") != "ok":
            lines.append(f"| {r['contrast']} | {r['checkpoint']} | missing | missing | missing | missing | missing | missing | missing |\n")
            continue
        lines.append(
            f"| {r['contrast']} | {r['checkpoint']} | {fmt(r['affected_delta'],100)} | {fmt(r['unaffected_delta'],100)} | {fmt(r['EEBF_delta'],100)} | {fmt(r['affected_margin_delta'])} | {fmt(r['unaffected_margin_delta'])} | {fmt(r['signed_bias_margin_delta'])} | {r['offset_like_signature']} |\n"
        )
    lines.append("\n## Late-window means\n\n")
    lines.append("| contrast | quantity | n | mean | min | max |\n")
    lines.append("|---|---|---:|---:|---:|---:|\n")
    for r in payload["window_summary"]:
        scale = 100.0 if r["quantity"].endswith("_delta") and "margin" not in r["quantity"] else 1.0
        lines.append(f"| {r['contrast']} | {r['quantity']} | {r['n']} | {fmt(r['mean'], scale)} | {fmt(r['min'], scale)} | {fmt(r['max'], scale)} |\n")
    lines.append("\n## Interpretation\n\n")
    for k, v in payload["interpretation"].items():
        lines.append(f"- **{k}**: {v}\n")
    lines.append("\n## Files\n\n")
    for k, v in payload["files"].items():
        lines.append(f"- {k}: `{v}`\n")
    path.write_text("".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT))
    ap.add_argument("--checkpoints", nargs="*", default=CHECKPOINTS)
    ap.add_argument("--arms", nargs="*", default=list(ARMS.keys()), choices=sorted(ARMS))
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--max-len", type=int, default=128)
    ap.add_argument("--plan-only", action="store_true")
    ap.add_argument("--allow-missing", action="store_true")
    args = ap.parse_args()

    out_dir = pathlib.Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoints = args.checkpoints or CHECKPOINTS
    selected_arms = args.arms or list(ARMS.keys())

    model_plan: list[dict[str, Any]] = []
    for arm in selected_arms:
        cfg = ARMS[arm]
        for ck in checkpoints:
            p = pathlib.Path(cfg["run_dir"]) / "hf_model" / ck
            model_plan.append({
                "arm": arm,
                "arm_role": cfg["arm_role"],
                "checkpoint": ck,
                "model_path": rel(p),
                "exists": p.exists(),
                "description": cfg["description"],
            })
    plan = {
        "status": "BREADTH_BINDING_AFFUNAFF_PLAN" if args.plan_only else "BREADTH_BINDING_AFFUNAFF_START",
        "created_utc": now(),
        "eval_path": rel(EVAL_PATH),
        "out_dir": rel(out_dir),
        "checkpoints": checkpoints,
        "arms": selected_arms,
        "model_plan": model_plan,
        "missing_models": [r for r in model_plan if not r["exists"]],
        "cpu_safety": {"CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"), "torch_cuda_available": bool(torch.cuda.is_available()), "torch_cuda_device_count": int(torch.cuda.device_count())},
        "no_training_official_eval_gpu_upload_or_leaderboard": True,
    }
    (out_dir / "binding_probe_plan.json").write_text(json.dumps(plan, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(plan, indent=2, ensure_ascii=False), flush=True)
    if args.plan_only:
        return
    if plan["missing_models"] and not args.allow_missing:
        raise SystemExit(f"missing {len(plan['missing_models'])} model checkpoints; refusing partial run")
    if torch.cuda.is_available() or torch.cuda.device_count() != 0:
        raise SystemExit("CPU safety failure before inference")

    raw_rows = load_jsonl(EVAL_PATH)
    eval_rows = eval_rows_for_binding(raw_rows)
    per_item_all: list[dict[str, Any]] = []
    per_model_records: list[dict[str, Any]] = []
    per_model_full: dict[tuple[str, str], dict[str, Any]] = {}

    for arm in selected_arms:
        cfg = ARMS[arm]
        for ck in checkpoints:
            model_path = pathlib.Path(cfg["run_dir"]) / "hf_model" / ck
            if not model_path.exists():
                continue
            model_key = f"{arm}:{ck}"
            rows, elapsed = evaluate_one(model_key, model_path, eval_rows, out_dir, args.batch_size, args.max_len)
            metrics = summarize(rows)
            rec = {
                "status": "ok",
                "arm": arm,
                "arm_role": cfg["arm_role"],
                "checkpoint": ck,
                "words": int(ck.replace("chck_", "").replace("M", "")) * 1_000_000,
                "model_path": rel(model_path),
                "elapsed_sec": round(elapsed, 3),
                "metrics": metrics,
            }
            per_model_full[(arm, ck)] = rec
            per_model_records.append({
                "arm": arm,
                "arm_role": cfg["arm_role"],
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
            for rr in rows:
                rr.update({"arm": arm, "checkpoint": ck})
            per_item_all.extend(rows)
            print(json.dumps({
                "model_key": model_key,
                "affected_acc": metrics["affected_accuracy"],
                "unaffected_acc": metrics["unaffected_accuracy"],
                "EEBF": metrics["EEBF_composite"],
                "elapsed_sec": round(elapsed, 3),
            }, ensure_ascii=False), flush=True)

    contrast_rows = build_contrasts(per_model_full)
    window_summary = build_window_summary(contrast_rows)

    def wmean(c: str, q: str) -> float | None:
        vals = [r["mean"] for r in window_summary if r["contrast"] == c and r["quantity"] == q]
        return vals[0] if vals else None

    interpretation = {
        "VminusB_binding": "If V-B were the same changed-state offset as V-R/B-R, it should show affected gains paired with unaffected losses. The saved table should therefore be read through affected_delta, unaffected_delta, and EEBF_delta rather than aggregate Entity alone.",
        "late_mean_VminusB": f"late 80/90/100 V-B mean affected_delta={wmean('VminusB','affected_delta')}, unaffected_delta={wmean('VminusB','unaffected_delta')}, EEBF_delta={wmean('VminusB','EEBF_delta')}",
        "permuted_companion_decision": "A balanced positive V-B on binding rows would strengthen the case for a permuted companion arm; an offset-like V-B would weaken correspondence/addressability and push interpretation toward decision bias or dataset/evaluation mixture.",
        "scope": "This remains a cheap binding panel over a synthetic-natural bridge substrate, not official BabyLM evaluation and not a completed general principle.",
    }
    files = {
        "plan_json": rel(out_dir / "binding_probe_plan.json"),
        "summary_json": rel(out_dir / "binding_probe_summary.json"),
        "summary_md": rel(out_dir / "binding_probe_summary.md"),
        "per_model_csv": rel(out_dir / "binding_per_model_summary.csv"),
        "contrast_csv": rel(out_dir / "binding_contrast_deltas.csv"),
        "window_summary_csv": rel(out_dir / "binding_window_summary.csv"),
        "per_item_csv": rel(out_dir / "binding_per_item_records.csv"),
    }
    payload = {
        **plan,
        "status": "BREADTH_BINDING_AFFUNAFF_DONE",
        "finished_utc": now(),
        "eval_row_count": len(eval_rows),
        "per_model_records": per_model_records,
        "per_model_full": {f"{k[0]}:{k[1]}": v for k, v in per_model_full.items()},
        "contrast_rows": contrast_rows,
        "window_summary": window_summary,
        "interpretation": interpretation,
        "files": files,
    }
    write_csv(out_dir / "binding_per_model_summary.csv", per_model_records)
    write_csv(out_dir / "binding_contrast_deltas.csv", contrast_rows)
    write_csv(out_dir / "binding_window_summary.csv", window_summary)
    write_csv(out_dir / "binding_per_item_records.csv", per_item_all)
    (out_dir / "binding_probe_summary.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_md(payload, out_dir / "binding_probe_summary.md")
    print(json.dumps({
        "status": payload["status"],
        "summary_md": files["summary_md"],
        "late_VminusB_affected_delta": wmean("VminusB", "affected_delta"),
        "late_VminusB_unaffected_delta": wmean("VminusB", "unaffected_delta"),
        "late_VminusB_EEBF_delta": wmean("VminusB", "EEBF_delta"),
        "no_training_official_eval_gpu_upload_or_leaderboard": True,
    }, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
