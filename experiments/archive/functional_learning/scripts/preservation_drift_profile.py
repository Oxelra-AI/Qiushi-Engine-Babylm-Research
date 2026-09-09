#!/usr/bin/env python3
"""research: parent-function drift on the ordinary full-row preservation rendering.

The clean preservation trainer can apply KL(teacher || student) on standard 15%
WWM renderings of full Qwen pair rows with source evidence still present. Before
launching a full controlled run, this script measures whether existing endpoints
actually drift from coherent86 on that exact rendering in eval mode.

If the acquisition-only (M,S) endpoint has only tiny eval/eval KL from the parent
on these full rows, ordinary full-row distillation is unlikely to repair the
observed evidence-absent/view-only/CDI costs. If the KL is substantial and aligned
with worse label CE, the rendering is a plausible preservation target.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import csv
import json
import os
import pathlib
import sys
import time
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn.functional as F

ROOT = _public_path('.')
STUDY = _public_path('experiments/archive/functional_learning')
SCRIPTS = _public_path('experiments/archive/functional_learning/scripts')
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import real_stream_train_weighted as s64  # noqa: E402
import corrected_bridge_trainer as bridge  # noqa: E402
import targeted_preservation_train as s88  # noqa: E402

DEFAULT_OUT = _public_path('experiments/archive/functional_learning/data/preservation_drift_profile')
MODEL_SPECS: Dict[str, Optional[pathlib.Path]] = {
    "coherent86": None,
    "sparse_focus_seed62064": _public_path('experiments/archive/functional_learning/data/unchanged_focus_weighted_train/correspondence_focus_weighted/checkpoints/update_0080'),
    "dense_focus_seed62064": _public_path('experiments/archive/functional_learning/data/unchanged_dense_focus_train/correspondence_focus_weighted/checkpoints/update_0080'),
    "dense_focus_seed62065": _public_path('experiments/archive/functional_learning/data/dense_focus_rep_seed62065_train/correspondence_focus_weighted/checkpoints/update_0080'),
    "densemask_sparselabel_seed62064": _public_path('experiments/archive/functional_learning/data/densemask_sparselabel_train_seed62064/correspondence_focus_weighted/checkpoints/update_0080'),
    "pres_lambda1_trainmode_confounded": _public_path('experiments/archive/functional_learning/data/preservation_lambda1p0_seed62064/checkpoints/update_0080'),
}


def rel(p: pathlib.Path | str | None) -> str | None:
    if p is None:
        return None
    try:
        return str(pathlib.Path(p).resolve().relative_to(ROOT))
    except Exception:
        return str(p)


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def setup_cache(out_dir: pathlib.Path) -> None:
    hf = out_dir / "hf_cache"
    os.environ["HF_HOME"] = str(hf.resolve())
    os.environ["HF_HUB_CACHE"] = str((hf / "hub").resolve())
    os.environ["HF_DATASETS_CACHE"] = str((hf / "datasets").resolve())
    os.environ["TRANSFORMERS_CACHE"] = str((hf / "transformers").resolve())
    os.environ["HF_MODULES_CACHE"] = str((hf / "modules").resolve())
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    for sub in ["hub", "datasets", "transformers", "modules"]:
        (hf / sub).mkdir(parents=True, exist_ok=True)


def select_qwen_rows(rows: List[Dict[str, Any]], max_examples: int) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    picked = []
    stats = {"rows_seen": 0, "qwen_seen": 0, "words_seen_until_last": 0}
    words = 0
    for r in rows:
        stats["rows_seen"] += 1
        words += int(r.get("words", s64.wc(r.get("text", ""))))
        if r.get("source") == "qwen_pair_packed" and bool(r.get("qwen_pair_segments")):
            stats["qwen_seen"] += 1
            picked.append(r)
            stats["words_seen_until_last"] = words
            if len(picked) >= int(max_examples):
                break
    return picked, stats


def load_endpoint(model_path: Optional[pathlib.Path], device: torch.device, private_scale: float):
    endpoint = pathlib.Path(model_path) if model_path is not None else bridge.PARENT_PATH
    model, missing, unexpected = bridge.base_loader.load_model(endpoint, device, 128, float(private_scale))
    ident = bridge.model_identity(model)
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()
    return model, {"path": rel(endpoint), "identity": ident, "missing": len(missing), "unexpected": len(unexpected)}


def score_model(student, teacher, examples: List[Dict[str, Any]], device: torch.device,
                batch_size: int, T: float) -> Dict[str, Any]:
    n_batches = 0
    n_tokens = 0
    kl_t_s = 0.0
    kl_s_t = 0.0
    ce_student = 0.0
    ce_teacher = 0.0
    rank_student_sum = 0.0
    rank_teacher_sum = 0.0
    improved_ce = 0
    improved_rank = 0
    per_batch = []
    for bi, start in enumerate(range(0, len(examples), int(batch_size))):
        mb = examples[start:start + int(batch_size)]
        inp = torch.stack([x["input_ids"] for x in mb]).to(device)
        att = torch.stack([x["attention_mask"] for x in mb]).to(device)
        lab = torch.stack([x["labels"] for x in mb]).to(device)
        mask = lab != -100
        if not mask.any():
            continue
        with torch.no_grad():
            s_logits_full = student(input_ids=inp, attention_mask=att).logits
            t_logits_full = teacher(input_ids=inp, attention_mask=att).logits
            s_logits = s_logits_full[mask]
            t_logits = t_logits_full[mask]
            labels = lab[mask]
            s_lp = F.log_softmax(s_logits / T, dim=-1)
            t_lp = F.log_softmax(t_logits / T, dim=-1)
            s_p = torch.exp(s_lp)
            t_p = torch.exp(t_lp)
            kts = F.kl_div(s_lp, t_p, reduction="sum")
            kst = F.kl_div(t_lp, s_p, reduction="sum")
            if T != 1.0:
                kts = kts * (T * T)
                kst = kst * (T * T)
            V = s_logits.shape[-1]
            s_ce_each = F.cross_entropy(s_logits.reshape(-1, V), labels.reshape(-1), reduction="none")
            t_ce_each = F.cross_entropy(t_logits.reshape(-1, V), labels.reshape(-1), reduction="none")
            s_lab_logits = s_logits.gather(1, labels.view(-1, 1)).squeeze(1)
            t_lab_logits = t_logits.gather(1, labels.view(-1, 1)).squeeze(1)
            s_rank_each = (s_logits > s_lab_logits.view(-1, 1)).sum(dim=1).float() + 1.0
            t_rank_each = (t_logits > t_lab_logits.view(-1, 1)).sum(dim=1).float() + 1.0
            nt = int(labels.numel())
            n_tokens += nt
            n_batches += 1
            kl_t_s += float(kts.detach().cpu())
            kl_s_t += float(kst.detach().cpu())
            ce_student += float(s_ce_each.sum().detach().cpu())
            ce_teacher += float(t_ce_each.sum().detach().cpu())
            rank_student_sum += float(s_rank_each.sum().detach().cpu())
            rank_teacher_sum += float(t_rank_each.sum().detach().cpu())
            improved_ce += int((s_ce_each < t_ce_each).sum().detach().cpu())
            improved_rank += int((s_rank_each < t_rank_each).sum().detach().cpu())
            per_batch.append({
                "batch_index": bi,
                "tokens": nt,
                "kl_teacher_to_student_mean": float(kts.detach().cpu()) / max(1, nt),
                "student_ce_mean": float(s_ce_each.mean().detach().cpu()),
                "teacher_ce_mean": float(t_ce_each.mean().detach().cpu()),
                "student_rank_mean": float(s_rank_each.mean().detach().cpu()),
                "teacher_rank_mean": float(t_rank_each.mean().detach().cpu()),
            })
        del inp, att, lab, mask, s_logits_full, t_logits_full
    return {
        "batches": n_batches,
        "tokens": n_tokens,
        "kl_teacher_to_student_mean": float(kl_t_s / max(1, n_tokens)),
        "kl_student_to_teacher_mean": float(kl_s_t / max(1, n_tokens)),
        "student_ce_label_mean": float(ce_student / max(1, n_tokens)),
        "teacher_ce_label_mean": float(ce_teacher / max(1, n_tokens)),
        "delta_ce_student_minus_teacher": float((ce_student - ce_teacher) / max(1, n_tokens)),
        "student_rank_mean": float(rank_student_sum / max(1, n_tokens)),
        "teacher_rank_mean": float(rank_teacher_sum / max(1, n_tokens)),
        "delta_rank_student_minus_teacher": float((rank_student_sum - rank_teacher_sum) / max(1, n_tokens)),
        "ce_improved_fraction_vs_teacher": float(improved_ce / max(1, n_tokens)),
        "rank_improved_fraction_vs_teacher": float(improved_rank / max(1, n_tokens)),
        "per_batch_head": per_batch[:8],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tail-jsonl", type=pathlib.Path, default=s64.DEFAULT_TAIL)
    ap.add_argument("--out-dir", type=pathlib.Path, default=DEFAULT_OUT)
    ap.add_argument("--max-updates", type=int, default=80)
    ap.add_argument("--words-per-update", type=int, default=39533)
    ap.add_argument("--max-qwen-examples", type=int, default=384)
    ap.add_argument("--seq-length", type=int, default=512)
    ap.add_argument("--train-seed", type=int, default=62064)
    ap.add_argument("--mask-prob", type=float, default=0.15)
    ap.add_argument("--private-scale", type=float, default=0.75)
    ap.add_argument("--batch-size", type=int, default=12)
    ap.add_argument("--kl-temperature", type=float, default=1.0)
    ap.add_argument("--models", nargs="+", default=["coherent86", "sparse_focus_seed62064", "densemask_sparselabel_seed62064", "dense_focus_seed62064", "pres_lambda1_trainmode_confounded"], choices=list(MODEL_SPECS.keys()))
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu", "auto"])
    ap.add_argument("--gpu", type=int, default=0)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    setup_cache(args.out_dir)
    torch.set_num_threads(max(1, min(8, int(os.environ.get("QIUSHI_TORCH_THREADS", "8")))))
    if args.device == "cuda" and torch.cuda.is_available():
        device = torch.device(f"cuda:{int(args.gpu)}")
    elif args.device == "auto" and torch.cuda.is_available():
        device = torch.device(f"cuda:{int(args.gpu)}")
    else:
        device = torch.device("cpu")

    rows, prefix_info = s64.load_prefix(args.tail_jsonl, int(args.max_updates), int(args.words_per_update))
    qwen_rows, qwen_stats = select_qwen_rows(rows, int(args.max_qwen_examples))
    tokenizer = bridge.AutoTokenizer.from_pretrained(str(bridge.PARENT_PATH), local_files_only=True, use_fast=True)
    wgb = bridge.WordGroupBuilder(tokenizer)
    pres_examples, pres_n, pres_w = s88.prepare_preservation(qwen_rows, tokenizer, int(args.seq_length), wgb, int(args.train_seed), float(args.mask_prob))
    example_digest = {
        "qwen_rows_selected": len(qwen_rows),
        "pres_examples": len(pres_examples),
        "pres_targets": pres_n,
        "pres_words": pres_w,
        "rendering": "ordinary 15% WWM on full packed Qwen pair rows; source evidence remains present",
    }

    teacher, teacher_load = load_endpoint(None, device, float(args.private_scale))
    results: Dict[str, Any] = {}
    load_infos: Dict[str, Any] = {"coherent86_teacher": teacher_load}
    for name in args.models:
        t0 = time.time()
        student, load_info = load_endpoint(MODEL_SPECS[name], device, float(args.private_scale))
        metrics = score_model(student, teacher, pres_examples, device, int(args.batch_size), float(args.kl_temperature))
        metrics["elapsed_sec"] = round(time.time() - t0, 1)
        metrics["model_path"] = rel(MODEL_SPECS[name] if MODEL_SPECS[name] is not None else bridge.PARENT_PATH)
        results[name] = metrics
        load_infos[name] = load_info
        print(json.dumps({"event": "model_done", "model": name, "kl_teacher_to_student": metrics["kl_teacher_to_student_mean"], "delta_ce": metrics["delta_ce_student_minus_teacher"], "delta_rank": metrics["delta_rank_student_minus_teacher"], "elapsed_sec": metrics["elapsed_sec"]}, ensure_ascii=False), flush=True)
        del student
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Deltas against sparse/acquisition baselines if present.
    comparison: Dict[str, Any] = {}
    if "densemask_sparselabel_seed62064" in results and "sparse_focus_seed62064" in results:
        dm = results["densemask_sparselabel_seed62064"]
        sp = results["sparse_focus_seed62064"]
        comparison["densemask_minus_sparse"] = {
            "kl_teacher_to_student_mean_delta": dm["kl_teacher_to_student_mean"] - sp["kl_teacher_to_student_mean"],
            "delta_ce_student_minus_teacher_delta": dm["delta_ce_student_minus_teacher"] - sp["delta_ce_student_minus_teacher"],
            "delta_rank_student_minus_teacher_delta": dm["delta_rank_student_minus_teacher"] - sp["delta_rank_student_minus_teacher"],
        }
    if "pres_lambda1_trainmode_confounded" in results and "densemask_sparselabel_seed62064" in results:
        pr = results["pres_lambda1_trainmode_confounded"]
        dm = results["densemask_sparselabel_seed62064"]
        comparison["lambda1_minus_densemask"] = {
            "kl_teacher_to_student_mean_delta": pr["kl_teacher_to_student_mean"] - dm["kl_teacher_to_student_mean"],
            "delta_ce_student_minus_teacher_delta": pr["delta_ce_student_minus_teacher"] - dm["delta_ce_student_minus_teacher"],
            "delta_rank_student_minus_teacher_delta": pr["delta_rank_student_minus_teacher"] - dm["delta_rank_student_minus_teacher"],
        }

    report = {
        "status": "PRESERVATION_DRIFT_PROFILE_DONE",
        "created_utc": now(),
        "scientific_question": "Does the ordinary full-row WWM preservation rendering expose meaningful parent-function drift after acquisition-only training?",
        "prefix_info": prefix_info,
        "qwen_selection": qwen_stats,
        "preservation_examples": example_digest,
        "implemented_kl_direction": "KL(teacher || student)",
        "device": str(device),
        "teacher": teacher_load,
        "load_infos": load_infos,
        "results": results,
        "comparisons": comparison,
        "interpretation_note": "This probes only ordinary full-row Qwen renderings where source evidence remains present; it does not measure CDI or genuinely evidence-absent view-only damage directly.",
    }
    out_json = args.out_dir / "preservation_drift_profile.json"
    out_md = args.out_dir / "preservation_drift_profile.md"
    out_csv = args.out_dir / "preservation_drift_profile.csv"
    out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        cols = ["model", "kl_teacher_to_student_mean", "kl_student_to_teacher_mean", "delta_ce_student_minus_teacher", "delta_rank_student_minus_teacher", "ce_improved_fraction_vs_teacher", "rank_improved_fraction_vs_teacher", "tokens", "elapsed_sec"]
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for name, m in results.items():
            w.writerow({k: (name if k == "model" else m.get(k)) for k in cols})
    lines = [
        "# research preservation drift profile",
        "",
        f"Status: `{report['status']}`",
        f"Rendering: {example_digest['rendering']}",
        f"Examples/targets/words: `{example_digest['pres_examples']}` / `{example_digest['pres_targets']}` / `{example_digest['pres_words']}`",
        "",
        "## Model results",
    ]
    for name, m in results.items():
        lines.append(f"- {name}: KL(t||s) `{m['kl_teacher_to_student_mean']:.8f}`, ΔCE(s-t) `{m['delta_ce_student_minus_teacher']:.6f}`, Δrank(s-t) `{m['delta_rank_student_minus_teacher']:.3f}`, CE-improved `{m['ce_improved_fraction_vs_teacher']:.3f}`, rank-improved `{m['rank_improved_fraction_vs_teacher']:.3f}`")
    if comparison:
        lines.append("\n## Comparisons")
        for name, c in comparison.items():
            lines.append(f"- {name}: {json.dumps(c, ensure_ascii=False)}")
    lines.append("\nThis is a preservation-target relevance probe, not official BabyLM scoring.")
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "out_json": rel(out_json), "out_md": rel(out_md), "out_csv": rel(out_csv), "results": {k: {"kl_teacher_to_student_mean": v["kl_teacher_to_student_mean"], "delta_ce": v["delta_ce_student_minus_teacher"], "delta_rank": v["delta_rank_student_minus_teacher"]} for k, v in results.items()}, "comparisons": comparison}, indent=2), flush=True)

    del teacher
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
