#!/usr/bin/env python3
"""research: measure identical-weight preservation KL under train/eval modes.

The research preservation branch reports KL(teacher || student) on standard-WWM
renderings of Qwen rows. At update 0, teacher and student are loaded from the
same coherent86 checkpoint, but the teacher is in eval mode while the student is
in train mode. This script measures how much nonzero KL exists before learning
purely from stochastic train-mode behavior (dropout), and records the exact KL
orientation implemented by F.kl_div(log_softmax(student), softmax(teacher)).

It does not train. It runs on a bounded number of preservation examples from the
first macro update by default.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import json
import os
import pathlib
import sys
import time
from typing import Any, Dict, List, Tuple

import torch
import torch.nn.functional as F

ROOT = _public_path('.')
STUDY = _public_path('experiments/archive/functional_learning')
SCRIPTS = _public_path('experiments/archive/functional_learning/scripts')
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import real_stream_train_weighted as s64  # noqa: E402
import corrected_bridge_trainer as bridge  # noqa: E402
# Reuse preparation function from research; import applies research patch but this script only uses preservation prep.
import targeted_preservation_train as s88  # noqa: E402

DEFAULT_OUT = _public_path('experiments/archive/functional_learning/data/preservation_kl_mode_probe')


def rel(p: pathlib.Path | str) -> str:
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


def first_macro_rows(rows: List[Dict[str, Any]], words_per_update: int) -> Tuple[List[Dict[str, Any]], int]:
    macro: List[Dict[str, Any]] = []
    words = 0
    for r in rows:
        if words >= int(words_per_update):
            break
        macro.append(r)
        words += int(r.get("words", s64.wc(r.get("text", ""))))
    return macro, words


def kl_metrics(model_a, model_b, examples: List[Dict[str, Any]], device: torch.device,
               micro_bs: int, T: float, seed: int | None, max_batches: int | None = None) -> Dict[str, Any]:
    """Return KL distributions for P_b || P_a and P_a || P_b.

    For research orientation, model_a is student and model_b is teacher, so
    F.kl_div(log_softmax(student), softmax(teacher)) = KL(teacher || student).
    """
    if seed is not None:
        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))
    total_forward = 0
    total_tokens = 0
    kl_b_to_a_sum = 0.0
    kl_a_to_b_sum = 0.0
    ce_teacher_on_labels = 0.0
    ce_student_on_labels = 0.0
    target_mass = 0
    per_batch = []
    for bi, start in enumerate(range(0, len(examples), micro_bs)):
        if max_batches is not None and bi >= max_batches:
            break
        mb = examples[start:start + micro_bs]
        inp = torch.stack([x["input_ids"] for x in mb]).to(device)
        att = torch.stack([x["attention_mask"] for x in mb]).to(device)
        lab = torch.stack([x["labels"] for x in mb]).to(device)
        mask = lab != -100
        if not mask.any():
            continue
        with torch.no_grad():
            out_a = model_a(input_ids=inp, attention_mask=att).logits[mask]
            out_b = model_b(input_ids=inp, attention_mask=att).logits[mask]
            a_logits = out_a / T
            b_logits = out_b / T
            a_lp = F.log_softmax(a_logits, dim=-1)
            b_lp = F.log_softmax(b_logits, dim=-1)
            a_p = torch.exp(a_lp)
            b_p = torch.exp(b_lp)
            kl_b_to_a = F.kl_div(a_lp, b_p, reduction="sum")
            kl_a_to_b = F.kl_div(b_lp, a_p, reduction="sum")
            if T != 1.0:
                kl_b_to_a = kl_b_to_a * (T * T)
                kl_a_to_b = kl_a_to_b * (T * T)
            V = out_a.shape[-1]
            ce_a = F.cross_entropy(out_a.reshape(-1, V), lab[mask].reshape(-1), reduction="sum")
            ce_b = F.cross_entropy(out_b.reshape(-1, V), lab[mask].reshape(-1), reduction="sum")
            n = int(mask.sum().item())
        total_tokens += n
        target_mass += n
        kl_b_to_a_sum += float(kl_b_to_a.detach().cpu())
        kl_a_to_b_sum += float(kl_a_to_b.detach().cpu())
        ce_student_on_labels += float(ce_a.detach().cpu())
        ce_teacher_on_labels += float(ce_b.detach().cpu())
        total_forward += 1
        per_batch.append({
            "batch_index": bi,
            "tokens": n,
            "kl_b_to_a_mean": float(kl_b_to_a.detach().cpu()) / max(1, n),
            "kl_a_to_b_mean": float(kl_a_to_b.detach().cpu()) / max(1, n),
            "student_ce_label_mean": float(ce_a.detach().cpu()) / max(1, n),
            "teacher_ce_label_mean": float(ce_b.detach().cpu()) / max(1, n),
        })
        del inp, att, lab, mask, out_a, out_b, a_logits, b_logits, a_lp, b_lp, a_p, b_p
    return {
        "forward_batches": total_forward,
        "tokens": total_tokens,
        "orientation": "KL(teacher || student) = F.kl_div(log_softmax(student_logits), softmax(teacher_logits))",
        "kl_b_to_a_mean": float(kl_b_to_a_sum / max(1, total_tokens)),
        "kl_a_to_b_mean": float(kl_a_to_b_sum / max(1, total_tokens)),
        "student_ce_label_mean": float(ce_student_on_labels / max(1, target_mass)),
        "teacher_ce_label_mean": float(ce_teacher_on_labels / max(1, target_mass)),
        "per_batch_head": per_batch[:10],
    }


def set_mode(model, mode: str) -> None:
    if mode == "train":
        model.train()
    elif mode == "eval":
        model.eval()
    else:
        raise ValueError(mode)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tail-jsonl", type=pathlib.Path, default=s64.DEFAULT_TAIL)
    ap.add_argument("--out-dir", type=pathlib.Path, default=DEFAULT_OUT)
    ap.add_argument("--words-per-update", type=int, default=39533)
    ap.add_argument("--seq-length", type=int, default=512)
    ap.add_argument("--train-seed", type=int, default=62064)
    ap.add_argument("--mask-prob", type=float, default=0.15)
    ap.add_argument("--private-scale", type=float, default=0.75)
    ap.add_argument("--micro-batch", type=int, default=4)
    ap.add_argument("--max-examples", type=int, default=33)
    ap.add_argument("--max-batches", type=int, default=None)
    ap.add_argument("--kl-temperature", type=float, default=1.0)
    ap.add_argument("--gpu", type=int, default=0)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    setup_cache(args.out_dir)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    rows, prefix_info = s64.load_prefix(args.tail_jsonl, 1, int(args.words_per_update))
    tokenizer = bridge.AutoTokenizer.from_pretrained(str(bridge.PARENT_PATH), local_files_only=True, use_fast=True)
    wgb = bridge.WordGroupBuilder(tokenizer)
    macro_rows, macro_words = first_macro_rows(rows, int(args.words_per_update))
    pres_examples, pres_n, pres_w = s88.prepare_preservation(
        macro_rows, tokenizer, int(args.seq_length), wgb, int(args.train_seed), float(args.mask_prob)
    )
    pres_examples = pres_examples[:int(args.max_examples)]
    limited_n = sum(int((x["labels"] != -100).sum().item()) for x in pres_examples)

    # Load two independent copies of the same coherent86 checkpoint.
    model_a, _m1, _u1 = bridge.load_model(device, float(args.private_scale))
    model_b, _m2, _u2 = bridge.load_model(device, float(args.private_scale))
    ident_a = bridge.model_identity(model_a)
    ident_b = bridge.model_identity(model_b)
    for p in model_a.parameters():
        p.requires_grad_(False)
    for p in model_b.parameters():
        p.requires_grad_(False)

    modes = [
        ("eval_eval", "eval", "eval", 12345),
        ("train_eval", "train", "eval", 12345),
        ("train_eval_repeat_same_seed", "train", "eval", 12345),
        ("train_eval_different_seed", "train", "eval", 54321),
        ("train_train_same_seed_reset", "train", "train", 777),
        ("eval_train", "eval", "train", 12345),
    ]
    results: Dict[str, Any] = {}
    for name, mode_a, mode_b, seed in modes:
        set_mode(model_a, mode_a)
        set_mode(model_b, mode_b)
        results[name] = {
            "student_mode_model_a": mode_a,
            "teacher_mode_model_b": mode_b,
            "rng_seed_before_eval": seed,
            **kl_metrics(model_a, model_b, pres_examples, device, int(args.micro_batch), float(args.kl_temperature), seed, args.max_batches),
        }

    report = {
        "status": "PRESERVATION_KL_MODE_PROBE_DONE",
        "created_utc": now(),
        "scientific_question": "How much identical-weight KL arises from train/eval stochastic mode mismatch before learning?",
        "prefix_info": prefix_info,
        "macro_words": macro_words,
        "preservation_examples_available": len(pres_examples),
        "preservation_targets_available_before_limit": pres_n,
        "preservation_words_first_macro": pres_w,
        "limited_examples": len(pres_examples),
        "limited_targets": limited_n,
        "model_a_identity": ident_a,
        "model_b_identity": ident_b,
        "implemented_kl_direction": "KL(teacher || student), not KL(student || teacher)",
        "mode_results": results,
        "interpretation": {
            "eval_eval": "Should be near numerical zero for identical weights; nonzero indicates loading mismatch.",
            "train_eval": "Measures dropout/stochastic consistency pressure present at update 0 in research-style teacher-eval/student-train KL.",
            "warning": "These KL magnitudes are not gradient-contribution percentages; they only quantify objective value under the probed modes.",
        },
    }
    out_json = args.out_dir / "preservation_kl_mode_probe.json"
    out_md = args.out_dir / "preservation_kl_mode_probe.md"
    out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = [
        "# research preservation KL mode probe",
        "",
        f"Status: `{report['status']}`",
        f"Limited examples/targets: `{len(pres_examples)}` / `{limited_n}`",
        "",
        "Implemented orientation: `KL(teacher || student)` via `F.kl_div(log_softmax(student), softmax(teacher))`.",
        "",
        "## Mode results",
    ]
    for name, res in results.items():
        lines.append(f"- {name}: KL teacher||student mean `{res['kl_b_to_a_mean']:.8f}`, reverse `{res['kl_a_to_b_mean']:.8f}`, student CE `{res['student_ce_label_mean']:.6f}`, teacher CE `{res['teacher_ce_label_mean']:.6f}`, tokens `{res['tokens']}`")
    lines.append("")
    lines.append("The train/eval identical-weight value is stochastic regularization pressure, not acquired functional drift.")
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "out_json": rel(out_json), "out_md": rel(out_md), "mode_results": {k: {"kl_b_to_a_mean": v["kl_b_to_a_mean"], "kl_a_to_b_mean": v["kl_a_to_b_mean"], "tokens": v["tokens"]} for k, v in results.items()}}, indent=2), flush=True)

    del model_a, model_b
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
