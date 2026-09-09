#!/usr/bin/env python3
"""research: common-support endpoint probe for acquisition and preservation.

Scientific purpose
------------------
research established a fixed set of token positions that are masked by both
clean-style ordinary WWM preservation and dense-corrupted non-label preservation.
At those same positions, the dense rendering made the coherent86 parent much less
accurate while the acquisition-only endpoint improved ground-truth prediction.
That means parent anchoring on dense states can oppose beneficial acquired fit,
not merely repair forgetting.

This script reconstructs the same deterministic common-support target positions
and scores multiple frozen endpoints against the coherent86 parent under both
ordinary-common and dense-common renderings.  It records parent KL together with
ground-truth CE and rank, so later results from the dense-corruption preservation
arm can be read as: preserved parent function, retained acquired fit, suppression
of acquired fit, or a cleaner separation between inherited-function protection
and useful learning.

The probe is not leaderboard scoring and does not modify any checkpoint.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import gc
import json
import math
import os
import pathlib
import sys
import time
from collections import Counter
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn.functional as F

ROOT = _public_path('.')
STUDY = _public_path('experiments/archive/functional_learning')
SCRIPTS = _public_path('experiments/archive/functional_learning/scripts')
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import preservation_geometry_matched_diagnostic as geom  # noqa: E402
import corrected_bridge_trainer as bridge  # noqa: E402
import real_stream_train_weighted as s64  # noqa: E402

DEFAULT_OUT = _public_path('experiments/archive/functional_learning/data/common_support_endpoint_probe')

DEFAULT_ENDPOINTS: List[Tuple[str, pathlib.Path]] = [
    ("coherent86_parent", bridge.PARENT_PATH),
    ("ordinary_inherited_wwm_seed62064", _public_path('experiments/archive/functional_learning/data/unchanged_focus_weighted_train/inherited_wwm/checkpoints/update_0080')),
    ("exact_ms_seed62064", _public_path('experiments/archive/functional_learning/data/densemask_sparselabel_train_seed62064/correspondence_focus_weighted/checkpoints/update_0080')),
    ("clean_ms_kl_seed62064", _public_path('experiments/archive/functional_learning/data/clean_preservation_lambda1_eval_full80/checkpoints/update_0080')),
    ("clean_ms_kl_seed62065", _public_path('experiments/archive/functional_learning/data/clean_preservation_lambda1_eval_seed62065_full80/checkpoints/update_0080')),
    ("densecorr_ms_kl_seed62064", _public_path('experiments/archive/functional_learning/data/densecorruption_preservation_lambda1_full80/checkpoints/update_0080')),
]


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


def parse_name_path(items: Iterable[str]) -> List[Tuple[str, pathlib.Path]]:
    out: List[Tuple[str, pathlib.Path]] = []
    for item in items:
        if "=" not in item:
            raise ValueError(f"Endpoint must be name=path, got {item!r}")
        name, path = item.split("=", 1)
        name = name.strip()
        if not name:
            raise ValueError(f"Endpoint name empty in {item!r}")
        out.append((name, pathlib.Path(path)))
    return out


def endpoint_list(extra: Iterable[str], include_defaults: bool = True) -> List[Tuple[str, pathlib.Path]]:
    pairs = list(DEFAULT_ENDPOINTS) if include_defaults else []
    pairs.extend(parse_name_path(extra))
    seen = set()
    clean: List[Tuple[str, pathlib.Path]] = []
    for name, path in pairs:
        if name in seen:
            continue
        seen.add(name)
        if path.exists():
            clean.append((name, path))
    return clean


def split_macros(rows: List[Dict[str, Any]], max_updates: int, words_per_update: int) -> List[Dict[str, Any]]:
    return geom.split_macros(rows, max_updates, words_per_update)


def stack_batch(examples: List[Dict[str, Any]], device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    inp = torch.stack([x["input_ids"] for x in examples]).to(device)
    att = torch.stack([x["attention_mask"] for x in examples]).to(device)
    lab = torch.stack([x["labels"] for x in examples]).to(device)
    mask = lab != -100
    return inp, att, lab, mask


def token_rank(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    lab_logits = logits.gather(1, labels.view(-1, 1)).squeeze(1)
    return (logits > lab_logits.view(-1, 1)).sum(dim=1).float() + 1.0


def score_against_parent(model, parent, examples: List[Dict[str, Any]], device: torch.device, micro_batch: int) -> Dict[str, Any]:
    model.eval()
    parent.eval()
    sums = Counter()
    macro_counts = Counter()
    with torch.no_grad():
        for mb in geom.batches(examples, int(micro_batch)):
            inp, att, lab, mask = stack_batch(mb, device)
            if not mask.any():
                del inp, att, lab, mask
                continue
            labels = lab[mask]
            p_out = parent(input_ids=inp, attention_mask=att)
            m_out = model(input_ids=inp, attention_mask=att)
            p_logits = p_out.logits[mask].float()
            m_logits = m_out.logits[mask].float()
            vocab = p_logits.shape[-1]
            p_lp = F.log_softmax(p_logits, dim=-1)
            m_lp = F.log_softmax(m_logits, dim=-1)
            p_prob = torch.exp(p_lp)
            m_prob = torch.exp(m_lp)
            parent_ce = F.cross_entropy(p_logits.reshape(-1, vocab), labels.reshape(-1), reduction="none")
            model_ce = F.cross_entropy(m_logits.reshape(-1, vocab), labels.reshape(-1), reduction="none")
            parent_rank = token_rank(p_logits, labels)
            model_rank = token_rank(m_logits, labels)
            parent_entropy = -(p_prob * p_lp).sum(dim=-1)
            model_entropy = -(m_prob * m_lp).sum(dim=-1)
            parent_top1 = p_prob.max(dim=-1).values
            model_top1 = m_prob.max(dim=-1).values
            kl_parent_to_model = F.kl_div(m_lp, p_prob, reduction="none").sum(dim=-1)
            kl_model_to_parent = F.kl_div(p_lp, m_prob, reduction="none").sum(dim=-1)
            n = int(labels.numel())
            sums["tokens"] += n
            sums["parent_ce"] += float(parent_ce.sum().detach().cpu())
            sums["model_ce"] += float(model_ce.sum().detach().cpu())
            sums["parent_rank"] += float(parent_rank.sum().detach().cpu())
            sums["model_rank"] += float(model_rank.sum().detach().cpu())
            sums["parent_entropy"] += float(parent_entropy.sum().detach().cpu())
            sums["model_entropy"] += float(model_entropy.sum().detach().cpu())
            sums["parent_top1"] += float(parent_top1.sum().detach().cpu())
            sums["model_top1"] += float(model_top1.sum().detach().cpu())
            sums["kl_parent_to_model"] += float(kl_parent_to_model.sum().detach().cpu())
            sums["kl_model_to_parent"] += float(kl_model_to_parent.sum().detach().cpu())
            for ex in mb:
                macro_counts[str(ex.get("macro_index0", "unknown"))] += int((ex["labels"] != -100).sum().item())
            del inp, att, lab, mask, labels, p_out, m_out, p_logits, m_logits, p_lp, m_lp, p_prob, m_prob
            del parent_ce, model_ce, parent_rank, model_rank, parent_entropy, model_entropy, parent_top1, model_top1
            del kl_parent_to_model, kl_model_to_parent
    n_tok = int(sums["tokens"])
    if n_tok <= 0:
        return {"tokens": 0}
    parent_ce = float(sums["parent_ce"] / n_tok)
    model_ce = float(sums["model_ce"] / n_tok)
    parent_rank = float(sums["parent_rank"] / n_tok)
    model_rank = float(sums["model_rank"] / n_tok)
    return {
        "tokens": n_tok,
        "parent_target_ce": parent_ce,
        "model_target_ce": model_ce,
        "ce_gain_vs_parent": parent_ce - model_ce,
        "ce_delta_model_minus_parent": model_ce - parent_ce,
        "parent_rank": parent_rank,
        "model_rank": model_rank,
        "rank_gain_vs_parent": parent_rank - model_rank,
        "rank_delta_model_minus_parent": model_rank - parent_rank,
        "parent_entropy": float(sums["parent_entropy"] / n_tok),
        "model_entropy": float(sums["model_entropy"] / n_tok),
        "parent_top1": float(sums["parent_top1"] / n_tok),
        "model_top1": float(sums["model_top1"] / n_tok),
        "kl_parent_to_model": float(sums["kl_parent_to_model"] / n_tok),
        "kl_model_to_parent": float(sums["kl_model_to_parent"] / n_tok),
        "macro_token_counts": dict(macro_counts),
    }


def add_macro_index(examples: List[Dict[str, Any]], macro_index0: int) -> List[Dict[str, Any]]:
    out = []
    for ex in examples:
        y = dict(ex)
        y["macro_index0"] = int(macro_index0)
        out.append(y)
    return out


def target_records(branch_examples_by_macro: Dict[str, List[List[Dict[str, Any]]]], branch_names: List[str]) -> List[Dict[str, Any]]:
    rows = []
    for branch in branch_names:
        for macro_slot, examples in enumerate(branch_examples_by_macro[branch]):
            for ex in examples:
                labels = ex["labels"]
                positions = torch.nonzero(labels != -100, as_tuple=False).view(-1).tolist()
                if not positions:
                    continue
                rows.append({
                    "branch": branch,
                    "macro_slot": int(macro_slot),
                    "macro_index0": int(ex.get("macro_index0", -1)),
                    "row_key": ex.get("row_key"),
                    "words": int(ex.get("words", 0)),
                    "positions": [int(p) for p in positions],
                    "target_ids": [int(labels[int(p)].item()) for p in positions],
                })
    return rows


def mean_fmt(x: Any, nd: int = 6) -> str:
    if x is None:
        return ""
    if isinstance(x, float):
        if math.isnan(x) or math.isinf(x):
            return str(x)
        return f"{x:.{nd}g}"
    return str(x)


def build_contrasts(scores: Dict[str, Any], branch_names: List[str]) -> Dict[str, Any]:
    contrasts: Dict[str, Any] = {}
    for name, by_branch in scores.items():
        if "ordinary_common" in by_branch and "dense_common" in by_branch:
            o = by_branch["ordinary_common"]
            d = by_branch["dense_common"]
            if o.get("tokens") and d.get("tokens"):
                contrasts[name] = {
                    "dense_minus_ordinary_parent_ce": d["parent_target_ce"] - o["parent_target_ce"],
                    "dense_minus_ordinary_model_ce": d["model_target_ce"] - o["model_target_ce"],
                    "dense_minus_ordinary_ce_gain_vs_parent": d["ce_gain_vs_parent"] - o["ce_gain_vs_parent"],
                    "dense_minus_ordinary_parent_rank": d["parent_rank"] - o["parent_rank"],
                    "dense_minus_ordinary_model_rank": d["model_rank"] - o["model_rank"],
                    "dense_minus_ordinary_rank_gain_vs_parent": d["rank_gain_vs_parent"] - o["rank_gain_vs_parent"],
                    "dense_minus_ordinary_kl_parent_to_model": d["kl_parent_to_model"] - o["kl_parent_to_model"],
                }
    if "exact_ms_seed62064" in scores:
        exact = scores["exact_ms_seed62064"]
        rel_exact: Dict[str, Any] = {}
        for name, by_branch in scores.items():
            if name == "exact_ms_seed62064":
                continue
            branch_comp: Dict[str, Any] = {}
            for branch in branch_names:
                if branch not in by_branch or branch not in exact:
                    continue
                a = by_branch[branch]
                b = exact[branch]
                if not a.get("tokens") or not b.get("tokens"):
                    continue
                branch_comp[branch] = {
                    "model_ce_minus_exact": a["model_target_ce"] - b["model_target_ce"],
                    "ce_gain_vs_parent_minus_exact": a["ce_gain_vs_parent"] - b["ce_gain_vs_parent"],
                    "model_rank_minus_exact": a["model_rank"] - b["model_rank"],
                    "rank_gain_vs_parent_minus_exact": a["rank_gain_vs_parent"] - b["rank_gain_vs_parent"],
                    "kl_parent_to_model_minus_exact": a["kl_parent_to_model"] - b["kl_parent_to_model"],
                }
            rel_exact[name] = branch_comp
        contrasts["relative_to_exact_ms_seed62064"] = rel_exact
    return contrasts


def write_outputs(out_dir: pathlib.Path, result: Dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / "common_support_endpoint_probe.json"
    out_md = out_dir / "common_support_endpoint_probe.md"
    out_targets = out_dir / "frozen_common_support_targets.jsonl"
    out_json.write_text(json.dumps(result, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    with out_targets.open("w", encoding="utf-8") as f:
        for rec in result.get("frozen_target_records", []):
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    lines = [
        "# research common-support endpoint probe",
        "",
        f"Created: `{result['created_utc']}`",
        "",
        "This probe reuses the deterministic common token positions from the research rendering comparison. It scores ground-truth CE/rank and parent KL for existing endpoints under ordinary-common and dense-common renderings. The positions are familiar training-row positions but were not sparse acquisition labels; their improvement is not held-out transfer evidence.",
        "",
        "## Target support",
        "",
        f"- Macro indices: `{result['selected_macro_indices0']}`",
        f"- Frozen target record JSONL: `{rel(out_targets)}`",
        "",
        "| branch | examples | target tokens |",
        "|---|---:|---:|",
    ]
    for branch, s in result["branch_support"].items():
        lines.append(f"| {branch} | {s['examples']} | {s['tokens']} |")
    lines += ["", "## Endpoint scores", "", "| endpoint | branch | KL(parent||endpoint) | parent CE | endpoint CE | CE gain vs parent | parent rank | endpoint rank | rank gain vs parent |", "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for endpoint, by_branch in result["endpoint_scores"].items():
        for branch in result["branches_scored"]:
            rec = by_branch.get(branch, {})
            if not rec.get("tokens"):
                continue
            lines.append(
                f"| {endpoint} | {branch} | {mean_fmt(rec.get('kl_parent_to_model'))} | {mean_fmt(rec.get('parent_target_ce'))} | {mean_fmt(rec.get('model_target_ce'))} | {mean_fmt(rec.get('ce_gain_vs_parent'))} | {mean_fmt(rec.get('parent_rank'))} | {mean_fmt(rec.get('model_rank'))} | {mean_fmt(rec.get('rank_gain_vs_parent'))} |"
            )
    lines += ["", "## Dense-minus-ordinary common contrast", "", "| endpoint | Δ endpoint CE | Δ CE gain vs parent | Δ endpoint rank | Δ rank gain vs parent | Δ KL(parent||endpoint) |", "|---|---:|---:|---:|---:|---:|"]
    for endpoint, rec in result["contrasts"].items():
        if endpoint == "relative_to_exact_ms_seed62064":
            continue
        lines.append(
            f"| {endpoint} | {mean_fmt(rec.get('dense_minus_ordinary_model_ce'))} | {mean_fmt(rec.get('dense_minus_ordinary_ce_gain_vs_parent'))} | {mean_fmt(rec.get('dense_minus_ordinary_model_rank'))} | {mean_fmt(rec.get('dense_minus_ordinary_rank_gain_vs_parent'))} | {mean_fmt(rec.get('dense_minus_ordinary_kl_parent_to_model'))} |"
        )
    if "relative_to_exact_ms_seed62064" in result["contrasts"]:
        lines += ["", "## Relative to exact `(M,S)` seed62064", "", "Negative CE/rank deltas mean the endpoint predicts the ground-truth token better than exact `(M,S)` on the same rendering and positions.", "", "| endpoint | branch | endpoint CE − exact CE | CE gain diff | endpoint rank − exact rank | rank gain diff | KL diff |", "|---|---|---:|---:|---:|---:|---:|"]
        for endpoint, by_branch in result["contrasts"]["relative_to_exact_ms_seed62064"].items():
            for branch, rec in by_branch.items():
                lines.append(
                    f"| {endpoint} | {branch} | {mean_fmt(rec.get('model_ce_minus_exact'))} | {mean_fmt(rec.get('ce_gain_vs_parent_minus_exact'))} | {mean_fmt(rec.get('model_rank_minus_exact'))} | {mean_fmt(rec.get('rank_gain_vs_parent_minus_exact'))} | {mean_fmt(rec.get('kl_parent_to_model_minus_exact'))} |"
                )
    lines += ["", "## Scientific reading", "", result["scientific_reading"], ""]
    out_md.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=pathlib.Path, default=DEFAULT_OUT)
    ap.add_argument("--tail-jsonl", type=pathlib.Path, default=s64.DEFAULT_TAIL)
    ap.add_argument("--macro-indices", default="0,20,40,60")
    ap.add_argument("--branches", nargs="+", default=["ordinary_common", "dense_common"], choices=["ordinary_common", "dense_common", "ordinary_full", "dense_nonlabel_full"])
    ap.add_argument("--extra-endpoint", action="append", default=[], help="Additional endpoint as name=path")
    ap.add_argument("--no-default-endpoints", action="store_true")
    ap.add_argument("--max-updates", type=int, default=80)
    ap.add_argument("--words-per-update", type=int, default=39533)
    ap.add_argument("--seq-length", type=int, default=512)
    ap.add_argument("--train-seed", type=int, default=62064)
    ap.add_argument("--mask-prob", type=float, default=0.15)
    ap.add_argument("--focus-prob", type=float, default=0.35)
    ap.add_argument("--max-focus-groups-per-row", type=int, default=16)
    ap.add_argument("--max-dense-mask-groups", type=int, default=128)
    ap.add_argument("--private-scale", type=float, default=0.75)
    ap.add_argument("--micro-batch", type=int, default=8)
    ap.add_argument("--device", choices=["cpu", "cuda", "auto"], default="cpu")
    ap.add_argument("--gpu", type=int, default=0)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    setup_cache(args.out_dir)
    torch.set_num_threads(max(1, min(12, int(os.environ.get("QIUSHI_TORCH_THREADS", "8")))))
    if args.device == "cuda" and torch.cuda.is_available():
        device = torch.device(f"cuda:{int(args.gpu)}")
    elif args.device == "auto" and torch.cuda.is_available():
        device = torch.device(f"cuda:{int(args.gpu)}")
    else:
        device = torch.device("cpu")

    geom.install_densemask_patch(int(args.max_dense_mask_groups))
    rows, prefix_info = s64.load_prefix(pathlib.Path(args.tail_jsonl), int(args.max_updates), int(args.words_per_update))
    macros_all = split_macros(rows, int(args.max_updates), int(args.words_per_update))
    wanted = geom.parse_macro_indices(args.macro_indices)
    selected = [macros_all[i] for i in wanted if 0 <= i < len(macros_all)]
    if not selected:
        raise RuntimeError(f"No selected macros from {wanted}")

    tokenizer = bridge.AutoTokenizer.from_pretrained(str(bridge.PARENT_PATH), local_files_only=True, use_fast=True)
    wgb = bridge.WordGroupBuilder(tokenizer)
    per_macro: Dict[str, List[List[Dict[str, Any]]]] = {k: [] for k in ["ordinary_common", "dense_common", "ordinary_full", "dense_nonlabel_full"]}
    support_counts = Counter()
    macro_support_records = []
    for macro in selected:
        rends, st = geom.build_preservation_renderings(macro["rows"], tokenizer, wgb, args)
        for branch, exs in rends.items():
            per_macro[branch].append(add_macro_index(exs, int(macro["update_index0"])))
        counts = st["counts"]
        support_counts.update(counts)
        support_counts["selected_rows"] += len(macro["rows"])
        support_counts["selected_words"] += int(macro["words"])
        macro_support_records.append({"update_index0": int(macro["update_index0"]), "rows": len(macro["rows"]), "words": int(macro["words"]), **counts})

    branch_examples: Dict[str, List[Dict[str, Any]]] = {}
    branch_support: Dict[str, Dict[str, int]] = {}
    for branch in args.branches:
        exs: List[Dict[str, Any]] = []
        for macro_exs in per_macro[branch]:
            exs.extend(macro_exs)
        branch_examples[branch] = exs
        branch_support[branch] = {"examples": len(exs), "tokens": int(sum(int((x["labels"] != -100).sum().item()) for x in exs))}

    endpoints = endpoint_list(args.extra_endpoint, include_defaults=not args.no_default_endpoints)
    parent, parent_load = geom.load_endpoint(bridge.PARENT_PATH, device, float(args.private_scale))
    for p in parent.parameters():
        p.requires_grad_(False)
    parent.eval()

    endpoint_scores: Dict[str, Any] = {}
    endpoint_loads: Dict[str, Any] = {"coherent86_parent_reference": parent_load}
    t0 = time.time()
    for name, path in endpoints:
        if pathlib.Path(path).resolve() == pathlib.Path(bridge.PARENT_PATH).resolve():
            model = parent
            load = parent_load
            loaded_separately = False
        else:
            model, load = geom.load_endpoint(pathlib.Path(path), device, float(args.private_scale))
            for p in model.parameters():
                p.requires_grad_(False)
            model.eval()
            loaded_separately = True
        endpoint_loads[name] = load
        by_branch: Dict[str, Any] = {}
        for branch, exs in branch_examples.items():
            by_branch[branch] = score_against_parent(model, parent, exs, device, int(args.micro_batch))
        endpoint_scores[name] = by_branch
        print(json.dumps({"event": "endpoint_scored", "endpoint": name, "branches": list(by_branch.keys()), "elapsed_sec": round(time.time() - t0, 1)}, ensure_ascii=False), flush=True)
        if loaded_separately:
            del model
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

    contrasts = build_contrasts(endpoint_scores, list(args.branches))
    frozen_records = target_records(per_macro, [b for b in args.branches if b in {"ordinary_common", "dense_common"}])

    reading = []
    exact_dense = endpoint_scores.get("exact_ms_seed62064", {}).get("dense_common")
    exact_ord = endpoint_scores.get("exact_ms_seed62064", {}).get("ordinary_common")
    if exact_dense and exact_dense.get("tokens"):
        reading.append(
            f"At dense-common positions, exact `(M,S)` improves ground-truth CE by {exact_dense['ce_gain_vs_parent']:.6f} and mean rank by {exact_dense['rank_gain_vs_parent']:.3f} relative to coherent86, while its parent KL is {exact_dense['kl_parent_to_model']:.6f}. This is the key signal: dense-state parent anchoring can constrain predictions where acquisition has made the ground-truth token easier, not only states where the student has drifted harmfully."
        )
    if exact_ord and exact_ord.get("tokens"):
        reading.append(
            f"On ordinary-common rendering at the same positions, exact `(M,S)` changes CE by only {exact_ord['ce_gain_vs_parent']:.6f} and rank by {exact_ord['rank_gain_vs_parent']:.3f}; the dense rendering is therefore where the acquired fit is concentrated."
        )
    for nm in ["clean_ms_kl_seed62064", "clean_ms_kl_seed62065", "ordinary_inherited_wwm_seed62064", "densecorr_ms_kl_seed62064"]:
        dense = endpoint_scores.get(nm, {}).get("dense_common")
        if dense and dense.get("tokens"):
            reading.append(
                f"{nm} has dense-common CE gain {dense['ce_gain_vs_parent']:.6f}, rank gain {dense['rank_gain_vs_parent']:.3f}, and parent KL {dense['kl_parent_to_model']:.6f}. Compare this triple with exact `(M,S)` rather than reading reduced KL alone as better preservation."
            )
    if "densecorr_ms_kl_seed62064" not in endpoint_scores:
        reading.append("The full dense-corruption endpoint was not present when this probe ran. Rerun the same script with the default endpoint set after `densecorruption_preservation_lambda1_full80/checkpoints/update_0080` exists; the frozen target JSONL and deterministic reconstruction keep the common positions fixed.")

    result = {
        "status": "COMMON_SUPPORT_ENDPOINT_PROBE_DONE",
        "created_utc": now(),
        "script": rel(_public_path('experiments/archive/functional_learning/scripts/common_support_endpoint_probe.py')),
        "scientific_question": "At fixed common preservation positions, do endpoints reduce parent KL by protecting inherited behavior or by suppressing useful dense-state ground-truth fit?",
        "scope": "frozen-checkpoint common-support probe on familiar training rows; not official BabyLM scoring and not held-out transfer evidence",
        "selected_macro_indices0": wanted,
        "prefix_info": prefix_info,
        "branches_scored": list(args.branches),
        "settings": {
            "branches": list(args.branches),
            "seq_length": int(args.seq_length),
            "train_seed": int(args.train_seed),
            "mask_prob": float(args.mask_prob),
            "focus_prob": float(args.focus_prob),
            "max_focus_groups_per_row": int(args.max_focus_groups_per_row),
            "max_dense_mask_groups": int(args.max_dense_mask_groups),
            "private_scale": float(args.private_scale),
            "micro_batch": int(args.micro_batch),
            "device": str(device),
        },
        "support_summary": dict(support_counts),
        "macro_support_records": macro_support_records,
        "branch_support": branch_support,
        "endpoints": [{"name": n, "path": rel(p)} for n, p in endpoints],
        "endpoint_loads": endpoint_loads,
        "endpoint_scores": endpoint_scores,
        "contrasts": contrasts,
        "frozen_target_records": frozen_records,
        "scientific_reading": " ".join(reading),
    }
    write_outputs(args.out_dir, result)
    print(json.dumps({"status": result["status"], "out_json": rel(args.out_dir / "common_support_endpoint_probe.json"), "out_md": rel(args.out_dir / "common_support_endpoint_probe.md"), "endpoints_scored": list(endpoint_scores.keys()), "branches": list(args.branches)}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
