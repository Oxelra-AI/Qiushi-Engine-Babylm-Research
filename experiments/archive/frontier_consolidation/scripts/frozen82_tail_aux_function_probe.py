#!/usr/bin/env python3
"""Source-free auxiliary functional probe for frozen-82M private-tail models.

This replays the deterministic tail batches/masks used by
frozen82_private_tail_trainer, rebuilds sparse top20 auxiliary views, and
measures conditioned and source-free NLL for completed tail checkpoints.  It tests
whether any aligned-tail benefit is present in the inference-time source-free view,
not merely in the source-conditioned view where the true source is visible.

Negative pairwise deltas mean the first model has lower NLL.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import random
import sys
import time
from typing import Any

# Writable import-time cache before transformers import.
_CACHE = pathlib.Path("experiments/archive/frontier_consolidation/data/frozen82_tail_aux_probe/import_cache").resolve()
for _k, _p in {
    "HF_HOME": _CACHE / "hf_home",
    "HF_HUB_CACHE": _CACHE / "hf_home" / "hub",
    "HUGGINGFACE_HUB_CACHE": _CACHE / "hf_home" / "hub",
    "TRANSFORMERS_CACHE": _CACHE / "transformers",
    "HF_MODULES_CACHE": _CACHE / "modules",
    "HF_DATASETS_CACHE": _CACHE / "datasets",
    "TMPDIR": _CACHE / "tmp",
}.items():
    _p.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault(_k, str(_p))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoModelForMaskedLM, AutoTokenizer

ROOT = pathlib.Path(".").resolve()
STUDY = ROOT / "experiments/archive/frontier_consolidation"
WORKSPACE = STUDY
SCRIPTS = WORKSPACE / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import frozen82_private_tail_trainer as tr  # noqa: E402

CHCK82_ENDPOINT = WORKSPACE / "training/runs/adapter128_scale1p75_h100M100M_seed43022_official_ladder/hf_model/chck_82M"
AUX_DATA = WORKSPACE / "data/sparse_aux_pair_data/top20/sparse_aux_pair_data_top20.json"
STREAM = WORKSPACE / "data/density_cleanqwen_overlay_medium_riskhard/cleanqwen_fineweb_compact_view_reinvest_100M.jsonl"
OUT_ROOT = WORKSPACE / "data/frozen82_tail_aux_probe"
RUNS = {
    "aligned": WORKSPACE / "training/runs/frozen82_tail4M_aligned_seed43022",
    "shuffled": WORKSPACE / "training/runs/frozen82_tail4M_shuffled_seed43022",
}


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def rel(p: pathlib.Path | str) -> str:
    p = pathlib.Path(p)
    try:
        return str(p.resolve().relative_to(ROOT))
    except Exception:
        return str(p)


def read_json(p: pathlib.Path) -> Any:
    return json.loads(p.read_text(encoding="utf-8"))


def reset_all(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_cache(out_dir: pathlib.Path) -> None:
    cache = out_dir / "hf_cache"
    for k, p in {
        "HF_HOME": cache / "hf_home",
        "HF_HUB_CACHE": cache / "hf_home" / "hub",
        "HUGGINGFACE_HUB_CACHE": cache / "hf_home" / "hub",
        "TRANSFORMERS_CACHE": cache / "transformers",
        "HF_MODULES_CACHE": cache / "modules",
        "HF_DATASETS_CACHE": cache / "datasets",
        "TMPDIR": cache / "tmp",
    }.items():
        p.mkdir(parents=True, exist_ok=True)
        os.environ[k] = str(p.resolve())
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def build_rows_split(units: list[tr.AuxUnit], mode: str, seed: int, step: int,
                     cls_id: int, sep_id: int, mask_id: int, pad_id: int,
                     max_len: int) -> tuple[list[dict[str, Any]], int]:
    if mode == "neutral_only" or not units:
        return [], 0
    assigned = tr.source_assignment(units, mode, seed, step)
    if not assigned:
        return [], 0
    rows: list[dict[str, Any]] = []
    charged_words = 0
    for unit, (source_ids, source_words) in zip(units, assigned):
        masked_rw = list(unit.rewrite_ids)
        lab_rw = [-100] * len(unit.rewrite_ids)
        for j, g in enumerate(unit.rewrite_word_group):
            if g in unit.rewrite_word_groups_to_mask:
                lab_rw[j] = int(unit.rewrite_ids[j])
                masked_rw[j] = int(mask_id)
        if all(x == -100 for x in lab_rw):
            continue
        cond_seq = [int(cls_id)] + list(source_ids) + [int(sep_id)] + masked_rw + [int(sep_id)]
        cond_lab = [-100] + [-100] * len(source_ids) + [-100] + lab_rw + [-100]
        free_seq = [int(cls_id)] + masked_rw + [int(sep_id)]
        free_lab = [-100] + lab_rw + [-100]
        if len(cond_seq) > max_len or len(free_seq) > max_len:
            continue
        rows.append({
            "cond_ids": cond_seq,
            "cond_labels": cond_lab,
            "free_ids": free_seq,
            "free_labels": free_lab,
            "target_count": sum(1 for x in lab_rw if x != -100),
            "source_words": int(source_words),
            "rewrite_words": int(unit.rewrite_words),
        })
        charged_words += int(source_words) + 2 * int(unit.rewrite_words)
    return rows, charged_words


def collect_rows(args, tokenizer, device: torch.device) -> dict[str, Any]:
    aux_data, aux_summary = tr.load_aux_pair_data(AUX_DATA)
    pair_eids = {int(k) for k in aux_data.keys()}
    examples = tr.load_examples_tail(STREAM, args.skip_rows, args.max_tail_charged_words)
    dataset = tr.TailDataset(examples, tokenizer, args.seq_length, pair_eids)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=tr.collate, num_workers=0, pin_memory=False)
    gen = torch.Generator(device=device)
    gen.manual_seed(args.train_rng_seed)
    mask_id = int(tokenizer.mask_token_id)
    cls_id = tokenizer.cls_token_id if tokenizer.cls_token_id is not None else tokenizer.bos_token_id
    sep_id = tokenizer.sep_token_id if tokenizer.sep_token_id is not None else tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    if cls_id is None or sep_id is None:
        raise RuntimeError("missing special ids")

    rows_by_mode = {"aligned": [], "shuffled": []}
    replay = []
    cum_main = 0
    cum_aux = 0
    for loader_step, batch in enumerate(loader, 1):
        if args.max_batches > 0 and loader_step > args.max_batches:
            break
        words = int(batch["words"].sum().item())
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        word_group = batch["word_group"].to(device)
        masked_inputs, labels = tr.apply_wwm(input_ids, attention_mask, word_group, tokenizer, args.mask_prob, gen)
        example_ids = batch["example_id"]
        is_pair = batch["is_pair"]
        pidx = is_pair.nonzero(as_tuple=False).squeeze(-1).tolist()
        if isinstance(pidx, int):
            pidx = [pidx]
        recs = []
        valid = []
        for bi in pidx:
            k = str(int(example_ids[bi].item()))
            if k in aux_data:
                valid.append(bi)
                recs.append(aux_data[k])
        units = tr.collect_aux_units(word_group, labels, valid, recs) if valid else []
        aligned_rows, aligned_words = build_rows_split(units, "aligned", args.aux_pair_shuffle_seed, loader_step,
                                                       cls_id, sep_id, mask_id, pad_id, args.aux_max_length)
        shuffled_rows, shuffled_words = build_rows_split(units, "shuffled", args.aux_pair_shuffle_seed, loader_step,
                                                         cls_id, sep_id, mask_id, pad_id, args.aux_max_length)
        next_charged = cum_main + cum_aux + words + aligned_words
        if next_charged > args.max_tail_charged_words:
            replay.append({"loader_step": loader_step, "event": "stop_before_cap", "current_charged": cum_main + cum_aux, "next_main_words": words, "next_aux_words": aligned_words})
            break
        if aligned_words != shuffled_words or len(aligned_rows) != len(shuffled_rows):
            raise RuntimeError(f"aligned/shuffled replay mismatch at step {loader_step}: words {aligned_words} {shuffled_words}, rows {len(aligned_rows)} {len(shuffled_rows)}")
        rows_by_mode["aligned"].extend(aligned_rows)
        rows_by_mode["shuffled"].extend(shuffled_rows)
        cum_main += words
        cum_aux += aligned_words
        replay.append({"loader_step": loader_step, "main_words": words, "aux_words": aligned_words, "n_rows": len(aligned_rows), "cum_main": cum_main, "cum_aux": cum_aux, "cum_charged": cum_main + cum_aux})
        del input_ids, attention_mask, word_group, masked_inputs, labels
    return {
        "rows_by_mode": rows_by_mode,
        "pad_id": pad_id,
        "replay_summary": {
            "batches": sum(1 for x in replay if x.get("event") != "stop_before_cap"),
            "cum_main_words": cum_main,
            "cum_aux_words": cum_aux,
            "cum_charged_words": cum_main + cum_aux,
            "aligned_aux_rows": len(rows_by_mode["aligned"]),
            "shuffled_aux_rows": len(rows_by_mode["shuffled"]),
            "aligned_targets": sum(r["target_count"] for r in rows_by_mode["aligned"]),
            "shuffled_targets": sum(r["target_count"] for r in rows_by_mode["shuffled"]),
            "aux_summary": aux_summary,
            "stop_record": replay[-1] if replay and replay[-1].get("event") == "stop_before_cap" else None,
        },
        "replay_log_sample": replay[:3] + replay[-3:],
    }


def pad_batch(seqs: list[list[int]], labels: list[list[int]], pad_id: int, max_len: int, device: torch.device):
    B = len(seqs)
    inp = torch.full((B, max_len), int(pad_id), dtype=torch.long, device=device)
    att = torch.zeros((B, max_len), dtype=torch.long, device=device)
    lab = torch.full((B, max_len), -100, dtype=torch.long, device=device)
    for i, (s, l) in enumerate(zip(seqs, labels)):
        inp[i, :len(s)] = torch.tensor(s, dtype=torch.long, device=device)
        att[i, :len(s)] = 1
        lab[i, :len(l)] = torch.tensor(l, dtype=torch.long, device=device)
    return inp, att, lab


def ce_sum_for_rows(model, rows: list[dict[str, Any]], view: str, pad_id: int, max_len: int, micro_batch: int, device: torch.device) -> tuple[float, int]:
    if not rows:
        return 0.0, 0
    ids_key = f"{view}_ids"
    lab_key = f"{view}_labels"
    total_loss = 0.0
    total_targets = 0
    model.eval()
    with torch.no_grad():
        for s in range(0, len(rows), micro_batch):
            chunk = rows[s:s + micro_batch]
            inp, att, lab = pad_batch([r[ids_key] for r in chunk], [r[lab_key] for r in chunk], pad_id, max_len, device)
            out = model(input_ids=inp, attention_mask=att)
            vocab = out.logits.shape[-1]
            loss_sum = F.cross_entropy(out.logits.reshape(-1, vocab), lab.reshape(-1), ignore_index=-100, reduction="sum")
            total_loss += float(loss_sum.detach().cpu())
            total_targets += int((lab != -100).sum().item())
            del inp, att, lab, out, loss_sum
    return total_loss, total_targets


def model_path(label: str, endpoint: str) -> pathlib.Path:
    if label == "base":
        return CHCK82_ENDPOINT
    root = RUNS[label] / "hf_model"
    ep = endpoint
    if ep == "auto":
        ep = "final"
    return root / ep


def set_private_enabled(model, enabled: bool) -> None:
    if hasattr(model, "set_private_enabled"):
        model.set_private_enabled(enabled)
    elif hasattr(model, "deberta"):
        for layer in model.deberta.encoder.layer:
            if hasattr(layer, "private_adapter"):
                layer.private_adapter.enabled = bool(enabled)
        if hasattr(model.config, "private_adapter_enabled"):
            model.config.private_adapter_enabled = bool(enabled)


def eval_model(label: str, rows_by_mode: dict[str, list[dict[str, Any]]], pad_id: int, args, device: torch.device) -> dict[str, Any]:
    mp = model_path(label, args.endpoint)
    if not (mp / "model.safetensors").exists() and not (mp / "pytorch_model.bin").exists():
        raise FileNotFoundError(mp)
    print(json.dumps({"event": "load_model", "label": label, "path": rel(mp), "disable_private": args.disable_private}), flush=True)
    model = AutoModelForMaskedLM.from_pretrained(str(mp), trust_remote_code=True, local_files_only=True)
    model.to(device)
    model.eval()
    if args.disable_private:
        set_private_enabled(model, False)
    metrics = None
    if label in RUNS and (RUNS[label] / "scientific_metrics.json").exists():
        metrics = read_json(RUNS[label] / "scientific_metrics.json")
    out = {"label": label, "model_path": rel(mp), "private_disabled": args.disable_private, "training_metrics": metrics, "views": {}}
    for source_mode, rows in rows_by_mode.items():
        view_rec = {}
        for view in ["cond", "free"]:
            loss_sum, targets = ce_sum_for_rows(model, rows, view, pad_id, args.aux_max_length, args.micro_batch, device)
            view_rec[view] = {"loss_sum": loss_sum, "targets": targets, "nll": None if targets == 0 else loss_sum / targets, "n_rows": len(rows)}
            print(json.dumps({"event": "eval_view", "label": label, "source_mode": source_mode, "view": view, "targets": targets, "nll": view_rec[view]["nll"]}), flush=True)
        out["views"][source_mode] = view_rec
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return out


def pairwise(results: dict[str, Any]) -> dict[str, Any]:
    out = {}
    labels = list(results)
    for a in labels:
        for b in labels:
            if a == b:
                continue
            key = f"{a}_minus_{b}"
            out[key] = {}
            for source_mode in ["aligned", "shuffled"]:
                out[key][source_mode] = {}
                for view in ["cond", "free"]:
                    av = results[a]["views"][source_mode][view]["nll"]
                    bv = results[b]["views"][source_mode][view]["nll"]
                    out[key][source_mode][view] = None if av is None or bv is None else av - bv
    return out


def build_md(output: dict[str, Any]) -> str:
    lines = [f"# research frozen-82M tail auxiliary functional probe — {output['label']}", "", f"Created: `{output['created_utc']}`", "", "## Replay", "", "```json", json.dumps(output["replay_summary"], indent=2), "```", "", "## View NLLs", "", "| model | source assignment | view | NLL | targets |", "|---|---|---|---:|---:|"]
    for lab, rec in output["model_results"].items():
        for sm, vrec in rec["views"].items():
            for view in ["cond", "free"]:
                r = vrec[view]
                lines.append(f"| {lab} | {sm} | {view} | {r['nll']:.6f} | {r['targets']} |")
    lines += ["", "## Key deltas (negative means first model lower NLL)", "", "```json", json.dumps(output["key_deltas"], indent=2), "```", "", f"JSON: `{output['out_json']}`"]
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", default="tail4M_adapter_on")
    ap.add_argument("--models", nargs="+", default=["aligned", "shuffled", "base"], choices=["aligned", "shuffled", "base"])
    ap.add_argument("--endpoint", default="final")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--max-tail-charged-words", type=int, default=4_000_000)
    ap.add_argument("--skip-rows", type=int, default=530_944)
    ap.add_argument("--max-batches", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--seq-length", type=int, default=256)
    ap.add_argument("--aux-max-length", type=int, default=464)
    ap.add_argument("--mask-prob", type=float, default=0.15)
    ap.add_argument("--train-rng-seed", type=int, default=43023)
    ap.add_argument("--aux-pair-shuffle-seed", type=int, default=43022)
    ap.add_argument("--micro-batch", type=int, default=32)
    ap.add_argument("--disable-private", action="store_true")
    args = ap.parse_args()

    out_dir = OUT_ROOT / args.label
    out_dir.mkdir(parents=True, exist_ok=True)
    ensure_cache(out_dir)
    if args.device == "cuda":
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(str(CHCK82_ENDPOINT), local_files_only=True, use_fast=True)
    reset_all(args.train_rng_seed)
    print(json.dumps({"event": "collect_tail_rows_start", "device": str(device), "label": args.label}), flush=True)
    row_pack = collect_rows(args, tokenizer, device)
    print(json.dumps({"event": "collect_tail_rows_done", **row_pack["replay_summary"]}), flush=True)

    results = {}
    for lab in args.models:
        results[lab] = eval_model(lab, row_pack["rows_by_mode"], row_pack["pad_id"], args, device)
    pw = pairwise(results)
    key_deltas = {
        "aligned_model_minus_shuffled_model_on_true_free_view": pw.get("aligned_minus_shuffled", {}).get("aligned", {}).get("free"),
        "aligned_model_minus_shuffled_model_on_true_conditioned_view": pw.get("aligned_minus_shuffled", {}).get("aligned", {}).get("cond"),
        "aligned_model_minus_base_on_true_free_view": pw.get("aligned_minus_base", {}).get("aligned", {}).get("free"),
        "aligned_model_minus_base_on_true_conditioned_view": pw.get("aligned_minus_base", {}).get("aligned", {}).get("cond"),
        "shuffled_model_minus_base_on_shuffled_free_view": pw.get("shuffled_minus_base", {}).get("shuffled", {}).get("free"),
        "aligned_model_minus_shuffled_model_on_shuffled_free_view": pw.get("aligned_minus_shuffled", {}).get("shuffled", {}).get("free"),
    }
    output = {
        "status": "FROZEN82_TAIL_AUX_FUNCTION_PROBE",
        "label": args.label,
        "created_utc": now(),
        "args": vars(args),
        "runs": {lab: (rel(CHCK82_ENDPOINT) if lab == "base" else rel(RUNS[lab])) for lab in ["aligned", "shuffled", "base"]},
        "replay_summary": row_pack["replay_summary"],
        "replay_log_sample": row_pack["replay_log_sample"],
        "model_results": results,
        "pairwise_nll_deltas": pw,
        "key_deltas": key_deltas,
        "interpretation": "Negative deltas mean the first model has lower held auxiliary NLL. Source-free true-view deltas test whether source-correspondence learning survives when the paired source is absent at inference time.",
    }
    suffix = "_private_off" if args.disable_private else ""
    out_json = out_dir / f"{args.label}{suffix}.json"
    out_md = out_dir / f"{args.label}{suffix}.md"
    output["out_json"] = rel(out_json)
    output["out_md"] = rel(out_md)
    out_json.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    out_md.write_text(build_md(output), encoding="utf-8")
    print(json.dumps({"status": output["status"], "label": args.label, "disable_private": args.disable_private, "key_deltas": key_deltas, "out_json": rel(out_json), "out_md": rel(out_md)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
