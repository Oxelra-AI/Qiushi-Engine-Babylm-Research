#!/usr/bin/env python3
"""research: score the acquired relation-state checkpoint on T/U/N state-margin probes.

The research plain-use answer harness scored generated answer fragments against
canonical foil phrases.  That is useful as masked-fragment learning evidence, but
many held answers are validator-hit windows rather than complete grammatical state
alternatives.  This script uses the T/U/N current-state template
instrument instead: every candidate phrase is placed in the same explicit state
slot, and source-only (N), true-update (T), and swapped-update (U) contexts are
preserved.

Scientific question: does the exact relation-first acquired state, before any
packet training, move held state-use margins relative to the coherent86
parent?  A positive T-vs-N or recipient/update-sensitive movement would be
transfer evidence for a reusable state-selection operation; a flat result means
The acquisition is still format/relation specific.
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
import sys
import time
from collections import defaultdict
from typing import Any, Iterable, List

# Set writable dynamic-module caches before importing transformers through the state-margin scorer.
ROOT = _public_path('.')
STUDY = _public_path('experiments/archive/functional_learning')
DEFAULT_OUT = _public_path('experiments/archive/functional_learning/data/tun_transfer')
HF_CACHE = _public_path('experiments/archive/functional_learning/data/tun_transfer/hf_cache')
for key, sub in [
    ("HF_HOME", "home"),
    ("HF_HUB_CACHE", "hub"),
    ("HF_DATASETS_CACHE", "datasets"),
    ("TRANSFORMERS_CACHE", "transformers"),
    ("HF_MODULES_CACHE", "modules"),
]:
    p = HF_CACHE / sub
    p.mkdir(parents=True, exist_ok=True)
    os.environ[key] = str(p.resolve())
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch  # noqa: E402
from transformers import AutoModelForMaskedLM, AutoTokenizer  # noqa: E402

A02_SCRIPTS = _public_path('experiments/archive/relation_learning/scripts')
if str(A02_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(A02_SCRIPTS))
import score_state_margin_probe as tun  # noqa: E402

PARENT = _public_path('models/frontier')
A01_SEED40040 = _public_path('experiments/archive/functional_learning/data/saved_state_replicate_neutral_threeentity/seed_40040/hf_model/final')
BALANCED = _public_path('experiments/archive/relation_learning/data/state_margin_probe/state_margin_probe_balanced_heldout.jsonl')
EXTENDED = _public_path('experiments/archive/relation_learning/data/state_margin_probe/state_margin_probe_extended_nontrain.jsonl')


def rel(path: pathlib.Path | str) -> str:
    try:
        return str(pathlib.Path(path).resolve().relative_to(ROOT))
    except Exception:
        return str(path)


def now_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def write_csv(path: pathlib.Path, rows: List[dict[str, Any]], fieldnames: List[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        keys: list[str] = []
        seen = set()
        for r in rows:
            for k in r.keys():
                if k not in seen:
                    seen.add(k)
                    keys.append(k)
        fieldnames = keys
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def finite_mean(vals: Iterable[Any]) -> float:
    xs = []
    for x in vals:
        try:
            xf = float(x)
        except Exception:
            continue
        if math.isfinite(xf):
            xs.append(xf)
    return sum(xs) / len(xs) if xs else float("nan")


def pct_pos(vals: Iterable[Any]) -> float:
    xs = []
    for x in vals:
        try:
            xf = float(x)
        except Exception:
            continue
        if math.isfinite(xf):
            xs.append(xf)
    return sum(1 for x in xs if x > 0) / len(xs) if xs else float("nan")


def load_model_identity(path: pathlib.Path, device: torch.device) -> tuple[Any, dict[str, Any]]:
    model = AutoModelForMaskedLM.from_pretrained(
        str(path),
        trust_remote_code=True,
        local_files_only=True,
        torch_dtype=torch.float32,
    ).eval().to(device)
    private_params = 0
    private_tensors = 0
    for n, p in model.named_parameters():
        if ".private_adapter." in n:
            private_params += int(p.numel())
            private_tensors += 1
    scales = []
    try:
        for layer in model.deberta.encoder.layer:
            scales.append(float(layer.private_adapter.scale))
    except Exception:
        pass
    ident = {
        "path": rel(path),
        "class": model.__class__.__name__,
        "module": model.__class__.__module__,
        "total_params": int(sum(p.numel() for p in model.parameters())),
        "private_adapter_params": int(private_params),
        "private_adapter_tensor_count": int(private_tensors),
        "executed_private_scales": scales,
    }
    return model, ident


def build_encoded(probe_paths: list[pathlib.Path], slot_modes: list[str], tokenizer, max_length: int, max_items: int) -> tuple[list[dict[str, Any]], dict[str, Any], dict[tuple[str, str], dict[str, Any]], dict[str, dict[str, Any]], list[dict[str, Any]]]:
    encoded_all: list[dict[str, Any]] = []
    encoding_stats: dict[str, Any] = {}
    swap_meta: dict[tuple[str, str], dict[str, Any]] = {}
    packet_meta: dict[str, dict[str, Any]] = {}
    swap_rows_out: list[dict[str, Any]] = []
    for p in probe_paths:
        rows = tun.read_jsonl(p)
        if max_items > 0:
            rows = rows[:max_items]
        probe_set = p.stem.replace("state_margin_probe_", "")
        examples, smeta = tun.build_examples(rows, probe_set, slot_modes)
        enc, estats = tun.encode_examples(tokenizer, examples, max_length)
        encoded_all.extend(enc)
        encoding_stats[probe_set] = estats
        for r in rows:
            packet_meta[f"{probe_set}::{r.get('pair_id')}"] = r
        for sr in smeta["swap_rows"]:
            swap_rows_out.append(sr)
            swap_meta[(probe_set, str(sr["pair_id"]))] = {
                "pair_id": sr.get("swap_pair_id"),
                "new_state": sr.get("swap_new_state"),
                "state_type_new": sr.get("swap_state_type_new"),
                "update_sentence": sr.get("swap_update_sentence"),
            }
    return encoded_all, encoding_stats, swap_meta, packet_meta, swap_rows_out


def compact_transfer_summary(comp_summary: list[dict[str, Any]], raw_summary: list[dict[str, Any]], margin_rows: list[dict[str, Any]]) -> dict[str, Any]:
    rows = []
    for r in comp_summary:
        if r.get("grouping") == "seed+checkpoint+probe_set+slot_mode+packet_type":
            rows.append({
                "probe_set": r.get("probe_set"),
                "slot_mode": r.get("slot_mode"),
                "packet_type": r.get("packet_type"),
                "n_packets": int(r.get("n_packets") or 0),
                "delta_T_update_use": r.get("mean_full_delta_T_update_use"),
                "se_T_update_use": r.get("se_full_delta_T_update_use"),
                "delta_U_retention": r.get("mean_full_delta_U_retention"),
                "se_U_retention": r.get("se_full_delta_U_retention"),
                "delta_N_update_prior": r.get("mean_full_delta_N_update_prior"),
                "delta_T_minus_N_update_use": r.get("mean_full_net_T_minus_N_update_use"),
                "delta_T_retention": r.get("mean_full_delta_T_retention"),
                "delta_N_retention_prior": r.get("mean_full_delta_N_retention_prior"),
                "delta_T_minus_N_retention": r.get("mean_full_net_T_minus_N_retention"),
                "binding_pair_sum_Tupdate_Uretention": r.get("mean_full_binding_pair_sum_Tupdate_Uretention"),
                "binding_pair_sum_Tretention_Uretention": r.get("mean_full_binding_pair_sum_Tretention_Uretention"),
            })
    raw = []
    for r in raw_summary:
        if r.get("grouping") == "seed+arm+checkpoint+probe_set+slot_mode+packet_type+condition":
            if r.get("packet_type") == "UPDATED_USE" and r.get("condition") in {"T", "N", "U"}:
                raw.append({
                    "arm": r.get("arm"),
                    "probe_set": r.get("probe_set"),
                    "slot_mode": r.get("slot_mode"),
                    "packet_type": r.get("packet_type"),
                    "condition": r.get("condition"),
                    "n_packets": int(r.get("n_packets") or 0),
                    "mean_new_minus_source": r.get("mean_full_m_orig_new_minus_source"),
                    "frac_new_over_source": r.get("fraction_positive_full_m_orig_new_minus_source"),
                })
            if r.get("packet_type") == "UNCHANGED_DISTRACTOR_USE" and r.get("condition") in {"T", "N", "U"}:
                raw.append({
                    "arm": r.get("arm"),
                    "probe_set": r.get("probe_set"),
                    "slot_mode": r.get("slot_mode"),
                    "packet_type": r.get("packet_type"),
                    "condition": r.get("condition"),
                    "n_packets": int(r.get("n_packets") or 0),
                    "mean_source_minus_new": r.get("mean_full_m_source_minus_orig_new"),
                    "frac_source_over_new": r.get("fraction_positive_full_m_source_minus_orig_new"),
                })
    # A very direct readout: for updated packets, does the relation-acquired checkpoint increase T more than N?
    updated_t_minus_n = [r.get("delta_T_minus_N_update_use") for r in rows if r.get("packet_type") == "UPDATED_USE"]
    return {
        "interpretation_boundary": "Paired deltas are the relation-acquired checkpoint minus coherent86 parent on nontraining packets. Template slot uses complete candidate phrases in the same explicit frame, avoiding research's partial-answer-vs-full-foil issue. Rows are not same-source recipient-only minimal pairs, so this is a portability readout rather than proof of binding in the transfer format.",
        "compact_rows": rows,
        "raw_rows_selected": raw,
        "mean_delta_T_minus_N_update_use_over_probe_sets": finite_mean(updated_t_minus_n),
        "fraction_probe_rows_positive_delta_T_minus_N_update_use": pct_pos(updated_t_minus_n),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT))
    ap.add_argument("--probe", action="append", default=[], help="Probe JSONL; defaults to balanced heldout.")
    ap.add_argument("--include-extended", action="store_true", help="Also score extended nontrain set.")
    ap.add_argument("--slot-modes", nargs="+", default=["template"], choices=["template", "use"])
    ap.add_argument("--max-items", type=int, default=0)
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--parent", default=str(PARENT))
    ap.add_argument("--acquired", default=str(A01_SEED40040))
    args = ap.parse_args()

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    probe_paths = [pathlib.Path(p) for p in args.probe] if args.probe else [BALANCED]
    if args.include_extended and EXTENDED not in probe_paths:
        probe_paths.append(EXTENDED)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(str(pathlib.Path(args.parent)), local_files_only=True, use_fast=True)
    encoded_all, encoding_stats, swap_meta, packet_meta, swap_rows = build_encoded(probe_paths, args.slot_modes, tokenizer, int(args.max_length), int(args.max_items))
    if not encoded_all:
        raise RuntimeError("No encoded records; cannot score transfer")
    write_csv(out_dir / "swap_assignments.csv", swap_rows)
    (out_dir / "encoding_metadata.json").write_text(json.dumps({
        "status": "A02_TUN_TRANSFER_ENCODING_READY",
        "created_utc": now_utc(),
        "probe_paths": [rel(p) for p in probe_paths],
        "slot_modes": args.slot_modes,
        "max_items": int(args.max_items),
        "encoded_candidate_records": len(encoded_all),
        "encoding_stats": encoding_stats,
        "candidate_records_expected_per_packet_per_slot_mode": 9,
    }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    model_specs = [
        {"model_name": "coherent86_parent", "arm": "base", "path": pathlib.Path(args.parent)},
        {"model_name": "a01_relation_acquired_seed40040", "arm": "intervention", "path": pathlib.Path(args.acquired)},
    ]
    all_score_rows: list[dict[str, Any]] = []
    identities: list[dict[str, Any]] = []
    t0 = time.time()
    for i, spec in enumerate(model_specs, start=1):
        print(json.dumps({"event": "load", "i": i, "n": len(model_specs), "model": spec["model_name"], "path": rel(spec["path"])}), flush=True)
        model, ident = load_model_identity(spec["path"], device)
        identities.append({"model_name": spec["model_name"], "arm": spec["arm"], **ident})
        rows = tun.score_encoded(model, encoded_all, tokenizer, device, int(args.batch_size))
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        for r in rows:
            pkt = packet_meta.get(f"{r['probe_set']}::{r['packet_id']}", {})
            sw = swap_meta.get((r["probe_set"], str(r["packet_id"])), {})
            r.update({
                "model_name": spec["model_name"],
                "seed": "a01_transfer",
                "arm": spec["arm"],
                "checkpoint": "relation80_seed40040_vs_parent",
                "model_path": rel(spec["path"]),
                "gold_state_type": pkt.get("gold_state_type"),
                "competitor_state_type": pkt.get("competitor_state_type"),
                "state_type_source": pkt.get("state_type_source"),
                "state_type_new": pkt.get("state_type_new"),
                "updated_conflict_heuristic": pkt.get("updated_conflict_heuristic"),
                "source_state": pkt.get("source_state"),
                "original_new_state": pkt.get("new_state"),
                "target_entity": pkt.get("target_entity"),
                "swapped_new_state": sw.get("new_state"),
                "swap_pair_id": sw.get("pair_id"),
                "swap_state_type_new": sw.get("state_type_new"),
            })
        all_score_rows.extend(rows)
        print(json.dumps({"event": "scored", "model": spec["model_name"], "candidate_records": len(rows), "class": ident["class"], "private_params": ident["private_adapter_params"]}), flush=True)

    score_path = out_dir / "candidate_phrase_score_rows.csv"
    write_csv(score_path, all_score_rows)
    margin_rows = tun.pivot_margins(all_score_rows)
    write_csv(out_dir / "raw_TUN_margin_rows.csv", margin_rows)
    raw_summary = tun.summarize_margins(margin_rows, out_dir)
    delta_rows, delta_summary, comp_summary = tun.paired_arm_deltas(margin_rows, out_dir)
    transfer = compact_transfer_summary(comp_summary, raw_summary, margin_rows)
    meta = {
        "status": "A02_TUN_TRANSFER_DONE",
        "created_utc": now_utc(),
        "elapsed_sec": round(time.time() - t0, 2),
        "scientific_question": "Does the exact relation-first acquired checkpoint move state-use T/U/N margins before any packet training?",
        "probe_paths": [rel(p) for p in probe_paths],
        "slot_modes": args.slot_modes,
        "encoded_candidate_records": len(encoded_all),
        "model_identities": identities,
        "transfer_summary": transfer,
        "outputs": {
            "encoding_metadata": rel(out_dir / "encoding_metadata.json"),
            "candidate_phrase_score_rows": rel(score_path),
            "raw_TUN_margin_rows": rel(out_dir / "raw_TUN_margin_rows.csv"),
            "raw_TUN_margin_summary": rel(out_dir / "raw_TUN_margin_summary.csv"),
            "paired_arm_delta_rows": rel(out_dir / "paired_arm_delta_rows.csv"),
            "paired_arm_delta_summary": rel(out_dir / "paired_arm_delta_summary.csv"),
            "relation_composite_delta_rows": rel(out_dir / "relation_composite_delta_rows.csv"),
            "relation_composite_delta_summary": rel(out_dir / "relation_composite_delta_summary.csv"),
            "transfer_summary": rel(out_dir / "transfer_summary.json"),
        },
    }
    (out_dir / "transfer_summary.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": meta["status"], "out": meta["outputs"]["transfer_summary"], "elapsed_sec": meta["elapsed_sec"], "compact": transfer["compact_rows"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
