#!/usr/bin/env python3
"""research official-like cheap7 scorer with fixed Reading subword recursion.

research's arbitrary-checkpoint cheap7 wrapper reused research's local scorer.  Its
Reading helper accidentally called the multi-token scorer recursively for each
subtoken, whereas the official BabyLM Reading implementation calls the single-token
MLM scorer for later subtokens.  Some decoded subtokens re-tokenize to multiple
pieces, causing infinite recursion on dense64_scale0p60.  This wrapper imports the
same research scorer, monkey-patches the Reading probability function to mirror the
official get_p2_mlm/get_p_mlm structure, and otherwise leaves the zero-shot and
Reading surfaces unchanged.

Use --only-reading to complete a prior partial cheap7 run without rescoring the
expensive zero-shot columns.  Use normal mode for newly repaired ladder checkpoints.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import importlib.util
import json
import math
import pathlib
import sys
import time
from typing import Any

import torch

ROOT = _public_path('.')
PATH = _public_path('experiments/archive/relation_learning/scripts/no_boundary_cheap7_diagnostic.py')
DEFAULT_OUT = _public_path('experiments/archive/relation_learning/data/custom_cheap7_fixed')


def rel(path: pathlib.Path | str) -> str:
    try:
        return str(pathlib.Path(path).resolve().relative_to(ROOT))
    except Exception:
        return str(path)


def import_step105(out_dir: pathlib.Path):
    # research sets writable HF caches from sys.argv before importing Transformers.
    old_argv = list(sys.argv)
    sys.argv = [str(PATH), "--out-dir", str(out_dir)]
    try:
        spec = importlib.util.spec_from_file_location("no_boundary_cheap7_for_step126", PATH)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"cannot import {PATH}")
        mod = importlib.util.module_from_spec(spec)
        sys.modules["no_boundary_cheap7_for_step126"] = mod
        spec.loader.exec_module(mod)
        return mod
    finally:
        sys.argv = old_argv


def patch_reading_probability(mod) -> None:
    """Patch research's Reading scorer to the official non-recursive subword form."""

    def _score_first_id(sentence: str, token_text: str, model, tokenizer, device: torch.device,
                        add_special_tokens: bool, num_mask_tokens: int = 3) -> float:
        text = "".join([sentence, "".join([tokenizer.mask_token for _ in range(num_mask_tokens)])])
        inpts = tokenizer(text, return_tensors="pt", add_special_tokens=add_special_tokens).to(device)
        if int(inpts.input_ids[:, -1].item()) == int(tokenizer.mask_token_id):
            position = -num_mask_tokens
        else:
            position = -(num_mask_tokens + 1)
        with torch.no_grad():
            outputs = model(**inpts)
            logits = mod.get_logits(outputs)[:, position, :].cpu()
        ids = tokenizer(token_text, add_special_tokens=False)["input_ids"]
        if not ids:
            return float("nan")
        return float(torch.softmax(logits[0], dim=-1)[int(ids[0])].item())

    def p_mlm_next_fixed(sentence: str, word: str, model, tokenizer, device: torch.device,
                         add_special_tokens: bool, num_mask_tokens: int = 3) -> tuple[float, int]:
        target = tokenizer(word, add_special_tokens=False)["input_ids"]
        if not target:
            return float("nan"), 0
        out_p: list[float] = []
        first_id = int(target[0])
        # First piece: probability of the first target id in the original word.
        text = "".join([sentence, "".join([tokenizer.mask_token for _ in range(num_mask_tokens)])])
        inpts = tokenizer(text, return_tensors="pt", add_special_tokens=add_special_tokens).to(device)
        if int(inpts.input_ids[:, -1].item()) == int(tokenizer.mask_token_id):
            position = -num_mask_tokens
        else:
            position = -(num_mask_tokens + 1)
        with torch.no_grad():
            outputs = model(**inpts)
            logits = mod.get_logits(outputs)[:, position, :].cpu()
        out_p.append(float(torch.softmax(logits[0], dim=-1)[first_id].item()))
        if len(target) == 1:
            return out_p[0], 0
        next_sentence = sentence + tokenizer.decode(first_id)
        # Later pieces: mirror official get_p2_mlm, which calls get_p_mlm rather than
        # recursively calling get_p2_mlm.  This avoids infinite recursion when a decoded
        # subtoken re-tokenizes into multiple ids.
        for tok in target[1:]:
            t = tokenizer.decode(int(tok))
            out_p.append(_score_first_id(next_sentence, t, model, tokenizer, device, add_special_tokens, num_mask_tokens))
            next_sentence = next_sentence + t
        prod = 1.0
        for p in out_p:
            if not math.isfinite(p):
                return float("nan"), 1
            prod *= float(p)
        return float(prod), 1

    mod.p_mlm_next = p_mlm_next_fixed


def run_only_reading(mod, label: str, ckpt: pathlib.Path, out_dir: pathlib.Path, input_forms: list[str],
                     batch_size: int, non_causal_batch_size: int, max_reading_rows: int,
                     cpu: bool) -> dict[str, Any]:
    # batch args are unused here but kept in the payload for comparability.
    device = torch.device("cuda" if torch.cuda.is_available() and not cpu else "cpu")
    tokenizer = mod.load_tokenizer(ckpt)
    mod.ensure_pad(tokenizer)
    model = mod.AutoModelForMaskedLM.from_pretrained(str(ckpt), trust_remote_code=True)
    model.to(device)
    model.eval()
    ident = mod.model_identity(model, ckpt)
    endpoint_out = out_dir / label
    endpoint_out.mkdir(parents=True, exist_ok=True)
    (endpoint_out / "model_identity.json").write_text(json.dumps(ident, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    readings: dict[str, Any] = {}
    for form in input_forms:
        add_special = form == "with_special"
        form_out = endpoint_out / form
        form_out.mkdir(parents=True, exist_ok=True)
        rrec = mod.score_reading(model, tokenizer, add_special, device, form_out / "Reading", max_rows=max_reading_rows)
        readings[form] = rrec
        print(json.dumps({"event": "column_done", "endpoint": label, "form": form, "column": "Reading", "score": rrec["score"], "elapsed_sec": rrec["elapsed_sec"]}), flush=True)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    payload = {
        "status": "READING_FIXED_ONLY_DONE",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "label": label,
        "checkpoint": rel(ckpt),
        "input_forms": input_forms,
        "model_identity": ident,
        "readings": readings,
        "batch_size": batch_size,
        "non_causal_batch_size": non_causal_batch_size,
        "max_reading_rows": max_reading_rows,
        "note": "Reading scored with official get_p2_mlm-style non-recursive subword continuation; zero-shot columns, if any, are from a separate run.",
    }
    (out_dir / f"{label}_reading_fixed_payload.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--label", required=True)
    ap.add_argument("--checkpoint", type=pathlib.Path, required=True)
    ap.add_argument("--out-dir", type=pathlib.Path, default=DEFAULT_OUT)
    ap.add_argument("--input-forms", nargs="*", default=["with_special"], choices=["with_special", "no_special"])
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--non-causal-batch-size", type=int, default=128)
    ap.add_argument("--max-items-per-file", type=int, default=0)
    ap.add_argument("--max-reading-rows", type=int, default=0)
    ap.add_argument("--skip-reading", action="store_true")
    ap.add_argument("--only-reading", action="store_true")
    ap.add_argument("--no-predictions", action="store_true")
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    out_dir = args.out_dir if args.out_dir.is_absolute() else ROOT / args.out_dir
    ckpt = args.checkpoint if args.checkpoint.is_absolute() else ROOT / args.checkpoint
    if not (ckpt / "config.json").is_file():
        raise FileNotFoundError(f"checkpoint config not found: {ckpt / 'config.json'}")
    out_dir.mkdir(parents=True, exist_ok=True)

    mod = import_step105(out_dir)
    patch_reading_probability(mod)
    mod.setup_cache(out_dir)
    torch.manual_seed(0)

    if args.only_reading:
        payload = run_only_reading(mod, args.label, ckpt, out_dir, list(args.input_forms), int(args.batch_size), int(args.non_causal_batch_size), int(args.max_reading_rows), bool(args.cpu))
        print(json.dumps({"status": payload["status"], "label": args.label, "out_json": rel(out_dir / f"{args.label}_reading_fixed_payload.json")}, indent=2, ensure_ascii=False), flush=True)
        return

    mod.ENDPOINTS = {args.label: {"model_path": ckpt, "known_official_scores": {}}}
    ns = argparse.Namespace(
        out_dir=str(out_dir),
        endpoints=[args.label],
        input_forms=list(args.input_forms),
        batch_size=int(args.batch_size),
        non_causal_batch_size=int(args.non_causal_batch_size),
        max_items_per_file=int(args.max_items_per_file),
        max_reading_rows=int(args.max_reading_rows),
        skip_reading=bool(args.skip_reading),
        no_predictions=bool(args.no_predictions),
        cpu=bool(args.cpu),
    )
    result = mod.evaluate_endpoint(args.label, list(args.input_forms), ns)
    summary = mod.build_overall_summary({args.label: result}, out_dir)
    payload: dict[str, Any] = {
        "status": "OFFICIAL_LIKE_CHEAP7_CUSTOM_FIXED_DONE",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "script": rel(_public_path('experiments/archive/relation_learning/scripts/official_like_cheap7_custom_fixed.py')),
        "label": args.label,
        "checkpoint": rel(ckpt),
        "input_forms": list(args.input_forms),
        "summary_json": rel(out_dir / "summary.json"),
        "summary_md": rel(out_dir / "summary.md"),
        "score_rows_csv": rel(out_dir / "score_rows.csv"),
        "model_identity": result.get("model_identity"),
        "scores": {form: data.get("scores") for form, data in result.get("forms", {}).items()},
        "cheap7": {form: data.get("cheap7") for form, data in result.get("forms", {}).items()},
        "note": "with_special is official-like local masked-LM/Reading; Reading uses the research non-recursive subword fix matching official get_p2_mlm structure.",
    }
    (out_dir / "custom_cheap7_fixed_payload.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "payload": payload, "summary_status": summary.get("status")}, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
