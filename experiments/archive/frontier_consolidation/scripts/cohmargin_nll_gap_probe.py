#!/usr/bin/env python3
"""research coherent-vs-disrupted NLL-gap probe.

Purpose: after the research coherence-margin pilot, measure whether a model actually
assigns lower masked-target NLL to the coherent legal-row context than to the
same row with unmasked context blocks shuffled.  This is a mechanism probe for
the training-time signal, not an official benchmark evaluator.

The probe uses only legal stream rows, a fixed WWM mask seed, and the same
block-shuffle construction as the trainer.  It can be applied unchanged to the
coherence-margin model, a lambda-zero same-charge arm, and standard references.
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
import os
import random
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

HERE = _public_path('experiments/archive/frontier_consolidation/scripts')
USER_ROOT = _public_path('.')
STUDY = _public_path('experiments/archive/frontier_consolidation')
WORKSPACE = _public_path('experiments/archive/frontier_consolidation')
TRAINER_PATH = _public_path('experiments/archive/frontier_consolidation/scripts/coherence_margin_trainer.py')
DEFAULT_STREAM = _public_path('experiments/archive/frontier_consolidation/data/density_cleanqwen_overlay_medium_riskhard/cleanqwen_fineweb_compact_view_reinvest_100M.jsonl')
DEFAULT_TOKENIZER = _public_path('experiments/archive/frontier_consolidation/data/compliant_tokenizer')


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def rel(path: Path | str) -> str:
    p = Path(path)
    try:
        return str(p.resolve().relative_to(USER_ROOT))
    except Exception:
        return str(p)


def read_trainer():
    spec = importlib.util.spec_from_file_location("cohmargin_trainer", TRAINER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load trainer module: {TRAINER_PATH}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


TRAINER = read_trainer()
BASE = TRAINER.BASE


def load_rows(path: Path, *, row_offset: int, skip_coherent_words: int | None, num_rows: int) -> tuple[list[Any], dict[str, Any]]:
    examples: list[Any] = []
    rows_seen = 0
    words_seen = 0
    rows_skipped_by_words = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            rows_seen += 1
            text = str(obj["text"])
            words = int(obj.get("words", len(text.split())))
            actual = len(text.split())
            if actual != words:
                raise RuntimeError({"word_mismatch_row": rows_seen, "field": words, "actual": actual})
            words_seen += words
            if rows_seen <= row_offset:
                continue
            if skip_coherent_words is not None and words_seen <= skip_coherent_words:
                rows_skipped_by_words += 1
                continue
            ex_id = int(obj.get("example_id", rows_seen - 1))
            source = str(obj.get("source", "example_jsonl"))
            examples.append(BASE.Example(text=text, words=words, example_id=ex_id, source=source))
            if len(examples) >= num_rows:
                break
    meta = {
        "rows_seen_until_stop": rows_seen,
        "words_seen_until_stop": words_seen,
        "row_offset": row_offset,
        "skip_coherent_words": skip_coherent_words,
        "rows_skipped_by_words": rows_skipped_by_words,
        "num_rows_requested": num_rows,
        "num_rows_loaded": len(examples),
        "loaded_words": int(sum(int(getattr(x, "words")) for x in examples)),
        "loaded_example_ids_first10": [int(getattr(x, "example_id")) for x in examples[:10]],
        "loaded_sources_first10": [str(getattr(x, "source")) for x in examples[:10]],
    }
    if not examples:
        raise RuntimeError({"no_examples_loaded": meta})
    return examples, meta


def load_model(model_path: Path, out_dir: Path, device: torch.device):
    cache = out_dir / "cache"
    hf_home = cache / "hf_home"
    hf_mod = cache / "hf_modules"
    hf_home.mkdir(parents=True, exist_ok=True)
    hf_mod.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(hf_home))
    os.environ.setdefault("HF_MODULES_CACHE", str(hf_mod))
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    from transformers import AutoModelForMaskedLM  # imported after writable cache env is set

    model = AutoModelForMaskedLM.from_pretrained(str(model_path), trust_remote_code=True, local_files_only=True)
    model.to(device)
    model.eval()
    return model


def safe_mean(xs: list[float]) -> float | None:
    return float(sum(xs) / len(xs)) if xs else None


def score_model(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    random.seed(args.probe_seed)
    torch.manual_seed(args.probe_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.probe_seed)

    if args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available")
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    tokenizer = BASE.make_portable_tokenizer(args.tokenizer_path)
    model = load_model(Path(args.model_path), out_dir, device)

    examples, row_meta = load_rows(
        Path(args.example_jsonl),
        row_offset=args.row_offset,
        skip_coherent_words=args.skip_coherent_words if args.skip_coherent_words >= 0 else None,
        num_rows=args.num_rows,
    )
    dataset = BASE.MaskedChunkDataset(examples, tokenizer, args.max_seq_length)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=BASE.collate,
                        num_workers=0, pin_memory=(device.type == "cuda"))
    curriculum_state = BASE.MaskingCurriculumState(curriculum="wwm_fixed", mask_prob_start=args.mask_prob, mask_prob_end=args.mask_prob)
    curriculum_state.initialize(vocab_size=len(tokenizer), total_steps=max(1, len(loader)))
    gen = torch.Generator(device=device)
    gen.manual_seed(args.probe_seed)

    token_gaps: list[float] = []
    token_coh: list[float] = []
    token_bad: list[float] = []
    batch_records: list[dict[str, Any]] = []
    smoke_totals = {
        "target_mismatch_tokens": 0,
        "special_or_pad_mismatch_tokens": 0,
        "mask_position_mismatch_tokens": 0,
        "rows_with_context_multiset_mismatch": 0,
        "moved_context_tokens_mean_values": [],
        "moved_context_tokens_min": None,
        "moved_context_tokens_max": None,
    }
    t0 = time.time()
    with torch.no_grad():
        for bi, batch in enumerate(loader, 1):
            input_ids = batch["input_ids"].to(device)[:, :args.seq_length].contiguous()
            attn = batch["attention_mask"].to(device)[:, :args.seq_length].contiguous()
            word_group = batch["word_group"].to(device)[:, :args.seq_length].contiguous()
            curriculum_state.current_step = bi - 1
            masked_inputs, labels = BASE.apply_masking_curriculum(input_ids, attn, word_group, tokenizer, curriculum_state, gen)
            bad_inputs = TRAINER.make_disrupted_context(masked_inputs, labels, attn, tokenizer, gen, args.disrupt_span_tokens)
            smoke = TRAINER.verify_disruption({"attention_mask": attn}, masked_inputs, labels, bad_inputs, tokenizer)
            smoke_totals["target_mismatch_tokens"] += int(smoke["target_mismatch_tokens"])
            smoke_totals["special_or_pad_mismatch_tokens"] += int(smoke["special_or_pad_mismatch_tokens"])
            smoke_totals["mask_position_mismatch_tokens"] += int(smoke["mask_position_mismatch_tokens"])
            smoke_totals["rows_with_context_multiset_mismatch"] += int(smoke["rows_with_context_multiset_mismatch"])
            smoke_totals["moved_context_tokens_mean_values"].append(float(smoke["moved_context_tokens_mean"]))
            mn = int(smoke["moved_context_tokens_min"])
            mx = int(smoke["moved_context_tokens_max"])
            smoke_totals["moved_context_tokens_min"] = mn if smoke_totals["moved_context_tokens_min"] is None else min(int(smoke_totals["moved_context_tokens_min"]), mn)
            smoke_totals["moved_context_tokens_max"] = mx if smoke_totals["moved_context_tokens_max"] is None else max(int(smoke_totals["moved_context_tokens_max"]), mx)
            if smoke["target_mismatch_tokens"] or smoke["special_or_pad_mismatch_tokens"] or smoke["rows_with_context_multiset_mismatch"]:
                raise RuntimeError({"bad_disruption_in_probe_batch": bi, "smoke": smoke})

            coh = model(input_ids=masked_inputs, attention_mask=attn)
            bad = model(input_ids=bad_inputs, attention_mask=attn)
            coh_ce = TRAINER.masked_ce_vec(coh.logits, labels).detach().float().cpu()
            bad_ce = TRAINER.masked_ce_vec(bad.logits, labels).detach().float().cpu()
            if coh_ce.numel() != bad_ce.numel():
                raise RuntimeError({"ce_length_mismatch": [int(coh_ce.numel()), int(bad_ce.numel())]})
            gaps = bad_ce - coh_ce
            token_gaps.extend(float(x) for x in gaps.tolist())
            token_coh.extend(float(x) for x in coh_ce.tolist())
            token_bad.extend(float(x) for x in bad_ce.tolist())
            batch_records.append({
                "batch_index": bi,
                "n_masked_tokens": int(gaps.numel()),
                "coh_nll_mean": float(coh_ce.mean().item()) if coh_ce.numel() else None,
                "bad_nll_mean": float(bad_ce.mean().item()) if bad_ce.numel() else None,
                "bad_minus_coh_nll_mean": float(gaps.mean().item()) if gaps.numel() else None,
                "gap_positive_fraction": float((gaps > 0).float().mean().item()) if gaps.numel() else None,
                "moved_context_tokens_mean": float(smoke["moved_context_tokens_mean"]),
            })
            print(json.dumps({"event": "probe_batch", **batch_records[-1]}), flush=True)

    n = len(token_gaps)
    gap_mean = safe_mean(token_gaps)
    gap_std = float(statistics.pstdev(token_gaps)) if len(token_gaps) > 1 else 0.0
    se = gap_std / math.sqrt(max(1, n))
    out = {
        "status": "COHMARGIN_NLL_GAP_PROBE_COMPLETE",
        "created_utc": now(),
        "label": args.label,
        "model_path": rel(args.model_path),
        "example_jsonl": rel(args.example_jsonl),
        "tokenizer_path": rel(args.tokenizer_path),
        "device": str(device),
        "gpu": args.gpu if args.device == "cuda" else None,
        "probe_seed": args.probe_seed,
        "row_meta": row_meta,
        "seq_length": args.seq_length,
        "max_seq_length": args.max_seq_length,
        "mask_prob": args.mask_prob,
        "disrupt_span_tokens": args.disrupt_span_tokens,
        "batch_size": args.batch_size,
        "n_batches": len(batch_records),
        "n_masked_tokens": n,
        "coherent_nll_mean": safe_mean(token_coh),
        "disrupted_nll_mean": safe_mean(token_bad),
        "bad_minus_coh_nll_mean": gap_mean,
        "bad_minus_coh_nll_std_population": gap_std,
        "bad_minus_coh_nll_se": se,
        "bad_minus_coh_nll_z_vs_zero": (gap_mean / se) if se > 0 and gap_mean is not None else None,
        "gap_positive_fraction": safe_mean([1.0 if x > 0 else 0.0 for x in token_gaps]),
        "smoke_totals": {
            **{k: v for k, v in smoke_totals.items() if k != "moved_context_tokens_mean_values"},
            "moved_context_tokens_mean": safe_mean([float(x) for x in smoke_totals["moved_context_tokens_mean_values"]]),
        },
        "batch_records": batch_records,
        "elapsed_sec": round(time.time() - t0, 3),
        "scientific_reading": "Positive bad_minus_coh_nll means the model assigns lower NLL to coherent context than to block-shuffled context for the same masked targets. Compare lambda>0 and lambda=0 same-charge arms before interpreting official-score movement as caused by the margin signal.",
    }
    out_json = out_dir / "cohmargin_nll_gap_probe.json"
    out_md = out_dir / "cohmargin_nll_gap_probe.md"
    out_json.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = [
        f"# research coherence-margin NLL gap probe — {args.label}",
        "",
        f"Status: **{out['status']}**",
        "",
        f"Model: `{out['model_path']}`",
        f"Rows loaded: {row_meta['num_rows_loaded']} (words {row_meta['loaded_words']}, row_offset {row_meta['row_offset']}, skip_coherent_words {row_meta['skip_coherent_words']})",
        f"Masked tokens: {n}",
        f"Coherent NLL mean: {out['coherent_nll_mean']:.6f}" if out['coherent_nll_mean'] is not None else "Coherent NLL mean: NA",
        f"Disrupted NLL mean: {out['disrupted_nll_mean']:.6f}" if out['disrupted_nll_mean'] is not None else "Disrupted NLL mean: NA",
        f"Bad-minus-coherent NLL mean: {out['bad_minus_coh_nll_mean']:.6f} (SE {out['bad_minus_coh_nll_se']:.6f}, z {out['bad_minus_coh_nll_z_vs_zero']:.3f})" if out['bad_minus_coh_nll_mean'] is not None else "Bad-minus-coherent NLL mean: NA",
        f"Gap positive fraction: {out['gap_positive_fraction']:.4f}" if out['gap_positive_fraction'] is not None else "Gap positive fraction: NA",
        "",
        "Positive gap means coherent context is preferred for the same masked targets. This file is a mechanism probe, not official evaluation.",
        "",
        f"JSON: `{rel(out_json)}`",
    ]
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": out["status"], "label": args.label, "out_json": rel(out_json), "out_md": rel(out_md), "bad_minus_coh_nll_mean": gap_mean, "n_masked_tokens": n}, indent=2), flush=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--example-jsonl", default=str(DEFAULT_STREAM))
    ap.add_argument("--tokenizer-path", default=str(DEFAULT_TOKENIZER))
    ap.add_argument("--num-rows", type=int, default=64)
    ap.add_argument("--row-offset", type=int, default=0)
    ap.add_argument("--skip-coherent-words", type=int, default=-1, help="If >=0, start after cumulative stream words exceed this value.")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--seq-length", type=int, default=256)
    ap.add_argument("--max-seq-length", type=int, default=256)
    ap.add_argument("--mask-prob", type=float, default=0.15)
    ap.add_argument("--disrupt-span-tokens", type=int, default=8)
    ap.add_argument("--probe-seed", type=int, default=15900)
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    ap.add_argument("--gpu", type=int, default=0)
    args = ap.parse_args()
    score_model(args)


if __name__ == "__main__":
    main()
