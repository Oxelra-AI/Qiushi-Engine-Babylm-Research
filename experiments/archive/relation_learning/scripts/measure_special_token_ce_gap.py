#!/usr/bin/env python3
"""research: measure whether research format screen is confounded by special-token exposure.

The corrected research trainer tokenizes rows with add_special_tokens=True to match the
sentence_zero_shot/official evaluator, but the v4 pretraining streams and coherent86
private replay used add_special_tokens=False. This script measures the frozen trunk's
masked-MLM CE on the same coherent leash/readout rows and the same first isolated
macro-batch with and without [CLS]/[SEP] under the research WWM grouping.

It does not train. It writes a JSON/Markdown record that can decide whether the
running isolated/half-format arms should be interpreted as isolation-format training
or as a mixed isolation + special-token familiarization screen.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import json
import math
import os
import pathlib
import random
import sys
import time
from typing import Any

import torch
import torch.nn.functional as F

SCRIPT = _public_path('experiments/archive/relation_learning/scripts/measure_special_token_ce_gap.py')
ROOT = _public_path('.')

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# Reuse the corrected loader and word-start convention without executing its trainer.
sys.path.insert(0, str(_public_path('experiments/archive/relation_learning/scripts')))
from word_paced_format_replay_trainer import (  # noqa: E402
    CHCK82,
    WORDS_PER_UPDATE,
    WordGrouper,
    apply_wwm,
    collate_dynamic,
    load_frozen_model,
    load_jsonl_rows,
    set_private_enabled,
    set_seed,
)


def rel(p: pathlib.Path | str) -> str:
    try:
        return str(pathlib.Path(p).resolve().relative_to(ROOT))
    except Exception:
        return str(p)


def load_rows_for_words(path: pathlib.Path, target_words: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    total = 0
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            text = str(obj["text"])
            words = int(obj.get("words", len(text.split())))
            if total + words > target_words and rows:
                break
            rows.append({"text": text, "words": words, "source": str(obj.get("source", ""))})
            total += words
            if total >= target_words:
                break
    return rows


def tokenized_stats(tokenizer, texts: list[str], max_length: int, add_special_tokens: bool) -> dict[str, Any]:
    full_lengths = []
    trunc_lengths = []
    for text in texts:
        enc_full = tokenizer(text, add_special_tokens=add_special_tokens, truncation=False)
        enc_trunc = tokenizer(text, add_special_tokens=add_special_tokens, truncation=True, max_length=max_length)
        full_lengths.append(len(enc_full["input_ids"]))
        trunc_lengths.append(len(enc_trunc["input_ids"]))
    n = len(texts)
    trunc_rows = sum(1 for a, b in zip(full_lengths, trunc_lengths) if a > b)
    lost = sum(max(0, a - b) for a, b in zip(full_lengths, trunc_lengths))
    return {
        "rows": n,
        "mean_full_len": sum(full_lengths) / max(1, n),
        "mean_trunc_len": sum(trunc_lengths) / max(1, n),
        "max_full_len": max(full_lengths) if full_lengths else 0,
        "max_trunc_len": max(trunc_lengths) if trunc_lengths else 0,
        "truncated_rows": trunc_rows,
        "truncated_row_fraction": trunc_rows / max(1, n),
        "lost_tokens_to_truncation": lost,
    }


def make_tokenized(rows: list[dict[str, Any]], grouper: WordGrouper, max_length: int, add_special_tokens: bool) -> list[dict[str, Any]]:
    toks: list[dict[str, Any]] = []
    for r in rows:
        ids, wg = grouper.tokenize(str(r["text"]), max_length=max_length, add_special_tokens=add_special_tokens)
        toks.append({"ids": ids, "wg": wg, "words": int(r["words"])})
    return toks


def batched_ce(model, tokenizer, tokenized: list[dict[str, Any]], device: torch.device,
               *, mask_seed: int, batch_size: int, mask_prob: float, label: str) -> dict[str, Any]:
    pad_id = int(tokenizer.pad_token_id)
    gen = torch.Generator(device=device)
    gen.manual_seed(int(mask_seed))
    set_private_enabled(model, False)
    model.eval()
    total_loss = 0.0
    total_targets = 0
    total_non_special = 0
    total_positions = 0
    n_batches = math.ceil(len(tokenized) / batch_size)
    t0 = time.time()
    with torch.no_grad():
        for bi in range(n_batches):
            batch_toks = tokenized[bi * batch_size:(bi + 1) * batch_size]
            batch = collate_dynamic(batch_toks, pad_id)
            ids = batch["input_ids"].to(device)
            att = batch["attention_mask"].to(device)
            wg = batch["word_group"].to(device)
            masked, labels, _select, candidate = apply_wwm(ids, att, wg, tokenizer, float(mask_prob), gen)
            out = model(input_ids=masked, attention_mask=att)
            vocab = out.logits.shape[-1]
            loss = F.cross_entropy(out.logits.reshape(-1, vocab), labels.reshape(-1), ignore_index=-100, reduction="sum")
            n_targets = int((labels != -100).sum().item())
            total_loss += float(loss.detach().cpu())
            total_targets += n_targets
            total_non_special += int(candidate.sum().item())
            total_positions += int(att.sum().item())
    ce = total_loss / max(1, total_targets)
    return {
        "label": label,
        "rows": len(tokenized),
        "tokens_with_attention": total_positions,
        "non_special_tokens": total_non_special,
        "targets": total_targets,
        "target_ratio": total_targets / max(1, total_non_special),
        "masked_ce": ce,
        "elapsed_sec": round(time.time() - t0, 3),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="experiments/archive/relation_learning/data/special_token_ce_gap")
    ap.add_argument("--device", default="cpu", help="cpu or cuda:N; default avoids interfering with running H100 tasks")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--mask-seed", type=int, default=99099)
    ap.add_argument("--mask-prob", type=float, default=0.15)
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--isolated-words", type=int, default=39553)
    args = ap.parse_args()

    out_dir = pathlib.Path(args.out)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    cache = out_dir / "hf_cache"
    (cache / "modules").mkdir(parents=True, exist_ok=True)
    os.environ["TRANSFORMERS_CACHE"] = str(cache)
    os.environ["HF_HOME"] = str(cache)
    os.environ["HF_MODULES_CACHE"] = str(cache / "modules")

    if args.device.startswith("cuda") and torch.cuda.is_available():
        device = torch.device(args.device)
    else:
        device = torch.device("cpu")
    set_seed(99099)
    t0 = time.time()
    model, tokenizer, total_params, trainable_params = load_frozen_model(pathlib.Path(CHCK82), private_scale=1.0, device=device)
    # This is a trunk measurement, not a private-adapter run.
    set_private_enabled(model, False)
    model.eval()
    grouper = WordGrouper(tokenizer)

    leash_path = _public_path('experiments/archive/relation_learning/data/held_coherent_sets/held_coherent_leash_256rows.jsonl')
    readout_path = _public_path('experiments/archive/relation_learning/data/held_coherent_sets/held_coherent_readout_256rows.jsonl')
    isolated_path = _public_path('experiments/archive/relation_learning/data/isolated_replay_streams/isolated_all_replay_3992800w.jsonl')

    datasets = {
        "coherent_leash": load_jsonl_rows(leash_path, max_words=0),
        "coherent_readout": load_jsonl_rows(readout_path, max_words=0),
        "isolated_first_macro": load_rows_for_words(isolated_path, int(args.isolated_words)),
    }

    results: dict[str, Any] = {
        "status": "SPECIAL_TOKEN_CE_GAP_DONE",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "purpose": "Measure frozen-trunk masked-MLM CE gap from add_special_tokens=True vs False on research coherent and isolated rows.",
        "model": rel(CHCK82),
        "device": str(device),
        "total_params": total_params,
        "trainable_params_loaded_but_disabled_for_measurement": trainable_params,
        "mask_seed": args.mask_seed,
        "mask_prob": args.mask_prob,
        "batch_size": args.batch_size,
        "max_length": args.max_length,
        "datasets": {},
    }

    for name, rows in datasets.items():
        texts = [str(r["text"]) for r in rows]
        words = sum(int(r["words"]) for r in rows)
        results["datasets"][name] = {
            "rows": len(rows),
            "words": words,
            "source_path": rel(leash_path if name == "coherent_leash" else readout_path if name == "coherent_readout" else isolated_path),
            "conditions": {},
            "gap_special_minus_no_special": {},
        }
        per_cond = {}
        for add_special in [False, True]:
            cond = "with_special_tokens" if add_special else "without_special_tokens"
            toks = make_tokenized(rows, grouper, args.max_length, add_special)
            stat = tokenized_stats(tokenizer, texts, args.max_length, add_special)
            ce = batched_ce(model, tokenizer, toks, device, mask_seed=args.mask_seed, batch_size=args.batch_size,
                            mask_prob=args.mask_prob, label=f"{name}:{cond}")
            per_cond[cond] = {"tokenization": stat, "ce": ce}
        gap = per_cond["with_special_tokens"]["ce"]["masked_ce"] - per_cond["without_special_tokens"]["ce"]["masked_ce"]
        target_gap = per_cond["with_special_tokens"]["ce"]["target_ratio"] - per_cond["without_special_tokens"]["ce"]["target_ratio"]
        results["datasets"][name]["conditions"] = per_cond
        results["datasets"][name]["gap_special_minus_no_special"] = {
            "masked_ce": gap,
            "target_ratio": target_gap,
            "interpretation_band": "few-hundredths" if abs(gap) < 0.05 else "tenths_or_larger" if abs(gap) >= 0.1 else "intermediate",
        }
        print(json.dumps({"event": "dataset_done", "dataset": name, "gap_ce": gap,
                          "without": per_cond["without_special_tokens"]["ce"]["masked_ce"],
                          "with": per_cond["with_special_tokens"]["ce"]["masked_ce"]}), flush=True)

    # Also record special-token embedding scale relative to normal tokens after raw embedding table.
    try:
        emb = model.deberta.embeddings.word_embeddings.weight.detach().cpu()
        special_ids = [int(x) for x in tokenizer.all_special_ids]
        normal_ids = [i for i in range(emb.shape[0]) if i not in set(special_ids)]
        sample_normals = normal_ids[: min(1000, len(normal_ids))]
        results["embedding_norms"] = {
            "special_ids": special_ids,
            "special_tokens": tokenizer.convert_ids_to_tokens(special_ids),
            "special_norms": {str(i): float(emb[i].norm()) for i in special_ids},
            "normal_first1000_mean_norm": float(emb[sample_normals].norm(dim=1).mean()),
            "normal_first1000_std_norm": float(emb[sample_normals].norm(dim=1).std()),
        }
    except Exception as e:
        results["embedding_norms_error"] = repr(e)

    results["elapsed_sec"] = round(time.time() - t0, 3)
    json_path = out_dir / "special_token_ce_gap.json"
    json_path.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    lines = ["# research special-token CE gap", ""]
    lines.append("This measures the frozen chck_82M slow path with private adapters disabled. Positive gap means adding official [CLS]/[SEP]-style special tokens raises masked-MLM CE on the same text rows.")
    lines.append("")
    for name, d in results["datasets"].items():
        lines.append(f"## {name}")
        lines.append(f"Rows `{d['rows']}`, words `{d['words']}`, source `{d['source_path']}`.")
        no = d["conditions"]["without_special_tokens"]["ce"]
        yes = d["conditions"]["with_special_tokens"]["ce"]
        gapd = d["gap_special_minus_no_special"]
        lines.append(f"- without specials: CE `{no['masked_ce']:.6f}`, targets `{no['targets']}`, target ratio `{no['target_ratio']:.6f}`, non-special tokens `{no['non_special_tokens']}`")
        lines.append(f"- with specials: CE `{yes['masked_ce']:.6f}`, targets `{yes['targets']}`, target ratio `{yes['target_ratio']:.6f}`, non-special tokens `{yes['non_special_tokens']}`")
        lines.append(f"- special-minus-no-special CE gap: `{gapd['masked_ce']:.6f}` ({gapd['interpretation_band']})")
        tn = d["conditions"]["without_special_tokens"]["tokenization"]
        ty = d["conditions"]["with_special_tokens"]["tokenization"]
        lines.append(f"- truncation without/with specials: `{tn['truncated_rows']}`/`{ty['truncated_rows']}` rows, lost tokens `{tn['lost_tokens_to_truncation']}`/`{ty['lost_tokens_to_truncation']}`")
        lines.append("")
    lines.append(f"JSON: `{rel(json_path)}`")
    (out_dir / "special_token_ce_gap.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": results["status"], "json": rel(json_path), "md": rel(out_dir / "special_token_ce_gap.md"), "elapsed_sec": results["elapsed_sec"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
