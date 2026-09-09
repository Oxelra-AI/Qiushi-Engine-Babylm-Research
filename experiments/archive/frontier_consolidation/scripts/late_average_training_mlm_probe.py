#!/usr/bin/env python3
"""research: CPU masked-token probe for the late-weight-average candidate.

This probes whether the already-built 80/82/84 same-trajectory weight average is a
functional low-pass point on ordinary legal-corpus masked-token likelihood, or a
weight-space failure that should not receive later selected BabyLM scoring.  It is
not an official BabyLM evaluation, not a selector for public endpoints, and performs
no upload or leaderboard submission.
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
import random
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

import torch
import torch.nn.functional as F


def find_user_root() -> Path:
    return _PUBLIC_ROOT


ROOT = find_user_root()
STUDY = ROOT / "experiments/archive" / 'frontier_consolidation'
WORKSPACE = STUDY
OUT_DEFAULT = WORKSPACE / "data" / "late_average_training_mlm_probe"
CACHE_ROOT = OUT_DEFAULT / "runtime_cache"
for key, sub in {
    "HF_HOME": "home",
    "HF_HUB_CACHE": "hub",
    "HUGGINGFACE_HUB_CACHE": "hub",
    "HF_DATASETS_CACHE": "datasets",
    "TRANSFORMERS_CACHE": "transformers",
    "XDG_CACHE_HOME": "xdg",
    "HF_MODULES_CACHE": "modules",
}.items():
    p = CACHE_ROOT / sub
    p.mkdir(parents=True, exist_ok=True)
    os.environ[key] = str(p.resolve())
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

CORPUS = WORKSPACE / "data/density_cleanqwen_overlay_medium_riskhard/cleanqwen_fineweb_compact_view_reinvest_100M.jsonl"
REF_RUN = WORKSPACE / "training/runs/adapter128_scale1p75_h100M100M_seed43022_official_ladder/hf_model"
MODELS = {
    "chck_80M": REF_RUN / "chck_80M",
    "chck_82M": REF_RUN / "chck_82M",
    "chck_84M": REF_RUN / "chck_84M",
    "chck_86M": REF_RUN / "chck_86M",
    "chck_100M": REF_RUN / "chck_100M",
    "avg_80_82_84_uniform": WORKSPACE / "data/late_weight_average_scaffold/candidates/reference_scale1p75_seed43022__center_80_82_84_uniform",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def rel(path: Path | str) -> str:
    p = Path(path)
    try:
        return str(p.resolve().relative_to(ROOT))
    except Exception:
        return str(p)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, sort_keys=False) + "\n", encoding="utf-8")


def reservoir_by_source(path: Path, per_source: int, seed: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rng = random.Random(seed)
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    counts: dict[str, int] = defaultdict(int)
    total = 0
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            if not line.strip():
                continue
            total += 1
            obj = json.loads(line)
            src = str(obj.get("source", "unknown"))
            counts[src] += 1
            rec = {"line_number": i, "source": src, "example_id": obj.get("example_id"), "words": obj.get("words"), "text": obj.get("text", "")}
            b = buckets[src]
            if len(b) < per_source:
                b.append(rec)
            else:
                j = rng.randrange(counts[src])
                if j < per_source:
                    b[j] = rec
    selected = []
    for src in sorted(buckets):
        selected.extend(buckets[src])
    selected.sort(key=lambda r: int(r["line_number"]))
    meta = {
        "corpus_path": rel(path),
        "total_nonempty_lines": total,
        "source_counts": dict(sorted(counts.items())),
        "per_source_requested": per_source,
        "selected_count": len(selected),
        "selected_by_source": {src: len(buckets[src]) for src in sorted(buckets)},
    }
    return selected, meta


def build_features(tokenizer, sample: list[dict[str, Any]], seed: int, max_length: int, mask_prob: float) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rng = random.Random(seed)
    features = []
    skipped = []
    mask_id = tokenizer.mask_token_id
    pad_id = tokenizer.pad_token_id
    if mask_id is None or pad_id is None:
        raise RuntimeError("Tokenizer lacks mask or pad token id")
    for ridx, row in enumerate(sample):
        enc = tokenizer(row["text"], truncation=True, max_length=max_length, add_special_tokens=True)
        input_ids = list(enc["input_ids"])
        if hasattr(enc, "word_ids"):
            word_ids = enc.word_ids()
        else:
            word_ids = [None] * len(input_ids)
        groups: dict[int, list[int]] = defaultdict(list)
        for pos, (tid, wid) in enumerate(zip(input_ids, word_ids)):
            if wid is None:
                continue
            if tid in set(tokenizer.all_special_ids):
                continue
            groups[int(wid)].append(pos)
        group_ids = list(groups.keys())
        if not group_ids:
            skipped.append({"row_index": ridx, "line_number": row["line_number"], "reason": "no_maskable_word_groups"})
            continue
        rng.shuffle(group_ids)
        n_content_tokens = sum(len(groups[g]) for g in group_ids)
        target_tokens = max(1, int(round(mask_prob * n_content_tokens)))
        chosen_positions: list[int] = []
        for g in group_ids:
            chosen_positions.extend(groups[g])
            if len(chosen_positions) >= target_tokens:
                break
        labels = [-100] * len(input_ids)
        masked = list(input_ids)
        for pos in chosen_positions:
            labels[pos] = int(input_ids[pos])
            masked[pos] = int(mask_id)
        features.append({
            "probe_row_index": len(features),
            "line_number": row["line_number"],
            "source": row["source"],
            "example_id": row.get("example_id"),
            "words": row.get("words"),
            "sequence_length": len(input_ids),
            "n_word_groups": len(group_ids),
            "n_masked_tokens": len(chosen_positions),
            "input_ids": masked,
            "attention_mask": [1] * len(masked),
            "labels": labels,
        })
    meta = {
        "seed": seed,
        "max_length": max_length,
        "mask_prob": mask_prob,
        "masking_reading": "deterministic whole-word-style random masked groups using tokenizer word_ids on sampled legal-corpus rows",
        "n_features": len(features),
        "n_skipped": len(skipped),
        "skipped": skipped[:20],
        "total_masked_tokens": sum(int(f["n_masked_tokens"]) for f in features),
        "mean_masked_tokens_per_row": mean([int(f["n_masked_tokens"]) for f in features]) if features else None,
        "mean_sequence_length": mean([int(f["sequence_length"]) for f in features]) if features else None,
    }
    return features, meta


def pad_batch(rows: list[dict[str, Any]], pad_id: int) -> dict[str, torch.Tensor]:
    maxlen = max(len(r["input_ids"]) for r in rows)
    ids = []
    attn = []
    labels = []
    for r in rows:
        n = len(r["input_ids"])
        padn = maxlen - n
        ids.append(r["input_ids"] + [pad_id] * padn)
        attn.append(r["attention_mask"] + [0] * padn)
        labels.append(r["labels"] + [-100] * padn)
    return {
        "input_ids": torch.tensor(ids, dtype=torch.long),
        "attention_mask": torch.tensor(attn, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


def score_model(label: str, model_dir: Path, features: list[dict[str, Any]], batch_size: int, torch_threads: int) -> dict[str, Any]:
    from transformers import AutoModelForMaskedLM, AutoTokenizer
    torch.set_num_threads(torch_threads)
    tok = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True)
    model = AutoModelForMaskedLM.from_pretrained(str(model_dir), trust_remote_code=True, local_files_only=True)
    model.eval()
    pad_id = tok.pad_token_id
    if pad_id is None:
        raise RuntimeError(f"No pad token for {label}")
    total_nll = 0.0
    total_tokens = 0
    by_source: dict[str, dict[str, float]] = defaultdict(lambda: {"nll": 0.0, "tokens": 0.0, "rows": 0.0})
    row_records = []
    with torch.no_grad():
        for start in range(0, len(features), batch_size):
            batch_rows = features[start:start + batch_size]
            batch = pad_batch(batch_rows, int(pad_id))
            outputs = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
            logits = outputs.logits
            labels = batch["labels"]
            vocab = logits.shape[-1]
            flat_loss = F.cross_entropy(logits.view(-1, vocab), labels.view(-1), reduction="none", ignore_index=-100)
            losses = flat_loss.view(labels.shape)
            mask = labels.ne(-100)
            for j, r in enumerate(batch_rows):
                toks = int(mask[j].sum().item())
                nll = float(losses[j][mask[j]].sum().item()) if toks else 0.0
                mean_nll = nll / toks if toks else None
                total_nll += nll
                total_tokens += toks
                src = str(r["source"])
                by_source[src]["nll"] += nll
                by_source[src]["tokens"] += toks
                by_source[src]["rows"] += 1
                row_records.append({
                    "probe_row_index": r["probe_row_index"],
                    "line_number": r["line_number"],
                    "source": src,
                    "n_masked_tokens": toks,
                    "mean_nll": mean_nll,
                })
    bys = {}
    for src, v in sorted(by_source.items()):
        toks = int(v["tokens"])
        bys[src] = {"rows": int(v["rows"]), "masked_tokens": toks, "mean_nll": (float(v["nll"]) / toks if toks else None)}
    out = {
        "label": label,
        "model_dir": rel(model_dir),
        "trusted_model_class": model.__class__.__name__,
        "param_count": int(sum(p.numel() for p in model.parameters())),
        "mean_nll": total_nll / total_tokens if total_tokens else None,
        "total_nll": total_nll,
        "masked_tokens": total_tokens,
        "rows": len(features),
        "by_source": bys,
        "row_records": row_records,
    }
    del model
    return out


def summarize_scores(scores: dict[str, Any]) -> dict[str, Any]:
    mean_nll = {label: block["mean_nll"] for label, block in scores.items()}
    sorted_labels = sorted(mean_nll, key=lambda k: float(mean_nll[k]))
    avg_label = "avg_80_82_84_uniform"
    endpoint_labels = [k for k in mean_nll if k.startswith("chck_")]
    avg_nll = float(mean_nll[avg_label]) if avg_label in mean_nll and mean_nll[avg_label] is not None else None
    rows_by_model = {label: {r["probe_row_index"]: r for r in block["row_records"]} for label, block in scores.items()}
    row_deltas = []
    if avg_label in rows_by_model and "chck_84M" in rows_by_model:
        for ridx, ar in rows_by_model[avg_label].items():
            if ridx in rows_by_model["chck_84M"]:
                delta = float(ar["mean_nll"] - rows_by_model["chck_84M"][ridx]["mean_nll"])
                row_deltas.append(delta)
    endpoint_mean_80_82_84 = None
    if all(k in mean_nll for k in ["chck_80M", "chck_82M", "chck_84M"]):
        endpoint_mean_80_82_84 = mean([float(mean_nll[k]) for k in ["chck_80M", "chck_82M", "chck_84M"]])
    return {
        "mean_nll_by_model": mean_nll,
        "ranked_by_sample_mean_nll": sorted_labels,
        "avg_minus_chck84_mean_nll": (avg_nll - float(mean_nll["chck_84M"])) if avg_nll is not None and "chck_84M" in mean_nll else None,
        "avg_minus_mean_endpoint_80_82_84_nll": (avg_nll - endpoint_mean_80_82_84) if avg_nll is not None and endpoint_mean_80_82_84 is not None else None,
        "row_delta_avg_minus_chck84": {
            "n": len(row_deltas),
            "mean": float(mean(row_deltas)) if row_deltas else None,
            "std": float(pstdev(row_deltas)) if len(row_deltas) > 1 else (0.0 if row_deltas else None),
            "fraction_avg_lower_nll_than_chck84": sum(1 for d in row_deltas if d < 0.0) / len(row_deltas) if row_deltas else None,
        },
        "interpretation": "Legal-corpus masked-token NLL only checks local functional smoothness/collapse risk. Earlier evidence shows MLM likelihood does not select BabyLM competence, so selected task scoring remains necessary for any endpoint decision.",
    }


def write_markdown(path: Path, result: dict[str, Any]) -> None:
    lines: list[str] = []
    lines.append("# research late average legal-corpus MLM probe\n\n")
    lines.append("This CPU probe uses deterministic masked-token likelihood on sampled legal-corpus rows. It is not an official BabyLM score, not public endpoint selection, and performs no upload or leaderboard submission.\n\n")
    meta = result["sample_meta"]
    feat = result["feature_meta"]
    lines.append(f"Sample: `{meta['selected_count']}` rows from `{meta['corpus_path']}`, per-source request `{meta['per_source_requested']}`, masked tokens `{feat['total_masked_tokens']}`, mean sequence length `{feat['mean_sequence_length']:.2f}`.\n\n")
    lines.append("## Mean masked-token NLL\n\n")
    lines.append("| model | mean NLL | masked tokens | rows |\n")
    lines.append("|---|---:|---:|---:|\n")
    for label in result["summary"]["ranked_by_sample_mean_nll"]:
        block = result["scores"][label]
        lines.append(f"| {label} | {block['mean_nll']:.6f} | {block['masked_tokens']} | {block['rows']} |\n")
    s = result["summary"]
    lines.append("\n## Comparisons involving the built 80/82/84 average\n\n")
    lines.append(f"- average minus chck_84M mean NLL: `{s['avg_minus_chck84_mean_nll']:.6f}`\n")
    lines.append(f"- average minus arithmetic mean of endpoint NLLs (80/82/84): `{s['avg_minus_mean_endpoint_80_82_84_nll']:.6f}`\n")
    rd = s["row_delta_avg_minus_chck84"]
    lines.append(f"- row-level avg lower NLL than chck84 fraction: `{rd['fraction_avg_lower_nll_than_chck84']:.4f}` over `{rd['n']}` rows\n")
    lines.append("\n## Per-source mean NLL\n\n")
    sources = sorted(next(iter(result["scores"].values()))["by_source"].keys())
    lines.append("| source | " + " | ".join(result["scores"].keys()) + " |\n")
    lines.append("|---" + "|---:" * len(result["scores"]) + "|\n")
    for src in sources:
        vals = []
        for label, block in result["scores"].items():
            v = block["by_source"].get(src, {}).get("mean_nll")
            vals.append("" if v is None else f"{float(v):.4f}")
        lines.append(f"| {src} | " + " | ".join(vals) + " |\n")
    lines.append("\n## Scientific reading\n\n")
    lines.append(result["summary"]["interpretation"] + "\n")
    lines.append(f"\nJSON: `{rel(path.with_suffix('.json'))}`\n")
    path.write_text("".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=OUT_DEFAULT)
    ap.add_argument("--per-source", type=int, default=32)
    ap.add_argument("--sample-seed", type=int, default=192001)
    ap.add_argument("--mask-seed", type=int, default=192002)
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--mask-prob", type=float, default=0.15)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--torch-threads", type=int, default=8)
    ap.add_argument("--models", nargs="+", default=list(MODELS.keys()), choices=sorted(MODELS.keys()))
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(MODELS[args.models[0]]), local_files_only=True)
    sample, sample_meta = reservoir_by_source(CORPUS, args.per_source, args.sample_seed)
    features, feature_meta = build_features(tok, sample, args.mask_seed, args.max_length, args.mask_prob)
    feature_public = [{k: v for k, v in f.items() if k not in ("input_ids", "attention_mask", "labels")} for f in features]
    write_json(args.out_dir / "probe_sample_rows.json", sample)
    write_json(args.out_dir / "probe_mask_features_meta.json", {"feature_meta": feature_meta, "features": feature_public})

    scores = {}
    for label in args.models:
        scores[label] = score_model(label, MODELS[label], features, args.batch_size, args.torch_threads)
    result = {
        "status": "LATE_AVERAGE_TRAINING_MLM_PROBE_COMPLETE",
        "created_utc": utc_now(),
        "models": {label: rel(MODELS[label]) for label in args.models},
        "sample_meta": sample_meta,
        "feature_meta": feature_meta,
        "scores": scores,
        "summary": summarize_scores(scores),
        "official_babylm_evaluation_performed": False,
        "model_upload_performed": False,
        "leaderboard_submission_performed": False,
    }
    out_json = args.out_dir / "late_average_training_mlm_probe.json"
    out_md = args.out_dir / "late_average_training_mlm_probe.md"
    # Store compact row records separately for easier inspection and keep main result readable.
    row_records = {label: result["scores"][label].pop("row_records") for label in list(result["scores"].keys())}
    write_json(args.out_dir / "row_level_model_nll_records.json", row_records)
    write_json(out_json, result)
    write_markdown(out_md, result)
    print(json.dumps({"status": result["status"], "out_json": rel(out_json), "out_md": rel(out_md)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
