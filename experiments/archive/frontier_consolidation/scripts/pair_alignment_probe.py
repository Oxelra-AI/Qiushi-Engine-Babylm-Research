#!/usr/bin/env python3
"""research: CPU source/compact-view representation alignment probe.

This reads only the frozen legal compact-view reinvest training stream, the research
training-side pair span map, and already-trained local checkpoints.  It performs no
training, no official evaluation, and no route launch.  Its purpose is to measure
whether the existing token-mean model already makes source and compact rewrite
representations close, and whether the word-mean objective changed that geometry.
The result informs the scientific leverage of a future shared-subspace consistency
loss relative to sequence/utilization or other optimization routes.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import hashlib
import json
import math
import pathlib
import random
import statistics
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

EXPECTED_TRAIN_SHA = "3dd19f09deeca44d6b07340e70cd15baa4b5c7bffe0bd462ff37536e470e7691"
EXPECTED_TOKENIZER_SHA = "91b775514b9f4e2d1f37c28445ab98181007e16d14f763955168547e293ee8f9"
CHANGED_SOURCE = "cleanqwen_fineweb_compact_view_reinvest"
SEQ_LEN = 256


def find_user_root() -> pathlib.Path:
    return _PUBLIC_ROOT


USER_ROOT = find_user_root()
WORKSPACE = USER_ROOT / "experiments/archive/frontier_consolidation"
TRAIN_100M = WORKSPACE / "data/density_cleanqwen_overlay_medium_riskhard/cleanqwen_fineweb_compact_view_reinvest_100M.jsonl"
SPAN_JSONL = WORKSPACE / "data/pair_span_map/pair_span_map.jsonl"
TOKENIZER_DIR = WORKSPACE / "data/compliant_tokenizer"
OUT_DIR = WORKSPACE / "data/pair_alignment_probe"
RUN_TOKENMEAN = WORKSPACE / "training/runs/complianttok_reinvest_seed43022_r2/hf_model"
RUN_WORDMEAN = WORKSPACE / "training/runs/wordmean_mlm_complianttok_reinvest_seed43022_80M/hf_model"

CHECKPOINTS = [
    {
        "name": "tokenmean_20M",
        "family": "tokenmean",
        "path": RUN_TOKENMEAN / "chck_20M",
        "score_context": "token-mean legal compact-view reinvest, before mature positive effect fully emerges",
    },
    {
        "name": "tokenmean_80M",
        "family": "tokenmean",
        "path": RUN_TOKENMEAN / "chck_80M",
        "score_context": "token-mean legal compact-view reinvest mature cheap-column reference",
    },
    {
        "name": "tokenmean_100M",
        "family": "tokenmean",
        "path": RUN_TOKENMEAN / "chck_100M",
        "score_context": "best fully legal complete endpoint, Overall 41.2578",
    },
    {
        "name": "wordmean_80M",
        "family": "wordmean",
        "path": RUN_WORDMEAN / "chck_80M",
        "score_context": "global word-mean objective, closed after broad 80M language-column loss versus token-mean",
    },
]


@dataclass
class ChangedExample:
    row_1based: int
    example_id: int
    text: str
    words: int
    span: dict[str, Any]


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_span_map(path: pathlib.Path) -> dict[int, dict[str, Any]]:
    out: dict[int, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                obj = json.loads(line)
                out[int(obj["example_id"])] = obj
    return out


def load_changed_examples(train_path: pathlib.Path, span_by_ex: dict[int, dict[str, Any]]) -> list[ChangedExample]:
    examples: list[ChangedExample] = []
    seen: set[int] = set()
    # The stream has ten shuffled passes; use the first occurrence of each changed
    # example_id to avoid pseudo-replication.
    with train_path.open("r", encoding="utf-8") as f:
        for row_1, line in enumerate(f, 1):
            obj = json.loads(line)
            ex_id = int(obj.get("example_id", row_1 - 1))
            if ex_id in seen or ex_id not in span_by_ex:
                continue
            if str(obj.get("source", "")) != CHANGED_SOURCE:
                continue
            examples.append(ChangedExample(
                row_1based=row_1,
                example_id=ex_id,
                text=str(obj["text"]),
                words=int(obj.get("words", len(str(obj["text"]).split()))),
                span=span_by_ex[ex_id],
            ))
            seen.add(ex_id)
            if len(seen) == len(span_by_ex):
                break
    examples.sort(key=lambda x: x.example_id)
    return examples


def select_examples(examples: list[ChangedExample], n: int, seed: int) -> list[ChangedExample]:
    if n <= 0 or n >= len(examples):
        return examples
    rng = random.Random(seed)
    idxs = sorted(rng.sample(range(len(examples)), n))
    return [examples[i] for i in idxs]


def build_batch(examples: list[ChangedExample], tokenizer) -> dict[str, Any]:
    texts = [ex.text for ex in examples]
    enc = tokenizer(
        texts,
        add_special_tokens=False,
        truncation=True,
        max_length=SEQ_LEN,
        padding="max_length",
        return_tensors="pt",
    )
    aux_records: list[dict[str, Any]] = []
    errors = defaultdict(int)
    attention = enc["attention_mask"]
    for bi, ex in enumerate(examples):
        actual_len = int(attention[bi].sum().item())
        span_len = int(ex.span.get("token_len_truncated", -1))
        if span_len != actual_len:
            errors["token_length_mismatch"] += 1
            continue
        for pair in ex.span.get("pairs") or []:
            if pair.get("visibility") != "both_visible":
                continue
            src_ranges = [[int(a), int(b)] for a, b in pair.get("source_token_ranges") or []]
            rew_ranges = [[int(a), int(b)] for a, b in pair.get("rewrite_token_ranges") or []]
            if not src_ranges or not rew_ranges:
                errors["empty_range"] += 1
                continue
            bad = False
            for a, b in src_ranges + rew_ranges:
                if a < 0 or b <= a or b > SEQ_LEN:
                    bad = True
                    errors["invalid_range"] += 1
                    break
                if int(attention[bi, a:b].sum().item()) != b - a:
                    bad = True
                    errors["range_crosses_padding"] += 1
                    break
            if bad:
                continue
            aux_records.append({
                "batch_row": bi,
                "example_id": int(ex.example_id),
                "global_row_1based": int(ex.row_1based),
                "pair_id": str(pair.get("pair_id", "")),
                "source_ranges": src_ranges,
                "rewrite_ranges": rew_ranges,
                "source_token_count": int(sum(b - a for a, b in src_ranges)),
                "rewrite_token_count": int(sum(b - a for a, b in rew_ranges)),
            })
    enc["aux_records"] = aux_records
    enc["aux_errors"] = dict(errors)
    return enc


def pool_ranges(hidden: torch.Tensor, batch_row: int, ranges: list[list[int]]) -> torch.Tensor:
    chunks = [hidden[batch_row, int(a):int(b), :] for a, b in ranges]
    return torch.cat(chunks, dim=0).mean(dim=0)


def summarize(vals: list[float]) -> dict[str, float | int]:
    clean = [float(v) for v in vals if math.isfinite(float(v))]
    if not clean:
        return {"n": 0, "mean": None, "median": None, "p05": None, "p95": None, "min": None, "max": None, "std": None}
    s = sorted(clean)
    def q(frac: float) -> float:
        if len(s) == 1:
            return s[0]
        pos = frac * (len(s) - 1)
        lo = int(math.floor(pos)); hi = int(math.ceil(pos))
        if lo == hi:
            return s[lo]
        return s[lo] * (hi - pos) + s[hi] * (pos - lo)
    return {
        "n": len(clean),
        "mean": float(statistics.mean(clean)),
        "median": float(statistics.median(clean)),
        "p05": float(q(0.05)),
        "p95": float(q(0.95)),
        "min": float(s[0]),
        "max": float(s[-1]),
        "std": float(statistics.pstdev(clean)),
    }


def cosine_rows(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    an = torch.nn.functional.normalize(a.float(), dim=1, eps=1e-6)
    bn = torch.nn.functional.normalize(b.float(), dim=1, eps=1e-6)
    return (an * bn).sum(dim=1)


def retrieval_stats(src: torch.Tensor, rew: torch.Tensor, centered: bool) -> dict[str, float | int]:
    x = src.float()
    y = rew.float()
    if centered:
        mean = torch.cat([x, y], dim=0).mean(dim=0, keepdim=True)
        x = x - mean
        y = y - mean
    x = torch.nn.functional.normalize(x, dim=1, eps=1e-6)
    y = torch.nn.functional.normalize(y, dim=1, eps=1e-6)
    sim = x @ y.t()
    n = sim.shape[0]
    ranks = []
    chunk = 512
    for start in range(0, n, chunk):
        end = min(n, start + chunk)
        block = sim[start:end]
        true_vals = block[torch.arange(end - start), torch.arange(start, end)].unsqueeze(1)
        rank = (block > true_vals).sum(dim=1) + 1
        ranks.extend([int(v) for v in rank.cpu().tolist()])
    return {
        "n": n,
        "top1": float(sum(1 for r in ranks if r == 1) / n) if n else 0.0,
        "top5": float(sum(1 for r in ranks if r <= 5) / n) if n else 0.0,
        "mean_rank": float(statistics.mean(ranks)) if ranks else None,
        "median_rank": float(statistics.median(ranks)) if ranks else None,
    }


def same_row_other_cos(src: torch.Tensor, rew: torch.Tensor, records: list[dict[str, Any]], centered: bool) -> list[float]:
    if centered:
        mean = torch.cat([src.float(), rew.float()], dim=0).mean(dim=0, keepdim=True)
        src_use = src.float() - mean
        rew_use = rew.float() - mean
    else:
        src_use = src.float()
        rew_use = rew.float()
    by_ex: dict[int, list[int]] = defaultdict(list)
    for i, r in enumerate(records):
        by_ex[int(r["example_id"])].append(i)
    vals: list[float] = []
    for idxs in by_ex.values():
        if len(idxs) < 2:
            continue
        for j, i in enumerate(idxs):
            k = idxs[(j + 1) % len(idxs)]
            vals.append(float(cosine_rows(src_use[i:i+1], rew_use[k:k+1]).item()))
    return vals


def summarize_layer(src: torch.Tensor, rew: torch.Tensor, records: list[dict[str, Any]], seed: int) -> dict[str, Any]:
    n = src.shape[0]
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    perm = torch.randperm(n, generator=gen)
    if n > 1:
        # Rotate any accidental fixed points so the shuffled control never compares
        # a pair to itself.
        fixed = perm == torch.arange(n)
        if bool(fixed.any()):
            perm = torch.roll(perm, shifts=1)
    raw_actual = cosine_rows(src, rew)
    raw_shuffled = cosine_rows(src, rew[perm])
    mean_vec = torch.cat([src.float(), rew.float()], dim=0).mean(dim=0, keepdim=True)
    src_c = src.float() - mean_vec
    rew_c = rew.float() - mean_vec
    cen_actual = cosine_rows(src_c, rew_c)
    cen_shuffled = cosine_rows(src_c, rew_c[perm])
    raw_same_row = same_row_other_cos(src, rew, records, centered=False)
    cen_same_row = same_row_other_cos(src, rew, records, centered=True)
    return {
        "n_pair_records": int(n),
        "n_examples": int(len({int(r["example_id"]) for r in records})),
        "source_tokens_per_pair": summarize([float(r["source_token_count"]) for r in records]),
        "rewrite_tokens_per_pair": summarize([float(r["rewrite_token_count"]) for r in records]),
        "raw_actual_cos": summarize([float(x) for x in raw_actual.cpu().tolist()]),
        "raw_shuffled_cos": summarize([float(x) for x in raw_shuffled.cpu().tolist()]),
        "raw_actual_minus_shuffled_mean": float(raw_actual.mean().item() - raw_shuffled.mean().item()),
        "raw_same_row_other_cos": summarize(raw_same_row),
        "centered_actual_cos": summarize([float(x) for x in cen_actual.cpu().tolist()]),
        "centered_shuffled_cos": summarize([float(x) for x in cen_shuffled.cpu().tolist()]),
        "centered_actual_minus_shuffled_mean": float(cen_actual.mean().item() - cen_shuffled.mean().item()),
        "centered_same_row_other_cos": summarize(cen_same_row),
        "retrieval_raw": retrieval_stats(src, rew, centered=False),
        "retrieval_centered": retrieval_stats(src, rew, centered=True),
    }


def analyze_checkpoint(ckpt: dict[str, Any], tokenizer, examples: list[ChangedExample], batch_size: int, layers: list[int], seed: int, torch_threads: int) -> dict[str, Any]:
    path = pathlib.Path(ckpt["path"])
    if not path.exists():
        return {"name": ckpt["name"], "status": "missing", "path": str(path)}
    if torch_threads > 0:
        torch.set_num_threads(torch_threads)
    t0 = time.time()
    model = AutoModelForMaskedLM.from_pretrained(str(path))
    model.eval()
    model.to("cpu")
    layer_src: dict[int, list[torch.Tensor]] = {li: [] for li in layers}
    layer_rew: dict[int, list[torch.Tensor]] = {li: [] for li in layers}
    all_records: list[dict[str, Any]] = []
    total_errors = defaultdict(int)
    with torch.inference_mode():
        for start in range(0, len(examples), batch_size):
            batch_examples = examples[start:start + batch_size]
            batch = build_batch(batch_examples, tokenizer)
            aux = batch.pop("aux_records")
            errors = batch.pop("aux_errors")
            for k, v in errors.items():
                total_errors[k] += int(v)
            if not aux:
                continue
            input_ids = batch["input_ids"].to("cpu")
            attention_mask = batch["attention_mask"].to("cpu")
            out = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True, return_dict=True)
            hidden_states = out.hidden_states
            n_layers = len(hidden_states)
            use_layers = []
            for li in layers:
                actual_li = li if li >= 0 else n_layers + li
                if actual_li < 0 or actual_li >= n_layers:
                    raise RuntimeError(f"layer index {li} not available for {ckpt['name']} with {n_layers} hidden states")
                use_layers.append((li, actual_li))
            for rec in aux:
                all_records.append(rec)
                br = int(rec["batch_row"])
                for li, actual_li in use_layers:
                    h = hidden_states[actual_li]
                    layer_src[li].append(pool_ranges(h, br, rec["source_ranges"]).detach().cpu())
                    layer_rew[li].append(pool_ranges(h, br, rec["rewrite_ranges"]).detach().cpu())
    if not all_records:
        return {"name": ckpt["name"], "status": "no_aux_records", "path": str(path), "aux_errors": dict(total_errors)}
    layer_summaries: dict[str, Any] = {}
    for li in layers:
        src = torch.stack(layer_src[li], dim=0)
        rew = torch.stack(layer_rew[li], dim=0)
        label = "embedding" if li == 0 else ("last" if li < 0 else f"layer_{li}")
        layer_summaries[label] = summarize_layer(src, rew, all_records, seed=seed + 1009 * (li if li >= 0 else 99))
    del model
    return {
        "name": ckpt["name"],
        "family": ckpt["family"],
        "path": str(path),
        "score_context": ckpt["score_context"],
        "status": "ok",
        "elapsed_sec": round(time.time() - t0, 3),
        "n_examples_input": len(examples),
        "n_pair_records": len(all_records),
        "aux_errors": dict(total_errors),
        "layers": layer_summaries,
    }


def make_note(summary: dict[str, Any], out_json: pathlib.Path) -> str:
    lines = [
        "# research source-view representation alignment probe",
        "",
        "CPU-only measurement over already trained checkpoints. It does not train, evaluate official tasks, change data, or select the next H100 run.",
        "",
        "## Purpose",
        "",
        "Measure whether source spans and their compact rewrite spans in the frozen changed block are already close in the model representation, using shuffled and same-row controls. This informs the possible leverage of a future low-weight shared-subspace consistency loss if the active minfreq50 screen is weak.",
        "",
        "## Sample",
        f"- changed examples available: `{summary['sample']['changed_examples_available']}`",
        f"- changed examples sampled: `{summary['sample']['changed_examples_sampled']}`",
        f"- pair span records in sampled rows: `{summary['sample']['sample_pair_records_both_visible']}`",
        f"- train SHA: `{summary['inputs']['train_sha256']}`",
        f"- tokenizer SHA: `{summary['inputs']['tokenizer_sha256']}`",
        "",
        "## Last-layer centered geometry",
    ]
    for r in summary["checkpoint_results"]:
        if r.get("status") != "ok":
            lines.append(f"- {r['name']}: status `{r.get('status')}`")
            continue
        last = r["layers"].get("last") or r["layers"].get("layer_8")
        if last is None:
            keys = list(r["layers"].keys())
            last = r["layers"][keys[-1]]
        lines.append(
            f"- {r['name']}: pairs={r['n_pair_records']}, centered actual cos mean={last['centered_actual_cos']['mean']:.4f}, "
            f"shuffled={last['centered_shuffled_cos']['mean']:.4f}, margin={last['centered_actual_minus_shuffled_mean']:.4f}, "
            f"same-row-other={last['centered_same_row_other_cos']['mean']:.4f}, centered retrieval top1={last['retrieval_centered']['top1']:.4f}, top5={last['retrieval_centered']['top5']:.4f}, elapsed={r['elapsed_sec']}s"
        )
    lines += ["", "## Interpretation"]
    interp = summary.get("scientific_interpretation") or []
    for item in interp:
        lines.append(f"- {item}")
    lines += ["", f"Full JSON: `{out_json}`"]
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample-examples", type=int, default=192)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--seed", type=int, default=6901)
    ap.add_argument("--layers", default="0,4,-1", help="hidden-state indices: 0=embedding, -1=last")
    ap.add_argument("--torch-threads", type=int, default=8)
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    args = ap.parse_args()
    t0 = time.time()
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_sha = sha256_file(TRAIN_100M)
    tok_sha = sha256_file(TOKENIZER_DIR / "tokenizer.json")
    if train_sha != EXPECTED_TRAIN_SHA:
        raise RuntimeError(f"train SHA mismatch: {train_sha}")
    if tok_sha != EXPECTED_TOKENIZER_SHA:
        raise RuntimeError(f"tokenizer SHA mismatch: {tok_sha}")
    span_by_ex = load_span_map(SPAN_JSONL)
    all_changed = load_changed_examples(TRAIN_100M, span_by_ex)
    sampled = select_examples(all_changed, args.sample_examples, args.seed)
    sample_pair_records = sum(1 for ex in sampled for p in (ex.span.get("pairs") or []) if p.get("visibility") == "both_visible")
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER_DIR), use_fast=True)
    layers = [int(x.strip()) for x in args.layers.split(",") if x.strip()]
    results = []
    for ckpt in CHECKPOINTS:
        results.append(analyze_checkpoint(ckpt, tokenizer, sampled, args.batch_size, layers, args.seed, args.torch_threads))

    # A compact cross-check: compare token-mean mature geometry to word-mean mature geometry.
    by_name = {r.get("name"): r for r in results if r.get("status") == "ok"}
    contrasts: list[dict[str, Any]] = []
    def layer_metric(name: str, layer: str, field: str) -> float | None:
        r = by_name.get(name)
        if not r:
            return None
        lay = r["layers"].get(layer)
        if not lay:
            return None
        return float(lay[field])
    for layer in ["embedding", "layer_4", "last"]:
        t80 = layer_metric("tokenmean_80M", layer, "centered_actual_minus_shuffled_mean")
        t20 = layer_metric("tokenmean_20M", layer, "centered_actual_minus_shuffled_mean")
        t100 = layer_metric("tokenmean_100M", layer, "centered_actual_minus_shuffled_mean")
        w80 = layer_metric("wordmean_80M", layer, "centered_actual_minus_shuffled_mean")
        row: dict[str, Any] = {"layer": layer}
        if t20 is not None and t80 is not None:
            row["tokenmean_80M_minus_20M_margin"] = t80 - t20
        if t80 is not None and t100 is not None:
            row["tokenmean_100M_minus_80M_margin"] = t100 - t80
        if t80 is not None and w80 is not None:
            row["wordmean_80M_minus_tokenmean_80M_margin"] = w80 - t80
        contrasts.append(row)

    interpretation = []
    last_t80 = by_name.get("tokenmean_80M", {}).get("layers", {}).get("last")
    last_w80 = by_name.get("wordmean_80M", {}).get("layers", {}).get("last")
    if last_t80:
        margin = float(last_t80["centered_actual_minus_shuffled_mean"])
        top1 = float(last_t80["retrieval_centered"]["top1"])
        same_row = last_t80["centered_same_row_other_cos"]["mean"]
        interpretation.append(
            f"Token-mean 80M has a measurable paired-view geometry in the last layer: centered actual-minus-shuffled cosine margin {margin:.4f}, retrieval top1 {top1:.4f}, and same-row-other mean {same_row:.4f}. This means the training stream already induces some source/rewrite abstraction; a future consistency loss should be low-weight and subspace-limited rather than full-vector forcing."
        )
    if last_t80 and last_w80:
        dm = float(last_w80["centered_actual_minus_shuffled_mean"]) - float(last_t80["centered_actual_minus_shuffled_mean"])
        dt = float(last_w80["retrieval_centered"]["top1"]) - float(last_t80["retrieval_centered"]["top1"])
        interpretation.append(
            f"Word-mean 80M changes last-layer paired geometry relative to token-mean by margin delta {dm:+.4f} and retrieval-top1 delta {dt:+.4f}. Because word-mean harmed broad language scores, any increase here would not by itself endorse stronger alignment; any decrease would suggest that its GlobalPIQA/COMPS signal is not source-view abstraction."
        )
    interpretation.append(
        "This CPU result is not behavior evidence for a source-view objective. It only tells the next route choice whether the validated compact-view mechanism still has representational headroom and what failure mode to avoid: forcing all source/rewrite information together instead of extracting a small invariant component while preserving residual details."
    )
    summary = {
        "status": "PAIR_ALIGNMENT_PROBE",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "purpose": "CPU-only source/compact-view representation alignment measurement for successor-route leverage while minfreq50 H100 work runs.",
        "inputs": {
            "train_100m": str(TRAIN_100M),
            "train_sha256": train_sha,
            "span_jsonl": str(SPAN_JSONL),
            "span_examples": len(span_by_ex),
            "tokenizer_dir": str(TOKENIZER_DIR),
            "tokenizer_sha256": tok_sha,
            "seq_len": SEQ_LEN,
            "layers": layers,
            "batch_size": args.batch_size,
            "torch_threads": args.torch_threads,
        },
        "sample": {
            "changed_examples_available": len(all_changed),
            "changed_examples_sampled": len(sampled),
            "sample_example_ids_first10": [ex.example_id for ex in sampled[:10]],
            "sample_pair_records_both_visible": sample_pair_records,
        },
        "checkpoint_results": results,
        "contrasts": contrasts,
        "scientific_interpretation": interpretation,
        "elapsed_sec": round(time.time() - t0, 3),
    }
    out_json = out_dir / "pair_alignment_probe.json"
    out_md = out_dir / "pair_alignment_probe.md"
    out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    out_md.write_text(make_note(summary, out_json), encoding="utf-8")
    print(json.dumps({
        "status": summary["status"],
        "out_json": str(out_json),
        "out_md": str(out_md),
        "sampled_examples": len(sampled),
        "sample_pair_records": sample_pair_records,
        "checkpoint_status": {r.get("name"): r.get("status") for r in results},
        "elapsed_sec": summary["elapsed_sec"],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
