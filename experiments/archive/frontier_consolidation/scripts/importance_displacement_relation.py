#!/usr/bin/env python3
"""research: research squared-gradient importance versus endpoint displacement.

This bounded analysis estimates which parameters the frozen legal research MLM loss is
locally sensitive to on legal-corpus samples, then compares that importance with
parameter displacements in two closed endpoint routes (scale1.75 and U256) and the
normal research 80M->100M late movement.

Interpretation limit: this is evidence about MLM-sensitive displacement only. It does
not identify which representations should be protected for BabyLM broad competence by
itself.
"""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import math
import os
import random
import statistics
import time
from pathlib import Path
from typing import Any

STUDY = Path("experiments/archive/frontier_consolidation")
WORKSPACE = STUDY
DEFAULT_CACHE = WORKSPACE / "data/hf_cache_importance"
os.environ.setdefault("HF_HOME", str(DEFAULT_CACHE / "hf_home"))
os.environ.setdefault("TRANSFORMERS_CACHE", str(DEFAULT_CACHE / "transformers"))
os.environ.setdefault("HF_MODULES_CACHE", str(DEFAULT_CACHE / "modules"))
for _p in [os.environ["HF_HOME"], os.environ["TRANSFORMERS_CACHE"], os.environ["HF_MODULES_CACHE"]]:
    Path(_p).mkdir(parents=True, exist_ok=True)

import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from transformers import AutoModelForMaskedLM, AutoTokenizer

POOL = WORKSPACE / "data/density_cleanqwen_overlay_medium_riskhard/cleanqwen_fineweb_compact_view_reinvest_10M.jsonl"
TOKENIZER_DIR = WORKSPACE / "data/compliant_tokenizer"
research = WORKSPACE / "training/runs/complianttok_reinvest_seed43022_r2/hf_model"
SCALE = WORKSPACE / "training/runs/adapter128_scale1p75_h100M100M_seed43022_official_ladder/hf_model"
U256 = WORKSPACE / "training/runs/eu_U256_legal16k_seed43022_100M/hf_model"
EXPECTED_POOL_SHA = "215944978157394dbecf2039f2c1e1806bfcbb9701421920f78605eb58975a23"
EXPECTED_TOKENIZER_SHA = "91b775514b9f4e2d1f37c28445ab98181007e16d14f763955168547e293ee8f9"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_jsonl_pool(max_per_source: int, seed: int) -> list[dict[str, Any]]:
    by_src: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    rng = random.Random(seed)
    with POOL.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            if not line.strip():
                continue
            obj = json.loads(line)
            obj["_row_index"] = i
            by_src[str(obj.get("source"))].append(obj)
    rows: list[dict[str, Any]] = []
    for src, xs in sorted(by_src.items()):
        rng.shuffle(xs)
        rows.extend(xs[:max_per_source])
    rng.shuffle(rows)
    return rows


def wwm_mask_batch(tokenizer, texts: list[str], seed: int, max_length: int, mask_prob: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
    gen = random.Random(seed)
    enc = tokenizer(texts, add_special_tokens=True, padding=True, truncation=True, max_length=max_length, return_offsets_mapping=True, return_tensors="pt")
    input_ids = enc["input_ids"].clone()
    labels = torch.full_like(input_ids, -100)
    attention = enc["attention_mask"]
    mask_id = int(tokenizer.mask_token_id)
    special_ids = set(int(x) for x in tokenizer.all_special_ids)
    vocab_size = int(tokenizer.vocab_size)
    total_groups = 0
    selected_groups = 0
    selected_tokens = 0
    for b in range(input_ids.shape[0]):
        offs = enc["offset_mapping"][b].tolist()
        groups: list[list[int]] = []
        cur: list[int] = []
        prev_end = None
        for i, (a, e) in enumerate(offs):
            tid = int(input_ids[b, i])
            if int(attention[b, i]) == 0 or tid in special_ids or e <= a:
                if cur:
                    groups.append(cur); cur = []
                prev_end = None
                continue
            # Byte-level tokenizer offsets: contiguous alphanumeric pieces usually have adjacent offsets.
            txt_piece = tokenizer.convert_ids_to_tokens([tid])[0]
            # group consecutive non-space pieces into word-like units; punctuation is its own group.
            if prev_end is not None and a == prev_end and cur:
                cur.append(i)
            else:
                if cur:
                    groups.append(cur)
                cur = [i]
            prev_end = e
        if cur:
            groups.append(cur)
        if not groups:
            continue
        total_groups += len(groups)
        chosen: list[int] = []
        for g in groups:
            if gen.random() < mask_prob:
                chosen.extend(g)
        if not chosen:
            chosen.extend(gen.choice(groups))
        for i in chosen:
            labels[b, i] = input_ids[b, i]
            r = gen.random()
            if r < 0.80:
                input_ids[b, i] = mask_id
            elif r < 0.90:
                input_ids[b, i] = gen.randrange(vocab_size)
            # else keep original
        selected_groups += len(chosen)  # token count proxy; exact group count not needed for importance
        selected_tokens += len(chosen)
    meta = {"masked_token_count": int((labels != -100).sum().item()), "candidate_group_count_proxy": total_groups, "selected_token_count": selected_tokens}
    return input_ids, attention, labels, meta


def group_name(n: str) -> str:
    if n.startswith("deberta.embeddings"):
        return "embeddings"
    if n.startswith("deberta.encoder.layer."):
        parts = n.split(".")
        layer = parts[3] if len(parts) > 3 else "?"
        if ".attention." in n:
            return f"layer{layer}_attention"
        if ".intermediate." in n or ".output." in n:
            return f"layer{layer}_ffn_output"
        return f"layer{layer}_other"
    if n.startswith("cls."):
        return "mlm_head"
    return "other"


def model_safetensors(root: Path, exp: str) -> Path:
    return root / f"chck_{exp}" / "model.safetensors"


def stock_common_names(ref: dict[str, torch.Tensor], other: dict[str, torch.Tensor]) -> list[str]:
    return sorted(n for n in (set(ref) & set(other)) if ".adapter." not in n and ref[n].shape == other[n].shape)


def summarize(vals: list[float]) -> dict[str, Any]:
    vals = [float(v) for v in vals if math.isfinite(float(v))]
    if not vals:
        return {"n": 0, "mean": None, "median": None, "p10": None, "p90": None, "min": None, "max": None}
    vals.sort()
    def q(p: float) -> float:
        if len(vals) == 1:
            return vals[0]
        pos = p * (len(vals) - 1)
        lo = int(math.floor(pos)); hi = int(math.ceil(pos))
        if lo == hi:
            return vals[lo]
        return vals[lo] * (hi - pos) + vals[hi] * (pos - lo)
    return {"n": len(vals), "mean": statistics.mean(vals), "median": statistics.median(vals), "p10": q(0.10), "p90": q(0.90), "min": vals[0], "max": vals[-1]}


def pearson(x: list[float], y: list[float]) -> float | None:
    if len(x) < 3 or len(y) != len(x):
        return None
    mx = statistics.mean(x); my = statistics.mean(y)
    vx = sum((a - mx) ** 2 for a in x); vy = sum((b - my) ** 2 for b in y)
    if vx <= 0 or vy <= 0:
        return None
    return sum((a - mx) * (b - my) for a, b in zip(x, y)) / math.sqrt(vx * vy)


def estimate_importance(args) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = "cuda" if (args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available())) else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER_DIR), use_fast=True)
    model = AutoModelForMaskedLM.from_pretrained(str(research / "chck_100M"))
    model.to(device)
    model.train(False)
    rows = read_jsonl_pool(args.max_rows_per_source, args.seed)
    # Accumulate per-batch squared gradients on CPU.
    grad2: dict[str, torch.Tensor] = {}
    source_mask_counts = collections.Counter()
    source_rows = collections.Counter()
    losses = []
    masked_counts = []
    for start in range(0, len(rows), args.batch_size):
        batch = rows[start:start + args.batch_size]
        texts = [str(r["text"]) for r in batch]
        input_ids, attention, labels, meta = wwm_mask_batch(tokenizer, texts, args.seed + start, args.max_length, args.mask_prob)
        if meta["masked_token_count"] <= 0:
            continue
        for r in batch:
            source_rows[str(r.get("source"))] += 1
        input_ids = input_ids.to(device); attention = attention.to(device); labels = labels.to(device)
        model.zero_grad(set_to_none=True)
        out = model(input_ids=input_ids, attention_mask=attention, labels=labels, return_dict=True)
        loss = out.loss
        loss.backward()
        losses.append(float(loss.detach().cpu()))
        masked_counts.append(meta["masked_token_count"])
        for n, p in model.named_parameters():
            if p.grad is None:
                continue
            g2 = p.grad.detach().float().cpu().square()
            if n not in grad2:
                grad2[n] = torch.zeros_like(g2)
            grad2[n].add_(g2)
        for r in batch:
            source_mask_counts[str(r.get("source"))] += meta["masked_token_count"] / max(1, len(batch))
    n_batches = max(1, len(losses))
    for n in list(grad2.keys()):
        grad2[n].div_(n_batches)
    info = {
        "device": device,
        "rows_sampled": len(rows),
        "batches": len(losses),
        "batch_size": args.batch_size,
        "max_rows_per_source": args.max_rows_per_source,
        "loss_mean": statistics.mean(losses) if losses else None,
        "loss_by_batch": losses,
        "masked_tokens_total": int(sum(masked_counts)),
        "masked_tokens_mean_per_batch": statistics.mean(masked_counts) if masked_counts else None,
        "source_rows": dict(source_rows),
        "source_mask_count_proxy": dict(source_mask_counts),
        "masking": "bounded approximation of research WWM 0.15 with 80/10/10 replacement on legal 10M rows; used only for local squared-gradient importance",
    }
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return grad2, info


def compare_route(route_label: str, ref: dict[str, torch.Tensor], other: dict[str, torch.Tensor], grad2: dict[str, torch.Tensor], rng: random.Random, max_samples_per_tensor: int) -> dict[str, Any]:
    names = [n for n in stock_common_names(ref, other) if n in grad2]
    groups: dict[str, dict[str, Any]] = collections.defaultdict(lambda: {"n_params": 0, "n_tensors": 0, "grad2_sum": 0.0, "delta2_sum": 0.0, "ref2_sum": 0.0})
    tensor_rows = []
    sampled_g: list[float] = []
    sampled_d: list[float] = []
    sampled_d_shuf: list[float] = []
    for n in names:
        g = grad2[n].flatten().float()
        r = ref[n].flatten().float()
        o = other[n].flatten().float()
        d2 = (o - r).square()
        g_sum = float(g.sum()); d_sum = float(d2.sum()); r_sum = float(r.square().sum())
        numel = int(g.numel())
        gr = groups[group_name(n)]
        gr["n_params"] += numel; gr["n_tensors"] += 1; gr["grad2_sum"] += g_sum; gr["delta2_sum"] += d_sum; gr["ref2_sum"] += r_sum
        tensor_rows.append({
            "route": route_label,
            "tensor": n,
            "group": group_name(n),
            "n_params": numel,
            "grad2_mean": g_sum / max(1, numel),
            "delta2_mean": d_sum / max(1, numel),
            "delta2_rel_to_ref2": d_sum / r_sum if r_sum else None,
            "grad2_sum": g_sum,
            "delta2_sum": d_sum,
        })
        k = min(numel, max_samples_per_tensor)
        if k <= 0:
            continue
        # deterministic sample indices per tensor, without materializing huge permutation for very large tensors.
        if numel <= k:
            idx = torch.arange(numel)
        else:
            # sample with random.Random to keep CPU memory bounded
            idx_list = rng.sample(range(numel), k)
            idx = torch.tensor(idx_list, dtype=torch.long)
        gv = g[idx]
        dv = d2[idx]
        gm = float(g.mean().item()) + 1e-30
        dm = float(d2.mean().item()) + 1e-30
        sampled_g.extend([float(x) for x in torch.log1p(gv / gm).tolist()])
        sampled_d.extend([float(x) for x in torch.log1p(dv / dm).tolist()])
        # null: pair sampled g with an independent within-tensor sampled d.
        if numel <= k:
            shidx = idx[torch.randperm(len(idx), generator=torch.Generator().manual_seed(len(n) + 3))]
        else:
            shidx = torch.tensor(rng.sample(range(numel), k), dtype=torch.long)
        sampled_d_shuf.extend([float(x) for x in torch.log1p(d2[shidx] / dm).tolist()])
    # Normalize groups after sums.
    total_grad = sum(v["grad2_sum"] for v in groups.values())
    total_delta = sum(v["delta2_sum"] for v in groups.values())
    total_ref = sum(v["ref2_sum"] for v in groups.values())
    group_rows = []
    for gname, rec in groups.items():
        row = dict(rec)
        row["group"] = gname
        row["grad2_fraction"] = rec["grad2_sum"] / total_grad if total_grad else 0.0
        row["delta2_fraction"] = rec["delta2_sum"] / total_delta if total_delta else 0.0
        row["param_fraction"] = rec["n_params"] / max(1, sum(v["n_params"] for v in groups.values()))
        row["grad2_per_param"] = rec["grad2_sum"] / max(1, rec["n_params"])
        row["delta2_per_param"] = rec["delta2_sum"] / max(1, rec["n_params"])
        row["delta2_rel_to_ref2"] = rec["delta2_sum"] / rec["ref2_sum"] if rec["ref2_sum"] else None
        row["delta_over_importance_fraction_ratio"] = (row["delta2_fraction"] / row["grad2_fraction"]) if row["grad2_fraction"] else None
        group_rows.append(row)
    group_rows.sort(key=lambda r: r["delta2_fraction"], reverse=True)
    # Deciles of normalized importance versus normalized displacement.
    pairs = sorted(zip(sampled_g, sampled_d, sampled_d_shuf), key=lambda x: x[0])
    deciles = []
    if pairs:
        n = len(pairs)
        for d in range(10):
            lo = d * n // 10; hi = (d + 1) * n // 10 if d < 9 else n
            chunk = pairs[lo:hi]
            deciles.append({
                "grad_decile_low_to_high": d + 1,
                "n": len(chunk),
                "mean_log1p_norm_grad2": statistics.mean(x[0] for x in chunk),
                "mean_log1p_norm_delta2": statistics.mean(x[1] for x in chunk),
                "mean_log1p_norm_delta2_within_tensor_shuffle_null": statistics.mean(x[2] for x in chunk),
            })
    return {
        "route": route_label,
        "n_tensors": len(names),
        "n_params": sum(int(grad2[n].numel()) for n in names),
        "group_rows": group_rows,
        "tensor_rows": sorted(tensor_rows, key=lambda r: r["delta2_sum"], reverse=True),
        "sampled_element_count": len(sampled_g),
        "sampled_log1p_norm_grad_delta_pearson": pearson(sampled_g, sampled_d),
        "sampled_log1p_norm_grad_delta_pearson_within_tensor_shuffle_null": pearson(sampled_g, sampled_d_shuf),
        "importance_deciles": deciles,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = sorted({k for r in rows for k in r.keys()})
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rows)


def run(args) -> dict[str, Any]:
    t0 = time.time()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.torch_threads)
    pool_sha = sha256_file(POOL)
    tok_sha = sha256_file(TOKENIZER_DIR / "tokenizer.json")
    if pool_sha != EXPECTED_POOL_SHA:
        raise RuntimeError(f"pool SHA mismatch {pool_sha}")
    if tok_sha != EXPECTED_TOKENIZER_SHA:
        raise RuntimeError(f"tokenizer SHA mismatch {tok_sha}")
    grad2, grad_info = estimate_importance(args)
    print(json.dumps({"event": "grad_importance_done", **{k: grad_info[k] for k in ["rows_sampled", "batches", "loss_mean", "masked_tokens_total"]}}), flush=True)
    ref100 = load_file(str(model_safetensors(research, "100M")), device="cpu")
    ref80 = load_file(str(model_safetensors(research, "80M")), device="cpu")
    scale100 = load_file(str(model_safetensors(SCALE, "100M")), device="cpu")
    u256100 = load_file(str(model_safetensors(U256, "100M")), device="cpu")
    rng = random.Random(args.seed + 99)
    route_results = []
    route_results.append(compare_route("scale1p75_100M_stock_minus_step35_100M", ref100, scale100, grad2, rng, args.max_samples_per_tensor))
    route_results.append(compare_route("u256_100M_minus_step35_100M", ref100, u256100, grad2, rng, args.max_samples_per_tensor))
    route_results.append(compare_route("reference_100M_minus_step35_80M_normal_late_movement", ref80, ref100, grad2, rng, args.max_samples_per_tensor))
    # Write tables.
    all_groups = []
    all_tensors = []
    all_deciles = []
    for rr in route_results:
        for r in rr["group_rows"]:
            all_groups.append({"route": rr["route"], **r})
        all_tensors.extend(rr["tensor_rows"][:200])
        for d in rr["importance_deciles"]:
            all_deciles.append({"route": rr["route"], **d})
    group_csv = out_dir / "importance_displacement_group_rows.csv"
    tensor_csv = out_dir / "importance_displacement_top_tensor_rows.csv"
    decile_csv = out_dir / "importance_displacement_deciles.csv"
    write_csv(group_csv, all_groups)
    write_csv(tensor_csv, all_tensors)
    write_csv(decile_csv, all_deciles)
    result = {
        "status": "IMPORTANCE_DISPLACEMENT_RELATION",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "purpose": "Estimate research legal-corpus MLM squared-gradient importance and compare it with endpoint displacements; interpret only as MLM-sensitive displacement evidence.",
        "legal_and_runtime": {
            "uses_eval_labels": False,
            "uses_external_text": False,
            "updates_model_parameters": False,
            "pool": str(POOL),
            "pool_sha256": pool_sha,
            "tokenizer_sha256": tok_sha,
            "checkpoint": str(research / "chck_100M"),
            "device": grad_info["device"],
            "hf_modules_cache": os.environ.get("HF_MODULES_CACHE"),
        },
        "args": vars(args),
        "gradient_importance": grad_info,
        "routes": route_results,
        "outputs": {"group_csv": str(group_csv), "top_tensor_csv": str(tensor_csv), "decile_csv": str(decile_csv)},
        "interpretation_limits": "Squared-gradient alignment identifies directions to which the research MLM loss is locally sensitive on a bounded legal-corpus sample. It cannot by itself establish which representations preserve BabyLM broad competence or authorize a long training route.",
        "elapsed_sec": round(time.time() - t0, 3),
    }
    out_json = out_dir / "importance_displacement_relation.json"
    out_md = out_dir / "importance_displacement_relation.md"
    out_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# research — squared-gradient importance vs endpoint displacement",
        "",
        "This bounded analysis uses legal-corpus MLM gradients from the frozen research endpoint. It is only evidence about MLM-sensitive displacement.",
        "",
        "## Gradient sample",
        f"- rows sampled: `{grad_info['rows_sampled']}`; batches `{grad_info['batches']}`; mean loss `{grad_info['loss_mean']}`; masked tokens `{grad_info['masked_tokens_total']}`.",
        f"- source rows: `{grad_info['source_rows']}`.",
        "",
        "## Element-level normalized grad²/displacement relation",
        "| route | sampled params | pearson(log grad², log delta²) | within-tensor shuffled null |",
        "|---|---:|---:|---:|",
    ]
    for rr in route_results:
        lines.append(f"| {rr['route']} | {rr['sampled_element_count']} | {rr['sampled_log1p_norm_grad_delta_pearson']} | {rr['sampled_log1p_norm_grad_delta_pearson_within_tensor_shuffle_null']} |")
    lines += ["", "## Top group movement/intensity", "", "| route | group | delta2_fraction | grad2_fraction | delta/importance ratio | delta2_rel_to_ref2 |", "|---|---|---:|---:|---:|---:|"]
    for rr in route_results:
        for r in rr["group_rows"][:8]:
            lines.append(f"| {rr['route']} | {r['group']} | {r['delta2_fraction']} | {r['grad2_fraction']} | {r['delta_over_importance_fraction_ratio']} | {r['delta2_rel_to_ref2']} |")
    lines += [
        "",
        "## Scientific reading",
        "Compare closed-route displacement to normal research late movement. If a closed route's displacement is not more enriched in high-MLM-importance directions than normal late movement, a simple importance-damping mechanism is weak; if it is enriched, protection may be mechanistically motivated but still needs token-structure and score-sentinel evidence.",
        "",
        f"Group CSV: `{group_csv}`",
        f"Decile CSV: `{decile_csv}`",
        f"JSON: `{out_json}`",
    ]
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": result["status"],
        "out_json": str(out_json),
        "out_md": str(out_md),
        "rows_sampled": grad_info["rows_sampled"],
        "batches": grad_info["batches"],
        "route_pearsons": {rr["route"]: rr["sampled_log1p_norm_grad_delta_pearson"] for rr in route_results},
        "null_pearsons": {rr["route"]: rr["sampled_log1p_norm_grad_delta_pearson_within_tensor_shuffle_null"] for rr in route_results},
        "elapsed_sec": result["elapsed_sec"],
    }, indent=2), flush=True)
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(WORKSPACE / "data/importance_displacement_relation"))
    ap.add_argument("--max-rows-per-source", type=int, default=64)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--mask-prob", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=8122)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--torch-threads", type=int, default=8)
    ap.add_argument("--max-samples-per-tensor", type=int, default=20000)
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
