#!/usr/bin/env python3
"""research: leakage-stripped paired-view private-channel probe.

Purpose
-------
The previous pair-geometry probes encoded the packed training row containing both the
source sentence and its compact rewrite.  With a bidirectional MLM encoder, the source
span can attend to the rewrite span and vice versa, so high source<->rewrite retrieval
may partly reflect co-present text rather than a separately acquired invariant.

This script encodes each source sentence and each compact rewrite as separate inputs
with the frozen research legal checkpoint.  It then trains only a small detached private
projection on source/rewrite pooled representations.  The language model backbone and
head are never updated and the learned channel is not connected to logits, so the
reference function is preserved exactly during this test.  The scientific question is
whether held-out true pairs beat same-row decoy rewrites after removing adjacency
leakage, and whether a private low-dimensional channel can acquire extra true-pair
structure without touching the shared backbone.

No pretraining, official evaluation, corpus change, tokenizer change, or model save is
performed.
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
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any

import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

EXPECTED_POOL_SHA = "215944978157394dbecf2039f2c1e1806bfcbb9701421920f78605eb58975a23"
EXPECTED_TOKENIZER_SHA = "91b775514b9f4e2d1f37c28445ab98181007e16d14f763955168547e293ee8f9"


def find_user_root() -> pathlib.Path:
    return _PUBLIC_ROOT


USER_ROOT = find_user_root()
WORKSPACE = USER_ROOT / "experiments/archive/frontier_consolidation"
PAIR_JSONL = WORKSPACE / "data/density_core_reinvestment_medium_riskhard/selected_compact_reinvest_pairs.jsonl"
ROW_META_JSONL = WORKSPACE / "data/density_cleanqwen_overlay_medium_riskhard/cleanqwen_fineweb_compact_view_reinvest_changed_block_rows_meta.jsonl"
POOL_10M = WORKSPACE / "data/density_cleanqwen_overlay_medium_riskhard/cleanqwen_fineweb_compact_view_reinvest_10M.jsonl"
TOKENIZER_DIR = WORKSPACE / "data/compliant_tokenizer"
CKPT = WORKSPACE / "training/runs/complianttok_reinvest_seed43022_r2/hf_model/chck_100M"
OUT_DIR = WORKSPACE / "data/separate_pair_private_channel_probe"


@dataclass
class PairRec:
    pair_id: str
    row_index: int
    example_id: int
    source_text: str
    rewrite_text: str
    source_words: int
    rewrite_words: int
    doc_id: str
    sentence_id: str
    content_overlap: float
    content_recall: float
    entity_recall: float
    number_recall: float
    domain_hits: tuple[str, ...]


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def norm_pair_id(pid: str) -> str:
    return str(pid).split(":", 1)[-1] if str(pid).startswith("compact:") else str(pid)


def load_rows() -> dict[str, tuple[int, int]]:
    out: dict[str, tuple[int, int]] = {}
    with ROW_META_JSONL.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            row_index = int(obj["row_index"])
            example_id = int(obj["example_id"])
            for pid in obj.get("pair_ids") or []:
                out[norm_pair_id(pid)] = (row_index, example_id)
    return out


def load_pairs() -> list[PairRec]:
    row_of = load_rows()
    pairs: list[PairRec] = []
    missing_rows = 0
    with PAIR_JSONL.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            pid = norm_pair_id(obj["pair_id"])
            if pid not in row_of:
                missing_rows += 1
                continue
            row_index, example_id = row_of[pid]
            pairs.append(PairRec(
                pair_id=pid,
                row_index=row_index,
                example_id=example_id,
                source_text=str(obj["source_text"]),
                rewrite_text=str(obj["rewrite_text"]),
                source_words=int(obj.get("source_words", len(str(obj["source_text"]).split()))),
                rewrite_words=int(obj.get("rewrite_words", len(str(obj["rewrite_text"]).split()))),
                doc_id=str(obj.get("doc_id", "")),
                sentence_id=str(obj.get("sentence_id", "")),
                content_overlap=float(obj.get("content_overlap", math.nan)),
                content_recall=float(obj.get("content_recall", math.nan)),
                entity_recall=float(obj.get("entity_recall", math.nan)),
                number_recall=float(obj.get("number_recall", math.nan)),
                domain_hits=tuple(obj.get("domain_hits") or []),
            ))
    if missing_rows:
        raise RuntimeError(f"missing row metadata for {missing_rows} pairs")
    pairs.sort(key=lambda r: (r.row_index, r.pair_id))
    return pairs


def select_rows(pairs: list[PairRec], max_rows: int, seed: int, low_overlap_focus: bool) -> list[PairRec]:
    by_row: dict[int, list[PairRec]] = defaultdict(list)
    for p in pairs:
        by_row[p.row_index].append(p)
    eligible_rows = [r for r, ps in by_row.items() if len(ps) >= 2]
    rng = random.Random(seed)
    if low_overlap_focus:
        # Prefer rows containing at least one strongly compressed/low-overlap pair, then fill randomly.
        scored = []
        for r in eligible_rows:
            vals = [p.content_overlap for p in by_row[r] if math.isfinite(p.content_overlap)]
            score = min(vals) if vals else 1.0
            scored.append((score, r))
        scored.sort(key=lambda x: (x[0], x[1]))
        head_n = min(len(scored), max_rows)
        selected_rows = [r for _, r in scored[:head_n]]
        rng.shuffle(selected_rows)
    else:
        selected_rows = eligible_rows[:]
        rng.shuffle(selected_rows)
        selected_rows = selected_rows[:max_rows]
    selected = []
    for r in sorted(selected_rows):
        selected.extend(by_row[r])
    return selected


def split_by_row(pairs: list[PairRec], train_frac: float, seed: int) -> tuple[list[int], list[int]]:
    rows = sorted(set(p.row_index for p in pairs))
    rng = random.Random(seed)
    rng.shuffle(rows)
    n_train = max(1, min(len(rows) - 1, int(round(len(rows) * train_frac))))
    train_rows = set(rows[:n_train])
    train_idx, test_idx = [], []
    for i, p in enumerate(pairs):
        (train_idx if p.row_index in train_rows else test_idx).append(i)
    return train_idx, test_idx


def encode_texts(model, tokenizer, texts: list[str], batch_size: int, max_length: int, device: str) -> torch.Tensor:
    outs: list[torch.Tensor] = []
    special = set(int(x) for x in tokenizer.all_special_ids)
    model.eval()
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch = texts[start:start + batch_size]
            enc = tokenizer(batch, add_special_tokens=True, truncation=True, max_length=max_length, padding=True, return_tensors="pt")
            input_ids = enc["input_ids"].to(device)
            attention = enc["attention_mask"].to(device)
            out = model(input_ids=input_ids, attention_mask=attention, output_hidden_states=True, return_dict=True)
            h = out.hidden_states[-1].float()
            # Pool non-pad, non-special tokens.  Fall back to attention mask if a very short text leaves no nonspecial token.
            nonspecial = attention.bool()
            for sid in special:
                nonspecial = nonspecial & (input_ids != sid)
            fallback = nonspecial.sum(dim=1) == 0
            if fallback.any():
                nonspecial[fallback] = attention.bool()[fallback]
            denom = nonspecial.sum(dim=1).clamp_min(1).view(-1, 1).float()
            pooled = (h * nonspecial.unsqueeze(-1).float()).sum(dim=1) / denom
            outs.append(pooled.cpu())
    return torch.cat(outs, dim=0)


def normalize(x: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.normalize(x.float(), dim=1, eps=1e-6)


def centered(x: torch.Tensor, train_idx: list[int]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mu = x[train_idx].mean(dim=0, keepdim=True)
    std = x[train_idx].std(dim=0, keepdim=True).clamp_min(1e-4)
    return (x - mu) / std, mu.squeeze(0), std.squeeze(0)


def summarize(vals: list[float]) -> dict[str, Any]:
    vals = [float(v) for v in vals if math.isfinite(float(v))]
    if not vals:
        return {"n": 0, "mean": None, "median": None, "p05": None, "p95": None, "min": None, "max": None, "std": None}
    s = sorted(vals)
    def q(frac: float) -> float:
        if len(s) == 1:
            return s[0]
        pos = frac * (len(s) - 1)
        lo = int(math.floor(pos)); hi = int(math.ceil(pos))
        if lo == hi:
            return s[lo]
        return s[lo] * (hi - pos) + s[hi] * (pos - lo)
    return {"n": len(vals), "mean": statistics.mean(vals), "median": statistics.median(vals), "p05": q(0.05), "p95": q(0.95), "min": min(vals), "max": max(vals), "std": statistics.pstdev(vals)}


def pair_metrics(src: torch.Tensor, rew: torch.Tensor, pairs: list[PairRec], idxs: list[int], label: str) -> dict[str, Any]:
    S = normalize(src[idxs])
    R = normalize(rew[idxs])
    idx_to_local = {idx: k for k, idx in enumerate(idxs)}
    by_row: dict[int, list[int]] = defaultdict(list)
    for idx in idxs:
        by_row[pairs[idx].row_index].append(idx)
    true_cos = (S * R).sum(dim=1)
    rand_cos = []
    same_decoy_vals = []
    top1 = 0
    ranks = []
    margins = []
    n_with_decoy = 0
    for local_i, idx in enumerate(idxs):
        members = [j for j in by_row[pairs[idx].row_index] if j != idx and j in idx_to_local]
        # deterministic random-other within the split
        rand_local = (local_i + 1) % len(idxs)
        rand_cos.append(float((S[local_i] * R[rand_local]).sum().item()))
        if not members:
            continue
        n_with_decoy += 1
        cand_locals = [idx_to_local[j] for j in members]
        d = torch.mv(R[cand_locals], S[local_i])
        same_decoy_vals.extend([float(v) for v in d.cpu().tolist()])
        tv = float(true_cos[local_i].item())
        max_decoy = float(d.max().item())
        margins.append(tv - max_decoy)
        rank = 1 + int((d > tv).sum().item())
        ranks.append(rank)
        if rank == 1:
            top1 += 1
    # Global retrieval among all split rewrites.
    sim = S @ R.t()
    n = sim.shape[0]
    global_ranks = []
    for start in range(0, n, 512):
        end = min(n, start + 512)
        block = sim[start:end]
        diag = block[torch.arange(end - start), torch.arange(start, end)].unsqueeze(1)
        rr = (block > diag).sum(dim=1) + 1
        global_ranks.extend(int(v) for v in rr.cpu().tolist())
    return {
        "label": label,
        "n_pairs": len(idxs),
        "n_rows": len(by_row),
        "true_cos": summarize([float(v) for v in true_cos.cpu().tolist()]),
        "random_other_cos": summarize(rand_cos),
        "same_row_decoy_cos": summarize(same_decoy_vals),
        "same_row_true_minus_max_decoy": summarize(margins),
        "same_row_top1": (top1 / n_with_decoy) if n_with_decoy else None,
        "same_row_mean_rank": statistics.mean(ranks) if ranks else None,
        "global_top1": sum(1 for r in global_ranks if r == 1) / len(global_ranks) if global_ranks else None,
        "global_top5": sum(1 for r in global_ranks if r <= 5) / len(global_ranks) if global_ranks else None,
        "global_mean_rank": statistics.mean(global_ranks) if global_ranks else None,
    }


class DualProjector(torch.nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.src = torch.nn.Linear(in_dim, out_dim, bias=False)
        self.rew = torch.nn.Linear(in_dim, out_dim, bias=False)
        torch.nn.init.orthogonal_(self.src.weight)
        torch.nn.init.orthogonal_(self.rew.weight)

    def forward(self, s: torch.Tensor, r: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return normalize(self.src(s)), normalize(self.rew(r))


def train_private_channel(src: torch.Tensor, rew: torch.Tensor, pairs: list[PairRec], train_idx: list[int], test_idx: list[int], args) -> tuple[dict[str, Any], torch.Tensor, torch.Tensor]:
    device = torch.device("cpu")
    # Standardize from train only; the projector never sees LM parameters.
    src_z, _, _ = centered(src, train_idx)
    rew_z, _, _ = centered(rew, train_idx)
    S_train = src_z[train_idx].to(device)
    R_train = rew_z[train_idx].to(device)
    model = DualProjector(src_z.shape[1], args.proj_dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.proj_lr, weight_decay=args.proj_wd)
    row_members_local: dict[int, list[int]] = defaultdict(list)
    for li, gi in enumerate(train_idx):
        row_members_local[pairs[gi].row_index].append(li)
    hard_pairs: list[tuple[int, list[int]]] = []
    for li, gi in enumerate(train_idx):
        decoys = [j for j in row_members_local[pairs[gi].row_index] if j != li]
        if decoys:
            hard_pairs.append((li, decoys))
    history = []
    gen = torch.Generator(device="cpu")
    gen.manual_seed(args.seed + 17)
    n = S_train.shape[0]
    for epoch in range(args.proj_epochs):
        perm = torch.randperm(n, generator=gen)
        total = 0.0
        nb = 0
        for start in range(0, n, args.proj_batch):
            ids = perm[start:start + args.proj_batch]
            s = S_train[ids]
            r = R_train[ids]
            ps, pr = model(s, r)
            logits = ps @ pr.t() / args.temperature
            targets = torch.arange(logits.shape[0], device=device)
            loss = 0.5 * (torch.nn.functional.cross_entropy(logits, targets) + torch.nn.functional.cross_entropy(logits.t(), targets))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += float(loss.detach())
            nb += 1
        if hard_pairs and (epoch % args.hard_every == 0 or epoch == args.proj_epochs - 1):
            # Small hard same-row margin pass on all train rows; updates only private projector.
            ps_all, pr_all = model(S_train, R_train)
            hard_losses = []
            for li, decoys in hard_pairs:
                true = (ps_all[li] * pr_all[li]).sum()
                d = torch.mv(pr_all[decoys], ps_all[li])
                hard_losses.append(torch.relu(args.hard_margin - true + d.max()))
            hard_loss = torch.stack(hard_losses).mean()
            opt.zero_grad(set_to_none=True)
            (args.hard_weight * hard_loss).backward()
            opt.step()
            hard_val = float(hard_loss.detach())
        else:
            hard_val = None
        if epoch in {0, 1, 2, 4, 9, args.proj_epochs - 1} or (epoch + 1) % 20 == 0:
            history.append({"epoch": epoch + 1, "nce_loss_mean": total / max(1, nb), "hard_margin_loss": hard_val})
    with torch.no_grad():
        psrc_all, prew_all = model(src_z.to(device), rew_z.to(device))
    train_metrics = pair_metrics(psrc_all.cpu(), prew_all.cpu(), pairs, train_idx, "private_projected_train")
    test_metrics = pair_metrics(psrc_all.cpu(), prew_all.cpu(), pairs, test_idx, "private_projected_test")
    summary = {
        "projector": "dual_linear_detached_private_channel",
        "proj_dim": args.proj_dim,
        "temperature": args.temperature,
        "epochs": args.proj_epochs,
        "train_pairs": len(train_idx),
        "test_pairs": len(test_idx),
        "hard_same_row_pairs": len(hard_pairs),
        "history": history,
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "backbone_updated": False,
        "lm_logits_connected_to_private_channel": False,
        "function_preservation": "Exact during this test: source/rewrite embeddings were detached from a frozen research checkpoint and the learned private channel is not connected to the MLM head.",
    }
    return summary, psrc_all.cpu(), prew_all.cpu()


def domain_counts(pairs: list[PairRec]) -> dict[str, int]:
    c = Counter()
    for p in pairs:
        if p.domain_hits:
            for d in p.domain_hits:
                c[d] += 1
        else:
            c["no_domain"] += 1
    return dict(c.most_common())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    ap.add_argument("--max-rows", type=int, default=1200)
    ap.add_argument("--seed", type=int, default=8120)
    ap.add_argument("--train-frac", type=float, default=0.70)
    ap.add_argument("--max-length", type=int, default=128)
    ap.add_argument("--encode-batch", type=int, default=64)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--low-overlap-focus", action="store_true")
    ap.add_argument("--proj-dim", type=int, default=64)
    ap.add_argument("--proj-epochs", type=int, default=80)
    ap.add_argument("--proj-batch", type=int, default=256)
    ap.add_argument("--proj-lr", type=float, default=3e-3)
    ap.add_argument("--proj-wd", type=float, default=1e-4)
    ap.add_argument("--temperature", type=float, default=0.07)
    ap.add_argument("--hard-margin", type=float, default=0.10)
    ap.add_argument("--hard-weight", type=float, default=0.30)
    ap.add_argument("--hard-every", type=int, default=2)
    args = ap.parse_args()
    t0 = time.time()
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pool_sha = sha256_file(POOL_10M)
    tok_sha = sha256_file(TOKENIZER_DIR / "tokenizer.json")
    if pool_sha != EXPECTED_POOL_SHA:
        raise RuntimeError(f"pool SHA mismatch: {pool_sha}")
    if tok_sha != EXPECTED_TOKENIZER_SHA:
        raise RuntimeError(f"tokenizer SHA mismatch: {tok_sha}")

    all_pairs = load_pairs()
    selected_pairs = select_rows(all_pairs, args.max_rows, args.seed, args.low_overlap_focus)
    train_idx, test_idx = split_by_row(selected_pairs, args.train_frac, args.seed + 1)
    row_counts = Counter(p.row_index for p in selected_pairs)
    dev = "cuda" if (args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available())) else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER_DIR), use_fast=True)
    model = AutoModelForMaskedLM.from_pretrained(str(CKPT))
    model.to(dev)
    model.eval()
    source_embeddings = encode_texts(model, tokenizer, [p.source_text for p in selected_pairs], args.encode_batch, args.max_length, dev)
    rewrite_embeddings = encode_texts(model, tokenizer, [p.rewrite_text for p in selected_pairs], args.encode_batch, args.max_length, dev)
    del model
    if dev == "cuda":
        torch.cuda.empty_cache()

    raw_src_z, _, _ = centered(source_embeddings, train_idx)
    raw_rew_z, _, _ = centered(rewrite_embeddings, train_idx)
    raw_train = pair_metrics(raw_src_z, raw_rew_z, selected_pairs, train_idx, "separate_identity_train")
    raw_test = pair_metrics(raw_src_z, raw_rew_z, selected_pairs, test_idx, "separate_identity_test")
    priv, psrc, prew = train_private_channel(source_embeddings, rewrite_embeddings, selected_pairs, train_idx, test_idx, args)

    # Evaluate a deliberately wrong control: shuffle rewrite partners within train/test rows. If private channel
    # simply encodes row/topic identity, true-vs-same-row margins will not exceed this by much.
    test_private = priv["test_metrics"]
    improvement = {
        "same_row_top1_private_minus_identity_test": None if raw_test["same_row_top1"] is None else test_private["same_row_top1"] - raw_test["same_row_top1"],
        "same_row_margin_private_minus_identity_test": None if raw_test["same_row_true_minus_max_decoy"]["mean"] is None else test_private["same_row_true_minus_max_decoy"]["mean"] - raw_test["same_row_true_minus_max_decoy"]["mean"],
        "global_top1_private_minus_identity_test": None if raw_test["global_top1"] is None else test_private["global_top1"] - raw_test["global_top1"],
    }
    route_readout = {
        "private_channel_signal_present": bool(
            test_private["same_row_top1"] is not None and test_private["same_row_top1"] >= max(0.60, raw_test["same_row_top1"] - 0.02)
            and test_private["same_row_true_minus_max_decoy"]["mean"] is not None
            and test_private["same_row_true_minus_max_decoy"]["mean"] > 0.02
        ),
        "private_channel_adds_over_identity": bool(
            improvement["same_row_margin_private_minus_identity_test"] is not None
            and improvement["same_row_margin_private_minus_identity_test"] > 0.01
        ),
        "interpretation": "A positive result supports a frozen/slow backbone plus private pair channel as a learnable corpus-derived signal; a non-positive result argues against spending a full run on paired-view alignment without a richer architecture or better supervision.",
    }

    result = {
        "status": "SEPARATE_PAIR_PRIVATE_CHANNEL_PROBE",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "purpose": "Remove packed-row attention leakage and test whether legal compact source/rewrite pairs support a detached private invariant channel while preserving the research LM function.",
        "inputs": {
            "pair_jsonl": str(PAIR_JSONL),
            "row_meta_jsonl": str(ROW_META_JSONL),
            "pool_10m": str(POOL_10M),
            "pool_sha256": pool_sha,
            "tokenizer": str(TOKENIZER_DIR),
            "tokenizer_sha256": tok_sha,
            "checkpoint": str(CKPT),
        },
        "args": vars(args),
        "sample": {
            "all_pairs": len(all_pairs),
            "selected_pairs": len(selected_pairs),
            "selected_rows": len(row_counts),
            "row_pair_count_stats": summarize([float(v) for v in row_counts.values()]),
            "train_rows": len(set(selected_pairs[i].row_index for i in train_idx)),
            "test_rows": len(set(selected_pairs[i].row_index for i in test_idx)),
            "train_pairs": len(train_idx),
            "test_pairs": len(test_idx),
            "domain_counts": domain_counts(selected_pairs),
            "content_overlap": summarize([p.content_overlap for p in selected_pairs]),
            "content_recall": summarize([p.content_recall for p in selected_pairs]),
        },
        "encoding": {
            "device": dev,
            "embedding_shape": list(source_embeddings.shape),
            "max_length": args.max_length,
            "source_norms": summarize([float(v) for v in source_embeddings.norm(dim=1).tolist()]),
            "rewrite_norms": summarize([float(v) for v in rewrite_embeddings.norm(dim=1).tolist()]),
            "packed_row_partner_attention_removed": True,
        },
        "identity_metrics": {"train": raw_train, "test": raw_test},
        "private_channel": priv,
        "private_vs_identity": improvement,
        "route_readout": route_readout,
        "elapsed_sec": round(time.time() - t0, 3),
    }
    out_json = out_dir / "separate_pair_private_channel_probe.json"
    out_md = out_dir / "separate_pair_private_channel_probe.md"
    out_json.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    lines = [
        "# research — separate-encoding paired-view private channel",
        "",
        "Each source sentence and compact rewrite was encoded as a separate input by the frozen research legal checkpoint. This removes same-row bidirectional attention to the partner text before measuring or learning source/rewrite alignment.",
        "",
        "## Legal corpus substrate",
        f"- selected pairs: `{len(selected_pairs)}` from `{len(row_counts)}` packed changed rows; train/test split by row = `{len(set(selected_pairs[i].row_index for i in train_idx))}`/`{len(set(selected_pairs[i].row_index for i in test_idx))}` rows.",
        f"- pool SHA: `{pool_sha}`; tokenizer SHA: `{tok_sha}`.",
        f"- content_overlap mean: `{result['sample']['content_overlap']['mean']:.4f}`; content_recall mean: `{result['sample']['content_recall']['mean']:.4f}`.",
        "",
        "## Held-out same-row retrieval after removing partner attention",
        "| Channel | same-row top1 | true - max same-row decoy | global top1 | global top5 |",
        "|---|---:|---:|---:|---:|",
        f"| identity frozen reps | {raw_test['same_row_top1']:.4f} | {raw_test['same_row_true_minus_max_decoy']['mean']:.4f} | {raw_test['global_top1']:.4f} | {raw_test['global_top5']:.4f} |",
        f"| detached private projection | {test_private['same_row_top1']:.4f} | {test_private['same_row_true_minus_max_decoy']['mean']:.4f} | {test_private['global_top1']:.4f} | {test_private['global_top5']:.4f} |",
        "",
        "## Private-channel readout",
        f"- same-row margin improvement over identity: `{improvement['same_row_margin_private_minus_identity_test']}`.",
        f"- same-row top1 improvement over identity: `{improvement['same_row_top1_private_minus_identity_test']}`.",
        f"- private_channel_signal_present: `{route_readout['private_channel_signal_present']}`.",
        f"- private_channel_adds_over_identity: `{route_readout['private_channel_adds_over_identity']}`.",
        "- Backbone preservation in this test is exact: the research model is frozen, embeddings are detached, and the private channel is not connected to LM logits.",
        "",
        "## Scientific use",
        "This is not BabyLM score evidence. It is a corpus-derived test of whether a private fast pathway can learn paired-view structure without displacing the protected language function. A full training run is not justified by this probe alone.",
        "",
        f"JSON: `{out_json}`",
    ]
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": result["status"],
        "out_json": str(out_json),
        "out_md": str(out_md),
        "selected_pairs": len(selected_pairs),
        "selected_rows": len(row_counts),
        "identity_test_same_row_top1": raw_test["same_row_top1"],
        "private_test_same_row_top1": test_private["same_row_top1"],
        "identity_test_margin": raw_test["same_row_true_minus_max_decoy"]["mean"],
        "private_test_margin": test_private["same_row_true_minus_max_decoy"]["mean"],
        "route_readout": route_readout,
        "elapsed_sec": result["elapsed_sec"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
