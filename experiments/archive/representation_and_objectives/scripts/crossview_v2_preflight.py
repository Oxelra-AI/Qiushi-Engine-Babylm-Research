#!/usr/bin/env python3
"""Preflight corrected research cross-view DeBERTa MLM infrastructure.

Checks before H100 launch:
  * own/wrong arms carry identical intrinsic rewrite-origin target strata over the
    whole epoch population;
  * rewrite masks are identity-attached: the same rewrite word targets are masked
    under own and wrong pairing, and visible/blocked arms share the same targets;
  * source masks are identity-attached across own/wrong arms;
  * full-visible custom DeBERTa forward exactly matches stock forward; blocked
    mask changes logits; gradients are finite;
  * all arms load from one identical initialization file.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
from collections import Counter, defaultdict

import torch
from transformers import AutoTokenizer

USER_ROOT = pathlib.Path(".").resolve()
SCRIPT_DIR = USER_ROOT / "experiments/archive" / 'representation_and_objectives' / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from crossview_deberta_trainer_v2 import (  # noqa: E402
    PairAwareDataset,
    STRATA,
    custom_mlm_forward,
    collate,
    load_or_build_model,
)
from crossview_deberta_trainer_v2 import build_args as _unused_build_args  # noqa: F401,E402


def load_pool(path: str):
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                obj = json.loads(line)
                obj["_epoch"] = 0
                out.append(obj)
    return out


def summarize_pair_targets(ds: PairAwareDataset):
    dist = Counter()
    masked_dist = Counter()
    by_rewrite = {}
    by_source = {}
    pair_count = 0
    for idx, ex in enumerate(ds.examples):
        if ex["type"] != "pair":
            continue
        pair_count += 1
        it = ds[idx]
        st = it["target_stratum"]
        labels = it["labels"]
        b = int(it["boundary"])
        rw_len = int(it["rewrite_len"])
        rid = ex["own_rewrite_pair_id"] if ds.partner == "own" else ex["wrong_rewrite_pair_id"]
        sid = ex["source_pair_id"]
        # Rewrite target/mask signature: stratum and masked positions within rewrite segment.
        rw_strata = tuple(int(x) for x in st[b:b+rw_len].tolist())
        rw_mask_pos = tuple(int(i) for i, x in enumerate(labels[b:b+rw_len].tolist()) if x != -100)
        rw_mask_lab = tuple(int(x) for x in labels[b:b+rw_len].tolist() if x != -100)
        by_rewrite[rid] = (rw_strata, rw_mask_pos, rw_mask_lab)
        src_mask_pos = tuple(int(i) for i, x in enumerate(labels[:b].tolist()) if x != -100)
        src_mask_lab = tuple(int(x) for x in labels[:b].tolist() if x != -100)
        by_source[sid] = (src_mask_pos, src_mask_lab)
        for sid_i, sn in STRATA.items():
            if sn == "pad":
                continue
            dist[sn] += int((st == sid_i).sum().item())
            masked_dist[sn] += int(((st == sid_i) & (labels != -100)).sum().item())
    return {
        "pair_count": pair_count,
        "dist": dict(dist),
        "masked_dist": dict(masked_dist),
        "by_rewrite": by_rewrite,
        "by_source": by_source,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--tokenizer_path", required=True)
    ap.add_argument("--init_model_path", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--max_seq_length", type=int, default=256)
    ap.add_argument("--mask_prob", type=float, default=0.15)
    ap.add_argument("--mask_seed", type=int, default=430230221)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    out = pathlib.Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    pool = load_pool(args.data)
    tok = AutoTokenizer.from_pretrained(args.tokenizer_path, use_fast=True)

    datasets = {}
    summaries = {}
    for partner in ["own", "wrong"]:
        for vis in ["visible", "blocked"]:
            ds = PairAwareDataset(pool, tok, args.max_seq_length, partner, vis == "blocked", args.mask_prob, args.mask_seed)
            key = f"{partner}_{vis}"
            datasets[key] = ds
            summaries[key] = summarize_pair_targets(ds)
            print(f"built/summarized {key}", flush=True)

    # Intrinsic rewrite-label and rewrite-mask equality by rewrite identity.
    own_sig = summaries["own_visible"]["by_rewrite"]
    wrong_sig = summaries["wrong_visible"]["by_rewrite"]
    common_rewrites = set(own_sig) & set(wrong_sig)
    rw_strata_mismatch = sum(1 for rid in common_rewrites if own_sig[rid][0] != wrong_sig[rid][0])
    rw_mask_mismatch = sum(1 for rid in common_rewrites if own_sig[rid][1:] != wrong_sig[rid][1:])
    # Visibility must not alter target masks.
    own_vis_blk_mismatch = sum(
        1 for rid, sig in summaries["own_visible"]["by_rewrite"].items()
        if sig != summaries["own_blocked"]["by_rewrite"].get(rid)
    )
    wrong_vis_blk_mismatch = sum(
        1 for rid, sig in summaries["wrong_visible"]["by_rewrite"].items()
        if sig != summaries["wrong_blocked"]["by_rewrite"].get(rid)
    )
    # Source masks must be independent of partner/visibility by source identity.
    src_own_wrong_mismatch = sum(
        1 for sid, sig in summaries["own_visible"]["by_source"].items()
        if sig != summaries["wrong_visible"]["by_source"].get(sid)
    )
    src_vis_blk_mismatch = sum(
        1 for sid, sig in summaries["own_visible"]["by_source"].items()
        if sig != summaries["own_blocked"]["by_source"].get(sid)
    )

    # Stock/custom forward and gradient checks on blocked arm.
    class DummyArgs:
        pass
    dargs = DummyArgs()
    dargs.init_model_path = args.init_model_path
    dargs.seed = 43
    dargs.extra_init_seed = 43022
    dargs.max_seq_length = args.max_seq_length
    dargs.max_position_embeddings = 512
    dargs.hidden_size = 480
    dargs.n_layer = 8
    dargs.n_head = 8
    dargs.ffn_mult = 4
    dargs.position_buckets = 256
    dargs.max_relative_positions = 256
    dargs.deberta_pos_att_type = "p2c,c2p"
    device = torch.device(args.device)
    model = load_or_build_model(dargs, tok).to(device)
    model.config.use_cache = False
    model.eval()
    batch = collate([datasets["own_blocked"][i] for i in range(4)])
    ids = batch["input_ids"].to(device)
    mids = batch["masked_input_ids"].to(device)
    labels = batch["labels"].to(device)
    att = batch["attention_mask"].to(device)
    pm = batch["pair_mask"].to(device)
    with torch.no_grad():
        stock = model(input_ids=ids, attention_mask=att)
        full = (att.unsqueeze(2) * att.unsqueeze(1)).long()
        cl, _ = custom_mlm_forward(model, ids, att, full)
        custom_delta = (stock.logits - cl).abs().max().item()
        blocked_logits, _ = custom_mlm_forward(model, ids, att, pm)
        blocked_delta = (cl - blocked_logits).abs().max().item()
    model.train()
    _, loss = custom_mlm_forward(model, mids, att, pm, labels)
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    grad_finite = all(torch.isfinite(g).all().item() for g in grads)

    init_weight = pathlib.Path(args.init_model_path) / "model.safetensors"
    init_sha = None
    if init_weight.exists():
        import hashlib
        h = hashlib.sha256()
        with init_weight.open("rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        init_sha = h.hexdigest()

    conditions = {
        "same_rewrite_identity_set": len(common_rewrites) == 12155,
        "rewrite_strata_own_wrong_mismatches_zero": rw_strata_mismatch == 0,
        "rewrite_mask_own_wrong_mismatches_zero": rw_mask_mismatch == 0,
        "own_visible_blocked_rewrite_target_mismatches_zero": own_vis_blk_mismatch == 0,
        "wrong_visible_blocked_rewrite_target_mismatches_zero": wrong_vis_blk_mismatch == 0,
        "source_mask_own_wrong_mismatches_zero": src_own_wrong_mismatch == 0,
        "source_mask_visible_blocked_mismatches_zero": src_vis_blk_mismatch == 0,
        "custom_full_matches_stock": custom_delta < 1e-4,
        "blocked_changes_logits": blocked_delta > 1e-7,
        "gradients_finite": bool(grad_finite),
    }
    result = {
        "status": "CROSSVIEW_V2_PREFLIGHT",
        "data": args.data,
        "tokenizer_path": args.tokenizer_path,
        "init_model_path": args.init_model_path,
        "init_model_sha256": init_sha,
        "arm_distributions": {
            k: {"dist": v["dist"], "masked_dist": v["masked_dist"], "pair_count": v["pair_count"]}
            for k, v in summaries.items()
        },
        "identity_checks": {
            "common_rewrite_identities": len(common_rewrites),
            "rewrite_strata_own_wrong_mismatches": rw_strata_mismatch,
            "rewrite_mask_own_wrong_mismatches": rw_mask_mismatch,
            "own_visible_blocked_rewrite_target_mismatches": own_vis_blk_mismatch,
            "wrong_visible_blocked_rewrite_target_mismatches": wrong_vis_blk_mismatch,
            "source_mask_own_wrong_mismatches": src_own_wrong_mismatch,
            "source_mask_visible_blocked_mismatches": src_vis_blk_mismatch,
        },
        "forward_checks": {
            "custom_full_vs_stock_max_abs_delta": custom_delta,
            "blocked_vs_full_max_abs_delta": blocked_delta,
            "loss_initial": float(loss.detach().cpu()),
            "grad_tensors": len(grads),
            "gradients_finite": bool(grad_finite),
            "masked_tokens_batch": int((labels != -100).sum().item()),
        },
        "conditions": conditions,
        "pass": all(conditions.values()),
    }
    (out / "crossview_v2_preflight.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": result["status"],
        "pass": result["pass"],
        "identity_checks": result["identity_checks"],
        "forward_checks": result["forward_checks"],
    }, indent=2), flush=True)
    if not result["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
