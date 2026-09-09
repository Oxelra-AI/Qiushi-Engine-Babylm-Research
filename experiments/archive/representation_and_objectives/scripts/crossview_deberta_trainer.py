#!/usr/bin/env python3
"""research: Pair-aware DeBERTa MLM trainer for cross-view causal separation.

Trains DeBERTa-v2 8x480 with pairwise attention masking. For pair rows, source and
rewrite tokens occupy separate segments; a 3D mask controls cross-boundary reachability.
Filler rows get full bidirectional attention. Per-target-stratum loss tracking isolates
source-absent content learning signals.

Arms (via --partner + --visibility):
  own_visible:   source_i + rewrite_i, full cross-boundary attention
  own_blocked:   source_i + rewrite_i, blocked cross-boundary attention
  wrong_visible: source_i + rewrite_π(i), full cross-boundary attention
  wrong_blocked: source_i + rewrite_π(i), blocked cross-boundary attention
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import random
import sys
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

USER_ROOT = pathlib.Path(".").resolve()
COMPACT_EXPERIENCE_SCRIPTS = USER_ROOT / "experiments/archive" / 'compact_experience' / "scripts"
if str(COMPACT_EXPERIENCE_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(COMPACT_EXPERIENCE_SCRIPTS))

from masking_curriculum_trainer import (          # noqa: E402
    make_portable_tokenizer, build_model, save_hf_checkpoint,
    is_word_start, apply_masking_curriculum, MaskingCurriculumState,
)
from transformers import get_cosine_schedule_with_warmup  # noqa: E402


# ── Strata ────────────────────────────────────────────────────────────────────
S_PAD, S_FILLER, S_SOURCE = 0, 1, 2
S_RW_COPIED, S_RW_ABS_CONTENT, S_RW_ABS_OTHER = 3, 4, 5
STRATA = {S_PAD: "pad", S_FILLER: "filler", S_SOURCE: "source",
           S_RW_COPIED: "rw_copied", S_RW_ABS_CONTENT: "rw_abs_content",
           S_RW_ABS_OTHER: "rw_abs_other"}

def origin_to_stratum(origin: str) -> int:
    if origin.startswith("copied"):
        return S_RW_COPIED
    elif origin == "absent_content":
        return S_RW_ABS_CONTENT
    return S_RW_ABS_OTHER


# ── Custom forward ────────────────────────────────────────────────────────────
def custom_mlm_forward(model, input_ids, pad_mask_2d, pair_mask_3d, labels=None):
    """DeBERTa MLM with 2D embedding mask and 3D encoder mask.
    research smoke: exact equivalence to stock forward under full visibility."""
    emb = model.deberta.embeddings(
        input_ids=input_ids, token_type_ids=None,
        position_ids=None, mask=pad_mask_2d, inputs_embeds=None)
    enc = model.deberta.encoder(
        emb, pair_mask_3d,
        output_hidden_states=False, output_attentions=False, return_dict=True)
    seq = enc[0]
    if getattr(model, "legacy", False):
        logits = model.cls(seq)
    else:
        logits = model.lm_predictions(seq, model.deberta.embeddings.word_embeddings)
    loss = None
    if labels is not None:
        loss = F.cross_entropy(logits.view(-1, model.config.vocab_size),
                               labels.view(-1), ignore_index=-100)
    return logits, loss


# ── Dataset ───────────────────────────────────────────────────────────────────
class PairAwareDataset(Dataset):
    """Handles both pair rows (source+rewrite with 3D mask) and filler rows."""
    def __init__(self, examples: list[dict], tokenizer, seq_len: int,
                 partner: str, blocked: bool):
        self.examples = examples
        self.tok = tokenizer
        self.T = seq_len
        self.partner = partner
        self.blocked = blocked
        self.special_ids = set(tokenizer.all_special_ids)
        self._ws = {}

    def _is_ws(self, tid: int) -> bool:
        v = self._ws.get(tid)
        if v is None:
            s = self.tok.convert_ids_to_tokens(int(tid))
            v = bool(s and is_word_start(str(s)))
            self._ws[tid] = v
        return v

    def __len__(self):
        return len(self.examples)

    def _word_groups(self, ids, attn, start_gid=0):
        T = ids.shape[0]
        g = torch.full((T,), -1, dtype=torch.long)
        gid = start_gid - 1
        for i in range(T):
            if attn[i] == 0: continue
            t = int(ids[i])
            if t in self.special_ids: continue
            if gid < start_gid or self._is_ws(t) or i == 0:
                gid += 1
            g[i] = gid
        return g, gid + 1

    def __getitem__(self, idx):
        ex = self.examples[idx]
        T = self.T
        if ex["type"] == "filler":
            return self._filler(ex, T)
        return self._pair(ex, T)

    def _filler(self, ex, T):
        enc = self.tok(ex["text"], add_special_tokens=False, truncation=True,
                       max_length=T, padding="max_length", return_tensors="pt")
        ids = enc["input_ids"].squeeze(0)
        att = enc["attention_mask"].squeeze(0)
        pm = (att.unsqueeze(1) * att.unsqueeze(0)).long()
        wg, _ = self._word_groups(ids, att)
        st = torch.where(att.bool(), torch.tensor(S_FILLER), torch.tensor(S_PAD))
        return {"input_ids": ids, "attention_mask": att, "pair_mask": pm,
                "word_group": wg, "target_stratum": st, "words": ex["words"]}

    def _pair(self, ex, T):
        src = ex["source_text"]
        if self.partner == "own":
            rw, origins, tw = ex["own_rewrite"], ex["own_rewrite_origins"], ex["own_total_words"]
        else:
            rw, origins, tw = ex["wrong_rewrite"], ex["wrong_rewrite_origins"], ex["wrong_total_words"]

        half = T // 2
        se = self.tok(src, add_special_tokens=False, truncation=True, max_length=half)
        re = self.tok(rw, add_special_tokens=False, truncation=True,
                      max_length=T - len(se["input_ids"]))
        si, ri = se["input_ids"], re["input_ids"]
        bnd = len(si)
        tl = min(bnd + len(ri), T)
        ri = ri[:tl - bnd]

        ids = torch.zeros(T, dtype=torch.long)
        att = torch.zeros(T, dtype=torch.long)
        ids[:bnd] = torch.tensor(si, dtype=torch.long)
        ids[bnd:tl] = torch.tensor(ri, dtype=torch.long)
        att[:tl] = 1

        # 3D pair mask
        pm = torch.zeros(T, T, dtype=torch.long)
        pm[:bnd, :bnd] = 1
        pm[bnd:tl, bnd:tl] = 1
        if not self.blocked:
            pm[:bnd, bnd:tl] = 1
            pm[bnd:tl, :bnd] = 1
        pm *= att.unsqueeze(1) * att.unsqueeze(0)

        # Word groups
        wg = torch.full((T,), -1, dtype=torch.long)
        sg, nx = self._word_groups(ids[:bnd], att[:bnd], 0)
        wg[:bnd] = sg
        rg, _ = self._word_groups(ids[bnd:tl], att[bnd:tl], nx)
        wg[bnd:tl] = rg

        # Strata
        st = torch.full((T,), S_PAD, dtype=torch.long)
        st[:bnd] = S_SOURCE
        wids = re.word_ids(batch_index=0) if hasattr(re, "word_ids") else None
        if wids is not None:
            for i, wid in enumerate(wids):
                if bnd + i >= tl: break
                if wid is not None and wid < len(origins):
                    st[bnd + i] = origin_to_stratum(origins[wid])
                else:
                    st[bnd + i] = S_RW_COPIED
        else:
            st[bnd:tl] = S_RW_COPIED  # fallback if word_ids unavailable

        return {"input_ids": ids, "attention_mask": att, "pair_mask": pm,
                "word_group": wg, "target_stratum": st, "words": tw}


def collate(batch):
    return {k: torch.stack([x[k] for x in batch]) if k != "words"
            else torch.tensor([x[k] for x in batch], dtype=torch.long)
            for k in batch[0]}


# ── Args ──────────────────────────────────────────────────────────────────────
def build_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--partner", required=True, choices=["own", "wrong"])
    p.add_argument("--visibility", required=True, choices=["visible", "blocked"])
    p.add_argument("--tokenizer_path", default="")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--max_word_exposure", type=int, default=20_000_000)
    p.add_argument("--checkpoint_words", type=int, default=1_000_000)
    p.add_argument("--max_seq_length", type=int, default=256)
    p.add_argument("--max_position_embeddings", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--hidden_size", type=int, default=480)
    p.add_argument("--n_layer", type=int, default=8)
    p.add_argument("--n_head", type=int, default=8)
    p.add_argument("--ffn_mult", type=int, default=4)
    p.add_argument("--position_buckets", type=int, default=256)
    p.add_argument("--max_relative_positions", type=int, default=256)
    p.add_argument("--deberta_pos_att_type", default="p2c,c2p")
    p.add_argument("--learning_rate", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_fraction", type=float, default=0.05)
    p.add_argument("--lr_total_steps", type=int, default=0)
    p.add_argument("--mask_prob", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=43)
    p.add_argument("--extra_init_seed", type=int, default=43022)
    p.add_argument("--train_rng_seed", type=int, default=43023)
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--smoke", action="store_true")
    return p.parse_args()


# ── Smoke ─────────────────────────────────────────────────────────────────────
def run_smoke(model, tokenizer, dataset, device, out, arm, args):
    print("\n=== SMOKE TEST ===", flush=True)
    T = args.max_seq_length
    res = {}

    # shapes
    item = dataset[0]
    assert item["input_ids"].shape == (T,)
    assert item["pair_mask"].shape == (T, T)
    res["shapes"] = True

    # stratum distribution on first 10 rows
    total_dist = {v: 0 for v in STRATA.values()}
    for i in range(min(10, len(dataset))):
        it = dataset[i]
        for sid, sn in STRATA.items():
            total_dist[sn] += int((it["target_stratum"] == sid).sum().item())
    res["stratum_first10"] = total_dist
    print(f"  Strata (first 10): {total_dist}", flush=True)

    # full-visible equivalence
    bs = min(4, len(dataset))
    batch = collate([dataset[i] for i in range(bs)])
    ids = batch["input_ids"].to(device)
    att = batch["attention_mask"].to(device)
    pm = batch["pair_mask"].to(device)

    model.eval()
    with torch.no_grad():
        stock = model(input_ids=ids, attention_mask=att)
        full = (att.unsqueeze(2) * att.unsqueeze(1)).long()
        cl, _ = custom_mlm_forward(model, ids, att, full)
        delta = (stock.logits - cl).abs().max().item()
    res["full_vis_delta"] = delta
    res["full_vis_pass"] = delta < 1e-4
    print(f"  Full-vis delta: {delta:.2e} (pass: {res['full_vis_pass']})", flush=True)

    # pair mask changes logits
    with torch.no_grad():
        pl, _ = custom_mlm_forward(model, ids, att, pm)
        pd = (cl - pl).abs().max().item()
    res["pair_mask_delta"] = pd
    has_pair = any(dataset.examples[i]["type"] == "pair" for i in range(bs))
    if has_pair and args.visibility == "blocked":
        res["pair_mask_changes"] = pd > 1e-7
    else:
        res["pair_mask_changes"] = True  # visible arm: pair_mask == full, delta ≈ 0
    print(f"  Pair mask delta: {pd:.2e} (has_pair: {has_pair})", flush=True)

    # gradient
    model.train()
    labs = ids.clone(); labs[:] = -100; labs[0, 3] = ids[0, 3]
    lo, ls = custom_mlm_forward(model, ids, att, pm, labs)
    ls.backward()
    ng = sum(1 for p in model.parameters() if p.grad is not None)
    af = all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    res["grad_n"] = ng; res["grad_finite"] = af
    print(f"  Grads: {ng} tensors, finite: {af}", flush=True)

    # WWM smoke
    gen = torch.Generator(device=device); gen.manual_seed(43023)
    cs = MaskingCurriculumState("wwm_fixed", 0.15, 0.15)
    cs.initialize(len(tokenizer), 100)
    mi, la = apply_masking_curriculum(ids, att, batch["word_group"].to(device),
                                      tokenizer, cs, gen)
    nm = int((la != -100).sum().item())
    res["wwm_masked"] = nm
    res["wwm_ok"] = nm > 0
    print(f"  WWM: {nm} masked tokens", flush=True)

    ok = res["full_vis_pass"] and res["pair_mask_changes"] and res["grad_finite"] and res["wwm_ok"]
    res["pass"] = ok
    sp = out / f"smoke_{arm}.json"
    sp.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(f"\n  SMOKE {'PASS' if ok else 'FAIL'}: {sp}", flush=True)
    if not ok: raise SystemExit(1)
    return res


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    args = build_args()
    t0 = time.time()
    out = pathlib.Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    arm = f"{args.partner}_{args.visibility}"
    blocked = args.visibility == "blocked"

    tokenizer = make_portable_tokenizer(args.tokenizer_path)
    print(f"Tokenizer: vocab {len(tokenizer)}", flush=True)

    # load pool
    pool = []
    with open(args.data) as f:
        for line in f:
            if line.strip(): pool.append(json.loads(line))
    pw = sum(r["own_total_words"] if r["type"] == "pair" else r["words"] for r in pool)
    print(f"Pool: {len(pool)} rows, {pw} words/epoch", flush=True)

    # build epoch list
    examples, aw, ep = [], 0, 0
    while aw < args.max_word_exposure:
        ep_pool = list(pool)
        random.Random(args.seed + 1000003 * ep).shuffle(ep_pool)
        for r in ep_pool:
            if aw >= args.max_word_exposure: break
            w = (r["own_total_words"] if r["type"] == "pair" and args.partner == "own"
                 else r.get("wrong_total_words", r.get("words", 0))
                 if r["type"] == "pair" else r["words"])
            if aw + w <= args.max_word_exposure:
                examples.append(r); aw += w
        ep += 1
    print(f"Stream: {len(examples)} examples, {aw} words, {ep} epochs", flush=True)

    dataset = PairAwareDataset(examples, tokenizer, args.max_seq_length,
                                args.partner, blocked)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        collate_fn=collate, num_workers=0,
                        pin_memory=torch.cuda.is_available())
    total_steps = len(loader)

    # model
    reset = lambda s: (random.seed(s), torch.manual_seed(s),
                       torch.cuda.manual_seed_all(s) if torch.cuda.is_available() else None)
    reset(args.seed)
    if args.extra_init_seed >= 0: reset(args.extra_init_seed)
    model = build_model(args, tokenizer)
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    model.config.use_cache = False
    if args.train_rng_seed >= 0: reset(args.train_rng_seed)

    device = torch.device(args.device)
    model.to(device)
    pc = sum(p.numel() for p in model.parameters())
    print(f"Model: {pc:,} params on {device}", flush=True)

    if args.smoke:
        run_smoke(model, tokenizer, dataset, device, out, arm, args)
        return

    # optimizer
    optim = torch.optim.AdamW(model.parameters(), lr=args.learning_rate,
                               weight_decay=args.weight_decay, betas=(0.9, 0.98))
    st_total = args.lr_total_steps if args.lr_total_steps > 0 else total_steps
    warmup = max(1, int(st_total * args.warmup_fraction))
    sched = get_cosine_schedule_with_warmup(optim, warmup, st_total)

    cs = MaskingCurriculumState("wwm_fixed", args.mask_prob, args.mask_prob)
    cs.initialize(len(tokenizer), total_steps)
    gen = torch.Generator(device=device)
    gen.manual_seed(args.train_rng_seed if args.train_rng_seed >= 0 else args.seed)

    # training
    log_path = out / "training_log.jsonl"
    cum, loss_first, loss_last = 0, None, None
    next_ckpt = args.checkpoint_words
    saved, stratum_acc = [], {v: [0.0, 0] for v in STRATA.values() if v != "pad"}

    model.train()
    with log_path.open("w", encoding="utf-8") as logf:
        for step, batch in enumerate(loader, 1):
            words = int(batch.pop("words").sum().item())
            ids = batch["input_ids"].to(device, non_blocking=True)
            att = batch["attention_mask"].to(device, non_blocking=True)
            pm = batch["pair_mask"].to(device, non_blocking=True)
            wg = batch["word_group"].to(device, non_blocking=True)
            ts = batch["target_stratum"].to(device, non_blocking=True)

            cs.current_step = step - 1
            mi, la = apply_masking_curriculum(ids, att, wg, tokenizer, cs, gen)

            optim.zero_grad(set_to_none=True)
            lo, ls = custom_mlm_forward(model, mi, att, pm, la)
            ls.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            sched.step()

            cum += words
            lf = float(ls.detach().cpu())
            if loss_first is None: loss_first = lf
            loss_last = lf

            # per-stratum losses
            srec = {}
            with torch.no_grad():
                mask = la != -100
                if mask.sum() > 0:
                    ptl = F.cross_entropy(lo.view(-1, model.config.vocab_size),
                                          la.view(-1), reduction="none",
                                          ignore_index=-100).view(la.shape)
                    for sid, sn in STRATA.items():
                        if sid == S_PAD: continue
                        sm = mask & (ts == sid)
                        cnt = int(sm.sum().item())
                        if cnt > 0:
                            sl = float(ptl[sm].mean().item())
                            srec[sn] = round(sl, 5)
                            srec[f"{sn}_n"] = cnt
                            stratum_acc[sn][0] += sl * cnt
                            stratum_acc[sn][1] += cnt

            np_ = int(mask.sum().item())
            emr = np_ / max(1, int(att.sum().item()))
            rec = {"step": step, "loss": round(lf, 5),
                   "lr": float(sched.get_last_lr()[0]),
                   "words": words, "cum": cum, "masked": np_,
                   "emr": round(emr, 4), **srec}
            logf.write(json.dumps(rec) + "\n")
            if step == 1 or step % args.log_every == 0 or step == total_steps:
                print(json.dumps({"e": "t", "s": step, "l": round(lf, 4),
                                  "c": cum, "t": round(time.time() - t0, 0),
                                  **{k: v for k, v in srec.items() if not k.endswith("_n")}}),
                      flush=True)

            while next_ckpt and cum >= next_ckpt and next_ckpt <= args.max_word_exposure:
                nm = f"chck_{next_ckpt // 1_000_000}M" if next_ckpt >= 1_000_000 else "chck_1M"
                cp = out / "hf_model" / nm
                save_hf_checkpoint(model, tokenizer, cp)
                saved.append({"name": nm, "cum": cum, "path": str(cp)})
                print(json.dumps({"e": "ckpt", "n": nm, "c": cum}), flush=True)
                next_ckpt += args.checkpoint_words

    save_hf_checkpoint(model, tokenizer, out / "hf_model")

    sm = {sn: round(a[0] / a[1], 5) if a[1] > 0 else None for sn, a in stratum_acc.items()}
    sn_ = {f"{sn}_n": a[1] for sn, a in stratum_acc.items()}
    summary = {
        "status": f"STEP220_{arm.upper()}_DONE", "arm": arm,
        "partner": args.partner, "visibility": args.visibility,
        "word_exposure": cum, "steps": total_steps, "epochs": ep,
        "loss_first": round(loss_first, 5) if loss_first else None,
        "loss_last": round(loss_last, 5) if loss_last else None,
        "params": pc, "checkpoints": saved,
        "stratum_mean_loss": sm, "stratum_counts": sn_,
        "elapsed_sec": round(time.time() - t0, 1),
    }
    (out / "scientific_metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"e": "done", "arm": arm, "cum": cum, "loss_last": loss_last,
                       "stratum": sm, "t": round(time.time() - t0, 0)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
