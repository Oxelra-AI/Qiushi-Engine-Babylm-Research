#!/usr/bin/env python3
"""research corrected pair-aware DeBERTa MLM trainer.

Repairs two confounds in the research trainer before any four-arm H100 study:
  1. rewrite target strata are intrinsic to the rewrite identity (computed against
     the rewrite's own true source), not relative to the receiving source in the
     wrong-partner arm;
  2. masking is deterministic and identity-attached.  Source masks are keyed by
     source identity; rewrite masks are keyed by rewrite identity; filler masks
     are keyed by filler identity.  Thus the same target words are trained across
     own/wrong pairings and visible/blocked arms, independent of row order.

Arms:
  own_visible / own_blocked: source_i + rewrite_i
  wrong_visible / wrong_blocked: source_i + rewrite_pi(i)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import pathlib
import random
import sys
import time
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import DebertaV2ForMaskedLM, get_cosine_schedule_with_warmup  # noqa: E402

USER_ROOT = pathlib.Path(".").resolve()
COMPACT_EXPERIENCE_SCRIPTS = USER_ROOT / "experiments/archive" / 'compact_experience' / "scripts"
if str(COMPACT_EXPERIENCE_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(COMPACT_EXPERIENCE_SCRIPTS))

from masking_curriculum_trainer import (  # noqa: E402
    make_portable_tokenizer,
    build_model,
    save_hf_checkpoint,
)

# ── Strata ────────────────────────────────────────────────────────────────────
S_PAD, S_FILLER, S_SOURCE = 0, 1, 2
S_RW_COPIED, S_RW_ABS_CONTENT, S_RW_ABS_OTHER = 3, 4, 5
STRATA = {
    S_PAD: "pad",
    S_FILLER: "filler",
    S_SOURCE: "source",
    S_RW_COPIED: "rw_copied",
    S_RW_ABS_CONTENT: "rw_abs_content",
    S_RW_ABS_OTHER: "rw_abs_other",
}


def origin_to_stratum(origin: str) -> int:
    if origin.startswith("copied"):
        return S_RW_COPIED
    if origin == "absent_content":
        return S_RW_ABS_CONTENT
    return S_RW_ABS_OTHER


# ── Stable deterministic masking ──────────────────────────────────────────────
def stable_u64(s: str) -> int:
    return int.from_bytes(hashlib.blake2b(s.encode("utf-8"), digest_size=8).digest(), "little")


def stable_uniform01(s: str) -> float:
    # Use top 53 bits for exactly representable double in [0,1).
    return ((stable_u64(s) >> 11) & ((1 << 53) - 1)) / float(1 << 53)


def stable_randint(s: str, n: int) -> int:
    return stable_u64(s) % n


def sha256_file(path: pathlib.Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ── Custom forward ────────────────────────────────────────────────────────────
def custom_mlm_forward(model, input_ids, pad_mask_2d, pair_mask_3d, labels=None):
    """DeBERTa MLM with 2D embedding mask and 3D encoder mask.

    research smoke established exact equivalence to stock forward when pair_mask_3d
    is full bidirectional padding reachability, and real logit change under a
    source↔rewrite block.
    """
    emb = model.deberta.embeddings(
        input_ids=input_ids,
        token_type_ids=None,
        position_ids=None,
        mask=pad_mask_2d,
        inputs_embeds=None,
    )
    enc = model.deberta.encoder(
        emb,
        pair_mask_3d,
        output_hidden_states=False,
        output_attentions=False,
        return_dict=True,
    )
    seq = enc[0]
    if getattr(model, "legacy", False):
        logits = model.cls(seq)
    else:
        logits = model.lm_predictions(seq, model.deberta.embeddings.word_embeddings)
    loss = None
    if labels is not None:
        loss = F.cross_entropy(logits.view(-1, model.config.vocab_size), labels.view(-1), ignore_index=-100)
    return logits, loss


# ── Dataset ───────────────────────────────────────────────────────────────────
class PairAwareDataset(Dataset):
    """Pair/filler dataset with identity-attached deterministic WWM masks."""

    def __init__(
        self,
        examples: list[dict[str, Any]],
        tokenizer,
        seq_len: int,
        partner: str,
        blocked: bool,
        mask_prob: float,
        mask_seed: int,
    ):
        self.examples = examples
        self.tok = tokenizer
        self.T = seq_len
        self.partner = partner
        self.blocked = blocked
        self.mask_prob = mask_prob
        self.mask_seed = mask_seed
        self.special_ids = set(int(x) for x in tokenizer.all_special_ids)
        self.mask_token_id = int(tokenizer.mask_token_id)
        self.vocab_size = len(tokenizer)

    def __len__(self):
        return len(self.examples)

    def _mask_tokens_by_identity(
        self,
        ids: torch.Tensor,
        word_ids: list[int | None],
        start: int,
        length: int,
        epoch: int,
        seg: str,
        identity: str,
        labels: torch.Tensor,
        masked: torch.Tensor,
    ) -> int:
        """Apply deterministic WWM to one segment; return masked-token count."""
        selected_cache: dict[int, bool] = {}
        tok_in_word: dict[int, int] = {}
        n_masked = 0
        for rel in range(length):
            wid = word_ids[rel] if rel < len(word_ids) else None
            pos = start + rel
            if wid is None:
                continue
            tid = int(ids[pos])
            if tid in self.special_ids:
                continue
            if wid not in selected_cache:
                key = f"s={self.mask_seed}|ep={epoch}|seg={seg}|id={identity}|word={wid}|select"
                selected_cache[wid] = stable_uniform01(key) < self.mask_prob
                tok_in_word[wid] = 0
            if not selected_cache[wid]:
                continue
            token_ord = tok_in_word.get(wid, 0)
            tok_in_word[wid] = token_ord + 1
            labels[pos] = tid
            rkey = f"s={self.mask_seed}|ep={epoch}|seg={seg}|id={identity}|word={wid}|tok={token_ord}|replace"
            r = stable_uniform01(rkey)
            if r < 0.8:
                masked[pos] = self.mask_token_id
            elif r < 0.9:
                masked[pos] = stable_randint(rkey + "|randtok", self.vocab_size)
            # else leave original
            n_masked += 1
        return n_masked

    def __getitem__(self, idx: int):
        ex = self.examples[idx]
        if ex["type"] == "filler":
            return self._filler(ex, idx)
        return self._pair(ex, idx)

    def _filler(self, ex: dict[str, Any], idx: int):
        T = self.T
        enc = self.tok(
            ex["text"],
            add_special_tokens=False,
            truncation=True,
            max_length=T,
            padding="max_length",
            return_tensors="pt",
        )
        ids = enc["input_ids"].squeeze(0)
        att = enc["attention_mask"].squeeze(0)
        pm = (att.unsqueeze(1) * att.unsqueeze(0)).long()
        st = torch.where(att.bool(), torch.tensor(S_FILLER), torch.tensor(S_PAD))
        labels = torch.full((T,), -100, dtype=torch.long)
        masked = ids.clone()
        epoch = int(ex.get("_epoch", 0))
        fid = str(ex.get("filler_id", f"filler_idx:{idx}"))
        wids = enc.word_ids(batch_index=0) if hasattr(enc, "word_ids") else list(range(int(att.sum().item())))
        self._mask_tokens_by_identity(ids, wids, 0, int(att.sum().item()), epoch, "filler", fid, labels, masked)
        return {
            "input_ids": ids,
            "masked_input_ids": masked,
            "labels": labels,
            "attention_mask": att,
            "pair_mask": pm,
            "target_stratum": st,
            "words": ex["words"],
            "boundary": 0,
            "rewrite_len": 0,
        }

    def _pair(self, ex: dict[str, Any], idx: int):
        T = self.T
        src = ex["source_text"]
        source_id = str(ex.get("source_pair_id", ex.get("pair_id")))
        if self.partner == "own":
            rw = ex["own_rewrite"]
            origins = ex["own_rewrite_origins_true"]
            rewrite_id = str(ex.get("own_rewrite_pair_id", source_id))
            words = int(ex["own_total_words"])
        else:
            rw = ex["wrong_rewrite"]
            origins = ex["wrong_rewrite_origins_true"]
            rewrite_id = str(ex["wrong_rewrite_pair_id"])
            words = int(ex["wrong_total_words"])

        half = T // 2
        se = self.tok(src, add_special_tokens=False, truncation=True, max_length=half)
        si = list(se["input_ids"])
        re = self.tok(rw, add_special_tokens=False, truncation=True, max_length=T - len(si))
        ri = list(re["input_ids"])
        bnd = len(si)
        tl = min(bnd + len(ri), T)
        ri = ri[: tl - bnd]
        rw_len = len(ri)

        ids = torch.zeros(T, dtype=torch.long)
        att = torch.zeros(T, dtype=torch.long)
        if bnd:
            ids[:bnd] = torch.tensor(si, dtype=torch.long)
        if rw_len:
            ids[bnd:tl] = torch.tensor(ri, dtype=torch.long)
        att[:tl] = 1

        pm = torch.zeros(T, T, dtype=torch.long)
        pm[:bnd, :bnd] = 1
        pm[bnd:tl, bnd:tl] = 1
        if not self.blocked:
            pm[:bnd, bnd:tl] = 1
            pm[bnd:tl, :bnd] = 1
        pm *= att.unsqueeze(1) * att.unsqueeze(0)

        st = torch.full((T,), S_PAD, dtype=torch.long)
        st[:bnd] = S_SOURCE
        rw_word_ids = re.word_ids(batch_index=0) if hasattr(re, "word_ids") else None
        if rw_word_ids is not None:
            for i, wid in enumerate(rw_word_ids[:rw_len]):
                if wid is not None and wid < len(origins):
                    st[bnd + i] = origin_to_stratum(origins[wid])
                else:
                    st[bnd + i] = S_RW_COPIED
        else:
            st[bnd:tl] = S_RW_COPIED

        labels = torch.full((T,), -100, dtype=torch.long)
        masked = ids.clone()
        epoch = int(ex.get("_epoch", 0))
        src_wids = se.word_ids(batch_index=0) if hasattr(se, "word_ids") else list(range(bnd))
        rw_wids = rw_word_ids if rw_word_ids is not None else list(range(rw_len))
        self._mask_tokens_by_identity(ids, src_wids, 0, bnd, epoch, "source", source_id, labels, masked)
        self._mask_tokens_by_identity(ids, rw_wids, bnd, rw_len, epoch, "rewrite", rewrite_id, labels, masked)

        return {
            "input_ids": ids,
            "masked_input_ids": masked,
            "labels": labels,
            "attention_mask": att,
            "pair_mask": pm,
            "target_stratum": st,
            "words": words,
            "boundary": bnd,
            "rewrite_len": rw_len,
        }


def collate(batch):
    out = {}
    for k in batch[0]:
        vals = [x[k] for x in batch]
        if torch.is_tensor(vals[0]):
            out[k] = torch.stack(vals)
        else:
            out[k] = torch.tensor(vals, dtype=torch.long)
    return out


# ── Args ──────────────────────────────────────────────────────────────────────
def build_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--partner", required=True, choices=["own", "wrong"])
    p.add_argument("--visibility", required=True, choices=["visible", "blocked"])
    p.add_argument("--tokenizer_path", default="")
    p.add_argument("--init_model_path", default="", help="Identical untrained model checkpoint for every arm")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--max_word_exposure", type=int, default=20_000_000)
    p.add_argument("--checkpoint_words", type=int, default=10_000_000)
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
    p.add_argument("--lr_total_steps", type=int, default=2529)
    p.add_argument("--mask_prob", type=float, default=0.15)
    p.add_argument("--mask_seed", type=int, default=430230221)
    p.add_argument("--seed", type=int, default=43)
    p.add_argument("--extra_init_seed", type=int, default=43022)
    p.add_argument("--train_rng_seed", type=int, default=43023)
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--smoke", action="store_true")
    return p.parse_args()


def reset_seeds(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_or_build_model(args, tokenizer):
    if args.init_model_path:
        model = DebertaV2ForMaskedLM.from_pretrained(args.init_model_path)
        return model
    reset_seeds(args.seed)
    if args.extra_init_seed >= 0:
        reset_seeds(args.extra_init_seed)
    model = build_model(args, tokenizer)
    return model


# ── Stream construction ───────────────────────────────────────────────────────
def load_pool(path: str) -> list[dict[str, Any]]:
    pool = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                pool.append(json.loads(line))
    return pool


def row_words(r: dict[str, Any], partner: str) -> int:
    if r["type"] == "pair":
        return int(r["own_total_words"] if partner == "own" else r["wrong_total_words"])
    return int(r["words"])


def build_stream(pool: list[dict[str, Any]], max_words: int, partner: str, seed: int) -> tuple[list[dict[str, Any]], int, int]:
    examples: list[dict[str, Any]] = []
    aw = 0
    ep = 0
    min_row_words = min(row_words(r, partner) for r in pool)
    while aw < max_words:
        ep_pool = list(pool)
        random.Random(seed + 1000003 * ep).shuffle(ep_pool)
        added_this_epoch = 0
        for r in ep_pool:
            if aw >= max_words:
                break
            w = row_words(r, partner)
            if aw + w <= max_words:
                rr = dict(r)
                rr["_epoch"] = ep
                examples.append(rr)
                aw += w
                added_this_epoch += w
        ep += 1
        # Arbitrary pilot budgets need not be exactly fillable by whole rows.
        # The official/short screens use exact 10M-epoch multiples and still fill exactly;
        # for nonmultiples, stop once the remaining budget is smaller than any row or no
        # row could be added, instead of looping forever.
        if aw < max_words and (max_words - aw < min_row_words or added_this_epoch == 0):
            break
    return examples, aw, ep


# ── Smoke ─────────────────────────────────────────────────────────────────────
def run_smoke(model, dataset, device, out, arm, args):
    print("\n=== research V2 SMOKE TEST ===", flush=True)
    res: dict[str, Any] = {}
    T = args.max_seq_length
    item = dataset[0]
    assert item["input_ids"].shape == (T,)
    assert item["pair_mask"].shape == (T, T)
    assert item["masked_input_ids"].shape == (T,)
    assert item["labels"].shape == (T,)
    res["shapes"] = True

    total_dist = {v: 0 for v in STRATA.values()}
    mask_dist = {v: 0 for v in STRATA.values()}
    for i in range(min(32, len(dataset))):
        it = dataset[i]
        for sid, sn in STRATA.items():
            total_dist[sn] += int((it["target_stratum"] == sid).sum().item())
            mask_dist[sn] += int(((it["target_stratum"] == sid) & (it["labels"] != -100)).sum().item())
    res["stratum_first32_tokens"] = total_dist
    res["stratum_first32_masked"] = mask_dist
    print(f"  Masked strata first32: {mask_dist}", flush=True)

    bs = min(4, len(dataset))
    batch = collate([dataset[i] for i in range(bs)])
    ids = batch["input_ids"].to(device)
    mids = batch["masked_input_ids"].to(device)
    labs = batch["labels"].to(device)
    att = batch["attention_mask"].to(device)
    pm = batch["pair_mask"].to(device)

    model.eval()
    with torch.no_grad():
        stock = model(input_ids=ids, attention_mask=att)
        full = (att.unsqueeze(2) * att.unsqueeze(1)).long()
        cl, _ = custom_mlm_forward(model, ids, att, full)
        delta = (stock.logits - cl).abs().max().item()
        pair_logits, _ = custom_mlm_forward(model, ids, att, pm)
        pdelta = (cl - pair_logits).abs().max().item()
    res["full_vis_delta"] = delta
    res["full_vis_pass"] = delta < 1e-4
    res["pair_mask_delta"] = pdelta
    has_pair = any(dataset.examples[i]["type"] == "pair" for i in range(bs))
    res["pair_mask_changes"] = bool(pdelta > 1e-7) if args.visibility == "blocked" and has_pair else True
    print(f"  Full-vis delta {delta:.2e}; pair-mask delta {pdelta:.2e}", flush=True)

    model.train()
    _, loss = custom_mlm_forward(model, mids, att, pm, labs)
    res["loss_initial"] = float(loss.detach().cpu())
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    res["grad_n"] = len(grads)
    res["grad_finite"] = all(torch.isfinite(g).all().item() for g in grads)
    res["masked_tokens_batch"] = int((labs != -100).sum().item())
    print(f"  Loss {res['loss_initial']:.4f}; grads {res['grad_n']} finite={res['grad_finite']}; masked={res['masked_tokens_batch']}", flush=True)

    ok = res["full_vis_pass"] and res["pair_mask_changes"] and res["grad_finite"] and res["masked_tokens_batch"] > 0
    res["pass"] = bool(ok)
    sp = out / f"smoke_{arm}.json"
    sp.write_text(json.dumps(res, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"  SMOKE {'PASS' if ok else 'FAIL'}: {sp}", flush=True)
    if not ok:
        raise SystemExit(1)
    return res


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    args = build_args()
    t0 = time.time()
    out = pathlib.Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    arm = f"{args.partner}_{args.visibility}"
    blocked = args.visibility == "blocked"

    tokenizer = make_portable_tokenizer(args.tokenizer_path)
    print(f"Tokenizer: vocab {len(tokenizer)}", flush=True)

    pool = load_pool(args.data)
    own_pool_words = sum(row_words(r, "own") for r in pool)
    wrong_pool_words = sum(row_words(r, "wrong") for r in pool)
    print(f"Pool: {len(pool)} rows; own_words={own_pool_words}; wrong_words={wrong_pool_words}", flush=True)
    if own_pool_words != wrong_pool_words:
        raise ValueError(f"Own/wrong epoch word mismatch: {own_pool_words} vs {wrong_pool_words}")

    examples, aw, ep = build_stream(pool, args.max_word_exposure, args.partner, args.seed)
    print(f"Stream: {len(examples)} examples, {aw} words, {ep} epochs", flush=True)

    dataset = PairAwareDataset(examples, tokenizer, args.max_seq_length, args.partner, blocked, args.mask_prob, args.mask_seed)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    total_steps = len(loader)

    if args.train_rng_seed >= 0:
        reset_seeds(args.train_rng_seed)
    model = load_or_build_model(args, tokenizer)
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    model.config.use_cache = False
    device = torch.device(args.device)
    model.to(device)
    pc = sum(p.numel() for p in model.parameters())
    init_path = pathlib.Path(args.init_model_path) if args.init_model_path else None
    init_sha = sha256_file(init_path / "model.safetensors") if init_path else None
    print(f"Model: {pc:,} params on {device}; init_sha={init_sha}", flush=True)

    if args.smoke:
        run_smoke(model, dataset, device, out, arm, args)
        return

    optim = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay, betas=(0.9, 0.98))
    st_total = args.lr_total_steps if args.lr_total_steps > 0 else total_steps
    warmup = max(1, int(st_total * args.warmup_fraction))
    sched = get_cosine_schedule_with_warmup(optim, warmup, st_total)

    log_path = out / "training_log.jsonl"
    cum = 0
    loss_first = None
    loss_last = None
    next_ckpt = args.checkpoint_words if args.checkpoint_words > 0 else None
    saved = []
    stratum_acc = {v: [0.0, 0] for v in STRATA.values() if v != "pad"}
    stratum_last = {v: None for v in STRATA.values() if v != "pad"}

    model.train()
    with log_path.open("w", encoding="utf-8") as logf:
        for step, batch in enumerate(loader, 1):
            words = int(batch.pop("words").sum().item())
            ids = batch["masked_input_ids"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            att = batch["attention_mask"].to(device, non_blocking=True)
            pm = batch["pair_mask"].to(device, non_blocking=True)
            ts = batch["target_stratum"].to(device, non_blocking=True)

            optim.zero_grad(set_to_none=True)
            logits, loss = custom_mlm_forward(model, ids, att, pm, labels)
            if not torch.isfinite(loss):
                raise RuntimeError(f"nonfinite loss at step {step}: {loss}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            sched.step()

            cum += words
            lf = float(loss.detach().cpu())
            if loss_first is None:
                loss_first = lf
            loss_last = lf

            srec = {}
            with torch.no_grad():
                mask = labels != -100
                ptl = F.cross_entropy(
                    logits.view(-1, model.config.vocab_size),
                    labels.view(-1),
                    reduction="none",
                    ignore_index=-100,
                ).view(labels.shape)
                for sid, sn in STRATA.items():
                    if sid == S_PAD:
                        continue
                    sm = mask & (ts == sid)
                    cnt = int(sm.sum().item())
                    if cnt > 0:
                        sl = float(ptl[sm].mean().item())
                        srec[sn] = round(sl, 5)
                        srec[f"{sn}_n"] = cnt
                        stratum_acc[sn][0] += sl * cnt
                        stratum_acc[sn][1] += cnt
                        stratum_last[sn] = sl
            nm = int((labels != -100).sum().item())
            emr = nm / max(1, int(att.sum().item()))
            rec = {
                "step": step,
                "loss": round(lf, 5),
                "lr": float(sched.get_last_lr()[0]),
                "words": words,
                "cum": cum,
                "masked": nm,
                "emr": round(emr, 4),
                **srec,
            }
            logf.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if step == 1 or step % args.log_every == 0 or step == total_steps:
                print(
                    json.dumps(
                        {
                            "e": "t",
                            "s": step,
                            "l": round(lf, 4),
                            "c": cum,
                            "lr": float(sched.get_last_lr()[0]),
                            "t": round(time.time() - t0, 0),
                            **{k: v for k, v in srec.items() if not k.endswith("_n")},
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
            while next_ckpt and cum >= next_ckpt and next_ckpt <= args.max_word_exposure:
                nm_ck = f"chck_{next_ckpt // 1_000_000}M" if next_ckpt >= 1_000_000 else "chck_1M"
                cp = out / "hf_model" / nm_ck
                save_hf_checkpoint(model, tokenizer, cp)
                saved.append({"name": nm_ck, "cum": cum, "path": str(cp)})
                print(json.dumps({"e": "ckpt", "n": nm_ck, "c": cum}), flush=True)
                next_ckpt += args.checkpoint_words

    save_hf_checkpoint(model, tokenizer, out / "hf_model")
    sm = {sn: round(a[0] / a[1], 5) if a[1] > 0 else None for sn, a in stratum_acc.items()}
    sn_ = {f"{sn}_n": a[1] for sn, a in stratum_acc.items()}
    slast = {f"{sn}_last": (round(v, 5) if v is not None else None) for sn, v in stratum_last.items()}
    summary = {
        "status": f"STEP221_{arm.upper()}_DONE",
        "arm": arm,
        "partner": args.partner,
        "visibility": args.visibility,
        "data": args.data,
        "init_model_path": args.init_model_path,
        "init_model_sha256": init_sha,
        "mask_schedule": "identity_attached_deterministic_wwm_v2",
        "mask_seed": args.mask_seed,
        "word_exposure": cum,
        "steps": total_steps,
        "epochs": ep,
        "lr_total_steps": st_total,
        "warmup_steps": warmup,
        "loss_first": round(loss_first, 5) if loss_first is not None else None,
        "loss_last": round(loss_last, 5) if loss_last is not None else None,
        "params": pc,
        "checkpoints": saved,
        "stratum_mean_loss": sm,
        "stratum_counts": sn_,
        "stratum_last_loss": slast,
        "elapsed_sec": round(time.time() - t0, 1),
    }
    (out / "scientific_metrics.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {"e": "done", "arm": arm, "cum": cum, "loss_last": loss_last, "stratum": sm, "t": round(time.time() - t0, 0)},
            indent=2,
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
