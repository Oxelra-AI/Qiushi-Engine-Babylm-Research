#!/usr/bin/env python3
"""research: Raw-token entity-event memory interface (no template metadata).

Interface limitation: the earlier learned arm still received entity
slot positions, initial-state positions, event verb/actor positions, and query
coordinates from template regex. Natural EWoK supplies none of these. This trainer
removes ALL template metadata from the memory module and lets it discover soft
entity / event / state assignment from raw token hidden states, using only the
legitimately-known mask position (where the answer is predicted).

Interface the module is allowed to see:
  - input_ids, attention_mask  (raw tokens only)
  - mask_pos                    (position of <mask>; legitimate at inference)

Everything else (which tokens are entities, which are events, which is the query
entity, the initial state) is DISCOVERED by learned attention. This makes the
module applicable to any text, including natural EWoK conditional-reversal items.

Arms:
  vanilla        : no memory (baseline transformer)
  rawmem         : raw-token entity-event memory (learned discovery)
  rawmem_noevent : ablation - memory slots but no event write (tests whether the
                   event-conditioned write is what matters, vs pure slot readout)

Eval: same selective-updating / quartet / multi-event metrics as research, plus
paraphrase-held robustness when a paraphrase eval file is present.
"""
from __future__ import annotations
import argparse, hashlib, json, os, random, time
from collections import defaultdict
from pathlib import Path
import numpy as np, torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer


def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""): h.update(c)
    return h.hexdigest()

def model_sha(m):
    h = hashlib.sha256()
    for n, p in sorted(m.state_dict().items()):
        h.update(n.encode()); h.update(p.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()[:16]

def read_jsonl(p):
    return [json.loads(x) for x in Path(p).read_text("utf-8").splitlines() if x.strip()]


def encode_row(row, tok, max_len):
    """Encode using ONLY raw text + mask position. No entity/event/state/query metadata."""
    text = row["text"]
    enc = tok(text, padding="max_length", truncation=True, max_length=max_len)
    mask_id = tok.mask_token_id
    try:
        mpos = enc["input_ids"].index(mask_id)
    except ValueError:
        raise ValueError(f"mask absent: {text}")
    aid = tok.encode(" " + row["answer"], add_special_tokens=False)
    fid = tok.encode(" " + row["foil"], add_special_tokens=False)
    if len(aid) < 2 or len(fid) < 2:
        raise ValueError(f"target too short: {row['answer']}={aid} {row['foil']}={fid}")
    return {**enc, "mask_pos": mpos, "answer_id": aid[1], "foil_id": fid[1]}


class Rows(Dataset):
    def __init__(self, rows, tok, max_len):
        self.items = []
        for r in rows:
            x = encode_row(r, tok, max_len)
            x["meta"] = r
            self.items.append(x)
    def __len__(self): return len(self.items)
    def __getitem__(self, i): return self.items[i]


def collate(xs):
    out = {}
    for k in ["input_ids", "attention_mask", "mask_pos", "answer_id", "foil_id"]:
        out[k] = torch.tensor([x[k] for x in xs], dtype=torch.long)
    out["meta"] = [x["meta"] for x in xs]
    return out


class RawTokenEntityMemory(nn.Module):
    """Discovers entities/events/state from raw hidden states.

    Slots: n_slots learnable slot query vectors. Each slot softly attends over all
    non-pad token positions to gather its entity representation and initial value.
    Events: a learned event detector scores each position; event value is a
    context-pooled vector; the write actor-attention routes each event to a slot by
    matching an event-derived query against slot keys. Read: the mask position's
    own representation forms a read query against slot keys; the read value is added
    back to the mask position. Nothing uses template positions.
    """
    def __init__(self, d, n_slots=2, n_event_slots=3, no_event=False):
        super().__init__()
        self.d = d; self.n_slots = n_slots; self.n_event_slots = n_event_slots
        self.no_event = no_event
        self.scale = d ** -0.5
        # learnable slot queries that attend over the sequence to localize entities
        self.slot_query = nn.Parameter(torch.randn(n_slots, d) * 0.02)
        self.key_proj = nn.Linear(d, d, bias=False)     # keys for slot localization
        self.val_proj = nn.Linear(d, d)                 # value gathered into slot init
        self.slot_key = nn.Linear(d, d, bias=False)     # slot identity key (write/read match)
        # event detection: score each position as an event trigger
        self.event_score = nn.Linear(d, 1)
        self.event_query = nn.Linear(d, d, bias=False)  # event->slot routing query
        self.event_value = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, d))
        self.gate = nn.Linear(2 * d, 1)
        self.read_query = nn.Linear(d, d, bias=False)
        self.out = nn.Linear(d, d)
        self.norm = nn.LayerNorm(d)

    def forward(self, h, mask_pos, pad, permute_write=False, return_aux=False):
        B, L, D = h.shape
        dev = h.device
        ar = torch.arange(B, device=dev)
        neg = torch.finfo(h.dtype).min
        padf = pad  # (B,L) True where pad

        K = self.key_proj(h)                             # (B,L,D)
        V = self.val_proj(h)                             # (B,L,D)
        # slot localization: each slot query attends over positions
        # (B, n_slots, L)
        loc = torch.einsum("sd,bld->bsl", self.slot_query, K) * self.scale
        loc = loc.masked_fill(padf[:, None, :], neg)
        loc_a = torch.softmax(loc, dim=-1)               # (B,n_slots,L)
        slot_ctx = torch.einsum("bsl,bld->bsd", loc_a, h)   # (B,n_slots,D) entity repr
        slots = torch.einsum("bsl,bld->bsd", loc_a, V)      # (B,n_slots,D) initial value
        keys = self.slot_key(slot_ctx)                   # (B,n_slots,D)

        write_hist = []
        if not self.no_event:
            # event detection over positions
            es = self.event_score(h).squeeze(-1)         # (B,L)
            es = es.masked_fill(padf, neg)
            # top n_event_slots soft events via repeated softmax with masking
            remaining = es.clone()
            for t in range(self.n_event_slots):
                ea = torch.softmax(remaining, dim=-1)    # (B,L) soft event position
                verb = torch.einsum("bl,bld->bd", ea, h) # (B,D) event trigger repr
                # route event to slot
                q = self.event_query(verb)               # (B,D)
                wa = torch.softmax((q[:, None, :] * keys).sum(-1) * self.scale, dim=-1)  # (B,n_slots)
                if permute_write:
                    wa = wa.flip(1)
                old = (wa[:, :, None] * slots).sum(1)     # (B,D)
                value = self.event_value(torch.cat([verb, old], -1))
                g = torch.sigmoid(self.gate(torch.cat([verb, old], -1)))
                updated = old + g * (value - old)
                slots = slots + wa[:, :, None] * (updated[:, None, :] - slots)
                write_hist.append(wa)
                # suppress the chosen event region for the next event slot
                remaining = remaining + (ea.detach().clamp_min(1e-9).log()) * 1.0

        # read at mask position
        query = h[ar, mask_pos]                          # (B,D)
        rq = self.read_query(query)
        ra = torch.softmax((rq[:, None, :] * keys).sum(-1) * self.scale, dim=-1)  # (B,n_slots)
        read = (ra[:, :, None] * slots).sum(1)           # (B,D)
        h = h.clone()
        h[ar, mask_pos] = self.norm(h[ar, mask_pos] + self.out(read))
        aux = {"write_attention": torch.stack(write_hist, 1) if write_hist else torch.zeros(B, 0, self.n_slots, device=dev),
               "read_attention": ra, "slot_localization": loc_a, "slots": slots}
        return (h, aux) if return_aux else (h, None)


class TinyMLM(nn.Module):
    def __init__(self, vocab, arm, d=192, layers=4, heads=6, ff=768, max_len=64,
                 n_slots=2, n_event_slots=3):
        super().__init__()
        self.arm = arm
        self.word = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(max_len, d)
        self.embnorm = nn.LayerNorm(d)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d, heads, ff, 0.1, batch_first=True, norm_first=True, activation="gelu")
            for _ in range(layers)])
        if arm == "vanilla":
            self.memory = None
        else:
            self.memory = RawTokenEntityMemory(d, n_slots, n_event_slots,
                                               no_event=(arm == "rawmem_noevent"))
        self.lm = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.LayerNorm(d), nn.Linear(d, vocab, bias=False))
        self.lm[-1].weight = self.word.weight

    def forward(self, b, permute_write=False, return_aux=False):
        ids = b["input_ids"]; B, L = ids.shape
        pos = torch.arange(L, device=ids.device)[None]
        h = self.embnorm(self.word(ids) + self.pos(pos))
        pad = b["attention_mask"].eq(0); aux = {}
        for i, layer in enumerate(self.layers):
            h = layer(h, src_key_padding_mask=pad)
            if i == 1 and self.memory is not None:
                h, aux = self.memory(h, b["mask_pos"], pad, permute_write, return_aux)
        return self.lm(h), aux


def todev(b, dev):
    return {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in b.items()}


@torch.no_grad()
def evaluate(model, loader, dev, permute=False):
    model.eval()
    rows = []; sums = defaultdict(lambda: [0, 0, 0.0])
    for b in loader:
        meta = b["meta"]; b = todev(b, dev)
        logits, aux = model(b, permute_write=permute, return_aux=True)
        ar = torch.arange(logits.shape[0], device=dev)
        z = logits[ar, b["mask_pos"]]
        margin = z[ar, b["answer_id"]] - z[ar, b["foil_id"]]
        ok = margin.gt(0)
        for i, m in enumerate(meta):
            group = m["split"]
            sums[group][0] += int(ok[i]); sums[group][1] += 1; sums[group][2] += float(margin[i])
            rows.append({"id": m["id"], "split": group, "kind": m["kind"],
                         "family": m["family"], "is_affected": m.get("is_affected_query"),
                         "quartet_id": m.get("quartet_id", ""),
                         "correct": bool(ok[i]), "margin": float(margin[i]), "permuted": permute})
    summary = {k: {"correct": v[0], "n": v[1], "accuracy": v[0] / v[1], "margin_mean": v[2] / v[1]}
               for k, v in sums.items()}

    def selective(splitname):
        hr = [r for r in rows if r["split"] == splitname and r["kind"] == "binding"]
        if not hr:
            return None
        affected = {r["quartet_id"]: r["correct"] for r in hr if r.get("is_affected") == True}
        unaffected = {r["quartet_id"]: r["correct"] for r in hr if r.get("is_affected") == False}
        aff_acc = sum(affected.values()) / max(len(affected), 1)
        unaff_acc = sum(unaffected.values()) / max(len(unaffected), 1)
        qgroups = defaultdict(list)
        for r in hr:
            base = "_".join(r["quartet_id"].rsplit("_", 2)[:1])
            qgroups[base].append(r)
        pair_both = sum(1 for g in qgroups.values()
                        if any(r["correct"] for r in g if r.get("is_affected") == True)
                        and any(r["correct"] for r in g if r.get("is_affected") == False))
        quartet_all = sum(1 for g in qgroups.values() if all(r["correct"] for r in g))
        return {"affected_accuracy": aff_acc, "unaffected_accuracy": unaff_acc,
                "composite": 0.5 * (aff_acc + unaff_acc),
                "pair_both_correct": pair_both, "pair_total": len(qgroups),
                "pair_both_rate": pair_both / max(len(qgroups), 1),
                "quartet_all_correct": quartet_all, "quartet_all_rate": quartet_all / max(len(qgroups), 1),
                "n_affected": len(affected), "n_unaffected": len(unaffected)}

    for sp in ["eval_held_recomb", "eval_held_paraphrase"]:
        s = selective(sp)
        if s is not None:
            summary["selective_" + sp] = s

    multi_rows = [r for r in rows if r["kind"] == "multi_event"]
    if multi_rows:
        mc = sum(r["correct"] for r in multi_rows)
        summary["multi_event"] = {"correct": mc, "n": len(multi_rows), "accuracy": mc / len(multi_rows)}
    return summary, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["vanilla", "rawmem", "rawmem_noevent"], required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--out-dir", default="")
    ap.add_argument("--seed", type=int, default=43022)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--max-len", type=int, default=64)
    ap.add_argument("--d-model", type=int, default=192)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--train-file", default="train.jsonl")
    ap.add_argument("--eval-file", default="eval.jsonl")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    seed_all(args.seed)
    out = Path(args.out_dir or os.environ.get("QIUSHI_AI_LAB_RUN_DIR", "."))
    out.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True, local_files_only=True)
    train_rows = read_jsonl(Path(args.data_dir) / args.train_file)
    eval_rows = read_jsonl(Path(args.data_dir) / args.eval_file)
    if args.dry_run:
        train_rows = train_rows[:64]; eval_rows = eval_rows[:64]; args.epochs = 1

    train = Rows(train_rows, tok, args.max_len)
    ev = Rows(eval_rows, tok, args.max_len)
    g = torch.Generator().manual_seed(args.seed)
    tl = DataLoader(train, batch_size=args.batch_size, shuffle=True, generator=g, collate_fn=collate, num_workers=0)
    el = DataLoader(ev, batch_size=args.batch_size, shuffle=False, collate_fn=collate, num_workers=0)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TinyMLM(len(tok), args.arm, args.d_model, args.layers, max_len=args.max_len).to(dev)
    nparams = sum(p.numel() for p in model.parameters())
    print(json.dumps({"arm": args.arm, "parameters": nparams, "device": str(dev),
                      "train_records": len(train), "eval_records": len(ev)}), flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    losses = []; t0 = time.time()
    for ep in range(args.epochs):
        model.train(); total = 0.0; n = 0
        for b in tl:
            b = todev(b, dev); opt.zero_grad(set_to_none=True)
            logits, _ = model(b)
            ar = torch.arange(logits.shape[0], device=dev)
            loss = F.cross_entropy(logits[ar, b["mask_pos"]], b["answer_id"])
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            total += float(loss) * len(ar); n += len(ar)
        losses.append(total / n)
        print(json.dumps({"epoch": ep + 1, "loss": losses[-1]}), flush=True)

    base, rows = evaluate(model, el, dev, False)
    if args.arm == "vanilla":
        perm = {}
    else:
        perm, _ = evaluate(model, el, dev, True)

    torch.save({"model": model.state_dict(), "args": vars(args)}, out / "model.pt")
    with (out / "eval_rows.jsonl").open("w") as f:
        for r in rows: f.write(json.dumps(r) + "\n")

    result = {
        "status": "RAWTOKEN_ENTITY_MEMORY",
        "arm": args.arm, "seed": args.seed, "device": str(dev), "dry_run": args.dry_run,
        "train_records": len(train), "eval_records": len(ev),
        "epochs": args.epochs, "losses": losses, "parameters": nparams,
        "model_sha256": model_sha(model),
        "data_sha256": {"train": sha(Path(args.data_dir) / args.train_file),
                        "eval": sha(Path(args.data_dir) / args.eval_file)},
        "baseline": base, "write_permutation": perm,
        "runtime_sec": time.time() - t0,
    }
    (out / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
