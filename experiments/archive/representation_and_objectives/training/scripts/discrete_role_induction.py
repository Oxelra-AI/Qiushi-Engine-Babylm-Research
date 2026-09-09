#!/usr/bin/env python3
"""research: Discrete Latent Role Induction — one bounded test.

Fourth and final raw-token attempt. Three prior attempts (rawmem, lexmem, lexrec)
all failed at chance-level held recombination with zero write sensitivity. This
attempt uses a genuinely different mechanism:

1. GUMBEL-SOFTMAX discrete position selection (prior used soft attention)
2. ITERATIVE slot competition (prior used single-pass)
3. SLOT-COMPETITION normalization (softmax over slots per position, forcing
   each position to be "owned" by at most one entity)

Predeclared pass/fail:
- held_recomb > 0.70 with BOTH affected > chance AND unaffected > chance
- held paraphrase > 0.55
- write-permutation delta on held < -0.15
- If ALL fail: close TinyMLM memory variant route

Architecture:
- Same TinyMLM encoder (4 layers, d=192, 6 heads, ff=768)
- DiscreteSlotMemory: iterative discrete entity assignment + event routing
- Same evaluation as research

Each sentence processed independently. No paired counterfactual access.
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


class DiscreteSlotMemory(nn.Module):
    """Iterative discrete entity/event role assignment from raw tokens.

    Key novelty vs rawmem/lexmem/lexrec:
    - Gumbel-Softmax forces discrete position selections
    - Iterative refinement allows slot representations to sharpen
    - Competition normalization (softmax over slots per position) forces
      each position to be owned by at most one entity
    - Event detection and entity-affectedness are separate discrete decisions
    """
    def __init__(self, d, n_slots=2, n_iters=3, no_event=False):
        super().__init__()
        self.d = d
        self.n_slots = n_slots
        self.n_iters = n_iters
        self.no_event = no_event
        self.scale = d ** -0.5

        # Learnable slot initializations
        self.slot_init = nn.Parameter(torch.randn(n_slots, d) * 0.02)

        # Per-iteration slot update
        self.slot_attn_q = nn.ModuleList([nn.Linear(d, d, bias=False) for _ in range(n_iters)])
        self.slot_attn_k = nn.Linear(d, d, bias=False)  # shared key projection
        self.slot_gru = nn.GRUCell(d, d)  # slot update after attention

        # Event detection
        self.event_q = nn.Parameter(torch.randn(1, d) * 0.02)
        # Affectedness: which entity is affected by the event
        self.affect_mlp = nn.Sequential(
            nn.Linear(d * 2, d), nn.GELU(), nn.Linear(d, 1))

        # State update network
        self.state_update = nn.Sequential(
            nn.Linear(d * 2, d), nn.GELU(), nn.Linear(d, d))
        self.update_gate = nn.Linear(d * 2, d)

        # Read query and output
        self.read_q = nn.Linear(d, d, bias=False)
        self.out_proj = nn.Linear(d, d)
        self.norm = nn.LayerNorm(d)

        # Temperature (set externally)
        self.tau = 1.0

    def forward(self, h, mask_pos, pad, permute_write=False, return_aux=False):
        """h: [B, L, D], mask_pos: [B], pad: [B, L] bool"""
        B, L, D = h.shape
        dev = h.device
        ar = torch.arange(B, device=dev)

        # Initialize slots from learnable parameters
        slots = self.slot_init.unsqueeze(0).expand(B, -1, -1).contiguous()  # [B, K, D]

        # Shared keys for all iterations
        keys = self.slot_attn_k(h)  # [B, L, D]

        aux_data = {"slot_assignments": [], "slot_entropies": []}

        # Iterative slot competition
        for it in range(self.n_iters):
            # Slot queries
            q = self.slot_attn_q[it](slots)  # [B, K, D]

            # Attention scores: [B, K, L]
            scores = torch.einsum('bkd,bld->bkl', q, keys) * self.scale
            scores = scores.masked_fill(pad.unsqueeze(1), -1e9)

            # COMPETITION: softmax over SLOTS for each position
            # This means each position is "claimed" by at most one slot
            # [B, K, L] -> softmax over dim=1 (slots)
            compete = F.softmax(scores, dim=1)  # [B, K, L]

            # SELECTION: Gumbel-Softmax over positions for each slot
            # This forces each slot to discretely select positions
            # [B, K, L] -> Gumbel over dim=2 (positions)
            if self.training:
                select = F.gumbel_softmax(scores, tau=self.tau, hard=True, dim=-1)  # [B, K, L]
            else:
                # At inference: hard argmax
                idx = scores.argmax(dim=-1)  # [B, K]
                select = torch.zeros_like(scores)
                select.scatter_(2, idx.unsqueeze(-1), 1.0)

            # Combined assignment: compete * select gives position ownership
            assign = compete * select  # [B, K, L]
            assign = assign / (assign.sum(dim=-1, keepdim=True) + 1e-8)

            # Gather slot representations from assigned positions
            gathered = torch.einsum('bkl,bld->bkd', assign, h)  # [B, K, D]

            # Update slots with GRU
            flat_slots = slots.reshape(B * self.n_slots, D)
            flat_gathered = gathered.reshape(B * self.n_slots, D)
            flat_updated = self.slot_gru(flat_gathered, flat_slots)
            slots = flat_updated.reshape(B, self.n_slots, D)

            if return_aux:
                aux_data["slot_assignments"].append(assign.detach())
                # Entropy of assignment over positions (lower = more peaked)
                probs = F.softmax(scores, dim=-1).clamp(1e-8)
                ent = -(probs * probs.log()).sum(-1).mean(-1).mean()  # scalar
                aux_data["slot_entropies"].append(float(ent))

        if not self.no_event:
            # Event detection: find the action verb position
            event_scores = torch.einsum('rd,bld->brl', self.event_q, h).squeeze(1)  # [B, L]
            event_scores = event_scores.masked_fill(pad, -1e9)
            if self.training:
                event_select = F.gumbel_softmax(event_scores.unsqueeze(1),
                                                tau=self.tau, hard=True, dim=-1).squeeze(1)
            else:
                eidx = event_scores.argmax(dim=-1)
                event_select = torch.zeros(B, L, device=dev)
                event_select.scatter_(1, eidx.unsqueeze(-1), 1.0)

            event_repr = torch.einsum('bl,bld->bd', event_select, h)  # [B, D]

            # Affectedness: which entity is affected?
            # [B, K, 2D] -> [B, K, 1] -> [B, K]
            affect_in = torch.cat([
                event_repr.unsqueeze(1).expand(-1, self.n_slots, -1),
                slots
            ], dim=-1)
            affect_logits = self.affect_mlp(affect_in).squeeze(-1)  # [B, K]

            if self.training:
                affect_select = F.gumbel_softmax(affect_logits, tau=self.tau,
                                                  hard=True, dim=-1)  # [B, K]
            else:
                aidx = affect_logits.argmax(dim=-1)
                affect_select = torch.zeros(B, self.n_slots, device=dev)
                affect_select.scatter_(1, aidx.unsqueeze(-1), 1.0)

            if permute_write:
                affect_select = affect_select.flip(1)

            # State update for affected entity
            affected = torch.einsum('bk,bkd->bd', affect_select, slots)  # [B, D]
            update_in = torch.cat([event_repr, affected], dim=-1)  # [B, 2D]
            new_state = self.state_update(update_in)  # [B, D]
            gate = torch.sigmoid(self.update_gate(update_in))  # [B, D]
            updated = affected + gate * (new_state - affected)

            # Apply update to only the affected entity slot
            slots = slots + affect_select.unsqueeze(-1) * (updated.unsqueeze(1) - slots)

        # Query at mask position
        mask_h = h[ar, mask_pos]  # [B, D]
        read_q = self.read_q(mask_h)  # [B, D]

        # Read: which entity is being queried?
        read_scores = torch.einsum('bd,bkd->bk', read_q, slots) * self.scale  # [B, K]
        read_attn = F.softmax(read_scores, dim=-1)  # [B, K]
        readout = torch.einsum('bk,bkd->bd', read_attn, slots)  # [B, D]

        # Combine with local context
        h = h.clone()
        h[ar, mask_pos] = self.norm(h[ar, mask_pos] + self.out_proj(readout))

        aux = {"read_attention": read_attn.detach(), **aux_data} if return_aux else {}
        return (h, aux) if return_aux else (h, None)


class TinyMLM(nn.Module):
    def __init__(self, vocab, arm, d=192, layers=4, heads=6, ff=768, max_len=64,
                 n_slots=2, n_iters=3):
        super().__init__()
        self.arm = arm
        self.word = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(max_len, d)
        self.embnorm = nn.LayerNorm(d)
        self.layers_list = nn.ModuleList([
            nn.TransformerEncoderLayer(d, heads, ff, 0.1, batch_first=True,
                                       norm_first=True, activation="gelu")
            for _ in range(layers)])
        if arm == "vanilla":
            self.memory = None
        elif arm == "discrete":
            self.memory = DiscreteSlotMemory(d, n_slots, n_iters, no_event=False)
        elif arm == "discrete_noevent":
            self.memory = DiscreteSlotMemory(d, n_slots, n_iters, no_event=True)
        else:
            raise ValueError(f"Unknown arm: {arm}")
        self.lm = nn.Sequential(nn.Linear(d, d), nn.GELU(),
                                nn.LayerNorm(d), nn.Linear(d, vocab, bias=False))
        self.lm[-1].weight = self.word.weight

    def forward(self, b, permute_write=False, return_aux=False):
        ids = b["input_ids"]; B, L = ids.shape
        pos = torch.arange(L, device=ids.device)[None]
        h = self.embnorm(self.word(ids) + self.pos(pos))
        pad = b["attention_mask"].eq(0)
        aux = {}
        for i, layer in enumerate(self.layers_list):
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
                         "family": m.get("family", ""), "is_affected": m.get("is_affected_query"),
                         "quartet_id": m.get("quartet_id", ""),
                         "correct": bool(ok[i]), "margin": float(margin[i]), "permuted": permute})
    summary = {k: {"correct": v[0], "n": v[1], "accuracy": v[0] / v[1], "margin_mean": v[2] / v[1]}
               for k, v in sums.items()}

    def selective(splitname):
        hr = [r for r in rows if r["split"] == splitname and r["kind"] == "binding"]
        if not hr: return None
        affected = {r["quartet_id"]: r["correct"] for r in hr if r.get("is_affected") is True}
        unaffected = {r["quartet_id"]: r["correct"] for r in hr if r.get("is_affected") is False}
        aff_acc = sum(affected.values()) / max(len(affected), 1)
        unaff_acc = sum(unaffected.values()) / max(len(unaffected), 1)
        qgroups = defaultdict(list)
        for r in hr:
            base = "_".join(r["quartet_id"].rsplit("_", 2)[:1])
            qgroups[base].append(r)
        pair_both = sum(1 for g in qgroups.values()
                        if any(r["correct"] for r in g if r.get("is_affected") is True)
                        and any(r["correct"] for r in g if r.get("is_affected") is False))
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
    ap.add_argument("--arm", choices=["vanilla", "discrete", "discrete_noevent"], required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--out-dir", default="")
    ap.add_argument("--seed", type=int, default=43022)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--max-len", type=int, default=64)
    ap.add_argument("--d-model", type=int, default=192)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--n-iters", type=int, default=3, help="Slot refinement iterations")
    ap.add_argument("--tau-init", type=float, default=2.0, help="Initial Gumbel temperature")
    ap.add_argument("--tau-final", type=float, default=0.3, help="Final Gumbel temperature")
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
    tl = DataLoader(train, batch_size=args.batch_size, shuffle=True, generator=g,
                    collate_fn=collate, num_workers=0)
    el = DataLoader(ev, batch_size=args.batch_size, shuffle=False, collate_fn=collate, num_workers=0)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TinyMLM(len(tok), args.arm, args.d_model, args.layers, max_len=args.max_len,
                    n_iters=args.n_iters).to(dev)
    nparams = sum(p.numel() for p in model.parameters())
    print(json.dumps({"arm": args.arm, "parameters": nparams, "device": str(dev),
                      "train_records": len(train), "eval_records": len(ev),
                      "n_iters": args.n_iters, "tau_init": args.tau_init, "tau_final": args.tau_final}),
          flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    losses = []; t0 = time.time()
    for ep in range(args.epochs):
        model.train()
        # Anneal Gumbel temperature
        if model.memory is not None:
            progress = ep / max(args.epochs - 1, 1)
            model.memory.tau = args.tau_init + (args.tau_final - args.tau_init) * progress

        total = 0.0; n = 0
        for b in tl:
            b = todev(b, dev); opt.zero_grad(set_to_none=True)
            logits, _ = model(b)
            ar = torch.arange(logits.shape[0], device=dev)
            loss = F.cross_entropy(logits[ar, b["mask_pos"]], b["answer_id"])
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            total += float(loss) * len(ar); n += len(ar)
        losses.append(total / n)
        tau_now = model.memory.tau if model.memory is not None else None
        print(json.dumps({"epoch": ep + 1, "loss": losses[-1], "tau": tau_now}), flush=True)

    base, rows = evaluate(model, el, dev, False)
    perm = {}
    if args.arm != "vanilla":
        perm, _ = evaluate(model, el, dev, True)

    # Compute write-perm deltas
    wp_deltas = {}
    for k in ["selective_eval_held_recomb", "selective_eval_held_paraphrase", "multi_event"]:
        if k in base and k in perm:
            field = "composite" if "selective" in k else "accuracy"
            wp_deltas[f"wp_delta_{k}"] = perm[k][field] - base[k][field]

    torch.save({"model": model.state_dict(), "args": vars(args)}, out / "model.pt")
    with (out / "eval_rows.jsonl").open("w") as f:
        for r in rows: f.write(json.dumps(r) + "\n")

    result = {
        "status": "DISCRETE_ROLE_INDUCTION",
        "arm": args.arm, "seed": args.seed, "device": str(dev), "dry_run": args.dry_run,
        "train_records": len(train), "eval_records": len(ev),
        "epochs": args.epochs, "losses": losses, "parameters": nparams,
        "n_iters": args.n_iters, "tau_init": args.tau_init, "tau_final": args.tau_final,
        "model_sha256": model_sha(model),
        "data_sha256": {"train": sha(Path(args.data_dir) / args.train_file),
                        "eval": sha(Path(args.data_dir) / args.eval_file)},
        "baseline": base, "write_permutation": perm,
        "wp_deltas": wp_deltas,
        "runtime_sec": time.time() - t0,
    }
    # Predeclared pass criteria
    hr = base.get("selective_eval_held_recomb", {})
    result["pass_criteria"] = {
        "held_recomb_above_0.70": hr.get("composite", 0) > 0.70,
        "affected_above_chance": hr.get("affected_accuracy", 0) > 0.52,
        "unaffected_above_chance": hr.get("unaffected_accuracy", 0) > 0.52,
        "paraphrase_above_0.55": base.get("selective_eval_held_paraphrase", {}).get("composite", 0) > 0.55,
        "wp_delta_below_neg0.15": wp_deltas.get("wp_delta_selective_eval_held_recomb", 0) < -0.15,
    }
    result["all_pass"] = all(result["pass_criteria"].values())
    result["route_decision"] = ("PASS: carry to DeBERTa" if result["all_pass"]
                                 else "FAIL: close TinyMLM memory variant route")

    (out / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
