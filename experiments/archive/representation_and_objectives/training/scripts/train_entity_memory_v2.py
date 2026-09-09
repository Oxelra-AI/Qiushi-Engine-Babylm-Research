#!/usr/bin/env python3
"""research: Counterbalanced entity-memory miniscreen with oracle/learned modes.

Fixes the research same-answer flaw. Both entities start in the same state;
the answer differs between affected and unaffected queries solely through
event-entity-query correspondence. No constant-prediction shortcut exists
(majority class 6.2%).

Arms: vanilla, shared_key, independent_key.
Modes: oracle (hard one-hot entity-to-slot), learned (attention-based).
Evaluation: selective-updating rate, quartet consistency, write-permutation.
"""
from __future__ import annotations
import argparse, hashlib, json, math, os, random, re, time
from collections import defaultdict
from pathlib import Path
import numpy as np, torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

STATES = {"empty","full","open","closed","clean","dirty","dry","wet",
          "cold","hot","dark","lit","flat","inflated","unlocked","locked"}

def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

def sha(p):
    h = hashlib.sha256()
    with open(p,"rb") as f:
        for c in iter(lambda: f.read(1<<20), b""): h.update(c)
    return h.hexdigest()

def model_sha(m):
    h = hashlib.sha256()
    for n,p in sorted(m.state_dict().items()):
        h.update(n.encode()); h.update(p.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()[:16]

def read_jsonl(p):
    return [json.loads(x) for x in Path(p).read_text("utf-8").splitlines() if x.strip()]

def find_token(offsets, text, term, start=0):
    m = re.search(r"\b"+re.escape(term)+r"\b", text[start:], re.I)
    if not m: return None, start
    a,b = start+m.start(), start+m.end()
    for i,(x,y) in enumerate(offsets):
        if x<b and y>a: return i,b
    return None,b


def positions(row, tok, max_len):
    text = row["text"]  # already contains <mask>
    enc = tok(text, padding="max_length", truncation=True,
              max_length=max_len, return_offsets_mapping=True)
    offs = enc.pop("offset_mapping")

    entities = row["entities"][:2]
    # Entity slot positions: first mentions
    slots = []
    for e in entities:
        p,_ = find_token(offs, text, e)
        slots.append(p if p is not None else 0)
    while len(slots) < 2: slots.append(slots[0] if slots else 0)
    slots = slots[:2]

    # Initial state positions (first two state-word tokens)
    state_hits = []
    for i,(a,b) in enumerate(offs):
        w = text[a:b].lower()
        if w in STATES and i not in state_hits: state_hits.append(i)
    if not state_hits: state_hits = [slots[0]]
    while len(state_hits) < 2: state_hits.append(state_hits[0])
    state_hits = state_hits[:2]

    # Event verb/entity positions
    ev_pos = []; cursor = 0
    for ev in row.get("events",[]):
        vp,end = find_token(offs, text, ev.get("verb",""), cursor)
        if vp is None: vp = slots[0]
        ep,end2 = find_token(offs, text, ev.get("entity",entities[0]), end or cursor)
        if ep is None: ep = slots[0]
        ev_pos.append((vp,ep)); cursor = end2 or cursor
    while len(ev_pos) < 3: ev_pos.append((-1,-1))
    ev_pos = ev_pos[:3]

    # Oracle: per-event actor slot and query slot
    actor_slots = []
    for ev in row.get("events",[]):
        ent = ev.get("entity","")
        if ent == entities[0]: actor_slots.append(0)
        elif len(entities) > 1 and ent == entities[1]: actor_slots.append(1)
        else: actor_slots.append(0)
    while len(actor_slots) < 3: actor_slots.append(-1)
    actor_slots = actor_slots[:3]

    qent = row.get("query_entity",entities[0])
    if qent == entities[0]: query_slot = 0
    elif len(entities) > 1 and qent == entities[1]: query_slot = 1
    else: query_slot = 0

    # Query entity position (last mention before mask)
    matches = list(re.finditer(r"\b"+re.escape(qent)+r"\b", text, re.I))
    qpos = slots[0]
    if matches:
        qa,qb = matches[-1].span()
        for i,(a,b) in enumerate(offs):
            if a < qb and b > qa: qpos = i; break

    mask_id = tok.mask_token_id
    try: mpos = enc["input_ids"].index(mask_id)
    except ValueError: raise ValueError(f"mask absent: {text}")

    # First discriminative BPE piece (skip shared space prefix)
    aid = tok.encode(" "+row["answer"], add_special_tokens=False)
    fid = tok.encode(" "+row["foil"], add_special_tokens=False)
    if len(aid) < 2 or len(fid) < 2:
        raise ValueError(f"target too short: {row['answer']}={aid} {row['foil']}={fid}")

    return {**enc, "slot_pos": slots, "state_pos": state_hits,
            "event_pos": ev_pos, "query_pos": qpos, "mask_pos": mpos,
            "answer_id": aid[1], "foil_id": fid[1],
            "actor_slots": actor_slots, "query_slot": query_slot}


class Rows(Dataset):
    def __init__(self, rows, tok, max_len):
        self.items = []
        for r in rows:
            x = positions(r, tok, max_len)
            x["meta"] = r
            self.items.append(x)
    def __len__(self): return len(self.items)
    def __getitem__(self,i): return self.items[i]


def collate(xs):
    out = {}
    for k in ["input_ids","attention_mask","slot_pos","state_pos","event_pos",
              "query_pos","mask_pos","answer_id","foil_id","actor_slots","query_slot"]:
        out[k] = torch.tensor([x[k] for x in xs], dtype=torch.long)
    out["meta"] = [x["meta"] for x in xs]
    return out


class EntityMemory(nn.Module):
    def __init__(self, d, arm, oracle=False):
        super().__init__()
        self.arm = arm; self.oracle = oracle
        self.scale = d**-0.5
        self.slot_key = nn.Linear(d,d,bias=False)
        if arm == "shared_key":
            self.write_query = self.slot_key
            self.read_query = self.slot_key
            self.capacity_match = nn.Parameter(torch.zeros(2,d,d))
        else:
            self.write_query = nn.Linear(d,d,bias=False)
            self.read_query = nn.Linear(d,d,bias=False)
            self.register_parameter("capacity_match",None)
        self.init_value = nn.Linear(d,d)
        self.event_value = nn.Sequential(nn.Linear(2*d,d),nn.GELU(),nn.Linear(d,d))
        self.gate = nn.Linear(2*d,1)
        self.out = nn.Linear(d,d)
        self.norm = nn.LayerNorm(d)

    def forward(self, h, batch, permute_write=False, return_aux=False):
        B,L,D = h.shape; dev = h.device
        ar = torch.arange(B, device=dev)
        sp = batch["slot_pos"]; st = batch["state_pos"]
        slot_h = h[ar[:,None],sp]
        keys = self.slot_key(slot_h)
        slots = self.init_value(h[ar[:,None],st])

        write_hist = []
        for t in range(batch["event_pos"].shape[1]):
            vp = batch["event_pos"][:,t,0]; ep = batch["event_pos"][:,t,1]
            valid = vp.ge(0)
            vpc = vp.clamp_min(0); epc = ep.clamp_min(0)
            verb = h[ar,vpc]; actor = h[ar,epc]

            if self.oracle:
                # Hard one-hot from ground truth
                aslot = batch["actor_slots"][:,t]  # (B,)
                wa = torch.zeros(B,2,device=dev)
                v = valid & aslot.ge(0)
                if v.any():
                    wa[v] = F.one_hot(aslot[v].clamp(0,1),2).float()
            else:
                q = self.write_query(actor)
                wa = torch.softmax((q[:,None,:]*keys).sum(-1)*self.scale, dim=-1)

            if permute_write: wa = wa.flip(1)

            old = (wa[:,:,None]*slots).sum(1)
            value = self.event_value(torch.cat([verb,old],-1))
            gate = torch.sigmoid(self.gate(torch.cat([verb,old],-1)))
            updated = old + gate*(value - old)
            slots = slots + valid[:,None,None]*wa[:,:,None]*(updated[:,None,:]-slots)
            write_hist.append(wa)

        qp = batch["query_pos"]; query = h[ar,qp]
        if self.oracle:
            qslot = batch["query_slot"]
            ra = F.one_hot(qslot.clamp(0,1),2).float()
        else:
            rq = self.read_query(query)
            ra = torch.softmax((rq[:,None,:]*keys).sum(-1)*self.scale, dim=-1)

        read = (ra[:,:,None]*slots).sum(1)
        mp = batch["mask_pos"]
        h = h.clone()
        h[ar,mp] = self.norm(h[ar,mp]+self.out(read))

        aux = {"write_attention": torch.stack(write_hist,1) if write_hist else torch.zeros(B,0,2,device=dev),
               "read_attention": ra, "slots": slots}
        return (h,aux) if return_aux else (h,None)


class TinyMLM(nn.Module):
    def __init__(self, vocab, arm, oracle=False, d=192, layers=4, heads=6, ff=768, max_len=64):
        super().__init__()
        self.arm = arm
        self.word = nn.Embedding(vocab,d)
        self.pos = nn.Embedding(max_len,d)
        self.embnorm = nn.LayerNorm(d)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d,heads,ff,0.1,batch_first=True,norm_first=True,activation="gelu")
            for _ in range(layers)])
        self.memory = None if arm=="vanilla" else EntityMemory(d,arm,oracle)
        self.lm = nn.Sequential(nn.Linear(d,d),nn.GELU(),nn.LayerNorm(d),nn.Linear(d,vocab,bias=False))
        self.lm[-1].weight = self.word.weight

    def forward(self, b, permute_write=False, return_aux=False):
        ids = b["input_ids"]; B,L = ids.shape
        pos = torch.arange(L,device=ids.device)[None]
        h = self.embnorm(self.word(ids)+self.pos(pos))
        pad = b["attention_mask"].eq(0); aux = {}
        for i,layer in enumerate(self.layers):
            h = layer(h, src_key_padding_mask=pad)
            if i == 1 and self.memory is not None:
                h,aux = self.memory(h,b,permute_write,return_aux)
        return self.lm(h), aux


def todev(b, dev):
    return {k:(v.to(dev) if torch.is_tensor(v) else v) for k,v in b.items()}


@torch.no_grad()
def evaluate(model, loader, dev, permute=False):
    model.eval()
    rows = []; sums = defaultdict(lambda:[0,0,0.0])
    wa_list, ra_list = [], []
    for b in loader:
        meta = b["meta"]; b = todev(b,dev)
        logits, aux = model(b, permute_write=permute, return_aux=True)
        ar = torch.arange(logits.shape[0],device=dev)
        z = logits[ar, b["mask_pos"]]
        margin = z[ar,b["answer_id"]] - z[ar,b["foil_id"]]
        ok = margin.gt(0)
        for i,m in enumerate(meta):
            group = m["split"]
            sums[group][0] += int(ok[i]); sums[group][1] += 1; sums[group][2] += float(margin[i])
            rows.append({"id":m["id"],"split":group,"kind":m["kind"],
                         "family":m["family"],"is_affected":m.get("is_affected_query"),
                         "quartet_id":m.get("quartet_id",""),
                         "correct":bool(ok[i]),"margin":float(margin[i]),
                         "permuted":permute})
        if aux:
            if aux.get("write_attention") is not None and aux["write_attention"].numel():
                wa_list.append(aux["write_attention"].cpu())
            if aux.get("read_attention") is not None:
                ra_list.append(aux["read_attention"].cpu())

    summary = {k:{"correct":v[0],"n":v[1],"accuracy":v[0]/v[1],"margin_mean":v[2]/v[1]}
               for k,v in sums.items()}

    # Selective-updating metrics for binding quartets
    held_rows = [r for r in rows if r["split"]=="eval_held_recomb" and r["kind"]=="binding"]
    if held_rows:
        affected = {r["quartet_id"]:r["correct"] for r in held_rows if r.get("is_affected")==True}
        unaffected = {r["quartet_id"]:r["correct"] for r in held_rows if r.get("is_affected")==False}
        aff_acc = sum(affected.values())/max(len(affected),1)
        unaff_acc = sum(unaffected.values())/max(len(unaffected),1)
        # Group by quartet base
        qgroups = defaultdict(list)
        for r in held_rows:
            base = "_".join(r["quartet_id"].rsplit("_",2)[:1])
            qgroups[base].append(r)
        pair_both = sum(1 for g in qgroups.values()
                        if any(r["correct"] for r in g if r.get("is_affected")==True)
                        and any(r["correct"] for r in g if r.get("is_affected")==False))
        quartet_all = sum(1 for g in qgroups.values() if all(r["correct"] for r in g))
        summary["selective_updating"] = {
            "affected_accuracy": aff_acc,
            "unaffected_accuracy": unaff_acc,
            "pair_both_correct": pair_both,
            "pair_total": len(qgroups),
            "pair_both_rate": pair_both/max(len(qgroups),1),
            "quartet_all_correct": quartet_all,
            "quartet_all_rate": quartet_all/max(len(qgroups),1),
        }

    # Multi-event metrics
    multi_rows = [r for r in rows if r["kind"]=="multi_event"]
    if multi_rows:
        multi_correct = sum(r["correct"] for r in multi_rows)
        summary["multi_event"] = {"correct":multi_correct,"n":len(multi_rows),
                                   "accuracy":multi_correct/len(multi_rows)}

    transport = {}
    if ra_list:
        r = torch.cat(ra_list)
        transport["read_slot0_mean"] = float(r[:,0].mean())
        transport["read_entropy_mean"] = float((-(r*r.clamp_min(1e-9).log()).sum(-1)).mean())
    if wa_list:
        w = torch.cat(wa_list)
        transport["write_entropy_mean"] = float((-(w*w.clamp_min(1e-9).log()).sum(-1)).mean())
    return summary, transport, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["vanilla","shared_key","independent_key"], required=True)
    ap.add_argument("--oracle", action="store_true", help="Oracle entity-to-slot (hard one-hot)")
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--out-dir", default="")
    ap.add_argument("--seed", type=int, default=43022)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--max-len", type=int, default=64)
    ap.add_argument("--d-model", type=int, default=192)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    # Oracle only applies to memory arms
    oracle = args.oracle and args.arm != "vanilla"
    seed_all(args.seed)

    out = Path(args.out_dir or os.environ.get("QIUSHI_AI_LAB_RUN_DIR","."))
    out.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True, local_files_only=True)
    train_rows = read_jsonl(Path(args.data_dir)/"train.jsonl")
    eval_rows = read_jsonl(Path(args.data_dir)/"eval.jsonl")
    if args.dry_run:
        train_rows = train_rows[:64]; eval_rows = eval_rows[:64]; args.epochs = 1

    train = Rows(train_rows, tok, args.max_len)
    ev = Rows(eval_rows, tok, args.max_len)
    g = torch.Generator().manual_seed(args.seed)
    tl = DataLoader(train, batch_size=args.batch_size, shuffle=True, generator=g, collate_fn=collate, num_workers=0)
    el = DataLoader(ev, batch_size=args.batch_size, shuffle=False, collate_fn=collate, num_workers=0)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TinyMLM(len(tok), args.arm, oracle, args.d_model, args.layers, max_len=args.max_len).to(dev)
    nparams = sum(p.numel() for p in model.parameters())
    label = f"{args.arm}{'_oracle' if oracle else '_learned' if args.arm!='vanilla' else ''}"
    print(json.dumps({"arm":args.arm,"oracle":oracle,"label":label,"parameters":nparams,
                      "device":str(dev),"train_records":len(train),"eval_records":len(ev)}),flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    losses = []; t0 = time.time()
    for ep in range(args.epochs):
        model.train(); total=0.0; n=0
        for b in tl:
            b = todev(b,dev); opt.zero_grad(set_to_none=True)
            logits,_ = model(b)
            ar = torch.arange(logits.shape[0],device=dev)
            loss = F.cross_entropy(logits[ar,b["mask_pos"]], b["answer_id"])
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
            total += float(loss)*len(ar); n += len(ar)
        losses.append(total/n)
        print(json.dumps({"epoch":ep+1,"loss":losses[-1]}),flush=True)

    base, transport, rows = evaluate(model, el, dev, False)
    if args.arm == "vanilla":
        perm, perm_transport = {}, {}
    else:
        perm, perm_transport, _ = evaluate(model, el, dev, True)

    torch.save({"model":model.state_dict(),"args":vars(args)}, out/"model.pt")
    with (out/"eval_rows.jsonl").open("w") as f:
        for r in rows: f.write(json.dumps(r)+"\n")

    result = {
        "status":"COUNTERBALANCED_MINISCREEN",
        "arm":args.arm, "oracle":oracle, "label":label,
        "seed":args.seed, "device":str(dev), "dry_run":args.dry_run,
        "train_records":len(train), "eval_records":len(ev),
        "epochs":args.epochs, "losses":losses, "parameters":nparams,
        "model_sha256":model_sha(model),
        "data_sha256":{"train":sha(Path(args.data_dir)/"train.jsonl"),
                       "eval":sha(Path(args.data_dir)/"eval.jsonl")},
        "baseline":base, "transport":transport,
        "write_permutation":perm, "write_permutation_transport":perm_transport,
        "runtime_sec":time.time()-t0,
    }
    (out/"result.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2),flush=True)


if __name__ == "__main__":
    main()
