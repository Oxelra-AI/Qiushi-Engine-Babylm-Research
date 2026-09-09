#!/usr/bin/env python3
"""research original-surface alias probe.

The main research test used minimal predicate sentences and expanded each event
through all templates to isolate relation/coreference learning. This companion
uses the actual research score-ablated sentence surfaces and their original
assigned templates. It asks whether per-item random aliases repair the fixed-name
shortcut on the source-like surface.
"""

from __future__ import annotations

import json
import math
import random
import re
import time
import hashlib
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

WORKSPACE = Path("experiments/archive/representation_and_objectives")
PILOT_DIR = WORKSPACE / "data/paired_world_pilot"
OUT_DIR = WORKSPACE / "data/original_surface_alias_probe"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ANCHOR_TEMPLATES = {1, 2, 3, 6, 7, 8, 11, 12, 13, 14}
PROBE_TEMPLATES = {4, 5, 9, 10, 15}
HELD_TEMPLATES = {16, 17, 18, 19, 20}
ALIAS_MODES = ["fixed_names", "per_item_alias"]
ARM_K = [("zero", 0), ("true", 8), ("shuffled", 8), ("exposure", 8)]
N_SEEDS = 5
MAX_LEN = 80
EMB_DIM = 48
HIDDEN_DIM = 48
EPOCHS = 22
BATCH_SIZE = 64
LR = 0.0025
TRAIN_FIT = 0.94
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

ALIAS_POOL = [
    "ava", "ben", "cai", "dia", "eli", "fay", "gus", "hao", "ivy", "jen", "kai", "lia", "max", "nia", "ori", "pia",
    "qio", "rae", "sam", "tia", "uma", "vex", "wye", "xia", "yan", "zoe", "ari", "bea", "cal", "dee", "eno", "flo",
    "gio", "hal", "ian", "jia", "kim", "len", "moe", "noa", "ola", "paz", "quin", "rio", "sue", "tam", "ulo", "viv",
    "wes", "xim", "yul", "zed", "alba", "brio", "cora", "dune", "emil", "fern", "gale", "hera", "isla", "juno", "kora", "luma",
]


def stable_int(s: str) -> int:
    return int(hashlib.sha256(s.encode("utf-8")).hexdigest()[:12], 16)


def tokenize(text: str) -> List[str]:
    text = text.lower().replace("[sep]", " [sep] ")
    for ch in ".,;:!?()/-":
        text = text.replace(ch, f" {ch} ")
    return text.split()


def load_families() -> List[dict]:
    fams = []
    for split in ["train", "held"]:
        p = PILOT_DIR / f"families_{split}.jsonl"
        for line in p.read_text().splitlines():
            if line.strip():
                fams.append(json.loads(line))
    return fams


def classify(tid: int) -> str:
    if tid in ANCHOR_TEMPLATES:
        return "anchor"
    if tid in PROBE_TEMPLATES:
        return "probe"
    if tid in HELD_TEMPLATES:
        return "held"
    return "other"


def item_alias(row_key: str) -> Tuple[str, str]:
    rng = random.Random(stable_int("ORIGSURF|" + row_key))
    return tuple(rng.sample(ALIAS_POOL, 2))  # type: ignore


def replace_names(text: str, pa: str, pb: str, aa: str, bb: str) -> str:
    # Exact replacement is enough for research generated strings. Longer first.
    pairs = sorted([(pa, aa), (pb, bb)], key=lambda x: -len(x[0]))
    out = text
    for old, new in pairs:
        out = out.replace(old, new)
    return out


def make_rows(fams: List[dict], mode: str) -> List[dict]:
    rows = []
    for fam in fams:
        pa, pb = fam["participant_a"], fam["participant_b"]
        for ck in ["context1", "context2"]:
            cx = fam[ck]
            tid = int(cx["template_id"])
            wl = cx["winner_label"]
            for hd in ["AB", "BA"]:
                row_key = f"{mode}|{fam['family_id']}|{ck}|T{tid}|{hd}"
                if mode == "fixed_names":
                    aa, bb = pa, pb
                elif mode == "per_item_alias":
                    aa, bb = item_alias(row_key)
                else:
                    raise ValueError(mode)
                context = replace_names(cx["text_score_ablated"], pa, pb, aa, bb)
                hyp = f"{aa} defeated {bb}." if hd == "AB" else f"{bb} defeated {aa}."
                label = 1 if (hd == "AB" and wl == "A") or (hd == "BA" and wl == "B") else 0
                rows.append({
                    "row_key": row_key,
                    "family_id": fam["family_id"],
                    "family_split": fam["family_split"],
                    "tid": tid,
                    "group": classify(tid),
                    "mode": mode,
                    "text": f"{context} [SEP] {hyp}",
                    "label": label,
                })
    return rows


def build_vocab(rows: List[dict]) -> Dict[str, int]:
    cnt = Counter()
    for r in rows:
        cnt.update(tokenize(r["text"]))
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for tok, _ in cnt.most_common():
        vocab[tok] = len(vocab)
    return vocab


def encode_rows(rows: List[dict], vocab: Dict[str, int]) -> Tuple[List[List[int]], List[int]]:
    xs, ys = [], []
    for r in rows:
        toks = tokenize(r["text"])[:MAX_LEN]
        ids = [vocab.get(t, 1) for t in toks] + [0] * (MAX_LEN - len(toks))
        xs.append(ids)
        ys.append(int(r["label"]))
    return xs, ys


class BiGRU(nn.Module):
    def __init__(self, vocab_size: int):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, EMB_DIM, padding_idx=0)
        self.gru = nn.GRU(EMB_DIM, HIDDEN_DIM, batch_first=True, bidirectional=True)
        self.attn = nn.Linear(HIDDEN_DIM * 2, 1)
        self.head = nn.Sequential(nn.Dropout(0.05), nn.Linear(HIDDEN_DIM * 2, HIDDEN_DIM), nn.ReLU(), nn.Linear(HIDDEN_DIM, 1))
    def forward(self, x):
        mask = x.ne(0)
        out, _ = self.gru(self.emb(x))
        s = self.attn(out).squeeze(-1).masked_fill(~mask, -1e9)
        w = torch.softmax(s, dim=-1)
        return self.head((out * w.unsqueeze(-1)).sum(1)).squeeze(-1)


def acc(logits, y):
    return logits.gt(0).float().eq(y).float().mean().item()


def train_eval(train_rows: List[dict], eval_sets: Dict[str, List[dict]], vocab: Dict[str, int], seed: int) -> Dict[str, Any]:
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    x, y = encode_rows(train_rows, vocab)
    tx = torch.tensor(x, dtype=torch.long, device=DEVICE)
    ty = torch.tensor(y, dtype=torch.float32, device=DEVICE)
    model = BiGRU(len(vocab)).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    loader = DataLoader(TensorDataset(tx, ty), batch_size=BATCH_SIZE, shuffle=True)
    best_state, best_acc, best_loss = None, -1.0, 1e9
    for ep in range(1, EPOCHS + 1):
        model.train()
        for xb, yb in loader:
            loss = F.binary_cross_entropy_with_logits(model(xb), yb)
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0); opt.step(); opt.zero_grad(set_to_none=True)
        model.eval()
        with torch.no_grad():
            logits = model(tx)
            tr_acc = acc(logits, ty)
            tr_loss = F.binary_cross_entropy_with_logits(logits, ty).item()
        if tr_acc > best_acc + 1e-5 or (abs(tr_acc - best_acc) < 1e-5 and tr_loss < best_loss):
            best_acc, best_loss = tr_acc, tr_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if best_acc >= 0.999:
            break
    if best_state:
        model.load_state_dict({k: v.to(DEVICE) for k, v in best_state.items()})
    model.eval()
    res: Dict[str, Any] = {"train_acc": best_acc, "train_loss": best_loss, "epochs": ep}
    with torch.no_grad():
        for name, rows in eval_sets.items():
            if not rows:
                res[name] = float("nan")
            else:
                ex, ey = encode_rows(rows, vocab)
                ex_t = torch.tensor(ex, dtype=torch.long, device=DEVICE)
                ey_t = torch.tensor(ey, dtype=torch.float32, device=DEVICE)
                res[name] = acc(model(ex_t), ey_t)
    return res


def choose_sparse(probe_rows: List[dict], k: int, arm: str, seed: int) -> List[dict]:
    if k <= 0:
        return []
    rng = random.Random(seed)
    by_tid_event = defaultdict(list)
    for r in probe_rows:
        by_tid_event[(r["tid"], r["family_id"], r["row_key"].split("|")[2])].append(r)
    # group by original context/event so AB+BA travel together
    by_tid = defaultdict(list)
    for (tid, _, _), group in by_tid_event.items():
        by_tid[tid].append(group)
    out = []
    for tid, groups in by_tid.items():
        rng.shuffle(groups)
        for group in groups[: min(k, len(groups))]:
            for r0 in group:
                r = dict(r0)
                if arm == "true":
                    pass
                elif arm == "shuffled":
                    r["label"] = 1 - r["label"]
                elif arm == "exposure":
                    bit = stable_int(f"ORIGEXP|{seed}|{r['row_key']}") % 2
                    r["label"] = bit
                else:
                    raise ValueError(arm)
                out.append(r)
    return out


def summarize(vals: List[float]) -> Dict[str, Any]:
    vals = [float(v) for v in vals if not (isinstance(v, float) and math.isnan(v))]
    return {"mean": round(float(np.mean(vals)), 4), "std": round(float(np.std(vals)), 4), "n": len(vals), "values": [round(v, 4) for v in vals]} if vals else {"mean": float("nan"), "std": float("nan"), "n": 0}


def main():
    t0 = time.time()
    if DEVICE == "cuda":
        torch.cuda.set_device(0)
    torch.set_num_threads(min(16, max(1, torch.get_num_threads())))
    fams = load_families()
    raw = []
    metadata = {"device": DEVICE, "arm_k": ARM_K, "modes": ALIAS_MODES, "n_seeds": N_SEEDS, "model": {"emb_dim": EMB_DIM, "hidden_dim": HIDDEN_DIM, "epochs": EPOCHS, "batch_size": BATCH_SIZE, "lr": LR}}
    print(json.dumps({"status": "ORIGINAL_SURFACE_ALIAS_PROBE_START", **metadata}), flush=True)
    for mode in ALIAS_MODES:
        rows = make_rows(fams, mode)
        vocab = build_vocab(rows)
        anchor_train = [r for r in rows if r["family_split"] == "train" and r["group"] == "anchor"]
        probe_train = [r for r in rows if r["family_split"] == "train" and r["group"] == "probe"]
        eval_sets = {
            "anchor_heldfam": [r for r in rows if r["family_split"] == "held" and r["group"] == "anchor"],
            "probe_heldfam": [r for r in rows if r["family_split"] == "held" and r["group"] == "probe"],
            "heldtemplate_heldfam": [r for r in rows if r["family_split"] == "held" and r["group"] == "held"],
            "probe_trainfam": probe_train,
        }
        for tid in sorted(PROBE_TEMPLATES):
            eval_sets[f"probe_T{tid:02d}_heldfam"] = [r for r in rows if r["family_split"] == "held" and r["tid"] == tid]
        print(f"\n=== {mode} vocab={len(vocab)} anchor_train={len(anchor_train)} probe_train={len(probe_train)} ===", flush=True)
        for arm, k in ARM_K:
            for seed_idx in range(N_SEEDS):
                seed = 257300 + seed_idx * 1013 + k * 29 + stable_int(mode + arm) % 701
                sparse = choose_sparse(probe_train, k, arm if arm != "zero" else "true", seed) if k else []
                train_rows = anchor_train + sparse
                res = train_eval(train_rows, eval_sets, vocab, seed)
                res.update({"mode": mode, "arm": arm, "k_per_template": k, "seed_idx": seed_idx, "seed": seed, "n_train_rows": len(train_rows), "n_sparse_rows": len(sparse), "train_fit": bool(res["train_acc"] >= TRAIN_FIT)})
                raw.append(res)
                print(json.dumps({"mode": mode, "arm": arm, "k": k, "seed_idx": seed_idx, "train": round(res["train_acc"], 4), "probeH": round(res["probe_heldfam"], 4), "heldT": round(res["heldtemplate_heldfam"], 4), "fit": res["train_fit"]}), flush=True)
    grouped = defaultdict(list)
    for r in raw:
        grouped[(r["mode"], r["arm"], r["k_per_template"])].append(r)
    metrics = ["train_acc", "anchor_heldfam", "probe_heldfam", "probe_trainfam", "heldtemplate_heldfam"] + [f"probe_T{tid:02d}_heldfam" for tid in sorted(PROBE_TEMPLATES)]
    summary = {}
    for (mode, arm, k), items in grouped.items():
        key = f"{mode}|{arm}|k{k}"
        summary[key] = {"mode": mode, "arm": arm, "k_per_template": k, "n_runs": len(items), "fit_count": sum(x["train_fit"] for x in items)}
        for m in metrics:
            summary[key][m] = summarize([x[m] for x in items if m in x])
    out = {"status": "ORIGINAL_SURFACE_ALIAS_PROBE", "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "elapsed_seconds": round(time.time()-t0, 2), "metadata": metadata, "summary": summary, "raw": raw}
    out_json = OUT_DIR / "original_surface_alias_probe_summary.json"
    out_json.write_text(json.dumps(out, indent=2, default=str))
    lines = ["# research original-surface alias probe\n\n", f"Device `{DEVICE}`, elapsed {out['elapsed_seconds']} s. Uses actual research score-ablated sentences and assigned templates.\n\n", "| mode | arm | k/template | fit | train | probe-held | held-template |\n", "|---|---:|---:|---:|---:|---:|---:|\n"]
    for mode in ALIAS_MODES:
        for arm, k in ARM_K:
            s = summary[f"{mode}|{arm}|k{k}"]
            lines.append(f"| {mode} | {arm} | {k} | {s['fit_count']}/{s['n_runs']} | {s['train_acc']['mean']:.3f} | {s['probe_heldfam']['mean']:.3f} | {s['heldtemplate_heldfam']['mean']:.3f} |\n")
    lines.append(f"\nSummary JSON: `{out_json}`\n")
    ((OUT_DIR.parents[4] / 'research/documents/representation_and_objectives/data/original_surface_alias_probe/original_surface_alias_probe_summary.md')).write_text("".join(lines))
    print("\nORIGINAL-SURFACE ALIAS PROBE SUMMARY")
    for mode in ALIAS_MODES:
        for arm, k in ARM_K:
            s = summary[f"{mode}|{arm}|k{k}"]
            print(f"{mode:<15s} {arm:<9s} k={k:<2d} fit={s['fit_count']}/{s['n_runs']} train={s['train_acc']['mean']:.3f} probeH={s['probe_heldfam']['mean']:.3f} heldT={s['heldtemplate_heldfam']['mean']:.3f}")
    print(json.dumps({"status": out["status"], "elapsed_seconds": out["elapsed_seconds"], "summary_json": str(out_json)}, indent=2))


if __name__ == "__main__":
    main()
