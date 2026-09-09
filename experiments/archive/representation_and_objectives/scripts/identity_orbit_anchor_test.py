#!/usr/bin/env python3
"""research: identity-orbit randomization x sparse role-coordinate anchors.

Scientific purpose
------------------
research showed that the fixed-name sports construction admits a dominant
identity-specific solution: a BiGRU can train well without transferring predicate
meaning to new names. The proposed repair randomizes the entity identity
orbit while preserving within-item coreference: resample fresh aliases for each
encoded item, compare against fixed natural names and family-stable aliases, and
cross this with correctly aligned versus anti-aligned/exposure-matched sparse
probe anchors.

This script is a deliberately cheap raw-text diagnostic. It uses minimal relation
sentences derived from the research source-attested paired worlds, removing dates,
tournaments, scores, and other event identifiers so that the remaining problem is
relation-role parsing and entity coreference.

Load-bearing outcomes:
  * fixed_names can fit by entity identity and need not transfer;
  * per_item_alias must fit anchor predicates by reusable predicate/coreference
    structure, because aliases are resampled independently per row;
  * if sparse true probe anchors beat anti-aligned and random exposure controls on
    held-family probe rows, then identity-orbit randomization + absolute
    role-coordinate anchoring is a live data-efficient principle candidate.
"""

from __future__ import annotations

import json
import math
import random
import time
import hashlib
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

WORKSPACE = Path("experiments/archive/representation_and_objectives")
PILOT_DIR = WORKSPACE / "data/paired_world_pilot"
OUT_DIR = WORKSPACE / "data/identity_orbit_anchor_test"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Minimal predicate templates: no date/tournament/score/event-id channels.
# first is only metadata for reading; labels always follow W/L arguments.
TEMPLATES: Dict[int, Tuple[str, str, str]] = {
    1:  ("{W} defeated {L}.", "w", "anchor"),
    2:  ("{W} beat {L}.", "w", "anchor"),
    3:  ("{W} won against {L}.", "w", "anchor"),
    4:  ("{W} overcame {L}.", "w", "probe"),
    5:  ("{W} proved too strong for {L}.", "w", "probe"),
    6:  ("{L} lost to {W}.", "l", "anchor"),
    7:  ("{L} fell to {W}.", "l", "anchor"),
    8:  ("{L} was defeated by {W}.", "l", "anchor"),
    9:  ("{L} was beaten by {W}.", "l", "probe"),
    10: ("{L} was unable to overcome {W}.", "l", "probe"),
    11: ("{W} triumphed over {L}.", "m", "anchor"),
    12: ("{W} emerged victorious over {L}.", "m", "anchor"),
    13: ("{W} prevailed against {L}.", "m", "anchor"),
    14: ("{W} came out on top against {L}.", "m", "anchor"),
    15: ("{W} was victorious over {L}.", "m", "probe"),
    16: ("{W} edged out {L}.", "w", "held"),
    17: ("{L} succumbed to {W}.", "l", "held"),
    18: ("The match resulted in a victory for {W} over {L}.", "m", "held"),
    19: ("{W} claimed the win against {L}.", "w", "held"),
    20: ("{L} went down to {W}.", "l", "held"),
}
ANCHOR_TEMPLATES = [tid for tid, (_, _, g) in TEMPLATES.items() if g == "anchor"]
PROBE_TEMPLATES = [tid for tid, (_, _, g) in TEMPLATES.items() if g == "probe"]
HELD_TEMPLATES = [tid for tid, (_, _, g) in TEMPLATES.items() if g == "held"]

# Low-cost but enough to see the representation/interface effect.
TRAIN_EVENT_LIMIT = 240       # contexts, not families; anchor rows = limit*10*2
EVAL_TRAIN_EVENT_LIMIT = 160  # same-family probe memory surface
EVAL_HELD_EVENT_LIMIT = 160   # new-family transfer surface
K_PER_PROBE_TEMPLATE = [0, 2, 8, 32]
ALIAS_MODES = ["fixed_names", "family_alias", "per_item_alias"]
ANCHOR_ARMS = ["true", "shuffled", "exposure"]  # shuffled = systematic anti-coordinate
N_SEEDS = 2
MAX_LEN = 32
EMB_DIM = 40
HIDDEN_DIM = 48
EPOCHS = 18
BATCH_SIZE = 128
LR = 0.003
TRAIN_FIT = 0.97
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# A finite alias vocabulary: every alias appears in many train/eval rows, but
# per-item resampling makes identity uninformative.
ALIAS_POOL = [
    "ava", "ben", "cai", "dia", "eli", "fay", "gus", "hao", "ivy", "jen", "kai", "lia", "max", "nia", "ori", "pia",
    "qio", "rae", "sam", "tia", "uma", "vex", "wye", "xia", "yan", "zoe", "ari", "bea", "cal", "dee", "eno", "flo",
    "gio", "hal", "ian", "jia", "kim", "len", "moe", "noa", "ola", "paz", "quin", "rio", "sue", "tam", "ulo", "viv",
    "wes", "xim", "yul", "zed", "alba", "brio", "cora", "dune", "emil", "fern", "gale", "hera", "isla", "juno", "kora", "luma",
]


def stable_int(s: str) -> int:
    return int(hashlib.sha256(s.encode("utf-8")).hexdigest()[:12], 16)


def tokenize(text: str) -> List[str]:
    # Keep words/punctuation simple; aliases and natural names become tokens.
    text = text.lower().replace("[sep]", " [sep] ")
    for ch in ".,;:!?()":
        text = text.replace(ch, f" {ch} ")
    return text.split()


def load_families() -> Tuple[List[dict], List[dict]]:
    fams = []
    for split in ["train", "held"]:
        p = PILOT_DIR / f"families_{split}.jsonl"
        for line in p.read_text().splitlines():
            if line.strip():
                fams.append(json.loads(line))
    train = [f for f in fams if f["family_split"] == "train"]
    held = [f for f in fams if f["family_split"] == "held"]
    return train, held


def extract_events(families: List[dict]) -> List[dict]:
    events = []
    for fam in families:
        for ck in ["context1", "context2"]:
            cx = fam[ck]
            events.append({
                "event_id": f"{fam['family_id']}::{ck}",
                "family_id": fam["family_id"],
                "participant_a": fam["participant_a"],
                "participant_b": fam["participant_b"],
                "winner_label": cx["winner_label"],
            })
    return events


def family_alias_map(family_id: str) -> Tuple[str, str]:
    # Stable within a family, drawn from the shared alias pool so held-family alias
    # tokens are still trained elsewhere. Avoid same alias for A/B.
    i = stable_int("A::" + family_id) % len(ALIAS_POOL)
    j = stable_int("B::" + family_id) % (len(ALIAS_POOL) - 1)
    if j >= i:
        j += 1
    return ALIAS_POOL[i], ALIAS_POOL[j]


def item_alias_map(row_key: str) -> Tuple[str, str]:
    # Fresh deterministic aliases per encoded item, consistent inside that item.
    rng = random.Random(stable_int("ITEM::" + row_key))
    a, b = rng.sample(ALIAS_POOL, 2)
    return a, b


def aliases_for(mode: str, ev: dict, row_key: str) -> Tuple[str, str]:
    if mode == "fixed_names":
        return ev["participant_a"], ev["participant_b"]
    if mode == "family_alias":
        return family_alias_map(ev["family_id"])
    if mode == "per_item_alias":
        return item_alias_map(row_key)
    raise ValueError(mode)


def render_context(tid: int, ev: dict, alias_a: str, alias_b: str) -> str:
    wl = ev["winner_label"]
    if wl == "A":
        W, L = alias_a, alias_b
    else:
        W, L = alias_b, alias_a
    return TEMPLATES[tid][0].format(W=W, L=L)


def render_row(ev: dict, tid: int, hyp_dir: str, mode: str, namespace: str) -> dict:
    # hyp_dir AB means hypothesis says participant A defeated participant B;
    # BA means participant B defeated participant A.
    row_key = f"{namespace}|{ev['event_id']}|T{tid}|{hyp_dir}|{mode}"
    aa, bb = aliases_for(mode, ev, row_key)
    context = render_context(tid, ev, aa, bb)
    hyp = f"{aa} defeated {bb}." if hyp_dir == "AB" else f"{bb} defeated {aa}."
    label = 1 if (hyp_dir == "AB" and ev["winner_label"] == "A") or (hyp_dir == "BA" and ev["winner_label"] == "B") else 0
    return {
        "row_key": row_key,
        "mode": mode,
        "event_id": ev["event_id"],
        "family_id": ev["family_id"],
        "tid": tid,
        "template_group": TEMPLATES[tid][2],
        "hyp_dir": hyp_dir,
        "text": f"{context} [SEP] {hyp}",
        "label": label,
    }


def make_rows(events: List[dict], tids: List[int], mode: str, namespace: str) -> List[dict]:
    rows = []
    for ev in events:
        for tid in tids:
            for hd in ["AB", "BA"]:
                rows.append(render_row(ev, tid, hd, mode, namespace))
    return rows


def build_vocab(rows_by_mode: Dict[str, List[dict]]) -> Dict[str, Dict[str, int]]:
    vocabs = {}
    for mode, rows in rows_by_mode.items():
        cnt = Counter()
        for r in rows:
            cnt.update(tokenize(r["text"]))
        vocab = {"<PAD>": 0, "<UNK>": 1}
        for tok, _ in cnt.most_common():
            if tok not in vocab:
                vocab[tok] = len(vocab)
        vocabs[mode] = vocab
    return vocabs


def encode_rows(rows: List[dict], vocab: Dict[str, int]) -> Tuple[List[List[int]], List[int]]:
    ids = []
    labs = []
    for r in rows:
        toks = tokenize(r["text"])[:MAX_LEN]
        arr = [vocab.get(t, 1) for t in toks]
        arr += [0] * (MAX_LEN - len(arr))
        ids.append(arr)
        labs.append(int(r["label"]))
    return ids, labs


class BiGRU(nn.Module):
    def __init__(self, vocab_size: int):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, EMB_DIM, padding_idx=0)
        self.gru = nn.GRU(EMB_DIM, HIDDEN_DIM, batch_first=True, bidirectional=True)
        self.attn = nn.Linear(HIDDEN_DIM * 2, 1)
        self.head = nn.Sequential(
            nn.Linear(HIDDEN_DIM * 2, HIDDEN_DIM), nn.ReLU(), nn.Linear(HIDDEN_DIM, 1)
        )

    def forward(self, x):
        mask = x.ne(0)
        out, _ = self.gru(self.emb(x))
        score = self.attn(out).squeeze(-1).masked_fill(~mask, -1e9)
        w = torch.softmax(score, dim=-1)
        pooled = (out * w.unsqueeze(-1)).sum(dim=1)
        return self.head(pooled).squeeze(-1)


def accuracy_for_logits(logits: torch.Tensor, y: torch.Tensor) -> float:
    return (logits.gt(0).float().eq(y).float().mean().item())


def train_eval(train_rows: List[dict], eval_sets: Dict[str, List[dict]], vocab: Dict[str, int], seed: int) -> Dict[str, Any]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    x, y = encode_rows(train_rows, vocab)
    tx = torch.tensor(x, dtype=torch.long, device=DEVICE)
    ty = torch.tensor(y, dtype=torch.float32, device=DEVICE)
    model = BiGRU(len(vocab)).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    loader = DataLoader(TensorDataset(tx, ty), batch_size=BATCH_SIZE, shuffle=True)

    best_state = None
    best_acc = -1.0
    best_loss = float("inf")
    no_improve = 0
    for ep in range(1, EPOCHS + 1):
        model.train()
        for xb, yb in loader:
            logits = model(xb)
            loss = F.binary_cross_entropy_with_logits(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
        model.eval()
        with torch.no_grad():
            logits = model(tx)
            loss = F.binary_cross_entropy_with_logits(logits, ty).item()
            acc = accuracy_for_logits(logits, ty)
        if acc > best_acc + 1e-5 or (abs(acc - best_acc) < 1e-5 and loss < best_loss):
            best_acc, best_loss = acc, loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
        if best_acc >= 0.999:
            break
        if ep >= 8 and no_improve >= 4 and best_acc > 0.80:
            break

    if best_state is not None:
        model.load_state_dict({k: v.to(DEVICE) for k, v in best_state.items()})
    model.eval()
    out: Dict[str, Any] = {"train_acc": best_acc, "train_loss": best_loss, "epochs": ep}

    # Training-kind fit: anchor versus sparse rows. This matters for interpretation.
    kinds = defaultdict(list)
    for i, r in enumerate(train_rows):
        kinds[r.get("train_kind", "unknown")].append(i)
    with torch.no_grad():
        logits_all = model(tx)
        for kind, idxs in kinds.items():
            ii = torch.tensor(idxs, dtype=torch.long, device=DEVICE)
            out[f"train_{kind}_acc"] = accuracy_for_logits(logits_all.index_select(0, ii), ty.index_select(0, ii))
        for name, rows in eval_sets.items():
            if not rows:
                out[name] = float("nan")
                continue
            ex, ey = encode_rows(rows, vocab)
            ex_t = torch.tensor(ex, dtype=torch.long, device=DEVICE)
            ey_t = torch.tensor(ey, dtype=torch.float32, device=DEVICE)
            out[name] = accuracy_for_logits(model(ex_t), ey_t)
    return out


def select_events(events: List[dict], n: int, seed: int) -> List[dict]:
    rng = random.Random(seed)
    evs = events[:]
    rng.shuffle(evs)
    return evs[: min(n, len(evs))]


def make_sparse_probe_rows(events: List[dict], mode: str, k_per_template: int, arm: str, seed: int) -> List[dict]:
    if k_per_template <= 0:
        return []
    rng = random.Random(seed)
    rows = []
    for tid in PROBE_TEMPLATES:
        evs = events[:]
        rng.shuffle(evs)
        for ev in evs[: min(k_per_template, len(evs))]:
            for hd in ["AB", "BA"]:
                r = render_row(ev, tid, hd, mode, f"sparse_{arm}_{k_per_template}")
                if arm == "true":
                    pass
                elif arm == "shuffled":
                    # Anti-aligned coordinate: systematic label flip, preserving balance and exposure.
                    r["label"] = 1 - r["label"]
                elif arm == "exposure":
                    # Exposure-matched but uninformative labels, balanced within the two hypotheses.
                    # Use deterministic random bit for AB and its complement for BA at event/template.
                    bit = stable_int(f"EXPO|{seed}|{ev['event_id']}|T{tid}") % 2
                    r["label"] = bit if hd == "AB" else 1 - bit
                else:
                    raise ValueError(arm)
                r["train_kind"] = f"probe_{arm}"
                rows.append(r)
    return rows


def summarize(vals: List[float]) -> Dict[str, Any]:
    vals = [float(v) for v in vals if not (isinstance(v, float) and math.isnan(v))]
    if not vals:
        return {"mean": float("nan"), "std": float("nan"), "n": 0}
    return {"mean": round(float(np.mean(vals)), 4), "std": round(float(np.std(vals)), 4), "n": len(vals), "values": [round(v, 4) for v in vals]}


def main():
    t0 = time.time()
    if DEVICE == "cuda":
        torch.cuda.set_device(0)
    torch.set_num_threads(min(16, max(1, torch.get_num_threads())))
    train_fams, held_fams = load_families()
    train_events_all = extract_events(train_fams)
    held_events_all = extract_events(held_fams)
    train_events = select_events(train_events_all, TRAIN_EVENT_LIMIT, 257)
    eval_train_events = select_events(train_events_all, EVAL_TRAIN_EVENT_LIMIT, 1257)
    eval_held_events = select_events(held_events_all, EVAL_HELD_EVENT_LIMIT, 2257)

    print(json.dumps({
        "status": "IDENTITY_ORBIT_ANCHOR_TEST_START",
        "device": DEVICE,
        "train_events": len(train_events),
        "eval_train_events": len(eval_train_events),
        "eval_held_events": len(eval_held_events),
        "anchor_templates": ANCHOR_TEMPLATES,
        "probe_templates": PROBE_TEMPLATES,
        "held_templates": HELD_TEMPLATES,
    }), flush=True)

    # Precompute all possible rows per mode for vocabulary and eval.
    all_rows_by_mode: Dict[str, List[dict]] = defaultdict(list)
    datasets: Dict[str, Dict[str, List[dict]]] = {}
    for mode in ALIAS_MODES:
        base_anchor = make_rows(train_events, ANCHOR_TEMPLATES, mode, "train_anchor")
        for r in base_anchor:
            r["train_kind"] = "anchor"
        evals = {
            "anchor_trainfam": make_rows(eval_train_events, ANCHOR_TEMPLATES, mode, "eval_train_anchor"),
            "probe_trainfam": make_rows(eval_train_events, PROBE_TEMPLATES, mode, "eval_train_probe"),
            "anchor_heldfam": make_rows(eval_held_events, ANCHOR_TEMPLATES, mode, "eval_held_anchor"),
            "probe_heldfam": make_rows(eval_held_events, PROBE_TEMPLATES, mode, "eval_held_probe"),
            "heldtemplate_heldfam": make_rows(eval_held_events, HELD_TEMPLATES, mode, "eval_heldtemplate"),
        }
        datasets[mode] = {"base_anchor": base_anchor, **evals}
        for part in datasets[mode].values():
            all_rows_by_mode[mode].extend(part)
        # Include possible sparse rows in the vocabulary for every arm/k. This prevents
        # accidental <UNK> differences across arms.
        for k in K_PER_PROBE_TEMPLATE:
            for arm in ANCHOR_ARMS:
                if k == 0 and arm != "true":
                    continue
                all_rows_by_mode[mode].extend(make_sparse_probe_rows(train_events, mode, k, arm, 999))
    vocabs = build_vocab(all_rows_by_mode)

    metadata = {
        "train_event_limit": TRAIN_EVENT_LIMIT,
        "eval_train_event_limit": EVAL_TRAIN_EVENT_LIMIT,
        "eval_held_event_limit": EVAL_HELD_EVENT_LIMIT,
        "k_per_probe_template": K_PER_PROBE_TEMPLATE,
        "alias_modes": ALIAS_MODES,
        "anchor_arms": ANCHOR_ARMS,
        "n_seeds": N_SEEDS,
        "model": {"type": "BiGRU_attention", "emb_dim": EMB_DIM, "hidden_dim": HIDDEN_DIM, "epochs": EPOCHS, "batch_size": BATCH_SIZE, "lr": LR},
        "device": DEVICE,
        "templates": {str(k): {"text": v[0], "first": v[1], "group": v[2]} for k, v in TEMPLATES.items()},
        "vocab_sizes": {m: len(v) for m, v in vocabs.items()},
    }
    (OUT_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2))

    raw = []
    for mode in ALIAS_MODES:
        vocab = vocabs[mode]
        base = datasets[mode]["base_anchor"]
        eval_sets = {k: v for k, v in datasets[mode].items() if k != "base_anchor"}
        print(f"\n=== MODE {mode} | vocab={len(vocab)} | base_anchor_rows={len(base)} ===", flush=True)
        for k in K_PER_PROBE_TEMPLATE:
            arms = ["true"] if k == 0 else ANCHOR_ARMS
            for arm in arms:
                for seed_idx in range(N_SEEDS):
                    seed = 257000 + seed_idx * 1009 + k * 37 + stable_int(mode + arm) % 997
                    sparse = make_sparse_probe_rows(train_events, mode, k, arm, seed)
                    train_rows = base + sparse
                    res = train_eval(train_rows, eval_sets, vocab, seed)
                    res.update({
                        "mode": mode,
                        "arm": "zero" if k == 0 else arm,
                        "k_per_probe_template": k,
                        "seed_idx": seed_idx,
                        "seed": seed,
                        "n_train_rows": len(train_rows),
                        "n_sparse_rows": len(sparse),
                        "train_fit": bool(res["train_acc"] >= TRAIN_FIT),
                    })
                    raw.append(res)
                    print(json.dumps({
                        "mode": mode,
                        "arm": res["arm"],
                        "k": k,
                        "seed_idx": seed_idx,
                        "train": round(res["train_acc"], 4),
                        "anchor_held": round(res["anchor_heldfam"], 4),
                        "probe_train": round(res["probe_trainfam"], 4),
                        "probe_held": round(res["probe_heldfam"], 4),
                        "heldtmpl": round(res["heldtemplate_heldfam"], 4),
                        "fit": res["train_fit"],
                    }), flush=True)

    # Aggregate by mode/arm/k.
    grouped = defaultdict(list)
    for r in raw:
        grouped[(r["mode"], r["arm"], r["k_per_probe_template"])].append(r)
    summary = {}
    for key, items in grouped.items():
        mode, arm, k = key
        skey = f"{mode}|{arm}|k{k}"
        summary[skey] = {
            "mode": mode,
            "arm": arm,
            "k_per_probe_template": k,
            "n_runs": len(items),
            "fit_count": sum(1 for x in items if x["train_fit"]),
        }
        metrics = [
            "train_acc", "train_anchor_acc", "train_probe_true_acc", "train_probe_shuffled_acc", "train_probe_exposure_acc",
            "anchor_trainfam", "probe_trainfam", "anchor_heldfam", "probe_heldfam", "heldtemplate_heldfam",
        ]
        for m in metrics:
            vals = [x[m] for x in items if m in x]
            if vals:
                summary[skey][m] = summarize(vals)
        fit_items = [x for x in items if x["train_fit"]]
        if fit_items:
            for m in ["anchor_heldfam", "probe_heldfam", "probe_trainfam", "heldtemplate_heldfam"]:
                summary[skey][m + "__fit_only"] = summarize([x[m] for x in fit_items])

    out = {
        "status": "IDENTITY_ORBIT_ANCHOR_TEST",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_seconds": round(time.time() - t0, 2),
        "metadata": metadata,
        "summary": summary,
        "raw": raw,
    }
    out_json = OUT_DIR / "identity_orbit_anchor_summary.json"
    out_json.write_text(json.dumps(out, indent=2, default=str))

    # Human-readable compact report.
    lines = []
    lines.append("# research identity-orbit alias x sparse anchor test\n")
    lines.append(f"Device: `{DEVICE}`; elapsed {out['elapsed_seconds']} s. Minimal predicate texts remove score/date/tournament event-ID channels.\n")
    lines.append("\n## Mean accuracies by alias mode and sparse-anchor arm\n")
    lines.append("\n| mode | arm | k/template | fit | train | anchor-held | probe-held | probe-trainfam | held-template |\n")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|\n")
    order = []
    for mode in ALIAS_MODES:
        for k in K_PER_PROBE_TEMPLATE:
            arms = ["zero"] if k == 0 else ANCHOR_ARMS
            for arm in arms:
                order.append((mode, arm, k))
    for mode, arm, k in order:
        skey = f"{mode}|{arm}|k{k}"
        if skey not in summary:
            continue
        s = summary[skey]
        def mean(metric):
            return s.get(metric, {}).get("mean", float("nan"))
        lines.append(
            f"| {mode} | {arm} | {k} | {s['fit_count']}/{s['n_runs']} | "
            f"{mean('train_acc'):.3f} | {mean('anchor_heldfam'):.3f} | {mean('probe_heldfam'):.3f} | "
            f"{mean('probe_trainfam'):.3f} | {mean('heldtemplate_heldfam'):.3f} |\n"
        )
    lines.append("\nInterpretation should use only train-fit runs. `probe_heldfam` is the mixed seen-hypothesis/held-context surface with new source families; `probe_trainfam` shows same-family shortcut sensitivity.\n")
    lines.append(f"\nSummary JSON: `{out_json}`\n")
    ((OUT_DIR.parents[4] / 'research/documents/representation_and_objectives/data/identity_orbit_anchor_test/identity_orbit_anchor_summary.md')).write_text("".join(lines))

    print("\n" + "=" * 92)
    print("IDENTITY-ORBIT x SPARSE ANCHOR SUMMARY")
    print("=" * 92)
    print("mode                 arm        k  fit train  anchorH probeH probeTrain heldT")
    for mode, arm, k in order:
        skey = f"{mode}|{arm}|k{k}"
        if skey not in summary:
            continue
        s = summary[skey]
        def mean(metric):
            return s.get(metric, {}).get("mean", float("nan"))
        print(f"{mode:<20s} {arm:<9s} {k:>2d} {s['fit_count']}/{s['n_runs']} "
              f"{mean('train_acc'):.3f}  {mean('anchor_heldfam'):.3f}   {mean('probe_heldfam'):.3f}   {mean('probe_trainfam'):.3f}    {mean('heldtemplate_heldfam'):.3f}")
    print(json.dumps({"status": out["status"], "elapsed_seconds": out["elapsed_seconds"], "summary_json": str(out_json)}, indent=2))


if __name__ == "__main__":
    main()
