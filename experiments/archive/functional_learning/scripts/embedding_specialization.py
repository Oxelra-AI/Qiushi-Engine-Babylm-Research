#!/usr/bin/env python3
"""research: separate held-symbol binding loss from tied embedding drift.

research showed a sharp dissociation: full-objective continuation can recover
excellent trained-entity binding while held-entity binding, already strong at the
answer-only preparation endpoint, collapses.  The model ties input embeddings to
the output classifier, so full next-token gradients can move held entity rows as
softmax output rows even though those entity tokens never appear as training
inputs or correct targets.  This experiment asks whether preserving or separating
held input rows selectively preserves held-symbol binding while full-objective
learning proceeds.
"""

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative

import argparse, copy, json, math, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

SCRIPT_DIR = _public_path('experiments/archive/functional_learning/scripts')
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
import loss_allocation_binding as Base
import query_first_binding as S16
import binding_branching as S17
import budget_matched_full_objective as S18

K, N_ATTR, VOCAB = Base.K, Base.N_ATTR, Base.VOCAB
TRAIN_E, HELD_E = Base.TRAIN_E, Base.HELD_E
BOS, PAD, SEP, HAS, IS = Base.BOS, Base.PAD, Base.SEP, Base.HAS, Base.IS
ENT, ATTR, RWT, RWT_TOKS = Base.ENT, Base.ATTR, Base.RWT, Base.RWT_TOKS
SL, IS_POS = 18, 15
DEFAULT_PREP_EPOCHS = {42: 300, 43: 500, 100: 400}

HELD_ENT_ROWS = [ENT(e) for e in HELD_E]
TRAIN_ENT_ROWS = [ENT(e) for e in TRAIN_E]
ATTR_ROWS = [ATTR(a) for a in range(N_ATTR)]
RWT_ROWS = [RWT(a) for a in range(N_ATTR)]
SPECIAL_ROWS = [BOS, PAD, SEP, HAS, IS]

BRANCHES = [
    dict(name="tied_carry_full", model="tied", opt="carry", freeze_held=False),
    dict(name="tied_carry_freeze_held", model="tied", opt="carry", freeze_held=True),
    dict(name="tied_reset_full", model="tied", opt="reset", freeze_held=False),
    dict(name="untied_reset_full", model="untied", opt="reset", freeze_held=False),
    dict(name="untied_reset_freeze_held_input", model="untied", opt="reset", freeze_held=True),
]


class UntiedCLM(nn.Module):
    """Same architecture as Base.CLM, but with an untied output classifier."""
    def __init__(self, V, d, nh, nl, ml):
        super().__init__()
        self.tok = nn.Embedding(V, d)
        self.pos = nn.Embedding(ml, d)
        self.blks = nn.ModuleList([Base.Blk(d, nh) for _ in range(nl)])
        self.ln = nn.LayerNorm(d)
        self.out = nn.Linear(d, V, bias=False)

    def forward(self, x, m=None):
        B, L = x.shape
        h = self.tok(x) + self.pos(torch.arange(L, device=x.device))
        for b in self.blks:
            h = b(h, m)
        return self.out(self.ln(h))


def parse_prep_epochs(text):
    if not text:
        return dict(DEFAULT_PREP_EPOCHS)
    out = {}
    for part in text.split(','):
        if not part.strip():
            continue
        k, v = part.split(':')
        out[int(k)] = int(v)
    return out


def make_tied(cfg, dev):
    return Base.CLM(VOCAB, cfg["d"], cfg["nh"], cfg["nl"], SL).to(dev)


def make_untied_from_tied_state(tied_state, cfg, dev):
    model = UntiedCLM(VOCAB, cfg["d"], cfg["nh"], cfg["nl"], SL).to(dev)
    own = model.state_dict()
    for k in list(own.keys()):
        if k == "out.weight":
            own[k].copy_(tied_state["tok.weight"])
        elif k in tied_state:
            own[k].copy_(tied_state[k])
        else:
            raise KeyError(k)
    model.load_state_dict(own)
    return model


def category_stats(W, W0, rows):
    rows_t = torch.tensor(rows, device=W.device, dtype=torch.long)
    X = W[rows_t].detach()
    X0 = W0[rows_t].detach()
    diff = X - X0
    l2 = torch.linalg.norm(diff, dim=1)
    base = torch.linalg.norm(X0, dim=1).clamp_min(1e-12)
    rel = l2 / base
    cos = F.cosine_similarity(X, X0, dim=1)
    return dict(
        l2_mean=round(float(l2.mean().item()), 6),
        l2_max=round(float(l2.max().item()), 6),
        rel_mean=round(float(rel.mean().item()), 6),
        cos_mean=round(float(cos.mean().item()), 6),
    )


def row_drift(model, prep_tok_weight, prep_out_weight=None):
    groups = dict(held_ent=HELD_ENT_ROWS, train_ent=TRAIN_ENT_ROWS,
                  attr=ATTR_ROWS, rwt=RWT_ROWS, special=SPECIAL_ROWS)
    W = model.tok.weight.detach()
    out = {"input": {g: category_stats(W, prep_tok_weight, rows) for g, rows in groups.items()}}
    if hasattr(model, "out"):
        Wout = model.out.weight.detach()
        base = prep_out_weight if prep_out_weight is not None else prep_tok_weight
        out["output"] = {g: category_stats(Wout, base, rows) for g, rows in groups.items()}
    else:
        out["output"] = {g: dict(v) for g, v in out["input"].items()}
    return out


def restore_rows_(model, rows, values):
    # Advanced indexing returns a copy in PyTorch; index_copy_ performs an in-place
    # scatter into the embedding matrix and is required for the freeze/restore
    # intervention to be real.
    with torch.no_grad():
        model.tok.weight.data.index_copy_(0, rows, values)


def train_one_epoch(model, opt, seqs, order_idx, dev, mode, bs, mask,
                    restore_rows=None, restore_values=None):
    model.train()
    st = torch.tensor(seqs, dtype=torch.long)[torch.tensor(order_idx, dtype=torch.long)]
    L = SL - 1
    pos_w = S17.weight_vec(mode, 1.0, dev)
    tot = 0.0; denom = 0.0; ans_s = 0.0; ctx_s = 0.0; ctx_d = 0.0; cnt = 0
    for i in range(0, st.size(0), bs):
        b = st[i:i + bs].to(dev)
        inp, tgt = b[:, :L], b[:, 1:]
        logits = model(inp, mask)
        ce = F.cross_entropy(logits.reshape(-1, VOCAB), tgt.reshape(-1),
                             reduction="none").view(tgt.shape)
        nonpad = (tgt != PAD).float()
        wt = nonpad * pos_w.view(1, -1)
        d = wt.sum().clamp_min(1.0)
        loss = (ce * wt).sum() / d
        opt.zero_grad(); loss.backward(); opt.step()
        if restore_rows is not None:
            restore_rows_(model, restore_rows, restore_values)
        tot += float((ce * wt).sum().detach()); denom += float(d.detach())
        ans_s += float(ce[:, IS_POS].sum().detach()); cnt += int(ce.size(0))
        cm2 = nonpad.clone(); cm2[:, IS_POS] = 0.0
        ctx_s += float((ce * cm2).sum().detach()); ctx_d += float(cm2.sum().detach())
    return dict(wloss=round(tot / max(denom, 1), 6),
                ans_ce=round(ans_s / max(cnt, 1), 6),
                ctx_ce=round(ctx_s / max(ctx_d, 1), 6))


def eval_with_optional_restore(model, probes, dev, cm_blk, prep_tok_weight,
                               prep_out_weight=None, restore=False):
    if restore:
        rows = torch.tensor(HELD_ENT_ROWS, device=dev, dtype=torch.long)
        old = model.tok.weight.data[rows].detach().clone()
        restore_rows_(model, rows, prep_tok_weight[rows])
    try:
        qf = S18.eval_extended(model, probes, "query_first", dev)
        qfb = S18.eval_extended(S17.MaskOverride(model, cm_blk), probes, "query_first", dev)
        drift = row_drift(model, prep_tok_weight, prep_out_weight)
    finally:
        if restore:
            restore_rows_(model, rows, old)
    return dict(qf=qf, qf_blocked=qfb, drift=drift)


def held_summary(pack):
    qf = pack["qf"]
    return dict(
        train_top4=qf["std"]["ctx_top1"],
        train_b=qf["bswap"]["mean"],
        train_q=qf["qswap"]["margin"],
        train_sel=qf["corrupt"]["novel_selectivity"],
        held_top4=qf["held"]["ctx_top1"],
        held_b=qf["held_bswap"]["mean"],
        held_q=qf["held_qswap"]["margin"],
        held_sel=qf["held_corrupt"]["novel_selectivity"],
        blocked_top4=pack["qf_blocked"]["std"]["ctx_top1"],
        held_ent_input_l2=pack["drift"]["input"]["held_ent"]["l2_mean"],
        held_ent_input_cos=pack["drift"]["input"]["held_ent"]["cos_mean"],
    )


def strong_train(pack):
    qf = pack["qf"]
    return (qf["std"]["ctx_top1"] >= 0.95 and
            qf["bswap"]["mean"] >= 5.0 and
            qf["qswap"]["margin"] >= 5.0 and
            qf["corrupt"]["novel_selectivity"] >= 0.8)


def held_preserved(pack):
    qf = pack["qf"]
    return (qf["held"]["ctx_top1"] >= 0.75 and
            qf["held_bswap"]["mean"] >= 5.0 and
            qf["held_corrupt"]["novel_selectivity"] >= 0.5)


def should_eval(global_epoch, prep_epoch, total_epochs, eval_every):
    special = {prep_epoch, prep_epoch + 1, prep_epoch + 25, prep_epoch + 50,
               prep_epoch + 100, total_epochs}
    return global_epoch in special or (global_epoch % eval_every == 0)


def train_prep(seed, prep_epoch, n_train, cfg, dev, cm_std, probes, cm_blk):
    torch.manual_seed(seed); np.random.seed(seed)
    model = make_tied(cfg, dev)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    prep_records = []
    ts = time.time()
    for ep in range(1, prep_epoch + 1):
        rows = S16.make_epoch_rows(seed, ep, n_train)
        seqs = S16.rows_to_seqs(rows, "query_first", "bound")
        idx = S16.common_order(seed, ep, n_train)
        info = train_one_epoch(model, opt, seqs, idx, dev, "answer_only", cfg["bs"], cm_std)
        if ep == 1 or ep == prep_epoch or ep % max(25, prep_epoch // 4) == 0:
            prep_tok = model.tok.weight.detach().clone()
            pack = eval_with_optional_restore(model, probes, dev, cm_blk, prep_tok)
            rec = dict(seed=seed, epoch=ep, elapsed=round(time.time() - ts, 1), **info)
            rec.update(pack)
            prep_records.append(rec)
            hs = held_summary(pack)
            print(f"  prep e={ep:4d} train4={hs['train_top4']:.3f} held4={hs['held_top4']:.3f} "
                  f"b={hs['train_b']:+.3f} hb={hs['held_b']:+.3f} sel={hs['held_sel']:+.3f}", flush=True)
    return (copy.deepcopy(model.state_dict()), copy.deepcopy(opt.state_dict()),
            model.tok.weight.detach().clone(), prep_records)


def train_branch(seed, branch, prep_epoch, total_epochs, n_train, cfg, dev,
                 cm_std, cm_blk, probes, tied_state, opt_state, prep_tok_weight):
    if branch["model"] == "tied":
        model = make_tied(cfg, dev)
        model.load_state_dict(copy.deepcopy(tied_state))
        prep_out_weight = prep_tok_weight
    elif branch["model"] == "untied":
        model = make_untied_from_tied_state(tied_state, cfg, dev)
        prep_out_weight = prep_tok_weight
    else:
        raise ValueError(branch["model"])

    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    if branch["opt"] == "carry":
        opt.load_state_dict(copy.deepcopy(opt_state))
    elif branch["opt"] != "reset":
        raise ValueError(branch["opt"])

    restore_rows = torch.tensor(HELD_ENT_ROWS, device=dev, dtype=torch.long) if branch["freeze_held"] else None
    restore_vals = prep_tok_weight[restore_rows].detach().clone() if restore_rows is not None else None

    records = []
    ts = time.time()

    def add_record(ep, info):
        pack = eval_with_optional_restore(model, probes, dev, cm_blk, prep_tok_weight, prep_out_weight, restore=False)
        rec = dict(seed=seed, branch=branch["name"], model_type=branch["model"],
                   opt_mode=branch["opt"], freeze_held=branch["freeze_held"],
                   prep_epoch=prep_epoch, epoch=ep, branch_epoch=ep - prep_epoch,
                   elapsed=round(time.time() - ts, 1), **info)
        rec.update(pack)
        # For every branch this measures whether replacing only held input rows by
        # their preparation values would change behavior.  For frozen branches it
        # should be nearly identical and acts as a sanity check.
        rest = eval_with_optional_restore(model, probes, dev, cm_blk, prep_tok_weight, prep_out_weight, restore=True)
        rec["heldrow_restore_eval"] = rest
        records.append(rec)
        hs = held_summary(pack)
        hrs = held_summary(rest)
        print(f"    {branch['name']:<32s} e={ep:4d} be={ep-prep_epoch:4d} "
              f"train4={hs['train_top4']:.3f} held4={hs['held_top4']:.3f} "
              f"held4R={hrs['held_top4']:.3f} hb={hs['held_b']:+.3f} "
              f"sel={hs['held_sel']:+.3f} rowL2={hs['held_ent_input_l2']:.3f}", flush=True)

    add_record(prep_epoch, dict(wloss=0.0, ans_ce=0.0, ctx_ce=0.0))
    for ep in range(prep_epoch + 1, total_epochs + 1):
        rows = S16.make_epoch_rows(seed, ep, n_train)
        seqs = S16.rows_to_seqs(rows, "query_first", "bound")
        idx = S16.common_order(seed, ep, n_train)
        info = train_one_epoch(model, opt, seqs, idx, dev, "full", cfg["bs"], cm_std,
                               restore_rows=restore_rows, restore_values=restore_vals)
        if should_eval(ep, prep_epoch, total_epochs, cfg["eval_every"]):
            add_record(ep, info)
    return records, copy.deepcopy(model.state_dict())


def summarize(data):
    seeds = data["config"]["seeds"]
    branches = [b["name"] for b in data["branches_spec"]]
    S = {"by_seed": {}, "by_branch": {}}
    for sd in seeds:
        S["by_seed"][str(sd)] = {}
        for br in branches:
            rs = [r for r in data["branches"] if r["seed"] == sd and r["branch"] == br]
            if not rs:
                continue
            rs = sorted(rs, key=lambda r: r["epoch"])
            prep = rs[0]
            final = rs[-1]
            S["by_seed"][str(sd)][br] = dict(
                prep=held_summary(prep),
                final=held_summary(final),
                final_restore=held_summary(final["heldrow_restore_eval"]),
                train_strong_final=strong_train(final),
                held_preserved_final=held_preserved(final),
                held_restore_delta=round(held_summary(final["heldrow_restore_eval"])["held_top4"] - held_summary(final)["held_top4"], 6),
            )
    for br in branches:
        finals = []
        rests = []
        for sd in seeds:
            b = S["by_seed"].get(str(sd), {}).get(br)
            if b:
                finals.append(b["final"]); rests.append(b["final_restore"])
        def mean(key, arr=finals):
            return None if not arr else round(float(np.mean([x[key] for x in arr])), 6)
        S["by_branch"][br] = dict(
            n=len(finals),
            final_train_strong=sum(1 for sd in seeds if S["by_seed"].get(str(sd), {}).get(br, {}).get("train_strong_final")),
            final_held_preserved=sum(1 for sd in seeds if S["by_seed"].get(str(sd), {}).get(br, {}).get("held_preserved_final")),
            final_mean_train_top4=mean("train_top4"),
            final_mean_held_top4=mean("held_top4"),
            final_mean_held_b=mean("held_b"),
            final_mean_held_sel=mean("held_sel"),
            final_mean_blocked_top4=mean("blocked_top4"),
            final_mean_held_row_l2=mean("held_ent_input_l2"),
            restore_mean_held_top4=mean("held_top4", rests),
            restore_mean_held_b=mean("held_b", rests),
            restore_delta_top4=None if not finals else round(float(np.mean([rests[i]["held_top4"] - finals[i]["held_top4"] for i in range(len(finals))])), 6),
        )
    return S


def make_figure(data, figpath):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    branches = [b["name"] for b in data["branches_spec"]]
    colours = {
        "tied_carry_full": "tab:red",
        "tied_carry_freeze_held": "tab:blue",
        "tied_reset_full": "tab:pink",
        "untied_reset_full": "tab:orange",
        "untied_reset_freeze_held_input": "tab:green",
    }
    plots = [
        (("qf", "std", "ctx_top1"), "Trained top4", 0.25, 0.95),
        (("qf", "held", "ctx_top1"), "Held-query top4", 0.25, 0.75),
        (("qf", "held_bswap", "mean"), "Held B-swap", 0.0, 5.0),
        (("qf", "held_corrupt", "novel_selectivity"), "Held corruption selectivity", 0.0, 0.5),
        (("qf_blocked", "std", "ctx_top1"), "Blocked q→ctx trained top4", 0.25, 0.95),
        (("drift", "input", "held_ent", "l2_mean"), "Held input row drift", 0.0, None),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(18, 9))
    for ax, (path, title, base_line, upper_line) in zip(axes.flat, plots):
        for br in branches:
            xs_all = sorted(set(r["branch_epoch"] for r in data["branches"] if r["branch"] == br))
            xs, ys = [], []
            for be in xs_all:
                vals = []
                for sd in data["config"]["seeds"]:
                    rr = [r for r in data["branches"] if r["branch"] == br and r["seed"] == sd and r["branch_epoch"] == be]
                    if not rr:
                        continue
                    obj = rr[0]
                    for k in path:
                        obj = obj[k]
                    vals.append(float(obj))
                if vals:
                    xs.append(be); ys.append(float(np.mean(vals)))
            ax.plot(xs, ys, label=br.replace("_", " "), color=colours.get(br), lw=1.6)
        if base_line is not None:
            ax.axhline(base_line, color="black", ls="--", alpha=0.28)
        if upper_line is not None:
            ax.axhline(upper_line, color="black", ls=":", alpha=0.25)
        ax.set_title(title, fontsize=10); ax.set_xlabel("epochs after switch"); ax.grid(alpha=0.25)
        ax.legend(fontsize=6)
    fig.suptitle("research: held-symbol binding versus tied embedding drift", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    Path(figpath).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figpath, dpi=160, bbox_inches="tight")
    plt.close(fig)


def write_note(data, path):
    S = data["summary"]
    lines = []
    lines.append("# research result: held-symbol binding and tied embedding drift")
    lines.append("")
    lines.append("This note is generated from `data/embedding_specialization/results.json`. It tests whether full-objective continuation loses held-symbol binding because tied output/input rows for unseen entity tokens drift, or because the contextual selector/readout itself specializes to familiar entities.")
    lines.append("")
    lines.append("## Branch-level final summary")
    lines.append("")
    lines.append("| branch | trained strong | held preserved | final train top4 | final held top4 | restored held top4 | restore Δtop4 | held B | held sel | blocked top4 | held row L2 |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for br, sm in S["by_branch"].items():
        lines.append(f"| {br} | {sm['final_train_strong']}/{sm['n']} | {sm['final_held_preserved']}/{sm['n']} | {sm['final_mean_train_top4']:.3f} | {sm['final_mean_held_top4']:.3f} | {sm['restore_mean_held_top4']:.3f} | {sm['restore_delta_top4']:+.3f} | {sm['final_mean_held_b']:+.3f} | {sm['final_mean_held_sel']:+.3f} | {sm['final_mean_blocked_top4']:.3f} | {sm['final_mean_held_row_l2']:.3f} |")
    lines.append("")
    lines.append("## Per-seed final summary")
    lines.append("")
    lines.append("| seed | branch | prep held4 | final train4 | final held4 | restored held4 | held B | held sel | held row L2 | blocked train4 |")
    lines.append("|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for sd, byb in S["by_seed"].items():
        for br, sm in byb.items():
            p, f, r = sm["prep"], sm["final"], sm["final_restore"]
            lines.append(f"| {sd} | {br} | {p['held_top4']:.3f} | {f['train_top4']:.3f} | {f['held_top4']:.3f} | {r['held_top4']:.3f} | {f['held_b']:+.3f} | {f['held_sel']:+.3f} | {f['held_ent_input_l2']:.3f} | {f['blocked_top4']:.3f} |")
    lines.append("")
    lines.append("## Reading guide")
    lines.append("")
    lines.append("A selective row-drift explanation is supported only if held-row restoration or held-row freezing preserves held-query binding while trained-entity binding remains strong. If freezing or untieing rows does not preserve held behavior, then the loss is more likely in the contextual network or RWT readout specialization. Blocked q→context evaluation tracks whether the prospective marker route remains behaviorally required after continuation.")
    Path(path).write_text("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--data", default=None)
    ap.add_argument("--fig", default=None)
    ap.add_argument("--seeds", default="42,43,100")
    ap.add_argument("--prep-epochs", default="")
    ap.add_argument("--total-epochs", type=int, default=1000)
    ap.add_argument("--eval-every", type=int, default=25)
    ap.add_argument("--n-train", type=int, default=500)
    ap.add_argument("--save-checkpoints", action="store_true")
    A = ap.parse_args()

    ws = _public_path('experiments/archive/functional_learning')
    if A.data is None:
        A.data = str(ws / "data" / "embedding_specialization")
    if A.fig is None:
        A.fig = str(ws / "figures" / "embedding_specialization.png")

    if A.smoke:
        seeds = [42]
        prep_epochs = {42: 8}
        total_epochs = 24
        eval_every = 4
        n_train = 64
        branches = BRANCHES[:3]
        n_std, n_small = 96, 48
    else:
        seeds = [int(x) for x in A.seeds.split(',') if x.strip()]
        p_in = parse_prep_epochs(A.prep_epochs)
        prep_epochs = {sd: int(p_in.get(sd, DEFAULT_PREP_EPOCHS.get(sd, 400))) for sd in seeds}
        total_epochs = A.total_epochs
        eval_every = A.eval_every
        n_train = A.n_train
        branches = BRANCHES
        n_std, n_small = 512, 256

    cfg = dict(d=64, nh=2, nl=3, lr=3e-4, wd=0.01, bs=64,
               seeds=seeds, prep_epochs={str(k): int(v) for k, v in prep_epochs.items()},
               total_epochs=total_epochs, eval_every=eval_every, n_train=n_train,
               held_ent_rows=HELD_ENT_ROWS)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cm_std = S17.standard_causal_mask(SL - 1, dev)
    cm_blk = S17.block_query_ctx_mask(SL - 1, dev)
    probes = S18.make_probes(seed=190019, n_std=n_std, n_small=n_small)
    out_dir = Path(A.data); out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = out_dir / "checkpoints"
    if A.save_checkpoints:
        ckpt_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device: {dev}")
    print(f"Seeds={seeds}; prep={prep_epochs}; total_epochs={total_epochs}; n_train={n_train}; branches={[b['name'] for b in branches]}", flush=True)
    t0 = time.time()
    data = dict(config=cfg, branches_spec=copy.deepcopy(branches), prep=[], branches=[])

    for sd in seeds:
        P = prep_epochs[sd]
        print("\n" + "=" * 80)
        print(f"Seed {sd}: query-first answer-only preparation to P={P}")
        print("=" * 80, flush=True)
        tied_state, opt_state, prep_tok, prep_records = train_prep(sd, P, n_train, cfg, dev, cm_std, probes, cm_blk)
        data["prep"].extend(prep_records)
        if A.save_checkpoints:
            torch.save(tied_state, ckpt_dir / f"seed{sd}_prep_tied.pt")
        for br in branches:
            print(f"\n  Branch {br['name']}", flush=True)
            recs, final_state = train_branch(sd, br, P, total_epochs, n_train, cfg, dev,
                                             cm_std, cm_blk, probes, tied_state, opt_state, prep_tok)
            data["branches"].extend(recs)
            if A.save_checkpoints:
                torch.save(final_state, ckpt_dir / f"seed{sd}_{br['name']}_final.pt")

    data["summary"] = summarize(data)
    outp = out_dir / "results.json"
    with open(outp, "w") as f:
        json.dump(data, f, indent=2)
    make_figure(data, A.fig)
    note = ws / "notes" / "embedding_specialization_result.md"
    write_note(data, note)

    print("\nSUMMARY")
    for br, sm in data["summary"]["by_branch"].items():
        print(f"  {br:<34s} train={sm['final_train_strong']}/{sm['n']} held={sm['final_held_preserved']}/{sm['n']} "
              f"held4={sm['final_mean_held_top4']:.3f} restore4={sm['restore_mean_held_top4']:.3f} "
              f"rowL2={sm['final_mean_held_row_l2']:.3f}")
    print(json.dumps(dict(status="DONE", out=str(outp), figure=A.fig,
                          note=str(note), elapsed=round(time.time() - t0, 1))))


if __name__ == "__main__":
    main()
