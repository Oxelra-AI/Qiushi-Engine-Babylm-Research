#!/usr/bin/env python3
"""research: Branch from bound query-first checkpoints.

Scientific questions:
  (1) Retention: does the binding computation persist when the objective changes?
  (2) Query access: does it depend on active query-conditioned context states?
  (3) Transfer: does a bound initialization make later original-order experience
      more effective than fresh or bag-level initializations?

Design:
  Phase 1: Acquire binding via query-first answer-only training (per seed).
  Phase 2: Branch from bound checkpoint into 5 arms (500 additional epochs):
    - continue_qfirst_ans_only  (retention baseline)
    - switch_qfirst_full        (retention under full objective)
    - switch_qfirst_ctx_only    (retention without answer pressure)
    - qfirst_block_qctx_ans     (query-access dependency)
    - transfer_orig_ans_only    (bound init -> original order)
  Phase 3: Control (matched total compute for transfer comparison):
    - Original-order answer-only from scratch for (acq_epoch + branch_epochs) total.
"""

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative

import argparse, copy, json, math, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

SCRIPT_DIR = _public_path('experiments/archive/functional_learning/scripts')
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
import loss_allocation_binding as Base
import query_first_binding as S16

# ─── constants ───
K, N_ATTR, VOCAB = Base.K, Base.N_ATTR, Base.VOCAB
TRAIN_E, HELD_E = Base.TRAIN_E, Base.HELD_E
BOS, PAD, SEP, HAS, IS = Base.BOS, Base.PAD, Base.SEP, Base.HAS, Base.IS
ENT, ATTR, RWT, RWT_TOKS = Base.ENT, Base.ATTR, Base.RWT, Base.RWT_TOKS
SL = 18; IS_POS = 15


# ─── attention masks ───
def standard_causal_mask(L, dev):
    return torch.triu(torch.ones(L, L, dtype=torch.bool, device=dev), diagonal=1)


def block_query_ctx_mask(L, dev):
    """Causal mask + block context positions (3..14) from attending to query
    region (positions 1=entity, 2=SEP). Blocks both direct and SEP-mediated
    indirect query access. IS (pos 15) and answer input (pos 16) keep full access."""
    m = standard_causal_mask(L, dev)
    for i in range(3, 15):          # context triple positions
        for j in [1, 2]:            # query entity + SEP
            m[i, j] = True
    return m


class MaskOverride:
    """Wrapper so S16 evaluation functions use a specified attention mask."""
    def __init__(self, model, mask):
        self._m = model; self._mask = mask
    def __call__(self, x, m=None):
        return self._m(x, self._mask)
    def eval(self):
        self._m.eval(); return self
    def train(self, mode=True):
        self._m.train(mode); return self
    def parameters(self):
        return self._m.parameters()


# ─── training ───
def weight_vec(mode, answer_weight, dev):
    L = SL - 1
    w = torch.zeros(L, dtype=torch.float32, device=dev)
    if mode == "full":
        w[:] = 1.0; w[-1] = 0.0
    elif mode == "answer_only":
        w[IS_POS] = 1.0
    elif mode == "context_only":
        w[:] = 1.0; w[IS_POS] = 0.0; w[-1] = 0.0
    else:
        raise ValueError(mode)
    return w


def train_one_epoch(model, opt, seqs, order_idx, dev, mode, bs, mask):
    model.train()
    st = torch.tensor(seqs, dtype=torch.long)[torch.tensor(order_idx, dtype=torch.long)]
    L = SL - 1
    pos_w = weight_vec(mode, 1.0, dev)
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
        tot += float((ce * wt).sum().detach()); denom += float(d.detach())
        ans_s += float(ce[:, IS_POS].sum().detach()); cnt += int(ce.size(0))
        cm2 = nonpad.clone(); cm2[:, IS_POS] = 0.0
        ctx_s += float((ce * cm2).sum().detach()); ctx_d += float(cm2.sum().detach())
    return dict(wloss=round(tot / max(denom, 1), 6),
                ans_ce=round(ans_s / max(cnt, 1), 6),
                ctx_ce=round(ctx_s / max(ctx_d, 1), 6))


# ─── evaluation ───
def eval_all(model, probes, dev, blocked_mask):
    """Evaluate in query-first (standard), original (standard), and qf-blocked."""
    model.eval()
    qf = S16.evaluate(model, probes, "query_first", dev)
    orig = S16.evaluate(model, probes, "original", dev)
    bm = MaskOverride(model, blocked_mask)
    qfb = S16.evaluate(bm, probes, "query_first", dev)
    return dict(qf=qf, orig=orig, qf_blocked=qfb)


def bound_threshold(qf_met):
    """Binding threshold for acquisition phase."""
    return (qf_met["std"]["ctx_top1"] >= 0.95 and
            qf_met["bswap"]["frac_pos"] >= 0.95 and
            qf_met["qswap"]["frac_pos"] >= 0.95 and
            qf_met["corrupt"]["novel_selectivity"] >= 0.8)


# ─── branch specifications ───
BRANCH_SPECS = [
    dict(name="continue_qfirst_ans_only", order="query_first",
         mode="answer_only", mask_type="standard"),
    dict(name="switch_qfirst_full", order="query_first",
         mode="full", mask_type="standard"),
    dict(name="switch_qfirst_ctx_only", order="query_first",
         mode="context_only", mask_type="standard"),
    dict(name="qfirst_block_qctx_ans", order="query_first",
         mode="answer_only", mask_type="block_query_ctx"),
    dict(name="transfer_orig_ans_only", order="original",
         mode="answer_only", mask_type="standard"),
]


# ─── figure ───
def make_figure(data, figpath):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(18, 9))

    # Colours per branch
    colours = {
        "continue_qfirst_ans_only": "tab:blue",
        "switch_qfirst_full": "tab:green",
        "switch_qfirst_ctx_only": "tab:purple",
        "qfirst_block_qctx_ans": "tab:red",
        "transfer_orig_ans_only": "tab:orange",
        "control_orig_ans_only": "tab:gray",
    }

    # ── panel helpers ──
    def branch_curve(records, branch_name, key_path, seeds):
        """Return (branch_epochs, seed-mean values) for a branch."""
        brs = [r for r in records if r["branch"] == branch_name]
        if not brs:
            return [], []
        eps_all = sorted(set(r["branch_epoch"] for r in brs))
        xs, ys = [], []
        for ep in eps_all:
            vals = []
            for sd in seeds:
                rr = [r for r in brs if r["seed"] == sd and r["branch_epoch"] == ep]
                if rr:
                    obj = rr[0]
                    for k in key_path:
                        obj = obj[k]
                    vals.append(float(obj))
            if vals:
                xs.append(ep); ys.append(float(np.mean(vals)))
        return xs, ys

    def ctrl_curve(records, key_path, seeds, offset=0):
        eps_all = sorted(set(r["epoch"] for r in records))
        xs, ys = [], []
        for ep in eps_all:
            vals = []
            for sd in seeds:
                rr = [r for r in records if r["seed"] == sd and r["epoch"] == ep]
                if rr:
                    obj = rr[0]
                    for k in key_path:
                        obj = obj[k]
                    vals.append(float(obj))
            if vals:
                xs.append(ep - offset); ys.append(float(np.mean(vals)))
        return xs, ys

    seeds = data["config"]["seeds"]

    # Panel 0: query-first correct NLL (retention)
    ax = axes[0, 0]
    for bn in ["continue_qfirst_ans_only", "switch_qfirst_full",
                "switch_qfirst_ctx_only", "qfirst_block_qctx_ans"]:
        x, y = branch_curve(data["branches"], bn, ["qf", "std", "correct_nll"], seeds)
        ax.plot(x, y, label=bn.replace("_", " "), color=colours[bn], lw=1.5)
    ax.axhline(math.log(4), color="black", ls="--", alpha=0.3, label="bag uniform")
    ax.set_title("QF correct NLL (retention)", fontsize=10)
    ax.set_xlabel("Branch epoch"); ax.set_ylabel("NLL"); ax.legend(fontsize=6); ax.grid(alpha=0.2)

    # Panel 1: query-first top4 (retention)
    ax = axes[0, 1]
    for bn in ["continue_qfirst_ans_only", "switch_qfirst_full",
                "switch_qfirst_ctx_only", "qfirst_block_qctx_ans"]:
        x, y = branch_curve(data["branches"], bn, ["qf", "std", "ctx_top1"], seeds)
        ax.plot(x, y, label=bn.replace("_", " "), color=colours[bn], lw=1.5)
    ax.axhline(0.25, color="black", ls="--", alpha=0.3, label="chance")
    ax.set_title("QF tie-safe top4 (retention)", fontsize=10)
    ax.set_xlabel("Branch epoch"); ax.set_ylabel("Top4"); ax.legend(fontsize=6); ax.grid(alpha=0.2)

    # Panel 2: query-first blocked top4 (query-access test)
    ax = axes[0, 2]
    for bn in ["continue_qfirst_ans_only", "qfirst_block_qctx_ans"]:
        x, y = branch_curve(data["branches"], bn, ["qf_blocked", "std", "ctx_top1"], seeds)
        ax.plot(x, y, label=f"{bn} (blocked eval)", color=colours[bn], lw=1.5, ls="--")
        x2, y2 = branch_curve(data["branches"], bn, ["qf", "std", "ctx_top1"], seeds)
        ax.plot(x2, y2, label=f"{bn} (std eval)", color=colours[bn], lw=1.5)
    ax.axhline(0.25, color="black", ls="--", alpha=0.3)
    ax.set_title("Query-access dependency", fontsize=10)
    ax.set_xlabel("Branch epoch"); ax.set_ylabel("Top4"); ax.legend(fontsize=6); ax.grid(alpha=0.2)

    # Panel 3: original-order top4 (transfer vs control)
    ax = axes[1, 0]
    x, y = branch_curve(data["branches"], "transfer_orig_ans_only",
                         ["orig", "std", "ctx_top1"], seeds)
    ax.plot(x, y, label="transfer (bound init)", color=colours["transfer_orig_ans_only"], lw=1.8)
    # Control: use epoch offset so branch_epoch 0 = acq_epoch
    if data.get("control"):
        acq_eps = sorted(set(r.get("acq_epoch_ref", 0) for r in data["control"]))
        offset = int(np.mean(acq_eps)) if acq_eps else 0
        x3, y3 = ctrl_curve(data["control"], ["orig", "std", "ctx_top1"], seeds, offset=offset)
        ax.plot(x3, y3, label="control (fresh init)", color=colours["control_orig_ans_only"], lw=1.5)
    ax.axhline(0.25, color="black", ls="--", alpha=0.3, label="chance")
    ax.set_title("Original-order top4 (transfer)", fontsize=10)
    ax.set_xlabel("Branch epoch (or epoch-offset)"); ax.set_ylabel("Top4")
    ax.legend(fontsize=6); ax.grid(alpha=0.2)

    # Panel 4: original-order B-swap (transfer vs control)
    ax = axes[1, 1]
    x, y = branch_curve(data["branches"], "transfer_orig_ans_only",
                         ["orig", "bswap", "mean"], seeds)
    ax.plot(x, y, label="transfer (bound init)", color=colours["transfer_orig_ans_only"], lw=1.8)
    if data.get("control"):
        x3, y3 = ctrl_curve(data["control"], ["orig", "bswap", "mean"], seeds, offset=offset)
        ax.plot(x3, y3, label="control (fresh init)", color=colours["control_orig_ans_only"], lw=1.5)
    ax.axhline(0, color="black", ls="--", alpha=0.3)
    ax.set_title("Original-order B-swap (transfer)", fontsize=10)
    ax.set_xlabel("Branch epoch"); ax.set_ylabel("B-swap margin")
    ax.legend(fontsize=6); ax.grid(alpha=0.2)

    # Panel 5: query-novel selectivity for key arms
    ax = axes[1, 2]
    for bn in ["continue_qfirst_ans_only", "switch_qfirst_full",
                "transfer_orig_ans_only"]:
        kp = ["qf", "corrupt", "novel_selectivity"] if "qfirst" in bn else \
             ["orig", "corrupt", "novel_selectivity"]
        x, y = branch_curve(data["branches"], bn, kp, seeds)
        ax.plot(x, y, label=bn.replace("_", " "), color=colours[bn], lw=1.5)
    ax.axhline(0, color="black", ls="--", alpha=0.3)
    ax.set_title("Query-novel selectivity", fontsize=10)
    ax.set_xlabel("Branch epoch"); ax.set_ylabel("Selectivity")
    ax.legend(fontsize=6); ax.grid(alpha=0.2)

    fig.suptitle("research: Branching from bound query-first checkpoints", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    Path(figpath).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figpath, dpi=160, bbox_inches="tight")
    plt.close(fig)


# ─── summary ───
def print_summary(data):
    seeds = data["config"]["seeds"]
    print("\n" + "=" * 72)
    print("ACQUISITION")
    for sd in seeds:
        recs = [r for r in data["acquisition"] if r["seed"] == sd]
        if recs:
            last = recs[-1]
            qf = last["qf"]["std"]
            print(f"  seed {sd}: acq_epoch={last['epoch']} cnll={qf['correct_nll']:.3f} "
                  f"top4={qf['ctx_top1']:.3f} "
                  f"bs={last['qf']['bswap']['mean']:+.3f} "
                  f"sel={last['qf']['corrupt']['novel_selectivity']:+.3f}")

    print("\nBRANCH FINALS (branch_epoch = branch_epochs or last)")
    for bn in [s["name"] for s in BRANCH_SPECS]:
        brs = [r for r in data["branches"] if r["branch"] == bn]
        if not brs:
            continue
        max_be = max(r["branch_epoch"] for r in brs)
        finals = [r for r in brs if r["branch_epoch"] == max_be]
        if not finals:
            continue
        # Primary format
        fmt = "qf" if "qfirst" in bn else "orig"
        cnll = np.mean([r[fmt]["std"]["correct_nll"] for r in finals])
        top4 = np.mean([r[fmt]["std"]["ctx_top1"] for r in finals])
        bs = np.mean([r[fmt]["bswap"]["mean"] for r in finals])
        qs = np.mean([r[fmt]["qswap"]["margin"] for r in finals])
        sel = np.mean([r[fmt]["corrupt"]["novel_selectivity"] for r in finals])
        print(f"  {bn:35s} ({fmt}) cnll={cnll:.3f} top4={top4:.3f} "
              f"bs={bs:+.3f} qs={qs:+.3f} sel={sel:+.3f}")

    if data.get("control"):
        print("\nCONTROL FINAL")
        max_ep = max(r["epoch"] for r in data["control"])
        finals = [r for r in data["control"] if r["epoch"] == max_ep]
        if finals:
            cnll = np.mean([r["orig"]["std"]["correct_nll"] for r in finals])
            top4 = np.mean([r["orig"]["std"]["ctx_top1"] for r in finals])
            bs = np.mean([r["orig"]["bswap"]["mean"] for r in finals])
            sel = np.mean([r["orig"]["corrupt"]["novel_selectivity"] for r in finals])
            print(f"  control_orig_ans_only (orig, ep={max_ep}) "
                  f"cnll={cnll:.3f} top4={top4:.3f} bs={bs:+.3f} sel={sel:+.3f}")

    print("=" * 72)


# ─── main ───
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--data", default=None)
    ap.add_argument("--fig", default=None)
    ap.add_argument("--seeds", default="42,43,100")
    A = ap.parse_args()

    ws = _public_path('experiments/archive/functional_learning')
    if A.data is None:
        A.data = str(ws / "data" / "branching")
    if A.fig is None:
        A.fig = str(ws / "figures" / "branching.png")

    seeds = [42] if A.smoke else [int(x) for x in A.seeds.split(",")]
    acq_max    = 30  if A.smoke else 600
    branch_ep  = 10  if A.smoke else 500
    eval_every = 5   if A.smoke else 25
    n_train    = 50  if A.smoke else 500
    n_probe    = 64  if A.smoke else 512
    n_probe_sm = 32  if A.smoke else 256
    cfg = dict(d=64, nh=2, nl=3, lr=3e-4, wd=0.01, bs=64,
               seeds=seeds, acq_max=acq_max, branch_epochs=branch_ep,
               eval_every=eval_every, n_train=n_train)

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {dev}")

    # Probe banks for both orders (seeded independently of training)
    erng = np.random.default_rng(170017)
    probes = {}
    for order in ["original", "query_first"]:
        probes[order] = dict(
            std=S16.gen_std(erng, n_probe, TRAIN_E, order),
            held=S16.gen_held(erng, n_probe_sm, order),
            bswap=S16.gen_bswap(erng, n_probe, TRAIN_E, order),
            qswap=S16.gen_qswap(erng, n_probe_sm, TRAIN_E, order),
            corrupt=S16.gen_corrupt(erng, n_probe_sm, TRAIN_E, order),
        )

    L = SL - 1
    cm_std = standard_causal_mask(L, dev)
    cm_blk = block_query_ctx_mask(L, dev)
    masks = {"standard": cm_std, "block_query_ctx": cm_blk}

    data = dict(config=cfg, acquisition=[], branches=[], control=[])
    t0 = time.time()
    Path(A.data).mkdir(parents=True, exist_ok=True)

    for seed in seeds:
        print(f"\n{'='*72}\nSeed {seed}\n{'='*72}")

        # ────── PHASE 1: Acquire binding ──────
        print(f"  PHASE 1: Acquire binding (qf ans-only, max {acq_max} ep)")
        torch.manual_seed(seed); np.random.seed(seed)
        model = Base.CLM(VOCAB, cfg["d"], cfg["nh"], cfg["nl"], SL).to(dev)
        opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])

        acq_records = []; ckpts = {}
        ts = time.time()

        for ep in range(1, acq_max + 1):
            rows = S16.make_epoch_rows(seed, ep, n_train)
            seqs = S16.rows_to_seqs(rows, "query_first", "bound")
            idx  = S16.common_order(seed, ep, n_train)
            info = train_one_epoch(model, opt, seqs, idx, dev, "answer_only",
                                   cfg["bs"], cm_std)
            if ep == 1 or ep % eval_every == 0 or ep == acq_max:
                met = eval_all(model, probes, dev, cm_blk)
                rec = dict(seed=seed, epoch=ep, **info); rec.update(met)
                acq_records.append(rec)
                ckpts[ep] = (copy.deepcopy(model.state_dict()),
                             copy.deepcopy(opt.state_dict()))
                qf = met["qf"]["std"]
                print(f"    e={ep:3d} ans={info['ans_ce']:.3f} "
                      f"cnll={qf['correct_nll']:.3f} t4={qf['ctx_top1']:.3f} "
                      f"bs={met['qf']['bswap']['mean']:+.4f} "
                      f"qf={met['qf']['qswap']['frac_pos']:.3f} "
                      f"sel={met['qf']['corrupt']['novel_selectivity']:+.3f} "
                      f"({time.time()-ts:.0f}s)")

        data["acquisition"].extend(acq_records)

        # Find acquisition epoch (first meeting threshold)
        acq_epoch = acq_max
        for rec in acq_records:
            if bound_threshold(rec["qf"]):
                acq_epoch = rec["epoch"]
                break
        # Pick nearest saved checkpoint (at or before acq_epoch)
        valid_eps = [e for e in sorted(ckpts.keys()) if e <= acq_epoch]
        ckpt_ep = valid_eps[-1] if valid_eps else max(ckpts.keys())
        bound_sd, bound_opt = ckpts[ckpt_ep]
        print(f"  ✓ Bound checkpoint at epoch {ckpt_ep}"
              f" (threshold {'met' if ckpt_ep == acq_epoch else 'NOT met'})")

        # ────── PHASE 2: Branch from bound checkpoint ──────
        print(f"  PHASE 2: Branch ({branch_ep} ep each)")

        for spec in BRANCH_SPECS:
            bn = spec["name"]
            print(f"    {bn} (order={spec['order']}, mode={spec['mode']}, "
                  f"mask={spec['mask_type']})")

            # Reload bound checkpoint
            torch.manual_seed(seed + 9999); np.random.seed(seed + 9999)
            model = Base.CLM(VOCAB, cfg["d"], cfg["nh"], cfg["nl"], SL).to(dev)
            model.load_state_dict(copy.deepcopy(bound_sd))
            opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"],
                                    weight_decay=cfg["wd"])
            opt.load_state_dict(copy.deepcopy(bound_opt))

            train_mask = masks[spec["mask_type"]]

            # Epoch-0 evaluation (before any branch training)
            met0 = eval_all(model, probes, dev, cm_blk)
            rec0 = dict(seed=seed, branch=bn, branch_epoch=0,
                        global_epoch=ckpt_ep, acq_epoch=ckpt_ep,
                        ans_ce=0, ctx_ce=0, wloss=0)
            rec0.update(met0)
            data["branches"].append(rec0)

            ts = time.time()
            for bep in range(1, branch_ep + 1):
                gep = ckpt_ep + bep
                rows = S16.make_epoch_rows(seed, gep, n_train)
                seqs = S16.rows_to_seqs(rows, spec["order"], "bound")
                idx  = S16.common_order(seed, gep, n_train)
                info = train_one_epoch(model, opt, seqs, idx, dev,
                                       spec["mode"], cfg["bs"], train_mask)
                if bep == 1 or bep % eval_every == 0 or bep == branch_ep:
                    met = eval_all(model, probes, dev, cm_blk)
                    rec = dict(seed=seed, branch=bn, branch_epoch=bep,
                               global_epoch=gep, acq_epoch=ckpt_ep)
                    rec.update(info); rec.update(met)
                    data["branches"].append(rec)

                    pm = met["qf"] if spec["order"] == "query_first" else met["orig"]
                    print(f"      bep={bep:3d} cnll={pm['std']['correct_nll']:.3f} "
                          f"t4={pm['std']['ctx_top1']:.3f} "
                          f"bs={pm['bswap']['mean']:+.4f} "
                          f"sel={pm['corrupt']['novel_selectivity']:+.3f} "
                          f"({time.time()-ts:.0f}s)")

        # ────── PHASE 3: Control (matched total compute) ──────
        total_epochs = ckpt_ep + branch_ep
        print(f"  PHASE 3: Control (orig ans-only, {total_epochs} total ep)")

        torch.manual_seed(seed); np.random.seed(seed)
        model = Base.CLM(VOCAB, cfg["d"], cfg["nh"], cfg["nl"], SL).to(dev)
        opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"],
                                weight_decay=cfg["wd"])
        ts = time.time()
        for ep in range(1, total_epochs + 1):
            rows = S16.make_epoch_rows(seed, ep, n_train)
            seqs = S16.rows_to_seqs(rows, "original", "bound")
            idx  = S16.common_order(seed, ep, n_train)
            info = train_one_epoch(model, opt, seqs, idx, dev, "answer_only",
                                   cfg["bs"], cm_std)
            if ep == 1 or ep % eval_every == 0 or ep == total_epochs:
                met = eval_all(model, probes, dev, cm_blk)
                rec = dict(seed=seed, control="control_orig_ans_only",
                           epoch=ep, total_epochs=total_epochs,
                           acq_epoch_ref=ckpt_ep)
                rec.update(info); rec.update(met)
                data["control"].append(rec)

                orig = met["orig"]["std"]
                print(f"      e={ep:3d} cnll={orig['correct_nll']:.3f} "
                      f"t4={orig['ctx_top1']:.3f} "
                      f"bs={met['orig']['bswap']['mean']:+.4f} "
                      f"sel={met['orig']['corrupt']['novel_selectivity']:+.3f} "
                      f"({time.time()-ts:.0f}s)")

    # ─── save ───
    outp = Path(A.data) / "results.json"
    with open(outp, "w") as f:
        json.dump(data, f, indent=2)

    print_summary(data)
    make_figure(data, A.fig)

    el = round(time.time() - t0, 1)
    print(json.dumps(dict(status="DONE", out=str(outp),
                          figure=A.fig, elapsed=el)))


if __name__ == "__main__":
    main()
