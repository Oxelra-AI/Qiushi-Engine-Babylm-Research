#!/usr/bin/env python3
"""research: Readout-calibration and compatible-trajectory test.

Tests the competing explanation for held-symbol binding loss:
  - Is it an inherent conflict between context prediction and the binding marker?
  - Or is it how the full objective is introduced (gradient shock)?

Context: After query-first answer-only preparation, the model has excellent
binding (trained and held).  First-epoch context CE is ~26-31 nats (essentially
random) vs answer CE ~0.07-0.14.  When switching to full objective, the body
receives massive context-position gradients.  research showed body-only full
collapses held binding within 1-25 epochs while body-only answer-only preserves it.
But context CE starts at 31 nats; the mismatch itself may cause the shock.

Arms (all untied, from Step019b preparation checkpoints):
  1. direct_full:  standard switch to full objective (baseline collapse)
  2. calibrate_out50_then_full:  readout-only calibration (out.weight) for 50 epochs
     under full objective, then unfreeze all for 450 epochs
  3. gradual_ctx100_then_full:  all params, context weight ramped 0.01 -> 1.0
     linearly over 100 epochs, then normal full for 400 epochs
  4. slow_lr25_then_full:  all params, lr=3e-5 for 25 epochs, then lr=3e-4
  5. interleaved_ans_full:  alternate answer-only and full every epoch (500 total)
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
import revision_019b_embedding_role_decomposition as S19b

VOCAB = Base.VOCAB
PAD   = Base.PAD
SL    = 18
IS_POS = 15
DEFAULT_PREP_EPOCHS = {42: 300, 43: 500, 100: 400}

# ── arm definitions ──────────────────────────────────────────────────
ARMS = [
    dict(name="direct_full", phases=[
        dict(epochs=500, train="all", lr=3e-4, ctx_mode="fixed", ctx_weight=1.0),
    ]),
    dict(name="calibrate_out50_then_full", phases=[
        dict(epochs=50, train="out_only", lr=3e-4, ctx_mode="fixed", ctx_weight=1.0),
        dict(epochs=450, train="all",     lr=3e-4, ctx_mode="fixed", ctx_weight=1.0),
    ]),
    dict(name="gradual_ctx100_then_full", phases=[
        dict(epochs=100, train="all", lr=3e-4, ctx_mode="ramp",
             ctx_start=0.01, ctx_end=1.0),
        dict(epochs=400, train="all", lr=3e-4, ctx_mode="fixed", ctx_weight=1.0),
    ]),
    dict(name="slow_lr25_then_full", phases=[
        dict(epochs=25,  train="all", lr=3e-5, ctx_mode="fixed", ctx_weight=1.0),
        dict(epochs=475, train="all", lr=3e-4, ctx_mode="fixed", ctx_weight=1.0),
    ]),
    dict(name="interleaved_ans_full", phases=[
        dict(epochs=500, train="all", lr=3e-4, ctx_mode="interleaved"),
    ]),
    dict(name="constant_ctx_half", phases=[
        dict(epochs=500, train="all", lr=3e-4, ctx_mode="fixed", ctx_weight=0.5),
    ]),
    dict(name="constant_ctx_tenth", phases=[
        dict(epochs=500, train="all", lr=3e-4, ctx_mode="fixed", ctx_weight=0.1),
    ]),
    # Static objective with the same nominal A:C coefficient ratio as alternating
    # answer-only epochs (A) with full epochs ((A+15C)/16):
    # 0.5*A + 0.5*(A+15C)/16 = (17A+15C)/32.
    # The normalized fixed-weight objective (A+15*w*C)/(1+15*w)
    # matches this when w = 1/17.
    dict(name="constant_ctx_1over17", phases=[
        dict(epochs=500, train="all", lr=3e-4, ctx_mode="fixed", ctx_weight=1.0/17.0),
    ]),
]
ARM_INDEX = {a["name"]: i for i, a in enumerate(ARMS)}


# ── helpers ──────────────────────────────────────────────────────────
def ctx_weight_for_epoch(phase, ep_in_phase):
    """Compute context-position loss weight for this epoch within the phase."""
    mode = phase.get("ctx_mode", "fixed")
    if mode == "fixed":
        return phase.get("ctx_weight", 1.0)
    elif mode == "ramp":
        t = ep_in_phase / max(phase["epochs"], 1)
        return phase["ctx_start"] + (phase["ctx_end"] - phase["ctx_start"]) * t
    elif mode == "interleaved":
        return 0.0 if ep_in_phase % 2 == 1 else 1.0
    raise ValueError(mode)


def set_trainable(model, spec):
    """Freeze everything then selectively unfreeze; return trainable params."""
    for p in model.parameters():
        p.requires_grad_(False)
    if spec == "all":
        for p in model.parameters():
            p.requires_grad_(True)
    elif spec == "out_only":
        model.out.weight.requires_grad_(True)
    elif spec == "out_ln":
        model.out.weight.requires_grad_(True)
        for p in model.ln.parameters():
            p.requires_grad_(True)
    else:
        raise ValueError(spec)
    return [p for p in model.parameters() if p.requires_grad]


def train_one_epoch_weighted(model, opt, seqs, order_idx, dev, bs, mask,
                             answer_weight=1.0, context_weight=1.0):
    """Train one epoch with explicit per-position weights."""
    model.train()
    L = SL - 1
    # build position-weight vector
    pw = torch.zeros(L, dtype=torch.float32, device=dev)
    pw[:] = context_weight
    pw[IS_POS] = answer_weight
    pw[-1] = 0.0   # terminal PAD
    st = torch.tensor(seqs, dtype=torch.long)[torch.tensor(order_idx, dtype=torch.long)]
    tot = 0.0; den = 0.0; ans_s = 0.0; ctx_s = 0.0; ctx_d = 0.0; cnt = 0
    for i in range(0, st.size(0), bs):
        b = st[i:i + bs].to(dev)
        inp, tgt = b[:, :L], b[:, 1:]
        logits = model(inp, mask)
        ce = F.cross_entropy(logits.reshape(-1, VOCAB), tgt.reshape(-1),
                             reduction="none").view(tgt.shape)
        nonpad = (tgt != PAD).float()
        wt = nonpad * pw.unsqueeze(0)
        d = wt.sum().clamp_min(1.0)
        loss = (ce * wt).sum() / d
        opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            tot += float((ce * wt).sum()); den += float(d)
            ans_s += float(ce[:, IS_POS].sum()); cnt += int(ce.size(0))
            cm2 = nonpad.clone(); cm2[:, IS_POS] = 0.0
            ctx_s += float((ce * cm2).sum()); ctx_d += float(cm2.sum())
    return dict(wloss=round(tot / max(den, 1), 6),
                ans_ce=round(ans_s / max(cnt, 1), 6),
                ctx_ce=round(ctx_s / max(ctx_d, 1), 6))


def eval_schedule(total):
    s = {0, 1, 5, 10, 25, 50, 75, 100, 125, 150, 200, 250, 300, 350, 400, 450, 500}
    return sorted(e for e in s if 0 <= e <= total)


def phase_label(ph):
    mode = ph.get("ctx_mode", "fixed")
    return f"{ph['train']}_{mode}"


# ── per-arm runner ───────────────────────────────────────────────────
def run_arm(seed, P, arm, tied_state, prep_tok, cfg,
            probes, dev, cm_std, cm_blk, n_train, total_cont):
    idx_offset = ARM_INDEX.get(arm["name"], 0)
    torch.manual_seed(seed + 210000 + 31 * idx_offset)
    np.random.seed(seed + 210000 + idx_offset)

    model = S19b.make_untied_from_tied_state(tied_state, cfg, dev)
    prep_out = model.out.weight.data.detach().clone()

    sched = set(eval_schedule(total_cont))
    recs = []; ts = time.time()

    def record(be, info, plabel=""):
        pack = S19b.eval_pack(model, probes, dev, cm_blk, prep_tok, prep_out)
        sm = S19b.metric_summary(pack)
        sm = S19b.add_drift_to_summary(sm, pack)
        rec = dict(seed=int(seed), arm=arm["name"], prep_epoch=int(P),
                   branch_epoch=int(be), epoch=int(P + be), phase=plabel,
                   elapsed=round(time.time() - ts, 2))
        rec.update(info)
        rec.update(pack)
        recs.append(rec)
        print(f"  {arm['name']:<38s} be={be:3d} [{plabel:<14s}] "
              f"tr4={sm['train_top4']:.3f} h4={sm['held_top4']:.3f} "
              f"hB={sm['held_b']:+.3f} hSel={sm['held_sel']:+.3f} "
              f"blk4={sm['blocked_train_top4']:.3f} "
              f"ans={info.get('ans_ce',0):.3f} ctx={info.get('ctx_ce',0):.3f} "
              f"cw={info.get('ctx_weight','?')}", flush=True)

    # be=0: evaluate preparation checkpoint
    record(0, dict(wloss=0.0, ans_ce=0.0, ctx_ce=0.0, ctx_weight=0.0), "prep")

    be = 0
    for ph in arm["phases"]:
        params = set_trainable(model, ph["train"])
        opt = torch.optim.AdamW(params, lr=ph["lr"], weight_decay=cfg["wd"])
        plabel = phase_label(ph)
        for ep_in_ph in range(1, ph["epochs"] + 1):
            be += 1
            ep = P + be
            cw = ctx_weight_for_epoch(ph, ep_in_ph)
            rows = S16.make_epoch_rows(seed, ep, n_train)
            seqs = S16.rows_to_seqs(rows, "query_first", "bound")
            idx  = S16.common_order(seed, ep, n_train)
            info = train_one_epoch_weighted(model, opt, seqs, idx, dev,
                                           cfg["bs"], cm_std,
                                           answer_weight=1.0,
                                           context_weight=cw)
            info["ctx_weight"] = round(cw, 4)
            if be in sched:
                record(be, info, plabel)
    return recs


# ── summarize ────────────────────────────────────────────────────────
def arm_summary(all_recs, arm_name, seeds):
    """Mean metrics across seeds for start and final."""
    recs = [r for r in all_recs if r["arm"] == arm_name]
    if not recs:
        return {}
    max_be = max(int(r["branch_epoch"]) for r in recs)

    def mean_k(subset, key):
        vals = []
        for r in subset:
            sm = S19b.metric_summary(r)
            sm = S19b.add_drift_to_summary(sm, r)
            vals.append(sm.get(key, 0))
        return round(float(np.mean(vals)), 6) if vals else None

    starts = [r for r in recs if int(r["branch_epoch"]) == 0]
    finals = [r for r in recs if int(r["branch_epoch"]) == max_be]
    out = dict(n=len(finals), max_be=max_be)
    for tag, arr in [("start", starts), ("final", finals)]:
        for k in ["train_top4", "held_top4", "held_b", "held_sel",
                   "blocked_train_top4"]:
            out[f"{tag}_{k}"] = mean_k(arr, k)
    out["final_ans_ce"] = round(float(np.mean([r.get("ans_ce", 0) for r in finals])), 4)
    out["final_ctx_ce"] = round(float(np.mean([r.get("ctx_ce", 0) for r in finals])), 4)
    return out


# ── note ─────────────────────────────────────────────────────────────
def write_note(data, path):
    def f(v):
        return "" if v is None else f"{v:.3f}"
    lines = [
        "# research calibration and compatible-trajectory test", "",
        "Tests whether held-symbol binding loss during full-objective continuation",
        "is inherent (body context prediction conflicts with the binding marker)",
        "or transitional (gradient shock from uncalibrated context positions).", "",
        "## Arm means at final epoch", "",
        "| arm | n | start h4 | final h4 | final hB | final hSel | "
        "final train4 | final blk4 | final ctx_ce |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, sm in data["summary"].items():
        lines.append(
            f"| {name} | {sm['n']} | {f(sm['start_held_top4'])} | "
            f"{f(sm['final_held_top4'])} | {f(sm['final_held_b'])} | "
            f"{f(sm['final_held_sel'])} | {f(sm['final_train_top4'])} | "
            f"{f(sm['final_blocked_train_top4'])} | {f(sm['final_ctx_ce'])} |")
    lines += ["", "## Per-seed trajectories", ""]
    seeds = [int(s) if isinstance(s, str) else s for s in data["config"]["seeds"]]
    for sd in seeds:
        lines.append(f"### Seed {sd}")
        lines.append("")
        for aspec in data["arms_spec"]:
            name = aspec["name"]
            arm_recs = sorted([r for r in data["records"]
                               if int(r["seed"]) == sd and r["arm"] == name],
                              key=lambda r: int(r["branch_epoch"]))
            if not arm_recs:
                continue
            lines.append(f"**{name}**")
            for r in arm_recs:
                sm = S19b.metric_summary(r)
                sm = S19b.add_drift_to_summary(sm, r)
                lines.append(
                    f"  be={int(r['branch_epoch']):3d} [{r.get('phase',''):<14s}] "
                    f"tr4={sm['train_top4']:.3f} h4={sm['held_top4']:.3f} "
                    f"hB={sm['held_b']:+.3f} hSel={sm['held_sel']:+.3f} "
                    f"ctx_ce={r.get('ctx_ce',0):.3f} cw={r.get('ctx_weight','?')}")
            lines.append("")
    lines += [
        "## Interpretation", "",
        "If any compatible trajectory (gradual ramp, slow lr, interleaved "
        "reinforcement) preserves held binding while context CE drops to normal "
        "levels (~1.5 nats), the conflict is transitional and a compatible learning "
        "path exists.  If all fail, context prediction through the body inherently "
        "conflicts with the binding computation.", "",
        "Readout calibration may fail to reduce context CE enough (research showed "
        "out_only barely reduced it from ~34 to ~28 in 25 epochs), so its failure "
        "alone does not establish inherent conflict.  The gradual ramp is the "
        "strongest test because it limits gradient magnitude at every epoch.", ""
    ]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines) + "\n")


# ── figure ───────────────────────────────────────────────────────────
def make_figure(data, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    arm_names = [a["name"] for a in data["arms_spec"]]
    cmap = plt.cm.tab10(np.linspace(0, 1, max(len(arm_names), 1)))
    arm_colors = {n: cmap[i] for i, n in enumerate(arm_names)}
    seeds = [int(s) if isinstance(s, str) else s for s in data["config"]["seeds"]]

    plots = [
        ("held_top4",           "Held top-4",         0.15, 1.05),
        ("held_b",              "Held B-swap",        -2.0, None),
        ("held_sel",            "Held selectivity",   -0.3, 1.05),
        ("train_top4",          "Train top-4",         0.0, 1.05),
        ("blocked_train_top4",  "Blocked train top-4", 0.2, 0.4),
        ("_ctx_ce",             "Context CE",          0.0, None),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    for ax, (mk, title, ylo, yhi) in zip(axes.flat, plots):
        for aname, color in arm_colors.items():
            for si, sd in enumerate(seeds):
                rs = sorted([r for r in data["records"]
                             if int(r["seed"]) == sd and r["arm"] == aname],
                            key=lambda r: int(r["branch_epoch"]))
                if not rs:
                    continue
                xs = [int(r["branch_epoch"]) for r in rs]
                if mk == "_ctx_ce":
                    ys = [float(r.get("ctx_ce", 0)) for r in rs]
                else:
                    ys = []
                    for r in rs:
                        sm = S19b.metric_summary(r)
                        sm = S19b.add_drift_to_summary(sm, r)
                        ys.append(float(sm[mk]))
                label = aname if si == 0 else None
                ax.plot(xs, ys, color=color, alpha=0.5, lw=1.1, label=label)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Branch epoch", fontsize=8)
        if ylo is not None:
            ax.set_ylim(bottom=ylo)
        if yhi is not None:
            ax.set_ylim(top=yhi)
        if mk == "held_top4":
            ax.legend(fontsize=6, loc="lower right", ncol=1)
    fig.suptitle("research: Calibration & Compatible-Trajectory Test", fontsize=13)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Figure saved: {path}")


# ── JSON encoder ─────────────────────────────────────────────────────
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


# ── main ─────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint-dir",
        default="experiments/archive/functional_learning/data"
                "revision_019b_embedding_role_decomposition/checkpoints")
    ap.add_argument("--data",
        default="experiments/archive/functional_learning/data/calibration_trajectory")
    ap.add_argument("--note",
        default="research/notes/functional_learning/calibration_trajectory.md")
    ap.add_argument("--figure",
        default="experiments/archive/functional_learning/figures/calibration_trajectory.png")
    ap.add_argument("--seeds", default="42,43,100")
    ap.add_argument("--arms", default="all")
    ap.add_argument("--total-epochs", type=int, default=500)
    ap.add_argument("--n-train", type=int, default=500)
    ap.add_argument("--probe-seed", type=int, default=180018)
    ap.add_argument("--smoke", action="store_true")
    A = ap.parse_args()

    if A.smoke:
        A.seeds = "100"
        A.arms = "direct_full,gradual_ctx100_then_full"
        A.total_epochs = 10
        A.n_train = 64
        n_std, n_small = 96, 48
    else:
        n_std, n_small = 512, 256

    seeds = [int(x) for x in A.seeds.split(",") if x.strip()]

    # select arms
    if A.arms == "all":
        arms = [copy.deepcopy(a) for a in ARMS]
    else:
        want = {x.strip() for x in A.arms.split(",") if x.strip()}
        arms = [copy.deepcopy(a) for a in ARMS if a["name"] in want]

    # scale phase epochs to requested total
    for arm in arms:
        orig_total = sum(p["epochs"] for p in arm["phases"])
        if orig_total != A.total_epochs:
            scale = A.total_epochs / orig_total
            for p in arm["phases"]:
                p["epochs"] = max(1, int(round(p["epochs"] * scale)))

    cfg = dict(d=64, nh=2, nl=3, lr=3e-4, wd=0.01, bs=64)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cm_std = S17.standard_causal_mask(SL - 1, dev)
    cm_blk = S17.block_query_ctx_mask(SL - 1, dev)
    probes = S18.make_probes(seed=A.probe_seed, n_std=n_std, n_small=n_small)
    out_dir = Path(A.data); out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = Path(A.checkpoint_dir)

    data = dict(
        config=dict(seeds=seeds,
                    prep_epochs={str(k): DEFAULT_PREP_EPOCHS[k] for k in seeds},
                    total_cont_epochs=A.total_epochs, n_train=A.n_train,
                    probe_seed=A.probe_seed),
        arms_spec=[dict(name=a["name"], phases=a["phases"]) for a in arms],
        records=[], summary={})

    print(f"Device: {dev}; seeds={seeds}; total_epochs={A.total_epochs}; "
          f"arms={[a['name'] for a in arms]}", flush=True)
    t0 = time.time()

    for sd in seeds:
        P = DEFAULT_PREP_EPOCHS[sd]
        ck = ckpt_dir / f"seed{sd}_prep_tied.pt"
        if not ck.exists():
            print(f"SKIP seed {sd}: {ck} not found"); continue
        tied_state = torch.load(ck, map_location=dev)
        prep_tok = tied_state["tok.weight"].detach().clone().to(dev)
        print("\n" + "=" * 80)
        print(f"Seed {sd}; prep P={P}")
        print("=" * 80, flush=True)
        for arm in arms:
            total_cont = sum(p["epochs"] for p in arm["phases"])
            recs = run_arm(sd, P, arm, tied_state, prep_tok, cfg, probes, dev,
                           cm_std, cm_blk, A.n_train, total_cont)
            data["records"].extend(recs)

    for arm in arms:
        data["summary"][arm["name"]] = arm_summary(data["records"], arm["name"], seeds)

    outp = out_dir / "results.json"
    outp.write_text(json.dumps(data, indent=2, cls=NpEncoder))
    write_note(data, A.note)
    try:
        make_figure(data, A.figure)
    except Exception as e:
        print(f"Figure error: {e}")
    print(json.dumps(dict(status="ok", out=str(outp), note=A.note,
                          figure=A.figure, elapsed=round(time.time() - t0, 1)),
                     indent=2))


if __name__ == "__main__":
    main()
