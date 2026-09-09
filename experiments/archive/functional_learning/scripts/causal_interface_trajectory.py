#!/usr/bin/env python3
"""research: causal interface trajectory for transfer retention.

This follows the research donor-query causal handle through continuation rather than
using familiar-symbol redirection as the endpoint.  It asks whether held-symbol
transfer loss arises from

  (i) weaker query-conditioned signal formation,
  (ii) changed downstream sensitivity to a supplied signal, or
  (iii) narrowing of the symbols/contexts that can generate a usable signal.

For each continuation checkpoint we measure behavior, L1 direction separability,
self donor-query redirection, and cross-time transplantation between the
preparation model and the continuation model, separately for train-only queries,
held-donor queries, and train-donor queries in held-containing contexts.
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

import query_first_binding as S16
import binding_branching as S17
import budget_matched_full_objective as S18
import revision_019b_embedding_role_decomposition as S19b
import causal_intervention as S25

SL = S25.SL
IS_POS = S25.IS_POS
VOCAB = S25.VOCAB
PAD = S25.PAD
RWT = S25.RWT
ATTR_POS = S25.ATTR_POS
K = S25.K
N_ATTR = S25.N_ATTR
TRAIN_E = S25.TRAIN_E
HELD_E = S25.HELD_E
DEFAULT_PREP = S25.DEFAULT_PREP

ARMS = {
    "direct_full": dict(kind="static", ctx_weight=1.0, lr=3e-4),
    "static_1over17": dict(kind="static", ctx_weight=1.0/17.0, lr=3e-4),
    "interleaved_ans_full": dict(kind="interleaved", ctx_weight=None, lr=3e-4),
}


class NpEnc(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating, float)):
            return round(float(o), 6)
        if isinstance(o, torch.Tensor):
            return o.detach().cpu().tolist()
        if isinstance(o, np.ndarray):
            return o.tolist()
        return super().default(o)


def r6(x):
    return round(float(x), 6)


def ctx_weight_for(arm_cfg, be):
    if arm_cfg["kind"] == "interleaved":
        # Same convention as research: odd epochs are answer-only, even epochs full.
        return 0.0 if be % 2 == 1 else 1.0
    return float(arm_cfg["ctx_weight"])


# ───────────────────────── fixed loss evaluation ─────────────────────────
def held_eval_rows(seed, n):
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n):
        he = int(rng.choice(HELD_E))
        others = rng.choice(TRAIN_E, K-1, replace=False).tolist()
        ce = list(rng.permutation(others + [he]))
        ca = rng.choice(N_ATTR, K, replace=False).tolist()
        qi = ce.index(he)
        bag_ti = int(rng.integers(K))
        rows.append((ce, ca, qi, bag_ti))
    return rows


@torch.no_grad()
def eval_ce(model, seqs, dev, mask, bs=128):
    model.eval()
    ans_s = 0.0; ans_n = 0
    ctx_s = 0.0; ctx_n = 0.0
    st = torch.tensor(seqs, dtype=torch.long)
    L = SL - 1
    for i in range(0, len(seqs), bs):
        b = st[i:i+bs].to(dev)
        inp, tgt = b[:, :L], b[:, 1:]
        logits = model(inp, mask)
        ce = F.cross_entropy(logits.reshape(-1, VOCAB), tgt.reshape(-1), reduction="none").view(tgt.shape)
        nonpad = (tgt != PAD).float()
        ans_s += float(ce[:, IS_POS].sum()); ans_n += int(b.size(0))
        cm = nonpad.clone(); cm[:, IS_POS] = 0.0
        ctx_s += float((ce * cm).sum()); ctx_n += float(cm.sum())
    return dict(ans_ce=r6(ans_s / max(ans_n, 1)), ctx_ce=r6(ctx_s / max(ctx_n, 1.0)))


# ───────────────────────── direction generation/eval ──────────────────────
def gen_dir_bank(seed, n, kind):
    rng = np.random.default_rng(seed)
    seqs, labels = [], []
    for _ in range(n):
        if kind == "train":
            ce = rng.choice(TRAIN_E, K, replace=False).tolist()
            ca = rng.choice(N_ATTR, K, replace=False).tolist()
            qi = int(rng.integers(K))
        elif kind == "held_query":
            he = int(rng.choice(HELD_E))
            others = rng.choice(TRAIN_E, K-1, replace=False).tolist()
            ce = list(rng.permutation(others + [he]))
            ca = rng.choice(N_ATTR, K, replace=False).tolist()
            qi = ce.index(he)
        elif kind == "train_in_held_ctx":
            he = int(rng.choice(HELD_E))
            others = rng.choice(TRAIN_E, K-1, replace=False).tolist()
            ce = list(rng.permutation(others + [he]))
            ca = rng.choice(N_ATTR, K, replace=False).tolist()
            hi = ce.index(he)
            choices = [i for i in range(K) if i != hi]
            qi = int(rng.choice(choices))
        else:
            raise ValueError(kind)
        seqs.append(S16.mk_seq("query_first", ce, ca, qi, PAD))
        labels.append(qi)
    return seqs, torch.tensor(labels, dtype=torch.long)


@torch.no_grad()
def extract_l1_feats(model, seqs, dev, mask, bs=128):
    feats = []
    for i in range(0, len(seqs), bs):
        inp = torch.tensor(seqs[i:i+bs], dtype=torch.long, device=dev)[:, :SL-1]
        _, stored = S25.forward_extract(model, inp, mask, [1])
        feats.append(stored[1].cpu())
    return torch.cat(feats, 0)


def learn_l1_direction(model, dev, mask, seed, n_train=800, n_test=300):
    tr_seqs, tr_lab = gen_dir_bank(seed, n_train, "train")
    te_seqs, te_lab = gen_dir_bank(seed + 1, n_test, "train")
    tr = extract_l1_feats(model, tr_seqs, dev, mask)
    te = extract_l1_feats(model, te_seqs, dev, mask)
    matched = torch.stack([tr[i, int(tr_lab[i])] for i in range(tr.size(0))])
    unmatched = torch.stack([tr[i, j] for i in range(tr.size(0)) for j in range(K) if j != int(tr_lab[i])])
    d = matched.mean(0) - unmatched.mean(0)
    d = d / d.norm().clamp_min(1e-8)
    def acc_margin(feats, lab):
        sc = (feats * d.view(1, 1, -1)).sum(-1)
        acc = float((sc.argmax(1) == lab).float().mean())
        true = sc[torch.arange(sc.size(0)), lab]
        masked = sc.clone(); masked[torch.arange(sc.size(0)), lab] = -1e9
        margin = float((true - masked.max(1).values).mean())
        return dict(acc=r6(acc), margin=r6(margin))
    return d, dict(train=acc_margin(tr, tr_lab), test=acc_margin(te, te_lab))


def direction_bank_metrics(model, direction, banks, dev, mask):
    out = {}
    for name, (seqs, lab) in banks.items():
        feats = extract_l1_feats(model, seqs, dev, mask)
        sc = (feats * direction.view(1, 1, -1)).sum(-1)
        pred = sc.argmax(1).cpu()
        true = sc[torch.arange(sc.size(0)), lab]
        masked = sc.clone(); masked[torch.arange(sc.size(0)), lab] = -1e9
        margin = true - masked.max(1).values
        out[name] = dict(acc=r6((pred == lab).float().mean()), margin=r6(margin.mean()))
    return out


# ───────────────────────── donor extraction and patching ──────────────────
@torch.no_grad()
def extract_donor_l1(model, pairs, dev, mask, bs=128):
    donor = []
    logits_a = []
    for i in range(0, len(pairs), bs):
        batch = pairs[i:i+bs]
        inp_a = torch.tensor([p["seq_a"] for p in batch], dtype=torch.long, device=dev)[:, :SL-1]
        logits, stored = S25.forward_extract(model, inp_a, mask, [1])
        donor.append(stored[1].cpu())
        logits_a.append(logits[:, IS_POS].cpu())
    logits_a = torch.cat(logits_a, 0)
    donor = torch.cat(donor, 0)
    ca = sum(1 for j, p in enumerate(pairs) if int(logits_a[j].argmax()) == RWT(p["target_a"]))
    return donor, r6(ca / max(len(pairs), 1))


@torch.no_grad()
def clean_b_acc(model, pairs, dev, mask, bs=128):
    cb = 0
    for i in range(0, len(pairs), bs):
        batch = pairs[i:i+bs]
        inp_b = torch.tensor([p["seq_b"] for p in batch], dtype=torch.long, device=dev)[:, :SL-1]
        logits = model(inp_b, mask)[:, IS_POS].cpu()
        for j, p in enumerate(batch):
            if int(logits[j].argmax()) == RWT(p["target_b"]):
                cb += 1
    return r6(cb / max(len(pairs), 1))


def patch_eval(recipient, pairs, donor_acts, dev, mask, direction=None, mode="full", bs=128):
    redir = 0; nat = 0; rnll = []; nnll = []
    patch_mode = "full" if mode == "full" else ("orth_only" if mode.startswith("orth") else "d_only")
    for i in range(0, len(pairs), bs):
        batch = pairs[i:i+bs]; bsz = len(batch)
        inp_b = torch.tensor([p["seq_b"] for p in batch], dtype=torch.long, device=dev)[:, :SL-1]
        donor = donor_acts[i:i+bsz].to(dev)
        logits = S25.forward_patch(recipient, inp_b, mask, 1, donor, patch_mode, direction)
        for j, p in enumerate(batch):
            ta = RWT(p["target_a"]); tb = RWT(p["target_b"])
            lp = F.log_softmax(logits[j, IS_POS], dim=-1)
            pred = int(logits[j, IS_POS].argmax())
            redir += int(pred == ta); nat += int(pred == tb)
            rnll.append(float(-lp[ta])); nnll.append(float(-lp[tb]))
    N = max(len(pairs), 1)
    return dict(redirect=r6(redir / N), natural=r6(nat / N),
                margin=r6(float(np.mean(nnll) - np.mean(rnll))),
                redirect_nll=r6(float(np.mean(rnll))), natural_nll=r6(float(np.mean(nnll))))


def eval_case_bundle(current, prep, pairs, prep_d, own_d, prep_donor_pack, current_donor_pack, dev, mask):
    # Donor packs are (acts, clean_a_acc).  Clean B depends on recipient.
    prep_donor, prep_clean_a = prep_donor_pack
    curr_donor, curr_clean_a = current_donor_pack
    out = {
        "clean": {
            "prep_a": prep_clean_a,
            "current_a": curr_clean_a,
            "prep_b": clean_b_acc(prep, pairs, dev, mask),
            "current_b": clean_b_acc(current, pairs, dev, mask),
        },
        "self_current": {},
        "prep_to_current": {},
        "current_to_prep": {},
    }
    # self: what the current model forms and uses.
    out["self_current"]["full"] = patch_eval(current, pairs, curr_donor, dev, mask, mode="full")
    out["self_current"]["d_prep"] = patch_eval(current, pairs, curr_donor, dev, mask, direction=prep_d, mode="d_only")
    out["self_current"]["orth_prep"] = patch_eval(current, pairs, curr_donor, dev, mask, direction=prep_d, mode="orth_only")
    out["self_current"]["d_own"] = patch_eval(current, pairs, curr_donor, dev, mask, direction=own_d, mode="d_only")
    # prep -> current: current downstream sensitivity to a well-formed preparation signal.
    out["prep_to_current"]["full"] = patch_eval(current, pairs, prep_donor, dev, mask, mode="full")
    out["prep_to_current"]["d_prep"] = patch_eval(current, pairs, prep_donor, dev, mask, direction=prep_d, mode="d_only")
    # current -> prep: information carried by the current signal, tested by preparation downstream.
    out["current_to_prep"]["full"] = patch_eval(prep, pairs, curr_donor, dev, mask, mode="full")
    out["current_to_prep"]["d_prep"] = patch_eval(prep, pairs, curr_donor, dev, mask, direction=prep_d, mode="d_only")
    out["current_to_prep"]["d_own"] = patch_eval(prep, pairs, curr_donor, dev, mask, direction=own_d, mode="d_only")
    return out


# ───────────────────────── behavior and record ────────────────────────────
def quick_behavior(model, probes, dev, cm_blk, prep_tok, prep_out):
    pack = S19b.eval_pack(model, probes, dev, cm_blk, prep_tok, prep_out)
    sm = S19b.metric_summary(pack)
    return {k: r6(sm.get(k, 0.0)) for k in [
        "train_top4", "train_b", "train_sel", "held_top4", "held_b", "held_sel",
        "blocked_train_top4", "blocked_train_b"]}


def evaluate_checkpoint(seed, arm_name, be, model, prep_model, prep_d, prep_dir_train, probes,
                        prep_tok, prep_out, dir_banks, pairs_by_case, prep_donors_by_case,
                        eval_train_seqs, eval_held_seqs, dev, cm, cm_blk, n_own_dir):
    own_d, own_train = learn_l1_direction(model, dev, cm, seed=360000 + seed*101 + be + 1000*len(arm_name),
                                          n_train=n_own_dir, n_test=max(100, n_own_dir//3))
    rec = dict(seed=int(seed), arm=arm_name, branch_epoch=int(be))
    rec["behavior"] = quick_behavior(model, probes, dev, cm_blk, prep_tok, prep_out)
    rec["fixed_loss_train"] = eval_ce(model, eval_train_seqs, dev, cm)
    rec["fixed_loss_heldquery"] = eval_ce(model, eval_held_seqs, dev, cm)
    rec["direction_training"] = {"prep_direction_on_prep": prep_dir_train, "own_direction_on_current": own_train}
    rec["dir_acc_prep_d"] = direction_bank_metrics(model, prep_d, dir_banks, dev, cm)
    rec["dir_acc_own_d"] = direction_bank_metrics(model, own_d, dir_banks, dev, cm)
    rec["cases"] = {}
    for cname, pairs in pairs_by_case.items():
        curr_pack = extract_donor_l1(model, pairs, dev, cm)
        rec["cases"][cname] = eval_case_bundle(model, prep_model, pairs, prep_d, own_d,
                                                prep_donors_by_case[cname], curr_pack, dev, cm)
    return rec


def train_and_trace(seed, arm_name, total_epochs, schedule, args, cfg, prep_state,
                    prep_model, prep_d, prep_dir_train, prep_tok, prep_out, probes,
                    dir_banks, pairs_by_case, prep_donors_by_case, eval_train_seqs,
                    eval_held_seqs, dev, cm, cm_blk):
    P = DEFAULT_PREP[seed]
    arm = ARMS[arm_name]
    model = S19b.make_untied_from_tied_state(prep_state, cfg, dev)
    opt = torch.optim.AdamW(model.parameters(), lr=arm["lr"], weight_decay=cfg["wd"])
    records = []
    sched = set(schedule)
    if 0 in sched:
        records.append(evaluate_checkpoint(seed, arm_name, 0, model, prep_model, prep_d, prep_dir_train,
                                           probes, prep_tok, prep_out, dir_banks, pairs_by_case,
                                           prep_donors_by_case, eval_train_seqs, eval_held_seqs,
                                           dev, cm, cm_blk, args.n_own_dir))
    last_info = None
    for be in range(1, total_epochs + 1):
        rows = S16.make_epoch_rows(seed, P + be, args.n_train)
        seqs = S16.rows_to_seqs(rows, "query_first", "bound")
        idx = S16.common_order(seed, P + be, args.n_train)
        cw = ctx_weight_for(arm, be)
        last_info = S25.train_epoch(model, opt, seqs, idx, dev, cfg["bs"], cm, "static", cw)
        if be in sched:
            rec = evaluate_checkpoint(seed, arm_name, be, model, prep_model, prep_d, prep_dir_train,
                                      probes, prep_tok, prep_out, dir_banks, pairs_by_case,
                                      prep_donors_by_case, eval_train_seqs, eval_held_seqs,
                                      dev, cm, cm_blk, args.n_own_dir)
            rec["last_train_epoch_loss"] = last_info
            rec["last_ctx_weight"] = r6(cw)
            records.append(rec)
            b = rec["behavior"]
            tr = rec["cases"]["train"]["self_current"]["d_prep"]["redirect"]
            hd = rec["cases"]["held_donor"]["self_current"]["d_prep"]["redirect"]
            ptch = rec["cases"]["held_donor"]["prep_to_current"]["d_prep"]["redirect"]
            print(f"  {arm_name:<20s} be={be:3d} h4={b['held_top4']:.3f} tr4={b['train_top4']:.3f} "
                  f"selfD train={tr:.3f} held={hd:.3f} prep→cur held={ptch:.3f} ctx={rec['fixed_loss_train']['ctx_ce']:.3f}",
                  flush=True)
    return records


# ───────────────────────── analysis/note ──────────────────────────────────
def _get_case_metric(rec, case, route, mode, metric="redirect"):
    return rec["cases"][case][route][mode][metric]


def summarize(data):
    out = {}
    for arm in data["config"]["arms"]:
        out[arm] = {}
        for be in data["config"]["schedule"]:
            rs = [r for r in data["records"] if r["arm"] == arm and int(r["branch_epoch"]) == int(be)]
            if not rs:
                continue
            def mean_path(fn):
                vals = [fn(r) for r in rs]
                return r6(np.mean(vals))
            out[arm][str(be)] = dict(
                n=len(rs),
                train_top4=mean_path(lambda r: r["behavior"]["train_top4"]),
                held_top4=mean_path(lambda r: r["behavior"]["held_top4"]),
                held_b=mean_path(lambda r: r["behavior"]["held_b"]),
                ctx_ce=mean_path(lambda r: r["fixed_loss_train"]["ctx_ce"]),
                dir_train=mean_path(lambda r: r["dir_acc_prep_d"]["train"]["acc"]),
                dir_held=mean_path(lambda r: r["dir_acc_prep_d"]["held_query"]["acc"]),
                self_train_d=mean_path(lambda r: _get_case_metric(r, "train", "self_current", "d_prep")),
                self_held_d=mean_path(lambda r: _get_case_metric(r, "held_donor", "self_current", "d_prep")),
                prep_to_cur_train_d=mean_path(lambda r: _get_case_metric(r, "train", "prep_to_current", "d_prep")),
                prep_to_cur_held_d=mean_path(lambda r: _get_case_metric(r, "held_donor", "prep_to_current", "d_prep")),
                cur_to_prep_train_d=mean_path(lambda r: _get_case_metric(r, "train", "current_to_prep", "d_prep")),
                cur_to_prep_held_d=mean_path(lambda r: _get_case_metric(r, "held_donor", "current_to_prep", "d_prep")),
            )
    data["summary"] = out
    return out


def write_note(data, path):
    summary = data.get("summary", {})
    lines = [
        "# research causal interface trajectory",
        "",
        "## Purpose",
        "",
        "This experiment follows the research donor-query causal interface through continuation. It does not treat familiar-symbol redirection as the endpoint. The measurements separate three questions: whether the model forms a query-conditioned L1 signal, whether later layers/readout remain sensitive to a supplied preparation signal, and whether the effect narrows from train symbols to held symbols.",
        "",
        "## Mean trajectory across seeds",
        "",
    ]
    for arm, by_epoch in summary.items():
        lines += [f"### {arm}", "", "| be | train4 | held4 | heldB | ctxCE | dir train/held | self d train/held | prep→cur d train/held | cur→prep d train/held |", "|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for be in sorted(by_epoch, key=lambda x: int(x)):
            s = by_epoch[be]
            lines.append(
                f"| {be} | {s['train_top4']:.3f} | {s['held_top4']:.3f} | {s['held_b']:+.3f} | {s['ctx_ce']:.3f} | "
                f"{s['dir_train']:.3f}/{s['dir_held']:.3f} | "
                f"{s['self_train_d']:.3f}/{s['self_held_d']:.3f} | "
                f"{s['prep_to_cur_train_d']:.3f}/{s['prep_to_cur_held_d']:.3f} | "
                f"{s['cur_to_prep_train_d']:.3f}/{s['cur_to_prep_held_d']:.3f} |"
            )
        lines.append("")
    lines += [
        "## Reading guide",
        "",
        "- `dir train/held` is L1 slot classification by the preparation-learned direction in the current model. It measures separability, not causal use.",
        "- `self d train/held` patches the current model's own donor activation into itself along the preparation direction. It measures formation plus current downstream use.",
        "- `prep→cur d` patches the preparation donor signal into the current recipient. High values mean current downstream layers can still use a well-formed preparation-era signal.",
        "- `cur→prep d` patches the current donor signal into the preparation recipient. Low values mean the current activation no longer carries the preparation-readable query-selection component.",
        "- Held columns use held-donor pairs: the donor query is a held entity in a context containing one held entity and three train entities; the recipient query is a train entity from the same context.",
        "",
        "## Main interpretation",
        "",
        "The causal interface should be read against behavioral held transfer. If familiar `self d` remains high while held `self d` falls, the continuation has narrowed formation of the reusable selector rather than globally preserving it. If `prep→cur d` remains high when `self d` is low, the recipient can use a supplied signal, so loss is upstream of or at signal formation rather than a complete failure of downstream readout. If `cur→prep d` is low, the current donor activation itself lacks the preparation-readable signal. Seed-level details in `results.json` are necessary for the subtle seed100 direct-full case, where direction classification can be near chance while d-only redirection remains functional.",
        "",
        "## Files",
        "",
        f"- Data: `{data['paths']['data']}`",
        f"- Figure: `{data['paths'].get('figure','')}`",
        f"- Script: `{data['paths']['script']}`",
    ]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines) + "\n")


def make_figure(data, path):
    import matplotlib.pyplot as plt
    schedule = [int(x) for x in data["config"]["schedule"]]
    arms = data["config"]["arms"]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True)
    panels = [
        ("held_top4", "Held behavioral top-4"),
        ("self_held_d", "Held self d-only redirect"),
        ("prep_to_cur_held_d", "Held prep→current d redirect"),
        ("train_top4", "Familiar behavioral top-4"),
        ("self_train_d", "Familiar self d-only redirect"),
        ("ctx_ce", "Fixed train-context CE"),
    ]
    colors = dict(direct_full="tab:red", static_1over17="tab:blue", interleaved_ans_full="tab:green")
    for ax, (key, title) in zip(axes.ravel(), panels):
        for arm in arms:
            xs, ys = [], []
            for be in schedule:
                s = data.get("summary", {}).get(arm, {}).get(str(be))
                if s is None:
                    continue
                xs.append(be); ys.append(s[key])
            if xs:
                ax.plot(xs, ys, marker="o", label=arm, color=colors.get(arm, None))
        ax.set_title(title)
        ax.set_xlabel("Continuation epoch")
        if key != "ctx_ce":
            ax.set_ylim(-0.03, 1.03)
        ax.grid(alpha=0.25)
    axes[0,0].legend(fontsize=8)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint-dir", default="experiments/archive/functional_learning/data/revision_019b_embedding_role_decomposition/checkpoints")
    ap.add_argument("--data", default="experiments/archive/functional_learning/data/causal_interface_trajectory/results.json")
    ap.add_argument("--note", default="research/notes/functional_learning/causal_interface_trajectory.md")
    ap.add_argument("--figure", default="experiments/archive/functional_learning/figures/causal_interface_trajectory.png")
    ap.add_argument("--seeds", default="43,100")
    ap.add_argument("--arms", default="direct_full,static_1over17,interleaved_ans_full")
    ap.add_argument("--total-epochs", type=int, default=500)
    ap.add_argument("--schedule", default="0,1,5,25,100,150,250,500")
    ap.add_argument("--n-train", type=int, default=500)
    ap.add_argument("--n-pairs-train", type=int, default=256)
    ap.add_argument("--n-pairs-held", type=int, default=160)
    ap.add_argument("--n-dir-bank", type=int, default=300)
    ap.add_argument("--n-own-dir", type=int, default=700)
    ap.add_argument("--n-probe-std", type=int, default=256)
    ap.add_argument("--n-probe-small", type=int, default=128)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    if args.smoke:
        args.seeds = "100"
        args.arms = "direct_full,static_1over17"
        args.total_epochs = 10
        args.schedule = "0,1,5,10"
        args.n_train = 64
        args.n_pairs_train = 64
        args.n_pairs_held = 40
        args.n_dir_bank = 80
        args.n_own_dir = 180
        args.n_probe_std = 96
        args.n_probe_small = 48

    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    arms = [x.strip() for x in args.arms.split(",") if x.strip()]
    schedule = sorted({int(x) for x in args.schedule.split(",") if x.strip() and int(x) <= args.total_epochs})
    cfg = dict(d=64, nh=2, nl=3, wd=0.01, bs=64)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cm = S17.standard_causal_mask(SL-1, dev)
    cm_blk = S17.block_query_ctx_mask(SL-1, dev)
    probes = S18.make_probes(seed=180018, n_std=args.n_probe_std, n_small=args.n_probe_small)
    ckpt_dir = Path(args.checkpoint_dir)

    # Fixed evaluation banks shared across seeds/arms.
    dir_banks = {
        "train": gen_dir_bank(260001, args.n_dir_bank, "train"),
        "held_query": gen_dir_bank(260002, args.n_dir_bank, "held_query"),
        "train_in_held_ctx": gen_dir_bank(260003, args.n_dir_bank, "train_in_held_ctx"),
    }
    pairs_by_case = {
        "train": S25.gen_train_pairs(260010, args.n_pairs_train),
    }
    hp = S25.gen_held_pairs(260011, args.n_pairs_held)
    pairs_by_case["held_donor"] = hp["held_donor"]
    pairs_by_case["train_in_held_ctx"] = hp["train_donor"]
    eval_train_rows = S16.make_epoch_rows(260020, 0, 512 if not args.smoke else 96)
    eval_train_seqs = S16.rows_to_seqs(eval_train_rows, "query_first", "bound")
    eval_held_seqs = [S16.mk_seq("query_first", ce, ca, qi, RWT(ca[qi]))
                      for ce, ca, qi, _ in held_eval_rows(260021, 384 if not args.smoke else 80)]

    data = dict(
        config=dict(seeds=seeds, arms=arms, total_epochs=args.total_epochs, schedule=schedule,
                    n_train=args.n_train, n_pairs_train=args.n_pairs_train,
                    n_pairs_held=args.n_pairs_held, n_dir_bank=args.n_dir_bank,
                    n_own_dir=args.n_own_dir),
        paths=dict(script="experiments/archive/functional_learning/scripts/causal_interface_trajectory.py",
                   data=args.data, note=args.note, figure=args.figure),
        records=[], summary={}
    )

    t0 = time.time()
    print(f"Device: {dev}; seeds={seeds}; arms={arms}; schedule={schedule}", flush=True)

    for sd in seeds:
        P = DEFAULT_PREP[sd]
        prep_state = torch.load(ckpt_dir / f"seed{sd}_prep_tied.pt", map_location=dev)
        prep_model = S19b.make_untied_from_tied_state(prep_state, cfg, dev)
        prep_tok = prep_state["tok.weight"].detach().clone().to(dev)
        prep_out = prep_model.out.weight.detach().clone()
        prep_d, prep_dir_train = learn_l1_direction(prep_model, dev, cm, seed=360000 + sd*101,
                                                    n_train=args.n_own_dir, n_test=max(100, args.n_own_dir//3))
        prep_donors_by_case = {c: extract_donor_l1(prep_model, p, dev, cm) for c, p in pairs_by_case.items()}
        print("\n" + "="*88)
        print(f"Seed {sd}; prep P={P}; prep L1 dir test={prep_dir_train['test']['acc']:.3f}/{prep_dir_train['test']['margin']:+.3f}")
        print("="*88, flush=True)
        for arm in arms:
            if arm not in ARMS:
                raise ValueError(f"unknown arm {arm}")
            recs = train_and_trace(sd, arm, args.total_epochs, schedule, args, cfg, prep_state,
                                   prep_model, prep_d, prep_dir_train, prep_tok, prep_out,
                                   probes, dir_banks, pairs_by_case, prep_donors_by_case,
                                   eval_train_seqs, eval_held_seqs, dev, cm, cm_blk)
            data["records"].extend(recs)

    summarize(data)
    outp = Path(args.data); outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(data, indent=2, cls=NpEnc))
    make_figure(data, args.figure)
    write_note(data, args.note)
    print(json.dumps(dict(status="ok", out=args.data, note=args.note, figure=args.figure,
                          elapsed=round(time.time() - t0, 1)), indent=2))


if __name__ == "__main__":
    main()
