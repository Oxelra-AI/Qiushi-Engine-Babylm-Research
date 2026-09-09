#!/usr/bin/env python3
"""Step019b: input/output embedding-role decomposition for held binding loss.

Scientific target. research showed that answer-only query-first preparation can
produce strong held-symbol binding, but subsequent full next-token training can
recover perfect familiar-symbol binding while held-symbol transfer collapses.
The CLM ties input embeddings to the output classifier.  A held entity token is
never a training input or correct target, yet full softmax training can still move
its tied row as a non-target output class; because the same row is used as input,
this may damage held-symbol matching without destroying the contextual selector.

This corrected script adds zero-training input/output hybrid evaluations that
separate held input-row drift from held output-row effects in final tied models.
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
    dict(name="tied_carry_freeze_held_shared", model="tied", opt="carry", freeze_held=True),
    dict(name="tied_reset_full", model="tied", opt="reset", freeze_held=False),
    dict(name="tied_reset_freeze_held_shared", model="tied", opt="reset", freeze_held=True),
    dict(name="untied_reset_full", model="untied", opt="reset", freeze_held=False),
    dict(name="untied_reset_freeze_held_input", model="untied", opt="reset", freeze_held=True),
]
BRANCH_SEED_OFFSETS = {b["name"]: 1000 + 17 * i for i, b in enumerate(BRANCHES)}


class UntiedCLM(nn.Module):
    """Same transformer as Base.CLM, but with a separate output classifier."""
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


def choose_branches(text, smoke=False):
    if smoke and text == "all":
        # Include both tied and untied code paths in smoke testing.
        names = ["tied_carry_full", "tied_carry_freeze_held_shared", "untied_reset_full"]
    elif text == "all":
        names = [b["name"] for b in BRANCHES]
    else:
        names = [x.strip() for x in text.split(',') if x.strip()]
    index = {b["name"]: b for b in BRANCHES}
    missing = [n for n in names if n not in index]
    if missing:
        raise ValueError(f"unknown branches: {missing}")
    return [copy.deepcopy(index[n]) for n in names]


def make_tied(cfg, dev):
    return Base.CLM(VOCAB, cfg["d"], cfg["nh"], cfg["nl"], SL).to(dev)


def make_untied_from_tied_state(tied_state, cfg, dev):
    model = UntiedCLM(VOCAB, cfg["d"], cfg["nh"], cfg["nl"], SL).to(dev)
    own = model.state_dict()
    for k in own.keys():
        if k == "out.weight":
            own[k].copy_(tied_state["tok.weight"])
        elif k in tied_state:
            own[k].copy_(tied_state[k])
        else:
            raise KeyError(f"unexpected untied state key {k}")
    model.load_state_dict(own)
    return model


def restore_rows_(matrix, rows, values):
    """Scatter values into selected rows of a Parameter or Tensor in-place."""
    with torch.no_grad():
        matrix.data.index_copy_(0, rows, values)


def category_stats(W, W0, rows):
    rows_t = torch.tensor(rows, device=W.device, dtype=torch.long)
    X = W[rows_t].detach(); X0 = W0[rows_t].detach()
    diff = X - X0
    l2 = torch.linalg.norm(diff, dim=1)
    base = torch.linalg.norm(X0, dim=1).clamp_min(1e-12)
    rel = l2 / base
    cos = F.cosine_similarity(X, X0, dim=1)
    out = dict(l2_mean=float(l2.mean().item()), l2_max=float(l2.max().item()),
               rel_mean=float(rel.mean().item()), cos_mean=float(cos.mean().item()))
    # Preserve per-row detail for the two held rows and compact detail for small groups.
    if len(rows) <= 8:
        out["per_row"] = [{"row": int(r), "l2": float(l2[i].item()),
                            "rel": float(rel[i].item()), "cos": float(cos[i].item()),
                            "norm": float(torch.linalg.norm(X[i]).item()),
                            "prep_norm": float(torch.linalg.norm(X0[i]).item())}
                           for i, r in enumerate(rows)]
    return out


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
        # In a tied model the same rows have both roles; label as shared-row drift.
        out["output"] = copy.deepcopy(out["input"])
    if len(HELD_ENT_ROWS) == 2:
        rows_t = torch.tensor(HELD_ENT_ROWS, device=W.device, dtype=torch.long)
        d = W[rows_t].detach() - prep_tok_weight[rows_t].detach()
        denom = (torch.linalg.norm(d[0]) * torch.linalg.norm(d[1])).clamp_min(1e-12)
        out["input"]["held_pair_displacement_cos"] = float(torch.dot(d[0], d[1]).item() / denom.item())
    return out


def train_one_epoch(model, opt, seqs, order_idx, dev, mode, bs, mask,
                    freeze_rows=None, freeze_values=None):
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
        if freeze_rows is not None:
            restore_rows_(model.tok.weight, freeze_rows, freeze_values)
        tot += float((ce * wt).sum().detach()); denom += float(d.detach())
        ans_s += float(ce[:, IS_POS].sum().detach()); cnt += int(ce.size(0))
        cm2 = nonpad.clone(); cm2[:, IS_POS] = 0.0
        ctx_s += float((ce * cm2).sum().detach()); ctx_d += float(cm2.sum().detach())
    return dict(wloss=round(tot / max(denom, 1), 6),
                ans_ce=round(ans_s / max(cnt, 1), 6),
                ctx_ce=round(ctx_s / max(ctx_d, 1), 6))


def eval_pack(model, probes, dev, cm_blk, prep_tok_weight=None, prep_out_weight=None):
    model.eval()
    qf = S18.eval_extended(model, probes, "query_first", dev)
    qfb = S18.eval_extended(S17.MaskOverride(model, cm_blk), probes, "query_first", dev)
    out = dict(qf=qf, qf_blocked=qfb)
    if prep_tok_weight is not None:
        out["drift"] = row_drift(model, prep_tok_weight, prep_out_weight)
    return out


def eval_untied_with_row_substitution(model, probes, dev, cm_blk,
                                      input_rows=None, input_values=None,
                                      output_rows=None, output_values=None,
                                      prep_tok_weight=None, prep_out_weight=None):
    old_in = None; old_out = None
    if input_rows is not None:
        old_in = model.tok.weight.data.index_select(0, input_rows).detach().clone()
        restore_rows_(model.tok.weight, input_rows, input_values)
    if output_rows is not None:
        old_out = model.out.weight.data.index_select(0, output_rows).detach().clone()
        restore_rows_(model.out.weight, output_rows, output_values)
    try:
        return eval_pack(model, probes, dev, cm_blk, prep_tok_weight, prep_out_weight)
    finally:
        if output_rows is not None:
            restore_rows_(model.out.weight, output_rows, old_out)
        if input_rows is not None:
            restore_rows_(model.tok.weight, input_rows, old_in)


def metric_summary(pack):
    qf = pack["qf"]
    return dict(
        train_nll=qf["std"]["correct_nll"], train_top4=qf["std"]["ctx_top1"],
        train_b=qf["bswap"]["mean"], train_b_frac=qf["bswap"].get("frac_pos", None),
        train_q=qf["qswap"]["margin"], train_q_frac=qf["qswap"].get("frac_pos", None),
        train_q_both=qf["qswap"].get("both", None), train_sel=qf["corrupt"]["novel_selectivity"],
        held_top4=qf["held"]["ctx_top1"], held_b=qf["held_bswap"]["mean"],
        held_b_frac=qf["held_bswap"].get("frac_pos", None), held_q=qf["held_qswap"]["margin"],
        held_sel=qf["held_corrupt"]["novel_selectivity"],
        blocked_train_top4=pack["qf_blocked"]["std"]["ctx_top1"],
        blocked_train_b=pack["qf_blocked"]["bswap"]["mean"],
    )


def add_drift_to_summary(sm, pack):
    if "drift" not in pack:
        return sm
    din = pack["drift"]["input"]
    sm["held_ent_input_l2"] = din["held_ent"]["l2_mean"]
    sm["held_ent_input_cos"] = din["held_ent"]["cos_mean"]
    sm["held_pair_disp_cos"] = din.get("held_pair_displacement_cos")
    sm["train_ent_input_l2"] = din["train_ent"]["l2_mean"]
    sm["rwt_input_l2"] = din["rwt"]["l2_mean"]
    if "output" in pack["drift"]:
        dout = pack["drift"]["output"]
        sm["held_ent_output_l2"] = dout["held_ent"]["l2_mean"]
        sm["rwt_output_l2"] = dout["rwt"]["l2_mean"]
    return sm


def strong_train(pack):
    q = pack["qf"]
    return (q["std"]["ctx_top1"] >= 0.95 and
            q["bswap"]["mean"] >= 5.0 and q["bswap"].get("frac_pos", 0.0) >= 0.95 and
            q["qswap"]["margin"] >= 5.0 and q["qswap"].get("frac_pos", 0.0) >= 0.95 and
            q["qswap"].get("both", 0.0) >= 0.90 and
            q["corrupt"]["novel_selectivity"] >= 0.80)


def absolute_held_strong(pack):
    q = pack["qf"]
    return (q["held"]["ctx_top1"] >= 0.75 and q["held_bswap"]["mean"] >= 5.0 and
            q["held_corrupt"]["novel_selectivity"] >= 0.5)


def gap_closed(prep, final, test, key):
    denom = prep[key] - final[key]
    if abs(denom) < 1e-9:
        return None
    return (test[key] - final[key]) / denom


def should_eval(ep, prep_epoch, total_epochs, eval_every):
    specials = {prep_epoch, prep_epoch + 1, prep_epoch + 25, prep_epoch + 50,
                prep_epoch + 100, prep_epoch + 250, prep_epoch + 500, total_epochs}
    return ep in specials or (ep % eval_every == 0)


def train_prep(seed, prep_epoch, n_train, cfg, dev, cm_std, cm_blk, probes):
    torch.manual_seed(seed); np.random.seed(seed)
    model = make_tied(cfg, dev)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    recs = []
    ts = time.time()
    for ep in range(1, prep_epoch + 1):
        rows = S16.make_epoch_rows(seed, ep, n_train)
        seqs = S16.rows_to_seqs(rows, "query_first", "bound")
        idx = S16.common_order(seed, ep, n_train)
        info = train_one_epoch(model, opt, seqs, idx, dev, "answer_only", cfg["bs"], cm_std)
        if ep == 1 or ep == prep_epoch or ep % cfg["prep_eval_every"] == 0:
            ref = model.tok.weight.detach().clone()
            pack = eval_pack(model, probes, dev, cm_blk, ref)
            rec = dict(seed=seed, epoch=ep, elapsed=round(time.time() - ts, 1), **info)
            rec.update(pack); recs.append(rec)
            sm = metric_summary(pack)
            print(f"  prep e={ep:4d} train4={sm['train_top4']:.3f} held4={sm['held_top4']:.3f} "
                  f"B={sm['train_b']:+.3f} hB={sm['held_b']:+.3f} hSel={sm['held_sel']:+.3f}", flush=True)
    return (copy.deepcopy(model.state_dict()), copy.deepcopy(opt.state_dict()),
            model.tok.weight.detach().clone(), recs)


def endpoint_hybrids_for_tied(final_tied_model, prep_tied_state, prep_tok_weight,
                              cfg, dev, cm_blk, probes):
    rows = torch.tensor(HELD_ENT_ROWS, device=dev, dtype=torch.long)
    prep_vals = prep_tok_weight[rows].detach().clone()
    final_vals = final_tied_model.tok.weight.detach()[rows].clone()

    final_state = copy.deepcopy(final_tied_model.state_dict())
    final_u = make_untied_from_tied_state(final_state, cfg, dev)
    prep_u = make_untied_from_tied_state(prep_tied_state, cfg, dev)

    out = {}
    # Final contextual/readout state. Labels indicate held input/output row source.
    out["final_FF"] = eval_untied_with_row_substitution(
        final_u, probes, dev, cm_blk, prep_tok_weight=prep_tok_weight, prep_out_weight=prep_tok_weight)
    out["final_PF_inputprep_outputfinal"] = eval_untied_with_row_substitution(
        final_u, probes, dev, cm_blk, input_rows=rows, input_values=prep_vals,
        prep_tok_weight=prep_tok_weight, prep_out_weight=prep_tok_weight)
    out["final_FP_inputfinal_outputprep"] = eval_untied_with_row_substitution(
        final_u, probes, dev, cm_blk, output_rows=rows, output_values=prep_vals,
        prep_tok_weight=prep_tok_weight, prep_out_weight=prep_tok_weight)
    out["final_PP_inputprep_outputprep"] = eval_untied_with_row_substitution(
        final_u, probes, dev, cm_blk, input_rows=rows, input_values=prep_vals,
        output_rows=rows, output_values=prep_vals,
        prep_tok_weight=prep_tok_weight, prep_out_weight=prep_tok_weight)

    # Reverse transplant: can final held input rows break the preparation network?
    out["prep_PP_baseline"] = eval_untied_with_row_substitution(
        prep_u, probes, dev, cm_blk, prep_tok_weight=prep_tok_weight, prep_out_weight=prep_tok_weight)
    out["prep_FP_inputfinal_outputprep"] = eval_untied_with_row_substitution(
        prep_u, probes, dev, cm_blk, input_rows=rows, input_values=final_vals,
        prep_tok_weight=prep_tok_weight, prep_out_weight=prep_tok_weight)
    return out


def train_branch(seed, branch, prep_epoch, total_epochs, n_train, cfg, dev,
                 cm_std, cm_blk, probes, tied_state, opt_state, prep_tok_weight):
    branch_seed = seed + 100000 + BRANCH_SEED_OFFSETS[branch["name"]]
    torch.manual_seed(branch_seed)
    np.random.seed(branch_seed)
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

    freeze_rows = torch.tensor(HELD_ENT_ROWS, device=dev, dtype=torch.long) if branch["freeze_held"] else None
    freeze_vals = prep_tok_weight[freeze_rows].detach().clone() if freeze_rows is not None else None
    recs = []
    ts = time.time()

    def add_rec(ep, info):
        pack = eval_pack(model, probes, dev, cm_blk, prep_tok_weight, prep_out_weight)
        rec = dict(seed=seed, branch=branch["name"], model_type=branch["model"],
                   opt_mode=branch["opt"], freeze_held=branch["freeze_held"],
                   epoch=ep, prep_epoch=prep_epoch, branch_epoch=ep - prep_epoch,
                   elapsed=round(time.time() - ts, 1), **info)
        rec.update(pack)
        recs.append(rec)
        sm = metric_summary(pack); sm = add_drift_to_summary(sm, pack)
        print(f"    {branch['name']:<35s} e={ep:4d} be={ep-prep_epoch:4d} "
              f"tr4={sm['train_top4']:.3f} h4={sm['held_top4']:.3f} "
              f"hB={sm['held_b']:+.3f} hSel={sm['held_sel']:+.3f} "
              f"blk4={sm['blocked_train_top4']:.3f} hL2={sm['held_ent_input_l2']:.3f}", flush=True)
        return pack

    final_pack = add_rec(prep_epoch, dict(wloss=0.0, ans_ce=0.0, ctx_ce=0.0))
    for ep in range(prep_epoch + 1, total_epochs + 1):
        rows = S16.make_epoch_rows(seed, ep, n_train)
        seqs = S16.rows_to_seqs(rows, "query_first", "bound")
        idx = S16.common_order(seed, ep, n_train)
        info = train_one_epoch(model, opt, seqs, idx, dev, "full", cfg["bs"], cm_std,
                               freeze_rows=freeze_rows, freeze_values=freeze_vals)
        if should_eval(ep, prep_epoch, total_epochs, cfg["eval_every"]):
            final_pack = add_rec(ep, info)

    hybrids = None
    if branch["model"] == "tied":
        hybrids = endpoint_hybrids_for_tied(model, tied_state, prep_tok_weight, cfg, dev, cm_blk, probes)
        hsm = {k: add_drift_to_summary(metric_summary(v), v) for k, v in hybrids.items()}
        pf = hsm["final_PF_inputprep_outputfinal"]
        ff = hsm["final_FF"]
        pp0 = hsm["prep_PP_baseline"]
        fp0 = hsm["prep_FP_inputfinal_outputprep"]
        print(f"      hybrid final FF h4={ff['held_top4']:.3f}; PF(input prep) h4={pf['held_top4']:.3f}; "
              f"prep->input final h4={fp0['held_top4']:.3f} vs prep {pp0['held_top4']:.3f}", flush=True)
    return recs, hybrids, copy.deepcopy(model.state_dict())


def nearest(rs, branch_epoch):
    return min(rs, key=lambda r: abs(int(r["branch_epoch"]) - int(branch_epoch))) if rs else None


def summarize(data):
    S = {"by_seed": {}, "by_branch": {}}
    branches = [b["name"] for b in data["branches_spec"]]
    for sd in data["config"]["seeds"]:
        S["by_seed"][str(sd)] = {}
        for br in branches:
            rs = sorted([r for r in data["branches"] if r["seed"] == sd and r["branch"] == br], key=lambda r: r["epoch"])
            if not rs:
                continue
            prep = metric_summary(rs[0]); prep = add_drift_to_summary(prep, rs[0])
            be500 = nearest(rs, 500)
            final = metric_summary(rs[-1]); final = add_drift_to_summary(final, rs[-1])
            item = dict(prep=prep, be500=None if be500 is None else add_drift_to_summary(metric_summary(be500), be500),
                        final=final, train_strong_final=strong_train(rs[-1]),
                        abs_held_strong_final=absolute_held_strong(rs[-1]),
                        delta_final_vs_prep={k: (final[k] - prep[k]) for k in ["held_top4", "held_b", "held_sel", "train_top4", "train_b", "train_sel"]})
            if br in data.get("hybrids", {}).get(str(sd), {}):
                H = data["hybrids"][str(sd)][br]
                hsum = {name: add_drift_to_summary(metric_summary(pack), pack) for name, pack in H.items()}
                final_ff = hsum["final_FF"]
                item["hybrid_summary"] = hsum
                item["input_restore_gap_closed"] = {
                    k: gap_closed(prep, final_ff, hsum["final_PF_inputprep_outputfinal"], k)
                    for k in ["held_top4", "held_b", "held_sel"]
                }
                item["reverse_input_final_delta"] = {
                    k: hsum["prep_FP_inputfinal_outputprep"][k] - hsum["prep_PP_baseline"][k]
                    for k in ["held_top4", "held_b", "held_sel"]
                }
            S["by_seed"][str(sd)][br] = item
    for br in branches:
        finals = [S["by_seed"][str(sd)][br]["final"] for sd in data["config"]["seeds"] if br in S["by_seed"].get(str(sd), {})]
        be500s = [S["by_seed"][str(sd)][br]["be500"] for sd in data["config"]["seeds"] if br in S["by_seed"].get(str(sd), {}) and S["by_seed"][str(sd)][br]["be500"] is not None]
        def mean(arr, key):
            return None if not arr else round(float(np.mean([x[key] for x in arr])), 6)
        S["by_branch"][br] = dict(
            n=len(finals),
            train_strong_final=sum(1 for sd in data["config"]["seeds"] if S["by_seed"].get(str(sd), {}).get(br, {}).get("train_strong_final")),
            abs_held_strong_final=sum(1 for sd in data["config"]["seeds"] if S["by_seed"].get(str(sd), {}).get(br, {}).get("abs_held_strong_final")),
            final_train_top4=mean(finals, "train_top4"), final_held_top4=mean(finals, "held_top4"),
            final_held_b=mean(finals, "held_b"), final_held_sel=mean(finals, "held_sel"),
            final_blocked_top4=mean(finals, "blocked_train_top4"),
            final_held_input_l2=mean(finals, "held_ent_input_l2"),
            be500_train_top4=mean(be500s, "train_top4"), be500_held_top4=mean(be500s, "held_top4"),
            be500_held_b=mean(be500s, "held_b"), be500_held_sel=mean(be500s, "held_sel"),
        )
    return S


def make_figure(data, figpath):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    branches = [b["name"] for b in data["branches_spec"]]
    colors = {
        "tied_carry_full": "tab:red",
        "tied_carry_freeze_held_shared": "tab:blue",
        "tied_reset_full": "tab:pink",
        "tied_reset_freeze_held_shared": "tab:purple",
        "untied_reset_full": "tab:orange",
        "untied_reset_freeze_held_input": "tab:green",
    }
    plots = [
        (("qf", "std", "ctx_top1"), "Trained top4", 0.25, 0.95),
        (("qf", "held", "ctx_top1"), "Held-query top4", 0.25, 0.75),
        (("qf", "held_bswap", "mean"), "Held B-swap", 0.0, 5.0),
        (("qf", "held_corrupt", "novel_selectivity"), "Held selectivity", 0.0, 0.5),
        (("qf_blocked", "std", "ctx_top1"), "Blocked q→ctx trained top4", 0.25, 0.95),
        (("drift", "input", "held_ent", "l2_mean"), "Held input-row drift", 0.0, None),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(18, 9))
    for ax, (path, title, base_line, strong_line) in zip(axes.flat, plots):
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
            ax.plot(xs, ys, color=colors.get(br), lw=1.5, label=br.replace("_", " "))
        if base_line is not None:
            ax.axhline(base_line, color="black", ls="--", alpha=0.28)
        if strong_line is not None:
            ax.axhline(strong_line, color="black", ls=":", alpha=0.25)
        ax.set_title(title, fontsize=10); ax.set_xlabel("epochs after switch"); ax.grid(alpha=0.25); ax.legend(fontsize=6)
    fig.suptitle("Step019b: held-symbol transfer under embedding-role interventions", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    Path(figpath).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figpath, dpi=160, bbox_inches="tight")
    plt.close(fig)


def fmt(x):
    if x is None:
        return ""
    return f"{x:.3f}" if isinstance(x, float) else str(x)


def write_note(data, path):
    S = data["summary"]
    lines = []
    lines.append("# Step019b result: embedding-role decomposition of held-symbol binding loss")
    lines.append("")
    lines.append("This note is generated from `data/revision_019b_embedding_role_decomposition/results.json`. It uses query-first answer-only prepared checkpoints and continues with the full next-token objective while manipulating held entity input rows and tied/untied output roles.")
    lines.append("")
    lines.append("## Branch-level final summary")
    lines.append("")
    lines.append("| branch | train strong | abs held strong | final train4 | final held4 | final heldB | final heldSel | blocked train4 | held input L2 | be500 held4 |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for br, sm in S["by_branch"].items():
        lines.append(f"| {br} | {sm['train_strong_final']}/{sm['n']} | {sm['abs_held_strong_final']}/{sm['n']} | {fmt(sm['final_train_top4'])} | {fmt(sm['final_held_top4'])} | {fmt(sm['final_held_b'])} | {fmt(sm['final_held_sel'])} | {fmt(sm['final_blocked_top4'])} | {fmt(sm['final_held_input_l2'])} | {fmt(sm['be500_held_top4'])} |")
    lines.append("")
    lines.append("## Per-seed final summary")
    lines.append("")
    lines.append("| seed | branch | prep held4 | final train4 | final held4 | Δheld4 | final heldB | ΔheldB | final heldSel | ΔheldSel | held input L2 | blocked train4 |")
    lines.append("|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for sd, byb in S["by_seed"].items():
        for br, item in byb.items():
            p = item["prep"]; f = item["final"]; d = item["delta_final_vs_prep"]
            lines.append(f"| {sd} | {br} | {p['held_top4']:.3f} | {f['train_top4']:.3f} | {f['held_top4']:.3f} | {d['held_top4']:+.3f} | {f['held_b']:+.3f} | {d['held_b']:+.3f} | {f['held_sel']:+.3f} | {d['held_sel']:+.3f} | {f.get('held_ent_input_l2', float('nan')):.3f} | {f['blocked_train_top4']:.3f} |")
    lines.append("")
    lines.append("## Tied-branch zero-training hybrids")
    lines.append("")
    lines.append("`final_FF` is an untied copy of the final tied model. `final_PF` restores only held input rows to preparation values while leaving output rows and the rest of the model final. `final_FP` restores held output rows only. `final_PP` restores both held input and output rows. `prep_FP` inserts final held input rows into the preparation network.")
    lines.append("")
    lines.append("| seed | branch | prep held4 | final_FF held4 | final_PF held4 | final_FP held4 | final_PP held4 | prep_FP held4 | PF gap-closed held4 | PF gap-closed heldB |")
    lines.append("|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for sd, byb in S["by_seed"].items():
        for br, item in byb.items():
            if "hybrid_summary" not in item:
                continue
            H = item["hybrid_summary"]
            prep = item["prep"]
            gc = item["input_restore_gap_closed"]
            lines.append(f"| {sd} | {br} | {prep['held_top4']:.3f} | {H['final_FF']['held_top4']:.3f} | {H['final_PF_inputprep_outputfinal']['held_top4']:.3f} | {H['final_FP_inputfinal_outputprep']['held_top4']:.3f} | {H['final_PP_inputprep_outputprep']['held_top4']:.3f} | {H['prep_FP_inputfinal_outputprep']['held_top4']:.3f} | {fmt(gc['held_top4'])} | {fmt(gc['held_b'])} |")
    lines.append("")
    lines.append("## Scientific reading")
    lines.append("")
    lines.append("Input-row drift is supported only if `final_PF` selectively restores held behavior, or if inserting final held input rows into the preparation network destroys held behavior. A null `final_PF` with strong familiar-symbol binding points to contextual network/readout specialization or to a more distributed geometry change. The tied shared-row freeze is not input-only; it anchors the same row in both input and output roles. Untied branches test global removal of weight sharing but also change optimization for the entire output classifier.")
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
    ap.add_argument("--prep-eval-every", type=int, default=75)
    ap.add_argument("--n-train", type=int, default=500)
    ap.add_argument("--probe-seed", type=int, default=180018)
    ap.add_argument("--branches", default="all")
    ap.add_argument("--save-checkpoints", action="store_true")
    A = ap.parse_args()

    ws = _public_path('experiments/archive/functional_learning')
    if A.data is None:
        A.data = str(ws / "data" / "revision_019b_embedding_role_decomposition")
    if A.fig is None:
        A.fig = str(ws / "figures" / "revision_019b_embedding_role_decomposition.png")

    if A.smoke:
        seeds = [42]
        prep_epochs = {42: 8}
        total_epochs = 24
        eval_every = 4
        prep_eval_every = 4
        n_train = 64
        n_std, n_small = 96, 48
    else:
        seeds = [int(x) for x in A.seeds.split(',') if x.strip()]
        pin = parse_prep_epochs(A.prep_epochs)
        prep_epochs = {sd: int(pin.get(sd, DEFAULT_PREP_EPOCHS.get(sd, 400))) for sd in seeds}
        total_epochs = A.total_epochs
        eval_every = A.eval_every
        prep_eval_every = A.prep_eval_every
        n_train = A.n_train
        n_std, n_small = 512, 256
    branches = choose_branches(A.branches, smoke=A.smoke)

    cfg = dict(d=64, nh=2, nl=3, lr=3e-4, wd=0.01, bs=64,
               seeds=seeds, prep_epochs={str(k): int(v) for k, v in prep_epochs.items()},
               total_epochs=total_epochs, eval_every=eval_every,
               prep_eval_every=prep_eval_every, n_train=n_train,
               probe_seed=A.probe_seed, held_ent_rows=HELD_ENT_ROWS)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cm_std = S17.standard_causal_mask(SL - 1, dev)
    cm_blk = S17.block_query_ctx_mask(SL - 1, dev)
    probes = S18.make_probes(seed=A.probe_seed, n_std=n_std, n_small=n_small)

    out_dir = Path(A.data); out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = out_dir / "checkpoints"
    if A.save_checkpoints:
        ckpt_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device: {dev}")
    print(f"Seeds={seeds}; prep={prep_epochs}; total={total_epochs}; n_train={n_train}; probe_seed={A.probe_seed}")
    print(f"Branches={[b['name'] for b in branches]}", flush=True)
    t0 = time.time()
    data = dict(config=cfg, branches_spec=copy.deepcopy(branches), prep=[], branches=[], hybrids={})

    for sd in seeds:
        P = prep_epochs[sd]
        if total_epochs <= P:
            raise ValueError(f"total_epochs {total_epochs} must exceed prep {P}")
        print("\n" + "=" * 80)
        print(f"Seed {sd}: prepare bound query-first answer-only to P={P}")
        print("=" * 80, flush=True)
        tied_state, opt_state, prep_tok, prep_recs = train_prep(sd, P, n_train, cfg, dev, cm_std, cm_blk, probes)
        data["prep"].extend(prep_recs)
        if A.save_checkpoints:
            torch.save(tied_state, ckpt_dir / f"seed{sd}_prep_tied.pt")
        data["hybrids"].setdefault(str(sd), {})
        for br in branches:
            print(f"\n  Branch {br['name']}", flush=True)
            recs, hybrids, final_state = train_branch(sd, br, P, total_epochs, n_train, cfg, dev,
                                                      cm_std, cm_blk, probes, tied_state,
                                                      opt_state, prep_tok)
            data["branches"].extend(recs)
            if hybrids is not None:
                data["hybrids"][str(sd)][br["name"]] = hybrids
            if A.save_checkpoints:
                torch.save(final_state, ckpt_dir / f"seed{sd}_{br['name']}_final.pt")

    data["summary"] = summarize(data)
    outp = out_dir / "results.json"
    with open(outp, "w") as f:
        json.dump(data, f, indent=2)
    make_figure(data, A.fig)
    note = ws / "notes" / "019b_embedding_role_decomposition_result.md"
    write_note(data, note)

    print("\nSUMMARY")
    for br, sm in data["summary"]["by_branch"].items():
        print(f"  {br:<38s} train={sm['train_strong_final']}/{sm['n']} held={sm['abs_held_strong_final']}/{sm['n']} "
              f"held4={sm['final_held_top4']:.3f} hB={sm['final_held_b']:+.3f} hL2={sm['final_held_input_l2']:.3f}")
    print(json.dumps(dict(status="STEP019B_DONE", out=str(outp), figure=A.fig,
                          note=str(note), elapsed=round(time.time() - t0, 1))))


if __name__ == "__main__":
    main()
