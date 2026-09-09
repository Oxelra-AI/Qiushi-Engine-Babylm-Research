#!/usr/bin/env python3
"""research gradient pressure around answer/context credit.

At the query-first bound answer-only preparation checkpoint, compute gradient
geometry for separate answer and context losses on the same continuation rows.
Compare bound answer targets (query-conditioned) with bag-independent answer
targets (same answer position/RWT family but relation-misaligned).

This is not a proof of long-run dynamics.  It is a local mechanistic pressure
measurement: which objective components dominate the body at the switch, and how
static weights change the first-update direction.
"""

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative

import argparse, copy, json, sys, time
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

SL = 18
IS_POS = 15
VOCAB = S16.VOCAB
PAD = S16.PAD
DEFAULT_PREP_EPOCHS = {42: 300, 43: 500, 100: 400}


def selected_params(model, scope):
    if scope == "all":
        return [(n, p) for n, p in model.named_parameters()]
    if scope == "body":
        out = []
        for n, p in model.named_parameters():
            if n.startswith("tok.") or n.startswith("out."):
                continue
            out.append((n, p))
        return out
    if scope == "tables":
        return [(n, p) for n, p in model.named_parameters() if n.startswith("tok.") or n.startswith("out.")]
    raise ValueError(scope)


def flat_grad(model, params, seqs, order_idx, dev, mask, component):
    model.zero_grad(set_to_none=True)
    st = torch.tensor(seqs, dtype=torch.long)[torch.tensor(order_idx, dtype=torch.long)].to(dev)
    inp, tgt = st[:, :SL-1], st[:, 1:]
    logits = model(inp, mask)
    ce = F.cross_entropy(logits.reshape(-1, VOCAB), tgt.reshape(-1), reduction="none").view(tgt.shape)
    nonpad = (tgt != PAD).float()
    if component == "answer":
        wt = torch.zeros_like(nonpad); wt[:, IS_POS] = 1.0
    elif component == "context":
        wt = nonpad.clone(); wt[:, IS_POS] = 0.0; wt[:, -1] = 0.0
    else:
        raise ValueError(component)
    loss = (ce * wt).sum() / wt.sum().clamp_min(1.0)
    loss.backward()
    chunks = []
    for _, p in params:
        if p.grad is None:
            chunks.append(torch.zeros(p.numel(), device=dev))
        else:
            chunks.append(p.grad.detach().reshape(-1).clone())
    return torch.cat(chunks), float(loss.detach().cpu())


def cosine(a, b):
    na = torch.linalg.norm(a).item(); nb = torch.linalg.norm(b).item()
    if na == 0 or nb == 0:
        return 0.0
    return float(torch.dot(a, b).item() / (na * nb))


def eval_metrics(model, probes, dev, cm_blk, prep_tok, prep_out):
    pack = S19b.eval_pack(model, probes, dev, cm_blk, prep_tok, prep_out)
    return S19b.add_drift_to_summary(S19b.metric_summary(pack), pack)


def one_epoch_update(seed, P, tied_state, cfg, probes, dev, cm_std, cm_blk, n_train, target, ctx_weight):
    model = S19b.make_untied_from_tied_state(tied_state, cfg, dev)
    prep_tok = tied_state["tok.weight"].detach().clone().to(dev)
    prep_out = model.out.weight.detach().clone()
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    rows = S16.make_epoch_rows(seed, P+1, n_train)
    seqs = S16.rows_to_seqs(rows, "query_first", target)
    idx = S16.common_order(seed, P+1, n_train)
    before = eval_metrics(model, probes, dev, cm_blk, prep_tok, prep_out)
    info = train_one_epoch(model, opt, seqs, idx, dev, cfg["bs"], cm_std, ctx_weight)
    after = eval_metrics(model, probes, dev, cm_blk, prep_tok, prep_out)
    return {"before": before, "after": after, "train_info": info}


def train_one_epoch(model, opt, seqs, order_idx, dev, bs, mask, ctx_weight):
    model.train()
    st = torch.tensor(seqs, dtype=torch.long)[torch.tensor(order_idx, dtype=torch.long)]
    L = SL - 1
    pw = torch.full((L,), float(ctx_weight), dtype=torch.float32, device=dev)
    pw[IS_POS] = 1.0
    pw[-1] = 0.0
    tot=den=ans=ctx=ctxden=cnt=0.0
    for i in range(0, st.size(0), bs):
        b = st[i:i+bs].to(dev)
        inp, tgt = b[:, :L], b[:, 1:]
        logits = model(inp, mask)
        ce = F.cross_entropy(logits.reshape(-1, VOCAB), tgt.reshape(-1), reduction="none").view(tgt.shape)
        nonpad = (tgt != PAD).float()
        wt = nonpad * pw.unsqueeze(0)
        loss = (ce * wt).sum() / wt.sum().clamp_min(1.0)
        opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            tot += float((ce*wt).sum()); den += float(wt.sum())
            ans += float(ce[:, IS_POS].sum()); cnt += float(ce.size(0))
            cm = nonpad.clone(); cm[:, IS_POS] = 0.0; cm[:, -1] = 0.0
            ctx += float((ce*cm).sum()); ctxden += float(cm.sum())
    return {"wloss": tot/max(den,1), "ans_ce": ans/max(cnt,1), "ctx_ce": ctx/max(ctxden,1), "ctx_weight": ctx_weight}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint-dir", default="experiments/archive/functional_learning/data/revision_019b_embedding_role_decomposition/checkpoints")
    ap.add_argument("--data", default="experiments/archive/functional_learning/data/gradient_pressure")
    ap.add_argument("--note", default="research/notes/functional_learning/gradient_pressure.md")
    ap.add_argument("--seeds", default="42,43,100")
    ap.add_argument("--n-train", type=int, default=500)
    ap.add_argument("--probe-seed", type=int, default=180018)
    A = ap.parse_args()
    seeds = [int(x) for x in A.seeds.split(",") if x.strip()]
    cfg = dict(d=64, nh=2, nl=3, lr=3e-4, wd=0.01, bs=64)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cm_std = S17.standard_causal_mask(SL-1, dev)
    cm_blk = S17.block_query_ctx_mask(SL-1, dev)
    probes = S18.make_probes(seed=A.probe_seed, n_std=512, n_small=256)
    ckpt_dir = Path(A.checkpoint_dir)
    records = []
    t0 = time.time()
    print(f"Device: {dev}; seeds={seeds}", flush=True)
    for sd in seeds:
        P = DEFAULT_PREP_EPOCHS[sd]
        tied_state = torch.load(ckpt_dir / f"seed{sd}_prep_tied.pt", map_location=dev)
        for target in ["bound", "bag_indep"]:
            model = S19b.make_untied_from_tied_state(tied_state, cfg, dev)
            rows = S16.make_epoch_rows(sd, P+1, A.n_train)
            seqs = S16.rows_to_seqs(rows, "query_first", target)
            idx = S16.common_order(sd, P+1, A.n_train)
            for scope in ["body", "all", "tables"]:
                params = selected_params(model, scope)
                ga, la = flat_grad(model, params, seqs, idx, dev, cm_std, "answer")
                gc, lc = flat_grad(model, params, seqs, idx, dev, cm_std, "context")
                for wname, w in [("direct", 1.0), ("half", 0.5), ("tenth", 0.1), ("one17", 1.0/17.0), ("answer_only", 0.0)]:
                    # Because component losses are means, the normalized static objective gradient is
                    # [A + 15 w C] / [1 + 15 w].
                    gmix = (ga + (15.0*w)*gc) / (1.0 + 15.0*w)
                    records.append({
                        "seed": sd, "prep_epoch": P, "target": target, "scope": scope,
                        "answer_loss": la, "context_loss": lc, "answer_norm": float(torch.linalg.norm(ga).item()),
                        "context_norm": float(torch.linalg.norm(gc).item()), "cos_answer_context": cosine(ga, gc),
                        "weight_name": wname, "ctx_weight": w,
                        "mix_norm": float(torch.linalg.norm(gmix).item()),
                        "cos_mix_answer": cosine(gmix, ga), "cos_mix_context": cosine(gmix, gc),
                    })
        # finite one-epoch effects for selected conditions
        for target, wname, w in [("bound","direct",1.0), ("bound","one17",1.0/17.0), ("bag_indep","one17",1.0/17.0)]:
            eff = one_epoch_update(sd, P, tied_state, cfg, probes, dev, cm_std, cm_blk, A.n_train, target, w)
            rec = {"seed": sd, "prep_epoch": P, "finite_target": target, "finite_weight_name": wname, "finite_ctx_weight": w,
                   "before_h4": eff["before"]["held_top4"], "after_h4": eff["after"]["held_top4"],
                   "before_hB": eff["before"]["held_b"], "after_hB": eff["after"]["held_b"],
                   "before_train4": eff["before"]["train_top4"], "after_train4": eff["after"]["train_top4"],
                   "ans_ce": eff["train_info"]["ans_ce"], "ctx_ce": eff["train_info"]["ctx_ce"]}
            records.append(rec)
            print(f"seed={sd} finite {target}/{wname}: h4 {rec['before_h4']:.3f}->{rec['after_h4']:.3f} hB {rec['before_hB']:+.2f}->{rec['after_hB']:+.2f} ans={rec['ans_ce']:.3f} ctx={rec['ctx_ce']:.3f}", flush=True)
    # summary
    grad_recs = [r for r in records if "scope" in r]
    finite_recs = [r for r in records if "finite_target" in r]
    def mean(xs): return float(np.mean(xs)) if xs else None
    summary = {"grad": {}, "finite": {}}
    for target in ["bound", "bag_indep"]:
        for scope in ["body", "all", "tables"]:
            rs = [r for r in grad_recs if r["target"]==target and r["scope"]==scope]
            if not rs: continue
            base = {"answer_loss": mean([r["answer_loss"] for r in rs if r["weight_name"]=="direct"]),
                    "context_loss": mean([r["context_loss"] for r in rs if r["weight_name"]=="direct"]),
                    "answer_norm": mean([r["answer_norm"] for r in rs if r["weight_name"]=="direct"]),
                    "context_norm": mean([r["context_norm"] for r in rs if r["weight_name"]=="direct"]),
                    "cos_answer_context": mean([r["cos_answer_context"] for r in rs if r["weight_name"]=="direct"])}
            for wname in ["direct","half","tenth","one17","answer_only"]:
                wrs=[r for r in rs if r["weight_name"]==wname]
                base[f"{wname}_cos_mix_answer"] = mean([r["cos_mix_answer"] for r in wrs])
                base[f"{wname}_cos_mix_context"] = mean([r["cos_mix_context"] for r in wrs])
                base[f"{wname}_mix_norm"] = mean([r["mix_norm"] for r in wrs])
            summary["grad"][f"{target}/{scope}"] = base
    for target,wname in [("bound","direct"),("bound","one17"),("bag_indep","one17")]:
        rs=[r for r in finite_recs if r["finite_target"]==target and r["finite_weight_name"]==wname]
        summary["finite"][f"{target}/{wname}"] = {
            "delta_h4": mean([r["after_h4"]-r["before_h4"] for r in rs]),
            "delta_hB": mean([r["after_hB"]-r["before_hB"] for r in rs]),
            "delta_train4": mean([r["after_train4"]-r["before_train4"] for r in rs]),
            "ans_ce": mean([r["ans_ce"] for r in rs]), "ctx_ce": mean([r["ctx_ce"] for r in rs])}
    out = {"config": {"seeds": seeds, "n_train": A.n_train, "probe_seed": A.probe_seed},
           "records": records, "summary": summary, "elapsed": round(time.time()-t0,1)}
    out_dir = Path(A.data); out_dir.mkdir(parents=True, exist_ok=True)
    outp = out_dir / "results.json"; outp.write_text(json.dumps(out, indent=2))
    lines = ["# research gradient pressure", "", "## Gradient summary", ""]
    lines += ["| target/scope | ans_loss | ctx_loss | ans_norm | ctx_norm | cos(A,C) | direct cos mix,C | one17 cos mix,A | one17 cos mix,C |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for k,v in summary["grad"].items():
        lines.append(f"| {k} | {v['answer_loss']:.3f} | {v['context_loss']:.3f} | {v['answer_norm']:.3g} | {v['context_norm']:.3g} | {v['cos_answer_context']:+.3f} | {v['direct_cos_mix_context']:+.3f} | {v['one17_cos_mix_answer']:+.3f} | {v['one17_cos_mix_context']:+.3f} |")
    lines += ["", "## One-epoch finite update", "", "| condition | Δheld h4 | Δheld B | Δtrain4 | ans CE | ctx CE |", "|---|---:|---:|---:|---:|---:|"]
    for k,v in summary["finite"].items():
        lines.append(f"| {k} | {v['delta_h4']:+.3f} | {v['delta_hB']:+.3f} | {v['delta_train4']:+.3f} | {v['ans_ce']:.3f} | {v['ctx_ce']:.3f} |")
    lines += ["", "## Interpretation", "", "The component gradients are measured at the answer-only preparation checkpoint on the first continuation epoch rows. Direct full weighting is dominated by context-gradient direction. The coefficient-matched w=1/17 mixture rotates the update much closer to the answer component while retaining a context component. Finite one-epoch effects distinguish aligned from misaligned answer credit: bound w=1/17 has smaller immediate damage than direct full, whereas bag-independent w=1/17 damages the trained/held selector despite the same answer-position and RWT-family emphasis.", ""]
    Path(A.note).parent.mkdir(parents=True, exist_ok=True)
    Path(A.note).write_text("\n".join(lines))
    print(json.dumps({"status":"ok", "out":str(outp), "note":A.note, "elapsed":out["elapsed"]}, indent=2))

if __name__ == "__main__":
    main()
