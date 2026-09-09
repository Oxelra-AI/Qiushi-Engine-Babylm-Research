#!/usr/bin/env python3
"""research: separate context-dose reduction from relation-compatible answer credit.

Starts from the same query-first bound answer-only preparation checkpoints.

Controls:
  1. context_only_lr_half: no answer loss; context positions only; lr is half of
     the direct-full lr, matching the reduction of the normalized context
     coefficient from 15/16 to 15/32 in the static w=1/17 objective.  If this
     preserves transfer while learning context, reduced context step size alone
     could explain Step021b.
  2. slot0_static_1over17: same nominal A/C coefficients as Step021b's bound
     static w=1/17, but answer target is the first context attribute, a visible
     deterministic target unrelated to the queried entity except when qi=0.
     This supplies learnable answer-position RWT pressure without maintaining
     the query-conditioned relation.

Existing comparator: Step021b bound static w=1/17, already saved in
data/revision_021b_credit_matched/results.json.
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
import calibration_trajectory as S21

SL = 18
IS_POS = 15
VOCAB = S16.VOCAB
PAD = S16.PAD
RWT = S16.RWT
DEFAULT_PREP_EPOCHS = {42: 300, 43: 500, 100: 400}

ARMS = [
    dict(name="context_only_lr_half", target="bound", kind="context_only", lr=1.5e-4,
         epochs=500, ctx_weight=1.0),
    dict(name="slot0_static_1over17", target="slot0", kind="static", lr=3e-4,
         epochs=500, ctx_weight=1.0/17.0),
]
ARM_INDEX = {a["name"]: i for i, a in enumerate(ARMS)}


def custom_seqs(rows, target):
    seqs = []
    for ce, ca, qi, bag_ti in rows:
        if target == "bound":
            tgt = RWT(ca[qi])
        elif target == "slot0":
            tgt = RWT(ca[0])
        else:
            raise ValueError(target)
        seqs.append(S16.mk_seq("query_first", ce, ca, qi, tgt))
    return seqs


def train_epoch(model, opt, seqs, order_idx, dev, bs, mask, kind, ctx_weight):
    model.train()
    st = torch.tensor(seqs, dtype=torch.long)[torch.tensor(order_idx, dtype=torch.long)]
    L = SL - 1
    ans_s=ctx_s=tot=den=ctx_den=0.0; cnt=0
    for i in range(0, st.size(0), bs):
        b = st[i:i+bs].to(dev)
        inp, tgt = b[:, :L], b[:, 1:]
        logits = model(inp, mask)
        ce = F.cross_entropy(logits.reshape(-1, VOCAB), tgt.reshape(-1), reduction="none").view(tgt.shape)
        nonpad = (tgt != PAD).float()
        if kind == "context_only":
            wt = nonpad.clone(); wt[:, IS_POS] = 0.0; wt[:, -1] = 0.0
        elif kind == "static":
            pw = torch.full((L,), float(ctx_weight), dtype=torch.float32, device=dev)
            pw[IS_POS] = 1.0; pw[-1] = 0.0
            wt = nonpad * pw.unsqueeze(0)
        else:
            raise ValueError(kind)
        loss = (ce * wt).sum() / wt.sum().clamp_min(1.0)
        opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            tot += float((ce*wt).sum()); den += float(wt.sum())
            ans_s += float(ce[:, IS_POS].sum()); cnt += int(ce.size(0))
            cm = nonpad.clone(); cm[:, IS_POS]=0.0; cm[:, -1]=0.0
            ctx_s += float((ce*cm).sum()); ctx_den += float(cm.sum())
    return dict(wloss=round(tot/max(den,1),6), ans_ce=round(ans_s/max(cnt,1),6),
                ctx_ce=round(ctx_s/max(ctx_den,1),6), ctx_weight=round(float(ctx_weight),6))


def eval_schedule(total):
    s = {0,1,5,10,25,50,75,100,125,150,200,250,300,350,400,450,500}
    return sorted(e for e in s if 0 <= e <= total)


def run_arm(seed, P, arm, tied_state, prep_tok, cfg, probes, dev, cm_std, cm_blk, n_train):
    torch.manual_seed(seed + 230000 + 41*ARM_INDEX[arm["name"]])
    np.random.seed(seed + 230000 + ARM_INDEX[arm["name"]])
    model = S19b.make_untied_from_tied_state(tied_state, cfg, dev)
    prep_out = model.out.weight.detach().clone()
    opt = torch.optim.AdamW(model.parameters(), lr=arm["lr"], weight_decay=cfg["wd"])
    sched = set(eval_schedule(arm["epochs"]))
    recs=[]; ts=time.time()
    def record(be, info):
        pack = S19b.eval_pack(model, probes, dev, cm_blk, prep_tok, prep_out)
        m = S19b.add_drift_to_summary(S19b.metric_summary(pack), pack)
        rec = dict(seed=int(seed), arm=arm["name"], target=arm["target"], kind=arm["kind"],
                   lr=arm["lr"], prep_epoch=int(P), branch_epoch=int(be), epoch=int(P+be),
                   elapsed=round(time.time()-ts,2))
        rec.update(info); rec.update(pack); recs.append(rec)
        print(f"  {arm['name']:<24s} be={be:3d} tr4={m['train_top4']:.3f} h4={m['held_top4']:.3f} "
              f"hB={m['held_b']:+.3f} hSel={m['held_sel']:+.3f} blk4={m['blocked_train_top4']:.3f} "
              f"ans={info.get('ans_ce',0):.3f} ctx={info.get('ctx_ce',0):.3f}", flush=True)
    record(0, dict(wloss=0.0, ans_ce=0.0, ctx_ce=0.0, ctx_weight=0.0))
    for be in range(1, arm["epochs"]+1):
        ep = P + be
        rows = S16.make_epoch_rows(seed, ep, n_train)
        seqs = custom_seqs(rows, arm["target"])
        idx = S16.common_order(seed, ep, n_train)
        info = train_epoch(model, opt, seqs, idx, dev, cfg["bs"], cm_std, arm["kind"], arm["ctx_weight"])
        if be in sched:
            record(be, info)
    return recs


def summarize(records, arms):
    out={}
    for a in arms:
        name=a["name"]
        rs=[r for r in records if r["arm"]==name]
        if not rs: continue
        max_be=max(int(r["branch_epoch"]) for r in rs)
        starts=[r for r in rs if int(r["branch_epoch"])==0]
        finals=[r for r in rs if int(r["branch_epoch"])==max_be]
        def mean_sm(arr,k):
            vals=[S19b.add_drift_to_summary(S19b.metric_summary(r), r)[k] for r in arr]
            return round(float(np.mean(vals)),6) if vals else None
        out[name]=dict(n=len(finals), max_be=max_be, lr=a["lr"], target=a["target"], kind=a["kind"])
        for tag, arr in [("start",starts),("final",finals)]:
            for k in ["train_top4","held_top4","held_b","held_sel","blocked_train_top4"]:
                out[name][f"{tag}_{k}"]=mean_sm(arr,k)
        out[name]["final_ans_ce"]=round(float(np.mean([r.get("ans_ce",0) for r in finals])),6)
        out[name]["final_ctx_ce"]=round(float(np.mean([r.get("ctx_ce",0) for r in finals])),6)
    return out


def write_note(data, path):
    lines=["# research alignment versus context-dose controls", "", "## Final means", "", "| arm | target/kind | lr | final h4 | final hB | final hSel | train4 | blocked4 | ctx CE | ans CE |", "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name,s in data["summary"].items():
        lines.append(f"| {name} | {s['target']}/{s['kind']} | {s['lr']:.1e} | {s['final_held_top4']:.3f} | {s['final_held_b']:+.3f} | {s['final_held_sel']:+.3f} | {s['final_train_top4']:.3f} | {s['final_blocked_train_top4']:.3f} | {s['final_ctx_ce']:.3f} | {s['final_ans_ce']:.3f} |")
    lines += ["", "## Per-seed final h4/hB", "", "| seed | context_only_lr_half | slot0_static_1over17 |", "|---:|---:|---:|"]
    for sd in data["config"]["seeds"]:
        row=[str(sd)]
        for name in ["context_only_lr_half","slot0_static_1over17"]:
            rs=sorted([r for r in data["records"] if int(r["seed"])==int(sd) and r["arm"]==name], key=lambda r:int(r["branch_epoch"]))
            if rs:
                m=S19b.add_drift_to_summary(S19b.metric_summary(rs[-1]), rs[-1])
                row.append(f"{m['held_top4']:.3f}/{m['held_b']:+.2f}")
            else:
                row.append("")
        lines.append("| " + " | ".join(row) + " |")
    lines += ["", "## Interpretation scaffold", "", "Compare these endpoints to Step021b's bound static w=1/17 arm (mean held_top4 0.694, held_B +7.548, context CE 1.238). If context_only_lr_half collapses, then reducing context-update scale without answer credit is insufficient. If slot0_static_1over17 collapses while learning its visible deterministic answer distribution, then learnable answer/RWT pressure at the matched coefficient is insufficient unless the answer remains compatible with the query-conditioned binding relation.", ""]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines))

class NpEncoder(json.JSONEncoder):
    def default(self,obj):
        if isinstance(obj,(np.integer,)): return int(obj)
        if isinstance(obj,(np.floating,)): return float(obj)
        if isinstance(obj,np.ndarray): return obj.tolist()
        return super().default(obj)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--checkpoint-dir", default="experiments/archive/functional_learning/data/revision_019b_embedding_role_decomposition/checkpoints")
    ap.add_argument("--data", default="experiments/archive/functional_learning/data/alignment_vs_dose")
    ap.add_argument("--note", default="research/notes/functional_learning/alignment_vs_dose.md")
    ap.add_argument("--seeds", default="42,43,100")
    ap.add_argument("--n-train", type=int, default=500)
    ap.add_argument("--probe-seed", type=int, default=180018)
    ap.add_argument("--smoke", action="store_true")
    A=ap.parse_args()
    if A.smoke:
        A.seeds="100"; A.n_train=64; n_std,n_small=96,48
        arms=[copy.deepcopy(a) for a in ARMS]
        for a in arms: a["epochs"]=10
    else:
        n_std,n_small=512,256
        arms=[copy.deepcopy(a) for a in ARMS]
    seeds=[int(x) for x in A.seeds.split(",") if x.strip()]
    cfg=dict(d=64, nh=2, nl=3, lr=3e-4, wd=0.01, bs=64)
    dev=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cm_std=S17.standard_causal_mask(SL-1, dev)
    cm_blk=S17.block_query_ctx_mask(SL-1, dev)
    probes=S18.make_probes(seed=A.probe_seed, n_std=n_std, n_small=n_small)
    ckpt_dir=Path(A.checkpoint_dir)
    data=dict(config=dict(seeds=seeds,n_train=A.n_train,probe_seed=A.probe_seed,prep_epochs={str(k):DEFAULT_PREP_EPOCHS[k] for k in seeds}), arms_spec=arms, records=[], summary={})
    t0=time.time()
    print(f"Device: {dev}; seeds={seeds}; arms={[a['name'] for a in arms]}", flush=True)
    for sd in seeds:
        P=DEFAULT_PREP_EPOCHS[sd]
        tied_state=torch.load(ckpt_dir / f"seed{sd}_prep_tied.pt", map_location=dev)
        prep_tok=tied_state["tok.weight"].detach().clone().to(dev)
        print("\n"+"="*80); print(f"Seed {sd}; prep P={P}"); print("="*80, flush=True)
        for arm in arms:
            data["records"].extend(run_arm(sd,P,arm,tied_state,prep_tok,cfg,probes,dev,cm_std,cm_blk,A.n_train))
    data["summary"]=summarize(data["records"], arms)
    out_dir=Path(A.data); out_dir.mkdir(parents=True, exist_ok=True)
    outp=out_dir/"results.json"; outp.write_text(json.dumps(data, indent=2, cls=NpEncoder))
    write_note(data, A.note)
    print(json.dumps({"status":"ok","out":str(outp),"note":A.note,"elapsed":round(time.time()-t0,1)}, indent=2))

if __name__ == "__main__":
    main()
