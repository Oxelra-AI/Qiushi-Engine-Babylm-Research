#!/usr/bin/env python3
"""research-style component ablation from query-first bound checkpoints.

Step019b ruled out held input-row drift as the immediate explanation of held-symbol
binding loss after switching from answer-only preparation to full next-token training.
This script asks where the destructive full-objective update must act: output readout,
contextual body/pos/LN, input table, or the objective itself.

It loads the Step019b saved preparation checkpoints, converts them to tied/untied
models as needed, and runs short continuations (default 25 epochs) with selective
trainable parameter sets.  The experiment is intentionally small: it focuses on the
first epoch and early collapse/recovery rather than long endpoint training.
"""

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative

import argparse, copy, json, math, sys, time
from pathlib import Path

import numpy as np
import torch

SCRIPT_DIR = _public_path('experiments/archive/functional_learning/scripts')
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
import loss_allocation_binding as Base
import query_first_binding as S16
import binding_branching as S17
import budget_matched_full_objective as S18
import revision_019b_embedding_role_decomposition as S19b

VOCAB, SL = Base.VOCAB, 18
DEFAULT_PREP_EPOCHS = {42: 300, 43: 500, 100: 400}

BRANCHES = [
    dict(name="tied_all_full", model="tied", mode="full", train="all"),
    dict(name="untied_all_full", model="untied", mode="full", train="all"),
    dict(name="untied_all_answer_only", model="untied", mode="answer_only", train="all"),
    dict(name="untied_all_context_only", model="untied", mode="context_only", train="all"),
    dict(name="untied_body_only_full", model="untied", mode="full", train="body_only"),
    dict(name="untied_body_only_answer_only", model="untied", mode="answer_only", train="body_only"),
    dict(name="untied_body_only_context_only", model="untied", mode="context_only", train="body_only"),
    dict(name="untied_out_only_full", model="untied", mode="full", train="out_only"),
    dict(name="untied_input_only_full", model="untied", mode="full", train="input_only"),
]


def parse_ints(s):
    return [int(x) for x in s.split(',') if x.strip()]


def choose_branches(s):
    idx = {b["name"]: b for b in BRANCHES}
    if s == "all":
        return copy.deepcopy(BRANCHES)
    names = [x.strip() for x in s.split(',') if x.strip()]
    missing = [x for x in names if x not in idx]
    if missing:
        raise ValueError(missing)
    return [copy.deepcopy(idx[x]) for x in names]


def get_metric_pack(model, probes, dev, cm_blk, prep_tok):
    pack = S19b.eval_pack(model, probes, dev, cm_blk, prep_tok, prep_tok if not hasattr(model, "out") else prep_tok)
    return pack


def summarize_pack(pack):
    m = S19b.metric_summary(pack)
    m = S19b.add_drift_to_summary(m, pack)
    return m


def set_trainable(model, spec):
    for p in model.parameters():
        p.requires_grad_(False)
    train = spec["train"]
    if train == "all":
        for p in model.parameters():
            p.requires_grad_(True)
    elif train == "body_only":
        # Keep input table and output classifier fixed; allow position embedding,
        # transformer blocks, and layer norm to adapt to full-objective losses.
        for name, p in model.named_parameters():
            if name.startswith("pos.") or name.startswith("blks.") or name.startswith("ln."):
                p.requires_grad_(True)
    elif train == "out_only":
        for name, p in model.named_parameters():
            if name.startswith("out."):
                p.requires_grad_(True)
    elif train == "input_only":
        for name, p in model.named_parameters():
            if name.startswith("tok."):
                p.requires_grad_(True)
    else:
        raise ValueError(train)
    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        raise ValueError(f"no trainable params for {spec}")
    return params


def make_model(spec, tied_state, cfg, dev):
    if spec["model"] == "tied":
        m = S19b.make_tied(cfg, dev)
        m.load_state_dict(copy.deepcopy(tied_state))
    elif spec["model"] == "untied":
        m = S19b.make_untied_from_tied_state(tied_state, cfg, dev)
    else:
        raise ValueError(spec["model"])
    return m


def eval_schedule(max_epochs):
    s = {0, 1, 5, 10, 25, max_epochs}
    return sorted(e for e in s if 0 <= e <= max_epochs)


def run_branch(seed, P, spec, tied_state, prep_tok, cfg, probes, dev, cm_std, cm_blk, n_train, epochs):
    torch.manual_seed(seed + 202000 + 31 * BRANCHES.index(next(b for b in BRANCHES if b['name'] == spec['name'])))
    np.random.seed(seed + 202000)
    model = make_model(spec, tied_state, cfg, dev)
    params = set_trainable(model, spec)
    opt = torch.optim.AdamW(params, lr=cfg["lr"], weight_decay=cfg["wd"])
    recs = []
    sched = set(eval_schedule(epochs))
    ts = time.time()

    def add(be, info):
        pack = get_metric_pack(model, probes, dev, cm_blk, prep_tok)
        sm = summarize_pack(pack)
        rec = dict(seed=seed, branch=spec["name"], model=spec["model"], mode=spec["mode"],
                   train=spec["train"], prep_epoch=P, branch_epoch=be, epoch=P + be,
                   elapsed=round(time.time()-ts, 2), **info)
        rec.update(pack)
        recs.append(rec)
        print(f"  {spec['name']:<28s} be={be:3d} tr4={sm['train_top4']:.3f} h4={sm['held_top4']:.3f} "
              f"hB={sm['held_b']:+.3f} hSel={sm['held_sel']:+.3f} blk4={sm['blocked_train_top4']:.3f} "
              f"hL2={sm.get('held_ent_input_l2', 0.0):.3f}", flush=True)

    add(0, dict(wloss=0.0, ans_ce=0.0, ctx_ce=0.0))
    for be in range(1, epochs + 1):
        ep = P + be
        rows = S16.make_epoch_rows(seed, ep, n_train)
        seqs = S16.rows_to_seqs(rows, "query_first", "bound")
        idx = S16.common_order(seed, ep, n_train)
        info = S19b.train_one_epoch(model, opt, seqs, idx, dev, spec["mode"], cfg["bs"], cm_std)
        if be in sched:
            add(be, info)
    return recs


def metrics(rec):
    return summarize_pack(rec)


def analyze(data):
    out = {"by_seed": {}, "by_branch": {}}
    branches = [b["name"] for b in data["branches_spec"]]
    for sd in data["config"]["seeds"]:
        out["by_seed"][str(sd)] = {}
        for br in branches:
            rs = sorted([r for r in data["records"] if int(r["seed"]) == int(sd) and r["branch"] == br], key=lambda r: int(r["branch_epoch"]))
            if not rs:
                continue
            start = metrics(rs[0]); final = metrics(rs[-1])
            r1 = next((metrics(r) for r in rs if int(r["branch_epoch"]) == 1), None)
            r25 = next((metrics(r) for r in rs if int(r["branch_epoch"]) == min(25, data["config"]["epochs"])), None)
            out["by_seed"][str(sd)][br] = dict(start=start, be1=r1, be25=r25, final=final,
                delta1={k: None if r1 is None else r1[k]-start[k] for k in ["train_top4","held_top4","held_b","held_sel"]},
                delta_final={k: final[k]-start[k] for k in ["train_top4","held_top4","held_b","held_sel"]})
    for br in branches:
        finals = [out["by_seed"][str(sd)][br]["final"] for sd in data["config"]["seeds"] if br in out["by_seed"].get(str(sd), {})]
        ones = [out["by_seed"][str(sd)][br]["be1"] for sd in data["config"]["seeds"] if br in out["by_seed"].get(str(sd), {}) and out["by_seed"][str(sd)][br]["be1"] is not None]
        def mean_key(arr, k):
            return None if not arr else round(float(np.mean([x[k] for x in arr])), 6)
        out["by_branch"][br] = dict(n=len(finals),
            be1_train_top4=mean_key(ones, "train_top4"), be1_held_top4=mean_key(ones, "held_top4"), be1_held_b=mean_key(ones, "held_b"), be1_held_sel=mean_key(ones, "held_sel"),
            final_train_top4=mean_key(finals, "train_top4"), final_held_top4=mean_key(finals, "held_top4"), final_held_b=mean_key(finals, "held_b"), final_held_sel=mean_key(finals, "held_sel"), final_blocked_top4=mean_key(finals, "blocked_train_top4"), final_held_l2=mean_key(finals, "held_ent_input_l2"))
    return out


def f3(x):
    return "" if x is None else f"{x:.3f}"


def write_note(data, path):
    A = data["analysis"]
    lines = []
    lines.append("# research component ablation after full-objective switch")
    lines.append("")
    lines.append("This analysis starts from the saved query-first answer-only preparation checkpoints from Step019b and runs short continuations with selective trainable parameter sets. It asks whether early held-symbol binding loss is caused by the tied held input rows, the output head, the contextual body, or the switch of objective pressure itself.")
    lines.append("")
    lines.append("## Branch means")
    lines.append("")
    lines.append("| branch | n | be1 train4 | be1 held4 | be1 heldB | be1 heldSel | final train4 | final held4 | final heldB | final heldSel | final blocked train4 | final held L2 |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for br, sm in A["by_branch"].items():
        lines.append(f"| {br} | {sm['n']} | {f3(sm['be1_train_top4'])} | {f3(sm['be1_held_top4'])} | {f3(sm['be1_held_b'])} | {f3(sm['be1_held_sel'])} | {f3(sm['final_train_top4'])} | {f3(sm['final_held_top4'])} | {f3(sm['final_held_b'])} | {f3(sm['final_held_sel'])} | {f3(sm['final_blocked_top4'])} | {f3(sm['final_held_l2'])} |")
    lines.append("")
    lines.append("## Per-seed one-epoch and final deltas")
    lines.append("")
    lines.append("| seed | branch | start h4 | be1 h4 | Δ1 h4 | be1 hB | Δ1 hB | final h4 | Δfinal h4 | final hB | Δfinal hB | final train4 |")
    lines.append("|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for sd, byb in A["by_seed"].items():
        for br, item in byb.items():
            st, r1, fi = item["start"], item["be1"], item["final"]
            d1, df = item["delta1"], item["delta_final"]
            lines.append(f"| {sd} | {br} | {f3(st['held_top4'])} | {f3(r1['held_top4'] if r1 else None)} | {f3(d1['held_top4'])} | {f3(r1['held_b'] if r1 else None)} | {f3(d1['held_b'])} | {f3(fi['held_top4'])} | {f3(df['held_top4'])} | {f3(fi['held_b'])} | {f3(df['held_b'])} | {f3(fi['train_top4'])} |")
    lines.append("")
    lines.append("## Interpretation scaffold")
    lines.append("")
    lines.append("If `untied_out_only_full` collapses held behavior while the body and input embeddings are fixed, then output/readout adaptation is sufficient. If `untied_body_only_full` collapses held behavior while input and output tables are fixed, then contextual state specialization is sufficient. If `untied_input_only_full` collapses despite fixed body/output and held rows absent, then training-symbol input geometry is sufficient. `untied_all_answer_only` is the continuing-answer-pressure control; `untied_all_context_only` tests whether answer loss is needed to keep the selector alive. These are early-dynamics tests; long reacquisition under full objective is a separate phenomenon.")
    Path(path).write_text("\n".join(lines)+"\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint-dir", default="experiments/archive/functional_learning/data/revision_019b_embedding_role_decomposition/checkpoints")
    ap.add_argument("--data", default="experiments/archive/functional_learning/data/component_ablation_after_switch")
    ap.add_argument("--note", default="research/notes/functional_learning/component_ablation_after_switch.md")
    ap.add_argument("--seeds", default="42,43,100")
    ap.add_argument("--branches", default="all")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--n-train", type=int, default=500)
    ap.add_argument("--probe-seed", type=int, default=180018)
    ap.add_argument("--smoke", action="store_true")
    A = ap.parse_args()
    if A.smoke:
        A.seeds = "100"; A.branches = "untied_all_full,untied_body_only_full,untied_out_only_full"; A.epochs = 1; A.n_train = 64
        n_std, n_small = 96, 48
    else:
        n_std, n_small = 512, 256
    seeds = parse_ints(A.seeds)
    branches = choose_branches(A.branches)
    cfg = dict(d=64, nh=2, nl=3, lr=3e-4, wd=0.01, bs=64)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cm_std = S17.standard_causal_mask(SL - 1, dev)
    cm_blk = S17.block_query_ctx_mask(SL - 1, dev)
    probes = S18.make_probes(seed=A.probe_seed, n_std=n_std, n_small=n_small)
    out_dir = Path(A.data); out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = Path(A.checkpoint_dir)
    data = dict(config=dict(seeds=seeds, prep_epochs={str(k): DEFAULT_PREP_EPOCHS[k] for k in seeds}, epochs=A.epochs, n_train=A.n_train, probe_seed=A.probe_seed),
                branches_spec=branches, records=[])
    print(f"Device: {dev}; seeds={seeds}; epochs={A.epochs}; branches={[b['name'] for b in branches]}", flush=True)
    t0 = time.time()
    for sd in seeds:
        P = DEFAULT_PREP_EPOCHS[sd]
        ck = ckpt_dir / f"seed{sd}_prep_tied.pt"
        tied_state = torch.load(ck, map_location=dev)
        prep_tok = tied_state["tok.weight"].detach().clone().to(dev)
        print("\n" + "="*80)
        print(f"Seed {sd}; prep checkpoint {ck}; P={P}")
        print("="*80, flush=True)
        for spec in branches:
            recs = run_branch(sd, P, spec, tied_state, prep_tok, cfg, probes, dev, cm_std, cm_blk, A.n_train, A.epochs)
            data["records"].extend(recs)
    data["analysis"] = analyze(data)
    outp = out_dir / "results.json"
    outp.write_text(json.dumps(data, indent=2))
    write_note(data, A.note)
    print(json.dumps({"status":"ok", "out":str(outp), "note":A.note, "elapsed":round(time.time()-t0,1)}, indent=2))

if __name__ == "__main__":
    main()
