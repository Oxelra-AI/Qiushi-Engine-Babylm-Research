#!/usr/bin/env python3
"""research: Does preservation require relation-aligned answer credit?

Step021b showed that a static objective with the same nominal A:C coefficients as
alternating answer/full (context weight w=1/17) preserves most held-symbol binding
while learning context prediction.  This script tests whether that is merely answer
position / RWT rehearsal or whether the answer credit must remain aligned with the
query-conditioned binding relation.

Starting from the same query-first bound answer-only preparation checkpoints, we
continue training with the same context rows and the same answer-position marginal
RWT vocabulary, but make the answer target bag-independent (one random context
attribute rather than the queried entity's attribute).  Held entities are still
absent from continuation rows.  If high answer credit or temporal answer exposure
alone preserves transfer, bag-independent answer credit should preserve it.  If the
shared computation is maintained by relation-aligned answer gradients, these arms
should collapse query-conditioned binding despite matching nominal allocation or
interleaving structure.
"""

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative

import argparse, copy, json, sys, time
from pathlib import Path

import numpy as np
import torch

SCRIPT_DIR = _public_path('experiments/archive/functional_learning/scripts')
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import query_first_binding as S16
import binding_branching as S17
import budget_matched_full_objective as S18
import revision_019b_embedding_role_decomposition as S19b
import calibration_trajectory as S21

SL = 18
DEFAULT_PREP_EPOCHS = {42: 300, 43: 500, 100: 400}

ARMS = [
    dict(name="bag_static_1over17", target="bag_indep", phases=[
        dict(epochs=500, train="all", lr=3e-4, ctx_mode="fixed", ctx_weight=1.0/17.0),
    ]),
    dict(name="bag_interleaved_ans_full", target="bag_indep", phases=[
        dict(epochs=500, train="all", lr=3e-4, ctx_mode="interleaved"),
    ]),
]
ARM_INDEX = {a["name"]: i for i, a in enumerate(ARMS)}


def eval_schedule(total):
    s = {0, 1, 5, 10, 25, 50, 75, 100, 125, 150, 200, 250, 300, 350, 400, 450, 500}
    return sorted(e for e in s if 0 <= e <= total)


def run_arm(seed, P, arm, tied_state, prep_tok, cfg, probes, dev, cm_std, cm_blk, n_train):
    torch.manual_seed(seed + 220000 + 37 * ARM_INDEX.get(arm["name"], 0))
    np.random.seed(seed + 220000 + ARM_INDEX.get(arm["name"], 0))
    model = S19b.make_untied_from_tied_state(tied_state, cfg, dev)
    prep_out = model.out.weight.data.detach().clone()
    total_cont = sum(p["epochs"] for p in arm["phases"])
    sched = set(eval_schedule(total_cont))
    recs = []
    ts = time.time()

    def record(be, info, plabel=""):
        pack = S19b.eval_pack(model, probes, dev, cm_blk, prep_tok, prep_out)
        m = S19b.add_drift_to_summary(S19b.metric_summary(pack), pack)
        rec = dict(seed=int(seed), arm=arm["name"], target=arm["target"], prep_epoch=int(P),
                   branch_epoch=int(be), epoch=int(P+be), phase=plabel,
                   elapsed=round(time.time() - ts, 2))
        rec.update(info)
        rec.update(pack)
        recs.append(rec)
        print(f"  {arm['name']:<31s} be={be:3d} [{plabel:<14s}] "
              f"tr4={m['train_top4']:.3f} h4={m['held_top4']:.3f} "
              f"hB={m['held_b']:+.3f} hSel={m['held_sel']:+.3f} "
              f"blk4={m['blocked_train_top4']:.3f} ans={info.get('ans_ce',0):.3f} "
              f"ctx={info.get('ctx_ce',0):.3f} cw={info.get('ctx_weight','?')}", flush=True)

    record(0, dict(wloss=0.0, ans_ce=0.0, ctx_ce=0.0, ctx_weight=0.0), "prep")
    be = 0
    for ph in arm["phases"]:
        params = S21.set_trainable(model, ph["train"])
        opt = torch.optim.AdamW(params, lr=ph["lr"], weight_decay=cfg["wd"])
        plabel = S21.phase_label(ph)
        for ep_in_ph in range(1, ph["epochs"] + 1):
            be += 1
            ep = P + be
            cw = S21.ctx_weight_for_epoch(ph, ep_in_ph)
            rows = S16.make_epoch_rows(seed, ep, n_train)
            seqs = S16.rows_to_seqs(rows, "query_first", arm["target"])
            idx = S16.common_order(seed, ep, n_train)
            info = S21.train_one_epoch_weighted(model, opt, seqs, idx, dev, cfg["bs"], cm_std,
                                               answer_weight=1.0, context_weight=cw)
            info["ctx_weight"] = round(float(cw), 4)
            if be in sched:
                record(be, info, plabel)
    return recs


def summarize(records, arm_names):
    out = {}
    for name in arm_names:
        rs = [r for r in records if r["arm"] == name]
        if not rs:
            continue
        max_be = max(int(r["branch_epoch"]) for r in rs)
        starts = [r for r in rs if int(r["branch_epoch"]) == 0]
        finals = [r for r in rs if int(r["branch_epoch"]) == max_be]
        def mean_sm(arr, key):
            vals = []
            for r in arr:
                vals.append(S19b.add_drift_to_summary(S19b.metric_summary(r), r)[key])
            return round(float(np.mean(vals)), 6) if vals else None
        out[name] = dict(n=len(finals), max_be=max_be)
        for tag, arr in [("start", starts), ("final", finals)]:
            for key in ["train_top4", "held_top4", "held_b", "held_sel", "blocked_train_top4"]:
                out[name][f"{tag}_{key}"] = mean_sm(arr, key)
        out[name]["final_ans_ce"] = round(float(np.mean([r.get("ans_ce", 0) for r in finals])), 6)
        out[name]["final_ctx_ce"] = round(float(np.mean([r.get("ctx_ce", 0) for r in finals])), 6)
    return out


def write_note(data, path):
    lines = [
        "# research answer-credit alignment test", "",
        "Continuation starts from the same query-first bound answer-only preparation checkpoints used in research.",
        "The two arms keep the context rows and answer-position marginal RWT vocabulary, but train the answer target as bag-independent rather than the queried entity's attribute.",
        "Held entity tokens remain absent from all continuation rows.", "",
        "## Final means", "",
        "| arm | final h4 | final hB | final hSel | train4 | blocked4 | ctx CE | ans CE |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, s in data["summary"].items():
        lines.append(f"| {name} | {s['final_held_top4']:.3f} | {s['final_held_b']:+.3f} | {s['final_held_sel']:+.3f} | "
                     f"{s['final_train_top4']:.3f} | {s['final_blocked_train_top4']:.3f} | {s['final_ctx_ce']:.3f} | {s['final_ans_ce']:.3f} |")
    lines += ["", "## Per-seed trajectories", ""]
    for sd in data["config"]["seeds"]:
        lines.append(f"### Seed {sd}")
        for arm in data["arms_spec"]:
            name = arm["name"]
            rs = sorted([r for r in data["records"] if int(r["seed"]) == int(sd) and r["arm"] == name], key=lambda r:int(r["branch_epoch"]))
            if not rs:
                continue
            lines.append(f"**{name}**")
            for r in rs:
                m = S19b.add_drift_to_summary(S19b.metric_summary(r), r)
                lines.append(f"  be={int(r['branch_epoch']):3d} h4={m['held_top4']:.3f} hB={m['held_b']:+.3f} "
                             f"hSel={m['held_sel']:+.3f} tr4={m['train_top4']:.3f} ctx={r.get('ctx_ce',0):.3f} cw={r.get('ctx_weight','?')}")
            lines.append("")
    lines += [
        "## Interpretation", "",
        "If these bag-independent answer-credit arms lose binding while Step021b's bound static w=1/17 and bound interleaving preserve it, the preservation cannot be explained by answer-token exposure, RWT-family rehearsal, or temporal alternation alone. It depends on answer gradients that remain aligned with the query-conditioned relation. Conversely, if they preserve, the previous result would reduce to generic answer/RWT rehearsal or loss allocation without relation-specific credit.", "",
    ]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines))


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint-dir", default="experiments/archive/functional_learning/data/revision_019b_embedding_role_decomposition/checkpoints")
    ap.add_argument("--data", default="experiments/archive/functional_learning/data/answer_credit_alignment")
    ap.add_argument("--note", default="research/notes/functional_learning/answer_credit_alignment.md")
    ap.add_argument("--seeds", default="42,43,100")
    ap.add_argument("--arms", default="all")
    ap.add_argument("--total-epochs", type=int, default=500)
    ap.add_argument("--n-train", type=int, default=500)
    ap.add_argument("--probe-seed", type=int, default=180018)
    ap.add_argument("--smoke", action="store_true")
    A = ap.parse_args()
    if A.smoke:
        A.seeds = "100"; A.total_epochs = 10; A.n_train = 64; n_std, n_small = 96, 48
    else:
        n_std, n_small = 512, 256
    seeds = [int(x) for x in A.seeds.split(",") if x.strip()]
    if A.arms == "all":
        arms = [copy.deepcopy(a) for a in ARMS]
    else:
        want = {x.strip() for x in A.arms.split(",") if x.strip()}
        arms = [copy.deepcopy(a) for a in ARMS if a["name"] in want]
    for arm in arms:
        orig = sum(p["epochs"] for p in arm["phases"])
        if orig != A.total_epochs:
            scale = A.total_epochs / orig
            for p in arm["phases"]:
                p["epochs"] = max(1, int(round(p["epochs"] * scale)))
    cfg = dict(d=64, nh=2, nl=3, lr=3e-4, wd=0.01, bs=64)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cm_std = S17.standard_causal_mask(SL - 1, dev)
    cm_blk = S17.block_query_ctx_mask(SL - 1, dev)
    probes = S18.make_probes(seed=A.probe_seed, n_std=n_std, n_small=n_small)
    ckpt_dir = Path(A.checkpoint_dir)
    out_dir = Path(A.data); out_dir.mkdir(parents=True, exist_ok=True)
    data = dict(config=dict(seeds=seeds, prep_epochs={str(k): DEFAULT_PREP_EPOCHS[k] for k in seeds},
                            total_cont_epochs=A.total_epochs, n_train=A.n_train, probe_seed=A.probe_seed),
                arms_spec=[dict(name=a["name"], target=a["target"], phases=a["phases"]) for a in arms],
                records=[], summary={})
    print(f"Device: {dev}; seeds={seeds}; total_epochs={A.total_epochs}; arms={[a['name'] for a in arms]}", flush=True)
    t0 = time.time()
    for sd in seeds:
        P = DEFAULT_PREP_EPOCHS[sd]
        ck = ckpt_dir / f"seed{sd}_prep_tied.pt"
        if not ck.exists():
            print(f"SKIP seed {sd}: {ck} not found")
            continue
        tied_state = torch.load(ck, map_location=dev)
        prep_tok = tied_state["tok.weight"].detach().clone().to(dev)
        print("\n" + "="*80)
        print(f"Seed {sd}; prep P={P}")
        print("="*80, flush=True)
        for arm in arms:
            data["records"].extend(run_arm(sd, P, arm, tied_state, prep_tok, cfg, probes, dev, cm_std, cm_blk, A.n_train))
    data["summary"] = summarize(data["records"], [a["name"] for a in arms])
    outp = out_dir / "results.json"
    outp.write_text(json.dumps(data, indent=2, cls=NpEncoder))
    write_note(data, A.note)
    print(json.dumps({"status":"ok", "out":str(outp), "note":A.note, "elapsed":round(time.time()-t0,1)}, indent=2))

if __name__ == "__main__":
    main()
