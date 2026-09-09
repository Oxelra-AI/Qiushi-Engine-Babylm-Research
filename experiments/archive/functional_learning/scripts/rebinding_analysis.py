#!/usr/bin/env python3
"""research: Rebinding and trajectory analysis for research saved models.

This is not a new training ingredient. It re-reads research curves and final
saved models to test what reusable computation was actually acquired.

Key correction motivating this analysis:
  * research U probes scored the old target after swapping bindings. That can
    measure source-specific disruption for an old answer, but not whether the
    model follows the queried entity's newly correct binding.
  * UNPAIRED_SRC in research used b != a, so it is anti-identity rather than an
    independence comparator.
  * IDENT_BLOCKED masks direct query->source edges but not multi-layer indirect
    paths through separator/other context positions.

This script therefore asks, with the same token bag but changed entity-attribute
binding: does probability/ranking move to the newly correct RWT token?
"""
import argparse, json, math
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Vocabulary and architecture copied from research
BOS, PAD, SEP, HAS, IS = 0, 1, 2, 3, 4
N_ENT, N_ATTR = 8, 10
ENT  = lambda e: 5 + e
SRC  = lambda a: 13 + a
RWT  = lambda a: 23 + a
COPY_CUE, REWRITE_CUE, NEUTRAL_CUE, CONST_CUE = 33, 34, 35, 36
VOCAB = 37
TRAIN_E = list(range(6)); HELD_E = [6, 7]; ALL_E = list(range(8))
N_CTX = 4
SRC_TOKS = list(range(13, 23))
RWT_TOKS = list(range(23, 33))
SL  = 19
IS_POS  = 16
CUE_POS = 15

ARMS = ["ident_full", "ident_blocked", "unpaired_src"]
CUES = ["typed", "uninformative"]


def mk_seq(ce, ca, qi, cue, tgt):
    s = [BOS]
    for i in range(N_CTX):
        s += [ENT(ce[i]), HAS, SRC(ca[i])]
    s += [SEP, ENT(ce[qi]), cue, IS, tgt, PAD]
    return s[:SL]


def causal(L, dev="cpu"):
    return torch.triu(torch.ones(L, L, dtype=torch.bool, device=dev), diagonal=1)


class Blk(nn.Module):
    def __init__(self, d, nh):
        super().__init__()
        self.attn = nn.MultiheadAttention(d, nh, batch_first=True, dropout=0)
        self.ln1 = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, 4*d), nn.GELU(), nn.Linear(4*d, d))
        self.ln2 = nn.LayerNorm(d)
    def forward(self, x, m):
        h = self.ln1(x)
        h, _ = self.attn(h, h, h, attn_mask=m)
        x = x + h
        return x + self.ff(self.ln2(x))


class CLM(nn.Module):
    def __init__(self, V, d, nh, nl, ml):
        super().__init__()
        self.tok = nn.Embedding(V, d)
        self.pos = nn.Embedding(ml, d)
        self.blks = nn.ModuleList([Blk(d, nh) for _ in range(nl)])
        self.ln = nn.LayerNorm(d)
    def forward(self, x, m=None):
        B, L = x.shape
        h = self.tok(x) + self.pos(torch.arange(L, device=x.device))
        for b in self.blks:
            h = b(h, m)
        return self.ln(h) @ self.tok.weight.T


def generate_rebinding_probes(seed, n_per=200):
    """Paired T/S probes with identical token bag and changed held-entity binding.

    T: held query entity has old_attr a_old, another context entity has a_new.
    S: swap these two attributes; held query now has a_new. Same sequence token
       multiset, same entity order, only entity--attribute binding changes.

    We score both RWT(a_old) and RWT(a_new) under T and S. True rebinding means
    T prefers old over new AND S prefers new over old.
    """
    rng = np.random.default_rng(seed)
    probes = []
    for he in HELD_E:
        made = 0
        while made < n_per:
            old = int(rng.integers(N_ATTR))
            new = int(rng.integers(N_ATTR - 1))
            if new >= old:
                new += 1
            oth = rng.choice(TRAIN_E, N_CTX - 1, replace=False).tolist()
            ce = list(rng.permutation(oth + [he]))
            qi = ce.index(he)
            other_positions = [i for i in range(N_CTX) if i != qi]
            si = int(rng.choice(other_positions))
            ca = []
            for i, e in enumerate(ce):
                if i == qi:
                    ca.append(old)
                elif i == si:
                    ca.append(new)
                else:
                    # Avoid accidental old/new duplicates so the only source of
                    # the two candidate attributes is the swapped pair.
                    choices = [x for x in range(N_ATTR) if x not in (old, new)]
                    ca.append(int(rng.choice(choices)))
            ca_sw = list(ca)
            ca_sw[qi], ca_sw[si] = ca[si], ca[qi]
            s_t = mk_seq(ce, ca, qi, PAD, PAD)
            s_s = mk_seq(ce, ca_sw, qi, PAD, PAD)
            probes.append({
                "s_t": s_t, "s_s": s_s,
                "old_attr": old, "new_attr": new,
                "old_rwt": RWT(old), "new_rwt": RWT(new),
                "old_src": SRC(old), "new_src": SRC(new),
                "he": he, "ce": ce, "qi": qi, "swap_i": si,
            })
            made += 1
    return probes


def eval_rebinding(model, probes, cue_tok, dev, batch_size=256):
    model.eval()
    seqs_t = [list(p["s_t"]) for p in probes]
    seqs_s = [list(p["s_s"]) for p in probes]
    for s in seqs_t + seqs_s:
        s[CUE_POS] = cue_tok
    out = {
        "T_old_nll": [], "T_new_nll": [], "S_old_nll": [], "S_new_nll": [],
        "T_logit_old_minus_new": [], "S_logit_new_minus_old": [],
        "T_p_old": [], "T_p_new": [], "S_p_old": [], "S_p_new": [],
        "T_rank_old": [], "T_rank_new": [], "S_rank_old": [], "S_rank_new": [],
        "T_top_old": [], "S_top_new": [], "rebinding_pair_success": [],
        "binding_margin_sum": [], "binding_flip": [],
        "T_rwt_fam": [], "S_rwt_fam": [], "T_src_old_p": [], "S_src_new_p": [],
    }
    with torch.no_grad():
        for side, seqs in [("T", seqs_t), ("S", seqs_s)]:
            st = torch.tensor(seqs, dtype=torch.long, device=dev)
            L = st.size(1) - 1
            cm = causal(L, dev)
            for i in range(0, len(seqs), batch_size):
                b = st[i:i+batch_size, :L]
                logits = model(b, cm)
                l = logits[:, IS_POS, :]
                lp = F.log_softmax(l, dim=-1)
                p = F.softmax(l, dim=-1)
                for j in range(b.size(0)):
                    pr = probes[i+j]
                    old_tok, new_tok = pr["old_rwt"], pr["new_rwt"]
                    old_ix, new_ix = pr["old_attr"], pr["new_attr"]
                    rps = p[j, RWT_TOKS]
                    order = rps.argsort(descending=True)
                    rank_old = int((order == old_ix).nonzero(as_tuple=True)[0].item() + 1)
                    rank_new = int((order == new_ix).nonzero(as_tuple=True)[0].item() + 1)
                    if side == "T":
                        out["T_old_nll"].append(-lp[j, old_tok].item())
                        out["T_new_nll"].append(-lp[j, new_tok].item())
                        out["T_logit_old_minus_new"].append((l[j, old_tok] - l[j, new_tok]).item())
                        out["T_p_old"].append(p[j, old_tok].item())
                        out["T_p_new"].append(p[j, new_tok].item())
                        out["T_rank_old"].append(rank_old)
                        out["T_rank_new"].append(rank_new)
                        out["T_top_old"].append(float(rank_old == 1))
                        out["T_rwt_fam"].append(p[j, RWT_TOKS].sum().item())
                        out["T_src_old_p"].append(p[j, pr["old_src"]].item())
                    else:
                        out["S_old_nll"].append(-lp[j, old_tok].item())
                        out["S_new_nll"].append(-lp[j, new_tok].item())
                        out["S_logit_new_minus_old"].append((l[j, new_tok] - l[j, old_tok]).item())
                        out["S_p_old"].append(p[j, old_tok].item())
                        out["S_p_new"].append(p[j, new_tok].item())
                        out["S_rank_old"].append(rank_old)
                        out["S_rank_new"].append(rank_new)
                        out["S_top_new"].append(float(rank_new == 1))
                        out["S_rwt_fam"].append(p[j, RWT_TOKS].sum().item())
                        out["S_src_new_p"].append(p[j, pr["new_src"]].item())
    # Pair-level quantities require arrays from both sides in same probe order.
    Tm = np.array(out["T_logit_old_minus_new"])
    Sm = np.array(out["S_logit_new_minus_old"])
    out["binding_margin_sum"] = list(Tm + Sm)
    out["rebinding_pair_success"] = list(((Tm > 0) & (Sm > 0)).astype(float))
    out["binding_flip"] = list(((np.array(out["T_rank_old"]) < np.array(out["T_rank_new"])) &
                                 (np.array(out["S_rank_new"]) < np.array(out["S_rank_old"]))).astype(float))
    summary = {}
    for k, vals in out.items():
        arr = np.asarray(vals, dtype=float)
        summary[k] = {"mean": float(np.mean(arr)), "std": float(np.std(arr))}
    # Useful derived deltas: positive S_new_minus_T_old means the swapped correct target has lower probability than the original correct target.
    summary["S_new_minus_T_old_nll"] = {
        "mean": float(np.mean(np.asarray(out["S_new_nll"]) - np.asarray(out["T_old_nll"]))),
        "std": float(np.std(np.asarray(out["S_new_nll"]) - np.asarray(out["T_old_nll"])))
    }
    summary["S_new_minus_S_old_nll"] = {
        "mean": float(np.mean(np.asarray(out["S_new_nll"]) - np.asarray(out["S_old_nll"]))),
        "std": float(np.std(np.asarray(out["S_new_nll"]) - np.asarray(out["S_old_nll"])))
    }
    summary["T_old_minus_T_new_nll"] = {
        "mean": float(np.mean(np.asarray(out["T_old_nll"]) - np.asarray(out["T_new_nll"]))),
        "std": float(np.std(np.asarray(out["T_old_nll"]) - np.asarray(out["T_new_nll"])))
    }
    return summary


def compact(x):
    return {"mean": round(x["mean"], 5), "std": round(x["std"], 5)}


def analyze_existing_curves(d):
    seeds = d["config"]["seeds"]
    metrics = ["rwt_nll", "rwt_fnll", "rwt_wnll", "rwt_mrr", "rwt_top1", "p_src_a"]
    curves = {}
    for cue_key in [k for k in d if k.startswith("cue_")]:
        cue_out = {}
        # assume all arms/seeds have same eval epochs
        epochs = [row["e"] for row in d[cue_key][f"seed{seeds[0]}"][ARMS[0]]["curves"]]
        for ep_i, ep in enumerate(epochs):
            ep_out = {"e": ep}
            # absolute summaries per arm for T correct old target
            for arm in ARMS:
                arm_out = {}
                for m in metrics:
                    vals = [d[cue_key][f"seed{s}"][arm]["curves"][ep_i]["T"][m] for s in seeds]
                    arm_out[m] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
                # T-U gap per arm
                gap_out = {}
                for m in metrics:
                    vals = [d[cue_key][f"seed{s}"][arm]["curves"][ep_i]["T"][m] -
                            d[cue_key][f"seed{s}"][arm]["curves"][ep_i]["U"][m]
                            for s in seeds]
                    gap_out[m] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
                arm_out["T_minus_U"] = gap_out
                ep_out[arm] = arm_out
            # research D interactions at this epoch
            int_out = {}
            for ia in ["ident_full", "ident_blocked"]:
                D = {}
                for m in metrics:
                    vals = []
                    for s in seeds:
                        it = d[cue_key][f"seed{s}"][ia]["curves"][ep_i]["T"][m]
                        iu = d[cue_key][f"seed{s}"][ia]["curves"][ep_i]["U"][m]
                        ut = d[cue_key][f"seed{s}"]["unpaired_src"]["curves"][ep_i]["T"][m]
                        uu = d[cue_key][f"seed{s}"]["unpaired_src"]["curves"][ep_i]["U"][m]
                        vals.append((it - ut) - (iu - uu))
                    D[m] = {"mean": float(np.mean(vals)), "std": float(np.std(vals)),
                            "vals": [float(v) for v in vals]}
                int_out[f"D_{ia}"] = D
            ep_out["interactions"] = int_out
            cue_out[str(ep)] = ep_out
        curves[cue_key] = cue_out
    return curves


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="experiments/archive/functional_learning/data/source_trigger/results.json")
    ap.add_argument("--out", default="experiments/archive/functional_learning/data/rebinding_analysis/results.json")
    ap.add_argument("--probe-n-per-held", type=int, default=240)
    args = ap.parse_args()
    data_path = Path(args.data)
    with open(data_path) as f:
        d = json.load(f)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    seeds = d["config"]["seeds"]
    config = d["config"]
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    curve_summary = analyze_existing_curves(d)
    rebinding = {}
    for cue in CUES:
        cue_key = f"cue_{cue}"
        cue_tok = REWRITE_CUE if cue == "typed" else CONST_CUE
        rebinding[cue_key] = {}
        for seed in seeds:
            probes = generate_rebinding_probes(seed * 3000 + 17, args.probe_n_per_held)
            rebinding[cue_key][f"seed{seed}"] = {}
            for arm in ARMS:
                mdir = data_path.parent / f"models_{cue}_s{seed}"
                mpath = mdir / f"{arm}.pt"
                model = CLM(VOCAB, config["d"], config["nh"], config["nl"], SL).to(dev)
                sd = torch.load(mpath, map_location=dev)
                model.load_state_dict(sd)
                res = eval_rebinding(model, probes, cue_tok, dev)
                rebinding[cue_key][f"seed{seed}"][arm] = {k: compact(v) for k, v in res.items()}
                del model
                if dev == "cuda":
                    torch.cuda.empty_cache()
    # Aggregate across seeds
    agg = {}
    key_metrics = [
        "T_old_nll", "T_new_nll", "S_old_nll", "S_new_nll",
        "T_logit_old_minus_new", "S_logit_new_minus_old", "binding_margin_sum",
        "rebinding_pair_success", "binding_flip", "T_top_old", "S_top_new",
        "S_new_minus_T_old_nll", "S_new_minus_S_old_nll", "T_old_minus_T_new_nll",
        "T_rwt_fam", "S_rwt_fam", "T_src_old_p", "S_src_new_p",
    ]
    for cue in CUES:
        ck = f"cue_{cue}"
        agg[ck] = {}
        for arm in ARMS:
            arm_agg = {}
            for km in key_metrics:
                vals = [rebinding[ck][f"seed{s}"][arm][km]["mean"] for s in seeds]
                arm_agg[km] = {"mean": round(float(np.mean(vals)), 5),
                               "std_across_seeds": round(float(np.std(vals)), 5),
                               "seed_means": [round(float(v), 5) for v in vals]}
            agg[ck][arm] = arm_agg
        # identity-full minus comparators for rebinding-sensitive metrics, noting comparators are imperfect
        contrasts = {}
        for comp in ["unpaired_src", "ident_blocked"]:
            c = {}
            for km in ["S_logit_new_minus_old", "binding_margin_sum", "rebinding_pair_success", "S_new_nll", "S_new_minus_S_old_nll", "S_top_new"]:
                vals = []
                for s in seeds:
                    vals.append(rebinding[ck][f"seed{s}"]["ident_full"][km]["mean"] -
                                rebinding[ck][f"seed{s}"][comp][km]["mean"])
                c[f"ident_full_minus_{comp}:{km}"] = {"mean": round(float(np.mean(vals)), 5),
                                                       "std_across_seeds": round(float(np.std(vals)), 5),
                                                       "seed_vals": [round(float(v), 5) for v in vals]}
            contrasts[comp] = c
        agg[ck]["contrasts"] = contrasts

    result = {
        "status": "REBINDING_ANALYSIS_DONE",
        "input_step012_results": str(data_path),
        "notes": {
            "unpaired_src_limitation": "research UNPAIRED_SRC sampled b != a and therefore teaches anti-identity, not source-target independence.",
            "blocked_limitation": "research IDENT_BLOCKED blocks direct query-segment attention to the source event but not all multi-layer indirect paths through SEP or other context positions.",
            "rebinding_definition": "T and swapped S have the same token bag; correct target changes from old RWT to new RWT. True binding should prefer old in T and new in S."
        },
        "curve_summary": curve_summary,
        "rebinding_final_models": rebinding,
        "rebinding_aggregate": agg,
    }
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    # Print a compact scientific summary
    print(json.dumps({"status": result["status"], "out": str(out_path)}, indent=2))
    print("\nExisting research trajectory: typed T RWT NLL / within NLL / MRR by epoch")
    for arm in ARMS:
        parts = []
        for ep in sorted(curve_summary["cue_typed"], key=lambda x: int(x)):
            row = curve_summary["cue_typed"][ep][arm]
            parts.append(f"e{ep}:NLL{row['rwt_nll']['mean']:.2f},W{row['rwt_wnll']['mean']:.2f},MRR{row['rwt_mrr']['mean']:.2f}")
        print(f"  {arm}: " + " | ".join(parts))
    print("\nFinal-model rebinding aggregate (same token bag, queried binding changed)")
    print("Columns: T_old_margin=logit(old)-logit(new) in original T; S_new_margin=logit(new)-logit(old) after swap; pair_success requires both >0.")
    for ck in ["cue_typed", "cue_uninformative"]:
        print(f"\n{ck}")
        for arm in ARMS:
            a = agg[ck][arm]
            print(f"  {arm:15s} T_old_margin {a['T_logit_old_minus_new']['mean']:+.3f}±{a['T_logit_old_minus_new']['std_across_seeds']:.3f}"
                  f" | S_new_margin {a['S_logit_new_minus_old']['mean']:+.3f}±{a['S_logit_new_minus_old']['std_across_seeds']:.3f}"
                  f" | pair_success {a['rebinding_pair_success']['mean']:.3f}±{a['rebinding_pair_success']['std_across_seeds']:.3f}"
                  f" | S_new_nll {a['S_new_nll']['mean']:.2f}"
                  f" | S_top_new {a['S_top_new']['mean']:.3f}")
        print("  contrasts ident_full - unpaired_src:")
        c = agg[ck]["contrasts"]["unpaired_src"]
        for name, val in c.items():
            if name.endswith("S_logit_new_minus_old") or name.endswith("binding_margin_sum") or name.endswith("rebinding_pair_success") or name.endswith("S_new_nll") or name.endswith("S_top_new"):
                print(f"    {name}: {val['mean']:+.3f} ± {val['std_across_seeds']:.3f}  seeds={val['seed_vals']}")

if __name__ == "__main__":
    main()
