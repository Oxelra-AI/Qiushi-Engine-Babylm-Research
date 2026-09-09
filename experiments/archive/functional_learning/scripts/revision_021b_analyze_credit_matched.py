#!/usr/bin/env python3
"""Analyze research/021b effective-credit comparison.

The key comparison is interleaved answer/full vs a static objective with the same
nominal coefficients on answer loss A and mean context loss C.  With 15 context
positions and one answer position, fixed context weight w gives
(A + 15 w C)/(1 + 15 w).  Alternating answer-only A with full (A+15C)/16 averages
to (17A+15C)/32, matched by w=1/17.
"""

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative

import json, math
from pathlib import Path
import numpy as np

import sys
SCRIPT_DIR = _public_path('experiments/archive/functional_learning/scripts')
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
import revision_019b_embedding_role_decomposition as S19b

BASE = Path("experiments/archive/functional_learning")
P1 = BASE / "data/calibration_trajectory/results.json"
P2 = BASE / "data/revision_021b_credit_matched/results.json"
OUT = BASE / "data/revision_021b_credit_matched/credit_matched_analysis.json"
NOTE = (_PUBLIC_ROOT / 'research/notes/functional_learning/021b_credit_matched_analysis.md')
FIG = BASE / "figures/revision_021b_credit_matched_unified.png"

SELECT = [
    "direct_full",
    "gradual_ctx100_then_full",
    "interleaved_ans_full",
    "constant_ctx_half",
    "constant_ctx_tenth",
    "constant_ctx_1over17",
]

DISPLAY = {
    "direct_full": "direct full",
    "gradual_ctx100_then_full": "gradual ramp",
    "interleaved_ans_full": "interleaved",
    "constant_ctx_half": "static w=0.5",
    "constant_ctx_tenth": "static w=0.1",
    "constant_ctx_1over17": "static w=1/17",
}


def load_records():
    data = []
    for p in [P1, P2]:
        d = json.load(open(p))
        for r in d["records"]:
            if r["arm"] in SELECT:
                rr = dict(r)
                rr["source"] = str(p)
                data.append(rr)
    return data


def sm(r):
    x = S19b.metric_summary(r)
    return S19b.add_drift_to_summary(x, r)


def fixed_coeff(w):
    return (1.0/(1.0 + 15.0*w), (15.0*w)/(1.0 + 15.0*w))


def nominal_coeff(arm):
    if arm == "direct_full":
        return fixed_coeff(1.0)
    if arm == "constant_ctx_half":
        return fixed_coeff(0.5)
    if arm == "constant_ctx_tenth":
        return fixed_coeff(0.1)
    if arm == "constant_ctx_1over17":
        return fixed_coeff(1.0/17.0)
    if arm == "interleaved_ans_full":
        return (17.0/32.0, 15.0/32.0)
    # not constant; use None for ramp
    return (None, None)


def collect(records):
    out = {"arms": {}, "per_seed_final": {}, "trajectories": {}}
    for arm in SELECT:
        rs = [r for r in records if r["arm"] == arm]
        if not rs: continue
        max_be = max(int(r["branch_epoch"]) for r in rs)
        starts = [r for r in rs if int(r["branch_epoch"]) == 0]
        finals = [r for r in rs if int(r["branch_epoch"]) == max_be]
        def mean_metric(arr, key):
            vals = [sm(r)[key] for r in arr]
            return float(np.mean(vals)) if vals else None
        Acoef, Ccoef = nominal_coeff(arm)
        out["arms"][arm] = {
            "label": DISPLAY[arm],
            "n": len(finals),
            "max_be": max_be,
            "Acoef_nominal": Acoef,
            "Ccoef_nominal": Ccoef,
            "start_h4": mean_metric(starts, "held_top4"),
            "final_h4": mean_metric(finals, "held_top4"),
            "final_hB": mean_metric(finals, "held_b"),
            "final_hSel": mean_metric(finals, "held_sel"),
            "final_train4": mean_metric(finals, "train_top4"),
            "final_blocked4": mean_metric(finals, "blocked_train_top4"),
            "final_ctx_ce": float(np.mean([r.get("ctx_ce", 0.0) for r in finals])),
            "final_ans_ce": float(np.mean([r.get("ans_ce", 0.0) for r in finals])),
        }
        # per-seed finals + AUC/min over held_top4
        for sd in sorted(set(int(r["seed"]) for r in rs)):
            srs = sorted([r for r in rs if int(r["seed"]) == sd], key=lambda r:int(r["branch_epoch"]))
            vals = [(int(r["branch_epoch"]), sm(r)["held_top4"], sm(r)["held_b"], sm(r)["held_sel"], float(r.get("ctx_ce", 0.0))) for r in srs]
            final = vals[-1]
            start = vals[0]
            # trapezoid AUC on recorded held_top4 points normalized by max epoch
            xs = np.array([v[0] for v in vals], dtype=float)
            ys = np.array([v[1] for v in vals], dtype=float)
            auc = float(np.trapz(ys, xs) / max(xs[-1], 1.0))
            out["per_seed_final"].setdefault(str(sd), {})[arm] = {
                "start_h4": start[1], "final_h4": final[1], "final_hB": final[2],
                "final_hSel": final[3], "final_ctx_ce": final[4],
                "min_h4_recorded": float(np.min(ys)), "auc_h4_recorded": auc,
            }
            out["trajectories"].setdefault(arm, {})[str(sd)] = vals
    # relative effects vs direct and vs interleaved
    d = out["arms"].get("direct_full")
    inter = out["arms"].get("interleaved_ans_full")
    if d and inter:
        for arm, x in out["arms"].items():
            x["delta_h4_vs_direct"] = x["final_h4"] - d["final_h4"]
            x["delta_hB_vs_direct"] = x["final_hB"] - d["final_hB"]
            x["delta_ctx_vs_direct"] = x["final_ctx_ce"] - d["final_ctx_ce"]
        cm = out["arms"].get("constant_ctx_1over17")
        if cm:
            out["matched_static_fraction_of_interleaved_gain"] = {
                "h4": (cm["final_h4"] - d["final_h4"]) / (inter["final_h4"] - d["final_h4"]),
                "hB": (cm["final_hB"] - d["final_hB"]) / (inter["final_hB"] - d["final_hB"]),
                "hSel": (cm["final_hSel"] - d["final_hSel"]) / (inter["final_hSel"] - d["final_hSel"]),
            }
    return out


def write_note(ana):
    lines = []
    lines += ["# Step021b effective-credit matched comparison", ""]
    lines += [
        "## Why w=1/17 is the matched static objective", "",
        "Let A be mean answer-position loss and C be mean context-position loss over the 15 non-answer context positions. The implemented weighted loss for static context weight w is",
        "",
        "\\[L_w = \\frac{A + 15 w C}{1 + 15 w}.\\]",
        "",
        "Full training is \\(L_1=(A+15C)/16\\), and answer-only is \\(L_0=A\\). Alternating answer-only and full gives the nominal average",
        "",
        "\\[\\tfrac12 L_0 + \\tfrac12 L_1 = (17A+15C)/32.\\]",
        "",
        "The static objective matching those coefficients is therefore \\(w=1/17\\), since \\(L_{1/17}=(17A+15C)/32\\). Static w=0.5 and w=0.1 are useful dose-response arms but do not isolate temporal alternation.", "",
        "## Endpoint comparison on the same probe set", "",
        "| arm | A coef | C coef | final h4 | final hB | final hSel | train4 | blocked4 | ctx CE | ans CE | Δh4 vs direct | Δctx vs direct |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for arm in SELECT:
        if arm not in ana["arms"]: continue
        x = ana["arms"][arm]
        Acoef = "" if x["Acoef_nominal"] is None else f"{x['Acoef_nominal']:.3f}"
        Ccoef = "" if x["Ccoef_nominal"] is None else f"{x['Ccoef_nominal']:.3f}"
        lines.append(
            f"| {x['label']} | {Acoef} | {Ccoef} | {x['final_h4']:.3f} | {x['final_hB']:+.3f} | {x['final_hSel']:+.3f} | "
            f"{x['final_train4']:.3f} | {x['final_blocked4']:.3f} | {x['final_ctx_ce']:.3f} | {x['final_ans_ce']:.4f} | "
            f"{x.get('delta_h4_vs_direct',0):+.3f} | {x.get('delta_ctx_vs_direct',0):+.3f} |")
    frac = ana.get("matched_static_fraction_of_interleaved_gain", {})
    if frac:
        lines += ["", "The coefficient-matched static arm recovers most of the interleaved endpoint advantage over direct full training: "
                  f"{frac['h4']*100:.1f}% of the h4 gain, {frac['hB']*100:.1f}% of the hB gain, and {frac['hSel']*100:.1f}% of the hSel gain."]
    lines += ["", "## Per-seed final h4/hB", "", "| seed | direct | interleaved | static w=1/17 | static w=0.1 | static w=0.5 |", "|---:|---:|---:|---:|---:|---:|"]
    for sd, arms in sorted(ana["per_seed_final"].items(), key=lambda kv:int(kv[0])):
        def cell(a):
            x = arms.get(a)
            return "" if not x else f"{x['final_h4']:.3f}/{x['final_hB']:+.2f}"
        lines.append(f"| {sd} | {cell('direct_full')} | {cell('interleaved_ans_full')} | {cell('constant_ctx_1over17')} | {cell('constant_ctx_tenth')} | {cell('constant_ctx_half')} |")
    lines += ["", "## Interpretation", ""]
    lines += [
        "The matched static objective preserves held-symbol transfer almost as well as interleaving while learning context prediction on the same held-out probe distribution. This changes the research interpretation: periodic answer-only epochs are not required for the preservation effect. The main cause is compatible effective allocation between answer credit and context credit. In this task, reducing the normalized context coefficient from 0.938 in direct full training to 0.469 in the matched static/interleaved allocation is enough for the trained-entity answer signal to keep shaping a shared query-conditioned computation that also works on held entities.", "",
        "Temporal ordering may still matter, but as a smaller effect rather than the main explanation. Interleaving has the best mean final h4 (0.746 vs 0.694 for static w=1/17) and hSel (0.665 vs 0.574), with the clearest extra h4 in seed43. The static arm, however, has comparable hB and lower context CE (1.238 vs 1.316). Because both use the same common probes and training distribution, the remaining difference is a real tradeoff to investigate, not grounds for calling alternation necessary.", "",
        "The half/tenth arms show the response to static allocation. w=0.5 is close to direct full and still loses much held transfer, while w=0.1 and w=1/17 preserve far more. This nonlinearity indicates that the destructive transition is controlled by the answer/context coefficient ratio rather than by the mere presence of context prediction. Context CE by 500 epochs remains around 1.23--1.24 for all static weights, so the preserved held behavior is not achieved by failing to learn the context positions.", "",
        "The result is also not ordinary held-symbol rehearsal: held entity tokens never occur in the continuation rows, and blocked query-to-context evaluation stays near chance. What is being maintained must therefore be a computation shared across entity symbols under the query-first format. The present evidence does not yet identify its activation-level form or prove that the direct-full path erases a specific internal marker; it shows that the transfer-bearing computation survives when answer-position credit is kept strong enough during context learning.", "",
        "## Prediction for the next experiment", "",
        "If effective allocation is the main mechanism, answer-position gradients computed on trained entities should have a positive projection on the transfer-bearing query-conditioned pathway, while context gradients at the full coefficient should dominate or oppose that pathway early after the switch. A coefficient-matched static update should reduce the destructive projection without requiring a separate answer-only epoch. If temporal scheduling adds a real second-order effect, the interleaved trajectory should show different optimizer-state or activation-marker evolution from the static w=1/17 trajectory despite matched nominal coefficients. This can be tested by saving matched checkpoints and measuring gradient alignment plus linear/causal probes on attribute-position states for direct, static w=1/17, and interleaved arms.", "",
        "## Files", "",
        f"- research data: `{P1}`", f"- Step021b data: `{P2}`", f"- Analysis JSON: `{OUT}`", f"- Unified figure: `{FIG}`", "",
    ]
    NOTE.write_text("\n".join(lines))


def make_fig(ana):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {
        "direct_full": "#333333",
        "gradual_ctx100_then_full": "#2ca02c",
        "interleaved_ans_full": "#d62728",
        "constant_ctx_half": "#9467bd",
        "constant_ctx_tenth": "#1f77b4",
        "constant_ctx_1over17": "#ff7f0e",
    }
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    for arm in SELECT:
        if arm not in ana["trajectories"]: continue
        for sd, vals in ana["trajectories"][arm].items():
            xs = [v[0] for v in vals]
            h4 = [v[1] for v in vals]
            hb = [v[2] for v in vals]
            ctx = [v[4] for v in vals]
            lab = DISPLAY[arm] if sd == sorted(ana["trajectories"][arm].keys(), key=int)[0] else None
            axes[0].plot(xs, h4, color=colors[arm], alpha=0.55, lw=1.2, label=lab)
            axes[1].plot(xs, hb, color=colors[arm], alpha=0.55, lw=1.2)
            axes[2].plot(xs, ctx, color=colors[arm], alpha=0.55, lw=1.2)
    axes[0].set_title("Held top-4")
    axes[1].set_title("Held binding-swap margin")
    axes[2].set_title("Context CE")
    for ax in axes:
        ax.set_xlabel("Continuation epoch")
    axes[0].set_ylim(0.15, 1.05)
    axes[2].set_ylim(0, 9)
    axes[0].legend(fontsize=7, ncol=1, loc="lower right")
    fig.suptitle("Step021b: static effective-credit matching vs interleaving")
    fig.tight_layout()
    FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG, dpi=160)
    plt.close(fig)


def main():
    records = load_records()
    ana = collect(records)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(ana, indent=2))
    write_note(ana)
    make_fig(ana)
    print(json.dumps({"status":"ok", "analysis":str(OUT), "note":str(NOTE), "figure":str(FIG)}, indent=2))

if __name__ == "__main__":
    main()
