#!/usr/bin/env python3
"""research analysis: load results and produce interpretable summaries and figure."""
import json, numpy as np, argparse
from pathlib import Path

def load(path):
    with open(path) as f: return json.load(f)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/source_trigger/results.json")
    ap.add_argument("--fig", default="figures/source_trigger.png")
    A = ap.parse_args()
    d = load(A.data)
    seeds = d["config"]["seeds"]
    arms = ["ident_full", "ident_blocked", "unpaired_src"]
    met_labels = {
        "rwt_nll": "RWT total NLL",
        "rwt_wnll": "Within-RWT NLL",
        "rwt_fnll": "RWT family NLL",
        "p_src_a": "p(SRC(a))",
        "src_fam": "SRC family mass",
        "copy_nll": "Copy NLL",
        "rwt_mrr": "Within-RWT MRR",
        "rwt_top1": "Within-RWT Top-1",
    }

    print("="*70)
    print("research SOURCE-TRIGGER EXPERIMENT ANALYSIS")
    print("="*70)

    for mkey in sorted(k for k in d if k.startswith("cue_")):
        print(f"\n{'='*60}\n{mkey}\n{'='*60}")
        
        # Final epoch metrics per arm × probe condition
        print("\n--- Final epoch metrics ---")
        for arm in arms:
            print(f"\n  {arm}:")
            for cond in ["T", "U", "NS"]:
                vals = {}
                for m in met_labels:
                    vs = [d[mkey][f"seed{s}"][arm]["curves"][-1][cond][m] for s in seeds]
                    vals[m] = (np.mean(vs), np.std(vs))
                print(f"    {cond}: rwt_nll={vals['rwt_nll'][0]:.3f}±{vals['rwt_nll'][1]:.3f}  "
                      f"wnll={vals['rwt_wnll'][0]:.3f}±{vals['rwt_wnll'][1]:.3f}  "
                      f"fnll={vals['rwt_fnll'][0]:.3f}±{vals['rwt_fnll'][1]:.3f}  "
                      f"psrc={vals['p_src_a'][0]:.4f}±{vals['p_src_a'][1]:.4f}  "
                      f"mrr={vals['rwt_mrr'][0]:.3f}±{vals['rwt_mrr'][1]:.3f}")

        # Source-specific interactions
        print("\n--- Source-specific interactions D_m = (I_T-UP_T) - (I_U-UP_U) ---")
        for ia in ["ident_full", "ident_blocked"]:
            print(f"\n  D_{ia}:")
            for m in met_labels:
                ds = []
                for s in seeds:
                    it = d[mkey][f"seed{s}"][ia]["curves"][-1]["T"][m]
                    iu = d[mkey][f"seed{s}"][ia]["curves"][-1]["U"][m]
                    ut = d[mkey][f"seed{s}"]["unpaired_src"]["curves"][-1]["T"][m]
                    uu = d[mkey][f"seed{s}"]["unpaired_src"]["curves"][-1]["U"][m]
                    ds.append((it-ut)-(iu-uu))
                print(f"    {met_labels[m]:20s}: {np.mean(ds):+.5f} ± {np.std(ds):.5f}  [{', '.join(f'{v:+.5f}' for v in ds)}]")

        # Local access contribution
        print("\n  Local access (D_FULL - D_BLOCKED):")
        for m in met_labels:
            df, db = [], []
            for s in seeds:
                it=d[mkey][f"seed{s}"]["ident_full"]["curves"][-1]["T"][m]
                iu=d[mkey][f"seed{s}"]["ident_full"]["curves"][-1]["U"][m]
                ut=d[mkey][f"seed{s}"]["unpaired_src"]["curves"][-1]["T"][m]
                uu=d[mkey][f"seed{s}"]["unpaired_src"]["curves"][-1]["U"][m]
                df.append((it-ut)-(iu-uu))
                it2=d[mkey][f"seed{s}"]["ident_blocked"]["curves"][-1]["T"][m]
                iu2=d[mkey][f"seed{s}"]["ident_blocked"]["curves"][-1]["U"][m]
                db.append((it2-ut)-(iu2-uu))
            la = [f-b for f,b in zip(df,db)]
            print(f"    {met_labels[m]:20s}: {np.mean(la):+.5f} ± {np.std(la):.5f}")

        # T-U gap per arm (simple true-source benefit)
        print("\n--- T-U gap (T minus U, per arm) ---")
        for arm in arms:
            for m in ["rwt_nll", "rwt_wnll", "p_src_a", "rwt_mrr"]:
                gaps = []
                for s in seeds:
                    t = d[mkey][f"seed{s}"][arm]["curves"][-1]["T"][m]
                    u = d[mkey][f"seed{s}"][arm]["curves"][-1]["U"][m]
                    gaps.append(t-u)
                print(f"  {arm:15s} T-U {met_labels[m]:20s}: {np.mean(gaps):+.5f} ± {np.std(gaps):.5f}")

        # Cue swap results (typed only)
        if mkey == "cue_typed":
            print("\n--- Cue swap at final epoch ---")
            for arm in arms:
                has_swap = "cue_swap" in d[mkey].get(f"seed{seeds[0]}", {}).get(arm, {}).get("curves", [{}])[-1]
                if not has_swap: continue
                for sn in ["copy", "const"]:
                    for cond in ["T", "U"]:
                        vals = {}
                        for m in ["rwt_nll", "p_src_a", "rwt_mrr"]:
                            vs = [d[mkey][f"seed{s}"][arm]["curves"][-1]["cue_swap"][sn][cond][m] for s in seeds]
                            vals[m] = (np.mean(vs), np.std(vs))
                        print(f"  {arm:15s} {sn}_cue {cond}: rwt_nll={vals['rwt_nll'][0]:.3f}  "
                              f"psrc={vals['p_src_a'][0]:.4f}  mrr={vals['rwt_mrr'][0]:.3f}")

    # Learning curves
    print("\n--- Learning curve snapshots (typed, seed-averaged) ---")
    if "cue_typed" in d:
        for arm in arms:
            epochs_data = {}
            for s in seeds:
                for row in d["cue_typed"][f"seed{s}"][arm]["curves"]:
                    ep = row["e"]
                    if ep not in epochs_data: epochs_data[ep] = {"T_rnll":[], "U_rnll":[], "T_psrc":[], "U_psrc":[], "T_mrr":[]}
                    epochs_data[ep]["T_rnll"].append(row["T"]["rwt_nll"])
                    epochs_data[ep]["U_rnll"].append(row["U"]["rwt_nll"])
                    epochs_data[ep]["T_psrc"].append(row["T"]["p_src_a"])
                    epochs_data[ep]["U_psrc"].append(row["U"]["p_src_a"])
                    epochs_data[ep]["T_mrr"].append(row["T"]["rwt_mrr"])
            print(f"\n  {arm}:")
            for ep in sorted(epochs_data):
                ed = epochs_data[ep]
                print(f"    e={ep:3d}  T_rnll={np.mean(ed['T_rnll']):.3f}  U_rnll={np.mean(ed['U_rnll']):.3f}  "
                      f"T_psrc={np.mean(ed['T_psrc']):.4f}  U_psrc={np.mean(ed['U_psrc']):.4f}  "
                      f"T_mrr={np.mean(ed['T_mrr']):.3f}")

    # Generate figure
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        fig.suptitle("research: Source-Trigger Experiment", fontsize=14, fontweight="bold")

        metrics_to_plot = [
            ("rwt_nll", "RWT Total NLL", axes[0, 0]),
            ("rwt_wnll", "Within-RWT NLL", axes[0, 1]),
            ("p_src_a", "p(SRC(a))", axes[0, 2]),
            ("rwt_mrr", "Within-RWT MRR", axes[1, 0]),
            ("copy_nll", "Copy NLL", axes[1, 1]),
        ]
        colors = {"ident_full": "C0", "ident_blocked": "C1", "unpaired_src": "C2"}
        styles = {"T": "-", "U": "--"}

        for m, label, ax in metrics_to_plot:
            if "cue_typed" not in d: continue
            for arm in arms:
                for cond in ["T", "U"]:
                    epochs_vals = {}
                    for s in seeds:
                        for row in d["cue_typed"][f"seed{s}"][arm]["curves"]:
                            ep = row["e"]
                            if ep not in epochs_vals: epochs_vals[ep] = []
                            epochs_vals[ep].append(row[cond][m])
                    eps = sorted(epochs_vals)
                    means = [np.mean(epochs_vals[e]) for e in eps]
                    ax.plot(eps, means, styles[cond], color=colors[arm],
                            label=f"{arm[:8]}_{cond}" if m == "rwt_nll" else "")
            ax.set_xlabel("Epoch"); ax.set_ylabel(label)
            ax.set_title(label)
        
        axes[0, 0].legend(fontsize=7, ncol=2)

        # Interaction bar chart
        ax = axes[1, 2]
        if "cue_typed" in d:
            bar_mets = ["rwt_nll", "rwt_wnll", "p_src_a"]
            x = np.arange(len(bar_mets))
            width = 0.35
            for ii, ia in enumerate(["ident_full", "ident_blocked"]):
                means_bar, stds_bar = [], []
                for m in bar_mets:
                    ds = []
                    for s in seeds:
                        it=d["cue_typed"][f"seed{s}"][ia]["curves"][-1]["T"][m]
                        iu=d["cue_typed"][f"seed{s}"][ia]["curves"][-1]["U"][m]
                        ut=d["cue_typed"][f"seed{s}"]["unpaired_src"]["curves"][-1]["T"][m]
                        uu=d["cue_typed"][f"seed{s}"]["unpaired_src"]["curves"][-1]["U"][m]
                        ds.append((it-ut)-(iu-uu))
                    means_bar.append(np.mean(ds)); stds_bar.append(np.std(ds))
                ax.bar(x + ii*width, means_bar, width, yerr=stds_bar,
                       label=ia[:8], color=colors[ia], alpha=0.7)
            ax.set_xticks(x + width/2)
            ax.set_xticklabels(["RWT NLL", "Within NLL", "p(SRC)"], fontsize=8)
            ax.axhline(0, color="black", linewidth=0.5)
            ax.set_title("Source-specific D_m")
            ax.legend(fontsize=8)

        plt.tight_layout()
        Path(A.fig).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(A.fig, dpi=150, bbox_inches="tight")
        print(f"\nFigure saved: {A.fig}")
    except Exception as e:
        print(f"\nFigure generation failed: {e}")

if __name__ == "__main__":
    main()
