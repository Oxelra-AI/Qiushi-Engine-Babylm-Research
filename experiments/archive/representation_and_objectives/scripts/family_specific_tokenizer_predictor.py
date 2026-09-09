#!/usr/bin/env python3
"""research: family-specific tokenizer frontier predictor.

This refines `tokenizer_support_frontier_predictor.py` by using the actual
research evaluation-family statistics rather than global pool support. It asks a narrow,
decision-changing question:

    Does a pure support-floored tokenizer (minfreq25/minfreq50) plausibly close the
    ~0.8-1.0 Overall gap to the 41.8 frontier by itself, or must the next H100 route
    combine support flooring with a distinct factor (depth/masking/curriculum/objective)?

It uses no new training and no new official model evaluation. It calibrates on the two
legal endpoints that already have full official vectors (legal16k, legal40k).
"""
import csv
import json
import pathlib

ROOT = pathlib.Path("experiments/archive/representation_and_objectives")
FAMILY = ROOT / "data/tokenizer_support_spectrum/eval_family_low_support.csv"
research = ROOT / "data/legal40k_endpoint_interpretation/legal40k_endpoint_interpretation.json"
OUT_DIR = ROOT / "data/family_specific_tokenizer_predictor"
OUT_JSON = OUT_DIR / "family_specific_tokenizer_predictor.json"
OUT_CSV = OUT_DIR / "family_specific_predictions.csv"
NOTE = (ROOT.parents[2] / 'research/notes/representation_and_objectives/family_specific_tokenizer_predictor.md')

COLUMNS = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "SuperGLUE", "GlobalPIQA", "Reading", "AoA"]
TOKENIZERS = [
    "legal_a01_16k",
    "legal_byte_bpe_40k",
    "legal_byte_bpe_40k_minfreq25",
    "legal_byte_bpe_40k_minfreq50",
    "legal_byte_bpe_24k",
    "legal_byte_bpe_32k",
]

# Score columns map directly to research family names.
FAMILY = {c: c for c in COLUMNS if c != "AoA"}


def clamp01(x):
    return max(0.0, min(1.0, x))


def load_family():
    d = {}
    with open(FAMILY) as f:
        for r in csv.DictReader(f):
            tok = r["tokenizer"]
            fam = r["family"]
            d[(tok, fam)] = {
                "eval_tokens": float(r["eval_tokens"]),
                "frac_lt50": float(r["frac_lt50"]),
                "frac_lt100": float(r["frac_lt100"]),
                "p10_support": float(r["p10_support"]),
                "mean_pool_support": float(r["mean_pool_support"]),
            }
    return d


def norm_between(value, lo, hi):
    # lo is legal16, hi is legal40. Return 0 at legal16 and 1 at legal40.
    if abs(hi - lo) < 1e-12:
        return 0.0
    return clamp01((value - lo) / (hi - lo))


def predict(interp, famstats, mode):
    v16 = interp["legal16k_mean"]
    v40 = interp["legal40k_mean"]
    deltas = interp["mean_delta_legal40k_minus_legal16k"]
    out = {}
    for tok in TOKENIZERS:
        vec = {}
        drivers = {}
        for col in COLUMNS:
            if col == "AoA":
                vec[col] = 0.0
                drivers[col] = {"driver": "fixed_aoa", "frac": 0.0}
                continue
            fam = FAMILY[col]
            x16 = famstats[("legal_a01_16k", fam)]
            x40 = famstats[("legal_byte_bpe_40k", fam)]
            xt = famstats[(tok, fam)]
            # Segmentation position: use the reduction in eval tokens from 16k to 40k;
            # normalized so 0=16k, 1=40k.
            f_seg = norm_between(xt["eval_tokens"], x16["eval_tokens"], x40["eval_tokens"])
            # Support position: use family-specific fraction of eval tokens whose pool
            # support is <50; normalized so 0=16k, 1=40k.
            f_supp = norm_between(xt["frac_lt50"], x16["frac_lt50"], x40["frac_lt50"])
            if mode == "sign_index":
                # If 40k improved a column, attribute the improvement to segmentation;
                # if it worsened, attribute the loss to rare-token support.
                frac = f_seg if deltas[col] >= 0 else f_supp
                driver = "seg_eval_tokens" if deltas[col] >= 0 else "support_frac_lt50"
            elif mode == "dominant_manual":
                if col in {"BLiMP", "Supplement", "EWoK"}:
                    frac = f_seg
                    driver = "seg_eval_tokens"
                elif col in {"GlobalPIQA", "Entity"}:
                    frac = f_supp
                    driver = "support_frac_lt50"
                else:
                    frac = 0.5 * (f_seg + f_supp)
                    driver = "avg_seg_support"
            else:
                raise ValueError(mode)
            vec[col] = v16[col] + frac * (v40[col] - v16[col])
            drivers[col] = {"driver": driver, "frac": frac, "f_seg": f_seg, "f_supp": f_supp}
        vec["Overall"] = sum(vec[c] for c in COLUMNS) / len(COLUMNS)
        out[tok] = {"vector": vec, "drivers": drivers}
    return out


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    interp = json.loads(research.read_text())
    famstats = load_family()
    modes = {m: predict(interp, famstats, m) for m in ["sign_index", "dominant_manual"]}

    v16 = interp["legal16k_mean"]
    v40 = interp["legal40k_mean"]
    anchor_ok = True
    for mode, preds in modes.items():
        anchor_ok &= abs(preds["legal_a01_16k"]["vector"]["Overall"] - v16["Overall"]) < 1e-8
        anchor_ok &= abs(preds["legal_byte_bpe_40k"]["vector"]["Overall"] - v40["Overall"]) < 1e-8

    seed_sd = interp.get("legal40k_overall_population_sd", 0.36022302756231284)
    # Add a model-form band equal to half the disagreement between the two predictors for the candidate.
    candidates = ["legal_byte_bpe_40k_minfreq25", "legal_byte_bpe_40k_minfreq50", "legal_byte_bpe_24k", "legal_byte_bpe_32k"]
    disagreements = {
        tok: abs(modes["sign_index"][tok]["vector"]["Overall"] - modes["dominant_manual"][tok]["vector"]["Overall"])
        for tok in candidates
    }
    summary = {}
    for tok in candidates:
        pred_vals = [modes[m][tok]["vector"]["Overall"] for m in modes]
        central = sum(pred_vals) / len(pred_vals)
        form_band = disagreements[tok] / 2.0
        band = (seed_sd ** 2 + form_band ** 2) ** 0.5
        summary[tok] = {
            "sign_index_overall": modes["sign_index"][tok]["vector"]["Overall"],
            "dominant_manual_overall": modes["dominant_manual"][tok]["vector"]["Overall"],
            "central_overall": central,
            "predictor_disagreement": disagreements[tok],
            "combined_band_plus_minus": band,
            "upper_vs_leader_41p8": central + band - 41.8,
        }

    best = max((summary[t]["central_overall"], t) for t in candidates)

    payload = {
        "status": "FAMILY_SPECIFIC_TOKENIZER_PREDICTOR",
        "method": "family-specific interpolation from measured legal16k/legal40k official vectors using research eval-family eval_tokens and frac_lt50 support exposure",
        "anchor_self_check_ok": anchor_ok,
        "modes": modes,
        "candidate_summary": summary,
        "best_central_candidate": {"tokenizer": best[1], "central_overall": best[0]},
        "visible_leader_overall": 41.8,
        "decision": {
            "pure_support_floor_crosses_leader_within_band": any(summary[t]["upper_vs_leader_41p8"] >= 0 for t in candidates),
            "reading": "Both family-specific predictors are calibrated on the measured endpoints. If all support-floor candidates remain below 41.8 even after seed/model-form uncertainty, pure tokenizer interpolation is unlikely to be an endpoint route by itself and should be combined with a new factor rather than consuming a whole A01 two-seed run alone.",
        },
        "sources": {
            "eval_family_low_support_csv": str(FAMILY),
            "legal40k_endpoint_interpretation_json": str(research),
        },
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2))

    with open(OUT_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["mode", "tokenizer"] + COLUMNS + ["Overall"])
        for mode, preds in modes.items():
            for tok in TOKENIZERS:
                v = preds[tok]["vector"]
                w.writerow([mode, tok] + [round(v[c], 4) for c in COLUMNS] + [round(v["Overall"], 4)])

    lines = []
    lines.append("# research — Family-specific legal tokenizer predictor\n")
    lines.append("This refines the global support-frontier predictor by using actual official-evaluation family statistics: family eval-token counts as a segmentation proxy and family `frac_lt50` as under-trained-token exposure. It is CPU-only and calibrates only on the measured legal16k/legal40k official endpoints.\n")
    lines.append(f"- Anchor self-check passed: **{anchor_ok}**")
    lines.append(f"- Uncertainty baseline: legal40k two-seed population SD {seed_sd:.4f}; model-form uncertainty is half the disagreement between the sign-index and manual-dominant predictors.\n")
    lines.append("## Candidate Overall predictions\n")
    lines.append("| tokenizer | sign-index | manual-dominant | central | ± band | upper vs 41.8 |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for tok in candidates:
        s = summary[tok]
        lines.append(f"| {tok} | {s['sign_index_overall']:.4f} | {s['dominant_manual_overall']:.4f} | {s['central_overall']:.4f} | {s['combined_band_plus_minus']:.4f} | {s['upper_vs_leader_41p8']:+.4f} |")
    lines.append("")
    lines.append(f"Best central candidate: **{best[1]}** at **{best[0]:.4f}**.")
    lines.append(f"Pure support-floor crosses 41.8 within band: **{payload['decision']['pure_support_floor_crosses_leader_within_band']}**.\n")
    lines.append("## Route implication\n")
    lines.append("The calibrated family-specific model does not support spending A01's next full two-seed slot on a pure tokenizer support-floor interpolation alone. The evidence supports the current single-seed 12×384 depth test as a distinct factor, and if depth alone does not cross, the strongest next route should combine a support-floored tokenizer with another leader factor (masking curriculum or faithfully implemented sequence curriculum) rather than run minfreq25/minfreq50 as an isolated pair.\n")
    lines.append(f"JSON: `{OUT_JSON}`")
    lines.append(f"CSV: `{OUT_CSV}`")
    NOTE.write_text("\n".join(lines) + "\n")

    print(json.dumps({
        "status": payload["status"],
        "anchor_self_check_ok": anchor_ok,
        "best_central_candidate": best[1],
        "best_central_overall": round(best[0], 4),
        "pure_support_floor_crosses_within_band": payload["decision"]["pure_support_floor_crosses_leader_within_band"],
        "summary": {k: {kk: round(vv, 4) if isinstance(vv, float) else vv for kk, vv in val.items()} for k, val in summary.items()},
        "out_json": str(OUT_JSON),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
