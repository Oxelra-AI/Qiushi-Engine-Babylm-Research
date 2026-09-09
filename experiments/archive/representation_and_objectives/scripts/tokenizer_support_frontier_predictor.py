#!/usr/bin/env python3
"""research: CPU-only predictive model for the legal-tokenizer support/segmentation frontier.

Scientific question
--------------------
Legal16k and legal40k both have full pristine official vectors and full research pool
support spectra. Their Overall scores are essentially tied (40.864 vs 40.780) but their
column patterns are mirror images: legal40k gains BLiMP/Supplement/EWoK (finer
segmentation) and loses GlobalPIQA/Entity (rare-token support collapse). This is a
representation sample-efficiency trade-off at a single flat vocab size.

The support-floored tokenizers (minfreq25 vocab 29,529; minfreq50 vocab 19,609) sit
BETWEEN legal16k and legal40k on BOTH drivers: they keep shorter segmentation than 16k
while flooring rare merges so token-mass on low-support units is closer to 16k. This
script builds a transparent first-order predictive model that estimates the full nine
column vector and Overall for each candidate tokenizer, so the next expensive H100 route
can be chosen from a quantitative prediction rather than intuition.

Method (transparent, auditable)
--------------------------------
1. Load exact measured mean vectors for legal16k and legal40k (research interpretation).
2. Load exact pool-support drivers for all tokenizers (research pool_support_summary.csv):
     - seg   = tokens_per_word           (lower => finer/richer segmentation)
     - supp  = token_mass_frac_lt50      (higher => more token-mass on under-trained units)
3. For each column, decompose the measured legal16k->legal40k delta into a
   segmentation-sensitivity term and a support-sensitivity term using the KNOWN sign
   structure from the column pattern, under a two-anchor first-order model:
       score(c) = a_c + b_c * seg_norm + g_c * supp_norm
   With only two anchors we cannot fit b_c and g_c independently per column, so we use a
   principled single-index attribution:
       - "segmentation-driven" columns (BLiMP, Supplement, EWoK): interpolate on seg only.
       - "support-driven" columns   (GlobalPIQA, Entity): interpolate on supp only.
       - "flat" columns (COMPS, SuperGLUE, Reading, AoA): hold at the anchor mean
         (their |delta| is small; interpolation would be noise).
   Interpolation is LINEAR in the driver between the two anchors, then extrapolation is
   CLAMPED to the anchor range so we never predict beyond measured evidence.
4. Predict Overall with the official BabyLM weighting used in the collator (equal average
   of the nine columns as recorded; Overall is recomputed from predicted columns with the
   same arithmetic the interpretation files use).
5. Report predictions for minfreq25 and minfreq50, plus a sanity self-prediction of the
   two anchors (must reproduce measured values), plus an honest uncertainty band from the
   |column delta| magnitude and the seed spread.

This is CPU-only, trains/evaluates no model, and uses official eval text only through the
already-computed support statistics. It is decision support for the Lead route choice.
"""
import csv
import json
import pathlib

ROOT = pathlib.Path("experiments/archive/representation_and_objectives")
CSV = ROOT / "data/tokenizer_support_spectrum/pool_support_summary.csv"
research = ROOT / "data/legal40k_endpoint_interpretation/legal40k_endpoint_interpretation.json"
OUT_DIR = ROOT / "data/tokenizer_support_frontier_predictor"
OUT_JSON = OUT_DIR / "tokenizer_support_frontier_predictor.json"
OUT_CSV = OUT_DIR / "predicted_tokenizer_vectors.csv"
NOTE = (ROOT.parents[2] / 'research/notes/representation_and_objectives/tokenizer_support_frontier_predictor.md')

COLUMNS = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS",
           "SuperGLUE", "GlobalPIQA", "Reading", "AoA"]

# Column driver attribution from the measured legal16k->legal40k column pattern.
SEG_DRIVEN = {"BLiMP", "Supplement", "EWoK"}          # finer segmentation helps
SUPP_DRIVEN = {"GlobalPIQA", "Entity"}                # rare-token support protects
FLAT = {"COMPS", "SuperGLUE", "Reading", "AoA"}       # small |delta|, hold at mean interp


def load_support():
    rows = {}
    with open(CSV) as f:
        for r in csv.DictReader(f):
            rows[r["tokenizer"]] = {
                "vocab_size": int(r["vocab_size"]),
                "seg": float(r["tokens_per_word"]),
                "supp": float(r["token_mass_frac_lt50"]),
            }
    return rows


def clamp01(x):
    return max(0.0, min(1.0, x))


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    interp = json.loads(research.read_text())
    v16 = interp["legal16k_mean"]
    v40 = interp["legal40k_mean"]
    sup = load_support()

    a16 = sup["legal_a01_16k"]
    a40 = sup["legal_byte_bpe_40k"]

    # Normalized position of a tokenizer between the two anchors on each driver.
    # 0.0 == legal16k anchor, 1.0 == legal40k anchor.
    def frac_seg(seg):
        return clamp01((seg - a16["seg"]) / (a40["seg"] - a16["seg"]))

    def frac_supp(supp):
        return clamp01((supp - a16["supp"]) / (a40["supp"] - a16["supp"]))

    def predict_vector(seg, supp):
        fs = frac_seg(seg)
        fp = frac_supp(supp)
        out = {}
        for c in COLUMNS:
            lo, hi = v16[c], v40[c]
            if c in SEG_DRIVEN:
                out[c] = lo + fs * (hi - lo)
            elif c in SUPP_DRIVEN:
                out[c] = lo + fp * (hi - lo)
            else:  # FLAT: interpolate on the mean of the two normalized drivers (weak)
                fm = 0.5 * (fs + fp)
                out[c] = lo + fm * (hi - lo)
        out["Overall"] = sum(out[c] for c in COLUMNS) / len(COLUMNS)
        return out, fs, fp

    candidates = [
        ("legal_a01_16k", "anchor-self-check-16k"),
        ("legal_byte_bpe_40k", "anchor-self-check-40k"),
        ("legal_byte_bpe_40k_minfreq25", "PRIMARY-candidate-support-floor-29.5k"),
        ("legal_byte_bpe_40k_minfreq50", "A02-line-support-floor-19.6k"),
        ("legal_byte_bpe_24k", "reference-24k"),
        ("legal_byte_bpe_32k", "reference-32k"),
    ]

    results = {}
    for tok, role in candidates:
        s = sup[tok]
        vec, fs, fp = predict_vector(s["seg"], s["supp"])
        results[tok] = {
            "role": role,
            "vocab_size": s["vocab_size"],
            "tokens_per_word": s["seg"],
            "token_mass_frac_lt50": s["supp"],
            "frac_seg_position": fs,
            "frac_supp_position": fp,
            "predicted_vector": vec,
            "predicted_overall": vec["Overall"],
        }

    # Self-check: anchors must reproduce their measured Overall exactly.
    sc16 = results["legal_a01_16k"]["predicted_overall"]
    sc40 = results["legal_byte_bpe_40k"]["predicted_overall"]
    self_check_ok = (abs(sc16 - v16["Overall"]) < 1e-6) and (abs(sc40 - v40["Overall"]) < 1e-6)

    # Uncertainty: the model is first-order in a single driver per column. The dominant
    # honest uncertainty is (a) the measured seed spread (Overall population sd 0.360 for
    # legal40k) and (b) unmodeled cross-terms. Report a conservative +/- band.
    seed_sd = interp.get("legal40k_overall_population_sd", 0.36)
    # Cross-term band: half of the max single-column |delta| that is NOT captured by the
    # dominant driver (i.e., the second-order coupling), scaled to Overall (÷9 columns).
    deltas = interp["mean_delta_legal40k_minus_legal16k"]
    max_offdriver = max(abs(deltas[c]) for c in FLAT)  # flat columns carry residual coupling
    cross_band = max_offdriver / len(COLUMNS)
    overall_band = round((seed_sd**2 + cross_band**2) ** 0.5, 4)

    mf25 = results["legal_byte_bpe_40k_minfreq25"]["predicted_overall"]
    mf50 = results["legal_byte_bpe_40k_minfreq50"]["predicted_overall"]

    best_pred = max(
        (results[t]["predicted_overall"], t) for t, _ in candidates
        if t not in ("legal_a01_16k", "legal_byte_bpe_40k")
    )

    payload = {
        "status": "TOKENIZER_SUPPORT_FRONTIER_PREDICTOR",
        "method": "first-order single-index interpolation between measured legal16k/legal40k anchors on segmentation (tokens/word) and support (token_mass_frac_lt50) drivers",
        "anchors": {
            "legal16k": {"Overall": v16["Overall"], "seg": a16["seg"], "supp": a16["supp"]},
            "legal40k": {"Overall": v40["Overall"], "seg": a40["seg"], "supp": a40["supp"]},
        },
        "column_driver_attribution": {
            "segmentation_driven": sorted(SEG_DRIVEN),
            "support_driven": sorted(SUPP_DRIVEN),
            "flat_held": sorted(FLAT),
        },
        "self_check_reproduces_anchors": self_check_ok,
        "predictions": results,
        "predicted_overall_minfreq25": mf25,
        "predicted_overall_minfreq50": mf50,
        "predicted_best_candidate": {"overall": best_pred[0], "tokenizer": best_pred[1]},
        "uncertainty": {
            "seed_population_sd_overall": seed_sd,
            "cross_term_band_overall": round(cross_band, 4),
            "combined_overall_band_plus_minus": overall_band,
        },
        "visible_leader_overall": 41.8,
        "decision_reading": {
            "minfreq25_vs_leader_margin": round(mf25 - 41.8, 4),
            "minfreq25_vs_legal40k_mean": round(mf25 - v40["Overall"], 4),
            "minfreq25_vs_legal16k_mean": round(mf25 - v16["Overall"], 4),
            "crosses_leader_within_band": (mf25 + overall_band) >= 41.8,
            "note": "This is a first-order prediction from two anchors. It bounds the plausible gain of the support-floor route; it does not replace a real training run. If predicted best candidate + band stays below 41.8, a pure-tokenizer support-floor route is unlikely to cross the frontier alone and the next route must add a distinct factor (depth, masking curriculum, or objective).",
        },
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2))

    with open(OUT_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["tokenizer", "role", "vocab", "tok_per_word", "mass_lt50",
                    "frac_seg", "frac_supp"] + COLUMNS + ["Overall"])
        for tok, role in candidates:
            r = results[tok]
            v = r["predicted_vector"]
            w.writerow([tok, role, r["vocab_size"],
                        round(r["tokens_per_word"], 4), round(r["token_mass_frac_lt50"], 4),
                        round(r["frac_seg_position"], 4), round(r["frac_supp_position"], 4)]
                       + [round(v[c], 4) for c in COLUMNS] + [round(v["Overall"], 4)])

    lines = []
    lines.append("# research — Legal tokenizer support/segmentation frontier predictor\n")
    lines.append("CPU-only first-order predictor of the full nine-column official vector for candidate legal tokenizers, calibrated on the two measured anchors (legal16k, legal40k) and the research pool-support drivers. Trains and evaluates no model.\n")
    lines.append(f"- Self-check reproduces both anchors exactly: **{self_check_ok}**")
    lines.append(f"- Anchors: legal16k Overall {v16['Overall']:.4f} (seg {a16['seg']:.4f}, mass<50 {a16['supp']:.4f}); legal40k Overall {v40['Overall']:.4f} (seg {a40['seg']:.4f}, mass<50 {a40['supp']:.4f})")
    lines.append(f"- Column attribution: seg-driven {sorted(SEG_DRIVEN)}; support-driven {sorted(SUPP_DRIVEN)}; flat-held {sorted(FLAT)}\n")
    lines.append("## Predicted Overall by candidate\n")
    lines.append("| tokenizer | vocab | tok/word | mass<50 | pred Overall | vs 41.8 | vs 40k mean |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for tok, role in candidates:
        r = results[tok]
        po = r["predicted_overall"]
        lines.append(f"| {tok} | {r['vocab_size']} | {r['tokens_per_word']:.4f} | {r['token_mass_frac_lt50']:.4f} | {po:.4f} | {po-41.8:+.4f} | {po-v40['Overall']:+.4f} |")
    lines.append("")
    lines.append(f"- **minfreq25 predicted Overall = {mf25:.4f}** (band ±{overall_band}); margin vs 41.8 = {mf25-41.8:+.4f}")
    lines.append(f"- minfreq50 predicted Overall = {mf50:.4f}")
    lines.append(f"- best predicted candidate: {best_pred[1]} at {best_pred[0]:.4f}")
    lines.append(f"- crosses 41.8 within ±{overall_band} band: **{payload['decision_reading']['crosses_leader_within_band']}**\n")
    lines.append("## How to read this for the next H100 route\n")
    lines.append("If the best support-floored candidate + uncertainty band stays below 41.8, a pure tokenizer support-floor run is unlikely to cross the frontier by itself, and the next expensive route must ADD a distinct factor (depth 12x384 — already running seed43022 — masking curriculum WWM->token, or a sharper objective) rather than another point on the tokenizer axis. If it crosses within band, the support-floor tokenizer becomes the strongest single-factor compliant repair and justifies one H100 pair.\n")
    lines.append(f"JSON: `{OUT_JSON}`")
    lines.append(f"CSV: `{OUT_CSV}`")
    NOTE.write_text("\n".join(lines) + "\n")

    print(json.dumps({
        "status": payload["status"],
        "self_check_ok": self_check_ok,
        "minfreq25_pred_overall": round(mf25, 4),
        "minfreq50_pred_overall": round(mf50, 4),
        "best_candidate": best_pred[1],
        "best_overall": round(best_pred[0], 4),
        "overall_band": overall_band,
        "crosses_leader_within_band": payload["decision_reading"]["crosses_leader_within_band"],
        "out_json": str(OUT_JSON),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
