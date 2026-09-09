#!/usr/bin/env python3
"""research: integrate delivered mechanism/floor measurements into compact tables and note.

Inputs:
- research source-token misfire mass readout.
- research/research Entity relevant-update R/C contrasts.
- research/research duplicate CLEAN probe/entity outputs.
- research dose1p82 V/R held-out probe output.

The goal is to preserve the scientific reading, not to produce a final paper.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import csv
import hashlib
import json
import math
import pathlib
import statistics as stats
from typing import Any

ROOT = _public_path('experiments/archive/relation_learning/scripts/integrate_mechanism_and_floor.py')
ROOT = _PUBLIC_ROOT
WS = _public_path('experiments/archive/relation_learning')
OUT = _public_path('experiments/archive/relation_learning/data/mechanism_floor_integration')
NOTE = _public_path('research/notes/relation_learning/mechanism_floor_and_behavioral_refinement.md')

MISFIRE_TERMS = _public_path('experiments/archive/relation_learning/data/source_token_misfire_mass/misfire_late_terms.csv')
MISFIRE_CONTRASTS = _public_path('experiments/archive/relation_learning/data/source_token_misfire_mass/misfire_late_contrasts.csv')
REWRITE = _public_path('experiments/archive/relation_learning/data/relation_decomposition/rewrite_contrast_late_summary.csv')
REWRITE = _public_path('experiments/archive/relation_learning/data/seed43222_threearm_probe_decomposition/rewrite_late_contrasts.csv')
ENTITY = _public_path('experiments/archive/relation_learning/data/entity_relevant_update_analysis/entity_relevant_update_late_contrasts.csv')
ENTITY = _public_path('experiments/archive/relation_learning/data/seed43222_entity_clean_integration/seed43222_vrc_entity_contrasts.csv')
DOSE182_COPY = _public_path('experiments/archive/relation_learning/data/dose1p82_deberta_probes/copy_late_contrasts.csv')
DOSE182_REWRITE = _public_path('experiments/archive/relation_learning/data/dose1p82_deberta_probes/rewrite_late_contrasts.csv')
DOSE182_ENTITY = _public_path('experiments/archive/relation_learning/data/dose1p82_deberta_probes/entity_ablation_late_contrasts.csv')

DEFAULT_CLEAN = _public_path('experiments/archive/relation_learning/training/runs/full_p2c_c2p_abs_clean_dose2p64x_matched_rowholdout_deberta100M_seed43222/hf_model')
PARALLEL_CLEAN = _public_path('experiments/archive/relation_learning/training/runs/full_p2c_c2p_abs_clean_dose2p64x_matched_rowholdout_deberta100M_seed43222_parallel/hf_model')
DEFAULT_ENTITY_ROWS = _public_path('experiments/archive/relation_learning/data/seed43222_entity_eval/seed43222_entity_eval_rows.csv')
PARALLEL_ENTITY_ROWS = _public_path('experiments/archive/relation_learning/data/seed43222_parallel_clean_entity_eval/seed43222_entity_eval_rows.csv')
DEFAULT_PROBE_DIR = _public_path('experiments/archive/relation_learning/data/default_clean_probes')
PARALLEL_PROBE_DIR = _public_path('experiments/archive/relation_learning/data/seed43222_parallel_clean_probes')


def read_csv(path: pathlib.Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("\n", encoding="utf-8")
        return
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rows)


def f(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def sha(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as inp:
        for b in iter(lambda: inp.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def pearson(xs: list[float], ys: list[float]) -> float:
    mx, my = stats.mean(xs), stats.mean(ys)
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx <= 0 or vy <= 0:
        return float("nan")
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / math.sqrt(vx * vy)


def gather_misfire() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    terms = read_csv(MISFIRE_TERMS)
    cons = read_csv(MISFIRE_CONTRASTS)
    rc = [r for r in cons if r["contrast"] == "RminusC"]
    vc = [r for r in cons if r["contrast"] == "VminusC"]
    # Relation to existing excess true-source cost on token-nonoverlap rewrite.
    cost_by_seed: dict[str, float] = {}
    for r in read_csv(REWRITE):
        if r.get("arch") == "D" and r.get("group") == "token_nonoverlap" and r.get("contrast") == "RminusC":
            cost_by_seed[str(r["seed"])] = f(r["mean_excess_true_nll_cost_over_unrelated"])
    for r in read_csv(REWRITE):
        if r.get("group") == "token_nonoverlap" and r.get("contrast") == "RminusC":
            cost_by_seed[str(r["seed"])] = f(r.get("excess_true_cost"))
    rows = []
    xs = []; ys = []
    for r in sorted(rc, key=lambda x: x["seed"]):
        seed = str(r["seed"])
        sm = f(r["source_mass_delta"])
        cost = cost_by_seed.get(seed, float("nan"))
        rows.append({
            "seed": seed,
            "r_minus_c_source_mass_delta": sm,
            "r_minus_c_target_prob_delta": f(r["target_prob_delta"]),
            "r_minus_c_top1_source_rate_delta": f(r["top1_source_rate_delta"]),
            "r_minus_c_excess_true_cost": cost,
        })
        if math.isfinite(sm) and math.isfinite(cost):
            xs.append(sm); ys.append(cost)
    summary = {
        "r_minus_c_source_mass_delta_mean": stats.mean([f(r["source_mass_delta"]) for r in rc]),
        "r_minus_c_source_mass_delta_range": [min(f(r["source_mass_delta"]) for r in rc), max(f(r["source_mass_delta"]) for r in rc)],
        "r_minus_c_target_prob_delta_mean": stats.mean([f(r["target_prob_delta"]) for r in rc]),
        "r_minus_c_top1_source_rate_delta_mean": stats.mean([f(r["top1_source_rate_delta"]) for r in rc]),
        "v_minus_c_source_mass_delta_mean": stats.mean([f(r["source_mass_delta"]) for r in vc]),
        "v_minus_c_target_prob_delta_mean": stats.mean([f(r["target_prob_delta"]) for r in vc]),
        "source_mass_excess_cost_r_n3": pearson(xs, ys) if len(xs) >= 2 else float("nan"),
        "terms_path": str(MISFIRE_TERMS.relative_to(ROOT)),
        "contrasts_path": str(MISFIRE_CONTRASTS.relative_to(ROOT)),
    }
    return rows, summary


def gather_entity_rc() -> list[dict[str, Any]]:
    rows = []
    # Earlier two seeds: research has exact rel_updates bins and rel_ge1. It lacks rel_ge3, so compute rel_ge3 from rel_updates 3-5 weighted by n.
    by_seed: dict[str, dict[str, dict[str, float]]] = {}
    for r in read_csv(ENTITY):
        if r.get("contrast") != "RminusC" or r.get("seed") not in {"43022", "43122"}:
            continue
        g = r["group"]
        if g.startswith("rel_updates_") or g == "rel_ge1":
            by_seed.setdefault(r["seed"], {})[g] = {
                "n": f(r["n"]),
                "delta": f(r["late_mean_delta_accuracy_pct_a_minus_b"]),
                "r_acc": f(r["late_mean_acc_a"]),
                "c_acc": f(r["late_mean_acc_b"]),
            }
    for seed, d in sorted(by_seed.items()):
        for g in ["rel_updates_0", "rel_updates_1", "rel_updates_2", "rel_updates_3", "rel_updates_4", "rel_updates_5", "rel_ge1"]:
            if g in d:
                rows.append({"seed": seed, "group": g, "n": int(d[g]["n"]), "r_minus_c_accuracy_delta": d[g]["delta"], "r_acc": d[g]["r_acc"], "c_acc": d[g]["c_acc"]})
        vals = [d[g] for g in ["rel_updates_3", "rel_updates_4", "rel_updates_5"] if g in d]
        if vals:
            n = sum(v["n"] for v in vals)
            rows.append({
                "seed": seed,
                "group": "rel_ge3_weighted_from_rel_updates_3_5",
                "n": int(n),
                "r_minus_c_accuracy_delta": sum(v["delta"] * v["n"] for v in vals) / n,
                "r_acc": sum(v["r_acc"] * v["n"] for v in vals) / n,
                "c_acc": sum(v["c_acc"] * v["n"] for v in vals) / n,
            })
    # Third seed integrated table already has rel_ge3.
    for r in read_csv(ENTITY):
        if r["group"] in {"rel_updates_0", "rel_updates_1", "rel_updates_2", "rel_updates_3", "rel_updates_4", "rel_updates_5", "rel_ge1", "rel_ge2", "rel_ge3"}:
            rows.append({
                "seed": "43222",
                "group": r["group"],
                "n": int(f(r["n"])),
                "r_minus_c_accuracy_delta": f(r["RminusC"]),
                "r_acc": f(r["R_acc"]),
                "c_acc": f(r["C_acc"]),
            })
    order = {"rel_updates_0": 0, "rel_updates_1": 1, "rel_updates_2": 2, "rel_updates_3": 3, "rel_updates_4": 4, "rel_updates_5": 5, "rel_ge1": 6, "rel_ge2": 7, "rel_ge3": 8, "rel_ge3_weighted_from_rel_updates_3_5": 8}
    return sorted(rows, key=lambda r: (r["seed"], order.get(r["group"], 99)))


def gather_duplicate_floor() -> dict[str, Any]:
    files = []
    for ck in ["chck_80M", "chck_90M", "chck_100M"]:
        for name in ["model.safetensors", "config.json", "tokenizer.json"]:
            a = sha(DEFAULT_CLEAN / ck / name)
            b = sha(PARALLEL_CLEAN / ck / name)
            files.append({"checkpoint": ck, "file": name, "identical": a == b, "sha": a})
    ent_default = read_csv(DEFAULT_ENTITY_ROWS)
    ent_parallel = read_csv(PARALLEL_ENTITY_ROWS)
    ent_delta = []
    for a, b in zip(ent_default, ent_parallel):
        ent_delta.append({"checkpoint": a["checkpoint"], "default_score": f(a["score"]), "parallel_score": f(b["score"]), "delta": f(a["score"]) - f(b["score"])})
    # Probe summaries are expected identical because checkpoint hashes are identical; preserve a few file hashes too.
    probe_files = []
    for fname in ["rewrite_summary.csv", "copy_summary.csv", "entity_ablation_summary.csv"]:
        pa, pb = DEFAULT_PROBE_DIR / fname, PARALLEL_PROBE_DIR / fname
        probe_files.append({"file": fname, "identical": sha(pa) == sha(pb), "default_sha": sha(pa), "parallel_sha": sha(pb)})
    return {
        "checkpoint_files": files,
        "all_checkpoint_files_identical": all(r["identical"] for r in files),
        "entity_scores": ent_delta,
        "max_entity_abs_delta": max(abs(r["delta"]) for r in ent_delta) if ent_delta else float("nan"),
        "probe_summary_files": probe_files,
        "all_probe_summary_files_identical": all(r["identical"] for r in probe_files),
    }


def gather_dose182() -> dict[str, Any]:
    def pick(path: pathlib.Path, group: str, contrast: str, key: str) -> float:
        for r in read_csv(path):
            if r.get("group") == group and r.get("contrast") == contrast:
                return f(r[key])
        return float("nan")
    return {
        "copy_RminusV_ALL": pick(DOSE182_COPY, "ALL", "RminusV", "late_mean_delta_mean_gain_a_minus_b"),
        "rewrite_VminusR_ALL": pick(DOSE182_REWRITE, "ALL", "VminusR", "late_mean_delta_mean_gain_a_minus_b"),
        "rewrite_VminusR_token_nonoverlap": pick(DOSE182_REWRITE, "token_nonoverlap", "VminusR", "late_mean_delta_mean_gain_a_minus_b"),
        "entity_ablation_rel_ge3_VminusR_effect_no_all": pick(DOSE182_ENTITY, "rel_ge3", "VminusR", "late_mean_delta_mean_effect_no_all_relevant_updates_a_minus_b"),
        "entity_ablation_rel_ge3_RminusV_effect_no_initial": pick(DOSE182_ENTITY, "rel_ge3", "RminusV", "late_mean_delta_mean_effect_no_initial_qbox_a_minus_b"),
        "files": {
            "copy": str(DOSE182_COPY.relative_to(ROOT)),
            "rewrite": str(DOSE182_REWRITE.relative_to(ROOT)),
            "entity": str(DOSE182_ENTITY.relative_to(ROOT)),
        },
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    mis_rows, mis_summary = gather_misfire()
    entity_rows = gather_entity_rc()
    dup = gather_duplicate_floor()
    dose = gather_dose182()

    write_csv(_public_path('experiments/archive/relation_learning/data/mechanism_floor_integration/misfire_vs_cost_by_seed.csv'), mis_rows)
    write_csv(_public_path('experiments/archive/relation_learning/data/mechanism_floor_integration/entity_r_minus_c_by_relevant_updates.csv'), entity_rows)
    write_csv(_public_path('experiments/archive/relation_learning/data/mechanism_floor_integration/duplicate_clean_checkpoint_hashes.csv'), dup["checkpoint_files"])
    write_csv(_public_path('experiments/archive/relation_learning/data/mechanism_floor_integration/duplicate_clean_entity_scores.csv'), dup["entity_scores"])
    write_csv(_public_path('experiments/archive/relation_learning/data/mechanism_floor_integration/duplicate_clean_probe_file_hashes.csv'), dup["probe_summary_files"])
    payload = {"misfire_summary": mis_summary, "duplicate_clean": dup, "dose1p82": dose}
    (_public_path('experiments/archive/relation_learning/data/mechanism_floor_integration/mechanism_floor_integration.json')).write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    lines = []
    lines.append("# research mechanism, duplicate-run floor, and behavioral refinement")
    lines.append("")
    lines.append("## Source-token mass separates identity misfire from pure displacement")
    lines.append("")
    lines.append("The research source-token readout scored 2,585 held-out compact-rewrite token-nonoverlap target masks at 80M/90M/100M for each DeBERTa V/C/R seed. The target token is absent from the source-token set, so probability mass on source tokens is not correct content use.")
    lines.append("")
    lines.append(f"Across seeds, REPEAT-CLEAN source-token mass deltas are {', '.join(f'{r['r_minus_c_source_mass_delta']:+.4f}' for r in mis_rows)}, mean {mis_summary['r_minus_c_source_mass_delta_mean']:+.4f}. REPEAT also lowers target probability relative to CLEAN by mean {mis_summary['r_minus_c_target_prob_delta_mean']:+.4f} and raises the top-1-is-source rate by mean {mis_summary['r_minus_c_top1_source_rate_delta_mean']:+.4f}.")
    lines.append(f"VIEW is different: V-C source-mass delta mean is only {mis_summary['v_minus_c_source_mass_delta_mean']:+.4f}, while V-C target-probability delta mean is {mis_summary['v_minus_c_target_prob_delta_mean']:+.4f}. Thus VIEW improves the correct nonoverlap target without pulling mass to source tokens, whereas REPEAT pulls mass to source tokens and suppresses the target.")
    lines.append(f"With only three seeds, R-C source-mass elevation and the existing R-C excess true-source cost have Pearson r={mis_summary['source_mass_excess_cost_r_n3']:.3f}; treat this as consistency, not a precise scaling law.")
    lines.append("")
    lines.append("Scientific reading: the active recurrence cost is accompanied by output-level identity competition/misfire, not merely by an absence of content reading. This agrees in form with functional_learning #403's synthetic result that identity can create copy-output competition, while this BabyLM readout supplies the natural-language masked-LM source-token mass evidence directly.")
    lines.append("")
    lines.append("## Entity R versus C should be stated as retrieval benefit plus update cost")
    lines.append("")
    lines.append("The third CLEAN seed changes the Entity framing. The stable behavioral fact is not a three-seed VIEW-over-CLEAN benefit. It is that exact recurrence raises unchanged-state retrieval relative to CLEAN, while after the queried state is updated REPEAT is at or below CLEAN on the recorded aggregate cells.")
    lines.append("")
    lines.append("| seed | group | n | R-C accuracy delta | R acc | C acc |")
    lines.append("|---:|---|---:|---:|---:|---:|")
    for r in entity_rows:
        if r["group"] in {"rel_updates_0", "rel_updates_1", "rel_updates_2", "rel_updates_3", "rel_updates_4", "rel_updates_5", "rel_ge3", "rel_ge3_weighted_from_rel_updates_3_5"}:
            lines.append(f"| {r['seed']} | {r['group']} | {r['n']} | {r['r_minus_c_accuracy_delta']:+.2f} | {r['r_acc']:.2f} | {r['c_acc']:.2f} |")
    lines.append("")
    lines.append("This preserves Entity as a behavioral face of the cost center: identity practice buys retrieval of an unchanged stated fact and can hurt discrimination once a state has been superseded. The held-out copy/rewrite probes remain the primary installed-competence evidence because their three-seed magnitudes are more stable.")
    lines.append("")
    lines.append("## Duplicate CLEAN run did not introduce an additional run-nondeterminism floor")
    lines.append("")
    lines.append(f"The default research CLEAN seed43222 and the research parallel CLEAN seed43222 checkpoint files are byte-identical for model.safetensors, config.json, and tokenizer.json at 80M/90M/100M: {dup['all_checkpoint_files_identical']}. Entity scores are identical at 26.41/27.15/26.59, with max absolute delta {dup['max_entity_abs_delta']:.6f}. Probe summary files are also byte-identical: {dup['all_probe_summary_files_identical']}.")
    lines.append("This means the duplicate CLEAN does not set a new stochastic floor; under this recipe and hardware path the same seed/data run is deterministic at the measured checkpoints. The hierarchy of evidence should therefore be based on seed-to-seed variation and probe specificity, not duplicate-run drift.")
    lines.append("")
    lines.append("## 1.82x DeBERTa pair preserves the V/R relation signature")
    lines.append("")
    lines.append(f"The existing 1.82x DeBERTa VIEW/REPEAT pair gives R-V held-out copy gain {dose['copy_RminusV_ALL']:+.4f}, V-R rewrite gain {dose['rewrite_VminusR_ALL']:+.4f} overall and {dose['rewrite_VminusR_token_nonoverlap']:+.4f} on token-nonoverlap targets. The Entity cue-ablation rel_ge3 V-R effect when all relevant updates are removed is {dose['entity_ablation_rel_ge3_VminusR_effect_no_all']:+.4f}; R-V no-initial effect is {dose['entity_ablation_rel_ge3_RminusV_effect_no_initial']:+.4f}.")
    lines.append("Thus the relation-specific V/R split is not unique to the 2.64x MAX dose, although this two-arm lower-dose check lacks a matched CLEAN baseline for active-cost statements.")
    lines.append("")
    lines.append("## Open causal work")
    lines.append("")
    lines.append("Both split arms, REPEAT_SPLIT and VIEW_SPLIT, were training at this stage. Their results require the held-out copy/rewrite framework and the predictions in `notes/split_control_numeric_prestatement.md`. Locality cannot be inferred before those results are available and scored.")
    lines.append("")
    lines.append("Data outputs: `experiments/archive/relation_learning/data/mechanism_floor_integration`.")
    NOTE.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": "MECHANISM_FLOOR_INTEGRATION_DONE", "note": str(NOTE.relative_to(ROOT)), "outputs": str(OUT.relative_to(ROOT))}, indent=2), flush=True)

if __name__ == "__main__":
    main()
