#!/usr/bin/env python3
"""research: fully same-coordinate 2x2 (treatment x seed) on the pristine
official coordinate.

Cells: clean x {43022,43122}, reinvest x {43022,43122}.

Coordinate consistency (matching reinvest research/research pristine collation):
  - Overall = arithmetic mean of nine leaderboard columns
      (BLiMP, Supplement, EWoK, Entity, COMPS, SuperGLUE, GlobalPIQA, Reading, AoA_leaderboard)
  - EWoK: pristine 7618-row official for both clean and reinvest cells
  - SuperGLUE: leaderboard PRIMARY-METRIC convention
      f1 for MRPC/QQP, accuracy for boolq/mnli/multirc/rte/wsc; mean*100
  - AoA: official min_context=0 -> all four cells report leaderboard 0.0
  - BLiMP/Supplement/Entity/COMPS/GlobalPIQA/Reading: from each cell's official
    full vector; clean zero-shot data is identical across the corrections we make,
    so clean cells reuse COMPACT_EXPERIENCE research columns except EWoK+SuperGLUE which are
    replaced with the same-coordinate versions.

CPU-only arithmetic over existing evidence. No training, no evaluation.
"""
import json
import pathlib

SESS = pathlib.Path("experiments/archive/representation_and_objectives")
COMPACT_EXPERIENCE = pathlib.Path("experiments/archive/compact_experience")
A02 = pathlib.Path("experiments/archive/frontier_consolidation")

NINE = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "SuperGLUE", "GlobalPIQA", "Reading", "AoA"]

SG_PRIMARY = {
    "boolq": "accuracy", "mnli": "accuracy", "multirc": "accuracy",
    "rte": "accuracy", "wsc": "accuracy", "mrpc": "f1", "qqp": "f1",
}
SG_TASKS = ["boolq", "mnli", "multirc", "rte", "wsc", "mrpc", "qqp"]


def parse_results_txt(text):
    d = {}
    for line in text.strip().splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            d[k.strip()] = float(v.strip())
    return d


def clean_superglue_primary(seed_dir_name):
    base = (COMPACT_EXPERIENCE / "data" / "full_eval" / "superglue_results"
            / seed_dir_name)
    vals = {}
    for task in SG_TASKS:
        p = base / task / "chck_100M" / "main" / "finetune" / task / "results.txt"
        d = parse_results_txt(p.read_text())
        vals[task] = d[SG_PRIMARY[task]]
    mean = sum(vals.values()) / len(vals) * 100.0
    return mean, {t: round(vals[t] * 100.0, 4) for t in SG_TASKS}


def load(p):
    return json.loads(pathlib.Path(p).read_text())


def overall(cols):
    return sum(cols[k] for k in NINE) / 9.0


def main():
    out_dir = SESS / "data" / "consistent_official_2x2"
    out_dir.mkdir(parents=True, exist_ok=True)

    # reinvest cells (already on pristine official coordinate)
    r430 = load(SESS / "data/pristine_collate_seed43022/pristine_collate_seed43022_summary.json")
    r431 = load(SESS / "data/pristine_collate_seed43122/pristine_collate_seed43122_summary.json")
    reinvest = {
        "43022": dict(r430["score_summary"]["official_overall"]["scores"]),
        "43122": dict(r431["score_summary"]["scores"]),
    }

    # clean cells: COMPACT_EXPERIENCE columns except same-coordinate EWoK + SuperGLUE
    s028 = load(COMPACT_EXPERIENCE / "data/full_eval/full_eval_summary.json")
    c430_src = s028["targets"]["qwen_clean_aligned"]
    c431_src = s028["targets"]["qwen_clean_aligned_seed43122"]

    ewok_2x2 = load(A02 / "data/official_ewok_2x2_domain_did/official_ewok_2x2_domain_did.json")
    clean_ewok = {
        "43022": ewok_2x2["official_macro_over_domains"]["clean_43022"],
        "43122": ewok_2x2["official_macro_over_domains"]["clean_43122"],
    }
    sg430, sg430_tasks = clean_superglue_primary("qwen_clean_aligned")
    sg431, sg431_tasks = clean_superglue_primary("qwen_clean_aligned_seed43122")

    def build_clean(src, seed, sg):
        return {
            "BLiMP": src["BLiMP"], "Supplement": src["Supplement"],
            "EWoK": clean_ewok[seed], "Entity": src["Entity"],
            "COMPS": src["COMPS"], "SuperGLUE": sg,
            "GlobalPIQA": src["GlobalPIQA"], "Reading": src["Reading"],
            "AoA": src["AoA"],
        }

    clean = {
        "43022": build_clean(c430_src, "43022", sg430),
        "43122": build_clean(c431_src, "43122", sg431),
    }

    cells = {
        "clean_43022": clean["43022"], "clean_43122": clean["43122"],
        "reinvest_43022": reinvest["43022"], "reinvest_43122": reinvest["43122"],
    }
    overalls = {k: round(overall(v), 6) for k, v in cells.items()}

    # treatment effect within seed (reinvest - clean)
    def col_delta(a, b):
        return {k: round(a[k] - b[k], 6) for k in NINE}

    TE_43022 = col_delta(reinvest["43022"], clean["43022"])
    TE_43122 = col_delta(reinvest["43122"], clean["43122"])
    TE_43022_overall = round(overalls["reinvest_43022"] - overalls["clean_43022"], 6)
    TE_43122_overall = round(overalls["reinvest_43122"] - overalls["clean_43122"], 6)

    # seed spread within treatment (43122 - 43022)
    seed_clean = col_delta(clean["43122"], clean["43022"])
    seed_reinvest = col_delta(reinvest["43122"], reinvest["43022"])
    seed_clean_overall = round(overalls["clean_43122"] - overalls["clean_43022"], 6)
    seed_reinvest_overall = round(overalls["reinvest_43122"] - overalls["reinvest_43022"], 6)

    # DiD = TE_43122 - TE_43022 (how treatment effect changes across seed)
    DiD = {k: round(TE_43122[k] - TE_43022[k], 6) for k in NINE}
    DiD_overall = round(TE_43122_overall - TE_43022_overall, 6)

    # average treatment effect across two seeds
    ATE_overall = round((TE_43022_overall + TE_43122_overall) / 2.0, 6)
    ATE_cols = {k: round((TE_43022[k] + TE_43122[k]) / 2.0, 6) for k in NINE}

    result = {
        "status": "CONSISTENT_OFFICIAL_2x2",
        "coordinate": {
            "overall_formula": "mean of nine leaderboard columns",
            "ewok": "pristine 7618-row official for all cells",
            "superglue": "primary-metric convention f1(MRPC,QQP)/accuracy(others), mean*100",
            "aoa": "official min_context=0; all cells 0.0",
            "clean_columns_source": "COMPACT_EXPERIENCE research zero-shot (BLiMP/Supplement/Entity/COMPS/GlobalPIQA/Reading), EWoK+SuperGLUE replaced on-coordinate",
            "reinvest_columns_source": "research/research pristine collation",
        },
        "cells": cells,
        "overalls": overalls,
        "clean_superglue_primary": {"43022": round(sg430, 6), "43122": round(sg431, 6),
                                    "43022_tasks": sg430_tasks, "43122_tasks": sg431_tasks},
        "clean_ewok_pristine": clean_ewok,
        "treatment_effect_within_seed": {
            "TE_43022_overall": TE_43022_overall, "TE_43022_cols": TE_43022,
            "TE_43122_overall": TE_43122_overall, "TE_43122_cols": TE_43122,
            "ATE_overall_two_seeds": ATE_overall, "ATE_cols": ATE_cols,
        },
        "seed_spread_within_treatment": {
            "clean_43122_minus_43022_overall": seed_clean_overall, "clean_cols": seed_clean,
            "reinvest_43122_minus_43022_overall": seed_reinvest_overall, "reinvest_cols": seed_reinvest,
        },
        "DiD_treatment_x_seed": {"DiD_overall": DiD_overall, "DiD_cols": DiD},
        "leader_reference": 41.8,
        "margins_over_leader": {k: round(v - 41.8, 6) for k, v in overalls.items()},
        "interpretation": {
            "endpoint_robustness": "Only reinvest_43022 exceeds the visible 41.8 leader; reinvest_43122 is 41.248 (below).",
            "treatment_replication": "See TE_43022_overall vs TE_43122_overall; positive at both seeds means the compact-view principle helps regardless of seed even if absolute 43122 endpoint is below leader.",
            "seed_variance_partition": "Compare clean seed spread vs reinvest seed spread to see whether treatment amplifies or dampens seed sensitivity.",
        },
    }

    out_json = out_dir / "consistent_official_2x2.json"
    out_json.write_text(json.dumps(result, indent=2))
    print(json.dumps({
        "overalls": overalls,
        "margins_over_leader": result["margins_over_leader"],
        "TE_43022_overall": TE_43022_overall,
        "TE_43122_overall": TE_43122_overall,
        "ATE_overall_two_seeds": ATE_overall,
        "clean_seed_spread_overall": seed_clean_overall,
        "reinvest_seed_spread_overall": seed_reinvest_overall,
        "DiD_overall": DiD_overall,
        "clean_sg_primary": {"43022": round(sg430, 4), "43122": round(sg431, 4)},
        "out_json": str(out_json),
    }, indent=2))


if __name__ == "__main__":
    main()
