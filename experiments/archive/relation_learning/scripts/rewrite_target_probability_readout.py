#!/usr/bin/env python3
"""research direct target-probability readout for compact rewrites.

Uses existing compact rewrite T/U rows. For tokenizer-overlap targets, the target
ID is present in the true source by construction; for nonoverlap targets it is
absent. We compute exp(-true_source_nll) and exp(-unrelated_source_nll) to ask
whether REPEAT actually raises probability on the correct source-present token in
nonidentical rewrite contexts, or instead mainly moves diffuse mass onto source
content while target probability stays below CLEAN.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import csv
import json
import math
import pathlib
import statistics
import time
from collections import defaultdict
from typing import Any

ROOT = _public_path('experiments/archive/relation_learning/scripts/rewrite_target_probability_readout.py')
ROOT = _PUBLIC_ROOT
WS = _public_path('experiments/archive/relation_learning')
SOURCES = [
    _public_path('experiments/archive/relation_learning/data/heldout_copy_rewrite_entity_ablation/rewrite_pair_rows.csv'),  # 43022/43122 V/C/R
    _public_path('experiments/archive/relation_learning/data/seed43222_probes/rewrite_pair_rows.csv'),  # 43222 V/R
    _public_path('experiments/archive/relation_learning/data/seed43222_parallel_clean_probes/rewrite_pair_rows.csv'),  # 43222 C
]
OUT_DIR = _public_path('experiments/archive/relation_learning/data/integrated_scope_and_mechanism')
NOTE = _public_path('research/notes/relation_learning/rewrite_target_probability_readout.md')
CKS = {"chck_80M", "chck_90M", "chck_100M"}
ROLES = {"C", "R", "V"}
SEEDS = {43022, 43122, 43222}


def rel(p: pathlib.Path) -> str:
    try: return str(p.relative_to(ROOT))
    except Exception: return str(p)


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def mean(xs):
    vals = [float(x) for x in xs if math.isfinite(float(x))]
    return statistics.mean(vals) if vals else float("nan")


def sd(xs):
    vals = [float(x) for x in xs if math.isfinite(float(x))]
    return statistics.stdev(vals) if len(vals) >= 2 else float("nan")


def write_csv(path: pathlib.Path, rows: list[dict[str, Any]], preferred: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("\n", encoding="utf-8")
        return
    fields = [f for f in preferred if any(f in r for r in rows)]
    fields += [f for f in sorted(set().union(*(r.keys() for r in rows))) if f not in fields]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rows)


def stream_rows():
    seen = set()
    for path in SOURCES:
        with path.open(encoding="utf-8") as f:
            for r in csv.DictReader(f):
                seed = int(r.get("seed", -1))
                role = str(r.get("role", ""))
                ck = str(r.get("checkpoint", ""))
                token_class = str(r.get("token_class", ""))
                if seed not in SEEDS or role not in ROLES or ck not in CKS or token_class not in {"overlap", "nonoverlap"}:
                    continue
                key = (seed, role, ck, r.get("probe_id"))
                if key in seen:
                    continue
                seen.add(key)
                T = float(r["true_source_nll"])
                U = float(r["unrelated_source_nll"])
                yield {
                    "seed": seed,
                    "role": role,
                    "checkpoint": ck,
                    "probe_id": r["probe_id"],
                    "pair_id": r.get("pair_id", ""),
                    "token_class": token_class,
                    "true_source_nll": T,
                    "unrelated_source_nll": U,
                    "gain": float(r["gain"]),
                    "true_target_prob": math.exp(-T) if T < 80 else 0.0,
                    "unrelated_target_prob": math.exp(-U) if U < 80 else 0.0,
                }


def main() -> None:
    rows = list(stream_rows())
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    bins: dict[tuple[int, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        bins[(r["seed"], r["role"], r["checkpoint"], r["token_class"])].append(r)
    ck_terms = []
    for (seed, role, ck, token_class), vals in sorted(bins.items()):
        ck_terms.append({
            "seed": seed, "role": role, "checkpoint": ck, "token_class": token_class,
            "n": len(vals),
            "mean_true_nll": mean([v["true_source_nll"] for v in vals]),
            "mean_unrelated_nll": mean([v["unrelated_source_nll"] for v in vals]),
            "mean_gain_U_minus_T": mean([v["gain"] for v in vals]),
            "mean_true_target_prob": mean([v["true_target_prob"] for v in vals]),
            "mean_unrelated_target_prob": mean([v["unrelated_target_prob"] for v in vals]),
        })
    late_bins: dict[tuple[int, str, str], list[dict[str, Any]]] = defaultdict(list)
    for r in ck_terms:
        late_bins[(r["seed"], r["role"], r["token_class"])].append(r)
    late_terms = []
    for (seed, role, token_class), vals in sorted(late_bins.items()):
        late_terms.append({
            "seed": seed, "role": role, "token_class": token_class,
            "n_min": min(int(v["n"]) for v in vals),
            "n_checkpoints": len(vals),
            "mean_true_nll": mean([v["mean_true_nll"] for v in vals]),
            "mean_unrelated_nll": mean([v["mean_unrelated_nll"] for v in vals]),
            "mean_gain_U_minus_T": mean([v["mean_gain_U_minus_T"] for v in vals]),
            "mean_true_target_prob": mean([v["mean_true_target_prob"] for v in vals]),
            "mean_unrelated_target_prob": mean([v["mean_unrelated_target_prob"] for v in vals]),
        })
    idx = {(r["seed"], r["role"], r["token_class"]): r for r in late_terms}
    contrasts = []
    for seed in sorted(SEEDS):
        for tc in ["overlap", "nonoverlap"]:
            for cname, a, b in [("RminusC", "R", "C"), ("VminusC", "V", "C"), ("VminusR", "V", "R")]:
                if (seed, a, tc) not in idx or (seed, b, tc) not in idx:
                    continue
                ra, rb = idx[(seed, a, tc)], idx[(seed, b, tc)]
                contrasts.append({
                    "seed": seed, "token_class": tc, "contrast": cname,
                    "n_min": min(int(ra["n_min"]), int(rb["n_min"])),
                    "delta_true_nll": ra["mean_true_nll"] - rb["mean_true_nll"],
                    "delta_unrelated_nll": ra["mean_unrelated_nll"] - rb["mean_unrelated_nll"],
                    "delta_gain_U_minus_T": ra["mean_gain_U_minus_T"] - rb["mean_gain_U_minus_T"],
                    "delta_true_target_prob": ra["mean_true_target_prob"] - rb["mean_true_target_prob"],
                    "delta_unrelated_target_prob": ra["mean_unrelated_target_prob"] - rb["mean_unrelated_target_prob"],
                    "a_true_target_prob": ra["mean_true_target_prob"],
                    "b_true_target_prob": rb["mean_true_target_prob"],
                })
    across = []
    for tc in ["overlap", "nonoverlap"]:
        for cname in ["RminusC", "VminusC", "VminusR"]:
            vals = [r for r in contrasts if r["token_class"] == tc and r["contrast"] == cname]
            if not vals:
                continue
            rec = {"token_class": tc, "contrast": cname, "n_seeds": len(vals)}
            for col in ["delta_true_nll", "delta_unrelated_nll", "delta_gain_U_minus_T", "delta_true_target_prob", "delta_unrelated_target_prob"]:
                xs = [float(v[col]) for v in vals]
                rec[col + "_mean"] = mean(xs)
                rec[col + "_seed_sd"] = sd(xs)
                rec[col + "_signs"] = "/".join("+" if x > 0 else "-" if x < 0 else "0" for x in xs)
            across.append(rec)
    write_csv(_public_path('experiments/archive/relation_learning/data/integrated_scope_and_mechanism/rewrite_target_probability_late_terms.csv'), late_terms,
              ["seed", "role", "token_class", "n_min", "mean_true_nll", "mean_unrelated_nll", "mean_gain_U_minus_T", "mean_true_target_prob", "mean_unrelated_target_prob"])
    write_csv(_public_path('experiments/archive/relation_learning/data/integrated_scope_and_mechanism/rewrite_target_probability_late_contrasts.csv'), contrasts,
              ["seed", "token_class", "contrast", "n_min", "delta_true_nll", "delta_unrelated_nll", "delta_gain_U_minus_T", "delta_true_target_prob", "delta_unrelated_target_prob", "a_true_target_prob", "b_true_target_prob"])
    write_csv(_public_path('experiments/archive/relation_learning/data/integrated_scope_and_mechanism/rewrite_target_probability_across_seed.csv'), across,
              ["token_class", "contrast", "n_seeds", "delta_true_target_prob_mean", "delta_true_target_prob_seed_sd", "delta_true_target_prob_signs", "delta_true_nll_mean", "delta_true_nll_seed_sd", "delta_true_nll_signs", "delta_gain_U_minus_T_mean", "delta_gain_U_minus_T_seed_sd", "delta_gain_U_minus_T_signs"])
    def get(tc: str, cn: str, col: str) -> str:
        for r in across:
            if r["token_class"] == tc and r["contrast"] == cn:
                return f"{float(r[col + '_mean']):+.5f} ± {float(r[col + '_seed_sd']):.5f} ({r[col + '_signs']})"
        return "NA"
    lines = []
    lines.append("# research direct target-probability readout for compact rewrites")
    lines.append("")
    lines.append(f"Created: {now()}")
    lines.append("")
    lines.append("This readout converts existing true-source/unrelated-source NLL rows into target probabilities. For tokenizer-overlap targets the target token ID is present in the true source; for tokenizer-nonoverlap targets it is absent. It tests whether REPEAT's source-content mass elevation corresponds to higher probability on the correct source-present token inside nonidentical rewrite contexts.")
    lines.append("")
    lines.append("## Across-seed contrasts")
    lines.append("")
    lines.append("| token class | contrast | Δ true target prob | Δ true-source NLL | Δ source-conditioned gain U−T |")
    lines.append("|---|---|---:|---:|---:|")
    for tc in ["overlap", "nonoverlap"]:
        for cn in ["RminusC", "VminusC", "VminusR"]:
            lines.append(f"| {tc} | {cn} | {get(tc, cn, 'delta_true_target_prob')} | {get(tc, cn, 'delta_true_nll')} | {get(tc, cn, 'delta_gain_U_minus_T')} |")
    lines.append("")
    lines.append("Reading: in the nonidentical rewrite format, REPEAT does not become a reliable precise answer copier. Even on tokenizer-overlap targets where the answer token is present in the source, R−C has negative source-conditioned gain in all three seeds and its direct true-source target-probability change is small/unstable, not a strong positive copy signal. On nonoverlap targets, R lowers true-source target probability and increases true-source NLL in all three seeds. VIEW raises true-source target probability and source-conditioned gain on both token classes, with the larger gain on nonoverlap tokens. The correct mechanism phrase is therefore a source-recognition trigger whose diffuse source-content pull transfers to related spans, while precise answer copying remains format-bound to exact natural-repeat contexts.")
    lines.append("")
    lines.append(f"Data: `{rel(_public_path('experiments/archive/relation_learning/data/integrated_scope_and_mechanism/rewrite_target_probability_across_seed.csv'))}`")
    NOTE.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status":"REWRITE_TARGET_PROBABILITY_DONE", "rows": len(rows), "note": rel(NOTE), "across": rel(_public_path('experiments/archive/relation_learning/data/integrated_scope_and_mechanism/rewrite_target_probability_across_seed.csv'))}, indent=2))

if __name__ == "__main__":
    main()
