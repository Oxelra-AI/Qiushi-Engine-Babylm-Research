#!/usr/bin/env python3
"""research: score HALF_VIEW on the compact restatement-dose curve.

HALF_VIEW is the matched half-dose VIEW/no-exact control materialized in research
and trained as the HALF_VIEW arm.  This script scores C/R/V/HM/HV seed43022
with the same compact T/U/N, Wikipedia T/U/N, and natural-copy records used by
the research COMPACT_EXPERIENCE mechanism scorer.  It intentionally does not run official
leaderboard evaluation and does not require Entity predictions for HALF_VIEW.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import json
import math
import pathlib
import sys
import time
from typing import Any

import pandas as pd
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

ROOT0 = _public_path('experiments/archive/relation_learning/scripts/score_half_view_curve.py')
ROOT = _PUBLIC_ROOT

WS = ROOT / "experiments/archive/relation_learning"
REPRESENTATION_FRONTIER_STUDIES = ROOT / "experiments/archive/frontier_consolidation"
sys.path.insert(0, str(WS / "scripts"))
import score_paired_context_relation_design as base  # noqa: E402

OUT = WS / "data/half_view_curve_probe"
NOTE = (_PUBLIC_ROOT / 'research/notes/relation_learning/half_view_curve_probe.md')
CKPTS = ["chck_80M", "chck_90M", "chck_100M"]
ARMS = {
    "C": {
        "description": "seed43022 designed-family CLEAN, zero designed compact relation dose",
        "run": REPRESENTATION_FRONTIER_STUDIES / "training/runs/full_p2c_c2p_abs_clean_dose2p64x_matched_rowholdout_deberta100M_seed43022",
    },
    "R": {
        "description": "seed43022 full exact-recurrence REPEAT",
        "run": REPRESENTATION_FRONTIER_STUDIES / "training/runs/full_p2c_c2p_abs_repeat_dose2p64x_matched_rowholdout_deberta100M_seed43022",
    },
    "V": {
        "description": "seed43022 full compact restatement VIEW, 33.3K local rewrite pairs",
        "run": REPRESENTATION_FRONTIER_STUDIES / "training/runs/full_p2c_c2p_abs_view_dose2p64x_matched_rowholdout_deberta100M_seed43022",
    },
    "HM": {
        "description": "seed43022 hash-mixed exact/rewrite, off-axis 16.7K rewrite plus 16.6K exact pairs",
        "run": WS / "training/runs/full_p2c_c2p_abs_hash_mixed_dose2p64x_matched_rowholdout_deberta100M_seed43022",
    },
    "HV": {
        "description": "seed43022 HALF_VIEW, 16.7K local rewrite pairs plus neutral no-exact slots",
        "run": WS / "training/runs/full_p2c_c2p_abs_half_view_noexact_dose2p64x_matched_rowholdout_deberta100M_seed43022",
    },
}
CONTRASTS = [
    ("RminusC", "R", "C"),
    ("VminusC", "V", "C"),
    ("HVminusC", "HV", "C"),
    ("HVminusV", "HV", "V"),
    ("VminusHV", "V", "HV"),
    ("HMminusC", "HM", "C"),
    ("HMminusHV", "HM", "HV"),
    ("HVminusHM", "HV", "HM"),
    ("HMminusV", "HM", "V"),
    ("HMminusR", "HM", "R"),
]


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def rel(path: pathlib.Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except Exception:
        return str(path)


def model_path(role: str, ck: str) -> pathlib.Path:
    return ARMS[role]["run"] / "hf_model" / ck


def write_json(path: pathlib.Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def fmt(x: Any, digits: int = 4) -> str:
    try:
        v = float(x)
    except Exception:
        return "NA"
    if not math.isfinite(v):
        return "NA"
    return f"{v:+.{digits}f}"


def one(df: pd.DataFrame, **kw) -> pd.Series | None:
    sub = df
    for k, v in kw.items():
        sub = sub[sub[k] == v]
    if len(sub) != 1:
        return None
    return sub.iloc[0]


def write_note(plan: dict[str, Any], compact_roles: pd.DataFrame, compact_con: pd.DataFrame, wiki_roles: pd.DataFrame, wiki_con: pd.DataFrame, copy_roles: pd.DataFrame, copy_con: pd.DataFrame) -> None:
    def cr(role: str, cls: str, col: str) -> str:
        r = one(compact_roles, role=role, token_class=cls)
        return "NA" if r is None else fmt(r[col])
    def cc(con: str, cls: str, col: str) -> str:
        r = one(compact_con, contrast=con, token_class=cls)
        return "NA" if r is None else fmt(r[col])
    def wc(con: str, cls: str, est: str = "gain_T_vs_N") -> str:
        r = one(wiki_con, overlap_bin="ALL", token_class=cls, contrast=con, estimand=est)
        return "NA" if r is None else f"{fmt(r['mean_difference'])} ± {float(r['se_pair_difference']):.4f}"
    def cp(role: str) -> str:
        r = one(copy_roles, role=role)
        return "NA" if r is None else fmt(r["gain"])
    def cpc(con: str) -> str:
        r = one(copy_con, contrast=con)
        return "NA" if r is None else fmt(r["delta_gain"])

    rows = []
    for role in ["C", "HV", "V", "HM", "R"]:
        rows.append((role, cr(role, "nonoverlap", "A_T"), cr(role, "nonoverlap", "A_U"), cr(role, "nonoverlap", "G"), cr(role, "overlap", "A_T"), cp(role)))
    contrast_rows = []
    for con in ["HVminusC", "VminusC", "HMminusC", "HVminusV", "HMminusHV", "RminusC"]:
        contrast_rows.append((con, cc(con, "nonoverlap", "delta_A_T"), cc(con, "nonoverlap", "delta_A_U"), cc(con, "nonoverlap", "delta_G"), cc(con, "overlap", "delta_A_T"), wc(con, "overlap"), wc(con, "nonoverlap"), cpc(con)))

    def getc(con: str, cls: str, col: str) -> float:
        r = one(compact_con, contrast=con, token_class=cls)
        return float("nan") if r is None else float(r[col])
    hv = getc("HVminusC", "nonoverlap", "delta_A_T")
    v = getc("VminusC", "nonoverlap", "delta_A_T")
    hm = getc("HMminusC", "nonoverlap", "delta_A_T")
    frac_hv = hv / v if math.isfinite(hv) and math.isfinite(v) and abs(v) > 1e-9 else float("nan")
    hm_minus_hv = getc("HMminusHV", "nonoverlap", "delta_A_T")

    lines = [
        "# research HALF_VIEW curve probe\n",
        f"Created: {now()}\n\n",
        "HALF_VIEW was trained after the hash-mixed arm to separate a half-dose restatement curve from active exact/restatement composition. It uses the same deterministic half assignment as hash-mix: the rewrite-assigned half receives local source+rewrite pairs; the other selected-source slots receive neutral no-exact companions. This note scores C/R/V/HM/HV with the same records so the compact designed-register curve and the off-axis hash-mixed placement are directly comparable.\n\n",
        "## Records and arms\n\n",
        f"- Per model records: copy {plan['records']['copy_records']}, compact T/U {plan['records']['rewrite_TU_records']}, compact N {plan['records']['rewrite_N_records']}, Wikipedia {plan['records']['wiki_records']}.\n",
        f"- Checkpoints: {', '.join(plan['checkpoints'])}.\n",
        "- C is zero designed compact-relation dose; HV is 16.7K local compact restatements with no exact companion; V is 33.3K local compact restatements; HM is off-axis because it has the same rewrite half as HV plus 16.6K local exact recurrence companions.\n\n",
        "## Late-checkpoint absolute source-use readouts\n\n",
        "Compact values use `A_T=N−T`, `A_U=N−U`, and `G=U−T`; positive `A_T` means the true source helps more relative to the neutral ordinary source.\n\n",
        "| role | compact nonoverlap A_T | compact nonoverlap A_U | compact nonoverlap G | compact overlap A_T | natural-copy gain |\n",
        "|---|---:|---:|---:|---:|---:|\n",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |\n")
    lines += [
        "\n## Dose and off-axis contrasts\n\n",
        "| contrast | compact nonoverlap ΔA_T | compact nonoverlap ΔA_U | compact nonoverlap ΔG | compact overlap ΔA_T | Wikipedia overlap Δ(N−T) | Wikipedia nonoverlap Δ(N−T) | natural-copy Δgain |\n",
        "|---|---:|---:|---:|---:|---:|---:|---:|\n",
    ]
    for row in contrast_rows:
        lines.append("| " + " | ".join(row) + " |\n")
    lines += [
        "\n## Scientific reading\n\n",
        f"On the designed FineWeb-register compact nonoverlap readout, HALF_VIEW is not merely half of VIEW in the late-checkpoint seed43022 curve: HV−C ΔA_T is {fmt(hv)}, compared with V−C {fmt(v)} (ratio {frac_hv:.2f}). This supports a concave or rapidly saturating benefit of corresponding compact restatement practice, not a linear dose rule.\n",
        f"Hash-mix lies below the half-dose no-exact point on the same true-source readout: HM−HV ΔA_T is {fmt(hm_minus_hv)}. Thus the earlier HM shortfall relative to VIEW is no longer interpretable as half-dose alone; with the matched half-dose point present, the off-axis exact-recurrence companions appear to reduce the compact restatement source-use routine. This remains one-seed evidence and should be presented as a composition effect, not as an architecture-free law.\n",
        "The Wikipedia readout must be kept separate from the compact curve because it is near the inherited Qwen/SimpleWiki register rather than the designed compact register. The paper-level principle should therefore distinguish reach: corresponding restatement competence is strong in the format/register practiced, while exact recurrence liability has traveled across compact, Wikipedia-substitution, and causal next-token readouts.\n\n",
        "## Output files\n\n",
        f"- `{rel(OUT / 'compact_TUN_late_roles.csv')}`\n",
        f"- `{rel(OUT / 'compact_TUN_late_contrasts.csv')}`\n",
        f"- `{rel(OUT / 'wikipedia_late_contrasts.csv')}`\n",
        f"- `{rel(OUT / 'copy_late_contrasts.csv')}`\n",
    ]
    NOTE.parent.mkdir(parents=True, exist_ok=True)
    NOTE.write_text("".join(lines), encoding="utf-8")
    summary = {
        "status": "HALF_VIEW_CURVE_INTEGRATED",
        "note": rel(NOTE),
        "hv_minus_c_compact_nonoverlap_delta_A_T": hv,
        "v_minus_c_compact_nonoverlap_delta_A_T": v,
        "hm_minus_c_compact_nonoverlap_delta_A_T": hm,
        "hv_over_v_ratio_compact_nonoverlap_A_T": frac_hv,
        "hm_minus_hv_compact_nonoverlap_delta_A_T": hm_minus_hv,
    }
    write_json(OUT / "half_view_curve_summary.json", summary)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arms", nargs="+", default=["C", "R", "V", "HM", "HV"])
    ap.add_argument("--checkpoints", nargs="+", default=CKPTS)
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--copy-n-per-span", type=int, default=1000)
    ap.add_argument("--copy-span-lengths", nargs="+", type=int, default=[1, 4])
    ap.add_argument("--copy-scan-rows", type=int, default=6992)
    ap.add_argument("--rewrite-max-pairs", type=int, default=0)
    ap.add_argument("--rewrite-tokens-per-class", type=int, default=2)
    ap.add_argument("--max-len-copy-rewrite", type=int, default=256)
    ap.add_argument("--wiki-max-records", type=int, default=0)
    ap.add_argument("--seed", type=int, default=930030)
    ap.add_argument("--plan-only", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    unknown = [a for a in args.arms if a not in ARMS]
    if unknown:
        raise KeyError(unknown)
    for role in args.arms:
        for ck in args.checkpoints:
            p = model_path(role, ck)
            if not p.exists():
                raise FileNotFoundError(p)

    # Patch the imported scorer's globals so its builders/aggregators can be reused.
    base.ARMS = {k: {"description": ARMS[k]["description"], "run": ARMS[k]["run"], "per_target": pathlib.Path("/dev/null")} for k in args.arms}
    base.CKPTS = list(args.checkpoints)
    base.CONTRASTS = [c for c in CONTRASTS if c[1] in args.arms and c[2] in args.arms]
    base.model_path = model_path  # type: ignore[assignment]

    tokenizer = AutoTokenizer.from_pretrained(str(model_path(args.arms[0], args.checkpoints[-1]).parent), use_fast=True)
    records, rec_stats = base.build_probe_records(tokenizer, args)
    plan = {
        "status": "HALF_VIEW_CURVE_PLAN",
        "created_utc": now(),
        "arms": {k: {"description": ARMS[k]["description"], "run": rel(ARMS[k]["run"])} for k in args.arms},
        "checkpoints": list(args.checkpoints),
        "records": rec_stats,
        "total_records_per_model": len(records),
        "out_dir": rel(OUT),
    }
    write_json(OUT / "score_plan.json", plan)
    if args.plan_only:
        print(json.dumps(plan, indent=2, ensure_ascii=False))
        return

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    pad_id = int(tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0)
    scored_all: list[dict[str, Any]] = []
    meta: list[dict[str, Any]] = []
    for role in args.arms:
        for ck in args.checkpoints:
            t0 = time.time()
            mp = model_path(role, ck)
            print(f"[LOAD] {role} {ck} {mp}", flush=True)
            model = AutoModelForMaskedLM.from_pretrained(str(mp), torch_dtype=torch.float32).eval().to(device)
            rows = base.score_records(model, records, device, pad_id, args.batch_size)
            for r in rows:
                r["role"] = role
                r["arm"] = role
                r["checkpoint"] = ck
            scored_all.extend(rows)
            elapsed = time.time() - t0
            meta.append({"role": role, "checkpoint": ck, "records": len(rows), "elapsed_sec": round(elapsed, 2), "device": str(device)})
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
            print(f"[DONE] {role} {ck}: {len(rows)} records in {elapsed:.1f}s", flush=True)
    base.write_csv(OUT / "score_meta.csv", meta)
    dfs = base.aggregate_all(scored_all, OUT)
    _m, compact_roles, compact_con = base.integrate_compact(dfs, OUT)
    copy_roles, copy_con = base.integrate_copy(dfs, OUT)
    _wt, wiki_roles, wiki_con = base.integrate_wikipedia(dfs, OUT)
    write_note(plan, compact_roles, compact_con, wiki_roles, wiki_con, copy_roles, copy_con)
    result = {"status": "HALF_VIEW_CURVE_DONE", "created_utc": now(), "out_dir": rel(OUT), "note": rel(NOTE), "score_rows": len(scored_all)}
    write_json(OUT / "score_summary.json", result)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
