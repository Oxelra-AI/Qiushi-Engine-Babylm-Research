#!/usr/bin/env python3
"""Close or summarize the research coherence-margin pilot route."""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import json
import pathlib
import time
from statistics import mean
from typing import Any

ROOT = _public_path('.')
WORKSPACE = _public_path('experiments/archive/frontier_consolidation')
OUT = _public_path('experiments/archive/frontier_consolidation/data/cohmargin_route_closure')
PILOT_PAYLOAD = _public_path('experiments/archive/frontier_consolidation/data/cohmargin4M_eval/per_target/cohmargin4M_scale1p75_seed43022.json')
TRAIN_TRACE = _public_path('experiments/archive/frontier_consolidation/data/cohmargin_training_trace_summary/cohmargin_training_trace_summary.json')
NLL_COMPARE = _public_path('experiments/archive/frontier_consolidation/data/cohmargin_nll_gap_comparison/cohmargin_nll_gap_comparison.json')
CHCK82 = _public_path('experiments/archive/frontier_consolidation/data/chck82_independent_verification/chck82_independent_verification.json')
LAMBDA0_TASK_STDOUT = _public_path('experiments/archive/frontier_consolidation/tasks/s159_t54_tool1/stdout.log')
CHEAP = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA", "Reading"]


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def rel(p: pathlib.Path | str) -> str:
    q = pathlib.Path(p)
    try:
        return str(q.resolve().relative_to(ROOT))
    except Exception:
        return str(q)


def read_json(p: pathlib.Path) -> Any:
    return json.loads(p.read_text(encoding="utf-8"))


def scores_from_payload(p: pathlib.Path) -> dict[str, float | None]:
    data = read_json(p)
    official = data.get("official_overall", {}).get("scores", {})
    scores = {c: (None if official.get(c) is None else float(official[c])) for c in CHEAP}
    if scores.get("GlobalPIQA") is None:
        tasks = data.get("tasks", {})
        gp = []
        for c in ["GlobalPIQA_parallel", "GlobalPIQA_nonparallel"]:
            if tasks.get(c, {}).get("score") is not None:
                gp.append(float(tasks[c]["score"]))
        if len(gp) == 2:
            scores["GlobalPIQA"] = float(mean(gp))
    if scores.get("Reading") is None:
        rd = data.get("tasks", {}).get("Reading", {})
        if isinstance(rd.get("scores"), dict) and rd["scores"].get("Reading") is not None:
            scores["Reading"] = float(rd["scores"]["Reading"])
    return scores


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    pilot_scores = scores_from_payload(PILOT_PAYLOAD)
    pilot_cheap7 = float(mean(float(pilot_scores[c]) for c in CHEAP))
    ch = read_json(CHCK82)
    ch_scores = {k: float(v) for k, v in ch["score_arithmetic"]["scores"].items() if v is not None}
    ch_cheap7 = float(mean(ch_scores[c] for c in CHEAP))
    trace = read_json(TRAIN_TRACE)
    nll = read_json(NLL_COMPARE)
    lambda0_stdout_tail = ""
    if LAMBDA0_TASK_STDOUT.exists():
        txt = LAMBDA0_TASK_STDOUT.read_text(encoding="utf-8", errors="replace")
        lambda0_stdout_tail = "\n".join(txt.splitlines()[-6:])
    deltas = {c: float(pilot_scores[c] - ch_scores[c]) for c in CHEAP}
    out = {
        "status": "COHERENCE_MARGIN_ROUTE_CLOSED_AFTER_4M_PILOT",
        "created_utc": now(),
        "pilot_payload": rel(PILOT_PAYLOAD),
        "pilot_scores": pilot_scores,
        "pilot_cheap7": pilot_cheap7,
        "protected_chck82_cheap7": ch_cheap7,
        "deltas_vs_chck82": deltas,
        "cheap7_delta_vs_chck82": pilot_cheap7 - ch_cheap7,
        "training_trace_summary": {
            "source": rel(TRAIN_TRACE),
            "charged_word_exposure": trace.get("charged_word_exposure"),
            "coherent_words": trace.get("coherent_words"),
            "optimizer_steps": trace.get("optimizer_steps"),
            "margin_loss_mean": trace.get("margin_loss_mean"),
            "last_10_margin_loss_mean": trace.get("last_10_margin_loss_mean"),
            "nll_bad_minus_coh_mean": trace.get("nll_bad_minus_coh_mean"),
            "last_10_nll_gap_mean": trace.get("last_10_nll_gap_mean"),
        },
        "nll_gap_comparison": {
            "source": rel(NLL_COMPARE),
            "pilot_train_gap": nll["records"]["pilot_train64"]["bad_minus_coh_nll_mean"],
            "pilot_holdout_gap": nll["records"]["pilot_holdout64_after2M"]["bad_minus_coh_nll_mean"],
            "ref2m_train_gap": nll["records"]["ref2m_train64"]["bad_minus_coh_nll_mean"],
            "ref2m_holdout_gap": nll["records"]["ref2m_holdout64_after2M"]["bad_minus_coh_nll_mean"],
            "ref4m_train_gap": nll["records"]["ref4m_train64"]["bad_minus_coh_nll_mean"],
            "ref4m_holdout_gap": nll["records"]["ref4m_holdout64_after2M"]["bad_minus_coh_nll_mean"],
        },
        "lambda0_isolate_status": {
            "task": "s159_t54_tool1",
            "state": "cancelled_after_pilot_destructive_result",
            "partial_trace_reading": "not a result; smoke and early logs only confirm same trainer/disruption geometry before cancellation",
            "stdout_tail": lambda0_stdout_tail,
        },
        "scientific_interpretation": "The pilot is broadly destructive on official-compatible cheap7 and fails the intended mechanism probe: the margin term stayed near softplus(0.20) and coherent-vs-disrupted NLL separation remained near zero. The route should not continue to 20M, nor should lambda-zero official evaluation be spent after this pilot result.",
    }
    js = _public_path('experiments/archive/frontier_consolidation/data/cohmargin_route_closure/cohmargin_route_closure.json')
    md = _public_path('research/documents/frontier_consolidation/data/cohmargin_route_closure/cohmargin_route_closure.md')
    js.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = [
        "# research coherence-margin route closure",
        "",
        f"Status: **{out['status']}**",
        "",
        f"Pilot cheap7: `{pilot_cheap7}`; protected chck82 cheap7: `{ch_cheap7}`; delta `{pilot_cheap7 - ch_cheap7:+.6f}`.",
        "",
        "| column | pilot | chck82 | delta |",
        "|---|---:|---:|---:|",
    ]
    for c in CHEAP:
        lines.append(f"| {c} | {pilot_scores[c]:.6f} | {ch_scores[c]:.6f} | {deltas[c]:+.6f} |")
    lines += [
        "",
        "## Mechanism readings",
        "",
        f"Training trace: margin_loss mean `{trace.get('margin_loss_mean')}`, last10 `{trace.get('last_10_margin_loss_mean')}`, logged NLL gap mean `{trace.get('nll_bad_minus_coh_mean')}`, last10 `{trace.get('last_10_nll_gap_mean')}`.",
        f"NLL probe gaps: pilot train64 `{out['nll_gap_comparison']['pilot_train_gap']}`, pilot holdout64 `{out['nll_gap_comparison']['pilot_holdout_gap']}`, ref4m train64 `{out['nll_gap_comparison']['ref4m_train_gap']}`, ref4m holdout64 `{out['nll_gap_comparison']['ref4m_holdout_gap']}`.",
        "",
        out["scientific_interpretation"],
        "",
        "The lambda-zero isolate task was cancelled once the pilot cheap7 result made the route noncompetitive and scientifically unpromising; its partial logs are not an endpoint result.",
        "",
        f"JSON: `{rel(js)}`",
    ]
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": out["status"], "pilot_cheap7": pilot_cheap7, "cheap7_delta_vs_chck82": pilot_cheap7 - ch_cheap7, "out_json": rel(js), "out_md": rel(md)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
