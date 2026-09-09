#!/usr/bin/env python3
"""research: summarize persistence and cross-loss evidence into a mechanism readout.

Inputs:
  - research own changed/unchanged MLM loss across view/repeat/breadth/clean.
  - research cross-loss of common text sets under clean/view/repeat/breadth models.

Output:
  - A compact JSON and Markdown reading of what the loss measurements do and do
    not establish about duplicate recurrence versus distinct-content persistence.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative

import json, pathlib, statistics, time
from typing import Any


def find_root() -> pathlib.Path:
    return _PUBLIC_ROOT

ROOT = find_root()
WS = ROOT / "experiments/archive" / 'frontier_consolidation'
OWN = WS / "data" / "persistence_loss_analysis" / "persistence_loss_results.json"
CROSS = WS / "data" / "persistence_crossloss_analysis" / "crossloss_results.json"
OUT = WS / "data" / "persistence_mechanism_readout"


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def rel(p: pathlib.Path) -> str:
    try: return str(p.relative_to(ROOT))
    except ValueError: return str(p)


def load(path: pathlib.Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def own_index(data: dict[str, Any]) -> dict[tuple[str, str, str], float]:
    out = {}
    for r in data["results"]:
        out[(r["arm"], r["checkpoint"], r["block_type"])] = float(r["mean_loss"])
    return out


def cross_index(data: dict[str, Any]) -> dict[tuple[str, str, str], float]:
    out = {}
    for r in data["results"]:
        out[(r["model_arm"], r["checkpoint"], r["text_set"])] = float(r["mean_loss"])
    return out


def rnd(x: float | None) -> float | None:
    return None if x is None else round(float(x), 6)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    own = load(OWN)
    cross = load(CROSS)
    oi = own_index(own)
    ci = cross_index(cross)

    checkpoints = ["chck_40M", "chck_80M", "chck_100M"]
    arms = ["view", "repeat", "breadth", "clean"]
    textsets = ["view_changed", "repeat_changed", "breadth_changed", "clean_displaced"]
    owner = {"view_changed": "view", "repeat_changed": "repeat", "breadth_changed": "breadth", "clean_displaced": "clean"}

    own_rows = []
    for arm in arms:
        for ck in ["chck_20M", "chck_40M", "chck_60M", "chck_80M", "chck_100M"]:
            ch = oi.get((arm, ck, "changed")); un = oi.get((arm, ck, "unchanged"))
            own_rows.append({
                "arm": arm, "checkpoint": ck,
                "changed_loss": rnd(ch), "unchanged_loss": rnd(un),
                "changed_minus_unchanged": rnd(ch - un) if ch is not None and un is not None else None,
            })

    cross_rows = []
    for ck in checkpoints:
        for ts in textsets:
            cl = ci.get(("clean", ck, ts))
            ol = ci.get((owner[ts], ck, ts))
            cross_rows.append({
                "checkpoint": ck,
                "text_set": ts,
                "clean_loss": rnd(cl),
                "owner_arm": owner[ts],
                "owner_loss": rnd(ol),
                "clean_minus_owner": rnd(cl - ol) if cl is not None and ol is not None else None,
            })

    displaced_rows = []
    for ck in checkpoints:
        clean_loss = ci.get(("clean", ck, "clean_displaced"))
        for arm in ["view", "repeat", "breadth"]:
            loss = ci.get((arm, ck, "clean_displaced"))
            displaced_rows.append({
                "checkpoint": ck,
                "arm": arm,
                "loss_on_clean_displaced": rnd(loss),
                "clean_loss_on_clean_displaced": rnd(clean_loss),
                "arm_minus_clean_on_displaced": rnd(loss - clean_loss) if loss is not None and clean_loss is not None else None,
            })

    # Key late numbers used in the reading.
    key = {
        "own_100M_changed_loss": {arm: oi.get((arm, "chck_100M", "changed")) for arm in arms},
        "own_100M_changed_minus_unchanged": {arm: oi.get((arm, "chck_100M", "changed")) - oi.get((arm, "chck_100M", "unchanged")) for arm in arms},
        "cross_100M_clean_minus_owner": {ts: ci.get(("clean", "chck_100M", ts)) - ci.get((owner[ts], "chck_100M", ts)) for ts in textsets},
        "cross_100M_clean_prior_difficulty": {ts: ci.get(("clean", "chck_100M", ts)) for ts in textsets},
        "displaced_100M_arm_minus_clean": {arm: ci.get((arm, "chck_100M", "clean_displaced")) - ci.get(("clean", "chck_100M", "clean_displaced")) for arm in ["view", "repeat", "breadth"]},
    }
    key = json.loads(json.dumps(key), parse_float=float)

    late_displaced = [ci.get((arm, "chck_100M", "clean_displaced")) - ci.get(("clean", "chck_100M", "clean_displaced")) for arm in ["view", "repeat", "breadth"]]
    mechanism = {
        "supports": [
            "Exact recurrence drives very low own loss on repeated changed rows by 100M, but this coincides with near-zero late broad downstream advantage from earlier research/287 scores; low training loss alone is not transferable competence.",
            "Distinct breadth rows remain high-loss even after training and are very hard for the clean model, matching the idea that distinct admitted material continues to supply residual prediction error instead of becoming exhausted.",
            "Compact view rows are intermediate: learned substantially better than clean, but not as over-compressed as exact repeat; this is compatible with why view retains late broad benefit while exact repeat decays.",
            "Sampled displaced clean rows are only about 0.10-0.12 loss worse under view/repeat/breadth than under clean at 100M, so the late V/B versus R difference is not explained by a large differential loss collapse on these sampled sacrificed rows.",
        ],
        "does_not_establish": [
            "This is forward-only fixed-mask MLM loss, not a direct gradient norm measurement; it estimates residual prediction error rather than the exact parameter update contribution.",
            "The samples are small deterministic row samples (research n=300 own rows, research n=200 cross rows), not full-corpus loss integrals.",
            "The readout does not by itself prove that the residual error is evaluation-relevant; official family scores from the corrected research scorer are still needed.",
            "It does not remove the known architecture boundary: RoBERTa did not reproduce the broad late DeBERTa view-clean sign.",
        ],
        "current_scientific_formulation": "Under this DeBERTa/WWM/fixed-budget coordinate, repeated exposure can make admitted duplicate text easy without maintaining broad downstream value; distinct adult-distribution content remains partly unpredictable across passes and can keep supplying useful learning pressure. The principle is therefore about persistent nonredundant error on useful distributions, plus the opportunity cost of what is removed, not simply word count, surface compactness, or recurrence.",
    }

    output = {
        "status": "PERSISTENCE_MECHANISM_READOUT",
        "created_utc": now(),
        "inputs": {"own_loss": rel(OWN), "cross_loss": rel(CROSS)},
        "key_late_numbers": key,
        "own_loss_rows": own_rows,
        "cross_loss_rows": cross_rows,
        "displaced_loss_rows": displaced_rows,
        "mechanism_reading": mechanism,
    }
    out_json = OUT / "persistence_mechanism_readout.json"
    out_json.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    def f(x: Any) -> str:
        return "NA" if x is None else f"{float(x):.4f}"

    lines = [
        "# research persistence mechanism readout",
        "",
        "This note joins the research own changed/unchanged row loss measurement with the research cross-loss measurement. It is a mechanism readout, not an official benchmark score.",
        "",
        "## Late 100M signal",
        "",
        "| arm/text set | clean or own prior loss | owner loss | delta |",
        "|---|---:|---:|---:|",
    ]
    for ts in textsets:
        cl = ci.get(("clean", "chck_100M", ts))
        ol = ci.get((owner[ts], "chck_100M", ts))
        lines.append(f"| {ts} | {f(cl)} | {f(ol)} ({owner[ts]}) | {f(cl-ol if cl is not None and ol is not None else None)} clean_minus_owner |")
    lines += ["", "### Own changed vs unchanged at 100M", "", "| arm | changed loss | unchanged loss | changed-minus-unchanged |", "|---|---:|---:|---:|"]
    for arm in arms:
        ch = oi.get((arm, "chck_100M", "changed")); un = oi.get((arm, "chck_100M", "unchanged"))
        lines.append(f"| {arm} | {f(ch)} | {f(un)} | {f(ch-un if ch is not None and un is not None else None)} |")
    lines += ["", "### Displaced clean rows at 100M", "", "| intervention arm | loss on clean-displaced rows | clean loss on same rows | intervention-minus-clean |", "|---|---:|---:|---:|"]
    for arm in ["view", "repeat", "breadth"]:
        loss = ci.get((arm, "chck_100M", "clean_displaced")); clean_loss = ci.get(("clean", "chck_100M", "clean_displaced"))
        lines.append(f"| {arm} | {f(loss)} | {f(clean_loss)} | {f(loss-clean_loss if loss is not None and clean_loss is not None else None)} |")
    lines += ["", "## Scientific reading", ""]
    for s in mechanism["supports"]:
        lines.append(f"- {s}")
    lines += ["", "## What remains unsettled", ""]
    for s in mechanism["does_not_establish"]:
        lines.append(f"- {s}")
    lines += ["", "## Current formulation", "", mechanism["current_scientific_formulation"], "", f"JSON: `{rel(out_json)}`"]
    out_md = OUT / "persistence_mechanism_readout.md"
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps({
        "status": output["status"],
        "key_late_numbers": key,
        "summary_md": rel(out_md),
        "summary_json": rel(out_json),
    }, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
