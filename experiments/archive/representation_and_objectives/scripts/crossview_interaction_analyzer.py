#!/usr/bin/env python3
"""research cross-view partner-visibility interaction analyzer (loss level).

Consumes per-stratum training logs from four arms:
  own_visible, own_blocked, wrong_visible, wrong_blocked.

For each rewrite target stratum X (rw_copied, rw_abs_content, rw_abs_other) and for
source/filler channels, computes exposure-weighted mean loss over the training run
and end-window mean loss over the final epoch (second 10M words), then:

  own_vis_effect(X)   = own_visible(X)   - own_blocked(X)      # partner visibility under correct partner
  wrong_vis_effect(X) = wrong_visible(X) - wrong_blocked(X)    # visibility under mismatched partner
  I_partner(X)        = own_vis_effect(X) - wrong_vis_effect(X)

Negative I_partner(X) means correct-partner visibility reduces loss on stratum X
beyond generic cross-boundary visibility. The decisive route signal is negative
I_partner on rw_abs_content (intrinsically source-absent semantic content), with
copy-only or null interaction closing the compact semantic-abstraction route.

This is loss-level evidence only; downstream Supplement/relational-EWoK readout is a
separate endpoint step, run only if this interaction is positive on source-absent
content.
"""
from __future__ import annotations

import argparse
import json
import pathlib

STRATA = ["filler", "source", "rw_copied", "rw_abs_content", "rw_abs_other"]


def load_log(run_dir: pathlib.Path):
    log = run_dir / "training_log.jsonl"
    recs = []
    with log.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                recs.append(json.loads(line))
    return recs


def weighted_and_endwindow(recs, end_word_start: int):
    """Return exposure-weighted mean and end-window (cum >= end_word_start) mean per stratum."""
    tot = {s: [0.0, 0] for s in STRATA}
    end = {s: [0.0, 0] for s in STRATA}
    for r in recs:
        cum = r.get("cum", 0)
        for s in STRATA:
            if s in r and f"{s}_n" in r:
                loss = r[s]
                n = r[f"{s}_n"]
                tot[s][0] += loss * n
                tot[s][1] += n
                if cum >= end_word_start:
                    end[s][0] += loss * n
                    end[s][1] += n
    wmean = {s: (tot[s][0] / tot[s][1] if tot[s][1] else None) for s in STRATA}
    emean = {s: (end[s][0] / end[s][1] if end[s][1] else None) for s in STRATA}
    counts = {s: tot[s][1] for s in STRATA}
    return wmean, emean, counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--own_visible", required=True)
    ap.add_argument("--own_blocked", required=True)
    ap.add_argument("--wrong_visible", required=True)
    ap.add_argument("--wrong_blocked", required=True)
    ap.add_argument("--end_word_start", type=int, default=10_000_000)
    ap.add_argument("--output_dir", required=True)
    args = ap.parse_args()
    out = pathlib.Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    arms = {
        "own_visible": pathlib.Path(args.own_visible),
        "own_blocked": pathlib.Path(args.own_blocked),
        "wrong_visible": pathlib.Path(args.wrong_visible),
        "wrong_blocked": pathlib.Path(args.wrong_blocked),
    }
    arm_stats = {}
    final_losses = {}
    for name, rd in arms.items():
        recs = load_log(rd)
        w, e, c = weighted_and_endwindow(recs, args.end_word_start)
        arm_stats[name] = {"weighted_mean_loss": w, "endwindow_mean_loss": e, "counts": c, "n_steps": len(recs)}
        sm = rd / "scientific_metrics.json"
        if sm.exists():
            final_losses[name] = json.loads(sm.read_text())

    def interaction(metric_key):
        res = {}
        for s in STRATA:
            ov = arm_stats["own_visible"][metric_key][s]
            ob = arm_stats["own_blocked"][metric_key][s]
            wv = arm_stats["wrong_visible"][metric_key][s]
            wb = arm_stats["wrong_blocked"][metric_key][s]
            if None in (ov, ob, wv, wb):
                res[s] = None
                continue
            own_eff = ov - ob
            wrong_eff = wv - wb
            res[s] = {
                "own_visible": round(ov, 6),
                "own_blocked": round(ob, 6),
                "wrong_visible": round(wv, 6),
                "wrong_blocked": round(wb, 6),
                "own_vis_effect": round(own_eff, 6),
                "wrong_vis_effect": round(wrong_eff, 6),
                "I_partner": round(own_eff - wrong_eff, 6),
            }
        return res

    weighted_I = interaction("weighted_mean_loss")
    endwindow_I = interaction("endwindow_mean_loss")

    def verdict(I):
        rac = I.get("rw_abs_content")
        rcp = I.get("rw_copied")
        if not rac or not rcp:
            return "incomplete"
        # route survives if correct-partner visibility helps source-absent content beyond copy channel
        if rac["I_partner"] < 0 and rac["I_partner"] < rcp["I_partner"]:
            return "source_absent_partner_signal"
        if rcp["I_partner"] < 0 and rac["I_partner"] >= 0:
            return "copy_only"
        return "null_or_adverse"

    result = {
        "status": "CROSSVIEW_PARTNER_VISIBILITY_INTERACTION",
        "arms": {k: str(v) for k, v in arms.items()},
        "end_word_start": args.end_word_start,
        "arm_stats": arm_stats,
        "weighted_interaction": weighted_I,
        "endwindow_interaction": endwindow_I,
        "verdict_weighted": verdict(weighted_I),
        "verdict_endwindow": verdict(endwindow_I),
        "interpretation": (
            "Negative I_partner on rw_abs_content that is also below the rw_copied I_partner indicates "
            "correct-partner visibility specifically helps intrinsically source-absent semantic content, "
            "the surviving-route signal. A copy-only or null/adverse interaction closes the compact "
            "semantic-abstraction route at loss level. Downstream Supplement/relational-EWoK endpoint "
            "readout is warranted only under source_absent_partner_signal."
        ),
        "final_arm_summaries": {k: v.get("stratum_mean_loss") for k, v in final_losses.items()},
    }
    (out / "crossview_partner_visibility_interaction.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "status": result["status"],
        "verdict_weighted": result["verdict_weighted"],
        "verdict_endwindow": result["verdict_endwindow"],
        "weighted_rw_abs_content": weighted_I.get("rw_abs_content"),
        "weighted_rw_copied": weighted_I.get("rw_copied"),
        "endwindow_rw_abs_content": endwindow_I.get("rw_abs_content"),
        "endwindow_rw_copied": endwindow_I.get("rw_copied"),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
