#!/usr/bin/env python3
"""research: Does MLM training-loss trajectory predict downstream Overall / winning seed?

CPU-only. Reads training_log.jsonl for the inherited-tokenizer reinvest runs
(both seeds, complete 100M) whose official Overall scores are known, plus the
in-progress corrected-tokenizer retrains for a same-window comparison.

Purpose: determine whether the pretraining loss curve is a usable seed-selection
signal. If loss cannot separate the seeds whose official Overalls differ by
~0.785, then intermediate DOWNSTREAM evaluation is the only reliable selector,
which justifies a specific cheap GPU protocol once a GPU frees up.

This does NOT launch any GPU work and does not modify training runs.
"""

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative

import json
import math
from pathlib import Path

HERE = _public_path('experiments/archive/representation_and_objectives/scripts/loss_trajectory_vs_overall.py')
# user root = .../ (parent of Sessions)
ROOT = HERE
while ROOT.name != "representation_and_objectives" and _public_path('experiments/archive/representation_and_objectives/scripts') != ROOT:
    ROOT = _public_path('experiments/archive/representation_and_objectives/scripts')
# ROOT now at experiments/archive/representation_and_objectives ; go up to user root
USER_ROOT = _public_path('experiments/archive/representation_and_objectives')

RUNS = {
    "old_tok_reinv_seed43022": {
        "log": "experiments/archive/frontier_consolidation/training/runs/cleanqwen_fineweb_compact_view_reinvest_16k_seed43022/training_log.jsonl",
        "official_overall": 42.0331347900748,
        "tokenizer": "inherited_strict100M",
    },
    "old_tok_reinv_seed43122": {
        "log": "experiments/archive/representation_and_objectives/training/runs/repl_compact_view_reinvest_seed43122/training_log.jsonl",
        "official_overall": 41.24823958912208,
        "tokenizer": "inherited_strict100M",
    },
    "new_tok_reinv_seed43022": {
        "log": "experiments/archive/representation_and_objectives/training/runs/strictsmalltok_compact_view_reinvest_seed43022/training_log.jsonl",
        "official_overall": None,  # in progress
        "tokenizer": "strictsmall_10M",
    },
    "new_tok_reinv_seed43122": {
        "log": "experiments/archive/representation_and_objectives/training/runs/strictsmalltok_compact_view_reinvest_seed43122/training_log.jsonl",
        "official_overall": None,  # in progress
        "tokenizer": "strictsmall_10M",
    },
}


def load_log(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            if d.get("event") == "train" or "loss" in d:
                rows.append(d)
    return rows


def window_mean(rows, lo, hi):
    """Mean loss over cumulative_word_exposure in [lo, hi)."""
    vals = [r["loss"] for r in rows
            if lo <= r.get("cumulative_word_exposure", -1) < hi and math.isfinite(r.get("loss", float("nan")))]
    return (sum(vals) / len(vals)) if vals else None


def summarize(rows):
    losses = [r["loss"] for r in rows if math.isfinite(r.get("loss", float("nan")))]
    max_exp = max((r.get("cumulative_word_exposure", 0) for r in rows), default=0)
    windows = {}
    # 10M windows up to whatever is available
    for w in range(0, 100_000_000, 10_000_000):
        m = window_mean(rows, w, w + 10_000_000)
        if m is not None:
            windows[f"{w//1_000_000}-{(w+10_000_000)//1_000_000}M"] = round(m, 6)
    return {
        "n_steps": len(rows),
        "max_word_exposure": max_exp,
        "final_loss": round(losses[-1], 6) if losses else None,
        "last50_mean_loss": round(sum(losses[-50:]) / len(losses[-50:]), 6) if losses else None,
        "all_step_mean_loss": round(sum(losses) / len(losses), 6) if losses else None,
        "window_means_10M": windows,
    }


def main():
    out = {"status": "LOSS_TRAJECTORY_VS_OVERALL", "runs": {}}
    loaded = {}
    for key, cfg in RUNS.items():
        p = USER_ROOT / cfg["log"]
        if not p.exists():
            out["runs"][key] = {"error": f"log not found: {cfg['log']}"}
            continue
        rows = load_log(p)
        loaded[key] = rows
        s = summarize(rows)
        s["official_overall"] = cfg["official_overall"]
        s["tokenizer"] = cfg["tokenizer"]
        out["runs"][key] = s

    # ── Key comparisons on the OLD tokenizer pair (known Overalls) ──
    a = "old_tok_reinv_seed43022"  # winner, 42.033
    b = "old_tok_reinv_seed43122"  # loser, 41.248
    comp = {}
    if a in loaded and b in loaded:
        # matched-window loss deltas (winner minus loser) over common windows
        wa = out["runs"][a]["window_means_10M"]
        wb = out["runs"][b]["window_means_10M"]
        deltas = {}
        for w in wa:
            if w in wb:
                deltas[w] = round(wa[w] - wb[w], 6)
        comp["old_pair_window_loss_delta_winner_minus_loser"] = deltas
        comp["old_pair_all_step_mean_loss_delta"] = round(
            out["runs"][a]["all_step_mean_loss"] - out["runs"][b]["all_step_mean_loss"], 6)
        comp["old_pair_last50_mean_loss_delta"] = round(
            out["runs"][a]["last50_mean_loss"] - out["runs"][b]["last50_mean_loss"], 6)
        comp["official_overall_delta_winner_minus_loser"] = round(
            RUNS[a]["official_overall"] - RUNS[b]["official_overall"], 6)
        # interpretation: sign of loss delta vs sign of Overall delta
        loss_lower_for_winner = comp["old_pair_all_step_mean_loss_delta"] < 0
        comp["winner_has_lower_all_step_loss"] = loss_lower_for_winner
        comp["loss_predicts_winner"] = (
            "NO — winner does NOT have consistently lower training loss; "
            "loss trajectory does not separate the seeds by the direction of their "
            "0.785 Overall gap"
            if not loss_lower_for_winner or abs(comp["old_pair_all_step_mean_loss_delta"]) < 0.01
            else "AMBIGUOUS — check magnitudes"
        )
    out["comparisons"] = comp

    # ── New tokenizer same-window loss divergence so far ──
    na, nb = "new_tok_reinv_seed43022", "new_tok_reinv_seed43122"
    if na in loaded and nb in loaded:
        wna = out["runs"][na]["window_means_10M"]
        wnb = out["runs"][nb]["window_means_10M"]
        ndeltas = {w: round(wna[w] - wnb[w], 6) for w in wna if w in wnb}
        out["new_pair_window_loss_delta_43022_minus_43122_inprogress"] = ndeltas

    out_dir = _public_path('experiments/archive/representation_and_objectives/data/loss_trajectory_vs_overall')
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "loss_trajectory_vs_overall.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps({
        "status": out["status"],
        "out_json": str(out_path.relative_to(USER_ROOT)),
        "old_pair_all_step_mean_loss_delta": comp.get("old_pair_all_step_mean_loss_delta"),
        "old_pair_last50_mean_loss_delta": comp.get("old_pair_last50_mean_loss_delta"),
        "official_overall_delta_winner_minus_loser": comp.get("official_overall_delta_winner_minus_loser"),
        "loss_predicts_winner": comp.get("loss_predicts_winner"),
    }, indent=2))


if __name__ == "__main__":
    main()
