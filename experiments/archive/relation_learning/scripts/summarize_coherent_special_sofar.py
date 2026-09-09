#!/usr/bin/env python3
"""research: summarize the first coherent-special format endpoint evidence.

This reads only completed or already written files.  It preserves the two-seed
coherent-unsplit-special signal before later format endpoints influence the
interpretation.  Missing columns are left missing rather than inferred.
"""
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
OUT = _public_path('experiments/archive/relation_learning/data/coherent_special_sofar')
TRAIN_ROOT = _public_path('experiments/archive/relation_learning/data/format_replay_corrected/coherent_unsplit_special')
EVAL_ROOT = _public_path('experiments/archive/relation_learning/data/eval_format_replay_corrected/coherent_unsplit_special')
research = _public_path('experiments/archive/relation_learning/data/format_and_v5_reading_matrix/format_and_v5_reading_matrix.json')
MD = _public_path('research/documents/relation_learning/data/pairwise_item_flips/pairwise_item_flips.md')

CHEAP = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA", "Reading"]
SEEDS = [98097, 98098]
CHCK82 = {
    "BLiMP": 68.49128403651986,
    "Supplement": 62.9378112562002,
    "EWoK": 50.05545332553276,
    "Entity": 28.314041930298774,
    "COMPS": 52.19117509443596,
    "GlobalPIQA": 37.57766990291262,
    "Reading": 8.148713589261902,
}
COH86 = {
    "BLiMP": 68.51,
    "Supplement": 63.64,
    "EWoK": 50.02,
    "Entity": 28.32,
    "COMPS": 52.05,
    "GlobalPIQA": 38.565,
    "Reading": 8.165,
}


def rel(p: pathlib.Path | str) -> str:
    try:
        return str(pathlib.Path(p).resolve().relative_to(ROOT))
    except Exception:
        return str(p)


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def read_json(path: pathlib.Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def extract_scores(payload: dict[str, Any]) -> dict[str, float]:
    out: dict[str, float] = {}
    tasks = payload.get("tasks", {})
    for c in ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS"]:
        rec = tasks.get(c)
        if isinstance(rec, dict) and rec.get("score") is not None:
            out[c] = float(rec["score"])
    gp_vals = []
    for c in ["GlobalPIQA_parallel", "GlobalPIQA_nonparallel"]:
        rec = tasks.get(c)
        if isinstance(rec, dict) and rec.get("score") is not None:
            gp_vals.append(float(rec["score"]))
    if len(gp_vals) == 2:
        out["GlobalPIQA"] = float(mean(gp_vals))
    rec = tasks.get("Reading")
    if isinstance(rec, dict):
        if isinstance(rec.get("scores"), dict) and rec["scores"].get("Reading") is not None:
            out["Reading"] = float(rec["scores"]["Reading"])
        elif rec.get("score") is not None:
            out["Reading"] = float(rec["score"])
    official = payload.get("official_overall", {}).get("scores", {})
    for c in CHEAP:
        if c not in out and official.get(c) is not None:
            out[c] = float(official[c])
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    reading = read_json(research)
    bands = reading["coherent_private_two_seed_band_vs_chck82"]
    records: dict[str, Any] = {}
    for seed in SEEDS:
        label = f"coherent_unsplit_special_seed{seed}_alpha0p75"
        train_summary_path = TRAIN_ROOT / f"seed{seed}" / "summary.json"
        payload_path = _public_path('experiments/archive/relation_learning/data/eval_format_replay_corrected/coherent_unsplit_special/eval') / label / "per_target" / f"{label}.json"
        cheap_summary_path = _public_path('experiments/archive/relation_learning/data/eval_format_replay_corrected/coherent_unsplit_special/summary') / f"{label}_summary.json"
        rec: dict[str, Any] = {
            "seed": seed,
            "label": label,
            "train_summary_path": rel(train_summary_path),
            "payload_path": rel(payload_path),
            "cheap_summary_path": rel(cheap_summary_path),
            "train_summary_exists": train_summary_path.exists(),
            "payload_exists": payload_path.exists(),
            "cheap_summary_exists": cheap_summary_path.exists(),
        }
        if train_summary_path.exists():
            ts = read_json(train_summary_path)
            rec["training"] = {
                "updates": ts.get("updates"),
                "total_words": ts.get("total_words"),
                "first_target_ratio": (ts.get("first_update") or {}).get("target_ratio"),
                "last_target_ratio": (ts.get("last_update") or {}).get("target_ratio"),
                "first_initial_private_slow_ce_absdiff": (ts.get("first_update") or {}).get("initial_private_slow_ce_absdiff"),
                "first_readout_kl": (ts.get("first_update") or {}).get("readout_neutral_kl"),
                "final_readout_kl_scale1p0": ts.get("final_readout_neutral_kl"),
                "final_leash_kl_scale1p0": (ts.get("last_update") or {}).get("leash_neutral_kl"),
                "private_rms_max_last": (ts.get("last_update") or {}).get("private_rms_max"),
                "alpha_endpoint_dir": ts.get("alpha_endpoint_dir"),
            }
        scores: dict[str, float] = {}
        if cheap_summary_path.exists():
            cs = read_json(cheap_summary_path)
            scores = {k: float(v) for k, v in (cs.get("scores") or {}).items() if v is not None}
            rec["cheap7"] = cs.get("cheap7")
            rec["cheap7_delta_vs_chck82"] = cs.get("cheap7_delta_vs_chck82")
        elif payload_path.exists():
            scores = extract_scores(read_json(payload_path))
        rec["available_scores"] = scores
        rec["available_deltas_vs_chck82"] = {c: float(scores[c] - CHCK82[c]) for c in scores if c in CHCK82}
        rec["available_deltas_vs_coherent86_s43022"] = {c: float(scores[c] - COH86[c]) for c in scores if c in COH86}
        rec["band_position_vs_chck82"] = {}
        for c, d in rec["available_deltas_vs_chck82"].items():
            b = bands.get(c)
            if not b:
                continue
            rec["band_position_vs_chck82"][c] = {
                "delta": d,
                "coherent_band_min": b["min_delta_vs_chck82"],
                "coherent_band_max": b["max_delta_vs_chck82"],
                "coherent_band_width": b["width"],
                "below_band": d < b["min_delta_vs_chck82"],
                "above_band": d > b["max_delta_vs_chck82"],
            }
        records[str(seed)] = rec

    result = {
        "status": "COHERENT_SPECIAL_SOFAR",
        "created_utc": now(),
        "question": "Does adding official special-token exposure to the exact coherent86 suffix through the private branch create a uniform format benefit, or a column trade?",
        "fixed_frame_sources": [rel(research), rel(MD)],
        "records": records,
        "interpretation": {
            "seed98097": "complete cheap7 evaluation is a trade: BLiMP and COMPS move up, but Supplement, EWoK, Entity, GlobalPIQA, and Reading move down against chck82; cheap7 is below chck82 and coherent86.",
            "seed98098": "available columns already repeat the large Supplement/EWoK damage and BLiMP lift; remaining columns were not inferred here.",
            "entry_result": "coherent_unsplit_special is not a surviving training lever under the fixed two-seed reading rule; do not search alpha values to rescue it.",
            "method_update": "coherent-row readout KL is too narrow for short official-format behavior, so a short isolated no-gradient readout should accompany the coherent readout for every format endpoint."
        },
    }
    out_json = _public_path('experiments/archive/relation_learning/data/coherent_special_sofar/coherent_special_sofar.json')
    out_md = _public_path('research/documents/relation_learning/data/coherent_special_sofar/coherent_special_sofar.md')
    out_json.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = [
        "# research coherent-unsplit-special evidence so far",
        "",
        "This record reads already written files before interpreting the other format arms.",
        "",
        "## Training invariants",
        "",
        "| seed | updates | words | first target ratio | last target ratio | initial CE diff | final coherent readout KL at scale1.0 |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for seed in SEEDS:
        r = records[str(seed)]
        t = r.get("training") or {}
        lines.append(f"| {seed} | {t.get('updates')} | {t.get('total_words')} | {t.get('first_target_ratio')} | {t.get('last_target_ratio')} | {t.get('first_initial_private_slow_ce_absdiff')} | {t.get('final_readout_kl_scale1p0')} |")
    lines += ["", "## Available column movement", "", "| seed | column | score | Δ vs chck82 | Δ vs coherent86_s43022 | relation to coherent two-seed band |", "|---:|---|---:|---:|---:|---|" ]
    for seed in SEEDS:
        r = records[str(seed)]
        scores = r.get("available_scores") or {}
        dch = r.get("available_deltas_vs_chck82") or {}
        dcoh = r.get("available_deltas_vs_coherent86_s43022") or {}
        bpos = r.get("band_position_vs_chck82") or {}
        for c in CHEAP:
            if c not in scores:
                continue
            bp = bpos.get(c, {})
            if bp.get("below_band"):
                pos = "below coherent band"
            elif bp.get("above_band"):
                pos = "above coherent band"
            else:
                pos = "inside coherent band"
            lines.append(f"| {seed} | {c} | {scores[c]:.6f} | {dch[c]:+.6f} | {dcoh[c]:+.6f} | {pos} |")
        if r.get("cheap7") is not None:
            lines.append(f"| {seed} | cheap7 | {float(r['cheap7']):.6f} | {float(r['cheap7_delta_vs_chck82']):+.6f} | {float(r['cheap7']) - mean(COH86[c] for c in CHEAP):+.6f} | below coherent86_s43022 |")
    lines += [
        "",
        "## Interpretation",
        "",
        "Seed 98097 has clean training invariants and a completed cheap7 score of 43.5000, which is -0.45945 against chck82 and -0.68143 against coherent86_s43022.  Its Supplement loss is -1.9478 against chck82 and -2.6500 against coherent86_s43022, while BLiMP rises +0.3787 against chck82.  Seed 98098 has clean training invariants and the available columns repeat the same main face: BLiMP +0.3187 against chck82, Supplement -2.1978, EWoK -0.8355, Entity +0.4060.  Because the repeated movement includes a large damaging Supplement/EWoK face, coherent-unsplit-special does not enter a composed candidate.  The result motivates the short isolated no-gradient readout and a no-special research-trainer control before attributing the loss specifically to special tokens.",
        "",
        f"JSON: `{rel(out_json)}`",
    ]
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "out_json": rel(out_json), "out_md": rel(out_md)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
