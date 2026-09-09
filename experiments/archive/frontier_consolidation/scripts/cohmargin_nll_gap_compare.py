#!/usr/bin/env python3
"""Summarize research coherent/disrupted NLL-gap probes.

Reads the mechanism-probe outputs for the research coherence-margin pilot and
ordinary scale1.75 references at coherent-word matched 2M and charged-word
matched 4M checkpoints. Produces a compact scientific interpretation for whether
margin training has opened coherent-over-disrupted likelihood separation.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import json
import math
import pathlib
import time
from typing import Any

ROOT = _public_path('.')
WORKSPACE = _public_path('experiments/archive/frontier_consolidation')
OUT = _public_path('experiments/archive/frontier_consolidation/data/cohmargin_nll_gap_comparison')

PATHS = {
    "pilot_train64": _public_path('experiments/archive/frontier_consolidation/data/cohmargin_nll_gap_pilot_train64/cohmargin_nll_gap_probe.json'),
    "pilot_holdout64_after2M": _public_path('experiments/archive/frontier_consolidation/data/cohmargin_nll_gap_pilot_holdout64/cohmargin_nll_gap_probe.json'),
    "ref2m_train64": _public_path('experiments/archive/frontier_consolidation/data/cohmargin_nll_gap_ref2m_train64/cohmargin_nll_gap_probe.json'),
    "ref2m_holdout64_after2M": _public_path('experiments/archive/frontier_consolidation/data/cohmargin_nll_gap_ref2m_holdout64/cohmargin_nll_gap_probe.json'),
    "ref4m_train64": _public_path('experiments/archive/frontier_consolidation/data/cohmargin_nll_gap_ref4m_train64/cohmargin_nll_gap_probe.json'),
    "ref4m_holdout64_after2M": _public_path('experiments/archive/frontier_consolidation/data/cohmargin_nll_gap_ref4m_holdout64/cohmargin_nll_gap_probe.json'),
}


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def rel(p: pathlib.Path | str) -> str:
    q = pathlib.Path(p)
    try:
        return str(q.resolve().relative_to(ROOT))
    except Exception:
        return str(q)


def read_json(p: pathlib.Path) -> dict[str, Any]:
    return json.loads(p.read_text(encoding="utf-8"))


def rec(j: dict[str, Any]) -> dict[str, Any]:
    return {
        "label": j.get("label"),
        "model_path": j.get("model_path"),
        "rows_loaded": j.get("row_meta", {}).get("num_rows_loaded"),
        "loaded_words": j.get("row_meta", {}).get("loaded_words"),
        "skip_coherent_words": j.get("row_meta", {}).get("skip_coherent_words"),
        "n_masked_tokens": int(j["n_masked_tokens"]),
        "coherent_nll_mean": float(j["coherent_nll_mean"]),
        "disrupted_nll_mean": float(j["disrupted_nll_mean"]),
        "bad_minus_coh_nll_mean": float(j["bad_minus_coh_nll_mean"]),
        "bad_minus_coh_nll_se": float(j["bad_minus_coh_nll_se"]),
        "z_vs_zero": float(j["bad_minus_coh_nll_z_vs_zero"]),
        "gap_positive_fraction": float(j["gap_positive_fraction"]),
        "source": rel(PATHS_BY_LABEL[j.get("label", "")]) if j.get("label") in PATHS_BY_LABEL else None,
    }


def diff(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    d = float(a["bad_minus_coh_nll_mean"] - b["bad_minus_coh_nll_mean"])
    se = math.sqrt(float(a["bad_minus_coh_nll_se"]) ** 2 + float(b["bad_minus_coh_nll_se"]) ** 2)
    return {"delta_gap": d, "se_independent": se, "z_independent": d / se if se > 0 else None}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    raw = {k: read_json(p) for k, p in PATHS.items()}
    by = {k: rec(v) for k, v in raw.items()}
    comparisons = {
        "train64_pilot_minus_ref2m": diff(by["pilot_train64"], by["ref2m_train64"]),
        "holdout64_pilot_minus_ref2m": diff(by["pilot_holdout64_after2M"], by["ref2m_holdout64_after2M"]),
        "train64_pilot_minus_ref4m": diff(by["pilot_train64"], by["ref4m_train64"]),
        "holdout64_pilot_minus_ref4m": diff(by["pilot_holdout64_after2M"], by["ref4m_holdout64_after2M"]),
        "train64_ref4m_minus_ref2m": diff(by["ref4m_train64"], by["ref2m_train64"]),
        "holdout64_ref4m_minus_ref2m": diff(by["ref4m_holdout64_after2M"], by["ref2m_holdout64_after2M"]),
    }
    interpretation = (
        "On these fixed legal-row probes, the research lambda>0 pilot has only near-zero coherent-over-disrupted "
        "preference and is close to the coherent-word matched 2M ordinary reference, while the 4M ordinary reference "
        "already shows a much larger positive preference. Therefore the short margin pilot has not yet established the "
        "intended coherent-context likelihood separation; any encouraging official score must be isolated against a "
        "same-charge same-row lambda-zero arm before being attributed to the margin signal."
    )
    out = {
        "status": "COMPLETE",
        "created_utc": now(),
        "records": by,
        "comparisons": comparisons,
        "scientific_interpretation": interpretation,
        "policy_implication": "Do not extend coherence-margin to 20M from pilot score alone. If cheap7 is promising, run the minimal 4M charged lambda-zero same-row arm first and compare both official cheap7 and this NLL-gap probe.",
    }
    js = _public_path('experiments/archive/frontier_consolidation/data/cohmargin_nll_gap_comparison/cohmargin_nll_gap_comparison.json')
    md = _public_path('research/documents/frontier_consolidation/data/cohmargin_nll_gap_comparison/cohmargin_nll_gap_comparison.md')
    js.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = [
        "# research coherence-margin NLL-gap comparison",
        "",
        f"Status: **{out['status']}**",
        "",
        "| probe | coherent NLL | disrupted NLL | bad-coh gap | SE | z | pos frac |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for k in ["pilot_train64", "ref2m_train64", "ref4m_train64", "pilot_holdout64_after2M", "ref2m_holdout64_after2M", "ref4m_holdout64_after2M"]:
        r = by[k]
        lines.append(
            f"| {k} | {r['coherent_nll_mean']:.6f} | {r['disrupted_nll_mean']:.6f} | {r['bad_minus_coh_nll_mean']:+.6f} | {r['bad_minus_coh_nll_se']:.6f} | {r['z_vs_zero']:+.2f} | {r['gap_positive_fraction']:.4f} |"
        )
    lines += [
        "",
        "## Key contrasts",
        "",
        "| contrast | delta gap | SE | z |",
        "|---|---:|---:|---:|",
    ]
    for k, v in comparisons.items():
        z = v["z_independent"]
        lines.append(f"| {k} | {v['delta_gap']:+.6f} | {v['se_independent']:.6f} | {z:+.2f} |")
    lines += ["", interpretation, "", out["policy_implication"], "", f"JSON: `{rel(js)}`"]
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": out["status"], "out_json": rel(js), "out_md": rel(md), "pilot_train_gap": by["pilot_train64"]["bad_minus_coh_nll_mean"], "ref2m_train_gap": by["ref2m_train64"]["bad_minus_coh_nll_mean"], "ref4m_train_gap": by["ref4m_train64"]["bad_minus_coh_nll_mean"]}, indent=2), flush=True)


PATHS_BY_LABEL: dict[str, pathlib.Path] = {}
# Populate after function definitions so the source path is recoverable in records.
for _name, _path in PATHS.items():
    try:
        _label = json.loads(_path.read_text(encoding="utf-8")).get("label")
        if _label:
            PATHS_BY_LABEL[str(_label)] = _path
    except Exception:
        pass


if __name__ == "__main__":
    main()
