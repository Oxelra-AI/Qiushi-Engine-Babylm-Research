#!/usr/bin/env python3
"""research: read corrected decisive GPU scores for register/in-corpus contrasts.

File-only readout for outputs of decisive_gpu_scorer.py. Joins scored
arms to the research MAX-geometry clean reference and to research pre-score
register predictions. It can run before all scores are present and will report
partial data state.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative

import csv, json, math, pathlib, statistics, time
from typing import Any


def find_root() -> pathlib.Path:
    return _PUBLIC_ROOT

ROOT = find_root()
WS = ROOT / "experiments/archive" / 'frontier_consolidation'
GPU = WS / "data" / "decisive_gpu_eval" / "per_target"
CLEAN = WS / "data" / "deberta_maxgeom_clean_stable_eval" / "eval" / "per_target"
PRED = WS / "data" / "distribution_proximity_prediction" / "prediction_commitment.json"
OUT = WS / "data" / "decisive_comparison_readout"
FAMS = ["BLiMP", "Supplement", "EWoK", "COMPS", "Entity"]
EXE = ["BLiMP", "Supplement", "EWoK", "COMPS"]


def now(): return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
def rel(p: pathlib.Path) -> str:
    try: return str(p.relative_to(ROOT))
    except Exception: return str(p)
def finite(x: Any) -> bool:
    try: return x is not None and math.isfinite(float(x))
    except Exception: return False

def load_score_file(path: pathlib.Path) -> dict[str, Any]:
    d = json.loads(path.read_text(encoding="utf-8"))
    arm = d.get("arm") or d.get("data_arm") or "clean"
    ck = d.get("checkpoint") or d.get("endpoint") or path.stem.split("_")[-1]
    tasks = d.get("tasks") or {}
    scores = {}
    for f in FAMS:
        rec = tasks.get(f) or {}
        s = rec.get("score") if isinstance(rec, dict) else None
        scores[f] = float(s) if finite(s) else None
    return {"arm": arm, "checkpoint": ck, "scores": scores, "path": rel(path)}

def load_gpu() -> dict[str, dict[str, dict[str, Any]]]:
    out = {}
    if not GPU.exists(): return out
    for p in sorted(GPU.glob("*.json")):
        r = load_score_file(p)
        out.setdefault(r["arm"], {})[r["checkpoint"]] = r
    return out

def load_clean() -> dict[str, dict[str, Any]]:
    out = {}
    if not CLEAN.exists(): return out
    for p in sorted(CLEAN.glob("*.json")):
        d = json.loads(p.read_text(encoding="utf-8"))
        ck = d.get("endpoint") or d.get("checkpoint") or p.stem.split("_")[-1]
        tasks = d.get("tasks") or {}
        scores = {}
        for f in FAMS:
            rec = tasks.get(f) or {}
            s = rec.get("score") if isinstance(rec, dict) else None
            scores[f] = float(s) if finite(s) else None
        out[ck] = {"scores": scores, "path": rel(p)}
    return out

def avg(scores: dict[str, Any], fams: list[str]) -> float | None:
    vals = [scores.get(f) for f in fams]
    return statistics.mean(float(v) for v in vals) if all(finite(v) for v in vals) else None

def diff(a: float | None, b: float | None) -> float | None:
    return round(a-b, 6) if a is not None and b is not None else None

def family_scores(r: dict[str, Any] | None) -> dict[str, float | None]:
    return (r or {}).get("scores", {}) if r else {}

def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    gpu = load_gpu(); clean = load_clean()
    pred = json.loads(PRED.read_text(encoding="utf-8")) if PRED.exists() else {}
    rows = []
    for arm, ckmap in sorted(gpu.items()):
        for ck, r in sorted(ckmap.items()):
            s = r["scores"]; cs = family_scores(clean.get(ck))
            row = {"arm": arm, "checkpoint": ck, "source": r["path"]}
            for f in FAMS:
                row[f] = s.get(f); row[f"delta_{f}_vs_clean"] = diff(s.get(f), cs.get(f))
            row["exEntity4"] = avg(s, EXE); row["cheap5_noReading"] = avg(s, FAMS)
            row["delta_exEntity4_vs_clean"] = diff(row["exEntity4"], avg(cs, EXE))
            row["delta_cheap5_vs_clean"] = diff(row["cheap5_noReading"], avg(cs, FAMS))
            row["complete_families"] = sum(1 for f in FAMS if finite(s.get(f)))
            rows.append(row)

    contrast_rows = []
    def add_contrast(name: str, arm_a: str, arm_b: str, ck: str):
        ra = gpu.get(arm_a, {}).get(ck); rb = gpu.get(arm_b, {}).get(ck)
        if not ra or not rb: return
        sa, sb = ra["scores"], rb["scores"]
        row = {"contrast": name, "checkpoint": ck, "arm_a": arm_a, "arm_b": arm_b}
        for f in FAMS: row[f] = diff(sa.get(f), sb.get(f))
        row["exEntity4"] = diff(avg(sa, EXE), avg(sb, EXE))
        row["cheap5_noReading"] = diff(avg(sa, FAMS), avg(sb, FAMS))
        row["complete_families"] = sum(1 for f in FAMS if finite(sa.get(f)) and finite(sb.get(f)))
        contrast_rows.append(row)
    for ck in ["chck_80M", "chck_100M"]:
        add_contrast("childspeech_minus_adultprose", "regmax_childspeech", "regmax_adultprose", ck)
    for ck in ["chck_70M"]:
        add_contrast("incorpus_minus_full1x", "incorpus_adultprose", "subdose_full", ck)
    for ck in ["chck_70M", "chck_80M", "chck_100M"]:
        # vs clean is kept as rows through deltas, but store aggregate too.
        ri = gpu.get("incorpus_adultprose", {}).get(ck)
        if ri and ck in clean:
            si, sc = ri["scores"], clean[ck]["scores"]
            contrast_rows.append({
                "contrast": "incorpus_minus_clean", "checkpoint": ck,
                "arm_a": "incorpus_adultprose", "arm_b": "clean_maxgeom",
                **{f: diff(si.get(f), sc.get(f)) for f in FAMS},
                "exEntity4": diff(avg(si, EXE), avg(sc, EXE)),
                "cheap5_noReading": diff(avg(si, FAMS), avg(sc, FAMS)),
                "complete_families": sum(1 for f in FAMS if finite(si.get(f)) and finite(sc.get(f))),
            })

    def write_csv(path, rs):
        if not rs: path.write_text("", encoding="utf-8"); return
        fields=[]
        for r in rs:
            for k in r:
                if k not in fields: fields.append(k)
        with path.open("w", encoding="utf-8", newline="") as f:
            w=csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rs)
    write_csv(OUT/"arm_scores_vs_clean.csv", rows)
    write_csv(OUT/"decisive_contrasts.csv", contrast_rows)
    summary = {
        "status": "DECISIVE_COMPARISON_READOUT",
        "created_utc": now(),
        "gpu_score_dir": rel(GPU),
        "clean_score_dir": rel(CLEAN),
        "arms_seen": {a: sorted(m) for a,m in gpu.items()},
        "n_arm_rows": len(rows),
        "n_contrast_rows": len(contrast_rows),
        "contrast_rows": contrast_rows,
        "prediction_commitment_excerpt": {
            "contrast_definition": pred.get("contrast_definition"),
            "profile_exEntity5": (((pred.get("committed_profile_js_model") or {}).get("predictions") or {}).get("exEntity5")),
            "strict_profile_exEntity5": (((pred.get("strict_eval_text_extraction_robustness") or {}).get("aggregate_predictions") or {}).get("profile_js", {}) or {}).get("exEntity5"),
            "word_control_exEntity5": (((pred.get("lexical_word_js_control") or {}).get("predictions") or {}).get("exEntity5")),
        },
        "files": {"json": rel(OUT/"decisive_comparison_readout.json"), "arm_csv": rel(OUT/"arm_scores_vs_clean.csv"), "contrast_csv": rel(OUT/"decisive_contrasts.csv"), "md": rel(OUT/"decisive_comparison_readout.md")},
        "no_model_loading_training_official_evaluation_gpu_globalpiqa_superglue_aoa_upload_or_leaderboard": True,
    }
    (OUT/"decisive_comparison_readout.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")
    lines=["# research decisive comparison readout", "", f"Arms seen: `{summary['arms_seen']}`", "", "## Contrasts", "", "| contrast | checkpoint | BLiMP | Supplement | EWoK | COMPS | Entity | exEntity4 | cheap5 | complete families |", "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    def fmt(x): return "NA" if x is None else f"{float(x):.4f}"
    for r in contrast_rows:
        lines.append(f"| {r['contrast']} | {r['checkpoint']} | {fmt(r.get('BLiMP'))} | {fmt(r.get('Supplement'))} | {fmt(r.get('EWoK'))} | {fmt(r.get('COMPS'))} | {fmt(r.get('Entity'))} | {fmt(r.get('exEntity4'))} | {fmt(r.get('cheap5_noReading'))} | {r.get('complete_families')} |")
    lines += ["", "## Prediction anchor", "", f"Register childspeech-minus-adultprose profile prediction exEntity5: {summary['prediction_commitment_excerpt']['profile_exEntity5']}; strict eval-text profile: {summary['prediction_commitment_excerpt']['strict_profile_exEntity5']}; word control: {summary['prediction_commitment_excerpt']['word_control_exEntity5']}.", "", f"JSON: `{summary['files']['json']}`"]
    (OUT/"decisive_comparison_readout.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    print(json.dumps({"status": summary["status"], "arms_seen": summary["arms_seen"], "n_contrast_rows": len(contrast_rows), "summary_md": summary["files"]["md"]}, indent=2), flush=True)

if __name__ == "__main__": main()
