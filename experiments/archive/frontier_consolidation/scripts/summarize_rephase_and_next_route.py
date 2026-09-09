#!/usr/bin/env python3
"""research: summarize tail-rephase score probe and route implications."""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import json
import pathlib
import time
from typing import Any


def find_user_root() -> pathlib.Path:
    return _PUBLIC_ROOT


USER_ROOT = find_user_root()
STUDY = USER_ROOT / "experiments/archive/frontier_consolidation"
WORKSPACE = STUDY
OUT = WORKSPACE / "data/tail_rephase_summary"
NOTE = WORKSPACE / "notes/tail_rephase_probe_and_experience_utilization_route.md"

LEGAL100 = {
    "label": "legal100M_ref",
    "BLiMP": 65.8707,
    "Supplement": 61.1657,
    "EWoK": 50.3932,
    "Entity": 27.4008,
    "COMPS": 52.0083,
    "GlobalPIQA": 36.065,
    "Reading": 8.14,
}

TARGETS = [
    ("continuous_90M", WORKSPACE / "data/rephase_eval/per_target/continuous_chck_90M.json"),
    ("matched_lr_chunkshuffle_90M", WORKSPACE / "data/rephase_eval/per_target/matched_lr_chck_90M.json"),
    ("base_lr_chunkshuffle_90M", WORKSPACE / "data/rephase_eval/per_target/base_lr_chck_90M.json"),
    ("matched_lr_original_tail_90M", WORKSPACE / "data/tail_replay_eval/per_target/tail_replay_matched_lr_chck_90M.json"),
]

DRYRUNS = {
    "U256": WORKSPACE / "data/experience_utilization_dryruns/U256/dryrun_metrics.json",
    "U64_128_256": WORKSPACE / "data/experience_utilization_dryruns/U64_128_256/dryrun_metrics.json",
}
INTERP = WORKSPACE / "data/rephase_probe_interpretation/rephase_probe_interpretation_and_tail_assets.json"


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def get_task_score(tasks: dict[str, Any], name: str) -> float:
    rec = tasks.get(name, {})
    if rec.get("accuracy") is not None:
        return float(rec["accuracy"])
    if rec.get("score") is not None:
        return float(rec["score"])
    scores = rec.get("scores") if isinstance(rec.get("scores"), dict) else {}
    if name == "Reading" and scores.get("Reading") is not None:
        return float(scores["Reading"])
    raise KeyError(f"missing score for {name}")


def parse_payload(label: str, path: pathlib.Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    tasks = payload.get("tasks", {})
    gpar = get_task_score(tasks, "GlobalPIQA_parallel")
    gnon = get_task_score(tasks, "GlobalPIQA_nonparallel")
    rec = {
        "label": label,
        "path": str(path.relative_to(USER_ROOT)),
        "BLiMP": get_task_score(tasks, "BLiMP"),
        "Supplement": get_task_score(tasks, "Supplement"),
        "EWoK": get_task_score(tasks, "EWoK"),
        "Entity": get_task_score(tasks, "Entity"),
        "COMPS": get_task_score(tasks, "COMPS"),
        "GlobalPIQA_parallel": gpar,
        "GlobalPIQA_nonparallel": gnon,
        "GlobalPIQA": (gpar + gnon) / 2.0,
        "Reading": get_task_score(tasks, "Reading"),
    }
    keys = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA", "Reading"]
    rec["cheap7"] = sum(rec[k] for k in keys) / 7.0
    return rec


def cheap7(rec: dict[str, Any]) -> float:
    keys = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA", "Reading"]
    return sum(float(rec[k]) for k in keys) / 7.0


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    ref = dict(LEGAL100)
    ref["cheap7"] = cheap7(ref)
    rows.append(ref)
    for label, path in TARGETS:
        if not path.exists():
            raise FileNotFoundError(path)
        rows.append(parse_payload(label, path))
    ref_c7 = ref["cheap7"]
    cont90 = next(r for r in rows if r["label"] == "continuous_90M")
    matched = next(r for r in rows if r["label"] == "matched_lr_chunkshuffle_90M")
    base = next(r for r in rows if r["label"] == "base_lr_chunkshuffle_90M")
    tail = next(r for r in rows if r["label"] == "matched_lr_original_tail_90M")
    for r in rows:
        r["delta_vs_legal100M_cheap7"] = r["cheap7"] - ref_c7
        r["delta_vs_continuous90M_cheap7"] = r["cheap7"] - cont90["cheap7"]
    interp = json.loads(INTERP.read_text(encoding="utf-8"))
    dry = {}
    for label, path in DRYRUNS.items():
        payload = json.loads(path.read_text(encoding="utf-8"))
        dry[label] = {
            "path": str(path.relative_to(USER_ROOT)),
            "total_steps_executed": payload.get("total_steps_executed"),
            "total_charged_words": payload.get("total_charged_words"),
            "raw_tokens_per_epoch": payload.get("raw_tokens_per_epoch"),
            "verification": payload.get("verification"),
            "stage_records": payload.get("stage_records"),
        }

    summary = {
        "status": "TAIL_REPHASE_SUMMARY",
        "created_utc": now(),
        "score_rows": rows,
        "scientific_reading": {
            "matched_lr_chunkshuffle_movement": matched["delta_vs_legal100M_cheap7"],
            "matched_lr_original_tail_movement": tail["delta_vs_legal100M_cheap7"],
            "original_tail_vs_continuous90M": tail["delta_vs_continuous90M_cheap7"],
            "base_lr_chunkshuffle_movement": base["delta_vs_legal100M_cheap7"],
            "reading": "The only large-ish restart movement appears in the research token-chunk shuffle presentation. It is not reproduced when the matched-LR restart consumes the original post-80M row stream. The optimizer-state/tail-LR package is therefore not a route to the legal SOTA gap.",
        },
        "probe_confounds": interp.get("trainer_differences"),
        "original_tail_segment": interp.get("tail_segment"),
        "experience_utilization_dryruns": dry,
        "next_route_candidate": {
            "name": "faithful word-boundary experience utilization on the protected legal16 compact-view reinvest stream",
            "why_distinct": "It changes which charged corpus tokens are visible and predictable under the same word budget, rather than changing mask rate, optimizer, benchmark-specific targets, or final-task engineering.",
            "current_cpu_status": "A01 trainer runs successfully on A02 legal16 pool in dry-run mode for U256 and U64_128_256, both with 100M charged words and 2,530 steps.",
            "not_yet_gpu_authorized": "Conditional on the completed SGCR endpoint: a one-arm U64_128_256 or paired U256/U64_128_256 comparison remains proposed, depending on the scientific contrast needed.",
        },
    }
    out_json = OUT / "tail_rephase_summary.json"
    out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    table = [
        "# research tail-rephase score probe and next-route reading",
        "",
        "## Score probe",
        "",
        "| model | BLiMP | Suppl | EWoK | Entity | COMPS | GPIQA | Reading | cheap7 | Δ vs legal100M | Δ vs continuous90M |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        table.append(
            f"| {r['label']} | {r['BLiMP']:.2f} | {r['Supplement']:.2f} | {r['EWoK']:.2f} | "
            f"{r['Entity']:.2f} | {r['COMPS']:.2f} | {r['GlobalPIQA']:.3f} | {r['Reading']:.3f} | "
            f"{r['cheap7']:.4f} | {r['delta_vs_legal100M_cheap7']:+.4f} | {r['delta_vs_continuous90M_cheap7']:+.4f} |"
        )
    table += [
        "",
        "## Reading",
        "",
        f"- research matched-LR token-chunk shuffle 90M: cheap7 `{matched['cheap7']:.4f}`, Δ vs legal100M `{matched['delta_vs_legal100M_cheap7']:+.4f}`, Δ vs continuous90M `{matched['delta_vs_continuous90M_cheap7']:+.4f}`.",
        f"- research matched-LR original-tail replay 90M: cheap7 `{tail['cheap7']:.4f}`, Δ vs legal100M `{tail['delta_vs_legal100M_cheap7']:+.4f}`, Δ vs continuous90M `{tail['delta_vs_continuous90M_cheap7']:+.4f}`.",
        f"- Base-LR token-chunk shuffle 90M: cheap7 `{base['cheap7']:.4f}`, Δ vs legal100M `{base['delta_vs_legal100M_cheap7']:+.4f}`.",
        "- The original-tail replay consumes row 518144 through 647399 of the frozen 100M stream, 19,965,632 words, ending at actual 100,000,000 words. It removes the post-80M data-order/presentation change from the matched-LR comparison.",
        "- The larger research matched-LR movement does not reproduce under the original row stream. Fresh optimizer state / low-LR tail rephase is not the missing broad mechanism.",
        "- No full official evaluation, second seed, 95M/100M endpoint sweep, base-LR replay, or optimizer/LR continuation is supported by these numbers.",
        "",
        "## CPU-only route asset preserved",
        "",
        "The distinct next candidate is faithful word-boundary experience utilization on the legal16 compact-view reinvest stream. It preserves the corpus and tokenizer but changes whether every charged word's tokenizer tokens become visible and predictable within the 100M-word allowance.",
        "",
        "| arm | charged words | steps | active tokens per epoch | verification |",
        "|---|---:|---:|---:|---|",
    ]
    for label, d in dry.items():
        ver = d["verification"]
        active = ver.get("active_tokens_per_epoch", [None])[0] if ver else None
        ok = bool(ver and ver.get("words_match_100M") and ver.get("steps_match_expected") and ver.get("all_epoch_words_10M") and ver.get("all_epoch_steps_253"))
        table.append(f"| {label} | {d['total_charged_words']} | {d['total_steps_executed']} | {active} | {ok} |")
    table += [
        "",
        f"JSON: `{out_json.relative_to(USER_ROOT)}`",
    ]
    out_md = OUT / "tail_rephase_summary.md"
    out_md.write_text("\n".join(table) + "\n", encoding="utf-8")
    NOTE.write_text("\n".join(table) + "\n", encoding="utf-8")

    print(json.dumps({
        "status": summary["status"],
        "out_json": str(out_json.relative_to(USER_ROOT)),
        "out_md": str(out_md.relative_to(USER_ROOT)),
        "note": str(NOTE.relative_to(USER_ROOT)),
        "matched_chunk_delta": matched["delta_vs_legal100M_cheap7"],
        "tail_replay_delta": tail["delta_vs_legal100M_cheap7"],
        "tail_vs_cont90": tail["delta_vs_continuous90M_cheap7"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
