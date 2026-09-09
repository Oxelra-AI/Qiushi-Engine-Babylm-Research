#!/usr/bin/env python3
"""Compare legal-40k compact_view_reinvest official results after full collation.

This script is CPU-only and safe to run repeatedly. It waits until both research
legal-40k accumulated-training collations exist, then reports:
  * two legal-40k official vectors and mean;
  * movement from the research legal-16k corrected-tokenizer vectors;
  * movement from the inherited-tokenizer mechanism coordinate (non-submission);
  * whether the route restored Supplement/EWoK without surrendering GlobalPIQA,
    Entity, and COMPS.

It must not be used before both hardened full official collations exist, and it
must not infer missing AoA/SuperGLUE values.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

ROOT = Path(".").resolve()
WS = ROOT / "experiments/archive/representation_and_objectives"
OUT_DIR = WS / "data/legal40k_two_seed_comparison"
NOTE = (ROOT / 'research/notes/representation_and_objectives/legal40k_two_seed_comparison.md')

LEADER = 41.8
TASKS = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "SuperGLUE", "GlobalPIQA", "Reading", "AoA", "Overall"]
FOCUS_RESTORE = ["Supplement", "EWoK"]
PRESERVE = ["GlobalPIQA", "Entity", "COMPS"]

COLLATE_SUMMARIES = {
    "legal40k_43022": WS / "data/legal40k_accum_seed43022_pristine_collate/pristine_collate_legal40k_seed43022_summary.json",
    "legal40k_43122": WS / "data/legal40k_accum_seed43122_pristine_collate/pristine_collate_legal40k_seed43122_summary.json",
}
LEGAL16_COMPARISON = WS / "data/corrected_tokenizer_two_seed_comparison/corrected_tokenizer_two_seed_comparison.json"
INHERITED_430 = WS / "data/pristine_collate_seed43022/pristine_collate_seed43022_summary.json"
INHERITED_431 = WS / "data/pristine_collate_seed43122/pristine_collate_seed43122_summary.json"


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def finite_float(v: Any) -> float | None:
    if isinstance(v, (int, float)) and math.isfinite(float(v)):
        return float(v)
    try:
        x = float(v)
        return x if math.isfinite(x) else None
    except Exception:
        return None


def vector_candidates(rec: Any) -> list[dict[str, Any]]:
    """Return score-vector candidates across the supported summary schemas.

    Legal-16k and legal-40k research-style summaries store the nine official
    columns under ``score_summary.scores`` and Overall at ``score_summary.Overall``.
    The older inherited-tokenizer research/038 summaries store them under
    ``score_summary.official_overall.scores`` and ``...Overall``.  The original
    version of this comparison script looked only for flat score_summary fields,
    which would fail exactly when the legal-40k collations arrived.
    """
    out: list[dict[str, Any]] = []

    def add(scores_obj: Any, top_obj: Any | None = None) -> None:
        if not isinstance(scores_obj, dict):
            return
        cand = dict(scores_obj)
        if isinstance(top_obj, dict):
            for k in ["Overall", "NLP_average", "Human_like_average", "margin_over_visible_leader_41p8", "margin_over_visible_leader"]:
                if k in top_obj and k not in cand:
                    cand[k] = top_obj[k]
        out.append(cand)

    if isinstance(rec, dict):
        ss = rec.get("score_summary")
        if isinstance(ss, dict):
            oo = ss.get("official_overall")
            if isinstance(oo, dict):
                add(oo.get("scores"), oo)
                add(oo, oo)
            add(ss.get("scores"), ss)
            add(ss, ss)
        add(rec.get("scores"), rec)
        add(rec, rec)
    return out


def read_summary_vector(path: Path) -> dict[str, float]:
    rec = load_json(path)
    candidates = vector_candidates(rec)
    for cand in candidates:
        vec: dict[str, float] = {}
        for t in TASKS:
            x = finite_float(cand.get(t))
            if x is not None:
                vec[t] = x
        if all(t in vec for t in TASKS):
            return vec
    available = sorted({k for cand in candidates for k in cand.keys()})
    raise RuntimeError({"missing_or_nonfinite_tasks": [t for t in TASKS if all(finite_float(c.get(t)) is None for c in candidates)], "path": str(path), "available_candidate_keys": available})


def mean_vec(vectors: list[dict[str, float]]) -> dict[str, float]:
    return {t: sum(v[t] for v in vectors) / len(vectors) for t in TASKS}


def delta_vec(a: dict[str, float], b: dict[str, float]) -> dict[str, float]:
    return {t: a[t] - b[t] for t in TASKS}


def read_legal16_vectors() -> dict[str, dict[str, float]]:
    rec = load_json(LEGAL16_COMPARISON)
    vectors: dict[str, dict[str, float]] = {}
    # research comparison stores the accepted legal-16k vectors under
    # ``corrected_tokenizer_results.<seed>.scores``; keep fallback support for
    # older schema names and individual collate summaries.
    if isinstance(rec, dict):
        ctr = rec.get("corrected_tokenizer_results")
        if isinstance(ctr, dict):
            for seed in ["43022", "43122"]:
                maybe = ctr.get(seed)
                if isinstance(maybe, dict):
                    cand = maybe.get("scores") or maybe
                    if isinstance(cand, dict) and all(finite_float(cand.get(t)) is not None for t in TASKS):
                        vectors[seed] = {t: float(cand[t]) for t in TASKS}
        for key in ("seed_vectors", "vectors", "corrected_vectors", "by_seed"):
            obj = rec.get(key)
            if isinstance(obj, dict):
                for seed_key, maybe in obj.items():
                    name = str(seed_key)
                    if "43022" in name or "43122" in name:
                        for cand in vector_candidates(maybe):
                            if all(finite_float(cand.get(t)) is not None for t in TASKS):
                                seed = "43022" if "43022" in name else "43122"
                                vectors[seed] = {t: float(cand[t]) for t in TASKS}
                                break
    if set(vectors) != {"43022", "43122"}:
        # Fallback to the known individual research collate summaries.
        fallback = {
            "43022": WS / "data/strictsmalltok_seed43022_pristine_collate/pristine_collate_strictsmalltok_seed43022_summary.json",
            "43122": WS / "data/strictsmalltok_seed43122_pristine_collate/pristine_collate_strictsmalltok_seed43122_summary.json",
        }
        vectors = {seed: read_summary_vector(path) for seed, path in fallback.items() if path.exists()}
    if set(vectors) != {"43022", "43122"}:
        raise RuntimeError({"could_not_load_legal16_vectors": str(LEGAL16_COMPARISON), "loaded": sorted(vectors)})
    return vectors


def load_inputs() -> tuple[dict[str, dict[str, float]], dict[str, dict[str, float]], dict[str, dict[str, float]], list[str]]:
    waiting = [str(p) for p in COLLATE_SUMMARIES.values() if not p.exists()]
    if waiting:
        return {}, {}, {}, waiting
    legal40 = {"43022": read_summary_vector(COLLATE_SUMMARIES["legal40k_43022"]), "43122": read_summary_vector(COLLATE_SUMMARIES["legal40k_43122"])}
    legal16 = read_legal16_vectors()
    inherited = {"43022": read_summary_vector(INHERITED_430), "43122": read_summary_vector(INHERITED_431)}
    return legal40, legal16, inherited, []


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    legal40, legal16, inherited, waiting = load_inputs()
    if waiting:
        payload = {
            "status": "WAITING_FOR_LEGAL40K_FULL_COLLATIONS",
            "missing": waiting,
            "purpose": "Run after both hardened research legal-40k official collations exist; do not infer missing scores.",
        }
        out_json = OUT_DIR / "legal40k_two_seed_comparison.json"
        out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(json.dumps({"status": payload["status"], "missing": waiting, "out_json": str(out_json)}, indent=2))
        return

    legal40_mean = mean_vec(list(legal40.values()))
    legal16_mean = mean_vec(list(legal16.values()))
    inherited_mean = mean_vec(list(inherited.values()))
    by_seed = {}
    for seed in ["43022", "43122"]:
        by_seed[seed] = {
            "legal40k": legal40[seed],
            "legal16k": legal16[seed],
            "inherited16k_non_submission": inherited[seed],
            "delta_legal40k_minus_legal16k": delta_vec(legal40[seed], legal16[seed]),
            "delta_legal40k_minus_inherited16k": delta_vec(legal40[seed], inherited[seed]),
            "margin_over_visible_leader_41p8": legal40[seed]["Overall"] - LEADER,
        }
    mean_delta_40_minus_16 = delta_vec(legal40_mean, legal16_mean)
    mean_delta_40_minus_inherited = delta_vec(legal40_mean, inherited_mean)

    restored = {t: mean_delta_40_minus_16[t] for t in FOCUS_RESTORE}
    preserved = {t: mean_delta_40_minus_16[t] for t in PRESERVE}
    both_above = all(legal40[s]["Overall"] > LEADER for s in legal40)
    any_above = any(legal40[s]["Overall"] > LEADER for s in legal40)
    scientific_reading = []
    if both_above:
        scientific_reading.append("Both legal-40k seeds clear the visible 41.8 leader; protect the route and proceed to endpoint robustness/provenance packaging after independent review.")
    elif any_above:
        scientific_reading.append("Only one legal-40k seed clears the visible leader; preserve it as a legal endpoint candidate while localizing seed-sensitive columns before any submission packaging.")
    else:
        scientific_reading.append("Neither legal-40k seed clears the visible leader; do not spend another full pair on intermediate vocabulary sizes. Move to a different curriculum or architecture route.")
    scientific_reading.append("The load-bearing pattern is whether Supplement/EWoK recover from legal-16k while GlobalPIQA/Entity/COMPS do not surrender the compact-view gains.")

    payload = {
        "status": "LEGAL40K_TWO_SEED_COMPARISON",
        "visible_leader": LEADER,
        "legal40k_mean": legal40_mean,
        "legal16k_mean": legal16_mean,
        "inherited16k_non_submission_mean": inherited_mean,
        "mean_delta_legal40k_minus_legal16k": mean_delta_40_minus_16,
        "mean_delta_legal40k_minus_inherited16k": mean_delta_40_minus_inherited,
        "restore_focus_delta_vs_legal16k": restored,
        "preserve_focus_delta_vs_legal16k": preserved,
        "by_seed": by_seed,
        "both_legal40k_seeds_above_visible_leader": both_above,
        "any_legal40k_seed_above_visible_leader": any_above,
        "scientific_reading": scientific_reading,
    }
    out_json = OUT_DIR / "legal40k_two_seed_comparison.json"
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    NOTE.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# research legal-40k two-seed comparison",
        "",
        f"Visible leader baseline: {LEADER:.3f}",
        "",
        "## Overall",
        f"- legal40k seed43022: {legal40['43022']['Overall']:.6f} (margin {legal40['43022']['Overall']-LEADER:+.6f})",
        f"- legal40k seed43122: {legal40['43122']['Overall']:.6f} (margin {legal40['43122']['Overall']-LEADER:+.6f})",
        f"- legal40k mean: {legal40_mean['Overall']:.6f}",
        f"- legal16k mean: {legal16_mean['Overall']:.6f}",
        f"- mean delta legal40k - legal16k: {mean_delta_40_minus_16['Overall']:+.6f}",
        "",
        "## Restore/preserve pattern vs legal16k",
    ]
    for t in FOCUS_RESTORE + PRESERVE:
        lines.append(f"- {t}: {mean_delta_40_minus_16[t]:+.6f}")
    lines.append("")
    lines.append("## Scientific reading")
    for item in scientific_reading:
        lines.append(f"- {item}")
    lines.append("")
    lines.append(f"JSON: `{out_json}`")
    NOTE.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps({
        "status": payload["status"],
        "seed43022_overall": legal40["43022"]["Overall"],
        "seed43122_overall": legal40["43122"]["Overall"],
        "mean_overall": legal40_mean["Overall"],
        "mean_delta_vs_legal16k": mean_delta_40_minus_16["Overall"],
        "both_above": both_above,
        "any_above": any_above,
        "out_json": str(out_json),
        "note": str(NOTE),
    }, indent=2))


if __name__ == "__main__":
    main()
