#!/usr/bin/env python3
"""Link tokenizer pressure to corrected 100M EWoK focus margins.

CPU-only, no model inference. Uses already-scored corrected-tokenizer 100M focused
EWoK margins and computes old-tokenizer vs legal-tokenizer token-length
features for the same candidate sentences/completions. Purpose: determine whether
focused relation behavior under the legal tokenizer is mainly explained by crude
fragmentation/length pressure, or whether official score movement should be read
as changed learned relation dynamics.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import json
import math
import statistics
import time
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

HERE = _public_path('experiments/archive/representation_and_objectives/scripts/ewok_focus_tokenization_margin_link.py')
A01 = _public_path('experiments/archive/representation_and_objectives')
USER_ROOT = _public_path('.')
WORKSPACE = _public_path('experiments/archive/representation_and_objectives')
OLD_TOKENIZER = _public_path('experiments/archive/initial_model_studies/training/runs/fullcycle_official_wwm_debertav2_8x480_seed43_100M_b256/hf_model')
NEW_TOKENIZER = _public_path('experiments/archive/representation_and_objectives/data/strictsmall_tokenizer_retrain/strictsmall_compact_reinvest_16k_tokenizer')
MARGINS = _public_path('experiments/archive/representation_and_objectives/data/strictsmalltok_midtrain_ewok_margin_probe/strictsmalltok_chck_100M_focus553_margins.json')
OUT_DIR = _public_path('experiments/archive/representation_and_objectives/data/ewok_focus_tokenization_margin_link')
OUT_JSON = _public_path('experiments/archive/representation_and_objectives/data/ewok_focus_tokenization_margin_link/ewok_focus_tokenization_margin_link.json')
OUT_CSV = _public_path('experiments/archive/representation_and_objectives/data/ewok_focus_tokenization_margin_link/ewok_focus_tokenization_margin_link_rows.csv')
OUT_MD = _public_path('research/notes/representation_and_objectives/ewok_focus_tokenization_margin_link.md')


def now_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def rel(p: Path | str) -> str:
    p = Path(p)
    try:
        return str(p.relative_to(USER_ROOT))
    except Exception:
        return str(p)


def pearson(xs: list[float], ys: list[float]) -> float | None:
    pairs = [(x, y) for x, y in zip(xs, ys) if math.isfinite(x) and math.isfinite(y)]
    if len(pairs) < 3:
        return None
    xs2, ys2 = zip(*pairs)
    mx, my = statistics.mean(xs2), statistics.mean(ys2)
    vx = sum((x - mx) ** 2 for x in xs2)
    vy = sum((y - my) ** 2 for y in ys2)
    if vx <= 0 or vy <= 0:
        return None
    return sum((x - mx) * (y - my) for x, y in pairs) / math.sqrt(vx * vy)


def tokenize_len(tok, text: str) -> int:
    return len(tok(text, add_special_tokens=False, truncation=False)["input_ids"])


def target_token_len(tok, sentence: str, completion: str) -> int:
    enc = tok(sentence, return_offsets_mapping=True, add_special_tokens=False, truncation=False)
    start_char_idx = len(sentence) - len(completion)
    return sum(1 for start, end in enc["offset_mapping"] if end > start_char_idx)


def summarize_vals(vals: list[float]) -> dict[str, Any]:
    vals = [float(v) for v in vals if math.isfinite(float(v))]
    if not vals:
        return {"n": 0}
    vals2 = sorted(vals)
    def q(p: float) -> float:
        return vals2[min(len(vals2)-1, max(0, round((len(vals2)-1)*p)))]
    return {"n": len(vals), "mean": statistics.mean(vals), "std": statistics.pstdev(vals) if len(vals)>1 else 0.0, "min": vals2[0], "p25": q(0.25), "p50": q(0.5), "p75": q(0.75), "max": vals2[-1]}


def score_rows(data: dict[str, Any]) -> list[dict[str, Any]]:
    r430 = {r["uid"]: r for r in data["results_by_model"]["strict50_43022"]["records"]}
    r431 = {r["uid"]: r for r in data["results_by_model"]["strict50_43122"]["records"]}
    out = []
    old_tok = AutoTokenizer.from_pretrained(str(OLD_TOKENIZER), use_fast=True, local_files_only=True)
    new_tok = AutoTokenizer.from_pretrained(str(NEW_TOKENIZER), use_fast=True, local_files_only=True)
    for uid, a in r430.items():
        b = r431[uid]
        cands = a["candidates"]
        sent0 = str(cands[0]["sentence"])
        sent1 = str(cands[1]["sentence"])
        completion = str(a.get("Target1") or "")
        if not completion:
            # Fallback: sentence suffix after final period is not safe; use new-token record length only.
            completion = sent0.split(". ")[-1]
        old_len0 = tokenize_len(old_tok, sent0)
        old_len1 = tokenize_len(old_tok, sent1)
        new_len0 = tokenize_len(new_tok, sent0)
        new_len1 = tokenize_len(new_tok, sent1)
        try:
            old_tgt0 = target_token_len(old_tok, sent0, completion)
            old_tgt1 = target_token_len(old_tok, sent1, completion)
            new_tgt0 = target_token_len(new_tok, sent0, completion)
            new_tgt1 = target_token_len(new_tok, sent1, completion)
        except Exception:
            old_tgt0 = old_tgt1 = new_tgt0 = new_tgt1 = None
        row = {
            "uid": uid,
            "domain": a.get("domain"),
            "ConceptA": a.get("ConceptA"),
            "ConceptB": a.get("ConceptB"),
            "ContextType": a.get("ContextType"),
            "ContextDiff": a.get("ContextDiff"),
            "TargetDiff": a.get("TargetDiff"),
            "old_instability_pattern": a.get("selection_pattern"),
            "correct_43022": int(a["correct"]),
            "correct_43122": int(b["correct"]),
            "margin_43022": float(a["margin_c0_minus_c1"]),
            "margin_43122": float(b["margin_c0_minus_c1"]),
            "margin_seed_delta_430_minus_431": float(a["margin_c0_minus_c1"]) - float(b["margin_c0_minus_c1"]),
            "old_sent_len0": old_len0,
            "old_sent_len1": old_len1,
            "new_sent_len0": new_len0,
            "new_sent_len1": new_len1,
            "delta_sent_len0_new_minus_old": new_len0 - old_len0,
            "delta_sent_len1_new_minus_old": new_len1 - old_len1,
            "mean_sentence_len_delta_new_minus_old": ((new_len0 - old_len0) + (new_len1 - old_len1)) / 2.0,
            "delta_candidate_length_asymmetry_new_minus_old": (new_len0 - new_len1) - (old_len0 - old_len1),
            "old_target_len0": old_tgt0,
            "old_target_len1": old_tgt1,
            "new_target_len0": new_tgt0,
            "new_target_len1": new_tgt1,
            "target_len_delta_new_minus_old": None if old_tgt0 is None or new_tgt0 is None else ((new_tgt0 - old_tgt0) + (new_tgt1 - old_tgt1)) / 2.0,
        }
        out.append(row)
    return out


def group_summary(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        groups.setdefault(str(r.get(key)), []).append(r)
    out: dict[str, Any] = {}
    for g, rs in sorted(groups.items()):
        out[g] = {
            "n": len(rs),
            "accuracy_43022": sum(r["correct_43022"] for r in rs) / len(rs) * 100.0,
            "accuracy_43122": sum(r["correct_43122"] for r in rs) / len(rs) * 100.0,
            "mean_margin_43022": statistics.mean(r["margin_43022"] for r in rs),
            "mean_margin_43122": statistics.mean(r["margin_43122"] for r in rs),
            "mean_sent_len_delta": statistics.mean(r["mean_sentence_len_delta_new_minus_old"] for r in rs),
            "mean_target_len_delta": statistics.mean(float(r["target_len_delta_new_minus_old"] or 0.0) for r in rs),
            "mean_length_asymmetry_delta": statistics.mean(r["delta_candidate_length_asymmetry_new_minus_old"] for r in rs),
        }
    return out


def write_csv(rows: list[dict[str, Any]]) -> None:
    import csv
    if not rows:
        return
    with OUT_CSV.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def write_md(payload: dict[str, Any]) -> None:
    lines = ["# research — EWoK focus tokenization/margin link\n\n"]
    lines.append("CPU-only analysis on already-scored corrected 100M focused EWoK margins. It does not evaluate a model; it computes tokenizer-length features for the same items.\n\n")
    g = payload["global"]
    lines.append("## Global correlations\n\n")
    for k, v in g.items():
        lines.append(f"- `{k}`: `{v}`\n")
    lines.append("\n## Domain summary\n\n")
    lines.append("| domain | n | acc430 | acc431 | mean sent Δ | mean target Δ | asym Δ |\n")
    lines.append("|---|---:|---:|---:|---:|---:|---:|\n")
    for dom, r in payload["by_domain"].items():
        lines.append(f"| {dom} | {r['n']} | {r['accuracy_43022']:.2f} | {r['accuracy_43122']:.2f} | {r['mean_sent_len_delta']:.3f} | {r['mean_target_len_delta']:.3f} | {r['mean_length_asymmetry_delta']:.3f} |\n")
    lines.append("\n## Interpretation\n\n")
    for x in payload["interpretation"]:
        lines.append(f"- {x}\n")
    OUT_MD.write_text("".join(lines), encoding="utf-8")


def main() -> None:
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data = json.loads(MARGINS.read_text(encoding="utf-8"))
    rows = score_rows(data)
    xs_sent = [r["mean_sentence_len_delta_new_minus_old"] for r in rows]
    xs_tgt = [float(r["target_len_delta_new_minus_old"] or 0.0) for r in rows]
    xs_asym = [r["delta_candidate_length_asymmetry_new_minus_old"] for r in rows]
    m430 = [r["margin_43022"] for r in rows]
    m431 = [r["margin_43122"] for r in rows]
    dseed = [r["margin_seed_delta_430_minus_431"] for r in rows]
    global_summary = {
        "n_rows": len(rows),
        "sent_len_delta_summary": summarize_vals(xs_sent),
        "target_len_delta_summary": summarize_vals(xs_tgt),
        "candidate_length_asymmetry_delta_summary": summarize_vals(xs_asym),
        "pearson_sent_len_delta_vs_margin430": pearson(xs_sent, m430),
        "pearson_sent_len_delta_vs_margin431": pearson(xs_sent, m431),
        "pearson_target_len_delta_vs_margin430": pearson(xs_tgt, m430),
        "pearson_target_len_delta_vs_margin431": pearson(xs_tgt, m431),
        "pearson_asym_delta_vs_margin430": pearson(xs_asym, m430),
        "pearson_asym_delta_vs_margin431": pearson(xs_asym, m431),
        "pearson_sent_len_delta_vs_seed_margin_delta": pearson(xs_sent, dseed),
        "pearson_target_len_delta_vs_seed_margin_delta": pearson(xs_tgt, dseed),
        "pearson_asym_delta_vs_seed_margin_delta": pearson(xs_asym, dseed),
    }
    max_abs_corr = max(abs(v) for k, v in global_summary.items() if k.startswith("pearson") and isinstance(v, (int, float)))
    interp = [
        f"Maximum absolute item-level correlation between simple tokenizer-length features and corrected 100M margins/seed-delta is {max_abs_corr:.3f}.",
    ]
    if max_abs_corr < 0.15:
        interp.append("Simple token length/fragmentation features do not explain the focused EWoK relation-margin behavior; official EWoK movement should be read mainly as learned relation dynamics or vocabulary-target geometry, not crude item length.")
    else:
        interp.append("Some simple token length feature correlates with focused EWoK margins; inspect row-level examples before attributing EWoK movement solely to the compact-view data mechanism.")
    payload = {
        "status": "EWOK_FOCUS_TOKENIZATION_MARGIN_LINK",
        "created_utc": now_utc(),
        "elapsed_sec": round(time.time() - t0, 3),
        "source_margins": rel(MARGINS),
        "old_tokenizer": rel(OLD_TOKENIZER),
        "new_tokenizer": rel(NEW_TOKENIZER),
        "global": global_summary,
        "by_domain": group_summary(rows, "domain"),
        "by_old_instability_pattern": group_summary(rows, "old_instability_pattern"),
        "interpretation": interp,
        "row_csv": rel(OUT_CSV),
        "note": rel(OUT_MD),
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_csv(rows)
    write_md(payload)
    print(json.dumps({
        "status": payload["status"],
        "out_json": rel(OUT_JSON),
        "row_csv": rel(OUT_CSV),
        "note": rel(OUT_MD),
        "elapsed_sec": payload["elapsed_sec"],
        "n_rows": len(rows),
        "max_abs_corr": max_abs_corr,
        "interpretation": interp,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
