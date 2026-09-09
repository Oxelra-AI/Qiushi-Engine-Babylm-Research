#!/usr/bin/env python3
"""research: EWoK evaluation-interface tokenizer audit.

The corrected Strict-Small tokenizer changes not only pretraining representation but
also the official MLM zero-shot scoring interface: every token overlapping the
completion span is masked and scored.  This script compares the inherited
Strict-100M tokenizer and the new 10M-trained tokenizer on official EWoK target
phrases and candidate sentences, with emphasis on the relation rows that showed
old-seed margin polarization.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import collections
import csv
import json
from pathlib import Path
import statistics
from typing import Any

from transformers import AutoTokenizer


def find_user_root() -> Path:
    return _PUBLIC_ROOT


ROOT = find_user_root()
A01 = ROOT / "experiments/archive/representation_and_objectives"
WORKSPACE = A01
EWOK_DIR = WORKSPACE / "data/pristine_official_coordinate/babylm-eval/strict/evaluation_data/full_eval/ewok_filtered"
FOCUS_CSV = WORKSPACE / "data/official_ewok_margin_subset/ewok_margin_focus_selection.csv"
OLD_TOK = ROOT / "experiments/archive/initial_model_studies/training/runs/fullcycle_official_wwm_debertav2_8x480_seed43_100M_b256/hf_model"
NEW_TOK = WORKSPACE / "data/strictsmall_tokenizer_retrain/strictsmall_compact_reinvest_16k_tokenizer"
OUT_DIR = WORKSPACE / "data/ewok_tokenizer_eval_interface_audit"


def load_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for p in sorted(EWOK_DIR.glob("*.jsonl")):
        dom = p.stem
        for idx, line in enumerate(p.read_text(encoding="utf-8", errors="replace").splitlines()):
            if not line.strip():
                continue
            raw = json.loads(line)
            c1, c2, t1 = raw["Context1"], raw["Context2"], raw["Target1"]
            rows.append({
                "uid": f"{dom}_{idx}",
                "domain": dom,
                "idx": idx,
                "ConceptA": raw.get("ConceptA"),
                "ConceptB": raw.get("ConceptB"),
                "ContextType": raw.get("ContextType"),
                "ContextDiff": raw.get("ContextDiff"),
                "TargetDiff": raw.get("TargetDiff"),
                "Context1": c1,
                "Context2": c2,
                "Target1": t1,
                "Target2": raw.get("Target2"),
                "sentences": [" ".join([c1, t1]).strip(), " ".join([c2, t1]).strip()],
                "completion": " " + t1,
            })
    return rows


def load_focus_keys() -> dict[tuple[str, int], dict[str, Any]]:
    if not FOCUS_CSV.exists():
        return {}
    out = {}
    with FOCUS_CSV.open("r", encoding="utf-8", errors="replace", newline="") as f:
        for r in csv.DictReader(f):
            out[(r["domain"], int(r["idx"]))] = r
    return out


def phrase_token_info(tok, sentence: str, completion: str) -> dict[str, Any]:
    enc = tok(text=sentence, return_offsets_mapping=True)
    ids = list(enc["input_ids"])
    offsets = list(enc["offset_mapping"])
    start_char_idx = len(sentence) - len(completion)
    phrase_indices = []
    target_ids = []
    for i, (start, end) in enumerate(offsets):
        if end > start_char_idx:
            phrase_indices.append(i)
            target_ids.append(ids[i])
    return {
        "sentence_tokens": len(ids),
        "completion_tokens": len(target_ids),
        "target_ids": target_ids,
        "target_tokens": tok.convert_ids_to_tokens(target_ids),
        "phrase_indices": phrase_indices,
    }


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        return {"n": 0}
    diffs = [r["new_completion_tokens_mean"] - r["old_completion_tokens_mean"] for r in records]
    absdiffs = [abs(x) for x in diffs]
    sentdiffs = [r["new_sentence_tokens_mean"] - r["old_sentence_tokens_mean"] for r in records]
    return {
        "n": len(records),
        "mean_completion_token_delta_new_minus_old": statistics.mean(diffs),
        "median_completion_token_delta_new_minus_old": statistics.median(diffs),
        "changed_completion_token_count_frac": sum(1 for x in absdiffs if x != 0) / len(records),
        "increased_completion_token_count_frac": sum(1 for x in diffs if x > 0) / len(records),
        "decreased_completion_token_count_frac": sum(1 for x in diffs if x < 0) / len(records),
        "mean_sentence_token_delta_new_minus_old": statistics.mean(sentdiffs),
        "median_sentence_token_delta_new_minus_old": statistics.median(sentdiffs),
        "old_mean_completion_tokens": statistics.mean([r["old_completion_tokens_mean"] for r in records]),
        "new_mean_completion_tokens": statistics.mean([r["new_completion_tokens_mean"] for r in records]),
        "old_mean_sentence_tokens": statistics.mean([r["old_sentence_tokens_mean"] for r in records]),
        "new_mean_sentence_tokens": statistics.mean([r["new_sentence_tokens_mean"] for r in records]),
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tok_old = AutoTokenizer.from_pretrained(str(OLD_TOK), local_files_only=True)
    tok_new = AutoTokenizer.from_pretrained(str(NEW_TOK), local_files_only=True)
    rows = load_rows()
    focus = load_focus_keys()
    recs: list[dict[str, Any]] = []
    concept_tokenizations: dict[str, dict[str, Any]] = {}
    concepts = sorted({str(r.get("ConceptA")) for r in rows if r.get("ConceptA")} | {str(r.get("ConceptB")) for r in rows if r.get("ConceptB")})
    for concept in concepts:
        o = tok_old(concept, add_special_tokens=False)["input_ids"]
        n = tok_new(concept, add_special_tokens=False)["input_ids"]
        concept_tokenizations[concept] = {
            "old_len": len(o), "new_len": len(n),
            "old_tokens": tok_old.convert_ids_to_tokens(o),
            "new_tokens": tok_new.convert_ids_to_tokens(n),
        }
    for r in rows:
        old_infos = [phrase_token_info(tok_old, s, r["completion"]) for s in r["sentences"]]
        new_infos = [phrase_token_info(tok_new, s, r["completion"]) for s in r["sentences"]]
        key = (r["domain"], int(r["idx"]))
        frec = focus.get(key, {})
        rec = {k: r.get(k) for k in ["uid", "domain", "idx", "ConceptA", "ConceptB", "ContextType", "ContextDiff", "TargetDiff", "Target1"]}
        rec.update({
            "is_focus": bool(frec),
            "focus_pattern": frec.get("pattern"),
            "focus_DiD_item": int(frec["DiD_item"]) if frec.get("DiD_item") not in (None, "") else None,
            "focus_selection_reason": frec.get("selection_reason"),
            "old_completion_tokens_c0": old_infos[0]["completion_tokens"],
            "old_completion_tokens_c1": old_infos[1]["completion_tokens"],
            "new_completion_tokens_c0": new_infos[0]["completion_tokens"],
            "new_completion_tokens_c1": new_infos[1]["completion_tokens"],
            "old_completion_tokens_mean": (old_infos[0]["completion_tokens"] + old_infos[1]["completion_tokens"]) / 2,
            "new_completion_tokens_mean": (new_infos[0]["completion_tokens"] + new_infos[1]["completion_tokens"]) / 2,
            "old_sentence_tokens_mean": (old_infos[0]["sentence_tokens"] + old_infos[1]["sentence_tokens"]) / 2,
            "new_sentence_tokens_mean": (new_infos[0]["sentence_tokens"] + new_infos[1]["sentence_tokens"]) / 2,
            "old_target_tokens_c0": old_infos[0]["target_tokens"],
            "new_target_tokens_c0": new_infos[0]["target_tokens"],
            "old_target_tokens_c1": old_infos[1]["target_tokens"],
            "new_target_tokens_c1": new_infos[1]["target_tokens"],
        })
        rec["completion_token_delta"] = rec["new_completion_tokens_mean"] - rec["old_completion_tokens_mean"]
        rec["sentence_token_delta"] = rec["new_sentence_tokens_mean"] - rec["old_sentence_tokens_mean"]
        recs.append(rec)

    by_domain = {d: summarize([r for r in recs if r["domain"] == d]) for d in sorted({r["domain"] for r in recs})}
    focus_recs = [r for r in recs if r["is_focus"]]
    focus_neg = [r for r in recs if r.get("focus_DiD_item") is not None and r["focus_DiD_item"] < 0]
    focus_pat0110 = [r for r in recs if r.get("focus_pattern") == "0110"]
    by_focus_reason = {reason: summarize([r for r in focus_recs if r.get("focus_selection_reason") == reason]) for reason in sorted({r.get("focus_selection_reason") for r in focus_recs})}

    payload = {
        "status": "EWOK_TOKENIZER_EVAL_INTERFACE_AUDIT",
        "old_tokenizer": str(OLD_TOK),
        "new_tokenizer": str(NEW_TOK),
        "ewok_rows": len(recs),
        "focus_rows": len(focus_recs),
        "overall": summarize(recs),
        "focus_subset": summarize(focus_recs),
        "focus_negative_rows": summarize(focus_neg),
        "focus_pattern_0110_rows": summarize(focus_pat0110),
        "by_domain": by_domain,
        "by_focus_reason": by_focus_reason,
        "concept_tokenizations": concept_tokenizations,
        "largest_completion_token_increases": sorted(recs, key=lambda r: (-r["completion_token_delta"], r["domain"], r["idx"]))[:60],
        "largest_completion_token_decreases": sorted(recs, key=lambda r: (r["completion_token_delta"], r["domain"], r["idx"]))[:60],
        "interpretation_limits": [
            "This measures tokenizer/evaluation geometry only, not model competence.",
            "For MLM EWoK, more target tokens means more masked predictions are summed; this can change margin scale and noise but does not determine correctness without trained weights.",
            "Use it to interpret corrected-tokenizer EWoK margins after official evaluation.",
        ],
    }
    out_json = OUT_DIR / "ewok_tokenizer_eval_interface_audit.json"
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    out_csv = OUT_DIR / "ewok_tokenizer_eval_interface_rows.csv"
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        fields = [
            "uid", "domain", "idx", "ConceptA", "ConceptB", "ContextType", "ContextDiff", "TargetDiff", "Target1",
            "is_focus", "focus_pattern", "focus_DiD_item", "focus_selection_reason",
            "old_completion_tokens_mean", "new_completion_tokens_mean", "completion_token_delta",
            "old_sentence_tokens_mean", "new_sentence_tokens_mean", "sentence_token_delta",
            "old_target_tokens_c0", "new_target_tokens_c0",
        ]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in recs:
            row = dict(r)
            row["old_target_tokens_c0"] = " ".join(r["old_target_tokens_c0"])
            row["new_target_tokens_c0"] = " ".join(r["new_target_tokens_c0"])
            w.writerow({k: row.get(k) for k in fields})

    note = WORKSPACE / "notes/ewok_tokenizer_eval_interface_audit.md"
    lines = [
        "# research — EWoK tokenizer evaluation-interface audit",
        "",
        f"JSON: `{out_json}`",
        f"Rows CSV: `{out_csv}`",
        "",
        "This compares the inherited Strict-100M tokenizer and the new 10M-trained Strict-Small tokenizer on the official EWoK MLM scoring interface. It measures target/completion token counts and sentence token counts; it is not a model score.",
        "",
        "## Overall EWoK geometry",
    ]
    ov = payload["overall"]
    lines.append(f"- rows: {ov['n']}")
    lines.append(f"- mean completion tokens: old {ov['old_mean_completion_tokens']:.3f}, new {ov['new_mean_completion_tokens']:.3f}, delta {ov['mean_completion_token_delta_new_minus_old']:.3f}")
    lines.append(f"- completion token count changed in {ov['changed_completion_token_count_frac']:.3f} of rows; increased {ov['increased_completion_token_count_frac']:.3f}, decreased {ov['decreased_completion_token_count_frac']:.3f}")
    lines.append(f"- mean candidate sentence tokens: old {ov['old_mean_sentence_tokens']:.3f}, new {ov['new_mean_sentence_tokens']:.3f}, delta {ov['mean_sentence_token_delta_new_minus_old']:.3f}")
    lines.extend(["", "## Focused unstable subset geometry"])
    for name, summ in [("focus_all", payload["focus_subset"]), ("focus_negative", payload["focus_negative_rows"]), ("focus_0110", payload["focus_pattern_0110_rows"] )]:
        lines.append(f"- {name}: n={summ['n']}, completion delta mean={summ['mean_completion_token_delta_new_minus_old']:.3f}, changed_frac={summ['changed_completion_token_count_frac']:.3f}, sentence delta mean={summ['mean_sentence_token_delta_new_minus_old']:.3f}")
    lines.extend(["", "## Worst relation domains"])
    for d in ["material-dynamics", "physical-dynamics", "spatial-relations", "physical-interactions", "social-relations", "agent-properties", "material-properties", "social-interactions"]:
        s = by_domain[d]
        lines.append(f"- {d}: n={s['n']}, completion delta mean={s['mean_completion_token_delta_new_minus_old']:.3f}, changed_frac={s['changed_completion_token_count_frac']:.3f}, sentence delta mean={s['mean_sentence_token_delta_new_minus_old']:.3f}")
    lines.extend(["", "## Selected concept tokenizations"])
    for c in ["subordinate", "boss", "parent", "child", "teacher", "student", "above", "below", "kick", "drop", "touch", "break", "stir", "accelerate", "slow down"]:
        if c in concept_tokenizations:
            t = concept_tokenizations[c]
            lines.append(f"- {c}: old({t['old_len']})={t['old_tokens']} ; new({t['new_len']})={t['new_tokens']}")
    lines.extend(["", "## Reading"])
    lines.append("The corrected tokenizer should be interpreted as a representation change at both training and evaluation time. If later official EWoK movement differs from the inherited-tokenizer endpoint, this audit tells whether target-token fragmentation is large enough to be a plausible contributor. The actual endpoint decision still requires completed retraining and official collation.")
    note.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": payload["status"],
        "out_json": str(out_json),
        "out_csv": str(out_csv),
        "note": str(note),
        "overall": payload["overall"],
        "focus_negative": payload["focus_negative_rows"],
        "selected_concepts": {c: concept_tokenizations.get(c) for c in ["subordinate", "boss", "parent", "child", "teacher", "student", "above", "below", "kick", "drop", "touch", "break", "stir", "accelerate"] if c in concept_tokenizations},
    }, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
