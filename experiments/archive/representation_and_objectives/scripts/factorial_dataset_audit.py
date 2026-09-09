#!/usr/bin/env python3
"""research CPU audit for factorial and information-budget substrates.

No model loading/training/evaluation.  The script checks exactly the artifacts that
matter for interpreting the running research learned probe and any future balanced
budget run:
- training pattern × relation × voice × static_slot balance;
- evaluation cell coverage;
- simple word-length deltas introduced by initial=same premise reconstruction;
- whether the current learned probe dataset class serializes metadata/IDs.
"""
from __future__ import annotations

import ast
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

ROOT = Path("experiments/archive/representation_and_objectives")
OWNERSHIP_FACTORIAL = ROOT / "data/factorial_initial_ownership"
INFORMATION_SUBSTRATE = ROOT / "data/information_budget_substrate"
PROBE_SCRIPT = ROOT / "training/scripts/factorial_probe.py"
OUT = ROOT / "data/factorial_dataset_audits"

TRAIN_CONDITIONS = [
    ("underdetermined", OWNERSHIP_FACTORIAL / "underdetermined"),
    ("disambiguated", OWNERSHIP_FACTORIAL / "disambiguated"),
    ("replace_k00_spread", INFORMATION_SUBSTRATE / "replace_k00_spread"),
    ("replace_k16_spread", INFORMATION_SUBSTRATE / "replace_k16_spread"),
    ("replace_k08_spread", INFORMATION_SUBSTRATE / "replace_k08_spread"),
    ("replace_k08_single_h0_dax", INFORMATION_SUBSTRATE / "replace_k08_single_h0_dax"),
]
ARMS = ["aligned_state_bridge", "inverted_state_bridge", "heldheld_only"]


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    with open(path) as f:
        for line in f:
            s = line.strip()
            if s:
                rows.append(json.loads(s))
    return rows


def words(s: str) -> int:
    return len(re.findall(r"\b\w+\b", s or ""))


def stat(xs: Sequence[int]) -> Dict[str, Any]:
    if not xs:
        return {"n": 0}
    return {"n": len(xs), "min": min(xs), "max": max(xs), "mean": sum(xs) / len(xs), "total": sum(xs)}


def cdict(counter: Counter) -> Dict[str, int]:
    return {str(k): int(v) for k, v in sorted(counter.items(), key=lambda kv: str(kv[0]))}


def row_key(r: Dict[str, Any], fields: Sequence[str]) -> str:
    vals = []
    for f in fields:
        v = r.get(f)
        if isinstance(v, list):
            v = "/".join(map(str, v))
        vals.append(str(v))
    return "|".join(vals)


def audit_rows(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    state = [r for r in rows if r.get("task") == "state_query"]
    comp = [r for r in rows if r.get("task") == "relation_comparison"]
    true_state = [r for r in state if r.get("label") is True]
    changed_true = [r for r in true_state if r.get("query_kind") == "changed"]
    report: Dict[str, Any] = {
        "n_rows": len(rows),
        "n_state_rows": len(state),
        "n_state_pairs": len({r.get("pair_id") for r in state}),
        "n_relation_comparison_rows": len(comp),
        "label_true_frac_all_labeled": (sum(bool(r.get("label")) for r in rows if "label" in r) / max(1, len([r for r in rows if "label" in r]))),
        "pattern_by_static_slot_rows": cdict(Counter(row_key(r, ["initial_pattern", "static_slot"]) for r in state)),
        "pattern_by_relation_voice_static_rows": cdict(Counter(row_key(r, ["relation", "voice", "static_slot", "initial_pattern"]) for r in state)),
        "true_changed_by_relation_voice_static_pattern": cdict(Counter(row_key(r, ["relation", "voice", "static_slot", "initial_pattern"]) for r in changed_true)),
        "true_changed_owner_slot_by_pattern": cdict(Counter(row_key(r, ["initial_pattern", "correct_slot"]) for r in changed_true)),
        "premise_word_count_state": stat([words(r.get("premise", "")) for r in state]),
        "hypothesis_word_count_state": stat([words(r.get("hypothesis", "")) for r in state]),
        "state_row_word_count_by_pattern": {},
    }
    for pat in sorted({r.get("initial_pattern") for r in state}, key=str):
        subset = [r for r in state if r.get("initial_pattern") == pat]
        report["state_row_word_count_by_pattern"][str(pat)] = {
            "premise": stat([words(r.get("premise", "")) for r in subset]),
            "hypothesis": stat([words(r.get("hypothesis", "")) for r in subset]),
        }
    return report


def audit_train_conditions() -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for name, cond in TRAIN_CONDITIONS:
        if not cond.exists():
            continue
        out[name] = {"path": str(cond), "arms": {}}
        for arm in ARMS:
            rows = load_jsonl(cond / "arms" / arm / "train_supervised.jsonl")
            out[name]["arms"][arm] = audit_rows(rows)
    return out


def audit_eval_conditions() -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for name, cond in [("underdetermined", OWNERSHIP_FACTORIAL / "underdetermined"), ("disambiguated", OWNERSHIP_FACTORIAL / "disambiguated"), ("replace_k16_spread", INFORMATION_SUBSTRATE / "replace_k16_spread")]:
        if not cond.exists():
            continue
        out[name] = {"path": str(cond), "suites": {}}
        for ef in sorted((cond / "eval").glob("*.jsonl")):
            rows = load_jsonl(ef)
            rep = audit_rows(rows)
            state = [r for r in rows if r.get("task") == "state_query"]
            true_changed = [r for r in state if r.get("query_kind") == "changed" and r.get("label") is True]
            rep["eval_true_changed_cells_pattern_static_relation_voice"] = cdict(Counter(row_key(r, ["initial_pattern", "static_slot", "relation", "voice"]) for r in true_changed))
            rep["eval_all_state_cells_pattern_static_relation_voice_query_label"] = cdict(Counter(row_key(r, ["initial_pattern", "static_slot", "relation", "voice", "query_kind", "label"]) for r in state))
            out[name]["suites"][ef.stem] = rep
    return out


def inspect_probe_serialization() -> Dict[str, Any]:
    text = PROBE_SCRIPT.read_text(encoding="utf-8") if PROBE_SCRIPT.exists() else ""
    parsed = ast.parse(text) if text else None
    # Conservative static inspection: report literal subscript/get keys inside NLIDataset.__init__.
    keys: List[str] = []
    class V(ast.NodeVisitor):
        in_target = False
        def visit_FunctionDef(self, node: ast.FunctionDef) -> Any:
            old = self.in_target
            if node.name == "__init__":
                # The script only has one Dataset __init__ that matters; still inspect literals generally.
                self.in_target = True
                self.generic_visit(node)
                self.in_target = old
            else:
                self.generic_visit(node)
        def visit_Subscript(self, node: ast.Subscript) -> Any:
            if self.in_target:
                sl = node.slice
                if isinstance(sl, ast.Constant) and isinstance(sl.value, str):
                    keys.append(sl.value)
            self.generic_visit(node)
    if parsed is not None:
        V().visit(parsed)
    uses = sorted(set(keys))
    return {
        "probe_script": str(PROBE_SCRIPT),
        "dataset_literal_row_keys_seen_in_init": uses,
        "serializes_only_premise_hypothesis_or_text": all(k in {"premise", "hypothesis", "text", "label"} for k in uses),
        "important_note": "research NLIDataset constructs examples from r['premise'], r['hypothesis'] or r['text'] and r['label']; row_id/pair_id/initial_pattern metadata are not serialized by this script. The script also ignores train_unsup_text.jsonl in train_one, so unsupervised exposure equality is inert in the current classification probe.",
    }


def make_summary(report: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append("# research factorial dataset audit")
    lines.append("")
    lines.append("## Probe serialization")
    ser = report["probe_serialization"]
    lines.append(f"- Script: `{ser['probe_script']}`")
    lines.append(f"- Literal row keys used inside dataset init: `{ser['dataset_literal_row_keys_seen_in_init']}`")
    lines.append(f"- Metadata/IDs serialized by this script: `{not ser['serializes_only_premise_hypothesis_or_text']}`")
    lines.append(f"- Note: {ser['important_note']}")
    lines.append("")
    lines.append("## Training balance highlights")
    lines.append("")
    lines.append("| condition | arm | state pairs | pattern×static rows | true changed relation×voice×static×pattern cells | premise words by pattern |")
    lines.append("|---|---|---:|---|---|---|")
    for cname, crep in report["train_conditions"].items():
        for arm in ["aligned_state_bridge", "inverted_state_bridge"]:
            a = crep["arms"].get(arm, {})
            pw = a.get("state_row_word_count_by_pattern", {})
            pw_compact = {k: {"min": v.get("premise", {}).get("min"), "max": v.get("premise", {}).get("max")} for k, v in pw.items()}
            lines.append(f"| {cname} | {arm} | {a.get('n_state_pairs')} | `{a.get('pattern_by_static_slot_rows')}` | `{a.get('true_changed_by_relation_voice_static_pattern')}` | `{pw_compact}` |")
    lines.append("")
    lines.append("## Evaluation balance highlights")
    lines.append("")
    lines.append("The state evaluation suites are copied across research/research conditions.  The most important question is whether all four pattern×static-slot cells exist so static-slot-switch heuristics can be rejected by off-diagonal rows.")
    lines.append("")
    lines.append("| condition | suite | state pairs | pattern×static rows | true changed cells |")
    lines.append("|---|---|---:|---|---|")
    for cname, crep in report["eval_conditions"].items():
        for suite, srep in crep["suites"].items():
            if srep.get("n_state_rows", 0) == 0:
                continue
            lines.append(f"| {cname} | {suite} | {srep.get('n_state_pairs')} | `{srep.get('pattern_by_static_slot_rows')}` | `{srep.get('eval_true_changed_cells_pattern_static_relation_voice')}` |")
    lines.append("")
    lines.append("## Scientific use")
    lines.append("")
    lines.append("- The research disambiguated learned probe is not a clean forced-event-role test because `initial_pattern` is diagonal with `static_slot` in train.  Its aggregate state metrics should be treated as preliminary; mixed held-seen orientation remains the more important coordinate readout.")
    lines.append("- `replace_k16_spread` removes the diagonal by giving equal opposite/same counts in every relation×voice×static-slot cell for aligned and inverted bridge arms.  A future learned confirmation should use this or a counterallocated equivalent and save per-row predictions.")
    lines.append("- The current research probe script will not allow post-hoc row-level rescoring because it does not save predictions or checkpoints.  Future runners must preserve per-row outputs and cell metrics.")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    report: Dict[str, Any] = {
        "status": "FACTORIAL_DATASET_AUDIT_COMPLETE",
        "train_conditions": audit_train_conditions(),
        "eval_conditions": audit_eval_conditions(),
        "probe_serialization": inspect_probe_serialization(),
        "no_model_loading_training_evaluation_upload_or_leaderboard": True,
    }
    (OUT / "factorial_dataset_audit.json").write_text(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    ((OUT.parents[4] / 'research/documents/representation_and_objectives/data/factorial_dataset_audits/factorial_dataset_audit_summary.md')).write_text(make_summary(report), encoding="utf-8")
    compact = {
        "status": report["status"],
        "summary": str((OUT.parents[4] / 'research/documents/representation_and_objectives/data/factorial_dataset_audits/factorial_dataset_audit_summary.md')),
        "serializes_metadata": not report["probe_serialization"]["serializes_only_premise_hypothesis_or_text"],
        "note": "research train has initial_pattern×static_slot diagonal; research k16 spread balances it. Future learned runner needs per-row predictions.",
    }
    print(json.dumps(compact, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
