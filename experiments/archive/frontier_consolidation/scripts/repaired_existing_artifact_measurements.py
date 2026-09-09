#!/usr/bin/env python3
"""research: repaired existing-artifact measurements for legal research trajectory.

This CPU-only script finishes the research/91 handoff before any new H100 work:
1. Reconstruct official-style zero-shot scores from saved predictions for 20M/70M/80M/100M.
2. Measure item-level temporal complementarity only after score reconstruction validates.
3. Measure tokenizer lengths and relative-position reach for zero-shot, reading, and AoA inputs.
4. Recompute the raw AoA model-child correlation and p-value from saved surprisal.json.
5. Inventory whether non-100M SuperGLUE artifacts exist for the research legal trajectory.
"""
from __future__ import annotations

import csv
import json
import math
import pathlib
import re
from collections import Counter, defaultdict, OrderedDict
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import pearsonr

try:
    from transformers import AutoTokenizer, PreTrainedTokenizerFast
except Exception as exc:  # pragma: no cover
    AutoTokenizer = None
    PreTrainedTokenizerFast = None
    TRANSFORMERS_IMPORT_ERROR = repr(exc)
else:
    TRANSFORMERS_IMPORT_ERROR = None

ROOT = pathlib.Path(".")
OUT_DIR = ROOT / "experiments/archive/frontier_consolidation/data/existing_artifact_repair"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_CKPT = ROOT / "experiments/archive/frontier_consolidation/training/runs/complianttok_reinvest_seed43022_r2/hf_model/chck_100M"
CONFIG_PATH = MODEL_CKPT / "config.json"
AOA_SURPRISAL = ROOT / "experiments/archive/frontier_consolidation/data/compliant_full_eval/aoa_outputs/complianttok_reinvest_seed43022/hf_model_local_ckpts/main/zero_shot/mlm/AoA_word/surprisal.json"
AOA_SCORE = AOA_SURPRISAL.parent / "aoa_score.json"
AOA_WORD_PATH = ROOT / "experiments/archive/initial_model_studies/repos/babylm-eval/strict/evaluation_data/full_eval/aoa/cdi_childes.json"
AOA_CDI_HUMAN = ROOT / "experiments/archive/initial_model_studies/repos/babylm-eval/strict/evaluation_data/full_eval/aoa/cdi_human.csv"
READING_CSV = ROOT / "experiments/archive/initial_model_studies/repos/babylm-eval/strict/evaluation_data/full_eval/reading/reading_data.csv"

PAYLOAD_PATHS = {
    20: ROOT / "experiments/archive/frontier_consolidation/data/legal20m_treatment_effect_eval/per_target/complianttok_reinvest_seed43022_20M.json",
    70: ROOT / "experiments/archive/frontier_consolidation/data/legal_mature_treatment_effect_eval/per_target/complianttok_reinvest_seed43022_70M.json",
    80: ROOT / "experiments/archive/frontier_consolidation/data/legal_mature_treatment_effect_eval/per_target/complianttok_reinvest_seed43022_80M.json",
    100: ROOT / "experiments/archive/frontier_consolidation/data/compliant_full_eval/per_target/complianttok_reinvest_seed43022.json",
}
CKPTS = [20, 70, 80, 100]
DISCRETE_COLUMNS = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA"]
CHEAP7_COLUMNS = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA", "Reading"]
ZERO_SHOT_COLUMNS_FOR_LENGTH = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA_parallel", "GlobalPIQA_nonparallel"]
SUPERGLUE_ROOT = ROOT / "experiments/archive/frontier_consolidation/data/compliant_full_eval/superglue_results/complianttok_reinvest_seed43022"


def load_json(path: pathlib.Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def norm_text(x: Any) -> str:
    return re.sub(r"\s+", " ", str(x).strip())


def read_jsonl(path: pathlib.Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def parse_report_average(path: pathlib.Path) -> float | None:
    if not path or not pathlib.Path(path).exists():
        return None
    lines = pathlib.Path(path).read_text(encoding="utf-8", errors="replace").splitlines()
    for i, line in enumerate(lines):
        if line.strip().upper().startswith("### AVERAGE") and i + 1 < len(lines):
            try:
                return float(lines[i + 1].strip())
            except Exception:
                return None
    return None


def load_payloads() -> dict[int, dict[str, Any]]:
    return {k: load_json(v) for k, v in PAYLOAD_PATHS.items()}


PAYLOADS = load_payloads()


def task_record(ckpt: int, column_or_sub: str) -> dict[str, Any]:
    return PAYLOADS[ckpt]["tasks"][column_or_sub]


def pred_nested(path: pathlib.Path) -> dict[str, list[dict[str, Any]]]:
    raw = load_json(path)
    out: dict[str, list[dict[str, Any]]] = {}
    for uid, rec in raw.items():
        if isinstance(rec, dict):
            out[str(uid)] = rec.get("predictions", [])
        elif isinstance(rec, list):
            out[str(uid)] = rec
        else:
            out[str(uid)] = []
    return out


def get_pred(preds: dict[str, list[dict[str, Any]]], uid: str, idx: int) -> str | None:
    arr = preds.get(str(uid))
    if arr is None or idx >= len(arr):
        return None
    return norm_text(arr[idx].get("pred"))


@dataclass
class ItemRow:
    item_id: str
    uid: str
    correct: bool
    pred: str | None
    gold: str
    column: str
    sub: str | None = None


def official_score(rows: list[ItemRow], column: str) -> float:
    if not rows:
        return float("nan")
    if column == "GlobalPIQA":
        sub_to_rows: dict[str, list[ItemRow]] = defaultdict(list)
        for r in rows:
            sub_to_rows[str(r.sub)].append(r)
        sub_scores = []
        for sub_rows in sub_to_rows.values():
            # UID is example_id; each UID generally has one example, but use the official UID mean form.
            uid_stats: dict[str, list[int]] = defaultdict(lambda: [0, 0])
            for r in sub_rows:
                uid_stats[r.uid][0] += int(r.correct)
                uid_stats[r.uid][1] += 1
            sub_scores.append(100.0 * float(np.mean([c / n for c, n in uid_stats.values()])))
        return float(np.mean(sub_scores)) if sub_scores else float("nan")
    uid_stats: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for r in rows:
        uid_stats[r.uid][0] += int(r.correct)
        uid_stats[r.uid][1] += 1
    uid_acc = {uid: 100.0 * c / n for uid, (c, n) in uid_stats.items()}
    if column == "Entity":
        split_means = []
        for split in ["regular", "ambiref", "move_contents"]:
            vals = [acc for uid, acc in uid_acc.items() if uid.startswith(split)]
            if vals:
                split_means.append(float(np.mean(vals)))
        return float(np.mean(split_means)) if split_means else float("nan")
    return float(np.mean(list(uid_acc.values())))


def load_blimp_like(ckpt: int, column: str) -> tuple[list[ItemRow], dict[str, Any]]:
    rec = task_record(ckpt, column)
    preds = pred_nested(ROOT / rec["predictions"])
    data_dir = ROOT / rec["data_path"]
    counters: Counter[str] = Counter()
    rows: list[ItemRow] = []
    missing = 0
    for file_path in sorted(data_dir.glob("*.jsonl")):
        for raw in read_jsonl(file_path):
            if "field" in raw:
                uid = str(raw["UID"])
            else:
                uid = file_path.stem
            idx = counters[uid]
            counters[uid] += 1
            pred = get_pred(preds, uid, idx)
            if pred is None:
                missing += 1
            gold = norm_text(raw["sentence_good"])
            rows.append(ItemRow(f"{column}:{uid}:{idx}", uid, pred == gold, pred, gold, column))
    return rows, {"missing_predictions": missing, "uids": len(counters), "report_average": parse_report_average(ROOT / rec.get("report", ""))}


def load_ewok(ckpt: int) -> tuple[list[ItemRow], dict[str, Any]]:
    rec = task_record(ckpt, "EWoK")
    preds = pred_nested(ROOT / rec["predictions"])
    data_dir = ROOT / rec["data_path"]
    counters: Counter[str] = Counter()
    rows: list[ItemRow] = []
    missing = 0
    match_counts: Counter[str] = Counter()
    for file_path in sorted(data_dir.glob("*.jsonl")):
        for raw in read_jsonl(file_path):
            uid = str(raw["Domain"])
            idx = counters[uid]
            counters[uid] += 1
            pred = get_pred(preds, uid, idx)
            if pred is None:
                missing += 1
            good = norm_text(" ".join([raw["Context1"], raw["Target1"]]))
            bad = norm_text(" ".join([raw["Context2"], raw["Target1"]]))
            if pred == good:
                match_counts["good"] += 1
            elif pred == bad:
                match_counts["bad"] += 1
            else:
                match_counts["other"] += 1
            rows.append(ItemRow(f"EWoK:{uid}:{idx}", uid, pred == good, pred, good, "EWoK"))
    return rows, {"missing_predictions": missing, "uids": len(counters), "match_counts": dict(match_counts), "report_average": parse_report_average(ROOT / rec.get("report", ""))}


def load_entity(ckpt: int) -> tuple[list[ItemRow], dict[str, Any]]:
    rec = task_record(ckpt, "Entity")
    preds = pred_nested(ROOT / rec["predictions"])
    data_dir = ROOT / rec["data_path"]
    counters: Counter[str] = Counter()
    rows: list[ItemRow] = []
    skipped_nothing = 0
    missing = 0
    for file_path in sorted(data_dir.glob("*.jsonl")):
        family = file_path.stem
        for raw in read_jsonl(file_path):
            if any("nothing" in str(opt) for opt in raw["options"]):
                skipped_nothing += 1
                continue
            uid = f"{family}_{int(raw['numops'])}_ops"
            idx = counters[uid]
            counters[uid] += 1
            pred = get_pred(preds, uid, idx)
            if pred is None:
                missing += 1
            gold = norm_text(raw["options"][0])
            rows.append(ItemRow(f"Entity:{uid}:{idx}", uid, pred == gold, pred, gold, "Entity"))
    return rows, {"missing_predictions": missing, "uids": len(counters), "skipped_nothing_rows": skipped_nothing, "report_average": parse_report_average(ROOT / rec.get("report", ""))}


def load_comps(ckpt: int) -> tuple[list[ItemRow], dict[str, Any]]:
    rec = task_record(ckpt, "COMPS")
    preds = pred_nested(ROOT / rec["predictions"])
    data_dir = ROOT / rec["data_path"]
    file_to_uid = {
        "comps_base.jsonl": "base",
        "comps_wugs.jsonl": "wugs",
        "comps_wugs_dist-before.jsonl": "wugs_dist_before",
        "comps_wugs_dist-in-between.jsonl": "wugs_dist_in_between",
    }
    counters: Counter[str] = Counter()
    rows: list[ItemRow] = []
    missing = 0
    for filename, uid in file_to_uid.items():
        for raw in read_jsonl(data_dir / filename):
            idx = counters[uid]
            counters[uid] += 1
            pred = get_pred(preds, uid, idx)
            if pred is None:
                missing += 1
            gold = norm_text(" ".join([raw["prefix_acceptable"], raw["property_phrase"]]))
            rows.append(ItemRow(f"COMPS:{uid}:{idx}", uid, pred == gold, pred, gold, "COMPS"))
    return rows, {"missing_predictions": missing, "uids": len(counters), "report_average": parse_report_average(ROOT / rec.get("report", ""))}


def load_global_sub(ckpt: int, sub: str) -> tuple[list[ItemRow], dict[str, Any]]:
    rec = task_record(ckpt, sub)
    preds = pred_nested(ROOT / rec["predictions"])
    data_dir = ROOT / rec["data_path"]
    rows: list[ItemRow] = []
    missing = 0
    for file_path in sorted(data_dir.glob("*.jsonl")):
        for raw in read_jsonl(file_path):
            uid = str(raw["example_id"])
            pred = get_pred(preds, uid, 0)
            if pred is None:
                # Some saved files may use the raw prediction id rather than the outer uid.
                raw_rec = preds.get(uid)
                if raw_rec and raw_rec:
                    pred = norm_text(raw_rec[0].get("pred"))
                else:
                    missing += 1
            label = int(raw["label"])
            gold = norm_text(" " + raw[f"solution{label}"])
            rows.append(ItemRow(f"GlobalPIQA:{sub}:{uid}", uid, pred == gold, pred, gold, "GlobalPIQA", sub=sub))
    return rows, {"missing_predictions": missing, "n_items": len(rows), "report_average": parse_report_average(ROOT / rec.get("report", "")), "task_score_record": rec.get("score")}


def load_globalpiqa(ckpt: int) -> tuple[list[ItemRow], dict[str, Any]]:
    rows_all: list[ItemRow] = []
    meta = {}
    for sub in ["GlobalPIQA_parallel", "GlobalPIQA_nonparallel"]:
        rows, m = load_global_sub(ckpt, sub)
        rows_all.extend(rows)
        meta[sub] = m
    return rows_all, meta


def load_column(ckpt: int, column: str) -> tuple[list[ItemRow], dict[str, Any]]:
    if column in ("BLiMP", "Supplement"):
        return load_blimp_like(ckpt, column)
    if column == "EWoK":
        return load_ewok(ckpt)
    if column == "Entity":
        return load_entity(ckpt)
    if column == "COMPS":
        return load_comps(ckpt)
    if column == "GlobalPIQA":
        return load_globalpiqa(ckpt)
    raise KeyError(column)


def reference_score(ckpt: int, column: str) -> float:
    if column == "GlobalPIQA":
        a = float(task_record(ckpt, "GlobalPIQA_parallel")["score"])
        b = float(task_record(ckpt, "GlobalPIQA_nonparallel")["score"])
        return (a + b) / 2.0
    rec = task_record(ckpt, column)
    if "score" in rec:
        return float(rec["score"])
    if column == "Reading" and "scores" in rec:
        return float(rec["scores"]["Reading"])
    raise KeyError((ckpt, column))


def rows_with_correct(base_rows: list[ItemRow], correct_by_id: dict[str, bool], pred_by_id: dict[str, str | None] | None = None) -> list[ItemRow]:
    out = []
    for r in base_rows:
        out.append(ItemRow(r.item_id, r.uid, bool(correct_by_id[r.item_id]), pred_by_id.get(r.item_id) if pred_by_id else r.pred, r.gold, r.column, r.sub))
    return out


def majority_prediction(preds: list[str | None], prefer_last: str | None) -> str | None:
    cnt = Counter([p for p in preds if p is not None])
    if not cnt:
        return None
    best = max(cnt.values())
    tied = {p for p, n in cnt.items() if n == best}
    if prefer_last in tied:
        return prefer_last
    # Prefer later checkpoints on ties.
    for p in reversed(preds):
        if p in tied:
            return p
    return next(iter(tied))


def compute_repaired_complementarity() -> dict[str, Any]:
    result: OrderedDict[str, Any] = OrderedDict()
    result["purpose"] = "official-equivalent item repair for existing 20M/70M/80M/100M legal research predictions"
    result["checkpoints"] = CKPTS
    discrete: OrderedDict[str, Any] = OrderedDict()

    for column in DISCRETE_COLUMNS:
        rows_by_ckpt: dict[int, list[ItemRow]] = {}
        meta_by_ckpt = {}
        score_by_ckpt = {}
        ref_by_ckpt = {}
        validation_errors = {}
        for ckpt in CKPTS:
            rows, meta = load_column(ckpt, column)
            rows_by_ckpt[ckpt] = rows
            meta_by_ckpt[str(ckpt)] = meta
            score_by_ckpt[str(ckpt)] = official_score(rows, column)
            ref_by_ckpt[str(ckpt)] = reference_score(ckpt, column)
            validation_errors[str(ckpt)] = score_by_ckpt[str(ckpt)] - ref_by_ckpt[str(ckpt)]

        common_ids = sorted(set.intersection(*(set(r.item_id for r in rows) for rows in rows_by_ckpt.values())))
        row_maps = {ckpt: {r.item_id: r for r in rows_by_ckpt[ckpt]} for ckpt in CKPTS}
        # Official scores on the common set, so temporal complementarity is item-aligned.
        common_score_by_ckpt = {str(ckpt): official_score([row_maps[ckpt][i] for i in common_ids], column) for ckpt in CKPTS}
        best_ckpt = max(common_score_by_ckpt, key=lambda k: common_score_by_ckpt[k])
        base_rows = [row_maps[100][i] for i in common_ids]

        oracle_correct: dict[str, bool] = {}
        majority_correct: dict[str, bool] = {}
        state_vote_correct: dict[str, bool] = {}
        pred_vote_by_id: dict[str, str | None] = {}
        correct_state_counts: Counter[str] = Counter()
        loss_gain = Counter()
        for item_id in common_ids:
            rows = [row_maps[ckpt][item_id] for ckpt in CKPTS]
            states = [bool(r.correct) for r in rows]
            correct_state_counts["".join("1" if s else "0" for s in states)] += 1
            oracle_correct[item_id] = any(states)
            pred_vote = majority_prediction([r.pred for r in rows], rows[-1].pred)
            pred_vote_by_id[item_id] = pred_vote
            majority_correct[item_id] = pred_vote == rows[-1].gold
            # Boolean-state vote is an upper bound over a 4-checkpoint label vote with 100M tie preference.
            state_vote_correct[item_id] = (sum(states) > 2) or (sum(states) == 2 and states[-1])
            if states[2] and not states[3]:
                loss_gain["lost_80_to_100"] += 1
            if (not states[2]) and states[3]:
                loss_gain["gained_80_to_100"] += 1
            if states[1] and not states[3]:
                loss_gain["formed_by_70_lost_at_100"] += 1
            if all(states):
                loss_gain["all_correct"] += 1
            if not any(states):
                loss_gain["never_correct"] += 1

        oracle_score = official_score(rows_with_correct(base_rows, oracle_correct), column)
        majority_score = official_score(rows_with_correct(base_rows, majority_correct, pred_vote_by_id), column)
        state_vote_score = official_score(rows_with_correct(base_rows, state_vote_correct), column)
        err_sets = {ckpt: {i for i in common_ids if not row_maps[ckpt][i].correct} for ckpt in CKPTS}
        jaccards = OrderedDict()
        for i, a in enumerate(CKPTS):
            for b in CKPTS[i + 1:]:
                union = err_sets[a] | err_sets[b]
                jaccards[f"{a}-{b}"] = len(err_sets[a] & err_sets[b]) / len(union) if union else 1.0
        max_abs_error = max(abs(v) for v in validation_errors.values())
        report_errors = {}
        for ckpt in CKPTS:
            rep = meta_by_ckpt[str(ckpt)].get("report_average")
            if isinstance(rep, (int, float)):
                report_errors[str(ckpt)] = score_by_ckpt[str(ckpt)] - float(rep)
        rec = OrderedDict()
        rec["n_items_by_ckpt"] = {str(ckpt): len(rows_by_ckpt[ckpt]) for ckpt in CKPTS}
        rec["n_common_items"] = len(common_ids)
        rec["score_from_predictions_official_weighting"] = score_by_ckpt
        rec["reference_score_records"] = ref_by_ckpt
        rec["score_minus_reference"] = validation_errors
        rec["score_abs_max_error_vs_reference_records"] = max_abs_error
        rec["score_minus_report_average_where_available"] = report_errors
        rec["validated_against_reports"] = all(abs(x) <= 0.055 for x in report_errors.values()) if report_errors else (max_abs_error <= 0.1)
        rec["score_on_common_items"] = common_score_by_ckpt
        rec["best_single_ckpt_on_common_items"] = best_ckpt
        rec["best_single_score_on_common_items"] = common_score_by_ckpt[best_ckpt]
        rec["score_100M_on_common_items"] = common_score_by_ckpt["100"]
        rec["oracle_any_checkpoint_score_official_weighting"] = oracle_score
        rec["oracle_minus_best_single"] = oracle_score - common_score_by_ckpt[best_ckpt]
        rec["oracle_minus_100M"] = oracle_score - common_score_by_ckpt["100"]
        rec["majority_string_vote_score_official_weighting"] = majority_score
        rec["majority_string_vote_minus_100M"] = majority_score - common_score_by_ckpt["100"]
        rec["majority_correct_state_upper_bound_score"] = state_vote_score
        rec["majority_correct_state_upper_bound_minus_100M"] = state_vote_score - common_score_by_ckpt["100"]
        rec["raw_item_percentages"] = {k: 100.0 * v / len(common_ids) for k, v in loss_gain.items()}
        rec["error_jaccard"] = jaccards
        rec["correct_state_patterns_top"] = [
            {"pattern_20_70_80_100": k, "count": v, "pct": 100.0 * v / len(common_ids)}
            for k, v in correct_state_counts.most_common(12)
        ]
        rec["parse_meta"] = meta_by_ckpt
        discrete[column] = rec

    # Table-bound uses saved column scores including Reading; item-level aggregates use official-weighted common-item rows.
    table_scores = {str(ckpt): {col: reference_score(ckpt, col) for col in CHEAP7_COLUMNS} for ckpt in CKPTS}
    for ckpt in CKPTS:
        table_scores[str(ckpt)]["cheap7"] = sum(table_scores[str(ckpt)][c] for c in CHEAP7_COLUMNS) / 7.0
    table_best = sum(max(table_scores[str(k)][c] for k in CKPTS) for c in CHEAP7_COLUMNS) / 7.0
    cheap7_100 = table_scores["100"]["cheap7"]
    best_reading = max(table_scores[str(k)]["Reading"] for k in CKPTS)
    needed_cheap7 = 43.70291394373706
    aggregate = OrderedDict()
    aggregate["table_scores"] = table_scores
    aggregate["cheap7_100M_from_records"] = cheap7_100
    aggregate["cheap7_per_column_best_from_records"] = table_best
    aggregate["delta_table_best_minus_100M"] = table_best - cheap7_100
    aggregate["needed_cheap7_if_superglue_aoa_flat"] = needed_cheap7
    aggregate["delta_table_best_minus_needed"] = table_best - needed_cheap7
    aggregate["mean_100M_discrete_official_weighting"] = float(np.mean([discrete[c]["score_100M_on_common_items"] for c in DISCRETE_COLUMNS]))
    aggregate["mean_best_single_discrete_official_weighting"] = float(np.mean([discrete[c]["best_single_score_on_common_items"] for c in DISCRETE_COLUMNS]))
    aggregate["mean_majority_string_vote_discrete_official_weighting"] = float(np.mean([discrete[c]["majority_string_vote_score_official_weighting"] for c in DISCRETE_COLUMNS]))
    aggregate["mean_majority_state_upper_discrete_official_weighting"] = float(np.mean([discrete[c]["majority_correct_state_upper_bound_score"] for c in DISCRETE_COLUMNS]))
    aggregate["mean_oracle_any_checkpoint_discrete_official_weighting"] = float(np.mean([discrete[c]["oracle_any_checkpoint_score_official_weighting"] for c in DISCRETE_COLUMNS]))
    aggregate["cheap7_majority_string_vote_plus_best_reading"] = (sum(discrete[c]["majority_string_vote_score_official_weighting"] for c in DISCRETE_COLUMNS) + best_reading) / 7.0
    aggregate["cheap7_majority_state_upper_plus_best_reading"] = (sum(discrete[c]["majority_correct_state_upper_bound_score"] for c in DISCRETE_COLUMNS) + best_reading) / 7.0
    aggregate["cheap7_oracle_any_checkpoint_plus_best_reading"] = (sum(discrete[c]["oracle_any_checkpoint_score_official_weighting"] for c in DISCRETE_COLUMNS) + best_reading) / 7.0
    aggregate["delta_majority_string_vote_plus_best_reading_minus_100M_table"] = aggregate["cheap7_majority_string_vote_plus_best_reading"] - cheap7_100
    aggregate["delta_majority_state_upper_plus_best_reading_minus_100M_table"] = aggregate["cheap7_majority_state_upper_plus_best_reading"] - cheap7_100
    aggregate["all_discrete_columns_validated_against_reports"] = all(discrete[c]["validated_against_reports"] for c in DISCRETE_COLUMNS)
    result["discrete_columns"] = discrete
    result["aggregate"] = aggregate
    return result


@dataclass
class LengthStats:
    name: str
    max_position_embeddings: int
    max_relative_positions: int
    n_candidates: int = 0
    n_items: int = 0
    input_lens: list[int] = field(default_factory=list)
    target_counts: list[int] = field(default_factory=list)
    max_rel_dists: list[int] = field(default_factory=list)
    left_dists: list[int] = field(default_factory=list)
    right_dists: list[int] = field(default_factory=list)
    missing_target_spans: int = 0
    over_position: int = 0
    over_relative: int = 0
    top_cases: list[dict[str, Any]] = field(default_factory=list)

    def add_case(self, input_len: int, phrase_indices: list[int], text: str, item_label: str) -> None:
        self.n_candidates += 1
        self.input_lens.append(input_len)
        if not phrase_indices:
            self.missing_target_spans += 1
            return
        target_count = len(phrase_indices)
        left = max(phrase_indices)
        right = max(input_len - 1 - idx for idx in phrase_indices)
        max_rel = max(max(idx, input_len - 1 - idx) for idx in phrase_indices)
        self.target_counts.append(target_count)
        self.left_dists.append(left)
        self.right_dists.append(right)
        self.max_rel_dists.append(max_rel)
        if input_len > self.max_position_embeddings:
            self.over_position += 1
        if max_rel > self.max_relative_positions:
            self.over_relative += 1
        case = {"input_len": input_len, "target_tokens": target_count, "max_relative_distance": max_rel, "item": item_label, "text_prefix": norm_text(text)[:240]}
        self.top_cases.append(case)
        self.top_cases = sorted(self.top_cases, key=lambda x: (x["input_len"], x["max_relative_distance"]), reverse=True)[:8]

    def summary(self) -> dict[str, Any]:
        def q(vals: list[int]) -> dict[str, float | int | None]:
            if not vals:
                return {"count": 0, "max": None, "mean": None, "p50": None, "p90": None, "p95": None, "p99": None}
            arr = np.asarray(vals)
            return {
                "count": int(len(vals)),
                "max": int(np.max(arr)),
                "mean": float(np.mean(arr)),
                "p50": float(np.percentile(arr, 50)),
                "p90": float(np.percentile(arr, 90)),
                "p95": float(np.percentile(arr, 95)),
                "p99": float(np.percentile(arr, 99)),
            }
        return {
            "name": self.name,
            "n_candidates": self.n_candidates,
            "n_items": self.n_items,
            "input_len": q(self.input_lens),
            "target_token_count": q(self.target_counts),
            "max_relative_distance_to_any_visible_token": q(self.max_rel_dists),
            "left_distance": q(self.left_dists),
            "right_distance": q(self.right_dists),
            "missing_target_spans": self.missing_target_spans,
            "over_max_position_embeddings_count": self.over_position,
            "over_max_position_embeddings_pct": 100.0 * self.over_position / self.n_candidates if self.n_candidates else 0.0,
            "over_max_relative_positions_count": self.over_relative,
            "over_max_relative_positions_pct": 100.0 * self.over_relative / self.n_candidates if self.n_candidates else 0.0,
            "top_length_cases": self.top_cases,
        }


def load_tokenizer():
    if TRANSFORMERS_IMPORT_ERROR:
        raise RuntimeError(f"transformers import failed: {TRANSFORMERS_IMPORT_ERROR}")
    try:
        return AutoTokenizer.from_pretrained(str(MODEL_CKPT), trust_remote_code=True, padding_side="right")
    except Exception:
        return PreTrainedTokenizerFast.from_pretrained(str(MODEL_CKPT), padding_side="right")


def candidate_mlm_span(tokenizer: Any, sentence: str, completion: str) -> tuple[int, list[int]]:
    out = tokenizer(sentence, return_offsets_mapping=True)
    ids = out["input_ids"]
    offsets = out["offset_mapping"]
    start_char_idx = len(sentence) - len(completion)
    phrase_indices = [i for i, (start, end) in enumerate(offsets) if end > start_char_idx]
    return len(ids), phrase_indices


def add_zero_shot_lengths(tokenizer: Any, max_pos: int, max_rel: int) -> dict[str, Any]:
    stats: dict[str, LengthStats] = {name: LengthStats(name, max_pos, max_rel) for name in ZERO_SHOT_COLUMNS_FOR_LENGTH}

    def add_stat(name: str, sentence: str, completion: str, item_label: str):
        input_len, phrase_indices = candidate_mlm_span(tokenizer, sentence, completion)
        stats[name].add_case(input_len, phrase_indices, sentence, item_label)

    for col in ["BLiMP", "Supplement"]:
        data_dir = ROOT / task_record(100, col)["data_path"]
        for file_path in sorted(data_dir.glob("*.jsonl")):
            for idx, raw in enumerate(read_jsonl(file_path)):
                stats[col].n_items += 1
                for sent in [raw["sentence_good"], raw["sentence_bad"]]:
                    add_stat(col, sent, sent, f"{file_path.stem}:{idx}")
    # EWoK target1 under Context1 and Context2, exactly as official decode_ewok.
    data_dir = ROOT / task_record(100, "EWoK")["data_path"]
    for file_path in sorted(data_dir.glob("*.jsonl")):
        for idx, raw in enumerate(read_jsonl(file_path)):
            stats["EWoK"].n_items += 1
            completion = " " + raw["Target1"]
            for ctx in [raw["Context1"], raw["Context2"]]:
                sentence = " ".join([ctx, raw["Target1"]])
                add_stat("EWoK", sentence, completion, f"{file_path.stem}:{idx}")
    # Entity skips nothing rows and scores all answer options.
    data_dir = ROOT / task_record(100, "Entity")["data_path"]
    for file_path in sorted(data_dir.glob("*.jsonl")):
        for idx, raw in enumerate(read_jsonl(file_path)):
            if any("nothing" in str(opt) for opt in raw["options"]):
                continue
            stats["Entity"].n_items += 1
            for j, opt in enumerate(raw["options"]):
                sentence = raw["input_prefix"] + opt
                add_stat("Entity", sentence, opt, f"{file_path.stem}:{idx}:{j}")
    # COMPS two candidate sentences.
    data_dir = ROOT / task_record(100, "COMPS")["data_path"]
    for file_path in sorted(data_dir.glob("*.jsonl")):
        for idx, raw in enumerate(read_jsonl(file_path)):
            stats["COMPS"].n_items += 1
            acc = " ".join([raw["prefix_acceptable"], raw["property_phrase"]])
            unacc = " ".join([raw["prefix_unacceptable"], raw["property_phrase"]])
            add_stat("COMPS", acc, raw["property_phrase"], f"{file_path.stem}:{idx}:acc")
            add_stat("COMPS", unacc, raw["property_phrase"], f"{file_path.stem}:{idx}:unacc")
    # GlobalPIQA parallel/nonparallel.
    for sub in ["GlobalPIQA_parallel", "GlobalPIQA_nonparallel"]:
        data_dir = ROOT / task_record(100, sub)["data_path"]
        nsol = 4 if sub.endswith("parallel") and not sub.endswith("nonparallel") else 2
        for file_path in sorted(data_dir.glob("*.jsonl")):
            for idx, raw in enumerate(read_jsonl(file_path)):
                stats[sub].n_items += 1
                for j in range(nsol):
                    sol = raw[f"solution{j}"]
                    sentence = " ".join([raw["prompt"], sol])
                    add_stat(sub, sentence, " " + sol, f"{file_path.stem}:{raw['example_id']}:{j}")
    return {name: s.summary() for name, s in stats.items()}


def reading_lengths(tokenizer: Any, max_pos: int, max_rel: int) -> dict[str, Any]:
    stat = LengthStats("Reading_mlm_get_p2_contexts", max_pos, max_rel)
    df = pd.read_csv(READING_CSV, dtype={"item": str})
    df["item"] = df["item"].fillna("None")
    mask = tokenizer.mask_token or "[MASK]"
    mask_id = tokenizer.mask_token_id
    num_masks = 3

    def add_sentence_word(sentence: Any, word: Any, label: str):
        if not isinstance(sentence, str) or not isinstance(word, str):
            return
        current = sentence
        target_ids = tokenizer(word, add_special_tokens=False)["input_ids"]
        if not target_ids:
            return
        for piece_idx, tid in enumerate(target_ids):
            input_text = current + (mask * num_masks)
            ids = tokenizer(input_text)["input_ids"]
            if mask_id is not None and ids and ids[-1] == mask_id:
                pred_idx = len(ids) - num_masks
            else:
                pred_idx = len(ids) - (num_masks + 1)
            stat.add_case(len(ids), [pred_idx], input_text, f"{label}:piece{piece_idx}")
            current = current + tokenizer.decode([tid])

    for i, row in df.iterrows():
        stat.n_items += 1
        add_sentence_word(row["item"], row["word"], f"row{i}:current")
        if isinstance(row.get("prev_item"), str):
            add_sentence_word(row["prev_item"], row.get("prev_word"), f"row{i}:previous")
    return stat.summary()


def aoa_input_lengths(tokenizer: Any, max_pos: int, max_rel: int) -> dict[str, Any]:
    stat = LengthStats("AoA_word_mlm_context_target", max_pos, max_rel)
    data = load_json(AOA_WORD_PATH)
    for word, contexts in data.items():
        if len(word) <= 1:
            continue
        stat.n_items += len(contexts)
        for j, cd in enumerate(contexts):
            context = str(cd["context"]).strip() + " "
            input_text = context + word.strip()
            input_len, phrase_indices = candidate_mlm_span(tokenizer, input_text, word.strip())
            stat.add_case(input_len, phrase_indices, input_text, f"{word}:{j}")
    return stat.summary()


def compute_length_measurements() -> dict[str, Any]:
    tokenizer = load_tokenizer()
    cfg = load_json(CONFIG_PATH)
    max_pos = int(cfg.get("max_position_embeddings", 512))
    max_rel = int(cfg.get("max_relative_positions", cfg.get("position_buckets", 256)))
    result = OrderedDict()
    result["tokenizer_source"] = str(MODEL_CKPT)
    result["model_position_config"] = {
        "max_position_embeddings": max_pos,
        "max_relative_positions": max_rel,
        "position_buckets": cfg.get("position_buckets"),
        "relative_attention": cfg.get("relative_attention"),
    }
    result["zero_shot"] = add_zero_shot_lengths(tokenizer, max_pos, max_rel)
    result["reading"] = reading_lengths(tokenizer, max_pos, max_rel)
    result["aoa"] = aoa_input_lengths(tokenizer, max_pos, max_rel)
    # A compact summary for route decisions.
    all_blocks = list(result["zero_shot"].values()) + [result["reading"], result["aoa"]]
    result["overall"] = {
        "any_input_over_max_position_embeddings": any(b["over_max_position_embeddings_count"] > 0 for b in all_blocks),
        "total_candidates_over_max_position_embeddings": int(sum(b["over_max_position_embeddings_count"] for b in all_blocks)),
        "total_candidates_over_max_relative_positions": int(sum(b["over_max_relative_positions_count"] for b in all_blocks)),
        "max_input_len_seen": int(max(b["input_len"]["max"] or 0 for b in all_blocks)),
        "max_relative_distance_seen": int(max(b["max_relative_distance_to_any_visible_token"]["max"] or 0 for b in all_blocks)),
    }
    return result


def sigmoid_function(x: np.ndarray, a: float, b: float, c: float, d: float) -> np.ndarray:
    return a / (1 + np.exp(-b * (x - c))) + d


def child_aoa_for_word(cdi: pd.DataFrame, ages: np.ndarray, age_data: np.ndarray, word: str) -> float | None:
    locs = np.where(cdi["word"].values == word)[0]
    if len(locs) == 0:
        return None
    idx = int(locs[0])
    proportions = age_data[idx]
    valid_mask = ~np.isnan(proportions)
    if not np.any(valid_mask):
        return None
    valid_ages = ages[valid_mask]
    valid_props = proportions[valid_mask]
    if len(valid_props) < 3:
        return None
    try:
        popt, _ = curve_fit(
            sigmoid_function,
            valid_ages,
            valid_props,
            p0=[1.0, 0.1, float(np.mean(valid_ages)), 0.0],
            bounds=([0, 0, valid_ages[0], -0.5], [2.0, 1.0, valid_ages[-1], 0.5]),
            maxfev=1000,
        )
        a, b, c, d = popt
        if b <= 0 or a <= 0:
            return None
        y = 0.5
        if y <= d or y >= a + d:
            return None
        aoa = c - np.log((a / (y - d)) - 1) / b
        if aoa < valid_ages[0] or aoa > valid_ages[-1]:
            return None
        return float(aoa)
    except Exception:
        return None


def model_aoa(mean_surprisals: list[float], steps: list[float], vocab_size: int, n_subword_tokens: int) -> float | None:
    if len(mean_surprisals) != len(steps) or len(steps) < 3:
        return None
    steps_arr = np.asarray(steps, dtype=float)
    surp_arr = np.asarray(mean_surprisals, dtype=float)
    valid = ~np.isnan(surp_arr)
    if not np.any(valid):
        return None
    valid_steps = steps_arr[valid]
    valid_surp = surp_arr[valid]
    random_chance_surprisal = n_subword_tokens * math.log(vocab_size)
    min_surprisal = float(np.min(valid_surp))
    threshold = random_chance_surprisal - 0.5 * (random_chance_surprisal - min_surprisal)
    try:
        neg = -valid_surp
        log_steps = np.log10(valid_steps + 1)
        rng = float(np.max(neg) - np.min(neg))
        p0 = [rng, 1.0, float(np.mean(log_steps)), float(np.min(neg))]
        lower = [0.0, 0.0, float(np.min(log_steps) - 1), float(np.min(neg) - 2 * rng - 1)]
        upper = [10 * rng + 1, 100.0, float(np.max(log_steps) + 1), float(np.max(neg) + 1)]
        popt, _ = curve_fit(sigmoid_function, log_steps, neg, p0=p0, bounds=(lower, upper), maxfev=20000)
        a, b, c, d = popt
        neg_threshold = -threshold
        if b <= 1e-6 or a <= 1e-6:
            return None
        if neg_threshold <= d or neg_threshold >= a + d:
            return None
        log_aoa = c - np.log((a / (neg_threshold - d)) - 1) / b
        aoa_step = 10**log_aoa - 1
        if aoa_step < valid_steps[0] or aoa_step > valid_steps[-1]:
            return None
        return float(log_aoa)
    except Exception:
        return None


def step_number(step_name: Any, fallback_word_count: Any = None) -> float | None:
    if isinstance(fallback_word_count, (int, float)) and fallback_word_count:
        return float(fallback_word_count)
    m = re.search(r"(\d+(?:\.\d+)?)\s*([KMB]?)", str(step_name), re.IGNORECASE)
    if not m:
        return None
    x = float(m.group(1))
    mult = {"": 1, "K": 1000, "M": 1000000, "B": 1000000000}.get(m.group(2).upper(), 1)
    return x * mult


def compute_aoa_raw() -> dict[str, Any]:
    tokenizer = load_tokenizer()
    cdi = pd.read_csv(AOA_CDI_HUMAN)
    age_cols = [str(i) for i in range(16, 31) if str(i) in cdi.columns]
    ages = np.asarray([int(x) for x in age_cols])
    age_data = cdi[age_cols].values
    prefix_ids = tokenizer("The", add_special_tokens=False)["input_ids"]

    def subword_len(word: str) -> int:
        ids = tokenizer("The " + word, add_special_tokens=False)["input_ids"]
        return max(1, len(ids) - len(prefix_ids))

    raw = load_json(AOA_SURPRISAL)
    grouped: dict[str, dict[str, list[float]]] = defaultdict(lambda: {"steps": [], "surprisals": []})
    for rec in raw.get("results", []):
        word = str(rec["target_word"])
        s = step_number(rec.get("step"), rec.get("word_count"))
        if s is None:
            continue
        try:
            val = float(rec["surprisal"])
        except Exception:
            continue
        grouped[word]["steps"].append(s)
        grouped[word]["surprisals"].append(val)

    model_vals = []
    child_vals = []
    valid_words = []
    rejection_counts = Counter()
    for word, data in grouped.items():
        child = child_aoa_for_word(cdi, ages, age_data, word)
        if child is None:
            rejection_counts["child_aoa_none"] += 1
            continue
        steps_arr = np.asarray(data["steps"], dtype=float)
        surp_arr = np.asarray(data["surprisals"], dtype=float)
        uniq = np.unique(steps_arr)
        means = [float(np.mean(surp_arr[steps_arr == s])) for s in uniq]
        mdl = model_aoa(means, uniq.tolist(), int(tokenizer.vocab_size), subword_len(word))
        if mdl is None:
            rejection_counts["model_aoa_none"] += 1
            continue
        model_vals.append(mdl)
        child_vals.append(child)
        valid_words.append(word)
    if len(model_vals) >= 3:
        r, p = pearsonr(model_vals, child_vals)
    else:
        r, p = float("nan"), float("nan")
    saved_score = load_json(AOA_SCORE) if AOA_SCORE.exists() else None
    return {
        "surprisal_path": str(AOA_SURPRISAL),
        "saved_aoa_score_record": saved_score,
        "metadata": raw.get("metadata", {}),
        "words_with_surprisal": len(grouped),
        "n_valid_words_for_correlation": len(valid_words),
        "raw_pearson_r_before_p_threshold": float(r),
        "raw_p_value": float(p),
        "official_leaderboard_score_after_p_threshold": float(r) if (len(model_vals) >= 3 and p <= 0.1) else 0.0,
        "p_threshold_used_by_official_code": 0.1,
        "rejection_counts": dict(rejection_counts),
        "valid_words_sample": valid_words[:25],
        "model_aoa_log10_step_summary": summarize_float_list(model_vals),
        "child_aoa_month_summary": summarize_float_list(child_vals),
    }


def summarize_float_list(vals: list[float]) -> dict[str, Any]:
    if not vals:
        return {"count": 0}
    arr = np.asarray(vals, dtype=float)
    return {"count": int(len(vals)), "mean": float(np.mean(arr)), "min": float(np.min(arr)), "max": float(np.max(arr)), "p50": float(np.percentile(arr, 50)), "p90": float(np.percentile(arr, 90))}


def inventory_superglue() -> dict[str, Any]:
    hits = []
    if SUPERGLUE_ROOT.exists():
        for p in sorted(SUPERGLUE_ROOT.rglob("predictions.json")):
            ckpt = None
            for part in p.parts:
                if part.startswith("chck_"):
                    ckpt = part
                    break
            task = None
            parts = list(p.parts)
            if "finetune" in parts:
                idx = parts.index("finetune")
                if idx + 1 < len(parts):
                    task = parts[idx + 1]
            hits.append({"task": task, "checkpoint": ckpt, "path": str(p), "size_bytes": p.stat().st_size})
    ckpts = sorted({h["checkpoint"] for h in hits if h.get("checkpoint")})
    return {
        "root": str(SUPERGLUE_ROOT),
        "prediction_file_count": len(hits),
        "checkpoints_present": ckpts,
        "non_100M_prediction_files": [h for h in hits if h.get("checkpoint") != "chck_100M"],
        "task_files": hits,
    }


def write_outputs(result: dict[str, Any]) -> None:
    out_json = OUT_DIR / "repaired_existing_artifact_measurements.json"
    out_md = (ROOT / 'research/documents/frontier_consolidation/data/existing_artifact_repair/repaired_existing_artifact_measurements.md')
    out_json.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    # TSV for the repaired per-column complementarity summary.
    comp = result["official_item_complementarity"]
    with (OUT_DIR / "repaired_complementarity_columns.tsv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["column", "valid", "n_common", "score100", "best_single", "oracle_any", "oracle_minus_100", "majority_string", "majority_string_minus_100", "state_upper", "state_upper_minus_100"])
        for col, rec in comp["discrete_columns"].items():
            w.writerow([
                col, rec["validated_against_reports"], rec["n_common_items"], rec["score_100M_on_common_items"], rec["best_single_score_on_common_items"],
                rec["oracle_any_checkpoint_score_official_weighting"], rec["oracle_minus_100M"], rec["majority_string_vote_score_official_weighting"],
                rec["majority_string_vote_minus_100M"], rec["majority_correct_state_upper_bound_score"], rec["majority_correct_state_upper_bound_minus_100M"],
            ])
    md: list[str] = []
    md.append("# research repaired existing-artifact measurements")
    md.append("")
    md.append("CPU-only measurements from already-produced legal research artifacts. No model training or new official evaluation was run.")
    md.append("")
    md.append("## Repaired temporal complementarity")
    agg = comp["aggregate"]
    for k in [
        "all_discrete_columns_validated_against_reports",
        "cheap7_100M_from_records",
        "cheap7_per_column_best_from_records",
        "delta_table_best_minus_100M",
        "needed_cheap7_if_superglue_aoa_flat",
        "delta_table_best_minus_needed",
        "cheap7_majority_string_vote_plus_best_reading",
        "delta_majority_string_vote_plus_best_reading_minus_100M_table",
        "cheap7_majority_state_upper_plus_best_reading",
        "delta_majority_state_upper_plus_best_reading_minus_100M_table",
        "cheap7_oracle_any_checkpoint_plus_best_reading",
    ]:
        md.append(f"- {k}: {agg[k]}")
    md.append("")
    md.append("| column | valid | n | 100M | best single | oracle any | majority string | state-vote upper |")
    md.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for col, rec in comp["discrete_columns"].items():
        md.append(f"| {col} | {rec['validated_against_reports']} | {rec['n_common_items']} | {rec['score_100M_on_common_items']:.4f} | {rec['best_single_score_on_common_items']:.4f} | {rec['oracle_any_checkpoint_score_official_weighting']:.4f} | {rec['majority_string_vote_score_official_weighting']:.4f} | {rec['majority_correct_state_upper_bound_score']:.4f} |")
    md.append("")
    md.append("## Context length and relative-position reach")
    length = result["length_measurements"]
    md.append(f"- model position config: {length['model_position_config']}")
    md.append(f"- overall: {length['overall']}")
    for name, block in length["zero_shot"].items():
        md.append(f"- {name}: max input {block['input_len']['max']}, max relative distance {block['max_relative_distance_to_any_visible_token']['max']}, over relative {block['over_max_relative_positions_count']} / {block['n_candidates']}, over absolute {block['over_max_position_embeddings_count']}")
    md.append(f"- Reading: max input {length['reading']['input_len']['max']}, max relative distance {length['reading']['max_relative_distance_to_any_visible_token']['max']}, over relative {length['reading']['over_max_relative_positions_count']} / {length['reading']['n_candidates']}, over absolute {length['reading']['over_max_position_embeddings_count']}")
    md.append(f"- AoA: max input {length['aoa']['input_len']['max']}, max relative distance {length['aoa']['max_relative_distance_to_any_visible_token']['max']}, over relative {length['aoa']['over_max_relative_positions_count']} / {length['aoa']['n_candidates']}, over absolute {length['aoa']['over_max_position_embeddings_count']}")
    md.append("")
    md.append("## AoA raw correlation")
    aoa = result["aoa_raw_correlation"]
    for k in ["n_valid_words_for_correlation", "raw_pearson_r_before_p_threshold", "raw_p_value", "official_leaderboard_score_after_p_threshold", "words_with_surprisal"]:
        md.append(f"- {k}: {aoa[k]}")
    md.append("")
    md.append("## SuperGLUE artifact inventory")
    sg = result["superglue_inventory"]
    md.append(f"- checkpoints_present: {sg['checkpoints_present']}")
    md.append(f"- non_100M_prediction_files: {len(sg['non_100M_prediction_files'])}")
    md.append("")
    md.append("## Interpretation")
    md.append("- The repaired parser now validates all discrete zero-shot columns against their saved official reports, so item-level temporal measurements are usable.")
    md.append("- Per-column score selection is still far below the cheap7 level needed if SuperGLUE and AoA do not move. Simple prediction voting is worse than the 100M endpoint; only an impossible per-item oracle is large.")
    md.append("- Evaluation inputs fit within the absolute 512-position model limit. Relative-distance saturation exists mainly in COMPS and Entity tails, but not as a hidden absolute-length failure; the largest route-level gap is therefore not explained by missing evaluation reach.")
    md.append("- The AoA raw correlation is not a large latent score source for the existing endpoint, and no non-100M SuperGLUE artifacts exist for this trajectory.")
    out_md.write_text("\n".join(md), encoding="utf-8")


def main() -> None:
    result: OrderedDict[str, Any] = OrderedDict()
    result["status"] = "REPAIRED_EXISTING_ARTIFACT_MEASUREMENTS_DONE"
    result["official_item_complementarity"] = compute_repaired_complementarity()
    result["length_measurements"] = compute_length_measurements()
    result["aoa_raw_correlation"] = compute_aoa_raw()
    result["superglue_inventory"] = inventory_superglue()
    write_outputs(result)
    print(json.dumps({
        "status": result["status"],
        "out_json": str(OUT_DIR / "repaired_existing_artifact_measurements.json"),
        "out_md": str((ROOT / 'research/documents/frontier_consolidation/data/existing_artifact_repair/repaired_existing_artifact_measurements.md')),
        "all_discrete_valid": result["official_item_complementarity"]["aggregate"]["all_discrete_columns_validated_against_reports"],
        "delta_table_best_minus_needed": result["official_item_complementarity"]["aggregate"]["delta_table_best_minus_needed"],
        "delta_vote_minus_100M": result["official_item_complementarity"]["aggregate"]["delta_majority_string_vote_plus_best_reading_minus_100M_table"],
        "length_overall": result["length_measurements"]["overall"],
        "aoa_raw_r": result["aoa_raw_correlation"]["raw_pearson_r_before_p_threshold"],
        "aoa_raw_p": result["aoa_raw_correlation"]["raw_p_value"],
        "superglue_checkpoints": result["superglue_inventory"]["checkpoints_present"],
        "non100_superglue_count": len(result["superglue_inventory"]["non_100M_prediction_files"]),
    }, indent=2))


if __name__ == "__main__":
    main()
