#!/usr/bin/env python3
"""research: late-resolving RoBERTa compact-vs-repeat transfer scaffold.

CPU/file-only audit and command manifest for the independent architecture-transfer
question: does the natural compact-view + reinvestment marginal reproduce under a
stock RoBERTa/BERT-style bidirectional MLM when evaluated across the late 100M
trajectory? This script does not launch training or evaluation.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import hashlib
import json
import os
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


def find_user_root() -> Path:
    return _PUBLIC_ROOT


USER_ROOT = find_user_root()
STUDY = USER_ROOT / "experiments/archive" / 'frontier_consolidation'
WORKSPACE = STUDY
DATA_ROOT = WORKSPACE / "data"
OUT_DIR = DATA_ROOT / "roberta_full100m_transfer_scaffold"
DENSITY_POOL = DATA_ROOT / "density_cleanqwen_overlay_medium_riskhard"
TOKENIZER_DIR = DATA_ROOT / "compliant_tokenizer"
TRAINER = WORKSPACE / "scripts" / "roberta_mlm_transfer_trainer.py"
EVAL_WRAPPER = WORKSPACE / "scripts" / "eval_custom_checkpoint.py"

COMPACT10 = DENSITY_POOL / "cleanqwen_fineweb_compact_view_reinvest_10M.jsonl"
REPEAT10 = DENSITY_POOL / "cleanqwen_fineweb_repeat_compact_reinvest_10M.jsonl"
COMPACT100 = DENSITY_POOL / "cleanqwen_fineweb_compact_view_reinvest_100M.jsonl"
REPEAT100 = DENSITY_POOL / "cleanqwen_fineweb_repeat_compact_reinvest_100M.jsonl"
SCAFFOLD_JSON = DATA_ROOT / "roberta_transfer_pair_scaffold" / "roberta_transfer_pair_scaffold.json"

EXPECTED = {
    "compact10_sha256": "215944978157394dbecf2039f2c1e1806bfcbb9701421920f78605eb58975a23",
    "repeat10_sha256": "0950c099e7227b8dc6940ea4bf749ce5604b28ff244ca3cf0bf9c102dab2a242",
    "tokenizer_sha256": "91b775514b9f4e2d1f37c28445ab98181007e16d14f763955168547e293ee8f9",
    "rows_10m": 64740,
    "words_10m": 10_000_000,
    "rows_100m": 647400,
    "words_100m": 100_000_000,
}

CHECKPOINTS = ["chck_10M", "chck_20M", "chck_30M", "chck_40M", "chck_50M", "chck_60M", "chck_70M", "chck_80M", "chck_90M", "chck_100M"]
PRIMARY_LATE_CHECKPOINTS = ["chck_60M", "chck_70M", "chck_80M", "chck_90M", "chck_100M"]


def now_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


@dataclass
class RowLite:
    text: str
    words: int
    example_id: int
    source: str


def read_row(raw: bytes | str) -> RowLite:
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    obj = json.loads(raw)
    text = str(obj["text"])
    words = int(obj.get("words", len(text.split())))
    actual = len(text.split())
    if words != actual:
        raise RuntimeError(f"word count mismatch: field={words} actual={actual} prefix={text[:100]!r}")
    return RowLite(text=text, words=words, example_id=int(obj.get("example_id", -1)), source=str(obj.get("source", "")))


def count_file(path: Path, with_example_hist: bool = False) -> dict[str, Any]:
    rows = 0
    words = 0
    source_words: dict[str, int] = defaultdict(int)
    example_counts: Counter[int] = Counter()
    first_rows: list[dict[str, Any]] = []
    last_rows: list[dict[str, Any]] = []
    with path.open("rb") as f:
        for raw in f:
            if not raw.strip():
                continue
            r = read_row(raw)
            rows += 1
            words += r.words
            source_words[r.source] += r.words
            if with_example_hist:
                example_counts[r.example_id] += 1
            meta = {"row": rows - 1, "example_id": r.example_id, "words": r.words, "source": r.source, "text_prefix": r.text[:120]}
            if len(first_rows) < 5:
                first_rows.append(meta)
            last_rows.append(meta)
            if len(last_rows) > 5:
                last_rows.pop(0)
    out: dict[str, Any] = {
        "path": str(path),
        "rows": rows,
        "words": words,
        "sha256": sha256_file(path),
        "source_words_top20": sorted(source_words.items(), key=lambda kv: (-kv[1], kv[0]))[:20],
        "first_rows": first_rows,
        "last_rows": last_rows,
    }
    if with_example_hist:
        hist = Counter(example_counts.values())
        out["unique_example_ids"] = len(example_counts)
        out["example_multiplicity_histogram"] = {str(k): v for k, v in sorted(hist.items())}
    return out


def compare_pair(a: Path, b: Path, sample_limit: int = 8, with_changed_hist: bool = True) -> dict[str, Any]:
    rows = 0
    words_a = 0
    words_b = 0
    id_mismatches = 0
    word_mismatches = 0
    text_equal = 0
    text_different = 0
    source_equal = 0
    source_different = 0
    source_pair_rows: Counter[str] = Counter()
    diff_source_pair_rows: Counter[str] = Counter()
    diff_example_counts: Counter[int] = Counter()
    diff_samples: list[dict[str, Any]] = []
    mismatch_samples: list[dict[str, Any]] = []
    with a.open("rb") as fa, b.open("rb") as fb:
        while True:
            la = fa.readline()
            lb = fb.readline()
            if not la and not lb:
                break
            if not la or not lb:
                raise RuntimeError(f"unequal number of raw lines at compared row {rows}")
            if not la.strip() and not lb.strip():
                continue
            ra = read_row(la)
            rb = read_row(lb)
            words_a += ra.words
            words_b += rb.words
            if ra.example_id != rb.example_id:
                id_mismatches += 1
                if len(mismatch_samples) < sample_limit:
                    mismatch_samples.append({"row": rows, "kind": "example_id", "compact": ra.example_id, "repeat": rb.example_id})
            if ra.words != rb.words:
                word_mismatches += 1
                if len(mismatch_samples) < sample_limit:
                    mismatch_samples.append({"row": rows, "kind": "words", "compact": ra.words, "repeat": rb.words})
            if ra.text == rb.text:
                text_equal += 1
            else:
                text_different += 1
                if with_changed_hist:
                    diff_example_counts[ra.example_id] += 1
                diff_source_pair_rows[f"{ra.source} || {rb.source}"] += 1
                if len(diff_samples) < sample_limit:
                    diff_samples.append({
                        "row": rows,
                        "example_id": ra.example_id,
                        "words": ra.words,
                        "compact_source": ra.source,
                        "repeat_source": rb.source,
                        "compact_prefix": ra.text[:180],
                        "repeat_prefix": rb.text[:180],
                    })
            if ra.source == rb.source:
                source_equal += 1
            else:
                source_different += 1
                source_pair_rows[f"{ra.source} || {rb.source}"] += 1
            rows += 1
    out: dict[str, Any] = {
        "rows_compared": rows,
        "compact_words": words_a,
        "repeat_words": words_b,
        "example_id_mismatches": id_mismatches,
        "word_mismatches": word_mismatches,
        "text_equal_rows": text_equal,
        "text_different_rows": text_different,
        "source_equal_rows": source_equal,
        "source_different_rows": source_different,
        "source_pair_rows_top20": source_pair_rows.most_common(20),
        "diff_source_pair_rows_top20": diff_source_pair_rows.most_common(20),
        "diff_samples": diff_samples,
        "mismatch_samples": mismatch_samples,
    }
    if with_changed_hist:
        hist = Counter(diff_example_counts.values())
        out["diff_unique_example_ids"] = len(diff_example_counts)
        out["diff_example_multiplicity_histogram"] = {str(k): v for k, v in sorted(hist.items())}
    return out


def load_step210_token_note() -> dict[str, Any] | None:
    if not SCAFFOLD_JSON.exists():
        return None
    data = json.loads(SCAFFOLD_JSON.read_text(encoding="utf-8"))
    ts = data.get("token_stats")
    if not ts:
        return None
    one = ts.get("compact_minus_repeat_one_pass", {})
    changed = ts.get("compact_minus_repeat_changed_block", {})
    est40 = ts.get("estimated_40m_total_diff_by_4x_one_pass", {})
    return {
        "source": str(SCAFFOLD_JSON),
        "one_10m_pass_compact_minus_repeat": one,
        "changed_block_10m_compact_minus_repeat": changed,
        "estimated_40m_total_diff_by_4x_one_pass": est40,
        "estimated_100m_total_diff_by_10x_one_pass": {k: (v * 10 if isinstance(v, (int, float)) else v) for k, v in one.items() if k in ["active_tokens", "candidate_tokens", "word_groups", "truncated_rows"]},
        "interpretation": "Compact has slightly more BPE active/candidate tokens for the same legal word count; the difference is part of the natural compact-vs-repeat treatment and is concentrated in compact-pair rows in the 10M audit.",
    }


def command_manifest() -> dict[str, Any]:
    compact_run = WORKSPACE / "training" / "runs" / "roberta_compact_reinvest_100M_seed43022"
    repeat_run = WORKSPACE / "training" / "runs" / "roberta_repeat_compact_reinvest_100M_seed43022"
    eval_out = DATA_ROOT / "roberta_full100m_selected_eval_if_authorized"
    common = [
        "python3", "-B", str(TRAINER),
        "--tokenizer_path", str(TOKENIZER_DIR),
        "--tokenizer_label", "compliant16k_reinvest10M",
        "--model_family", "roberta",
        "--hidden_size", "480",
        "--n_layer", "8",
        "--n_head", "8",
        "--ffn_mult", "4",
        "--seed", "43",
        "--extra_init_seed", "43022",
        "--train_rng_seed", "43023",
        "--batch_size", "256",
        "--seq_length", "256",
        "--max_seq_length", "256",
        "--learning_rate", "0.001",
        "--weight_decay", "0.01",
        "--warmup_fraction", "0.06",
        "--mask_prob", "0.15",
        "--max_word_exposure", "100000000",
        "--checkpoint_words", "10000000",
        "--lr_total_steps", "2529",
        "--log_every", "50",
    ]
    def train_cmd(path: Path, label: str, run_dir: Path, gpu: int) -> str:
        return "CUDA_VISIBLE_DEVICES=%d " % gpu + " ".join(common + ["--example_jsonl", str(path), "--example_jsonl_label", label, "--output_dir", str(run_dir)])
    def eval_cmd(run_dir: Path, arm: str, ck: str, gpu: int) -> str:
        target = f"roberta_{arm}_{ck}"
        out_base = eval_out / arm / ck
        return "CUDA_VISIBLE_DEVICES=%d " % gpu + " ".join([
            "python3", "-B", str(EVAL_WRAPPER),
            "--run-dir", str(run_dir),
            "--endpoint", ck,
            "--target", target,
            "--out-base", str(out_base),
            "--gpu", str(gpu),
            "--force",
        ])
    return {
        "training_commands_not_launched": {
            "compact_reinvest_roberta100M_gpu0": train_cmd(COMPACT100, "compact_reinvest100M_roberta_transfer", compact_run, 0),
            "repeat_compact_roberta100M_gpu1": train_cmd(REPEAT100, "repeat_compact_reinvest100M_roberta_transfer", repeat_run, 1),
        },
        "run_dirs_if_launched": {"compact": str(compact_run), "repeat": str(repeat_run)},
        "checkpoint_plan": CHECKPOINTS,
        "primary_late_checkpoints": PRIMARY_LATE_CHECKPOINTS,
        "selected_eval_commands_not_launched": {
            f"compact_{ck}": eval_cmd(compact_run, "compact_reinvest", ck, 0) for ck in CHECKPOINTS
        } | {
            f"repeat_{ck}": eval_cmd(repeat_run, "repeat_compact", ck, 1) for ck in CHECKPOINTS
        },
        "eval_out_if_launched": str(eval_out),
        "scientific_readout_after_launch": {
            "primary": ["mean compact-minus-repeat over chck_60M/chck_70M/chck_80M/chck_90M/chck_100M", "best late compact-minus-repeat with family decomposition"],
            "stable_metrics": ["cheap6_no_GlobalPIQA", "cheap5_no_GlobalPIQA_Reading", "EWoK_plus_Entity", "Supplement", "Entity", "COMPS"],
            "volatile_metrics_to_not_overweight": ["GlobalPIQA", "Reading"],
            "continue_if": "Compact is positive on late stable metrics and not carried solely by GlobalPIQA/Reading.",
            "localize_if": "Compact is neutral/negative on late stable metrics despite full exposure; this bounds the bundled compact-vs-repeat effect in this tested stock absolute-position RoBERTa coordinate and motivates mechanism decomposition without identifying the missing cause.",
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", type=Path, default=OUT_DIR)
    args = ap.parse_args()
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    required = {
        "compact10": COMPACT10,
        "repeat10": REPEAT10,
        "compact100": COMPACT100,
        "repeat100": REPEAT100,
        "tokenizer_json": TOKENIZER_DIR / "tokenizer.json",
        "trainer": TRAINER,
        "eval_wrapper": EVAL_WRAPPER,
    }
    missing = {k: str(v) for k, v in required.items() if not v.exists()}
    if missing:
        raise SystemExit(f"missing required files: {missing}")

    counts = {
        "compact10": count_file(COMPACT10, with_example_hist=True),
        "repeat10": count_file(REPEAT10, with_example_hist=True),
        "compact100": count_file(COMPACT100, with_example_hist=True),
        "repeat100": count_file(REPEAT100, with_example_hist=True),
    }
    pair10 = compare_pair(COMPACT10, REPEAT10)
    pair100 = compare_pair(COMPACT100, REPEAT100)
    tok_sha = sha256_file(TOKENIZER_DIR / "tokenizer.json")
    token_note = load_step210_token_note()
    commands = command_manifest()

    ok = True
    problems: list[str] = []
    if counts["compact10"]["sha256"] != EXPECTED["compact10_sha256"]:
        ok = False; problems.append("compact10 SHA mismatch")
    if counts["repeat10"]["sha256"] != EXPECTED["repeat10_sha256"]:
        ok = False; problems.append("repeat10 SHA mismatch")
    if tok_sha != EXPECTED["tokenizer_sha256"]:
        ok = False; problems.append("research tokenizer SHA mismatch")
    for name in ["compact10", "repeat10"]:
        if counts[name]["rows"] != EXPECTED["rows_10m"]:
            ok = False; problems.append(f"{name} rows {counts[name]['rows']} != {EXPECTED['rows_10m']}")
        if counts[name]["words"] != EXPECTED["words_10m"]:
            ok = False; problems.append(f"{name} words {counts[name]['words']} != {EXPECTED['words_10m']}")
    for name in ["compact100", "repeat100"]:
        if counts[name]["rows"] != EXPECTED["rows_100m"]:
            ok = False; problems.append(f"{name} rows {counts[name]['rows']} != {EXPECTED['rows_100m']}")
        if counts[name]["words"] != EXPECTED["words_100m"]:
            ok = False; problems.append(f"{name} words {counts[name]['words']} != {EXPECTED['words_100m']}")
    if pair10["example_id_mismatches"] or pair10["word_mismatches"]:
        ok = False; problems.append("10M compact/repeat row id/word mismatch")
    if pair100["example_id_mismatches"] or pair100["word_mismatches"]:
        ok = False; problems.append("100M compact/repeat row id/word mismatch")
    if pair100["text_different_rows"] != pair10["text_different_rows"] * 10:
        ok = False; problems.append(f"100M text-different rows {pair100['text_different_rows']} != 10x 10M {pair10['text_different_rows']}")

    payload = {
        "status": "ROBERTA_FULL100M_TRANSFER_SCAFFOLD",
        "ok": ok,
        "problems": problems,
        "created_utc": now_utc(),
        "scientific_purpose": "Prepare a late-resolving compact-vs-repeat RoBERTa MLM architecture-transfer test after compact-order evidence showed order is not the downstream cause.",
        "no_training_started": True,
        "no_selected_eval_started": True,
        "human_submission_boundary": "No leaderboard submission is made or authorized.",
        "counts": counts,
        "pair10": pair10,
        "pair100": pair100,
        "tokenizer": {"path": str(TOKENIZER_DIR), "tokenizer_json_sha256": tok_sha},
        "tokenization_note_from_step210": token_note,
        "future_command_manifest": commands,
        "cost_admission": {
            "why_expensive_work_would_be_admitted": "Two full 100M RoBERTa runs resolve an independent architecture-transfer question for the validated compact-vs-repeat data marginal; 20M/40M cannot close it because the original effect emerged late.",
            "minimum_reliable_method": "CPU/file preflight plus matched full 100M pair audit before launch; then full 100M training on both H100s with 10M checkpoints and selected official-compatible scoring on the late band.",
            "would_continue_route_if": "compact beats repeat on late stable metrics, not just GlobalPIQA/Reading",
            "would_change_route_if": "compact is neutral/negative on late stable metrics after full exposure, bounding the bundled compact-vs-repeat effect in this tested RoBERTa coordinate and requiring a different mechanism decomposition; this alone does not identify relative position as the cause",
        },
    }
    out_json = out_dir / "roberta_full100m_transfer_scaffold.json"
    out_md = out_dir / "roberta_full100m_transfer_scaffold.md"
    write_json(out_json, payload)

    lines = [
        "# research RoBERTa full-100M compact-vs-repeat transfer scaffold",
        "",
        f"Status: {'OK' if ok else 'NOT OK'}",
        "",
        "No training or selected evaluation was launched by this scaffold.",
        "",
        "## Matched data pair",
        f"- compact100: `{COMPACT100}` rows {counts['compact100']['rows']} words {counts['compact100']['words']} sha `{counts['compact100']['sha256']}`",
        f"- repeat100: `{REPEAT100}` rows {counts['repeat100']['rows']} words {counts['repeat100']['words']} sha `{counts['repeat100']['sha256']}`",
        f"- row id mismatches: {pair100['example_id_mismatches']}",
        f"- word-count mismatches: {pair100['word_mismatches']}",
        f"- text-different rows: {pair100['text_different_rows']} (10M diff rows {pair10['text_different_rows']} x10)",
        f"- text-equal rows: {pair100['text_equal_rows']}",
        "",
        "## Scientific reason",
        "The ordered-vs-scrambled 40M result removes coherent compact order as a downstream-positive cause, but it does not test the natural compact-vs-repeat data marginal. Because the original compact benefit emerged late, this transfer test must train to 100M and evaluate the late trajectory.",
        "",
        "## Future training commands (not launched)",
        "```",
        commands["training_commands_not_launched"]["compact_reinvest_roberta100M_gpu0"],
        commands["training_commands_not_launched"]["repeat_compact_roberta100M_gpu1"],
        "```",
        "",
        "Primary late readout: compact-minus-repeat over chck_60M/chck_70M/chck_80M/chck_90M/chck_100M on cheap6_no_GlobalPIQA, cheap5_no_GlobalPIQA_Reading, EWoK+Entity, Supplement, Entity, and COMPS.",
    ]
    if token_note:
        est = token_note.get("estimated_100m_total_diff_by_10x_one_pass", {})
        lines += [
            "",
            "## Tokenization note",
            f"Estimated compact-minus-repeat over 100M: active tokens {est.get('active_tokens')}, candidate tokens {est.get('candidate_tokens')}, word groups {est.get('word_groups')}, truncated rows {est.get('truncated_rows')}. This is part of the natural treatment.",
        ]
    if problems:
        lines += ["", "## Problems", *[f"- {p}" for p in problems]]
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "ok": ok, "out": str(out_json), "compact100_sha256": counts["compact100"]["sha256"], "repeat100_sha256": counts["repeat100"]["sha256"], "pair_text_diff_rows": pair100["text_different_rows"]}, indent=2), flush=True)
    if not ok:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
