#!/usr/bin/env python3
"""research: prepare the matched compact-vs-repeat data scaffold for a possible RoBERTa/BERT MLM transfer test.

This script is deliberately CPU/file-only.  It does not launch training or
selected evaluation.  It materializes and audits the 40M repeat-control stream
needed to pair with the existing research compact_ordered_40M stream if the
pending compact ordered-vs-scrambled result justifies a model-coordinate
transfer experiment.

Scientific question enabled later:
  Does the compact-view/reinvestment data distribution transfer to a different
  bidirectional MLM encoder (RoBERTa/BERT-style absolute-position attention), or
  was the DeBERTa-v2 result tied to its particular architecture coordinate?

The future experiment should train under ordinary WWM, not source-absent target
priority.  This scaffold only records data identities, row invariants, tokenizer
active-token/group differences, and exact future commands.
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
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from transformers import AutoTokenizer, PreTrainedTokenizerFast


def find_user_root() -> Path:
    return _PUBLIC_ROOT


USER_ROOT = find_user_root()
STUDY = USER_ROOT / "experiments/archive" / 'frontier_consolidation'
WORKSPACE = STUDY
DATA_ROOT = WORKSPACE / "data"
SCAFFOLD_DIR = DATA_ROOT / "roberta_transfer_pair_scaffold"
DENSITY_POOL = DATA_ROOT / "density_cleanqwen_overlay_medium_riskhard"
ORDER_FACTORIAL_POOL = DATA_ROOT / "compact_order_factorial_pool_scaffold"
TOKENIZER_DIR = DATA_ROOT / "compliant_tokenizer"
TRAINER = WORKSPACE / "scripts" / "roberta_mlm_transfer_trainer.py"
EVAL_WRAPPER = WORKSPACE / "scripts" / "eval_custom_checkpoint.py"

COMPACT40 = ORDER_FACTORIAL_POOL / "compact_ordered_40M.jsonl"
COMPACT10 = ORDER_FACTORIAL_POOL / "compact_ordered_10M.jsonl"
REPEAT10 = DENSITY_POOL / "cleanqwen_fineweb_repeat_compact_reinvest_10M.jsonl"
REPEAT100 = DENSITY_POOL / "cleanqwen_fineweb_repeat_compact_reinvest_100M.jsonl"
REPEAT40 = SCAFFOLD_DIR / "repeat_compact_reinvest_40M.jsonl"

EXPECTED = {
    "compact40_sha256": "48313173a4cd93e768f490c8a5eaaa850fc828fabc9cfb21ebad73651bef85dc",
    "compact10_sha256": "215944978157394dbecf2039f2c1e1806bfcbb9701421920f78605eb58975a23",
    "repeat10_sha256": "0950c099e7227b8dc6940ea4bf749ce5604b28ff244ca3cf0bf9c102dab2a242",
    "tokenizer_sha256": "91b775514b9f4e2d1f37c28445ab98181007e16d14f763955168547e293ee8f9",
    "rows_per_10m": 64740,
    "words_per_10m": 10_000_000,
    "rows_40m": 258960,
    "words_40m": 40_000_000,
    "changed_rows_per_pass": 3006,
    "compact_pair_rows_per_pass": 3005,
    "neutral_topup_rows_per_pass": 1,
    "filler_rows_per_pass": 61734,
}


def now_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_lines(lines: Iterable[bytes]) -> str:
    h = hashlib.sha256()
    for line in lines:
        h.update(line)
    return h.hexdigest()


def read_json_line(line: bytes | str) -> dict[str, Any]:
    if isinstance(line, bytes):
        line = line.decode("utf-8")
    return json.loads(line)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


@dataclass
class RowLite:
    text: str
    words: int
    example_id: int
    source: str


def row_lite(line: bytes | str) -> RowLite:
    obj = read_json_line(line)
    text = str(obj["text"])
    words = int(obj.get("words", len(text.split())))
    actual = len(text.split())
    if words != actual:
        raise RuntimeError(f"word mismatch: field={words} actual={actual} text_prefix={text[:80]!r}")
    return RowLite(text=text, words=words, example_id=int(obj.get("example_id", -1)), source=str(obj.get("source", "")))


def count_file(path: Path) -> dict[str, Any]:
    rows = 0
    words = 0
    source_words: dict[str, int] = {}
    first_rows: list[dict[str, Any]] = []
    last_rows: list[dict[str, Any]] = []
    with path.open("rb") as f:
        for raw in f:
            if not raw.strip():
                continue
            r = row_lite(raw)
            rows += 1
            words += r.words
            source_words[r.source] = source_words.get(r.source, 0) + r.words
            meta = {"row": rows - 1, "example_id": r.example_id, "words": r.words, "source": r.source, "text_prefix": r.text[:120]}
            if len(first_rows) < 5:
                first_rows.append(meta)
            last_rows.append(meta)
            if len(last_rows) > 5:
                last_rows.pop(0)
    return {"path": str(path), "rows": rows, "words": words, "source_words": source_words, "first_rows": first_rows, "last_rows": last_rows, "sha256": sha256_file(path)}


def materialize_repeat40(overwrite: bool = False) -> dict[str, Any]:
    SCAFFOLD_DIR.mkdir(parents=True, exist_ok=True)
    if REPEAT40.exists() and not overwrite:
        return {"materialized": False, "reason": "exists", "path": str(REPEAT40), "sha256": sha256_file(REPEAT40), "bytes": REPEAT40.stat().st_size}
    tmp = REPEAT40.with_suffix(".jsonl.tmp")
    with tmp.open("wb") as out:
        for _ in range(4):
            with REPEAT10.open("rb") as f:
                for line in f:
                    out.write(line)
    os.replace(tmp, REPEAT40)
    return {"materialized": True, "path": str(REPEAT40), "sha256": sha256_file(REPEAT40), "bytes": REPEAT40.stat().st_size}


def prefix_sha(path: Path, n_lines: int) -> tuple[str, int]:
    h = hashlib.sha256()
    rows = 0
    with path.open("rb") as f:
        for line in f:
            if not line.strip():
                continue
            h.update(line)
            rows += 1
            if rows >= n_lines:
                break
    return h.hexdigest(), rows


def compare_pair(compact_path: Path, repeat_path: Path) -> dict[str, Any]:
    rows = 0
    words = 0
    example_id_mismatches = 0
    word_mismatches = 0
    filler_text_mismatches = 0
    filler_source_mismatches = 0
    changed_text_equal = 0
    changed_text_different = 0
    changed_source_pairs: dict[str, int] = {}
    first_changed_samples: list[dict[str, Any]] = []
    first_filler_samples: list[dict[str, Any]] = []
    rows_per_pass = EXPECTED["rows_per_10m"]
    changed_per_pass = EXPECTED["changed_rows_per_pass"]

    with compact_path.open("rb") as fc, repeat_path.open("rb") as fr:
        for lc, lr in zip(fc, fr):
            if not lc.strip() and not lr.strip():
                continue
            c = row_lite(lc)
            r = row_lite(lr)
            pass_idx = rows // rows_per_pass
            pos = rows % rows_per_pass
            in_changed = pos < changed_per_pass
            words += c.words
            if c.example_id != r.example_id:
                example_id_mismatches += 1
            if c.words != r.words:
                word_mismatches += 1
            if in_changed:
                if c.text == r.text:
                    changed_text_equal += 1
                else:
                    changed_text_different += 1
                key = f"{c.source} || {r.source}"
                changed_source_pairs[key] = changed_source_pairs.get(key, 0) + 1
                if len(first_changed_samples) < 6:
                    first_changed_samples.append({
                        "row": rows,
                        "pass": pass_idx,
                        "pos": pos,
                        "example_id": c.example_id,
                        "words": c.words,
                        "compact_source": c.source,
                        "repeat_source": r.source,
                        "compact_prefix": c.text[:180],
                        "repeat_prefix": r.text[:180],
                    })
            else:
                if c.text != r.text:
                    filler_text_mismatches += 1
                    if len(first_filler_samples) < 6:
                        first_filler_samples.append({"row": rows, "pass": pass_idx, "pos": pos, "example_id_compact": c.example_id, "example_id_repeat": r.example_id, "compact_prefix": c.text[:120], "repeat_prefix": r.text[:120]})
                if c.source != r.source:
                    filler_source_mismatches += 1
            rows += 1

    return {
        "rows_compared": rows,
        "words_seen_from_compact": words,
        "rows_per_pass": rows_per_pass,
        "changed_rows_per_pass": changed_per_pass,
        "passes": rows / rows_per_pass if rows_per_pass else None,
        "example_id_mismatches": example_id_mismatches,
        "word_mismatches": word_mismatches,
        "changed_text_equal": changed_text_equal,
        "changed_text_different": changed_text_different,
        "filler_text_mismatches": filler_text_mismatches,
        "filler_source_mismatches": filler_source_mismatches,
        "changed_source_pairs": changed_source_pairs,
        "first_changed_samples": first_changed_samples,
        "first_filler_mismatch_samples": first_filler_samples,
    }


def make_portable_tokenizer(tokenizer_path: Path) -> PreTrainedTokenizerFast:
    base = AutoTokenizer.from_pretrained(str(tokenizer_path), use_fast=True)
    backend = getattr(base, "_tokenizer", None) or getattr(base, "backend_tokenizer", None)
    if backend is None:
        raise RuntimeError(f"Tokenizer at {tokenizer_path} has no fast backend")
    tok = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        bos_token=base.bos_token if base.bos_token else "<s>",
        eos_token=base.eos_token if base.eos_token else "</s>",
        unk_token=base.unk_token if base.unk_token else "<unk>",
        pad_token=base.pad_token if base.pad_token else "<pad>",
        mask_token=getattr(base, "mask_token", None) or "<mask>",
        model_max_length=getattr(base, "model_max_length", 1024) or 1024,
    )
    return tok


def is_word_start(tok: str) -> bool:
    return tok.startswith("Ġ") or tok.startswith("▁")


def token_stats_one_pass(compact10_path: Path, repeat10_path: Path, tokenizer_path: Path, max_length: int, sample_rows: int | None) -> dict[str, Any]:
    tok = make_portable_tokenizer(tokenizer_path)
    special = set(tok.all_special_ids)
    changed_per_pass = EXPECTED["changed_rows_per_pass"]
    out: dict[str, Any] = {}
    for label, path in [("compact", compact10_path), ("repeat", repeat10_path)]:
        rows = 0
        words = 0
        trunc = 0
        active = 0
        candidate = 0
        groups = 0
        changed = {"rows": 0, "words": 0, "active_tokens": 0, "candidate_tokens": 0, "word_groups": 0, "truncated_rows": 0}
        filler = {"rows": 0, "words": 0, "active_tokens": 0, "candidate_tokens": 0, "word_groups": 0, "truncated_rows": 0}
        with path.open("rb") as f:
            for raw in f:
                if not raw.strip():
                    continue
                r = row_lite(raw)
                enc_full = tok(r.text, add_special_tokens=False, truncation=False)["input_ids"]
                enc = enc_full[:max_length]
                was_trunc = len(enc_full) > max_length
                cand = [x for x in enc if int(x) not in special]
                grp = 0
                prev_started = False
                for i, tid in enumerate(enc):
                    tid = int(tid)
                    if tid in special:
                        continue
                    s = tok.convert_ids_to_tokens(tid)
                    if grp == 0 or is_word_start(str(s)) or i == 0:
                        grp += 1
                bucket = changed if rows < changed_per_pass else filler
                bucket["rows"] += 1
                bucket["words"] += r.words
                bucket["active_tokens"] += len(enc)
                bucket["candidate_tokens"] += len(cand)
                bucket["word_groups"] += grp
                bucket["truncated_rows"] += int(was_trunc)
                rows += 1
                words += r.words
                active += len(enc)
                candidate += len(cand)
                groups += grp
                trunc += int(was_trunc)
                if sample_rows is not None and rows >= sample_rows:
                    break
        def finish_bucket(b: dict[str, int]) -> dict[str, Any]:
            rr = max(1, int(b["rows"]))
            ww = max(1, int(b["words"]))
            return {**b, "active_tokens_per_row": b["active_tokens"] / rr, "word_groups_per_row": b["word_groups"] / rr, "active_tokens_per_word": b["active_tokens"] / ww}
        out[label] = {
            "rows": rows,
            "words": words,
            "active_tokens": active,
            "candidate_tokens": candidate,
            "word_groups": groups,
            "truncated_rows": trunc,
            "active_tokens_per_row": active / max(1, rows),
            "word_groups_per_row": groups / max(1, rows),
            "active_tokens_per_word": active / max(1, words),
            "changed": finish_bucket(changed),
            "filler": finish_bucket(filler),
        }
    diff = {}
    for key in ["active_tokens", "candidate_tokens", "word_groups", "truncated_rows", "active_tokens_per_row", "word_groups_per_row", "active_tokens_per_word"]:
        diff[key] = out["compact"][key] - out["repeat"][key]
    diff_changed = {}
    diff_filler = {}
    for key in ["active_tokens", "candidate_tokens", "word_groups", "truncated_rows", "active_tokens_per_row", "word_groups_per_row", "active_tokens_per_word"]:
        diff_changed[key] = out["compact"]["changed"][key] - out["repeat"]["changed"][key]
        diff_filler[key] = out["compact"]["filler"][key] - out["repeat"]["filler"][key]
    return {
        "tokenizer_path": str(tokenizer_path),
        "tokenizer_json_sha256": sha256_file(tokenizer_path / "tokenizer.json"),
        "vocab_size": len(tok),
        "max_length": max_length,
        "sample_rows": sample_rows,
        "one_pass_stats": out,
        "compact_minus_repeat_one_pass": diff,
        "compact_minus_repeat_changed_block": diff_changed,
        "compact_minus_repeat_filler": diff_filler,
        "estimated_40m_total_diff_by_4x_one_pass": {k: (v * 4 if isinstance(v, (int, float)) else v) for k, v in diff.items() if k in ["active_tokens", "candidate_tokens", "word_groups", "truncated_rows"]},
    }


def command_manifest(compact_path: Path, repeat_path: Path, out_base: Path) -> dict[str, Any]:
    compact_run = WORKSPACE / "training" / "runs" / "roberta_compact_reinvest_40M_seed43022"
    repeat_run = WORKSPACE / "training" / "runs" / "roberta_repeat_compact_reinvest_40M_seed43022"
    eval_out = DATA_ROOT / "roberta_transfer_selected_eval_if_authorized"
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
        "--max_word_exposure", "40000000",
        "--checkpoint_words", "20000000",
        "--lr_total_steps", "2529",
        "--log_every", "50",
    ]
    def cmd_for(path: Path, label: str, run: Path) -> list[str]:
        return common + ["--example_jsonl", str(path), "--example_jsonl_label", label, "--output_dir", str(run)]
    def eval_cmd(run: Path, arm: str, ck: str, gpu: int) -> str:
        target = f"roberta_{arm}_{ck}"
        out_base = eval_out / arm / ck
        return " ".join([
            "python3", "-B", str(EVAL_WRAPPER),
            "--run-dir", str(run),
            "--endpoint", ck,
            "--target", target,
            "--out-base", str(out_base),
            "--gpu", str(gpu),
        ])
    return {
        "training_commands_not_launched": {
            "compact_reinvest_roberta40M_gpu0": "CUDA_VISIBLE_DEVICES=0 " + " ".join(cmd_for(compact_path, "compact_reinvest_ordered40M_roberta_transfer", compact_run)),
            "repeat_compact_roberta40M_gpu1": "CUDA_VISIBLE_DEVICES=1 " + " ".join(cmd_for(repeat_path, "repeat_compact_reinvest40M_roberta_transfer", repeat_run)),
        },
        "selected_eval_commands_not_launched": {
            "compact_chck20_gpu0": eval_cmd(compact_run, "compact_reinvest", "chck_20M", 0),
            "repeat_chck20_gpu1": eval_cmd(repeat_run, "repeat_compact", "chck_20M", 1),
            "compact_chck40_gpu0": eval_cmd(compact_run, "compact_reinvest", "chck_40M", 0),
            "repeat_chck40_gpu1": eval_cmd(repeat_run, "repeat_compact", "chck_40M", 1),
        },
        "run_dirs_if_launched": {"compact": str(compact_run), "repeat": str(repeat_run)},
        "eval_out_if_launched": str(eval_out),
        "evaluation_note": "Run selected scoring after both arms finish, ideally in two waves chck20 compact/repeat then chck40 compact/repeat. The research wrapper evaluates one endpoint per command.",
    }


def main() -> None:
    global SCAFFOLD_DIR, REPEAT40
    ap = argparse.ArgumentParser()
    ap.add_argument("--output_dir", type=Path, default=SCAFFOLD_DIR)
    ap.add_argument("--materialize", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--compare-existing-100m-prefix", action="store_true")
    ap.add_argument("--token-stats", choices=["none", "sample", "full"], default="sample")
    ap.add_argument("--sample-rows", type=int, default=2000)
    ap.add_argument("--max-length", type=int, default=256)
    args = ap.parse_args()

    SCAFFOLD_DIR = args.output_dir
    REPEAT40 = SCAFFOLD_DIR / "repeat_compact_reinvest_40M.jsonl"
    SCAFFOLD_DIR.mkdir(parents=True, exist_ok=True)

    required = {
        "compact40": COMPACT40,
        "compact10": COMPACT10,
        "repeat10": REPEAT10,
        "repeat100": REPEAT100,
        "tokenizer_json": TOKENIZER_DIR / "tokenizer.json",
        "trainer": TRAINER,
        "eval_wrapper": EVAL_WRAPPER,
    }
    missing = {k: str(v) for k, v in required.items() if not v.exists()}
    if missing:
        raise SystemExit(f"missing required files: {missing}")

    mat = None
    if args.materialize or not REPEAT40.exists():
        mat = materialize_repeat40(overwrite=args.overwrite)
    else:
        mat = {"materialized": False, "reason": "existing file retained", "path": str(REPEAT40), "sha256": sha256_file(REPEAT40), "bytes": REPEAT40.stat().st_size}

    counts = {
        "compact40": count_file(COMPACT40),
        "repeat40": count_file(REPEAT40),
        "compact10": count_file(COMPACT10),
        "repeat10": count_file(REPEAT10),
    }
    repeat100_prefix = None
    if args.compare_existing_100m_prefix:
        psha, prows = prefix_sha(REPEAT100, EXPECTED["rows_40m"])
        repeat100_prefix = {"path": str(REPEAT100), "first_rows": prows, "first_40m_sha256": psha, "matches_materialized_repeat40": psha == counts["repeat40"]["sha256"]}

    pair = compare_pair(COMPACT40, REPEAT40)
    token_stats = None
    if args.token_stats != "none":
        sample = None if args.token_stats == "full" else args.sample_rows
        token_stats = token_stats_one_pass(COMPACT10, REPEAT10, TOKENIZER_DIR, max_length=args.max_length, sample_rows=sample)

    ok = True
    problems: list[str] = []
    warnings: list[str] = []
    if counts["compact40"]["sha256"] != EXPECTED["compact40_sha256"]:
        ok = False; problems.append("compact40 sha mismatch")
    if counts["compact10"]["sha256"] != EXPECTED["compact10_sha256"]:
        ok = False; problems.append("compact10 sha mismatch")
    if counts["repeat10"]["sha256"] != EXPECTED["repeat10_sha256"]:
        ok = False; problems.append("repeat10 sha mismatch")
    tok_sha = sha256_file(TOKENIZER_DIR / "tokenizer.json")
    if tok_sha != EXPECTED["tokenizer_sha256"]:
        ok = False; problems.append("research tokenizer sha mismatch")
    for name, exp_rows, exp_words in [("compact40", EXPECTED["rows_40m"], EXPECTED["words_40m"]), ("repeat40", EXPECTED["rows_40m"], EXPECTED["words_40m"]), ("compact10", EXPECTED["rows_per_10m"], EXPECTED["words_per_10m"]), ("repeat10", EXPECTED["rows_per_10m"], EXPECTED["words_per_10m"] )]:
        if counts[name]["rows"] != exp_rows:
            ok = False; problems.append(f"{name} rows {counts[name]['rows']} != {exp_rows}")
        if counts[name]["words"] != exp_words:
            ok = False; problems.append(f"{name} words {counts[name]['words']} != {exp_words}")
    if pair["example_id_mismatches"] != 0 or pair["word_mismatches"] != 0:
        ok = False; problems.append("compact/repeat row id or word mismatch")
    if pair["filler_text_mismatches"] != 0:
        ok = False; problems.append("compact/repeat filler text mismatch")
    expected_changed_diff = EXPECTED["compact_pair_rows_per_pass"] * 4
    expected_changed_equal = EXPECTED["neutral_topup_rows_per_pass"] * 4
    if pair["changed_text_different"] != expected_changed_diff or pair["changed_text_equal"] != expected_changed_equal:
        ok = False; problems.append("changed block compact-vs-repeat/topup counts unexpected")
    if repeat100_prefix and not repeat100_prefix["matches_materialized_repeat40"]:
        warnings.append("materialized repeat40 deliberately matches research ordered 4x10M order; legacy repeat100 has a different row order and is not used for the future paired RoBERTa screen")
    if pair["filler_source_mismatches"] and pair["filler_text_mismatches"] == 0:
        warnings.append("filler source labels differ only because research adds pass prefixes; filler text/example_id/word counts are identical")

    commands = command_manifest(COMPACT40, REPEAT40, SCAFFOLD_DIR)
    payload = {
        "status": "ROBERTA_TRANSFER_PAIR_SCAFFOLD",
        "ok": ok,
        "problems": problems,
        "warnings": warnings,
        "created_utc": now_utc(),
        "scientific_purpose": "Prepare a matched compact-vs-repeat RoBERTa/BERT MLM transfer screen without launching GPU work; future use depends on pending compact-order evidence.",
        "no_training_started": True,
        "no_selected_eval_started": True,
        "human_submission_boundary": "No leaderboard submission is made or authorized by this scaffold.",
        "inputs": {k: str(v) for k, v in required.items()},
        "materialization": mat,
        "counts": counts,
        "repeat100_prefix_check": repeat100_prefix,
        "pair_invariants": pair,
        "token_stats": token_stats,
        "future_command_manifest": commands,
        "future_decision_use": {
            "launch_only_if": "Pending A02 ordered/scrambled selected scores plus channel/exposure-split show stable-family movement and source_absent_content advantage over retained/function controls.",
            "stop_if": "The pending result is close, volatile-column-only, or lacks source_absent-minus-control selectivity; then fluent compact order should not motivate architecture-transfer H100 training.",
            "primary_readouts_after_launch": ["cheap6_no_GlobalPIQA", "cheap5_no_GlobalPIQA_Reading", "Supplement", "EWoK", "Entity", "COMPS", "EWoK_plus_Entity", "category_interactions"],
        },
    }
    out_json = SCAFFOLD_DIR / "roberta_transfer_pair_scaffold.json"
    out_md = SCAFFOLD_DIR / "roberta_transfer_pair_scaffold.md"
    write_json(out_json, payload)
    lines = [
        "# research RoBERTa transfer pair scaffold",
        "",
        f"Status: {'OK' if ok else 'NOT OK'}",
        "",
        "No training or selected evaluation was launched.",
        "",
        "## Data identities",
        f"- compact40: `{COMPACT40}` sha `{counts['compact40']['sha256']}` rows {counts['compact40']['rows']} words {counts['compact40']['words']}",
        f"- repeat40: `{REPEAT40}` sha `{counts['repeat40']['sha256']}` rows {counts['repeat40']['rows']} words {counts['repeat40']['words']}",
        f"- tokenizer: `{TOKENIZER_DIR}` sha `{tok_sha}`",
        "",
        "## Pair invariants",
        f"- row id mismatches: {pair['example_id_mismatches']}",
        f"- word-count mismatches: {pair['word_mismatches']}",
        f"- changed rows different: {pair['changed_text_different']} / {EXPECTED['compact_pair_rows_per_pass'] * 4}",
        f"- changed rows equal topup: {pair['changed_text_equal']} / {EXPECTED['neutral_topup_rows_per_pass'] * 4}",
        f"- filler text mismatches: {pair['filler_text_mismatches']}",
    ]
    if token_stats is not None:
        d = token_stats["compact_minus_repeat_changed_block"]
        td = token_stats["compact_minus_repeat_one_pass"]
        lines += [
            "",
            "## Tokenizer active-token/group differences (one 10M pass; compact minus repeat)",
            f"- total active tokens: {td['active_tokens']}",
            f"- total candidate tokens: {td['candidate_tokens']}",
            f"- total word groups: {td['word_groups']}",
            f"- changed-block active tokens: {d['active_tokens']}",
            f"- changed-block candidate tokens: {d['candidate_tokens']}",
            f"- changed-block word groups: {d['word_groups']}",
            f"- changed-block active tokens/word: {d['active_tokens_per_word']:.6f}",
        ]
    lines += [
        "",
        "## Future commands (not launched)",
        "```",
        commands["training_commands_not_launched"]["compact_reinvest_roberta40M_gpu0"],
        commands["training_commands_not_launched"]["repeat_compact_roberta40M_gpu1"],
        "```",
        "",
        "Future launch remains conditional on the pending compact-order evidence.",
        "",
        "Warnings are contextual rather than failures when legacy 100M row order differs or pass-prefixed source labels differ while text/id/word invariants hold.",
    ]
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "ok": ok, "out": str(out_json), "repeat40_sha256": counts["repeat40"]["sha256"], "pair_ok": not problems, "token_stats": args.token_stats}, indent=2), flush=True)
    if not ok:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
