#!/usr/bin/env python3
"""research legal representation route map for BabyLM Strict-Small.

CPU-only representation-interface study after the compliant 16k tokenizer endpoint
failed to preserve the old inherited-tokenizer score.  It trains alternative
legal byte-level BPE tokenizers on exactly the same 10M pool and measures how
representation capacity changes pretraining exposure, whole-word-mask grouping,
and official evaluation text segmentation.  It does not evaluate or train a
language model.
"""
from __future__ import annotations

import collections
import csv
import hashlib
import json
import math
import pathlib
import statistics
import time
from typing import Any, Iterable

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer
from transformers import AutoTokenizer, PreTrainedTokenizerFast

ROOT = pathlib.Path(".")
STUDY = pathlib.Path("experiments/archive/representation_and_objectives")
WS = STUDY
OUT = WS / "data/legal_representation_route_map"
NOTE = (ROOT / 'research/notes/representation_and_objectives/59_legal_representation_route_map.md')
POOL_10M = pathlib.Path("experiments/archive/frontier_consolidation/data/density_cleanqwen_overlay_medium_riskhard/cleanqwen_fineweb_compact_view_reinvest_10M.jsonl")
OLD_TOKENIZER = pathlib.Path("experiments/archive/initial_model_studies/training/runs/fullcycle_official_wwm_debertav2_8x480_seed43_100M_b256/hf_model")
LEGAL16 = pathlib.Path("experiments/archive/representation_and_objectives/data/strictsmall_tokenizer_retrain/strictsmall_compact_reinvest_16k_tokenizer")
A02_BYTE16 = pathlib.Path("experiments/archive/frontier_consolidation/data/compliant_tokenizer_bytealphabet")
PRISTINE_FULL = pathlib.Path("experiments/archive/representation_and_objectives/data/pristine_official_coordinate/babylm-eval/strict/evaluation_data/full_eval")
GLOBALPIQA_FULL = pathlib.Path("experiments/archive/representation_and_objectives/data/globalpiqa_official_lineage/official_dl_scratch/generated_by_current_official_dl/evaluation_data/full_eval")
COMPARE_JSON = pathlib.Path("experiments/archive/representation_and_objectives/data/corrected_tokenizer_two_seed_comparison/corrected_tokenizer_two_seed_comparison.json")
SURFACE_JSON = pathlib.Path("experiments/archive/representation_and_objectives/data/tokenizer_surface_contingency/tokenizer_surface_contingency.json")
TRAINER = pathlib.Path("experiments/archive/compact_experience/scripts/masking_curriculum_trainer.py")
TEMPLATE_TOK_JSON = OLD_TOKENIZER / "tokenizer.json"

SPECIAL_TOKENS = ["<unk>", "<s>", "</s>", "<pad>", "<mask>"]
SEQ_PRETRAIN = 256
SEQ_FINETUNE = 512
EXPECTED_POOL_SHA = "215944978157394dbecf2039f2c1e1806bfcbb9701421920f78605eb58975a23"

CANDIDATE_SPECS = [
    {"label": "legal_byte_bpe_24k", "vocab_size": 24576},
    {"label": "legal_byte_bpe_32k", "vocab_size": 32768},
    {"label": "legal_byte_bpe_40k", "vocab_size": 40000},
]

SCORE_COLUMNS = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "SuperGLUE", "GlobalPIQA", "Reading", "AoA"]


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def iter_jsonl(path: pathlib.Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def text_iterator(path: pathlib.Path) -> Iterable[str]:
    for row in iter_jsonl(path):
        text = str(row.get("text", ""))
        if text:
            yield text


def q(vals: list[float], p: float) -> float | None:
    if not vals:
        return None
    vals = sorted(vals)
    if len(vals) == 1:
        return float(vals[0])
    pos = p * (len(vals) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(vals[lo])
    return float(vals[lo] * (hi - pos) + vals[hi] * (pos - lo))


def stats(vals: Iterable[float]) -> dict[str, Any]:
    xs = [float(x) for x in vals if x is not None and not (isinstance(x, float) and math.isnan(x))]
    if not xs:
        return {"n": 0}
    return {
        "n": len(xs),
        "mean": float(sum(xs) / len(xs)),
        "std": float(statistics.pstdev(xs)) if len(xs) > 1 else 0.0,
        "min": float(min(xs)),
        "p05": q(xs, 0.05),
        "p50": q(xs, 0.50),
        "p95": q(xs, 0.95),
        "p99": q(xs, 0.99),
        "max": float(max(xs)),
        "sum": float(sum(xs)),
    }


def source_group(row: dict[str, Any]) -> str:
    src = str(row.get("source", row.get("source_name", "unknown")))
    if src == "cleanqwen_fineweb_compact_view_reinvest":
        return "compact_changed_block"
    if src.startswith("neutral_cleanqwen_topup_compact_reinvest"):
        return "compact_neutral_topup"
    if src == "qwen_pair_packed":
        return "inherited_qwen_pairs"
    if src in {"childes", "gutenberg", "open_subtitles", "simple_wiki", "bnc_spoken", "switchboard"}:
        return f"official::{src}"
    return f"other::{src}"


def collect_pool_rows(path: pathlib.Path) -> list[dict[str, Any]]:
    rows = []
    words = 0
    for idx, row in enumerate(iter_jsonl(path)):
        text = str(row.get("text", ""))
        w = int(row.get("words", len(text.split())))
        if w != len(text.split()):
            raise RuntimeError(f"word-count mismatch at row {idx}: field={w} actual={len(text.split())}")
        words += w
        rows.append({"text": text, "words": w, "source_group": source_group(row), "example_id": row.get("example_id")})
    return rows


def train_byte_tokenizer(label: str, vocab_size: int) -> pathlib.Path:
    out_dir = OUT / "tokenizers" / label
    tok_file = out_dir / "tokenizer.json"
    if tok_file.exists():
        return out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    template = Tokenizer.from_file(str(TEMPLATE_TOK_JSON))
    tok = Tokenizer(BPE(unk_token="<unk>"))
    tok.normalizer = template.normalizer
    tok.pre_tokenizer = template.pre_tokenizer
    tok.post_processor = template.post_processor
    tok.decoder = template.decoder
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=SPECIAL_TOKENS,
        min_frequency=2,
        initial_alphabet=ByteLevel.alphabet(),
        show_progress=True,
    )
    t0 = time.time()
    tok.train_from_iterator(text_iterator(POOL_10M), trainer=trainer)
    tok.save(str(tok_file))
    hf_tok = PreTrainedTokenizerFast(
        tokenizer_file=str(tok_file),
        unk_token="<unk>", bos_token="<s>", eos_token="</s>", pad_token="<pad>", mask_token="<mask>",
        model_max_length=1024,
    )
    hf_tok.save_pretrained(str(out_dir))
    meta = {
        "status": "trained",
        "label": label,
        "vocab_size_requested": vocab_size,
        "vocab_size_actual": tok.get_vocab_size(),
        "training_pool": str(POOL_10M),
        "training_pool_sha256": sha256_file(POOL_10M),
        "tokenizer_json_sha256": sha256_file(tok_file),
        "special_token_ids": {s: tok.token_to_id(s) for s in SPECIAL_TOKENS},
        "missing_bytelevel_alphabet": sorted(set(ByteLevel.alphabet()) - set(tok.get_vocab().keys())),
        "elapsed_sec": round(time.time() - t0, 3),
    }
    (out_dir / "tokenizer_route_metadata.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return out_dir


def load_tokenizers() -> dict[str, Any]:
    tokenizers: dict[str, Any] = {}
    tokenizers["old_inherited_16k"] = AutoTokenizer.from_pretrained(str(OLD_TOKENIZER), use_fast=True)
    tokenizers["legal_a01_16k"] = AutoTokenizer.from_pretrained(str(LEGAL16), use_fast=True)
    if A02_BYTE16.exists():
        tokenizers["legal_a02_bytealpha_16k"] = AutoTokenizer.from_pretrained(str(A02_BYTE16), use_fast=True)
    for spec in CANDIDATE_SPECS:
        p = train_byte_tokenizer(spec["label"], int(spec["vocab_size"]))
        tokenizers[spec["label"]] = AutoTokenizer.from_pretrained(str(p), use_fast=True)
    return tokenizers


def tok_ids(tok, text: str) -> list[int]:
    return tok(text, add_special_tokens=False, truncation=False)["input_ids"]


def token_strings(tok, ids: list[int]) -> list[str]:
    return [str(x) for x in tok.convert_ids_to_tokens(ids)]


def is_word_start(t: str) -> bool:
    return t.startswith("Ġ") or t.startswith("▁")


def word_group_lengths(tok, text: str) -> tuple[int, list[int]]:
    ids = tok_ids(tok, text)
    toks = token_strings(tok, ids)
    lengths: list[int] = []
    cur = 0
    for i, t in enumerate(toks):
        if i == 0 or is_word_start(t):
            if cur:
                lengths.append(cur)
            cur = 1
        else:
            cur += 1
    if cur:
        lengths.append(cur)
    return len(ids), lengths


def visible_prefix_stats(tok, text: str, max_len: int) -> tuple[int, int]:
    words = text.split()
    if not words:
        return 0, 0
    ids = tok_ids(tok, text)
    if len(ids) <= max_len:
        n_groups = len(word_group_lengths(tok, text)[1])
        return len(words), n_groups
    lo, hi = 0, len(words)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if len(tok_ids(tok, " ".join(words[:mid]))) <= max_len:
            lo = mid
        else:
            hi = mid - 1
    _, groups = word_group_lengths(tok, " ".join(words[:lo])) if lo else (0, [])
    return lo, len(groups)


def analyze_pool_for_tok(rows: list[dict[str, Any]], tok, max_len: int = SEQ_PRETRAIN) -> dict[str, Any]:
    total_words = 0
    total_tokens = 0
    total_groups = 0
    visible_words = 0
    visible_groups = 0
    rows_over = 0
    per_group: dict[str, dict[str, Any]] = collections.defaultdict(lambda: {
        "rows": 0, "words": 0, "tokens": 0, "groups": 0, "visible_words": 0, "visible_groups": 0,
        "rows_over": 0, "tok_per_word_vals": [], "groups_per_word_vals": [], "group_len_vals": []
    })
    high_fragment_examples = []
    for i, row in enumerate(rows):
        text = row["text"]
        words = int(row["words"])
        n_tok, group_lens = word_group_lengths(tok, text)
        n_groups = len(group_lens)
        vw, vg = visible_prefix_stats(tok, text, max_len)
        total_words += words
        total_tokens += n_tok
        total_groups += n_groups
        visible_words += vw
        visible_groups += vg
        over = n_tok > max_len
        rows_over += int(over)
        g = per_group[row["source_group"]]
        g["rows"] += 1
        g["words"] += words
        g["tokens"] += n_tok
        g["groups"] += n_groups
        g["visible_words"] += vw
        g["visible_groups"] += vg
        g["rows_over"] += int(over)
        if words:
            g["tok_per_word_vals"].append(n_tok / words)
            g["groups_per_word_vals"].append(n_groups / words)
        g["group_len_vals"].extend(group_lens)
        if len(high_fragment_examples) < 12 and words and (n_tok / words >= 1.85 or max(group_lens or [0]) >= 6):
            high_fragment_examples.append({"row_index": i, "source_group": row["source_group"], "words": words, "tokens": n_tok, "groups": n_groups, "max_group_len": max(group_lens or [0]), "text_excerpt": text[:220]})
    group_rows = {}
    for name, g in sorted(per_group.items()):
        group_rows[name] = {
            "rows": g["rows"],
            "words": g["words"],
            "tokens": g["tokens"],
            "tokens_per_word": g["tokens"] / max(1, g["words"]),
            "wwm_groups": g["groups"],
            "groups_per_word": g["groups"] / max(1, g["words"]),
            "visible_word_fraction_seq256": g["visible_words"] / max(1, g["words"]),
            "visible_group_fraction_seq256": g["visible_groups"] / max(1, g["groups"]),
            "rows_over_seq256": g["rows_over"],
            "token_per_word_distribution": stats(g["tok_per_word_vals"]),
            "group_len_distribution": stats(g["group_len_vals"]),
        }
    return {
        "rows": len(rows),
        "words": total_words,
        "tokens": total_tokens,
        "tokens_per_word": total_tokens / max(1, total_words),
        "wwm_groups": total_groups,
        "groups_per_word": total_groups / max(1, total_words),
        "visible_word_fraction_seq256": visible_words / max(1, total_words),
        "visible_group_fraction_seq256": visible_groups / max(1, total_groups),
        "rows_over_seq256": rows_over,
        "by_source_group": group_rows,
        "high_fragment_examples": high_fragment_examples,
    }


def collect_eval_records() -> Iterable[dict[str, Any]]:
    for family, dname in [("BLiMP", "blimp_filtered"), ("Supplement", "supplement_filtered")]:
        d = PRISTINE_FULL / dname
        for p in sorted(d.glob("*.jsonl")):
            sub = p.stem
            for i, obj in enumerate(iter_jsonl(p)):
                for role, key in [("good", "sentence_good"), ("bad", "sentence_bad")]:
                    if key in obj:
                        yield {"family": family, "subtask": sub, "role": role, "row": i, "text": str(obj[key])}
    d = PRISTINE_FULL / "ewok_filtered"
    for p in sorted(d.glob("*.jsonl")):
        sub = p.stem
        for i, obj in enumerate(iter_jsonl(p)):
            for key in ["Context1", "Context2", "Target1", "Target2"]:
                if key in obj:
                    yield {"family": "EWoK", "subtask": sub, "role": key, "row": i, "text": str(obj[key])}
            if all(k in obj for k in ["Context1", "Target1", "Context2", "Target2"]):
                yield {"family": "EWoK_concat", "subtask": sub, "role": "ctx1_target1", "row": i, "text": str(obj["Context1"]) + " " + str(obj["Target1"])}
                yield {"family": "EWoK_concat", "subtask": sub, "role": "ctx2_target2", "row": i, "text": str(obj["Context2"]) + " " + str(obj["Target2"])}
    d = PRISTINE_FULL / "entity_tracking"
    for p in sorted(d.glob("*.jsonl")):
        sub = p.stem
        for i, obj in enumerate(iter_jsonl(p)):
            prefix = str(obj.get("input_prefix", ""))
            if prefix:
                yield {"family": "Entity", "subtask": sub, "role": "input_prefix", "row": i, "text": prefix}
            for j, opt in enumerate(obj.get("options", [])):
                yield {"family": "Entity", "subtask": sub, "role": f"prefix_option{j}", "row": i, "text": prefix + str(opt)}
    d = PRISTINE_FULL / "comps"
    for p in sorted(d.glob("*.jsonl")):
        sub = p.stem
        for i, obj in enumerate(iter_jsonl(p)):
            prop = str(obj.get("property_phrase", obj.get("property", "")))
            for role, pk in [("acceptable", "prefix_acceptable"), ("unacceptable", "prefix_unacceptable")]:
                if pk in obj:
                    yield {"family": "COMPS", "subtask": sub, "role": role, "row": i, "text": str(obj[pk]) + " " + prop}
    for family, dname in [("GlobalPIQA_parallel", "global_piqa_parallel"), ("GlobalPIQA_nonparallel", "global_piqa_nonparallel")]:
        d = GLOBALPIQA_FULL / dname
        for p in sorted(d.glob("*.jsonl")):
            for i, obj in enumerate(iter_jsonl(p)):
                prompt = str(obj.get("prompt", ""))
                if prompt:
                    yield {"family": family, "subtask": p.stem, "role": "prompt", "row": i, "text": prompt}
                for k, v in obj.items():
                    if k.startswith("solution") and isinstance(v, str):
                        yield {"family": family, "subtask": p.stem, "role": k, "row": i, "text": prompt + " " + v}
    rp = PRISTINE_FULL / "reading/reading_data.csv"
    with rp.open("r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            sent = str(row.get("sentence", ""))
            word = str(row.get("word", ""))
            if sent:
                yield {"family": "Reading_sentence", "subtask": "reading_data", "role": "sentence", "row": i, "text": sent}
            if word:
                yield {"family": "Reading_word", "subtask": "reading_data", "role": "word", "row": i, "text": word}
    gd = PRISTINE_FULL / "glue_filtered"
    for p in sorted(gd.glob("*.jsonl")):
        parts = p.name.split(".")
        task = parts[0]
        split = parts[1] if len(parts) > 2 else "unknown"
        for i, obj in enumerate(iter_jsonl(p)):
            fields = []
            for k, v in obj.items():
                if k == "label":
                    continue
                if isinstance(v, str):
                    fields.append(v)
            if fields:
                yield {"family": "SuperGLUE", "subtask": task, "role": "all_text_fields", "row": i, "text": " </s> ".join(fields)}


class EvalAcc:
    def __init__(self):
        self.n = 0
        self.tokens = 0
        self.group_count = 0
        self.lens: list[float] = []
        self.groups: list[float] = []
        self.unks = 0
        self.rows_with_unk = 0
        self.over256 = 0
        self.over512 = 0
        self.word_start_tokens = 0
        self.max_group_lens: list[float] = []

    def add(self, tok, text: str) -> None:
        ids = tok_ids(tok, text)
        toks = token_strings(tok, ids)
        unk_id = tok.unk_token_id
        unk_here = sum(1 for x in ids if unk_id is not None and int(x) == int(unk_id))
        _, groups = word_group_lengths(tok, text)
        self.n += 1
        self.tokens += len(ids)
        self.group_count += len(groups)
        self.lens.append(float(len(ids)))
        self.groups.append(float(len(groups)))
        self.unks += unk_here
        self.rows_with_unk += int(unk_here > 0)
        self.over256 += int(len(ids) > 256)
        self.over512 += int(len(ids) > 512)
        self.word_start_tokens += sum(1 for s in toks if is_word_start(s))
        self.max_group_lens.append(float(max(groups or [0])))

    def finish(self) -> dict[str, Any]:
        return {
            "n_texts": self.n,
            "total_tokens": self.tokens,
            "mean_tokens_per_text": self.tokens / max(1, self.n),
            "token_count_distribution": stats(self.lens),
            "wwm_groups_total": self.group_count,
            "mean_groups_per_text": self.group_count / max(1, self.n),
            "mean_tokens_per_wwm_group": self.tokens / max(1, self.group_count),
            "max_group_len_distribution": stats(self.max_group_lens),
            "word_start_token_fraction": self.word_start_tokens / max(1, self.tokens),
            "unk_count": self.unks,
            "rows_with_unk": self.rows_with_unk,
            "rows_over256": self.over256,
            "rows_over512": self.over512,
        }


def analyze_eval_for_tokenizers(tokenizers: dict[str, Any]) -> dict[str, Any]:
    acc: dict[str, dict[str, EvalAcc]] = {label: collections.defaultdict(EvalAcc) for label in tokenizers}
    total_records = 0
    for rec in collect_eval_records():
        total_records += 1
        fam = rec["family"]
        sub = rec["subtask"]
        groups = [f"family::{fam}", f"subtask::{fam}::{sub}"]
        if fam in ("GlobalPIQA_parallel", "GlobalPIQA_nonparallel"):
            groups.append("family::GlobalPIQA")
        if fam.startswith("Reading"):
            groups.append("family::Reading")
        for label, tok in tokenizers.items():
            for g in groups:
                acc[label][g].add(tok, rec["text"])
    out: dict[str, Any] = {"total_eval_text_records": total_records, "by_tokenizer": {}}
    for label, groups in acc.items():
        out["by_tokenizer"][label] = {g: a.finish() for g, a in sorted(groups.items())}
    return out


def read_scores() -> dict[str, Any]:
    cmp = json.loads(COMPARE_JSON.read_text(encoding="utf-8"))
    return cmp.get("comparisons", {})


def compare_tokenizer_surfaces(eval_summary: dict[str, Any], ref_label: str, other_label: str) -> dict[str, Any]:
    ref = eval_summary["by_tokenizer"][ref_label]
    other = eval_summary["by_tokenizer"][other_label]
    rows = {}
    for g in sorted(set(ref) & set(other)):
        r = ref[g]
        o = other[g]
        rows[g] = {
            "new_over_ref_total_tokens": o["total_tokens"] / max(1, r["total_tokens"]),
            "delta_mean_tokens_per_text": o["mean_tokens_per_text"] - r["mean_tokens_per_text"],
            "new_over_ref_wwm_groups": o["wwm_groups_total"] / max(1, r["wwm_groups_total"]),
            "delta_mean_tokens_per_wwm_group": o["mean_tokens_per_wwm_group"] - r["mean_tokens_per_wwm_group"],
            "unk_count_other": o["unk_count"],
            "rows_with_unk_other": o["rows_with_unk"],
            "rows_over256_other": o["rows_over256"],
            "rows_over512_other": o["rows_over512"],
        }
    return rows


def summarize_candidate_routes(pool_summary: dict[str, Any], eval_summary: dict[str, Any], score_movements: dict[str, Any]) -> dict[str, Any]:
    tokenizers = list(pool_summary.keys())
    routes = []
    ref = "legal_a01_16k"
    for label in tokenizers:
        if label == ref:
            continue
        tr = pool_summary[label]
        rr = pool_summary[ref]
        eval_cmp = compare_tokenizer_surfaces(eval_summary, ref, label)
        fams = {}
        for fam in ["family::Supplement", "family::EWoK", "family::Entity", "family::COMPS", "family::GlobalPIQA", "family::SuperGLUE", "family::BLiMP", "family::Reading"]:
            if fam in eval_cmp:
                fams[fam] = eval_cmp[fam]
        routes.append({
            "candidate": label,
            "train_tokens_over_legal16": tr["tokens"] / max(1, rr["tokens"]),
            "train_groups_over_legal16": tr["wwm_groups"] / max(1, rr["wwm_groups"]),
            "train_visible_word_fraction": tr["visible_word_fraction_seq256"],
            "train_visible_group_fraction": tr["visible_group_fraction_seq256"],
            "rows_over_seq256": tr["rows_over_seq256"],
            "eval_family_surface_vs_legal16": fams,
        })
    return {
        "score_movements_to_explain": {
            "corrected_mean_minus_inherited_mean": score_movements.get("corrected_mean_minus_inherited_mean"),
            "corrected_two_seed_mean": score_movements.get("corrected_two_seed_mean"),
        },
        "candidate_representation_surfaces": routes,
        "route_interpretation": [
            "The legal 16k tokenizer loses Supplement and EWoK while improving GlobalPIQA on average; a pure eval-length story is insufficient because Supplement barely lengthened and GlobalPIQA also lengthened but improved.",
            "A legal larger byte-BPE tokenizer is a distinct route from copying the public leader: it changes capacity and MLM target geometry while preserving byte coverage and the exact 10M source pool.",
            "Sequence-length and masking curricula should be tested as a separate route because they alter when the model predicts words versus subword pieces, not merely which vocabulary entries exist.",
        ],
    }


def write_tables(payload: dict[str, Any]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    # Training surface table.
    with (OUT / "training_pool_tokenizer_surface.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["tokenizer", "vocab_size", "tokens_per_word", "groups_per_word", "visible_word_fraction_seq256", "visible_group_fraction_seq256", "rows_over_seq256"])
        writer.writeheader()
        for label, r in payload["training_pool_surface"].items():
            writer.writerow({
                "tokenizer": label,
                "vocab_size": payload["tokenizers"][label]["vocab_size"],
                "tokens_per_word": r["tokens_per_word"],
                "groups_per_word": r["groups_per_word"],
                "visible_word_fraction_seq256": r["visible_word_fraction_seq256"],
                "visible_group_fraction_seq256": r["visible_group_fraction_seq256"],
                "rows_over_seq256": r["rows_over_seq256"],
            })
    # Eval family table relative to legal 16k.
    with (OUT / "eval_family_surface_vs_legal16.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["tokenizer", "family", "token_ratio_vs_legal16", "wwm_group_ratio_vs_legal16", "delta_mean_tokens_per_text", "delta_mean_tokens_per_wwm_group", "unk_count", "rows_with_unk"])
        writer.writeheader()
        for label, rel in payload["surface_relative_to_legal16"].items():
            for g, row in rel.items():
                if not g.startswith("family::"):
                    continue
                writer.writerow({
                    "tokenizer": label,
                    "family": g.replace("family::", ""),
                    "token_ratio_vs_legal16": row["new_over_ref_total_tokens"],
                    "wwm_group_ratio_vs_legal16": row["new_over_ref_wwm_groups"],
                    "delta_mean_tokens_per_text": row["delta_mean_tokens_per_text"],
                    "delta_mean_tokens_per_wwm_group": row["delta_mean_tokens_per_wwm_group"],
                    "unk_count": row["unk_count_other"],
                    "rows_with_unk": row["rows_with_unk_other"],
                })
    # Supplement and EWoK subtasks for candidate route reading.
    with (OUT / "supplement_ewok_subtask_surface_vs_legal16.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["tokenizer", "subtask", "token_ratio_vs_legal16", "wwm_group_ratio_vs_legal16", "delta_mean_tokens_per_text", "unk_count"])
        writer.writeheader()
        for label, rel in payload["surface_relative_to_legal16"].items():
            for g, row in rel.items():
                if g.startswith("subtask::Supplement::") or g.startswith("subtask::EWoK::"):
                    writer.writerow({
                        "tokenizer": label,
                        "subtask": g.replace("subtask::", ""),
                        "token_ratio_vs_legal16": row["new_over_ref_total_tokens"],
                        "wwm_group_ratio_vs_legal16": row["new_over_ref_wwm_groups"],
                        "delta_mean_tokens_per_text": row["delta_mean_tokens_per_text"],
                        "unk_count": row["unk_count_other"],
                    })


def write_note(payload: dict[str, Any]) -> None:
    cmp_mean = payload["candidate_route_synthesis"]["score_movements_to_explain"].get("corrected_mean_minus_inherited_mean") or {}
    legal_mean = payload["candidate_route_synthesis"]["score_movements_to_explain"].get("corrected_two_seed_mean") or {}
    lines: list[str] = []
    lines.append("# research legal representation route map\n\n")
    lines.append("This note rebuilds the route after the two legal-tokenizer compact-view endpoints scored 40.704 and 41.024. It uses only CPU tokenizer training and text-interface measurement; it is not a model score.\n\n")
    lines.append("## Score movement that must be explained\n\n")
    lines.append(f"The legal 16k two-seed mean is Overall {legal_mean.get('Overall'):.6f} with Supplement {legal_mean.get('Supplement'):.3f}, EWoK {legal_mean.get('EWoK'):.3f}, Entity {legal_mean.get('Entity'):.3f}, COMPS {legal_mean.get('COMPS'):.3f}, GlobalPIQA {legal_mean.get('GlobalPIQA'):.3f}. Relative to the inherited-tokenizer mechanism coordinate, the mean movement is Supplement {cmp_mean.get('Supplement'):+.3f}, EWoK {cmp_mean.get('EWoK'):+.3f}, SuperGLUE {cmp_mean.get('SuperGLUE'):+.3f}, GlobalPIQA {cmp_mean.get('GlobalPIQA'):+.3f}, Entity {cmp_mean.get('Entity'):+.3f}, COMPS {cmp_mean.get('COMPS'):+.3f}, Overall {cmp_mean.get('Overall'):+.3f}.\n\n")
    lines.append("## Legal tokenizers trained on the exact same 10M pool\n\n")
    lines.append("| tokenizer | vocab | train tokens/word | train WWM groups/word | visible words@256 | visible groups@256 | rows >256 |\n")
    lines.append("|---|---:|---:|---:|---:|---:|---:|\n")
    for label, r in payload["training_pool_surface"].items():
        lines.append(f"| {label} | {payload['tokenizers'][label]['vocab_size']} | {r['tokens_per_word']:.4f} | {r['groups_per_word']:.4f} | {r['visible_word_fraction_seq256']:.4f} | {r['visible_group_fraction_seq256']:.4f} | {r['rows_over_seq256']} |\n")
    lines.append("\nThe trainer forms whole-word-mask units by token-string prefixes (`Ġ` or `▁`). A tokenizer change therefore changes both embedding/output vocabulary and the units receiving early WWM prediction pressure.\n\n")
    lines.append("## Official evaluation text relative to A01 legal 16k\n\n")
    lines.append("| tokenizer | family | token ratio | WWM-unit ratio | mean token delta/text | token/group delta | unknown tokens |\n")
    lines.append("|---|---|---:|---:|---:|---:|---:|\n")
    for label, rel in payload["surface_relative_to_legal16"].items():
        if label == "legal_a01_16k":
            continue
        for fam in ["family::Supplement", "family::EWoK", "family::Entity", "family::COMPS", "family::GlobalPIQA", "family::SuperGLUE", "family::BLiMP", "family::Reading"]:
            row = rel.get(fam)
            if not row:
                continue
            lines.append(f"| {label} | {fam.replace('family::','')} | {row['new_over_ref_total_tokens']:.4f} | {row['new_over_ref_wwm_groups']:.4f} | {row['delta_mean_tokens_per_text']:+.3f} | {row['delta_mean_tokens_per_wwm_group']:+.4f} | {row['unk_count_other']} |\n")
    lines.append("\n## Route synthesis\n\n")
    lines.append("The paired old/new-tokenizer evidence does not support a narrow relation-retention replacement as the next expensive step. The broad loss is not just a few compact pairs: Supplement collapses despite almost unchanged length under A01 legal16, EWoK falls even though crude token-length features were weakly linked to focused EWoK margins, while GlobalPIQA improves under the same legal representation. The central mechanism is representation-learning interaction: which wordpieces exist, how WWM groups are formed, and how compact/source pairs allocate prediction pressure across rare relational, dialogue, morphology, and practical-action surfaces.\n\n")
    lines.append("The next H100 run should be selected by cheap evidence among three genuinely distinct legal routes: (1) higher-capacity legal byte-BPE trained on the same exact 10M pool, especially 32k/40k, testing whether the old representation effect was capacity/target-geometry rather than illegal corpus knowledge; (2) joint corpus-tokenizer redesign, changing the 10M pool so the tokenizer and compact views co-adapt rather than fitting a tokenizer to a corpus dominated by official filler; (3) sequence/masking curriculum, testing whether early short contexts plus late token-level masking recovers Supplement/EWoK without sacrificing the GlobalPIQA/Entity gains.\n\n")
    lines.append("Recommended immediate continuation: run only short early-dynamics experiments, not full 100M. Train one seed for 20M exposure for legal_byte_bpe_40k under the current data/recipe and, separately, legal_a01_16k with WWM7→token3/64→256 curriculum if the trainer can encode the length schedule. Evaluate cheap zero-shot columns plus focused EWoK/Supplement at 10M/20M. Continue a route only if it moves the same broad columns that failed here; do not launch a small relation-retention swap as the next H100 experiment.\n\n")
    lines.append("## Files\n\n")
    lines.append(f"JSON: `{OUT / 'legal_representation_route_map.json'}`\n")
    lines.append(f"Training surface CSV: `{OUT / 'training_pool_tokenizer_surface.csv'}`\n")
    lines.append(f"Evaluation family surface CSV: `{OUT / 'eval_family_surface_vs_legal16.csv'}`\n")
    lines.append(f"Supplement/EWoK subtask surface CSV: `{OUT / 'supplement_ewok_subtask_surface_vs_legal16.csv'}`\n")
    NOTE.write_text("".join(lines), encoding="utf-8")


def main() -> None:
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    if sha256_file(POOL_10M) != EXPECTED_POOL_SHA:
        raise RuntimeError("10M pool SHA mismatch")
    if not COMPARE_JSON.exists():
        raise FileNotFoundError(COMPARE_JSON)
    pool_rows = collect_pool_rows(POOL_10M)
    tokenizers = load_tokenizers()
    tok_meta = {}
    for label, tok in tokenizers.items():
        tok_dir = pathlib.Path(getattr(tok, "name_or_path", ""))
        tok_json = tok_dir / "tokenizer.json" if tok_dir else pathlib.Path("")
        tok_meta[label] = {
            "name_or_path": str(getattr(tok, "name_or_path", "")),
            "vocab_size": len(tok),
            "special_ids": tok.all_special_ids,
            "unk_token_id": tok.unk_token_id,
            "tokenizer_json_sha256": sha256_file(tok_json) if tok_json.exists() else None,
        }
    pool_summary = {label: analyze_pool_for_tok(pool_rows, tok) for label, tok in tokenizers.items()}
    eval_summary = analyze_eval_for_tokenizers(tokenizers)
    relative = {}
    for label in tokenizers:
        relative[label] = compare_tokenizer_surfaces(eval_summary, "legal_a01_16k", label)
    score_movements = read_scores()
    synthesis = summarize_candidate_routes(pool_summary, eval_summary, score_movements)
    payload = {
        "status": "LEGAL_REPRESENTATION_ROUTE_MAP",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "purpose": "Use cheap representation-interface evidence to choose the next legal route after the compliant 16k compact-view endpoint failed below the 41.8 leader.",
        "pool_10M": {"path": str(POOL_10M), "sha256": sha256_file(POOL_10M), "rows": len(pool_rows), "words": sum(r["words"] for r in pool_rows)},
        "tokenizers": tok_meta,
        "training_pool_surface": pool_summary,
        "eval_surface": eval_summary,
        "surface_relative_to_legal16": relative,
        "candidate_route_synthesis": synthesis,
        "external_method_context": {
            "leader_model_card": "go76dof/wwm_curriculum_simplification_40k uses 40k SentencePiece BPE trained for its 9,999,969-word FineWeb simplification-pair condition, 12x384 DeBERTa-v2, LAMB LR 0.007, sequence 64->256, WWM epochs 1-7 then token masking epochs 8-10.",
            "mask_and_receive": "Prior BabyLM evidence found vocabulary-size effects depend on dataset; 40k was stronger on the new paraphrase-like dataset for BLiMP and Entity, while 16k was not uniformly worse on the original dataset. This argues for testing representation-data interaction, not copying 40k as a universal rule.",
        },
        "elapsed_sec": round(time.time() - t0, 3),
    }
    write_tables(payload)
    write_note(payload)
    (OUT / "legal_representation_route_map.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": payload["status"],
        "elapsed_sec": payload["elapsed_sec"],
        "tokenizers": {k: v["vocab_size"] for k, v in tok_meta.items()},
        "out_json": str(OUT / "legal_representation_route_map.json"),
        "note": str(NOTE),
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
