#!/usr/bin/env python3
"""research: token-support spectra for legal compact_view_reinvest tokenizers.

Purpose
-------
The legal-40k compact_view_reinvest trainings are running. If their full official
vectors fail, the next expensive route should not be another vocabulary-size
sweep; it should attack a distinct mechanism. One plausible mechanism from the
research independent review memo is support density: a larger 10M-trained vocabulary shortens
strings but may allocate many embeddings/output rows to very low-count units.

This CPU-only script trains support-floored 40k byte-BPE candidates strictly on
the exact allowed 10M pool and compares them with the existing legal 16/24/32/40k
byte-BPE tokenizers. It counts token occurrence support on the 10M pool, separates
new vocabulary entries relative to legal16k, and measures how often official
evaluation texts are encoded with low-support training tokens. The evaluation
texts are used only for post-hoc surface analysis, never for tokenizer training.
"""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import time
from typing import Any, Iterable

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer
from transformers import PreTrainedTokenizerFast

ROOT = Path(".").resolve()
STUDY = ROOT / "experiments/archive/representation_and_objectives"
WS = STUDY
OUT = WS / "data/tokenizer_support_spectrum"
NOTE = (ROOT / 'research/notes/representation_and_objectives/tokenizer_support_spectrum.md')
POOL_10M = ROOT / "experiments/archive/frontier_consolidation/data/density_cleanqwen_overlay_medium_riskhard/cleanqwen_fineweb_compact_view_reinvest_10M.jsonl"
EXPECTED_POOL_SHA = "215944978157394dbecf2039f2c1e1806bfcbb9701421920f78605eb58975a23"
TEMPLATE_TOK_JSON = ROOT / "experiments/archive/initial_model_studies/training/runs/fullcycle_official_wwm_debertav2_8x480_seed43_100M_b256/hf_model/tokenizer.json"
LEGAL16 = WS / "data/strictsmall_tokenizer_retrain/strictsmall_compact_reinvest_16k_tokenizer"
ROUTE_MAP_TOK = WS / "data/legal_representation_route_map/tokenizers"
PRISTINE_FULL = WS / "data/pristine_official_coordinate/babylm-eval/strict/evaluation_data/full_eval"
GLOBALPIQA_FULL = WS / "data/globalpiqa_official_lineage/official_dl_scratch/generated_by_current_official_dl/evaluation_data/full_eval"

SPECIAL_TOKENS = ["<unk>", "<s>", "</s>", "<pad>", "<mask>"]
THRESHOLDS = [5, 10, 20, 50, 100, 200, 500]
BATCH_SIZE = 512

EXISTING_TOKENIZERS = {
    "legal_a01_16k": LEGAL16,
    "legal_byte_bpe_24k": ROUTE_MAP_TOK / "legal_byte_bpe_24k",
    "legal_byte_bpe_32k": ROUTE_MAP_TOK / "legal_byte_bpe_32k",
    "legal_byte_bpe_40k": ROUTE_MAP_TOK / "legal_byte_bpe_40k",
}

SUPPORT_FLOOR_SPECS = [
    {"label": "legal_byte_bpe_40k_minfreq10", "vocab_size": 40000, "min_frequency": 10},
    {"label": "legal_byte_bpe_40k_minfreq25", "vocab_size": 40000, "min_frequency": 25},
    {"label": "legal_byte_bpe_40k_minfreq50", "vocab_size": 40000, "min_frequency": 50},
]


def now_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def iter_pool_rows() -> Iterable[dict[str, Any]]:
    for i, row in enumerate(iter_jsonl(POOL_10M)):
        text = str(row.get("text", ""))
        words = int(row.get("words", len(text.split())))
        actual = len(text.split())
        if words != actual:
            raise RuntimeError(f"word count mismatch row {i}: field={words} actual={actual}")
        yield {"index": i, "text": text, "words": words, "source": str(row.get("source", "unknown")), "example_id": row.get("example_id")}


def pool_text_iterator() -> Iterable[str]:
    for row in iter_pool_rows():
        yield row["text"]


def quantiles(vals: list[float], ps=(0.0, 0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 1.0)) -> dict[str, float | None]:
    xs = sorted(float(x) for x in vals if math.isfinite(float(x)))
    if not xs:
        return {f"p{int(p*100):02d}": None for p in ps}
    out: dict[str, float | None] = {}
    for p in ps:
        pos = p * (len(xs) - 1)
        lo = int(math.floor(pos))
        hi = int(math.ceil(pos))
        if lo == hi:
            val = xs[lo]
        else:
            val = xs[lo] * (hi - pos) + xs[hi] * (pos - lo)
        out[f"p{int(p*100):02d}"] = float(val)
    return out


def train_support_floor_tokenizer(label: str, vocab_size: int, min_frequency: int, force: bool = False) -> Path:
    out_dir = OUT / "tokenizers" / label
    tok_file = out_dir / "tokenizer.json"
    if tok_file.exists() and not force:
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
        min_frequency=min_frequency,
        initial_alphabet=ByteLevel.alphabet(),
        show_progress=True,
    )
    t0 = time.time()
    tok.train_from_iterator(pool_text_iterator(), trainer=trainer)
    tok.save(str(tok_file))
    hf = PreTrainedTokenizerFast(
        tokenizer_file=str(tok_file),
        unk_token="<unk>", bos_token="<s>", eos_token="</s>", pad_token="<pad>", mask_token="<mask>",
        model_max_length=1024,
    )
    hf.save_pretrained(str(out_dir))
    meta = {
        "status": "support_floor_tokenizer_trained_on_allowed_10M_only",
        "created_utc": now_utc(),
        "label": label,
        "requested_vocab_size": vocab_size,
        "actual_vocab_size": tok.get_vocab_size(),
        "min_frequency": min_frequency,
        "training_pool": str(POOL_10M),
        "training_pool_sha256": sha256_file(POOL_10M),
        "tokenizer_json_sha256": sha256_file(tok_file),
        "special_token_ids": {s: tok.token_to_id(s) for s in SPECIAL_TOKENS},
        "missing_bytelevel_alphabet_count": len(set(ByteLevel.alphabet()) - set(tok.get_vocab().keys())),
        "elapsed_sec": round(time.time() - t0, 3),
        "compliance_note": "Vocabulary learned only from the exact compact_view_reinvest 10M pool; official evaluation text is not used for training.",
    }
    (out_dir / "tokenizer_support_floor_metadata.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return out_dir


def tokenizer_json_path(tok_dir: Path) -> Path:
    p = tok_dir / "tokenizer.json"
    if not p.exists():
        raise FileNotFoundError(p)
    return p


def load_tokenizer(tok_dir: Path) -> Tokenizer:
    tok = Tokenizer.from_file(str(tokenizer_json_path(tok_dir)))
    # Some HF-saved tokenizer.json files preserve padding/truncation state.  The
    # support spectrum must count real unpadded, untruncated text tokens, matching
    # the earlier tokenizer audits and training/evaluation calls that pass
    # add_special_tokens=False, padding disabled, and truncation disabled for raw
    # surface measurement.  Leaving raw padding enabled makes short eval strings
    # appear as exactly 256 tokens and corrupts support statistics.
    try:
        tok.no_padding()
    except Exception:
        pass
    try:
        tok.no_truncation()
    except Exception:
        pass
    return tok


def encode_batch(tok: Tokenizer, texts: list[str]) -> list[list[int]]:
    return [enc.ids for enc in tok.encode_batch(texts, add_special_tokens=False)]


def special_id_set(tok: Tokenizer) -> set[int]:
    return {tok.token_to_id(s) for s in SPECIAL_TOKENS if tok.token_to_id(s) is not None}


def count_pool_support(label: str, tok: Tokenizer) -> dict[str, Any]:
    vocab = tok.get_vocab()
    max_id = max(vocab.values())
    counts = [0] * (max_id + 1)
    by_source: dict[str, dict[str, int]] = collections.defaultdict(lambda: {"rows": 0, "words": 0, "tokens": 0, "unk": 0})
    total_rows = 0
    total_words = 0
    total_tokens = 0
    unk_id = tok.token_to_id("<unk>")
    batch_rows: list[dict[str, Any]] = []
    for row in iter_pool_rows():
        batch_rows.append(row)
        if len(batch_rows) >= BATCH_SIZE:
            total_rows, total_words, total_tokens = consume_pool_batch(tok, batch_rows, counts, by_source, unk_id, total_rows, total_words, total_tokens)
            batch_rows = []
    if batch_rows:
        total_rows, total_words, total_tokens = consume_pool_batch(tok, batch_rows, counts, by_source, unk_id, total_rows, total_words, total_tokens)
    specials = special_id_set(tok)
    used_counts = [c for i, c in enumerate(counts) if i not in specials and c > 0]
    all_nonspecial = [counts[i] if i < len(counts) else 0 for i in range(max_id + 1) if i not in specials]
    threshold_summary = {}
    for th in THRESHOLDS:
        low_ids = [i for i in range(max_id + 1) if i not in specials and counts[i] > 0 and counts[i] < th]
        low_mass = sum(counts[i] for i in low_ids)
        threshold_summary[str(th)] = {
            "used_vocab_below": len(low_ids),
            "used_vocab_below_fraction": len(low_ids) / max(1, len(used_counts)),
            "token_occurrences_below": low_mass,
            "token_occurrence_fraction_below": low_mass / max(1, total_tokens),
        }
    by_source_finish = {
        src: {
            **d,
            "tokens_per_word": d["tokens"] / max(1, d["words"]),
            "unk_fraction": d["unk"] / max(1, d["tokens"]),
        }
        for src, d in sorted(by_source.items())
    }
    return {
        "label": label,
        "vocab_size": tok.get_vocab_size(),
        "max_token_id": max_id,
        "special_ids": {s: tok.token_to_id(s) for s in SPECIAL_TOKENS},
        "rows": total_rows,
        "words": total_words,
        "tokens": total_tokens,
        "tokens_per_word": total_tokens / max(1, total_words),
        "unk_count": counts[unk_id] if unk_id is not None and unk_id < len(counts) else None,
        "used_nonspecial_vocab": len(used_counts),
        "unused_nonspecial_vocab": sum(1 for c in all_nonspecial if c == 0),
        "used_occurrence_quantiles": quantiles([float(x) for x in used_counts]),
        "threshold_summary": threshold_summary,
        "by_source": by_source_finish,
        "counts_by_id": counts,
        "id_to_token": {str(i): t for t, i in vocab.items()},
    }


def consume_pool_batch(tok: Tokenizer, rows: list[dict[str, Any]], counts: list[int], by_source: dict[str, dict[str, int]], unk_id: int | None, total_rows: int, total_words: int, total_tokens: int) -> tuple[int, int, int]:
    encs = encode_batch(tok, [r["text"] for r in rows])
    for row, ids in zip(rows, encs):
        src = row["source"]
        d = by_source[src]
        d["rows"] += 1
        d["words"] += int(row["words"])
        d["tokens"] += len(ids)
        if unk_id is not None:
            unk_here = sum(1 for x in ids if x == unk_id)
            d["unk"] += unk_here
        for x in ids:
            if x >= len(counts):
                counts.extend([0] * (x + 1 - len(counts)))
            counts[x] += 1
        total_rows += 1
        total_words += int(row["words"])
        total_tokens += len(ids)
    return total_rows, total_words, total_tokens


def summarize_new_vs_base(label: str, rec: dict[str, Any], base_vocab: set[str]) -> dict[str, Any]:
    id_to_token = {int(k): v for k, v in rec["id_to_token"].items()}
    counts = rec["counts_by_id"]
    specials = set(rec["special_ids"].values())
    new_counts = []
    new_used_counts = []
    new_ids = []
    for i, tok in id_to_token.items():
        if i in specials or tok in base_vocab:
            continue
        c = counts[i] if i < len(counts) else 0
        new_ids.append(i)
        new_counts.append(c)
        if c > 0:
            new_used_counts.append(c)
    threshold_summary = {}
    for th in THRESHOLDS:
        vals = [c for c in new_used_counts if c < th]
        threshold_summary[str(th)] = {
            "new_used_vocab_below": len(vals),
            "new_used_vocab_below_fraction": len(vals) / max(1, len(new_used_counts)),
            "new_token_occurrences_below": sum(vals),
            "new_token_occurrence_fraction_of_all_tokens": sum(vals) / max(1, rec["tokens"]),
        }
    top_new = sorted(
        [{"token": id_to_token[i], "id": i, "pool_count": counts[i]} for i in new_ids if counts[i] > 0],
        key=lambda x: x["pool_count"], reverse=True
    )[:80]
    rare_new = sorted(
        [{"token": id_to_token[i], "id": i, "pool_count": counts[i]} for i in new_ids if 0 < counts[i] < 20],
        key=lambda x: (x["pool_count"], x["token"])
    )[:80]
    return {
        "label": label,
        "new_vocab_entries_vs_legal16": len(new_ids),
        "new_used_entries_vs_legal16": len(new_used_counts),
        "new_unused_entries_vs_legal16": sum(1 for c in new_counts if c == 0),
        "new_occurrence_mass": sum(new_used_counts),
        "new_occurrence_mass_fraction_of_all_tokens": sum(new_used_counts) / max(1, rec["tokens"]),
        "new_used_occurrence_quantiles": quantiles([float(x) for x in new_used_counts]),
        "threshold_summary": threshold_summary,
        "top_new_tokens_by_pool_count": top_new,
        "rare_new_tokens_pool_count_lt20_examples": rare_new,
    }


def iter_eval_texts() -> Iterable[tuple[str, str]]:
    # BLiMP + Supplement sentence pairs.
    for family, dname in [("BLiMP", "blimp_filtered"), ("Supplement", "supplement_filtered")]:
        d = PRISTINE_FULL / dname
        for p in sorted(d.glob("*.jsonl")):
            for obj in iter_jsonl(p):
                for key in ["sentence_good", "sentence_bad"]:
                    val = obj.get(key)
                    if isinstance(val, str) and val.strip():
                        yield family, val
    # EWoK contexts and targets.
    d = PRISTINE_FULL / "ewok_filtered"
    for p in sorted(d.glob("*.jsonl")):
        for obj in iter_jsonl(p):
            for key in ["Context1", "Target1", "Context2", "Target2"]:
                val = obj.get(key)
                if isinstance(val, str) and val.strip():
                    yield "EWoK", val
    # Entity tracking text fields.
    d = PRISTINE_FULL / "entity_tracking"
    for p in sorted(d.glob("*.jsonl")):
        for obj in iter_jsonl(p):
            prefix = obj.get("input_prefix")
            if isinstance(prefix, str) and prefix.strip():
                yield "Entity", prefix
            opts = obj.get("options")
            if isinstance(opts, list):
                for opt in opts:
                    if isinstance(opt, str) and opt.strip():
                        yield "Entity", (prefix or "") + opt
    # COMPS: generic string fields are acceptable for surface/support measurement.
    d = PRISTINE_FULL / "comps"
    for p in sorted(d.glob("*.jsonl")):
        for obj in iter_jsonl(p):
            fields = [v for k, v in obj.items() if isinstance(v, str) and v.strip() and k not in {"uid", "label"}]
            if fields:
                yield "COMPS", " ".join(fields)
    # GlobalPIQA generated current official files.
    for family, dname in [("GlobalPIQA_parallel", "global_piqa_parallel"), ("GlobalPIQA_nonparallel", "global_piqa_nonparallel")]:
        d = GLOBALPIQA_FULL / dname
        for p in sorted(d.glob("*.jsonl")):
            for obj in iter_jsonl(p):
                fields = [v for k, v in obj.items() if isinstance(v, str) and v.strip() and k not in {"uid", "label"}]
                if fields:
                    yield "GlobalPIQA", " ".join(fields)
    # Reading CSV.
    rp = PRISTINE_FULL / "reading/reading_data.csv"
    if rp.exists():
        with rp.open("r", encoding="utf-8", errors="replace", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                sent = row.get("sentence") or ""
                word = row.get("word") or ""
                if sent.strip():
                    yield "Reading", sent
                if word.strip():
                    yield "Reading", word
    # SuperGLUE all text fields from train/valid/predict jsonl.
    gd = PRISTINE_FULL / "glue_filtered"
    for p in sorted(gd.glob("*.jsonl")):
        for obj in iter_jsonl(p):
            fields = [v for k, v in obj.items() if isinstance(v, str) and v.strip() and k != "label"]
            if fields:
                yield "SuperGLUE", " </s> ".join(fields)


def eval_low_support(tok: Tokenizer, pool_counts: list[int]) -> dict[str, Any]:
    acc: dict[str, dict[str, Any]] = collections.defaultdict(lambda: {
        "texts": 0, "tokens": 0, "unk": 0, "support_values": [],
        **{f"below_{th}": 0 for th in THRESHOLDS},
    })
    unk_id = tok.token_to_id("<unk>")
    batch: list[tuple[str, str]] = []
    for family, text in iter_eval_texts():
        batch.append((family, text))
        if len(batch) >= BATCH_SIZE:
            consume_eval_batch(tok, pool_counts, acc, batch, unk_id)
            batch = []
    if batch:
        consume_eval_batch(tok, pool_counts, acc, batch, unk_id)
    out: dict[str, Any] = {}
    for family, d in sorted(acc.items()):
        toks = d["tokens"]
        out[family] = {
            "texts": d["texts"],
            "tokens": toks,
            "unk": d["unk"],
            "mean_pool_support_per_eval_token": sum(d["support_values"]) / max(1, len(d["support_values"])),
            "support_quantiles_per_eval_token": quantiles([float(x) for x in d["support_values"]]),
            **{f"fraction_eval_tokens_with_pool_count_below_{th}": d[f"below_{th}"] / max(1, toks) for th in THRESHOLDS},
        }
    return out


def consume_eval_batch(tok: Tokenizer, pool_counts: list[int], acc: dict[str, dict[str, Any]], batch: list[tuple[str, str]], unk_id: int | None) -> None:
    encs = encode_batch(tok, [x[1] for x in batch])
    for (family, _text), ids in zip(batch, encs):
        d = acc[family]
        d["texts"] += 1
        d["tokens"] += len(ids)
        for x in ids:
            if unk_id is not None and x == unk_id:
                d["unk"] += 1
            c = pool_counts[x] if x < len(pool_counts) else 0
            d["support_values"].append(c)
            for th in THRESHOLDS:
                if c < th:
                    d[f"below_{th}"] += 1


def strip_heavy(rec: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in rec.items() if k not in {"counts_by_id", "id_to_token"}}


def write_tables(payload: dict[str, Any]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "pool_support_summary.csv").open("w", encoding="utf-8", newline="") as f:
        cols = ["tokenizer", "vocab_size", "tokens_per_word", "used_nonspecial_vocab", "unused_nonspecial_vocab", "median_used_count", "p05_used_count", "p10_used_count"]
        for th in [20, 50, 100, 200]:
            cols.extend([f"vocab_frac_lt{th}", f"token_mass_frac_lt{th}"])
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        for label, rec in payload["pool_support"].items():
            q = rec["used_occurrence_quantiles"]
            row = {
                "tokenizer": label,
                "vocab_size": rec["vocab_size"],
                "tokens_per_word": rec["tokens_per_word"],
                "used_nonspecial_vocab": rec["used_nonspecial_vocab"],
                "unused_nonspecial_vocab": rec["unused_nonspecial_vocab"],
                "median_used_count": q.get("p50"),
                "p05_used_count": q.get("p05"),
                "p10_used_count": q.get("p10"),
            }
            for th in [20, 50, 100, 200]:
                ts = rec["threshold_summary"][str(th)]
                row[f"vocab_frac_lt{th}"] = ts["used_vocab_below_fraction"]
                row[f"token_mass_frac_lt{th}"] = ts["token_occurrence_fraction_below"]
            writer.writerow(row)
    with (OUT / "new_vs_legal16_support_summary.csv").open("w", encoding="utf-8", newline="") as f:
        cols = ["tokenizer", "new_vocab", "new_used", "new_mass_fraction", "new_median_used_count", "new_p05_used_count", "new_p10_used_count"]
        for th in [20, 50, 100, 200]:
            cols.extend([f"new_used_vocab_frac_lt{th}", f"new_mass_frac_lt{th}"])
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        for label, rec in payload["new_vs_legal16"].items():
            q = rec["new_used_occurrence_quantiles"]
            row = {
                "tokenizer": label,
                "new_vocab": rec["new_vocab_entries_vs_legal16"],
                "new_used": rec["new_used_entries_vs_legal16"],
                "new_mass_fraction": rec["new_occurrence_mass_fraction_of_all_tokens"],
                "new_median_used_count": q.get("p50"),
                "new_p05_used_count": q.get("p05"),
                "new_p10_used_count": q.get("p10"),
            }
            for th in [20, 50, 100, 200]:
                ts = rec["threshold_summary"][str(th)]
                row[f"new_used_vocab_frac_lt{th}"] = ts["new_used_vocab_below_fraction"]
                row[f"new_mass_frac_lt{th}"] = ts["new_token_occurrence_fraction_of_all_tokens"]
            writer.writerow(row)
    with (OUT / "eval_family_low_support.csv").open("w", encoding="utf-8", newline="") as f:
        cols = ["tokenizer", "family", "texts", "eval_tokens", "mean_pool_support", "p10_support", "p50_support", "frac_lt20", "frac_lt50", "frac_lt100", "frac_lt200", "unk"]
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        for label, fams in payload["eval_low_support"].items():
            for fam, rec in fams.items():
                q = rec["support_quantiles_per_eval_token"]
                writer.writerow({
                    "tokenizer": label,
                    "family": fam,
                    "texts": rec["texts"],
                    "eval_tokens": rec["tokens"],
                    "mean_pool_support": rec["mean_pool_support_per_eval_token"],
                    "p10_support": q.get("p10"),
                    "p50_support": q.get("p50"),
                    "frac_lt20": rec["fraction_eval_tokens_with_pool_count_below_20"],
                    "frac_lt50": rec["fraction_eval_tokens_with_pool_count_below_50"],
                    "frac_lt100": rec["fraction_eval_tokens_with_pool_count_below_100"],
                    "frac_lt200": rec["fraction_eval_tokens_with_pool_count_below_200"],
                    "unk": rec["unk"],
                })


def write_note(payload: dict[str, Any]) -> None:
    lines: list[str] = []
    lines.append("# research tokenizer support spectrum\n")
    lines.append("\nThis CPU-only evidence asset was built while the legal-40k 100M trainings were running. It does not train or evaluate a language model and does not use official evaluation text to train tokenizers. Support-floored tokenizers are learned only from the exact allowed 10M compact_view_reinvest pool.\n")
    lines.append("\n## Pool support summary\n\n")
    lines.append("| tokenizer | vocab | tok/word | used non-special | median count | p10 count | vocab<50 | mass<50 | vocab<100 | mass<100 |\n")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
    for label, rec in payload["pool_support"].items():
        q = rec["used_occurrence_quantiles"]
        t50 = rec["threshold_summary"]["50"]
        t100 = rec["threshold_summary"]["100"]
        lines.append(
            f"| {label} | {rec['vocab_size']} | {rec['tokens_per_word']:.4f} | {rec['used_nonspecial_vocab']} | "
            f"{q.get('p50'):.1f} | {q.get('p10'):.1f} | {t50['used_vocab_below_fraction']:.3f} | "
            f"{t50['token_occurrence_fraction_below']:.4f} | {t100['used_vocab_below_fraction']:.3f} | {t100['token_occurrence_fraction_below']:.4f} |\n"
        )
    lines.append("\n## New-token support relative to A01 legal16k\n\n")
    lines.append("| tokenizer | new vocab | new used | new mass | median new count | p10 new count | new vocab<50 | new mass<50 | new vocab<100 | new mass<100 |\n")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
    for label, rec in payload["new_vs_legal16"].items():
        q = rec["new_used_occurrence_quantiles"]
        t50 = rec["threshold_summary"]["50"]
        t100 = rec["threshold_summary"]["100"]
        lines.append(
            f"| {label} | {rec['new_vocab_entries_vs_legal16']} | {rec['new_used_entries_vs_legal16']} | "
            f"{rec['new_occurrence_mass_fraction_of_all_tokens']:.4f} | {q.get('p50'):.1f} | {q.get('p10'):.1f} | "
            f"{t50['new_used_vocab_below_fraction']:.3f} | {t50['new_token_occurrence_fraction_of_all_tokens']:.4f} | "
            f"{t100['new_used_vocab_below_fraction']:.3f} | {t100['new_token_occurrence_fraction_of_all_tokens']:.4f} |\n"
        )
    lines.append("\n## Evaluation-family exposure to low-support training tokens\n\n")
    lines.append("| tokenizer | family | eval tokens | mean pool count | p10 | p50 | frac<50 | frac<100 | frac<200 |\n")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|\n")
    focus = ["Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA", "SuperGLUE", "BLiMP", "Reading"]
    for label, fams in payload["eval_low_support"].items():
        for fam in focus:
            rec = fams.get(fam)
            if not rec:
                continue
            q = rec["support_quantiles_per_eval_token"]
            lines.append(
                f"| {label} | {fam} | {rec['tokens']} | {rec['mean_pool_support_per_eval_token']:.1f} | "
                f"{q.get('p10'):.1f} | {q.get('p50'):.1f} | "
                f"{rec['fraction_eval_tokens_with_pool_count_below_50']:.4f} | "
                f"{rec['fraction_eval_tokens_with_pool_count_below_100']:.4f} | "
                f"{rec['fraction_eval_tokens_with_pool_count_below_200']:.4f} |\n"
            )
    lines.append("\n## Scientific use\n\n")
    lines.append("- This file does not settle the 40k route. The running legal-40k models still need full pristine official evaluation.\n")
    lines.append("- If legal 40k improves Supplement/EWoK but loses GlobalPIQA/Entity/COMPS, these support spectra can test whether rare, low-support new units are a plausible failure mode.\n")
    lines.append("- If legal 40k fails broadly, the spectra help decide whether a support-floored tokenizer is scientifically distinct enough to justify a later experiment, or whether the next route should move to masking/sequence curriculum or depth-over-width architecture.\n")
    lines.append("- Evaluation-family support columns are interpretation-only; tokenizer learning used only the allowed 10M pool.\n")
    lines.append(f"\nJSON: `{OUT / 'tokenizer_support_spectrum.json'}`\n")
    lines.append("CSV tables: `pool_support_summary.csv`, `new_vs_legal16_support_summary.csv`, `eval_family_low_support.csv`.\n")
    NOTE.parent.mkdir(parents=True, exist_ok=True)
    NOTE.write_text("".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force-train", action="store_true")
    ap.add_argument("--skip-eval-surface", action="store_true")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    pool_sha = sha256_file(POOL_10M)
    if pool_sha != EXPECTED_POOL_SHA:
        raise RuntimeError({"pool_sha_mismatch": pool_sha, "expected": EXPECTED_POOL_SHA})

    tokenizer_dirs: dict[str, Path] = dict(EXISTING_TOKENIZERS)
    trained_records = []
    for spec in SUPPORT_FLOOR_SPECS:
        td = train_support_floor_tokenizer(spec["label"], int(spec["vocab_size"]), int(spec["min_frequency"]), force=args.force_train)
        tokenizer_dirs[spec["label"]] = td
        meta_path = td / "tokenizer_support_floor_metadata.json"
        if meta_path.exists():
            trained_records.append(json.loads(meta_path.read_text(encoding="utf-8")))

    tokenizers = {label: load_tokenizer(path) for label, path in tokenizer_dirs.items()}
    base_vocab = set(tokenizers["legal_a01_16k"].get_vocab().keys())

    pool_support_heavy: dict[str, dict[str, Any]] = {}
    pool_support_light: dict[str, dict[str, Any]] = {}
    new_vs_legal16: dict[str, dict[str, Any]] = {}
    eval_support: dict[str, dict[str, Any]] = {}
    for label, tok in tokenizers.items():
        rec = count_pool_support(label, tok)
        pool_support_heavy[label] = rec
        pool_support_light[label] = strip_heavy(rec)
        if label != "legal_a01_16k":
            new_vs_legal16[label] = summarize_new_vs_base(label, rec, base_vocab)
        if not args.skip_eval_surface:
            eval_support[label] = eval_low_support(tok, rec["counts_by_id"])

    payload = {
        "status": "TOKENIZER_SUPPORT_SPECTRUM",
        "created_utc": now_utc(),
        "elapsed_sec": None,
        "pool_10m": str(POOL_10M),
        "pool_10m_sha256": pool_sha,
        "training_compliance": "All support-floored tokenizer candidates were trained only on the exact allowed 10M pool; evaluation text is used only for analysis of token support after tokenizer training.",
        "tokenizer_dirs": {label: str(path) for label, path in tokenizer_dirs.items()},
        "support_floor_training_records": trained_records,
        "pool_support": pool_support_light,
        "new_vs_legal16": new_vs_legal16,
        "eval_low_support": eval_support,
        "interpretation_boundary": [
            "Support spectra are mechanistic evidence about representation-unit training density, not downstream scores.",
            "The running legal-40k two-seed full official evaluation remains decisive for the current route.",
            "If 40k fails, do not infer that all representation routes failed; compare low-support exposure and route to support-floor tokenizer, masking/sequence curriculum, or depth-over-width architecture based on the full score pattern.",
        ],
    }
    out_json = OUT / "tokenizer_support_spectrum.json"
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_tables(payload)
    write_note(payload)
    # Mark elapsed after all writes.
    payload["elapsed_sec"] = None
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(json.dumps({
        "status": payload["status"],
        "tokenizers": list(tokenizer_dirs),
        "out_json": str(out_json),
        "note": str(NOTE),
        "tables": [
            str(OUT / "pool_support_summary.csv"),
            str(OUT / "new_vs_legal16_support_summary.csv"),
            str(OUT / "eval_family_low_support.csv"),
        ],
    }, indent=2), flush=True)


if __name__ == "__main__":
    t0 = time.time()
    main()
