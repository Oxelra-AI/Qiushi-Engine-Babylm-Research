#!/usr/bin/env python3
"""research: frequency of byte-alphabet repair tokens for research <unk> spans.

The byte-alphabet tokenizer removes research-tokenizer <unk> events, but some
repair tokens may be absent or rare in the allowed 10M pretraining pool. This
CPU-only analysis supports later interpretation of research vs byte-alphabet
endpoint scores; it does not choose or modify vocabulary.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import hashlib
import importlib.util
import json
import pathlib
import time
from collections import Counter, defaultdict
from typing import Any, Iterable

from tokenizers import Tokenizer


def find_user_root() -> pathlib.Path:
    return _PUBLIC_ROOT

USER_ROOT = find_user_root()
STUDY = USER_ROOT / "experiments/archive/frontier_consolidation"
STRICT_REPO = USER_ROOT / "experiments/archive/representation_and_objectives/data/pristine_official_coordinate/babylm-eval/strict"
EVAL_ROOT = STRICT_REPO / "evaluation_data/full_eval"
GLOBALPIQA_ROOT = USER_ROOT / "experiments/archive/initial_model_studies/repos/babylm-eval/strict/evaluation_data/full_eval"
POOL = STUDY / "data/density_cleanqwen_overlay_medium_riskhard/cleanqwen_fineweb_compact_view_reinvest_10M.jsonl"
TOK = STUDY / "data/compliant_tokenizer/tokenizer.json"
BYTEALPHA_TOK = STUDY / "data/compliant_tokenizer_bytealphabet/tokenizer.json"
OUT_DIR = STUDY / "data/bytealpha_repair_token_frequency"
READ_FILES_PATH = STRICT_REPO / "evaluation_pipeline/sentence_zero_shot/read_files.py"

spec = importlib.util.spec_from_file_location("strict_sentence_read_files", READ_FILES_PATH)
read_mod = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(read_mod)  # type: ignore[arg-type]

ZERO_SPECS = [
    ("BLiMP", "blimp", EVAL_ROOT / "blimp_filtered"),
    ("Supplement", "blimp", EVAL_ROOT / "supplement_filtered"),
    ("EWoK", "ewok", EVAL_ROOT / "ewok_filtered"),
    ("Entity", "entity_tracking", EVAL_ROOT / "entity_tracking"),
    ("COMPS", "comps", EVAL_ROOT / "comps"),
    ("GlobalPIQA_parallel", "global_piqa_parallel", GLOBALPIQA_ROOT / "global_piqa_parallel"),
    ("GlobalPIQA_nonparallel", "global_piqa_nonparallel", GLOBALPIQA_ROOT / "global_piqa_nonparallel"),
]
SUPERGLUE_ROOT = EVAL_ROOT / "glue_filtered"


def sha256_file(p: pathlib.Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load_jsonl(path: pathlib.Path) -> Iterable[tuple[int, dict[str, Any], str]]:
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            raw = line.rstrip("\n")
            if not raw:
                continue
            yield line_no, json.loads(raw), raw


def train_bytealpha_freq(tok: Tokenizer) -> tuple[Counter[str], Counter[int], dict[str, Any]]:
    tok_freq: Counter[str] = Counter()
    id_freq: Counter[int] = Counter()
    n_rows = n_words = n_tokens = 0
    with POOL.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            text = row.get("text", "")
            if not isinstance(text, str):
                continue
            n_rows += 1
            n_words += int(row.get("words", len(text.split())))
            enc = tok.encode(text, add_special_tokens=True)
            n_tokens += len(enc.tokens)
            tok_freq.update(enc.tokens)
            id_freq.update(enc.ids)
    meta = {"pool": str(POOL), "pool_sha256": sha256_file(POOL), "rows": n_rows, "words": n_words, "tokens": n_tokens, "unique_tokens_seen": len(tok_freq)}
    return tok_freq, id_freq, meta


def bytealpha_tokens_for_step35_unk_spans(tok35: Tokenizer, tokb: Tokenizer, text: str) -> list[dict[str, Any]]:
    a = tok35.encode(text, add_special_tokens=True)
    b = tokb.encode(text, add_special_tokens=True)
    out = []
    for i, (ta, (sa, ea)) in enumerate(zip(a.tokens, a.offsets)):
        if ta != "<unk>":
            continue
        repair = []
        for tb, ib, (sb, eb) in zip(b.tokens, b.ids, b.offsets):
            if eb <= sa or sb >= ea:
                continue
            # Ignore special zero-offset wrappers unless they are truly overlapping nonempty spans.
            if eb <= sb:
                continue
            repair.append({"token": tb, "id": ib, "offset": [sb, eb], "surface": text[sb:eb]})
        out.append({"unk_index": i, "unk_offset": [sa, ea], "unk_surface": text[sa:ea], "repair_tokens": repair})
    return out


def update_with_repair(counter: Counter[str], family_stats: Counter, examples: list[dict[str, Any]], train_freq: Counter[str], family: str, task: str, path: pathlib.Path, line_no: int, field: str, text: str, repair_spans: list[dict[str, Any]], *, target: bool | None = None) -> None:
    if not repair_spans:
        return
    family_stats["texts_with_step35_unk"] += 1
    family_stats["unk_spans"] += len(repair_spans)
    if target is True:
        family_stats["target_texts_with_step35_unk"] += 1
        family_stats["target_step35_unk_spans"] += len(repair_spans)
    for span in repair_spans:
        if span.get("unk_surface") == "\n":
            family_stats["newline_unk_spans"] += 1
        for rt in span["repair_tokens"]:
            token = rt["token"]
            counter[token] += 1
            family_stats["repair_token_occurrences"] += 1
            freq = train_freq.get(token, 0)
            if freq == 0:
                family_stats["repair_token_occurrences_train_freq_0"] += 1
            if freq < 10:
                family_stats["repair_token_occurrences_train_freq_lt10"] += 1
            if freq < 100:
                family_stats["repair_token_occurrences_train_freq_lt100"] += 1
    if len(examples) < 16:
        examples.append({
            "family": family, "task": task, "path": str(path), "line": line_no, "field": field, "target_span": target,
            "text_prefix": text[:240].replace("\n", "\\n"),
            "repair_spans": [
                {"unk_surface": s["unk_surface"].replace("\n", "\\n"), "repair_tokens": [{"token": rt["token"], "surface": rt["surface"].replace("\n", "\\n"), "train_freq": train_freq.get(rt["token"], 0)} for rt in s["repair_tokens"]]} for s in repair_spans[:4]
            ],
        })


def completion_indices(text: str, completion: str, offsets: list[tuple[int, int]]) -> set[int]:
    start_char_idx = len(text) - len(completion)
    return {i for i, (start, end) in enumerate(offsets) if end > start_char_idx}


def scan_zero(tok35: Tokenizer, tokb: Tokenizer, train_freq: Counter[str]) -> dict[str, Any]:
    global_counter: Counter[str] = Counter()
    by_family: dict[str, Counter] = defaultdict(Counter)
    by_task: dict[str, Counter] = defaultdict(Counter)
    examples: list[dict[str, Any]] = []
    for family, task, root in ZERO_SPECS:
        if not root.exists():
            continue
        for path in sorted(root.glob("*.jsonl")):
            task_name = path.stem
            for line_no, raw, raw_line in load_jsonl(path):
                for dec in read_mod.decode(raw_line, path, task, False, None):
                    for cand_i, (text, completion) in enumerate(zip(dec["sentences"], dec["completions"])):
                        enc35 = tok35.encode(text, add_special_tokens=True)
                        target_idxs = completion_indices(text, completion, enc35.offsets)
                        all_spans = bytealpha_tokens_for_step35_unk_spans(tok35, tokb, text)
                        if not all_spans:
                            continue
                        target_spans = [s for s in all_spans if s["unk_index"] in target_idxs]
                        context_spans = [s for s in all_spans if s["unk_index"] not in target_idxs]
                        update_with_repair(global_counter, by_family[family], examples, train_freq, family, task_name, path, line_no, f"candidate[{cand_i}]", text, all_spans, target=bool(target_spans))
                        update_with_repair(Counter(), by_task[f"{family}/{task_name}"], [], train_freq, family, task_name, path, line_no, f"candidate[{cand_i}]", text, all_spans, target=bool(target_spans))
                        by_family[family]["target_repair_token_occurrences"] += sum(len(s["repair_tokens"]) for s in target_spans)
                        by_family[family]["context_repair_token_occurrences"] += sum(len(s["repair_tokens"]) for s in context_spans)
                        by_task[f"{family}/{task_name}"]["target_repair_token_occurrences"] += sum(len(s["repair_tokens"]) for s in target_spans)
                        by_task[f"{family}/{task_name}"]["context_repair_token_occurrences"] += sum(len(s["repair_tokens"]) for s in context_spans)
    return {"token_counter": global_counter, "by_family": {k: dict(v) for k, v in by_family.items()}, "by_task": {k: dict(v) for k, v in by_task.items()}, "examples": examples}


def scan_superglue(tok35: Tokenizer, tokb: Tokenizer, train_freq: Counter[str]) -> dict[str, Any]:
    counter: Counter[str] = Counter(); by_split: dict[str, Counter] = defaultdict(Counter); examples=[]
    for path in sorted(SUPERGLUE_ROOT.glob("*.jsonl")):
        parts = path.name.split(".")
        split = f"{parts[0]}.{parts[1] if len(parts)>2 else 'unknown'}"
        for line_no, row, _ in load_jsonl(path):
            for field, text in [(k, v) for k, v in row.items() if isinstance(v, str) and k not in {"label", "idx"}]:
                spans = bytealpha_tokens_for_step35_unk_spans(tok35, tokb, text)
                update_with_repair(counter, by_split[split], examples, train_freq, "SuperGLUE", split, path, line_no, field, text, spans, target=None)
    return {"token_counter": counter, "by_task_split": {k: dict(v) for k,v in by_split.items()}, "examples": examples}


def summarize_tokens(counter: Counter[str], train_freq: Counter[str]) -> dict[str, Any]:
    total = sum(counter.values())
    unique = len(counter)
    zero_types = sorted([t for t in counter if train_freq.get(t, 0) == 0])
    low_types = sorted([t for t in counter if 0 < train_freq.get(t, 0) < 10], key=lambda t: train_freq[t])
    top = []
    for tok, c in counter.most_common(50):
        top.append({"token": tok, "official_repair_occurrences": c, "train_freq": train_freq.get(tok, 0)})
    return {
        "official_repair_token_occurrences": total,
        "unique_repair_tokens": unique,
        "occurrences_with_train_freq_0": sum(c for t, c in counter.items() if train_freq.get(t, 0) == 0),
        "occurrences_with_train_freq_lt10": sum(c for t, c in counter.items() if train_freq.get(t, 0) < 10),
        "occurrences_with_train_freq_lt100": sum(c for t, c in counter.items() if train_freq.get(t, 0) < 100),
        "unique_train_freq_0": len(zero_types),
        "unique_train_freq_lt10": sum(1 for t in counter if train_freq.get(t, 0) < 10),
        "unique_train_freq_lt100": sum(1 for t in counter if train_freq.get(t, 0) < 100),
        "zero_freq_types": zero_types[:100],
        "low_freq_types": [{"token": t, "train_freq": train_freq[t], "official_repair_occurrences": counter[t]} for t in low_types[:100]],
        "top_repair_tokens": top,
    }


def add_family_rates(stats: dict[str, Counter]) -> dict[str, Any]:
    out = {}
    for k, v in stats.items():
        d = dict(v)
        occ = d.get("repair_token_occurrences", 0) or 0
        if occ:
            d["frac_repair_occ_train_freq_0"] = d.get("repair_token_occurrences_train_freq_0", 0) / occ
            d["frac_repair_occ_train_freq_lt10"] = d.get("repair_token_occurrences_train_freq_lt10", 0) / occ
            d["frac_repair_occ_train_freq_lt100"] = d.get("repair_token_occurrences_train_freq_lt100", 0) / occ
        out[k] = d
    return out


def write_md(payload: dict[str, Any], out_md: pathlib.Path) -> None:
    lines = ["# research byte-alphabet repair-token training frequency", ""]
    lines.append("This analysis asks whether byte-alphabet tokens that replace research `<unk>` spans in official inputs were actually seen in the legal 10M pretraining pool. It is interpretation-only and does not alter the frozen tokenizer design.")
    lines.append("")
    lines.append(f"Allowed pool tokens under byte-alphabet tokenizer: {payload['train_pool']['tokens']} over {payload['train_pool']['rows']} rows; unique tokens seen {payload['train_pool']['unique_tokens_seen']}.")
    lines.append("")
    for name in ["zero_shot", "superglue"]:
        s = payload[name]["repair_token_summary"]
        lines.append(f"## {name}")
        lines.append("")
        lines.append(f"Official repair-token occurrences: {s['official_repair_token_occurrences']}; unique repair tokens: {s['unique_repair_tokens']}; occurrences with training frequency 0: {s['occurrences_with_train_freq_0']}; <10: {s['occurrences_with_train_freq_lt10']}; <100: {s['occurrences_with_train_freq_lt100']}.")
        lines.append("")
        lines.append("Top repair tokens:")
        lines.append("")
        lines.append("| token | official occurrences | train frequency |")
        lines.append("|---|---:|---:|")
        for r in s["top_repair_tokens"][:20]:
            lines.append(f"| `{r['token'].replace('`','` `')}` | {r['official_repair_occurrences']} | {r['train_freq']} |")
        lines.append("")
    lines.append("## Zero-shot by family")
    lines.append("")
    lines.append("| family | research unk texts | unk spans | newline spans | repair-token occ | freq0 frac | <10 frac | <100 frac |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for fam, d in sorted(payload["zero_shot"]["by_family"].items()):
        lines.append(f"| {fam} | {d.get('texts_with_step35_unk',0)} | {d.get('unk_spans',0)} | {d.get('newline_unk_spans',0)} | {d.get('repair_token_occurrences',0)} | {d.get('frac_repair_occ_train_freq_0',0):.3f} | {d.get('frac_repair_occ_train_freq_lt10',0):.3f} | {d.get('frac_repair_occ_train_freq_lt100',0):.3f} |")
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append("If the byte-alphabet endpoint differs from the research endpoint, the score movement should be read as a real interaction between legal tokenizer construction and learning. Removing `<unk>` does not guarantee improvement when the replacing byte tokens are rare or absent as positive MLM targets in the 10M pool; conversely, research's `<unk>` token is also unseen as a pretraining target. The official endpoint results remain decisive.")
    lines.append("")
    lines.append(f"Full JSON: `{out_md.with_suffix('.json')}`")
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    t0 = time.time(); OUT_DIR.mkdir(parents=True, exist_ok=True)
    tok35 = Tokenizer.from_file(str(TOK)); tokb = Tokenizer.from_file(str(BYTEALPHA_TOK))
    train_freq, _, train_meta = train_bytealpha_freq(tokb)
    zero = scan_zero(tok35, tokb, train_freq)
    sg = scan_superglue(tok35, tokb, train_freq)
    zero_counter = zero.pop("token_counter")
    sg_counter = sg.pop("token_counter")
    payload = {
        "status": "BYTEALPHA_REPAIR_TOKEN_FREQUENCY",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "purpose": "Count legal-pool frequencies of byte-alphabet tokens that replace research-tokenizer <unk> spans in official inputs; no model inference and no vocabulary changes.",
        "tokenizers": {"research": str(TOK), "sha": sha256_file(TOK), "bytealpha": str(BYTEALPHA_TOK), "bytealpha_sha": sha256_file(BYTEALPHA_TOK)},
        "train_pool": train_meta,
        "zero_shot": {**zero, "by_family": add_family_rates(zero["by_family"]), "by_task": add_family_rates(zero["by_task"]), "repair_token_summary": summarize_tokens(zero_counter, train_freq)},
        "superglue": {**sg, "by_task_split": add_family_rates(sg["by_task_split"]), "repair_token_summary": summarize_tokens(sg_counter, train_freq)},
        "elapsed_sec": round(time.time() - t0, 3),
    }
    out_json = OUT_DIR / "bytealpha_repair_token_frequency.json"
    out_md = OUT_DIR / "bytealpha_repair_token_frequency.md"
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_md(payload, out_md)
    print(json.dumps({
        "status": payload["status"], "out_json": str(out_json), "out_md": str(out_md),
        "train_unique_tokens_seen": train_meta["unique_tokens_seen"],
        "zero_repair_summary": payload["zero_shot"]["repair_token_summary"],
        "superglue_repair_summary": payload["superglue"]["repair_token_summary"],
        "elapsed_sec": payload["elapsed_sec"],
    }, indent=2, ensure_ascii=False)[:4000], flush=True)

if __name__ == "__main__":
    main()
