#!/usr/bin/env python3
"""research: tokenizer-level exposure audit for the seqsafe96 FineWeb source-breadth
contrast.

Scientific purpose
------------------
The independent review verifier (research) flagged as mandatory that word-matched and row-length-matched
arms do NOT guarantee matched *token-level* exposure under the baseline16k tokenizer.
Two arms can have identical whitespace-word budgets and identical row lengths yet differ
in:
  - total non-padding tokens (real gradient signal per epoch)
  - tokens/word (FineWeb tends ~1.44-1.50 vs official ~1.28)
  - truncation loss at seq256 (rows exceeding 256 tokens lose their tail)
  - masked-target token counts under WWM (the actual learned prediction budget)

If the treatment (FineWeb-injected) arm and control (official-filler) arm differ
materially in non-padding tokens or masked targets, the seqsafe96 downstream contrast
is partly a token-budget contrast, not a pure source-breadth contrast. This audit
quantifies that asymmetry so the pending source-breadth result can be interpreted
correctly.

This uses the 10M single-epoch pool (each arm's exact training text, one pass) with the
real baseline16k tokenizer. It does not train and does not touch GPUs.
"""
import json
import pathlib
import sys
from collections import defaultdict

from transformers import AutoTokenizer

STUDY = pathlib.Path("experiments/archive/representation_and_objectives")
DATA_DIR = STUDY / "training/data/cached_fineweb_seqsafe96_candidate"
TREAT = DATA_DIR / "cleanqwen_seqsafe_fineweb_single_doc_10M.jsonl"
CTRL = DATA_DIR / "cleanqwen_official_lengthmatched_seqsafe_control_10M.jsonl"
TOKENIZER = pathlib.Path(
    "experiments/archive/initial_model_studies/training/runs"
    "fullcycle_official_wwm_debertav2_8x480_seed43_100M_b256/hf_model"
)
OUT_DIR = STUDY / "data/tokenizer_exposure_audit"
OUT_JSON = OUT_DIR / "tokenizer_exposure_audit.json"
NOTE = (STUDY.parents[2] / 'research/notes/representation_and_objectives/tokenizer_exposure_audit.md')
SEQ_LEN = 256  # max_seq_length used in training


def audit_arm(path: pathlib.Path, tok, seq_len: int):
    """Tokenize every row once, measure non-padding tokens, truncation, tokens/word.

    Returns aggregate stats and per-source-prefix stats. The special tokens ([CLS]/[SEP])
    are counted because training adds them; truncation at seq_len is applied so we measure
    what the model actually sees.
    """
    total_rows = 0
    total_words = 0
    total_tokens_raw = 0          # tokens before seq_len truncation (with specials)
    total_tokens_seen = 0         # tokens after seq_len truncation (what model sees)
    total_truncated_tokens = 0    # tokens lost to truncation
    rows_truncated = 0
    by_source = defaultdict(lambda: {
        "rows": 0, "words": 0, "tokens_raw": 0, "tokens_seen": 0,
        "truncated_tokens": 0, "rows_truncated": 0,
    })

    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            text = str(r.get("text", ""))
            words = int(r.get("words", len(text.split())))
            source = str(r.get("source", "unknown"))
            # source prefix grouping: collapse the seqsafe fineweb / official prefixes
            src_key = source

            enc = tok(text, add_special_tokens=True, truncation=False)
            n_raw = len(enc["input_ids"])
            n_seen = min(n_raw, seq_len)
            n_trunc = max(0, n_raw - seq_len)

            total_rows += 1
            total_words += words
            total_tokens_raw += n_raw
            total_tokens_seen += n_seen
            total_truncated_tokens += n_trunc
            if n_trunc > 0:
                rows_truncated += 1

            b = by_source[src_key]
            b["rows"] += 1
            b["words"] += words
            b["tokens_raw"] += n_raw
            b["tokens_seen"] += n_seen
            b["truncated_tokens"] += n_trunc
            if n_trunc > 0:
                b["rows_truncated"] += 1

    agg = {
        "rows": total_rows,
        "words": total_words,
        "tokens_raw": total_tokens_raw,
        "tokens_seen": total_tokens_seen,
        "truncated_tokens": total_truncated_tokens,
        "rows_truncated": rows_truncated,
        "tokens_per_word_raw": (total_tokens_raw / total_words) if total_words else 0.0,
        "tokens_per_word_seen": (total_tokens_seen / total_words) if total_words else 0.0,
        "truncation_frac_tokens": (total_truncated_tokens / total_tokens_raw) if total_tokens_raw else 0.0,
        "truncation_frac_rows": (rows_truncated / total_rows) if total_rows else 0.0,
    }
    # finalize per-source ratios
    per_source = {}
    for k, b in by_source.items():
        per_source[k] = dict(b)
        per_source[k]["tokens_per_word_raw"] = (b["tokens_raw"] / b["words"]) if b["words"] else 0.0
        per_source[k]["tokens_per_word_seen"] = (b["tokens_seen"] / b["words"]) if b["words"] else 0.0
    return agg, per_source


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for p in (TREAT, CTRL, TOKENIZER):
        if not p.exists():
            print(json.dumps({"error": f"missing path {p}"}))
            sys.exit(1)

    tok = AutoTokenizer.from_pretrained(str(TOKENIZER))

    treat_agg, treat_src = audit_arm(TREAT, tok, SEQ_LEN)
    ctrl_agg, ctrl_src = audit_arm(CTRL, tok, SEQ_LEN)

    # Deltas that matter for interpretation
    delta = {
        "tokens_seen_treat_minus_ctrl": treat_agg["tokens_seen"] - ctrl_agg["tokens_seen"],
        "tokens_seen_rel_pct": (
            100.0 * (treat_agg["tokens_seen"] - ctrl_agg["tokens_seen"]) / ctrl_agg["tokens_seen"]
            if ctrl_agg["tokens_seen"] else 0.0
        ),
        "tokens_per_word_seen_treat_minus_ctrl": (
            treat_agg["tokens_per_word_seen"] - ctrl_agg["tokens_per_word_seen"]
        ),
        "truncation_frac_tokens_treat_minus_ctrl": (
            treat_agg["truncation_frac_tokens"] - ctrl_agg["truncation_frac_tokens"]
        ),
    }

    # Identify the changed block source keys (FineWeb replacement vs official filler)
    fineweb_keys = [k for k in treat_src if "fineweb" in k.lower()]
    official_keys = [k for k in ctrl_src if "official" in k.lower() or "lengthmatched" in k.lower()]

    payload = {
        "status": "TOKENIZER_EXPOSURE_AUDIT",
        "tokenizer": str(TOKENIZER),
        "seq_len": SEQ_LEN,
        "treatment": {"agg": treat_agg, "by_source": treat_src},
        "control": {"agg": ctrl_agg, "by_source": ctrl_src},
        "delta": delta,
        "changed_block_source_keys": {
            "fineweb_keys": fineweb_keys,
            "official_filler_keys": official_keys,
        },
        "interpretation": (
            "If tokens_seen_rel_pct is small (|.|<~1%) and tokens_per_word delta is small, "
            "the seqsafe96 arms are token-matched and the downstream contrast is a clean "
            "source-breadth contrast. A large positive treatment token surplus would mean "
            "FineWeb injection also increased the real gradient/prediction budget, "
            "confounding source breadth with token exposure."
        ),
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False))

    lines = [
        "# research tokenizer-level exposure audit (seqsafe96 FineWeb source-breadth)",
        "",
        f"Tokenizer: `{TOKENIZER}` (baseline16k, vocab {tok.vocab_size}), seq_len={SEQ_LEN}.",
        "",
        "## Aggregate per arm (10M single-epoch pool)",
        "",
        "| arm | rows | words | tokens_seen | tok/word (seen) | trunc frac (tokens) | trunc frac (rows) |",
        "|-----|------|-------|-------------|-----------------|---------------------|-------------------|",
        (f"| treatment (FineWeb) | {treat_agg['rows']} | {treat_agg['words']} | "
         f"{treat_agg['tokens_seen']} | {treat_agg['tokens_per_word_seen']:.4f} | "
         f"{treat_agg['truncation_frac_tokens']:.5f} | {treat_agg['truncation_frac_rows']:.5f} |"),
        (f"| control (official) | {ctrl_agg['rows']} | {ctrl_agg['words']} | "
         f"{ctrl_agg['tokens_seen']} | {ctrl_agg['tokens_per_word_seen']:.4f} | "
         f"{ctrl_agg['truncation_frac_tokens']:.5f} | {ctrl_agg['truncation_frac_rows']:.5f} |"),
        "",
        "## Key deltas (treatment - control)",
        "",
        f"- tokens_seen delta: {delta['tokens_seen_treat_minus_ctrl']} "
        f"({delta['tokens_seen_rel_pct']:.3f}% relative)",
        f"- tokens/word (seen) delta: {delta['tokens_per_word_seen_treat_minus_ctrl']:.4f}",
        f"- truncation frac (tokens) delta: {delta['truncation_frac_tokens_treat_minus_ctrl']:.5f}",
        "",
        "## Interpretation",
        "",
        payload["interpretation"],
        "",
        f"JSON: `{OUT_JSON}`",
    ]
    NOTE.write_text("\n".join(lines))
    print(json.dumps({
        "status": payload["status"],
        "out_json": str(OUT_JSON),
        "note": str(NOTE),
        "tokens_seen_rel_pct": delta["tokens_seen_rel_pct"],
        "tokens_per_word_seen_treat": treat_agg["tokens_per_word_seen"],
        "tokens_per_word_seen_ctrl": ctrl_agg["tokens_per_word_seen"],
    }, indent=2))


if __name__ == "__main__":
    main()
