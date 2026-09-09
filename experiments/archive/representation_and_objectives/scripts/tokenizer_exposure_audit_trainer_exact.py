#!/usr/bin/env python3
"""research trainer-exact tokenizer/masking exposure audit for the seqsafe96 FineWeb
source-breadth contrast.

This supersedes the earlier quick audit because the actual trainer tokenizes with:
  tokenizer(text, add_special_tokens=False, truncation=True, max_length=256,
            padding='max_length')
not add_special_tokens=True. It also builds WWM groups from token word-start markers.

Scientific purpose
------------------
Word-matched arms can differ in the true prediction budget. This audit measures the
exact per-epoch token exposure and the *expected* WWM prediction budget under the
same tokenizer/chunking logic used by `masking_curriculum_trainer.py` for the pending
FineWeb seqsafe96 source-breadth contrast:

  - candidate tokens seen after seq256 truncation
  - truncation losses relative to no-truncation tokenization
  - WWM word-group counts and group size statistics
  - expected masked tokens per epoch at p=0.15 (p * candidate_tokens)
  - expected masked WWM groups per epoch at p=0.15 (p * word_groups)

A positive downstream result is easier to interpret as source-breadth if treatment and
control are matched at this level. A treatment token surplus would be a confound; a
tiny treatment deficit would make positive score movement conservative with respect to
prediction-token exposure.
"""
import json
import pathlib
import sys
from collections import defaultdict
from math import sqrt

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
OUT_JSON = OUT_DIR / "tokenizer_exposure_audit_trainer_exact.json"
NOTE = (STUDY.parents[2] / 'research/notes/representation_and_objectives/tokenizer_exposure_audit_trainer_exact.md')
SEQ_LEN = 256
MASK_PROB = 0.15


def is_word_start(token_str: str) -> bool:
    return token_str.startswith("Ġ") or token_str.startswith("▁")


def empty_stats():
    return {
        "rows": 0,
        "words": 0,
        "tokens_raw_no_special": 0,
        "candidate_tokens_seen": 0,
        "padding_tokens": 0,
        "truncated_tokens": 0,
        "rows_truncated": 0,
        "wwm_groups_seen": 0,
        "max_group_len": 0,
        "sum_group_len_sq": 0,
        "rows_with_zero_groups": 0,
    }


def add_stats(dst, words, n_raw, n_seen, n_trunc, groups):
    dst["rows"] += 1
    dst["words"] += words
    dst["tokens_raw_no_special"] += n_raw
    dst["candidate_tokens_seen"] += n_seen
    dst["padding_tokens"] += max(0, SEQ_LEN - n_seen)
    dst["truncated_tokens"] += n_trunc
    if n_trunc > 0:
        dst["rows_truncated"] += 1
    if not groups:
        dst["rows_with_zero_groups"] += 1
    dst["wwm_groups_seen"] += len(groups)
    if groups:
        mx = max(groups)
        dst["max_group_len"] = max(dst["max_group_len"], mx)
        dst["sum_group_len_sq"] += sum(g*g for g in groups)


def finalize_stats(s):
    words = s["words"] or 1
    rows = s["rows"] or 1
    raw = s["tokens_raw_no_special"] or 1
    seen = s["candidate_tokens_seen"] or 1
    groups = s["wwm_groups_seen"] or 1
    out = dict(s)
    out.update({
        "tokens_per_word_raw_no_special": s["tokens_raw_no_special"] / words,
        "candidate_tokens_per_word_seen": s["candidate_tokens_seen"] / words,
        "padding_tokens_per_row": s["padding_tokens"] / rows,
        "truncation_frac_tokens": s["truncated_tokens"] / raw,
        "truncation_frac_rows": s["rows_truncated"] / rows,
        "wwm_groups_per_word_seen": s["wwm_groups_seen"] / words,
        "tokens_per_wwm_group_seen": s["candidate_tokens_seen"] / groups,
        "group_len_rms": sqrt(s["sum_group_len_sq"] / groups),
        "expected_masked_tokens_per_epoch_p015": MASK_PROB * s["candidate_tokens_seen"],
        "expected_masked_groups_per_epoch_p015": MASK_PROB * s["wwm_groups_seen"],
    })
    return out


def row_token_stats(text: str, tok, special_ids):
    # Raw tokenization with no specials, exactly the trainer's tokenization convention.
    input_ids = tok(text, add_special_tokens=False, truncation=False)["input_ids"]
    n_raw = len(input_ids)
    seen_ids = input_ids[:SEQ_LEN]
    n_seen = len(seen_ids)
    n_trunc = max(0, n_raw - SEQ_LEN)
    # In practice there should be no special tokens because add_special_tokens=False, but
    # keep the candidate definition identical to trainer masking: attention && !special.
    candidate_ids = [tid for tid in seen_ids if int(tid) not in special_ids]
    # Recreate word_group logic from masking_curriculum_trainer.py.
    groups = []
    gid = -1
    current_len = 0
    for i, tid in enumerate(seen_ids):
        tid = int(tid)
        if tid in special_ids:
            continue
        tokstr = str(tok.convert_ids_to_tokens(tid))
        if gid < 0 or is_word_start(tokstr) or i == 0:
            if current_len > 0:
                groups.append(current_len)
            gid += 1
            current_len = 1
        else:
            current_len += 1
    if current_len > 0:
        groups.append(current_len)
    # candidate_ids length should equal n_seen unless tokenizer emitted special IDs in text.
    return n_raw, len(candidate_ids), n_trunc, groups


def audit_arm(path: pathlib.Path, tok):
    total = empty_stats()
    by_source = defaultdict(empty_stats)
    examples_checked = []
    special_ids = set(tok.all_special_ids)
    with path.open("r", encoding="utf-8") as fh:
        for i, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            text = str(r["text"])
            words = int(r.get("words", len(text.split())))
            actual_words = len(text.split())
            if words != actual_words:
                raise RuntimeError(f"word mismatch {path} line {i}: field={words} actual={actual_words}")
            src = str(r.get("source", "unknown"))
            n_raw, n_seen, n_trunc, groups = row_token_stats(text, tok, special_ids)
            add_stats(total, words, n_raw, n_seen, n_trunc, groups)
            add_stats(by_source[src], words, n_raw, n_seen, n_trunc, groups)
            if len(examples_checked) < 3:
                examples_checked.append({
                    "line": i, "source": src, "words": words,
                    "tokens_raw_no_special": n_raw,
                    "candidate_tokens_seen": n_seen,
                    "truncated_tokens": n_trunc,
                    "wwm_groups": len(groups),
                    "first_group_lens": groups[:10],
                    "text_prefix": text[:180],
                })
    return finalize_stats(total), {k: finalize_stats(v) for k, v in by_source.items()}, examples_checked


def rel_delta(a, b):
    return (100.0 * (a - b) / b) if b else 0.0


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for p in (TREAT, CTRL, TOKENIZER):
        if not p.exists():
            print(json.dumps({"error": f"missing path {p}"}))
            sys.exit(1)
    tok = AutoTokenizer.from_pretrained(str(TOKENIZER), use_fast=True)

    treat_agg, treat_src, treat_samples = audit_arm(TREAT, tok)
    ctrl_agg, ctrl_src, ctrl_samples = audit_arm(CTRL, tok)

    deltas = {
        "candidate_tokens_seen_treat_minus_ctrl": treat_agg["candidate_tokens_seen"] - ctrl_agg["candidate_tokens_seen"],
        "candidate_tokens_seen_rel_pct": rel_delta(treat_agg["candidate_tokens_seen"], ctrl_agg["candidate_tokens_seen"]),
        "candidate_tokens_per_word_seen_treat_minus_ctrl": treat_agg["candidate_tokens_per_word_seen"] - ctrl_agg["candidate_tokens_per_word_seen"],
        "truncation_frac_tokens_treat_minus_ctrl": treat_agg["truncation_frac_tokens"] - ctrl_agg["truncation_frac_tokens"],
        "wwm_groups_seen_treat_minus_ctrl": treat_agg["wwm_groups_seen"] - ctrl_agg["wwm_groups_seen"],
        "wwm_groups_seen_rel_pct": rel_delta(treat_agg["wwm_groups_seen"], ctrl_agg["wwm_groups_seen"]),
        "expected_masked_tokens_p015_treat_minus_ctrl": treat_agg["expected_masked_tokens_per_epoch_p015"] - ctrl_agg["expected_masked_tokens_per_epoch_p015"],
        "expected_masked_groups_p015_treat_minus_ctrl": treat_agg["expected_masked_groups_per_epoch_p015"] - ctrl_agg["expected_masked_groups_per_epoch_p015"],
    }

    payload = {
        "status": "TOKENIZER_EXPOSURE_AUDIT_TRAINER_EXACT",
        "supersedes": "experiments/archive/representation_and_objectives/data/tokenizer_exposure_audit/tokenizer_exposure_audit.json",
        "why_supersedes": "Earlier quick audit used add_special_tokens=True; actual trainer uses add_special_tokens=False with seq256 padding/truncation.",
        "tokenizer": str(TOKENIZER),
        "tokenizer_vocab_size": tok.vocab_size,
        "seq_len": SEQ_LEN,
        "mask_prob": MASK_PROB,
        "treatment": {"aggregate": treat_agg, "by_source": treat_src, "samples": treat_samples},
        "control": {"aggregate": ctrl_agg, "by_source": ctrl_src, "samples": ctrl_samples},
        "deltas_treatment_minus_control": deltas,
        "scientific_interpretation": (
            "Under the exact trainer tokenization, the two seqsafe96 arms are materially token-matched if "
            "candidate_tokens_seen_rel_pct is near zero. A positive downstream FineWeb effect cannot be "
            "explained by larger treatment token exposure if this delta is near-zero or negative. WWM expected "
            "masked-token budget is proportional to candidate tokens, while expected masked-group budget follows "
            "WWM group counts and reflects lexical segmentation differences."
        ),
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    def row(label, s):
        return (
            f"| {label} | {s['rows']} | {s['words']} | {s['candidate_tokens_seen']} | "
            f"{s['candidate_tokens_per_word_seen']:.4f} | {s['wwm_groups_seen']} | "
            f"{s['tokens_per_wwm_group_seen']:.4f} | {s['truncation_frac_tokens']:.5f} | "
            f"{s['padding_tokens_per_row']:.2f} |"
        )

    lines = [
        "# research trainer-exact tokenizer/masking exposure audit",
        "",
        "This supersedes the earlier quick exposure note because the trainer uses `add_special_tokens=False`, `max_length=256`, and padding to 256.",
        "",
        f"Tokenizer: `{TOKENIZER}`; vocab={tok.vocab_size}; seq_len={SEQ_LEN}; fixed WWM mask probability={MASK_PROB}.",
        "",
        "## Aggregate exposure per 10M-word epoch",
        "",
        "| arm | rows | words | candidate tokens seen | candidate tok/word | WWM groups | tok/group | token trunc frac | padding tokens/row |",
        "|-----|------|-------|-----------------------|--------------------|------------|-----------|------------------|--------------------|",
        row("treatment FineWeb", treat_agg),
        row("control official", ctrl_agg),
        "",
        "## Deltas (treatment - control)",
        "",
        f"- candidate tokens seen: {deltas['candidate_tokens_seen_treat_minus_ctrl']} ({deltas['candidate_tokens_seen_rel_pct']:.3f}% relative)",
        f"- candidate tok/word: {deltas['candidate_tokens_per_word_seen_treat_minus_ctrl']:.4f}",
        f"- token truncation fraction: {deltas['truncation_frac_tokens_treat_minus_ctrl']:.5f}",
        f"- WWM groups seen: {deltas['wwm_groups_seen_treat_minus_ctrl']} ({deltas['wwm_groups_seen_rel_pct']:.3f}% relative)",
        f"- expected masked tokens/epoch at p=0.15: {deltas['expected_masked_tokens_p015_treat_minus_ctrl']:.1f}",
        f"- expected masked WWM groups/epoch at p=0.15: {deltas['expected_masked_groups_p015_treat_minus_ctrl']:.1f}",
        "",
        "## Source-block exposure",
        "",
        "### Treatment",
        "",
    ]
    for k, s in sorted(treat_src.items(), key=lambda kv: -kv[1]["words"]):
        lines.append(f"- `{k}`: words={s['words']}, candidate_tok/word={s['candidate_tokens_per_word_seen']:.4f}, WWM_groups/word={s['wwm_groups_per_word_seen']:.4f}, trunc_frac={s['truncation_frac_tokens']:.5f}")
    lines += ["", "### Control", ""]
    for k, s in sorted(ctrl_src.items(), key=lambda kv: -kv[1]["words"]):
        lines.append(f"- `{k}`: words={s['words']}, candidate_tok/word={s['candidate_tokens_per_word_seen']:.4f}, WWM_groups/word={s['wwm_groups_per_word_seen']:.4f}, trunc_frac={s['truncation_frac_tokens']:.5f}")
    lines += [
        "",
        "## Interpretation",
        "",
        payload["scientific_interpretation"],
        "",
        f"Full JSON: `{OUT_JSON}`",
    ]
    NOTE.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps({
        "status": payload["status"],
        "out_json": str(OUT_JSON),
        "note": str(NOTE),
        "candidate_tokens_seen_rel_pct": deltas["candidate_tokens_seen_rel_pct"],
        "wwm_groups_seen_rel_pct": deltas["wwm_groups_seen_rel_pct"],
        "treat_candidate_tok_per_word": treat_agg["candidate_tokens_per_word_seen"],
        "ctrl_candidate_tok_per_word": ctrl_agg["candidate_tokens_per_word_seen"],
    }, indent=2))


if __name__ == "__main__":
    main()
