#!/usr/bin/env python3
"""research: static quality and scoring-readiness report for the frozen research
structural contrast-margin probe.

This script does not alter the frozen probe.  It checks whether the focused-PLL
scorer can see the intended focus spans under the legal tokenizer, reports length
and focus-token imbalances that may affect margin interpretation, and surfaces
family-specific construction artifacts before the full checkpoint panel is read.
"""
from __future__ import annotations
import json, hashlib, math, re, statistics, time, os
from pathlib import Path
from collections import Counter, defaultdict

# Reuse writable cache pattern for tokenizer loading.
CACHE = Path("experiments/archive/frontier_consolidation/.hf_cache_probe_quality")
os.environ["HF_HOME"] = str(CACHE / "home")
os.environ["HF_MODULES_CACHE"] = str(CACHE / "modules")
os.environ["TRANSFORMERS_CACHE"] = str(CACHE / "transformers")
for d in [CACHE / "home", CACHE / "modules", CACHE / "transformers"]:
    d.mkdir(parents=True, exist_ok=True)

from transformers import AutoTokenizer

ROOT = Path("experiments/archive/frontier_consolidation")
PROBE = ROOT / "data/structural_contrast_probe/structural_contrast_probe.json"
TOKENIZER_CKPT = ROOT / "training/runs/adapter128_scale1p75_h100M100M_seed43022_official_ladder/hf_model/chck_82M"
OUT_DIR = ROOT / "data/structural_probe_quality"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def extract_spans(text: str, spans: list[list[int]]) -> list[str]:
    out = []
    for a, b in spans:
        a = max(0, min(len(text), int(a)))
        b = max(0, min(len(text), int(b)))
        out.append(text[a:b])
    return out


def focus_token_count(tokenizer, text: str, spans: list[list[int]], max_length: int = 256):
    enc = tokenizer(text, return_offsets_mapping=True, add_special_tokens=True,
                    truncation=True, max_length=max_length, return_attention_mask=True)
    offsets = enc["offset_mapping"]
    focus = set()
    for a, b in spans:
        for ti, (ta, tb) in enumerate(offsets):
            if ta < b and tb > a and ta != tb:
                focus.add(ti)
    nonzero_offsets = [(ta, tb) for ta, tb in offsets if ta != tb]
    max_seen = max((tb for ta, tb in nonzero_offsets), default=0)
    truncated_focus = any(b > max_seen for a, b in spans)
    return {
        "n_input_tokens": len(enc["input_ids"]),
        "n_focus_tokens": len(focus),
        "truncated_focus": truncated_focus,
        "max_seen_char": max_seen,
    }


def mean(xs):
    return sum(xs) / len(xs) if xs else None


def q(xs, frac):
    if not xs:
        return None
    xs = sorted(xs)
    return xs[min(len(xs)-1, max(0, int(round(frac * (len(xs)-1)))))]


def summarize_nums(xs):
    if not xs:
        return {"n": 0}
    return {
        "n": len(xs),
        "mean": round(mean(xs), 6),
        "median": round(statistics.median(xs), 6),
        "q10": round(q(xs, 0.10), 6),
        "q90": round(q(xs, 0.90), 6),
        "min": round(min(xs), 6),
        "max": round(max(xs), 6),
    }


def family_quality_flags(pair):
    fam = pair.get("family")
    c = pair.get("coherent_text", "")
    p = pair.get("perturbed_text", "")
    flags = []
    if c == p:
        flags.append("identical_text")
    if len(c) < 15 or len(p) < 15:
        flags.append("very_short")
    if c.count("..") or p.count(".."):
        flags.append("double_period")
    if re.search(r"\s[.,;:]", c) or re.search(r"\s[.,;:]", p):
        flags.append("space_before_punct")
    if fam == "polarity_relation":
        if not re.search(r"\b(no|not|never|without|cannot|can't|n't)\b", c, re.I):
            flags.append("coherent_lacks_negator")
        if re.search(r"\b(no|not|never|without|cannot|can't|n't)\b", p, re.I):
            flags.append("perturbed_keeps_negator")
    if fam == "entity_state":
        if " is now " not in c or " is now " not in p:
            flags.append("entity_missing_final_state_phrase")
        # Synthetic templates sometimes use "puts X to Y"; record, do not invalidate.
        if re.search(r"\bputs?\b[^.]{0,40}\bto\b", c):
            flags.append("entity_put_to_template")
    if fam == "temporal_order":
        if not re.search(r"\b(after|before|then)\b", c, re.I):
            flags.append("temporal_lacks_marker")
        if p[:1].islower():
            flags.append("perturbed_lowercase_start")
    if fam == "directed_relation":
        if not re.search(r"\b(more|less|from|to|into|onto|toward|towards)\b", c, re.I):
            flags.append("directed_lacks_marker")
        if len(c.split()) > 0 and len(p.split()) > 0 and abs(len(c.split()) - len(p.split())) > 5:
            flags.append("directed_length_shift")
    if fam == "belief_report":
        if re.search(r"[*\[\]]", c+p):
            flags.append("childes_markup")
    return flags


def main():
    t0 = time.time()
    probe = json.load(open(PROBE, encoding="utf-8"))
    pairs = probe["pairs"]
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER_CKPT), trust_remote_code=True)

    rows = []
    flag_counts = Counter()
    fam_flag_counts = defaultdict(Counter)
    by_family = defaultdict(list)
    subtype_counts = Counter()
    source_counts = defaultdict(Counter)
    span_text_samples = defaultdict(list)

    for pair in pairs:
        coh = focus_token_count(tokenizer, pair["coherent_text"], pair["focus_spans_coherent"])
        pert = focus_token_count(tokenizer, pair["perturbed_text"], pair["focus_spans_perturbed"])
        flags = family_quality_flags(pair)
        if coh["n_focus_tokens"] == 0:
            flags.append("zero_coherent_focus_tokens")
        if pert["n_focus_tokens"] == 0:
            flags.append("zero_perturbed_focus_tokens")
        if coh["truncated_focus"]:
            flags.append("coherent_focus_truncated")
        if pert["truncated_focus"]:
            flags.append("perturbed_focus_truncated")
        if abs(coh["n_focus_tokens"] - pert["n_focus_tokens"]) >= 3:
            flags.append("focus_token_count_imbalance_ge3")
        if abs(len(pair["coherent_text"].split()) - len(pair["perturbed_text"].split())) >= 4:
            flags.append("word_count_imbalance_ge4")

        fam = pair["family"]
        subtype_counts[(fam, pair.get("sub_type", ""))] += 1
        source_counts[fam][pair.get("corpus_source", "")] += 1
        for fl in flags:
            flag_counts[fl] += 1
            fam_flag_counts[fam][fl] += 1
        row = {
            "pair_id": pair["pair_id"],
            "family": fam,
            "sub_type": pair.get("sub_type", ""),
            "source": pair.get("corpus_source", ""),
            "coherent_words": len(pair["coherent_text"].split()),
            "perturbed_words": len(pair["perturbed_text"].split()),
            "word_delta_abs": abs(len(pair["coherent_text"].split()) - len(pair["perturbed_text"].split())),
            "coherent_tokens": coh["n_input_tokens"],
            "perturbed_tokens": pert["n_input_tokens"],
            "coherent_focus_tokens": coh["n_focus_tokens"],
            "perturbed_focus_tokens": pert["n_focus_tokens"],
            "focus_delta_abs": abs(coh["n_focus_tokens"] - pert["n_focus_tokens"]),
            "flags": flags,
            "coherent_focus_text": extract_spans(pair["coherent_text"], pair["focus_spans_coherent"]),
            "perturbed_focus_text": extract_spans(pair["perturbed_text"], pair["focus_spans_perturbed"]),
        }
        rows.append(row)
        by_family[fam].append(row)
        if len(span_text_samples[fam]) < 5:
            span_text_samples[fam].append({
                "pair_id": pair["pair_id"],
                "coh_focus": row["coherent_focus_text"],
                "pert_focus": row["perturbed_focus_text"],
                "flags": flags,
            })

    fam_summary = {}
    for fam, frs in sorted(by_family.items()):
        fam_summary[fam] = {
            "n_pairs": len(frs),
            "coherent_words": summarize_nums([r["coherent_words"] for r in frs]),
            "perturbed_words": summarize_nums([r["perturbed_words"] for r in frs]),
            "coherent_tokens": summarize_nums([r["coherent_tokens"] for r in frs]),
            "perturbed_tokens": summarize_nums([r["perturbed_tokens"] for r in frs]),
            "coherent_focus_tokens": summarize_nums([r["coherent_focus_tokens"] for r in frs]),
            "perturbed_focus_tokens": summarize_nums([r["perturbed_focus_tokens"] for r in frs]),
            "focus_imbalance_ge3": sum(1 for r in frs if r["focus_delta_abs"] >= 3),
            "zero_focus_pairs": sum(1 for r in frs if r["coherent_focus_tokens"] == 0 or r["perturbed_focus_tokens"] == 0),
            "n_with_any_flag": sum(1 for r in frs if r["flags"]),
            "top_flags": dict(fam_flag_counts[fam].most_common(12)),
        }

    result = {
        "status": "STRUCTURAL_PROBE_STATIC_QUALITY_REPORTED",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "probe_path": str(PROBE),
        "probe_sha256": sha256_file(PROBE),
        "probe_content_sha256": probe.get("content_sha256"),
        "probe_id": probe.get("probe_id"),
        "tokenizer_checkpoint": str(TOKENIZER_CKPT),
        "tokenizer_vocab_size": tokenizer.vocab_size,
        "n_pairs": len(pairs),
        "family_counts": dict(Counter(p["family"] for p in pairs)),
        "subtype_counts": {f"{k[0]}/{k[1]}": v for k, v in sorted(subtype_counts.items())},
        "source_counts_by_family": {fam: dict(cnt) for fam, cnt in source_counts.items()},
        "global_flag_counts": dict(flag_counts.most_common()),
        "family_summary": fam_summary,
        "focus_text_samples": dict(span_text_samples),
        "flagged_examples": rows[:0],
        "elapsed_sec": round(time.time() - t0, 3),
    }
    # Save a bounded set of examples per important flag.
    examples = defaultdict(list)
    for r in rows:
        for fl in r["flags"]:
            if len(examples[fl]) < 8:
                examples[fl].append({
                    "pair_id": r["pair_id"], "family": r["family"], "sub_type": r["sub_type"],
                    "source": r["source"], "coh_focus": r["coherent_focus_text"],
                    "pert_focus": r["perturbed_focus_text"],
                    "coh_focus_tokens": r["coherent_focus_tokens"],
                    "pert_focus_tokens": r["perturbed_focus_tokens"],
                })
    result["flagged_examples_by_flag"] = dict(examples)

    out_json = OUT_DIR / "structural_probe_quality_report.json"
    out_json.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    md = []
    md.append("# research static quality report for structural contrast probe v1")
    md.append("")
    md.append(f"Probe: `{PROBE}`")
    md.append(f"Probe SHA256: `{result['probe_sha256']}`")
    md.append(f"Content SHA256 recorded by builder: `{result['probe_content_sha256']}`")
    md.append(f"Pairs: **{len(pairs)}**; tokenizer vocab: **{tokenizer.vocab_size}**")
    md.append("")
    md.append("## Family readiness summary")
    md.append("")
    md.append("| family | n | mean coh toks | mean pert toks | mean coh focus | mean pert focus | zero focus | focus imbalance >=3 | pairs with flags | top flags |")
    md.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---|")
    for fam, fs in sorted(fam_summary.items()):
        topf = ", ".join(f"{k}:{v}" for k, v in list(fs["top_flags"].items())[:5])
        md.append(
            f"| {fam} | {fs['n_pairs']} | {fs['coherent_tokens'].get('mean')} | {fs['perturbed_tokens'].get('mean')} | "
            f"{fs['coherent_focus_tokens'].get('mean')} | {fs['perturbed_focus_tokens'].get('mean')} | "
            f"{fs['zero_focus_pairs']} | {fs['focus_imbalance_ge3']} | {fs['n_with_any_flag']} | {topf} |"
        )
    md.append("")
    md.append("## Global construction flags")
    md.append("")
    if flag_counts:
        for k, v in flag_counts.most_common():
            md.append(f"- {k}: {v}")
    else:
        md.append("- none")
    md.append("")
    md.append("## Interpretation notes")
    md.append("")
    md.append("- This report does not change the frozen v1 pair set or the frozen scoring weights.")
    md.append("- Zero-focus and truncated-focus counts directly affect whether focused PLL can score a pair; they should remain zero or very small.")
    md.append("- Markup, double punctuation, and template artifacts are interpretation context for the v1 panel, not permission to alter v1 after seeing checkpoint scores.")
    md.append("- The `entity_put_to_template` flag records a known synthetic-template awkwardness; entity-state margins still test stale-location preference versus final-state update.")
    md.append("")
    md.append(f"JSON: `{out_json}`")
    (OUT_DIR / "structural_probe_quality_report.md").write_text("\n".join(md), encoding="utf-8")

    print(json.dumps({
        "status": result["status"],
        "out_json": str(out_json),
        "n_pairs": len(pairs),
        "global_flags": dict(flag_counts.most_common(10)),
        "elapsed_sec": result["elapsed_sec"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
