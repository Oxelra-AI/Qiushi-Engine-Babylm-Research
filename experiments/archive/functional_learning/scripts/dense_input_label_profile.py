#!/usr/bin/env python3
"""research: quantify sparse/dense/dense-mask label and input geometry.

Dense unchanged-Qwen focus differs from sparse focus in two coupled ways: target
coverage and masked input geometry.  This CPU script walks the exact 80-update
unchanged-Qwen legal prefix once, tokenizes each Qwen row once, and reproduces the
row-keyed group selection policies without training.  It reports how many
second-view content groups/tokens are labelled, how many are masked, how much
mask-only corruption the dense-mask/sparse-label control would add, and how
copied-from-source vs noncopied view tokens are distributed.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import json
import pathlib
import random
import re
import sys
from collections import Counter
from typing import Any, Dict, Iterable, List

SCRIPT = _public_path('experiments/archive/functional_learning/scripts/dense_input_label_profile.py')
ROOT = _public_path('.')
STUDY = _public_path('experiments/archive/functional_learning')
SCRIPTS = _public_path('experiments/archive/functional_learning/scripts')
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import real_stream_train_weighted as base  # noqa: E402
import corrected_bridge_trainer as bridge  # noqa: E402

OUT = _public_path('experiments/archive/functional_learning/data/dense_input_label_profile')
SPARSE_SUMMARY = _public_path('experiments/archive/functional_learning/data/unchanged_focus_weighted_train/correspondence_focus_weighted/train_summary.json')
DENSE64_SUMMARY = _public_path('experiments/archive/functional_learning/data/unchanged_dense_focus_train/correspondence_focus_weighted/train_summary.json')
DENSE65_SUMMARY = _public_path('experiments/archive/functional_learning/data/dense_focus_rep_seed62065_train/correspondence_focus_weighted/train_summary.json')
WORD_NORM_RE = re.compile(r"[a-z0-9]+(?:['’][a-z0-9]+)?")

POLICIES = {
    "sparse_seed62064": {"train_seed": 62064, "mode": "sparse", "focus_prob": 0.35, "max_label_groups": 16, "max_mask_groups": 16},
    "dense_seed62064": {"train_seed": 62064, "mode": "dense", "focus_prob": 1.0, "max_label_groups": 128, "max_mask_groups": 128},
    "dense_seed62065": {"train_seed": 62065, "mode": "dense", "focus_prob": 1.0, "max_label_groups": 128, "max_mask_groups": 128},
    "densemask_sparselabel_seed62064": {"train_seed": 62064, "mode": "densemask_sparse_label", "focus_prob": 0.35, "max_label_groups": 16, "max_mask_groups": 128},
    "densemask_sparselabel_seed62065": {"train_seed": 62065, "mode": "densemask_sparse_label", "focus_prob": 0.35, "max_label_groups": 16, "max_mask_groups": 128},
}


def rel(path: pathlib.Path | str) -> str:
    try:
        return str(pathlib.Path(path).resolve().relative_to(ROOT))
    except Exception:
        return str(path)


def load_json(path: pathlib.Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def norm_word(w: str) -> str:
    m = WORD_NORM_RE.search((w or "").lower())
    return m.group(0).strip("'’") if m else ""


def source_norm_set(source_text: str) -> set[str]:
    return {norm_word(m.group(0)) for m in WORD_NORM_RE.finditer(source_text or "") if norm_word(m.group(0))}


def locate_groups(row: Dict[str, Any], tok: Dict[str, Any]) -> List[Dict[str, Any]]:
    offsets = tok["offsets"]
    attention_mask = tok["attention_mask"]
    text = str(row.get("text", ""))
    groups: List[Dict[str, Any]] = []
    for seg_i, seg in enumerate(row.get("qwen_pair_segments") or []):
        view_text = str(seg.get("view_text", ""))
        source_text = str(seg.get("source_text", ""))
        vstart = int(seg.get("view_start", -1))
        vend = int(seg.get("view_end", -1))
        if vstart < 0 or vend <= vstart or vend > len(text):
            continue
        if text[vstart:vend] != view_text:
            continue
        source_words = source_norm_set(source_text)
        for a, b, word in base.content_word_spans(view_text):
            pos = base.locate_positions(offsets, vstart + a, vstart + b)
            pos = [int(p) for p in pos if int(attention_mask[p]) == 1]
            if not pos:
                continue
            nw = norm_word(word)
            groups.append({
                "segment_index": seg_i,
                "pair_id": str(seg.get("pair_id", "")),
                "view_kind": seg.get("view_kind"),
                "candidate_kind": seg.get("candidate_kind"),
                "word": word,
                "norm_word": nw,
                "positions": pos,
                "n_tokens": len(pos),
                "copied_binary": "copied" if nw and nw in source_words else "not_copied",
            })
    return groups


def select_groups(groups: List[Dict[str, Any]], row: Dict[str, Any], train_seed: int, focus_prob: float, max_focus_groups_per_row: int) -> List[Dict[str, Any]]:
    rng = random.Random(base.stable_seed("view-focus", int(train_seed), bridge.row_key(row)))
    candidates = list(groups)
    if len(candidates) > int(max_focus_groups_per_row):
        candidates = rng.sample(candidates, int(max_focus_groups_per_row))
    chosen = [g for g in candidates if rng.random() < float(focus_prob)]
    if not chosen and candidates:
        chosen = [rng.choice(candidates)]
    return chosen


def capped_candidates(groups: List[Dict[str, Any]], row: Dict[str, Any], train_seed: int, max_focus_groups_per_row: int) -> List[Dict[str, Any]]:
    rng = random.Random(base.stable_seed("view-focus", int(train_seed), bridge.row_key(row)))
    candidates = list(groups)
    if len(candidates) > int(max_focus_groups_per_row):
        candidates = rng.sample(candidates, int(max_focus_groups_per_row))
    return candidates


def densemask_masks(groups: List[Dict[str, Any]], labels: List[Dict[str, Any]], row: Dict[str, Any], train_seed: int, max_mask_groups: int) -> List[Dict[str, Any]]:
    row_seed = base.stable_seed("view-focus", int(train_seed), bridge.row_key(row))
    masks = list(groups)
    if len(masks) > max_mask_groups:
        mrng = random.Random(base.stable_seed("densemask-mask-cap", row_seed, bridge.row_key(row), max_mask_groups))
        sampled = mrng.sample(masks, max_mask_groups)
        for g in labels:
            if g not in sampled:
                sampled.append(g)
        masks = sampled
    return masks


def positions(groups: Iterable[Dict[str, Any]]) -> set[int]:
    out = set()
    for g in groups:
        out.update(int(p) for p in g.get("positions", []))
    return out


def add_distribution(counter: Counter, groups: Iterable[Dict[str, Any]], prefix: str) -> None:
    for g in groups:
        cb = str(g.get("copied_binary", "unknown"))
        counter[f"{prefix}_groups_{cb}"] += 1
        counter[f"{prefix}_tokens_{cb}"] += int(g.get("n_tokens", 0))
        kind = str(g.get("candidate_kind", "unknown"))
        counter[f"{prefix}_groups_kind::{kind}"] += 1


def init_counter(policy: Dict[str, Any]) -> Counter:
    c = Counter()
    c.update({
        "train_seed": int(policy["train_seed"]),
        "focus_prob_for_labels_x1000000": int(round(float(policy["focus_prob"]) * 1_000_000)),
        "max_label_groups_per_row": int(policy["max_label_groups"]),
        "max_mask_groups_per_row": int(policy["max_mask_groups"]),
    })
    return c


def finalize_profile(name: str, policy: Dict[str, Any], c: Counter, macro_count: int) -> Dict[str, Any]:
    d = dict(c)
    # Convert stored integer focus prob back into an ordinary float field.
    d["focus_prob_for_labels"] = float(d.pop("focus_prob_for_labels_x1000000")) / 1_000_000.0
    d.update({
        "policy": name,
        "mode": policy["mode"],
        "completed_macro_updates_profiled": macro_count,
        "label_to_mask_token_ratio": float(d.get("label_tokens", 0) / d.get("mask_tokens", 1)) if d.get("mask_tokens", 0) else None,
        "label_group_fraction_of_raw": float(d.get("label_groups", 0) / d.get("raw_groups", 1)) if d.get("raw_groups", 0) else None,
        "mask_group_fraction_of_raw": float(d.get("mask_groups", 0) / d.get("raw_groups", 1)) if d.get("raw_groups", 0) else None,
        "candidate_group_fraction_after_label_cap": float(d.get("label_candidate_groups_after_cap", 0) / d.get("raw_groups", 1)) if d.get("raw_groups", 0) else None,
        "label_copied_token_fraction": float(d.get("label_tokens_copied", 0) / d.get("label_tokens", 1)) if d.get("label_tokens", 0) else None,
        "mask_copied_token_fraction": float(d.get("mask_tokens_copied", 0) / d.get("mask_tokens", 1)) if d.get("mask_tokens", 0) else None,
        "mask_only_token_fraction": float(d.get("mask_only_tokens", 0) / d.get("mask_tokens", 1)) if d.get("mask_tokens", 0) else None,
    })
    return d


def profile_once(rows: List[Dict[str, Any]], tokenizer) -> Dict[str, Dict[str, Any]]:
    wgb = bridge.WordGroupBuilder(tokenizer)
    counters = {name: init_counter(pol) for name, pol in POLICIES.items()}
    cursor = 0
    macro_count = 0
    for update_i in range(80):
        words = 0
        current: List[Dict[str, Any]] = []
        while cursor < len(rows) and words < 39533:
            r = rows[cursor]
            current.append(r)
            words += int(r.get("words", base.wc(r.get("text", ""))))
            cursor += 1
        if not current:
            break
        macro_count += 1
        for r in current:
            is_qwen = r.get("source") == "qwen_pair_packed" and bool(r.get("qwen_pair_segments"))
            if not is_qwen:
                continue
            tok = bridge.tokenize_row(r, tokenizer, 512, wgb)
            groups = locate_groups(r, tok)
            raw_pos = positions(groups)
            for name, pol in POLICIES.items():
                c = counters[name]
                c["qwen_rows"] += 1
                c["raw_groups"] += len(groups)
                c["raw_tokens"] += len(raw_pos)
                c["label_candidate_groups_after_cap"] += len(capped_candidates(groups, r, int(pol["train_seed"]), int(pol["max_label_groups"])))
                if len(groups) > int(pol["max_label_groups"]):
                    c["rows_over_label_cap"] += 1
                if len(groups) > int(pol["max_mask_groups"]):
                    c["rows_over_mask_cap"] += 1
                add_distribution(c, groups, "raw")
                if pol["mode"] == "sparse":
                    labels = select_groups(groups, r, int(pol["train_seed"]), float(pol["focus_prob"]), int(pol["max_label_groups"]))
                    masks = labels
                elif pol["mode"] == "dense":
                    labels = select_groups(groups, r, int(pol["train_seed"]), 1.0, int(pol["max_mask_groups"]))
                    masks = labels
                else:
                    labels = select_groups(groups, r, int(pol["train_seed"]), float(pol["focus_prob"]), int(pol["max_label_groups"]))
                    masks = densemask_masks(groups, labels, r, int(pol["train_seed"]), int(pol["max_mask_groups"]))
                label_pos = positions(labels)
                mask_pos = positions(masks)
                c["label_groups"] += len(labels)
                c["label_tokens"] += len(label_pos)
                c["mask_groups"] += len(masks)
                c["mask_tokens"] += len(mask_pos)
                c["mask_only_tokens"] += len(mask_pos - label_pos)
                c["mask_only_groups_lower_bound"] += max(0, len(masks) - len(labels))
                if not labels:
                    c["zero_label_qwen_rows"] += 1
                add_distribution(c, labels, "label")
                add_distribution(c, masks, "mask")
    return {name: finalize_profile(name, pol, counters[name], macro_count) for name, pol in POLICIES.items()}


def validate_against_training_profiles(profiles: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    actuals = {
        "sparse_seed62064": load_json(SPARSE_SUMMARY),
        "dense_seed62064": load_json(DENSE64_SUMMARY),
        "dense_seed62065": load_json(DENSE65_SUMMARY),
    }
    out = {}
    for key, actual in actuals.items():
        if not actual:
            out[key] = {"status": "missing_actual"}
            continue
        prof = profiles.get(key)
        out[key] = {
            "actual_train_summary_path": rel({
                "sparse_seed62064": SPARSE_SUMMARY,
                "dense_seed62064": DENSE64_SUMMARY,
                "dense_seed62065": DENSE65_SUMMARY,
            }[key]),
            "actual_total_focus_targets": actual.get("total_focus_targets"),
            "profile_label_tokens": prof.get("label_tokens"),
            "actual_focus_selected_groups_total": actual.get("focus_selected_groups_total"),
            "profile_label_groups": prof.get("label_groups"),
            "actual_focus_candidate_groups_total": actual.get("focus_candidate_groups_total"),
            "profile_label_candidate_groups_after_cap": prof.get("label_candidate_groups_after_cap"),
            "focus_target_token_diff_profile_minus_actual": int(prof.get("label_tokens", 0)) - int(actual.get("total_focus_targets", 0)),
            "focus_group_diff_profile_minus_actual": int(prof.get("label_groups", 0)) - int(actual.get("focus_selected_groups_total", 0)),
            "candidate_group_diff_profile_minus_actual": int(prof.get("label_candidate_groups_after_cap", 0)) - int(actual.get("focus_candidate_groups_total", 0)),
        }
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows, prefix_info = base.load_prefix(base.DEFAULT_TAIL, 80, 39533)
    tokenizer = bridge.AutoTokenizer.from_pretrained(str(bridge.PARENT_PATH), local_files_only=True, use_fast=True)
    profiles = profile_once(rows, tokenizer)
    validation = validate_against_training_profiles(profiles)
    result = {
        "status": "DENSE_INPUT_LABEL_PROFILE",
        "script": rel(SCRIPT),
        "prefix_info": prefix_info,
        "policies": {
            "sparse": "research sparse labels and masks: focus_prob=0.35, max_focus_groups=16.",
            "dense": "research/075 dense labels and masks: focus_prob=1.0, max_focus_groups=128.",
            "densemask_sparselabel": "Future control: sparse labels but dense masks over all detected/capped second-view content groups.",
        },
        "profiles": profiles,
        "validation_against_actual_train_summaries": validation,
        "interpretation": {
            "frontier_status": "This is a design readout for causal controls; no official benchmark result is inferred.",
            "main_use": "Quantifies whether the future dense-mask/sparse-label arm mainly changes input corruption while keeping sparse supervised coverage, and whether dense extra targets are copied or noncopied from the source.",
        },
    }
    out_json = _public_path('experiments/archive/functional_learning/data/dense_input_label_profile/dense_input_label_profile.json')
    out_md = _public_path('research/documents/functional_learning/data/dense_input_label_profile/dense_input_label_profile.md')
    out_json.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = ["# research dense/sparse input-label profile\n"]
    lines.append(f"Prefix: `{prefix_info['prefix_rows']}` rows, `{prefix_info['prefix_words']}` words, Qwen rows `{prefix_info['qwen_rows']}`, Qwen pair segments `{prefix_info['qwen_pair_segments']}`.\n")
    lines.append("## Policy totals\n")
    for key, p in profiles.items():
        lines.append(f"- {key}: label groups `{p.get('label_groups')}`, label tokens `{p.get('label_tokens')}`, mask groups `{p.get('mask_groups')}`, mask tokens `{p.get('mask_tokens')}`, mask-only tokens `{p.get('mask_only_tokens')}`, label/mask token ratio `{p.get('label_to_mask_token_ratio')}`, label copied-token fraction `{p.get('label_copied_token_fraction')}`, mask copied-token fraction `{p.get('mask_copied_token_fraction')}`.")
    lines.append("")
    lines.append("## Validation against actual completed training summaries\n")
    for key, rec in validation.items():
        lines.append(f"- {key}: target-token diff profile-actual `{rec.get('focus_target_token_diff_profile_minus_actual')}`, selected-group diff `{rec.get('focus_group_diff_profile_minus_actual')}`, candidate-group diff `{rec.get('candidate_group_diff_profile_minus_actual')}`, actual targets `{rec.get('actual_total_focus_targets')}`, profile label tokens `{rec.get('profile_label_tokens')}`.")
    lines.append("")
    lines.append("## Scientific use\n")
    lines.append("The dense-mask/sparse-label control keeps supervised focus coverage near the sparse policy while exposing the model to dense second-view corruption. If full official evaluation retains source-responsive/Entity gains but costs erase the aggregate, this profile specifies a focused repair experiment separating input-side clue removal from dense supervised coverage.")
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "out_json": rel(out_json), "out_md": rel(out_md)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
