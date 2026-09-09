#!/usr/bin/env python3
"""research: mirrored four-row relation-fact rows with assignment-position variation.

This is the research relation-fact family with one repair before training:
for k stranger operations, the true assignment operation is placed at a balanced
position among the k+1 operations.  The position is identical for C_A and C_B
inside a quad, so every context still appears once with an updated answer and
once with a retained answer; first-operation, last-operation, k, frame, query
entity, candidate presence, and context length are therefore not sufficient
answer-role predictors.
"""

from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = _public_path('.')
MAPS_PATH = _public_path('experiments/archive/functional_learning/data/assignment_reversal_export/a01_assignment_reversal_operation_maps.jsonl')
OUT_DIR = _public_path('experiments/archive/relation_learning/data/mirrored_four_row_family_position_var')
SEED = 92092
K_VALUES = [0, 1, 2, 3, 4]

FRAMES = {
    "death_place": [
        ("f00_dp", "train_seen", "The place where {entity} died is ", "."),
        ("f01_dp", "train_seen", "{entity} died in ", "."),
        ("f02_dp", "train_seen", "The death place of {entity} is ", "."),
        ("f03_dp", "train_seen", "Where {entity} passed away is ", "."),
        ("f04_dp", "train_seen", "The location of {entity}'s death is ", "."),
        ("f05_dp", "eval_unseen", "{entity}'s place of death is ", "."),
        ("f06_dp", "eval_unseen", "According to records, {entity} died in ", "."),
        ("f07_dp", "eval_unseen", "The city where {entity} died is ", "."),
    ],
    "birthplace": [
        ("f00_bp", "train_seen", "The birthplace of {entity} is ", "."),
        ("f01_bp", "train_seen", "{entity} was born in ", "."),
        ("f02_bp", "train_seen", "The place where {entity} was born is ", "."),
        ("f03_bp", "train_seen", "Where {entity} was born is ", "."),
        ("f04_bp", "train_seen", "The birth location of {entity} is ", "."),
        ("f05_bp", "eval_unseen", "{entity}'s place of birth is ", "."),
        ("f06_bp", "eval_unseen", "According to records, {entity} was born in ", "."),
        ("f07_bp", "eval_unseen", "The city where {entity} was born is ", "."),
    ],
    "founded_year": [
        ("f00_fy", "train_seen", "{entity} was founded in ", "."),
        ("f01_fy", "train_seen", "The founding year of {entity} is ", "."),
        ("f02_fy", "train_seen", "{entity} was established in ", "."),
        ("f03_fy", "train_seen", "The year {entity} was founded is ", "."),
        ("f04_fy", "train_seen", "The year of {entity}'s creation is ", "."),
        ("f05_fy", "eval_unseen", "According to records, {entity} was established in ", "."),
        ("f06_fy", "eval_unseen", "{entity} started in ", "."),
        ("f07_fy", "eval_unseen", "The origin year of {entity} is ", "."),
    ],
}


def whole_word_in(text: str, word: str) -> bool:
    return bool(re.search(r"\b" + re.escape(word) + r"\b", text))


def validate(m: dict[str, Any]) -> list[str]:
    issues: list[str] = []
    sc = m["source_context"]
    if not whole_word_in(sc, m["entity_a"]):
        issues.append(f"entity_a '{m['entity_a']}' not whole-word in source")
    if not whole_word_in(sc, m["entity_b"]):
        issues.append(f"entity_b '{m['entity_b']}' not whole-word in source")
    if m["value_a"] not in sc:
        issues.append("value_a not in source")
    if m["value_b"] not in sc:
        issues.append("value_b not in source")
    if m["shared_new_value"] in sc:
        issues.append("shared_new already in source")
    ua, ub = m["update_a_sentence"], m["update_b_sentence"]
    if m["entity_a"] not in ua:
        issues.append("entity_a not in update_a")
    if m["entity_b"] in ua:
        issues.append("entity_b leaks into update_a")
    if m["entity_b"] not in ub:
        issues.append("entity_b not in update_b")
    if m["entity_a"] in ub:
        issues.append("entity_a leaks into update_b")
    if m["shared_new_value"] not in ua or m["shared_new_value"] not in ub:
        issues.append("shared_new missing from an update")
    if len({m["value_a"], m["value_b"], m["shared_new_value"]}) < 3:
        issues.append("duplicate current values")
    return issues


def choose_strangers(pool: list[dict[str, str]], current: dict[str, Any], k: int, rng: random.Random) -> list[dict[str, str]]:
    if k == 0:
        return []
    cur_ents = {current["entity_a"], current["entity_b"]}
    cur_vals = {current["value_a"], current["value_b"], current["shared_new_value"]}
    eligible = [s for s in pool
                if s["map_id"] != current["base_pair_id"]
                and s["entity"] not in cur_ents
                and s["value"] not in cur_vals]
    rng.shuffle(eligible)
    out: list[dict[str, str]] = []
    used_values: set[str] = set(cur_vals)
    used_entities: set[str] = set(cur_ents)
    for s in eligible:
        if s["value"] in used_values or s["entity"] in used_entities:
            continue
        out.append(s)
        used_values.add(s["value"])
        used_entities.add(s["entity"])
        if len(out) == k:
            break
    if len(out) != k:
        raise RuntimeError(f"not enough unique strangers for {current['base_pair_id']} k={k}")
    return out


def make_context(source: str, assignment_sentence: str, stranger_sentences: list[str], assignment_pos: int) -> tuple[str, list[dict[str, Any]]]:
    operations: list[dict[str, Any]] = []
    si = 0
    for pos in range(len(stranger_sentences) + 1):
        if pos == assignment_pos:
            operations.append({"kind": "assignment", "sentence": assignment_sentence})
        else:
            operations.append({"kind": "stranger", "sentence": stranger_sentences[si]})
            si += 1
    ctx = source + " " + " ".join(op["sentence"] for op in operations)
    return ctx, operations


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    maps = [json.loads(line) for line in MAPS_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
    valid: list[dict[str, Any]] = []
    skipped: list[tuple[str, list[str]]] = []
    for m in maps:
        issues = validate(m)
        if issues:
            skipped.append((m["base_pair_id"], issues))
        else:
            valid.append(m)

    pool: list[dict[str, str]] = []
    for m in valid:
        pool.append({"entity": m["entity_a"], "value": m["shared_new_value"],
                     "sentence": m["update_a_sentence"], "map_id": m["base_pair_id"]})
        pool.append({"entity": m["entity_b"], "value": m["shared_new_value"],
                     "sentence": m["update_b_sentence"], "map_id": m["base_pair_id"]})

    rng = random.Random(SEED)
    split_k_counts: dict[tuple[str, int], int] = defaultdict(int)
    all_rows: list[dict[str, Any]] = []
    all_quads: list[dict[str, Any]] = []

    for m in valid:
        rel = m["relation"]
        frames = FRAMES.get(rel, [])
        if not frames:
            continue
        mid = m["base_pair_id"]
        for k in K_VALUES:
            # Balanced, deterministic assignment-position variation within each split and k.
            idx = split_k_counts[(m["split"], k)]
            split_k_counts[(m["split"], k)] += 1
            assignment_pos = idx % (k + 1)

            strangers = choose_strangers(pool, m, k, rng)
            stranger_sents = [s["sentence"] for s in strangers]
            ca, ops_a = make_context(m["source_context"], m["update_a_sentence"], stranger_sents, assignment_pos)
            cb, ops_b = make_context(m["source_context"], m["update_b_sentence"], stranger_sents, assignment_pos)
            all_present = sorted({m["value_a"], m["value_b"], m["shared_new_value"], *[s["value"] for s in strangers]})

            for fid, fsplit, ftpl, fsuf in frames:
                qid = f"{mid}:k{k}:pos{assignment_pos}:{fid}"
                qp_a = ftpl.replace("{entity}", m["entity_a"])
                qp_b = ftpl.replace("{entity}", m["entity_b"])

                def mk(rid: str, ctx: str, qp: str, queried: str, partner: str, ans: str,
                       role: str, svq: str, svp: str, variant: str, recipient: str,
                       ops: list[dict[str, Any]]) -> dict[str, Any]:
                    row_text = ctx + " " + qp + ans + fsuf
                    answer_start = len(ctx) + 1 + len(qp)
                    answer_end = answer_start + len(ans)
                    if row_text[answer_start:answer_end] != ans:
                        raise AssertionError("answer span mismatch")
                    return {
                        "row_id": rid,
                        "quad_id": qid,
                        "split": m["split"],
                        "map_id": mid,
                        "relation": rel,
                        "k": k,
                        "assignment_position": assignment_pos,
                        "frame_id": fid,
                        "frame_split": fsplit,
                        "context_variant": variant,
                        "entity_queried": queried,
                        "entity_partner": partner,
                        "answer": ans,
                        "answer_role": role,
                        "source_value_queried": svq,
                        "source_value_partner": svp,
                        "shared_new_value": m["shared_new_value"],
                        "update_recipient": recipient,
                        "stranger_entities": [s["entity"] for s in strangers],
                        "stranger_values": [s["value"] for s in strangers],
                        "all_present_values": all_present,
                        "operation_sequence": ops,
                        "context_text": ctx,
                        "query_prefix": qp,
                        "answer_text": ans,
                        "answer_suffix": fsuf,
                        "row_text": row_text,
                        "answer_char_start": answer_start,
                        "answer_char_end": answer_end,
                        "row_words": len(row_text.split()),
                        "context_words": len(ctx.split()),
                    }

                r1 = mk(f"{m['split']}:{mid}:k{k}:pos{assignment_pos}:ca:ask_a:{fid}", ca, qp_a,
                        m["entity_a"], m["entity_b"], m["shared_new_value"], "updated",
                        m["value_a"], m["value_b"], "ca", m["entity_a"], ops_a)
                r2 = mk(f"{m['split']}:{mid}:k{k}:pos{assignment_pos}:ca:ask_b:{fid}", ca, qp_b,
                        m["entity_b"], m["entity_a"], m["value_b"], "retained",
                        m["value_b"], m["value_a"], "ca", m["entity_a"], ops_a)
                r3 = mk(f"{m['split']}:{mid}:k{k}:pos{assignment_pos}:cb:ask_a:{fid}", cb, qp_a,
                        m["entity_a"], m["entity_b"], m["value_a"], "retained",
                        m["value_a"], m["value_b"], "cb", m["entity_b"], ops_b)
                r4 = mk(f"{m['split']}:{mid}:k{k}:pos{assignment_pos}:cb:ask_b:{fid}", cb, qp_b,
                        m["entity_b"], m["entity_a"], m["shared_new_value"], "updated",
                        m["value_b"], m["value_a"], "cb", m["entity_b"], ops_b)
                all_rows.extend([r1, r2, r3, r4])
                all_quads.append({
                    "quad_id": qid,
                    "map_id": mid,
                    "split": m["split"],
                    "relation": rel,
                    "k": k,
                    "assignment_position": assignment_pos,
                    "frame_id": fid,
                    "frame_split": fsplit,
                    "rows": {"ca_ask_a": r1["row_id"], "ca_ask_b": r2["row_id"],
                             "cb_ask_a": r3["row_id"], "cb_ask_b": r4["row_id"]},
                    "shared_context_pairs": [[r1["row_id"], r2["row_id"]], [r3["row_id"], r4["row_id"]]],
                    "shared_query_pairs": [[r1["row_id"], r3["row_id"]], [r2["row_id"], r4["row_id"]]],
                })

    def save_jsonl(data: list[dict[str, Any]], path: Path) -> None:
        with path.open("w", encoding="utf-8") as f:
            for d in data:
                f.write(json.dumps(d, ensure_ascii=False) + "\n")

    train_seen = [r for r in all_rows if r["split"] == "train" and r["frame_split"] == "train_seen"]
    train_all = [r for r in all_rows if r["split"] == "train"]
    heldout_seen = [r for r in all_rows if r["split"] == "heldout" and r["frame_split"] == "train_seen"]
    heldout_all = [r for r in all_rows if r["split"] == "heldout"]
    q_train_seen = [q for q in all_quads if q["split"] == "train" and q["frame_split"] == "train_seen"]
    q_held_seen = [q for q in all_quads if q["split"] == "heldout" and q["frame_split"] == "train_seen"]
    q_held_all = [q for q in all_quads if q["split"] == "heldout"]

    save_jsonl(train_seen, _public_path('experiments/archive/relation_learning/data/mirrored_four_row_family_position_var/mirrored_train_frame_seen.jsonl'))
    save_jsonl(train_all, _public_path('experiments/archive/relation_learning/data/mirrored_four_row_family_position_var/mirrored_train_frame_all.jsonl'))
    save_jsonl(heldout_seen, _public_path('experiments/archive/relation_learning/data/mirrored_four_row_family_position_var/mirrored_heldout_frame_seen.jsonl'))
    save_jsonl(heldout_all, _public_path('experiments/archive/relation_learning/data/mirrored_four_row_family_position_var/mirrored_heldout_frame_all.jsonl'))
    save_jsonl(q_train_seen, _public_path('experiments/archive/relation_learning/data/mirrored_four_row_family_position_var/binding_quads_train_frame_seen.jsonl'))
    save_jsonl(q_held_seen, _public_path('experiments/archive/relation_learning/data/mirrored_four_row_family_position_var/binding_quads_heldout_frame_seen.jsonl'))
    save_jsonl(q_held_all, _public_path('experiments/archive/relation_learning/data/mirrored_four_row_family_position_var/binding_quads_heldout_frame_all.jsonl'))

    def summarize(rows: list[dict[str, Any]], label: str) -> dict[str, Any]:
        by_k = defaultdict(lambda: {"n": 0, "updated": 0})
        by_pos = defaultdict(lambda: {"n": 0, "updated": 0})
        ctx_group = defaultdict(list)
        query_group = defaultdict(dict)
        for r in rows:
            by_k[r["k"]]["n"] += 1
            by_k[r["k"]]["updated"] += int(r["answer_role"] == "updated")
            by_pos[(r["k"], r["assignment_position"] )]["n"] += 1
            by_pos[(r["k"], r["assignment_position"] )]["updated"] += int(r["answer_role"] == "updated")
            ctx_group[(r["quad_id"], r["context_variant"])].append(r["context_words"])
            query_group[(r["quad_id"], r["entity_queried"] )][r["context_variant"]] = r["context_words"]
        ca_cb = [v["ca"] - v["cb"] for v in query_group.values() if "ca" in v and "cb" in v]
        upd_words = [len(r["answer"].split()) for r in rows if r["answer_role"] == "updated"]
        ret_words = [len(r["answer"].split()) for r in rows if r["answer_role"] == "retained"]
        return {
            "label": label,
            "n": len(rows),
            "words": sum(r["row_words"] for r in rows),
            "P_updated": sum(1 for r in rows if r["answer_role"] == "updated") / len(rows) if rows else 0,
            "by_k": {str(k): {"n": v["n"], "P_updated": v["updated"] / v["n"]}
                     for k, v in sorted(by_k.items())},
            "by_k_assignment_position": {f"k{k}_pos{p}": {"n": v["n"], "P_updated": v["updated"] / v["n"]}
                                         for (k, p), v in sorted(by_pos.items())},
            "shared_context_word_mismatches": sum(1 for vals in ctx_group.values() if len(set(vals)) > 1),
            "ca_cb_context_word_delta_mean": sum(ca_cb) / len(ca_cb) if ca_cb else 0,
            "ca_cb_context_word_delta_min": min(ca_cb) if ca_cb else 0,
            "ca_cb_context_word_delta_max": max(ca_cb) if ca_cb else 0,
            "answer_words_updated_mean": sum(upd_words) / len(upd_words) if upd_words else 0,
            "answer_words_retained_mean": sum(ret_words) / len(ret_words) if ret_words else 0,
        }

    f_train = summarize(train_seen, "train_frame_seen")
    f_held = summarize(heldout_all, "heldout_frame_all")
    manifest = {
        "status": "MIRRORED_FOUR_ROW_POSITION_VARIED",
        "source": str(MAPS_PATH.relative_to(ROOT)),
        "valid_maps": len(valid),
        "total_maps": len(maps),
        "skipped": skipped,
        "seed": SEED,
        "k_values": K_VALUES,
        "relations": dict(Counter(m["relation"] for m in valid)),
        "counts": {
            "train_frame_seen": {"rows": len(train_seen), "quads": len(q_train_seen), "words": f_train["words"]},
            "train_frame_all": {"rows": len(train_all)},
            "heldout_frame_seen": {"rows": len(heldout_seen), "quads": len(q_held_seen)},
            "heldout_frame_all": {"rows": len(heldout_all), "quads": len(q_held_all), "words": f_held["words"]},
        },
        "features_train": f_train,
        "features_heldout": f_held,
    }
    (_public_path('experiments/archive/relation_learning/data/mirrored_four_row_family_position_var/manifest.json')).write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        "# research mirrored four-row family with assignment-position variation",
        "",
        f"Valid maps: {len(valid)}/{len(maps)}",
        f"Relations: {dict(Counter(m['relation'] for m in valid))}",
        "",
        "The real assignment update is inserted at a balanced position among the k+1 operation sentences, with the same position for C_A and C_B inside each quad.",
        "Each context still appears once with an updated answer and once with a retained answer; each query entity appears in both roles; k and assignment position have P(updated)=0.5.",
        "",
    ]
    for stat in [f_train, f_held]:
        lines.append(f"## {stat['label']}: {stat['n']} rows, {stat['words']} words")
        lines.append(f"P(updated) = {stat['P_updated']:.3f}")
        lines.append(f"Shared-context word mismatches: {stat['shared_context_word_mismatches']}")
        lines.append("By k:")
        for k, v in stat["by_k"].items():
            lines.append(f"  k={k}: n={v['n']}, P(updated)={v['P_updated']:.3f}")
        lines.append("Assignment-position cells:")
        for cell, v in stat["by_k_assignment_position"].items():
            lines.append(f"  {cell}: n={v['n']}, P(updated)={v['P_updated']:.3f}")
        lines.append(f"C_A-C_B context word delta mean={stat['ca_cb_context_word_delta_mean']:.3f}, range=[{stat['ca_cb_context_word_delta_min']}, {stat['ca_cb_context_word_delta_max']}]")
        lines.append(f"Answer words updated={stat['answer_words_updated_mean']:.3f}, retained={stat['answer_words_retained_mean']:.3f}")
        lines.append("")
    if skipped:
        lines.append("## Skipped maps")
        for mid, issues in skipped:
            lines.append(f"  {mid}: {'; '.join(issues)}")
    (_public_path('research/documents/relation_learning/data/mirrored_four_row_family_position_var/summary.md')).write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps({
        "status": manifest["status"],
        "valid_maps": len(valid),
        "counts": manifest["counts"],
        "train_P_updated": f_train["P_updated"],
        "heldout_P_updated": f_held["P_updated"],
        "train_shared_context_word_mismatches": f_train["shared_context_word_mismatches"],
        "heldout_shared_context_word_mismatches": f_held["shared_context_word_mismatches"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
