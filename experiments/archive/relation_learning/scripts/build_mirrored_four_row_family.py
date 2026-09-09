#!/usr/bin/env python3
"""research: Mirrored four-row relation-fact binding family from maps.

Each map with entities A,B and values v_A, v_B, shared new nv produces:
  C_A = source + update_a(+strangers)   C_B = source + update_b(+strangers)
  Row 1: (C_A, ask A) -> nv  [updated]   Row 2: (C_A, ask B) -> v_B [retained]
  Row 3: (C_B, ask A) -> v_A [retained]  Row 4: (C_B, ask B) -> nv  [updated]

By construction:
 - Same context appears in both answer roles -> length/tokens carry no info
 - Each entity queried in both roles -> entity name carries no info
 - Stranger updates identical for both contexts -> op count neutral
 - Only predictor: identity-conditioned state assignment
"""

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import json, os, re, random
from pathlib import Path
from collections import Counter, defaultdict

ROOT = _public_path('.')
MAPS_PATH = _public_path('experiments/archive/functional_learning/data/assignment_reversal_export/a01_assignment_reversal_operation_maps.jsonl')
OUT_DIR = _public_path('experiments/archive/relation_learning/data/mirrored_four_row_family')
SEED = 91091
K_VALUES = [0, 1, 2, 3, 4]

# ── Frame templates: (id, split, prefix_template, suffix) ──
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
    return bool(re.search(r'\b' + re.escape(word) + r'\b', text))


def validate(m: dict) -> list[str]:
    issues = []
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
    vals = {m["value_a"], m["value_b"], m["shared_new_value"]}
    if len(vals) < 3:
        issues.append("duplicate values")
    return issues


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load maps ──
    maps = []
    with open(MAPS_PATH) as f:
        for line in f:
            maps.append(json.loads(line.strip()))
    print(f"Loaded {len(maps)} maps")

    # ── Validate ──
    valid, skipped = [], []
    for m in maps:
        iss = validate(m)
        if iss:
            skipped.append((m["base_pair_id"], iss))
            print(f"  SKIP {m['base_pair_id']}: {'; '.join(iss)}")
        else:
            valid.append(m)
    print(f"Valid: {len(valid)}/{len(maps)}")

    # ── Stranger pool ──
    pool = []
    for m in valid:
        pool.append({"entity": m["entity_a"], "value": m["shared_new_value"],
                      "sentence": m["update_a_sentence"], "map_id": m["base_pair_id"]})
        pool.append({"entity": m["entity_b"], "value": m["shared_new_value"],
                      "sentence": m["update_b_sentence"], "map_id": m["base_pair_id"]})
    print(f"Stranger pool: {len(pool)} entries")

    # ── Build rows and quads ──
    rng = random.Random(SEED)
    all_rows: list[dict] = []
    all_quads: list[dict] = []

    for m in valid:
        rel = m["relation"]
        frames = FRAMES.get(rel, [])
        if not frames:
            continue
        cur_ents = {m["entity_a"], m["entity_b"]}
        mid = m["base_pair_id"]

        for k in K_VALUES:
            eligible = [s for s in pool
                        if s["entity"] not in cur_ents and s["map_id"] != mid]
            strangers = rng.sample(eligible, min(k, len(eligible)))
            stranger_sents = [s["sentence"] for s in strangers]
            stranger_suffix = (" " + " ".join(stranger_sents)) if stranger_sents else ""

            ca = m["source_context"] + " " + m["update_a_sentence"] + stranger_suffix
            cb = m["source_context"] + " " + m["update_b_sentence"] + stranger_suffix

            all_present = sorted(set([m["value_a"], m["value_b"], m["shared_new_value"]]
                                     + [s["value"] for s in strangers]))

            for fid, fsplit, ftpl, fsuf in frames:
                qid = f"{mid}:k{k}:{fid}"
                qp_a = ftpl.replace("{entity}", m["entity_a"])
                qp_b = ftpl.replace("{entity}", m["entity_b"])

                def mk(rid, ctx, qp, eq, ep, ans, arole, svq, svp, cv, ur):
                    rt = ctx + " " + qp + ans + fsuf
                    acs = len(ctx) + 1 + len(qp)
                    ace = acs + len(ans)
                    assert rt[acs:ace] == ans, f"Span mismatch: '{rt[acs:ace]}' != '{ans}'"
                    return {
                        "row_id": rid, "quad_id": qid,
                        "split": m["split"], "map_id": mid, "relation": rel,
                        "k": k, "frame_id": fid, "frame_split": fsplit,
                        "context_variant": cv,
                        "entity_queried": eq, "entity_partner": ep,
                        "answer": ans, "answer_role": arole,
                        "source_value_queried": svq, "source_value_partner": svp,
                        "shared_new_value": m["shared_new_value"],
                        "update_recipient": ur,
                        "stranger_entities": [s["entity"] for s in strangers],
                        "stranger_values": [s["value"] for s in strangers],
                        "all_present_values": all_present,
                        "context_text": ctx, "query_prefix": qp,
                        "answer_text": ans, "answer_suffix": fsuf,
                        "row_text": rt,
                        "answer_char_start": acs, "answer_char_end": ace,
                        "row_words": len(rt.split()),
                        "context_words": len(ctx.split()),
                    }

                r1 = mk(f"{m['split']}:{mid}:k{k}:ca:ask_a:{fid}",
                         ca, qp_a, m["entity_a"], m["entity_b"],
                         m["shared_new_value"], "updated",
                         m["value_a"], m["value_b"], "ca", m["entity_a"])
                r2 = mk(f"{m['split']}:{mid}:k{k}:ca:ask_b:{fid}",
                         ca, qp_b, m["entity_b"], m["entity_a"],
                         m["value_b"], "retained",
                         m["value_b"], m["value_a"], "ca", m["entity_a"])
                r3 = mk(f"{m['split']}:{mid}:k{k}:cb:ask_a:{fid}",
                         cb, qp_a, m["entity_a"], m["entity_b"],
                         m["value_a"], "retained",
                         m["value_a"], m["value_b"], "cb", m["entity_b"])
                r4 = mk(f"{m['split']}:{mid}:k{k}:cb:ask_b:{fid}",
                         cb, qp_b, m["entity_b"], m["entity_a"],
                         m["shared_new_value"], "updated",
                         m["value_b"], m["value_a"], "cb", m["entity_b"])

                all_rows.extend([r1, r2, r3, r4])
                all_quads.append({
                    "quad_id": qid, "map_id": mid,
                    "split": m["split"], "relation": rel,
                    "k": k, "frame_id": fid, "frame_split": fsplit,
                    "rows": {
                        "ca_ask_a": r1["row_id"], "ca_ask_b": r2["row_id"],
                        "cb_ask_a": r3["row_id"], "cb_ask_b": r4["row_id"],
                    },
                    "shared_context_pairs": [
                        [r1["row_id"], r2["row_id"]],
                        [r3["row_id"], r4["row_id"]],
                    ],
                    "shared_query_pairs": [
                        [r1["row_id"], r3["row_id"]],
                        [r2["row_id"], r4["row_id"]],
                    ],
                })

    # ── Split into output files ──
    train_seen = [r for r in all_rows
                  if r["split"] == "train" and r["frame_split"] == "train_seen"]
    train_all = [r for r in all_rows if r["split"] == "train"]
    heldout_seen = [r for r in all_rows
                    if r["split"] == "heldout" and r["frame_split"] == "train_seen"]
    heldout_all = [r for r in all_rows if r["split"] == "heldout"]

    qt_s = [q for q in all_quads
            if q["split"] == "train" and q["frame_split"] == "train_seen"]
    qh_s = [q for q in all_quads
            if q["split"] == "heldout" and q["frame_split"] == "train_seen"]
    qh_a = [q for q in all_quads if q["split"] == "heldout"]

    def save_jsonl(data, path):
        with open(path, "w", encoding="utf-8") as f:
            for d in data:
                f.write(json.dumps(d, ensure_ascii=False) + "\n")

    save_jsonl(train_seen, _public_path('experiments/archive/relation_learning/data/mirrored_four_row_family/mirrored_train_frame_seen.jsonl'))
    save_jsonl(train_all,  _public_path('experiments/archive/relation_learning/data/mirrored_four_row_family/mirrored_train_frame_all.jsonl'))
    save_jsonl(heldout_seen, _public_path('experiments/archive/relation_learning/data/mirrored_four_row_family/mirrored_heldout_frame_seen.jsonl'))
    save_jsonl(heldout_all,  _public_path('experiments/archive/relation_learning/data/mirrored_four_row_family/mirrored_heldout_frame_all.jsonl'))
    save_jsonl(qt_s, _public_path('experiments/archive/relation_learning/data/mirrored_four_row_family/binding_quads_train_frame_seen.jsonl'))
    save_jsonl(qh_s, _public_path('experiments/archive/relation_learning/data/mirrored_four_row_family/binding_quads_heldout_frame_seen.jsonl'))
    save_jsonl(qh_a, _public_path('experiments/archive/relation_learning/data/mirrored_four_row_family/binding_quads_heldout_frame_all.jsonl'))

    # ── Feature table ──
    def feat(rows, label):
        if not rows:
            return {"label": label, "n": 0}
        n = len(rows)
        n_upd = sum(1 for r in rows if r["answer_role"] == "updated")
        by_k = defaultdict(lambda: {"n": 0, "upd": 0})
        for r in rows:
            by_k[r["k"]]["n"] += 1
            if r["answer_role"] == "updated":
                by_k[r["k"]]["upd"] += 1

        # Shared-context word mismatches (should be 0)
        cg = defaultdict(list)
        for r in rows:
            cg[(r["quad_id"], r["context_variant"])].append(r["context_words"])
        ctx_mm = sum(1 for v in cg.values() if len(set(v)) > 1)

        # C_A vs C_B word delta per shared-query pair
        qg = defaultdict(dict)
        for r in rows:
            key = (r["map_id"], r["k"], r["frame_id"], r["entity_queried"])
            qg[key][r["context_variant"]] = r["context_words"]
        deltas = [v["ca"] - v["cb"] for v in qg.values()
                  if "ca" in v and "cb" in v]

        # Answer length by role
        upd_al = [len(r["answer"].split()) for r in rows if r["answer_role"] == "updated"]
        ret_al = [len(r["answer"].split()) for r in rows if r["answer_role"] == "retained"]

        return {
            "label": label, "n": n,
            "words": sum(r["row_words"] for r in rows),
            "P_updated": n_upd / n,
            "by_k": {k: {"n": v["n"], "P_upd": v["upd"] / v["n"]}
                     for k, v in sorted(by_k.items())},
            "shared_ctx_word_mismatches": ctx_mm,
            "ca_cb_delta_mean": sum(deltas) / len(deltas) if deltas else 0,
            "ca_cb_delta_min": min(deltas) if deltas else 0,
            "ca_cb_delta_max": max(deltas) if deltas else 0,
            "ans_words_upd": sum(upd_al) / len(upd_al) if upd_al else 0,
            "ans_words_ret": sum(ret_al) / len(ret_al) if ret_al else 0,
        }

    ts = feat(train_seen, "train_frame_seen")
    ha = feat(heldout_all, "heldout_frame_all")

    manifest = {
        "status": "MIRRORED_FOUR_ROW_FAMILY",
        "source": str(MAPS_PATH.relative_to(ROOT)),
        "valid_maps": len(valid), "total_maps": len(maps),
        "skipped": [(mid, iss) for mid, iss in skipped],
        "k_values": K_VALUES, "seed": SEED,
        "relations": dict(Counter(m["relation"] for m in valid)),
        "counts": {
            "train_frame_seen": {"rows": len(train_seen), "quads": len(qt_s),
                                 "words": ts.get("words", 0)},
            "train_frame_all": {"rows": len(train_all)},
            "heldout_frame_seen": {"rows": len(heldout_seen), "quads": len(qh_s)},
            "heldout_frame_all": {"rows": len(heldout_all), "quads": len(qh_a),
                                  "words": ha.get("words", 0)},
        },
        "features_train": ts, "features_heldout": ha,
    }
    with open(_public_path('experiments/archive/relation_learning/data/mirrored_four_row_family/manifest.json'), "w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    # ── Summary ──
    lines = [
        "# research mirrored four-row relation-fact binding family\n",
        "From maps: values present as literal strings in context.",
        f"Valid maps: {len(valid)}/{len(maps)}",
        f"Relations: {dict(Counter(m['relation'] for m in valid))}",
        "",
        "## Design",
        "C_A = source + update_a + strangers; C_B = source + update_b + strangers",
        "1. (C_A, ask A) -> nv [updated]   2. (C_A, ask B) -> v_B [retained]",
        "3. (C_B, ask A) -> v_A [retained]  4. (C_B, ask B) -> nv [updated]",
        "",
        "Structural balance: same context in both answer roles; each entity in both roles;",
        "operation count identical within quad; stranger updates identical for both contexts.",
        "Only sufficient predictor: identity-conditioned state assignment.",
        "",
    ]
    for stat in [ts, ha]:
        if stat["n"] == 0:
            continue
        lines.append(f"## {stat['label']}: {stat['n']} rows, {stat['words']} words")
        lines.append(f"P(updated) = {stat['P_updated']:.3f}")
        lines.append(f"Shared-context word mismatches: {stat['shared_ctx_word_mismatches']}")
        lines.append(f"C_A vs C_B word delta: mean={stat['ca_cb_delta_mean']:.2f}, "
                      f"range=[{stat['ca_cb_delta_min']}, {stat['ca_cb_delta_max']}]")
        lines.append(f"Answer words: updated={stat['ans_words_upd']:.2f}, "
                      f"retained={stat['ans_words_ret']:.2f}")
        for k, kb in stat["by_k"].items():
            lines.append(f"  k={k}: n={kb['n']}, P(updated)={kb['P_upd']:.3f}")
        lines.append("")

    if skipped:
        lines.append("## Skipped maps")
        for mid, iss in skipped:
            lines.append(f"  {mid}: {'; '.join(iss)}")

    with open(_public_path('research/documents/relation_learning/data/mirrored_four_row_family/summary.md'), "w") as f:
        f.write("\n".join(lines) + "\n")

    print(json.dumps({
        "status": manifest["status"],
        "valid_maps": len(valid),
        "counts": manifest["counts"],
        "P_updated_train": ts.get("P_updated", 0),
        "P_updated_heldout": ha.get("P_updated", 0),
        "shared_ctx_mismatches_train": ts.get("shared_ctx_word_mismatches", 0),
        "shared_ctx_mismatches_heldout": ha.get("shared_ctx_word_mismatches", 0),
    }, indent=2))


if __name__ == "__main__":
    main()
