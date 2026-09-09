#!/usr/bin/env python3
"""Step038c: tiny curated source-grounded recipient-contrast probe.

The goal is not to create a training set. It is to materialize a few rows that
satisfy the research contract better than research, so the scorer can be exercised
on semantically grounded natural-style contexts while larger generation/validation
work is repaired.
"""
from __future__ import annotations

import json
import pathlib
import re
import time
from typing import Any

OUT_DIR = pathlib.Path("experiments/archive/functional_learning/data/curated_source_grounded_probe")
ROWS_PATH = OUT_DIR / "curated_source_grounded_training_rows.jsonl"
PAIRS_PATH = OUT_DIR / "curated_source_grounded_pairs.jsonl"
SUMMARY_PATH = OUT_DIR / "curated_source_grounded_summary.json"
NOTE_PATH = pathlib.Path("research/notes/functional_learning/curated_source_grounded_probe.md")

PAIRS = [
    {
        "pair_id": "sgp_chart_netherlands_australia",
        "source_origin": "simple_wiki_step058_raw",
        "raw_source_text": "It went to number 1 in New Zealand, number 3 in the United States, number 11 in Australia, number 14 in Canada and the Netherlands and number 20 in Belgium.",
        "target_entity": "Netherlands",
        "distractor_entity": "Australia",
        "relation_type": "chart_position",
        "target_source_answer": "number 14",
        "distractor_source_answer": "number 11",
        "shared_new_answer": "number 5",
        "update_event_frame": "After the reissue, the current chart position for {ENTITY} changed to number 5.",
        "target_use_frame": "The current chart position for Netherlands is {STATE}.",
        "target_evidence_quote": "number 14 in Canada and the Netherlands",
        "distractor_evidence_quote": "number 11 in Australia",
    },
    {
        "pair_id": "sgp_chart_italy_billboard",
        "source_origin": "simple_wiki_step058_raw",
        "raw_source_text": "It is the second single from his 16th studio album \"Working on a Dream\" and went to number 41 in Italy and number 18 on the Billboard Adult Alternative Songs chart.",
        "target_entity": "Italy",
        "distractor_entity": "Billboard Adult Alternative Songs",
        "relation_type": "chart_position",
        "target_source_answer": "number 41",
        "distractor_source_answer": "number 18",
        "shared_new_answer": "number 1",
        "update_event_frame": "After the reissue, the current chart position for {ENTITY} changed to number 1.",
        "target_use_frame": "The current chart position for Italy is {STATE}.",
        "target_evidence_quote": "number 41 in Italy",
        "distractor_evidence_quote": "number 18 on the Billboard Adult Alternative Songs chart",
    },
    {
        "pair_id": "sgp_pope_action_nicholas_photios",
        "source_origin": "simple_wiki_step058_raw",
        "raw_source_text": "Pope Nicholas I had refused to recognize Patriarch Photios I of Constantinople, who in turn had attacked the pope as a heretic.",
        "target_entity": "Pope Nicholas I",
        "distractor_entity": "Patriarch Photios I",
        "relation_type": "recorded_action_toward_other_leader",
        "target_source_answer": "refused to recognize",
        "distractor_source_answer": "attacked the pope as a heretic",
        "shared_new_answer": "exchanged diplomatic letters",
        "update_event_frame": "After the settlement, the current recorded action for {ENTITY} changed to exchanged diplomatic letters.",
        "target_use_frame": "The current recorded action for Pope Nicholas I is {STATE}.",
        "target_evidence_quote": "Pope Nicholas I had refused to recognize Patriarch Photios I",
        "distractor_evidence_quote": "Patriarch Photios I of Constantinople, who in turn had attacked the pope as a heretic",
    },
    {
        "pair_id": "sgp_music_video_fatima_trainor",
        "source_origin": "simple_wiki_step058_raw",
        "raw_source_text": "Fatima Robinson directed the music video for \"No\", which features Trainor performing choreographed dances in a warehouse and entwining her arms with accompanying female dancers.",
        "target_entity": "Fatima Robinson",
        "distractor_entity": "Trainor",
        "relation_type": "music_video_contribution",
        "target_source_answer": "directed",
        "distractor_source_answer": "choreographed dances",
        "shared_new_answer": "performed live vocals",
        "update_event_frame": "The production notes updated the current contribution for {ENTITY} to performed live vocals.",
        "target_use_frame": "The current contribution for Fatima Robinson is {STATE}.",
        "target_evidence_quote": "Fatima Robinson directed the music video",
        "distractor_evidence_quote": "Trainor performing choreographed dances",
    },
    {
        "pair_id": "sgp_military_listing_iraqi_bombers",
        "source_origin": "bnc_spoken_step058_raw",
        "raw_source_text": "At the same time, the three thousand five hundred tanks and three hundred and ninety thousand Iraqi troops have dug into defensive positions which make them difficult to knock out, even with the sophistication of British and United States ground attack bombers.",
        "target_entity": "Iraqi troops",
        "distractor_entity": "British and United States ground attack bombers",
        "relation_type": "military_listing",
        "target_source_answer": "defensive positions",
        "distractor_source_answer": "ground attack bombers",
        "shared_new_answer": "mobile artillery units",
        "update_event_frame": "The current military listing for {ENTITY} changed to mobile artillery units.",
        "target_use_frame": "The current military listing for Iraqi troops is {STATE}.",
        "target_evidence_quote": "Iraqi troops have dug into defensive positions",
        "distractor_evidence_quote": "British and United States ground attack bombers",
    },
    {
        "pair_id": "sgp_brand_label_mcafee_intel",
        "source_origin": "simple_wiki_step058_raw",
        "raw_source_text": "This bore the McAfee brand-name for years, until it was bought by Intel and given the Intel name.",
        "target_entity": "McAfee",
        "distractor_entity": "Intel",
        "relation_type": "product_label",
        "target_source_answer": "McAfee brand-name",
        "distractor_source_answer": "Intel name",
        "shared_new_answer": "corporate logo",
        "update_event_frame": "After the redesign, the current product label for {ENTITY} changed to corporate logo.",
        "target_use_frame": "The current product label for McAfee is {STATE}.",
        "target_evidence_quote": "bore the McAfee brand-name",
        "distractor_evidence_quote": "given the Intel name",
    },
]


def now_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def spans(text: str, sub: str) -> list[list[int]]:
    out = []
    text_l, sub_l = text.lower(), sub.lower()
    start = 0
    while True:
        i = text_l.find(sub_l, start)
        if i < 0:
            break
        out.append([i, i + len(sub_l)])
        start = i + max(1, len(sub_l))
    return out


def norm_ws(x: str) -> str:
    return " ".join(str(x or "").replace("\u00a0", " ").split())


def render_pair(p: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    source = norm_ws(p["raw_source_text"])
    target = p["target_entity"]
    distractor = p["distractor_entity"]
    up_frame = p["update_event_frame"]
    use_frame = p["target_use_frame"]
    assert up_frame.count("{ENTITY}") == 1, p["pair_id"]
    assert use_frame.count("{STATE}") == 1, p["pair_id"]
    validations = {
        "target_entity_spans": spans(source, target),
        "distractor_entity_spans": spans(source, distractor),
        "target_source_answer_spans": spans(source, p["target_source_answer"]),
        "distractor_source_answer_spans": spans(source, p["distractor_source_answer"]),
        "target_evidence_quote_spans": spans(source, p["target_evidence_quote"]),
        "distractor_evidence_quote_spans": spans(source, p["distractor_evidence_quote"]),
    }
    missing = [k for k, v in validations.items() if not v]
    if missing:
        raise ValueError(f"{p['pair_id']} missing raw spans: {missing}")
    target_update = norm_ws(up_frame.replace("{ENTITY}", target))
    distractor_update = norm_ws(up_frame.replace("{ENTITY}", distractor))
    update_use = norm_ws(use_frame.replace("{STATE}", p["shared_new_answer"]))
    retain_use = norm_ws(use_frame.replace("{STATE}", p["target_source_answer"]))
    pair_obj = {
        "pair_id": p["pair_id"],
        "source_sentence": source,
        "source_origin": p["source_origin"],
        "target_entity": target,
        "distractor_entity": distractor,
        "relation_type": p["relation_type"],
        "target_source_answer": p["target_source_answer"],
        "distractor_source_answer": p["distractor_source_answer"],
        "shared_new_answer": p["shared_new_answer"],
        "update_event_frame": up_frame,
        "use_sentence_frame": use_frame,
        "target_update_sentence": target_update,
        "distractor_update_sentence": distractor_update,
        "update_use_sentence": update_use,
        "retain_use_sentence": retain_use,
        "target_evidence_quote": p["target_evidence_quote"],
        "distractor_evidence_quote": p["distractor_evidence_quote"],
        "validation": validations,
        "semantic_intent": "TARGET_UPDATE should answer shared_new_answer; DISTRACTOR_UPDATE should answer target_source_answer. Both candidate phrases are visible; correctness requires selecting which entity the update applied to.",
    }
    rows = [
        {
            "pair_id": p["pair_id"],
            "packet_type": "UPDATE",
            "source_sentence": source,
            "update_sentence": target_update,
            "use_sentence": update_use,
            "use_sentence_frame": use_frame,
            "full_text": f"{source} {target_update} {update_use}",
            "answer_text": p["shared_new_answer"],
            "foil_text": p["target_source_answer"],
            "entity_name": target,
            "distractor_entity": distractor,
            "update_entity": target,
            "target_source_state": p["relation_type"],
            "target_source_answer": p["target_source_answer"],
            "distractor_source_state": p["relation_type"],
            "distractor_source_answer": p["distractor_source_answer"],
            "target_new_update_state": p["shared_new_answer"],
            "update_state_text": p["shared_new_answer"],
            "source_state_text": p["target_source_answer"],
            "answer_state_kind": "target_new",
            "role_position_relation": "manual_balanced_unknown",
            "answer_in_use_spans": spans(update_use, p["shared_new_answer"]),
            "answer_elsewhere_spans": spans(source + " " + target_update, p["shared_new_answer"]),
            "foil_spans": spans(source + " " + target_update + " " + update_use, p["target_source_answer"]),
            "answer_also_in_source_or_update": bool(spans(source + " " + target_update, p["shared_new_answer"])),
        },
        {
            "pair_id": p["pair_id"],
            "packet_type": "RETAIN",
            "source_sentence": source,
            "update_sentence": distractor_update,
            "use_sentence": retain_use,
            "use_sentence_frame": use_frame,
            "full_text": f"{source} {distractor_update} {retain_use}",
            "answer_text": p["target_source_answer"],
            "foil_text": p["shared_new_answer"],
            "entity_name": target,
            "distractor_entity": distractor,
            "update_entity": distractor,
            "target_source_state": p["relation_type"],
            "target_source_answer": p["target_source_answer"],
            "distractor_source_state": p["relation_type"],
            "distractor_source_answer": p["distractor_source_answer"],
            "target_new_update_state": p["shared_new_answer"],
            "update_state_text": p["shared_new_answer"],
            "source_state_text": p["target_source_answer"],
            "answer_state_kind": "target_source",
            "role_position_relation": "manual_balanced_unknown",
            "answer_in_use_spans": spans(retain_use, p["target_source_answer"]),
            "answer_elsewhere_spans": spans(source + " " + distractor_update, p["target_source_answer"]),
            "foil_spans": spans(source + " " + distractor_update + " " + retain_use, p["shared_new_answer"]),
            "answer_also_in_source_or_update": bool(spans(source + " " + distractor_update, p["target_source_answer"])),
        },
    ]
    return pair_obj, rows


def write_jsonl(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pair_objs: list[dict[str, Any]] = []
    train_rows: list[dict[str, Any]] = []
    for p in PAIRS:
        pair_obj, rows = render_pair(p)
        pair_objs.append(pair_obj)
        train_rows.extend(rows)
    write_jsonl(PAIRS_PATH, pair_objs)
    write_jsonl(ROWS_PATH, train_rows)
    summary = {
        "status": "STEP038C_CURATED_SOURCE_GROUNDED_PROBE",
        "created_utc": now_utc(),
        "n_pairs": len(pair_objs),
        "n_training_rows": len(train_rows),
        "rows_path": str(ROWS_PATH),
        "pairs_path": str(PAIRS_PATH),
        "construction": "manually curated from raw research source sentences with exact raw entity/answer/evidence spans and explicit current replacement/update events",
        "semantic_intent": "a small probe only; not a training corpus or proof of general natural-data quality",
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    NOTE_PATH.parent.mkdir(parents=True, exist_ok=True)
    md = ["# Step038c curated source-grounded probe\n\n"]
    md.append("This tiny probe materializes six source-grounded recipient contrasts under the research contract. It is not a training set; it checks that the row schema and scorer can operate on semantically clearer examples while generation/validation is repaired.\n\n")
    md.append(f"Rows: `{ROWS_PATH}`; pairs: `{PAIRS_PATH}`.\n\n")
    for po in pair_objs:
        md.append(f"## {po['pair_id']}\n\n")
        md.append(f"SOURCE: {po['source_sentence']}\n\n")
        md.append(f"TARGET={po['target_entity']} source answer `{po['target_source_answer']}`; DISTRACTOR={po['distractor_entity']} source answer `{po['distractor_source_answer']}`; new answer `{po['shared_new_answer']}`.\n\n")
        md.append(f"UPDATE frame: {po['update_event_frame']}\n\n")
        md.append(f"USE frame: {po['use_sentence_frame']}\n\n")
    NOTE_PATH.write_text("".join(md), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
