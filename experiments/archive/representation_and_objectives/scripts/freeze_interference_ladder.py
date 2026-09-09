#!/usr/bin/env python3
"""research: freeze an interference-controlled state-update measurement.

Builds a systematic ladder of context conditions around state-change scenarios,
holding the final update fixed while varying the amount and type of prior
conflicting/consistent evidence.  Uses the same slot vocabulary as v3 D frames
(open/closed, empty/full) but generates more base frames and applies 10
interference conditions to each.

The object is frozen before any model scoring.  All targets are one-token under
both legal16k and legal40k tokenizers.  All vocabulary is corpus-attested.

Conditions (per base frame):
  1. no_context          — empty context (pure target prior)
  2. explicit_final      — direct statement of final state
  3. last_event          — single event, no prior state
  4. consistent_prior    — prior agrees + event
  5. distractor_event    — unrelated distractor + final event
  6. contradict_bare     — contradictory prior + final event
  7. contradict_temporal — contradictory prior + "Then" + final event
  8. contradict_reinforced — contradictory prior + reinforcement + final event
  9. explicit_override   — contradictory prior + explicit final-state sentence
  10. reported_event     — third-party report of final event
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import hashlib
import json
import os
import random
from pathlib import Path
from typing import Any

USER_ROOT = _public_path('.')
os.chdir(USER_ROOT)

A01_WS = _public_path('experiments/archive/representation_and_objectives')
A02_WS = _public_path('experiments/archive/frontier_consolidation')
OUT_DIR = _public_path('experiments/archive/representation_and_objectives/data/interference_ladder')
NOTE_PATH = _public_path('research/notes/representation_and_objectives/interference_ladder_freeze.md')

POOL_PATH = _public_path('experiments/archive/frontier_consolidation/data/density_cleanqwen_overlay_medium_riskhard/cleanqwen_fineweb_compact_view_reinvest_10M.jsonl')
TOK16_PATH = str(_public_path('experiments/archive/frontier_consolidation/training/runs/adapter128_scale1p75_h100M100M_seed43022_official_ladder/hf_model/chck_82M'))
TOK40_PATH = str(_public_path('experiments/archive/representation_and_objectives/training/runs/legal40k_accum_compact_view_reinvest_seed43022/hf_model/chck_100M'))

SEED = 18602

# --- Slot vocabulary ---
FAMILIES = {
    "open_closed": {
        "state_a": "open", "state_b": "closed",
        "event_a": "opened", "event_b": "closed",
        "objects": ["box", "bag", "window", "door", "drawer"],
    },
    "empty_full": {
        "state_a": "empty", "state_b": "full",
        "event_a": "emptied", "event_b": "filled",
        "objects": ["glass", "cup", "bottle"],
    },
}

ACTORS = ["Kate", "Jack", "Anna", "Emma", "Mary", "Seth", "Amy", "Rose", "Alice", "Tom"]
REPORTERS = ["Sam", "John", "Max", "Ben"]
DISTRACTOR_OBJECTS = ["cup", "ball", "pen", "key", "book", "coin", "plate", "hat"]

CONDITIONS = [
    "no_context",
    "explicit_final",
    "last_event",
    "consistent_prior",
    "distractor_event",
    "contradict_bare",
    "contradict_temporal",
    "contradict_reinforced",
    "explicit_override",
    "reported_event",
]


def build_contexts(obj: str, actor: str, fam: dict, reporter: str,
                   distractor: str) -> dict[str, tuple[str, str]]:
    """Return {condition: (context1, context2)} where c1→state_a correct, c2→state_b correct."""
    sa, sb = fam["state_a"], fam["state_b"]
    ea, eb = fam["event_a"], fam["event_b"]

    out = {}
    # 1. no_context
    out["no_context"] = ("", "")

    # 2. explicit_final
    out["explicit_final"] = (
        f"The {obj} is now {sa}.",
        f"The {obj} is now {sb}.",
    )

    # 3. last_event
    out["last_event"] = (
        f"{actor} {ea} the {obj}.",
        f"{actor} {eb} the {obj}.",
    )

    # 4. consistent_prior (prior agrees with final, then event)
    out["consistent_prior"] = (
        f"The {obj} was already {sa}. Then {actor} {ea} the {obj}.",
        f"The {obj} was already {sb}. Then {actor} {eb} the {obj}.",
    )

    # 5. distractor_event (unrelated action, then final event)
    out["distractor_event"] = (
        f"The {distractor} fell on the floor. {actor} {ea} the {obj}.",
        f"The {distractor} fell on the floor. {actor} {eb} the {obj}.",
    )

    # 6. contradict_bare (prior contradicts final, then final event)
    out["contradict_bare"] = (
        f"The {obj} was {sb}. {actor} {ea} the {obj}.",
        f"The {obj} was {sa}. {actor} {eb} the {obj}.",
    )

    # 7. contradict_temporal (same as bare but with "Then")
    out["contradict_temporal"] = (
        f"The {obj} was {sb}. Then {actor} {ea} the {obj}.",
        f"The {obj} was {sa}. Then {actor} {eb} the {obj}.",
    )

    # 8. contradict_reinforced (prior + reinforcement + final event)
    out["contradict_reinforced"] = (
        f"The {obj} was {sb}. The {obj} stayed {sb}. Then {actor} {ea} the {obj}.",
        f"The {obj} was {sa}. The {obj} stayed {sa}. Then {actor} {eb} the {obj}.",
    )

    # 9. explicit_override (prior contradicts, then explicit final statement)
    out["explicit_override"] = (
        f"The {obj} was {sb}. The {obj} is now {sa}.",
        f"The {obj} was {sa}. The {obj} is now {sb}.",
    )

    # 10. reported_event (third-party report)
    out["reported_event"] = (
        f"{reporter} said that {actor} {ea} the {obj}.",
        f"{reporter} said that {actor} {eb} the {obj}.",
    )

    return out


def single_token_in_context(tok, word: str) -> bool:
    """Check if word occupies exactly one token when rendered in a realistic sentence context.
    Uses offset_mapping, matching the actual scorer's target identification."""
    test_sent = f"The item is now {word}."
    enc = tok(test_sent, return_offsets_mapping=True, return_tensors=None)
    start = test_sent.index(word)
    end = start + len(word)
    selected = [i for i, (a, b) in enumerate(enc["offset_mapping"])
                if not (a == b == 0) and b > start and a < end]
    return len(selected) == 1


def verify_targets(tok16, tok40, word: str) -> bool:
    return single_token_in_context(tok16, word) and single_token_in_context(tok40, word)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_str(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def main():
    import time
    from transformers import AutoTokenizer

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(json.dumps({"event": "load_tokenizers", "tok16": TOK16_PATH, "tok40": TOK40_PATH}), flush=True)
    tok16 = AutoTokenizer.from_pretrained(TOK16_PATH, trust_remote_code=True, use_fast=True)
    tok40 = AutoTokenizer.from_pretrained(TOK40_PATH, trust_remote_code=True, use_fast=True)

    rng = random.Random(SEED)

    # Verify target words are one-token under both tokenizers
    target_checks = {}
    for fam_name, fam in FAMILIES.items():
        for w in [fam["state_a"], fam["state_b"]]:
            ok = verify_targets(tok16, tok40, w)
            target_checks[w] = ok
            if not ok:
                raise RuntimeError(f"Target word '{w}' is not single-token under both tokenizers")

    # Generate base frames
    base_frames = []
    frame_counter = 0
    rejects = {"distractor_same_as_object": 0}

    for fam_name, fam in sorted(FAMILIES.items()):
        for obj in fam["objects"]:
            for actor in ACTORS:
                # Pick a reporter (different from actor)
                available_reporters = [r for r in REPORTERS if r != actor]
                reporter = rng.choice(available_reporters)

                # Pick a distractor object (different from frame object)
                available_distractors = [d for d in DISTRACTOR_OBJECTS if d != obj]
                distractor = rng.choice(available_distractors)

                frame_counter += 1
                base_id = f"BASE_{frame_counter:04d}"

                base_frames.append({
                    "base_id": base_id,
                    "family": fam_name,
                    "object": obj,
                    "actor": actor,
                    "reporter": reporter,
                    "distractor": distractor,
                    "state_a": fam["state_a"],
                    "state_b": fam["state_b"],
                    "event_a": fam["event_a"],
                    "event_b": fam["event_b"],
                })

    print(json.dumps({"event": "base_frames", "count": len(base_frames)}), flush=True)

    # Generate interference conditions for each base frame
    all_frames = []
    cond_counter = 0
    for bf in base_frames:
        contexts = build_contexts(
            bf["object"], bf["actor"], bf, bf["reporter"], bf["distractor"]
        )
        query = f"The {bf['object']} is now {{target}}."

        for cond in CONDITIONS:
            c1, c2 = contexts[cond]
            cond_counter += 1
            frame = {
                "frame_id": f"IF_{cond_counter:05d}",
                "base_id": bf["base_id"],
                "condition": cond,
                "family": bf["family"],
                "object": bf["object"],
                "actor": bf["actor"],
                "alt_a": bf["state_a"],
                "alt_b": bf["state_b"],
                "context1": c1,
                "context2": c2,
                "query_template": query,
                "correct_c1": "a",
                "correct_c2": "b",
                "reporter": bf["reporter"] if cond == "reported_event" else None,
                "distractor": bf["distractor"] if cond == "distractor_event" else None,
            }
            all_frames.append(frame)

    print(json.dumps({"event": "total_frames", "count": len(all_frames), "conditions": len(CONDITIONS), "bases": len(base_frames)}), flush=True)

    # Write frames JSONL
    frames_path = _public_path('experiments/archive/representation_and_objectives/data/interference_ladder/interference_ladder_frames.jsonl')
    content_lines = []
    for f in all_frames:
        line = json.dumps(f, ensure_ascii=False, sort_keys=True)
        content_lines.append(line)
    content_str = "\n".join(content_lines) + "\n"
    frames_path.write_text(content_str, encoding="utf-8")
    frames_sha = sha256_str(content_str)

    # Generate renaming controls: for each base frame, swap object and actor
    # Use a deterministic rotation within each family
    rename_map = {}
    for fam_name, fam in sorted(FAMILIES.items()):
        objs = fam["objects"]
        # Rotate objects by 1
        for i, o in enumerate(objs):
            rename_map[f"{fam_name}_{o}"] = objs[(i + 1) % len(objs)]
    # Rotate actors by 3
    actor_rename = {}
    for i, a in enumerate(ACTORS):
        actor_rename[a] = ACTORS[(i + 3) % len(ACTORS)]

    renamed_frames = []
    for f in all_frames:
        fam_name = f["family"]
        new_obj = rename_map.get(f"{fam_name}_{f['object']}", f["object"])
        new_actor = actor_rename.get(f["actor"], f["actor"])
        new_reporter = actor_rename.get(f.get("reporter", ""), f.get("reporter"))

        # Rebuild contexts with renamed slots
        fam = FAMILIES[fam_name]
        new_distractor = f.get("distractor", "ball")
        if new_distractor == new_obj:
            available = [d for d in DISTRACTOR_OBJECTS if d != new_obj]
            new_distractor = available[0]

        new_contexts = build_contexts(new_obj, new_actor, fam, new_reporter or "Sam", new_distractor)
        cond = f["condition"]
        c1r, c2r = new_contexts[cond]

        renamed_frames.append({
            **f,
            "frame_id": f["frame_id"] + "_renamed",
            "object": new_obj,
            "actor": new_actor,
            "context1": c1r,
            "context2": c2r,
            "query_template": f"The {new_obj} is now {{target}}.",
            "reporter": new_reporter if cond == "reported_event" else None,
            "distractor": new_distractor if cond == "distractor_event" else None,
            "renamed_from": f["frame_id"],
        })

    renamed_path = _public_path('experiments/archive/representation_and_objectives/data/interference_ladder/interference_ladder_renamed_controls.jsonl')
    renamed_lines = [json.dumps(r, ensure_ascii=False, sort_keys=True) for r in renamed_frames]
    renamed_str = "\n".join(renamed_lines) + "\n"
    renamed_path.write_text(renamed_str, encoding="utf-8")
    renamed_sha = sha256_str(renamed_str)

    # Count frames per condition
    cond_counts = {}
    for f in all_frames:
        c = f["condition"]
        cond_counts[c] = cond_counts.get(c, 0) + 1

    # Manifest
    manifest = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "scientific_object": "interference-controlled state-update measurement",
        "description": "Systematic ladder of 10 context conditions around state-change scenarios, varying prior-state interference while holding the final update fixed",
        "seed": SEED,
        "pool_path": str(POOL_PATH.relative_to(USER_ROOT)),
        "pool_sha256": "215944978157394dbecf2039f2c1e1806bfcbb9701421920f78605eb58975a23",
        "n_base_frames": len(base_frames),
        "n_conditions": len(CONDITIONS),
        "n_total_frames": len(all_frames),
        "n_renamed_controls": len(renamed_frames),
        "conditions": CONDITIONS,
        "condition_counts": cond_counts,
        "families": {k: {kk: vv for kk, vv in v.items() if kk != "objects"} for k, v in FAMILIES.items()},
        "family_objects": {k: v["objects"] for k, v in FAMILIES.items()},
        "actors": ACTORS,
        "reporters": REPORTERS,
        "distractor_objects": DISTRACTOR_OBJECTS,
        "target_token_checks": target_checks,
        "frame_path": str(frames_path.relative_to(USER_ROOT)),
        "frame_sha256": frames_sha,
        "renamed_path": str(renamed_path.relative_to(USER_ROOT)),
        "renamed_sha256": renamed_sha,
        "content_sha256": sha256_str(content_str + renamed_str),
        "rejects": rejects,
    }
    man_path = _public_path('experiments/archive/representation_and_objectives/data/interference_ladder/interference_ladder_manifest.json')
    man_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")

    # Note
    note_lines = [
        "# research interference-controlled state-update measurement",
        "",
        f"Status: **FROZEN**",
        "",
        "Systematic ladder of 10 interference conditions applied to state-change scenarios.",
        "Each base frame has one object, one actor, one state-change pair (open/closed or empty/full).",
        "Conditions vary prior-state interference while holding the final update fixed.",
        "",
        f"- Base frames: {len(base_frames)} ({sum(1 for f in base_frames if f['family']=='open_closed')} open_closed, {sum(1 for f in base_frames if f['family']=='empty_full')} empty_full)",
        f"- Conditions: {len(CONDITIONS)}",
        f"- Total frames: {len(all_frames)}",
        f"- Renamed controls: {len(renamed_frames)}",
        f"- Frame SHA256: `{frames_sha}`",
        f"- Renamed SHA256: `{renamed_sha}`",
        f"- Content SHA256: `{manifest['content_sha256']}`",
        "",
        "## Conditions",
        "",
        "| # | Condition | Description | Example (open_closed, box, Alice) |",
        "|---|---|---|---|",
    ]

    example_contexts = build_contexts("box", "Alice", FAMILIES["open_closed"], "Sam", "ball")
    for i, c in enumerate(CONDITIONS, 1):
        c1, c2 = example_contexts[c]
        desc_map = {
            "no_context": "Pure target prior (empty context)",
            "explicit_final": "Direct statement of final state",
            "last_event": "Single event, no prior state",
            "consistent_prior": "Prior agrees with final + event",
            "distractor_event": "Unrelated distractor + final event",
            "contradict_bare": "Contradictory prior + final event",
            "contradict_temporal": "Contradictory prior + 'Then' + final event",
            "contradict_reinforced": "Contradictory prior + reinforcement + final event",
            "explicit_override": "Contradictory prior + explicit final statement",
            "reported_event": "Third-party report of final event",
        }
        desc = desc_map.get(c, "")
        note_lines.append(f"| {i} | `{c}` | {desc} | c1: \"{c1}\" / c2: \"{c2}\" |")

    note_lines.extend([
        "",
        "## Interference hypothesis",
        "",
        "research D-state ablation found:",
        "- `explicit_final` crossed=1.0 for all mature models",
        "- `last_event_only` crossed=0.03 to 0.89, trajectory-dependent",
        "- `initial_plus_last` (= contradict_bare) crossed=0.0 for all mature models",
        "- `full_v3` (= contradict_reinforced approx) crossed=0.0 for all",
        "",
        "The interference boundary is between no-prior-state and any-prior-state conditions.",
        "This ladder tests whether specific interference mitigations (temporal markers,",
        "explicit overrides, consistent evidence, reported framing, distractors) change",
        "the boundary, and whether the pattern varies across training trajectories.",
        "",
        "## Next: score existing checkpoints before any training",
        "",
        f"Frames: `{manifest['frame_path']}`",
        f"Renamed: `{manifest['renamed_path']}`",
        f"Manifest: `{str(man_path.relative_to(USER_ROOT))}`",
    ])

    _public_path('research/notes/representation_and_objectives').mkdir(parents=True, exist_ok=True)
    NOTE_PATH.write_text("\n".join(note_lines) + "\n", encoding="utf-8")

    print(json.dumps({
        "event": "frozen",
        "frames": str(frames_path.relative_to(USER_ROOT)),
        "renamed": str(renamed_path.relative_to(USER_ROOT)),
        "manifest": str(man_path.relative_to(USER_ROOT)),
        "note": str(NOTE_PATH.relative_to(USER_ROOT)),
        "content_sha256": manifest["content_sha256"],
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
