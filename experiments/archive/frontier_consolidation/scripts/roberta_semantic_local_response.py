#!/usr/bin/env python3
"""research: semantic stratification of RoBERTa compact-vs-repeat local response.

Reads per-event local NLL losses produced by roberta_pair_stratified_response_probe.py
with --write_event_losses and joins them to the frozen event set.  The purpose is to test
whether the local compact-trained advantage on compact source-absent events is concentrated
on relational/event-state words, as found for target-selective DeBERTa labels.

This script performs no training, no official evaluation, no upload, and no leaderboard action.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import collections
import hashlib
import json
import math
import random
import statistics
import sys
import time
from pathlib import Path
from typing import Any


def find_user_root() -> Path:
    return _PUBLIC_ROOT


ROOT = find_user_root()
WS = ROOT / "experiments/archive/frontier_consolidation"
A01_SCRIPTS = ROOT / "experiments/archive/representation_and_objectives/scripts"
if str(A01_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(A01_SCRIPTS))

try:
    from source_absent_lexical_profile import flags as lexical_flags  # type: ignore
    from annotate_packed_pool import norm_word, word_class  # type: ignore
except Exception as exc:  # pragma: no cover - explicit fallback for portability
    lexical_flags = None
    _IMPORT_ERROR = repr(exc)

    def norm_word(w: str) -> str:  # type: ignore
        import re
        parts = re.findall(r"[A-Za-z0-9]+(?:['\u2019-][A-Za-z0-9]+)?", w)
        return "".join(parts).lower() if parts else ""

    def word_class(w: str) -> str:  # type: ignore
        nw = norm_word(w)
        if not nw:
            return "other"
        if nw.isdigit():
            return "number"
        if w[:1].isupper():
            return "capitalized"
        return "word"
else:
    _IMPORT_ERROR = None

DEFAULT_EVENTS = WS / "data/roberta_pair_stratified_response_probe/frozen_events.jsonl"
DEFAULT_EVENT_LOSS_DIR = WS / "data/roberta_pair_stratified_response_late_cpu_eventlosses"
DEFAULT_MANIFEST = WS / "data/roberta_pair_stratified_response_probe/stratum_manifest.json"
DEFAULT_OUT = WS / "data/roberta_semantic_local_response"
LATE = ["chck_60M", "chck_70M", "chck_80M", "chck_90M", "chck_100M"]


def now_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
    return out


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def finite(x: Any) -> bool:
    try:
        return math.isfinite(float(x))
    except Exception:
        return False


def q(vals: list[float], p: float) -> float | None:
    if not vals:
        return None
    vals = sorted(vals)
    if len(vals) == 1:
        return vals[0]
    k = p * (len(vals) - 1)
    lo = int(math.floor(k)); hi = int(math.ceil(k))
    if lo == hi:
        return vals[lo]
    return vals[lo] * (hi - k) + vals[hi] * (k - lo)


def flag_groups(word: str) -> set[str]:
    nw = norm_word(word)
    if lexical_flags is not None:
        fs = set(lexical_flags(word, nw))
    else:
        fs = set()
        if word[:1].isupper():
            fs.add("capitalized")
        if any(ch.isdigit() for ch in word):
            fs.add("number_or_year")
    rel_event = bool({"relational_or_abstracting_word", "event_or_state_word"} & fs)
    groups = {"all"}
    groups.update(fs)
    groups.add("relational_or_event_state" if rel_event else "not_relational_or_event_state")
    groups.add("capitalized_or_number" if {"capitalized", "number_or_year"} & fs else "not_capitalized_or_number")
    if not ({"relational_or_abstracting_word", "event_or_state_word", "capitalized", "number_or_year", "generic_entity_word"} & fs):
        groups.add("ordinary_nonrel_nonentity")
    groups.add(f"class_{word_class(word)}")
    return groups


def weighted_adv(rows: list[dict[str, Any]]) -> float | None:
    # repeat_minus_compact_advantage = (compact_loss_sum - repeat_loss_sum) / pieces, negated from compact_minus_repeat_nll.
    den = sum(int(r.get("pieces", 0)) for r in rows)
    if den <= 0:
        return None
    delta = sum(float(r["compact_loss_sum"]) - float(r["repeat_loss_sum"]) for r in rows)
    return -delta / den


def summarize(rows: list[dict[str, Any]], reps: int, seed: int) -> dict[str, Any]:
    adv = weighted_adv(rows)
    den = sum(int(r.get("pieces", 0)) for r in rows)
    by_pair: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for r in rows:
        by_pair[str(r.get("pair_id"))].append(r)
    boot: list[float] = []
    if reps > 0 and len(by_pair) >= 2:
        keys = list(by_pair)
        rng = random.Random(seed)
        for _ in range(reps):
            sample: list[dict[str, Any]] = []
            for _j in keys:
                sample.extend(by_pair[rng.choice(keys)])
            v = weighted_adv(sample)
            if v is not None and finite(v):
                boot.append(float(v))
    out = {
        "events": len(rows),
        "clusters": len(by_pair),
        "pieces": den,
        "repeat_minus_compact_advantage": adv,
    }
    if boot:
        out["bootstrap"] = {
            "reps": len(boot),
            "p05": q(boot, 0.05),
            "p50": q(boot, 0.50),
            "p95": q(boot, 0.95),
            "p_gt_0": sum(1 for x in boot if x > 0) / len(boot),
        }
    return out


def enrich_records(events: dict[int, dict[str, Any]], loss_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for lr in loss_rows:
        idx = int(lr["event_index"])
        e = events.get(idx)
        if e is None:
            continue
        word = str(e.get("word", ""))
        groups = sorted(flag_groups(word))
        rec = dict(lr)
        rec.update({
            "word": word,
            "norm_word": norm_word(word),
            "word_class": word_class(word),
            "semantic_groups": groups,
            "relational_or_event_state": "relational_or_event_state" in groups,
            "capitalized_or_number": "capitalized_or_number" in groups,
        })
        out.append(rec)
    return out


def by_group(rows: list[dict[str, Any]], reps: int, seed: int) -> dict[str, Any]:
    groups = [
        "all",
        "relational_or_event_state",
        "not_relational_or_event_state",
        "ordinary_nonrel_nonentity",
        "capitalized_or_number",
        "relational_or_abstracting_word",
        "event_or_state_word",
        "generic_entity_word",
    ]
    viewcats = sorted({f"{r.get('view_type')}|{r.get('category')}" for r in rows})
    out: dict[str, Any] = {}
    for vc in viewcats:
        vrows = [r for r in rows if f"{r.get('view_type')}|{r.get('category')}" == vc]
        out[vc] = {}
        for g in groups:
            grows = [r for r in vrows if g in r.get("semantic_groups", [])]
            if grows:
                out[vc][g] = summarize(grows, reps=reps, seed=seed + 101 * (len(out[vc]) + 1))
    return out


def interaction_from_table(table: dict[str, Any]) -> dict[str, Any]:
    def adv(vc: str, grp: str) -> float | None:
        v = table.get(vc, {}).get(grp, {}).get("repeat_minus_compact_advantage")
        return float(v) if finite(v) else None
    sa_rel = adv("compact|source_absent_content", "relational_or_event_state")
    sa_other = adv("compact|source_absent_content", "not_relational_or_event_state")
    rc_rel = adv("compact|retained_content", "relational_or_event_state")
    rc_other = adv("compact|retained_content", "not_relational_or_event_state")
    cap = adv("compact|source_absent_content", "capitalized_or_number")
    ordinary = adv("compact|source_absent_content", "ordinary_nonrel_nonentity")
    interaction = None
    if all(finite(x) for x in [sa_rel, sa_other, rc_rel, rc_other]):
        interaction = (float(sa_rel) - float(sa_other)) - (float(rc_rel) - float(rc_other))
    return {
        "source_absent_rel_event": sa_rel,
        "source_absent_other": sa_other,
        "retained_rel_event": rc_rel,
        "retained_other": rc_other,
        "source_absent_capitalized_or_number": cap,
        "source_absent_ordinary_nonrel_nonentity": ordinary,
        "novelty_x_rel_event_interaction_using_compact_view_events": interaction,
        "reading": "Positive interaction means the compact-trained local advantage is especially concentrated on source-absent relational/event words beyond any retained relational/event advantage.",
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", type=Path, default=DEFAULT_EVENTS)
    ap.add_argument("--event_loss_dir", type=Path, default=DEFAULT_EVENT_LOSS_DIR)
    ap.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    ap.add_argument("--out_dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--checkpoints", nargs="+", default=LATE)
    ap.add_argument("--bootstrap_reps", type=int, default=300)
    ap.add_argument("--seed", type=int, default=21943023)
    args = ap.parse_args()

    events_list = read_jsonl(args.events)
    events = {int(e["event_index"]): e for e in events_list}
    manifest = read_json(args.manifest) if args.manifest.exists() else {}
    ck_payload: dict[str, Any] = {}
    late_interactions: list[dict[str, Any]] = []
    missing: list[str] = []
    for i, ck in enumerate(args.checkpoints):
        p = args.event_loss_dir / f"event_losses_{ck}.jsonl"
        if not p.exists():
            ck_payload[ck] = {"missing": True, "path": str(p)}
            missing.append(ck)
            continue
        rows = enrich_records(events, read_jsonl(p))
        table = by_group(rows, reps=args.bootstrap_reps, seed=args.seed + 1009 * i)
        inter = interaction_from_table(table)
        ck_payload[ck] = {
            "missing": False,
            "event_losses_path": str(p),
            "records": len(rows),
            "table": table,
            "key_interaction": inter,
        }
        late_interactions.append(inter)

    def mean_key(key: str) -> float | None:
        xs = [d.get(key) for d in late_interactions if finite(d.get(key))]
        return sum(float(x) for x in xs) / len(xs) if xs else None

    late_mean_interaction = {k: mean_key(k) for k in [
        "source_absent_rel_event",
        "source_absent_other",
        "retained_rel_event",
        "retained_other",
        "source_absent_capitalized_or_number",
        "source_absent_ordinary_nonrel_nonentity",
        "novelty_x_rel_event_interaction_using_compact_view_events",
    ]}
    payload = {
        "status": "ROBERTA_SEMANTIC_LOCAL_RESPONSE",
        "created_utc": now_utc(),
        "meaning": "Semantic stratification of RoBERTa compact-vs-repeat local response; tests whether source-absent local advantage concentrates on relational/event words. Local mechanism evidence only, to be read with official selected trajectory.",
        "inputs": {
            "events": str(args.events),
            "events_sha256": sha256_file(args.events),
            "manifest_events_sha256": manifest.get("frozen_events_sha256"),
            "event_loss_dir": str(args.event_loss_dir),
            "checkpoints": args.checkpoints,
            "lexical_flags_import_error": _IMPORT_ERROR,
        },
        "event_sha_matches_manifest": bool(manifest.get("frozen_events_sha256") == sha256_file(args.events)),
        "per_checkpoint": ck_payload,
        "late_mean_key_interaction": late_mean_interaction,
        "scientific_reading": "Use only as local pseudolikelihood evidence. Concordance would require late official selected compact-minus-repeat improvements on stable families; otherwise a semantic local advantage is another local-vs-downstream dissociation.",
        "no_training_or_official_eval_or_submission": True,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.out_dir / "roberta_semantic_local_response.json", payload)
    lines = [
        "# research RoBERTa semantic local response",
        "",
        f"Created: `{payload['created_utc']}`",
        "",
        f"Event SHA matches manifest: `{payload['event_sha_matches_manifest']}`",
        f"Missing checkpoints: `{missing}`",
        "",
        "## Late mean key interaction",
        "",
    ]
    for k, v in late_mean_interaction.items():
        lines.append(f"- `{k}`: `{v}`")
    lines += [
        "",
        "## Reading",
        payload["scientific_reading"],
        "",
        "Positive advantage means compact-trained model has lower local NLL than repeat-trained model on that event subset. Positive novelty×rel/event interaction means source-absent relational/event words receive a larger compact-trained advantage beyond retained relational/event words.",
    ]
    (args.out_dir / "roberta_semantic_local_response.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": payload["status"],
        "out": str(args.out_dir / "roberta_semantic_local_response.json"),
        "event_sha_matches_manifest": payload["event_sha_matches_manifest"],
        "missing": missing,
        "late_mean_key_interaction": late_mean_interaction,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
