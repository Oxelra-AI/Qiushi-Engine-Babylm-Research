#!/usr/bin/env python3
"""research: validate same-source contrastive binding generation outputs.

Input: Qwen outputs for prompts from build_contrastive_binding_prompts.py.
Output: accepted contrastive packet objects and two training rows per accepted
source:
  UPDATE = source + target_update + use_frame(target_new_answer)
  RETAIN = source + distractor_update + use_frame(target_source_answer)
The two rows keep the source and use frame fixed; only update target and answer
phrase differ.  This is the data side for the crossed same-source/credit-
placement entity-binding experiment with functional_learning.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import pathlib
import random
import re
import statistics
import time
from collections import Counter, defaultdict
from typing import Any

ROOT = pathlib.Path.cwd()
DEFAULT_OUT = ROOT / "experiments/archive/relation_learning/data/contrastive_binding_validated"

REQ_KEYS = [
    "target_entity", "distractor_entity", "target_source_state", "target_source_answer",
    "target_new_update_state", "target_new_answer", "distractor_source_state",
    "distractor_new_update_state", "target_update_sentence", "distractor_update_sentence",
    "use_sentence_frame",
]
WORD_RE = re.compile(r"[A-Za-z0-9]+(?:[\u2019'][A-Za-z0-9]+)?")
STOP = {
    "a","an","and","are","as","at","be","been","being","but","by","can","could","did","do","does","for","from",
    "had","has","have","he","her","hers","him","his","i","if","in","into","is","it","its","just","may","might","must",
    "not","of","on","or","our","she","should","so","some","such","than","that","the","their","them","then","there",
    "these","they","this","to","too","under","up","very","was","we","were","what","when","where","which","who","will",
    "with","would","you","your","before","after","while","over","also","once","only","own","each","any","all","one",
    "two","three","day","time","way","thing","person","people","made","make","makes","felt","feel","found","find",
}
CUE_REGEXES: list[tuple[str, re.Pattern[str]]] = [
    ("still", re.compile(r"\bstill\b", re.I)),
    ("remain", re.compile(r"\bremains?\b|\bremained\b|\bremaining\b", re.I)),
    ("continue", re.compile(r"\bcontinues?\b|\bcontinued\b|\bcontinuing\b", re.I)),
    ("stay", re.compile(r"\bstays?\b|\bstayed\b|\bstaying\b", re.I)),
    ("keep", re.compile(r"\bkeeps?\b|\bkept\b", re.I)),
    ("retain", re.compile(r"\bretains?\b|\bretained\b|\bretaining\b", re.I)),
    ("maintain", re.compile(r"\bmaintains?\b|\bmaintained\b|\bmaintaining\b", re.I)),
    ("unchanged", re.compile(r"\bunchanged\b|\bas before\b", re.I)),
    ("today", re.compile(r"\btoday\b|\bnowadays\b", re.I)),
    ("now", re.compile(r"\bnow\b", re.I)),
    ("current", re.compile(r"\bcurrently\b|\bcurrent\b|\bpresently\b|\bat present\b", re.I)),
    ("new", re.compile(r"\bnew\b|\bnewly\b", re.I)),
    ("no_longer", re.compile(r"\bno longer\b|\banymore\b", re.I)),
    ("again_later", re.compile(r"\bagain\b|\blater\b", re.I)),
    ("former", re.compile(r"\bformer\b|\bformerly\b", re.I)),
]
TOY_RE = re.compile(r"\b(box|basket|container|marble)\b", re.I)


def now_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def norm_ws(x: str) -> str:
    return " ".join(str(x or "").replace("\u00a0", " ").split())


def words(x: str) -> list[str]:
    return [m.group(0) for m in WORD_RE.finditer(str(x or ""))]


def norm_tokens(x: str) -> list[str]:
    return [re.sub(r"[^a-z0-9]+", "", w.lower().replace("\u2019", "'")) for w in words(x) if re.sub(r"[^a-z0-9]+", "", w.lower())]


def content_tokens(x: str) -> list[str]:
    return [t for t in norm_tokens(x) if t not in STOP and len(t) > 1]


def norm_phrase(x: str) -> str:
    return " ".join(norm_tokens(x))


def cue_hits(x: str) -> list[str]:
    return [name for name, rx in CUE_REGEXES if rx.search(x or "")]


def jaccard(a: list[str], b: list[str]) -> float:
    A, B = set(a), set(b)
    if not A and not B:
        return 0.0
    return len(A & B) / max(1, len(A | B))


def overlap_min(a: list[str], b: list[str]) -> float:
    A, B = set(a), set(b)
    if not A or not B:
        return 0.0
    return len(A & B) / max(1, min(len(A), len(B)))


def contiguous_lcs(a: list[str], b: list[str]) -> tuple[int, str]:
    best = 0; best_i = 0
    dp = [0] * (len(b) + 1)
    for i, x in enumerate(a, 1):
        ndp = [0] * (len(b) + 1)
        for j, y in enumerate(b, 1):
            if x == y:
                ndp[j] = dp[j-1] + 1
                if ndp[j] > best:
                    best = ndp[j]; best_i = i - best
        dp = ndp
    return best, " ".join(a[best_i:best_i+best]) if best else ""


def parse_jsonish(raw: str) -> tuple[dict[str, Any] | None, str]:
    s = (raw or "").strip()
    if not s:
        return None, "empty_output"
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?", "", s.strip(), flags=re.I).strip()
        s = re.sub(r"```$", "", s.strip()).strip()
    try:
        obj = json.loads(s)
        return (obj, "") if isinstance(obj, dict) else (None, "json_not_object")
    except json.JSONDecodeError:
        pass
    # Recover a single JSON object if the model added a prefix/suffix.
    a = s.find("{"); b = s.rfind("}")
    if a >= 0 and b > a:
        try:
            obj = json.loads(s[a:b+1])
            return (obj, "") if isinstance(obj, dict) else (None, "json_not_object")
        except json.JSONDecodeError:
            return None, "json_parse_failed"
    return None, "json_parse_failed"


def entity_match(entity: str, text: str) -> bool:
    et = [t.rstrip("s") for t in norm_tokens(entity) if t not in {"the", "a", "an"}]
    tt = set(t.rstrip("s") for t in norm_tokens(text))
    return bool(et) and all(t in tt for t in et)


def phrase_in(phrase: str, text: str) -> bool:
    p = norm_phrase(phrase)
    t = norm_phrase(text)
    return bool(p) and p in t


def spans(text: str, sub: str) -> list[list[int]]:
    out: list[list[int]] = []
    if not sub:
        return out
    start = 0
    low, sublow = text.lower(), sub.lower()
    while True:
        i = low.find(sublow, start)
        if i < 0:
            break
        out.append([i, i + len(sub)])
        start = i + max(1, len(sub))
    return out


def validate_obj(obj: dict[str, Any], prompt: dict[str, Any]) -> tuple[dict[str, Any] | None, list[str], dict[str, Any]]:
    reasons: list[str] = []
    detail: dict[str, Any] = {}
    missing = [k for k in REQ_KEYS if k not in obj]
    if missing:
        return None, ["missing_keys"], {"missing": missing}

    vals = {k: norm_ws(str(obj.get(k, ""))) for k in REQ_KEYS}
    if any(not vals[k] for k in REQ_KEYS):
        reasons.append("empty_required_value")

    target = vals["target_entity"]
    distractor = vals["distractor_entity"]
    source = norm_ws(prompt.get("source_sentence", ""))
    candidates = [norm_ws(x) for x in prompt.get("candidate_entities", [])]
    source_answer = vals["target_source_answer"]
    new_answer = vals["target_new_answer"]
    target_update = vals["target_update_sentence"]
    distractor_update = vals["distractor_update_sentence"]
    frame = vals["use_sentence_frame"]
    update_use = frame.replace("{STATE}", new_answer)
    retain_use = frame.replace("{STATE}", source_answer)

    # Entity and frame structure.
    if norm_phrase(target) == norm_phrase(distractor):
        reasons.append("target_equals_distractor")
    if candidates:
        if not any(norm_phrase(target) == norm_phrase(c) or entity_match(target, c) or entity_match(c, target) for c in candidates):
            reasons.append("target_not_in_candidates")
        if not any(norm_phrase(distractor) == norm_phrase(c) or entity_match(distractor, c) or entity_match(c, distractor) for c in candidates):
            reasons.append("distractor_not_in_candidates")
    if not entity_match(target, source):
        reasons.append("target_not_in_source")
    if not entity_match(distractor, source):
        reasons.append("distractor_not_in_source")
    if not entity_match(target, target_update):
        reasons.append("target_update_missing_target")
    if entity_match(distractor, target_update):
        reasons.append("target_update_mentions_distractor")
    if not entity_match(distractor, distractor_update):
        reasons.append("distractor_update_missing_distractor")
    if entity_match(target, distractor_update):
        reasons.append("distractor_update_mentions_target")
    if frame.count("{STATE}") != 1:
        reasons.append("frame_placeholder_count")
    if not entity_match(target, frame):
        reasons.append("frame_missing_target")
    if entity_match(distractor, frame):
        reasons.append("frame_mentions_distractor")

    # Cues and toy states.
    cue_fields = {"frame": frame, "target_update": target_update, "distractor_update": distractor_update, "update_use": update_use, "retain_use": retain_use}
    field_hits = {k: cue_hits(v) for k, v in cue_fields.items() if cue_hits(v)}
    if field_hits:
        reasons.append("cue_word_present")
    if any(TOY_RE.search(v) for v in [target, distractor, vals["target_source_state"], source_answer, vals["target_new_update_state"], new_answer, vals["distractor_new_update_state"], frame, target_update, distractor_update]):
        reasons.append("toy_state_word")

    # Lengths.
    tuw, duw, uuw, ruw = map(lambda x: len(words(x)), [target_update, distractor_update, update_use, retain_use])
    detail.update({"target_update_words": tuw, "distractor_update_words": duw, "update_use_words": uuw, "retain_use_words": ruw})
    if not (8 <= tuw <= 24): reasons.append("target_update_length")
    if not (8 <= duw <= 24): reasons.append("distractor_update_length")
    if not (7 <= uuw <= 20): reasons.append("update_use_length")
    if not (7 <= ruw <= 20): reasons.append("retain_use_length")

    # State-answer relation and recurrence controls.
    src_ans_ct = content_tokens(source_answer); new_ans_ct = content_tokens(new_answer)
    if len(src_ans_ct) < 1 or len(new_ans_ct) < 1:
        reasons.append("answer_low_content")
    if norm_phrase(source_answer) == norm_phrase(new_answer) or jaccard(src_ans_ct, new_ans_ct) > 0.50:
        reasons.append("source_new_answer_too_similar")
    if overlap_min(src_ans_ct, content_tokens(source + " " + vals["target_source_state"])) < 0.34:
        reasons.append("source_answer_not_source_grounded")
    if overlap_min(new_ans_ct, content_tokens(target_update + " " + vals["target_new_update_state"])) < 0.34:
        reasons.append("new_answer_not_update_grounded")
    # Strong phrase-recurrence filters: completed use should not simply replay the long state phrase.
    if phrase_in(vals["target_source_state"], retain_use) and len(content_tokens(vals["target_source_state"])) >= 2:
        reasons.append("source_state_phrase_recurred_in_retain_use")
    if phrase_in(vals["target_new_update_state"], update_use) and len(content_tokens(vals["target_new_update_state"])) >= 2:
        reasons.append("new_state_phrase_recurred_in_update_use")
    if len(src_ans_ct) >= 2 and phrase_in(source_answer, source):
        reasons.append("source_answer_exactly_copied_from_source")
    if len(new_ans_ct) >= 2 and phrase_in(new_answer, target_update):
        reasons.append("new_answer_exactly_copied_from_update")

    # Surface copy controls against the natural source and update sentences.
    for label, text_a, text_b in [
        ("update_use_vs_source", update_use, source),
        ("retain_use_vs_source", retain_use, source),
        ("update_use_vs_target_update", update_use, target_update),
        ("retain_use_vs_distractor_update", retain_use, distractor_update),
    ]:
        lcs, span = contiguous_lcs(norm_tokens(text_a), norm_tokens(text_b))
        detail[label] = {"contiguous_lcs": lcs, "span": span, "content_jaccard": jaccard(content_tokens(text_a), content_tokens(text_b)), "overlap_min": overlap_min(content_tokens(text_a), content_tokens(text_b))}
        if lcs >= 6:
            reasons.append(f"{label}_lcs_ge6")
    if detail["update_use_vs_target_update"]["content_jaccard"] >= 0.80:
        reasons.append("update_use_high_jaccard_with_update")
    if detail["retain_use_vs_source"]["content_jaccard"] >= 0.85:
        reasons.append("retain_use_high_jaccard_with_source")

    if reasons:
        return None, reasons, detail

    pid = str(prompt.get("pair_id") or prompt.get("id", ""))
    full_update = f"{source} {target_update} {update_use}"
    full_retain = f"{source} {distractor_update} {retain_use}"
    pair_obj: dict[str, Any] = {
        "pair_id": pid,
        "source_sentence": source,
        "qwen_rewrite": prompt.get("qwen_rewrite"),
        "source": prompt.get("source"),
        "example_id": prompt.get("example_id"),
        "cohort": prompt.get("cohort"),
        "candidate_entities": candidates,
        **vals,
        "update_use_sentence": update_use,
        "retain_use_sentence": retain_use,
        "full_text_UPDATE": full_update,
        "full_text_RETAIN": full_retain,
        "surface": detail,
    }
    rows = []
    for typ, update_sent, use_sent, full_text, answer, foil, update_entity, update_state, answer_state_kind in [
        ("UPDATE", target_update, update_use, full_update, new_answer, source_answer, target, vals["target_new_update_state"], "target_new"),
        ("RETAIN", distractor_update, retain_use, full_retain, source_answer, new_answer, distractor, vals["distractor_new_update_state"], "target_source"),
    ]:
        rows.append({
            "pair_id": pid,
            "packet_type": typ,
            "source_sentence": source,
            "update_sentence": update_sent,
            "use_sentence": use_sent,
            "use_sentence_frame": frame,
            "full_text": full_text,
            "answer_text": answer,
            "foil_text": foil,
            "entity_name": target,
            "distractor_entity": distractor,
            "update_entity": update_entity,
            "target_source_state": vals["target_source_state"],
            "target_new_update_state": vals["target_new_update_state"],
            "distractor_new_update_state": vals["distractor_new_update_state"],
            "answer_state_kind": answer_state_kind,
            "answer_in_use_spans": spans(use_sent, answer),
            "answer_elsewhere_spans": spans((source + " " + update_sent), answer),
            "foil_spans": spans(full_text, foil),
            "answer_also_in_source_or_update": bool(spans((source + " " + update_sent), answer)),
            "word_count": len(words(full_text)),
        })
    pair_obj["training_rows"] = rows
    return pair_obj, [], detail


def read_jsonl(path: pathlib.Path, tolerate_partial: bool = False) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip(): continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                if tolerate_partial: continue
                raise RuntimeError(f"Bad JSON in {path} line {line_no}")
    return rows


def write_jsonl(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def stat(xs: list[float]) -> dict[str, Any]:
    ys = [float(x) for x in xs if math.isfinite(float(x))]
    if not ys:
        return {"n": 0, "mean": None, "median": None, "p95": None, "max": None}
    ys = sorted(ys)
    p95 = ys[min(len(ys)-1, int(math.ceil(0.95*len(ys))) - 1)]
    return {"n": len(ys), "mean": round(statistics.mean(ys), 4), "median": round(statistics.median(ys), 4), "p95": round(p95, 4), "max": round(max(ys), 4)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--outputs", required=True)
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT))
    ap.add_argument("--tag", default="pilot")
    ap.add_argument("--tolerate-partial", action="store_true")
    ap.add_argument("--manual-sample-n", type=int, default=40)
    args = ap.parse_args()

    t0 = time.time()
    out_dir = pathlib.Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    outputs = read_jsonl(pathlib.Path(args.outputs), tolerate_partial=args.tolerate_partial)
    prompts = read_jsonl(pathlib.Path(args.prompts))
    accepted_pairs: list[dict[str, Any]] = []
    train_rows: list[dict[str, Any]] = []
    reject_reasons = Counter(); rejected_examples = []
    for local_i, rec in enumerate(outputs):
        idx = rec.get("index")
        if not isinstance(idx, int) or not (0 <= idx < len(prompts)):
            reject_reasons["index_out_of_range"] += 1
            continue
        prompt = prompts[idx]
        raw = str(rec.get("output", rec.get("generated_text", rec.get("generated", ""))))
        obj, parse_reason = parse_jsonish(raw)
        if obj is None:
            reject_reasons[parse_reason] += 1
            if len(rejected_examples) < 800:
                rejected_examples.append({"index": idx, "pair_id": prompt.get("pair_id"), "reason": parse_reason, "raw_output": raw, "source_sentence": prompt.get("source_sentence")})
            continue
        pair, reasons, detail = validate_obj(obj, prompt)
        if pair is None:
            for r in reasons:
                reject_reasons[r] += 1
            if len(rejected_examples) < 800:
                rejected_examples.append({"index": idx, "pair_id": prompt.get("pair_id"), "reasons": reasons, "detail": detail, "obj": obj, "raw_output": raw, "source_sentence": prompt.get("source_sentence")})
            continue
        pair["generation_index"] = idx
        accepted_pairs.append(pair)
        train_rows.extend(pair["training_rows"])

    pairs_path = out_dir / f"accepted_contrastive_binding_pairs_{args.tag}.jsonl"
    rows_path = out_dir / f"accepted_contrastive_binding_training_rows_{args.tag}.jsonl"
    rej_path = out_dir / f"rejected_examples_{args.tag}.jsonl"
    meta_path = out_dir / f"validation_metadata_{args.tag}.json"
    sample_path = out_dir / f"accepted_manual_read_sample_{args.tag}.md"
    write_jsonl(pairs_path, accepted_pairs)
    write_jsonl(rows_path, train_rows)
    write_jsonl(rej_path, rejected_examples)

    rng = random.Random(57058)
    sample = accepted_pairs[:]
    rng.shuffle(sample)
    lines = [f"# research contrastive binding accepted sample ({args.tag})\n\n"]
    for i, p in enumerate(sample[:args.manual_sample_n], 1):
        lines.append(f"## {i}. {p['pair_id']} ({p.get('source')})\n\n")
        lines.append(f"SOURCE: {p['source_sentence']}\n\n")
        lines.append(f"TARGET={p['target_entity']} DISTRACTOR={p['distractor_entity']}\n\n")
        lines.append(f"SOURCE_STATE={p['target_source_state']} | SOURCE_ANSWER={p['target_source_answer']}\n\n")
        lines.append(f"NEW_STATE={p['target_new_update_state']} | NEW_ANSWER={p['target_new_answer']}\n\n")
        lines.append(f"TARGET_UPDATE: {p['target_update_sentence']}\n\n")
        lines.append(f"DISTRACTOR_UPDATE: {p['distractor_update_sentence']}\n\n")
        lines.append(f"FRAME: {p['use_sentence_frame']}\n\n")
        lines.append(f"UPDATE_USE: {p['update_use_sentence']}\n\n")
        lines.append(f"RETAIN_USE: {p['retain_use_sentence']}\n\n")
    sample_path.write_text("".join(lines), encoding="utf-8")

    by_source = Counter(p.get("source", "") for p in accepted_pairs)
    by_cohort = Counter(p.get("cohort", "") for p in accepted_pairs)
    word_counts = [r["word_count"] for r in train_rows]
    meta = {
        "status": "CONTRASTIVE_BINDING_VALIDATED",
        "created_utc": now_utc(),
        "tag": args.tag,
        "outputs_path": args.outputs,
        "prompts_path": args.prompts,
        "outputs_read": len(outputs),
        "prompts_read": len(prompts),
        "accepted_pairs": len(accepted_pairs),
        "accepted_training_rows": len(train_rows),
        "acceptance_rate_per_output": round(len(accepted_pairs) / max(1, len(outputs)), 4),
        "rejection_reasons": dict(reject_reasons.most_common()),
        "accepted_source_distribution": dict(by_source.most_common()),
        "accepted_cohort_distribution": dict(by_cohort.most_common()),
        "training_row_word_count": stat([float(x) for x in word_counts]),
        "outputs": {"accepted_pairs": str(pairs_path), "training_rows": str(rows_path), "rejected_examples": str(rej_path), "manual_sample": str(sample_path)},
        "sha256": {"accepted_pairs": sha256_file(pairs_path), "training_rows": sha256_file(rows_path), "manual_sample": sha256_file(sample_path)},
        "elapsed_sec": round(time.time() - t0, 2),
    }
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(meta, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
