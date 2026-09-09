#!/usr/bin/env python3
"""research: deterministic, legally inspectable PVDM labels and matched control.

The research spaCy pilot established that pivot->dependent supervision is plausible,
but a pretrained dependency parser is not allowed for the BabyLM Strict-Small
training pipeline because it was learned on external language data. This script
replaces it with a deterministic extractor: fixed lexical/pattern rules, corpus
frequency counts computed only from the allowed compact 10M pool, and no learned
parser/POS model.

Output labels are for the 70M->100M tail of the compact stream. The matched
control uses exactly the same dependent target word indices as PVDM and therefore
preserves the dependent-target distribution by construction; it differs only in
which visible anchor is protected from masking (true pivot versus matched
surrogate anchor). Masked mass equality is enforced later in the trainer by a
per-batch/per-row probability normalizer.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

STUDY = Path("experiments/archive/representation_and_objectives")
WS = STUDY
STREAM_100M = WS / "data/fw_source_breadth_arm/fw_preserved_compact_view_100M.jsonl"
POOL_10M = WS / "data/fw_full_arms/fw_preserved_compact_view_10M.jsonl"
OUT = WS / "data/pvdm_deterministic_labels"
NOTE = (WS.parents[2] / 'research/notes/representation_and_objectives/pvdm_compliance_and_control_design.md')
TOKENIZER_DIR = WS / "data/shared_tokenizer/shared_16k_tokenizer"
EXPECTED_STREAM_SHA = "c8d7f24b5edd2dad21f589c8cde72671d0b76038d9632fcd3d6c79178a24be68"
EXPECTED_POOL_SHA = "ee08a1f8d974248aa9bdda14dfe48724b93873e6961db137d9bbdd078e5ef914"
EXPECTED_TOKENIZER_SHA = "e70d167f620668130f88744998bd1bacbe05f6d0735012144fe501b271b04366"
START_WORDS = 70_000_000
TAIL_WORDS = 30_000_000

LEAD_TRAIL = " \t\n\r\"'“”‘’()[]{}<>.,;:!?/\\|`~*_=+"

STOP = {
    "a","an","the","and","or","but","if","then","else","of","for","to","from","with","without","by","as","at","be","am","is","are","was","were","been","being",
    "i","you","he","she","it","we","they","me","him","her","us","them","my","your","his","its","our","their","this","that","these","those","there","here",
    "do","does","did","have","has","had","not","no","yes","can","could","may","might","must","shall","should","will","would","just","only","also","very"
}
NONREL_FUNCTION = {
    "the","a","an","and","or","but","of","for","by","as","at","this","that","these","those","there","here","also","very","just","only","all","some","many","most"
}

CHANGE = {
    "cause","causes","caused","causing","lead","leads","led","leading","result","results","resulted","resulting",
    "make","makes","made","making","force","forces","forced","forcing","prevent","prevents","prevented","preventing",
    "allow","allows","allowed","allowing","enable","enables","enabled","enabling","melt","melts","melted","melting",
    "freeze","freezes","froze","frozen","freezing","heat","heats","heated","heating","cool","cools","cooled","cooling",
    "break","breaks","broke","broken","breaking","fix","fixes","fixed","fixing","open","opens","opened","opening",
    "close","closes","closed","closing","increase","increases","increased","increasing","decrease","decreases","decreased","decreasing",
    "rise","rises","rose","risen","rising","fall","falls","fell","fallen","falling","grow","grows","grew","grown","growing",
    "shrink","shrinks","shrank","shrunk","shrinking","expand","expands","expanded","expanding","contract","contracts","contracted","contracting",
    "push","pushes","pushed","pushing","pull","pulls","pulled","pulling","lift","lifts","lifted","lifting","drop","drops","dropped","dropping",
    "move","moves","moved","moving","enter","enters","entered","entering","leave","leaves","left","leaving","give","gives","gave","given","giving",
    "take","takes","took","taken","taking","add","adds","added","adding","remove","removes","removed","removing","build","builds","built","building",
    "destroy","destroys","destroyed","destroying","create","creates","created","creating","turn","turns","turned","turning","change","changes","changed","changing"
}
SPATIAL = {
    "above","below","over","under","behind","beside","between","inside","outside","near","within","across","through","toward","towards","into","onto","around","along","up","down","off","on","in"
}
TEMPORAL = {"before","after","during","until","while","when","then","later","earlier","subsequently","since","once","again"}
COMPARATIVE = {"more","less","greater","smaller","larger","higher","lower","faster","slower","longer","shorter","better","worse","than","least","most"}
POLARITY_MODAL = {"not","never","cannot","can't","can","could","may","might","must","should","would","will","won't","wouldn't","couldn't","shouldn't","without"}
CAUSAL_CONNECTOR = {"because","so","therefore","thereby","hence","thus","unless","although","though","whereas","while","if","since"}
PIVOT_WORDS = CHANGE | SPATIAL | TEMPORAL | COMPARATIVE | POLARITY_MODAL | CAUSAL_CONNECTOR

CONTENT_SUFFIX = ("tion","ment","ness","ity","ance","ence","ship","age","ism","ist","er","or","ing","ed","ive","al","ous","ful","less","able","ible","ly")
PUNCT_RE = re.compile(r"^[\W_]+$", re.UNICODE)
NUM_RE = re.compile(r"^[+-]?(?:\d+[\d,]*(?:\.\d+)?|\.\d+)(?:[%°])?$")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def norm_word(w: str) -> str:
    return w.strip(LEAD_TRAIL).lower()


def is_punct_or_empty(w: str) -> bool:
    n = norm_word(w)
    return not n or bool(PUNCT_RE.match(n))


def is_number(n: str) -> bool:
    return bool(NUM_RE.match(n.replace(",", "")))


def freq_bin(freq: int) -> str:
    if freq <= 0: return "0"
    if freq <= 2: return "1-2"
    if freq <= 5: return "3-5"
    if freq <= 10: return "6-10"
    if freq <= 25: return "11-25"
    if freq <= 50: return "26-50"
    if freq <= 100: return "51-100"
    if freq <= 250: return "101-250"
    if freq <= 500: return "251-500"
    if freq <= 1000: return "501-1000"
    return "1001+"


def freq_bin_id(freq: int) -> int:
    bins = [0,2,5,10,25,50,100,250,500,1000]
    for i, hi in enumerate(bins):
        if freq <= hi:
            return i
    return len(bins)


def coarse_class(word: str, norm: str | None = None) -> str:
    n = norm if norm is not None else norm_word(word)
    if not n: return "empty"
    if is_number(n): return "number"
    if n in POLARITY_MODAL: return "modal_polarity"
    if n in CAUSAL_CONNECTOR or n in TEMPORAL: return "clause_connector"
    if n in SPATIAL: return "adposition_like"
    if n in COMPARATIVE or n.endswith("er") or n.endswith("est"): return "comparative_like"
    if n in CHANGE: return "event_verb_like"
    if n in NONREL_FUNCTION: return "function"
    if word[:1].isupper(): return "capitalized"
    if n.endswith(("ing","ed","en","ize","ise","ify")): return "verbish"
    if n.endswith(("ly",)): return "adverbish"
    if n.endswith(("ive","al","ous","ful","less","able","ible")): return "adjectival"
    if n.endswith(("tion","ment","ness","ity","ance","ence","ship","age","ism")): return "nominal"
    return "content"


def pivot_category(n: str, words: list[str], i: int) -> str | None:
    if n in CHANGE: return "change_causal_verb"
    if n in CAUSAL_CONNECTOR: return "causal_connector"
    if n in TEMPORAL: return "temporal"
    if n in SPATIAL: return "spatial"
    if n in POLARITY_MODAL: return "polarity_modal"
    # Comparatives: include explicit words and simple -er/-est if supported by local than/more/less context.
    if n in COMPARATIVE: return "comparative"
    if (n.endswith("er") or n.endswith("est")) and any(norm_word(w) in {"than","more","less"} for w in words[max(0,i-3):min(len(words),i+4)]):
        return "comparative"
    return None


def is_content_target(w: str) -> bool:
    n = norm_word(w)
    if not n or n in STOP or n in PIVOT_WORDS: return False
    if len(n) <= 1: return False
    if PUNCT_RE.match(n): return False
    return True


def collect_targets(words: list[str], i: int, cat: str) -> list[int]:
    """Conservative local deterministic approximation to pivot-dependent links."""
    out: list[int] = []
    n = len(words)

    def add_right(window: int, cap: int, skip_first_stop: bool = True):
        for j in range(i + 1, min(n, i + 1 + window)):
            nw = norm_word(words[j])
            if not nw:
                continue
            # A punctuation-bearing word can delimit weak clause windows.
            if words[j].strip().startswith((".",";",":","?","!")):
                break
            if is_content_target(words[j]):
                out.append(j)
                if len(out) >= cap:
                    return

    def add_left(window: int, cap: int):
        for j in range(i - 1, max(-1, i - 1 - window), -1):
            if is_content_target(words[j]):
                out.append(j)
                if len(out) >= cap:
                    return

    if cat in {"spatial", "temporal", "causal_connector"}:
        add_right(10, 2)
        # For after/before/while clauses, one left-side content often supplies the governed event/state.
        if cat in {"temporal", "causal_connector"} and len(out) < 2:
            add_left(6, 2 - len(out))
    elif cat == "change_causal_verb":
        add_right(8, 2)
        if len(out) < 1:
            add_left(5, 1)
    elif cat == "comparative":
        # Prefer compared property/object on the right; also include item after 'than' if close.
        add_right(8, 2)
        for j in range(i + 1, min(n, i + 8)):
            if norm_word(words[j]) == "than":
                for k in range(j + 1, min(n, j + 6)):
                    if is_content_target(words[k]):
                        out.append(k); break
                break
    elif cat == "polarity_modal":
        add_right(6, 2)
    # Deduplicate and drop pivot.
    seen = set(); final = []
    for j in out:
        if j == i or j in seen: continue
        if not (0 <= j < n): continue
        seen.add(j); final.append(j)
        if len(final) >= 3:
            break
    return final


def distance_bin(d: int) -> str:
    if d <= 1: return "1"
    if d <= 2: return "2"
    if d <= 4: return "3-4"
    if d <= 8: return "5-8"
    if d <= 16: return "9-16"
    return "17+"


def build_freqs(path: Path) -> Counter[str]:
    c: Counter[str] = Counter()
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            words = json.loads(line)["text"].split()
            for w in words:
                n = norm_word(w)
                if n:
                    c[n] += 1
    return c


def control_anchor_for(words: list[str], pivot_i: int, target_i: int, pivot_cat: str, freqs: Counter[str], banned: set[int]) -> tuple[int | None, dict[str, Any]]:
    pivot_n = norm_word(words[pivot_i])
    pivot_freq = freqs.get(pivot_n, 0)
    pivot_fbin = freq_bin_id(pivot_freq)
    pivot_class = coarse_class(words[pivot_i], pivot_n)
    desired_dist = abs(target_i - pivot_i)
    desired_side = 1 if pivot_i > target_i else -1
    pivot_functionish = pivot_class in {"modal_polarity", "clause_connector", "adposition_like", "function"}
    best: tuple[float, int, dict[str, Any]] | None = None
    relational_positions = {idx for idx, w in enumerate(words) if norm_word(w) in PIVOT_WORDS}
    n = len(words)
    for j, w in enumerate(words):
        if j in banned or j == pivot_i or j == target_i: continue
        nw = norm_word(w)
        if not nw or PUNCT_RE.match(nw): continue
        # First preference: not a pivot word. If function-like matching is impossible, allow a weakly functional word.
        is_rel = j in relational_positions
        cls = coarse_class(w, nw)
        if is_rel:
            continue
        jf = freqs.get(nw, 0)
        jbin = freq_bin_id(jf)
        d = abs(target_i - j)
        if d == 0 or d > 24: continue
        side = 1 if j > target_i else -1
        # Match coarse syntactic class without a learned POS tagger. For function-like pivots use nonrelational function words;
        # for event/comparative pivots accept deterministic suffix/content classes.
        class_ok = (cls == pivot_class) or (pivot_functionish and cls == "function") or (not pivot_functionish and cls in {pivot_class, "content", "verbish", "adjectival", "nominal", "capitalized"})
        if not class_ok:
            continue
        score = 0.0
        score += 3.0 * abs(math.log1p(d) - math.log1p(desired_dist))
        score += 1.2 * abs(jbin - pivot_fbin)
        score += 0.4 * (side != desired_side)
        score += 0.8 * (cls != pivot_class)
        score += 0.02 * abs(j - target_i)
        detail = {
            "control_norm": nw,
            "control_class": cls,
            "control_freq_bin": freq_bin(jf),
            "control_distance": d,
            "distance_abs_diff": abs(d - desired_dist),
            "freq_bin_abs_diff": abs(jbin - pivot_fbin),
            "class_exact": cls == pivot_class,
            "score": round(score, 4),
        }
        if best is None or score < best[0]:
            best = (score, j, detail)
    if best is None:
        return None, {"matched": False}
    return best[1], {"matched": True, **best[2]}


def events_for_row(words: list[str], freqs: Counter[str], max_events: int) -> tuple[list[dict[str, Any]], Counter[str]]:
    stats: Counter[str] = Counter()
    raw_events: list[tuple[int, int, str]] = []
    for i, w in enumerate(words):
        nw = norm_word(w)
        cat = pivot_category(nw, words, i)
        if cat is None: continue
        tgts = collect_targets(words, i, cat)
        if not tgts: continue
        for t in tgts:
            raw_events.append((i, t, cat))
    # Keep broad coverage but prevent a few long rows from becoming all-target rows.
    raw_events = sorted(raw_events, key=lambda x: (x[0], abs(x[1]-x[0])))[: max_events * 3]
    events: list[dict[str, Any]] = []
    banned_targets: set[int] = set()
    for pivot_i, target_i, cat in raw_events:
        if len(events) >= max_events: break
        if target_i in banned_targets: continue
        banned = set(banned_targets) | {pivot_i, target_i}
        ctrl_i, ctrl_detail = control_anchor_for(words, pivot_i, target_i, cat, freqs, banned)
        if ctrl_i is None:
            stats["events_without_control"] += 1
            continue
        pivot_n = norm_word(words[pivot_i]); target_n = norm_word(words[target_i]); ctrl_n = norm_word(words[ctrl_i])
        ev = {
            "pivot_i": pivot_i,
            "target_i": target_i,
            "control_i": ctrl_i,
            "category": cat,
            "pivot": words[pivot_i],
            "target": words[target_i],
            "control": words[ctrl_i],
            "pivot_norm": pivot_n,
            "target_norm": target_n,
            "control_norm": ctrl_n,
            "pivot_class": coarse_class(words[pivot_i], pivot_n),
            "target_class": coarse_class(words[target_i], target_n),
            "target_freq_bin": freq_bin(freqs.get(target_n, 0)),
            "pivot_freq_bin": freq_bin(freqs.get(pivot_n, 0)),
            "control_freq_bin": freq_bin(freqs.get(ctrl_n, 0)),
            "pivot_target_distance": abs(target_i - pivot_i),
            "control_target_distance": abs(target_i - ctrl_i),
            "distance_bin": distance_bin(abs(target_i - pivot_i)),
            "control_match": ctrl_detail,
        }
        events.append(ev)
        banned_targets.add(target_i)
        stats["events_with_control"] += 1
    return events, stats


def stream_tail_rows(path: Path, start_words: int, tail_words: int):
    cumulative = 0
    emitted = 0
    with path.open(encoding="utf-8") as f:
        for row_idx, line in enumerate(f):
            if not line.strip(): continue
            obj = json.loads(line)
            words = int(obj.get("words", len(str(obj["text"]).split())))
            actual = len(str(obj["text"]).split())
            if words != actual:
                raise RuntimeError(f"word-count mismatch row {row_idx}: field={words} actual={actual}")
            next_cum = cumulative + words
            if next_cum <= start_words:
                cumulative = next_cum
                continue
            if cumulative < start_words:
                raise RuntimeError(f"tail begins inside row {row_idx}: cumulative={cumulative}, row_words={words}, start={start_words}")
            if emitted + words > tail_words:
                raise RuntimeError(f"tail would end inside row {row_idx}: emitted={emitted}, row_words={words}, tail={tail_words}")
            yield row_idx, obj
            emitted += words
            cumulative = next_cum
            if emitted == tail_words:
                return
    if emitted != tail_words:
        raise RuntimeError(f"tail emitted {emitted} words != {tail_words}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-events-per-row", type=int, default=8)
    ap.add_argument("--sample-limit", type=int, default=400)
    ap.add_argument("--skip-sha-check", action="store_true")
    args = ap.parse_args()
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    if not args.skip_sha_check:
        stream_sha = sha256_file(STREAM_100M)
        pool_sha = sha256_file(POOL_10M)
        tok_sha = sha256_file(TOKENIZER_DIR / "tokenizer.json")
        if stream_sha != EXPECTED_STREAM_SHA:
            raise RuntimeError(f"stream sha mismatch {stream_sha} != {EXPECTED_STREAM_SHA}")
        if pool_sha != EXPECTED_POOL_SHA:
            raise RuntimeError(f"pool sha mismatch {pool_sha} != {EXPECTED_POOL_SHA}")
        if tok_sha != EXPECTED_TOKENIZER_SHA:
            raise RuntimeError(f"tokenizer sha mismatch {tok_sha} != {EXPECTED_TOKENIZER_SHA}")
    else:
        stream_sha = sha256_file(STREAM_100M); pool_sha = sha256_file(POOL_10M); tok_sha = sha256_file(TOKENIZER_DIR / "tokenizer.json")

    print(json.dumps({"event": "freq_count_start", "pool": str(POOL_10M)}), flush=True)
    freqs = build_freqs(POOL_10M)
    print(json.dumps({"event": "freq_count_done", "types": len(freqs), "elapsed_sec": round(time.time()-t0, 1)}), flush=True)

    out_labels = OUT / "pvdm_tail_70M_100M_labels.jsonl"
    sample_path = OUT / "pvdm_tail_event_samples.jsonl"
    stats: Counter[str] = Counter()
    cat_counts: Counter[str] = Counter()
    target_class_counts: Counter[str] = Counter()
    target_freq_bins: Counter[str] = Counter()
    pivot_freq_bins: Counter[str] = Counter()
    ctrl_freq_bins: Counter[str] = Counter()
    dist_bins: Counter[str] = Counter()
    ctrl_dist_abs_diff: Counter[str] = Counter()
    ctrl_freq_diff: Counter[str] = Counter()
    ctrl_class_exact = 0
    samples: list[dict[str, Any]] = []
    target_norms: Counter[str] = Counter()
    control_norms: Counter[str] = Counter()
    pivot_norms: Counter[str] = Counter()

    with out_labels.open("w", encoding="utf-8") as out_f:
        for local_idx, (orig_row_idx, obj) in enumerate(stream_tail_rows(STREAM_100M, START_WORDS, TAIL_WORDS)):
            text = str(obj["text"])
            words = text.split()
            events, row_stats = events_for_row(words, freqs, args.max_events_per_row)
            stats.update(row_stats)
            targets = sorted({ev["target_i"] for ev in events})
            pivots = sorted({ev["pivot_i"] for ev in events})
            controls = sorted({ev["control_i"] for ev in events})
            rec = {
                "tail_row_idx": local_idx,
                "orig_row_idx": orig_row_idx,
                "example_id": int(obj.get("example_id", orig_row_idx)),
                "source": obj.get("source", ""),
                "words": int(obj.get("words", len(words))),
                "text": text,
                "pvdm_pivot_indices": pivots,
                "dependent_target_indices": targets,
                "control_anchor_indices": controls,
                "events": events,
            }
            out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            stats["rows"] += 1
            stats["words"] += len(words)
            stats["events"] += len(events)
            stats["target_positions_unique_sum"] += len(targets)
            stats["pivot_positions_unique_sum"] += len(pivots)
            stats["control_positions_unique_sum"] += len(controls)
            if events:
                stats["eligible_rows"] += 1
                if len(samples) < args.sample_limit:
                    samples.append({"tail_row_idx": local_idx, "orig_row_idx": orig_row_idx, "text": text, "events": events[:6]})
            for ev in events:
                cat_counts[ev["category"]] += 1
                target_class_counts[ev["target_class"]] += 1
                target_freq_bins[ev["target_freq_bin"]] += 1
                pivot_freq_bins[ev["pivot_freq_bin"]] += 1
                ctrl_freq_bins[ev["control_freq_bin"]] += 1
                dist_bins[ev["distance_bin"]] += 1
                target_norms[ev["target_norm"]] += 1
                control_norms[ev["control_norm"]] += 1
                pivot_norms[ev["pivot_norm"]] += 1
                cm = ev["control_match"]
                ctrl_dist_abs_diff[str(cm.get("distance_abs_diff", "NA"))] += 1
                ctrl_freq_diff[str(cm.get("freq_bin_abs_diff", "NA"))] += 1
                ctrl_class_exact += int(bool(cm.get("class_exact", False)))
            if (local_idx + 1) % 25000 == 0:
                print(json.dumps({"event": "rows", "rows": local_idx + 1, "words": stats["words"], "events": stats["events"], "elapsed_sec": round(time.time()-t0, 1)}), flush=True)

    with sample_path.open("w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    # Hash the target identity sequence used by both treatment and control to document exact equality.
    target_hash = hashlib.sha256()
    with out_labels.open(encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            payload = {"row": rec["tail_row_idx"], "targets": rec["dependent_target_indices"]}
            target_hash.update(json.dumps(payload, sort_keys=True).encode())
            target_hash.update(b"\n")
    label_sha = sha256_file(out_labels)
    sample_sha = sha256_file(sample_path)
    summary = {
        "status": "DETERMINISTIC_PVDM_LABELS_READY",
        "created_utc_unix": time.time(),
        "compliance": {
            "no_learned_parser_or_pos_tagger": True,
            "extractor_type": "deterministic fixed lexical/window rules plus in-corpus frequency bins",
            "frequency_source": str(POOL_10M),
            "frequency_source_sha256": pool_sha,
            "training_stream_source": str(STREAM_100M),
            "training_stream_sha256": stream_sha,
            "tokenizer_sha256": tok_sha,
            "official_faq_basis": "BabyLM FAQ says off-the-shelf POS taggers are not allowed and language-learned parsers count toward the word budget; spaCy pilot is therefore excluded from training labels.",
        },
        "tail": {"start_word_exposure": START_WORDS, "tail_words": TAIL_WORDS, "rows": stats["rows"]},
        "counts": dict(stats),
        "rates": {
            "eligible_row_fraction": stats["eligible_rows"] / max(1, stats["rows"]),
            "events_per_eligible_row": stats["events"] / max(1, stats["eligible_rows"]),
            "unique_targets_per_row": stats["target_positions_unique_sum"] / max(1, stats["rows"]),
            "target_positions_per_1000_words": 1000.0 * stats["target_positions_unique_sum"] / max(1, stats["words"]),
        },
        "events_by_category": dict(cat_counts),
        "target_class_distribution": dict(target_class_counts),
        "target_frequency_bins": dict(target_freq_bins),
        "pivot_frequency_bins": dict(pivot_freq_bins),
        "control_frequency_bins": dict(ctrl_freq_bins),
        "pivot_target_distance_bins": dict(dist_bins),
        "control_match_quality": {
            "class_exact_fraction": ctrl_class_exact / max(1, stats["events"]),
            "distance_abs_diff_counts": dict(ctrl_dist_abs_diff),
            "freq_bin_abs_diff_counts": dict(ctrl_freq_diff),
        },
        "target_distribution_equality": {
            "treatment_and_control_use_identical_dependent_target_indices": True,
            "target_identity_sequence_sha256": target_hash.hexdigest(),
            "top_targets": target_norms.most_common(40),
            "top_pivots": pivot_norms.most_common(40),
            "top_control_anchors": control_norms.most_common(40),
        },
        "files": {
            "labels_jsonl": str(out_labels),
            "labels_sha256": label_sha,
            "samples_jsonl": str(sample_path),
            "samples_sha256": sample_sha,
        },
        "runtime_sec": round(time.time() - t0, 2),
    }
    (OUT / "pvdm_label_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    note = f"""# research — PVDM compliance and deterministic control design

## Rule/provenance resolution

The research spaCy dependency pilot is not used to create training labels. The official BabyLM FAQ says that ancillary language-learned models count toward the 100M word budget and gives the concrete example that an off-the-shelf POS tagger cannot be used in the pipeline. The installed `en_core_web_sm` metadata reports OntoNotes 5, ClearNLP conversion guidelines, and WordNet sources; it is therefore an external language-learned parser/tagger for our purpose.

## Replacement extractor

This step uses only deterministic pattern rules plus word-frequency bins computed from the allowed compact 10M pool. There are no learned parser, tagger, NER, or external model outputs in the labels. The rule list is saved in `scripts/pvdm_deterministic_label_builder.py` and is inspectable submission-side code.

## Matched-control construction

For each accepted PVDM event, the target word index is identical in treatment and control. PVDM protects the true pivot from masking; the control protects a surrogate visible anchor chosen from the same row by deterministic class/frequency/distance scoring. Thus the dependent-target identity sequence is exactly shared by treatment and control, and the trainer will normalize masking probabilities so total masked mass is matched per batch.

## Tail label facts

- tail: {START_WORDS:,}→{START_WORDS + TAIL_WORDS:,} words from `{STREAM_100M}`
- rows: {stats['rows']:,}; words: {stats['words']:,}
- eligible rows: {stats['eligible_rows']:,} ({summary['rates']['eligible_row_fraction']:.3f})
- accepted events: {stats['events']:,}; unique target positions: {stats['target_positions_unique_sum']:,}
- target positions per 1000 words: {summary['rates']['target_positions_per_1000_words']:.2f}
- target identity sequence hash shared by both arms: `{summary['target_distribution_equality']['target_identity_sequence_sha256']}`
- label file SHA: `{label_sha}`

Full summary: `{OUT/'pvdm_label_summary.json'}`
Samples: `{sample_path}`
"""
    NOTE.write_text(note, encoding="utf-8")
    print(json.dumps({"status": summary["status"], "labels": str(out_labels), "summary": str(OUT/"pvdm_label_summary.json"), "events": stats["events"], "eligible_rows": stats["eligible_rows"], "target_hash": summary["target_distribution_equality"]["target_identity_sequence_sha256"], "runtime_sec": summary["runtime_sec"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
