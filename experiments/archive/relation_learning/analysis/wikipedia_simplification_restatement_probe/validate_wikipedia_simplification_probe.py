#!/usr/bin/env python3
"""Independent structural, provenance, and surface-holdout validation."""

from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import collections
import hashlib
import json
import os
import re
import time
from pathlib import Path


HERE = _public_path('experiments/archive/relation_learning/analysis/wikipedia_simplification_restatement_probe')
ROOT = _public_path('.')
HF_CACHE = _public_path('experiments/archive/relation_learning/analysis/wikipedia_simplification_restatement_probe/_hf_cache')
os.environ.setdefault("HF_HOME", str(HF_CACHE))
os.environ.setdefault("HF_DATASETS_CACHE", str(_public_path('experiments/archive/relation_learning/analysis/wikipedia_simplification_restatement_probe/_hf_cache/datasets')))

from datasets import load_dataset  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402


DATASET = "eilamc14/wikilarge-clean"
REVISION = "216fedb399e10141b390c8b89039737c934d43be"
TOKENIZER_DIR = _public_path('experiments/archive/frontier_consolidation/data/compliant_tokenizer')
POOL_DIR = _public_path('experiments/archive/frontier_consolidation/data/dose_2p64x_rowholdout_pools')
POOLS = {
    "clean": _public_path('experiments/archive/frontier_consolidation/data/dose_2p64x_rowholdout_pools/cleanqwen_lengthmatched_dose2p64x_10M.jsonl'),
    "view": _public_path('experiments/archive/frontier_consolidation/data/dose_2p64x_rowholdout_pools/compact_view_dose2p64x_10M.jsonl'),
    "repeat": _public_path('experiments/archive/frontier_consolidation/data/dose_2p64x_rowholdout_pools/compact_repeat_dose2p64x_10M.jsonl'),
}
NEUTRAL_ROWS = _public_path('experiments/archive/frontier_consolidation/data/dose_2p64x_rowholdout_pools/heldout_cleanqwen_rows.jsonl')
PAIR_FILE = _public_path('experiments/archive/relation_learning/analysis/wikipedia_simplification_restatement_probe/wikipedia_simplification_pairs.jsonl')
RECORD_FILE = _public_path('experiments/archive/relation_learning/analysis/wikipedia_simplification_restatement_probe/wikipedia_simplification_probe_records.jsonl')
STATS_FILE = _public_path('experiments/archive/relation_learning/analysis/wikipedia_simplification_restatement_probe/probe_stats.json')
SAMPLE_FILE = _public_path('experiments/archive/relation_learning/analysis/wikipedia_simplification_restatement_probe/manual_sample.md')
WORD_RE = re.compile(r"[^\W\d_]+(?:['’][^\W\d_]+)*", re.UNICODE)
STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "if", "then", "than", "that", "this", "these", "those",
    "of", "in", "on", "at", "to", "for", "from", "with", "without", "by", "as", "is", "are", "was",
    "were", "be", "been", "being", "it", "its", "they", "them", "their", "he", "she", "his", "her",
    "we", "you", "i", "not", "no", "do", "does", "did", "can", "could", "would", "should", "will",
    "have", "has", "had", "just", "so", "very", "also", "about", "more", "some", "any", "all", "each",
    "every", "both", "few", "many", "much", "such", "own", "other", "up", "out",
}


def normalize(text: str) -> str:
    return " ".join(str(text).replace("\u00a0", " ").split()).strip()


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def content_set(text: str) -> set[str]:
    return {
        m.group(0).lower().replace("’", "'") for m in WORD_RE.finditer(text)
        if len(m.group(0)) >= 2 and m.group(0).lower().replace("’", "'") not in STOPWORDS
    }


def expected_bin(jaccard: float) -> str:
    return "high" if jaccard >= 0.5 else "medium" if jaccard >= 0.25 else "low"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


class TokenAho:
    def __init__(self, patterns: dict[tuple[int, ...], set[str]]):
        self.go: list[dict[int, int]] = [{}]
        self.fail = [0]
        self.out = [set()]
        for pattern, labels in patterns.items():
            node = 0
            for token in pattern:
                if token not in self.go[node]:
                    self.go[node][token] = len(self.go)
                    self.go.append({}); self.fail.append(0); self.out.append(set())
                node = self.go[node][token]
            self.out[node].update(labels)
        queue = collections.deque(self.go[0].values())
        while queue:
            node = queue.popleft()
            for token, child in self.go[node].items():
                queue.append(child)
                fallback = self.fail[node]
                while fallback and token not in self.go[fallback]:
                    fallback = self.fail[fallback]
                self.fail[child] = self.go[fallback].get(token, 0)
                self.out[child].update(self.out[self.fail[child]])

    def scan(self, tokens: list[int]) -> set[str]:
        node = 0; found: set[str] = set()
        for token in tokens:
            while node and token not in self.go[node]:
                node = self.fail[node]
            node = self.go[node].get(token, 0)
            found.update(self.out[node])
        return found


def scan_exposure(pairs: list[dict], tokenizer) -> tuple[dict, dict]:
    patterns: dict[tuple[int, ...], set[str]] = collections.defaultdict(set)
    for pair in pairs:
        pid = pair["pair_id"]
        patterns[tuple(pair["target_token_ids"])].add(pid + "|target")
        patterns[tuple(pair["source_token_ids"] + pair["target_token_ids"])].add(pid + "|pair")
        patterns[tuple(pair["source_token_ids"])].add(pid + "|source")
    matcher = TokenAho(patterns)
    hits = {arm: {"target": set(), "pair": set(), "source": set()} for arm in POOLS}
    scanned = {}
    for arm, path in POOLS.items():
        nrows = nwords = 0; batch = []
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    text = str(json.loads(line)["text"]); batch.append(text)
                    nrows += 1; nwords += len(text.split())
                if len(batch) == 256:
                    for ids in tokenizer(batch, add_special_tokens=False)["input_ids"]:
                        for label in matcher.scan(ids):
                            _, kind = label.rsplit("|", 1); hits[arm][kind].add(label.rsplit("|", 1)[0])
                    batch = []
            if batch:
                for ids in tokenizer(batch, add_special_tokens=False)["input_ids"]:
                    for label in matcher.scan(ids):
                        _, kind = label.rsplit("|", 1); hits[arm][kind].add(label.rsplit("|", 1)[0])
        scanned[arm] = {"rows": nrows, "whitespace_words": nwords}
    return {
        arm: {kind: len(values) for kind, values in kinds.items()} for arm, kinds in hits.items()
    }, scanned


def main() -> None:
    started = time.time()
    pairs = load_jsonl(PAIR_FILE); records = load_jsonl(RECORD_FILE)
    stats = json.loads(STATS_FILE.read_text(encoding="utf-8"))
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR, local_files_only=True, use_fast=True)
    dataset = load_dataset(DATASET, revision=REVISION, cache_dir=str(_public_path('experiments/archive/relation_learning/analysis/wikipedia_simplification_restatement_probe/_hf_cache/datasets')))
    failures: list[str] = []

    def fail(message: str) -> None:
        if len(failures) < 250:
            failures.append(message)

    pair_ids = [p["pair_id"] for p in pairs]
    record_ids = [r["record_id"] for r in records]
    if len(pair_ids) != len(set(pair_ids)): fail("duplicate pair_id")
    if len(record_ids) != len(set(record_ids)): fail("duplicate record_id")
    if len({(p["dataset_split"], p["dataset_row_index"]) for p in pairs}) != len(pairs):
        fail("duplicate dataset provenance key")
    pair_map = {p["pair_id"]: p for p in pairs}

    neutral_rows = {str(x.get("example_id")): x for x in load_jsonl(NEUTRAL_ROWS)}
    pair_checks = collections.Counter()
    pair_token_cache = {}
    for p in pairs:
        pid = p["pair_id"]
        raw = dataset[p["dataset_split"]][int(p["dataset_row_index"])]
        if normalize(raw["source"]) != p["source_text"] or normalize(raw["target"]) != p["target_text"]:
            fail(f"dataset provenance mismatch {pid}")
        uid = p["unrelated_source_id"]
        uraw = dataset[p["unrelated_source_dataset_split"]][int(p["unrelated_source_dataset_row_index"])]
        if normalize(uraw["source"]) != p["unrelated_source_text"] or uid == pid:
            fail(f"U provenance/different-pair failure {pid}")
        src_ids = [int(x) for x in tokenizer(p["source_text"], add_special_tokens=False)["input_ids"]]
        tgt_enc = tokenizer(p["target_text"], add_special_tokens=False, return_offsets_mapping=True)
        tgt_ids = [int(x) for x in tgt_enc["input_ids"]]
        u_ids = [int(x) for x in tokenizer(p["unrelated_source_text"], add_special_tokens=False)["input_ids"]]
        n_ids = [int(x) for x in tokenizer(p["neutral_source_text"], add_special_tokens=False)["input_ids"]]
        if src_ids != p["source_token_ids"] or tgt_ids != p["target_token_ids"] or u_ids != p["unrelated_source_token_ids"] or n_ids != p["neutral_source_token_ids"]:
            fail(f"stored tokenization mismatch {pid}")
        if len(src_ids) != len(u_ids) or len(src_ids) != len(n_ids) or len(src_ids) != p["source_token_len"]:
            fail(f"T/U/N source-slot token length mismatch {pid}")
        source_content = content_set(p["source_text"]); target_content = content_set(p["target_text"])
        shared = source_content & target_content; novel = target_content - source_content
        jac = len(shared) / len(source_content | target_content)
        if sorted(shared) != p["overlap_words"] or sorted(novel) != p["nonoverlap_words"]:
            fail(f"content-overlap mismatch {pid}")
        if abs(jac - float(p["content_jaccard"])) > 1e-12 or expected_bin(jac) != p["overlap_bin"]:
            fail(f"Jaccard/bin mismatch {pid}")
        if float(p["semantic_similarity"]) < 0.65 or float(p["nli_entailment_probability"]) < 0.80 or p["nli_predicted_label"] != "entailment":
            fail(f"semantic gate failure {pid}")
        u_overlap = sorted(target_content & content_set(p["unrelated_source_text"]))
        n_overlap = sorted(target_content & content_set(p["neutral_source_text"]))
        if u_overlap != p["unrelated_target_content_overlap_words"] or u_overlap:
            fail(f"U target-content overlap {pid}: {u_overlap}")
        if n_overlap != p["neutral_target_content_overlap_words"] or n_overlap:
            fail(f"N target-content overlap {pid}: {n_overlap}")
        neutral_raw = neutral_rows.get(str(p["neutral_row_example_id"]))
        if neutral_raw is None or p["neutral_source_text"] not in normalize(neutral_raw["text"]):
            fail(f"N row provenance mismatch {pid}")
        if neutral_raw is not None and p["neutral_row_source"] != neutral_raw.get("source"):
            fail(f"N source label mismatch {pid}")
        pair_token_cache[pid] = (source_content, target_content, tgt_enc, tgt_ids)
        pair_checks[p["overlap_bin"]] += 1

    by_target: dict[str, list[dict]] = collections.defaultdict(list)
    per_pair_classes = collections.Counter()
    condition_counts = collections.Counter()
    vocab = len(tokenizer); specials = set(tokenizer.all_special_ids)
    for r in records:
        pid = r["pair_id"]; pair = pair_map.get(pid)
        if pair is None:
            fail(f"orphan record {r['record_id']}"); continue
        source_content, target_content, tgt_enc, tgt_ids = pair_token_cache[pid]
        by_target[r["target_key"]].append(r)
        per_pair_classes[(pid, r["token_class"], r["condition_code"])] += 1
        condition_counts[(r["overlap_bin"], r["token_class"], r["condition_code"])] += 1
        slots = {
            "T": ("true_source", pair["pair_id"], pair["source_token_ids"]),
            "U": ("unrelated_source", pair["unrelated_source_id"], pair["unrelated_source_token_ids"]),
            "N": ("neutral_ordinary", pair["neutral_source_id"], pair["neutral_source_token_ids"]),
        }
        if r["condition_code"] not in slots:
            fail(f"bad condition {r['record_id']}"); continue
        condition, source_id, source_ids = slots[r["condition_code"]]
        if r["condition"] != condition or r["source_slot_id"] != source_id:
            fail(f"condition/source id mismatch {r['record_id']}")
        canonical = [tokenizer.bos_token_id] + source_ids + pair["target_token_ids"] + [tokenizer.eos_token_id]
        restored = list(r["input_ids"]); pos = int(r["mask_position"]); target_id = int(r["target_token_id"])
        if not (0 <= pos < len(restored)) or restored[pos] != tokenizer.mask_token_id or restored.count(tokenizer.mask_token_id) != 1:
            fail(f"mask failure {r['record_id']}"); continue
        restored[pos] = target_id
        if restored != canonical:
            fail(f"canonical reconstruction failure {r['record_id']}")
        if pos != 1 + len(source_ids) + int(r["target_token_index"]):
            fail(f"target position shift {r['record_id']}")
        if len(r["input_ids"]) != r["seq_len"] or len(r["attention_mask"]) != r["seq_len"] or any(x != 1 for x in r["attention_mask"]) or r["seq_len"] > 256:
            fail(f"sequence length/attention failure {r['record_id']}")
        if target_id < 0 or target_id >= vocab or target_id in specials:
            fail(f"invalid target id {r['record_id']}")
        word = pair["target_text"][int(r["target_char_start"]):int(r["target_char_end"])].lower().replace("’", "'")
        if word != r["target_word"] or word not in target_content:
            fail(f"target character provenance failure {r['record_id']}")
        covering = [i for i, (a, b) in enumerate(tgt_enc["offset_mapping"]) if a < int(r["target_char_end"]) and b > int(r["target_char_start"])]
        if covering != [int(r["target_token_index"])] or tgt_ids[int(r["target_token_index"])] != target_id:
            fail(f"target not an exact single token {r['record_id']}")
        expected_class = "overlap" if target_id in set(pair["source_token_ids"]) else "nonoverlap"
        if r["token_class"] != expected_class:
            fail(f"target class failure {r['record_id']}")
        expected_surface_class = "overlap" if word in source_content else "nonoverlap"
        if r.get("surface_word_class") != expected_surface_class:
            fail(f"target surface-word class failure {r['record_id']}")

    for key, group in by_target.items():
        if {r["condition_code"] for r in group} != {"T", "U", "N"} or len(group) != 3:
            fail(f"unaligned T/U/N target {key}")
        invariant = {(r["pair_id"], r["target_token_id"], r["target_word"], r["target_token_index"], r["token_class"], r["seq_len"], r["mask_position"]) for r in group}
        if len(invariant) != 1:
            fail(f"T/U/N target metadata drift {key}")
    for pid in pair_ids:
        expected = {("overlap", "T"): 1, ("overlap", "U"): 1, ("overlap", "N"): 1,
                    ("nonoverlap", "T"): 2, ("nonoverlap", "U"): 2, ("nonoverlap", "N"): 2}
        for (target_class, condition), count in expected.items():
            if per_pair_classes[(pid, target_class, condition)] != count:
                fail(f"per-pair denominator failure {pid} {target_class} {condition}")

    exposure_hits, scanned = scan_exposure(pairs, tokenizer)
    if any(exposure_hits[a][k] for a in POOLS for k in ("target", "pair")):
        fail("target or full T-body surface exposure found")
    stats_counts = stats["record_counts_by_bin_class_condition"]
    recomputed_counts = {"|".join(key): value for key, value in sorted(condition_counts.items())}
    if stats_counts != recomputed_counts or stats["selected_pairs"] != len(pairs) or stats["records"] != len(records):
        fail("probe_stats count mismatch")
    sample_bins = collections.Counter()
    for line in SAMPLE_FILE.read_text(encoding="utf-8").splitlines():
        if line.startswith("| wikilarge_"):
            # Pair lookup avoids trusting headings in the rendered table.
            pid = line.split("|", 2)[1].strip(); sample_bins[pair_map[pid]["overlap_bin"]] += 1
    if sum(sample_bins.values()) < 25 or any(sample_bins[b] == 0 for b in ("low", "medium", "high")):
        fail("manual sample lacks >=25 examples spanning all bins")

    result = {
        "status": "PASS" if not failures else "FAIL",
        "validated_local": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "pairs": len(pairs), "records": len(records), "paired_targets": len(by_target),
        "pair_bins": dict(pair_checks), "record_counts": recomputed_counts,
        "manual_sample_pairs_by_bin": dict(sample_bins),
        "surface_exposure_hits": exposure_hits, "surface_scan_counts": scanned,
        "checks": {
            "unique_pair_ids": len(pair_ids) == len(set(pair_ids)),
            "unique_record_ids": len(record_ids) == len(set(record_ids)),
            "dataset_rows_reverified": True,
            "all_targets_TUN_aligned": all(len(v) == 3 and {r["condition_code"] for r in v} == {"T", "U", "N"} for v in by_target.values()),
            "T_U_N_exact_source_token_length": all(p["source_token_len"] == len(p["unrelated_source_token_ids"]) == len(p["neutral_source_token_ids"]) for p in pairs),
            "all_U_from_different_pair": all(p["pair_id"] != p["unrelated_source_id"] for p in pairs),
            "all_U_target_content_disjoint": all(not p["unrelated_target_content_overlap_words"] for p in pairs),
            "all_N_target_content_disjoint": all(not p["neutral_target_content_overlap_words"] for p in pairs),
            "all_selected_directionally_entailed_at_threshold": all(float(p["nli_entailment_probability"]) >= 0.80 for p in pairs),
            "target_and_full_pair_surface_unexposed_all_arms": not any(exposure_hits[a][k] for a in POOLS for k in ("target", "pair")),
        },
        "files_sha256": {path.name: sha256(path) for path in (PAIR_FILE, RECORD_FILE, STATS_FILE, SAMPLE_FILE)},
        "failure_count": len(failures), "failures": failures,
        "elapsed_seconds": time.time() - started,
    }
    (_public_path('experiments/archive/relation_learning/analysis/wikipedia_simplification_restatement_probe/validation_results.json')).write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
