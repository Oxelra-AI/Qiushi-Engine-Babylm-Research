#!/usr/bin/env python3
"""research: lightweight inventory of downloaded structured role sources.

Compares available structured sources for role-equivariant world-pair construction.
No training. Outputs only source-shape evidence and small samples.
"""
from __future__ import annotations

import csv
import gzip
import io
import json
import tarfile
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path("experiments/archive/representation_and_objectives")
OUT_DIR = ROOT / "data/world_pair_source_feasibility"
OUT_DIR.mkdir(parents=True, exist_ok=True)
DOWNLOADS = ROOT / "staging/downloads"

CODEx_TARS = [
    DOWNLOADS / "3d83e17314e5_codex-s.tar.gz",
    DOWNLOADS / "c2ccc659629a_codex-m.tar.gz",
    DOWNLOADS / "2ccc0fc6b95f_codex-l.tar.gz",
]
KG_CAL = DOWNLOADS / "8ff6bbd2eb24_kg-calibration.zip"


def safe_read_text_from_tar(tf: tarfile.TarFile, member_name_contains: str, max_bytes: int = 5_000_000):
    members = [m for m in tf.getmembers() if member_name_contains in m.name and m.isfile()]
    if not members:
        return None, None
    m = members[0]
    f = tf.extractfile(m)
    if f is None:
        return m.name, None
    data = f.read(max_bytes)
    return m.name, data.decode("utf-8", errors="replace")


def codex_inventory(path: Path):
    if not path.exists():
        return {"path": str(path), "exists": False}
    res = {"path": str(path), "exists": True, "members": [], "files": {}}
    with tarfile.open(path, "r:gz") as tf:
        res["members"] = [m.name for m in tf.getmembers()[:80]]
        # CoDEx archives commonly contain triples/train.txt and entities.json/relations.json.
        for key in ["entities", "relations", "train", "valid", "test"]:
            candidates = [m for m in tf.getmembers() if key in m.name.lower() and m.isfile()]
            res["files"][key] = [m.name for m in candidates[:5]]
        triple_file = None
        for m in tf.getmembers():
            low = m.name.lower()
            if m.isfile() and (low.endswith("train.txt") or low.endswith("triples.txt")):
                triple_file = m
                break
        if triple_file:
            data = tf.extractfile(triple_file).read(2_000_000).decode("utf-8", errors="replace")
            lines = [ln.strip() for ln in data.splitlines() if ln.strip()]
            triples = []
            for ln in lines[:20000]:
                parts = ln.split("\t") if "\t" in ln else ln.split()
                if len(parts) >= 3:
                    triples.append(tuple(parts[:3]))
            rel_counts = Counter(r for _, r, _ in triples)
            directed = {(h, r, t) for h, r, t in triples}
            reciprocal_same_rel = sum(1 for h, r, t in directed if (t, r, h) in directed) // 2
            res["sample_triple_file"] = triple_file.name
            res["sample_triples_scanned"] = len(triples)
            res["sample_relation_count"] = len(rel_counts)
            res["sample_top_relations"] = rel_counts.most_common(12)
            res["sample_reciprocal_same_relation_pairs"] = reciprocal_same_rel
            res["sample_triples"] = [list(x) for x in triples[:8]]
    return res


def kg_cal_inventory(path: Path):
    if not path.exists():
        return {"path": str(path), "exists": False}
    res = {"path": str(path), "exists": True, "members": [], "candidate_files": {}}
    with zipfile.ZipFile(path) as zf:
        names = zf.namelist()
        res["members"] = names[:120]
        for key in ["wikidata", "entities", "relations", "triples", "train", "test"]:
            res["candidate_files"][key] = [n for n in names if key in n.lower()][:10]
        # Try readable TSV/CSV triple-like files.
        samples = []
        for n in names:
            low = n.lower()
            if not (low.endswith(".tsv") or low.endswith(".csv") or low.endswith(".txt")):
                continue
            if not any(k in low for k in ["triple", "train", "valid", "test"]):
                continue
            try:
                data = zf.read(n)[:2_000_000].decode("utf-8", errors="replace")
            except Exception:
                continue
            lines = [ln.strip() for ln in data.splitlines() if ln.strip()]
            triples = []
            for ln in lines[:20000]:
                parts = ln.split("\t") if "\t" in ln else ln.split(",") if "," in ln else ln.split()
                if len(parts) >= 3:
                    triples.append(tuple(parts[:3]))
            if triples:
                rel_counts = Counter(r for _, r, _ in triples)
                directed = {(h, r, t) for h, r, t in triples}
                reciprocal_same_rel = sum(1 for h, r, t in directed if (t, r, h) in directed) // 2
                samples.append({
                    "file": n,
                    "triples_scanned": len(triples),
                    "relation_count": len(rel_counts),
                    "top_relations": rel_counts.most_common(8),
                    "reciprocal_same_relation_pairs": reciprocal_same_rel,
                    "sample_triples": [list(x) for x in triples[:5]],
                })
                if len(samples) >= 8:
                    break
        res["triple_like_samples"] = samples
    return res


def main():
    inv = {
        "status": "STRUCTURED_SOURCE_INVENTORY",
        "codex": [codex_inventory(p) for p in CODEx_TARS],
        "kg_calibration": kg_cal_inventory(KG_CAL),
        "interpretation": {
            "sports_records": "provide repeated-event same-participant role flips with explicit dates/scores; narrow but clean world pairs",
            "knowledge_graph_static_triples": "usually provide asymmetric positives and negatives, but same entity-pair role-reversed positives are rare or absent unless temporal/event qualifiers exist; useful for relation vocabulary and negative probes, not automatically for world-pair role reassignment",
            "nli_role_swap_datasets": "provide hard role-swap negatives from a source context; useful as verification probes but not by themselves paired worlds unless a mate context with reversed true relation is available",
        },
    }
    out = OUT_DIR / "structured_source_inventory.json"
    out.write_text(json.dumps(inv, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({
        "status": inv["status"],
        "codex_archives": len(inv["codex"]),
        "kg_calibration_exists": inv["kg_calibration"].get("exists"),
        "out": str(out),
    }, indent=2))

if __name__ == "__main__":
    main()
