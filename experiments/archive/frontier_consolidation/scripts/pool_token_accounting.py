#!/usr/bin/env python3
"""research: whole-pool token/WWM accounting for every 2.64x MAX dose arm.

The research changed-block audit revealed that compact rewrites have higher
legal16k fertility (about 1.529 tokens/word) than source sentences (1.385) or
independently sampled FineWeb sentences (1.431).  Because the BabyLM budget is
counted in words, arms that are exactly word-matched are not automatically
token-matched: the view arm can process more subword targets per row than the
repeat, clean, or breadth arms.

This script measures, over each complete 10M pool (all rows, not only the
changed block), under the exact trainer tokenization/WWM semantics:

  * total legal16k tokens and visible tokens after truncation to seq_length,
  * total whole-word-masking groups, i.e. total mask opportunity,
  * truncation loss,
  * derived per-word and per-arm relative shifts against the view arm.

The result states how token-matched the dose instrument actually is, which is a
condition on every leg of view-clean = (view-repeat) + (repeat-clean) and on
view-breadth.  CPU only: no training, GPU evaluation, SuperGLUE, AoA, upload, or
leaderboard action.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import json
import pathlib
import time
from typing import Any

from transformers import AutoTokenizer


def find_user_root() -> pathlib.Path:
    return _PUBLIC_ROOT


ROOT = find_user_root()
WS = ROOT / "experiments/archive/frontier_consolidation"
TOKENIZER_DIR = WS / "data/compliant_tokenizer"
MAXD = WS / "data/dose_2p64x_rowholdout_pools"
BRD = WS / "data/dose_2p64x_breadth_rowholdout_pools"
MID = WS / "data/dose_1p82x_rowholdout_pools"
OUT_DIR = WS / "data/pool_token_accounting"
SEQ_LENGTH = 256

POOLS = {
    "max_view": MAXD / "compact_view_dose2p64x_10M.jsonl",
    "max_repeat": MAXD / "compact_repeat_dose2p64x_10M.jsonl",
    "max_clean": MAXD / "cleanqwen_lengthmatched_dose2p64x_10M.jsonl",
    "max_breadth": BRD / "compact_breadth_dose2p64x_10M.jsonl",
}
OPTIONAL_POOLS = {
    "mid_view": MID / "compact_view_dose1p82x_10M.jsonl",
    "mid_repeat": MID / "compact_repeat_dose1p82x_10M.jsonl",
}


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def rel(p: pathlib.Path) -> str:
    try:
        return str(p.relative_to(ROOT))
    except ValueError:
        return str(p)


class Geo:
    def __init__(self, tokenizer, seq_length: int) -> None:
        self.tok = tokenizer
        self.seq = seq_length
        self.special = set(tokenizer.all_special_ids)
        self._ws: dict[int, bool] = {}

    def word_start(self, tid: int) -> bool:
        v = self._ws.get(tid)
        if v is None:
            s = self.tok.convert_ids_to_tokens(int(tid))
            v = bool(s is not None and (str(s).startswith("\u0120") or str(s).startswith("\u2581")))
            self._ws[tid] = v
        return v

    def pool(self, path: pathlib.Path, batch: int = 512) -> dict[str, Any]:
        rows = 0
        words = 0
        tokens_full = 0
        tokens_visible = 0
        groups = 0
        truncated_rows = 0
        tokens_dropped = 0
        buf: list[str] = []

        def flush(texts: list[str]) -> None:
            nonlocal tokens_full, tokens_visible, groups, truncated_rows, tokens_dropped
            if not texts:
                return
            enc = self.tok(texts, add_special_tokens=False)["input_ids"]
            for ids in enc:
                tokens_full += len(ids)
                vis = ids[: self.seq]
                tokens_visible += len(vis)
                if len(ids) > self.seq:
                    truncated_rows += 1
                    tokens_dropped += len(ids) - self.seq
                gid = -1
                for i, tid in enumerate(vis):
                    if tid in self.special:
                        continue
                    if gid < 0 or self.word_start(int(tid)) or i == 0:
                        gid += 1
                        groups += 1

        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                text = str(obj.get("text") or "")
                rows += 1
                words += len(text.split())
                buf.append(text)
                if len(buf) >= batch:
                    flush(buf)
                    buf = []
        flush(buf)
        return {
            "path": rel(path),
            "rows": rows,
            "words": words,
            "tokens_legal16k": tokens_full,
            "tokens_visible_seq256": tokens_visible,
            "wwm_groups_visible": groups,
            "truncated_rows": truncated_rows,
            "tokens_dropped_by_truncation": tokens_dropped,
            "tokens_per_word": (tokens_full / words) if words else None,
            "visible_tokens_per_word": (tokens_visible / words) if words else None,
            "wwm_groups_per_word": (groups / words) if words else None,
            "expected_masked_groups_at_p015": 0.15 * groups,
        }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    ap.add_argument("--include-midpoint", action="store_true")
    args = ap.parse_args()
    out_dir = pathlib.Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(str(TOKENIZER_DIR), use_fast=True)
    geo = Geo(tok, SEQ_LENGTH)

    targets = dict(POOLS)
    if args.include_midpoint:
        for k, p in OPTIONAL_POOLS.items():
            if p.exists():
                targets[k] = p

    results: dict[str, Any] = {}
    for name, path in targets.items():
        if not path.exists():
            results[name] = {"path": rel(path), "exists": False}
            continue
        t0 = time.time()
        results[name] = geo.pool(path)
        results[name]["elapsed_sec"] = round(time.time() - t0, 1)
        print(json.dumps({name: results[name]}, ensure_ascii=False), flush=True)

    ref = results.get("max_view") or {}
    shifts: dict[str, Any] = {}
    for name, rec in results.items():
        if name == "max_view" or not rec.get("rows"):
            continue
        row: dict[str, Any] = {}
        for key in ["tokens_legal16k", "tokens_visible_seq256", "wwm_groups_visible", "words"]:
            a = ref.get(key)
            b = rec.get(key)
            if isinstance(a, (int, float)) and isinstance(b, (int, float)) and a:
                row[f"{key}_minus_view"] = b - a
                row[f"{key}_pct_vs_view"] = 100.0 * (b - a) / a
        shifts[name] = row

    payload = {
        "status": "POOL_TOKEN_ACCOUNTING_DONE",
        "created_utc": now(),
        "tokenizer": rel(TOKENIZER_DIR),
        "seq_length": SEQ_LENGTH,
        "wwm_mask_prob": 0.15,
        "pools": results,
        "relative_shift_vs_max_view": shifts,
        "interpretation": [
            "BabyLM budget is counted in words, so exactly word-matched arms are not automatically token-matched.",
            "Compact rewrites tokenize less efficiently than raw sentences under the fixed legal16k tokenizer, so the view arm carries more subword targets per row than repeat/clean/breadth.",
            "Any view-repeat, view-clean, or view-breadth reading therefore includes a small systematic token/compute asymmetry that must be stated with the effect size.",
        ],
        "no_training_gpu_eval_superglue_aoa_upload_or_leaderboard": True,
    }
    (out_dir / "pool_token_accounting.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    lines = [
        "# research whole-pool token / WWM accounting for the 2.64x dose arms",
        "",
        f"Fixed research legal16k tokenizer; seq_length {SEQ_LENGTH}; WWM p=0.15; complete 10M pools.",
        "",
        "| arm | rows | words | legal16k tokens | visible tokens | WWM groups | tokens/word | groups/word |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for name, rec in results.items():
        if not rec.get("rows"):
            lines.append(f"| {name} | missing | | | | | | |")
            continue
        lines.append(
            f"| {name} | {rec['rows']} | {rec['words']} | {rec['tokens_legal16k']} | {rec['tokens_visible_seq256']} | "
            f"{rec['wwm_groups_visible']} | {rec['tokens_per_word']:.5f} | {rec['wwm_groups_per_word']:.5f} |"
        )
    lines += ["", "## Relative shift versus the MAX view arm", "", "| arm | tokens % | visible tokens % | WWM groups % | word delta |", "|---|---|---|---|---|"]
    for name, row in shifts.items():
        lines.append(
            f"| {name} | {row.get('tokens_legal16k_pct_vs_view'):+.4f} | {row.get('tokens_visible_seq256_pct_vs_view'):+.4f} | "
            f"{row.get('wwm_groups_visible_pct_vs_view'):+.4f} | {row.get('words_minus_view')} |"
        )
    lines += ["", "## Interpretation", ""] + [f"- {s}" for s in payload["interpretation"]]
    (out_dir / "pool_token_accounting.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "relative_shift_vs_max_view": shifts, "out": rel(out_dir)}, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
