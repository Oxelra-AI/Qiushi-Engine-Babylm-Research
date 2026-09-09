#!/usr/bin/env python3
"""research: token-frequency concentration probe for source-recurring effects.

functional_learning asked whether the BabyLM source-recurring gains concentrate on rare or
low-exposure lexical rows, given that the models tie input embeddings to output
readouts.  This CPU script tests a simple lexical-row explanation for the COMPACT_EXPERIENCE
ALN-vs-OFF Wikipedia effect: count target-token frequencies in the 10M training
pools and correlate those counts with per-target ALN-OFF source-use changes.

The result cannot identify internal circuits.  It can only say whether the large
source-recurring effect is visibly concentrated in low-frequency tokens or in
tokens whose marginal frequency increased most in the aligned-Qwen pool.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import csv
import json
import math
import pathlib
import time
from collections import Counter
from typing import Any

import numpy as np
import pandas as pd
from transformers import AutoTokenizer

ROOT0 = _public_path('experiments/archive/relation_learning/scripts/frequency_concentration_probe.py')
ROOT = _PUBLIC_ROOT

WS = ROOT / "experiments/archive/relation_learning"
COMPACT_EXPERIENCE = ROOT / "experiments/archive/compact_experience"
TOKENIZER = COMPACT_EXPERIENCE / "training/runs/official_lengthmatched_16k_seed43122/hf_model"
# The path above is intentionally corrected below if running from ROOT.
TOKENIZER = ROOT / "experiments/archive/compact_experience/training/runs/official_lengthmatched_16k_seed43122/hf_model"
POOLS = {
    "OFF": ROOT / "experiments/archive/compact_experience/data/qwen_clean_aligned/training_corpora/official_only_10M.jsonl",
    "ALN": ROOT / "experiments/archive/compact_experience/data/qwen_clean_aligned/training_corpora/qwen_aligned_10M.jsonl",
}
WIKI_RUNS = {
    "seed43022": WS / "data/paired_context_relation_design_probe/wikipedia_target_TUN_rows.csv",
    "seed43122": WS / "data/paired_context_aln_off_seed43122_probe/wikipedia_target_TUN_rows.csv",
}
OUT = WS / "data/frequency_concentration_probe"
NOTE = (_PUBLIC_ROOT / 'research/notes/relation_learning/frequency_concentration_probe.md')


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def rel(path: pathlib.Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except Exception:
        return str(path)


def read_jsonl_texts(path: pathlib.Path):
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            yield str(json.loads(line).get("text") or "")


def count_pool_tokens(tok, path: pathlib.Path, batch_size: int = 512) -> dict[str, Any]:
    counts: Counter[int] = Counter()
    rows = 0
    words = 0
    batch: list[str] = []
    def flush():
        nonlocal batch
        if not batch:
            return
        enc = tok(batch, add_special_tokens=False)["input_ids"]
        for ids in enc:
            counts.update(int(x) for x in ids)
        batch = []
    for text in read_jsonl_texts(path):
        rows += 1
        words += len(text.split())
        batch.append(text)
        if len(batch) >= batch_size:
            flush()
    flush()
    return {"rows": rows, "words": words, "total_tokens": int(sum(counts.values())), "counts": counts}


def se(vals: pd.Series) -> float:
    x = pd.Series(vals, dtype=float).dropna()
    return float(x.std(ddof=0) / math.sqrt(len(x))) if len(x) > 1 else float("nan")


def corr(x: pd.Series, y: pd.Series) -> float:
    xx = pd.Series(x, dtype=float)
    yy = pd.Series(y, dtype=float)
    ok = np.isfinite(xx.to_numpy()) & np.isfinite(yy.to_numpy())
    if ok.sum() < 3 or float(np.std(xx.to_numpy()[ok])) == 0.0 or float(np.std(yy.to_numpy()[ok])) == 0.0:
        return float("nan")
    return float(np.corrcoef(xx.to_numpy()[ok], yy.to_numpy()[ok])[0, 1])


def load_wiki_effects() -> pd.DataFrame:
    frames = []
    for seed, path in WIKI_RUNS.items():
        df = pd.read_csv(path)
        df = df[df["role"].isin(["ALN", "OFF"])].copy()
        # Average late checkpoints per target within role.
        g = df.groupby(["role", "pair_id", "target_key", "token_class", "overlap_bin", "target_token_id", "target_word"], dropna=False).agg(
            n_checkpoints=("checkpoint", "nunique"),
            T=("T", "mean"), U=("U", "mean"), N=("N", "mean"),
            gain_T_vs_N=("gain_T_vs_N", "mean"), gain_U_vs_N=("gain_U_vs_N", "mean"), gain_T_vs_U=("gain_T_vs_U", "mean"),
        ).reset_index()
        g = g[g["n_checkpoints"] == 3].copy()
        aln = g[g.role == "ALN"].drop(columns=["role"]).add_suffix("_ALN")
        off = g[g.role == "OFF"].drop(columns=["role"]).add_suffix("_OFF")
        key_cols = ["pair_id", "target_key", "token_class", "overlap_bin", "target_token_id", "target_word"]
        left = aln.rename(columns={f"{c}_ALN": c for c in key_cols})
        right = off.rename(columns={f"{c}_OFF": c for c in key_cols})
        m = left.merge(right, on=key_cols)
        m["seed"] = seed
        for col in ["T", "U", "N", "gain_T_vs_N", "gain_U_vs_N", "gain_T_vs_U"]:
            m[f"delta_{col}_ALNminusOFF"] = m[f"{col}_ALN"] - m[f"{col}_OFF"]
        frames.append(m)
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(str(TOKENIZER), use_fast=True)
    pool_infos = {}
    freq_rows = []
    for role, path in POOLS.items():
        t0 = time.time()
        info = count_pool_tokens(tok, path)
        counts = info.pop("counts")
        pool_infos[role] = {**info, "path": rel(path), "elapsed_sec": round(time.time() - t0, 1)}
        for tid, c in counts.items():
            freq_rows.append({"role": role, "target_token_id": int(tid), "freq": int(c)})
        print(json.dumps({role: pool_infos[role]}, ensure_ascii=False), flush=True)
    freq = pd.DataFrame(freq_rows).pivot(index="target_token_id", columns="role", values="freq").fillna(0).reset_index()
    for role in ["OFF", "ALN"]:
        if role not in freq.columns:
            freq[role] = 0
    freq["freq_OFF"] = freq["OFF"].astype(float)
    freq["freq_ALN"] = freq["ALN"].astype(float)
    freq["delta_freq_ALNminusOFF"] = freq["freq_ALN"] - freq["freq_OFF"]
    freq["log1p_OFF"] = np.log1p(freq["freq_OFF"])
    freq["log1p_ALN"] = np.log1p(freq["freq_ALN"])
    freq["delta_log1p_ALNminusOFF"] = freq["log1p_ALN"] - freq["log1p_OFF"]
    freq[["target_token_id", "freq_OFF", "freq_ALN", "delta_freq_ALNminusOFF", "log1p_OFF", "log1p_ALN", "delta_log1p_ALNminusOFF"]].to_csv(OUT / "paired_context_off_aln_token_frequencies.csv", index=False)

    eff = load_wiki_effects()
    merged = eff.merge(freq[["target_token_id", "freq_OFF", "freq_ALN", "delta_freq_ALNminusOFF", "log1p_OFF", "log1p_ALN", "delta_log1p_ALNminusOFF"]], on="target_token_id", how="left")
    # Token IDs absent from a 10M pool are genuine zero-frequency rows, not missing data.
    for col in ["freq_OFF", "freq_ALN", "delta_freq_ALNminusOFF", "log1p_OFF", "log1p_ALN", "delta_log1p_ALNminusOFF"]:
        merged[col] = merged[col].fillna(0.0)
    merged.to_csv(OUT / "aln_off_wikipedia_target_effects_with_frequency.csv", index=False)

    summaries = []
    for (seed, token_class), sub in merged.groupby(["seed", "token_class"], dropna=False):
        y = sub["delta_gain_T_vs_N_ALNminusOFF"]
        summaries.append({
            "seed": seed,
            "token_class": token_class,
            "n_targets": int(len(sub)),
            "mean_delta_gain_T_vs_N": float(y.mean()),
            "se_delta_gain_T_vs_N": se(y),
            "corr_with_log1p_OFF_freq": corr(sub["log1p_OFF"], y),
            "corr_with_delta_log1p_freq": corr(sub["delta_log1p_ALNminusOFF"], y),
            "corr_with_ALN_log1p_freq": corr(sub["log1p_ALN"], y),
            "mean_OFF_freq": float(sub["freq_OFF"].mean()),
            "median_OFF_freq": float(sub["freq_OFF"].median()),
            "mean_delta_freq": float(sub["delta_freq_ALNminusOFF"].mean()),
            "median_delta_freq": float(sub["delta_freq_ALNminusOFF"].median()),
            "frac_delta_freq_positive": float((sub["delta_freq_ALNminusOFF"] > 0).mean()),
        })
    summ = pd.DataFrame(summaries)
    summ.to_csv(OUT / "frequency_effect_correlations_by_seed_class.csv", index=False)

    # Frequency quartiles pooled within each seed/class by OFF frequency.
    qrows = []
    for (seed, token_class), sub in merged.groupby(["seed", "token_class"], dropna=False):
        sub = sub.copy()
        try:
            sub["off_freq_quartile"] = pd.qcut(sub["freq_OFF"].rank(method="first"), 4, labels=["Q1_low", "Q2", "Q3", "Q4_high"])
        except Exception:
            sub["off_freq_quartile"] = "all"
        for q, g in sub.groupby("off_freq_quartile", dropna=False):
            qrows.append({
                "seed": seed,
                "token_class": token_class,
                "off_freq_quartile": str(q),
                "n_targets": int(len(g)),
                "mean_delta_gain_T_vs_N": float(g["delta_gain_T_vs_N_ALNminusOFF"].mean()),
                "se_delta_gain_T_vs_N": se(g["delta_gain_T_vs_N_ALNminusOFF"]),
                "mean_OFF_freq": float(g["freq_OFF"].mean()),
                "mean_delta_freq": float(g["delta_freq_ALNminusOFF"].mean()),
            })
    qdf = pd.DataFrame(qrows)
    qdf.to_csv(OUT / "frequency_quartile_effects.csv", index=False)

    # Two-seed mean correlations by class.
    class_rows = []
    for token_class, sub in summ.groupby("token_class"):
        class_rows.append({
            "token_class": token_class,
            "n_seed_rows": int(len(sub)),
            "mean_effect": float(sub["mean_delta_gain_T_vs_N"].mean()),
            "mean_corr_log1p_OFF_freq": float(sub["corr_with_log1p_OFF_freq"].mean()),
            "mean_corr_delta_log1p_freq": float(sub["corr_with_delta_log1p_freq"].mean()),
            "mean_corr_ALN_log1p_freq": float(sub["corr_with_ALN_log1p_freq"].mean()),
        })
    class_df = pd.DataFrame(class_rows)
    class_df.to_csv(OUT / "frequency_effect_correlations_two_seed_summary.csv", index=False)

    def tab(df: pd.DataFrame, cols: list[str]) -> str:
        lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
        for r in df.to_dict("records"):
            vals = []
            for c in cols:
                v = r[c]
                if isinstance(v, float):
                    vals.append(f"{v:+.4f}" if math.isfinite(v) else "NA")
                else:
                    vals.append(str(v))
            lines.append("| " + " | ".join(vals) + " |")
        return "\n".join(lines)

    note_lines = [
        "# research token-frequency concentration probe\n",
        f"Created: {now()}\n\n",
        "This probe responds to functional_learning's tied-softmax mechanism question. The representative DeBERTa/RoBERTa/GPT models tie input embeddings and output readout rows, so a lexical-row path is possible. Here I test a simple version for the COMPACT_EXPERIENCE ALN-vs-OFF Wikipedia effect: whether per-target source-use changes concentrate on low-frequency token IDs or on tokens whose marginal frequency changed most in the aligned-Qwen 10M pool.\n\n",
        "## Pool token counts\n\n",
        "| role | rows | words | total tokens | elapsed s | pool |\n|---|---:|---:|---:|---:|---|\n",
    ]
    for role in ["OFF", "ALN"]:
        p = pool_infos[role]
        note_lines.append(f"| {role} | {p['rows']} | {p['words']} | {p['total_tokens']} | {p['elapsed_sec']:.1f} | `{p['path']}` |\n")
    note_lines += [
        "\n## Correlations by seed and target class\n\n",
        "The effect column is per-target ALN−OFF `gain_T_vs_N`; positive means aligned-Qwen training gave more true-source benefit. Correlations are Pearson correlations over target tokens within the class.\n\n",
        tab(summ[["seed", "token_class", "n_targets", "mean_delta_gain_T_vs_N", "corr_with_log1p_OFF_freq", "corr_with_delta_log1p_freq", "corr_with_ALN_log1p_freq", "frac_delta_freq_positive"]], ["seed", "token_class", "n_targets", "mean_delta_gain_T_vs_N", "corr_with_log1p_OFF_freq", "corr_with_delta_log1p_freq", "corr_with_ALN_log1p_freq", "frac_delta_freq_positive"]),
        "\n\n## Two-seed summary\n\n",
        tab(class_df, ["token_class", "n_seed_rows", "mean_effect", "mean_corr_log1p_OFF_freq", "mean_corr_delta_log1p_freq", "mean_corr_ALN_log1p_freq"]),
        "\n\n## Reading\n\n",
        "The large source-recurring ALN−OFF effect is not explained by a simple rare-token concentration pattern if it is similar across OFF-frequency quartiles and only weakly correlated with OFF token frequency or marginal ALN−OFF token-frequency change. A strong positive correlation with frequency change would instead make lexical-row exposure a serious alternative. This probe is only lexical-row evidence; it cannot identify attention or residual circuits.\n\n",
        "Quartile table: `" + rel(OUT / "frequency_quartile_effects.csv") + "`. Per-target merged rows: `" + rel(OUT / "aln_off_wikipedia_target_effects_with_frequency.csv") + "`.\n",
    ]
    NOTE.write_text("".join(note_lines), encoding="utf-8")
    result = {"status": "FREQUENCY_CONCENTRATION_DONE", "note": rel(NOTE), "out_dir": rel(OUT), "pool_infos": pool_infos}
    (OUT / "frequency_concentration_summary.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
