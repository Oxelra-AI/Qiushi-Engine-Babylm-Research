#!/usr/bin/env python3
"""research — Patched Reading evaluator for causal models with space-prefixed BPE.

The standard Reading scorer tokenizes target words WITHOUT leading space, but causal
BPE models (like RecGPT) predict space-prefixed continuation tokens. This script
uses " " + word for target tokenization to match what the causal model actually predicts.

Runs the full Reading evaluation pipeline (eye tracking + self-paced) on the patched
public RecGPT and compares with leaderboard values (eye 9.35, self-paced 4.49).
"""
from __future__ import annotations
import json, math, pathlib, argparse
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = pathlib.Path("experiments/archive/initial_model_studies")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_logits(outputs):
    if type(outputs) is tuple:
        return outputs[0]
    if hasattr(outputs, "logits"):
        return outputs.logits
    raise ValueError("Cannot extract logits")


def get_p_causal_space(sentence: str, word: str, model, tokenizer) -> float:
    """Score P(word | sentence) for causal model with space-prefix."""
    inpts = tokenizer(sentence, return_tensors="pt", add_special_tokens=False).to(DEVICE)
    with torch.no_grad():
        outputs = model(**inpts)
        logits = get_logits(outputs)[:, -1, :].cpu()
    target_id = tokenizer(word, add_special_tokens=False)["input_ids"][0]
    p = torch.softmax(logits[0], dim=-1)[target_id].item()
    return p


def get_p2_causal_space(sentence: str, word: str, model, tokenizer):
    """Score P(word | sentence) with space-prefixed target tokenization for causal BPE.

    The key fix: tokenize " " + word to get the continuation form the model predicts.
    """
    inpts = tokenizer(sentence, return_tensors="pt", add_special_tokens=False).to(DEVICE)
    with torch.no_grad():
        outputs = model(**inpts)
        logits = get_logits(outputs)[:, -1, :].cpu()

    # Space-prefixed tokenization for continuation
    target = tokenizer(" " + word, add_special_tokens=False)["input_ids"]

    if len(target) == 1:
        target_id = target[0]
        p = torch.softmax(logits[0], dim=-1)[target_id].item()
        return p, 0
    else:
        out_p = []
        target_id = target[0]
        p = torch.softmax(logits[0], dim=-1)[target_id].item()
        out_p.append(p)
        # For subsequent subtokens, append decoded first token and score next
        sentence = sentence + tokenizer.decode(target_id)
        for token in target[1:]:
            t = tokenizer.decode(token)
            # Subsequent tokens are mid-word continuations, no extra space needed
            inpts2 = tokenizer(sentence, return_tensors="pt", add_special_tokens=False).to(DEVICE)
            with torch.no_grad():
                out2 = model(**inpts2)
                logits2 = get_logits(out2)[:, -1, :].cpu()
            p = torch.softmax(logits2[0], dim=-1)[token].item()
            out_p.append(p)
            sentence = sentence + t
        p_multi = np.prod(out_p)
        return p_multi, 1


def run_reading_eval(model_path: str, data_path: str, output_dir: pathlib.Path):
    """Replicate the official Reading evaluation with space-prefix fix."""
    output_dir.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=True)
    if tok.pad_token_id is None:
        tok.pad_token = "<pad>"
    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.float32)
    model.to(DEVICE).eval()

    df = pd.read_csv(data_path, dtype={"item": str})
    df["item"] = df["item"].fillna("None")

    out = []
    prev_p2 = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Reading (space-fix)"):
        p, _ = get_p2_causal_space(row["item"], row["word"], model, tok)
        out.append(p)
        if isinstance(row["prev_item"], str):
            try:
                prev_p, _ = get_p2_causal_space(row["prev_item"], row["prev_word"], model, tok)
            except Exception:
                prev_p = 1e-10
            prev_p2.append(-math.log(max(prev_p, 1e-30)))
        else:
            prev_p2.append(float("NaN"))

    p2 = [-math.log(max(p, 1e-30)) for p in out]
    df["pred"] = p2
    df["prev_pred"] = prev_p2

    # Save predictions
    pred_file = output_dir / "prediction.jsonl"
    with pred_file.open("w") as fj:
        for index, row in df.iterrows():
            print(json.dumps({"Index": index, "Sentence": row["item"], "Word": row["word"],
                              "Logprob": row["pred"], "Prev_Logprob": row["prev_pred"]}), file=fj)

    # Eye tracking regression
    variables = ['RTfirstfix', 'RTfirstpass', 'RTgopast', 'RTrightbound',
                 'self_paced_reading_time', 'ELAN', 'LAN', 'N400', 'P600', 'EPNP', 'PNP']

    report_values = []
    results_eye = []
    for dv in variables:
        temp = df[[dv, "Subtlex_log10", "length", "context_length"]].dropna()
        OLS_baseline = smf.ols(
            formula=dv + ' ~ Subtlex_log10 + length + context_length + Subtlex_log10:length + Subtlex_log10:context_length + length:context_length',
            data=temp).fit()
        R2_baseline = float(OLS_baseline.rsquared)
        temp = df[["pred", dv, "Subtlex_log10", "length", "context_length"]].dropna()
        OLS_model = smf.ols(
            formula=dv + ' ~ Subtlex_log10 + length + context_length + Subtlex_log10:length + Subtlex_log10:context_length + length:context_length + pred',
            data=temp).fit()
        R2_model = float(OLS_model.rsquared)
        results_eye.append({
            "Predicted variable": dv,
            "Coefficient": float(OLS_model.params["pred"]),
            "t-value": float(OLS_model.tvalues["pred"]),
            "P-value": float(OLS_model.pvalues["pred"]),
            "R2": R2_model,
            "Delta_R2": R2_model - R2_baseline,
        })
        if "RT" in dv:
            report_values.append(((R2_model - R2_baseline) / (1 - R2_baseline)) * 100)

    eye_score = sum(report_values) / len(report_values) if report_values else 0.0

    # Self-paced with spillover
    report_spr = 0.0
    results_spr = []
    for dv in variables:
        temp = df[[dv, "Subtlex_log10", "length", "context_length", "prev_length", "prev_pred"]].dropna()
        OLS_baseline = smf.ols(
            formula=dv + ' ~ Subtlex_log10 + length + context_length + prev_length + prev_pred + Subtlex_log10:length + Subtlex_log10:context_length + Subtlex_log10:prev_length + Subtlex_log10:prev_pred + length:context_length + length:prev_length + length:prev_pred + context_length:prev_length + context_length:prev_pred + prev_length:prev_pred',
            data=temp).fit()
        R2_baseline = float(OLS_baseline.rsquared)
        temp = df[["pred", dv, "Subtlex_log10", "length", "context_length", "prev_length", "prev_pred"]].dropna()
        OLS_model = smf.ols(
            formula=dv + ' ~ Subtlex_log10 + length + context_length + prev_length + prev_pred + Subtlex_log10:length + Subtlex_log10:context_length + Subtlex_log10:prev_length + Subtlex_log10:prev_pred + length:context_length + length:prev_length + length:prev_pred + context_length:prev_length + context_length:prev_pred + prev_length:prev_pred + pred',
            data=temp).fit()
        R2_model = float(OLS_model.rsquared)
        results_spr.append({
            "Predicted variable": dv,
            "Coefficient": float(OLS_model.params["pred"]),
            "t-value": float(OLS_model.tvalues["pred"]),
            "P-value": float(OLS_model.pvalues["pred"]),
            "R2": R2_model,
            "Delta_R2": R2_model - R2_baseline,
        })
        if "self" in dv:
            report_spr = ((R2_model - R2_baseline) / (1 - R2_baseline)) * 100

    # Save report
    report_file = output_dir / "report.txt"
    with report_file.open("w") as f:
        print(f"EYE TRACKING SCORE: {eye_score:.2f}", file=f)
        print(f"SELF-PACED READING SCORE: {report_spr:.2f}", file=f)
    print(f"EYE TRACKING SCORE: {eye_score:.2f}")
    print(f"SELF-PACED READING SCORE: {report_spr:.2f}")
    print(f"READING MEAN: {(eye_score + report_spr) / 2:.2f}")

    # Save detailed results
    detail_file = output_dir / "detailed_results.json"
    detail_file.write_text(json.dumps({
        "eye_tracking_score": round(eye_score, 2),
        "self_paced_score": round(report_spr, 2),
        "reading_mean": round((eye_score + report_spr) / 2, 2),
        "leaderboard_eye": 9.35,
        "leaderboard_spr": 4.49,
        "leaderboard_reading": 6.92,
        "delta_eye": round(eye_score - 9.35, 2),
        "delta_spr": round(report_spr - 4.49, 2),
        "eye_results": results_eye,
        "spr_results": results_spr,
        "pred_summary": {
            "mean": float(df["pred"].mean()),
            "std": float(df["pred"].std()),
            "min": float(df["pred"].min()),
            "max": float(df["pred"].max()),
        },
        "fix_applied": "Space-prefixed target tokenization: tokenizer(' ' + word) instead of tokenizer(word)",
    }, indent=2) + "\n")
    print(f"Saved: {detail_file}")
    return eye_score, report_spr


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", default=str(ROOT / "data/recgpt_local/patched_model"))
    p.add_argument("--data_path", default=str(ROOT / "repos/babylm-eval/strict/evaluation_data/full_eval/reading/reading_data.csv"))
    p.add_argument("--output_dir", default=str(ROOT / "training/runs/recgpt_reading_spacefix"))
    args = p.parse_args()
    run_reading_eval(args.model_path, args.data_path, pathlib.Path(args.output_dir))


if __name__ == "__main__":
    main()
