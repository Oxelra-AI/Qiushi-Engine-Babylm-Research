#!/usr/bin/env python3
"""research: single-entity counterfactual probe for entity-event-state binding.

Inference-only. Uses annotated binding eval rows and compares existing
DeBERTa checkpoints on:
  - original two-entity affected-query rows;
  - affected-query rows rewritten as a single-entity state-change sentence;
  - original two-entity unaffected-query rows;
  - no-change single-entity rows for the unaffected query entity.

Purpose: distinguish whether existing checkpoints lack the verb->result-state
transition mapping even in isolation, or mainly fail under two-entity binding /
competition. No model weights are changed; no BabyLM training/evaluation/upload.
"""

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative

import argparse
import csv
import json
import os
from collections import defaultdict
from pathlib import Path

import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

ROOT = _public_path('.')
DEFAULT_DATA = _public_path('experiments/archive/representation_and_objectives/data/annotated_corpus/eval.jsonl')
DEFAULT_OUT = _public_path('experiments/archive/frontier_consolidation/data/binding_single_entity_probe')
DEFAULT_MODELS = {
    "scale1p75_chck82": _public_path('experiments/archive/frontier_consolidation/training/runs/adapter128_scale1p75_h100M100M_seed43022_official_ladder/hf_model/chck_82M'),
    "scale1p75_chck84": _public_path('experiments/archive/frontier_consolidation/training/runs/adapter128_scale1p75_h100M100M_seed43022_official_ladder/hf_model/chck_84M'),
    "scale1p75_chck100": _public_path('experiments/archive/frontier_consolidation/training/runs/adapter128_scale1p75_h100M100M_seed43022_official_ladder/hf_model/chck_100M'),
}


def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def first_answer_token(tok, word):
    ids = tok.encode(" " + word, add_special_tokens=False)
    if not ids:
        raise ValueError(f"empty encoding for {word!r}")
    # Match the binding evaluator convention: use second piece when byte-level BPE
    # emits a separate prefix piece, otherwise first piece.
    return ids[1] if len(ids) > 1 else ids[0]


def make_probe_rows(eval_rows):
    probes = []
    for r in eval_rows:
        if r.get("split") != "eval_held_recomb" or r.get("kind") != "binding":
            continue
        q = r["query_entity"]
        actor = r["actor_entity"]
        is_aff = bool(r["is_affected_query"])
        event = r.get("events", [{}])[0]
        verb = event.get("verb") or "changed"
        answer = r["answer"]
        foil = r["foil"]
        if is_aff:
            # For affected rows the foil is the initial state and answer is the result state.
            initial = foil
            single_text = f"The {q} was {initial}. Mira {verb} the {q}. The {q} is now <mask>."
            probes.append({
                "source_id": r["id"],
                "variant": "original_affected_two_entity",
                "text": r["text"],
                "answer": answer,
                "foil": foil,
                "family": r["family"],
                "template_id": r.get("template_id"),
                "query_entity": q,
                "actor_entity": actor,
                "is_affected_query": True,
                "verb": verb,
            })
            probes.append({
                "source_id": r["id"],
                "variant": "single_affected_update",
                "text": single_text,
                "answer": answer,
                "foil": foil,
                "family": r["family"],
                "template_id": r.get("template_id"),
                "query_entity": q,
                "actor_entity": actor,
                "is_affected_query": True,
                "verb": verb,
            })
        else:
            # For unaffected rows the answer is the initial state. A single no-change
            # variant tests whether preserving the entity's own state is easy absent
            # competitor/event binding.
            initial = answer
            single_text = f"The {q} was {initial}. Nothing changed the {q}. The {q} is now <mask>."
            probes.append({
                "source_id": r["id"],
                "variant": "original_unaffected_two_entity",
                "text": r["text"],
                "answer": answer,
                "foil": foil,
                "family": r["family"],
                "template_id": r.get("template_id"),
                "query_entity": q,
                "actor_entity": actor,
                "is_affected_query": False,
                "verb": verb,
            })
            probes.append({
                "source_id": r["id"],
                "variant": "single_unaffected_nochange",
                "text": single_text,
                "answer": answer,
                "foil": foil,
                "family": r["family"],
                "template_id": r.get("template_id"),
                "query_entity": q,
                "actor_entity": actor,
                "is_affected_query": False,
                "verb": verb,
            })
    return probes


def evaluate_model(model_name, model_path, probes, out_dir, device="cpu", batch_size=64):
    model_path = Path(model_path)
    cache_home = out_dir / f"hf_home_{model_name}"
    os.environ.setdefault("HF_HOME", str(cache_home))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(cache_home / "transformers"))
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)
    model = AutoModelForMaskedLM.from_pretrained(model_path, trust_remote_code=True, local_files_only=True).to(device).eval()
    mask_id = tok.mask_token_id
    for p in probes:
        p["_answer_id"] = first_answer_token(tok, p["answer"])
        p["_foil_id"] = first_answer_token(tok, p["foil"])
    rows = []
    for i in range(0, len(probes), batch_size):
        batch = probes[i:i+batch_size]
        enc = tok([p["text"] for p in batch], padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
        with torch.no_grad():
            logits = model(**enc).logits
        for j, p in enumerate(batch):
            ids = enc["input_ids"][j]
            mp = (ids == mask_id).nonzero(as_tuple=True)[0]
            if len(mp) == 0:
                margin = float("nan")
                correct = False
                err = "no_mask"
            else:
                pos = mp[0].item()
                margin = float(logits[j, pos, p["_answer_id"]].item() - logits[j, pos, p["_foil_id"]].item())
                correct = margin > 0
                err = ""
            row = {k: p[k] for k in ["source_id", "variant", "family", "template_id", "query_entity", "actor_entity", "is_affected_query", "verb", "answer", "foil", "text"]}
            row.update({"model": model_name, "correct": correct, "margin": margin, "error": err})
            rows.append(row)
    del model
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return rows


def summarize(rows):
    def acc(rs):
        return sum(1 for r in rs if r["correct"]) / len(rs) if rs else None
    def mean_margin(rs):
        return sum(r["margin"] for r in rs) / len(rs) if rs else None
    by_model_variant = defaultdict(list)
    by_model_variant_family = defaultdict(list)
    for r in rows:
        by_model_variant[(r["model"], r["variant"])].append(r)
        by_model_variant_family[(r["model"], r["variant"], r["family"])].append(r)
    models = sorted(set(r["model"] for r in rows))
    variants = sorted(set(r["variant"] for r in rows))
    summary = {"by_model_variant": {}, "by_model_variant_family": {}, "derived": {}}
    for m in models:
        summary["by_model_variant"][m] = {}
        for v in variants:
            rs = by_model_variant.get((m, v), [])
            summary["by_model_variant"][m][v] = {"n": len(rs), "accuracy": acc(rs), "margin_mean": mean_margin(rs)}
        oa = summary["by_model_variant"][m].get("original_affected_two_entity", {}).get("accuracy")
        sa = summary["by_model_variant"][m].get("single_affected_update", {}).get("accuracy")
        ou = summary["by_model_variant"][m].get("original_unaffected_two_entity", {}).get("accuracy")
        su = summary["by_model_variant"][m].get("single_unaffected_nochange", {}).get("accuracy")
        summary["derived"][m] = {
            "affected_single_minus_two_entity_acc": None if oa is None or sa is None else sa - oa,
            "unaffected_nochange_minus_two_entity_acc": None if ou is None or su is None else su - ou,
            "single_update_minus_single_nochange_acc": None if sa is None or su is None else sa - su,
        }
    for (m, v, fam), rs in sorted(by_model_variant_family.items()):
        summary["by_model_variant_family"].setdefault(m, {}).setdefault(v, {})[fam] = {
            "n": len(rs), "accuracy": acc(rs), "margin_mean": mean_margin(rs)
        }
    return summary


def write_md(summary, path):
    lines = []
    lines.append("# research single-entity binding probe\n")
    lines.append("Inference-only probe over A01 research held recombination binding rows. Affected rows are rewritten as one-entity update sentences; unaffected rows are rewritten as one-entity no-change sentences.\n")
    lines.append("## Main variant accuracies\n")
    lines.append("| model | original affected two-entity | single affected update | original unaffected two-entity | single unaffected no-change | affected single-two delta |\n")
    lines.append("|---|---:|---:|---:|---:|---:|\n")
    for m, vals in summary["by_model_variant"].items():
        oa = vals.get("original_affected_two_entity", {}).get("accuracy")
        sa = vals.get("single_affected_update", {}).get("accuracy")
        ou = vals.get("original_unaffected_two_entity", {}).get("accuracy")
        su = vals.get("single_unaffected_nochange", {}).get("accuracy")
        d = summary["derived"][m]["affected_single_minus_two_entity_acc"]
        def f(x): return "None" if x is None else f"{x:.3f}"
        lines.append(f"| {m} | {f(oa)} | {f(sa)} | {f(ou)} | {f(su)} | {f(d)} |\n")
    lines.append("\n## Interpretation notes\n")
    lines.append("- If `single_affected_update` is high while original affected two-entity is low, the existing model knows many verb->result mappings but fails entity binding under competition.\n")
    lines.append("- If `single_affected_update` remains low, the route needs transition-result signal or representation learning, not only slot separation.\n")
    lines.append("- `single_unaffected_nochange` is an easy preservation control; high values there do not establish binding, only retention without competing event.\n")
    lines.append("- This is not BabyLM official evaluation and not model training. It only sharpens the next mechanism route.\n")
    path.write_text("".join(lines), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(DEFAULT_DATA))
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--model", action="append", default=[], help="name=path; may be repeated. Defaults to chck82/84/100")
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if args.model:
        models = {}
        for item in args.model:
            name, path = item.split("=", 1)
            models[name] = Path(path)
    else:
        models = DEFAULT_MODELS
    eval_rows = load_jsonl(Path(args.data))
    probes = make_probe_rows(eval_rows)
    (out / "probe_rows.jsonl").write_text("\n".join(json.dumps(p, ensure_ascii=False) for p in probes) + "\n", encoding="utf-8")
    all_rows = []
    for name, path in models.items():
        rows = evaluate_model(name, path, probes, out, device=args.device, batch_size=args.batch_size)
        all_rows.extend(rows)
        with open(out / f"per_item_{name}.csv", "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
    summary = summarize(all_rows)
    summary.update({
        "status": "BINDING_SINGLE_ENTITY_PROBE",
        "data": str(Path(args.data)),
        "n_probe_rows": len(probes),
        "models": {k: str(v) for k, v in models.items()},
        "device": args.device,
        "no_training_official_eval_upload_or_leaderboard": True,
    })
    (out / "single_entity_probe_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_md(summary, out / "single_entity_probe_summary.md")
    print(json.dumps({
        "status": summary["status"],
        "out_dir": str(out),
        "n_probe_rows": len(probes),
        "derived": summary["derived"],
        "no_training_official_eval_upload_or_leaderboard": True,
    }, indent=2))

if __name__ == "__main__":
    main()
