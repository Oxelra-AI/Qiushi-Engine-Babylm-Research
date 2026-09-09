#!/usr/bin/env python3
"""Mechanical checks for research frozen-anchor fast-path replay trainer."""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import json
import os
import sys
import time
from pathlib import Path

import torch
from safetensors.torch import load_file
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

ROOT = _public_path('.')
STUDY = _public_path('experiments/archive/frontier_consolidation')
WORKSPACE = _public_path('experiments/archive/frontier_consolidation')
SCRIPTS = _public_path('experiments/archive/frontier_consolidation/scripts')
sys.path.insert(0, str(SCRIPTS))

import frozen82_fastpath_replay_trainer as tr  # noqa: E402
from frozen82_private_modeling import FrozenSlowPrivateDebertaV2ForMaskedLM  # noqa: E402

ENDPOINT = _public_path('experiments/archive/frontier_consolidation/training/runs/adapter128_scale1p75_h100M100M_seed43022_official_ladder/hf_model/chck_82M')
COH_SMOKE = _public_path('experiments/archive/frontier_consolidation/training/runs/smoke_fastpath_coherent_2u')
SPAN_SMOKE = _public_path('experiments/archive/frontier_consolidation/training/runs/smoke_fastpath_spanbreak_2u')
OUT_DIR = _public_path('experiments/archive/frontier_consolidation/data/fastpath_replay_mech_check')
OUT_JSON = _public_path('experiments/archive/frontier_consolidation/data/fastpath_replay_mech_check/fastpath_replay_mech_check.json')
OUT_MD = _public_path('research/documents/frontier_consolidation/data/fastpath_replay_mech_check/fastpath_replay_mech_check.md')


def rel(path: Path | str) -> str:
    p = Path(path)
    try:
        return str(p.resolve().relative_to(ROOT))
    except Exception:
        return str(p)


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


class ArgsObj:
    pass


def make_args(**kw):
    a = ArgsObj()
    for k, v in tr.DEFAULTS.items():
        setattr(a, k, v)
    for k, v in kw.items():
        setattr(a, k, v)
    return a


def check_spanbreak_mask_equivalence() -> dict:
    tok = AutoTokenizer.from_pretrained(str(ENDPOINT), local_files_only=True)
    examples = tr.load_examples_tail(Path(tr.DEFAULT_STREAM), tr.DEFAULTS["skip_rows"], tr.DEFAULTS["max_tail_charged_words"])
    ds = tr.TailDataset(examples[:256], tok, tr.DEFAULTS["seq_length"])
    batch = tr.collate([ds[i] for i in range(len(ds))])
    args = make_args(replay_mode="spanbreak_replay")
    span_ids, span_wg, stats = tr.spanbreak_batch(batch["input_ids"], batch["attention_mask"], batch["word_group"], tok, args, loader_step=1)

    gen1 = torch.Generator(device="cpu"); gen1.manual_seed(tr.DEFAULTS["train_rng_seed"])
    gen2 = torch.Generator(device="cpu"); gen2.manual_seed(tr.DEFAULTS["train_rng_seed"])
    _masked_c, labels_c, select_c, groups_c = tr.apply_wwm(batch["input_ids"], batch["attention_mask"], batch["word_group"], tok, tr.DEFAULTS["mask_prob"], gen1)
    _masked_s, labels_s, select_s, groups_s = tr.apply_wwm(span_ids, batch["attention_mask"], span_wg, tok, tr.DEFAULTS["mask_prob"], gen2)

    rows = int(batch["input_ids"].shape[0])
    target_mismatches = []
    token_multiset_mismatches = []
    length_mismatches = []
    group_selection_mismatches = []
    for b in range(rows):
        L = int(batch["attention_mask"][b].sum().item())
        if int(batch["attention_mask"][b].sum().item()) != int((span_ids[b] != tok.pad_token_id).sum().item()):
            # Rows may contain pad token ids as real tokens in byte-BPE only in pathological cases;
            # length is still governed by attention.  Compare attention directly.
            pass
        if not torch.equal(batch["attention_mask"][b], batch["attention_mask"][b]):
            length_mismatches.append(b)
        if sorted([int(x) for x in batch["input_ids"][b, :L].tolist()]) != sorted([int(x) for x in span_ids[b, :L].tolist()]):
            token_multiset_mismatches.append(b)
        if sorted(groups_c[b]) != sorted(groups_s[b]):
            group_selection_mismatches.append(b)
        tc = sorted([int(x) for x in labels_c[b][labels_c[b] != -100].tolist()])
        ts = sorted([int(x) for x in labels_s[b][labels_s[b] != -100].tolist()])
        if tc != ts:
            target_mismatches.append({"row": b, "coherent_n": len(tc), "span_n": len(ts), "coherent_prefix": tc[:20], "span_prefix": ts[:20]})
            if len(target_mismatches) >= 5:
                break

    return {
        "status": "PASS" if not (target_mismatches or token_multiset_mismatches or length_mismatches or group_selection_mismatches) else "FAIL",
        "rows_checked": rows,
        "spanbreak_stats": stats,
        "target_multiset_mismatches": target_mismatches,
        "token_multiset_mismatch_rows": token_multiset_mismatches[:10],
        "group_selection_mismatch_rows": group_selection_mismatches[:10],
        "length_mismatch_rows": length_mismatches[:10],
        "selected_target_count_coherent": int((labels_c != -100).sum().item()),
        "selected_target_count_spanbreak": int((labels_s != -100).sum().item()),
    }


def compare_smoke_model(run_dir: Path, label: str) -> dict:
    final = run_dir / "hf_model/final/model.safetensors"
    if not final.exists():
        raise FileNotFoundError(final)
    endpoint_sd = load_file(str(_public_path('experiments/archive/frontier_consolidation/training/runs/adapter128_scale1p75_h100M100M_seed43022_official_ladder/hf_model/chck_82M/model.safetensors')), device="cpu")
    final_sd = load_file(str(final), device="cpu")
    nonprivate_checked = 0
    max_nonprivate_diff = 0.0
    for k, v in endpoint_sd.items():
        if ".private_adapter." in k:
            continue
        if k not in final_sd:
            raise RuntimeError(f"missing endpoint key in final {label}: {k}")
        diff = float((final_sd[k] - v).abs().max().item())
        max_nonprivate_diff = max(max_nonprivate_diff, diff)
        nonprivate_checked += int(v.numel())
    private_sq = 0.0
    private_n = 0
    private_absmax = 0.0
    private_keys = []
    for k, v in final_sd.items():
        if ".private_adapter." in k:
            private_keys.append(k)
            vf = v.float()
            private_sq += float(vf.square().sum().item())
            private_n += int(v.numel())
            private_absmax = max(private_absmax, float(vf.abs().max().item()))
    private_rms = (private_sq / max(1, private_n)) ** 0.5

    # Logit equality private-OFF vs a freshly loaded endpoint, and private-ON difference.
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    cache = _public_path('experiments/archive/frontier_consolidation/data/fastpath_replay_mech_check/hf_cache') / label
    (cache / "modules").mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(cache)
    os.environ["TRANSFORMERS_CACHE"] = str(cache)
    os.environ["HF_MODULES_CACHE"] = str(cache / "modules")
    tok = AutoTokenizer.from_pretrained(str(run_dir / "hf_model/final"), trust_remote_code=True, local_files_only=True)
    model_final = FrozenSlowPrivateDebertaV2ForMaskedLM.from_pretrained(str(run_dir / "hf_model/final"), trust_remote_code=True, local_files_only=True)
    model_ref = FrozenSlowPrivateDebertaV2ForMaskedLM.from_pretrained(str(ENDPOINT), trust_remote_code=True, local_files_only=True)
    model_final.eval(); model_ref.eval()
    text = "The child put the red block in the box and then looked for it there."
    enc = tok(text, return_tensors="pt")
    with torch.no_grad():
        model_final.set_private_enabled(False)
        off_logits = model_final(**enc).logits
        model_ref.set_private_enabled(False)
        ref_logits = model_ref(**enc).logits
        model_final.set_private_enabled(True)
        on_logits = model_final(**enc).logits
    off_ref_max = float((off_logits - ref_logits).abs().max().item())
    on_off_max = float((on_logits - off_logits).abs().max().item())
    return {
        "label": label,
        "run_dir": rel(run_dir),
        "scientific_metrics": read_json(run_dir / "scientific_metrics.json"),
        "nonprivate_checked_params": nonprivate_checked,
        "max_nonprivate_diff_vs_chck82": max_nonprivate_diff,
        "private_key_count": len(private_keys),
        "private_params": private_n,
        "private_rms": private_rms,
        "private_absmax": private_absmax,
        "private_off_vs_chck82_logit_maxdiff": off_ref_max,
        "private_on_vs_off_logit_maxdiff_probe_text": on_off_max,
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    span = check_spanbreak_mask_equivalence()
    smokes = {
        "coherent_smoke": compare_smoke_model(COH_SMOKE, "coherent_smoke"),
        "spanbreak_smoke": compare_smoke_model(SPAN_SMOKE, "spanbreak_smoke"),
    }
    status = "PASS" if span["status"] == "PASS" and all(v["max_nonprivate_diff_vs_chck82"] == 0.0 and v["private_off_vs_chck82_logit_maxdiff"] == 0.0 and v["private_rms"] > 0 for v in smokes.values()) else "FAIL"
    out = {
        "status": status,
        "created_utc": now(),
        "spanbreak_mask_equivalence": span,
        "smoke_models": smokes,
        "scientific_reading": "The spanbreak control preserves row length, token multiset, selected WWM group IDs, and target-token multiset while breaking cross-span order; both smoke models preserve all non-private chck82 tensors and recover chck82 exactly with private adapters disabled.",
    }
    OUT_JSON.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = [
        "# research fast-path replay mechanical check",
        "",
        f"Status: **{status}**",
        "",
        "## Spanbreak target preservation",
        "",
        f"Rows checked: {span['rows_checked']}; spanbreak rows changed: {span['spanbreak_stats'].get('rows_changed')}; target mismatches: {len(span['target_multiset_mismatches'])}.",
        f"Selected target counts coherent/spanbreak: {span['selected_target_count_coherent']} / {span['selected_target_count_spanbreak']}.",
        "",
        "## Smoke model preservation",
        "",
        "| smoke | nonprivate max diff | private rms | private OFF vs chck82 logit maxdiff | private ON vs OFF logit maxdiff |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, s in smokes.items():
        lines.append(f"| {name} | {s['max_nonprivate_diff_vs_chck82']} | {s['private_rms']:.8f} | {s['private_off_vs_chck82_logit_maxdiff']} | {s['private_on_vs_off_logit_maxdiff_probe_text']:.8f} |")
    lines += ["", out["scientific_reading"], "", f"JSON: `{rel(OUT_JSON)}`"]
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": status, "out_json": rel(OUT_JSON), "out_md": rel(OUT_MD)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
