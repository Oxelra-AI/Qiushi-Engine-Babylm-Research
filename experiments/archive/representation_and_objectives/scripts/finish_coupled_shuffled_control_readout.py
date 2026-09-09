#!/usr/bin/env python3
"""research: finish and assemble the coupled-sparse20 shuffled-control readout.

The research readout timed out after completing GlobalPIQA for all targets and EWoK
for MLM-only plus coupled-aligned.  This script runs only the missing EWoK readout
for the coupled-shuffled checkpoint, then assembles the final correspondence
summary using the already written row-level outputs.  It does no training and does
not touch the chck_82M endpoint.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import importlib.util
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

USER_ROOT = _public_path('.')
os.chdir(USER_ROOT)
A01_WS = _public_path('experiments/archive/representation_and_objectives')
SCRIPT_DIR = _public_path('experiments/archive/representation_and_objectives/scripts')
OUT_ROOT = _public_path('experiments/archive/representation_and_objectives/data/coupled_shuffled_control_readout')


def import_from(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def rel(p: Path | str | None) -> str | None:
    if p is None:
        return None
    pp = Path(p)
    try:
        return str(pp.resolve().relative_to(USER_ROOT))
    except Exception:
        return str(pp)


def now_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


S177 = import_from(_public_path('experiments/archive/representation_and_objectives/scripts/coupled_shuffled_control_readout.py'), "readout_for_step180")
S175 = S177.S175
S175.TARGETS["coupled_sparse20_shuffled_20M"] = S177.COUPLED_SHUFFLED


def set_safe_caches(out_root: Path) -> None:
    cache_root = out_root / "hf_cache"
    for d in [out_root / "home", out_root / "xdg_cache", cache_root, cache_root / "modules", out_root / "torch_cache", out_root / "tmp"]:
        d.mkdir(parents=True, exist_ok=True)
    os.environ["HOME"] = str((out_root / "home").resolve())
    os.environ["XDG_CACHE_HOME"] = str((out_root / "xdg_cache").resolve())
    os.environ["HF_HOME"] = str(cache_root.resolve())
    os.environ["TRANSFORMERS_CACHE"] = str(cache_root.resolve())
    os.environ["HF_MODULES_CACHE"] = str((cache_root / "modules").resolve())
    os.environ["TORCH_HOME"] = str((out_root / "torch_cache").resolve())
    os.environ["TMPDIR"] = str((out_root / "tmp").resolve())


def load_gp(name: str) -> dict[str, Any]:
    path = _public_path('experiments/archive/representation_and_objectives/data/coupled_shuffled_control_readout/globalpiqa') / f"{name}_globalpiqa_margins.json"
    if not path.exists():
        raise FileNotFoundError(path)
    return S175.compact_gp(read_json(path))


def load_ewok(name: str) -> dict[str, Any]:
    path = _public_path('experiments/archive/representation_and_objectives/data/coupled_shuffled_control_readout/ewok_stable_subset') / name / "ewok_stable_subset_summary.json"
    if not path.exists():
        raise FileNotFoundError(path)
    d = read_json(path)
    return {k: v for k, v in d.items() if k not in {"by_domain", "by_context_diff"}}


def target_ready(name: str, meta: dict[str, Any]) -> dict[str, Any]:
    mp = Path(meta["model_path"])
    return {
        "label": meta.get("label"),
        "family": meta.get("family"),
        "model_path": rel(mp),
        "ready": mp.exists() and (mp / "model.safetensors").exists() and (mp / "config.json").exists(),
        "training_metrics": S175.read_json(Path(meta["metrics_path"])),
    }


def hard_summary_gp(gp: dict[str, Any]) -> dict[str, Any]:
    par = gp.get("modes", {}).get("parallel", {}) if isinstance(gp, dict) else {}
    hard = par.get("always_wrong_subset", {}) if isinstance(par, dict) else {}
    return {
        "parallel_accuracy": par.get("accuracy"),
        "parallel_rank_counts": par.get("correct_rank_counts"),
        "hard52_accuracy": hard.get("accuracy"),
        "hard52_rank_counts": hard.get("correct_rank_counts"),
        "hard52_mean_top_minus_correct": hard.get("mean_top_minus_correct"),
        "hard52_median_top_minus_correct": hard.get("median_top_minus_correct"),
    }


def hard_summary_ewok(ew: dict[str, Any]) -> dict[str, Any]:
    s = ew.get("summary", {}) if isinstance(ew, dict) else {}
    iw = s.get("interaction_sum_wrong", {}) if isinstance(s.get("interaction_sum_wrong"), dict) else {}
    return {
        "n": s.get("n"),
        "accuracy": s.get("accuracy"),
        "saved_wrong": s.get("saved_wrong"),
        "stable_failure": s.get("stable_failure"),
        "stable_failure_frac_all": s.get("stable_failure_frac_all"),
        "within_both_positive_wrong_frac": s.get("within_both_positive_wrong_frac"),
        "interaction_sum_wrong_mean": iw.get("mean"),
        "interaction_sum_wrong_median": iw.get("median"),
        "interaction_sum_wrong_p05": iw.get("p05"),
        "interaction_sum_wrong_p95": iw.get("p95"),
    }


def make_note(summary: dict[str, Any], out_md: Path) -> None:
    lines: list[str] = []
    lines.append("# research coupled sparse20 shuffled-control readout")
    lines.append("")
    lines.append(f"Status: **{summary['status']}**")
    lines.append("")
    lines.append("This readout finishes the matched coupled-shuffled control from research. It uses existing checkpoints only, keeps the fixed research EWoK stable-reversal rows and fixed research GlobalPIQA hard52 rows, and leaves the reproduced `chck_82M` endpoint untouched.")
    lines.append("")
    lines.append("## Training-coordinate match")
    for name, st in summary["targets"].items():
        m = st.get("training_metrics", {}) if isinstance(st.get("training_metrics"), dict) else {}
        lines.append(f"- `{name}`: ready={st.get('ready')} mode={m.get('mode')} updates={m.get('updates')} main={m.get('total_main_word_exposure')} aux={m.get('total_aux_word_exposure')} charged={m.get('total_charged_words')} first_loss={m.get('first_loss')} aux_batches={m.get('aux_loss_batches')}")
    lines.append("")
    lines.append("## GlobalPIQA rank/margin surface")
    for name in summary["globalpiqa"]:
        lines.append(f"- `{name}`: {json.dumps(hard_summary_gp(summary['globalpiqa'][name]), ensure_ascii=False)}")
    gp_delta = summary.get("deltas", {}).get("globalpiqa", {}).get("coupled_aligned_minus_coupled_shuffled", {}).get("parallel", {})
    lines.append(f"- `coupled_aligned_minus_coupled_shuffled` parallel delta: {json.dumps(gp_delta, ensure_ascii=False)}")
    lines.append("")
    lines.append("## EWoK stable-reversal surface")
    for name in summary["ewok_stable_subset"]:
        lines.append(f"- `{name}`: {json.dumps(hard_summary_ewok(summary['ewok_stable_subset'][name]), ensure_ascii=False)}")
    ew_delta = summary.get("deltas", {}).get("ewok_stable_subset", {}).get("coupled_aligned_minus_coupled_shuffled", {})
    lines.append(f"- `coupled_aligned_minus_coupled_shuffled` delta: {json.dumps(ew_delta, ensure_ascii=False)}")
    lines.append("")
    lines.append("## Scientific reading")
    for item in summary.get("interpretation", []):
        lines.append(f"- {item}")
    lines.append("")
    lines.append(f"JSON: `{rel(summary['summary_json'])}`")
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    set_safe_caches(OUT_ROOT)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    ewok_root = _public_path('experiments/archive/representation_and_objectives/data/coupled_shuffled_control_readout/ewok_stable_subset')
    shuf_file = ewok_root / "coupled_sparse20_shuffled_20M" / "ewok_stable_subset_summary.json"
    if not shuf_file.exists():
        S175.run_ewok_subset(
            {"coupled_sparse20_shuffled_20M": S177.COUPLED_SHUFFLED},
            ewok_root,
            max_rows=None,
            threads=24,
            row_batch_size=64,
            masked_batch_size=160,
        )

    target_names = ["mlm_only_20M", "coupled_sparse20_aligned_20M", "coupled_sparse20_shuffled_20M"]
    targets = {
        "mlm_only_20M": S175.TARGETS["mlm_only_20M"],
        "coupled_sparse20_aligned_20M": S175.TARGETS["coupled_sparse20_aligned_20M"],
        "coupled_sparse20_shuffled_20M": S177.COUPLED_SHUFFLED,
    }
    globalpiqa = {name: load_gp(name) for name in target_names}
    ewok = {name: load_ewok(name) for name in target_names}
    status = {name: target_ready(name, targets[name]) for name in target_names}
    indices, ew_meta = S175.load_stable_indices(max_rows=None)
    summary: dict[str, Any] = {
        "status": "COUPLED_SHUFFLED_CONTROL_READOUT_DONE",
        "created_utc": now_utc(),
        "boundary": "posthoc matched coupled-correspondence hard-surface readout on existing checkpoints; no training",
        "out_root": rel(OUT_ROOT),
        "summary_json": str(_public_path('experiments/archive/representation_and_objectives/data/coupled_shuffled_control_readout/coupled_shuffled_control_summary.json')),
        "targets": status,
        "globalpiqa_fixed_hard_set": {
            "source": "experiments/archive/representation_and_objectives/data/globalpiqa_margin_synthesis/globalpiqa_margin_synthesis.json",
            "definition": "fixed research cross-endpoint GlobalPIQA_parallel hard52 rows",
        },
        "ewok_stable_subset_definition": ew_meta,
        "globalpiqa": globalpiqa,
        "ewok_stable_subset": ewok,
        "deltas": S177.build_step177_deltas(globalpiqa, ewok),
        "interpretation": [],
    }
    summary["interpretation"] = S177.interpret_step177(summary)
    out_json = _public_path('experiments/archive/representation_and_objectives/data/coupled_shuffled_control_readout/coupled_shuffled_control_summary.json')
    summary["summary_json"] = str(out_json)
    out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    make_note(summary, _public_path('research/documents/representation_and_objectives/data/coupled_shuffled_control_readout/coupled_shuffled_control_summary.md'))
    print(json.dumps({"status": summary["status"], "summary_json": rel(out_json), "summary_md": rel(_public_path('research/documents/representation_and_objectives/data/coupled_shuffled_control_readout/coupled_shuffled_control_summary.md'))}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
