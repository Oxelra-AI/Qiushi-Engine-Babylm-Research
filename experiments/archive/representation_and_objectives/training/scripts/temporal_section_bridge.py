#!/usr/bin/env python3
"""research: section-address ceiling for the research temporal-change bridge.

This wrapper imports the repaired research bridge and replaces the natural
initial/later language with a discourse-section address:

  Background: ...   Update: ...
  Based on the background/update, ...

Scientific boundary: this is a supplied-address ceiling.  It tests whether the
full fixed-meaning temporal bridge can work when the before/after state is made
addressable by familiar discourse keys.  It is not by itself evidence that the
model has learned a general time-indexed state principle, because exact keys and
fixed positions remain available.
"""
from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path

STUDY = Path("experiments/archive/representation_and_objectives")
BASE_PATH = STUDY / "training/scripts/temporal_change_bridge_probe.py"
spec = importlib.util.spec_from_file_location("temporal_section_base", BASE_PATH)
base = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = base
spec.loader.exec_module(base)

base.DEFAULT_OUT = STUDY / "data/temporal_section_full_bridge"

# Use discourse-functional labels for every template group used by the base
# script, while preserving the truth relation and held/train wording split.
base.INIT_CTX = {
    "anchor": [
        "Background: {H} was ranked above {Lo} in the ATP list.",
        "Background record: {Lo} was ranked below {H} in the ATP list.",
    ],
    "train": [
        "Background: {H} was ranked above {Lo} in the ATP list.",
        "Background fact: {Lo} was ranked below {H} in the ATP list.",
    ],
    "held": [
        "Background information: {H} had the higher ATP ranking than {Lo}.",
        "Background note: {Lo} trailed {H} in the ATP rankings.",
    ],
    "dirpara": [
        "Background summary: {H} led {Lo} in the ATP ranking order.",
        "Background summary: {Lo} followed {H} on the ATP ranking list.",
    ],
    "nonpara": [
        "Background note mentioned {H} and {Lo}.",
        "Background note listed {Lo} near {H}.",
    ],
}
base.LATER_CTX = {
    "anchor": [
        "Update: {H} was ranked above {Lo} in the ATP list.",
        "Update record: {Lo} was ranked below {H} in the ATP list.",
    ],
    "train": [
        "Update: {H} was ranked above {Lo} in the ATP list.",
        "Update fact: {Lo} was ranked below {H} in the ATP list.",
    ],
    "held": [
        "Update information: {H} had the higher ATP ranking than {Lo}.",
        "Update note: {Lo} trailed {H} in the ATP rankings.",
    ],
    "dirpara": [
        "Update summary: {H} led {Lo} in the ATP ranking order.",
        "Update summary: {Lo} followed {H} on the ATP ranking list.",
    ],
    "nonpara": [
        "Update note mentioned {H} and {Lo}.",
        "Update note listed {Lo} near {H}.",
    ],
}
base.HYP_BEFORE = {
    "train": "Based on the background, {X} was ranked higher than {Y}.",
    "held": "According to the background record, {X} outranked {Y}.",
}
base.HYP_AFTER = {
    "train": "Based on the update, {X} was ranked higher than {Y}.",
    "held": "According to the update record, {X} outranked {Y}.",
}


def section_temporal_context(f, s, am, ctx_grp: str, vi: int, vl: int,
                             order: str = "standard", include_initial: bool = True,
                             include_later: bool = True,
                             include_secondary: bool = True):
    """Replace fixed initial/later addresses by background/update addresses."""
    parts = []
    focal_init = f"Focal background record: {base.sent_initial(f, am, ctx_grp, vi)}"
    focal_later = f"Focal update record: {base.sent_later(f, am, ctx_grp, vl)}"
    sec_init = f"Separate background record: {base.sent_initial(s, am, ctx_grp, vi + 1)}"
    sec_later = f"Separate update record: {base.sent_later(s, am, ctx_grp, vl + 1)}"
    if order == "secondary_first":
        cand = [(include_secondary and include_initial, sec_init), (include_secondary and include_later, sec_later),
                (include_initial, focal_init), (include_later, focal_later)]
    elif order == "later_first":
        cand = [(include_later, focal_later), (include_initial, focal_init),
                (include_secondary and include_later, sec_later), (include_secondary and include_initial, sec_init)]
    else:
        cand = [(include_initial, focal_init), (include_later, focal_later),
                (include_secondary and include_initial, sec_init), (include_secondary and include_later, sec_later)]
    for ok, txt in cand:
        if ok:
            parts.append(txt)
    tpl = f"section_ctx={ctx_grp}|vi={vi}|vl={vl}|order={order}|init={include_initial}|later={include_later}|sec={include_secondary}"
    return " ".join(parts), tpl

base.temporal_context = section_temporal_context

_orig_construction_summary = base.construction_summary

def construction_summary(*args, **kwargs):
    s = _orig_construction_summary(*args, **kwargs)
    out = args[-1] if args else kwargs.get("out")
    s["format_patch"] = "section_address_ceiling"
    s["scientific_boundary"] = (
        "Supplied-address section format: background/update keys recur in context and query; "
        "success is a ceiling, not evidence of general temporal meaning."
    )
    if out is not None:
        Path(out).mkdir(parents=True, exist_ok=True)
        (Path(out) / "construction_summary.json").write_text(json.dumps(s, indent=2, ensure_ascii=False) + "\n", "utf-8")
    return s

base.construction_summary = construction_summary

if __name__ == "__main__":
    base.main()
