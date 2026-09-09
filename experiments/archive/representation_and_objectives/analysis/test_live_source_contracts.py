"""Static launch-contract tests against the live research sources as audited."""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import ast
from pathlib import Path


ROOT = _public_path('.')
MODULE = _public_path('experiments/archive/representation_and_objectives/scripts/sgcr_module.py')
TRAINER = _public_path('experiments/archive/representation_and_objectives/scripts/sgcr_trainer.py')
BASE = _public_path('experiments/archive/representation_and_objectives/scripts/accumulated_masking_curriculum_trainer.py')


def test_live_module_parses():
    ast.parse(MODULE.read_text(encoding="utf-8"), filename=str(MODULE))


def test_live_trainer_parses():
    ast.parse(TRAINER.read_text(encoding="utf-8"), filename=str(TRAINER))


def test_live_trainer_uses_valid_curriculum_mode_accessor():
    text = TRAINER.read_text(encoding="utf-8")
    assert "curriculum_state.current_mode_label" not in text
    assert "curriculum_state.get_current_mask_mode()" in text


def test_live_trainer_preserves_matched_gradient_clipping():
    baseline = BASE.read_text(encoding="utf-8")
    treatment = TRAINER.read_text(encoding="utf-8")
    assert "clip_grad_norm_" in baseline
    assert "clip_grad_norm_" in treatment


def test_live_trainer_exposes_random_decomposition_control():
    text = TRAINER.read_text(encoding="utf-8")
    assert "random_decomposition" in text
