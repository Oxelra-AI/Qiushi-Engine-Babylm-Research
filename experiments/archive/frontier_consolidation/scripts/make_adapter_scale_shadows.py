#!/usr/bin/env python3
"""research: create inference-scale shadow checkpoints for the research live adapter.

The model weights are exactly the research live adapter128 matched-horizon 20M
checkpoint.  Only config.json plus the local modeling source are changed so that
adapter output is multiplied by `adapter_scale` at inference.  This is an endpoint
sensitivity map, not a causal decomposition of branch output vs backbone drift.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import json
import os
import shutil
from pathlib import Path

USER_ROOT = _public_path('.')
STUDY = _public_path('experiments/archive/frontier_consolidation')
WORKSPACE = _public_path('experiments/archive/frontier_consolidation')
SRC_RUN = _public_path('experiments/archive/frontier_consolidation/training/runs/adapter128_live_h100M20M_seed43022')
SRC_CKPT = _public_path('experiments/archive/frontier_consolidation/training/runs/adapter128_live_h100M20M_seed43022/hf_model/chck_20M')
OUT_ROOT = _public_path('experiments/archive/frontier_consolidation/data/adapter_scale_sweep/shadow_runs')

MODEL_CODE = '''#!/usr/bin/env python3
"""research scaleable residual-adapter DeBERTa-v2 MLM shadow.

This source is checkpoint-local.  It is identical in state-dict layout to the
research/103 adapter model but multiplies adapter updates by config.adapter_scale
at inference/training time.  Used only for endpoint sensitivity maps.
"""
from __future__ import annotations
import shutil
from pathlib import Path
from typing import Optional
import torch
from torch import nn
from transformers import DebertaV2Config, DebertaV2ForMaskedLM
from transformers.activations import ACT2FN
from transformers.models.deberta_v2.modeling_deberta_v2 import DebertaV2Layer

_THIS_FILE = Path(__file__).resolve()

class ZeroOutputBottleneckAdapter(nn.Module):
    def __init__(self, config: DebertaV2Config):
        super().__init__()
        width = int(getattr(config, "adapter_bottleneck", 64))
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.down = nn.Linear(config.hidden_size, width)
        self.up = nn.Linear(width, config.hidden_size)
        self.activation = ACT2FN[getattr(config, "adapter_activation", "gelu")]
        self.enabled = bool(getattr(config, "adapter_enabled", True))
        self.scale = float(getattr(config, "adapter_scale", 1.0))
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)
        self.last_rms = 0.0
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if not self.enabled or self.scale == 0.0:
            self.last_rms = 0.0
            return torch.zeros_like(hidden_states)
        update = self.up(self.activation(self.down(self.layer_norm(hidden_states)))) * self.scale
        if not torch.jit.is_tracing():
            self.last_rms = float(update.detach().float().square().mean().sqrt().cpu())
        return update

class AdapterDebertaV2Layer(DebertaV2Layer):
    def forward(self, hidden_states, attention_mask, query_states=None, relative_pos=None,
                rel_embeddings=None, output_attentions: bool = False) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        attention_output, att_matrix = self.attention(
            hidden_states, attention_mask, output_attentions=output_attentions,
            query_states=query_states, relative_pos=relative_pos, rel_embeddings=rel_embeddings)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        layer_output = layer_output + self.adapter(layer_output)
        return (layer_output, att_matrix if output_attentions else None)

class AdapterDebertaV2ForMaskedLM(DebertaV2ForMaskedLM):
    config_class = DebertaV2Config
    def __init__(self, config: DebertaV2Config):
        super().__init__(config)
        config.adapter_bottleneck = int(getattr(config, "adapter_bottleneck", 64))
        config.adapter_activation = getattr(config, "adapter_activation", "gelu")
        config.adapter_enabled = bool(getattr(config, "adapter_enabled", True))
        config.adapter_scale = float(getattr(config, "adapter_scale", 1.0))
        config.architectures = [self.__class__.__name__]
        config.auto_map = {"AutoModelForMaskedLM": "adapter_modeling.AdapterDebertaV2ForMaskedLM"}
        for layer in self.deberta.encoder.layer:
            layer.__class__ = AdapterDebertaV2Layer
            layer.adapter = ZeroOutputBottleneckAdapter(config)
    def adapter_parameters(self):
        for name, parameter in self.named_parameters():
            if ".adapter." in name:
                yield name, parameter
    def adapter_rms(self) -> list[float]:
        return [float(layer.adapter.last_rms) for layer in self.deberta.encoder.layer]
    def save_pretrained(self, save_directory, *args, **kwargs):
        super().save_pretrained(save_directory, *args, **kwargs)
        dst = Path(save_directory) / _THIS_FILE.name
        if not dst.exists():
            shutil.copy2(_THIS_FILE, dst)
'''


def scale_tag(scale: float) -> str:
    s = f"{scale:.2f}".replace(".", "p")
    return s.replace("-", "m")


def link_or_copy(src: Path, dst: Path) -> None:
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def make_shadow(scale: float, force: bool = False) -> dict:
    if not (_public_path('experiments/archive/frontier_consolidation/training/runs/adapter128_live_h100M20M_seed43022/hf_model/chck_20M/model.safetensors')).exists():
        raise FileNotFoundError(_public_path('experiments/archive/frontier_consolidation/training/runs/adapter128_live_h100M20M_seed43022/hf_model/chck_20M/model.safetensors'))
    tag = scale_tag(scale)
    run_dir = OUT_ROOT / f"adapter128_scale_{tag}_from_live20M"
    ckpt = run_dir / "hf_model/chck_20M"
    ckpt.mkdir(parents=True, exist_ok=True)
    for name in ["model.safetensors", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"]:
        src = SRC_CKPT / name
        dst = ckpt / name
        if src.exists():
            if name == "model.safetensors":
                if force or not dst.exists():
                    link_or_copy(src, dst)
            else:
                shutil.copy2(src, dst)
    cfg = json.loads((_public_path('experiments/archive/frontier_consolidation/training/runs/adapter128_live_h100M20M_seed43022/hf_model/chck_20M/config.json')).read_text(encoding="utf-8"))
    cfg["adapter_scale"] = float(scale)
    cfg["adapter_enabled"] = True
    cfg["architectures"] = ["AdapterDebertaV2ForMaskedLM"]
    cfg["auto_map"] = {"AutoModelForMaskedLM": "adapter_modeling.AdapterDebertaV2ForMaskedLM"}
    (ckpt / "config.json").write_text(json.dumps(cfg, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (ckpt / "adapter_modeling.py").write_text(MODEL_CODE, encoding="utf-8")
    src_metrics = _public_path('experiments/archive/frontier_consolidation/training/runs/adapter128_live_h100M20M_seed43022/scientific_metrics.json')
    metrics = json.loads(src_metrics.read_text(encoding="utf-8")) if src_metrics.exists() else {}
    metrics["shadow_source_run"] = str(SRC_RUN)
    metrics["shadow_source_checkpoint"] = str(SRC_CKPT)
    metrics["adapter_scale"] = float(scale)
    metrics["variant"] = f"inference_scaled_adapter128_scale_{tag}_from_step103_live20M"
    (run_dir / "scientific_metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return {"scale": float(scale), "tag": tag, "run_dir": str(run_dir), "checkpoint": str(ckpt)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scales", nargs="*", type=float, default=[0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5])
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    records = [make_shadow(s, args.force) for s in args.scales]
    out_json = _public_path('experiments/archive/frontier_consolidation/data/adapter_scale_sweep/adapter_scale_shadows.json')
    out_json.write_text(json.dumps({"status": "ok", "source_checkpoint": str(SRC_CKPT), "records": records}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": "ok", "out_json": str(out_json), "n": len(records), "records": records}, indent=2), flush=True)


if __name__ == "__main__":
    main()
