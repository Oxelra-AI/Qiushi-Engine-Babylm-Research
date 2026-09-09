
from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative

from pathlib import Path
import os

import torch
import torch.distributed as dist

def get_base_dir() -> Path:
    # Return the project root directory.
    return _public_path('experiments/archive/initial_model_studies/data/public_recgpt_runtime')


def setup_distributed() -> tuple[torch.device, int, int, int, bool]:
    ddp = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    if not torch.cuda.is_available():
        raise RuntimeError("Training requires CUDA/ROCm; CPU/MPS are intentionally unsupported.")
    if not ddp:
        return torch.device("cuda"), 0, 0, 1, False

    dist.init_process_group(backend="nccl")
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    return torch.device("cuda", local_rank), rank, local_rank, world_size, True


def print0(*args, rank: int, **kwargs):
    if rank == 0:
        print(*args, **kwargs)
