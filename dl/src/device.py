"""Sélection du device et détection machine — pilier État (DL) OscarIA.

Deux besoins distincts :
- `get_device()` : choix du device **au runtime** (MPS sur Mac, CUDA sur le PC, repli CPU).
- `describe_machine()` / `recommend_install()` : **check machine** pour savoir quel
  environnement PyTorch installer (wheels MPS par défaut sur Mac Apple Silicon, index CUDA
  12.8 sur un PC Linux/Windows avec GPU NVIDIA Blackwell type RTX 5070 Ti).
"""

from __future__ import annotations

import platform


def get_device(prefer: str = "auto"):
    """Renvoie le meilleur `torch.device` disponible.

    - `prefer="auto"` : MPS (Mac) > CUDA (PC) > CPU.
    - `prefer` ∈ {"mps", "cuda", "cpu"} force le choix (repli silencieux sur CPU si indispo).
    Import de torch différé pour que `describe_machine()` reste utilisable sans torch installé.
    """
    import torch

    if prefer == "cpu":
        return torch.device("cpu")
    if prefer in ("mps", "auto") and torch.backends.mps.is_available():
        return torch.device("mps")
    if prefer in ("cuda", "auto") and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def describe_machine() -> dict:
    """Infos machine utiles au choix d'environnement (sans dépendre de torch)."""
    return {
        "system": platform.system(),        # Darwin / Linux / Windows
        "machine": platform.machine(),       # arm64 / x86_64 ...
        "processor": platform.processor(),
        "python": platform.python_version(),
    }


def recommend_install() -> str:
    """Recommande l'installation PyTorch selon la machine (heuristique, pas d'exécution).

    Ne détecte pas le GPU NVIDIA lui-même (nvidia-smi non appelé) : on se base sur l'OS/arch.
    Le smoke test confirme ensuite que le device attendu est bien exploitable.
    """
    m = describe_machine()
    if m["system"] == "Darwin" and m["machine"] == "arm64":
        return (
            "Mac Apple Silicon → PyTorch par défaut (wheels PyPI, backend MPS). "
            "Ex : `uv add torch torchvision` (device attendu : mps)."
        )
    if m["system"] in ("Linux", "Windows"):
        return (
            "PC avec GPU NVIDIA (RTX 5070 Ti = Blackwell sm_120) → build CUDA 12.8. "
            "Ex : `uv add torch torchvision --index-url "
            "https://download.pytorch.org/whl/cu128` (device attendu : cuda). "
            "Nightly si sm_120 non couvert par le stable."
        )
    return "Machine non reconnue → repli CPU (`uv add torch torchvision`, device : cpu)."
