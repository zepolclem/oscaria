"""Sélection du device et détection machine — pilier DL OscarIA.

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
            "PC avec GPU NVIDIA (RTX 5070 Ti = Blackwell sm_120) → build CUDA 12.8.\n"
            "  1. `uv sync --all-packages`            # installe la variante CPU\n"
            "  2. `uv pip install --torch-backend=cu128 torch torchvision`\n"
            "  (device attendu : cuda ; nightly si sm_120 n'est pas couvert par le stable)\n"
            "\n"
            "  Pourquoi deux étapes. Sous Linux, `pyproject.toml` épingle torch à l'index\n"
            "  `pytorch-cpu` : c'est ce qui fait tomber l'image Docker de 5,53 Go à 1,5 Go\n"
            "  en écartant 2,9 Go de pilotes `nvidia-*` inutiles dans un conteneur CPU.\n"
            "  L'image de l'application et ce PC étant tous deux sous Linux, aucun marqueur\n"
            "  de plateforme ne peut les distinguer.\n"
            "\n"
            "  La forme documentée par uv — deux extras `cpu`/`cu128` plus\n"
            "  `[tool.uv] conflicts` — a été essayée et ne fonctionne pas dans cet espace de\n"
            "  travail : uv n'y lit pas `conflicts` (membres en `package = false`, vérifié —\n"
            "  un nom de paquet volontairement faux n'y déclenche aucune erreur), et rejette\n"
            "  alors deux index explicites pour un même paquet.\n"
            "  D'où la surcouche `uv pip`, seule interface qui accepte `--torch-backend`.\n"
            "  Conséquence à connaître : l'étape 2 installe une version que `uv.lock` ne\n"
            "  décrit pas, et un `uv sync` ultérieur la réécrasera par la variante CPU."
        )
    return "Machine non reconnue → repli CPU (`uv add torch torchvision`, device : cpu)."
