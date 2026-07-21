"""CarDD (COCO) — dataset multi-label « type de dégât », pilier État OscarIA.

Cible = vecteur binaire 6-dim (présence de chaque type de dégât sur l'image).
6 classes dans l'ordre des category_id COCO (id 1..6), figé pour un encodage stable.
Décision : ADR 0006.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset

# Ordre figé = category_id COCO 1..6 (voir EDA). Ne pas réordonner : casse l'encodage.
CLASSES = ["dent", "scratch", "crack", "glass shatter", "lamp broken", "tire flat"]
SPLIT_DIR = {"train": "train2017", "val": "val2017", "test": "test2017"}


class CarDDMultiLabel(Dataset):
    """Dataset multi-label CarDD à partir des annotations COCO.

    `coco_root` pointe sur `.../CarDD_release/CarDD_COCO`.
    Renvoie `(image_transformée, cible_6d_float)`.
    """

    def __init__(self, coco_root: str | Path, split: str = "train", transform=None):
        self.root = Path(coco_root)
        self.split = split
        self.transform = transform
        self.img_dir = self.root / SPLIT_DIR[split]
        self.cls_index = {name: i for i, name in enumerate(CLASSES)}

        ann = json.load(open(self.root / "annotations" / f"instances_{SPLIT_DIR[split]}.json"))
        cats = {c["id"]: c["name"] for c in ann["categories"]}
        self.files = {img["id"]: img["file_name"] for img in ann["images"]}

        # vecteur cible par image (0 partout puis on allume les classes présentes)
        self.labels = {iid: torch.zeros(len(CLASSES)) for iid in self.files}
        for a in ann["annotations"]:
            self.labels[a["image_id"]][self.cls_index[cats[a["category_id"]]]] = 1.0

        self.ids = list(self.files)

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx: int):
        iid = self.ids[idx]
        img = Image.open(self.img_dir / self.files[iid]).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, self.labels[iid]

    def label_matrix(self) -> torch.Tensor:
        """Matrice (N, 6) de toutes les cibles — pour stats/pondération."""
        return torch.stack([self.labels[i] for i in self.ids])

    def pos_weight(self) -> torch.Tensor:
        """`pos_weight` pour `BCEWithLogitsLoss` = nb_négatifs / nb_positifs par classe.

        Compense le déséquilibre (scratch ~7× tire flat) : pèse plus les classes rares.
        """
        y = self.label_matrix()
        pos = y.sum(0)
        neg = len(self.ids) - pos
        return neg / pos.clamp(min=1)
