from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import json
import random

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset


class CocoDetectionDataset(Dataset):
    """
    Simple COCO-style detection dataset tailored for the Wheres-Wally data.

    It expects a Roboflow-style COCO JSON (like `_annotations.coco.json`) with:
    - `images`: list of {id, file_name, width, height, ...}
    - `annotations`: list of {image_id, category_id, bbox, area, iscrowd, ...}
    - `categories`: list of {id, name, ...}

    Each `__getitem__` returns:
        image: FloatTensor (C, H, W) in [0, 1] (unless your transform changes this)
        target: dict with keys:
            - boxes: FloatTensor [N, 4] in (x_min, y_min, x_max, y_max)
            - labels: LongTensor [N]
            - image_id: Tensor [1]
            - area: FloatTensor [N]
            - iscrowd: LongTensor [N]
            - (optionally) orig_size: Tensor [2] (H, W)

    You can pass transforms that receive and return (image, target) pairs,
    which keeps it compatible with torchvision detection models.
    """

    def __init__(
        self,
        images_dir: str | Path,
        annotations_file: str | Path,
        transforms: Optional[
            Callable[[Tensor, Dict[str, Tensor]], Tuple[Tensor, Dict[str, Tensor]]]
        ] = None,
    ) -> None:
        super().__init__()

        self.images_dir = Path(images_dir)
        self.annotations_file = Path(annotations_file)
        self.transforms = transforms

        # Load COCO-style JSON
        with self.annotations_file.open("r", encoding="utf-8") as f:
            coco = json.load(f)

        images = coco.get("images", [])
        annotations = coco.get("annotations", [])
        categories = coco.get("categories", [])

        # Index images by internal index -> image dict
        # and by image_id -> annotation list
        self.images: List[Dict[str, Any]] = images
        self.id_to_image: Dict[int, Dict[str, Any]] = {img["id"]: img for img in images}

        self.image_id_to_annots: Dict[int, List[Dict[str, Any]]] = {}
        for ann in annotations:
            img_id = ann["image_id"]
            self.image_id_to_annots.setdefault(img_id, []).append(ann)

        # Build category mapping in case you want to use contiguous labels 1..K
        self.categories: List[Dict[str, Any]] = categories
        self.cat_id_to_name: Dict[int, str] = {c["id"]: c["name"] for c in categories}

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.images)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Dict[str, Tensor]]:  # type: ignore[override]
        img_info = self.images[idx]
        img_id = img_info["id"]
        file_name = img_info["file_name"]

        # Load image
        img_path = self.images_dir / file_name
        image = Image.open(img_path).convert("RGB")
        image = self._pil_to_tensor(image)

        # Collect annotations for this image
        annots = self.image_id_to_annots.get(img_id, [])

        boxes: List[List[float]] = []
        labels: List[int] = []
        areas: List[float] = []
        iscrowd: List[int] = []

        for ann in annots:
            # COCO bbox format is [x, y, w, h]
            x, y, w, h = ann["bbox"]
            x2 = x + w
            y2 = y + h
            boxes.append([x, y, x2, y2])

            labels.append(int(ann["category_id"]))
            areas.append(float(ann.get("area", w * h)))
            iscrowd.append(int(ann.get("iscrowd", 0)))

        target: Dict[str, Tensor] = {}
        if boxes:
            target["boxes"] = torch.tensor(boxes, dtype=torch.float32)
            target["labels"] = torch.tensor(labels, dtype=torch.int64)
            target["area"] = torch.tensor(areas, dtype=torch.float32)
            target["iscrowd"] = torch.tensor(iscrowd, dtype=torch.int64)
        else:
            # Ensure empty tensors have the right shapes/dtypes
            target["boxes"] = torch.zeros((0, 4), dtype=torch.float32)
            target["labels"] = torch.zeros((0,), dtype=torch.int64)
            target["area"] = torch.zeros((0,), dtype=torch.float32)
            target["iscrowd"] = torch.zeros((0,), dtype=torch.int64)

        target["image_id"] = torch.tensor([img_id], dtype=torch.int64)
        # Optional: keep original size; handy for metrics or visualization
        h = img_info.get("height")
        w = img_info.get("width")
        if h is not None and w is not None:
            target["orig_size"] = torch.tensor([h, w], dtype=torch.int64)

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target

    @staticmethod
    def _pil_to_tensor(image: Image.Image) -> Tensor:
        """Convert a PIL image to float tensor in [0, 1] with shape (C, H, W)."""
        # Using torchvision is standard, but to keep this dataset self-contained
        # we manually convert here.
        img = torch.as_tensor(
            bytearray(image.tobytes()),
            dtype=torch.uint8,
        ).view(image.size[1], image.size[0], 3)  # (H, W, C)
        img = img.permute(2, 0, 1).float() / 255.0
        return img


def coco_detection_collate_fn(
    batch: List[Tuple[Tensor, Dict[str, Tensor]]]
) -> Tuple[Tensor, List[Dict[str, Tensor]]]:
    """
    Collate function for detection models.

    - Stacks images into a batch tensor [B, C, H, W] (assumes all same size).
    - Leaves targets as a list of dicts (one per image).

    Use with DataLoader:
        DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=coco_detection_collate_fn)
    """
    images, targets = zip(*batch)
    images_tensor = torch.stack(images, dim=0)
    return images_tensor, list(targets)


def print_examples(
    dataset: CocoDetectionDataset,
    figsize: Tuple[int, int] = (12, 12),
) -> None:
    """
    Visualize a few examples from the dataset with bounding boxes.

    Parameters
    ----------
    dataset : CocoDetectionDataset
        The dataset to visualize (train split recommended).
    figsize : (w, h)
        Matplotlib figure size.
    """
    n = min(4, len(dataset))
    if n <= 0:
        print("No examples to display.")
        return

    # Randomly sample distinct indices
    indices = random.sample(range(len(dataset)), n)

    for idx in indices:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        image, target = dataset[idx]
        img_np = image.permute(1, 2, 0).cpu().numpy()  # (H, W, C)

        ax.imshow(img_np)

        boxes = target.get("boxes")
        labels = target.get("labels")

        if boxes is not None and labels is not None and len(boxes) > 0:
            for box, label in zip(boxes, labels):
                x1, y1, x2, y2 = box.tolist()
                w = x2 - x1
                h = y2 - y1

                cat_id = int(label.item())
                name = dataset.cat_id_to_name.get(cat_id, str(cat_id))
                name_lower = name.lower()

                # Fixed colors per character class:
                # - wally      -> lime
                # - wenda      -> red
                # - odwally    (odwald) -> yellow
                # - wizard_*   -> cyan
                if "wenda" in name_lower:
                    color = "red"
                elif name_lower.startswith("odw"):
                    color = "yellow"
                elif "wizard" in name_lower:
                    color = "cyan"
                elif "wally" in name_lower:
                    color = "lime"
                else:
                    color = "white"

                rect = patches.Rectangle(
                    (x1, y1),
                    w,
                    h,
                    linewidth=2,
                    edgecolor=color,
                    facecolor="none",
                )
                ax.add_patch(rect)

                ax.text(
                    x1,
                    y1,
                    name,
                    fontsize=8,
                    color="black",
                    bbox=dict(facecolor=color, alpha=0.6, pad=1),
                )

        ax.set_title(f"idx={idx}")
        ax.axis("off")

    # Show all created figures; each example is in its own window.
    plt.show()


__all__ = [
    "CocoDetectionDataset",
    "coco_detection_collate_fn",
    "slice_coco_dataset",
    "print_examples",
]


