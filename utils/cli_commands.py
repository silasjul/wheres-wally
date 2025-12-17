import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image

from utils.dataset import CocoDetectionDataset, print_examples


def cmd_examples(args: argparse.Namespace) -> None:
    # Always use the train split and show 4 random examples.
    base_dir = Path("data")
    split_dir = base_dir / "train"
    ann_file = split_dir / "_annotations.coco.json"

    if not ann_file.exists():
        raise FileNotFoundError(f"Annotations file not found: {ann_file}")

    dataset = CocoDetectionDataset(
        images_dir=split_dir,
        annotations_file=ann_file,
    )

    print("Showing 4 random example(s) from 'train'")
    print_examples(dataset=dataset)


def cmd_find_wally(args: argparse.Namespace) -> None:
    """
    Show one of the original images (1-12) without annotations,
    so the user can try to find Wally manually.
    """
    img_number = args.image_number
    if img_number < 1 or img_number > 12:
        raise ValueError("image_number must be between 1 and 12")

    base_dir = Path("data") / "original_images"
    img_path = base_dir / f"{img_number}.jpg"

    if not img_path.exists():
        raise FileNotFoundError(f"Original image not found: {img_path}")

    print(f"Opening original image #{img_number}: {img_path}")

    image = Image.open(img_path).convert("RGB")

    plt.figure(figsize=(10, 6))
    plt.imshow(image)
    plt.axis("off")
    plt.title(f"Find Wally! (image {img_number})")
    plt.show()