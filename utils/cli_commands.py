import argparse
from pathlib import Path

from utils.slicing import balance_dataset, slice_coco_dataset


def cmd_find_wally(args: argparse.Namespace) -> None:
    """
    Show one of the original images (1-12) without annotations,
    so the user can try to find Wally manually.
    """
    img_number = args.image_number
    if img_number < 1 or img_number > 12:
        raise ValueError("image_number must be between 1 and 12")

    base_dir = Path("original_images")
    img_path = base_dir / f"{img_number}.jpg"

    if not img_path.exists():
        raise FileNotFoundError(f"Original image not found: {img_path}")

    print(f"Original image path: {img_path}")
    print("Displaying the image is not yet implemented in this CLI.")


def cmd_build_dataset(args: argparse.Namespace) -> None:
    """
    Build the sliced and balanced dataset.

    Assumes that the COCO annotations and images have already been
    unzipped into the `data/train`, `data/valid`, and `data/test`
    folders.
    """
    print("Running slice_coco_dataset() ...")
    slice_coco_dataset()
    print("Running balance_dataset() ...")
    balance_dataset()
    print("Dataset build completed.")