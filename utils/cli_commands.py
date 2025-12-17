import argparse
from pathlib import Path

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