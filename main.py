from __future__ import annotations

import argparse

from utils.cli_commands import cmd_build_dataset, cmd_find_wally


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Wheres-Wally training and visualization CLI"
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # examples: show a few annotated images with bounding boxes
    subparsers.add_parser(
        "examples",
        help="Show random annotated training images with bounding boxes",
    )

    # find-wally: show one original page by number (1-12)
    find_parser = subparsers.add_parser(
        "find-wally",
        help="Show an original image (1-12) so you can try to find Wally",
    )
    find_parser.add_argument(
        "image_number",
        type=int,
        help="Image number from 1 to 12 (corresponds to data/original_images/<n>.jpg)",
    )

    # build-dataset: slice and balance the dataset after data has been unzipped
    subparsers.add_parser(
        "build-dataset",
        help="Slice COCO datasets and balance backgrounds after unzipping data into data/train, data/valid, and data/test",
    )

    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.command == "examples":
        cmd_examples(args)
    elif args.command == "find-wally":
        cmd_find_wally(args)
    elif args.command == "build-dataset":
        cmd_build_dataset(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
