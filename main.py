from __future__ import annotations

import argparse

from utils.cli_commands import cmd_examples


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Wheres-Wally training and visualization CLI"
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # examples: show a few annotated images
    examples_parser = subparsers.add_parser(
        "examples", help="Show example images with bounding boxes"
    )

    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.command == "examples":
        cmd_examples(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
