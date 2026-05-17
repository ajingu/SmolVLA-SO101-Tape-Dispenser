from __future__ import annotations

import argparse
import sys

from svla import cameras, ports, soarm


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="so101")
    subparsers = parser.add_subparsers(dest="command", required=True)

    cameras.register_parsers(subparsers)
    ports.register_parsers(subparsers)
    soarm.register_parsers(subparsers)

    return parser


def main() -> int:
    args = build_parser().parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
