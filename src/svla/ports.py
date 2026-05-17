from __future__ import annotations

import argparse

from serial.tools import list_ports


def list_serial_ports() -> int:
    ports = list(list_ports.comports())
    if not ports:
        print("No serial ports found.")
        return 1

    for port in ports:
        print(f"{port.device}: {port.description}")
        print(f"  hwid: {port.hwid}")
    return 0


def list_ports_command(_: argparse.Namespace) -> int:
    return list_serial_ports()


def register_parsers(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("list-ports")
    parser.set_defaults(func=list_ports_command)
