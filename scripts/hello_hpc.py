"""hello_hpc.py — introspect what Python sees inside a PBS job."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import platform
import shutil
import socket
import sys
from pathlib import Path
from textwrap import indent


def section(title: str) -> None:
    print(f"\n---------- {title} ----------")


def kv(label: str, value) -> None:
    print(f"  {label:<22}: {value}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scratch",
        type=Path,
        required=True,
        help="Scratch directory created by the shell job script.",
    )
    args = parser.parse_args()


if __name__ == "__main__":
    main()
