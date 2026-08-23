#!/usr/bin/env python3
"""Generate deterministic Hotspot input files for large H100 runs."""

from __future__ import annotations

import argparse
import math
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Hotspot temperature and power input files."
    )
    parser.add_argument("rows", type=int)
    parser.add_argument("cols", type=int)
    parser.add_argument("temp_file")
    parser.add_argument("power_file")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing files.",
    )
    return parser.parse_args()


def write_inputs(rows: int, cols: int, temp_path: Path, power_path: Path) -> None:
    temp_path.parent.mkdir(parents=True, exist_ok=True)
    power_path.parent.mkdir(parents=True, exist_ok=True)
    with temp_path.open("w", buffering=1024 * 1024) as temp_file, power_path.open(
        "w", buffering=1024 * 1024
    ) as power_file:
        row_scale = max(rows - 1, 1)
        col_scale = max(cols - 1, 1)
        for row in range(rows):
            y = row / row_scale
            row_hotspot = math.exp(-((y - 0.47) ** 2) / 0.018)
            for col in range(cols):
                x = col / col_scale
                wave = math.sin(2.0 * math.pi * x) * math.cos(math.pi * y)
                bump = math.exp(-(((x - 0.62) ** 2) + ((y - 0.38) ** 2)) / 0.012)
                temp = 318.0 + 9.0 * x + 6.0 * y + 2.5 * wave + 4.0 * bump
                power = 3.2e-5 + 1.1e-5 * bump + 4.0e-6 * row_hotspot
                temp_file.write(f"{temp:.9f}\n")
                power_file.write(f"{power:.12f}\n")


def main() -> None:
    args = parse_args()
    if args.rows <= 1 or args.cols <= 1:
        raise SystemExit("rows and cols must both be greater than one")

    temp_path = Path(args.temp_file)
    power_path = Path(args.power_file)
    if not args.force and temp_path.exists() and power_path.exists():
        print(f"Hotspot inputs already exist: {temp_path} {power_path}")
        return

    write_inputs(args.rows, args.cols, temp_path, power_path)
    print(f"Wrote Hotspot inputs: {temp_path} {power_path}")


if __name__ == "__main__":
    main()
