#!/usr/bin/env python3
"""Plot H100 mixed-precision ratios by configuration and significant digit.

Run from mp_tests/h100_results, for example:

    python plot_h100_ratios.py 747957 --font-size 12

The script reads cuda_h100_ratios.csv and writes one time-ratio figure and
one memory-ratio figure for each benchmark. If ratio stddev columns are
present, they are rendered as error bars.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path

plt = None


CASE_RE = re.compile(r"^digit(?P<combination>\d+)_(?P<digit>\d+)$")

METRICS = {
    "time_ratio_vs_double": {
        "ylabel": "Time ratio to FP64",
        "filename": "time_ratio",
        "note": "Lower is faster",
        "stddev": "time_ratio_vs_double_stddev",
    },
    "memory_ratio_vs_double": {
        "ylabel": "Memory ratio to FP64",
        "filename": "memory_ratio",
        "note": "Lower is more memory efficient",
        "stddev": None,
    },
}

COLORS = {
    1: "#0072B2",
    2: "#D55E00",
}

BENCHMARK_LABELS = {
    "backprop": "Backprop",
    "dense_lu": "Dense LU",
    "hotspot": "Hotspot",
}

MARKERS = {
    1: "o",
    2: "s",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create publication-quality H100 ratio plots from "
            "cuda_h100_ratios.csv."
        )
    )
    parser.add_argument(
        "result_dir",
        nargs="?",
        default=None,
        help=(
            "Result directory under the current directory, e.g. 747957. "
            "If omitted, the newest directory containing cuda_h100_ratios.csv "
            "is used."
        ),
    )
    parser.add_argument(
        "--csv",
        dest="csv_path",
        default=None,
        help="Explicit path to cuda_h100_ratios.csv.",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Output directory. Default: <result_dir>/figures.",
    )
    parser.add_argument(
        "--font-size",
        type=float,
        default=12.0,
        help="Unified font size for labels, ticks, legends, and annotations.",
    )
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        default=None,
        help="Benchmarks to plot. Default: all benchmarks in the CSV.",
    )
    parser.add_argument(
        "--combinations",
        nargs="+",
        type=int,
        default=[1, 2],
        help="Configuration numbers to plot. Default: 1 2.",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        default=["pdf", "png"],
        choices=["pdf", "png", "svg"],
        help="Output formats. Default: pdf png.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=600,
        help="DPI for raster outputs. Default: 600.",
    )
    return parser.parse_args()


def load_matplotlib() -> None:
    global plt
    try:
        import matplotlib.pyplot as pyplot
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "This script requires matplotlib. Install it with "
            "`python -m pip install matplotlib` or run it in a Python "
            "environment where matplotlib is available."
        ) from exc
    plt = pyplot


def find_default_csv(cwd: Path) -> tuple[Path, Path]:
    candidates = []
    for child in cwd.iterdir():
        csv_path = child / "cuda_h100_ratios.csv"
        if child.is_dir() and csv_path.is_file():
            candidates.append((child.stat().st_mtime, child, csv_path))
    if not candidates:
        raise FileNotFoundError(
            "No result directory containing cuda_h100_ratios.csv was found."
        )
    _, result_dir, csv_path = sorted(candidates)[-1]
    return result_dir, csv_path


def resolve_paths(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    cwd = Path.cwd()
    if args.csv_path:
        csv_path = Path(args.csv_path)
        if not csv_path.is_absolute():
            csv_path = cwd / csv_path
        result_dir = csv_path.parent
    elif args.result_dir:
        result_dir = Path(args.result_dir)
        if not result_dir.is_absolute():
            result_dir = cwd / result_dir
        csv_path = result_dir / "cuda_h100_ratios.csv"
    else:
        result_dir, csv_path = find_default_csv(cwd)

    if not csv_path.is_file():
        raise FileNotFoundError(f"Missing CSV: {csv_path}")

    out_dir = Path(args.out_dir) if args.out_dir else result_dir / "figures"
    if not out_dir.is_absolute():
        out_dir = cwd / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    return result_dir, csv_path, out_dir


def configure_matplotlib(font_size: float) -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": font_size,
            "axes.labelsize": font_size,
            "axes.titlesize": font_size,
            "xtick.labelsize": font_size,
            "ytick.labelsize": font_size,
            "legend.fontsize": font_size,
            "figure.titlesize": font_size,
            "axes.linewidth": 0.9,
            "lines.linewidth": 1.9,
            "lines.markersize": 5.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
        }
    )


def load_rows(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {"benchmark", "case", *METRICS.keys()}
    missing = required - set(rows[0].keys() if rows else [])
    if missing:
        raise ValueError(f"CSV is missing required columns: {sorted(missing)}")
    return rows


def collect_series(
    rows: list[dict[str, str]],
    benchmark: str,
    metric: str,
    combinations: list[int],
) -> tuple[list[int], dict[int, dict[int, float]], dict[int, dict[int, float]]]:
    series: dict[int, dict[int, float]] = {combo: {} for combo in combinations}
    errors: dict[int, dict[int, float]] = {combo: {} for combo in combinations}
    digits = set()
    stddev_col = METRICS[metric]["stddev"]

    for row in rows:
        if row["benchmark"] != benchmark:
            continue
        match = CASE_RE.match(row["case"])
        if not match:
            continue
        combo = int(match.group("combination"))
        digit = int(match.group("digit"))
        if combo not in series:
            continue
        value = float(row[metric])
        series[combo][digit] = value
        if stddev_col and stddev_col in row and row[stddev_col]:
            errors[combo][digit] = float(row[stddev_col])
        digits.add(digit)

    return sorted(digits), series, errors


def finite_values(series: dict[int, dict[int, float]]) -> list[float]:
    values = []
    for by_digit in series.values():
        values.extend(v for v in by_digit.values() if math.isfinite(v))
    return values


def set_y_limits(ax: plt.Axes, values: list[float]) -> None:
    if not values:
        return
    low = min(values + [1.0])
    high = max(values + [1.0])
    pad = max((high - low) * 0.12, 0.02)
    ax.set_ylim(max(0.0, low - pad), high + pad)


def plot_metric(
    rows: list[dict[str, str]],
    benchmark: str,
    metric: str,
    combinations: list[int],
    out_dir: Path,
    formats: list[str],
    dpi: int,
) -> list[Path]:
    digits, series, errors = collect_series(rows, benchmark, metric, combinations)
    if not digits:
        return []

    fig, ax = plt.subplots(figsize=(3.7, 2.75), constrained_layout=True)
    ax.axhline(1.0, color="0.35", linewidth=0.9, linestyle=(0, (3, 2)), zorder=1)

    for combo in combinations:
        y_values = [series[combo].get(digit, math.nan) for digit in digits]
        y_errors = [errors[combo].get(digit, 0.0) for digit in digits]
        common_style = {
            "color": COLORS.get(combo, None),
            "marker": MARKERS.get(combo, "o"),
            "markerfacecolor": "white",
            "markeredgewidth": 1.2,
            "label": f"Combination {combo}",
            "zorder": 3,
        }
        if any(err > 0.0 for err in y_errors):
            ax.errorbar(
                digits,
                y_values,
                yerr=y_errors,
                capsize=2.8,
                elinewidth=0.9,
                capthick=0.9,
                **common_style,
            )
        else:
            ax.plot(digits, y_values, **common_style)

    ax.set_xlabel("Significant digits")
    ax.set_ylabel(METRICS[metric]["ylabel"])
    ax.set_title(BENCHMARK_LABELS.get(benchmark, benchmark.replace("_", " ")))
    ax.set_xticks(digits)
    ax.grid(True, axis="y", color="0.86", linewidth=0.7)
    ax.grid(True, axis="x", color="0.93", linewidth=0.5)
    ax.tick_params(direction="out", length=3.5, width=0.8)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    values = finite_values(series)
    set_y_limits(ax, values)

    ax.text(
        0.98,
        0.96,
        METRICS[metric]["note"],
        transform=ax.transAxes,
        ha="right",
        va="top",
        color="0.25",
        fontsize=plt.rcParams["font.size"] * 0.86,
    )
    ax.legend(frameon=False, loc="best", handlelength=2.3)

    saved = []
    combo_tag = "_".join(str(combo) for combo in combinations)
    stem = f"{benchmark}_{METRICS[metric]['filename']}_combinations_{combo_tag}"
    for fmt in formats:
        path = out_dir / f"{stem}.{fmt}"
        fig.savefig(path, dpi=dpi if fmt == "png" else None)
        saved.append(path)
    plt.close(fig)
    return saved


def main() -> None:
    args = parse_args()
    _, csv_path, out_dir = resolve_paths(args)
    load_matplotlib()
    configure_matplotlib(args.font_size)
    rows = load_rows(csv_path)

    benchmarks = args.benchmarks or sorted({row["benchmark"] for row in rows})
    saved_paths = []
    for benchmark in benchmarks:
        for metric in METRICS:
            saved_paths.extend(
                plot_metric(
                    rows,
                    benchmark,
                    metric,
                    args.combinations,
                    out_dir,
                    args.formats,
                    args.dpi,
                )
            )

    if not saved_paths:
        raise RuntimeError("No figures were generated.")

    print(f"Read: {csv_path}")
    print(f"Wrote {len(saved_paths)} figure files to: {out_dir}")
    for path in saved_paths:
        print(path)


if __name__ == "__main__":
    main()
