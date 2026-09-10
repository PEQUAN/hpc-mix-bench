#!/usr/bin/env python3
"""Plot compact H100 memory and time ratios by benchmark.

Run from papers/h100_results, for example:

    python plot_h100_benchmark_panels.py --font-size 10

By default, the script scans all result directories under papers/h100_results
that contain cuda_h100_ratios.csv, combines their benchmark rows, and writes a
publication-oriented figure with one panel per benchmark.  Each panel overlays
memory-ratio line plots with time-ratio bars, both normalized to FP64.
Accuracy/error columns are ignored on purpose: the paper figure should focus on
resource use and runtime only.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path

plt = None


CASE_RE = re.compile(r"^digit(?P<combination>\d+)_(?P<digit>\d+)$")

BENCHMARK_ORDER = ["backprop", "hotspot", "dense_lu"]

BENCHMARK_LABELS = {
    "backprop": "BackProp",
    "hotspot": "HotSpot",
    "dense_lu": "Dense LU",
}

COMBINATION_LABELS = {
    1: "Combination I",
    2: "Combination II",
}

COLORS = {
    1: "#2B6CB0",
    2: "#B83280",
}

MARKERS = {
    1: "o",
    2: "s",
}

HATCHES = {
    1: "",
    2: "///",
}

LINESTYLES = {
    1: "-",
    2: (0, (4, 2)),
}

REQUIRED_COLUMNS = {
    "benchmark",
    "case",
    "memory_ratio_vs_double",
    "time_ratio_vs_double",
}


def combination_label(combo: int) -> str:
    """Return a readable label for a precision combination."""
    return COMBINATION_LABELS.get(combo, f"Combination {combo}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create compact H100 benchmark panels from cuda_h100_ratios.csv, "
            "showing memory as lines and runtime as bars."
        )
    )
    parser.add_argument(
        "result_dirs",
        nargs="*",
        default=None,
        help=(
            "Optional result directories under the current directory, e.g. "
            "1284441 1284476 1326840. Unique prefixes are also accepted. "
            "If omitted, all directories containing cuda_h100_ratios.csv are used."
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
        help=(
            "Output directory. Default: <result_dir>/figures for one input "
            "directory, otherwise ./figures."
        ),
    )
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        default=None,
        help="Benchmarks to plot. Default: BackProp, HotSpot, Dense LU if present.",
    )
    parser.add_argument(
        "--combinations",
        nargs="+",
        type=int,
        default=[1, 2],
        help="Precision combination numbers to plot. Default: 1 2.",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        default=["pdf", "png"],
        choices=["pdf", "png", "svg"],
        help="Output formats. Default: pdf png.",
    )
    parser.add_argument(
        "--font-size",
        type=float,
        default=9.5,
        help="Unified font size for labels, ticks, legends, and annotations.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=600,
        help="DPI for raster outputs. Default: 600.",
    )
    parser.add_argument(
        "--separate",
        action="store_true",
        help="Also write one standalone figure per benchmark.",
    )
    return parser.parse_args()


def load_matplotlib() -> None:
    global plt
    try:
        import matplotlib.pyplot as pyplot
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "This script requires matplotlib. Install it with "
            "`python -m pip install matplotlib` or run it in an environment "
            "where matplotlib is available."
        ) from exc
    plt = pyplot


def find_result_csvs(cwd: Path) -> list[tuple[float, Path, Path]]:
    candidates = []
    for child in cwd.iterdir():
        csv_path = child / "cuda_h100_ratios.csv"
        if child.is_dir() and csv_path.is_file():
            candidates.append((child.stat().st_mtime, child, csv_path))
    return sorted(candidates)


def format_result_dirs(candidates: list[tuple[float, Path, Path]]) -> str:
    if not candidates:
        return "none"
    return ", ".join(result_dir.name for _, result_dir, _ in candidates)


def resolve_result_dir(
    cwd: Path,
    result_dir_arg: str,
    candidates: list[tuple[float, Path, Path]],
) -> tuple[Path, Path]:
    result_dir = Path(result_dir_arg)
    if result_dir.is_absolute():
        csv_path = result_dir / "cuda_h100_ratios.csv"
        if csv_path.is_file():
            return result_dir, csv_path
    else:
        exact_dir = cwd / result_dir
        exact_csv = exact_dir / "cuda_h100_ratios.csv"
        if exact_csv.is_file():
            return exact_dir, exact_csv

        matches = [
            (candidate_dir, csv_path)
            for _, candidate_dir, csv_path in candidates
            if candidate_dir.name.startswith(result_dir_arg)
        ]
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise FileNotFoundError(
                f"Ambiguous result directory prefix: {result_dir_arg}\n"
                "Matching result directories: "
                f"{', '.join(path.name for path, _ in matches)}"
            )

    available = format_result_dirs(candidates)
    raise FileNotFoundError(
        f"Could not find result directory '{result_dir_arg}' containing "
        f"cuda_h100_ratios.csv.\n"
        f"Available result directories: {available}\n"
        "Pass one or more of those directory names, omit all positional "
        "arguments to combine every available result directory, or pass "
        "--csv /path/to/cuda_h100_ratios.csv."
    )


def resolve_paths(args: argparse.Namespace) -> tuple[list[Path], Path]:
    cwd = Path.cwd()
    candidates = find_result_csvs(cwd)
    if args.csv_path:
        csv_path = Path(args.csv_path)
        if not csv_path.is_absolute():
            csv_path = cwd / csv_path
        if not csv_path.is_file():
            raise FileNotFoundError(f"Missing CSV: {csv_path}")
        csv_paths = [csv_path]
        default_out_dir = csv_path.parent / "figures"
    elif args.result_dirs:
        resolved = [
            resolve_result_dir(cwd, result_dir_arg, candidates)
            for result_dir_arg in args.result_dirs
        ]
        csv_paths = [csv_path for _, csv_path in resolved]
        default_out_dir = (
            resolved[0][0] / "figures" if len(resolved) == 1 else cwd / "figures"
        )
    else:
        if not candidates:
            raise FileNotFoundError(
                "No result directory containing cuda_h100_ratios.csv was found."
            )
        csv_paths = [csv_path for _, _, csv_path in candidates]
        default_out_dir = cwd / "figures"

    out_dir = Path(args.out_dir) if args.out_dir else default_out_dir
    if not out_dir.is_absolute():
        out_dir = cwd / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    return csv_paths, out_dir


def configure_matplotlib(font_size: float) -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": font_size,
            "axes.labelsize": font_size,
            "axes.titlesize": font_size * 1.05,
            "xtick.labelsize": font_size,
            "ytick.labelsize": font_size,
            "legend.fontsize": font_size,
            "figure.titlesize": font_size,
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.7,
            "lines.markersize": 5.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
        }
    )


def load_rows(csv_paths: list[Path]) -> list[dict[str, str]]:
    rows = []
    for csv_path in csv_paths:
        with csv_path.open(newline="") as handle:
            csv_rows = list(csv.DictReader(handle))
        columns = set(csv_rows[0].keys() if csv_rows else [])
        missing = REQUIRED_COLUMNS - columns
        if missing:
            raise ValueError(
                f"CSV is missing required columns: {csv_path}: {sorted(missing)}"
            )
        rows.extend(csv_rows)
    return rows


def parse_float(value: str | None) -> float:
    if value is None or value == "":
        return math.nan
    return float(value)


def available_benchmarks(rows: list[dict[str, str]]) -> list[str]:
    present = {row["benchmark"] for row in rows if CASE_RE.match(row.get("case", ""))}
    ordered = [benchmark for benchmark in BENCHMARK_ORDER if benchmark in present]
    extras = sorted(present - set(ordered))
    return ordered + extras


def collect_benchmark_series(
    rows: list[dict[str, str]],
    benchmark: str,
    combinations: list[int],
) -> tuple[list[int], dict[int, dict[int, float]], dict[int, dict[int, float]]]:
    memory_series: dict[int, dict[int, float]] = {combo: {} for combo in combinations}
    time_series: dict[int, dict[int, float]] = {combo: {} for combo in combinations}
    digits = set()

    for row in rows:
        if row.get("benchmark") != benchmark:
            continue
        match = CASE_RE.match(row.get("case", ""))
        if not match:
            continue
        combo = int(match.group("combination"))
        if combo not in memory_series:
            continue
        digit = int(match.group("digit"))
        memory_value = parse_float(row.get("memory_ratio_vs_double"))
        time_value = parse_float(row.get("time_ratio_vs_double"))
        if not (math.isfinite(memory_value) or math.isfinite(time_value)):
            continue
        memory_series[combo][digit] = memory_value
        time_series[combo][digit] = time_value
        digits.add(digit)

    return sorted(digits), memory_series, time_series


def finite_values(*series_maps: dict[int, dict[int, float]]) -> list[float]:
    values = []
    for series in series_maps:
        for by_digit in series.values():
            values.extend(value for value in by_digit.values() if math.isfinite(value))
    return values


def set_ratio_axis(ax, values: list[float]) -> None:
    if values:
        high = max(values + [1.0])
        ax.set_ylim(0.0, max(1.05, high * 1.12))
    else:
        ax.set_ylim(0.0, 1.05)
    ax.axhline(1.0, color="0.35", linewidth=0.85, linestyle=(0, (3, 2)), zorder=1)


def style_axis(ax) -> None:
    ax.grid(True, axis="y", color="0.86", linewidth=0.65, zorder=0)
    ax.grid(True, axis="x", color="0.93", linewidth=0.45, zorder=0)
    ax.tick_params(direction="out", length=3.2, width=0.75, color="0.25")
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)


def plot_benchmark_panel(
    ax,
    rows: list[dict[str, str]],
    benchmark: str,
    combinations: list[int],
    *,
    show_ylabel: bool,
) -> bool:
    digits, memory_series, time_series = collect_benchmark_series(
        rows, benchmark, combinations
    )
    if not digits:
        ax.set_visible(False)
        return False

    bar_width = min(0.34, 0.78 / max(len(combinations), 1))
    offsets = {
        combo: (index - (len(combinations) - 1) / 2.0) * bar_width
        for index, combo in enumerate(combinations)
    }

    for combo in combinations:
        color = COLORS.get(combo, "0.35")
        time_values = [time_series[combo].get(digit, math.nan) for digit in digits]
        memory_values = [memory_series[combo].get(digit, math.nan) for digit in digits]
        x_bars = [digit + offsets[combo] for digit in digits]

        ax.bar(
            x_bars,
            time_values,
            width=bar_width * 0.88,
            color=color,
            edgecolor=color,
            linewidth=0.9,
            hatch=HATCHES.get(combo, ""),
            alpha=0.42,
            zorder=2,
        )
        ax.plot(
            digits,
            memory_values,
            color=color,
            linestyle=LINESTYLES.get(combo, "-"),
            marker=MARKERS.get(combo, "o"),
            markerfacecolor="white",
            markeredgecolor=color,
            markeredgewidth=1.05,
            linewidth=1.75,
            zorder=3,
        )

    ax.set_title(BENCHMARK_LABELS.get(benchmark, benchmark.replace("_", " ").title()))
    ax.set_xlabel("Number of required digits")
    if show_ylabel:
        ax.set_ylabel("Ratio to FP64")
    ax.set_xticks(digits)
    ax.set_xlim(min(digits) - 0.45, max(digits) + 0.45)
    set_ratio_axis(ax, finite_values(memory_series, time_series))
    style_axis(ax)
    return True


def save_figure(fig, out_dir: Path, stem: str, formats: list[str], dpi: int) -> list[Path]:
    saved = []
    for fmt in formats:
        path = out_dir / f"{stem}.{fmt}"
        fig.savefig(path, dpi=dpi if fmt == "png" else None, bbox_inches="tight")
        saved.append(path)
    plt.close(fig)
    return saved


def add_shared_legend(fig, combinations: list[int], ncol: int) -> None:
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    metric_handles = [
        Line2D(
            [0],
            [0],
            color="0.22",
            marker="o",
            markerfacecolor="white",
            linewidth=1.8,
            label="Memory ratio",
        ),
        Patch(
            facecolor="0.55",
            edgecolor="0.35",
            alpha=0.42,
            label="Time ratio",
        ),
    ]
    combo_handles = [
        Line2D(
            [0],
            [0],
            color=COLORS.get(combo, "0.35"),
            marker=MARKERS.get(combo, "o"),
            markerfacecolor="white",
            linestyle=LINESTYLES.get(combo, "-"),
            linewidth=1.7,
            label=combination_label(combo),
        )
        for combo in combinations
    ]
    handles = metric_handles + combo_handles
    fig.legend(
        handles,
        [handle.get_label() for handle in handles],
        loc="lower center",
        bbox_to_anchor=(0.04, -0.035, 0.92, 0.1),
        ncol=ncol,
        mode="expand",
        frameon=False,
        handlelength=2.2,
        handletextpad=0.55,
        columnspacing=1.4,
        borderaxespad=0.0,
    )


def plot_benchmark_row(
    rows: list[dict[str, str]],
    benchmarks: list[str],
    combinations: list[int],
    out_dir: Path,
    formats: list[str],
    dpi: int,
) -> list[Path]:
    fig_width = max(6.6, 2.35 * len(benchmarks))
    fig, axes = plt.subplots(1, len(benchmarks), figsize=(fig_width, 2.55), squeeze=False)
    axes = list(axes[0])

    plotted = 0
    for index, (ax, benchmark) in enumerate(zip(axes, benchmarks)):
        plotted += int(
            plot_benchmark_panel(
                ax,
                rows,
                benchmark,
                combinations,
                show_ylabel=index == 0,
            )
        )
    if plotted == 0:
        plt.close(fig)
        return []

    add_shared_legend(fig, combinations, ncol=min(4, len(combinations) + 2))
    fig.subplots_adjust(left=0.065, right=0.995, bottom=0.31, top=0.88, wspace=0.28)

    benchmark_tag = "_".join(benchmark for benchmark in benchmarks)
    combo_tag = "_".join(str(combo) for combo in combinations)
    stem = f"h100_{benchmark_tag}_memory_lines_time_bars_combinations_{combo_tag}"
    return save_figure(fig, out_dir, stem, formats, dpi)


def plot_separate_benchmarks(
    rows: list[dict[str, str]],
    benchmarks: list[str],
    combinations: list[int],
    out_dir: Path,
    formats: list[str],
    dpi: int,
) -> list[Path]:
    saved_paths = []
    for benchmark in benchmarks:
        fig, ax = plt.subplots(1, 1, figsize=(3.15, 2.55))
        if not plot_benchmark_panel(
            ax,
            rows,
            benchmark,
            combinations,
            show_ylabel=True,
        ):
            plt.close(fig)
            continue
        add_shared_legend(fig, combinations, ncol=2)
        fig.subplots_adjust(left=0.18, right=0.99, bottom=0.32, top=0.86)
        combo_tag = "_".join(str(combo) for combo in combinations)
        stem = f"{benchmark}_memory_lines_time_bars_combinations_{combo_tag}"
        saved_paths.extend(save_figure(fig, out_dir, stem, formats, dpi))
    return saved_paths


def main() -> None:
    args = parse_args()
    try:
        csv_paths, out_dir = resolve_paths(args)
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc

    load_matplotlib()
    configure_matplotlib(args.font_size)
    rows = load_rows(csv_paths)

    benchmarks = args.benchmarks or available_benchmarks(rows)
    if not benchmarks:
        raise RuntimeError("No benchmark rows were found in the ratio CSV.")

    saved_paths = plot_benchmark_row(
        rows,
        benchmarks,
        args.combinations,
        out_dir,
        args.formats,
        args.dpi,
    )
    if args.separate:
        saved_paths.extend(
            plot_separate_benchmarks(
                rows,
                benchmarks,
                args.combinations,
                out_dir,
                args.formats,
                args.dpi,
            )
        )

    if not saved_paths:
        raise RuntimeError("No figures were generated.")

    print("Read:")
    for csv_path in csv_paths:
        print(f"  {csv_path}")
    print(f"Wrote {len(saved_paths)} figure files to: {out_dir}")
    for path in saved_paths:
        print(path)


if __name__ == "__main__":
    main()
