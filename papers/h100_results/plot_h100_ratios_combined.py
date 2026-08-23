#!/usr/bin/env python3
"""Plot H100 mixed-precision ratios and optional accuracy errors.

Run from mp_tests/h100_results, for example:

    python plot_h100_ratios_combined.py 747957 --font-size 12

The script reads cuda_h100_ratios.csv and writes one combined ratio figure per
benchmark with memory ratio on the left and time ratio on the right.  When the
CSV also contains benchmark-specific accuracy columns, it additionally writes a
three-panel figure with memory ratio, time ratio, and error trends separated.
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
    "memory_ratio_vs_double": {
        "ylabel": "Memory ratio to FP64",
        "filename": "memory_ratio",
        "note": "Lower is more memory efficient",
        "stddev": None,
        "position": 0,  # left subplot
    },
    "time_ratio_vs_double": {
        "ylabel": "Time ratio to FP64",
        "filename": "time_ratio",
        "note": "Lower is faster",
        "stddev": "time_ratio_vs_double_stddev",
        "position": 1,  # right subplot
    },
}

ACCURACY_METRICS = {
    "backprop": (
        ("output_delta_mse_vs_double", "Mean squared error"),
        ("output_delta_l2_error_vs_double", r"Relative $\ell_2$ error"),
    ),
    "dense_lu": (
        ("solution_l2_error_vs_double", r"Relative $\ell_2$ error"),
        ("solution_linf_error_vs_double", r"Relative $\ell_{\infty}$ error"),
    ),
    "hotspot": (
        ("output_l2_error_vs_double", r"Relative $\ell_2$ error"),
        ("output_linf_error_vs_double", r"Relative $\ell_{\infty}$ error"),
    ),
}

COLORS = {
    1: "#2B6CB0",
    2: "#B83280",
}

METRIC_COLORS = {
    "memory": "#005CA8",
    "time": "#C44E52",
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

COMBINATION_LINESTYLES = {
    1: "-",
    2: (0, (4, 2)),
}

COMBINATION_HATCHES = {
    1: "",
    2: "///",
}

COMBINATION_LABELS = {
    1: "Combination I",
    2: "Combination II",
}


def combination_label(combo: int) -> str:
    return COMBINATION_LABELS.get(combo, f"Combination {combo}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create publication-quality H100 combined ratio plots from "
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
    parser.add_argument(
        "--skip-ratio-error",
        action="store_true",
        help="Only write the original side-by-side ratio figures.",
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


def parse_float(text) -> float:
    if text is None or text == "":
        return math.nan
    return float(text)


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


def positive_values_by_combo(series: dict[int, dict[int, float]]) -> list[float]:
    return [
        value
        for by_digit in series.values()
        for value in by_digit.values()
        if math.isfinite(value) and value > 0.0
    ]


def set_y_limits(ax: plt.Axes, values: list[float]) -> None:
    if not values:
        return
    low = min(values + [1.0])
    high = max(values + [1.0])
    pad = max((high - low) * 0.12, 0.02)
    ax.set_ylim(max(0.0, low - pad), high + pad)


def set_ratio_axis(ax) -> None:
    ax.set_ylim(0.0, 1.02)
    ax.set_yticks([index / 10.0 for index in range(0, 11)])


def set_log_y_limits(ax, values: list[float]) -> None:
    positives = [value for value in values if math.isfinite(value) and value > 0.0]
    if not positives:
        return
    low = min(positives)
    high = max(positives)
    if high <= low:
        ax.set_ylim(low / 2.0, high * 10.0)
    else:
        ax.set_ylim(low / 1.8, high * 1.8)


def collect_metric_series(
    rows: list[dict[str, str]],
    benchmark: str,
    metric: str,
    combinations: list[int],
) -> tuple[list[int], dict[int, dict[int, float]]]:
    series: dict[int, dict[int, float]] = {combo: {} for combo in combinations}
    digits = set()
    for row in rows:
        if row.get("benchmark") != benchmark:
            continue
        match = CASE_RE.match(row.get("case", ""))
        if not match:
            continue
        combo = int(match.group("combination"))
        digit = int(match.group("digit"))
        if combo not in series:
            continue
        value = parse_float(row.get(metric))
        if not math.isfinite(value):
            continue
        series[combo][digit] = value
        digits.add(digit)
    return sorted(digits), series


def has_positive_data(series: dict[int, dict[int, float]]) -> bool:
    return any(
        math.isfinite(value) and value > 0.0
        for by_digit in series.values()
        for value in by_digit.values()
    )


def style_axis(ax, color=None) -> None:
    ax.grid(True, axis="y", color="0.88", linewidth=0.65)
    ax.grid(True, axis="x", color="0.94", linewidth=0.45)
    ax.tick_params(direction="out", length=3.5, width=0.8, color="0.25")
    if color:
        ax.tick_params(axis="y", colors=color)
        ax.yaxis.label.set_color(color)
        ax.spines["left"].set_color(color)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)


def style_twin_axis(ax, color: str) -> None:
    ax.tick_params(axis="y", direction="out", length=3.5, width=0.8, colors=color)
    ax.yaxis.label.set_color(color)
    ax.spines["right"].set_color(color)
    ax.spines["left"].set_visible(False)
    ax.spines["top"].set_visible(False)


def plot_ratio_bars(
    ax,
    digits: list[int],
    series: dict[int, dict[int, float]],
    combinations: list[int],
    *,
    ylabel: str,
    color: str,
) -> None:
    bar_width = min(0.34, 0.78 / max(len(combinations), 1))
    combo_offsets = {
        combo: (index - (len(combinations) - 1) / 2.0) * bar_width
        for index, combo in enumerate(combinations)
    }
    for combo in combinations:
        values = [series[combo].get(digit, math.nan) for digit in digits]
        linestyle = COMBINATION_LINESTYLES.get(combo, "-")
        hatch = COMBINATION_HATCHES.get(combo, "")
        combo_color = COLORS.get(combo, color)
        x_values = [digit + combo_offsets[combo] for digit in digits]
        ax.bar(
            x_values,
            values,
            width=bar_width * 0.88,
            color=combo_color,
            edgecolor=combo_color,
            linewidth=1.0,
            linestyle=linestyle,
            hatch=hatch,
            alpha=0.72,
            zorder=2,
        )

    ax.set_xlabel("Number of required digits")
    ax.set_ylabel(ylabel)
    ax.yaxis.labelpad = 8
    ax.set_xticks(digits)
    ax.set_xlim(min(digits) - 0.45, max(digits) + 0.45)
    set_ratio_axis(ax)
    ax.axhline(1.0, color="0.36", linewidth=0.9, linestyle=(0, (3, 2)), zorder=1)
    style_axis(ax, color)


def plot_error_lines(
    ax,
    digits: list[int],
    first_series: dict[int, dict[int, float]],
    second_series: dict[int, dict[int, float]],
    combinations: list[int],
    metric_labels: tuple[str, str],
) -> None:
    line_dodge = min(0.06, 0.14 / max(len(combinations), 1))
    combo_offsets = {
        combo: (index - (len(combinations) - 1) / 2.0) * line_dodge
        for index, combo in enumerate(combinations)
    }
    metric_specs = [
        (first_series, metric_labels[0], "o", "-"),
        (second_series, metric_labels[1], "s", (0, (2, 1))),
    ]
    for metric_series, _, marker, metric_linestyle in metric_specs:
        for combo in combinations:
            x_values = [digit + combo_offsets[combo] for digit in digits]
            values = [
                metric_series[combo].get(digit, math.nan)
                for digit in digits
            ]
            values = [
                value if math.isfinite(value) and value > 0.0 else math.nan
                for value in values
            ]
            ax.plot(
                x_values,
                values,
                color=COLORS.get(combo, "0.25"),
                linestyle=metric_linestyle,
                marker=marker,
                markerfacecolor="white",
                markeredgecolor=COLORS.get(combo, "0.25"),
                markeredgewidth=1.15,
                markersize=5.8,
                linewidth=1.65,
                alpha=0.9,
                zorder=3,
            )

    ax.set_yscale("log")
    set_log_y_limits(
        ax,
        positive_values_by_combo(first_series) + positive_values_by_combo(second_series),
    )
    ax.set_xlabel("Number of required digits")
    ax.set_ylabel("Error")
    ax.yaxis.labelpad = 8
    ax.set_xticks(digits)
    ax.set_xlim(min(digits) - 0.25, max(digits) + 0.25)
    style_axis(ax)


def plot_ratio_error_benchmark(
    rows: list[dict[str, str]],
    benchmark: str,
    combinations: list[int],
    out_dir: Path,
    formats: list[str],
    dpi: int,
) -> list[Path]:
    accuracy = ACCURACY_METRICS.get(benchmark)
    if not accuracy:
        return []
    required_columns = {
        "memory_ratio_vs_double",
        "time_ratio_vs_double",
        accuracy[0][0],
        accuracy[1][0],
    }
    if not rows or not required_columns.issubset(rows[0].keys()):
        return []

    perf_digits, memory_series = collect_metric_series(
        rows, benchmark, "memory_ratio_vs_double", combinations
    )
    time_digits, time_series = collect_metric_series(
        rows, benchmark, "time_ratio_vs_double", combinations
    )
    err_left_digits, err_left_series = collect_metric_series(
        rows, benchmark, accuracy[0][0], combinations
    )
    err_right_digits, err_right_series = collect_metric_series(
        rows, benchmark, accuracy[1][0], combinations
    )

    panel_digits = sorted(
        set(perf_digits) & set(time_digits) & set(err_left_digits) & set(err_right_digits)
    )
    if not panel_digits:
        return []
    if not (has_positive_data(err_left_series) or has_positive_data(err_right_series)):
        return []

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(10.8, 3.2),
        constrained_layout=False,
        gridspec_kw={"width_ratios": [1.0, 1.0, 1.2]},
    )

    plot_ratio_bars(
        axes[0],
        panel_digits,
        memory_series,
        combinations,
        ylabel="Memory ratio to FP64",
        color=METRIC_COLORS["memory"],
    )
    plot_ratio_bars(
        axes[1],
        panel_digits,
        time_series,
        combinations,
        ylabel="Time ratio to FP64",
        color=METRIC_COLORS["time"],
    )
    plot_error_lines(
        axes[2],
        panel_digits,
        err_left_series,
        err_right_series,
        combinations,
        (accuracy[0][1], accuracy[1][1]),
    )

    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    combo_handles = [
        Patch(
            facecolor=COLORS.get(combo, "0.78"),
            edgecolor=COLORS.get(combo, "0.22"),
            linestyle=COMBINATION_LINESTYLES.get(combo, "-"),
            linewidth=1.1,
            hatch=COMBINATION_HATCHES.get(combo, ""),
            label=combination_label(combo),
        )
        for combo in combinations
    ]
    metric_handles = [
        Line2D(
            [0],
            [0],
            color="0.22",
            linestyle="-",
            marker="o",
            markerfacecolor="white",
            markeredgecolor="0.22",
            markeredgewidth=1.15,
            linewidth=1.65,
            markersize=5.8,
            label=accuracy[0][1],
        ),
        Line2D(
            [0],
            [0],
            color="0.22",
            linestyle=(0, (2, 1)),
            marker="s",
            markerfacecolor="white",
            markeredgecolor="0.22",
            markeredgewidth=1.15,
            linewidth=1.65,
            markersize=5.8,
            label=accuracy[1][1],
        ),
    ]
    legend = fig.legend(
        combo_handles + metric_handles,
        [handle.get_label() for handle in combo_handles + metric_handles],
        loc="lower center",
        bbox_to_anchor=(0.055, -0.025, 0.92, 0.1),
        ncol=4,
        mode="expand",
        frameon=False,
        columnspacing=2.4,
        handlelength=2.7,
        handletextpad=0.7,
        borderaxespad=0.0,
    )
    legend.get_frame().set_linewidth(0.0)

    fig.subplots_adjust(left=0.06, right=0.99, bottom=0.29, top=0.94, wspace=0.36)

    saved = []
    combo_tag = "_".join(str(combo) for combo in combinations)
    stem = f"{benchmark}_ratio_error_combinations_{combo_tag}_three_panel"
    for fmt in formats:
        path = out_dir / f"{stem}.{fmt}"
        fig.savefig(path, dpi=dpi if fmt == "png" else None, bbox_inches="tight")
        saved.append(path)
    plt.close(fig)
    return saved


def plot_combined_benchmark(
    rows: list[dict[str, str]],
    benchmark: str,
    combinations: list[int],
    out_dir: Path,
    formats: list[str],
    dpi: int,
) -> list[Path]:
    # Prepare data for both metrics
    all_data = {}
    for metric in METRICS:
        digits, series, errors = collect_series(rows, benchmark, metric, combinations)
        if not digits:
            return []
        all_data[metric] = (digits, series, errors)

    # Create figure with 2 subplots side-by-side
    fig, axes = plt.subplots(1, 2, figsize=(7.8, 2.75), constrained_layout=True)
    
    # Plot each metric
    handles_labels = []
    for metric, ax in zip(METRICS.keys(), axes):
        digits, series, errors = all_data[metric]
        
        ax.axhline(1.0, color="0.35", linewidth=0.9, linestyle=(0, (3, 2)), zorder=1)
        
        for combo in combinations:
            y_values = [series[combo].get(digit, math.nan) for digit in digits]
            y_errors = [errors[combo].get(digit, 0.0) for digit in digits]
            common_style = {
                "color": COLORS.get(combo, None),
                "linestyle": COMBINATION_LINESTYLES.get(combo, "-"),
                "marker": MARKERS.get(combo, "o"),
                "markerfacecolor": "white",
                "markeredgewidth": 1.2,
                "label": combination_label(combo),
                "zorder": 3,
            }
            if any(err > 0.0 for err in y_errors):
                line = ax.errorbar(
                    digits,
                    y_values,
                    yerr=y_errors,
                    capsize=2.8,
                    elinewidth=0.9,
                    capthick=0.9,
                    **common_style,
                )
            else:
                line = ax.plot(digits, y_values, **common_style)[0]
            
            # Collect handles and labels from first subplot only
            if metric == "memory_ratio_vs_double":
                handles_labels.append((line, combination_label(combo)))
        
        ax.set_xlabel("Number of required digits")
        ax.set_ylabel(METRICS[metric]["ylabel"])
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
    
    # Add shared legend at the top center
    if handles_labels:
        handles, labels = zip(*handles_labels)
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.12),
            ncol=len(combinations),
            frameon=False,
            handlelength=2.3,
        )
    
    # Add overall title
    #fig.suptitle(
    #    BENCHMARK_LABELS.get(benchmark, benchmark.replace("_", " ")),
    #    y=1.18,
    #    fontsize=plt.rcParams["font.size"] * 1.1,
    #)

    saved = []
    combo_tag = "_".join(str(combo) for combo in combinations)
    stem = f"{benchmark}_ratio_combinations_{combo_tag}_combined"
    for fmt in formats:
        path = out_dir / f"{stem}.{fmt}"
        fig.savefig(path, dpi=dpi if fmt == "png" else None, bbox_inches="tight")
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
        saved_paths.extend(
            plot_combined_benchmark(
                rows,
                benchmark,
                args.combinations,
                out_dir,
                args.formats,
                args.dpi,
            )
        )
        if not args.skip_ratio_error:
            saved_paths.extend(
                plot_ratio_error_benchmark(
                    rows,
                    benchmark,
                    args.combinations,
                    out_dir,
                    args.formats,
                    args.dpi,
                )
            )

    if not saved_paths:
        raise RuntimeError("No figures were generated.")

    print(f"Read: {csv_path}")
    print(f"Wrote {len(saved_paths)} combined figure files to: {out_dir}")
    for path in saved_paths:
        print(path)


if __name__ == "__main__":
    main()
