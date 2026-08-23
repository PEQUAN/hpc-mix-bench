#!/usr/bin/env python3
"""Plot H100 accuracy errors recorded in ratio CSV files.

The script reads cuda_h100_ratios.csv or direct_case_ratios.csv and plots
benchmark-specific errors against the FP64 CUDA output:

- Backprop: output_delta_mse_vs_double and output_delta_l2_error_vs_double.
- Dense LU: solution_l2_error_vs_double and solution_linf_error_vs_double.
- Hotspot: output_l2_error_vs_double and output_linf_error_vs_double.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path

plt = None


CASE_RE = re.compile(r"^digit(?P<combination>\d+)_(?P<digit>\d+)$")

COLORS = {
    1: "#2B6CB0",
    2: "#B83280",
}

MARKERS = {
    1: "o",
    2: "s",
}

COMBINATION_LINESTYLES = {
    1: "-",
    2: (0, (4, 2)),
}

COMBINATION_LABELS = {
    1: "Combination I",
    2: "Combination II",
}


def combination_label(combo: int) -> str:
    return COMBINATION_LABELS.get(combo, f"Combination {combo}")

BENCHMARKS = {
    "backprop": {
        "label": "Backprop",
        "stem": "backprop_output_delta_error",
        "metrics": (
            ("output_delta_mse_vs_double", "Mean squared error"),
            ("output_delta_l2_error_vs_double", r"Relative $\ell_2$ error"),
        ),
    },
    "dense_lu": {
        "label": "Dense LU",
        "stem": "dense_lu_solution_error",
        "metrics": (
            ("solution_l2_error_vs_double", r"Relative $\ell_2$ error"),
            ("solution_linf_error_vs_double", r"Relative $\ell_{\infty}$ error"),
        ),
    },
    "hotspot": {
        "label": "Hotspot",
        "stem": "hotspot_output_error",
        "metrics": (
            ("output_l2_error_vs_double", r"Relative $\ell_2$ error"),
            ("output_linf_error_vs_double", r"Relative $\ell_{\infty}$ error"),
        ),
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create H100 accuracy-error plots from ratio CSV files."
    )
    parser.add_argument(
        "result_dir",
        nargs="?",
        default=None,
        help=(
            "Result directory under the current directory. If omitted, the "
            "newest directory containing cuda_h100_ratios.csv is used."
        ),
    )
    parser.add_argument(
        "--csv",
        dest="csv_path",
        default=None,
        help="Explicit path to cuda_h100_ratios.csv or direct_case_ratios.csv.",
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
        choices=sorted(BENCHMARKS),
        help="Benchmarks to plot. Default: all supported benchmarks in the CSV.",
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
            "`python -m pip install matplotlib` or run it in an environment "
            "where matplotlib is available."
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
    if not rows:
        raise ValueError(f"Empty CSV: {csv_path}")
    return rows


def parse_float(text: str) -> float:
    if text == "":
        return math.nan
    return float(text)


def collect_rows(
    rows: list[dict[str, str]],
    benchmark: str,
    combinations: list[int],
) -> list[dict[str, int | float]]:
    metrics = [metric for metric, _ in BENCHMARKS[benchmark]["metrics"]]
    out: list[dict[str, int | float]] = []
    for row in rows:
        if row.get("benchmark") != benchmark:
            continue
        match = CASE_RE.match(row.get("case", ""))
        if not match:
            continue
        combination = int(match.group("combination"))
        digit = int(match.group("digit"))
        if combination not in combinations:
            continue
        record: dict[str, int | float] = {
            "combination": combination,
            "digit": digit,
        }
        has_error = False
        for metric in metrics:
            value = parse_float(row.get(metric, ""))
            record[metric] = value
            has_error = has_error or (math.isfinite(value) and value > 0.0)
        if has_error:
            out.append(record)
    return sorted(out, key=lambda row: (int(row["combination"]), int(row["digit"])))


def positive_values(values: list[float]) -> list[float]:
    return [value for value in values if math.isfinite(value) and value > 0.0]


def set_log_limits(ax, values: list[float]) -> None:
    positives = positive_values(values)
    if not positives:
        return
    low = min(positives)
    high = max(positives)
    ax.set_ylim(low / 2.0, high * 2.0 if high > low else high * 10.0)


def plot_benchmark(
    rows: list[dict[str, int | float]],
    benchmark: str,
    combinations: list[int],
    out_dir: Path,
    formats: list[str],
    dpi: int,
) -> list[Path]:
    if not rows:
        return []

    config = BENCHMARKS[benchmark]
    metrics = list(config["metrics"])
    fig, axes = plt.subplots(1, 2, figsize=(7.8, 2.75), constrained_layout=True)
    handles_labels = []
    all_digits = sorted({int(row["digit"]) for row in rows})

    for ax, (metric, ylabel) in zip(axes, metrics):
        all_values = []
        for combination in combinations:
            combo_rows = [
                row for row in rows if int(row["combination"]) == combination
            ]
            if not combo_rows:
                continue
            digits = [int(row["digit"]) for row in combo_rows]
            values = [float(row[metric]) for row in combo_rows]
            all_values.extend(values)
            line = ax.plot(
                digits,
                values,
                color=COLORS.get(combination),
                linestyle=COMBINATION_LINESTYLES.get(combination, "-"),
                marker=MARKERS.get(combination, "o"),
                markerfacecolor="white",
                markeredgewidth=1.2,
                label=combination_label(combination),
                zorder=3,
            )[0]
            if metric == metrics[0][0]:
                handles_labels.append((line, combination_label(combination)))

        ax.set_yscale("log")
        ax.set_xlabel("Number of required digits")
        ax.set_ylabel(ylabel)
        ax.set_xticks(all_digits)
        set_log_limits(ax, all_values)
        ax.grid(True, which="major", axis="y", color="0.86", linewidth=0.7)
        ax.grid(True, which="minor", axis="y", color="0.92", linewidth=0.45)
        ax.grid(True, axis="x", color="0.93", linewidth=0.5)
        ax.tick_params(direction="out", length=3.5, width=0.8)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)

    if handles_labels:
        handles, labels = zip(*handles_labels)
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.12),
            ncol=len(labels),
            frameon=False,
            handlelength=2.3,
        )

    saved = []
    combo_tag = "_".join(str(combo) for combo in combinations)
    for fmt in formats:
        path = out_dir / f"{config['stem']}_combinations_{combo_tag}.{fmt}"
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

    benchmarks = args.benchmarks or [
        benchmark
        for benchmark in BENCHMARKS
        if any(row.get("benchmark") == benchmark for row in rows)
    ]
    saved_paths = []
    for benchmark in benchmarks:
        saved_paths.extend(
            plot_benchmark(
                collect_rows(rows, benchmark, args.combinations),
                benchmark,
                args.combinations,
                out_dir,
                args.formats,
                args.dpi,
            )
        )

    if not saved_paths:
        raise RuntimeError("No accuracy-error figures were generated.")

    print(f"Read: {csv_path}")
    print(f"Wrote {len(saved_paths)} figure files to: {out_dir}")
    for path in saved_paths:
        print(path)


if __name__ == "__main__":
    main()
