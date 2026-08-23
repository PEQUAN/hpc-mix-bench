#!/usr/bin/env python3
"""Plot complementary H100 Tensor Core experiment results.

Run from papers/complement, for example:

    python plot_complement_h100.py results/1254344 --font-size 12

The script reads tensorcore_complement.csv, computes ratios against the FP64
blocked-update baseline, and writes Tensor Core ratio and accuracy figures.
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create Tensor Core complement plots from H100 CSV results."
    )
    parser.add_argument(
        "result_dir",
        nargs="?",
        default=None,
        help=(
            "Complement result directory, e.g. results/1254344. If omitted, "
            "the newest directory containing tensorcore_complement.csv is used."
        ),
    )
    parser.add_argument(
        "--csv",
        dest="csv_path",
        default=None,
        help="Explicit path to tensorcore_complement.csv.",
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
    roots = [cwd / "results", cwd]
    for root in roots:
        if not root.is_dir():
            continue
        for csv_path in root.glob("*/tensorcore_complement.csv"):
            candidates.append((csv_path.parent.stat().st_mtime, csv_path.parent, csv_path))
    if not candidates:
        raise FileNotFoundError(
            "No result directory containing tensorcore_complement.csv was found."
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
        csv_path = result_dir / "tensorcore_complement.csv"
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
    required = {
        "source_benchmark",
        "source_case",
        "status",
        "time_ms",
        "device_allocation_bytes",
        "relative_l2_error_vs_fp64",
        "relative_linf_error_vs_fp64",
    }
    missing = required - set(rows[0].keys() if rows else [])
    if missing:
        raise ValueError(f"CSV is missing required columns: {sorted(missing)}")
    return rows


def parse_float(row: dict[str, str], key: str) -> float:
    value = row.get(key, "")
    if value == "":
        return math.nan
    return float(value)


def completed_dense_lu_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return [
        row
        for row in rows
        if row.get("source_benchmark") == "dense_lu" and row.get("status") == "completed"
    ]


def find_baseline(rows: list[dict[str, str]]) -> dict[str, str]:
    for row in completed_dense_lu_rows(rows):
        if row.get("source_case") == "double":
            return row
    raise ValueError("Missing dense_lu/double FP64 Tensor Core complement baseline.")


def collect_digit_rows(
    rows: list[dict[str, str]],
    combinations: list[int],
) -> list[dict[str, str | int | float]]:
    baseline = find_baseline(rows)
    baseline_time = parse_float(baseline, "time_ms")
    baseline_bytes = parse_float(baseline, "device_allocation_bytes")
    out: list[dict[str, str | int | float]] = []

    for row in completed_dense_lu_rows(rows):
        match = CASE_RE.match(row.get("source_case", ""))
        if not match:
            continue
        combination = int(match.group("combination"))
        digit = int(match.group("digit"))
        if combination not in combinations:
            continue
        time_ms = parse_float(row, "time_ms")
        bytes_value = parse_float(row, "device_allocation_bytes")
        out.append(
            {
                "case": row["source_case"],
                "combination": combination,
                "digit": digit,
                "mode": row.get("mode", ""),
                "time_ratio_vs_fp64_update": time_ms / baseline_time
                if baseline_time > 0.0
                else math.nan,
                "memory_ratio_vs_fp64_update": bytes_value / baseline_bytes
                if baseline_bytes > 0.0
                else math.nan,
                "relative_l2_error_vs_fp64": parse_float(
                    row, "relative_l2_error_vs_fp64"
                ),
                "relative_linf_error_vs_fp64": parse_float(
                    row, "relative_linf_error_vs_fp64"
                ),
                "gflops": parse_float(row, "gflops"),
            }
        )
    return sorted(out, key=lambda row: (int(row["combination"]), int(row["digit"])))


def write_ratio_summary(rows: list[dict[str, str | int | float]], out_csv: Path) -> None:
    fields = [
        "case",
        "combination",
        "digit",
        "mode",
        "time_ratio_vs_fp64_update",
        "memory_ratio_vs_fp64_update",
        "relative_l2_error_vs_fp64",
        "relative_linf_error_vs_fp64",
        "gflops",
    ]
    with out_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def finite_values(values: list[float]) -> list[float]:
    return [value for value in values if math.isfinite(value)]


def positive_values(values: list[float]) -> list[float]:
    return [value for value in finite_values(values) if value > 0.0]


def set_linear_limits(ax, values: list[float]) -> None:
    finite = finite_values(values)
    if not finite:
        return
    low = min(finite + [1.0])
    high = max(finite + [1.0])
    pad = max((high - low) * 0.12, 0.02)
    ax.set_ylim(max(0.0, low - pad), high + pad)


def set_log_limits(ax, values: list[float]) -> None:
    positives = positive_values(values)
    if not positives:
        return
    low = min(positives)
    high = max(positives)
    ax.set_ylim(low / 2.0, high * 2.0 if high > low else high * 10.0)


def plot_metric_pair(
    rows: list[dict[str, str | int | float]],
    combinations: list[int],
    metrics: list[tuple[str, str]],
    stem: str,
    out_dir: Path,
    formats: list[str],
    dpi: int,
    *,
    log_scale: bool,
) -> list[Path]:
    if not rows:
        return []

    fig, axes = plt.subplots(1, 2, figsize=(7.8, 2.75), constrained_layout=True)
    handles_labels = []
    all_digits = sorted({int(row["digit"]) for row in rows})

    for ax, (metric, ylabel) in zip(axes, metrics):
        all_values = []
        if not log_scale:
            ax.axhline(1.0, color="0.35", linewidth=0.9, linestyle=(0, (3, 2)), zorder=1)
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
                label=COMBINATION_LABELS.get(combination, f"Combination {combination}"),
                zorder=3,
            )[0]
            if metric == metrics[0][0]:
                handles_labels.append(
                    (
                        line,
                        COMBINATION_LABELS.get(
                            combination, f"Combination {combination}"
                        ),
                    )
                )

        if log_scale:
            ax.set_yscale("log")
            set_log_limits(ax, all_values)
        else:
            set_linear_limits(ax, all_values)
        ax.set_xlabel("Number of required digits")
        ax.set_ylabel(ylabel)
        ax.set_xticks(all_digits)
        ax.grid(True, which="major", axis="y", color="0.86", linewidth=0.7)
        if log_scale:
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
        path = out_dir / f"{stem}_combinations_{combo_tag}.{fmt}"
        fig.savefig(path, dpi=dpi if fmt == "png" else None, bbox_inches="tight")
        saved.append(path)
    plt.close(fig)
    return saved


def main() -> None:
    args = parse_args()
    result_dir, csv_path, out_dir = resolve_paths(args)
    load_matplotlib()
    configure_matplotlib(args.font_size)
    rows = collect_digit_rows(load_rows(csv_path), args.combinations)
    if not rows:
        raise RuntimeError(f"No completed dense_lu digit rows were found in: {csv_path}")

    summary_csv = result_dir / "tensorcore_ratio_summary.csv"
    write_ratio_summary(rows, summary_csv)

    saved_paths = []
    saved_paths.extend(
        plot_metric_pair(
            rows,
            args.combinations,
            [
                ("memory_ratio_vs_fp64_update", "Memory ratio to FP64 update"),
                ("time_ratio_vs_fp64_update", "Time ratio to FP64 update"),
            ],
            "tensorcore_dense_lu_ratio",
            out_dir,
            args.formats,
            args.dpi,
            log_scale=False,
        )
    )
    saved_paths.extend(
        plot_metric_pair(
            rows,
            args.combinations,
            [
                ("relative_l2_error_vs_fp64", r"Relative $\ell_2$ error"),
                ("relative_linf_error_vs_fp64", r"Relative $\ell_{\infty}$ error"),
            ],
            "tensorcore_dense_lu_error",
            out_dir,
            args.formats,
            args.dpi,
            log_scale=True,
        )
    )

    print(f"Read: {csv_path}")
    print(f"Wrote: {summary_csv}")
    print(f"Wrote {len(saved_paths)} figure files to: {out_dir}")
    for path in saved_paths:
        print(path)


if __name__ == "__main__":
    main()
