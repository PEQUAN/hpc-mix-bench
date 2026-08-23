#!/usr/bin/env python3
"""Plot Dense LU solution errors from saved H100 solution vectors.

Run from papers/h100_results, for example:

    python plot_dense_lu_solution_errors.py 1254344 --font-size 12

The script reads <result_dir>/dense_lu_solutions/double_solution.txt and the
corresponding digit<i>_<j>_solution.txt files, writes a compact CSV summary, and
generates a publication-style L2/Linf error figure.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path

plt = None


CASE_RE = re.compile(r"^digit(?P<combination>\d+)_(?P<digit>\d+)_solution\.txt$")

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot Dense LU solution-vector errors against the FP64 run."
    )
    parser.add_argument(
        "result_dir",
        nargs="?",
        default=None,
        help=(
            "Result directory under the current directory. If omitted, the "
            "newest directory containing dense_lu_solutions is used."
        ),
    )
    parser.add_argument(
        "--solutions-dir",
        default=None,
        help="Explicit dense_lu_solutions directory.",
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


def find_default_solutions_dir(cwd: Path) -> tuple[Path, Path]:
    candidates = []
    for child in cwd.iterdir():
        solutions_dir = child / "dense_lu_solutions"
        ref_path = solutions_dir / "double_solution.txt"
        if child.is_dir() and ref_path.is_file():
            candidates.append((child.stat().st_mtime, child, solutions_dir))
    if not candidates:
        raise FileNotFoundError(
            "No result directory containing dense_lu_solutions/double_solution.txt "
            "was found."
        )
    _, result_dir, solutions_dir = sorted(candidates)[-1]
    return result_dir, solutions_dir


def resolve_paths(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    cwd = Path.cwd()
    if args.solutions_dir:
        solutions_dir = Path(args.solutions_dir)
        if not solutions_dir.is_absolute():
            solutions_dir = cwd / solutions_dir
        result_dir = solutions_dir.parent
    elif args.result_dir:
        result_dir = Path(args.result_dir)
        if not result_dir.is_absolute():
            result_dir = cwd / result_dir
        solutions_dir = result_dir / "dense_lu_solutions"
    else:
        result_dir, solutions_dir = find_default_solutions_dir(cwd)

    ref_path = solutions_dir / "double_solution.txt"
    if not ref_path.is_file():
        raise FileNotFoundError(f"Missing FP64 reference solution: {ref_path}")

    out_dir = Path(args.out_dir) if args.out_dir else result_dir / "figures"
    if not out_dir.is_absolute():
        out_dir = cwd / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    return result_dir, solutions_dir, out_dir


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


def read_vector(path: Path) -> list[float]:
    values = []
    with path.open() as handle:
        for line in handle:
            text = line.strip()
            if text:
                values.append(float(text))
    if not values:
        raise ValueError(f"Empty solution vector: {path}")
    return values


def compare_vectors(ref: list[float], candidate: list[float]) -> tuple[float, float]:
    if len(ref) != len(candidate):
        raise ValueError(
            f"Solution length mismatch: reference has {len(ref)} values, "
            f"candidate has {len(candidate)} values."
        )
    ref_l2 = 0.0
    err_l2 = 0.0
    ref_linf = 0.0
    err_linf = 0.0
    for ref_value, cand_value in zip(ref, candidate):
        diff = cand_value - ref_value
        ref_l2 += ref_value * ref_value
        err_l2 += diff * diff
        ref_linf = max(ref_linf, abs(ref_value))
        err_linf = max(err_linf, abs(diff))
    rel_l2 = math.sqrt(err_l2) / math.sqrt(ref_l2) if ref_l2 > 0.0 else math.sqrt(err_l2)
    rel_linf = err_linf / ref_linf if ref_linf > 0.0 else err_linf
    return rel_l2, rel_linf


def collect_errors(
    solutions_dir: Path,
    combinations: list[int],
) -> list[dict[str, str | int | float]]:
    ref = read_vector(solutions_dir / "double_solution.txt")
    rows: list[dict[str, str | int | float]] = []
    for path in sorted(solutions_dir.glob("digit*_*_solution.txt")):
        match = CASE_RE.match(path.name)
        if not match:
            continue
        combination = int(match.group("combination"))
        digit = int(match.group("digit"))
        if combination not in combinations:
            continue
        rel_l2, rel_linf = compare_vectors(ref, read_vector(path))
        rows.append(
            {
                "case": f"digit{combination}_{digit}",
                "combination": combination,
                "digit": digit,
                "relative_l2_error_vs_double": rel_l2,
                "relative_linf_error_vs_double": rel_linf,
            }
        )
    return sorted(rows, key=lambda row: (int(row["combination"]), int(row["digit"])))


def write_csv(rows: list[dict[str, str | int | float]], out_csv: Path) -> None:
    fields = [
        "case",
        "combination",
        "digit",
        "relative_l2_error_vs_double",
        "relative_linf_error_vs_double",
    ]
    with out_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def positive_values(values: list[float]) -> list[float]:
    return [value for value in values if math.isfinite(value) and value > 0.0]


def set_log_limits(ax, values: list[float]) -> None:
    positives = positive_values(values)
    if not positives:
        return
    low = min(positives)
    high = max(positives)
    ax.set_ylim(low / 2.0, high * 2.0 if high > low else high * 10.0)


def plot_errors(
    rows: list[dict[str, str | int | float]],
    combinations: list[int],
    out_dir: Path,
    formats: list[str],
    dpi: int,
) -> list[Path]:
    if not rows:
        return []

    metrics = [
        ("relative_l2_error_vs_double", r"Relative $\ell_2$ error"),
        ("relative_linf_error_vs_double", r"Relative $\ell_{\infty}$ error"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(7.8, 2.75), constrained_layout=True)
    handles_labels = []

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
            if metric == "relative_l2_error_vs_double":
                handles_labels.append((line, combination_label(combination)))

        ax.set_yscale("log")
        ax.set_xlabel("Number of required digits")
        ax.set_ylabel(ylabel)
        ax.set_xticks(sorted({int(row["digit"]) for row in rows}))
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
    stem = f"dense_lu_solution_error_combinations_{combo_tag}"
    for fmt in formats:
        path = out_dir / f"{stem}.{fmt}"
        fig.savefig(path, dpi=dpi if fmt == "png" else None, bbox_inches="tight")
        saved.append(path)
    plt.close(fig)
    return saved


def main() -> None:
    args = parse_args()
    result_dir, solutions_dir, out_dir = resolve_paths(args)
    load_matplotlib()
    configure_matplotlib(args.font_size)
    rows = collect_errors(solutions_dir, args.combinations)
    if not rows:
        raise RuntimeError(f"No digit solution files were found in: {solutions_dir}")

    summary_csv = result_dir / "dense_lu_solution_errors.csv"
    write_csv(rows, summary_csv)
    saved_paths = plot_errors(rows, args.combinations, out_dir, args.formats, args.dpi)

    print(f"Read: {solutions_dir}")
    print(f"Wrote: {summary_csv}")
    print(f"Wrote {len(saved_paths)} figure files to: {out_dir}")
    for path in saved_paths:
        print(path)


if __name__ == "__main__":
    main()
