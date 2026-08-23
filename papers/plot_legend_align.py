#!/usr/bin/env python3
"""Create side-by-side precision/runtime plots for selected mp_tests benchmarks."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


CATEGORY_DISPLAY_NAMES = {
    "double": "FP64",
    "float": "FP32",
    "half_float::half": "FP16",
    "flx::floatx<5, 10>": "FP16",
    "flx::floatx<8, 7>": "BF16",
    "flx::floatx<4, 3>": "E4M3",
    "flx::floatx<5, 2>": "E5M2",
}

CATEGORY_COLORS = {
    "double": "#81D4FAB3",
    "float": "#FFAB91B3",
    "half_float::half": "#BA68C8B3",
    "flx::floatx<5, 10>": "#7262F0B3",
    "flx::floatx<8, 7>": "#F06292B3",
    "flx::floatx<4, 3>": "#AED581B3",
    "flx::floatx<5, 2>": "#FFF176B3",
}

CATEGORY_ORDER = [
    "flx::floatx<4, 3>",
    "flx::floatx<5, 2>",
    "flx::floatx<8, 7>",
    "flx::floatx<5, 10>",
    "half_float::half",
    "float",
    "double",
]

BENCHMARKS = {
    "dense_lu": "dense_lu",
    "hotspot": "hotspot",
    "backprop": "backprop",
    "srad_v2": "srad_v2",
    "particle_filter": "particle_filter",
    "sparse_lu": "sparse_lu",
}

DEFAULT_BENCHMARKS = [
    "dense_lu",
    "hotspot",
    "backprop",
    "srad_v2",
    "particle_filter",
    "sparse_lu",
]

DIGITS = list(range(1, 11))
Y_AXIS_HEADROOM = 1.02
COMBINED_PANEL_W_PAD = 0.5
# Keep fallback cropping shallow enough to preserve the top y-axis tick labels.
DEFAULT_IMAGE_CROP_TOP_FRACTION = 0.08


def _import_matplotlib():
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    return plt, Line2D, Patch


def use_original_style(plt):
    available_styles = plt.style.available
    preferred_style = (
        "seaborn"
        if "seaborn" in available_styles
        else "seaborn-v0_8"
        if "seaborn-v0_8" in available_styles
        else "ggplot"
    )
    try:
        plt.style.use(preferred_style)
    except OSError:
        plt.style.use("default")


def load_precision_settings(path: Path):
    with path.open() as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{path} must contain a list")
    for setting in data:
        if not isinstance(setting, dict):
            raise ValueError(f"{path} must contain a list of objects")
        for values in setting.values():
            if not isinstance(values, list):
                raise ValueError(f"{path} precision categories must contain lists")
    return data


def load_runtimes(path: Path):
    if not path.exists():
        return None

    runtimes = []
    with path.open(newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if header != ["Digit", "Runtime (seconds)"]:
            raise ValueError(f"{path} has an invalid CSV header")
        for row in reader:
            if not row or row[0] == "Average":
                continue
            if len(row) != 2:
                raise ValueError(f"{path} has an invalid row: {row}")
            runtimes.append(float(row[1]))
    return runtimes


def active_categories(*settings_groups):
    categories = set()
    for settings in settings_groups:
        for setting in settings:
            categories.update(setting.keys())

    def order_key(category):
        return CATEGORY_ORDER.index(category) if category in CATEGORY_ORDER else len(CATEGORY_ORDER)

    return sorted(categories, key=order_key)


def category_heights(settings, categories):
    heights = {category: [] for category in categories}
    for setting in settings:
        for category in categories:
            heights[category].append(len(setting.get(category, [])))
    return heights


def max_stack_height(settings, categories):
    heights = category_heights(settings, categories)
    totals = [
        sum(heights[category][index] for category in categories)
        for index in range(len(settings))
    ]
    return max(totals, default=1)


def add_precision_runtime_panel(
    ax,
    settings,
    runtimes,
    digits,
    categories,
    title=None,
    precision_ylim=None,
    runtime_ylim=None,
    show_precision_ylabel=True,
    show_runtime_ylabel=True,
):
    fontsize = 25
    ax2 = ax.twinx()
    x_indices = list(range(len(digits)))
    heights = category_heights(settings, categories)
    bottom = [0] * len(digits)

    for category in categories:
        color = CATEGORY_COLORS.get(category, "#808080")
        bars = ax.bar(
            x_indices,
            heights[category],
            bottom=bottom,
            color=color,
            width=0.6,
            edgecolor="white",
        )
        for bar, bar_height, bottom_height in zip(bars, heights[category], bottom):
            if bar_height > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bottom_height + bar_height / 2,
                    f"{int(bar_height)}",
                    ha="center",
                    va="center",
                    fontsize=22,
                    weight="bold",
                    color="black",
                )
        bottom = [old + new for old, new in zip(bottom, heights[category])]

    ax2.plot(
        x_indices,
        runtimes,
        color="red",
        marker="o",
        linestyle="-",
        linewidth=2,
        markersize=8,
        zorder=10,
    )

    ax.set_ylim(
        precision_ylim if precision_ylim is not None else (0, max(max(bottom), 1))
    )
    ax2.set_ylim(
        runtime_ylim
        if runtime_ylim is not None
        else (0, max(runtimes) * 1.5 if runtimes else 1.0)
    )
    ax.set_xticks(x_indices)
    ax.set_xticklabels(digits)
    ax.set_xlim(-0.5, len(digits) - 0.5)
    ax.set_xlabel("Number of required digits", fontsize=fontsize, weight="bold")
    if show_precision_ylabel:
        ax.set_ylabel("Number of variables of each type", fontsize=fontsize, weight="bold")
    if show_runtime_ylabel:
        ax2.set_ylabel("Runtime (seconds)", fontsize=fontsize, weight="bold", color="red")
    ax2.tick_params(axis="y", labelcolor="red")
    ax.tick_params(axis="both", which="major", labelsize=fontsize)
    ax2.tick_params(axis="both", which="major", labelsize=fontsize)
    ax.grid(True, axis="y", linestyle="--", alpha=0.7)
    if title:
        ax.set_title(title, fontsize=fontsize, weight="bold", pad=20)


def legend_handles(categories, Line2D, Patch):
    handles = [
        Patch(
            facecolor=CATEGORY_COLORS.get(category, "#808080"),
            edgecolor="white",
            label=CATEGORY_DISPLAY_NAMES.get(category, category),
        )
        for category in categories
    ]
    handles.append(
        Line2D(
            [0],
            [0],
            color="red",
            marker="o",
            linestyle="-",
            linewidth=2,
            markersize=8,
            label="Runtime",
        )
    )
    return handles


def generate_from_data(benchmark_dir: Path, output_path: Path, digits):
    plt, Line2D, Patch = _import_matplotlib()
    use_original_style(plt)

    settings_1 = load_precision_settings(benchmark_dir / "prec_setting_1.json")
    settings_2 = load_precision_settings(benchmark_dir / "prec_setting_2.json")
    runtimes_1 = load_runtimes(benchmark_dir / "runtimes1.csv")
    runtimes_2 = load_runtimes(benchmark_dir / "runtimes2.csv")
    if runtimes_1 is None or runtimes_2 is None:
        return False

    for label, values in {
        "prec_setting_1.json": settings_1,
        "prec_setting_2.json": settings_2,
        "runtimes1.csv": runtimes_1,
        "runtimes2.csv": runtimes_2,
    }.items():
        if len(values) != len(digits):
            raise ValueError(f"{benchmark_dir / label} has {len(values)} rows, expected {len(digits)}")

    categories = active_categories(settings_1, settings_2)
    precision_ylim = (
        0,
        max(
            max_stack_height(settings_1, categories),
            max_stack_height(settings_2, categories),
            1,
        )
        * Y_AXIS_HEADROOM,
    )
    runtime_ylim = (
        0,
        max(runtimes_1 + runtimes_2) * 1.5 if runtimes_1 or runtimes_2 else 1.0,
    )
    fig, axes = plt.subplots(1, 2, figsize=(22, 8.5))
    add_precision_runtime_panel(
        axes[0],
        settings_1,
        runtimes_1,
        digits,
        categories,
        title="Combination I",
        precision_ylim=precision_ylim,
        runtime_ylim=runtime_ylim,
        show_precision_ylabel=False,
        show_runtime_ylabel=False,
    )
    add_precision_runtime_panel(
        axes[1],
        settings_2,
        runtimes_2,
        digits,
        categories,
        title="Combination II",
        precision_ylim=precision_ylim,
        runtime_ylim=runtime_ylim,
        show_precision_ylabel=False,
        show_runtime_ylabel=False,
    )
    fig.text(
        0.05,
        0.5,
        "Number of variables of each type",
        va="center",
        rotation="vertical",
        fontsize=25,
        weight="bold",
    )
    fig.text(
        0.94,
        0.5,
        "Runtime (seconds)",
        va="center",
        rotation=-90,
        fontsize=25,
        weight="bold",
        color="red",
    )
    fig.legend(
        handles=legend_handles(categories, Line2D, Patch),
        loc="upper center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=min(len(categories) + 1, 6),
        fontsize=22,
        frameon=True,
        edgecolor="black",
    )
    fig.tight_layout(rect=[0.07, 0, 0.93, 0.9], w_pad=COMBINED_PANEL_W_PAD)
    fig.savefig(output_path, bbox_inches="tight", dpi=300, transparent=False)
    plt.close(fig)
    return True


def generate_from_existing_images(benchmark_dir: Path, output_path: Path, crop_top_fraction: float):
    plt, Line2D, Patch = _import_matplotlib()
    use_original_style(plt)

    image_paths = [
        benchmark_dir / "precision1_with_runtime.jpg",
        benchmark_dir / "precision2_with_runtime.jpg",
    ]
    missing = [str(path) for path in image_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing existing plots: {', '.join(missing)}")

    settings_1 = load_precision_settings(benchmark_dir / "prec_setting_1.json")
    settings_2 = load_precision_settings(benchmark_dir / "prec_setting_2.json")
    categories = active_categories(settings_1, settings_2)

    fig, axes = plt.subplots(1, 2, figsize=(22, 8.5))
    for ax, image_path, title in zip(
        axes, image_paths, ["Combination I", "Combination II"]
    ):
        image = plt.imread(image_path)
        crop_rows = int(image.shape[0] * crop_top_fraction)
        ax.imshow(image[crop_rows:])
        ax.set_title(title, fontsize=25, weight="bold", pad=20)
        ax.axis("off")

    fig.legend(
        handles=legend_handles(categories, Line2D, Patch),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.99),
        ncol=min(len(categories) + 1, 6),
        fontsize=22,
        frameon=True,
        edgecolor="black",
    )
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.88, wspace=0.02)
    fig.savefig(output_path, bbox_inches="tight", dpi=300, transparent=False)
    plt.close(fig)


def generate_benchmark_plot(
    base_dir: Path, benchmark: str, output_name: str | None, crop_top_fraction: float
):
    benchmark_key = BENCHMARKS.get(benchmark)
    if benchmark_key is None:
        raise ValueError(f"Unknown benchmark: {benchmark}")

    benchmark_dir = base_dir / benchmark_key
    output_filename = output_name or f"precision_{benchmark_key}_with_runtime.jpg"
    output_path = benchmark_dir / output_filename
    generated_from_data = generate_from_data(benchmark_dir, output_path, DIGITS)
    if not generated_from_data:
        generate_from_existing_images(benchmark_dir, output_path, crop_top_fraction)
    print(f"Saved {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Combine precision1_with_runtime.jpg and precision2_with_runtime.jpg into one side-by-side plot."
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Path to the mp_tests directory.",
    )
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        default=DEFAULT_BENCHMARKS,
        help="Benchmark names to process. Defaults to dense_lu, hotspot, backprop, srad, Particle Filter, sparse_lu.",
    )
    parser.add_argument(
        "--output-name",
        default=None,
        help=(
            "Combined image filename to write inside each benchmark directory. "
            "Defaults to precision_[benchmark_name]_with_runtime.jpg."
        ),
    )
    parser.add_argument(
        "--image-crop-top-fraction",
        type=float,
        default=DEFAULT_IMAGE_CROP_TOP_FRACTION,
        help="Top fraction to crop from existing images when runtime CSV files are missing.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    for benchmark in args.benchmarks:
        generate_benchmark_plot(
            args.base_dir,
            benchmark,
            args.output_name,
            args.image_crop_top_fraction,
        )


if __name__ == "__main__":
    main()
