#!/usr/bin/env python
"""Generate 1-bit granularity PROMISE precision-analysis figures.

This script sweeps one custom PROMISE precision format against double
precision.  Each PROMISE run uses only two precision choices: double precision
(`d`, exponent 11 and explicit significand 52) and one custom precision (`w`).
The custom exponent/significand bit counts are rewritten in a temporary fp.json
for each point of the sweep.  This matches the custom format layout documented
in cadnaPromise/README.rst.

How to run:
    cd 1-bit-exps
    python3 onebit_precision_analysis.py --repo-root ../mp_tests [options]

Run PROMISE data collection and then plot all default benchmarks:
    python3 onebit_precision_analysis.py --repo-root ../mp_tests --run

Plot from existing CSV files without rerunning PROMISE:
    python3 onebit_precision_analysis.py --repo-root ../mp_tests

Run only two benchmarks, for example hotspot and dense_lu:
    python3 onebit_precision_analysis.py --repo-root ../mp_tests --run --benchmark hotspot --benchmark dense_lu

Run from the repository root instead of 1-bit-exps:
    python3 1-bit-exps/onebit_precision_analysis.py --repo-root mp_tests --run

Options:
    --repo-root PATH
        Benchmark root path containing the benchmark folders (for example
        mp_tests).  The default is the current directory, so pass
        --repo-root explicitly unless the benchmark folders are alongside
        this script.

    --benchmark NAME
        Benchmark folder name under --repo-root.  This option is repeatable.
        If omitted, the default benchmark list is backprop and dense_lu.
        Any existing benchmark folder under --repo-root can be selected, such
        as hotspot or dense_lu.

    --run
        Run PROMISE sweeps before plotting.  If omitted, the script only reads
        existing onebit_precision_results.csv files and regenerates figures.
        Default: disabled.

    --digits LIST
        Required significant digits to sweep.  Accepts comma-separated values
        and ranges, for example 1-10 or 2,4,6.  Default: 1-10.

    --nb-digits N
        Run one required significant digit value.  This is ignored when
        --digits is provided.  Default: unset.

    --cadna-path PATH
        Optional CADNA_PATH override for PROMISE.  If omitted, the script first
        uses the existing CADNA_PATH environment variable, then bundled CADNA
        from the imported cadnaPromise package when available.

    --timeout SECONDS
        Timeout for each PROMISE run.  Default: 300 seconds.

Outputs:
    For each selected benchmark, data are read from or written to
    <benchmark>/onebit_precision_results.csv.  Figures and
    onebit_precision_summary.txt are written to <repo-root>/figures/.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import math
import os
import signal
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterator, List, Optional, Tuple

BENCHMARKS = {
    "backprop": Path("backprop"),
    "dense_lu": Path("dense_lu"),
}

COLORS = {
    "custom": "#1f77b4",
    "runtime": "#d62728",
}

DOUBLE_EXPONENT_BITS = 11
DOUBLE_SIGNIFICAND_BITS = 52
SIGNIFICAND_SWEEP_BITS = DOUBLE_SIGNIFICAND_BITS
EXPONENT_SWEEP_BITS = DOUBLE_EXPONENT_BITS
CUSTOM_SYMBOL = "w"
DOUBLE_SYMBOL = "d"
DOUBLE_TYPE = "double"
RUNTIME_YAXIS_PADDING = 1.2
DEFAULT_SIGNIFICANT_DIGITS = tuple(range(1, 11))
SWEEP_MAX_XTICKS = 8
HEATMAP_MAX_XTICKS = 9
XTICK_LABEL_ROTATION = 35
FONT_SIZE = 14


@dataclass
class SweepConfig:
    benchmark: str
    benchmark_dir: Path
    csv_path: Path
    repo_root: Path


def benchmark_display_name(benchmark: str) -> str:
    if benchmark == "dense_lu":
        return "Dense LU"
    if benchmark == "backprop":
        return "Backprop"
    return benchmark.replace("_", " ").title()


def resolve_benchmarks(repo_root: Path, selected: Optional[List[str]]) -> List[Tuple[str, Path]]:
    if not selected:
        return [(name, repo_root / rel_path) for name, rel_path in BENCHMARKS.items()]

    resolved: List[Tuple[str, Path]] = []
    for name in selected:
        bench_dir = repo_root / name
        if not bench_dir.is_dir():
            raise RuntimeError(f"benchmark '{name}' not found under {repo_root}")
        resolved.append((name, bench_dir))
    return resolved


def parse_digits(raw_digits: Optional[str], nb_digits: Optional[int]) -> List[int]:
    if raw_digits:
        digits: List[int] = []
        for chunk in raw_digits.split(","):
            chunk = chunk.strip()
            if not chunk:
                continue
            if "-" in chunk:
                start, end = (int(part.strip()) for part in chunk.split("-", 1))
                if start > end:
                    raise ValueError(f"invalid descending digit range: {chunk}")
                digits.extend(range(start, end + 1))
            else:
                digits.append(int(chunk))
    elif nb_digits is not None:
        digits = [nb_digits]
    else:
        digits = list(DEFAULT_SIGNIFICANT_DIGITS)

    unique_digits = sorted(set(digits))
    if not unique_digits or any(digit <= 0 for digit in unique_digits):
        raise ValueError("significant digits must be positive integers")
    return unique_digits


def configure_style() -> None:
    plt, _ = get_plotting_dependencies()
    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "axes.labelsize": FONT_SIZE,
            "axes.titlesize": FONT_SIZE,
            "xtick.labelsize": FONT_SIZE,
            "ytick.labelsize": FONT_SIZE,
            "legend.fontsize": 11,
            "lines.linewidth": 2.2,
            "lines.markersize": 6,
            "axes.grid": True,
            "grid.color": "#d9d9d9",
            "grid.alpha": 0.5,
            "grid.linewidth": 0.8,
        }
    )


def get_plotting_dependencies() -> Tuple[object, object]:
    import matplotlib.pyplot as pyplot
    import numpy as numpy

    return pyplot, numpy


def custom_type_name(e: int, t: int) -> str:
    return f"flx::floatx<{e}, {t}>"


def write_temp_fp_json(path: Path, e: int, t: int) -> None:
    payload = {
        CUSTOM_SYMBOL: [e, t],
        DOUBLE_SYMBOL: [DOUBLE_EXPONENT_BITS, DOUBLE_SIGNIFICAND_BITS],
    }
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def clean_promise_result(result: object) -> Dict[str, List[str]]:
    if not isinstance(result, dict):
        return {}
    cleaned: Dict[str, List[str]] = {}
    for key, value in result.items():
        if isinstance(value, (set, list, tuple)):
            cleaned[str(key)] = sorted(str(item) for item in value)
        else:
            cleaned[str(key)] = [str(value)]
    return cleaned


@contextmanager
def promise_timeout(seconds: int) -> Iterator[None]:
    if seconds <= 0 or not hasattr(signal, "SIGALRM"):
        yield
        return

    def _raise_timeout(_signum: int, _frame: object) -> None:
        # This interrupts PROMISE execution; benchmark temp files are still
        # removed by the caller's finally block.
        raise TimeoutError(f"PROMISE run exceeded {seconds} seconds")

    start_time = time.monotonic()
    previous_handler = signal.signal(signal.SIGALRM, _raise_timeout)
    previous_timer = signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)
        previous_remaining = previous_timer[0] - (time.monotonic() - start_time)
        if previous_remaining > 0:
            signal.setitimer(signal.ITIMER_REAL, previous_remaining, previous_timer[1])


def local_cadna_promise_search_paths(repo_root: Path) -> List[Path]:
    return [repo_root.parent / "cadnaPromise", repo_root / "cadnaPromise"]


def import_run_promise(repo_root: Path) -> Callable[[Optional[List[str]]], object]:
    try:
        module = importlib.import_module("cadnaPromise.run")
    except ImportError as installed_error:
        last_error: ImportError = installed_error
    else:
        configure_cadna_path_from_module(module)
        return getattr(module, "runPromise")

    for local_package in local_cadna_promise_search_paths(repo_root):
        local_package_str = str(local_package)
        inserted = False
        if local_package.is_dir() and local_package_str not in sys.path:
            sys.path.insert(0, local_package_str)
            inserted = True
        try:
            module = importlib.import_module("cadnaPromise.run")
        except ImportError as exc:
            last_error = exc
            if inserted:
                sys.path.remove(local_package_str)
            continue

        configure_cadna_path_from_module(module)
        return getattr(module, "runPromise")

    raise ImportError("cadnaPromise.run could not be imported") from last_error


def configure_cadna_path_from_module(module: object) -> None:
    if os.environ.get("CADNA_PATH"):
        return

    module_file = getattr(module, "__file__", None)
    if not module_file:
        return

    bundled_cadna = Path(module_file).resolve().parent / "cadna"
    if next((bundled_cadna / "lib").glob("libcadnaC.*"), None) is not None:
        os.environ["CADNA_PATH"] = str(bundled_cadna)


def run_promise_once(
    benchmark_dir: Path,
    repo_root: Path,
    fp_json_path: Path,
    nb_digits: int,
    cadna_path: Optional[str],
    timeout: int,
) -> Tuple[Dict[str, List[str]], float, str, str]:
    env_cadna = os.environ.get("CADNA_PATH")
    old_cwd = Path.cwd()
    start_time = time.time()
    try:
        if cadna_path:
            os.environ["CADNA_PATH"] = cadna_path
        run_promise = import_run_promise(repo_root)
        os.chdir(benchmark_dir)
        testargs = [
            f"--precs={CUSTOM_SYMBOL}{DOUBLE_SYMBOL}",
            f"--nbDigits={nb_digits}",
            "--conf=promise.yml",
            f"--fp={fp_json_path}",
        ]
        with promise_timeout(timeout):
            result = run_promise(testargs)
        elapsed = time.time() - start_time
        cleaned = clean_promise_result(result)
        return cleaned, elapsed, "ok" if cleaned else "no_result", ""
    except TimeoutError as exc:
        return {}, time.time() - start_time, "timeout", str(exc)
    except Exception as exc:
        return {}, time.time() - start_time, "error", str(exc)
    finally:
        os.chdir(old_cwd)
        if cadna_path:
            if env_cadna is None:
                os.environ.pop("CADNA_PATH", None)
            else:
                os.environ["CADNA_PATH"] = env_cadna


def count_precision_assignments(setting: Dict[str, List[str]], custom_type: str) -> Tuple[int, int]:
    return len(setting.get(custom_type, [])), len(setting.get(DOUBLE_TYPE, []))


def append_run_row(
    rows: List[Dict[str, object]],
    mode: str,
    nb_digits: int,
    e: int,
    t: int,
    setting: Dict[str, List[str]],
    runtime: float,
    status: str,
    error: str,
) -> None:
    custom_type = custom_type_name(e, t)
    custom_count, double_count = count_precision_assignments(setting, custom_type)
    rows.append(
        {
            "mode": mode,
            "nb_digits": nb_digits,
            "e": e,
            "t": t,
            "custom_type": custom_type,
            "custom_count": custom_count,
            "double_count": double_count,
            "runtime": f"{runtime:.6f}",
            "status": status,
            "setting_json": json.dumps(setting, sort_keys=True),
            "error": error,
        }
    )


def collect_one_point(
    cfg: SweepConfig,
    tmp_fp: Path,
    e: int,
    t: int,
    nb_digits: int,
    cadna_path: Optional[str],
    timeout: int,
) -> Tuple[Dict[str, List[str]], float, str, str]:
    write_temp_fp_json(tmp_fp, e, t)
    return run_promise_once(cfg.benchmark_dir, cfg.repo_root, tmp_fp, nb_digits, cadna_path, timeout)


def maybe_run_data_collection(
    cfg: SweepConfig,
    run: bool,
    digits: List[int],
    cadna_path: Optional[str],
    timeout: int,
) -> None:
    if not run:
        return

    rows: List[Dict[str, object]] = []
    tmp_fp = cfg.benchmark_dir / "fp_1bit_tmp.json"

    try:
        for nb_digits in digits:
            for t in range(1, SIGNIFICAND_SWEEP_BITS + 1):
                setting, runtime, status, error = collect_one_point(
                    cfg, tmp_fp, DOUBLE_EXPONENT_BITS, t, nb_digits, cadna_path, timeout
                )
                append_run_row(rows, "sig", nb_digits, DOUBLE_EXPONENT_BITS, t, setting, runtime, status, error)

            for e in range(1, EXPONENT_SWEEP_BITS + 1):
                setting, runtime, status, error = collect_one_point(
                    cfg, tmp_fp, e, DOUBLE_SIGNIFICAND_BITS, nb_digits, cadna_path, timeout
                )
                append_run_row(rows, "exp", nb_digits, e, DOUBLE_SIGNIFICAND_BITS, setting, runtime, status, error)
    finally:
        if tmp_fp.exists():
            tmp_fp.unlink()

    cfg.csv_path.parent.mkdir(parents=True, exist_ok=True)
    with cfg.csv_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "mode",
            "nb_digits",
            "e",
            "t",
            "custom_type",
            "custom_count",
            "double_count",
            "runtime",
            "status",
            "setting_json",
            "error",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_rows(csv_path: Path) -> List[Dict[str, str]]:
    if not csv_path.exists():
        return []
    with csv_path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def available_digits(rows: List[Dict[str, str]], fallback: List[int]) -> List[int]:
    digits = set()
    saw_legacy_rows = False
    for row in rows:
        if "nb_digits" not in row:
            saw_legacy_rows = True
        try:
            digits.add(int(row.get("nb_digits", "")))
        except Exception:
            continue
    if not digits and saw_legacy_rows:
        return [6]
    return sorted(digits) if digits else fallback


def rows_to_maps(
    rows: List[Dict[str, str]],
) -> Tuple[
    Dict[Tuple[int, int, int], float],
    Dict[Tuple[int, int, int], float],
    Dict[Tuple[int, int, int], float],
]:
    custom_counts: Dict[Tuple[int, int, int], float] = {}
    double_counts: Dict[Tuple[int, int, int], float] = {}
    runtimes: Dict[Tuple[int, int, int], float] = {}
    for row in rows:
        try:
            nb_digits = int(row.get("nb_digits", "6"))
            e = int(row["e"])
            t = int(row["t"])
        except Exception:
            continue

        key = (nb_digits, e, t)
        try:
            custom_counts[key] = float(row["custom_count"])
        except Exception:
            custom_counts[key] = math.nan

        try:
            double_counts[key] = float(row["double_count"])
        except Exception:
            double_counts[key] = math.nan

        try:
            runtimes[key] = float(row.get("runtime", "nan"))
        except Exception:
            runtimes[key] = math.nan
    return custom_counts, double_counts, runtimes


def build_sweep_arrays(
    count_map: Dict[Tuple[int, int, int], float],
    runtime_map: Dict[Tuple[int, int, int], float],
    nb_digits: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    _, np = get_plotting_dependencies()
    tvals = np.arange(1, SIGNIFICAND_SWEEP_BITS + 1)
    evals = np.arange(1, EXPONENT_SWEEP_BITS + 1)
    sig_counts = np.array(
        [count_map.get((nb_digits, DOUBLE_EXPONENT_BITS, int(t)), math.nan) for t in tvals], dtype=float
    )
    exp_counts = np.array(
        [count_map.get((nb_digits, int(e), DOUBLE_SIGNIFICAND_BITS), math.nan) for e in evals], dtype=float
    )
    sig_runtime = np.array(
        [runtime_map.get((nb_digits, DOUBLE_EXPONENT_BITS, int(t)), math.nan) for t in tvals], dtype=float
    )
    exp_runtime = np.array(
        [runtime_map.get((nb_digits, int(e), DOUBLE_SIGNIFICAND_BITS), math.nan) for e in evals], dtype=float
    )
    return tvals, sig_counts, sig_runtime, evals, exp_counts, exp_runtime


def build_digit_matrices(
    custom_map: Dict[Tuple[int, int, int], float],
    double_map: Dict[Tuple[int, int, int], float],
    digits: List[int],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    _, np = get_plotting_dependencies()
    sig_custom = np.full((len(digits), SIGNIFICAND_SWEEP_BITS), np.nan, dtype=float)
    sig_double = np.full((len(digits), SIGNIFICAND_SWEEP_BITS), np.nan, dtype=float)
    exp_custom = np.full((len(digits), EXPONENT_SWEEP_BITS), np.nan, dtype=float)
    exp_double = np.full((len(digits), EXPONENT_SWEEP_BITS), np.nan, dtype=float)

    for row_idx, nb_digits in enumerate(digits):
        for t in range(1, SIGNIFICAND_SWEEP_BITS + 1):
            key = (nb_digits, DOUBLE_EXPONENT_BITS, t)
            sig_custom[row_idx, t - 1] = custom_map.get(key, math.nan)
            sig_double[row_idx, t - 1] = double_map.get(key, math.nan)
        for e in range(1, EXPONENT_SWEEP_BITS + 1):
            key = (nb_digits, e, DOUBLE_SIGNIFICAND_BITS)
            exp_custom[row_idx, e - 1] = custom_map.get(key, math.nan)
            exp_double[row_idx, e - 1] = double_map.get(key, math.nan)

    return sig_custom, sig_double, exp_custom, exp_double


def plot_sweep_panel(ax, xs: np.ndarray, counts: np.ndarray, runtimes: np.ndarray, title: str, xlabel: str) -> None:
    _, np = get_plotting_dependencies()
    count_line = ax.plot(xs, counts, marker="o", color=COLORS["custom"], label="Custom variables")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Variables assigned to custom precision")
    ax.set_title(title)
    ax.set_ylim(bottom=0)
    ax.margins(x=0.03)

    ax2 = ax.twinx()
    runtime_line = ax2.plot(xs, runtimes, marker="s", linestyle="--", color=COLORS["runtime"], label="Runtime")
    ax2.set_ylabel("Runtime (seconds)", color=COLORS["runtime"])
    ax2.tick_params(axis="y", labelcolor=COLORS["runtime"])
    finite_runtime = runtimes[np.isfinite(runtimes)]
    if finite_runtime.size:
        ax2.set_ylim(0, max(float(np.max(finite_runtime)) * RUNTIME_YAXIS_PADDING, 0.1))

    lines = count_line + runtime_line
    ax.legend(lines, [line.get_label() for line in lines], loc="best")


def adaptive_sweep_ticks(max_bits: int, max_ticks: int = 13) -> np.ndarray:
    _, np = get_plotting_dependencies()
    if max_bits <= max_ticks:
        return np.arange(1, max_bits + 1)

    raw_step = max(1, math.ceil(max_bits / max(max_ticks - 2, 1)))
    nice_steps = (1, 2, 5, 10, 20, 25, 50)
    step = next((value for value in nice_steps if value >= raw_step), raw_step)
    ticks = np.concatenate(([1], np.arange(step, max_bits + 1, step), [max_bits]))
    ticks = np.unique(ticks)
    if ticks.size > 1 and max_bits - ticks[-2] < step:
        ticks = np.delete(ticks, -2)
    return np.unique(ticks)


def apply_bit_xticks(ax, max_bits: int, max_ticks: int, offset: int = 0) -> None:
    ticks = adaptive_sweep_ticks(max_bits, max_ticks=max_ticks)
    ax.set_xticks(ticks - offset)
    ax.set_xticklabels([str(int(tick)) for tick in ticks], rotation=XTICK_LABEL_ROTATION, ha="right")
    ax.tick_params(axis="x", labelsize=FONT_SIZE)


def apply_bit_yticks(ax, max_bits: int, max_ticks: int, offset: int = 0) -> None:
    ticks = adaptive_sweep_ticks(max_bits, max_ticks=max_ticks)
    ax.set_yticks(ticks - offset)
    ax.set_yticklabels([str(int(tick)) for tick in ticks])
    ax.tick_params(axis="y", labelsize=FONT_SIZE)


def plot_sweep(
    benchmark: str,
    nb_digits: int,
    tvals: np.ndarray,
    sig_counts: np.ndarray,
    sig_runtime: np.ndarray,
    evals: np.ndarray,
    exp_counts: np.ndarray,
    exp_runtime: np.ndarray,
    outdir: Path,
) -> None:
    plt, np = get_plotting_dependencies()
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), constrained_layout=True)

    plot_sweep_panel(
        axes[0],
        tvals,
        sig_counts,
        sig_runtime,
        f"Custom significand sweep, e = {DOUBLE_EXPONENT_BITS}",
        "Custom trailing significand bits",
    )
    apply_bit_xticks(axes[0], SIGNIFICAND_SWEEP_BITS, max_ticks=SWEEP_MAX_XTICKS)

    plot_sweep_panel(
        axes[1],
        evals,
        exp_counts,
        exp_runtime,
        f"Custom exponent sweep, t = {DOUBLE_SIGNIFICAND_BITS}",
        "Custom exponent bits",
    )
    apply_bit_xticks(axes[1], EXPONENT_SWEEP_BITS, max_ticks=SWEEP_MAX_XTICKS)

    fig.suptitle(f"{benchmark_display_name(benchmark)}: custom precision vs double ({nb_digits} digits)")

    pdf = outdir / f"{benchmark}_1bit_sweep.pdf"
    png = outdir / f"{benchmark}_1bit_sweep.png"
    fig.savefig(pdf)
    fig.savefig(png, dpi=600)
    plt.close(fig)


def plot_digit_count_heatmap(
    ax,
    data: np.ndarray,
    digits: List[int],
    y_label: str,
    title: str,
) -> object:
    plt, np = get_plotting_dependencies()
    cmap = plt.cm.magma.copy()
    cmap.set_bad(color="#bdbdbd")
    img = ax.imshow(data.T, origin="lower", cmap=cmap, aspect="auto")
    ax.set_xlabel("Required significant digits")
    ax.set_ylabel(y_label)
    ax.set_xticks(np.arange(len(digits)))
    ax.set_xticklabels(digits)
    ax.set_title(title)
    return img


def plot_digit_counts(
    benchmark: str,
    digits: List[int],
    sig_custom: np.ndarray,
    sig_double: np.ndarray,
    exp_custom: np.ndarray,
    exp_double: np.ndarray,
    outdir: Path,
) -> None:
    plt, np = get_plotting_dependencies()
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)

    img = plot_digit_count_heatmap(
        axes[0, 0],
        sig_custom,
        digits,
        f"Custom significand bits (e = {DOUBLE_EXPONENT_BITS})",
        "Custom precision variables",
    )
    apply_bit_yticks(axes[0, 0], SIGNIFICAND_SWEEP_BITS, max_ticks=HEATMAP_MAX_XTICKS, offset=1)
    fig.colorbar(img, ax=axes[0, 0])

    img = plot_digit_count_heatmap(
        axes[0, 1],
        sig_double,
        digits,
        f"Custom significand bits (e = {DOUBLE_EXPONENT_BITS})",
        "Double precision variables",
    )
    apply_bit_yticks(axes[0, 1], SIGNIFICAND_SWEEP_BITS, max_ticks=HEATMAP_MAX_XTICKS, offset=1)
    fig.colorbar(img, ax=axes[0, 1])

    img = plot_digit_count_heatmap(
        axes[1, 0],
        exp_custom,
        digits,
        f"Custom exponent bits (t = {DOUBLE_SIGNIFICAND_BITS})",
        "Custom precision variables",
    )
    apply_bit_yticks(axes[1, 0], EXPONENT_SWEEP_BITS, max_ticks=HEATMAP_MAX_XTICKS, offset=1)
    fig.colorbar(img, ax=axes[1, 0])

    img = plot_digit_count_heatmap(
        axes[1, 1],
        exp_double,
        digits,
        f"Custom exponent bits (t = {DOUBLE_SIGNIFICAND_BITS})",
        "Double precision variables",
    )
    apply_bit_yticks(axes[1, 1], EXPONENT_SWEEP_BITS, max_ticks=HEATMAP_MAX_XTICKS, offset=1)
    fig.colorbar(img, ax=axes[1, 1])

    fig.suptitle(f"{benchmark_display_name(benchmark)}: precision counts across significant digits")
    pdf = outdir / f"{benchmark}_digit_precision_counts.pdf"
    png = outdir / f"{benchmark}_digit_precision_counts.png"
    fig.savefig(pdf)
    fig.savefig(png, dpi=600)
    plt.close(fig)


def first_saturation(xs: np.ndarray, ys: np.ndarray, tol: float = 0.2) -> Optional[int]:
    _, np = get_plotting_dependencies()
    finite_idx = np.where(np.isfinite(ys))[0]
    if finite_idx.size < 3:
        return None
    y = ys[finite_idx]
    x = xs[finite_idx]
    for i in range(2, len(y)):
        if abs(y[i] - y[i - 1]) <= tol and abs(y[i - 1] - y[i - 2]) <= tol:
            return int(x[i])
    return None


def summarize(benchmark: str, nb_digits: int, sig_counts: np.ndarray, exp_counts: np.ndarray) -> str:
    _, np = get_plotting_dependencies()
    sig_valid = sig_counts[np.isfinite(sig_counts)]
    exp_valid = exp_counts[np.isfinite(exp_counts)]

    if sig_valid.size == 0 and exp_valid.size == 0:
        return f"{benchmark}: no valid PROMISE custom/double sweep data were available in CSV for {nb_digits} digits."

    sig_sat = first_saturation(np.arange(1, SIGNIFICAND_SWEEP_BITS + 1), sig_counts)
    exp_sat = first_saturation(np.arange(1, EXPONENT_SWEEP_BITS + 1), exp_counts)
    sig_max = int(np.max(sig_valid)) if sig_valid.size else 0
    exp_max = int(np.max(exp_valid)) if exp_valid.size else 0

    return (
        f"{benchmark} ({nb_digits} digits): max variables assigned to custom precision were {sig_max} in the significand sweep "
        f"and {exp_max} in the exponent sweep; saturation at t={sig_sat if sig_sat is not None else 'n/a'}, "
        f"e={exp_sat if exp_sat is not None else 'n/a'}."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="1-bit custom precision analysis for PROMISE benchmarks")
    parser.add_argument("--repo-root", default=".", help="benchmark root path (default: current directory)")
    parser.add_argument(
        "--benchmark",
        action="append",
        help="Benchmark folder name under mp_tests (repeatable); defaults to backprop and dense_lu",
    )
    parser.add_argument("--run", action="store_true", help="Run PROMISE sweeps before plotting")
    parser.add_argument(
        "--digits",
        default=None,
        help="Required significant digits to sweep, e.g. '1-10' or '2,4,6' (default: 1-10)",
    )
    parser.add_argument(
        "--nb-digits",
        type=int,
        default=None,
        help="Run one required significant digit value; ignored when --digits is set",
    )
    parser.add_argument(
        "--cadna-path",
        default=os.environ.get("CADNA_PATH"),
        help="Optional CADNA_PATH override for PROMISE",
    )
    parser.add_argument("--timeout", type=int, default=300, help="Timeout per PROMISE run (seconds)")
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    requested_digits = parse_digits(args.digits, args.nb_digits)
    benchmarks = resolve_benchmarks(repo_root, args.benchmark)
    outdir = repo_root / "figures"
    outdir.mkdir(parents=True, exist_ok=True)

    configure_style()
    summaries: List[str] = []

    for bench, bench_dir in benchmarks:
        csv_path = bench_dir / "onebit_precision_results.csv"
        cfg = SweepConfig(bench, bench_dir, csv_path, repo_root)

        maybe_run_data_collection(
            cfg,
            run=args.run,
            digits=requested_digits,
            cadna_path=args.cadna_path,
            timeout=args.timeout,
        )
        rows = load_rows(csv_path)
        digits = available_digits(rows, requested_digits)
        if not digits:
            summaries.append(f"{bench}: no significant digit values were available.")
            continue
        summary_digit = max(digits)
        custom_map, double_map, runtime_map = rows_to_maps(rows)

        tvals, sig_counts, sig_runtime, evals, exp_counts, exp_runtime = build_sweep_arrays(
            custom_map, runtime_map, summary_digit
        )
        sig_custom, sig_double, exp_custom, exp_double = build_digit_matrices(custom_map, double_map, digits)

        plot_sweep(bench, summary_digit, tvals, sig_counts, sig_runtime, evals, exp_counts, exp_runtime, outdir)
        plot_digit_counts(bench, digits, sig_custom, sig_double, exp_custom, exp_double, outdir)

        summaries.append(summarize(bench, summary_digit, sig_counts, exp_counts))

    summary_path = outdir / "onebit_precision_summary.txt"
    summary_path.write_text("\n".join(summaries) + "\n", encoding="utf-8")
    print("\n".join(summaries))


if __name__ == "__main__":
    main()
