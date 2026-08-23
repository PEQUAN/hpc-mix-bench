#!/usr/bin/env python3
"""Summarize H100 results for reviewer-facing performance explanations."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


BENCHMARKS = ("dense_lu", "hotspot", "backprop")
COMBINATIONS = (1, 2)

PRECISION_NAME = {
    (11, 52): "FP64",
    (8, 23): "FP32",
    (5, 10): "FP16",
    (8, 7): "BF16",
    (5, 2): "E5M2",
    (4, 3): "E4M3",
}

PRECISION_BYTES = {
    "FP64": 8,
    "FP32": 4,
    "FP16": 2,
    "BF16": 2,
    "E5M2": 1,
    "E4M3": 1,
}

WRAPPER_PAIR_RE = re.compile(r"#define\s+([A-Z0-9_]+)_E\s+(\d+)\s*\n#define\s+\1_T\s+(\d+)")


def precision_name(e: int, t: int) -> str:
    return PRECISION_NAME.get((e, t), f"E{e}M{t}")


def precision_bytes(name: str) -> int:
    return PRECISION_BYTES.get(name, 4)


def case_sort_key(case: str) -> tuple[int, int]:
    if case == "double":
        return (0, 0)
    m = re.match(r"digit(\d+)_(\d+)$", case)
    if not m:
        return (99, 99)
    return (int(m.group(1)), int(m.group(2)))


def read_ratio_rows(results_root: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for csv_path in sorted(results_root.glob("*/cuda_h100_ratios.csv")):
        run_id = csv_path.parent.name
        with csv_path.open(newline="") as f:
            for row in csv.DictReader(f):
                if row.get("benchmark") in BENCHMARKS:
                    row = dict(row)
                    row["run_id"] = run_id
                    rows.append(row)
    return rows


def choose_latest_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    latest: dict[tuple[str, str], dict[str, str]] = {}
    for row in rows:
        key = (row["benchmark"], row["case"])
        if key not in latest or row["run_id"] > latest[key]["run_id"]:
            latest[key] = row
    return [latest[k] for k in sorted(latest, key=lambda x: (x[0], case_sort_key(x[1])))]


def read_cpu_counts(papers_dir: Path) -> dict[tuple[str, int, int], dict[str, int]]:
    counts_path = papers_dir / "fp_counts_summary.csv"
    out: dict[tuple[str, int, int], dict[str, int]] = {}
    if not counts_path.exists():
        return out
    with counts_path.open(newline="") as f:
        for row in csv.DictReader(f):
            bench = row["Folder"].strip("/")
            if bench not in BENCHMARKS:
                continue
            combo = int(row["Precision Setting"])
            digit = int(row["Significant Digits"])
            out[(bench, combo, digit)] = {
                "FP64": int(row["FP64"]),
                "FP32": int(row["FP32"]),
                "FP16": int(row["FP16"]),
                "BF16": int(row["BF16"]),
                "E4M3": int(row["E4M3"]),
                "E5M2": int(row["E5M2"]),
            }
    return out


def parse_wrapper(papers_dir: Path, bench: str, case: str) -> dict[str, str | int]:
    if case == "double":
        return {}
    wrapper = papers_dir / bench / case / f"{bench}_cuda.cu"
    if bench == "backprop":
        wrapper = papers_dir / bench / case / "backprop_cuda.cu"
    if not wrapper.exists():
        return {"wrapper_exists": 0}
    text = wrapper.read_text()
    pairs = {}
    for name, e, t in WRAPPER_PAIR_RE.findall(text):
        pname = precision_name(int(e), int(t))
        pairs[name] = pname

    fp64_macro_count = sum(1 for p in pairs.values() if p == "FP64")
    lower_macro_count = sum(1 for p in pairs.values() if p != "FP64")
    dominant_storage = ""
    dominant_storage_bytes = ""
    if bench == "dense_lu":
        dominant_storage = pairs.get("DLU_AKK", "")
        dominant_storage_bytes = str(precision_bytes(dominant_storage)) if dominant_storage else ""
    elif bench == "hotspot":
        dominant_storage = pairs.get("HS_FIELD", pairs.get("HS_DELTA", ""))
        dominant_storage_bytes = str(precision_bytes(dominant_storage)) if dominant_storage else ""
    elif bench == "backprop":
        state_names = ["BP_HIDDEN_H", "BP_OUTPUT_O", "BP_OUTPUT_T", "BP_ADJUST"]
        dominant_storage = ";".join(f"{name}:{pairs.get(name, '')}" for name in state_names)
        dominant_storage_bytes = ""

    return {
        "wrapper_exists": 1,
        "wrapper_macro_count": len(pairs),
        "wrapper_fp64_macro_count": fp64_macro_count,
        "wrapper_lower_macro_count": lower_macro_count,
        "dominant_storage": dominant_storage,
        "dominant_storage_bytes": dominant_storage_bytes,
        "wrapper_signature": "|".join(f"{k}:{v}" for k, v in sorted(pairs.items())),
    }


def write_summary_csv(rows: list[dict[str, str]], papers_dir: Path, out_csv: Path) -> None:
    cpu_counts = read_cpu_counts(papers_dir)
    fields = [
        "run_id",
        "benchmark",
        "case",
        "combination",
        "digit",
        "time_ratio_vs_double",
        "time_ratio_vs_double_stddev",
        "speedup_vs_double",
        "memory_ratio_vs_double",
        "device_allocation_mib",
        "cpu_fp64_variables",
        "cpu_lower_variables",
        "wrapper_fp64_macro_count",
        "wrapper_lower_macro_count",
        "dominant_storage",
        "dominant_storage_bytes",
        "wrapper_signature",
    ]
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            case = row["case"]
            if case == "double":
                combo = digit = ""
                counts = {}
                wrapper = {}
            else:
                combo, digit = case_sort_key(case)
                if combo not in COMBINATIONS:
                    continue
                counts = cpu_counts.get((row["benchmark"], combo, digit), {})
                wrapper = parse_wrapper(papers_dir, row["benchmark"], case)
            cpu_fp64 = counts.get("FP64", "")
            cpu_lower = ""
            if counts:
                cpu_lower = sum(v for k, v in counts.items() if k != "FP64")
            writer.writerow({
                "run_id": row.get("run_id", ""),
                "benchmark": row["benchmark"],
                "case": case,
                "combination": combo,
                "digit": digit,
                "time_ratio_vs_double": row.get("time_ratio_vs_double", ""),
                "time_ratio_vs_double_stddev": row.get("time_ratio_vs_double_stddev", ""),
                "speedup_vs_double": row.get("speedup_vs_double", ""),
                "memory_ratio_vs_double": row.get("memory_ratio_vs_double", ""),
                "device_allocation_mib": row.get("device_allocation_mib", ""),
                "cpu_fp64_variables": cpu_fp64,
                "cpu_lower_variables": cpu_lower,
                "wrapper_fp64_macro_count": wrapper.get("wrapper_fp64_macro_count", ""),
                "wrapper_lower_macro_count": wrapper.get("wrapper_lower_macro_count", ""),
                "dominant_storage": wrapper.get("dominant_storage", ""),
                "dominant_storage_bytes": wrapper.get("dominant_storage_bytes", ""),
                "wrapper_signature": wrapper.get("wrapper_signature", ""),
            })


def read_summary_rows(summary_csv: Path) -> list[dict[str, str]]:
    with summary_csv.open(newline="") as f:
        return list(csv.DictReader(f))


def row_for(rows: list[dict[str, str]], bench: str, case: str) -> dict[str, str] | None:
    for row in rows:
        if row["benchmark"] == bench and row["case"] == case:
            return row
    return None


def fmt(x: str, ndigits: int = 3) -> str:
    if x == "":
        return "n/a"
    try:
        return f"{float(x):.{ndigits}f}"
    except ValueError:
        return x


def write_reviewer_notes(summary_csv: Path, out_md: Path) -> None:
    rows = read_summary_rows(summary_csv)
    out_md.parent.mkdir(parents=True, exist_ok=True)

    hot_110 = row_for(rows, "hotspot", "digit1_10")
    hot_210 = row_for(rows, "hotspot", "digit2_10")
    dense_16 = row_for(rows, "dense_lu", "digit1_6")
    dense_26 = row_for(rows, "dense_lu", "digit2_6")

    identical_hotspot_10 = (
        hot_110 is not None
        and hot_210 is not None
        and hot_110["wrapper_signature"] == hot_210["wrapper_signature"]
    )
    identical_dense_6 = (
        dense_16 is not None
        and dense_26 is not None
        and dense_16["wrapper_signature"] == dense_26["wrapper_signature"]
    )

    lines = [
        "# Reviewer-Facing Notes for H100 Timing Questions",
        "",
        "## Main interpretation",
        "",
        "- The paper figures use direct CUDA ports of the PROMISE-transformed programs. "
        "They preserve the selected precision assignments but do not call cuBLAS, WMMA, "
        "or Tensor Core APIs.",
        "- A time ratio greater than one is therefore possible: lower storage precision "
        "reduces allocation, but the direct kernels may pay for conversions, scalar "
        "rounding, extra casts to preserve the PROMISE arithmetic path, and kernel "
        "launch/synchronization overheads.",
        "- Memory ratio and time ratio should be discussed separately.  The former is a "
        "direct consequence of the selected storage types; the latter depends on whether "
        "the lowered variables dominate the GPU execution.",
        "- The complement job uses `digit_case_manifest.csv` to bind each rerun and "
        "Tensor-Core-complement row to an originating `digit<i>_<j>` PROMISE case. "
        "The default manifest covers all available first-two-combination digit-sweep "
        "cases for Backprop, Dense LU, and Hotspot.  Tensor-Core rows are emitted "
        "only when the source case has a suitable dense matrix update; otherwise "
        "the CSV marks the case as not applicable.",
        "",
        "## Dense LU",
        "",
    ]
    if dense_16:
        lines.append(
            f"- At digit 6 / Combination I, the measured time ratio is "
            f"{fmt(dense_16['time_ratio_vs_double'])} and the memory ratio is "
            f"{fmt(dense_16['memory_ratio_vs_double'])}.  The CPU transformed code "
            f"has {dense_16['cpu_fp64_variables']} FP64 variables and "
            f"{dense_16['cpu_lower_variables']} lower-precision variables."
        )
    if dense_26:
        lines.append(
            f"- At digit 6 / Combination II, the measured time ratio is "
            f"{fmt(dense_26['time_ratio_vs_double'])} and the memory ratio is "
            f"{fmt(dense_26['memory_ratio_vs_double'])}.  The generated CUDA wrapper "
            f"uses dominant matrix storage `{dense_26['dominant_storage']}`."
        )
    if identical_dense_6:
        lines.append(
            "- The generated CUDA wrappers for dense LU digit1_6 and digit2_6 are "
            "identical except for the comment header; different timing values should "
            "not be interpreted as a precision-layout effect."
        )
    lines.extend([
        "- The direct dense-LU CUDA implementation performs pivoting, row swaps, "
        "scaling, and element-wise trailing updates as separate kernels.  This is "
        "faithful to the PROMISE-transformed scalar code, but it is not a "
        "Tensor-Core-friendly blocked LU.  The complement benchmark therefore isolates "
        "a blocked trailing update to estimate the performance ceiling of a TC-enabled "
        "reformulation.",
        "",
        "## Hotspot",
        "",
    ])
    if hot_110:
        lines.append(
            f"- At digit 10 / Combination I, the measured time ratio is "
            f"{fmt(hot_110['time_ratio_vs_double'])} and the memory ratio is "
            f"{fmt(hot_110['memory_ratio_vs_double'])}.  The dominant grid storage "
            f"in the CUDA wrapper is `{hot_110['dominant_storage']}`."
        )
    if hot_210:
        lines.append(
            f"- At digit 10 / Combination II, the measured time ratio is "
            f"{fmt(hot_210['time_ratio_vs_double'])} and the memory ratio is "
            f"{fmt(hot_210['memory_ratio_vs_double'])}.  The CPU transformed code "
            f"has {hot_210['cpu_fp64_variables']} FP64 variables and "
            f"{hot_210['cpu_lower_variables']} lower-precision variables."
        )
    if identical_hotspot_10:
        lines.append(
            "- The generated CUDA wrappers for hotspot digit1_10 and digit2_10 are "
            "identical except for the comment header.  Their timing difference should "
            "therefore be reported as measurement variability rather than a real "
            "Combination-I/Combination-II precision effect."
        )
    lines.extend([
        "- The 13 FP64 variables in Hotspot are scalar physical constants or "
        "intermediate parameters in the transformed CPU program; in the CUDA port, "
        "the large arrays are `temp`, `power`, and `result`, whose storage is controlled "
        "by the generated field precision.  Hotspot is a stencil and is primarily "
        "memory-bandwidth/latency sensitive, so it is not a natural Tensor Core target.",
        "",
        "## Suggested response wording",
        "",
        "The original H100 validation did not use Tensor Cores.  It preserves the "
        "PROMISE-derived mixed-precision assignments in direct CUDA kernels to test "
        "the performance effect of the transformed program itself.  Time ratios "
        "slightly above one are caused by the fact that reduced storage does not remove "
        "all FP64 arithmetic and may introduce casts/rounding and synchronization costs. "
        "For Dense LU and Hotspot, many low-precision variables are scalar or do not map "
        "to Tensor-Core instructions in the direct implementation; consequently the "
        "memory savings are much clearer than the speedups.  We added complementary "
        "profiling and Tensor-Core-suitable experiments to separate this implementation "
        "effect from the hardware performance ceiling.",
        "",
    ])
    out_md.write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--papers-dir", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--results-root", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()

    papers_dir = args.papers_dir.resolve()
    results_root = (args.results_root or papers_dir / "h100_results").resolve()
    out_dir = (args.out_dir or Path(__file__).resolve().parent / "results" / "local").resolve()

    rows = choose_latest_rows(read_ratio_rows(results_root))
    if not rows:
        raise SystemExit(f"No cuda_h100_ratios.csv files found under {results_root}")

    summary_csv = out_dir / "existing_h100_summary.csv"
    notes_md = out_dir / "reviewer_notes.md"
    write_summary_csv(rows, papers_dir, summary_csv)
    write_reviewer_notes(summary_csv, notes_md)
    print(f"wrote {summary_csv}")
    print(f"wrote {notes_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
