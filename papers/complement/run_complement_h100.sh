#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PAPERS_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"

OUT_DIR="${OUT_DIR:-$SCRIPT_DIR/results/manual}"
BENCHMARKS="${BENCHMARKS:-backprop dense_lu hotspot}"
COMBINATIONS="${COMBINATIONS:-1 2}"
MANIFEST="${MANIFEST:-$SCRIPT_DIR/digit_case_manifest.csv}"
CUDA_ARCH="${CUDA_ARCH:-sm_90}"
FORCE_REBUILD="${FORCE_REBUILD:-1}"
WARMUP_RUNS="${WARMUP_RUNS:-1}"
MEASURED_RUNS="${MEASURED_RUNS:-5}"
PLOT_FONT_SIZE="${PLOT_FONT_SIZE:-12}"
RUN_PLOTS="${RUN_PLOTS:-1}"

BACKPROP_SIZE="${BACKPROP_SIZE:-65536}"
BACKPROP_OUTPUT_DIR="${BACKPROP_OUTPUT_DIR:-$OUT_DIR/backprop_outputs}"
DENSE_LU_SIZE="${DENSE_LU_SIZE:-5000}"
DENSE_LU_OUTPUT_DIR="${DENSE_LU_OUTPUT_DIR:-$OUT_DIR/dense_lu_solutions}"
HOTSPOT_ROWS="${HOTSPOT_ROWS:-1024}"
HOTSPOT_COLS="${HOTSPOT_COLS:-1024}"
HOTSPOT_ITERS="${HOTSPOT_ITERS:-2}"
HOTSPOT_TEMP_FILE="${HOTSPOT_TEMP_FILE:-$OUT_DIR/hotspot_inputs/temp_${HOTSPOT_ROWS}}"
HOTSPOT_POWER_FILE="${HOTSPOT_POWER_FILE:-$OUT_DIR/hotspot_inputs/power_${HOTSPOT_ROWS}}"
HOTSPOT_INPUT_GENERATOR="${HOTSPOT_INPUT_GENERATOR:-$PAPERS_DIR/hotspot/generate_hotspot_input.py}"
HOTSPOT_GENERATE_INPUTS="${HOTSPOT_GENERATE_INPUTS:-1}"

DENSE_LU_PANEL="${DENSE_LU_PANEL:-64}"
ENABLE_NCU="${ENABLE_NCU:-0}"

mkdir -p "$OUT_DIR"

echo "out_dir=$OUT_DIR"
echo "papers_dir=$PAPERS_DIR"
echo "benchmarks=$BENCHMARKS"
echo "combinations=$COMBINATIONS"
echo "manifest=$MANIFEST"
echo "cuda_arch=$CUDA_ARCH"
echo "warmup_runs=$WARMUP_RUNS"
echo "measured_runs=$MEASURED_RUNS"
echo "plot_font_size=$PLOT_FONT_SIZE"
echo "run_plots=$RUN_PLOTS"
echo "hotspot_rows=$HOTSPOT_ROWS"
echo "hotspot_cols=$HOTSPOT_COLS"
echo "hotspot_iters=$HOTSPOT_ITERS"
echo "hotspot_temp_file=$HOTSPOT_TEMP_FILE"
echo "hotspot_power_file=$HOTSPOT_POWER_FILE"
echo "hotspot_generate_inputs=$HOTSPOT_GENERATE_INPUTS"

if [[ ! -f "$MANIFEST" ]]; then
  echo "missing manifest: $MANIFEST" >&2
  exit 1
fi

python3 "$SCRIPT_DIR/analyze_h100_results.py" \
  --papers-dir "$PAPERS_DIR" \
  --results-root "$PAPERS_DIR/h100_results" \
  --out-dir "$OUT_DIR"

if [[ "$FORCE_REBUILD" == "1" ]]; then
  echo "Cleaning direct CUDA targets..."
  for bench in $BENCHMARKS; do
    if [[ -f "$PAPERS_DIR/$bench/Makefile.cuda" ]]; then
      make -C "$PAPERS_DIR/$bench" -f Makefile.cuda clean
    fi
    for dir in "$PAPERS_DIR/$bench"/digit*_*; do
      [[ -d "$dir" && -f "$dir/Makefile.cuda" ]] || continue
      make -C "$dir" -f Makefile.cuda clean
    done
  done
  make -C "$SCRIPT_DIR/dense_lu_tensorcore" clean
fi

echo "Building direct CUDA ports..."
BENCHMARKS="$BENCHMARKS" CUDA_ARCH="$CUDA_ARCH" \
  sh "$PAPERS_DIR/build_cuda_h100.sh" > "$OUT_DIR/build_direct_cuda.log" 2>&1

echo "Building complement Tensor Core benchmarks..."
make -C "$SCRIPT_DIR/dense_lu_tensorcore" CUDA_ARCH="$CUDA_ARCH" \
  > "$OUT_DIR/build_dense_lu_tensorcore.log" 2>&1

extract_metric() {
  local text="$1"
  local key="$2"
  printf '%s\n' "$text" | sed -n "s/^${key}=//p" | head -n 1
}

compare_solution_to_double() {
  local ref_file="$1"
  local candidate_file="$2"
  awk '
    NR == FNR {
      ref[++n] = $1 + 0
      ref_l2 += ref[n] * ref[n]
      next
    }
    {
      i++
      diff = ($1 + 0) - ref[i]
      err_l2 += diff * diff
      absdiff = diff < 0 ? -diff : diff
      if (absdiff > err_linf) err_linf = absdiff
      absref = ref[i] < 0 ? -ref[i] : ref[i]
      if (absref > ref_linf) ref_linf = absref
    }
    END {
      if (i != n || n == 0) exit 2
      l2 = (ref_l2 > 0) ? sqrt(err_l2) / sqrt(ref_l2) : sqrt(err_l2)
      linf = (ref_linf > 0) ? err_linf / ref_linf : err_linf
      printf "%.9g,%.9g", l2, linf
    }' "$ref_file" "$candidate_file"
}

compare_vector_to_double() {
  local ref_file="$1"
  local candidate_file="$2"
  awk '
    NR == FNR {
      ref[++n] = $1 + 0
      ref_l2 += ref[n] * ref[n]
      next
    }
    {
      i++
      diff = ($1 + 0) - ref[i]
      mse += diff * diff
      err_l2 += diff * diff
      absdiff = diff < 0 ? -diff : diff
      if (absdiff > err_linf) err_linf = absdiff
      absref = ref[i] < 0 ? -ref[i] : ref[i]
      if (absref > ref_linf) ref_linf = absref
    }
    END {
      if (i != n || n == 0) exit 2
      mse /= n
      l2 = (ref_l2 > 0) ? sqrt(err_l2) / sqrt(ref_l2) : sqrt(err_l2)
      linf = (ref_linf > 0) ? err_linf / ref_linf : err_linf
      printf "%.9g,%.9g,%.9g", mse, l2, linf
    }' "$ref_file" "$candidate_file"
}

compare_indexed_output_to_double() {
  local ref_file="$1"
  local candidate_file="$2"
  awk '
    NR == FNR {
      ref[++n] = $2 + 0
      ref_l2 += ref[n] * ref[n]
      next
    }
    {
      i++
      diff = ($2 + 0) - ref[i]
      err_l2 += diff * diff
      absdiff = diff < 0 ? -diff : diff
      if (absdiff > err_linf) err_linf = absdiff
      absref = ref[i] < 0 ? -ref[i] : ref[i]
      if (absref > ref_linf) ref_linf = absref
    }
    END {
      if (i != n || n == 0) exit 2
      l2 = (ref_l2 > 0) ? sqrt(err_l2) / sqrt(ref_l2) : sqrt(err_l2)
      linf = (ref_linf > 0) ? err_linf / ref_linf : err_linf
      printf "%.9g,%.9g", l2, linf
    }' "$ref_file" "$candidate_file"
}

ensure_hotspot_inputs() {
  if [[ -f "$HOTSPOT_TEMP_FILE" && -f "$HOTSPOT_POWER_FILE" ]]; then
    return
  fi
  if [[ "$HOTSPOT_GENERATE_INPUTS" != "1" ]]; then
    echo "missing Hotspot input files: $HOTSPOT_TEMP_FILE $HOTSPOT_POWER_FILE" >&2
    return 1
  fi
  if [[ ! -f "$HOTSPOT_INPUT_GENERATOR" ]]; then
    echo "missing Hotspot input generator: $HOTSPOT_INPUT_GENERATOR" >&2
    return 1
  fi
  python3 "$HOTSPOT_INPUT_GENERATOR" \
    "$HOTSPOT_ROWS" "$HOTSPOT_COLS" "$HOTSPOT_TEMP_FILE" "$HOTSPOT_POWER_FILE" >&2
}

summarize_values() {
  awk '
    NF {
      x = $1 + 0
      vals[++n] = x
      sum += x
      if (n == 1 || x < min) min = x
      if (n == 1 || x > max) max = x
    }
    END {
      mean = sum / n
      if (n > 1) {
        for (i = 1; i <= n; ++i) {
          diff = vals[i] - mean
          ss += diff * diff
        }
        stddev = sqrt(ss / (n - 1))
      } else {
        stddev = 0
      }
      printf "%.9g,%.9g,%.9g,%.9g", mean, stddev, min, max
    }'
}

run_direct_case() {
  local bench="$1"
  local case_name="$2"
  local exe="$3"
  local time_key="$4"
  shift 4

  if [[ ! -x "$exe" ]]; then
    echo "missing executable: $exe" >&2
    return 1
  fi

  local i=0
  while [[ "$i" -lt "$WARMUP_RUNS" ]]; do
    "$exe" "$@" >/dev/null
    i=$((i + 1))
  done

  local times=""
  local wall_times=""
  local bytes=""
  local mib=""
  local rel_residual=""
  local rel_error=""
  i=0
  while [[ "$i" -lt "$MEASURED_RUNS" ]]; do
    local output
    local start_ns
    local end_ns
    start_ns="$(date +%s%N)"
    output="$("$exe" "$@")"
    end_ns="$(date +%s%N)"
    local time_ms
    local wall_ms
    time_ms="$(extract_metric "$output" "$time_key")"
    wall_ms="$(awk -v start="$start_ns" -v end="$end_ns" 'BEGIN { printf "%.9g", (end - start) / 1000000.0 }')"
    bytes="$(extract_metric "$output" "device_allocation_bytes")"
    mib="$(extract_metric "$output" "device_allocation_mib")"
    rel_residual="$(extract_metric "$output" "relative_residual")"
    rel_error="$(extract_metric "$output" "relative_error")"
    if [[ -z "$time_ms" || -z "$bytes" || -z "$mib" ]]; then
      echo "missing metrics in ${bench}/${case_name}" >&2
      return 1
    fi
    times="${times}${time_ms}"$'\n'
    wall_times="${wall_times}${wall_ms}"$'\n'
    i=$((i + 1))
  done

  if [[ "$bench" == "backprop" && -n "${BACKPROP_CURRENT_OUTPUT_DELTA:-}" ]]; then
    "$exe" "$@" "$BACKPROP_CURRENT_OUTPUT_DELTA" >/dev/null
  fi

  local stats
  local wall_stats
  local kernel_mean
  local wall_mean
  local overhead_ms
  stats="$(printf '%s' "$times" | summarize_values)"
  wall_stats="$(printf '%s' "$wall_times" | summarize_values)"
  kernel_mean="${stats%%,*}"
  wall_mean="${wall_stats%%,*}"
  overhead_ms="$(awk -v wall="$wall_mean" -v kernel="$kernel_mean" 'BEGIN { v = wall - kernel; if (v < 0) v = 0; printf "%.9g", v }')"
  local solution_l2=""
  local solution_linf=""
  local output_delta_mse=""
  local output_delta_l2=""
  local output_delta_linf=""
  local output_l2=""
  local output_linf=""
  if [[ "$bench" == "backprop" ]]; then
    if [[ "$case_name" == "double" ]]; then
      output_delta_mse="0"
      output_delta_l2="0"
      output_delta_linf="0"
    elif [[ -n "${BACKPROP_REFERENCE_OUTPUT_DELTA:-}" && -n "${BACKPROP_CURRENT_OUTPUT_DELTA:-}" ]]; then
      local compare_stats
      compare_stats="$(compare_vector_to_double "$BACKPROP_REFERENCE_OUTPUT_DELTA" "$BACKPROP_CURRENT_OUTPUT_DELTA")"
      output_delta_mse="${compare_stats%%,*}"
      local rest="${compare_stats#*,}"
      output_delta_l2="${rest%,*}"
      output_delta_linf="${rest#*,}"
    fi
  elif [[ "$bench" == "dense_lu" ]]; then
    if [[ "$case_name" == "double" ]]; then
      solution_l2="0"
      solution_linf="0"
    elif [[ -n "${DENSE_LU_REFERENCE_SOLUTION:-}" && -n "${DENSE_LU_CURRENT_SOLUTION:-}" ]]; then
      local compare_stats
      compare_stats="$(compare_solution_to_double "$DENSE_LU_REFERENCE_SOLUTION" "$DENSE_LU_CURRENT_SOLUTION")"
      solution_l2="${compare_stats%,*}"
      solution_linf="${compare_stats#*,}"
    fi
  elif [[ "$bench" == "hotspot" ]]; then
    if [[ "$case_name" == "double" ]]; then
      output_l2="0"
      output_linf="0"
    elif [[ -n "${HOTSPOT_REFERENCE_OUTPUT:-}" && -n "${HOTSPOT_CURRENT_OUTPUT:-}" ]]; then
      local compare_stats
      compare_stats="$(compare_indexed_output_to_double "$HOTSPOT_REFERENCE_OUTPUT" "$HOTSPOT_CURRENT_OUTPUT")"
      output_l2="${compare_stats%,*}"
      output_linf="${compare_stats#*,}"
    fi
  fi
  printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
    "$bench" "$case_name" "$stats" "$wall_stats" "$overhead_ms" "$bytes" "$mib" \
    "${rel_residual:-}" "${rel_error:-}" "$solution_l2" "$solution_linf" \
    "$output_delta_mse" "$output_delta_l2" "$output_delta_linf" \
    "$output_l2" "$output_linf" "$WARMUP_RUNS" "$MEASURED_RUNS"

  if [[ "$ENABLE_NCU" == "1" && "$(command -v ncu || true)" != "" ]]; then
    mkdir -p "$OUT_DIR/ncu"
    ncu --target-processes all \
      --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed,dram__throughput.avg.pct_of_peak_sustained_elapsed \
      --csv --log-file "$OUT_DIR/ncu/${bench}_${case_name}.csv" \
      "$exe" "$@" >/dev/null || true
  fi
}

run_manifest_direct_case() {
  local bench="$1"
  local case_name="$2"
  case "$bench" in
    backprop)
      local output_delta_path="$BACKPROP_OUTPUT_DIR/${case_name}_output_delta.txt"
      BACKPROP_REFERENCE_OUTPUT_DELTA="$BACKPROP_REFERENCE_OUTPUT_DELTA" \
      BACKPROP_CURRENT_OUTPUT_DELTA="$output_delta_path" \
        run_direct_case "$bench" "$case_name" "$PAPERS_DIR/backprop/$case_name/backprop_cuda" \
        kernel_time_ms "$BACKPROP_SIZE"
      ;;
    dense_lu)
      local solution_path="$DENSE_LU_OUTPUT_DIR/${case_name}_solution.txt"
      DENSE_LU_REFERENCE_SOLUTION="$DENSE_LU_REFERENCE_SOLUTION" \
      DENSE_LU_CURRENT_SOLUTION="$solution_path" \
        run_direct_case "$bench" "$case_name" "$PAPERS_DIR/dense_lu/$case_name/dense_lu_cuda" \
        factorization_time_ms "$DENSE_LU_SIZE" "$solution_path"
      ;;
    hotspot)
      mkdir -p "$OUT_DIR/hotspot_outputs"
      local output_path="$OUT_DIR/hotspot_outputs/${case_name}.out"
      HOTSPOT_REFERENCE_OUTPUT="$HOTSPOT_REFERENCE_OUTPUT" \
      HOTSPOT_CURRENT_OUTPUT="$output_path" \
      run_direct_case "$bench" "$case_name" "$PAPERS_DIR/hotspot/$case_name/hotspot_cuda" \
        kernel_time_ms "$HOTSPOT_ROWS" "$HOTSPOT_COLS" "$HOTSPOT_ITERS" \
        "$HOTSPOT_TEMP_FILE" "$HOTSPOT_POWER_FILE" "$output_path"
      ;;
    *)
      echo "unknown benchmark in manifest: $bench" >&2
      return 1
      ;;
  esac
}

echo "Running selected direct CUDA cases..."
{
  printf 'benchmark,case,kernel_time_ms,kernel_time_ms_stddev,kernel_time_ms_min,kernel_time_ms_max,total_time_ms,total_time_ms_stddev,total_time_ms_min,total_time_ms_max,non_kernel_overhead_ms,device_allocation_bytes,device_allocation_mib,relative_residual,relative_error,solution_l2_error_vs_double,solution_linf_error_vs_double,output_delta_mse_vs_double,output_delta_l2_error_vs_double,output_delta_linf_error_vs_double,output_l2_error_vs_double,output_linf_error_vs_double,warmup_runs,measured_runs\n'
  mkdir -p "$BACKPROP_OUTPUT_DIR"
  BACKPROP_REFERENCE_OUTPUT_DELTA="$BACKPROP_OUTPUT_DIR/double_output_delta.txt"
  BACKPROP_CURRENT_OUTPUT_DELTA="$BACKPROP_REFERENCE_OUTPUT_DELTA" \
    run_direct_case backprop double "$PAPERS_DIR/backprop/backprop_cuda_double" kernel_time_ms \
    "$BACKPROP_SIZE"
  mkdir -p "$DENSE_LU_OUTPUT_DIR"
  DENSE_LU_REFERENCE_SOLUTION="$DENSE_LU_OUTPUT_DIR/double_solution.txt"
  DENSE_LU_CURRENT_SOLUTION="$DENSE_LU_REFERENCE_SOLUTION" \
    run_direct_case dense_lu double "$PAPERS_DIR/dense_lu/dense_lu_cuda_double" factorization_time_ms \
    "$DENSE_LU_SIZE" "$DENSE_LU_REFERENCE_SOLUTION"
  mkdir -p "$OUT_DIR/hotspot_outputs"
  ensure_hotspot_inputs
  HOTSPOT_REFERENCE_OUTPUT="$OUT_DIR/hotspot_outputs/double.out"
  HOTSPOT_CURRENT_OUTPUT="$HOTSPOT_REFERENCE_OUTPUT" \
    run_direct_case hotspot double "$PAPERS_DIR/hotspot/hotspot_cuda_double" kernel_time_ms \
    "$HOTSPOT_ROWS" "$HOTSPOT_COLS" "$HOTSPOT_ITERS" \
    "$HOTSPOT_TEMP_FILE" "$HOTSPOT_POWER_FILE" "$OUT_DIR/hotspot_outputs/double.out"

  while IFS=, read -r bench case_name tc_mode notes; do
    [[ "$bench" != "benchmark" ]] || continue
    run_manifest_direct_case "$bench" "$case_name"
  done < "$MANIFEST"
} > "$OUT_DIR/direct_case_rerun.csv"

awk -F, \
  -v bp_size="$BACKPROP_SIZE" \
  -v dl_size="$DENSE_LU_SIZE" \
  -v hs_rows="$HOTSPOT_ROWS" \
  -v hs_cols="$HOTSPOT_COLS" \
  -v hs_iters="$HOTSPOT_ITERS" \
  -v warmup_runs="$WARMUP_RUNS" \
  -v measured_runs="$MEASURED_RUNS" '
BEGIN { OFS = "," }
NR == 1 { next }
$2 == "double" {
  double_time[$1] = $3 + 0
  double_time_stddev[$1] = $4 + 0
  double_bytes[$1] = $12 + 0
}
{ rows[++n] = $0 }
END {
  print "benchmark,case,precision,input_size,time_ms,time_ms_stddev,time_ms_min,time_ms_max,time_ms_runs,device_allocation_bytes,device_allocation_mib,speedup_vs_double,speedup_vs_double_stddev,time_ratio_vs_double,time_ratio_vs_double_stddev,memory_ratio_vs_double,relative_residual,relative_error,solution_l2_error_vs_double,solution_linf_error_vs_double,output_delta_mse_vs_double,output_delta_l2_error_vs_double,output_delta_linf_error_vs_double,output_l2_error_vs_double,output_linf_error_vs_double,warmup_runs,measured_runs"
  for (i = 1; i <= n; i++) {
    split(rows[i], a, ",")
    bench = a[1]
    case_name = a[2]
    time_ms = a[3] + 0
    time_stddev = a[4] + 0
    time_min = a[5] + 0
    time_max = a[6] + 0
    bytes = a[12] + 0
    mib = a[13] + 0
    relative_residual = a[14]
    relative_error = a[15]
    solution_l2_error = a[16]
    solution_linf_error = a[17]
    output_delta_mse = a[18]
    output_delta_l2_error = a[19]
    output_delta_linf_error = a[20]
    output_l2_error = a[21]
    output_linf_error = a[22]
    precision = (case_name == "double") ? "double" : "mixed"
    if (bench == "backprop") {
      input_size = bp_size
    } else if (bench == "dense_lu") {
      input_size = dl_size
    } else {
      input_size = hs_rows "x" hs_cols "x" hs_iters
    }
    speedup = (time_ms > 0) ? double_time[bench] / time_ms : 0
    time_ratio = (double_time[bench] > 0) ? time_ms / double_time[bench] : 0
    memory_ratio = (double_bytes[bench] > 0) ? bytes / double_bytes[bench] : 0
    if (time_ms > 0 && double_time[bench] > 0) {
      rel = sqrt((time_stddev / time_ms) ^ 2 + (double_time_stddev[bench] / double_time[bench]) ^ 2)
      speedup_stddev = speedup * rel
      time_ratio_stddev = time_ratio * rel
    } else {
      speedup_stddev = 0
      time_ratio_stddev = 0
    }
    if (case_name == "double") {
      speedup = 1
      speedup_stddev = 0
      time_ratio = 1
      time_ratio_stddev = 0
      memory_ratio = 1
    }
    printf "%s,%s,%s,%s,%.9g,%.9g,%.9g,%.9g,%d,%.0f,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%s,%s,%s,%s,%s,%s,%s,%s,%s,%d,%d\n", \
      bench, case_name, precision, input_size, time_ms, time_stddev, time_min, time_max, measured_runs, bytes, mib, speedup, speedup_stddev, time_ratio, time_ratio_stddev, memory_ratio, relative_residual, relative_error, solution_l2_error, solution_linf_error, output_delta_mse, output_delta_l2_error, output_delta_linf_error, output_l2_error, output_linf_error, warmup_runs, measured_runs
  }
}
' "$OUT_DIR/direct_case_rerun.csv" > "$OUT_DIR/direct_case_ratios.csv"

awk -F, '
BEGIN {
  OFS = ","
  print "benchmark,case,total_time_ms,pure_computation_time_ms,non_kernel_overhead_ms,memory_transfer_time_ms,type_conversion_time_ms,profiling_method,notes"
}
NR == 1 { next }
{
  print $1,$2,$7,$3,$11,"","","cuda_event_kernel_plus_host_wall_clock","memory transfer and type conversion require instrumented CUDA regions or Nsight traces"
}
' "$OUT_DIR/direct_case_rerun.csv" > "$OUT_DIR/profile_breakdown.csv"

if [[ "$RUN_PLOTS" == "1" && -f "$PAPERS_DIR/h100_results/plot_h100_ratios.py" ]]; then
  python3 "$PAPERS_DIR/h100_results/plot_h100_ratios.py" \
    --csv "$OUT_DIR/direct_case_ratios.csv" \
    --out-dir "$OUT_DIR/figures" \
    --font-size "$PLOT_FONT_SIZE" \
    --combinations $COMBINATIONS \
    --formats pdf png || echo "WARNING: plot_h100_ratios.py failed" >&2
fi

if [[ "$RUN_PLOTS" == "1" && -f "$PAPERS_DIR/h100_results/plot_h100_ratios_combined.py" ]]; then
  python3 "$PAPERS_DIR/h100_results/plot_h100_ratios_combined.py" \
    --csv "$OUT_DIR/direct_case_ratios.csv" \
    --out-dir "$OUT_DIR/figures" \
    --font-size "$PLOT_FONT_SIZE" \
    --combinations $COMBINATIONS \
    --formats pdf png || echo "WARNING: plot_h100_ratios_combined.py failed" >&2
fi

if [[ "$RUN_PLOTS" == "1" \
  && -f "$PAPERS_DIR/h100_results/plot_dense_lu_solution_errors.py" \
  && -f "$DENSE_LU_OUTPUT_DIR/double_solution.txt" ]]; then
  python3 "$PAPERS_DIR/h100_results/plot_dense_lu_solution_errors.py" \
    --solutions-dir "$DENSE_LU_OUTPUT_DIR" \
    --out-dir "$OUT_DIR/figures" \
    --font-size "$PLOT_FONT_SIZE" \
    --combinations $COMBINATIONS \
    --formats pdf png || echo "WARNING: plot_dense_lu_solution_errors.py failed" >&2
fi

if [[ "$RUN_PLOTS" == "1" && -f "$PAPERS_DIR/h100_results/plot_h100_accuracy_errors.py" ]]; then
  python3 "$PAPERS_DIR/h100_results/plot_h100_accuracy_errors.py" \
    --csv "$OUT_DIR/direct_case_ratios.csv" \
    --out-dir "$OUT_DIR/figures" \
    --font-size "$PLOT_FONT_SIZE" \
    --combinations $COMBINATIONS \
    --formats pdf png || echo "WARNING: plot_h100_accuracy_errors.py failed" >&2
fi

run_tc_case() {
  local source_benchmark="$1"
  local source_case="$2"
  local accelerator="$3"
  local mode="$4"
  local exe="$5"
  shift 5
  local output
  output="$("$exe" "$@")"
  local time_ms bytes mib gflops tc
  time_ms="$(extract_metric "$output" "kernel_time_ms")"
  bytes="$(extract_metric "$output" "device_allocation_bytes")"
  mib="$(extract_metric "$output" "device_allocation_mib")"
  local rel_l2 rel_linf
  rel_l2="$(extract_metric "$output" "relative_l2_error_vs_fp64")"
  rel_linf="$(extract_metric "$output" "relative_linf_error_vs_fp64")"
  gflops="$(extract_metric "$output" "gflops")"
  tc="$(extract_metric "$output" "uses_tensor_core_candidate")"
  printf '%s,%s,%s,%s,completed,%s,%s,%s,%s,%s,%s,\n' \
    "$source_benchmark" "$source_case" "$accelerator" "$mode" "$time_ms" "$bytes" "$mib" "${rel_l2:-}" "${rel_linf:-}" "${gflops:-}"
  printf '%s\n' "$output" > "$OUT_DIR/${source_benchmark}_${source_case}_${accelerator}_${mode}.log"
  if [[ "${tc:-0}" != "1" ]]; then
    echo "note: $source_benchmark/$source_case $accelerator/$mode is a non-Tensor-Core baseline" >&2
  fi
}

emit_tc_not_applicable() {
  local source_benchmark="$1"
  local source_case="$2"
  local notes="$3"
  printf '%s,%s,none,not_applicable,not_applicable,,,,,,%s\n' \
    "$source_benchmark" "$source_case" "$notes"
}

echo "Running Tensor Core complement benchmarks..."
{
  printf 'source_benchmark,source_case,accelerator,mode,status,time_ms,device_allocation_bytes,device_allocation_mib,relative_l2_error_vs_fp64,relative_linf_error_vs_fp64,gflops,notes\n'
  dense_baseline_done=0
  while IFS=, read -r bench case_name tc_mode notes; do
    [[ "$bench" != "benchmark" ]] || continue
    if [[ "$bench" == "dense_lu" && "$tc_mode" != "not_applicable" ]]; then
      if [[ "$dense_baseline_done" == "0" ]]; then
        run_tc_case dense_lu double dense_lu_blocked_update fp64 \
          "$SCRIPT_DIR/dense_lu_tensorcore/dense_lu_panel_update_tc" \
          "$DENSE_LU_SIZE" "$DENSE_LU_PANEL" fp64 "$WARMUP_RUNS" "$MEASURED_RUNS"
        dense_baseline_done=1
      fi
      run_tc_case "$bench" "$case_name" dense_lu_blocked_update "$tc_mode" \
        "$SCRIPT_DIR/dense_lu_tensorcore/dense_lu_panel_update_tc" \
        "$DENSE_LU_SIZE" "$DENSE_LU_PANEL" "$tc_mode" "$WARMUP_RUNS" "$MEASURED_RUNS"
    else
      emit_tc_not_applicable "$bench" "$case_name" "$notes"
    fi
  done < "$MANIFEST"
} > "$OUT_DIR/tensorcore_complement.csv"

if [[ "$RUN_PLOTS" == "1" && -f "$SCRIPT_DIR/plot_complement_h100.py" ]]; then
  python3 "$SCRIPT_DIR/plot_complement_h100.py" \
    --csv "$OUT_DIR/tensorcore_complement.csv" \
    --out-dir "$OUT_DIR/figures" \
    --font-size "$PLOT_FONT_SIZE" \
    --combinations $COMBINATIONS \
    --formats pdf png || echo "WARNING: plot_complement_h100.py failed" >&2
fi

echo "Done."
echo "summary_csv=$OUT_DIR/existing_h100_summary.csv"
echo "reviewer_notes=$OUT_DIR/reviewer_notes.md"
echo "direct_case_rerun=$OUT_DIR/direct_case_rerun.csv"
echo "direct_case_ratios=$OUT_DIR/direct_case_ratios.csv"
echo "profile_breakdown=$OUT_DIR/profile_breakdown.csv"
echo "tensorcore_complement=$OUT_DIR/tensorcore_complement.csv"
echo "figures=$OUT_DIR/figures"
