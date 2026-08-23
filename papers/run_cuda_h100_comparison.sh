#!/usr/bin/env sh
set -eu

ROOT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
BENCHMARKS="${BENCHMARKS:-backprop dense_lu hotspot}"
COMBINATIONS="${COMBINATIONS:-1 2}"
BACKPROP_SIZE="${BACKPROP_SIZE:-65536}"
BACKPROP_OUTPUT_DIR="${BACKPROP_OUTPUT_DIR:-$ROOT_DIR/backprop/cuda_outputs}"
DENSE_LU_SIZE="${DENSE_LU_SIZE:-5000}"
DENSE_LU_OUTPUT_DIR="${DENSE_LU_OUTPUT_DIR:-$ROOT_DIR/dense_lu/cuda_outputs}"
HOTSPOT_ROWS="${HOTSPOT_ROWS:-1024}"
HOTSPOT_COLS="${HOTSPOT_COLS:-1024}"
HOTSPOT_ITERS="${HOTSPOT_ITERS:-2}"
HOTSPOT_TEMP_FILE="${HOTSPOT_TEMP_FILE:-$ROOT_DIR/hotspot/temp_${HOTSPOT_ROWS}}"
HOTSPOT_POWER_FILE="${HOTSPOT_POWER_FILE:-$ROOT_DIR/hotspot/power_${HOTSPOT_ROWS}}"
HOTSPOT_OUTPUT_DIR="${HOTSPOT_OUTPUT_DIR:-$ROOT_DIR/hotspot/cuda_outputs}"
HOTSPOT_INPUT_GENERATOR="${HOTSPOT_INPUT_GENERATOR:-$ROOT_DIR/hotspot/generate_hotspot_input.py}"
HOTSPOT_GENERATE_INPUTS="${HOTSPOT_GENERATE_INPUTS:-1}"
WARMUP_RUNS="${WARMUP_RUNS:-1}"
MEASURED_RUNS="${MEASURED_RUNS:-3}"

case "$WARMUP_RUNS" in
  ''|*[!0-9]*)
    echo "WARMUP_RUNS must be a non-negative integer" >&2
    exit 1
    ;;
esac

case "$MEASURED_RUNS" in
  ''|*[!0-9]*|0)
    echo "MEASURED_RUNS must be a positive integer" >&2
    exit 1
    ;;
esac

extract_metric() {
  printf '%s\n' "$1" | sed -n "s/^$2=//p" | head -n 1
}

case_enabled() {
  case_name="$1"
  combo=$(printf '%s\n' "$case_name" | sed -n 's/^digit\([0-9][0-9]*\)_[0-9][0-9]*$/\1/p')
  [ -n "$combo" ] || return 1
  for enabled_combo in $COMBINATIONS; do
    if [ "$combo" = "$enabled_combo" ]; then
      return 0
    fi
  done
  return 1
}

compare_solution_to_double() {
  ref_file="$1"
  candidate_file="$2"
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
  ref_file="$1"
  candidate_file="$2"
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
  ref_file="$1"
  candidate_file="$2"
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
  if [ -f "$HOTSPOT_TEMP_FILE" ] && [ -f "$HOTSPOT_POWER_FILE" ]; then
    return
  fi
  if [ "$HOTSPOT_GENERATE_INPUTS" != "1" ]; then
    echo "missing Hotspot input files: $HOTSPOT_TEMP_FILE $HOTSPOT_POWER_FILE" >&2
    exit 1
  fi
  if [ ! -f "$HOTSPOT_INPUT_GENERATOR" ]; then
    echo "missing Hotspot input generator: $HOTSPOT_INPUT_GENERATOR" >&2
    exit 1
  fi
  python3 "$HOTSPOT_INPUT_GENERATOR" \
    "$HOTSPOT_ROWS" "$HOTSPOT_COLS" "$HOTSPOT_TEMP_FILE" "$HOTSPOT_POWER_FILE" >&2
}

run_case() {
  bench="$1"
  case_name="$2"
  exe="$3"
  time_key="$4"
  shift 4

  if [ ! -x "$exe" ]; then
    echo "missing executable: $exe" >&2
    exit 1
  fi

  i=0
  while [ "$i" -lt "$WARMUP_RUNS" ]; do
    "$exe" "$@" >/dev/null
    i=$((i + 1))
  done

  times=""
  bytes=""
  mib=""
  rel_residual=""
  rel_error=""
  i=0
  while [ "$i" -lt "$MEASURED_RUNS" ]; do
    output=$("$exe" "$@")
    time_ms=$(extract_metric "$output" "$time_key")
    bytes=$(extract_metric "$output" "device_allocation_bytes")
    mib=$(extract_metric "$output" "device_allocation_mib")
    rel_residual=$(extract_metric "$output" "relative_residual")
    rel_error=$(extract_metric "$output" "relative_error")
    if [ -z "$time_ms" ] || [ -z "$bytes" ] || [ -z "$mib" ]; then
      echo "missing metric in ${bench}/${case_name} run $((i + 1))" >&2
      exit 1
    fi
    times="${times}${time_ms}
"
    i=$((i + 1))
  done

  if [ "$bench" = "backprop" ] && [ -n "${BACKPROP_CURRENT_OUTPUT_DELTA:-}" ]; then
    "$exe" "$@" "$BACKPROP_CURRENT_OUTPUT_DELTA" >/dev/null
  fi

  stats=$(printf '%s' "$times" | awk '
    NF {
      x = $1 + 0
      vals[++n] = x
      sum += x
      if (n == 1 || x < min) min = x
      if (n == 1 || x > max) max = x
    }
    END {
      if (n == 0) exit 1
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
      printf "%.9g,%.9g,%.9g,%.9g,%d", mean, stddev, min, max, n
    }
  ')
  solution_l2=""
  solution_linf=""
  output_delta_mse=""
  output_delta_l2=""
  output_delta_linf=""
  output_l2=""
  output_linf=""
  if [ "$bench" = "backprop" ]; then
    if [ "$case_name" = "double" ]; then
      output_delta_mse="0"
      output_delta_l2="0"
      output_delta_linf="0"
    elif [ -n "${BACKPROP_REFERENCE_OUTPUT_DELTA:-}" ] && [ -n "${BACKPROP_CURRENT_OUTPUT_DELTA:-}" ]; then
      compare_stats=$(compare_vector_to_double "$BACKPROP_REFERENCE_OUTPUT_DELTA" "$BACKPROP_CURRENT_OUTPUT_DELTA")
      output_delta_mse=${compare_stats%%,*}
      rest=${compare_stats#*,}
      output_delta_l2=${rest%,*}
      output_delta_linf=${rest#*,}
    fi
  elif [ "$bench" = "dense_lu" ]; then
    if [ "$case_name" = "double" ]; then
      solution_l2="0"
      solution_linf="0"
    elif [ -n "${DENSE_LU_REFERENCE_SOLUTION:-}" ] && [ -n "${DENSE_LU_CURRENT_SOLUTION:-}" ]; then
      compare_stats=$(compare_solution_to_double "$DENSE_LU_REFERENCE_SOLUTION" "$DENSE_LU_CURRENT_SOLUTION")
      solution_l2=${compare_stats%,*}
      solution_linf=${compare_stats#*,}
    fi
  elif [ "$bench" = "hotspot" ]; then
    if [ "$case_name" = "double" ]; then
      output_l2="0"
      output_linf="0"
    elif [ -n "${HOTSPOT_REFERENCE_OUTPUT:-}" ] && [ -n "${HOTSPOT_CURRENT_OUTPUT:-}" ]; then
      compare_stats=$(compare_indexed_output_to_double "$HOTSPOT_REFERENCE_OUTPUT" "$HOTSPOT_CURRENT_OUTPUT")
      output_l2=${compare_stats%,*}
      output_linf=${compare_stats#*,}
    fi
  fi
  printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
    "$bench" "$case_name" "$stats" "$bytes" "$mib" "${rel_residual:-}" "${rel_error:-}" \
    "$solution_l2" "$solution_linf" "$output_delta_mse" "$output_delta_l2" "$output_delta_linf" \
    "$output_l2" "$output_linf" "$WARMUP_RUNS" "$MEASURED_RUNS"
}

printf 'benchmark,case,time_ms,time_ms_stddev,time_ms_min,time_ms_max,time_ms_runs,device_allocation_bytes,device_allocation_mib,relative_residual,relative_error,solution_l2_error_vs_double,solution_linf_error_vs_double,output_delta_mse_vs_double,output_delta_l2_error_vs_double,output_delta_linf_error_vs_double,output_l2_error_vs_double,output_linf_error_vs_double,warmup_runs,measured_runs\n'

for bench in $BENCHMARKS; do
  case "$bench" in
    backprop)
      mkdir -p "$BACKPROP_OUTPUT_DIR"
      BACKPROP_REFERENCE_OUTPUT_DELTA="$BACKPROP_OUTPUT_DIR/double_output_delta.txt"
      BACKPROP_CURRENT_OUTPUT_DELTA="$BACKPROP_REFERENCE_OUTPUT_DELTA" \
        run_case backprop double "$ROOT_DIR/backprop/backprop_cuda_double" kernel_time_ms \
        "$BACKPROP_SIZE"
      for dir in "$ROOT_DIR/backprop"/digit*_*; do
        [ -d "$dir" ] || continue
        case_enabled "${dir##*/}" || continue
        case_output_delta="$BACKPROP_OUTPUT_DIR/${dir##*/}_output_delta.txt"
        BACKPROP_REFERENCE_OUTPUT_DELTA="$BACKPROP_REFERENCE_OUTPUT_DELTA" \
        BACKPROP_CURRENT_OUTPUT_DELTA="$case_output_delta" \
          run_case backprop "${dir##*/}" "$dir/backprop_cuda" kernel_time_ms \
          "$BACKPROP_SIZE"
      done
      ;;
    dense_lu)
      mkdir -p "$DENSE_LU_OUTPUT_DIR"
      DENSE_LU_REFERENCE_SOLUTION="$DENSE_LU_OUTPUT_DIR/double_solution.txt"
      DENSE_LU_CURRENT_SOLUTION="$DENSE_LU_REFERENCE_SOLUTION" \
        run_case dense_lu double "$ROOT_DIR/dense_lu/dense_lu_cuda_double" factorization_time_ms \
        "$DENSE_LU_SIZE" "$DENSE_LU_REFERENCE_SOLUTION"
      for dir in "$ROOT_DIR/dense_lu"/digit*_*; do
        [ -d "$dir" ] || continue
        case_enabled "${dir##*/}" || continue
        case_solution="$DENSE_LU_OUTPUT_DIR/${dir##*/}_solution.txt"
        DENSE_LU_REFERENCE_SOLUTION="$DENSE_LU_REFERENCE_SOLUTION" \
        DENSE_LU_CURRENT_SOLUTION="$case_solution" \
          run_case dense_lu "${dir##*/}" "$dir/dense_lu_cuda" factorization_time_ms \
          "$DENSE_LU_SIZE" "$case_solution"
      done
      ;;
    hotspot)
      ensure_hotspot_inputs
      mkdir -p "$HOTSPOT_OUTPUT_DIR"
      HOTSPOT_REFERENCE_OUTPUT="$HOTSPOT_OUTPUT_DIR/hotspot_double.out"
      HOTSPOT_CURRENT_OUTPUT="$HOTSPOT_REFERENCE_OUTPUT" \
        run_case hotspot double "$ROOT_DIR/hotspot/hotspot_cuda_double" kernel_time_ms \
        "$HOTSPOT_ROWS" "$HOTSPOT_COLS" "$HOTSPOT_ITERS" \
        "$HOTSPOT_TEMP_FILE" "$HOTSPOT_POWER_FILE" \
        "$HOTSPOT_REFERENCE_OUTPUT"
      for dir in "$ROOT_DIR/hotspot"/digit*_*; do
        [ -d "$dir" ] || continue
        case_enabled "${dir##*/}" || continue
        case_output="$HOTSPOT_OUTPUT_DIR/${dir##*/}.out"
        HOTSPOT_REFERENCE_OUTPUT="$HOTSPOT_REFERENCE_OUTPUT" \
        HOTSPOT_CURRENT_OUTPUT="$case_output" \
          run_case hotspot "${dir##*/}" "$dir/hotspot_cuda" kernel_time_ms \
          "$HOTSPOT_ROWS" "$HOTSPOT_COLS" "$HOTSPOT_ITERS" \
          "$HOTSPOT_TEMP_FILE" "$HOTSPOT_POWER_FILE" \
          "$case_output"
      done
      ;;
    *)
      echo "unknown benchmark: $bench" >&2
      exit 1
      ;;
  esac
done
