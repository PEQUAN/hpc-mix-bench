#!/usr/bin/env sh
set -eu

ROOT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
BENCHMARKS="${BENCHMARKS:-backprop dense_lu hotspot}"
BACKPROP_SIZE="${BACKPROP_SIZE:-65536}"
DENSE_LU_SIZE="${DENSE_LU_SIZE:-500}"
HOTSPOT_ROWS="${HOTSPOT_ROWS:-512}"
HOTSPOT_COLS="${HOTSPOT_COLS:-512}"
HOTSPOT_ITERS="${HOTSPOT_ITERS:-2}"
HOTSPOT_TEMP_FILE="${HOTSPOT_TEMP_FILE:-$ROOT_DIR/hotspot/temp_512}"
HOTSPOT_POWER_FILE="${HOTSPOT_POWER_FILE:-$ROOT_DIR/hotspot/power_512}"
HOTSPOT_OUTPUT_DIR="${HOTSPOT_OUTPUT_DIR:-$ROOT_DIR/hotspot/cuda_outputs}"
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
  i=0
  while [ "$i" -lt "$MEASURED_RUNS" ]; do
    output=$("$exe" "$@")
    time_ms=$(extract_metric "$output" "$time_key")
    bytes=$(extract_metric "$output" "device_allocation_bytes")
    mib=$(extract_metric "$output" "device_allocation_mib")
    if [ -z "$time_ms" ] || [ -z "$bytes" ] || [ -z "$mib" ]; then
      echo "missing metric in ${bench}/${case_name} run $((i + 1))" >&2
      exit 1
    fi
    times="${times}${time_ms}
"
    i=$((i + 1))
  done

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
  printf '%s,%s,%s,%s,%s,%s,%s\n' "$bench" "$case_name" "$stats" "$bytes" "$mib" "$WARMUP_RUNS" "$MEASURED_RUNS"
}

printf 'benchmark,case,time_ms,time_ms_stddev,time_ms_min,time_ms_max,time_ms_runs,device_allocation_bytes,device_allocation_mib,warmup_runs,measured_runs\n'

for bench in $BENCHMARKS; do
  case "$bench" in
    backprop)
      run_case backprop double "$ROOT_DIR/backprop/backprop_cuda_double" kernel_time_ms "$BACKPROP_SIZE"
      for dir in "$ROOT_DIR/backprop"/digit*_*; do
        [ -d "$dir" ] || continue
        run_case backprop "${dir##*/}" "$dir/backprop_cuda" kernel_time_ms "$BACKPROP_SIZE"
      done
      ;;
    dense_lu)
      run_case dense_lu double "$ROOT_DIR/dense_lu/dense_lu_cuda_double" factorization_time_ms "$DENSE_LU_SIZE"
      for dir in "$ROOT_DIR/dense_lu"/digit*_*; do
        [ -d "$dir" ] || continue
        run_case dense_lu "${dir##*/}" "$dir/dense_lu_cuda" factorization_time_ms "$DENSE_LU_SIZE"
      done
      ;;
    hotspot)
      mkdir -p "$HOTSPOT_OUTPUT_DIR"
      run_case hotspot double "$ROOT_DIR/hotspot/hotspot_cuda_double" kernel_time_ms \
        "$HOTSPOT_ROWS" "$HOTSPOT_COLS" "$HOTSPOT_ITERS" \
        "$HOTSPOT_TEMP_FILE" "$HOTSPOT_POWER_FILE" \
        "$HOTSPOT_OUTPUT_DIR/hotspot_double.out"
      for dir in "$ROOT_DIR/hotspot"/digit*_*; do
        [ -d "$dir" ] || continue
        run_case hotspot "${dir##*/}" "$dir/hotspot_cuda" kernel_time_ms \
          "$HOTSPOT_ROWS" "$HOTSPOT_COLS" "$HOTSPOT_ITERS" \
          "$HOTSPOT_TEMP_FILE" "$HOTSPOT_POWER_FILE" \
          "$HOTSPOT_OUTPUT_DIR/${dir##*/}.out"
      done
      ;;
    *)
      echo "unknown benchmark: $bench" >&2
      exit 1
      ;;
  esac
done
