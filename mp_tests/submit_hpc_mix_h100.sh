#!/bin/bash
#SBATCH --job-name=hpcmix-h100
##SBATCH --account=YOUR_PROJECT@h100
#SBATCH -C h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --hint=nomultithread
#SBATCH --qos=qos_gpu_h100-t3
#SBATCH --time=04:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail

# You can submit this script from the hpc-mix-bench/mp_tests directory with:
#   sbatch --account=${IDRPROJ}@h100 --export=ALL,REPO_DIR="$(pwd)/.." submit_hpc_mix_h100.sh
REPO_DIR="${REPO_DIR:-}"
BENCHMARKS="${BENCHMARKS:-backprop dense_lu hotspot}"
BACKPROP_SIZE="${BACKPROP_SIZE:-65536}"
DENSE_LU_SIZE="${DENSE_LU_SIZE:-500}"
HOTSPOT_ROWS="${HOTSPOT_ROWS:-512}"
HOTSPOT_COLS="${HOTSPOT_COLS:-512}"
HOTSPOT_ITERS="${HOTSPOT_ITERS:-2}"
HOTSPOT_TEMP_FILE="${HOTSPOT_TEMP_FILE:-}"
HOTSPOT_POWER_FILE="${HOTSPOT_POWER_FILE:-}"
CUDA_ARCH="${CUDA_ARCH:-sm_90}"
FORCE_REBUILD="${FORCE_REBUILD:-1}"
WARMUP_RUNS="${WARMUP_RUNS:-1}"
MEASURED_RUNS="${MEASURED_RUNS:-3}"

if [[ -z "$REPO_DIR" ]]; then
  for candidate in "$SLURM_SUBMIT_DIR/.." "$SLURM_SUBMIT_DIR" "$SLURM_SUBMIT_DIR/hpc-mix-bench" "$PWD/.." "$PWD" "$PWD/hpc-mix-bench"; do
    if [[ -d "$candidate/mp_tests" ]]; then
      REPO_DIR="$candidate"
      break
    fi
  done
fi

if [[ -z "$REPO_DIR" || ! -d "$REPO_DIR/mp_tests" ]]; then
  echo "ERROR: could not find hpc-mix-bench/mp_tests." >&2
  echo "Submit from mp_tests with: sbatch --account=\${IDRPROJ}@h100 --export=ALL,REPO_DIR=\"\$(pwd)/..\" submit_hpc_mix_h100.sh" >&2
  exit 1
fi

MP_TESTS="$REPO_DIR/mp_tests"
OUT_DIR="${OUT_DIR:-$SLURM_SUBMIT_DIR/h100_results/${SLURM_JOB_ID:-manual}}"
BUILD_LOG="$OUT_DIR/build_cuda_h100.log"
RAW_CSV="$OUT_DIR/cuda_h100_raw.csv"
RATIO_CSV="$OUT_DIR/cuda_h100_ratios.csv"
HOTSPOT_TEMP_FILE="${HOTSPOT_TEMP_FILE:-$MP_TESTS/hotspot/temp_512}"
HOTSPOT_POWER_FILE="${HOTSPOT_POWER_FILE:-$MP_TESTS/hotspot/power_512}"
HOTSPOT_OUTPUT_DIR="$OUT_DIR/hotspot_outputs"

mkdir -p "$OUT_DIR"

echo "job_id=${SLURM_JOB_ID:-manual}"
echo "host=$(hostname)"
echo "repo_dir=$REPO_DIR"
echo "out_dir=$OUT_DIR"
echo "benchmarks=$BENCHMARKS"
echo "backprop_size=$BACKPROP_SIZE"
echo "dense_lu_size=$DENSE_LU_SIZE"
echo "hotspot_rows=$HOTSPOT_ROWS"
echo "hotspot_cols=$HOTSPOT_COLS"
echo "hotspot_iters=$HOTSPOT_ITERS"
echo "hotspot_temp_file=$HOTSPOT_TEMP_FILE"
echo "hotspot_power_file=$HOTSPOT_POWER_FILE"
echo "cuda_arch=$CUDA_ARCH"
echo "warmup_runs=$WARMUP_RUNS"
echo "measured_runs=$MEASURED_RUNS"

module purge
module load arch/h100
module load cuda/12.4.1 || module load cuda/12.2.0 || module load cuda
module list

which nvcc
nvcc --version
nvidia-smi

cd "$MP_TESTS"

if [[ "$FORCE_REBUILD" == "1" ]]; then
  echo "Cleaning previous CUDA H100 targets..."
  for bench in $BENCHMARKS; do
    if [[ -f "$bench/Makefile.cuda" ]]; then
      make -C "$bench" -f Makefile.cuda clean
    fi
    for dir in "$bench"/digit*_*; do
      [[ -d "$dir" ]] || continue
      make -C "$dir" -f Makefile.cuda clean
    done
  done
fi

echo "Building CUDA H100 benchmarks..."
BENCHMARKS="$BENCHMARKS" CUDA_ARCH="$CUDA_ARCH" \
  sh "$MP_TESTS/build_cuda_h100.sh" > "$BUILD_LOG" 2>&1

echo "Running CUDA H100 comparisons..."
BENCHMARKS="$BENCHMARKS" \
BACKPROP_SIZE="$BACKPROP_SIZE" \
DENSE_LU_SIZE="$DENSE_LU_SIZE" \
HOTSPOT_ROWS="$HOTSPOT_ROWS" \
HOTSPOT_COLS="$HOTSPOT_COLS" \
HOTSPOT_ITERS="$HOTSPOT_ITERS" \
HOTSPOT_TEMP_FILE="$HOTSPOT_TEMP_FILE" \
HOTSPOT_POWER_FILE="$HOTSPOT_POWER_FILE" \
HOTSPOT_OUTPUT_DIR="$HOTSPOT_OUTPUT_DIR" \
WARMUP_RUNS="$WARMUP_RUNS" \
MEASURED_RUNS="$MEASURED_RUNS" \
  sh "$MP_TESTS/run_cuda_h100_comparison.sh" > "$RAW_CSV"

awk -F, \
  -v bp_size="$BACKPROP_SIZE" \
  -v dl_size="$DENSE_LU_SIZE" \
  -v hs_rows="$HOTSPOT_ROWS" \
  -v hs_cols="$HOTSPOT_COLS" \
  -v hs_iters="$HOTSPOT_ITERS" \
  -v warmup_runs="$WARMUP_RUNS" \
  -v measured_runs="$MEASURED_RUNS" '
BEGIN {
  OFS = ","
}
NR == 1 {
  next
}
$2 == "double" {
  double_time[$1] = $3 + 0
  double_time_stddev[$1] = $4 + 0
  double_bytes[$1] = $8 + 0
}
{
  rows[++n] = $0
}
END {
  print "benchmark,case,precision,input_size,time_ms,time_ms_stddev,time_ms_min,time_ms_max,time_ms_runs,device_allocation_bytes,device_allocation_mib,speedup_vs_double,speedup_vs_double_stddev,time_ratio_vs_double,time_ratio_vs_double_stddev,memory_ratio_vs_double,warmup_runs,measured_runs"
  for (i = 1; i <= n; i++) {
    split(rows[i], a, ",")
    bench = a[1]
    case_name = a[2]
    time_ms = a[3] + 0
    time_stddev = a[4] + 0
    time_min = a[5] + 0
    time_max = a[6] + 0
    time_runs = a[7] + 0
    bytes = a[8] + 0
    mib = a[9] + 0
    precision = (case_name == "double") ? "double" : "mixed"
    if (bench == "backprop") {
      input_size = bp_size
    } else if (bench == "dense_lu") {
      input_size = dl_size
    } else {
      input_size = hs_rows "x" hs_cols "x" hs_iters
    }

    if (!(bench in double_time) || !(bench in double_bytes)) {
      printf("missing double baseline for %s\n", bench) > "/dev/stderr"
      exit 2
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

    printf "%s,%s,%s,%s,%.9g,%.9g,%.9g,%.9g,%d,%.0f,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%d,%d\n", \
      bench, case_name, precision, input_size, time_ms, time_stddev, time_min, time_max, time_runs, bytes, mib, speedup, speedup_stddev, time_ratio, time_ratio_stddev, memory_ratio, warmup_runs, measured_runs
  }
}
' "$RAW_CSV" > "$RATIO_CSV"

echo "Done."
echo "Raw CSV:   $RAW_CSV"
echo "Ratio CSV: $RATIO_CSV"
