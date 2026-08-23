#!/bin/bash
#SBATCH --job-name=hpcmix-comp
##SBATCH --account=YOUR_PROJECT@h100
#SBATCH -C h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --hint=nomultithread
#SBATCH --qos=qos_gpu_h100-t3
#SBATCH --time=20:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail

# Submit from hpc-mix-bench/papers with:
#   sbatch --account=${IDRPROJ}@h100 \
#     --export=ALL,REPO_DIR="$(pwd)/..",WARMUP_RUNS=1,MEASURED_RUNS=5 \
#     complement/submit_complement_h100.sh

REPO_DIR="${REPO_DIR:-}"
COMBINATIONS="${COMBINATIONS:-1 2}"
CUDA_ARCH="${CUDA_ARCH:-sm_90}"
WARMUP_RUNS="${WARMUP_RUNS:-1}"
MEASURED_RUNS="${MEASURED_RUNS:-5}"
FORCE_REBUILD="${FORCE_REBUILD:-1}"
BACKPROP_SIZE="${BACKPROP_SIZE:-65536}"
DENSE_LU_SIZE="${DENSE_LU_SIZE:-5000}"
HOTSPOT_ROWS="${HOTSPOT_ROWS:-1024}"
HOTSPOT_COLS="${HOTSPOT_COLS:-1024}"
HOTSPOT_ITERS="${HOTSPOT_ITERS:-2}"
HOTSPOT_TEMP_FILE="${HOTSPOT_TEMP_FILE:-}"
HOTSPOT_POWER_FILE="${HOTSPOT_POWER_FILE:-}"
HOTSPOT_GENERATE_INPUTS="${HOTSPOT_GENERATE_INPUTS:-1}"
PLOT_FONT_SIZE="${PLOT_FONT_SIZE:-12}"
RUN_PLOTS="${RUN_PLOTS:-1}"

if [[ -z "$REPO_DIR" ]]; then
  for candidate in "$SLURM_SUBMIT_DIR/.." "$SLURM_SUBMIT_DIR" "$PWD/.." "$PWD"; do
    if [[ -d "$candidate/papers/complement" ]]; then
      REPO_DIR="$candidate"
      break
    fi
  done
fi

if [[ -z "$REPO_DIR" || ! -d "$REPO_DIR/papers/complement" ]]; then
  echo "ERROR: could not find hpc-mix-bench/papers/complement." >&2
  echo "Submit from papers with REPO_DIR=\"\$(pwd)/..\"." >&2
  exit 1
fi

PAPERS_DIR="$REPO_DIR/papers"
OUT_DIR="${OUT_DIR:-$PAPERS_DIR/complement/results/${SLURM_JOB_ID:-manual}}"
mkdir -p "$OUT_DIR"

echo "job_id=${SLURM_JOB_ID:-manual}"
echo "host=$(hostname)"
echo "repo_dir=$REPO_DIR"
echo "papers_dir=$PAPERS_DIR"
echo "out_dir=$OUT_DIR"
echo "combinations=$COMBINATIONS"
echo "cuda_arch=$CUDA_ARCH"
echo "backprop_size=$BACKPROP_SIZE"
echo "dense_lu_size=$DENSE_LU_SIZE"
echo "hotspot_rows=$HOTSPOT_ROWS"
echo "hotspot_cols=$HOTSPOT_COLS"
echo "hotspot_iters=$HOTSPOT_ITERS"
echo "hotspot_temp_file=$HOTSPOT_TEMP_FILE"
echo "hotspot_power_file=$HOTSPOT_POWER_FILE"
echo "hotspot_generate_inputs=$HOTSPOT_GENERATE_INPUTS"
echo "warmup_runs=$WARMUP_RUNS"
echo "measured_runs=$MEASURED_RUNS"
echo "plot_font_size=$PLOT_FONT_SIZE"
echo "run_plots=$RUN_PLOTS"

module purge
module load arch/h100
module load cuda/12.4.1 || module load cuda/12.2.0 || module load cuda
module list

which nvcc
nvcc --version
nvidia-smi

cd "$PAPERS_DIR"

OUT_DIR="$OUT_DIR" \
COMBINATIONS="$COMBINATIONS" \
CUDA_ARCH="$CUDA_ARCH" \
WARMUP_RUNS="$WARMUP_RUNS" \
MEASURED_RUNS="$MEASURED_RUNS" \
BACKPROP_SIZE="$BACKPROP_SIZE" \
DENSE_LU_SIZE="$DENSE_LU_SIZE" \
HOTSPOT_ROWS="$HOTSPOT_ROWS" \
HOTSPOT_COLS="$HOTSPOT_COLS" \
HOTSPOT_ITERS="$HOTSPOT_ITERS" \
HOTSPOT_TEMP_FILE="$HOTSPOT_TEMP_FILE" \
HOTSPOT_POWER_FILE="$HOTSPOT_POWER_FILE" \
HOTSPOT_GENERATE_INPUTS="$HOTSPOT_GENERATE_INPUTS" \
PLOT_FONT_SIZE="$PLOT_FONT_SIZE" \
RUN_PLOTS="$RUN_PLOTS" \
FORCE_REBUILD="$FORCE_REBUILD" \
  bash "$PAPERS_DIR/complement/run_complement_h100.sh"
