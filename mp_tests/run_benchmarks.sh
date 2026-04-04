#!/usr/bin/env bash

# ------------------------------------------------------------
# Usage:
#   ./run_benchmarks.sh <run_exp> <run_plot> <run_debug> [folder1 folder2 ...] [--parallel] [--jobs N]
#
# Arguments:
#   • run_exp   : 1|true|y → run experiments
#   • run_plot  : 1|true|y → run plotting
#   • run_debug : 1|true|y → run matching run_debug_i.sh after run_setting_i.py
#
# Options:
#   • folders    : optional target folders; if omitted, auto-detect all valid folders
#   • --parallel : run tasks in parallel (requires GNU parallel)
#   • --jobs N   : number of parallel jobs/processes
#                  also supports: --jobs=N
#
# Folder requirements:
#   Each folder must contain:
#     - promise.yml
#     - run_setting_*.py
#
# Optional files:
#   • run_debug_{i}.sh
#       - Executed only if run_debug=true
#       - Matched to run_setting_{i}.py by index
#
# Notes:
#   • Boolean arguments accept: 1/0, true/false, yes/no (case-insensitive)
#   • Logs are saved to: logs/<folder>/run_<i>.log
#   • Missing run_debug_i.sh will be skipped gracefully
#   • If --jobs is not provided, default is:
#         JOBS environment variable, otherwise nproc, otherwise 4
# ------------------------------------------------------------
# Author: Xinye Chen (xinyechenai@gmail.com)
# Last Updated: November 18, 2025

# ---------- 1. Parse arguments ----------

set -euo pipefail


RUN_EXPERIMENTS=${1:-true}
RUN_PLOTTING=${2:-true}
RUN_DEBUG=${3:-false}

shift 3 || true

PARALLEL=false
TARGET_FOLDERS=()

# default jobs: use env JOBS if set, otherwise nproc, fallback 4
JOBS=${JOBS:-$(nproc 2>/dev/null || echo 4)}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --parallel)
            PARALLEL=true
            shift
            ;;
        --jobs)
            if [[ $# -lt 2 ]]; then
                echo "Error: --jobs requires an integer argument"
                exit 1
            fi
            JOBS="$2"
            shift 2
            ;;
        --jobs=*)
            JOBS="${1#*=}"
            shift
            ;;
        *)
            TARGET_FOLDERS+=("$1")
            shift
            ;;
    esac
done

normalize_bool() {
    case "$1" in
        1|true|True|TRUE|y|Y|yes|Yes|YES) echo "true" ;;
        0|false|False|FALSE|n|N|no|No|NO) echo "false" ;;
        *) echo "true" ;;
    esac
}

RUN_EXPERIMENTS=$(normalize_bool "$RUN_EXPERIMENTS")
RUN_PLOTTING=$(normalize_bool "$RUN_PLOTTING")
RUN_DEBUG=$(normalize_bool "$RUN_DEBUG")

# validate jobs
if ! [[ "$JOBS" =~ ^[0-9]+$ ]] || (( JOBS <= 0 )); then
    echo "Error: --jobs must be a positive integer"
    exit 1
fi

# avoid nested oversubscription from math libraries
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}

# ------------------------------------------------------------
# Resume detection
# ------------------------------------------------------------
experiment_already_done() {
    dir="$1"

    shopt -s nullglob
    csvs=("$dir"/runtimes*.csv)
    shopt -u nullglob

    (( ${#csvs[@]} > 0 ))
}

# ------------------------------------------------------------
# Run one task = run_setting_i.py + optional run_debug_i.sh
# ------------------------------------------------------------
run_task() {
    input_dir="$1"
    script_path="$2"

    dir=$(cd "$input_dir" && pwd)
    name=$(basename "$dir")
    script_base=$(basename "$script_path")

    mkdir -p logs
    mkdir -p "logs/$name"

    if [[ ! "$script_base" =~ run_setting_([0-9]+)\.py ]]; then
        echo "[Skip] invalid script name: $script_base"
        return
    fi

    i="${BASH_REMATCH[1]}"
    debug_script="run_debug_${i}.sh"
    logfile="logs/$name/run_${i}.log"

    echo "================================================"
    echo "Benchmark folder : $name"
    echo "Task             : $script_base"
    echo "Log              : $logfile"
    echo "================================================"

    if [[ ! -f "$dir/promise.yml" ]]; then
        echo "[Skip] missing promise.yml in $dir"
        return
    fi

    if [[ "$RUN_EXPERIMENTS" == "true" ]]; then
        if experiment_already_done "$dir"; then
            echo "[Resume] experiment results already exist in $name"
        fi
    fi

    task_start=$(date +%s)

    (
        cd "$dir"
        echo "[RUN] python3 $script_base $RUN_EXPERIMENTS $RUN_PLOTTING"
        python3 "$script_base" "$RUN_EXPERIMENTS" "$RUN_PLOTTING"
    ) >> "$logfile" 2>&1

    if (( $? != 0 )); then
        echo "[FAILED] $name / $script_base"
        return 1
    fi

    if [[ "$RUN_DEBUG" == "true" ]]; then
        if [[ -f "$dir/$debug_script" ]]; then
            (
                cd "$dir"
                chmod +x "$debug_script"
                echo "[RUN] ./$debug_script"
                "./$debug_script"
            ) >> "$logfile" 2>&1

            if (( $? != 0 )); then
                echo "[FAILED] $name / $debug_script"
                return 1
            fi
        else
            echo "[Skip] missing $debug_script in $name" | tee -a "$logfile"
        fi
    fi

    task_end=$(date +%s)
    elapsed=$((task_end - task_start))
    h=$((elapsed / 3600))
    m=$(((elapsed % 3600) / 60))
    s=$((elapsed % 60))

    echo "[DONE] $name / $script_base (${h}h ${m}m ${s}s)"
}

export -f run_task
export -f experiment_already_done

export RUN_EXPERIMENTS
export RUN_PLOTTING
export RUN_DEBUG
export OMP_NUM_THREADS
export MKL_NUM_THREADS
export OPENBLAS_NUM_THREADS

# ------------------------------------------------------------
# Banner
# ------------------------------------------------------------
echo "======================================"
echo "Run experiments : $RUN_EXPERIMENTS"
echo "Run plotting    : $RUN_PLOTTING"
echo "Run debug       : $RUN_DEBUG"
echo "Parallel        : $PARALLEL"
echo "Jobs            : $JOBS"
echo "OMP threads     : $OMP_NUM_THREADS"
echo "MKL threads     : $MKL_NUM_THREADS"
echo "OPENBLAS thrds  : $OPENBLAS_NUM_THREADS"

if (( ${#TARGET_FOLDERS[@]} == 0 )); then
    echo "Folders         : auto detect"
else
    echo "Folders         : ${TARGET_FOLDERS[*]}"
fi
echo "======================================"

# ------------------------------------------------------------
# Discover valid folders
# ------------------------------------------------------------
valid_folders=()

if (( ${#TARGET_FOLDERS[@]} == 0 )); then
    while IFS= read -r script; do
        dir=$(dirname "$script")
        if [[ -f "$dir/promise.yml" ]]; then
            valid_folders+=("$dir")
        fi
    done < <(find . -maxdepth 2 -type f -name "run_setting_*.py")
else
    for folder in "${TARGET_FOLDERS[@]}"; do
        if [[ ! -d "$folder" ]]; then
            echo "Invalid folder: $folder"
            continue
        fi
        valid_folders+=("$folder")
    done
fi

mapfile -t valid_folders < <(printf "%s\n" "${valid_folders[@]}" | sort -u)

if (( ${#valid_folders[@]} == 0 )); then
    echo "No valid benchmarks found."
    exit 0
fi

# ------------------------------------------------------------
# Build task list: one line = folder + script
# ------------------------------------------------------------
TASK_FILE=$(mktemp)

for dir in "${valid_folders[@]}"; do
    [[ -f "$dir/promise.yml" ]] || continue

    shopt -s nullglob
    scripts=("$dir"/run_setting_*.py)
    shopt -u nullglob

    if (( ${#scripts[@]} == 0 )); then
        continue
    fi

    mapfile -t sorted_scripts < <(printf "%s\n" "${scripts[@]}" | sort -V)

    for script in "${sorted_scripts[@]}"; do
        printf '%s\t%s\n' "$dir" "$script" >> "$TASK_FILE"
    done
done

TASK_COUNT=$(wc -l < "$TASK_FILE" | tr -d ' ')

if (( TASK_COUNT == 0 )); then
    rm -f "$TASK_FILE"
    echo "No runnable tasks found."
    exit 0
fi

echo "Discovered $TASK_COUNT task(s)."

START_TIME=$(date +%s)
START_HUMAN=$(date '+%Y-%m-%d %H:%M:%S')

# ------------------------------------------------------------
# Run tasks
# ------------------------------------------------------------
if [[ "$PARALLEL" == "true" ]] && command -v parallel >/dev/null; then
    echo "Running $TASK_COUNT tasks in parallel..."
    parallel -j "$JOBS" --colsep '\t' --lb run_task {1} {2} :::: "$TASK_FILE"
else
    echo "Running sequentially..."
    while IFS=$'\t' read -r dir script; do
        run_task "$dir" "$script"
    done < "$TASK_FILE"
fi

END_TIME=$(date +%s)
END_HUMAN=$(date '+%Y-%m-%d %H:%M:%S')
ELAPSED=$((END_TIME - START_TIME))

HOURS=$((ELAPSED / 3600))
MINUTES=$(((ELAPSED % 3600) / 60))
SECONDS=$((ELAPSED % 60))

rm -f "$TASK_FILE"

echo "======================================"
echo "All benchmark tasks finished."
echo "Logs in ./logs/"
echo "Started at      : $START_HUMAN"
echo "Finished at     : $END_HUMAN"
echo "Total elapsed   : ${HOURS}h ${MINUTES}m ${SECONDS}s"
echo "======================================"