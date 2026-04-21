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
JOBS=${JOBS:-$(nproc 2>/dev/null || echo 4)}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --parallel)
            PARALLEL=true
            shift
            ;;
        --jobs)
            [[ $# -ge 2 ]] || { echo "Error: --jobs requires an integer argument"; exit 1; }
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
    case "${1,,}" in
        1|true|y|yes) echo "true" ;;
        0|false|n|no) echo "false" ;;
        *)
            echo "Error: invalid boolean value: $1" >&2
            exit 1
            ;;
    esac
}

RUN_EXPERIMENTS=$(normalize_bool "$RUN_EXPERIMENTS")
RUN_PLOTTING=$(normalize_bool "$RUN_PLOTTING")
RUN_DEBUG=$(normalize_bool "$RUN_DEBUG")

if ! [[ "$JOBS" =~ ^[0-9]+$ ]] || (( JOBS <= 0 )); then
    echo "Error: --jobs must be a positive integer"
    exit 1
fi

export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}

# -----------------------------
# Discover valid folders
# -----------------------------
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
        if [[ ! -f "$folder/promise.yml" ]]; then
            echo "Skip $folder: missing promise.yml"
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

folder_to_logname() {
    local dir="$1"
    dir="${dir#./}"
    echo "${dir//\//__}"
}

run_setting_task() {
    local input_dir="$1"
    local script_path="$2"
    local dir name script_base i logfile

    dir=$(cd "$input_dir" && pwd)
    name=$(folder_to_logname "$input_dir")
    script_base=$(basename "$script_path")

    mkdir -p "logs/$name"

    if [[ ! "$script_base" =~ run_setting_([0-9]+)\.py ]]; then
        echo "[Skip] invalid script name: $script_base"
        return 0
    fi

    i="${BASH_REMATCH[1]}"
    logfile="logs/$name/run_${i}.log"

    echo "================================================"
    echo "Benchmark folder : $input_dir"
    echo "Task             : $script_base"
    echo "Log              : $logfile"
    echo "================================================"

    if [[ "$RUN_EXPERIMENTS" != "true" && "$RUN_PLOTTING" != "true" ]]; then
        echo "[Skip] setting phase skipped" | tee -a "$logfile"
        return 0
    fi

    if ! (
        cd "$dir"
        echo "[RUN] python3 $script_base $RUN_EXPERIMENTS $RUN_PLOTTING"
        python3 "$script_base" "$RUN_EXPERIMENTS" "$RUN_PLOTTING"
    ) >> "$logfile" 2>&1; then
        echo "[FAILED] $input_dir / $script_base"
        return 1
    fi

    echo "[DONE] $input_dir / $script_base"
}

run_debug_folder() {
    local input_dir="$1"
    local dir name debug_scripts=()
    local debug_script base i logfile

    dir=$(cd "$input_dir" && pwd)
    name=$(folder_to_logname "$input_dir")

    mkdir -p "logs/$name"

    shopt -s nullglob
    debug_scripts=("$dir"/run_debug_*.sh)
    shopt -u nullglob

    if (( ${#debug_scripts[@]} == 0 )); then
        echo "[Skip] no debug scripts in $input_dir"
        return 0
    fi

    mapfile -t debug_scripts < <(printf "%s\n" "${debug_scripts[@]}" | sort -V)

    echo "------------------------------------------------"
    echo "Debug folder     : $input_dir"
    echo "Mode             : folder-level parallel, in-folder sequential"
    echo "------------------------------------------------"

    for debug_script in "${debug_scripts[@]}"; do
        base=$(basename "$debug_script")

        if [[ ! "$base" =~ run_debug_([0-9]+)\.sh ]]; then
            echo "[Skip] invalid debug script name: $base"
            continue
        fi

        i="${BASH_REMATCH[1]}"
        logfile="logs/$name/run_${i}.log"

        echo "[RUN-DEBUG] $input_dir / $base"

        if ! (
            cd "$dir"
            chmod +x "$base"
            echo "[RUN] ./$base"
            "./$base"
        ) >> "$logfile" 2>&1; then
            echo "[FAILED] $input_dir / $base"
            return 1
        fi

        echo "[DONE-DEBUG] $input_dir / $base"
    done
}

export -f folder_to_logname
export -f run_setting_task
export -f run_debug_folder

export RUN_EXPERIMENTS
export RUN_PLOTTING
export RUN_DEBUG
export OMP_NUM_THREADS
export MKL_NUM_THREADS
export OPENBLAS_NUM_THREADS

SETTING_TASK_FILE=$(mktemp)
FOLDER_FILE=$(mktemp)

for dir in "${valid_folders[@]}"; do
    printf '%s\n' "$dir" >> "$FOLDER_FILE"

    shopt -s nullglob
    scripts=("$dir"/run_setting_*.py)
    shopt -u nullglob

    if (( ${#scripts[@]} > 0 )); then
        mapfile -t sorted_scripts < <(printf "%s\n" "${scripts[@]}" | sort -V)
        for script in "${sorted_scripts[@]}"; do
            printf '%s\t%s\n' "$dir" "$script" >> "$SETTING_TASK_FILE"
        done
    fi
done

START_TIME=$(date +%s)
START_HUMAN=$(date '+%Y-%m-%d %H:%M:%S')

echo "======================================"
echo "Run experiments : $RUN_EXPERIMENTS"
echo "Run plotting    : $RUN_PLOTTING"
echo "Run debug       : $RUN_DEBUG"
echo "Parallel        : $PARALLEL"
echo "Jobs            : $JOBS"
echo "======================================"

# -----------------------------
# Phase 1: run settings
# -----------------------------
if [[ "$RUN_EXPERIMENTS" == "true" || "$RUN_PLOTTING" == "true" ]]; then
    setting_count=$(wc -l < "$SETTING_TASK_FILE" | tr -d ' ')
    echo "Discovered $setting_count setting task(s)."

    if [[ "$PARALLEL" == "true" ]] && command -v parallel >/dev/null; then
        echo "Running setting tasks in parallel..."
        parallel -j "$JOBS" --colsep '\t' --lb run_setting_task {1} {2} :::: "$SETTING_TASK_FILE"
    else
        echo "Running setting tasks sequentially..."
        while IFS=$'\t' read -r dir script; do
            run_setting_task "$dir" "$script"
        done < "$SETTING_TASK_FILE"
    fi
else
    echo "Setting phase skipped."
fi

# -----------------------------
# Phase 2: run debug by folder
# -----------------------------
if [[ "$RUN_DEBUG" == "true" ]]; then
    folder_count=$(wc -l < "$FOLDER_FILE" | tr -d ' ')
    echo "Discovered $folder_count debug folder(s)."

    if [[ "$PARALLEL" == "true" ]] && command -v parallel >/dev/null; then
        echo "Running debug folders in parallel, one debug at a time per folder..."
        parallel -j "$JOBS" --lb run_debug_folder {} :::: "$FOLDER_FILE"
    else
        echo "Running debug folders sequentially..."
        while IFS= read -r dir; do
            run_debug_folder "$dir"
        done < "$FOLDER_FILE"
    fi
else
    echo "Debug phase skipped."
fi

END_TIME=$(date +%s)
END_HUMAN=$(date '+%Y-%m-%d %H:%M:%S')
ELAPSED=$((END_TIME - START_TIME))

HOURS=$((ELAPSED / 3600))
MINUTES=$(((ELAPSED % 3600) / 60))
SECONDS=$((ELAPSED % 60))

rm -f "$SETTING_TASK_FILE" "$FOLDER_FILE"

echo "======================================"
echo "All benchmark tasks finished."
echo "Logs in ./logs/"
echo "Started at      : $START_HUMAN"
echo "Finished at     : $END_HUMAN"
echo "Total elapsed   : ${HOURS}h ${MINUTES}m ${SECONDS}s"
echo "======================================"