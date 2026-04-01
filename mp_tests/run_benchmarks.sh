#!/usr/bin/env bash
# ------------------------------------------------------------
# Usage:
#   ./run_benchmarks.sh <run_exp> <run_plot> <run_debug> [folder1 folder2 ...] [--parallel]
#
# Arguments:
#   • run_exp   : 1|true|y → run experiments (execute run_setting_*.py)
#   • run_plot  : 1|true|y → enable plotting inside run_setting_*.py
#   • run_debug : 1|true|y → after each run_setting_i.py, run run_debug_i.sh (if exists)
#
# Options:
#   • folders    : optional list of target folders
#                  if none provided → auto-detect all valid folders
#   • --parallel : (optional) run folders in parallel (requires GNU parallel)
#
# Folder requirements:
#   Each folder must contain:
#     - promise.yml
#     - run_setting_*.py (any number, matched by index i)
#
# Optional files:
#   • run_debug_{i}.sh
#       - Executed only if run_debug=true
#       - Matched to run_setting_{i}.py by index
#
#   • For plotting:
#       - prec_setting_{i}.json
#       - runtimes{i}.csv
#       (must exist as matching pairs)
#
# Execution behavior:
#   For each folder:
#     run_setting_1.py → (optional) run_debug_1.sh
#     run_setting_2.py → (optional) run_debug_2.sh
#     ...
#
# Notes:
#   • Boolean arguments accept: 1/0, true/false, yes/no (case-insensitive)
#   • Logs are saved to: logs/<folder>.log
#   • Folders without promise.yml are skipped
#   • Missing run_debug_i.sh will be skipped gracefully
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

for arg in "$@"; do
    if [[ "$arg" == "--parallel" ]]; then
        PARALLEL=true
    else
        TARGET_FOLDERS+=("$arg")
    fi
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

# parallel jobs
JOBS=${JOBS:-$(nproc 2>/dev/null || echo 4)}

# ------------------------------------------------------------
# Helper: detect plotting data
# ------------------------------------------------------------

check_plot_pairs() {
    dir="$1"

    shopt -s nullglob
    prec_files=("$dir"/prec_setting_*.json)
    shopt -u nullglob

    for prec in "${prec_files[@]}"; do
        base=$(basename "$prec")

        if [[ $base =~ prec_setting_([0-9]+)\.json ]]; then
            i="${BASH_REMATCH[1]}"

            if [[ -f "$dir/runtimes${i}.csv" ]]; then
                return 0
            fi
        fi
    done

    return 1
}

# ------------------------------------------------------------
# Resume detection
# ------------------------------------------------------------

experiment_already_done() {
    dir="$1"

    shopt -s nullglob
    csvs=("$dir"/runtimes*.csv)
    shopt -u nullglob

    if (( ${#csvs[@]} > 0 )); then
        return 0
    else
        return 1
    fi
}

# ------------------------------------------------------------
# Run benchmark folder
# ------------------------------------------------------------

run_folder() {
    input="$1"
    dir=$(cd "$input" && pwd)
    name=$(basename "$dir")

    mkdir -p logs
    LOGFILE="logs/${name}.log"

    echo "================================================"
    echo "Running benchmark: $name"
    echo "Log: $LOGFILE"
    echo "================================================"

    if [[ ! -f "$dir/promise.yml" ]]; then
        echo "[Skip] missing promise.yml"
        return
    fi

    shopt -s nullglob
    scripts=("$dir"/run_setting_*.py)
    shopt -u nullglob

    if (( ${#scripts[@]} == 0 )); then
        echo "[Skip] no run_setting_*.py"
        return
    fi

    IFS=$'\n' scripts=($(printf "%s\n" "${scripts[@]}" | sort -V))

    if [[ "$RUN_EXPERIMENTS" == "true" ]]; then
        if experiment_already_done "$dir"; then
            echo "[Resume] experiment results already exist"
        fi
    fi

    for script in "${scripts[@]}"; do
        script_base=$(basename "$script")
        echo "→ $script_base"

        (
            cd "$dir"
            python3 "$script_base" "$RUN_EXPERIMENTS" "$RUN_PLOTTING"
        ) >> "$LOGFILE" 2>&1

        if (( $? != 0 )); then
            echo "[FAILED] $script_base"
            return
        fi

        if [[ "$RUN_DEBUG" == "true" ]]; then
            if [[ "$script_base" =~ run_setting_([0-9]+)\.py ]]; then
                i="${BASH_REMATCH[1]}"
                debug_script="run_debug_${i}.sh"

                if [[ -f "$dir/$debug_script" ]]; then
                    echo "→ $debug_script"

                    (
                        cd "$dir"
                        chmod +x "$debug_script"
                        "./$debug_script"
                    ) >> "$LOGFILE" 2>&1

                    if (( $? != 0 )); then
                        echo "[FAILED] $debug_script"
                        return
                    fi
                else
                    echo "[Skip] missing $debug_script" | tee -a "$LOGFILE"
                fi
            fi
        fi
    done

    echo "[DONE] $name"
}

export -f run_folder
export -f check_plot_pairs
export -f experiment_already_done

export RUN_EXPERIMENTS
export RUN_PLOTTING
export RUN_DEBUG

# ------------------------------------------------------------
# Banner
# ------------------------------------------------------------

echo "======================================"
echo "Run experiments : $RUN_EXPERIMENTS"
echo "Run plotting    : $RUN_PLOTTING"
echo "Run debug       : $RUN_DEBUG"
echo "Parallel        : $PARALLEL"
echo "Jobs            : $JOBS"

if (( ${#TARGET_FOLDERS[@]} == 0 )); then
    echo "Folders         : auto detect"
else
    echo "Folders         : ${TARGET_FOLDERS[*]}"
fi

echo "======================================"

# ------------------------------------------------------------
# Folder discovery
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
# Global timing
# ------------------------------------------------------------

START_TIME=$(date +%s)
START_HUMAN=$(date '+%Y-%m-%d %H:%M:%S')

# ------------------------------------------------------------
# Run
# ------------------------------------------------------------

if [[ "$PARALLEL" == "true" ]] && command -v parallel >/dev/null; then
    echo "Running ${#valid_folders[@]} benchmarks in parallel..."
    printf "%s\n" "${valid_folders[@]}" | parallel -j "$JOBS" --lb run_folder {}
else
    echo "Running sequentially..."
    for f in "${valid_folders[@]}"; do
        run_folder "$f"
    done
fi

END_TIME=$(date +%s)
END_HUMAN=$(date '+%Y-%m-%d %H:%M:%S')
ELAPSED=$((END_TIME - START_TIME))

HOURS=$((ELAPSED / 3600))
MINUTES=$(((ELAPSED % 3600) / 60))
SECONDS=$((ELAPSED % 60))

echo "======================================"
echo "All benchmarks finished."
echo "Logs in ./logs/"
echo "Started at      : $START_HUMAN"
echo "Finished at     : $END_HUMAN"
echo "Total elapsed   : ${HOURS}h ${MINUTES}m ${SECONDS}s"
echo "======================================"