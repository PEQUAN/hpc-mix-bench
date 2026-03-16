#!/usr/bin/env bash

# ------------------------------------------------------------
# Usage:
#   ./run_benchmarks.sh <run_exp> <run_plot> [folder1 folder2 ...] [--parallel]
#
#   • run_exp: 1|true|y → run experiments
#   • run_plot: 1|true|y → run plots
#   • folders: optional, if none → all valid folders
#   • --parallel: (optional) run folders in parallel (requires GNU parallel)
#
#   Folder must contain:
#     - run_setting_*.py (any number, any index)
#     - promise.yml
#     - For plotting:
#           matching pairs: prec_setting_{i}.json + runtimes{i}.csv
# ------------------------------------------------------------
# Author: Xinye Chen (xinyechenai@gmail.com)
# Last Updated: November 18, 2025

# ---------- 1. Parse arguments ----------

set -euo pipefail


RUN_EXPERIMENTS=${1:-true}
RUN_PLOTTING=${2:-true}

shift 2 || true

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


    # resume check

    if [[ "$RUN_EXPERIMENTS" == "true" ]]; then

        if experiment_already_done "$dir"; then
            echo "[Resume] experiment results already exist"
        fi

    fi


    for script in "${scripts[@]}"; do

        echo "→ $(basename "$script")"

        (
            cd "$dir"

            python3 "$script" "$RUN_EXPERIMENTS" "$RUN_PLOTTING"

        ) >> "$LOGFILE" 2>&1

        if (( $? != 0 )); then
            echo "[FAILED] $(basename "$script")"
            return
        fi

    done

    echo "[DONE] $name"

}



export -f run_folder
export -f check_plot_pairs
export -f experiment_already_done

export RUN_EXPERIMENTS
export RUN_PLOTTING



# ------------------------------------------------------------
# Banner
# ------------------------------------------------------------

echo "======================================"
echo "Run experiments : $RUN_EXPERIMENTS"
echo "Run plotting    : $RUN_PLOTTING"
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


echo "======================================"
echo "All benchmarks finished."
echo "Logs in ./logs/"
echo "======================================"