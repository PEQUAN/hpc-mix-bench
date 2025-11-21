#!/bin/bash

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
RUN_EXPERIMENTS=${1:-true}
RUN_PLOTTING=${2:-true}
shift 2

PARALLEL=false
TARGET_FOLDERS=()
for arg in "$@"; do
    if [[ "$arg" == "--parallel" ]]; then
        PARALLEL=true
    else
        TARGET_FOLDERS+=("$arg")
    fi
done

# ---------- 2. Normalize booleans ----------
normalize_bool() {
    case "$1" in
        1|true|True|TRUE|y|Y|yes|Yes|YES|t|T) echo "true" ;;
        0|false|False|FALSE|n|N|no|No|NO|f|F) echo "false" ;;
        *) echo "true" ;;
    esac
}
RUN_EXPERIMENTS=$(normalize_bool "$RUN_EXPERIMENTS")
RUN_PLOTTING=$(normalize_bool "$RUN_PLOTTING")

# ---------- 3. Helper: detect at least one valid plotting pair ----------
check_plot_pairs() {
    local dir="$1"

    shopt -s nullglob
    local prec_files=("$dir"/prec_setting_*.json)
    local runtime_files=("$dir"/runtimes*.csv)
    shopt -u nullglob

    local matched_pairs=()

    for prec in "${prec_files[@]}"; do
        local base=$(basename "$prec")
        if [[ $base =~ prec_setting_([0-9]+)\.json ]]; then
            local i="${BASH_REMATCH[1]}"
            if [[ -f "$dir/runtimes${i}.csv" ]]; then
                matched_pairs+=("$i")
            fi
        fi
    done

    if (( ${#matched_pairs[@]} > 0 )); then
        return 0  # success
    else
        return 1  # failure
    fi
}

# ---------- 4. Run one folder ----------
run_folder() {
    local input_dir="$1"
    local dir=$(realpath "$input_dir")

    echo "=== Processing folder: $dir ==="

    missing_data=()

    # Always require promise.yml
    [[ -f "$dir/promise.yml" ]] || missing_data+=("promise.yml")

    # Plotting requires matching pairs
    if [[ "$RUN_PLOTTING" == "true" ]]; then
        if ! check_plot_pairs "$dir"; then
            missing_data+=("prec_setting_*.json + runtimes*.csv (matching index required)")
        fi
    fi

    # If missing something → skip
    if (( ${#missing_data[@]} > 0 )); then
        echo "  [Missing required files] ${missing_data[*]}"
        echo "  Skipping $dir"
        echo
        return 1
    fi

    # Find all run_setting_*.py
    shopt -s nullglob
    scripts=("$dir"/run_setting_*.py)
    shopt -u nullglob

    if (( ${#scripts[@]} == 0 )); then
        echo "  [No run_setting_*.py files found]"
        echo "  Skipping $dir"
        echo
        return 1
    fi

    # Sort numerically (script index)
    IFS=$'\n' scripts=($(printf "%s\n" "${scripts[@]}" | sort -V))

    for script in "${scripts[@]}"; do
        echo "  → Running: $(basename "$script")"
        (
            cd "$dir"
            python3 "$script" "$RUN_EXPERIMENTS" "$RUN_PLOTTING"
        )
        if (( $? != 0 )); then
            echo "  [Failed] $(basename "$script")"
        else
            echo "  [Success] $(basename "$script")"
        fi
    done

    echo
}

# ---------- 5. Export for parallel ----------
export -f normalize_bool check_plot_pairs run_folder
export RUN_EXPERIMENTS RUN_PLOTTING

# ---------- 6. Main banner ----------
echo "=========================================="
echo "Run experiments : $RUN_EXPERIMENTS"
echo "Run plotting    : $RUN_PLOTTING"
echo "Parallel mode   : $PARALLEL"

if (( ${#TARGET_FOLDERS[@]} == 0 )); then
    echo "Target folders  : ALL valid folders"
else
    echo "Target folders  : ${TARGET_FOLDERS[*]}"
fi

echo "=========================================="

# ---------- 7. Auto-discovery mode ----------
if (( ${#TARGET_FOLDERS[@]} == 0 )); then
    valid_folders=()

    # detect directories containing run_setting_*.py
    while IFS= read -r script; do
        dir=$(dirname "$script")
        dir=$(realpath "$dir")

        # must have promise.yml
        [[ -f "$dir/promise.yml" ]] || continue

        # must have valid plotting pairs (if plotting)
        if [[ "$RUN_PLOTTING" == "true" ]]; then
            check_plot_pairs "$dir" || continue
        fi

        valid_folders+=("$dir")
    done < <(find . -maxdepth 2 -type f -name "run_setting_*.py")

    # remove duplicates
    mapfile -t valid_folders < <(printf "%s\n" "${valid_folders[@]}" | sort -u)

    if (( ${#valid_folders[@]} == 0 )); then
        echo "Warning: No valid folders found."
    else
        if [[ "$PARALLEL" == "true" ]] && command -v parallel >/dev/null 2>&1; then
            echo "Running ${#valid_folders[@]} folders in parallel (max 4 jobs)..."
            printf "%s\n" "${valid_folders[@]}" | parallel -j 4 run_folder {}
        else
            echo "Running ${#valid_folders[@]} folders sequentially..."
            for dir in "${valid_folders[@]}"; do run_folder "$dir"; done
        fi
    fi

# ---------- 8. Manual folder selection ----------
else
    valid_folders=()

    for folder in "${TARGET_FOLDERS[@]}"; do
        [[ -d "$folder" ]] || { echo "Error: '$folder' is not a directory."; continue; }

        # require run_setting_*.py
        shopt -s nullglob
        scripts=("$folder"/run_setting_*.py)
        shopt -u nullglob
        if (( ${#scripts[@]} == 0 )); then
            echo "Error: '$folder' missing run_setting_*.py. Skipping."
            continue
        fi

        # require promise.yml
        [[ -f "$folder/promise.yml" ]] || { echo "Error: '$folder' missing promise.yml. Skipping."; continue; }

        # require matching plot pairs if plotting
        if [[ "$RUN_PLOTTING" == "true" ]]; then
            if ! check_plot_pairs "$folder"; then
                echo "Error: '$folder' missing matching prec_setting_*.json + runtimes*.csv. Skipping."
                continue
            fi
        fi

        valid_folders+=("$folder")
    done

    # run validated folders
    if (( ${#valid_folders[@]} == 0 )); then
        echo "No valid folders specified."
    else
        if [[ "$PARALLEL" == "true" ]] && command -v parallel >/dev/null 2>&1; then
            echo "Running ${#valid_folders[@]} folders in parallel (max 4 jobs)..."
            printf "%s\n" "${valid_folders[@]}" | parallel -j 4 run_folder {}
        else
            echo "Running ${#valid_folders[@]} folders sequentially..."
            for dir in "${valid_folders[@]}"; do run_folder "$dir"; done
        fi
    fi
fi

echo "=========================================="
echo "All done!"
