#!/bin/bash

# ------------------------------------------------------------
# Usage:
#   ./run_plots.sh <run_exp> <run_plot> [folder1 folder2 ...]
#
#   • run_exp: 1|true|y → run experiments
#   • run_plot: 1|true|y → run plots
#   • folders: optional, if none → all valid folders
#
#   Each folder must contain:
#     - plot_1.py to plot_4.py
#     - precision_settings_1.json
#     - promise.yml
#
#   Uses GNU parallel to run folders concurrently.
# ------------------------------------------------------------

# ---------- 1. Check for parallel ----------
if ! command -v parallel >/dev/null; then
    echo "Error: GNU parallel is not installed."
    echo "Install it with: sudo apt install parallel (Linux) or brew install parallel (MacOS)"
    exit 1
fi

# ---------- 2. Parse arguments ----------
RUN_EXPERIMENTS=${1:-true}
RUN_PLOTTING=${2:-true}
shift 2
TARGET_FOLDERS=("$@")

# ---------- 3. Normalize booleans ----------
normalize_bool() {
    case "$1" in
        1|true|True|TRUE|y|Y|yes|Yes|YES|t|T) echo "true" ;;
        0|false|False|FALSE|n|N|no|No|NO|f|F) echo "false" ;;
        *) echo "true" ;;
    esac
}
RUN_EXPERIMENTS=$(normalize_bool "$RUN_EXPERIMENTS")
RUN_PLOTTING=$(normalize_bool "$RUN_PLOTTING")

# ---------- 4. Helper: run one folder ----------
run_folder() {
    local dir="$1"
    local abs_dir=$(realpath "$dir") 2>/dev/null || {
        echo "Error: Cannot resolve path for '$dir'. Skipping."
        return 1
    }

    echo "=== Processing folder: $abs_dir ==="

    # Check required files
    local missing=()
    for file in plot_{1,2,3,4}.py precision_settings_1.json promise.yml; do
        [[ -f "$dir/$file" ]] || missing+=("$file")
    done

    if (( ${#missing[@]} > 0 )); then
        echo "  [Missing files] ${missing[*]}"
        echo "  Skipping $dir"
        echo
        return 1
    fi

    # Run scripts sequentially INSIDE folder
    for script in plot_{1,2,3,4}.py; do
        echo "  → Running: $script"
        (
            cd "$dir" || { echo "  [Failed] Cannot cd to $dir"; exit 1; }
            python3 "$script" "$RUN_EXPERIMENTS" "$RUN_PLOTTING"
        )
        if (( $? != 0 )); then
            echo "  [Failed] $script"
        else
            echo "  [Success] $script"
        fi
    done
    echo
}

# Export function and variables for parallel
export -f run_folder
export RUN_EXPERIMENTS RUN_PLOTTING

# ---------- 5. Main logic ----------
echo "=========================================="
echo "Run experiments : $RUN_EXPERIMENTS"
echo "Run plotting    : $RUN_PLOTTING"
if (( ${#TARGET_FOLDERS[@]} == 0 )); then
    echo "Target folders  : ALL matching folders"
else
    echo "Target folders  : ${TARGET_FOLDERS[*]}"
fi
echo "Running folders in parallel (jobs: $(nproc))"
echo "=========================================="

# ---------- 6. Run folders in parallel ----------
if (( ${#TARGET_FOLDERS[@]} == 0 )); then
    # Find all valid folders
    valid_folders=()
    while IFS= read -r plot1; do
        dir=$(dirname "$plot1")
        if [[ -f "$dir/plot_2.py" && -f "$dir/plot_3.py" && -f "$dir/plot_4.py" && \
              -f "$dir/precision_settings_1.json" && -f "$dir/promise.yml" ]]; then
            valid_folders+=("$dir")
        fi
    done < <(find . -maxdepth 2 -type f -name "plot_1.py")

    if (( ${#valid_folders[@]} == 0 )); then
        echo "Warning: No complete folder found (missing files)."
    else
        # Run folders in parallel
        printf "%s\n" "${valid_folders[@]}" | parallel -j "$(nproc)" run_folder
    fi
else
    # Run only specified folders in parallel
    valid_folders=()
    for folder in "${TARGET_FOLDERS[@]}"; do
        if [[ ! -d "$folder" ]]; then
            echo "Error: '$folder' is not a directory. Skipping."
            continue
        fi
        if [[ -f "$folder/plot_1.py" && -f "$folder/plot_2.py" && \
              -f "$folder/plot_3.py" && -f "$folder/plot_4.py" && \
              -f "$folder/precision_settings_1.json" && -f "$folder/promise.yml" ]]; then
            valid_folders+=("$folder")
        else
            echo "Error: '$folder' missing required files. Skipping."
        fi
    done

    if (( ${#valid_folders[@]} > 0 )); then
        printf "%s\n" "${valid_folders[@]}" | parallel -j "$(nproc)" run_folder
    else
        echo "Error: No valid folders specified."
    fi
fi

echo "=========================================="
echo "All done!"