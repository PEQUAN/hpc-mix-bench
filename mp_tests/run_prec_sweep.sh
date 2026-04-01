#!/bin/bash
# Usage:
#   ./run_prec_sweep.sh [-r] [-b] [-x] [-p] [folders...]
#   ./run_prec_sweep.sh [--remove] [--broadcast] [--execute] [--parallel] [folders...]
#
# Actions:
#   -r, --remove       Remove run_debug_1.sh ~ run_debug_4.sh from each subdirectory
#   -b, --broadcast    Copy run_settings/run_debug_1.sh ~ run_debug_4.sh into each subdirectory
#   -x, --execute      Run run_debug_1.sh ~ run_debug_4.sh inside each subdirectory
#                      Only runs in subdirectories containing promise.yml
#   -p, --parallel     Execute in parallel (default is sequential)
#
# Optional:
#   [folders...]       Specific folders to process; if none given, all subdirectories are processed
#
# Short options can be bundled: e.g., -rbxp
# ------------------------------------------------------------
# Author: Xinye Chen (xinyechenai@gmail.com)
# Last Updated: April 1, 2026
# ------------------------------------------------------------

REMOVE=false
BROADCAST=false
EXECUTE=false
PARALLEL=false

shopt -s nullglob

# Files to manage
SCRIPT_FILES=(
    "run_debug_1.sh"
    "run_debug_2.sh"
    "run_debug_3.sh"
    "run_debug_4.sh"
)

# ---------------------------------------
# Parse parameters (support bundled short options)
# ---------------------------------------
POSITIONAL=()
for arg in "$@"; do
    if [[ "$arg" == --* ]]; then
        case "$arg" in
            --remove) REMOVE=true ;;
            --broadcast) BROADCAST=true ;;
            --execute) EXECUTE=true ;;
            --parallel) PARALLEL=true ;;
            *) echo "Unknown option: $arg"; exit 1 ;;
        esac
    elif [[ "$arg" == -* ]]; then
        for (( i=1; i<${#arg}; i++ )); do
            c="${arg:i:1}"
            case "$c" in
                r) REMOVE=true ;;
                b) BROADCAST=true ;;
                x) EXECUTE=true ;;
                p) PARALLEL=true ;;
                *) echo "Unknown option: -$c"; exit 1 ;;
            esac
        done
    else
        POSITIONAL+=("$arg")
    fi
done

# Restore positional arguments
set -- "${POSITIONAL[@]}"

# Ensure at least one action is selected
if ! $REMOVE && ! $BROADCAST && ! $EXECUTE; then
    echo "No action selected. Choose -r, -b, -x or any combination."
    exit 1
fi

# Determine folders to process
if [ $# -gt 0 ]; then
    FOLDERS=("$@")
else
    FOLDERS=(*/)
fi

if [ ${#FOLDERS[@]} -eq 0 ]; then
    echo "No folders to process."
    exit 0
fi

# Check source scripts before broadcasting
if $BROADCAST; then
    for script in "${SCRIPT_FILES[@]}"; do
        PARENT_SCRIPT="../run_settings/$script"
        if [ ! -f "$PARENT_SCRIPT" ]; then
            echo "Error: $PARENT_SCRIPT not found. Cannot broadcast."
            exit 1
        fi
    done
fi

echo "Processing folders: ${FOLDERS[*]}"

PIDS=()

for d in "${FOLDERS[@]}"; do
    [ -d "$d" ] || { echo "Skipping $d: not a directory"; continue; }
    [[ "$d" == "run_settings/" ]] && continue

    # Step 1 — Remove
    if $REMOVE; then
        for script in "${SCRIPT_FILES[@]}"; do
            if [ -f "$d/$script" ]; then
                rm -f "$d/$script"
                echo "Removed $d/$script"
            else
                echo "Skipping remove in $d: $script not found"
            fi
        done
    fi

    # Step 2 — Broadcast
    if $BROADCAST; then
        for script in "${SCRIPT_FILES[@]}"; do
            cp "../run_settings/$script" "$d"
            echo "Broadcasted $script to $d"
        done
    fi

    # Step 3 — Execute
    if $EXECUTE; then
        if [ ! -f "$d/promise.yml" ]; then
            echo "Skipping $d: missing promise.yml"
            continue
        fi

        if $PARALLEL; then
            (
                cd "$d" || exit
                for script in "${SCRIPT_FILES[@]}"; do
                    if [ -f "$script" ]; then
                        chmod +x "$script"
                        echo "Executing $script in $(pwd)"
                        "./$script"
                    else
                        echo "Skipping $(pwd)/$script: file not found"
                    fi
                done
            ) &
            PIDS+=("$!")
            echo "Started execution in parallel for: $d"
        else
            echo "Executing sequentially in: $d"
            (
                cd "$d" || exit
                for script in "${SCRIPT_FILES[@]}"; do
                    if [ -f "$script" ]; then
                        chmod +x "$script"
                        echo "Executing $script in $(pwd)"
                        "./$script"
                    else
                        echo "Skipping $(pwd)/$script: file not found"
                    fi
                done
            )
        fi
    fi
done

# Wait for parallel jobs
if $EXECUTE && $PARALLEL && [ ${#PIDS[@]} -gt 0 ]; then
    echo "Waiting for all parallel executions to finish..."
    wait "${PIDS[@]}"
    echo "All executions completed."
fi

echo "Done."