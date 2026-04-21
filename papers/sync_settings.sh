#!/usr/bin/env bash


# There is a folder named run_settings in the parent directory containing all run_setting_*.py files
# This script will delete all run_setting_*.py files in each subfolder under the current directory
# Then copy all run_setting_*.py files from ../run_settings/ to each subfolder

# Overview:
# This Bash script manages `run_setting_*.py` files across multiple subfolders in the current working directory.
# It performs two main steps:
# 1. Delete Step: Deletes all files matching `run_setting_*.py` in each subfolder under the current directory.
# 2. Copy Step: Copies `run_setting_*.py` files from the parent directory's `run_settings/` folder to each subfolder,
#    ensuring all subfolders use the same configuration files.
#
# The script is useful for automating the synchronization of experiment or run settings files across multiple folders,
# such as in machine learning experiments or batch tasks.
#
# Prerequisites:
# - Run in the current working directory containing multiple subfolders (targets).
# - Parent directory (`..`) must have a `run_settings` folder with `run_setting_*.py` files (for copy step).
# - Uses `find` and `cp` commands (standard on Unix-like systems).
#
# Usage:
# bash sync_settings.sh [options]
#
# Options:
#   --delete or -d: Execute Step 1 (delete files). Default: enabled if no options.
#   --broadcast or -b: Execute Step 2 (copy files).   Default: enabled if no options.
#
# --optimal, -o       Enable broadcast for advanced folders
#
# Examples:
#   # Full run (delete + broadcast settings)
#   bash sync_settings.sh
#
#   # Delete only
#   bash sync_settings.sh --delete
#   # Or short form
#   bash sync_settings.sh -d
#
#   # Broadcast only
#   bash sync_settings.sh --broadcast
#   # Or short form
#   bash sync_settings.sh -b
#
#   # Both explicitly (mix long/short forms)
#   bash sync_settings.sh --delete -b
#
# Notes:
# - If no options provided, both steps run.
# - Supports mixing long (--delete, --copy) and short (-d, -c) options.
# - Backup files before delete step!
# - Grant execute: chmod +x sync_settings.sh
#
# Author: Xinye Chen (xinyechenai@gmail.com)
# Last Updated: November 16, 2025
#!/usr/bin/env bash
#!/usr/bin/env bash
DO_DELETE=false
DO_BROADCAST=false
ADVANCED=false

usage() {
    echo "Usage: $0 [options]"
    echo ""
    echo "Options:"
    echo "  --delete, -d        Delete run_setting_*.py, run_debug_*.sh and fp.json"
    echo "  --broadcast, -b     Broadcast run_setting_*.py, run_debug_*.sh and fp.json"
    echo "  --advanced, -a      Include advanced/run_setting_*.py"
    echo ""
    echo "If no options are given, all operations run."
}

# ---------------------------------------
# Parse args
# ---------------------------------------
while [[ $# -gt 0 ]]; do
    case $1 in
        --delete|-d) DO_DELETE=true ;;
        --broadcast|-b) DO_BROADCAST=true ;;
        --advanced|-a) ADVANCED=true ;;
        *) echo "Unknown option: $1"; usage; exit 1 ;;
    esac
    shift
done

# default everything
if ! $DO_DELETE && ! $DO_BROADCAST; then
    DO_DELETE=true
    DO_BROADCAST=true
fi

# ---------------------------------------
# Check run_settings
# ---------------------------------------
if $DO_BROADCAST && [ ! -d "../run_settings" ]; then
    echo "Error: ../run_settings folder missing."
    exit 1
fi

# ---------------------------------------
# Collect subdirs
# ---------------------------------------
shopt -s nullglob
subdirs=(*/)

if [ ${#subdirs[@]} -eq 0 ]; then
    echo "No subfolders found."
    exit 1
fi

#############################################
# Step 1: Delete files
#############################################
if $DO_DELETE; then
    echo "Deleting run_setting_*.py, run_debug_*.sh and fp.json..."
    for subdir in "${subdirs[@]}"; do
        find "$subdir" -maxdepth 1 \( \
            -name "run_setting_*.py" -o \
            -name "run_debug_*.sh" -o \
            -name "fp.json" \
        \) -delete
        echo "Cleaned $subdir"
    done
fi

#############################################
# Step 2: Broadcast files + fp.json
#############################################
if $DO_BROADCAST; then
    echo "Broadcasting run_setting_*.py, run_debug_*.sh and fp.json..."

    py_files=(../run_settings/run_setting_*.py)
    sh_files=(../run_settings/run_debug_*.sh)
    FP_SOURCE="../run_settings/fp.json"

    if [ ! -f "$FP_SOURCE" ]; then
        echo "Warning: fp.json not found."
    fi

    for subdir in "${subdirs[@]}"; do
        # copy python files
        if (( ${#py_files[@]} > 0 )); then
            cp "${py_files[@]}" "$subdir/"
        fi

        # copy shell scripts
        if (( ${#sh_files[@]} > 0 )); then
            cp "${sh_files[@]}" "$subdir/"
        fi

        # copy fp.json
        if [ -f "$FP_SOURCE" ]; then
            cp "$FP_SOURCE" "$subdir/"
        fi

        echo "Broadcasted scripts and fp.json to $subdir"
    done

    # advanced python configs
    if $ADVANCED && [ -d "../run_settings/advanced" ]; then
        adv_files=(../run_settings/advanced/run_setting_*.py)
        if (( ${#adv_files[@]} > 0 )); then
            for subdir in "${subdirs[@]}"; do
                cp "${adv_files[@]}" "$subdir/"
                echo "Broadcasted advanced configs to $subdir"
            done
        else
            echo "No run_setting_*.py found in advanced/"
        fi
    fi
fi

echo "Operation completed!"