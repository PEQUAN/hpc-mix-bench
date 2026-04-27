#!/usr/bin/env bash

set -euo pipefail

# ------------------------------------------------------------
# sync_settings.sh
#
# Synchronize benchmark configuration files from ../run_settings/
# to benchmark folders under the current directory.
#
# Operations:
#   1. Delete existing files in each benchmark folder:
#        - run_setting_*.py
#        - run_debug_*.sh
#        - fp.json
#
#   2. Broadcast files from ../run_settings/ to each benchmark folder:
#        - run_setting_*.py
#        - run_debug_*.sh
#        - fp.json
#
#   3. Optionally broadcast advanced configs from:
#        - ../run_settings/advanced/run_setting_*.py
#
# Target folders:
#   By default, only subfolders containing promise.yml are treated as
#   benchmark folders.
#
# Usage:
#   bash sync_settings.sh [options]
#
# Options:
#   --delete, -d        Delete existing run_setting_*.py, run_debug_*.sh, fp.json
#   --broadcast, -b     Copy files from ../run_settings/ to benchmark folders
#   --advanced, -a      Also copy ../run_settings/advanced/run_setting_*.py
#   --help, -h          Show this help message
#
# Default:
#   If no options are given, both delete and broadcast are enabled.
#
# Examples:
#   bash sync_settings.sh
#   bash sync_settings.sh --delete
#   bash sync_settings.sh --broadcast
#   bash sync_settings.sh --delete --broadcast --advanced
#
# Author: Xinye Chen
# Last Updated: November 16, 2025
# ------------------------------------------------------------

DO_DELETE=false
DO_BROADCAST=false
ADVANCED=false

usage() {
    echo "Usage: $0 [options]"
    echo ""
    echo "Options:"
    echo "  --delete, -d        Delete run_setting_*.py, run_debug_*.sh and fp.json"
    echo "  --broadcast, -b     Broadcast run_setting_*.py, run_debug_*.sh and fp.json"
    echo "  --advanced, -a      Include ../run_settings/advanced/run_setting_*.py"
    echo "  --help, -h          Show this help message"
    echo ""
    echo "If no options are given, delete + broadcast are both enabled."
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --delete|-d)
            DO_DELETE=true
            ;;
        --broadcast|-b)
            DO_BROADCAST=true
            ;;
        --advanced|-a)
            ADVANCED=true
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
    shift
done

if ! $DO_DELETE && ! $DO_BROADCAST; then
    DO_DELETE=true
    DO_BROADCAST=true
fi

if $DO_BROADCAST && [[ ! -d "../run_settings" ]]; then
    echo "Error: ../run_settings folder missing."
    exit 1
fi

shopt -s nullglob

subdirs=()
for d in */; do
    if [[ -f "$d/promise.yml" ]]; then
        subdirs+=("$d")
    fi
done

if (( ${#subdirs[@]} == 0 )); then
    echo "No benchmark folders found."
    echo "Expected subfolders containing promise.yml."
    exit 1
fi

if $DO_DELETE; then
    echo "Deleting run_setting_*.py, run_debug_*.sh and fp.json..."

    for subdir in "${subdirs[@]}"; do
        find "$subdir" -maxdepth 1 -type f \( \
            -name "run_setting_*.py" -o \
            -name "run_debug_*.sh" -o \
            -name "fp.json" \
        \) -delete

        echo "Cleaned $subdir"
    done
fi

if $DO_BROADCAST; then
    echo "Broadcasting run_setting_*.py, run_debug_*.sh and fp.json..."

    py_files=(../run_settings/run_setting_*.py)
    sh_files=(../run_settings/run_debug_*.sh)
    fp_source="../run_settings/fp.json"

    if (( ${#py_files[@]} == 0 )); then
        echo "Warning: no run_setting_*.py found in ../run_settings/"
    fi

    if (( ${#sh_files[@]} == 0 )); then
        echo "Warning: no run_debug_*.sh found in ../run_settings/"
    fi

    if [[ ! -f "$fp_source" ]]; then
        echo "Warning: fp.json not found in ../run_settings/"
    fi

    for subdir in "${subdirs[@]}"; do
        if (( ${#py_files[@]} > 0 )); then
            cp "${py_files[@]}" "$subdir/"
        fi

        if (( ${#sh_files[@]} > 0 )); then
            cp "${sh_files[@]}" "$subdir/"
            chmod +x "$subdir"/run_debug_*.sh 2>/dev/null || true
        fi

        if [[ -f "$fp_source" ]]; then
            cp "$fp_source" "$subdir/"
        fi

        echo "Broadcasted files to $subdir"
    done

    if $ADVANCED; then
        if [[ -d "../run_settings/advanced" ]]; then
            adv_files=(../run_settings/advanced/run_setting_*.py)

            if (( ${#adv_files[@]} > 0 )); then
                for subdir in "${subdirs[@]}"; do
                    cp "${adv_files[@]}" "$subdir/"
                    echo "Broadcasted advanced configs to $subdir"
                done
            else
                echo "Warning: no run_setting_*.py found in ../run_settings/advanced/"
            fi
        else
            echo "Warning: ../run_settings/advanced folder missing."
        fi
    fi
fi

echo "Operation completed!"