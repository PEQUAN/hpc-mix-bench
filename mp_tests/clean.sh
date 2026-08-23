#!/bin/bash
# Usage:
# ./clean.sh [operations] [folders...]
# operations = combination of characters:
# c = remove CSV files (*.csv)
# d = remove debug folders
# p = remove prec_setting_{k}.json
# r = remove runtimes{k}.csv
# f = remove fp.json
# j = remove JPG files (*.jpg / *.jpeg)
#
# If operations are omitted → all operations are enabled.
# Remaining arguments = folders to target (optional)
# If no folders given → recursively clean ALL directories under current path
set -euo pipefail

OPS=${1:-"cdprfj"} # default: all operations
shift $([ $# -gt 0 ] && echo 1 || echo 0) # shift if first argument was operations
TARGET_DIRS=("$@") # remaining args = folder paths

# -------------------------------
# Determine target directories
# -------------------------------
if [ ${#TARGET_DIRS[@]} -gt 0 ]; then
    echo "Using explicitly provided target directories:"
    for t in "${TARGET_DIRS[@]}"; do
        echo " - $t"
        if [ ! -d "$t" ]; then
            echo "Error: $t is not a directory."
            exit 1
        fi
    done
    TARGETS=("${TARGET_DIRS[@]}")
else
    echo "No folders provided → recursively operating on ALL directories under current path."
    # 兼容旧版 bash（macOS 默认 bash 3.2）的写法
    TARGETS=()
    while IFS= read -r dir; do
        TARGETS+=("$dir")
    done < <(find . -type d)
fi

if [ ${#TARGETS[@]} -eq 0 ]; then
    echo "No target directories found."
    exit 1
fi

echo "Target directories (${#TARGETS[@]} found):"
printf ' - %s\n' "${TARGETS[@]}"

# -------------------------------
# Helper: check if operation is enabled
# -------------------------------
op_enabled() {
    [[ "$OPS" == *"$1"* ]]
}

# -------------------------------
# 1. Remove *.csv
# -------------------------------
if op_enabled "c"; then
    echo "Removing *.csv files..."
    for dir in "${TARGETS[@]}"; do
        find "$dir" -maxdepth 1 -type f -name "*.csv" -print -delete
    done
    echo "CSV files deleted."
else
    echo "Skipping CSV deletion."
fi

# -------------------------------
# 2. Remove debug folders + logs
# -------------------------------
if op_enabled "d"; then
    echo "Removing top-level logs directory..."
    if [ -d "logs" ]; then
        rm -rf logs
        echo "Removed ./logs/"
    else
        echo "No top-level logs/ found."
    fi
    echo "Removing debug folders and log.txt..."
    for dir in "${TARGETS[@]}"; do
        find "$dir" -maxdepth 1 -type d \
            \( -name "compileErrors" -o -name "digit*" -o -name "logs" \) \
            -print -exec rm -rf {} +
        find "$dir" -maxdepth 1 -type f -name "log.txt" \
            -print -delete
    done
    echo "Debug folders and log.txt deleted."
else
    echo "Skipping debug folder deletion."
fi

# -------------------------------
# 3. Remove prec_setting_{k}.json
# -------------------------------
if op_enabled "p"; then
    echo "Removing prec_setting_{k}.json..."
    for dir in "${TARGETS[@]}"; do
        find "$dir" -maxdepth 1 -type f \
            -regex ".*/prec_setting_[0-9][0-9]*\.json" \
            -print -delete
    done
    echo "prec_setting_{k}.json files deleted."
else
    echo "Skipping prec_setting deletion."
fi

# -------------------------------
# 4. Remove runtimes{k}.csv
# -------------------------------
if op_enabled "r"; then
    echo "Removing runtimes{k}.csv..."
    for dir in "${TARGETS[@]}"; do
        find "$dir" -maxdepth 1 -type f -name "runtimes*.csv" -print -delete
    done
    echo "runtimes{k}.csv deleted."
else
    echo "Skipping runtimes deletion."
fi

# -------------------------------
# 5. Remove fp.json
# -------------------------------
if op_enabled "f"; then
    echo "Removing fp.json..."
    for dir in "${TARGETS[@]}"; do
        find "$dir" -maxdepth 1 -type f -name "fp.json" -print -delete
    done
    echo "fp.json files deleted."
else
    echo "Skipping fp.json deletion."
fi

# -------------------------------
# 6. Remove *.jpg / *.jpeg
# -------------------------------
if op_enabled "j"; then
    echo "Removing *.jpg / *.jpeg files..."
    for dir in "${TARGETS[@]}"; do
        find "$dir" -maxdepth 1 -type f \( -iname "*.jpg" -o -iname "*.jpeg" \) -print -delete
    done
    echo "JPEG files deleted."
else
    echo "Skipping JPG deletion."
fi

echo "Cleanup completed successfully!"