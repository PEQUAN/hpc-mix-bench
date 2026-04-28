#!/bin/bash
set -u

shopt -s nullglob
dirs=(digit1_*/)
if [ ${#dirs[@]} -gt 0 ]; then
    rm -rf "${dirs[@]}"
fi
shopt -u nullglob

mkdir -p logs

for n in {1..10}; do
    echo "[RUN] promise --precs=wpsd --nbDigits=$n"

    if ! promise --precs=wpsd --nbDigits="$n" > "logs/log_1_${n}.txt" 2>&1; then
        echo "[FAILED] promise nbDigits=$n, see logs/log_1_${n}.txt"
        continue
    fi

    if [ ! -d debug ]; then
        echo "[FAILED] no debug/ generated for nbDigits=$n, see logs/log_1_${n}.txt"
        continue
    fi

    rm -rf "digit1_${n}"
    mv debug "digit1_${n}"
    echo "[DONE] digit1_${n}"
done