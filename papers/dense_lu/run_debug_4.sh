#!/bin/bash
set -u

shopt -s nullglob
dirs=(digit4_*/)
if [ ${#dirs[@]} -gt 0 ]; then
    rm -rf "${dirs[@]}"
fi
shopt -u nullglob

mkdir -p logs

for n in {1..10}; do
    echo "[RUN] promise --precs=wpsd --nbDigits=$n"

    if ! promise --precs=wpsd --nbDigits="$n" > "logs/log_4_${n}.txt" 2>&1; then
        echo "[FAILED] promise nbDigits=$n, see logs/log_4_${n}.txt"
        continue
    fi

    if [ ! -d debug ]; then
        echo "[FAILED] no debug/ generated for nbDigits=$n, see logs/log_4_${n}.txt"
        continue
    fi

    rm -rf "digit4_${n}"
    mv debug "digit4_${n}"
    echo "[DONE] digit4_${n}"
done