#!/bin/bash
set -u

shopt -s nullglob
dirs=(digit3_*/)
if [ ${#dirs[@]} -gt 0 ]; then
    rm -rf "${dirs[@]}"
fi
shopt -u nullglob

mkdir -p logs

for n in {1..10}; do
    echo "[RUN] promise --precs=wpsd --nbDigits=$n"

    if ! promise --precs=wpsd --nbDigits="$n" > "logs/log_3_${n}.txt" 2>&1; then
        echo "[FAILED] promise nbDigits=$n, see logs/log_3_${n}.txt"
        continue
    fi

    if [ ! -d debug ]; then
        echo "[FAILED] no debug/ generated for nbDigits=$n, see logs/log_3_${n}.txt"
        continue
    fi

    rm -rf "digit3_${n}"
    mv debug "digit3_${n}"
    echo "[DONE] digit3_${n}"
done