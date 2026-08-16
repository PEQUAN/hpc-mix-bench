#!/bin/bash
set -u

shopt -s nullglob
dirs=(digit1_*/)
if [ ${#dirs[@]} -gt 0 ]; then
    rm -rf "${dirs[@]}"
fi
shopt -u nullglob

rm -rf digit1_* debug
mkdir -p logs

for n in {1..10}; do
    echo "[RUN] promise --precs=wpsd --nbDigits=$n"

    rm -rf debug

    promise --precs=wpsd --nbDigits="$n" > "logs/log_1_${n}.txt" 2>&1
    echo "Done nbDigits=$n, see logs/log_1_${n}.txt"

    mv debug "digit1_${n}"
    echo "[DONE] digit1_${n}"
done