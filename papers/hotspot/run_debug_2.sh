#!/bin/bash
set -u

shopt -s nullglob
dirs=(digit2_*/)
if [ ${#dirs[@]} -gt 0 ]; then
    rm -rf "${dirs[@]}"
fi
shopt -u nullglob

rm -rf digit2_* debug
mkdir -p logs

for n in {1..10}; do
    echo "[RUN] promise --precs=wpsd --nbDigits=$n"

    rm -rf debug

    promise --precs=wpsd --nbDigits="$n" > "logs/log_2_${n}.txt" 2>&1
    echo "Done nbDigits=$n, see logs/log_2_${n}.txt"

    mv debug "digit2_${n}"
    echo "[DONE] digit2_${n}"
done