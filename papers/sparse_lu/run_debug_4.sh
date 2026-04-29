#!/bin/bash
set -u

shopt -s nullglob
dirs=(digit4_*/)
if [ ${#dirs[@]} -gt 0 ]; then
    rm -rf "${dirs[@]}"
fi
shopt -u nullglob

rm -rf digit4_* debug
mkdir -p logs

for n in {1..10}; do
    echo "[RUN] promise --precs=wpsd --nbDigits=$n"

    rm -rf debug

    promise --precs=wpsd --nbDigits="$n" > "logs/log_4_${n}.txt" 2>&1
    echo "Done nbDigits=$n, see logs/log_4_${n}.txt"

    mv debug "digit4_${n}"
    echo "[DONE] digit4_${n}"
done