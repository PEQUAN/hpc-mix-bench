#!/bin/bash
set -u

shopt -s nullglob
dirs=(digit3_*/)
if [ ${#dirs[@]} -gt 0 ]; then
    rm -rf "${dirs[@]}"
fi
shopt -u nullglob

rm -rf digit3_* debug
mkdir -p logs

for n in {1..10}; do
    echo "[RUN] promise --precs=wpsd --nbDigits=$n"

    rm -rf debug

    promise --precs=wpsd --nbDigits="$n" > "logs/log_3_${n}.txt" 2>&1
    echo "Done nbDigits=$n, see logs/log_3_${n}.txt"

    mv debug "digit3_${n}"
    echo "[DONE] digit3_${n}"
done