#!/bin/bash
ls -d digit_*/
shopt -s nullglob
dirs=(digit_*/)
if [ ${#dirs[@]} -gt 0 ]; then
    rm -r "${dirs[@]}"
fi

if [ -d logs ]; then
    rm -rf logs
fi
mkdir logs

promise --precs=wpsd --nbDigits=1 >> logs/log_1_1.txt
mv debug digit1_1
promise --precs=wpsd --nbDigits=2 >> logs/log_1_2.txt
mv debug digit1_2
promise --precs=wpsd --nbDigits=3 >> logs/log_1_3.txt
mv debug digit1_3
promise --precs=wpsd --nbDigits=4 >> logs/log_1_4.txt
mv debug digit1_4
promise --precs=wpsd --nbDigits=5 >> logs/log_1_5.txt
mv debug digit1_5
promise --precs=wpsd --nbDigits=6 >> logs/log_1_6.txt
mv debug digit1_6
promise --precs=wpsd --nbDigits=7 >> logs/log_1_7.txt
mv debug digit1_7
promise --precs=wpsd --nbDigits=8 >> logs/log_1_8.txt
mv debug digit1_8
promise --precs=wpsd --nbDigits=9 >> logs/log_1_9.txt
mv debug digit1_9
promise --precs=wpsd --nbDigits=10 >> logs/log_1_10.txt
mv debug digit1_10
