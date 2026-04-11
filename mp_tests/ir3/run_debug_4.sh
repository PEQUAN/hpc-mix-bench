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

promise --precs=cbsd --nbDigits=1 >> logs/log_4_1.txt
mv debug digit4_1
promise --precs=cbsd --nbDigits=2 >> logs/log_4_2.txt
mv debug digit4_2
promise --precs=cbsd --nbDigits=3 >> logs/log_4_3.txt
mv debug digit4_3
promise --precs=cbsd --nbDigits=4 >> logs/log_4_4.txt
mv debug digit4_4
promise --precs=cbsd --nbDigits=5 >> logs/log_4_5.txt
mv debug digit4_5
promise --precs=cbsd --nbDigits=6 >> logs/log_4_6.txt
mv debug digit4_6
promise --precs=cbsd --nbDigits=7 >> logs/log_4_7.txt
mv debug digit4_7
promise --precs=cbsd --nbDigits=8 >> logs/log_4_8.txt
mv debug digit4_8
promise --precs=cbsd --nbDigits=9 >> logs/log_4_9.txt
mv debug digit4_9
promise --precs=cbsd --nbDigits=10 >> logs/log_4_10.txt
mv debug digit4_10
