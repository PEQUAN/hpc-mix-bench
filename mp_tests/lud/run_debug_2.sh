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

promise --precs=wbsd --nbDigits=1 >> logs/log_2_1.txt
mv debug digit2_1
promise --precs=wbsd --nbDigits=2 >> logs/log_2_2.txt
mv debug digit2_2
promise --precs=wbsd --nbDigits=3 >> logs/log_2_3.txt
mv debug digit2_3
promise --precs=wbsd --nbDigits=4 >> logs/log_2_4.txt
mv debug digit2_4
promise --precs=wbsd --nbDigits=5 >> logs/log_2_5.txt
mv debug digit2_5
promise --precs=wbsd --nbDigits=6 >> logs/log_2_6.txt
mv debug digit2_6
promise --precs=wbsd --nbDigits=7 >> logs/log_2_7.txt
mv debug digit2_7
promise --precs=wbsd --nbDigits=8 >> logs/log_2_8.txt
mv debug digit2_8
promise --precs=wbsd --nbDigits=9 >> logs/log_2_9.txt
mv debug digit2_9
promise --precs=wbsd --nbDigits=10 >> logs/log_2_10.txt
mv debug digit2_10
