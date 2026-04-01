#!/bin/bash
ls -d digit_*/
shopt -s nullglob
dirs=(digit_*/)
if [ ${#dirs[@]} -gt 0 ]; then
    rm -r "${dirs[@]}"
fi

if [ ! -d logs ]; then
    mkdir logs
fi


promise --digits=cpsd --nbDigits=1 >> logs/log_3_1.txt
mv debug digit3_1
promise --digits=cpsd --nbDigits=2 >> logs/log_3_2.txt
mv debug digit3_2
promise --digits=cpsd --nbDigits=3 >> logs/log_3_3.txt
mv debug digit3_3
promise --digits=cpsd --nbDigits=4 >> logs/log_3_4.txt
mv debug digit3_4
promise --digits=cpsd --nbDigits=5 >> logs/log_3_5.txt
mv debug digit3_5
promise --digits=cpsd --nbDigits=6 >> logs/log_3_6.txt
mv debug digit3_6
promise --digits=cpsd --nbDigits=7 >> logs/log_3_7.txt
mv debug digit3_7
promise --digits=cpsd --nbDigits=8 >> logs/log_3_8.txt
mv debug digit3_8
promise --digits=cpsd --nbDigits=9 >> logs/log_3_9.txt
mv debug digit3_9
promise --digits=cpsd --nbDigits=10 >> logs/log_3_10.txt
mv debug digit3_10
