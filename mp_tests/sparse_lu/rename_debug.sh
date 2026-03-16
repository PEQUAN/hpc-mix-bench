#!/bin/bash
ls -d prec_*/
rm -r prec_*/

mkdir logs

promise --precs=wpsd --nbDigits=1 >> logs/log_1_1.txt
mv debug prec1_1
promise --precs=wpsd --nbDigits=2 >> logs/log_1_2.txt
mv debug prec1_2
promise --precs=wpsd --nbDigits=3 >> logs/log_1_3.txt
mv debug prec1_3
promise --precs=wpsd --nbDigits=4 >> logs/log_1_4.txt
mv debug prec1_4
promise --precs=wpsd --nbDigits=5 >> logs/log_1_5.txt
mv debug prec1_5
promise --precs=wpsd --nbDigits=6 >> logs/log_1_6.txt
mv debug prec1_6
promise --precs=wpsd --nbDigits=7 >> logs/log_1_7.txt
mv debug prec1_7
promise --precs=wpsd --nbDigits=8 >> logs/log_1_8.txt
mv debug prec1_8
promise --precs=wpsd --nbDigits=9 >> logs/log_1_9.txt
mv debug prec1_9
promise --precs=wpsd --nbDigits=10 >> logs/log_1_10.txt
mv debug prec1_10

promise --precs=wbsd --nbDigits=1 >> logs/log_2_1.txt
mv debug prec2_1
promise --precs=wbsd --nbDigits=2 >> logs/log_2_2.txt
mv debug prec2_2
promise --precs=wbsd --nbDigits=3 >> logs/log_2_3.txt
mv debug prec2_3
promise --precs=wbsd --nbDigits=4 >> logs/log_2_4.txt
mv debug prec2_4
promise --precs=wbsd --nbDigits=5 >> logs/log_2_5.txt
mv debug prec2_5
promise --precs=wbsd --nbDigits=6 >> logs/log_2_6.txt
mv debug prec2_6
promise --precs=wbsd --nbDigits=7 >> logs/log_2_7.txt
mv debug prec2_7
promise --precs=wbsd --nbDigits=8 >> logs/log_2_8.txt
mv debug prec2_8
promise --precs=wbsd --nbDigits=9 >> logs/log_2_9.txt
mv debug prec2_9
promise --precs=wbsd --nbDigits=10 >> logs/log_2_10.txt
mv debug prec2_10

promise --precs=cpsd --nbDigits=1 >> logs/log_3_1.txt
mv debug prec3_1
promise --precs=cpsd --nbDigits=2 >> logs/log_3_2.txt
mv debug prec3_2
promise --precs=cpsd --nbDigits=3 >> logs/log_3_3.txt
mv debug prec3_3
promise --precs=cpsd --nbDigits=4 >> logs/log_3_4.txt
mv debug prec3_4
promise --precs=cpsd --nbDigits=5 >> logs/log_3_5.txt
mv debug prec3_5
promise --precs=cpsd --nbDigits=6 >> logs/log_3_6.txt
mv debug prec3_6
promise --precs=cpsd --nbDigits=7 >> logs/log_3_7.txt
mv debug prec3_7
promise --precs=cpsd --nbDigits=8 >> logs/log_3_8.txt
mv debug prec3_8
promise --precs=cpsd --nbDigits=9 >> logs/log_3_9.txt
mv debug prec3_9
promise --precs=cpsd --nbDigits=10 >> logs/log_3_10.txt
mv debug prec3_10

promise --precs=cbsd --nbDigits=1 >> logs/log_4_1.txt
mv debug prec4_1
promise --precs=cbsd --nbDigits=2 >> logs/log_4_2.txt
mv debug prec4_2
promise --precs=cbsd --nbDigits=3 >> logs/log_4_3.txt
mv debug prec4_3
promise --precs=cbsd --nbDigits=4 >> logs/log_4_4.txt
mv debug prec4_4
promise --precs=cbsd --nbDigits=5 >> logs/log_4_5.txt
mv debug prec4_5
promise --precs=cbsd --nbDigits=6 >> logs/log_4_6.txt
mv debug prec4_6
promise --precs=cbsd --nbDigits=7 >> logs/log_4_7.txt
mv debug prec4_7
promise --precs=cbsd --nbDigits=8 >> logs/log_4_8.txt
mv debug prec4_8
promise --precs=cbsd --nbDigits=9 >> logs/log_4_9.txt
mv debug prec4_9
promise --precs=cbsd --nbDigits=10 >> logs/log_4_10.txt
mv debug prec4_10
