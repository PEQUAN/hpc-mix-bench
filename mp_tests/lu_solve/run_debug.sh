export CADNA_PATH=/home/xinye/.local/lib/python3.12/site-packages/cadnaPromise/cadna
g++ -O3 lu.cpp -frounding-math -m64 -o lu.out -lcadnaC -L$CADNA_PATH/lib -I$CADNA_PATH/include
./lu.out