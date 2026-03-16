export CADNA_PATH=/home/xinye/.local/lib/python3.12/site-packages/cadnaPromise/cadna


g++ srad.cpp -frounding-math -m64 -o srad.out -lcadnaC -L$CADNA_PATH/lib -I$CADNA_PATH/include
./srad.out 2048 2048 0 127 0 127 2 0.5 2