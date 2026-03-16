export CADNA_PATH=/home/xinye/.local/lib/python3.12/site-packages/cadnaPromise/cadna


g++ sor.cpp -frounding-math -m64 -o sor.out -lcadnaC -L$CADNA_PATH/lib -I$CADNA_PATH/include
./sor.out