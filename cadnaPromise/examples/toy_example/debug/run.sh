export CADNA_PATH=/home/xinye/.local/lib/python3.12/site-packages/cadnaPromise/cadna
g++ toy.cpp -O3 -frounding-math -m64 -o toy.out -lcadnaC -L$CADNA_PATH/lib -I$CADNA_PATH/include 
./toy.out