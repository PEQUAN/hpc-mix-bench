g++ backprop.c backprop_kernel.c facetrain.c imagenet.c -frounding-math -m64 -o backprop -fopenmp
./backprop 65536

