export OMP_NUM_THREADS=10
g++ -fopenmp -O3 lu_sparse2.cpp -o lu_sparse
./lu_sparse