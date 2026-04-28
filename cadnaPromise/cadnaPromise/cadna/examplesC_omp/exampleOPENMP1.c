#include <stdlib.h>
#include <stdio.h>
#include <omp.h>

int main(int argc, char *argv[]) {
  
  const int n = 1000000;
  int i;
  float *A; 
  float S;

  A=(float*)malloc(sizeof(float)*n*2);
  omp_set_num_threads(4); //the nb of threads can be changed
  
  printf("Computation with OpenMP without CADNA:\n");
#pragma omp parallel for 
   for (i=0;i<2*n;i=i+2) {
     A[i+1]=(float)i+1.;
     A[i]=-(float)i;
   }

  S=0.;
#pragma omp parallel for reduction(+:S) schedule(static,1)
  for (i=0;i<2*n;i++)
   S=S+A[i];
  
  printf("Sum:          %.7e\nExact result: 1.0000000e+06\n", S);
  free(A);
  return 0;
}
