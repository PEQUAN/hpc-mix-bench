#include <stdlib.h>
#include <stdio.h>
#include <cadna.h>
#include <omp.h>

int main(int argc, char *argv[]) {

  cadna_init(-1); //all instabilities
  const int n = 1000000;
  int i;
  float_st *A; 
  float_st S;

  A=(float_st*)malloc(sizeof(float_st)*n*2);
  omp_set_num_threads(4); //the nb of threads can be changed
  
  printf("Computation with OpenMP using CADNA:\n");
#pragma omp parallel for 
   for (i=0;i<2*n;i=i+2) {
     A[i+1]=(float_st)i+1.;
     A[i]=-(float_st)i;
   }

  S=0.;
#pragma omp parallel for reduction(+:S) schedule(static,1)
  for (i=0;i<2*n;i++)
   S=S+A[i];
  
  printf("Sum:         %s\nExact result: 1.0000000e+06\n", strp(S));
  cadna_end();
  free(A);
  return 0;
}
