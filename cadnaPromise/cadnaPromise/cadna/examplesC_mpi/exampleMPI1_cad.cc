#include <stdlib.h>
#include <stdio.h>
#include <mpi.h>

#include <cadna.h>
#include <cadna_mpi.h>

int main(int argc, char *argv[]) {

int p, np;
MPI_Status status;

 /* MPI Initialization */
MPI_Init( &argc, &argv);
MPI_Comm_rank(MPI_COMM_WORLD, &p);
MPI_Comm_size(MPI_COMM_WORLD, &np);

cadna_mpi_init(p, -1);

  if( p ==0 ){
     fprintf( stderr, "Computation with MPI using CADNA:\n");
     fprintf( stderr, "I am the Master\n");
  } 
  if( np !=4 ) {
    if( p ==0 ) fprintf( stderr, "4 processes are necessary\n");
    MPI_Finalize();
    return 0;
  }
  
  double_st x=10864., y=18817.;
  
  if (p!=0) {
    double_st res;
    switch(p){
    case 1 : res=9.*x*x*x*x; break;
    case 2 : res=- y*y*y*y; break;
    case 3: res= 2.*y*y; break;
    } 
   printf("I am thread %d and I send %s\n",p,strp(res));
   MPI_Send( &res, 1, MPI_DOUBLE_ST, 0, 0, MPI_COMM_WORLD);
  }
  else{
    double_st res=0.; 
    double_st tmp;
    int i;
    for (i=1; i<=3; i++) {
      MPI_Recv( &tmp, 1, MPI_DOUBLE_ST, MPI_ANY_SOURCE, 0, 
                MPI_COMM_WORLD, &status);
      res+=tmp;
      printf("I have received %s\n",strp(tmp));
    }
  printf("-->Final result: %s\n", strp(res));
  printf("Exact result:  1.000000000000000e+00\n");
 }
  
  cadna_mpi_end();
  MPI_Finalize();  
  return 0;
}
