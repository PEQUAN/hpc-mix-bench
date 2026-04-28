#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>

#include <mpi.h>
#include <omp.h>

// code executed by process 1
double proc1(double x, double y)
{
  double res1=1;

#pragma omp parallel sections  reduction(*:res1) num_threads(3)
  {
#pragma omp section 
    {
      res1*=9.*x ;
      printf("MPI 1.1: thread %d  Result res1=%.15e\n", omp_get_thread_num(),res1);
    }
#pragma omp section 
    {
      res1*=x;
      printf("MPI 1.2: thread %d  Result res1=%.15e\n", omp_get_thread_num(),res1);
    }
#pragma omp section 
    {
      res1*=x*x;
      printf("MPI 1.3: thread %d Result res1=%.15e\n", omp_get_thread_num(),res1);
    }
  }
#pragma omp barrier
      printf("MPI 1.4: Final result:  9*x*x*x*x=%.15e\n", res1);
      return (res1);
}

// code executed by process 2
double proc2(double x, double y)
{
  double res2=1; 
  omp_set_num_threads(10);
#pragma omp parallel sections   reduction(*:res2) num_threads(10)
  {
#pragma omp section 
    {
      res2*=-y; 
      printf("MPI 2.1: thread %d  Result  res2=%.15e\n", omp_get_thread_num(),res2);
    }
#pragma omp section 
    { 
      res2*=y;
      printf("MPI 2.2: thread %d  Result res2=%.15e\n", omp_get_thread_num(),res2);
    }
#pragma omp section 
    { 
      res2*=y*y; 
      printf("MPI 2.3: thread %d Result res2=%.15e\n", omp_get_thread_num(),res2);
    }
  }
  
#pragma omp barrier
  printf("MPI 2.4: Final result:  -y*y*y*y=%.15e\n", res2);
  return (res2);
}

// code executed by process 3
double proc3(double x, double y)
{
  double res3=1; 
  omp_set_num_threads(10);
#pragma omp  parallel sections   reduction(*:res3) num_threads(10)
  {
#pragma omp section 
    {
      res3*=2. ; 
      printf("MPI 3.1: thread %d  Result  res3=%.15e\n", omp_get_thread_num(),res3);
    }
#pragma omp section 
    { 
      res3*=y*y; 
      printf("MPI 3.2: thread %d  Result  res3=%.15e\n", omp_get_thread_num(),res3);
    }
  }
  
#pragma omp barrier
  printf("MPI 3.3: Final result: 2*y*y=%.15e\n", res3);
  return (res3);
}
    
int main(int argc, char *argv[]) {

  int p, np, mpisupport;
  MPI_Status status;

  /* Initialization */
  MPI_Init_thread( &argc, &argv, MPI_THREAD_FUNNELED, &mpisupport);
  MPI_Comm_rank(MPI_COMM_WORLD, &p);
  MPI_Comm_size(MPI_COMM_WORLD, &np);
  
  if( p ==0 ) {
      fprintf( stderr, "Computation with MPI and OpenMP without CADNA:\n");
      fprintf( stderr, "MPI: I am the Master\n");
  }
  if( np !=4 ) {
    if( p ==0 ) fprintf( stderr, "4 processes are necessary\n");
    MPI_Finalize();
    return 0;
  }
    
  if (mpisupport <MPI_THREAD_FUNNELED){
    fprintf( stderr, "PB MPI\n");
    return 0;
  }
  double x=10864., y=18817.;
  
  if (p!=0) {
    switch(p) {
      
    case 1 : {
      double res1;
      res1=proc1(x,y);
#pragma omp master
      MPI_Send( &res1, 1, MPI_DOUBLE, 0,  0, MPI_COMM_WORLD);
    }
      break; 
      
  case 2 : { 
    double res2;
    res2=proc2(x,y);
#pragma omp master
    MPI_Send( &res2, 1, MPI_DOUBLE, 0,  0, MPI_COMM_WORLD);
  }
    break;
      
    case 3 : {
      double res3;
      res3=proc3(x,y);
#pragma omp master
      MPI_Send( &res3, 1, MPI_DOUBLE, 0,  0, MPI_COMM_WORLD);
    }
   break;
    }  
  }    
  
  else{
    double res=0.; 
    double tmp;
    int i;
    for (i=1; i<=3; i++) {
#pragma omp master
      MPI_Recv( &tmp, 1, MPI_DOUBLE, i, 0, MPI_COMM_WORLD, &status);
#pragma omp barrier
      res+=tmp;
      printf("MPI 4.1: I have received from %d  %.15e\n",i, tmp);
    }
    printf("MPI 4.2: Final result: %.15e\n", res); //Exact result: 1
  }
  
  MPI_Finalize();  
  return 0;
}
