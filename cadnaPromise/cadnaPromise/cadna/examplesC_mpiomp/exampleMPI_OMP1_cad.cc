#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>

#include <mpi.h>
#include <omp.h>

#include <cadna.h>
#include <cadna_mpi.h>

// code executed by process 1
double_st proc1(double_st x, double_st y)
{
  double_st res1=1;

#pragma omp parallel sections  reduction(*:res1) num_threads(3)
  {
    // res1*= is used to avoid a bug if one thread runs several sections
#pragma omp section 
    {
      res1*=9.*x ;
      printf("MPI 1.1: thread %d  Result res1=%s\n", omp_get_thread_num(),strp(res1));
    }
#pragma omp section 
    {
      res1*=x;
      printf("MPI 1.2: thread %d  Result res1=%s\n", omp_get_thread_num(),strp(res1));
    }
#pragma omp section 
    {
      res1*=x*x;
      printf("MPI 1.3: thread %d  Result  res1=%s\n", omp_get_thread_num(),strp(res1));
    }
  }
#pragma omp barrier
      printf("MPI 1.4: Final result: 9*x*x*x*x=%s\n", strp(res1));
      return (res1);
}

// code executed by process 2
double_st proc2(double_st x, double_st y)
{
  double_st res2=1; 

#pragma omp parallel sections   reduction(*:res2) num_threads(3)
  {
#pragma omp section 
    {
      res2*=-y; 
      printf("MPI 2.1:  thread %d  Result res2=%s\n", omp_get_thread_num(),strp(res2));
    }
#pragma omp section 
    { 
      res2*=y;
      printf("MPI 2.2: thread %d  Result res2=%s\n", omp_get_thread_num(),strp(res2));
    }
#pragma omp section 
    { 
      res2*=y*y; 
      printf("MPI 2.3: thread %d  Result res2=%s\n", omp_get_thread_num(),strp(res2));
    }
  }
  
#pragma omp barrier
  printf("MPI 2.4: Final result: -y*y*y*y=%s\n", strp(res2));
  return (res2);
}

// code executed by process 3
double_st proc3(double_st x, double_st y)
{
  double_st res3=1; 
#pragma omp  parallel sections   reduction(*:res3) num_threads(3)
  {
#pragma omp section 
    {
      res3*=2. ; 
      printf("MPI 3.1: thread %d  Result res3=%s\n", omp_get_thread_num(),strp(res3));
    }
#pragma omp section 
    { 
      res3*=y*y; 
      printf("MPI 3.2: thread %d Result res3=%s\n", omp_get_thread_num(),strp(res3));
    }
  }
  
#pragma omp barrier
  printf("MPI 3.3: Final result: 2*y*y=%s\n", strp(res3));
  return (res3);
}
  
int main(int argc, char *argv[]) {

  int p, np, mpisupport;
  MPI_Status status;

  /* Initialization */
  MPI_Init_thread( &argc, &argv, MPI_THREAD_FUNNELED, &mpisupport);
  //MPI_Init_thread( &argc, &argv, MPI_THREAD_MULTIPLE, &mpisupport);
  MPI_Comm_rank(MPI_COMM_WORLD, &p);
  MPI_Comm_size(MPI_COMM_WORLD, &np);
  
  cadna_mpi_init(p, -1);
  
  if( p ==0 ) {
      fprintf( stderr, "Computation with MPI and OpenMP using CADNA:\n");
      fprintf( stderr, "MPI: I am the Master\n");
  }
  if( np !=4 ) {
    /* seul le maitre parle */
    if( p ==0 ) fprintf( stderr, "4 processes are necessary\n");
    MPI_Finalize();
    return 0;
  }
  
  if (mpisupport <MPI_THREAD_FUNNELED){
    fprintf( stderr, "PB MPI\n");
    return 0;
  }
  double_st x=10864., y=18817.;
  
  if (p!=0) {
    switch(p) {
      
    case 1 : {
      double_st res1;
      res1=proc1(x,y);
#pragma omp master
      MPI_Send( &res1, 1, MPI_DOUBLE_ST, 0,  0, MPI_COMM_WORLD);
    }
      break; 
      
  case 2 : { 
    double_st res2;
    res2=proc2(x,y);
#pragma omp master
    MPI_Send( &res2, 1, MPI_DOUBLE_ST, 0,  0, MPI_COMM_WORLD);
  }
    break;
      
    case 3 : {
      double_st res3;
      res3=proc3(x,y);
#pragma omp master
      MPI_Send( &res3, 1, MPI_DOUBLE_ST, 0,  0, MPI_COMM_WORLD);
    }
   break;
    }  
  }    
  
  else{
    double_st res=0; 
    double_st tmp;
    int i;
    for (i=1; i<=3; i++) {
#pragma omp master
      MPI_Recv( &tmp, 1, MPI_DOUBLE_ST, i, 0, MPI_COMM_WORLD, &status);
#pragma omp barrier
      res+=tmp;
      printf("MPI 4.1: I have received from %d %s\n",i, strp(tmp));
    }
    printf("-->MPI 4.2: Final result: %s\n", strp(res)); //Exact result: 1
  }

  cadna_mpi_end();  
  MPI_Finalize();  
  return 0;
}
