#include <iostream>

using namespace std;

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
    cerr <<  "Computation with MPI using CADNA:" << endl;
    cerr <<  "I am the Master" << endl;   
  }
  if( np !=4 ) {
    if( p==0 ) cerr <<  "4 processes are necessary" << endl;
    MPI_Finalize();
    return 0;
  }
  
  float_st x=10864., y=18817.;
  
  if (p!=0) {
    float_st res;
    switch(p){
    case 1 : {
      res=9.*x*x*x*x; 
      break;
    }
    case 2 : {
      res=- y*y*y*y;      
      break;
    }
    case 3 : {
      res= 2.*y*y; 
      break;
    }
    }
    //printf("I am thread %d and I send %s\n",p,strp(res));
    cout << "I am thread " << p << " and I send " << strp(res) << endl;
    MPI_Send( &res, 1, MPI_FLOAT_ST, 0, 0, MPI_COMM_WORLD);
  }
  else{
    float_st res=0.; 
    float_st tmp;
    int i;
    for (i=1; i<=3; i++) {
      MPI_Recv( &tmp, 1, MPI_FLOAT_ST, MPI_ANY_SOURCE, 0, 
                MPI_COMM_WORLD, &status);
      res+=tmp;
      cout << "I have received " <<  strp(tmp) << endl ;
    }
    cout <<  "-->Final result: " << strp(res) << endl;
    cout << "Exact result:  1" << endl;
  }

  cadna_mpi_end();
  MPI_Finalize();  
  return 0;
}
