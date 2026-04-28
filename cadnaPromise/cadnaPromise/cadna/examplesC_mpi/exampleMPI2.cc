#include <iostream>

using namespace std;

#include <mpi.h>

int main(int argc, char *argv[]) {

  int p, np;
  MPI_Status status;

  /* MPI Initialization */
  MPI_Init( &argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &p);
  MPI_Comm_size(MPI_COMM_WORLD, &np);

  if( p ==0 ) {
    cerr <<  "Computation with MPI without CADNA:" << endl;
    cerr <<  "I am the Master" << endl;
  }
  if( np !=4 ) {
    if( p ==0 ) cerr << "4 processes are necessary" << endl;
    MPI_Finalize();
    return 0;
  }
  
  double x=10864., y=18817.;
  
  if (p!=0) {
    double res;
    switch(p){
    case 1 : res=9.*x*x*x*x; break;
    case 2 : res=- y*y*y*y; break;
    case 3 : res= 2.*y*y; break;
    } 
    // printf("I am thread %d and I send %.15e\n",p,res);
    cout << "I am thread " << p<< " and I send "<<res << endl;
    MPI_Send( &res, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
  }
  else{
    double res=0.; 
    double tmp;
    int i;
    for (i=1; i<=3; i++) {
      MPI_Recv( &tmp, 1, MPI_DOUBLE, MPI_ANY_SOURCE, 0, 
                MPI_COMM_WORLD, &status);
      res+=tmp;
      cout << "I have received " << tmp << endl;
    }
    cout << "Final result: " << res << endl;
    cout << "Exact result: 1" << endl;
  }

  MPI_Finalize();  
  return 0;
}
