#include <iostream>
#include <cadna.h>

using namespace std;
int main()
{
  half_st amat[3][3];
  int i,j,k;
  half_st s, aux, det;

  cadna_init(-1);
  cout << "--------------------------------------------------------" << endl;
  cout << "|  Computation of the determinant of Hilbert's matrix  |" << endl; 
  cout << "|  using Gaussian elimination with CADNA               |" << endl;
  cout << "--------------------------------------------------------" << endl;

  for(i=1;i<=3;i++)
    for(j=1;j<=3;j++){
       s = i+j-1;
      amat[i-1][j-1] = 1./s;
      amat[i-1][j-1].data_st();
    }
  
  det = 1.;
  for(i=0;i<2;i++){
    cout << "Pivot number " << i << " = " << amat[i][i] << endl;
    det = det*amat[i][i];
    aux = 1./amat[i][i];
    for(j=i+1;j<3;j++){
      amat[i][j] = amat[i][j]*aux;
    }
    for(j=i+1;j<3;j++){
      aux = amat[j][i];
      for(k=i+1;k<3;k++){
	amat[j][k] = amat[j][k] - aux*amat[i][k];
      }
    }
  }
  cout << "Pivot number " << i << " = "  << amat[i][i] << endl;
  det = det*amat[i][i];
  cout << "Determinant      = " << det << endl;
  //cout << "Remark : only the first three digits are exact : 3.02e-65"<< endl; 
  cadna_end();	  
}
