#include <iostream>
#include <cadna.h>

using namespace std;
int main()
{
  double_st amat[11][11];
  int i,j,k;
  double_st aux, det;

  cadna_init(-1);
  cout << "--------------------------------------------------------" << endl;
  cout << "|  Computation of the determinant of Hilbert's matrix  |" << endl; 
  cout << "|  using Gaussian elimination with CADNA               |" << endl;
  cout << "--------------------------------------------------------" << endl;

  for(i=1;i<=11;i++)
    for(j=1;j<=11;j++)
      amat[i-1][j-1] = 1./(double_st)(i+j-1);
  
  det = 1.;

  for(i=0;i<10;i++){
    cout << "Pivot number " << i << " = " << amat[i][i] << endl;
    det = det*amat[i][i];
    aux = 1./amat[i][i];
    for(j=i+1;j<11;j++)
      amat[i][j] = amat[i][j]*aux;
    
    for(j=i+1;j<11;j++){
      aux = amat[j][i];
      for(k=i+1;k<11;k++)
	amat[j][k] = amat[j][k] - aux*amat[i][k];
    }
  }
  cout << "Pivot number " << i << " = "  << amat[i][i] << endl;
  det = det*amat[i][i];
  cout << "Determinant      = " << det << endl;
  cout << "Remark : only the first three digits are exact : 3.02e-65"<< endl; 
  cadna_end();
	  
}
