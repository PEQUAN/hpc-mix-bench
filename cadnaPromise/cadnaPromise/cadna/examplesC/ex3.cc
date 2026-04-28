#include <iostream>

using namespace std;

int main()
{
  double amat[11][11];
  int i,j,k;
  double aux, det;

  cout << "--------------------------------------------------------" << endl;
  cout << "|  Computation of the determinant of Hilbert's matrix  |" << endl; 
  cout << "|  using Gaussian elimination without CADNA            |" << endl;
  cout << "--------------------------------------------------------" << endl;
  cout.precision(15);
  cout.setf(ios::scientific,ios::floatfield);

  for(i=1;i<=11;i++)
    for(j=1;j<=11;j++)
      amat[i-1][j-1] = 1./(double)(i+j-1);
  
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
  cout << "Determinant     = " << det << endl;
  cout << "Question: how many digits are correct?" << endl; 
  return 0;
}
