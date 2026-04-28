#include <math.h>
#include <stdio.h>

int main()
{
  int i, nmax=100;
  double y, x, eps=1.e-12; 
  printf("-------------------------------------------------------------\n");
  printf("|  Computation of a root of a polynomial by Newton's method |\n");
  printf("|  without CADNA                                            |\n");
  printf("-------------------------------------------------------------\n");
  
  y = 0.5;
  for(i = 1;i<=nmax;i++){
    x = y;
    y = x-(1.47*x*x*x+1.19*x*x-1.83*x+0.45) / (4.41*x*x+2.38*x-1.83); 
    if (fabs(x-y)<=eps) break;
  }
  printf("x(%3d) = %+.15e\n",i-1,x);
  printf("x(%3d) = %+.15e\n",i,y);
  return 0;
}
