#include <cadna.h>
#include <math.h>
#include <stdio.h>

int main()
{
  int i, nmax=100;
  double_st y, x;
  
  cadna_init(-1);

  printf("-------------------------------------------------------------\n");
  printf("|  Computation of a root of a polynomial by Newton's method |\n");
  printf("|  with optimization and CADNA                              |\n");
  printf("-------------------------------------------------------------\n");
  
  y = 0.5;
  for(i = 1;i<=nmax;i++){
    x = y;
    y = ((4.2*x + 3.5)*x + 1.5)/(6.3*x + 6.1);
    if (x==y)break;
  }
  printf("x(%3d) = %s\n",i-1,strp(x)); 
  printf("x(%3d) = %s\n",i, strp(y));  

  cadna_end();
}







