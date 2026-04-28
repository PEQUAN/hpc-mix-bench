#include "stdio.h"
#include <math.h>

int  main()
{
  double r=0,r1,x,y,z;
  int i, nloop, ierr;
  printf("Example created on purpose to make CADNA fail\n");
  printf("The same result r is computed for a number of iterations chosen by the user\n");
  printf("The exact result is 1.4E-10\n");
  printf("Enter the number of iterations: ");
  scanf("%d",&nloop);
  ierr = 0;
  for(i=0;i<nloop;i++){
    x=6.83561e+5;
    y=6.83560e+5;
    z=1.00000000007;
    r = z - x;
    r1 = z - y;
    r = r + y;
    r1 = r1 + x;
    r1 = r1 - 2;
    r = r + r1;
    //      r = ((z-x)+y) + ((z-y)+x-2)
    if(r != 1.4e-10) ierr = ierr + 1;
  }
  printf("Last value of r: %.15e\nNumber of iterations when r is different from the exact result: %d\n", r ,ierr);
  return 0;
}

