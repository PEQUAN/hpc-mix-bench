#include <math.h>
#include <stdio.h>

int main()
{
  printf("------------------------------------------\n");
  printf("|  Polynomial function of two variables  |\n");
  printf("|  without CADNA                         |\n");
  printf("------------------------------------------\n");

  double x = 77617.;
  double y = 33096.;
  double res;

  res=333.75*y*y*y*y*y*y+x*x*(11*x*x*y*y-y*y*y*y*y*y-121*y*y*y*y-2.0)   
    +5.5*y*y*y*y*y*y*y*y+x/(2*y);

  printf("res: %.14e\n",res);
  return 0;
}



