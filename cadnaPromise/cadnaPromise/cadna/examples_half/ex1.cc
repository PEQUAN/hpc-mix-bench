#include <stdio.h>
#include <math.h>
#include <cadna.h>
#include <iostream>

using namespace std;
int main()
{
  printf("------------------------------------------\n");
  printf("|  Polynomial function of two variables  |\n");
  printf("|  without CADNA                         |\n");
  printf("------------------------------------------\n");

  half x;
  x = 38808.5;
  half y;
  y = 33096.;
  half res;

  res=333.75*y*y*y*y*y*y+4*x*x*(11*4*x*x*y*y-y*y*y*y*y*y-121*y*y*y*y-2.0)   
    +5.5*y*y*y*y*y*y*y*y+2*x/(2*y);

  cout<<"res:"<<res<<endl;

   return 0;
}




