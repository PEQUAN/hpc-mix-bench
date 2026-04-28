#include <cadna.h>
#include <stdio.h>
#include <iostream>


using namespace std;

int main()
{

  cadna_init(-1);
  printf("------------------------------------------\n");
  printf("|  Polynomial function of two variables  |\n");
  printf("|  with CADNA                            |\n");
  printf("------------------------------------------\n");

  half_st x= 38808.5;
  half_st y= 33096.;
  half_st res;



res=333.75*y*y*y*y*y*y+4*x*x*(11*4*x*x*y*y-y*y*y*y*y*y-121*y*y*y*y-2.0)+5.5*y*y*y*y*y*y*y*y+(2*x)/(2*y);

  cout << "res="<<res<<endl;
  printf("Exact result: -0.82739605994682...\n");
  cadna_end();
}




