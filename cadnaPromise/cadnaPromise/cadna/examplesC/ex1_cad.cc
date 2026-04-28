#include <cadna.h>
#include <stdio.h>

using namespace std;

int main()
{
  cadna_init(-1);
  printf("------------------------------------------\n");
  printf("|  Polynomial function of two variables  |\n");
  printf("|  with CADNA                            |\n");
  printf("------------------------------------------\n");
  double_st x=77617.;
  double_st y=33096.;
  double_st res;

  res=333.75*y*y*y*y*y*y+x*x*(11*x*x*y*y-y*y*y*y*y*y-121*y*y*y*y-2.0)   
    +5.5*y*y*y*y*y*y*y*y+x/(2*y);

  printf("res: %s\n",strp(res));
  printf("Exact result: -0.82739605994682...\n");
  //  cout<< "x.x="<<x.getx()<< ",x.y="<<x.gety()<< ",x.z="<<x.getz()<<",x.accuracy="<<x.getaccuracy()<<endl;
  cadna_end();
}




