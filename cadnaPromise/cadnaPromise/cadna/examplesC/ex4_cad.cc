#include <cadna.h>
#include <math.h>
#include <stdio.h>

using  namespace std;

int main()
{
  int i;
  double_st a,b,c;

  cadna_init(-1);

  printf("-------------------------------------\n");
  printf("| A second order recurrent sequence |\n");
  printf("| with CADNA                        |\n");
  printf("-------------------------------------\n");

  a = 5.5;
  b = 61./11.; 
  for(i=3;i<=30;i++){
    c = b;
    b = 111. - 1130./b + 3000./(a*b);
    a = c;
    printf("U(%d) = %s\n",i,strp(b));
  }
  printf("The exact limit is 6.\n");

  cadna_end();
}






