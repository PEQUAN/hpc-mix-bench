#include <stdio.h>
#include <math.h>

int main()
{      
  float a = 0.3;
  float b = -2.1;
  float c = 3.675;
  float d, x1,x2;

  printf("----------------------------------\n");
  printf("|  Second order equation         |\n");
  printf("|  without CADNA                 |\n");
  printf("----------------------------------\n");

  //
  //      CASE : A = 0
  //
  if (a==0.)
    if (b==0.) {
      if (c==0.) printf("Every complex value is solution.\n");
      else printf("There is no solution.\n");
    }
    else {
      x1 = - c/b;
      printf("The equation is degenerated.\n");
      printf("There is one real solution %+.6e\n",x1);
    }
  else {
    //
    //      CASE : A /= 0
    //
    b = b/a;
    c = c/a;
    d = b*b - 4.0*c;
    printf("d = %+.6e\n",d);
    //
    //   DISCRIMINANT = 0
    //
    if (d==0.) {
      x1 = -b*0.5;
      printf("Discriminant is zero.\n");
      printf("The double solution is %+.6e\n",x1);
    }
    else {
      //
      //      DISCRIMINANT > 0
      //
      if (d>0.) {
          x1 = ( - b - sqrtf(d))*0.5;
          x2 = ( - b + sqrtf(d))*0.5;
	  printf("There are two real solutions.\n");
	  printf("x1 = %+.6e x2 = %+.6e\n",x1,x2);
      }
      else {
	//
	//      DISCRIMINANT < 0
	//
	x1 = - b*0.5;
	x2 = sqrtf(-d)*0.5;
	printf("There are two complex solutions.\n");
	printf("z1 = %+.6e  +  i * %+.6e\n",x1,x2);
	printf("z2 = %+.6e  +  i * %+.6e\n",x1, -x2);
      }
    }
  }
  printf("The exact discriminant value is 0.\n");
  return 0;
}








