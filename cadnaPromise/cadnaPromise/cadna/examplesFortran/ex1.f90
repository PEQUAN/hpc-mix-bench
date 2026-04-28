      program ex1
      implicit none
      double precision :: y,x,res
      print *,'Polynomial function of two variables without CADNA: '
      x=77617e0
      y=33096e0
      res=333.75*y*y*y*y*y*y+x*x*(11*x*x*y*y-y*y*y*y*y*y-121 &
     &    *y*y*y*y-2.0) &
     &  +5.5*y*y*y*y*y*y*y*y+x/(2*y)
      print *,'res = ',res
      end program ex1
