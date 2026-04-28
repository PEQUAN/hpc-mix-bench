      program ex1_cad
      use cadna
      implicit none
      type (double_st) :: y,x,res
      call cadna_init(-1)
      print *,'Polynomial function of two variables with CADNA: '
      x=77617e0
      y=33096e0
      res=333.75*y*y*y*y*y*y+x*x*(11*x*x*y*y-y*y*y*y*y*y-121 &
     &   *y*y*y*y-2.0) &
     &   +5.5*y*y*y*y*y*y*y*y+x/(2*y)
      print *,'res = ',str(res)
      call cadna_end()
      end program ex1_cad
