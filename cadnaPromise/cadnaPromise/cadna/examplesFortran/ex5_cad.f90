      program ex5_cad
      use cadna
      implicit type (double_st) (a-h,o-z)
      double precision eps
      parameter (eps = 1.0d-12, nmax = 100)
!
      call cadna_init(-1)
      print *,'Computation of a root of a polynomial by Newton method'
      print *,'with CADNA:'
      y = 0.5d0
      do i = 1,nmax
        x = y
        y = x - (1.47d0*x**3 + 1.19d0*x**2 - 1.83d0*x + 0.45d0)&
     &         /(4.41d0*x**2 + 2.38d0*x - 1.83d0)
        if (abs(x-y).le.eps) goto 1
      enddo
!
 1    continue
!
      write(*,*)'x(',i-1,') = ',str(x)
      write(*,*)'x(',i,') = ',str(y)
      call cadna_end()
!
      end
