      program ex5
      implicit double precision (a-h,o-z)
      parameter (eps = 1.0d-12, nmax = 100)
      !
      print *,'Computation of a root of a polynomial by Newton method'
      print *,'without CADNA:'
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
      write(*,*)'x(',i-1,') = ',x
      write(*,*)'x(',i,') = ',y
!
      end
