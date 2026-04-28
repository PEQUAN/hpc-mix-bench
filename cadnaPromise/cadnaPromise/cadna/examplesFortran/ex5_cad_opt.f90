      program ex5_cad_opt
      use cadna
      implicit type (double_st) (a-h,o-z)
      parameter (nmax = 100)
!
      call cadna_init(-1)
      y = 0.5d0
      do i = 1,nmax
        x = y
          y = ((4.2d0*x + 3.5d0)*x + 1.5d0)/(6.3d0*x + 6.1d0)
          if (x.eq.y) goto 1
      enddo
!
 1    continue
!
      write(*,*)'x(',i-1,') = ',str(x)
      write(*,*)'x(',i,') = ',str(y)
      call cadna_end()
!
      end
