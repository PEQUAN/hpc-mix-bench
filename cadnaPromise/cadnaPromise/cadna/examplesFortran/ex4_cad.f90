      program ex4_cad
      use cadna
      implicit type (double_st) (a-h,o-z)
      call cadna_init(-1)
      !
      print *,'A second order recurrent sequence with CADNA:'
      a = 5.5d0
      b = 61.d0/11.d0
      do i=3,30
        c = b
        b = 111.d0 - 1130.d0/b + 3000.d0/(a*b)
        a = c
        write(*,'(a2,i2,a3,a23)')'U(',i,') = ',str(b)
      enddo
      write(*,*)'The exact limit is 6.'
      call cadna_end()
!
      end
