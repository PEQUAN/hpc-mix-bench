      program ex4
      implicit double precision (a-h,o-z)
      !
      print *,'A second order recurrent sequence without CADNA:'
      a = 5.5d0
      b = 61.d0/11.d0
      do i=3,30
        c = b
        b = 111.d0 - 1130.d0/b + 3000.d0/(a*b)
        a = c
        write(*,'(a2,i2,a3,d23.16)')'U(',i,') = ',b
      enddo
      write(*,*)'The exact limit is 6.'
!
      end
