      program ex2_cad
      use cadna
      implicit type (single_st) (a-h,o-z)
      call cadna_init(-1)
      print *,'Second order equation with CADNA:' 
      a = 0.3
      call data_st(a)
      b = -2.1
      call data_st(b)
      c = 3.675
!
      if (a.eq.0.) then
        if (b.eq.0.) then
          if (c.eq.0.) then
            write(*,*)'Every complex value is solution.'
          else
            write(*,*)'There is no solution.'
          endif
        else
          x1 = - c/b
          write(*,1000) str(x1)
        endif
!
      else
!
        b = b/a
        c = c/a
        d = b*b - 4.0*c
        write(*,*)'d = ',str(d)
!        call display(d)
!
        if (d.eq.0.) then
          x1 = -b*0.5
          write(*,1001) str(x1)
!
        elseif (d.gt.0.) then
          x1 = ( - b - sqrt(d))*0.5
          x2 = ( - b + sqrt(d))*0.5
          write(*,1002) str(x1), str(x2)
!
        else
          x1 = - b*0.5
          x2 = sqrt(-d)*0.5
          write(*,1003) str(x1), str(x2), str(x1), str(-x2)
        endif
!
      endif
!
 1000 format('The equation is degenerated.'/, &
     &       'There is one real solution ',a15)
!
 1001 format('Discriminant is zero.'/, &
     &       'The double solution is ',a15)
!
 1002 format('There are two real solutions.'/, &
     &       'x1 = ',a15/,'x2 = ',a15)
!
 1003 format('There are two complex solutions.'/, &
     &       'z1 = ',a15,'  +  i * ',a15/, &
     &       'z2 = ',a15,'  +  i * ',a15)
      call cadna_end()
!
      End    
