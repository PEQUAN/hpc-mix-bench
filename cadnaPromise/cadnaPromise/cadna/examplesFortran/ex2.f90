      program ex2
      print *,'Second order equation without CADNA:' 
      a = 0.3
      b = -2.1
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
          write(*,1000) x1
        endif
!
      else
!
        b = b/a
        c = c/a
        d = b*b - 4.0*c
        write(*,*)'d = ',d
!
        if (d.eq.0.) then
          x1 = -b*0.5
          write(*,1001) x1
!
        elseif (d.gt.0.) then
          x1 = ( - b - sqrt(d))*0.5
          x2 = ( - b + sqrt(d))*0.5
          write(*,1002) x1, x2
!
        else
          x1 = - b*0.5
          x2 = sqrt(-d)*0.5
          write(*,1003) x1, x2, x1, -x2
        endif
!
      endif
!
 1000 format('The equation is degenerated.'/, &
     &       'There is one real solution ',e14.7)
!
 1001 format('Discriminant is zero.'/, &
     &       'The double solution is ',e14.7)
!
 1002 format('There are two real solutions.'/, &
     &       'x1 = ',e14.7/,'x2 = ',e14.7)
!
 1003 format('There are two complex solutions.'/, & 
     &       'z1 = ',e14.7,'  +  i * ',e14.7/, &
     &       'z2 = ',e14.7,'  +  i * ',e14.7)
!
      end    
