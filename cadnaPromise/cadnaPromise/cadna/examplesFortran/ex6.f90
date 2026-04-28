      program ex6
      parameter(idim=4,idim1=5)
      dimension a(idim,idim1),xsol(idim)
      !
      print *,'Solving a linear system using Gaussian elimination'
      print *,'with partial pivoting, without CADNA:'
      xsol(1) = 1.
      xsol(2) = 1.
      xsol(3) = 1.e-8
      xsol(4) = 1.
      a(1,1) = 21.0
      a(1,2) = 130.0
      a(1,3) = 0.0
      a(1,4) = 2.1
      a(1,5) = 153.1
      a(2,1) = 13.0
      a(2,2) = 80.0
      a(2,3) = 4.74e+8
      a(2,4) = 752.0
      a(2,5) = 849.74
      a(3,1) = 0.0
      a(3,2) = -0.4
      a(3,3) = 3.9816e+8
      a(3,4) = 4.2
      a(3,5) = 7.7816
      a(4,1) = 0.0
      a(4,2) = 0.0
      a(4,3) = 1.7
      a(4,4) = 9.0e-9
      a(4,5) = 2.6e-8
!
      do i=1,idim-1
!
        pmax = 0.0
        do j=i,idim
          if(abs(a(j,i)).gt.pmax) then
            pmax = abs(a(j,i))
            ll = j
          endif
        enddo
!
        if (ll.ne.i) then
          do j=i,idim1
            aux = a(i,j)
            a(i,j) = a(ll,j)
            a(ll,j) = aux
          enddo
        endif
!
        aux = a(i,i)
        do j =i+1,idim1
           a(i,j) = a(i,j)/aux
        end do
!
        do k=i+1,idim
          aux = a(k,i)
          do j=i+1,idim1
            a(k,j)=a(k,j) - aux*a(i,j)
          end do
        end do
!
      end do
!
      a(idim,idim1) = a(idim,idim1)/a(idim,idim)
!
      do i=idim-1,1,-1
        do j=i+1,idim
          a(i,idim1) = a(i,idim1) - a(i,j)*a(j,idim1)
        enddo
      enddo
!
      do i=1,idim
        write(*,'(a6,i1,a4,e17.7,a22,e17.7,a1)')'x_sol(',i,') = ', &
     &   a(i,idim1),'     (exact solution: ',xsol(i),')'
      enddo
! 
      end



