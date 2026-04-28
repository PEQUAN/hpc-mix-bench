      program ex3
      implicit double precision (a-h,o-z)
      dimension amat(11,11)
      !
      print *,'Computation of the determinant of Hilbert matrix'
      print *,'using Gaussian elimination without CADNA:'
      !
      do i=1,11
        do j=1,11
          amat(i,j) = 1.d0/(i+j-1)
        enddo
      enddo
      det = 1.d0
!
      do i=1,10
        write(*,'(a13,i2,a4,d23.16)')'Pivot number ',i,'  = ',amat(i,i)
        det = det*amat(i,i)
        aux = 1.d0/amat(i,i)
        do j=i+1,11
          amat(i,j) = amat(i,j)*aux
        enddo
!
        do j=i+1,11
          aux = amat(j,i)
          do k=i+1,11
            amat(j,k) = amat(j,k) - aux*amat(i,k)
          enddo
        enddo
!
      enddo
!
      write(*,'(a13,i2,a4,d23.16)')'Pivot number ',i,'  = ',amat(i,i)
      det = det*amat(i,i)
      write(*,'(a19,d23.16)')'Determinant      = ',det
!
      end
