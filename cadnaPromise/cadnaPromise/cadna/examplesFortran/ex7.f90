program ex7
  implicit none
  double precision :: r,r1,x,y,z
  integer :: i, nloop, ierr
  print *,'Example created on purpose to make CADNA fail'
  print *,'The same result r is computed for a number of iterations chosen by the user'
  print *,'The exact result is 1.4E-10'
  print *,'Enter the number of iterations'
  read *,nloop
  ierr = 0
  do i=1,nloop
    x=6.83561d+05
    y=6.83560d+05
    z=1.00000000007d0
    r = z - x
    r1 = z - y
    r = r + y
    r1 = r1 + x
    r1 = r1 - 2
    r = r + r1
    !      r = ((z-x)+y) + ((z-y)+x-2)
    if(r.ne.1.4d-10) then 
        ierr = ierr + 1
     endif
  enddo
  print *, 'Last value of r: ', r
  print *, 'Number of iterations when r is different from the exact result:', ierr 
end program ex7
