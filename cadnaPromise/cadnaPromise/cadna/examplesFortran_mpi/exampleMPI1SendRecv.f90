program exampleMPI1
  use ISO_FORTRAN_ENV, only : REAL64, OUTPUT_UNIT, ERROR_UNIT
  
  use mpi
  
  integer status(mpi_status_size)
  integer nb_procs, rank, code
  REAL(kind=REAL64) x, y
  REAL(kind=REAL64) res, tmp

  ! MPI Initialization
  call MPI_Init(code)
  call MPI_Comm_rank(MPI_COMM_WORLD, rank, code)
  call MPI_Comm_size(MPI_COMM_WORLD, nb_procs, code)

  if( rank == 0 ) then
     write( ERROR_UNIT, '(a)' ) "I am the Master"
     write( ERROR_UNIT, '(a)' ) "Computation with MPI without CADNA:"
  end if
  if( nb_procs /= 4 ) then
    if( rank == 0 ) write( ERROR_UNIT, '(a)' ) "4 processes are necessary"
    call MPI_Finalize(code)
    CALL MPI_ABORT ( MPI_COMM_WORLD, 4, code)
  end if
  
  x=10864.d0; y=18817.d0
  if ( rank /= 0 ) then
    select case(rank)
      case (1)
        res=9.d0*x*x*x*x
      case (2)
        res=- y*y*y*y
      case (3)
        res= 2.d0*y*y
    end select
    write( OUTPUT_UNIT, '(a, i0, a, e22.15)' ) "I am thread ", rank, " and I send ", res
    call MPI_Send( res, 1, MPI_DOUBLE_PRECISION, 0, 0, MPI_COMM_WORLD, code)
    
  else
    res=0.d0
    do i=1,3
      call MPI_Recv( tmp, 1, MPI_DOUBLE_PRECISION, MPI_ANY_SOURCE, 0, & 
                     MPI_COMM_WORLD, status, code)
      res = res + tmp
      write( OUTPUT_UNIT, '(a, e22.15)' ) "I have received ", tmp
    end do
    write( OUTPUT_UNIT, '(a, e22.15)' ) "Final result: ", res
    write( OUTPUT_UNIT, '(a)' )  "Exact result:  1.000000000000000e+00"
  end if
  
  call MPI_Finalize(code)
end program exampleMPI1
