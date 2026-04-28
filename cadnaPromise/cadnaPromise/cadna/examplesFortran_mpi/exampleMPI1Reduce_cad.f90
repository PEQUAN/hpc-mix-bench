program exampleMPI1_cad
  use ISO_FORTRAN_ENV, only : OUTPUT_UNIT, ERROR_UNIT
  use mpi
  use cadna
  use cadna_mpi
  implicit none


  integer nb_procs, rank, code
  type(double_st) x, y
  type(double_st) res, somme

  ! MPI Initialization
  call MPI_Init(code)
  call MPI_Comm_rank(MPI_COMM_WORLD, rank, code)
  call MPI_Comm_size(MPI_COMM_WORLD, nb_procs, code)

  call cadna_mpi_init(rank, -1)

  if( rank == 0 ) then
     write( ERROR_UNIT, '(a)' ) "I am the Master"
     write( ERROR_UNIT, '(a)' ) "Computation with MPI using CADNA:"
  end if
  if( nb_procs /= 4 ) then
    if( rank == 0 ) write( ERROR_UNIT, '(a)' ) "4 processes are necessary"
    !CALL MPI_ABORT ( MPI_COMM_WORLD, 4, code)
    call MPI_Finalize(code)
    stop 4
  end if
  
  x=10864.d0; y=18817.d0
  res=0.d0 ; somme = 0.d0
  select case(rank)
    case (1)
      res=9.d0*x*x*x*x
    case (2)
      res=- y*y*y*y
    case (3)
      res= 2.d0*y*y
  end select
  write( OUTPUT_UNIT, '(a, i0, 2a)' ) "I am thread ", rank, " and I have res = ", str(res)
  call MPI_Reduce( res, somme, 1, MPIF_DOUBLE_ST, MPIF_CADNA_SUM_DP, 0, MPI_COMM_WORLD, code)
  if ( rank == 0 ) then
    write( OUTPUT_UNIT, '(2a)' ) "-->Final result: ", str(somme)
    write( OUTPUT_UNIT, '(a)' )  "Exact result:  1.000000000000000e+00"
  end if
  
  call cadna_mpi_end
  call MPI_Finalize(code)
end program exampleMPI1_cad
