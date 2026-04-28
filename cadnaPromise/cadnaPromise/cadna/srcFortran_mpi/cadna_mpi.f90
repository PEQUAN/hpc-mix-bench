!Copyright 2019 R. CARPENTIER, P. CORDE, F. JEZEQUEL, J.-L. LAMOTTE
!
!This file is part of CADNA.
!
!    CADNA is free software: you can redistribute it and/or modify it
!    under the terms of the GNU Lesser General Public License as
!    published by the Free Software Foundation, either version 3 of the
!    License, or (at your option) any later version.
!
!    CADNA is distributed in the hope that it will be useful, but WITHOUT
!    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
!    or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General
!    Public License for more details.
!
!    You should have received a copy of the GNU Lesser General Public
!    License along with CADNA. If not, see
!    <http://www.gnu.org/licenses/>.
!


! Module d'interfaçage MPI


module cadna_mpi
  use ISO_C_BINDING, only : C_INT

  integer(C_INT), bind(C,name="MPIF_SINGLE_ST")     :: MPIF_SINGLE_ST
  integer(C_INT), bind(C,name="MPIF_DOUBLE_ST")     :: MPIF_DOUBLE_ST

  integer(C_INT), bind(C,name="MPIF_CADNA_SUM_SP")  :: MPIF_CADNA_SUM_SP
  integer(C_INT), bind(C,name="MPIF_CADNA_SUM_DP")  :: MPIF_CADNA_SUM_DP
  integer(C_INT), bind(C,name="MPIF_CADNA_PROD_SP") :: MPIF_CADNA_PROD_SP
  integer(C_INT), bind(C,name="MPIF_CADNA_PROD_DP") :: MPIF_CADNA_PROD_DP
  integer(C_INT), bind(C,name="MPIF_CADNA_MIN_SP")  :: MPIF_CADNA_MIN_SP
  integer(C_INT), bind(C,name="MPIF_CADNA_MIN_DP")  :: MPIF_CADNA_MIN_DP
  integer(C_INT), bind(C,name="MPIF_CADNA_MAX_SP")  :: MPIF_CADNA_MAX_SP
  integer(C_INT), bind(C,name="MPIF_CADNA_MAX_DP")  :: MPIF_CADNA_MAX_DP

  integer(C_INT), bind(C) :: f_cadna_comm

  interface
    subroutine cadna_mpi_init( rank, max_nb_instability ) bind(C, name="cpp_cadna_mpi_init")
      import C_INT
      integer(C_INT), intent(in) :: rank
      integer(C_INT), value      :: max_nb_instability
    end subroutine cadna_mpi_init
    subroutine cadna_mpi_end() bind (C, name="cpp_cadna_mpi_end")
    end subroutine cadna_mpi_end
  end interface
end module cadna_mpi

