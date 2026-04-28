!Copyright 2017 R. CARPENTIER, P. CORDE, F. JEZEQUEL, J.-L. LAMOTTE
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


module cadna_types
  use ISO_C_BINDING, only : C_INT, C_FLOAT, C_DOUBLE
  type, BIND(C) :: single_st
    real(kind=C_FLOAT)  :: x=0._C_FLOAT
    real(kind=C_FLOAT)  :: y=0._C_FLOAT
    real(kind=C_FLOAT)  :: z=0._C_FLOAT
    integer(kind=C_INT) :: accuracy=7_C_INT
  end type single_st
  type, BIND(C) :: double_st
    real(kind=C_DOUBLE) :: x=0._C_DOUBLE
    real(kind=C_DOUBLE) :: y=0._C_DOUBLE
    real(kind=C_DOUBLE) :: z=0._C_DOUBLE
    integer(kind=C_INT) :: accuracy=15_C_INT
    integer(kind=C_INT) :: pad=0_C_INT
  end type double_st

  type complex_single_st
    type(single_st) :: x
    type(single_st) :: y
  end type complex_single_st
  type complex_double_st
    type(double_st) :: x
    type(double_st) :: y
 end type complex_double_st

 integer, parameter :: &
       & cadna_branching = 8, &
       & cadna_mathematic = 16, &
       & cadna_intrinsic = 32, &
       & cadna_cancellation = 64, &
       & cadna_division = 128, &
       & cadna_power = 256, &
       & cadna_multiplication = 512, &
       & cadna_all = 2048 

 integer, parameter :: DIGIT_NOT_COMPUTED = -1
end module cadna_types
