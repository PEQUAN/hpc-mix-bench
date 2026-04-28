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



! Module de surcharge de l'opérateur "=="




module cadna_cmplx_eq
  use ISO_C_BINDING, only : C_BOOL, C_LONG_LONG, C_LONG, C_INT, C_SHORT, C_FLOAT, C_DOUBLE
  !`use ISO_C_BINDING, only : C_BOOL, C_LONG_LONG, C_LONG, C_INT, C_SHORT, C_FLOAT'
  use cadna_types
  use cadna_eq
  implicit none

!INIT(eq, C_LONG_LONG, C_LONG, C_INT, C_SHORT, C_FLOAT, C_DOUBLE)



  interface operator(==)
    module procedure eq_complex_double_st_complex_double_st
  end interface operator(==)

  interface operator(==)
    module procedure eq_complex_double_st_complex_single_st
  end interface operator(==)

  interface operator(==)
    module procedure eq_complex_double_st_double_st
  end interface operator(==)

  interface operator(==)
    module procedure eq_complex_double_st_single_st
  end interface operator(==)

  interface operator(==)
    module procedure eq_complex_double_st_double
  end interface operator(==)

  interface operator(==)
    module procedure eq_complex_double_st_float
  end interface operator(==)

  interface operator(==)
    module procedure eq_complex_double_st_long_long
  end interface operator(==)

!INTERFACE_OP(==, eq, complex_double_st, long)
  interface operator(==)
    module procedure eq_complex_double_st_int
  end interface operator(==)

  interface operator(==)
    module procedure eq_complex_double_st_short
  end interface operator(==)

!
  interface operator(==)
    module procedure eq_complex_single_st_complex_double_st
  end interface operator(==)

  interface operator(==)
    module procedure eq_complex_single_st_complex_single_st
  end interface operator(==)

  interface operator(==)
    module procedure eq_complex_single_st_double_st
  end interface operator(==)

  interface operator(==)
    module procedure eq_complex_single_st_single_st
  end interface operator(==)

  interface operator(==)
    module procedure eq_complex_single_st_double
  end interface operator(==)

  interface operator(==)
    module procedure eq_complex_single_st_float
  end interface operator(==)

  interface operator(==)
    module procedure eq_complex_single_st_long_long
  end interface operator(==)

!INTERFACE_OP(==, eq, complex_single_st, long)
  interface operator(==)
    module procedure eq_complex_single_st_int
  end interface operator(==)

  interface operator(==)
    module procedure eq_complex_single_st_short
  end interface operator(==)

!
  interface operator(==)
    module procedure eq_single_st_complex_double_st
  end interface operator(==)

  interface operator(==)
    module procedure eq_single_st_complex_single_st
  end interface operator(==)


  interface operator(==)
    module procedure eq_double_st_complex_double_st
  end interface operator(==)

  interface operator(==)
    module procedure eq_double_st_complex_single_st
  end interface operator(==)


  interface operator(==)
    module procedure eq_double_complex_double_st
  end interface operator(==)

  interface operator(==)
    module procedure eq_double_complex_single_st
  end interface operator(==)

!
  interface operator(==)
    module procedure eq_float_complex_double_st
  end interface operator(==)

  interface operator(==)
    module procedure eq_float_complex_single_st
  end interface operator(==)

!
  interface operator(==)
    module procedure eq_long_long_complex_double_st
  end interface operator(==)

  interface operator(==)
    module procedure eq_long_long_complex_single_st
  end interface operator(==)

!
!INTERFACE_OP(==, eq, long,              complex_double_st)
!INTERFACE_OP(==, eq, long,              complex_single_st)
!
  interface operator(==)
    module procedure eq_int_complex_double_st
  end interface operator(==)

  interface operator(==)
    module procedure eq_int_complex_single_st
  end interface operator(==)

!
  interface operator(==)
    module procedure eq_short_complex_double_st
  end interface operator(==)

  interface operator(==)
    module procedure eq_short_complex_single_st
  end interface operator(==)


contains



  elemental function eq_complex_double_st_complex_double_st(a, b)
    type(complex_double_st), intent(in) :: a
    type(complex_double_st), intent(in) :: b
    !logical(C_BOOL) eq_complex_double_st_complex_double_st
    logical eq_complex_double_st_complex_double_st

    eq_complex_double_st_complex_double_st = a%x == b%x .AND. a%y == b%y
  end function eq_complex_double_st_complex_double_st

  elemental function eq_complex_double_st_complex_single_st(a, b)
    type(complex_double_st), intent(in) :: a
    type(complex_single_st), intent(in) :: b
    !logical(C_BOOL) eq_complex_double_st_complex_single_st
    logical eq_complex_double_st_complex_single_st

    eq_complex_double_st_complex_single_st = a%x == b%x .AND. a%y == b%y
  end function eq_complex_double_st_complex_single_st

  elemental function eq_complex_double_st_double_st(a, b)
    type(complex_double_st), intent(in) :: a
    type(double_st), intent(in) :: b
    !logical(C_BOOL) eq_complex_double_st_double_st
    logical eq_complex_double_st_double_st

    eq_complex_double_st_double_st = a%x == b .AND. a%y == 0.
  end function eq_complex_double_st_double_st

  elemental function eq_complex_double_st_single_st(a, b)
    type(complex_double_st), intent(in) :: a
    type(single_st), intent(in) :: b
    !logical(C_BOOL) eq_complex_double_st_single_st
    logical eq_complex_double_st_single_st

    eq_complex_double_st_single_st = a%x == b .AND. a%y == 0.
  end function eq_complex_double_st_single_st

  elemental function eq_complex_double_st_double(a, b)
    type(complex_double_st), intent(in) :: a
    real(C_DOUBLE), intent(in) :: b
    !logical(C_BOOL) eq_complex_double_st_double
    logical eq_complex_double_st_double

    eq_complex_double_st_double = a%x == b .AND. a%y == 0.
  end function eq_complex_double_st_double

  elemental function eq_complex_double_st_float(a, b)
    type(complex_double_st), intent(in) :: a
    real(C_FLOAT), intent(in) :: b
    !logical(C_BOOL) eq_complex_double_st_float
    logical eq_complex_double_st_float

    eq_complex_double_st_float = a%x == b .AND. a%y == 0.
  end function eq_complex_double_st_float

  elemental function eq_complex_double_st_long_long(a, b)
    type(complex_double_st), intent(in) :: a
    integer(C_LONG_LONG), intent(in) :: b
    !logical(C_BOOL) eq_complex_double_st_long_long
    logical eq_complex_double_st_long_long

    eq_complex_double_st_long_long = a%x == b .AND. a%y == 0.
  end function eq_complex_double_st_long_long

!CONTAINS_OP(eq, complex_double_st, long,              complex_double_st)
  elemental function eq_complex_double_st_int(a, b)
    type(complex_double_st), intent(in) :: a
    integer(C_INT), intent(in) :: b
    !logical(C_BOOL) eq_complex_double_st_int
    logical eq_complex_double_st_int

    eq_complex_double_st_int = a%x == b .AND. a%y == 0.
  end function eq_complex_double_st_int

  elemental function eq_complex_double_st_short(a, b)
    type(complex_double_st), intent(in) :: a
    integer(C_SHORT), intent(in) :: b
    !logical(C_BOOL) eq_complex_double_st_short
    logical eq_complex_double_st_short

    eq_complex_double_st_short = a%x == b .AND. a%y == 0.
  end function eq_complex_double_st_short

!
  elemental function eq_complex_single_st_complex_double_st(a, b)
    type(complex_single_st), intent(in) :: a
    type(complex_double_st), intent(in) :: b
    !logical(C_BOOL) eq_complex_single_st_complex_double_st
    logical eq_complex_single_st_complex_double_st

    eq_complex_single_st_complex_double_st = a%x == b%x .AND. a%y == b%y
  end function eq_complex_single_st_complex_double_st

  elemental function eq_complex_single_st_complex_single_st(a, b)
    type(complex_single_st), intent(in) :: a
    type(complex_single_st), intent(in) :: b
    !logical(C_BOOL) eq_complex_single_st_complex_single_st
    logical eq_complex_single_st_complex_single_st

    eq_complex_single_st_complex_single_st = a%x == b%x .AND. a%y == b%y
  end function eq_complex_single_st_complex_single_st

  elemental function eq_complex_single_st_double_st(a, b)
    type(complex_single_st), intent(in) :: a
    type(double_st), intent(in) :: b
    !logical(C_BOOL) eq_complex_single_st_double_st
    logical eq_complex_single_st_double_st

    eq_complex_single_st_double_st = a%x == b .AND. a%y == 0.
  end function eq_complex_single_st_double_st

  elemental function eq_complex_single_st_single_st(a, b)
    type(complex_single_st), intent(in) :: a
    type(single_st), intent(in) :: b
    !logical(C_BOOL) eq_complex_single_st_single_st
    logical eq_complex_single_st_single_st

    eq_complex_single_st_single_st = a%x == b .AND. a%y == 0.
  end function eq_complex_single_st_single_st

  elemental function eq_complex_single_st_double(a, b)
    type(complex_single_st), intent(in) :: a
    real(C_DOUBLE), intent(in) :: b
    !logical(C_BOOL) eq_complex_single_st_double
    logical eq_complex_single_st_double

    eq_complex_single_st_double = a%x == b .AND. a%y == 0.
  end function eq_complex_single_st_double

  elemental function eq_complex_single_st_float(a, b)
    type(complex_single_st), intent(in) :: a
    real(C_FLOAT), intent(in) :: b
    !logical(C_BOOL) eq_complex_single_st_float
    logical eq_complex_single_st_float

    eq_complex_single_st_float = a%x == b .AND. a%y == 0.
  end function eq_complex_single_st_float

  elemental function eq_complex_single_st_long_long(a, b)
    type(complex_single_st), intent(in) :: a
    integer(C_LONG_LONG), intent(in) :: b
    !logical(C_BOOL) eq_complex_single_st_long_long
    logical eq_complex_single_st_long_long

    eq_complex_single_st_long_long = a%x == b .AND. a%y == 0.
  end function eq_complex_single_st_long_long

!CONTAINS_OP(eq, complex_single_st, long,              complex_single_st)
  elemental function eq_complex_single_st_int(a, b)
    type(complex_single_st), intent(in) :: a
    integer(C_INT), intent(in) :: b
    !logical(C_BOOL) eq_complex_single_st_int
    logical eq_complex_single_st_int

    eq_complex_single_st_int = a%x == b .AND. a%y == 0.
  end function eq_complex_single_st_int

  elemental function eq_complex_single_st_short(a, b)
    type(complex_single_st), intent(in) :: a
    integer(C_SHORT), intent(in) :: b
    !logical(C_BOOL) eq_complex_single_st_short
    logical eq_complex_single_st_short

    eq_complex_single_st_short = a%x == b .AND. a%y == 0.
  end function eq_complex_single_st_short

!
  elemental function eq_single_st_complex_double_st(a, b)
    type(single_st), intent(in) :: a
    type(complex_double_st), intent(in) :: b
    !logical(C_BOOL) eq_single_st_complex_double_st
    logical eq_single_st_complex_double_st

    eq_single_st_complex_double_st = b%x == a .AND. b%y == 0.
  end function eq_single_st_complex_double_st

  elemental function eq_single_st_complex_single_st(a, b)
    type(single_st), intent(in) :: a
    type(complex_single_st), intent(in) :: b
    !logical(C_BOOL) eq_single_st_complex_single_st
    logical eq_single_st_complex_single_st

    eq_single_st_complex_single_st = b%x == a .AND. b%y == 0.
  end function eq_single_st_complex_single_st


  elemental function eq_double_st_complex_double_st(a, b)
    type(double_st), intent(in) :: a
    type(complex_double_st), intent(in) :: b
    !logical(C_BOOL) eq_double_st_complex_double_st
    logical eq_double_st_complex_double_st

    eq_double_st_complex_double_st = b%x == a .AND. b%y == 0.
  end function eq_double_st_complex_double_st

  elemental function eq_double_st_complex_single_st(a, b)
    type(double_st), intent(in) :: a
    type(complex_single_st), intent(in) :: b
    !logical(C_BOOL) eq_double_st_complex_single_st
    logical eq_double_st_complex_single_st

    eq_double_st_complex_single_st = b%x == a .AND. b%y == 0.
  end function eq_double_st_complex_single_st


  elemental function eq_double_complex_double_st(a, b)
    real(C_DOUBLE), intent(in) :: a
    type(complex_double_st), intent(in) :: b
    !logical(C_BOOL) eq_double_complex_double_st
    logical eq_double_complex_double_st

    eq_double_complex_double_st = b%x == a .AND. b%y == 0.
  end function eq_double_complex_double_st

  elemental function eq_double_complex_single_st(a, b)
    real(C_DOUBLE), intent(in) :: a
    type(complex_single_st), intent(in) :: b
    !logical(C_BOOL) eq_double_complex_single_st
    logical eq_double_complex_single_st

    eq_double_complex_single_st = b%x == a .AND. b%y == 0.
  end function eq_double_complex_single_st

!
  elemental function eq_float_complex_double_st(a, b)
    real(C_FLOAT), intent(in) :: a
    type(complex_double_st), intent(in) :: b
    !logical(C_BOOL) eq_float_complex_double_st
    logical eq_float_complex_double_st

    eq_float_complex_double_st = b%x == a .AND. b%y == 0.
  end function eq_float_complex_double_st

  elemental function eq_float_complex_single_st(a, b)
    real(C_FLOAT), intent(in) :: a
    type(complex_single_st), intent(in) :: b
    !logical(C_BOOL) eq_float_complex_single_st
    logical eq_float_complex_single_st

    eq_float_complex_single_st = b%x == a .AND. b%y == 0.
  end function eq_float_complex_single_st

!
  elemental function eq_long_long_complex_double_st(a, b)
    integer(C_LONG_LONG), intent(in) :: a
    type(complex_double_st), intent(in) :: b
    !logical(C_BOOL) eq_long_long_complex_double_st
    logical eq_long_long_complex_double_st

    eq_long_long_complex_double_st = b%x == a .AND. b%y == 0.
  end function eq_long_long_complex_double_st

  elemental function eq_long_long_complex_single_st(a, b)
    integer(C_LONG_LONG), intent(in) :: a
    type(complex_single_st), intent(in) :: b
    !logical(C_BOOL) eq_long_long_complex_single_st
    logical eq_long_long_complex_single_st

    eq_long_long_complex_single_st = b%x == a .AND. b%y == 0.
  end function eq_long_long_complex_single_st

!
!CONTAINS_OP(eq, long,              complex_double_st,         complex_double_st)
!CONTAINS_OP(eq, long,              complex_single_st,         complex_single_st)
!
  elemental function eq_int_complex_double_st(a, b)
    integer(C_INT), intent(in) :: a
    type(complex_double_st), intent(in) :: b
    !logical(C_BOOL) eq_int_complex_double_st
    logical eq_int_complex_double_st

    eq_int_complex_double_st = b%x == a .AND. b%y == 0.
  end function eq_int_complex_double_st

  elemental function eq_int_complex_single_st(a, b)
    integer(C_INT), intent(in) :: a
    type(complex_single_st), intent(in) :: b
    !logical(C_BOOL) eq_int_complex_single_st
    logical eq_int_complex_single_st

    eq_int_complex_single_st = b%x == a .AND. b%y == 0.
  end function eq_int_complex_single_st

!
  elemental function eq_short_complex_double_st(a, b)
    integer(C_SHORT), intent(in) :: a
    type(complex_double_st), intent(in) :: b
    !logical(C_BOOL) eq_short_complex_double_st
    logical eq_short_complex_double_st

    eq_short_complex_double_st = b%x == a .AND. b%y == 0.
  end function eq_short_complex_double_st

  elemental function eq_short_complex_single_st(a, b)
    integer(C_SHORT), intent(in) :: a
    type(complex_single_st), intent(in) :: b
    !logical(C_BOOL) eq_short_complex_single_st
    logical eq_short_complex_single_st

    eq_short_complex_single_st = b%x == a .AND. b%y == 0.
  end function eq_short_complex_single_st




end module cadna_cmplx_eq

