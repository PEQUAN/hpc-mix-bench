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



! Module de surcharge de l'opérateur "**"




module cadna_cmplx_pow
  use ISO_C_BINDING, only : C_LONG_LONG, C_LONG, C_INT, C_SHORT, C_FLOAT, C_DOUBLE
  use cadna_types
  use cadna_math
  use cadna_cmplx_affect
  use cadna_cmplx_mult
  use cadna_cmplx_math
  implicit none




  interface operator(**)
    module procedure pow_complex_double_st_complex_double_st
  end interface operator(**)

  interface operator(**)
    module procedure pow_complex_double_st_complex_single_st
  end interface operator(**)

  interface operator(**)
    module procedure pow_complex_double_st_double_st
  end interface operator(**)

  interface operator(**)
    module procedure pow_complex_double_st_single_st
  end interface operator(**)

  interface operator(**)
    module procedure pow_complex_double_st_double
  end interface operator(**)

  interface operator(**)
    module procedure pow_complex_double_st_float
  end interface operator(**)

  interface operator(**)
    module procedure pow_complex_double_st_long_long
  end interface operator(**)

!INTERFACE_OP(**, pow, complex_double_st, long)
  interface operator(**)
    module procedure pow_complex_double_st_int
  end interface operator(**)

  interface operator(**)
    module procedure pow_complex_double_st_short
  end interface operator(**)

!
  interface operator(**)
    module procedure pow_complex_single_st_complex_double_st
  end interface operator(**)

  interface operator(**)
    module procedure pow_complex_single_st_complex_single_st
  end interface operator(**)

  interface operator(**)
    module procedure pow_complex_single_st_double_st
  end interface operator(**)

  interface operator(**)
    module procedure pow_complex_single_st_single_st
  end interface operator(**)

  interface operator(**)
    module procedure pow_complex_single_st_double
  end interface operator(**)

  interface operator(**)
    module procedure pow_complex_single_st_float
  end interface operator(**)

  interface operator(**)
    module procedure pow_complex_single_st_long_long
  end interface operator(**)

!INTERFACE_OP(**, pow, complex_single_st, long)
  interface operator(**)
    module procedure pow_complex_single_st_int
  end interface operator(**)

  interface operator(**)
    module procedure pow_complex_single_st_short
  end interface operator(**)

!
  interface operator(**)
    module procedure pow_single_st_complex_double_st
  end interface operator(**)

  interface operator(**)
    module procedure pow_single_st_complex_single_st
  end interface operator(**)


  interface operator(**)
    module procedure pow_double_st_complex_double_st
  end interface operator(**)

  interface operator(**)
    module procedure pow_double_st_complex_single_st
  end interface operator(**)


  interface operator(**)
    module procedure pow_double_complex_double_st
  end interface operator(**)

  interface operator(**)
    module procedure pow_double_complex_single_st
  end interface operator(**)

!
  interface operator(**)
    module procedure pow_float_complex_double_st
  end interface operator(**)

  interface operator(**)
    module procedure pow_float_complex_single_st
  end interface operator(**)

!
  interface operator(**)
    module procedure pow_long_long_complex_double_st
  end interface operator(**)

  interface operator(**)
    module procedure pow_long_long_complex_single_st
  end interface operator(**)

!
!INTERFACE_OP(**, pow, long,              complex_double_st)
!INTERFACE_OP(**, pow, long,              complex_single_st)
!
  interface operator(**)
    module procedure pow_int_complex_double_st
  end interface operator(**)

  interface operator(**)
    module procedure pow_int_complex_single_st
  end interface operator(**)

!
  interface operator(**)
    module procedure pow_short_complex_double_st
  end interface operator(**)

  interface operator(**)
    module procedure pow_short_complex_single_st
  end interface operator(**)



contains



  elemental function pow_complex_double_st_complex_double_st(a, b)
    type(complex_double_st), intent(in) :: a
    type(complex_double_st), intent(in) :: b
    type(complex_double_st) pow_complex_double_st_complex_double_st

    pow_complex_double_st_complex_double_st = exp(b*log(a))
  end function pow_complex_double_st_complex_double_st

  elemental function pow_complex_double_st_complex_single_st(a, b)
    type(complex_double_st), intent(in) :: a
    type(complex_single_st), intent(in) :: b
    type(complex_double_st) pow_complex_double_st_complex_single_st

    pow_complex_double_st_complex_single_st = exp(b*log(a))
  end function pow_complex_double_st_complex_single_st

  elemental function pow_complex_double_st_double_st(a, b)
    type(complex_double_st), intent(in) :: a
    type(double_st), intent(in) :: b
    type(complex_double_st) pow_complex_double_st_double_st

    pow_complex_double_st_double_st = exp(b*log(a))
  end function pow_complex_double_st_double_st

  elemental function pow_complex_double_st_single_st(a, b)
    type(complex_double_st), intent(in) :: a
    type(single_st), intent(in) :: b
    type(complex_double_st) pow_complex_double_st_single_st

    pow_complex_double_st_single_st = exp(b*log(a))
  end function pow_complex_double_st_single_st

  elemental function pow_complex_double_st_double(a, b)
    type(complex_double_st), intent(in) :: a
    real(C_DOUBLE), intent(in) :: b
    type(complex_double_st) pow_complex_double_st_double

    pow_complex_double_st_double = exp(b*log(a))
  end function pow_complex_double_st_double

  elemental function pow_complex_double_st_float(a, b)
    type(complex_double_st), intent(in) :: a
    real(C_FLOAT), intent(in) :: b
    type(complex_double_st) pow_complex_double_st_float

    pow_complex_double_st_float = exp(b*log(a))
  end function pow_complex_double_st_float

  elemental function pow_complex_double_st_long_long(a, b)
    type(complex_double_st), intent(in) :: a
    integer(C_LONG_LONG), intent(in) :: b
    type(complex_double_st) pow_complex_double_st_long_long

    pow_complex_double_st_long_long = exp(b*log(a))
  end function pow_complex_double_st_long_long

!CONTAINS_OP(pow, complex_double_st, long,              complex_double_st)
  elemental function pow_complex_double_st_int(a, b)
    type(complex_double_st), intent(in) :: a
    integer(C_INT), intent(in) :: b
    type(complex_double_st) pow_complex_double_st_int

    pow_complex_double_st_int = exp(b*log(a))
  end function pow_complex_double_st_int

  elemental function pow_complex_double_st_short(a, b)
    type(complex_double_st), intent(in) :: a
    integer(C_SHORT), intent(in) :: b
    type(complex_double_st) pow_complex_double_st_short

    pow_complex_double_st_short = exp(b*log(a))
  end function pow_complex_double_st_short

!
  elemental function pow_complex_single_st_complex_double_st(a, b)
    type(complex_single_st), intent(in) :: a
    type(complex_double_st), intent(in) :: b
    type(complex_double_st) pow_complex_single_st_complex_double_st

    pow_complex_single_st_complex_double_st = exp(b*log(a))
  end function pow_complex_single_st_complex_double_st

  elemental function pow_complex_single_st_complex_single_st(a, b)
    type(complex_single_st), intent(in) :: a
    type(complex_single_st), intent(in) :: b
    type(complex_single_st) pow_complex_single_st_complex_single_st

    pow_complex_single_st_complex_single_st = exp(b*log(a))
  end function pow_complex_single_st_complex_single_st

  elemental function pow_complex_single_st_double_st(a, b)
    type(complex_single_st), intent(in) :: a
    type(double_st), intent(in) :: b
    type(complex_double_st) pow_complex_single_st_double_st

    pow_complex_single_st_double_st = exp(b*log(a))
  end function pow_complex_single_st_double_st

  elemental function pow_complex_single_st_single_st(a, b)
    type(complex_single_st), intent(in) :: a
    type(single_st), intent(in) :: b
    type(complex_single_st) pow_complex_single_st_single_st

    pow_complex_single_st_single_st = exp(b*log(a))
  end function pow_complex_single_st_single_st

  elemental function pow_complex_single_st_double(a, b)
    type(complex_single_st), intent(in) :: a
    real(C_DOUBLE), intent(in) :: b
    type(complex_single_st) pow_complex_single_st_double

    pow_complex_single_st_double = exp(b*log(a))
  end function pow_complex_single_st_double

  elemental function pow_complex_single_st_float(a, b)
    type(complex_single_st), intent(in) :: a
    real(C_FLOAT), intent(in) :: b
    type(complex_single_st) pow_complex_single_st_float

    pow_complex_single_st_float = exp(b*log(a))
  end function pow_complex_single_st_float

  elemental function pow_complex_single_st_long_long(a, b)
    type(complex_single_st), intent(in) :: a
    integer(C_LONG_LONG), intent(in) :: b
    type(complex_single_st) pow_complex_single_st_long_long

    pow_complex_single_st_long_long = exp(b*log(a))
  end function pow_complex_single_st_long_long

!CONTAINS_OP(pow, complex_single_st, long,              complex_single_st)
  elemental function pow_complex_single_st_int(a, b)
    type(complex_single_st), intent(in) :: a
    integer(C_INT), intent(in) :: b
    type(complex_single_st) pow_complex_single_st_int

    pow_complex_single_st_int = exp(b*log(a))
  end function pow_complex_single_st_int

  elemental function pow_complex_single_st_short(a, b)
    type(complex_single_st), intent(in) :: a
    integer(C_SHORT), intent(in) :: b
    type(complex_single_st) pow_complex_single_st_short

    pow_complex_single_st_short = exp(b*log(a))
  end function pow_complex_single_st_short

!
  elemental function pow_single_st_complex_double_st(a, b)
    type(single_st), intent(in) :: a
    type(complex_double_st), intent(in) :: b
    type(complex_double_st) pow_single_st_complex_double_st

    pow_single_st_complex_double_st = exp(b*log(a))
  end function pow_single_st_complex_double_st

  elemental function pow_single_st_complex_single_st(a, b)
    type(single_st), intent(in) :: a
    type(complex_single_st), intent(in) :: b
    type(complex_single_st) pow_single_st_complex_single_st

    pow_single_st_complex_single_st = exp(b*log(a))
  end function pow_single_st_complex_single_st


  elemental function pow_double_st_complex_double_st(a, b)
    type(double_st), intent(in) :: a
    type(complex_double_st), intent(in) :: b
    type(complex_double_st) pow_double_st_complex_double_st

    pow_double_st_complex_double_st = exp(b*log(a))
  end function pow_double_st_complex_double_st

  elemental function pow_double_st_complex_single_st(a, b)
    type(double_st), intent(in) :: a
    type(complex_single_st), intent(in) :: b
    type(complex_double_st) pow_double_st_complex_single_st

    pow_double_st_complex_single_st = exp(b*log(a))
  end function pow_double_st_complex_single_st


  elemental function pow_double_complex_double_st(a, b)
    real(C_DOUBLE), intent(in) :: a
    type(complex_double_st), intent(in) :: b
    type(complex_double_st) pow_double_complex_double_st

    pow_double_complex_double_st = exp(b*log(a))
  end function pow_double_complex_double_st

  elemental function pow_double_complex_single_st(a, b)
    real(C_DOUBLE), intent(in) :: a
    type(complex_single_st), intent(in) :: b
    type(complex_single_st) pow_double_complex_single_st

    pow_double_complex_single_st = exp(b*log(a))
  end function pow_double_complex_single_st

!
  elemental function pow_float_complex_double_st(a, b)
    real(C_FLOAT), intent(in) :: a
    type(complex_double_st), intent(in) :: b
    type(complex_double_st) pow_float_complex_double_st

    pow_float_complex_double_st = exp(b*log(a))
  end function pow_float_complex_double_st

  elemental function pow_float_complex_single_st(a, b)
    real(C_FLOAT), intent(in) :: a
    type(complex_single_st), intent(in) :: b
    type(complex_single_st) pow_float_complex_single_st

    pow_float_complex_single_st = exp(b*log(a))
  end function pow_float_complex_single_st

!
  elemental function pow_long_long_complex_double_st(a, b)
    integer(C_LONG_LONG), intent(in) :: a
    type(complex_double_st), intent(in) :: b
    type(complex_double_st) pow_long_long_complex_double_st

    pow_long_long_complex_double_st = exp(b*log(real(a)))
  end function pow_long_long_complex_double_st

  elemental function pow_long_long_complex_single_st(a, b)
    integer(C_LONG_LONG), intent(in) :: a
    type(complex_single_st), intent(in) :: b
    type(complex_single_st) pow_long_long_complex_single_st

    pow_long_long_complex_single_st = exp(b*log(real(a)))
  end function pow_long_long_complex_single_st

!
!CONTAINS_OP(pow, long, complex_double_st,             complex_double_st)
!CONTAINS_OP(pow, long, complex_single_st,             complex_single_st)
!
  elemental function pow_int_complex_double_st(a, b)
    integer(C_INT), intent(in) :: a
    type(complex_double_st), intent(in) :: b
    type(complex_double_st) pow_int_complex_double_st

    pow_int_complex_double_st = exp(b*log(real(a)))
  end function pow_int_complex_double_st

  elemental function pow_int_complex_single_st(a, b)
    integer(C_INT), intent(in) :: a
    type(complex_single_st), intent(in) :: b
    type(complex_single_st) pow_int_complex_single_st

    pow_int_complex_single_st = exp(b*log(real(a)))
  end function pow_int_complex_single_st

!
  elemental function pow_short_complex_double_st(a, b)
    integer(C_SHORT), intent(in) :: a
    type(complex_double_st), intent(in) :: b
    type(complex_double_st) pow_short_complex_double_st

    pow_short_complex_double_st = exp(b*log(real(a)))
  end function pow_short_complex_double_st

  elemental function pow_short_complex_single_st(a, b)
    integer(C_SHORT), intent(in) :: a
    type(complex_single_st), intent(in) :: b
    type(complex_single_st) pow_short_complex_single_st

    pow_short_complex_single_st = exp(b*log(real(a)))
  end function pow_short_complex_single_st




end module cadna_cmplx_pow

