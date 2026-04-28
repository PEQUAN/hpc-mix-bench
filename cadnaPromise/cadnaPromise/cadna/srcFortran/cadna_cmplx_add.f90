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


! Module de surcharge de l'opérateur "+"




module cadna_cmplx_add
  use ISO_C_BINDING,   only : C_LONG_LONG, C_LONG, C_INT, C_SHORT, C_FLOAT, C_DOUBLE
  use ISO_FORTRAN_ENV, only : REAL32, REAL64
  use cadna_types
  use cadna_add
  use cadna_affect
  implicit none




  interface operator(+)
    module procedure add_complex_double_st_complex_double_st
  end interface operator(+)

  interface operator(+)
    module procedure add_complex_double_st_complex_single_st
  end interface operator(+)

  interface operator(+)
    module procedure add_complex_double_st_double_st
  end interface operator(+)

  interface operator(+)
    module procedure add_complex_double_st_single_st
  end interface operator(+)

  interface operator(+)
    module procedure add_complex_double_st_complex_float
  end interface operator(+)

  interface operator(+)
    module procedure add_complex_double_st_complex_double
  end interface operator(+)

  interface operator(+)
    module procedure add_complex_double_st_double
  end interface operator(+)

  interface operator(+)
    module procedure add_complex_double_st_float
  end interface operator(+)

  interface operator(+)
    module procedure add_complex_double_st_long_long
  end interface operator(+)

!INTERFACE_OP(+, add, complex_double_st, long)
  interface operator(+)
    module procedure add_complex_double_st_int
  end interface operator(+)

  interface operator(+)
    module procedure add_complex_double_st_short
  end interface operator(+)

!
  interface operator(+)
    module procedure add_complex_single_st_complex_double_st
  end interface operator(+)

  interface operator(+)
    module procedure add_complex_single_st_complex_single_st
  end interface operator(+)

  interface operator(+)
    module procedure add_complex_single_st_double_st
  end interface operator(+)

  interface operator(+)
    module procedure add_complex_single_st_single_st
  end interface operator(+)

  interface operator(+)
    module procedure add_complex_single_st_complex_float
  end interface operator(+)

  interface operator(+)
    module procedure add_complex_single_st_complex_double
  end interface operator(+)

  interface operator(+)
    module procedure add_complex_single_st_double
  end interface operator(+)

  interface operator(+)
    module procedure add_complex_single_st_float
  end interface operator(+)

  interface operator(+)
    module procedure add_complex_single_st_long_long
  end interface operator(+)

!INTERFACE_OP(+, add, complex_single_st, long)
  interface operator(+)
    module procedure add_complex_single_st_int
  end interface operator(+)

  interface operator(+)
    module procedure add_complex_single_st_short
  end interface operator(+)

!
  interface operator(+)
    module procedure add_single_st_complex_double_st
  end interface operator(+)

  interface operator(+)
    module procedure add_single_st_complex_single_st
  end interface operator(+)


  interface operator(+)
    module procedure add_double_st_complex_double_st
  end interface operator(+)

  interface operator(+)
    module procedure add_double_st_complex_single_st
  end interface operator(+)


  interface operator(+)
    module procedure add_complex_float_complex_single_st
  end interface operator(+)

  interface operator(+)
    module procedure add_complex_float_complex_double_st
  end interface operator(+)


  interface operator(+)
    module procedure add_complex_double_complex_single_st
  end interface operator(+)

  interface operator(+)
    module procedure add_complex_double_complex_double_st
  end interface operator(+)

!
  interface operator(+)
    module procedure add_double_complex_double_st
  end interface operator(+)

  interface operator(+)
    module procedure add_double_complex_single_st
  end interface operator(+)

!
  interface operator(+)
    module procedure add_float_complex_double_st
  end interface operator(+)

  interface operator(+)
    module procedure add_float_complex_single_st
  end interface operator(+)

!
  interface operator(+)
    module procedure add_long_long_complex_double_st
  end interface operator(+)

  interface operator(+)
    module procedure add_long_long_complex_single_st
  end interface operator(+)

!
!INTERFACE_OP(+, add, long,              complex_double_st)
!INTERFACE_OP(+, add, long,              complex_single_st)
!
  interface operator(+)
    module procedure add_int_complex_double_st
  end interface operator(+)

  interface operator(+)
    module procedure add_int_complex_single_st
  end interface operator(+)

!
  interface operator(+)
    module procedure add_short_complex_double_st
  end interface operator(+)

  interface operator(+)
    module procedure add_short_complex_single_st
  end interface operator(+)



contains



  elemental function add_complex_double_st_complex_double_st(a, b)
    type(complex_double_st), intent(in) :: a
    type(complex_double_st), intent(in) :: b
    type(complex_double_st) add_complex_double_st_complex_double_st
     
     

    add_complex_double_st_complex_double_st%x = a%x+b%x
    add_complex_double_st_complex_double_st%y = a%y+b%y 
  end function add_complex_double_st_complex_double_st

  elemental function add_complex_double_st_complex_single_st(a, b)
    type(complex_double_st), intent(in) :: a
    type(complex_single_st), intent(in) :: b
    type(complex_double_st) add_complex_double_st_complex_single_st
     
     

    add_complex_double_st_complex_single_st%x = a%x+b%x
    add_complex_double_st_complex_single_st%y = a%y+b%y 
  end function add_complex_double_st_complex_single_st

  elemental function add_complex_double_st_double_st(a, b)
    type(complex_double_st), intent(in) :: a
    type(double_st), intent(in) :: b
    type(complex_double_st) add_complex_double_st_double_st
     
     

    add_complex_double_st_double_st%x = a%x+b
    add_complex_double_st_double_st%y = a%y
  end function add_complex_double_st_double_st

  elemental function add_complex_double_st_single_st(a, b)
    type(complex_double_st), intent(in) :: a
    type(single_st), intent(in) :: b
    type(complex_double_st) add_complex_double_st_single_st
     
     

    add_complex_double_st_single_st%x = a%x+b
    add_complex_double_st_single_st%y = a%y
  end function add_complex_double_st_single_st

  elemental function add_complex_double_st_complex_float(a, b)
    type(complex_double_st), intent(in) :: a
    complex(REAL32), intent(in) :: b
    type(complex_double_st) add_complex_double_st_complex_float
     
    real(REAL32) re, im

    re = real(b) ; im = aimag(b)
    add_complex_double_st_complex_float%x = a%x+re
    add_complex_double_st_complex_float%y = a%y+im
  end function add_complex_double_st_complex_float

  elemental function add_complex_double_st_complex_double(a, b)
    type(complex_double_st), intent(in) :: a
    complex(REAL64), intent(in) :: b
    type(complex_double_st) add_complex_double_st_complex_double
     
    real(REAL64) re, im

    re = real(b) ; im = aimag(b)
    add_complex_double_st_complex_double%x = a%x+re
    add_complex_double_st_complex_double%y = a%y+im
  end function add_complex_double_st_complex_double

  elemental function add_complex_double_st_double(a, b)
    type(complex_double_st), intent(in) :: a
    real(C_DOUBLE), intent(in) :: b
    type(complex_double_st) add_complex_double_st_double
     
     

    add_complex_double_st_double%x = a%x+b
    add_complex_double_st_double%y = a%y
  end function add_complex_double_st_double

  elemental function add_complex_double_st_float(a, b)
    type(complex_double_st), intent(in) :: a
    real(C_FLOAT), intent(in) :: b
    type(complex_double_st) add_complex_double_st_float
     
     

    add_complex_double_st_float%x = a%x+b
    add_complex_double_st_float%y = a%y
  end function add_complex_double_st_float

  elemental function add_complex_double_st_long_long(a, b)
    type(complex_double_st), intent(in) :: a
    integer(C_LONG_LONG), intent(in) :: b
    type(complex_double_st) add_complex_double_st_long_long
     
     

    add_complex_double_st_long_long%x = a%x+b
    add_complex_double_st_long_long%y = a%y
  end function add_complex_double_st_long_long

!CONTAINS_OP(add, complex_double_st, long,              complex_double_st)
  elemental function add_complex_double_st_int(a, b)
    type(complex_double_st), intent(in) :: a
    integer(C_INT), intent(in) :: b
    type(complex_double_st) add_complex_double_st_int
     
     

    add_complex_double_st_int%x = a%x+b
    add_complex_double_st_int%y = a%y
  end function add_complex_double_st_int

  elemental function add_complex_double_st_short(a, b)
    type(complex_double_st), intent(in) :: a
    integer(C_SHORT), intent(in) :: b
    type(complex_double_st) add_complex_double_st_short
     
     

    add_complex_double_st_short%x = a%x+b
    add_complex_double_st_short%y = a%y
  end function add_complex_double_st_short

!
  elemental function add_complex_single_st_complex_double_st(a, b)
    type(complex_single_st), intent(in) :: a
    type(complex_double_st), intent(in) :: b
    type(complex_double_st) add_complex_single_st_complex_double_st
     
     

    add_complex_single_st_complex_double_st%x = a%x+b%x
    add_complex_single_st_complex_double_st%y = a%y+b%y 
  end function add_complex_single_st_complex_double_st

  elemental function add_complex_single_st_complex_single_st(a, b)
    type(complex_single_st), intent(in) :: a
    type(complex_single_st), intent(in) :: b
    type(complex_single_st) add_complex_single_st_complex_single_st
     
     

    add_complex_single_st_complex_single_st%x = a%x+b%x
    add_complex_single_st_complex_single_st%y = a%y+b%y 
  end function add_complex_single_st_complex_single_st

  elemental function add_complex_single_st_double_st(a, b)
    type(complex_single_st), intent(in) :: a
    type(double_st), intent(in) :: b
    type(complex_double_st) add_complex_single_st_double_st
     
     

    add_complex_single_st_double_st%x = a%x+b
    add_complex_single_st_double_st%y = a%y
  end function add_complex_single_st_double_st

  elemental function add_complex_single_st_single_st(a, b)
    type(complex_single_st), intent(in) :: a
    type(single_st), intent(in) :: b
    type(complex_single_st) add_complex_single_st_single_st
     
     

    add_complex_single_st_single_st%x = a%x+b
    add_complex_single_st_single_st%y = a%y
  end function add_complex_single_st_single_st

  elemental function add_complex_single_st_complex_float(a, b)
    type(complex_single_st), intent(in) :: a
    complex(REAL32), intent(in) :: b
    type(complex_single_st) add_complex_single_st_complex_float
     
    real(REAL32) re, im

    re = real(b) ; im = aimag(b)
    add_complex_single_st_complex_float%x = a%x+re
    add_complex_single_st_complex_float%y = a%y+im
  end function add_complex_single_st_complex_float

  elemental function add_complex_single_st_complex_double(a, b)
    type(complex_single_st), intent(in) :: a
    complex(REAL64), intent(in) :: b
    type(complex_single_st) add_complex_single_st_complex_double
     
    real(REAL64) re, im

    re = real(b) ; im = aimag(b)
    add_complex_single_st_complex_double%x = a%x+re
    add_complex_single_st_complex_double%y = a%y+im
  end function add_complex_single_st_complex_double

  elemental function add_complex_single_st_double(a, b)
    type(complex_single_st), intent(in) :: a
    real(C_DOUBLE), intent(in) :: b
    type(complex_single_st) add_complex_single_st_double
     
     

    add_complex_single_st_double%x = a%x+b
    add_complex_single_st_double%y = a%y
  end function add_complex_single_st_double

  elemental function add_complex_single_st_float(a, b)
    type(complex_single_st), intent(in) :: a
    real(C_FLOAT), intent(in) :: b
    type(complex_single_st) add_complex_single_st_float
     
     

    add_complex_single_st_float%x = a%x+b
    add_complex_single_st_float%y = a%y
  end function add_complex_single_st_float

  elemental function add_complex_single_st_long_long(a, b)
    type(complex_single_st), intent(in) :: a
    integer(C_LONG_LONG), intent(in) :: b
    type(complex_single_st) add_complex_single_st_long_long
     
     

    add_complex_single_st_long_long%x = a%x+b
    add_complex_single_st_long_long%y = a%y
  end function add_complex_single_st_long_long

!CONTAINS_OP(add, complex_single_st, long,              complex_single_st)
  elemental function add_complex_single_st_int(a, b)
    type(complex_single_st), intent(in) :: a
    integer(C_INT), intent(in) :: b
    type(complex_single_st) add_complex_single_st_int
     
     

    add_complex_single_st_int%x = a%x+b
    add_complex_single_st_int%y = a%y
  end function add_complex_single_st_int

  elemental function add_complex_single_st_short(a, b)
    type(complex_single_st), intent(in) :: a
    integer(C_SHORT), intent(in) :: b
    type(complex_single_st) add_complex_single_st_short
     
     

    add_complex_single_st_short%x = a%x+b
    add_complex_single_st_short%y = a%y
  end function add_complex_single_st_short

!
  elemental function add_single_st_complex_double_st(a, b)
    type(single_st), intent(in) :: a
    type(complex_double_st), intent(in) :: b
    type(complex_double_st) add_single_st_complex_double_st
     
     

    add_single_st_complex_double_st%x = a+b%x
    add_single_st_complex_double_st%y = b%y
  end function add_single_st_complex_double_st

  elemental function add_single_st_complex_single_st(a, b)
    type(single_st), intent(in) :: a
    type(complex_single_st), intent(in) :: b
    type(complex_single_st) add_single_st_complex_single_st
     
     

    add_single_st_complex_single_st%x = a+b%x
    add_single_st_complex_single_st%y = b%y
  end function add_single_st_complex_single_st


  elemental function add_double_st_complex_double_st(a, b)
    type(double_st), intent(in) :: a
    type(complex_double_st), intent(in) :: b
    type(complex_double_st) add_double_st_complex_double_st
     
     

    add_double_st_complex_double_st%x = a+b%x
    add_double_st_complex_double_st%y = b%y
  end function add_double_st_complex_double_st

  elemental function add_double_st_complex_single_st(a, b)
    type(double_st), intent(in) :: a
    type(complex_single_st), intent(in) :: b
    type(complex_double_st) add_double_st_complex_single_st
     
     

    add_double_st_complex_single_st%x = a+b%x
    add_double_st_complex_single_st%y = b%y
  end function add_double_st_complex_single_st


  elemental function add_complex_float_complex_double_st(a, b)
    complex(REAL32), intent(in) :: a
    type(complex_double_st), intent(in) :: b
    type(complex_double_st) add_complex_float_complex_double_st
    real(REAL32) re, im
     

    re = real(a) ; im = aimag(a)
    add_complex_float_complex_double_st%x = re+b%x
    add_complex_float_complex_double_st%y = im+b%y
  end function add_complex_float_complex_double_st

  elemental function add_complex_float_complex_single_st(a, b)
    complex(REAL32), intent(in) :: a
    type(complex_single_st), intent(in) :: b
    type(complex_single_st) add_complex_float_complex_single_st
    real(REAL32) re, im
     

    re = real(a) ; im = aimag(a)
    add_complex_float_complex_single_st%x = re+b%x
    add_complex_float_complex_single_st%y = im+b%y
  end function add_complex_float_complex_single_st


  elemental function add_complex_double_complex_double_st(a, b)
    complex(REAL64), intent(in) :: a
    type(complex_double_st), intent(in) :: b
    type(complex_double_st) add_complex_double_complex_double_st
    real(REAL64) re, im
     

    re = real(a) ; im = aimag(a)
    add_complex_double_complex_double_st%x = re+b%x
    add_complex_double_complex_double_st%y = im+b%y
  end function add_complex_double_complex_double_st

  elemental function add_complex_double_complex_single_st(a, b)
    complex(REAL64), intent(in) :: a
    type(complex_single_st), intent(in) :: b
    type(complex_single_st) add_complex_double_complex_single_st
    real(REAL64) re, im
     

    re = real(a) ; im = aimag(a)
    add_complex_double_complex_single_st%x = re+b%x
    add_complex_double_complex_single_st%y = im+b%y
  end function add_complex_double_complex_single_st

!
  elemental function add_double_complex_double_st(a, b)
    real(C_DOUBLE), intent(in) :: a
    type(complex_double_st), intent(in) :: b
    type(complex_double_st) add_double_complex_double_st
     
     

    add_double_complex_double_st%x = a+b%x
    add_double_complex_double_st%y = b%y
  end function add_double_complex_double_st

  elemental function add_double_complex_single_st(a, b)
    real(C_DOUBLE), intent(in) :: a
    type(complex_single_st), intent(in) :: b
    type(complex_single_st) add_double_complex_single_st
     
     

    add_double_complex_single_st%x = a+b%x
    add_double_complex_single_st%y = b%y
  end function add_double_complex_single_st

!
  elemental function add_float_complex_double_st(a, b)
    real(C_FLOAT), intent(in) :: a
    type(complex_double_st), intent(in) :: b
    type(complex_double_st) add_float_complex_double_st
     
     

    add_float_complex_double_st%x = a+b%x
    add_float_complex_double_st%y = b%y
  end function add_float_complex_double_st

  elemental function add_float_complex_single_st(a, b)
    real(C_FLOAT), intent(in) :: a
    type(complex_single_st), intent(in) :: b
    type(complex_single_st) add_float_complex_single_st
     
     

    add_float_complex_single_st%x = a+b%x
    add_float_complex_single_st%y = b%y
  end function add_float_complex_single_st

!
  elemental function add_long_long_complex_double_st(a, b)
    integer(C_LONG_LONG), intent(in) :: a
    type(complex_double_st), intent(in) :: b
    type(complex_double_st) add_long_long_complex_double_st
     
     

    add_long_long_complex_double_st%x = a+b%x
    add_long_long_complex_double_st%y = b%y
  end function add_long_long_complex_double_st

  elemental function add_long_long_complex_single_st(a, b)
    integer(C_LONG_LONG), intent(in) :: a
    type(complex_single_st), intent(in) :: b
    type(complex_single_st) add_long_long_complex_single_st
     
     

    add_long_long_complex_single_st%x = a+b%x
    add_long_long_complex_single_st%y = b%y
  end function add_long_long_complex_single_st

!
!CONTAINS_OP(add, long,              complex_double_st, complex_double_st)
!CONTAINS_OP(add, long,              complex_single_st, complex_single_st)
!
  elemental function add_int_complex_double_st(a, b)
    integer(C_INT), intent(in) :: a
    type(complex_double_st), intent(in) :: b
    type(complex_double_st) add_int_complex_double_st
     
     

    add_int_complex_double_st%x = a+b%x
    add_int_complex_double_st%y = b%y
  end function add_int_complex_double_st

  elemental function add_int_complex_single_st(a, b)
    integer(C_INT), intent(in) :: a
    type(complex_single_st), intent(in) :: b
    type(complex_single_st) add_int_complex_single_st
     
     

    add_int_complex_single_st%x = a+b%x
    add_int_complex_single_st%y = b%y
  end function add_int_complex_single_st

!
  elemental function add_short_complex_double_st(a, b)
    integer(C_SHORT), intent(in) :: a
    type(complex_double_st), intent(in) :: b
    type(complex_double_st) add_short_complex_double_st
     
     

    add_short_complex_double_st%x = a+b%x
    add_short_complex_double_st%y = b%y
  end function add_short_complex_double_st

  elemental function add_short_complex_single_st(a, b)
    integer(C_SHORT), intent(in) :: a
    type(complex_single_st), intent(in) :: b
    type(complex_single_st) add_short_complex_single_st
     
     

    add_short_complex_single_st%x = a+b%x
    add_short_complex_single_st%y = b%y
  end function add_short_complex_single_st




end module cadna_cmplx_add

