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


! Module de surcharge de l'opérateur "*"




module cadna_mult
  use ISO_C_BINDING, only : C_LONG_LONG, C_LONG, C_INT, C_SHORT, C_FLOAT, C_DOUBLE
  use cadna_types
  implicit none




  interface operator(*)
    module procedure mult_double_st_double_st
  end interface operator(*)
  interface
    pure function cpp_mult_double_st_double_st(a, b) bind(C)
      import double_st
      
      type(double_st), intent(in) :: a
      type(double_st), intent(in) :: b
      type(double_st) cpp_mult_double_st_double_st
    end function cpp_mult_double_st_double_st
  end interface

  interface operator(*)
    module procedure mult_double_st_single_st
  end interface operator(*)
  interface
    pure function cpp_mult_double_st_single_st(a, b) bind(C)
      import double_st
      import single_st
      type(double_st), intent(in) :: a
      type(single_st), intent(in) :: b
      type(double_st) cpp_mult_double_st_single_st
    end function cpp_mult_double_st_single_st
  end interface

  interface operator(*)
    module procedure mult_double_st_double
  end interface operator(*)
  interface
    pure function cpp_mult_double_st_double(a, b) bind(C)
      import double_st
      import C_DOUBLE
      type(double_st), intent(in) :: a
      real(C_DOUBLE), intent(in) :: b
      type(double_st) cpp_mult_double_st_double
    end function cpp_mult_double_st_double
  end interface

  interface operator(*)
    module procedure mult_double_st_float
  end interface operator(*)
  interface
    pure function cpp_mult_double_st_float(a, b) bind(C)
      import double_st
      import C_FLOAT
      type(double_st), intent(in) :: a
      real(C_FLOAT), intent(in) :: b
      type(double_st) cpp_mult_double_st_float
    end function cpp_mult_double_st_float
  end interface

  interface operator(*)
    module procedure mult_double_st_long_long
  end interface operator(*)
  interface
    pure function cpp_mult_double_st_long_long(a, b) bind(C)
      import double_st
      import C_LONG_LONG
      type(double_st), intent(in) :: a
      integer(C_LONG_LONG), intent(in) :: b
      type(double_st) cpp_mult_double_st_long_long
    end function cpp_mult_double_st_long_long
  end interface

!INTERFACE_OP(*, mult, double_st, long,      double_st)
  interface operator(*)
    module procedure mult_double_st_int
  end interface operator(*)
  interface
    pure function cpp_mult_double_st_int(a, b) bind(C)
      import double_st
      import C_INT
      type(double_st), intent(in) :: a
      integer(C_INT), intent(in) :: b
      type(double_st) cpp_mult_double_st_int
    end function cpp_mult_double_st_int
  end interface

  interface operator(*)
    module procedure mult_double_st_short
  end interface operator(*)
  interface
    pure function cpp_mult_double_st_short(a, b) bind(C)
      import double_st
      import C_SHORT
      type(double_st), intent(in) :: a
      integer(C_SHORT), intent(in) :: b
      type(double_st) cpp_mult_double_st_short
    end function cpp_mult_double_st_short
  end interface


  interface operator(*)
    module procedure mult_single_st_double_st
  end interface operator(*)
  interface
    pure function cpp_mult_single_st_double_st(a, b) bind(C)
      import single_st
      import double_st
      type(single_st), intent(in) :: a
      type(double_st), intent(in) :: b
      type(double_st) cpp_mult_single_st_double_st
    end function cpp_mult_single_st_double_st
  end interface

  interface operator(*)
    module procedure mult_single_st_single_st
  end interface operator(*)
  interface
    pure function cpp_mult_single_st_single_st(a, b) bind(C)
      import single_st
      
      type(single_st), intent(in) :: a
      type(single_st), intent(in) :: b
      type(single_st) cpp_mult_single_st_single_st
    end function cpp_mult_single_st_single_st
  end interface

  interface operator(*)
    module procedure mult_single_st_double
  end interface operator(*)
  interface
    pure function cpp_mult_single_st_double(a, b) bind(C)
      import single_st
      import C_DOUBLE
      type(single_st), intent(in) :: a
      real(C_DOUBLE), intent(in) :: b
      type(single_st) cpp_mult_single_st_double
    end function cpp_mult_single_st_double
  end interface

  interface operator(*)
    module procedure mult_single_st_float
  end interface operator(*)
  interface
    pure function cpp_mult_single_st_float(a, b) bind(C)
      import single_st
      import C_FLOAT
      type(single_st), intent(in) :: a
      real(C_FLOAT), intent(in) :: b
      type(single_st) cpp_mult_single_st_float
    end function cpp_mult_single_st_float
  end interface

  interface operator(*)
    module procedure mult_single_st_long_long
  end interface operator(*)
  interface
    pure function cpp_mult_single_st_long_long(a, b) bind(C)
      import single_st
      import C_LONG_LONG
      type(single_st), intent(in) :: a
      integer(C_LONG_LONG), intent(in) :: b
      type(single_st) cpp_mult_single_st_long_long
    end function cpp_mult_single_st_long_long
  end interface

!INTERFACE_OP(*, mult, single_st, long,      single_st)
  interface operator(*)
    module procedure mult_single_st_int
  end interface operator(*)
  interface
    pure function cpp_mult_single_st_int(a, b) bind(C)
      import single_st
      import C_INT
      type(single_st), intent(in) :: a
      integer(C_INT), intent(in) :: b
      type(single_st) cpp_mult_single_st_int
    end function cpp_mult_single_st_int
  end interface

  interface operator(*)
    module procedure mult_single_st_short
  end interface operator(*)
  interface
    pure function cpp_mult_single_st_short(a, b) bind(C)
      import single_st
      import C_SHORT
      type(single_st), intent(in) :: a
      integer(C_SHORT), intent(in) :: b
      type(single_st) cpp_mult_single_st_short
    end function cpp_mult_single_st_short
  end interface


  interface operator(*)
    module procedure mult_double_double_st
  end interface operator(*)
  interface
    pure function cpp_mult_double_double_st(a, b) bind(C)
      import C_DOUBLE
      import double_st
      real(C_DOUBLE), intent(in) :: a
      type(double_st), intent(in) :: b
      type(double_st) cpp_mult_double_double_st
    end function cpp_mult_double_double_st
  end interface

  interface operator(*)
    module procedure mult_double_single_st
  end interface operator(*)
  interface
    pure function cpp_mult_double_single_st(a, b) bind(C)
      import C_DOUBLE
      import single_st
      real(C_DOUBLE), intent(in) :: a
      type(single_st), intent(in) :: b
      type(single_st) cpp_mult_double_single_st
    end function cpp_mult_double_single_st
  end interface


  interface operator(*)
    module procedure mult_float_double_st
  end interface operator(*)
  interface
    pure function cpp_mult_float_double_st(a, b) bind(C)
      import C_FLOAT
      import double_st
      real(C_FLOAT), intent(in) :: a
      type(double_st), intent(in) :: b
      type(double_st) cpp_mult_float_double_st
    end function cpp_mult_float_double_st
  end interface

  interface operator(*)
    module procedure mult_float_single_st
  end interface operator(*)
  interface
    pure function cpp_mult_float_single_st(a, b) bind(C)
      import C_FLOAT
      import single_st
      real(C_FLOAT), intent(in) :: a
      type(single_st), intent(in) :: b
      type(single_st) cpp_mult_float_single_st
    end function cpp_mult_float_single_st
  end interface


  interface operator(*)
    module procedure mult_long_long_double_st
  end interface operator(*)
  interface
    pure function cpp_mult_long_long_double_st(a, b) bind(C)
      import C_LONG_LONG
      import double_st
      integer(C_LONG_LONG), intent(in) :: a
      type(double_st), intent(in) :: b
      type(double_st) cpp_mult_long_long_double_st
    end function cpp_mult_long_long_double_st
  end interface

  interface operator(*)
    module procedure mult_long_long_single_st
  end interface operator(*)
  interface
    pure function cpp_mult_long_long_single_st(a, b) bind(C)
      import C_LONG_LONG
      import single_st
      integer(C_LONG_LONG), intent(in) :: a
      type(single_st), intent(in) :: b
      type(single_st) cpp_mult_long_long_single_st
    end function cpp_mult_long_long_single_st
  end interface


!INTERFACE_OP(*, mult, long, double_st,      double_st)
!INTERFACE_OP(*, mult, long, single_st,      single_st)

  interface operator(*)
    module procedure mult_int_double_st
  end interface operator(*)
  interface
    pure function cpp_mult_int_double_st(a, b) bind(C)
      import C_INT
      import double_st
      integer(C_INT), intent(in) :: a
      type(double_st), intent(in) :: b
      type(double_st) cpp_mult_int_double_st
    end function cpp_mult_int_double_st
  end interface

  interface operator(*)
    module procedure mult_int_single_st
  end interface operator(*)
  interface
    pure function cpp_mult_int_single_st(a, b) bind(C)
      import C_INT
      import single_st
      integer(C_INT), intent(in) :: a
      type(single_st), intent(in) :: b
      type(single_st) cpp_mult_int_single_st
    end function cpp_mult_int_single_st
  end interface


  interface operator(*)
    module procedure mult_short_double_st
  end interface operator(*)
  interface
    pure function cpp_mult_short_double_st(a, b) bind(C)
      import C_SHORT
      import double_st
      integer(C_SHORT), intent(in) :: a
      type(double_st), intent(in) :: b
      type(double_st) cpp_mult_short_double_st
    end function cpp_mult_short_double_st
  end interface

  interface operator(*)
    module procedure mult_short_single_st
  end interface operator(*)
  interface
    pure function cpp_mult_short_single_st(a, b) bind(C)
      import C_SHORT
      import single_st
      integer(C_SHORT), intent(in) :: a
      type(single_st), intent(in) :: b
      type(single_st) cpp_mult_short_single_st
    end function cpp_mult_short_single_st
  end interface


contains



  elemental function mult_double_st_double_st(a, b)
    type(double_st), intent(in) :: a
    type(double_st), intent(in) :: b
    type(double_st) mult_double_st_double_st

    mult_double_st_double_st = cpp_mult_double_st_double_st(a, b)
  end function mult_double_st_double_st

  elemental function mult_double_st_single_st(a, b)
    type(double_st), intent(in) :: a
    type(single_st), intent(in) :: b
    type(double_st) mult_double_st_single_st

    mult_double_st_single_st = cpp_mult_double_st_single_st(a, b)
  end function mult_double_st_single_st

  elemental function mult_double_st_double(a, b)
    type(double_st), intent(in) :: a
    real(C_DOUBLE), intent(in) :: b
    type(double_st) mult_double_st_double

    mult_double_st_double = cpp_mult_double_st_double(a, b)
  end function mult_double_st_double

  elemental function mult_double_st_float(a, b)
    type(double_st), intent(in) :: a
    real(C_FLOAT), intent(in) :: b
    type(double_st) mult_double_st_float

    mult_double_st_float = cpp_mult_double_st_float(a, b)
  end function mult_double_st_float

  elemental function mult_double_st_long_long(a, b)
    type(double_st), intent(in) :: a
    integer(C_LONG_LONG), intent(in) :: b
    type(double_st) mult_double_st_long_long

    mult_double_st_long_long = cpp_mult_double_st_long_long(a, b)
  end function mult_double_st_long_long

!CONTAINS_OP(mult, double_st, long,      double_st)
  elemental function mult_double_st_int(a, b)
    type(double_st), intent(in) :: a
    integer(C_INT), intent(in) :: b
    type(double_st) mult_double_st_int

    mult_double_st_int = cpp_mult_double_st_int(a, b)
  end function mult_double_st_int

  elemental function mult_double_st_short(a, b)
    type(double_st), intent(in) :: a
    integer(C_SHORT), intent(in) :: b
    type(double_st) mult_double_st_short

    mult_double_st_short = cpp_mult_double_st_short(a, b)
  end function mult_double_st_short


  elemental function mult_single_st_double_st(a, b)
    type(single_st), intent(in) :: a
    type(double_st), intent(in) :: b
    type(double_st) mult_single_st_double_st

    mult_single_st_double_st = cpp_mult_single_st_double_st(a, b)
  end function mult_single_st_double_st

  elemental function mult_single_st_single_st(a, b)
    type(single_st), intent(in) :: a
    type(single_st), intent(in) :: b
    type(single_st) mult_single_st_single_st

    mult_single_st_single_st = cpp_mult_single_st_single_st(a, b)
  end function mult_single_st_single_st

  elemental function mult_single_st_double(a, b)
    type(single_st), intent(in) :: a
    real(C_DOUBLE), intent(in) :: b
    type(single_st) mult_single_st_double

    mult_single_st_double = cpp_mult_single_st_double(a, b)
  end function mult_single_st_double

  elemental function mult_single_st_float(a, b)
    type(single_st), intent(in) :: a
    real(C_FLOAT), intent(in) :: b
    type(single_st) mult_single_st_float

    mult_single_st_float = cpp_mult_single_st_float(a, b)
  end function mult_single_st_float

  elemental function mult_single_st_long_long(a, b)
    type(single_st), intent(in) :: a
    integer(C_LONG_LONG), intent(in) :: b
    type(single_st) mult_single_st_long_long

    mult_single_st_long_long = cpp_mult_single_st_long_long(a, b)
  end function mult_single_st_long_long

!CONTAINS_OP(mult, single_st, long,      single_st)
  elemental function mult_single_st_int(a, b)
    type(single_st), intent(in) :: a
    integer(C_INT), intent(in) :: b
    type(single_st) mult_single_st_int

    mult_single_st_int = cpp_mult_single_st_int(a, b)
  end function mult_single_st_int

  elemental function mult_single_st_short(a, b)
    type(single_st), intent(in) :: a
    integer(C_SHORT), intent(in) :: b
    type(single_st) mult_single_st_short

    mult_single_st_short = cpp_mult_single_st_short(a, b)
  end function mult_single_st_short


  elemental function mult_double_double_st(a, b)
    real(C_DOUBLE), intent(in) :: a
    type(double_st), intent(in) :: b
    type(double_st) mult_double_double_st

    mult_double_double_st = cpp_mult_double_double_st(a, b)
  end function mult_double_double_st

  elemental function mult_double_single_st(a, b)
    real(C_DOUBLE), intent(in) :: a
    type(single_st), intent(in) :: b
    type(single_st) mult_double_single_st

    mult_double_single_st = cpp_mult_double_single_st(a, b)
  end function mult_double_single_st


  elemental function mult_float_double_st(a, b)
    real(C_FLOAT), intent(in) :: a
    type(double_st), intent(in) :: b
    type(double_st) mult_float_double_st

    mult_float_double_st = cpp_mult_float_double_st(a, b)
  end function mult_float_double_st

  elemental function mult_float_single_st(a, b)
    real(C_FLOAT), intent(in) :: a
    type(single_st), intent(in) :: b
    type(single_st) mult_float_single_st

    mult_float_single_st = cpp_mult_float_single_st(a, b)
  end function mult_float_single_st


  elemental function mult_long_long_double_st(a, b)
    integer(C_LONG_LONG), intent(in) :: a
    type(double_st), intent(in) :: b
    type(double_st) mult_long_long_double_st

    mult_long_long_double_st = cpp_mult_long_long_double_st(a, b)
  end function mult_long_long_double_st

  elemental function mult_long_long_single_st(a, b)
    integer(C_LONG_LONG), intent(in) :: a
    type(single_st), intent(in) :: b
    type(single_st) mult_long_long_single_st

    mult_long_long_single_st = cpp_mult_long_long_single_st(a, b)
  end function mult_long_long_single_st


!CONTAINS_OP(mult, long, double_st,      double_st)
!CONTAINS_OP(mult, long, single_st,      single_st)

  elemental function mult_int_double_st(a, b)
    integer(C_INT), intent(in) :: a
    type(double_st), intent(in) :: b
    type(double_st) mult_int_double_st

    mult_int_double_st = cpp_mult_int_double_st(a, b)
  end function mult_int_double_st

  elemental function mult_int_single_st(a, b)
    integer(C_INT), intent(in) :: a
    type(single_st), intent(in) :: b
    type(single_st) mult_int_single_st

    mult_int_single_st = cpp_mult_int_single_st(a, b)
  end function mult_int_single_st


  elemental function mult_short_double_st(a, b)
    integer(C_SHORT), intent(in) :: a
    type(double_st), intent(in) :: b
    type(double_st) mult_short_double_st

    mult_short_double_st = cpp_mult_short_double_st(a, b)
  end function mult_short_double_st

  elemental function mult_short_single_st(a, b)
    integer(C_SHORT), intent(in) :: a
    type(single_st), intent(in) :: b
    type(single_st) mult_short_single_st

    mult_short_single_st = cpp_mult_short_single_st(a, b)
  end function mult_short_single_st




end module cadna_mult

