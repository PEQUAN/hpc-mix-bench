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


! Module de surcharge de l'opérateur "/="




module cadna_ne
  use ISO_C_BINDING, only : C_BOOL, C_LONG_LONG, C_LONG, C_INT, C_SHORT, C_FLOAT, C_DOUBLE
  !`use ISO_C_BINDING, only : C_BOOL, C_LONG_LONG, C_LONG, C_INT, C_SHORT, C_FLOAT'
  use cadna_types
  implicit none

!INIT(ne, C_LONG_LONG, C_LONG, C_INT, C_SHORT, C_FLOAT, C_DOUBLE)



  interface operator(/=)
    module procedure ne_double_st_double_st
  end interface operator(/=)
  interface
    pure function cpp_ne_double_st_double_st(a, b) bind(C)
      import C_BOOL
      import double_st
      
      type(double_st), intent(in) :: a
      type(double_st), intent(in) :: b
      logical(C_BOOL) :: cpp_ne_double_st_double_st
      !logical :: cpp_ne_double_st_double_st
    end function cpp_ne_double_st_double_st
  end interface

  interface operator(/=)
    module procedure ne_double_st_single_st
  end interface operator(/=)
  interface
    pure function cpp_ne_double_st_single_st(a, b) bind(C)
      import C_BOOL
      import double_st
      import single_st
      type(double_st), intent(in) :: a
      type(single_st), intent(in) :: b
      logical(C_BOOL) :: cpp_ne_double_st_single_st
      !logical :: cpp_ne_double_st_single_st
    end function cpp_ne_double_st_single_st
  end interface

  interface operator(/=)
    module procedure ne_double_st_double
  end interface operator(/=)
  interface
    pure function cpp_ne_double_st_double(a, b) bind(C)
      import C_BOOL
      import double_st
      import C_DOUBLE
      type(double_st), intent(in) :: a
      real(C_DOUBLE), intent(in) :: b
      logical(C_BOOL) :: cpp_ne_double_st_double
      !logical :: cpp_ne_double_st_double
    end function cpp_ne_double_st_double
  end interface

  interface operator(/=)
    module procedure ne_double_st_float
  end interface operator(/=)
  interface
    pure function cpp_ne_double_st_float(a, b) bind(C)
      import C_BOOL
      import double_st
      import C_FLOAT
      type(double_st), intent(in) :: a
      real(C_FLOAT), intent(in) :: b
      logical(C_BOOL) :: cpp_ne_double_st_float
      !logical :: cpp_ne_double_st_float
    end function cpp_ne_double_st_float
  end interface

  interface operator(/=)
    module procedure ne_double_st_long_long
  end interface operator(/=)
  interface
    pure function cpp_ne_double_st_long_long(a, b) bind(C)
      import C_BOOL
      import double_st
      import C_LONG_LONG
      type(double_st), intent(in) :: a
      integer(C_LONG_LONG), intent(in) :: b
      logical(C_BOOL) :: cpp_ne_double_st_long_long
      !logical :: cpp_ne_double_st_long_long
    end function cpp_ne_double_st_long_long
  end interface

!INTERFACE_OP(/=, ne, double_st, long)
  interface operator(/=)
    module procedure ne_double_st_int
  end interface operator(/=)
  interface
    pure function cpp_ne_double_st_int(a, b) bind(C)
      import C_BOOL
      import double_st
      import C_INT
      type(double_st), intent(in) :: a
      integer(C_INT), intent(in) :: b
      logical(C_BOOL) :: cpp_ne_double_st_int
      !logical :: cpp_ne_double_st_int
    end function cpp_ne_double_st_int
  end interface

  interface operator(/=)
    module procedure ne_double_st_short
  end interface operator(/=)
  interface
    pure function cpp_ne_double_st_short(a, b) bind(C)
      import C_BOOL
      import double_st
      import C_SHORT
      type(double_st), intent(in) :: a
      integer(C_SHORT), intent(in) :: b
      logical(C_BOOL) :: cpp_ne_double_st_short
      !logical :: cpp_ne_double_st_short
    end function cpp_ne_double_st_short
  end interface


  interface operator(/=)
    module procedure ne_single_st_double_st
  end interface operator(/=)
  interface
    pure function cpp_ne_single_st_double_st(a, b) bind(C)
      import C_BOOL
      import single_st
      import double_st
      type(single_st), intent(in) :: a
      type(double_st), intent(in) :: b
      logical(C_BOOL) :: cpp_ne_single_st_double_st
      !logical :: cpp_ne_single_st_double_st
    end function cpp_ne_single_st_double_st
  end interface

  interface operator(/=)
    module procedure ne_single_st_single_st
  end interface operator(/=)
  interface
    pure function cpp_ne_single_st_single_st(a, b) bind(C)
      import C_BOOL
      import single_st
      
      type(single_st), intent(in) :: a
      type(single_st), intent(in) :: b
      logical(C_BOOL) :: cpp_ne_single_st_single_st
      !logical :: cpp_ne_single_st_single_st
    end function cpp_ne_single_st_single_st
  end interface

  interface operator(/=)
    module procedure ne_single_st_double
  end interface operator(/=)
  interface
    pure function cpp_ne_single_st_double(a, b) bind(C)
      import C_BOOL
      import single_st
      import C_DOUBLE
      type(single_st), intent(in) :: a
      real(C_DOUBLE), intent(in) :: b
      logical(C_BOOL) :: cpp_ne_single_st_double
      !logical :: cpp_ne_single_st_double
    end function cpp_ne_single_st_double
  end interface

  interface operator(/=)
    module procedure ne_single_st_float
  end interface operator(/=)
  interface
    pure function cpp_ne_single_st_float(a, b) bind(C)
      import C_BOOL
      import single_st
      import C_FLOAT
      type(single_st), intent(in) :: a
      real(C_FLOAT), intent(in) :: b
      logical(C_BOOL) :: cpp_ne_single_st_float
      !logical :: cpp_ne_single_st_float
    end function cpp_ne_single_st_float
  end interface

  interface operator(/=)
    module procedure ne_single_st_long_long
  end interface operator(/=)
  interface
    pure function cpp_ne_single_st_long_long(a, b) bind(C)
      import C_BOOL
      import single_st
      import C_LONG_LONG
      type(single_st), intent(in) :: a
      integer(C_LONG_LONG), intent(in) :: b
      logical(C_BOOL) :: cpp_ne_single_st_long_long
      !logical :: cpp_ne_single_st_long_long
    end function cpp_ne_single_st_long_long
  end interface

!INTERFACE_OP(/=, ne, single_st, long)
  interface operator(/=)
    module procedure ne_single_st_int
  end interface operator(/=)
  interface
    pure function cpp_ne_single_st_int(a, b) bind(C)
      import C_BOOL
      import single_st
      import C_INT
      type(single_st), intent(in) :: a
      integer(C_INT), intent(in) :: b
      logical(C_BOOL) :: cpp_ne_single_st_int
      !logical :: cpp_ne_single_st_int
    end function cpp_ne_single_st_int
  end interface

  interface operator(/=)
    module procedure ne_single_st_short
  end interface operator(/=)
  interface
    pure function cpp_ne_single_st_short(a, b) bind(C)
      import C_BOOL
      import single_st
      import C_SHORT
      type(single_st), intent(in) :: a
      integer(C_SHORT), intent(in) :: b
      logical(C_BOOL) :: cpp_ne_single_st_short
      !logical :: cpp_ne_single_st_short
    end function cpp_ne_single_st_short
  end interface


  interface operator(/=)
    module procedure ne_double_double_st
  end interface operator(/=)
  interface
    pure function cpp_ne_double_double_st(a, b) bind(C)
      import C_BOOL
      import C_DOUBLE
      import double_st
      real(C_DOUBLE), intent(in) :: a
      type(double_st), intent(in) :: b
      logical(C_BOOL) :: cpp_ne_double_double_st
      !logical :: cpp_ne_double_double_st
    end function cpp_ne_double_double_st
  end interface

  interface operator(/=)
    module procedure ne_double_single_st
  end interface operator(/=)
  interface
    pure function cpp_ne_double_single_st(a, b) bind(C)
      import C_BOOL
      import C_DOUBLE
      import single_st
      real(C_DOUBLE), intent(in) :: a
      type(single_st), intent(in) :: b
      logical(C_BOOL) :: cpp_ne_double_single_st
      !logical :: cpp_ne_double_single_st
    end function cpp_ne_double_single_st
  end interface


  interface operator(/=)
    module procedure ne_float_double_st
  end interface operator(/=)
  interface
    pure function cpp_ne_float_double_st(a, b) bind(C)
      import C_BOOL
      import C_FLOAT
      import double_st
      real(C_FLOAT), intent(in) :: a
      type(double_st), intent(in) :: b
      logical(C_BOOL) :: cpp_ne_float_double_st
      !logical :: cpp_ne_float_double_st
    end function cpp_ne_float_double_st
  end interface

  interface operator(/=)
    module procedure ne_float_single_st
  end interface operator(/=)
  interface
    pure function cpp_ne_float_single_st(a, b) bind(C)
      import C_BOOL
      import C_FLOAT
      import single_st
      real(C_FLOAT), intent(in) :: a
      type(single_st), intent(in) :: b
      logical(C_BOOL) :: cpp_ne_float_single_st
      !logical :: cpp_ne_float_single_st
    end function cpp_ne_float_single_st
  end interface


  interface operator(/=)
    module procedure ne_long_long_double_st
  end interface operator(/=)
  interface
    pure function cpp_ne_long_long_double_st(a, b) bind(C)
      import C_BOOL
      import C_LONG_LONG
      import double_st
      integer(C_LONG_LONG), intent(in) :: a
      type(double_st), intent(in) :: b
      logical(C_BOOL) :: cpp_ne_long_long_double_st
      !logical :: cpp_ne_long_long_double_st
    end function cpp_ne_long_long_double_st
  end interface

  interface operator(/=)
    module procedure ne_long_long_single_st
  end interface operator(/=)
  interface
    pure function cpp_ne_long_long_single_st(a, b) bind(C)
      import C_BOOL
      import C_LONG_LONG
      import single_st
      integer(C_LONG_LONG), intent(in) :: a
      type(single_st), intent(in) :: b
      logical(C_BOOL) :: cpp_ne_long_long_single_st
      !logical :: cpp_ne_long_long_single_st
    end function cpp_ne_long_long_single_st
  end interface


!INTERFACE_OP(/=, ne, long,     double_st)
!INTERFACE_OP(/=, ne, long,     single_st)

  interface operator(/=)
    module procedure ne_int_double_st
  end interface operator(/=)
  interface
    pure function cpp_ne_int_double_st(a, b) bind(C)
      import C_BOOL
      import C_INT
      import double_st
      integer(C_INT), intent(in) :: a
      type(double_st), intent(in) :: b
      logical(C_BOOL) :: cpp_ne_int_double_st
      !logical :: cpp_ne_int_double_st
    end function cpp_ne_int_double_st
  end interface

  interface operator(/=)
    module procedure ne_int_single_st
  end interface operator(/=)
  interface
    pure function cpp_ne_int_single_st(a, b) bind(C)
      import C_BOOL
      import C_INT
      import single_st
      integer(C_INT), intent(in) :: a
      type(single_st), intent(in) :: b
      logical(C_BOOL) :: cpp_ne_int_single_st
      !logical :: cpp_ne_int_single_st
    end function cpp_ne_int_single_st
  end interface


  interface operator(/=)
    module procedure ne_short_double_st
  end interface operator(/=)
  interface
    pure function cpp_ne_short_double_st(a, b) bind(C)
      import C_BOOL
      import C_SHORT
      import double_st
      integer(C_SHORT), intent(in) :: a
      type(double_st), intent(in) :: b
      logical(C_BOOL) :: cpp_ne_short_double_st
      !logical :: cpp_ne_short_double_st
    end function cpp_ne_short_double_st
  end interface

  interface operator(/=)
    module procedure ne_short_single_st
  end interface operator(/=)
  interface
    pure function cpp_ne_short_single_st(a, b) bind(C)
      import C_BOOL
      import C_SHORT
      import single_st
      integer(C_SHORT), intent(in) :: a
      type(single_st), intent(in) :: b
      logical(C_BOOL) :: cpp_ne_short_single_st
      !logical :: cpp_ne_short_single_st
    end function cpp_ne_short_single_st
  end interface


contains



  elemental function ne_double_st_double_st(a, b)
    type(double_st), intent(in) :: a
    type(double_st), intent(in) :: b
    !logical(C_BOOL) ne_double_st_double_st
    logical ne_double_st_double_st

    ne_double_st_double_st = cpp_ne_double_st_double_st(a, b)
  end function ne_double_st_double_st

  elemental function ne_double_st_single_st(a, b)
    type(double_st), intent(in) :: a
    type(single_st), intent(in) :: b
    !logical(C_BOOL) ne_double_st_single_st
    logical ne_double_st_single_st

    ne_double_st_single_st = cpp_ne_double_st_single_st(a, b)
  end function ne_double_st_single_st

  elemental function ne_double_st_double(a, b)
    type(double_st), intent(in) :: a
    real(C_DOUBLE), intent(in) :: b
    !logical(C_BOOL) ne_double_st_double
    logical ne_double_st_double

    ne_double_st_double = cpp_ne_double_st_double(a, b)
  end function ne_double_st_double

  elemental function ne_double_st_float(a, b)
    type(double_st), intent(in) :: a
    real(C_FLOAT), intent(in) :: b
    !logical(C_BOOL) ne_double_st_float
    logical ne_double_st_float

    ne_double_st_float = cpp_ne_double_st_float(a, b)
  end function ne_double_st_float

  elemental function ne_double_st_long_long(a, b)
    type(double_st), intent(in) :: a
    integer(C_LONG_LONG), intent(in) :: b
    !logical(C_BOOL) ne_double_st_long_long
    logical ne_double_st_long_long

    ne_double_st_long_long = cpp_ne_double_st_long_long(a, b)
  end function ne_double_st_long_long

!CONTAINS_OP(ne, double_st, long)
  elemental function ne_double_st_int(a, b)
    type(double_st), intent(in) :: a
    integer(C_INT), intent(in) :: b
    !logical(C_BOOL) ne_double_st_int
    logical ne_double_st_int

    ne_double_st_int = cpp_ne_double_st_int(a, b)
  end function ne_double_st_int

  elemental function ne_double_st_short(a, b)
    type(double_st), intent(in) :: a
    integer(C_SHORT), intent(in) :: b
    !logical(C_BOOL) ne_double_st_short
    logical ne_double_st_short

    ne_double_st_short = cpp_ne_double_st_short(a, b)
  end function ne_double_st_short


  elemental function ne_single_st_double_st(a, b)
    type(single_st), intent(in) :: a
    type(double_st), intent(in) :: b
    !logical(C_BOOL) ne_single_st_double_st
    logical ne_single_st_double_st

    ne_single_st_double_st = cpp_ne_single_st_double_st(a, b)
  end function ne_single_st_double_st

  elemental function ne_single_st_single_st(a, b)
    type(single_st), intent(in) :: a
    type(single_st), intent(in) :: b
    !logical(C_BOOL) ne_single_st_single_st
    logical ne_single_st_single_st

    ne_single_st_single_st = cpp_ne_single_st_single_st(a, b)
  end function ne_single_st_single_st

  elemental function ne_single_st_double(a, b)
    type(single_st), intent(in) :: a
    real(C_DOUBLE), intent(in) :: b
    !logical(C_BOOL) ne_single_st_double
    logical ne_single_st_double

    ne_single_st_double = cpp_ne_single_st_double(a, b)
  end function ne_single_st_double

  elemental function ne_single_st_float(a, b)
    type(single_st), intent(in) :: a
    real(C_FLOAT), intent(in) :: b
    !logical(C_BOOL) ne_single_st_float
    logical ne_single_st_float

    ne_single_st_float = cpp_ne_single_st_float(a, b)
  end function ne_single_st_float

  elemental function ne_single_st_long_long(a, b)
    type(single_st), intent(in) :: a
    integer(C_LONG_LONG), intent(in) :: b
    !logical(C_BOOL) ne_single_st_long_long
    logical ne_single_st_long_long

    ne_single_st_long_long = cpp_ne_single_st_long_long(a, b)
  end function ne_single_st_long_long

!CONTAINS_OP(ne, single_st, long)
  elemental function ne_single_st_int(a, b)
    type(single_st), intent(in) :: a
    integer(C_INT), intent(in) :: b
    !logical(C_BOOL) ne_single_st_int
    logical ne_single_st_int

    ne_single_st_int = cpp_ne_single_st_int(a, b)
  end function ne_single_st_int

  elemental function ne_single_st_short(a, b)
    type(single_st), intent(in) :: a
    integer(C_SHORT), intent(in) :: b
    !logical(C_BOOL) ne_single_st_short
    logical ne_single_st_short

    ne_single_st_short = cpp_ne_single_st_short(a, b)
  end function ne_single_st_short


  elemental function ne_double_double_st(a, b)
    real(C_DOUBLE), intent(in) :: a
    type(double_st), intent(in) :: b
    !logical(C_BOOL) ne_double_double_st
    logical ne_double_double_st

    ne_double_double_st = cpp_ne_double_double_st(a, b)
  end function ne_double_double_st

  elemental function ne_double_single_st(a, b)
    real(C_DOUBLE), intent(in) :: a
    type(single_st), intent(in) :: b
    !logical(C_BOOL) ne_double_single_st
    logical ne_double_single_st

    ne_double_single_st = cpp_ne_double_single_st(a, b)
  end function ne_double_single_st


  elemental function ne_float_double_st(a, b)
    real(C_FLOAT), intent(in) :: a
    type(double_st), intent(in) :: b
    !logical(C_BOOL) ne_float_double_st
    logical ne_float_double_st

    ne_float_double_st = cpp_ne_float_double_st(a, b)
  end function ne_float_double_st

  elemental function ne_float_single_st(a, b)
    real(C_FLOAT), intent(in) :: a
    type(single_st), intent(in) :: b
    !logical(C_BOOL) ne_float_single_st
    logical ne_float_single_st

    ne_float_single_st = cpp_ne_float_single_st(a, b)
  end function ne_float_single_st


  elemental function ne_long_long_double_st(a, b)
    integer(C_LONG_LONG), intent(in) :: a
    type(double_st), intent(in) :: b
    !logical(C_BOOL) ne_long_long_double_st
    logical ne_long_long_double_st

    ne_long_long_double_st = cpp_ne_long_long_double_st(a, b)
  end function ne_long_long_double_st

  elemental function ne_long_long_single_st(a, b)
    integer(C_LONG_LONG), intent(in) :: a
    type(single_st), intent(in) :: b
    !logical(C_BOOL) ne_long_long_single_st
    logical ne_long_long_single_st

    ne_long_long_single_st = cpp_ne_long_long_single_st(a, b)
  end function ne_long_long_single_st


!CONTAINS_OP(ne, long,     double_st)
!CONTAINS_OP(ne, long,     single_st)

  elemental function ne_int_double_st(a, b)
    integer(C_INT), intent(in) :: a
    type(double_st), intent(in) :: b
    !logical(C_BOOL) ne_int_double_st
    logical ne_int_double_st

    ne_int_double_st = cpp_ne_int_double_st(a, b)
  end function ne_int_double_st

  elemental function ne_int_single_st(a, b)
    integer(C_INT), intent(in) :: a
    type(single_st), intent(in) :: b
    !logical(C_BOOL) ne_int_single_st
    logical ne_int_single_st

    ne_int_single_st = cpp_ne_int_single_st(a, b)
  end function ne_int_single_st


  elemental function ne_short_double_st(a, b)
    integer(C_SHORT), intent(in) :: a
    type(double_st), intent(in) :: b
    !logical(C_BOOL) ne_short_double_st
    logical ne_short_double_st

    ne_short_double_st = cpp_ne_short_double_st(a, b)
  end function ne_short_double_st

  elemental function ne_short_single_st(a, b)
    integer(C_SHORT), intent(in) :: a
    type(single_st), intent(in) :: b
    !logical(C_BOOL) ne_short_single_st
    logical ne_short_single_st

    ne_short_single_st = cpp_ne_short_single_st(a, b)
  end function ne_short_single_st




end module cadna_ne

