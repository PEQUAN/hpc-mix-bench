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



! Module de surcharge de l'opérateur "<"




module cadna_lt
  use ISO_C_BINDING, only : C_BOOL, C_LONG_LONG, C_LONG, C_INT, C_SHORT, C_FLOAT, C_DOUBLE
  !`use ISO_C_BINDING, only : C_BOOL, C_LONG_LONG, C_LONG, C_INT, C_SHORT, C_FLOAT'
  use cadna_types
  implicit none

!INIT(lt, C_LONG_LONG, C_LONG, C_INT, C_SHORT, C_FLOAT, C_DOUBLE)



  interface operator(<)
    module procedure lt_double_st_double_st
  end interface operator(<)
  interface
    pure function cpp_lt_double_st_double_st(a, b) bind(C)
      import C_BOOL
      import double_st
      
      type(double_st), intent(in) :: a
      type(double_st), intent(in) :: b
      logical(C_BOOL) :: cpp_lt_double_st_double_st
      !logical :: cpp_lt_double_st_double_st
    end function cpp_lt_double_st_double_st
  end interface

  interface operator(<)
    module procedure lt_double_st_single_st
  end interface operator(<)
  interface
    pure function cpp_lt_double_st_single_st(a, b) bind(C)
      import C_BOOL
      import double_st
      import single_st
      type(double_st), intent(in) :: a
      type(single_st), intent(in) :: b
      logical(C_BOOL) :: cpp_lt_double_st_single_st
      !logical :: cpp_lt_double_st_single_st
    end function cpp_lt_double_st_single_st
  end interface

  interface operator(<)
    module procedure lt_double_st_double
  end interface operator(<)
  interface
    pure function cpp_lt_double_st_double(a, b) bind(C)
      import C_BOOL
      import double_st
      import C_DOUBLE
      type(double_st), intent(in) :: a
      real(C_DOUBLE), intent(in) :: b
      logical(C_BOOL) :: cpp_lt_double_st_double
      !logical :: cpp_lt_double_st_double
    end function cpp_lt_double_st_double
  end interface

  interface operator(<)
    module procedure lt_double_st_float
  end interface operator(<)
  interface
    pure function cpp_lt_double_st_float(a, b) bind(C)
      import C_BOOL
      import double_st
      import C_FLOAT
      type(double_st), intent(in) :: a
      real(C_FLOAT), intent(in) :: b
      logical(C_BOOL) :: cpp_lt_double_st_float
      !logical :: cpp_lt_double_st_float
    end function cpp_lt_double_st_float
  end interface

  interface operator(<)
    module procedure lt_double_st_long_long
  end interface operator(<)
  interface
    pure function cpp_lt_double_st_long_long(a, b) bind(C)
      import C_BOOL
      import double_st
      import C_LONG_LONG
      type(double_st), intent(in) :: a
      integer(C_LONG_LONG), intent(in) :: b
      logical(C_BOOL) :: cpp_lt_double_st_long_long
      !logical :: cpp_lt_double_st_long_long
    end function cpp_lt_double_st_long_long
  end interface

!INTERFACE_OP(<, lt, double_st, long)
  interface operator(<)
    module procedure lt_double_st_int
  end interface operator(<)
  interface
    pure function cpp_lt_double_st_int(a, b) bind(C)
      import C_BOOL
      import double_st
      import C_INT
      type(double_st), intent(in) :: a
      integer(C_INT), intent(in) :: b
      logical(C_BOOL) :: cpp_lt_double_st_int
      !logical :: cpp_lt_double_st_int
    end function cpp_lt_double_st_int
  end interface

  interface operator(<)
    module procedure lt_double_st_short
  end interface operator(<)
  interface
    pure function cpp_lt_double_st_short(a, b) bind(C)
      import C_BOOL
      import double_st
      import C_SHORT
      type(double_st), intent(in) :: a
      integer(C_SHORT), intent(in) :: b
      logical(C_BOOL) :: cpp_lt_double_st_short
      !logical :: cpp_lt_double_st_short
    end function cpp_lt_double_st_short
  end interface


  interface operator(<)
    module procedure lt_single_st_double_st
  end interface operator(<)
  interface
    pure function cpp_lt_single_st_double_st(a, b) bind(C)
      import C_BOOL
      import single_st
      import double_st
      type(single_st), intent(in) :: a
      type(double_st), intent(in) :: b
      logical(C_BOOL) :: cpp_lt_single_st_double_st
      !logical :: cpp_lt_single_st_double_st
    end function cpp_lt_single_st_double_st
  end interface

  interface operator(<)
    module procedure lt_single_st_single_st
  end interface operator(<)
  interface
    pure function cpp_lt_single_st_single_st(a, b) bind(C)
      import C_BOOL
      import single_st
      
      type(single_st), intent(in) :: a
      type(single_st), intent(in) :: b
      logical(C_BOOL) :: cpp_lt_single_st_single_st
      !logical :: cpp_lt_single_st_single_st
    end function cpp_lt_single_st_single_st
  end interface

  interface operator(<)
    module procedure lt_single_st_double
  end interface operator(<)
  interface
    pure function cpp_lt_single_st_double(a, b) bind(C)
      import C_BOOL
      import single_st
      import C_DOUBLE
      type(single_st), intent(in) :: a
      real(C_DOUBLE), intent(in) :: b
      logical(C_BOOL) :: cpp_lt_single_st_double
      !logical :: cpp_lt_single_st_double
    end function cpp_lt_single_st_double
  end interface

  interface operator(<)
    module procedure lt_single_st_float
  end interface operator(<)
  interface
    pure function cpp_lt_single_st_float(a, b) bind(C)
      import C_BOOL
      import single_st
      import C_FLOAT
      type(single_st), intent(in) :: a
      real(C_FLOAT), intent(in) :: b
      logical(C_BOOL) :: cpp_lt_single_st_float
      !logical :: cpp_lt_single_st_float
    end function cpp_lt_single_st_float
  end interface

  interface operator(<)
    module procedure lt_single_st_long_long
  end interface operator(<)
  interface
    pure function cpp_lt_single_st_long_long(a, b) bind(C)
      import C_BOOL
      import single_st
      import C_LONG_LONG
      type(single_st), intent(in) :: a
      integer(C_LONG_LONG), intent(in) :: b
      logical(C_BOOL) :: cpp_lt_single_st_long_long
      !logical :: cpp_lt_single_st_long_long
    end function cpp_lt_single_st_long_long
  end interface

!INTERFACE_OP(<, lt, single_st, long)
  interface operator(<)
    module procedure lt_single_st_int
  end interface operator(<)
  interface
    pure function cpp_lt_single_st_int(a, b) bind(C)
      import C_BOOL
      import single_st
      import C_INT
      type(single_st), intent(in) :: a
      integer(C_INT), intent(in) :: b
      logical(C_BOOL) :: cpp_lt_single_st_int
      !logical :: cpp_lt_single_st_int
    end function cpp_lt_single_st_int
  end interface

  interface operator(<)
    module procedure lt_single_st_short
  end interface operator(<)
  interface
    pure function cpp_lt_single_st_short(a, b) bind(C)
      import C_BOOL
      import single_st
      import C_SHORT
      type(single_st), intent(in) :: a
      integer(C_SHORT), intent(in) :: b
      logical(C_BOOL) :: cpp_lt_single_st_short
      !logical :: cpp_lt_single_st_short
    end function cpp_lt_single_st_short
  end interface


  interface operator(<)
    module procedure lt_double_double_st
  end interface operator(<)
  interface
    pure function cpp_lt_double_double_st(a, b) bind(C)
      import C_BOOL
      import C_DOUBLE
      import double_st
      real(C_DOUBLE), intent(in) :: a
      type(double_st), intent(in) :: b
      logical(C_BOOL) :: cpp_lt_double_double_st
      !logical :: cpp_lt_double_double_st
    end function cpp_lt_double_double_st
  end interface

  interface operator(<)
    module procedure lt_double_single_st
  end interface operator(<)
  interface
    pure function cpp_lt_double_single_st(a, b) bind(C)
      import C_BOOL
      import C_DOUBLE
      import single_st
      real(C_DOUBLE), intent(in) :: a
      type(single_st), intent(in) :: b
      logical(C_BOOL) :: cpp_lt_double_single_st
      !logical :: cpp_lt_double_single_st
    end function cpp_lt_double_single_st
  end interface


  interface operator(<)
    module procedure lt_float_double_st
  end interface operator(<)
  interface
    pure function cpp_lt_float_double_st(a, b) bind(C)
      import C_BOOL
      import C_FLOAT
      import double_st
      real(C_FLOAT), intent(in) :: a
      type(double_st), intent(in) :: b
      logical(C_BOOL) :: cpp_lt_float_double_st
      !logical :: cpp_lt_float_double_st
    end function cpp_lt_float_double_st
  end interface

  interface operator(<)
    module procedure lt_float_single_st
  end interface operator(<)
  interface
    pure function cpp_lt_float_single_st(a, b) bind(C)
      import C_BOOL
      import C_FLOAT
      import single_st
      real(C_FLOAT), intent(in) :: a
      type(single_st), intent(in) :: b
      logical(C_BOOL) :: cpp_lt_float_single_st
      !logical :: cpp_lt_float_single_st
    end function cpp_lt_float_single_st
  end interface


  interface operator(<)
    module procedure lt_long_long_double_st
  end interface operator(<)
  interface
    pure function cpp_lt_long_long_double_st(a, b) bind(C)
      import C_BOOL
      import C_LONG_LONG
      import double_st
      integer(C_LONG_LONG), intent(in) :: a
      type(double_st), intent(in) :: b
      logical(C_BOOL) :: cpp_lt_long_long_double_st
      !logical :: cpp_lt_long_long_double_st
    end function cpp_lt_long_long_double_st
  end interface

  interface operator(<)
    module procedure lt_long_long_single_st
  end interface operator(<)
  interface
    pure function cpp_lt_long_long_single_st(a, b) bind(C)
      import C_BOOL
      import C_LONG_LONG
      import single_st
      integer(C_LONG_LONG), intent(in) :: a
      type(single_st), intent(in) :: b
      logical(C_BOOL) :: cpp_lt_long_long_single_st
      !logical :: cpp_lt_long_long_single_st
    end function cpp_lt_long_long_single_st
  end interface


!INTERFACE_OP(<, lt, long,     double_st)
!INTERFACE_OP(<, lt, long,     single_st)

  interface operator(<)
    module procedure lt_int_double_st
  end interface operator(<)
  interface
    pure function cpp_lt_int_double_st(a, b) bind(C)
      import C_BOOL
      import C_INT
      import double_st
      integer(C_INT), intent(in) :: a
      type(double_st), intent(in) :: b
      logical(C_BOOL) :: cpp_lt_int_double_st
      !logical :: cpp_lt_int_double_st
    end function cpp_lt_int_double_st
  end interface

  interface operator(<)
    module procedure lt_int_single_st
  end interface operator(<)
  interface
    pure function cpp_lt_int_single_st(a, b) bind(C)
      import C_BOOL
      import C_INT
      import single_st
      integer(C_INT), intent(in) :: a
      type(single_st), intent(in) :: b
      logical(C_BOOL) :: cpp_lt_int_single_st
      !logical :: cpp_lt_int_single_st
    end function cpp_lt_int_single_st
  end interface


  interface operator(<)
    module procedure lt_short_double_st
  end interface operator(<)
  interface
    pure function cpp_lt_short_double_st(a, b) bind(C)
      import C_BOOL
      import C_SHORT
      import double_st
      integer(C_SHORT), intent(in) :: a
      type(double_st), intent(in) :: b
      logical(C_BOOL) :: cpp_lt_short_double_st
      !logical :: cpp_lt_short_double_st
    end function cpp_lt_short_double_st
  end interface

  interface operator(<)
    module procedure lt_short_single_st
  end interface operator(<)
  interface
    pure function cpp_lt_short_single_st(a, b) bind(C)
      import C_BOOL
      import C_SHORT
      import single_st
      integer(C_SHORT), intent(in) :: a
      type(single_st), intent(in) :: b
      logical(C_BOOL) :: cpp_lt_short_single_st
      !logical :: cpp_lt_short_single_st
    end function cpp_lt_short_single_st
  end interface


contains



  elemental function lt_double_st_double_st(a, b)
    type(double_st), intent(in) :: a
    type(double_st), intent(in) :: b
    !logical(C_BOOL) lt_double_st_double_st
    logical lt_double_st_double_st

    lt_double_st_double_st = cpp_lt_double_st_double_st(a, b)
  end function lt_double_st_double_st

  elemental function lt_double_st_single_st(a, b)
    type(double_st), intent(in) :: a
    type(single_st), intent(in) :: b
    !logical(C_BOOL) lt_double_st_single_st
    logical lt_double_st_single_st

    lt_double_st_single_st = cpp_lt_double_st_single_st(a, b)
  end function lt_double_st_single_st

  elemental function lt_double_st_double(a, b)
    type(double_st), intent(in) :: a
    real(C_DOUBLE), intent(in) :: b
    !logical(C_BOOL) lt_double_st_double
    logical lt_double_st_double

    lt_double_st_double = cpp_lt_double_st_double(a, b)
  end function lt_double_st_double

  elemental function lt_double_st_float(a, b)
    type(double_st), intent(in) :: a
    real(C_FLOAT), intent(in) :: b
    !logical(C_BOOL) lt_double_st_float
    logical lt_double_st_float

    lt_double_st_float = cpp_lt_double_st_float(a, b)
  end function lt_double_st_float

  elemental function lt_double_st_long_long(a, b)
    type(double_st), intent(in) :: a
    integer(C_LONG_LONG), intent(in) :: b
    !logical(C_BOOL) lt_double_st_long_long
    logical lt_double_st_long_long

    lt_double_st_long_long = cpp_lt_double_st_long_long(a, b)
  end function lt_double_st_long_long

!CONTAINS_OP(lt, double_st, long)
  elemental function lt_double_st_int(a, b)
    type(double_st), intent(in) :: a
    integer(C_INT), intent(in) :: b
    !logical(C_BOOL) lt_double_st_int
    logical lt_double_st_int

    lt_double_st_int = cpp_lt_double_st_int(a, b)
  end function lt_double_st_int

  elemental function lt_double_st_short(a, b)
    type(double_st), intent(in) :: a
    integer(C_SHORT), intent(in) :: b
    !logical(C_BOOL) lt_double_st_short
    logical lt_double_st_short

    lt_double_st_short = cpp_lt_double_st_short(a, b)
  end function lt_double_st_short


  elemental function lt_single_st_double_st(a, b)
    type(single_st), intent(in) :: a
    type(double_st), intent(in) :: b
    !logical(C_BOOL) lt_single_st_double_st
    logical lt_single_st_double_st

    lt_single_st_double_st = cpp_lt_single_st_double_st(a, b)
  end function lt_single_st_double_st

  elemental function lt_single_st_single_st(a, b)
    type(single_st), intent(in) :: a
    type(single_st), intent(in) :: b
    !logical(C_BOOL) lt_single_st_single_st
    logical lt_single_st_single_st

    lt_single_st_single_st = cpp_lt_single_st_single_st(a, b)
  end function lt_single_st_single_st

  elemental function lt_single_st_double(a, b)
    type(single_st), intent(in) :: a
    real(C_DOUBLE), intent(in) :: b
    !logical(C_BOOL) lt_single_st_double
    logical lt_single_st_double

    lt_single_st_double = cpp_lt_single_st_double(a, b)
  end function lt_single_st_double

  elemental function lt_single_st_float(a, b)
    type(single_st), intent(in) :: a
    real(C_FLOAT), intent(in) :: b
    !logical(C_BOOL) lt_single_st_float
    logical lt_single_st_float

    lt_single_st_float = cpp_lt_single_st_float(a, b)
  end function lt_single_st_float

  elemental function lt_single_st_long_long(a, b)
    type(single_st), intent(in) :: a
    integer(C_LONG_LONG), intent(in) :: b
    !logical(C_BOOL) lt_single_st_long_long
    logical lt_single_st_long_long

    lt_single_st_long_long = cpp_lt_single_st_long_long(a, b)
  end function lt_single_st_long_long

!CONTAINS_OP(lt, single_st, long)
  elemental function lt_single_st_int(a, b)
    type(single_st), intent(in) :: a
    integer(C_INT), intent(in) :: b
    !logical(C_BOOL) lt_single_st_int
    logical lt_single_st_int

    lt_single_st_int = cpp_lt_single_st_int(a, b)
  end function lt_single_st_int

  elemental function lt_single_st_short(a, b)
    type(single_st), intent(in) :: a
    integer(C_SHORT), intent(in) :: b
    !logical(C_BOOL) lt_single_st_short
    logical lt_single_st_short

    lt_single_st_short = cpp_lt_single_st_short(a, b)
  end function lt_single_st_short


  elemental function lt_double_double_st(a, b)
    real(C_DOUBLE), intent(in) :: a
    type(double_st), intent(in) :: b
    !logical(C_BOOL) lt_double_double_st
    logical lt_double_double_st

    lt_double_double_st = cpp_lt_double_double_st(a, b)
  end function lt_double_double_st

  elemental function lt_double_single_st(a, b)
    real(C_DOUBLE), intent(in) :: a
    type(single_st), intent(in) :: b
    !logical(C_BOOL) lt_double_single_st
    logical lt_double_single_st

    lt_double_single_st = cpp_lt_double_single_st(a, b)
  end function lt_double_single_st


  elemental function lt_float_double_st(a, b)
    real(C_FLOAT), intent(in) :: a
    type(double_st), intent(in) :: b
    !logical(C_BOOL) lt_float_double_st
    logical lt_float_double_st

    lt_float_double_st = cpp_lt_float_double_st(a, b)
  end function lt_float_double_st

  elemental function lt_float_single_st(a, b)
    real(C_FLOAT), intent(in) :: a
    type(single_st), intent(in) :: b
    !logical(C_BOOL) lt_float_single_st
    logical lt_float_single_st

    lt_float_single_st = cpp_lt_float_single_st(a, b)
  end function lt_float_single_st


  elemental function lt_long_long_double_st(a, b)
    integer(C_LONG_LONG), intent(in) :: a
    type(double_st), intent(in) :: b
    !logical(C_BOOL) lt_long_long_double_st
    logical lt_long_long_double_st

    lt_long_long_double_st = cpp_lt_long_long_double_st(a, b)
  end function lt_long_long_double_st

  elemental function lt_long_long_single_st(a, b)
    integer(C_LONG_LONG), intent(in) :: a
    type(single_st), intent(in) :: b
    !logical(C_BOOL) lt_long_long_single_st
    logical lt_long_long_single_st

    lt_long_long_single_st = cpp_lt_long_long_single_st(a, b)
  end function lt_long_long_single_st


!CONTAINS_OP(lt, long,     double_st)
!CONTAINS_OP(lt, long,     single_st)

  elemental function lt_int_double_st(a, b)
    integer(C_INT), intent(in) :: a
    type(double_st), intent(in) :: b
    !logical(C_BOOL) lt_int_double_st
    logical lt_int_double_st

    lt_int_double_st = cpp_lt_int_double_st(a, b)
  end function lt_int_double_st

  elemental function lt_int_single_st(a, b)
    integer(C_INT), intent(in) :: a
    type(single_st), intent(in) :: b
    !logical(C_BOOL) lt_int_single_st
    logical lt_int_single_st

    lt_int_single_st = cpp_lt_int_single_st(a, b)
  end function lt_int_single_st


  elemental function lt_short_double_st(a, b)
    integer(C_SHORT), intent(in) :: a
    type(double_st), intent(in) :: b
    !logical(C_BOOL) lt_short_double_st
    logical lt_short_double_st

    lt_short_double_st = cpp_lt_short_double_st(a, b)
  end function lt_short_double_st

  elemental function lt_short_single_st(a, b)
    integer(C_SHORT), intent(in) :: a
    type(single_st), intent(in) :: b
    !logical(C_BOOL) lt_short_single_st
    logical lt_short_single_st

    lt_short_single_st = cpp_lt_short_single_st(a, b)
  end function lt_short_single_st




end module cadna_lt

