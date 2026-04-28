
! Module de surcharge de l'opérateur "="




module cadna_affect
  use ISO_C_BINDING, only : C_LONG_LONG, C_LONG, C_INT, C_SHORT, C_FLOAT, C_DOUBLE
  use cadna_types
  implicit none




!INTERFACE_OP(affect, double_st, double_st, double_st)
  interface assignment(=)
    module procedure affect_double_st_single_st
  end interface assignment(=)
  interface
    pure subroutine cpp_affect_double_st_single_st(a, b) bind(C)
      import double_st
      import single_st
      type(double_st), intent(out) :: a
      type(single_st), intent(in) :: b
    end subroutine cpp_affect_double_st_single_st
  end interface

  interface assignment(=)
    module procedure affect_double_st_double
  end interface assignment(=)
  interface
    pure subroutine cpp_affect_double_st_double(a, b) bind(C)
      import double_st
      import C_DOUBLE
      type(double_st), intent(out) :: a
      real(C_DOUBLE), intent(in) :: b
    end subroutine cpp_affect_double_st_double
  end interface

  interface assignment(=)
    module procedure affect_double_st_float
  end interface assignment(=)
  interface
    pure subroutine cpp_affect_double_st_float(a, b) bind(C)
      import double_st
      import C_FLOAT
      type(double_st), intent(out) :: a
      real(C_FLOAT), intent(in) :: b
    end subroutine cpp_affect_double_st_float
  end interface

  interface assignment(=)
    module procedure affect_double_st_long_long
  end interface assignment(=)
  interface
    pure subroutine cpp_affect_double_st_long_long(a, b) bind(C)
      import double_st
      import C_LONG_LONG
      type(double_st), intent(out) :: a
      integer(C_LONG_LONG), intent(in) :: b
    end subroutine cpp_affect_double_st_long_long
  end interface

!INTERFACE_OP(affect, double_st, long,      double_st)
  interface assignment(=)
    module procedure affect_double_st_int
  end interface assignment(=)
  interface
    pure subroutine cpp_affect_double_st_int(a, b) bind(C)
      import double_st
      import C_INT
      type(double_st), intent(out) :: a
      integer(C_INT), intent(in) :: b
    end subroutine cpp_affect_double_st_int
  end interface

  interface assignment(=)
    module procedure affect_double_st_short
  end interface assignment(=)
  interface
    pure subroutine cpp_affect_double_st_short(a, b) bind(C)
      import double_st
      import C_SHORT
      type(double_st), intent(out) :: a
      integer(C_SHORT), intent(in) :: b
    end subroutine cpp_affect_double_st_short
  end interface


  interface assignment(=)
    module procedure affect_single_st_double_st
  end interface assignment(=)
  interface
    pure subroutine cpp_affect_single_st_double_st(a, b) bind(C)
      import single_st
      import double_st
      type(single_st), intent(out) :: a
      type(double_st), intent(in) :: b
    end subroutine cpp_affect_single_st_double_st
  end interface

!INTERFACE_OP(affect, single_st, single_st, single_st)
  interface assignment(=)
    module procedure affect_single_st_double
  end interface assignment(=)
  interface
    pure subroutine cpp_affect_single_st_double(a, b) bind(C)
      import single_st
      import C_DOUBLE
      type(single_st), intent(out) :: a
      real(C_DOUBLE), intent(in) :: b
    end subroutine cpp_affect_single_st_double
  end interface

  interface assignment(=)
    module procedure affect_single_st_float
  end interface assignment(=)
  interface
    pure subroutine cpp_affect_single_st_float(a, b) bind(C)
      import single_st
      import C_FLOAT
      type(single_st), intent(out) :: a
      real(C_FLOAT), intent(in) :: b
    end subroutine cpp_affect_single_st_float
  end interface

  interface assignment(=)
    module procedure affect_single_st_long_long
  end interface assignment(=)
  interface
    pure subroutine cpp_affect_single_st_long_long(a, b) bind(C)
      import single_st
      import C_LONG_LONG
      type(single_st), intent(out) :: a
      integer(C_LONG_LONG), intent(in) :: b
    end subroutine cpp_affect_single_st_long_long
  end interface

!INTERFACE_OP(affect, single_st, long,      single_st)
  interface assignment(=)
    module procedure affect_single_st_int
  end interface assignment(=)
  interface
    pure subroutine cpp_affect_single_st_int(a, b) bind(C)
      import single_st
      import C_INT
      type(single_st), intent(out) :: a
      integer(C_INT), intent(in) :: b
    end subroutine cpp_affect_single_st_int
  end interface

  interface assignment(=)
    module procedure affect_single_st_short
  end interface assignment(=)
  interface
    pure subroutine cpp_affect_single_st_short(a, b) bind(C)
      import single_st
      import C_SHORT
      type(single_st), intent(out) :: a
      integer(C_SHORT), intent(in) :: b
    end subroutine cpp_affect_single_st_short
  end interface


  interface assignment(=)
    module procedure affect_double_double_st
  end interface assignment(=)
  interface
    pure subroutine cpp_affect_double_double_st(a, b) bind(C)
      import C_DOUBLE
      import double_st
      real(C_DOUBLE), intent(out) :: a
      type(double_st), intent(in) :: b
    end subroutine cpp_affect_double_double_st
  end interface

  interface assignment(=)
    module procedure affect_double_single_st
  end interface assignment(=)
  interface
    pure subroutine cpp_affect_double_single_st(a, b) bind(C)
      import C_DOUBLE
      import single_st
      real(C_DOUBLE), intent(out) :: a
      type(single_st), intent(in) :: b
    end subroutine cpp_affect_double_single_st
  end interface


  interface assignment(=)
    module procedure affect_float_double_st
  end interface assignment(=)
  interface
    pure subroutine cpp_affect_float_double_st(a, b) bind(C)
      import C_FLOAT
      import double_st
      real(C_FLOAT), intent(out) :: a
      type(double_st), intent(in) :: b
    end subroutine cpp_affect_float_double_st
  end interface

  interface assignment(=)
    module procedure affect_float_single_st
  end interface assignment(=)
  interface
    pure subroutine cpp_affect_float_single_st(a, b) bind(C)
      import C_FLOAT
      import single_st
      real(C_FLOAT), intent(out) :: a
      type(single_st), intent(in) :: b
    end subroutine cpp_affect_float_single_st
  end interface


  interface assignment(=)
    module procedure affect_long_long_double_st
  end interface assignment(=)
  interface
    pure subroutine cpp_affect_long_long_double_st(a, b) bind(C)
      import C_LONG_LONG
      import double_st
      integer(C_LONG_LONG), intent(out) :: a
      type(double_st), intent(in) :: b
    end subroutine cpp_affect_long_long_double_st
  end interface

  interface assignment(=)
    module procedure affect_long_long_single_st
  end interface assignment(=)
  interface
    pure subroutine cpp_affect_long_long_single_st(a, b) bind(C)
      import C_LONG_LONG
      import single_st
      integer(C_LONG_LONG), intent(out) :: a
      type(single_st), intent(in) :: b
    end subroutine cpp_affect_long_long_single_st
  end interface


!INTERFACE_OP(affect, long, double_st,      double_st)
!INTERFACE_OP(affect, long, single_st,      single_st)

  interface assignment(=)
    module procedure affect_int_double_st
  end interface assignment(=)
  interface
    pure subroutine cpp_affect_int_double_st(a, b) bind(C)
      import C_INT
      import double_st
      integer(C_INT), intent(out) :: a
      type(double_st), intent(in) :: b
    end subroutine cpp_affect_int_double_st
  end interface

  interface assignment(=)
    module procedure affect_int_single_st
  end interface assignment(=)
  interface
    pure subroutine cpp_affect_int_single_st(a, b) bind(C)
      import C_INT
      import single_st
      integer(C_INT), intent(out) :: a
      type(single_st), intent(in) :: b
    end subroutine cpp_affect_int_single_st
  end interface


  interface assignment(=)
    module procedure affect_short_double_st
  end interface assignment(=)
  interface
    pure subroutine cpp_affect_short_double_st(a, b) bind(C)
      import C_SHORT
      import double_st
      integer(C_SHORT), intent(out) :: a
      type(double_st), intent(in) :: b
    end subroutine cpp_affect_short_double_st
  end interface

  interface assignment(=)
    module procedure affect_short_single_st
  end interface assignment(=)
  interface
    pure subroutine cpp_affect_short_single_st(a, b) bind(C)
      import C_SHORT
      import single_st
      integer(C_SHORT), intent(out) :: a
      type(single_st), intent(in) :: b
    end subroutine cpp_affect_short_single_st
  end interface


contains



!CONTAINS_OP(affect, double_st, double_st)
  elemental subroutine affect_double_st_single_st(a, b)
    type(double_st), intent(out) :: a
    type(single_st), intent(in) :: b

    call cpp_affect_double_st_single_st(a, b)
  end subroutine affect_double_st_single_st

  elemental subroutine affect_double_st_double(a, b)
    type(double_st), intent(out) :: a
    real(C_DOUBLE), intent(in) :: b

    call cpp_affect_double_st_double(a, b)
  end subroutine affect_double_st_double

  elemental subroutine affect_double_st_float(a, b)
    type(double_st), intent(out) :: a
    real(C_FLOAT), intent(in) :: b

    call cpp_affect_double_st_float(a, b)
  end subroutine affect_double_st_float

  elemental subroutine affect_double_st_long_long(a, b)
    type(double_st), intent(out) :: a
    integer(C_LONG_LONG), intent(in) :: b

    call cpp_affect_double_st_long_long(a, b)
  end subroutine affect_double_st_long_long

!CONTAINS_OP(affect, double_st, long)
  elemental subroutine affect_double_st_int(a, b)
    type(double_st), intent(out) :: a
    integer(C_INT), intent(in) :: b

    call cpp_affect_double_st_int(a, b)
  end subroutine affect_double_st_int

  elemental subroutine affect_double_st_short(a, b)
    type(double_st), intent(out) :: a
    integer(C_SHORT), intent(in) :: b

    call cpp_affect_double_st_short(a, b)
  end subroutine affect_double_st_short


  elemental subroutine affect_single_st_double_st(a, b)
    type(single_st), intent(out) :: a
    type(double_st), intent(in) :: b

    call cpp_affect_single_st_double_st(a, b)
  end subroutine affect_single_st_double_st

!CONTAINS_OP(affect, single_st, single_st)
  elemental subroutine affect_single_st_double(a, b)
    type(single_st), intent(out) :: a
    real(C_DOUBLE), intent(in) :: b

    call cpp_affect_single_st_double(a, b)
  end subroutine affect_single_st_double

  elemental subroutine affect_single_st_float(a, b)
    type(single_st), intent(out) :: a
    real(C_FLOAT), intent(in) :: b

    call cpp_affect_single_st_float(a, b)
  end subroutine affect_single_st_float

  elemental subroutine affect_single_st_long_long(a, b)
    type(single_st), intent(out) :: a
    integer(C_LONG_LONG), intent(in) :: b

    call cpp_affect_single_st_long_long(a, b)
  end subroutine affect_single_st_long_long

!CONTAINS_OP(affect, single_st, long)
  elemental subroutine affect_single_st_int(a, b)
    type(single_st), intent(out) :: a
    integer(C_INT), intent(in) :: b

    call cpp_affect_single_st_int(a, b)
  end subroutine affect_single_st_int

  elemental subroutine affect_single_st_short(a, b)
    type(single_st), intent(out) :: a
    integer(C_SHORT), intent(in) :: b

    call cpp_affect_single_st_short(a, b)
  end subroutine affect_single_st_short


  elemental subroutine affect_double_double_st(a, b)
    real(C_DOUBLE), intent(out) :: a
    type(double_st), intent(in) :: b

    call cpp_affect_double_double_st(a, b)
  end subroutine affect_double_double_st

  elemental subroutine affect_double_single_st(a, b)
    real(C_DOUBLE), intent(out) :: a
    type(single_st), intent(in) :: b

    call cpp_affect_double_single_st(a, b)
  end subroutine affect_double_single_st


  elemental subroutine affect_float_double_st(a, b)
    real(C_FLOAT), intent(out) :: a
    type(double_st), intent(in) :: b

    call cpp_affect_float_double_st(a, b)
  end subroutine affect_float_double_st

  elemental subroutine affect_float_single_st(a, b)
    real(C_FLOAT), intent(out) :: a
    type(single_st), intent(in) :: b

    call cpp_affect_float_single_st(a, b)
  end subroutine affect_float_single_st


  elemental subroutine affect_long_long_double_st(a, b)
    integer(C_LONG_LONG), intent(out) :: a
    type(double_st), intent(in) :: b

    call cpp_affect_long_long_double_st(a, b)
  end subroutine affect_long_long_double_st

  elemental subroutine affect_long_long_single_st(a, b)
    integer(C_LONG_LONG), intent(out) :: a
    type(single_st), intent(in) :: b

    call cpp_affect_long_long_single_st(a, b)
  end subroutine affect_long_long_single_st


!CONTAINS_OP(affect, long, double_st)
!CONTAINS_OP(affect, long, single_st)

  elemental subroutine affect_int_double_st(a, b)
    integer(C_INT), intent(out) :: a
    type(double_st), intent(in) :: b

    call cpp_affect_int_double_st(a, b)
  end subroutine affect_int_double_st

  elemental subroutine affect_int_single_st(a, b)
    integer(C_INT), intent(out) :: a
    type(single_st), intent(in) :: b

    call cpp_affect_int_single_st(a, b)
  end subroutine affect_int_single_st


  elemental subroutine affect_short_double_st(a, b)
    integer(C_SHORT), intent(out) :: a
    type(double_st), intent(in) :: b

    call cpp_affect_short_double_st(a, b)
  end subroutine affect_short_double_st

  elemental subroutine affect_short_single_st(a, b)
    integer(C_SHORT), intent(out) :: a
    type(single_st), intent(in) :: b

    call cpp_affect_short_single_st(a, b)
  end subroutine affect_short_single_st




end module cadna_affect

