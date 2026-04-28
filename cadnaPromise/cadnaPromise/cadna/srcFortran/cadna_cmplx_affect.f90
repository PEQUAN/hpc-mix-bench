
! Module de surcharge de l'opérateur "="





module cadna_cmplx_affect
  use ISO_C_BINDING, only : C_LONG_LONG, C_LONG, C_INT, C_SHORT, C_FLOAT, C_DOUBLE
  use cadna_types
  use cadna_affect
  implicit none




!INTERFACE_OP(=, affect, complex_double_st, complex_double_st)
  interface assignment(=)
    module procedure affect_complex_double_st_complex_single_st
  end interface assignment(=)

  interface assignment(=)
    module procedure affect_complex_double_st_complex
  end interface assignment(=)

  interface assignment(=)
    module procedure affect_complex_double_st_double_st
  end interface assignment(=)

  interface assignment(=)
    module procedure affect_complex_double_st_single_st
  end interface assignment(=)

  interface assignment(=)
    module procedure affect_complex_double_st_double
  end interface assignment(=)

  interface assignment(=)
    module procedure affect_complex_double_st_float
  end interface assignment(=)

  interface assignment(=)
    module procedure affect_complex_double_st_long_long
  end interface assignment(=)

!INTERFACE_OP(=, affect, complex_double_st, long)
  interface assignment(=)
    module procedure affect_complex_double_st_int
  end interface assignment(=)

  interface assignment(=)
    module procedure affect_complex_double_st_short
  end interface assignment(=)

!
  interface assignment(=)
    module procedure affect_complex_single_st_complex_double_st
  end interface assignment(=)

!INTERFACE_OP(=, affect, complex_single_st, complex_single_st)
  interface assignment(=)
    module procedure affect_complex_single_st_complex
  end interface assignment(=)

  interface assignment(=)
    module procedure affect_complex_single_st_double_st
  end interface assignment(=)

  interface assignment(=)
    module procedure affect_complex_single_st_single_st
  end interface assignment(=)

  interface assignment(=)
    module procedure affect_complex_single_st_double
  end interface assignment(=)

  interface assignment(=)
    module procedure affect_complex_single_st_float
  end interface assignment(=)

  interface assignment(=)
    module procedure affect_complex_single_st_long_long
  end interface assignment(=)

!INTERFACE_OP(=, affect, complex_single_st, long)
  interface assignment(=)
    module procedure affect_complex_single_st_int
  end interface assignment(=)

  interface assignment(=)
    module procedure affect_complex_single_st_short
  end interface assignment(=)

!
  interface assignment(=)
    module procedure affect_single_st_complex_double_st
  end interface assignment(=)

  interface assignment(=)
    module procedure affect_single_st_complex_single_st
  end interface assignment(=)


  interface assignment(=)
    module procedure affect_double_st_complex_double_st
  end interface assignment(=)

  interface assignment(=)
    module procedure affect_double_st_complex_single_st
  end interface assignment(=)


  interface assignment(=)
    module procedure affect_complex_complex_double_st
  end interface assignment(=)

  interface assignment(=)
    module procedure affect_complex_complex_single_st
  end interface assignment(=)


  interface assignment(=)
    module procedure affect_double_complex_double_st
  end interface assignment(=)

  interface assignment(=)
    module procedure affect_double_complex_single_st
  end interface assignment(=)

!
  interface assignment(=)
    module procedure affect_float_complex_double_st
  end interface assignment(=)

  interface assignment(=)
    module procedure affect_float_complex_single_st
  end interface assignment(=)

!
  interface assignment(=)
    module procedure affect_long_long_complex_double_st
  end interface assignment(=)

  interface assignment(=)
    module procedure affect_long_long_complex_single_st
  end interface assignment(=)

!
!INTERFACE_OP(=, affect, long,              complex_double_st)
!INTERFACE_OP(=, affect, long,              complex_single_st)
!
  interface assignment(=)
    module procedure affect_int_complex_double_st
  end interface assignment(=)

  interface assignment(=)
    module procedure affect_int_complex_single_st
  end interface assignment(=)

!
  interface assignment(=)
    module procedure affect_short_complex_double_st
  end interface assignment(=)

  interface assignment(=)
    module procedure affect_short_complex_single_st
  end interface assignment(=)


contains



!CONTAINS_OP(affect, complex_double_st, complex_double_st)
  elemental subroutine affect_complex_double_st_complex_single_st(a, b)
    type(complex_double_st), intent(out) :: a
    type(complex_single_st), intent(in) :: b
    a%x = b%x ; a%y = b%y
  end subroutine affect_complex_double_st_complex_single_st

  elemental subroutine affect_complex_double_st_complex(a, b)
    type(complex_double_st), intent(out) :: a
    complex, intent(in) :: b
    a%x = REAL(b)
    a%y = AIMAG(b)
  end subroutine affect_complex_double_st_complex

  elemental subroutine affect_complex_double_st_double_st(a, b)
    type(complex_double_st), intent(out) :: a
    type(double_st), intent(in) :: b
    a%x = b ; a%y = 0.
  end subroutine affect_complex_double_st_double_st

  elemental subroutine affect_complex_double_st_single_st(a, b)
    type(complex_double_st), intent(out) :: a
    type(single_st), intent(in) :: b
    a%x = b ; a%y = 0.
  end subroutine affect_complex_double_st_single_st

  elemental subroutine affect_complex_double_st_double(a, b)
    type(complex_double_st), intent(out) :: a
    real(C_DOUBLE), intent(in) :: b
    a%x = b ; a%y = 0.
  end subroutine affect_complex_double_st_double

  elemental subroutine affect_complex_double_st_float(a, b)
    type(complex_double_st), intent(out) :: a
    real(C_FLOAT), intent(in) :: b
    a%x = b ; a%y = 0.
  end subroutine affect_complex_double_st_float

  elemental subroutine affect_complex_double_st_long_long(a, b)
    type(complex_double_st), intent(out) :: a
    integer(C_LONG_LONG), intent(in) :: b
    a%x = b ; a%y = 0.
  end subroutine affect_complex_double_st_long_long

!CONTAINS_OP(affect, complex_double_st, long)
  elemental subroutine affect_complex_double_st_int(a, b)
    type(complex_double_st), intent(out) :: a
    integer(C_INT), intent(in) :: b
    a%x = b ; a%y = 0.
  end subroutine affect_complex_double_st_int

  elemental subroutine affect_complex_double_st_short(a, b)
    type(complex_double_st), intent(out) :: a
    integer(C_SHORT), intent(in) :: b
    a%x = b ; a%y = 0.
  end subroutine affect_complex_double_st_short

!
  elemental subroutine affect_complex_single_st_complex_double_st(a, b)
    type(complex_single_st), intent(out) :: a
    type(complex_double_st), intent(in) :: b
    a%x = b%x ; a%y = b%y
  end subroutine affect_complex_single_st_complex_double_st

!CONTAINS_OP(affect, complex_single_st, complex_single_st)
  elemental subroutine affect_complex_single_st_complex(a, b)
    type(complex_single_st), intent(out) :: a
    complex, intent(in) :: b
    a%x = REAL(b)
    a%y = AIMAG(b)
  end subroutine affect_complex_single_st_complex

  elemental subroutine affect_complex_single_st_double_st(a, b)
    type(complex_single_st), intent(out) :: a
    type(double_st), intent(in) :: b
    a%x = b ; a%y = 0.
  end subroutine affect_complex_single_st_double_st

  elemental subroutine affect_complex_single_st_single_st(a, b)
    type(complex_single_st), intent(out) :: a
    type(single_st), intent(in) :: b
    a%x = b ; a%y = 0.
  end subroutine affect_complex_single_st_single_st

  elemental subroutine affect_complex_single_st_double(a, b)
    type(complex_single_st), intent(out) :: a
    real(C_DOUBLE), intent(in) :: b
    a%x = b ; a%y = 0.
  end subroutine affect_complex_single_st_double

  elemental subroutine affect_complex_single_st_float(a, b)
    type(complex_single_st), intent(out) :: a
    real(C_FLOAT), intent(in) :: b
    a%x = b ; a%y = 0.
  end subroutine affect_complex_single_st_float

  elemental subroutine affect_complex_single_st_long_long(a, b)
    type(complex_single_st), intent(out) :: a
    integer(C_LONG_LONG), intent(in) :: b
    a%x = b ; a%y = 0.
  end subroutine affect_complex_single_st_long_long

!CONTAINS_OP(affect, complex_single_st, long)
  elemental subroutine affect_complex_single_st_int(a, b)
    type(complex_single_st), intent(out) :: a
    integer(C_INT), intent(in) :: b
    a%x = b ; a%y = 0.
  end subroutine affect_complex_single_st_int

  elemental subroutine affect_complex_single_st_short(a, b)
    type(complex_single_st), intent(out) :: a
    integer(C_SHORT), intent(in) :: b
    a%x = b ; a%y = 0.
  end subroutine affect_complex_single_st_short

!
  elemental subroutine affect_single_st_complex_double_st(a, b)
    type(single_st), intent(out) :: a
    type(complex_double_st), intent(in) :: b
    a = b%x
  end subroutine affect_single_st_complex_double_st

  elemental subroutine affect_single_st_complex_single_st(a, b)
    type(single_st), intent(out) :: a
    type(complex_single_st), intent(in) :: b
    a = b%x
  end subroutine affect_single_st_complex_single_st


  elemental subroutine affect_double_st_complex_double_st(a, b)
    type(double_st), intent(out) :: a
    type(complex_double_st), intent(in) :: b
    a = b%x
  end subroutine affect_double_st_complex_double_st

  elemental subroutine affect_double_st_complex_single_st(a, b)
    type(double_st), intent(out) :: a
    type(complex_single_st), intent(in) :: b
    a = b%x
  end subroutine affect_double_st_complex_single_st


  elemental subroutine affect_complex_complex_double_st(a, b)
    complex, intent(out) :: a
    type(complex_double_st), intent(in) :: b
    real x, y
    
    x = b%x ; y = b%y
    a = cmplx(x, y)
  end subroutine affect_complex_complex_double_st

  elemental subroutine affect_complex_complex_single_st(a, b)
    complex, intent(out) :: a
    type(complex_single_st), intent(in) :: b
    real x, y
    
    x = b%x ; y = b%y
    a = cmplx(x, y)
  end subroutine affect_complex_complex_single_st


  elemental subroutine affect_double_complex_double_st(a, b)
    real(C_DOUBLE), intent(out) :: a
    type(complex_double_st), intent(in) :: b
    a = b%x
  end subroutine affect_double_complex_double_st

  elemental subroutine affect_double_complex_single_st(a, b)
    real(C_DOUBLE), intent(out) :: a
    type(complex_single_st), intent(in) :: b
    a = b%x
  end subroutine affect_double_complex_single_st

!
  elemental subroutine affect_float_complex_double_st(a, b)
    real(C_FLOAT), intent(out) :: a
    type(complex_double_st), intent(in) :: b
    a = b%x
  end subroutine affect_float_complex_double_st

  elemental subroutine affect_float_complex_single_st(a, b)
    real(C_FLOAT), intent(out) :: a
    type(complex_single_st), intent(in) :: b
    a = b%x
  end subroutine affect_float_complex_single_st

!
  elemental subroutine affect_long_long_complex_double_st(a, b)
    integer(C_LONG_LONG), intent(out) :: a
    type(complex_double_st), intent(in) :: b
    a = b%x
  end subroutine affect_long_long_complex_double_st

  elemental subroutine affect_long_long_complex_single_st(a, b)
    integer(C_LONG_LONG), intent(out) :: a
    type(complex_single_st), intent(in) :: b
    a = b%x
  end subroutine affect_long_long_complex_single_st

!
!CONTAINS_OP(affect, long,             complex_double_st)
!CONTAINS_OP(affect, long,             complex_single_st)
!
  elemental subroutine affect_int_complex_double_st(a, b)
    integer(C_INT), intent(out) :: a
    type(complex_double_st), intent(in) :: b
    a = b%x
  end subroutine affect_int_complex_double_st

  elemental subroutine affect_int_complex_single_st(a, b)
    integer(C_INT), intent(out) :: a
    type(complex_single_st), intent(in) :: b
    a = b%x
  end subroutine affect_int_complex_single_st

!
  elemental subroutine affect_short_complex_double_st(a, b)
    integer(C_SHORT), intent(out) :: a
    type(complex_double_st), intent(in) :: b
    a = b%x
  end subroutine affect_short_complex_double_st

  elemental subroutine affect_short_complex_single_st(a, b)
    integer(C_SHORT), intent(out) :: a
    type(complex_single_st), intent(in) :: b
    a = b%x
  end subroutine affect_short_complex_single_st




end module cadna_cmplx_affect

