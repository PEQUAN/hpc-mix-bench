
! Module de surcharge de l'opérateur "/"





module cadna_cmplx_div
  use ISO_C_BINDING, only : C_LONG_LONG, C_LONG, C_INT, C_SHORT, C_FLOAT, C_DOUBLE
  use ISO_FORTRAN_ENV, only : REAL32, REAL64
  use cadna_types
  use cadna_add
  use cadna_div
  use cadna_affect
  use cadna_pow
  use cadna_cmplx_mult
  use cadna_cmplx_convert
  implicit none




  interface operator(/)
    module procedure div_complex_double_st_complex_double_st
  end interface operator(/)

  interface operator(/)
    module procedure div_complex_double_st_complex_single_st
  end interface operator(/)

  interface operator(/)
    module procedure div_complex_double_st_double_st
  end interface operator(/)

  interface operator(/)
    module procedure div_complex_double_st_single_st
  end interface operator(/)

  interface operator(/)
    module procedure div_complex_double_st_complex_float
  end interface operator(/)

  interface operator(/)
    module procedure div_complex_double_st_complex_double
  end interface operator(/)

  interface operator(/)
    module procedure div_complex_double_st_double
  end interface operator(/)

  interface operator(/)
    module procedure div_complex_double_st_float
  end interface operator(/)

  interface operator(/)
    module procedure div_complex_double_st_long_long
  end interface operator(/)

!INTERFACE_OP(/, div, complex_double_st, long)
  interface operator(/)
    module procedure div_complex_double_st_int
  end interface operator(/)

  interface operator(/)
    module procedure div_complex_double_st_short
  end interface operator(/)

!
  interface operator(/)
    module procedure div_complex_single_st_complex_double_st
  end interface operator(/)

  interface operator(/)
    module procedure div_complex_single_st_complex_single_st
  end interface operator(/)

  interface operator(/)
    module procedure div_complex_single_st_double_st
  end interface operator(/)

  interface operator(/)
    module procedure div_complex_single_st_single_st
  end interface operator(/)

  interface operator(/)
    module procedure div_complex_single_st_complex_float
  end interface operator(/)

  interface operator(/)
    module procedure div_complex_single_st_complex_double
  end interface operator(/)

  interface operator(/)
    module procedure div_complex_single_st_double
  end interface operator(/)

  interface operator(/)
    module procedure div_complex_single_st_float
  end interface operator(/)

  interface operator(/)
    module procedure div_complex_single_st_long_long
  end interface operator(/)

!INTERFACE_OP(/, div, complex_single_st, long)
  interface operator(/)
    module procedure div_complex_single_st_int
  end interface operator(/)

  interface operator(/)
    module procedure div_complex_single_st_short
  end interface operator(/)

!
  interface operator(/)
    module procedure div_single_st_complex_double_st
  end interface operator(/)

  interface operator(/)
    module procedure div_single_st_complex_single_st
  end interface operator(/)


  interface operator(/)
    module procedure div_double_st_complex_double_st
  end interface operator(/)

  interface operator(/)
    module procedure div_double_st_complex_single_st
  end interface operator(/)


  interface operator(/)
    module procedure div_complex_float_complex_single_st
  end interface operator(/)

  interface operator(/)
    module procedure div_complex_float_complex_double_st
  end interface operator(/)


  interface operator(/)
    module procedure div_complex_double_complex_single_st
  end interface operator(/)

  interface operator(/)
    module procedure div_complex_double_complex_double_st
  end interface operator(/)


  interface operator(/)
    module procedure div_double_complex_double_st
  end interface operator(/)

  interface operator(/)
    module procedure div_double_complex_single_st
  end interface operator(/)

!
  interface operator(/)
    module procedure div_float_complex_double_st
  end interface operator(/)

  interface operator(/)
    module procedure div_float_complex_single_st
  end interface operator(/)

!
  interface operator(/)
    module procedure div_long_long_complex_double_st
  end interface operator(/)

  interface operator(/)
    module procedure div_long_long_complex_single_st
  end interface operator(/)

!
!INTERFACE_OP(/, div, long,              complex_double_st)
!INTERFACE_OP(/, div, long,              complex_single_st)
!
  interface operator(/)
    module procedure div_int_complex_double_st
  end interface operator(/)

  interface operator(/)
    module procedure div_int_complex_single_st
  end interface operator(/)

!
  interface operator(/)
    module procedure div_short_complex_double_st
  end interface operator(/)

  interface operator(/)
    module procedure div_short_complex_single_st
  end interface operator(/)


contains



  elemental function div_complex_double_st_complex_double_st(a, b)
    type(complex_double_st), intent(in) :: a
    type(complex_double_st), intent(in) :: b
    type(complex_double_st) div_complex_double_st_complex_double_st
    ! Local declarations
    type(double_st) norme2

    norme2 = b%x**2 + b%y**2
    div_complex_double_st_complex_double_st = a*conjg(b)/norme2
  end function div_complex_double_st_complex_double_st

  elemental function div_complex_double_st_complex_single_st(a, b)
    type(complex_double_st), intent(in) :: a
    type(complex_single_st), intent(in) :: b
    type(complex_double_st) div_complex_double_st_complex_single_st
    ! Local declarations
    type(double_st) norme2

    norme2 = b%x**2 + b%y**2
    div_complex_double_st_complex_single_st = a*conjg(b)/norme2
  end function div_complex_double_st_complex_single_st

  elemental function div_complex_double_st_double_st(a, b)
    type(complex_double_st), intent(in) :: a
    type(double_st), intent(in) :: b
    type(complex_double_st) div_complex_double_st_double_st
    ! Local declarations
    

    div_complex_double_st_double_st%x = a%x/b
    div_complex_double_st_double_st%y = a%y/b
  end function div_complex_double_st_double_st

  elemental function div_complex_double_st_single_st(a, b)
    type(complex_double_st), intent(in) :: a
    type(single_st), intent(in) :: b
    type(complex_double_st) div_complex_double_st_single_st
    ! Local declarations
    

    div_complex_double_st_single_st%x = a%x/b
    div_complex_double_st_single_st%y = a%y/b
  end function div_complex_double_st_single_st

  elemental function div_complex_double_st_complex_float(a, b)
    type(complex_double_st), intent(in) :: a
    complex(REAL32), intent(in) :: b
    type(complex_double_st) div_complex_double_st_complex_float
    ! Local declarations
    real(REAL32) norme2

    norme2 = real(b)**2 + aimag(b)**2
    div_complex_double_st_complex_float = a*conjg(b)/norme2
  end function div_complex_double_st_complex_float

  elemental function div_complex_double_st_complex_double(a, b)
    type(complex_double_st), intent(in) :: a
    complex(REAL64), intent(in) :: b
    type(complex_double_st) div_complex_double_st_complex_double
    ! Local declarations
    real(REAL64) norme2

    norme2 = real(b)**2 + aimag(b)**2
    div_complex_double_st_complex_double = a*conjg(b)/norme2
  end function div_complex_double_st_complex_double

  elemental function div_complex_double_st_double(a, b)
    type(complex_double_st), intent(in) :: a
    real(C_DOUBLE), intent(in) :: b
    type(complex_double_st) div_complex_double_st_double
    ! Local declarations
    

    div_complex_double_st_double%x = a%x/b
    div_complex_double_st_double%y = a%y/b
  end function div_complex_double_st_double

  elemental function div_complex_double_st_float(a, b)
    type(complex_double_st), intent(in) :: a
    real(C_FLOAT), intent(in) :: b
    type(complex_double_st) div_complex_double_st_float
    ! Local declarations
    

    div_complex_double_st_float%x = a%x/b
    div_complex_double_st_float%y = a%y/b
  end function div_complex_double_st_float

  elemental function div_complex_double_st_long_long(a, b)
    type(complex_double_st), intent(in) :: a
    integer(C_LONG_LONG), intent(in) :: b
    type(complex_double_st) div_complex_double_st_long_long
    ! Local declarations
    

    div_complex_double_st_long_long%x = a%x/b
    div_complex_double_st_long_long%y = a%y/b
  end function div_complex_double_st_long_long

!CONTAINS_OP(div, complex_double_st, long,              complex_double_st)
  elemental function div_complex_double_st_int(a, b)
    type(complex_double_st), intent(in) :: a
    integer(C_INT), intent(in) :: b
    type(complex_double_st) div_complex_double_st_int
    ! Local declarations
    

    div_complex_double_st_int%x = a%x/b
    div_complex_double_st_int%y = a%y/b
  end function div_complex_double_st_int

  elemental function div_complex_double_st_short(a, b)
    type(complex_double_st), intent(in) :: a
    integer(C_SHORT), intent(in) :: b
    type(complex_double_st) div_complex_double_st_short
    ! Local declarations
    

    div_complex_double_st_short%x = a%x/b
    div_complex_double_st_short%y = a%y/b
  end function div_complex_double_st_short

!
  elemental function div_complex_single_st_complex_double_st(a, b)
    type(complex_single_st), intent(in) :: a
    type(complex_double_st), intent(in) :: b
    type(complex_double_st) div_complex_single_st_complex_double_st
    ! Local declarations
    type(double_st) norme2

    norme2 = b%x**2 + b%y**2
    div_complex_single_st_complex_double_st = a*conjg(b)/norme2
  end function div_complex_single_st_complex_double_st

  elemental function div_complex_single_st_complex_single_st(a, b)
    type(complex_single_st), intent(in) :: a
    type(complex_single_st), intent(in) :: b
    type(complex_single_st) div_complex_single_st_complex_single_st
    ! Local declarations
    type(single_st) norme2

    norme2 = b%x**2 + b%y**2
    div_complex_single_st_complex_single_st = a*conjg(b)/norme2
  end function div_complex_single_st_complex_single_st

  elemental function div_complex_single_st_double_st(a, b)
    type(complex_single_st), intent(in) :: a
    type(double_st), intent(in) :: b
    type(complex_double_st) div_complex_single_st_double_st
    ! Local declarations
    

    div_complex_single_st_double_st%x = a%x/b
    div_complex_single_st_double_st%y = a%y/b
  end function div_complex_single_st_double_st

  elemental function div_complex_single_st_single_st(a, b)
    type(complex_single_st), intent(in) :: a
    type(single_st), intent(in) :: b
    type(complex_single_st) div_complex_single_st_single_st
    ! Local declarations
    

    div_complex_single_st_single_st%x = a%x/b
    div_complex_single_st_single_st%y = a%y/b
  end function div_complex_single_st_single_st

  elemental function div_complex_single_st_complex_float(a, b)
    type(complex_single_st), intent(in) :: a
    complex(REAL32), intent(in) :: b
    type(complex_single_st) div_complex_single_st_complex_float
    ! Local declarations
    real(REAL32) norme2

    norme2 = real(b)**2 + aimag(b)**2
    div_complex_single_st_complex_float = a*conjg(b)/norme2
  end function div_complex_single_st_complex_float

  elemental function div_complex_single_st_complex_double(a, b)
    type(complex_single_st), intent(in) :: a
    complex(REAL64), intent(in) :: b
    type(complex_single_st) div_complex_single_st_complex_double
    ! Local declarations
    real(REAL64) norme2

    norme2 = real(b)**2 + aimag(b)**2
    div_complex_single_st_complex_double = a*conjg(b)/norme2
  end function div_complex_single_st_complex_double

  elemental function div_complex_single_st_double(a, b)
    type(complex_single_st), intent(in) :: a
    real(C_DOUBLE), intent(in) :: b
    type(complex_single_st) div_complex_single_st_double
    ! Local declarations
    

    div_complex_single_st_double%x = a%x/b
    div_complex_single_st_double%y = a%y/b
  end function div_complex_single_st_double

  elemental function div_complex_single_st_float(a, b)
    type(complex_single_st), intent(in) :: a
    real(C_FLOAT), intent(in) :: b
    type(complex_single_st) div_complex_single_st_float
    ! Local declarations
    

    div_complex_single_st_float%x = a%x/b
    div_complex_single_st_float%y = a%y/b
  end function div_complex_single_st_float

  elemental function div_complex_single_st_long_long(a, b)
    type(complex_single_st), intent(in) :: a
    integer(C_LONG_LONG), intent(in) :: b
    type(complex_single_st) div_complex_single_st_long_long
    ! Local declarations
    

    div_complex_single_st_long_long%x = a%x/b
    div_complex_single_st_long_long%y = a%y/b
  end function div_complex_single_st_long_long

!CONTAINS_OP(div, complex_single_st, long,              complex_single_st)
  elemental function div_complex_single_st_int(a, b)
    type(complex_single_st), intent(in) :: a
    integer(C_INT), intent(in) :: b
    type(complex_single_st) div_complex_single_st_int
    ! Local declarations
    

    div_complex_single_st_int%x = a%x/b
    div_complex_single_st_int%y = a%y/b
  end function div_complex_single_st_int

  elemental function div_complex_single_st_short(a, b)
    type(complex_single_st), intent(in) :: a
    integer(C_SHORT), intent(in) :: b
    type(complex_single_st) div_complex_single_st_short
    ! Local declarations
    

    div_complex_single_st_short%x = a%x/b
    div_complex_single_st_short%y = a%y/b
  end function div_complex_single_st_short

!
  elemental function div_single_st_complex_double_st(a, b)
    type(single_st), intent(in) :: a
    type(complex_double_st), intent(in) :: b
    type(complex_double_st) div_single_st_complex_double_st
    ! Local declarations
    type(double_st) norme2

    norme2 = b%x**2 + b%y**2
    div_single_st_complex_double_st = a*conjg(b)/norme2
  end function div_single_st_complex_double_st

  elemental function div_single_st_complex_single_st(a, b)
    type(single_st), intent(in) :: a
    type(complex_single_st), intent(in) :: b
    type(complex_single_st) div_single_st_complex_single_st
    ! Local declarations
    type(single_st) norme2

    norme2 = b%x**2 + b%y**2
    div_single_st_complex_single_st = a*conjg(b)/norme2
  end function div_single_st_complex_single_st


  elemental function div_double_st_complex_double_st(a, b)
    type(double_st), intent(in) :: a
    type(complex_double_st), intent(in) :: b
    type(complex_double_st) div_double_st_complex_double_st
    ! Local declarations
    type(double_st) norme2

    norme2 = b%x**2 + b%y**2
    div_double_st_complex_double_st = a*conjg(b)/norme2
  end function div_double_st_complex_double_st

  elemental function div_double_st_complex_single_st(a, b)
    type(double_st), intent(in) :: a
    type(complex_single_st), intent(in) :: b
    type(complex_double_st) div_double_st_complex_single_st
    ! Local declarations
    type(double_st) norme2

    norme2 = b%x**2 + b%y**2
    div_double_st_complex_single_st = a*conjg(b)/norme2
  end function div_double_st_complex_single_st


  elemental function div_complex_float_complex_double_st(a, b)
    complex(REAL32), intent(in) :: a
    type(complex_double_st), intent(in) :: b
    type(complex_double_st) div_complex_float_complex_double_st
    ! Local declarations
    type(double_st) norme2

    norme2 = b%x**2 + b%y**2
    div_complex_float_complex_double_st = a*conjg(b)/norme2
  end function div_complex_float_complex_double_st

  elemental function div_complex_float_complex_single_st(a, b)
    complex(REAL32), intent(in) :: a
    type(complex_single_st), intent(in) :: b
    type(complex_single_st) div_complex_float_complex_single_st
    ! Local declarations
    type(single_st) norme2

    norme2 = b%x**2 + b%y**2
    div_complex_float_complex_single_st = a*conjg(b)/norme2
  end function div_complex_float_complex_single_st


  elemental function div_complex_double_complex_double_st(a, b)
    complex(REAL64), intent(in) :: a
    type(complex_double_st), intent(in) :: b
    type(complex_double_st) div_complex_double_complex_double_st
    ! Local declarations
    type(double_st) norme2

    norme2 = b%x**2 + b%y**2
    div_complex_double_complex_double_st = a*conjg(b)/norme2
  end function div_complex_double_complex_double_st

  elemental function div_complex_double_complex_single_st(a, b)
    complex(REAL64), intent(in) :: a
    type(complex_single_st), intent(in) :: b
    type(complex_single_st) div_complex_double_complex_single_st
    ! Local declarations
    type(single_st) norme2

    norme2 = b%x**2 + b%y**2
    div_complex_double_complex_single_st = a*conjg(b)/norme2
  end function div_complex_double_complex_single_st


  elemental function div_double_complex_double_st(a, b)
    real(C_DOUBLE), intent(in) :: a
    type(complex_double_st), intent(in) :: b
    type(complex_double_st) div_double_complex_double_st
    ! Local declarations
    type(double_st) norme2

    norme2 = b%x**2 + b%y**2
    div_double_complex_double_st = a*conjg(b)/norme2
  end function div_double_complex_double_st

  elemental function div_double_complex_single_st(a, b)
    real(C_DOUBLE), intent(in) :: a
    type(complex_single_st), intent(in) :: b
    type(complex_single_st) div_double_complex_single_st
    ! Local declarations
    type(single_st) norme2

    norme2 = b%x**2 + b%y**2
    div_double_complex_single_st = a*conjg(b)/norme2
  end function div_double_complex_single_st

!
  elemental function div_float_complex_double_st(a, b)
    real(C_FLOAT), intent(in) :: a
    type(complex_double_st), intent(in) :: b
    type(complex_double_st) div_float_complex_double_st
    ! Local declarations
    type(double_st) norme2

    norme2 = b%x**2 + b%y**2
    div_float_complex_double_st = a*conjg(b)/norme2
  end function div_float_complex_double_st

  elemental function div_float_complex_single_st(a, b)
    real(C_FLOAT), intent(in) :: a
    type(complex_single_st), intent(in) :: b
    type(complex_single_st) div_float_complex_single_st
    ! Local declarations
    type(single_st) norme2

    norme2 = b%x**2 + b%y**2
    div_float_complex_single_st = a*conjg(b)/norme2
  end function div_float_complex_single_st

!
  elemental function div_long_long_complex_double_st(a, b)
    integer(C_LONG_LONG), intent(in) :: a
    type(complex_double_st), intent(in) :: b
    type(complex_double_st) div_long_long_complex_double_st
    ! Local declarations
    type(double_st) norme2

    norme2 = b%x**2 + b%y**2
    div_long_long_complex_double_st = a*conjg(b)/norme2
  end function div_long_long_complex_double_st

  elemental function div_long_long_complex_single_st(a, b)
    integer(C_LONG_LONG), intent(in) :: a
    type(complex_single_st), intent(in) :: b
    type(complex_single_st) div_long_long_complex_single_st
    ! Local declarations
    type(single_st) norme2

    norme2 = b%x**2 + b%y**2
    div_long_long_complex_single_st = a*conjg(b)/norme2
  end function div_long_long_complex_single_st

!
!CONTAINS_OP(div, long,              complex_double_st, complex_double_st)
!CONTAINS_OP(div, long,              complex_single_st, complex_single_st)
!
  elemental function div_int_complex_double_st(a, b)
    integer(C_INT), intent(in) :: a
    type(complex_double_st), intent(in) :: b
    type(complex_double_st) div_int_complex_double_st
    ! Local declarations
    type(double_st) norme2

    norme2 = b%x**2 + b%y**2
    div_int_complex_double_st = a*conjg(b)/norme2
  end function div_int_complex_double_st

  elemental function div_int_complex_single_st(a, b)
    integer(C_INT), intent(in) :: a
    type(complex_single_st), intent(in) :: b
    type(complex_single_st) div_int_complex_single_st
    ! Local declarations
    type(single_st) norme2

    norme2 = b%x**2 + b%y**2
    div_int_complex_single_st = a*conjg(b)/norme2
  end function div_int_complex_single_st

!
  elemental function div_short_complex_double_st(a, b)
    integer(C_SHORT), intent(in) :: a
    type(complex_double_st), intent(in) :: b
    type(complex_double_st) div_short_complex_double_st
    ! Local declarations
    type(double_st) norme2

    norme2 = b%x**2 + b%y**2
    div_short_complex_double_st = a*conjg(b)/norme2
  end function div_short_complex_double_st

  elemental function div_short_complex_single_st(a, b)
    integer(C_SHORT), intent(in) :: a
    type(complex_single_st), intent(in) :: b
    type(complex_single_st) div_short_complex_single_st
    ! Local declarations
    type(single_st) norme2

    norme2 = b%x**2 + b%y**2
    div_short_complex_single_st = a*conjg(b)/norme2
  end function div_short_complex_single_st



end module cadna_cmplx_div

