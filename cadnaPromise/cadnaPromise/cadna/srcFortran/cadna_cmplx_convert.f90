
! Module de conversion




module cadna_cmplx_convert
  use ISO_C_BINDING, only : C_INT
  use cadna_types
  use cadna_affect
  use cadna_cmplx_affect
  use cadna_convert
  use cadna_opp

  implicit none




  interface single_st
    module procedure single_st_complex_single_st
  end interface

  interface double_st
    module procedure double_st_complex_single_st
  end interface

  interface single_st
    module procedure single_st_complex_double_st
  end interface

  interface double_st
    module procedure double_st_complex_double_st
  end interface

  interface complex_single_st
    module procedure complex_single_st_complex_single_st
  end interface

  interface complex_single_st
    module procedure complex_single_st_complex_double_st
  end interface

  interface complex_single_st
    module procedure complex_single_st_single_st
  end interface

  interface complex_single_st
    module procedure complex_single_st_double_st
  end interface

  interface complex_double_st
    module procedure complex_double_st_complex_single_st
  end interface

  interface complex_double_st
    module procedure complex_double_st_complex_double_st
  end interface

  interface complex_double_st
    module procedure complex_double_st_single_st
  end interface

  interface complex_double_st
    module procedure complex_double_st_double_st
  end interface




  interface cmplx
    module procedure cmplx_complex_single_st_single_st
  end interface

  interface cmplx
    module procedure cmplx_complex_double_st_double_st
  end interface




  interface int
    module procedure int_complex_single_st
  end interface

  interface int
    module procedure int_complex_double_st
  end interface

  interface real
    module procedure single_st_complex_single_st
  end interface

  interface real
    module procedure double_st_complex_double_st
  end interface

  interface dble
    module procedure double_st_complex_single_st
  end interface

  interface dble
    module procedure double_st_complex_double_st
  end interface

  interface aimag
    module procedure aimag_complex_single_st
  end interface

  interface aimag
    module procedure aimag_complex_double_st
  end interface

  interface conjg
    module procedure conjg_complex_single_st
  end interface

  interface conjg
    module procedure conjg_complex_double_st
  end interface


contains



! REAL(complex_single_st)
  elemental function single_st_complex_single_st(z)
    type(complex_single_st), intent(in) :: z
    type(single_st) single_st_complex_single_st

    single_st_complex_single_st = z
  end function single_st_complex_single_st

! REAL(complex_single_st, KIND=8), DBLE(complex_single_st)
  elemental function double_st_complex_single_st(z)
    type(complex_single_st), intent(in) :: z
    type(double_st) double_st_complex_single_st

    double_st_complex_single_st = z
  end function double_st_complex_single_st

! REAL(complex_double_st, KIND=4)
  elemental function single_st_complex_double_st(z)
    type(complex_double_st), intent(in) :: z
    type(single_st) single_st_complex_double_st

    single_st_complex_double_st = z
  end function single_st_complex_double_st

! REAL(complex_double_st), DBLE(complex_double_st)
  elemental function double_st_complex_double_st(z)
    type(complex_double_st), intent(in) :: z
    type(double_st) double_st_complex_double_st

    double_st_complex_double_st = z
  end function double_st_complex_double_st

! CMPLX(complex_single_st, KIND=4)
  elemental function complex_single_st_complex_single_st(z)
    type(complex_single_st), intent(in) :: z
    type(complex_single_st) complex_single_st_complex_single_st

    complex_single_st_complex_single_st = z
  end function complex_single_st_complex_single_st

! CMPLX(complex_double_st, KIND=8)
  elemental function complex_double_st_complex_double_st(z)
    type(complex_double_st), intent(in) :: z
    type(complex_double_st) complex_double_st_complex_double_st

    complex_double_st_complex_double_st = z
  end function complex_double_st_complex_double_st

! CMPLX(complex_double_st, KIND=4)
  elemental function complex_single_st_complex_double_st(z)
    type(complex_double_st), intent(in) :: z
    type(complex_single_st) complex_single_st_complex_double_st

    complex_single_st_complex_double_st = z
  end function complex_single_st_complex_double_st

! CMPLX(complex_single_st, KIND=8)
  elemental function complex_double_st_complex_single_st(z)
    type(complex_single_st), intent(in) :: z
    type(complex_double_st) complex_double_st_complex_single_st

    complex_double_st_complex_single_st = z
  end function complex_double_st_complex_single_st





! surcharge des constructeurs par défaut pour permettre des arguments optionnels.
! CMPLX(single_st[, single_st], KIND=4)
  elemental function complex_single_st_single_st(x, y)
    type(single_st),           intent(in) :: x
    type(single_st), optional, intent(in) :: y
    type(complex_single_st) complex_single_st_single_st
    type(single_st) y_local

    y_local = MERGE(tsource=y, fsource=single_st(), mask=present(y))
    complex_single_st_single_st%x = x; complex_single_st_single_st%y = y_local
  end function complex_single_st_single_st

! CMPLX(double_st[, double_st], KIND=4)
  elemental function complex_single_st_double_st(x, y)
    type(double_st),           intent(in) :: x
    type(double_st), optional, intent(in) :: y
    type(complex_single_st) complex_single_st_double_st
    type(double_st) y_local

    y_local = MERGE(tsource=y, fsource=double_st(), mask=present(y))
    complex_single_st_double_st%x = x; complex_single_st_double_st%y = y_local
  end function complex_single_st_double_st

! CMPLX(single_st[, single_st], KIND=8)
  elemental function complex_double_st_single_st(x, y)
    type(single_st),           intent(in) :: x
    type(single_st), optional, intent(in) :: y
    type(complex_double_st) complex_double_st_single_st
    type(single_st) y_local

    y_local = MERGE(tsource=y, fsource=single_st(), mask=present(y))
    complex_double_st_single_st%x = x; complex_double_st_single_st%y = y_local
  end function complex_double_st_single_st

! CMPLX(double_st[, double_st], KIND=8)
  elemental function complex_double_st_double_st(x, y)
    type(double_st),           intent(in) :: x
    type(double_st), optional, intent(in) :: y
    type(complex_double_st) complex_double_st_double_st
    type(double_st) y_local

    y_local = MERGE(tsource=y, fsource=double_st(), mask=present(y))
    complex_double_st_double_st%x = x; complex_double_st_double_st%y = y_local
  end function complex_double_st_double_st




! CMPLX(single_st[, single_st])
  elemental function cmplx_complex_single_st_single_st(x, y)
    type(single_st),           intent(in) :: x
    type(single_st), optional, intent(in) :: y
    type(complex_single_st) cmplx_complex_single_st_single_st
    type(single_st) y_local

    y_local = MERGE(tsource=y, fsource=single_st(), mask=present(y))
    cmplx_complex_single_st_single_st%x = x
    cmplx_complex_single_st_single_st%y = y_local
  end function cmplx_complex_single_st_single_st

! CMPLX(double_st[, double_st])
  elemental function cmplx_complex_double_st_double_st(x, y)
    type(double_st),           intent(in) :: x
    type(double_st), optional, intent(in) :: y
    type(complex_double_st) cmplx_complex_double_st_double_st
    type(double_st) y_local

    y_local = MERGE(tsource=y, fsource=double_st(), mask=present(y))
    cmplx_complex_double_st_double_st%x = x
    cmplx_complex_double_st_double_st%y = y_local
  end function cmplx_complex_double_st_double_st




  elemental function int_complex_single_st(a)
    type(complex_single_st), intent(in) :: a
    integer(C_INT) int_complex_single_st

    int_complex_single_st = int(a%x) 
  end function int_complex_single_st

  elemental function int_complex_double_st(a)
    type(complex_double_st), intent(in) :: a
    integer(C_INT) int_complex_double_st

    int_complex_double_st = int(a%x) 
  end function int_complex_double_st

  elemental function aimag_complex_single_st(z)
    type(complex_single_st), intent(in) :: z
    type(single_st) aimag_complex_single_st

    aimag_complex_single_st = z%y
  end function aimag_complex_single_st

  elemental function aimag_complex_double_st(z)
    type(complex_double_st), intent(in) :: z
    type(double_st) aimag_complex_double_st

    aimag_complex_double_st = z%y
  end function aimag_complex_double_st

  elemental function conjg_complex_single_st(z)
    type(complex_single_st), intent(in) :: z
    type(complex_single_st) conjg_complex_single_st

    conjg_complex_single_st%x = z%x ; conjg_complex_single_st%y = -z%y
  end function conjg_complex_single_st

  elemental function conjg_complex_double_st(z)
    type(complex_double_st), intent(in) :: z
    type(complex_double_st) conjg_complex_double_st

    conjg_complex_double_st%x = z%x ; conjg_complex_double_st%y = -z%y
  end function conjg_complex_double_st




end module cadna_cmplx_convert

