
! Module de conversion




module cadna_convert
  use ISO_C_BINDING, only : C_INT
  use cadna_types
  use cadna_affect

  implicit none




  interface single_st
    module procedure single_st_double_st
  end interface

  interface single_st
    module procedure single_st_single_st
  end interface

  interface double_st
    module procedure double_st_single_st
  end interface

  interface double_st
    module procedure double_st_double_st
  end interface




  interface int
    module procedure int_single_st
  end interface
  interface
    pure function cpp_int_single_st(a) bind(C)
      import single_st
      import C_INT
      type(single_st), intent(in) :: a
      integer(C_INT) cpp_int_single_st
    end function cpp_int_single_st
  end interface

  interface int
    module procedure int_double_st
  end interface
  interface
    pure function cpp_int_double_st(a) bind(C)
      import double_st
      import C_INT
      type(double_st), intent(in) :: a
      integer(C_INT) cpp_int_double_st
    end function cpp_int_double_st
  end interface




  interface real
    module procedure single_st_double_st
  end interface

  interface real
    module procedure single_st_single_st
  end interface

  interface dble
    module procedure double_st_single_st
  end interface

  interface dble
    module procedure double_st_double_st
  end interface


contains



  elemental function int_single_st(a)
    type(single_st), intent(in) :: a
    integer(C_INT) int_single_st

    int_single_st = cpp_int_single_st(a)
  end function int_single_st

  elemental function int_double_st(a)
    type(double_st), intent(in) :: a
    integer(C_INT) int_double_st

    int_double_st = cpp_int_double_st(a)
  end function int_double_st




  elemental function single_st_double_st(a)
    type(double_st), intent(in) :: a
    type(single_st) single_st_double_st

    single_st_double_st = a
  end function single_st_double_st

  elemental function single_st_single_st(a)
    type(single_st), intent(in) :: a
    type(single_st) single_st_single_st

    single_st_single_st = a
  end function single_st_single_st

  elemental function double_st_single_st(a)
    type(single_st), intent(in) :: a
    type(double_st) double_st_single_st

    double_st_single_st = a
  end function double_st_single_st

  elemental function double_st_double_st(a)
    type(double_st), intent(in) :: a
    type(double_st) double_st_double_st

    double_st_double_st = a
  end function double_st_double_st




end module cadna_convert

