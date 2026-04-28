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


module cadna_utils
  use ISO_C_BINDING, only : C_INT, C_DOUBLE, C_CHAR, C_NULL_CHAR
  use cadna_types

  implicit none

  interface nb_significant_digit
    function nb_significant_digit_single_st(a) bind(C, name="cpp_nb_significant_digit_single_st")
      import single_st, C_INT
      type(single_st), intent(in) :: a
      integer(kind=C_INT) :: nb_significant_digit_single_st
    end function nb_significant_digit_single_st
    function nb_significant_digit_double_st(a) bind(C, name="cpp_nb_significant_digit_double_st")
      import double_st, C_INT
      type(double_st), intent(in) :: a
      integer(kind=C_INT) :: nb_significant_digit_double_st
    end function nb_significant_digit_double_st
  end interface nb_significant_digit
  !
  !=== DATA_ST ===
  interface data_st
    module procedure f_data_st_single_st, f_data_st_single_st_double_int, &
                     f_data_st_double_st, f_data_st_double_st_double_int
  end interface data_st
  !
  interface
    pure subroutine data_st_single_st(a) bind(C,name="cpp_data_st_single_st")
      import single_st
      type(single_st), intent(inout) :: a
    end subroutine data_st_single_st
    pure subroutine data_st_single_st_double_int(a, e, error_type) bind(C,name="cpp_data_st_single_st_double_int")
      import single_st, C_INT, C_DOUBLE
      type(single_st), intent(inout) :: a
      real(C_DOUBLE),  intent(in)    :: e
      integer(C_INT),  intent(in)    :: error_type
    end subroutine data_st_single_st_double_int
    pure subroutine data_st_double_st(a) bind(C,name="cpp_data_st_double_st")
      import double_st
      type(double_st), intent(inout) :: a
    end subroutine data_st_double_st
    pure subroutine data_st_double_st_double_int(a, e, error_type) bind(C,name="cpp_data_st_double_st_double_int")
      import double_st, C_INT, C_DOUBLE
      type(double_st), intent(inout) :: a
      real(C_DOUBLE),  intent(in)    :: e
      integer(C_INT),  intent(in)    :: error_type
    end subroutine data_st_double_st_double_int
  end interface
  !=== END DATA_ST ===

  !=== DISPLAY ===
  interface display
    module procedure f_display_single_st, f_display_char_single_st, &
                     f_display_double_st, f_display_char_double_st
  end interface display
  !
  interface
    pure subroutine display_single_st(a) bind(C, name="cpp_display_single_st")
      import single_st
      type(single_st), intent(in) :: a
    end subroutine display_single_st
    pure subroutine display_char_single_st(s, a) bind(C, name="cpp_display_char_single_st")
      import single_st, C_CHAR
      character(kind=C_CHAR), dimension(*), intent(in) :: s
      type(single_st), intent(in) :: a
    end subroutine display_char_single_st
    !
    pure subroutine display_double_st(a) bind(C, name="cpp_display_double_st")
      import double_st
      type(double_st), intent(in) :: a
    end subroutine display_double_st
    pure subroutine display_char_double_st(s, a) bind(C, name="cpp_display_char_double_st")
      import double_st, C_CHAR
      character(kind=C_CHAR), dimension(*), intent(in) :: s
      type(double_st), intent(in) :: a
    end subroutine display_char_double_st
  end interface
  !=== END DISPLAY ===
  !
  !=== STR ===
  interface str
    module procedure f_str_single_st, f_str_double_st
  end interface str
  !
  interface
    pure subroutine str_single_st(a,str) bind(C, name="cpp_str_single_st")
      import single_st, C_CHAR
      type(single_st), intent(in) :: a
      character(kind=C_CHAR), dimension(*), intent(out) :: str
    end subroutine str_single_st
    pure subroutine str_double_st(a,str) bind(C, name="cpp_str_double_st")
      import double_st, C_CHAR
      type(double_st), intent(in) :: a
      character(kind=C_CHAR), dimension(*), intent(out) :: str
    end subroutine str_double_st
  end interface
  !=== END STR ===

contains

  !=== DISPLAY ===
  elemental subroutine f_display_single_st(a)
    type(single_st), intent(in) :: a

    call display_single_st(a)
  end subroutine f_display_single_st

  elemental subroutine f_display_char_single_st(s, a)
    character(kind=C_CHAR, len=*), intent(in) :: s
    type(single_st), intent(in) :: a

    call display_char_single_st(s, a)
  end subroutine f_display_char_single_st
  !
  elemental subroutine f_display_double_st(a)
    type(double_st), intent(in) :: a

    call display_double_st(a)
  end subroutine f_display_double_st

  elemental subroutine f_display_char_double_st(s, a)
    character(kind=C_CHAR, len=*), intent(in) :: s
    type(double_st), intent(in) :: a

    call display_char_double_st(s, a)
  end subroutine f_display_char_double_st
  !=== END DISPLAY ===

  !=== STR ===
  elemental function f_str_single_st(a)
    type(single_st), intent(in) :: a
    !character(len=:), allocatable :: f_str_single_st
    character(kind=C_CHAR,len=64) :: f_str_single_st
    integer i

    call str_single_st(a, f_str_single_st)
    i = index(f_str_single_st, C_NULL_CHAR )
    f_str_single_st(i:) = ' '
  end function f_str_single_st
  elemental function f_str_double_st(a)
    type(double_st), intent(in) :: a
    !character(len=:), allocatable :: f_str_double_st
    character(kind=C_CHAR,len=64) :: f_str_double_st
    integer i

    call str_double_st(a, f_str_double_st)
    i = index(f_str_double_st, C_NULL_CHAR )
    f_str_double_st(i:) = ' '
  end function f_str_double_st
  !=== END STR ===
  
  !=== DATA_ST ===
  elemental subroutine f_data_st_single_st(a)
    type(single_st), intent(inout) :: a

    call data_st_single_st(a)
  end  subroutine f_data_st_single_st

  elemental subroutine f_data_st_single_st_double_int(a, e, error_type)
    type(single_st), intent(inout) :: a
    real(C_DOUBLE),  intent(in)    :: e
    integer(C_INT),  intent(in)    :: error_type

    call data_st_single_st_double_int(a, e, error_type)
  end  subroutine f_data_st_single_st_double_int
  elemental subroutine f_data_st_double_st(a)
    type(double_st), intent(inout) :: a

    call data_st_double_st(a)
  end  subroutine f_data_st_double_st

  elemental subroutine f_data_st_double_st_double_int(a, e, error_type)
    type(double_st), intent(inout) :: a
    real(C_DOUBLE),  intent(in)    :: e
    integer(C_INT),  intent(in)    :: error_type

    call data_st_double_st_double_int(a, e, error_type)
  end  subroutine f_data_st_double_st_double_int
  !=== END DATA_ST ===

end module cadna_utils
