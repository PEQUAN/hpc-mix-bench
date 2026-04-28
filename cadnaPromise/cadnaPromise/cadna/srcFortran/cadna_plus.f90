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



! Module de surcharge de l'opérateur unaire "+"




module cadna_plus
  use cadna_types
  implicit none




  interface operator(+)
    module procedure plus_single_st
  end interface operator(+)
  interface
    pure function cpp_plus_single_st(a) bind(C)
      import single_st
      type(single_st), intent(in) :: a
      type(single_st) cpp_plus_single_st
    end function cpp_plus_single_st
  end interface

  interface operator(+)
    module procedure plus_double_st
  end interface operator(+)
  interface
    pure function cpp_plus_double_st(a) bind(C)
      import double_st
      type(double_st), intent(in) :: a
      type(double_st) cpp_plus_double_st
    end function cpp_plus_double_st
  end interface


contains



  elemental function plus_single_st(a)
    type(single_st), intent(in) :: a
    type(single_st) plus_single_st

    plus_single_st = cpp_plus_single_st(a)
  end function plus_single_st

  elemental function plus_double_st(a)
    type(double_st), intent(in) :: a
    type(double_st) plus_double_st

    plus_double_st = cpp_plus_double_st(a)
  end function plus_double_st




end module cadna_plus

