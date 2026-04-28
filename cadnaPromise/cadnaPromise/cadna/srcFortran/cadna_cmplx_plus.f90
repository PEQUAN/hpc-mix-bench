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




module cadna_cmplx_plus
  use cadna_types
  use cadna_plus
  implicit none




  interface operator(+)
    module procedure plus_complex_single_st
  end interface operator(+)

  interface operator(+)
    module procedure plus_complex_double_st
  end interface operator(+)


contains



  elemental function plus_complex_single_st(a)
    type(complex_single_st), intent(in) :: a
    type(complex_single_st) plus_complex_single_st

    plus_complex_single_st%x = +a%x
    plus_complex_single_st%y = +a%y
  end function plus_complex_single_st

  elemental function plus_complex_double_st(a)
    type(complex_double_st), intent(in) :: a
    type(complex_double_st) plus_complex_double_st

    plus_complex_double_st%x = +a%x
    plus_complex_double_st%y = +a%y
  end function plus_complex_double_st




end module cadna_cmplx_plus

