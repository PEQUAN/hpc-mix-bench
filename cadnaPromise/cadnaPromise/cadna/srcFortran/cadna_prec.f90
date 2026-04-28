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


! Module de surcharge des fonctions de précision




module cadna_prec
  use cadna_types
  implicit none




  interface precision
   module procedure precision_single_st
  end interface precision

  interface precision
   module procedure precision_double_st
  end interface precision

  interface range
   module procedure range_single_st
  end interface range

  interface range
   module procedure range_double_st
  end interface range

  interface epsilon
   module procedure epsilon_single_st
  end interface epsilon

  interface epsilon
   module procedure epsilon_double_st
  end interface epsilon

  interface spacing
   module procedure spacing_single_st
  end interface spacing

  interface spacing
   module procedure spacing_double_st
  end interface spacing

  interface nearest
   module procedure nearest_single_st
  end interface nearest

  interface nearest
   module procedure nearest_double_st
  end interface nearest




  interface precision
   module procedure precision_single_st_1D
  end interface precision

  interface precision
   module procedure precision_double_st_1D
  end interface precision

  interface precision
   module procedure precision_single_st_2D
  end interface precision

  interface precision
   module procedure precision_double_st_2D
  end interface precision

  interface precision
   module procedure precision_single_st_3D
  end interface precision

  interface precision
   module procedure precision_double_st_3D
  end interface precision

  interface precision
   module procedure precision_single_st_4D
  end interface precision

  interface precision
   module procedure precision_double_st_4D
  end interface precision

  interface precision
   module procedure precision_single_st_5D
  end interface precision

  interface precision
   module procedure precision_double_st_5D
  end interface precision

  interface precision
   module procedure precision_single_st_6D
  end interface precision

  interface precision
   module procedure precision_double_st_6D
  end interface precision

  interface precision
   module procedure precision_single_st_7D
  end interface precision

  interface precision
   module procedure precision_double_st_7D
  end interface precision

  interface range
   module procedure range_single_st_1D
  end interface range

  interface range
   module procedure range_double_st_1D
  end interface range

  interface range
   module procedure range_single_st_2D
  end interface range

  interface range
   module procedure range_double_st_2D
  end interface range

  interface range
   module procedure range_single_st_3D
  end interface range

  interface range
   module procedure range_double_st_3D
  end interface range

  interface range
   module procedure range_single_st_4D
  end interface range

  interface range
   module procedure range_double_st_4D
  end interface range

  interface range
   module procedure range_single_st_5D
  end interface range

  interface range
   module procedure range_double_st_5D
  end interface range

  interface range
   module procedure range_single_st_6D
  end interface range

  interface range
   module procedure range_double_st_6D
  end interface range

  interface range
   module procedure range_single_st_7D
  end interface range

  interface range
   module procedure range_double_st_7D
  end interface range


contains



  function precision_single_st(x )
     
    type(single_st), intent(in) :: x
     
    integer precision_single_st

    precision_single_st = precision((x%x+x%y+x%z)/3.)
  end function precision_single_st

  function precision_double_st(x )
     
    type(double_st), intent(in) :: x
     
    integer precision_double_st

    precision_double_st = precision((x%x+x%y+x%z)/3.)
  end function precision_double_st

  function range_single_st(x )
     
    type(single_st), intent(in) :: x
     
    integer range_single_st

    range_single_st = range((x%x+x%y+x%z)/3.)
  end function range_single_st

  function range_double_st(x )
     
    type(double_st), intent(in) :: x
     
    integer range_double_st

    range_double_st = range((x%x+x%y+x%z)/3.)
  end function range_double_st

  elemental function epsilon_single_st(x )
    use ISO_FORTRAN_ENV, only : REAL32
    type(single_st), intent(in) :: x
     
    real(kind=REAL32) epsilon_single_st

    epsilon_single_st = epsilon((x%x+x%y+x%z)/3.)
  end function epsilon_single_st

  elemental function epsilon_double_st(x )
    use ISO_FORTRAN_ENV, only : REAL64
    type(double_st), intent(in) :: x
     
    real(kind=REAL64) epsilon_double_st

    epsilon_double_st = epsilon((x%x+x%y+x%z)/3.)
  end function epsilon_double_st

  elemental function spacing_single_st(x )
    use ISO_FORTRAN_ENV, only : REAL32
    type(single_st), intent(in) :: x
     
    real(kind=REAL32) spacing_single_st

    spacing_single_st = spacing((x%x+x%y+x%z)/3.)
  end function spacing_single_st

  elemental function spacing_double_st(x )
    use ISO_FORTRAN_ENV, only : REAL64
    type(double_st), intent(in) :: x
     
    real(kind=REAL64) spacing_double_st

    spacing_double_st = spacing((x%x+x%y+x%z)/3.)
  end function spacing_double_st

  elemental function nearest_single_st(x ,s)
    use ISO_FORTRAN_ENV, only : REAL32
    type(single_st), intent(in) :: x
    real,            intent(in) :: s
    real(kind=REAL32) nearest_single_st

    nearest_single_st = nearest((x%x+x%y+x%z)/3.,s)
  end function nearest_single_st

  elemental function nearest_double_st(x ,s)
    use ISO_FORTRAN_ENV, only : REAL64
    type(double_st), intent(in) :: x
    real,            intent(in) :: s
    real(kind=REAL64) nearest_double_st

    nearest_double_st = nearest((x%x+x%y+x%z)/3.,s)
  end function nearest_double_st




  function precision_single_st_1D(x)
    type(single_st), dimension (:), intent(in), target, contiguous :: x
    type(single_st), dimension(:), pointer :: p_array
    integer precision_single_st_1D

    p_array(1:size(x)) => x
    precision_single_st_1D = precision((p_array(1)%x))
  end function precision_single_st_1D

  function precision_double_st_1D(x)
    type(double_st), dimension (:), intent(in), target, contiguous :: x
    type(double_st), dimension(:), pointer :: p_array
    integer precision_double_st_1D

    p_array(1:size(x)) => x
    precision_double_st_1D = precision((p_array(1)%x))
  end function precision_double_st_1D

  function precision_single_st_2D(x)
    type(single_st), dimension (:,:), intent(in), target, contiguous :: x
    type(single_st), dimension(:), pointer :: p_array
    integer precision_single_st_2D

    p_array(1:size(x)) => x
    precision_single_st_2D = precision((p_array(1)%x))
  end function precision_single_st_2D

  function precision_double_st_2D(x)
    type(double_st), dimension (:,:), intent(in), target, contiguous :: x
    type(double_st), dimension(:), pointer :: p_array
    integer precision_double_st_2D

    p_array(1:size(x)) => x
    precision_double_st_2D = precision((p_array(1)%x))
  end function precision_double_st_2D

  function precision_single_st_3D(x)
    type(single_st), dimension (:,:,:), intent(in), target, contiguous :: x
    type(single_st), dimension(:), pointer :: p_array
    integer precision_single_st_3D

    p_array(1:size(x)) => x
    precision_single_st_3D = precision((p_array(1)%x))
  end function precision_single_st_3D

  function precision_double_st_3D(x)
    type(double_st), dimension (:,:,:), intent(in), target, contiguous :: x
    type(double_st), dimension(:), pointer :: p_array
    integer precision_double_st_3D

    p_array(1:size(x)) => x
    precision_double_st_3D = precision((p_array(1)%x))
  end function precision_double_st_3D

  function precision_single_st_4D(x)
    type(single_st), dimension (:,:,:,:), intent(in), target, contiguous :: x
    type(single_st), dimension(:), pointer :: p_array
    integer precision_single_st_4D

    p_array(1:size(x)) => x
    precision_single_st_4D = precision((p_array(1)%x))
  end function precision_single_st_4D

  function precision_double_st_4D(x)
    type(double_st), dimension (:,:,:,:), intent(in), target, contiguous :: x
    type(double_st), dimension(:), pointer :: p_array
    integer precision_double_st_4D

    p_array(1:size(x)) => x
    precision_double_st_4D = precision((p_array(1)%x))
  end function precision_double_st_4D

  function precision_single_st_5D(x)
    type(single_st), dimension (:,:,:,:,:), intent(in), target, contiguous :: x
    type(single_st), dimension(:), pointer :: p_array
    integer precision_single_st_5D

    p_array(1:size(x)) => x
    precision_single_st_5D = precision((p_array(1)%x))
  end function precision_single_st_5D

  function precision_double_st_5D(x)
    type(double_st), dimension (:,:,:,:,:), intent(in), target, contiguous :: x
    type(double_st), dimension(:), pointer :: p_array
    integer precision_double_st_5D

    p_array(1:size(x)) => x
    precision_double_st_5D = precision((p_array(1)%x))
  end function precision_double_st_5D

  function precision_single_st_6D(x)
    type(single_st), dimension (:,:,:,:,:,:), intent(in), target, contiguous :: x
    type(single_st), dimension(:), pointer :: p_array
    integer precision_single_st_6D

    p_array(1:size(x)) => x
    precision_single_st_6D = precision((p_array(1)%x))
  end function precision_single_st_6D

  function precision_double_st_6D(x)
    type(double_st), dimension (:,:,:,:,:,:), intent(in), target, contiguous :: x
    type(double_st), dimension(:), pointer :: p_array
    integer precision_double_st_6D

    p_array(1:size(x)) => x
    precision_double_st_6D = precision((p_array(1)%x))
  end function precision_double_st_6D

  function precision_single_st_7D(x)
    type(single_st), dimension (:,:,:,:,:,:,:), intent(in), target, contiguous :: x
    type(single_st), dimension(:), pointer :: p_array
    integer precision_single_st_7D

    p_array(1:size(x)) => x
    precision_single_st_7D = precision((p_array(1)%x))
  end function precision_single_st_7D

  function precision_double_st_7D(x)
    type(double_st), dimension (:,:,:,:,:,:,:), intent(in), target, contiguous :: x
    type(double_st), dimension(:), pointer :: p_array
    integer precision_double_st_7D

    p_array(1:size(x)) => x
    precision_double_st_7D = precision((p_array(1)%x))
  end function precision_double_st_7D

  function range_single_st_1D(x)
    type(single_st), dimension (:), intent(in), target, contiguous :: x
    type(single_st), dimension(:), pointer :: p_array
    integer range_single_st_1D

    p_array(1:size(x)) => x
    range_single_st_1D = range((p_array(1)%x))
  end function range_single_st_1D

  function range_double_st_1D(x)
    type(double_st), dimension (:), intent(in), target, contiguous :: x
    type(double_st), dimension(:), pointer :: p_array
    integer range_double_st_1D

    p_array(1:size(x)) => x
    range_double_st_1D = range((p_array(1)%x))
  end function range_double_st_1D

  function range_single_st_2D(x)
    type(single_st), dimension (:,:), intent(in), target, contiguous :: x
    type(single_st), dimension(:), pointer :: p_array
    integer range_single_st_2D

    p_array(1:size(x)) => x
    range_single_st_2D = range((p_array(1)%x))
  end function range_single_st_2D

  function range_double_st_2D(x)
    type(double_st), dimension (:,:), intent(in), target, contiguous :: x
    type(double_st), dimension(:), pointer :: p_array
    integer range_double_st_2D

    p_array(1:size(x)) => x
    range_double_st_2D = range((p_array(1)%x))
  end function range_double_st_2D

  function range_single_st_3D(x)
    type(single_st), dimension (:,:,:), intent(in), target, contiguous :: x
    type(single_st), dimension(:), pointer :: p_array
    integer range_single_st_3D

    p_array(1:size(x)) => x
    range_single_st_3D = range((p_array(1)%x))
  end function range_single_st_3D

  function range_double_st_3D(x)
    type(double_st), dimension (:,:,:), intent(in), target, contiguous :: x
    type(double_st), dimension(:), pointer :: p_array
    integer range_double_st_3D

    p_array(1:size(x)) => x
    range_double_st_3D = range((p_array(1)%x))
  end function range_double_st_3D

  function range_single_st_4D(x)
    type(single_st), dimension (:,:,:,:), intent(in), target, contiguous :: x
    type(single_st), dimension(:), pointer :: p_array
    integer range_single_st_4D

    p_array(1:size(x)) => x
    range_single_st_4D = range((p_array(1)%x))
  end function range_single_st_4D

  function range_double_st_4D(x)
    type(double_st), dimension (:,:,:,:), intent(in), target, contiguous :: x
    type(double_st), dimension(:), pointer :: p_array
    integer range_double_st_4D

    p_array(1:size(x)) => x
    range_double_st_4D = range((p_array(1)%x))
  end function range_double_st_4D

  function range_single_st_5D(x)
    type(single_st), dimension (:,:,:,:,:), intent(in), target, contiguous :: x
    type(single_st), dimension(:), pointer :: p_array
    integer range_single_st_5D

    p_array(1:size(x)) => x
    range_single_st_5D = range((p_array(1)%x))
  end function range_single_st_5D

  function range_double_st_5D(x)
    type(double_st), dimension (:,:,:,:,:), intent(in), target, contiguous :: x
    type(double_st), dimension(:), pointer :: p_array
    integer range_double_st_5D

    p_array(1:size(x)) => x
    range_double_st_5D = range((p_array(1)%x))
  end function range_double_st_5D

  function range_single_st_6D(x)
    type(single_st), dimension (:,:,:,:,:,:), intent(in), target, contiguous :: x
    type(single_st), dimension(:), pointer :: p_array
    integer range_single_st_6D

    p_array(1:size(x)) => x
    range_single_st_6D = range((p_array(1)%x))
  end function range_single_st_6D

  function range_double_st_6D(x)
    type(double_st), dimension (:,:,:,:,:,:), intent(in), target, contiguous :: x
    type(double_st), dimension(:), pointer :: p_array
    integer range_double_st_6D

    p_array(1:size(x)) => x
    range_double_st_6D = range((p_array(1)%x))
  end function range_double_st_6D

  function range_single_st_7D(x)
    type(single_st), dimension (:,:,:,:,:,:,:), intent(in), target, contiguous :: x
    type(single_st), dimension(:), pointer :: p_array
    integer range_single_st_7D

    p_array(1:size(x)) => x
    range_single_st_7D = range((p_array(1)%x))
  end function range_single_st_7D

  function range_double_st_7D(x)
    type(double_st), dimension (:,:,:,:,:,:,:), intent(in), target, contiguous :: x
    type(double_st), dimension(:), pointer :: p_array
    integer range_double_st_7D

    p_array(1:size(x)) => x
    range_double_st_7D = range((p_array(1)%x))
  end function range_double_st_7D




end module cadna_prec

