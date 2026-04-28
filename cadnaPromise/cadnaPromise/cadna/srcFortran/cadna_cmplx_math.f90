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



! Module de surcharge des fonctions mathématiques




module cadna_cmplx_math
  use cadna_types
  use cadna_affect
  use cadna_add
  use cadna_cmplx_add
  use cadna_sub
  use cadna_cmplx_sub
  use cadna_mult
  use cadna_cmplx_mult
! `use cadna_cmplx_pow'
  use cadna_div
  use cadna_cmplx_div
  use cadna_opp
  use cadna_math
  use cadna_lt
  use cadna_gt

  implicit none


!define(INTERFACE_MATH_UNAIRE_DOUBLE,
!`  interface $1'
!    `module procedure $1_double_st'
!  `end interface'
!)
!
!INTERFACE_MATH_UNAIRE_DOUBLE(dabs)
!INTERFACE_MATH_UNAIRE_DOUBLE(dsqrt)
!INTERFACE_MATH_UNAIRE_DOUBLE(dsin)
!INTERFACE_MATH_UNAIRE_DOUBLE(dcos)
!INTERFACE_MATH_UNAIRE_DOUBLE(dtan)
!INTERFACE_MATH_UNAIRE_DOUBLE(dexp)
!INTERFACE_MATH_UNAIRE_DOUBLE(dlog)



  interface abs
    module procedure abs_complex_single_st
  end interface abs

  interface abs
    module procedure abs_complex_double_st
  end interface abs

  interface acos
    module procedure acos_complex_single_st
  end interface acos

  interface acos
    module procedure acos_complex_double_st
  end interface acos

  interface acosh
    module procedure acosh_complex_single_st
  end interface acosh

  interface acosh
    module procedure acosh_complex_double_st
  end interface acosh

  interface asin
    module procedure asin_complex_single_st
  end interface asin

  interface asin
    module procedure asin_complex_double_st
  end interface asin

  interface asinh
    module procedure asinh_complex_single_st
  end interface asinh

  interface asinh
    module procedure asinh_complex_double_st
  end interface asinh

  interface atan
    module procedure atan_complex_single_st
  end interface atan

  interface atan
    module procedure atan_complex_double_st
  end interface atan

  interface atanh
    module procedure atanh_complex_single_st
  end interface atanh

  interface atanh
    module procedure atanh_complex_double_st
  end interface atanh

  interface cos
    module procedure cos_complex_single_st
  end interface cos

  interface cos
    module procedure cos_complex_double_st
  end interface cos

  interface cosh
    module procedure cosh_complex_single_st
  end interface cosh

  interface cosh
    module procedure cosh_complex_double_st
  end interface cosh

  interface exp
    module procedure exp_complex_single_st
  end interface exp

  interface exp
    module procedure exp_complex_double_st
  end interface exp

  interface log
    module procedure log_complex_single_st
  end interface log

  interface log
    module procedure log_complex_double_st
  end interface log

  interface sin
    module procedure sin_complex_single_st
  end interface sin

  interface sin
    module procedure sin_complex_double_st
  end interface sin

  interface sinh
    module procedure sinh_complex_single_st
  end interface sinh

  interface sinh
    module procedure sinh_complex_double_st
  end interface sinh

  interface sqrt
    module procedure sqrt_complex_single_st
  end interface sqrt

  interface sqrt
    module procedure sqrt_complex_double_st
  end interface sqrt

  interface tan
    module procedure tan_complex_single_st
  end interface tan

  interface tan
    module procedure tan_complex_double_st
  end interface tan

  interface tanh
    module procedure tanh_complex_single_st
  end interface tanh

  interface tanh
    module procedure tanh_complex_double_st
  end interface tanh


contains

!define(CONTAINS_MATH_UNAIRE_DOUBLE,
!`  elemental function $1_double_st(a)'
!`    type(double_st), intent(in) :: a'
!`    type(double_st) $1_double_st'
!
!`    $1_double_st = translit( $1, `^d' )(a)'
!`  end function $1_double_st'
!)
!
!CONTAINS_MATH_UNAIRE_DOUBLE(dabs)
!CONTAINS_MATH_UNAIRE_DOUBLE(dsqrt)
!CONTAINS_MATH_UNAIRE_DOUBLE(dsin)
!CONTAINS_MATH_UNAIRE_DOUBLE(dcos)
!CONTAINS_MATH_UNAIRE_DOUBLE(dtan)
!CONTAINS_MATH_UNAIRE_DOUBLE(dexp)
!CONTAINS_MATH_UNAIRE_DOUBLE(dlog)



  elemental function abs_complex_single_st(z)
    type(complex_single_st), intent(in) :: z
    type(single_st) abs_complex_single_st
    

    abs_complex_single_st = (z%x**2 + z%y**2)**0.5
  end function abs_complex_single_st

  elemental function abs_complex_double_st(z)
    type(complex_double_st), intent(in) :: z
    type(double_st) abs_complex_double_st
    

    abs_complex_double_st = (z%x**2 + z%y**2)**0.5
  end function abs_complex_double_st

  elemental function acos_complex_single_st(z)
    type(complex_single_st), intent(in) :: z
    type(complex_single_st) acos_complex_single_st 
    

    acos_complex_single_st = (0.,-1.)*log(z + (0.,1.)*sqrt(1.-z*z))
  end function acos_complex_single_st

  elemental function acos_complex_double_st(z)
    type(complex_double_st), intent(in) :: z
    type(complex_double_st) acos_complex_double_st 
    

    acos_complex_double_st = (0.,-1.)*log(z + (0.,1.)*sqrt(1.-z*z))
  end function acos_complex_double_st

  elemental function acosh_complex_single_st(z)
    type(complex_single_st), intent(in) :: z
    type(complex_single_st) acosh_complex_single_st 
    

    acosh_complex_single_st = log(z + sqrt(z-1.)*sqrt(z+1.))
  end function acosh_complex_single_st

  elemental function acosh_complex_double_st(z)
    type(complex_double_st), intent(in) :: z
    type(complex_double_st) acosh_complex_double_st 
    

    acosh_complex_double_st = log(z + sqrt(z-1.)*sqrt(z+1.))
  end function acosh_complex_double_st

  elemental function asin_complex_single_st(z)
    type(complex_single_st), intent(in) :: z
    type(complex_single_st) asin_complex_single_st 
    

    asin_complex_single_st = (0.,-1.)*log((0.,1.)*z + sqrt(1.-z*z))
  end function asin_complex_single_st

  elemental function asin_complex_double_st(z)
    type(complex_double_st), intent(in) :: z
    type(complex_double_st) asin_complex_double_st 
    

    asin_complex_double_st = (0.,-1.)*log((0.,1.)*z + sqrt(1.-z*z))
  end function asin_complex_double_st

  elemental function asinh_complex_single_st(z)
    type(complex_single_st), intent(in) :: z
    type(complex_single_st) asinh_complex_single_st 
    

    asinh_complex_single_st = log(z + sqrt(z*z+1.))
  end function asinh_complex_single_st

  elemental function asinh_complex_double_st(z)
    type(complex_double_st), intent(in) :: z
    type(complex_double_st) asinh_complex_double_st 
    

    asinh_complex_double_st = log(z + sqrt(z*z+1.))
  end function asinh_complex_double_st

  elemental function atan_complex_single_st(z)
    type(complex_single_st), intent(in) :: z
    type(complex_single_st) atan_complex_single_st 
    

    atan_complex_single_st = asin(z)/acos(z)
  end function atan_complex_single_st

  elemental function atan_complex_double_st(z)
    type(complex_double_st), intent(in) :: z
    type(complex_double_st) atan_complex_double_st 
    

    atan_complex_double_st = asin(z)/acos(z)
  end function atan_complex_double_st

  elemental function atanh_complex_single_st(z)
    type(complex_single_st), intent(in) :: z
    type(complex_single_st) atanh_complex_single_st 
    

    atanh_complex_single_st = asinh(z)/acosh(z) 
  end function atanh_complex_single_st

  elemental function atanh_complex_double_st(z)
    type(complex_double_st), intent(in) :: z
    type(complex_double_st) atanh_complex_double_st 
    

    atanh_complex_double_st = asinh(z)/acosh(z) 
  end function atanh_complex_double_st

  elemental function cos_complex_single_st(z)
    type(complex_single_st), intent(in) :: z
    type(complex_single_st) cos_complex_single_st 
    

    cos_complex_single_st%x =  cos(z%x)*cosh(z%y)
    cos_complex_single_st%y = -sin(z%x)*sinh(z%y)
  end function cos_complex_single_st

  elemental function cos_complex_double_st(z)
    type(complex_double_st), intent(in) :: z
    type(complex_double_st) cos_complex_double_st 
    

    cos_complex_double_st%x =  cos(z%x)*cosh(z%y)
    cos_complex_double_st%y = -sin(z%x)*sinh(z%y)
  end function cos_complex_double_st

  elemental function cosh_complex_single_st(z)
    type(complex_single_st), intent(in) :: z
    type(complex_single_st) cosh_complex_single_st 
    

    cosh_complex_single_st%x = cosh(z%x)*cos(z%y)
    cosh_complex_single_st%y = sinh(z%x)*sin(z%y)
  end function cosh_complex_single_st

  elemental function cosh_complex_double_st(z)
    type(complex_double_st), intent(in) :: z
    type(complex_double_st) cosh_complex_double_st 
    

    cosh_complex_double_st%x = cosh(z%x)*cos(z%y)
    cosh_complex_double_st%y = sinh(z%x)*sin(z%y)
  end function cosh_complex_double_st

  elemental function exp_complex_single_st(z)
    type(complex_single_st), intent(in) :: z
    type(complex_single_st) exp_complex_single_st 
    

    exp_complex_single_st%x = exp(z%x)*cos(z%y)
    exp_complex_single_st%y = exp(z%x)*sin(z%y)
  end function exp_complex_single_st

  elemental function exp_complex_double_st(z)
    type(complex_double_st), intent(in) :: z
    type(complex_double_st) exp_complex_double_st 
    

    exp_complex_double_st%x = exp(z%x)*cos(z%y)
    exp_complex_double_st%y = exp(z%x)*sin(z%y)
  end function exp_complex_double_st

  elemental function log_complex_single_st(z)
    type(complex_single_st), intent(in) :: z
    type(complex_single_st) log_complex_single_st 
    

    log_complex_single_st%x = log(abs(z))
    log_complex_single_st%y = atan(z%y/z%x)
    if (z%x<0. .AND. z%y<0.) log_complex_single_st%y = log_complex_single_st%y - acos(-z%x/z%x)
    if (z%x<0. .AND. z%y>0.) log_complex_single_st%y = log_complex_single_st%y + acos(-z%x/z%x)
  end function log_complex_single_st

  elemental function log_complex_double_st(z)
    type(complex_double_st), intent(in) :: z
    type(complex_double_st) log_complex_double_st 
    

    log_complex_double_st%x = log(abs(z))
    log_complex_double_st%y = atan(z%y/z%x)
    if (z%x<0. .AND. z%y<0.) log_complex_double_st%y = log_complex_double_st%y - acos(-z%x/z%x)
    if (z%x<0. .AND. z%y>0.) log_complex_double_st%y = log_complex_double_st%y + acos(-z%x/z%x)
  end function log_complex_double_st

  elemental function sin_complex_single_st(z)
    type(complex_single_st), intent(in) :: z
    type(complex_single_st) sin_complex_single_st 
    

    sin_complex_single_st%x = sin(z%x)*cosh(z%y)
    sin_complex_single_st%y = cos(z%x)*sinh(z%y)
  end function sin_complex_single_st

  elemental function sin_complex_double_st(z)
    type(complex_double_st), intent(in) :: z
    type(complex_double_st) sin_complex_double_st 
    

    sin_complex_double_st%x = sin(z%x)*cosh(z%y)
    sin_complex_double_st%y = cos(z%x)*sinh(z%y)
  end function sin_complex_double_st

  elemental function sinh_complex_single_st(z)
    type(complex_single_st), intent(in) :: z
    type(complex_single_st) sinh_complex_single_st 
    

    sinh_complex_single_st%x = sinh(z%x)*cos(z%y)
    sinh_complex_single_st%y = cosh(z%x)*sin(z%y)
  end function sinh_complex_single_st

  elemental function sinh_complex_double_st(z)
    type(complex_double_st), intent(in) :: z
    type(complex_double_st) sinh_complex_double_st 
    

    sinh_complex_double_st%x = sinh(z%x)*cos(z%y)
    sinh_complex_double_st%y = cosh(z%x)*sin(z%y)
  end function sinh_complex_double_st

  elemental function sqrt_complex_single_st(z)
    type(complex_single_st), intent(in) :: z
    type(complex_single_st) sqrt_complex_single_st 
    type(single_st) rho, theta

    rho = abs(z)
    theta = atan(z%y/z%x)
    if (z%x<0. .AND. z%y<0.) theta = theta - acos(-z%x/z%x)
    if (z%x<0. .AND. z%y>0.) theta = theta + acos(-z%x/z%x)
    rho = sqrt(rho)
    theta = theta/2
    sqrt_complex_single_st%x = rho*cos(theta)
    sqrt_complex_single_st%y = rho*sin(theta)
  end function sqrt_complex_single_st

  elemental function sqrt_complex_double_st(z)
    type(complex_double_st), intent(in) :: z
    type(complex_double_st) sqrt_complex_double_st 
    type(double_st) rho, theta

    rho = abs(z)
    theta = atan(z%y/z%x)
    if (z%x<0. .AND. z%y<0.) theta = theta - acos(-z%x/z%x)
    if (z%x<0. .AND. z%y>0.) theta = theta + acos(-z%x/z%x)
    rho = sqrt(rho)
    theta = theta/2
    sqrt_complex_double_st%x = rho*cos(theta)
    sqrt_complex_double_st%y = rho*sin(theta)
  end function sqrt_complex_double_st

  elemental function tan_complex_single_st(z)
    type(complex_single_st), intent(in) :: z
    type(complex_single_st) tan_complex_single_st 
    

    tan_complex_single_st =  sin(z)/cos(z)
  end function tan_complex_single_st

  elemental function tan_complex_double_st(z)
    type(complex_double_st), intent(in) :: z
    type(complex_double_st) tan_complex_double_st 
    

    tan_complex_double_st =  sin(z)/cos(z)
  end function tan_complex_double_st

  elemental function tanh_complex_single_st(z)
    type(complex_single_st), intent(in) :: z
    type(complex_single_st) tanh_complex_single_st 
    

    tanh_complex_single_st =  sinh(z)/cosh(z)
  end function tanh_complex_single_st

  elemental function tanh_complex_double_st(z)
    type(complex_double_st), intent(in) :: z
    type(complex_double_st) tanh_complex_double_st 
    

    tanh_complex_double_st =  sinh(z)/cosh(z)
  end function tanh_complex_double_st




end module cadna_cmplx_math

