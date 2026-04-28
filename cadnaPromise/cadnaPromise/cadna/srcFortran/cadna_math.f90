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




module cadna_math
  use ISO_C_BINDING, only : C_SHORT, C_INT, C_LONG_LONG
  use cadna_types
  use cadna_affect
  use cadna_add
  use cadna_sub
  use cadna_mult
  use cadna_pow

  implicit none




  interface dabs
    module procedure dabs_double_st
  end interface

  interface dsqrt
    module procedure dsqrt_double_st
  end interface

  interface dsin
    module procedure dsin_double_st
  end interface

  interface dcos
    module procedure dcos_double_st
  end interface

  interface dtan
    module procedure dtan_double_st
  end interface

  interface dexp
    module procedure dexp_double_st
  end interface

  interface dlog
    module procedure dlog_double_st
  end interface




  interface abs
    module procedure abs_single_st
  end interface
  interface
    pure function cpp_abs_single_st(a) bind(C)
      import single_st
      
      type(single_st), intent(in) :: a
      type(single_st)  cpp_abs_single_st
    end function cpp_abs_single_st
  end interface

  interface abs
    module procedure abs_double_st
  end interface
  interface
    pure function cpp_abs_double_st(a) bind(C)
      import double_st
      
      type(double_st), intent(in) :: a
      type(double_st)  cpp_abs_double_st
    end function cpp_abs_double_st
  end interface

  interface acos
    module procedure acos_single_st
  end interface
  interface
    pure function cpp_acos_single_st(a) bind(C)
      import single_st
      
      type(single_st), intent(in) :: a
      type(single_st)  cpp_acos_single_st
    end function cpp_acos_single_st
  end interface

  interface acos
    module procedure acos_double_st
  end interface
  interface
    pure function cpp_acos_double_st(a) bind(C)
      import double_st
      
      type(double_st), intent(in) :: a
      type(double_st)  cpp_acos_double_st
    end function cpp_acos_double_st
  end interface

  interface acosh
    module procedure acosh_single_st
  end interface
  interface
    pure function cpp_acosh_single_st(a) bind(C)
      import single_st
      
      type(single_st), intent(in) :: a
      type(single_st)  cpp_acosh_single_st
    end function cpp_acosh_single_st
  end interface

  interface acosh
    module procedure acosh_double_st
  end interface
  interface
    pure function cpp_acosh_double_st(a) bind(C)
      import double_st
      
      type(double_st), intent(in) :: a
      type(double_st)  cpp_acosh_double_st
    end function cpp_acosh_double_st
  end interface

  interface aint
    module procedure aint_single_st
  end interface
  interface
    pure function cpp_aint_single_st(a) bind(C)
      import single_st
      
      type(single_st), intent(in) :: a
      type(single_st)  cpp_aint_single_st
    end function cpp_aint_single_st
  end interface

  interface aint
    module procedure aint_double_st
  end interface
  interface
    pure function cpp_aint_double_st(a) bind(C)
      import double_st
      
      type(double_st), intent(in) :: a
      type(double_st)  cpp_aint_double_st
    end function cpp_aint_double_st
  end interface

!INTERFACE_MATH_UNAIRE(anint,   single_st, int)
!INTERFACE_MATH_UNAIRE(anint,   double_st, int)
  interface asin
    module procedure asin_single_st
  end interface
  interface
    pure function cpp_asin_single_st(a) bind(C)
      import single_st
      
      type(single_st), intent(in) :: a
      type(single_st)  cpp_asin_single_st
    end function cpp_asin_single_st
  end interface

  interface asin
    module procedure asin_double_st
  end interface
  interface
    pure function cpp_asin_double_st(a) bind(C)
      import double_st
      
      type(double_st), intent(in) :: a
      type(double_st)  cpp_asin_double_st
    end function cpp_asin_double_st
  end interface

  interface asinh
    module procedure asinh_single_st
  end interface
  interface
    pure function cpp_asinh_single_st(a) bind(C)
      import single_st
      
      type(single_st), intent(in) :: a
      type(single_st)  cpp_asinh_single_st
    end function cpp_asinh_single_st
  end interface

  interface asinh
    module procedure asinh_double_st
  end interface
  interface
    pure function cpp_asinh_double_st(a) bind(C)
      import double_st
      
      type(double_st), intent(in) :: a
      type(double_st)  cpp_asinh_double_st
    end function cpp_asinh_double_st
  end interface

  interface atan
    module procedure atan_single_st
  end interface
  interface
    pure function cpp_atan_single_st(a) bind(C)
      import single_st
      
      type(single_st), intent(in) :: a
      type(single_st)  cpp_atan_single_st
    end function cpp_atan_single_st
  end interface

  interface atan
    module procedure atan_double_st
  end interface
  interface
    pure function cpp_atan_double_st(a) bind(C)
      import double_st
      
      type(double_st), intent(in) :: a
      type(double_st)  cpp_atan_double_st
    end function cpp_atan_double_st
  end interface

  interface atanh
    module procedure atanh_single_st
  end interface
  interface
    pure function cpp_atanh_single_st(a) bind(C)
      import single_st
      
      type(single_st), intent(in) :: a
      type(single_st)  cpp_atanh_single_st
    end function cpp_atanh_single_st
  end interface

  interface atanh
    module procedure atanh_double_st
  end interface
  interface
    pure function cpp_atanh_double_st(a) bind(C)
      import double_st
      
      type(double_st), intent(in) :: a
      type(double_st)  cpp_atanh_double_st
    end function cpp_atanh_double_st
  end interface

  interface ceiling
    module procedure ceiling_single_st
  end interface
  interface
    pure function cpp_ceiling_single_st(a) bind(C)
      import single_st
      import C_INT
      type(single_st), intent(in) :: a
      integer(C_INT) cpp_ceiling_single_st
    end function cpp_ceiling_single_st
  end interface

  interface ceiling
    module procedure ceiling_double_st
  end interface
  interface
    pure function cpp_ceiling_double_st(a) bind(C)
      import double_st
      import C_INT
      type(double_st), intent(in) :: a
      integer(C_INT) cpp_ceiling_double_st
    end function cpp_ceiling_double_st
  end interface

  interface cos
    module procedure cos_single_st
  end interface
  interface
    pure function cpp_cos_single_st(a) bind(C)
      import single_st
      
      type(single_st), intent(in) :: a
      type(single_st)  cpp_cos_single_st
    end function cpp_cos_single_st
  end interface

  interface cos
    module procedure cos_double_st
  end interface
  interface
    pure function cpp_cos_double_st(a) bind(C)
      import double_st
      
      type(double_st), intent(in) :: a
      type(double_st)  cpp_cos_double_st
    end function cpp_cos_double_st
  end interface

  interface cosh
    module procedure cosh_single_st
  end interface
  interface
    pure function cpp_cosh_single_st(a) bind(C)
      import single_st
      
      type(single_st), intent(in) :: a
      type(single_st)  cpp_cosh_single_st
    end function cpp_cosh_single_st
  end interface

  interface cosh
    module procedure cosh_double_st
  end interface
  interface
    pure function cpp_cosh_double_st(a) bind(C)
      import double_st
      
      type(double_st), intent(in) :: a
      type(double_st)  cpp_cosh_double_st
    end function cpp_cosh_double_st
  end interface

  interface erf
    module procedure erf_single_st
  end interface
  interface
    pure function cpp_erf_single_st(a) bind(C)
      import single_st
      
      type(single_st), intent(in) :: a
      type(single_st)  cpp_erf_single_st
    end function cpp_erf_single_st
  end interface

  interface erf
    module procedure erf_double_st
  end interface
  interface
    pure function cpp_erf_double_st(a) bind(C)
      import double_st
      
      type(double_st), intent(in) :: a
      type(double_st)  cpp_erf_double_st
    end function cpp_erf_double_st
  end interface

  interface erfc
    module procedure erfc_single_st
  end interface
  interface
    pure function cpp_erfc_single_st(a) bind(C)
      import single_st
      
      type(single_st), intent(in) :: a
      type(single_st)  cpp_erfc_single_st
    end function cpp_erfc_single_st
  end interface

  interface erfc
    module procedure erfc_double_st
  end interface
  interface
    pure function cpp_erfc_double_st(a) bind(C)
      import double_st
      
      type(double_st), intent(in) :: a
      type(double_st)  cpp_erfc_double_st
    end function cpp_erfc_double_st
  end interface

  interface exp
    module procedure exp_single_st
  end interface
  interface
    pure function cpp_exp_single_st(a) bind(C)
      import single_st
      
      type(single_st), intent(in) :: a
      type(single_st)  cpp_exp_single_st
    end function cpp_exp_single_st
  end interface

  interface exp
    module procedure exp_double_st
  end interface
  interface
    pure function cpp_exp_double_st(a) bind(C)
      import double_st
      
      type(double_st), intent(in) :: a
      type(double_st)  cpp_exp_double_st
    end function cpp_exp_double_st
  end interface

  interface floor
    module procedure floor_single_st
  end interface
  interface
    pure function cpp_floor_single_st(a) bind(C)
      import single_st
      import C_INT
      type(single_st), intent(in) :: a
      integer(C_INT) cpp_floor_single_st
    end function cpp_floor_single_st
  end interface

  interface floor
    module procedure floor_double_st
  end interface
  interface
    pure function cpp_floor_double_st(a) bind(C)
      import double_st
      import C_INT
      type(double_st), intent(in) :: a
      integer(C_INT) cpp_floor_double_st
    end function cpp_floor_double_st
  end interface

  interface log
    module procedure log_single_st
  end interface
  interface
    pure function cpp_log_single_st(a) bind(C)
      import single_st
      
      type(single_st), intent(in) :: a
      type(single_st)  cpp_log_single_st
    end function cpp_log_single_st
  end interface

  interface log
    module procedure log_double_st
  end interface
  interface
    pure function cpp_log_double_st(a) bind(C)
      import double_st
      
      type(double_st), intent(in) :: a
      type(double_st)  cpp_log_double_st
    end function cpp_log_double_st
  end interface

  interface log10
    module procedure log10_single_st
  end interface
  interface
    pure function cpp_log10_single_st(a) bind(C)
      import single_st
      
      type(single_st), intent(in) :: a
      type(single_st)  cpp_log10_single_st
    end function cpp_log10_single_st
  end interface

  interface log10
    module procedure log10_double_st
  end interface
  interface
    pure function cpp_log10_double_st(a) bind(C)
      import double_st
      
      type(double_st), intent(in) :: a
      type(double_st)  cpp_log10_double_st
    end function cpp_log10_double_st
  end interface

  interface nint
    module procedure nint_single_st
  end interface
  interface
    pure function cpp_nint_single_st(a) bind(C)
      import single_st
      import C_INT
      type(single_st), intent(in) :: a
      integer(C_INT) cpp_nint_single_st
    end function cpp_nint_single_st
  end interface

  interface nint
    module procedure nint_double_st
  end interface
  interface
    pure function cpp_nint_double_st(a) bind(C)
      import double_st
      import C_INT
      type(double_st), intent(in) :: a
      integer(C_INT) cpp_nint_double_st
    end function cpp_nint_double_st
  end interface

  interface sin
    module procedure sin_single_st
  end interface
  interface
    pure function cpp_sin_single_st(a) bind(C)
      import single_st
      
      type(single_st), intent(in) :: a
      type(single_st)  cpp_sin_single_st
    end function cpp_sin_single_st
  end interface

  interface sin
    module procedure sin_double_st
  end interface
  interface
    pure function cpp_sin_double_st(a) bind(C)
      import double_st
      
      type(double_st), intent(in) :: a
      type(double_st)  cpp_sin_double_st
    end function cpp_sin_double_st
  end interface

  interface sinh
    module procedure sinh_single_st
  end interface
  interface
    pure function cpp_sinh_single_st(a) bind(C)
      import single_st
      
      type(single_st), intent(in) :: a
      type(single_st)  cpp_sinh_single_st
    end function cpp_sinh_single_st
  end interface

  interface sinh
    module procedure sinh_double_st
  end interface
  interface
    pure function cpp_sinh_double_st(a) bind(C)
      import double_st
      
      type(double_st), intent(in) :: a
      type(double_st)  cpp_sinh_double_st
    end function cpp_sinh_double_st
  end interface

  interface sqrt
    module procedure sqrt_single_st
  end interface
  interface
    pure function cpp_sqrt_single_st(a) bind(C)
      import single_st
      
      type(single_st), intent(in) :: a
      type(single_st)  cpp_sqrt_single_st
    end function cpp_sqrt_single_st
  end interface

  interface sqrt
    module procedure sqrt_double_st
  end interface
  interface
    pure function cpp_sqrt_double_st(a) bind(C)
      import double_st
      
      type(double_st), intent(in) :: a
      type(double_st)  cpp_sqrt_double_st
    end function cpp_sqrt_double_st
  end interface

  interface tan
    module procedure tan_single_st
  end interface
  interface
    pure function cpp_tan_single_st(a) bind(C)
      import single_st
      
      type(single_st), intent(in) :: a
      type(single_st)  cpp_tan_single_st
    end function cpp_tan_single_st
  end interface

  interface tan
    module procedure tan_double_st
  end interface
  interface
    pure function cpp_tan_double_st(a) bind(C)
      import double_st
      
      type(double_st), intent(in) :: a
      type(double_st)  cpp_tan_double_st
    end function cpp_tan_double_st
  end interface

  interface tanh
    module procedure tanh_single_st
  end interface
  interface
    pure function cpp_tanh_single_st(a) bind(C)
      import single_st
      
      type(single_st), intent(in) :: a
      type(single_st)  cpp_tanh_single_st
    end function cpp_tanh_single_st
  end interface

  interface tanh
    module procedure tanh_double_st
  end interface
  interface
    pure function cpp_tanh_double_st(a) bind(C)
      import double_st
      
      type(double_st), intent(in) :: a
      type(double_st)  cpp_tanh_double_st
    end function cpp_tanh_double_st
  end interface




  interface atan2
    module procedure atan2_single_st_single_st
  end interface
  interface
    pure function cpp_atan2_single_st_single_st(a, b) bind(C)
      import single_st
      
      type(single_st), intent(in) :: a
      type(single_st), intent(in) :: b
      type(single_st) cpp_atan2_single_st_single_st
    end function cpp_atan2_single_st_single_st
  end interface

  interface atan2
    module procedure atan2_double_st_double_st
  end interface
  interface
    pure function cpp_atan2_double_st_double_st(a, b) bind(C)
      import double_st
      
      type(double_st), intent(in) :: a
      type(double_st), intent(in) :: b
      type(double_st) cpp_atan2_double_st_double_st
    end function cpp_atan2_double_st_double_st
  end interface

  interface max
    module procedure max_single_st_single_st
  end interface
  interface
    pure function cpp_max_single_st_single_st(a, b) bind(C)
      import single_st
      
      type(single_st), intent(in) :: a
      type(single_st), intent(in) :: b
      type(single_st) cpp_max_single_st_single_st
    end function cpp_max_single_st_single_st
  end interface

  interface max
    module procedure max_double_st_double_st
  end interface
  interface
    pure function cpp_max_double_st_double_st(a, b) bind(C)
      import double_st
      
      type(double_st), intent(in) :: a
      type(double_st), intent(in) :: b
      type(double_st) cpp_max_double_st_double_st
    end function cpp_max_double_st_double_st
  end interface

  interface max
    module procedure max_float_single_st
  end interface
  interface
    pure function cpp_max_float_single_st(a, b) bind(C)
      import C_FLOAT
      import single_st
      real(C_FLOAT), intent(in) :: a
      type(single_st), intent(in) :: b
      type(single_st) cpp_max_float_single_st
    end function cpp_max_float_single_st
  end interface

  interface max
    module procedure max_double_double_st
  end interface
  interface
    pure function cpp_max_double_double_st(a, b) bind(C)
      import C_DOUBLE
      import double_st
      real(C_DOUBLE), intent(in) :: a
      type(double_st), intent(in) :: b
      type(double_st) cpp_max_double_double_st
    end function cpp_max_double_double_st
  end interface

  interface max
    module procedure max_single_st_float
  end interface
  interface
    pure function cpp_max_single_st_float(a, b) bind(C)
      import single_st
      import C_FLOAT
      type(single_st), intent(in) :: a
      real(C_FLOAT), intent(in) :: b
      type(single_st) cpp_max_single_st_float
    end function cpp_max_single_st_float
  end interface

  interface max
    module procedure max_double_st_double
  end interface
  interface
    pure function cpp_max_double_st_double(a, b) bind(C)
      import double_st
      import C_DOUBLE
      type(double_st), intent(in) :: a
      real(C_DOUBLE), intent(in) :: b
      type(double_st) cpp_max_double_st_double
    end function cpp_max_double_st_double
  end interface

  interface min
    module procedure min_single_st_single_st
  end interface
  interface
    pure function cpp_min_single_st_single_st(a, b) bind(C)
      import single_st
      
      type(single_st), intent(in) :: a
      type(single_st), intent(in) :: b
      type(single_st) cpp_min_single_st_single_st
    end function cpp_min_single_st_single_st
  end interface

  interface min
    module procedure min_double_st_double_st
  end interface
  interface
    pure function cpp_min_double_st_double_st(a, b) bind(C)
      import double_st
      
      type(double_st), intent(in) :: a
      type(double_st), intent(in) :: b
      type(double_st) cpp_min_double_st_double_st
    end function cpp_min_double_st_double_st
  end interface

  interface min
    module procedure min_float_single_st
  end interface
  interface
    pure function cpp_min_float_single_st(a, b) bind(C)
      import C_FLOAT
      import single_st
      real(C_FLOAT), intent(in) :: a
      type(single_st), intent(in) :: b
      type(single_st) cpp_min_float_single_st
    end function cpp_min_float_single_st
  end interface

  interface min
    module procedure min_double_double_st
  end interface
  interface
    pure function cpp_min_double_double_st(a, b) bind(C)
      import C_DOUBLE
      import double_st
      real(C_DOUBLE), intent(in) :: a
      type(double_st), intent(in) :: b
      type(double_st) cpp_min_double_double_st
    end function cpp_min_double_double_st
  end interface

  interface min
    module procedure min_single_st_float
  end interface
  interface
    pure function cpp_min_single_st_float(a, b) bind(C)
      import single_st
      import C_FLOAT
      type(single_st), intent(in) :: a
      real(C_FLOAT), intent(in) :: b
      type(single_st) cpp_min_single_st_float
    end function cpp_min_single_st_float
  end interface

  interface min
    module procedure min_double_st_double
  end interface
  interface
    pure function cpp_min_double_st_double(a, b) bind(C)
      import double_st
      import C_DOUBLE
      type(double_st), intent(in) :: a
      real(C_DOUBLE), intent(in) :: b
      type(double_st) cpp_min_double_st_double
    end function cpp_min_double_st_double
  end interface

  interface mod
    module procedure mod_single_st_single_st
  end interface
  interface
    pure function cpp_mod_single_st_single_st(a, b) bind(C)
      import single_st
      
      type(single_st), intent(in) :: a
      type(single_st), intent(in) :: b
      type(single_st) cpp_mod_single_st_single_st
    end function cpp_mod_single_st_single_st
  end interface

  interface mod
    module procedure mod_double_st_double_st
  end interface
  interface
    pure function cpp_mod_double_st_double_st(a, b) bind(C)
      import double_st
      
      type(double_st), intent(in) :: a
      type(double_st), intent(in) :: b
      type(double_st) cpp_mod_double_st_double_st
    end function cpp_mod_double_st_double_st
  end interface




  interface dim
    module procedure dim_single_st_single_st
  end interface dim

  interface dim
    module procedure dim_double_st_double_st
  end interface dim

  interface dim
    module procedure dim_float_single_st
  end interface dim

  interface dim
    module procedure dim_double_double_st
  end interface dim

  interface dim
    module procedure dim_single_st_float
  end interface dim

  interface dim
    module procedure dim_double_st_double
  end interface dim

  interface hypot
    module procedure hypot_single_st_single_st
  end interface hypot

  interface hypot
    module procedure hypot_double_st_double_st
  end interface hypot

  interface hypot
    module procedure hypot_float_single_st
  end interface hypot

  interface hypot
    module procedure hypot_double_double_st
  end interface hypot

  interface hypot
    module procedure hypot_single_st_float
  end interface hypot

  interface hypot
    module procedure hypot_double_st_double
  end interface hypot

  interface sign
    module procedure sign_single_st_single_st
  end interface sign

  interface sign
    module procedure sign_double_st_double_st
  end interface sign

  interface sign
    module procedure sign_float_single_st
  end interface sign

  interface sign
    module procedure sign_double_double_st
  end interface sign

  interface sign
    module procedure sign_single_st_float
  end interface sign

  interface sign
    module procedure sign_double_st_double
  end interface sign


contains



  elemental function dabs_double_st(a)
    type(double_st), intent(in) :: a
    type(double_st) dabs_double_st

    dabs_double_st = abs(a)
  end function dabs_double_st

  elemental function dsqrt_double_st(a)
    type(double_st), intent(in) :: a
    type(double_st) dsqrt_double_st

    dsqrt_double_st = sqrt(a)
  end function dsqrt_double_st

  elemental function dsin_double_st(a)
    type(double_st), intent(in) :: a
    type(double_st) dsin_double_st

    dsin_double_st = sin(a)
  end function dsin_double_st

  elemental function dcos_double_st(a)
    type(double_st), intent(in) :: a
    type(double_st) dcos_double_st

    dcos_double_st = cos(a)
  end function dcos_double_st

  elemental function dtan_double_st(a)
    type(double_st), intent(in) :: a
    type(double_st) dtan_double_st

    dtan_double_st = tan(a)
  end function dtan_double_st

  elemental function dexp_double_st(a)
    type(double_st), intent(in) :: a
    type(double_st) dexp_double_st

    dexp_double_st = exp(a)
  end function dexp_double_st

  elemental function dlog_double_st(a)
    type(double_st), intent(in) :: a
    type(double_st) dlog_double_st

    dlog_double_st = log(a)
  end function dlog_double_st




  elemental function abs_single_st(a)
    type(single_st), intent(in) :: a
    type(single_st) abs_single_st

    abs_single_st = cpp_abs_single_st(a)
  end function abs_single_st

  elemental function abs_double_st(a)
    type(double_st), intent(in) :: a
    type(double_st) abs_double_st

    abs_double_st = cpp_abs_double_st(a)
  end function abs_double_st

  elemental function acos_single_st(a)
    type(single_st), intent(in) :: a
    type(single_st) acos_single_st

    acos_single_st = cpp_acos_single_st(a)
  end function acos_single_st

  elemental function acos_double_st(a)
    type(double_st), intent(in) :: a
    type(double_st) acos_double_st

    acos_double_st = cpp_acos_double_st(a)
  end function acos_double_st

  elemental function acosh_single_st(a)
    type(single_st), intent(in) :: a
    type(single_st) acosh_single_st

    acosh_single_st = cpp_acosh_single_st(a)
  end function acosh_single_st

  elemental function acosh_double_st(a)
    type(double_st), intent(in) :: a
    type(double_st) acosh_double_st

    acosh_double_st = cpp_acosh_double_st(a)
  end function acosh_double_st

  elemental function aint_single_st(a)
    type(single_st), intent(in) :: a
    type(single_st) aint_single_st

    aint_single_st = cpp_aint_single_st(a)
  end function aint_single_st

  elemental function aint_double_st(a)
    type(double_st), intent(in) :: a
    type(double_st) aint_double_st

    aint_double_st = cpp_aint_double_st(a)
  end function aint_double_st

!CONTAINS_MATH_UNAIRE(anint,   single_st, single_st)
!CONTAINS_MATH_UNAIRE(anint,   double_st, double_st)
  elemental function asin_single_st(a)
    type(single_st), intent(in) :: a
    type(single_st) asin_single_st

    asin_single_st = cpp_asin_single_st(a)
  end function asin_single_st

  elemental function asin_double_st(a)
    type(double_st), intent(in) :: a
    type(double_st) asin_double_st

    asin_double_st = cpp_asin_double_st(a)
  end function asin_double_st

  elemental function asinh_single_st(a)
    type(single_st), intent(in) :: a
    type(single_st) asinh_single_st

    asinh_single_st = cpp_asinh_single_st(a)
  end function asinh_single_st

  elemental function asinh_double_st(a)
    type(double_st), intent(in) :: a
    type(double_st) asinh_double_st

    asinh_double_st = cpp_asinh_double_st(a)
  end function asinh_double_st

  elemental function atan_single_st(a)
    type(single_st), intent(in) :: a
    type(single_st) atan_single_st

    atan_single_st = cpp_atan_single_st(a)
  end function atan_single_st

  elemental function atan_double_st(a)
    type(double_st), intent(in) :: a
    type(double_st) atan_double_st

    atan_double_st = cpp_atan_double_st(a)
  end function atan_double_st

  elemental function atanh_single_st(a)
    type(single_st), intent(in) :: a
    type(single_st) atanh_single_st

    atanh_single_st = cpp_atanh_single_st(a)
  end function atanh_single_st

  elemental function atanh_double_st(a)
    type(double_st), intent(in) :: a
    type(double_st) atanh_double_st

    atanh_double_st = cpp_atanh_double_st(a)
  end function atanh_double_st

  elemental function ceiling_single_st(a)
    type(single_st), intent(in) :: a
    integer(C_INT) ceiling_single_st

    ceiling_single_st = int(cpp_ceiling_single_st(a), C_INT)
  end function ceiling_single_st

  elemental function ceiling_double_st(a)
    type(double_st), intent(in) :: a
    integer(C_INT) ceiling_double_st

    ceiling_double_st = int(cpp_ceiling_double_st(a), C_INT)
  end function ceiling_double_st

  elemental function cos_single_st(a)
    type(single_st), intent(in) :: a
    type(single_st) cos_single_st

    cos_single_st = cpp_cos_single_st(a)
  end function cos_single_st

  elemental function cos_double_st(a)
    type(double_st), intent(in) :: a
    type(double_st) cos_double_st

    cos_double_st = cpp_cos_double_st(a)
  end function cos_double_st

  elemental function cosh_single_st(a)
    type(single_st), intent(in) :: a
    type(single_st) cosh_single_st

    cosh_single_st = cpp_cosh_single_st(a)
  end function cosh_single_st

  elemental function cosh_double_st(a)
    type(double_st), intent(in) :: a
    type(double_st) cosh_double_st

    cosh_double_st = cpp_cosh_double_st(a)
  end function cosh_double_st

  elemental function erf_single_st(a)
    type(single_st), intent(in) :: a
    type(single_st) erf_single_st

    erf_single_st = cpp_erf_single_st(a)
  end function erf_single_st

  elemental function erf_double_st(a)
    type(double_st), intent(in) :: a
    type(double_st) erf_double_st

    erf_double_st = cpp_erf_double_st(a)
  end function erf_double_st

  elemental function erfc_single_st(a)
    type(single_st), intent(in) :: a
    type(single_st) erfc_single_st

    erfc_single_st = cpp_erfc_single_st(a)
  end function erfc_single_st

  elemental function erfc_double_st(a)
    type(double_st), intent(in) :: a
    type(double_st) erfc_double_st

    erfc_double_st = cpp_erfc_double_st(a)
  end function erfc_double_st

  elemental function exp_single_st(a)
    type(single_st), intent(in) :: a
    type(single_st) exp_single_st

    exp_single_st = cpp_exp_single_st(a)
  end function exp_single_st

  elemental function exp_double_st(a)
    type(double_st), intent(in) :: a
    type(double_st) exp_double_st

    exp_double_st = cpp_exp_double_st(a)
  end function exp_double_st

  elemental function floor_single_st(a)
    type(single_st), intent(in) :: a
    integer(C_INT) floor_single_st

    floor_single_st = int(cpp_floor_single_st(a), C_INT)
  end function floor_single_st

  elemental function floor_double_st(a)
    type(double_st), intent(in) :: a
    integer(C_INT) floor_double_st

    floor_double_st = int(cpp_floor_double_st(a), C_INT)
  end function floor_double_st

  elemental function log_single_st(a)
    type(single_st), intent(in) :: a
    type(single_st) log_single_st

    log_single_st = cpp_log_single_st(a)
  end function log_single_st

  elemental function log_double_st(a)
    type(double_st), intent(in) :: a
    type(double_st) log_double_st

    log_double_st = cpp_log_double_st(a)
  end function log_double_st

  elemental function log10_single_st(a)
    type(single_st), intent(in) :: a
    type(single_st) log10_single_st

    log10_single_st = cpp_log10_single_st(a)
  end function log10_single_st

  elemental function log10_double_st(a)
    type(double_st), intent(in) :: a
    type(double_st) log10_double_st

    log10_double_st = cpp_log10_double_st(a)
  end function log10_double_st

  elemental function nint_single_st(a)
    type(single_st), intent(in) :: a
    integer(C_INT) nint_single_st

    nint_single_st = int(cpp_nint_single_st(a), C_INT)
  end function nint_single_st

  elemental function nint_double_st(a)
    type(double_st), intent(in) :: a
    integer(C_INT) nint_double_st

    nint_double_st = int(cpp_nint_double_st(a), C_INT)
  end function nint_double_st

  elemental function sin_single_st(a)
    type(single_st), intent(in) :: a
    type(single_st) sin_single_st

    sin_single_st = cpp_sin_single_st(a)
  end function sin_single_st

  elemental function sin_double_st(a)
    type(double_st), intent(in) :: a
    type(double_st) sin_double_st

    sin_double_st = cpp_sin_double_st(a)
  end function sin_double_st

  elemental function sinh_single_st(a)
    type(single_st), intent(in) :: a
    type(single_st) sinh_single_st

    sinh_single_st = cpp_sinh_single_st(a)
  end function sinh_single_st

  elemental function sinh_double_st(a)
    type(double_st), intent(in) :: a
    type(double_st) sinh_double_st

    sinh_double_st = cpp_sinh_double_st(a)
  end function sinh_double_st

  elemental function sqrt_single_st(a)
    type(single_st), intent(in) :: a
    type(single_st) sqrt_single_st

    sqrt_single_st = cpp_sqrt_single_st(a)
  end function sqrt_single_st

  elemental function sqrt_double_st(a)
    type(double_st), intent(in) :: a
    type(double_st) sqrt_double_st

    sqrt_double_st = cpp_sqrt_double_st(a)
  end function sqrt_double_st

  elemental function tan_single_st(a)
    type(single_st), intent(in) :: a
    type(single_st) tan_single_st

    tan_single_st = cpp_tan_single_st(a)
  end function tan_single_st

  elemental function tan_double_st(a)
    type(double_st), intent(in) :: a
    type(double_st) tan_double_st

    tan_double_st = cpp_tan_double_st(a)
  end function tan_double_st

  elemental function tanh_single_st(a)
    type(single_st), intent(in) :: a
    type(single_st) tanh_single_st

    tanh_single_st = cpp_tanh_single_st(a)
  end function tanh_single_st

  elemental function tanh_double_st(a)
    type(double_st), intent(in) :: a
    type(double_st) tanh_double_st

    tanh_double_st = cpp_tanh_double_st(a)
  end function tanh_double_st




  elemental function atan2_single_st_single_st(a, b)
    type(single_st), intent(in) :: a
    type(single_st), intent(in) :: b
    type(single_st) atan2_single_st_single_st

    atan2_single_st_single_st = cpp_atan2_single_st_single_st(a, b)
  end function atan2_single_st_single_st

  elemental function atan2_double_st_double_st(a, b)
    type(double_st), intent(in) :: a
    type(double_st), intent(in) :: b
    type(double_st) atan2_double_st_double_st

    atan2_double_st_double_st = cpp_atan2_double_st_double_st(a, b)
  end function atan2_double_st_double_st

  elemental function max_single_st_single_st(a, b)
    type(single_st), intent(in) :: a
    type(single_st), intent(in) :: b
    type(single_st) max_single_st_single_st

    max_single_st_single_st = cpp_max_single_st_single_st(a, b)
  end function max_single_st_single_st

  elemental function max_double_st_double_st(a, b)
    type(double_st), intent(in) :: a
    type(double_st), intent(in) :: b
    type(double_st) max_double_st_double_st

    max_double_st_double_st = cpp_max_double_st_double_st(a, b)
  end function max_double_st_double_st

  elemental function max_float_single_st(a, b)
    real(C_FLOAT), intent(in) :: a
    type(single_st), intent(in) :: b
    type(single_st) max_float_single_st

    max_float_single_st = cpp_max_float_single_st(a, b)
  end function max_float_single_st

  elemental function max_double_double_st(a, b)
    real(C_DOUBLE), intent(in) :: a
    type(double_st), intent(in) :: b
    type(double_st) max_double_double_st

    max_double_double_st = cpp_max_double_double_st(a, b)
  end function max_double_double_st

  elemental function max_single_st_float(a, b)
    type(single_st), intent(in) :: a
    real(C_FLOAT), intent(in) :: b
    type(single_st) max_single_st_float

    max_single_st_float = cpp_max_single_st_float(a, b)
  end function max_single_st_float

  elemental function max_double_st_double(a, b)
    type(double_st), intent(in) :: a
    real(C_DOUBLE), intent(in) :: b
    type(double_st) max_double_st_double

    max_double_st_double = cpp_max_double_st_double(a, b)
  end function max_double_st_double

  elemental function min_single_st_single_st(a, b)
    type(single_st), intent(in) :: a
    type(single_st), intent(in) :: b
    type(single_st) min_single_st_single_st

    min_single_st_single_st = cpp_min_single_st_single_st(a, b)
  end function min_single_st_single_st

  elemental function min_double_st_double_st(a, b)
    type(double_st), intent(in) :: a
    type(double_st), intent(in) :: b
    type(double_st) min_double_st_double_st

    min_double_st_double_st = cpp_min_double_st_double_st(a, b)
  end function min_double_st_double_st

  elemental function min_float_single_st(a, b)
    real(C_FLOAT), intent(in) :: a
    type(single_st), intent(in) :: b
    type(single_st) min_float_single_st

    min_float_single_st = cpp_min_float_single_st(a, b)
  end function min_float_single_st

  elemental function min_double_double_st(a, b)
    real(C_DOUBLE), intent(in) :: a
    type(double_st), intent(in) :: b
    type(double_st) min_double_double_st

    min_double_double_st = cpp_min_double_double_st(a, b)
  end function min_double_double_st

  elemental function min_single_st_float(a, b)
    type(single_st), intent(in) :: a
    real(C_FLOAT), intent(in) :: b
    type(single_st) min_single_st_float

    min_single_st_float = cpp_min_single_st_float(a, b)
  end function min_single_st_float

  elemental function min_double_st_double(a, b)
    type(double_st), intent(in) :: a
    real(C_DOUBLE), intent(in) :: b
    type(double_st) min_double_st_double

    min_double_st_double = cpp_min_double_st_double(a, b)
  end function min_double_st_double

  elemental function mod_single_st_single_st(a, b)
    type(single_st), intent(in) :: a
    type(single_st), intent(in) :: b
    type(single_st) mod_single_st_single_st

    mod_single_st_single_st = cpp_mod_single_st_single_st(a, b)
  end function mod_single_st_single_st

  elemental function mod_double_st_double_st(a, b)
    type(double_st), intent(in) :: a
    type(double_st), intent(in) :: b
    type(double_st) mod_double_st_double_st

    mod_double_st_double_st = cpp_mod_double_st_double_st(a, b)
  end function mod_double_st_double_st




  elemental function dim_single_st_single_st(a, b)
    type(single_st), intent(in) :: a
    type(single_st), intent(in) :: b
    type(single_st) dim_single_st_single_st

    dim_single_st_single_st = MAX(a-b, 0.)
  end function dim_single_st_single_st

  elemental function dim_double_st_double_st(a, b)
    type(double_st), intent(in) :: a
    type(double_st), intent(in) :: b
    type(double_st) dim_double_st_double_st

    dim_double_st_double_st = MAX(a-b, 0.d0)
  end function dim_double_st_double_st

  elemental function dim_float_single_st(a, b)
    real(C_FLOAT), intent(in) :: a
    type(single_st), intent(in) :: b
    type(single_st) dim_float_single_st

    dim_float_single_st = MAX(a-b, 0.)
  end function dim_float_single_st

  elemental function dim_double_double_st(a, b)
    real(C_DOUBLE), intent(in) :: a
    type(double_st), intent(in) :: b
    type(double_st) dim_double_double_st

    dim_double_double_st = MAX(a-b, 0.d0)
  end function dim_double_double_st

  elemental function dim_single_st_float(a, b)
    type(single_st), intent(in) :: a
    real(C_FLOAT), intent(in) :: b
    type(single_st) dim_single_st_float

    dim_single_st_float = MAX(a-b, 0.)
  end function dim_single_st_float

  elemental function dim_double_st_double(a, b)
    type(double_st), intent(in) :: a
    real(C_DOUBLE), intent(in) :: b
    type(double_st) dim_double_st_double

    dim_double_st_double = MAX(a-b, 0.d0)
  end function dim_double_st_double

  elemental function hypot_single_st_single_st(a, b)
    type(single_st), intent(in) :: a
    type(single_st), intent(in) :: b
    type(single_st) hypot_single_st_single_st

    hypot_single_st_single_st = SQRT( a**2 + b**2 )
  end function hypot_single_st_single_st

  elemental function hypot_double_st_double_st(a, b)
    type(double_st), intent(in) :: a
    type(double_st), intent(in) :: b
    type(double_st) hypot_double_st_double_st

    hypot_double_st_double_st = SQRT( a**2 + b**2 )
  end function hypot_double_st_double_st

  elemental function hypot_float_single_st(a, b)
    real(C_FLOAT), intent(in) :: a
    type(single_st), intent(in) :: b
    type(single_st) hypot_float_single_st

    hypot_float_single_st = SQRT( a**2 + b**2 )
  end function hypot_float_single_st

  elemental function hypot_double_double_st(a, b)
    real(C_DOUBLE), intent(in) :: a
    type(double_st), intent(in) :: b
    type(double_st) hypot_double_double_st

    hypot_double_double_st = SQRT( a**2 + b**2 )
  end function hypot_double_double_st

  elemental function hypot_single_st_float(a, b)
    type(single_st), intent(in) :: a
    real(C_FLOAT), intent(in) :: b
    type(single_st) hypot_single_st_float

    hypot_single_st_float = SQRT( a**2 + b**2 )
  end function hypot_single_st_float

  elemental function hypot_double_st_double(a, b)
    type(double_st), intent(in) :: a
    real(C_DOUBLE), intent(in) :: b
    type(double_st) hypot_double_st_double

    hypot_double_st_double = SQRT( a**2 + b**2 )
  end function hypot_double_st_double

  elemental function sign_single_st_single_st(a, b)
    type(single_st), intent(in) :: a
    type(single_st), intent(in) :: b
    type(single_st) sign_single_st_single_st

    sign_single_st_single_st%x = sign(a%x, b%x) 
    sign_single_st_single_st%y = sign(a%y, b%y) 
    sign_single_st_single_st%z = sign(a%z, b%z) 
    sign_single_st_single_st%accuracy = DIGIT_NOT_COMPUTED
  end function sign_single_st_single_st

  elemental function sign_double_st_double_st(a, b)
    type(double_st), intent(in) :: a
    type(double_st), intent(in) :: b
    type(double_st) sign_double_st_double_st

    sign_double_st_double_st%x = sign(a%x, b%x) 
    sign_double_st_double_st%y = sign(a%y, b%y) 
    sign_double_st_double_st%z = sign(a%z, b%z) 
    sign_double_st_double_st%accuracy = DIGIT_NOT_COMPUTED
  end function sign_double_st_double_st

  elemental function sign_float_single_st(a, b)
    real(C_FLOAT), intent(in) :: a
    type(single_st), intent(in) :: b
    type(single_st) sign_float_single_st

    sign_float_single_st%x = sign(a, b%x)
    sign_float_single_st%y = sign(a, b%y)
    sign_float_single_st%z = sign(a, b%z)
    sign_float_single_st%accuracy = DIGIT_NOT_COMPUTED
  end function sign_float_single_st

  elemental function sign_double_double_st(a, b)
    real(C_DOUBLE), intent(in) :: a
    type(double_st), intent(in) :: b
    type(double_st) sign_double_double_st

    sign_double_double_st%x = sign(a, b%x)
    sign_double_double_st%y = sign(a, b%y)
    sign_double_double_st%z = sign(a, b%z)
    sign_double_double_st%accuracy = DIGIT_NOT_COMPUTED
  end function sign_double_double_st

  elemental function sign_single_st_float(a, b)
    type(single_st), intent(in) :: a
    real(C_FLOAT), intent(in) :: b
    type(single_st) sign_single_st_float

    sign_single_st_float%x = sign(a%x, b)
    sign_single_st_float%y = sign(a%y, b)
    sign_single_st_float%z = sign(a%z, b)
    sign_single_st_float%accuracy = DIGIT_NOT_COMPUTED
  end function sign_single_st_float

  elemental function sign_double_st_double(a, b)
    type(double_st), intent(in) :: a
    real(C_DOUBLE), intent(in) :: b
    type(double_st) sign_double_st_double

    sign_double_st_double%x = sign(a%x, b)
    sign_double_st_double%y = sign(a%y, b)
    sign_double_st_double%z = sign(a%z, b)
    sign_double_st_double%accuracy = DIGIT_NOT_COMPUTED
  end function sign_double_st_double




end module cadna_math

