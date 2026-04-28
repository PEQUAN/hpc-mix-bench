/*-----------------------------------------------------------------------
Copyright 2017 R. CARPENTIER, P. CORDE, F. JEZEQUEL, J.-L. LAMOTTE
Modifications:
2019 F. JEZEQUEL

This file is part of CADNA.

    CADNA is free software: you can redistribute it and/or modify it
    under the terms of the GNU Lesser General Public License as
    published by the Free Software Foundation, either version 3 of the
    License, or (at your option) any later version.

    CADNA is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
    or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General
    Public License for more details.

    You should have received a copy of the GNU Lesser General Public
    License along with CADNA. If not, see
    <http://www.gnu.org/licenses/>.
-----------------------------------------------------------------------*/

// Module de surcharge des fonctions mathématiques




#include <iostream>
#include "cadna.h"

using namespace std;

extern "C"
{




  float_st cpp_abs_single_st( float_st &a )
  {
     return abs(a);
  }

  double_st cpp_abs_double_st( double_st &a )
  {
     return abs(a);
  }

  float_st cpp_acos_single_st( float_st &a )
  {
     return acos(a);
  }

  double_st cpp_acos_double_st( double_st &a )
  {
     return acos(a);
  }

  float_st cpp_acosh_single_st( float_st &a )
  {
     return acosh(a);
  }

  double_st cpp_acosh_double_st( double_st &a )
  {
     return acosh(a);
  }

  float_st cpp_aint_single_st( float_st &a )
  {
     return trunc(a);
  }

  double_st cpp_aint_double_st( double_st &a )
  {
     return trunc(a);
  }

//FUNCTION_UNAIRE(anint, round, single_st, float_st)
//FUNCTION_UNAIRE(anint, round, double_st, double_st)
  float_st cpp_asin_single_st( float_st &a )
  {
     return asin(a);
  }

  double_st cpp_asin_double_st( double_st &a )
  {
     return asin(a);
  }

  float_st cpp_asinh_single_st( float_st &a )
  {
     return asinh(a);
  }

  double_st cpp_asinh_double_st( double_st &a )
  {
     return asinh(a);
  }

  float_st cpp_atan_single_st( float_st &a )
  {
     return atan(a);
  }

  double_st cpp_atan_double_st( double_st &a )
  {
     return atan(a);
  }

  float_st cpp_atanh_single_st( float_st &a )
  {
     return atanh(a);
  }

  double_st cpp_atanh_double_st( double_st &a )
  {
     return atanh(a);
  }

  float_st cpp_cos_single_st( float_st &a )
  {
     return cos(a);
  }

  double_st cpp_cos_double_st( double_st &a )
  {
     return cos(a);
  }

  float_st cpp_cosh_single_st( float_st &a )
  {
     return cosh(a);
  }

  double_st cpp_cosh_double_st( double_st &a )
  {
     return cosh(a);
  }

  double_st cpp_erf_double_st( double_st &a )
  {
     return erf(a);
  }

  float_st cpp_erf_single_st( float_st &a )
  {
     return erf(a);
  }

  double_st cpp_erfc_double_st( double_st &a )
  {
     return erfc(a);
  }

  float_st cpp_erfc_single_st( float_st &a )
  {
     return erfc(a);
  }

  double_st cpp_exp_double_st( double_st &a )
  {
     return exp(a);
  }

  float_st cpp_exp_single_st( float_st &a )
  {
     return exp(a);
  }

//FUNCTION_UNAIRE(exponent, frexp, double_st, double_st)
//FUNCTION_UNAIRE(exponent, frexp, single_st, float_st)
//FUNCTION_UNAIRE(fraction, frexp, single_st, float_st)
//FUNCTION_UNAIRE(fraction, frexp, double_st, double_st)
//FUNCTION_UNAIRE(int, trunc, single_st, float_st)
//FUNCTION_UNAIRE(int, trunc, double_st, double_st)
  double_st cpp_log_double_st( double_st &a )
  {
     return log(a);
  }

  float_st cpp_log_single_st( float_st &a )
  {
     return log(a);
  }

//FUNCTION_UNAIRE(log_gamma, lgamma, single_st, float_st)
//FUNCTION_UNAIRE(log_gamma, lgamma, double_st, double_st)
  double_st cpp_log10_double_st( double_st &a )
  {
     return log10(a);
  }

  float_st cpp_log10_single_st( float_st &a )
  {
     return log10(a);
  }

  float_st cpp_sin_single_st( float_st &a )
  {
     return sin(a);
  }

  double_st cpp_sin_double_st( double_st &a )
  {
     return sin(a);
  }

  float_st cpp_sinh_single_st( float_st &a )
  {
     return sinh(a);
  }

  double_st cpp_sinh_double_st( double_st &a )
  {
     return sinh(a);
  }

  float_st cpp_sqrt_single_st( float_st &a )
  {
     return sqrt(a);
  }

  double_st cpp_sqrt_double_st( double_st &a )
  {
     return sqrt(a);
  }

  float_st cpp_tan_single_st( float_st &a )
  {
     return tan(a);
  }

  double_st cpp_tan_double_st( double_st &a )
  {
     return tan(a);
  }

  float_st cpp_tanh_single_st( float_st &a )
  {
     return tanh(a);
  }

  double_st cpp_tanh_double_st( double_st &a )
  {
     return tanh(a);
  }





  int cpp_floor_single_st( float_st &a )
  {
     return (int)floor(a);
  }

  int cpp_floor_double_st( double_st &a )
  {
     return (int)floor(a);
  }

  int cpp_ceiling_single_st( float_st &a )
  {
     return (int)ceil(a);
  }

  int cpp_ceiling_double_st( double_st &a )
  {
     return (int)ceil(a);
  }

  int cpp_nint_single_st( float_st &a )
  {
     return (int)nearbyint(a);
  }

  int cpp_nint_double_st( double_st &a )
  {
     return (int)nearbyint(a);
  }



            

  float_st cpp_atan2_single_st_single_st( float_st &a,  float_st &b )
  {
     return atan2(a, b);
  }

  double_st cpp_atan2_double_st_double_st( double_st &a,  double_st &b )
  {
     return atan2(a, b);
  }

  double_st cpp_max_double_st_double_st( double_st &a,  double_st &b )
  {
     return fmax(a, b);
  }

  double_st cpp_max_double_st_double( double_st &a,  double &b )
  {
     return fmax(a, b);
  }

  double_st cpp_max_double_double_st( double &a,  double_st &b )
  {
     return fmax(a, b);
  }

  float_st cpp_max_single_st_single_st( float_st &a,  float_st &b )
  {
     return fmax(a, b);
  }

  float_st cpp_max_single_st_float( float_st &a,  float &b) 
  {
     return fmax(a, b);
  }

  float_st cpp_max_float_single_st( float &a,  float_st &b )
  {
     return fmax(a, b);
  }

//FUNCTION_BINAIRE(max, fmax, double_st, float, double_st)
//FUNCTION_BINAIRE(max, fmax, float, double_st, double_st)
//FUNCTION_BINAIRE(max, fmax, single_st, double, float_st)
//FUNCTION_BINAIRE(max, fmax, double, single_st, float_st)
  double_st cpp_min_double_st_double_st( double_st &a,  double_st &b )
  {
     return fmin(a, b);
  }

  double_st cpp_min_double_st_double( double_st &a,  double &b )
  {
     return fmin(a, b);
  }

  double_st cpp_min_double_double_st( double &a,  double_st &b )
  {
     return fmin(a, b);
  }

  float_st cpp_min_single_st_single_st( float_st &a,  float_st &b )
  {
     return fmin(a, b);
  }

  float_st cpp_min_single_st_float( float_st &a,  float &b) 
  {
     return fmin(a, b);
  }

  float_st cpp_min_float_single_st( float &a,  float_st &b )
  {
     return fmin(a, b);
  }

//FUNCTION_BINAIRE(min, fmin, double_st, float, double_st)
//FUNCTION_BINAIRE(min, fmin, float, double_st, double_st)
//FUNCTION_BINAIRE(min, fmin, single_st, double, float_st)
//FUNCTION_BINAIRE(min, fmin, double, single_st, float_st)
// fmod n'existe que pour des types identiques dans la bibliothèque cadnaC
  double_st cpp_mod_double_st_double_st( double_st &a,  double_st &b )
  {
     return fmod(a, b);
  }

//FUNCTION_BINAIRE(mod, fmod, double_st, double, double_st)
//FUNCTION_BINAIRE(mod, fmod, double, double_st, double_st)
  float_st cpp_mod_single_st_single_st( float_st &a,  float_st &b )
  {
     return fmod(a, b);
  }

//FUNCTION_BINAIRE(mod, fmod, single_st, float, float_st)
//FUNCTION_BINAIRE(mod, fmod, float, single_st, float_st)
//FUNCTION_BINAIRE(mod, fmod, double_st, float, double_st)
//FUNCTION_BINAIRE(mod, fmod, float, double_st, double_st)
//FUNCTION_BINAIRE(mod, fmod, single_st, double, float_st)
//FUNCTION_BINAIRE(mod, fmod, double, single_st, float_st)
//FUNCTION_BINAIRE(modulo, remainder, double_st, double_st, double_st)
//FUNCTION_BINAIRE(modulo, remainder, double_st, double, double_st)
//FUNCTION_BINAIRE(modulo, remainder, double, double_st, double_st)
//FUNCTION_BINAIRE(modulo, remainder, single_st, single_st, float_st)
//FUNCTION_BINAIRE(modulo, remainder, single_st, float, float_st)
//FUNCTION_BINAIRE(modulo, remainder, float, single_st, float_st)
//FUNCTION_BINAIRE(modulo, remainder, double_st, float, double_st)
//FUNCTION_BINAIRE(modulo, remainder, float, double_st, double_st)
//FUNCTION_BINAIRE(modulo, remainder, single_st, double, float_st)
//FUNCTION_BINAIRE(modulo, remainder, double, single_st, float_st)



}

