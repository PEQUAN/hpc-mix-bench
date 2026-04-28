/*-----------------------------------------------------------------------
Copyright 2017 R. CARPENTIER, P. CORDE, F. JEZEQUEL, J.-L. LAMOTTE

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

// Module de surcharge de l'opérateur ">="




#include <iostream>
#include <stdbool.h>
#include "cadna.h"

using namespace std;

extern "C"
{



            

  bool cpp_ge_double_st_double_st( double_st &a,  double_st &b )
  {
    return a>=b;
  }

  bool cpp_ge_double_st_single_st( double_st &a,  float_st &b )
  {
    return a>=b;
  }

  bool cpp_ge_double_st_double( double_st &a,  double &b )
  {
    return a>=b;
  }

  bool cpp_ge_double_st_float( double_st &a,  float &b) 
  {
    return a>=b;
  }

  bool cpp_ge_double_st_long_long( double_st &a,  long long &b )
  {
    return a>=b;
  }

//FUNCTIONS_OP(ge, double_st, long)
  bool cpp_ge_double_st_int( double_st &a,  int &b )
  {
    return a>=b;
  }

  bool cpp_ge_double_st_short( double_st &a,  short &b ) 
  {
    return a>=b;
  }


  bool cpp_ge_single_st_double_st( float_st &a,  double_st &b )
  {
    return a>=b;
  }

  bool cpp_ge_single_st_single_st( float_st &a,  float_st &b )
  {
    return a>=b;
  }

  bool cpp_ge_single_st_double( float_st &a,  double &b )
  {
    return a>=b;
  }

  bool cpp_ge_single_st_float( float_st &a,  float &b) 
  {
    return a>=b;
  }

  bool cpp_ge_single_st_long_long( float_st &a,  long long &b )
  {
    return a>=b;
  }

//FUNCTIONS_OP(ge, single_st, long)
  bool cpp_ge_single_st_int( float_st &a,  int &b )
  {
    return a>=b;
  }

  bool cpp_ge_single_st_short( float_st &a,  short &b ) 
  {
    return a>=b;
  }


  bool cpp_ge_double_double_st( double &a,  double_st &b )
  {
    return a>=b;
  }

  bool cpp_ge_double_single_st( double &a,  float_st &b )
  {
    return a>=b;
  }


  bool cpp_ge_float_double_st( float &a,  double_st &b )
  {
    return a>=b;
  }

  bool cpp_ge_float_single_st( float &a,  float_st &b )
  {
    return a>=b;
  }


  bool cpp_ge_long_long_double_st( long long &a,  double_st &b )
  {
    return a>=b;
  }

  bool cpp_ge_long_long_single_st( long long &a,  float_st &b )
  {
    return a>=b;
  }


//FUNCTIONS_OP(ge, long, double_st)
//FUNCTIONS_OP(ge, long, single_st)

  bool cpp_ge_int_double_st( int &a,  double_st &b )
  {
    return a>=b;
  }

  bool cpp_ge_int_single_st( int &a,  float_st &b )
  {
    return a>=b;
  }


  bool cpp_ge_short_double_st( short &a,   double_st &b )
  {
    return a>=b;
  }

  bool cpp_ge_short_single_st( short &a,   float_st &b )
  {
    return a>=b;
  }




}

