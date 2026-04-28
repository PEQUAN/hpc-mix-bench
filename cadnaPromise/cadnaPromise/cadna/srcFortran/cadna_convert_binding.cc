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

// Module de conversion




#include <iostream>
#include "cadna.h"

using namespace std;

extern "C"
{



            

  int cpp_int_single_st( float_st &a )
  {
    return (int)a;
  }

  int cpp_int_double_st( double_st &a )
  {
    return (int)a;
  }

  float cpp_real_single_st( float_st &a )
  {
    return (float)a;
  }

  float cpp_real_double_st( double_st &a )
  {
    return (float)a;
  }

  double cpp_dble_single_st( float_st &a )
  {
    return (double)a;
  }

  double cpp_dble_double_st( double_st &a )
  {
    return (double)a;
  }




}

