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
#include <iostream>
#include "cadna.h"

using namespace std;

extern "C"
{
  int cpp_nb_significant_digit_single_st( float_st &a )
  {
    return a.nb_significant_digit();
  }
  int cpp_nb_significant_digit_double_st( double_st &a )
  {
    return a.nb_significant_digit();
  }

  void cpp_display_single_st( float_st &a )
  {
    a.display();
  }
  void cpp_display_double_st( double_st &a )
  {
    a.display();
  }

  void cpp_display_char_single_st( const char *s, float_st &a )
  {
    a.display(s);
  }
  void cpp_display_char_double_st( const char *s, double_st &a )
  {
    a.display(s);
  }

  void cpp_str_single_st( float_st &a, char *s )
  {
    strcpy(s, strp(a));
  }
  void cpp_str_double_st( double_st &a, char *s )
  {
    strcpy(s, strp(a));
  }

  void cpp_data_st_single_st( float_st &a )
  {
    a.data_st();
  }
  void cpp_data_st_double_st( double_st &a )
  {
    a.data_st();
  }

  void cpp_data_st_single_st_double_int( float_st &a, const double &e, const int &error_type )
  {
    a.data_st(e, error_type);
  }
  void cpp_data_st_double_st_double_int( double_st &a, const double &e, const int &error_type )
  {
    a.data_st(e, error_type);
  }
}
