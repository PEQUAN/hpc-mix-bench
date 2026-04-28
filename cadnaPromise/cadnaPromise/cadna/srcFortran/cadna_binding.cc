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
  void cpp_cadna_init_int( int max_nb_instability )
  {
    cadna_init(max_nb_instability);
  }
  
  void cpp_cadna_init_int_int( int          max_nb_instability,
                               unsigned int tag)
  {
    cadna_init(max_nb_instability, tag);
  }
  
  void cpp_cadna_init_int_int_int_int( int          max_nb_instability,
                                       unsigned int tag,
                                       unsigned int seed,
                                       unsigned int cancellation)
  {
    cadna_init(max_nb_instability, tag, seed, cancellation );
  }
  
  void cpp_cadna_end()
  {
    cadna_end();
  }

  void cpp_cadna_enable( unsigned int tag )
  {
    cadna_enable(tag);
  }

  void cpp_cadna_disable( unsigned int tag )
  {
    cadna_disable(tag);
  }
}
