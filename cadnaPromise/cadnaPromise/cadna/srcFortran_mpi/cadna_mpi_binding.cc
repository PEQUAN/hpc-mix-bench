/*-----------------------------------------------------------------------
Copyright 2019 P. CORDE, J.-L. LAMOTTE

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

// Module d'interfaçage MPI pour Fortran.




//#include <iostream>
#include "cadna_mpi.h"

extern "C"
{
  void cpp_cadna_mpi_init(const int& rank, int max_nb_instability)
  {
    cadna_mpi_init(rank, max_nb_instability);

    return;
  }

  void cpp_cadna_mpi_end()
  {
    cadna_mpi_end();

    return;
  }
}

