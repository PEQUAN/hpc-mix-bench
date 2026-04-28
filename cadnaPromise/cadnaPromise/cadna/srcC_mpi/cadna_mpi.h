/*****************************************************************************************
!  COPYRIGHT
!      2010 - Sethy Montan, sethy.montan@gmail.com
!      2018 -  F. Jezequel, J.-L. Lamotte
!	 
!    This file is part of CADNA.
!
!    CADNA is free software: you can redistribute it and/or modify
!    it under the terms of the GNU Lesser General Public License as published by
!    the Free Software Foundation, either version 3 of the License, or
!    (at your option) any later version.
!
!    CADNA is distributed in the hope that it will be useful,
!    but WITHOUT ANY WARRANTY; without even the implied warranty of
!    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
!    GNU Lesser General Public License for more details.
!
!    You should have received a copy of the GNU Lesser General Public License
!   along with CADNA.  If not, see <http://www.gnu.org/licenses/>.
!
!  USAGE
!    Can be used by inserting "#include "cadna_mpi.h"" at the beginning of each file
!    which uses the CADNA Library  & MPI
!
!  DESCRIPTION
!    Allows to use the types and functions defined by the CADNA library with MPI routines  
!
!*****************************************************************************************/

#ifndef __CADNA_MPI__
#define __CADNA_MPI__

#include <cadna.h>
#include <mpi.h>

/* new types */ 
//MPI_Datatype
extern MPI_Datatype  MPI_FLOAT_ST; 
extern MPI_Datatype  MPI_DOUBLE_ST;
//extern MPI_Datatype  MPI_DATA;

/* Reduction operators */ 
extern MPI_Op MPI_CADNA_SUM_SP,MPI_CADNA_SUM_DP;
extern MPI_Op MPI_CADNA_PROD_SP,MPI_CADNA_PROD_DP;
extern MPI_Op MPI_CADNA_MAX_SP,MPI_CADNA_MAX_DP;
extern MPI_Op MPI_CADNA_MIN_SP,MPI_CADNA_MIN_DP;


/* initialization and end functions */ 

void  cadna_mpi_init(const int& rank,int max_nb_instability);
void  cadna_mpi_end();

/* Functions for reduction operators*/ 
//SUM 
void SP_SUM(float_st*,float_st*,int*,MPI_Datatype*);
void DP_SUM(double_st*,double_st*,int* len ,MPI_Datatype*);

//Product 
void SP_PROD(float_st*,float_st*,int*,MPI_Datatype*);
void DP_PROD(double_st*,double_st*,int* len ,MPI_Datatype*);

//MIN 
void SP_MIN(float_st*,float_st*,int*,MPI_Datatype*);
void DP_MIN(double_st* in,double_st* inout,int* len ,MPI_Datatype* dptr);

//MAX  
void SP_MAX(float_st*,float_st*,int*,MPI_Datatype*);
//void DP_MAX(float_st*,float_st*,int*,MPI_Datatype*);
void DP_MAX(double_st* in,double_st* inout,int* len ,MPI_Datatype* dptr);


extern "C" void CADNA_MAX(void* in,void* inout,int* len ,MPI_Datatype* dptr);
#endif
