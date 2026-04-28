
/*****************************************************************************************
!  COPYRIGHT
!	2010 - Sethy Montan, sethy.montan@gmail.com
!       2018 -  F. Jezequel, J.-L. Lamotte
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

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#define CADNA_MPI_TYPE_AND_OP

#include "cadna_mpi.h"
#include "cadna.h"
#include "cadna-error.h"
#include "unistd.h"
using namespace std;


//MPI_Datatype  MPI_DATA;
//MPI_Datatype  MPI_SINGLE_ST; //FJ
MPI_Datatype  MPI_FLOAT_ST; //FJ
MPI_Datatype  MPI_DOUBLE_ST; 

/* reduction operators */
MPI_Op MPI_CADNA_SUM_SP,  MPI_CADNA_SUM_DP;
MPI_Op MPI_CADNA_PROD_SP, MPI_CADNA_PROD_DP;
MPI_Op MPI_CADNA_MAX_SP,  MPI_CADNA_MAX_DP;
MPI_Op MPI_CADNA_MIN_SP,  MPI_CADNA_MIN_DP;  

MPI_Comm cadna_comm;

// Ajouts IDRIS
MPI_Fint MPIF_SINGLE_ST, MPIF_DOUBLE_ST;

MPI_Fint MPIF_CADNA_SUM_SP,  MPIF_CADNA_SUM_DP;
MPI_Fint MPIF_CADNA_PROD_SP, MPIF_CADNA_PROD_DP;
MPI_Fint MPIF_CADNA_MAX_SP,  MPIF_CADNA_MAX_DP;
MPI_Fint MPIF_CADNA_MIN_SP,  MPIF_CADNA_MIN_DP;

MPI_Fint f_cadna_comm;
// Fin Ajouts IDRIS



/****** cadna_mpi/cadna_mpi_init
!  NAME
!    cadna_mpi_int
!
!  FUNCTION
!    Subroutine to create 2 mpi derived types
!    	 MPI_FLOAT_ST for float_st
!    	 MPI_DOUBLE_ST for double_st
!    and the MPI_CADNA reduction operators
!        MPI_CADNA_SUM_SP,MPI_CADNA_SUM_DP
!        MPI_CADNA_PROD_SP,MPI_CADNA_PROD_DP
!        MPI_CADNA_MAX_SP,MPI_CADNA_MAX_DP
!        MPI_CADNA_MIN_SP,MPI_CADNA_MIN_DP
!    and init cadna
!
!*****************************************/
void  cadna_mpi_init(const int& rank,int max_nb_instability)
{
  //cout<<"\nCADNA MPI INIT FOR PROC : "<<rank<<endl ;

// MPI_Comm_split(MPI_COMM_WORLD, cadna_rank, cadna_rank, &cadna_comm);
 MPI_Comm_dup(MPI_COMM_WORLD, &cadna_comm);


//first init CADNA 
cadna_init(max_nb_instability);

int nb_block = 2 ; 
int block_lengths[nb_block];
MPI_Datatype typelist[nb_block]; 
MPI_Aint displacements[nb_block]; 
MPI_Aint lb , extent ; 

/**** MPI_FLOAT_ST ***/////  
block_lengths[0]= 3 ;
typelist[0]= MPI_FLOAT ; 
displacements[0] = 0 ; 
block_lengths[1]= 1 ;
typelist[1]= MPI_INT ; 
MPI_Type_get_extent(MPI_FLOAT,&lb,&extent) ;
	displacements[1] = 3 * extent ; 

	//	MPI_Type_create_struct(nb_block,block_lengths,displacements,typelist,&MPI_SINGLE_ST); //
	//      MPI_Type_commit(&MPI_SINGLE_ST); //FJ
	MPI_Type_create_struct(nb_block,block_lengths,displacements,typelist,&MPI_FLOAT_ST);//FJ
	MPI_Type_commit(&MPI_FLOAT_ST);//FJ
	
	/**** MPI_DOUBLE_ST ***/////  
	block_lengths[0]= 3 ;
	typelist[0]= MPI_DOUBLE ; 
	displacements[0] = 0 ; 
	block_lengths[1]= 2 ;
	typelist[1]= MPI_INT ; 
	MPI_Type_get_extent(MPI_DOUBLE,&lb,&extent) ;
	displacements[1] = 3 * extent ; 
	
	
	MPI_Type_create_struct(nb_block,block_lengths,displacements,typelist,&MPI_DOUBLE_ST);
	MPI_Type_commit(&MPI_DOUBLE_ST);
	
	/*
	//  ** test type dérivé de type dérivé **  
	block_lengths[0]= 3 ;
	typelist[0]= MPI_DOUBLE_ST ; 
	displacements[0] = 0 ; 
	block_lengths[1]= 1 ;
	typelist[1]= MPI_INT ; 
	MPI_Type_get_extent(MPI_DOUBLE_ST,&lb,&extent) ;
	cout<<" MPI_DOUBLE_ST "<<"lb = "<<lb<<" extent = "<<extent<<endl;
	displacements[1] = 3 * extent ; 

	MPI_Type_create_struct(nb_block,block_lengths,displacements,typelist,&MPI_DATA);
	MPI_Type_commit(&MPI_DATA);
	
	*/
	
	/******* REDUCTION OPERATORS ****//////////////
	int commute = false ;
	//SUM
	MPI_Op_create((MPI_User_function *)SP_SUM,commute,&MPI_CADNA_SUM_SP);
	MPI_Op_create((MPI_User_function *)DP_SUM,commute,&MPI_CADNA_SUM_DP);
        
	//PROD
	MPI_Op_create((MPI_User_function *)SP_PROD,commute,&MPI_CADNA_PROD_SP);
	MPI_Op_create((MPI_User_function *)DP_PROD,commute,&MPI_CADNA_PROD_DP);
	//MIN
	MPI_Op_create((MPI_User_function *)SP_MIN,commute,&MPI_CADNA_MIN_SP);
	MPI_Op_create((MPI_User_function *)DP_MIN,commute,&MPI_CADNA_MIN_DP);
	//MAX
	MPI_Op_create((MPI_User_function *)SP_MAX,commute,&MPI_CADNA_MAX_SP);
	MPI_Op_create((MPI_User_function *)DP_MAX,commute,&MPI_CADNA_MAX_DP);

        // Ajouts IDRIS
        f_cadna_comm       = MPI_Comm_c2f(cadna_comm);

	//MPIF_SINGLE_ST     = MPI_Type_c2f(MPI_SINGLE_ST); //FJ
        MPIF_SINGLE_ST     = MPI_Type_c2f(MPI_FLOAT_ST); //FJ
        MPIF_DOUBLE_ST     = MPI_Type_c2f(MPI_DOUBLE_ST);

        MPIF_CADNA_SUM_SP  = MPI_Op_c2f(MPI_CADNA_SUM_SP);
        MPIF_CADNA_SUM_DP  = MPI_Op_c2f(MPI_CADNA_SUM_DP);
        MPIF_CADNA_PROD_SP = MPI_Op_c2f(MPI_CADNA_PROD_SP);
        MPIF_CADNA_PROD_DP = MPI_Op_c2f(MPI_CADNA_PROD_DP);
        MPIF_CADNA_MIN_SP  = MPI_Op_c2f(MPI_CADNA_MIN_SP);
        MPIF_CADNA_MIN_DP  = MPI_Op_c2f(MPI_CADNA_MIN_DP);
        MPIF_CADNA_MAX_SP  = MPI_Op_c2f(MPI_CADNA_MAX_SP);
        MPIF_CADNA_MAX_DP  = MPI_Op_c2f(MPI_CADNA_MAX_DP);
        // Fin Ajouts IDRIS

}

/****** cadna_mpi/cadna_mpi_end **********************
!  NAME
!    cadna_mpi_end
!
!  FUNCTION
!    Subroutine to free 
!	the 2 mpi derived types 
!	THE MPI_CADNA REDUCTION OPERATORS 
!   and end cadna

!
******************************************************/

static void cadna_message_instability_(unsigned long count, unsigned int change,  string s)
{
  if (change !=0) cout << "Instability detection has been deactivated during the run.  " << count << "   " << s << endl; 
  else if (count !=0 )  cout << count << "   " << s << endl; 
}

static void cadna_message__(int proc, unsigned long tabcount[], unsigned int tabchange[])
{
  unsigned long nbcount=0;
  unsigned int nbchange=0;

  cout << "Proc " << proc <<"     " ;
      
  for (int i=0;i<CADNA_INSTABILITIES_NUMBER; i++){
    if  (tabcount[i] != 0) nbcount ++;
    if  (tabchange[i] != 0) nbchange ++;
  }

  if (nbcount==0 && nbchange==0) {
    cout  << "no instability" << endl ;
    return;
  }

  cadna_message_instability_(tabcount[0], tabchange[0], _cadna_div_err_msg);
  cadna_message_instability_(tabcount[1], tabchange[1], _cadna_mul_err_msg);
  cadna_message_instability_(tabcount[2], tabchange[2], _cadna_power_err_msg);
  cadna_message_instability_(tabcount[3], tabchange[3], _cadna_branching_err_msg);
  cadna_message_instability_(tabcount[4], tabchange[4], _cadna_cancel_err_msg);
  cadna_message_instability_(tabcount[5], tabchange[5], _cadna_intrinsic_err_msg);
  cadna_message_instability_(tabcount[6], tabchange[6], _cadna_math_err_msg);
}	   


void  cadna_mpi_end()
{
  
  // FREE DATATYPE 
  //	MPI_Type_free(&MPI_SINGLE_ST);
  //	MPI_Type_free(&MPI_DOUBLE_ST);
  // FREE OPERATOR 
  //	MPI_Op_free(&MPI_CADNA_SUM_SP);
  //	MPI_Op_free(&MPI_CADNA_SUM_DP);
  //	MPI_Op_free(&MPI_CADNA_PROD_SP);
  //	MPI_Op_free(&MPI_CADNA_PROD_DP);
  //	MPI_Op_free(&MPI_CADNA_MIN_SP);
  //	MPI_Op_free(&MPI_CADNA_MIN_DP);
  //	MPI_Op_free(&MPI_CADNA_MAX_SP);
  //	MPI_Op_free(&MPI_CADNA_MAX_DP);
  //MPI_Op_free(&MPI_CADNA_MAX);
  //Now ending CADNA 
  // all the instabilities are sent to the master (0) with
  // a specifc MPI communicator.
  
  int cadna_rank=999, cadna_size=888;
  
  MPI_Comm_rank(cadna_comm, &cadna_rank);
  MPI_Comm_size(cadna_comm, &cadna_size);
  
   unsigned long  tab_count[7]={
    _cadna_div_count,  _cadna_mul_count,  
    _cadna_power_count,  _cadna_branching_count, 
    _cadna_cancel_count, _cadna_intrinsic_count, 
    _cadna_math_count}; 

  unsigned int  tab_change[7]={
    _cadna_div_change,  _cadna_mul_change,  
    _cadna_power_change,  _cadna_branching_change, 
    _cadna_cancel_change, _cadna_intrinsic_change, 
    _cadna_math_change}; 
  
  unsigned long *tab_global_count=NULL;
  unsigned int  *tab_global_change=NULL;
  
  if  (cadna_rank==0) 
    if (((tab_global_count=(unsigned long*)malloc(sizeof(unsigned long)*7*(cadna_size+1)))==NULL)||
	((tab_global_change=(unsigned int*)malloc(sizeof(unsigned int)*7*(cadna_size+1)))==NULL))
	cerr << "cadna_end : allocation problem\n" << endl;

  MPI_Gather(tab_count,7,MPI_UNSIGNED_LONG,
	     tab_global_count,7,MPI_UNSIGNED_LONG, 0, cadna_comm);
  MPI_Gather(tab_change,7,MPI_UNSIGNED,
	     tab_global_change,7,MPI_UNSIGNED, 0, cadna_comm);

  if (cadna_rank==0) {
    cout << "----------------------------------------------------------------" << endl;
    cout << PACKAGE_STRING << " software --- MPI -- Proc size " << cadna_size << endl;
    

    // affichage pour le proc 0
	cadna_message__(0, &tab_global_count[0], &tab_global_change[0]);
    
    for (int i=1; i<cadna_size;i++) {

       // affichage pour le proc i <> 0
	cadna_message__(i, &tab_global_count[i*7], &tab_global_change[i*7]);
	// Only the proc 0 runs this part
	// CADNA instabilities have just been written on the screen
	// It is possible to use the  _cadna_div_count, ... 
	// to reduce sum of all the instabilities
	// on somme pour les proc i<> 0 (i allant de 1 à cadna_size -1
	_cadna_div_count+=tab_global_count[i*7+0]; 
	_cadna_mul_count+=tab_global_count[i*7+1]; 
	_cadna_power_count+=tab_global_count[i*7+2]; 
	_cadna_branching_count+=tab_global_count[i*7+3]; 
	_cadna_cancel_count+=tab_global_count[i*7+4]; 
	_cadna_intrinsic_count+=tab_global_count[i*7+5]; 
	_cadna_math_count+=tab_global_count[i*7+6];
	
	_cadna_div_change|=tab_global_change[i*7+0]; 
	_cadna_mul_change|=tab_global_change[i*7+1]; 
	_cadna_power_change|=tab_global_change[i*7+2]; 
	_cadna_branching_change|=tab_global_change[i*7+3]; 
	_cadna_cancel_change|=tab_global_change[i*7+4]; 
	_cadna_intrinsic_change|=tab_global_change[i*7+5]; 
	_cadna_math_change|=tab_global_count[i*7+6];
    }
    cout << "----------------------------------------------------------------" << endl;
    cout << " All processes : "<< endl;
    cadna_end();
  }
}
	/******* REDUCTION OPERATORS Functions ****//////////////
//SUM
//Single PRec  
void SP_SUM(float_st* in,float_st* inout,int* len ,MPI_Datatype* dptr) 
{
	for(int i =0; i<*len; i++) inout[i]+= in[i] ;
} 
//Double Prec  
void DP_SUM(double_st* in,double_st* inout,int* len ,MPI_Datatype* dptr) 
{
	for(int i =0; i<*len; i++) inout[i]+= in[i] ;
}   

//Product
//Single PRec 
void SP_PROD(float_st* in,float_st* inout,int* len ,MPI_Datatype* dptr) 
{
	for(int i =0; i<*len; i++) inout[i]*= in[i] ;
} 
//Double Prec   
void DP_PROD(double_st* in,double_st* inout,int* len ,MPI_Datatype* dptr) 
{
	for(int i =0; i<*len; i++) inout[i]*= in[i] ;
}   

//MIN 
//Single Prec
void SP_MIN(float_st* in,float_st* inout,int* len ,MPI_Datatype* dptr) 
{
	for(int i =0; i<*len; i++) inout[i] = fminf(inout[i],in[i] );
} 
//Double Prec  
void DP_MIN(double_st* in,double_st* inout,int* len ,MPI_Datatype* dptr) 
{
	for(int i =0; i<*len; i++) inout[i] = fmin(inout[i],in[i] );	
} 

//MAX  
//Single Precision
void SP_MAX(float_st* in,float_st* inout,int* len ,MPI_Datatype* dptr) 
{
	for(int i =0; i<*len; i++) inout[i] = fmaxf(inout[i],in[i] );	
} 
//Double Precision  
void DP_MAX(double_st* in,double_st* inout,int* len ,MPI_Datatype* dptr) 
{
	for(int i =0; i<*len; i++) inout[i] = fmax(inout[i],in[i] );
}   
/*
void DP_MAX(float_st* in,float_st* inout,int* len ,MPI_Datatype* dptr) 
{
	for(int i =0; i<*len; i++) inout[i] = fmaxf(inout[i],in[i] );	
}
   */
