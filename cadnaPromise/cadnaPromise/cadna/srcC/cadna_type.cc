// Copyright 2015-2020 J.-M. Chesneaux, P. Eberhart, F. Jezequel, J.-L. Lamotte, S. Hoseininasab

// This file is part of CADNA.

// CADNA is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.

// CADNA is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Lesser General Public License for more details.

// You should have received a copy of the GNU Lesser General Public License
// along with CADNA.  If not, see <http://www.gnu.org/licenses/>.

#include <iostream>
//#include <stdlib.h>
#include "cadna.h"
#include "cadna-error.h"
#ifdef CADNA_HALF_EMULATION
#include "half.hpp"
using half_float::half;
#endif
#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

// #ifdef _OPENMP
//#include <omp.h>
// #endif //_OPENMP
using namespace std;

//=====================================================
//   "CADNA_INIT"
//=====================================================

//  global variables


// INSTABILITIES MANAGEMENT

// The RANDOM BIT GENERATOR

#if defined(_OPENMP)
#pragma omp threadprivate(_cadna_random)
#pragma omp threadprivate(_cadna_random_counter)
#pragma omp threadprivate(_cadna_recurrence)
#endif


unsigned int _cadna_random;
unsigned int _cadna_random1;
int _cadna_random_counter;
unsigned int _cadna_recurrence;

using namespace std;
double max_value = 65504;

int _cadna_max_instability, _cadna_instability_detected;
int nb_stability_count = 0;

unsigned long _cadna_div_count;
unsigned long _cadna_mul_count;  
unsigned long _cadna_power_count;  
unsigned long _cadna_math_count; 
unsigned long _cadna_intrinsic_count;  
unsigned long _cadna_cancel_count;  
unsigned long _cadna_branching_count; 
#ifdef CADNA_HALF_EMULATION
unsigned long _cadna_half_overflow_count; 
unsigned long _cadna_half_underflow_count; 
#endif
#ifdef CADNA_HALF_NATIVE
unsigned long _cadna_half_overflow_count; 
unsigned long _cadna_half_underflow_count; 
#endif

unsigned int _cadna_div_tag=1;
unsigned int _cadna_mul_tag=1;
unsigned int _cadna_power_tag=1;
unsigned int _cadna_math_tag=1;
unsigned int _cadna_intrinsic_tag=1;
unsigned int _cadna_cancel_tag=1;
unsigned int _cadna_branching_tag=1;
#ifdef CADNA_HALF_EMULATION
unsigned int _cadna_half_overflow_tag=1; 
unsigned int _cadna_half_underflow_tag=1; 
#endif
#ifdef CADNA_HALF_NATIVE
unsigned int _cadna_half_overflow_tag=1; 
unsigned int _cadna_half_underflow_tag=1; 
#endif

unsigned int _cadna_div_change=0;
unsigned int _cadna_mul_change=0;
unsigned int _cadna_power_change=0;
unsigned int _cadna_math_change=0;
unsigned int _cadna_intrinsic_change=0;
unsigned int _cadna_cancel_change=0;
unsigned int _cadna_branching_change=0;
#ifdef CADNA_HALF_EMULATION
unsigned int _cadna_half_overflow_change=0; 
unsigned int _cadna_half_underflow_change=0;
#endif
#ifdef CADNA_HALF_NATIVE
unsigned int _cadna_half_overflow_change=0; 
unsigned int _cadna_half_underflow_change=0;
#endif

int _cadna_cancel_value=4;

extern int _cadna_cancel_value;

const char* _cadna_div_err_msg       = "UNSTABLE DIVISION(S)";
const char* _cadna_mul_err_msg       = "UNSTABLE MULTIPLICATION(S)";
const char* _cadna_power_err_msg     = "UNSTABLE POWER FUNCTION(S)";
const char* _cadna_branching_err_msg = "UNSTABLE BRANCHING(S)";
const char* _cadna_cancel_err_msg    = "LOSS(ES) OF ACCURACY DUE TO CANCELLATION(S)";
const char* _cadna_intrinsic_err_msg = "UNSTABLE INTRINSIC FUNCTION(S)";
const char* _cadna_math_err_msg      = "UNSTABLE MATHEMATICAL FUNCTION(S)";
#ifdef CADNA_HALF_EMULATION
const char* _cadna_half_overflow_err_msg = "OVERFLOW(S) IS(ARE) DETECTED";
const char* _cadna_half_underflow_err_msg = "UNDERFLOW(S) IS(ARE) DETECTED";
#endif
#ifdef CADNA_HALF_NATIVE
const char* _cadna_half_overflow_err_msg = "OVERFLOW(S) IS(ARE) DETECTED";
const char* _cadna_half_underflow_err_msg = "UNDERFLOW(S) IS(ARE) DETECTED";
#endif

//****f* cadna_type/cadna_init
//    NAME
//     cadna_init
//
//    SYNOPSIS
//      void cadna_init(int max_nb_instability) 
//      void cadna_init(int max_nb_instability, unsigned int tag) 
//      void cadna_init(int max_instability, unsigned int tag,
//                      unsigned int seed, unsigned int cancellation)
//
//    FUNCTION
//     The cadna_init function initializes the CADNA library
//     (see documentation).
//      
//    INPUTS
//   
//    RESULT

//*****
//   You can use this space for remarks that should not be included
//   in the documentation.
//    EXAMPLE
//      
//  
//    NOTES
//  
//  
//    BUGS
//  
//  
//      
//      
//  /

void  cadna_init(int max_nb_instability) 
{
  cadna_init(max_nb_instability,0,51,_cadna_cancel_value);
}

void  cadna_init(int max_nb_instability,unsigned int tag) 
{
  cadna_init(max_nb_instability,tag,51,_cadna_cancel_value);
}




 void  cadna_init(int max_instability, unsigned int tag, unsigned int seed, unsigned int cancellation)
{
  _cadna_cancel_value=cancellation;
  
  _cadna_max_instability=max_instability;
  _cadna_instability_detected=0;
  
  _cadna_div_count=0;
  _cadna_mul_count=0;  
  _cadna_power_count=0;  
  _cadna_math_count=0; 
  _cadna_intrinsic_count=0;  
  _cadna_cancel_count=0;  
  _cadna_branching_count=0; 
#ifdef CADNA_HALF_EMULATION
  _cadna_half_overflow_count=0; 
  _cadna_half_underflow_count=0; 
#endif
#ifdef CADNA_HALF_NATIVE
  _cadna_half_overflow_count=0; 
  _cadna_half_underflow_count=0; 
#endif
  
  if ( max_instability == 0){
    _cadna_div_change=1; 
    _cadna_mul_change=1;  
    _cadna_power_change=1;  
    _cadna_branching_change=1;
    _cadna_cancel_change=1; 
    _cadna_intrinsic_change=1; 
    _cadna_math_change=1;
#ifdef CADNA_HALF_EMULATION
    _cadna_half_overflow_change=1;
    _cadna_half_underflow_change=1;
#endif
#ifdef CADNA_HALF_NATIVE
    _cadna_half_overflow_change=1;
    _cadna_half_underflow_change=1;
#endif
    _cadna_div_tag=0; 
    _cadna_mul_tag=0;  
    _cadna_power_tag=0;  
    _cadna_branching_tag=0;
    _cadna_cancel_tag=0; 
    _cadna_intrinsic_tag=0; 
    _cadna_math_tag=0; 
#ifdef CADNA_HALF_EMULATION
    _cadna_half_overflow_tag=0;
    _cadna_half_underflow_tag=0;
#endif
#ifdef CADNA_HALF_NATIVE
    _cadna_half_overflow_tag=0;
    _cadna_half_underflow_tag=0;
#endif

  }
  else {
    if (tag & CADNA_DIV)   {
      _cadna_div_tag=0; 
      _cadna_div_change=1; 
    }
    if (tag & CADNA_MUL) {
      _cadna_mul_tag=0; 
      _cadna_mul_change=1; 
    }
    if (tag & CADNA_POWER) {
      _cadna_power_tag=0; 
      _cadna_power_change=1; 
    }
    if (tag & CADNA_MATH) {
      _cadna_math_tag=0; 
      _cadna_math_change=1; 
    }
    if (tag & CADNA_INTRINSIC) {
      _cadna_intrinsic_tag=0; 
      _cadna_intrinsic_change=1; 
    }
    if (tag & CADNA_CANCEL) {
      _cadna_cancel_tag=0; 
      _cadna_cancel_change=1; 
    }
    if (tag & CADNA_BRANCHING) {
      _cadna_branching_tag=0; 
      _cadna_branching_change=1; 
    }
#ifdef CADNA_HALF_EMULATION
    if (tag & CADNA_HALF_OVERFLOW) {
      _cadna_half_overflow_tag=0; 
      _cadna_half_overflow_change=1; 
    }
    
    if (tag & CADNA_HALF_UNDERFLOW) {
      _cadna_half_underflow_tag=0; 
      _cadna_half_underflow_change=1; 
    }
#endif
#ifdef CADNA_HALF_NATIVE
    if (tag & CADNA_HALF_OVERFLOW) {
      _cadna_half_overflow_tag=0; 
      _cadna_half_overflow_change=1; 
    }
    
    if (tag & CADNA_HALF_UNDERFLOW) {
      _cadna_half_underflow_tag=0; 
      _cadna_half_underflow_change=1; 
    }
#endif

  }
  cout << "----------------------------------------------------------------" << endl;
  
  cout << PACKAGE_STRING << " software" << endl;
  
  if (_cadna_div_tag & _cadna_mul_tag & _cadna_power_tag) 
    cout << "Self-validation detection: ON" << endl;
  else     cout << "Self-validation detection: OFF" << endl;
  cout << "Mathematical instabilities detection: " <<  (_cadna_math_tag ? "ON": "OFF") << endl;
  cout << "Branching instabilities detection: " << (_cadna_branching_tag ? "ON": "OFF") << endl;
  cout << "Intrinsic instabilities detection: " << (_cadna_intrinsic_tag ? "ON": "OFF") << endl;
  cout << "Cancellation instabilities detection: " << (_cadna_cancel_tag ? "ON": "OFF") << endl;
#ifdef CADNA_HALF_EMULATION
  cout << "Overflow detection for half-precision: " << (_cadna_half_overflow_tag ? "ON": "OFF") << endl;
  cout << "Underflow detection for half-precision: " << (_cadna_half_underflow_tag ? "ON": "OFF") << endl;
#endif
#ifdef CADNA_HALF_NATIVE
  cout << "Overflow detection for half-precision: " << (_cadna_half_overflow_tag ? "ON": "OFF") << endl;
  cout << "Underflow detection for half-precision: " << (_cadna_half_underflow_tag ? "ON": "OFF") << endl;
#endif
  cout << "----------------------------------------------------------------" << endl;


  fesetround(FE_UPWARD);

////////////////// on vérifie que tout le monde  passe bien en mode arrondi vers + l'infini.

#ifdef _OPENMP
  unsigned int rounded_up = 1;
#pragma omp parallel reduction(&&:rounded_up)
  {
    double test = 1.;
    test += 1.e-20;
    rounded_up = rounded_up && (test != 1.0);
  }
// Pierre : barriere a priori inutile car hors d'une region parallele OpenMP :
#pragma omp barrier 

  /////////////////////


  if (!rounded_up) {
#pragma omp parallel
    {
      fesetround(FE_UPWARD);
    }
// Pierre : barriere a priori inutile car hors d'une region parallele OpenMP :
#pragma omp barrier

    rounded_up = 1;
#pragma omp parallel reduction(&&:rounded_up)
    {
      double test = 1.;
      test += 1.e-20;
      rounded_up = rounded_up && (test != 1.0);
    }
// Pierre : barriere a priori inutile car hors d'une region parallele OpenMP :
#pragma omp barrier


    ////////////////////
    if (!rounded_up) {
      cout << "The optimisation options used to compile the code are too efficient for CADNA-OpenMP" << endl;
      cout << "Try to use a cadna debug library or -O0 option for the compiler" << endl;
      exit(-1);
    }
  }
#endif //_OPENMP



#ifdef _OPENMP

#pragma omp parallel shared(seed)
  {
  _cadna_recurrence=omp_get_thread_num()*1099087573UL+seed;
  _cadna_random=0;
   _cadna_random_counter=0;
  }
#pragma omp barrier

#else //_OPENMP
  //  _cadna_recurrence+=seed; //modif Sept. 2021
  _cadna_recurrence=seed;
  _cadna_random=0;
  _cadna_random_counter=0;
#endif //_OPENMP

}


//****f* cadna_type/cadna_end
//    NAME
//      cadna_end
//
//    SYNOPSIS
//      void cadna_end()
//    FUNCTION
//      The cadna_end function "closes" the CADNA library
//      (see documentation).
//    INPUTS
//      no input parameter  
//    RESULT
//      void
//*****
//   You can use this space for remarks that should not be included
//   in the documentation.
//    EXAMPLE
//      
//  
//    NOTES
//  
//  
//    BUGS
//  
//  
//      
//      
//  /


void cadna_end()
{
  fesetround(FE_TONEAREST);
  unsigned long long sum;
  cout << "----------------------------------------------------------------" << endl;
  cout << PACKAGE_STRING << " software" << endl;
  sum= _cadna_div_count+_cadna_mul_count+_cadna_math_count+_cadna_power_count +
    _cadna_intrinsic_count+_cadna_cancel_count+_cadna_branching_count;
  
  if (_cadna_div_change || _cadna_mul_change || _cadna_power_change){
    cout << endl;
    cout << "BE CAREFUL: the self-validation has been deactivated." << endl;
    cout << "The results are NOT guaranteed." << endl ;
    cout << endl;
  }
  
  if (!sum) cout << "No instability detected" << endl;
  else {
    if (_cadna_div_count+_cadna_mul_count+_cadna_power_count){
      cout << endl;
      cout <<  "CRITICAL WARNING: the self-validation detects major problem(s)." << endl;
      cout << "The results are NOT guaranteed." << endl ;
      cout << endl;
    }
    
    
    
    if (sum==1) cout << "There is  1 numerical instability" << endl;
    else cout << "There are " << sum << " numerical instabilities"<<endl;
    
    if  (_cadna_div_tag && _cadna_div_count) 
      cout <<  _cadna_div_count << " " <<  _cadna_div_err_msg << endl;
    if  (_cadna_mul_tag && _cadna_mul_count) 
      cout <<  _cadna_mul_count << " " <<  _cadna_mul_err_msg << endl;
    if  (_cadna_power_tag && _cadna_power_count) 
      cout <<  _cadna_power_count << " " <<  _cadna_power_err_msg << endl;
    if  (_cadna_branching_tag && _cadna_branching_count) 
      cout <<  _cadna_branching_count << " " <<  _cadna_branching_err_msg << endl;
    if (_cadna_intrinsic_tag && _cadna_intrinsic_count) 
      cout <<  _cadna_intrinsic_count << " " <<  _cadna_intrinsic_err_msg << endl;
    if  (_cadna_math_tag && _cadna_math_count) 
      cout <<  _cadna_math_count << " " <<  _cadna_math_err_msg << endl;
    if  (_cadna_cancel_tag && _cadna_cancel_count) 
      cout <<  _cadna_cancel_count << " " <<  _cadna_cancel_err_msg << endl;
  }
#ifdef CADNA_HALF_EMULATION
  if (_cadna_half_overflow_tag && _cadna_half_overflow_count) {
    cout << endl;
    cout << "BE CAREFUL: over range values have been detected." << endl;
    cout << "The results are NOT guaranteed." << endl ;
    cout << _cadna_half_overflow_count << " " << _cadna_half_overflow_err_msg << endl; 
    cout << endl;
  }
  
    if (_cadna_half_underflow_tag && _cadna_half_underflow_count) {
    cout << endl;
    cout << "BE CAREFUL: under range values have been detected." << endl;
    cout << "The results are NOT guaranteed." << endl ;
    cout << _cadna_half_underflow_count << " " << _cadna_half_underflow_err_msg << endl; 
    cout << endl;
  }
#endif  
#ifdef CADNA_HALF_NATIVE
  if (_cadna_half_overflow_tag && _cadna_half_overflow_count) {
    cout << endl;
    cout << "BE CAREFUL: over range values have been detected." << endl;
    cout << "The results are NOT guaranteed." << endl ;
    cout << _cadna_half_overflow_count << " " << _cadna_half_overflow_err_msg << endl; 
    cout << endl;
  }
  
    if (_cadna_half_underflow_tag && _cadna_half_underflow_count) {
    cout << endl;
    cout << "BE CAREFUL: under range values have been detected." << endl;
    cout << "The results are NOT guaranteed." << endl ;
    cout << _cadna_half_underflow_count << " " << _cadna_half_underflow_err_msg << endl; 
    cout << endl;
  }
#endif 
  cout << "----------------------------------------------------------------" << endl;

}


//****f* cadna_type/cadna_enable
//    NAME
//     cadna_enable
//
//    SYNOPSIS
//      void cadna_enable(unsigned int tag)
//    FUNCTION
//      The cadna_enable function enables the detection of a kind of
//      numerical instability.
//    INPUTS
//      unsigned int tag (see documentation)
//    RESULT
//      
//    SEE ALSO
//        cadna_disable(3)
//*****
//   You can use this space for remarks that should not be included
//   in the documentation.
//    EXAMPLE
//      
//  
//    NOTES
//  
//  
//    BUGS
//  
//  
//      
//      
//  /

void cadna_enable(unsigned int tag)
{
  if (_cadna_max_instability==-2) {
    cerr << "-----------------------------------------------------------" << endl;
    cerr << " CADNA: The maximum number of instabilities has been reached" << endl;
    cerr << " The cadna_select_instability is not performed"    << endl;
    cerr << "-----------------------------------------------------------" << endl;
  }
  
  if (tag & CADNA_DIV)  { 
    if (!_cadna_div_tag) _cadna_div_change=1;
    _cadna_div_tag=1;
  }
  
  if (tag & CADNA_MUL)  { 
    if (!_cadna_mul_tag) _cadna_mul_change=1;
    _cadna_mul_tag=1;
  }

  if (tag & CADNA_POWER)  { 
    if (!_cadna_power_tag) _cadna_power_change=1;
    _cadna_power_tag=1;
  }
  
  if (tag & CADNA_BRANCHING){ 
    if (!_cadna_branching_tag) _cadna_branching_change=1;
    _cadna_branching_tag=1;
  }
  
  if (tag & CADNA_CANCEL){ 
    if (!_cadna_cancel_tag) _cadna_cancel_change=1;
    _cadna_cancel_tag=1;
  }
  
  if (tag & CADNA_INTRINSIC){ 
    if (!_cadna_intrinsic_tag) _cadna_intrinsic_change=1;
    _cadna_intrinsic_tag=1;
  }
  
  if (tag & CADNA_MATH){ 
    if (!_cadna_math_tag) _cadna_math_change=1;
    _cadna_math_tag=1;
  }
#ifdef CADNA_HALF_EMULATION
  if (tag & CADNA_HALF_OVERFLOW){ 
    if (!_cadna_half_overflow_tag) _cadna_half_overflow_change=1;
    _cadna_half_overflow_tag=1;
  }
  
    if (tag & CADNA_HALF_UNDERFLOW){ 
    if (!_cadna_half_underflow_tag) _cadna_half_underflow_change=1;
    _cadna_half_underflow_tag=1;
  }
#endif
#ifdef CADNA_HALF_NATIVE
  if (tag & CADNA_HALF_OVERFLOW){ 
    if (!_cadna_half_overflow_tag) _cadna_half_overflow_change=1;
    _cadna_half_overflow_tag=1;
  }
  
    if (tag & CADNA_HALF_UNDERFLOW){ 
    if (!_cadna_half_underflow_tag) _cadna_half_underflow_change=1;
    _cadna_half_underflow_tag=1;
  }
#endif
}




//****f* cadna_type/cadna_disable
//    NAME
//     cadna_disable
//
//    SYNOPSIS
//      void cadna_disable(unsigned int tag)
//    FUNCTION
//      The cadna_disable function disables the detection of a kind of
//      numerical instability.
//    INPUTS
//      unsigned int tag (see documentation)
//    RESULT
//    
//    SEE ALSO
//      cadna_enable(3)
//*****
//   You can use this space for remarks that should not be included
//   in the documentation.
//    EXAMPLE
//      
//  
//    NOTES
//  
//  
//    BUGS
//  
//  
//      
//      
//  /


void cadna_disable(unsigned int tag)
{
  if (tag & CADNA_DIV){ 
    if (_cadna_div_tag) _cadna_div_change=1;
    _cadna_div_tag=0;
  }
  
  if (tag & CADNA_MUL){ 
    _cadna_mul_change=1;
    _cadna_mul_tag=0;
  }

  if (tag & CADNA_POWER){ 
    if (_cadna_power_tag) _cadna_power_change=1;
    _cadna_power_tag=0;
  }
  
  if (tag & CADNA_BRANCHING){ 
    if (_cadna_branching_tag) _cadna_branching_change=1;
    _cadna_branching_tag=0;
  }

  if (tag & CADNA_CANCEL){ 
    if (_cadna_cancel_tag) _cadna_cancel_change=1;
    _cadna_cancel_tag=0;
  }
  
  if (tag & CADNA_INTRINSIC){ 
    if (_cadna_intrinsic_tag) _cadna_intrinsic_change=1;
    _cadna_intrinsic_tag=0;
  }
  
  if (tag & CADNA_MATH){ 
    if (_cadna_math_tag) _cadna_math_change=1;
    _cadna_math_tag=0;
  }
#ifdef CADNA_HALF_EMULATION
  if (tag & CADNA_HALF_OVERFLOW){ 
    if (_cadna_half_overflow_tag) _cadna_half_overflow_change=1;
    _cadna_half_overflow_tag=0;
  }
  
    if (tag & CADNA_HALF_UNDERFLOW){ 
    if (_cadna_half_underflow_tag) _cadna_half_underflow_change=1;
    _cadna_half_underflow_tag=0;
  }
#endif
#ifdef CADNA_HALF_NATIVE
  if (tag & CADNA_HALF_OVERFLOW){ 
    if (_cadna_half_overflow_tag) _cadna_half_overflow_change=1;
    _cadna_half_overflow_tag=0;
  }
  
    if (tag & CADNA_HALF_UNDERFLOW){ 
    if (_cadna_half_underflow_tag) _cadna_half_underflow_change=1;
    _cadna_half_underflow_tag=0;
  }
#endif
}

