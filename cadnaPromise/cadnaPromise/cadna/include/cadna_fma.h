// Copyright 2015-2017 J.-M. Chesneaux, P. Eberhart, F. Jezequel, J.-L. Lamotte, R. Picot
// Copyright 2022 F. Jezequel, J.-L. Lamotte

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

#ifdef CADNA_FMA


#include "cadna.h"
#include <math.h>

///////////////////////////////////////////////////
//****m* cadna_fma/fma
//    NAME
//      fma
//    SYNOPSIS
//      res = a * b + c
//    FUNCTION
//      Defines all the functions involving at least one argument
//      of stochastic type which overload the "fma" operator
//      in a statement such as "a*b+c".
//
//    INPUTS
//      a           - an integer, a float, a double or a stochastic number
//      b           - an integer, a float, a double or a stochastic number
//      c           - an integer, a float, a double or a stochastic number
//      At least one argument must be of stochastic type.
//    RESULT
//      res         - a stochastic number
//
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
//    SEE ALSO
//
//
//  /






///////////////////////////////////////////////////
//****m* cadna_fma/fma_no_instab
//    NAME
//      fma
//    SYNOPSIS
//      res = a * b + c
//    FUNCTION
//      Defines all the functions involving at least one argument
//      of stochastic type which overload the "fma" operator
//      in a statement such as "a*b+c" without instability detection.
//
//    INPUTS
//      a           - an integer, a float, a double or a stochastic number
//      b           - an integer, a float, a double or a stochastic number
//      c           - an integer, a float, a double or a stochastic number
//      At least one argument must be of stochastic type.
//    RESULT
//      res         - a stochastic number
//
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
//    SEE ALSO
//
//
//  /


inline double_st fma( double_st const& a,  double_st const& b,  double_st const& c){
  double_st res;
  unsigned int r1;
  unsigned int r2;
  unsigned int r3;
  double tmpa;
  double tmpb;
  double tmpc;

  res.accuracy=DIGIT_NOT_COMPUTED;

  r1 = RANDOM;
  tmpa = a.x;
  tmpb = b.x;
  tmpc = c.x;
  DOUBLE_BIT_FLIP(tmpa, r1);
  DOUBLE_BIT_FLIP(tmpc, r1);
  res.x = (double)fma((double)tmpa, (double)tmpb, (double)tmpc);
  DOUBLE_BIT_FLIP(res.x, r1);

  r2 = RANDOM;
  tmpa = a.y;
  tmpb = b.y;
  tmpc = c.y;
  DOUBLE_BIT_FLIP(tmpa, r2);
  DOUBLE_BIT_FLIP(tmpc, r2);
  res.y = (double)fma((double)tmpa, (double)tmpb, (double)tmpc);
  DOUBLE_BIT_FLIP(res.y, r2);


  r3 = 1^r2;
  tmpa = a.z;
  tmpb = b.z;
  tmpc = c.z;
  DOUBLE_BIT_FLIP(tmpa, r3);
  DOUBLE_BIT_FLIP(tmpc, r3);
  res.z = (double)fma((double)tmpa, (double)tmpb, (double)tmpc);
  DOUBLE_BIT_FLIP(res.z, r3);

  if(_cadna_mul_tag) { 
    if (a.accuracy==DIGIT_NOT_COMPUTED) a.approx_digit();	 
    if (a.accuracy==0){ 
      if (b.accuracy==DIGIT_NOT_COMPUTED) b.approx_digit(); 
      if (b.accuracy==0) 
	instability(&_cadna_mul_count); 
    } 
  } 

  if(_cadna_cancel_tag){
    if (a.accuracy==DIGIT_NOT_COMPUTED || a.accuracy==RELIABLE_RESULT)	
      a.nb_significant_digit();						
    if (b.accuracy==DIGIT_NOT_COMPUTED || b.accuracy==RELIABLE_RESULT)	
      b.nb_significant_digit();						
    if (c.accuracy==DIGIT_NOT_COMPUTED || c.accuracy==RELIABLE_RESULT)	
      c.nb_significant_digit();						
    if ((a.accuracy < b.accuracy ? (a.accuracy < c.accuracy ? a.accuracy : c.accuracy):
	 (b.accuracy < c.accuracy ? b.accuracy : c.accuracy)) >=		
	res.nb_significant_digit()+_cadna_cancel_value)			
      _cadna_cancel_count++;
  }


  return res;
}
inline double_st fma( float_st const& a,  double_st const& b,  double_st const& c){
  double_st res;
  unsigned int r1;
  unsigned int r2;
  unsigned int r3;
  double tmpa;
  double tmpb;
  double tmpc;

  res.accuracy=DIGIT_NOT_COMPUTED;

  r1 = RANDOM;
  tmpa = a.x;
  tmpb = b.x;
  tmpc = c.x;
  DOUBLE_BIT_FLIP(tmpa, r1);
  DOUBLE_BIT_FLIP(tmpc, r1);
  res.x = (double)fma((double)tmpa, (double)tmpb, (double)tmpc);
  DOUBLE_BIT_FLIP(res.x, r1);

  r2 = RANDOM;
  tmpa = a.y;
  tmpb = b.y;
  tmpc = c.y;
  DOUBLE_BIT_FLIP(tmpa, r2);
  DOUBLE_BIT_FLIP(tmpc, r2);
  res.y = (double)fma((double)tmpa, (double)tmpb, (double)tmpc);
  DOUBLE_BIT_FLIP(res.y, r2);


  r3 = 1^r2;
  tmpa = a.z;
  tmpb = b.z;
  tmpc = c.z;
  DOUBLE_BIT_FLIP(tmpa, r3);
  DOUBLE_BIT_FLIP(tmpc, r3);
  res.z = (double)fma((double)tmpa, (double)tmpb, (double)tmpc);
  DOUBLE_BIT_FLIP(res.z, r3);

  if(_cadna_mul_tag) { 
    if (a.accuracy==DIGIT_NOT_COMPUTED) a.approx_digit();	 
    if (a.accuracy==0){ 
      if (b.accuracy==DIGIT_NOT_COMPUTED) b.approx_digit(); 
      if (b.accuracy==0) 
	instability(&_cadna_mul_count); 
    } 
  } 

  if(_cadna_cancel_tag){
    if (a.accuracy==DIGIT_NOT_COMPUTED || a.accuracy==RELIABLE_RESULT)	
      a.nb_significant_digit();						
    if (b.accuracy==DIGIT_NOT_COMPUTED || b.accuracy==RELIABLE_RESULT)	
      b.nb_significant_digit();						
    if (c.accuracy==DIGIT_NOT_COMPUTED || c.accuracy==RELIABLE_RESULT)	
      c.nb_significant_digit();						
    if ((a.accuracy < b.accuracy ? (a.accuracy < c.accuracy ? a.accuracy : c.accuracy):
	 (b.accuracy < c.accuracy ? b.accuracy : c.accuracy)) >=		
	res.nb_significant_digit()+_cadna_cancel_value)			
      _cadna_cancel_count++;
  }


  return res;
}
inline double_st fma( double_st const& a,  float_st const& b,  double_st const& c){
  double_st res;
  unsigned int r1;
  unsigned int r2;
  unsigned int r3;
  double tmpa;
  double tmpb;
  double tmpc;

  res.accuracy=DIGIT_NOT_COMPUTED;

  r1 = RANDOM;
  tmpa = a.x;
  tmpb = b.x;
  tmpc = c.x;
  DOUBLE_BIT_FLIP(tmpa, r1);
  DOUBLE_BIT_FLIP(tmpc, r1);
  res.x = (double)fma((double)tmpa, (double)tmpb, (double)tmpc);
  DOUBLE_BIT_FLIP(res.x, r1);

  r2 = RANDOM;
  tmpa = a.y;
  tmpb = b.y;
  tmpc = c.y;
  DOUBLE_BIT_FLIP(tmpa, r2);
  DOUBLE_BIT_FLIP(tmpc, r2);
  res.y = (double)fma((double)tmpa, (double)tmpb, (double)tmpc);
  DOUBLE_BIT_FLIP(res.y, r2);


  r3 = 1^r2;
  tmpa = a.z;
  tmpb = b.z;
  tmpc = c.z;
  DOUBLE_BIT_FLIP(tmpa, r3);
  DOUBLE_BIT_FLIP(tmpc, r3);
  res.z = (double)fma((double)tmpa, (double)tmpb, (double)tmpc);
  DOUBLE_BIT_FLIP(res.z, r3);

  if(_cadna_mul_tag) { 
    if (a.accuracy==DIGIT_NOT_COMPUTED) a.approx_digit();	 
    if (a.accuracy==0){ 
      if (b.accuracy==DIGIT_NOT_COMPUTED) b.approx_digit(); 
      if (b.accuracy==0) 
	instability(&_cadna_mul_count); 
    } 
  } 

  if(_cadna_cancel_tag){
    if (a.accuracy==DIGIT_NOT_COMPUTED || a.accuracy==RELIABLE_RESULT)	
      a.nb_significant_digit();						
    if (b.accuracy==DIGIT_NOT_COMPUTED || b.accuracy==RELIABLE_RESULT)	
      b.nb_significant_digit();						
    if (c.accuracy==DIGIT_NOT_COMPUTED || c.accuracy==RELIABLE_RESULT)	
      c.nb_significant_digit();						
    if ((a.accuracy < b.accuracy ? (a.accuracy < c.accuracy ? a.accuracy : c.accuracy):
	 (b.accuracy < c.accuracy ? b.accuracy : c.accuracy)) >=		
	res.nb_significant_digit()+_cadna_cancel_value)			
      _cadna_cancel_count++;
  }


  return res;
}
inline double_st fma( double_st const& a,  double_st const& b,  float_st const& c){
  double_st res;
  unsigned int r1;
  unsigned int r2;
  unsigned int r3;
  double tmpa;
  double tmpb;
  double tmpc;

  res.accuracy=DIGIT_NOT_COMPUTED;

  r1 = RANDOM;
  tmpa = a.x;
  tmpb = b.x;
  tmpc = c.x;
  DOUBLE_BIT_FLIP(tmpa, r1);
  DOUBLE_BIT_FLIP(tmpc, r1);
  res.x = (double)fma((double)tmpa, (double)tmpb, (double)tmpc);
  DOUBLE_BIT_FLIP(res.x, r1);

  r2 = RANDOM;
  tmpa = a.y;
  tmpb = b.y;
  tmpc = c.y;
  DOUBLE_BIT_FLIP(tmpa, r2);
  DOUBLE_BIT_FLIP(tmpc, r2);
  res.y = (double)fma((double)tmpa, (double)tmpb, (double)tmpc);
  DOUBLE_BIT_FLIP(res.y, r2);


  r3 = 1^r2;
  tmpa = a.z;
  tmpb = b.z;
  tmpc = c.z;
  DOUBLE_BIT_FLIP(tmpa, r3);
  DOUBLE_BIT_FLIP(tmpc, r3);
  res.z = (double)fma((double)tmpa, (double)tmpb, (double)tmpc);
  DOUBLE_BIT_FLIP(res.z, r3);

  if(_cadna_mul_tag) { 
    if (a.accuracy==DIGIT_NOT_COMPUTED) a.approx_digit();	 
    if (a.accuracy==0){ 
      if (b.accuracy==DIGIT_NOT_COMPUTED) b.approx_digit(); 
      if (b.accuracy==0) 
	instability(&_cadna_mul_count); 
    } 
  } 

  if(_cadna_cancel_tag){
    if (a.accuracy==DIGIT_NOT_COMPUTED || a.accuracy==RELIABLE_RESULT)	
      a.nb_significant_digit();						
    if (b.accuracy==DIGIT_NOT_COMPUTED || b.accuracy==RELIABLE_RESULT)	
      b.nb_significant_digit();						
    if (c.accuracy==DIGIT_NOT_COMPUTED || c.accuracy==RELIABLE_RESULT)	
      c.nb_significant_digit();						
    if ((a.accuracy < b.accuracy ? (a.accuracy < c.accuracy ? a.accuracy : c.accuracy):
	 (b.accuracy < c.accuracy ? b.accuracy : c.accuracy)) >=		
	res.nb_significant_digit()+_cadna_cancel_value)			
      _cadna_cancel_count++;
  }


  return res;
}
inline double_st fma( float_st const& a,  float_st const& b,  double_st const& c){
  double_st res;
  unsigned int r1;
  unsigned int r2;
  unsigned int r3;
  double tmpa;
  double tmpb;
  double tmpc;

  res.accuracy=DIGIT_NOT_COMPUTED;

  r1 = RANDOM;
  tmpa = a.x;
  tmpb = b.x;
  tmpc = c.x;
  DOUBLE_BIT_FLIP(tmpa, r1);
  DOUBLE_BIT_FLIP(tmpc, r1);
  res.x = (double)fma((double)tmpa, (double)tmpb, (double)tmpc);
  DOUBLE_BIT_FLIP(res.x, r1);

  r2 = RANDOM;
  tmpa = a.y;
  tmpb = b.y;
  tmpc = c.y;
  DOUBLE_BIT_FLIP(tmpa, r2);
  DOUBLE_BIT_FLIP(tmpc, r2);
  res.y = (double)fma((double)tmpa, (double)tmpb, (double)tmpc);
  DOUBLE_BIT_FLIP(res.y, r2);


  r3 = 1^r2;
  tmpa = a.z;
  tmpb = b.z;
  tmpc = c.z;
  DOUBLE_BIT_FLIP(tmpa, r3);
  DOUBLE_BIT_FLIP(tmpc, r3);
  res.z = (double)fma((double)tmpa, (double)tmpb, (double)tmpc);
  DOUBLE_BIT_FLIP(res.z, r3);

  if(_cadna_mul_tag) { 
    if (a.accuracy==DIGIT_NOT_COMPUTED) a.approx_digit();	 
    if (a.accuracy==0){ 
      if (b.accuracy==DIGIT_NOT_COMPUTED) b.approx_digit(); 
      if (b.accuracy==0) 
	instability(&_cadna_mul_count); 
    } 
  } 

  if(_cadna_cancel_tag){
    if (a.accuracy==DIGIT_NOT_COMPUTED || a.accuracy==RELIABLE_RESULT)	
      a.nb_significant_digit();						
    if (b.accuracy==DIGIT_NOT_COMPUTED || b.accuracy==RELIABLE_RESULT)	
      b.nb_significant_digit();						
    if (c.accuracy==DIGIT_NOT_COMPUTED || c.accuracy==RELIABLE_RESULT)	
      c.nb_significant_digit();						
    if ((a.accuracy < b.accuracy ? (a.accuracy < c.accuracy ? a.accuracy : c.accuracy):
	 (b.accuracy < c.accuracy ? b.accuracy : c.accuracy)) >=		
	res.nb_significant_digit()+_cadna_cancel_value)			
      _cadna_cancel_count++;
  }


  return res;
}
inline double_st fma( float_st const& a,  double_st const& b,  float_st const& c){
  double_st res;
  unsigned int r1;
  unsigned int r2;
  unsigned int r3;
  double tmpa;
  double tmpb;
  double tmpc;

  res.accuracy=DIGIT_NOT_COMPUTED;

  r1 = RANDOM;
  tmpa = a.x;
  tmpb = b.x;
  tmpc = c.x;
  DOUBLE_BIT_FLIP(tmpa, r1);
  DOUBLE_BIT_FLIP(tmpc, r1);
  res.x = (double)fma((double)tmpa, (double)tmpb, (double)tmpc);
  DOUBLE_BIT_FLIP(res.x, r1);

  r2 = RANDOM;
  tmpa = a.y;
  tmpb = b.y;
  tmpc = c.y;
  DOUBLE_BIT_FLIP(tmpa, r2);
  DOUBLE_BIT_FLIP(tmpc, r2);
  res.y = (double)fma((double)tmpa, (double)tmpb, (double)tmpc);
  DOUBLE_BIT_FLIP(res.y, r2);


  r3 = 1^r2;
  tmpa = a.z;
  tmpb = b.z;
  tmpc = c.z;
  DOUBLE_BIT_FLIP(tmpa, r3);
  DOUBLE_BIT_FLIP(tmpc, r3);
  res.z = (double)fma((double)tmpa, (double)tmpb, (double)tmpc);
  DOUBLE_BIT_FLIP(res.z, r3);

  if(_cadna_mul_tag) { 
    if (a.accuracy==DIGIT_NOT_COMPUTED) a.approx_digit();	 
    if (a.accuracy==0){ 
      if (b.accuracy==DIGIT_NOT_COMPUTED) b.approx_digit(); 
      if (b.accuracy==0) 
	instability(&_cadna_mul_count); 
    } 
  } 

  if(_cadna_cancel_tag){
    if (a.accuracy==DIGIT_NOT_COMPUTED || a.accuracy==RELIABLE_RESULT)	
      a.nb_significant_digit();						
    if (b.accuracy==DIGIT_NOT_COMPUTED || b.accuracy==RELIABLE_RESULT)	
      b.nb_significant_digit();						
    if (c.accuracy==DIGIT_NOT_COMPUTED || c.accuracy==RELIABLE_RESULT)	
      c.nb_significant_digit();						
    if ((a.accuracy < b.accuracy ? (a.accuracy < c.accuracy ? a.accuracy : c.accuracy):
	 (b.accuracy < c.accuracy ? b.accuracy : c.accuracy)) >=		
	res.nb_significant_digit()+_cadna_cancel_value)			
      _cadna_cancel_count++;
  }


  return res;
}
inline double_st fma( double_st const& a,  float_st const& b,  float_st const& c){
  double_st res;
  unsigned int r1;
  unsigned int r2;
  unsigned int r3;
  double tmpa;
  double tmpb;
  double tmpc;

  res.accuracy=DIGIT_NOT_COMPUTED;

  r1 = RANDOM;
  tmpa = a.x;
  tmpb = b.x;
  tmpc = c.x;
  DOUBLE_BIT_FLIP(tmpa, r1);
  DOUBLE_BIT_FLIP(tmpc, r1);
  res.x = (double)fma((double)tmpa, (double)tmpb, (double)tmpc);
  DOUBLE_BIT_FLIP(res.x, r1);

  r2 = RANDOM;
  tmpa = a.y;
  tmpb = b.y;
  tmpc = c.y;
  DOUBLE_BIT_FLIP(tmpa, r2);
  DOUBLE_BIT_FLIP(tmpc, r2);
  res.y = (double)fma((double)tmpa, (double)tmpb, (double)tmpc);
  DOUBLE_BIT_FLIP(res.y, r2);


  r3 = 1^r2;
  tmpa = a.z;
  tmpb = b.z;
  tmpc = c.z;
  DOUBLE_BIT_FLIP(tmpa, r3);
  DOUBLE_BIT_FLIP(tmpc, r3);
  res.z = (double)fma((double)tmpa, (double)tmpb, (double)tmpc);
  DOUBLE_BIT_FLIP(res.z, r3);

  if(_cadna_mul_tag) { 
    if (a.accuracy==DIGIT_NOT_COMPUTED) a.approx_digit();	 
    if (a.accuracy==0){ 
      if (b.accuracy==DIGIT_NOT_COMPUTED) b.approx_digit(); 
      if (b.accuracy==0) 
	instability(&_cadna_mul_count); 
    } 
  } 

  if(_cadna_cancel_tag){
    if (a.accuracy==DIGIT_NOT_COMPUTED || a.accuracy==RELIABLE_RESULT)	
      a.nb_significant_digit();						
    if (b.accuracy==DIGIT_NOT_COMPUTED || b.accuracy==RELIABLE_RESULT)	
      b.nb_significant_digit();						
    if (c.accuracy==DIGIT_NOT_COMPUTED || c.accuracy==RELIABLE_RESULT)	
      c.nb_significant_digit();						
    if ((a.accuracy < b.accuracy ? (a.accuracy < c.accuracy ? a.accuracy : c.accuracy):
	 (b.accuracy < c.accuracy ? b.accuracy : c.accuracy)) >=		
	res.nb_significant_digit()+_cadna_cancel_value)			
      _cadna_cancel_count++;
  }


  return res;
}
inline float_st fma( float_st const& a,  float_st const& b,  float_st const& c){
  float_st res;
  unsigned int r1;
  unsigned int r2;
  unsigned int r3;
  float tmpa;
  float tmpb;
  float tmpc;

  res.accuracy=DIGIT_NOT_COMPUTED;

  r1 = RANDOM;
  tmpa = a.x;
  tmpb = b.x;
  tmpc = c.x;
  FLOAT_BIT_FLIP(tmpa, r1);
  FLOAT_BIT_FLIP(tmpc, r1);
  res.x = (float)fma((float)tmpa, (float)tmpb, (float)tmpc);
  FLOAT_BIT_FLIP(res.x, r1);

  r2 = RANDOM;
  tmpa = a.y;
  tmpb = b.y;
  tmpc = c.y;
  FLOAT_BIT_FLIP(tmpa, r2);
  FLOAT_BIT_FLIP(tmpc, r2);
  res.y = (float)fma((float)tmpa, (float)tmpb, (float)tmpc);
  FLOAT_BIT_FLIP(res.y, r2);


  r3 = 1^r2;
  tmpa = a.z;
  tmpb = b.z;
  tmpc = c.z;
  FLOAT_BIT_FLIP(tmpa, r3);
  FLOAT_BIT_FLIP(tmpc, r3);
  res.z = (float)fma((float)tmpa, (float)tmpb, (float)tmpc);
  FLOAT_BIT_FLIP(res.z, r3);

  if(_cadna_mul_tag) { 
    if (a.accuracy==DIGIT_NOT_COMPUTED) a.approx_digit();	 
    if (a.accuracy==0){ 
      if (b.accuracy==DIGIT_NOT_COMPUTED) b.approx_digit(); 
      if (b.accuracy==0) 
	instability(&_cadna_mul_count); 
    } 
  } 

  if(_cadna_cancel_tag){
    if (a.accuracy==DIGIT_NOT_COMPUTED || a.accuracy==RELIABLE_RESULT)	
      a.nb_significant_digit();						
    if (b.accuracy==DIGIT_NOT_COMPUTED || b.accuracy==RELIABLE_RESULT)	
      b.nb_significant_digit();						
    if (c.accuracy==DIGIT_NOT_COMPUTED || c.accuracy==RELIABLE_RESULT)	
      c.nb_significant_digit();						
    if ((a.accuracy < b.accuracy ? (a.accuracy < c.accuracy ? a.accuracy : c.accuracy):
	 (b.accuracy < c.accuracy ? b.accuracy : c.accuracy)) >=		
	res.nb_significant_digit()+_cadna_cancel_value)			
      _cadna_cancel_count++;
  }


  return res;
}




inline double_st fma_no_instab( double_st const& a,  double_st const& b,  double_st const& c){
  double_st res;
  unsigned int r1;
  unsigned int r2;
  unsigned int r3;
  double tmpa;
  double tmpb;
  double tmpc;

  res.accuracy=DIGIT_NOT_COMPUTED;

  r1 = RANDOM;
  tmpa = a.x;
  tmpb = b.x;
  tmpc = c.x;
  DOUBLE_BIT_FLIP(tmpa, r1);
  DOUBLE_BIT_FLIP(tmpc, r1);
  res.x = (double)fma((double)tmpa, (double)tmpb, (double)tmpc);
  DOUBLE_BIT_FLIP(res.x, r1);

  r2 = RANDOM;
  tmpa = a.y;
  tmpb = b.y;
  tmpc = c.y;
  DOUBLE_BIT_FLIP(tmpa, r2);
  DOUBLE_BIT_FLIP(tmpc, r2);
  res.y = (double)fma((double)tmpa, (double)tmpb, (double)tmpc);
  DOUBLE_BIT_FLIP(res.y, r2);


  r3 = 1^r2;
  tmpa = a.z;
  tmpb = b.z;
  tmpc = c.z;
  DOUBLE_BIT_FLIP(tmpa, r3);
  DOUBLE_BIT_FLIP(tmpc, r3);
  res.z = (double)fma((double)tmpa, (double)tmpb, (double)tmpc);
  DOUBLE_BIT_FLIP(res.z, r3);

  if(_cadna_mul_tag) { 
    if (a.accuracy==DIGIT_NOT_COMPUTED) a.approx_digit();	 
    if (a.accuracy==0){ 
      if (b.accuracy==DIGIT_NOT_COMPUTED) b.approx_digit(); 
      if (b.accuracy==0) 
	instability(&_cadna_mul_count); 
    } 
  } 

  if(_cadna_cancel_tag){
    if (a.accuracy==DIGIT_NOT_COMPUTED || a.accuracy==RELIABLE_RESULT)	
      a.nb_significant_digit();						
    if (b.accuracy==DIGIT_NOT_COMPUTED || b.accuracy==RELIABLE_RESULT)	
      b.nb_significant_digit();						
    if (c.accuracy==DIGIT_NOT_COMPUTED || c.accuracy==RELIABLE_RESULT)	
      c.nb_significant_digit();						
    if ((a.accuracy < b.accuracy ? (a.accuracy < c.accuracy ? a.accuracy : c.accuracy):
	 (b.accuracy < c.accuracy ? b.accuracy : c.accuracy)) >=		
	res.nb_significant_digit()+_cadna_cancel_value)			
      _cadna_cancel_count++;
  }


  return res;
}
inline double_st fma_no_instab( float_st const& a,  double_st const& b,  double_st const& c){
  double_st res;
  unsigned int r1;
  unsigned int r2;
  unsigned int r3;
  double tmpa;
  double tmpb;
  double tmpc;

  res.accuracy=DIGIT_NOT_COMPUTED;

  r1 = RANDOM;
  tmpa = a.x;
  tmpb = b.x;
  tmpc = c.x;
  DOUBLE_BIT_FLIP(tmpa, r1);
  DOUBLE_BIT_FLIP(tmpc, r1);
  res.x = (double)fma((double)tmpa, (double)tmpb, (double)tmpc);
  DOUBLE_BIT_FLIP(res.x, r1);

  r2 = RANDOM;
  tmpa = a.y;
  tmpb = b.y;
  tmpc = c.y;
  DOUBLE_BIT_FLIP(tmpa, r2);
  DOUBLE_BIT_FLIP(tmpc, r2);
  res.y = (double)fma((double)tmpa, (double)tmpb, (double)tmpc);
  DOUBLE_BIT_FLIP(res.y, r2);


  r3 = 1^r2;
  tmpa = a.z;
  tmpb = b.z;
  tmpc = c.z;
  DOUBLE_BIT_FLIP(tmpa, r3);
  DOUBLE_BIT_FLIP(tmpc, r3);
  res.z = (double)fma((double)tmpa, (double)tmpb, (double)tmpc);
  DOUBLE_BIT_FLIP(res.z, r3);

  if(_cadna_mul_tag) { 
    if (a.accuracy==DIGIT_NOT_COMPUTED) a.approx_digit();	 
    if (a.accuracy==0){ 
      if (b.accuracy==DIGIT_NOT_COMPUTED) b.approx_digit(); 
      if (b.accuracy==0) 
	instability(&_cadna_mul_count); 
    } 
  } 

  if(_cadna_cancel_tag){
    if (a.accuracy==DIGIT_NOT_COMPUTED || a.accuracy==RELIABLE_RESULT)	
      a.nb_significant_digit();						
    if (b.accuracy==DIGIT_NOT_COMPUTED || b.accuracy==RELIABLE_RESULT)	
      b.nb_significant_digit();						
    if (c.accuracy==DIGIT_NOT_COMPUTED || c.accuracy==RELIABLE_RESULT)	
      c.nb_significant_digit();						
    if ((a.accuracy < b.accuracy ? (a.accuracy < c.accuracy ? a.accuracy : c.accuracy):
	 (b.accuracy < c.accuracy ? b.accuracy : c.accuracy)) >=		
	res.nb_significant_digit()+_cadna_cancel_value)			
      _cadna_cancel_count++;
  }


  return res;
}
inline double_st fma_no_instab( double_st const& a,  float_st const& b,  double_st const& c){
  double_st res;
  unsigned int r1;
  unsigned int r2;
  unsigned int r3;
  double tmpa;
  double tmpb;
  double tmpc;

  res.accuracy=DIGIT_NOT_COMPUTED;

  r1 = RANDOM;
  tmpa = a.x;
  tmpb = b.x;
  tmpc = c.x;
  DOUBLE_BIT_FLIP(tmpa, r1);
  DOUBLE_BIT_FLIP(tmpc, r1);
  res.x = (double)fma((double)tmpa, (double)tmpb, (double)tmpc);
  DOUBLE_BIT_FLIP(res.x, r1);

  r2 = RANDOM;
  tmpa = a.y;
  tmpb = b.y;
  tmpc = c.y;
  DOUBLE_BIT_FLIP(tmpa, r2);
  DOUBLE_BIT_FLIP(tmpc, r2);
  res.y = (double)fma((double)tmpa, (double)tmpb, (double)tmpc);
  DOUBLE_BIT_FLIP(res.y, r2);


  r3 = 1^r2;
  tmpa = a.z;
  tmpb = b.z;
  tmpc = c.z;
  DOUBLE_BIT_FLIP(tmpa, r3);
  DOUBLE_BIT_FLIP(tmpc, r3);
  res.z = (double)fma((double)tmpa, (double)tmpb, (double)tmpc);
  DOUBLE_BIT_FLIP(res.z, r3);

  if(_cadna_mul_tag) { 
    if (a.accuracy==DIGIT_NOT_COMPUTED) a.approx_digit();	 
    if (a.accuracy==0){ 
      if (b.accuracy==DIGIT_NOT_COMPUTED) b.approx_digit(); 
      if (b.accuracy==0) 
	instability(&_cadna_mul_count); 
    } 
  } 

  if(_cadna_cancel_tag){
    if (a.accuracy==DIGIT_NOT_COMPUTED || a.accuracy==RELIABLE_RESULT)	
      a.nb_significant_digit();						
    if (b.accuracy==DIGIT_NOT_COMPUTED || b.accuracy==RELIABLE_RESULT)	
      b.nb_significant_digit();						
    if (c.accuracy==DIGIT_NOT_COMPUTED || c.accuracy==RELIABLE_RESULT)	
      c.nb_significant_digit();						
    if ((a.accuracy < b.accuracy ? (a.accuracy < c.accuracy ? a.accuracy : c.accuracy):
	 (b.accuracy < c.accuracy ? b.accuracy : c.accuracy)) >=		
	res.nb_significant_digit()+_cadna_cancel_value)			
      _cadna_cancel_count++;
  }


  return res;
}
inline double_st fma_no_instab( double_st const& a,  double_st const& b,  float_st const& c){
  double_st res;
  unsigned int r1;
  unsigned int r2;
  unsigned int r3;
  double tmpa;
  double tmpb;
  double tmpc;

  res.accuracy=DIGIT_NOT_COMPUTED;

  r1 = RANDOM;
  tmpa = a.x;
  tmpb = b.x;
  tmpc = c.x;
  DOUBLE_BIT_FLIP(tmpa, r1);
  DOUBLE_BIT_FLIP(tmpc, r1);
  res.x = (double)fma((double)tmpa, (double)tmpb, (double)tmpc);
  DOUBLE_BIT_FLIP(res.x, r1);

  r2 = RANDOM;
  tmpa = a.y;
  tmpb = b.y;
  tmpc = c.y;
  DOUBLE_BIT_FLIP(tmpa, r2);
  DOUBLE_BIT_FLIP(tmpc, r2);
  res.y = (double)fma((double)tmpa, (double)tmpb, (double)tmpc);
  DOUBLE_BIT_FLIP(res.y, r2);


  r3 = 1^r2;
  tmpa = a.z;
  tmpb = b.z;
  tmpc = c.z;
  DOUBLE_BIT_FLIP(tmpa, r3);
  DOUBLE_BIT_FLIP(tmpc, r3);
  res.z = (double)fma((double)tmpa, (double)tmpb, (double)tmpc);
  DOUBLE_BIT_FLIP(res.z, r3);

  if(_cadna_mul_tag) { 
    if (a.accuracy==DIGIT_NOT_COMPUTED) a.approx_digit();	 
    if (a.accuracy==0){ 
      if (b.accuracy==DIGIT_NOT_COMPUTED) b.approx_digit(); 
      if (b.accuracy==0) 
	instability(&_cadna_mul_count); 
    } 
  } 

  if(_cadna_cancel_tag){
    if (a.accuracy==DIGIT_NOT_COMPUTED || a.accuracy==RELIABLE_RESULT)	
      a.nb_significant_digit();						
    if (b.accuracy==DIGIT_NOT_COMPUTED || b.accuracy==RELIABLE_RESULT)	
      b.nb_significant_digit();						
    if (c.accuracy==DIGIT_NOT_COMPUTED || c.accuracy==RELIABLE_RESULT)	
      c.nb_significant_digit();						
    if ((a.accuracy < b.accuracy ? (a.accuracy < c.accuracy ? a.accuracy : c.accuracy):
	 (b.accuracy < c.accuracy ? b.accuracy : c.accuracy)) >=		
	res.nb_significant_digit()+_cadna_cancel_value)			
      _cadna_cancel_count++;
  }


  return res;
}
inline double_st fma_no_instab( float_st const& a,  float_st const& b,  double_st const& c){
  double_st res;
  unsigned int r1;
  unsigned int r2;
  unsigned int r3;
  double tmpa;
  double tmpb;
  double tmpc;

  res.accuracy=DIGIT_NOT_COMPUTED;

  r1 = RANDOM;
  tmpa = a.x;
  tmpb = b.x;
  tmpc = c.x;
  DOUBLE_BIT_FLIP(tmpa, r1);
  DOUBLE_BIT_FLIP(tmpc, r1);
  res.x = (double)fma((double)tmpa, (double)tmpb, (double)tmpc);
  DOUBLE_BIT_FLIP(res.x, r1);

  r2 = RANDOM;
  tmpa = a.y;
  tmpb = b.y;
  tmpc = c.y;
  DOUBLE_BIT_FLIP(tmpa, r2);
  DOUBLE_BIT_FLIP(tmpc, r2);
  res.y = (double)fma((double)tmpa, (double)tmpb, (double)tmpc);
  DOUBLE_BIT_FLIP(res.y, r2);


  r3 = 1^r2;
  tmpa = a.z;
  tmpb = b.z;
  tmpc = c.z;
  DOUBLE_BIT_FLIP(tmpa, r3);
  DOUBLE_BIT_FLIP(tmpc, r3);
  res.z = (double)fma((double)tmpa, (double)tmpb, (double)tmpc);
  DOUBLE_BIT_FLIP(res.z, r3);

  if(_cadna_mul_tag) { 
    if (a.accuracy==DIGIT_NOT_COMPUTED) a.approx_digit();	 
    if (a.accuracy==0){ 
      if (b.accuracy==DIGIT_NOT_COMPUTED) b.approx_digit(); 
      if (b.accuracy==0) 
	instability(&_cadna_mul_count); 
    } 
  } 

  if(_cadna_cancel_tag){
    if (a.accuracy==DIGIT_NOT_COMPUTED || a.accuracy==RELIABLE_RESULT)	
      a.nb_significant_digit();						
    if (b.accuracy==DIGIT_NOT_COMPUTED || b.accuracy==RELIABLE_RESULT)	
      b.nb_significant_digit();						
    if (c.accuracy==DIGIT_NOT_COMPUTED || c.accuracy==RELIABLE_RESULT)	
      c.nb_significant_digit();						
    if ((a.accuracy < b.accuracy ? (a.accuracy < c.accuracy ? a.accuracy : c.accuracy):
	 (b.accuracy < c.accuracy ? b.accuracy : c.accuracy)) >=		
	res.nb_significant_digit()+_cadna_cancel_value)			
      _cadna_cancel_count++;
  }


  return res;
}
inline double_st fma_no_instab( float_st const& a,  double_st const& b,  float_st const& c){
  double_st res;
  unsigned int r1;
  unsigned int r2;
  unsigned int r3;
  double tmpa;
  double tmpb;
  double tmpc;

  res.accuracy=DIGIT_NOT_COMPUTED;

  r1 = RANDOM;
  tmpa = a.x;
  tmpb = b.x;
  tmpc = c.x;
  DOUBLE_BIT_FLIP(tmpa, r1);
  DOUBLE_BIT_FLIP(tmpc, r1);
  res.x = (double)fma((double)tmpa, (double)tmpb, (double)tmpc);
  DOUBLE_BIT_FLIP(res.x, r1);

  r2 = RANDOM;
  tmpa = a.y;
  tmpb = b.y;
  tmpc = c.y;
  DOUBLE_BIT_FLIP(tmpa, r2);
  DOUBLE_BIT_FLIP(tmpc, r2);
  res.y = (double)fma((double)tmpa, (double)tmpb, (double)tmpc);
  DOUBLE_BIT_FLIP(res.y, r2);


  r3 = 1^r2;
  tmpa = a.z;
  tmpb = b.z;
  tmpc = c.z;
  DOUBLE_BIT_FLIP(tmpa, r3);
  DOUBLE_BIT_FLIP(tmpc, r3);
  res.z = (double)fma((double)tmpa, (double)tmpb, (double)tmpc);
  DOUBLE_BIT_FLIP(res.z, r3);

  if(_cadna_mul_tag) { 
    if (a.accuracy==DIGIT_NOT_COMPUTED) a.approx_digit();	 
    if (a.accuracy==0){ 
      if (b.accuracy==DIGIT_NOT_COMPUTED) b.approx_digit(); 
      if (b.accuracy==0) 
	instability(&_cadna_mul_count); 
    } 
  } 

  if(_cadna_cancel_tag){
    if (a.accuracy==DIGIT_NOT_COMPUTED || a.accuracy==RELIABLE_RESULT)	
      a.nb_significant_digit();						
    if (b.accuracy==DIGIT_NOT_COMPUTED || b.accuracy==RELIABLE_RESULT)	
      b.nb_significant_digit();						
    if (c.accuracy==DIGIT_NOT_COMPUTED || c.accuracy==RELIABLE_RESULT)	
      c.nb_significant_digit();						
    if ((a.accuracy < b.accuracy ? (a.accuracy < c.accuracy ? a.accuracy : c.accuracy):
	 (b.accuracy < c.accuracy ? b.accuracy : c.accuracy)) >=		
	res.nb_significant_digit()+_cadna_cancel_value)			
      _cadna_cancel_count++;
  }


  return res;
}
inline double_st fma_no_instab( double_st const& a,  float_st const& b,  float_st const& c){
  double_st res;
  unsigned int r1;
  unsigned int r2;
  unsigned int r3;
  double tmpa;
  double tmpb;
  double tmpc;

  res.accuracy=DIGIT_NOT_COMPUTED;

  r1 = RANDOM;
  tmpa = a.x;
  tmpb = b.x;
  tmpc = c.x;
  DOUBLE_BIT_FLIP(tmpa, r1);
  DOUBLE_BIT_FLIP(tmpc, r1);
  res.x = (double)fma((double)tmpa, (double)tmpb, (double)tmpc);
  DOUBLE_BIT_FLIP(res.x, r1);

  r2 = RANDOM;
  tmpa = a.y;
  tmpb = b.y;
  tmpc = c.y;
  DOUBLE_BIT_FLIP(tmpa, r2);
  DOUBLE_BIT_FLIP(tmpc, r2);
  res.y = (double)fma((double)tmpa, (double)tmpb, (double)tmpc);
  DOUBLE_BIT_FLIP(res.y, r2);


  r3 = 1^r2;
  tmpa = a.z;
  tmpb = b.z;
  tmpc = c.z;
  DOUBLE_BIT_FLIP(tmpa, r3);
  DOUBLE_BIT_FLIP(tmpc, r3);
  res.z = (double)fma((double)tmpa, (double)tmpb, (double)tmpc);
  DOUBLE_BIT_FLIP(res.z, r3);

  if(_cadna_mul_tag) { 
    if (a.accuracy==DIGIT_NOT_COMPUTED) a.approx_digit();	 
    if (a.accuracy==0){ 
      if (b.accuracy==DIGIT_NOT_COMPUTED) b.approx_digit(); 
      if (b.accuracy==0) 
	instability(&_cadna_mul_count); 
    } 
  } 

  if(_cadna_cancel_tag){
    if (a.accuracy==DIGIT_NOT_COMPUTED || a.accuracy==RELIABLE_RESULT)	
      a.nb_significant_digit();						
    if (b.accuracy==DIGIT_NOT_COMPUTED || b.accuracy==RELIABLE_RESULT)	
      b.nb_significant_digit();						
    if (c.accuracy==DIGIT_NOT_COMPUTED || c.accuracy==RELIABLE_RESULT)	
      c.nb_significant_digit();						
    if ((a.accuracy < b.accuracy ? (a.accuracy < c.accuracy ? a.accuracy : c.accuracy):
	 (b.accuracy < c.accuracy ? b.accuracy : c.accuracy)) >=		
	res.nb_significant_digit()+_cadna_cancel_value)			
      _cadna_cancel_count++;
  }


  return res;
}
inline float_st fma_no_instab( float_st const& a,  float_st const& b,  float_st const& c){
  float_st res;
  unsigned int r1;
  unsigned int r2;
  unsigned int r3;
  float tmpa;
  float tmpb;
  float tmpc;

  res.accuracy=DIGIT_NOT_COMPUTED;

  r1 = RANDOM;
  tmpa = a.x;
  tmpb = b.x;
  tmpc = c.x;
  FLOAT_BIT_FLIP(tmpa, r1);
  FLOAT_BIT_FLIP(tmpc, r1);
  res.x = (float)fma((float)tmpa, (float)tmpb, (float)tmpc);
  FLOAT_BIT_FLIP(res.x, r1);

  r2 = RANDOM;
  tmpa = a.y;
  tmpb = b.y;
  tmpc = c.y;
  FLOAT_BIT_FLIP(tmpa, r2);
  FLOAT_BIT_FLIP(tmpc, r2);
  res.y = (float)fma((float)tmpa, (float)tmpb, (float)tmpc);
  FLOAT_BIT_FLIP(res.y, r2);


  r3 = 1^r2;
  tmpa = a.z;
  tmpb = b.z;
  tmpc = c.z;
  FLOAT_BIT_FLIP(tmpa, r3);
  FLOAT_BIT_FLIP(tmpc, r3);
  res.z = (float)fma((float)tmpa, (float)tmpb, (float)tmpc);
  FLOAT_BIT_FLIP(res.z, r3);

  if(_cadna_mul_tag) { 
    if (a.accuracy==DIGIT_NOT_COMPUTED) a.approx_digit();	 
    if (a.accuracy==0){ 
      if (b.accuracy==DIGIT_NOT_COMPUTED) b.approx_digit(); 
      if (b.accuracy==0) 
	instability(&_cadna_mul_count); 
    } 
  } 

  if(_cadna_cancel_tag){
    if (a.accuracy==DIGIT_NOT_COMPUTED || a.accuracy==RELIABLE_RESULT)	
      a.nb_significant_digit();						
    if (b.accuracy==DIGIT_NOT_COMPUTED || b.accuracy==RELIABLE_RESULT)	
      b.nb_significant_digit();						
    if (c.accuracy==DIGIT_NOT_COMPUTED || c.accuracy==RELIABLE_RESULT)	
      c.nb_significant_digit();						
    if ((a.accuracy < b.accuracy ? (a.accuracy < c.accuracy ? a.accuracy : c.accuracy):
	 (b.accuracy < c.accuracy ? b.accuracy : c.accuracy)) >=		
	res.nb_significant_digit()+_cadna_cancel_value)			
      _cadna_cancel_count++;
  }


  return res;
}

#endif // CADNA_FMA
