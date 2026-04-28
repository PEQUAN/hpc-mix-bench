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
#include <cadna_plugin.h>
#ifdef CADNA_HALF_EMULATION
#include <half.hpp>
#include <cadna_half.hpp>
using half_float::half;
#endif
#include <cmath>
///////////////////////////////////////////////////
//****m* cadna_mul/operator*
//    NAME
//      operator*
//    SYNOPSIS
//      res = a * b
//    FUNCTION
//      Defines all the functions involving at least one argument
//      of stochastic type which overload the "*" operator
//      in a statement such as "a*b".
//
//    INPUTS
//      a           - an integer, a float, a double or a stochastic number
//      b           - an integer, a float, a double or a stochastic number
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







/////////////////////////////////////////////////////////////////////////








inline float_st& float_st::operator*=(const double& a){
  unsigned int random;
  float tmp;

  accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = x;
  FLOAT_BIT_FLIP(tmp, random);
  x = tmp;
  tmp = a;
  x *= tmp;
  FLOAT_BIT_FLIP(x, random);

  random = RANDOM;
  tmp = y;
  FLOAT_BIT_FLIP(tmp, random);
  y = tmp;
  tmp = a;
  y *= tmp;
  FLOAT_BIT_FLIP(y, random);

  random = 1^random;
  tmp = z;
  FLOAT_BIT_FLIP(tmp, random);
  z = tmp;
  tmp = a;
  z *= tmp;
  FLOAT_BIT_FLIP(z, random);

  return *this;
}
inline float_st& float_st::operator*=(const float& a){
  unsigned int random;
  float tmp;

  accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = x;
  FLOAT_BIT_FLIP(tmp, random);
  x = tmp;
  tmp = a;
  x *= tmp;
  FLOAT_BIT_FLIP(x, random);

  random = RANDOM;
  tmp = y;
  FLOAT_BIT_FLIP(tmp, random);
  y = tmp;
  tmp = a;
  y *= tmp;
  FLOAT_BIT_FLIP(y, random);

  random = 1^random;
  tmp = z;
  FLOAT_BIT_FLIP(tmp, random);
  z = tmp;
  tmp = a;
  z *= tmp;
  FLOAT_BIT_FLIP(z, random);

  return *this;
}
inline float_st& float_st::operator*=(const unsigned long long& a){
  unsigned int random;
  float tmp;

  accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = x;
  FLOAT_BIT_FLIP(tmp, random);
  x = tmp;
  tmp = a;
  x *= tmp;
  FLOAT_BIT_FLIP(x, random);

  random = RANDOM;
  tmp = y;
  FLOAT_BIT_FLIP(tmp, random);
  y = tmp;
  tmp = a;
  y *= tmp;
  FLOAT_BIT_FLIP(y, random);

  random = 1^random;
  tmp = z;
  FLOAT_BIT_FLIP(tmp, random);
  z = tmp;
  tmp = a;
  z *= tmp;
  FLOAT_BIT_FLIP(z, random);

  return *this;
}
inline float_st& float_st::operator*=(const long long& a){
  unsigned int random;
  float tmp;

  accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = x;
  FLOAT_BIT_FLIP(tmp, random);
  x = tmp;
  tmp = a;
  x *= tmp;
  FLOAT_BIT_FLIP(x, random);

  random = RANDOM;
  tmp = y;
  FLOAT_BIT_FLIP(tmp, random);
  y = tmp;
  tmp = a;
  y *= tmp;
  FLOAT_BIT_FLIP(y, random);

  random = 1^random;
  tmp = z;
  FLOAT_BIT_FLIP(tmp, random);
  z = tmp;
  tmp = a;
  z *= tmp;
  FLOAT_BIT_FLIP(z, random);

  return *this;
}
inline float_st& float_st::operator*=(const unsigned long& a){
  unsigned int random;
  float tmp;

  accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = x;
  FLOAT_BIT_FLIP(tmp, random);
  x = tmp;
  tmp = a;
  x *= tmp;
  FLOAT_BIT_FLIP(x, random);

  random = RANDOM;
  tmp = y;
  FLOAT_BIT_FLIP(tmp, random);
  y = tmp;
  tmp = a;
  y *= tmp;
  FLOAT_BIT_FLIP(y, random);

  random = 1^random;
  tmp = z;
  FLOAT_BIT_FLIP(tmp, random);
  z = tmp;
  tmp = a;
  z *= tmp;
  FLOAT_BIT_FLIP(z, random);

  return *this;
}
inline float_st& float_st::operator*=(const long& a){
  unsigned int random;
  float tmp;

  accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = x;
  FLOAT_BIT_FLIP(tmp, random);
  x = tmp;
  tmp = a;
  x *= tmp;
  FLOAT_BIT_FLIP(x, random);

  random = RANDOM;
  tmp = y;
  FLOAT_BIT_FLIP(tmp, random);
  y = tmp;
  tmp = a;
  y *= tmp;
  FLOAT_BIT_FLIP(y, random);

  random = 1^random;
  tmp = z;
  FLOAT_BIT_FLIP(tmp, random);
  z = tmp;
  tmp = a;
  z *= tmp;
  FLOAT_BIT_FLIP(z, random);

  return *this;
}
inline float_st& float_st::operator*=(const unsigned int& a){
  unsigned int random;
  float tmp;

  accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = x;
  FLOAT_BIT_FLIP(tmp, random);
  x = tmp;
  tmp = a;
  x *= tmp;
  FLOAT_BIT_FLIP(x, random);

  random = RANDOM;
  tmp = y;
  FLOAT_BIT_FLIP(tmp, random);
  y = tmp;
  tmp = a;
  y *= tmp;
  FLOAT_BIT_FLIP(y, random);

  random = 1^random;
  tmp = z;
  FLOAT_BIT_FLIP(tmp, random);
  z = tmp;
  tmp = a;
  z *= tmp;
  FLOAT_BIT_FLIP(z, random);

  return *this;
}
inline float_st& float_st::operator*=(const int& a){
  unsigned int random;
  float tmp;

  accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = x;
  FLOAT_BIT_FLIP(tmp, random);
  x = tmp;
  tmp = a;
  x *= tmp;
  FLOAT_BIT_FLIP(x, random);

  random = RANDOM;
  tmp = y;
  FLOAT_BIT_FLIP(tmp, random);
  y = tmp;
  tmp = a;
  y *= tmp;
  FLOAT_BIT_FLIP(y, random);

  random = 1^random;
  tmp = z;
  FLOAT_BIT_FLIP(tmp, random);
  z = tmp;
  tmp = a;
  z *= tmp;
  FLOAT_BIT_FLIP(z, random);

  return *this;
}
inline float_st& float_st::operator*=(const unsigned short& a){
  unsigned int random;
  float tmp;

  accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = x;
  FLOAT_BIT_FLIP(tmp, random);
  x = tmp;
  tmp = a;
  x *= tmp;
  FLOAT_BIT_FLIP(x, random);

  random = RANDOM;
  tmp = y;
  FLOAT_BIT_FLIP(tmp, random);
  y = tmp;
  tmp = a;
  y *= tmp;
  FLOAT_BIT_FLIP(y, random);

  random = 1^random;
  tmp = z;
  FLOAT_BIT_FLIP(tmp, random);
  z = tmp;
  tmp = a;
  z *= tmp;
  FLOAT_BIT_FLIP(z, random);

  return *this;
}
inline float_st& float_st::operator*=(const short& a){
  unsigned int random;
  float tmp;

  accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = x;
  FLOAT_BIT_FLIP(tmp, random);
  x = tmp;
  tmp = a;
  x *= tmp;
  FLOAT_BIT_FLIP(x, random);

  random = RANDOM;
  tmp = y;
  FLOAT_BIT_FLIP(tmp, random);
  y = tmp;
  tmp = a;
  y *= tmp;
  FLOAT_BIT_FLIP(y, random);

  random = 1^random;
  tmp = z;
  FLOAT_BIT_FLIP(tmp, random);
  z = tmp;
  tmp = a;
  z *= tmp;
  FLOAT_BIT_FLIP(z, random);

  return *this;
}
inline float_st& float_st::operator*=(const unsigned char& a){
  unsigned int random;
  float tmp;

  accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = x;
  FLOAT_BIT_FLIP(tmp, random);
  x = tmp;
  tmp = a;
  x *= tmp;
  FLOAT_BIT_FLIP(x, random);

  random = RANDOM;
  tmp = y;
  FLOAT_BIT_FLIP(tmp, random);
  y = tmp;
  tmp = a;
  y *= tmp;
  FLOAT_BIT_FLIP(y, random);

  random = 1^random;
  tmp = z;
  FLOAT_BIT_FLIP(tmp, random);
  z = tmp;
  tmp = a;
  z *= tmp;
  FLOAT_BIT_FLIP(z, random);

  return *this;
}
inline float_st& float_st::operator*=(const char& a){
  unsigned int random;
  float tmp;

  accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = x;
  FLOAT_BIT_FLIP(tmp, random);
  x = tmp;
  tmp = a;
  x *= tmp;
  FLOAT_BIT_FLIP(x, random);

  random = RANDOM;
  tmp = y;
  FLOAT_BIT_FLIP(tmp, random);
  y = tmp;
  tmp = a;
  y *= tmp;
  FLOAT_BIT_FLIP(y, random);

  random = 1^random;
  tmp = z;
  FLOAT_BIT_FLIP(tmp, random);
  z = tmp;
  tmp = a;
  z *= tmp;
  FLOAT_BIT_FLIP(z, random);

  return *this;
}
#ifdef CADNA_HALF_EMULATION
inline float_st& float_st::operator*=(const half& a){
  unsigned int random;
  float tmp;

  accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = x;
  FLOAT_BIT_FLIP(tmp, random);
  x = tmp;
  tmp = a;
  x *= tmp;
  FLOAT_BIT_FLIP(x, random);

  random = RANDOM;
  tmp = y;
  FLOAT_BIT_FLIP(tmp, random);
  y = tmp;
  tmp = a;
  y *= tmp;
  FLOAT_BIT_FLIP(y, random);

  random = 1^random;
  tmp = z;
  FLOAT_BIT_FLIP(tmp, random);
  z = tmp;
  tmp = a;
  z *= tmp;
  FLOAT_BIT_FLIP(z, random);

  return *this;
}
#endif // CADNA_HALF_EMULATION

#ifdef CADNA_HALF_NATIVE
inline float_st& float_st::operator*=(const half& a){
  unsigned int random;
  float tmp;

  accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = x;
  FLOAT_BIT_FLIP(tmp, random);
  x = tmp;
  tmp = a;
  x *= tmp;
  FLOAT_BIT_FLIP(x, random);

  random = RANDOM;
  tmp = y;
  FLOAT_BIT_FLIP(tmp, random);
  y = tmp;
  tmp = a;
  y *= tmp;
  FLOAT_BIT_FLIP(y, random);

  random = 1^random;
  tmp = z;
  FLOAT_BIT_FLIP(tmp, random);
  z = tmp;
  tmp = a;
  z *= tmp;
  FLOAT_BIT_FLIP(z, random);

  return *this;
}
#endif //

inline double_st& double_st::operator*=(const double& a){
  unsigned int random;
  double tmp;

  accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = x;
  DOUBLE_BIT_FLIP(tmp, random);
  x = tmp;
  tmp = a;
  x *= tmp;
  DOUBLE_BIT_FLIP(x, random);

  random = RANDOM;
  tmp = y;
  DOUBLE_BIT_FLIP(tmp, random);
  y = tmp;
  tmp = a;
  y *= tmp;
  DOUBLE_BIT_FLIP(y, random);

  random = 1^random;
  tmp = z;
  DOUBLE_BIT_FLIP(tmp, random);
  z = tmp;
  tmp = a;
  z *= tmp;
  DOUBLE_BIT_FLIP(z, random);

  return *this;
}
inline double_st& double_st::operator*=(const float& a){
  unsigned int random;
  double tmp;

  accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = x;
  DOUBLE_BIT_FLIP(tmp, random);
  x = tmp;
  tmp = a;
  x *= tmp;
  DOUBLE_BIT_FLIP(x, random);

  random = RANDOM;
  tmp = y;
  DOUBLE_BIT_FLIP(tmp, random);
  y = tmp;
  tmp = a;
  y *= tmp;
  DOUBLE_BIT_FLIP(y, random);

  random = 1^random;
  tmp = z;
  DOUBLE_BIT_FLIP(tmp, random);
  z = tmp;
  tmp = a;
  z *= tmp;
  DOUBLE_BIT_FLIP(z, random);

  return *this;
}
inline double_st& double_st::operator*=(const unsigned long long& a){
  unsigned int random;
  double tmp;

  accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = x;
  DOUBLE_BIT_FLIP(tmp, random);
  x = tmp;
  tmp = a;
  x *= tmp;
  DOUBLE_BIT_FLIP(x, random);

  random = RANDOM;
  tmp = y;
  DOUBLE_BIT_FLIP(tmp, random);
  y = tmp;
  tmp = a;
  y *= tmp;
  DOUBLE_BIT_FLIP(y, random);

  random = 1^random;
  tmp = z;
  DOUBLE_BIT_FLIP(tmp, random);
  z = tmp;
  tmp = a;
  z *= tmp;
  DOUBLE_BIT_FLIP(z, random);

  return *this;
}
inline double_st& double_st::operator*=(const long long& a){
  unsigned int random;
  double tmp;

  accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = x;
  DOUBLE_BIT_FLIP(tmp, random);
  x = tmp;
  tmp = a;
  x *= tmp;
  DOUBLE_BIT_FLIP(x, random);

  random = RANDOM;
  tmp = y;
  DOUBLE_BIT_FLIP(tmp, random);
  y = tmp;
  tmp = a;
  y *= tmp;
  DOUBLE_BIT_FLIP(y, random);

  random = 1^random;
  tmp = z;
  DOUBLE_BIT_FLIP(tmp, random);
  z = tmp;
  tmp = a;
  z *= tmp;
  DOUBLE_BIT_FLIP(z, random);

  return *this;
}
inline double_st& double_st::operator*=(const unsigned long& a){
  unsigned int random;
  double tmp;

  accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = x;
  DOUBLE_BIT_FLIP(tmp, random);
  x = tmp;
  tmp = a;
  x *= tmp;
  DOUBLE_BIT_FLIP(x, random);

  random = RANDOM;
  tmp = y;
  DOUBLE_BIT_FLIP(tmp, random);
  y = tmp;
  tmp = a;
  y *= tmp;
  DOUBLE_BIT_FLIP(y, random);

  random = 1^random;
  tmp = z;
  DOUBLE_BIT_FLIP(tmp, random);
  z = tmp;
  tmp = a;
  z *= tmp;
  DOUBLE_BIT_FLIP(z, random);

  return *this;
}
inline double_st& double_st::operator*=(const long& a){
  unsigned int random;
  double tmp;

  accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = x;
  DOUBLE_BIT_FLIP(tmp, random);
  x = tmp;
  tmp = a;
  x *= tmp;
  DOUBLE_BIT_FLIP(x, random);

  random = RANDOM;
  tmp = y;
  DOUBLE_BIT_FLIP(tmp, random);
  y = tmp;
  tmp = a;
  y *= tmp;
  DOUBLE_BIT_FLIP(y, random);

  random = 1^random;
  tmp = z;
  DOUBLE_BIT_FLIP(tmp, random);
  z = tmp;
  tmp = a;
  z *= tmp;
  DOUBLE_BIT_FLIP(z, random);

  return *this;
}
inline double_st& double_st::operator*=(const unsigned int& a){
  unsigned int random;
  double tmp;

  accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = x;
  DOUBLE_BIT_FLIP(tmp, random);
  x = tmp;
  tmp = a;
  x *= tmp;
  DOUBLE_BIT_FLIP(x, random);

  random = RANDOM;
  tmp = y;
  DOUBLE_BIT_FLIP(tmp, random);
  y = tmp;
  tmp = a;
  y *= tmp;
  DOUBLE_BIT_FLIP(y, random);

  random = 1^random;
  tmp = z;
  DOUBLE_BIT_FLIP(tmp, random);
  z = tmp;
  tmp = a;
  z *= tmp;
  DOUBLE_BIT_FLIP(z, random);

  return *this;
}
inline double_st& double_st::operator*=(const int& a){
  unsigned int random;
  double tmp;

  accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = x;
  DOUBLE_BIT_FLIP(tmp, random);
  x = tmp;
  tmp = a;
  x *= tmp;
  DOUBLE_BIT_FLIP(x, random);

  random = RANDOM;
  tmp = y;
  DOUBLE_BIT_FLIP(tmp, random);
  y = tmp;
  tmp = a;
  y *= tmp;
  DOUBLE_BIT_FLIP(y, random);

  random = 1^random;
  tmp = z;
  DOUBLE_BIT_FLIP(tmp, random);
  z = tmp;
  tmp = a;
  z *= tmp;
  DOUBLE_BIT_FLIP(z, random);

  return *this;
}
inline double_st& double_st::operator*=(const unsigned short& a){
  unsigned int random;
  double tmp;

  accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = x;
  DOUBLE_BIT_FLIP(tmp, random);
  x = tmp;
  tmp = a;
  x *= tmp;
  DOUBLE_BIT_FLIP(x, random);

  random = RANDOM;
  tmp = y;
  DOUBLE_BIT_FLIP(tmp, random);
  y = tmp;
  tmp = a;
  y *= tmp;
  DOUBLE_BIT_FLIP(y, random);

  random = 1^random;
  tmp = z;
  DOUBLE_BIT_FLIP(tmp, random);
  z = tmp;
  tmp = a;
  z *= tmp;
  DOUBLE_BIT_FLIP(z, random);

  return *this;
}
inline double_st& double_st::operator*=(const short& a){
  unsigned int random;
  double tmp;

  accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = x;
  DOUBLE_BIT_FLIP(tmp, random);
  x = tmp;
  tmp = a;
  x *= tmp;
  DOUBLE_BIT_FLIP(x, random);

  random = RANDOM;
  tmp = y;
  DOUBLE_BIT_FLIP(tmp, random);
  y = tmp;
  tmp = a;
  y *= tmp;
  DOUBLE_BIT_FLIP(y, random);

  random = 1^random;
  tmp = z;
  DOUBLE_BIT_FLIP(tmp, random);
  z = tmp;
  tmp = a;
  z *= tmp;
  DOUBLE_BIT_FLIP(z, random);

  return *this;
}
inline double_st& double_st::operator*=(const unsigned char& a){
  unsigned int random;
  double tmp;

  accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = x;
  DOUBLE_BIT_FLIP(tmp, random);
  x = tmp;
  tmp = a;
  x *= tmp;
  DOUBLE_BIT_FLIP(x, random);

  random = RANDOM;
  tmp = y;
  DOUBLE_BIT_FLIP(tmp, random);
  y = tmp;
  tmp = a;
  y *= tmp;
  DOUBLE_BIT_FLIP(y, random);

  random = 1^random;
  tmp = z;
  DOUBLE_BIT_FLIP(tmp, random);
  z = tmp;
  tmp = a;
  z *= tmp;
  DOUBLE_BIT_FLIP(z, random);

  return *this;
}
inline double_st& double_st::operator*=(const char& a){
  unsigned int random;
  double tmp;

  accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = x;
  DOUBLE_BIT_FLIP(tmp, random);
  x = tmp;
  tmp = a;
  x *= tmp;
  DOUBLE_BIT_FLIP(x, random);

  random = RANDOM;
  tmp = y;
  DOUBLE_BIT_FLIP(tmp, random);
  y = tmp;
  tmp = a;
  y *= tmp;
  DOUBLE_BIT_FLIP(y, random);

  random = 1^random;
  tmp = z;
  DOUBLE_BIT_FLIP(tmp, random);
  z = tmp;
  tmp = a;
  z *= tmp;
  DOUBLE_BIT_FLIP(z, random);

  return *this;
}
#ifdef CADNA_HALF_EMULATION
inline double_st& double_st::operator*=(const half& a){
  unsigned int random;
  double tmp;

  accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = x;
  DOUBLE_BIT_FLIP(tmp, random);
  x = tmp;
  tmp = a;
  x *= tmp;
  DOUBLE_BIT_FLIP(x, random);

  random = RANDOM;
  tmp = y;
  DOUBLE_BIT_FLIP(tmp, random);
  y = tmp;
  tmp = a;
  y *= tmp;
  DOUBLE_BIT_FLIP(y, random);

  random = 1^random;
  tmp = z;
  DOUBLE_BIT_FLIP(tmp, random);
  z = tmp;
  tmp = a;
  z *= tmp;
  DOUBLE_BIT_FLIP(z, random);

  return *this;
}
#endif // CADNA_HALF_EMULATION

#ifdef CADNA_HALF_NATIVE
inline double_st& double_st::operator*=(const half& a){
  unsigned int random;
  double tmp;

  accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = x;
  DOUBLE_BIT_FLIP(tmp, random);
  x = tmp;
  tmp = a;
  x *= tmp;
  DOUBLE_BIT_FLIP(x, random);

  random = RANDOM;
  tmp = y;
  DOUBLE_BIT_FLIP(tmp, random);
  y = tmp;
  tmp = a;
  y *= tmp;
  DOUBLE_BIT_FLIP(y, random);

  random = 1^random;
  tmp = z;
  DOUBLE_BIT_FLIP(tmp, random);
  z = tmp;
  tmp = a;
  z *= tmp;
  DOUBLE_BIT_FLIP(z, random);

  return *this;
}
#endif


#ifdef CADNA_HALF_EMULATION
inline half_st& half_st::operator*=(const double& a){
  unsigned int random;
  half tmp;

  accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = x;
  HALF_BIT_FLIP(tmp, random);
  x = tmp;
  x *= a;
  HALF_BIT_FLIP(x, random);

  random = RANDOM;
  tmp = y;
  HALF_BIT_FLIP(tmp, random);
  y = tmp;
  y *= a;
  HALF_BIT_FLIP(y, random);

  random = 1^random;
  tmp = z;
  HALF_BIT_FLIP(tmp, random);
  z = tmp;
  z *= a;
  HALF_BIT_FLIP(z, random);
  
  
  
  if(_cadna_half_overflow_tag){
   if (std::isinf(x) or std::isinf(y) or std::isinf(z)){
    instability(&_cadna_half_overflow_count);
    }
}



  return *this;
  

}
inline half_st& half_st::operator*=(const float& a){
  unsigned int random;
  half tmp;

  accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = x;
  HALF_BIT_FLIP(tmp, random);
  x = tmp;
  x *= a;
  HALF_BIT_FLIP(x, random);

  random = RANDOM;
  tmp = y;
  HALF_BIT_FLIP(tmp, random);
  y = tmp;
  y *= a;
  HALF_BIT_FLIP(y, random);

  random = 1^random;
  tmp = z;
  HALF_BIT_FLIP(tmp, random);
  z = tmp;
  z *= a;
  HALF_BIT_FLIP(z, random);
  
  
  
  if(_cadna_half_overflow_tag){
   if (std::isinf(x) or std::isinf(y) or std::isinf(z)){
    instability(&_cadna_half_overflow_count);
    }
}



  return *this;
  

}
inline half_st& half_st::operator*=(const unsigned long long& a){
  unsigned int random;
  half tmp;

  accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = x;
  HALF_BIT_FLIP(tmp, random);
  x = tmp;
  x *= a;
  HALF_BIT_FLIP(x, random);

  random = RANDOM;
  tmp = y;
  HALF_BIT_FLIP(tmp, random);
  y = tmp;
  y *= a;
  HALF_BIT_FLIP(y, random);

  random = 1^random;
  tmp = z;
  HALF_BIT_FLIP(tmp, random);
  z = tmp;
  z *= a;
  HALF_BIT_FLIP(z, random);
  
  
  
  if(_cadna_half_overflow_tag){
   if (std::isinf(x) or std::isinf(y) or std::isinf(z)){
    instability(&_cadna_half_overflow_count);
    }
}



  return *this;
  

}
inline half_st& half_st::operator*=(const long long& a){
  unsigned int random;
  half tmp;

  accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = x;
  HALF_BIT_FLIP(tmp, random);
  x = tmp;
  x *= a;
  HALF_BIT_FLIP(x, random);

  random = RANDOM;
  tmp = y;
  HALF_BIT_FLIP(tmp, random);
  y = tmp;
  y *= a;
  HALF_BIT_FLIP(y, random);

  random = 1^random;
  tmp = z;
  HALF_BIT_FLIP(tmp, random);
  z = tmp;
  z *= a;
  HALF_BIT_FLIP(z, random);
  
  
  
  if(_cadna_half_overflow_tag){
   if (std::isinf(x) or std::isinf(y) or std::isinf(z)){
    instability(&_cadna_half_overflow_count);
    }
}



  return *this;
  

}
inline half_st& half_st::operator*=(const unsigned long& a){
  unsigned int random;
  half tmp;

  accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = x;
  HALF_BIT_FLIP(tmp, random);
  x = tmp;
  x *= a;
  HALF_BIT_FLIP(x, random);

  random = RANDOM;
  tmp = y;
  HALF_BIT_FLIP(tmp, random);
  y = tmp;
  y *= a;
  HALF_BIT_FLIP(y, random);

  random = 1^random;
  tmp = z;
  HALF_BIT_FLIP(tmp, random);
  z = tmp;
  z *= a;
  HALF_BIT_FLIP(z, random);
  
  
  
  if(_cadna_half_overflow_tag){
   if (std::isinf(x) or std::isinf(y) or std::isinf(z)){
    instability(&_cadna_half_overflow_count);
    }
}



  return *this;
  

}
inline half_st& half_st::operator*=(const long& a){
  unsigned int random;
  half tmp;

  accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = x;
  HALF_BIT_FLIP(tmp, random);
  x = tmp;
  x *= a;
  HALF_BIT_FLIP(x, random);

  random = RANDOM;
  tmp = y;
  HALF_BIT_FLIP(tmp, random);
  y = tmp;
  y *= a;
  HALF_BIT_FLIP(y, random);

  random = 1^random;
  tmp = z;
  HALF_BIT_FLIP(tmp, random);
  z = tmp;
  z *= a;
  HALF_BIT_FLIP(z, random);
  
  
  
  if(_cadna_half_overflow_tag){
   if (std::isinf(x) or std::isinf(y) or std::isinf(z)){
    instability(&_cadna_half_overflow_count);
    }
}



  return *this;
  

}
inline half_st& half_st::operator*=(const unsigned int& a){
  unsigned int random;
  half tmp;

  accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = x;
  HALF_BIT_FLIP(tmp, random);
  x = tmp;
  x *= a;
  HALF_BIT_FLIP(x, random);

  random = RANDOM;
  tmp = y;
  HALF_BIT_FLIP(tmp, random);
  y = tmp;
  y *= a;
  HALF_BIT_FLIP(y, random);

  random = 1^random;
  tmp = z;
  HALF_BIT_FLIP(tmp, random);
  z = tmp;
  z *= a;
  HALF_BIT_FLIP(z, random);
  
  
  
  if(_cadna_half_overflow_tag){
   if (std::isinf(x) or std::isinf(y) or std::isinf(z)){
    instability(&_cadna_half_overflow_count);
    }
}



  return *this;
  

}
inline half_st& half_st::operator*=(const int& a){
  unsigned int random;
  half tmp;

  accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = x;
  HALF_BIT_FLIP(tmp, random);
  x = tmp;
  x *= a;
  HALF_BIT_FLIP(x, random);

  random = RANDOM;
  tmp = y;
  HALF_BIT_FLIP(tmp, random);
  y = tmp;
  y *= a;
  HALF_BIT_FLIP(y, random);

  random = 1^random;
  tmp = z;
  HALF_BIT_FLIP(tmp, random);
  z = tmp;
  z *= a;
  HALF_BIT_FLIP(z, random);
  
  
  
  if(_cadna_half_overflow_tag){
   if (std::isinf(x) or std::isinf(y) or std::isinf(z)){
    instability(&_cadna_half_overflow_count);
    }
}



  return *this;
  

}
inline half_st& half_st::operator*=(const unsigned short& a){
  unsigned int random;
  half tmp;

  accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = x;
  HALF_BIT_FLIP(tmp, random);
  x = tmp;
  x *= a;
  HALF_BIT_FLIP(x, random);

  random = RANDOM;
  tmp = y;
  HALF_BIT_FLIP(tmp, random);
  y = tmp;
  y *= a;
  HALF_BIT_FLIP(y, random);

  random = 1^random;
  tmp = z;
  HALF_BIT_FLIP(tmp, random);
  z = tmp;
  z *= a;
  HALF_BIT_FLIP(z, random);
  
  
  
  if(_cadna_half_overflow_tag){
   if (std::isinf(x) or std::isinf(y) or std::isinf(z)){
    instability(&_cadna_half_overflow_count);
    }
}



  return *this;
  

}
inline half_st& half_st::operator*=(const short& a){
  unsigned int random;
  half tmp;

  accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = x;
  HALF_BIT_FLIP(tmp, random);
  x = tmp;
  x *= a;
  HALF_BIT_FLIP(x, random);

  random = RANDOM;
  tmp = y;
  HALF_BIT_FLIP(tmp, random);
  y = tmp;
  y *= a;
  HALF_BIT_FLIP(y, random);

  random = 1^random;
  tmp = z;
  HALF_BIT_FLIP(tmp, random);
  z = tmp;
  z *= a;
  HALF_BIT_FLIP(z, random);
  
  
  
  if(_cadna_half_overflow_tag){
   if (std::isinf(x) or std::isinf(y) or std::isinf(z)){
    instability(&_cadna_half_overflow_count);
    }
}



  return *this;
  

}
inline half_st& half_st::operator*=(const unsigned char& a){
  unsigned int random;
  half tmp;

  accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = x;
  HALF_BIT_FLIP(tmp, random);
  x = tmp;
  x *= a;
  HALF_BIT_FLIP(x, random);

  random = RANDOM;
  tmp = y;
  HALF_BIT_FLIP(tmp, random);
  y = tmp;
  y *= a;
  HALF_BIT_FLIP(y, random);

  random = 1^random;
  tmp = z;
  HALF_BIT_FLIP(tmp, random);
  z = tmp;
  z *= a;
  HALF_BIT_FLIP(z, random);
  
  
  
  if(_cadna_half_overflow_tag){
   if (std::isinf(x) or std::isinf(y) or std::isinf(z)){
    instability(&_cadna_half_overflow_count);
    }
}



  return *this;
  

}
inline half_st& half_st::operator*=(const char& a){
  unsigned int random;
  half tmp;

  accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = x;
  HALF_BIT_FLIP(tmp, random);
  x = tmp;
  x *= a;
  HALF_BIT_FLIP(x, random);

  random = RANDOM;
  tmp = y;
  HALF_BIT_FLIP(tmp, random);
  y = tmp;
  y *= a;
  HALF_BIT_FLIP(y, random);

  random = 1^random;
  tmp = z;
  HALF_BIT_FLIP(tmp, random);
  z = tmp;
  z *= a;
  HALF_BIT_FLIP(z, random);
  
  
  
  if(_cadna_half_overflow_tag){
   if (std::isinf(x) or std::isinf(y) or std::isinf(z)){
    instability(&_cadna_half_overflow_count);
    }
}



  return *this;
  

}
inline half_st& half_st::operator*=(const half& a){
  unsigned int random;
  half tmp;

  accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = x;
  HALF_BIT_FLIP(tmp, random);
  x = tmp;
  x *= a;
  HALF_BIT_FLIP(x, random);

  random = RANDOM;
  tmp = y;
  HALF_BIT_FLIP(tmp, random);
  y = tmp;
  y *= a;
  HALF_BIT_FLIP(y, random);

  random = 1^random;
  tmp = z;
  HALF_BIT_FLIP(tmp, random);
  z = tmp;
  z *= a;
  HALF_BIT_FLIP(z, random);
  
  
  
  if(_cadna_half_overflow_tag){
   if (std::isinf(x) or std::isinf(y) or std::isinf(z)){
    instability(&_cadna_half_overflow_count);
    }
}



  return *this;
  

}
#endif // CADNA_HALF_EMULATION


#ifdef CADNA_HALF_NATIVE
inline half_st& half_st::operator*=(const double& a){
  unsigned int random;
  half tmp;

  accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = x;
  HALF_BIT_FLIP(tmp, random);
  x = tmp;
  x *= a;
  HALF_BIT_FLIP(x, random);

  random = RANDOM;
  tmp = y;
  HALF_BIT_FLIP(tmp, random);
  y = tmp;
  y *= a;
  HALF_BIT_FLIP(y, random);

  random = 1^random;
  tmp = z;
  HALF_BIT_FLIP(tmp, random);
  z = tmp;
  z *= a;
  HALF_BIT_FLIP(z, random);
  
  
  
  if(_cadna_half_overflow_tag){
   if (std::isinf(x) or std::isinf(y) or std::isinf(z)){
    instability(&_cadna_half_overflow_count);
    }
}



  return *this;
  

}
inline half_st& half_st::operator*=(const float& a){
  unsigned int random;
  half tmp;

  accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = x;
  HALF_BIT_FLIP(tmp, random);
  x = tmp;
  x *= a;
  HALF_BIT_FLIP(x, random);

  random = RANDOM;
  tmp = y;
  HALF_BIT_FLIP(tmp, random);
  y = tmp;
  y *= a;
  HALF_BIT_FLIP(y, random);

  random = 1^random;
  tmp = z;
  HALF_BIT_FLIP(tmp, random);
  z = tmp;
  z *= a;
  HALF_BIT_FLIP(z, random);
  
  
  
  if(_cadna_half_overflow_tag){
   if (std::isinf(x) or std::isinf(y) or std::isinf(z)){
    instability(&_cadna_half_overflow_count);
    }
}



  return *this;
  

}
inline half_st& half_st::operator*=(const unsigned long long& a){
  unsigned int random;
  half tmp;

  accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = x;
  HALF_BIT_FLIP(tmp, random);
  x = tmp;
  x *= a;
  HALF_BIT_FLIP(x, random);

  random = RANDOM;
  tmp = y;
  HALF_BIT_FLIP(tmp, random);
  y = tmp;
  y *= a;
  HALF_BIT_FLIP(y, random);

  random = 1^random;
  tmp = z;
  HALF_BIT_FLIP(tmp, random);
  z = tmp;
  z *= a;
  HALF_BIT_FLIP(z, random);
  
  
  
  if(_cadna_half_overflow_tag){
   if (std::isinf(x) or std::isinf(y) or std::isinf(z)){
    instability(&_cadna_half_overflow_count);
    }
}



  return *this;
  

}
inline half_st& half_st::operator*=(const long long& a){
  unsigned int random;
  half tmp;

  accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = x;
  HALF_BIT_FLIP(tmp, random);
  x = tmp;
  x *= a;
  HALF_BIT_FLIP(x, random);

  random = RANDOM;
  tmp = y;
  HALF_BIT_FLIP(tmp, random);
  y = tmp;
  y *= a;
  HALF_BIT_FLIP(y, random);

  random = 1^random;
  tmp = z;
  HALF_BIT_FLIP(tmp, random);
  z = tmp;
  z *= a;
  HALF_BIT_FLIP(z, random);
  
  
  
  if(_cadna_half_overflow_tag){
   if (std::isinf(x) or std::isinf(y) or std::isinf(z)){
    instability(&_cadna_half_overflow_count);
    }
}



  return *this;
  

}
inline half_st& half_st::operator*=(const unsigned long& a){
  unsigned int random;
  half tmp;

  accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = x;
  HALF_BIT_FLIP(tmp, random);
  x = tmp;
  x *= a;
  HALF_BIT_FLIP(x, random);

  random = RANDOM;
  tmp = y;
  HALF_BIT_FLIP(tmp, random);
  y = tmp;
  y *= a;
  HALF_BIT_FLIP(y, random);

  random = 1^random;
  tmp = z;
  HALF_BIT_FLIP(tmp, random);
  z = tmp;
  z *= a;
  HALF_BIT_FLIP(z, random);
  
  
  
  if(_cadna_half_overflow_tag){
   if (std::isinf(x) or std::isinf(y) or std::isinf(z)){
    instability(&_cadna_half_overflow_count);
    }
}



  return *this;
  

}
inline half_st& half_st::operator*=(const long& a){
  unsigned int random;
  half tmp;

  accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = x;
  HALF_BIT_FLIP(tmp, random);
  x = tmp;
  x *= a;
  HALF_BIT_FLIP(x, random);

  random = RANDOM;
  tmp = y;
  HALF_BIT_FLIP(tmp, random);
  y = tmp;
  y *= a;
  HALF_BIT_FLIP(y, random);

  random = 1^random;
  tmp = z;
  HALF_BIT_FLIP(tmp, random);
  z = tmp;
  z *= a;
  HALF_BIT_FLIP(z, random);
  
  
  
  if(_cadna_half_overflow_tag){
   if (std::isinf(x) or std::isinf(y) or std::isinf(z)){
    instability(&_cadna_half_overflow_count);
    }
}



  return *this;
  

}
inline half_st& half_st::operator*=(const unsigned int& a){
  unsigned int random;
  half tmp;

  accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = x;
  HALF_BIT_FLIP(tmp, random);
  x = tmp;
  x *= a;
  HALF_BIT_FLIP(x, random);

  random = RANDOM;
  tmp = y;
  HALF_BIT_FLIP(tmp, random);
  y = tmp;
  y *= a;
  HALF_BIT_FLIP(y, random);

  random = 1^random;
  tmp = z;
  HALF_BIT_FLIP(tmp, random);
  z = tmp;
  z *= a;
  HALF_BIT_FLIP(z, random);
  
  
  
  if(_cadna_half_overflow_tag){
   if (std::isinf(x) or std::isinf(y) or std::isinf(z)){
    instability(&_cadna_half_overflow_count);
    }
}



  return *this;
  

}
inline half_st& half_st::operator*=(const int& a){
  unsigned int random;
  half tmp;

  accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = x;
  HALF_BIT_FLIP(tmp, random);
  x = tmp;
  x *= a;
  HALF_BIT_FLIP(x, random);

  random = RANDOM;
  tmp = y;
  HALF_BIT_FLIP(tmp, random);
  y = tmp;
  y *= a;
  HALF_BIT_FLIP(y, random);

  random = 1^random;
  tmp = z;
  HALF_BIT_FLIP(tmp, random);
  z = tmp;
  z *= a;
  HALF_BIT_FLIP(z, random);
  
  
  
  if(_cadna_half_overflow_tag){
   if (std::isinf(x) or std::isinf(y) or std::isinf(z)){
    instability(&_cadna_half_overflow_count);
    }
}



  return *this;
  

}
inline half_st& half_st::operator*=(const unsigned short& a){
  unsigned int random;
  half tmp;

  accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = x;
  HALF_BIT_FLIP(tmp, random);
  x = tmp;
  x *= a;
  HALF_BIT_FLIP(x, random);

  random = RANDOM;
  tmp = y;
  HALF_BIT_FLIP(tmp, random);
  y = tmp;
  y *= a;
  HALF_BIT_FLIP(y, random);

  random = 1^random;
  tmp = z;
  HALF_BIT_FLIP(tmp, random);
  z = tmp;
  z *= a;
  HALF_BIT_FLIP(z, random);
  
  
  
  if(_cadna_half_overflow_tag){
   if (std::isinf(x) or std::isinf(y) or std::isinf(z)){
    instability(&_cadna_half_overflow_count);
    }
}



  return *this;
  

}
inline half_st& half_st::operator*=(const short& a){
  unsigned int random;
  half tmp;

  accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = x;
  HALF_BIT_FLIP(tmp, random);
  x = tmp;
  x *= a;
  HALF_BIT_FLIP(x, random);

  random = RANDOM;
  tmp = y;
  HALF_BIT_FLIP(tmp, random);
  y = tmp;
  y *= a;
  HALF_BIT_FLIP(y, random);

  random = 1^random;
  tmp = z;
  HALF_BIT_FLIP(tmp, random);
  z = tmp;
  z *= a;
  HALF_BIT_FLIP(z, random);
  
  
  
  if(_cadna_half_overflow_tag){
   if (std::isinf(x) or std::isinf(y) or std::isinf(z)){
    instability(&_cadna_half_overflow_count);
    }
}



  return *this;
  

}
inline half_st& half_st::operator*=(const unsigned char& a){
  unsigned int random;
  half tmp;

  accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = x;
  HALF_BIT_FLIP(tmp, random);
  x = tmp;
  x *= a;
  HALF_BIT_FLIP(x, random);

  random = RANDOM;
  tmp = y;
  HALF_BIT_FLIP(tmp, random);
  y = tmp;
  y *= a;
  HALF_BIT_FLIP(y, random);

  random = 1^random;
  tmp = z;
  HALF_BIT_FLIP(tmp, random);
  z = tmp;
  z *= a;
  HALF_BIT_FLIP(z, random);
  
  
  
  if(_cadna_half_overflow_tag){
   if (std::isinf(x) or std::isinf(y) or std::isinf(z)){
    instability(&_cadna_half_overflow_count);
    }
}



  return *this;
  

}
inline half_st& half_st::operator*=(const char& a){
  unsigned int random;
  half tmp;

  accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = x;
  HALF_BIT_FLIP(tmp, random);
  x = tmp;
  x *= a;
  HALF_BIT_FLIP(x, random);

  random = RANDOM;
  tmp = y;
  HALF_BIT_FLIP(tmp, random);
  y = tmp;
  y *= a;
  HALF_BIT_FLIP(y, random);

  random = 1^random;
  tmp = z;
  HALF_BIT_FLIP(tmp, random);
  z = tmp;
  z *= a;
  HALF_BIT_FLIP(z, random);
  
  
  
  if(_cadna_half_overflow_tag){
   if (std::isinf(x) or std::isinf(y) or std::isinf(z)){
    instability(&_cadna_half_overflow_count);
    }
}



  return *this;
  

}
inline half_st& half_st::operator*=(const half& a){
  unsigned int random;
  half tmp;

  accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = x;
  HALF_BIT_FLIP(tmp, random);
  x = tmp;
  x *= a;
  HALF_BIT_FLIP(x, random);

  random = RANDOM;
  tmp = y;
  HALF_BIT_FLIP(tmp, random);
  y = tmp;
  y *= a;
  HALF_BIT_FLIP(y, random);

  random = 1^random;
  tmp = z;
  HALF_BIT_FLIP(tmp, random);
  z = tmp;
  z *= a;
  HALF_BIT_FLIP(z, random);
  
  
  
  if(_cadna_half_overflow_tag){
   if (std::isinf(x) or std::isinf(y) or std::isinf(z)){
    instability(&_cadna_half_overflow_count);
    }
}



  return *this;
  

}
#endif

/////////////////////////////////////////////////////////////////////////






inline double_st& double_st::operator*=(const double_st& a){
  unsigned int random;
  double tmp;

  if(_cadna_mul_tag) {
    if (accuracy==DIGIT_NOT_COMPUTED) approx_digit();
    if (accuracy==0){
      if (a.accuracy==DIGIT_NOT_COMPUTED) a.approx_digit();
      if (a.accuracy==0)
	instability(&_cadna_mul_count);
    }
  }

  accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = x;
  DOUBLE_BIT_FLIP(tmp, random);
  x = tmp;
  tmp = a.x;
  x *= tmp;
  DOUBLE_BIT_FLIP(x, random);

  random = RANDOM;
  tmp = y;
  DOUBLE_BIT_FLIP(tmp, random);
  y = tmp;
  tmp = a.y;
  y *= tmp;
  DOUBLE_BIT_FLIP(y, random);

  random = 1^random;
  tmp = z;
  DOUBLE_BIT_FLIP(tmp, random);
  z = tmp;
  tmp = a.z;
  z *= tmp;
  DOUBLE_BIT_FLIP(z, random);

  return *this;
}
inline double_st& double_st::operator*=(const float_st& a){
  unsigned int random;
  double tmp;

  if(_cadna_mul_tag) {
    if (accuracy==DIGIT_NOT_COMPUTED) approx_digit();
    if (accuracy==0){
      if (a.accuracy==DIGIT_NOT_COMPUTED) a.approx_digit();
      if (a.accuracy==0)
	instability(&_cadna_mul_count);
    }
  }

  accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = x;
  DOUBLE_BIT_FLIP(tmp, random);
  x = tmp;
  tmp = a.x;
  x *= tmp;
  DOUBLE_BIT_FLIP(x, random);

  random = RANDOM;
  tmp = y;
  DOUBLE_BIT_FLIP(tmp, random);
  y = tmp;
  tmp = a.y;
  y *= tmp;
  DOUBLE_BIT_FLIP(y, random);

  random = 1^random;
  tmp = z;
  DOUBLE_BIT_FLIP(tmp, random);
  z = tmp;
  tmp = a.z;
  z *= tmp;
  DOUBLE_BIT_FLIP(z, random);

  return *this;
}
inline float_st& float_st::operator*=(const double_st& a){
  unsigned int random;
  float tmp;

  if(_cadna_mul_tag) {
    if (accuracy==DIGIT_NOT_COMPUTED) approx_digit();
    if (accuracy==0){
      if (a.accuracy==DIGIT_NOT_COMPUTED) a.approx_digit();
      if (a.accuracy==0)
	instability(&_cadna_mul_count);
    }
  }

  accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = x;
  FLOAT_BIT_FLIP(tmp, random);
  x = tmp;
  tmp = a.x;
  x *= tmp;
  FLOAT_BIT_FLIP(x, random);

  random = RANDOM;
  tmp = y;
  FLOAT_BIT_FLIP(tmp, random);
  y = tmp;
  tmp = a.y;
  y *= tmp;
  FLOAT_BIT_FLIP(y, random);

  random = 1^random;
  tmp = z;
  FLOAT_BIT_FLIP(tmp, random);
  z = tmp;
  tmp = a.z;
  z *= tmp;
  FLOAT_BIT_FLIP(z, random);

  return *this;
}
inline float_st& float_st::operator*=(const float_st& a){
  unsigned int random;
  float tmp;

  if(_cadna_mul_tag) {
    if (accuracy==DIGIT_NOT_COMPUTED) approx_digit();
    if (accuracy==0){
      if (a.accuracy==DIGIT_NOT_COMPUTED) a.approx_digit();
      if (a.accuracy==0)
	instability(&_cadna_mul_count);
    }
  }

  accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = x;
  FLOAT_BIT_FLIP(tmp, random);
  x = tmp;
  tmp = a.x;
  x *= tmp;
  FLOAT_BIT_FLIP(x, random);

  random = RANDOM;
  tmp = y;
  FLOAT_BIT_FLIP(tmp, random);
  y = tmp;
  tmp = a.y;
  y *= tmp;
  FLOAT_BIT_FLIP(y, random);

  random = 1^random;
  tmp = z;
  FLOAT_BIT_FLIP(tmp, random);
  z = tmp;
  tmp = a.z;
  z *= tmp;
  FLOAT_BIT_FLIP(z, random);

  return *this;
}

#ifdef CADNA_HALF_EMULATION
inline half_st& half_st::operator*=(const half_st& a){
  unsigned int random;
  half tmp;

  if(_cadna_mul_tag) {
    if (accuracy==DIGIT_NOT_COMPUTED) approx_digit();
    if (accuracy==0){
      if (a.accuracy==DIGIT_NOT_COMPUTED) a.approx_digit();
      if (a.accuracy==0)
	instability(&_cadna_mul_count);
    }
  }

  accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = x;
  HALF_BIT_FLIP(tmp, random);
  x = tmp;
  tmp = a.x;
  x *= tmp;
  HALF_BIT_FLIP(x, random);

  random = RANDOM;
  tmp = y;
  HALF_BIT_FLIP(tmp, random);
  y = tmp;
  tmp = a.y;
  y *= tmp;
  HALF_BIT_FLIP(y, random);

  random = 1^random;
  tmp = z;
  HALF_BIT_FLIP(tmp, random);
  z = tmp;
  tmp = a.z;
  z *= tmp;
  HALF_BIT_FLIP(z, random);
  
     
    
    if(_cadna_half_overflow_tag){
    if (std::isinf(x) or std::isinf(y) or std::isinf(z)){
    instability(&_cadna_half_overflow_count);
    }
}



  return *this;


}
inline double_st& double_st::operator*=(const half_st& a){
  unsigned int random;
  double tmp;

  if(_cadna_mul_tag) {
    if (accuracy==DIGIT_NOT_COMPUTED) approx_digit();
    if (accuracy==0){
      if (a.accuracy==DIGIT_NOT_COMPUTED) a.approx_digit();
      if (a.accuracy==0)
	instability(&_cadna_mul_count);
    }
  }

  accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = x;
  DOUBLE_BIT_FLIP(tmp, random);
  x = tmp;
  tmp = a.x;
  x *= tmp;
  DOUBLE_BIT_FLIP(x, random);

  random = RANDOM;
  tmp = y;
  DOUBLE_BIT_FLIP(tmp, random);
  y = tmp;
  tmp = a.y;
  y *= tmp;
  DOUBLE_BIT_FLIP(y, random);

  random = 1^random;
  tmp = z;
  DOUBLE_BIT_FLIP(tmp, random);
  z = tmp;
  tmp = a.z;
  z *= tmp;
  DOUBLE_BIT_FLIP(z, random);

  return *this;
}
inline float_st& float_st::operator*=(const half_st& a){
  unsigned int random;
  float tmp;

  if(_cadna_mul_tag) {
    if (accuracy==DIGIT_NOT_COMPUTED) approx_digit();
    if (accuracy==0){
      if (a.accuracy==DIGIT_NOT_COMPUTED) a.approx_digit();
      if (a.accuracy==0)
	instability(&_cadna_mul_count);
    }
  }

  accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = x;
  FLOAT_BIT_FLIP(tmp, random);
  x = tmp;
  tmp = a.x;
  x *= tmp;
  FLOAT_BIT_FLIP(x, random);

  random = RANDOM;
  tmp = y;
  FLOAT_BIT_FLIP(tmp, random);
  y = tmp;
  tmp = a.y;
  y *= tmp;
  FLOAT_BIT_FLIP(y, random);

  random = 1^random;
  tmp = z;
  FLOAT_BIT_FLIP(tmp, random);
  z = tmp;
  tmp = a.z;
  z *= tmp;
  FLOAT_BIT_FLIP(z, random);

  return *this;
}
inline half_st& half_st::operator*=(const float_st& a){
  unsigned int random;
  half tmp;

  if(_cadna_mul_tag) {
    if (accuracy==DIGIT_NOT_COMPUTED) approx_digit();
    if (accuracy==0){
      if (a.accuracy==DIGIT_NOT_COMPUTED) a.approx_digit();
      if (a.accuracy==0)
	instability(&_cadna_mul_count);
    }
  }

  accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = x;
  HALF_BIT_FLIP(tmp, random);
  x = tmp;
  tmp = a.x;
  x *= tmp;
  HALF_BIT_FLIP(x, random);

  random = RANDOM;
  tmp = y;
  HALF_BIT_FLIP(tmp, random);
  y = tmp;
  tmp = a.y;
  y *= tmp;
  HALF_BIT_FLIP(y, random);

  random = 1^random;
  tmp = z;
  HALF_BIT_FLIP(tmp, random);
  z = tmp;
  tmp = a.z;
  z *= tmp;
  HALF_BIT_FLIP(z, random);
  
     
    
    if(_cadna_half_overflow_tag){
    if (std::isinf(x) or std::isinf(y) or std::isinf(z)){
    instability(&_cadna_half_overflow_count);
    }
}



  return *this;


}
inline half_st& half_st::operator*=(const double_st& a){
  unsigned int random;
  half tmp;

  if(_cadna_mul_tag) {
    if (accuracy==DIGIT_NOT_COMPUTED) approx_digit();
    if (accuracy==0){
      if (a.accuracy==DIGIT_NOT_COMPUTED) a.approx_digit();
      if (a.accuracy==0)
	instability(&_cadna_mul_count);
    }
  }

  accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = x;
  HALF_BIT_FLIP(tmp, random);
  x = tmp;
  tmp = a.x;
  x *= tmp;
  HALF_BIT_FLIP(x, random);

  random = RANDOM;
  tmp = y;
  HALF_BIT_FLIP(tmp, random);
  y = tmp;
  tmp = a.y;
  y *= tmp;
  HALF_BIT_FLIP(y, random);

  random = 1^random;
  tmp = z;
  HALF_BIT_FLIP(tmp, random);
  z = tmp;
  tmp = a.z;
  z *= tmp;
  HALF_BIT_FLIP(z, random);
  
     
    
    if(_cadna_half_overflow_tag){
    if (std::isinf(x) or std::isinf(y) or std::isinf(z)){
    instability(&_cadna_half_overflow_count);
    }
}



  return *this;


}
#endif


#ifdef CADNA_HALF_NATIVE
inline half_st& half_st::operator*=(const half_st& a){
  unsigned int random;
  half tmp;

  if(_cadna_mul_tag) {
    if (accuracy==DIGIT_NOT_COMPUTED) approx_digit();
    if (accuracy==0){
      if (a.accuracy==DIGIT_NOT_COMPUTED) a.approx_digit();
      if (a.accuracy==0)
	instability(&_cadna_mul_count);
    }
  }

  accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = x;
  HALF_BIT_FLIP(tmp, random);
  x = tmp;
  tmp = a.x;
  x *= tmp;
  HALF_BIT_FLIP(x, random);

  random = RANDOM;
  tmp = y;
  HALF_BIT_FLIP(tmp, random);
  y = tmp;
  tmp = a.y;
  y *= tmp;
  HALF_BIT_FLIP(y, random);

  random = 1^random;
  tmp = z;
  HALF_BIT_FLIP(tmp, random);
  z = tmp;
  tmp = a.z;
  z *= tmp;
  HALF_BIT_FLIP(z, random);
  
     
    
    if(_cadna_half_overflow_tag){
    if (std::isinf(x) or std::isinf(y) or std::isinf(z)){
    instability(&_cadna_half_overflow_count);
    }
}



  return *this;


}
inline double_st& double_st::operator*=(const half_st& a){
  unsigned int random;
  double tmp;

  if(_cadna_mul_tag) {
    if (accuracy==DIGIT_NOT_COMPUTED) approx_digit();
    if (accuracy==0){
      if (a.accuracy==DIGIT_NOT_COMPUTED) a.approx_digit();
      if (a.accuracy==0)
	instability(&_cadna_mul_count);
    }
  }

  accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = x;
  DOUBLE_BIT_FLIP(tmp, random);
  x = tmp;
  tmp = a.x;
  x *= tmp;
  DOUBLE_BIT_FLIP(x, random);

  random = RANDOM;
  tmp = y;
  DOUBLE_BIT_FLIP(tmp, random);
  y = tmp;
  tmp = a.y;
  y *= tmp;
  DOUBLE_BIT_FLIP(y, random);

  random = 1^random;
  tmp = z;
  DOUBLE_BIT_FLIP(tmp, random);
  z = tmp;
  tmp = a.z;
  z *= tmp;
  DOUBLE_BIT_FLIP(z, random);

  return *this;
}
inline float_st& float_st::operator*=(const half_st& a){
  unsigned int random;
  float tmp;

  if(_cadna_mul_tag) {
    if (accuracy==DIGIT_NOT_COMPUTED) approx_digit();
    if (accuracy==0){
      if (a.accuracy==DIGIT_NOT_COMPUTED) a.approx_digit();
      if (a.accuracy==0)
	instability(&_cadna_mul_count);
    }
  }

  accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = x;
  FLOAT_BIT_FLIP(tmp, random);
  x = tmp;
  tmp = a.x;
  x *= tmp;
  FLOAT_BIT_FLIP(x, random);

  random = RANDOM;
  tmp = y;
  FLOAT_BIT_FLIP(tmp, random);
  y = tmp;
  tmp = a.y;
  y *= tmp;
  FLOAT_BIT_FLIP(y, random);

  random = 1^random;
  tmp = z;
  FLOAT_BIT_FLIP(tmp, random);
  z = tmp;
  tmp = a.z;
  z *= tmp;
  FLOAT_BIT_FLIP(z, random);

  return *this;
}
inline half_st& half_st::operator*=(const float_st& a){
  unsigned int random;
  half tmp;

  if(_cadna_mul_tag) {
    if (accuracy==DIGIT_NOT_COMPUTED) approx_digit();
    if (accuracy==0){
      if (a.accuracy==DIGIT_NOT_COMPUTED) a.approx_digit();
      if (a.accuracy==0)
	instability(&_cadna_mul_count);
    }
  }

  accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = x;
  HALF_BIT_FLIP(tmp, random);
  x = tmp;
  tmp = a.x;
  x *= tmp;
  HALF_BIT_FLIP(x, random);

  random = RANDOM;
  tmp = y;
  HALF_BIT_FLIP(tmp, random);
  y = tmp;
  tmp = a.y;
  y *= tmp;
  HALF_BIT_FLIP(y, random);

  random = 1^random;
  tmp = z;
  HALF_BIT_FLIP(tmp, random);
  z = tmp;
  tmp = a.z;
  z *= tmp;
  HALF_BIT_FLIP(z, random);
  
     
    
    if(_cadna_half_overflow_tag){
    if (std::isinf(x) or std::isinf(y) or std::isinf(z)){
    instability(&_cadna_half_overflow_count);
    }
}



  return *this;


}
inline half_st& half_st::operator*=(const double_st& a){
  unsigned int random;
  half tmp;

  if(_cadna_mul_tag) {
    if (accuracy==DIGIT_NOT_COMPUTED) approx_digit();
    if (accuracy==0){
      if (a.accuracy==DIGIT_NOT_COMPUTED) a.approx_digit();
      if (a.accuracy==0)
	instability(&_cadna_mul_count);
    }
  }

  accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = x;
  HALF_BIT_FLIP(tmp, random);
  x = tmp;
  tmp = a.x;
  x *= tmp;
  HALF_BIT_FLIP(x, random);

  random = RANDOM;
  tmp = y;
  HALF_BIT_FLIP(tmp, random);
  y = tmp;
  tmp = a.y;
  y *= tmp;
  HALF_BIT_FLIP(y, random);

  random = 1^random;
  tmp = z;
  HALF_BIT_FLIP(tmp, random);
  z = tmp;
  tmp = a.z;
  z *= tmp;
  HALF_BIT_FLIP(z, random);
  
     
    
    if(_cadna_half_overflow_tag){
    if (std::isinf(x) or std::isinf(y) or std::isinf(z)){
    instability(&_cadna_half_overflow_count);
    }
}



  return *this;


}
#endif

/////////////////////////////////////////////////////////////////////////







inline double_st operator*(const double& a, const double_st& b){
  double_st res;
  unsigned int random;
  double tmp;

  res.accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = a;
  DOUBLE_BIT_FLIP(tmp, random);
  res.x = tmp;
  tmp = b.x;
  res.x *= tmp;
  DOUBLE_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp = a;
  DOUBLE_BIT_FLIP(tmp, random);
  res.y = tmp;
  tmp = b.y;
  res.y *= tmp;
  DOUBLE_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp = a;
  DOUBLE_BIT_FLIP(tmp, random);
  res.z = tmp;
  tmp = b.z;
  res.z *= tmp;
  DOUBLE_BIT_FLIP(res.z, random);

  return res ;
}
inline double_st operator*(const float& a, const double_st& b){
  double_st res;
  unsigned int random;
  double tmp;

  res.accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = a;
  DOUBLE_BIT_FLIP(tmp, random);
  res.x = tmp;
  tmp = b.x;
  res.x *= tmp;
  DOUBLE_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp = a;
  DOUBLE_BIT_FLIP(tmp, random);
  res.y = tmp;
  tmp = b.y;
  res.y *= tmp;
  DOUBLE_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp = a;
  DOUBLE_BIT_FLIP(tmp, random);
  res.z = tmp;
  tmp = b.z;
  res.z *= tmp;
  DOUBLE_BIT_FLIP(res.z, random);

  return res ;
}
inline double_st operator*(const unsigned long long& a, const double_st& b){
  double_st res;
  unsigned int random;
  double tmp;

  res.accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = a;
  DOUBLE_BIT_FLIP(tmp, random);
  res.x = tmp;
  tmp = b.x;
  res.x *= tmp;
  DOUBLE_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp = a;
  DOUBLE_BIT_FLIP(tmp, random);
  res.y = tmp;
  tmp = b.y;
  res.y *= tmp;
  DOUBLE_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp = a;
  DOUBLE_BIT_FLIP(tmp, random);
  res.z = tmp;
  tmp = b.z;
  res.z *= tmp;
  DOUBLE_BIT_FLIP(res.z, random);

  return res ;
}
inline double_st operator*(const long long& a, const double_st& b){
  double_st res;
  unsigned int random;
  double tmp;

  res.accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = a;
  DOUBLE_BIT_FLIP(tmp, random);
  res.x = tmp;
  tmp = b.x;
  res.x *= tmp;
  DOUBLE_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp = a;
  DOUBLE_BIT_FLIP(tmp, random);
  res.y = tmp;
  tmp = b.y;
  res.y *= tmp;
  DOUBLE_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp = a;
  DOUBLE_BIT_FLIP(tmp, random);
  res.z = tmp;
  tmp = b.z;
  res.z *= tmp;
  DOUBLE_BIT_FLIP(res.z, random);

  return res ;
}
inline double_st operator*(const unsigned long& a, const double_st& b){
  double_st res;
  unsigned int random;
  double tmp;

  res.accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = a;
  DOUBLE_BIT_FLIP(tmp, random);
  res.x = tmp;
  tmp = b.x;
  res.x *= tmp;
  DOUBLE_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp = a;
  DOUBLE_BIT_FLIP(tmp, random);
  res.y = tmp;
  tmp = b.y;
  res.y *= tmp;
  DOUBLE_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp = a;
  DOUBLE_BIT_FLIP(tmp, random);
  res.z = tmp;
  tmp = b.z;
  res.z *= tmp;
  DOUBLE_BIT_FLIP(res.z, random);

  return res ;
}
inline double_st operator*(const long& a, const double_st& b){
  double_st res;
  unsigned int random;
  double tmp;

  res.accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = a;
  DOUBLE_BIT_FLIP(tmp, random);
  res.x = tmp;
  tmp = b.x;
  res.x *= tmp;
  DOUBLE_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp = a;
  DOUBLE_BIT_FLIP(tmp, random);
  res.y = tmp;
  tmp = b.y;
  res.y *= tmp;
  DOUBLE_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp = a;
  DOUBLE_BIT_FLIP(tmp, random);
  res.z = tmp;
  tmp = b.z;
  res.z *= tmp;
  DOUBLE_BIT_FLIP(res.z, random);

  return res ;
}
inline double_st operator*(const unsigned int & a, const double_st& b){
  double_st res;
  unsigned int random;
  double tmp;

  res.accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = a;
  DOUBLE_BIT_FLIP(tmp, random);
  res.x = tmp;
  tmp = b.x;
  res.x *= tmp;
  DOUBLE_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp = a;
  DOUBLE_BIT_FLIP(tmp, random);
  res.y = tmp;
  tmp = b.y;
  res.y *= tmp;
  DOUBLE_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp = a;
  DOUBLE_BIT_FLIP(tmp, random);
  res.z = tmp;
  tmp = b.z;
  res.z *= tmp;
  DOUBLE_BIT_FLIP(res.z, random);

  return res ;
}
inline double_st operator*(const int & a, const double_st& b){
  double_st res;
  unsigned int random;
  double tmp;

  res.accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = a;
  DOUBLE_BIT_FLIP(tmp, random);
  res.x = tmp;
  tmp = b.x;
  res.x *= tmp;
  DOUBLE_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp = a;
  DOUBLE_BIT_FLIP(tmp, random);
  res.y = tmp;
  tmp = b.y;
  res.y *= tmp;
  DOUBLE_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp = a;
  DOUBLE_BIT_FLIP(tmp, random);
  res.z = tmp;
  tmp = b.z;
  res.z *= tmp;
  DOUBLE_BIT_FLIP(res.z, random);

  return res ;
}
inline double_st operator*(const unsigned short& a, const double_st& b){
  double_st res;
  unsigned int random;
  double tmp;

  res.accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = a;
  DOUBLE_BIT_FLIP(tmp, random);
  res.x = tmp;
  tmp = b.x;
  res.x *= tmp;
  DOUBLE_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp = a;
  DOUBLE_BIT_FLIP(tmp, random);
  res.y = tmp;
  tmp = b.y;
  res.y *= tmp;
  DOUBLE_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp = a;
  DOUBLE_BIT_FLIP(tmp, random);
  res.z = tmp;
  tmp = b.z;
  res.z *= tmp;
  DOUBLE_BIT_FLIP(res.z, random);

  return res ;
}
inline double_st operator*(const short& a, const double_st& b){
  double_st res;
  unsigned int random;
  double tmp;

  res.accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = a;
  DOUBLE_BIT_FLIP(tmp, random);
  res.x = tmp;
  tmp = b.x;
  res.x *= tmp;
  DOUBLE_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp = a;
  DOUBLE_BIT_FLIP(tmp, random);
  res.y = tmp;
  tmp = b.y;
  res.y *= tmp;
  DOUBLE_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp = a;
  DOUBLE_BIT_FLIP(tmp, random);
  res.z = tmp;
  tmp = b.z;
  res.z *= tmp;
  DOUBLE_BIT_FLIP(res.z, random);

  return res ;
}
inline double_st operator*(const unsigned char& a, const double_st& b){
  double_st res;
  unsigned int random;
  double tmp;

  res.accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = a;
  DOUBLE_BIT_FLIP(tmp, random);
  res.x = tmp;
  tmp = b.x;
  res.x *= tmp;
  DOUBLE_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp = a;
  DOUBLE_BIT_FLIP(tmp, random);
  res.y = tmp;
  tmp = b.y;
  res.y *= tmp;
  DOUBLE_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp = a;
  DOUBLE_BIT_FLIP(tmp, random);
  res.z = tmp;
  tmp = b.z;
  res.z *= tmp;
  DOUBLE_BIT_FLIP(res.z, random);

  return res ;
}
inline double_st operator*(const char& a, const double_st& b){
  double_st res;
  unsigned int random;
  double tmp;

  res.accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = a;
  DOUBLE_BIT_FLIP(tmp, random);
  res.x = tmp;
  tmp = b.x;
  res.x *= tmp;
  DOUBLE_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp = a;
  DOUBLE_BIT_FLIP(tmp, random);
  res.y = tmp;
  tmp = b.y;
  res.y *= tmp;
  DOUBLE_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp = a;
  DOUBLE_BIT_FLIP(tmp, random);
  res.z = tmp;
  tmp = b.z;
  res.z *= tmp;
  DOUBLE_BIT_FLIP(res.z, random);

  return res ;
}
#ifdef CADNA_HALF_EMULATION
inline double_st operator*(const half& a, const double_st& b){
  double_st res;
  unsigned int random;
  double tmp;

  res.accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = a;
  DOUBLE_BIT_FLIP(tmp, random);
  res.x = tmp;
  tmp = b.x;
  res.x *= tmp;
  DOUBLE_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp = a;
  DOUBLE_BIT_FLIP(tmp, random);
  res.y = tmp;
  tmp = b.y;
  res.y *= tmp;
  DOUBLE_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp = a;
  DOUBLE_BIT_FLIP(tmp, random);
  res.z = tmp;
  tmp = b.z;
  res.z *= tmp;
  DOUBLE_BIT_FLIP(res.z, random);

  return res ;
}
#endif // CADNA_HALF_EMULATION

#ifdef CADNA_HALF_NATIVE
inline double_st operator*(const half& a, const double_st& b){
  double_st res;
  unsigned int random;
  double tmp;

  res.accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = a;
  DOUBLE_BIT_FLIP(tmp, random);
  res.x = tmp;
  tmp = b.x;
  res.x *= tmp;
  DOUBLE_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp = a;
  DOUBLE_BIT_FLIP(tmp, random);
  res.y = tmp;
  tmp = b.y;
  res.y *= tmp;
  DOUBLE_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp = a;
  DOUBLE_BIT_FLIP(tmp, random);
  res.z = tmp;
  tmp = b.z;
  res.z *= tmp;
  DOUBLE_BIT_FLIP(res.z, random);

  return res ;
}
#endif 

inline float_st operator*(const double& a, const float_st& b){
  float_st res;
  unsigned int random;
  float tmp;

  res.accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = a;
  FLOAT_BIT_FLIP(tmp, random);
  res.x = tmp;
  tmp = b.x;
  res.x *= tmp;
  FLOAT_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp = a;
  FLOAT_BIT_FLIP(tmp, random);
  res.y = tmp;
  tmp = b.y;
  res.y *= tmp;
  FLOAT_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp = a;
  FLOAT_BIT_FLIP(tmp, random);
  res.z = tmp;
  tmp = b.z;
  res.z *= tmp;
  FLOAT_BIT_FLIP(res.z, random);

  return res ;
}
inline float_st operator*(const float& a, const float_st& b){
  float_st res;
  unsigned int random;
  float tmp;

  res.accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = a;
  FLOAT_BIT_FLIP(tmp, random);
  res.x = tmp;
  tmp = b.x;
  res.x *= tmp;
  FLOAT_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp = a;
  FLOAT_BIT_FLIP(tmp, random);
  res.y = tmp;
  tmp = b.y;
  res.y *= tmp;
  FLOAT_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp = a;
  FLOAT_BIT_FLIP(tmp, random);
  res.z = tmp;
  tmp = b.z;
  res.z *= tmp;
  FLOAT_BIT_FLIP(res.z, random);

  return res ;
}
inline float_st operator*(const unsigned long long& a, const float_st& b){
  float_st res;
  unsigned int random;
  float tmp;

  res.accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = a;
  FLOAT_BIT_FLIP(tmp, random);
  res.x = tmp;
  tmp = b.x;
  res.x *= tmp;
  FLOAT_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp = a;
  FLOAT_BIT_FLIP(tmp, random);
  res.y = tmp;
  tmp = b.y;
  res.y *= tmp;
  FLOAT_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp = a;
  FLOAT_BIT_FLIP(tmp, random);
  res.z = tmp;
  tmp = b.z;
  res.z *= tmp;
  FLOAT_BIT_FLIP(res.z, random);

  return res ;
}
inline float_st operator*(const long long& a, const float_st& b){
  float_st res;
  unsigned int random;
  float tmp;

  res.accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = a;
  FLOAT_BIT_FLIP(tmp, random);
  res.x = tmp;
  tmp = b.x;
  res.x *= tmp;
  FLOAT_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp = a;
  FLOAT_BIT_FLIP(tmp, random);
  res.y = tmp;
  tmp = b.y;
  res.y *= tmp;
  FLOAT_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp = a;
  FLOAT_BIT_FLIP(tmp, random);
  res.z = tmp;
  tmp = b.z;
  res.z *= tmp;
  FLOAT_BIT_FLIP(res.z, random);

  return res ;
}
inline float_st operator*(const unsigned long& a, const float_st& b){
  float_st res;
  unsigned int random;
  float tmp;

  res.accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = a;
  FLOAT_BIT_FLIP(tmp, random);
  res.x = tmp;
  tmp = b.x;
  res.x *= tmp;
  FLOAT_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp = a;
  FLOAT_BIT_FLIP(tmp, random);
  res.y = tmp;
  tmp = b.y;
  res.y *= tmp;
  FLOAT_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp = a;
  FLOAT_BIT_FLIP(tmp, random);
  res.z = tmp;
  tmp = b.z;
  res.z *= tmp;
  FLOAT_BIT_FLIP(res.z, random);

  return res ;
}
inline float_st operator*(const long& a, const float_st& b){
  float_st res;
  unsigned int random;
  float tmp;

  res.accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = a;
  FLOAT_BIT_FLIP(tmp, random);
  res.x = tmp;
  tmp = b.x;
  res.x *= tmp;
  FLOAT_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp = a;
  FLOAT_BIT_FLIP(tmp, random);
  res.y = tmp;
  tmp = b.y;
  res.y *= tmp;
  FLOAT_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp = a;
  FLOAT_BIT_FLIP(tmp, random);
  res.z = tmp;
  tmp = b.z;
  res.z *= tmp;
  FLOAT_BIT_FLIP(res.z, random);

  return res ;
}
inline float_st operator*(const unsigned int & a, const float_st& b){
  float_st res;
  unsigned int random;
  float tmp;

  res.accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = a;
  FLOAT_BIT_FLIP(tmp, random);
  res.x = tmp;
  tmp = b.x;
  res.x *= tmp;
  FLOAT_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp = a;
  FLOAT_BIT_FLIP(tmp, random);
  res.y = tmp;
  tmp = b.y;
  res.y *= tmp;
  FLOAT_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp = a;
  FLOAT_BIT_FLIP(tmp, random);
  res.z = tmp;
  tmp = b.z;
  res.z *= tmp;
  FLOAT_BIT_FLIP(res.z, random);

  return res ;
}
inline float_st operator*(const int & a, const float_st& b){
  float_st res;
  unsigned int random;
  float tmp;

  res.accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = a;
  FLOAT_BIT_FLIP(tmp, random);
  res.x = tmp;
  tmp = b.x;
  res.x *= tmp;
  FLOAT_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp = a;
  FLOAT_BIT_FLIP(tmp, random);
  res.y = tmp;
  tmp = b.y;
  res.y *= tmp;
  FLOAT_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp = a;
  FLOAT_BIT_FLIP(tmp, random);
  res.z = tmp;
  tmp = b.z;
  res.z *= tmp;
  FLOAT_BIT_FLIP(res.z, random);

  return res ;
}
inline float_st operator*(const unsigned short& a, const float_st& b){
  float_st res;
  unsigned int random;
  float tmp;

  res.accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = a;
  FLOAT_BIT_FLIP(tmp, random);
  res.x = tmp;
  tmp = b.x;
  res.x *= tmp;
  FLOAT_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp = a;
  FLOAT_BIT_FLIP(tmp, random);
  res.y = tmp;
  tmp = b.y;
  res.y *= tmp;
  FLOAT_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp = a;
  FLOAT_BIT_FLIP(tmp, random);
  res.z = tmp;
  tmp = b.z;
  res.z *= tmp;
  FLOAT_BIT_FLIP(res.z, random);

  return res ;
}
inline float_st operator*(const short& a, const float_st& b){
  float_st res;
  unsigned int random;
  float tmp;

  res.accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = a;
  FLOAT_BIT_FLIP(tmp, random);
  res.x = tmp;
  tmp = b.x;
  res.x *= tmp;
  FLOAT_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp = a;
  FLOAT_BIT_FLIP(tmp, random);
  res.y = tmp;
  tmp = b.y;
  res.y *= tmp;
  FLOAT_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp = a;
  FLOAT_BIT_FLIP(tmp, random);
  res.z = tmp;
  tmp = b.z;
  res.z *= tmp;
  FLOAT_BIT_FLIP(res.z, random);

  return res ;
}
inline float_st operator*(const unsigned char& a, const float_st& b){
  float_st res;
  unsigned int random;
  float tmp;

  res.accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = a;
  FLOAT_BIT_FLIP(tmp, random);
  res.x = tmp;
  tmp = b.x;
  res.x *= tmp;
  FLOAT_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp = a;
  FLOAT_BIT_FLIP(tmp, random);
  res.y = tmp;
  tmp = b.y;
  res.y *= tmp;
  FLOAT_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp = a;
  FLOAT_BIT_FLIP(tmp, random);
  res.z = tmp;
  tmp = b.z;
  res.z *= tmp;
  FLOAT_BIT_FLIP(res.z, random);

  return res ;
}
inline float_st operator*(const char& a, const float_st& b){
  float_st res;
  unsigned int random;
  float tmp;

  res.accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = a;
  FLOAT_BIT_FLIP(tmp, random);
  res.x = tmp;
  tmp = b.x;
  res.x *= tmp;
  FLOAT_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp = a;
  FLOAT_BIT_FLIP(tmp, random);
  res.y = tmp;
  tmp = b.y;
  res.y *= tmp;
  FLOAT_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp = a;
  FLOAT_BIT_FLIP(tmp, random);
  res.z = tmp;
  tmp = b.z;
  res.z *= tmp;
  FLOAT_BIT_FLIP(res.z, random);

  return res ;
}
#ifdef CADNA_HALF_EMULATION
inline float_st operator*(const half& a, const float_st& b){
  float_st res;
  unsigned int random;
  float tmp;

  res.accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = a;
  FLOAT_BIT_FLIP(tmp, random);
  res.x = tmp;
  tmp = b.x;
  res.x *= tmp;
  FLOAT_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp = a;
  FLOAT_BIT_FLIP(tmp, random);
  res.y = tmp;
  tmp = b.y;
  res.y *= tmp;
  FLOAT_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp = a;
  FLOAT_BIT_FLIP(tmp, random);
  res.z = tmp;
  tmp = b.z;
  res.z *= tmp;
  FLOAT_BIT_FLIP(res.z, random);

  return res ;
}
#endif // CADNA_HALF_EMULATION


#ifdef CADNA_HALF_NATIVE
inline float_st operator*(const half& a, const float_st& b){
  float_st res;
  unsigned int random;
  float tmp;

  res.accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = a;
  FLOAT_BIT_FLIP(tmp, random);
  res.x = tmp;
  tmp = b.x;
  res.x *= tmp;
  FLOAT_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp = a;
  FLOAT_BIT_FLIP(tmp, random);
  res.y = tmp;
  tmp = b.y;
  res.y *= tmp;
  FLOAT_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp = a;
  FLOAT_BIT_FLIP(tmp, random);
  res.z = tmp;
  tmp = b.z;
  res.z *= tmp;
  FLOAT_BIT_FLIP(res.z, random);

  return res ;
}
#endif

#ifdef CADNA_HALF_EMULATION
inline half_st operator*(const double& a, const half_st& b){
  half_st res;
  unsigned int random;
  half tmp;

  res.accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = b.x;
  HALF_BIT_FLIP(tmp, random);
  res.x = a * tmp;
  HALF_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp = b.y;
  HALF_BIT_FLIP(tmp, random);
  res.y = a * tmp;
  HALF_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp = b.z;
  HALF_BIT_FLIP(tmp, random);
  res.z = a * tmp;
  HALF_BIT_FLIP(res.z, random);
  
    
    
    if(_cadna_half_overflow_tag){
    if (std::isinf(res.x) or std::isinf(res.y) or std::isinf(res.z)){
    instability(&_cadna_half_overflow_count);
    }
}



  return res ;


}
inline half_st operator*(const float& a, const half_st& b){
  half_st res;
  unsigned int random;
  half tmp;

  res.accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = b.x;
  HALF_BIT_FLIP(tmp, random);
  res.x = a * tmp;
  HALF_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp = b.y;
  HALF_BIT_FLIP(tmp, random);
  res.y = a * tmp;
  HALF_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp = b.z;
  HALF_BIT_FLIP(tmp, random);
  res.z = a * tmp;
  HALF_BIT_FLIP(res.z, random);
  
    
    
    if(_cadna_half_overflow_tag){
    if (std::isinf(res.x) or std::isinf(res.y) or std::isinf(res.z)){
    instability(&_cadna_half_overflow_count);
    }
}



  return res ;


}
inline half_st operator*(const unsigned long long& a, const half_st& b){
  half_st res;
  unsigned int random;
  half tmp;

  res.accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = b.x;
  HALF_BIT_FLIP(tmp, random);
  res.x = a * tmp;
  HALF_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp = b.y;
  HALF_BIT_FLIP(tmp, random);
  res.y = a * tmp;
  HALF_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp = b.z;
  HALF_BIT_FLIP(tmp, random);
  res.z = a * tmp;
  HALF_BIT_FLIP(res.z, random);
  
    
    
    if(_cadna_half_overflow_tag){
    if (std::isinf(res.x) or std::isinf(res.y) or std::isinf(res.z)){
    instability(&_cadna_half_overflow_count);
    }
}



  return res ;


}
inline half_st operator*(const long long& a, const half_st& b){
  half_st res;
  unsigned int random;
  half tmp;

  res.accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = b.x;
  HALF_BIT_FLIP(tmp, random);
  res.x = a * tmp;
  HALF_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp = b.y;
  HALF_BIT_FLIP(tmp, random);
  res.y = a * tmp;
  HALF_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp = b.z;
  HALF_BIT_FLIP(tmp, random);
  res.z = a * tmp;
  HALF_BIT_FLIP(res.z, random);
  
    
    
    if(_cadna_half_overflow_tag){
    if (std::isinf(res.x) or std::isinf(res.y) or std::isinf(res.z)){
    instability(&_cadna_half_overflow_count);
    }
}



  return res ;


}
inline half_st operator*(const unsigned long& a, const half_st& b){
  half_st res;
  unsigned int random;
  half tmp;

  res.accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = b.x;
  HALF_BIT_FLIP(tmp, random);
  res.x = a * tmp;
  HALF_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp = b.y;
  HALF_BIT_FLIP(tmp, random);
  res.y = a * tmp;
  HALF_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp = b.z;
  HALF_BIT_FLIP(tmp, random);
  res.z = a * tmp;
  HALF_BIT_FLIP(res.z, random);
  
    
    
    if(_cadna_half_overflow_tag){
    if (std::isinf(res.x) or std::isinf(res.y) or std::isinf(res.z)){
    instability(&_cadna_half_overflow_count);
    }
}



  return res ;


}
inline half_st operator*(const long& a, const half_st& b){
  half_st res;
  unsigned int random;
  half tmp;

  res.accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = b.x;
  HALF_BIT_FLIP(tmp, random);
  res.x = a * tmp;
  HALF_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp = b.y;
  HALF_BIT_FLIP(tmp, random);
  res.y = a * tmp;
  HALF_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp = b.z;
  HALF_BIT_FLIP(tmp, random);
  res.z = a * tmp;
  HALF_BIT_FLIP(res.z, random);
  
    
    
    if(_cadna_half_overflow_tag){
    if (std::isinf(res.x) or std::isinf(res.y) or std::isinf(res.z)){
    instability(&_cadna_half_overflow_count);
    }
}



  return res ;


}
inline half_st operator*(const unsigned int & a, const half_st& b){
  half_st res;
  unsigned int random;
  half tmp;

  res.accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = b.x;
  HALF_BIT_FLIP(tmp, random);
  res.x = a * tmp;
  HALF_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp = b.y;
  HALF_BIT_FLIP(tmp, random);
  res.y = a * tmp;
  HALF_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp = b.z;
  HALF_BIT_FLIP(tmp, random);
  res.z = a * tmp;
  HALF_BIT_FLIP(res.z, random);
  
    
    
    if(_cadna_half_overflow_tag){
    if (std::isinf(res.x) or std::isinf(res.y) or std::isinf(res.z)){
    instability(&_cadna_half_overflow_count);
    }
}



  return res ;


}
inline half_st operator*(const int & a, const half_st& b){
  half_st res;
  unsigned int random;
  half tmp;

  res.accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = b.x;
  HALF_BIT_FLIP(tmp, random);
  res.x = a * tmp;
  HALF_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp = b.y;
  HALF_BIT_FLIP(tmp, random);
  res.y = a * tmp;
  HALF_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp = b.z;
  HALF_BIT_FLIP(tmp, random);
  res.z = a * tmp;
  HALF_BIT_FLIP(res.z, random);
  
    
    
    if(_cadna_half_overflow_tag){
    if (std::isinf(res.x) or std::isinf(res.y) or std::isinf(res.z)){
    instability(&_cadna_half_overflow_count);
    }
}



  return res ;


}
inline half_st operator*(const unsigned short& a, const half_st& b){
  half_st res;
  unsigned int random;
  half tmp;

  res.accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = b.x;
  HALF_BIT_FLIP(tmp, random);
  res.x = a * tmp;
  HALF_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp = b.y;
  HALF_BIT_FLIP(tmp, random);
  res.y = a * tmp;
  HALF_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp = b.z;
  HALF_BIT_FLIP(tmp, random);
  res.z = a * tmp;
  HALF_BIT_FLIP(res.z, random);
  
    
    
    if(_cadna_half_overflow_tag){
    if (std::isinf(res.x) or std::isinf(res.y) or std::isinf(res.z)){
    instability(&_cadna_half_overflow_count);
    }
}



  return res ;


}
inline half_st operator*(const short& a, const half_st& b){
  half_st res;
  unsigned int random;
  half tmp;

  res.accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = b.x;
  HALF_BIT_FLIP(tmp, random);
  res.x = a * tmp;
  HALF_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp = b.y;
  HALF_BIT_FLIP(tmp, random);
  res.y = a * tmp;
  HALF_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp = b.z;
  HALF_BIT_FLIP(tmp, random);
  res.z = a * tmp;
  HALF_BIT_FLIP(res.z, random);
  
    
    
    if(_cadna_half_overflow_tag){
    if (std::isinf(res.x) or std::isinf(res.y) or std::isinf(res.z)){
    instability(&_cadna_half_overflow_count);
    }
}



  return res ;


}
inline half_st operator*(const unsigned char& a, const half_st& b){
  half_st res;
  unsigned int random;
  half tmp;

  res.accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = b.x;
  HALF_BIT_FLIP(tmp, random);
  res.x = a * tmp;
  HALF_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp = b.y;
  HALF_BIT_FLIP(tmp, random);
  res.y = a * tmp;
  HALF_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp = b.z;
  HALF_BIT_FLIP(tmp, random);
  res.z = a * tmp;
  HALF_BIT_FLIP(res.z, random);
  
    
    
    if(_cadna_half_overflow_tag){
    if (std::isinf(res.x) or std::isinf(res.y) or std::isinf(res.z)){
    instability(&_cadna_half_overflow_count);
    }
}



  return res ;


}
inline half_st operator*(const char& a, const half_st& b){
  half_st res;
  unsigned int random;
  half tmp;

  res.accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = b.x;
  HALF_BIT_FLIP(tmp, random);
  res.x = a * tmp;
  HALF_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp = b.y;
  HALF_BIT_FLIP(tmp, random);
  res.y = a * tmp;
  HALF_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp = b.z;
  HALF_BIT_FLIP(tmp, random);
  res.z = a * tmp;
  HALF_BIT_FLIP(res.z, random);
  
    
    
    if(_cadna_half_overflow_tag){
    if (std::isinf(res.x) or std::isinf(res.y) or std::isinf(res.z)){
    instability(&_cadna_half_overflow_count);
    }
}



  return res ;


}
inline half_st operator*(const half& a, const half_st& b){
  half_st res;
  unsigned int random;
  half tmp;

  res.accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = b.x;
  HALF_BIT_FLIP(tmp, random);
  res.x = a * tmp;
  HALF_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp = b.y;
  HALF_BIT_FLIP(tmp, random);
  res.y = a * tmp;
  HALF_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp = b.z;
  HALF_BIT_FLIP(tmp, random);
  res.z = a * tmp;
  HALF_BIT_FLIP(res.z, random);
  
    
    
    if(_cadna_half_overflow_tag){
    if (std::isinf(res.x) or std::isinf(res.y) or std::isinf(res.z)){
    instability(&_cadna_half_overflow_count);
    }
}



  return res ;


}
#endif // CADNA_HALF_EMULATION


#ifdef CADNA_HALF_NATIVE
inline half_st operator*(const double& a, const half_st& b){
  half_st res;
  unsigned int random;
  half tmp;

  res.accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = b.x;
  HALF_BIT_FLIP(tmp, random);
  res.x = a * tmp;
  HALF_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp = b.y;
  HALF_BIT_FLIP(tmp, random);
  res.y = a * tmp;
  HALF_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp = b.z;
  HALF_BIT_FLIP(tmp, random);
  res.z = a * tmp;
  HALF_BIT_FLIP(res.z, random);
  
    
    
    if(_cadna_half_overflow_tag){
    if (std::isinf(res.x) or std::isinf(res.y) or std::isinf(res.z)){
    instability(&_cadna_half_overflow_count);
    }
}



  return res ;


}
inline half_st operator*(const float& a, const half_st& b){
  half_st res;
  unsigned int random;
  half tmp;

  res.accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = b.x;
  HALF_BIT_FLIP(tmp, random);
  res.x = a * tmp;
  HALF_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp = b.y;
  HALF_BIT_FLIP(tmp, random);
  res.y = a * tmp;
  HALF_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp = b.z;
  HALF_BIT_FLIP(tmp, random);
  res.z = a * tmp;
  HALF_BIT_FLIP(res.z, random);
  
    
    
    if(_cadna_half_overflow_tag){
    if (std::isinf(res.x) or std::isinf(res.y) or std::isinf(res.z)){
    instability(&_cadna_half_overflow_count);
    }
}



  return res ;


}
inline half_st operator*(const unsigned long long& a, const half_st& b){
  half_st res;
  unsigned int random;
  half tmp;

  res.accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = b.x;
  HALF_BIT_FLIP(tmp, random);
  res.x = a * tmp;
  HALF_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp = b.y;
  HALF_BIT_FLIP(tmp, random);
  res.y = a * tmp;
  HALF_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp = b.z;
  HALF_BIT_FLIP(tmp, random);
  res.z = a * tmp;
  HALF_BIT_FLIP(res.z, random);
  
    
    
    if(_cadna_half_overflow_tag){
    if (std::isinf(res.x) or std::isinf(res.y) or std::isinf(res.z)){
    instability(&_cadna_half_overflow_count);
    }
}



  return res ;


}
inline half_st operator*(const long long& a, const half_st& b){
  half_st res;
  unsigned int random;
  half tmp;

  res.accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = b.x;
  HALF_BIT_FLIP(tmp, random);
  res.x = a * tmp;
  HALF_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp = b.y;
  HALF_BIT_FLIP(tmp, random);
  res.y = a * tmp;
  HALF_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp = b.z;
  HALF_BIT_FLIP(tmp, random);
  res.z = a * tmp;
  HALF_BIT_FLIP(res.z, random);
  
    
    
    if(_cadna_half_overflow_tag){
    if (std::isinf(res.x) or std::isinf(res.y) or std::isinf(res.z)){
    instability(&_cadna_half_overflow_count);
    }
}



  return res ;


}
inline half_st operator*(const unsigned long& a, const half_st& b){
  half_st res;
  unsigned int random;
  half tmp;

  res.accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = b.x;
  HALF_BIT_FLIP(tmp, random);
  res.x = a * tmp;
  HALF_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp = b.y;
  HALF_BIT_FLIP(tmp, random);
  res.y = a * tmp;
  HALF_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp = b.z;
  HALF_BIT_FLIP(tmp, random);
  res.z = a * tmp;
  HALF_BIT_FLIP(res.z, random);
  
    
    
    if(_cadna_half_overflow_tag){
    if (std::isinf(res.x) or std::isinf(res.y) or std::isinf(res.z)){
    instability(&_cadna_half_overflow_count);
    }
}



  return res ;


}
inline half_st operator*(const long& a, const half_st& b){
  half_st res;
  unsigned int random;
  half tmp;

  res.accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = b.x;
  HALF_BIT_FLIP(tmp, random);
  res.x = a * tmp;
  HALF_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp = b.y;
  HALF_BIT_FLIP(tmp, random);
  res.y = a * tmp;
  HALF_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp = b.z;
  HALF_BIT_FLIP(tmp, random);
  res.z = a * tmp;
  HALF_BIT_FLIP(res.z, random);
  
    
    
    if(_cadna_half_overflow_tag){
    if (std::isinf(res.x) or std::isinf(res.y) or std::isinf(res.z)){
    instability(&_cadna_half_overflow_count);
    }
}



  return res ;


}
inline half_st operator*(const unsigned int & a, const half_st& b){
  half_st res;
  unsigned int random;
  half tmp;

  res.accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = b.x;
  HALF_BIT_FLIP(tmp, random);
  res.x = a * tmp;
  HALF_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp = b.y;
  HALF_BIT_FLIP(tmp, random);
  res.y = a * tmp;
  HALF_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp = b.z;
  HALF_BIT_FLIP(tmp, random);
  res.z = a * tmp;
  HALF_BIT_FLIP(res.z, random);
  
    
    
    if(_cadna_half_overflow_tag){
    if (std::isinf(res.x) or std::isinf(res.y) or std::isinf(res.z)){
    instability(&_cadna_half_overflow_count);
    }
}



  return res ;


}
inline half_st operator*(const int & a, const half_st& b){
  half_st res;
  unsigned int random;
  half tmp;

  res.accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = b.x;
  HALF_BIT_FLIP(tmp, random);
  res.x = a * tmp;
  HALF_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp = b.y;
  HALF_BIT_FLIP(tmp, random);
  res.y = a * tmp;
  HALF_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp = b.z;
  HALF_BIT_FLIP(tmp, random);
  res.z = a * tmp;
  HALF_BIT_FLIP(res.z, random);
  
    
    
    if(_cadna_half_overflow_tag){
    if (std::isinf(res.x) or std::isinf(res.y) or std::isinf(res.z)){
    instability(&_cadna_half_overflow_count);
    }
}



  return res ;


}
inline half_st operator*(const unsigned short& a, const half_st& b){
  half_st res;
  unsigned int random;
  half tmp;

  res.accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = b.x;
  HALF_BIT_FLIP(tmp, random);
  res.x = a * tmp;
  HALF_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp = b.y;
  HALF_BIT_FLIP(tmp, random);
  res.y = a * tmp;
  HALF_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp = b.z;
  HALF_BIT_FLIP(tmp, random);
  res.z = a * tmp;
  HALF_BIT_FLIP(res.z, random);
  
    
    
    if(_cadna_half_overflow_tag){
    if (std::isinf(res.x) or std::isinf(res.y) or std::isinf(res.z)){
    instability(&_cadna_half_overflow_count);
    }
}



  return res ;


}
inline half_st operator*(const short& a, const half_st& b){
  half_st res;
  unsigned int random;
  half tmp;

  res.accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = b.x;
  HALF_BIT_FLIP(tmp, random);
  res.x = a * tmp;
  HALF_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp = b.y;
  HALF_BIT_FLIP(tmp, random);
  res.y = a * tmp;
  HALF_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp = b.z;
  HALF_BIT_FLIP(tmp, random);
  res.z = a * tmp;
  HALF_BIT_FLIP(res.z, random);
  
    
    
    if(_cadna_half_overflow_tag){
    if (std::isinf(res.x) or std::isinf(res.y) or std::isinf(res.z)){
    instability(&_cadna_half_overflow_count);
    }
}



  return res ;


}
inline half_st operator*(const unsigned char& a, const half_st& b){
  half_st res;
  unsigned int random;
  half tmp;

  res.accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = b.x;
  HALF_BIT_FLIP(tmp, random);
  res.x = a * tmp;
  HALF_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp = b.y;
  HALF_BIT_FLIP(tmp, random);
  res.y = a * tmp;
  HALF_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp = b.z;
  HALF_BIT_FLIP(tmp, random);
  res.z = a * tmp;
  HALF_BIT_FLIP(res.z, random);
  
    
    
    if(_cadna_half_overflow_tag){
    if (std::isinf(res.x) or std::isinf(res.y) or std::isinf(res.z)){
    instability(&_cadna_half_overflow_count);
    }
}



  return res ;


}
inline half_st operator*(const char& a, const half_st& b){
  half_st res;
  unsigned int random;
  half tmp;

  res.accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = b.x;
  HALF_BIT_FLIP(tmp, random);
  res.x = a * tmp;
  HALF_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp = b.y;
  HALF_BIT_FLIP(tmp, random);
  res.y = a * tmp;
  HALF_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp = b.z;
  HALF_BIT_FLIP(tmp, random);
  res.z = a * tmp;
  HALF_BIT_FLIP(res.z, random);
  
    
    
    if(_cadna_half_overflow_tag){
    if (std::isinf(res.x) or std::isinf(res.y) or std::isinf(res.z)){
    instability(&_cadna_half_overflow_count);
    }
}



  return res ;


}
inline half_st operator*(const half& a, const half_st& b){
  half_st res;
  unsigned int random;
  half tmp;

  res.accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = b.x;
  HALF_BIT_FLIP(tmp, random);
  res.x = a * tmp;
  HALF_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp = b.y;
  HALF_BIT_FLIP(tmp, random);
  res.y = a * tmp;
  HALF_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp = b.z;
  HALF_BIT_FLIP(tmp, random);
  res.z = a * tmp;
  HALF_BIT_FLIP(res.z, random);
  
    
    
    if(_cadna_half_overflow_tag){
    if (std::isinf(res.x) or std::isinf(res.y) or std::isinf(res.z)){
    instability(&_cadna_half_overflow_count);
    }
}



  return res ;


}
#endif // CADNA_HALF_NATIVE
/////////////////////////////////////////////////////////////////////////






inline double_st operator*(const double_st& a, const double& b){
  double_st res;
  unsigned int random;
  double tmp;

  res.accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = a.x;
  DOUBLE_BIT_FLIP(tmp, random);
  res.x = tmp;
  tmp = b;
  res.x *= tmp;
  DOUBLE_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp = a.y;
  DOUBLE_BIT_FLIP(tmp, random);
  res.y = tmp;
  tmp = b;
  res.y *= tmp;
  DOUBLE_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp = a.z;
  DOUBLE_BIT_FLIP(tmp, random);
  res.z = tmp;
  tmp = b;
  res.z *= tmp;
  DOUBLE_BIT_FLIP(res.z, random);

  return res ;
}
inline double_st operator*(const double_st& a, const float& b){
  double_st res;
  unsigned int random;
  double tmp;

  res.accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = a.x;
  DOUBLE_BIT_FLIP(tmp, random);
  res.x = tmp;
  tmp = b;
  res.x *= tmp;
  DOUBLE_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp = a.y;
  DOUBLE_BIT_FLIP(tmp, random);
  res.y = tmp;
  tmp = b;
  res.y *= tmp;
  DOUBLE_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp = a.z;
  DOUBLE_BIT_FLIP(tmp, random);
  res.z = tmp;
  tmp = b;
  res.z *= tmp;
  DOUBLE_BIT_FLIP(res.z, random);

  return res ;
}
inline double_st operator*(const double_st& a, const unsigned long long& b){
  double_st res;
  unsigned int random;
  double tmp;

  res.accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = a.x;
  DOUBLE_BIT_FLIP(tmp, random);
  res.x = tmp;
  tmp = b;
  res.x *= tmp;
  DOUBLE_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp = a.y;
  DOUBLE_BIT_FLIP(tmp, random);
  res.y = tmp;
  tmp = b;
  res.y *= tmp;
  DOUBLE_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp = a.z;
  DOUBLE_BIT_FLIP(tmp, random);
  res.z = tmp;
  tmp = b;
  res.z *= tmp;
  DOUBLE_BIT_FLIP(res.z, random);

  return res ;
}
inline double_st operator*(const double_st& a, const long long& b){
  double_st res;
  unsigned int random;
  double tmp;

  res.accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = a.x;
  DOUBLE_BIT_FLIP(tmp, random);
  res.x = tmp;
  tmp = b;
  res.x *= tmp;
  DOUBLE_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp = a.y;
  DOUBLE_BIT_FLIP(tmp, random);
  res.y = tmp;
  tmp = b;
  res.y *= tmp;
  DOUBLE_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp = a.z;
  DOUBLE_BIT_FLIP(tmp, random);
  res.z = tmp;
  tmp = b;
  res.z *= tmp;
  DOUBLE_BIT_FLIP(res.z, random);

  return res ;
}
inline double_st operator*(const double_st& a, const unsigned long& b){
  double_st res;
  unsigned int random;
  double tmp;

  res.accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = a.x;
  DOUBLE_BIT_FLIP(tmp, random);
  res.x = tmp;
  tmp = b;
  res.x *= tmp;
  DOUBLE_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp = a.y;
  DOUBLE_BIT_FLIP(tmp, random);
  res.y = tmp;
  tmp = b;
  res.y *= tmp;
  DOUBLE_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp = a.z;
  DOUBLE_BIT_FLIP(tmp, random);
  res.z = tmp;
  tmp = b;
  res.z *= tmp;
  DOUBLE_BIT_FLIP(res.z, random);

  return res ;
}
inline double_st operator*(const double_st& a, const long& b){
  double_st res;
  unsigned int random;
  double tmp;

  res.accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = a.x;
  DOUBLE_BIT_FLIP(tmp, random);
  res.x = tmp;
  tmp = b;
  res.x *= tmp;
  DOUBLE_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp = a.y;
  DOUBLE_BIT_FLIP(tmp, random);
  res.y = tmp;
  tmp = b;
  res.y *= tmp;
  DOUBLE_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp = a.z;
  DOUBLE_BIT_FLIP(tmp, random);
  res.z = tmp;
  tmp = b;
  res.z *= tmp;
  DOUBLE_BIT_FLIP(res.z, random);

  return res ;
}
inline double_st operator*(const double_st& a, const unsigned int & b){
  double_st res;
  unsigned int random;
  double tmp;

  res.accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = a.x;
  DOUBLE_BIT_FLIP(tmp, random);
  res.x = tmp;
  tmp = b;
  res.x *= tmp;
  DOUBLE_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp = a.y;
  DOUBLE_BIT_FLIP(tmp, random);
  res.y = tmp;
  tmp = b;
  res.y *= tmp;
  DOUBLE_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp = a.z;
  DOUBLE_BIT_FLIP(tmp, random);
  res.z = tmp;
  tmp = b;
  res.z *= tmp;
  DOUBLE_BIT_FLIP(res.z, random);

  return res ;
}
inline double_st operator*(const double_st& a, const int & b){
  double_st res;
  unsigned int random;
  double tmp;

  res.accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = a.x;
  DOUBLE_BIT_FLIP(tmp, random);
  res.x = tmp;
  tmp = b;
  res.x *= tmp;
  DOUBLE_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp = a.y;
  DOUBLE_BIT_FLIP(tmp, random);
  res.y = tmp;
  tmp = b;
  res.y *= tmp;
  DOUBLE_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp = a.z;
  DOUBLE_BIT_FLIP(tmp, random);
  res.z = tmp;
  tmp = b;
  res.z *= tmp;
  DOUBLE_BIT_FLIP(res.z, random);

  return res ;
}
inline double_st operator*(const double_st& a, const unsigned short& b){
  double_st res;
  unsigned int random;
  double tmp;

  res.accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = a.x;
  DOUBLE_BIT_FLIP(tmp, random);
  res.x = tmp;
  tmp = b;
  res.x *= tmp;
  DOUBLE_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp = a.y;
  DOUBLE_BIT_FLIP(tmp, random);
  res.y = tmp;
  tmp = b;
  res.y *= tmp;
  DOUBLE_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp = a.z;
  DOUBLE_BIT_FLIP(tmp, random);
  res.z = tmp;
  tmp = b;
  res.z *= tmp;
  DOUBLE_BIT_FLIP(res.z, random);

  return res ;
}
inline double_st operator*(const double_st& a, const short& b){
  double_st res;
  unsigned int random;
  double tmp;

  res.accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = a.x;
  DOUBLE_BIT_FLIP(tmp, random);
  res.x = tmp;
  tmp = b;
  res.x *= tmp;
  DOUBLE_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp = a.y;
  DOUBLE_BIT_FLIP(tmp, random);
  res.y = tmp;
  tmp = b;
  res.y *= tmp;
  DOUBLE_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp = a.z;
  DOUBLE_BIT_FLIP(tmp, random);
  res.z = tmp;
  tmp = b;
  res.z *= tmp;
  DOUBLE_BIT_FLIP(res.z, random);

  return res ;
}
inline double_st operator*(const double_st& a, const unsigned char& b){
  double_st res;
  unsigned int random;
  double tmp;

  res.accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = a.x;
  DOUBLE_BIT_FLIP(tmp, random);
  res.x = tmp;
  tmp = b;
  res.x *= tmp;
  DOUBLE_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp = a.y;
  DOUBLE_BIT_FLIP(tmp, random);
  res.y = tmp;
  tmp = b;
  res.y *= tmp;
  DOUBLE_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp = a.z;
  DOUBLE_BIT_FLIP(tmp, random);
  res.z = tmp;
  tmp = b;
  res.z *= tmp;
  DOUBLE_BIT_FLIP(res.z, random);

  return res ;
}
inline double_st operator*(const double_st& a, const char& b){
  double_st res;
  unsigned int random;
  double tmp;

  res.accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = a.x;
  DOUBLE_BIT_FLIP(tmp, random);
  res.x = tmp;
  tmp = b;
  res.x *= tmp;
  DOUBLE_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp = a.y;
  DOUBLE_BIT_FLIP(tmp, random);
  res.y = tmp;
  tmp = b;
  res.y *= tmp;
  DOUBLE_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp = a.z;
  DOUBLE_BIT_FLIP(tmp, random);
  res.z = tmp;
  tmp = b;
  res.z *= tmp;
  DOUBLE_BIT_FLIP(res.z, random);

  return res ;
}
#ifdef CADNA_HALF_EMULATION
inline double_st operator*(const double_st& a, const half& b){
  double_st res;
  unsigned int random;
  double tmp;

  res.accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = a.x;
  DOUBLE_BIT_FLIP(tmp, random);
  res.x = tmp;
  tmp = b;
  res.x *= tmp;
  DOUBLE_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp = a.y;
  DOUBLE_BIT_FLIP(tmp, random);
  res.y = tmp;
  tmp = b;
  res.y *= tmp;
  DOUBLE_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp = a.z;
  DOUBLE_BIT_FLIP(tmp, random);
  res.z = tmp;
  tmp = b;
  res.z *= tmp;
  DOUBLE_BIT_FLIP(res.z, random);

  return res ;
}
#endif // CADNA_HALF_EMULATION


#ifdef CADNA_HALF_NATIVE
inline double_st operator*(const double_st& a, const half& b){
  double_st res;
  unsigned int random;
  double tmp;

  res.accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = a.x;
  DOUBLE_BIT_FLIP(tmp, random);
  res.x = tmp;
  tmp = b;
  res.x *= tmp;
  DOUBLE_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp = a.y;
  DOUBLE_BIT_FLIP(tmp, random);
  res.y = tmp;
  tmp = b;
  res.y *= tmp;
  DOUBLE_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp = a.z;
  DOUBLE_BIT_FLIP(tmp, random);
  res.z = tmp;
  tmp = b;
  res.z *= tmp;
  DOUBLE_BIT_FLIP(res.z, random);

  return res ;
}
#endif


inline float_st operator*(const float_st& a, const double& b){
  float_st res;
  unsigned int random;
  float tmp;

  res.accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = a.x;
  FLOAT_BIT_FLIP(tmp, random);
  res.x = tmp;
  tmp = b;
  res.x *= tmp;
  FLOAT_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp = a.y;
  FLOAT_BIT_FLIP(tmp, random);
  res.y = tmp;
  tmp = b;
  res.y *= tmp;
  FLOAT_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp = a.z;
  FLOAT_BIT_FLIP(tmp, random);
  res.z = tmp;
  tmp = b;
  res.z *= tmp;
  FLOAT_BIT_FLIP(res.z, random);

  return res ;
}
inline float_st operator*(const float_st& a, const float& b){
  float_st res;
  unsigned int random;
  float tmp;

  res.accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = a.x;
  FLOAT_BIT_FLIP(tmp, random);
  res.x = tmp;
  tmp = b;
  res.x *= tmp;
  FLOAT_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp = a.y;
  FLOAT_BIT_FLIP(tmp, random);
  res.y = tmp;
  tmp = b;
  res.y *= tmp;
  FLOAT_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp = a.z;
  FLOAT_BIT_FLIP(tmp, random);
  res.z = tmp;
  tmp = b;
  res.z *= tmp;
  FLOAT_BIT_FLIP(res.z, random);

  return res ;
}
inline float_st operator*(const float_st& a, const unsigned long long& b){
  float_st res;
  unsigned int random;
  float tmp;

  res.accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = a.x;
  FLOAT_BIT_FLIP(tmp, random);
  res.x = tmp;
  tmp = b;
  res.x *= tmp;
  FLOAT_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp = a.y;
  FLOAT_BIT_FLIP(tmp, random);
  res.y = tmp;
  tmp = b;
  res.y *= tmp;
  FLOAT_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp = a.z;
  FLOAT_BIT_FLIP(tmp, random);
  res.z = tmp;
  tmp = b;
  res.z *= tmp;
  FLOAT_BIT_FLIP(res.z, random);

  return res ;
}
inline float_st operator*(const float_st& a, const long long& b){
  float_st res;
  unsigned int random;
  float tmp;

  res.accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = a.x;
  FLOAT_BIT_FLIP(tmp, random);
  res.x = tmp;
  tmp = b;
  res.x *= tmp;
  FLOAT_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp = a.y;
  FLOAT_BIT_FLIP(tmp, random);
  res.y = tmp;
  tmp = b;
  res.y *= tmp;
  FLOAT_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp = a.z;
  FLOAT_BIT_FLIP(tmp, random);
  res.z = tmp;
  tmp = b;
  res.z *= tmp;
  FLOAT_BIT_FLIP(res.z, random);

  return res ;
}
inline float_st operator*(const float_st& a, const unsigned long& b){
  float_st res;
  unsigned int random;
  float tmp;

  res.accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = a.x;
  FLOAT_BIT_FLIP(tmp, random);
  res.x = tmp;
  tmp = b;
  res.x *= tmp;
  FLOAT_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp = a.y;
  FLOAT_BIT_FLIP(tmp, random);
  res.y = tmp;
  tmp = b;
  res.y *= tmp;
  FLOAT_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp = a.z;
  FLOAT_BIT_FLIP(tmp, random);
  res.z = tmp;
  tmp = b;
  res.z *= tmp;
  FLOAT_BIT_FLIP(res.z, random);

  return res ;
}
inline float_st operator*(const float_st& a, const long& b){
  float_st res;
  unsigned int random;
  float tmp;

  res.accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = a.x;
  FLOAT_BIT_FLIP(tmp, random);
  res.x = tmp;
  tmp = b;
  res.x *= tmp;
  FLOAT_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp = a.y;
  FLOAT_BIT_FLIP(tmp, random);
  res.y = tmp;
  tmp = b;
  res.y *= tmp;
  FLOAT_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp = a.z;
  FLOAT_BIT_FLIP(tmp, random);
  res.z = tmp;
  tmp = b;
  res.z *= tmp;
  FLOAT_BIT_FLIP(res.z, random);

  return res ;
}
inline float_st operator*(const float_st& a, const unsigned int & b){
  float_st res;
  unsigned int random;
  float tmp;

  res.accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = a.x;
  FLOAT_BIT_FLIP(tmp, random);
  res.x = tmp;
  tmp = b;
  res.x *= tmp;
  FLOAT_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp = a.y;
  FLOAT_BIT_FLIP(tmp, random);
  res.y = tmp;
  tmp = b;
  res.y *= tmp;
  FLOAT_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp = a.z;
  FLOAT_BIT_FLIP(tmp, random);
  res.z = tmp;
  tmp = b;
  res.z *= tmp;
  FLOAT_BIT_FLIP(res.z, random);

  return res ;
}
inline float_st operator*(const float_st& a, const int & b){
  float_st res;
  unsigned int random;
  float tmp;

  res.accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = a.x;
  FLOAT_BIT_FLIP(tmp, random);
  res.x = tmp;
  tmp = b;
  res.x *= tmp;
  FLOAT_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp = a.y;
  FLOAT_BIT_FLIP(tmp, random);
  res.y = tmp;
  tmp = b;
  res.y *= tmp;
  FLOAT_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp = a.z;
  FLOAT_BIT_FLIP(tmp, random);
  res.z = tmp;
  tmp = b;
  res.z *= tmp;
  FLOAT_BIT_FLIP(res.z, random);

  return res ;
}
inline float_st operator*(const float_st& a, const unsigned short& b){
  float_st res;
  unsigned int random;
  float tmp;

  res.accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = a.x;
  FLOAT_BIT_FLIP(tmp, random);
  res.x = tmp;
  tmp = b;
  res.x *= tmp;
  FLOAT_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp = a.y;
  FLOAT_BIT_FLIP(tmp, random);
  res.y = tmp;
  tmp = b;
  res.y *= tmp;
  FLOAT_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp = a.z;
  FLOAT_BIT_FLIP(tmp, random);
  res.z = tmp;
  tmp = b;
  res.z *= tmp;
  FLOAT_BIT_FLIP(res.z, random);

  return res ;
}
inline float_st operator*(const float_st& a, const short& b){
  float_st res;
  unsigned int random;
  float tmp;

  res.accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = a.x;
  FLOAT_BIT_FLIP(tmp, random);
  res.x = tmp;
  tmp = b;
  res.x *= tmp;
  FLOAT_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp = a.y;
  FLOAT_BIT_FLIP(tmp, random);
  res.y = tmp;
  tmp = b;
  res.y *= tmp;
  FLOAT_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp = a.z;
  FLOAT_BIT_FLIP(tmp, random);
  res.z = tmp;
  tmp = b;
  res.z *= tmp;
  FLOAT_BIT_FLIP(res.z, random);

  return res ;
}
inline float_st operator*(const float_st& a, const unsigned char& b){
  float_st res;
  unsigned int random;
  float tmp;

  res.accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = a.x;
  FLOAT_BIT_FLIP(tmp, random);
  res.x = tmp;
  tmp = b;
  res.x *= tmp;
  FLOAT_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp = a.y;
  FLOAT_BIT_FLIP(tmp, random);
  res.y = tmp;
  tmp = b;
  res.y *= tmp;
  FLOAT_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp = a.z;
  FLOAT_BIT_FLIP(tmp, random);
  res.z = tmp;
  tmp = b;
  res.z *= tmp;
  FLOAT_BIT_FLIP(res.z, random);

  return res ;
}
inline float_st operator*(const float_st& a, const char& b){
  float_st res;
  unsigned int random;
  float tmp;

  res.accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = a.x;
  FLOAT_BIT_FLIP(tmp, random);
  res.x = tmp;
  tmp = b;
  res.x *= tmp;
  FLOAT_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp = a.y;
  FLOAT_BIT_FLIP(tmp, random);
  res.y = tmp;
  tmp = b;
  res.y *= tmp;
  FLOAT_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp = a.z;
  FLOAT_BIT_FLIP(tmp, random);
  res.z = tmp;
  tmp = b;
  res.z *= tmp;
  FLOAT_BIT_FLIP(res.z, random);

  return res ;
}
#ifdef CADNA_HALF_EMULATION
inline float_st operator*(const float_st& a, const half& b){
  float_st res;
  unsigned int random;
  float tmp;

  res.accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = a.x;
  FLOAT_BIT_FLIP(tmp, random);
  res.x = tmp;
  tmp = b;
  res.x *= tmp;
  FLOAT_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp = a.y;
  FLOAT_BIT_FLIP(tmp, random);
  res.y = tmp;
  tmp = b;
  res.y *= tmp;
  FLOAT_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp = a.z;
  FLOAT_BIT_FLIP(tmp, random);
  res.z = tmp;
  tmp = b;
  res.z *= tmp;
  FLOAT_BIT_FLIP(res.z, random);

  return res ;
}
#endif // CADNA_HALF_EMULATION

#ifdef CADNA_HALF_NATIVE
inline float_st operator*(const float_st& a, const half& b){
  float_st res;
  unsigned int random;
  float tmp;

  res.accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = a.x;
  FLOAT_BIT_FLIP(tmp, random);
  res.x = tmp;
  tmp = b;
  res.x *= tmp;
  FLOAT_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp = a.y;
  FLOAT_BIT_FLIP(tmp, random);
  res.y = tmp;
  tmp = b;
  res.y *= tmp;
  FLOAT_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp = a.z;
  FLOAT_BIT_FLIP(tmp, random);
  res.z = tmp;
  tmp = b;
  res.z *= tmp;
  FLOAT_BIT_FLIP(res.z, random);

  return res ;
}
#endif

#ifdef CADNA_HALF_EMULATION
inline half_st operator*(const half_st& a, const double& b){
  half_st res;
  unsigned int random;
  half tmp;

  res.accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = a.x;
  HALF_BIT_FLIP(tmp, random);
  res.x = tmp;
  res.x *= b;
  HALF_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp = a.y;
  HALF_BIT_FLIP(tmp, random);
  res.y = tmp;
  res.y *= b;
  HALF_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp = a.z;
  HALF_BIT_FLIP(tmp, random);
  res.z = tmp;
  res.z *= b;
  HALF_BIT_FLIP(res.z, random);
  
  
  
    if(_cadna_half_overflow_tag){
    if (std::isinf(res.x) or std::isinf(res.y) or std::isinf(res.z)){
    instability(&_cadna_half_overflow_count);
    }
}



  return res ;


}
inline half_st operator*(const half_st& a, const float& b){
  half_st res;
  unsigned int random;
  half tmp;

  res.accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = a.x;
  HALF_BIT_FLIP(tmp, random);
  res.x = tmp;
  res.x *= b;
  HALF_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp = a.y;
  HALF_BIT_FLIP(tmp, random);
  res.y = tmp;
  res.y *= b;
  HALF_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp = a.z;
  HALF_BIT_FLIP(tmp, random);
  res.z = tmp;
  res.z *= b;
  HALF_BIT_FLIP(res.z, random);
  
  
  
    if(_cadna_half_overflow_tag){
    if (std::isinf(res.x) or std::isinf(res.y) or std::isinf(res.z)){
    instability(&_cadna_half_overflow_count);
    }
}



  return res ;


}
inline half_st operator*(const half_st& a, const unsigned long long& b){
  half_st res;
  unsigned int random;
  half tmp;

  res.accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = a.x;
  HALF_BIT_FLIP(tmp, random);
  res.x = tmp;
  res.x *= b;
  HALF_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp = a.y;
  HALF_BIT_FLIP(tmp, random);
  res.y = tmp;
  res.y *= b;
  HALF_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp = a.z;
  HALF_BIT_FLIP(tmp, random);
  res.z = tmp;
  res.z *= b;
  HALF_BIT_FLIP(res.z, random);
  
  
  
    if(_cadna_half_overflow_tag){
    if (std::isinf(res.x) or std::isinf(res.y) or std::isinf(res.z)){
    instability(&_cadna_half_overflow_count);
    }
}



  return res ;


}
inline half_st operator*(const half_st& a, const long long& b){
  half_st res;
  unsigned int random;
  half tmp;

  res.accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = a.x;
  HALF_BIT_FLIP(tmp, random);
  res.x = tmp;
  res.x *= b;
  HALF_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp = a.y;
  HALF_BIT_FLIP(tmp, random);
  res.y = tmp;
  res.y *= b;
  HALF_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp = a.z;
  HALF_BIT_FLIP(tmp, random);
  res.z = tmp;
  res.z *= b;
  HALF_BIT_FLIP(res.z, random);
  
  
  
    if(_cadna_half_overflow_tag){
    if (std::isinf(res.x) or std::isinf(res.y) or std::isinf(res.z)){
    instability(&_cadna_half_overflow_count);
    }
}



  return res ;


}
inline half_st operator*(const half_st& a, const unsigned long& b){
  half_st res;
  unsigned int random;
  half tmp;

  res.accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = a.x;
  HALF_BIT_FLIP(tmp, random);
  res.x = tmp;
  res.x *= b;
  HALF_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp = a.y;
  HALF_BIT_FLIP(tmp, random);
  res.y = tmp;
  res.y *= b;
  HALF_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp = a.z;
  HALF_BIT_FLIP(tmp, random);
  res.z = tmp;
  res.z *= b;
  HALF_BIT_FLIP(res.z, random);
  
  
  
    if(_cadna_half_overflow_tag){
    if (std::isinf(res.x) or std::isinf(res.y) or std::isinf(res.z)){
    instability(&_cadna_half_overflow_count);
    }
}



  return res ;


}
inline half_st operator*(const half_st& a, const long& b){
  half_st res;
  unsigned int random;
  half tmp;

  res.accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = a.x;
  HALF_BIT_FLIP(tmp, random);
  res.x = tmp;
  res.x *= b;
  HALF_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp = a.y;
  HALF_BIT_FLIP(tmp, random);
  res.y = tmp;
  res.y *= b;
  HALF_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp = a.z;
  HALF_BIT_FLIP(tmp, random);
  res.z = tmp;
  res.z *= b;
  HALF_BIT_FLIP(res.z, random);
  
  
  
    if(_cadna_half_overflow_tag){
    if (std::isinf(res.x) or std::isinf(res.y) or std::isinf(res.z)){
    instability(&_cadna_half_overflow_count);
    }
}



  return res ;


}
inline half_st operator*(const half_st& a, const unsigned int & b){
  half_st res;
  unsigned int random;
  half tmp;

  res.accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = a.x;
  HALF_BIT_FLIP(tmp, random);
  res.x = tmp;
  res.x *= b;
  HALF_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp = a.y;
  HALF_BIT_FLIP(tmp, random);
  res.y = tmp;
  res.y *= b;
  HALF_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp = a.z;
  HALF_BIT_FLIP(tmp, random);
  res.z = tmp;
  res.z *= b;
  HALF_BIT_FLIP(res.z, random);
  
  
  
    if(_cadna_half_overflow_tag){
    if (std::isinf(res.x) or std::isinf(res.y) or std::isinf(res.z)){
    instability(&_cadna_half_overflow_count);
    }
}



  return res ;


}
inline half_st operator*(const half_st& a, const int & b){
  half_st res;
  unsigned int random;
  half tmp;

  res.accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = a.x;
  HALF_BIT_FLIP(tmp, random);
  res.x = tmp;
  res.x *= b;
  HALF_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp = a.y;
  HALF_BIT_FLIP(tmp, random);
  res.y = tmp;
  res.y *= b;
  HALF_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp = a.z;
  HALF_BIT_FLIP(tmp, random);
  res.z = tmp;
  res.z *= b;
  HALF_BIT_FLIP(res.z, random);
  
  
  
    if(_cadna_half_overflow_tag){
    if (std::isinf(res.x) or std::isinf(res.y) or std::isinf(res.z)){
    instability(&_cadna_half_overflow_count);
    }
}



  return res ;


}
inline half_st operator*(const half_st& a, const unsigned short& b){
  half_st res;
  unsigned int random;
  half tmp;

  res.accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = a.x;
  HALF_BIT_FLIP(tmp, random);
  res.x = tmp;
  res.x *= b;
  HALF_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp = a.y;
  HALF_BIT_FLIP(tmp, random);
  res.y = tmp;
  res.y *= b;
  HALF_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp = a.z;
  HALF_BIT_FLIP(tmp, random);
  res.z = tmp;
  res.z *= b;
  HALF_BIT_FLIP(res.z, random);
  
  
  
    if(_cadna_half_overflow_tag){
    if (std::isinf(res.x) or std::isinf(res.y) or std::isinf(res.z)){
    instability(&_cadna_half_overflow_count);
    }
}



  return res ;


}
inline half_st operator*(const half_st& a, const short& b){
  half_st res;
  unsigned int random;
  half tmp;

  res.accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = a.x;
  HALF_BIT_FLIP(tmp, random);
  res.x = tmp;
  res.x *= b;
  HALF_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp = a.y;
  HALF_BIT_FLIP(tmp, random);
  res.y = tmp;
  res.y *= b;
  HALF_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp = a.z;
  HALF_BIT_FLIP(tmp, random);
  res.z = tmp;
  res.z *= b;
  HALF_BIT_FLIP(res.z, random);
  
  
  
    if(_cadna_half_overflow_tag){
    if (std::isinf(res.x) or std::isinf(res.y) or std::isinf(res.z)){
    instability(&_cadna_half_overflow_count);
    }
}



  return res ;


}
inline half_st operator*(const half_st& a, const unsigned char& b){
  half_st res;
  unsigned int random;
  half tmp;

  res.accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = a.x;
  HALF_BIT_FLIP(tmp, random);
  res.x = tmp;
  res.x *= b;
  HALF_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp = a.y;
  HALF_BIT_FLIP(tmp, random);
  res.y = tmp;
  res.y *= b;
  HALF_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp = a.z;
  HALF_BIT_FLIP(tmp, random);
  res.z = tmp;
  res.z *= b;
  HALF_BIT_FLIP(res.z, random);
  
  
  
    if(_cadna_half_overflow_tag){
    if (std::isinf(res.x) or std::isinf(res.y) or std::isinf(res.z)){
    instability(&_cadna_half_overflow_count);
    }
}



  return res ;


}
inline half_st operator*(const half_st& a, const char& b){
  half_st res;
  unsigned int random;
  half tmp;

  res.accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = a.x;
  HALF_BIT_FLIP(tmp, random);
  res.x = tmp;
  res.x *= b;
  HALF_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp = a.y;
  HALF_BIT_FLIP(tmp, random);
  res.y = tmp;
  res.y *= b;
  HALF_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp = a.z;
  HALF_BIT_FLIP(tmp, random);
  res.z = tmp;
  res.z *= b;
  HALF_BIT_FLIP(res.z, random);
  
  
  
    if(_cadna_half_overflow_tag){
    if (std::isinf(res.x) or std::isinf(res.y) or std::isinf(res.z)){
    instability(&_cadna_half_overflow_count);
    }
}



  return res ;


}
inline half_st operator*(const half_st& a, const half& b){
  half_st res;
  unsigned int random;
  half tmp;

  res.accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = a.x;
  HALF_BIT_FLIP(tmp, random);
  res.x = tmp;
  res.x *= b;
  HALF_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp = a.y;
  HALF_BIT_FLIP(tmp, random);
  res.y = tmp;
  res.y *= b;
  HALF_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp = a.z;
  HALF_BIT_FLIP(tmp, random);
  res.z = tmp;
  res.z *= b;
  HALF_BIT_FLIP(res.z, random);
  
  
  
    if(_cadna_half_overflow_tag){
    if (std::isinf(res.x) or std::isinf(res.y) or std::isinf(res.z)){
    instability(&_cadna_half_overflow_count);
    }
}



  return res ;


}
#endif // CADNA_HALF_EMULATION


#ifdef CADNA_HALF_NATIVE
inline half_st operator*(const half_st& a, const double& b){
  half_st res;
  unsigned int random;
  half tmp;

  res.accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = a.x;
  HALF_BIT_FLIP(tmp, random);
  res.x = tmp;
  res.x *= b;
  HALF_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp = a.y;
  HALF_BIT_FLIP(tmp, random);
  res.y = tmp;
  res.y *= b;
  HALF_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp = a.z;
  HALF_BIT_FLIP(tmp, random);
  res.z = tmp;
  res.z *= b;
  HALF_BIT_FLIP(res.z, random);
  
  
  
    if(_cadna_half_overflow_tag){
    if (std::isinf(res.x) or std::isinf(res.y) or std::isinf(res.z)){
    instability(&_cadna_half_overflow_count);
    }
}



  return res ;


}
inline half_st operator*(const half_st& a, const float& b){
  half_st res;
  unsigned int random;
  half tmp;

  res.accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = a.x;
  HALF_BIT_FLIP(tmp, random);
  res.x = tmp;
  res.x *= b;
  HALF_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp = a.y;
  HALF_BIT_FLIP(tmp, random);
  res.y = tmp;
  res.y *= b;
  HALF_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp = a.z;
  HALF_BIT_FLIP(tmp, random);
  res.z = tmp;
  res.z *= b;
  HALF_BIT_FLIP(res.z, random);
  
  
  
    if(_cadna_half_overflow_tag){
    if (std::isinf(res.x) or std::isinf(res.y) or std::isinf(res.z)){
    instability(&_cadna_half_overflow_count);
    }
}



  return res ;


}
inline half_st operator*(const half_st& a, const unsigned long long& b){
  half_st res;
  unsigned int random;
  half tmp;

  res.accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = a.x;
  HALF_BIT_FLIP(tmp, random);
  res.x = tmp;
  res.x *= b;
  HALF_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp = a.y;
  HALF_BIT_FLIP(tmp, random);
  res.y = tmp;
  res.y *= b;
  HALF_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp = a.z;
  HALF_BIT_FLIP(tmp, random);
  res.z = tmp;
  res.z *= b;
  HALF_BIT_FLIP(res.z, random);
  
  
  
    if(_cadna_half_overflow_tag){
    if (std::isinf(res.x) or std::isinf(res.y) or std::isinf(res.z)){
    instability(&_cadna_half_overflow_count);
    }
}



  return res ;


}
inline half_st operator*(const half_st& a, const long long& b){
  half_st res;
  unsigned int random;
  half tmp;

  res.accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = a.x;
  HALF_BIT_FLIP(tmp, random);
  res.x = tmp;
  res.x *= b;
  HALF_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp = a.y;
  HALF_BIT_FLIP(tmp, random);
  res.y = tmp;
  res.y *= b;
  HALF_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp = a.z;
  HALF_BIT_FLIP(tmp, random);
  res.z = tmp;
  res.z *= b;
  HALF_BIT_FLIP(res.z, random);
  
  
  
    if(_cadna_half_overflow_tag){
    if (std::isinf(res.x) or std::isinf(res.y) or std::isinf(res.z)){
    instability(&_cadna_half_overflow_count);
    }
}



  return res ;


}
inline half_st operator*(const half_st& a, const unsigned long& b){
  half_st res;
  unsigned int random;
  half tmp;

  res.accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = a.x;
  HALF_BIT_FLIP(tmp, random);
  res.x = tmp;
  res.x *= b;
  HALF_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp = a.y;
  HALF_BIT_FLIP(tmp, random);
  res.y = tmp;
  res.y *= b;
  HALF_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp = a.z;
  HALF_BIT_FLIP(tmp, random);
  res.z = tmp;
  res.z *= b;
  HALF_BIT_FLIP(res.z, random);
  
  
  
    if(_cadna_half_overflow_tag){
    if (std::isinf(res.x) or std::isinf(res.y) or std::isinf(res.z)){
    instability(&_cadna_half_overflow_count);
    }
}



  return res ;


}
inline half_st operator*(const half_st& a, const long& b){
  half_st res;
  unsigned int random;
  half tmp;

  res.accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = a.x;
  HALF_BIT_FLIP(tmp, random);
  res.x = tmp;
  res.x *= b;
  HALF_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp = a.y;
  HALF_BIT_FLIP(tmp, random);
  res.y = tmp;
  res.y *= b;
  HALF_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp = a.z;
  HALF_BIT_FLIP(tmp, random);
  res.z = tmp;
  res.z *= b;
  HALF_BIT_FLIP(res.z, random);
  
  
  
    if(_cadna_half_overflow_tag){
    if (std::isinf(res.x) or std::isinf(res.y) or std::isinf(res.z)){
    instability(&_cadna_half_overflow_count);
    }
}



  return res ;


}
inline half_st operator*(const half_st& a, const unsigned int & b){
  half_st res;
  unsigned int random;
  half tmp;

  res.accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = a.x;
  HALF_BIT_FLIP(tmp, random);
  res.x = tmp;
  res.x *= b;
  HALF_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp = a.y;
  HALF_BIT_FLIP(tmp, random);
  res.y = tmp;
  res.y *= b;
  HALF_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp = a.z;
  HALF_BIT_FLIP(tmp, random);
  res.z = tmp;
  res.z *= b;
  HALF_BIT_FLIP(res.z, random);
  
  
  
    if(_cadna_half_overflow_tag){
    if (std::isinf(res.x) or std::isinf(res.y) or std::isinf(res.z)){
    instability(&_cadna_half_overflow_count);
    }
}



  return res ;


}
inline half_st operator*(const half_st& a, const int & b){
  half_st res;
  unsigned int random;
  half tmp;

  res.accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = a.x;
  HALF_BIT_FLIP(tmp, random);
  res.x = tmp;
  res.x *= b;
  HALF_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp = a.y;
  HALF_BIT_FLIP(tmp, random);
  res.y = tmp;
  res.y *= b;
  HALF_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp = a.z;
  HALF_BIT_FLIP(tmp, random);
  res.z = tmp;
  res.z *= b;
  HALF_BIT_FLIP(res.z, random);
  
  
  
    if(_cadna_half_overflow_tag){
    if (std::isinf(res.x) or std::isinf(res.y) or std::isinf(res.z)){
    instability(&_cadna_half_overflow_count);
    }
}



  return res ;


}
inline half_st operator*(const half_st& a, const unsigned short& b){
  half_st res;
  unsigned int random;
  half tmp;

  res.accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = a.x;
  HALF_BIT_FLIP(tmp, random);
  res.x = tmp;
  res.x *= b;
  HALF_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp = a.y;
  HALF_BIT_FLIP(tmp, random);
  res.y = tmp;
  res.y *= b;
  HALF_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp = a.z;
  HALF_BIT_FLIP(tmp, random);
  res.z = tmp;
  res.z *= b;
  HALF_BIT_FLIP(res.z, random);
  
  
  
    if(_cadna_half_overflow_tag){
    if (std::isinf(res.x) or std::isinf(res.y) or std::isinf(res.z)){
    instability(&_cadna_half_overflow_count);
    }
}



  return res ;


}
inline half_st operator*(const half_st& a, const short& b){
  half_st res;
  unsigned int random;
  half tmp;

  res.accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = a.x;
  HALF_BIT_FLIP(tmp, random);
  res.x = tmp;
  res.x *= b;
  HALF_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp = a.y;
  HALF_BIT_FLIP(tmp, random);
  res.y = tmp;
  res.y *= b;
  HALF_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp = a.z;
  HALF_BIT_FLIP(tmp, random);
  res.z = tmp;
  res.z *= b;
  HALF_BIT_FLIP(res.z, random);
  
  
  
    if(_cadna_half_overflow_tag){
    if (std::isinf(res.x) or std::isinf(res.y) or std::isinf(res.z)){
    instability(&_cadna_half_overflow_count);
    }
}



  return res ;


}
inline half_st operator*(const half_st& a, const unsigned char& b){
  half_st res;
  unsigned int random;
  half tmp;

  res.accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = a.x;
  HALF_BIT_FLIP(tmp, random);
  res.x = tmp;
  res.x *= b;
  HALF_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp = a.y;
  HALF_BIT_FLIP(tmp, random);
  res.y = tmp;
  res.y *= b;
  HALF_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp = a.z;
  HALF_BIT_FLIP(tmp, random);
  res.z = tmp;
  res.z *= b;
  HALF_BIT_FLIP(res.z, random);
  
  
  
    if(_cadna_half_overflow_tag){
    if (std::isinf(res.x) or std::isinf(res.y) or std::isinf(res.z)){
    instability(&_cadna_half_overflow_count);
    }
}



  return res ;


}
inline half_st operator*(const half_st& a, const char& b){
  half_st res;
  unsigned int random;
  half tmp;

  res.accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = a.x;
  HALF_BIT_FLIP(tmp, random);
  res.x = tmp;
  res.x *= b;
  HALF_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp = a.y;
  HALF_BIT_FLIP(tmp, random);
  res.y = tmp;
  res.y *= b;
  HALF_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp = a.z;
  HALF_BIT_FLIP(tmp, random);
  res.z = tmp;
  res.z *= b;
  HALF_BIT_FLIP(res.z, random);
  
  
  
    if(_cadna_half_overflow_tag){
    if (std::isinf(res.x) or std::isinf(res.y) or std::isinf(res.z)){
    instability(&_cadna_half_overflow_count);
    }
}



  return res ;


}
inline half_st operator*(const half_st& a, const half& b){
  half_st res;
  unsigned int random;
  half tmp;

  res.accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = a.x;
  HALF_BIT_FLIP(tmp, random);
  res.x = tmp;
  res.x *= b;
  HALF_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp = a.y;
  HALF_BIT_FLIP(tmp, random);
  res.y = tmp;
  res.y *= b;
  HALF_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp = a.z;
  HALF_BIT_FLIP(tmp, random);
  res.z = tmp;
  res.z *= b;
  HALF_BIT_FLIP(res.z, random);
  
  
  
    if(_cadna_half_overflow_tag){
    if (std::isinf(res.x) or std::isinf(res.y) or std::isinf(res.z)){
    instability(&_cadna_half_overflow_count);
    }
}



  return res ;


}
#endif
/////////////////////////////////////////////////////////////////////////







inline double_st operator*(const double_st& a, const double_st& b){
  double_st res;
  unsigned int random;
  double tmp;

  res.accuracy=DIGIT_NOT_COMPUTED;

  if(_cadna_mul_tag) {
    if (b.accuracy==DIGIT_NOT_COMPUTED) b.approx_digit();
    if (b.accuracy==0){
      if (a.accuracy==DIGIT_NOT_COMPUTED) a.approx_digit();
      if (a.accuracy==0)
	instability(&_cadna_mul_count);
    }
  }

  random = RANDOM;
  tmp = a.x;
  DOUBLE_BIT_FLIP(tmp, random);
  res.x = tmp;
  tmp = b.x;
  res.x *= tmp;
  DOUBLE_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp = a.y;
  DOUBLE_BIT_FLIP(tmp, random);
  res.y = tmp;
  tmp = b.y;
  res.y *= tmp;
  DOUBLE_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp = a.z;
  DOUBLE_BIT_FLIP(tmp, random);
  res.z = tmp;
  tmp = b.z;
  res.z *= tmp;
  DOUBLE_BIT_FLIP(res.z, random);

  return res ;
}
inline double_st operator*(const double_st& a, const float_st& b){
  double_st res;
  unsigned int random;
  double tmp;

  res.accuracy=DIGIT_NOT_COMPUTED;

  if(_cadna_mul_tag) {
    if (b.accuracy==DIGIT_NOT_COMPUTED) b.approx_digit();
    if (b.accuracy==0){
      if (a.accuracy==DIGIT_NOT_COMPUTED) a.approx_digit();
      if (a.accuracy==0)
	instability(&_cadna_mul_count);
    }
  }

  random = RANDOM;
  tmp = a.x;
  DOUBLE_BIT_FLIP(tmp, random);
  res.x = tmp;
  tmp = b.x;
  res.x *= tmp;
  DOUBLE_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp = a.y;
  DOUBLE_BIT_FLIP(tmp, random);
  res.y = tmp;
  tmp = b.y;
  res.y *= tmp;
  DOUBLE_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp = a.z;
  DOUBLE_BIT_FLIP(tmp, random);
  res.z = tmp;
  tmp = b.z;
  res.z *= tmp;
  DOUBLE_BIT_FLIP(res.z, random);

  return res ;
}
inline double_st operator*(const float_st& a, const double_st& b){
  double_st res;
  unsigned int random;
  double tmp;

  res.accuracy=DIGIT_NOT_COMPUTED;

  if(_cadna_mul_tag) {
    if (b.accuracy==DIGIT_NOT_COMPUTED) b.approx_digit();
    if (b.accuracy==0){
      if (a.accuracy==DIGIT_NOT_COMPUTED) a.approx_digit();
      if (a.accuracy==0)
	instability(&_cadna_mul_count);
    }
  }

  random = RANDOM;
  tmp = a.x;
  DOUBLE_BIT_FLIP(tmp, random);
  res.x = tmp;
  tmp = b.x;
  res.x *= tmp;
  DOUBLE_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp = a.y;
  DOUBLE_BIT_FLIP(tmp, random);
  res.y = tmp;
  tmp = b.y;
  res.y *= tmp;
  DOUBLE_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp = a.z;
  DOUBLE_BIT_FLIP(tmp, random);
  res.z = tmp;
  tmp = b.z;
  res.z *= tmp;
  DOUBLE_BIT_FLIP(res.z, random);

  return res ;
}
inline float_st operator*(const float_st& a, const float_st& b){
  float_st res;
  unsigned int random;
  float tmp;

  res.accuracy=DIGIT_NOT_COMPUTED;

  if(_cadna_mul_tag) {
    if (b.accuracy==DIGIT_NOT_COMPUTED) b.approx_digit();
    if (b.accuracy==0){
      if (a.accuracy==DIGIT_NOT_COMPUTED) a.approx_digit();
      if (a.accuracy==0)
	instability(&_cadna_mul_count);
    }
  }

  random = RANDOM;
  tmp = a.x;
  FLOAT_BIT_FLIP(tmp, random);
  res.x = tmp;
  tmp = b.x;
  res.x *= tmp;
  FLOAT_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp = a.y;
  FLOAT_BIT_FLIP(tmp, random);
  res.y = tmp;
  tmp = b.y;
  res.y *= tmp;
  FLOAT_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp = a.z;
  FLOAT_BIT_FLIP(tmp, random);
  res.z = tmp;
  tmp = b.z;
  res.z *= tmp;
  FLOAT_BIT_FLIP(res.z, random);

  return res ;
}
#ifdef CADNA_HALF_EMULATION
inline half_st operator*(const half_st& a, const half_st& b){
  half_st res;
  unsigned int random;
  half tmp;

  res.accuracy=DIGIT_NOT_COMPUTED;

  if(_cadna_mul_tag) {
    if (b.accuracy==DIGIT_NOT_COMPUTED) b.approx_digit();
    if (b.accuracy==0){
      if (a.accuracy==DIGIT_NOT_COMPUTED) a.approx_digit();
      if (a.accuracy==0)
	instability(&_cadna_mul_count);
    }
  }

  random = RANDOM;
  tmp = a.x;
  HALF_BIT_FLIP(tmp, random);
  res.x = tmp;
  tmp = b.x;
  res.x *= tmp;
  HALF_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp = a.y;
  HALF_BIT_FLIP(tmp, random);
  res.y = tmp;
  tmp = b.y;
  res.y *= tmp;
  HALF_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp = a.z;
  HALF_BIT_FLIP(tmp, random);
  res.z = tmp;
  tmp = b.z;
  res.z *= tmp;
  HALF_BIT_FLIP(res.z, random);
  
  
  
    if(_cadna_half_overflow_tag){
    if (std::isinf(res.x) or std::isinf(res.y) or std::isinf(res.z)){
    instability(&_cadna_half_overflow_count);
    }
}

  if(_cadna_half_underflow_tag){
  /*
  double val = res.x + res.y + res.z;
  double val_a = a.x+a.y+a.z;
  double val_b = b.x+b.y+b.z;
  if ( (val_a!=0) &&  (val_b!=0) && (val==0)){
  */
  //result is zero but a and b are not exact zeroes
  if ( ( (res.x==0) || (res.y==0) || (res.z==0) ) && (a.x!=0) && (b.x!=0) ){
    instability(&_cadna_half_underflow_count);
    }
}
  return res ;

}
inline float_st operator*(const half_st& a, const float_st& b){
  float_st res;
  unsigned int random;
  float tmp;

  res.accuracy=DIGIT_NOT_COMPUTED;

  if(_cadna_mul_tag) {
    if (b.accuracy==DIGIT_NOT_COMPUTED) b.approx_digit();
    if (b.accuracy==0){
      if (a.accuracy==DIGIT_NOT_COMPUTED) a.approx_digit();
      if (a.accuracy==0)
	instability(&_cadna_mul_count);
    }
  }

  random = RANDOM;
  tmp = a.x;
  FLOAT_BIT_FLIP(tmp, random);
  res.x = tmp;
  tmp = b.x;
  res.x *= tmp;
  FLOAT_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp = a.y;
  FLOAT_BIT_FLIP(tmp, random);
  res.y = tmp;
  tmp = b.y;
  res.y *= tmp;
  FLOAT_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp = a.z;
  FLOAT_BIT_FLIP(tmp, random);
  res.z = tmp;
  tmp = b.z;
  res.z *= tmp;
  FLOAT_BIT_FLIP(res.z, random);
  
  
  
    if(_cadna_half_overflow_tag){
    if (std::isinf(res.x) or std::isinf(res.y) or std::isinf(res.z)){
    instability(&_cadna_half_overflow_count);
    }
}

  if(_cadna_half_underflow_tag){
  /*
  double val = res.x + res.y + res.z;
  double val_a = a.x+a.y+a.z;
  double val_b = b.x+b.y+b.z;
  if ( (val_a!=0) &&  (val_b!=0) && (val==0)){
  */
  //result is zero but a and b are not exact zeroes
  if ( ( (res.x==0) || (res.y==0) || (res.z==0) ) && (a.x!=0) && (b.x!=0) ){
    instability(&_cadna_half_underflow_count);
    }
}
  return res ;

}
inline float_st operator*(const float_st& a, const half_st& b){
  float_st res;
  unsigned int random;
  float tmp;

  res.accuracy=DIGIT_NOT_COMPUTED;

  if(_cadna_mul_tag) {
    if (b.accuracy==DIGIT_NOT_COMPUTED) b.approx_digit();
    if (b.accuracy==0){
      if (a.accuracy==DIGIT_NOT_COMPUTED) a.approx_digit();
      if (a.accuracy==0)
	instability(&_cadna_mul_count);
    }
  }

  random = RANDOM;
  tmp = a.x;
  FLOAT_BIT_FLIP(tmp, random);
  res.x = tmp;
  tmp = b.x;
  res.x *= tmp;
  FLOAT_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp = a.y;
  FLOAT_BIT_FLIP(tmp, random);
  res.y = tmp;
  tmp = b.y;
  res.y *= tmp;
  FLOAT_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp = a.z;
  FLOAT_BIT_FLIP(tmp, random);
  res.z = tmp;
  tmp = b.z;
  res.z *= tmp;
  FLOAT_BIT_FLIP(res.z, random);
  
  
  
    if(_cadna_half_overflow_tag){
    if (std::isinf(res.x) or std::isinf(res.y) or std::isinf(res.z)){
    instability(&_cadna_half_overflow_count);
    }
}

  if(_cadna_half_underflow_tag){
  /*
  double val = res.x + res.y + res.z;
  double val_a = a.x+a.y+a.z;
  double val_b = b.x+b.y+b.z;
  if ( (val_a!=0) &&  (val_b!=0) && (val==0)){
  */
  //result is zero but a and b are not exact zeroes
  if ( ( (res.x==0) || (res.y==0) || (res.z==0) ) && (a.x!=0) && (b.x!=0) ){
    instability(&_cadna_half_underflow_count);
    }
}
  return res ;

}
inline double_st operator*(const half_st& a, const double_st& b){
  double_st res;
  unsigned int random;
  double tmp;

  res.accuracy=DIGIT_NOT_COMPUTED;

  if(_cadna_mul_tag) {
    if (b.accuracy==DIGIT_NOT_COMPUTED) b.approx_digit();
    if (b.accuracy==0){
      if (a.accuracy==DIGIT_NOT_COMPUTED) a.approx_digit();
      if (a.accuracy==0)
	instability(&_cadna_mul_count);
    }
  }

  random = RANDOM;
  tmp = a.x;
  DOUBLE_BIT_FLIP(tmp, random);
  res.x = tmp;
  tmp = b.x;
  res.x *= tmp;
  DOUBLE_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp = a.y;
  DOUBLE_BIT_FLIP(tmp, random);
  res.y = tmp;
  tmp = b.y;
  res.y *= tmp;
  DOUBLE_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp = a.z;
  DOUBLE_BIT_FLIP(tmp, random);
  res.z = tmp;
  tmp = b.z;
  res.z *= tmp;
  DOUBLE_BIT_FLIP(res.z, random);
  
  
  
    if(_cadna_half_overflow_tag){
    if (std::isinf(res.x) or std::isinf(res.y) or std::isinf(res.z)){
    instability(&_cadna_half_overflow_count);
    }
}

  if(_cadna_half_underflow_tag){
  /*
  double val = res.x + res.y + res.z;
  double val_a = a.x+a.y+a.z;
  double val_b = b.x+b.y+b.z;
  if ( (val_a!=0) &&  (val_b!=0) && (val==0)){
  */
  //result is zero but a and b are not exact zeroes
  if ( ( (res.x==0) || (res.y==0) || (res.z==0) ) && (a.x!=0) && (b.x!=0) ){
    instability(&_cadna_half_underflow_count);
    }
}
  return res ;

}
inline double_st operator*(const double_st& a, const half_st& b){
  double_st res;
  unsigned int random;
  double tmp;

  res.accuracy=DIGIT_NOT_COMPUTED;

  if(_cadna_mul_tag) {
    if (b.accuracy==DIGIT_NOT_COMPUTED) b.approx_digit();
    if (b.accuracy==0){
      if (a.accuracy==DIGIT_NOT_COMPUTED) a.approx_digit();
      if (a.accuracy==0)
	instability(&_cadna_mul_count);
    }
  }

  random = RANDOM;
  tmp = a.x;
  DOUBLE_BIT_FLIP(tmp, random);
  res.x = tmp;
  tmp = b.x;
  res.x *= tmp;
  DOUBLE_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp = a.y;
  DOUBLE_BIT_FLIP(tmp, random);
  res.y = tmp;
  tmp = b.y;
  res.y *= tmp;
  DOUBLE_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp = a.z;
  DOUBLE_BIT_FLIP(tmp, random);
  res.z = tmp;
  tmp = b.z;
  res.z *= tmp;
  DOUBLE_BIT_FLIP(res.z, random);
  
  
  
    if(_cadna_half_overflow_tag){
    if (std::isinf(res.x) or std::isinf(res.y) or std::isinf(res.z)){
    instability(&_cadna_half_overflow_count);
    }
}

  if(_cadna_half_underflow_tag){
  /*
  double val = res.x + res.y + res.z;
  double val_a = a.x+a.y+a.z;
  double val_b = b.x+b.y+b.z;
  if ( (val_a!=0) &&  (val_b!=0) && (val==0)){
  */
  //result is zero but a and b are not exact zeroes
  if ( ( (res.x==0) || (res.y==0) || (res.z==0) ) && (a.x!=0) && (b.x!=0) ){
    instability(&_cadna_half_underflow_count);
    }
}
  return res ;

}
#endif

#ifdef CADNA_HALF_NATIVE
inline half_st operator*(const half_st& a, const half_st& b){
  half_st res;
  unsigned int random;
  half tmp;

  res.accuracy=DIGIT_NOT_COMPUTED;

  if(_cadna_mul_tag) {
    if (b.accuracy==DIGIT_NOT_COMPUTED) b.approx_digit();
    if (b.accuracy==0){
      if (a.accuracy==DIGIT_NOT_COMPUTED) a.approx_digit();
      if (a.accuracy==0)
	instability(&_cadna_mul_count);
    }
  }

  random = RANDOM;
  tmp = a.x;
  HALF_BIT_FLIP(tmp, random);
  res.x = tmp;
  tmp = b.x;
  res.x *= tmp;
  HALF_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp = a.y;
  HALF_BIT_FLIP(tmp, random);
  res.y = tmp;
  tmp = b.y;
  res.y *= tmp;
  HALF_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp = a.z;
  HALF_BIT_FLIP(tmp, random);
  res.z = tmp;
  tmp = b.z;
  res.z *= tmp;
  HALF_BIT_FLIP(res.z, random);
  
  
  
    if(_cadna_half_overflow_tag){
    if (std::isinf(res.x) or std::isinf(res.y) or std::isinf(res.z)){
    instability(&_cadna_half_overflow_count);
    }
}

  if(_cadna_half_underflow_tag){
  /*
  double val = res.x + res.y + res.z;
  double val_a = a.x+a.y+a.z;
  double val_b = b.x+b.y+b.z;
  if ( (val_a!=0) &&  (val_b!=0) && (val==0)){
  */
  //result is zero but a and b are not exact zeroes
  if ( ( (res.x==0) || (res.y==0) || (res.z==0) ) && (a.x!=0) && (b.x!=0) ){
    instability(&_cadna_half_underflow_count);
    }
}
  return res ;

}
inline float_st operator*(const half_st& a, const float_st& b){
  float_st res;
  unsigned int random;
  float tmp;

  res.accuracy=DIGIT_NOT_COMPUTED;

  if(_cadna_mul_tag) {
    if (b.accuracy==DIGIT_NOT_COMPUTED) b.approx_digit();
    if (b.accuracy==0){
      if (a.accuracy==DIGIT_NOT_COMPUTED) a.approx_digit();
      if (a.accuracy==0)
	instability(&_cadna_mul_count);
    }
  }

  random = RANDOM;
  tmp = a.x;
  FLOAT_BIT_FLIP(tmp, random);
  res.x = tmp;
  tmp = b.x;
  res.x *= tmp;
  FLOAT_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp = a.y;
  FLOAT_BIT_FLIP(tmp, random);
  res.y = tmp;
  tmp = b.y;
  res.y *= tmp;
  FLOAT_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp = a.z;
  FLOAT_BIT_FLIP(tmp, random);
  res.z = tmp;
  tmp = b.z;
  res.z *= tmp;
  FLOAT_BIT_FLIP(res.z, random);
  
  
  
    if(_cadna_half_overflow_tag){
    if (std::isinf(res.x) or std::isinf(res.y) or std::isinf(res.z)){
    instability(&_cadna_half_overflow_count);
    }
}

  if(_cadna_half_underflow_tag){
  /*
  double val = res.x + res.y + res.z;
  double val_a = a.x+a.y+a.z;
  double val_b = b.x+b.y+b.z;
  if ( (val_a!=0) &&  (val_b!=0) && (val==0)){
  */
  //result is zero but a and b are not exact zeroes
  if ( ( (res.x==0) || (res.y==0) || (res.z==0) ) && (a.x!=0) && (b.x!=0) ){
    instability(&_cadna_half_underflow_count);
    }
}
  return res ;

}
inline float_st operator*(const float_st& a, const half_st& b){
  float_st res;
  unsigned int random;
  float tmp;

  res.accuracy=DIGIT_NOT_COMPUTED;

  if(_cadna_mul_tag) {
    if (b.accuracy==DIGIT_NOT_COMPUTED) b.approx_digit();
    if (b.accuracy==0){
      if (a.accuracy==DIGIT_NOT_COMPUTED) a.approx_digit();
      if (a.accuracy==0)
	instability(&_cadna_mul_count);
    }
  }

  random = RANDOM;
  tmp = a.x;
  FLOAT_BIT_FLIP(tmp, random);
  res.x = tmp;
  tmp = b.x;
  res.x *= tmp;
  FLOAT_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp = a.y;
  FLOAT_BIT_FLIP(tmp, random);
  res.y = tmp;
  tmp = b.y;
  res.y *= tmp;
  FLOAT_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp = a.z;
  FLOAT_BIT_FLIP(tmp, random);
  res.z = tmp;
  tmp = b.z;
  res.z *= tmp;
  FLOAT_BIT_FLIP(res.z, random);
  
  
  
    if(_cadna_half_overflow_tag){
    if (std::isinf(res.x) or std::isinf(res.y) or std::isinf(res.z)){
    instability(&_cadna_half_overflow_count);
    }
}

  if(_cadna_half_underflow_tag){
  /*
  double val = res.x + res.y + res.z;
  double val_a = a.x+a.y+a.z;
  double val_b = b.x+b.y+b.z;
  if ( (val_a!=0) &&  (val_b!=0) && (val==0)){
  */
  //result is zero but a and b are not exact zeroes
  if ( ( (res.x==0) || (res.y==0) || (res.z==0) ) && (a.x!=0) && (b.x!=0) ){
    instability(&_cadna_half_underflow_count);
    }
}
  return res ;

}
inline double_st operator*(const half_st& a, const double_st& b){
  double_st res;
  unsigned int random;
  double tmp;

  res.accuracy=DIGIT_NOT_COMPUTED;

  if(_cadna_mul_tag) {
    if (b.accuracy==DIGIT_NOT_COMPUTED) b.approx_digit();
    if (b.accuracy==0){
      if (a.accuracy==DIGIT_NOT_COMPUTED) a.approx_digit();
      if (a.accuracy==0)
	instability(&_cadna_mul_count);
    }
  }

  random = RANDOM;
  tmp = a.x;
  DOUBLE_BIT_FLIP(tmp, random);
  res.x = tmp;
  tmp = b.x;
  res.x *= tmp;
  DOUBLE_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp = a.y;
  DOUBLE_BIT_FLIP(tmp, random);
  res.y = tmp;
  tmp = b.y;
  res.y *= tmp;
  DOUBLE_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp = a.z;
  DOUBLE_BIT_FLIP(tmp, random);
  res.z = tmp;
  tmp = b.z;
  res.z *= tmp;
  DOUBLE_BIT_FLIP(res.z, random);
  
  
  
    if(_cadna_half_overflow_tag){
    if (std::isinf(res.x) or std::isinf(res.y) or std::isinf(res.z)){
    instability(&_cadna_half_overflow_count);
    }
}

  if(_cadna_half_underflow_tag){
  /*
  double val = res.x + res.y + res.z;
  double val_a = a.x+a.y+a.z;
  double val_b = b.x+b.y+b.z;
  if ( (val_a!=0) &&  (val_b!=0) && (val==0)){
  */
  //result is zero but a and b are not exact zeroes
  if ( ( (res.x==0) || (res.y==0) || (res.z==0) ) && (a.x!=0) && (b.x!=0) ){
    instability(&_cadna_half_underflow_count);
    }
}
  return res ;

}
inline double_st operator*(const double_st& a, const half_st& b){
  double_st res;
  unsigned int random;
  double tmp;

  res.accuracy=DIGIT_NOT_COMPUTED;

  if(_cadna_mul_tag) {
    if (b.accuracy==DIGIT_NOT_COMPUTED) b.approx_digit();
    if (b.accuracy==0){
      if (a.accuracy==DIGIT_NOT_COMPUTED) a.approx_digit();
      if (a.accuracy==0)
	instability(&_cadna_mul_count);
    }
  }

  random = RANDOM;
  tmp = a.x;
  DOUBLE_BIT_FLIP(tmp, random);
  res.x = tmp;
  tmp = b.x;
  res.x *= tmp;
  DOUBLE_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp = a.y;
  DOUBLE_BIT_FLIP(tmp, random);
  res.y = tmp;
  tmp = b.y;
  res.y *= tmp;
  DOUBLE_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp = a.z;
  DOUBLE_BIT_FLIP(tmp, random);
  res.z = tmp;
  tmp = b.z;
  res.z *= tmp;
  DOUBLE_BIT_FLIP(res.z, random);
  
  
  
    if(_cadna_half_overflow_tag){
    if (std::isinf(res.x) or std::isinf(res.y) or std::isinf(res.z)){
    instability(&_cadna_half_overflow_count);
    }
}

  if(_cadna_half_underflow_tag){
  /*
  double val = res.x + res.y + res.z;
  double val_a = a.x+a.y+a.z;
  double val_b = b.x+b.y+b.z;
  if ( (val_a!=0) &&  (val_b!=0) && (val==0)){
  */
  //result is zero but a and b are not exact zeroes
  if ( ( (res.x==0) || (res.y==0) || (res.z==0) ) && (a.x!=0) && (b.x!=0) ){
    instability(&_cadna_half_underflow_count);
    }
}
  return res ;

}
#endif
