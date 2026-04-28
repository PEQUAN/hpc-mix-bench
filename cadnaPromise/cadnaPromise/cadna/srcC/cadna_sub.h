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
//****m* cadna_sub/operator-
//    NAME
//      operator-
//    SYNOPSIS
//      res = a - b
//    FUNCTION
//      Define all the functions involving at least one argument
//      of stochastic type which overload the "-" operator
//      in a statement such as "a-b".
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








inline float_st float_st::operator-() const{
  float_st res;
  res.x=-x;
  res.y=-y;
  res.z=-z;
  return res;
}
inline double_st double_st::operator-() const{
  double_st res;
  res.x=-x;
  res.y=-y;
  res.z=-z;
  return res;
}
#ifdef CADNA_HALF_EMULATION
inline half_st half_st::operator-() const{
  half_st res;
  res.x=-x;
  res.y=-y;
  res.z=-z;
  return res;
}
#endif
#ifdef CADNA_HALF_NATIVE
inline half_st half_st::operator-() const{
  half_st res;
  res.x=-x;
  res.y=-y;
  res.z=-z;
  return res;
}
#endif
/////////////////////////////////////////////////////////////////////////





inline float_st float_st::operator--(){
  float_st initial=*this;

  unsigned int random;
  float tmp;

  accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = x;
  FLOAT_BIT_FLIP(tmp, random);
  x = tmp;
  tmp = 1;
  FLOAT_BIT_FLIP(tmp, random);
  x -= tmp;
  FLOAT_BIT_FLIP(x, random);

  random = RANDOM;
  tmp = y;
  FLOAT_BIT_FLIP(tmp, random);
  y = tmp;
  tmp = 1;
  FLOAT_BIT_FLIP(tmp, random);
  y -= tmp;
  FLOAT_BIT_FLIP(y, random);

  random = 1^random;
  tmp = z;
  FLOAT_BIT_FLIP(tmp, random);
  z = tmp;
  tmp = 1;
  FLOAT_BIT_FLIP(tmp, random);
  z -= tmp;
  FLOAT_BIT_FLIP(z, random);

  if (_cadna_cancel_tag){
    if (initial.accuracy==DIGIT_NOT_COMPUTED || initial.accuracy==RELIABLE_RESULT)
      initial.nb_significant_digit();
    if (initial.accuracy >=  this->nb_significant_digit()+_cadna_cancel_value)
      instability(&_cadna_cancel_count);
  }
  return *this;
};
inline double_st double_st::operator--(){
  double_st initial=*this;

  unsigned int random;
  double tmp;

  accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = x;
  DOUBLE_BIT_FLIP(tmp, random);
  x = tmp;
  tmp = 1;
  DOUBLE_BIT_FLIP(tmp, random);
  x -= tmp;
  DOUBLE_BIT_FLIP(x, random);

  random = RANDOM;
  tmp = y;
  DOUBLE_BIT_FLIP(tmp, random);
  y = tmp;
  tmp = 1;
  DOUBLE_BIT_FLIP(tmp, random);
  y -= tmp;
  DOUBLE_BIT_FLIP(y, random);

  random = 1^random;
  tmp = z;
  DOUBLE_BIT_FLIP(tmp, random);
  z = tmp;
  tmp = 1;
  DOUBLE_BIT_FLIP(tmp, random);
  z -= tmp;
  DOUBLE_BIT_FLIP(z, random);

  if (_cadna_cancel_tag){
    if (initial.accuracy==DIGIT_NOT_COMPUTED || initial.accuracy==RELIABLE_RESULT)
      initial.nb_significant_digit();
    if (initial.accuracy >=  this->nb_significant_digit()+_cadna_cancel_value)
      instability(&_cadna_cancel_count);
  }
  return *this;
};
#ifdef CADNA_HALF_EMULATION
inline half_st half_st::operator--(){
  half_st initial=*this;

  unsigned int random;
  half tmp;

  accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = x;
  HALF_BIT_FLIP(tmp, random);
  x = tmp;
  tmp = 1;
  HALF_BIT_FLIP(tmp, random);
  x -= tmp;
  HALF_BIT_FLIP(x, random);

  random = RANDOM;
  tmp = y;
  HALF_BIT_FLIP(tmp, random);
  y = tmp;
  tmp = 1;
  HALF_BIT_FLIP(tmp, random);
  y -= tmp;
  HALF_BIT_FLIP(y, random);

  random = 1^random;
  tmp = z;
  HALF_BIT_FLIP(tmp, random);
  z = tmp;
  tmp = 1;
  HALF_BIT_FLIP(tmp, random);
  z -= tmp;
  HALF_BIT_FLIP(z, random);

  if (_cadna_cancel_tag){
    if (initial.accuracy==DIGIT_NOT_COMPUTED || initial.accuracy==RELIABLE_RESULT)
      initial.nb_significant_digit();
    if (initial.accuracy >=  this->nb_significant_digit()+_cadna_cancel_value)
      instability(&_cadna_cancel_count);
  }
  return *this;
};
#endif

#ifdef CADNA_HALF_NATIVE
inline half_st half_st::operator--(){
  half_st initial=*this;

  unsigned int random;
  half tmp;

  accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = x;
  HALF_BIT_FLIP(tmp, random);
  x = tmp;
  tmp = 1;
  HALF_BIT_FLIP(tmp, random);
  x -= tmp;
  HALF_BIT_FLIP(x, random);

  random = RANDOM;
  tmp = y;
  HALF_BIT_FLIP(tmp, random);
  y = tmp;
  tmp = 1;
  HALF_BIT_FLIP(tmp, random);
  y -= tmp;
  HALF_BIT_FLIP(y, random);

  random = 1^random;
  tmp = z;
  HALF_BIT_FLIP(tmp, random);
  z = tmp;
  tmp = 1;
  HALF_BIT_FLIP(tmp, random);
  z -= tmp;
  HALF_BIT_FLIP(z, random);

  if (_cadna_cancel_tag){
    if (initial.accuracy==DIGIT_NOT_COMPUTED || initial.accuracy==RELIABLE_RESULT)
      initial.nb_significant_digit();
    if (initial.accuracy >=  this->nb_significant_digit()+_cadna_cancel_value)
      instability(&_cadna_cancel_count);
  }
  return *this;
};
#endif
/////////////////////////////////////////////////////////////////////////




inline float_st float_st::operator--(int){
  float_st initial=*this;

  unsigned int random;
  float tmp;

  accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = x;
  FLOAT_BIT_FLIP(tmp, random);
  x = tmp;
  tmp = 1;
  FLOAT_BIT_FLIP(tmp, random);
  x -= tmp;
  FLOAT_BIT_FLIP(x, random);

  random = RANDOM;
  tmp = y;
  FLOAT_BIT_FLIP(tmp, random);
  y = tmp;
  tmp = 1;
  FLOAT_BIT_FLIP(tmp, random);
  y -= tmp;
  FLOAT_BIT_FLIP(y, random);

  random = 1^random;
  tmp = z;
  FLOAT_BIT_FLIP(tmp, random);
  z = tmp;
  tmp = 1;
  FLOAT_BIT_FLIP(tmp, random);
  z -= tmp;
  FLOAT_BIT_FLIP(z, random);

  if (_cadna_cancel_tag){
    if (initial.accuracy==DIGIT_NOT_COMPUTED || initial.accuracy==RELIABLE_RESULT)
      initial.nb_significant_digit();
    if (initial.accuracy >=  this->nb_significant_digit()+_cadna_cancel_value)
      instability(&_cadna_cancel_count);
  }
  return initial;
};
inline double_st double_st::operator--(int){
  double_st initial=*this;

  unsigned int random;
  double tmp;

  accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = x;
  DOUBLE_BIT_FLIP(tmp, random);
  x = tmp;
  tmp = 1;
  DOUBLE_BIT_FLIP(tmp, random);
  x -= tmp;
  DOUBLE_BIT_FLIP(x, random);

  random = RANDOM;
  tmp = y;
  DOUBLE_BIT_FLIP(tmp, random);
  y = tmp;
  tmp = 1;
  DOUBLE_BIT_FLIP(tmp, random);
  y -= tmp;
  DOUBLE_BIT_FLIP(y, random);

  random = 1^random;
  tmp = z;
  DOUBLE_BIT_FLIP(tmp, random);
  z = tmp;
  tmp = 1;
  DOUBLE_BIT_FLIP(tmp, random);
  z -= tmp;
  DOUBLE_BIT_FLIP(z, random);

  if (_cadna_cancel_tag){
    if (initial.accuracy==DIGIT_NOT_COMPUTED || initial.accuracy==RELIABLE_RESULT)
      initial.nb_significant_digit();
    if (initial.accuracy >=  this->nb_significant_digit()+_cadna_cancel_value)
      instability(&_cadna_cancel_count);
  }
  return initial;
};
#ifdef CADNA_HALF_EMULATION
inline half_st half_st::operator--(int){
  half_st initial=*this;

  unsigned int random;
  half tmp;

  accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = x;
  HALF_BIT_FLIP(tmp, random);
  x = tmp;
  tmp = 1;
  HALF_BIT_FLIP(tmp, random);
  x -= tmp;
  HALF_BIT_FLIP(x, random);

  random = RANDOM;
  tmp = y;
  HALF_BIT_FLIP(tmp, random);
  y = tmp;
  tmp = 1;
  HALF_BIT_FLIP(tmp, random);
  y -= tmp;
  HALF_BIT_FLIP(y, random);

  random = 1^random;
  tmp = z;
  HALF_BIT_FLIP(tmp, random);
  z = tmp;
  tmp = 1;
  HALF_BIT_FLIP(tmp, random);
  z -= tmp;
  HALF_BIT_FLIP(z, random);

  if (_cadna_cancel_tag){
    if (initial.accuracy==DIGIT_NOT_COMPUTED || initial.accuracy==RELIABLE_RESULT)
      initial.nb_significant_digit();
    if (initial.accuracy >=  this->nb_significant_digit()+_cadna_cancel_value)
      instability(&_cadna_cancel_count);
  }
  return initial;
};
#endif
#ifdef CADNA_HALF_NATIVE
inline half_st half_st::operator--(int){
  half_st initial=*this;

  unsigned int random;
  half tmp;

  accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = x;
  HALF_BIT_FLIP(tmp, random);
  x = tmp;
  tmp = 1;
  HALF_BIT_FLIP(tmp, random);
  x -= tmp;
  HALF_BIT_FLIP(x, random);

  random = RANDOM;
  tmp = y;
  HALF_BIT_FLIP(tmp, random);
  y = tmp;
  tmp = 1;
  HALF_BIT_FLIP(tmp, random);
  y -= tmp;
  HALF_BIT_FLIP(y, random);

  random = 1^random;
  tmp = z;
  HALF_BIT_FLIP(tmp, random);
  z = tmp;
  tmp = 1;
  HALF_BIT_FLIP(tmp, random);
  z -= tmp;
  HALF_BIT_FLIP(z, random);

  if (_cadna_cancel_tag){
    if (initial.accuracy==DIGIT_NOT_COMPUTED || initial.accuracy==RELIABLE_RESULT)
      initial.nb_significant_digit();
    if (initial.accuracy >=  this->nb_significant_digit()+_cadna_cancel_value)
      instability(&_cadna_cancel_count);
  }
  return initial;
};
#endif
/////////////////////////////////////////////////////////////////////////





inline double_st& double_st::operator-=(const double& a){
  double_st initial=*this;

  unsigned int random;
  double tmp;

  accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = x;
  DOUBLE_BIT_FLIP(tmp, random);
  x = tmp;
  tmp = a;
  DOUBLE_BIT_FLIP(tmp, random);
  x -= tmp;
  DOUBLE_BIT_FLIP(x, random);

  random = RANDOM;
  tmp = y;
  DOUBLE_BIT_FLIP(tmp, random);
  y = tmp;
  tmp = a;
  DOUBLE_BIT_FLIP(tmp, random);
  y -= tmp;
  DOUBLE_BIT_FLIP(y, random);

  random = 1^random;
  tmp = z;
  DOUBLE_BIT_FLIP(tmp, random);
  z = tmp;
  tmp = a;
  DOUBLE_BIT_FLIP(tmp, random);
  z -= tmp;
  DOUBLE_BIT_FLIP(z, random);

  if (_cadna_cancel_tag){
    if (initial.accuracy==DIGIT_NOT_COMPUTED || initial.accuracy==RELIABLE_RESULT)
      initial.nb_significant_digit();
    if (initial.accuracy >=  this->nb_significant_digit()+_cadna_cancel_value)
      instability(&_cadna_cancel_count);
  }
  
  return *this;
}
inline double_st& double_st::operator-=(const float& a){
  double_st initial=*this;

  unsigned int random;
  double tmp;

  accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = x;
  DOUBLE_BIT_FLIP(tmp, random);
  x = tmp;
  tmp = a;
  DOUBLE_BIT_FLIP(tmp, random);
  x -= tmp;
  DOUBLE_BIT_FLIP(x, random);

  random = RANDOM;
  tmp = y;
  DOUBLE_BIT_FLIP(tmp, random);
  y = tmp;
  tmp = a;
  DOUBLE_BIT_FLIP(tmp, random);
  y -= tmp;
  DOUBLE_BIT_FLIP(y, random);

  random = 1^random;
  tmp = z;
  DOUBLE_BIT_FLIP(tmp, random);
  z = tmp;
  tmp = a;
  DOUBLE_BIT_FLIP(tmp, random);
  z -= tmp;
  DOUBLE_BIT_FLIP(z, random);

  if (_cadna_cancel_tag){
    if (initial.accuracy==DIGIT_NOT_COMPUTED || initial.accuracy==RELIABLE_RESULT)
      initial.nb_significant_digit();
    if (initial.accuracy >=  this->nb_significant_digit()+_cadna_cancel_value)
      instability(&_cadna_cancel_count);
  }
  
  return *this;
}
inline double_st& double_st::operator-=(const unsigned long long& a){
  double_st initial=*this;

  unsigned int random;
  double tmp;

  accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = x;
  DOUBLE_BIT_FLIP(tmp, random);
  x = tmp;
  tmp = a;
  DOUBLE_BIT_FLIP(tmp, random);
  x -= tmp;
  DOUBLE_BIT_FLIP(x, random);

  random = RANDOM;
  tmp = y;
  DOUBLE_BIT_FLIP(tmp, random);
  y = tmp;
  tmp = a;
  DOUBLE_BIT_FLIP(tmp, random);
  y -= tmp;
  DOUBLE_BIT_FLIP(y, random);

  random = 1^random;
  tmp = z;
  DOUBLE_BIT_FLIP(tmp, random);
  z = tmp;
  tmp = a;
  DOUBLE_BIT_FLIP(tmp, random);
  z -= tmp;
  DOUBLE_BIT_FLIP(z, random);

  if (_cadna_cancel_tag){
    if (initial.accuracy==DIGIT_NOT_COMPUTED || initial.accuracy==RELIABLE_RESULT)
      initial.nb_significant_digit();
    if (initial.accuracy >=  this->nb_significant_digit()+_cadna_cancel_value)
      instability(&_cadna_cancel_count);
  }
  
  return *this;
}
inline double_st& double_st::operator-=(const long long& a){
  double_st initial=*this;

  unsigned int random;
  double tmp;

  accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = x;
  DOUBLE_BIT_FLIP(tmp, random);
  x = tmp;
  tmp = a;
  DOUBLE_BIT_FLIP(tmp, random);
  x -= tmp;
  DOUBLE_BIT_FLIP(x, random);

  random = RANDOM;
  tmp = y;
  DOUBLE_BIT_FLIP(tmp, random);
  y = tmp;
  tmp = a;
  DOUBLE_BIT_FLIP(tmp, random);
  y -= tmp;
  DOUBLE_BIT_FLIP(y, random);

  random = 1^random;
  tmp = z;
  DOUBLE_BIT_FLIP(tmp, random);
  z = tmp;
  tmp = a;
  DOUBLE_BIT_FLIP(tmp, random);
  z -= tmp;
  DOUBLE_BIT_FLIP(z, random);

  if (_cadna_cancel_tag){
    if (initial.accuracy==DIGIT_NOT_COMPUTED || initial.accuracy==RELIABLE_RESULT)
      initial.nb_significant_digit();
    if (initial.accuracy >=  this->nb_significant_digit()+_cadna_cancel_value)
      instability(&_cadna_cancel_count);
  }
  
  return *this;
}
inline double_st& double_st::operator-=(const unsigned long& a){
  double_st initial=*this;

  unsigned int random;
  double tmp;

  accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = x;
  DOUBLE_BIT_FLIP(tmp, random);
  x = tmp;
  tmp = a;
  DOUBLE_BIT_FLIP(tmp, random);
  x -= tmp;
  DOUBLE_BIT_FLIP(x, random);

  random = RANDOM;
  tmp = y;
  DOUBLE_BIT_FLIP(tmp, random);
  y = tmp;
  tmp = a;
  DOUBLE_BIT_FLIP(tmp, random);
  y -= tmp;
  DOUBLE_BIT_FLIP(y, random);

  random = 1^random;
  tmp = z;
  DOUBLE_BIT_FLIP(tmp, random);
  z = tmp;
  tmp = a;
  DOUBLE_BIT_FLIP(tmp, random);
  z -= tmp;
  DOUBLE_BIT_FLIP(z, random);

  if (_cadna_cancel_tag){
    if (initial.accuracy==DIGIT_NOT_COMPUTED || initial.accuracy==RELIABLE_RESULT)
      initial.nb_significant_digit();
    if (initial.accuracy >=  this->nb_significant_digit()+_cadna_cancel_value)
      instability(&_cadna_cancel_count);
  }
  
  return *this;
}
inline double_st& double_st::operator-=(const long& a){
  double_st initial=*this;

  unsigned int random;
  double tmp;

  accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = x;
  DOUBLE_BIT_FLIP(tmp, random);
  x = tmp;
  tmp = a;
  DOUBLE_BIT_FLIP(tmp, random);
  x -= tmp;
  DOUBLE_BIT_FLIP(x, random);

  random = RANDOM;
  tmp = y;
  DOUBLE_BIT_FLIP(tmp, random);
  y = tmp;
  tmp = a;
  DOUBLE_BIT_FLIP(tmp, random);
  y -= tmp;
  DOUBLE_BIT_FLIP(y, random);

  random = 1^random;
  tmp = z;
  DOUBLE_BIT_FLIP(tmp, random);
  z = tmp;
  tmp = a;
  DOUBLE_BIT_FLIP(tmp, random);
  z -= tmp;
  DOUBLE_BIT_FLIP(z, random);

  if (_cadna_cancel_tag){
    if (initial.accuracy==DIGIT_NOT_COMPUTED || initial.accuracy==RELIABLE_RESULT)
      initial.nb_significant_digit();
    if (initial.accuracy >=  this->nb_significant_digit()+_cadna_cancel_value)
      instability(&_cadna_cancel_count);
  }
  
  return *this;
}
inline double_st& double_st::operator-=(const unsigned int& a){
  double_st initial=*this;

  unsigned int random;
  double tmp;

  accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = x;
  DOUBLE_BIT_FLIP(tmp, random);
  x = tmp;
  tmp = a;
  DOUBLE_BIT_FLIP(tmp, random);
  x -= tmp;
  DOUBLE_BIT_FLIP(x, random);

  random = RANDOM;
  tmp = y;
  DOUBLE_BIT_FLIP(tmp, random);
  y = tmp;
  tmp = a;
  DOUBLE_BIT_FLIP(tmp, random);
  y -= tmp;
  DOUBLE_BIT_FLIP(y, random);

  random = 1^random;
  tmp = z;
  DOUBLE_BIT_FLIP(tmp, random);
  z = tmp;
  tmp = a;
  DOUBLE_BIT_FLIP(tmp, random);
  z -= tmp;
  DOUBLE_BIT_FLIP(z, random);

  if (_cadna_cancel_tag){
    if (initial.accuracy==DIGIT_NOT_COMPUTED || initial.accuracy==RELIABLE_RESULT)
      initial.nb_significant_digit();
    if (initial.accuracy >=  this->nb_significant_digit()+_cadna_cancel_value)
      instability(&_cadna_cancel_count);
  }
  
  return *this;
}
inline double_st& double_st::operator-=(const int& a){
  double_st initial=*this;

  unsigned int random;
  double tmp;

  accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = x;
  DOUBLE_BIT_FLIP(tmp, random);
  x = tmp;
  tmp = a;
  DOUBLE_BIT_FLIP(tmp, random);
  x -= tmp;
  DOUBLE_BIT_FLIP(x, random);

  random = RANDOM;
  tmp = y;
  DOUBLE_BIT_FLIP(tmp, random);
  y = tmp;
  tmp = a;
  DOUBLE_BIT_FLIP(tmp, random);
  y -= tmp;
  DOUBLE_BIT_FLIP(y, random);

  random = 1^random;
  tmp = z;
  DOUBLE_BIT_FLIP(tmp, random);
  z = tmp;
  tmp = a;
  DOUBLE_BIT_FLIP(tmp, random);
  z -= tmp;
  DOUBLE_BIT_FLIP(z, random);

  if (_cadna_cancel_tag){
    if (initial.accuracy==DIGIT_NOT_COMPUTED || initial.accuracy==RELIABLE_RESULT)
      initial.nb_significant_digit();
    if (initial.accuracy >=  this->nb_significant_digit()+_cadna_cancel_value)
      instability(&_cadna_cancel_count);
  }
  
  return *this;
}
inline double_st& double_st::operator-=(const unsigned short& a){
  double_st initial=*this;

  unsigned int random;
  double tmp;

  accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = x;
  DOUBLE_BIT_FLIP(tmp, random);
  x = tmp;
  tmp = a;
  DOUBLE_BIT_FLIP(tmp, random);
  x -= tmp;
  DOUBLE_BIT_FLIP(x, random);

  random = RANDOM;
  tmp = y;
  DOUBLE_BIT_FLIP(tmp, random);
  y = tmp;
  tmp = a;
  DOUBLE_BIT_FLIP(tmp, random);
  y -= tmp;
  DOUBLE_BIT_FLIP(y, random);

  random = 1^random;
  tmp = z;
  DOUBLE_BIT_FLIP(tmp, random);
  z = tmp;
  tmp = a;
  DOUBLE_BIT_FLIP(tmp, random);
  z -= tmp;
  DOUBLE_BIT_FLIP(z, random);

  if (_cadna_cancel_tag){
    if (initial.accuracy==DIGIT_NOT_COMPUTED || initial.accuracy==RELIABLE_RESULT)
      initial.nb_significant_digit();
    if (initial.accuracy >=  this->nb_significant_digit()+_cadna_cancel_value)
      instability(&_cadna_cancel_count);
  }
  
  return *this;
}
inline double_st& double_st::operator-=(const short& a){
  double_st initial=*this;

  unsigned int random;
  double tmp;

  accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = x;
  DOUBLE_BIT_FLIP(tmp, random);
  x = tmp;
  tmp = a;
  DOUBLE_BIT_FLIP(tmp, random);
  x -= tmp;
  DOUBLE_BIT_FLIP(x, random);

  random = RANDOM;
  tmp = y;
  DOUBLE_BIT_FLIP(tmp, random);
  y = tmp;
  tmp = a;
  DOUBLE_BIT_FLIP(tmp, random);
  y -= tmp;
  DOUBLE_BIT_FLIP(y, random);

  random = 1^random;
  tmp = z;
  DOUBLE_BIT_FLIP(tmp, random);
  z = tmp;
  tmp = a;
  DOUBLE_BIT_FLIP(tmp, random);
  z -= tmp;
  DOUBLE_BIT_FLIP(z, random);

  if (_cadna_cancel_tag){
    if (initial.accuracy==DIGIT_NOT_COMPUTED || initial.accuracy==RELIABLE_RESULT)
      initial.nb_significant_digit();
    if (initial.accuracy >=  this->nb_significant_digit()+_cadna_cancel_value)
      instability(&_cadna_cancel_count);
  }
  
  return *this;
}
inline double_st& double_st::operator-=(const unsigned char& a){
  double_st initial=*this;

  unsigned int random;
  double tmp;

  accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = x;
  DOUBLE_BIT_FLIP(tmp, random);
  x = tmp;
  tmp = a;
  DOUBLE_BIT_FLIP(tmp, random);
  x -= tmp;
  DOUBLE_BIT_FLIP(x, random);

  random = RANDOM;
  tmp = y;
  DOUBLE_BIT_FLIP(tmp, random);
  y = tmp;
  tmp = a;
  DOUBLE_BIT_FLIP(tmp, random);
  y -= tmp;
  DOUBLE_BIT_FLIP(y, random);

  random = 1^random;
  tmp = z;
  DOUBLE_BIT_FLIP(tmp, random);
  z = tmp;
  tmp = a;
  DOUBLE_BIT_FLIP(tmp, random);
  z -= tmp;
  DOUBLE_BIT_FLIP(z, random);

  if (_cadna_cancel_tag){
    if (initial.accuracy==DIGIT_NOT_COMPUTED || initial.accuracy==RELIABLE_RESULT)
      initial.nb_significant_digit();
    if (initial.accuracy >=  this->nb_significant_digit()+_cadna_cancel_value)
      instability(&_cadna_cancel_count);
  }
  
  return *this;
}
inline double_st& double_st::operator-=(const char& a){
  double_st initial=*this;

  unsigned int random;
  double tmp;

  accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = x;
  DOUBLE_BIT_FLIP(tmp, random);
  x = tmp;
  tmp = a;
  DOUBLE_BIT_FLIP(tmp, random);
  x -= tmp;
  DOUBLE_BIT_FLIP(x, random);

  random = RANDOM;
  tmp = y;
  DOUBLE_BIT_FLIP(tmp, random);
  y = tmp;
  tmp = a;
  DOUBLE_BIT_FLIP(tmp, random);
  y -= tmp;
  DOUBLE_BIT_FLIP(y, random);

  random = 1^random;
  tmp = z;
  DOUBLE_BIT_FLIP(tmp, random);
  z = tmp;
  tmp = a;
  DOUBLE_BIT_FLIP(tmp, random);
  z -= tmp;
  DOUBLE_BIT_FLIP(z, random);

  if (_cadna_cancel_tag){
    if (initial.accuracy==DIGIT_NOT_COMPUTED || initial.accuracy==RELIABLE_RESULT)
      initial.nb_significant_digit();
    if (initial.accuracy >=  this->nb_significant_digit()+_cadna_cancel_value)
      instability(&_cadna_cancel_count);
  }
  
  return *this;
}
#ifdef CADNA_HALF_EMULATION
inline double_st& double_st::operator-=(const half& a){
  double_st initial=*this;

  unsigned int random;
  double tmp;

  accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = x;
  DOUBLE_BIT_FLIP(tmp, random);
  x = tmp;
  tmp = a;
  DOUBLE_BIT_FLIP(tmp, random);
  x -= tmp;
  DOUBLE_BIT_FLIP(x, random);

  random = RANDOM;
  tmp = y;
  DOUBLE_BIT_FLIP(tmp, random);
  y = tmp;
  tmp = a;
  DOUBLE_BIT_FLIP(tmp, random);
  y -= tmp;
  DOUBLE_BIT_FLIP(y, random);

  random = 1^random;
  tmp = z;
  DOUBLE_BIT_FLIP(tmp, random);
  z = tmp;
  tmp = a;
  DOUBLE_BIT_FLIP(tmp, random);
  z -= tmp;
  DOUBLE_BIT_FLIP(z, random);

  if (_cadna_cancel_tag){
    if (initial.accuracy==DIGIT_NOT_COMPUTED || initial.accuracy==RELIABLE_RESULT)
      initial.nb_significant_digit();
    if (initial.accuracy >=  this->nb_significant_digit()+_cadna_cancel_value)
      instability(&_cadna_cancel_count);
  }
  
  return *this;
}
#endif // CADNA_HALF_EMULATION

#ifdef CADNA_HALF_NATIVE
inline double_st& double_st::operator-=(const half& a){
  double_st initial=*this;

  unsigned int random;
  double tmp;

  accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = x;
  DOUBLE_BIT_FLIP(tmp, random);
  x = tmp;
  tmp = a;
  DOUBLE_BIT_FLIP(tmp, random);
  x -= tmp;
  DOUBLE_BIT_FLIP(x, random);

  random = RANDOM;
  tmp = y;
  DOUBLE_BIT_FLIP(tmp, random);
  y = tmp;
  tmp = a;
  DOUBLE_BIT_FLIP(tmp, random);
  y -= tmp;
  DOUBLE_BIT_FLIP(y, random);

  random = 1^random;
  tmp = z;
  DOUBLE_BIT_FLIP(tmp, random);
  z = tmp;
  tmp = a;
  DOUBLE_BIT_FLIP(tmp, random);
  z -= tmp;
  DOUBLE_BIT_FLIP(z, random);

  if (_cadna_cancel_tag){
    if (initial.accuracy==DIGIT_NOT_COMPUTED || initial.accuracy==RELIABLE_RESULT)
      initial.nb_significant_digit();
    if (initial.accuracy >=  this->nb_significant_digit()+_cadna_cancel_value)
      instability(&_cadna_cancel_count);
  }
  
  return *this;
}
#endif

inline float_st& float_st::operator-=(const double& a){
  float_st initial=*this;

  unsigned int random;
  float tmp;

  accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = x;
  FLOAT_BIT_FLIP(tmp, random);
  x = tmp;
  tmp = a;
  FLOAT_BIT_FLIP(tmp, random);
  x -= tmp;
  FLOAT_BIT_FLIP(x, random);

  random = RANDOM;
  tmp = y;
  FLOAT_BIT_FLIP(tmp, random);
  y = tmp;
  tmp = a;
  FLOAT_BIT_FLIP(tmp, random);
  y -= tmp;
  FLOAT_BIT_FLIP(y, random);

  random = 1^random;
  tmp = z;
  FLOAT_BIT_FLIP(tmp, random);
  z = tmp;
  tmp = a;
  FLOAT_BIT_FLIP(tmp, random);
  z -= tmp;
  FLOAT_BIT_FLIP(z, random);

  if (_cadna_cancel_tag){
    if (initial.accuracy==DIGIT_NOT_COMPUTED || initial.accuracy==RELIABLE_RESULT)
      initial.nb_significant_digit();
    if (initial.accuracy >=  this->nb_significant_digit()+_cadna_cancel_value)
      instability(&_cadna_cancel_count);
  }
  
  return *this;
}
inline float_st& float_st::operator-=(const float& a){
  float_st initial=*this;

  unsigned int random;
  float tmp;

  accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = x;
  FLOAT_BIT_FLIP(tmp, random);
  x = tmp;
  tmp = a;
  FLOAT_BIT_FLIP(tmp, random);
  x -= tmp;
  FLOAT_BIT_FLIP(x, random);

  random = RANDOM;
  tmp = y;
  FLOAT_BIT_FLIP(tmp, random);
  y = tmp;
  tmp = a;
  FLOAT_BIT_FLIP(tmp, random);
  y -= tmp;
  FLOAT_BIT_FLIP(y, random);

  random = 1^random;
  tmp = z;
  FLOAT_BIT_FLIP(tmp, random);
  z = tmp;
  tmp = a;
  FLOAT_BIT_FLIP(tmp, random);
  z -= tmp;
  FLOAT_BIT_FLIP(z, random);

  if (_cadna_cancel_tag){
    if (initial.accuracy==DIGIT_NOT_COMPUTED || initial.accuracy==RELIABLE_RESULT)
      initial.nb_significant_digit();
    if (initial.accuracy >=  this->nb_significant_digit()+_cadna_cancel_value)
      instability(&_cadna_cancel_count);
  }
  
  return *this;
}
inline float_st& float_st::operator-=(const unsigned long long& a){
  float_st initial=*this;

  unsigned int random;
  float tmp;

  accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = x;
  FLOAT_BIT_FLIP(tmp, random);
  x = tmp;
  tmp = a;
  FLOAT_BIT_FLIP(tmp, random);
  x -= tmp;
  FLOAT_BIT_FLIP(x, random);

  random = RANDOM;
  tmp = y;
  FLOAT_BIT_FLIP(tmp, random);
  y = tmp;
  tmp = a;
  FLOAT_BIT_FLIP(tmp, random);
  y -= tmp;
  FLOAT_BIT_FLIP(y, random);

  random = 1^random;
  tmp = z;
  FLOAT_BIT_FLIP(tmp, random);
  z = tmp;
  tmp = a;
  FLOAT_BIT_FLIP(tmp, random);
  z -= tmp;
  FLOAT_BIT_FLIP(z, random);

  if (_cadna_cancel_tag){
    if (initial.accuracy==DIGIT_NOT_COMPUTED || initial.accuracy==RELIABLE_RESULT)
      initial.nb_significant_digit();
    if (initial.accuracy >=  this->nb_significant_digit()+_cadna_cancel_value)
      instability(&_cadna_cancel_count);
  }
  
  return *this;
}
inline float_st& float_st::operator-=(const long long& a){
  float_st initial=*this;

  unsigned int random;
  float tmp;

  accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = x;
  FLOAT_BIT_FLIP(tmp, random);
  x = tmp;
  tmp = a;
  FLOAT_BIT_FLIP(tmp, random);
  x -= tmp;
  FLOAT_BIT_FLIP(x, random);

  random = RANDOM;
  tmp = y;
  FLOAT_BIT_FLIP(tmp, random);
  y = tmp;
  tmp = a;
  FLOAT_BIT_FLIP(tmp, random);
  y -= tmp;
  FLOAT_BIT_FLIP(y, random);

  random = 1^random;
  tmp = z;
  FLOAT_BIT_FLIP(tmp, random);
  z = tmp;
  tmp = a;
  FLOAT_BIT_FLIP(tmp, random);
  z -= tmp;
  FLOAT_BIT_FLIP(z, random);

  if (_cadna_cancel_tag){
    if (initial.accuracy==DIGIT_NOT_COMPUTED || initial.accuracy==RELIABLE_RESULT)
      initial.nb_significant_digit();
    if (initial.accuracy >=  this->nb_significant_digit()+_cadna_cancel_value)
      instability(&_cadna_cancel_count);
  }
  
  return *this;
}
inline float_st& float_st::operator-=(const unsigned long& a){
  float_st initial=*this;

  unsigned int random;
  float tmp;

  accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = x;
  FLOAT_BIT_FLIP(tmp, random);
  x = tmp;
  tmp = a;
  FLOAT_BIT_FLIP(tmp, random);
  x -= tmp;
  FLOAT_BIT_FLIP(x, random);

  random = RANDOM;
  tmp = y;
  FLOAT_BIT_FLIP(tmp, random);
  y = tmp;
  tmp = a;
  FLOAT_BIT_FLIP(tmp, random);
  y -= tmp;
  FLOAT_BIT_FLIP(y, random);

  random = 1^random;
  tmp = z;
  FLOAT_BIT_FLIP(tmp, random);
  z = tmp;
  tmp = a;
  FLOAT_BIT_FLIP(tmp, random);
  z -= tmp;
  FLOAT_BIT_FLIP(z, random);

  if (_cadna_cancel_tag){
    if (initial.accuracy==DIGIT_NOT_COMPUTED || initial.accuracy==RELIABLE_RESULT)
      initial.nb_significant_digit();
    if (initial.accuracy >=  this->nb_significant_digit()+_cadna_cancel_value)
      instability(&_cadna_cancel_count);
  }
  
  return *this;
}
inline float_st& float_st::operator-=(const long& a){
  float_st initial=*this;

  unsigned int random;
  float tmp;

  accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = x;
  FLOAT_BIT_FLIP(tmp, random);
  x = tmp;
  tmp = a;
  FLOAT_BIT_FLIP(tmp, random);
  x -= tmp;
  FLOAT_BIT_FLIP(x, random);

  random = RANDOM;
  tmp = y;
  FLOAT_BIT_FLIP(tmp, random);
  y = tmp;
  tmp = a;
  FLOAT_BIT_FLIP(tmp, random);
  y -= tmp;
  FLOAT_BIT_FLIP(y, random);

  random = 1^random;
  tmp = z;
  FLOAT_BIT_FLIP(tmp, random);
  z = tmp;
  tmp = a;
  FLOAT_BIT_FLIP(tmp, random);
  z -= tmp;
  FLOAT_BIT_FLIP(z, random);

  if (_cadna_cancel_tag){
    if (initial.accuracy==DIGIT_NOT_COMPUTED || initial.accuracy==RELIABLE_RESULT)
      initial.nb_significant_digit();
    if (initial.accuracy >=  this->nb_significant_digit()+_cadna_cancel_value)
      instability(&_cadna_cancel_count);
  }
  
  return *this;
}
inline float_st& float_st::operator-=(const unsigned int& a){
  float_st initial=*this;

  unsigned int random;
  float tmp;

  accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = x;
  FLOAT_BIT_FLIP(tmp, random);
  x = tmp;
  tmp = a;
  FLOAT_BIT_FLIP(tmp, random);
  x -= tmp;
  FLOAT_BIT_FLIP(x, random);

  random = RANDOM;
  tmp = y;
  FLOAT_BIT_FLIP(tmp, random);
  y = tmp;
  tmp = a;
  FLOAT_BIT_FLIP(tmp, random);
  y -= tmp;
  FLOAT_BIT_FLIP(y, random);

  random = 1^random;
  tmp = z;
  FLOAT_BIT_FLIP(tmp, random);
  z = tmp;
  tmp = a;
  FLOAT_BIT_FLIP(tmp, random);
  z -= tmp;
  FLOAT_BIT_FLIP(z, random);

  if (_cadna_cancel_tag){
    if (initial.accuracy==DIGIT_NOT_COMPUTED || initial.accuracy==RELIABLE_RESULT)
      initial.nb_significant_digit();
    if (initial.accuracy >=  this->nb_significant_digit()+_cadna_cancel_value)
      instability(&_cadna_cancel_count);
  }
  
  return *this;
}
inline float_st& float_st::operator-=(const int& a){
  float_st initial=*this;

  unsigned int random;
  float tmp;

  accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = x;
  FLOAT_BIT_FLIP(tmp, random);
  x = tmp;
  tmp = a;
  FLOAT_BIT_FLIP(tmp, random);
  x -= tmp;
  FLOAT_BIT_FLIP(x, random);

  random = RANDOM;
  tmp = y;
  FLOAT_BIT_FLIP(tmp, random);
  y = tmp;
  tmp = a;
  FLOAT_BIT_FLIP(tmp, random);
  y -= tmp;
  FLOAT_BIT_FLIP(y, random);

  random = 1^random;
  tmp = z;
  FLOAT_BIT_FLIP(tmp, random);
  z = tmp;
  tmp = a;
  FLOAT_BIT_FLIP(tmp, random);
  z -= tmp;
  FLOAT_BIT_FLIP(z, random);

  if (_cadna_cancel_tag){
    if (initial.accuracy==DIGIT_NOT_COMPUTED || initial.accuracy==RELIABLE_RESULT)
      initial.nb_significant_digit();
    if (initial.accuracy >=  this->nb_significant_digit()+_cadna_cancel_value)
      instability(&_cadna_cancel_count);
  }
  
  return *this;
}
inline float_st& float_st::operator-=(const unsigned short& a){
  float_st initial=*this;

  unsigned int random;
  float tmp;

  accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = x;
  FLOAT_BIT_FLIP(tmp, random);
  x = tmp;
  tmp = a;
  FLOAT_BIT_FLIP(tmp, random);
  x -= tmp;
  FLOAT_BIT_FLIP(x, random);

  random = RANDOM;
  tmp = y;
  FLOAT_BIT_FLIP(tmp, random);
  y = tmp;
  tmp = a;
  FLOAT_BIT_FLIP(tmp, random);
  y -= tmp;
  FLOAT_BIT_FLIP(y, random);

  random = 1^random;
  tmp = z;
  FLOAT_BIT_FLIP(tmp, random);
  z = tmp;
  tmp = a;
  FLOAT_BIT_FLIP(tmp, random);
  z -= tmp;
  FLOAT_BIT_FLIP(z, random);

  if (_cadna_cancel_tag){
    if (initial.accuracy==DIGIT_NOT_COMPUTED || initial.accuracy==RELIABLE_RESULT)
      initial.nb_significant_digit();
    if (initial.accuracy >=  this->nb_significant_digit()+_cadna_cancel_value)
      instability(&_cadna_cancel_count);
  }
  
  return *this;
}
inline float_st& float_st::operator-=(const short& a){
  float_st initial=*this;

  unsigned int random;
  float tmp;

  accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = x;
  FLOAT_BIT_FLIP(tmp, random);
  x = tmp;
  tmp = a;
  FLOAT_BIT_FLIP(tmp, random);
  x -= tmp;
  FLOAT_BIT_FLIP(x, random);

  random = RANDOM;
  tmp = y;
  FLOAT_BIT_FLIP(tmp, random);
  y = tmp;
  tmp = a;
  FLOAT_BIT_FLIP(tmp, random);
  y -= tmp;
  FLOAT_BIT_FLIP(y, random);

  random = 1^random;
  tmp = z;
  FLOAT_BIT_FLIP(tmp, random);
  z = tmp;
  tmp = a;
  FLOAT_BIT_FLIP(tmp, random);
  z -= tmp;
  FLOAT_BIT_FLIP(z, random);

  if (_cadna_cancel_tag){
    if (initial.accuracy==DIGIT_NOT_COMPUTED || initial.accuracy==RELIABLE_RESULT)
      initial.nb_significant_digit();
    if (initial.accuracy >=  this->nb_significant_digit()+_cadna_cancel_value)
      instability(&_cadna_cancel_count);
  }
  
  return *this;
}
inline float_st& float_st::operator-=(const unsigned char& a){
  float_st initial=*this;

  unsigned int random;
  float tmp;

  accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = x;
  FLOAT_BIT_FLIP(tmp, random);
  x = tmp;
  tmp = a;
  FLOAT_BIT_FLIP(tmp, random);
  x -= tmp;
  FLOAT_BIT_FLIP(x, random);

  random = RANDOM;
  tmp = y;
  FLOAT_BIT_FLIP(tmp, random);
  y = tmp;
  tmp = a;
  FLOAT_BIT_FLIP(tmp, random);
  y -= tmp;
  FLOAT_BIT_FLIP(y, random);

  random = 1^random;
  tmp = z;
  FLOAT_BIT_FLIP(tmp, random);
  z = tmp;
  tmp = a;
  FLOAT_BIT_FLIP(tmp, random);
  z -= tmp;
  FLOAT_BIT_FLIP(z, random);

  if (_cadna_cancel_tag){
    if (initial.accuracy==DIGIT_NOT_COMPUTED || initial.accuracy==RELIABLE_RESULT)
      initial.nb_significant_digit();
    if (initial.accuracy >=  this->nb_significant_digit()+_cadna_cancel_value)
      instability(&_cadna_cancel_count);
  }
  
  return *this;
}
inline float_st& float_st::operator-=(const char& a){
  float_st initial=*this;

  unsigned int random;
  float tmp;

  accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = x;
  FLOAT_BIT_FLIP(tmp, random);
  x = tmp;
  tmp = a;
  FLOAT_BIT_FLIP(tmp, random);
  x -= tmp;
  FLOAT_BIT_FLIP(x, random);

  random = RANDOM;
  tmp = y;
  FLOAT_BIT_FLIP(tmp, random);
  y = tmp;
  tmp = a;
  FLOAT_BIT_FLIP(tmp, random);
  y -= tmp;
  FLOAT_BIT_FLIP(y, random);

  random = 1^random;
  tmp = z;
  FLOAT_BIT_FLIP(tmp, random);
  z = tmp;
  tmp = a;
  FLOAT_BIT_FLIP(tmp, random);
  z -= tmp;
  FLOAT_BIT_FLIP(z, random);

  if (_cadna_cancel_tag){
    if (initial.accuracy==DIGIT_NOT_COMPUTED || initial.accuracy==RELIABLE_RESULT)
      initial.nb_significant_digit();
    if (initial.accuracy >=  this->nb_significant_digit()+_cadna_cancel_value)
      instability(&_cadna_cancel_count);
  }
  
  return *this;
}
#ifdef CADNA_HALF_EMULATION
inline float_st& float_st::operator-=(const half& a){
  float_st initial=*this;

  unsigned int random;
  float tmp;

  accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = x;
  FLOAT_BIT_FLIP(tmp, random);
  x = tmp;
  tmp = a;
  FLOAT_BIT_FLIP(tmp, random);
  x -= tmp;
  FLOAT_BIT_FLIP(x, random);

  random = RANDOM;
  tmp = y;
  FLOAT_BIT_FLIP(tmp, random);
  y = tmp;
  tmp = a;
  FLOAT_BIT_FLIP(tmp, random);
  y -= tmp;
  FLOAT_BIT_FLIP(y, random);

  random = 1^random;
  tmp = z;
  FLOAT_BIT_FLIP(tmp, random);
  z = tmp;
  tmp = a;
  FLOAT_BIT_FLIP(tmp, random);
  z -= tmp;
  FLOAT_BIT_FLIP(z, random);

  if (_cadna_cancel_tag){
    if (initial.accuracy==DIGIT_NOT_COMPUTED || initial.accuracy==RELIABLE_RESULT)
      initial.nb_significant_digit();
    if (initial.accuracy >=  this->nb_significant_digit()+_cadna_cancel_value)
      instability(&_cadna_cancel_count);
  }
  
  return *this;
}
#endif // CADNA_HALF_EMULATION

#ifdef CADNA_HALF_NATIVE
inline float_st& float_st::operator-=(const half& a){
  float_st initial=*this;

  unsigned int random;
  float tmp;

  accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = x;
  FLOAT_BIT_FLIP(tmp, random);
  x = tmp;
  tmp = a;
  FLOAT_BIT_FLIP(tmp, random);
  x -= tmp;
  FLOAT_BIT_FLIP(x, random);

  random = RANDOM;
  tmp = y;
  FLOAT_BIT_FLIP(tmp, random);
  y = tmp;
  tmp = a;
  FLOAT_BIT_FLIP(tmp, random);
  y -= tmp;
  FLOAT_BIT_FLIP(y, random);

  random = 1^random;
  tmp = z;
  FLOAT_BIT_FLIP(tmp, random);
  z = tmp;
  tmp = a;
  FLOAT_BIT_FLIP(tmp, random);
  z -= tmp;
  FLOAT_BIT_FLIP(z, random);

  if (_cadna_cancel_tag){
    if (initial.accuracy==DIGIT_NOT_COMPUTED || initial.accuracy==RELIABLE_RESULT)
      initial.nb_significant_digit();
    if (initial.accuracy >=  this->nb_significant_digit()+_cadna_cancel_value)
      instability(&_cadna_cancel_count);
  }
  
  return *this;
}
#endif

#ifdef CADNA_HALF_EMULATION
inline half_st& half_st::operator-=(const double& a){
  half_st initial=*this;

  unsigned int random;
  half tmp;

  accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = x;
  HALF_BIT_FLIP(tmp, random);
  x = tmp;
  tmp = a;
  HALF_BIT_FLIP(tmp, random);
  x -= tmp;
  HALF_BIT_FLIP(x, random);

  random = RANDOM;
  tmp = y;
  HALF_BIT_FLIP(tmp, random);
  y = tmp;
  tmp = a;
  HALF_BIT_FLIP(tmp, random);
  y -= tmp;
  HALF_BIT_FLIP(y, random);

  random = 1^random;
  tmp = z;
  HALF_BIT_FLIP(tmp, random);
  z = tmp;
  tmp = a;
  HALF_BIT_FLIP(tmp, random);
  z -= tmp;
  HALF_BIT_FLIP(z, random);

  if (_cadna_cancel_tag){
    if (initial.accuracy==DIGIT_NOT_COMPUTED || initial.accuracy==RELIABLE_RESULT)
      initial.nb_significant_digit();
    if (initial.accuracy >=  this->nb_significant_digit()+_cadna_cancel_value)
      instability(&_cadna_cancel_count);
  }
  
  return *this;
}
inline half_st& half_st::operator-=(const float& a){
  half_st initial=*this;

  unsigned int random;
  half tmp;

  accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = x;
  HALF_BIT_FLIP(tmp, random);
  x = tmp;
  tmp = a;
  HALF_BIT_FLIP(tmp, random);
  x -= tmp;
  HALF_BIT_FLIP(x, random);

  random = RANDOM;
  tmp = y;
  HALF_BIT_FLIP(tmp, random);
  y = tmp;
  tmp = a;
  HALF_BIT_FLIP(tmp, random);
  y -= tmp;
  HALF_BIT_FLIP(y, random);

  random = 1^random;
  tmp = z;
  HALF_BIT_FLIP(tmp, random);
  z = tmp;
  tmp = a;
  HALF_BIT_FLIP(tmp, random);
  z -= tmp;
  HALF_BIT_FLIP(z, random);

  if (_cadna_cancel_tag){
    if (initial.accuracy==DIGIT_NOT_COMPUTED || initial.accuracy==RELIABLE_RESULT)
      initial.nb_significant_digit();
    if (initial.accuracy >=  this->nb_significant_digit()+_cadna_cancel_value)
      instability(&_cadna_cancel_count);
  }
  
  return *this;
}
inline half_st& half_st::operator-=(const unsigned long long& a){
  half_st initial=*this;
  unsigned long long tmp2;
  unsigned int random;
  half tmp;

  accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp2=(random) ? -a : a;
  tmp = x;
  HALF_BIT_FLIP(tmp, random);
  x = tmp;
  x -= tmp2;
  HALF_BIT_FLIP(x, random);

  random = RANDOM;
  tmp2=(random) ? -a : a;
  tmp = y;
  HALF_BIT_FLIP(tmp, random);
  y = tmp;
  y -= tmp2;
  HALF_BIT_FLIP(y, random);

  random = 1^random;
  tmp2=(random) ? -a : a;
  tmp = z;
  HALF_BIT_FLIP(tmp, random);
  z = tmp;
  z -= tmp2;
  HALF_BIT_FLIP(z, random);

  
  
  if (_cadna_cancel_tag){
    if (initial.accuracy==DIGIT_NOT_COMPUTED || initial.accuracy==RELIABLE_RESULT)
      initial.nb_significant_digit();
    if (initial.accuracy >=  this->nb_significant_digit()+_cadna_cancel_value)
      instability(&_cadna_cancel_count);
  }
  if (_cadna_half_overflow_tag){
  if (std::isinf(x) or std::isinf(y) or std::isinf(z)){
  instability(&_cadna_half_overflow_count);
  }
}


  return *this;
}
inline half_st& half_st::operator-=(const long long& a){
  half_st initial=*this;
  long long tmp2;
  unsigned int random;
  half tmp;

  accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp2=(random) ? -a : a;
  tmp = x;
  HALF_BIT_FLIP(tmp, random);
  x = tmp;
  x -= tmp2;
  HALF_BIT_FLIP(x, random);

  random = RANDOM;
  tmp2=(random) ? -a : a;
  tmp = y;
  HALF_BIT_FLIP(tmp, random);
  y = tmp;
  y -= tmp2;
  HALF_BIT_FLIP(y, random);

  random = 1^random;
  tmp2=(random) ? -a : a;
  tmp = z;
  HALF_BIT_FLIP(tmp, random);
  z = tmp;
  z -= tmp2;
  HALF_BIT_FLIP(z, random);

  
  
  if (_cadna_cancel_tag){
    if (initial.accuracy==DIGIT_NOT_COMPUTED || initial.accuracy==RELIABLE_RESULT)
      initial.nb_significant_digit();
    if (initial.accuracy >=  this->nb_significant_digit()+_cadna_cancel_value)
      instability(&_cadna_cancel_count);
  }
  if (_cadna_half_overflow_tag){
  if (std::isinf(x) or std::isinf(y) or std::isinf(z)){
  instability(&_cadna_half_overflow_count);
  }
}


  return *this;
}
inline half_st& half_st::operator-=(const unsigned long& a){
  half_st initial=*this;
  unsigned long tmp2;
  unsigned int random;
  half tmp;

  accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp2=(random) ? -a : a;
  tmp = x;
  HALF_BIT_FLIP(tmp, random);
  x = tmp;
  x -= tmp2;
  HALF_BIT_FLIP(x, random);

  random = RANDOM;
  tmp2=(random) ? -a : a;
  tmp = y;
  HALF_BIT_FLIP(tmp, random);
  y = tmp;
  y -= tmp2;
  HALF_BIT_FLIP(y, random);

  random = 1^random;
  tmp2=(random) ? -a : a;
  tmp = z;
  HALF_BIT_FLIP(tmp, random);
  z = tmp;
  z -= tmp2;
  HALF_BIT_FLIP(z, random);

  
  
  if (_cadna_cancel_tag){
    if (initial.accuracy==DIGIT_NOT_COMPUTED || initial.accuracy==RELIABLE_RESULT)
      initial.nb_significant_digit();
    if (initial.accuracy >=  this->nb_significant_digit()+_cadna_cancel_value)
      instability(&_cadna_cancel_count);
  }
  if (_cadna_half_overflow_tag){
  if (std::isinf(x) or std::isinf(y) or std::isinf(z)){
  instability(&_cadna_half_overflow_count);
  }
}


  return *this;
}
inline half_st& half_st::operator-=(const long& a){
  half_st initial=*this;
  long tmp2;
  unsigned int random;
  half tmp;

  accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp2=(random) ? -a : a;
  tmp = x;
  HALF_BIT_FLIP(tmp, random);
  x = tmp;
  x -= tmp2;
  HALF_BIT_FLIP(x, random);

  random = RANDOM;
  tmp2=(random) ? -a : a;
  tmp = y;
  HALF_BIT_FLIP(tmp, random);
  y = tmp;
  y -= tmp2;
  HALF_BIT_FLIP(y, random);

  random = 1^random;
  tmp2=(random) ? -a : a;
  tmp = z;
  HALF_BIT_FLIP(tmp, random);
  z = tmp;
  z -= tmp2;
  HALF_BIT_FLIP(z, random);

  
  
  if (_cadna_cancel_tag){
    if (initial.accuracy==DIGIT_NOT_COMPUTED || initial.accuracy==RELIABLE_RESULT)
      initial.nb_significant_digit();
    if (initial.accuracy >=  this->nb_significant_digit()+_cadna_cancel_value)
      instability(&_cadna_cancel_count);
  }
  if (_cadna_half_overflow_tag){
  if (std::isinf(x) or std::isinf(y) or std::isinf(z)){
  instability(&_cadna_half_overflow_count);
  }
}


  return *this;
}
inline half_st& half_st::operator-=(const unsigned int& a){
  half_st initial=*this;
  unsigned int tmp2;
  unsigned int random;
  half tmp;

  accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp2=(random) ? -a : a;
  tmp = x;
  HALF_BIT_FLIP(tmp, random);
  x = tmp;
  x -= tmp2;
  HALF_BIT_FLIP(x, random);

  random = RANDOM;
  tmp2=(random) ? -a : a;
  tmp = y;
  HALF_BIT_FLIP(tmp, random);
  y = tmp;
  y -= tmp2;
  HALF_BIT_FLIP(y, random);

  random = 1^random;
  tmp2=(random) ? -a : a;
  tmp = z;
  HALF_BIT_FLIP(tmp, random);
  z = tmp;
  z -= tmp2;
  HALF_BIT_FLIP(z, random);

  
  
  if (_cadna_cancel_tag){
    if (initial.accuracy==DIGIT_NOT_COMPUTED || initial.accuracy==RELIABLE_RESULT)
      initial.nb_significant_digit();
    if (initial.accuracy >=  this->nb_significant_digit()+_cadna_cancel_value)
      instability(&_cadna_cancel_count);
  }
  if (_cadna_half_overflow_tag){
  if (std::isinf(x) or std::isinf(y) or std::isinf(z)){
  instability(&_cadna_half_overflow_count);
  }
}


  return *this;
}
inline half_st& half_st::operator-=(const int& a){
  half_st initial=*this;
  int tmp2;
  unsigned int random;
  half tmp;

  accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp2=(random) ? -a : a;
  tmp = x;
  HALF_BIT_FLIP(tmp, random);
  x = tmp;
  x -= tmp2;
  HALF_BIT_FLIP(x, random);

  random = RANDOM;
  tmp2=(random) ? -a : a;
  tmp = y;
  HALF_BIT_FLIP(tmp, random);
  y = tmp;
  y -= tmp2;
  HALF_BIT_FLIP(y, random);

  random = 1^random;
  tmp2=(random) ? -a : a;
  tmp = z;
  HALF_BIT_FLIP(tmp, random);
  z = tmp;
  z -= tmp2;
  HALF_BIT_FLIP(z, random);

  
  
  if (_cadna_cancel_tag){
    if (initial.accuracy==DIGIT_NOT_COMPUTED || initial.accuracy==RELIABLE_RESULT)
      initial.nb_significant_digit();
    if (initial.accuracy >=  this->nb_significant_digit()+_cadna_cancel_value)
      instability(&_cadna_cancel_count);
  }
  if (_cadna_half_overflow_tag){
  if (std::isinf(x) or std::isinf(y) or std::isinf(z)){
  instability(&_cadna_half_overflow_count);
  }
}


  return *this;
}
inline half_st& half_st::operator-=(const unsigned short& a){
  half_st initial=*this;
  unsigned short tmp2;
  unsigned int random;
  half tmp;

  accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp2=(random) ? -a : a;
  tmp = x;
  HALF_BIT_FLIP(tmp, random);
  x = tmp;
  x -= tmp2;
  HALF_BIT_FLIP(x, random);

  random = RANDOM;
  tmp2=(random) ? -a : a;
  tmp = y;
  HALF_BIT_FLIP(tmp, random);
  y = tmp;
  y -= tmp2;
  HALF_BIT_FLIP(y, random);

  random = 1^random;
  tmp2=(random) ? -a : a;
  tmp = z;
  HALF_BIT_FLIP(tmp, random);
  z = tmp;
  z -= tmp2;
  HALF_BIT_FLIP(z, random);

  
  
  if (_cadna_cancel_tag){
    if (initial.accuracy==DIGIT_NOT_COMPUTED || initial.accuracy==RELIABLE_RESULT)
      initial.nb_significant_digit();
    if (initial.accuracy >=  this->nb_significant_digit()+_cadna_cancel_value)
      instability(&_cadna_cancel_count);
  }
  if (_cadna_half_overflow_tag){
  if (std::isinf(x) or std::isinf(y) or std::isinf(z)){
  instability(&_cadna_half_overflow_count);
  }
}


  return *this;
}
inline half_st& half_st::operator-=(const short& a){
  half_st initial=*this;
  short tmp2;
  unsigned int random;
  half tmp;

  accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp2=(random) ? -a : a;
  tmp = x;
  HALF_BIT_FLIP(tmp, random);
  x = tmp;
  x -= tmp2;
  HALF_BIT_FLIP(x, random);

  random = RANDOM;
  tmp2=(random) ? -a : a;
  tmp = y;
  HALF_BIT_FLIP(tmp, random);
  y = tmp;
  y -= tmp2;
  HALF_BIT_FLIP(y, random);

  random = 1^random;
  tmp2=(random) ? -a : a;
  tmp = z;
  HALF_BIT_FLIP(tmp, random);
  z = tmp;
  z -= tmp2;
  HALF_BIT_FLIP(z, random);

  
  
  if (_cadna_cancel_tag){
    if (initial.accuracy==DIGIT_NOT_COMPUTED || initial.accuracy==RELIABLE_RESULT)
      initial.nb_significant_digit();
    if (initial.accuracy >=  this->nb_significant_digit()+_cadna_cancel_value)
      instability(&_cadna_cancel_count);
  }
  if (_cadna_half_overflow_tag){
  if (std::isinf(x) or std::isinf(y) or std::isinf(z)){
  instability(&_cadna_half_overflow_count);
  }
}


  return *this;
}
inline half_st& half_st::operator-=(const unsigned char& a){
  half_st initial=*this;
  unsigned char tmp2;
  unsigned int random;
  half tmp;

  accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp2=(random) ? -a : a;
  tmp = x;
  HALF_BIT_FLIP(tmp, random);
  x = tmp;
  x -= tmp2;
  HALF_BIT_FLIP(x, random);

  random = RANDOM;
  tmp2=(random) ? -a : a;
  tmp = y;
  HALF_BIT_FLIP(tmp, random);
  y = tmp;
  y -= tmp2;
  HALF_BIT_FLIP(y, random);

  random = 1^random;
  tmp2=(random) ? -a : a;
  tmp = z;
  HALF_BIT_FLIP(tmp, random);
  z = tmp;
  z -= tmp2;
  HALF_BIT_FLIP(z, random);

  
  
  if (_cadna_cancel_tag){
    if (initial.accuracy==DIGIT_NOT_COMPUTED || initial.accuracy==RELIABLE_RESULT)
      initial.nb_significant_digit();
    if (initial.accuracy >=  this->nb_significant_digit()+_cadna_cancel_value)
      instability(&_cadna_cancel_count);
  }
  if (_cadna_half_overflow_tag){
  if (std::isinf(x) or std::isinf(y) or std::isinf(z)){
  instability(&_cadna_half_overflow_count);
  }
}


  return *this;
}
inline half_st& half_st::operator-=(const char& a){
  half_st initial=*this;
  char tmp2;
  unsigned int random;
  half tmp;

  accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp2=(random) ? -a : a;
  tmp = x;
  HALF_BIT_FLIP(tmp, random);
  x = tmp;
  x -= tmp2;
  HALF_BIT_FLIP(x, random);

  random = RANDOM;
  tmp2=(random) ? -a : a;
  tmp = y;
  HALF_BIT_FLIP(tmp, random);
  y = tmp;
  y -= tmp2;
  HALF_BIT_FLIP(y, random);

  random = 1^random;
  tmp2=(random) ? -a : a;
  tmp = z;
  HALF_BIT_FLIP(tmp, random);
  z = tmp;
  z -= tmp2;
  HALF_BIT_FLIP(z, random);

  
  
  if (_cadna_cancel_tag){
    if (initial.accuracy==DIGIT_NOT_COMPUTED || initial.accuracy==RELIABLE_RESULT)
      initial.nb_significant_digit();
    if (initial.accuracy >=  this->nb_significant_digit()+_cadna_cancel_value)
      instability(&_cadna_cancel_count);
  }
  if (_cadna_half_overflow_tag){
  if (std::isinf(x) or std::isinf(y) or std::isinf(z)){
  instability(&_cadna_half_overflow_count);
  }
}


  return *this;
}
inline half_st& half_st::operator-=(const half& a){
  half_st initial=*this;
  half tmp2;
  unsigned int random;
  half tmp;

  accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp2=(random) ? -a : a;
  tmp = x;
  HALF_BIT_FLIP(tmp, random);
  x = tmp;
  x -= tmp2;
  HALF_BIT_FLIP(x, random);

  random = RANDOM;
  tmp2=(random) ? -a : a;
  tmp = y;
  HALF_BIT_FLIP(tmp, random);
  y = tmp;
  y -= tmp2;
  HALF_BIT_FLIP(y, random);

  random = 1^random;
  tmp2=(random) ? -a : a;
  tmp = z;
  HALF_BIT_FLIP(tmp, random);
  z = tmp;
  z -= tmp2;
  HALF_BIT_FLIP(z, random);

  
  
  if (_cadna_cancel_tag){
    if (initial.accuracy==DIGIT_NOT_COMPUTED || initial.accuracy==RELIABLE_RESULT)
      initial.nb_significant_digit();
    if (initial.accuracy >=  this->nb_significant_digit()+_cadna_cancel_value)
      instability(&_cadna_cancel_count);
  }
  if (_cadna_half_overflow_tag){
  if (std::isinf(x) or std::isinf(y) or std::isinf(z)){
  instability(&_cadna_half_overflow_count);
  }
}


  return *this;
}
#endif // CADNA_HALF_EMULATION

#ifdef CADNA_HALF_NATIVE
inline half_st& half_st::operator-=(const double& a){
  half_st initial=*this;

  unsigned int random;
  half tmp;

  accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = x;
  HALF_BIT_FLIP(tmp, random);
  x = tmp;
  tmp = a;
  HALF_BIT_FLIP(tmp, random);
  x -= tmp;
  HALF_BIT_FLIP(x, random);

  random = RANDOM;
  tmp = y;
  HALF_BIT_FLIP(tmp, random);
  y = tmp;
  tmp = a;
  HALF_BIT_FLIP(tmp, random);
  y -= tmp;
  HALF_BIT_FLIP(y, random);

  random = 1^random;
  tmp = z;
  HALF_BIT_FLIP(tmp, random);
  z = tmp;
  tmp = a;
  HALF_BIT_FLIP(tmp, random);
  z -= tmp;
  HALF_BIT_FLIP(z, random);

  if (_cadna_cancel_tag){
    if (initial.accuracy==DIGIT_NOT_COMPUTED || initial.accuracy==RELIABLE_RESULT)
      initial.nb_significant_digit();
    if (initial.accuracy >=  this->nb_significant_digit()+_cadna_cancel_value)
      instability(&_cadna_cancel_count);
  }
  
  return *this;
}
inline half_st& half_st::operator-=(const float& a){
  half_st initial=*this;

  unsigned int random;
  half tmp;

  accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = x;
  HALF_BIT_FLIP(tmp, random);
  x = tmp;
  tmp = a;
  HALF_BIT_FLIP(tmp, random);
  x -= tmp;
  HALF_BIT_FLIP(x, random);

  random = RANDOM;
  tmp = y;
  HALF_BIT_FLIP(tmp, random);
  y = tmp;
  tmp = a;
  HALF_BIT_FLIP(tmp, random);
  y -= tmp;
  HALF_BIT_FLIP(y, random);

  random = 1^random;
  tmp = z;
  HALF_BIT_FLIP(tmp, random);
  z = tmp;
  tmp = a;
  HALF_BIT_FLIP(tmp, random);
  z -= tmp;
  HALF_BIT_FLIP(z, random);

  if (_cadna_cancel_tag){
    if (initial.accuracy==DIGIT_NOT_COMPUTED || initial.accuracy==RELIABLE_RESULT)
      initial.nb_significant_digit();
    if (initial.accuracy >=  this->nb_significant_digit()+_cadna_cancel_value)
      instability(&_cadna_cancel_count);
  }
  
  return *this;
}
inline half_st& half_st::operator-=(const unsigned long long& a){
  half_st initial=*this;
  unsigned long long tmp2;
  unsigned int random;
  half tmp;

  accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp2=(random) ? -a : a;
  tmp = x;
  HALF_BIT_FLIP(tmp, random);
  x = tmp;
  x -= tmp2;
  HALF_BIT_FLIP(x, random);

  random = RANDOM;
  tmp2=(random) ? -a : a;
  tmp = y;
  HALF_BIT_FLIP(tmp, random);
  y = tmp;
  y -= tmp2;
  HALF_BIT_FLIP(y, random);

  random = 1^random;
  tmp2=(random) ? -a : a;
  tmp = z;
  HALF_BIT_FLIP(tmp, random);
  z = tmp;
  z -= tmp2;
  HALF_BIT_FLIP(z, random);

  
  
  if (_cadna_cancel_tag){
    if (initial.accuracy==DIGIT_NOT_COMPUTED || initial.accuracy==RELIABLE_RESULT)
      initial.nb_significant_digit();
    if (initial.accuracy >=  this->nb_significant_digit()+_cadna_cancel_value)
      instability(&_cadna_cancel_count);
  }
  if (_cadna_half_overflow_tag){
  if (std::isinf(x) or std::isinf(y) or std::isinf(z)){
  instability(&_cadna_half_overflow_count);
  }
}


  return *this;
}
inline half_st& half_st::operator-=(const long long& a){
  half_st initial=*this;
  long long tmp2;
  unsigned int random;
  half tmp;

  accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp2=(random) ? -a : a;
  tmp = x;
  HALF_BIT_FLIP(tmp, random);
  x = tmp;
  x -= tmp2;
  HALF_BIT_FLIP(x, random);

  random = RANDOM;
  tmp2=(random) ? -a : a;
  tmp = y;
  HALF_BIT_FLIP(tmp, random);
  y = tmp;
  y -= tmp2;
  HALF_BIT_FLIP(y, random);

  random = 1^random;
  tmp2=(random) ? -a : a;
  tmp = z;
  HALF_BIT_FLIP(tmp, random);
  z = tmp;
  z -= tmp2;
  HALF_BIT_FLIP(z, random);

  
  
  if (_cadna_cancel_tag){
    if (initial.accuracy==DIGIT_NOT_COMPUTED || initial.accuracy==RELIABLE_RESULT)
      initial.nb_significant_digit();
    if (initial.accuracy >=  this->nb_significant_digit()+_cadna_cancel_value)
      instability(&_cadna_cancel_count);
  }
  if (_cadna_half_overflow_tag){
  if (std::isinf(x) or std::isinf(y) or std::isinf(z)){
  instability(&_cadna_half_overflow_count);
  }
}


  return *this;
}
inline half_st& half_st::operator-=(const unsigned long& a){
  half_st initial=*this;
  unsigned long tmp2;
  unsigned int random;
  half tmp;

  accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp2=(random) ? -a : a;
  tmp = x;
  HALF_BIT_FLIP(tmp, random);
  x = tmp;
  x -= tmp2;
  HALF_BIT_FLIP(x, random);

  random = RANDOM;
  tmp2=(random) ? -a : a;
  tmp = y;
  HALF_BIT_FLIP(tmp, random);
  y = tmp;
  y -= tmp2;
  HALF_BIT_FLIP(y, random);

  random = 1^random;
  tmp2=(random) ? -a : a;
  tmp = z;
  HALF_BIT_FLIP(tmp, random);
  z = tmp;
  z -= tmp2;
  HALF_BIT_FLIP(z, random);

  
  
  if (_cadna_cancel_tag){
    if (initial.accuracy==DIGIT_NOT_COMPUTED || initial.accuracy==RELIABLE_RESULT)
      initial.nb_significant_digit();
    if (initial.accuracy >=  this->nb_significant_digit()+_cadna_cancel_value)
      instability(&_cadna_cancel_count);
  }
  if (_cadna_half_overflow_tag){
  if (std::isinf(x) or std::isinf(y) or std::isinf(z)){
  instability(&_cadna_half_overflow_count);
  }
}


  return *this;
}
inline half_st& half_st::operator-=(const long& a){
  half_st initial=*this;
  long tmp2;
  unsigned int random;
  half tmp;

  accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp2=(random) ? -a : a;
  tmp = x;
  HALF_BIT_FLIP(tmp, random);
  x = tmp;
  x -= tmp2;
  HALF_BIT_FLIP(x, random);

  random = RANDOM;
  tmp2=(random) ? -a : a;
  tmp = y;
  HALF_BIT_FLIP(tmp, random);
  y = tmp;
  y -= tmp2;
  HALF_BIT_FLIP(y, random);

  random = 1^random;
  tmp2=(random) ? -a : a;
  tmp = z;
  HALF_BIT_FLIP(tmp, random);
  z = tmp;
  z -= tmp2;
  HALF_BIT_FLIP(z, random);

  
  
  if (_cadna_cancel_tag){
    if (initial.accuracy==DIGIT_NOT_COMPUTED || initial.accuracy==RELIABLE_RESULT)
      initial.nb_significant_digit();
    if (initial.accuracy >=  this->nb_significant_digit()+_cadna_cancel_value)
      instability(&_cadna_cancel_count);
  }
  if (_cadna_half_overflow_tag){
  if (std::isinf(x) or std::isinf(y) or std::isinf(z)){
  instability(&_cadna_half_overflow_count);
  }
}


  return *this;
}
inline half_st& half_st::operator-=(const unsigned int& a){
  half_st initial=*this;
  unsigned int tmp2;
  unsigned int random;
  half tmp;

  accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp2=(random) ? -a : a;
  tmp = x;
  HALF_BIT_FLIP(tmp, random);
  x = tmp;
  x -= tmp2;
  HALF_BIT_FLIP(x, random);

  random = RANDOM;
  tmp2=(random) ? -a : a;
  tmp = y;
  HALF_BIT_FLIP(tmp, random);
  y = tmp;
  y -= tmp2;
  HALF_BIT_FLIP(y, random);

  random = 1^random;
  tmp2=(random) ? -a : a;
  tmp = z;
  HALF_BIT_FLIP(tmp, random);
  z = tmp;
  z -= tmp2;
  HALF_BIT_FLIP(z, random);

  
  
  if (_cadna_cancel_tag){
    if (initial.accuracy==DIGIT_NOT_COMPUTED || initial.accuracy==RELIABLE_RESULT)
      initial.nb_significant_digit();
    if (initial.accuracy >=  this->nb_significant_digit()+_cadna_cancel_value)
      instability(&_cadna_cancel_count);
  }
  if (_cadna_half_overflow_tag){
  if (std::isinf(x) or std::isinf(y) or std::isinf(z)){
  instability(&_cadna_half_overflow_count);
  }
}


  return *this;
}
inline half_st& half_st::operator-=(const int& a){
  half_st initial=*this;
  int tmp2;
  unsigned int random;
  half tmp;

  accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp2=(random) ? -a : a;
  tmp = x;
  HALF_BIT_FLIP(tmp, random);
  x = tmp;
  x -= tmp2;
  HALF_BIT_FLIP(x, random);

  random = RANDOM;
  tmp2=(random) ? -a : a;
  tmp = y;
  HALF_BIT_FLIP(tmp, random);
  y = tmp;
  y -= tmp2;
  HALF_BIT_FLIP(y, random);

  random = 1^random;
  tmp2=(random) ? -a : a;
  tmp = z;
  HALF_BIT_FLIP(tmp, random);
  z = tmp;
  z -= tmp2;
  HALF_BIT_FLIP(z, random);

  
  
  if (_cadna_cancel_tag){
    if (initial.accuracy==DIGIT_NOT_COMPUTED || initial.accuracy==RELIABLE_RESULT)
      initial.nb_significant_digit();
    if (initial.accuracy >=  this->nb_significant_digit()+_cadna_cancel_value)
      instability(&_cadna_cancel_count);
  }
  if (_cadna_half_overflow_tag){
  if (std::isinf(x) or std::isinf(y) or std::isinf(z)){
  instability(&_cadna_half_overflow_count);
  }
}


  return *this;
}
inline half_st& half_st::operator-=(const unsigned short& a){
  half_st initial=*this;
  unsigned short tmp2;
  unsigned int random;
  half tmp;

  accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp2=(random) ? -a : a;
  tmp = x;
  HALF_BIT_FLIP(tmp, random);
  x = tmp;
  x -= tmp2;
  HALF_BIT_FLIP(x, random);

  random = RANDOM;
  tmp2=(random) ? -a : a;
  tmp = y;
  HALF_BIT_FLIP(tmp, random);
  y = tmp;
  y -= tmp2;
  HALF_BIT_FLIP(y, random);

  random = 1^random;
  tmp2=(random) ? -a : a;
  tmp = z;
  HALF_BIT_FLIP(tmp, random);
  z = tmp;
  z -= tmp2;
  HALF_BIT_FLIP(z, random);

  
  
  if (_cadna_cancel_tag){
    if (initial.accuracy==DIGIT_NOT_COMPUTED || initial.accuracy==RELIABLE_RESULT)
      initial.nb_significant_digit();
    if (initial.accuracy >=  this->nb_significant_digit()+_cadna_cancel_value)
      instability(&_cadna_cancel_count);
  }
  if (_cadna_half_overflow_tag){
  if (std::isinf(x) or std::isinf(y) or std::isinf(z)){
  instability(&_cadna_half_overflow_count);
  }
}


  return *this;
}
inline half_st& half_st::operator-=(const short& a){
  half_st initial=*this;
  short tmp2;
  unsigned int random;
  half tmp;

  accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp2=(random) ? -a : a;
  tmp = x;
  HALF_BIT_FLIP(tmp, random);
  x = tmp;
  x -= tmp2;
  HALF_BIT_FLIP(x, random);

  random = RANDOM;
  tmp2=(random) ? -a : a;
  tmp = y;
  HALF_BIT_FLIP(tmp, random);
  y = tmp;
  y -= tmp2;
  HALF_BIT_FLIP(y, random);

  random = 1^random;
  tmp2=(random) ? -a : a;
  tmp = z;
  HALF_BIT_FLIP(tmp, random);
  z = tmp;
  z -= tmp2;
  HALF_BIT_FLIP(z, random);

  
  
  if (_cadna_cancel_tag){
    if (initial.accuracy==DIGIT_NOT_COMPUTED || initial.accuracy==RELIABLE_RESULT)
      initial.nb_significant_digit();
    if (initial.accuracy >=  this->nb_significant_digit()+_cadna_cancel_value)
      instability(&_cadna_cancel_count);
  }
  if (_cadna_half_overflow_tag){
  if (std::isinf(x) or std::isinf(y) or std::isinf(z)){
  instability(&_cadna_half_overflow_count);
  }
}


  return *this;
}
inline half_st& half_st::operator-=(const unsigned char& a){
  half_st initial=*this;
  unsigned char tmp2;
  unsigned int random;
  half tmp;

  accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp2=(random) ? -a : a;
  tmp = x;
  HALF_BIT_FLIP(tmp, random);
  x = tmp;
  x -= tmp2;
  HALF_BIT_FLIP(x, random);

  random = RANDOM;
  tmp2=(random) ? -a : a;
  tmp = y;
  HALF_BIT_FLIP(tmp, random);
  y = tmp;
  y -= tmp2;
  HALF_BIT_FLIP(y, random);

  random = 1^random;
  tmp2=(random) ? -a : a;
  tmp = z;
  HALF_BIT_FLIP(tmp, random);
  z = tmp;
  z -= tmp2;
  HALF_BIT_FLIP(z, random);

  
  
  if (_cadna_cancel_tag){
    if (initial.accuracy==DIGIT_NOT_COMPUTED || initial.accuracy==RELIABLE_RESULT)
      initial.nb_significant_digit();
    if (initial.accuracy >=  this->nb_significant_digit()+_cadna_cancel_value)
      instability(&_cadna_cancel_count);
  }
  if (_cadna_half_overflow_tag){
  if (std::isinf(x) or std::isinf(y) or std::isinf(z)){
  instability(&_cadna_half_overflow_count);
  }
}


  return *this;
}
inline half_st& half_st::operator-=(const char& a){
  half_st initial=*this;
  char tmp2;
  unsigned int random;
  half tmp;

  accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp2=(random) ? -a : a;
  tmp = x;
  HALF_BIT_FLIP(tmp, random);
  x = tmp;
  x -= tmp2;
  HALF_BIT_FLIP(x, random);

  random = RANDOM;
  tmp2=(random) ? -a : a;
  tmp = y;
  HALF_BIT_FLIP(tmp, random);
  y = tmp;
  y -= tmp2;
  HALF_BIT_FLIP(y, random);

  random = 1^random;
  tmp2=(random) ? -a : a;
  tmp = z;
  HALF_BIT_FLIP(tmp, random);
  z = tmp;
  z -= tmp2;
  HALF_BIT_FLIP(z, random);

  
  
  if (_cadna_cancel_tag){
    if (initial.accuracy==DIGIT_NOT_COMPUTED || initial.accuracy==RELIABLE_RESULT)
      initial.nb_significant_digit();
    if (initial.accuracy >=  this->nb_significant_digit()+_cadna_cancel_value)
      instability(&_cadna_cancel_count);
  }
  if (_cadna_half_overflow_tag){
  if (std::isinf(x) or std::isinf(y) or std::isinf(z)){
  instability(&_cadna_half_overflow_count);
  }
}


  return *this;
}
inline half_st& half_st::operator-=(const half& a){
  half_st initial=*this;
  half tmp2;
  unsigned int random;
  half tmp;

  accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp2=(random) ? -a : a;
  tmp = x;
  HALF_BIT_FLIP(tmp, random);
  x = tmp;
  x -= tmp2;
  HALF_BIT_FLIP(x, random);

  random = RANDOM;
  tmp2=(random) ? -a : a;
  tmp = y;
  HALF_BIT_FLIP(tmp, random);
  y = tmp;
  y -= tmp2;
  HALF_BIT_FLIP(y, random);

  random = 1^random;
  tmp2=(random) ? -a : a;
  tmp = z;
  HALF_BIT_FLIP(tmp, random);
  z = tmp;
  z -= tmp2;
  HALF_BIT_FLIP(z, random);

  
  
  if (_cadna_cancel_tag){
    if (initial.accuracy==DIGIT_NOT_COMPUTED || initial.accuracy==RELIABLE_RESULT)
      initial.nb_significant_digit();
    if (initial.accuracy >=  this->nb_significant_digit()+_cadna_cancel_value)
      instability(&_cadna_cancel_count);
  }
  if (_cadna_half_overflow_tag){
  if (std::isinf(x) or std::isinf(y) or std::isinf(z)){
  instability(&_cadna_half_overflow_count);
  }
}


  return *this;
}
#endif

/////////////////////////////////////////////////////////////////////////





inline double_st& double_st::operator-=(const double_st& a){
  double_st initial=*this;

  unsigned int random;
  double tmp;

  accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = x;
  DOUBLE_BIT_FLIP(tmp, random);
  x = tmp;
  tmp = a.x;
  DOUBLE_BIT_FLIP(tmp, random);
  x -= tmp;
  DOUBLE_BIT_FLIP(x, random);

  random = RANDOM;
  tmp = y;
  DOUBLE_BIT_FLIP(tmp, random);
  y = tmp;
  tmp = a.y;
  DOUBLE_BIT_FLIP(tmp, random);
  y -= tmp;
  DOUBLE_BIT_FLIP(y, random);

  random = 1^random;
  tmp = z;
  DOUBLE_BIT_FLIP(tmp, random);
  z = tmp;
  tmp = a.z;
  DOUBLE_BIT_FLIP(tmp, random);
  z -= tmp;
  DOUBLE_BIT_FLIP(z, random);

  if (_cadna_cancel_tag){
    if (initial.accuracy==DIGIT_NOT_COMPUTED || initial.accuracy==RELIABLE_RESULT) initial.nb_significant_digit();
    if (a.accuracy==DIGIT_NOT_COMPUTED || a.accuracy==RELIABLE_RESULT) a.nb_significant_digit();
    if ((initial.accuracy < a.accuracy ? initial.accuracy : a.accuracy) >=
         nb_significant_digit()+_cadna_cancel_value)
      instability(&_cadna_cancel_count);
  }
  return *this;
}
inline double_st& double_st::operator-=(const float_st& a){
  double_st initial=*this;

  unsigned int random;
  double tmp;

  accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = x;
  DOUBLE_BIT_FLIP(tmp, random);
  x = tmp;
  tmp = a.x;
  DOUBLE_BIT_FLIP(tmp, random);
  x -= tmp;
  DOUBLE_BIT_FLIP(x, random);

  random = RANDOM;
  tmp = y;
  DOUBLE_BIT_FLIP(tmp, random);
  y = tmp;
  tmp = a.y;
  DOUBLE_BIT_FLIP(tmp, random);
  y -= tmp;
  DOUBLE_BIT_FLIP(y, random);

  random = 1^random;
  tmp = z;
  DOUBLE_BIT_FLIP(tmp, random);
  z = tmp;
  tmp = a.z;
  DOUBLE_BIT_FLIP(tmp, random);
  z -= tmp;
  DOUBLE_BIT_FLIP(z, random);

  if (_cadna_cancel_tag){
    if (initial.accuracy==DIGIT_NOT_COMPUTED || initial.accuracy==RELIABLE_RESULT) initial.nb_significant_digit();
    if (a.accuracy==DIGIT_NOT_COMPUTED || a.accuracy==RELIABLE_RESULT) a.nb_significant_digit();
    if ((initial.accuracy < a.accuracy ? initial.accuracy : a.accuracy) >=
         nb_significant_digit()+_cadna_cancel_value)
      instability(&_cadna_cancel_count);
  }
  return *this;
}
inline float_st& float_st::operator-=(const double_st& a){
  float_st initial=*this;

  unsigned int random;
  float tmp;

  accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = x;
  FLOAT_BIT_FLIP(tmp, random);
  x = tmp;
  tmp = a.x;
  FLOAT_BIT_FLIP(tmp, random);
  x -= tmp;
  FLOAT_BIT_FLIP(x, random);

  random = RANDOM;
  tmp = y;
  FLOAT_BIT_FLIP(tmp, random);
  y = tmp;
  tmp = a.y;
  FLOAT_BIT_FLIP(tmp, random);
  y -= tmp;
  FLOAT_BIT_FLIP(y, random);

  random = 1^random;
  tmp = z;
  FLOAT_BIT_FLIP(tmp, random);
  z = tmp;
  tmp = a.z;
  FLOAT_BIT_FLIP(tmp, random);
  z -= tmp;
  FLOAT_BIT_FLIP(z, random);

  if (_cadna_cancel_tag){
    if (initial.accuracy==DIGIT_NOT_COMPUTED || initial.accuracy==RELIABLE_RESULT) initial.nb_significant_digit();
    if (a.accuracy==DIGIT_NOT_COMPUTED || a.accuracy==RELIABLE_RESULT) a.nb_significant_digit();
    if ((initial.accuracy < a.accuracy ? initial.accuracy : a.accuracy) >=
         nb_significant_digit()+_cadna_cancel_value)
      instability(&_cadna_cancel_count);
  }
  return *this;
}
inline float_st& float_st::operator-=(const float_st& a){
  float_st initial=*this;

  unsigned int random;
  float tmp;

  accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = x;
  FLOAT_BIT_FLIP(tmp, random);
  x = tmp;
  tmp = a.x;
  FLOAT_BIT_FLIP(tmp, random);
  x -= tmp;
  FLOAT_BIT_FLIP(x, random);

  random = RANDOM;
  tmp = y;
  FLOAT_BIT_FLIP(tmp, random);
  y = tmp;
  tmp = a.y;
  FLOAT_BIT_FLIP(tmp, random);
  y -= tmp;
  FLOAT_BIT_FLIP(y, random);

  random = 1^random;
  tmp = z;
  FLOAT_BIT_FLIP(tmp, random);
  z = tmp;
  tmp = a.z;
  FLOAT_BIT_FLIP(tmp, random);
  z -= tmp;
  FLOAT_BIT_FLIP(z, random);

  if (_cadna_cancel_tag){
    if (initial.accuracy==DIGIT_NOT_COMPUTED || initial.accuracy==RELIABLE_RESULT) initial.nb_significant_digit();
    if (a.accuracy==DIGIT_NOT_COMPUTED || a.accuracy==RELIABLE_RESULT) a.nb_significant_digit();
    if ((initial.accuracy < a.accuracy ? initial.accuracy : a.accuracy) >=
         nb_significant_digit()+_cadna_cancel_value)
      instability(&_cadna_cancel_count);
  }
  return *this;
}
#ifdef CADNA_HALF_EMULATION
inline half_st& half_st::operator-=(const half_st& a){
  half_st initial=*this;

  unsigned int random;
  half tmp;

  accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = x;
  HALF_BIT_FLIP(tmp, random);
  x = tmp;
  tmp = a.x;
  HALF_BIT_FLIP(tmp, random);
  x -= tmp;
  HALF_BIT_FLIP(x, random);

  random = RANDOM;
  tmp = y;
  HALF_BIT_FLIP(tmp, random);
  y = tmp;
  tmp = a.y;
  HALF_BIT_FLIP(tmp, random);
  y -= tmp;
  HALF_BIT_FLIP(y, random);

  random = 1^random;
  tmp = z;
  HALF_BIT_FLIP(tmp, random);
  z = tmp;
  tmp = a.z;
  HALF_BIT_FLIP(tmp, random);
  z -= tmp;
  HALF_BIT_FLIP(z, random);

  
  
  if (_cadna_cancel_tag){
    if (initial.accuracy==DIGIT_NOT_COMPUTED || initial.accuracy==RELIABLE_RESULT) initial.nb_significant_digit();
    if (a.accuracy==DIGIT_NOT_COMPUTED || a.accuracy==RELIABLE_RESULT) a.nb_significant_digit();
    if ((initial.accuracy < a.accuracy ? initial.accuracy : a.accuracy) >=
         nb_significant_digit()+_cadna_cancel_value)
      instability(&_cadna_cancel_count);
  }
  if (_cadna_half_overflow_tag){
  if (std::isinf(x) or std::isinf(y) or std::isinf(z)){
  instability(&_cadna_half_overflow_count);
  }
}



  return *this;
}
inline half_st& half_st::operator-=(const double_st& a){
  half_st initial=*this;

  unsigned int random;
  half tmp;

  accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = x;
  HALF_BIT_FLIP(tmp, random);
  x = tmp;
  tmp = a.x;
  HALF_BIT_FLIP(tmp, random);
  x -= tmp;
  HALF_BIT_FLIP(x, random);

  random = RANDOM;
  tmp = y;
  HALF_BIT_FLIP(tmp, random);
  y = tmp;
  tmp = a.y;
  HALF_BIT_FLIP(tmp, random);
  y -= tmp;
  HALF_BIT_FLIP(y, random);

  random = 1^random;
  tmp = z;
  HALF_BIT_FLIP(tmp, random);
  z = tmp;
  tmp = a.z;
  HALF_BIT_FLIP(tmp, random);
  z -= tmp;
  HALF_BIT_FLIP(z, random);

  
  
  if (_cadna_cancel_tag){
    if (initial.accuracy==DIGIT_NOT_COMPUTED || initial.accuracy==RELIABLE_RESULT) initial.nb_significant_digit();
    if (a.accuracy==DIGIT_NOT_COMPUTED || a.accuracy==RELIABLE_RESULT) a.nb_significant_digit();
    if ((initial.accuracy < a.accuracy ? initial.accuracy : a.accuracy) >=
         nb_significant_digit()+_cadna_cancel_value)
      instability(&_cadna_cancel_count);
  }
  if (_cadna_half_overflow_tag){
  if (std::isinf(x) or std::isinf(y) or std::isinf(z)){
  instability(&_cadna_half_overflow_count);
  }
}



  return *this;
}
inline double_st& double_st::operator-=(const half_st& a){
  double_st initial=*this;

  unsigned int random;
  double tmp;

  accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = x;
  DOUBLE_BIT_FLIP(tmp, random);
  x = tmp;
  tmp = a.x;
  DOUBLE_BIT_FLIP(tmp, random);
  x -= tmp;
  DOUBLE_BIT_FLIP(x, random);

  random = RANDOM;
  tmp = y;
  DOUBLE_BIT_FLIP(tmp, random);
  y = tmp;
  tmp = a.y;
  DOUBLE_BIT_FLIP(tmp, random);
  y -= tmp;
  DOUBLE_BIT_FLIP(y, random);

  random = 1^random;
  tmp = z;
  DOUBLE_BIT_FLIP(tmp, random);
  z = tmp;
  tmp = a.z;
  DOUBLE_BIT_FLIP(tmp, random);
  z -= tmp;
  DOUBLE_BIT_FLIP(z, random);

  if (_cadna_cancel_tag){
    if (initial.accuracy==DIGIT_NOT_COMPUTED || initial.accuracy==RELIABLE_RESULT) initial.nb_significant_digit();
    if (a.accuracy==DIGIT_NOT_COMPUTED || a.accuracy==RELIABLE_RESULT) a.nb_significant_digit();
    if ((initial.accuracy < a.accuracy ? initial.accuracy : a.accuracy) >=
         nb_significant_digit()+_cadna_cancel_value)
      instability(&_cadna_cancel_count);
  }
  return *this;
}
inline half_st& half_st::operator-=(const float_st& a){
  half_st initial=*this;

  unsigned int random;
  half tmp;

  accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = x;
  HALF_BIT_FLIP(tmp, random);
  x = tmp;
  tmp = a.x;
  HALF_BIT_FLIP(tmp, random);
  x -= tmp;
  HALF_BIT_FLIP(x, random);

  random = RANDOM;
  tmp = y;
  HALF_BIT_FLIP(tmp, random);
  y = tmp;
  tmp = a.y;
  HALF_BIT_FLIP(tmp, random);
  y -= tmp;
  HALF_BIT_FLIP(y, random);

  random = 1^random;
  tmp = z;
  HALF_BIT_FLIP(tmp, random);
  z = tmp;
  tmp = a.z;
  HALF_BIT_FLIP(tmp, random);
  z -= tmp;
  HALF_BIT_FLIP(z, random);

  
  
  if (_cadna_cancel_tag){
    if (initial.accuracy==DIGIT_NOT_COMPUTED || initial.accuracy==RELIABLE_RESULT) initial.nb_significant_digit();
    if (a.accuracy==DIGIT_NOT_COMPUTED || a.accuracy==RELIABLE_RESULT) a.nb_significant_digit();
    if ((initial.accuracy < a.accuracy ? initial.accuracy : a.accuracy) >=
         nb_significant_digit()+_cadna_cancel_value)
      instability(&_cadna_cancel_count);
  }
  if (_cadna_half_overflow_tag){
  if (std::isinf(x) or std::isinf(y) or std::isinf(z)){
  instability(&_cadna_half_overflow_count);
  }
}



  return *this;
}
inline float_st& float_st::operator-=(const half_st& a){
  float_st initial=*this;

  unsigned int random;
  float tmp;

  accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = x;
  FLOAT_BIT_FLIP(tmp, random);
  x = tmp;
  tmp = a.x;
  FLOAT_BIT_FLIP(tmp, random);
  x -= tmp;
  FLOAT_BIT_FLIP(x, random);

  random = RANDOM;
  tmp = y;
  FLOAT_BIT_FLIP(tmp, random);
  y = tmp;
  tmp = a.y;
  FLOAT_BIT_FLIP(tmp, random);
  y -= tmp;
  FLOAT_BIT_FLIP(y, random);

  random = 1^random;
  tmp = z;
  FLOAT_BIT_FLIP(tmp, random);
  z = tmp;
  tmp = a.z;
  FLOAT_BIT_FLIP(tmp, random);
  z -= tmp;
  FLOAT_BIT_FLIP(z, random);

  if (_cadna_cancel_tag){
    if (initial.accuracy==DIGIT_NOT_COMPUTED || initial.accuracy==RELIABLE_RESULT) initial.nb_significant_digit();
    if (a.accuracy==DIGIT_NOT_COMPUTED || a.accuracy==RELIABLE_RESULT) a.nb_significant_digit();
    if ((initial.accuracy < a.accuracy ? initial.accuracy : a.accuracy) >=
         nb_significant_digit()+_cadna_cancel_value)
      instability(&_cadna_cancel_count);
  }
  return *this;
}
#endif

#ifdef CADNA_HALF_NATIVE
inline half_st& half_st::operator-=(const half_st& a){
  half_st initial=*this;

  unsigned int random;
  half tmp;

  accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = x;
  HALF_BIT_FLIP(tmp, random);
  x = tmp;
  tmp = a.x;
  HALF_BIT_FLIP(tmp, random);
  x -= tmp;
  HALF_BIT_FLIP(x, random);

  random = RANDOM;
  tmp = y;
  HALF_BIT_FLIP(tmp, random);
  y = tmp;
  tmp = a.y;
  HALF_BIT_FLIP(tmp, random);
  y -= tmp;
  HALF_BIT_FLIP(y, random);

  random = 1^random;
  tmp = z;
  HALF_BIT_FLIP(tmp, random);
  z = tmp;
  tmp = a.z;
  HALF_BIT_FLIP(tmp, random);
  z -= tmp;
  HALF_BIT_FLIP(z, random);

  
  
  if (_cadna_cancel_tag){
    if (initial.accuracy==DIGIT_NOT_COMPUTED || initial.accuracy==RELIABLE_RESULT) initial.nb_significant_digit();
    if (a.accuracy==DIGIT_NOT_COMPUTED || a.accuracy==RELIABLE_RESULT) a.nb_significant_digit();
    if ((initial.accuracy < a.accuracy ? initial.accuracy : a.accuracy) >=
         nb_significant_digit()+_cadna_cancel_value)
      instability(&_cadna_cancel_count);
  }
  if (_cadna_half_overflow_tag){
  if (std::isinf(x) or std::isinf(y) or std::isinf(z)){
  instability(&_cadna_half_overflow_count);
  }
}



  return *this;
}
inline half_st& half_st::operator-=(const double_st& a){
  half_st initial=*this;

  unsigned int random;
  half tmp;

  accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = x;
  HALF_BIT_FLIP(tmp, random);
  x = tmp;
  tmp = a.x;
  HALF_BIT_FLIP(tmp, random);
  x -= tmp;
  HALF_BIT_FLIP(x, random);

  random = RANDOM;
  tmp = y;
  HALF_BIT_FLIP(tmp, random);
  y = tmp;
  tmp = a.y;
  HALF_BIT_FLIP(tmp, random);
  y -= tmp;
  HALF_BIT_FLIP(y, random);

  random = 1^random;
  tmp = z;
  HALF_BIT_FLIP(tmp, random);
  z = tmp;
  tmp = a.z;
  HALF_BIT_FLIP(tmp, random);
  z -= tmp;
  HALF_BIT_FLIP(z, random);

  
  
  if (_cadna_cancel_tag){
    if (initial.accuracy==DIGIT_NOT_COMPUTED || initial.accuracy==RELIABLE_RESULT) initial.nb_significant_digit();
    if (a.accuracy==DIGIT_NOT_COMPUTED || a.accuracy==RELIABLE_RESULT) a.nb_significant_digit();
    if ((initial.accuracy < a.accuracy ? initial.accuracy : a.accuracy) >=
         nb_significant_digit()+_cadna_cancel_value)
      instability(&_cadna_cancel_count);
  }
  if (_cadna_half_overflow_tag){
  if (std::isinf(x) or std::isinf(y) or std::isinf(z)){
  instability(&_cadna_half_overflow_count);
  }
}



  return *this;
}
inline double_st& double_st::operator-=(const half_st& a){
  double_st initial=*this;

  unsigned int random;
  double tmp;

  accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = x;
  DOUBLE_BIT_FLIP(tmp, random);
  x = tmp;
  tmp = a.x;
  DOUBLE_BIT_FLIP(tmp, random);
  x -= tmp;
  DOUBLE_BIT_FLIP(x, random);

  random = RANDOM;
  tmp = y;
  DOUBLE_BIT_FLIP(tmp, random);
  y = tmp;
  tmp = a.y;
  DOUBLE_BIT_FLIP(tmp, random);
  y -= tmp;
  DOUBLE_BIT_FLIP(y, random);

  random = 1^random;
  tmp = z;
  DOUBLE_BIT_FLIP(tmp, random);
  z = tmp;
  tmp = a.z;
  DOUBLE_BIT_FLIP(tmp, random);
  z -= tmp;
  DOUBLE_BIT_FLIP(z, random);

  if (_cadna_cancel_tag){
    if (initial.accuracy==DIGIT_NOT_COMPUTED || initial.accuracy==RELIABLE_RESULT) initial.nb_significant_digit();
    if (a.accuracy==DIGIT_NOT_COMPUTED || a.accuracy==RELIABLE_RESULT) a.nb_significant_digit();
    if ((initial.accuracy < a.accuracy ? initial.accuracy : a.accuracy) >=
         nb_significant_digit()+_cadna_cancel_value)
      instability(&_cadna_cancel_count);
  }
  return *this;
}
inline half_st& half_st::operator-=(const float_st& a){
  half_st initial=*this;

  unsigned int random;
  half tmp;

  accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = x;
  HALF_BIT_FLIP(tmp, random);
  x = tmp;
  tmp = a.x;
  HALF_BIT_FLIP(tmp, random);
  x -= tmp;
  HALF_BIT_FLIP(x, random);

  random = RANDOM;
  tmp = y;
  HALF_BIT_FLIP(tmp, random);
  y = tmp;
  tmp = a.y;
  HALF_BIT_FLIP(tmp, random);
  y -= tmp;
  HALF_BIT_FLIP(y, random);

  random = 1^random;
  tmp = z;
  HALF_BIT_FLIP(tmp, random);
  z = tmp;
  tmp = a.z;
  HALF_BIT_FLIP(tmp, random);
  z -= tmp;
  HALF_BIT_FLIP(z, random);

  
  
  if (_cadna_cancel_tag){
    if (initial.accuracy==DIGIT_NOT_COMPUTED || initial.accuracy==RELIABLE_RESULT) initial.nb_significant_digit();
    if (a.accuracy==DIGIT_NOT_COMPUTED || a.accuracy==RELIABLE_RESULT) a.nb_significant_digit();
    if ((initial.accuracy < a.accuracy ? initial.accuracy : a.accuracy) >=
         nb_significant_digit()+_cadna_cancel_value)
      instability(&_cadna_cancel_count);
  }
  if (_cadna_half_overflow_tag){
  if (std::isinf(x) or std::isinf(y) or std::isinf(z)){
  instability(&_cadna_half_overflow_count);
  }
}



  return *this;
}
inline float_st& float_st::operator-=(const half_st& a){
  float_st initial=*this;

  unsigned int random;
  float tmp;

  accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = x;
  FLOAT_BIT_FLIP(tmp, random);
  x = tmp;
  tmp = a.x;
  FLOAT_BIT_FLIP(tmp, random);
  x -= tmp;
  FLOAT_BIT_FLIP(x, random);

  random = RANDOM;
  tmp = y;
  FLOAT_BIT_FLIP(tmp, random);
  y = tmp;
  tmp = a.y;
  FLOAT_BIT_FLIP(tmp, random);
  y -= tmp;
  FLOAT_BIT_FLIP(y, random);

  random = 1^random;
  tmp = z;
  FLOAT_BIT_FLIP(tmp, random);
  z = tmp;
  tmp = a.z;
  FLOAT_BIT_FLIP(tmp, random);
  z -= tmp;
  FLOAT_BIT_FLIP(z, random);

  if (_cadna_cancel_tag){
    if (initial.accuracy==DIGIT_NOT_COMPUTED || initial.accuracy==RELIABLE_RESULT) initial.nb_significant_digit();
    if (a.accuracy==DIGIT_NOT_COMPUTED || a.accuracy==RELIABLE_RESULT) a.nb_significant_digit();
    if ((initial.accuracy < a.accuracy ? initial.accuracy : a.accuracy) >=
         nb_significant_digit()+_cadna_cancel_value)
      instability(&_cadna_cancel_count);
  }
  return *this;
}
#endif
/////////////////////////////////////////////////////////////////////////




inline double_st operator-(const double& a, const double_st& b){
  double_st res;

  unsigned int random;
  double tmp;

  res.accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = a;
  DOUBLE_BIT_FLIP(tmp, random);
  res.x = tmp;
  tmp = b.x;
  DOUBLE_BIT_FLIP(tmp, random);
  res.x -= tmp;
  DOUBLE_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp = a;
  DOUBLE_BIT_FLIP(tmp, random);
  res.y = tmp;
  tmp = b.y;
  DOUBLE_BIT_FLIP(tmp, random);
  res.y -= tmp;
  DOUBLE_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp = a;
  DOUBLE_BIT_FLIP(tmp, random);
  res.z = tmp;
  tmp = b.z;
  DOUBLE_BIT_FLIP(tmp, random);
  res.z -= tmp;
  DOUBLE_BIT_FLIP(res.z, random);

  if (_cadna_cancel_tag){
    if (b.accuracy==DIGIT_NOT_COMPUTED || b.accuracy==RELIABLE_RESULT)
	b.nb_significant_digit();
    if (b.accuracy >= 	res.nb_significant_digit()+_cadna_cancel_value)
	instability(&_cadna_cancel_count);
  }
  return res;
}
inline double_st operator-(const float& a, const double_st& b){
  double_st res;

  unsigned int random;
  double tmp;

  res.accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = a;
  DOUBLE_BIT_FLIP(tmp, random);
  res.x = tmp;
  tmp = b.x;
  DOUBLE_BIT_FLIP(tmp, random);
  res.x -= tmp;
  DOUBLE_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp = a;
  DOUBLE_BIT_FLIP(tmp, random);
  res.y = tmp;
  tmp = b.y;
  DOUBLE_BIT_FLIP(tmp, random);
  res.y -= tmp;
  DOUBLE_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp = a;
  DOUBLE_BIT_FLIP(tmp, random);
  res.z = tmp;
  tmp = b.z;
  DOUBLE_BIT_FLIP(tmp, random);
  res.z -= tmp;
  DOUBLE_BIT_FLIP(res.z, random);

  if (_cadna_cancel_tag){
    if (b.accuracy==DIGIT_NOT_COMPUTED || b.accuracy==RELIABLE_RESULT)
	b.nb_significant_digit();
    if (b.accuracy >= 	res.nb_significant_digit()+_cadna_cancel_value)
	instability(&_cadna_cancel_count);
  }
  return res;
}
inline double_st operator-(const unsigned long long& a, const double_st& b){
  double_st res;

  unsigned int random;
  double tmp;

  res.accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = a;
  DOUBLE_BIT_FLIP(tmp, random);
  res.x = tmp;
  tmp = b.x;
  DOUBLE_BIT_FLIP(tmp, random);
  res.x -= tmp;
  DOUBLE_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp = a;
  DOUBLE_BIT_FLIP(tmp, random);
  res.y = tmp;
  tmp = b.y;
  DOUBLE_BIT_FLIP(tmp, random);
  res.y -= tmp;
  DOUBLE_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp = a;
  DOUBLE_BIT_FLIP(tmp, random);
  res.z = tmp;
  tmp = b.z;
  DOUBLE_BIT_FLIP(tmp, random);
  res.z -= tmp;
  DOUBLE_BIT_FLIP(res.z, random);

  if (_cadna_cancel_tag){
    if (b.accuracy==DIGIT_NOT_COMPUTED || b.accuracy==RELIABLE_RESULT)
	b.nb_significant_digit();
    if (b.accuracy >= 	res.nb_significant_digit()+_cadna_cancel_value)
	instability(&_cadna_cancel_count);
  }
  return res;
}
inline double_st operator-(const long long& a, const double_st& b){
  double_st res;

  unsigned int random;
  double tmp;

  res.accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = a;
  DOUBLE_BIT_FLIP(tmp, random);
  res.x = tmp;
  tmp = b.x;
  DOUBLE_BIT_FLIP(tmp, random);
  res.x -= tmp;
  DOUBLE_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp = a;
  DOUBLE_BIT_FLIP(tmp, random);
  res.y = tmp;
  tmp = b.y;
  DOUBLE_BIT_FLIP(tmp, random);
  res.y -= tmp;
  DOUBLE_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp = a;
  DOUBLE_BIT_FLIP(tmp, random);
  res.z = tmp;
  tmp = b.z;
  DOUBLE_BIT_FLIP(tmp, random);
  res.z -= tmp;
  DOUBLE_BIT_FLIP(res.z, random);

  if (_cadna_cancel_tag){
    if (b.accuracy==DIGIT_NOT_COMPUTED || b.accuracy==RELIABLE_RESULT)
	b.nb_significant_digit();
    if (b.accuracy >= 	res.nb_significant_digit()+_cadna_cancel_value)
	instability(&_cadna_cancel_count);
  }
  return res;
}
inline double_st operator-(const unsigned long& a, const double_st& b){
  double_st res;

  unsigned int random;
  double tmp;

  res.accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = a;
  DOUBLE_BIT_FLIP(tmp, random);
  res.x = tmp;
  tmp = b.x;
  DOUBLE_BIT_FLIP(tmp, random);
  res.x -= tmp;
  DOUBLE_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp = a;
  DOUBLE_BIT_FLIP(tmp, random);
  res.y = tmp;
  tmp = b.y;
  DOUBLE_BIT_FLIP(tmp, random);
  res.y -= tmp;
  DOUBLE_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp = a;
  DOUBLE_BIT_FLIP(tmp, random);
  res.z = tmp;
  tmp = b.z;
  DOUBLE_BIT_FLIP(tmp, random);
  res.z -= tmp;
  DOUBLE_BIT_FLIP(res.z, random);

  if (_cadna_cancel_tag){
    if (b.accuracy==DIGIT_NOT_COMPUTED || b.accuracy==RELIABLE_RESULT)
	b.nb_significant_digit();
    if (b.accuracy >= 	res.nb_significant_digit()+_cadna_cancel_value)
	instability(&_cadna_cancel_count);
  }
  return res;
}
inline double_st operator-(const long& a, const double_st& b){
  double_st res;

  unsigned int random;
  double tmp;

  res.accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = a;
  DOUBLE_BIT_FLIP(tmp, random);
  res.x = tmp;
  tmp = b.x;
  DOUBLE_BIT_FLIP(tmp, random);
  res.x -= tmp;
  DOUBLE_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp = a;
  DOUBLE_BIT_FLIP(tmp, random);
  res.y = tmp;
  tmp = b.y;
  DOUBLE_BIT_FLIP(tmp, random);
  res.y -= tmp;
  DOUBLE_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp = a;
  DOUBLE_BIT_FLIP(tmp, random);
  res.z = tmp;
  tmp = b.z;
  DOUBLE_BIT_FLIP(tmp, random);
  res.z -= tmp;
  DOUBLE_BIT_FLIP(res.z, random);

  if (_cadna_cancel_tag){
    if (b.accuracy==DIGIT_NOT_COMPUTED || b.accuracy==RELIABLE_RESULT)
	b.nb_significant_digit();
    if (b.accuracy >= 	res.nb_significant_digit()+_cadna_cancel_value)
	instability(&_cadna_cancel_count);
  }
  return res;
}
inline double_st operator-(const unsigned int & a, const double_st& b){
  double_st res;

  unsigned int random;
  double tmp;

  res.accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = a;
  DOUBLE_BIT_FLIP(tmp, random);
  res.x = tmp;
  tmp = b.x;
  DOUBLE_BIT_FLIP(tmp, random);
  res.x -= tmp;
  DOUBLE_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp = a;
  DOUBLE_BIT_FLIP(tmp, random);
  res.y = tmp;
  tmp = b.y;
  DOUBLE_BIT_FLIP(tmp, random);
  res.y -= tmp;
  DOUBLE_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp = a;
  DOUBLE_BIT_FLIP(tmp, random);
  res.z = tmp;
  tmp = b.z;
  DOUBLE_BIT_FLIP(tmp, random);
  res.z -= tmp;
  DOUBLE_BIT_FLIP(res.z, random);

  if (_cadna_cancel_tag){
    if (b.accuracy==DIGIT_NOT_COMPUTED || b.accuracy==RELIABLE_RESULT)
	b.nb_significant_digit();
    if (b.accuracy >= 	res.nb_significant_digit()+_cadna_cancel_value)
	instability(&_cadna_cancel_count);
  }
  return res;
}
inline double_st operator-(const int & a, const double_st& b){
  double_st res;

  unsigned int random;
  double tmp;

  res.accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = a;
  DOUBLE_BIT_FLIP(tmp, random);
  res.x = tmp;
  tmp = b.x;
  DOUBLE_BIT_FLIP(tmp, random);
  res.x -= tmp;
  DOUBLE_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp = a;
  DOUBLE_BIT_FLIP(tmp, random);
  res.y = tmp;
  tmp = b.y;
  DOUBLE_BIT_FLIP(tmp, random);
  res.y -= tmp;
  DOUBLE_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp = a;
  DOUBLE_BIT_FLIP(tmp, random);
  res.z = tmp;
  tmp = b.z;
  DOUBLE_BIT_FLIP(tmp, random);
  res.z -= tmp;
  DOUBLE_BIT_FLIP(res.z, random);

  if (_cadna_cancel_tag){
    if (b.accuracy==DIGIT_NOT_COMPUTED || b.accuracy==RELIABLE_RESULT)
	b.nb_significant_digit();
    if (b.accuracy >= 	res.nb_significant_digit()+_cadna_cancel_value)
	instability(&_cadna_cancel_count);
  }
  return res;
}
inline double_st operator-(const unsigned short& a, const double_st& b){
  double_st res;

  unsigned int random;
  double tmp;

  res.accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = a;
  DOUBLE_BIT_FLIP(tmp, random);
  res.x = tmp;
  tmp = b.x;
  DOUBLE_BIT_FLIP(tmp, random);
  res.x -= tmp;
  DOUBLE_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp = a;
  DOUBLE_BIT_FLIP(tmp, random);
  res.y = tmp;
  tmp = b.y;
  DOUBLE_BIT_FLIP(tmp, random);
  res.y -= tmp;
  DOUBLE_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp = a;
  DOUBLE_BIT_FLIP(tmp, random);
  res.z = tmp;
  tmp = b.z;
  DOUBLE_BIT_FLIP(tmp, random);
  res.z -= tmp;
  DOUBLE_BIT_FLIP(res.z, random);

  if (_cadna_cancel_tag){
    if (b.accuracy==DIGIT_NOT_COMPUTED || b.accuracy==RELIABLE_RESULT)
	b.nb_significant_digit();
    if (b.accuracy >= 	res.nb_significant_digit()+_cadna_cancel_value)
	instability(&_cadna_cancel_count);
  }
  return res;
}
inline double_st operator-(const short& a, const double_st& b){
  double_st res;

  unsigned int random;
  double tmp;

  res.accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = a;
  DOUBLE_BIT_FLIP(tmp, random);
  res.x = tmp;
  tmp = b.x;
  DOUBLE_BIT_FLIP(tmp, random);
  res.x -= tmp;
  DOUBLE_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp = a;
  DOUBLE_BIT_FLIP(tmp, random);
  res.y = tmp;
  tmp = b.y;
  DOUBLE_BIT_FLIP(tmp, random);
  res.y -= tmp;
  DOUBLE_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp = a;
  DOUBLE_BIT_FLIP(tmp, random);
  res.z = tmp;
  tmp = b.z;
  DOUBLE_BIT_FLIP(tmp, random);
  res.z -= tmp;
  DOUBLE_BIT_FLIP(res.z, random);

  if (_cadna_cancel_tag){
    if (b.accuracy==DIGIT_NOT_COMPUTED || b.accuracy==RELIABLE_RESULT)
	b.nb_significant_digit();
    if (b.accuracy >= 	res.nb_significant_digit()+_cadna_cancel_value)
	instability(&_cadna_cancel_count);
  }
  return res;
}
inline double_st operator-(const unsigned char& a, const double_st& b){
  double_st res;

  unsigned int random;
  double tmp;

  res.accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = a;
  DOUBLE_BIT_FLIP(tmp, random);
  res.x = tmp;
  tmp = b.x;
  DOUBLE_BIT_FLIP(tmp, random);
  res.x -= tmp;
  DOUBLE_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp = a;
  DOUBLE_BIT_FLIP(tmp, random);
  res.y = tmp;
  tmp = b.y;
  DOUBLE_BIT_FLIP(tmp, random);
  res.y -= tmp;
  DOUBLE_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp = a;
  DOUBLE_BIT_FLIP(tmp, random);
  res.z = tmp;
  tmp = b.z;
  DOUBLE_BIT_FLIP(tmp, random);
  res.z -= tmp;
  DOUBLE_BIT_FLIP(res.z, random);

  if (_cadna_cancel_tag){
    if (b.accuracy==DIGIT_NOT_COMPUTED || b.accuracy==RELIABLE_RESULT)
	b.nb_significant_digit();
    if (b.accuracy >= 	res.nb_significant_digit()+_cadna_cancel_value)
	instability(&_cadna_cancel_count);
  }
  return res;
}
inline double_st operator-(const char& a, const double_st& b){
  double_st res;

  unsigned int random;
  double tmp;

  res.accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = a;
  DOUBLE_BIT_FLIP(tmp, random);
  res.x = tmp;
  tmp = b.x;
  DOUBLE_BIT_FLIP(tmp, random);
  res.x -= tmp;
  DOUBLE_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp = a;
  DOUBLE_BIT_FLIP(tmp, random);
  res.y = tmp;
  tmp = b.y;
  DOUBLE_BIT_FLIP(tmp, random);
  res.y -= tmp;
  DOUBLE_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp = a;
  DOUBLE_BIT_FLIP(tmp, random);
  res.z = tmp;
  tmp = b.z;
  DOUBLE_BIT_FLIP(tmp, random);
  res.z -= tmp;
  DOUBLE_BIT_FLIP(res.z, random);

  if (_cadna_cancel_tag){
    if (b.accuracy==DIGIT_NOT_COMPUTED || b.accuracy==RELIABLE_RESULT)
	b.nb_significant_digit();
    if (b.accuracy >= 	res.nb_significant_digit()+_cadna_cancel_value)
	instability(&_cadna_cancel_count);
  }
  return res;
}
#ifdef CADNA_HALF_EMULATION
inline double_st operator-(const half& a, const double_st& b){
  double_st res;

  unsigned int random;
  double tmp;

  res.accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = a;
  DOUBLE_BIT_FLIP(tmp, random);
  res.x = tmp;
  tmp = b.x;
  DOUBLE_BIT_FLIP(tmp, random);
  res.x -= tmp;
  DOUBLE_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp = a;
  DOUBLE_BIT_FLIP(tmp, random);
  res.y = tmp;
  tmp = b.y;
  DOUBLE_BIT_FLIP(tmp, random);
  res.y -= tmp;
  DOUBLE_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp = a;
  DOUBLE_BIT_FLIP(tmp, random);
  res.z = tmp;
  tmp = b.z;
  DOUBLE_BIT_FLIP(tmp, random);
  res.z -= tmp;
  DOUBLE_BIT_FLIP(res.z, random);

  if (_cadna_cancel_tag){
    if (b.accuracy==DIGIT_NOT_COMPUTED || b.accuracy==RELIABLE_RESULT)
	b.nb_significant_digit();
    if (b.accuracy >= 	res.nb_significant_digit()+_cadna_cancel_value)
	instability(&_cadna_cancel_count);
  }
  return res;
}
#endif // CADNA_HALF_EMULATION

#ifdef CADNA_HALF_NATIVE
inline double_st operator-(const half& a, const double_st& b){
  double_st res;

  unsigned int random;
  double tmp;

  res.accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = a;
  DOUBLE_BIT_FLIP(tmp, random);
  res.x = tmp;
  tmp = b.x;
  DOUBLE_BIT_FLIP(tmp, random);
  res.x -= tmp;
  DOUBLE_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp = a;
  DOUBLE_BIT_FLIP(tmp, random);
  res.y = tmp;
  tmp = b.y;
  DOUBLE_BIT_FLIP(tmp, random);
  res.y -= tmp;
  DOUBLE_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp = a;
  DOUBLE_BIT_FLIP(tmp, random);
  res.z = tmp;
  tmp = b.z;
  DOUBLE_BIT_FLIP(tmp, random);
  res.z -= tmp;
  DOUBLE_BIT_FLIP(res.z, random);

  if (_cadna_cancel_tag){
    if (b.accuracy==DIGIT_NOT_COMPUTED || b.accuracy==RELIABLE_RESULT)
	b.nb_significant_digit();
    if (b.accuracy >= 	res.nb_significant_digit()+_cadna_cancel_value)
	instability(&_cadna_cancel_count);
  }
  return res;
}
#endif

inline float_st operator-(const double& a, const float_st& b){
  float_st res;

  unsigned int random;
  float tmp;

  res.accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = a;
  FLOAT_BIT_FLIP(tmp, random);
  res.x = tmp;
  tmp = b.x;
  FLOAT_BIT_FLIP(tmp, random);
  res.x -= tmp;
  FLOAT_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp = a;
  FLOAT_BIT_FLIP(tmp, random);
  res.y = tmp;
  tmp = b.y;
  FLOAT_BIT_FLIP(tmp, random);
  res.y -= tmp;
  FLOAT_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp = a;
  FLOAT_BIT_FLIP(tmp, random);
  res.z = tmp;
  tmp = b.z;
  FLOAT_BIT_FLIP(tmp, random);
  res.z -= tmp;
  FLOAT_BIT_FLIP(res.z, random);

  if (_cadna_cancel_tag){
    if (b.accuracy==DIGIT_NOT_COMPUTED || b.accuracy==RELIABLE_RESULT)
	b.nb_significant_digit();
    if (b.accuracy >= 	res.nb_significant_digit()+_cadna_cancel_value)
	instability(&_cadna_cancel_count);
  }
  return res;
}
inline float_st operator-(const float& a, const float_st& b){
  float_st res;

  unsigned int random;
  float tmp;

  res.accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = a;
  FLOAT_BIT_FLIP(tmp, random);
  res.x = tmp;
  tmp = b.x;
  FLOAT_BIT_FLIP(tmp, random);
  res.x -= tmp;
  FLOAT_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp = a;
  FLOAT_BIT_FLIP(tmp, random);
  res.y = tmp;
  tmp = b.y;
  FLOAT_BIT_FLIP(tmp, random);
  res.y -= tmp;
  FLOAT_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp = a;
  FLOAT_BIT_FLIP(tmp, random);
  res.z = tmp;
  tmp = b.z;
  FLOAT_BIT_FLIP(tmp, random);
  res.z -= tmp;
  FLOAT_BIT_FLIP(res.z, random);

  if (_cadna_cancel_tag){
    if (b.accuracy==DIGIT_NOT_COMPUTED || b.accuracy==RELIABLE_RESULT)
	b.nb_significant_digit();
    if (b.accuracy >= 	res.nb_significant_digit()+_cadna_cancel_value)
	instability(&_cadna_cancel_count);
  }
  return res;
}
inline float_st operator-(const unsigned long long& a, const float_st& b){
  float_st res;

  unsigned int random;
  float tmp;

  res.accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = a;
  FLOAT_BIT_FLIP(tmp, random);
  res.x = tmp;
  tmp = b.x;
  FLOAT_BIT_FLIP(tmp, random);
  res.x -= tmp;
  FLOAT_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp = a;
  FLOAT_BIT_FLIP(tmp, random);
  res.y = tmp;
  tmp = b.y;
  FLOAT_BIT_FLIP(tmp, random);
  res.y -= tmp;
  FLOAT_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp = a;
  FLOAT_BIT_FLIP(tmp, random);
  res.z = tmp;
  tmp = b.z;
  FLOAT_BIT_FLIP(tmp, random);
  res.z -= tmp;
  FLOAT_BIT_FLIP(res.z, random);

  if (_cadna_cancel_tag){
    if (b.accuracy==DIGIT_NOT_COMPUTED || b.accuracy==RELIABLE_RESULT)
	b.nb_significant_digit();
    if (b.accuracy >= 	res.nb_significant_digit()+_cadna_cancel_value)
	instability(&_cadna_cancel_count);
  }
  return res;
}
inline float_st operator-(const long long& a, const float_st& b){
  float_st res;

  unsigned int random;
  float tmp;

  res.accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = a;
  FLOAT_BIT_FLIP(tmp, random);
  res.x = tmp;
  tmp = b.x;
  FLOAT_BIT_FLIP(tmp, random);
  res.x -= tmp;
  FLOAT_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp = a;
  FLOAT_BIT_FLIP(tmp, random);
  res.y = tmp;
  tmp = b.y;
  FLOAT_BIT_FLIP(tmp, random);
  res.y -= tmp;
  FLOAT_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp = a;
  FLOAT_BIT_FLIP(tmp, random);
  res.z = tmp;
  tmp = b.z;
  FLOAT_BIT_FLIP(tmp, random);
  res.z -= tmp;
  FLOAT_BIT_FLIP(res.z, random);

  if (_cadna_cancel_tag){
    if (b.accuracy==DIGIT_NOT_COMPUTED || b.accuracy==RELIABLE_RESULT)
	b.nb_significant_digit();
    if (b.accuracy >= 	res.nb_significant_digit()+_cadna_cancel_value)
	instability(&_cadna_cancel_count);
  }
  return res;
}
inline float_st operator-(const unsigned long& a, const float_st& b){
  float_st res;

  unsigned int random;
  float tmp;

  res.accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = a;
  FLOAT_BIT_FLIP(tmp, random);
  res.x = tmp;
  tmp = b.x;
  FLOAT_BIT_FLIP(tmp, random);
  res.x -= tmp;
  FLOAT_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp = a;
  FLOAT_BIT_FLIP(tmp, random);
  res.y = tmp;
  tmp = b.y;
  FLOAT_BIT_FLIP(tmp, random);
  res.y -= tmp;
  FLOAT_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp = a;
  FLOAT_BIT_FLIP(tmp, random);
  res.z = tmp;
  tmp = b.z;
  FLOAT_BIT_FLIP(tmp, random);
  res.z -= tmp;
  FLOAT_BIT_FLIP(res.z, random);

  if (_cadna_cancel_tag){
    if (b.accuracy==DIGIT_NOT_COMPUTED || b.accuracy==RELIABLE_RESULT)
	b.nb_significant_digit();
    if (b.accuracy >= 	res.nb_significant_digit()+_cadna_cancel_value)
	instability(&_cadna_cancel_count);
  }
  return res;
}
inline float_st operator-(const long& a, const float_st& b){
  float_st res;

  unsigned int random;
  float tmp;

  res.accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = a;
  FLOAT_BIT_FLIP(tmp, random);
  res.x = tmp;
  tmp = b.x;
  FLOAT_BIT_FLIP(tmp, random);
  res.x -= tmp;
  FLOAT_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp = a;
  FLOAT_BIT_FLIP(tmp, random);
  res.y = tmp;
  tmp = b.y;
  FLOAT_BIT_FLIP(tmp, random);
  res.y -= tmp;
  FLOAT_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp = a;
  FLOAT_BIT_FLIP(tmp, random);
  res.z = tmp;
  tmp = b.z;
  FLOAT_BIT_FLIP(tmp, random);
  res.z -= tmp;
  FLOAT_BIT_FLIP(res.z, random);

  if (_cadna_cancel_tag){
    if (b.accuracy==DIGIT_NOT_COMPUTED || b.accuracy==RELIABLE_RESULT)
	b.nb_significant_digit();
    if (b.accuracy >= 	res.nb_significant_digit()+_cadna_cancel_value)
	instability(&_cadna_cancel_count);
  }
  return res;
}
inline float_st operator-(const unsigned int & a, const float_st& b){
  float_st res;

  unsigned int random;
  float tmp;

  res.accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = a;
  FLOAT_BIT_FLIP(tmp, random);
  res.x = tmp;
  tmp = b.x;
  FLOAT_BIT_FLIP(tmp, random);
  res.x -= tmp;
  FLOAT_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp = a;
  FLOAT_BIT_FLIP(tmp, random);
  res.y = tmp;
  tmp = b.y;
  FLOAT_BIT_FLIP(tmp, random);
  res.y -= tmp;
  FLOAT_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp = a;
  FLOAT_BIT_FLIP(tmp, random);
  res.z = tmp;
  tmp = b.z;
  FLOAT_BIT_FLIP(tmp, random);
  res.z -= tmp;
  FLOAT_BIT_FLIP(res.z, random);

  if (_cadna_cancel_tag){
    if (b.accuracy==DIGIT_NOT_COMPUTED || b.accuracy==RELIABLE_RESULT)
	b.nb_significant_digit();
    if (b.accuracy >= 	res.nb_significant_digit()+_cadna_cancel_value)
	instability(&_cadna_cancel_count);
  }
  return res;
}
inline float_st operator-(const int & a, const float_st& b){
  float_st res;

  unsigned int random;
  float tmp;

  res.accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = a;
  FLOAT_BIT_FLIP(tmp, random);
  res.x = tmp;
  tmp = b.x;
  FLOAT_BIT_FLIP(tmp, random);
  res.x -= tmp;
  FLOAT_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp = a;
  FLOAT_BIT_FLIP(tmp, random);
  res.y = tmp;
  tmp = b.y;
  FLOAT_BIT_FLIP(tmp, random);
  res.y -= tmp;
  FLOAT_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp = a;
  FLOAT_BIT_FLIP(tmp, random);
  res.z = tmp;
  tmp = b.z;
  FLOAT_BIT_FLIP(tmp, random);
  res.z -= tmp;
  FLOAT_BIT_FLIP(res.z, random);

  if (_cadna_cancel_tag){
    if (b.accuracy==DIGIT_NOT_COMPUTED || b.accuracy==RELIABLE_RESULT)
	b.nb_significant_digit();
    if (b.accuracy >= 	res.nb_significant_digit()+_cadna_cancel_value)
	instability(&_cadna_cancel_count);
  }
  return res;
}
inline float_st operator-(const unsigned short& a, const float_st& b){
  float_st res;

  unsigned int random;
  float tmp;

  res.accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = a;
  FLOAT_BIT_FLIP(tmp, random);
  res.x = tmp;
  tmp = b.x;
  FLOAT_BIT_FLIP(tmp, random);
  res.x -= tmp;
  FLOAT_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp = a;
  FLOAT_BIT_FLIP(tmp, random);
  res.y = tmp;
  tmp = b.y;
  FLOAT_BIT_FLIP(tmp, random);
  res.y -= tmp;
  FLOAT_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp = a;
  FLOAT_BIT_FLIP(tmp, random);
  res.z = tmp;
  tmp = b.z;
  FLOAT_BIT_FLIP(tmp, random);
  res.z -= tmp;
  FLOAT_BIT_FLIP(res.z, random);

  if (_cadna_cancel_tag){
    if (b.accuracy==DIGIT_NOT_COMPUTED || b.accuracy==RELIABLE_RESULT)
	b.nb_significant_digit();
    if (b.accuracy >= 	res.nb_significant_digit()+_cadna_cancel_value)
	instability(&_cadna_cancel_count);
  }
  return res;
}
inline float_st operator-(const short& a, const float_st& b){
  float_st res;

  unsigned int random;
  float tmp;

  res.accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = a;
  FLOAT_BIT_FLIP(tmp, random);
  res.x = tmp;
  tmp = b.x;
  FLOAT_BIT_FLIP(tmp, random);
  res.x -= tmp;
  FLOAT_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp = a;
  FLOAT_BIT_FLIP(tmp, random);
  res.y = tmp;
  tmp = b.y;
  FLOAT_BIT_FLIP(tmp, random);
  res.y -= tmp;
  FLOAT_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp = a;
  FLOAT_BIT_FLIP(tmp, random);
  res.z = tmp;
  tmp = b.z;
  FLOAT_BIT_FLIP(tmp, random);
  res.z -= tmp;
  FLOAT_BIT_FLIP(res.z, random);

  if (_cadna_cancel_tag){
    if (b.accuracy==DIGIT_NOT_COMPUTED || b.accuracy==RELIABLE_RESULT)
	b.nb_significant_digit();
    if (b.accuracy >= 	res.nb_significant_digit()+_cadna_cancel_value)
	instability(&_cadna_cancel_count);
  }
  return res;
}
inline float_st operator-(const unsigned char& a, const float_st& b){
  float_st res;

  unsigned int random;
  float tmp;

  res.accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = a;
  FLOAT_BIT_FLIP(tmp, random);
  res.x = tmp;
  tmp = b.x;
  FLOAT_BIT_FLIP(tmp, random);
  res.x -= tmp;
  FLOAT_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp = a;
  FLOAT_BIT_FLIP(tmp, random);
  res.y = tmp;
  tmp = b.y;
  FLOAT_BIT_FLIP(tmp, random);
  res.y -= tmp;
  FLOAT_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp = a;
  FLOAT_BIT_FLIP(tmp, random);
  res.z = tmp;
  tmp = b.z;
  FLOAT_BIT_FLIP(tmp, random);
  res.z -= tmp;
  FLOAT_BIT_FLIP(res.z, random);

  if (_cadna_cancel_tag){
    if (b.accuracy==DIGIT_NOT_COMPUTED || b.accuracy==RELIABLE_RESULT)
	b.nb_significant_digit();
    if (b.accuracy >= 	res.nb_significant_digit()+_cadna_cancel_value)
	instability(&_cadna_cancel_count);
  }
  return res;
}
inline float_st operator-(const char& a, const float_st& b){
  float_st res;

  unsigned int random;
  float tmp;

  res.accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = a;
  FLOAT_BIT_FLIP(tmp, random);
  res.x = tmp;
  tmp = b.x;
  FLOAT_BIT_FLIP(tmp, random);
  res.x -= tmp;
  FLOAT_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp = a;
  FLOAT_BIT_FLIP(tmp, random);
  res.y = tmp;
  tmp = b.y;
  FLOAT_BIT_FLIP(tmp, random);
  res.y -= tmp;
  FLOAT_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp = a;
  FLOAT_BIT_FLIP(tmp, random);
  res.z = tmp;
  tmp = b.z;
  FLOAT_BIT_FLIP(tmp, random);
  res.z -= tmp;
  FLOAT_BIT_FLIP(res.z, random);

  if (_cadna_cancel_tag){
    if (b.accuracy==DIGIT_NOT_COMPUTED || b.accuracy==RELIABLE_RESULT)
	b.nb_significant_digit();
    if (b.accuracy >= 	res.nb_significant_digit()+_cadna_cancel_value)
	instability(&_cadna_cancel_count);
  }
  return res;
}
#ifdef CADNA_HALF_EMULATION
inline float_st operator-(const half& a, const float_st& b){
  float_st res;

  unsigned int random;
  float tmp;

  res.accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = a;
  FLOAT_BIT_FLIP(tmp, random);
  res.x = tmp;
  tmp = b.x;
  FLOAT_BIT_FLIP(tmp, random);
  res.x -= tmp;
  FLOAT_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp = a;
  FLOAT_BIT_FLIP(tmp, random);
  res.y = tmp;
  tmp = b.y;
  FLOAT_BIT_FLIP(tmp, random);
  res.y -= tmp;
  FLOAT_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp = a;
  FLOAT_BIT_FLIP(tmp, random);
  res.z = tmp;
  tmp = b.z;
  FLOAT_BIT_FLIP(tmp, random);
  res.z -= tmp;
  FLOAT_BIT_FLIP(res.z, random);

  if (_cadna_cancel_tag){
    if (b.accuracy==DIGIT_NOT_COMPUTED || b.accuracy==RELIABLE_RESULT)
	b.nb_significant_digit();
    if (b.accuracy >= 	res.nb_significant_digit()+_cadna_cancel_value)
	instability(&_cadna_cancel_count);
  }
  return res;
}
#endif // CADNA_HALF_EMULATION
#ifdef CADNA_HALF_NATIVE
inline float_st operator-(const half& a, const float_st& b){
  float_st res;

  unsigned int random;
  float tmp;

  res.accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = a;
  FLOAT_BIT_FLIP(tmp, random);
  res.x = tmp;
  tmp = b.x;
  FLOAT_BIT_FLIP(tmp, random);
  res.x -= tmp;
  FLOAT_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp = a;
  FLOAT_BIT_FLIP(tmp, random);
  res.y = tmp;
  tmp = b.y;
  FLOAT_BIT_FLIP(tmp, random);
  res.y -= tmp;
  FLOAT_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp = a;
  FLOAT_BIT_FLIP(tmp, random);
  res.z = tmp;
  tmp = b.z;
  FLOAT_BIT_FLIP(tmp, random);
  res.z -= tmp;
  FLOAT_BIT_FLIP(res.z, random);

  if (_cadna_cancel_tag){
    if (b.accuracy==DIGIT_NOT_COMPUTED || b.accuracy==RELIABLE_RESULT)
	b.nb_significant_digit();
    if (b.accuracy >= 	res.nb_significant_digit()+_cadna_cancel_value)
	instability(&_cadna_cancel_count);
  }
  return res;
}
#endif

#ifdef CADNA_HALF_EMULATION
inline half_st operator-(const double& a, const half_st& b){
  half_st res;
  double tmp2;

  unsigned int random;
  half tmp;

  res.accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp2=(random) ? -a : a;
  tmp = b.x;
  HALF_BIT_FLIP(tmp, random);
  res.x = tmp2 - tmp;
  HALF_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp2=(random) ? -a : a;
  tmp = b.y;
  HALF_BIT_FLIP(tmp, random);
  res.y = tmp2 - tmp;
  HALF_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp2=(random) ? -a : a;
  tmp = b.z;
  HALF_BIT_FLIP(tmp, random);
  res.z = tmp2 - tmp;
  HALF_BIT_FLIP(res.z, random);

  
  
  if (_cadna_cancel_tag){
    if (b.accuracy==DIGIT_NOT_COMPUTED || b.accuracy==RELIABLE_RESULT)
	b.nb_significant_digit();
    if (b.accuracy >= 	res.nb_significant_digit()+_cadna_cancel_value)
	instability(&_cadna_cancel_count);
  }
  if (_cadna_half_overflow_tag){
  if (std::isinf(res.x) or std::isinf(res.y) or std::isinf(res.z)){
  instability(&_cadna_half_overflow_count);
  }
}

  return res;
}
inline half_st operator-(const float& a, const half_st& b){
  half_st res;
  float tmp2;

  unsigned int random;
  half tmp;

  res.accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp2=(random) ? -a : a;
  tmp = b.x;
  HALF_BIT_FLIP(tmp, random);
  res.x = tmp2 - tmp;
  HALF_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp2=(random) ? -a : a;
  tmp = b.y;
  HALF_BIT_FLIP(tmp, random);
  res.y = tmp2 - tmp;
  HALF_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp2=(random) ? -a : a;
  tmp = b.z;
  HALF_BIT_FLIP(tmp, random);
  res.z = tmp2 - tmp;
  HALF_BIT_FLIP(res.z, random);

  
  
  if (_cadna_cancel_tag){
    if (b.accuracy==DIGIT_NOT_COMPUTED || b.accuracy==RELIABLE_RESULT)
	b.nb_significant_digit();
    if (b.accuracy >= 	res.nb_significant_digit()+_cadna_cancel_value)
	instability(&_cadna_cancel_count);
  }
  if (_cadna_half_overflow_tag){
  if (std::isinf(res.x) or std::isinf(res.y) or std::isinf(res.z)){
  instability(&_cadna_half_overflow_count);
  }
}

  return res;
}
inline half_st operator-(const unsigned long long& a, const half_st& b){
  half_st res;
  unsigned long long tmp2;

  unsigned int random;
  half tmp;

  res.accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp2=(random) ? -a : a;
  tmp = b.x;
  HALF_BIT_FLIP(tmp, random);
  res.x = tmp2 - tmp;
  HALF_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp2=(random) ? -a : a;
  tmp = b.y;
  HALF_BIT_FLIP(tmp, random);
  res.y = tmp2 - tmp;
  HALF_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp2=(random) ? -a : a;
  tmp = b.z;
  HALF_BIT_FLIP(tmp, random);
  res.z = tmp2 - tmp;
  HALF_BIT_FLIP(res.z, random);

  
  
  if (_cadna_cancel_tag){
    if (b.accuracy==DIGIT_NOT_COMPUTED || b.accuracy==RELIABLE_RESULT)
	b.nb_significant_digit();
    if (b.accuracy >= 	res.nb_significant_digit()+_cadna_cancel_value)
	instability(&_cadna_cancel_count);
  }
  if (_cadna_half_overflow_tag){
  if (std::isinf(res.x) or std::isinf(res.y) or std::isinf(res.z)){
  instability(&_cadna_half_overflow_count);
  }
}

  return res;
}
inline half_st operator-(const long long& a, const half_st& b){
  half_st res;
  long long tmp2;

  unsigned int random;
  half tmp;

  res.accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp2=(random) ? -a : a;
  tmp = b.x;
  HALF_BIT_FLIP(tmp, random);
  res.x = tmp2 - tmp;
  HALF_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp2=(random) ? -a : a;
  tmp = b.y;
  HALF_BIT_FLIP(tmp, random);
  res.y = tmp2 - tmp;
  HALF_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp2=(random) ? -a : a;
  tmp = b.z;
  HALF_BIT_FLIP(tmp, random);
  res.z = tmp2 - tmp;
  HALF_BIT_FLIP(res.z, random);

  
  
  if (_cadna_cancel_tag){
    if (b.accuracy==DIGIT_NOT_COMPUTED || b.accuracy==RELIABLE_RESULT)
	b.nb_significant_digit();
    if (b.accuracy >= 	res.nb_significant_digit()+_cadna_cancel_value)
	instability(&_cadna_cancel_count);
  }
  if (_cadna_half_overflow_tag){
  if (std::isinf(res.x) or std::isinf(res.y) or std::isinf(res.z)){
  instability(&_cadna_half_overflow_count);
  }
}

  return res;
}
inline half_st operator-(const unsigned long& a, const half_st& b){
  half_st res;
  unsigned long tmp2;

  unsigned int random;
  half tmp;

  res.accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp2=(random) ? -a : a;
  tmp = b.x;
  HALF_BIT_FLIP(tmp, random);
  res.x = tmp2 - tmp;
  HALF_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp2=(random) ? -a : a;
  tmp = b.y;
  HALF_BIT_FLIP(tmp, random);
  res.y = tmp2 - tmp;
  HALF_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp2=(random) ? -a : a;
  tmp = b.z;
  HALF_BIT_FLIP(tmp, random);
  res.z = tmp2 - tmp;
  HALF_BIT_FLIP(res.z, random);

  
  
  if (_cadna_cancel_tag){
    if (b.accuracy==DIGIT_NOT_COMPUTED || b.accuracy==RELIABLE_RESULT)
	b.nb_significant_digit();
    if (b.accuracy >= 	res.nb_significant_digit()+_cadna_cancel_value)
	instability(&_cadna_cancel_count);
  }
  if (_cadna_half_overflow_tag){
  if (std::isinf(res.x) or std::isinf(res.y) or std::isinf(res.z)){
  instability(&_cadna_half_overflow_count);
  }
}

  return res;
}
inline half_st operator-(const long& a, const half_st& b){
  half_st res;
  long tmp2;

  unsigned int random;
  half tmp;

  res.accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp2=(random) ? -a : a;
  tmp = b.x;
  HALF_BIT_FLIP(tmp, random);
  res.x = tmp2 - tmp;
  HALF_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp2=(random) ? -a : a;
  tmp = b.y;
  HALF_BIT_FLIP(tmp, random);
  res.y = tmp2 - tmp;
  HALF_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp2=(random) ? -a : a;
  tmp = b.z;
  HALF_BIT_FLIP(tmp, random);
  res.z = tmp2 - tmp;
  HALF_BIT_FLIP(res.z, random);

  
  
  if (_cadna_cancel_tag){
    if (b.accuracy==DIGIT_NOT_COMPUTED || b.accuracy==RELIABLE_RESULT)
	b.nb_significant_digit();
    if (b.accuracy >= 	res.nb_significant_digit()+_cadna_cancel_value)
	instability(&_cadna_cancel_count);
  }
  if (_cadna_half_overflow_tag){
  if (std::isinf(res.x) or std::isinf(res.y) or std::isinf(res.z)){
  instability(&_cadna_half_overflow_count);
  }
}

  return res;
}
inline half_st operator-(const unsigned int & a, const half_st& b){
  half_st res;
  unsigned int  tmp2;

  unsigned int random;
  half tmp;

  res.accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp2=(random) ? -a : a;
  tmp = b.x;
  HALF_BIT_FLIP(tmp, random);
  res.x = tmp2 - tmp;
  HALF_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp2=(random) ? -a : a;
  tmp = b.y;
  HALF_BIT_FLIP(tmp, random);
  res.y = tmp2 - tmp;
  HALF_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp2=(random) ? -a : a;
  tmp = b.z;
  HALF_BIT_FLIP(tmp, random);
  res.z = tmp2 - tmp;
  HALF_BIT_FLIP(res.z, random);

  
  
  if (_cadna_cancel_tag){
    if (b.accuracy==DIGIT_NOT_COMPUTED || b.accuracy==RELIABLE_RESULT)
	b.nb_significant_digit();
    if (b.accuracy >= 	res.nb_significant_digit()+_cadna_cancel_value)
	instability(&_cadna_cancel_count);
  }
  if (_cadna_half_overflow_tag){
  if (std::isinf(res.x) or std::isinf(res.y) or std::isinf(res.z)){
  instability(&_cadna_half_overflow_count);
  }
}

  return res;
}
inline half_st operator-(const int & a, const half_st& b){
  half_st res;
  int  tmp2;

  unsigned int random;
  half tmp;

  res.accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp2=(random) ? -a : a;
  tmp = b.x;
  HALF_BIT_FLIP(tmp, random);
  res.x = tmp2 - tmp;
  HALF_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp2=(random) ? -a : a;
  tmp = b.y;
  HALF_BIT_FLIP(tmp, random);
  res.y = tmp2 - tmp;
  HALF_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp2=(random) ? -a : a;
  tmp = b.z;
  HALF_BIT_FLIP(tmp, random);
  res.z = tmp2 - tmp;
  HALF_BIT_FLIP(res.z, random);

  
  
  if (_cadna_cancel_tag){
    if (b.accuracy==DIGIT_NOT_COMPUTED || b.accuracy==RELIABLE_RESULT)
	b.nb_significant_digit();
    if (b.accuracy >= 	res.nb_significant_digit()+_cadna_cancel_value)
	instability(&_cadna_cancel_count);
  }
  if (_cadna_half_overflow_tag){
  if (std::isinf(res.x) or std::isinf(res.y) or std::isinf(res.z)){
  instability(&_cadna_half_overflow_count);
  }
}

  return res;
}
inline half_st operator-(const unsigned short& a, const half_st& b){
  half_st res;
  unsigned short tmp2;

  unsigned int random;
  half tmp;

  res.accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp2=(random) ? -a : a;
  tmp = b.x;
  HALF_BIT_FLIP(tmp, random);
  res.x = tmp2 - tmp;
  HALF_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp2=(random) ? -a : a;
  tmp = b.y;
  HALF_BIT_FLIP(tmp, random);
  res.y = tmp2 - tmp;
  HALF_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp2=(random) ? -a : a;
  tmp = b.z;
  HALF_BIT_FLIP(tmp, random);
  res.z = tmp2 - tmp;
  HALF_BIT_FLIP(res.z, random);

  
  
  if (_cadna_cancel_tag){
    if (b.accuracy==DIGIT_NOT_COMPUTED || b.accuracy==RELIABLE_RESULT)
	b.nb_significant_digit();
    if (b.accuracy >= 	res.nb_significant_digit()+_cadna_cancel_value)
	instability(&_cadna_cancel_count);
  }
  if (_cadna_half_overflow_tag){
  if (std::isinf(res.x) or std::isinf(res.y) or std::isinf(res.z)){
  instability(&_cadna_half_overflow_count);
  }
}

  return res;
}
inline half_st operator-(const short& a, const half_st& b){
  half_st res;
  short tmp2;

  unsigned int random;
  half tmp;

  res.accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp2=(random) ? -a : a;
  tmp = b.x;
  HALF_BIT_FLIP(tmp, random);
  res.x = tmp2 - tmp;
  HALF_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp2=(random) ? -a : a;
  tmp = b.y;
  HALF_BIT_FLIP(tmp, random);
  res.y = tmp2 - tmp;
  HALF_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp2=(random) ? -a : a;
  tmp = b.z;
  HALF_BIT_FLIP(tmp, random);
  res.z = tmp2 - tmp;
  HALF_BIT_FLIP(res.z, random);

  
  
  if (_cadna_cancel_tag){
    if (b.accuracy==DIGIT_NOT_COMPUTED || b.accuracy==RELIABLE_RESULT)
	b.nb_significant_digit();
    if (b.accuracy >= 	res.nb_significant_digit()+_cadna_cancel_value)
	instability(&_cadna_cancel_count);
  }
  if (_cadna_half_overflow_tag){
  if (std::isinf(res.x) or std::isinf(res.y) or std::isinf(res.z)){
  instability(&_cadna_half_overflow_count);
  }
}

  return res;
}
inline half_st operator-(const unsigned char& a, const half_st& b){
  half_st res;
  unsigned char tmp2;

  unsigned int random;
  half tmp;

  res.accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp2=(random) ? -a : a;
  tmp = b.x;
  HALF_BIT_FLIP(tmp, random);
  res.x = tmp2 - tmp;
  HALF_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp2=(random) ? -a : a;
  tmp = b.y;
  HALF_BIT_FLIP(tmp, random);
  res.y = tmp2 - tmp;
  HALF_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp2=(random) ? -a : a;
  tmp = b.z;
  HALF_BIT_FLIP(tmp, random);
  res.z = tmp2 - tmp;
  HALF_BIT_FLIP(res.z, random);

  
  
  if (_cadna_cancel_tag){
    if (b.accuracy==DIGIT_NOT_COMPUTED || b.accuracy==RELIABLE_RESULT)
	b.nb_significant_digit();
    if (b.accuracy >= 	res.nb_significant_digit()+_cadna_cancel_value)
	instability(&_cadna_cancel_count);
  }
  if (_cadna_half_overflow_tag){
  if (std::isinf(res.x) or std::isinf(res.y) or std::isinf(res.z)){
  instability(&_cadna_half_overflow_count);
  }
}

  return res;
}
inline half_st operator-(const char& a, const half_st& b){
  half_st res;
  char tmp2;

  unsigned int random;
  half tmp;

  res.accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp2=(random) ? -a : a;
  tmp = b.x;
  HALF_BIT_FLIP(tmp, random);
  res.x = tmp2 - tmp;
  HALF_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp2=(random) ? -a : a;
  tmp = b.y;
  HALF_BIT_FLIP(tmp, random);
  res.y = tmp2 - tmp;
  HALF_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp2=(random) ? -a : a;
  tmp = b.z;
  HALF_BIT_FLIP(tmp, random);
  res.z = tmp2 - tmp;
  HALF_BIT_FLIP(res.z, random);

  
  
  if (_cadna_cancel_tag){
    if (b.accuracy==DIGIT_NOT_COMPUTED || b.accuracy==RELIABLE_RESULT)
	b.nb_significant_digit();
    if (b.accuracy >= 	res.nb_significant_digit()+_cadna_cancel_value)
	instability(&_cadna_cancel_count);
  }
  if (_cadna_half_overflow_tag){
  if (std::isinf(res.x) or std::isinf(res.y) or std::isinf(res.z)){
  instability(&_cadna_half_overflow_count);
  }
}

  return res;
}
inline half_st operator-(const half& a, const half_st& b){
  half_st res;
  half tmp2;

  unsigned int random;
  half tmp;

  res.accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp2=(random) ? -a : a;
  tmp = b.x;
  HALF_BIT_FLIP(tmp, random);
  res.x = tmp2 - tmp;
  HALF_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp2=(random) ? -a : a;
  tmp = b.y;
  HALF_BIT_FLIP(tmp, random);
  res.y = tmp2 - tmp;
  HALF_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp2=(random) ? -a : a;
  tmp = b.z;
  HALF_BIT_FLIP(tmp, random);
  res.z = tmp2 - tmp;
  HALF_BIT_FLIP(res.z, random);

  
  
  if (_cadna_cancel_tag){
    if (b.accuracy==DIGIT_NOT_COMPUTED || b.accuracy==RELIABLE_RESULT)
	b.nb_significant_digit();
    if (b.accuracy >= 	res.nb_significant_digit()+_cadna_cancel_value)
	instability(&_cadna_cancel_count);
  }
  if (_cadna_half_overflow_tag){
  if (std::isinf(res.x) or std::isinf(res.y) or std::isinf(res.z)){
  instability(&_cadna_half_overflow_count);
  }
}

  return res;
}
#endif // CADNA_HALF_EMULATION

#ifdef CADNA_HALF_NATIVE
inline half_st operator-(const double& a, const half_st& b){
  half_st res;
  double tmp2;

  unsigned int random;
  half tmp;

  res.accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp2=(random) ? -a : a;
  tmp = b.x;
  HALF_BIT_FLIP(tmp, random);
  res.x = tmp2 - tmp;
  HALF_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp2=(random) ? -a : a;
  tmp = b.y;
  HALF_BIT_FLIP(tmp, random);
  res.y = tmp2 - tmp;
  HALF_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp2=(random) ? -a : a;
  tmp = b.z;
  HALF_BIT_FLIP(tmp, random);
  res.z = tmp2 - tmp;
  HALF_BIT_FLIP(res.z, random);

  
  
  if (_cadna_cancel_tag){
    if (b.accuracy==DIGIT_NOT_COMPUTED || b.accuracy==RELIABLE_RESULT)
	b.nb_significant_digit();
    if (b.accuracy >= 	res.nb_significant_digit()+_cadna_cancel_value)
	instability(&_cadna_cancel_count);
  }
  if (_cadna_half_overflow_tag){
  if (std::isinf(res.x) or std::isinf(res.y) or std::isinf(res.z)){
  instability(&_cadna_half_overflow_count);
  }
}

  return res;
}
inline half_st operator-(const float& a, const half_st& b){
  half_st res;
  float tmp2;

  unsigned int random;
  half tmp;

  res.accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp2=(random) ? -a : a;
  tmp = b.x;
  HALF_BIT_FLIP(tmp, random);
  res.x = tmp2 - tmp;
  HALF_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp2=(random) ? -a : a;
  tmp = b.y;
  HALF_BIT_FLIP(tmp, random);
  res.y = tmp2 - tmp;
  HALF_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp2=(random) ? -a : a;
  tmp = b.z;
  HALF_BIT_FLIP(tmp, random);
  res.z = tmp2 - tmp;
  HALF_BIT_FLIP(res.z, random);

  
  
  if (_cadna_cancel_tag){
    if (b.accuracy==DIGIT_NOT_COMPUTED || b.accuracy==RELIABLE_RESULT)
	b.nb_significant_digit();
    if (b.accuracy >= 	res.nb_significant_digit()+_cadna_cancel_value)
	instability(&_cadna_cancel_count);
  }
  if (_cadna_half_overflow_tag){
  if (std::isinf(res.x) or std::isinf(res.y) or std::isinf(res.z)){
  instability(&_cadna_half_overflow_count);
  }
}

  return res;
}
inline half_st operator-(const unsigned long long& a, const half_st& b){
  half_st res;
  unsigned long long tmp2;

  unsigned int random;
  half tmp;

  res.accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp2=(random) ? -a : a;
  tmp = b.x;
  HALF_BIT_FLIP(tmp, random);
  res.x = tmp2 - tmp;
  HALF_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp2=(random) ? -a : a;
  tmp = b.y;
  HALF_BIT_FLIP(tmp, random);
  res.y = tmp2 - tmp;
  HALF_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp2=(random) ? -a : a;
  tmp = b.z;
  HALF_BIT_FLIP(tmp, random);
  res.z = tmp2 - tmp;
  HALF_BIT_FLIP(res.z, random);

  
  
  if (_cadna_cancel_tag){
    if (b.accuracy==DIGIT_NOT_COMPUTED || b.accuracy==RELIABLE_RESULT)
	b.nb_significant_digit();
    if (b.accuracy >= 	res.nb_significant_digit()+_cadna_cancel_value)
	instability(&_cadna_cancel_count);
  }
  if (_cadna_half_overflow_tag){
  if (std::isinf(res.x) or std::isinf(res.y) or std::isinf(res.z)){
  instability(&_cadna_half_overflow_count);
  }
}

  return res;
}
inline half_st operator-(const long long& a, const half_st& b){
  half_st res;
  long long tmp2;

  unsigned int random;
  half tmp;

  res.accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp2=(random) ? -a : a;
  tmp = b.x;
  HALF_BIT_FLIP(tmp, random);
  res.x = tmp2 - tmp;
  HALF_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp2=(random) ? -a : a;
  tmp = b.y;
  HALF_BIT_FLIP(tmp, random);
  res.y = tmp2 - tmp;
  HALF_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp2=(random) ? -a : a;
  tmp = b.z;
  HALF_BIT_FLIP(tmp, random);
  res.z = tmp2 - tmp;
  HALF_BIT_FLIP(res.z, random);

  
  
  if (_cadna_cancel_tag){
    if (b.accuracy==DIGIT_NOT_COMPUTED || b.accuracy==RELIABLE_RESULT)
	b.nb_significant_digit();
    if (b.accuracy >= 	res.nb_significant_digit()+_cadna_cancel_value)
	instability(&_cadna_cancel_count);
  }
  if (_cadna_half_overflow_tag){
  if (std::isinf(res.x) or std::isinf(res.y) or std::isinf(res.z)){
  instability(&_cadna_half_overflow_count);
  }
}

  return res;
}
inline half_st operator-(const unsigned long& a, const half_st& b){
  half_st res;
  unsigned long tmp2;

  unsigned int random;
  half tmp;

  res.accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp2=(random) ? -a : a;
  tmp = b.x;
  HALF_BIT_FLIP(tmp, random);
  res.x = tmp2 - tmp;
  HALF_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp2=(random) ? -a : a;
  tmp = b.y;
  HALF_BIT_FLIP(tmp, random);
  res.y = tmp2 - tmp;
  HALF_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp2=(random) ? -a : a;
  tmp = b.z;
  HALF_BIT_FLIP(tmp, random);
  res.z = tmp2 - tmp;
  HALF_BIT_FLIP(res.z, random);

  
  
  if (_cadna_cancel_tag){
    if (b.accuracy==DIGIT_NOT_COMPUTED || b.accuracy==RELIABLE_RESULT)
	b.nb_significant_digit();
    if (b.accuracy >= 	res.nb_significant_digit()+_cadna_cancel_value)
	instability(&_cadna_cancel_count);
  }
  if (_cadna_half_overflow_tag){
  if (std::isinf(res.x) or std::isinf(res.y) or std::isinf(res.z)){
  instability(&_cadna_half_overflow_count);
  }
}

  return res;
}
inline half_st operator-(const long& a, const half_st& b){
  half_st res;
  long tmp2;

  unsigned int random;
  half tmp;

  res.accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp2=(random) ? -a : a;
  tmp = b.x;
  HALF_BIT_FLIP(tmp, random);
  res.x = tmp2 - tmp;
  HALF_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp2=(random) ? -a : a;
  tmp = b.y;
  HALF_BIT_FLIP(tmp, random);
  res.y = tmp2 - tmp;
  HALF_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp2=(random) ? -a : a;
  tmp = b.z;
  HALF_BIT_FLIP(tmp, random);
  res.z = tmp2 - tmp;
  HALF_BIT_FLIP(res.z, random);

  
  
  if (_cadna_cancel_tag){
    if (b.accuracy==DIGIT_NOT_COMPUTED || b.accuracy==RELIABLE_RESULT)
	b.nb_significant_digit();
    if (b.accuracy >= 	res.nb_significant_digit()+_cadna_cancel_value)
	instability(&_cadna_cancel_count);
  }
  if (_cadna_half_overflow_tag){
  if (std::isinf(res.x) or std::isinf(res.y) or std::isinf(res.z)){
  instability(&_cadna_half_overflow_count);
  }
}

  return res;
}
inline half_st operator-(const unsigned int & a, const half_st& b){
  half_st res;
  unsigned int  tmp2;

  unsigned int random;
  half tmp;

  res.accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp2=(random) ? -a : a;
  tmp = b.x;
  HALF_BIT_FLIP(tmp, random);
  res.x = tmp2 - tmp;
  HALF_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp2=(random) ? -a : a;
  tmp = b.y;
  HALF_BIT_FLIP(tmp, random);
  res.y = tmp2 - tmp;
  HALF_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp2=(random) ? -a : a;
  tmp = b.z;
  HALF_BIT_FLIP(tmp, random);
  res.z = tmp2 - tmp;
  HALF_BIT_FLIP(res.z, random);

  
  
  if (_cadna_cancel_tag){
    if (b.accuracy==DIGIT_NOT_COMPUTED || b.accuracy==RELIABLE_RESULT)
	b.nb_significant_digit();
    if (b.accuracy >= 	res.nb_significant_digit()+_cadna_cancel_value)
	instability(&_cadna_cancel_count);
  }
  if (_cadna_half_overflow_tag){
  if (std::isinf(res.x) or std::isinf(res.y) or std::isinf(res.z)){
  instability(&_cadna_half_overflow_count);
  }
}

  return res;
}
inline half_st operator-(const int & a, const half_st& b){
  half_st res;
  int  tmp2;

  unsigned int random;
  half tmp;

  res.accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp2=(random) ? -a : a;
  tmp = b.x;
  HALF_BIT_FLIP(tmp, random);
  res.x = tmp2 - tmp;
  HALF_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp2=(random) ? -a : a;
  tmp = b.y;
  HALF_BIT_FLIP(tmp, random);
  res.y = tmp2 - tmp;
  HALF_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp2=(random) ? -a : a;
  tmp = b.z;
  HALF_BIT_FLIP(tmp, random);
  res.z = tmp2 - tmp;
  HALF_BIT_FLIP(res.z, random);

  
  
  if (_cadna_cancel_tag){
    if (b.accuracy==DIGIT_NOT_COMPUTED || b.accuracy==RELIABLE_RESULT)
	b.nb_significant_digit();
    if (b.accuracy >= 	res.nb_significant_digit()+_cadna_cancel_value)
	instability(&_cadna_cancel_count);
  }
  if (_cadna_half_overflow_tag){
  if (std::isinf(res.x) or std::isinf(res.y) or std::isinf(res.z)){
  instability(&_cadna_half_overflow_count);
  }
}

  return res;
}
inline half_st operator-(const unsigned short& a, const half_st& b){
  half_st res;
  unsigned short tmp2;

  unsigned int random;
  half tmp;

  res.accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp2=(random) ? -a : a;
  tmp = b.x;
  HALF_BIT_FLIP(tmp, random);
  res.x = tmp2 - tmp;
  HALF_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp2=(random) ? -a : a;
  tmp = b.y;
  HALF_BIT_FLIP(tmp, random);
  res.y = tmp2 - tmp;
  HALF_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp2=(random) ? -a : a;
  tmp = b.z;
  HALF_BIT_FLIP(tmp, random);
  res.z = tmp2 - tmp;
  HALF_BIT_FLIP(res.z, random);

  
  
  if (_cadna_cancel_tag){
    if (b.accuracy==DIGIT_NOT_COMPUTED || b.accuracy==RELIABLE_RESULT)
	b.nb_significant_digit();
    if (b.accuracy >= 	res.nb_significant_digit()+_cadna_cancel_value)
	instability(&_cadna_cancel_count);
  }
  if (_cadna_half_overflow_tag){
  if (std::isinf(res.x) or std::isinf(res.y) or std::isinf(res.z)){
  instability(&_cadna_half_overflow_count);
  }
}

  return res;
}
inline half_st operator-(const short& a, const half_st& b){
  half_st res;
  short tmp2;

  unsigned int random;
  half tmp;

  res.accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp2=(random) ? -a : a;
  tmp = b.x;
  HALF_BIT_FLIP(tmp, random);
  res.x = tmp2 - tmp;
  HALF_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp2=(random) ? -a : a;
  tmp = b.y;
  HALF_BIT_FLIP(tmp, random);
  res.y = tmp2 - tmp;
  HALF_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp2=(random) ? -a : a;
  tmp = b.z;
  HALF_BIT_FLIP(tmp, random);
  res.z = tmp2 - tmp;
  HALF_BIT_FLIP(res.z, random);

  
  
  if (_cadna_cancel_tag){
    if (b.accuracy==DIGIT_NOT_COMPUTED || b.accuracy==RELIABLE_RESULT)
	b.nb_significant_digit();
    if (b.accuracy >= 	res.nb_significant_digit()+_cadna_cancel_value)
	instability(&_cadna_cancel_count);
  }
  if (_cadna_half_overflow_tag){
  if (std::isinf(res.x) or std::isinf(res.y) or std::isinf(res.z)){
  instability(&_cadna_half_overflow_count);
  }
}

  return res;
}
inline half_st operator-(const unsigned char& a, const half_st& b){
  half_st res;
  unsigned char tmp2;

  unsigned int random;
  half tmp;

  res.accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp2=(random) ? -a : a;
  tmp = b.x;
  HALF_BIT_FLIP(tmp, random);
  res.x = tmp2 - tmp;
  HALF_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp2=(random) ? -a : a;
  tmp = b.y;
  HALF_BIT_FLIP(tmp, random);
  res.y = tmp2 - tmp;
  HALF_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp2=(random) ? -a : a;
  tmp = b.z;
  HALF_BIT_FLIP(tmp, random);
  res.z = tmp2 - tmp;
  HALF_BIT_FLIP(res.z, random);

  
  
  if (_cadna_cancel_tag){
    if (b.accuracy==DIGIT_NOT_COMPUTED || b.accuracy==RELIABLE_RESULT)
	b.nb_significant_digit();
    if (b.accuracy >= 	res.nb_significant_digit()+_cadna_cancel_value)
	instability(&_cadna_cancel_count);
  }
  if (_cadna_half_overflow_tag){
  if (std::isinf(res.x) or std::isinf(res.y) or std::isinf(res.z)){
  instability(&_cadna_half_overflow_count);
  }
}

  return res;
}
inline half_st operator-(const char& a, const half_st& b){
  half_st res;
  char tmp2;

  unsigned int random;
  half tmp;

  res.accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp2=(random) ? -a : a;
  tmp = b.x;
  HALF_BIT_FLIP(tmp, random);
  res.x = tmp2 - tmp;
  HALF_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp2=(random) ? -a : a;
  tmp = b.y;
  HALF_BIT_FLIP(tmp, random);
  res.y = tmp2 - tmp;
  HALF_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp2=(random) ? -a : a;
  tmp = b.z;
  HALF_BIT_FLIP(tmp, random);
  res.z = tmp2 - tmp;
  HALF_BIT_FLIP(res.z, random);

  
  
  if (_cadna_cancel_tag){
    if (b.accuracy==DIGIT_NOT_COMPUTED || b.accuracy==RELIABLE_RESULT)
	b.nb_significant_digit();
    if (b.accuracy >= 	res.nb_significant_digit()+_cadna_cancel_value)
	instability(&_cadna_cancel_count);
  }
  if (_cadna_half_overflow_tag){
  if (std::isinf(res.x) or std::isinf(res.y) or std::isinf(res.z)){
  instability(&_cadna_half_overflow_count);
  }
}

  return res;
}
inline half_st operator-(const half& a, const half_st& b){
  half_st res;
  half tmp2;

  unsigned int random;
  half tmp;

  res.accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp2=(random) ? -a : a;
  tmp = b.x;
  HALF_BIT_FLIP(tmp, random);
  res.x = tmp2 - tmp;
  HALF_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp2=(random) ? -a : a;
  tmp = b.y;
  HALF_BIT_FLIP(tmp, random);
  res.y = tmp2 - tmp;
  HALF_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp2=(random) ? -a : a;
  tmp = b.z;
  HALF_BIT_FLIP(tmp, random);
  res.z = tmp2 - tmp;
  HALF_BIT_FLIP(res.z, random);

  
  
  if (_cadna_cancel_tag){
    if (b.accuracy==DIGIT_NOT_COMPUTED || b.accuracy==RELIABLE_RESULT)
	b.nb_significant_digit();
    if (b.accuracy >= 	res.nb_significant_digit()+_cadna_cancel_value)
	instability(&_cadna_cancel_count);
  }
  if (_cadna_half_overflow_tag){
  if (std::isinf(res.x) or std::isinf(res.y) or std::isinf(res.z)){
  instability(&_cadna_half_overflow_count);
  }
}

  return res;
}
#endif
/////////////////////////////////////////////////////////////////////////




inline double_st operator-(const double_st& a, const double& b){
  double_st res;

  unsigned int random;
  double tmp;

  res.accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = a.x;
  DOUBLE_BIT_FLIP(tmp, random);
  res.x = tmp;
  tmp = b;
  DOUBLE_BIT_FLIP(tmp, random);
  res.x -= tmp;
  DOUBLE_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp = a.y;
  DOUBLE_BIT_FLIP(tmp, random);
  res.y = tmp;
  tmp = b;
  DOUBLE_BIT_FLIP(tmp, random);
  res.y -= tmp;
  DOUBLE_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp = a.z;
  DOUBLE_BIT_FLIP(tmp, random);
  res.z = tmp;
  tmp = b;
  DOUBLE_BIT_FLIP(tmp, random);
  res.z -= tmp;
  DOUBLE_BIT_FLIP(res.z, random);

  if (_cadna_cancel_tag){
    if (a.accuracy==DIGIT_NOT_COMPUTED || a.accuracy==RELIABLE_RESULT)
	a.nb_significant_digit();
    if (a.accuracy >= 	res.nb_significant_digit()+_cadna_cancel_value)
	instability(&_cadna_cancel_count);
  }
  return res;
}
inline double_st operator-(const double_st& a, const float& b){
  double_st res;

  unsigned int random;
  double tmp;

  res.accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = a.x;
  DOUBLE_BIT_FLIP(tmp, random);
  res.x = tmp;
  tmp = b;
  DOUBLE_BIT_FLIP(tmp, random);
  res.x -= tmp;
  DOUBLE_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp = a.y;
  DOUBLE_BIT_FLIP(tmp, random);
  res.y = tmp;
  tmp = b;
  DOUBLE_BIT_FLIP(tmp, random);
  res.y -= tmp;
  DOUBLE_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp = a.z;
  DOUBLE_BIT_FLIP(tmp, random);
  res.z = tmp;
  tmp = b;
  DOUBLE_BIT_FLIP(tmp, random);
  res.z -= tmp;
  DOUBLE_BIT_FLIP(res.z, random);

  if (_cadna_cancel_tag){
    if (a.accuracy==DIGIT_NOT_COMPUTED || a.accuracy==RELIABLE_RESULT)
	a.nb_significant_digit();
    if (a.accuracy >= 	res.nb_significant_digit()+_cadna_cancel_value)
	instability(&_cadna_cancel_count);
  }
  return res;
}
inline double_st operator-(const double_st& a, const unsigned long long& b){
  double_st res;

  unsigned int random;
  double tmp;

  res.accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = a.x;
  DOUBLE_BIT_FLIP(tmp, random);
  res.x = tmp;
  tmp = b;
  DOUBLE_BIT_FLIP(tmp, random);
  res.x -= tmp;
  DOUBLE_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp = a.y;
  DOUBLE_BIT_FLIP(tmp, random);
  res.y = tmp;
  tmp = b;
  DOUBLE_BIT_FLIP(tmp, random);
  res.y -= tmp;
  DOUBLE_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp = a.z;
  DOUBLE_BIT_FLIP(tmp, random);
  res.z = tmp;
  tmp = b;
  DOUBLE_BIT_FLIP(tmp, random);
  res.z -= tmp;
  DOUBLE_BIT_FLIP(res.z, random);

  if (_cadna_cancel_tag){
    if (a.accuracy==DIGIT_NOT_COMPUTED || a.accuracy==RELIABLE_RESULT)
	a.nb_significant_digit();
    if (a.accuracy >= 	res.nb_significant_digit()+_cadna_cancel_value)
	instability(&_cadna_cancel_count);
  }
  return res;
}
inline double_st operator-(const double_st& a, const long long& b){
  double_st res;

  unsigned int random;
  double tmp;

  res.accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = a.x;
  DOUBLE_BIT_FLIP(tmp, random);
  res.x = tmp;
  tmp = b;
  DOUBLE_BIT_FLIP(tmp, random);
  res.x -= tmp;
  DOUBLE_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp = a.y;
  DOUBLE_BIT_FLIP(tmp, random);
  res.y = tmp;
  tmp = b;
  DOUBLE_BIT_FLIP(tmp, random);
  res.y -= tmp;
  DOUBLE_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp = a.z;
  DOUBLE_BIT_FLIP(tmp, random);
  res.z = tmp;
  tmp = b;
  DOUBLE_BIT_FLIP(tmp, random);
  res.z -= tmp;
  DOUBLE_BIT_FLIP(res.z, random);

  if (_cadna_cancel_tag){
    if (a.accuracy==DIGIT_NOT_COMPUTED || a.accuracy==RELIABLE_RESULT)
	a.nb_significant_digit();
    if (a.accuracy >= 	res.nb_significant_digit()+_cadna_cancel_value)
	instability(&_cadna_cancel_count);
  }
  return res;
}
inline double_st operator-(const double_st& a, const unsigned long& b){
  double_st res;

  unsigned int random;
  double tmp;

  res.accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = a.x;
  DOUBLE_BIT_FLIP(tmp, random);
  res.x = tmp;
  tmp = b;
  DOUBLE_BIT_FLIP(tmp, random);
  res.x -= tmp;
  DOUBLE_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp = a.y;
  DOUBLE_BIT_FLIP(tmp, random);
  res.y = tmp;
  tmp = b;
  DOUBLE_BIT_FLIP(tmp, random);
  res.y -= tmp;
  DOUBLE_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp = a.z;
  DOUBLE_BIT_FLIP(tmp, random);
  res.z = tmp;
  tmp = b;
  DOUBLE_BIT_FLIP(tmp, random);
  res.z -= tmp;
  DOUBLE_BIT_FLIP(res.z, random);

  if (_cadna_cancel_tag){
    if (a.accuracy==DIGIT_NOT_COMPUTED || a.accuracy==RELIABLE_RESULT)
	a.nb_significant_digit();
    if (a.accuracy >= 	res.nb_significant_digit()+_cadna_cancel_value)
	instability(&_cadna_cancel_count);
  }
  return res;
}
inline double_st operator-(const double_st& a, const long& b){
  double_st res;

  unsigned int random;
  double tmp;

  res.accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = a.x;
  DOUBLE_BIT_FLIP(tmp, random);
  res.x = tmp;
  tmp = b;
  DOUBLE_BIT_FLIP(tmp, random);
  res.x -= tmp;
  DOUBLE_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp = a.y;
  DOUBLE_BIT_FLIP(tmp, random);
  res.y = tmp;
  tmp = b;
  DOUBLE_BIT_FLIP(tmp, random);
  res.y -= tmp;
  DOUBLE_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp = a.z;
  DOUBLE_BIT_FLIP(tmp, random);
  res.z = tmp;
  tmp = b;
  DOUBLE_BIT_FLIP(tmp, random);
  res.z -= tmp;
  DOUBLE_BIT_FLIP(res.z, random);

  if (_cadna_cancel_tag){
    if (a.accuracy==DIGIT_NOT_COMPUTED || a.accuracy==RELIABLE_RESULT)
	a.nb_significant_digit();
    if (a.accuracy >= 	res.nb_significant_digit()+_cadna_cancel_value)
	instability(&_cadna_cancel_count);
  }
  return res;
}
inline double_st operator-(const double_st& a, const unsigned int & b){
  double_st res;

  unsigned int random;
  double tmp;

  res.accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = a.x;
  DOUBLE_BIT_FLIP(tmp, random);
  res.x = tmp;
  tmp = b;
  DOUBLE_BIT_FLIP(tmp, random);
  res.x -= tmp;
  DOUBLE_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp = a.y;
  DOUBLE_BIT_FLIP(tmp, random);
  res.y = tmp;
  tmp = b;
  DOUBLE_BIT_FLIP(tmp, random);
  res.y -= tmp;
  DOUBLE_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp = a.z;
  DOUBLE_BIT_FLIP(tmp, random);
  res.z = tmp;
  tmp = b;
  DOUBLE_BIT_FLIP(tmp, random);
  res.z -= tmp;
  DOUBLE_BIT_FLIP(res.z, random);

  if (_cadna_cancel_tag){
    if (a.accuracy==DIGIT_NOT_COMPUTED || a.accuracy==RELIABLE_RESULT)
	a.nb_significant_digit();
    if (a.accuracy >= 	res.nb_significant_digit()+_cadna_cancel_value)
	instability(&_cadna_cancel_count);
  }
  return res;
}
inline double_st operator-(const double_st& a, const int & b){
  double_st res;

  unsigned int random;
  double tmp;

  res.accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = a.x;
  DOUBLE_BIT_FLIP(tmp, random);
  res.x = tmp;
  tmp = b;
  DOUBLE_BIT_FLIP(tmp, random);
  res.x -= tmp;
  DOUBLE_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp = a.y;
  DOUBLE_BIT_FLIP(tmp, random);
  res.y = tmp;
  tmp = b;
  DOUBLE_BIT_FLIP(tmp, random);
  res.y -= tmp;
  DOUBLE_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp = a.z;
  DOUBLE_BIT_FLIP(tmp, random);
  res.z = tmp;
  tmp = b;
  DOUBLE_BIT_FLIP(tmp, random);
  res.z -= tmp;
  DOUBLE_BIT_FLIP(res.z, random);

  if (_cadna_cancel_tag){
    if (a.accuracy==DIGIT_NOT_COMPUTED || a.accuracy==RELIABLE_RESULT)
	a.nb_significant_digit();
    if (a.accuracy >= 	res.nb_significant_digit()+_cadna_cancel_value)
	instability(&_cadna_cancel_count);
  }
  return res;
}
inline double_st operator-(const double_st& a, const unsigned short& b){
  double_st res;

  unsigned int random;
  double tmp;

  res.accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = a.x;
  DOUBLE_BIT_FLIP(tmp, random);
  res.x = tmp;
  tmp = b;
  DOUBLE_BIT_FLIP(tmp, random);
  res.x -= tmp;
  DOUBLE_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp = a.y;
  DOUBLE_BIT_FLIP(tmp, random);
  res.y = tmp;
  tmp = b;
  DOUBLE_BIT_FLIP(tmp, random);
  res.y -= tmp;
  DOUBLE_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp = a.z;
  DOUBLE_BIT_FLIP(tmp, random);
  res.z = tmp;
  tmp = b;
  DOUBLE_BIT_FLIP(tmp, random);
  res.z -= tmp;
  DOUBLE_BIT_FLIP(res.z, random);

  if (_cadna_cancel_tag){
    if (a.accuracy==DIGIT_NOT_COMPUTED || a.accuracy==RELIABLE_RESULT)
	a.nb_significant_digit();
    if (a.accuracy >= 	res.nb_significant_digit()+_cadna_cancel_value)
	instability(&_cadna_cancel_count);
  }
  return res;
}
inline double_st operator-(const double_st& a, const short& b){
  double_st res;

  unsigned int random;
  double tmp;

  res.accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = a.x;
  DOUBLE_BIT_FLIP(tmp, random);
  res.x = tmp;
  tmp = b;
  DOUBLE_BIT_FLIP(tmp, random);
  res.x -= tmp;
  DOUBLE_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp = a.y;
  DOUBLE_BIT_FLIP(tmp, random);
  res.y = tmp;
  tmp = b;
  DOUBLE_BIT_FLIP(tmp, random);
  res.y -= tmp;
  DOUBLE_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp = a.z;
  DOUBLE_BIT_FLIP(tmp, random);
  res.z = tmp;
  tmp = b;
  DOUBLE_BIT_FLIP(tmp, random);
  res.z -= tmp;
  DOUBLE_BIT_FLIP(res.z, random);

  if (_cadna_cancel_tag){
    if (a.accuracy==DIGIT_NOT_COMPUTED || a.accuracy==RELIABLE_RESULT)
	a.nb_significant_digit();
    if (a.accuracy >= 	res.nb_significant_digit()+_cadna_cancel_value)
	instability(&_cadna_cancel_count);
  }
  return res;
}
inline double_st operator-(const double_st& a, const unsigned char& b){
  double_st res;

  unsigned int random;
  double tmp;

  res.accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = a.x;
  DOUBLE_BIT_FLIP(tmp, random);
  res.x = tmp;
  tmp = b;
  DOUBLE_BIT_FLIP(tmp, random);
  res.x -= tmp;
  DOUBLE_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp = a.y;
  DOUBLE_BIT_FLIP(tmp, random);
  res.y = tmp;
  tmp = b;
  DOUBLE_BIT_FLIP(tmp, random);
  res.y -= tmp;
  DOUBLE_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp = a.z;
  DOUBLE_BIT_FLIP(tmp, random);
  res.z = tmp;
  tmp = b;
  DOUBLE_BIT_FLIP(tmp, random);
  res.z -= tmp;
  DOUBLE_BIT_FLIP(res.z, random);

  if (_cadna_cancel_tag){
    if (a.accuracy==DIGIT_NOT_COMPUTED || a.accuracy==RELIABLE_RESULT)
	a.nb_significant_digit();
    if (a.accuracy >= 	res.nb_significant_digit()+_cadna_cancel_value)
	instability(&_cadna_cancel_count);
  }
  return res;
}
inline double_st operator-(const double_st& a, const char& b){
  double_st res;

  unsigned int random;
  double tmp;

  res.accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = a.x;
  DOUBLE_BIT_FLIP(tmp, random);
  res.x = tmp;
  tmp = b;
  DOUBLE_BIT_FLIP(tmp, random);
  res.x -= tmp;
  DOUBLE_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp = a.y;
  DOUBLE_BIT_FLIP(tmp, random);
  res.y = tmp;
  tmp = b;
  DOUBLE_BIT_FLIP(tmp, random);
  res.y -= tmp;
  DOUBLE_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp = a.z;
  DOUBLE_BIT_FLIP(tmp, random);
  res.z = tmp;
  tmp = b;
  DOUBLE_BIT_FLIP(tmp, random);
  res.z -= tmp;
  DOUBLE_BIT_FLIP(res.z, random);

  if (_cadna_cancel_tag){
    if (a.accuracy==DIGIT_NOT_COMPUTED || a.accuracy==RELIABLE_RESULT)
	a.nb_significant_digit();
    if (a.accuracy >= 	res.nb_significant_digit()+_cadna_cancel_value)
	instability(&_cadna_cancel_count);
  }
  return res;
}
#ifdef CADNA_HALF_EMULATION
inline double_st operator-(const double_st& a, const half& b){
  double_st res;

  unsigned int random;
  double tmp;

  res.accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = a.x;
  DOUBLE_BIT_FLIP(tmp, random);
  res.x = tmp;
  tmp = b;
  DOUBLE_BIT_FLIP(tmp, random);
  res.x -= tmp;
  DOUBLE_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp = a.y;
  DOUBLE_BIT_FLIP(tmp, random);
  res.y = tmp;
  tmp = b;
  DOUBLE_BIT_FLIP(tmp, random);
  res.y -= tmp;
  DOUBLE_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp = a.z;
  DOUBLE_BIT_FLIP(tmp, random);
  res.z = tmp;
  tmp = b;
  DOUBLE_BIT_FLIP(tmp, random);
  res.z -= tmp;
  DOUBLE_BIT_FLIP(res.z, random);

  if (_cadna_cancel_tag){
    if (a.accuracy==DIGIT_NOT_COMPUTED || a.accuracy==RELIABLE_RESULT)
	a.nb_significant_digit();
    if (a.accuracy >= 	res.nb_significant_digit()+_cadna_cancel_value)
	instability(&_cadna_cancel_count);
  }
  return res;
}
#endif // CADNA_HALF_EMULATION
#ifdef CADNA_HALF_NATIVE
inline double_st operator-(const double_st& a, const half& b){
  double_st res;

  unsigned int random;
  double tmp;

  res.accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = a.x;
  DOUBLE_BIT_FLIP(tmp, random);
  res.x = tmp;
  tmp = b;
  DOUBLE_BIT_FLIP(tmp, random);
  res.x -= tmp;
  DOUBLE_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp = a.y;
  DOUBLE_BIT_FLIP(tmp, random);
  res.y = tmp;
  tmp = b;
  DOUBLE_BIT_FLIP(tmp, random);
  res.y -= tmp;
  DOUBLE_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp = a.z;
  DOUBLE_BIT_FLIP(tmp, random);
  res.z = tmp;
  tmp = b;
  DOUBLE_BIT_FLIP(tmp, random);
  res.z -= tmp;
  DOUBLE_BIT_FLIP(res.z, random);

  if (_cadna_cancel_tag){
    if (a.accuracy==DIGIT_NOT_COMPUTED || a.accuracy==RELIABLE_RESULT)
	a.nb_significant_digit();
    if (a.accuracy >= 	res.nb_significant_digit()+_cadna_cancel_value)
	instability(&_cadna_cancel_count);
  }
  return res;
}
#endif

inline float_st operator-(const float_st& a, const double& b){
  float_st res;

  unsigned int random;
  float tmp;

  res.accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = a.x;
  FLOAT_BIT_FLIP(tmp, random);
  res.x = tmp;
  tmp = b;
  FLOAT_BIT_FLIP(tmp, random);
  res.x -= tmp;
  FLOAT_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp = a.y;
  FLOAT_BIT_FLIP(tmp, random);
  res.y = tmp;
  tmp = b;
  FLOAT_BIT_FLIP(tmp, random);
  res.y -= tmp;
  FLOAT_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp = a.z;
  FLOAT_BIT_FLIP(tmp, random);
  res.z = tmp;
  tmp = b;
  FLOAT_BIT_FLIP(tmp, random);
  res.z -= tmp;
  FLOAT_BIT_FLIP(res.z, random);

  if (_cadna_cancel_tag){
    if (a.accuracy==DIGIT_NOT_COMPUTED || a.accuracy==RELIABLE_RESULT)
	a.nb_significant_digit();
    if (a.accuracy >= 	res.nb_significant_digit()+_cadna_cancel_value)
	instability(&_cadna_cancel_count);
  }
  return res;
}
inline float_st operator-(const float_st& a, const float& b){
  float_st res;

  unsigned int random;
  float tmp;

  res.accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = a.x;
  FLOAT_BIT_FLIP(tmp, random);
  res.x = tmp;
  tmp = b;
  FLOAT_BIT_FLIP(tmp, random);
  res.x -= tmp;
  FLOAT_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp = a.y;
  FLOAT_BIT_FLIP(tmp, random);
  res.y = tmp;
  tmp = b;
  FLOAT_BIT_FLIP(tmp, random);
  res.y -= tmp;
  FLOAT_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp = a.z;
  FLOAT_BIT_FLIP(tmp, random);
  res.z = tmp;
  tmp = b;
  FLOAT_BIT_FLIP(tmp, random);
  res.z -= tmp;
  FLOAT_BIT_FLIP(res.z, random);

  if (_cadna_cancel_tag){
    if (a.accuracy==DIGIT_NOT_COMPUTED || a.accuracy==RELIABLE_RESULT)
	a.nb_significant_digit();
    if (a.accuracy >= 	res.nb_significant_digit()+_cadna_cancel_value)
	instability(&_cadna_cancel_count);
  }
  return res;
}
inline float_st operator-(const float_st& a, const unsigned long long& b){
  float_st res;

  unsigned int random;
  float tmp;

  res.accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = a.x;
  FLOAT_BIT_FLIP(tmp, random);
  res.x = tmp;
  tmp = b;
  FLOAT_BIT_FLIP(tmp, random);
  res.x -= tmp;
  FLOAT_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp = a.y;
  FLOAT_BIT_FLIP(tmp, random);
  res.y = tmp;
  tmp = b;
  FLOAT_BIT_FLIP(tmp, random);
  res.y -= tmp;
  FLOAT_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp = a.z;
  FLOAT_BIT_FLIP(tmp, random);
  res.z = tmp;
  tmp = b;
  FLOAT_BIT_FLIP(tmp, random);
  res.z -= tmp;
  FLOAT_BIT_FLIP(res.z, random);

  if (_cadna_cancel_tag){
    if (a.accuracy==DIGIT_NOT_COMPUTED || a.accuracy==RELIABLE_RESULT)
	a.nb_significant_digit();
    if (a.accuracy >= 	res.nb_significant_digit()+_cadna_cancel_value)
	instability(&_cadna_cancel_count);
  }
  return res;
}
inline float_st operator-(const float_st& a, const long long& b){
  float_st res;

  unsigned int random;
  float tmp;

  res.accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = a.x;
  FLOAT_BIT_FLIP(tmp, random);
  res.x = tmp;
  tmp = b;
  FLOAT_BIT_FLIP(tmp, random);
  res.x -= tmp;
  FLOAT_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp = a.y;
  FLOAT_BIT_FLIP(tmp, random);
  res.y = tmp;
  tmp = b;
  FLOAT_BIT_FLIP(tmp, random);
  res.y -= tmp;
  FLOAT_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp = a.z;
  FLOAT_BIT_FLIP(tmp, random);
  res.z = tmp;
  tmp = b;
  FLOAT_BIT_FLIP(tmp, random);
  res.z -= tmp;
  FLOAT_BIT_FLIP(res.z, random);

  if (_cadna_cancel_tag){
    if (a.accuracy==DIGIT_NOT_COMPUTED || a.accuracy==RELIABLE_RESULT)
	a.nb_significant_digit();
    if (a.accuracy >= 	res.nb_significant_digit()+_cadna_cancel_value)
	instability(&_cadna_cancel_count);
  }
  return res;
}
inline float_st operator-(const float_st& a, const unsigned long& b){
  float_st res;

  unsigned int random;
  float tmp;

  res.accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = a.x;
  FLOAT_BIT_FLIP(tmp, random);
  res.x = tmp;
  tmp = b;
  FLOAT_BIT_FLIP(tmp, random);
  res.x -= tmp;
  FLOAT_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp = a.y;
  FLOAT_BIT_FLIP(tmp, random);
  res.y = tmp;
  tmp = b;
  FLOAT_BIT_FLIP(tmp, random);
  res.y -= tmp;
  FLOAT_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp = a.z;
  FLOAT_BIT_FLIP(tmp, random);
  res.z = tmp;
  tmp = b;
  FLOAT_BIT_FLIP(tmp, random);
  res.z -= tmp;
  FLOAT_BIT_FLIP(res.z, random);

  if (_cadna_cancel_tag){
    if (a.accuracy==DIGIT_NOT_COMPUTED || a.accuracy==RELIABLE_RESULT)
	a.nb_significant_digit();
    if (a.accuracy >= 	res.nb_significant_digit()+_cadna_cancel_value)
	instability(&_cadna_cancel_count);
  }
  return res;
}
inline float_st operator-(const float_st& a, const long& b){
  float_st res;

  unsigned int random;
  float tmp;

  res.accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = a.x;
  FLOAT_BIT_FLIP(tmp, random);
  res.x = tmp;
  tmp = b;
  FLOAT_BIT_FLIP(tmp, random);
  res.x -= tmp;
  FLOAT_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp = a.y;
  FLOAT_BIT_FLIP(tmp, random);
  res.y = tmp;
  tmp = b;
  FLOAT_BIT_FLIP(tmp, random);
  res.y -= tmp;
  FLOAT_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp = a.z;
  FLOAT_BIT_FLIP(tmp, random);
  res.z = tmp;
  tmp = b;
  FLOAT_BIT_FLIP(tmp, random);
  res.z -= tmp;
  FLOAT_BIT_FLIP(res.z, random);

  if (_cadna_cancel_tag){
    if (a.accuracy==DIGIT_NOT_COMPUTED || a.accuracy==RELIABLE_RESULT)
	a.nb_significant_digit();
    if (a.accuracy >= 	res.nb_significant_digit()+_cadna_cancel_value)
	instability(&_cadna_cancel_count);
  }
  return res;
}
inline float_st operator-(const float_st& a, const unsigned int & b){
  float_st res;

  unsigned int random;
  float tmp;

  res.accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = a.x;
  FLOAT_BIT_FLIP(tmp, random);
  res.x = tmp;
  tmp = b;
  FLOAT_BIT_FLIP(tmp, random);
  res.x -= tmp;
  FLOAT_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp = a.y;
  FLOAT_BIT_FLIP(tmp, random);
  res.y = tmp;
  tmp = b;
  FLOAT_BIT_FLIP(tmp, random);
  res.y -= tmp;
  FLOAT_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp = a.z;
  FLOAT_BIT_FLIP(tmp, random);
  res.z = tmp;
  tmp = b;
  FLOAT_BIT_FLIP(tmp, random);
  res.z -= tmp;
  FLOAT_BIT_FLIP(res.z, random);

  if (_cadna_cancel_tag){
    if (a.accuracy==DIGIT_NOT_COMPUTED || a.accuracy==RELIABLE_RESULT)
	a.nb_significant_digit();
    if (a.accuracy >= 	res.nb_significant_digit()+_cadna_cancel_value)
	instability(&_cadna_cancel_count);
  }
  return res;
}
inline float_st operator-(const float_st& a, const int & b){
  float_st res;

  unsigned int random;
  float tmp;

  res.accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = a.x;
  FLOAT_BIT_FLIP(tmp, random);
  res.x = tmp;
  tmp = b;
  FLOAT_BIT_FLIP(tmp, random);
  res.x -= tmp;
  FLOAT_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp = a.y;
  FLOAT_BIT_FLIP(tmp, random);
  res.y = tmp;
  tmp = b;
  FLOAT_BIT_FLIP(tmp, random);
  res.y -= tmp;
  FLOAT_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp = a.z;
  FLOAT_BIT_FLIP(tmp, random);
  res.z = tmp;
  tmp = b;
  FLOAT_BIT_FLIP(tmp, random);
  res.z -= tmp;
  FLOAT_BIT_FLIP(res.z, random);

  if (_cadna_cancel_tag){
    if (a.accuracy==DIGIT_NOT_COMPUTED || a.accuracy==RELIABLE_RESULT)
	a.nb_significant_digit();
    if (a.accuracy >= 	res.nb_significant_digit()+_cadna_cancel_value)
	instability(&_cadna_cancel_count);
  }
  return res;
}
inline float_st operator-(const float_st& a, const unsigned short& b){
  float_st res;

  unsigned int random;
  float tmp;

  res.accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = a.x;
  FLOAT_BIT_FLIP(tmp, random);
  res.x = tmp;
  tmp = b;
  FLOAT_BIT_FLIP(tmp, random);
  res.x -= tmp;
  FLOAT_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp = a.y;
  FLOAT_BIT_FLIP(tmp, random);
  res.y = tmp;
  tmp = b;
  FLOAT_BIT_FLIP(tmp, random);
  res.y -= tmp;
  FLOAT_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp = a.z;
  FLOAT_BIT_FLIP(tmp, random);
  res.z = tmp;
  tmp = b;
  FLOAT_BIT_FLIP(tmp, random);
  res.z -= tmp;
  FLOAT_BIT_FLIP(res.z, random);

  if (_cadna_cancel_tag){
    if (a.accuracy==DIGIT_NOT_COMPUTED || a.accuracy==RELIABLE_RESULT)
	a.nb_significant_digit();
    if (a.accuracy >= 	res.nb_significant_digit()+_cadna_cancel_value)
	instability(&_cadna_cancel_count);
  }
  return res;
}
inline float_st operator-(const float_st& a, const short& b){
  float_st res;

  unsigned int random;
  float tmp;

  res.accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = a.x;
  FLOAT_BIT_FLIP(tmp, random);
  res.x = tmp;
  tmp = b;
  FLOAT_BIT_FLIP(tmp, random);
  res.x -= tmp;
  FLOAT_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp = a.y;
  FLOAT_BIT_FLIP(tmp, random);
  res.y = tmp;
  tmp = b;
  FLOAT_BIT_FLIP(tmp, random);
  res.y -= tmp;
  FLOAT_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp = a.z;
  FLOAT_BIT_FLIP(tmp, random);
  res.z = tmp;
  tmp = b;
  FLOAT_BIT_FLIP(tmp, random);
  res.z -= tmp;
  FLOAT_BIT_FLIP(res.z, random);

  if (_cadna_cancel_tag){
    if (a.accuracy==DIGIT_NOT_COMPUTED || a.accuracy==RELIABLE_RESULT)
	a.nb_significant_digit();
    if (a.accuracy >= 	res.nb_significant_digit()+_cadna_cancel_value)
	instability(&_cadna_cancel_count);
  }
  return res;
}
inline float_st operator-(const float_st& a, const unsigned char& b){
  float_st res;

  unsigned int random;
  float tmp;

  res.accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = a.x;
  FLOAT_BIT_FLIP(tmp, random);
  res.x = tmp;
  tmp = b;
  FLOAT_BIT_FLIP(tmp, random);
  res.x -= tmp;
  FLOAT_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp = a.y;
  FLOAT_BIT_FLIP(tmp, random);
  res.y = tmp;
  tmp = b;
  FLOAT_BIT_FLIP(tmp, random);
  res.y -= tmp;
  FLOAT_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp = a.z;
  FLOAT_BIT_FLIP(tmp, random);
  res.z = tmp;
  tmp = b;
  FLOAT_BIT_FLIP(tmp, random);
  res.z -= tmp;
  FLOAT_BIT_FLIP(res.z, random);

  if (_cadna_cancel_tag){
    if (a.accuracy==DIGIT_NOT_COMPUTED || a.accuracy==RELIABLE_RESULT)
	a.nb_significant_digit();
    if (a.accuracy >= 	res.nb_significant_digit()+_cadna_cancel_value)
	instability(&_cadna_cancel_count);
  }
  return res;
}
inline float_st operator-(const float_st& a, const char& b){
  float_st res;

  unsigned int random;
  float tmp;

  res.accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = a.x;
  FLOAT_BIT_FLIP(tmp, random);
  res.x = tmp;
  tmp = b;
  FLOAT_BIT_FLIP(tmp, random);
  res.x -= tmp;
  FLOAT_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp = a.y;
  FLOAT_BIT_FLIP(tmp, random);
  res.y = tmp;
  tmp = b;
  FLOAT_BIT_FLIP(tmp, random);
  res.y -= tmp;
  FLOAT_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp = a.z;
  FLOAT_BIT_FLIP(tmp, random);
  res.z = tmp;
  tmp = b;
  FLOAT_BIT_FLIP(tmp, random);
  res.z -= tmp;
  FLOAT_BIT_FLIP(res.z, random);

  if (_cadna_cancel_tag){
    if (a.accuracy==DIGIT_NOT_COMPUTED || a.accuracy==RELIABLE_RESULT)
	a.nb_significant_digit();
    if (a.accuracy >= 	res.nb_significant_digit()+_cadna_cancel_value)
	instability(&_cadna_cancel_count);
  }
  return res;
}
#ifdef CADNA_HALF_EMULATION
inline float_st operator-(const float_st& a, const half& b){
  float_st res;

  unsigned int random;
  float tmp;

  res.accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = a.x;
  FLOAT_BIT_FLIP(tmp, random);
  res.x = tmp;
  tmp = b;
  FLOAT_BIT_FLIP(tmp, random);
  res.x -= tmp;
  FLOAT_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp = a.y;
  FLOAT_BIT_FLIP(tmp, random);
  res.y = tmp;
  tmp = b;
  FLOAT_BIT_FLIP(tmp, random);
  res.y -= tmp;
  FLOAT_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp = a.z;
  FLOAT_BIT_FLIP(tmp, random);
  res.z = tmp;
  tmp = b;
  FLOAT_BIT_FLIP(tmp, random);
  res.z -= tmp;
  FLOAT_BIT_FLIP(res.z, random);

  if (_cadna_cancel_tag){
    if (a.accuracy==DIGIT_NOT_COMPUTED || a.accuracy==RELIABLE_RESULT)
	a.nb_significant_digit();
    if (a.accuracy >= 	res.nb_significant_digit()+_cadna_cancel_value)
	instability(&_cadna_cancel_count);
  }
  return res;
}
#endif // CADNA_HALF_EMULATION	
#ifdef CADNA_HALF_NATIVE
inline float_st operator-(const float_st& a, const half& b){
  float_st res;

  unsigned int random;
  float tmp;

  res.accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = a.x;
  FLOAT_BIT_FLIP(tmp, random);
  res.x = tmp;
  tmp = b;
  FLOAT_BIT_FLIP(tmp, random);
  res.x -= tmp;
  FLOAT_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp = a.y;
  FLOAT_BIT_FLIP(tmp, random);
  res.y = tmp;
  tmp = b;
  FLOAT_BIT_FLIP(tmp, random);
  res.y -= tmp;
  FLOAT_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp = a.z;
  FLOAT_BIT_FLIP(tmp, random);
  res.z = tmp;
  tmp = b;
  FLOAT_BIT_FLIP(tmp, random);
  res.z -= tmp;
  FLOAT_BIT_FLIP(res.z, random);

  if (_cadna_cancel_tag){
    if (a.accuracy==DIGIT_NOT_COMPUTED || a.accuracy==RELIABLE_RESULT)
	a.nb_significant_digit();
    if (a.accuracy >= 	res.nb_significant_digit()+_cadna_cancel_value)
	instability(&_cadna_cancel_count);
  }
  return res;
}
#endif

#ifdef CADNA_HALF_EMULATION
inline half_st operator-(const half_st& a, const double& b){
  half_st res;
  double tmp2;
  unsigned int random;
  half tmp;

  res.accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp2=(random) ? -b : b;
  tmp = a.x;
  HALF_BIT_FLIP(tmp, random);
  res.x = tmp;
  res.x -= tmp2;
  HALF_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp2=(random) ? -b : b;
  tmp = a.y;
  HALF_BIT_FLIP(tmp, random);
  res.y = tmp;
  res.y -= tmp2;
  HALF_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp2=(random) ? -b : b;
  tmp = a.z;
  HALF_BIT_FLIP(tmp, random);
  res.z = tmp;
  res.z -= tmp2;
  HALF_BIT_FLIP(res.z, random);

  
  
  if (_cadna_cancel_tag){
    if (a.accuracy==DIGIT_NOT_COMPUTED || a.accuracy==RELIABLE_RESULT)
	a.nb_significant_digit();
    if (a.accuracy >= 	res.nb_significant_digit()+_cadna_cancel_value)
	instability(&_cadna_cancel_count);
  }
  if (_cadna_half_overflow_tag){
  if (std::isinf(res.x) or std::isinf(res.y) or std::isinf(res.z)){
  instability(&_cadna_half_overflow_count);
  }
}



  return res;
}
inline half_st operator-(const half_st& a, const float& b){
  half_st res;
  float tmp2;
  unsigned int random;
  half tmp;

  res.accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp2=(random) ? -b : b;
  tmp = a.x;
  HALF_BIT_FLIP(tmp, random);
  res.x = tmp;
  res.x -= tmp2;
  HALF_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp2=(random) ? -b : b;
  tmp = a.y;
  HALF_BIT_FLIP(tmp, random);
  res.y = tmp;
  res.y -= tmp2;
  HALF_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp2=(random) ? -b : b;
  tmp = a.z;
  HALF_BIT_FLIP(tmp, random);
  res.z = tmp;
  res.z -= tmp2;
  HALF_BIT_FLIP(res.z, random);

  
  
  if (_cadna_cancel_tag){
    if (a.accuracy==DIGIT_NOT_COMPUTED || a.accuracy==RELIABLE_RESULT)
	a.nb_significant_digit();
    if (a.accuracy >= 	res.nb_significant_digit()+_cadna_cancel_value)
	instability(&_cadna_cancel_count);
  }
  if (_cadna_half_overflow_tag){
  if (std::isinf(res.x) or std::isinf(res.y) or std::isinf(res.z)){
  instability(&_cadna_half_overflow_count);
  }
}



  return res;
}
inline half_st operator-(const half_st& a, const unsigned long long& b){
  half_st res;
  unsigned long long tmp2;
  unsigned int random;
  half tmp;

  res.accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp2=(random) ? -b : b;
  tmp = a.x;
  HALF_BIT_FLIP(tmp, random);
  res.x = tmp;
  res.x -= tmp2;
  HALF_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp2=(random) ? -b : b;
  tmp = a.y;
  HALF_BIT_FLIP(tmp, random);
  res.y = tmp;
  res.y -= tmp2;
  HALF_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp2=(random) ? -b : b;
  tmp = a.z;
  HALF_BIT_FLIP(tmp, random);
  res.z = tmp;
  res.z -= tmp2;
  HALF_BIT_FLIP(res.z, random);

  
  
  if (_cadna_cancel_tag){
    if (a.accuracy==DIGIT_NOT_COMPUTED || a.accuracy==RELIABLE_RESULT)
	a.nb_significant_digit();
    if (a.accuracy >= 	res.nb_significant_digit()+_cadna_cancel_value)
	instability(&_cadna_cancel_count);
  }
  if (_cadna_half_overflow_tag){
  if (std::isinf(res.x) or std::isinf(res.y) or std::isinf(res.z)){
  instability(&_cadna_half_overflow_count);
  }
}



  return res;
}
inline half_st operator-(const half_st& a, const long long& b){
  half_st res;
  long long tmp2;
  unsigned int random;
  half tmp;

  res.accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp2=(random) ? -b : b;
  tmp = a.x;
  HALF_BIT_FLIP(tmp, random);
  res.x = tmp;
  res.x -= tmp2;
  HALF_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp2=(random) ? -b : b;
  tmp = a.y;
  HALF_BIT_FLIP(tmp, random);
  res.y = tmp;
  res.y -= tmp2;
  HALF_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp2=(random) ? -b : b;
  tmp = a.z;
  HALF_BIT_FLIP(tmp, random);
  res.z = tmp;
  res.z -= tmp2;
  HALF_BIT_FLIP(res.z, random);

  
  
  if (_cadna_cancel_tag){
    if (a.accuracy==DIGIT_NOT_COMPUTED || a.accuracy==RELIABLE_RESULT)
	a.nb_significant_digit();
    if (a.accuracy >= 	res.nb_significant_digit()+_cadna_cancel_value)
	instability(&_cadna_cancel_count);
  }
  if (_cadna_half_overflow_tag){
  if (std::isinf(res.x) or std::isinf(res.y) or std::isinf(res.z)){
  instability(&_cadna_half_overflow_count);
  }
}



  return res;
}
inline half_st operator-(const half_st& a, const unsigned long& b){
  half_st res;
  unsigned long tmp2;
  unsigned int random;
  half tmp;

  res.accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp2=(random) ? -b : b;
  tmp = a.x;
  HALF_BIT_FLIP(tmp, random);
  res.x = tmp;
  res.x -= tmp2;
  HALF_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp2=(random) ? -b : b;
  tmp = a.y;
  HALF_BIT_FLIP(tmp, random);
  res.y = tmp;
  res.y -= tmp2;
  HALF_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp2=(random) ? -b : b;
  tmp = a.z;
  HALF_BIT_FLIP(tmp, random);
  res.z = tmp;
  res.z -= tmp2;
  HALF_BIT_FLIP(res.z, random);

  
  
  if (_cadna_cancel_tag){
    if (a.accuracy==DIGIT_NOT_COMPUTED || a.accuracy==RELIABLE_RESULT)
	a.nb_significant_digit();
    if (a.accuracy >= 	res.nb_significant_digit()+_cadna_cancel_value)
	instability(&_cadna_cancel_count);
  }
  if (_cadna_half_overflow_tag){
  if (std::isinf(res.x) or std::isinf(res.y) or std::isinf(res.z)){
  instability(&_cadna_half_overflow_count);
  }
}



  return res;
}
inline half_st operator-(const half_st& a, const long& b){
  half_st res;
  long tmp2;
  unsigned int random;
  half tmp;

  res.accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp2=(random) ? -b : b;
  tmp = a.x;
  HALF_BIT_FLIP(tmp, random);
  res.x = tmp;
  res.x -= tmp2;
  HALF_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp2=(random) ? -b : b;
  tmp = a.y;
  HALF_BIT_FLIP(tmp, random);
  res.y = tmp;
  res.y -= tmp2;
  HALF_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp2=(random) ? -b : b;
  tmp = a.z;
  HALF_BIT_FLIP(tmp, random);
  res.z = tmp;
  res.z -= tmp2;
  HALF_BIT_FLIP(res.z, random);

  
  
  if (_cadna_cancel_tag){
    if (a.accuracy==DIGIT_NOT_COMPUTED || a.accuracy==RELIABLE_RESULT)
	a.nb_significant_digit();
    if (a.accuracy >= 	res.nb_significant_digit()+_cadna_cancel_value)
	instability(&_cadna_cancel_count);
  }
  if (_cadna_half_overflow_tag){
  if (std::isinf(res.x) or std::isinf(res.y) or std::isinf(res.z)){
  instability(&_cadna_half_overflow_count);
  }
}



  return res;
}
inline half_st operator-(const half_st& a, const unsigned int & b){
  half_st res;
  unsigned int  tmp2;
  unsigned int random;
  half tmp;

  res.accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp2=(random) ? -b : b;
  tmp = a.x;
  HALF_BIT_FLIP(tmp, random);
  res.x = tmp;
  res.x -= tmp2;
  HALF_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp2=(random) ? -b : b;
  tmp = a.y;
  HALF_BIT_FLIP(tmp, random);
  res.y = tmp;
  res.y -= tmp2;
  HALF_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp2=(random) ? -b : b;
  tmp = a.z;
  HALF_BIT_FLIP(tmp, random);
  res.z = tmp;
  res.z -= tmp2;
  HALF_BIT_FLIP(res.z, random);

  
  
  if (_cadna_cancel_tag){
    if (a.accuracy==DIGIT_NOT_COMPUTED || a.accuracy==RELIABLE_RESULT)
	a.nb_significant_digit();
    if (a.accuracy >= 	res.nb_significant_digit()+_cadna_cancel_value)
	instability(&_cadna_cancel_count);
  }
  if (_cadna_half_overflow_tag){
  if (std::isinf(res.x) or std::isinf(res.y) or std::isinf(res.z)){
  instability(&_cadna_half_overflow_count);
  }
}



  return res;
}
inline half_st operator-(const half_st& a, const int & b){
  half_st res;
  int  tmp2;
  unsigned int random;
  half tmp;

  res.accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp2=(random) ? -b : b;
  tmp = a.x;
  HALF_BIT_FLIP(tmp, random);
  res.x = tmp;
  res.x -= tmp2;
  HALF_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp2=(random) ? -b : b;
  tmp = a.y;
  HALF_BIT_FLIP(tmp, random);
  res.y = tmp;
  res.y -= tmp2;
  HALF_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp2=(random) ? -b : b;
  tmp = a.z;
  HALF_BIT_FLIP(tmp, random);
  res.z = tmp;
  res.z -= tmp2;
  HALF_BIT_FLIP(res.z, random);

  
  
  if (_cadna_cancel_tag){
    if (a.accuracy==DIGIT_NOT_COMPUTED || a.accuracy==RELIABLE_RESULT)
	a.nb_significant_digit();
    if (a.accuracy >= 	res.nb_significant_digit()+_cadna_cancel_value)
	instability(&_cadna_cancel_count);
  }
  if (_cadna_half_overflow_tag){
  if (std::isinf(res.x) or std::isinf(res.y) or std::isinf(res.z)){
  instability(&_cadna_half_overflow_count);
  }
}



  return res;
}
inline half_st operator-(const half_st& a, const unsigned short& b){
  half_st res;
  unsigned short tmp2;
  unsigned int random;
  half tmp;

  res.accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp2=(random) ? -b : b;
  tmp = a.x;
  HALF_BIT_FLIP(tmp, random);
  res.x = tmp;
  res.x -= tmp2;
  HALF_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp2=(random) ? -b : b;
  tmp = a.y;
  HALF_BIT_FLIP(tmp, random);
  res.y = tmp;
  res.y -= tmp2;
  HALF_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp2=(random) ? -b : b;
  tmp = a.z;
  HALF_BIT_FLIP(tmp, random);
  res.z = tmp;
  res.z -= tmp2;
  HALF_BIT_FLIP(res.z, random);

  
  
  if (_cadna_cancel_tag){
    if (a.accuracy==DIGIT_NOT_COMPUTED || a.accuracy==RELIABLE_RESULT)
	a.nb_significant_digit();
    if (a.accuracy >= 	res.nb_significant_digit()+_cadna_cancel_value)
	instability(&_cadna_cancel_count);
  }
  if (_cadna_half_overflow_tag){
  if (std::isinf(res.x) or std::isinf(res.y) or std::isinf(res.z)){
  instability(&_cadna_half_overflow_count);
  }
}



  return res;
}
inline half_st operator-(const half_st& a, const short& b){
  half_st res;
  short tmp2;
  unsigned int random;
  half tmp;

  res.accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp2=(random) ? -b : b;
  tmp = a.x;
  HALF_BIT_FLIP(tmp, random);
  res.x = tmp;
  res.x -= tmp2;
  HALF_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp2=(random) ? -b : b;
  tmp = a.y;
  HALF_BIT_FLIP(tmp, random);
  res.y = tmp;
  res.y -= tmp2;
  HALF_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp2=(random) ? -b : b;
  tmp = a.z;
  HALF_BIT_FLIP(tmp, random);
  res.z = tmp;
  res.z -= tmp2;
  HALF_BIT_FLIP(res.z, random);

  
  
  if (_cadna_cancel_tag){
    if (a.accuracy==DIGIT_NOT_COMPUTED || a.accuracy==RELIABLE_RESULT)
	a.nb_significant_digit();
    if (a.accuracy >= 	res.nb_significant_digit()+_cadna_cancel_value)
	instability(&_cadna_cancel_count);
  }
  if (_cadna_half_overflow_tag){
  if (std::isinf(res.x) or std::isinf(res.y) or std::isinf(res.z)){
  instability(&_cadna_half_overflow_count);
  }
}



  return res;
}
inline half_st operator-(const half_st& a, const unsigned char& b){
  half_st res;
  unsigned char tmp2;
  unsigned int random;
  half tmp;

  res.accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp2=(random) ? -b : b;
  tmp = a.x;
  HALF_BIT_FLIP(tmp, random);
  res.x = tmp;
  res.x -= tmp2;
  HALF_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp2=(random) ? -b : b;
  tmp = a.y;
  HALF_BIT_FLIP(tmp, random);
  res.y = tmp;
  res.y -= tmp2;
  HALF_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp2=(random) ? -b : b;
  tmp = a.z;
  HALF_BIT_FLIP(tmp, random);
  res.z = tmp;
  res.z -= tmp2;
  HALF_BIT_FLIP(res.z, random);

  
  
  if (_cadna_cancel_tag){
    if (a.accuracy==DIGIT_NOT_COMPUTED || a.accuracy==RELIABLE_RESULT)
	a.nb_significant_digit();
    if (a.accuracy >= 	res.nb_significant_digit()+_cadna_cancel_value)
	instability(&_cadna_cancel_count);
  }
  if (_cadna_half_overflow_tag){
  if (std::isinf(res.x) or std::isinf(res.y) or std::isinf(res.z)){
  instability(&_cadna_half_overflow_count);
  }
}



  return res;
}
inline half_st operator-(const half_st& a, const char& b){
  half_st res;
  char tmp2;
  unsigned int random;
  half tmp;

  res.accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp2=(random) ? -b : b;
  tmp = a.x;
  HALF_BIT_FLIP(tmp, random);
  res.x = tmp;
  res.x -= tmp2;
  HALF_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp2=(random) ? -b : b;
  tmp = a.y;
  HALF_BIT_FLIP(tmp, random);
  res.y = tmp;
  res.y -= tmp2;
  HALF_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp2=(random) ? -b : b;
  tmp = a.z;
  HALF_BIT_FLIP(tmp, random);
  res.z = tmp;
  res.z -= tmp2;
  HALF_BIT_FLIP(res.z, random);

  
  
  if (_cadna_cancel_tag){
    if (a.accuracy==DIGIT_NOT_COMPUTED || a.accuracy==RELIABLE_RESULT)
	a.nb_significant_digit();
    if (a.accuracy >= 	res.nb_significant_digit()+_cadna_cancel_value)
	instability(&_cadna_cancel_count);
  }
  if (_cadna_half_overflow_tag){
  if (std::isinf(res.x) or std::isinf(res.y) or std::isinf(res.z)){
  instability(&_cadna_half_overflow_count);
  }
}



  return res;
}
inline half_st operator-(const half_st& a, const half& b){
  half_st res;
  half tmp2;
  unsigned int random;
  half tmp;

  res.accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp2=(random) ? -b : b;
  tmp = a.x;
  HALF_BIT_FLIP(tmp, random);
  res.x = tmp;
  res.x -= tmp2;
  HALF_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp2=(random) ? -b : b;
  tmp = a.y;
  HALF_BIT_FLIP(tmp, random);
  res.y = tmp;
  res.y -= tmp2;
  HALF_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp2=(random) ? -b : b;
  tmp = a.z;
  HALF_BIT_FLIP(tmp, random);
  res.z = tmp;
  res.z -= tmp2;
  HALF_BIT_FLIP(res.z, random);

  
  
  if (_cadna_cancel_tag){
    if (a.accuracy==DIGIT_NOT_COMPUTED || a.accuracy==RELIABLE_RESULT)
	a.nb_significant_digit();
    if (a.accuracy >= 	res.nb_significant_digit()+_cadna_cancel_value)
	instability(&_cadna_cancel_count);
  }
  if (_cadna_half_overflow_tag){
  if (std::isinf(res.x) or std::isinf(res.y) or std::isinf(res.z)){
  instability(&_cadna_half_overflow_count);
  }
}



  return res;
}
#endif // CADNA_HALF_EMULATION

#ifdef CADNA_HALF_NATIVE
inline half_st operator-(const half_st& a, const double& b){
  half_st res;
  double tmp2;
  unsigned int random;
  half tmp;

  res.accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp2=(random) ? -b : b;
  tmp = a.x;
  HALF_BIT_FLIP(tmp, random);
  res.x = tmp;
  res.x -= tmp2;
  HALF_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp2=(random) ? -b : b;
  tmp = a.y;
  HALF_BIT_FLIP(tmp, random);
  res.y = tmp;
  res.y -= tmp2;
  HALF_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp2=(random) ? -b : b;
  tmp = a.z;
  HALF_BIT_FLIP(tmp, random);
  res.z = tmp;
  res.z -= tmp2;
  HALF_BIT_FLIP(res.z, random);

  
  
  if (_cadna_cancel_tag){
    if (a.accuracy==DIGIT_NOT_COMPUTED || a.accuracy==RELIABLE_RESULT)
	a.nb_significant_digit();
    if (a.accuracy >= 	res.nb_significant_digit()+_cadna_cancel_value)
	instability(&_cadna_cancel_count);
  }
  if (_cadna_half_overflow_tag){
  if (std::isinf(res.x) or std::isinf(res.y) or std::isinf(res.z)){
  instability(&_cadna_half_overflow_count);
  }
}



  return res;
}
inline half_st operator-(const half_st& a, const float& b){
  half_st res;
  float tmp2;
  unsigned int random;
  half tmp;

  res.accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp2=(random) ? -b : b;
  tmp = a.x;
  HALF_BIT_FLIP(tmp, random);
  res.x = tmp;
  res.x -= tmp2;
  HALF_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp2=(random) ? -b : b;
  tmp = a.y;
  HALF_BIT_FLIP(tmp, random);
  res.y = tmp;
  res.y -= tmp2;
  HALF_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp2=(random) ? -b : b;
  tmp = a.z;
  HALF_BIT_FLIP(tmp, random);
  res.z = tmp;
  res.z -= tmp2;
  HALF_BIT_FLIP(res.z, random);

  
  
  if (_cadna_cancel_tag){
    if (a.accuracy==DIGIT_NOT_COMPUTED || a.accuracy==RELIABLE_RESULT)
	a.nb_significant_digit();
    if (a.accuracy >= 	res.nb_significant_digit()+_cadna_cancel_value)
	instability(&_cadna_cancel_count);
  }
  if (_cadna_half_overflow_tag){
  if (std::isinf(res.x) or std::isinf(res.y) or std::isinf(res.z)){
  instability(&_cadna_half_overflow_count);
  }
}



  return res;
}
inline half_st operator-(const half_st& a, const unsigned long long& b){
  half_st res;
  unsigned long long tmp2;
  unsigned int random;
  half tmp;

  res.accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp2=(random) ? -b : b;
  tmp = a.x;
  HALF_BIT_FLIP(tmp, random);
  res.x = tmp;
  res.x -= tmp2;
  HALF_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp2=(random) ? -b : b;
  tmp = a.y;
  HALF_BIT_FLIP(tmp, random);
  res.y = tmp;
  res.y -= tmp2;
  HALF_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp2=(random) ? -b : b;
  tmp = a.z;
  HALF_BIT_FLIP(tmp, random);
  res.z = tmp;
  res.z -= tmp2;
  HALF_BIT_FLIP(res.z, random);

  
  
  if (_cadna_cancel_tag){
    if (a.accuracy==DIGIT_NOT_COMPUTED || a.accuracy==RELIABLE_RESULT)
	a.nb_significant_digit();
    if (a.accuracy >= 	res.nb_significant_digit()+_cadna_cancel_value)
	instability(&_cadna_cancel_count);
  }
  if (_cadna_half_overflow_tag){
  if (std::isinf(res.x) or std::isinf(res.y) or std::isinf(res.z)){
  instability(&_cadna_half_overflow_count);
  }
}



  return res;
}
inline half_st operator-(const half_st& a, const long long& b){
  half_st res;
  long long tmp2;
  unsigned int random;
  half tmp;

  res.accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp2=(random) ? -b : b;
  tmp = a.x;
  HALF_BIT_FLIP(tmp, random);
  res.x = tmp;
  res.x -= tmp2;
  HALF_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp2=(random) ? -b : b;
  tmp = a.y;
  HALF_BIT_FLIP(tmp, random);
  res.y = tmp;
  res.y -= tmp2;
  HALF_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp2=(random) ? -b : b;
  tmp = a.z;
  HALF_BIT_FLIP(tmp, random);
  res.z = tmp;
  res.z -= tmp2;
  HALF_BIT_FLIP(res.z, random);

  
  
  if (_cadna_cancel_tag){
    if (a.accuracy==DIGIT_NOT_COMPUTED || a.accuracy==RELIABLE_RESULT)
	a.nb_significant_digit();
    if (a.accuracy >= 	res.nb_significant_digit()+_cadna_cancel_value)
	instability(&_cadna_cancel_count);
  }
  if (_cadna_half_overflow_tag){
  if (std::isinf(res.x) or std::isinf(res.y) or std::isinf(res.z)){
  instability(&_cadna_half_overflow_count);
  }
}



  return res;
}
inline half_st operator-(const half_st& a, const unsigned long& b){
  half_st res;
  unsigned long tmp2;
  unsigned int random;
  half tmp;

  res.accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp2=(random) ? -b : b;
  tmp = a.x;
  HALF_BIT_FLIP(tmp, random);
  res.x = tmp;
  res.x -= tmp2;
  HALF_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp2=(random) ? -b : b;
  tmp = a.y;
  HALF_BIT_FLIP(tmp, random);
  res.y = tmp;
  res.y -= tmp2;
  HALF_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp2=(random) ? -b : b;
  tmp = a.z;
  HALF_BIT_FLIP(tmp, random);
  res.z = tmp;
  res.z -= tmp2;
  HALF_BIT_FLIP(res.z, random);

  
  
  if (_cadna_cancel_tag){
    if (a.accuracy==DIGIT_NOT_COMPUTED || a.accuracy==RELIABLE_RESULT)
	a.nb_significant_digit();
    if (a.accuracy >= 	res.nb_significant_digit()+_cadna_cancel_value)
	instability(&_cadna_cancel_count);
  }
  if (_cadna_half_overflow_tag){
  if (std::isinf(res.x) or std::isinf(res.y) or std::isinf(res.z)){
  instability(&_cadna_half_overflow_count);
  }
}



  return res;
}
inline half_st operator-(const half_st& a, const long& b){
  half_st res;
  long tmp2;
  unsigned int random;
  half tmp;

  res.accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp2=(random) ? -b : b;
  tmp = a.x;
  HALF_BIT_FLIP(tmp, random);
  res.x = tmp;
  res.x -= tmp2;
  HALF_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp2=(random) ? -b : b;
  tmp = a.y;
  HALF_BIT_FLIP(tmp, random);
  res.y = tmp;
  res.y -= tmp2;
  HALF_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp2=(random) ? -b : b;
  tmp = a.z;
  HALF_BIT_FLIP(tmp, random);
  res.z = tmp;
  res.z -= tmp2;
  HALF_BIT_FLIP(res.z, random);

  
  
  if (_cadna_cancel_tag){
    if (a.accuracy==DIGIT_NOT_COMPUTED || a.accuracy==RELIABLE_RESULT)
	a.nb_significant_digit();
    if (a.accuracy >= 	res.nb_significant_digit()+_cadna_cancel_value)
	instability(&_cadna_cancel_count);
  }
  if (_cadna_half_overflow_tag){
  if (std::isinf(res.x) or std::isinf(res.y) or std::isinf(res.z)){
  instability(&_cadna_half_overflow_count);
  }
}



  return res;
}
inline half_st operator-(const half_st& a, const unsigned int & b){
  half_st res;
  unsigned int  tmp2;
  unsigned int random;
  half tmp;

  res.accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp2=(random) ? -b : b;
  tmp = a.x;
  HALF_BIT_FLIP(tmp, random);
  res.x = tmp;
  res.x -= tmp2;
  HALF_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp2=(random) ? -b : b;
  tmp = a.y;
  HALF_BIT_FLIP(tmp, random);
  res.y = tmp;
  res.y -= tmp2;
  HALF_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp2=(random) ? -b : b;
  tmp = a.z;
  HALF_BIT_FLIP(tmp, random);
  res.z = tmp;
  res.z -= tmp2;
  HALF_BIT_FLIP(res.z, random);

  
  
  if (_cadna_cancel_tag){
    if (a.accuracy==DIGIT_NOT_COMPUTED || a.accuracy==RELIABLE_RESULT)
	a.nb_significant_digit();
    if (a.accuracy >= 	res.nb_significant_digit()+_cadna_cancel_value)
	instability(&_cadna_cancel_count);
  }
  if (_cadna_half_overflow_tag){
  if (std::isinf(res.x) or std::isinf(res.y) or std::isinf(res.z)){
  instability(&_cadna_half_overflow_count);
  }
}



  return res;
}
inline half_st operator-(const half_st& a, const int & b){
  half_st res;
  int  tmp2;
  unsigned int random;
  half tmp;

  res.accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp2=(random) ? -b : b;
  tmp = a.x;
  HALF_BIT_FLIP(tmp, random);
  res.x = tmp;
  res.x -= tmp2;
  HALF_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp2=(random) ? -b : b;
  tmp = a.y;
  HALF_BIT_FLIP(tmp, random);
  res.y = tmp;
  res.y -= tmp2;
  HALF_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp2=(random) ? -b : b;
  tmp = a.z;
  HALF_BIT_FLIP(tmp, random);
  res.z = tmp;
  res.z -= tmp2;
  HALF_BIT_FLIP(res.z, random);

  
  
  if (_cadna_cancel_tag){
    if (a.accuracy==DIGIT_NOT_COMPUTED || a.accuracy==RELIABLE_RESULT)
	a.nb_significant_digit();
    if (a.accuracy >= 	res.nb_significant_digit()+_cadna_cancel_value)
	instability(&_cadna_cancel_count);
  }
  if (_cadna_half_overflow_tag){
  if (std::isinf(res.x) or std::isinf(res.y) or std::isinf(res.z)){
  instability(&_cadna_half_overflow_count);
  }
}



  return res;
}
inline half_st operator-(const half_st& a, const unsigned short& b){
  half_st res;
  unsigned short tmp2;
  unsigned int random;
  half tmp;

  res.accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp2=(random) ? -b : b;
  tmp = a.x;
  HALF_BIT_FLIP(tmp, random);
  res.x = tmp;
  res.x -= tmp2;
  HALF_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp2=(random) ? -b : b;
  tmp = a.y;
  HALF_BIT_FLIP(tmp, random);
  res.y = tmp;
  res.y -= tmp2;
  HALF_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp2=(random) ? -b : b;
  tmp = a.z;
  HALF_BIT_FLIP(tmp, random);
  res.z = tmp;
  res.z -= tmp2;
  HALF_BIT_FLIP(res.z, random);

  
  
  if (_cadna_cancel_tag){
    if (a.accuracy==DIGIT_NOT_COMPUTED || a.accuracy==RELIABLE_RESULT)
	a.nb_significant_digit();
    if (a.accuracy >= 	res.nb_significant_digit()+_cadna_cancel_value)
	instability(&_cadna_cancel_count);
  }
  if (_cadna_half_overflow_tag){
  if (std::isinf(res.x) or std::isinf(res.y) or std::isinf(res.z)){
  instability(&_cadna_half_overflow_count);
  }
}



  return res;
}
inline half_st operator-(const half_st& a, const short& b){
  half_st res;
  short tmp2;
  unsigned int random;
  half tmp;

  res.accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp2=(random) ? -b : b;
  tmp = a.x;
  HALF_BIT_FLIP(tmp, random);
  res.x = tmp;
  res.x -= tmp2;
  HALF_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp2=(random) ? -b : b;
  tmp = a.y;
  HALF_BIT_FLIP(tmp, random);
  res.y = tmp;
  res.y -= tmp2;
  HALF_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp2=(random) ? -b : b;
  tmp = a.z;
  HALF_BIT_FLIP(tmp, random);
  res.z = tmp;
  res.z -= tmp2;
  HALF_BIT_FLIP(res.z, random);

  
  
  if (_cadna_cancel_tag){
    if (a.accuracy==DIGIT_NOT_COMPUTED || a.accuracy==RELIABLE_RESULT)
	a.nb_significant_digit();
    if (a.accuracy >= 	res.nb_significant_digit()+_cadna_cancel_value)
	instability(&_cadna_cancel_count);
  }
  if (_cadna_half_overflow_tag){
  if (std::isinf(res.x) or std::isinf(res.y) or std::isinf(res.z)){
  instability(&_cadna_half_overflow_count);
  }
}



  return res;
}
inline half_st operator-(const half_st& a, const unsigned char& b){
  half_st res;
  unsigned char tmp2;
  unsigned int random;
  half tmp;

  res.accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp2=(random) ? -b : b;
  tmp = a.x;
  HALF_BIT_FLIP(tmp, random);
  res.x = tmp;
  res.x -= tmp2;
  HALF_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp2=(random) ? -b : b;
  tmp = a.y;
  HALF_BIT_FLIP(tmp, random);
  res.y = tmp;
  res.y -= tmp2;
  HALF_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp2=(random) ? -b : b;
  tmp = a.z;
  HALF_BIT_FLIP(tmp, random);
  res.z = tmp;
  res.z -= tmp2;
  HALF_BIT_FLIP(res.z, random);

  
  
  if (_cadna_cancel_tag){
    if (a.accuracy==DIGIT_NOT_COMPUTED || a.accuracy==RELIABLE_RESULT)
	a.nb_significant_digit();
    if (a.accuracy >= 	res.nb_significant_digit()+_cadna_cancel_value)
	instability(&_cadna_cancel_count);
  }
  if (_cadna_half_overflow_tag){
  if (std::isinf(res.x) or std::isinf(res.y) or std::isinf(res.z)){
  instability(&_cadna_half_overflow_count);
  }
}



  return res;
}
inline half_st operator-(const half_st& a, const char& b){
  half_st res;
  char tmp2;
  unsigned int random;
  half tmp;

  res.accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp2=(random) ? -b : b;
  tmp = a.x;
  HALF_BIT_FLIP(tmp, random);
  res.x = tmp;
  res.x -= tmp2;
  HALF_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp2=(random) ? -b : b;
  tmp = a.y;
  HALF_BIT_FLIP(tmp, random);
  res.y = tmp;
  res.y -= tmp2;
  HALF_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp2=(random) ? -b : b;
  tmp = a.z;
  HALF_BIT_FLIP(tmp, random);
  res.z = tmp;
  res.z -= tmp2;
  HALF_BIT_FLIP(res.z, random);

  
  
  if (_cadna_cancel_tag){
    if (a.accuracy==DIGIT_NOT_COMPUTED || a.accuracy==RELIABLE_RESULT)
	a.nb_significant_digit();
    if (a.accuracy >= 	res.nb_significant_digit()+_cadna_cancel_value)
	instability(&_cadna_cancel_count);
  }
  if (_cadna_half_overflow_tag){
  if (std::isinf(res.x) or std::isinf(res.y) or std::isinf(res.z)){
  instability(&_cadna_half_overflow_count);
  }
}



  return res;
}
inline half_st operator-(const half_st& a, const half& b){
  half_st res;
  half tmp2;
  unsigned int random;
  half tmp;

  res.accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp2=(random) ? -b : b;
  tmp = a.x;
  HALF_BIT_FLIP(tmp, random);
  res.x = tmp;
  res.x -= tmp2;
  HALF_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp2=(random) ? -b : b;
  tmp = a.y;
  HALF_BIT_FLIP(tmp, random);
  res.y = tmp;
  res.y -= tmp2;
  HALF_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp2=(random) ? -b : b;
  tmp = a.z;
  HALF_BIT_FLIP(tmp, random);
  res.z = tmp;
  res.z -= tmp2;
  HALF_BIT_FLIP(res.z, random);

  
  
  if (_cadna_cancel_tag){
    if (a.accuracy==DIGIT_NOT_COMPUTED || a.accuracy==RELIABLE_RESULT)
	a.nb_significant_digit();
    if (a.accuracy >= 	res.nb_significant_digit()+_cadna_cancel_value)
	instability(&_cadna_cancel_count);
  }
  if (_cadna_half_overflow_tag){
  if (std::isinf(res.x) or std::isinf(res.y) or std::isinf(res.z)){
  instability(&_cadna_half_overflow_count);
  }
}



  return res;
}
#endif

/////////////////////////////////////////////////////////////////////////





inline double_st operator-(const double_st& a, const double_st& b){
  double_st res;

  unsigned int random;
  double tmp;

  res.accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = a.x;
  DOUBLE_BIT_FLIP(tmp, random);
  res.x = tmp;
  tmp = b.x;
  DOUBLE_BIT_FLIP(tmp, random);
  res.x -= tmp;
  DOUBLE_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp = a.y;
  DOUBLE_BIT_FLIP(tmp, random);
  res.y = tmp;
  tmp = b.y;
  DOUBLE_BIT_FLIP(tmp, random);
  res.y -= tmp;
  DOUBLE_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp = a.z;
  DOUBLE_BIT_FLIP(tmp, random);
  res.z = tmp;
  tmp = b.z;
  DOUBLE_BIT_FLIP(tmp, random);
  res.z -= tmp;
  DOUBLE_BIT_FLIP(res.z, random);

  if (_cadna_cancel_tag){
    if (a.accuracy==DIGIT_NOT_COMPUTED || a.accuracy==RELIABLE_RESULT)
	a.nb_significant_digit();
    if (b.accuracy==DIGIT_NOT_COMPUTED || b.accuracy==RELIABLE_RESULT)
	b.nb_significant_digit();
    if ((a.accuracy < b.accuracy ? a.accuracy: b.accuracy) >=
	  res.nb_significant_digit()+_cadna_cancel_value)
	instability(&_cadna_cancel_count);
  }
  return res;
}
inline double_st operator-(const double_st& a, const float_st& b){
  double_st res;

  unsigned int random;
  double tmp;

  res.accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = a.x;
  DOUBLE_BIT_FLIP(tmp, random);
  res.x = tmp;
  tmp = b.x;
  DOUBLE_BIT_FLIP(tmp, random);
  res.x -= tmp;
  DOUBLE_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp = a.y;
  DOUBLE_BIT_FLIP(tmp, random);
  res.y = tmp;
  tmp = b.y;
  DOUBLE_BIT_FLIP(tmp, random);
  res.y -= tmp;
  DOUBLE_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp = a.z;
  DOUBLE_BIT_FLIP(tmp, random);
  res.z = tmp;
  tmp = b.z;
  DOUBLE_BIT_FLIP(tmp, random);
  res.z -= tmp;
  DOUBLE_BIT_FLIP(res.z, random);

  if (_cadna_cancel_tag){
    if (a.accuracy==DIGIT_NOT_COMPUTED || a.accuracy==RELIABLE_RESULT)
	a.nb_significant_digit();
    if (b.accuracy==DIGIT_NOT_COMPUTED || b.accuracy==RELIABLE_RESULT)
	b.nb_significant_digit();
    if ((a.accuracy < b.accuracy ? a.accuracy: b.accuracy) >=
	  res.nb_significant_digit()+_cadna_cancel_value)
	instability(&_cadna_cancel_count);
  }
  return res;
}
inline double_st operator-(const float_st& a, const double_st& b){
  double_st res;

  unsigned int random;
  double tmp;

  res.accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = a.x;
  DOUBLE_BIT_FLIP(tmp, random);
  res.x = tmp;
  tmp = b.x;
  DOUBLE_BIT_FLIP(tmp, random);
  res.x -= tmp;
  DOUBLE_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp = a.y;
  DOUBLE_BIT_FLIP(tmp, random);
  res.y = tmp;
  tmp = b.y;
  DOUBLE_BIT_FLIP(tmp, random);
  res.y -= tmp;
  DOUBLE_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp = a.z;
  DOUBLE_BIT_FLIP(tmp, random);
  res.z = tmp;
  tmp = b.z;
  DOUBLE_BIT_FLIP(tmp, random);
  res.z -= tmp;
  DOUBLE_BIT_FLIP(res.z, random);

  if (_cadna_cancel_tag){
    if (a.accuracy==DIGIT_NOT_COMPUTED || a.accuracy==RELIABLE_RESULT)
	a.nb_significant_digit();
    if (b.accuracy==DIGIT_NOT_COMPUTED || b.accuracy==RELIABLE_RESULT)
	b.nb_significant_digit();
    if ((a.accuracy < b.accuracy ? a.accuracy: b.accuracy) >=
	  res.nb_significant_digit()+_cadna_cancel_value)
	instability(&_cadna_cancel_count);
  }
  return res;
}
inline float_st operator-(const float_st& a, const float_st& b){
  float_st res;

  unsigned int random;
  float tmp;

  res.accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = a.x;
  FLOAT_BIT_FLIP(tmp, random);
  res.x = tmp;
  tmp = b.x;
  FLOAT_BIT_FLIP(tmp, random);
  res.x -= tmp;
  FLOAT_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp = a.y;
  FLOAT_BIT_FLIP(tmp, random);
  res.y = tmp;
  tmp = b.y;
  FLOAT_BIT_FLIP(tmp, random);
  res.y -= tmp;
  FLOAT_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp = a.z;
  FLOAT_BIT_FLIP(tmp, random);
  res.z = tmp;
  tmp = b.z;
  FLOAT_BIT_FLIP(tmp, random);
  res.z -= tmp;
  FLOAT_BIT_FLIP(res.z, random);

  if (_cadna_cancel_tag){
    if (a.accuracy==DIGIT_NOT_COMPUTED || a.accuracy==RELIABLE_RESULT)
	a.nb_significant_digit();
    if (b.accuracy==DIGIT_NOT_COMPUTED || b.accuracy==RELIABLE_RESULT)
	b.nb_significant_digit();
    if ((a.accuracy < b.accuracy ? a.accuracy: b.accuracy) >=
	  res.nb_significant_digit()+_cadna_cancel_value)
	instability(&_cadna_cancel_count);
  }
  return res;
}

#ifdef CADNA_HALF_EMULATION
inline half_st operator-(const half_st& a, const half_st& b){
  half_st res;

  unsigned int random;
  half tmp;

  res.accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = a.x;
  HALF_BIT_FLIP(tmp, random);
  res.x = tmp;
  tmp = b.x;
  HALF_BIT_FLIP(tmp, random);
  res.x -= tmp;
  HALF_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp = a.y;
  HALF_BIT_FLIP(tmp, random);
  res.y = tmp;
  tmp = b.y;
  HALF_BIT_FLIP(tmp, random);
  res.y -= tmp;
  HALF_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp = a.z;
  HALF_BIT_FLIP(tmp, random);
  res.z = tmp;
  tmp = b.z;
  HALF_BIT_FLIP(tmp, random);
  res.z -= tmp;
  HALF_BIT_FLIP(res.z, random);

  
  
  if (_cadna_cancel_tag){
    if (a.accuracy==DIGIT_NOT_COMPUTED || a.accuracy==RELIABLE_RESULT)
	a.nb_significant_digit();
    if (b.accuracy==DIGIT_NOT_COMPUTED || b.accuracy==RELIABLE_RESULT)
	b.nb_significant_digit();
    if ((a.accuracy < b.accuracy ? a.accuracy: b.accuracy) >=
	  res.nb_significant_digit()+_cadna_cancel_value)
	instability(&_cadna_cancel_count);
  }
  if (_cadna_half_overflow_tag){
  if (std::isinf(res.x) or std::isinf(res.y) or std::isinf(res.z)){
  instability(&_cadna_half_overflow_count);
  }
}

  if (_cadna_half_underflow_tag){
  /*
  double val = res.x + res.y + res.z;
  double val_a = a.x+a.y+a.z;
  double val_b = b.x+b.y+b.z;
  if ( (val==0) &&  (val_a!=val_b) ){
  */
  //res is zero but a is different from b
  if ( ( (res.x==0) || (res.y==0) || (res.z==0) )  &&  (a.x!=b.x) ){
  instability(&_cadna_half_underflow_count);
  }
}

  return res;
}
inline double_st operator-(const half_st& a, const double_st& b){
  double_st res;

  unsigned int random;
  double tmp;

  res.accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = a.x;
  DOUBLE_BIT_FLIP(tmp, random);
  res.x = tmp;
  tmp = b.x;
  DOUBLE_BIT_FLIP(tmp, random);
  res.x -= tmp;
  DOUBLE_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp = a.y;
  DOUBLE_BIT_FLIP(tmp, random);
  res.y = tmp;
  tmp = b.y;
  DOUBLE_BIT_FLIP(tmp, random);
  res.y -= tmp;
  DOUBLE_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp = a.z;
  DOUBLE_BIT_FLIP(tmp, random);
  res.z = tmp;
  tmp = b.z;
  DOUBLE_BIT_FLIP(tmp, random);
  res.z -= tmp;
  DOUBLE_BIT_FLIP(res.z, random);

  
  
  if (_cadna_cancel_tag){
    if (a.accuracy==DIGIT_NOT_COMPUTED || a.accuracy==RELIABLE_RESULT)
	a.nb_significant_digit();
    if (b.accuracy==DIGIT_NOT_COMPUTED || b.accuracy==RELIABLE_RESULT)
	b.nb_significant_digit();
    if ((a.accuracy < b.accuracy ? a.accuracy: b.accuracy) >=
	  res.nb_significant_digit()+_cadna_cancel_value)
	instability(&_cadna_cancel_count);
  }
  if (_cadna_half_overflow_tag){
  if (std::isinf(res.x) or std::isinf(res.y) or std::isinf(res.z)){
  instability(&_cadna_half_overflow_count);
  }
}

  if (_cadna_half_underflow_tag){
  /*
  double val = res.x + res.y + res.z;
  double val_a = a.x+a.y+a.z;
  double val_b = b.x+b.y+b.z;
  if ( (val==0) &&  (val_a!=val_b) ){
  */
  //res is zero but a is different from b
  if ( ( (res.x==0) || (res.y==0) || (res.z==0) )  &&  (a.x!=b.x) ){
  instability(&_cadna_half_underflow_count);
  }
}

  return res;
}
inline double_st operator-(const double_st& a, const half_st& b){
  double_st res;

  unsigned int random;
  double tmp;

  res.accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = a.x;
  DOUBLE_BIT_FLIP(tmp, random);
  res.x = tmp;
  tmp = b.x;
  DOUBLE_BIT_FLIP(tmp, random);
  res.x -= tmp;
  DOUBLE_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp = a.y;
  DOUBLE_BIT_FLIP(tmp, random);
  res.y = tmp;
  tmp = b.y;
  DOUBLE_BIT_FLIP(tmp, random);
  res.y -= tmp;
  DOUBLE_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp = a.z;
  DOUBLE_BIT_FLIP(tmp, random);
  res.z = tmp;
  tmp = b.z;
  DOUBLE_BIT_FLIP(tmp, random);
  res.z -= tmp;
  DOUBLE_BIT_FLIP(res.z, random);

  
  
  if (_cadna_cancel_tag){
    if (a.accuracy==DIGIT_NOT_COMPUTED || a.accuracy==RELIABLE_RESULT)
	a.nb_significant_digit();
    if (b.accuracy==DIGIT_NOT_COMPUTED || b.accuracy==RELIABLE_RESULT)
	b.nb_significant_digit();
    if ((a.accuracy < b.accuracy ? a.accuracy: b.accuracy) >=
	  res.nb_significant_digit()+_cadna_cancel_value)
	instability(&_cadna_cancel_count);
  }
  if (_cadna_half_overflow_tag){
  if (std::isinf(res.x) or std::isinf(res.y) or std::isinf(res.z)){
  instability(&_cadna_half_overflow_count);
  }
}

  if (_cadna_half_underflow_tag){
  /*
  double val = res.x + res.y + res.z;
  double val_a = a.x+a.y+a.z;
  double val_b = b.x+b.y+b.z;
  if ( (val==0) &&  (val_a!=val_b) ){
  */
  //res is zero but a is different from b
  if ( ( (res.x==0) || (res.y==0) || (res.z==0) )  &&  (a.x!=b.x) ){
  instability(&_cadna_half_underflow_count);
  }
}

  return res;
}
inline float_st operator-(const half_st& a, const float_st& b){
  float_st res;

  unsigned int random;
  float tmp;

  res.accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = a.x;
  FLOAT_BIT_FLIP(tmp, random);
  res.x = tmp;
  tmp = b.x;
  FLOAT_BIT_FLIP(tmp, random);
  res.x -= tmp;
  FLOAT_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp = a.y;
  FLOAT_BIT_FLIP(tmp, random);
  res.y = tmp;
  tmp = b.y;
  FLOAT_BIT_FLIP(tmp, random);
  res.y -= tmp;
  FLOAT_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp = a.z;
  FLOAT_BIT_FLIP(tmp, random);
  res.z = tmp;
  tmp = b.z;
  FLOAT_BIT_FLIP(tmp, random);
  res.z -= tmp;
  FLOAT_BIT_FLIP(res.z, random);

  
  
  if (_cadna_cancel_tag){
    if (a.accuracy==DIGIT_NOT_COMPUTED || a.accuracy==RELIABLE_RESULT)
	a.nb_significant_digit();
    if (b.accuracy==DIGIT_NOT_COMPUTED || b.accuracy==RELIABLE_RESULT)
	b.nb_significant_digit();
    if ((a.accuracy < b.accuracy ? a.accuracy: b.accuracy) >=
	  res.nb_significant_digit()+_cadna_cancel_value)
	instability(&_cadna_cancel_count);
  }
  if (_cadna_half_overflow_tag){
  if (std::isinf(res.x) or std::isinf(res.y) or std::isinf(res.z)){
  instability(&_cadna_half_overflow_count);
  }
}

  if (_cadna_half_underflow_tag){
  /*
  double val = res.x + res.y + res.z;
  double val_a = a.x+a.y+a.z;
  double val_b = b.x+b.y+b.z;
  if ( (val==0) &&  (val_a!=val_b) ){
  */
  //res is zero but a is different from b
  if ( ( (res.x==0) || (res.y==0) || (res.z==0) )  &&  (a.x!=b.x) ){
  instability(&_cadna_half_underflow_count);
  }
}

  return res;
}
inline float_st operator-(const float_st& a, const half_st& b){
  float_st res;

  unsigned int random;
  float tmp;

  res.accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = a.x;
  FLOAT_BIT_FLIP(tmp, random);
  res.x = tmp;
  tmp = b.x;
  FLOAT_BIT_FLIP(tmp, random);
  res.x -= tmp;
  FLOAT_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp = a.y;
  FLOAT_BIT_FLIP(tmp, random);
  res.y = tmp;
  tmp = b.y;
  FLOAT_BIT_FLIP(tmp, random);
  res.y -= tmp;
  FLOAT_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp = a.z;
  FLOAT_BIT_FLIP(tmp, random);
  res.z = tmp;
  tmp = b.z;
  FLOAT_BIT_FLIP(tmp, random);
  res.z -= tmp;
  FLOAT_BIT_FLIP(res.z, random);

  
  
  if (_cadna_cancel_tag){
    if (a.accuracy==DIGIT_NOT_COMPUTED || a.accuracy==RELIABLE_RESULT)
	a.nb_significant_digit();
    if (b.accuracy==DIGIT_NOT_COMPUTED || b.accuracy==RELIABLE_RESULT)
	b.nb_significant_digit();
    if ((a.accuracy < b.accuracy ? a.accuracy: b.accuracy) >=
	  res.nb_significant_digit()+_cadna_cancel_value)
	instability(&_cadna_cancel_count);
  }
  if (_cadna_half_overflow_tag){
  if (std::isinf(res.x) or std::isinf(res.y) or std::isinf(res.z)){
  instability(&_cadna_half_overflow_count);
  }
}

  if (_cadna_half_underflow_tag){
  /*
  double val = res.x + res.y + res.z;
  double val_a = a.x+a.y+a.z;
  double val_b = b.x+b.y+b.z;
  if ( (val==0) &&  (val_a!=val_b) ){
  */
  //res is zero but a is different from b
  if ( ( (res.x==0) || (res.y==0) || (res.z==0) )  &&  (a.x!=b.x) ){
  instability(&_cadna_half_underflow_count);
  }
}

  return res;
}
#endif

#ifdef CADNA_HALF_NATIVE
inline half_st operator-(const half_st& a, const half_st& b){
  half_st res;

  unsigned int random;
  half tmp;

  res.accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = a.x;
  HALF_BIT_FLIP(tmp, random);
  res.x = tmp;
  tmp = b.x;
  HALF_BIT_FLIP(tmp, random);
  res.x -= tmp;
  HALF_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp = a.y;
  HALF_BIT_FLIP(tmp, random);
  res.y = tmp;
  tmp = b.y;
  HALF_BIT_FLIP(tmp, random);
  res.y -= tmp;
  HALF_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp = a.z;
  HALF_BIT_FLIP(tmp, random);
  res.z = tmp;
  tmp = b.z;
  HALF_BIT_FLIP(tmp, random);
  res.z -= tmp;
  HALF_BIT_FLIP(res.z, random);

  
  
  if (_cadna_cancel_tag){
    if (a.accuracy==DIGIT_NOT_COMPUTED || a.accuracy==RELIABLE_RESULT)
	a.nb_significant_digit();
    if (b.accuracy==DIGIT_NOT_COMPUTED || b.accuracy==RELIABLE_RESULT)
	b.nb_significant_digit();
    if ((a.accuracy < b.accuracy ? a.accuracy: b.accuracy) >=
	  res.nb_significant_digit()+_cadna_cancel_value)
	instability(&_cadna_cancel_count);
  }
  if (_cadna_half_overflow_tag){
  if (std::isinf(res.x) or std::isinf(res.y) or std::isinf(res.z)){
  instability(&_cadna_half_overflow_count);
  }
}

  if (_cadna_half_underflow_tag){
  /*
  double val = res.x + res.y + res.z;
  double val_a = a.x+a.y+a.z;
  double val_b = b.x+b.y+b.z;
  if ( (val==0) &&  (val_a!=val_b) ){
  */
  //res is zero but a is different from b
  if ( ( (res.x==0) || (res.y==0) || (res.z==0) )  &&  (a.x!=b.x) ){
  instability(&_cadna_half_underflow_count);
  }
}

  return res;
}
inline double_st operator-(const half_st& a, const double_st& b){
  double_st res;

  unsigned int random;
  double tmp;

  res.accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = a.x;
  DOUBLE_BIT_FLIP(tmp, random);
  res.x = tmp;
  tmp = b.x;
  DOUBLE_BIT_FLIP(tmp, random);
  res.x -= tmp;
  DOUBLE_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp = a.y;
  DOUBLE_BIT_FLIP(tmp, random);
  res.y = tmp;
  tmp = b.y;
  DOUBLE_BIT_FLIP(tmp, random);
  res.y -= tmp;
  DOUBLE_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp = a.z;
  DOUBLE_BIT_FLIP(tmp, random);
  res.z = tmp;
  tmp = b.z;
  DOUBLE_BIT_FLIP(tmp, random);
  res.z -= tmp;
  DOUBLE_BIT_FLIP(res.z, random);

  
  
  if (_cadna_cancel_tag){
    if (a.accuracy==DIGIT_NOT_COMPUTED || a.accuracy==RELIABLE_RESULT)
	a.nb_significant_digit();
    if (b.accuracy==DIGIT_NOT_COMPUTED || b.accuracy==RELIABLE_RESULT)
	b.nb_significant_digit();
    if ((a.accuracy < b.accuracy ? a.accuracy: b.accuracy) >=
	  res.nb_significant_digit()+_cadna_cancel_value)
	instability(&_cadna_cancel_count);
  }
  if (_cadna_half_overflow_tag){
  if (std::isinf(res.x) or std::isinf(res.y) or std::isinf(res.z)){
  instability(&_cadna_half_overflow_count);
  }
}

  if (_cadna_half_underflow_tag){
  /*
  double val = res.x + res.y + res.z;
  double val_a = a.x+a.y+a.z;
  double val_b = b.x+b.y+b.z;
  if ( (val==0) &&  (val_a!=val_b) ){
  */
  //res is zero but a is different from b
  if ( ( (res.x==0) || (res.y==0) || (res.z==0) )  &&  (a.x!=b.x) ){
  instability(&_cadna_half_underflow_count);
  }
}

  return res;
}
inline double_st operator-(const double_st& a, const half_st& b){
  double_st res;

  unsigned int random;
  double tmp;

  res.accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = a.x;
  DOUBLE_BIT_FLIP(tmp, random);
  res.x = tmp;
  tmp = b.x;
  DOUBLE_BIT_FLIP(tmp, random);
  res.x -= tmp;
  DOUBLE_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp = a.y;
  DOUBLE_BIT_FLIP(tmp, random);
  res.y = tmp;
  tmp = b.y;
  DOUBLE_BIT_FLIP(tmp, random);
  res.y -= tmp;
  DOUBLE_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp = a.z;
  DOUBLE_BIT_FLIP(tmp, random);
  res.z = tmp;
  tmp = b.z;
  DOUBLE_BIT_FLIP(tmp, random);
  res.z -= tmp;
  DOUBLE_BIT_FLIP(res.z, random);

  
  
  if (_cadna_cancel_tag){
    if (a.accuracy==DIGIT_NOT_COMPUTED || a.accuracy==RELIABLE_RESULT)
	a.nb_significant_digit();
    if (b.accuracy==DIGIT_NOT_COMPUTED || b.accuracy==RELIABLE_RESULT)
	b.nb_significant_digit();
    if ((a.accuracy < b.accuracy ? a.accuracy: b.accuracy) >=
	  res.nb_significant_digit()+_cadna_cancel_value)
	instability(&_cadna_cancel_count);
  }
  if (_cadna_half_overflow_tag){
  if (std::isinf(res.x) or std::isinf(res.y) or std::isinf(res.z)){
  instability(&_cadna_half_overflow_count);
  }
}

  if (_cadna_half_underflow_tag){
  /*
  double val = res.x + res.y + res.z;
  double val_a = a.x+a.y+a.z;
  double val_b = b.x+b.y+b.z;
  if ( (val==0) &&  (val_a!=val_b) ){
  */
  //res is zero but a is different from b
  if ( ( (res.x==0) || (res.y==0) || (res.z==0) )  &&  (a.x!=b.x) ){
  instability(&_cadna_half_underflow_count);
  }
}

  return res;
}
inline float_st operator-(const half_st& a, const float_st& b){
  float_st res;

  unsigned int random;
  float tmp;

  res.accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = a.x;
  FLOAT_BIT_FLIP(tmp, random);
  res.x = tmp;
  tmp = b.x;
  FLOAT_BIT_FLIP(tmp, random);
  res.x -= tmp;
  FLOAT_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp = a.y;
  FLOAT_BIT_FLIP(tmp, random);
  res.y = tmp;
  tmp = b.y;
  FLOAT_BIT_FLIP(tmp, random);
  res.y -= tmp;
  FLOAT_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp = a.z;
  FLOAT_BIT_FLIP(tmp, random);
  res.z = tmp;
  tmp = b.z;
  FLOAT_BIT_FLIP(tmp, random);
  res.z -= tmp;
  FLOAT_BIT_FLIP(res.z, random);

  
  
  if (_cadna_cancel_tag){
    if (a.accuracy==DIGIT_NOT_COMPUTED || a.accuracy==RELIABLE_RESULT)
	a.nb_significant_digit();
    if (b.accuracy==DIGIT_NOT_COMPUTED || b.accuracy==RELIABLE_RESULT)
	b.nb_significant_digit();
    if ((a.accuracy < b.accuracy ? a.accuracy: b.accuracy) >=
	  res.nb_significant_digit()+_cadna_cancel_value)
	instability(&_cadna_cancel_count);
  }
  if (_cadna_half_overflow_tag){
  if (std::isinf(res.x) or std::isinf(res.y) or std::isinf(res.z)){
  instability(&_cadna_half_overflow_count);
  }
}

  if (_cadna_half_underflow_tag){
  /*
  double val = res.x + res.y + res.z;
  double val_a = a.x+a.y+a.z;
  double val_b = b.x+b.y+b.z;
  if ( (val==0) &&  (val_a!=val_b) ){
  */
  //res is zero but a is different from b
  if ( ( (res.x==0) || (res.y==0) || (res.z==0) )  &&  (a.x!=b.x) ){
  instability(&_cadna_half_underflow_count);
  }
}

  return res;
}
inline float_st operator-(const float_st& a, const half_st& b){
  float_st res;

  unsigned int random;
  float tmp;

  res.accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = a.x;
  FLOAT_BIT_FLIP(tmp, random);
  res.x = tmp;
  tmp = b.x;
  FLOAT_BIT_FLIP(tmp, random);
  res.x -= tmp;
  FLOAT_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp = a.y;
  FLOAT_BIT_FLIP(tmp, random);
  res.y = tmp;
  tmp = b.y;
  FLOAT_BIT_FLIP(tmp, random);
  res.y -= tmp;
  FLOAT_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp = a.z;
  FLOAT_BIT_FLIP(tmp, random);
  res.z = tmp;
  tmp = b.z;
  FLOAT_BIT_FLIP(tmp, random);
  res.z -= tmp;
  FLOAT_BIT_FLIP(res.z, random);

  
  
  if (_cadna_cancel_tag){
    if (a.accuracy==DIGIT_NOT_COMPUTED || a.accuracy==RELIABLE_RESULT)
	a.nb_significant_digit();
    if (b.accuracy==DIGIT_NOT_COMPUTED || b.accuracy==RELIABLE_RESULT)
	b.nb_significant_digit();
    if ((a.accuracy < b.accuracy ? a.accuracy: b.accuracy) >=
	  res.nb_significant_digit()+_cadna_cancel_value)
	instability(&_cadna_cancel_count);
  }
  if (_cadna_half_overflow_tag){
  if (std::isinf(res.x) or std::isinf(res.y) or std::isinf(res.z)){
  instability(&_cadna_half_overflow_count);
  }
}

  if (_cadna_half_underflow_tag){
  /*
  double val = res.x + res.y + res.z;
  double val_a = a.x+a.y+a.z;
  double val_b = b.x+b.y+b.z;
  if ( (val==0) &&  (val_a!=val_b) ){
  */
  //res is zero but a is different from b
  if ( ( (res.x==0) || (res.y==0) || (res.z==0) )  &&  (a.x!=b.x) ){
  instability(&_cadna_half_underflow_count);
  }
}

  return res;
}
#endif
