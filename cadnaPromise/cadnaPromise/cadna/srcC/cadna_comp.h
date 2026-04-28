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


#include "cadna.h"





//////////////////////////////////////////////////
//****m* cadna_comp/sub
//    NAME
//      add
//    SYNOPSIS
//      res = a - b
//    FUNCTION
//    Defines all the functions involving at least one
//    stochastic argument which compute "a-b" statement 
//    without instability detection
//
//
//    INPUTS
//      a           - an integer, a float, a double or a stochastic number
//      b           - an integer, a float, a double or a stochastic number
//      At least one argument must be of stochastic type.
//
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




inline double_st sub(const double_st& a, const double_st& b){
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

  return res;
}
inline double_st sub(const double_st& a, const float_st& b){
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

  return res;
}
inline double_st sub(const float_st& a, const double_st& b){
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

  return res;
}
inline float_st sub(const float_st& a, const float_st& b){
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

  return res;
}
#ifdef CADNA_QUAD
inline float128_st sub(const float128_st& a, const float128_st& b){
  float128_st res;

  unsigned int random;
  float128 tmp;

  res.accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = a.x;
  FLOAT128_BIT_FLIP(tmp, random);
  res.x = tmp;
  tmp = b.x;
  FLOAT128_BIT_FLIP(tmp, random);
  res.x -= tmp;
  FLOAT128_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp = a.y;
  FLOAT128_BIT_FLIP(tmp, random);
  res.y = tmp;
  tmp = b.y;
  FLOAT128_BIT_FLIP(tmp, random);
  res.y -= tmp;
  FLOAT128_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp = a.z;
  FLOAT128_BIT_FLIP(tmp, random);
  res.z = tmp;
  tmp = b.z;
  FLOAT128_BIT_FLIP(tmp, random);
  res.z -= tmp;
  FLOAT128_BIT_FLIP(res.z, random);

  return res;
}
inline float128_st sub(const float128_st& a, const double_st& b){
  float128_st res;

  unsigned int random;
  float128 tmp;

  res.accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = a.x;
  FLOAT128_BIT_FLIP(tmp, random);
  res.x = tmp;
  tmp = b.x;
  FLOAT128_BIT_FLIP(tmp, random);
  res.x -= tmp;
  FLOAT128_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp = a.y;
  FLOAT128_BIT_FLIP(tmp, random);
  res.y = tmp;
  tmp = b.y;
  FLOAT128_BIT_FLIP(tmp, random);
  res.y -= tmp;
  FLOAT128_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp = a.z;
  FLOAT128_BIT_FLIP(tmp, random);
  res.z = tmp;
  tmp = b.z;
  FLOAT128_BIT_FLIP(tmp, random);
  res.z -= tmp;
  FLOAT128_BIT_FLIP(res.z, random);

  return res;
}
inline float128_st sub(const double_st& a, const float128_st& b){
  float128_st res;

  unsigned int random;
  float128 tmp;

  res.accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = a.x;
  FLOAT128_BIT_FLIP(tmp, random);
  res.x = tmp;
  tmp = b.x;
  FLOAT128_BIT_FLIP(tmp, random);
  res.x -= tmp;
  FLOAT128_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp = a.y;
  FLOAT128_BIT_FLIP(tmp, random);
  res.y = tmp;
  tmp = b.y;
  FLOAT128_BIT_FLIP(tmp, random);
  res.y -= tmp;
  FLOAT128_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp = a.z;
  FLOAT128_BIT_FLIP(tmp, random);
  res.z = tmp;
  tmp = b.z;
  FLOAT128_BIT_FLIP(tmp, random);
  res.z -= tmp;
  FLOAT128_BIT_FLIP(res.z, random);

  return res;
}
inline float128_st sub(const float128_st& a, const float_st& b){
  float128_st res;

  unsigned int random;
  float128 tmp;

  res.accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = a.x;
  FLOAT128_BIT_FLIP(tmp, random);
  res.x = tmp;
  tmp = b.x;
  FLOAT128_BIT_FLIP(tmp, random);
  res.x -= tmp;
  FLOAT128_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp = a.y;
  FLOAT128_BIT_FLIP(tmp, random);
  res.y = tmp;
  tmp = b.y;
  FLOAT128_BIT_FLIP(tmp, random);
  res.y -= tmp;
  FLOAT128_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp = a.z;
  FLOAT128_BIT_FLIP(tmp, random);
  res.z = tmp;
  tmp = b.z;
  FLOAT128_BIT_FLIP(tmp, random);
  res.z -= tmp;
  FLOAT128_BIT_FLIP(res.z, random);

  return res;
}
inline float128_st sub(const float_st& a, const float128_st& b){
  float128_st res;

  unsigned int random;
  float128 tmp;

  res.accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = a.x;
  FLOAT128_BIT_FLIP(tmp, random);
  res.x = tmp;
  tmp = b.x;
  FLOAT128_BIT_FLIP(tmp, random);
  res.x -= tmp;
  FLOAT128_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp = a.y;
  FLOAT128_BIT_FLIP(tmp, random);
  res.y = tmp;
  tmp = b.y;
  FLOAT128_BIT_FLIP(tmp, random);
  res.y -= tmp;
  FLOAT128_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp = a.z;
  FLOAT128_BIT_FLIP(tmp, random);
  res.z = tmp;
  tmp = b.z;
  FLOAT128_BIT_FLIP(tmp, random);
  res.z -= tmp;
  FLOAT128_BIT_FLIP(res.z, random);

  return res;
}
#endif

/////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////
//****m* cadna_comp/add
//    NAME
//      add
//    SYNOPSIS
//      res = a + b
//    FUNCTION
//    Defines all the functions involving at least one
//    stochastic argument which overload the  "+" operator
//    in a statement such as "a+b" without instability detection.
//
//
//    INPUTS
//      a           - an integer, a float, a double or a stochastic number
//      b           - an integer, a float, a double or a stochastic number
//      At least one argument must be of stochastic type.
//
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

///////////////////////////////////////////////////
//****m* cadna_comp/add
//    NAME
//      add
//    SYNOPSIS
//      res = a + b
//    FUNCTION
//    Defines all the functions involving at least one
//    stochastic argument which compute "a+b" statement 
//    without instability detection
//
//
//    INPUTS
//      a           - an integer, a float, a double or a stochastic number
//      b           - an integer, a float, a double or a stochastic number
//      At least one argument must be of stochastic type.
//
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




inline double_st add(const double& a, const double_st& b){
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
  res.x += tmp;
  DOUBLE_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp = a;
  DOUBLE_BIT_FLIP(tmp, random);
  res.y = tmp;
  tmp = b.y;
  DOUBLE_BIT_FLIP(tmp, random);
  res.y += tmp;
  DOUBLE_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp = a;
  DOUBLE_BIT_FLIP(tmp, random);
  res.z = tmp;
  tmp = b.z;
  DOUBLE_BIT_FLIP(tmp, random);
  res.z += tmp;
  DOUBLE_BIT_FLIP(res.z, random);

return res;
}
inline double_st add(const float& a, const double_st& b){
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
  res.x += tmp;
  DOUBLE_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp = a;
  DOUBLE_BIT_FLIP(tmp, random);
  res.y = tmp;
  tmp = b.y;
  DOUBLE_BIT_FLIP(tmp, random);
  res.y += tmp;
  DOUBLE_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp = a;
  DOUBLE_BIT_FLIP(tmp, random);
  res.z = tmp;
  tmp = b.z;
  DOUBLE_BIT_FLIP(tmp, random);
  res.z += tmp;
  DOUBLE_BIT_FLIP(res.z, random);

return res;
}
inline double_st add(const unsigned long long& a, const double_st& b){
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
  res.x += tmp;
  DOUBLE_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp = a;
  DOUBLE_BIT_FLIP(tmp, random);
  res.y = tmp;
  tmp = b.y;
  DOUBLE_BIT_FLIP(tmp, random);
  res.y += tmp;
  DOUBLE_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp = a;
  DOUBLE_BIT_FLIP(tmp, random);
  res.z = tmp;
  tmp = b.z;
  DOUBLE_BIT_FLIP(tmp, random);
  res.z += tmp;
  DOUBLE_BIT_FLIP(res.z, random);

return res;
}
inline double_st add(const long long& a, const double_st& b){
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
  res.x += tmp;
  DOUBLE_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp = a;
  DOUBLE_BIT_FLIP(tmp, random);
  res.y = tmp;
  tmp = b.y;
  DOUBLE_BIT_FLIP(tmp, random);
  res.y += tmp;
  DOUBLE_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp = a;
  DOUBLE_BIT_FLIP(tmp, random);
  res.z = tmp;
  tmp = b.z;
  DOUBLE_BIT_FLIP(tmp, random);
  res.z += tmp;
  DOUBLE_BIT_FLIP(res.z, random);

return res;
}
inline double_st add(const unsigned long& a, const double_st& b){
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
  res.x += tmp;
  DOUBLE_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp = a;
  DOUBLE_BIT_FLIP(tmp, random);
  res.y = tmp;
  tmp = b.y;
  DOUBLE_BIT_FLIP(tmp, random);
  res.y += tmp;
  DOUBLE_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp = a;
  DOUBLE_BIT_FLIP(tmp, random);
  res.z = tmp;
  tmp = b.z;
  DOUBLE_BIT_FLIP(tmp, random);
  res.z += tmp;
  DOUBLE_BIT_FLIP(res.z, random);

return res;
}
inline double_st add(const long& a, const double_st& b){
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
  res.x += tmp;
  DOUBLE_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp = a;
  DOUBLE_BIT_FLIP(tmp, random);
  res.y = tmp;
  tmp = b.y;
  DOUBLE_BIT_FLIP(tmp, random);
  res.y += tmp;
  DOUBLE_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp = a;
  DOUBLE_BIT_FLIP(tmp, random);
  res.z = tmp;
  tmp = b.z;
  DOUBLE_BIT_FLIP(tmp, random);
  res.z += tmp;
  DOUBLE_BIT_FLIP(res.z, random);

return res;
}
inline double_st add(const unsigned int& a, const double_st& b){
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
  res.x += tmp;
  DOUBLE_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp = a;
  DOUBLE_BIT_FLIP(tmp, random);
  res.y = tmp;
  tmp = b.y;
  DOUBLE_BIT_FLIP(tmp, random);
  res.y += tmp;
  DOUBLE_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp = a;
  DOUBLE_BIT_FLIP(tmp, random);
  res.z = tmp;
  tmp = b.z;
  DOUBLE_BIT_FLIP(tmp, random);
  res.z += tmp;
  DOUBLE_BIT_FLIP(res.z, random);

return res;
}
inline double_st add(const int& a, const double_st& b){
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
  res.x += tmp;
  DOUBLE_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp = a;
  DOUBLE_BIT_FLIP(tmp, random);
  res.y = tmp;
  tmp = b.y;
  DOUBLE_BIT_FLIP(tmp, random);
  res.y += tmp;
  DOUBLE_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp = a;
  DOUBLE_BIT_FLIP(tmp, random);
  res.z = tmp;
  tmp = b.z;
  DOUBLE_BIT_FLIP(tmp, random);
  res.z += tmp;
  DOUBLE_BIT_FLIP(res.z, random);

return res;
}
inline double_st add(const unsigned short& a, const double_st& b){
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
  res.x += tmp;
  DOUBLE_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp = a;
  DOUBLE_BIT_FLIP(tmp, random);
  res.y = tmp;
  tmp = b.y;
  DOUBLE_BIT_FLIP(tmp, random);
  res.y += tmp;
  DOUBLE_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp = a;
  DOUBLE_BIT_FLIP(tmp, random);
  res.z = tmp;
  tmp = b.z;
  DOUBLE_BIT_FLIP(tmp, random);
  res.z += tmp;
  DOUBLE_BIT_FLIP(res.z, random);

return res;
}
inline double_st add(const short& a, const double_st& b){
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
  res.x += tmp;
  DOUBLE_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp = a;
  DOUBLE_BIT_FLIP(tmp, random);
  res.y = tmp;
  tmp = b.y;
  DOUBLE_BIT_FLIP(tmp, random);
  res.y += tmp;
  DOUBLE_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp = a;
  DOUBLE_BIT_FLIP(tmp, random);
  res.z = tmp;
  tmp = b.z;
  DOUBLE_BIT_FLIP(tmp, random);
  res.z += tmp;
  DOUBLE_BIT_FLIP(res.z, random);

return res;
}
inline double_st add(const unsigned char& a, const double_st& b){
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
  res.x += tmp;
  DOUBLE_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp = a;
  DOUBLE_BIT_FLIP(tmp, random);
  res.y = tmp;
  tmp = b.y;
  DOUBLE_BIT_FLIP(tmp, random);
  res.y += tmp;
  DOUBLE_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp = a;
  DOUBLE_BIT_FLIP(tmp, random);
  res.z = tmp;
  tmp = b.z;
  DOUBLE_BIT_FLIP(tmp, random);
  res.z += tmp;
  DOUBLE_BIT_FLIP(res.z, random);

return res;
}
inline double_st add(const char& a, const double_st& b){
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
  res.x += tmp;
  DOUBLE_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp = a;
  DOUBLE_BIT_FLIP(tmp, random);
  res.y = tmp;
  tmp = b.y;
  DOUBLE_BIT_FLIP(tmp, random);
  res.y += tmp;
  DOUBLE_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp = a;
  DOUBLE_BIT_FLIP(tmp, random);
  res.z = tmp;
  tmp = b.z;
  DOUBLE_BIT_FLIP(tmp, random);
  res.z += tmp;
  DOUBLE_BIT_FLIP(res.z, random);

return res;
}

inline float_st add(const double& a, const float_st& b){
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
  res.x += tmp;
  FLOAT_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp = a;
  FLOAT_BIT_FLIP(tmp, random);
  res.y = tmp;
  tmp = b.y;
  FLOAT_BIT_FLIP(tmp, random);
  res.y += tmp;
  FLOAT_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp = a;
  FLOAT_BIT_FLIP(tmp, random);
  res.z = tmp;
  tmp = b.z;
  FLOAT_BIT_FLIP(tmp, random);
  res.z += tmp;
  FLOAT_BIT_FLIP(res.z, random);

return res;
}
inline float_st add(const float& a, const float_st& b){
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
  res.x += tmp;
  FLOAT_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp = a;
  FLOAT_BIT_FLIP(tmp, random);
  res.y = tmp;
  tmp = b.y;
  FLOAT_BIT_FLIP(tmp, random);
  res.y += tmp;
  FLOAT_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp = a;
  FLOAT_BIT_FLIP(tmp, random);
  res.z = tmp;
  tmp = b.z;
  FLOAT_BIT_FLIP(tmp, random);
  res.z += tmp;
  FLOAT_BIT_FLIP(res.z, random);

return res;
}
inline float_st add(const unsigned long long& a, const float_st& b){
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
  res.x += tmp;
  FLOAT_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp = a;
  FLOAT_BIT_FLIP(tmp, random);
  res.y = tmp;
  tmp = b.y;
  FLOAT_BIT_FLIP(tmp, random);
  res.y += tmp;
  FLOAT_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp = a;
  FLOAT_BIT_FLIP(tmp, random);
  res.z = tmp;
  tmp = b.z;
  FLOAT_BIT_FLIP(tmp, random);
  res.z += tmp;
  FLOAT_BIT_FLIP(res.z, random);

return res;
}
inline float_st add(const long long& a, const float_st& b){
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
  res.x += tmp;
  FLOAT_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp = a;
  FLOAT_BIT_FLIP(tmp, random);
  res.y = tmp;
  tmp = b.y;
  FLOAT_BIT_FLIP(tmp, random);
  res.y += tmp;
  FLOAT_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp = a;
  FLOAT_BIT_FLIP(tmp, random);
  res.z = tmp;
  tmp = b.z;
  FLOAT_BIT_FLIP(tmp, random);
  res.z += tmp;
  FLOAT_BIT_FLIP(res.z, random);

return res;
}
inline float_st add(const unsigned long& a, const float_st& b){
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
  res.x += tmp;
  FLOAT_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp = a;
  FLOAT_BIT_FLIP(tmp, random);
  res.y = tmp;
  tmp = b.y;
  FLOAT_BIT_FLIP(tmp, random);
  res.y += tmp;
  FLOAT_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp = a;
  FLOAT_BIT_FLIP(tmp, random);
  res.z = tmp;
  tmp = b.z;
  FLOAT_BIT_FLIP(tmp, random);
  res.z += tmp;
  FLOAT_BIT_FLIP(res.z, random);

return res;
}
inline float_st add(const long& a, const float_st& b){
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
  res.x += tmp;
  FLOAT_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp = a;
  FLOAT_BIT_FLIP(tmp, random);
  res.y = tmp;
  tmp = b.y;
  FLOAT_BIT_FLIP(tmp, random);
  res.y += tmp;
  FLOAT_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp = a;
  FLOAT_BIT_FLIP(tmp, random);
  res.z = tmp;
  tmp = b.z;
  FLOAT_BIT_FLIP(tmp, random);
  res.z += tmp;
  FLOAT_BIT_FLIP(res.z, random);

return res;
}
inline float_st add(const unsigned int& a, const float_st& b){
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
  res.x += tmp;
  FLOAT_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp = a;
  FLOAT_BIT_FLIP(tmp, random);
  res.y = tmp;
  tmp = b.y;
  FLOAT_BIT_FLIP(tmp, random);
  res.y += tmp;
  FLOAT_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp = a;
  FLOAT_BIT_FLIP(tmp, random);
  res.z = tmp;
  tmp = b.z;
  FLOAT_BIT_FLIP(tmp, random);
  res.z += tmp;
  FLOAT_BIT_FLIP(res.z, random);

return res;
}
inline float_st add(const int& a, const float_st& b){
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
  res.x += tmp;
  FLOAT_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp = a;
  FLOAT_BIT_FLIP(tmp, random);
  res.y = tmp;
  tmp = b.y;
  FLOAT_BIT_FLIP(tmp, random);
  res.y += tmp;
  FLOAT_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp = a;
  FLOAT_BIT_FLIP(tmp, random);
  res.z = tmp;
  tmp = b.z;
  FLOAT_BIT_FLIP(tmp, random);
  res.z += tmp;
  FLOAT_BIT_FLIP(res.z, random);

return res;
}
inline float_st add(const unsigned short& a, const float_st& b){
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
  res.x += tmp;
  FLOAT_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp = a;
  FLOAT_BIT_FLIP(tmp, random);
  res.y = tmp;
  tmp = b.y;
  FLOAT_BIT_FLIP(tmp, random);
  res.y += tmp;
  FLOAT_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp = a;
  FLOAT_BIT_FLIP(tmp, random);
  res.z = tmp;
  tmp = b.z;
  FLOAT_BIT_FLIP(tmp, random);
  res.z += tmp;
  FLOAT_BIT_FLIP(res.z, random);

return res;
}
inline float_st add(const short& a, const float_st& b){
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
  res.x += tmp;
  FLOAT_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp = a;
  FLOAT_BIT_FLIP(tmp, random);
  res.y = tmp;
  tmp = b.y;
  FLOAT_BIT_FLIP(tmp, random);
  res.y += tmp;
  FLOAT_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp = a;
  FLOAT_BIT_FLIP(tmp, random);
  res.z = tmp;
  tmp = b.z;
  FLOAT_BIT_FLIP(tmp, random);
  res.z += tmp;
  FLOAT_BIT_FLIP(res.z, random);

return res;
}
inline float_st add(const unsigned char& a, const float_st& b){
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
  res.x += tmp;
  FLOAT_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp = a;
  FLOAT_BIT_FLIP(tmp, random);
  res.y = tmp;
  tmp = b.y;
  FLOAT_BIT_FLIP(tmp, random);
  res.y += tmp;
  FLOAT_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp = a;
  FLOAT_BIT_FLIP(tmp, random);
  res.z = tmp;
  tmp = b.z;
  FLOAT_BIT_FLIP(tmp, random);
  res.z += tmp;
  FLOAT_BIT_FLIP(res.z, random);

return res;
}
inline float_st add(const char& a, const float_st& b){
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
  res.x += tmp;
  FLOAT_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp = a;
  FLOAT_BIT_FLIP(tmp, random);
  res.y = tmp;
  tmp = b.y;
  FLOAT_BIT_FLIP(tmp, random);
  res.y += tmp;
  FLOAT_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp = a;
  FLOAT_BIT_FLIP(tmp, random);
  res.z = tmp;
  tmp = b.z;
  FLOAT_BIT_FLIP(tmp, random);
  res.z += tmp;
  FLOAT_BIT_FLIP(res.z, random);

return res;
}

#ifdef CADNA_QUAD
inline double_st add(const float128& a, const double_st& b){
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
  res.x += tmp;
  DOUBLE_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp = a;
  DOUBLE_BIT_FLIP(tmp, random);
  res.y = tmp;
  tmp = b.y;
  DOUBLE_BIT_FLIP(tmp, random);
  res.y += tmp;
  DOUBLE_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp = a;
  DOUBLE_BIT_FLIP(tmp, random);
  res.z = tmp;
  tmp = b.z;
  DOUBLE_BIT_FLIP(tmp, random);
  res.z += tmp;
  DOUBLE_BIT_FLIP(res.z, random);

return res;
}
inline float_st add(const float128& a, const float_st& b){
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
  res.x += tmp;
  FLOAT_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp = a;
  FLOAT_BIT_FLIP(tmp, random);
  res.y = tmp;
  tmp = b.y;
  FLOAT_BIT_FLIP(tmp, random);
  res.y += tmp;
  FLOAT_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp = a;
  FLOAT_BIT_FLIP(tmp, random);
  res.z = tmp;
  tmp = b.z;
  FLOAT_BIT_FLIP(tmp, random);
  res.z += tmp;
  FLOAT_BIT_FLIP(res.z, random);

return res;
}
inline float128_st add(const double& a, const float128_st& b){
  float128_st res;

  unsigned int random;
  float128 tmp;

  res.accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = a;
  FLOAT128_BIT_FLIP(tmp, random);
  res.x = tmp;
  tmp = b.x;
  FLOAT128_BIT_FLIP(tmp, random);
  res.x += tmp;
  FLOAT128_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp = a;
  FLOAT128_BIT_FLIP(tmp, random);
  res.y = tmp;
  tmp = b.y;
  FLOAT128_BIT_FLIP(tmp, random);
  res.y += tmp;
  FLOAT128_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp = a;
  FLOAT128_BIT_FLIP(tmp, random);
  res.z = tmp;
  tmp = b.z;
  FLOAT128_BIT_FLIP(tmp, random);
  res.z += tmp;
  FLOAT128_BIT_FLIP(res.z, random);

return res;
}
inline float128_st add(const float& a, const float128_st& b){
  float128_st res;

  unsigned int random;
  float128 tmp;

  res.accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = a;
  FLOAT128_BIT_FLIP(tmp, random);
  res.x = tmp;
  tmp = b.x;
  FLOAT128_BIT_FLIP(tmp, random);
  res.x += tmp;
  FLOAT128_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp = a;
  FLOAT128_BIT_FLIP(tmp, random);
  res.y = tmp;
  tmp = b.y;
  FLOAT128_BIT_FLIP(tmp, random);
  res.y += tmp;
  FLOAT128_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp = a;
  FLOAT128_BIT_FLIP(tmp, random);
  res.z = tmp;
  tmp = b.z;
  FLOAT128_BIT_FLIP(tmp, random);
  res.z += tmp;
  FLOAT128_BIT_FLIP(res.z, random);

return res;
}
inline float128_st add(const unsigned long long& a, const float128_st& b){
  float128_st res;

  unsigned int random;
  float128 tmp;

  res.accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = a;
  FLOAT128_BIT_FLIP(tmp, random);
  res.x = tmp;
  tmp = b.x;
  FLOAT128_BIT_FLIP(tmp, random);
  res.x += tmp;
  FLOAT128_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp = a;
  FLOAT128_BIT_FLIP(tmp, random);
  res.y = tmp;
  tmp = b.y;
  FLOAT128_BIT_FLIP(tmp, random);
  res.y += tmp;
  FLOAT128_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp = a;
  FLOAT128_BIT_FLIP(tmp, random);
  res.z = tmp;
  tmp = b.z;
  FLOAT128_BIT_FLIP(tmp, random);
  res.z += tmp;
  FLOAT128_BIT_FLIP(res.z, random);

return res;
}
inline float128_st add(const long long& a, const float128_st& b){
  float128_st res;

  unsigned int random;
  float128 tmp;

  res.accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = a;
  FLOAT128_BIT_FLIP(tmp, random);
  res.x = tmp;
  tmp = b.x;
  FLOAT128_BIT_FLIP(tmp, random);
  res.x += tmp;
  FLOAT128_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp = a;
  FLOAT128_BIT_FLIP(tmp, random);
  res.y = tmp;
  tmp = b.y;
  FLOAT128_BIT_FLIP(tmp, random);
  res.y += tmp;
  FLOAT128_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp = a;
  FLOAT128_BIT_FLIP(tmp, random);
  res.z = tmp;
  tmp = b.z;
  FLOAT128_BIT_FLIP(tmp, random);
  res.z += tmp;
  FLOAT128_BIT_FLIP(res.z, random);

return res;
}
inline float128_st add(const unsigned long& a, const float128_st& b){
  float128_st res;

  unsigned int random;
  float128 tmp;

  res.accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = a;
  FLOAT128_BIT_FLIP(tmp, random);
  res.x = tmp;
  tmp = b.x;
  FLOAT128_BIT_FLIP(tmp, random);
  res.x += tmp;
  FLOAT128_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp = a;
  FLOAT128_BIT_FLIP(tmp, random);
  res.y = tmp;
  tmp = b.y;
  FLOAT128_BIT_FLIP(tmp, random);
  res.y += tmp;
  FLOAT128_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp = a;
  FLOAT128_BIT_FLIP(tmp, random);
  res.z = tmp;
  tmp = b.z;
  FLOAT128_BIT_FLIP(tmp, random);
  res.z += tmp;
  FLOAT128_BIT_FLIP(res.z, random);

return res;
}
inline float128_st add(const long& a, const float128_st& b){
  float128_st res;

  unsigned int random;
  float128 tmp;

  res.accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = a;
  FLOAT128_BIT_FLIP(tmp, random);
  res.x = tmp;
  tmp = b.x;
  FLOAT128_BIT_FLIP(tmp, random);
  res.x += tmp;
  FLOAT128_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp = a;
  FLOAT128_BIT_FLIP(tmp, random);
  res.y = tmp;
  tmp = b.y;
  FLOAT128_BIT_FLIP(tmp, random);
  res.y += tmp;
  FLOAT128_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp = a;
  FLOAT128_BIT_FLIP(tmp, random);
  res.z = tmp;
  tmp = b.z;
  FLOAT128_BIT_FLIP(tmp, random);
  res.z += tmp;
  FLOAT128_BIT_FLIP(res.z, random);

return res;
}
inline float128_st add(const unsigned int& a, const float128_st& b){
  float128_st res;

  unsigned int random;
  float128 tmp;

  res.accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = a;
  FLOAT128_BIT_FLIP(tmp, random);
  res.x = tmp;
  tmp = b.x;
  FLOAT128_BIT_FLIP(tmp, random);
  res.x += tmp;
  FLOAT128_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp = a;
  FLOAT128_BIT_FLIP(tmp, random);
  res.y = tmp;
  tmp = b.y;
  FLOAT128_BIT_FLIP(tmp, random);
  res.y += tmp;
  FLOAT128_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp = a;
  FLOAT128_BIT_FLIP(tmp, random);
  res.z = tmp;
  tmp = b.z;
  FLOAT128_BIT_FLIP(tmp, random);
  res.z += tmp;
  FLOAT128_BIT_FLIP(res.z, random);

return res;
}
inline float128_st add(const int& a, const float128_st& b){
  float128_st res;

  unsigned int random;
  float128 tmp;

  res.accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = a;
  FLOAT128_BIT_FLIP(tmp, random);
  res.x = tmp;
  tmp = b.x;
  FLOAT128_BIT_FLIP(tmp, random);
  res.x += tmp;
  FLOAT128_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp = a;
  FLOAT128_BIT_FLIP(tmp, random);
  res.y = tmp;
  tmp = b.y;
  FLOAT128_BIT_FLIP(tmp, random);
  res.y += tmp;
  FLOAT128_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp = a;
  FLOAT128_BIT_FLIP(tmp, random);
  res.z = tmp;
  tmp = b.z;
  FLOAT128_BIT_FLIP(tmp, random);
  res.z += tmp;
  FLOAT128_BIT_FLIP(res.z, random);

return res;
}
inline float128_st add(const unsigned short& a, const float128_st& b){
  float128_st res;

  unsigned int random;
  float128 tmp;

  res.accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = a;
  FLOAT128_BIT_FLIP(tmp, random);
  res.x = tmp;
  tmp = b.x;
  FLOAT128_BIT_FLIP(tmp, random);
  res.x += tmp;
  FLOAT128_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp = a;
  FLOAT128_BIT_FLIP(tmp, random);
  res.y = tmp;
  tmp = b.y;
  FLOAT128_BIT_FLIP(tmp, random);
  res.y += tmp;
  FLOAT128_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp = a;
  FLOAT128_BIT_FLIP(tmp, random);
  res.z = tmp;
  tmp = b.z;
  FLOAT128_BIT_FLIP(tmp, random);
  res.z += tmp;
  FLOAT128_BIT_FLIP(res.z, random);

return res;
}
inline float128_st add(const short& a, const float128_st& b){
  float128_st res;

  unsigned int random;
  float128 tmp;

  res.accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = a;
  FLOAT128_BIT_FLIP(tmp, random);
  res.x = tmp;
  tmp = b.x;
  FLOAT128_BIT_FLIP(tmp, random);
  res.x += tmp;
  FLOAT128_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp = a;
  FLOAT128_BIT_FLIP(tmp, random);
  res.y = tmp;
  tmp = b.y;
  FLOAT128_BIT_FLIP(tmp, random);
  res.y += tmp;
  FLOAT128_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp = a;
  FLOAT128_BIT_FLIP(tmp, random);
  res.z = tmp;
  tmp = b.z;
  FLOAT128_BIT_FLIP(tmp, random);
  res.z += tmp;
  FLOAT128_BIT_FLIP(res.z, random);

return res;
}
inline float128_st add(const unsigned char& a, const float128_st& b){
  float128_st res;

  unsigned int random;
  float128 tmp;

  res.accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = a;
  FLOAT128_BIT_FLIP(tmp, random);
  res.x = tmp;
  tmp = b.x;
  FLOAT128_BIT_FLIP(tmp, random);
  res.x += tmp;
  FLOAT128_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp = a;
  FLOAT128_BIT_FLIP(tmp, random);
  res.y = tmp;
  tmp = b.y;
  FLOAT128_BIT_FLIP(tmp, random);
  res.y += tmp;
  FLOAT128_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp = a;
  FLOAT128_BIT_FLIP(tmp, random);
  res.z = tmp;
  tmp = b.z;
  FLOAT128_BIT_FLIP(tmp, random);
  res.z += tmp;
  FLOAT128_BIT_FLIP(res.z, random);

return res;
}
inline float128_st add(const char& a, const float128_st& b){
  float128_st res;

  unsigned int random;
  float128 tmp;

  res.accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = a;
  FLOAT128_BIT_FLIP(tmp, random);
  res.x = tmp;
  tmp = b.x;
  FLOAT128_BIT_FLIP(tmp, random);
  res.x += tmp;
  FLOAT128_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp = a;
  FLOAT128_BIT_FLIP(tmp, random);
  res.y = tmp;
  tmp = b.y;
  FLOAT128_BIT_FLIP(tmp, random);
  res.y += tmp;
  FLOAT128_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp = a;
  FLOAT128_BIT_FLIP(tmp, random);
  res.z = tmp;
  tmp = b.z;
  FLOAT128_BIT_FLIP(tmp, random);
  res.z += tmp;
  FLOAT128_BIT_FLIP(res.z, random);

return res;
}
#endif //CADNA_QUAD

/////////////////////////////////////////////////////////////////////////



inline double_st add(const double_st& a, const double& b){
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
  res.x += tmp;
  DOUBLE_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp = a.y;
  DOUBLE_BIT_FLIP(tmp, random);
  res.y = tmp;
  tmp = b;
  DOUBLE_BIT_FLIP(tmp, random);
  res.y += tmp;
  DOUBLE_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp = a.z;
  DOUBLE_BIT_FLIP(tmp, random);
  res.z = tmp;
  tmp = b;
  DOUBLE_BIT_FLIP(tmp, random);
  res.z += tmp;
  DOUBLE_BIT_FLIP(res.z, random);

  return res;
}
inline double_st add(const double_st& a, const float& b){
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
  res.x += tmp;
  DOUBLE_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp = a.y;
  DOUBLE_BIT_FLIP(tmp, random);
  res.y = tmp;
  tmp = b;
  DOUBLE_BIT_FLIP(tmp, random);
  res.y += tmp;
  DOUBLE_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp = a.z;
  DOUBLE_BIT_FLIP(tmp, random);
  res.z = tmp;
  tmp = b;
  DOUBLE_BIT_FLIP(tmp, random);
  res.z += tmp;
  DOUBLE_BIT_FLIP(res.z, random);

  return res;
}
inline double_st add(const double_st& a, const unsigned long long& b){
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
  res.x += tmp;
  DOUBLE_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp = a.y;
  DOUBLE_BIT_FLIP(tmp, random);
  res.y = tmp;
  tmp = b;
  DOUBLE_BIT_FLIP(tmp, random);
  res.y += tmp;
  DOUBLE_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp = a.z;
  DOUBLE_BIT_FLIP(tmp, random);
  res.z = tmp;
  tmp = b;
  DOUBLE_BIT_FLIP(tmp, random);
  res.z += tmp;
  DOUBLE_BIT_FLIP(res.z, random);

  return res;
}
inline double_st add(const double_st& a, const long long& b){
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
  res.x += tmp;
  DOUBLE_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp = a.y;
  DOUBLE_BIT_FLIP(tmp, random);
  res.y = tmp;
  tmp = b;
  DOUBLE_BIT_FLIP(tmp, random);
  res.y += tmp;
  DOUBLE_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp = a.z;
  DOUBLE_BIT_FLIP(tmp, random);
  res.z = tmp;
  tmp = b;
  DOUBLE_BIT_FLIP(tmp, random);
  res.z += tmp;
  DOUBLE_BIT_FLIP(res.z, random);

  return res;
}
inline double_st add(const double_st& a, const unsigned long& b){
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
  res.x += tmp;
  DOUBLE_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp = a.y;
  DOUBLE_BIT_FLIP(tmp, random);
  res.y = tmp;
  tmp = b;
  DOUBLE_BIT_FLIP(tmp, random);
  res.y += tmp;
  DOUBLE_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp = a.z;
  DOUBLE_BIT_FLIP(tmp, random);
  res.z = tmp;
  tmp = b;
  DOUBLE_BIT_FLIP(tmp, random);
  res.z += tmp;
  DOUBLE_BIT_FLIP(res.z, random);

  return res;
}
inline double_st add(const double_st& a, const long& b){
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
  res.x += tmp;
  DOUBLE_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp = a.y;
  DOUBLE_BIT_FLIP(tmp, random);
  res.y = tmp;
  tmp = b;
  DOUBLE_BIT_FLIP(tmp, random);
  res.y += tmp;
  DOUBLE_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp = a.z;
  DOUBLE_BIT_FLIP(tmp, random);
  res.z = tmp;
  tmp = b;
  DOUBLE_BIT_FLIP(tmp, random);
  res.z += tmp;
  DOUBLE_BIT_FLIP(res.z, random);

  return res;
}
inline double_st add(const double_st& a, const unsigned int & b){
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
  res.x += tmp;
  DOUBLE_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp = a.y;
  DOUBLE_BIT_FLIP(tmp, random);
  res.y = tmp;
  tmp = b;
  DOUBLE_BIT_FLIP(tmp, random);
  res.y += tmp;
  DOUBLE_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp = a.z;
  DOUBLE_BIT_FLIP(tmp, random);
  res.z = tmp;
  tmp = b;
  DOUBLE_BIT_FLIP(tmp, random);
  res.z += tmp;
  DOUBLE_BIT_FLIP(res.z, random);

  return res;
}
inline double_st add(const double_st& a, const int & b){
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
  res.x += tmp;
  DOUBLE_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp = a.y;
  DOUBLE_BIT_FLIP(tmp, random);
  res.y = tmp;
  tmp = b;
  DOUBLE_BIT_FLIP(tmp, random);
  res.y += tmp;
  DOUBLE_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp = a.z;
  DOUBLE_BIT_FLIP(tmp, random);
  res.z = tmp;
  tmp = b;
  DOUBLE_BIT_FLIP(tmp, random);
  res.z += tmp;
  DOUBLE_BIT_FLIP(res.z, random);

  return res;
}
inline double_st add(const double_st& a, const unsigned short& b){
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
  res.x += tmp;
  DOUBLE_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp = a.y;
  DOUBLE_BIT_FLIP(tmp, random);
  res.y = tmp;
  tmp = b;
  DOUBLE_BIT_FLIP(tmp, random);
  res.y += tmp;
  DOUBLE_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp = a.z;
  DOUBLE_BIT_FLIP(tmp, random);
  res.z = tmp;
  tmp = b;
  DOUBLE_BIT_FLIP(tmp, random);
  res.z += tmp;
  DOUBLE_BIT_FLIP(res.z, random);

  return res;
}
inline double_st add(const double_st& a, const short& b){
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
  res.x += tmp;
  DOUBLE_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp = a.y;
  DOUBLE_BIT_FLIP(tmp, random);
  res.y = tmp;
  tmp = b;
  DOUBLE_BIT_FLIP(tmp, random);
  res.y += tmp;
  DOUBLE_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp = a.z;
  DOUBLE_BIT_FLIP(tmp, random);
  res.z = tmp;
  tmp = b;
  DOUBLE_BIT_FLIP(tmp, random);
  res.z += tmp;
  DOUBLE_BIT_FLIP(res.z, random);

  return res;
}
inline double_st add(const double_st& a, const unsigned char& b){
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
  res.x += tmp;
  DOUBLE_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp = a.y;
  DOUBLE_BIT_FLIP(tmp, random);
  res.y = tmp;
  tmp = b;
  DOUBLE_BIT_FLIP(tmp, random);
  res.y += tmp;
  DOUBLE_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp = a.z;
  DOUBLE_BIT_FLIP(tmp, random);
  res.z = tmp;
  tmp = b;
  DOUBLE_BIT_FLIP(tmp, random);
  res.z += tmp;
  DOUBLE_BIT_FLIP(res.z, random);

  return res;
}
inline double_st add(const double_st& a, const char& b){
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
  res.x += tmp;
  DOUBLE_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp = a.y;
  DOUBLE_BIT_FLIP(tmp, random);
  res.y = tmp;
  tmp = b;
  DOUBLE_BIT_FLIP(tmp, random);
  res.y += tmp;
  DOUBLE_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp = a.z;
  DOUBLE_BIT_FLIP(tmp, random);
  res.z = tmp;
  tmp = b;
  DOUBLE_BIT_FLIP(tmp, random);
  res.z += tmp;
  DOUBLE_BIT_FLIP(res.z, random);

  return res;
}

inline float_st add(const float_st& a, const double& b){
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
  res.x += tmp;
  FLOAT_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp = a.y;
  FLOAT_BIT_FLIP(tmp, random);
  res.y = tmp;
  tmp = b;
  FLOAT_BIT_FLIP(tmp, random);
  res.y += tmp;
  FLOAT_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp = a.z;
  FLOAT_BIT_FLIP(tmp, random);
  res.z = tmp;
  tmp = b;
  FLOAT_BIT_FLIP(tmp, random);
  res.z += tmp;
  FLOAT_BIT_FLIP(res.z, random);

  return res;
}
inline float_st add(const float_st& a, const float& b){
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
  res.x += tmp;
  FLOAT_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp = a.y;
  FLOAT_BIT_FLIP(tmp, random);
  res.y = tmp;
  tmp = b;
  FLOAT_BIT_FLIP(tmp, random);
  res.y += tmp;
  FLOAT_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp = a.z;
  FLOAT_BIT_FLIP(tmp, random);
  res.z = tmp;
  tmp = b;
  FLOAT_BIT_FLIP(tmp, random);
  res.z += tmp;
  FLOAT_BIT_FLIP(res.z, random);

  return res;
}
inline float_st add(const float_st& a, const unsigned long long& b){
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
  res.x += tmp;
  FLOAT_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp = a.y;
  FLOAT_BIT_FLIP(tmp, random);
  res.y = tmp;
  tmp = b;
  FLOAT_BIT_FLIP(tmp, random);
  res.y += tmp;
  FLOAT_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp = a.z;
  FLOAT_BIT_FLIP(tmp, random);
  res.z = tmp;
  tmp = b;
  FLOAT_BIT_FLIP(tmp, random);
  res.z += tmp;
  FLOAT_BIT_FLIP(res.z, random);

  return res;
}
inline float_st add(const float_st& a, const long long& b){
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
  res.x += tmp;
  FLOAT_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp = a.y;
  FLOAT_BIT_FLIP(tmp, random);
  res.y = tmp;
  tmp = b;
  FLOAT_BIT_FLIP(tmp, random);
  res.y += tmp;
  FLOAT_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp = a.z;
  FLOAT_BIT_FLIP(tmp, random);
  res.z = tmp;
  tmp = b;
  FLOAT_BIT_FLIP(tmp, random);
  res.z += tmp;
  FLOAT_BIT_FLIP(res.z, random);

  return res;
}
inline float_st add(const float_st& a, const unsigned long& b){
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
  res.x += tmp;
  FLOAT_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp = a.y;
  FLOAT_BIT_FLIP(tmp, random);
  res.y = tmp;
  tmp = b;
  FLOAT_BIT_FLIP(tmp, random);
  res.y += tmp;
  FLOAT_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp = a.z;
  FLOAT_BIT_FLIP(tmp, random);
  res.z = tmp;
  tmp = b;
  FLOAT_BIT_FLIP(tmp, random);
  res.z += tmp;
  FLOAT_BIT_FLIP(res.z, random);

  return res;
}
inline float_st add(const float_st& a, const long& b){
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
  res.x += tmp;
  FLOAT_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp = a.y;
  FLOAT_BIT_FLIP(tmp, random);
  res.y = tmp;
  tmp = b;
  FLOAT_BIT_FLIP(tmp, random);
  res.y += tmp;
  FLOAT_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp = a.z;
  FLOAT_BIT_FLIP(tmp, random);
  res.z = tmp;
  tmp = b;
  FLOAT_BIT_FLIP(tmp, random);
  res.z += tmp;
  FLOAT_BIT_FLIP(res.z, random);

  return res;
}
inline float_st add(const float_st& a, const unsigned int & b){
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
  res.x += tmp;
  FLOAT_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp = a.y;
  FLOAT_BIT_FLIP(tmp, random);
  res.y = tmp;
  tmp = b;
  FLOAT_BIT_FLIP(tmp, random);
  res.y += tmp;
  FLOAT_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp = a.z;
  FLOAT_BIT_FLIP(tmp, random);
  res.z = tmp;
  tmp = b;
  FLOAT_BIT_FLIP(tmp, random);
  res.z += tmp;
  FLOAT_BIT_FLIP(res.z, random);

  return res;
}
inline float_st add(const float_st& a, const int & b){
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
  res.x += tmp;
  FLOAT_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp = a.y;
  FLOAT_BIT_FLIP(tmp, random);
  res.y = tmp;
  tmp = b;
  FLOAT_BIT_FLIP(tmp, random);
  res.y += tmp;
  FLOAT_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp = a.z;
  FLOAT_BIT_FLIP(tmp, random);
  res.z = tmp;
  tmp = b;
  FLOAT_BIT_FLIP(tmp, random);
  res.z += tmp;
  FLOAT_BIT_FLIP(res.z, random);

  return res;
}
inline float_st add(const float_st& a, const unsigned short& b){
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
  res.x += tmp;
  FLOAT_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp = a.y;
  FLOAT_BIT_FLIP(tmp, random);
  res.y = tmp;
  tmp = b;
  FLOAT_BIT_FLIP(tmp, random);
  res.y += tmp;
  FLOAT_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp = a.z;
  FLOAT_BIT_FLIP(tmp, random);
  res.z = tmp;
  tmp = b;
  FLOAT_BIT_FLIP(tmp, random);
  res.z += tmp;
  FLOAT_BIT_FLIP(res.z, random);

  return res;
}
inline float_st add(const float_st& a, const short& b){
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
  res.x += tmp;
  FLOAT_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp = a.y;
  FLOAT_BIT_FLIP(tmp, random);
  res.y = tmp;
  tmp = b;
  FLOAT_BIT_FLIP(tmp, random);
  res.y += tmp;
  FLOAT_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp = a.z;
  FLOAT_BIT_FLIP(tmp, random);
  res.z = tmp;
  tmp = b;
  FLOAT_BIT_FLIP(tmp, random);
  res.z += tmp;
  FLOAT_BIT_FLIP(res.z, random);

  return res;
}
inline float_st add(const float_st& a, const unsigned char& b){
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
  res.x += tmp;
  FLOAT_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp = a.y;
  FLOAT_BIT_FLIP(tmp, random);
  res.y = tmp;
  tmp = b;
  FLOAT_BIT_FLIP(tmp, random);
  res.y += tmp;
  FLOAT_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp = a.z;
  FLOAT_BIT_FLIP(tmp, random);
  res.z = tmp;
  tmp = b;
  FLOAT_BIT_FLIP(tmp, random);
  res.z += tmp;
  FLOAT_BIT_FLIP(res.z, random);

  return res;
}
inline float_st add(const float_st& a, const char& b){
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
  res.x += tmp;
  FLOAT_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp = a.y;
  FLOAT_BIT_FLIP(tmp, random);
  res.y = tmp;
  tmp = b;
  FLOAT_BIT_FLIP(tmp, random);
  res.y += tmp;
  FLOAT_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp = a.z;
  FLOAT_BIT_FLIP(tmp, random);
  res.z = tmp;
  tmp = b;
  FLOAT_BIT_FLIP(tmp, random);
  res.z += tmp;
  FLOAT_BIT_FLIP(res.z, random);

  return res;
}


#ifdef CADNA_QUAD
inline double_st add(const double_st& a, const float128& b){
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
  res.x += tmp;
  DOUBLE_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp = a.y;
  DOUBLE_BIT_FLIP(tmp, random);
  res.y = tmp;
  tmp = b;
  DOUBLE_BIT_FLIP(tmp, random);
  res.y += tmp;
  DOUBLE_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp = a.z;
  DOUBLE_BIT_FLIP(tmp, random);
  res.z = tmp;
  tmp = b;
  DOUBLE_BIT_FLIP(tmp, random);
  res.z += tmp;
  DOUBLE_BIT_FLIP(res.z, random);

  return res;
}
inline float_st add(const float_st& a, const float128& b){
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
  res.x += tmp;
  FLOAT_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp = a.y;
  FLOAT_BIT_FLIP(tmp, random);
  res.y = tmp;
  tmp = b;
  FLOAT_BIT_FLIP(tmp, random);
  res.y += tmp;
  FLOAT_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp = a.z;
  FLOAT_BIT_FLIP(tmp, random);
  res.z = tmp;
  tmp = b;
  FLOAT_BIT_FLIP(tmp, random);
  res.z += tmp;
  FLOAT_BIT_FLIP(res.z, random);

  return res;
}
inline float128_st add(const float128_st& a, const double& b){
  float128_st res;

  unsigned int random;
  float128 tmp;

  res.accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = a.x;
  FLOAT128_BIT_FLIP(tmp, random);
  res.x = tmp;
  tmp = b;
  FLOAT128_BIT_FLIP(tmp, random);
  res.x += tmp;
  FLOAT128_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp = a.y;
  FLOAT128_BIT_FLIP(tmp, random);
  res.y = tmp;
  tmp = b;
  FLOAT128_BIT_FLIP(tmp, random);
  res.y += tmp;
  FLOAT128_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp = a.z;
  FLOAT128_BIT_FLIP(tmp, random);
  res.z = tmp;
  tmp = b;
  FLOAT128_BIT_FLIP(tmp, random);
  res.z += tmp;
  FLOAT128_BIT_FLIP(res.z, random);

  return res;
}
inline float128_st add(const float128_st& a, const float& b){
  float128_st res;

  unsigned int random;
  float128 tmp;

  res.accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = a.x;
  FLOAT128_BIT_FLIP(tmp, random);
  res.x = tmp;
  tmp = b;
  FLOAT128_BIT_FLIP(tmp, random);
  res.x += tmp;
  FLOAT128_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp = a.y;
  FLOAT128_BIT_FLIP(tmp, random);
  res.y = tmp;
  tmp = b;
  FLOAT128_BIT_FLIP(tmp, random);
  res.y += tmp;
  FLOAT128_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp = a.z;
  FLOAT128_BIT_FLIP(tmp, random);
  res.z = tmp;
  tmp = b;
  FLOAT128_BIT_FLIP(tmp, random);
  res.z += tmp;
  FLOAT128_BIT_FLIP(res.z, random);

  return res;
}
inline float128_st add(const float128_st& a, const unsigned long long& b){
  float128_st res;

  unsigned int random;
  float128 tmp;

  res.accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = a.x;
  FLOAT128_BIT_FLIP(tmp, random);
  res.x = tmp;
  tmp = b;
  FLOAT128_BIT_FLIP(tmp, random);
  res.x += tmp;
  FLOAT128_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp = a.y;
  FLOAT128_BIT_FLIP(tmp, random);
  res.y = tmp;
  tmp = b;
  FLOAT128_BIT_FLIP(tmp, random);
  res.y += tmp;
  FLOAT128_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp = a.z;
  FLOAT128_BIT_FLIP(tmp, random);
  res.z = tmp;
  tmp = b;
  FLOAT128_BIT_FLIP(tmp, random);
  res.z += tmp;
  FLOAT128_BIT_FLIP(res.z, random);

  return res;
}
inline float128_st add(const float128_st& a, const long long& b){
  float128_st res;

  unsigned int random;
  float128 tmp;

  res.accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = a.x;
  FLOAT128_BIT_FLIP(tmp, random);
  res.x = tmp;
  tmp = b;
  FLOAT128_BIT_FLIP(tmp, random);
  res.x += tmp;
  FLOAT128_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp = a.y;
  FLOAT128_BIT_FLIP(tmp, random);
  res.y = tmp;
  tmp = b;
  FLOAT128_BIT_FLIP(tmp, random);
  res.y += tmp;
  FLOAT128_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp = a.z;
  FLOAT128_BIT_FLIP(tmp, random);
  res.z = tmp;
  tmp = b;
  FLOAT128_BIT_FLIP(tmp, random);
  res.z += tmp;
  FLOAT128_BIT_FLIP(res.z, random);

  return res;
}
inline float128_st add(const float128_st& a, const unsigned long& b){
  float128_st res;

  unsigned int random;
  float128 tmp;

  res.accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = a.x;
  FLOAT128_BIT_FLIP(tmp, random);
  res.x = tmp;
  tmp = b;
  FLOAT128_BIT_FLIP(tmp, random);
  res.x += tmp;
  FLOAT128_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp = a.y;
  FLOAT128_BIT_FLIP(tmp, random);
  res.y = tmp;
  tmp = b;
  FLOAT128_BIT_FLIP(tmp, random);
  res.y += tmp;
  FLOAT128_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp = a.z;
  FLOAT128_BIT_FLIP(tmp, random);
  res.z = tmp;
  tmp = b;
  FLOAT128_BIT_FLIP(tmp, random);
  res.z += tmp;
  FLOAT128_BIT_FLIP(res.z, random);

  return res;
}
inline float128_st add(const float128_st& a, const long& b){
  float128_st res;

  unsigned int random;
  float128 tmp;

  res.accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = a.x;
  FLOAT128_BIT_FLIP(tmp, random);
  res.x = tmp;
  tmp = b;
  FLOAT128_BIT_FLIP(tmp, random);
  res.x += tmp;
  FLOAT128_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp = a.y;
  FLOAT128_BIT_FLIP(tmp, random);
  res.y = tmp;
  tmp = b;
  FLOAT128_BIT_FLIP(tmp, random);
  res.y += tmp;
  FLOAT128_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp = a.z;
  FLOAT128_BIT_FLIP(tmp, random);
  res.z = tmp;
  tmp = b;
  FLOAT128_BIT_FLIP(tmp, random);
  res.z += tmp;
  FLOAT128_BIT_FLIP(res.z, random);

  return res;
}
inline float128_st add(const float128_st& a, const unsigned int & b){
  float128_st res;

  unsigned int random;
  float128 tmp;

  res.accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = a.x;
  FLOAT128_BIT_FLIP(tmp, random);
  res.x = tmp;
  tmp = b;
  FLOAT128_BIT_FLIP(tmp, random);
  res.x += tmp;
  FLOAT128_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp = a.y;
  FLOAT128_BIT_FLIP(tmp, random);
  res.y = tmp;
  tmp = b;
  FLOAT128_BIT_FLIP(tmp, random);
  res.y += tmp;
  FLOAT128_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp = a.z;
  FLOAT128_BIT_FLIP(tmp, random);
  res.z = tmp;
  tmp = b;
  FLOAT128_BIT_FLIP(tmp, random);
  res.z += tmp;
  FLOAT128_BIT_FLIP(res.z, random);

  return res;
}
inline float128_st add(const float128_st& a, const int & b){
  float128_st res;

  unsigned int random;
  float128 tmp;

  res.accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = a.x;
  FLOAT128_BIT_FLIP(tmp, random);
  res.x = tmp;
  tmp = b;
  FLOAT128_BIT_FLIP(tmp, random);
  res.x += tmp;
  FLOAT128_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp = a.y;
  FLOAT128_BIT_FLIP(tmp, random);
  res.y = tmp;
  tmp = b;
  FLOAT128_BIT_FLIP(tmp, random);
  res.y += tmp;
  FLOAT128_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp = a.z;
  FLOAT128_BIT_FLIP(tmp, random);
  res.z = tmp;
  tmp = b;
  FLOAT128_BIT_FLIP(tmp, random);
  res.z += tmp;
  FLOAT128_BIT_FLIP(res.z, random);

  return res;
}
inline float128_st add(const float128_st& a, const unsigned short& b){
  float128_st res;

  unsigned int random;
  float128 tmp;

  res.accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = a.x;
  FLOAT128_BIT_FLIP(tmp, random);
  res.x = tmp;
  tmp = b;
  FLOAT128_BIT_FLIP(tmp, random);
  res.x += tmp;
  FLOAT128_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp = a.y;
  FLOAT128_BIT_FLIP(tmp, random);
  res.y = tmp;
  tmp = b;
  FLOAT128_BIT_FLIP(tmp, random);
  res.y += tmp;
  FLOAT128_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp = a.z;
  FLOAT128_BIT_FLIP(tmp, random);
  res.z = tmp;
  tmp = b;
  FLOAT128_BIT_FLIP(tmp, random);
  res.z += tmp;
  FLOAT128_BIT_FLIP(res.z, random);

  return res;
}
inline float128_st add(const float128_st& a, const short& b){
  float128_st res;

  unsigned int random;
  float128 tmp;

  res.accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = a.x;
  FLOAT128_BIT_FLIP(tmp, random);
  res.x = tmp;
  tmp = b;
  FLOAT128_BIT_FLIP(tmp, random);
  res.x += tmp;
  FLOAT128_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp = a.y;
  FLOAT128_BIT_FLIP(tmp, random);
  res.y = tmp;
  tmp = b;
  FLOAT128_BIT_FLIP(tmp, random);
  res.y += tmp;
  FLOAT128_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp = a.z;
  FLOAT128_BIT_FLIP(tmp, random);
  res.z = tmp;
  tmp = b;
  FLOAT128_BIT_FLIP(tmp, random);
  res.z += tmp;
  FLOAT128_BIT_FLIP(res.z, random);

  return res;
}
inline float128_st add(const float128_st& a, const unsigned char& b){
  float128_st res;

  unsigned int random;
  float128 tmp;

  res.accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = a.x;
  FLOAT128_BIT_FLIP(tmp, random);
  res.x = tmp;
  tmp = b;
  FLOAT128_BIT_FLIP(tmp, random);
  res.x += tmp;
  FLOAT128_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp = a.y;
  FLOAT128_BIT_FLIP(tmp, random);
  res.y = tmp;
  tmp = b;
  FLOAT128_BIT_FLIP(tmp, random);
  res.y += tmp;
  FLOAT128_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp = a.z;
  FLOAT128_BIT_FLIP(tmp, random);
  res.z = tmp;
  tmp = b;
  FLOAT128_BIT_FLIP(tmp, random);
  res.z += tmp;
  FLOAT128_BIT_FLIP(res.z, random);

  return res;
}
inline float128_st add(const float128_st& a, const char& b){
  float128_st res;

  unsigned int random;
  float128 tmp;

  res.accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = a.x;
  FLOAT128_BIT_FLIP(tmp, random);
  res.x = tmp;
  tmp = b;
  FLOAT128_BIT_FLIP(tmp, random);
  res.x += tmp;
  FLOAT128_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp = a.y;
  FLOAT128_BIT_FLIP(tmp, random);
  res.y = tmp;
  tmp = b;
  FLOAT128_BIT_FLIP(tmp, random);
  res.y += tmp;
  FLOAT128_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp = a.z;
  FLOAT128_BIT_FLIP(tmp, random);
  res.z = tmp;
  tmp = b;
  FLOAT128_BIT_FLIP(tmp, random);
  res.z += tmp;
  FLOAT128_BIT_FLIP(res.z, random);

  return res;
}
#endif //CADNA_QUAD


/////////////////////////////////////////////////////////////////////////




inline double_st add(const double_st& a, const double_st& b){
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
  res.x += tmp;
  DOUBLE_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp = a.y;
  DOUBLE_BIT_FLIP(tmp, random);
  res.y = tmp;
  tmp = b.y;
  DOUBLE_BIT_FLIP(tmp, random);
  res.y += tmp;
  DOUBLE_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp = a.z;
  DOUBLE_BIT_FLIP(tmp, random);
  res.z = tmp;
  tmp = b.z;
  DOUBLE_BIT_FLIP(tmp, random);
  res.z += tmp;
  DOUBLE_BIT_FLIP(res.z, random);

  return res;
}
inline double_st add(const double_st& a, const float_st& b){
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
  res.x += tmp;
  DOUBLE_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp = a.y;
  DOUBLE_BIT_FLIP(tmp, random);
  res.y = tmp;
  tmp = b.y;
  DOUBLE_BIT_FLIP(tmp, random);
  res.y += tmp;
  DOUBLE_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp = a.z;
  DOUBLE_BIT_FLIP(tmp, random);
  res.z = tmp;
  tmp = b.z;
  DOUBLE_BIT_FLIP(tmp, random);
  res.z += tmp;
  DOUBLE_BIT_FLIP(res.z, random);

  return res;
}
inline double_st add(const float_st& a, const double_st& b){
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
  res.x += tmp;
  DOUBLE_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp = a.y;
  DOUBLE_BIT_FLIP(tmp, random);
  res.y = tmp;
  tmp = b.y;
  DOUBLE_BIT_FLIP(tmp, random);
  res.y += tmp;
  DOUBLE_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp = a.z;
  DOUBLE_BIT_FLIP(tmp, random);
  res.z = tmp;
  tmp = b.z;
  DOUBLE_BIT_FLIP(tmp, random);
  res.z += tmp;
  DOUBLE_BIT_FLIP(res.z, random);

  return res;
}
inline float_st add(const float_st& a, const float_st& b){
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
  res.x += tmp;
  FLOAT_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp = a.y;
  FLOAT_BIT_FLIP(tmp, random);
  res.y = tmp;
  tmp = b.y;
  FLOAT_BIT_FLIP(tmp, random);
  res.y += tmp;
  FLOAT_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp = a.z;
  FLOAT_BIT_FLIP(tmp, random);
  res.z = tmp;
  tmp = b.z;
  FLOAT_BIT_FLIP(tmp, random);
  res.z += tmp;
  FLOAT_BIT_FLIP(res.z, random);

  return res;
}

#ifdef CADNA_QUAD
inline float128_st add(const float128_st& a, const float128_st& b){
  float128_st res;

  unsigned int random;
  float128 tmp;

  res.accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = a.x;
  FLOAT128_BIT_FLIP(tmp, random);
  res.x = tmp;
  tmp = b.x;
  FLOAT128_BIT_FLIP(tmp, random);
  res.x += tmp;
  FLOAT128_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp = a.y;
  FLOAT128_BIT_FLIP(tmp, random);
  res.y = tmp;
  tmp = b.y;
  FLOAT128_BIT_FLIP(tmp, random);
  res.y += tmp;
  FLOAT128_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp = a.z;
  FLOAT128_BIT_FLIP(tmp, random);
  res.z = tmp;
  tmp = b.z;
  FLOAT128_BIT_FLIP(tmp, random);
  res.z += tmp;
  FLOAT128_BIT_FLIP(res.z, random);

  return res;
}
inline float128_st add(const float128_st& a, const float_st& b){
  float128_st res;

  unsigned int random;
  float128 tmp;

  res.accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = a.x;
  FLOAT128_BIT_FLIP(tmp, random);
  res.x = tmp;
  tmp = b.x;
  FLOAT128_BIT_FLIP(tmp, random);
  res.x += tmp;
  FLOAT128_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp = a.y;
  FLOAT128_BIT_FLIP(tmp, random);
  res.y = tmp;
  tmp = b.y;
  FLOAT128_BIT_FLIP(tmp, random);
  res.y += tmp;
  FLOAT128_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp = a.z;
  FLOAT128_BIT_FLIP(tmp, random);
  res.z = tmp;
  tmp = b.z;
  FLOAT128_BIT_FLIP(tmp, random);
  res.z += tmp;
  FLOAT128_BIT_FLIP(res.z, random);

  return res;
}
inline float128_st add(const float_st& a, const float128_st& b){
  float128_st res;

  unsigned int random;
  float128 tmp;

  res.accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = a.x;
  FLOAT128_BIT_FLIP(tmp, random);
  res.x = tmp;
  tmp = b.x;
  FLOAT128_BIT_FLIP(tmp, random);
  res.x += tmp;
  FLOAT128_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp = a.y;
  FLOAT128_BIT_FLIP(tmp, random);
  res.y = tmp;
  tmp = b.y;
  FLOAT128_BIT_FLIP(tmp, random);
  res.y += tmp;
  FLOAT128_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp = a.z;
  FLOAT128_BIT_FLIP(tmp, random);
  res.z = tmp;
  tmp = b.z;
  FLOAT128_BIT_FLIP(tmp, random);
  res.z += tmp;
  FLOAT128_BIT_FLIP(res.z, random);

  return res;
}
inline float128_st add(const float128_st& a, const double_st& b){
  float128_st res;

  unsigned int random;
  float128 tmp;

  res.accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = a.x;
  FLOAT128_BIT_FLIP(tmp, random);
  res.x = tmp;
  tmp = b.x;
  FLOAT128_BIT_FLIP(tmp, random);
  res.x += tmp;
  FLOAT128_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp = a.y;
  FLOAT128_BIT_FLIP(tmp, random);
  res.y = tmp;
  tmp = b.y;
  FLOAT128_BIT_FLIP(tmp, random);
  res.y += tmp;
  FLOAT128_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp = a.z;
  FLOAT128_BIT_FLIP(tmp, random);
  res.z = tmp;
  tmp = b.z;
  FLOAT128_BIT_FLIP(tmp, random);
  res.z += tmp;
  FLOAT128_BIT_FLIP(res.z, random);

  return res;
}
inline float128_st add(const double_st& a, const float128_st& b){
  float128_st res;

  unsigned int random;
  float128 tmp;

  res.accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = a.x;
  FLOAT128_BIT_FLIP(tmp, random);
  res.x = tmp;
  tmp = b.x;
  FLOAT128_BIT_FLIP(tmp, random);
  res.x += tmp;
  FLOAT128_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp = a.y;
  FLOAT128_BIT_FLIP(tmp, random);
  res.y = tmp;
  tmp = b.y;
  FLOAT128_BIT_FLIP(tmp, random);
  res.y += tmp;
  FLOAT128_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp = a.z;
  FLOAT128_BIT_FLIP(tmp, random);
  res.z = tmp;
  tmp = b.z;
  FLOAT128_BIT_FLIP(tmp, random);
  res.z += tmp;
  FLOAT128_BIT_FLIP(res.z, random);

  return res;
}
#endif


/////////////////////////////////////////////////////////////////////////


///////////////////////////////////////////////////
//****m* cadna_comp/mul
//    NAME
//      mul
//    SYNOPSIS
//      res = a * b
//    FUNCTION
//      Defines all the functions involving at least one argument
//      of stochastic type which overload the "*" operator
//      in a statement such as "a*b" without instability detection.
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


///////////////////////////////////////////////////
//****m* cadna_comp/mul
//    NAME
//      add
//    SYNOPSIS
//      res = a * b
//    FUNCTION
//    Defines all the functions involving at least one
//    stochastic argument which compute "a+b" statement 
//    without instability detection
//
//
//    INPUTS
//      a           - an integer, a float, a double or a stochastic number
//      b           - an integer, a float, a double or a stochastic number
//      At least one argument must be of stochastic type.
//
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




inline double_st mul(const double& a, const double_st& b){
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
inline double_st mul(const float& a, const double_st& b){
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
inline double_st mul(const unsigned long long& a, const double_st& b){
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
inline double_st mul(const long long& a, const double_st& b){
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
inline double_st mul(const unsigned long& a, const double_st& b){
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
inline double_st mul(const long& a, const double_st& b){
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
inline double_st mul(const unsigned int & a, const double_st& b){
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
inline double_st mul(const int & a, const double_st& b){
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
inline double_st mul(const unsigned short& a, const double_st& b){
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
inline double_st mul(const short& a, const double_st& b){
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
inline double_st mul(const unsigned char& a, const double_st& b){
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
inline double_st mul(const char& a, const double_st& b){
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
#ifdef CADNA_QUAD
inline double_st mul(const float128& a, const double_st& b){
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
#endif // CADNA_QUAD

inline float_st mul(const double& a, const float_st& b){
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
inline float_st mul(const float& a, const float_st& b){
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
inline float_st mul(const unsigned long long& a, const float_st& b){
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
inline float_st mul(const long long& a, const float_st& b){
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
inline float_st mul(const unsigned long& a, const float_st& b){
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
inline float_st mul(const long& a, const float_st& b){
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
inline float_st mul(const unsigned int & a, const float_st& b){
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
inline float_st mul(const int & a, const float_st& b){
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
inline float_st mul(const unsigned short& a, const float_st& b){
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
inline float_st mul(const short& a, const float_st& b){
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
inline float_st mul(const unsigned char& a, const float_st& b){
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
inline float_st mul(const char& a, const float_st& b){
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
#ifdef CADNA_QUAD
inline float_st mul(const float128& a, const float_st& b){
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
#endif // CADNA_QUAD

#ifdef CADNA_QUAD
inline float128_st mul(const double& a, const float128_st& b){
  float128_st res;
  unsigned int random;
  float128 tmp;

  res.accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = a;
  FLOAT128_BIT_FLIP(tmp, random);
  res.x = tmp;
  tmp = b.x;
  res.x *= tmp;
  FLOAT128_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp = a;
  FLOAT128_BIT_FLIP(tmp, random);
  res.y = tmp;
  tmp = b.y;
  res.y *= tmp;
  FLOAT128_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp = a;
  FLOAT128_BIT_FLIP(tmp, random);
  res.z = tmp;
  tmp = b.z;
  res.z *= tmp;
  FLOAT128_BIT_FLIP(res.z, random);

  return res ;
}
inline float128_st mul(const float& a, const float128_st& b){
  float128_st res;
  unsigned int random;
  float128 tmp;

  res.accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = a;
  FLOAT128_BIT_FLIP(tmp, random);
  res.x = tmp;
  tmp = b.x;
  res.x *= tmp;
  FLOAT128_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp = a;
  FLOAT128_BIT_FLIP(tmp, random);
  res.y = tmp;
  tmp = b.y;
  res.y *= tmp;
  FLOAT128_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp = a;
  FLOAT128_BIT_FLIP(tmp, random);
  res.z = tmp;
  tmp = b.z;
  res.z *= tmp;
  FLOAT128_BIT_FLIP(res.z, random);

  return res ;
}
inline float128_st mul(const unsigned long long& a, const float128_st& b){
  float128_st res;
  unsigned int random;
  float128 tmp;

  res.accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = a;
  FLOAT128_BIT_FLIP(tmp, random);
  res.x = tmp;
  tmp = b.x;
  res.x *= tmp;
  FLOAT128_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp = a;
  FLOAT128_BIT_FLIP(tmp, random);
  res.y = tmp;
  tmp = b.y;
  res.y *= tmp;
  FLOAT128_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp = a;
  FLOAT128_BIT_FLIP(tmp, random);
  res.z = tmp;
  tmp = b.z;
  res.z *= tmp;
  FLOAT128_BIT_FLIP(res.z, random);

  return res ;
}
inline float128_st mul(const long long& a, const float128_st& b){
  float128_st res;
  unsigned int random;
  float128 tmp;

  res.accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = a;
  FLOAT128_BIT_FLIP(tmp, random);
  res.x = tmp;
  tmp = b.x;
  res.x *= tmp;
  FLOAT128_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp = a;
  FLOAT128_BIT_FLIP(tmp, random);
  res.y = tmp;
  tmp = b.y;
  res.y *= tmp;
  FLOAT128_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp = a;
  FLOAT128_BIT_FLIP(tmp, random);
  res.z = tmp;
  tmp = b.z;
  res.z *= tmp;
  FLOAT128_BIT_FLIP(res.z, random);

  return res ;
}
inline float128_st mul(const unsigned long& a, const float128_st& b){
  float128_st res;
  unsigned int random;
  float128 tmp;

  res.accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = a;
  FLOAT128_BIT_FLIP(tmp, random);
  res.x = tmp;
  tmp = b.x;
  res.x *= tmp;
  FLOAT128_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp = a;
  FLOAT128_BIT_FLIP(tmp, random);
  res.y = tmp;
  tmp = b.y;
  res.y *= tmp;
  FLOAT128_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp = a;
  FLOAT128_BIT_FLIP(tmp, random);
  res.z = tmp;
  tmp = b.z;
  res.z *= tmp;
  FLOAT128_BIT_FLIP(res.z, random);

  return res ;
}
inline float128_st mul(const long& a, const float128_st& b){
  float128_st res;
  unsigned int random;
  float128 tmp;

  res.accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = a;
  FLOAT128_BIT_FLIP(tmp, random);
  res.x = tmp;
  tmp = b.x;
  res.x *= tmp;
  FLOAT128_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp = a;
  FLOAT128_BIT_FLIP(tmp, random);
  res.y = tmp;
  tmp = b.y;
  res.y *= tmp;
  FLOAT128_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp = a;
  FLOAT128_BIT_FLIP(tmp, random);
  res.z = tmp;
  tmp = b.z;
  res.z *= tmp;
  FLOAT128_BIT_FLIP(res.z, random);

  return res ;
}
inline float128_st mul(const unsigned int & a, const float128_st& b){
  float128_st res;
  unsigned int random;
  float128 tmp;

  res.accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = a;
  FLOAT128_BIT_FLIP(tmp, random);
  res.x = tmp;
  tmp = b.x;
  res.x *= tmp;
  FLOAT128_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp = a;
  FLOAT128_BIT_FLIP(tmp, random);
  res.y = tmp;
  tmp = b.y;
  res.y *= tmp;
  FLOAT128_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp = a;
  FLOAT128_BIT_FLIP(tmp, random);
  res.z = tmp;
  tmp = b.z;
  res.z *= tmp;
  FLOAT128_BIT_FLIP(res.z, random);

  return res ;
}
inline float128_st mul(const int & a, const float128_st& b){
  float128_st res;
  unsigned int random;
  float128 tmp;

  res.accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = a;
  FLOAT128_BIT_FLIP(tmp, random);
  res.x = tmp;
  tmp = b.x;
  res.x *= tmp;
  FLOAT128_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp = a;
  FLOAT128_BIT_FLIP(tmp, random);
  res.y = tmp;
  tmp = b.y;
  res.y *= tmp;
  FLOAT128_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp = a;
  FLOAT128_BIT_FLIP(tmp, random);
  res.z = tmp;
  tmp = b.z;
  res.z *= tmp;
  FLOAT128_BIT_FLIP(res.z, random);

  return res ;
}
inline float128_st mul(const unsigned short& a, const float128_st& b){
  float128_st res;
  unsigned int random;
  float128 tmp;

  res.accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = a;
  FLOAT128_BIT_FLIP(tmp, random);
  res.x = tmp;
  tmp = b.x;
  res.x *= tmp;
  FLOAT128_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp = a;
  FLOAT128_BIT_FLIP(tmp, random);
  res.y = tmp;
  tmp = b.y;
  res.y *= tmp;
  FLOAT128_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp = a;
  FLOAT128_BIT_FLIP(tmp, random);
  res.z = tmp;
  tmp = b.z;
  res.z *= tmp;
  FLOAT128_BIT_FLIP(res.z, random);

  return res ;
}
inline float128_st mul(const short& a, const float128_st& b){
  float128_st res;
  unsigned int random;
  float128 tmp;

  res.accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = a;
  FLOAT128_BIT_FLIP(tmp, random);
  res.x = tmp;
  tmp = b.x;
  res.x *= tmp;
  FLOAT128_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp = a;
  FLOAT128_BIT_FLIP(tmp, random);
  res.y = tmp;
  tmp = b.y;
  res.y *= tmp;
  FLOAT128_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp = a;
  FLOAT128_BIT_FLIP(tmp, random);
  res.z = tmp;
  tmp = b.z;
  res.z *= tmp;
  FLOAT128_BIT_FLIP(res.z, random);

  return res ;
}
inline float128_st mul(const unsigned char& a, const float128_st& b){
  float128_st res;
  unsigned int random;
  float128 tmp;

  res.accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = a;
  FLOAT128_BIT_FLIP(tmp, random);
  res.x = tmp;
  tmp = b.x;
  res.x *= tmp;
  FLOAT128_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp = a;
  FLOAT128_BIT_FLIP(tmp, random);
  res.y = tmp;
  tmp = b.y;
  res.y *= tmp;
  FLOAT128_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp = a;
  FLOAT128_BIT_FLIP(tmp, random);
  res.z = tmp;
  tmp = b.z;
  res.z *= tmp;
  FLOAT128_BIT_FLIP(res.z, random);

  return res ;
}
inline float128_st mul(const char& a, const float128_st& b){
  float128_st res;
  unsigned int random;
  float128 tmp;

  res.accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = a;
  FLOAT128_BIT_FLIP(tmp, random);
  res.x = tmp;
  tmp = b.x;
  res.x *= tmp;
  FLOAT128_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp = a;
  FLOAT128_BIT_FLIP(tmp, random);
  res.y = tmp;
  tmp = b.y;
  res.y *= tmp;
  FLOAT128_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp = a;
  FLOAT128_BIT_FLIP(tmp, random);
  res.z = tmp;
  tmp = b.z;
  res.z *= tmp;
  FLOAT128_BIT_FLIP(res.z, random);

  return res ;
}
inline float128_st mul(const float128& a, const float128_st& b){
  float128_st res;
  unsigned int random;
  float128 tmp;

  res.accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = a;
  FLOAT128_BIT_FLIP(tmp, random);
  res.x = tmp;
  tmp = b.x;
  res.x *= tmp;
  FLOAT128_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp = a;
  FLOAT128_BIT_FLIP(tmp, random);
  res.y = tmp;
  tmp = b.y;
  res.y *= tmp;
  FLOAT128_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp = a;
  FLOAT128_BIT_FLIP(tmp, random);
  res.z = tmp;
  tmp = b.z;
  res.z *= tmp;
  FLOAT128_BIT_FLIP(res.z, random);

  return res ;
}
#endif // CADNA_QUAD


/////////////////////////////////////////////////////////////////////////



inline double_st mul(const double_st& a, const double& b){
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
inline double_st mul(const double_st& a, const float& b){
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
inline double_st mul(const double_st& a, const unsigned long long& b){
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
inline double_st mul(const double_st& a, const long long& b){
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
inline double_st mul(const double_st& a, const unsigned long& b){
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
inline double_st mul(const double_st& a, const long& b){
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
inline double_st mul(const double_st& a, const unsigned int & b){
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
inline double_st mul(const double_st& a, const int & b){
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
inline double_st mul(const double_st& a, const unsigned short& b){
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
inline double_st mul(const double_st& a, const short& b){
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
inline double_st mul(const double_st& a, const unsigned char& b){
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
inline double_st mul(const double_st& a, const char& b){
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
#ifdef CADNA_QUAD
inline double_st mul(const double_st& a, const float128& b){
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
#endif // CADNA_QUAD


inline float_st mul(const float_st& a, const double& b){
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
inline float_st mul(const float_st& a, const float& b){
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
inline float_st mul(const float_st& a, const unsigned long long& b){
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
inline float_st mul(const float_st& a, const long long& b){
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
inline float_st mul(const float_st& a, const unsigned long& b){
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
inline float_st mul(const float_st& a, const long& b){
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
inline float_st mul(const float_st& a, const unsigned int & b){
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
inline float_st mul(const float_st& a, const int & b){
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
inline float_st mul(const float_st& a, const unsigned short& b){
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
inline float_st mul(const float_st& a, const short& b){
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
inline float_st mul(const float_st& a, const unsigned char& b){
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
inline float_st mul(const float_st& a, const char& b){
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
#ifdef CADNA_QUAD
inline float_st mul(const float_st& a, const float128& b){
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
#endif // CADNA_QUAD

#ifdef CADNA_QUAD
inline float128_st mul(const float128_st& a, const double& b){
  float128_st res;
  unsigned int random;
  float128 tmp;

  res.accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = a.x;
  FLOAT128_BIT_FLIP(tmp, random);
  res.x = tmp;
  tmp = b;
  res.x *= tmp;
  FLOAT128_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp = a.y;
  FLOAT128_BIT_FLIP(tmp, random);
  res.y = tmp;
  tmp = b;
  res.y *= tmp;
  FLOAT128_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp = a.z;
  FLOAT128_BIT_FLIP(tmp, random);
  res.z = tmp;
  tmp = b;
  res.z *= tmp;
  FLOAT128_BIT_FLIP(res.z, random);

  return res ;
}
inline float128_st mul(const float128_st& a, const float& b){
  float128_st res;
  unsigned int random;
  float128 tmp;

  res.accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = a.x;
  FLOAT128_BIT_FLIP(tmp, random);
  res.x = tmp;
  tmp = b;
  res.x *= tmp;
  FLOAT128_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp = a.y;
  FLOAT128_BIT_FLIP(tmp, random);
  res.y = tmp;
  tmp = b;
  res.y *= tmp;
  FLOAT128_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp = a.z;
  FLOAT128_BIT_FLIP(tmp, random);
  res.z = tmp;
  tmp = b;
  res.z *= tmp;
  FLOAT128_BIT_FLIP(res.z, random);

  return res ;
}
inline float128_st mul(const float128_st& a, const unsigned long long& b){
  float128_st res;
  unsigned int random;
  float128 tmp;

  res.accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = a.x;
  FLOAT128_BIT_FLIP(tmp, random);
  res.x = tmp;
  tmp = b;
  res.x *= tmp;
  FLOAT128_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp = a.y;
  FLOAT128_BIT_FLIP(tmp, random);
  res.y = tmp;
  tmp = b;
  res.y *= tmp;
  FLOAT128_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp = a.z;
  FLOAT128_BIT_FLIP(tmp, random);
  res.z = tmp;
  tmp = b;
  res.z *= tmp;
  FLOAT128_BIT_FLIP(res.z, random);

  return res ;
}
inline float128_st mul(const float128_st& a, const long long& b){
  float128_st res;
  unsigned int random;
  float128 tmp;

  res.accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = a.x;
  FLOAT128_BIT_FLIP(tmp, random);
  res.x = tmp;
  tmp = b;
  res.x *= tmp;
  FLOAT128_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp = a.y;
  FLOAT128_BIT_FLIP(tmp, random);
  res.y = tmp;
  tmp = b;
  res.y *= tmp;
  FLOAT128_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp = a.z;
  FLOAT128_BIT_FLIP(tmp, random);
  res.z = tmp;
  tmp = b;
  res.z *= tmp;
  FLOAT128_BIT_FLIP(res.z, random);

  return res ;
}
inline float128_st mul(const float128_st& a, const unsigned long& b){
  float128_st res;
  unsigned int random;
  float128 tmp;

  res.accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = a.x;
  FLOAT128_BIT_FLIP(tmp, random);
  res.x = tmp;
  tmp = b;
  res.x *= tmp;
  FLOAT128_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp = a.y;
  FLOAT128_BIT_FLIP(tmp, random);
  res.y = tmp;
  tmp = b;
  res.y *= tmp;
  FLOAT128_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp = a.z;
  FLOAT128_BIT_FLIP(tmp, random);
  res.z = tmp;
  tmp = b;
  res.z *= tmp;
  FLOAT128_BIT_FLIP(res.z, random);

  return res ;
}
inline float128_st mul(const float128_st& a, const long& b){
  float128_st res;
  unsigned int random;
  float128 tmp;

  res.accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = a.x;
  FLOAT128_BIT_FLIP(tmp, random);
  res.x = tmp;
  tmp = b;
  res.x *= tmp;
  FLOAT128_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp = a.y;
  FLOAT128_BIT_FLIP(tmp, random);
  res.y = tmp;
  tmp = b;
  res.y *= tmp;
  FLOAT128_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp = a.z;
  FLOAT128_BIT_FLIP(tmp, random);
  res.z = tmp;
  tmp = b;
  res.z *= tmp;
  FLOAT128_BIT_FLIP(res.z, random);

  return res ;
}
inline float128_st mul(const float128_st& a, const unsigned int & b){
  float128_st res;
  unsigned int random;
  float128 tmp;

  res.accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = a.x;
  FLOAT128_BIT_FLIP(tmp, random);
  res.x = tmp;
  tmp = b;
  res.x *= tmp;
  FLOAT128_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp = a.y;
  FLOAT128_BIT_FLIP(tmp, random);
  res.y = tmp;
  tmp = b;
  res.y *= tmp;
  FLOAT128_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp = a.z;
  FLOAT128_BIT_FLIP(tmp, random);
  res.z = tmp;
  tmp = b;
  res.z *= tmp;
  FLOAT128_BIT_FLIP(res.z, random);

  return res ;
}
inline float128_st mul(const float128_st& a, const int & b){
  float128_st res;
  unsigned int random;
  float128 tmp;

  res.accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = a.x;
  FLOAT128_BIT_FLIP(tmp, random);
  res.x = tmp;
  tmp = b;
  res.x *= tmp;
  FLOAT128_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp = a.y;
  FLOAT128_BIT_FLIP(tmp, random);
  res.y = tmp;
  tmp = b;
  res.y *= tmp;
  FLOAT128_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp = a.z;
  FLOAT128_BIT_FLIP(tmp, random);
  res.z = tmp;
  tmp = b;
  res.z *= tmp;
  FLOAT128_BIT_FLIP(res.z, random);

  return res ;
}
inline float128_st mul(const float128_st& a, const unsigned short& b){
  float128_st res;
  unsigned int random;
  float128 tmp;

  res.accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = a.x;
  FLOAT128_BIT_FLIP(tmp, random);
  res.x = tmp;
  tmp = b;
  res.x *= tmp;
  FLOAT128_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp = a.y;
  FLOAT128_BIT_FLIP(tmp, random);
  res.y = tmp;
  tmp = b;
  res.y *= tmp;
  FLOAT128_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp = a.z;
  FLOAT128_BIT_FLIP(tmp, random);
  res.z = tmp;
  tmp = b;
  res.z *= tmp;
  FLOAT128_BIT_FLIP(res.z, random);

  return res ;
}
inline float128_st mul(const float128_st& a, const short& b){
  float128_st res;
  unsigned int random;
  float128 tmp;

  res.accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = a.x;
  FLOAT128_BIT_FLIP(tmp, random);
  res.x = tmp;
  tmp = b;
  res.x *= tmp;
  FLOAT128_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp = a.y;
  FLOAT128_BIT_FLIP(tmp, random);
  res.y = tmp;
  tmp = b;
  res.y *= tmp;
  FLOAT128_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp = a.z;
  FLOAT128_BIT_FLIP(tmp, random);
  res.z = tmp;
  tmp = b;
  res.z *= tmp;
  FLOAT128_BIT_FLIP(res.z, random);

  return res ;
}
inline float128_st mul(const float128_st& a, const unsigned char& b){
  float128_st res;
  unsigned int random;
  float128 tmp;

  res.accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = a.x;
  FLOAT128_BIT_FLIP(tmp, random);
  res.x = tmp;
  tmp = b;
  res.x *= tmp;
  FLOAT128_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp = a.y;
  FLOAT128_BIT_FLIP(tmp, random);
  res.y = tmp;
  tmp = b;
  res.y *= tmp;
  FLOAT128_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp = a.z;
  FLOAT128_BIT_FLIP(tmp, random);
  res.z = tmp;
  tmp = b;
  res.z *= tmp;
  FLOAT128_BIT_FLIP(res.z, random);

  return res ;
}
inline float128_st mul(const float128_st& a, const char& b){
  float128_st res;
  unsigned int random;
  float128 tmp;

  res.accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = a.x;
  FLOAT128_BIT_FLIP(tmp, random);
  res.x = tmp;
  tmp = b;
  res.x *= tmp;
  FLOAT128_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp = a.y;
  FLOAT128_BIT_FLIP(tmp, random);
  res.y = tmp;
  tmp = b;
  res.y *= tmp;
  FLOAT128_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp = a.z;
  FLOAT128_BIT_FLIP(tmp, random);
  res.z = tmp;
  tmp = b;
  res.z *= tmp;
  FLOAT128_BIT_FLIP(res.z, random);

  return res ;
}
inline float128_st mul(const float128_st& a, const float128& b){
  float128_st res;
  unsigned int random;
  float128 tmp;

  res.accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = a.x;
  FLOAT128_BIT_FLIP(tmp, random);
  res.x = tmp;
  tmp = b;
  res.x *= tmp;
  FLOAT128_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp = a.y;
  FLOAT128_BIT_FLIP(tmp, random);
  res.y = tmp;
  tmp = b;
  res.y *= tmp;
  FLOAT128_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp = a.z;
  FLOAT128_BIT_FLIP(tmp, random);
  res.z = tmp;
  tmp = b;
  res.z *= tmp;
  FLOAT128_BIT_FLIP(res.z, random);

  return res ;
}
#endif // CADNA_QUAD


/////////////////////////////////////////////////////////////////////////



inline double_st mul(const double_st& a, const double_st& b){
  double_st res;
  unsigned int random;
  double tmp;

  res.accuracy=DIGIT_NOT_COMPUTED;

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
inline double_st mul(const double_st& a, const float_st& b){
  double_st res;
  unsigned int random;
  double tmp;

  res.accuracy=DIGIT_NOT_COMPUTED;

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
inline double_st mul(const float_st& a, const double_st& b){
  double_st res;
  unsigned int random;
  double tmp;

  res.accuracy=DIGIT_NOT_COMPUTED;

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
inline float_st mul(const float_st& a, const float_st& b){
  float_st res;
  unsigned int random;
  float tmp;

  res.accuracy=DIGIT_NOT_COMPUTED;

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
#ifdef CADNA_QUAD
inline float128_st mul(const float128_st& a, const float128_st& b){
  float128_st res;
  unsigned int random;
  float128 tmp;

  res.accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = a.x;
  FLOAT128_BIT_FLIP(tmp, random);
  res.x = tmp;
  tmp = b.x;
  res.x *= tmp;
  FLOAT128_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp = a.y;
  FLOAT128_BIT_FLIP(tmp, random);
  res.y = tmp;
  tmp = b.y;
  res.y *= tmp;
  FLOAT128_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp = a.z;
  FLOAT128_BIT_FLIP(tmp, random);
  res.z = tmp;
  tmp = b.z;
  res.z *= tmp;
  FLOAT128_BIT_FLIP(res.z, random);

  return res ;
}
inline float128_st mul(const float128_st& a, const float_st& b){
  float128_st res;
  unsigned int random;
  float128 tmp;

  res.accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = a.x;
  FLOAT128_BIT_FLIP(tmp, random);
  res.x = tmp;
  tmp = b.x;
  res.x *= tmp;
  FLOAT128_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp = a.y;
  FLOAT128_BIT_FLIP(tmp, random);
  res.y = tmp;
  tmp = b.y;
  res.y *= tmp;
  FLOAT128_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp = a.z;
  FLOAT128_BIT_FLIP(tmp, random);
  res.z = tmp;
  tmp = b.z;
  res.z *= tmp;
  FLOAT128_BIT_FLIP(res.z, random);

  return res ;
}
inline float128_st mul(const float_st& a, const float128_st& b){
  float128_st res;
  unsigned int random;
  float128 tmp;

  res.accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = a.x;
  FLOAT128_BIT_FLIP(tmp, random);
  res.x = tmp;
  tmp = b.x;
  res.x *= tmp;
  FLOAT128_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp = a.y;
  FLOAT128_BIT_FLIP(tmp, random);
  res.y = tmp;
  tmp = b.y;
  res.y *= tmp;
  FLOAT128_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp = a.z;
  FLOAT128_BIT_FLIP(tmp, random);
  res.z = tmp;
  tmp = b.z;
  res.z *= tmp;
  FLOAT128_BIT_FLIP(res.z, random);

  return res ;
}
inline float128_st mul(const float128_st& a, const double_st& b){
  float128_st res;
  unsigned int random;
  float128 tmp;

  res.accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = a.x;
  FLOAT128_BIT_FLIP(tmp, random);
  res.x = tmp;
  tmp = b.x;
  res.x *= tmp;
  FLOAT128_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp = a.y;
  FLOAT128_BIT_FLIP(tmp, random);
  res.y = tmp;
  tmp = b.y;
  res.y *= tmp;
  FLOAT128_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp = a.z;
  FLOAT128_BIT_FLIP(tmp, random);
  res.z = tmp;
  tmp = b.z;
  res.z *= tmp;
  FLOAT128_BIT_FLIP(res.z, random);

  return res ;
}
inline float128_st mul(const double_st& a, const float128_st& b){
  float128_st res;
  unsigned int random;
  float128 tmp;

  res.accuracy=DIGIT_NOT_COMPUTED;

  random = RANDOM;
  tmp = a.x;
  FLOAT128_BIT_FLIP(tmp, random);
  res.x = tmp;
  tmp = b.x;
  res.x *= tmp;
  FLOAT128_BIT_FLIP(res.x, random);

  random = RANDOM;
  tmp = a.y;
  FLOAT128_BIT_FLIP(tmp, random);
  res.y = tmp;
  tmp = b.y;
  res.y *= tmp;
  FLOAT128_BIT_FLIP(res.y, random);

  random = 1^random;
  tmp = a.z;
  FLOAT128_BIT_FLIP(tmp, random);
  res.z = tmp;
  tmp = b.z;
  res.z *= tmp;
  FLOAT128_BIT_FLIP(res.z, random);

  return res ;
}
#endif

/////////////////////////////////////////////////////////////////////////


///////////////////////////////////////////////////
//****m* cadna_comp/twoSum
//    NAME
//      add
//    SYNOPSIS
//      
//    FUNCTION
//    Defines all the functions involving at least one
//    stochastic argument which compute the twoSum algorithm
//
//
//    INPUTS/OUTPUTS
//      a           - an integer, a float, a double or a stochastic number
//      b           - an integer, a float, a double or a stochastic number
//	s	    - a stochastic number: the result of "a+b"
//	e	    - a stochastic number: the computed error between "s"
//		      	and the real value of "a+b"
//      At least one argument must be of stochastic type.
//
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




inline void twoSum( double_st& a,  double_st& b, double_st& s, double_st&e){
  double_st t;
  s = (a) + b;
  t = sub(s,a);
  e = add((sub(a, sub(s, t))),(sub(b, t)));
}
inline void twoSum( double_st& a,  double_st& b, double_st& s, float_st&e){
  double_st t;
  s = (a) + b;
  t = sub(s,a);
  e = add((sub(a, sub(s, t))),(sub(b, t)));
}
inline void twoSum( float_st& a,  double_st& b, double_st& s, double_st&e){
  double_st t;
  s = (a) + b;
  t = sub(s,a);
  e = add((sub(a, sub(s, t))),(sub(b, t)));
}
inline void twoSum( float_st& a,  double_st& b, double_st& s, float_st&e){
  double_st t;
  s = (a) + b;
  t = sub(s,a);
  e = add((sub(a, sub(s, t))),(sub(b, t)));
}
inline void twoSum( double_st& a,  float_st& b, double_st& s, double_st&e){
  double_st t;
  s = (a) + b;
  t = sub(s,a);
  e = add((sub(a, sub(s, t))),(sub(b, t)));
}
inline void twoSum( double_st& a,  float_st& b, double_st& s, float_st&e){
  double_st t;
  s = (a) + b;
  t = sub(s,a);
  e = add((sub(a, sub(s, t))),(sub(b, t)));
}
inline void twoSum( double_st& a,  double_st& b, float_st& s, double_st&e){
  double_st t;
  s = (a) + b;
  t = sub(s,a);
  e = add((sub(a, sub(s, t))),(sub(b, t)));
}
inline void twoSum( double_st& a,  double_st& b, float_st& s, float_st&e){
  double_st t;
  s = (a) + b;
  t = sub(s,a);
  e = add((sub(a, sub(s, t))),(sub(b, t)));
}
inline void twoSum( float_st& a,  float_st& b, double_st& s, double_st&e){
  double_st t;
  s = (a) + b;
  t = sub(s,a);
  e = add((sub(a, sub(s, t))),(sub(b, t)));
}
inline void twoSum( float_st& a,  float_st& b, double_st& s, float_st&e){
  double_st t;
  s = (a) + b;
  t = sub(s,a);
  e = add((sub(a, sub(s, t))),(sub(b, t)));
}
inline void twoSum( float_st& a,  double_st& b, float_st& s, double_st&e){
  double_st t;
  s = (a) + b;
  t = sub(s,a);
  e = add((sub(a, sub(s, t))),(sub(b, t)));
}
inline void twoSum( float_st& a,  double_st& b, float_st& s, float_st&e){
  double_st t;
  s = (a) + b;
  t = sub(s,a);
  e = add((sub(a, sub(s, t))),(sub(b, t)));
}
inline void twoSum( double_st& a,  float_st& b, float_st& s, double_st&e){
  double_st t;
  s = (a) + b;
  t = sub(s,a);
  e = add((sub(a, sub(s, t))),(sub(b, t)));
}
inline void twoSum( double_st& a,  float_st& b, float_st& s, float_st&e){
  double_st t;
  s = (a) + b;
  t = sub(s,a);
  e = add((sub(a, sub(s, t))),(sub(b, t)));
}
inline void twoSum( float_st& a,  float_st& b, float_st& s, float_st&e){
  float_st t;
  s = (a) + b;
  t = sub(s,a);
  e = add((sub(a, sub(s, t))),(sub(b, t)));
}


////////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////
//****m* cadna_comp/fastTwoSum
//    NAME
//      add
//    SYNOPSIS
//      
//    FUNCTION
//    Defines all the functions involving at least one
//    stochastic argument which compute the fastTwoSum algorithm
//
//
//    INPUTS/OUTPUTS
//      a           - an integer, a float, a double or a stochastic number
//      b           - an integer, a float, a double or a stochastic number
//	s	    - a stochastic number: the result of "a+b"
//	e	    - a stochastic number: the computed error between "s"
//		      	and the real value of "a+b"
//      At least one argument must be of stochastic type.
//
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




inline void fastTwoSum( double_st& a,  double_st& b, double_st& s, double_st&e){
  double_st a1;
  double_st b1;		

  if(fabs(a.x) < fabs(b.x)){
    b1.x = a.x;
    a1.x = b.x;
  } else {
    a1.x = a.x;
    b1.x = b.x;
  }

  if(fabs(a.y) < fabs(b.y)){
    b1.y = a.y;
    a1.y = b.y;
  } else {
    a1.y = a.y;
    b1.y = b.y;
  }

  if(fabs(a.z) < fabs(b.z)){
    b1.z = a.z;
    a1.z = b.z;
  } else {
    a1.z = a.z;
    b1.z = b.z;
  }
  
  s = (a1) + b1; 
  e = sub(b1, sub(s, a1));
}
inline void fastTwoSum( double_st& a,  double_st& b, double_st& s, float_st&e){
  double_st a1;
  double_st b1;		

  if(fabs(a.x) < fabs(b.x)){
    b1.x = a.x;
    a1.x = b.x;
  } else {
    a1.x = a.x;
    b1.x = b.x;
  }

  if(fabs(a.y) < fabs(b.y)){
    b1.y = a.y;
    a1.y = b.y;
  } else {
    a1.y = a.y;
    b1.y = b.y;
  }

  if(fabs(a.z) < fabs(b.z)){
    b1.z = a.z;
    a1.z = b.z;
  } else {
    a1.z = a.z;
    b1.z = b.z;
  }
  
  s = (a1) + b1; 
  e = sub(b1, sub(s, a1));
}
inline void fastTwoSum( float_st& a,  double_st& b, double_st& s, double_st&e){
  double_st a1;
  double_st b1;		

  if(fabs(a.x) < fabs(b.x)){
    b1.x = a.x;
    a1.x = b.x;
  } else {
    a1.x = a.x;
    b1.x = b.x;
  }

  if(fabs(a.y) < fabs(b.y)){
    b1.y = a.y;
    a1.y = b.y;
  } else {
    a1.y = a.y;
    b1.y = b.y;
  }

  if(fabs(a.z) < fabs(b.z)){
    b1.z = a.z;
    a1.z = b.z;
  } else {
    a1.z = a.z;
    b1.z = b.z;
  }
  
  s = (a1) + b1; 
  e = sub(b1, sub(s, a1));
}
inline void fastTwoSum( float_st& a,  double_st& b, double_st& s, float_st&e){
  double_st a1;
  double_st b1;		

  if(fabs(a.x) < fabs(b.x)){
    b1.x = a.x;
    a1.x = b.x;
  } else {
    a1.x = a.x;
    b1.x = b.x;
  }

  if(fabs(a.y) < fabs(b.y)){
    b1.y = a.y;
    a1.y = b.y;
  } else {
    a1.y = a.y;
    b1.y = b.y;
  }

  if(fabs(a.z) < fabs(b.z)){
    b1.z = a.z;
    a1.z = b.z;
  } else {
    a1.z = a.z;
    b1.z = b.z;
  }
  
  s = (a1) + b1; 
  e = sub(b1, sub(s, a1));
}
inline void fastTwoSum( double_st& a,  float_st& b, double_st& s, double_st&e){
  double_st a1;
  double_st b1;		

  if(fabs(a.x) < fabs(b.x)){
    b1.x = a.x;
    a1.x = b.x;
  } else {
    a1.x = a.x;
    b1.x = b.x;
  }

  if(fabs(a.y) < fabs(b.y)){
    b1.y = a.y;
    a1.y = b.y;
  } else {
    a1.y = a.y;
    b1.y = b.y;
  }

  if(fabs(a.z) < fabs(b.z)){
    b1.z = a.z;
    a1.z = b.z;
  } else {
    a1.z = a.z;
    b1.z = b.z;
  }
  
  s = (a1) + b1; 
  e = sub(b1, sub(s, a1));
}
inline void fastTwoSum( double_st& a,  float_st& b, double_st& s, float_st&e){
  double_st a1;
  double_st b1;		

  if(fabs(a.x) < fabs(b.x)){
    b1.x = a.x;
    a1.x = b.x;
  } else {
    a1.x = a.x;
    b1.x = b.x;
  }

  if(fabs(a.y) < fabs(b.y)){
    b1.y = a.y;
    a1.y = b.y;
  } else {
    a1.y = a.y;
    b1.y = b.y;
  }

  if(fabs(a.z) < fabs(b.z)){
    b1.z = a.z;
    a1.z = b.z;
  } else {
    a1.z = a.z;
    b1.z = b.z;
  }
  
  s = (a1) + b1; 
  e = sub(b1, sub(s, a1));
}
inline void fastTwoSum( double_st& a,  double_st& b, float_st& s, double_st&e){
  double_st a1;
  double_st b1;		

  if(fabs(a.x) < fabs(b.x)){
    b1.x = a.x;
    a1.x = b.x;
  } else {
    a1.x = a.x;
    b1.x = b.x;
  }

  if(fabs(a.y) < fabs(b.y)){
    b1.y = a.y;
    a1.y = b.y;
  } else {
    a1.y = a.y;
    b1.y = b.y;
  }

  if(fabs(a.z) < fabs(b.z)){
    b1.z = a.z;
    a1.z = b.z;
  } else {
    a1.z = a.z;
    b1.z = b.z;
  }
  
  s = (a1) + b1; 
  e = sub(b1, sub(s, a1));
}
inline void fastTwoSum( double_st& a,  double_st& b, float_st& s, float_st&e){
  double_st a1;
  double_st b1;		

  if(fabs(a.x) < fabs(b.x)){
    b1.x = a.x;
    a1.x = b.x;
  } else {
    a1.x = a.x;
    b1.x = b.x;
  }

  if(fabs(a.y) < fabs(b.y)){
    b1.y = a.y;
    a1.y = b.y;
  } else {
    a1.y = a.y;
    b1.y = b.y;
  }

  if(fabs(a.z) < fabs(b.z)){
    b1.z = a.z;
    a1.z = b.z;
  } else {
    a1.z = a.z;
    b1.z = b.z;
  }
  
  s = (a1) + b1; 
  e = sub(b1, sub(s, a1));
}
inline void fastTwoSum( float_st& a,  float_st& b, double_st& s, double_st&e){
  double_st a1;
  double_st b1;		

  if(fabs(a.x) < fabs(b.x)){
    b1.x = a.x;
    a1.x = b.x;
  } else {
    a1.x = a.x;
    b1.x = b.x;
  }

  if(fabs(a.y) < fabs(b.y)){
    b1.y = a.y;
    a1.y = b.y;
  } else {
    a1.y = a.y;
    b1.y = b.y;
  }

  if(fabs(a.z) < fabs(b.z)){
    b1.z = a.z;
    a1.z = b.z;
  } else {
    a1.z = a.z;
    b1.z = b.z;
  }
  
  s = (a1) + b1; 
  e = sub(b1, sub(s, a1));
}
inline void fastTwoSum( float_st& a,  float_st& b, double_st& s, float_st&e){
  double_st a1;
  double_st b1;		

  if(fabs(a.x) < fabs(b.x)){
    b1.x = a.x;
    a1.x = b.x;
  } else {
    a1.x = a.x;
    b1.x = b.x;
  }

  if(fabs(a.y) < fabs(b.y)){
    b1.y = a.y;
    a1.y = b.y;
  } else {
    a1.y = a.y;
    b1.y = b.y;
  }

  if(fabs(a.z) < fabs(b.z)){
    b1.z = a.z;
    a1.z = b.z;
  } else {
    a1.z = a.z;
    b1.z = b.z;
  }
  
  s = (a1) + b1; 
  e = sub(b1, sub(s, a1));
}
inline void fastTwoSum( float_st& a,  double_st& b, float_st& s, double_st&e){
  double_st a1;
  double_st b1;		

  if(fabs(a.x) < fabs(b.x)){
    b1.x = a.x;
    a1.x = b.x;
  } else {
    a1.x = a.x;
    b1.x = b.x;
  }

  if(fabs(a.y) < fabs(b.y)){
    b1.y = a.y;
    a1.y = b.y;
  } else {
    a1.y = a.y;
    b1.y = b.y;
  }

  if(fabs(a.z) < fabs(b.z)){
    b1.z = a.z;
    a1.z = b.z;
  } else {
    a1.z = a.z;
    b1.z = b.z;
  }
  
  s = (a1) + b1; 
  e = sub(b1, sub(s, a1));
}
inline void fastTwoSum( float_st& a,  double_st& b, float_st& s, float_st&e){
  double_st a1;
  double_st b1;		

  if(fabs(a.x) < fabs(b.x)){
    b1.x = a.x;
    a1.x = b.x;
  } else {
    a1.x = a.x;
    b1.x = b.x;
  }

  if(fabs(a.y) < fabs(b.y)){
    b1.y = a.y;
    a1.y = b.y;
  } else {
    a1.y = a.y;
    b1.y = b.y;
  }

  if(fabs(a.z) < fabs(b.z)){
    b1.z = a.z;
    a1.z = b.z;
  } else {
    a1.z = a.z;
    b1.z = b.z;
  }
  
  s = (a1) + b1; 
  e = sub(b1, sub(s, a1));
}
inline void fastTwoSum( double_st& a,  float_st& b, float_st& s, double_st&e){
  double_st a1;
  double_st b1;		

  if(fabs(a.x) < fabs(b.x)){
    b1.x = a.x;
    a1.x = b.x;
  } else {
    a1.x = a.x;
    b1.x = b.x;
  }

  if(fabs(a.y) < fabs(b.y)){
    b1.y = a.y;
    a1.y = b.y;
  } else {
    a1.y = a.y;
    b1.y = b.y;
  }

  if(fabs(a.z) < fabs(b.z)){
    b1.z = a.z;
    a1.z = b.z;
  } else {
    a1.z = a.z;
    b1.z = b.z;
  }
  
  s = (a1) + b1; 
  e = sub(b1, sub(s, a1));
}
inline void fastTwoSum( double_st& a,  float_st& b, float_st& s, float_st&e){
  double_st a1;
  double_st b1;		

  if(fabs(a.x) < fabs(b.x)){
    b1.x = a.x;
    a1.x = b.x;
  } else {
    a1.x = a.x;
    b1.x = b.x;
  }

  if(fabs(a.y) < fabs(b.y)){
    b1.y = a.y;
    a1.y = b.y;
  } else {
    a1.y = a.y;
    b1.y = b.y;
  }

  if(fabs(a.z) < fabs(b.z)){
    b1.z = a.z;
    a1.z = b.z;
  } else {
    a1.z = a.z;
    b1.z = b.z;
  }
  
  s = (a1) + b1; 
  e = sub(b1, sub(s, a1));
}
inline void fastTwoSum( float_st& a,  float_st& b, float_st& s, float_st&e){
  float_st a1;
  float_st b1;		

  if(fabs(a.x) < fabs(b.x)){
    b1.x = a.x;
    a1.x = b.x;
  } else {
    a1.x = a.x;
    b1.x = b.x;
  }

  if(fabs(a.y) < fabs(b.y)){
    b1.y = a.y;
    a1.y = b.y;
  } else {
    a1.y = a.y;
    b1.y = b.y;
  }

  if(fabs(a.z) < fabs(b.z)){
    b1.z = a.z;
    a1.z = b.z;
  } else {
    a1.z = a.z;
    b1.z = b.z;
  }
  
  s = (a1) + b1; 
  e = sub(b1, sub(s, a1));
}


////////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////
//****m* cadna_comp/twoSumPriest
//    NAME
//      add
//    SYNOPSIS
//      
//    FUNCTION
//    Defines all the functions involving at least one
//    stochastic argument which compute the twoSumPriest algorithm
//
//
//    INPUTS/OUTPUTS
//      a           - an integer, a float, a double or a stochastic number
//      b           - an integer, a float, a double or a stochastic number
//	s	    - a stochastic number: the result of "a+b"
//	e	    - a stochastic number: the computed error between "s"
//		      	and the real value of "a+b"
//      At least one argument must be of stochastic type.
//
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




inline void twoSumPriest( double_st& a,  double_st& b, double_st& c, double_st& d){
  double_st a1;
  double_st b1;
  double_st e;
  double_st f;
  double_st g;
  double_st h;
  double_st tmp;
    
  if (fabs(a.x) < fabs(b.x))
    {
      a1.x = b.x;
      b1.x = a.x;
    }
  else
    {
      a1.x = a.x;
      b1.x = b.x;
    }

 if (fabs(a.y) < fabs(b.y))
    {
      a1.y = b.y;
      b1.y = a.y;
    }
  else
    {
      a1.y = a.y;
      b1.y = b.y;
    }

 if (fabs(a.z) < fabs(b.z))
    {
      a1.z = b.z;
      b1.z = a.z;
    }
  else
    {
      a1.z = a.z;
      b1.z = b.z;
    }

  c = add(a1, b1);
  e = sub(c, a1);
  g = sub(c, e);
  h = sub(g, a1);
  f = sub(b1, h);
  d = sub(f, e);

  tmp = add(d,e);
  if((tmp.x)!=f.x)
    {
      c.x = a1.x;
      d.x = b1.x;
    }
 if((tmp.y)!=f.y)
    {
      c.y = a1.y;
      d.y = b1.y;
    }
 if((tmp.z)!=f.z)
    {
      c.z = a1.z;
      d.z = b1.z;
    }

 c.accuracy=DIGIT_NOT_COMPUTED;
   if (_cadna_cancel_tag){
    if (a.accuracy==DIGIT_NOT_COMPUTED || a.accuracy==RELIABLE_RESULT)
	a.nb_significant_digit();
    if (b.accuracy==DIGIT_NOT_COMPUTED || b.accuracy==RELIABLE_RESULT)
	b.nb_significant_digit();
    if ((a.accuracy < b.accuracy ? a.accuracy: b.accuracy) >=
	  c.nb_significant_digit()+_cadna_cancel_value)
	instability(&_cadna_cancel_count);
  }


}
inline void twoSumPriest( double_st& a,  double_st& b, double_st& c, float_st& d){
  double_st a1;
  double_st b1;
  double_st e;
  double_st f;
  double_st g;
  double_st h;
  double_st tmp;
    
  if (fabs(a.x) < fabs(b.x))
    {
      a1.x = b.x;
      b1.x = a.x;
    }
  else
    {
      a1.x = a.x;
      b1.x = b.x;
    }

 if (fabs(a.y) < fabs(b.y))
    {
      a1.y = b.y;
      b1.y = a.y;
    }
  else
    {
      a1.y = a.y;
      b1.y = b.y;
    }

 if (fabs(a.z) < fabs(b.z))
    {
      a1.z = b.z;
      b1.z = a.z;
    }
  else
    {
      a1.z = a.z;
      b1.z = b.z;
    }

  c = add(a1, b1);
  e = sub(c, a1);
  g = sub(c, e);
  h = sub(g, a1);
  f = sub(b1, h);
  d = sub(f, e);

  tmp = add(d,e);
  if((tmp.x)!=f.x)
    {
      c.x = a1.x;
      d.x = b1.x;
    }
 if((tmp.y)!=f.y)
    {
      c.y = a1.y;
      d.y = b1.y;
    }
 if((tmp.z)!=f.z)
    {
      c.z = a1.z;
      d.z = b1.z;
    }

 c.accuracy=DIGIT_NOT_COMPUTED;
   if (_cadna_cancel_tag){
    if (a.accuracy==DIGIT_NOT_COMPUTED || a.accuracy==RELIABLE_RESULT)
	a.nb_significant_digit();
    if (b.accuracy==DIGIT_NOT_COMPUTED || b.accuracy==RELIABLE_RESULT)
	b.nb_significant_digit();
    if ((a.accuracy < b.accuracy ? a.accuracy: b.accuracy) >=
	  c.nb_significant_digit()+_cadna_cancel_value)
	instability(&_cadna_cancel_count);
  }


}
inline void twoSumPriest( float_st& a,  double_st& b, double_st& c, double_st& d){
  double_st a1;
  double_st b1;
  double_st e;
  double_st f;
  double_st g;
  double_st h;
  double_st tmp;
    
  if (fabs(a.x) < fabs(b.x))
    {
      a1.x = b.x;
      b1.x = a.x;
    }
  else
    {
      a1.x = a.x;
      b1.x = b.x;
    }

 if (fabs(a.y) < fabs(b.y))
    {
      a1.y = b.y;
      b1.y = a.y;
    }
  else
    {
      a1.y = a.y;
      b1.y = b.y;
    }

 if (fabs(a.z) < fabs(b.z))
    {
      a1.z = b.z;
      b1.z = a.z;
    }
  else
    {
      a1.z = a.z;
      b1.z = b.z;
    }

  c = add(a1, b1);
  e = sub(c, a1);
  g = sub(c, e);
  h = sub(g, a1);
  f = sub(b1, h);
  d = sub(f, e);

  tmp = add(d,e);
  if((tmp.x)!=f.x)
    {
      c.x = a1.x;
      d.x = b1.x;
    }
 if((tmp.y)!=f.y)
    {
      c.y = a1.y;
      d.y = b1.y;
    }
 if((tmp.z)!=f.z)
    {
      c.z = a1.z;
      d.z = b1.z;
    }

 c.accuracy=DIGIT_NOT_COMPUTED;
   if (_cadna_cancel_tag){
    if (a.accuracy==DIGIT_NOT_COMPUTED || a.accuracy==RELIABLE_RESULT)
	a.nb_significant_digit();
    if (b.accuracy==DIGIT_NOT_COMPUTED || b.accuracy==RELIABLE_RESULT)
	b.nb_significant_digit();
    if ((a.accuracy < b.accuracy ? a.accuracy: b.accuracy) >=
	  c.nb_significant_digit()+_cadna_cancel_value)
	instability(&_cadna_cancel_count);
  }


}
inline void twoSumPriest( float_st& a,  double_st& b, double_st& c, float_st& d){
  double_st a1;
  double_st b1;
  double_st e;
  double_st f;
  double_st g;
  double_st h;
  double_st tmp;
    
  if (fabs(a.x) < fabs(b.x))
    {
      a1.x = b.x;
      b1.x = a.x;
    }
  else
    {
      a1.x = a.x;
      b1.x = b.x;
    }

 if (fabs(a.y) < fabs(b.y))
    {
      a1.y = b.y;
      b1.y = a.y;
    }
  else
    {
      a1.y = a.y;
      b1.y = b.y;
    }

 if (fabs(a.z) < fabs(b.z))
    {
      a1.z = b.z;
      b1.z = a.z;
    }
  else
    {
      a1.z = a.z;
      b1.z = b.z;
    }

  c = add(a1, b1);
  e = sub(c, a1);
  g = sub(c, e);
  h = sub(g, a1);
  f = sub(b1, h);
  d = sub(f, e);

  tmp = add(d,e);
  if((tmp.x)!=f.x)
    {
      c.x = a1.x;
      d.x = b1.x;
    }
 if((tmp.y)!=f.y)
    {
      c.y = a1.y;
      d.y = b1.y;
    }
 if((tmp.z)!=f.z)
    {
      c.z = a1.z;
      d.z = b1.z;
    }

 c.accuracy=DIGIT_NOT_COMPUTED;
   if (_cadna_cancel_tag){
    if (a.accuracy==DIGIT_NOT_COMPUTED || a.accuracy==RELIABLE_RESULT)
	a.nb_significant_digit();
    if (b.accuracy==DIGIT_NOT_COMPUTED || b.accuracy==RELIABLE_RESULT)
	b.nb_significant_digit();
    if ((a.accuracy < b.accuracy ? a.accuracy: b.accuracy) >=
	  c.nb_significant_digit()+_cadna_cancel_value)
	instability(&_cadna_cancel_count);
  }


}
inline void twoSumPriest( double_st& a,  float_st& b, double_st& c, double_st& d){
  double_st a1;
  double_st b1;
  double_st e;
  double_st f;
  double_st g;
  double_st h;
  double_st tmp;
    
  if (fabs(a.x) < fabs(b.x))
    {
      a1.x = b.x;
      b1.x = a.x;
    }
  else
    {
      a1.x = a.x;
      b1.x = b.x;
    }

 if (fabs(a.y) < fabs(b.y))
    {
      a1.y = b.y;
      b1.y = a.y;
    }
  else
    {
      a1.y = a.y;
      b1.y = b.y;
    }

 if (fabs(a.z) < fabs(b.z))
    {
      a1.z = b.z;
      b1.z = a.z;
    }
  else
    {
      a1.z = a.z;
      b1.z = b.z;
    }

  c = add(a1, b1);
  e = sub(c, a1);
  g = sub(c, e);
  h = sub(g, a1);
  f = sub(b1, h);
  d = sub(f, e);

  tmp = add(d,e);
  if((tmp.x)!=f.x)
    {
      c.x = a1.x;
      d.x = b1.x;
    }
 if((tmp.y)!=f.y)
    {
      c.y = a1.y;
      d.y = b1.y;
    }
 if((tmp.z)!=f.z)
    {
      c.z = a1.z;
      d.z = b1.z;
    }

 c.accuracy=DIGIT_NOT_COMPUTED;
   if (_cadna_cancel_tag){
    if (a.accuracy==DIGIT_NOT_COMPUTED || a.accuracy==RELIABLE_RESULT)
	a.nb_significant_digit();
    if (b.accuracy==DIGIT_NOT_COMPUTED || b.accuracy==RELIABLE_RESULT)
	b.nb_significant_digit();
    if ((a.accuracy < b.accuracy ? a.accuracy: b.accuracy) >=
	  c.nb_significant_digit()+_cadna_cancel_value)
	instability(&_cadna_cancel_count);
  }


}
inline void twoSumPriest( double_st& a,  float_st& b, double_st& c, float_st& d){
  double_st a1;
  double_st b1;
  double_st e;
  double_st f;
  double_st g;
  double_st h;
  double_st tmp;
    
  if (fabs(a.x) < fabs(b.x))
    {
      a1.x = b.x;
      b1.x = a.x;
    }
  else
    {
      a1.x = a.x;
      b1.x = b.x;
    }

 if (fabs(a.y) < fabs(b.y))
    {
      a1.y = b.y;
      b1.y = a.y;
    }
  else
    {
      a1.y = a.y;
      b1.y = b.y;
    }

 if (fabs(a.z) < fabs(b.z))
    {
      a1.z = b.z;
      b1.z = a.z;
    }
  else
    {
      a1.z = a.z;
      b1.z = b.z;
    }

  c = add(a1, b1);
  e = sub(c, a1);
  g = sub(c, e);
  h = sub(g, a1);
  f = sub(b1, h);
  d = sub(f, e);

  tmp = add(d,e);
  if((tmp.x)!=f.x)
    {
      c.x = a1.x;
      d.x = b1.x;
    }
 if((tmp.y)!=f.y)
    {
      c.y = a1.y;
      d.y = b1.y;
    }
 if((tmp.z)!=f.z)
    {
      c.z = a1.z;
      d.z = b1.z;
    }

 c.accuracy=DIGIT_NOT_COMPUTED;
   if (_cadna_cancel_tag){
    if (a.accuracy==DIGIT_NOT_COMPUTED || a.accuracy==RELIABLE_RESULT)
	a.nb_significant_digit();
    if (b.accuracy==DIGIT_NOT_COMPUTED || b.accuracy==RELIABLE_RESULT)
	b.nb_significant_digit();
    if ((a.accuracy < b.accuracy ? a.accuracy: b.accuracy) >=
	  c.nb_significant_digit()+_cadna_cancel_value)
	instability(&_cadna_cancel_count);
  }


}
inline void twoSumPriest( double_st& a,  double_st& b, float_st& c, double_st& d){
  double_st a1;
  double_st b1;
  double_st e;
  double_st f;
  double_st g;
  double_st h;
  double_st tmp;
    
  if (fabs(a.x) < fabs(b.x))
    {
      a1.x = b.x;
      b1.x = a.x;
    }
  else
    {
      a1.x = a.x;
      b1.x = b.x;
    }

 if (fabs(a.y) < fabs(b.y))
    {
      a1.y = b.y;
      b1.y = a.y;
    }
  else
    {
      a1.y = a.y;
      b1.y = b.y;
    }

 if (fabs(a.z) < fabs(b.z))
    {
      a1.z = b.z;
      b1.z = a.z;
    }
  else
    {
      a1.z = a.z;
      b1.z = b.z;
    }

  c = add(a1, b1);
  e = sub(c, a1);
  g = sub(c, e);
  h = sub(g, a1);
  f = sub(b1, h);
  d = sub(f, e);

  tmp = add(d,e);
  if((tmp.x)!=f.x)
    {
      c.x = a1.x;
      d.x = b1.x;
    }
 if((tmp.y)!=f.y)
    {
      c.y = a1.y;
      d.y = b1.y;
    }
 if((tmp.z)!=f.z)
    {
      c.z = a1.z;
      d.z = b1.z;
    }

 c.accuracy=DIGIT_NOT_COMPUTED;
   if (_cadna_cancel_tag){
    if (a.accuracy==DIGIT_NOT_COMPUTED || a.accuracy==RELIABLE_RESULT)
	a.nb_significant_digit();
    if (b.accuracy==DIGIT_NOT_COMPUTED || b.accuracy==RELIABLE_RESULT)
	b.nb_significant_digit();
    if ((a.accuracy < b.accuracy ? a.accuracy: b.accuracy) >=
	  c.nb_significant_digit()+_cadna_cancel_value)
	instability(&_cadna_cancel_count);
  }


}
inline void twoSumPriest( double_st& a,  double_st& b, float_st& c, float_st& d){
  double_st a1;
  double_st b1;
  double_st e;
  double_st f;
  double_st g;
  double_st h;
  double_st tmp;
    
  if (fabs(a.x) < fabs(b.x))
    {
      a1.x = b.x;
      b1.x = a.x;
    }
  else
    {
      a1.x = a.x;
      b1.x = b.x;
    }

 if (fabs(a.y) < fabs(b.y))
    {
      a1.y = b.y;
      b1.y = a.y;
    }
  else
    {
      a1.y = a.y;
      b1.y = b.y;
    }

 if (fabs(a.z) < fabs(b.z))
    {
      a1.z = b.z;
      b1.z = a.z;
    }
  else
    {
      a1.z = a.z;
      b1.z = b.z;
    }

  c = add(a1, b1);
  e = sub(c, a1);
  g = sub(c, e);
  h = sub(g, a1);
  f = sub(b1, h);
  d = sub(f, e);

  tmp = add(d,e);
  if((tmp.x)!=f.x)
    {
      c.x = a1.x;
      d.x = b1.x;
    }
 if((tmp.y)!=f.y)
    {
      c.y = a1.y;
      d.y = b1.y;
    }
 if((tmp.z)!=f.z)
    {
      c.z = a1.z;
      d.z = b1.z;
    }

 c.accuracy=DIGIT_NOT_COMPUTED;
   if (_cadna_cancel_tag){
    if (a.accuracy==DIGIT_NOT_COMPUTED || a.accuracy==RELIABLE_RESULT)
	a.nb_significant_digit();
    if (b.accuracy==DIGIT_NOT_COMPUTED || b.accuracy==RELIABLE_RESULT)
	b.nb_significant_digit();
    if ((a.accuracy < b.accuracy ? a.accuracy: b.accuracy) >=
	  c.nb_significant_digit()+_cadna_cancel_value)
	instability(&_cadna_cancel_count);
  }


}
inline void twoSumPriest( float_st& a,  float_st& b, double_st& c, double_st& d){
  double_st a1;
  double_st b1;
  double_st e;
  double_st f;
  double_st g;
  double_st h;
  double_st tmp;
    
  if (fabs(a.x) < fabs(b.x))
    {
      a1.x = b.x;
      b1.x = a.x;
    }
  else
    {
      a1.x = a.x;
      b1.x = b.x;
    }

 if (fabs(a.y) < fabs(b.y))
    {
      a1.y = b.y;
      b1.y = a.y;
    }
  else
    {
      a1.y = a.y;
      b1.y = b.y;
    }

 if (fabs(a.z) < fabs(b.z))
    {
      a1.z = b.z;
      b1.z = a.z;
    }
  else
    {
      a1.z = a.z;
      b1.z = b.z;
    }

  c = add(a1, b1);
  e = sub(c, a1);
  g = sub(c, e);
  h = sub(g, a1);
  f = sub(b1, h);
  d = sub(f, e);

  tmp = add(d,e);
  if((tmp.x)!=f.x)
    {
      c.x = a1.x;
      d.x = b1.x;
    }
 if((tmp.y)!=f.y)
    {
      c.y = a1.y;
      d.y = b1.y;
    }
 if((tmp.z)!=f.z)
    {
      c.z = a1.z;
      d.z = b1.z;
    }

 c.accuracy=DIGIT_NOT_COMPUTED;
   if (_cadna_cancel_tag){
    if (a.accuracy==DIGIT_NOT_COMPUTED || a.accuracy==RELIABLE_RESULT)
	a.nb_significant_digit();
    if (b.accuracy==DIGIT_NOT_COMPUTED || b.accuracy==RELIABLE_RESULT)
	b.nb_significant_digit();
    if ((a.accuracy < b.accuracy ? a.accuracy: b.accuracy) >=
	  c.nb_significant_digit()+_cadna_cancel_value)
	instability(&_cadna_cancel_count);
  }


}
inline void twoSumPriest( float_st& a,  float_st& b, double_st& c, float_st& d){
  double_st a1;
  double_st b1;
  double_st e;
  double_st f;
  double_st g;
  double_st h;
  double_st tmp;
    
  if (fabs(a.x) < fabs(b.x))
    {
      a1.x = b.x;
      b1.x = a.x;
    }
  else
    {
      a1.x = a.x;
      b1.x = b.x;
    }

 if (fabs(a.y) < fabs(b.y))
    {
      a1.y = b.y;
      b1.y = a.y;
    }
  else
    {
      a1.y = a.y;
      b1.y = b.y;
    }

 if (fabs(a.z) < fabs(b.z))
    {
      a1.z = b.z;
      b1.z = a.z;
    }
  else
    {
      a1.z = a.z;
      b1.z = b.z;
    }

  c = add(a1, b1);
  e = sub(c, a1);
  g = sub(c, e);
  h = sub(g, a1);
  f = sub(b1, h);
  d = sub(f, e);

  tmp = add(d,e);
  if((tmp.x)!=f.x)
    {
      c.x = a1.x;
      d.x = b1.x;
    }
 if((tmp.y)!=f.y)
    {
      c.y = a1.y;
      d.y = b1.y;
    }
 if((tmp.z)!=f.z)
    {
      c.z = a1.z;
      d.z = b1.z;
    }

 c.accuracy=DIGIT_NOT_COMPUTED;
   if (_cadna_cancel_tag){
    if (a.accuracy==DIGIT_NOT_COMPUTED || a.accuracy==RELIABLE_RESULT)
	a.nb_significant_digit();
    if (b.accuracy==DIGIT_NOT_COMPUTED || b.accuracy==RELIABLE_RESULT)
	b.nb_significant_digit();
    if ((a.accuracy < b.accuracy ? a.accuracy: b.accuracy) >=
	  c.nb_significant_digit()+_cadna_cancel_value)
	instability(&_cadna_cancel_count);
  }


}
inline void twoSumPriest( float_st& a,  double_st& b, float_st& c, double_st& d){
  double_st a1;
  double_st b1;
  double_st e;
  double_st f;
  double_st g;
  double_st h;
  double_st tmp;
    
  if (fabs(a.x) < fabs(b.x))
    {
      a1.x = b.x;
      b1.x = a.x;
    }
  else
    {
      a1.x = a.x;
      b1.x = b.x;
    }

 if (fabs(a.y) < fabs(b.y))
    {
      a1.y = b.y;
      b1.y = a.y;
    }
  else
    {
      a1.y = a.y;
      b1.y = b.y;
    }

 if (fabs(a.z) < fabs(b.z))
    {
      a1.z = b.z;
      b1.z = a.z;
    }
  else
    {
      a1.z = a.z;
      b1.z = b.z;
    }

  c = add(a1, b1);
  e = sub(c, a1);
  g = sub(c, e);
  h = sub(g, a1);
  f = sub(b1, h);
  d = sub(f, e);

  tmp = add(d,e);
  if((tmp.x)!=f.x)
    {
      c.x = a1.x;
      d.x = b1.x;
    }
 if((tmp.y)!=f.y)
    {
      c.y = a1.y;
      d.y = b1.y;
    }
 if((tmp.z)!=f.z)
    {
      c.z = a1.z;
      d.z = b1.z;
    }

 c.accuracy=DIGIT_NOT_COMPUTED;
   if (_cadna_cancel_tag){
    if (a.accuracy==DIGIT_NOT_COMPUTED || a.accuracy==RELIABLE_RESULT)
	a.nb_significant_digit();
    if (b.accuracy==DIGIT_NOT_COMPUTED || b.accuracy==RELIABLE_RESULT)
	b.nb_significant_digit();
    if ((a.accuracy < b.accuracy ? a.accuracy: b.accuracy) >=
	  c.nb_significant_digit()+_cadna_cancel_value)
	instability(&_cadna_cancel_count);
  }


}
inline void twoSumPriest( float_st& a,  double_st& b, float_st& c, float_st& d){
  double_st a1;
  double_st b1;
  double_st e;
  double_st f;
  double_st g;
  double_st h;
  double_st tmp;
    
  if (fabs(a.x) < fabs(b.x))
    {
      a1.x = b.x;
      b1.x = a.x;
    }
  else
    {
      a1.x = a.x;
      b1.x = b.x;
    }

 if (fabs(a.y) < fabs(b.y))
    {
      a1.y = b.y;
      b1.y = a.y;
    }
  else
    {
      a1.y = a.y;
      b1.y = b.y;
    }

 if (fabs(a.z) < fabs(b.z))
    {
      a1.z = b.z;
      b1.z = a.z;
    }
  else
    {
      a1.z = a.z;
      b1.z = b.z;
    }

  c = add(a1, b1);
  e = sub(c, a1);
  g = sub(c, e);
  h = sub(g, a1);
  f = sub(b1, h);
  d = sub(f, e);

  tmp = add(d,e);
  if((tmp.x)!=f.x)
    {
      c.x = a1.x;
      d.x = b1.x;
    }
 if((tmp.y)!=f.y)
    {
      c.y = a1.y;
      d.y = b1.y;
    }
 if((tmp.z)!=f.z)
    {
      c.z = a1.z;
      d.z = b1.z;
    }

 c.accuracy=DIGIT_NOT_COMPUTED;
   if (_cadna_cancel_tag){
    if (a.accuracy==DIGIT_NOT_COMPUTED || a.accuracy==RELIABLE_RESULT)
	a.nb_significant_digit();
    if (b.accuracy==DIGIT_NOT_COMPUTED || b.accuracy==RELIABLE_RESULT)
	b.nb_significant_digit();
    if ((a.accuracy < b.accuracy ? a.accuracy: b.accuracy) >=
	  c.nb_significant_digit()+_cadna_cancel_value)
	instability(&_cadna_cancel_count);
  }


}
inline void twoSumPriest( double_st& a,  float_st& b, float_st& c, double_st& d){
  double_st a1;
  double_st b1;
  double_st e;
  double_st f;
  double_st g;
  double_st h;
  double_st tmp;
    
  if (fabs(a.x) < fabs(b.x))
    {
      a1.x = b.x;
      b1.x = a.x;
    }
  else
    {
      a1.x = a.x;
      b1.x = b.x;
    }

 if (fabs(a.y) < fabs(b.y))
    {
      a1.y = b.y;
      b1.y = a.y;
    }
  else
    {
      a1.y = a.y;
      b1.y = b.y;
    }

 if (fabs(a.z) < fabs(b.z))
    {
      a1.z = b.z;
      b1.z = a.z;
    }
  else
    {
      a1.z = a.z;
      b1.z = b.z;
    }

  c = add(a1, b1);
  e = sub(c, a1);
  g = sub(c, e);
  h = sub(g, a1);
  f = sub(b1, h);
  d = sub(f, e);

  tmp = add(d,e);
  if((tmp.x)!=f.x)
    {
      c.x = a1.x;
      d.x = b1.x;
    }
 if((tmp.y)!=f.y)
    {
      c.y = a1.y;
      d.y = b1.y;
    }
 if((tmp.z)!=f.z)
    {
      c.z = a1.z;
      d.z = b1.z;
    }

 c.accuracy=DIGIT_NOT_COMPUTED;
   if (_cadna_cancel_tag){
    if (a.accuracy==DIGIT_NOT_COMPUTED || a.accuracy==RELIABLE_RESULT)
	a.nb_significant_digit();
    if (b.accuracy==DIGIT_NOT_COMPUTED || b.accuracy==RELIABLE_RESULT)
	b.nb_significant_digit();
    if ((a.accuracy < b.accuracy ? a.accuracy: b.accuracy) >=
	  c.nb_significant_digit()+_cadna_cancel_value)
	instability(&_cadna_cancel_count);
  }


}
inline void twoSumPriest( double_st& a,  float_st& b, float_st& c, float_st& d){
  double_st a1;
  double_st b1;
  double_st e;
  double_st f;
  double_st g;
  double_st h;
  double_st tmp;
    
  if (fabs(a.x) < fabs(b.x))
    {
      a1.x = b.x;
      b1.x = a.x;
    }
  else
    {
      a1.x = a.x;
      b1.x = b.x;
    }

 if (fabs(a.y) < fabs(b.y))
    {
      a1.y = b.y;
      b1.y = a.y;
    }
  else
    {
      a1.y = a.y;
      b1.y = b.y;
    }

 if (fabs(a.z) < fabs(b.z))
    {
      a1.z = b.z;
      b1.z = a.z;
    }
  else
    {
      a1.z = a.z;
      b1.z = b.z;
    }

  c = add(a1, b1);
  e = sub(c, a1);
  g = sub(c, e);
  h = sub(g, a1);
  f = sub(b1, h);
  d = sub(f, e);

  tmp = add(d,e);
  if((tmp.x)!=f.x)
    {
      c.x = a1.x;
      d.x = b1.x;
    }
 if((tmp.y)!=f.y)
    {
      c.y = a1.y;
      d.y = b1.y;
    }
 if((tmp.z)!=f.z)
    {
      c.z = a1.z;
      d.z = b1.z;
    }

 c.accuracy=DIGIT_NOT_COMPUTED;
   if (_cadna_cancel_tag){
    if (a.accuracy==DIGIT_NOT_COMPUTED || a.accuracy==RELIABLE_RESULT)
	a.nb_significant_digit();
    if (b.accuracy==DIGIT_NOT_COMPUTED || b.accuracy==RELIABLE_RESULT)
	b.nb_significant_digit();
    if ((a.accuracy < b.accuracy ? a.accuracy: b.accuracy) >=
	  c.nb_significant_digit()+_cadna_cancel_value)
	instability(&_cadna_cancel_count);
  }


}
inline void twoSumPriest( float_st& a,  float_st& b, float_st& c, float_st& d){
  float_st a1;
  float_st b1;
  float_st e;
  float_st f;
  float_st g;
  float_st h;
  float_st tmp;
    
  if (fabs(a.x) < fabs(b.x))
    {
      a1.x = b.x;
      b1.x = a.x;
    }
  else
    {
      a1.x = a.x;
      b1.x = b.x;
    }

 if (fabs(a.y) < fabs(b.y))
    {
      a1.y = b.y;
      b1.y = a.y;
    }
  else
    {
      a1.y = a.y;
      b1.y = b.y;
    }

 if (fabs(a.z) < fabs(b.z))
    {
      a1.z = b.z;
      b1.z = a.z;
    }
  else
    {
      a1.z = a.z;
      b1.z = b.z;
    }

  c = add(a1, b1);
  e = sub(c, a1);
  g = sub(c, e);
  h = sub(g, a1);
  f = sub(b1, h);
  d = sub(f, e);

  tmp = add(d,e);
  if((tmp.x)!=f.x)
    {
      c.x = a1.x;
      d.x = b1.x;
    }
 if((tmp.y)!=f.y)
    {
      c.y = a1.y;
      d.y = b1.y;
    }
 if((tmp.z)!=f.z)
    {
      c.z = a1.z;
      d.z = b1.z;
    }

 c.accuracy=DIGIT_NOT_COMPUTED;
   if (_cadna_cancel_tag){
    if (a.accuracy==DIGIT_NOT_COMPUTED || a.accuracy==RELIABLE_RESULT)
	a.nb_significant_digit();
    if (b.accuracy==DIGIT_NOT_COMPUTED || b.accuracy==RELIABLE_RESULT)
	b.nb_significant_digit();
    if ((a.accuracy < b.accuracy ? a.accuracy: b.accuracy) >=
	  c.nb_significant_digit()+_cadna_cancel_value)
	instability(&_cadna_cancel_count);
  }


}


////////////////////////////////////////////////////////////////////////////////


///////////////////////////////////////////////////
//****m* cadna_comp/twoProdFma
//    NAME
//      add
//    SYNOPSIS
//      
//    FUNCTION
//    Defines all the functions involving at least one
//    stochastic argument which compute the twoProdFma algorithm
//
//
//    INPUTS/OUTPUTS
//      a           - an integer, a float, a double or a stochastic number
//      b           - an integer, a float, a double or a stochastic number
//	res	    - a stochastic number: the result of "a*b"
//	err	    - a stochastic number: the computed error between "res"
//		      	and the real value of "a*b"
//      At least one argument must be of stochastic type.
//
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


#ifdef CADNA_FMA



inline void twoProdFma( double_st& a,  double_st&b, double_st& res, double_st& err){
  res = a * b;
  double_st tmp = -res;
  err = fma_no_instab(a,b,tmp);
}
inline void twoProdFma( double_st& a,  double_st&b, double_st& res, float_st& err){
  res = a * b;
  double_st tmp = -res;
  err = fma_no_instab(a,b,tmp);
}
inline void twoProdFma( float_st& a,  double_st&b, double_st& res, double_st& err){
  res = a * b;
  double_st tmp = -res;
  err = fma_no_instab(a,b,tmp);
}
inline void twoProdFma( float_st& a,  double_st&b, double_st& res, float_st& err){
  res = a * b;
  double_st tmp = -res;
  err = fma_no_instab(a,b,tmp);
}
inline void twoProdFma( double_st& a,  float_st&b, double_st& res, double_st& err){
  res = a * b;
  double_st tmp = -res;
  err = fma_no_instab(a,b,tmp);
}
inline void twoProdFma( double_st& a,  float_st&b, double_st& res, float_st& err){
  res = a * b;
  double_st tmp = -res;
  err = fma_no_instab(a,b,tmp);
}
inline void twoProdFma( double_st& a,  double_st&b, float_st& res, double_st& err){
  res = a * b;
  double_st tmp = -res;
  err = fma_no_instab(a,b,tmp);
}
inline void twoProdFma( double_st& a,  double_st&b, float_st& res, float_st& err){
  res = a * b;
  double_st tmp = -res;
  err = fma_no_instab(a,b,tmp);
}
inline void twoProdFma( float_st& a,  float_st&b, double_st& res, double_st& err){
  res = a * b;
  double_st tmp = -res;
  err = fma_no_instab(a,b,tmp);
}
inline void twoProdFma( float_st& a,  float_st&b, double_st& res, float_st& err){
  res = a * b;
  double_st tmp = -res;
  err = fma_no_instab(a,b,tmp);
}
inline void twoProdFma( float_st& a,  double_st&b, float_st& res, double_st& err){
  res = a * b;
  double_st tmp = -res;
  err = fma_no_instab(a,b,tmp);
}
inline void twoProdFma( float_st& a,  double_st&b, float_st& res, float_st& err){
  res = a * b;
  double_st tmp = -res;
  err = fma_no_instab(a,b,tmp);
}
inline void twoProdFma( double_st& a,  float_st&b, float_st& res, double_st& err){
  res = a * b;
  double_st tmp = -res;
  err = fma_no_instab(a,b,tmp);
}
inline void twoProdFma( double_st& a,  float_st&b, float_st& res, float_st& err){
  res = a * b;
  double_st tmp = -res;
  err = fma_no_instab(a,b,tmp);
}
inline void twoProdFma( float_st& a,  float_st&b, float_st& res, float_st& err){
  res = a * b;
  float_st tmp = -res;
  err = fma_no_instab(a,b,tmp);
}

#endif // CADNA_FMA

////////////////////////////////////////////////////////////////////////////////


///////////////////////////////////////////////////
//****m* cadna_comp/twoProd
//    NAME
//      add
//    SYNOPSIS
//      
//    FUNCTION
//    Defines all the functions involving at least one
//    stochastic argument which compute the twoProd algorithm
//
//
//    INPUTS/OUTPUTS
//      a           - an integer, a float, a double or a stochastic number
//      b           - an integer, a float, a double or a stochastic number
//	res	    - a stochastic number: the result of "a*b"
//	err	    - a stochastic number: the computed error between "res"
//		      	and the real value of "a*b"
//      At least one argument must be of stochastic type.
//
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



inline void twoProd( double_st& a,  double_st& b, double_st& p, double_st& e){
  double_st ashift;
  double_st bshift;
  double_st ahi;
  double_st alo;
  double_st bhi;
  double_st blo;
  double splitter=134217729.0 ;   
  p.accuracy=DIGIT_NOT_COMPUTED;
  p = (a) * (b);       
  ashift  = (a) *  splitter ; 
  ahi  = sub(ashift,sub(ashift, a)); 
  alo  = sub(a, ahi) ;                  
  bshift  = (b) *  splitter ;       
  bhi  = sub(bshift,sub(bshift,b));  
  blo  = sub(b,bhi) ;                   
  e = add(add(add(sub(mul(ahi, bhi), p),            
	      mul(ahi, blo)),              
	      mul(alo, bhi)),              
	      mul(alo, blo));

}
inline void twoProd( double_st& a,  double_st& b, double_st& p, float_st& e){
  double_st ashift;
  double_st bshift;
  double_st ahi;
  double_st alo;
  double_st bhi;
  double_st blo;
  double splitter=134217729.0 ;   
  p.accuracy=DIGIT_NOT_COMPUTED;
  p = (a) * (b);       
  ashift  = (a) *  splitter ; 
  ahi  = sub(ashift,sub(ashift, a)); 
  alo  = sub(a, ahi) ;                  
  bshift  = (b) *  splitter ;       
  bhi  = sub(bshift,sub(bshift,b));  
  blo  = sub(b,bhi) ;                   
  e = add(add(add(sub(mul(ahi, bhi), p),            
	      mul(ahi, blo)),              
	      mul(alo, bhi)),              
	      mul(alo, blo));

}
inline void twoProd( float_st& a,  double_st& b, double_st& p, double_st& e){
  double_st ashift;
  double_st bshift;
  double_st ahi;
  double_st alo;
  double_st bhi;
  double_st blo;
  double splitter=134217729.0 ;   
  p.accuracy=DIGIT_NOT_COMPUTED;
  p = (a) * (b);       
  ashift  = (a) *  splitter ; 
  ahi  = sub(ashift,sub(ashift, a)); 
  alo  = sub(a, ahi) ;                  
  bshift  = (b) *  splitter ;       
  bhi  = sub(bshift,sub(bshift,b));  
  blo  = sub(b,bhi) ;                   
  e = add(add(add(sub(mul(ahi, bhi), p),            
	      mul(ahi, blo)),              
	      mul(alo, bhi)),              
	      mul(alo, blo));

}
inline void twoProd( float_st& a,  double_st& b, double_st& p, float_st& e){
  double_st ashift;
  double_st bshift;
  double_st ahi;
  double_st alo;
  double_st bhi;
  double_st blo;
  double splitter=134217729.0 ;   
  p.accuracy=DIGIT_NOT_COMPUTED;
  p = (a) * (b);       
  ashift  = (a) *  splitter ; 
  ahi  = sub(ashift,sub(ashift, a)); 
  alo  = sub(a, ahi) ;                  
  bshift  = (b) *  splitter ;       
  bhi  = sub(bshift,sub(bshift,b));  
  blo  = sub(b,bhi) ;                   
  e = add(add(add(sub(mul(ahi, bhi), p),            
	      mul(ahi, blo)),              
	      mul(alo, bhi)),              
	      mul(alo, blo));

}
inline void twoProd( double_st& a,  float_st& b, double_st& p, double_st& e){
  double_st ashift;
  double_st bshift;
  double_st ahi;
  double_st alo;
  double_st bhi;
  double_st blo;
  double splitter=134217729.0 ;   
  p.accuracy=DIGIT_NOT_COMPUTED;
  p = (a) * (b);       
  ashift  = (a) *  splitter ; 
  ahi  = sub(ashift,sub(ashift, a)); 
  alo  = sub(a, ahi) ;                  
  bshift  = (b) *  splitter ;       
  bhi  = sub(bshift,sub(bshift,b));  
  blo  = sub(b,bhi) ;                   
  e = add(add(add(sub(mul(ahi, bhi), p),            
	      mul(ahi, blo)),              
	      mul(alo, bhi)),              
	      mul(alo, blo));

}
inline void twoProd( double_st& a,  float_st& b, double_st& p, float_st& e){
  double_st ashift;
  double_st bshift;
  double_st ahi;
  double_st alo;
  double_st bhi;
  double_st blo;
  double splitter=134217729.0 ;   
  p.accuracy=DIGIT_NOT_COMPUTED;
  p = (a) * (b);       
  ashift  = (a) *  splitter ; 
  ahi  = sub(ashift,sub(ashift, a)); 
  alo  = sub(a, ahi) ;                  
  bshift  = (b) *  splitter ;       
  bhi  = sub(bshift,sub(bshift,b));  
  blo  = sub(b,bhi) ;                   
  e = add(add(add(sub(mul(ahi, bhi), p),            
	      mul(ahi, blo)),              
	      mul(alo, bhi)),              
	      mul(alo, blo));

}
inline void twoProd( double_st& a,  double_st& b, float_st& p, double_st& e){
  double_st ashift;
  double_st bshift;
  double_st ahi;
  double_st alo;
  double_st bhi;
  double_st blo;
  double splitter=134217729.0 ;   
  p.accuracy=DIGIT_NOT_COMPUTED;
  p = (a) * (b);       
  ashift  = (a) *  splitter ; 
  ahi  = sub(ashift,sub(ashift, a)); 
  alo  = sub(a, ahi) ;                  
  bshift  = (b) *  splitter ;       
  bhi  = sub(bshift,sub(bshift,b));  
  blo  = sub(b,bhi) ;                   
  e = add(add(add(sub(mul(ahi, bhi), p),            
	      mul(ahi, blo)),              
	      mul(alo, bhi)),              
	      mul(alo, blo));

}
inline void twoProd( double_st& a,  double_st& b, float_st& p, float_st& e){
  double_st ashift;
  double_st bshift;
  double_st ahi;
  double_st alo;
  double_st bhi;
  double_st blo;
  double splitter=134217729.0 ;   
  p.accuracy=DIGIT_NOT_COMPUTED;
  p = (a) * (b);       
  ashift  = (a) *  splitter ; 
  ahi  = sub(ashift,sub(ashift, a)); 
  alo  = sub(a, ahi) ;                  
  bshift  = (b) *  splitter ;       
  bhi  = sub(bshift,sub(bshift,b));  
  blo  = sub(b,bhi) ;                   
  e = add(add(add(sub(mul(ahi, bhi), p),            
	      mul(ahi, blo)),              
	      mul(alo, bhi)),              
	      mul(alo, blo));

}
inline void twoProd( float_st& a,  float_st& b, double_st& p, double_st& e){
  double_st ashift;
  double_st bshift;
  double_st ahi;
  double_st alo;
  double_st bhi;
  double_st blo;
  double splitter=134217729.0 ;   
  p.accuracy=DIGIT_NOT_COMPUTED;
  p = (a) * (b);       
  ashift  = (a) *  splitter ; 
  ahi  = sub(ashift,sub(ashift, a)); 
  alo  = sub(a, ahi) ;                  
  bshift  = (b) *  splitter ;       
  bhi  = sub(bshift,sub(bshift,b));  
  blo  = sub(b,bhi) ;                   
  e = add(add(add(sub(mul(ahi, bhi), p),            
	      mul(ahi, blo)),              
	      mul(alo, bhi)),              
	      mul(alo, blo));

}
inline void twoProd( float_st& a,  float_st& b, double_st& p, float_st& e){
  double_st ashift;
  double_st bshift;
  double_st ahi;
  double_st alo;
  double_st bhi;
  double_st blo;
  double splitter=134217729.0 ;   
  p.accuracy=DIGIT_NOT_COMPUTED;
  p = (a) * (b);       
  ashift  = (a) *  splitter ; 
  ahi  = sub(ashift,sub(ashift, a)); 
  alo  = sub(a, ahi) ;                  
  bshift  = (b) *  splitter ;       
  bhi  = sub(bshift,sub(bshift,b));  
  blo  = sub(b,bhi) ;                   
  e = add(add(add(sub(mul(ahi, bhi), p),            
	      mul(ahi, blo)),              
	      mul(alo, bhi)),              
	      mul(alo, blo));

}
inline void twoProd( float_st& a,  double_st& b, float_st& p, double_st& e){
  double_st ashift;
  double_st bshift;
  double_st ahi;
  double_st alo;
  double_st bhi;
  double_st blo;
  double splitter=134217729.0 ;   
  p.accuracy=DIGIT_NOT_COMPUTED;
  p = (a) * (b);       
  ashift  = (a) *  splitter ; 
  ahi  = sub(ashift,sub(ashift, a)); 
  alo  = sub(a, ahi) ;                  
  bshift  = (b) *  splitter ;       
  bhi  = sub(bshift,sub(bshift,b));  
  blo  = sub(b,bhi) ;                   
  e = add(add(add(sub(mul(ahi, bhi), p),            
	      mul(ahi, blo)),              
	      mul(alo, bhi)),              
	      mul(alo, blo));

}
inline void twoProd( float_st& a,  double_st& b, float_st& p, float_st& e){
  double_st ashift;
  double_st bshift;
  double_st ahi;
  double_st alo;
  double_st bhi;
  double_st blo;
  double splitter=134217729.0 ;   
  p.accuracy=DIGIT_NOT_COMPUTED;
  p = (a) * (b);       
  ashift  = (a) *  splitter ; 
  ahi  = sub(ashift,sub(ashift, a)); 
  alo  = sub(a, ahi) ;                  
  bshift  = (b) *  splitter ;       
  bhi  = sub(bshift,sub(bshift,b));  
  blo  = sub(b,bhi) ;                   
  e = add(add(add(sub(mul(ahi, bhi), p),            
	      mul(ahi, blo)),              
	      mul(alo, bhi)),              
	      mul(alo, blo));

}
inline void twoProd( double_st& a,  float_st& b, float_st& p, double_st& e){
  double_st ashift;
  double_st bshift;
  double_st ahi;
  double_st alo;
  double_st bhi;
  double_st blo;
  double splitter=134217729.0 ;   
  p.accuracy=DIGIT_NOT_COMPUTED;
  p = (a) * (b);       
  ashift  = (a) *  splitter ; 
  ahi  = sub(ashift,sub(ashift, a)); 
  alo  = sub(a, ahi) ;                  
  bshift  = (b) *  splitter ;       
  bhi  = sub(bshift,sub(bshift,b));  
  blo  = sub(b,bhi) ;                   
  e = add(add(add(sub(mul(ahi, bhi), p),            
	      mul(ahi, blo)),              
	      mul(alo, bhi)),              
	      mul(alo, blo));

}
inline void twoProd( double_st& a,  float_st& b, float_st& p, float_st& e){
  double_st ashift;
  double_st bshift;
  double_st ahi;
  double_st alo;
  double_st bhi;
  double_st blo;
  double splitter=134217729.0 ;   
  p.accuracy=DIGIT_NOT_COMPUTED;
  p = (a) * (b);       
  ashift  = (a) *  splitter ; 
  ahi  = sub(ashift,sub(ashift, a)); 
  alo  = sub(a, ahi) ;                  
  bshift  = (b) *  splitter ;       
  bhi  = sub(bshift,sub(bshift,b));  
  blo  = sub(b,bhi) ;                   
  e = add(add(add(sub(mul(ahi, bhi), p),            
	      mul(ahi, blo)),              
	      mul(alo, bhi)),              
	      mul(alo, blo));

}
inline void twoProd( float_st& a,  float_st& b, float_st& p, float_st& e){
  float_st ashift;
  float_st bshift;
  float_st ahi;
  float_st alo;
  float_st bhi;
  float_st blo;
  double splitter=134217729.0 ;   
  p.accuracy=DIGIT_NOT_COMPUTED;
  p = (a) * (b);       
  ashift  = (a) *  splitter ; 
  ahi  = sub(ashift,sub(ashift, a)); 
  alo  = sub(a, ahi) ;                  
  bshift  = (b) *  splitter ;       
  bhi  = sub(bshift,sub(bshift,b));  
  blo  = sub(b,bhi) ;                   
  e = add(add(add(sub(mul(ahi, bhi), p),            
	      mul(ahi, blo)),              
	      mul(alo, bhi)),              
	      mul(alo, blo));

}


////////////////////////////////////////////////////////////////////////////////

