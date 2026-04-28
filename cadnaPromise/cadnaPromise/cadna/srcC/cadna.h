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


#ifndef __CADNA__
#define __CADNA__

#ifdef _OPENMP
#include <omp.h>
#endif

#include <iostream>
#include <cmath>
#include <complex>
#include <cstdlib>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include "cadna_plugin.h"
#ifdef CADNA_HALF_EMULATION
#include <half.hpp>
#include <cadna_half.hpp>
using half_float::half;
#endif
#include <fenv.h>

#ifdef CADNA_HALF_NATIVE
typedef __fp16 half;
#endif

#ifdef __CADNA_DEBUG__

#define __CADNA_ALWAYS_INLINE__
#define __CADNA_INLINE__

#else

#define __CADNA_ALWAYS_INLINE__ __attribute__((always_inline))
#define __CADNA_INLINE__ inline
#endif

#define CADNA_DIV       0x1
#define CADNA_MUL       0x2
#define CADNA_POWER     0x4
#define CADNA_MATH      0x8
#define CADNA_BRANCHING 0x10
#define CADNA_INTRINSIC 0x20
#define CADNA_CANCEL    0x40
#ifdef CADNA_HALF_EMULATION
#define CADNA_HALF_OVERFLOW    0x80
#define CADNA_HALF_UNDERFLOW    0x100
#endif
#ifdef CADNA_HALF_NATIVE
#define CADNA_HALF_OVERFLOW    0x80
#define CADNA_HALF_UNDERFLOW    0x100
#endif
#define CADNA_ALL       0xFFFFFFFF



class float_st;
class half_st;



class double_st 
{  
  friend class float_st;

#ifdef CADNA_HALF_EMULATION
  friend class half_st;
#endif

#ifdef CADNA_HALF_NATIVE
  friend class half_st;
#endif
  
  typedef double TYPEBASE;
  
public:
  double x,y,z;
  mutable int accuracy;
  int pad;
  inline double_st(void) :x(0),y(0),z(0),accuracy(15) {};
  inline double_st(double a, double b, double c):x(a),y(b),z(c),accuracy(-1) {};
  inline double_st(const double &a): x(a),y(a),z(a),accuracy(15) {};

  inline double_st (const float_st&);
#ifdef CADNA_HALF_EMULATION
  inline double_st (const half_st&);
#endif

#ifdef CADNA_HALF_NATIVE
  inline double_st (const half_st&);
#endif
  double getx() {return(this->x);}
  double gety() {return(this->y);}
  double getz() {return(this->z);}
  int getaccuracy() {return(this->accuracy);}

  //complex
//1)mulplication
  friend std::complex<double_st> operator*(double const&,std::complex<double_st> const&);
  friend std::complex<double_st> operator*(std::complex<double_st> const&,double const&);
  friend std::complex<float_st> operator*( float const&,std::complex<float_st> const&);
  friend std::complex<float_st> operator*( std::complex<float_st> const&,float const&);
//2)division
  friend std::complex<double_st> operator/(double const&,std::complex<double_st> const&);
  friend std::complex<double_st> operator/(std::complex<double_st> const&,double const&);
  friend std::complex<float_st> operator/( float const&,std::complex<float_st> const&);
  friend std::complex<float_st> operator/( std::complex<float_st> const&,float const&);
//3)addition
  friend std::complex<double_st> operator+(double const&,std::complex<double_st> const&);
  friend std::complex<double_st> operator+(std::complex<double_st> const&,double const&);
  friend std::complex<float_st> operator+( float const&,std::complex<float_st> const&);
  friend std::complex<float_st> operator+( std::complex<float_st> const&,float const&);
//4)SUBTRACTION
  friend std::complex<double_st> operator-(double const&,std::complex<double_st> const&);
  friend std::complex<double_st> operator-(std::complex<double_st> const&,double const&);
  friend std::complex<float_st> operator-( float const&,std::complex<float_st> const&);
  friend std::complex<float_st> operator-( std::complex<float_st> const&,float const&);

  // ASSIGNMENT
  //inline double_st& operator=(const double_st&)__attribute__((always_inline));
  inline double_st& operator=(const float_st&)__attribute__((always_inline));
  inline double_st& operator=(const double&)__attribute__((always_inline));
  inline double_st& operator=(const float&)__attribute__((always_inline));
  inline double_st& operator=(const unsigned long long&)__attribute__((always_inline));
  inline double_st& operator=(const long long&)__attribute__((always_inline));
  inline double_st& operator=(const unsigned long&)__attribute__((always_inline));
  inline double_st& operator=(const long&)__attribute__((always_inline));
  inline double_st& operator=(const unsigned int&)__attribute__((always_inline));
  inline double_st& operator=(const int&)__attribute__((always_inline));
  inline double_st& operator=(const unsigned short&)__attribute__((always_inline));
  inline double_st& operator=(const short&)__attribute__((always_inline));
  inline double_st& operator=(const unsigned char&)__attribute__((always_inline));
  inline double_st& operator=(const char&)__attribute__((always_inline));
#ifdef CADNA_HALF_EMULATION
  inline double_st& operator=(const half&)__attribute__((always_inline));
  inline double_st& operator=(const half_st&)__attribute__((always_inline));
#endif 

#ifdef CADNA_HALF_NATIVE
  inline double_st& operator=(const half&)__attribute__((always_inline));
  inline double_st& operator=(const half_st&)__attribute__((always_inline));
#endif 

  // ADDITION
  inline double_st operator++()__attribute__((always_inline));
  inline double_st operator++(int)__attribute__((always_inline));
  inline double_st operator+() const __attribute__((always_inline));

#ifdef CADNA_HALF_EMULATION
  inline double_st& operator+=(const half_st& a)__attribute__((always_inline));
#endif 

#ifdef CADNA_HALF_NATIVE
  inline double_st& operator+=(const half_st& a)__attribute__((always_inline));
#endif
  inline double_st& operator+=(const double_st&)__attribute__((always_inline));
  inline double_st& operator+=(const float_st&)__attribute__((always_inline));
  inline double_st& operator+=(const double&)__attribute__((always_inline));
  inline double_st& operator+=(const float&)__attribute__((always_inline));
  inline double_st& operator+=(const unsigned long long&)__attribute__((always_inline));
  inline double_st& operator+=(const long long&)__attribute__((always_inline));
  inline double_st& operator+=(const unsigned long&)__attribute__((always_inline));
  inline double_st& operator+=(const long&)__attribute__((always_inline));
  inline double_st& operator+=(const unsigned int&)__attribute__((always_inline));
  inline double_st& operator+=(const int&)__attribute__((always_inline));
  inline double_st& operator+=(const unsigned short&)__attribute__((always_inline));
  inline double_st& operator+=(const short&)__attribute__((always_inline));
  inline double_st& operator+=(const unsigned char&)__attribute__((always_inline));
  inline double_st& operator+=(const char&)__attribute__((always_inline));

#ifdef CADNA_HALF_EMULATION
  inline double_st& operator+=(const half& a)__attribute__((always_inline));
#endif 

#ifdef CADNA_HALF_NATIVE
  inline double_st& operator+=(const half& a)__attribute__((always_inline));
#endif 

  inline friend double_st operator+(const double_st&, const double_st&)__attribute__((always_inline)); 
  inline friend double_st operator+(const double_st&, const float_st&)__attribute__((always_inline));
  inline friend double_st operator+(const float_st&,  const double_st&)__attribute__((always_inline));
  
  inline friend double_st operator+(const double_st&, const double&)__attribute__((always_inline));
  inline friend double_st operator+(const double_st&, const float&)__attribute__((always_inline));
  inline friend double_st operator+(const double_st&, const unsigned long long&)__attribute__((always_inline));
  inline friend double_st operator+(const double_st&, const long long&)__attribute__((always_inline));
  inline friend double_st operator+(const double_st&, const unsigned long&)__attribute__((always_inline));
  inline friend double_st operator+(const double_st&, const long&)__attribute__((always_inline));
  inline friend double_st operator+(const double_st&, const unsigned int&)__attribute__((always_inline));
  inline friend double_st operator+(const double_st&, const int&)__attribute__((always_inline));
  inline friend double_st operator+(const double_st&, const unsigned short&)__attribute__((always_inline));
  inline friend double_st operator+(const double_st&, const short&)__attribute__((always_inline));
  inline friend double_st operator+(const double_st&, const unsigned char&)__attribute__((always_inline));
  inline friend double_st operator+(const double_st&, const char&)__attribute__((always_inline));

  inline friend double_st operator+(const double&, const double_st&)__attribute__((always_inline));
  inline friend double_st operator+(const float&, const double_st&)__attribute__((always_inline));
  inline friend double_st operator+(const unsigned long long&, const double_st&)__attribute__((always_inline));
  inline friend double_st operator+(const long long&, const double_st&)__attribute__((always_inline));
  inline friend double_st operator+(const unsigned long&, const double_st&)__attribute__((always_inline));
  inline friend double_st operator+(const long&, const double_st&)__attribute__((always_inline));
  inline friend double_st operator+(const unsigned int&, const double_st&)__attribute__((always_inline));
  inline friend double_st operator+(const int&, const double_st&)__attribute__((always_inline));
  inline friend double_st operator+(const unsigned short&, const double_st&)__attribute__((always_inline));
  inline friend double_st operator+(const short&, const double_st&)__attribute__((always_inline));
  inline friend double_st operator+(const unsigned char&, const double_st&)__attribute__((always_inline));
  inline friend double_st operator+(const char&, const double_st&)__attribute__((always_inline));

#ifdef CADNA_HALF_EMULATION
  inline friend double_st operator+(const half_st&, const double_st&)__attribute__((always_inline));
  inline friend double_st operator+(const double_st&, const half_st&)__attribute__((always_inline));
  inline friend double_st operator+(const half&, const double_st&)__attribute__((always_inline));
  inline friend double_st operator+(const double_st&, const half&)__attribute__((always_inline));
#endif 
#ifdef CADNA_HALF_NATIVE
  inline friend double_st operator+(const half_st&, const double_st&)__attribute__((always_inline));
  inline friend double_st operator+(const double_st&, const half_st&)__attribute__((always_inline));
  inline friend double_st operator+(const half&, const double_st&)__attribute__((always_inline));
  inline friend double_st operator+(const double_st&, const half&)__attribute__((always_inline));
#endif
  
      
  // SUBTRACTION
  inline double_st operator--()__attribute__((always_inline));
  inline double_st operator--(int)__attribute__((always_inline));
  inline double_st operator-() const __attribute__((always_inline));

  inline double_st& operator-=(const double_st&)__attribute__((always_inline));
  inline double_st& operator-=(const float_st&)__attribute__((always_inline));
  inline double_st& operator-=(const double&)__attribute__((always_inline));
  inline double_st& operator-=(const float&)__attribute__((always_inline));
  inline double_st& operator-=(const unsigned long long&)__attribute__((always_inline));
  inline double_st& operator-=(const long long&)__attribute__((always_inline));
  inline double_st& operator-=(const unsigned long&)__attribute__((always_inline));
  inline double_st& operator-=(const long&)__attribute__((always_inline));
  inline double_st& operator-=(const unsigned int&)__attribute__((always_inline));
  inline double_st& operator-=(const int&)__attribute__((always_inline));
  inline double_st& operator-=(const unsigned short&)__attribute__((always_inline));
  inline double_st& operator-=(const short&)__attribute__((always_inline));
  inline double_st& operator-=(const unsigned char&)__attribute__((always_inline));
  inline double_st& operator-=(const char&)__attribute__((always_inline));


#ifdef CADNA_HALF_EMULATION
  inline double_st& operator-=(const half&)__attribute__((always_inline));
  inline double_st& operator-=(const half_st&)__attribute__((always_inline));
#endif 

#ifdef CADNA_HALF_NATIVE
  inline double_st& operator-=(const half&)__attribute__((always_inline));
  inline double_st& operator-=(const half_st&)__attribute__((always_inline));
#endif

  inline friend double_st operator-(const double_st&, const double_st&)__attribute__((always_inline)); 
  inline friend double_st operator-(const double_st&, const float_st&)__attribute__((always_inline));
  inline friend double_st operator-(const float_st&, const double_st&)__attribute__((always_inline));
  inline friend double_st operator-(const double_st&, const double&)__attribute__((always_inline));
  inline friend double_st operator-(const double_st&, const float&)__attribute__((always_inline));
  inline friend double_st operator-(const double_st&, const unsigned long long&)__attribute__((always_inline));
  inline friend double_st operator-(const double_st&, const long long&)__attribute__((always_inline));
  inline friend double_st operator-(const double_st&, const unsigned long&)__attribute__((always_inline));
  inline friend double_st operator-(const double_st&, const long&)__attribute__((always_inline));
  inline friend double_st operator-(const double_st&, const unsigned int&)__attribute__((always_inline));
  inline friend double_st operator-(const double_st&, const int&)__attribute__((always_inline));
  inline friend double_st operator-(const double_st&, const unsigned short&)__attribute__((always_inline));
  inline friend double_st operator-(const double_st&, const short&)__attribute__((always_inline));
  inline friend double_st operator-(const double_st&, const unsigned char&)__attribute__((always_inline));
  inline friend double_st operator-(const double_st&, const char&)__attribute__((always_inline));
#ifdef CADNA_HALF_EMULATION
  inline friend double_st operator-(const double_st&, const half&)__attribute__((always_inline)); 
  inline friend double_st operator-(const double_st&, const half_st&)__attribute__((always_inline)); 
  inline friend double_st operator-(const half_st&, const double_st&)__attribute__((always_inline)); 
#endif 

#ifdef CADNA_HALF_NATIVE
  inline friend double_st operator-(const double_st&, const half&)__attribute__((always_inline)); 
  inline friend double_st operator-(const double_st&, const half_st&)__attribute__((always_inline)); 
  inline friend double_st operator-(const half_st&, const double_st&)__attribute__((always_inline)); 
#endif 

  inline friend double_st operator-(const double&, const double_st&)__attribute__((always_inline));
  inline friend double_st operator-(const float&, const double_st&)__attribute__((always_inline));
  inline friend double_st operator-(const unsigned long long&, const double_st&)__attribute__((always_inline));
  inline friend double_st operator-(const long long&, const double_st&)__attribute__((always_inline));
  inline friend double_st operator-(const unsigned long&, const double_st&)__attribute__((always_inline));
  inline friend double_st operator-(const long&, const double_st&)__attribute__((always_inline));
  inline friend double_st operator-(const unsigned int&, const double_st&)__attribute__((always_inline));
  inline friend double_st operator-(const int&, const double_st&)__attribute__((always_inline));
  inline friend double_st operator-(const unsigned short&, const double_st&)__attribute__((always_inline));
  inline friend double_st operator-(const short&, const double_st&)__attribute__((always_inline));
  inline friend double_st operator-(const unsigned char&, const double_st&)__attribute__((always_inline));
  inline friend double_st operator-(const char&, const double_st&)__attribute__((always_inline));
#ifdef CADNA_HALF_EMULATION
  inline friend double_st operator-(const half&, const double_st&)__attribute__((always_inline));
#endif 

#ifdef CADNA_HALF_NATIVE
  inline friend double_st operator-(const half&, const double_st&)__attribute__((always_inline));
#endif

  // MULTIPLICATION

#ifdef CADNA_HALF_EMULATION
  inline double_st& operator*=(const half_st&)__attribute__((always_inline));
#endif 

#ifdef CADNA_HALF_NATIVE
  inline double_st& operator*=(const half_st&)__attribute__((always_inline));
#endif

  inline double_st& operator*=(const double_st&)__attribute__((always_inline));
  inline double_st& operator*=(const float_st&)__attribute__((always_inline));
  inline double_st& operator*=(const double&)__attribute__((always_inline));
  inline double_st& operator*=(const float&)__attribute__((always_inline));
  inline double_st& operator*=(const unsigned long long&)__attribute__((always_inline));
  inline double_st& operator*=(const long long&)__attribute__((always_inline));
  inline double_st& operator*=(const unsigned long&)__attribute__((always_inline));
  inline double_st& operator*=(const long&)__attribute__((always_inline));
  inline double_st& operator*=(const unsigned int&)__attribute__((always_inline));
  inline double_st& operator*=(const int&)__attribute__((always_inline));
  inline double_st& operator*=(const unsigned short&)__attribute__((always_inline));
  inline double_st& operator*=(const short&)__attribute__((always_inline));
  inline double_st& operator*=(const unsigned char&)__attribute__((always_inline));
  inline double_st& operator*=(const char&)__attribute__((always_inline));
#ifdef CADNA_HALF_EMULATION
  inline double_st& operator*=(const half&)__attribute__((always_inline));
  inline friend double_st operator*(const half_st&, const double_st&)__attribute__((always_inline)); 
  inline friend double_st operator*(const double_st&, const half_st&)__attribute__((always_inline));
#endif  

#ifdef CADNA_HALF_NATIVE
  inline double_st& operator*=(const half&)__attribute__((always_inline));
  inline friend double_st operator*(const half_st&, const double_st&)__attribute__((always_inline)); 
  inline friend double_st operator*(const double_st&, const half_st&)__attribute__((always_inline));
#endif



  inline friend double_st operator*(const double_st&, const double_st&)__attribute__((always_inline)); 
  inline friend double_st operator*(const double_st&, const float_st&)__attribute__((always_inline));
  inline friend double_st operator*(const float_st&, const double_st&)__attribute__((always_inline));

  inline friend double_st operator*(const double_st&, const double&)__attribute__((always_inline));
  inline friend double_st operator*(const double_st&, const float&)__attribute__((always_inline));
  inline friend double_st operator*(const double_st&, const unsigned long long&)__attribute__((always_inline));
  inline friend double_st operator*(const double_st&, const long long&)__attribute__((always_inline));
  inline friend double_st operator*(const double_st&, const unsigned long&)__attribute__((always_inline));
  inline friend double_st operator*(const double_st&, const long&)__attribute__((always_inline));
  inline friend double_st operator*(const double_st&, const unsigned int&)__attribute__((always_inline));
  inline friend double_st operator*(const double_st&, const int&)__attribute__((always_inline));
  inline friend double_st operator*(const double_st&, const unsigned short&)__attribute__((always_inline));
  inline friend double_st operator*(const double_st&, const short&)__attribute__((always_inline));
  inline friend double_st operator*(const double_st&, const unsigned char&)__attribute__((always_inline));
  inline friend double_st operator*(const double_st&, const char&)__attribute__((always_inline));
#ifdef CADNA_HALF_EMULATION
  inline friend double_st operator*(const double_st&, const half&)__attribute__((always_inline));
#endif 

#ifdef CADNA_HALF_NATIVE
  inline friend double_st operator*(const double_st&, const half&)__attribute__((always_inline));
#endif 

  inline friend double_st operator*(const double&, const double_st&)__attribute__((always_inline));
  inline friend double_st operator*(const float&, const double_st&)__attribute__((always_inline));
  inline friend double_st operator*(const unsigned long long&, const double_st&)__attribute__((always_inline));
  inline friend double_st operator*(const long long&, const double_st&)__attribute__((always_inline));
  inline friend double_st operator*(const unsigned long&, const double_st&)__attribute__((always_inline));
  inline friend double_st operator*(const long&, const double_st&)__attribute__((always_inline));
  inline friend double_st operator*(const unsigned int&, const double_st&)__attribute__((always_inline));
  inline friend double_st operator*(const int&, const double_st&)__attribute__((always_inline));
  inline friend double_st operator*(const unsigned short&, const double_st&)__attribute__((always_inline));
  inline friend double_st operator*(const short&, const double_st&)__attribute__((always_inline));
  inline friend double_st operator*(const unsigned char&, const double_st&)__attribute__((always_inline));
  inline friend double_st operator*(const char&, const double_st&)__attribute__((always_inline));
#ifdef CADNA_HALF_EMULATION
  inline friend double_st operator*(const half&, const double_st&)__attribute__((always_inline));
#endif 

#ifdef CADNA_HALF_NATIVE
  inline friend double_st operator*(const half&, const double_st&)__attribute__((always_inline));
#endif 

  // DIVISION

  inline double_st& operator/=(const double_st&)__attribute__((always_inline));
  inline double_st& operator/=(const float_st&)__attribute__((always_inline));
  inline double_st& operator/=(const double&)__attribute__((always_inline));
  inline double_st& operator/=(const float&)__attribute__((always_inline));
  inline double_st& operator/=(const unsigned long long&)__attribute__((always_inline));
  inline double_st& operator/=(const long long&)__attribute__((always_inline));
  inline double_st& operator/=(const unsigned long&)__attribute__((always_inline));
  inline double_st& operator/=(const long&)__attribute__((always_inline));
  inline double_st& operator/=(const unsigned int&)__attribute__((always_inline));
  inline double_st& operator/=(const int&)__attribute__((always_inline));
  inline double_st& operator/=(const unsigned short&)__attribute__((always_inline));
  inline double_st& operator/=(const short&)__attribute__((always_inline));
  inline double_st& operator/=(const unsigned char&)__attribute__((always_inline));
  inline double_st& operator/=(const char&)__attribute__((always_inline));
#ifdef CADNA_HALF_EMULATION
  inline double_st& operator/=(const half&)__attribute__((always_inline));
  inline double_st& operator/=(const half_st&)__attribute__((always_inline));
#endif 

#ifdef CADNA_HALF_NATIVE
  inline double_st& operator/=(const half&)__attribute__((always_inline));
  inline double_st& operator/=(const half_st&)__attribute__((always_inline));
#endif

  inline friend double_st operator/(const double_st&, const double_st&)__attribute__((always_inline)); 
  inline friend double_st operator/(const double_st&, const float_st&)__attribute__((always_inline));
  inline friend double_st operator/(const float_st&, const double_st&)__attribute__((always_inline));
  inline friend double_st operator/(const double_st&, const double&)__attribute__((always_inline));
  inline friend double_st operator/(const double_st&, const float&)__attribute__((always_inline));
  inline friend double_st operator/(const double_st&, const unsigned long long&)__attribute__((always_inline));
  inline friend double_st operator/(const double_st&, const long long&)__attribute__((always_inline));
  inline friend double_st operator/(const double_st&, const unsigned long&)__attribute__((always_inline));
  inline friend double_st operator/(const double_st&, const long&)__attribute__((always_inline));
  inline friend double_st operator/(const double_st&, const unsigned int&)__attribute__((always_inline));
  inline friend double_st operator/(const double_st&, const int&)__attribute__((always_inline));
  inline friend double_st operator/(const double_st&, const unsigned short&)__attribute__((always_inline));
  inline friend double_st operator/(const double_st&, const short&)__attribute__((always_inline));
  inline friend double_st operator/(const double_st&, const unsigned char&)__attribute__((always_inline));
  inline friend double_st operator/(const double_st&, const char&)__attribute__((always_inline));
#ifdef CADNA_HALF_EMULATION
  inline friend double_st operator/(const double_st&, const half&)__attribute__((always_inline));
  inline friend double_st operator/(const double_st&, const half_st&)__attribute__((always_inline));
  inline friend double_st operator/(const half_st&, const double_st&)__attribute__((always_inline));
#endif 

#ifdef CADNA_HALF_NATIVE
  inline friend double_st operator/(const double_st&, const half&)__attribute__((always_inline));
  inline friend double_st operator/(const double_st&, const half_st&)__attribute__((always_inline));
  inline friend double_st operator/(const half_st&, const double_st&)__attribute__((always_inline));
#endif
  
  inline friend double_st operator/(const double&, const double_st&)__attribute__((always_inline));
  inline friend double_st operator/(const float&, const double_st&)__attribute__((always_inline));
  inline friend double_st operator/(const unsigned long long&, const double_st&)__attribute__((always_inline));
  inline friend double_st operator/(const long long&, const double_st&)__attribute__((always_inline));
  inline friend double_st operator/(const unsigned long&, const double_st&)__attribute__((always_inline));
  inline friend double_st operator/(const long&, const double_st&)__attribute__((always_inline));
  inline friend double_st operator/(const unsigned int&, const double_st&)__attribute__((always_inline));
  inline friend double_st operator/(const int&, const double_st&)__attribute__((always_inline));
  inline friend double_st operator/(const unsigned short&, const double_st&)__attribute__((always_inline));
  inline friend double_st operator/(const short&, const double_st&)__attribute__((always_inline));
  inline friend double_st operator/(const unsigned char&, const double_st&)__attribute__((always_inline));
  inline friend double_st operator/(const char&, const double_st&)__attribute__((always_inline));
#ifdef CADNA_HALF_EMULATION
  inline friend double_st operator/(const half&, const double_st&)__attribute__((always_inline));
#endif

#ifdef CADNA_HALF_NATIVE
  inline friend double_st operator/(const half&, const double_st&)__attribute__((always_inline));
#endif

  // RELATIONAL OPERATORS

  inline friend int operator==(const double_st&, const double_st&)__attribute__((always_inline));
  inline friend int operator==(const double_st&, const float_st&)__attribute__((always_inline));
  inline friend int operator==(const float_st&, const double_st&)__attribute__((always_inline));

  inline friend int operator==(const double_st&, const double&)__attribute__((always_inline));
  inline friend int operator==(const double_st&, const float&)__attribute__((always_inline));
  inline friend int operator==(const double_st&, const unsigned long long&)__attribute__((always_inline));
  inline friend int operator==(const double_st&, const long long&)__attribute__((always_inline));
  inline friend int operator==(const double_st&, const unsigned long&)__attribute__((always_inline));
  inline friend int operator==(const double_st&, const long&)__attribute__((always_inline));
  inline friend int operator==(const double_st&, const unsigned int&)__attribute__((always_inline));
  inline friend int operator==(const double_st&, const int&)__attribute__((always_inline));
  inline friend int operator==(const double_st&, const unsigned short&)__attribute__((always_inline));
  inline friend int operator==(const double_st&, const short&)__attribute__((always_inline));
  inline friend int operator==(const double_st&, const unsigned char&)__attribute__((always_inline));
  inline friend int operator==(const double_st&, const char&)__attribute__((always_inline));
#ifdef CDANA_HALF_EMULATION
  inline friend int operator==(const double_st&, const half&)__attribute__((always_inline));
#endif

#ifdef CDANA_HALF_NATIVE
  inline friend int operator==(const double_st&, const half&)__attribute__((always_inline));
#endif

  inline friend int operator==(const double&, const double_st&)__attribute__((always_inline));
  inline friend int operator==(const float&, const double_st&)__attribute__((always_inline));
  inline friend int operator==(const unsigned long long&, const double_st&)__attribute__((always_inline));
  inline friend int operator==(const long long&, const double_st&)__attribute__((always_inline));
  inline friend int operator==(const unsigned long&, const double_st&)__attribute__((always_inline));
  inline friend int operator==(const long&, const double_st&)__attribute__((always_inline));
  inline friend int operator==(const unsigned int&, const double_st&)__attribute__((always_inline));
  inline friend int operator==(const int&, const double_st&)__attribute__((always_inline));
  inline friend int operator==(const unsigned short&, const double_st&)__attribute__((always_inline));
  inline friend int operator==(const short&, const double_st&)__attribute__((always_inline));
  inline friend int operator==(const unsigned char&, const double_st&)__attribute__((always_inline));
  inline friend int operator==(const char&, const double_st&)__attribute__((always_inline));
#ifdef CADNA_HALF_EMULATION
  inline friend int operator==(const half&, const double_st&)__attribute__((always_inline));
#endif

#ifdef CADNA_HALF_NATIVE
  inline friend int operator==(const half&, const double_st&)__attribute__((always_inline));
#endif

  inline friend int operator!=(const double_st&, const double_st&)__attribute__((always_inline));

#ifdef CADNA_HALF_EMULATION
  inline friend int operator!=(const double_st&, const half_st&)__attribute__((always_inline));
  inline friend int operator!=(const half_st&, const double_st&)__attribute__((always_inline));
#endif 

#ifdef CADNA_HALF_NATIVE
  inline friend int operator!=(const double_st&, const half_st&)__attribute__((always_inline));
  inline friend int operator!=(const half_st&, const double_st&)__attribute__((always_inline));
#endif

  inline friend int operator!=(const double_st&, const float_st&)__attribute__((always_inline));
  inline friend int operator!=(const float_st&, const double_st&)__attribute__((always_inline));

  inline friend int operator!=(const double_st&, const double&)__attribute__((always_inline));
  inline friend int operator!=(const double_st&, const float&)__attribute__((always_inline));
  inline friend int operator!=(const double_st&, const unsigned long long&)__attribute__((always_inline));
  inline friend int operator!=(const double_st&, const long long&)__attribute__((always_inline));
  inline friend int operator!=(const double_st&, const unsigned long&)__attribute__((always_inline));
  inline friend int operator!=(const double_st&, const long&)__attribute__((always_inline));
  inline friend int operator!=(const double_st&, const unsigned int&)__attribute__((always_inline));
  inline friend int operator!=(const double_st&, const int&)__attribute__((always_inline));
  inline friend int operator!=(const double_st&, const unsigned short&)__attribute__((always_inline));
  inline friend int operator!=(const double_st&, const short&)__attribute__((always_inline));
  inline friend int operator!=(const double_st&, const unsigned char&)__attribute__((always_inline));
  inline friend int operator!=(const double_st&, const char&)__attribute__((always_inline));
#ifdef CADNA_HALF_EMULATION
    inline friend int operator!=(const double_st&, const half&)__attribute__((always_inline));
#endif 

#ifdef CADNA_HALF_NATIVE
    inline friend int operator!=(const double_st&, const half&)__attribute__((always_inline));
#endif 

  inline friend int operator!=(const double&, const double_st&)__attribute__((always_inline));
  inline friend int operator!=(const float&, const double_st&)__attribute__((always_inline));
  inline friend int operator!=(const unsigned long long&, const double_st&)__attribute__((always_inline));
  inline friend int operator!=(const long long&, const double_st&)__attribute__((always_inline));
  inline friend int operator!=(const unsigned long&, const double_st&)__attribute__((always_inline));
  inline friend int operator!=(const long&, const double_st&)__attribute__((always_inline));
  inline friend int operator!=(const unsigned int&, const double_st&)__attribute__((always_inline));
  inline friend int operator!=(const int&, const double_st&)__attribute__((always_inline));
  inline friend int operator!=(const unsigned short&, const double_st&)__attribute__((always_inline));
  inline friend int operator!=(const short&, const double_st&)__attribute__((always_inline));
  inline friend int operator!=(const unsigned char&, const double_st&)__attribute__((always_inline));
  inline friend int operator!=(const char&, const double_st&)__attribute__((always_inline));
#ifdef CADNA_HALF_EMULATION
  inline friend int operator!=(const half&, const double_st&)__attribute__((always_inline));
#endif 

#ifdef CADNA_HALF_NATIVE
  inline friend int operator!=(const half&, const double_st&)__attribute__((always_inline));
#endif 

  inline friend int operator>=(const double_st&, const double_st&)__attribute__((always_inline));
  inline friend int operator>=(const double_st&, const float_st&)__attribute__((always_inline));
  inline friend int operator>=(const float_st&, const double_st&)__attribute__((always_inline));

  inline friend int operator>=(const double_st&, const double&)__attribute__((always_inline));
  inline friend int operator>=(const double_st&, const float&)__attribute__((always_inline));
  inline friend int operator>=(const double_st&, const unsigned long long&)__attribute__((always_inline));
  inline friend int operator>=(const double_st&, const long long&)__attribute__((always_inline));
  inline friend int operator>=(const double_st&, const unsigned long&)__attribute__((always_inline));
  inline friend int operator>=(const double_st&, const long&)__attribute__((always_inline));
  inline friend int operator>=(const double_st&, const unsigned int&)__attribute__((always_inline));
  inline friend int operator>=(const double_st&, const int&)__attribute__((always_inline));
  inline friend int operator>=(const double_st&, const unsigned short&)__attribute__((always_inline));
  inline friend int operator>=(const double_st&, const short&)__attribute__((always_inline));
  inline friend int operator>=(const double_st&, const unsigned char&)__attribute__((always_inline));
  inline friend int operator>=(const double_st&, const char&)__attribute__((always_inline));
#ifdef CADNA_HALF_EMULATION
  inline friend int operator>=(const double_st&, const half_st&)__attribute__((always_inline));  
  inline friend int operator>=(const half_st&, const double_st&)__attribute__((always_inline));  
  inline friend int operator>=(const double_st&, const half&)__attribute__((always_inline));
#endif

#ifdef CADNA_HALF_NATIVE
  inline friend int operator>=(const double_st&, const half_st&)__attribute__((always_inline));  
  inline friend int operator>=(const half_st&, const double_st&)__attribute__((always_inline));  
  inline friend int operator>=(const double_st&, const half&)__attribute__((always_inline));
#endif


  inline friend int operator>=(const double&, const double_st&)__attribute__((always_inline));
  inline friend int operator>=(const float&, const double_st&)__attribute__((always_inline));
  inline friend int operator>=(const unsigned long long&, const double_st&)__attribute__((always_inline));
  inline friend int operator>=(const long long&, const double_st&)__attribute__((always_inline));
  inline friend int operator>=(const unsigned long&, const double_st&)__attribute__((always_inline));
  inline friend int operator>=(const long&, const double_st&)__attribute__((always_inline));
  inline friend int operator>=(const unsigned int&, const double_st&)__attribute__((always_inline));
  inline friend int operator>=(const int&, const double_st&)__attribute__((always_inline));
  inline friend int operator>=(const unsigned short&, const double_st&)__attribute__((always_inline));
  inline friend int operator>=(const short&, const double_st&)__attribute__((always_inline));
  inline friend int operator>=(const unsigned char&, const double_st&)__attribute__((always_inline));
  inline friend int operator>=(const char&, const double_st&)__attribute__((always_inline));
#ifdef CADNA_HALF_EMULATION
  inline friend int operator>=(const half&, const double_st&)__attribute__((always_inline));
#endif

#ifdef CADNA_HALF_NATIVE
  inline friend int operator>=(const half&, const double_st&)__attribute__((always_inline));
#endif

  inline friend int operator>(const double_st&, const double_st&)__attribute__((always_inline));
  inline friend int operator>(const double_st&, const float_st&)__attribute__((always_inline));
  inline friend int operator>(const float_st&, const double_st&)__attribute__((always_inline));

  inline friend int operator>(const double_st&, const double&)__attribute__((always_inline));
  inline friend int operator>(const double_st&, const float&)__attribute__((always_inline));
  inline friend int operator>(const double_st&, const unsigned long long&)__attribute__((always_inline));
  inline friend int operator>(const double_st&, const long long&)__attribute__((always_inline));
  inline friend int operator>(const double_st&, const unsigned long&)__attribute__((always_inline));
  inline friend int operator>(const double_st&, const long&)__attribute__((always_inline));
  inline friend int operator>(const double_st&, const unsigned int&)__attribute__((always_inline));
  inline friend int operator>(const double_st&, const int&)__attribute__((always_inline));
  inline friend int operator>(const double_st&, const unsigned short&)__attribute__((always_inline));
  inline friend int operator>(const double_st&, const short&)__attribute__((always_inline));
  inline friend int operator>(const double_st&, const unsigned char&)__attribute__((always_inline));
  inline friend int operator>(const double_st&, const char&)__attribute__((always_inline));
#ifdef CADNA_HALF_EMULATION
  inline friend int operator>(const double_st&, const half_st&)__attribute__((always_inline));  
  inline friend int operator>(const half_st&, const double_st&)__attribute__((always_inline));  
  inline friend int operator>(const double_st&, const half&)__attribute__((always_inline));
#endif

#ifdef CADNA_HALF_NATIVE
  inline friend int operator>(const double_st&, const half_st&)__attribute__((always_inline));  
  inline friend int operator>(const half_st&, const double_st&)__attribute__((always_inline));  
  inline friend int operator>(const double_st&, const half&)__attribute__((always_inline));
#endif

  inline friend int operator>(const double&, const double_st&)__attribute__((always_inline));
  inline friend int operator>(const float&, const double_st&)__attribute__((always_inline));
  inline friend int operator>(const unsigned long long&, const double_st&)__attribute__((always_inline));
  inline friend int operator>(const long long&, const double_st&)__attribute__((always_inline));
  inline friend int operator>(const unsigned long&, const double_st&)__attribute__((always_inline));
  inline friend int operator>(const long&, const double_st&)__attribute__((always_inline));
  inline friend int operator>(const unsigned int&, const double_st&)__attribute__((always_inline));
  inline friend int operator>(const int&, const double_st&)__attribute__((always_inline));
  inline friend int operator>(const unsigned short&, const double_st&)__attribute__((always_inline));
  inline friend int operator>(const short&, const double_st&)__attribute__((always_inline));
  inline friend int operator>(const unsigned char&, const double_st&)__attribute__((always_inline));
  inline friend int operator>(const char&, const double_st&)__attribute__((always_inline));
#ifdef CADNA_HALF_EMULATION
  inline friend int operator>(const half&, const double_st&)__attribute__((always_inline));
#endif

#ifdef CADNA_HALF_NATIVE
  inline friend int operator>(const half&, const double_st&)__attribute__((always_inline));
#endif

  inline friend int operator<=(const double_st&, const double_st&)__attribute__((always_inline));
  inline friend int operator<=(const double_st&, const float_st&)__attribute__((always_inline));
  inline friend int operator<=(const float_st&, const double_st&)__attribute__((always_inline));

  inline friend int operator<=(const double_st&, const double&)__attribute__((always_inline));
  inline friend int operator<=(const double_st&, const float&)__attribute__((always_inline));
  inline friend int operator<=(const double_st&, const unsigned long long&)__attribute__((always_inline));
  inline friend int operator<=(const double_st&, const long long&)__attribute__((always_inline));
  inline friend int operator<=(const double_st&, const unsigned long&)__attribute__((always_inline));
  inline friend int operator<=(const double_st&, const long&)__attribute__((always_inline));
  inline friend int operator<=(const double_st&, const unsigned int&)__attribute__((always_inline));
  inline friend int operator<=(const double_st&, const int&)__attribute__((always_inline));
  inline friend int operator<=(const double_st&, const unsigned short&)__attribute__((always_inline));
  inline friend int operator<=(const double_st&, const short&)__attribute__((always_inline));
  inline friend int operator<=(const double_st&, const unsigned char&)__attribute__((always_inline));
  inline friend int operator<=(const double_st&, const char&)__attribute__((always_inline));
#ifdef CADNA_HALF_EMULATION
  inline friend int operator<=(const double_st&, const half_st&)__attribute__((always_inline));  
  inline friend int operator<=(const half_st&, const double_st&)__attribute__((always_inline));  
  inline friend int operator<=(const double_st&, const half&)__attribute__((always_inline));
#endif

#ifdef CADNA_HALF_NATIVE
  inline friend int operator<=(const double_st&, const half_st&)__attribute__((always_inline));  
  inline friend int operator<=(const half_st&, const double_st&)__attribute__((always_inline));  
  inline friend int operator<=(const double_st&, const half&)__attribute__((always_inline));
#endif

  inline friend int operator<=(const double&, const double_st&)__attribute__((always_inline));
  inline friend int operator<=(const float&, const double_st&)__attribute__((always_inline));
  inline friend int operator<=(const unsigned long long&, const double_st&)__attribute__((always_inline));
  inline friend int operator<=(const long long&, const double_st&)__attribute__((always_inline));
  inline friend int operator<=(const unsigned long&, const double_st&)__attribute__((always_inline));
  inline friend int operator<=(const long&, const double_st&)__attribute__((always_inline));
  inline friend int operator<=(const unsigned int&, const double_st&)__attribute__((always_inline));
  inline friend int operator<=(const int&, const double_st&)__attribute__((always_inline));
  inline friend int operator<=(const unsigned short&, const double_st&)__attribute__((always_inline));
  inline friend int operator<=(const short&, const double_st&)__attribute__((always_inline));
  inline friend int operator<=(const unsigned char&, const double_st&)__attribute__((always_inline));
  inline friend int operator<=(const char&, const double_st&)__attribute__((always_inline));
#ifdef CADNA_HALF_EMULATION
  inline friend int operator<=(const half&, const double_st&)__attribute__((always_inline));
#endif

#ifdef CADNA_HALF_NATIVE
  inline friend int operator<=(const half&, const double_st&)__attribute__((always_inline));
#endif

  inline friend int operator<(const double_st&, const double_st&)__attribute__((always_inline));
  inline friend int operator<(const double_st&, const float_st&)__attribute__((always_inline));
  inline friend int operator<(const float_st&, const double_st&)__attribute__((always_inline));

  inline friend int operator<(const double_st&, const double&)__attribute__((always_inline));
  inline friend int operator<(const double_st&, const float&)__attribute__((always_inline));
  inline friend int operator<(const double_st&, const unsigned long long&)__attribute__((always_inline));
  inline friend int operator<(const double_st&, const long long&)__attribute__((always_inline));
  inline friend int operator<(const double_st&, const unsigned long&)__attribute__((always_inline));
  inline friend int operator<(const double_st&, const long&)__attribute__((always_inline));
  inline friend int operator<(const double_st&, const unsigned int&)__attribute__((always_inline));
  inline friend int operator<(const double_st&, const int&)__attribute__((always_inline));
  inline friend int operator<(const double_st&, const unsigned short&)__attribute__((always_inline));
  inline friend int operator<(const double_st&, const short&)__attribute__((always_inline));
  inline friend int operator<(const double_st&, const unsigned char&)__attribute__((always_inline));
  inline friend int operator<(const double_st&, const char&)__attribute__((always_inline));
#ifdef CADNA_HALF_EMULATION
  inline friend int operator<(const double_st&, const half_st&)__attribute__((always_inline));  
  inline friend int operator<(const half_st&, const double_st&)__attribute__((always_inline));  
  inline friend int operator<(const double_st&, const half&)__attribute__((always_inline));
#endif

#ifdef CADNA_HALF_NATIVE
  inline friend int operator<(const double_st&, const half_st&)__attribute__((always_inline));  
  inline friend int operator<(const half_st&, const double_st&)__attribute__((always_inline));  
  inline friend int operator<(const double_st&, const half&)__attribute__((always_inline));
#endif

  inline friend int operator<(const double&, const double_st&)__attribute__((always_inline));
  inline friend int operator<(const float&, const double_st&)__attribute__((always_inline));
  inline friend int operator<(const unsigned long long&, const double_st&)__attribute__((always_inline));
  inline friend int operator<(const long long&, const double_st&)__attribute__((always_inline));
  inline friend int operator<(const unsigned long&, const double_st&)__attribute__((always_inline));
  inline friend int operator<(const long&, const double_st&)__attribute__((always_inline));
  inline friend int operator<(const unsigned int&, const double_st&)__attribute__((always_inline));
  inline friend int operator<(const int&, const double_st&)__attribute__((always_inline));
  inline friend int operator<(const unsigned short&, const double_st&)__attribute__((always_inline));
  inline friend int operator<(const short&, const double_st&)__attribute__((always_inline));
  inline friend int operator<(const unsigned char&, const double_st&)__attribute__((always_inline));
  inline friend int operator<(const char&, const double_st&)__attribute__((always_inline));
#ifdef CADNA_HALF_EMULATION
  inline friend int operator<(const half&, const double_st&)__attribute__((always_inline));
#endif

#ifdef CADNA_HALF_NATIVE
  inline friend int operator<(const half&, const double_st&)__attribute__((always_inline));
#endif

  // MATHEMATICAL FUNCTIONS 

  friend double_st  log(const double_st&);
  friend double_st  log2(const double_st&);
  friend double_st  log10(const double_st&);
  friend double_st  log1p(const double_st&);
  friend double_st  logb(const double_st&);
  friend double_st  exp(const double_st&); 
  friend double_st  exp2(const double_st&); 
  friend double_st  expm1(const double_st&); 
  friend double_st  sqrt(const double_st&);
  friend double_st  cbrt(const double_st&);
  friend double_st  sin(const double_st&);
  friend double_st  cos(const double_st&);
  friend double_st  tan(const double_st&);
  friend double_st  asin(const double_st&);
  friend double_st  acos(const double_st&);
  friend double_st  atan(const double_st&);
  friend double_st  atan2(const double_st&, const double_st&);
  friend double_st  atan2(const double_st&, const float_st&);
  friend double_st  atan2(const float_st&, const double_st&);
  friend double_st  sinh(const double_st&);
  friend double_st  cosh(const double_st&);
  friend double_st  tanh(const double_st&);
  friend double_st  asinh(const double_st&);
  friend double_st  acosh(const double_st&);
  friend double_st  atanh(const double_st&);
  friend double_st  hypot(const double_st&, const double_st&);
  friend double_st  hypot(const double_st&, const float_st&);
  friend double_st  hypot(const float_st&, const double_st&);

  
  friend double_st pow(const double_st&, const double_st&);  
  friend double_st pow(const double_st&, const float_st&);  
  friend double_st pow(const float_st&, const double_st&);  
  friend double_st pow(const double_st&, const double&);
  friend double_st pow(const double_st&, const float&);
  friend double_st pow(const double_st&, const unsigned long long&);
  friend double_st pow(const double_st&, const long long&);
  friend double_st pow(const double_st&, const unsigned long&);
  friend double_st pow(const double_st&, const long&);
  friend double_st pow(const double_st&, const unsigned int&);
  friend double_st pow(const double_st&, const int&);
  friend double_st pow(const double_st&, const unsigned short&);
  friend double_st pow(const double_st&, const short&);
  friend double_st pow(const double_st&, const unsigned char&);
  friend double_st pow(const double_st&, const char&);
#ifdef CADNA_HALF_EMULATION
  friend double_st pow(const double_st&, const half&);
  friend half_st pow(const half_st&, const double_st&);  
  friend half_st pow(const double_st&, const half_st&);  
#endif

  friend double_st pow(const double&, const double_st&);
  friend double_st pow(const float&, const double_st&);
  friend double_st pow(const unsigned long long&, const double_st&);
  friend double_st pow(const long long&, const double_st&);
  friend double_st pow(const unsigned long&, const double_st&);
  friend double_st pow(const long&, const double_st&);
  friend double_st pow(const unsigned int&, const double_st&);
  friend double_st pow(const int&, const double_st&);
  friend double_st pow(const unsigned short&, const double_st&);
  friend double_st pow(const short&, const double_st&);
  friend double_st pow(const unsigned char&, const double_st&);
  friend double_st pow(const char&, const double_st&);


  friend double_st fmax(const double_st&, const double_st&);  
  friend double_st fmax(const double_st&, const float_st&);  
  friend double_st fmax(const float_st&, const double_st&);  
  friend double_st fmax(const double_st&, const double&);
  friend double_st fmax(const double_st&, const float&);
  friend double_st fmax(const double_st&, const unsigned long long&);
  friend double_st fmax(const double_st&, const long long&);
  friend double_st fmax(const double_st&, const unsigned long&);
  friend double_st fmax(const double_st&, const long&);
  friend double_st fmax(const double_st&, const unsigned int&);
  friend double_st fmax(const double_st&, const int&);
  friend double_st fmax(const double_st&, const unsigned short&);
  friend double_st fmax(const double_st&, const short&);
  friend double_st fmax(const double_st&, const unsigned char&);
  friend double_st fmax(const double_st&, const char&);

  friend double_st fmax(const double&, const double_st&);
  friend double_st fmax(const float&, const double_st&);
  friend double_st fmax(const unsigned long long&, const double_st&);
  friend double_st fmax(const long long&, const double_st&);
  friend double_st fmax(const unsigned long&, const double_st&);
  friend double_st fmax(const long&, const double_st&);
  friend double_st fmax(const unsigned int&, const double_st&);
  friend double_st fmax(const int&, const double_st&);
  friend double_st fmax(const unsigned short&, const double_st&);
  friend double_st fmax(const short&, const double_st&);
  friend double_st fmax(const unsigned char&, const double_st&);
  friend double_st fmax(const char&, const double_st&);
 

  friend double_st fmin(const double_st&, const double_st&);  
  friend double_st fmin(const double_st&, const float_st&);  
  friend double_st fmin(const float_st&, const double_st&);  
  friend double_st fmin(const double_st&, const double&);
  friend double_st fmin(const double_st&, const float&);
  friend double_st fmin(const double_st&, const unsigned long long&);
  friend double_st fmin(const double_st&, const long long&);
  friend double_st fmin(const double_st&, const unsigned long&);
  friend double_st fmin(const double_st&, const long&);
  friend double_st fmin(const double_st&, const unsigned int&);
  friend double_st fmin(const double_st&, const int&);
  friend double_st fmin(const double_st&, const unsigned short&);
  friend double_st fmin(const double_st&, const short&);
  friend double_st fmin(const double_st&, const unsigned char&);
  friend double_st fmin(const double_st&, const char&);

  
  friend double_st fmin(const double&, const double_st&);
  friend double_st fmin(const float&, const double_st&);
  friend double_st fmin(const unsigned long long&, const double_st&);
  friend double_st fmin(const long long&, const double_st&);
  friend double_st fmin(const unsigned long&, const double_st&);
  friend double_st fmin(const long&, const double_st&);
  friend double_st fmin(const unsigned int&, const double_st&);
  friend double_st fmin(const int&, const double_st&);
  friend double_st fmin(const unsigned short&, const double_st&);
  friend double_st fmin(const short&, const double_st&);
  friend double_st fmin(const unsigned char&, const double_st&);
  friend double_st fmin(const char&, const double_st&);

  
  friend double_st frexp(const double_st&, int* );
  friend double_st modf(const  double_st&,  double_st*);
  friend double_st ldexp(const double_st&, const int);
  friend double_st scalbn(const double_st&, const int);
  friend double_st fmod(const double_st&, const double_st&);
  friend double_st copysign(const double_st&, const double_st&);

  friend double_st  erf(const double_st&);
  friend double_st  erfc(const double_st&);

  // INTRINSIC FUNCTIONS
  
  friend double_st fabs(const double_st&);
  friend double_st abs(const double_st&);
  friend double_st floor(const double_st&);
  friend double_st ceil(const double_st&);
  friend double_st trunc(const double_st&);
  friend double_st nearbyint(const double_st&);
  friend double_st rint(const double_st&);
  friend long int  lrint(const double_st&);
  friend long long int llrint(const double_st&);

  friend int finite(const double_st&);
  friend int isfinite(const double_st&);
  friend int isnan(const double_st&);
  friend int isinf(const double_st&);


  // Conversion
  operator char();  
  operator unsigned char();  
  operator short();  
  operator unsigned short();  
  operator int();  
  operator unsigned int();  
  operator long();  
  operator unsigned long();  
  operator long long();  
  operator unsigned long long();  
  operator float();
  operator double();
  operator float_st();
#ifdef CADNA_HALF_EMULATION
  operator half();
  operator half_st();
#endif

#ifdef CADNA_HALF_NATIVE
  operator half();
  operator half_st();
#endif
  
  operator float_st() const;
  operator char() const;  
  operator unsigned char() const;  
  operator short() const;  
  operator unsigned short() const;  
  operator int() const;  
  operator unsigned int() const;  
  operator long() const;  
  operator unsigned long() const;  
  operator long long() const;  
  operator unsigned long long() const;  
  operator float() const;
  operator double() const;
#ifdef CADNA_HALF_EMULATION
  operator half() const;
  operator half_st() const;
#endif

#ifdef CADNA_HALF_NATIVE
  operator half() const;
  operator half_st() const;
#endif
  
  inline int nb_significant_digit() const __attribute__((always_inline));
  inline int approx_digit() const __attribute__((always_inline));
  inline int computedzero() const __attribute__((always_inline));
  inline int numericalnoise() const __attribute__((always_inline));
  void display() const ; 
  void display(const char *) const ; 
  char* str( char *)  const ;
  
  friend char* strp(const double_st&);
  friend char* str( char *, const double_st&);
  
  friend std::istream& operator >>(std::istream& s, double_st& );  
  
  void data_st();
  void data_st(const double&, const int&);
  
};

std::ostream& operator<<(std::ostream&, const double_st&);
std::istream& operator >>(std::istream&, double_st& );

///////////////////////////////
///////////////////////////////
///////////////////////////////

class float_st {
  friend class double_st;

#ifdef CADNA_HALF_EMULATION
  friend class half_st;
#endif 

#ifdef CADNA_HALF_NATIVE
  friend class half_st;
#endif 
  
  typedef float TYPEBASE;
  
public:
  float x,y,z;
  mutable int accuracy;
  inline float_st(void) :x(0),y(0),z(0),accuracy(7) {};
  inline float_st(const double &a, const double &b, const double &c):x(a),y(b),z(c),accuracy(-1) {};

  inline float_st(const double &a): x(a),y(a),z(a),accuracy(7) {};
  inline float_st (const double_st&a) : x(a.x),y(a.y),z(a.z),accuracy(a.accuracy>=8?7:a.accuracy) {};

#ifdef CADNA_HALF_EMULATION
  inline float_st(const half_st&);
#endif

#ifdef CADNA_HALF_NATIVE
  inline float_st(const half_st&);
#endif

  float getx() {return(this->x);}
  float gety() {return(this->y);}
  float getz() {return(this->z);}
  int getaccuracy() {return(this->accuracy);}


  // ASSIGNMENT

 // inline float_st& operator=(const float_st&)__attribute__((always_inline)); 
  inline float_st& operator=(const double&)__attribute__((always_inline));
  inline float_st& operator=(const float&)__attribute__((always_inline));
  inline float_st& operator=(const unsigned long long&)__attribute__((always_inline));
  inline float_st& operator=(const long long&)__attribute__((always_inline));
  inline float_st& operator=(const unsigned long&)__attribute__((always_inline));
  inline float_st& operator=(const long&)__attribute__((always_inline));
  inline float_st& operator=(const unsigned int&)__attribute__((always_inline));
  inline float_st& operator=(const int&)__attribute__((always_inline));
  inline float_st& operator=(const unsigned short&)__attribute__((always_inline));
  inline float_st& operator=(const short&)__attribute__((always_inline));
  inline float_st& operator=(const unsigned char&)__attribute__((always_inline));
  inline float_st& operator=(const char&)__attribute__((always_inline));
#ifdef CADNA_HALF_EMULATION
  inline float_st& operator=(const half_st&)__attribute__((always_inline));
  inline float_st& operator=(const half&)__attribute__((always_inline));
#endif 

#ifdef CADNA_HALF_NATIVE
  inline float_st& operator=(const half_st&)__attribute__((always_inline));
  inline float_st& operator=(const half&)__attribute__((always_inline));
#endif 
  inline float_st& operator=(const double_st&)__attribute__((always_inline));


  // ADDITION
  inline float_st operator++()__attribute__((always_inline));
  inline float_st operator++(int)__attribute__((always_inline));
  inline float_st operator+() const __attribute__((always_inline)); 
#ifdef CADNA_HALF_EMULATION
  inline float_st& operator+=(const half_st& a)__attribute__((always_inline));
#endif 
#ifdef CADNA_HALF_NATIVE
  inline float_st& operator+=(const half_st& a)__attribute__((always_inline));
#endif 
  inline float_st& operator+=(const double_st&)__attribute__((always_inline));
  inline float_st& operator+=(const float_st&)__attribute__((always_inline));
  inline float_st& operator+=(const double&)__attribute__((always_inline));
  inline float_st& operator+=(const float&)__attribute__((always_inline));
  inline float_st& operator+=(const unsigned long long&)__attribute__((always_inline));
  inline float_st& operator+=(const long long&)__attribute__((always_inline));
  inline float_st& operator+=(const unsigned long&)__attribute__((always_inline));
  inline float_st& operator+=(const long&)__attribute__((always_inline));
  inline float_st& operator+=(const unsigned int&)__attribute__((always_inline));
  inline float_st& operator+=(const int&)__attribute__((always_inline));
  inline float_st& operator+=(const unsigned short&)__attribute__((always_inline));
  inline float_st& operator+=(const short&)__attribute__((always_inline));
  inline float_st& operator+=(const unsigned char&)__attribute__((always_inline));
  inline float_st& operator+=(const char&)__attribute__((always_inline));
#ifdef CADNA_HALF_EMULATION
  inline float_st& operator+=(const half& a)__attribute__((always_inline));
#endif 
#ifdef CADNA_HALF_NATIVE
  inline float_st& operator+=(const half& a)__attribute__((always_inline));
#endif 
  
  inline friend double_st operator+(const double_st&, const float_st&)__attribute__((always_inline));  
  inline friend double_st operator+(const float_st&,  const double_st&)__attribute__((always_inline));  
  inline friend float_st operator+(const float_st&, const float_st&)__attribute__((always_inline));  

  inline friend float_st operator+(const float_st&, const double&)__attribute__((always_inline));
  inline friend float_st operator+(const float_st&, const float&)__attribute__((always_inline));
  inline friend float_st operator+(const float_st&, const unsigned long long&)__attribute__((always_inline));
  inline friend float_st operator+(const float_st&, const long long&)__attribute__((always_inline));
  inline friend float_st operator+(const float_st&, const unsigned long&)__attribute__((always_inline));
  inline friend float_st operator+(const float_st&, const long&)__attribute__((always_inline));
  inline friend float_st operator+(const float_st&, const unsigned int&)__attribute__((always_inline));
  inline friend float_st operator+(const float_st&, const int&)__attribute__((always_inline));
  inline friend float_st operator+(const float_st&, const unsigned short&)__attribute__((always_inline));
  inline friend float_st operator+(const float_st&, const short&)__attribute__((always_inline));
  inline friend float_st operator+(const float_st&, const unsigned char&)__attribute__((always_inline));
  inline friend float_st operator+(const float_st&, const char&)__attribute__((always_inline));

  inline friend float_st operator+(const double&, const float_st&)__attribute__((always_inline));
  inline friend float_st operator+(const float&, const float_st&)__attribute__((always_inline));
  inline friend float_st operator+(const unsigned long long&, const float_st&)__attribute__((always_inline));
  inline friend float_st operator+(const long long&, const float_st&)__attribute__((always_inline));
  inline friend float_st operator+(const unsigned long&, const float_st&)__attribute__((always_inline));
  inline friend float_st operator+(const long&, const float_st&)__attribute__((always_inline));
  inline friend float_st operator+(const unsigned int&, const float_st&)__attribute__((always_inline));
  inline friend float_st operator+(const int&, const float_st&)__attribute__((always_inline));
  inline friend float_st operator+(const unsigned short&, const float_st&)__attribute__((always_inline));
  inline friend float_st operator+(const short&, const float_st&)__attribute__((always_inline));
  inline friend float_st operator+(const unsigned char&, const float_st&)__attribute__((always_inline));
  inline friend float_st operator+(const char&, const float_st&)__attribute__((always_inline));



#ifdef CADNA_HALF_EMULATION
  inline friend float_st operator+(const half_st&, const float_st&)__attribute__((always_inline));
  inline friend float_st operator+(const float_st&, const half_st&)__attribute__((always_inline));
  inline friend float_st operator+(const half&, const float_st&)__attribute__((always_inline));
  inline friend float_st operator+(const float_st&, const half&)__attribute__((always_inline));
#endif 

#ifdef CADNA_HALF_NATIVE
  inline friend float_st operator+(const half_st&, const float_st&)__attribute__((always_inline));
  inline friend float_st operator+(const float_st&, const half_st&)__attribute__((always_inline));
  inline friend float_st operator+(const half&, const float_st&)__attribute__((always_inline));
  inline friend float_st operator+(const float_st&, const half&)__attribute__((always_inline));
#endif 


  // SUBTRACTION

  inline float_st operator--()__attribute__((always_inline));
  inline float_st operator--(int)__attribute__((always_inline));
  inline float_st operator-() const __attribute__((always_inline)); 

  inline float_st& operator-=(const double_st&)__attribute__((always_inline));
  inline float_st& operator-=(const float_st&)__attribute__((always_inline));
  inline float_st& operator-=(const double&)__attribute__((always_inline));
  inline float_st& operator-=(const float&)__attribute__((always_inline));
  inline float_st& operator-=(const unsigned long long&)__attribute__((always_inline));
  inline float_st& operator-=(const long long&)__attribute__((always_inline));
  inline float_st& operator-=(const unsigned long&)__attribute__((always_inline));
  inline float_st& operator-=(const long&)__attribute__((always_inline));
  inline float_st& operator-=(const unsigned int&)__attribute__((always_inline));
  inline float_st& operator-=(const int&)__attribute__((always_inline));
  inline float_st& operator-=(const unsigned short&)__attribute__((always_inline));
  inline float_st& operator-=(const short&)__attribute__((always_inline));
  inline float_st& operator-=(const unsigned char&)__attribute__((always_inline));
  inline float_st& operator-=(const char&)__attribute__((always_inline));
#ifdef CADNA_HALF_EMULATION
  inline float_st& operator-=(const half&)__attribute__((always_inline));
  inline float_st& operator-=(const half_st&)__attribute__((always_inline));
#endif 

#ifdef CADNA_HALF_NATIVE
  inline float_st& operator-=(const half&)__attribute__((always_inline));
  inline float_st& operator-=(const half_st&)__attribute__((always_inline));
#endif

  inline friend double_st operator-(const double_st&, const float_st&)__attribute__((always_inline));  
  inline friend double_st operator-(const float_st&, const double_st&)__attribute__((always_inline));  
  inline friend float_st operator-(const float_st&, const float_st&)__attribute__((always_inline));  

  inline friend float_st operator-(const float_st&, const double&)__attribute__((always_inline));
  inline friend float_st operator-(const float_st&, const float&)__attribute__((always_inline));
  inline friend float_st operator-(const float_st&, const unsigned long long&)__attribute__((always_inline));
  inline friend float_st operator-(const float_st&, const long long&)__attribute__((always_inline));
  inline friend float_st operator-(const float_st&, const unsigned long&)__attribute__((always_inline));
  inline friend float_st operator-(const float_st&, const long&)__attribute__((always_inline));
  inline friend float_st operator-(const float_st&, const unsigned int&)__attribute__((always_inline));
  inline friend float_st operator-(const float_st&, const int&)__attribute__((always_inline));
  inline friend float_st operator-(const float_st&, const unsigned short&)__attribute__((always_inline));
  inline friend float_st operator-(const float_st&, const short&)__attribute__((always_inline));
  inline friend float_st operator-(const float_st&, const unsigned char&)__attribute__((always_inline));
  inline friend float_st operator-(const float_st&, const char&)__attribute__((always_inline));

#ifdef CADNA_HALF_EMULATION
  inline friend float_st operator-(const float_st&, const half&)__attribute__((always_inline));
#endif 
#ifdef CADNA_HALF_NATIVE
  inline friend float_st operator-(const float_st&, const half&)__attribute__((always_inline));
#endif 
  inline friend float_st operator-(const double&, const float_st&)__attribute__((always_inline));
  inline friend float_st operator-(const float&, const float_st&)__attribute__((always_inline));
  inline friend float_st operator-(const unsigned long long&, const float_st&)__attribute__((always_inline));
  inline friend float_st operator-(const long long&, const float_st&)__attribute__((always_inline));
  inline friend float_st operator-(const unsigned long&, const float_st&)__attribute__((always_inline));
  inline friend float_st operator-(const long&, const float_st&)__attribute__((always_inline));
  inline friend float_st operator-(const unsigned int&, const float_st&)__attribute__((always_inline));
  inline friend float_st operator-(const int&, const float_st&)__attribute__((always_inline));
  inline friend float_st operator-(const unsigned short&, const float_st&)__attribute__((always_inline));
  inline friend float_st operator-(const short&, const float_st&)__attribute__((always_inline));
  inline friend float_st operator-(const unsigned char&, const float_st&)__attribute__((always_inline));
  inline friend float_st operator-(const char&, const float_st&)__attribute__((always_inline));
#ifdef CADNA_HALF_EMULATION
  inline friend float_st operator-(const half&, const float_st&)__attribute__((always_inline));
#endif 
#ifdef CADNA_HALF_NATIVE
  inline friend float_st operator-(const half&, const float_st&)__attribute__((always_inline));
#endif

#ifdef CADNA_HALF_EMULATION
  inline friend float_st operator-(const half_st&, const float_st&)__attribute__((always_inline));
  inline friend float_st operator-(const float_st&, const half_st&)__attribute__((always_inline));
#endif 

#ifdef CADNA_HALF_NATIVE
  inline friend float_st operator-(const half_st&, const float_st&)__attribute__((always_inline));
  inline friend float_st operator-(const float_st&, const half_st&)__attribute__((always_inline));
#endif
  
  // PRODUCT
#ifdef CADNA_HALF_EMULATION
  inline float_st& operator*=(const half_st&)__attribute__((always_inline));
#endif 

#ifdef CADNA_HALF_NATIVE
  inline float_st& operator*=(const half_st&)__attribute__((always_inline));
#endif

  inline float_st& operator*=(const double_st&)__attribute__((always_inline));
  inline float_st& operator*=(const float_st&)__attribute__((always_inline));
  inline float_st& operator*=(const double&)__attribute__((always_inline));
  inline float_st& operator*=(const float&)__attribute__((always_inline));
  inline float_st& operator*=(const unsigned long long&)__attribute__((always_inline));
  inline float_st& operator*=(const long long&)__attribute__((always_inline));
  inline float_st& operator*=(const unsigned long&)__attribute__((always_inline));
  inline float_st& operator*=(const long&)__attribute__((always_inline));
  inline float_st& operator*=(const unsigned int&)__attribute__((always_inline));
  inline float_st& operator*=(const int&)__attribute__((always_inline));
  inline float_st& operator*=(const unsigned short&)__attribute__((always_inline));
  inline float_st& operator*=(const short&)__attribute__((always_inline));
  inline float_st& operator*=(const unsigned char&)__attribute__((always_inline));
  inline float_st& operator*=(const char&)__attribute__((always_inline));
#ifdef CADNA_HALF_EMULATION
  inline float_st& operator*=(const half&)__attribute__((always_inline));
  inline friend half_st& operator*=(const half_st&, const float_st)__attribute__((always_inline));
  inline friend half_st& operator*=(const float_st&, const half_st)__attribute__((always_inline));
#endif
#ifdef CADNA_HALF_NATIVE
  inline float_st& operator*=(const half&)__attribute__((always_inline));
  inline friend float_st& operator*=(const half_st&, const float_st)__attribute__((always_inline));
  inline friend float_st& operator*=(const float_st&, const half_st)__attribute__((always_inline));
#endif

  inline friend double_st operator*(const double_st&, const float_st&)__attribute__((always_inline));  
  inline friend double_st operator*(const float_st&, const double_st&)__attribute__((always_inline));  
  inline friend float_st operator*(const float_st&, const float_st&)__attribute__((always_inline));  

  inline friend float_st operator*(const float_st&, const double&)__attribute__((always_inline));
  inline friend float_st operator*(const float_st&, const float&)__attribute__((always_inline));
  inline friend float_st operator*(const float_st&, const unsigned long long&)__attribute__((always_inline));
  inline friend float_st operator*(const float_st&, const long long&)__attribute__((always_inline));
  inline friend float_st operator*(const float_st&, const unsigned long&)__attribute__((always_inline));
  inline friend float_st operator*(const float_st&, const long&)__attribute__((always_inline));
  inline friend float_st operator*(const float_st&, const unsigned int&)__attribute__((always_inline));
  inline friend float_st operator*(const float_st&, const int&)__attribute__((always_inline));
  inline friend float_st operator*(const float_st&, const unsigned short&)__attribute__((always_inline));
  inline friend float_st operator*(const float_st&, const short&)__attribute__((always_inline));
  inline friend float_st operator*(const float_st&, const unsigned char&)__attribute__((always_inline));
  inline friend float_st operator*(const float_st&, const char&)__attribute__((always_inline));
#ifdef CADNA_HALF_EMULATION
  inline friend float_st operator*(const float_st&, const half&)__attribute__((always_inline));
  inline friend float_st operator*(const half_st&, const float_st&)__attribute__((always_inline));  
  inline friend float_st operator*(const float_st&, const half_st&)__attribute__((always_inline));  
#endif
#ifdef CADNA_HALF_NATIVE
  inline friend float_st operator*(const float_st&, const half&)__attribute__((always_inline));
  inline friend float_st operator*(const half_st&, const float_st&)__attribute__((always_inline));  
  inline friend float_st operator*(const float_st&, const half_st&)__attribute__((always_inline));  
#endif 

  inline friend float_st operator*(const double&, const float_st&)__attribute__((always_inline));
  inline friend float_st operator*(const float&, const float_st&)__attribute__((always_inline));
  inline friend float_st operator*(const unsigned long long&, const float_st&)__attribute__((always_inline));
  inline friend float_st operator*(const long long&, const float_st&)__attribute__((always_inline));
  inline friend float_st operator*(const unsigned long&, const float_st&)__attribute__((always_inline));
  inline friend float_st operator*(const long&, const float_st&)__attribute__((always_inline));
  inline friend float_st operator*(const unsigned int&, const float_st&)__attribute__((always_inline));
  inline friend float_st operator*(const int&, const float_st&)__attribute__((always_inline));
  inline friend float_st operator*(const unsigned short&, const float_st&)__attribute__((always_inline));
  inline friend float_st operator*(const short&, const float_st&)__attribute__((always_inline));
  inline friend float_st operator*(const unsigned char&, const float_st&)__attribute__((always_inline));
  inline friend float_st operator*(const char&, const float_st&)__attribute__((always_inline));
#ifdef CADNA_HALF_EMULATION
  inline friend float_st operator*(const half&, const float_st&)__attribute__((always_inline));
#endif 
#ifdef CADNA_HALF_NATIVE
  inline friend float_st operator*(const half&, const float_st&)__attribute__((always_inline));
#endif 

  // DIVISION
  
  inline float_st& operator/=(const double_st&)__attribute__((always_inline));
  inline float_st& operator/=(const float_st&)__attribute__((always_inline));
  inline float_st& operator/=(const double&)__attribute__((always_inline));
  inline float_st& operator/=(const float&)__attribute__((always_inline));
  inline float_st& operator/=(const unsigned long long&)__attribute__((always_inline));
  inline float_st& operator/=(const long long&)__attribute__((always_inline));
  inline float_st& operator/=(const unsigned long&)__attribute__((always_inline));
  inline float_st& operator/=(const long&)__attribute__((always_inline));
  inline float_st& operator/=(const unsigned int&)__attribute__((always_inline));
  inline float_st& operator/=(const int&)__attribute__((always_inline));
  inline float_st& operator/=(const unsigned short&)__attribute__((always_inline));
  inline float_st& operator/=(const short&)__attribute__((always_inline));
  inline float_st& operator/=(const unsigned char&)__attribute__((always_inline));
  inline float_st& operator/=(const char&)__attribute__((always_inline));
#ifdef CADNA_HALF_EMULATION
  inline float_st& operator/=(const half&)__attribute__((always_inline));
  inline float_st& operator/=(const half_st&)__attribute__((always_inline));
#endif
#ifdef CADNA_HALF_NATIVE
  inline float_st& operator/=(const half&)__attribute__((always_inline));
  inline float_st& operator/=(const half_st&)__attribute__((always_inline));
#endif
  inline friend double_st operator/(const double_st&, const float_st&)__attribute__((always_inline));  
  inline friend double_st operator/(const float_st&, const double_st&)__attribute__((always_inline));  
  inline friend float_st operator/(const float_st&, const float_st&)__attribute__((always_inline));  

  inline friend float_st operator/(const float_st&, const double&)__attribute__((always_inline));
  inline friend float_st operator/(const float_st&, const float&)__attribute__((always_inline));
  inline friend float_st operator/(const float_st&, const unsigned long long&)__attribute__((always_inline));
  inline friend float_st operator/(const float_st&, const long long&)__attribute__((always_inline));
  inline friend float_st operator/(const float_st&, const unsigned long&)__attribute__((always_inline));
  inline friend float_st operator/(const float_st&, const long&)__attribute__((always_inline));
  inline friend float_st operator/(const float_st&, const unsigned int&)__attribute__((always_inline));
  inline friend float_st operator/(const float_st&, const int&)__attribute__((always_inline));
  inline friend float_st operator/(const float_st&, const unsigned short&)__attribute__((always_inline));
  inline friend float_st operator/(const float_st&, const short&)__attribute__((always_inline));
  inline friend float_st operator/(const float_st&, const unsigned char&)__attribute__((always_inline));
  inline friend float_st operator/(const float_st&, const char&)__attribute__((always_inline));
#ifdef CADNA_HALF_EMULATION
  inline friend float_st operator/(const float_st&, const half&)__attribute__((always_inline));
  inline friend float_st operator/(const half_st&, const float_st&)__attribute__((always_inline));
  inline friend float_st operator/(const float_st&, const half_st&)__attribute__((always_inline));
#endif
#ifdef CADNA_HALF_NATIVE
  inline friend float_st operator/(const float_st&, const half&)__attribute__((always_inline));
  inline friend float_st operator/(const half_st&, const float_st&)__attribute__((always_inline));
  inline friend float_st operator/(const float_st&, const half_st&)__attribute__((always_inline));
#endif

  inline friend float_st operator/(const double&, const float_st&)__attribute__((always_inline));
  inline friend float_st operator/(const float&, const float_st&)__attribute__((always_inline));
  inline friend float_st operator/(const unsigned long long&, const float_st&)__attribute__((always_inline));
  inline friend float_st operator/(const long long&, const float_st&)__attribute__((always_inline));
  inline friend float_st operator/(const unsigned long&, const float_st&)__attribute__((always_inline));
  inline friend float_st operator/(const long&, const float_st&)__attribute__((always_inline));
  inline friend float_st operator/(const unsigned int&, const float_st&)__attribute__((always_inline));
  inline friend float_st operator/(const int&, const float_st&)__attribute__((always_inline));
  inline friend float_st operator/(const unsigned short&, const float_st&)__attribute__((always_inline));
  inline friend float_st operator/(const short&, const float_st&)__attribute__((always_inline));
  inline friend float_st operator/(const unsigned char&, const float_st&)__attribute__((always_inline));
  inline friend float_st operator/(const char&, const float_st&)__attribute__((always_inline));
#ifdef CADNA_HALF_EMULATION
  inline friend float_st operator/(const half&, const float_st&)__attribute__((always_inline));
#endif
#ifdef CADNA_HALF_NATIVE
  inline friend float_st operator/(const half&, const float_st&)__attribute__((always_inline));
#endif



  // RELATIONAL OPERATORS

  inline friend int operator==(const double_st&, const float_st&)__attribute__((always_inline));  
  inline friend int operator==(const float_st&, const double_st&)__attribute__((always_inline));  
  inline friend int operator==(const float_st&, const float_st&)__attribute__((always_inline));  


  inline friend int operator==(const float_st&, const double&)__attribute__((always_inline));
  inline friend int operator==(const float_st&, const float&)__attribute__((always_inline));
  inline friend int operator==(const float_st&, const unsigned long long&)__attribute__((always_inline));
  inline friend int operator==(const float_st&, const long long&)__attribute__((always_inline));
  inline friend int operator==(const float_st&, const unsigned long&)__attribute__((always_inline));
  inline friend int operator==(const float_st&, const long&)__attribute__((always_inline));
  inline friend int operator==(const float_st&, const unsigned int&)__attribute__((always_inline));
  inline friend int operator==(const float_st&, const int&)__attribute__((always_inline));
  inline friend int operator==(const float_st&, const unsigned short&)__attribute__((always_inline));
  inline friend int operator==(const float_st&, const short&)__attribute__((always_inline));
  inline friend int operator==(const float_st&, const unsigned char&)__attribute__((always_inline));
  inline friend int operator==(const float_st&, const char&)__attribute__((always_inline));
#ifdef CADNA_HALF_EMULATION
  inline friend int operator==(const float_st&, const half&)__attribute__((always_inline));
#endif

#ifdef CADNA_HALF_NATIVE
  inline friend int operator==(const float_st&, const half&)__attribute__((always_inline));
#endif

  inline friend int operator==(const double&, const float_st&)__attribute__((always_inline));
  inline friend int operator==(const float&, const float_st&)__attribute__((always_inline));
  inline friend int operator==(const unsigned long long&, const float_st&)__attribute__((always_inline));
  inline friend int operator==(const long long&, const float_st&)__attribute__((always_inline));
  inline friend int operator==(const unsigned long&, const float_st&)__attribute__((always_inline));
  inline friend int operator==(const long&, const float_st&)__attribute__((always_inline));
  inline friend int operator==(const unsigned int&, const float_st&)__attribute__((always_inline));
  inline friend int operator==(const int&, const float_st&)__attribute__((always_inline));
  inline friend int operator==(const unsigned short&, const float_st&)__attribute__((always_inline));
  inline friend int operator==(const short&, const float_st&)__attribute__((always_inline));
  inline friend int operator==(const unsigned char&, const float_st&)__attribute__((always_inline));
  inline friend int operator==(const char&, const float_st&)__attribute__((always_inline));
#ifdef CADNA_HALF_EMULATION
  inline friend int operator==(const half&, const float_st&)__attribute__((always_inline));
#endif

#ifdef CADNA_HALF_NATIVE
  inline friend int operator==(const half&, const float_st&)__attribute__((always_inline));
#endif

  
#ifdef CADNA_HALF_EMULATION
  inline friend int operator!=(const half_st&, const float_st&)__attribute__((always_inline));  
  inline friend int operator!=(const float_st&, const half_st&)__attribute__((always_inline));
#endif 
#ifdef CADNA_HALF_NATIVE
  inline friend int operator!=(const half_st&, const float_st&)__attribute__((always_inline));  
  inline friend int operator!=(const float_st&, const half_st&)__attribute__((always_inline));
#endif

  inline friend int operator!=(const double_st&, const float_st&)__attribute__((always_inline));  
  inline friend int operator!=(const float_st&, const double_st&)__attribute__((always_inline));  
  inline friend int operator!=(const float_st&, const float_st&)__attribute__((always_inline));  

  inline friend int operator!=(const float_st&, const double&)__attribute__((always_inline));
  inline friend int operator!=(const float_st&, const float&)__attribute__((always_inline));
  inline friend int operator!=(const float_st&, const unsigned long long&)__attribute__((always_inline));
  inline friend int operator!=(const float_st&, const long long&)__attribute__((always_inline));
  inline friend int operator!=(const float_st&, const unsigned long&)__attribute__((always_inline));
  inline friend int operator!=(const float_st&, const long&)__attribute__((always_inline));
  inline friend int operator!=(const float_st&, const unsigned int&)__attribute__((always_inline));
  inline friend int operator!=(const float_st&, const int&)__attribute__((always_inline));
  inline friend int operator!=(const float_st&, const unsigned short&)__attribute__((always_inline));
  inline friend int operator!=(const float_st&, const short&)__attribute__((always_inline));
  inline friend int operator!=(const float_st&, const unsigned char&)__attribute__((always_inline));
  inline friend int operator!=(const float_st&, const char&)__attribute__((always_inline));
#ifdef CADNA_HALF_EMULATION
  inline friend int operator!=(const float_st&, const half&)__attribute__((always_inline));
#endif 
#ifdef CADNA_HALF_NATIVE
  inline friend int operator!=(const float_st&, const half&)__attribute__((always_inline));
#endif 

  inline friend int operator!=(const double&, const float_st&)__attribute__((always_inline));
  inline friend int operator!=(const float&, const float_st&)__attribute__((always_inline));
  inline friend int operator!=(const unsigned long long&, const float_st&)__attribute__((always_inline));
  inline friend int operator!=(const long long&, const float_st&)__attribute__((always_inline));
  inline friend int operator!=(const unsigned long&, const float_st&)__attribute__((always_inline));
  inline friend int operator!=(const long&, const float_st&)__attribute__((always_inline));
  inline friend int operator!=(const unsigned int&, const float_st&)__attribute__((always_inline));
  inline friend int operator!=(const int&, const float_st&)__attribute__((always_inline));
  inline friend int operator!=(const unsigned short&, const float_st&)__attribute__((always_inline));
  inline friend int operator!=(const short&, const float_st&)__attribute__((always_inline));
  inline friend int operator!=(const unsigned char&, const float_st&)__attribute__((always_inline));
  inline friend int operator!=(const char&, const float_st&)__attribute__((always_inline));
#ifdef CADNA_HALF_EMULATION
  inline friend int operator!=(const half&, const float_st&)__attribute__((always_inline));
#endif 
#ifdef CADNA_HALF_NATIVE
  inline friend int operator!=(const half&, const float_st&)__attribute__((always_inline));
#endif 

  inline friend int operator>=(const double_st&, const float_st&)__attribute__((always_inline));  
  inline friend int operator>=(const float_st&, const double_st&)__attribute__((always_inline));  
  inline friend int operator>=(const float_st&, const float_st&)__attribute__((always_inline));  

  inline friend int operator>=(const float_st&, const double&)__attribute__((always_inline));
  inline friend int operator>=(const float_st&, const float&)__attribute__((always_inline));
  inline friend int operator>=(const float_st&, const unsigned long long&)__attribute__((always_inline));
  inline friend int operator>=(const float_st&, const long long&)__attribute__((always_inline));
  inline friend int operator>=(const float_st&, const unsigned long&)__attribute__((always_inline));
  inline friend int operator>=(const float_st&, const long&)__attribute__((always_inline));
  inline friend int operator>=(const float_st&, const unsigned int&)__attribute__((always_inline));
  inline friend int operator>=(const float_st&, const int&)__attribute__((always_inline));
  inline friend int operator>=(const float_st&, const unsigned short&)__attribute__((always_inline));
  inline friend int operator>=(const float_st&, const short&)__attribute__((always_inline));
  inline friend int operator>=(const float_st&, const unsigned char&)__attribute__((always_inline));
  inline friend int operator>=(const float_st&, const char&)__attribute__((always_inline));
#ifdef CADNA_HALF_EMULATION
  inline friend int operator>=(const float_st&, const half&)__attribute__((always_inline));  
  inline friend int operator>=(const half_st&, const float_st&)__attribute__((always_inline));  
  inline friend int operator>=(const float_st&, const half_st&)__attribute__((always_inline));
#endif

#ifdef CADNA_HALF_NATIVE
  inline friend int operator>=(const float_st&, const half&)__attribute__((always_inline));  
  inline friend int operator>=(const half_st&, const float_st&)__attribute__((always_inline));  
  inline friend int operator>=(const float_st&, const half_st&)__attribute__((always_inline));
#endif

  inline friend int operator>=(const double&, const float_st&)__attribute__((always_inline));
  inline friend int operator>=(const float&, const float_st&)__attribute__((always_inline));
  inline friend int operator>=(const unsigned long long&, const float_st&)__attribute__((always_inline));
  inline friend int operator>=(const long long&, const float_st&)__attribute__((always_inline));
  inline friend int operator>=(const unsigned long&, const float_st&)__attribute__((always_inline));
  inline friend int operator>=(const long&, const float_st&)__attribute__((always_inline));
  inline friend int operator>=(const unsigned int&, const float_st&)__attribute__((always_inline));
  inline friend int operator>=(const int&, const float_st&)__attribute__((always_inline));
  inline friend int operator>=(const unsigned short&, const float_st&)__attribute__((always_inline));
  inline friend int operator>=(const short&, const float_st&)__attribute__((always_inline));
  inline friend int operator>=(const unsigned char&, const float_st&)__attribute__((always_inline));
  inline friend int operator>=(const char&, const float_st&)__attribute__((always_inline));
#ifdef CADNA_HALF_EMULATION
  inline friend int operator>=(const half&, const float_st&)__attribute__((always_inline));
#endif

#ifdef CADNA_HALF_NATIVE
  inline friend int operator>=(const half&, const float_st&)__attribute__((always_inline));
#endif

  inline friend int operator>(const double_st&, const float_st&)__attribute__((always_inline));  
  inline friend int operator>(const float_st&, const double_st&)__attribute__((always_inline));  
  inline friend int operator>(const float_st&, const float_st&)__attribute__((always_inline));
  
  inline friend int operator>(const float_st&, const double&)__attribute__((always_inline));
  inline friend int operator>(const float_st&, const float&)__attribute__((always_inline));
  inline friend int operator>(const float_st&, const unsigned long long&)__attribute__((always_inline));
  inline friend int operator>(const float_st&, const long long&)__attribute__((always_inline));
  inline friend int operator>(const float_st&, const unsigned long&)__attribute__((always_inline));
  inline friend int operator>(const float_st&, const long&)__attribute__((always_inline));
  inline friend int operator>(const float_st&, const unsigned int&)__attribute__((always_inline));
  inline friend int operator>(const float_st&, const int&)__attribute__((always_inline));
  inline friend int operator>(const float_st&, const unsigned short&)__attribute__((always_inline));
  inline friend int operator>(const float_st&, const short&)__attribute__((always_inline));
  inline friend int operator>(const float_st&, const unsigned char&)__attribute__((always_inline));
  inline friend int operator>(const float_st&, const char&)__attribute__((always_inline));
#ifdef CADNA_HALF_EMULATION
  inline friend int operator>(const float_st&, const half&)__attribute__((always_inline));  
  inline friend int operator>(const half_st&, const float_st&)__attribute__((always_inline));  
  inline friend int operator>(const float_st&, const half_st&)__attribute__((always_inline));
#endif
#ifdef CADNA_HALF_NATIVE
  inline friend int operator>(const float_st&, const half&)__attribute__((always_inline));  
  inline friend int operator>(const half_st&, const float_st&)__attribute__((always_inline));  
  inline friend int operator>(const float_st&, const half_st&)__attribute__((always_inline));
#endif

  inline friend int operator>(const double&, const float_st&)__attribute__((always_inline));
  inline friend int operator>(const float&, const float_st&)__attribute__((always_inline));
  inline friend int operator>(const unsigned long long&, const float_st&)__attribute__((always_inline));
  inline friend int operator>(const long long&, const float_st&)__attribute__((always_inline));
  inline friend int operator>(const unsigned long&, const float_st&)__attribute__((always_inline));
  inline friend int operator>(const long&, const float_st&)__attribute__((always_inline));
  inline friend int operator>(const unsigned int&, const float_st&)__attribute__((always_inline));
  inline friend int operator>(const int&, const float_st&)__attribute__((always_inline));
  inline friend int operator>(const unsigned short&, const float_st&)__attribute__((always_inline));
  inline friend int operator>(const short&, const float_st&)__attribute__((always_inline));
  inline friend int operator>(const unsigned char&, const float_st&)__attribute__((always_inline));
  inline friend int operator>(const char&, const float_st&)__attribute__((always_inline));
#ifdef CADNA_HALF_EMULATION
  inline friend int operator>(const half&, const float_st&)__attribute__((always_inline));
#endif
#ifdef CADNA_HALF_NATIVE
  inline friend int operator>(const half&, const float_st&)__attribute__((always_inline));
#endif


  inline friend int operator<=(const double_st&, const float_st&)__attribute__((always_inline));  
  inline friend int operator<=(const float_st&, const double_st&)__attribute__((always_inline));  
  inline friend int operator<=(const float_st&, const float_st&)__attribute__((always_inline));  

  inline friend int operator<=(const float_st&, const double&)__attribute__((always_inline));
  inline friend int operator<=(const float_st&, const float&)__attribute__((always_inline));
  inline friend int operator<=(const float_st&, const unsigned long long&)__attribute__((always_inline));
  inline friend int operator<=(const float_st&, const long long&)__attribute__((always_inline));
  inline friend int operator<=(const float_st&, const unsigned long&)__attribute__((always_inline));
  inline friend int operator<=(const float_st&, const long&)__attribute__((always_inline));
  inline friend int operator<=(const float_st&, const unsigned int&)__attribute__((always_inline));
  inline friend int operator<=(const float_st&, const int&)__attribute__((always_inline));
  inline friend int operator<=(const float_st&, const unsigned short&)__attribute__((always_inline));
  inline friend int operator<=(const float_st&, const short&)__attribute__((always_inline));
  inline friend int operator<=(const float_st&, const unsigned char&)__attribute__((always_inline));
  inline friend int operator<=(const float_st&, const char&)__attribute__((always_inline));
#ifdef CADNA_HALF_EMULATION
  inline friend int operator<=(const float_st&, const half&)__attribute__((always_inline));  
  inline friend int operator<=(const half_st&, const float_st&)__attribute__((always_inline));  
  inline friend int operator<=(const float_st&, const half_st&)__attribute__((always_inline));
#endif

#ifdef CADNA_HALF_NATIVE
  inline friend int operator<=(const float_st&, const half&)__attribute__((always_inline));  
  inline friend int operator<=(const half_st&, const float_st&)__attribute__((always_inline));  
  inline friend int operator<=(const float_st&, const half_st&)__attribute__((always_inline));
#endif

  inline friend int operator<=(const double&, const float_st&)__attribute__((always_inline));
  inline friend int operator<=(const float&, const float_st&)__attribute__((always_inline));
  inline friend int operator<=(const unsigned long long&, const float_st&)__attribute__((always_inline));
  inline friend int operator<=(const long long&, const float_st&)__attribute__((always_inline));
  inline friend int operator<=(const unsigned long&, const float_st&)__attribute__((always_inline));
  inline friend int operator<=(const long&, const float_st&)__attribute__((always_inline));
  inline friend int operator<=(const unsigned int&, const float_st&)__attribute__((always_inline));
  inline friend int operator<=(const int&, const float_st&)__attribute__((always_inline));
  inline friend int operator<=(const unsigned short&, const float_st&)__attribute__((always_inline));
  inline friend int operator<=(const short&, const float_st&)__attribute__((always_inline));
  inline friend int operator<=(const unsigned char&, const float_st&)__attribute__((always_inline));
  inline friend int operator<=(const char&, const float_st&)__attribute__((always_inline));
#ifdef CADNA_HALF_EMULATION
  inline friend int operator<=(const half&, const float_st&)__attribute__((always_inline));
#endif
#ifdef CADNA_HALF_NATIVE
  inline friend int operator<=(const half&, const float_st&)__attribute__((always_inline));
#endif

  inline friend int operator<(const double_st&, const float_st&)__attribute__((always_inline));  
  inline friend int operator<(const float_st&, const double_st&)__attribute__((always_inline));  
  inline friend int operator<(const float_st&, const float_st&)__attribute__((always_inline));
  
  inline friend int operator<(const float_st&, const double&)__attribute__((always_inline));
  inline friend int operator<(const float_st&, const float&)__attribute__((always_inline));
  inline friend int operator<(const float_st&, const unsigned long long&)__attribute__((always_inline));
  inline friend int operator<(const float_st&, const long long&)__attribute__((always_inline));
  inline friend int operator<(const float_st&, const unsigned long&)__attribute__((always_inline));
  inline friend int operator<(const float_st&, const long&)__attribute__((always_inline));
  inline friend int operator<(const float_st&, const unsigned int&)__attribute__((always_inline));
  inline friend int operator<(const float_st&, const int&)__attribute__((always_inline));
  inline friend int operator<(const float_st&, const unsigned short&)__attribute__((always_inline));
  inline friend int operator<(const float_st&, const short&)__attribute__((always_inline));
  inline friend int operator<(const float_st&, const unsigned char&)__attribute__((always_inline));
  inline friend int operator<(const float_st&, const char&)__attribute__((always_inline));
#ifdef CADNA_HALF_EMULATION
  inline friend int operator<(const float_st&, const half&)__attribute__((always_inline));  
  inline friend int operator<(const half_st&, const float_st&)__attribute__((always_inline));  
  inline friend int operator<(const float_st&, const half_st&)__attribute__((always_inline));
#endif
#ifdef CADNA_HALF_NATIVE
  inline friend int operator<(const float_st&, const half&)__attribute__((always_inline));  
  inline friend int operator<(const half_st&, const float_st&)__attribute__((always_inline));  
  inline friend int operator<(const float_st&, const half_st&)__attribute__((always_inline));
#endif

  inline friend int operator<(const double&, const float_st&)__attribute__((always_inline));
  inline friend int operator<(const float&, const float_st&)__attribute__((always_inline));
  inline friend int operator<(const unsigned long long&, const float_st&)__attribute__((always_inline));
  inline friend int operator<(const long long&, const float_st&)__attribute__((always_inline));
  inline friend int operator<(const unsigned long&, const float_st&)__attribute__((always_inline));
  inline friend int operator<(const long&, const float_st&)__attribute__((always_inline));
  inline friend int operator<(const unsigned int&, const float_st&)__attribute__((always_inline));
  inline friend int operator<(const int&, const float_st&)__attribute__((always_inline));
  inline friend int operator<(const unsigned short&, const float_st&)__attribute__((always_inline));
  inline friend int operator<(const short&, const float_st&)__attribute__((always_inline));
  inline friend int operator<(const unsigned char&, const float_st&)__attribute__((always_inline));
  inline friend int operator<(const char&, const float_st&)__attribute__((always_inline));
#ifdef CADNA_HALF_EMULATION
  inline friend int operator<(const half&, const float_st&)__attribute__((always_inline));
#endif
#ifdef CADNA_HALF_NATIVE
  inline friend int operator<(const half&, const float_st&)__attribute__((always_inline));
#endif

  // MATHEMATICAL FUNCTIONS 

  friend float_st  logf(const float_st&);
  friend float_st  log2f(const float_st&);
  friend float_st  log10f(const float_st&);
  friend float_st  log1pf(const float_st&);
  friend float_st  logbf(const float_st&);
  friend float_st  expf(const float_st&); 
  friend float_st  exp2f(const float_st&); 
  friend float_st  expm1f(const float_st&); 
  friend float_st  sqrtf(const float_st&);
  friend float_st  cbrtf(const float_st&);
  friend float_st  sinf(const float_st&);
  friend float_st  cosf(const float_st&);
  friend float_st  tanf(const float_st&);
  friend float_st  asinf(const float_st&);
  friend float_st  acosf(const float_st&);
  friend float_st  atanf(const float_st&);
  friend double_st atan2f(const double_st&, const float_st&);
  friend double_st atan2f(const float_st&, const double_st&);
  friend float_st  atan2f(const float_st&, const float_st&);
  friend float_st  sinhf(const float_st&);
  friend float_st  coshf(const float_st&);
  friend float_st  tanhf(const float_st&);
  friend float_st  asinhf(const float_st&);
  friend float_st  acoshf(const float_st&);
  friend float_st  atanhf(const float_st&);
  friend double_st  hypot(const double_st&, const float_st&);
  friend double_st  hypot(const float_st&, const double_st&);  
  friend float_st  hypotf(const float_st&, const float_st&);


  
  friend float_st  log(const float_st&);
  friend float_st  log2(const float_st&);
  friend float_st  log10(const float_st&);
  friend float_st  log1p(const float_st&);
  friend float_st  logb(const float_st&);
  friend float_st  exp(const float_st&); 
  friend float_st  exp2(const float_st&); 
  friend float_st  expm1(const float_st&); 
  friend float_st  sqrt(const float_st&);
  friend float_st  cbrt(const float_st&);
  friend float_st  sin(const float_st&);
  friend float_st  cos(const float_st&);
  friend float_st  tan(const float_st&);
  friend float_st  asin(const float_st&);
  friend float_st  acos(const float_st&);
  friend float_st  atan(const float_st&);
  friend double_st atan2(const double_st&, const float_st&);
  friend double_st atan2(const float_st&, const double_st&);
  friend float_st  atan2(const float_st&, const float_st&);
  friend float_st  sinh(const float_st&);
  friend float_st  cosh(const float_st&);
  friend float_st  tanh(const float_st&);
  friend float_st  asinh(const float_st&);
  friend float_st  acosh(const float_st&);
  friend float_st  atanh(const float_st&);  
  friend float_st  hypot(const float_st&, const float_st&);



  friend double_st pow(const double_st&, const float_st&);
  friend double_st pow(const float_st&, const double_st&);
  friend float_st powf(const float_st&, const float_st&);
  friend float_st pow(const float_st&, const float_st&);

  
  friend float_st powf(const float_st&, const double&);
  friend float_st powf(const float_st&, const float&);
  friend float_st powf(const float_st&, const unsigned long long&);
  friend float_st powf(const float_st&, const long long&);
  friend float_st powf(const float_st&, const unsigned long&);
  friend float_st powf(const float_st&, const long&);
  friend float_st powf(const float_st&, const unsigned int&);
  friend float_st powf(const float_st&, const int&);
  friend float_st powf(const float_st&, const unsigned short&);
  friend float_st powf(const float_st&, const short&);
  friend float_st powf(const float_st&, const unsigned char&);
  friend float_st powf(const float_st&, const char&);
#ifdef CADNA_HALF_EMULATION
  friend float_st powf(const float_st&, const half&);
  friend half_st pow(const half_st&, const float_st&);  
  friend half_st pow(const float_st&, const half_st&);
#endif
#ifdef CADNA_HALF_NATIVE
  friend float_st powf(const float_st&, const half&);
  friend half_st pow(const half_st&, const float_st&); 
  friend half_st pow(const float_st&, const half_st&);
#endif
  
  friend float_st pow(const float_st&, const double&);
  friend float_st pow(const float_st&, const float&);
  friend float_st pow(const float_st&, const unsigned long long&);
  friend float_st pow(const float_st&, const long long&);
  friend float_st pow(const float_st&, const unsigned long&);
  friend float_st pow(const float_st&, const long&);
  friend float_st pow(const float_st&, const unsigned int&);
  friend float_st pow(const float_st&, const int&);
  friend float_st pow(const float_st&, const unsigned short&);
  friend float_st pow(const float_st&, const short&);
  friend float_st pow(const float_st&, const unsigned char&);
  friend float_st pow(const float_st&, const char&);
#ifdef CADNA_HALF_EMULATION
  friend float_st pow(const float_st&, const half&);
#endif
#ifdef CADNA_HALF_NATIVE
  friend float_st pow(const float_st&, const half&);
#endif
  friend float_st powf(const double&, const float_st&);
  friend float_st powf(const float&, const float_st&);
  friend float_st powf(const unsigned long long&, const float_st&);
  friend float_st powf(const long long&, const float_st&);
  friend float_st powf(const unsigned long&, const float_st&);
  friend float_st powf(const long&, const float_st&);
  friend float_st powf(const unsigned int&, const float_st&);
  friend float_st powf(const int&, const float_st&);
  friend float_st powf(const unsigned short&, const float_st&);
  friend float_st powf(const short&, const float_st&);
  friend float_st powf(const unsigned char&, const float_st&);
  friend float_st powf(const char&, const float_st&);
#ifdef CADNA_HALF_EMULATION
  friend float_st powf(const float_st&, const half&);
#endif
#ifdef CADNA_HALF_NATIVE
  friend float_st powf(const float_st&, const half&);
#endif  
  friend float_st pow(const double&, const float_st&);
  friend float_st pow(const float&, const float_st&);
  friend float_st pow(const unsigned long long&, const float_st&);
  friend float_st pow(const long long&, const float_st&);
  friend float_st pow(const unsigned long&, const float_st&);
  friend float_st pow(const long&, const float_st&);
  friend float_st pow(const unsigned int&, const float_st&);
  friend float_st pow(const int&, const float_st&);
  friend float_st pow(const unsigned short&, const float_st&);
  friend float_st pow(const short&, const float_st&);
  friend float_st pow(const unsigned char&, const float_st&);
  friend float_st pow(const char&, const float_st&);
#ifdef CADNA_HALF_EMULATION
  friend float_st pow(const float_st&, const half&);
#endif
#ifdef CADNA_HALF_NATIVE
  friend float_st pow(const float_st&, const half&);
#endif
  friend float_st fmaxf(const float_st&, const float_st&);  
  friend double_st fmax(const double_st&, const float_st&);  
  friend double_st fmax(const float_st&, const double_st&);  
  friend float_st fmax(const float_st&, const float_st&);  
  friend float_st fmaxf(const float_st&, const double&);
  friend float_st fmaxf(const float_st&, const float&);
  friend float_st fmaxf(const float_st&, const unsigned long long&);
  friend float_st fmaxf(const float_st&, const long long&);
  friend float_st fmaxf(const float_st&, const unsigned long&);
  friend float_st fmaxf(const float_st&, const long&);
  friend float_st fmaxf(const float_st&, const unsigned int&);
  friend float_st fmaxf(const float_st&, const int&);
  friend float_st fmaxf(const float_st&, const unsigned short&);
  friend float_st fmaxf(const float_st&, const short&);
  friend float_st fmaxf(const float_st&, const unsigned char&);
  friend float_st fmaxf(const float_st&, const char&);


  
  friend float_st fmax(const float_st&, const double&);
  friend float_st fmax(const float_st&, const float&);
  friend float_st fmax(const float_st&, const unsigned long long&);
  friend float_st fmax(const float_st&, const long long&);
  friend float_st fmax(const float_st&, const unsigned long&);
  friend float_st fmax(const float_st&, const long&);
  friend float_st fmax(const float_st&, const unsigned int&);
  friend float_st fmax(const float_st&, const int&);
  friend float_st fmax(const float_st&, const unsigned short&);
  friend float_st fmax(const float_st&, const short&);
  friend float_st fmax(const float_st&, const unsigned char&);
  friend float_st fmax(const float_st&, const char&);


  friend float_st fmaxf(const double&, const float_st&);
  friend float_st fmaxf(const float&, const float_st&);
  friend float_st fmaxf(const unsigned long long&, const float_st&);
  friend float_st fmaxf(const long long&, const float_st&);
  friend float_st fmaxf(const unsigned long&, const float_st&);
  friend float_st fmaxf(const long&, const float_st&);
  friend float_st fmaxf(const unsigned int&, const float_st&);
  friend float_st fmaxf(const int&, const float_st&);
  friend float_st fmaxf(const unsigned short&, const float_st&);
  friend float_st fmaxf(const short&, const float_st&);
  friend float_st fmaxf(const unsigned char&, const float_st&);
  friend float_st fmaxf(const char&, const float_st&);

  
  friend float_st fmax(const double&, const float_st&);
  friend float_st fmax(const float&, const float_st&);
  friend float_st fmax(const unsigned long long&, const float_st&);
  friend float_st fmax(const long long&, const float_st&);
  friend float_st fmax(const unsigned long&, const float_st&);
  friend float_st fmax(const long&, const float_st&);
  friend float_st fmax(const unsigned int&, const float_st&);
  friend float_st fmax(const int&, const float_st&);
  friend float_st fmax(const unsigned short&, const float_st&);
  friend float_st fmax(const short&, const float_st&);
  friend float_st fmax(const unsigned char&, const float_st&);
  friend float_st fmax(const char&, const float_st&);


  friend float_st fminf(const float_st&, const float_st&);  
  friend float_st fmin(const float_st&, const float_st&);  
  friend double_st fmin(const double_st&, const float_st&);  
  friend double_st fmin(const float_st&, const double_st&);  

  friend float_st fminf(const float_st&, const double&);
  friend float_st fminf(const float_st&, const float&);
  friend float_st fminf(const float_st&, const unsigned long long&);
  friend float_st fminf(const float_st&, const long long&);
  friend float_st fminf(const float_st&, const unsigned long&);
  friend float_st fminf(const float_st&, const long&);
  friend float_st fminf(const float_st&, const unsigned int&);
  friend float_st fminf(const float_st&, const int&);
  friend float_st fminf(const float_st&, const unsigned short&);
  friend float_st fminf(const float_st&, const short&);
  friend float_st fminf(const float_st&, const unsigned char&);
  friend float_st fminf(const float_st&, const char&);

  
  friend float_st fmin(const float_st&, const double&);
  friend float_st fmin(const float_st&, const float&);
  friend float_st fmin(const float_st&, const unsigned long long&);
  friend float_st fmin(const float_st&, const long long&);
  friend float_st fmin(const float_st&, const unsigned long&);
  friend float_st fmin(const float_st&, const long&);
  friend float_st fmin(const float_st&, const unsigned int&);
  friend float_st fmin(const float_st&, const int&);
  friend float_st fmin(const float_st&, const unsigned short&);
  friend float_st fmin(const float_st&, const short&);
  friend float_st fmin(const float_st&, const unsigned char&);
  friend float_st fmin(const float_st&, const char&);

  friend float_st fminf(const double&, const float_st&);
  friend float_st fminf(const float&, const float_st&);
  friend float_st fminf(const unsigned long long&, const float_st&);
  friend float_st fminf(const long long&, const float_st&);
  friend float_st fminf(const unsigned long&, const float_st&);
  friend float_st fminf(const long&, const float_st&);
  friend float_st fminf(const unsigned int&, const float_st&);
  friend float_st fminf(const int&, const float_st&);
  friend float_st fminf(const unsigned short&, const float_st&);
  friend float_st fminf(const short&, const float_st&);
  friend float_st fminf(const unsigned char&, const float_st&);
  friend float_st fminf(const char&, const float_st&);

  
  friend float_st fmin(const double&, const float_st&);
  friend float_st fmin(const float&, const float_st&);
  friend float_st fmin(const unsigned long long&, const float_st&);
  friend float_st fmin(const long long&, const float_st&);
  friend float_st fmin(const unsigned long&, const float_st&);
  friend float_st fmin(const long&, const float_st&);
  friend float_st fmin(const unsigned int&, const float_st&);
  friend float_st fmin(const int&, const float_st&);
  friend float_st fmin(const unsigned short&, const float_st&);
  friend float_st fmin(const short&, const float_st&);
  friend float_st fmin(const unsigned char&, const float_st&);
  friend float_st fmin(const char&, const float_st&);

  friend float_st  erf(const float_st&);
  friend float_st  erfc(const float_st&);
  // INTRINSIC FUNCTIONS
 
  friend float_st fabsf(const float_st&);
  friend float_st floorf(const float_st&);
  friend float_st ceilf(const float_st&);
  friend float_st truncf(const float_st&);
  friend float_st nearbyintf(const float_st&);
  friend float_st rintf(const float_st&);
  friend long int  lrintf(const float_st&);
  friend long long int  llrintf(const float_st&);
  friend float_st frexp(const float_st&, int*);
  friend float_st ldexp(const float_st&, const int);
  friend float_st fmod(const float_st&, const float_st&);
  friend float_st frexpf(const float_st&, int*);
  friend float_st modff(const  float_st&,  float_st*);
  friend float_st ldexpf(const float_st&, const int);
  friend float_st fmodf(const float_st&, const float_st&);
  friend float_st fabs(const float_st&);
  friend float_st abs(const float_st&);
  friend float_st floor(const float_st&);
  friend float_st ceil(const float_st&);
  friend float_st trunc(const float_st&);
  friend float_st nearbyint(const float_st&);
  friend float_st rint(const float_st&);
  friend long int  lrint(const float_st&);
  friend long long int  llrint(const float_st&);
  friend int finite(const float_st&);
  friend int isfinitef(const float_st&);
  friend int isnan(const float_st&);
  friend int isinf(const float_st&);


  // Conversion
  operator char();  
  operator unsigned char();  
  operator short();  
  operator unsigned short();  
  operator int();  
  operator unsigned();  
  operator long();  
  operator unsigned long();  
  operator long long();  
  operator unsigned long long();  
  operator float();
  operator double();
#ifdef CADNA_HALF_EMULATION
  operator half();
  operator half_st();
#endif // CADNA_HALF_EMULATION
#ifdef CADNA_HALF_NATIVE
  operator half();
  operator half_st();
#endif
  operator double_st();
  
  operator char() const;  
  operator unsigned char() const;  
  operator short() const;  
  operator unsigned short() const;  
  operator int() const;  
  operator unsigned() const;  
  operator long() const;  
  operator unsigned long() const;  
  operator long long() const;  
  operator unsigned long long() const;  
  operator float() const;
  operator double() const;
#ifdef CADNA_HALF_EMULATION
  operator half_st() const;
  operator half() const;
#endif // CADNA_HALF_EMULATION

#ifdef CADNA_HALF_NATIVE
  operator half_st() const;
  operator half() const;
#endif 
  operator double_st() const;



  
  inline int nb_significant_digit() const __attribute__((always_inline));  
  inline int approx_digit() const __attribute__((always_inline));
  inline int computedzero() const __attribute__((always_inline));
  inline int numericalnoise() const __attribute__((always_inline));
  void display() const ; 
  void display(const char *) const ; 
  char* str( char *)  const ;
  
  friend char* strp(const float_st&);
  friend char* str( char *, const float_st&);
  
  friend std::istream& operator >>(std::istream& s, float_st& );  
  
  void data_st();
  void data_st(const double&, const int &);
  
};

std::ostream& operator<<(std::ostream&, const float_st&);
std::istream& operator >>(std::istream&, float_st& );

#ifdef CADNA_HALF_EMULATION
class half_st 
{
  friend class float_st;
  friend class double_st;



  typedef half TYPEBASE;

public:
  half x,y,z;
  mutable int16_t accuracy;
  inline half_st(void) :x(0),y(0),z(0),accuracy(3) {};
  inline half_st(half a, half b, half c):x((half)a),y((half)b),z((half)c),accuracy(-1) {};
  inline half_st(const half &a): x((half)a), y((half)a), z((half)a),accuracy(3) {};
  inline half_st(const double &a): x(a), y(a), z(a),accuracy(3) {};
  inline half_st(const double_st&a):x(a.x), y(a.y), z(a.z), accuracy(a.accuracy>=4?3:a.accuracy){};
  inline half_st(const float_st&a):x(a.x), y(a.y), z(a.z), accuracy(a.accuracy>=4?3:a.accuracy){};


  half getx() {return(this->x);}
  half gety() {return(this->y);}
  half getz() {return(this->z);}
  int getaccuracy() {return(this->accuracy);}
 // ASSIGNMENT
  
  inline half_st& operator=(const double&)__attribute__((always_inline));
  inline half_st& operator=(const float&)__attribute__((always_inline));
  inline half_st& operator=(const unsigned long long&)__attribute__((always_inline));
  inline half_st& operator=(const long long&)__attribute__((always_inline));
  inline half_st& operator=(const unsigned long&)__attribute__((always_inline));
  inline half_st& operator=(const long&)__attribute__((always_inline));
  inline half_st& operator=(const unsigned int&)__attribute__((always_inline));
  inline half_st& operator=(const int&)__attribute__((always_inline));
  inline half_st& operator=(const unsigned short&)__attribute__((always_inline));
  inline half_st& operator=(const short&)__attribute__((always_inline));
  inline half_st& operator=(const unsigned char&)__attribute__((always_inline));
  inline half_st& operator=(const char&)__attribute__((always_inline));
  inline half_st& operator=(const half&)__attribute__((always_inline));    
  inline half_st& operator=(const float_st&)__attribute__((always_inline));
  inline half_st& operator=(const double_st&)__attribute__((always_inline));

// ADDITION
  inline half_st operator++()__attribute__((always_inline));
  inline half_st operator++(int)__attribute__((always_inline));
  inline half_st operator+() const __attribute__((always_inline));
 
  inline half_st& operator+=(const half_st&)__attribute__((always_inline));
  inline half_st& operator+=(const double_st&)__attribute__((always_inline));
  inline half_st& operator+=(const float_st&)__attribute__((always_inline));
  inline half_st& operator+=(const double&)__attribute__((always_inline));
  inline half_st& operator+=(const float&)__attribute__((always_inline));
  inline half_st& operator+=(const unsigned long long&)__attribute__((always_inline));
  inline half_st& operator+=(const long long&)__attribute__((always_inline));
  inline half_st& operator+=(const unsigned long&)__attribute__((always_inline));
  inline half_st& operator+=(const long&)__attribute__((always_inline));
  inline half_st& operator+=(const unsigned int&)__attribute__((always_inline));
  inline half_st& operator+=(const int&)__attribute__((always_inline));
  inline half_st& operator+=(const unsigned short&)__attribute__((always_inline));
  inline half_st& operator+=(const short&)__attribute__((always_inline));
  inline half_st& operator+=(const unsigned char&)__attribute__((always_inline));
  inline half_st& operator+=(const char&)__attribute__((always_inline));
  inline half_st& operator+=(const half& a)__attribute__((always_inline));

  inline friend half_st operator+(const half_st&, const half_st&)__attribute__((always_inline)); 
  inline friend float_st operator+(const half_st&, const float_st&)__attribute__((always_inline));
  inline friend float_st operator+(const float_st&,  const half_st&)__attribute__((always_inline));
  inline friend double_st operator+(const half_st&, const double_st&)__attribute__((always_inline));
  inline friend double_st operator+(const double_st&,  const half_st&)__attribute__((always_inline));


  inline friend half_st operator+(const half_st&, const double&)__attribute__((always_inline));
  inline friend half_st operator+(const half_st&, const float&)__attribute__((always_inline));
  inline friend half_st operator+(const half_st&, const unsigned long long&)__attribute__((always_inline));
  inline friend half_st operator+(const half_st&, const long long&)__attribute__((always_inline));
  inline friend half_st operator+(const half_st&, const unsigned long&)__attribute__((always_inline));
  inline friend half_st operator+(const half_st&, const long&)__attribute__((always_inline));
  inline friend half_st operator+(const half_st&, const unsigned int&)__attribute__((always_inline));
  inline friend half_st operator+(const half_st&, const int&)__attribute__((always_inline));
  inline friend half_st operator+(const half_st&, const unsigned short&)__attribute__((always_inline));
  inline friend half_st operator+(const half_st&, const short&)__attribute__((always_inline));
  inline friend half_st operator+(const half_st&, const unsigned char&)__attribute__((always_inline));
  inline friend half_st operator+(const half_st&, const char&)__attribute__((always_inline));

  inline friend half_st operator+(const double&, const half_st&)__attribute__((always_inline));
  inline friend half_st operator+(const float&, const half_st&)__attribute__((always_inline));

  inline friend half_st operator+(const unsigned long long&, const half_st&)__attribute__((always_inline));
  inline friend half_st operator+(const long long&, const half_st&)__attribute__((always_inline));
  inline friend half_st operator+(const unsigned long&, const half_st&)__attribute__((always_inline));
  inline friend half_st operator+(const long&, const half_st&)__attribute__((always_inline));
  inline friend half_st operator+(const unsigned int&, const half_st&)__attribute__((always_inline));
  inline friend half_st operator+(const int&, const half_st&)__attribute__((always_inline));
  inline friend half_st operator+(const unsigned short&, const half_st&)__attribute__((always_inline));
  inline friend half_st operator+(const short&, const half_st&)__attribute__((always_inline));
  inline friend half_st operator+(const unsigned char&, const half_st&)__attribute__((always_inline));
  inline friend half_st operator+(const char&, const half_st&)__attribute__((always_inline));

  inline friend half_st operator+(const half&, const half_st&)__attribute__((always_inline));
  inline friend half_st operator+(const half_st&, const half&)__attribute__((always_inline));

// SUBTRACTION

  inline half_st operator--()__attribute__((always_inline));
  inline half_st operator--(int)__attribute__((always_inline));
  inline half_st operator-() const __attribute__((always_inline)); 

  inline half_st& operator-=(const half_st&)__attribute__((always_inline));
  inline half_st& operator-=(const double_st&)__attribute__((always_inline));
  inline half_st& operator-=(const float_st&)__attribute__((always_inline));

  inline half_st& operator-=(const double&)__attribute__((always_inline));
  inline half_st& operator-=(const float&)__attribute__((always_inline));
  inline half_st& operator-=(const unsigned long long&)__attribute__((always_inline));
  inline half_st& operator-=(const long long&)__attribute__((always_inline));
  inline half_st& operator-=(const unsigned long&)__attribute__((always_inline));
  inline half_st& operator-=(const long&)__attribute__((always_inline));
  inline half_st& operator-=(const unsigned int&)__attribute__((always_inline));
  inline half_st& operator-=(const int&)__attribute__((always_inline));
  inline half_st& operator-=(const unsigned short&)__attribute__((always_inline));
  inline half_st& operator-=(const short&)__attribute__((always_inline));
  inline half_st& operator-=(const unsigned char&)__attribute__((always_inline));
  inline half_st& operator-=(const char&)__attribute__((always_inline));
  inline half_st& operator-=(const half&)__attribute__((always_inline));

  inline friend half_st operator-(const half_st&, const half_st&)__attribute__((always_inline));  

  inline friend half_st operator-(const half_st&, const double&)__attribute__((always_inline));
  inline friend half_st operator-(const half_st&, const float&)__attribute__((always_inline));

  inline friend half_st operator-(const half_st&, const unsigned long long&)__attribute__((always_inline));
  inline friend half_st operator-(const half_st&, const long long&)__attribute__((always_inline));
  inline friend half_st operator-(const half_st&, const unsigned long&)__attribute__((always_inline));
  inline friend half_st operator-(const half_st&, const long&)__attribute__((always_inline));
  inline friend half_st operator-(const half_st&, const unsigned int&)__attribute__((always_inline));
  inline friend half_st operator-(const half_st&, const int&)__attribute__((always_inline));
  inline friend half_st operator-(const half_st&, const unsigned short&)__attribute__((always_inline));
  inline friend half_st operator-(const half_st&, const short&)__attribute__((always_inline));
  inline friend half_st operator-(const half_st&, const unsigned char&)__attribute__((always_inline));
  inline friend half_st operator-(const half_st&, const char&)__attribute__((always_inline));
  inline friend half_st operator-(const half_st&, const half_st&)__attribute__((always_inline));
  inline friend float_st operator-(const float_st&, const half_st&)__attribute__((always_inline));  
  inline friend float_st operator-(const half_st&, const float_st&)__attribute__((always_inline));  
  inline friend double_st operator-(const double_st&, const half_st&)__attribute__((always_inline));  
  inline friend double_st operator-(const half_st&, const double_st&)__attribute__((always_inline));   
 


  inline friend half_st operator-(const double&, const half_st&)__attribute__((always_inline));
  inline friend half_st operator-(const float&, const half_st&)__attribute__((always_inline));
  inline friend half_st operator-(const unsigned long long&, const half_st&)__attribute__((always_inline));
  inline friend half_st operator-(const long long&, const half_st&)__attribute__((always_inline));
  inline friend half_st operator-(const unsigned long&, const half_st&)__attribute__((always_inline));
  inline friend half_st operator-(const long&, const half_st&)__attribute__((always_inline));
  inline friend half_st operator-(const unsigned int&, const half_st&)__attribute__((always_inline));
  inline friend half_st operator-(const int&, const half_st&)__attribute__((always_inline));
  inline friend half_st operator-(const unsigned short&, const half_st&)__attribute__((always_inline));
  inline friend half_st operator-(const short&, const half_st&)__attribute__((always_inline));
  inline friend half_st operator-(const unsigned char&, const half_st&)__attribute__((always_inline));
  inline friend half_st operator-(const char&, const half_st&)__attribute__((always_inline));
  inline friend half_st operator-(const half&, const half_st&)__attribute__((always_inline));
  
  // MULTIPLICATION
  inline half_st& operator*=(const half_st&)__attribute__((always_inline)); 
  inline half_st& operator*=(const double_st&)__attribute__((always_inline));
  inline half_st& operator*=(const float_st&)__attribute__((always_inline));
  inline half_st& operator*=(const double&)__attribute__((always_inline));
  inline half_st& operator*=(const float&)__attribute__((always_inline));
  inline half_st& operator*=(const unsigned long long&)__attribute__((always_inline));
  inline half_st& operator*=(const long long&)__attribute__((always_inline));
  inline half_st& operator*=(const unsigned long&)__attribute__((always_inline));
  inline half_st& operator*=(const long&)__attribute__((always_inline));
  inline half_st& operator*=(const unsigned int&)__attribute__((always_inline));
  inline half_st& operator*=(const int&)__attribute__((always_inline));
  inline half_st& operator*=(const unsigned short&)__attribute__((always_inline));
  inline half_st& operator*=(const short&)__attribute__((always_inline));
  inline half_st& operator*=(const unsigned char&)__attribute__((always_inline));
  inline half_st& operator*=(const char&)__attribute__((always_inline));
  inline half_st& operator*=(const half&)__attribute__((always_inline));

  inline friend double_st operator*(const double_st&, const half_st&)__attribute__((always_inline));  
  inline friend double_st operator*(const half_st&, const double_st&)__attribute__((always_inline));  
  inline friend half_st operator*(const half_st&, const half_st&)__attribute__((always_inline));  
  inline friend float_st operator*(const float_st&, const half_st&)__attribute__((always_inline));  
  inline friend float_st operator*(const half_st&, const float_st&)__attribute__((always_inline));  

  inline friend half_st operator*(const half_st&, const double&)__attribute__((always_inline));
  inline friend half_st operator*(const half_st&, const float&)__attribute__((always_inline));
  inline friend half_st operator*(const half_st&, const unsigned long long&)__attribute__((always_inline));
  inline friend half_st operator*(const half_st&, const long long&)__attribute__((always_inline));
  inline friend half_st operator*(const half_st&, const unsigned long&)__attribute__((always_inline));
  inline friend half_st operator*(const half_st&, const long&)__attribute__((always_inline));
  inline friend half_st operator*(const half_st&, const unsigned int&)__attribute__((always_inline));
  inline friend half_st operator*(const half_st&, const int&)__attribute__((always_inline));
  inline friend half_st operator*(const half_st&, const unsigned short&)__attribute__((always_inline));
  inline friend half_st operator*(const half_st&, const short&)__attribute__((always_inline));
  inline friend half_st operator*(const half_st&, const unsigned char&)__attribute__((always_inline));
  inline friend half_st operator*(const half_st&, const char&)__attribute__((always_inline));
  inline friend half_st operator*(const half_st&, const half&)__attribute__((always_inline));

  inline friend half_st operator*(const double&, const half_st&)__attribute__((always_inline));
  inline friend half_st operator*(const float&, const half_st&)__attribute__((always_inline));
  inline friend half_st operator*(const unsigned long long&, const half_st&)__attribute__((always_inline));
  inline friend half_st operator*(const long long&, const half_st&)__attribute__((always_inline));
  inline friend half_st operator*(const unsigned long&, const half_st&)__attribute__((always_inline));
  inline friend half_st operator*(const long&, const half_st&)__attribute__((always_inline));
  inline friend half_st operator*(const unsigned int&, const half_st&)__attribute__((always_inline));
  inline friend half_st operator*(const int&, const half_st&)__attribute__((always_inline));
  inline friend half_st operator*(const unsigned short&, const half_st&)__attribute__((always_inline));
  inline friend half_st operator*(const short&, const half_st&)__attribute__((always_inline));
  inline friend half_st operator*(const unsigned char&, const half_st&)__attribute__((always_inline));
  inline friend half_st operator*(const char&, const half_st&)__attribute__((always_inline));
  inline friend half_st operator*(const half_st&, const half_st&)__attribute__((always_inline));

// DIVISION
  inline half_st& operator/=(const double_st&)__attribute__((always_inline));
  inline half_st& operator/=(const float_st&)__attribute__((always_inline));
  inline half_st& operator/=(const half_st&)__attribute__((always_inline));
  inline half_st& operator/=(const double&)__attribute__((always_inline));
  inline half_st& operator/=(const float&)__attribute__((always_inline));
  inline half_st& operator/=(const unsigned long long&)__attribute__((always_inline));
  inline half_st& operator/=(const long long&)__attribute__((always_inline));
  inline half_st& operator/=(const unsigned long&)__attribute__((always_inline));
  inline half_st& operator/=(const long&)__attribute__((always_inline));
  inline half_st& operator/=(const unsigned int&)__attribute__((always_inline));
  inline half_st& operator/=(const int&)__attribute__((always_inline));
  inline half_st& operator/=(const unsigned short&)__attribute__((always_inline));
  inline half_st& operator/=(const short&)__attribute__((always_inline));
  inline half_st& operator/=(const unsigned char&)__attribute__((always_inline));
  inline half_st& operator/=(const char&)__attribute__((always_inline));
  inline half_st& operator/=(const half&)__attribute__((always_inline));

  inline friend half_st operator/(const half_st&, const half_st&)__attribute__((always_inline));
  inline friend float_st operator/(const half_st&, const float_st&)__attribute__((always_inline));
  inline friend double_st operator/(const half_st&, const double_st&)__attribute__((always_inline));
  inline friend double_st operator/(const double_st&, const half_st&)__attribute__((always_inline));
  inline friend float_st operator/(const float_st&, const half_st&)__attribute__((always_inline));

  inline friend half_st operator/(const half_st&, const double&)__attribute__((always_inline));
  inline friend half_st operator/(const half_st&, const float&)__attribute__((always_inline));
  inline friend half_st operator/(const half_st&, const unsigned long long&)__attribute__((always_inline));
  inline friend half_st operator/(const half_st&, const long long&)__attribute__((always_inline));
  inline friend half_st operator/(const half_st&, const unsigned long&)__attribute__((always_inline));
  inline friend half_st operator/(const half_st&, const long&)__attribute__((always_inline));
  inline friend half_st operator/(const half_st&, const unsigned int&)__attribute__((always_inline));
  inline friend half_st operator/(const half_st&, const int&)__attribute__((always_inline));
  inline friend half_st operator/(const half_st&, const unsigned short&)__attribute__((always_inline));
  inline friend half_st operator/(const half_st&, const short&)__attribute__((always_inline));
  inline friend half_st operator/(const half_st&, const unsigned char&)__attribute__((always_inline));
  inline friend half_st operator/(const half_st&, const char&)__attribute__((always_inline));
  inline friend half_st operator/(const half_st&, const half&)__attribute__((always_inline));

  inline friend half_st operator/(const double&, const half_st&)__attribute__((always_inline));
  inline friend half_st operator/(const float&, const half_st&)__attribute__((always_inline));
  inline friend half_st operator/(const unsigned long long&, const half_st&)__attribute__((always_inline));
  inline friend half_st operator/(const long long&, const half_st&)__attribute__((always_inline));
  inline friend half_st operator/(const unsigned long&, const half_st&)__attribute__((always_inline));
  inline friend half_st operator/(const long&, const half_st&)__attribute__((always_inline));
  inline friend half_st operator/(const unsigned int&, const half_st&)__attribute__((always_inline));
  inline friend half_st operator/(const int&, const half_st&)__attribute__((always_inline));
  inline friend half_st operator/(const unsigned short&, const half_st&)__attribute__((always_inline));
  inline friend half_st operator/(const short&, const half_st&)__attribute__((always_inline));
  inline friend half_st operator/(const unsigned char&, const half_st&)__attribute__((always_inline));
  inline friend half_st operator/(const char&, const half_st&)__attribute__((always_inline));
  inline friend half_st operator/(const half&, const half_st&)__attribute__((always_inline));

// RELATIONAL OPERATORS

  inline friend int operator==(const half_st&, const half_st&)__attribute__((always_inline));  
  inline friend int operator==(const half_st&, const double_st&)__attribute__((always_inline)); 
  inline friend int operator==(const double_st&, const half_st&)__attribute__((always_inline)); 
  inline friend int operator==(const half_st&, const float_st&)__attribute__((always_inline)); 
  inline friend int operator==(const float_st&, const half_st&)__attribute__((always_inline)); 

  inline friend int operator==(const half_st&, const double&)__attribute__((always_inline));
  inline friend int operator==(const half_st&, const float&)__attribute__((always_inline));
  inline friend int operator==(const half_st&, const unsigned long long&)__attribute__((always_inline));
  inline friend int operator==(const half_st&, const long long&)__attribute__((always_inline));
  inline friend int operator==(const half_st&, const unsigned long&)__attribute__((always_inline));
  inline friend int operator==(const half_st&, const long&)__attribute__((always_inline));
  inline friend int operator==(const half_st&, const unsigned int&)__attribute__((always_inline));
  inline friend int operator==(const half_st&, const int&)__attribute__((always_inline));
  inline friend int operator==(const half_st&, const unsigned short&)__attribute__((always_inline));
  inline friend int operator==(const half_st&, const short&)__attribute__((always_inline));
  inline friend int operator==(const half_st&, const unsigned char&)__attribute__((always_inline));
  inline friend int operator==(const half_st&, const char&)__attribute__((always_inline));
  inline friend int operator==(const half_st&, const half&)__attribute__((always_inline));

  inline friend int operator==(const double&, const half_st&)__attribute__((always_inline));
  inline friend int operator==(const float&, const half_st&)__attribute__((always_inline));
  inline friend int operator==(const unsigned long long&, const half_st&)__attribute__((always_inline));
  inline friend int operator==(const long long&, const half_st&)__attribute__((always_inline));
  inline friend int operator==(const unsigned long&, const half_st&)__attribute__((always_inline));
  inline friend int operator==(const long&, const half_st&)__attribute__((always_inline));
  inline friend int operator==(const unsigned int&, const half_st&)__attribute__((always_inline));
  inline friend int operator==(const int&, const half_st&)__attribute__((always_inline));
  inline friend int operator==(const unsigned short&, const half_st&)__attribute__((always_inline));
  inline friend int operator==(const short&, const half_st&)__attribute__((always_inline));
  inline friend int operator==(const unsigned char&, const half_st&)__attribute__((always_inline));
  inline friend int operator==(const char&, const half_st&)__attribute__((always_inline));
  inline friend int operator==(const half&, const half_st&)__attribute__((always_inline));
  

  inline friend int operator!=(const half_st&, const half_st&)__attribute__((always_inline));
  inline friend int operator!=(const double_st&, const half_st&)__attribute__((always_inline));
  inline friend int operator!=(const half_st&, const double_st&)__attribute__((always_inline));
  inline friend int operator!=(const half_st&, const float_st&)__attribute__((always_inline));
  inline friend int operator!=(const float_st&, const half_st&)__attribute__((always_inline));

  inline friend int operator!=(const half_st&, const double&)__attribute__((always_inline));
  inline friend int operator!=(const half_st&, const float&)__attribute__((always_inline));
  inline friend int operator!=(const half_st&, const unsigned long long&)__attribute__((always_inline));
  inline friend int operator!=(const half_st&, const long long&)__attribute__((always_inline));
  inline friend int operator!=(const half_st&, const unsigned long&)__attribute__((always_inline));
  inline friend int operator!=(const half_st&, const long&)__attribute__((always_inline));
  inline friend int operator!=(const half_st&, const unsigned int&)__attribute__((always_inline));
  inline friend int operator!=(const half_st&, const int&)__attribute__((always_inline));
  inline friend int operator!=(const half_st&, const unsigned short&)__attribute__((always_inline));
  inline friend int operator!=(const half_st&, const short&)__attribute__((always_inline));
  inline friend int operator!=(const half_st&, const unsigned char&)__attribute__((always_inline));
  inline friend int operator!=(const half_st&, const char&)__attribute__((always_inline));
  inline friend int operator!=(const half_st&, const half&)__attribute__((always_inline));

  inline friend int operator!=(const double&, const half_st&)__attribute__((always_inline));
  inline friend int operator!=(const float&, const half_st&)__attribute__((always_inline));
  inline friend int operator!=(const unsigned long long&, const half_st&)__attribute__((always_inline));
  inline friend int operator!=(const long long&, const half_st&)__attribute__((always_inline));
  inline friend int operator!=(const unsigned long&, const half_st&)__attribute__((always_inline));
  inline friend int operator!=(const long&, const half_st&)__attribute__((always_inline));
  inline friend int operator!=(const unsigned int&, const half_st&)__attribute__((always_inline));
  inline friend int operator!=(const int&, const half_st&)__attribute__((always_inline));
  inline friend int operator!=(const unsigned short&, const half_st&)__attribute__((always_inline));
  inline friend int operator!=(const short&, const half_st&)__attribute__((always_inline));
  inline friend int operator!=(const unsigned char&, const half_st&)__attribute__((always_inline));
  inline friend int operator!=(const char&, const half_st&)__attribute__((always_inline));
  inline friend int operator!=(const half&, const half_st&)__attribute__((always_inline));

  inline friend int operator>=(const double_st&, const half_st&)__attribute__((always_inline));
  inline friend int operator>=(const half_st&, const double_st&)__attribute__((always_inline));  
  inline friend int operator>=(const half_st&, const float_st&)__attribute__((always_inline));
  inline friend int operator>=(const float_st&, const half_st&)__attribute__((always_inline));

  inline friend int operator>=(const half_st&, const double&)__attribute__((always_inline));
  inline friend int operator>=(const half_st&, const float&)__attribute__((always_inline));
  inline friend int operator>=(const half_st&, const unsigned long long&)__attribute__((always_inline));
  inline friend int operator>=(const half_st&, const long long&)__attribute__((always_inline));
  inline friend int operator>=(const half_st&, const unsigned long&)__attribute__((always_inline));
  inline friend int operator>=(const half_st&, const long&)__attribute__((always_inline));
  inline friend int operator>=(const half_st&, const unsigned int&)__attribute__((always_inline));
  inline friend int operator>=(const half_st&, const int&)__attribute__((always_inline));
  inline friend int operator>=(const half_st&, const unsigned short&)__attribute__((always_inline));
  inline friend int operator>=(const half_st&, const short&)__attribute__((always_inline));
  inline friend int operator>=(const half_st&, const unsigned char&)__attribute__((always_inline));
  inline friend int operator>=(const half_st&, const char&)__attribute__((always_inline));
  inline friend int operator>=(const half_st&, const half_st&)__attribute__((always_inline));  
  inline friend int operator>=(const half_st&, const half&)__attribute__((always_inline));


  inline friend int operator>=(const double&, const half_st&)__attribute__((always_inline));
  inline friend int operator>=(const float&, const half_st&)__attribute__((always_inline));
  inline friend int operator>=(const unsigned long long&, const half_st&)__attribute__((always_inline));
  inline friend int operator>=(const long long&, const half_st&)__attribute__((always_inline));
  inline friend int operator>=(const unsigned long&, const half_st&)__attribute__((always_inline));
  inline friend int operator>=(const long&, const half_st&)__attribute__((always_inline));
  inline friend int operator>=(const unsigned int&, const half_st&)__attribute__((always_inline));
  inline friend int operator>=(const int&, const half_st&)__attribute__((always_inline));
  inline friend int operator>=(const unsigned short&, const half_st&)__attribute__((always_inline));
  inline friend int operator>=(const short&, const half_st&)__attribute__((always_inline));
  inline friend int operator>=(const unsigned char&, const half_st&)__attribute__((always_inline));
  inline friend int operator>=(const char&, const half_st&)__attribute__((always_inline));
  inline friend int operator>=(const half&, const half_st&)__attribute__((always_inline));
  
  inline friend int operator>(const double_st&, const half_st&)__attribute__((always_inline));
  inline friend int operator>(const half_st&, const double_st&)__attribute__((always_inline));  
  inline friend int operator>(const half_st&, const float_st&)__attribute__((always_inline));
  inline friend int operator>(const float_st&, const half_st&)__attribute__((always_inline));

  inline friend int operator>(const half_st&, const double&)__attribute__((always_inline));
  inline friend int operator>(const half_st&, const float&)__attribute__((always_inline));
  inline friend int operator>(const half_st&, const unsigned long long&)__attribute__((always_inline));
  inline friend int operator>(const half_st&, const long long&)__attribute__((always_inline));
  inline friend int operator>(const half_st&, const unsigned long&)__attribute__((always_inline));
  inline friend int operator>(const half_st&, const long&)__attribute__((always_inline));
  inline friend int operator>(const half_st&, const unsigned int&)__attribute__((always_inline));
  inline friend int operator>(const half_st&, const int&)__attribute__((always_inline));
  inline friend int operator>(const half_st&, const unsigned short&)__attribute__((always_inline));
  inline friend int operator>(const half_st&, const short&)__attribute__((always_inline));
  inline friend int operator>(const half_st&, const unsigned char&)__attribute__((always_inline));
  inline friend int operator>(const half_st&, const char&)__attribute__((always_inline));
  inline friend int operator>(const half_st&, const half_st&)__attribute__((always_inline));  
  inline friend int operator>(const half_st&, const half&)__attribute__((always_inline));


  inline friend int operator>(const double&, const half_st&)__attribute__((always_inline));
  inline friend int operator>(const float&, const half_st&)__attribute__((always_inline));
  inline friend int operator>(const unsigned long long&, const half_st&)__attribute__((always_inline));
  inline friend int operator>(const long long&, const half_st&)__attribute__((always_inline));
  inline friend int operator>(const unsigned long&, const half_st&)__attribute__((always_inline));
  inline friend int operator>(const long&, const half_st&)__attribute__((always_inline));
  inline friend int operator>(const unsigned int&, const half_st&)__attribute__((always_inline));
  inline friend int operator>(const int&, const half_st&)__attribute__((always_inline));
  inline friend int operator>(const unsigned short&, const half_st&)__attribute__((always_inline));
  inline friend int operator>(const short&, const half_st&)__attribute__((always_inline));
  inline friend int operator>(const unsigned char&, const half_st&)__attribute__((always_inline));
  inline friend int operator>(const char&, const half_st&)__attribute__((always_inline));
  inline friend int operator>(const half&, const half_st&)__attribute__((always_inline));

 
  inline friend int operator<=(const double_st&, const half_st&)__attribute__((always_inline));
  inline friend int operator<=(const half_st&, const double_st&)__attribute__((always_inline));  
  inline friend int operator<=(const half_st&, const float_st&)__attribute__((always_inline));
  inline friend int operator<=(const float_st&, const half_st&)__attribute__((always_inline));

  inline friend int operator<=(const half_st&, const double&)__attribute__((always_inline));
  inline friend int operator<=(const half_st&, const float&)__attribute__((always_inline));
  inline friend int operator<=(const half_st&, const unsigned long long&)__attribute__((always_inline));
  inline friend int operator<=(const half_st&, const long long&)__attribute__((always_inline));
  inline friend int operator<=(const half_st&, const unsigned long&)__attribute__((always_inline));
  inline friend int operator<=(const half_st&, const long&)__attribute__((always_inline));
  inline friend int operator<=(const half_st&, const unsigned int&)__attribute__((always_inline));
  inline friend int operator<=(const half_st&, const int&)__attribute__((always_inline));
  inline friend int operator<=(const half_st&, const unsigned short&)__attribute__((always_inline));
  inline friend int operator<=(const half_st&, const short&)__attribute__((always_inline));
  inline friend int operator<=(const half_st&, const unsigned char&)__attribute__((always_inline));
  inline friend int operator<=(const half_st&, const char&)__attribute__((always_inline));
  inline friend int operator<=(const half_st&, const half_st&)__attribute__((always_inline));  
  inline friend int operator<=(const half_st&, const half&)__attribute__((always_inline));


  inline friend int operator<=(const double&, const half_st&)__attribute__((always_inline));
  inline friend int operator<=(const float&, const half_st&)__attribute__((always_inline));
  inline friend int operator<=(const unsigned long long&, const half_st&)__attribute__((always_inline));
  inline friend int operator<=(const long long&, const half_st&)__attribute__((always_inline));
  inline friend int operator<=(const unsigned long&, const half_st&)__attribute__((always_inline));
  inline friend int operator<=(const long&, const half_st&)__attribute__((always_inline));
  inline friend int operator<=(const unsigned int&, const half_st&)__attribute__((always_inline));
  inline friend int operator<=(const int&, const half_st&)__attribute__((always_inline));
  inline friend int operator<=(const unsigned short&, const half_st&)__attribute__((always_inline));
  inline friend int operator<=(const short&, const half_st&)__attribute__((always_inline));
  inline friend int operator<=(const unsigned char&, const half_st&)__attribute__((always_inline));
  inline friend int operator<=(const char&, const half_st&)__attribute__((always_inline));
  inline friend int operator<=(const half&, const half_st&)__attribute__((always_inline));

  inline friend int operator<(const double_st&, const half_st&)__attribute__((always_inline));
  inline friend int operator<(const half_st&, const double_st&)__attribute__((always_inline));  
  inline friend int operator<(const half_st&, const float_st&)__attribute__((always_inline));
  inline friend int operator<(const float_st&, const half_st&)__attribute__((always_inline));

  inline friend int operator<(const half_st&, const double&)__attribute__((always_inline));
  inline friend int operator<(const half_st&, const float&)__attribute__((always_inline));
  inline friend int operator<(const half_st&, const unsigned long long&)__attribute__((always_inline));
  inline friend int operator<(const half_st&, const long long&)__attribute__((always_inline));
  inline friend int operator<(const half_st&, const unsigned long&)__attribute__((always_inline));
  inline friend int operator<(const half_st&, const long&)__attribute__((always_inline));
  inline friend int operator<(const half_st&, const unsigned int&)__attribute__((always_inline));
  inline friend int operator<(const half_st&, const int&)__attribute__((always_inline));
  inline friend int operator<(const half_st&, const unsigned short&)__attribute__((always_inline));
  inline friend int operator<(const half_st&, const short&)__attribute__((always_inline));
  inline friend int operator<(const half_st&, const unsigned char&)__attribute__((always_inline));
  inline friend int operator<(const half_st&, const char&)__attribute__((always_inline));
  inline friend int operator<(const half_st&, const half_st&)__attribute__((always_inline));  
  inline friend int operator<(const half_st&, const half&)__attribute__((always_inline));


  inline friend int operator<(const double&, const half_st&)__attribute__((always_inline));
  inline friend int operator<(const float&, const half_st&)__attribute__((always_inline));
  inline friend int operator<(const unsigned long long&, const half_st&)__attribute__((always_inline));
  inline friend int operator<(const long long&, const half_st&)__attribute__((always_inline));
  inline friend int operator<(const unsigned long&, const half_st&)__attribute__((always_inline));
  inline friend int operator<(const long&, const half_st&)__attribute__((always_inline));
  inline friend int operator<(const unsigned int&, const half_st&)__attribute__((always_inline));
  inline friend int operator<(const int&, const half_st&)__attribute__((always_inline));
  inline friend int operator<(const unsigned short&, const half_st&)__attribute__((always_inline));
  inline friend int operator<(const short&, const half_st&)__attribute__((always_inline));
  inline friend int operator<(const unsigned char&, const half_st&)__attribute__((always_inline));
  inline friend int operator<(const char&, const half_st&)__attribute__((always_inline));
  inline friend int operator<(const half&, const half_st&)__attribute__((always_inline));

//MATHEMATICAL FUNCTIONS
  
  friend half_st  log(const half_st&);
  friend half_st  log2(const half_st&);
  friend half_st  log10(const half_st&);
  friend half_st  log1p(const half_st&);
  friend half_st  logb(const half_st&);
  friend half_st  exp(const half_st&); 
  friend half_st  exp2(const half_st&); 
  friend half_st  expm1(const half_st&); 
  friend half_st  sqrt(const half_st&);
  friend half_st  cbrt(const half_st&);
  friend half_st  sin(const half_st&);
  friend half_st  cos(const half_st&);
  friend half_st  tan(const half_st&);
  friend half_st  asin(const half_st&);
  friend half_st  acos(const half_st&);
  friend half_st  atan(const half_st&);
  friend half_st  atan2(const half_st&, const half_st&);
  friend half_st  sinh(const half_st&);
  friend half_st  cosh(const half_st&);
  friend half_st  tanh(const half_st&);
  friend half_st  asinh(const half_st&);
  friend half_st  acosh(const half_st&);
  friend half_st  atanh(const half_st&);
  friend half_st  hypot(const half_st&, const half_st&);


  
  friend half_st fmax(const half_st&, const half_st&);  
  friend half_st fmax(const half_st&, const half&);
  friend half_st fmax(const half&, const half_st&);


  friend half_st fmin(const half_st&, const half_st&);  
  friend half_st fmin(const half_st&, const half&);
  friend half_st fmin(const half&, const half_st&);
  
  friend half_st frexp(const half_st&, int* );
  friend half_st modf(const  half_st&,  half_st*);
  friend half_st ldexp(const half_st&, const int);
  friend half_st fmod(const half_st&, const half_st&);
  friend half_st copysign(const half_st&, const half_st&);

  friend half_st  erf(const half_st&);
  friend half_st  erfc(const half_st&);

  friend half_st pow(const half_st&, const half_st&);  
  friend half_st pow(const half_st&, const float_st&);  
  friend half_st pow(const float_st&, const half_st&);  
  friend half_st pow(const half_st&, const double_st&);  
  friend half_st pow(const double_st&, const half_st&);
  friend half_st pow(const half_st&, const double&);
  friend half_st pow(const half_st&, const float&);
  friend half_st pow(const half_st&, const unsigned long long&);
  friend half_st pow(const half_st&, const long long&);
  friend half_st pow(const half_st&, const unsigned long&);
  friend half_st pow(const half_st&, const long&);
  friend half_st pow(const half_st&, const unsigned int&);
  friend half_st pow(const half_st&, const int&);
  friend half_st pow(const half_st&, const unsigned short&);
  friend half_st pow(const half_st&, const short&);
  friend half_st pow(const half_st&, const unsigned char&);
  friend half_st pow(const half_st&, const char&);
  friend half_st pow(const half_st&, const half&);

  friend half_st pow(const double&, const half_st&);
  friend half_st pow(const float&, const half_st&);
  friend half_st pow(const unsigned long long&, const half_st&);
  friend half_st pow(const long long&, const half_st&);
  friend half_st pow(const unsigned long&, const half_st&);
  friend half_st pow(const long&, const half_st&);
  friend half_st pow(const unsigned int&, const half_st&);
  friend half_st pow(const int&, const half_st&);
  friend half_st pow(const unsigned short&, const half_st&);
  friend half_st pow(const short&, const half_st&);
  friend half_st pow(const unsigned char&, const half_st&);
  friend half_st pow(const char&, const half_st&);
  friend half_st pow(const half&, const half_st&);



  // INTRINSIC FUNCTIONS
  
  friend half_st fabs(const half_st&);
  friend half_st abs(const half_st&);
  friend half_st floor(const half_st&);
  friend half_st ceil(const half_st&);
  friend half_st trunc(const half_st&);
  friend half_st nearbyint(const half_st&);
  friend half_st rint(const half_st&);
  friend long int  lrint(const half_st&);
  friend long long int llrint(const half_st&);
  
  friend int isfinite(const half_st&);
  friend int isnan(const half_st&);
  friend int isinf(const half_st&);
  
  // Conversion
  operator char();  
  operator unsigned char();  
  operator short();  
  operator unsigned short();  
  operator int();  
  operator unsigned int();  
  operator long();  
  operator unsigned long();  
  operator long long();  
  operator unsigned long long();  
  operator float();
  operator double();
  operator half();
  operator float_st();
  operator double_st();

  operator double_st() const;
  operator float_st() const;
  operator char() const;  
  operator unsigned char() const;  
  operator short() const;  
  operator unsigned short() const;  
  operator int() const;  
  operator unsigned int() const;  
  operator long() const;  
  operator unsigned long() const;  
  operator long long() const;  
  operator unsigned long long() const;  
  operator float() const;
  operator double() const;
  operator half() const; 

  inline int nb_significant_digit() const __attribute__((always_inline));  
  inline int approx_digit() const __attribute__((always_inline));
  inline int computedzero() const __attribute__((always_inline));
  inline int numericalnoise() const __attribute__((always_inline));
  void display() const ; 
  void display(const char *) const ; 
  char* str( char *)  const ;
  friend char* strp(const half_st&);
  friend char* str( char *, const half_st&);
  friend std::istream& operator >>(std::istream& s, half_st&);
  
  void data_st();
  void data_st(const double&, const int&);
  
};

std::ostream& operator<<(std::ostream&, const half_st&);
std::istream& operator >>(std::istream&, half_st& );
#endif

#ifdef CADNA_HALF_NATIVE


class half_st 
{
  friend class float_st;
  friend class double_st;

typedef half TYPEBASE;


public:
  half x,y,z;
  mutable int16_t accuracy;
  inline half_st(void) :x(0),y(0),z(0),accuracy(3) {};
  inline half_st(half a, half b, half c):x((half)a),y((half)b),z((half)c),accuracy(-1) {};
  inline half_st(const half &a): x((half)a), y((half)a), z((half)a),accuracy(3) {};
  inline half_st(const double &a): x(a), y(a), z(a),accuracy(3) {};
  inline half_st(const double_st&a):x(a.x), y(a.y), z(a.z), accuracy(a.accuracy>=4?3:a.accuracy){};
  inline half_st(const float_st&a):x(a.x), y(a.y), z(a.z), accuracy(a.accuracy>=4?3:a.accuracy){};


  half getx() {return(this->x);}
  half gety() {return(this->y);}
  half getz() {return(this->z);}
  int getaccuracy() {return(this->accuracy);}
 // ASSIGNMENT
  
  inline half_st& operator=(const double&)__attribute__((always_inline));
  inline half_st& operator=(const float&)__attribute__((always_inline));
  inline half_st& operator=(const unsigned long long&)__attribute__((always_inline));
  inline half_st& operator=(const long long&)__attribute__((always_inline));
  inline half_st& operator=(const unsigned long&)__attribute__((always_inline));
  inline half_st& operator=(const long&)__attribute__((always_inline));
  inline half_st& operator=(const unsigned int&)__attribute__((always_inline));
  inline half_st& operator=(const int&)__attribute__((always_inline));
  inline half_st& operator=(const unsigned short&)__attribute__((always_inline));
  inline half_st& operator=(const short&)__attribute__((always_inline));
  inline half_st& operator=(const unsigned char&)__attribute__((always_inline));
  inline half_st& operator=(const char&)__attribute__((always_inline));
  inline half_st& operator=(const half&)__attribute__((always_inline));    
  inline half_st& operator=(const float_st&)__attribute__((always_inline));
  inline half_st& operator=(const double_st&)__attribute__((always_inline));

// ADDITION
  inline half_st operator++()__attribute__((always_inline));
  inline half_st operator++(int)__attribute__((always_inline));
  inline half_st operator+() const __attribute__((always_inline));
 
  inline half_st& operator+=(const half_st&)__attribute__((always_inline));
  inline half_st& operator+=(const double_st&)__attribute__((always_inline));
  inline half_st& operator+=(const float_st&)__attribute__((always_inline));
  inline half_st& operator+=(const double&)__attribute__((always_inline));
  inline half_st& operator+=(const float&)__attribute__((always_inline));
  inline half_st& operator+=(const unsigned long long&)__attribute__((always_inline));
  inline half_st& operator+=(const long long&)__attribute__((always_inline));
  inline half_st& operator+=(const unsigned long&)__attribute__((always_inline));
  inline half_st& operator+=(const long&)__attribute__((always_inline));
  inline half_st& operator+=(const unsigned int&)__attribute__((always_inline));
  inline half_st& operator+=(const int&)__attribute__((always_inline));
  inline half_st& operator+=(const unsigned short&)__attribute__((always_inline));
  inline half_st& operator+=(const short&)__attribute__((always_inline));
  inline half_st& operator+=(const unsigned char&)__attribute__((always_inline));
  inline half_st& operator+=(const char&)__attribute__((always_inline));
  inline half_st& operator+=(const half& a)__attribute__((always_inline));

  inline friend half_st operator+(const half_st&, const half_st&)__attribute__((always_inline)); 
  inline friend float_st operator+(const half_st&, const float_st&)__attribute__((always_inline));
  inline friend float_st operator+(const float_st&,  const half_st&)__attribute__((always_inline));
  inline friend double_st operator+(const half_st&, const double_st&)__attribute__((always_inline));
  inline friend double_st operator+(const double_st&,  const half_st&)__attribute__((always_inline));


  inline friend half_st operator+(const half_st&, const double&)__attribute__((always_inline));
  inline friend half_st operator+(const half_st&, const float&)__attribute__((always_inline));
  inline friend half_st operator+(const half_st&, const unsigned long long&)__attribute__((always_inline));
  inline friend half_st operator+(const half_st&, const long long&)__attribute__((always_inline));
  inline friend half_st operator+(const half_st&, const unsigned long&)__attribute__((always_inline));
  inline friend half_st operator+(const half_st&, const long&)__attribute__((always_inline));
  inline friend half_st operator+(const half_st&, const unsigned int&)__attribute__((always_inline));
  inline friend half_st operator+(const half_st&, const int&)__attribute__((always_inline));
  inline friend half_st operator+(const half_st&, const unsigned short&)__attribute__((always_inline));
  inline friend half_st operator+(const half_st&, const short&)__attribute__((always_inline));
  inline friend half_st operator+(const half_st&, const unsigned char&)__attribute__((always_inline));
  inline friend half_st operator+(const half_st&, const char&)__attribute__((always_inline));

  inline friend half_st operator+(const double&, const half_st&)__attribute__((always_inline));
  inline friend half_st operator+(const float&, const half_st&)__attribute__((always_inline));

  inline friend half_st operator+(const unsigned long long&, const half_st&)__attribute__((always_inline));
  inline friend half_st operator+(const long long&, const half_st&)__attribute__((always_inline));
  inline friend half_st operator+(const unsigned long&, const half_st&)__attribute__((always_inline));
  inline friend half_st operator+(const long&, const half_st&)__attribute__((always_inline));
  inline friend half_st operator+(const unsigned int&, const half_st&)__attribute__((always_inline));
  inline friend half_st operator+(const int&, const half_st&)__attribute__((always_inline));
  inline friend half_st operator+(const unsigned short&, const half_st&)__attribute__((always_inline));
  inline friend half_st operator+(const short&, const half_st&)__attribute__((always_inline));
  inline friend half_st operator+(const unsigned char&, const half_st&)__attribute__((always_inline));
  inline friend half_st operator+(const char&, const half_st&)__attribute__((always_inline));

  inline friend half_st operator+(const half&, const half_st&)__attribute__((always_inline));
  inline friend half_st operator+(const half_st&, const half&)__attribute__((always_inline));

// SUBTRACTION

  inline half_st operator--()__attribute__((always_inline));
  inline half_st operator--(int)__attribute__((always_inline));
  inline half_st operator-() const __attribute__((always_inline)); 

  inline half_st& operator-=(const half_st&)__attribute__((always_inline));
  inline half_st& operator-=(const double_st&)__attribute__((always_inline));
  inline half_st& operator-=(const float_st&)__attribute__((always_inline));

  inline half_st& operator-=(const double&)__attribute__((always_inline));
  inline half_st& operator-=(const float&)__attribute__((always_inline));
  inline half_st& operator-=(const unsigned long long&)__attribute__((always_inline));
  inline half_st& operator-=(const long long&)__attribute__((always_inline));
  inline half_st& operator-=(const unsigned long&)__attribute__((always_inline));
  inline half_st& operator-=(const long&)__attribute__((always_inline));
  inline half_st& operator-=(const unsigned int&)__attribute__((always_inline));
  inline half_st& operator-=(const int&)__attribute__((always_inline));
  inline half_st& operator-=(const unsigned short&)__attribute__((always_inline));
  inline half_st& operator-=(const short&)__attribute__((always_inline));
  inline half_st& operator-=(const unsigned char&)__attribute__((always_inline));
  inline half_st& operator-=(const char&)__attribute__((always_inline));
  inline half_st& operator-=(const half&)__attribute__((always_inline));

  inline friend half_st operator-(const half_st&, const half_st&)__attribute__((always_inline));  

  inline friend half_st operator-(const half_st&, const double&)__attribute__((always_inline));
  inline friend half_st operator-(const half_st&, const float&)__attribute__((always_inline));

  inline friend half_st operator-(const half_st&, const unsigned long long&)__attribute__((always_inline));
  inline friend half_st operator-(const half_st&, const long long&)__attribute__((always_inline));
  inline friend half_st operator-(const half_st&, const unsigned long&)__attribute__((always_inline));
  inline friend half_st operator-(const half_st&, const long&)__attribute__((always_inline));
  inline friend half_st operator-(const half_st&, const unsigned int&)__attribute__((always_inline));
  inline friend half_st operator-(const half_st&, const int&)__attribute__((always_inline));
  inline friend half_st operator-(const half_st&, const unsigned short&)__attribute__((always_inline));
  inline friend half_st operator-(const half_st&, const short&)__attribute__((always_inline));
  inline friend half_st operator-(const half_st&, const unsigned char&)__attribute__((always_inline));
  inline friend half_st operator-(const half_st&, const char&)__attribute__((always_inline));
  inline friend half_st operator-(const half_st&, const half_st&)__attribute__((always_inline));
  inline friend float_st operator-(const float_st&, const half_st&)__attribute__((always_inline));  
  inline friend float_st operator-(const half_st&, const float_st&)__attribute__((always_inline));  
  inline friend double_st operator-(const double_st&, const half_st&)__attribute__((always_inline));  
  inline friend double_st operator-(const half_st&, const double_st&)__attribute__((always_inline));   
 


  inline friend half_st operator-(const double&, const half_st&)__attribute__((always_inline));
  inline friend half_st operator-(const float&, const half_st&)__attribute__((always_inline));
  inline friend half_st operator-(const unsigned long long&, const half_st&)__attribute__((always_inline));
  inline friend half_st operator-(const long long&, const half_st&)__attribute__((always_inline));
  inline friend half_st operator-(const unsigned long&, const half_st&)__attribute__((always_inline));
  inline friend half_st operator-(const long&, const half_st&)__attribute__((always_inline));
  inline friend half_st operator-(const unsigned int&, const half_st&)__attribute__((always_inline));
  inline friend half_st operator-(const int&, const half_st&)__attribute__((always_inline));
  inline friend half_st operator-(const unsigned short&, const half_st&)__attribute__((always_inline));
  inline friend half_st operator-(const short&, const half_st&)__attribute__((always_inline));
  inline friend half_st operator-(const unsigned char&, const half_st&)__attribute__((always_inline));
  inline friend half_st operator-(const char&, const half_st&)__attribute__((always_inline));
  inline friend half_st operator-(const half&, const half_st&)__attribute__((always_inline));
  
  // MULTIPLICATION
  inline half_st& operator*=(const half_st&)__attribute__((always_inline)); 
  inline half_st& operator*=(const double_st&)__attribute__((always_inline));
  inline half_st& operator*=(const float_st&)__attribute__((always_inline));
  inline half_st& operator*=(const double&)__attribute__((always_inline));
  inline half_st& operator*=(const float&)__attribute__((always_inline));
  inline half_st& operator*=(const unsigned long long&)__attribute__((always_inline));
  inline half_st& operator*=(const long long&)__attribute__((always_inline));
  inline half_st& operator*=(const unsigned long&)__attribute__((always_inline));
  inline half_st& operator*=(const long&)__attribute__((always_inline));
  inline half_st& operator*=(const unsigned int&)__attribute__((always_inline));
  inline half_st& operator*=(const int&)__attribute__((always_inline));
  inline half_st& operator*=(const unsigned short&)__attribute__((always_inline));
  inline half_st& operator*=(const short&)__attribute__((always_inline));
  inline half_st& operator*=(const unsigned char&)__attribute__((always_inline));
  inline half_st& operator*=(const char&)__attribute__((always_inline));
  inline half_st& operator*=(const half&)__attribute__((always_inline));

  inline friend double_st operator*(const double_st&, const half_st&)__attribute__((always_inline));  
  inline friend double_st operator*(const half_st&, const double_st&)__attribute__((always_inline));  
  inline friend half_st operator*(const half_st&, const half_st&)__attribute__((always_inline));  
  inline friend float_st operator*(const float_st&, const half_st&)__attribute__((always_inline));  
  inline friend float_st operator*(const half_st&, const float_st&)__attribute__((always_inline));  

  inline friend half_st operator*(const half_st&, const double&)__attribute__((always_inline));
  inline friend half_st operator*(const half_st&, const float&)__attribute__((always_inline));
  inline friend half_st operator*(const half_st&, const unsigned long long&)__attribute__((always_inline));
  inline friend half_st operator*(const half_st&, const long long&)__attribute__((always_inline));
  inline friend half_st operator*(const half_st&, const unsigned long&)__attribute__((always_inline));
  inline friend half_st operator*(const half_st&, const long&)__attribute__((always_inline));
  inline friend half_st operator*(const half_st&, const unsigned int&)__attribute__((always_inline));
  inline friend half_st operator*(const half_st&, const int&)__attribute__((always_inline));
  inline friend half_st operator*(const half_st&, const unsigned short&)__attribute__((always_inline));
  inline friend half_st operator*(const half_st&, const short&)__attribute__((always_inline));
  inline friend half_st operator*(const half_st&, const unsigned char&)__attribute__((always_inline));
  inline friend half_st operator*(const half_st&, const char&)__attribute__((always_inline));
  inline friend half_st operator*(const half_st&, const half&)__attribute__((always_inline));

  inline friend half_st operator*(const double&, const half_st&)__attribute__((always_inline));
  inline friend half_st operator*(const float&, const half_st&)__attribute__((always_inline));
  inline friend half_st operator*(const unsigned long long&, const half_st&)__attribute__((always_inline));
  inline friend half_st operator*(const long long&, const half_st&)__attribute__((always_inline));
  inline friend half_st operator*(const unsigned long&, const half_st&)__attribute__((always_inline));
  inline friend half_st operator*(const long&, const half_st&)__attribute__((always_inline));
  inline friend half_st operator*(const unsigned int&, const half_st&)__attribute__((always_inline));
  inline friend half_st operator*(const int&, const half_st&)__attribute__((always_inline));
  inline friend half_st operator*(const unsigned short&, const half_st&)__attribute__((always_inline));
  inline friend half_st operator*(const short&, const half_st&)__attribute__((always_inline));
  inline friend half_st operator*(const unsigned char&, const half_st&)__attribute__((always_inline));
  inline friend half_st operator*(const char&, const half_st&)__attribute__((always_inline));
  inline friend half_st operator*(const half_st&, const half_st&)__attribute__((always_inline));

// DIVISION
  inline half_st& operator/=(const double_st&)__attribute__((always_inline));
  inline half_st& operator/=(const float_st&)__attribute__((always_inline));
  inline half_st& operator/=(const half_st&)__attribute__((always_inline));
  inline half_st& operator/=(const double&)__attribute__((always_inline));
  inline half_st& operator/=(const float&)__attribute__((always_inline));
  inline half_st& operator/=(const unsigned long long&)__attribute__((always_inline));
  inline half_st& operator/=(const long long&)__attribute__((always_inline));
  inline half_st& operator/=(const unsigned long&)__attribute__((always_inline));
  inline half_st& operator/=(const long&)__attribute__((always_inline));
  inline half_st& operator/=(const unsigned int&)__attribute__((always_inline));
  inline half_st& operator/=(const int&)__attribute__((always_inline));
  inline half_st& operator/=(const unsigned short&)__attribute__((always_inline));
  inline half_st& operator/=(const short&)__attribute__((always_inline));
  inline half_st& operator/=(const unsigned char&)__attribute__((always_inline));
  inline half_st& operator/=(const char&)__attribute__((always_inline));
  inline half_st& operator/=(const half&)__attribute__((always_inline));

  inline friend half_st operator/(const half_st&, const half_st&)__attribute__((always_inline));
  inline friend float_st operator/(const half_st&, const float_st&)__attribute__((always_inline));
  inline friend double_st operator/(const half_st&, const double_st&)__attribute__((always_inline));
  inline friend double_st operator/(const double_st&, const half_st&)__attribute__((always_inline));
  inline friend float_st operator/(const float_st&, const half_st&)__attribute__((always_inline));

  inline friend half_st operator/(const half_st&, const double&)__attribute__((always_inline));
  inline friend half_st operator/(const half_st&, const float&)__attribute__((always_inline));
  inline friend half_st operator/(const half_st&, const unsigned long long&)__attribute__((always_inline));
  inline friend half_st operator/(const half_st&, const long long&)__attribute__((always_inline));
  inline friend half_st operator/(const half_st&, const unsigned long&)__attribute__((always_inline));
  inline friend half_st operator/(const half_st&, const long&)__attribute__((always_inline));
  inline friend half_st operator/(const half_st&, const unsigned int&)__attribute__((always_inline));
  inline friend half_st operator/(const half_st&, const int&)__attribute__((always_inline));
  inline friend half_st operator/(const half_st&, const unsigned short&)__attribute__((always_inline));
  inline friend half_st operator/(const half_st&, const short&)__attribute__((always_inline));
  inline friend half_st operator/(const half_st&, const unsigned char&)__attribute__((always_inline));
  inline friend half_st operator/(const half_st&, const char&)__attribute__((always_inline));
  inline friend half_st operator/(const half_st&, const half&)__attribute__((always_inline));

  inline friend half_st operator/(const double&, const half_st&)__attribute__((always_inline));
  inline friend half_st operator/(const float&, const half_st&)__attribute__((always_inline));
  inline friend half_st operator/(const unsigned long long&, const half_st&)__attribute__((always_inline));
  inline friend half_st operator/(const long long&, const half_st&)__attribute__((always_inline));
  inline friend half_st operator/(const unsigned long&, const half_st&)__attribute__((always_inline));
  inline friend half_st operator/(const long&, const half_st&)__attribute__((always_inline));
  inline friend half_st operator/(const unsigned int&, const half_st&)__attribute__((always_inline));
  inline friend half_st operator/(const int&, const half_st&)__attribute__((always_inline));
  inline friend half_st operator/(const unsigned short&, const half_st&)__attribute__((always_inline));
  inline friend half_st operator/(const short&, const half_st&)__attribute__((always_inline));
  inline friend half_st operator/(const unsigned char&, const half_st&)__attribute__((always_inline));
  inline friend half_st operator/(const char&, const half_st&)__attribute__((always_inline));
  inline friend half_st operator/(const half&, const half_st&)__attribute__((always_inline));

// RELATIONAL OPERATORS

  inline friend int operator==(const half_st&, const half_st&)__attribute__((always_inline));  
  inline friend int operator==(const half_st&, const double_st&)__attribute__((always_inline)); 
  inline friend int operator==(const double_st&, const half_st&)__attribute__((always_inline)); 
  inline friend int operator==(const half_st&, const float_st&)__attribute__((always_inline)); 
  inline friend int operator==(const float_st&, const half_st&)__attribute__((always_inline)); 

  inline friend int operator==(const half_st&, const double&)__attribute__((always_inline));
  inline friend int operator==(const half_st&, const float&)__attribute__((always_inline));
  inline friend int operator==(const half_st&, const unsigned long long&)__attribute__((always_inline));
  inline friend int operator==(const half_st&, const long long&)__attribute__((always_inline));
  inline friend int operator==(const half_st&, const unsigned long&)__attribute__((always_inline));
  inline friend int operator==(const half_st&, const long&)__attribute__((always_inline));
  inline friend int operator==(const half_st&, const unsigned int&)__attribute__((always_inline));
  inline friend int operator==(const half_st&, const int&)__attribute__((always_inline));
  inline friend int operator==(const half_st&, const unsigned short&)__attribute__((always_inline));
  inline friend int operator==(const half_st&, const short&)__attribute__((always_inline));
  inline friend int operator==(const half_st&, const unsigned char&)__attribute__((always_inline));
  inline friend int operator==(const half_st&, const char&)__attribute__((always_inline));
  inline friend int operator==(const half_st&, const half&)__attribute__((always_inline));

  inline friend int operator==(const double&, const half_st&)__attribute__((always_inline));
  inline friend int operator==(const float&, const half_st&)__attribute__((always_inline));
  inline friend int operator==(const unsigned long long&, const half_st&)__attribute__((always_inline));
  inline friend int operator==(const long long&, const half_st&)__attribute__((always_inline));
  inline friend int operator==(const unsigned long&, const half_st&)__attribute__((always_inline));
  inline friend int operator==(const long&, const half_st&)__attribute__((always_inline));
  inline friend int operator==(const unsigned int&, const half_st&)__attribute__((always_inline));
  inline friend int operator==(const int&, const half_st&)__attribute__((always_inline));
  inline friend int operator==(const unsigned short&, const half_st&)__attribute__((always_inline));
  inline friend int operator==(const short&, const half_st&)__attribute__((always_inline));
  inline friend int operator==(const unsigned char&, const half_st&)__attribute__((always_inline));
  inline friend int operator==(const char&, const half_st&)__attribute__((always_inline));
  inline friend int operator==(const half&, const half_st&)__attribute__((always_inline));
  

  inline friend int operator!=(const half_st&, const half_st&)__attribute__((always_inline));
  inline friend int operator!=(const double_st&, const half_st&)__attribute__((always_inline));
  inline friend int operator!=(const half_st&, const double_st&)__attribute__((always_inline));
  inline friend int operator!=(const half_st&, const float_st&)__attribute__((always_inline));
  inline friend int operator!=(const float_st&, const half_st&)__attribute__((always_inline));

  inline friend int operator!=(const half_st&, const double&)__attribute__((always_inline));
  inline friend int operator!=(const half_st&, const float&)__attribute__((always_inline));
  inline friend int operator!=(const half_st&, const unsigned long long&)__attribute__((always_inline));
  inline friend int operator!=(const half_st&, const long long&)__attribute__((always_inline));
  inline friend int operator!=(const half_st&, const unsigned long&)__attribute__((always_inline));
  inline friend int operator!=(const half_st&, const long&)__attribute__((always_inline));
  inline friend int operator!=(const half_st&, const unsigned int&)__attribute__((always_inline));
  inline friend int operator!=(const half_st&, const int&)__attribute__((always_inline));
  inline friend int operator!=(const half_st&, const unsigned short&)__attribute__((always_inline));
  inline friend int operator!=(const half_st&, const short&)__attribute__((always_inline));
  inline friend int operator!=(const half_st&, const unsigned char&)__attribute__((always_inline));
  inline friend int operator!=(const half_st&, const char&)__attribute__((always_inline));
  inline friend int operator!=(const half_st&, const half&)__attribute__((always_inline));

  inline friend int operator!=(const double&, const half_st&)__attribute__((always_inline));
  inline friend int operator!=(const float&, const half_st&)__attribute__((always_inline));
  inline friend int operator!=(const unsigned long long&, const half_st&)__attribute__((always_inline));
  inline friend int operator!=(const long long&, const half_st&)__attribute__((always_inline));
  inline friend int operator!=(const unsigned long&, const half_st&)__attribute__((always_inline));
  inline friend int operator!=(const long&, const half_st&)__attribute__((always_inline));
  inline friend int operator!=(const unsigned int&, const half_st&)__attribute__((always_inline));
  inline friend int operator!=(const int&, const half_st&)__attribute__((always_inline));
  inline friend int operator!=(const unsigned short&, const half_st&)__attribute__((always_inline));
  inline friend int operator!=(const short&, const half_st&)__attribute__((always_inline));
  inline friend int operator!=(const unsigned char&, const half_st&)__attribute__((always_inline));
  inline friend int operator!=(const char&, const half_st&)__attribute__((always_inline));
  inline friend int operator!=(const half&, const half_st&)__attribute__((always_inline));

  inline friend int operator>=(const double_st&, const half_st&)__attribute__((always_inline));
  inline friend int operator>=(const half_st&, const double_st&)__attribute__((always_inline));  
  inline friend int operator>=(const half_st&, const float_st&)__attribute__((always_inline));
  inline friend int operator>=(const float_st&, const half_st&)__attribute__((always_inline));

  inline friend int operator>=(const half_st&, const double&)__attribute__((always_inline));
  inline friend int operator>=(const half_st&, const float&)__attribute__((always_inline));
  inline friend int operator>=(const half_st&, const unsigned long long&)__attribute__((always_inline));
  inline friend int operator>=(const half_st&, const long long&)__attribute__((always_inline));
  inline friend int operator>=(const half_st&, const unsigned long&)__attribute__((always_inline));
  inline friend int operator>=(const half_st&, const long&)__attribute__((always_inline));
  inline friend int operator>=(const half_st&, const unsigned int&)__attribute__((always_inline));
  inline friend int operator>=(const half_st&, const int&)__attribute__((always_inline));
  inline friend int operator>=(const half_st&, const unsigned short&)__attribute__((always_inline));
  inline friend int operator>=(const half_st&, const short&)__attribute__((always_inline));
  inline friend int operator>=(const half_st&, const unsigned char&)__attribute__((always_inline));
  inline friend int operator>=(const half_st&, const char&)__attribute__((always_inline));
  inline friend int operator>=(const half_st&, const half_st&)__attribute__((always_inline));  
  inline friend int operator>=(const half_st&, const half&)__attribute__((always_inline));


  inline friend int operator>=(const double&, const half_st&)__attribute__((always_inline));
  inline friend int operator>=(const float&, const half_st&)__attribute__((always_inline));
  inline friend int operator>=(const unsigned long long&, const half_st&)__attribute__((always_inline));
  inline friend int operator>=(const long long&, const half_st&)__attribute__((always_inline));
  inline friend int operator>=(const unsigned long&, const half_st&)__attribute__((always_inline));
  inline friend int operator>=(const long&, const half_st&)__attribute__((always_inline));
  inline friend int operator>=(const unsigned int&, const half_st&)__attribute__((always_inline));
  inline friend int operator>=(const int&, const half_st&)__attribute__((always_inline));
  inline friend int operator>=(const unsigned short&, const half_st&)__attribute__((always_inline));
  inline friend int operator>=(const short&, const half_st&)__attribute__((always_inline));
  inline friend int operator>=(const unsigned char&, const half_st&)__attribute__((always_inline));
  inline friend int operator>=(const char&, const half_st&)__attribute__((always_inline));
  inline friend int operator>=(const half&, const half_st&)__attribute__((always_inline));
  
  inline friend int operator>(const double_st&, const half_st&)__attribute__((always_inline));
  inline friend int operator>(const half_st&, const double_st&)__attribute__((always_inline));  
  inline friend int operator>(const half_st&, const float_st&)__attribute__((always_inline));
  inline friend int operator>(const float_st&, const half_st&)__attribute__((always_inline));

  inline friend int operator>(const half_st&, const double&)__attribute__((always_inline));
  inline friend int operator>(const half_st&, const float&)__attribute__((always_inline));
  inline friend int operator>(const half_st&, const unsigned long long&)__attribute__((always_inline));
  inline friend int operator>(const half_st&, const long long&)__attribute__((always_inline));
  inline friend int operator>(const half_st&, const unsigned long&)__attribute__((always_inline));
  inline friend int operator>(const half_st&, const long&)__attribute__((always_inline));
  inline friend int operator>(const half_st&, const unsigned int&)__attribute__((always_inline));
  inline friend int operator>(const half_st&, const int&)__attribute__((always_inline));
  inline friend int operator>(const half_st&, const unsigned short&)__attribute__((always_inline));
  inline friend int operator>(const half_st&, const short&)__attribute__((always_inline));
  inline friend int operator>(const half_st&, const unsigned char&)__attribute__((always_inline));
  inline friend int operator>(const half_st&, const char&)__attribute__((always_inline));
  inline friend int operator>(const half_st&, const half_st&)__attribute__((always_inline));  
  inline friend int operator>(const half_st&, const half&)__attribute__((always_inline));


  inline friend int operator>(const double&, const half_st&)__attribute__((always_inline));
  inline friend int operator>(const float&, const half_st&)__attribute__((always_inline));
  inline friend int operator>(const unsigned long long&, const half_st&)__attribute__((always_inline));
  inline friend int operator>(const long long&, const half_st&)__attribute__((always_inline));
  inline friend int operator>(const unsigned long&, const half_st&)__attribute__((always_inline));
  inline friend int operator>(const long&, const half_st&)__attribute__((always_inline));
  inline friend int operator>(const unsigned int&, const half_st&)__attribute__((always_inline));
  inline friend int operator>(const int&, const half_st&)__attribute__((always_inline));
  inline friend int operator>(const unsigned short&, const half_st&)__attribute__((always_inline));
  inline friend int operator>(const short&, const half_st&)__attribute__((always_inline));
  inline friend int operator>(const unsigned char&, const half_st&)__attribute__((always_inline));
  inline friend int operator>(const char&, const half_st&)__attribute__((always_inline));
  inline friend int operator>(const half&, const half_st&)__attribute__((always_inline));

 
  inline friend int operator<=(const double_st&, const half_st&)__attribute__((always_inline));
  inline friend int operator<=(const half_st&, const double_st&)__attribute__((always_inline));  
  inline friend int operator<=(const half_st&, const float_st&)__attribute__((always_inline));
  inline friend int operator<=(const float_st&, const half_st&)__attribute__((always_inline));

  inline friend int operator<=(const half_st&, const double&)__attribute__((always_inline));
  inline friend int operator<=(const half_st&, const float&)__attribute__((always_inline));
  inline friend int operator<=(const half_st&, const unsigned long long&)__attribute__((always_inline));
  inline friend int operator<=(const half_st&, const long long&)__attribute__((always_inline));
  inline friend int operator<=(const half_st&, const unsigned long&)__attribute__((always_inline));
  inline friend int operator<=(const half_st&, const long&)__attribute__((always_inline));
  inline friend int operator<=(const half_st&, const unsigned int&)__attribute__((always_inline));
  inline friend int operator<=(const half_st&, const int&)__attribute__((always_inline));
  inline friend int operator<=(const half_st&, const unsigned short&)__attribute__((always_inline));
  inline friend int operator<=(const half_st&, const short&)__attribute__((always_inline));
  inline friend int operator<=(const half_st&, const unsigned char&)__attribute__((always_inline));
  inline friend int operator<=(const half_st&, const char&)__attribute__((always_inline));
  inline friend int operator<=(const half_st&, const half_st&)__attribute__((always_inline));  
  inline friend int operator<=(const half_st&, const half&)__attribute__((always_inline));


  inline friend int operator<=(const double&, const half_st&)__attribute__((always_inline));
  inline friend int operator<=(const float&, const half_st&)__attribute__((always_inline));
  inline friend int operator<=(const unsigned long long&, const half_st&)__attribute__((always_inline));
  inline friend int operator<=(const long long&, const half_st&)__attribute__((always_inline));
  inline friend int operator<=(const unsigned long&, const half_st&)__attribute__((always_inline));
  inline friend int operator<=(const long&, const half_st&)__attribute__((always_inline));
  inline friend int operator<=(const unsigned int&, const half_st&)__attribute__((always_inline));
  inline friend int operator<=(const int&, const half_st&)__attribute__((always_inline));
  inline friend int operator<=(const unsigned short&, const half_st&)__attribute__((always_inline));
  inline friend int operator<=(const short&, const half_st&)__attribute__((always_inline));
  inline friend int operator<=(const unsigned char&, const half_st&)__attribute__((always_inline));
  inline friend int operator<=(const char&, const half_st&)__attribute__((always_inline));
  inline friend int operator<=(const half&, const half_st&)__attribute__((always_inline));

  inline friend int operator<(const double_st&, const half_st&)__attribute__((always_inline));
  inline friend int operator<(const half_st&, const double_st&)__attribute__((always_inline));  
  inline friend int operator<(const half_st&, const float_st&)__attribute__((always_inline));
  inline friend int operator<(const float_st&, const half_st&)__attribute__((always_inline));

  inline friend int operator<(const half_st&, const double&)__attribute__((always_inline));
  inline friend int operator<(const half_st&, const float&)__attribute__((always_inline));
  inline friend int operator<(const half_st&, const unsigned long long&)__attribute__((always_inline));
  inline friend int operator<(const half_st&, const long long&)__attribute__((always_inline));
  inline friend int operator<(const half_st&, const unsigned long&)__attribute__((always_inline));
  inline friend int operator<(const half_st&, const long&)__attribute__((always_inline));
  inline friend int operator<(const half_st&, const unsigned int&)__attribute__((always_inline));
  inline friend int operator<(const half_st&, const int&)__attribute__((always_inline));
  inline friend int operator<(const half_st&, const unsigned short&)__attribute__((always_inline));
  inline friend int operator<(const half_st&, const short&)__attribute__((always_inline));
  inline friend int operator<(const half_st&, const unsigned char&)__attribute__((always_inline));
  inline friend int operator<(const half_st&, const char&)__attribute__((always_inline));
  inline friend int operator<(const half_st&, const half_st&)__attribute__((always_inline));  
  inline friend int operator<(const half_st&, const half&)__attribute__((always_inline));


  inline friend int operator<(const double&, const half_st&)__attribute__((always_inline));
  inline friend int operator<(const float&, const half_st&)__attribute__((always_inline));
  inline friend int operator<(const unsigned long long&, const half_st&)__attribute__((always_inline));
  inline friend int operator<(const long long&, const half_st&)__attribute__((always_inline));
  inline friend int operator<(const unsigned long&, const half_st&)__attribute__((always_inline));
  inline friend int operator<(const long&, const half_st&)__attribute__((always_inline));
  inline friend int operator<(const unsigned int&, const half_st&)__attribute__((always_inline));
  inline friend int operator<(const int&, const half_st&)__attribute__((always_inline));
  inline friend int operator<(const unsigned short&, const half_st&)__attribute__((always_inline));
  inline friend int operator<(const short&, const half_st&)__attribute__((always_inline));
  inline friend int operator<(const unsigned char&, const half_st&)__attribute__((always_inline));
  inline friend int operator<(const char&, const half_st&)__attribute__((always_inline));
  inline friend int operator<(const half&, const half_st&)__attribute__((always_inline));

//MATHEMATICAL FUNCTIONS
  friend half_st  log(const half_st&);
  friend half_st  log2(const half_st&);
  friend half_st  log10(const half_st&);
  friend half_st  log1p(const half_st&);
  friend half_st  logb(const half_st&);
  friend half_st  exp(const half_st&); 
  friend half_st  exp2(const half_st&); 
  friend half_st  expm1(const half_st&); 
  friend half_st  sqrt(const half_st&);
  friend half_st  cbrt(const half_st&);
  friend half_st  sin(const half_st&);
  friend half_st  cos(const half_st&);
  friend half_st  tan(const half_st&);
  friend half_st  asin(const half_st&);
  friend half_st  acos(const half_st&);
  friend half_st  atan(const half_st&);
  friend half_st  atan2(const half_st&, const half&);
  friend half_st  atan2(const half_st&, const float&);
  friend half_st  atan2(const float_st&, const half&);
  friend half_st  atan2(const half_st&, const double&);
  friend half_st  atan2(const double_st&, const half&);
  friend half_st  sinh(const half_st&);
  friend half_st  cosh(const half_st&);
  friend half_st  tanh(const half_st&);
  friend half_st  asinh(const half_st&);
  friend half_st  acosh(const half_st&);
  friend half_st  atanh(const half_st&);
  friend half_st  hypot(const half_st&, const half&);
  friend half_st  hypot(const half_st&, const float&);
  friend half_st  hypot(const float_st&, const half&);
  friend half_st  hypot(const half_st&, const double&);
  friend half_st  hypot(const double_st&, const half&);

  
  friend half_st pow(const half_st&, const half_st&);
  friend half_st pow(const half_st&, const double_st&);
  friend half_st pow(const double_st&, const half_st&);
  friend half_st pow(const half_st&, const float_st&);
  friend half_st pow(const float_st&, const half_st&);
  friend half_st pow(const half_st&, const half&);
  friend half_st pow(const half_st&, const double&);
  friend half_st pow(const half_st&, const float&);
  friend half_st pow(const half_st&, const unsigned long long&);
  friend half_st pow(const half_st&, const long long&);
  friend half_st pow(const half_st&, const unsigned long&);
  friend half_st pow(const half_st&, const long&);
  friend half_st pow(const half_st&, const unsigned int&);
  friend half_st pow(const half_st&, const int&);
  friend half_st pow(const half_st&, const unsigned short&);
  friend half_st pow(const half_st&, const short&);
  friend half_st pow(const half_st&, const unsigned char);
  friend half_st pow(const half_st&, const char&);

  friend half_st pow(const half&, const half_st&);
  friend half_st pow(const double&, const half_st&);
  friend half_st pow(const float&, const half_st&);
  friend half_st pow(const unsigned long long&, const half_st&);
  friend half_st pow(const long long&, const half_st&);
  friend half_st pow(const unsigned long&, const half_st&);
  friend half_st pow(const long&, const half_st&);
  friend half_st pow(const unsigned int&, const half_st&);
  friend half_st pow(const int&, const half_st&);
  friend half_st pow(const unsigned short&, const half_st&);
  friend half_st pow(const short&, const half_st&);
  friend half_st pow(const unsigned char&, const half_st&);
  friend half_st pow(const char&, const half_st&);


  friend half_st fmax(const half_st&, const half_st&);  
  /*friend half_st fmax(const half_st&, const float_st&);  
  friend half_st fmax(const float_st&, const half_st&); 
  friend half_st fmax(const half_st&, const double_st&);  
  friend half_st fmax(const double_st&, const half_st&);  
  friend half_st fmax(const half_st&, const half&);
  friend half_st fmax(const half_st&, const double&);
  friend half_st fmax(const half_st&, const float&);
  friend half_st fmax(const half_st&, const unsigned long long&);
  friend half_st fmax(const half_st&, const long long&);
  friend half_st fmax(const half_st&, const unsigned long&);
  friend half_st fmax(const half_st&, const long&);
  friend half_st fmax(const half_st&, const unsigned int&);
  friend half_st fmax(const half_st&, const int&);
  friend half_st fmax(const half_st&, const unsigned short&);
  friend half_st fmax(const half_st&, const short&);
  friend half_st fmax(const half_st&, const unsigned char&);
  friend half_st fmax(const half_st&, const char&);
*/
  friend half_st fmax(const half&, const half_st&);
/*  friend half_st fmax(const double&, const half_st&);
  friend half_st fmax(const float&, const half_st&);
  friend half_st fmax(const unsigned long long&, const half_st&);
  friend half_st fmax(const long long&, const half_st&);
  friend half_st fmax(const unsigned long&, const half_st&);
  friend half_st fmax(const long&, const half_st&);
  friend half_st fmax(const unsigned int&, const half_st&);
  friend half_st fmax(const int&, const half_st&);
  friend half_st fmax(const unsigned short&, const half_st&);
  friend half_st fmax(const short&, const half_st&);
  friend half_st fmax(const unsigned char&, const half_st&);
  friend half_st fmax(const char&, const half_st&);
 
*/
  friend half_st fmin(const half_st&, const half_st&); 
 /* friend half_st fmin(const half_st&, const double_st&);  
  friend half_st fmin(const double_st&, const half_st&);  
  friend half_st fmin(const half_st&, const float_st&);  
  friend half_st fmin(const float_st&, const half_st&);  
  friend half_st fmin(const half_st&, const half&);
  friend half_st fmin(const half_st&, const double&);
  friend half_st fmin(const half_st&, const float&);
  friend half_st fmin(const half_st&, const unsigned long long&);
  friend half_st fmin(const half_st&, const long long&);
  friend half_st fmin(const half_st&, const unsigned long&);
  friend half_st fmin(const half_st&, const long&);
  friend half_st fmin(const half_st&, const unsigned int&);
  friend half_st fmin(const half_st&, const int&);
  friend half_st fmin(const half_st&, const unsigned short&);
  friend half_st fmin(const half_st&, const short&);
  friend half_st fmin(const half_st&, const unsigned char&);
  friend half_st fmin(const half_st&, const char&);
*/
  
  friend half_st fmin(const half&, const half_st&);
 /* friend half_st fmin(const double&, const half_st&);
  friend half_st fmin(const float&, const half_st&);
  friend half_st fmin(const unsigned long long&, const half_st&);
  friend half_st fmin(const long long&, const half_st&);
  friend half_st fmin(const unsigned long&, const half_st&);
  friend half_st fmin(const long&, const half_st&);
  friend half_st fmin(const unsigned int&, const half_st&);
  friend half_st fmin(const int&, const half_st&);
  friend half_st fmin(const unsigned short&, const half_st&);
  friend half_st fmin(const short&, const half_st&);
  friend half_st fmin(const unsigned char&, const half_st&);
  friend half_st fmin(const char&, const half_st&);
*/
  
  friend half_st frexp(const half_st&, int* );
 // friend half_st modf(const  half_st&,  half_st*);
  friend half_st ldexp(const half_st&, const int);
  friend half_st scalbn(const half_st&, const int);
  friend half_st fmod(const half_st&, const half_st&);
  friend half_st copysign(const half_st&, const half_st&);
  //friend half_st  erf(const half_st&);
  //friend half_st  erfc(const half_st&);


  // FMA
#ifdef CADNA_FMA  
  inline friend double_st fma( double_st const& a,  double_st const& b,  double_st const& c)__attribute__((always_inline));
  inline friend double_st fma( float_st const& a,  double_st const& b,  double_st const& c)__attribute__((always_inline));
  inline friend double_st fma( double_st const& a,  float_st const& b,  double_st const& c)__attribute__((always_inline));
  inline friend double_st fma( double_st const& a,  double_st const& b,  float_st const& c)__attribute__((always_inline));
  inline friend double_st fma( float_st const& a,  float_st const& b,  double_st const& c)__attribute__((always_inline));
  inline friend double_st fma( float_st const& a,  double_st const& b,  float_st const& c)__attribute__((always_inline));
  inline friend double_st fma( double_st const& a,  float_st const& b,  float_st const& c)__attribute__((always_inline));
  inline friend float_st fma( float_st const& a,  float_st const& b,  float_st const& c)__attribute__((always_inline));

  inline friend double_st fma_no_instab( double_st const& a,  double_st const& b,  double_st const& c)__attribute__((always_inline));
  inline friend double_st fma_no_instab( float_st const& a,  double_st const& b,  double_st const& c)__attribute__((always_inline));
  inline friend double_st fma_no_instab( double_st const& a,  float_st const& b,  double_st const& c)__attribute__((always_inline));
  inline friend double_st fma_no_instab( double_st const& a,  double_st const& b,  float_st const& c)__attribute__((always_inline));
  inline friend double_st fma_no_instab( float_st const& a,  float_st const& b,  double_st const& c)__attribute__((always_inline));
  inline friend double_st fma_no_instab( float_st const& a,  double_st const& b,  float_st const& c)__attribute__((always_inline));
  inline friend double_st fma_no_instab( double_st const& a,  float_st const& b,  float_st const& c)__attribute__((always_inline));
  inline friend float_st fma_no_instab( float_st const& a,  float_st const& b,  float_st const& c)__attribute__((always_inline));
#endif // CADNA_FMA

  // COMPENSATED ALGORITHMS
  inline friend void twoSum( double_st&a,  double_st&b, double_st&s, double_st&e)__attribute__((always_inline));
  inline friend void twoSum( double_st&a,  double_st&b, double_st&s, float_st&e)__attribute__((always_inline));
  inline friend void twoSum( float_st&a,  double_st&b, double_st&s, double_st&e)__attribute__((always_inline));
  inline friend void twoSum( float_st&a,  double_st&b, double_st&s, float_st&e)__attribute__((always_inline));
  inline friend void twoSum( double_st&a,  float_st&b, double_st&s, double_st&e)__attribute__((always_inline));
  inline friend void twoSum( double_st&a,  float_st&b, double_st&s, float_st&e)__attribute__((always_inline));
  inline friend void twoSum( double_st&a,  double_st&b, float_st&s, double_st&e)__attribute__((always_inline));
  inline friend void twoSum( double_st&a,  double_st&b, float_st&s, float_st&e)__attribute__((always_inline));
  inline friend void twoSum( float_st&a,  float_st&b, double_st&s, double_st&e)__attribute__((always_inline));
  inline friend void twoSum( float_st&a,  float_st&b, double_st&s, float_st&e)__attribute__((always_inline));
  inline friend void twoSum( float_st&a,  double_st&b, float_st&s, double_st&e)__attribute__((always_inline));
  inline friend void twoSum( float_st&a,  double_st&b, float_st&s, float_st&e)__attribute__((always_inline));
  inline friend void twoSum( double_st&a,  float_st&b, float_st&s, double_st&e)__attribute__((always_inline));
  inline friend void twoSum( double_st&a,  float_st&b, float_st&s, float_st&e)__attribute__((always_inline));
  inline friend void twoSum( float_st&a,  float_st&b, float_st&s, float_st&e)__attribute__((always_inline));

  inline friend void fastTwoSum( double_st&a,  double_st&b, double_st&s, double_st&e)__attribute__((always_inline));
  inline friend void fastTwoSum( double_st&a,  double_st&b, double_st&s, float_st&e)__attribute__((always_inline));
  inline friend void fastTwoSum( float_st&a,  double_st&b, double_st&s, double_st&e)__attribute__((always_inline));
  inline friend void fastTwoSum( float_st&a,  double_st&b, double_st&s, float_st&e)__attribute__((always_inline));
  inline friend void fastTwoSum( double_st&a,  float_st&b, double_st&s, double_st&e)__attribute__((always_inline));
  inline friend void fastTwoSum( double_st&a,  float_st&b, double_st&s, float_st&e)__attribute__((always_inline));
  inline friend void fastTwoSum( double_st&a,  double_st&b, float_st&s, double_st&e)__attribute__((always_inline));
  inline friend void fastTwoSum( double_st&a,  double_st&b, float_st&s, float_st&e)__attribute__((always_inline));
  inline friend void fastTwoSum( float_st&a,  float_st&b, double_st&s, double_st&e)__attribute__((always_inline));
  inline friend void fastTwoSum( float_st&a,  float_st&b, double_st&s, float_st&e)__attribute__((always_inline));
  inline friend void fastTwoSum( float_st&a,  double_st&b, float_st&s, double_st&e)__attribute__((always_inline));
  inline friend void fastTwoSum( float_st&a,  double_st&b, float_st&s, float_st&e)__attribute__((always_inline));
  inline friend void fastTwoSum( double_st&a,  float_st&b, float_st&s, double_st&e)__attribute__((always_inline));
  inline friend void fastTwoSum( double_st&a,  float_st&b, float_st&s, float_st&e)__attribute__((always_inline));
  inline friend void fastTwoSum( float_st&a,  float_st&b, float_st&s, float_st&e)__attribute__((always_inline));

  inline friend void twoSumPriest( double_st&a,  double_st&b, double_st&c, double_st&d)__attribute__((always_inline));
  inline friend void twoSumPriest( double_st&a,  double_st&b, double_st&c, float_st&d)__attribute__((always_inline));
  inline friend void twoSumPriest( float_st&a,  double_st&b, double_st&c, double_st&d)__attribute__((always_inline));
  inline friend void twoSumPriest( float_st&a,  double_st&b, double_st&c, float_st&d)__attribute__((always_inline));
  inline friend void twoSumPriest( double_st&a,  float_st&b, double_st&c, double_st&d)__attribute__((always_inline));
  inline friend void twoSumPriest( double_st&a,  float_st&b, double_st&c, float_st&d)__attribute__((always_inline));
  inline friend void twoSumPriest( double_st&a,  double_st&b, float_st&c, double_st&d)__attribute__((always_inline));
  inline friend void twoSumPriest( double_st&a,  double_st&b, float_st&c, float_st&d)__attribute__((always_inline));
  inline friend void twoSumPriest( float_st&a,  float_st&b, double_st&c, double_st&d)__attribute__((always_inline));
  inline friend void twoSumPriest( float_st&a,  float_st&b, double_st&c, float_st&d)__attribute__((always_inline));
  inline friend void twoSumPriest( float_st&a,  double_st&b, float_st&c, double_st&d)__attribute__((always_inline));
  inline friend void twoSumPriest( float_st&a,  double_st&b, float_st&c, float_st&d)__attribute__((always_inline));
  inline friend void twoSumPriest( double_st&a,  float_st&b, float_st&c, double_st&d)__attribute__((always_inline));
  inline friend void twoSumPriest( double_st&a,  float_st&b, float_st&c, float_st&d)__attribute__((always_inline));
  inline friend void twoSumPriest( float_st&a,  float_st&b, float_st&c, float_st&d)__attribute__((always_inline));

#ifdef CADNA_FMA
  inline friend void twoProdFma( double_st&a,  double_st&b, double_st&res, double_st&err)__attribute__((always_inline));
  inline friend void twoProdFma( double_st&a,  double_st&b, double_st&res, float_st&err)__attribute__((always_inline));
  inline friend void twoProdFma( float_st&a,  double_st&b, double_st&res, double_st&err)__attribute__((always_inline));
  inline friend void twoProdFma( float_st&a,  double_st&b, double_st&res, float_st&err)__attribute__((always_inline));
  inline friend void twoProdFma( double_st&a,  float_st&b, double_st&res, double_st&err)__attribute__((always_inline));
  inline friend void twoProdFma( double_st&a,  float_st&b, double_st&res, float_st&err)__attribute__((always_inline));
  inline friend void twoProdFma( double_st&a,  double_st&b, float_st&res, double_st&err)__attribute__((always_inline));
  inline friend void twoProdFma( double_st&a,  double_st&b, float_st&res, float_st&err)__attribute__((always_inline));
  inline friend void twoProdFma( float_st&a,  float_st&b, double_st&res, double_st&err)__attribute__((always_inline));
  inline friend void twoProdFma( float_st&a,  float_st&b, double_st&res, float_st&err)__attribute__((always_inline));
  inline friend void twoProdFma( float_st&a,  double_st&b, float_st&res, double_st&err)__attribute__((always_inline));
  inline friend void twoProdFma( float_st&a,  double_st&b, float_st&res, float_st&err)__attribute__((always_inline));
  inline friend void twoProdFma( double_st&a,  float_st&b, float_st&res, double_st&err)__attribute__((always_inline));
  inline friend void twoProdFma( double_st&a,  float_st&b, float_st&res, float_st&err)__attribute__((always_inline));
  inline friend void twoProdFma( float_st&a,  float_st&b, float_st&res, float_st&err)__attribute__((always_inline));
#endif //CADNA_FMA
 
  inline friend void twoProd( double_st&a,  double_st&b, double_st&res, double_st&err)__attribute__((always_inline));
  inline friend void twoProd( double_st&a,  double_st&b, double_st&res, float_st&err)__attribute__((always_inline));
  inline friend void twoProd( float_st&a,  double_st&b, double_st&res, double_st&err)__attribute__((always_inline));
  inline friend void twoProd( float_st&a,  double_st&b, double_st&res, float_st&err)__attribute__((always_inline));
  inline friend void twoProd( double_st&a,  float_st&b, double_st&res, double_st&err)__attribute__((always_inline));
  inline friend void twoProd( double_st&a,  float_st&b, double_st&res, float_st&err)__attribute__((always_inline));
  inline friend void twoProd( double_st&a,  double_st&b, float_st&res, double_st&err)__attribute__((always_inline));
  inline friend void twoProd( double_st&a,  double_st&b, float_st&res, float_st&err)__attribute__((always_inline));
  inline friend void twoProd( float_st&a,  float_st&b, double_st&res, double_st&err)__attribute__((always_inline));
  inline friend void twoProd( float_st&a,  float_st&b, double_st&res, float_st&err)__attribute__((always_inline));
  inline friend void twoProd( float_st&a,  double_st&b, float_st&res, double_st&err)__attribute__((always_inline));
  inline friend void twoProd( float_st&a,  double_st&b, float_st&res, float_st&err)__attribute__((always_inline));
  inline friend void twoProd( double_st&a,  float_st&b, float_st&res, double_st&err)__attribute__((always_inline));
  inline friend void twoProd( double_st&a,  float_st&b, float_st&res, float_st&err)__attribute__((always_inline));
  inline friend void twoProd( float_st&a,  float_st&b, float_st&res, float_st&err)__attribute__((always_inline));
  
  
  // INTRINSIC FUNCTIONS
  
  friend half_st fabs(const half_st&);
  friend half_st abs(const half_st&);
  friend half_st floor(const half_st&);
  friend half_st ceil(const half_st&);
  friend half_st trunc(const half_st&);
  friend half_st nearbyint(const half_st&);
  friend half_st rint(const half_st&);
  friend long int  lrint(const half_st&);
  friend long long int llrint(const half_st&);
  friend int finite(const half_st&);
  friend int isfinite(const half_st&);
  friend int isnan(const half_st&);
  friend int isinf(const half_st&);
  
  // Conversion
  operator char();  
  operator unsigned char();  
  operator short();  
  operator unsigned short();  
  operator int();  
  operator unsigned int();  
  operator long();  
  operator unsigned long();  
  operator long long();  
  operator unsigned long long();  
  operator float();
  operator double();
  operator half();
  operator float_st();
  operator double_st();

  operator double_st() const;
  operator float_st() const;
  operator char() const;  
  operator unsigned char() const;  
  operator short() const;  
  operator unsigned short() const;  
  operator int() const;  
  operator unsigned int() const;  
  operator long() const;  
  operator unsigned long() const;  
  operator long long() const;  
  operator unsigned long long() const;  
  operator float() const;
  operator double() const;
  operator half() const; 

  inline int nb_significant_digit() const __attribute__((always_inline));  
  inline int approx_digit() const __attribute__((always_inline));
  inline int computedzero() const __attribute__((always_inline));
  inline int numericalnoise() const __attribute__((always_inline));
  void display() const ; 
  void display(const char *) const ; 
  char* str( char *)  const ;

  friend char* strp(const half_st&);
  friend char* str( char *, const half_st&);
  friend std::istream& operator >>(std::istream& s, half_st&);
  
  void data_st();
  void data_st(const double&, const int&);
  
};

std::ostream& operator<<(std::ostream&, const half_st&);
std::istream& operator >>(std::istream&, half_st& );
#endif

inline double_st::double_st (const float_st&a) : x(a.x),y(a.y),z(a.z),accuracy(a.accuracy) {};
#ifdef CADNA_HALF_EMULATION
inline double_st::double_st (const half_st&a) : x(a.x),y(a.y),z(a.z),accuracy(a.accuracy) {};
inline float_st::float_st (const half_st&a) : x(a.x),y(a.y),z(a.z),accuracy(a.accuracy) {};
#endif

#ifdef CADNA_HALF_NATIVE
inline double_st::double_st (const half_st&a) : x(a.x),y(a.y),z(a.z),accuracy(a.accuracy) {};
inline float_st::float_st (const half_st&a) : x(a.x),y(a.y),z(a.z),accuracy(a.accuracy) {};
#endif

void  cadna_init(int, unsigned int, unsigned int, unsigned int);
void  cadna_init(int, unsigned int);
void  cadna_init(int);
void  cadna_end();

void cadna_enable(unsigned int);
void cadna_disable(unsigned int);

inline void  instability(unsigned long*) __attribute__((always_inline));
#ifdef CADNA_HALF_EMULATION
extern double max_value;
#endif

#ifdef CADNA_HALF_NATIVE
extern double max_value;
#endif
// INSTABILITIES MANAGEMENT
extern unsigned long _cadna_div_count;
extern unsigned long _cadna_mul_count;  
extern unsigned long _cadna_power_count;  
extern unsigned long _cadna_branching_count; 
extern unsigned long _cadna_cancel_count;  
extern unsigned long _cadna_intrinsic_count;  
extern unsigned long _cadna_math_count; 
#ifdef CADNA_HALF_EMULATION
extern unsigned long _cadna_half_overflow_count; 
extern unsigned long _cadna_half_underflow_count;
#endif 
#ifdef CADNA_HALF_NATIVE
extern unsigned long _cadna_half_overflow_count; 
extern unsigned long _cadna_half_underflow_count;
#endif 

extern unsigned int _cadna_div_tag;
extern unsigned int _cadna_mul_tag;  
extern unsigned int _cadna_power_tag;  
extern unsigned int _cadna_branching_tag;
extern unsigned int _cadna_cancel_tag;
extern unsigned int _cadna_intrinsic_tag; 
extern unsigned int _cadna_math_tag; 
#ifdef CADNA_HALF_EMULATION
extern unsigned int _cadna_half_overflow_tag;
extern unsigned int _cadna_half_underflow_tag;
#endif
#ifdef CADNA_HALF_NATIVE
extern unsigned int _cadna_half_overflow_tag;
extern unsigned int _cadna_half_underflow_tag;
#endif

extern unsigned int _cadna_div_change; 
extern unsigned int _cadna_mul_change;  
extern unsigned int _cadna_power_change;  
extern unsigned int _cadna_branching_change;
extern unsigned int _cadna_cancel_change; 
extern unsigned int _cadna_intrinsic_change; 
extern unsigned int _cadna_math_change; 
#ifdef CADNA_HALF_EMULATION
extern unsigned int _cadna_half_overflow_change; 
extern unsigned int _cadna_half_underflow_change; 
#endif
#ifdef CADNA_HALF_NATIVE
extern unsigned int _cadna_half_overflow_change; 
extern unsigned int _cadna_half_underflow_change; 
#endif
extern int _cadna_cancel_value;

extern int nbre_trace_cadna;
extern int nbre_stability_count;

extern int  _cadna_max_instability;
extern int  _cadna_instability_detected;

// The RANDOM BIT GENERATOR
extern unsigned int _cadna_random;
extern int _cadna_random_counter;
extern unsigned int _cadna_recurrence;

#define RANDOM _cadna_random_function()


#define DIGIT_NOT_COMPUTED -1
#define RELIABLE_RESULT  255

typedef union u_floatbf {
  float fl;
  unsigned int bf;
} floatbf;

typedef union u_doublebf {
  double fl;
  unsigned long bf;
} doublebf;



#ifdef CADNA_HALF_EMULATION
typedef union u_halfbf {
    half fl;
    uint16_t bf;
} halfbf;
#endif //CADNA_HALF_EMULATION

#ifdef CADNA_HALF_NATIVE
typedef union u_halfbf {
    half fl;
    unsigned short int bf;
} halfbf;
#endif

#ifdef _OPENMP
#pragma omp threadprivate (_cadna_random, _cadna_random_counter, _cadna_recurrence)
#endif //_OPENMP

#ifdef _OPENMP
#pragma omp declare reduction(+:float_st : omp_out=omp_in+omp_out)	\
  initializer(omp_priv=float_st(0.f))
#pragma omp declare reduction(+:double_st: omp_out=omp_in+omp_out)	\
  initializer(omp_priv=double_st(0.))
#pragma omp declare reduction(-:float_st : omp_out=omp_in+omp_out)	\
  initializer(omp_priv=float_st(0.f))
#pragma omp declare reduction(-:double_st: omp_out=omp_in+omp_out)	\
  initializer(omp_priv=double_st(0.))
#pragma omp declare reduction(*:float_st : omp_out=omp_in*omp_out)	\
  initializer(omp_priv=float_st(1.f))
#pragma omp declare reduction(*:double_st: omp_out=omp_in*omp_out)	\
  initializer(omp_priv=double_st(1.))
#endif //_OPENMP

#define FLOAT_BIT_FLIP(f, b) ((floatbf*)&f)->bf^=((unsigned int)b)<<31;
#define DOUBLE_BIT_FLIP(d, b) ((doublebf*)&d)->bf^=((unsigned long)b)<<63;

#ifdef CADNA_HALF_EMULATION
#define HALF_BIT_FLIP(f, b) ((halfbf*)&f)->bf^=((uint16_t)b)<<15;
#endif //CADNA_HALF_EMULATION

#ifdef CADNA_HALF_NATIVE
#define HALF_BIT_FLIP(f, b) ((halfbf*)&f)->bf^=((unsigned short int)b)<<15;
#endif 

#include <cadna_unstab.h>
#include <cadna_random.h>
#include <cadna_digitnumber.h>
#include <cadna_computedzero.h>
#include <cadna_numericalnoise.h>
#include <cadna_to.h>
#include <cadna_add.h>
#include <cadna_sub.h>
#include <cadna_mul.h>
#include <cadna_div.h>

#include <cadna_fma.h>

//compensated methods:
#include <cadna_comp.h>


//#ifndef __CADNA_LIB__

#include <cadna_eq.h>
#include <cadna_ne.h>
#include <cadna_ge.h>
#include <cadna_gt.h>
#include <cadna_le.h>
#include <cadna_lt.h>
#include <cadna_str.h>


//#endif  __CADNA_LIB 



#endif











