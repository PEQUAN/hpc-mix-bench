/*This file is part of PROMISE.

	PROMISE is free software: you can redistribute it and/or modify it
	under the terms of the GNU Lesser General Public License as
	published by the Free Software Foundation, either version 3 of the
	License, or (at your option) any later version.

	PROMISE is distributed in the hope that it will be useful, but WITHOUT
	ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
	or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General
	Public License for more details.

	You should have received a copy of the GNU Lesser General Public
	License along with PROMISE. If not, see
	<http://www.gnu.org/licenses/>.

Promise v1 was written by Romain Picot
Promise v2 has been written from v1 by Thibault Hilaire and Sara Hoseininasab
Promise v3 has been written from v2 by Thibault Hilaire, Fabienne JÉZÉQUEL and Xinye Chen
	Promise v3 enables version check, pip installing, arbitrary precision, etc., features.
	Sorbonne Université
	LIP6 (Computing Science Laboratory)
	Paris, France. Contact: thibault.hilaire@lip6.fr, fabienne.jezequel@lip6.fr, xinyechenai@gmail.com


Contains some macros used to check the precision (and compare with the expectations):
- PROMISE_CHECK_VAR(v)   to check the value of a variable v
- PROMISE_CHECK_ARRAY(x,n)  to check the values of an array x of size n
- PROMISE_CHECK_ARRAY2D(x,n,m)    to check the values of an 2D-array x of size n by m (added by Xinye Chen, xinyechenai@gmail.com)

THe same as promise.h, but without the function code

© Thibault Hilaire and Fabienne JÉZÉQUEL, April 2024
*/


#ifndef __PROMISE_DUMP__
#define __PROMISE_DUMP__

#include <stdio.h>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <type_traits>
#include <limits>
#include <cmath>
#include <half_promise.hpp>
#include <floatx.hpp>

/* These two macros export the value(s) to check to Promise2
Usage:
- `PROMISE_CHECK_VAR(var);`     to export the variable `var`
- `PROMISE_CHECK_ARRAY(var);`   to export the n values of the array `var`
*/
#define PROMISE_CHECK_VAR(var) promise_dump(#var, &var, 1)
#define PROMISE_CHECK_ARRAY(var, n) promise_dump(#var, (var), n)
#define PROMISE_CHECK_ARRAY2D(var, n, m) promise_dump_arr(#var, (var), n, m)

/* dump (to stdout) a variable:
 - varName: name of the variable dump (obtained with a C macro)
 - a: pointer to the variable (scalar or array) to dump
 - size: size of the array (1 for a scalar)*/
template<typename T>
void promise_dump(char const* varName, T* a, std::size_t size){
	/* export the stochastic double and significant digits */
	for(std::size_t i=0; i<size; i++){
		std::cout << "[PROMISE_DUMP] " << varName;
		if (size>1)
			std::cout << "[" << i << "]";
		std::cout << " = ";

		// Handle special values before applying hexfloat
		if constexpr (std::is_floating_point_v<T>) {
			double val_as_double = static_cast<double>(a[i]);
			if (std::isnan(val_as_double)) {
				std::cout << "nan";
			} else if (std::isinf(val_as_double)) {
				std::cout << (a[i] > static_cast<T>(0) ? "+inf" : "-inf");
			} else {
				// Normal value: use hexfloat for exact rep
				std::cout << std::hexfloat << a[i] << std::defaultfloat;
			}
		} else {
			// Non-float types: print as-is (no hexfloat)
			std::cout << a[i];
		}
		std::cout << std::endl;
	}
}

template<typename T>
void promise_dump_arr(char const* varName, T** a, std::size_t rows, std::size_t cols){
	/* export the stochastic double and significant digits */
	for(std::size_t i=0; i<rows; i++){
		promise_dump(varName, a[i], cols);
	}
}

/* explicitly instantiates the promise_dump function for half, single and double */
template void promise_dump<float>(char const*, float*, std::size_t);
template void promise_dump<double>(char const*, double*, std::size_t);
template void promise_dump<half_float::half>(char const*, half_float::half*, std::size_t);
template void promise_dump<flx::floatx<5, 10>>(char const*, flx::floatx<5, 10>*, std::size_t);
template void promise_dump_arr<float>(char const*, float**, std::size_t, std::size_t);
template void promise_dump_arr<double>(char const*, double**, std::size_t, std::size_t);
template void promise_dump_arr<half_float::half>(char const*, half_float::half**, std::size_t, std::size_t);
template void promise_dump_arr<flx::floatx<5, 10>>(char const*, flx::floatx<5, 10>**, std::size_t, std::size_t);

#ifndef __CADNA__
extern const char* strp(double a){
  return "";
}

extern const char* strp(half_float::half a){
  return "";
} 

extern const char* strp(flx::floatx<5, 10> a){
	return "";
} 
  

extern void cadna_init(int i){
  return ;
}

extern void cadna_end(){
  return ;
}
#endif

#ifdef __CADNA__
/* dump (to stdout) a variable, *but* for a stochastic variable:
 - varName: name of the variable dump (obtained with a C macro)
 - a: pointer to the variable (scalar or array) to dump
 - size: size of the array (1 for a scalar)*/
void promise_dump(char const* varName, double_st* a, std::size_t size){
	/* export the stochastic double and significant digits */
	for(std::size_t i=0; i<size; i++){
	    /* display name  */
		std::cout << "[PROMISE_DUMP_ST] " << varName;
		if (size>1)
			std::cout << "[" << i << "]";
		std::cout << " = ";

		// Check if all components are NaN
		double x_d = static_cast<double>(a[i].getx());
		double y_d = static_cast<double>(a[i].gety());
		double z_d = static_cast<double>(a[i].getz());
		if (std::isnan(x_d) && std::isnan(y_d) && std::isnan(z_d)) {
			std::cout << "(nan, nan, nan)";
		} else {
			/* display stochastic values (using `hexfloat` output) */
			std::cout << std::hexfloat << "(" << a[i].getx() << "," << a[i].gety() << "," << a[i].getz() << ")" << std::defaultfloat;
		}
		/* display number of significant digits */
		std::cout << ", nb significant digits=" << a[i].nb_significant_digit() << std::endl;
	}
}

void promise_dump(char const* varName, float_st* a, std::size_t size){
	/* export the stochastic float and significant digits */
	for(std::size_t i=0; i<size; i++){
	    /* display name  */
		std::cout << "[PROMISE_DUMP_ST] " << varName;
		if (size>1)
			std::cout << "[" << i << "]";
		std::cout << " = ";

		// Check if all components are NaN
		double x_d = static_cast<double>(a[i].getx());
		double y_d = static_cast<double>(a[i].gety());
		double z_d = static_cast<double>(a[i].getz());
		if (std::isnan(x_d) && std::isnan(y_d) && std::isnan(z_d)) {
			std::cout << "(nan, nan, nan)";
		} else {
			/* display stochastic values (using `hexfloat` output) */
			std::cout << std::hexfloat << "(" << a[i].getx() << "," << a[i].gety() << "," << a[i].getz() << ")" << std::defaultfloat;
		}
		/* display number of significant digits */
		std::cout << ", nb significant digits=" << a[i].nb_significant_digit() << std::endl;
	}
}

// Remove redundant forward declaration (function already defined above)

void promise_dump_arr(char const* varName, double** a, std::size_t rows, std::size_t cols){
	for(std::size_t i=0; i<rows; i++){
	    /* display name  */
		promise_dump(varName, a[i], cols);
	}
}
#endif
#endif