#                                               -*- Autoconf -*-	
# Process this file with autoconf to produce a configure script.

AC_PREREQ([2.71])
AC_INIT([CADNA_C_HALF],[3.1.12],[cadna-team@lip6.fr])
AC_CONFIG_AUX_DIR(admin)

AC_CONFIG_HEADERS(config.h)
AM_INIT_AUTOMAKE
AC_CONFIG_MACRO_DIR([./m4])

# Checks for programs.

# USER_XXXFLAGS  allows to avoid -g -O2
USER_CFLAGS="$CFLAGS"
USER_CXXFLAGS="$CXXFLAGS"
USER_FCFLAGS="$FCFLAGS"


AC_PROG_CC
AC_PROG_CXX
AC_PROG_FC 
AC_PROG_MAKE_SET
AM_PROG_AR
AM_PROG_AS
AC_PROG_INSTALL
AC_PROG_RANLIB

CFLAGS="$USER_CFLAGS"
CXXFLAGS="$USER_CXXFLAGS"
FCFLAGS="$USER_FCFLAGS"


# Parameters for 64 bits architecture
architecture="64"
architecture_m=""
architecture_q="-q64"

CADNA_MATH_FILE=srcC/cadna_math_gnu.cc	

AC_MSG_CHECKING([if 32bit support is requested])
AC_ARG_ENABLE(32bit,[  --enable-32bit          enable 32bit support (where applicable)],
		       [ 
		       architecture="32" 
                       architecture_m="-m32"
                       architecture_q="-q32"
		       CADNA_MATH_FILE=srcC/cadna_math_other.cc		
		       ])
AC_DEFINE_UNQUOTED(CADNA_ARCHITECTURE,  $architecture , [ "Architecture 32bit or 64bit" ] )
AC_MSG_RESULT([ $architecture bit support])


#inserted by FJ
AC_MSG_CHECKING([if FMA support is requested])
AC_ARG_ENABLE(fma,[  --enable-fma          enable fma support (where applicable)])
AC_MSG_RESULT($enable_fma)
#

AC_MSG_CHECKING([if half-emulation support is requested])
AC_ARG_ENABLE(half-emulation,[  --enable-half-emulation          enable half-emulation support (where applicable)])
AC_MSG_RESULT($enable_half_emulation)


AC_MSG_CHECKING([if half-native support is requested])
AC_ARG_ENABLE(half-native,[  --enable-half-native          enable half-native support (where applicable)])
AC_MSG_RESULT($enable_half_native)

#Modif ordre des compilateurs IDRIS/PC
case "$CC" in
	  *pgcc*)
	  AC_SUBST([AM_CFLAGS], ["-Kieee"])			
              ;;	      	       
          *gcc*)
	  AC_SUBST([AM_CFLAGS], ["-Wall -frounding-math $architecture_m"])		  
	  ;;
          *clang)
	  AC_SUBST([AM_CFLAGS], ["-Wall $architecture_m"])		
              ;;
	  *icc*)
	  AC_SUBST([AM_CFLAGS], ["-fp-modelstrict -fp-modelno-except"])			
              ;;
	  *xlc*)
	  AC_SUBST([AM_CFLAGS], ["-qstrict=ieeefp -qfloat=rrm $architecture_q"])			
              ;;
esac;

#Modif ordre des compilateurs IDRIS/PC
case "$CXX" in
	  *pgc++)
	  AC_SUBST([AM_CXXFLAGS], ["-Kieee"])			
           ;;	      	      
          *clang++)
	  AC_SUBST([AM_CXXFLAGS], [" -Wall $architecture_m"])		 			
          ;;
          *g++*)
	  AC_SUBST([AM_CXXFLAGS], ["-Wall -frounding-math $architecture_m"])
	  ;;
	   *icpc*)
	  AC_SUBST([AM_CXXFLAGS], ["-fp-modelstrict -fp-modelno-except"])			
          ;;
	  *xlC*)
	  AC_SUBST([AM_CXXFLAGS], ["-qstrict=ieeefp -qfloat=rrm -q$architecture"])			
          ;;
esac;


#Modif ordre des compilateurs IDRIS/PC
case "$FC" in
	  *pgfortran | pgf90)
	  AC_SUBST([AM_FCFLAGS], ["-Kieee"])			
          ;;	      	      
          *gfortran*)
	  AC_SUBST([AM_FCFLAGS], ["-Wall -frounding-math $architecture_m"])	  
	  ;;
	  *ifort*)
	  AC_SUBST([AM_FCFLAGS], ["-fp-modelstrict -fp-modelno-except" ])		  
	  ;;
	  *xlf*)
	  AC_SUBST([AM_FCFLAGS], ["-qstrict=ieeefp -qfloat=rrm $architecture_q"])			
          ;;	      	      

esac;



echo "CFLAGS=$CFLAGS"
echo "AM_CFLAGS=$AM_CFLAGS"

echo "CXXFLAGS=$CXXFLAGS"
echo "AM_CXXFLAGS=$AM_CXXFLAGS"

echo "FCFLAGS=$FCFLAGS"
echo "AM_FCFLAGS=$AM_FCFLAGS"



AC_CHECK_SIZEOF([unsigned long]) 
AC_CHECK_SIZEOF([unsigned int]) 
AC_CHECK_SIZEOF([float]) 
AC_CHECK_SIZEOF([double]) 

# Test to know if the sizeof(double)==sizeof( unsigned long)==8  and
# sizeof(float)==sizeof(int)==4  

AS_IF([test "$ac_cv_sizeof_unsigned_long" -ne 8 -o "$ac_cv_sizeof_unsigned_int" -ne 4 -o  \
	    "$ac_cv_sizeof_double" -ne 8 -o  "$ac_cv_sizeof_float" -ne 4 ],
    [AC_MSG_ERROR(INTEGER SIZEOF is not compatible with CADNA source code)]) 

rm -f srcC/cadna_plugin.h
touch srcC/cadna_plugin.h
rm -rf tmp_cadna__

#block inserted by FJ
#  -mfma suppressed because unrecognized
#   CXXFLAGS="$CXXFLAGS -std=c++11 -mfma -ffp-contract=off -D_CADNA_FMA"
AS_IF([test "$enable_fma" = "yes"], [
   CXXFLAGS="$CXXFLAGS -std=c++11 -ffp-contract=off -D_CADNA_FMA"
   (echo "#define CADNA_FMA"; cat srcC/cadna_plugin.h) > tmp_cadna__
   mv tmp_cadna__   srcC/cadna_plugin.h
])
#

AS_IF([test "$enable_half_emulation" = "yes"], [
   	AC_CHECK_FILE(srcC/half.hpp, [sed -i 's/#define HALF_ROUND_STYLE	1/#define HALF_ROUND_STYLE	2/g' srcC/half.hpp],
	[AC_MSG_ERROR([cannot find half.hpp, download it from http://half.sourceforge.net/ and put it in srcC directory])])

   CXXFLAGS="$CXXFLAGS -std=c++11"
   (echo "#define CADNA_HALF_EMULATION"; cat srcC/cadna_plugin.h) > tmp_cadna__
   mv tmp_cadna__   srcC/cadna_plugin.h
])

AS_IF([test "$enable_half_native" = "yes"], [
       CXXFLAGS="$CXXFLAGS -march=armv8.2-a+fp16"
       (echo "#define CADNA_HALF_NATIVE"; cat srcC/cadna_plugin.h) > tmp_cadna__
   mv tmp_cadna__   srcC/cadna_plugin.h
])

AC_MSG_CHECKING([for CADNA fortran library])
AC_ARG_ENABLE(fortran,[  --enable-fortran          enable fortran support (where applicable)],cadnafortran="yes",cadnafortran="no")
AC_MSG_RESULT($architecture)

AM_CONDITIONAL([HALF_EMULATION], [test "$enable_half_emulation" = "yes" ])

#AC_CHECK_HEADERS(math.h,,
#[AC_MSG_ERROR([cannot find math.h])])
#
# AC_CHECK_LIB(m, acos,AC_MSG_RESULT([]),AC_MSG_ERROR(You need libm))
# AC_CHECK_LIB(m, acosf,AC_MSG_RESULT([]),AC_MSG_ERROR(You need libm with functions on float (powf, cosf, expf, ....)))
# AC_CHECK_LIB(m,acosh,AC_MSG_RESULT([]),AC_MSG_ERROR(You need libm))
# AC_CHECK_LIB(m,acoshf,AC_MSG_RESULT([]),AC_MSG_ERROR(You need libm with functions on float (powf, cosf, expf, ....)))
# AC_CHECK_LIB(m,asin,AC_MSG_RESULT([]),AC_MSG_ERROR(You need libm))
# AC_CHECK_LIB(m,asinf,AC_MSG_RESULT([]),AC_MSG_ERROR(You need libm with functions on float (powf, cosf, expf, ....)))
# AC_CHECK_LIB(m,asinh,AC_MSG_RESULT([]),AC_MSG_ERROR(You need libm))
# AC_CHECK_LIB(m,asinhf,AC_MSG_RESULT([]),AC_MSG_ERROR(You need libm with functions on float (powf, cosf, expf, ....)))
# AC_CHECK_LIB(m,atan,AC_MSG_RESULT([]),AC_MSG_ERROR(You need libm))
# AC_CHECK_LIB(m,atan2,AC_MSG_RESULT([]),AC_MSG_ERROR(You need libm with functions on float (powf, cosf, expf, ....)))
# AC_CHECK_LIB(m,atan2f,AC_MSG_RESULT([]),AC_MSG_ERROR(You need libm))
# AC_CHECK_LIB(m,atanf,AC_MSG_RESULT([]),AC_MSG_ERROR(You need libm with functions on float (powf, cosf, expf, ....)))
# AC_CHECK_LIB(m,atanh,AC_MSG_RESULT([]),AC_MSG_ERROR(You need libm))
# AC_CHECK_LIB(m,cbrt,AC_MSG_RESULT([]),AC_MSG_ERROR(You need libm))
# AC_CHECK_LIB(m,cbrtf,AC_MSG_RESULT([]),AC_MSG_ERROR(You need libm with functions on float (powf, cosf, expf, ....)))
# AC_CHECK_LIB(m,cos,AC_MSG_RESULT([]),AC_MSG_ERROR(You need libm))
# AC_CHECK_LIB(m,cosf,AC_MSG_RESULT([]),AC_MSG_ERROR(You need libm with functions on float (powf, cosf, expf, ....)))
# AC_CHECK_LIB(m,cosh,AC_MSG_RESULT([]),AC_MSG_ERROR(You need libm))
# AC_CHECK_LIB(m,coshf,AC_MSG_RESULT([]),AC_MSG_ERROR(You need libm with functions on float (powf, cosf, expf, ....)))
# AC_CHECK_LIB(m,exp,AC_MSG_RESULT([]),AC_MSG_ERROR(You need libm))
# AC_CHECK_LIB(m,exp2,AC_MSG_RESULT([]),AC_MSG_ERROR(You need libm with functions on float (powf, cosf, expf, ....)))
# AC_CHECK_LIB(m,exp2f,AC_MSG_RESULT([]),AC_MSG_ERROR(You need libm))
# AC_CHECK_LIB(m,expf,AC_MSG_RESULT([]),AC_MSG_ERROR(You need libm with functions on float (powf, cosf, expf, ....)))
# AC_CHECK_LIB(m,expm1,AC_MSG_RESULT([]),AC_MSG_ERROR(You need libm))
# AC_CHECK_LIB(m,expm1f,AC_MSG_RESULT([]),AC_MSG_ERROR(You need libm with functions on float (powf, cosf, expf, ....)))
# AC_CHECK_LIB(m,fmax,AC_MSG_RESULT([]),AC_MSG_ERROR(You need libm))
# AC_CHECK_LIB(m,fmaxf,AC_MSG_RESULT([]),AC_MSG_ERROR(You need libm with functions on float (powf, cosf, expf, ....)))
# AC_CHECK_LIB(m,fmin,AC_MSG_RESULT([]),AC_MSG_ERROR(You need libm))
# AC_CHECK_LIB(m,fminf,AC_MSG_RESULT([]),AC_MSG_ERROR(You need libm with functions on float (powf, cosf, expf, ....)))
# AC_CHECK_LIB(m,hypot,AC_MSG_RESULT([]),AC_MSG_ERROR(You need libm))
# AC_CHECK_LIB(m,hypotf,AC_MSG_RESULT([]),AC_MSG_ERROR(You need libm with functions on float (powf, cosf, expf, ....)))
# AC_CHECK_LIB(m,log,AC_MSG_RESULT([]),AC_MSG_ERROR(You need libm))
# AC_CHECK_LIB(m,log10,AC_MSG_RESULT([]),AC_MSG_ERROR(You need libm with functions on float (powf, cosf, expf, ....)))
# AC_CHECK_LIB(m,log10f,AC_MSG_RESULT([]),AC_MSG_ERROR(You need libm))
# AC_CHECK_LIB(m,log1p,AC_MSG_RESULT([]),AC_MSG_ERROR(You need libm with functions on float (powf, cosf, expf, ....)))
# AC_CHECK_LIB(m,log1pf,AC_MSG_RESULT([]),AC_MSG_ERROR(You need libm))
# AC_CHECK_LIB(m,log2,AC_MSG_RESULT([]),AC_MSG_ERROR(You need libm with functions on float (powf, cosf, expf, ....)))
# AC_CHECK_LIB(m,log2f,AC_MSG_RESULT([]),AC_MSG_ERROR(You need libm))
# AC_CHECK_LIB(m,logb,AC_MSG_RESULT([]),AC_MSG_ERROR(You need libm with functions on float (powf, cosf, expf, ....)))
# AC_CHECK_LIB(m,logbf,AC_MSG_RESULT([]),AC_MSG_ERROR(You need libm))
# AC_CHECK_LIB(m,logf,AC_MSG_RESULT([]),AC_MSG_ERROR(You need libm with functions on float (powf, cosf, expf, ....)))
# AC_CHECK_LIB(m,pow,AC_MSG_RESULT([]),AC_MSG_ERROR(You need libm))
# AC_CHECK_LIB(m,powf,AC_MSG_RESULT([]),AC_MSG_ERROR(You need libm with functions on float (powf, cosf, expf, ....)))
# AC_CHECK_LIB(m,sin,AC_MSG_RESULT([]),AC_MSG_ERROR(You need libm))
# AC_CHECK_LIB(m,sinf,AC_MSG_RESULT([]),AC_MSG_ERROR(You need libm with functions on float (powf, cosf, expf, ....)))
# AC_CHECK_LIB(m,sinh,AC_MSG_RESULT([]),AC_MSG_ERROR(You need libm))
# AC_CHECK_LIB(m,sinhf,AC_MSG_RESULT([]),AC_MSG_ERROR(You need libm with functions on float (powf, cosf, expf, ....)))
# AC_CHECK_LIB(m,sqrt,AC_MSG_RESULT([]),AC_MSG_ERROR(You need libm))
# AC_CHECK_LIB(m,sqrtf,AC_MSG_RESULT([]),AC_MSG_ERROR(You need libm with functions on float (powf, cosf, expf, ....)))
# AC_CHECK_LIB(m,tan,AC_MSG_RESULT([]),AC_MSG_ERROR(You need libm))
# AC_CHECK_LIB(m,tanf,AC_MSG_RESULT([]),AC_MSG_ERROR(You need libm with functions on float (powf, cosf, expf, ....)))
# AC_CHECK_LIB(m,tanh,AC_MSG_RESULT([]),AC_MSG_ERROR(You need libm))
# AC_CHECK_LIB(m,tanhf,AC_MSG_RESULT([]),AC_MSG_ERROR(You need libm with functions on float (powf, cosf, expf, ....)))
# AC_CHECK_LIB(m,tanf,AC_MSG_RESULT([]),AC_MSG_ERROR(You need libm with functions on float (powf, cosf, expf, ....)))
# AC_CHECK_LIB(m,tanh,AC_MSG_RESULT([]),AC_MSG_ERROR(You need libm))
# AC_CHECK_LIB(m,tanhf,AC_MSG_RESULT([]),AC_MSG_ERROR(You need libm with functions on float (powf, cosf, expf, ....)))
# AC_CHECK_LIB(m,frexp,AC_MSG_RESULT([]),AC_MSG_ERROR(You need libm))
# AC_CHECK_LIB(m,frexpf,AC_MSG_RESULT([]),AC_MSG_ERROR(You need libm with functions on float (frexpf, ....)))
# AC_CHECK_LIB(m,ldexp,AC_MSG_RESULT([]),AC_MSG_ERROR(You need libm))
# AC_CHECK_LIB(m,ldexpf,AC_MSG_RESULT([]),AC_MSG_ERROR(You need libm with functions on float (ldexpf, ....)))
# AC_CHECK_LIB(m,fmod,AC_MSG_RESULT([]),AC_MSG_ERROR(You need libm))
# AC_CHECK_LIB(m,fmodf,AC_MSG_RESULT([]),AC_MSG_ERROR(You need libm with functions on float (fmodf, ....)))
# AC_CHECK_LIB(m,modf,AC_MSG_RESULT([]),AC_MSG_ERROR(You need libm))
# AC_CHECK_LIB(m,modff,AC_MSG_RESULT([]),AC_MSG_ERROR(You need libm with functions on float (fmodf, ....)))
#
#

AC_CONFIG_FILES([
	Makefile
	srcC/Makefile
	srcC_mpi/Makefile
	srcFortran/Makefile
	srcFortran_mpi/Makefile
	examplesC/Makefile
	examplesC_mpi/Makefile
	examplesC_omp/Makefile
	examplesC_mpiomp/Makefile	
	examplesFortran/Makefile
	examplesFortran_mpi/Makefile
    examples_half/Makefile
	])

AC_CANONICAL_HOST

__cadna_compiler_problem=1	

CXXCADNAOPTIMFLAGS="-O3"
CXXCADNADEBUGFLAGS="-O0 -g "
 


################
# C  FLAGS test
################

AC_LANG_PUSH([C])
echo "CHECK C"
#Modif IDRIS/PC
#CADNA_CHECK_COMPILER_FLAG([ "$CFLAGS" ],[CFLAGS])
CADNA_CHECK_C_COMPILER_FLAG([ "$CFLAGS" ],[CFLAGS])
#CADNA_CHECK_COMPILER_FLAG([ "$AM_CFLAGS" ],[AM_CFLAGS])
CADNA_CHECK_C_COMPILER_FLAG([ "$AM_CFLAGS" ],[AM_CFLAGS])

AC_LANG_POP




################
# CXXFLAGS test
################

AC_LANG_PUSH([C++])

CADNA_CHECK_COMPILER_FLAG([ "$CXXFLAGS" ],[CXXFLAGS])
CADNA_CHECK_COMPILER_FLAG([ "$AM_CXXFLAGS" ],[AM_CXXFLAGS])

CADNA_CHECK_COMPILER_FLAG([ "$CXXCADNAOPTIMFLAGS" ],[CXXCADNAOPTIMFLAG])
CADNA_CHECK_COMPILER_FLAG([ "$CXXCADNADEBUGFLAGS" ],[CXXCADNADEBUGFLAG])


################ find the C++ standard lib

CADNA_CHECK_CXX_LIB(["the c++ standard  library : -lc++ or -lstdc++"])
echo "CADNA_CXX_LIB = $CADNA_CXX_LIB"
echo $CXXFLAGS

AC_LANG_POP



######################## FORTRAN

if test "x$cadnafortran" = "xyes" ;
then

echo "FORTRAN1 : $FC $cadnafortran "
AC_LANG_PUSH([Fortran])
AC_MSG_CHECKING([whether $FC is available and compiles a fortran program])
AC_COMPILE_IFELSE([AC_LANG_PROGRAM([],[])],
                  [AC_MSG_RESULT([yes]) ],
                 [AC_MSG_ERROR([Fortran  compiler not available or unable to compile])])

CADNA_CHECK_FC_COMPILER_FLAG([ $AM_FCFLAGS ], [AM_FCFLAGS])
CADNA_CHECK_FC_COMPILER_FLAG([ $FCFLAGS ], [FCFLAGS])

AC_LANG_POP
fi

echo "FORTRAN : $FC $cadnafortran "
AM_CONDITIONAL([CADNA_FORTRAN], [test x$cadnafortran  = "xyes" ]  )


##################### OPENMP

use_openmp="no" 

######### TEST OPENMP with C
AC_LANG_PUSH([C])
C_SAVED_FLAGS=$CFLAGS	
CXX_SAVED_FLAGS=$CXXFLAGS  #added by S. Hoseininasab for half-precision

########## test for specific OpenMP_CFLAGS	
if test "x$OPENMP_CFLAGS" != "x" -o  "x$OPENMP_LDFLAGS" != "x" ; then
  	#echo "LANCEMENT       cadna_check_openmp_flags  "	
        CADNA_CHECK_OPENMP_FLAGS
         if test "x$use_openmp" == "xno"; then
            	 AC_MSG_FAILURE(Specific OpenMP FLAGS are unusable)
	else 
		use_openmp="yes"
	fi   	  
else   
     	  ########## test for standard OpenMP_CXXFLAGS	
            
     	  AC_OPENMP
	  #echo "ac_cv_prog_cc_openmp $ac_cv_prog_c_openmp"
	  if test "x$ac_cv_prog_c_openmp" != "xunsupported"; then 
	      use_openmp="yes"
	  fi
	   AC_MSG_RESULT( Test OPENMP $CC : $use_openmp)
fi


CFLAGS=$C_SAVED_FLAGS

AC_LANG_POP

if test "x$ac_cv_prog_c_openmp" != "xunsupported"; then
	########## TEST OPENMP with CXX
  	AC_LANG_PUSH([C++])
  	 ########## test for specific OpenMP_CXXFLAGS	
 	 if test "x$OPENMP_CXXFLAGS" != "x" -o  "x$OPENMP_LDFLAGS" != "x"; then
     	     #echo "LANCEMENT       cadna_check_openmp_flags   _AC_LANG_ABBREV $OPENMP_CXXFLAGS "	
    	      CADNA_CHECK_OPENMP_FLAGS
    	      if test "x$use_openmp" == "xno"; then
              	  AC_MSG_FAILURE(Specific OpenMP FLAGS are unusable)
	     else 
	     	   use_openmp="yes"
		    AC_MSG_RESULT( Test OPENMP $CXX : $use_openmp)
	      fi   	  
	 else   
     	 	 ########## test for standard OpenMP_CXXFLAGS	
     	        echo "Test OPENMP _AC_LANG_ABBREV"    
      		AC_OPENMP
		#echo "Test OPENMP _AC_LANG_ABBREV : $ac_cv_prog_cxx_openmp"	
		if test "x$ac_cv_prog_cxx_openmp" != "xunsupported"; then 
		   	use_openmp="yes"
		else
			use_openmp="no"
		fi
	fi
	# OpenMP is OK     	
	 if test "x$use_openmp" == "xyes"; then
	       # is Openmp4 ?    
	       # echo "OPENMP4  OpenMP_CXXFLAGS :  $OPENMP_CXXFLAGS"
		CADNA_CHECK_OPENMP4	
	
		if test "x$use_openmp" = "xyes"; then	
		    CADNA_CHECK_OPENMP_ROUNDINGMODE
        	   # if not use_openmp is set to no	
 	        fi		  
         fi
      AC_LANG_POP

fi 


CXXFLAGS=$CXX_SAVED_FLAGS
   	

AM_CONDITIONAL([CADNA_OPENMP], [test "x$use_openmp" = "xyes" ])
	   


##################### MPI

AC_LANG_PUSH([C])
AX_MPI( 
[ use_mpicc=yes 
 AC_MSG_RESULT([MPI compiler is OK ])  ],
[ use_mpicx=no 
  AC_MSG_WARN([No MPI compiler found, won't use MPI.]) ])
AC_LANG_POP

AC_LANG_PUSH([C++])
AX_MPI( 
[ use_mpicxx=yes 
 AC_MSG_RESULT([MPI compiler is OK ])  ],
[ use_mpicxx=no 
  AC_MSG_WARN([No MPI compiler found, won't use MPI.]) ])
AC_LANG_POP
echo "Suite MPI"



AC_LANG_PUSH([Fortran])
AX_MPI( 
[ use_mpif90=yes 
 AC_MSG_RESULT([MPI compiler is OK ])  ],
[ use_mpif90=no 
  AC_MSG_WARN([No MPI compiler found, won't use MPI.]) ])
AC_LANG_POP
echo "Suite MPI"




 echo "Openmp ${use_openmp}   MPI ${use_mpi}    $MPICXX $MPICC"


############################
AC_MSG_CHECKING([for compatibility between $CXX and $MPICXX ])

cat > conftest.cpp << EOF
#include <iostream>
int main(){
  using  namespace std;
  cout << endl;
  return 0;
}
EOF

$CXX conftest.cpp -c 
if $MPICXX  conftest.o    2>&1 | grep -iE "error|conftest.o" >/dev/null
then 
    use_mpicc=no	
    use_mpicxx=no	 
    AC_MSG_RESULT([NO. ])	 
    AC_MSG_RESULT([***])
    AC_MSG_RESULT([***])
    AC_MSG_RESULT([***])
    AC_MSG_RESULT([MPI part of CADNA will not be generated ])
    AC_MSG_RESULT([***])
    AC_MSG_RESULT([***])
    AC_MSG_RESULT([***])
    	 
else
   AC_MSG_RESULT([yes])    
fi
######################

AM_CONDITIONAL([CADNA_MPI], [test x$use_mpicc = xyes  -a x$use_mpicxx = xyes ])
AM_CONDITIONAL([CADNA_FORTRAN_MPI], [test x$use_mpicc = xyes  -a x$use_mpicxx = xyes -a x$use_mpif90  = "xyes" ]  ) 

AC_SUBST([AM_CFLAGS])
AC_SUBST([AM_CXXFLAGS])
AC_SUBST([AM_FCFLAGS])

AC_SUBST([CFLAGS])
AC_SUBST([CXXFLAGS])
AC_SUBST([FCFLAGS])

AC_SUBST([FCCADNACOMMONFLAGS])
AC_SUBST([CXXCADNAOPTIMFLAGS])
AC_SUBST([CXXCADNADEBUGFLAGS])	  

AC_SUBST([OPENMP_CFLAGS])	   
AC_SUBST([OPENMP_CXXFLAGS])	   
AC_SUBST([OPENMP_LDFLAGS])	   
AC_SUBST([CADNA_CXX_LIB])

echo "CFLAGS=$CFLAGS" 
echo "CXXFLAGS=$CXXFLAGS" 
echo "FCFLAGS=$FCFLAGS"
echo "CXXCADNAOPTIMFLAGS=$CXXCADNAOPTIMFLAGS"
echo "CXXCADNADEBUGFLAGS	  =$CXXCADNADEBUGFLAGS	  "

echo "OPENMP_CFLAGS	   =$OPENMP_CFLAGS	   "
echo "OPENMP_CXXFLAGS	   =$OPENMP_CXXFLAGS	   "
echo "OPENMP_LDFLAGS	   =$OPENMP_LDFLAGS	   "

echo "CADNA_CXX_LIB=$CADNA_CXX_LIB"

AC_CONFIG_LINKS([srcC/cadna_math.cc:$CADNA_MATH_FILE])

AC_OUTPUT


