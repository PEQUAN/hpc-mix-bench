
AC_DEFUN([CADNA_CHECK_OPENMP4],[


AC_MSG_CHECKING([OpenMP v4"])

CADNA_CHECK_OPENMP4_CXXFLAGS="$CXXFLAGS"

CXXFLAGS="$CXXFLAGS $AM_CXXFLAGS $OPENMP_CXXFLAGS   -Werror $OPENMP_LDFLAGS "



AC_RUN_IFELSE([
	AC_LANG_PROGRAM(
		[[ #include "omp.h" ]], 
		[[int i,sum=0;
	#pragma omp declare reduction(add:int : omp_out=omp_in+omp_out)	\
		initializer(omp_priv=0) 
	#pragma omp parallel for reduction(add:sum)
		for (i=0; i<10; i++) sum+=i;
		    return sum-45;]])],
           	    [
		    use_openmp="yes"
                    CXXCADNACOMMONFLAGS="$CXXCADNACOMMONFLAGS  "
               	    AC_MSG_RESULT(yes)	
                    ],
                    [
		     use_openmp="no" 
		     AC_MSG_RESULT(no)
               	    ])

])

CXX_FLAGS="$CADNA_CHECK_OPENMP4_CXXFLAGS"

