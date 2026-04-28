
AC_DEFUN([CADNA_CHECK_OPENMP_ROUNDINGMODE],[


AC_MSG_CHECKING([OpenMP and rounding mode"])

CADNA_CHECK_OPENMP_ROUNDINGMODE_CXXFLAGS="$CXXFLAGS "

CXXFLAGS="$CXXFLAGS $OPENMP_CXXFLAGS  -Werror  $OPENMP_LDFLAGS"


AC_RUN_IFELSE([
	AC_LANG_PROGRAM(
		[[ #include "omp.h" 
		   #include <stdio.h>
		   #include <fenv.h> ] ], 
		[[

  omp_set_num_threads(3);
  //  printf("openmp threads %d\n", omp_get_num_threads());
  fesetround(FE_UPWARD);
  unsigned int rounded_up = 1;
#pragma omp parallel reduction(&&:rounded_up) num_threads(3)
  {
    double test = 1.;
    test += 1.e-20;
    rounded_up = rounded_up && (test != 1.0);
    //printf("l233 test=%25.18e   %d\n",test, test!=1.0);
  }
if (rounded_up) return(0);
else {
#pragma omp parallel num_threads(3)
    {
      fesetround(FE_UPWARD);
    }

    rounded_up = 1;
#pragma omp parallel reduction(&&:rounded_up) num_threads(3)
    {
      double test = 1.;
      test += 1.e-20;
      //      printf("l249 test=%25.18e %d \n",test, test!=1.0);
      rounded_up = rounded_up && (test != 1.0);
    }
     if (rounded_up) return(0);
     else return(1);
}
	]])],
           	    [
#		    cat conftest.c*
		    use_openmp="yes"
                    CXXCADNACOMMONFLAGS="$CXXCADNACOMMONFLAGS  "
               	    AC_MSG_RESULT(yes)	
                    ],
                    [
		     use_openmp="no" 
		     AC_MSG_RESULT(no)
               	    ])

])

CXX_FLAGS="$CADNA_CHECK_OPENMP_ROUNDINGMODE_CXXFLAGS"

