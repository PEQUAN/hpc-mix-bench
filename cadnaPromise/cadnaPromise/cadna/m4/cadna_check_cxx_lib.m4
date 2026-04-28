
AC_DEFUN([CADNA_CHECK_CXX_LIB],[

AC_MSG_CHECKING([for $1 ])

cat > conftest.cpp << EOF
int main(){ return 0;}
EOF


if $CXX -v  conftest.cpp   2>&1 | grep lstdc >/dev/null
then 
   CADNA_CXX_LIB="-lstdc++"	
   AC_MSG_RESULT([libstdc++ found ])    
else
    CADNA_CXX_LIB="-lc++"
    AC_MSG_RESULT([libc++ found ])  
fi

])


