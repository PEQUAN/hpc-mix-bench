
AC_DEFUN([CADNA_CHECK_COMPILER_FLAG],[

AC_MSG_CHECKING([for $2 option $1  ])
AC_MSG_RESULT([:])


case "_AC_LANG_PREFIX" in
          C)
	  compiler=$CC	  
	  ;;
          CXX)
	  compiler=$CXX			
              ;;
esac;



cat > conftest.cpp << EOF
int main(){ return 0;}
EOF

IFS=' ' read -a array <<< $1

# [[@]] if not we have ${array@} in configure
# test of the compilation options one by one



for i in "${array[[@]]}"  ;
do
     AC_MSG_CHECKING([$2 option $i : ])	
    if $compiler  -Wformat -Werror  $i  conftest.cpp   2>&1 | grep -iE "warning|error" >/dev/null
    then 
    	 echo  "$i is unknown"
	 AC_MSG_ERROR([])
    else
	AC_MSG_RESULT([yes])
    fi    
done


])

