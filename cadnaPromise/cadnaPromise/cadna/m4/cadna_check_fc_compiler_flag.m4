
AC_DEFUN([CADNA_CHECK_FC_COMPILER_FLAG],[

#AC_MSG_CHECKING([for $1 ])



#if $FC   
#then 
 #   $3
  #  AC_MSG_RESULT([no])	 
#else
 #  $2
  # AC_MSG_RESULT([yes])    
#fi



AC_MSG_CHECKING([for $2 option $1  ])
AC_MSG_RESULT([:])

cat > conftest.f90 << EOF
program test
call exit(0)
end
EOF


IFS=' ' read -a array <<< $1

# [[@]] if not we have ${array@} in configure
# test of the compilation options one by one

for i in "${array[[@]]}"  ;
do
     AC_MSG_CHECKING([$2 option $i : ])	
    if $FC  $i conftest.f90   2>&1 | grep -iE "warning|error" >/dev/null
    then 
    	 echo  "$i is unknown"
	 AC_MSG_ERROR([])
    else
	AC_MSG_RESULT([yes])
    fi    
done



])

