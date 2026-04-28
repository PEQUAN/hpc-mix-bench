CC=$(MPICC)
CXX=$(MPICXX)

bin_PROGRAMS= exampleMPI_OMP1 exampleMPI_OMP1_cad  
#bin_PROGRAMS= exampleMPI1_cad  exampleMPI2_cad

AM_CFLAGS= -g  -I${includedir}    $(OPENMP_CFLAGS)  
AM_CXXFLAGS+= -I${includedir}    $(OPENMP_CXXFLAGS)   $(CXXCADNACOMMONFLAGS) $(CXXCADNADEBUGFLAGS)

MPI_CFLAGS=$(AM_CFLAGS) 
MPICFLAGS=$(AM_CFLAGS) 



## exampleMPI1.c  exampleMPI2.cc are the same. 
##  exampleMPI1 is a C version
##  exampleMPI2 is a C++ version

exampleMPI_OMP1_SOURCES=exampleMPI_OMP1.c
exampleMPI_OMP1_LDADD= -L${libdir}    $(OPENMP_LDFLAGS)

exampleMPI_OMP1_cad_SOURCES=exampleMPI_OMP1_cad.cc
exampleMPI_OMP1_cad_CXXFLAGS=$(AM_CXXFLAGS) 
exampleMPI_OMP1_cad_LDADD=$(LDFLAGSCADNA) -L${libdir}  -lcadnaMPICforOpenMP  -lcadnaOpenmpCdebug  $(OPENMP_LDFLAGS) $(CADNA_CXX_LIB)









