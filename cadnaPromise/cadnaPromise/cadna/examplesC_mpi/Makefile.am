CC=$(MPICC)
CXX=$(MPICXX)




bin_PROGRAMS=exampleMPI1 exampleMPI1_cad exampleMPI2 exampleMPI2_cad  \
	exampleMPI1_float  exampleMPI1_float_cad exampleMPI2_float  exampleMPI2_float_cad

#bin_PROGRAMS= exampleMPI1_cad  exampleMPI2_cad

AM_CXXFLAGS+= -I${includedir}  -g   $(CXXCADNACOMMONFLAGS) $(CXXCADNADEBUGFLAGS)

AM_CFLAGS= -g 


## exampleMPI1.c and exampleMPI2.cc do the same computation.
##  exampleMPI1 is a C version
##  exampleMPI2 is a C++ version

exampleMPI1_SOURCES=exampleMPI1.c
exampleMPI2_SOURCES=exampleMPI2.cc
exampleMPI1_float_SOURCES=exampleMPI1_float.c
exampleMPI2_float_SOURCES=exampleMPI2_float.cc

exampleMPI1_cad_SOURCES=exampleMPI1_cad.cc
exampleMPI1_cad_CXXFLAGS=$(AM_CXXFLAGS) 
exampleMPI1_cad_LDADD=$(LDFLAGSCADNA) -L${libdir}   -lcadnaMPIC -lcadnaC  $(CADNA_CXX_LIB)

exampleMPI2_cad_SOURCES=exampleMPI2_cad.cc
exampleMPI2_cad_CXXFLAGS=$(AM_CXXFLAGS) 
exampleMPI2_cad_LDADD=$(LDFLAGSCADNA) -L${libdir}  -lcadnaMPIC -lcadnaC  $(CADNA_CXX_LIB)


exampleMPI1_float_cad_SOURCES=exampleMPI1_float_cad.cc
exampleMPI1_float_cad_CXXFLAGS=$(AM_CXXFLAGS) 
exampleMPI1_float_cad_LDADD=$(LDFLAGSCADNA) -L${libdir}   -lcadnaMPIC -lcadnaC  $(CADNA_CXX_LIB)

exampleMPI2_float_cad_SOURCES=exampleMPI2_float_cad.cc
exampleMPI2_float_cad_CXXFLAGS=$(AM_CXXFLAGS) 
exampleMPI2_float_cad_LDADD=$(LDFLAGSCADNA) -L${libdir}  -lcadnaMPIC -lcadnaC  $(CADNA_CXX_LIB)



