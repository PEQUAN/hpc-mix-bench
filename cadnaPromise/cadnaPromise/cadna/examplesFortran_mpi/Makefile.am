FC=$(MPIFC)



bin_PROGRAMS=\
	exampleMPI1Reduce_cad \
	exampleMPI1Reduce_float_cad  \
	exampleMPI1SendRecv \
	exampleMPI1SendRecv_float \
	exampleMPI1SendRecv_cad  \
	exampleMPI1SendRecv_float_cad 



AM_FCFLAGS+= -I${includedir}  -g   $(CXXCADNACOMMONFLAGS) $(CXXCADNADEBUGFLAGS)

exampleMPI1Reduce_cad_SOURCES=exampleMPI1Reduce_cad.f90
exampleMPI1Reduce_cad_FCFLAGS=$(AM_FCFLAGS) 
exampleMPI1Reduce_cad_LDADD=$(LDFLAGSCADNA) -L${libdir}   -lcadnaMPIFortran -lcadnaMPIC -lcadnaFortran -lcadnaC   $(CADNA_CXX_LIB) 

exampleMPI1Reduce_float_cad_SOURCES=exampleMPI1Reduce_float_cad.f90
exampleMPI1Reduce_float_cad_FCFLAGS=$(AM_FCFLAGS) 
exampleMPI1Reduce_float_cad_LDADD=$(LDFLAGSCADNA) -L${libdir}   -lcadnaMPIFortran -lcadnaMPIC -lcadnaFortran -lcadnaC   $(CADNA_CXX_LIB)


exampleMPI1SendRecv_SOURCES=exampleMPI1SendRecv.f90
exampleMPI1SendRecv_float_SOURCES=exampleMPI1SendRecv_float.f90


exampleMPI1SendRecv_cad_SOURCES=exampleMPI1SendRecv_cad.f90
exampleMPI1SendRecv_cad_FCFLAGS=$(AM_FCFLAGS) 
exampleMPI1SendRecv_cad_LDADD=$(LDFLAGSCADNA) -L${libdir}   -lcadnaMPIFortran -lcadnaMPIC -lcadnaFortran -lcadnaC   $(CADNA_CXX_LIB) 

exampleMPI1SendRecv_float_cad_SOURCES=exampleMPI1SendRecv_float_cad.f90
exampleMPI1SendRecv_float_cad_FCFLAGS=$(AM_FCFLAGS) 
exampleMPI1SendRecv_float_cad_LDADD=$(LDFLAGSCADNA) -L${libdir}   -lcadnaMPIFortran -lcadnaMPIC -lcadnaFortran -lcadnaC   $(CADNA_CXX_LIB) 


