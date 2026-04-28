



CXX=$(MPICXX)
FC=$(MPIFC)

if CADNA_FORTRAN
if CADNA_FORTRAN_MPI

lib_LIBRARIES=libcadnaMPIFortran.a

libcadnaMPIFortran_a_SOURCES=  \
cadna_mpi_binding.cc \
cadna_mpi.f90


nodist_include_HEADERS = cadna_mpi.mod


libcadnaMPIFortran_a_FCFLAGS=  $(AM_FCFLAGS) -I../srcFortran   $(FCCADNACOMMONFLAGS) $(FCFLAGS)

libcadnaMPIFortran_a_CXXFLAGS= $(AM_CXXFLAGS)  -I../srcC_mpi -I../srcC   $(CXXCADNACOMMONFLAGS) $(CXXCADNADEBUGFLAGS) $(CXXFLAGS)

CLEANFILES  =   $(nodist_include_HEADERS)

endif
endif
