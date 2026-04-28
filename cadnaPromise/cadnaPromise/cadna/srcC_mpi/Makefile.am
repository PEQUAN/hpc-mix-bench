
CXX=$(MPICXX)


if CADNA_MPI
lib_LIBRARIES = libcadnaMPIC.a libcadnaMPICdebug.a  



##########  MP%MPII 

libcadnaMPIC_a_SOURCES=  cadna_mpi.cc
libcadnaMPIC_a_CXXFLAGS=  $(AM_CXXFLAGS) -I$(top_builddir)/srcC $(CXXCADNAOPTIMFLAGS) $(CXXFLAGS) 


##########  MPI DEBUG

libcadnaMPICdebug_a_SOURCES=  cadna_mpi.cc
libcadnaMPICdebug_a_CXXFLAGS=  $(AM_CXXFLAGS) -I$(top_builddir)/srcC   $(CXXCADNADEBUGFLAGS) $(CXXFLAGS) 


if CADNA_OPENMP

lib_LIBRARIES +=  libcadnaMPICforOpenMP.a libcadnaMPICdebugforOpenMP.a   



##########  MPI for OPENMP 

libcadnaMPICforOpenMP_a_SOURCES=  cadna_mpi.cc
libcadnaMPICforOpenMP_a_CXXFLAGS=  $(AM_CXXFLAGS) $(OPENMP_CXXFLAGS) -I$(top_builddir)/srcC  $(CXXCADNAOPTIMFLAGS)  $(CXXFLAGS) 


##########  MPI DEBUG for OPENMP

libcadnaMPICdebugforOpenMP_a_SOURCES=  cadna_mpi.cc
libcadnaMPICdebugforOpenMP_a_CXXFLAGS=  $(AM_CXXFLAGS) $(OPENMP_CXXFLAGS) -I$(top_builddir)/srcC  $(CXXCADNADEBUGFLAGS) $(CXXFLAGS) 

endif
endif


EXTRA_libcadnaMPIC_a_SOURCES = \
cadna_mpi.cc

include_HEADERS = \
 cadna_mpi.h


