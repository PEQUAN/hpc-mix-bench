lib_LIBRARIES=libcadnaCdebug.a  libcadnaC.a


if CADNA_OPENMP
lib_LIBRARIES += libcadnaOpenmpCdebug.a  libcadnaOpenmpC.a 
endif

########## 

libcadnaC_a_SOURCES= \
cadna_intr.cc \
cadna_math.cc\
cadna_convert.cc \
cadna_type.cc \
cadna_complex.cc

libcadnaC_a_CXXFLAGS=  $(AM_CXXFLAGS)  $(CXXCADNAOPTIMFLAGS) $(CXXFLAGS) 


########### Debug

libcadnaCdebug_a_SOURCES= \
cadna_intr.cc \
cadna_math.cc \
cadna_convert.cc \
cadna_type.cc \
cadna_complex.cc

libcadnaCdebug_a_CPPFLAGS= -D__CADNA_DEBUG__
libcadnaCdebug_a_CXXFLAGS=  $(AM_CXXFLAGS)   $(CXXCADNADEBUGFLAGS)  $(CXXFLAGS) 


########### OpenMP 

libcadnaOpenmpC_a_SOURCES= \
cadna_intr.cc \
cadna_math.cc\
cadna_convert.cc \
cadna_type.cc \
cadna_complex.cc

libcadnaOpenmpC_a_CXXFLAGS=  $(AM_CXXFLAGS) $(CXXCADNAOPTIMFLAGS) $(OPENMP_CXXFLAGS) $(CXXFLAGS) 

########### OpenMP Debug

libcadnaOpenmpCdebug_a_SOURCES= \
cadna_intr.cc \
cadna_math.cc \
cadna_convert.cc \
cadna_type.cc \
cadna_complex.cc

libcadnaOpenmpCdebug_a_CPPFLAGS= -D__CADNA_DEBUG__
libcadnaOpenmpCdebug_a_CXXFLAGS=  $(AM_CXXFLAGS)  $(CXXCADNADEBUGFLAGS) $(OPENMP_CXXFLAGS) $(CXXFLAGS)  


EXTRA_libcadnaC_a_SOURCES = \
cadna_math_other.cc cadna_math_gnu.cc 

include_HEADERS = \
cadna.h  cadna-error.h \
cadna_add.h cadna_sub.h cadna_mul.h cadna_div.h \
cadna_eq.h cadna_ne.h \
cadna_ge.h cadna_gt.h cadna_le.h cadna_lt.h \
cadna_digitnumber.h cadna_computedzero.h cadna_numericalnoise.h \
cadna_unstab.h cadna_str.h cadna_to.h cadna_random.h cadna_plugin.h cadna_comp.h cadna_fma.h

if HALF_EMULATION
include_HEADERS += half.hpp cadna_half.hpp
endif
