


if CADNA_FORTRAN


lib_LIBRARIES=libcadnaFortran.a

libcadnaFortran_a_SOURCES=  \
cadna_binding.cc \
cadna_utils_binding.cc \
$(CCSOURCES) \
$(F90SOURCES)


nodist_include_HEADERS = $(F90SOURCES:.f90=.mod)


CCSOURCES= \
cadna_add_binding.cc \
cadna_affect_binding.cc \
cadna_convert_binding.cc \
cadna_div_binding.cc \
cadna_eq_binding.cc \
cadna_ge_binding.cc \
cadna_gt_binding.cc \
cadna_le_binding.cc \
cadna_lt_binding.cc \
cadna_math_binding.cc \
cadna_mult_binding.cc \
cadna_ne_binding.cc \
cadna_opp_binding.cc \
cadna_pow_binding.cc \
cadna_sub_binding.cc \
cadna_plus_binding.cc


F90SOURCES=cadna_types.f90  cadna_add.f90 cadna_affect.f90 cadna_div.f90 \
cadna_gt.f90  cadna_ge.f90 cadna_lt.f90 cadna_le.f90  \
cadna_eq.f90 cadna_ne.f90  \
cadna_convert.f90 cadna_prec.f90 \
cadna_mult.f90 cadna_sub.f90 cadna_pow.f90 cadna_opp.f90 \
cadna_math.f90 cadna_utils.f90 cadna_array.f90  cadna_plus.f90 \
cadna_cmplx_add.f90  cadna_cmplx_sub.f90    cadna_cmplx_opp.f90 \
cadna_cmplx_affect.f90 cadna_cmplx_convert.f90 cadna_cmplx_mult.f90   cadna_cmplx_div.f90   \
cadna_cmplx_eq.f90       cadna_cmplx_ne.f90 \
cadna_cmplx_math.f90    cadna_cmplx_pow.f90 cadna_cmplx_array.f90 cadna_cmplx_plus.f90    cadna.f90


libcadnaFortran_a_FCFLAGS=  $(AM_FCFLAGS)  $(FCFLAGS) 

libcadnaFortran_a_CXXFLAGS=  -I../srcC   $(AM_CXXFLAGS) $(CXXCADNADEBUGFLAGS) $(CXXFLAGS) 

CLEANFILES  =   $(nodist_include_HEADERS)

endif
