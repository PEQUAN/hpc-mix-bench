bin_PROGRAMS=ex1 ex1_cad ex2 ex2_cad ex3 ex3_cad ex4 ex4_cad \
             ex5 ex5_cad ex5_cad_opt ex6 ex6_cad ex7 ex7_cad 



LDFLAGSCADNA=-L${libdir} -lcadnaFortran   -lcadnaCdebug  -lm  $(CADNA_CXX_LIB)

AM_FCFLAGS+= -I${includedir}    -g

#$(CXXCADNACOMMONFLAGS) $(CXXCADNADEBUGFLAGS)



ex1_SOURCES=ex1.f90
ex2_SOURCES=ex2.f90
ex3_SOURCES=ex3.f90
ex4_SOURCES=ex4.f90
ex5_SOURCES=ex5.f90
ex6_SOURCES=ex6.f90
ex7_SOURCES=ex7.f90


ex1_cad_SOURCES=ex1_cad.f90
ex2_cad_SOURCES=ex2_cad.f90
ex3_cad_SOURCES=ex3_cad.f90
ex4_cad_SOURCES=ex4_cad.f90
ex5_cad_SOURCES=ex5_cad.f90
ex5_cad_opt_SOURCES=ex5_cad_opt.f90
ex6_cad_SOURCES=ex6_cad.f90
ex7_cad_SOURCES=ex7_cad.f90


ex1_cad_LDADD=$(LDFLAGSCADNA) $(LDFLAGS)
ex2_cad_LDADD=$(LDFLAGSCADNA) $(LDFLAGS)
ex3_cad_LDADD=$(LDFLAGSCADNA) $(LDFLAGS)
ex4_cad_LDADD=$(LDFLAGSCADNA) $(LDFLAGS)
ex5_cad_LDADD=$(LDFLAGSCADNA) $(LDFLAGS)
ex5_cad_opt_LDADD=$(LDFLAGSCADNA) $(LDFLAGS)
ex6_cad_LDADD=$(LDFLAGSCADNA) $(LDFLAGS)
ex7_cad_LDADD=$(LDFLAGSCADNA) $(LDFLAGS)


ex5_LDADD=-lm
