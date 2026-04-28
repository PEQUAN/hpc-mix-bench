bin_PROGRAMS=ex1 ex1_cad ex2 ex2_cad  

LDFLAGSCADNA=-L${libdir} -lcadnaCdebug -lm
AM_CXXFLAGS+= -I${includedir} -D__CADNA_DEBUG__    $(CXXCADNADEBUGFLAGS) $(CXXFLAGS) 
   


ex1_SOURCES=ex1.cc
ex2_SOURCES=ex2.cc


ex1_CFLAGS=$(CXXCADNACOMMONFLAGS)  
ex2_CFLAGS=$(CXXCADNACOMMONFLAGS)


ex1_cad_SOURCES=ex1_cad.cc
ex2_cad_SOURCES=ex2_cad.cc

ex1_cad_LDADD=$(LDFLAGSCADNA) $(CADNA_CXX_LIB)
ex2_cad_LDADD=$(LDFLAGSCADNA) $(CADNA_CXX_LIB)

ex1_LDADD=-lm 
ex2_LDADD=-lm 



