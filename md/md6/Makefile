FC  = ifort

EXE = fe211
OBJ = run0.o md.o ran.o nhvars.o cell.o eam_fe.o cmm_parms.o

$(EXE):$(OBJ)
	$(FC) -o $@ $^

run0.o   : md.o cmm_parms.o
md.o     : ran.o nhvars.o cell.o eam_fe.o cmm_parms.o 
nhvars.o : cmm_parms.o 
cell.o   : cmm_parms.o

%.o: %.f90
	$(FC) -c $<

.PHONY : clean veryclean 
clean:
	rm -f *.o *.mod *.MOD *~ *.dat fe211
	rm -f ../data6/*

veryclean:
	rm -f *.o *.mod *.MOD $(EXE)
	rm -f ../data6/*
