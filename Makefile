LIBS=-lucp -luct -lucs -lmpi -lpthread

default : put_bench rand_put_bench

% : %.cc ucxctrl.o
	g++ -O3 -fopenmp $^ -o $@ $(LIBS)

%.o : %.cc
	g++ -O3 -c $< -o $@
