LIBS=-lucp -luct -lucs -lmpi

test : test.cc ucxctrl.o
	g++ $^ -o $@ $(LIBS)

%.o : %.cc
	g++ -c $< -o $@

% : %.cc
	g++ $< -o $@ $(LIBS)
