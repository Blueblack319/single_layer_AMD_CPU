all:
	g++ -o nnfc main.cpp fc_layer.cpp -Wall -pedantic -O2 -mavx -mavx2 -mfma

clean:
	rm -f nnfc


