all:
	g++ -o nnfc main.cpp fc_layer.cpp -Wall -pedantic -O3 -mavx -mavx2 -mfma -march=native

clean:
	rm -f nnfc


