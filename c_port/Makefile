CC = gcc
CFLAGS = -O2 -std=c99 -Wall -Wextra
LDFLAGS = -lm

all: test

generated/luts.h generated/luts.c: gen_luts.py
	python3 gen_luts.py

generated/test_vectors.h: gen_vectors.py ../weights/geometric_backbone.npz
	python3 gen_vectors.py

test_phi_int: phi_int.c generated/luts.c test_phi_int.c generated/luts.h generated/test_vectors.h
	$(CC) $(CFLAGS) -o test_phi_int phi_int.c generated/luts.c test_phi_int.c $(LDFLAGS)

test_fib: phi_int.c fib_solve.c generated/luts.c test_fib.c generated/luts.h generated/test_vectors.h
	$(CC) $(CFLAGS) -o test_fib phi_int.c fib_solve.c generated/luts.c test_fib.c $(LDFLAGS)

generated/test_fixed_nn.h: gen_fixed_vectors.py ../weights/geometric_backbone.npz
	python3 gen_fixed_vectors.py

test_fixed_nn: phi_int.c fixed_nn.c generated/luts.c test_fixed_nn.c generated/luts.h generated/test_fixed_nn.h
	$(CC) $(CFLAGS) -o test_fixed_nn phi_int.c fixed_nn.c generated/luts.c test_fixed_nn.c $(LDFLAGS)

test: test_phi_int test_fib test_fixed_nn
	./test_phi_int
	./test_fib
	./test_fixed_nn

clean:
	rm -f test_phi_int test_fib test_fixed_nn
	rm -rf generated __pycache__

.PHONY: all test clean
