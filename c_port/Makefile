CC = gcc
CFLAGS = -O2 -std=c99 -Wall -Wextra
LDFLAGS = -lm

all: test

generated/luts.h: gen_luts.py
	python3 gen_luts.py

generated/test_vectors.h: gen_vectors.py ../weights/geometric_backbone.npz
	python3 gen_vectors.py

test_phi_int: phi_int.c test_phi_int.c generated/luts.h generated/test_vectors.h
	$(CC) $(CFLAGS) -o test_phi_int phi_int.c test_phi_int.c $(LDFLAGS)

test: test_phi_int
	./test_phi_int

clean:
	rm -f test_phi_int
	rm -rf generated __pycache__

.PHONY: all test clean
