CC = mpicc
CFLAGS = -O3 -fopenmp -Wall
LDFLAGS = -lmetis -lm -lgvc -lcgraph

all: Q1

Q1: Q1.c
	rm -f q1 tree_T*.png
	rm -f q1 *.txt
	$(CC) $(CFLAGS) -o Q1 Q1.c $(LDFLAGS)

clean:
	rm -f q1 tree_T*.png
	rm -f q1 *.txt *.tmp
