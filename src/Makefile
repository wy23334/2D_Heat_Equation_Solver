CC = gcc
CFLAGS = -O3 -Wall
LDFLAGS = -lm
OMPFLAGS = -fopenmp -DSCHED_GUIDED

all: serial omp

serial: jacobi_serial.c
	$(CC) $(CFLAGS) jacobi_serial.c -o jacobi_serial $(LDFLAGS)

omp: jacobi_omp.c
	$(CC) $(CFLAGS) $(OMPFLAGS) jacobi_omp.c -o jacobi_omp $(LDFLAGS)

clean:
	rm -f jacobi_serial jacobi_omp heatmap_*.txt heatmap_*.png *.csv *.png