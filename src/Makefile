CC = gcc
# 1. 基础优化：用于真实的“原始串行”基准测试
CFLAGS_BASE = -O3 -Wall
# 2. 极限优化：用于我们精心调优的“分块+SIMD”版本
CFLAGS_EXTREME = -O3 -Wall -march=native -ffast-math
LDFLAGS = -lm
OMP_BASE = -fopenmp

all: serial serial_tiled omp_static omp_dynamic omp_guided

# 编译“原始串行”（作为真正的 Baseline）
serial: jacobi_serial.c
	$(CC) $(CFLAGS_BASE) jacobi_serial.c -o jacobi_serial $(LDFLAGS)

# 编译“极限优化版”
serial_tiled: jacobi_serial_tiled.c
	$(CC) $(CFLAGS_EXTREME) jacobi_serial_tiled.c -o jacobi_serial_tiled $(LDFLAGS)

# OpenMP 相关编译保持不变
omp_static: jacobi_omp.c
	$(CC) $(CFLAGS_EXTREME) $(OMP_BASE) jacobi_omp.c -o jacobi_omp_static $(LDFLAGS)

omp_dynamic: jacobi_omp.c
	$(CC) $(CFLAGS_EXTREME) $(OMP_BASE) -DSCHED_DYNAMIC jacobi_omp.c -o jacobi_omp_dynamic $(LDFLAGS)

omp_guided: jacobi_omp.c
	$(CC) $(CFLAGS_EXTREME) $(OMP_BASE) -DSCHED_GUIDED jacobi_omp.c -o jacobi_omp_guided $(LDFLAGS)

clean:
	rm -f jacobi_serial jacobi_serial_tiled jacobi_omp_* heatmap_*.txt heatmap_*.png *.csv *.png