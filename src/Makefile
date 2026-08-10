CC = gcc
MPICC = mpicc
# 性能对比必须使用相同的编译选项。
CFLAGS_COMMON = -O3 -Wall -march=native -ffast-math
CFLAGS_BASE = $(CFLAGS_COMMON)
CFLAGS_EXTREME = $(CFLAGS_COMMON)
LDFLAGS = -lm
OMP_BASE = -fopenmp

all: serial serial_colmajor serial_tiled omp_naive omp_colmajor omp_static omp_tiled omp_dynamic omp_guided mpi

# 编译“原始串行”（作为真正的 Baseline）
serial: jacobi_serial.c
	$(CC) $(CFLAGS_BASE) jacobi_serial.c -o jacobi_serial $(LDFLAGS)

# Phase 3 专用反面基准：对 C 行主序数组做按列跨步访问
serial_colmajor: jacobi_serial_colmajor.c
	$(CC) $(CFLAGS_BASE) jacobi_serial_colmajor.c -o jacobi_serial_colmajor $(LDFLAGS)

# 编译“极限优化版”
serial_tiled: jacobi_serial_tiled.c
	$(CC) $(CFLAGS_EXTREME) jacobi_serial_tiled.c -o jacobi_serial_tiled $(LDFLAGS)

# OpenMP 相关编译保持不变
omp_naive: jacobi_omp_naive.c
	$(CC) $(CFLAGS_EXTREME) $(OMP_BASE) jacobi_omp_naive.c -o jacobi_omp_naive $(LDFLAGS)

# Phase 3 专用的列优先 OpenMP 反面基准
omp_colmajor: jacobi_omp_colmajor.c
	$(CC) $(CFLAGS_EXTREME) $(OMP_BASE) jacobi_omp_colmajor.c -o jacobi_omp_colmajor $(LDFLAGS)

omp_static: jacobi_omp.c
	$(CC) $(CFLAGS_EXTREME) $(OMP_BASE) jacobi_omp.c -o jacobi_omp_static $(LDFLAGS)

# [新增] 编译 OpenMP + 分块版
omp_tiled: jacobi_omp_tiled.c
	$(CC) $(CFLAGS_EXTREME) $(OMP_BASE) jacobi_omp_tiled.c -o jacobi_omp_tiled $(LDFLAGS)

omp_dynamic: jacobi_omp.c
	$(CC) $(CFLAGS_EXTREME) $(OMP_BASE) -DSCHED_DYNAMIC jacobi_omp.c -o jacobi_omp_dynamic $(LDFLAGS)

omp_guided: jacobi_omp.c
	$(CC) $(CFLAGS_EXTREME) $(OMP_BASE) -DSCHED_GUIDED jacobi_omp.c -o jacobi_omp_guided $(LDFLAGS)

mpi: jacobi_mpi.c
	$(MPICC) $(CFLAGS_EXTREME) jacobi_mpi.c -o jacobi_mpi $(LDFLAGS)

# 🌟 核心修复：移除危险的通配符，精确指定要删除的可执行文件
clean:
	rm -f jacobi_serial jacobi_serial_colmajor jacobi_serial_tiled jacobi_omp_naive jacobi_omp_colmajor jacobi_omp_static jacobi_omp_tiled jacobi_omp_dynamic jacobi_omp_guided jacobi_mpi heatmap_*.txt heatmap_*.png *.csv *.png
