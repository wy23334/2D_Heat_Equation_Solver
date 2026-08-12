CC := gcc
MPICC := mpicc

# 所有性能版本采用相同优化选项，保证不同实现之间的计时可比。
CFLAGS := -O3 -Wall -march=native -ffast-math
OMPFLAGS := -fopenmp
LDLIBS := -lm

.PHONY: all phase0 phase1 phase2 phase3 phase4 clean

# 编译全部阶段。也可使用 make phaseN 单独编译指定阶段。
all: phase0 phase1 phase2 phase3 phase4

# -----------------------------------------------------------------------------
# Phase 0：OpenMP 环境验证
# -----------------------------------------------------------------------------
phase0: warm_up

warm_up: warm_up.c
	$(CC) -Wall $(OMPFLAGS) $< -o $@

# -----------------------------------------------------------------------------
# Phase 1：串行 Jacobi 求解器
# -----------------------------------------------------------------------------
phase1: serial

serial: jacobi_serial.c
	$(CC) $(CFLAGS) $< -o jacobi_serial $(LDLIBS)

# -----------------------------------------------------------------------------
# Phase 2：OpenMP 并行化与调度策略对比
# 同一份 jacobi_omp.c 通过编译宏生成 static、dynamic、guided 三个版本。
# -----------------------------------------------------------------------------
phase2: omp_static omp_dynamic omp_guided

omp_static: jacobi_omp.c
	$(CC) $(CFLAGS) $(OMPFLAGS) $< -o jacobi_omp_static $(LDLIBS)

omp_dynamic: jacobi_omp.c
	$(CC) $(CFLAGS) $(OMPFLAGS) -DSCHED_DYNAMIC $< -o jacobi_omp_dynamic $(LDLIBS)

omp_guided: jacobi_omp.c
	$(CC) $(CFLAGS) $(OMPFLAGS) -DSCHED_GUIDED $< -o jacobi_omp_guided $(LDLIBS)

# -----------------------------------------------------------------------------
# Phase 3：缓存局部性与二维分块
# 题目要求比较四种版本，因此保留以下四个独立目标：
#   1. serial_colmajor：列优先串行基准；
#   2. serial_tiled：串行分块版本；
#   3. omp_colmajor：列优先 OpenMP 基准；
#   4. omp_tiled：OpenMP 与分块结合版本。
# -----------------------------------------------------------------------------
phase3: serial_colmajor serial_tiled omp_colmajor omp_tiled

serial_colmajor: jacobi_serial_colmajor.c
	$(CC) $(CFLAGS) $< -o jacobi_serial_colmajor $(LDLIBS)

serial_tiled: jacobi_tiled.c
	$(CC) $(CFLAGS) $< -o jacobi_serial_tiled $(LDLIBS)

omp_colmajor: jacobi_omp_colmajor.c
	$(CC) $(CFLAGS) $(OMPFLAGS) $< -o jacobi_omp_colmajor $(LDLIBS)

omp_tiled: jacobi_omp_tiled.c
	$(CC) $(CFLAGS) $(OMPFLAGS) $< -o jacobi_omp_tiled $(LDLIBS)

# -----------------------------------------------------------------------------
# Phase 4：MPI 分布式扩展
# -----------------------------------------------------------------------------
phase4: mpi

mpi: jacobi_mpi.c
	$(MPICC) $(CFLAGS) $< -o jacobi_mpi $(LDLIBS)

# 仅删除本 Makefile 生成的可执行文件，不使用可能影响源码或数据的通配符。
clean:
	rm -f warm_up \
		jacobi_serial \
		jacobi_omp_static jacobi_omp_dynamic jacobi_omp_guided \
		jacobi_serial_colmajor jacobi_serial_tiled \
		jacobi_omp_colmajor jacobi_omp_tiled \
		jacobi_mpi
