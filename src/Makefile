CC = gcc
CFLAGS = -O3 -Wall
LDFLAGS = -lm
# 基础的 OpenMP 编译选项，不再绑定特定的宏
OMP_BASE = -fopenmp

# 默认目标：一次性编译出所有版本
all: serial omp_static omp_dynamic omp_guided

# 1. 编译串行版本
serial: jacobi_serial.c
	$(CC) $(CFLAGS) jacobi_serial.c -o jacobi_serial $(LDFLAGS)

# 2. 编译静态调度版本 (C代码中默认走 #else 分支，所以无需传宏)
omp_static: jacobi_omp.c
	$(CC) $(CFLAGS) $(OMP_BASE) jacobi_omp.c -o jacobi_omp_static $(LDFLAGS)

# 3. 编译动态调度版本 (传入 -DSCHED_DYNAMIC 宏激活对应代码)
omp_dynamic: jacobi_omp.c
	$(CC) $(CFLAGS) $(OMP_BASE) -DSCHED_DYNAMIC jacobi_omp.c -o jacobi_omp_dynamic $(LDFLAGS)

# 4. 编译向导调度版本 (传入 -DSCHED_GUIDED 宏激活对应代码)
omp_guided: jacobi_omp.c
	$(CC) $(CFLAGS) $(OMP_BASE) -DSCHED_GUIDED jacobi_omp.c -o jacobi_omp_guided $(LDFLAGS)

# 清理所有生成的文件（注意通配符 jacobi_omp_* 会清理掉三个版本的omp）
clean:
	rm -f jacobi_serial jacobi_omp_* heatmap_*.txt heatmap_*.png *.csv *.png