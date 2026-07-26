// /home/wy/Projects/2D_Heat_Equation_Solver/src/jacobi_serial_tiled.c
// Phase 1 - Tiling & SIMD Optimization (极致性能版 + 内存安全检查)
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>

#define DEFAULT_N 1024
#define MAX_ITER 200000
#define TOLERANCE 1e-6
#define IDX(i, j, N) ((i) * (N) + (j))
#define MIN_VAL(a, b) ((a) < (b) ? (a) : (b))

double get_time() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

void run_jacobi_tiled(int N, int B, FILE *fp) {
    printf("\n=============================================\n");
    printf("   Phase 1: 串行二维 Jacobi (极致分块 B=%d, N=%d)\n", B, N);
    printf("=============================================\n");

    // 为了使用高级 AVX 指令，内存对齐到 64 字节，并添加强制的返回值安全检查
    double *u = NULL;
    double *u_new = NULL;

    if (posix_memalign((void**)&u, 64, N * N * sizeof(double)) != 0) {
        fprintf(stderr, "[!] 错误: 无法为 u 分配对齐的内存！\n");
        exit(EXIT_FAILURE);
    }
    if (posix_memalign((void**)&u_new, 64, N * N * sizeof(double)) != 0) {
        fprintf(stderr, "[!] 错误: 无法为 u_new 分配对齐的内存！\n");
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            u[IDX(i, j, N)] = 0.0; u_new[IDX(i, j, N)] = 0.0;
        }
    }

    for (int i = 0; i < N; i++) {
        u[IDX(i, 0, N)] = 100.0;     u_new[IDX(i, 0, N)] = 100.0;
        u[IDX(i, N-1, N)] = 0.0;     u_new[IDX(i, N-1, N)] = 0.0;
    }
    for (int j = 0; j < N; j++) {
        u[IDX(0, j, N)] = 50.0;      u_new[IDX(0, j, N)] = 50.0;
        u[IDX(N-1, j, N)] = 75.0;    u_new[IDX(N-1, j, N)] = 75.0;
    }

    double start_time = get_time();
    int iter = 0;
    double max_diff = 0.0;

    for (iter = 1; iter <= MAX_ITER; iter++) {
        max_diff = 0.0;

        for (int ii = 1; ii < N - 1; ii += B) {
            int i_end = MIN_VAL(ii + B, N - 1);

            for (int jj = 1; jj < N - 1; jj += B) {
                int j_end = MIN_VAL(jj + B, N - 1);

                for (int i = ii; i < i_end; i++) {
                    double * restrict curr_row = __builtin_assume_aligned(&u[i * N], 64);
                    double * restrict top_row  = __builtin_assume_aligned(&u[(i - 1) * N], 64);
                    double * restrict bot_row  = __builtin_assume_aligned(&u[(i + 1) * N], 64);
                    double * restrict next_row = __builtin_assume_aligned(&u_new[i * N], 64);

                    double row_max = 0.0;

                    #pragma GCC ivdep
                    #pragma GCC unroll 4
                    for (int j = jj; j < j_end; j++) {
                        double val = (top_row[j] + bot_row[j] + curr_row[j - 1] + curr_row[j + 1]) * 0.25;
                        next_row[j] = val;

                        double diff = fabs(val - curr_row[j]);
                        if (diff > row_max) {
                            row_max = diff;
                        }
                    }

                    if (row_max > max_diff) {
                        max_diff = row_max;
                    }
                }
            }
        }

        double *temp = u; u = u_new; u_new = temp;
        if (max_diff < TOLERANCE) break;
    }

    double elapsed = get_time() - start_time;

    printf("\n--- 🏁 性能验证 (Sanity Check) ---\n");
    printf("[指标 1] 串行耗时    : %.4f 秒\n", elapsed);
    printf("[指标 2] 最终迭代次数: %d 次\n", iter > MAX_ITER ? MAX_ITER : iter);

    free(u);
    free(u_new);
}

int main(int argc, char *argv[]) {
    int N = DEFAULT_N;
    int B = 32;

    if (argc > 1) N = atoi(argv[1]);
    if (argc > 2) B = atoi(argv[2]);

    run_jacobi_tiled(N, B, NULL);
    return 0;
}