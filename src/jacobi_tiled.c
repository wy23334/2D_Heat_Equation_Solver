// Phase 3: 串行二维分块（tiling）与 SIMD 优化版本。
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

void run_jacobi_tiled(int N, int B, int max_iters, FILE *fp) {
    printf("\nPhase 3: 串行二维 Jacobi 分块版本 (B=%d, N=%d)\n", B, N);

    // [Phase 3 - Req 2] 64 字节对齐双缓冲数组，便于编译器向量化。
    double *u = NULL;
    double *u_new = NULL;

    if (posix_memalign((void**)&u, 64, N * N * sizeof(double)) != 0) {
        fprintf(stderr, "[ERROR] 无法为 u 分配对齐内存。\n");
        exit(EXIT_FAILURE);
    }
    if (posix_memalign((void**)&u_new, 64, N * N * sizeof(double)) != 0) {
        fprintf(stderr, "[ERROR] 无法为 u_new 分配对齐内存。\n");
        exit(EXIT_FAILURE);
    }

    // [Phase 1 - Req 2] 内部初值为 0℃。
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            u[IDX(i, j, N)] = 0.0; u_new[IDX(i, j, N)] = 0.0;
        }
    }

    // [Phase 1 - Req 2] 固定四条边界，且两个缓冲区保持一致。
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

    for (iter = 1; iter <= max_iters; iter++) {
        max_diff = 0.0;

        // [Phase 3 - Req 2] 分块只发生在一次 Jacobi 迭代内部；每个 tile
        // 内仍按 i 外层、j 内层的行优先顺序遍历，保持双缓冲逻辑不变。
        for (int ii = 1; ii < N - 1; ii += B) {
            int i_end = MIN_VAL(ii + B, N - 1);

            for (int jj = 1; jj < N - 1; jj += B) {
                int j_end = MIN_VAL(jj + B, N - 1);

                for (int i = ii; i < i_end; i++) {
                    // 基址按 64 字节对齐，但任意 N 时每一行不一定仍对齐，
                    // 因此不对行指针作不安全的 assume_aligned 假设。
                    double * restrict curr_row = &u[i * N];
                    double * restrict top_row  = &u[(i - 1) * N];
                    double * restrict bot_row  = &u[(i + 1) * N];
                    double * restrict next_row = &u_new[i * N];

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

        // [Phase 1 - Req 3/4] 每轮交换指针，并检查最大残差收敛条件。
        double *temp = u; u = u_new; u_new = temp;
        if (max_diff < TOLERANCE) break;
    }

    double elapsed = get_time() - start_time;

    printf("\n--- 性能验证 (Sanity Check) ---\n");
    // [Phase 3 - Req 3/4] 输出时间和实际迭代数供 tile size 脚本解析。
    printf("[指标 1] 串行耗时    : %.4f 秒\n", elapsed);
    printf("[指标 2] 最终迭代次数: %d 次\n", iter > max_iters ? max_iters : iter);

    free(u);
    free(u_new);
}

int main(int argc, char *argv[]) {
    int N = DEFAULT_N;
    int B = 32;
    int max_iters = MAX_ITER;

    if (argc > 1) N = atoi(argv[1]);
    if (argc > 2) B = atoi(argv[2]);
    if (argc > 3) max_iters = atoi(argv[3]);
    if (N < 3 || B < 1 || max_iters < 1) return EXIT_FAILURE;

    run_jacobi_tiled(N, B, max_iters, NULL);
    return 0;
}
