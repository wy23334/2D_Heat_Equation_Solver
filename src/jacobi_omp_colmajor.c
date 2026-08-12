// Phase 3: OpenMP 列优先负面对照。
// 与正确的 Phase 2 行并行版本隔离，仅用于题目要求的缓存局部性实验。
#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

#define DEFAULT_N 1024
#define DEFAULT_MAX_ITERS 200000
#define TOLERANCE 1e-6
#define IDX(i, j, n) ((i) * (n) + (j))

int main(int argc, char **argv) {
    int n = DEFAULT_N;
    int threads = 8;
    int max_iters = DEFAULT_MAX_ITERS;
    if (argc > 1) n = atoi(argv[1]);
    if (argc > 2) threads = atoi(argv[2]);
    if (argc > 3) max_iters = atoi(argv[3]);
    if (n < 3 || threads < 1 || max_iters < 1) return EXIT_FAILURE;

    // [Phase 3 - Req 3] 线程数由参数传入，用于 8 线程四版本对比。
    omp_set_num_threads(threads);
    double *u = calloc((size_t)n * n, sizeof(*u));
    double *u_new = calloc((size_t)n * n, sizeof(*u_new));
    if (u == NULL || u_new == NULL) {
        free(u);
        free(u_new);
        return EXIT_FAILURE;
    }
    // [Phase 1 - Req 2/3] 与其他版本保持相同的初值、边界和双缓冲。
    for (int i = 0; i < n; i++) {
        u[IDX(i, 0, n)] = u_new[IDX(i, 0, n)] = 100.0;
        u[IDX(i, n - 1, n)] = u_new[IDX(i, n - 1, n)] = 0.0;
    }
    for (int j = 0; j < n; j++) {
        u[IDX(0, j, n)] = u_new[IDX(0, j, n)] = 50.0;
        u[IDX(n - 1, j, n)] = u_new[IDX(n - 1, j, n)] = 75.0;
    }

    double start = omp_get_wtime();
    int iter;
    for (iter = 1; iter <= max_iters; iter++) {
        double max_diff = 0.0;

        // [Phase 3 - Req 1] 并行外层 j，内层 i 跨行访问；相邻元素相距
        // n 个 double，构成 OpenMP 低局部性基准。
        #pragma omp parallel for schedule(static) reduction(max:max_diff)
        for (int j = 1; j < n - 1; j++) {
            for (int i = 1; i < n - 1; i++) {
                size_t index = IDX(i, j, n);
                double value = 0.25 * (
                    u[IDX(i - 1, j, n)] + u[IDX(i + 1, j, n)] +
                    u[IDX(i, j - 1, n)] + u[IDX(i, j + 1, n)]
                );
                u_new[index] = value;
                double diff = fabs(value - u[index]);
                if (diff > max_diff) max_diff = diff;
            }
        }
        double *temp = u;
        u = u_new;
        u_new = temp;
        if (max_diff < TOLERANCE) break;
    }
    double elapsed = omp_get_wtime() - start;
    int final_iter = iter > max_iters ? max_iters : iter;

    double center;
    if (n % 2) {
        center = u[IDX(n / 2, n / 2, n)];
    } else {
        int m = n / 2;
        center = 0.25 * (u[IDX(m - 1, m - 1, n)] + u[IDX(m - 1, m, n)] +
                         u[IDX(m, m - 1, n)] + u[IDX(m, m, n)]);
    }
    // [Phase 3 - Req 3] 输出统一格式供性能和 perf 脚本解析。
    printf("RESULT,N=%d,Threads=%d,Order=Column-Major,Iters=%d,Time=%.6f,Center=%.6f\n",
           n, threads, final_iter, elapsed, center);
    free(u);
    free(u_new);
    return EXIT_SUCCESS;
}
