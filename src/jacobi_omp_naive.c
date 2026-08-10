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
    int num_threads = 8;
    int max_iters = DEFAULT_MAX_ITERS;
    if (argc > 1) n = atoi(argv[1]);
    if (argc > 2) num_threads = atoi(argv[2]);
    if (argc > 3) max_iters = atoi(argv[3]);
    if (n < 3 || num_threads < 1 || max_iters < 1) {
        fprintf(stderr, "Usage: %s [N>=3] [threads>=1] [max_iters>=1]\n", argv[0]);
        return EXIT_FAILURE;
    }

    omp_set_num_threads(num_threads);
    double *u = calloc((size_t)n * n, sizeof(*u));
    double *u_new = calloc((size_t)n * n, sizeof(*u_new));
    if (u == NULL || u_new == NULL) {
        free(u);
        free(u_new);
        return EXIT_FAILURE;
    }

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

        // Phase 2 基础实现：每轮创建 parallel 区域，每个线程维护
        // 局部最大值，再按题目允许的 critical 方式合并。
        #pragma omp parallel
        {
            double local_max = 0.0;
            #pragma omp for schedule(static)
            for (int i = 1; i < n - 1; i++) {
                for (int j = 1; j < n - 1; j++) {
                    size_t index = IDX(i, j, n);
                    double value = 0.25 * (
                        u[IDX(i - 1, j, n)] + u[IDX(i + 1, j, n)] +
                        u[IDX(i, j - 1, n)] + u[IDX(i, j + 1, n)]
                    );
                    u_new[index] = value;
                    double diff = fabs(value - u[index]);
                    if (diff > local_max) local_max = diff;
                }
            }
            #pragma omp critical
            {
                if (local_max > max_diff) max_diff = local_max;
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
    if (n % 2 != 0) {
        center = u[IDX(n / 2, n / 2, n)];
    } else {
        int m = n / 2;
        center = 0.25 * (u[IDX(m - 1, m - 1, n)] + u[IDX(m - 1, m, n)] +
                         u[IDX(m, m - 1, n)] + u[IDX(m, m, n)]);
    }
    printf("RESULT,N=%d,Threads=%d,Iters=%d,Time=%.6f,Center=%.6f\n",
           n, num_threads, final_iter, elapsed, center);

    free(u);
    free(u_new);
    return EXIT_SUCCESS;
}
