// Phase 3: 串行列优先负面对照。
// 在 C 行主序数组上以内层 i 跨行访问，用于验证低局部性带来的 cache miss。
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#define DEFAULT_N 1024
#define DEFAULT_MAX_ITERS 200000
#define TOLERANCE 1e-6
#define IDX(i, j, n) ((i) * (n) + (j))

static double get_time(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

int main(int argc, char **argv) {
    int n = DEFAULT_N;
    int max_iters = DEFAULT_MAX_ITERS;
    if (argc > 1) n = atoi(argv[1]);
    if (argc > 2) max_iters = atoi(argv[2]);
    if (n < 3 || max_iters < 1) {
        fprintf(stderr, "Usage: %s [N>=3] [max_iters>=1]\n", argv[0]);
        return EXIT_FAILURE;
    }

    // [Phase 3 - Req 1] 使用与 Phase 1 相同的双缓冲、边界和收敛条件，
    // 仅改变内部点遍历顺序，保证性能对比的数学工作量一致。
    double *u = calloc((size_t)n * n, sizeof(*u));
    double *u_new = calloc((size_t)n * n, sizeof(*u_new));
    if (u == NULL || u_new == NULL) {
        free(u);
        free(u_new);
        return EXIT_FAILURE;
    }

    // [Phase 1 - Req 2] 固定边界：左=100℃，右=0℃，上=50℃，下=75℃。
    for (int i = 0; i < n; i++) {
        u[IDX(i, 0, n)] = u_new[IDX(i, 0, n)] = 100.0;
        u[IDX(i, n - 1, n)] = u_new[IDX(i, n - 1, n)] = 0.0;
    }
    for (int j = 0; j < n; j++) {
        u[IDX(0, j, n)] = u_new[IDX(0, j, n)] = 50.0;
        u[IDX(n - 1, j, n)] = u_new[IDX(n - 1, j, n)] = 75.0;
    }

    double start = get_time();
    int iter;
    for (iter = 1; iter <= max_iters; iter++) {
        double max_diff = 0.0;

        // [Phase 3 - Req 1] 故意采用 j 外层、i 内层；相邻访问跨越 n 个
        // double，用作 perf 和分块优化的低局部性基准。
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

        // [Phase 3 - Req 2] 保持 Jacobi 双缓冲逻辑，不跨迭代做分块。
        double *temp = u;
        u = u_new;
        u_new = temp;
        if (max_diff < TOLERANCE) break;
    }
    double elapsed = get_time() - start;
    int final_iter = iter > max_iters ? max_iters : iter;

    double center;
    if (n % 2 != 0) {
        center = u[IDX(n / 2, n / 2, n)];
    } else {
        int m = n / 2;
        center = 0.25 * (u[IDX(m - 1, m - 1, n)] + u[IDX(m - 1, m, n)] +
                         u[IDX(m, m - 1, n)] + u[IDX(m, m, n)]);
    }
    // [Phase 3 - Req 3] 输出统一格式供四版本性能脚本解析。
    printf("RESULT,N=%d,Order=Column-Major,Iters=%d,Time=%.6f,Center=%.6f\n",
           n, final_iter, elapsed, center);

    free(u);
    free(u_new);
    return EXIT_SUCCESS;
}
