#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>       // 必须引入的 OpenMP 头文件

// 响应加分挑战：通过宏开关切换不同的 schedule 策略
#if defined(SCHED_DYNAMIC)
    #define SCHED_CLAUSE schedule(dynamic)
#elif defined(SCHED_GUIDED)
    #define SCHED_CLAUSE schedule(guided)
#else
    #define SCHED_CLAUSE schedule(static)
#endif

#define TOLERANCE 1e-6
#define IDX(i, j, N) ((i) * (N) + (j))

int main(int argc, char *argv[]) {
    int N = 1024;
    int num_threads = 4;
    int max_iters = 200000;

    // 支持命令行传参，方便 Python 自动化评测
    if (argc > 1) N = atoi(argv[1]);
    if (argc > 2) num_threads = atoi(argv[2]);
    if (argc > 3) max_iters = atoi(argv[3]);

    omp_set_num_threads(num_threads);

    double *u = (double *)malloc(N * N * sizeof(double));
    double *u_new = (double *)malloc(N * N * sizeof(double));

    // 【HPC极限优化】：并行初始化 (First-Touch Policy)
    // 这会让内存页绑定到距离各个 CPU 核心最近的物理内存上，极大提升内存带宽！
    #pragma omp parallel for SCHED_CLAUSE
    for (int i = 0; i < N * N; i++) {
        u[i] = 0.0; u_new[i] = 0.0;
    }
    for (int i = 0; i < N; i++) { u[IDX(i, 0, N)] = 100.0; u_new[IDX(i, 0, N)] = 100.0; }
    for (int j = 0; j < N; j++) { u[IDX(0, j, N)] = 50.0;  u_new[IDX(0, j, N)] = 50.0;  }
    for (int i = 0; i < N; i++) { u[IDX(i, N-1, N)] = 0.0; u_new[IDX(i, N-1, N)] = 0.0; }
    for (int j = 0; j < N; j++) { u[IDX(N-1, j, N)] = 75.0;u_new[IDX(N-1, j, N)] = 75.0;}

    double start_time = omp_get_wtime(); // 使用 OpenMP 的高精度挂钟时间
    int iter = 0;
    double max_diff = 0.0;

    for (iter = 1; iter <= max_iters; iter++) {
        max_diff = 0.0;

        // 【核心魔法指令】外层循环并行化 + 安全规约
        #pragma omp parallel for SCHED_CLAUSE reduction(max:max_diff)
        for (int i = 1; i < N - 1; i++) {
            for (int j = 1; j < N - 1; j++) {
                int idx = IDX(i, j, N);
                u_new[idx] = 0.25 * (
                    u[IDX(i - 1, j, N)] + u[IDX(i + 1, j, N)] +
                    u[IDX(i, j - 1, N)] + u[IDX(i, j + 1, N)]
                );
                double diff = fabs(u_new[idx] - u[idx]);
                if (diff > max_diff) {
                    max_diff = diff;
                }
            }
        }
        double *temp = u; u = u_new; u_new = temp;
        if (max_diff < TOLERANCE) break;
    }

    double elapsed = omp_get_wtime() - start_time;

    // 提取中心点温度验证正确性
    double center_temp;
    if (N % 2 != 0) center_temp = u[IDX(N/2, N/2, N)];
    else {
        int m = N / 2;
        center_temp = (u[IDX(m-1, m-1, N)] + u[IDX(m, m, N)] + u[IDX(m-1, m, N)] + u[IDX(m, m-1, N)]) / 4.0;
    }

    // 严谨机器格式，专供 Python 正则表达式抓取
    printf("RESULT,N=%d,Threads=%d,Iters=%d,Time=%.6f,Center=%.6f\n", N, num_threads, iter-1, elapsed, center_temp);

    free(u); free(u_new);
    return 0;
}