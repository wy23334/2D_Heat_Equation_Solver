// /home/wy/Projects/2D_Heat_Equation_Solver/src/jacobi_omp.c
// phase 2
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// [Req 1] 使用 OpenMP 进行并行化
#include <omp.h>

// [Req 6] 尝试不同的 schedule 子句（static, dynamic, guided）
// 说明：利用宏定义，编译时可通过 -DSCHED_DYNAMIC 等参数动态切换。
// 默认为 static 调度。
#if defined(SCHED_DYNAMIC)
    #define SCHED_CLAUSE schedule(dynamic)
#elif defined(SCHED_GUIDED)
    #define SCHED_CLAUSE schedule(guided)
#elif defined(SCHED_RUNTIME)
    #define SCHED_CLAUSE schedule(runtime) // 终极方案：运行时通过 export OMP_SCHEDULE="dynamic" 切换
#else
    #define SCHED_CLAUSE schedule(static)
#endif


// [Req 6] 检查并消除 false sharing
// 说明：大多数现代 CPU 的 Cache Line 大小为 64 字节。
// 一个 double 占 8 字节，所以 64 / 8 = 8。
// 设定跨度 PAD = 8，可以保证不同线程的数据绝对落在不同的 Cache Line 上。

#define CACHE_LINE_DOUBLES 8

#define TOLERANCE 1e-6
#define IDX(i, j, N) ((i) * (N) + (j))

int main(int argc, char *argv[]) {
    int N = 1024;
    int num_threads = 8;
    int max_iters = 200000;

    // 动态传参
    if (argc > 1) N = atoi(argv[1]);
    if (argc > 2) num_threads = atoi(argv[2]);
    if (argc > 3) max_iters = atoi(argv[3]);

    omp_set_num_threads(num_threads);

    double *u = (double *)malloc(N * N * sizeof(double));
    double *u_new = (double *)malloc(N * N * sizeof(double));

    // 【HPC极限优化】：并行初始化 (First-Touch Policy)
    #pragma omp parallel for SCHED_CLAUSE
    for (int i = 0; i < N * N; i++) {
        u[i] = 0.0; u_new[i] = 0.0;
    }
    for (int i = 0; i < N; i++) { u[IDX(i, 0, N)] = 100.0; u_new[IDX(i, 0, N)] = 100.0; }
    for (int j = 0; j < N; j++) { u[IDX(0, j, N)] = 50.0;  u_new[IDX(0, j, N)] = 50.0;  }
    for (int i = 0; i < N; i++) { u[IDX(i, N-1, N)] = 0.0; u_new[IDX(i, N-1, N)] = 0.0; }
    for (int j = 0; j < N; j++) { u[IDX(N-1, j, N)] = 75.0;u_new[IDX(N-1, j, N)] = 75.0;}

    // [Req 6] 手动分配填充过 (Padded) 的数组，用于替代 reduction 消除伪共享
    // 数组总大小：线程数 * 8
    double *padded_max_diff = (double *)calloc(num_threads * CACHE_LINE_DOUBLES, sizeof(double));

    double start_time = omp_get_wtime();
    int iter = 0;
    double global_max_diff = 0.0;

    for (iter = 1; iter <= max_iters; iter++) {
        global_max_diff = 0.0;

        // [Req 2] 并行化外层 i 循环
        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            double my_local_max = 0.0; // 寄存器变量，最快

            // [Req 6] 使用宏定义的 SCHED_CLAUSE 尝试不同的调度策略
            // 注意：这里去掉了自带的 reduction，改用我们手动实现的伪共享消除方案
            #pragma omp for SCHED_CLAUSE
            for (int i = 1; i < N - 1; i++) {
                for (int j = 1; j < N - 1; j++) {
                    int idx = IDX(i, j, N);
                    u_new[idx] = 0.25 * (
                        u[IDX(i - 1, j, N)] + u[IDX(i + 1, j, N)] +
                        u[IDX(i, j - 1, N)] + u[IDX(i, j + 1, N)]
                    );
                    double diff = fabs(u_new[idx] - u[idx]);
                    if (diff > my_local_max) {
                        my_local_max = diff;
                    }
                }
            }

            // [Req 6] 将每个线程的局部 diff 变量填充到不同的 cache line
            // tid * CACHE_LINE_DOUBLES 保证了间距为 64 字节，彻底消除 False Sharing
            padded_max_diff[tid * CACHE_LINE_DOUBLES] = my_local_max;
        }

        // [Req 6] 在串行区域手动进行最大值归约
        for (int t = 0; t < num_threads; t++) {
            if (padded_max_diff[t * CACHE_LINE_DOUBLES] > global_max_diff) {
                global_max_diff = padded_max_diff[t * CACHE_LINE_DOUBLES];
            }
        }

        double *temp = u; u = u_new; u_new = temp;
        if (global_max_diff < TOLERANCE) break;
    }

    double elapsed = omp_get_wtime() - start_time;

    double center_temp;
    if (N % 2 != 0) center_temp = u[IDX(N/2, N/2, N)];
    else {
        int m = N / 2;
        center_temp = (u[IDX(m-1, m-1, N)] + u[IDX(m, m, N)] + u[IDX(m-1, m, N)] + u[IDX(m, m-1, N)]) / 4.0;
    }

    int final_iter = (iter > max_iters) ? max_iters : iter;
    printf("RESULT,N=%d,Threads=%d,Iters=%d,Time=%.6f,Center=%.6f\n", N, num_threads, final_iter, elapsed, center_temp);

    free(u); free(u_new); free(padded_max_diff);
    return 0;
}