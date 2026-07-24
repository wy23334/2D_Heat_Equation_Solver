// /home/wy/Projects/2D_Heat_Equation_Solver/src/jacobi_omp.c
// phase 2
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// [Req 1] 使用 OpenMP 进行并行化 (引入必备的头文件)
#include <omp.h>


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
    int num_threads = 8;
    int max_iters = 200000;

    // [Req 4 & Req 5 的基础支撑]
    // 支持命令行传参动态修改 N 和 线程数。
    // 这使得外部自动化评测脚本可以通过改变 N 测“弱标(Req 5)”，通过改变 num_threads 测“强标(Req 4)”。
    if (argc > 1) N = atoi(argv[1]);
    if (argc > 2) num_threads = atoi(argv[2]);
    if (argc > 3) max_iters = atoi(argv[3]);

    // [Req 1] 使用 OpenMP 进行并行化 (在程序运行时动态设置使用多少个线程)
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

    // [计时方法]：采用 OpenMP 专用的高精度并行计时器
    double start_time = omp_get_wtime();
    int iter = 0;
    double max_diff = 0.0;

    for (iter = 1; iter <= max_iters; iter++) {
        max_diff = 0.0;

        // [Req 2] 并行化外层 i 循环（行方向），使用 #pragma omp parallel for
        // [Req 3] 处理收敛判据中的归约问题，使用 reduction(max:max_diff) 安全合并局部最大值
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

    // [Req 4 & Req 5] 修正了 iter 的准确性。这段紧凑的输出同样是为了方便 Python 抓取
    int final_iter = (iter > max_iters) ? max_iters : iter;
    printf("RESULT,N=%d,Threads=%d,Iters=%d,Time=%.6f,Center=%.6f\n", N, num_threads, final_iter, elapsed, center_temp);

    free(u); free(u_new);
    return 0;
}