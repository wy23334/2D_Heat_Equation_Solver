#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <unistd.h> // 提供 usleep 函数

// req6: 尝试不同的 schedule子句（static, dynamic, guided），对比性能。
#if defined(SCHED_DYNAMIC)
    #define SCHED_CLAUSE schedule(dynamic, chunk_size)
#elif defined(SCHED_GUIDED)
    #define SCHED_CLAUSE schedule(guided, chunk_size)
#elif defined(SCHED_RUNTIME)
    #define SCHED_CLAUSE schedule(runtime)
#else
    #define SCHED_CLAUSE schedule(static)
#endif

#define TOLERANCE 1e-6
// req6: 检查并消除 false sharing（将每个线程的局部 diff 变量填充到不同 cache line）。
#define CACHE_LINE_DOUBLES 16 // 扩展到 128 字节，防止现代 CPU 的激进预取引发伪共享
#define IDX(i, j, stride) ((i) * (stride) + (j))

int main(int argc, char *argv[]) {
    // req4: 测试不同线程数（1,2,4,8）的性能，记录强标（固定 N=1024）的加速比。
    // req5: 测试弱标（每线程工作量固定：N=256, 362, 512, 724 分别对应 1,2,4,8 线程），观察并行效率。
    int N = 1024;
    int num_threads = 8;
    int max_iters = 200000;

    if (argc > 1) N = atoi(argv[1]);
    if (argc > 2) num_threads = atoi(argv[2]);
    if (argc > 3) max_iters = atoi(argv[3]);

    omp_set_num_threads(num_threads);

    // 散热降频保护
    usleep(100000);

    // 向下取整的均衡分配，把队列抢占次数降到最低
    int chunk_size = (N - 2) / num_threads;
    if (chunk_size < 1) chunk_size = 1;

    // Stride 内存对齐，避免 False Sharing 和 Cache 冲突
    int stride = (N + 7) & ~7;
    if (stride % 32 == 0) stride += 8;

    double *u = NULL;
    double *u_new = NULL;
    if (posix_memalign((void**)&u, 64, N * stride * sizeof(double)) != 0) exit(EXIT_FAILURE);
    if (posix_memalign((void**)&u_new, 64, N * stride * sizeof(double)) != 0) exit(EXIT_FAILURE);

    // 分配无锁缓存行对齐的局部最大值数组
    double *padded_max_diff = NULL;
    if (posix_memalign((void**)&padded_max_diff, 64, num_threads * CACHE_LINE_DOUBLES * sizeof(double)) != 0) exit(EXIT_FAILURE);
    for (int i = 0; i < num_threads * CACHE_LINE_DOUBLES; i++) padded_max_diff[i] = 0.0;

    // First-Touch 对齐
    // 初始化时使用与计算完全一致的调度策略，确保物理内存分配在对应的核心上
    #pragma omp parallel for SCHED_CLAUSE
    for (int i = 1; i < N - 1; i++) {
        for (int j = 0; j < stride; j++) {
            u[i * stride + j] = 0.0;
            u_new[i * stride + j] = 0.0;
        }
    }

    // 初始化首尾两行（不参与并行核心计算的行）
    for (int j = 0; j < stride; j++) {
        u[0 * stride + j] = 0.0;
        u_new[0 * stride + j] = 0.0;
        u[(N-1) * stride + j] = 0.0;
        u_new[(N-1) * stride + j] = 0.0;
    }

    // 初始化真实的边界条件
    for (int i = 0; i < N; i++) { u[i * stride + 0] = 100.0;     u_new[i * stride + 0] = 100.0; }
    for (int j = 0; j < N; j++) { u[0 * stride + j] = 50.0;      u_new[0 * stride + j] = 50.0;  }
    for (int i = 0; i < N; i++) { u[i * stride + (N-1)] = 0.0;   u_new[i * stride + (N-1)] = 0.0; }
    for (int j = 0; j < N; j++) { u[(N-1) * stride + j] = 75.0;  u_new[(N-1) * stride + j] = 75.0;}

    double start_time = omp_get_wtime();
    int iter = 0;

    // req1: 基于串行代码，使用 OpenMP 进行并行化。
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        double * restrict local_u = u;
        double * restrict local_u_new = u_new;
        int step;

        for (step = 1; step <= max_iters; step++) {
            double my_local_max = 0.0;

            // nowait 避免多余同步
            // req2: 并行化外层 i 循环（行方向），使用 #pragma omp parallel for。
            // (注: 这里在 parallel 区域内部，所以直接使用 #pragma omp for)
            #pragma omp for SCHED_CLAUSE nowait
            for (int i = 1; i < N - 1; i++) {
                double * restrict curr_row = &local_u[i * stride];
                double * restrict next_row = &local_u_new[i * stride];
                double * restrict top_row  = &local_u[(i - 1) * stride];
                double * restrict bot_row  = &local_u[(i + 1) * stride];

                #pragma omp simd reduction(max:my_local_max)
                for (int j = 1; j < N - 1; j++) {
                    double val = (top_row[j] + bot_row[j] + curr_row[j - 1] + curr_row[j + 1]) * 0.25;
                    next_row[j] = val;

                    // 回归原生的 fabs，让编译器生成最佳的 SIMD 指令
                    double diff = fabs(val - curr_row[j]);
                    if (diff > my_local_max) {
                        my_local_max = diff;
                    }
                }
            }

            // req3: 处理收敛判据中的归约问题：计算 max diff 时，每个线程维护自己的局部最大值，最后使用 #pragma omp critical或 reduction(max:diff)合并。
            // (注: 此处代码采用了填充缓存行的自定义合并方式，避免了 critical 带来的性能开销)
            padded_max_diff[tid * CACHE_LINE_DOUBLES] = my_local_max;

            // 每轮迭代唯一的全局同步屏障
            #pragma omp barrier

            // 无锁读取全局最大值
            double step_global_max = 0.0;
            for (int t = 0; t < num_threads; t++) {
                if (padded_max_diff[t * CACHE_LINE_DOUBLES] > step_global_max) {
                    step_global_max = padded_max_diff[t * CACHE_LINE_DOUBLES];
                }
            }

            if (step_global_max < TOLERANCE) {
                break;
            }

            // 交换指针
            double *temp = local_u; local_u = local_u_new; local_u_new = temp;
        }

        #pragma omp master
        {
            iter = (step > max_iters) ? max_iters : step;
            u = local_u;
            u_new = local_u_new;
        }
    }

    double elapsed = omp_get_wtime() - start_time;

    double center_temp;
    if (N % 2 != 0) center_temp = u[IDX(N/2, N/2, stride)];
    else {
        int m = N / 2;
        center_temp = (u[IDX(m-1, m-1, stride)] + u[IDX(m, m, stride)] +
                       u[IDX(m-1, m, stride)] + u[IDX(m, m-1, stride)]) * 0.25;
    }

    printf("RESULT,N=%d,Threads=%d,Iters=%d,Time=%.6f,Center=%.6f\n", N, num_threads, iter, elapsed, center_temp);

    free(u); free(u_new); free(padded_max_diff);
    return 0;
}