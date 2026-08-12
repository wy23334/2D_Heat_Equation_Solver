// Phase 2: OpenMP Jacobi 求解器，支持 static/dynamic/guided 调度对比。
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <unistd.h> // 提供 usleep 函数

// [Phase 2 - Req 6] 通过编译宏选择 static、dynamic 或 guided 调度。
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
// [Phase 2 - Req 7] 将各线程局部残差放入不同缓存行，消除 false sharing。
#define CACHE_LINE_DOUBLES 16 // 扩展到 128 字节，防止现代 CPU 的激进预取引发伪共享
#define IDX(i, j, stride) ((i) * (stride) + (j))

int main(int argc, char *argv[]) {
    // [Phase 2 - Req 4] threads 参数用于 1/2/4/8 线程强标测试（固定 N=1024）。
    // [Phase 2 - Req 5] N 参数也支持 256/362/512/724 的弱标测试。
    int N = 1024;
    int num_threads = 8;
    int max_iters = 200000;

    if (argc > 1) N = atoi(argv[1]);
    if (argc > 2) num_threads = atoi(argv[2]);
    if (argc > 3) max_iters = atoi(argv[3]);

    if (N < 3 || num_threads < 1 || max_iters < 1) {
        fprintf(stderr, "Usage: %s [N>=3] [threads>=1] [max_iters>=1]\n", argv[0]);
        return EXIT_FAILURE;
    }

    omp_set_num_threads(num_threads);

    // 性能测试前短暂让出 CPU；该时间不计入核心迭代计时。
    usleep(100000);

    // dynamic/guided 调度使用近似均衡的块大小，减少运行时任务分派开销。
    int chunk_size = (N - 2) / num_threads;
    if (chunk_size < 1) chunk_size = 1;

    // [Phase 2 - Req 7] 使用带填充 stride，改善行对齐并降低缓存冲突。
    int stride = (N + 7) & ~7;
    if (stride % 32 == 0) stride += 8;

    double *u = NULL;
    double *u_new = NULL;
    if (posix_memalign((void**)&u, 64, N * stride * sizeof(double)) != 0) exit(EXIT_FAILURE);
    if (posix_memalign((void**)&u_new, 64, N * stride * sizeof(double)) != 0) exit(EXIT_FAILURE);

    // [Phase 2 - Req 3] 为每个线程分配缓存行隔离的局部最大残差。
    double *padded_max_diff = NULL;
    if (posix_memalign((void**)&padded_max_diff, 64, num_threads * CACHE_LINE_DOUBLES * sizeof(double)) != 0) exit(EXIT_FAILURE);
    for (int i = 0; i < num_threads * CACHE_LINE_DOUBLES; i++) padded_max_diff[i] = 0.0;

    // [Phase 2 - Req 7] First-touch 初始化使用相同调度，改善 NUMA 页归属。
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

    // [Phase 1 - Req 2] 与串行版保持相同的固定边界条件。
    for (int i = 0; i < N; i++) { u[i * stride + 0] = 100.0;     u_new[i * stride + 0] = 100.0; }
    for (int j = 0; j < N; j++) { u[0 * stride + j] = 50.0;      u_new[0 * stride + j] = 50.0;  }
    for (int i = 0; i < N; i++) { u[i * stride + (N-1)] = 0.0;   u_new[i * stride + (N-1)] = 0.0; }
    for (int j = 0; j < N; j++) { u[(N-1) * stride + j] = 75.0;  u_new[(N-1) * stride + j] = 75.0;}

    double start_time = omp_get_wtime();
    int iter = 0;

    // [Phase 2 - Req 1] 在 Phase 1 双缓冲 Jacobi 基础上建立持久 parallel 区域。
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        double * restrict local_u = u;
        double * restrict local_u_new = u_new;
        int step;

        for (step = 1; step <= max_iters; step++) {
            double my_local_max = 0.0;

            // nowait 避免多余同步
            // [Phase 2 - Req 2] 并行化外层 i 循环（行方向）。
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

                    // 使用标准 fabs，便于编译器识别并生成向量化绝对值运算。
                    double diff = fabs(val - curr_row[j]);
                    if (diff > my_local_max) {
                        my_local_max = diff;
                    }
                }
            }

            // [Phase 2 - Req 3] 每线程先计算局部最大值，再通过独立槽位合并；
            // 其语义等价于题目允许的 critical/reduction(max:diff)。
            padded_max_diff[tid * CACHE_LINE_DOUBLES] = my_local_max;

            // 每轮迭代唯一的全局同步屏障
            #pragma omp barrier

            // 每个线程读取所有独立槽位并得到相同的全局最大残差。
            double step_global_max = 0.0;
            for (int t = 0; t < num_threads; t++) {
                if (padded_max_diff[t * CACHE_LINE_DOUBLES] > step_global_max) {
                    step_global_max = padded_max_diff[t * CACHE_LINE_DOUBLES];
                }
            }

            // 本轮结果位于 local_u_new；即使本轮达到收敛条件也必须先交换，
            // 否则最终输出的会是上一轮网格。
            double *temp = local_u; local_u = local_u_new; local_u_new = temp;

            if (step_global_max < TOLERANCE) {
                break;
            }
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

    // [Phase 2 - Req 4/5] 统一 RESULT 格式供强标、弱标和正确性脚本解析。
    printf("RESULT,N=%d,Threads=%d,Iters=%d,Time=%.6f,Center=%.6f\n", N, num_threads, iter, elapsed, center_temp);

    free(u); free(u_new); free(padded_max_diff);
    return 0;
}
