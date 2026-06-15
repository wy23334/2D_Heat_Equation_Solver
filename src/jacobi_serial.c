#include <stdio.h>
#include <stdlib.h>
#include <math.h>      // 用于 fabs() 函数
#include <sys/time.h>  // 用于 gettimeofday() 高精度计时

#define DEFAULT_N 1024
#define MAX_ITER 20000
#define TOLERANCE 1e-6

// 一维数组模拟二维数组宏定义
#define IDX(i, j, N) ((i) * (N) + (j))

// 高精度计时器（返回秒数）
double get_time() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

int main(int argc, char *argv[]) {
    int N = DEFAULT_N;
    if (argc > 1) {
        N = atoi(argv[1]);
        if (N < 3) {
            printf("错误：网格大小必须 >= 3\n");
            return 1;
        }
    }

    printf("\n=== Phase 1: 串行二维 Jacobi 求解器 ===\n");
    printf("当前网格大小: %d x %d\n", N, N);

    double *u = (double *)malloc(N * N * sizeof(double));
    double *u_new = (double *)malloc(N * N * sizeof(double));

    if (u == NULL || u_new == NULL) {
        printf("内存分配失败！\n");
        return -1;
    }

    // 初始化内部与边界
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            u[IDX(i, j, N)] = 0.0;
            u_new[IDX(i, j, N)] = 0.0;
        }
    }
    for (int i = 0; i < N; i++) {
        u[IDX(i, 0, N)] = 100.0;     u_new[IDX(i, 0, N)] = 100.0;     // 左
        u[IDX(i, N-1, N)] = 0.0;     u_new[IDX(i, N-1, N)] = 0.0;     // 右
    }
    for (int j = 0; j < N; j++) {
        u[IDX(0, j, N)] = 50.0;      u_new[IDX(0, j, N)] = 50.0;      // 上
        u[IDX(N-1, j, N)] = 75.0;    u_new[IDX(N-1, j, N)] = 75.0;    // 下
    }

    double start_time = get_time();
    int iter = 0;
    double max_diff = 0.0;

    // 核心迭代
    for (iter = 1; iter <= MAX_ITER; iter++) {
        max_diff = 0.0;

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

        // 双缓冲技术：交换指针
        double *temp = u;
        u = u_new;
        u_new = temp;

        // 日志打印区分：小规模每100步打印，大规模每1000步打印
        if (iter == 1 || (N > 10 && iter % 1000 == 0) || (N <= 10 && iter % 100 == 0)) {
            printf("Iter %5d: max_diff = %.8f\n", iter, max_diff);
        }

        if (max_diff < TOLERANCE) {
            printf("=> 达到收敛标准！在第 %d 次迭代提前退出。\n", iter);
            break;
        }
    }

    double end_time = get_time();
    if (iter > MAX_ITER) iter = MAX_ITER;

    double elapsed = end_time - start_time;
    printf("\n=== 计算报告 (N=%d) ===\n", N);
    printf("最终迭代次数: %d\n", iter);
    printf("最终最大残差: %e\n", max_diff);
    printf("程序总运行时间: %.3f 秒 (约 %.1f 毫秒)\n", elapsed, elapsed * 1000.0);

    // 【修改点】动态生成文件名：heatmap_N.txt
    char filename[64];
    sprintf(filename, "heatmap_%d.txt", N);
    printf("\n正在写入 %s...\n", filename);

    FILE *fp = fopen(filename, "w");
    if (fp) {
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                fprintf(fp, "%.5f ", u[IDX(i, j, N)]);
            }
            fprintf(fp, "\n");
        }
        fclose(fp);
        printf("写入完成！\n");
    }

    free(u);
    free(u_new);
    return 0;
}