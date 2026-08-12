// /home/wy/Projects/2D_Heat_Equation_Solver/src/jacobi_serial.c
// phase 1
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <sys/time.h>

// [Phase 1 - Req 1] 网格大小为 N × N，默认为 1024，可由命令行传入。
#define DEFAULT_N 1024 
// [Phase 1 - Req 4] 最大迭代次数为 200000。
#define MAX_ITER 200000 
// [Phase 1 - Req 4] 绝对收敛阈值为 1e-6。
#define TOLERANCE 1e-6 
#define IDX(i, j, N) ((i) * (N) + (j))

double get_time() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

// 执行串行 Jacobi 迭代；fp 非空时同时输出最终温度场。
void run_jacobi(int N, int max_iters, FILE *fp) {
    printf("\nPhase 1: 串行二维 Jacobi 求解器 (N=%d)\n", N);

    // [Phase 1 - Req 3] 使用双缓冲技术（两个二维数组 u 和 u_new）。
    double *u = (double *)malloc(N * N * sizeof(double));
    double *u_new = (double *)malloc(N * N * sizeof(double));
    if (u == NULL || u_new == NULL) {
        fprintf(stderr, "[ERROR] 无法为 %d x %d 双缓冲网格分配内存。\n", N, N);
        free(u);
        free(u_new);
        return;
    }

    // [Phase 1 - Req 2] 内部初始温度为 0℃。
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            u[IDX(i, j, N)] = 0.0; u_new[IDX(i, j, N)] = 0.0;
        }
    }

    // [Phase 1 - Req 2] 固定边界：左=100℃，右=0℃，上=50℃，下=75℃。
    for (int i = 0; i < N; i++) {
        u[IDX(i, 0, N)] = 100.0;     u_new[IDX(i, 0, N)] = 100.0;  // 左边界
        u[IDX(i, N-1, N)] = 0.0;     u_new[IDX(i, N-1, N)] = 0.0;  // 右边界
    }
    for (int j = 0; j < N; j++) {
        u[IDX(0, j, N)] = 50.0;      u_new[IDX(0, j, N)] = 50.0;   // 上边界
        u[IDX(N-1, j, N)] = 75.0;    u_new[IDX(N-1, j, N)] = 75.0; // 下边界
    }

    double start_time = get_time();
    int iter = 0;
    double max_diff = 0.0;

    printf("[INFO] 开始核心迭代计算。\n");
    // [Phase 1 - Req 1] 五点差分 Jacobi 更新，仅更新内部网格点。
    // [Phase 1 - Req 4] 达到精度或最大迭代次数时停止。
    for (iter = 1; iter <= max_iters; iter++) {
        max_diff = 0.0;
        for (int i = 1; i < N - 1; i++) {
            for (int j = 1; j < N - 1; j++) {
                int idx = IDX(i, j, N);
                u_new[idx] = 0.25 * (
                    u[IDX(i - 1, j, N)] + u[IDX(i + 1, j, N)] +
                    u[IDX(i, j - 1, N)] + u[IDX(i, j + 1, N)]
                );
                double diff = fabs(u_new[idx] - u[idx]);
                if (diff > max_diff) max_diff = diff;
            }
        }

        // [Phase 1 - Req 3] 每次迭代结束后交换指针，不复制整个数组。
        double *temp = u; u = u_new; u_new = temp;

        // [Phase 1 - Req 4] 收敛判据：所有内部点最大变化量 < 1e-6。
        if (max_diff < TOLERANCE) break;
    }

    // [Phase 1 - Req 5] 输出核心计算总运行时间，文件 I/O 不计时。
    double elapsed = get_time() - start_time;

    // --- 提取中心点温度 ---
    double center_temp;
    if (N % 2 != 0) {
        center_temp = u[IDX(N/2, N/2, N)];
    } else {
        int m = N / 2;
        center_temp = (u[IDX(m-1, m-1, N)] + u[IDX(m, m, N)] +
                       u[IDX(m-1, m, N)] + u[IDX(m, m-1, N)]) / 4.0;
    }
    double target = 56.25;
    double error_pct = fabs(center_temp - target) / target * 100.0;

    printf("\n--- 正确性与性能验证 (Sanity Check) ---\n");
    // [Phase 1 - Req 5] 终端打印总耗时、迭代次数和中心温度。
    printf("[指标 1] 串行耗时    : %.4f 秒 (Req <= 5秒)\n", elapsed);

    int final_iter = iter > max_iters ? max_iters : iter;
    printf("[指标 1] 最终迭代次数: %d 次\n", final_iter);

    if (max_diff < TOLERANCE) {
        printf("[指标 1] 迭代停止判据: 精度达标 (当前最大误差 %.2e < 1e-6)\n", max_diff);
    } else {
        printf("[指标 1] 迭代停止判据: 达到迭代次数上限 (%d次，此时误差 %.2e)\n", max_iters, max_diff);
    }

    printf("[指标 2] 理论中心温度: %.4f ℃\n", target);
    printf("[指标 2] 实际计算温度: %.6f ℃\n", center_temp);
    printf("[指标 2] 相对误差百分比: %.6f%%\n", error_pct);

    // [Phase 1 - Req 5] 将最终温度场保存为每行 N 个浮点数的文本矩阵。
    if (fp) {
        printf("[INFO] 写入 N=%d 的温度场数据。\n", N);
        // 以注释行标记 N；Phase 1 绘图脚本据此读取不同规模的矩阵。
        fprintf(fp, "### N=%d\n", N);
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                fprintf(fp, "%.5f ", u[IDX(i, j, N)]);
            }
            fprintf(fp, "\n");
        }
        printf("[INFO] 温度场写入完成。\n");
    }

    free(u);
    free(u_new);
}

// 解析命令行参数，管理结果文件并调用核心求解函数。
int main(int argc, char *argv[]) {
    int N = DEFAULT_N;
    int max_iters = MAX_ITER;
    int write_heatmap = 1;
    // [Phase 1 - Req 1] N、迭代上限和是否输出热力图均可由命令行传入。
    if (argc > 1) {
        N = atoi(argv[1]);
        if (N < 3) return 1;
    }
    if (argc > 2) max_iters = atoi(argv[2]);
    if (argc > 3) write_heatmap = atoi(argv[3]);
    if (max_iters < 1) return 1;

    // 1. 在主函数统一打开文件（"w" 模式覆盖之前的旧文件）
    // 以可执行文件所在的 src 目录为基准，因此从项目根目录或 src
    // 目录启动都能写入同一个 data/phase_1_heatmap.txt。
    char filename[4096];
    const char *slash = strrchr(argv[0], '/');
    if (slash != NULL) {
        size_t directory_length = (size_t)(slash - argv[0]);
        snprintf(filename, sizeof(filename), "%.*s/../data/phase_1_heatmap.txt",
                 (int)directory_length, argv[0]);
    } else {
        snprintf(filename, sizeof(filename), "../data/phase_1_heatmap.txt");
    }
    FILE *fp = write_heatmap ? fopen(filename, "w") : NULL;
    if (write_heatmap && !fp) {
        printf("[WARNING] 无法创建或打开输出文件 %s。\n", filename);
    }

    // 2. 依次运行三种规模，把文件指针传给它们
    // 仅 phase 1 才取消下两行注释，取消注释后须重新编译
//    run_jacobi(16, fp);
//    run_jacobi(128, fp);
    run_jacobi(N, max_iters, fp);

    // 3. 运行结束后，统一关闭文件
    if (fp) {
        fclose(fp);
        printf("\n温度场数据已保存至: %s\n", filename);
    }

    return 0;
}
