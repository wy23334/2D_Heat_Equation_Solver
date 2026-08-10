// /home/wy/Projects/2D_Heat_Equation_Solver/src/jacobi_serial.c
// phase 1
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>

// [Req 1]: 网格大小为 N × N，默认为 1024
#define DEFAULT_N 1024 
// [Req 4]: 迭代次数达到上限 200000 次
#define MAX_ITER 200000 
// [Req 4]: 所有网格点最大差值 < 1e-6
#define TOLERANCE 1e-6 
#define IDX(i, j, N) ((i) * (N) + (j))

double get_time() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

// ==========================================
// 核心计算函数：接收 FILE *fp 作为参数
// ==========================================
void run_jacobi(int N, int max_iters, FILE *fp) {
    printf("\n=============================================\n");
    printf("   Phase 1: 串行二维 Jacobi 求解器 (N=%d)\n", N);
    printf("=============================================\n");

    // [Req 3]: 使用双缓冲技术（两个二维数组 u 和 u_new）
    double *u = (double *)malloc(N * N * sizeof(double));
    double *u_new = (double *)malloc(N * N * sizeof(double));

    // [Req 2]: 内部初始温度为 0℃
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            u[IDX(i, j, N)] = 0.0; u_new[IDX(i, j, N)] = 0.0;
        }
    }

    // [Req 2]: 边界条件：四周固定温度（左=100℃，右=0℃，上=50℃，下=75℃）
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

    printf("[-] 正在进行核心迭代计算...\n");
    // [Req 1]: 实现二维 Jacobi 迭代求解器
    // [Req 4]: 迭代次数达到上限
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

        // [Req 3]: 每次迭代结束后交换指针
        double *temp = u; u = u_new; u_new = temp;

        // [Req 4]: 收敛判据：两次迭代间所有网格点最大差值 < 1e-6
        if (max_diff < TOLERANCE) break;
    }

    // [Req 5]: 输出程序总运行时间
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

    printf("\n--- 🏁 正确性与性能验证 (Sanity Check) ---\n");
    // [Req 5]: 终端打印总耗时
    printf("[指标 1] 串行耗时    : %.4f 秒 (Req <= 5秒)\n", elapsed);

    int final_iter = iter > max_iters ? max_iters : iter;
    printf("[指标 1] 最终迭代次数: %d 次\n", final_iter);

    if (max_diff < TOLERANCE) {
        printf("[指标 1] 迭代停止判据: 精度达标 (当前最大误差 %.2e < 1e-6)\n", max_diff);
    } else {
        printf("[指标 1] 迭代停止判据: ⚠️ 达到迭代次数上限 (%d次，此时误差 %.2e)\n", max_iters, max_diff);
    }
    printf("---------------------------------------------\n");

    printf("[指标 2] 理论中心温度: %.4f ℃\n", target);
    printf("[指标 2] 实际计算温度: %.6f ℃\n", center_temp);
    printf("[指标 2] 相对误差百分比: %.6f%%\n", error_pct);

    // [Req 5]: 如果传入了有效的文件指针，则统一保存到这个文件中
    if (fp) {
        printf("[-] 正在追加 N=%d 的热力图数据到统一文件...\n", N);
        // 添加一个头部标记，方便后续读取时切分不同大小的矩阵
        fprintf(fp, "### N=%d\n", N);
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                fprintf(fp, "%.5f ", u[IDX(i, j, N)]);
            }
            fprintf(fp, "\n");
        }
        printf("[-] 数据追加完毕。\n");
    }

    free(u);
    free(u_new);
}

// ==========================================
// 主函数：统一打开文件并按顺序传递指针
// ==========================================
int main(int argc, char *argv[]) {
    int N = DEFAULT_N;
    int max_iters = MAX_ITER;
    int write_heatmap = 1;
    // [Req 1]: N 由命令行参数传入
    if (argc > 1) {
        N = atoi(argv[1]);
        if (N < 3) return 1;
    }
    if (argc > 2) max_iters = atoi(argv[2]);
    if (argc > 3) write_heatmap = atoi(argv[3]);
    if (max_iters < 1) return 1;

    // 1. 在主函数统一打开文件（"w" 模式覆盖之前的旧文件）
    char filename[] = "../data/heatmap.txt";
    FILE *fp = write_heatmap ? fopen(filename, "w") : NULL;
    if (write_heatmap && !fp) {
        printf("[!] 警告: 无法创建或打开 %s 准备写入。\n", filename);
    }

    // 2. 依次运行三种规模，把文件指针传给它们
    // 仅 phase 1 才取消下两行注释，取消注释后须重新编译
//    run_jacobi(16, fp);
//    run_jacobi(128, fp);
    run_jacobi(N, max_iters, fp);

    // 3. 运行结束后，统一关闭文件
    if (fp) {
        fclose(fp);
        printf("\n✅ 所有规模的数据均已保存至: %s\n", filename);
    }

    return 0;
}
