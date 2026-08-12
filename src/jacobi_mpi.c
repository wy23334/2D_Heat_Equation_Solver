// Phase 4: MPI 行分解 Jacobi 求解器，使用非阻塞 halo 交换。
#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#define DEFAULT_N 2048
#define DEFAULT_MAX_ITERS 200000
#define TOLERANCE 1e-6
#define IDX(i, j, n) ((size_t)(i) * (n) + (j))

static void abort_all(const char *message, int rank) {
    if (rank == 0) fprintf(stderr, "%s\n", message);
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
}

static inline double update_row(const double *restrict u,
                                double *restrict u_new, int i, int n) {
    double row_max = 0.0;
    for (int j = 1; j < n - 1; j++) {
        size_t index = IDX(i, j, n);
        double value = 0.25 * (
            u[IDX(i - 1, j, n)] + u[IDX(i + 1, j, n)] +
            u[IDX(i, j - 1, n)] + u[IDX(i, j + 1, n)]
        );
        u_new[index] = value;
        double diff = fabs(value - u[index]);
        if (diff > row_max) row_max = diff;
    }
    return row_max;
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    int n = DEFAULT_N;
    int max_iters = DEFAULT_MAX_ITERS;
    if (argc > 1) n = atoi(argv[1]);
    if (argc > 2) max_iters = atoi(argv[2]);
    if (n < 3 || max_iters < 1) {
        abort_all("Usage: jacobi_mpi [N>=3] [max_iters>=1]", rank);
    }
    if (world_size > n - 2) {
        abort_all("MPI process count must not exceed the number of interior rows", rank);
    }

    // [Phase 4 - Req 1] 将 N-2 个内部行连续分块；前 remainder 个进程
    // 多负责一行，从而使各进程负载之差不超过一行。
    int interior_rows = n - 2;
    int base = interior_rows / world_size;
    int remainder = interior_rows % world_size;
    int local_rows = base + (rank < remainder ? 1 : 0);
    int global_start = 1 + rank * base + (rank < remainder ? rank : remainder);
    int global_end = global_start + local_rows;  // exclusive

    // [Phase 4 - Req 1/2] 每个局部子域额外分配上下两行 halo。
    size_t elements = (size_t)(local_rows + 2) * n;
    double *u = calloc(elements, sizeof(*u));
    double *u_new = calloc(elements, sizeof(*u_new));
    if (u == NULL || u_new == NULL) {
        free(u);
        free(u_new);
        abort_all("Unable to allocate MPI local grids", rank);
    }

    // [Phase 1 - Req 2] 为本进程拥有的行设置固定左右边界。
    for (int local_i = 1; local_i <= local_rows; local_i++) {
        u[IDX(local_i, 0, n)] = u_new[IDX(local_i, 0, n)] = 100.0;
        u[IDX(local_i, n - 1, n)] = u_new[IDX(local_i, n - 1, n)] = 0.0;
    }
    // [Phase 1 - Req 2] 首、末进程分别持有物理上、下边界。
    if (rank == 0) {
        for (int j = 0; j < n; j++) {
            u[IDX(0, j, n)] = u_new[IDX(0, j, n)] = 50.0;
        }
    }
    if (rank == world_size - 1) {
        for (int j = 0; j < n; j++) {
            u[IDX(local_rows + 1, j, n)] =
                u_new[IDX(local_rows + 1, j, n)] = 75.0;
        }
    }

    int previous = rank == 0 ? MPI_PROC_NULL : rank - 1;
    int next = rank == world_size - 1 ? MPI_PROC_NULL : rank + 1;
    MPI_Barrier(MPI_COMM_WORLD);
    double start = MPI_Wtime();

    int iter;
    double global_max_diff = 0.0;
    for (iter = 1; iter <= max_iters; iter++) {
        // [Phase 4 - Req 2] 与相邻进程交换边界行。非阻塞 Irecv/Isend +
        // Waitall 完成与 MPI_Sendrecv 相同的 halo 交换，并允许通信计算重叠。
        MPI_Request requests[4];
        MPI_Irecv(&u[IDX(0, 0, n)], n, MPI_DOUBLE, previous, 20,
                  MPI_COMM_WORLD, &requests[0]);
        MPI_Irecv(&u[IDX(local_rows + 1, 0, n)], n, MPI_DOUBLE, next, 10,
                  MPI_COMM_WORLD, &requests[1]);
        MPI_Isend(&u[IDX(1, 0, n)], n, MPI_DOUBLE, previous, 10,
                  MPI_COMM_WORLD, &requests[2]);
        MPI_Isend(&u[IDX(local_rows, 0, n)], n, MPI_DOUBLE, next, 20,
                  MPI_COMM_WORLD, &requests[3]);

        // 先计算不依赖新 halo 的内部行，使通信隐藏在 stencil 计算之后。
        double local_max_diff = 0.0;
        for (int i = 2; i < local_rows; i++) {
            double row_max = update_row(u, u_new, i, n);
            if (row_max > local_max_diff) local_max_diff = row_max;
        }
        MPI_Waitall(4, requests, MPI_STATUSES_IGNORE);

        double row_max = update_row(u, u_new, 1, n);
        if (row_max > local_max_diff) local_max_diff = row_max;
        if (local_rows > 1) {
            row_max = update_row(u, u_new, local_rows, n);
            if (row_max > local_max_diff) local_max_diff = row_max;
        }

        // [Phase 1 - Req 4] 所有进程以 MPI_MAX 归约全局最大残差，确保
        // 每个进程在同一轮作出一致的停止决定。
        MPI_Allreduce(&local_max_diff, &global_max_diff, 1, MPI_DOUBLE,
                      MPI_MAX, MPI_COMM_WORLD);
        double *temp = u;
        u = u_new;
        u_new = temp;
        if (global_max_diff < TOLERANCE) break;
    }
    double local_elapsed = MPI_Wtime() - start;
    double elapsed = 0.0;
    MPI_Reduce(&local_elapsed, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0,
               MPI_COMM_WORLD);
    int final_iter = iter > max_iters ? max_iters : iter;

    // [Phase 4 - Correctness] 中心点由唯一进程持有，归约到 rank 0 后
    // 与串行版中心温度对比；偶数 N 同样取中央四点平均值。
    int center_rows[2];
    int center_cols[2];
    int center_count;
    if (n % 2) {
        center_rows[0] = center_cols[0] = n / 2;
        center_count = 1;
    } else {
        center_rows[0] = center_cols[0] = n / 2 - 1;
        center_rows[1] = center_cols[1] = n / 2;
        center_count = 2;
    }
    double local_center_sum = 0.0;
    for (int ri = 0; ri < center_count; ri++) {
        int global_i = center_rows[ri];
        if (global_i >= global_start && global_i < global_end) {
            int local_i = global_i - global_start + 1;
            for (int cj = 0; cj < center_count; cj++) {
                local_center_sum += u[IDX(local_i, center_cols[cj], n)];
            }
        }
    }
    double center_sum = 0.0;
    MPI_Reduce(&local_center_sum, &center_sum, 1, MPI_DOUBLE, MPI_SUM, 0,
               MPI_COMM_WORLD);

    if (rank == 0) {
        double center = center_sum / (center_count * center_count);
        // [Phase 4 - Req 3/4] 统一输出进程数、时间和正确性指标，供
        // 1/2/4/8 进程强标脚本生成 CSV 与加速比图。
        printf("RESULT,N=%d,Processes=%d,Iters=%d,Time=%.6f,Center=%.6f,MaxDiff=%.6e\n",
               n, world_size, final_iter, elapsed, center, global_max_diff);
    }

    free(u);
    free(u_new);
    MPI_Finalize();
    return EXIT_SUCCESS;
}
