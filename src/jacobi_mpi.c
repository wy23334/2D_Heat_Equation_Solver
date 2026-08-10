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

    // Block-distribute the N-2 interior rows. Ranks [0,remainder) own one extra row.
    int interior_rows = n - 2;
    int base = interior_rows / world_size;
    int remainder = interior_rows % world_size;
    int local_rows = base + (rank < remainder ? 1 : 0);
    int global_start = 1 + rank * base + (rank < remainder ? rank : remainder);
    int global_end = global_start + local_rows;  // exclusive

    // Two additional rows hold the upper and lower halos.
    size_t elements = (size_t)(local_rows + 2) * n;
    double *u = calloc(elements, sizeof(*u));
    double *u_new = calloc(elements, sizeof(*u_new));
    if (u == NULL || u_new == NULL) {
        free(u);
        free(u_new);
        abort_all("Unable to allocate MPI local grids", rank);
    }

    // Fixed left/right boundaries for locally owned rows.
    for (int local_i = 1; local_i <= local_rows; local_i++) {
        u[IDX(local_i, 0, n)] = u_new[IDX(local_i, 0, n)] = 100.0;
        u[IDX(local_i, n - 1, n)] = u_new[IDX(local_i, n - 1, n)] = 0.0;
    }
    // Physical top/bottom boundaries live in the outer halo rows.
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
        // Exchange halos asynchronously. Interior rows do not depend on the
        // incoming halos and are computed while communication is in flight.
        MPI_Request requests[4];
        MPI_Irecv(&u[IDX(0, 0, n)], n, MPI_DOUBLE, previous, 20,
                  MPI_COMM_WORLD, &requests[0]);
        MPI_Irecv(&u[IDX(local_rows + 1, 0, n)], n, MPI_DOUBLE, next, 10,
                  MPI_COMM_WORLD, &requests[1]);
        MPI_Isend(&u[IDX(1, 0, n)], n, MPI_DOUBLE, previous, 10,
                  MPI_COMM_WORLD, &requests[2]);
        MPI_Isend(&u[IDX(local_rows, 0, n)], n, MPI_DOUBLE, next, 20,
                  MPI_COMM_WORLD, &requests[3]);

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

    // Each center point is owned by exactly one rank; sum contributions to rank 0.
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
        printf("RESULT,N=%d,Processes=%d,Iters=%d,Time=%.6f,Center=%.6f,MaxDiff=%.6e\n",
               n, world_size, final_iter, elapsed, center, global_max_diff);
    }

    free(u);
    free(u_new);
    MPI_Finalize();
    return EXIT_SUCCESS;
}
