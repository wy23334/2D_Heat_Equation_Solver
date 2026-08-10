#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <unistd.h>

#define TOLERANCE 1e-6
#define CACHE_LINE_DOUBLES 16
#define IDX(i, j, stride) ((i) * (stride) + (j))

int main(int argc, char *argv[]) {
    int N = 1024;
    int num_threads = 8;
    int max_iters = 20000;
    int B = 64;

    if (argc > 1) N = atoi(argv[1]);
    if (argc > 2) num_threads = atoi(argv[2]);
    if (argc > 3) max_iters = atoi(argv[3]);
    if (argc > 4) B = atoi(argv[4]);

    if (N < 3 || num_threads < 1 || max_iters < 1 || B < 1) {
        fprintf(stderr, "Usage: %s [N>=3] [threads>=1] [max_iters>=1] [B>=1]\n", argv[0]);
        return EXIT_FAILURE;
    }

    omp_set_num_threads(num_threads);
    usleep(100000);

    int stride = (N + 7) & ~7;
    if (stride % 32 == 0) stride += 8;

    double *u = NULL;
    double *u_new = NULL;
    if (posix_memalign((void**)&u, 64, N * stride * sizeof(double)) != 0) exit(EXIT_FAILURE);
    if (posix_memalign((void**)&u_new, 64, N * stride * sizeof(double)) != 0) exit(EXIT_FAILURE);

    double *padded_max_diff = NULL;
    if (posix_memalign((void**)&padded_max_diff, 64, num_threads * CACHE_LINE_DOUBLES * sizeof(double)) != 0) exit(EXIT_FAILURE);
    for (int i = 0; i < num_threads * CACHE_LINE_DOUBLES; i++) padded_max_diff[i] = 0.0;

    double start_time = 0.0;
    int iter = 0;
    int actual_threads = 1;

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();

        #pragma omp single
        actual_threads = omp_get_num_threads();

        int total_rows = N - 2;
        int chunk = (total_rows + actual_threads - 1) / actual_threads;
        int my_start = 1 + tid * chunk;
        int my_end = my_start + chunk;
        if (my_end > N - 1) my_end = N - 1;
        if (my_start >= N - 1) my_start = my_end = 0;

        // 线程局部 First-Touch NUMA 物理页绑定
        if (my_start < my_end) {
            for (int i = my_start; i < my_end; i++) {
                for (int j = 0; j < stride; j++) {
                    u[i * stride + j] = 0.0;
                    u_new[i * stride + j] = 0.0;
                }
            }
        }
        #pragma omp barrier

        #pragma omp master
        {
            for (int i = 0; i < N; i++) { u[i * stride + 0] = 100.0;     u_new[i * stride + 0] = 100.0; }
            for (int j = 0; j < N; j++) { u[0 * stride + j] = 50.0;      u_new[0 * stride + j] = 50.0;  }
            for (int i = 0; i < N; i++) { u[i * stride + (N-1)] = 0.0;   u_new[i * stride + (N-1)] = 0.0; }
            for (int j = 0; j < N; j++) { u[(N-1) * stride + j] = 75.0;  u_new[(N-1) * stride + j] = 75.0;}
        }
        #pragma omp barrier

        #pragma omp master
        start_time = omp_get_wtime();
        #pragma omp barrier

        double * restrict local_u = u;
        double * restrict local_u_new = u_new;
        int step;

        for (step = 1; step <= max_iters; step++) {
            double my_local_max = 0.0;

            if (my_start < my_end) {
                // 纯正的 2D Grid Tiling
                for (int ii = my_start; ii < my_end; ii += B) {
                    int i_end = (ii + B < my_end) ? ii + B : my_end;

                    for (int jj = 1; jj < N - 1; jj += B) {
                        int j_end = (jj + B < N - 1) ? jj + B : N - 1;

                        for (int i = ii; i < i_end; i++) {
                            double * restrict r_top  = &local_u[(i - 1) * stride];
                            double * restrict r_curr = &local_u[i * stride];
                            double * restrict r_bot  = &local_u[(i + 1) * stride];
                            double * restrict w_curr = &local_u_new[i * stride];

                            #pragma omp simd reduction(max:my_local_max)
                            for (int j = jj; j < j_end; j++) {
                                double val = (r_top[j] + r_bot[j] + r_curr[j - 1] + r_curr[j + 1]) * 0.25;
                                w_curr[j] = val;
                                double diff = fabs(val - r_curr[j]);
                                if (diff > my_local_max) my_local_max = diff;
                            }
                        }
                    }
                }
            }

            padded_max_diff[tid * CACHE_LINE_DOUBLES] = my_local_max;

            #pragma omp barrier

            double step_global_max = 0.0;
            for (int t = 0; t < actual_threads; t++) {
                if (padded_max_diff[t * CACHE_LINE_DOUBLES] > step_global_max) {
                    step_global_max = padded_max_diff[t * CACHE_LINE_DOUBLES];
                }
            }

            double *temp = local_u; local_u = local_u_new; local_u_new = temp;

            if (step_global_max < TOLERANCE) break;
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
        center_temp = (u[IDX(m-1, m-1, stride)] + u[IDX(m, m, stride)] + u[IDX(m-1, m, stride)] + u[IDX(m, m-1, stride)]) * 0.25;
    }

    printf("RESULT,N=%d,Threads=%d,Iters=%d,Time=%.6f,Center=%.6f\n", N, num_threads, iter, elapsed, center_temp);

    free(u); free(u_new); free(padded_max_diff);
    return 0;
}
