// Phase 0: OpenMP 环境热身与线程创建验证。
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

int main(void) {
    printf("主线程: 准备创建 4 个线程...\n\n");

    // [Phase 0 - Req 1] 创建 4 个 OpenMP 线程。
    #pragma omp parallel num_threads(4)
    {
        // [Phase 0 - Req 2] 每个线程输出自己的线程编号。
        int thread_id = omp_get_thread_num();
        printf("Hello from thread %d\n", thread_id);
    }  // [Phase 0 - Req 3] 并行区末尾存在隐式屏障，4 个线程在此 join。

    printf("\n主线程: 4个线程已在并行区末尾完成同步。\n");
    return EXIT_SUCCESS;
}
