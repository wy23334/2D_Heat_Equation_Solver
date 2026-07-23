// /home/wy/Projects/2D_Heat_Equation_Solver/src/warm_up.c
#include <stdio.h>
#include <omp.h>  // 必须引入 OpenMP 的专属头文件，里面有获取线程ID的函数

int main() {
    printf("主线程: 准备创建 4 个线程...\n\n");

    // #pragma omp parallel: 告诉编译器下面大括号里的代码要开启多线程并行
    // num_threads(4): 强制要求派生（Fork）出 4 个线程同时去跑
    #pragma omp parallel num_threads(4)
    {
        // 获取当前正在跑这行代码的线程自己的编号 (0, 1, 2, 3)
        int thread_id = omp_get_thread_num();

        // 每个线程独立打印自己的 ID
        printf("Hello from thread %d\n", thread_id);

    } // 【隐式 Join 机制】
      // ⚠️ 重点：在并行区大括号 `}` 结束的位置，OpenMP 会自动设置一个“同步屏障(Barrier)”。
      // 跑得快的线程必须在这里停下等跑得慢的，直到 4 个线程全部到达这里完成汇合 (Join)，
      // 才会变回 1 个主线程继续往下走。

    printf("\n主线程: 4个线程已全部 Join 完毕，程序安全退出。\n");

    // 不是严格按照 0, 1, 2, 3 排列的（比如可能是 0, 2, 3, 1）。重新多运行几次，每次输出的顺序都可能不一样！

    return 0;
}