// SHA-256 brute-force using OpenMP (placeholder)
#include <stdio.h>
#include <omp.h>

int main() {
    printf("SHA-256 OpenMP brute-force (placeholder)\n");
    
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int nthreads = omp_get_num_threads();
        #pragma omp single
        printf("Running with %d threads\n", nthreads);
        printf("Hello from thread %d\n", tid);
    }
    
    return 0;
}
