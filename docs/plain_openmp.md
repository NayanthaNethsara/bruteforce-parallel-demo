# Plain OpenMP Brute Force Documentation

## 1. Parallelization Strategies
**Approach:**
The OpenMP implementation utilizes a shared-memory parallelization strategy. The primary approach involves partitioning the search space at the highest level of the candidate generation loop. Specifically, the first character of the candidate password is used as the basis for dividing work among threads.

**Design Decisions:**
- **Outer Loop Parallelization:** We parallelize the loop that iterates over the first character of the password (indices[0]). This ensures that each thread works on a distinct, non-overlapping subset of the search space (e.g., Thread 0 checks "a...", Thread 1 checks "b...", etc.).
- **Dynamic Scheduling:** While static scheduling is simple, dynamic scheduling (or the implicit round-robin assignment used in the code `start_char = thread_id; start_char < CS_LEN; start_char += num_threads`) ensures better load balancing if some sub-trees are pruned early (though in a pure brute force without pruning, static is fine). The current implementation uses a cyclic distribution manually.
- **Critical Section:** A `critical` section is used when a thread finds the password to safely update the shared `found` flag and print the result without race conditions.

**Load Balancing:**
The cyclic distribution (`start_char += num_threads`) helps distribute the workload evenly. Since all passwords of the same length take roughly the same time to generate and check in the plain version (simple string comparison), the load is naturally well-balanced.

## 2. Runtime Configurations
**Hardware Specifications:**
- **Development:** macOS (Apple Silicon M1/M2)
- **Target:** Linux VM (x86_64)

**Software Environment:**
- **Compiler:** GCC (with OpenMP support, e.g., `gcc-14` or `clang` with `libomp`)
- **Libraries:** Standard C Library (`libc`), OpenMP Runtime (`libomp`)
- **Version:** OpenMP 4.5+

**Configuration Parameters:**
- **Threads:** Configurable via `OMP_NUM_THREADS` environment variable.
- **Max Length:** Command-line argument (e.g., 4, 5).
- **Charset:** Lowercase alphanumeric (36 characters).

## 3. Performance Analysis
**Speedup & Efficiency:**
- **Ideal Speedup:** Linear speedup is expected (Speedup ≈ N, where N is the number of threads) because the task is embarrassingly parallel with minimal communication.
- **Observed:** On a 4-core system, we expect close to 3.5x-3.8x speedup. The overhead of thread creation is negligible compared to the exponential growth of the search space.

**Bottlenecks:**
- **False Sharing:** Minimal, as threads work on private candidate buffers.
- **Synchronization:** The only synchronization point is the `found` check, which is a simple volatile read, and the final critical section. This is highly efficient.

**Scalability:**
- The solution scales well with the number of cores up to the size of the charset (36). Beyond 36 threads, the simple partitioning by first character would need to be deeper (e.g., partitioning by first two characters) to utilize more threads effectively.

## 4. Critical Reflection
**Challenges:**
- **MacOS Support:** OpenMP is not built-in to the default Apple Clang, requiring `libomp` installation and specific compiler flags.
- **Loop Collapse:** Standard `omp parallel for` doesn't easily apply to the recursive or while-loop nature of incrementing indices for arbitrary lengths. The manual partitioning strategy was chosen to overcome this.

**Limitations:**
- **Granularity:** Partitioning only by the first character limits parallelism to 36 threads. For massive clusters, a more granular approach is needed.

**Lessons Learned:**
- Shared-memory parallelism is the easiest to implement for brute-force tasks but requires careful memory management (private variables) to avoid race conditions.
