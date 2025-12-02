# Plain MPI Brute Force Documentation

## 1. Parallelization Strategies
**Approach:**
The MPI (Message Passing Interface) implementation uses a distributed-memory model. The search space is partitioned among available processes (ranks). Similar to the OpenMP approach, the partitioning is done based on the first character of the candidate string.

**Design Decisions:**
- **Rank-based Partitioning:** Each process calculates which starting characters it is responsible for using a cyclic distribution: `start_char = rank + k * size`. This ensures that even if the number of processes is small or large, the work is distributed.
- **Independence:** Each process works completely independently on its assigned portion of the search space. There is no communication during the search phase, which maximizes efficiency.
- **Termination:** When a process finds the password, it prints the result and calls `MPI_Abort`. While abrupt, this is the most efficient way to stop all other processes immediately in a brute-force scenario without complex non-blocking communication polling.

**Load Balancing:**
The cyclic distribution ensures that if the search space is uneven (e.g., "z" takes longer to reach than "a" in a linear scan, though in brute force they are equal depth), the load is spread. Since all branches of the brute force tree (for a fixed length) are equal size, static cyclic partitioning provides near-perfect load balancing.

## 2. Runtime Configurations
**Hardware Specifications:**
- **Development:** macOS (Apple Silicon M1/M2)
- **Target:** Linux Cluster or VM with MPI support.

**Software Environment:**
- **Compiler:** `mpicc` (wrapper around gcc/clang)
- **Library:** OpenMPI or MPICH
- **Version:** MPI 3.0+

**Configuration Parameters:**
- **Processes:** Configured via `mpirun -np <N>`.
- **Max Length:** Command-line argument.
- **Charset:** Lowercase alphanumeric (36 characters).

## 3. Performance Analysis
**Speedup & Efficiency:**
- **Speedup:** Linear speedup is achieved. Since there is zero communication during the computation phase, N processes will search the space N times faster.
- **Efficiency:** High efficiency (>90%). The only overhead is MPI initialization (`MPI_Init`) and finalization, which is negligible for long-running brute force tasks.

**Bottlenecks:**
- **Startup Latency:** MPI has a higher startup cost than OpenMP.
- **I/O:** If multiple processes try to print to stdout simultaneously (e.g., debugging), it can cause contention. In the release version, only the finder prints.

**Scalability:**
- Scales linearly with the number of nodes/cores.
- Limited by the charset size (36) for the current partitioning strategy. To scale to hundreds of nodes, a recursive partitioning strategy (splitting the first 2 or 3 characters) would be required.

## 4. Critical Reflection
**Challenges:**
- **Process Management:** Unlike threads, processes do not share memory. We had to ensure the full context (target, max_len) was available to all ranks (passed via arguments).
- **Termination:** Stopping a distributed search is harder than a shared-memory one. `MPI_Abort` is a "nuclear option" but effective for this demo. A cleaner exit would involve `MPI_Iprobe` which adds overhead.

**Limitations:**
- **Memory:** Each process has its own copy of data, but for brute force, memory usage is very low, so this is not a practical limitation.

**Lessons Learned:**
- MPI is powerful for scaling across nodes but requires a "share-nothing" mindset.
- Static partitioning is often sufficient and more efficient than dynamic load balancing for uniform search spaces.
